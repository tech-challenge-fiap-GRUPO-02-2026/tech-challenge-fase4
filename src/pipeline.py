"""Orquestrador do monitoramento multimodal de um paciente.

    ┌ vídeo (fisio/cirurgia) ─ YOLOv8-pose / YOLOv8 ─────────────┐
    ├ áudio da consulta ─ biomarcadores + AudioSet ──┐            │
    │                    └ Azure Speech → Azure Text ┴─ achados ──┤
    ├ sinais vitais (PhysioNet) ─ regras/z/CUSUM/IForest ─────────┼─► fusão ─► alertas ─► equipe
    ├ prescrições ─ farmacovigilância ────────────────────────────┤             │
    └ movimentação (actigrafia) ─ regras + IForest ───────────────┘        auditoria

Os modelos populacionais (Isolation Forest de vitais e de movimentação) são
treinados uma vez e reaproveitados por todos os pacientes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from src.alerts.alert_manager import AlertManager
from src.anomaly import movement, prescriptions, vitals
from src.audio.audio_pipeline import analyze_consultation
from src.config import AUDIO_DIR, OUTPUT_DIR, PATIENTS_CSV, REPORTS_DIR, VIDEO_DIR
from src.datasets import physionet
from src.fusion.multimodal_fusion import fuse
from src.models import SEVERITY_ORDER, Finding
from src.safety.audit_logger import AuditLogger
from src.video.video_pipeline import analyze_video


def load_patients() -> pd.DataFrame:
    return pd.read_csv(PATIENTS_CSV, dtype=str).fillna("")


@dataclass
class PatientAssessment:
    patient: dict
    modalities: dict[str, dict]
    findings: dict[str, list[Finding]]
    fusion: dict
    alerts: list[dict]

    def all_findings(self) -> list[Finding]:
        return [f for fs in self.findings.values() for f in fs]

    def to_dict(self) -> dict:
        return {
            "paciente": self.patient,
            "modalidades": self.modalities,
            "achados": {m: [f.to_dict() for f in fs] for m, fs in self.findings.items()},
            "fusao": self.fusion,
            "alertas": self.alerts,
        }


class MonitoringSystem:
    def __init__(self, alert_manager: AlertManager | None = None, prefer_azure: bool = True,
                 use_audioset_model: bool = False, audit: AuditLogger | None = None):
        self.audit = audit or AuditLogger()
        self.alerts = alert_manager or AlertManager(audit=self.audit)
        self.prefer_azure = prefer_azure
        self.use_audioset_model = use_audioset_model
        self.patients = load_patients()

    # -- modelos/dados compartilhados (lazy) ----------------------------------------------
    @cached_property
    def vitals_population(self) -> vitals.PopulationModel:
        monitored = set(self.patients["bidmc_record"])
        return vitals.train_population_model(physionet.load_population(exclude=monitored))

    @cached_property
    def prescriptions_df(self) -> pd.DataFrame:
        return prescriptions.load()

    @cached_property
    def prescription_stats(self) -> dict:
        return prescriptions.population_stats(self.prescriptions_df)

    @cached_property
    def movement_df(self) -> pd.DataFrame:
        return movement.load()

    @cached_property
    def movement_model(self):
        return movement.train_population_model(self.movement_df)

    def patient(self, patient_id: str) -> dict:
        row = self.patients[self.patients["patient_id"] == patient_id]
        if row.empty:
            raise KeyError(f"Paciente {patient_id} não encontrado")
        return row.iloc[0].to_dict()

    def vitals_frame(self, patient_id: str) -> pd.DataFrame:
        return physionet.load_patient_vitals(patient_id, self.patient(patient_id)["bidmc_record"])

    # -- análise por modalidade -------------------------------------------------------------
    def analyze_vitals(self, patient_id: str) -> tuple[dict, list[Finding]]:
        df = self.vitals_frame(patient_id)
        findings, _ = vitals.detect(df, patient_id, self.vitals_population)
        return vitals.summary(df, findings), findings

    def analyze_audio(self, patient_id: str) -> tuple[dict, list[Finding]]:
        p = self.patient(patient_id)
        path = AUDIO_DIR / p["audio"]
        if not p["audio"] or not path.exists():
            return {"disponivel": False}, []
        report = analyze_consultation(path, patient_id, prefer_azure=self.prefer_azure,
                                      use_audioset_model=self.use_audioset_model)
        return report.to_dict(), report.findings

    def analyze_video(self, patient_id: str) -> tuple[dict, list[Finding]]:
        p = self.patient(patient_id)
        path = VIDEO_DIR / p["video"]
        if not p["video"] or not (path.exists() or path.with_name(f"{path.stem}_keypoints.json").exists()):
            return {"disponivel": False}, []
        report = analyze_video(path, patient_id)
        return report.to_dict(), report.findings

    def analyze_prescriptions(self, patient_id: str) -> tuple[dict, list[Finding]]:
        p = self.patient(patient_id)
        findings = prescriptions.detect(self.prescriptions_df, patient_id, p["alergias"], self.prescription_stats)
        return prescriptions.summary(self.prescriptions_df, patient_id, findings), findings

    def analyze_movement(self, patient_id: str) -> tuple[dict, list[Finding]]:
        findings = movement.detect(self.movement_df, patient_id, self.movement_model)
        return movement.summary(self.movement_df, patient_id, findings), findings

    # -- paciente completo ------------------------------------------------------------------
    def assess(self, patient_id: str, emit_alerts: bool = True) -> PatientAssessment:
        modalities, findings = {}, {}
        for name, fn in [("vitals", self.analyze_vitals), ("audio", self.analyze_audio), ("video", self.analyze_video),
                         ("prescriptions", self.analyze_prescriptions), ("movement", self.analyze_movement)]:
            modalities[name], findings[name] = fn(patient_id)
        fusion = fuse(patient_id, findings)
        findings["fusion"] = fusion.syndromes
        alerts = []
        if emit_alerts:
            alerts = [a.to_dict() for a in self.alerts.emit_all([f for fs in findings.values() for f in fs])]
        self.audit.log(
            "patient_assessment", patient_id,
            fused_score=fusion.score, level=fusion.level, modality_risks=fusion.modality_risks,
            n_findings={m: len(fs) for m, fs in findings.items()}, n_alerts=len(alerts),
            providers={
                "speech_to_text": modalities["audio"].get("transcricao", {}).get("provider"),
                "text_analytics": modalities["audio"].get("analise_texto", {}).get("provider"),
                "video": modalities["video"].get("fonte_deteccao"),
            },
        )
        return PatientAssessment(self.patient(patient_id), modalities, findings, fusion.to_dict(), alerts)

    def run_all(self, out_dir: Path = OUTPUT_DIR) -> list[PatientAssessment]:
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for pid in self.patients["patient_id"]:
            a = self.assess(pid)
            (out_dir / f"{pid}.json").write_text(json.dumps(a.to_dict(), ensure_ascii=False, indent=2, default=str),
                                                 encoding="utf-8")
            results.append(a)
        write_summary(results, REPORTS_DIR / "resultados_execucao.md")
        return results


def write_summary(results: list[PatientAssessment], path: Path) -> None:
    lines = ["# Resultados da execução — monitoramento multimodal", "",
             "Gerado por `python -m src.main run-all`. Valores reais desta execução.", "",
             "| Paciente | Cenário | Vitais | Áudio | Vídeo | Prescr. | Mov. | **Risco fundido** | Nível | Alertas |",
             "|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for a in results:
        r = a.fusion["risco_por_modalidade"]
        lines.append(
            f"| {a.patient['patient_id']} | {a.patient['diagnostico']} | {r['vitals']:.2f} | {r['audio']:.2f} | "
            f"{r['video']:.2f} | {r['prescriptions']:.2f} | {r['movement']:.2f} | **{a.fusion['score']:.2f}** | "
            f"{a.fusion['nivel']} | {len(a.alerts)} |")
    for a in results:
        lines += ["", f"## {a.patient['patient_id']} — {a.patient['diagnostico']}", "",
                  f"_Cenário injetado:_ {a.patient['cenario']}", ""]
        syn = a.fusion["sindromes"]
        if syn:
            lines.append("**Síndromes (fusão multimodal):**")
            lines += [f"- 🧩 **{s['message']}** → {s['evidence']['acao_sugerida']}" for s in syn]
            lines.append("")
        ordered = sorted((f for m, fs in a.findings.items() if m != "fusion" for f in fs),
                         key=lambda f: -SEVERITY_ORDER[f.severity])
        if ordered:
            lines.append("| Severidade | Modalidade | Achado | Método |")
            lines.append("|---|---|---|---|")
            lines += [f"| {f.severity} | {f.modality} | {f.message} | {f.method or ''} |" for f in ordered]
        else:
            lines.append("Nenhuma anomalia detectada.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
