"""Pipeline de áudio da consulta: sinal → biomarcadores → transcrição → texto → achados.

    WAV ─┬─ features.py ── voice_analyzer.py ──────────────┐
         │   (eventos AudioSet: tosse/respiração)          ├─► achados (Finding) + risco de áudio
         ├─ audioset_classifier.py (AST, opcional) ────────┤
         └─ Azure Speech to Text ── Azure Text Analytics ──┘
                (termos críticos, sentimento)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.audio import audioset_classifier
from src.audio.azure_speech import Transcript, transcribe
from src.audio.azure_text_analytics import TextAnalysis, analyze_text
from src.audio.features import VoiceFeatures, extract_features
from src.audio.voice_analyzer import LABELS, VoiceAssessment, assess
from src.audio.wav_io import read_wav
from src.datasets.audioset import PT_NAMES, mid_for
from src.models import Finding, modality_risk

TERM_SEVERITY_TO_FINDING = {"critico": "critico", "alto": "alto", "moderado": "moderado"}


@dataclass
class AudioReport:
    patient_id: str | None
    source: str
    features: VoiceFeatures
    voice: VoiceAssessment
    transcript: Transcript
    text: TextAnalysis
    audioset_model: dict | None
    findings: list[Finding] = field(default_factory=list)

    @property
    def risk(self) -> float:
        return modality_risk(self.findings)

    def to_dict(self) -> dict:
        events = [
            {**e, "mid": mid_for(e["label"]), "nome_pt": PT_NAMES.get(e["label"], e["label"])}
            for e in self.features.to_dict()["events"]
        ]
        return {
            "patient_id": self.patient_id,
            "arquivo": self.source,
            "duracao_s": round(self.features.duration_s, 2),
            "biomarcadores": {k: v for k, v in self.features.to_dict().items() if k != "events"},
            "eventos_audioset": events,
            "classificador_audioset_ast": self.audioset_model,
            "avaliacao_vocal": self.voice.to_dict(),
            "transcricao": self.transcript.to_dict(),
            "analise_texto": self.text.to_dict(),
            "achados": [f.to_dict() for f in self.findings],
            "risco_audio": self.risk,
        }


def _voice_severity(score: float) -> str:
    return "alto" if score >= 0.75 else "moderado"


def build_findings(patient_id, voice: VoiceAssessment, text: TextAnalysis, features: VoiceFeatures) -> list[Finding]:
    findings: list[Finding] = []
    for indicator, score in voice.scores.items():
        if score < 0.5:
            continue
        findings.append(Finding(
            modality="audio", kind=f"voz_{indicator}", severity=_voice_severity(score), patient_id=patient_id,
            message=f"{LABELS[indicator]} detectada na voz (score {score:.2f})",
            method="biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet)",
            evidence={"score": round(score, 3), "marcadores": voice.triggered[indicator]},
        ))
    coughs = int(voice.metrics.get("cough_episodes", 0))
    if coughs:
        findings.append(Finding(
            modality="audio", kind="tosse_detectada", severity="moderado", patient_id=patient_id,
            message=f"{coughs} episódio(s) de tosse durante a consulta",
            method="detector acústico (classe AudioSet Cough /m/01b_21)",
            evidence={"inicios_s": [e.start_s for e in features.events if e.label == "Cough"]},
        ))
    seen = set()
    for term in text.critical_terms:
        key = term.signal
        if key in seen:
            continue
        seen.add(key)
        findings.append(Finding(
            modality="audio", kind=f"termo_{term.signal}", severity=TERM_SEVERITY_TO_FINDING[term.severity],
            patient_id=patient_id, message=f"Paciente relatou \"{term.text}\" ({term.signal.replace('_', ' ')})",
            method=f"{text.provider} sobre a transcrição",
            evidence={"categoria": term.category, "confianca": term.confidence},
        ))
    neg = text.sentiment.get("negativo", 0.0)
    if text.sentiment.get("sentimento") == "negative" and neg >= 0.6:
        findings.append(Finding(
            modality="audio", kind="sentimento_negativo", severity="info", patient_id=patient_id,
            message=f"Sentimento predominantemente negativo na consulta ({neg:.0%})",
            method=text.provider, evidence=text.sentiment,
        ))
    return findings


def analyze_consultation(wav_path: str | Path, patient_id: str | None = None, *, prefer_azure: bool = True,
                         use_audioset_model: bool = False, baseline_wav: str | Path | None = None) -> AudioReport:
    signal, sr = read_wav(wav_path)
    features = extract_features(signal, sr)
    baseline = extract_features(*read_wav(baseline_wav)) if baseline_wav else None
    voice = assess(features, baseline)
    transcript = transcribe(wav_path, prefer_azure=prefer_azure)
    text = analyze_text(transcript.text, prefer_azure=prefer_azure) if transcript.text else analyze_text("", False)

    ast = None
    if use_audioset_model and audioset_classifier.available():
        detections = audioset_classifier.classify(signal, sr)
        ast = {"modelo": audioset_classifier.MODEL_ID, "deteccoes": detections,
               "resumo": audioset_classifier.summarize(detections)}

    findings = build_findings(patient_id, voice, text, features)
    return AudioReport(patient_id, str(Path(wav_path).name), features, voice, transcript, text, ast, findings)
