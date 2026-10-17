"""Análise de vídeo de cirurgia: contagem de instrumental e áreas críticas.

Regras de segurança cirúrgica (baseadas no checklist da OMS "Cirurgia
Segura" e nas recomendações AORN de contagem de instrumentais):

1. **Contagem de instrumental** — a contagem na mesa no fechamento deve
   ser igual à inicial. Divergência = possível item retido (evento
   "never event"); se o item faltante é visto na área da incisão, o
   alerta é crítico e aponta a localização;
2. **Instrumento fora do campo estéril** — instrumento detectado fora da
   mesa e do campo (ex.: caiu no chão) → contaminação, não pode voltar;
3. **Quebra de área estéril** — pessoa não paramentada detectada dentro
   da mesa de instrumental ou do campo cirúrgico.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.models import Finding
from src.video.objects import inside, overlaps

INSTRUMENT_CLASSES = {"bisturi", "tesoura", "pinca", "porta_agulha", "afastador", "compressa"}
UNSTERILE_CLASSES = {"pessoa_nao_paramentada"}
COUNT_WINDOW_S = 1.5


@dataclass
class ProcedureReport:
    metrics: dict
    timeline: list[dict]
    findings: list[Finding] = field(default_factory=list)
    key_frames: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"metricas": self.metrics, "linha_do_tempo": self.timeline,
                "achados": [f.to_dict() for f in self.findings], "frames_chave": self.key_frames}


def _count_on_table(frames: list[list[dict]], table) -> Counter:
    """Contagem modal por classe na mesa ao longo da janela (robusta a oclusão de 1 frame)."""
    per_frame = [Counter(d["classe"] for d in dets if d["classe"] in INSTRUMENT_CLASSES and inside(d["bbox"], table))
                 for dets in frames]
    classes = set().union(*per_frame) if per_frame else set()
    result = Counter()
    for c in classes:
        values = Counter(pf[c] for pf in per_frame)
        result[c] = values.most_common(1)[0][0]
    return result


def _first_time(frames, fps, predicate) -> float | None:
    for i, dets in enumerate(frames):
        if any(predicate(d) for d in dets):
            return round(i / fps, 2)
    return None


def analyze_surgery(frames: list[list[dict]], fps: float, zones: dict, patient_id: str | None = None) -> ProcedureReport:
    table, field_zone, incision = zones["mesa_instrumental"], zones["campo_cirurgico"], zones["incisao"]
    win = int(COUNT_WINDOW_S * fps)
    start_count = _count_on_table(frames[:win], table)
    end_count = _count_on_table(frames[-win:], table)
    findings, timeline, key_frames = [], [], []

    missing = start_count - end_count
    if missing:
        last = frames[-1]
        located = []
        for cls in missing:
            in_incision = [d for d in last if d["classe"] == cls and inside(d["bbox"], incision)]
            if in_incision:
                located.append(f"{cls} na área da incisão")
        n_total = sum(missing.values())
        msg = f"Contagem final de instrumental não confere: {sum(end_count.values())}/{sum(start_count.values())} na mesa — faltam {dict(missing)}"
        if located:
            msg += f"; detectado(s): {', '.join(located)} (possível item retido)"
        findings.append(Finding(
            "video", "falha_contagem_instrumental", "critico", patient_id=patient_id, message=msg,
            time_ref=f"fechamento ({len(frames) / fps:.1f}s)", method="YOLO: contagem por classe na mesa (início × fechamento)",
            evidence={"inicial": dict(start_count), "final": dict(end_count), "faltantes": dict(missing),
                      "localizados": located, "n_faltantes": n_total},
        ))
        key_frames.append({"frame": len(frames) - 1, "motivo": "contagem final divergente"})
        timeline.append({"t": round(len(frames) / fps, 2), "evento": "contagem_final_divergente"})

    # instrumento fora de qualquer área estéril (sustentado por ≥ 0,5 s)
    outside_since: dict[str, int] = {}
    reported = set()
    for i, dets in enumerate(frames):
        outside = {d["classe"]: d for d in dets if d["classe"] in INSTRUMENT_CLASSES
                   and not inside(d["bbox"], table) and not inside(d["bbox"], field_zone)}
        for key in list(outside_since):
            if key not in outside:
                outside_since.pop(key)
        for key, d in outside.items():
            outside_since.setdefault(key, i)
            if i - outside_since[key] >= fps * 0.5 and key not in reported:
                reported.add(key)
                t0 = outside_since[key] / fps
                findings.append(Finding(
                    "video", "instrumento_fora_campo_esteril", "alto", patient_id=patient_id,
                    message=f"{key} saiu do campo estéril (provável queda) em {t0:.1f}s — considerar contaminado",
                    time_ref=f"{t0:.1f}s", method="YOLO: posição do instrumento vs. zonas estéreis",
                    evidence={"bbox": d["bbox"]},
                ))
                key_frames.append({"frame": i, "motivo": f"{key} fora do campo estéril"})
                timeline.append({"t": round(t0, 2), "evento": f"{key}_fora_campo"})

    # quebra de área estéril
    breach = [i for i, dets in enumerate(frames)
              if any(d["classe"] in UNSTERILE_CLASSES and (overlaps(d["bbox"], table) or overlaps(d["bbox"], field_zone))
                     for d in dets)]
    if breach:
        t0, t1 = breach[0] / fps, breach[-1] / fps
        findings.append(Finding(
            "video", "quebra_area_esteril", "alto", patient_id=patient_id,
            message=f"Pessoa não paramentada na área estéril entre {t0:.1f}s e {t1:.1f}s ({t1 - t0:.1f}s)",
            time_ref=f"{t0:.1f}s–{t1:.1f}s", method="YOLO: pessoa × zonas estéreis",
            evidence={"inicio_s": round(t0, 2), "fim_s": round(t1, 2)},
        ))
        key_frames.append({"frame": breach[len(breach) // 2], "motivo": "quebra de área estéril"})
        timeline.append({"t": round(t0, 2), "evento": "quebra_area_esteril"})

    metrics = {
        "duracao_s": round(len(frames) / fps, 2),
        "contagem_inicial": dict(start_count), "contagem_final": dict(end_count),
        "primeira_deteccao_incisao_s": _first_time(frames, fps, lambda d: d["classe"] in INSTRUMENT_CLASSES and inside(d["bbox"], incision)),
    }
    return ProcedureReport(metrics, sorted(timeline, key=lambda e: e["t"]), findings, key_frames)
