"""Indicadores clínicos de voz: fadiga, disartria e dificuldade respiratória.

Cada indicador é uma combinação ponderada de biomarcadores vocais
(``features.py``), cada um normalizado em 0–1 por uma rampa linear entre
o valor típico de fala saudável e o valor típico do quadro alterado.
As âncoras foram calibradas na gravação de controle (PAC-0001) e estão
documentadas abaixo — num hospital real seriam recalibradas com a voz
basal do próprio paciente (admissão), o que o parâmetro ``baseline``
permite.

O resultado é **triagem**, não diagnóstico: o relatório sempre indica
quais marcadores dispararam, para que o fonoaudiólogo/médico avalie.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.audio.features import VoiceFeatures

# (feature, valor_normal, valor_alterado, peso)
INDICATORS: dict[str, list[tuple[str, float, float, float]]] = {
    "fadiga": [
        ("speech_rate_syl_s", 2.6, 1.8, 0.25),
        ("pause_ratio", 0.20, 0.45, 0.20),
        ("energy_slope_db_min", -11.0, -18.0, 0.30),
        ("long_pauses_per_min", 1.0, 6.0, 0.15),
        ("mean_phrase_s", 1.5, 0.9, 0.10),
    ],
    "disartria": [
        ("tremor_index", 0.07, 0.12, 0.30),
        ("hf_energy_db", -28.0, -36.0, 0.30),
        ("articulation_rate_syl_s", 3.5, 3.0, 0.15),
        ("jitter_pct", 1.5, 2.5, 0.10),
        ("mean_phrase_s", 1.2, 0.6, 0.15),
    ],
    "respiratorio": [
        ("cough_episodes", 0.0, 2.0, 0.35),
        ("breaths_per_min", 2.0, 15.0, 0.35),
        ("mean_phrase_s", 1.5, 0.9, 0.15),
        ("pause_ratio", 0.20, 0.45, 0.15),
    ],
}

LABELS = {
    "fadiga": "Fadiga / cansaço vocal",
    "disartria": "Disartria (fala arrastada/imprecisa)",
    "respiratorio": "Dificuldade respiratória",
}

FEATURE_NAMES = {
    "speech_rate_syl_s": "taxa de fala (sílabas/s)",
    "articulation_rate_syl_s": "taxa de articulação (sílabas/s)",
    "pause_ratio": "proporção de pausas",
    "energy_slope_db_min": "queda de energia vocal (dB/min)",
    "long_pauses_per_min": "pausas longas por minuto",
    "mean_phrase_s": "duração média das frases (s)",
    "tremor_index": "índice de tremor vocal (8–12 Hz)",
    "hf_energy_db": "energia de consoantes (dB)",
    "jitter_pct": "perturbação de F0 (jitter %)",
    "cough_episodes": "episódios de tosse",
    "breaths_per_min": "inspirações audíveis por minuto",
}

ALERT_THRESHOLD = 0.5


@dataclass
class VoiceAssessment:
    scores: dict[str, float]
    triggered: dict[str, list[str]]
    metrics: dict[str, float]
    flags: list[str] = field(default_factory=list)

    @property
    def risk(self) -> float:
        return max(self.scores.values()) if self.scores else 0.0

    def to_dict(self) -> dict:
        return {
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "marcadores_disparados": self.triggered,
            "metricas": {k: round(v, 3) for k, v in self.metrics.items()},
            "flags": self.flags,
            "risco_voz": round(self.risk, 3),
        }


def _ramp(value: float, normal: float, altered: float) -> float:
    if altered == normal:
        return 0.0
    return float(min(1.0, max(0.0, (value - normal) / (altered - normal))))


def cough_episodes(features: VoiceFeatures, gap_s: float = 0.6) -> int:
    """Agrupa rajadas de tosse próximas em episódios."""
    episodes, last_end = 0, float("-inf")
    for e in sorted((e for e in features.events if e.label == "Cough"), key=lambda e: e.start_s):
        if e.start_s - last_end > gap_s:
            episodes += 1
        last_end = e.end_s
    return episodes


def derived_metrics(features: VoiceFeatures) -> dict[str, float]:
    metrics = {k: v for k, v in features.to_dict().items() if isinstance(v, (int, float))}
    minutes = max(features.duration_s / 60, 1e-6)
    metrics["cough_episodes"] = float(cough_episodes(features))
    metrics["breaths_per_min"] = (features.count("Breathing") + features.count("Sigh")) / minutes
    return metrics


def assess(features: VoiceFeatures, baseline: VoiceFeatures | None = None) -> VoiceAssessment:
    metrics = derived_metrics(features)
    base = derived_metrics(baseline) if baseline else None
    scores, triggered = {}, {}
    for indicator, components in INDICATORS.items():
        total, hits = 0.0, []
        for feat, normal, altered, weight in components:
            if base is not None and feat in base:
                # desloca as âncoras para a voz basal do próprio paciente
                shift = base[feat] - normal
                normal, altered = normal + shift, altered + shift
            s = _ramp(metrics[feat], normal, altered)
            total += weight * s
            if s >= 0.5:
                hits.append(f"{FEATURE_NAMES[feat]} = {metrics[feat]:.2f}")
        scores[indicator] = total
        triggered[indicator] = hits
    flags = [f"voz_{k}" for k, v in scores.items() if v >= ALERT_THRESHOLD]
    return VoiceAssessment(scores=scores, triggered=triggered, metrics=metrics, flags=flags)
