"""Tipos comuns a todas as modalidades: achados (findings) e relatórios."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEVERITY_ORDER = {"info": 0, "moderado": 1, "alto": 2, "critico": 3}
SEVERITY_SCORE = {"info": 0.1, "moderado": 0.4, "alto": 0.7, "critico": 0.95}


@dataclass
class Finding:
    """Um achado anômalo de qualquer modalidade — a unidade que vira alerta."""

    modality: str           # vitals | audio | video | prescriptions | movement | fusion
    kind: str               # identificador estável do tipo de anomalia (ex.: "spo2_baixa")
    severity: str           # info | moderado | alto | critico
    message: str            # texto legível para a equipe médica
    patient_id: str | None = None
    time_ref: str | None = None   # "t=245s", timestamp ISO, "00:12 do vídeo"...
    method: str | None = None     # regra clínica, z-score robusto, IsolationForest...
    evidence: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return SEVERITY_SCORE[self.severity]

    def to_dict(self) -> dict:
        return asdict(self)


def max_severity(findings: list[Finding]) -> str:
    if not findings:
        return "info"
    return max((f.severity for f in findings), key=SEVERITY_ORDER.__getitem__)


def modality_risk(findings: list[Finding]) -> float:
    """Risco 0–1 da modalidade: severidade máxima + reforço leve por achados adicionais."""
    relevant = [f for f in findings if f.severity != "info"]
    if not relevant:
        return 0.0
    top = max(f.score for f in relevant)
    return round(min(1.0, top + 0.03 * (len(relevant) - 1)), 3)
