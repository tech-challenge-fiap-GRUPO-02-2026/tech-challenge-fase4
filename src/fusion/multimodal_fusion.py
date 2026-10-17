"""Fusão multimodal (late fusion) dos achados de vídeo, áudio, texto, vitais, prescrições e movimentação.

Cada modalidade já entrega um risco 0–1. A fusão:

1. **Não dilui achado crítico** — o risco do paciente parte do *máximo*
   entre as modalidades (um SpO₂ de 83% é crítico mesmo que a voz esteja
   normal), e soma um reforço proporcional à **corroboração** das demais,
   ponderada por ``FUSION_WEIGHTS``;
2. **Reconhece síndromes cruzando modalidades** — padrões que nenhuma
   fonte isolada confirma, mas cuja combinação é clinicamente forte
   (ex.: disartria na voz + "boca torta" no relato + assimetria motora no
   vídeo → suspeita de AVC). Cada síndrome vira um achado ``fusion`` com
   a lista de evidências de cada modalidade (explicabilidade);
3. Sugere a **resposta da equipe** (nunca prescreve; aciona protocolos).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import FUSION_WEIGHTS, risk_level
from src.models import Finding, modality_risk

MODALITY_LABELS = {
    "vitals": "Sinais vitais", "audio": "Áudio/voz + texto", "video": "Vídeo",
    "prescriptions": "Prescrições", "movement": "Movimentação",
}


@dataclass
class Syndrome:
    kind: str
    title: str
    severity: str
    action: str
    requires: list[list[str]]  # lista de grupos; cada grupo = prefixos de kind (OR); todos os grupos (AND)
    min_modalities: int = 2    # corroboração mínima entre fontes distintas


SYNDROMES = [
    Syndrome(
        "suspeita_avc", "Suspeita de AVC / déficit neurológico agudo", "critico",
        "Acionar protocolo AVC (código AVC): avaliação neurológica imediata e escala NIHSS",
        [["voz_disartria", "termo_suspeita_avc"], ["assimetria_membros", "adm_reduzida", "termo_suspeita_avc", "termo_deficit_neurologico"]],
    ),
    Syndrome(
        "insuficiencia_respiratoria", "Desconforto/insuficiência respiratória", "critico",
        "Avaliação médica imediata da via aérea e oxigenação; considerar gasometria",
        [["voz_respiratorio", "termo_desconforto_respiratorio", "tosse_detectada"], ["spo2_baixo", "resp_alto"]],
    ),
    Syndrome(
        "possivel_choque", "Possível choque (séptico/hemodinâmico)", "critico",
        "Acionar time de resposta rápida; reavaliar perfusão e protocolo de sepse",
        [["hr_alto"], ["sbp_baixo", "dbp_baixo"]], min_modalities=1,  # índice de choque: FC × PA
    ),
    Syndrome(
        "queda_confirmada", "Queda com possível lesão", "critico",
        "Avaliar o paciente no leito (trauma, nível de consciência); notificar evento adverso",
        [["queda_detectada", "queda_leito", "termo_queda"], ["queda_detectada", "queda_leito", "termo_possivel_fratura", "termo_queda", "agitacao_noturna"]],
    ),
    Syndrome(
        "delirium", "Risco de delirium", "alto",
        "Avaliar com escala CAM-ICU; revisar medicações e privação de sono",
        [["agitacao_noturna"], ["termo_delirium", "voz_fadiga"]],
    ),
    Syndrome(
        "risco_seguranca_medicamentosa", "Risco de evento adverso medicamentoso", "critico",
        "Farmácia clínica + médico prescritor revisarem a prescrição antes da próxima dose",
        [["conflito_alergia", "dose_acima_maxima", "duplicidade_terapeutica", "salto_de_dose", "suspensao_anticoagulante"]],
        min_modalities=1,
    ),
]


@dataclass
class FusionResult:
    patient_id: str
    score: float
    level: str
    modality_risks: dict[str, float]
    syndromes: list[Finding] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id, "score": self.score, "nivel": self.level,
            "risco_por_modalidade": self.modality_risks,
            "sindromes": [s.to_dict() for s in self.syndromes], "acoes_sugeridas": self.actions,
        }


def _matching(findings: list[Finding], prefixes: list[str]) -> list[Finding]:
    return [f for f in findings if any(f.kind.startswith(p) for p in prefixes)]


def detect_syndromes(patient_id: str, findings: list[Finding]) -> list[Finding]:
    out = []
    for syn in SYNDROMES:
        groups = [_matching(findings, g) for g in syn.requires]
        if not all(groups):
            continue
        evidence = sorted({f"{MODALITY_LABELS.get(f.modality, f.modality)}: {f.message}" for g in groups for f in g})
        modalities = sorted({f.modality for g in groups for f in g})
        if len(modalities) < syn.min_modalities:
            continue
        # síndrome de um só grupo precisa de ≥ 2 achados (ou 1 crítico) para não repetir um achado isolado
        if len(groups) == 1 and len(groups[0]) < 2 and groups[0][0].severity != "critico":
            continue
        out.append(Finding(
            "fusion", syn.kind, syn.severity, patient_id=patient_id,
            message=f"{syn.title} — corroborado por {len(modalities)} modalidade(s): {', '.join(MODALITY_LABELS.get(m, m) for m in modalities)}",
            method="fusão multimodal (regras de síndrome)",
            evidence={"acao_sugerida": syn.action, "evidencias": evidence, "modalidades": modalities},
        ))
    return out


def fuse(patient_id: str, by_modality: dict[str, list[Finding]]) -> FusionResult:
    risks = {m: modality_risk(by_modality.get(m, [])) for m in FUSION_WEIGHTS}
    top_mod = max(risks, key=risks.get)
    top = risks[top_mod]
    corroboration = sum(FUSION_WEIGHTS[m] * r for m, r in risks.items() if m != top_mod)
    all_findings = [f for fs in by_modality.values() for f in fs]
    syndromes = detect_syndromes(patient_id, all_findings)
    syndrome_boost = 0.1 * len(syndromes)
    score = round(min(1.0, top + 0.5 * corroboration + syndrome_boost), 3)
    actions = [s.evidence["acao_sugerida"] for s in syndromes]
    if not actions and score >= 0.5:
        actions.append("Reavaliação médica à beira-leito dos achados listados")
    return FusionResult(patient_id, score, risk_level(score), risks, syndromes, actions)
