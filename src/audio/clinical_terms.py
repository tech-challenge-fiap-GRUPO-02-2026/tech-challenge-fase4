"""Léxico de termos clínicos críticos em pt-BR (com detecção de negação).

Serve a dois propósitos:
1. **Fallback local** do Azure Text Analytics for Health quando não há chave;
2. **Graduação de severidade** das entidades que o Azure devolve — o Azure
   diz *o que* é (SymptomOrSign, Diagnosis...), o léxico do hospital diz
   *quão urgente* é para a equipe (critico/alto/moderado).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SEVERITY_WEIGHT = {"critico": 1.0, "alto": 0.7, "moderado": 0.4}

# termo normalizado (sem acento) → (categoria, severidade, sinal de alerta associado)
TERMS: dict[str, tuple[str, str, str]] = {
    "dor no peito": ("SymptomOrSign", "critico", "possivel_sindrome_coronariana"),
    "dor toracica": ("SymptomOrSign", "critico", "possivel_sindrome_coronariana"),
    "falta de ar": ("SymptomOrSign", "critico", "desconforto_respiratorio"),
    "nao consigo respirar": ("SymptomOrSign", "critico", "desconforto_respiratorio"),
    "chiado no peito": ("SymptomOrSign", "alto", "broncoespasmo"),
    "fala esta enrolada": ("SymptomOrSign", "critico", "suspeita_avc"),
    "fala enrolada": ("SymptomOrSign", "critico", "suspeita_avc"),
    "boca esta torta": ("SymptomOrSign", "critico", "suspeita_avc"),
    "boca torta": ("SymptomOrSign", "critico", "suspeita_avc"),
    "braco esquerdo fraco": ("SymptomOrSign", "critico", "suspeita_avc"),
    "braco direito fraco": ("SymptomOrSign", "critico", "suspeita_avc"),
    "formigamento": ("SymptomOrSign", "alto", "deficit_neurologico"),
    "engasgo": ("SymptomOrSign", "alto", "disfagia_risco_aspiracao"),
    "desmaio": ("SymptomOrSign", "critico", "sincope"),
    "desmaiei": ("SymptomOrSign", "critico", "sincope"),
    "confusa": ("SymptomOrSign", "alto", "delirium"),
    "confuso": ("SymptomOrSign", "alto", "delirium"),
    "cai": ("Event", "alto", "queda"),  # "caí" após normalização
    "queda": ("Event", "alto", "queda"),
    "sangramento": ("SymptomOrSign", "critico", "hemorragia"),
    "febre": ("SymptomOrSign", "alto", "sinal_infeccioso"),
    "calafrios": ("SymptomOrSign", "alto", "sinal_infeccioso"),
    "coracao esta acelerado": ("SymptomOrSign", "alto", "taquicardia_referida"),
    "palpitacao": ("SymptomOrSign", "alto", "taquicardia_referida"),
    "tontura": ("SymptomOrSign", "moderado", "instabilidade_hemodinamica"),
    "cansada": ("SymptomOrSign", "moderado", "fadiga"),
    "cansado": ("SymptomOrSign", "moderado", "fadiga"),
    "sem forca": ("SymptomOrSign", "moderado", "fadiga"),
    "fraca": ("SymptomOrSign", "moderado", "fadiga"),
    "fraco": ("SymptomOrSign", "moderado", "fadiga"),
    "tosse": ("SymptomOrSign", "moderado", "sintoma_respiratorio"),
    "catarro": ("SymptomOrSign", "moderado", "sintoma_respiratorio"),
    "dor no quadril": ("SymptomOrSign", "alto", "possivel_fratura_pos_queda"),
    "dor": ("SymptomOrSign", "moderado", "dor"),
    "doendo": ("SymptomOrSign", "moderado", "dor"),
    "ansioso": ("SymptomOrSign", "moderado", "ansiedade"),
    "ansiosa": ("SymptomOrSign", "moderado", "ansiedade"),
    "medo": ("SymptomOrSign", "moderado", "ansiedade"),
}

NEGATIONS = ("nao", "nem", "nunca", "sem", "nenhum", "nenhuma")
POSITIVE_WORDS = {"bem", "melhor", "melhorou", "diminuiu", "consegui", "otimo", "tranquilo", "correu bem", "alivio"}
NEGATIVE_WORDS = {"piorou", "piorar", "pior", "muita", "medo", "nao consigo", "cansada", "cansado", "dor", "fraca",
                  "fraco", "tontura", "febre", "confusa", "ansioso", "ansiosa", "enrolada", "torta", "engasgo", "sem forca"}


@dataclass
class ClinicalTerm:
    text: str
    category: str
    severity: str
    signal: str
    negated: bool
    offset: int
    confidence: float = 1.0
    source: str = "lexico_local"

    def to_dict(self) -> dict:
        return {
            "texto": self.text, "categoria": self.category, "severidade": self.severity,
            "sinal": self.signal, "negado": self.negated, "offset": self.offset,
            "confianca": self.confidence, "fonte": self.source,
        }


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _is_negated(norm: str, start: int) -> bool:
    window = norm[max(0, start - 25):start]
    tokens = re.findall(r"[a-z]+", window)[-3:]
    return any(t in NEGATIONS for t in tokens)


def find_terms(text: str) -> list[ClinicalTerm]:
    """Busca gulosa (termos mais longos primeiro), sem sobreposição."""
    norm = normalize(text)
    taken = [False] * len(norm)
    found: list[ClinicalTerm] = []
    for term in sorted(TERMS, key=len, reverse=True):
        nterm = normalize(term)
        for m in re.finditer(rf"\b{re.escape(nterm)}\b", norm):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            category, severity, signal = TERMS[term]
            found.append(ClinicalTerm(text[m.start():m.end()], category, severity, signal,
                                      _is_negated(norm, m.start()), m.start()))
    return sorted(found, key=lambda t: t.offset)


def lexicon_sentiment(text: str) -> dict:
    """Sentimento por léxico (fallback do Azure SentimentAnalysis)."""
    norm = normalize(text)
    pos = sum(len(re.findall(rf"\b{re.escape(w)}\b", norm)) for w in POSITIVE_WORDS)
    neg = sum(len(re.findall(rf"\b{re.escape(w)}\b", norm)) for w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return {"sentimento": "neutral", "positivo": 0.0, "neutro": 1.0, "negativo": 0.0}
    p, n = pos / total, neg / total
    label = "positive" if p > 0.6 else "negative" if n > 0.6 else "mixed"
    return {"sentimento": label, "positivo": round(p, 2), "neutro": 0.0, "negativo": round(n, 2)}
