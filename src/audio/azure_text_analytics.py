"""Termos críticos e sentimento — Azure Text Analytics (Language) com fallback local.

Provider Azure (Language Service, ``api-version=2023-04-01``):
* ``SentimentAnalysis`` (com *opinion mining*) → sentimento por frase;
* ``KeyPhraseExtraction`` → frases-chave;
* ``Healthcare`` (Text Analytics for Health, job assíncrono) → entidades
  clínicas (sintomas, diagnósticos, medicamentos...) com asserção de
  negação/certeza.

As entidades do Azure são cruzadas com o léxico do hospital
(``clinical_terms.py``) para receber uma severidade operacional. Sem
chave, o léxico local faz o papel completo (provider ``local_lexico``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.audio.clinical_terms import SEVERITY_WEIGHT, ClinicalTerm, find_terms, lexicon_sentiment, normalize
from src.cloud.azure_client import AzureError, AzureSettings, http_request, post_json

API_VERSION = "2023-04-01"


@dataclass
class TextAnalysis:
    provider: str
    sentiment: dict
    terms: list[ClinicalTerm]
    key_phrases: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def critical_terms(self) -> list[ClinicalTerm]:
        return [t for t in self.terms if not t.negated and t.severity in ("critico", "alto")]

    @property
    def risk(self) -> float:
        """Risco textual: termo mais grave não negado, reforçado por sentimento negativo."""
        active = [SEVERITY_WEIGHT[t.severity] for t in self.terms if not t.negated]
        base = max(active, default=0.0)
        return round(min(1.0, base * (0.85 + 0.15 * self.sentiment.get("negativo", 0.0))), 3)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "sentimento": self.sentiment,
            "frases_chave": self.key_phrases,
            "termos": [t.to_dict() for t in self.terms],
            "termos_criticos": sorted({t.text.lower() for t in self.critical_terms}),
            "risco_texto": self.risk,
            "erro_azure": self.error,
        }


class AzureTextAnalytics:
    def __init__(self, settings: AzureSettings | None = None):
        self.settings = settings or AzureSettings.from_env()

    @property
    def available(self) -> bool:
        return self.settings.language_available

    def _docs(self, text: str, language: str | None = None) -> dict:
        return {"documents": [{"id": "1", "language": language or self.settings.language, "text": text}]}

    def _sync(self, kind: str, text: str, parameters: dict | None = None) -> dict:
        url = f"{self.settings.language_endpoint}/language/:analyze-text?api-version={API_VERSION}"
        body = {"kind": kind, "analysisInput": self._docs(text), "parameters": parameters or {}}
        _, _, payload = post_json(url, self.settings.language_key, body)
        docs = payload.get("results", {}).get("documents", [])
        if not docs:
            errors = payload.get("results", {}).get("errors", [])
            raise AzureError(f"{kind}: sem documentos no retorno ({errors})")
        return docs[0]

    def sentiment(self, text: str) -> dict:
        doc = self._sync("SentimentAnalysis", text, {"opinionMining": True})
        scores = doc.get("confidenceScores", {})
        return {
            "sentimento": doc.get("sentiment"),
            "positivo": scores.get("positive", 0.0),
            "neutro": scores.get("neutral", 0.0),
            "negativo": scores.get("negative", 0.0),
            "frases": [
                {"texto": s.get("text"), "sentimento": s.get("sentiment")} for s in doc.get("sentences", [])
            ],
        }

    def key_phrases(self, text: str) -> list[str]:
        return self._sync("KeyPhraseExtraction", text).get("keyPhrases", [])

    def healthcare_entities(self, text: str, poll_s: float = 1.0, timeout_s: float = 60.0) -> list[ClinicalTerm]:
        url = f"{self.settings.language_endpoint}/language/analyze-text/jobs?api-version={API_VERSION}"
        body = {
            "displayName": "consulta-monitoramento",
            "analysisInput": self._docs(text),
            "tasks": [{"kind": "Healthcare", "taskName": "ta4h"}],
        }
        _, headers, _ = post_json(url, self.settings.language_key, body)
        op = headers.get("operation-location") or headers.get("Operation-Location")
        if not op:
            raise AzureError("Healthcare: resposta sem operation-location")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            _, _, job = http_request("GET", op, headers={"Ocp-Apim-Subscription-Key": self.settings.language_key})
            status = job.get("status")
            if status == "succeeded":
                items = job.get("tasks", {}).get("items", [])
                docs = items[0].get("results", {}).get("documents", []) if items else []
                return [self._to_term(e) for e in (docs[0].get("entities", []) if docs else [])]
            if status in ("failed", "cancelled"):
                raise AzureError(f"Healthcare job {status}: {job.get('errors')}")
            time.sleep(poll_s)
        raise AzureError("Healthcare job excedeu o tempo limite")

    @staticmethod
    def _to_term(entity: dict) -> ClinicalTerm:
        text = entity.get("text", "")
        # severidade operacional vem do léxico do hospital; entidade desconhecida = moderado
        local = find_terms(text)
        severity = local[0].severity if local else "moderado"
        signal = local[0].signal if local else normalize(entity.get("category", "achado")).lower()
        certainty = (entity.get("assertion") or {}).get("certainty", "")
        return ClinicalTerm(
            text=text, category=entity.get("category", ""), severity=severity, signal=signal,
            negated=certainty.startswith("negative"), offset=entity.get("offset", 0),
            confidence=entity.get("confidenceScore", 0.0), source="azure_text_analytics_for_health",
        )

    def analyze(self, text: str) -> TextAnalysis:
        sentiment = self.sentiment(text)
        phrases = self.key_phrases(text)
        try:
            entities = self.healthcare_entities(text)
        except AzureError:
            entities = []
        # termos do léxico que o modelo genérico não reconheceu (ex.: "boca torta") também entram
        known = {normalize(t.text) for t in entities}
        extra = [t for t in find_terms(text) if normalize(t.text) not in known]
        return TextAnalysis(provider="azure_text_analytics", sentiment=sentiment,
                            terms=sorted(entities + extra, key=lambda t: t.offset), key_phrases=phrases)


def analyze_local(text: str) -> TextAnalysis:
    return TextAnalysis(provider="local_lexico", sentiment=lexicon_sentiment(text), terms=find_terms(text))


def analyze_text(text: str, prefer_azure: bool = True) -> TextAnalysis:
    azure = AzureTextAnalytics()
    if not (prefer_azure and azure.available):
        return analyze_local(text)
    try:
        return azure.analyze(text)
    except AzureError as exc:
        result = analyze_local(text)
        result.error = str(exc)
        return result
