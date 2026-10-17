"""Configuração e cliente HTTP mínimo para os Azure Cognitive Services.

Usa só ``urllib`` (stdlib): nenhuma SDK é obrigatória. As credenciais vêm
de variáveis de ambiente (ou de um arquivo ``.env`` na raiz, nunca
versionado):

    AZURE_SPEECH_KEY, AZURE_SPEECH_REGION        → Speech to Text
    AZURE_LANGUAGE_KEY, AZURE_LANGUAGE_ENDPOINT  → Language / Text Analytics

Sem credenciais, cada serviço declara ``available = False`` e o pipeline
usa automaticamente o fallback local equivalente — o resultado indica
sempre qual provider foi usado (``provider: "azure"`` ou ``"local"``).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.config import BASE_DIR


def load_dotenv(path=BASE_DIR / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


@dataclass
class AzureSettings:
    speech_key: str | None
    speech_region: str
    language_key: str | None
    language_endpoint: str | None
    language: str

    @classmethod
    def from_env(cls) -> "AzureSettings":
        endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        return cls(
            speech_key=os.getenv("AZURE_SPEECH_KEY"),
            speech_region=os.getenv("AZURE_SPEECH_REGION", "brazilsouth"),
            language_key=os.getenv("AZURE_LANGUAGE_KEY"),
            language_endpoint=endpoint.rstrip("/") if endpoint else None,
            language=os.getenv("AZURE_LANGUAGE", "pt-BR"),
        )

    @property
    def speech_available(self) -> bool:
        return bool(self.speech_key)

    @property
    def language_available(self) -> bool:
        return bool(self.language_key and self.language_endpoint)

    def status(self) -> dict:
        return {
            "speech_to_text": {"configurado": self.speech_available, "regiao": self.speech_region},
            "text_analytics": {"configurado": self.language_available, "endpoint": self.language_endpoint},
            "idioma": self.language,
        }


class AzureError(RuntimeError):
    pass


def http_request(method: str, url: str, *, headers: dict, body: bytes | None = None,
                 timeout: float = 30.0) -> tuple[int, dict, dict]:
    """Executa a requisição e devolve (status, headers, json)."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            payload = json.loads(raw) if raw else {}
            return resp.status, dict(resp.headers), payload
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise AzureError(f"HTTP {exc.code} em {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AzureError(f"Falha de rede em {url}: {exc.reason}") from exc


def post_json(url: str, key: str, payload: dict, timeout: float = 30.0) -> tuple[int, dict, dict]:
    return http_request(
        "POST", url,
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"},
        body=json.dumps(payload).encode("utf-8"), timeout=timeout,
    )
