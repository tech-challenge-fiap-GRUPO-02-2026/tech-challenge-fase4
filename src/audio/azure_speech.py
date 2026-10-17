"""Transcrição das consultas — Azure Speech to Text (REST) com fallback local.

Provider Azure: API REST de áudio curto
(``/speech/recognition/conversation/cognitiveservices/v1``), formato
``detailed`` para obter a confiança. Gravações acima de ~55 s são
divididas em blocos nos silêncios antes do envio (limite de 60 s da API).

Fallback local: usa a *transcrição de referência* (``.txt`` ao lado do
``.wav``), marcada como ``provider="local_referencia"`` — o resultado
nunca finge ter vindo do Azure.

Quando ambos existem, calcula o **WER** (word error rate) do Azure contra
a referência, métrica reportada no relatório técnico.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.audio.wav_io import TARGET_SR, read_wav
from src.cloud.azure_client import AzureError, AzureSettings, http_request

MAX_CHUNK_S = 55.0


@dataclass
class Transcript:
    text: str
    provider: str
    confidence: float | None = None
    segments: list[dict] = field(default_factory=list)
    wer_vs_reference: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "texto": self.text,
            "provider": self.provider,
            "confianca": self.confidence,
            "segmentos": self.segments,
            "wer_vs_referencia": self.wer_vs_reference,
            "erro_azure": self.error,
        }


def _normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalize_words(reference), _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[-1, -1] / len(ref))


def _pcm_wav(signal: np.ndarray) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes((np.clip(signal, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def split_on_silence(signal: np.ndarray, max_s: float = MAX_CHUNK_S) -> list[np.ndarray]:
    """Divide em blocos <= max_s cortando no ponto de menor energia perto do limite."""
    max_n = int(max_s * TARGET_SR)
    chunks, start = [], 0
    while len(signal) - start > max_n:
        window = signal[start + max_n - 5 * TARGET_SR: start + max_n]
        frame = 800
        energies = [np.mean(window[i:i + frame] ** 2) for i in range(0, len(window) - frame, frame)]
        cut = start + max_n - 5 * TARGET_SR + int(np.argmin(energies)) * frame
        chunks.append(signal[start:cut])
        start = cut
    chunks.append(signal[start:])
    return chunks


class AzureSpeechToText:
    def __init__(self, settings: AzureSettings | None = None):
        self.settings = settings or AzureSettings.from_env()

    @property
    def available(self) -> bool:
        return self.settings.speech_available

    def _url(self) -> str:
        return (
            f"https://{self.settings.speech_region}.stt.speech.microsoft.com/speech/recognition/"
            f"conversation/cognitiveservices/v1?language={self.settings.language}&format=detailed&profanity=raw"
        )

    def transcribe(self, wav_path: str | Path) -> Transcript:
        signal, _ = read_wav(wav_path)
        texts, confs, segments, offset = [], [], [], 0.0
        for chunk in split_on_silence(signal):
            _, _, payload = http_request(
                "POST", self._url(),
                headers={
                    "Ocp-Apim-Subscription-Key": self.settings.speech_key,
                    "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                    "Accept": "application/json",
                },
                body=_pcm_wav(chunk), timeout=60,
            )
            status = payload.get("RecognitionStatus")
            if status not in ("Success", "NoMatch", "InitialSilenceTimeout"):
                raise AzureError(f"Azure Speech retornou status {status}")
            best = (payload.get("NBest") or [{}])[0]
            text = payload.get("DisplayText") or best.get("Display", "")
            if text:
                texts.append(text)
                confs.append(best.get("Confidence", 0.0))
                segments.append({
                    "inicio_s": round(offset + payload.get("Offset", 0) / 1e7, 2),
                    "duracao_s": round(payload.get("Duration", 0) / 1e7, 2),
                    "texto": text,
                    "confianca": best.get("Confidence"),
                })
            offset += len(chunk) / TARGET_SR
        return Transcript(
            text=" ".join(texts), provider="azure_speech_to_text",
            confidence=round(float(np.mean(confs)), 3) if confs else None, segments=segments,
        )


class LocalReferenceTranscriber:
    """Fallback: transcrição de referência gravada junto ao áudio (roteiro da consulta)."""

    available = True

    def transcribe(self, wav_path: str | Path) -> Transcript:
        ref = Path(wav_path).with_suffix(".txt")
        text = ref.read_text(encoding="utf-8").strip() if ref.exists() else ""
        return Transcript(text=text, provider="local_referencia", confidence=None)


def transcribe(wav_path: str | Path, prefer_azure: bool = True) -> Transcript:
    """Azure quando configurado; senão (ou em caso de erro) a referência local."""
    local = LocalReferenceTranscriber().transcribe(wav_path)
    azure = AzureSpeechToText()
    if not (prefer_azure and azure.available):
        return local
    try:
        result = azure.transcribe(wav_path)
    except AzureError as exc:
        local.error = str(exc)
        return local
    if local.text:
        result.wer_vs_reference = round(word_error_rate(local.text, result.text), 3)
    return result
