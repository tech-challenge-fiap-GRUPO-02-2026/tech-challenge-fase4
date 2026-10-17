"""Classificador de eventos sonoros treinado no AudioSet (opcional).

Usa o **Audio Spectrogram Transformer** ``MIT/ast-finetuned-audioset-10-10-0.4593``
(Gong et al., 2021 — mAP 0.459 no AudioSet), que prediz as 527 classes da
ontologia. Janelas deslizantes de 4 s percorrem a consulta e só as classes
do catálogo clínico (tosse, respiração, chiado, arquejo...) são mantidas.

É uma segunda opinião independente do detector heurístico de
``features.detect_events`` — os dois usam os mesmos rótulos AudioSet, então
o relatório pode mostrar concordância entre eles. Requer
``requirements-ml.txt`` (torch + transformers) e baixa ~350 MB na 1ª vez;
sem isso, ``available()`` retorna False e o pipeline segue sem ele.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.datasets.audioset import load_catalog

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
WINDOW_S = 4.0
HOP_S = 2.0
MIN_SCORE = 0.05  # janelas de 4 s misturam fala e evento; a fala domina a probabilidade


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline

    return pipeline("audio-classification", model=MODEL_ID, top_k=10)


def classify(signal: np.ndarray, sr: int = 16000, include_speech: bool = False) -> list[dict]:
    clinical = {c["name"]: c for c in load_catalog()["classes_clinicas"]}
    if not include_speech:
        clinical = {k: v for k, v in clinical.items() if v["indicador"] != "fala"}
    clf = _pipeline()
    win, hop = int(WINDOW_S * sr), int(HOP_S * sr)
    detections = []
    for start in range(0, max(1, len(signal) - win + hop), hop):
        chunk = signal[start:start + win]
        if len(chunk) < sr:
            break
        for pred in clf({"raw": chunk.astype(np.float32), "sampling_rate": sr}):
            if pred["label"] in clinical and pred["score"] >= MIN_SCORE:
                cls = clinical[pred["label"]]
                detections.append({
                    "label": pred["label"], "nome_pt": cls["nome_pt"], "mid": cls["mid"],
                    "inicio_s": round(start / sr, 2), "fim_s": round((start + len(chunk)) / sr, 2),
                    "score": round(float(pred["score"]), 3),
                })
    return detections


def summarize(detections: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for d in detections:
        s = summary.setdefault(d["label"], {"nome_pt": d["nome_pt"], "mid": d["mid"], "janelas": 0, "score_max": 0.0})
        s["janelas"] += 1
        s["score_max"] = max(s["score_max"], d["score"])
    return summary
