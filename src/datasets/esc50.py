"""ESC-50 — sons reais de tosse e respiração (complemento ao AudioSet).

O AudioSet só distribui rótulos e IDs do YouTube, não o áudio. Para ter
**tosses e respirações reais** nas consultas usamos um subconjunto do
ESC-50 (Piczak, 2015 — clipes do Freesound.org rotulados manualmente;
licença CC BY-NC 3.0, uso acadêmico com atribuição), cujas classes
``coughing`` e ``breathing`` correspondem às classes ``Cough`` e
``Breathing`` da ontologia AudioSet.

Os clipes abaixo foram escolhidos ranqueando os 80 clipes dessas duas
classes pelo próprio classificador AudioSet (AST) — os que o modelo
reconhece com mais confiança como sons respiratórios.

Referência: K. J. Piczak. "ESC: Dataset for Environmental Sound
Classification", ACM Multimedia 2015. https://github.com/karolpiczak/ESC-50
"""
from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from src.audio.wav_io import TARGET_SR, read_wav, write_wav
from src.config import RAW_DIR

ESC50_DIR = RAW_DIR / "esc50"
BASE_URL = "https://github.com/karolpiczak/ESC-50/raw/master/audio/{name}"

CLIPS = {
    "Cough": ["4-155650-A-24.wav", "1-53663-A-24.wav", "1-19118-A-24.wav", "5-184871-A-24.wav"],
    "Breathing": ["5-260164-A-23.wav", "4-210000-B-23.wav", "3-112557-B-23.wav", "3-151206-A-23.wav"],
}


def _active_region(x: np.ndarray, max_s: float, rel_db: float = -30.0) -> np.ndarray:
    frame = 320
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    active = np.where(db > db.max() + rel_db)[0]
    if active.size == 0:
        return x[: int(max_s * TARGET_SR)]
    a, b = active[0] * frame, (active[-1] + 1) * frame
    return x[a: min(b, a + int(max_s * TARGET_SR))]


def download(overwrite: bool = False) -> list[Path]:
    """Baixa os clipes, converte para 16 kHz mono e recorta a região ativa."""
    ESC50_DIR.mkdir(parents=True, exist_ok=True)
    written, meta = [], []
    for label, names in CLIPS.items():
        for name in names:
            out = ESC50_DIR / f"{label.lower()}_{name}"
            if out.exists() and not overwrite:
                written.append(out)
                continue
            with tempfile.TemporaryDirectory() as tmp:
                raw = Path(tmp) / name
                urllib.request.urlretrieve(BASE_URL.format(name=name), raw)
                x, _ = read_wav(raw)
            x = _active_region(x, max_s=1.6 if label == "Cough" else 1.2)
            write_wav(out, x)
            written.append(out)
            meta.append({"arquivo": out.name, "original": name, "classe_audioset": label})
    if meta:
        (ESC50_DIR / "README.json").write_text(json.dumps({
            "fonte": "ESC-50 (Piczak 2015) — clipes Freesound.org",
            "licenca": "CC BY-NC 3.0 — uso acadêmico, não comercial, com atribuição",
            "url": "https://github.com/karolpiczak/ESC-50",
            "clipes": meta,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return written


def load_clips(label: str) -> list[np.ndarray]:
    return [read_wav(p)[0] for p in sorted(ESC50_DIR.glob(f"{label.lower()}_*.wav"))]
