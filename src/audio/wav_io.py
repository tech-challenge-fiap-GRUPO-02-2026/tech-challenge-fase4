"""Leitura/escrita de WAV PCM 16-bit mono só com stdlib + numpy."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16000  # taxa exigida pelo Azure Speech REST (PCM 16 kHz mono)


def read_wav(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"Apenas PCM 16-bit é suportado (arquivo tem {8 * width} bits)")
    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    if target_sr and sr != target_sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, target_sr)
        signal = resample_poly(signal, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return signal, sr


def write_wav(path: str | Path, signal: np.ndarray, sr: int = TARGET_SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def wav_bytes(path: str | Path) -> bytes:
    """Bytes de um WAV já normalizado para 16 kHz mono (formato aceito pelo Azure)."""
    import io

    signal, sr = read_wav(path)
    buf = io.BytesIO()
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
