"""Extração de biomarcadores vocais (numpy/scipy puro, sem dependências pesadas).

Marcadores usados na literatura de análise de voz clínica:

* **Prosódia / ritmo** — taxa de fala e de articulação (núcleos silábicos
  por segundo), proporção de pausas, pausas longas por minuto;
* **Fonte glótica** — F0 por autocorrelação, variabilidade de F0 em
  semitons (monotonia), perturbação de F0 (*jitter proxy*), perturbação
  de amplitude (*shimmer proxy*), razão harmônico-ruído (HNR);
* **Articulação** — energia de alta frequência (3–4.8 kHz, em dB relativos à banda de voz) nos trechos de
  fala, que cai quando consoantes ficam imprecisas;
* **Energia** — inclinação da energia vocal ao longo da consulta
  (dB/min), negativa quando o paciente "vai perdendo a voz" de cansaço;
* **Tremor** — modulação de amplitude 8–12 Hz no envelope da fala;
* **Eventos não-verbais** — tosse e inspirações audíveis, detectados nos
  intervalos entre falas e rotulados com as classes do AudioSet.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

FRAME = 400  # 25 ms @16 kHz
HOP = 160    # 10 ms
F0_MIN, F0_MAX = 75.0, 400.0


@dataclass
class AcousticEvent:
    label: str          # nome da classe AudioSet (Cough, Breathing, ...)
    start_s: float
    end_s: float
    peak_db: float

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


@dataclass
class VoiceFeatures:
    duration_s: float
    phonation_time_s: float
    speech_rate_syl_s: float
    articulation_rate_syl_s: float
    pause_ratio: float
    mean_pause_s: float
    long_pauses_per_min: float
    f0_median_hz: float
    f0_std_semitones: float
    jitter_pct: float
    shimmer_db: float
    hnr_db: float
    hf_energy_db: float
    energy_slope_db_min: float
    tremor_index: float
    mean_phrase_s: float
    events: list[AcousticEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [{**asdict(e), "duration_s": e.duration_s} for e in self.events]
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    def count(self, label: str) -> int:
        return sum(1 for e in self.events if e.label == label)


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Intervalos [ini, fim) contíguos onde mask é True."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _periodicity(frames: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Pico da autocorrelação normalizada (0–1) e F0 correspondente, por frame."""
    lag_min, lag_max = int(sr / F0_MAX), int(sr / F0_MIN)
    win = frames * np.hanning(FRAME)[None, :]
    win = win - win.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(win, n=2 * FRAME, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :FRAME]
    ac = ac / (ac[:, :1] + 1e-12)
    seg = ac[:, lag_min:lag_max]
    best = seg.argmax(axis=1)
    r = seg[np.arange(len(seg)), best]
    f0 = sr / (best + lag_min)
    return np.clip(r, 0, 1), f0


def extract_features(signal: np.ndarray, sr: int = 16000) -> VoiceFeatures:
    frames = _frames(signal)
    hop_s = HOP / sr
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 10)
    active = db > max(floor + 12, -50)

    r, f0 = _periodicity(frames, sr)
    voiced = active & (r > 0.45)

    # --- segmentos de fala: frames vozeados + consoantes vizinhas (gaps < 200 ms)
    speech_runs = _merge_runs([(a, b) for a, b in _runs(voiced) if b - a >= 3], max_gap=int(0.2 / hop_s))
    speech = np.zeros(len(frames), dtype=bool)
    for a, b in speech_runs:
        speech[a:b] = True

    duration = len(signal) / sr
    phonation = speech.sum() * hop_s
    first = speech_runs[0][0] if speech_runs else 0
    last = speech_runs[-1][1] if speech_runs else len(frames)
    pauses = [(speech_runs[i + 1][0] - speech_runs[i][1]) * hop_s for i in range(len(speech_runs) - 1)]
    pauses = [p for p in pauses if p >= 0.25]
    talk_span = max((last - first) * hop_s, 1e-6)

    # --- núcleos silábicos: picos do envelope suavizado dentro da fala
    from scipy.signal import find_peaks

    env = np.convolve(db, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(env, prominence=3.0, distance=int(0.12 / hop_s))
    syllables = int(np.sum(speech[peaks] & voiced[peaks])) if len(peaks) else 0

    # --- fonte glótica
    vf0 = f0[voiced]
    f0_med = float(np.median(vf0)) if vf0.size else 0.0
    semis = 12 * np.log2(vf0 / f0_med) if vf0.size else np.array([0.0])
    semis = semis[np.abs(semis) < 12]  # descarta erros de oitava
    jit, shim = [], []
    for a, b in _runs(voiced):
        if b - a < 4:
            continue
        periods = 1.0 / f0[a:b]
        jit.append(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
        shim.append(np.mean(np.abs(np.diff(db[a:b]))))
    rv = np.clip(r[voiced], 1e-3, 0.999)
    hnr = float(np.mean(10 * np.log10(rv / (1 - rv)))) if rv.size else 0.0

    # --- articulação: energia de consoantes (3–4.8 kHz) relativa à banda de voz (0.1–3 kHz), em dB
    spec = np.abs(np.fft.rfft(frames[speech] * np.hanning(FRAME), axis=1)) ** 2 if speech.any() else np.zeros((1, FRAME // 2 + 1))
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    hf = spec[:, (freqs >= 3000) & (freqs <= 4800)].sum()
    lf = spec[:, (freqs >= 100) & (freqs < 3000)].sum()
    hf_ratio = float(10 * np.log10((hf + 1e-12) / (lf + 1e-12)))

    # --- inclinação de energia (dB/min) ao longo das frases
    seg_t = np.array([(a + b) / 2 * hop_s for a, b in speech_runs])
    seg_db = np.array([np.mean(db[a:b]) for a, b in speech_runs])
    slope = float(np.polyfit(seg_t, seg_db, 1)[0] * 60) if len(speech_runs) >= 3 else 0.0

    # --- tremor: modulação 8–12 Hz do envelope linear nos trechos de fala
    tremor = _tremor_index(rms, speech_runs, 1 / hop_s)

    events = detect_events(db, r, speech, floor, hop_s, frames, sr)

    return VoiceFeatures(
        duration_s=duration,
        phonation_time_s=float(phonation),
        speech_rate_syl_s=syllables / talk_span,
        articulation_rate_syl_s=syllables / max(phonation, 1e-6),
        pause_ratio=float(sum(pauses) / talk_span),
        mean_pause_s=float(np.mean(pauses)) if pauses else 0.0,
        long_pauses_per_min=float(sum(p > 1.0 for p in pauses) / (talk_span / 60)),
        f0_median_hz=f0_med,
        f0_std_semitones=float(np.std(semis)) if semis.size else 0.0,
        jitter_pct=float(np.mean(jit) * 100) if jit else 0.0,
        shimmer_db=float(np.mean(shim)) if shim else 0.0,
        hnr_db=hnr,
        hf_energy_db=hf_ratio,
        energy_slope_db_min=slope,
        tremor_index=tremor,
        mean_phrase_s=float(np.mean([(b - a) * hop_s for a, b in speech_runs])) if speech_runs else 0.0,
        events=events,
    )


def _tremor_index(rms: np.ndarray, runs: list[tuple[int, int]], fs: float) -> float:
    ratios = []
    for a, b in runs:
        seg = rms[a:b]
        if len(seg) < int(0.6 * fs):
            continue
        seg = (seg - seg.mean()) / (seg.std() + 1e-9)
        p = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / fs)
        band = p[(f >= 8) & (f <= 12)].sum()
        total = p[(f >= 1) & (f <= 20)].sum() + 1e-12
        ratios.append(band / total)
    return float(np.mean(ratios)) if ratios else 0.0


def detect_events(db, r, speech, floor, hop_s, frames, sr) -> list[AcousticEvent]:
    """Eventos não-verbais fora da fala: tosse (explosiva, alta) e respiração audível (longa, baixa)."""
    speech_level = np.median(db[speech]) if speech.any() else floor + 30
    candidate = (db > floor + 10) & ~speech & (r < 0.5)
    events = []
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    for a, b in _merge_runs(_runs(candidate), max_gap=3):
        dur = (b - a) * hop_s
        if dur < 0.08:
            continue
        seg_db = db[a:b]
        peak = float(seg_db.max())
        onset = float(seg_db[: min(4, len(seg_db))].max() - db[max(0, a - 3)])
        spec = np.abs(np.fft.rfft(frames[a:b] * np.hanning(FRAME), axis=1)) ** 2
        centroid = float((spec.sum(axis=0) * freqs).sum() / (spec.sum() + 1e-12))
        if peak >= speech_level - 3 and onset > 12 and dur <= 1.2 and centroid > 800:
            label = "Cough"
        elif 0.3 <= dur <= 2.0 and peak < speech_level - 4:
            label = "Sigh" if dur > 1.0 else "Breathing"
        else:
            continue
        events.append(AcousticEvent(label, round(a * hop_s, 2), round(b * hop_s, 2), round(peak, 1)))
    return events
