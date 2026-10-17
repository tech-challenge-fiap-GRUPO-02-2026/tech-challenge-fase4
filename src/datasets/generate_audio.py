"""Gera as gravações de consulta (áudio) de cada paciente.

Não existe dataset público de consultas médicas em português com
disartria/fadiga rotuladas, então as gravações são produzidas assim:

1. **Fala real sintetizada** em pt-BR pela voz neural/SAPI do Windows
   ("Microsoft Maria") a partir de um roteiro de consulta por paciente —
   o roteiro vira também a *transcrição de referência* (``.txt``), usada
   para medir a qualidade do Azure Speech to Text (WER);
2. **Perfil vocal** aplicado por processamento de sinal, reproduzindo os
   marcadores acústicos descritos na literatura:
   - *fadiga*: fala lenta, energia decrescente, pausas longas, suspiros;
   - *disartria*: fala muito lenta, articulação imprecisa (passa-baixa),
     tremor de amplitude ~5 Hz, instabilidade de F0 (jitter), pausas
     irregulares no meio das frases;
   - *respiratório*: frases curtas interrompidas por inspirações audíveis
     e episódios de tosse (classes ``Breathing`` e ``Cough`` do AudioSet).

Fora do Windows (sem SAPI), cai para um sintetizador de vogais por
formantes — suficiente para os testes, mas sem palavras reais.

O ``.json`` ao lado de cada ``.wav`` guarda o perfil injetado (ground truth).
"""
from __future__ import annotations

import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfilt

from src.audio.wav_io import TARGET_SR, read_wav, write_wav
from src.config import AUDIO_DIR

SEED = 7

SCRIPTS = {
    "PAC-0001": ("normal", [
        "Bom dia, doutor.",
        "Estou me sentindo bem melhor hoje.",
        "A dor da cirurgia diminuiu bastante e consegui caminhar pelo corredor.",
        "Dormi bem a noite toda e estou com fome.",
        "Acho que já posso ir para casa em breve.",
    ]),
    "PAC-0002": ("respiratorio", [
        "Doutora, estou com muita falta de ar.",
        "Não consigo falar uma frase inteira sem parar.",
        "A tosse piorou de noite, com catarro amarelado.",
        "Sinto um chiado no peito quando respiro.",
        "Estou cansada só de sentar na cama.",
    ]),
    "PAC-0003": ("disartria", [
        "Doutor, minha fala está enrolada desde ontem.",
        "Sinto o braço esquerdo fraco e formigamento no rosto.",
        "Às vezes engasgo quando tomo água.",
        "Minha família disse que minha boca está torta.",
    ]),
    "PAC-0004": ("fadiga", [
        "Estou muito cansada, sem força nenhuma.",
        "Tive febre e calafrios a noite inteira.",
        "Sinto tontura quando levanto e meu coração está acelerado.",
        "Estou com medo de piorar.",
    ]),
    "PAC-0005": ("fadiga_respiratorio", [
        "Estou fraca e com sono o tempo todo.",
        "Ontem à noite fiquei confusa e levantei sozinha para ir ao banheiro.",
        "Acho que caí, estou com dor no quadril.",
        "Ainda sinto falta de ar e tosse.",
    ]),
    "PAC-0006": ("normal", [
        "Doutor, a cirurgia correu bem, mas sinto uma dor no peito quando respiro fundo.",
        "Estou um pouco ansioso.",
        "A cicatriz está doendo, mas consigo me mexer na cama.",
        "Quero saber quando vou sair da UTI.",
    ]),
}

# rate/pitch no formato SSML <prosody>. Atenção: no System.Speech, "0%"
# é interpretado como a velocidade mínima — por isso o perfil normal usa "medium".
PROSODY = {
    "normal": {"rate": "medium", "pitch": "medium"},
    "respiratorio": {"rate": "medium", "pitch": "medium"},
    "fadiga": {"rate": "slow", "pitch": "low"},
    "fadiga_respiratorio": {"rate": "-30%", "pitch": "low"},
    "disartria": {"rate": "-45%", "pitch": "-8%"},
}

_PS_TEMPLATE = r"""
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 '{manifest}' | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($it in $items) {{
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {{ $s.SelectVoice('Microsoft Maria Desktop') }} catch {{}}
  $s.SetOutputToWaveFile($it.out, $fmt)
  $s.SpeakSsml($it.ssml)
  $s.Dispose()
}}
"""


# ---------------------------------------------------------------------------
# Síntese de fala
# ---------------------------------------------------------------------------
def _sapi_available() -> bool:
    return platform.system() == "Windows"


def _synthesize_sapi(sentences: list[str], prosody: dict, workdir: Path) -> list[np.ndarray]:
    items = []
    for i, text in enumerate(sentences):
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">'
            f'<prosody rate="{prosody["rate"]}" pitch="{prosody["pitch"]}">{text}</prosody></speak>'
        )
        items.append({"out": str(workdir / f"seg_{i:02d}.wav"), "ssml": ssml})
    manifest = workdir / "manifest.json"
    manifest.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    script = workdir / "tts.ps1"
    script.write_text(_PS_TEMPLATE.format(manifest=manifest), encoding="utf-8-sig")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)
    return [_trim_silence(read_wav(it["out"])[0]) for it in items]


def synth_vowels(duration_s: float, f0: float, rng: np.random.Generator, sr: int = TARGET_SR,
                 jitter: float = 0.005) -> np.ndarray:
    """Sintetizador simples por formantes (pulso glotal + 3 ressonâncias) — fallback/testes."""
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    pos = 0.0
    # contorno de entonação + jitter ciclo a ciclo
    while pos < n:
        idx = int(pos)
        out[idx] = 1.0
        f = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * idx / sr)) * (1 + rng.normal(0, jitter))
        pos += sr / f
    for fc, bw in [(700, 110), (1220, 120), (2600, 160)]:
        sos = butter(2, [max(50, fc - bw), fc + bw], btype="bandpass", fs=sr, output="sos")
        out = out + 0.6 * sosfilt(sos, out)
    # "sílabas": envelope de ~4 Hz
    t = np.arange(n) / sr
    env = 0.55 + 0.45 * np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None)
    out = out * env
    return (0.3 * out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def _synthesize_fallback(sentences: list[str], profile: str, rng: np.random.Generator) -> list[np.ndarray]:
    f0 = 120 if profile != "fadiga" else 105
    return [synth_vowels(0.06 * len(s), f0, rng, jitter=0.03 if profile == "disartria" else 0.005) for s in sentences]


def _trim_silence(x: np.ndarray, thr: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(x) > thr)[0]
    return x[idx[0]: idx[-1] + 1] if idx.size else x


# ---------------------------------------------------------------------------
# Eventos não-verbais (classes AudioSet)
# ---------------------------------------------------------------------------
def breath_sound(rng: np.random.Generator, duration_s: float = 0.7, level: float = 0.06) -> np.ndarray:
    """Inspiração audível: ruído filtrado 300–2000 Hz com envelope em sino (AudioSet: Breathing)."""
    n = int(duration_s * TARGET_SR)
    noise = rng.normal(0, 1, n)
    sos = butter(3, [300, 2000], btype="bandpass", fs=TARGET_SR, output="sos")
    env = np.hanning(n)
    out = sosfilt(sos, noise) * env
    return (level * out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def cough_sound(rng: np.random.Generator, bursts: int = 2) -> np.ndarray:
    """Episódio de tosse: rajadas explosivas de banda larga com decaimento rápido (AudioSet: Cough)."""
    parts = []
    for _ in range(bursts):
        n = int(0.22 * TARGET_SR)
        t = np.arange(n) / TARGET_SR
        burst = rng.normal(0, 1, n) * np.exp(-t / 0.05)
        sos = butter(2, [150, 5000], btype="bandpass", fs=TARGET_SR, output="sos")
        burst = sosfilt(sos, burst)
        parts.append(0.7 * burst / (np.abs(burst).max() + 1e-9))
        parts.append(np.zeros(int(0.12 * TARGET_SR)))
    return np.concatenate(parts).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * TARGET_SR), dtype=np.float32)


def _peak_rms(x: np.ndarray, frame: int = 400) -> float:
    n = max(1, len(x) // frame)
    return float(np.sqrt((x[: n * frame].reshape(n, -1) ** 2).mean(axis=1)).max()) + 1e-9


class SoundBank:
    """Sons não-verbais: clipes reais do ESC-50 quando baixados, senão síntese."""

    def __init__(self, rng: np.random.Generator, speech_level: float, use_real: bool = True):
        from src.datasets.esc50 import load_clips

        self.rng = rng
        self.speech_level = speech_level
        self.real = {"Cough": load_clips("Cough"), "Breathing": load_clips("Breathing")} if use_real else {}
        self.source = "ESC-50 (gravações reais)" if all(self.real.get(k) for k in ("Cough", "Breathing")) else "sintético"

    def _pick(self, label: str, level: float) -> np.ndarray | None:
        clips = self.real.get(label) or []
        if not clips:
            return None
        clip = clips[self.rng.integers(len(clips))]
        return (clip * (level / _peak_rms(clip))).astype(np.float32)

    def cough(self, bursts: int) -> np.ndarray:
        real = self._pick("Cough", 1.4 * self.speech_level)
        return real if real is not None else cough_sound(self.rng, bursts)

    def breath(self, duration_s: float, level_rel: float = 0.25) -> np.ndarray:
        real = self._pick("Breathing", level_rel * self.speech_level)
        if real is not None:
            return real[: int(duration_s * 1.4 * TARGET_SR)] * np.hanning(min(len(real), int(duration_s * 1.4 * TARGET_SR))).astype(np.float32)
        return breath_sound(self.rng, duration_s)


# ---------------------------------------------------------------------------
# Perfis vocais
# ---------------------------------------------------------------------------
def _lowpass(x: np.ndarray, fc: float) -> np.ndarray:
    return sosfilt(butter(4, fc, btype="lowpass", fs=TARGET_SR, output="sos"), x).astype(np.float32)


def _tremor(x: np.ndarray, hz: float = 5.0, depth: float = 0.35) -> np.ndarray:
    t = np.arange(len(x)) / TARGET_SR
    return (x * (1 - depth + depth * np.sin(2 * np.pi * hz * t) ** 2)).astype(np.float32)


def _pitch_instability(x: np.ndarray, rng: np.random.Generator, chunk_ms: int = 40, spread: float = 0.04) -> np.ndarray:
    """Reamostra blocos curtos com fatores aleatórios → F0 instável (jitter macro)."""
    chunk = int(chunk_ms / 1000 * TARGET_SR)
    out = []
    for i in range(0, len(x), chunk):
        seg = x[i:i + chunk]
        if len(seg) < 8:
            continue
        factor = 1 + rng.normal(0, spread)
        out.append(resample(seg, max(8, int(len(seg) * factor))))
    return np.concatenate(out).astype(np.float32)


def _split_words(seg: np.ndarray, rng: np.random.Generator, parts: int) -> list[np.ndarray]:
    cuts = sorted(rng.choice(np.arange(1, 10), size=parts - 1, replace=False) / 10 * len(seg))
    bounds = [0, *[int(c) for c in cuts], len(seg)]
    return [seg[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def compose(profile: str, segments: list[np.ndarray], rng: np.random.Generator,
            bank: SoundBank | None = None) -> tuple[np.ndarray, list[dict]]:
    """Monta a gravação final aplicando o perfil vocal; devolve sinal + eventos injetados."""
    if bank is None:
        bank = SoundBank(rng, float(np.median([_peak_rms(s) for s in segments])), use_real=False)
    pieces: list[np.ndarray] = [_silence(0.4)]
    events: list[dict] = []

    def cursor() -> float:
        return sum(len(p) for p in pieces) / TARGET_SR

    for i, seg in enumerate(segments):
        if profile == "disartria":
            seg = _pitch_instability(_tremor(_lowpass(seg, 2400)), rng)
            for chunk in _split_words(seg, rng, parts=3):
                pieces.append(chunk)
                pieces.append(_silence(rng.uniform(0.35, 0.7)))
            pieces.append(_silence(rng.uniform(0.6, 1.0)))
            continue

        if profile in ("respiratorio", "fadiga_respiratorio"):
            # frases interrompidas por inspirações audíveis
            for chunk in _split_words(seg, rng, parts=2):
                pieces.append(chunk)
                pieces.append(_silence(0.15))
                events.append({"tipo": "Breathing", "inicio_s": round(cursor(), 2)})
                pieces.append(bank.breath(rng.uniform(0.55, 0.85)))
                pieces.append(_silence(0.15))
            if i in (1, 3):
                pieces.append(_silence(0.3))
                events.append({"tipo": "Cough", "inicio_s": round(cursor(), 2)})
                pieces.append(bank.cough(bursts=2 if i == 1 else 3))
                pieces.append(_silence(0.3))
        else:
            pieces.append(seg)

        if profile.startswith("fadiga"):
            pieces.append(_silence(rng.uniform(1.0, 1.6)))
            if i == 1:
                events.append({"tipo": "Breathing", "inicio_s": round(cursor(), 2)})
                pieces.append(bank.breath(1.0, level_rel=0.2))
        else:
            pieces.append(_silence(rng.uniform(0.35, 0.6)))

    signal = np.concatenate(pieces)
    if profile.startswith("fadiga"):
        # energia vocal decrescente ao longo da consulta
        signal = (signal * np.linspace(1.0, 0.4, len(signal))).astype(np.float32)
        signal = _lowpass(signal, 5000)
    signal = signal + rng.normal(0, 0.0015, len(signal)).astype(np.float32)  # ruído de sala
    return signal, events


def generate_all(output_dir: Path = AUDIO_DIR, use_tts: bool | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    use_tts = _sapi_available() if use_tts is None else use_tts
    written = []
    for pid, (profile, sentences) in SCRIPTS.items():
        if use_tts:
            with tempfile.TemporaryDirectory() as tmp:
                segments = _synthesize_sapi(sentences, PROSODY[profile], Path(tmp))
        else:
            segments = _synthesize_fallback(sentences, profile, rng)
        bank = SoundBank(rng, float(np.median([_peak_rms(s) for s in segments])))
        signal, events = compose(profile, segments, rng, bank)
        wav_path = output_dir / f"{pid}_consulta.wav"
        write_wav(wav_path, signal)
        wav_path.with_suffix(".txt").write_text(" ".join(sentences), encoding="utf-8")
        wav_path.with_suffix(".json").write_text(json.dumps({
            "patient_id": pid,
            "perfil_injetado": profile,
            "sintese": "Windows SAPI pt-BR (Microsoft Maria)" if use_tts else "sintetizador de formantes (fallback)",
            "prosodia": PROSODY[profile],
            "sons_nao_verbais": bank.source,
            "eventos_injetados": events,
            "duracao_s": round(len(signal) / TARGET_SR, 2),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(wav_path)
    return written


def main() -> None:
    paths = generate_all()
    print(f"{len(paths)} gravações de consulta geradas em {AUDIO_DIR}")


if __name__ == "__main__":
    main()
