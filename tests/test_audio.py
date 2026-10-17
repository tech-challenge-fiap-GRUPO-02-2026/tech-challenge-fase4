import numpy as np

from src.audio.azure_speech import LocalReferenceTranscriber, split_on_silence, transcribe, word_error_rate
from src.audio.azure_text_analytics import analyze_local, analyze_text
from src.audio.clinical_terms import find_terms
from src.audio.features import extract_features
from src.audio.voice_analyzer import assess
from src.audio.wav_io import read_wav, write_wav
from src.config import AUDIO_DIR
from src.datasets.generate_audio import cough_sound, synth_vowels


def test_f0_estimation_on_synthetic_voice():
    rng = np.random.default_rng(0)
    x = np.concatenate([synth_vowels(1.5, 150, rng), np.zeros(8000, np.float32), synth_vowels(1.5, 150, rng)])
    f = extract_features(x)
    assert 130 < f.f0_median_hz < 175


def test_cough_detected_between_phrases():
    rng = np.random.default_rng(1)
    speech = synth_vowels(1.2, 140, rng)
    x = np.concatenate([speech, np.zeros(6000, np.float32), cough_sound(rng), np.zeros(6000, np.float32), speech])
    labels = [e.label for e in extract_features(x).events]
    assert "Cough" in labels


def test_wav_roundtrip(tmp_path):
    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    write_wav(tmp_path / "a.wav", x)
    y, sr = read_wav(tmp_path / "a.wav")
    assert sr == 16000 and len(y) == len(x) and np.allclose(x, y, atol=1e-3)


def test_voice_indicators_match_injected_profiles():
    expected = {"PAC-0001": set(), "PAC-0003": {"voz_disartria"}, "PAC-0004": {"voz_fadiga"},
                "PAC-0002": {"voz_respiratorio"}}
    for pid, flags in expected.items():
        a = assess(extract_features(*read_wav(AUDIO_DIR / f"{pid}_consulta.wav")))
        assert flags <= set(a.flags), (pid, a.flags)
        if not flags:
            assert a.flags == []


def test_wer():
    assert word_error_rate("falta de ar", "falta de ar") == 0
    assert abs(word_error_rate("dor no peito forte", "dor no peito") - 0.25) < 1e-9
    assert word_error_rate("Coração", "coracao") == 0  # acentos/maiúsculas normalizados


def test_split_on_silence_respects_limit():
    x = np.random.default_rng(0).normal(0, 0.1, 16000 * 130).astype(np.float32)
    chunks = split_on_silence(x, max_s=55)
    assert sum(len(c) for c in chunks) == len(x)
    assert all(len(c) <= 55 * 16000 for c in chunks)


def test_transcribe_falls_back_without_azure(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    t = transcribe(AUDIO_DIR / "PAC-0003_consulta.wav")
    assert t.provider == "local_referencia" and "enrolada" in t.text
    assert LocalReferenceTranscriber().transcribe(AUDIO_DIR / "inexistente.wav").text == ""


def test_critical_terms_and_negation():
    terms = {t.signal: t for t in find_terms("Não sinto dor no peito, mas estou com falta de ar e a boca torta.")}
    assert terms["possivel_sindrome_coronariana"].negated
    assert not terms["desconforto_respiratorio"].negated
    assert terms["suspeita_avc"].severity == "critico"


def test_text_analysis_local(monkeypatch):
    monkeypatch.delenv("AZURE_LANGUAGE_KEY", raising=False)
    r = analyze_text("Estou muito cansada e com tontura, tive febre.")
    assert r.provider == "local_lexico"
    assert r.sentiment["sentimento"] == "negative"
    assert r.risk >= 0.6
    assert analyze_local("Estou me sentindo bem melhor hoje.").sentiment["sentimento"] == "positive"
