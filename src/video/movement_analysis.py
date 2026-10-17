"""Análise postural de sessões de fisioterapia a partir dos keypoints.

Para cada exercício há um **protocolo esperado** (amplitude-alvo, simetria,
estabilidade de tronco). O vídeo é segmentado em repetições e cada
repetição é comparada ao protocolo; desvios viram achados:

* amplitude de movimento (ADM) abaixo do esperado;
* assimetria entre lados (ex.: hemiparesia);
* compensação de tronco (inclinação lateral para "ajudar" o braço);
* tremor (energia 4–8 Hz no punho, normalizada pela largura dos ombros);
* oscilação crescente do tronco (instabilidade → risco de queda);
* **queda**: tronco > 60° da vertical com descida brusca do quadril.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from src.models import Finding
from src.video.pose import PoseSequence

PROTOCOLS = {
    "abducao_ombro": {
        "nome": "Abdução de ombro bilateral",
        "adm_alvo_graus": 150, "adm_minima_pct": 0.6, "assimetria_max_graus": 20,
        "inclinacao_tronco_max_graus": 10, "tremor_max": 0.2,
    },
    "agachamento": {
        "nome": "Agachamento (sentar-levantar)",
        "flexao_joelho_alvo_graus": 80, "oscilacao_tronco_max_graus": 3,
    },
}


@dataclass
class ExerciseReport:
    exercise: str
    n_reps: int
    metrics: dict
    reps: list[dict]
    findings: list[Finding] = field(default_factory=list)
    key_frames: list[dict] = field(default_factory=list)  # frames notáveis para o relatório

    def to_dict(self) -> dict:
        return {
            "exercicio": self.exercise, "protocolo": PROTOCOLS.get(self.exercise, {}),
            "repeticoes": self.n_reps, "metricas": self.metrics, "por_repeticao": self.reps,
            "achados": [f.to_dict() for f in self.findings], "frames_chave": self.key_frames,
        }


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ângulo (graus) entre vetores a e b, por frame."""
    dot = (a * b).sum(-1)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))


def _smooth(x: np.ndarray, n: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():  # preenche lacunas (frames sem detecção) por interpolação
        idx = np.arange(len(x))
        ok = ~np.isnan(x)
        x = np.interp(idx, idx[ok], x[ok]) if ok.any() else np.zeros_like(x)
    return np.convolve(x, np.ones(n) / n, mode="same")


def trunk_angle(seq: PoseSequence) -> np.ndarray:
    """Inclinação do tronco em relação à vertical (graus, com sinal: + = para a esquerda da imagem)."""
    sh = (seq.xy("ombro_esq") + seq.xy("ombro_dir")) / 2
    hip = (seq.xy("quadril_esq") + seq.xy("quadril_dir")) / 2
    v = sh - hip
    return np.degrees(np.arctan2(v[:, 0], -v[:, 1]))


def shoulder_abduction(seq: PoseSequence, side: str) -> np.ndarray:
    """Abdução = ângulo entre o braço (ombro→cotovelo) e o eixo do tronco apontando para baixo."""
    sh, el = seq.xy(f"ombro_{side}"), seq.xy(f"cotovelo_{side}")
    mid_sh = (seq.xy("ombro_esq") + seq.xy("ombro_dir")) / 2
    mid_hip = (seq.xy("quadril_esq") + seq.xy("quadril_dir")) / 2
    return _angle(mid_hip - mid_sh, el - sh)


def knee_flexion(seq: PoseSequence, side: str = "esq") -> np.ndarray:
    hip, knee, ankle = seq.xy(f"quadril_{side}"), seq.xy(f"joelho_{side}"), seq.xy(f"tornozelo_{side}")
    return 180 - _angle(hip - knee, ankle - knee)


def tremor_index(seq: PoseSequence, side: str, band: tuple[float, float] = (4, 8)) -> float:
    """Energia na banda de tremor do punho, normalizada pela largura dos ombros."""
    w = seq.xy(f"punho_{side}")
    scale = np.nanmedian(np.linalg.norm(seq.xy("ombro_esq") - seq.xy("ombro_dir"), axis=1)) + 1e-9
    total = 0.0
    for axis in (0, 1):
        x = _smooth(w[:, axis], 1)
        x = x - _smooth(x, int(seq.fps * 0.3) | 1)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / seq.fps)
        total += spec[(f >= band[0]) & (f <= band[1])].sum() / len(x)
    return float(np.sqrt(total) / scale)


def _segment_reps(signal: np.ndarray, fps: float, prominence: float) -> list[tuple[int, int, int]]:
    """(início, pico, fim) de cada repetição."""
    peaks, _ = find_peaks(signal, prominence=prominence, distance=int(1.5 * fps))
    reps = []
    for i, p in enumerate(peaks):
        a = 0 if i == 0 else (peaks[i - 1] + p) // 2
        b = len(signal) - 1 if i == len(peaks) - 1 else (p + peaks[i + 1]) // 2
        reps.append((int(a), int(p), int(b)))
    return reps


def _t(frame: int, fps: float) -> str:
    return f"{frame / fps:05.1f}s"


def analyze_shoulder_abduction(seq: PoseSequence, patient_id: str | None = None) -> ExerciseReport:
    proto = PROTOCOLS["abducao_ombro"]
    abd_l, abd_r = _smooth(shoulder_abduction(seq, "esq")), _smooth(shoulder_abduction(seq, "dir"))
    trunk = _smooth(trunk_angle(seq))
    reps = _segment_reps(np.maximum(abd_l, abd_r), seq.fps, prominence=40)
    rep_rows, findings, key_frames = [], [], []
    for n, (a, p, b) in enumerate(reps, 1):
        rom_l, rom_r = float(abd_l[a:b].max()), float(abd_r[a:b].max())
        lean = float(np.abs(trunk[a:b]).max())
        rep_rows.append({"rep": n, "adm_esq": round(rom_l, 1), "adm_dir": round(rom_r, 1),
                         "assimetria": round(abs(rom_l - rom_r), 1), "inclinacao_tronco_max": round(lean, 1),
                         "pico_s": round(p / seq.fps, 2)})
    if not rep_rows:
        return ExerciseReport("abducao_ombro", 0, {}, [], [Finding(
            "video", "sem_repeticoes", "moderado", "Nenhuma repetição do exercício identificada no vídeo",
            patient_id, method="segmentação de repetições")])

    mean_l = float(np.mean([r["adm_esq"] for r in rep_rows]))
    mean_r = float(np.mean([r["adm_dir"] for r in rep_rows]))
    asym = float(np.mean([r["assimetria"] for r in rep_rows]))
    lean = float(np.max([r["inclinacao_tronco_max"] for r in rep_rows]))
    trem_l, trem_r = tremor_index(seq, "esq"), tremor_index(seq, "dir")
    worst = max(rep_rows, key=lambda r: r["assimetria"])
    worst_frame = int(worst["pico_s"] * seq.fps)

    target = proto["adm_alvo_graus"]
    for side, rom in (("esquerdo", mean_l), ("direito", mean_r)):
        if rom < proto["adm_minima_pct"] * target:
            findings.append(Finding(
                "video", f"adm_reduzida_ombro_{side}", "alto", patient_id=patient_id,
                message=f"ADM do ombro {side} {rom:.0f}° — {rom / target:.0%} do alvo do protocolo ({target}°)",
                time_ref=f"pior repetição em {worst['pico_s']:.1f}s", method="pose COCO-17 → ângulo ombro-cotovelo/tronco",
                evidence={"adm_media": round(rom, 1), "alvo": target},
            ))
    if asym > proto["assimetria_max_graus"]:
        findings.append(Finding(
            "video", "assimetria_membros_superiores", "alto" if asym > 2 * proto["assimetria_max_graus"] else "moderado",
            patient_id=patient_id,
            message=f"Assimetria média de {asym:.0f}° entre os braços (esq {mean_l:.0f}° × dir {mean_r:.0f}°)",
            time_ref=f"{worst['pico_s']:.1f}s", method="comparação bilateral por repetição",
            evidence={"assimetria_media": round(asym, 1), "limite": proto["assimetria_max_graus"]},
        ))
    if lean > proto["inclinacao_tronco_max_graus"]:
        findings.append(Finding(
            "video", "compensacao_tronco", "moderado", patient_id=patient_id,
            message=f"Compensação com inclinação lateral do tronco de até {lean:.0f}° durante a elevação",
            method="ângulo quadril→ombros vs. vertical",
            evidence={"inclinacao_max": round(lean, 1), "limite": proto["inclinacao_tronco_max_graus"]},
        ))
    for side, trem in (("esquerdo", trem_l), ("direito", trem_r)):
        if trem > proto["tremor_max"]:
            findings.append(Finding(
                "video", f"tremor_punho_{side}", "moderado", patient_id=patient_id,
                message=f"Tremor no punho {side} (índice {trem:.3f}, limite {proto['tremor_max']})",
                method="energia 4–8 Hz da trajetória do punho",
                evidence={"indice": round(trem, 4)},
            ))
    if findings:
        key_frames.append({"frame": worst_frame, "motivo": "repetição com maior assimetria"})
    metrics = {
        "adm_media_esq": round(mean_l, 1), "adm_media_dir": round(mean_r, 1), "assimetria_media": round(asym, 1),
        "inclinacao_tronco_max": round(lean, 1), "tremor_esq": round(trem_l, 4), "tremor_dir": round(trem_r, 4),
    }
    return ExerciseReport("abducao_ombro", len(rep_rows), metrics, rep_rows, findings, key_frames)


def detect_fall(seq: PoseSequence) -> dict | None:
    """Queda: tronco > 60° da vertical + descida do quadril > 30% da altura corporal em ≤ 1,5 s."""
    trunk = np.abs(_smooth(trunk_angle(seq)))
    hip_y = _smooth(((seq.xy("quadril_esq") + seq.xy("quadril_dir")) / 2)[:, 1])
    head_y = _smooth(seq.xy("nariz")[:, 1])
    ankle_y = _smooth(((seq.xy("tornozelo_esq") + seq.xy("tornozelo_dir")) / 2)[:, 1])
    height = float(np.nanmedian((ankle_y - head_y)[: int(seq.fps)]))
    win = int(1.5 * seq.fps)
    for i in range(win, len(trunk)):
        if trunk[i] > 60 and hip_y[i] - hip_y[i - win] > 0.3 * height:
            onset = i - win + int(np.argmax(np.diff(hip_y[i - win:i + 1]) > 1))
            return {"frame": int(i), "inicio_frame": int(onset), "tronco_graus": round(float(trunk[i]), 1),
                    "queda_quadril_px": round(float(hip_y[i] - hip_y[i - win]), 1)}
    return None


def analyze_squat(seq: PoseSequence, patient_id: str | None = None) -> ExerciseReport:
    proto = PROTOCOLS["agachamento"]
    fall = detect_fall(seq)
    end = fall["inicio_frame"] if fall else seq.n_frames
    knee = _smooth(knee_flexion(seq))[:end]
    trunk = trunk_angle(seq)[:end]
    reps = _segment_reps(knee, seq.fps, prominence=30)
    rep_rows, findings, key_frames = [], [], []
    for n, (a, p, b) in enumerate(reps, 1):
        seg = trunk[a:b]
        sway = float(np.std(seg - _smooth(seg, int(seq.fps * 0.8) | 1)))  # oscilação rápida do tronco
        rep_rows.append({"rep": n, "flexao_joelho_max": round(float(knee[a:b].max()), 1),
                         "oscilacao_tronco": round(sway, 2), "pico_s": round(p / seq.fps, 2)})
    sways = [r["oscilacao_tronco"] for r in rep_rows]
    if sways and max(sways) > proto["oscilacao_tronco_max_graus"]:
        worst = max(rep_rows, key=lambda r: r["oscilacao_tronco"])
        trend = sways[-1] - sways[0] if len(sways) > 1 else 0.0
        findings.append(Finding(
            "video", "instabilidade_postural", "alto", patient_id=patient_id,
            message=f"Oscilação do tronco crescente (até {max(sways):.1f}°, limite {proto['oscilacao_tronco_max_graus']}°) — instabilidade/risco de queda",
            time_ref=f"rep {worst['rep']} em {worst['pico_s']:.1f}s", method="desvio-padrão da oscilação rápida do tronco",
            evidence={"oscilacao_por_rep": sways, "tendencia": round(trend, 2)},
        ))
        key_frames.append({"frame": int(worst["pico_s"] * seq.fps), "motivo": "maior oscilação do tronco"})
    if fall:
        findings.append(Finding(
            "video", "queda_detectada", "critico", patient_id=patient_id,
            message=f"QUEDA durante o exercício: tronco a {fall['tronco_graus']:.0f}° da vertical, quadril desceu {fall['queda_quadril_px']:.0f}px",
            time_ref=_t(fall["frame"], seq.fps), method="pose: ângulo do tronco + velocidade vertical do quadril",
            evidence=fall,
        ))
        key_frames.append({"frame": fall["frame"], "motivo": "queda"})
    metrics = {
        "flexao_joelho_media": round(float(np.mean([r["flexao_joelho_max"] for r in rep_rows])), 1) if rep_rows else 0.0,
        "oscilacao_tronco_max": round(max(sways), 2) if sways else 0.0,
        "queda": bool(fall),
    }
    return ExerciseReport("agachamento", len(rep_rows), metrics, rep_rows, findings, key_frames)


ANALYZERS = {"abducao_ombro": analyze_shoulder_abduction, "agachamento": analyze_squat}
