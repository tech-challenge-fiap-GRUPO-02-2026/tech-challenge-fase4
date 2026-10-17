"""Gera os vídeos clínicos de demonstração (fisioterapia e cirurgia).

Gravações reais de pacientes não podem ser versionadas (LGPD/sigilo), e
não há dataset público de fisioterapia hospitalar com anomalias rotuladas.
Por isso cada vídeo é gerado a partir de um **modelo cinemático** do
exercício, com a anomalia injetada de forma controlada:

* ``abducao_ombro`` — elevação lateral dos braços (5 repetições).
  PAC-0001: execução normal. PAC-0003 (pós-AVC): hemiparesia esquerda
  (braço esquerdo só chega a ~75°), compensação com inclinação do tronco
  e tremor no punho;
* ``agachamento`` (vista lateral) — PAC-0005: oscilação crescente do
  tronco, perda de equilíbrio e **queda para trás** na 4ª repetição;
* ``cirurgia`` (vista superior do campo cirúrgico) — PAC-0006: contagem
  de instrumental que não fecha (tesoura retida na área da incisão),
  instrumento que cai fora do campo estéril e invasão da área estéril
  por profissional não paramentado.

Os keypoints seguem o formato **COCO-17** (o mesmo do YOLOv8-pose e do
OpenPose COCO, ver ``src/video/pose.py``) e são salvos em
``<video>_keypoints.json``; as detecções de objetos no formato YOLO
(classe, bbox, confiança) em ``<video>_detections.json``. O ``.mp4``
renderizado (H.264) serve para visualização no painel e para o vídeo de
demonstração — num vídeo real, esses mesmos arquivos são produzidos pelo
YOLOv8-pose/YOLOv8 (``python -m src.main video --file seu_video.mp4``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import VIDEO_DIR

FPS = 30
W, H = 640, 480
SEED = 11

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


# ---------------------------------------------------------------------------
# Cinemática
# ---------------------------------------------------------------------------
def _rot(p: np.ndarray, center: np.ndarray, ang_deg: float) -> np.ndarray:
    a = np.radians(ang_deg)
    c, s = np.cos(a), np.sin(a)
    d = p - center
    return center + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def frontal_pose(theta_l: float, theta_r: float, lean: float = 0.0, tremor: tuple[float, float] = (0, 0)) -> np.ndarray:
    """Pose frontal (COCO-17, x,y em pixels). theta = abdução do ombro (0 = braço junto ao corpo)."""
    k = np.zeros((17, 2))
    hip_c = np.array([320.0, 290.0])
    k[11], k[12] = [345, 290], [295, 290]          # quadril E/D (esquerda do paciente = direita da imagem)
    k[13], k[14] = [348, 375], [292, 375]
    k[15], k[16] = [350, 455], [290, 455]
    k[5], k[6] = [370, 150], [270, 150]           # ombros
    k[0] = [320, 100]
    k[1], k[2] = [330, 92], [310, 92]
    k[3], k[4] = [342, 98], [298, 98]
    for side, sh, el, wr, th, sign in [("l", 5, 7, 9, theta_l, 1), ("r", 6, 8, 10, theta_r, -1)]:
        a = np.radians(th)
        d = np.array([sign * np.sin(a), np.cos(a)])
        k[el] = k[sh] + 70 * d
        k[wr] = k[el] + 65 * d
    k[9] += tremor
    upper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in upper:  # inclinação lateral do tronco em torno do quadril
        k[i] = _rot(k[i], hip_c, lean)
    return k


def sagittal_pose(knee_flex: float, sway: float = 0.0, fall: float = 0.0) -> np.ndarray:
    """Pose lateral para agachamento. fall ∈ [0,1]: rotação do corpo para trás até o chão."""
    ankle = np.array([320.0, 450.0])
    a = np.radians(knee_flex * 0.45)              # tíbia inclina para frente
    b = np.radians(knee_flex * 0.55)              # coxa inclina para trás
    knee = ankle + 90 * np.array([np.sin(a), -np.cos(a)])
    hip = knee + 90 * np.array([-np.sin(b), -np.cos(b)])
    tau = np.radians(knee_flex * 0.35 + sway)     # tronco inclina para frente (+ oscilação)
    shoulder = hip + 120 * np.array([np.sin(tau), -np.cos(tau)])
    head = shoulder + 45 * np.array([np.sin(tau), -np.cos(tau)])
    elbow = shoulder + np.array([60, 5])
    wrist = elbow + np.array([55, 0])
    k = np.zeros((17, 2))
    off = np.array([4.0, 0])                       # lados quase sobrepostos na vista lateral
    k[0] = head + [10, 0]
    k[1] = k[2] = head + [8, -6]
    k[3] = k[4] = head + [-4, -2]
    k[5], k[6] = shoulder + off, shoulder - off
    k[7], k[8] = elbow + off, elbow - off
    k[9], k[10] = wrist + off, wrist - off
    k[11], k[12] = hip + off, hip - off
    k[13], k[14] = knee + off, knee - off
    k[15], k[16] = ankle + off, ankle - off
    if fall > 0:
        # rotação para trás em torno do tornozelo; corpo termina deitado no chão
        k = np.array([_rot(p, ankle, -88 * fall) for p in k])
        k[:, 1] = np.minimum(k[:, 1], 462)
    return k


def _noisy(k: np.ndarray, rng: np.random.Generator, sigma: float = 1.2) -> list[list[float]]:
    pts = k + rng.normal(0, sigma, k.shape)
    conf = rng.uniform(0.82, 0.98, (17, 1))
    return np.hstack([pts, conf]).round(2).tolist()


def _rep_curve(t: np.ndarray, period: float, start: float = 1.0) -> np.ndarray:
    """0→1→0 suave por repetição (cosseno)."""
    phase = np.clip(t - start, 0, None) % period / period
    active = t >= start
    return np.where(active, 0.5 - 0.5 * np.cos(2 * np.pi * phase), 0.0)


def shoulder_session(profile: str, rng: np.random.Generator, reps: int = 5) -> list[list[list[float]]]:
    period = 4.0 if profile == "normal" else 5.0
    t = np.arange(int((1 + reps * period + 1) * FPS)) / FPS
    c = _rep_curve(t, period)
    c[t > 1 + reps * period] = 0
    frames = []
    for i, ti in enumerate(t):
        if profile == "normal":
            k = frontal_pose(165 * c[i], 162 * c[i], lean=1.5 * np.sin(2 * np.pi * 0.3 * ti))
        else:  # hemiparesia esquerda + compensação + tremor
            tremor = (4 * np.sin(2 * np.pi * 6 * ti) * c[i], 3 * np.cos(2 * np.pi * 6 * ti) * c[i])
            k = frontal_pose(75 * c[i], 158 * c[i], lean=-14 * c[i], tremor=tremor)
        frames.append(_noisy(k, rng))
    return frames


def squat_session(rng: np.random.Generator, reps: int = 4, fall_rep: int = 4) -> list[list[list[float]]]:
    period = 4.0
    fall_start = 1 + (fall_rep - 1) * period + period * 0.45
    total = fall_start + 4.0
    t = np.arange(int(total * FPS)) / FPS
    c = _rep_curve(t, period)
    frames = []
    for i, ti in enumerate(t):
        rep = int((ti - 1) // period) + 1 if ti >= 1 else 0
        sway_amp = 0 if rep < 3 else (5 if rep == 3 else 12)
        sway = sway_amp * np.sin(2 * np.pi * 1.5 * ti)
        if ti < fall_start:
            k = sagittal_pose(100 * c[i], sway=sway)
        else:
            f = min(1.0, (ti - fall_start) / 0.8)   # queda em 0,8 s
            k = sagittal_pose(100 * c[int(fall_start * FPS)], sway=-15, fall=f ** 2)
        frames.append(_noisy(k, rng))
    return frames


# ---------------------------------------------------------------------------
# Cirurgia — detecções no formato YOLO
# ---------------------------------------------------------------------------
TABLE = (430, 40, 620, 440)       # mesa de instrumental (área estéril)
FIELD = (40, 40, 420, 440)        # campo cirúrgico (área estéril)
INCISION = (170, 180, 300, 300)   # área da incisão (área crítica)
INSTRUMENTS = ["bisturi", "tesoura", "pinca", "pinca", "porta_agulha", "afastador"]


def _box(cx, cy, w, h):
    return [round(cx - w / 2, 1), round(cy - h / 2, 1), round(cx + w / 2, 1), round(cy + h / 2, 1)]


def surgery_session(rng: np.random.Generator, duration_s: float = 24.0) -> list[list[dict]]:
    n = int(duration_s * FPS)
    slots = [(470 + 90 * (j % 2), 90 + 60 * (j // 2)) for j in range(len(INSTRUMENTS))]
    frames = []
    for i in range(n):
        t = i / FPS
        dets = []
        for j, name in enumerate(INSTRUMENTS):
            home = np.array(slots[j], dtype=float)
            pos = home.copy()
            # uso: cada instrumento vai à incisão e volta numa janela própria
            t0 = 3 + 2.2 * j
            if name == "tesoura" and t >= t0:
                # tesoura vai para a incisão e NÃO volta (item retido)
                pos = np.array([235.0, 240.0]) if t > t0 + 1 else home + (np.array([235, 240]) - home) * (t - t0)
            elif name == "pinca" and j == 3 and 12.5 <= t:
                # pinça cai fora do campo estéril (chão) aos 13 s
                pos = np.array([300.0, 465.0]) if t > 13.3 else home + (np.array([300, 465]) - home) * (t - 12.5) / 0.8
            elif t0 <= t < t0 + 2.0:
                pos = np.array([235.0, 240.0]) + rng.normal(0, 2, 2)
            dets.append({"classe": name, "bbox": _box(*pos, 46, 18), "conf": round(float(rng.uniform(0.7, 0.95)), 2)})
        # mãos da equipe paramentada sempre no campo
        dets.append({"classe": "maos_equipe", "bbox": _box(235 + 30 * np.sin(t), 250, 120, 60), "conf": 0.9})
        # profissional não paramentado entra na área estéril entre 9 s e 12 s
        if 9.0 <= t <= 12.0:
            x = 640 - 160 * min(1.0, (t - 9.0) / 1.0)
            dets.append({"classe": "pessoa_nao_paramentada", "bbox": _box(x, 380, 90, 120), "conf": 0.88})
        frames.append(dets)
    return frames


# ---------------------------------------------------------------------------
# Renderização
# ---------------------------------------------------------------------------
def render_pose_video(frames, path: Path, title: str) -> None:
    import cv2

    from src.video.video_io import video_writer

    with video_writer(path, FPS, (W, H)) as wr:
        _draw_pose_frames(cv2, wr, frames, title)


def _draw_pose_frames(cv2, wr, frames, title: str) -> None:
    for i, kp in enumerate(frames):
        img = np.full((H, W, 3), (38, 30, 24), np.uint8)
        cv2.line(img, (0, 463), (W, 463), (90, 90, 90), 2)
        pts = [(int(x), int(y)) for x, y, _ in kp]
        for a, b in SKELETON:
            cv2.line(img, pts[a], pts[b], (120, 220, 160), 6, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 5, (240, 240, 240), -1, cv2.LINE_AA)
        cv2.circle(img, pts[0], 18, (120, 220, 160), 3, cv2.LINE_AA)
        cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, f"t={i / FPS:5.1f}s", (W - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        wr.write(img)


def render_surgery_video(frames, path: Path, title: str) -> None:
    import cv2

    from src.video.video_io import video_writer

    with video_writer(path, FPS, (W, H)) as wr:
        _draw_surgery_frames(cv2, wr, frames, title)


def _draw_surgery_frames(cv2, wr, frames, title: str) -> None:
    colors = {"bisturi": (200, 200, 255), "tesoura": (80, 200, 255), "pinca": (255, 210, 120),
              "porta_agulha": (180, 255, 180), "afastador": (220, 160, 255)}
    for i, dets in enumerate(frames):
        img = np.full((H, W, 3), (70, 110, 60), np.uint8)
        cv2.rectangle(img, FIELD[:2], FIELD[2:], (90, 140, 80), -1)
        cv2.rectangle(img, TABLE[:2], TABLE[2:], (150, 150, 140), -1)
        cv2.ellipse(img, (235, 240), (60, 50), 0, 0, 360, (90, 90, 190), -1)
        for d in dets:
            x1, y1, x2, y2 = map(int, d["bbox"])
            if d["classe"] == "maos_equipe":
                cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), (35, 22), 0, 0, 360, (210, 230, 240), -1)
            elif d["classe"] == "pessoa_nao_paramentada":
                cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), (40, 55), 0, 0, 360, (60, 60, 200), -1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(d["classe"], (255, 255, 255)), -1)
        cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(img, f"t={i / FPS:5.1f}s", (W - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        wr.write(img)


VIDEOS = {
    "PAC-0001_fisio_ombro": {"tipo": "abducao_ombro", "perfil": "normal"},
    "PAC-0003_fisio_ombro": {"tipo": "abducao_ombro", "perfil": "hemiparesia_esquerda"},
    "PAC-0005_fisio_agachamento": {"tipo": "agachamento", "perfil": "perda_equilibrio_queda"},
    "PAC-0006_cirurgia": {"tipo": "cirurgia", "perfil": "falha_contagem_instrumental"},
}


def generate_all(output_dir: Path = VIDEO_DIR, render: bool = True) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    written = []
    for name, spec in VIDEOS.items():
        meta = {"video": f"{name}.mp4", "fps": FPS, "largura": W, "altura": H, **spec}
        if spec["tipo"] == "cirurgia":
            frames = surgery_session(rng)
            meta.update({"zonas": {"mesa_instrumental": TABLE, "campo_cirurgico": FIELD, "incisao": INCISION},
                         "instrumentos_iniciais": INSTRUMENTS, "frames": frames})
            (output_dir / f"{name}_detections.json").write_text(json.dumps(meta), encoding="utf-8")
            if render:
                render_surgery_video(frames, output_dir / f"{name}.mp4", "Campo cirurgico - vista superior")
        else:
            frames = shoulder_session("normal" if spec["perfil"] == "normal" else "paretico", rng) \
                if spec["tipo"] == "abducao_ombro" else squat_session(rng)
            meta.update({"formato": "COCO-17 (x, y, confianca)", "frames": frames})
            (output_dir / f"{name}_keypoints.json").write_text(json.dumps(meta), encoding="utf-8")
            if render:
                label = "Fisioterapia - abducao de ombro" if spec["tipo"] == "abducao_ombro" else "Fisioterapia - agachamento"
                render_pose_video(frames, output_dir / f"{name}.mp4", label)
        written.append(output_dir / f"{name}.mp4")
    return written


def main() -> None:
    paths = generate_all()
    print(f"{len(paths)} vídeos clínicos gerados em {VIDEO_DIR}")


if __name__ == "__main__":
    main()
