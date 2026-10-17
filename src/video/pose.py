"""Estimação de pose: YOLOv8-pose (COCO-17) com mapeamento para o formato OpenPose.

O enunciado sugere **OpenPose** para análise postural. O OpenPose original
(CMU, Caffe) é pesado de compilar e não tem build oficial para Python 3.12+;
usamos o **YOLOv8-pose** (Ultralytics), que estima os mesmos 17 keypoints
do modelo COCO do OpenPose em tempo real na CPU. ``to_openpose18`` converte
para o layout COCO-18 do OpenPose (com o ponto "pescoço" sintetizado), de
modo que qualquer código escrito para a saída do OpenPose funciona aqui.

Fontes de pose suportadas (``get_pose_sequence``):
* ``sidecar`` — ``<video>_keypoints.json`` (vídeos de demonstração gerados);
* ``yolo``    — YOLOv8-pose rodando sobre os frames do vídeo (vídeo real).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COCO_KEYPOINTS = [
    "nariz", "olho_esq", "olho_dir", "orelha_esq", "orelha_dir",
    "ombro_esq", "ombro_dir", "cotovelo_esq", "cotovelo_dir", "punho_esq", "punho_dir",
    "quadril_esq", "quadril_dir", "joelho_esq", "joelho_dir", "tornozelo_esq", "tornozelo_dir",
]
KP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# OpenPose COCO-18: 0 nariz, 1 pescoço, 2 ombro_dir, 3 cotovelo_dir, 4 punho_dir, 5 ombro_esq,
# 6 cotovelo_esq, 7 punho_esq, 8 quadril_dir, 9 joelho_dir, 10 tornozelo_dir, 11 quadril_esq,
# 12 joelho_esq, 13 tornozelo_esq, 14 olho_dir, 15 olho_esq, 16 orelha_dir, 17 orelha_esq
_OPENPOSE18_FROM_COCO = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

YOLO_POSE_MODEL = "yolov8n-pose.pt"


@dataclass
class PoseSequence:
    keypoints: np.ndarray  # (frames, 17, 3) → x, y, confiança (NaN onde não detectado)
    fps: float
    source: str

    @property
    def n_frames(self) -> int:
        return len(self.keypoints)

    def xy(self, name: str) -> np.ndarray:
        return self.keypoints[:, KP[name], :2]


def to_openpose18(kp17: np.ndarray) -> np.ndarray:
    """(…,17,3) COCO → (…,18,3) layout OpenPose, com pescoço = média dos ombros."""
    out = np.zeros(kp17.shape[:-2] + (18, 3))
    for op_idx, coco_idx in enumerate(_OPENPOSE18_FROM_COCO):
        if coco_idx is None:
            out[..., op_idx, :2] = (kp17[..., 5, :2] + kp17[..., 6, :2]) / 2
            out[..., op_idx, 2] = np.minimum(kp17[..., 5, 2], kp17[..., 6, 2])
        else:
            out[..., op_idx, :] = kp17[..., coco_idx, :]
    return out


def sidecar_path(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.with_name(f"{p.stem}_keypoints.json")


def load_sidecar(video_path: str | Path) -> tuple[PoseSequence, dict]:
    meta = json.loads(sidecar_path(video_path).read_text(encoding="utf-8"))
    kps = np.array(meta.pop("frames"), dtype=float)
    return PoseSequence(kps, float(meta.get("fps", 30)), "keypoints_sidecar (COCO-17)"), meta


class YoloPoseEstimator:
    def __init__(self, model: str = YOLO_POSE_MODEL, conf: float = 0.3):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf = conf

    def estimate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Keypoints (17,3) da pessoa com maior bbox no frame (NaN se ninguém)."""
        res = self.model(frame, verbose=False, conf=self.conf)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return np.full((17, 3), np.nan)
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return res.keypoints.data.cpu().numpy()[int(areas.argmax())]

    def estimate_video(self, video_path: str | Path, stride: int = 1, max_frames: int | None = None) -> PoseSequence:
        from src.video.video_io import read_frames, video_info

        fps = video_info(video_path)["fps"]
        kps = [self.estimate_frame(f) for _, f in read_frames(video_path, stride, max_frames)]
        return PoseSequence(np.array(kps), fps / stride, f"yolov8-pose ({YOLO_POSE_MODEL})")


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def get_pose_sequence(video_path: str | Path, prefer: str = "auto", stride: int = 1) -> tuple[PoseSequence, dict]:
    """``auto``: usa o sidecar se existir (vídeo de demonstração); senão YOLOv8-pose."""
    if prefer in ("auto", "sidecar") and sidecar_path(video_path).exists():
        return load_sidecar(video_path)
    if prefer == "sidecar":
        raise FileNotFoundError(f"Sem keypoints para {video_path}")
    if not yolo_available():
        raise RuntimeError("ultralytics não instalado — pip install -r requirements-video.txt")
    return YoloPoseEstimator().estimate_video(video_path, stride=stride), {}
