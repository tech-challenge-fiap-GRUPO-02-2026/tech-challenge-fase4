"""Detecção de objetos e áreas críticas — YOLOv8 (Ultralytics).

Dois detectores com a mesma saída (lista de ``{"classe", "bbox", "conf"}``
por frame):

* ``YoloObjectDetector`` — YOLOv8n pré-treinado no COCO. Das 80 classes do
  COCO, as úteis no centro cirúrgico são ``scissors`` (tesoura), ``knife``
  (bisturi/lâmina) e ``person``. Instrumental específico (pinça,
  porta-agulha, afastador) exige um YOLOv8 *fine-tuned* em datasets de
  instrumentos cirúrgicos (ex.: m2cai16-tool-locations, Cholec80) — basta
  apontar ``--weights`` para o ``.pt`` treinado, o restante do pipeline
  não muda;
* ``SidecarDetector`` — lê ``<video>_detections.json`` (vídeos de
  demonstração gerados, que emulam a saída desse modelo fine-tuned).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COCO_TO_CLINICAL = {"scissors": "tesoura", "knife": "bisturi", "person": "pessoa"}
YOLO_DET_MODEL = "yolov8n.pt"


def sidecar_path(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.with_name(f"{p.stem}_detections.json")


class SidecarDetector:
    source = "detections_sidecar (formato YOLO)"

    def __init__(self, video_path: str | Path):
        self.meta = json.loads(sidecar_path(video_path).read_text(encoding="utf-8"))
        self.frames: list[list[dict]] = self.meta.pop("frames")
        self.fps = float(self.meta.get("fps", 30))


class YoloObjectDetector:
    def __init__(self, weights: str = YOLO_DET_MODEL, conf: float = 0.25, class_map: dict | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.class_map = class_map if class_map is not None else COCO_TO_CLINICAL
        self.source = f"yolov8 ({Path(weights).name})"

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        res = self.model(frame, verbose=False, conf=self.conf)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            name = res.names[int(cls)]
            if self.class_map and name not in self.class_map:
                continue
            out.append({"classe": self.class_map.get(name, name), "bbox": [round(float(v), 1) for v in box],
                        "conf": round(float(conf), 3)})
        return out

    def detect_video(self, video_path: str | Path, stride: int = 1, max_frames: int | None = None) -> tuple[list[list[dict]], float]:
        from src.video.video_io import read_frames, video_info

        fps = video_info(video_path)["fps"]
        frames = [self.detect_frame(f) for _, f in read_frames(video_path, stride, max_frames)]
        return frames, fps / stride


def center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def inside(bbox: list[float], zone: list[float] | tuple) -> bool:
    cx, cy = center(bbox)
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def overlaps(bbox: list[float], zone: list[float] | tuple) -> bool:
    return not (bbox[2] < zone[0] or bbox[0] > zone[2] or bbox[3] < zone[1] or bbox[1] > zone[3])
