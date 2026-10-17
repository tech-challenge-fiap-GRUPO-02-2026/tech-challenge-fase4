"""Valida a integração real com o YOLOv8 (pose + detecção) em imagens públicas da Ultralytics.

Os vídeos de demonstração usam keypoints/detecções gravados em sidecar
(boneco renderizado não é "gente" para o YOLO). Este script prova que o
caminho com modelo real funciona ponta a ponta: baixa imagens de exemplo
da Ultralytics (pessoas reais), roda YOLOv8n-pose e YOLOv8n, converte a
pose para o layout OpenPose-18, calcula os ângulos usados na fisioterapia
e salva as imagens anotadas + um JSON com os resultados.

    python scripts/validate_yolo.py      → reports/output/yolo/
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import OUTPUT_DIR  # noqa: E402
from src.video.movement_analysis import shoulder_abduction, trunk_angle  # noqa: E402
from src.video.objects import YoloObjectDetector  # noqa: E402
from src.video.pose import COCO_KEYPOINTS, PoseSequence, YoloPoseEstimator, to_openpose18  # noqa: E402
from src.video.video_io import write_image  # noqa: E402

IMAGES = {"bus.jpg": "https://ultralytics.com/images/bus.jpg", "zidane.jpg": "https://ultralytics.com/images/zidane.jpg"}


def main() -> None:
    import cv2

    out_dir = OUTPUT_DIR / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = YoloPoseEstimator()
    det = YoloObjectDetector(class_map={})  # todas as classes COCO
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, url in IMAGES.items():
            path = Path(tmp) / name
            urllib.request.urlretrieve(url, path)
            frame = cv2.imread(str(path))
            res = pose.model(frame, verbose=False)[0]
            n_people = 0 if res.keypoints is None else len(res.keypoints)
            kp = pose.estimate_frame(frame)
            seq = PoseSequence(kp[None], 30, "yolov8-pose")
            objects = det.detect_frame(frame)
            annotated = res.plot()
            for d in objects:
                x1, y1, x2, y2 = map(int, d["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)
            write_image(out_dir / f"{Path(name).stem}_yolo.png", annotated)
            results.append({
                "imagem": name,
                "pessoas_detectadas_pose": n_people,
                "keypoints_confiaveis_pessoa_principal": int((kp[:, 2] > 0.5).sum()),
                "openpose18_shape": list(to_openpose18(kp).shape),
                "abducao_ombro_esq_graus": round(float(shoulder_abduction(seq, "esq")[0]), 1),
                "abducao_ombro_dir_graus": round(float(shoulder_abduction(seq, "dir")[0]), 1),
                "inclinacao_tronco_graus": round(float(trunk_angle(seq)[0]), 1),
                "objetos_coco": sorted({d["classe"] for d in objects}),
                "keypoints": {COCO_KEYPOINTS[i]: [round(float(v), 1) for v in kp[i]] for i in range(17)},
            })
    (out_dir / "validacao_yolo.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        print({k: v for k, v in r.items() if k != "keypoints"})
    print(f"Imagens anotadas e JSON em {out_dir}")


if __name__ == "__main__":
    main()
