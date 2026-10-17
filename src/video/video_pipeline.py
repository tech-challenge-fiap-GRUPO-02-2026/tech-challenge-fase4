"""Pipeline de vídeo: frames → pose/objetos → análise do protocolo → relatório automático.

    vídeo ─┬─ pose (YOLOv8-pose / OpenPose COCO) ── movement_analysis ──┐
           └─ objetos (YOLOv8) ── procedure_analysis ───────────────────┴─► achados + relatório .md
                                                                           + frames-chave anotados
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import OUTPUT_DIR
from src.models import Finding, modality_risk
from src.video import objects, pose
from src.video.movement_analysis import ANALYZERS, PROTOCOLS
from src.video.procedure_analysis import analyze_surgery

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


@dataclass
class VideoReport:
    patient_id: str | None
    video: str
    kind: str
    source: str
    analysis: dict
    findings: list[Finding] = field(default_factory=list)
    key_frame_images: list[str] = field(default_factory=list)
    report_path: str | None = None

    @property
    def risk(self) -> float:
        return modality_risk(self.findings)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id, "video": self.video, "tipo": self.kind, "fonte_deteccao": self.source,
            "analise": self.analysis, "achados": [f.to_dict() for f in self.findings],
            "frames_chave": self.key_frame_images, "relatorio": self.report_path, "risco_video": self.risk,
        }


def _annotate(frame: np.ndarray, kp: np.ndarray | None, dets: list[dict] | None, label: str) -> np.ndarray:
    import cv2

    img = frame.copy()
    if kp is not None and not np.isnan(kp).all():
        for a, b in SKELETON:
            if np.isnan(kp[[a, b], :2]).any():
                continue
            cv2.line(img, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), (0, 200, 255), 3, cv2.LINE_AA)
    for d in dets or []:
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = (0, 0, 255) if d["classe"] == "pessoa_nao_paramentada" else (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d['classe']} {d['conf']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), (0, 0, 160), -1)
    cv2.putText(img, label[:80], (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _save_key_frames(video_path: Path, key_frames: list[dict], kps: np.ndarray | None,
                     det_frames: list | None, out_dir: Path) -> list[str]:
    from src.video.video_io import read_frames, write_image

    wanted = {k["frame"]: k["motivo"] for k in key_frames}
    if not wanted or not video_path.exists():
        return []
    saved = []
    for i, frame in read_frames(video_path):
        if i in wanted:
            kp = kps[i] if kps is not None and i < len(kps) else None
            dets = det_frames[i] if det_frames is not None and i < len(det_frames) else None
            path = out_dir / f"{video_path.stem}_f{i:05d}.png"
            write_image(path, _annotate(frame, kp, dets, f"t={i / 30:.1f}s  {wanted[i]}"))
            saved.append(str(path.relative_to(out_dir.parents[1])).replace("\\", "/"))
        if i > max(wanted):
            break
    return saved


def write_markdown(report: VideoReport, out_dir: Path) -> Path:
    lines = [f"# Relatório automático de vídeo — {report.video}", "",
             f"- **Paciente:** {report.patient_id or '—'}",
             f"- **Tipo:** {PROTOCOLS.get(report.kind, {}).get('nome', report.kind)}",
             f"- **Fonte de detecção:** {report.source}",
             f"- **Risco da modalidade:** {report.risk:.2f}", ""]
    metrics = report.analysis.get("metricas", {})
    if metrics:
        lines += ["## Métricas", "", "| Métrica | Valor |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in metrics.items()]
        lines.append("")
    reps = report.analysis.get("por_repeticao", [])
    if reps:
        cols = list(reps[0].keys())
        lines += ["## Repetições", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in reps]
        lines.append("")
    lines += ["## Desvios / falhas detectados", ""]
    if report.findings:
        lines += [f"- **[{f.severity.upper()}]** {f.message}" + (f" _(em {f.time_ref})_" if f.time_ref else "")
                  for f in report.findings]
    else:
        lines.append("- Nenhum desvio em relação ao protocolo esperado.")
    if report.key_frame_images:
        lines += ["", "## Frames-chave", ""] + [f"![frame](../../{p})" for p in report.key_frame_images]
    lines += ["", "> Relatório gerado automaticamente — requer revisão do fisioterapeuta/cirurgião responsável."]
    path = out_dir / f"{Path(report.video).stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def analyze_video(video_path: str | Path, patient_id: str | None = None, exercise: str | None = None,
                  prefer: str = "auto", stride: int = 1, weights: str | None = None,
                  out_dir: Path = OUTPUT_DIR / "video", save_frames: bool = True) -> VideoReport:
    video_path = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_sidecar_det = prefer in ("auto", "sidecar") and objects.sidecar_path(video_path).exists()
    kind = exercise

    if use_sidecar_det or exercise == "cirurgia":
        if use_sidecar_det:
            det = objects.SidecarDetector(video_path)
            frames, fps, zones, source = det.frames, det.fps, det.meta["zonas"], det.source
        else:
            yolo = objects.YoloObjectDetector(weights or objects.YOLO_DET_MODEL)
            frames, fps = yolo.detect_video(video_path, stride)
            from src.datasets.generate_video import FIELD, INCISION, TABLE

            zones, source = {"mesa_instrumental": TABLE, "campo_cirurgico": FIELD, "incisao": INCISION}, yolo.source
        proc = analyze_surgery(frames, fps, zones, patient_id)
        report = VideoReport(patient_id, video_path.name, "cirurgia", source, proc.to_dict(), proc.findings)
        if save_frames:
            report.key_frame_images = _save_key_frames(video_path, proc.key_frames, None, frames, out_dir)
    else:
        seq, meta = pose.get_pose_sequence(video_path, prefer=prefer, stride=stride)
        kind = exercise or meta.get("tipo")
        if kind not in ANALYZERS:
            raise ValueError(f"Exercício desconhecido: {kind!r} — use --exercise {sorted(ANALYZERS)}")
        ex = ANALYZERS[kind](seq, patient_id)
        report = VideoReport(patient_id, video_path.name, kind, seq.source, ex.to_dict(), ex.findings)
        if save_frames:
            report.key_frame_images = _save_key_frames(video_path, ex.key_frames, seq.keypoints, None, out_dir)

    report.report_path = str(write_markdown(report, out_dir).relative_to(out_dir.parents[1])).replace("\\", "/")
    return report
