"""E/S de vídeo com OpenCV tolerante a caminhos não-ASCII.

O backend FFMPEG do OpenCV no Windows não abre caminhos com acentos
(ex.: ``C:\\...\\programação\\...``). Estas funções gravam/leem através de
um arquivo temporário em caminho ASCII e movem o resultado.
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def _is_ascii(path: Path) -> bool:
    return all(ord(c) < 128 for c in str(path))


class _FfmpegH264Writer:
    """H.264 (libx264, yuv420p) via imageio-ffmpeg — reproduzível em qualquer navegador."""

    def __init__(self, path: Path, fps: int, size: tuple[int, int]):
        import imageio_ffmpeg

        self._gen = imageio_ffmpeg.write_frames(
            str(path), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
            output_params=["-crf", "28", "-preset", "veryfast", "-movflags", "+faststart"],
        )
        self._gen.send(None)

    def write(self, frame_bgr: np.ndarray) -> None:
        self._gen.send(np.ascontiguousarray(frame_bgr[:, :, ::-1]))

    def release(self) -> None:
        self._gen.close()


def _open_writer(target: Path, fps: int, size: tuple[int, int]):
    try:
        return _FfmpegH264Writer(target, fps, size)
    except ImportError:
        import cv2

        # sem imageio-ffmpeg: MPEG-4 Part 2 (toca no VLC, mas não em todos os navegadores)
        return cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


@contextmanager
def video_writer(path: str | Path, fps: int, size: tuple[int, int]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = None
    target = path
    if not _is_ascii(path):
        tmpdir = tempfile.mkdtemp(prefix="tc4_")
        target = Path(tmpdir) / f"out{path.suffix}"
    writer = _open_writer(target, fps, size)
    try:
        yield writer
    finally:
        writer.release()
        if tmpdir:
            shutil.move(str(target), str(path))
            shutil.rmtree(tmpdir, ignore_errors=True)


def read_frames(path: str | Path, stride: int = 1, max_frames: int | None = None):
    """Itera (índice, frame BGR) e devolve também o fps via atributo da função geradora."""
    import cv2

    path = Path(path)
    tmpdir = None
    source = path
    if not _is_ascii(path):
        tmpdir = tempfile.mkdtemp(prefix="tc4_")
        source = Path(tmpdir) / f"in{path.suffix}"
        shutil.copy(path, source)
    cap = cv2.VideoCapture(str(source))
    try:
        i = 0
        yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, frame
                yielded += 1
                if max_frames and yielded >= max_frames:
                    break
            i += 1
    finally:
        cap.release()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def video_info(path: str | Path) -> dict:
    import cv2

    path = Path(path)
    tmpdir = None
    source = path
    if not _is_ascii(path):
        tmpdir = tempfile.mkdtemp(prefix="tc4_")
        source = Path(tmpdir) / f"in{path.suffix}"
        shutil.copy(path, source)
    cap = cv2.VideoCapture(str(source))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "largura": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "altura": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return info


def write_image(path: str | Path, image: np.ndarray) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if ok:
        path.write_bytes(buf.tobytes())
