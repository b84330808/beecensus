"""Frame access for long field videos (10 min at 1080p does not fit in memory at full rate)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Window:
    frames: list[np.ndarray]  # RGB uint8, full resolution
    index: list[int]  # source frame number of each frame
    fps: float


def info(path: str | Path) -> tuple[float, int, int, int]:
    """(fps, frame_count, width, height). frame_count can be 0 for AVIs without an index."""
    cap = cv2.VideoCapture(str(path))
    out = (cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
           int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return out


def iter_frames(path: str | Path, t_start: float, t_end: float, sample_fps: float):
    """(frame number, RGB frame) between t_start and t_end seconds, every (fps / sample_fps)-th frame, one at a time."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    first, last = int(t_start * fps), int(t_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    try:
        for i in range(first, last):
            if not cap.grab():
                break
            if i % step == 0:
                ok, bgr = cap.retrieve()
                if ok:
                    yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_window(path: str | Path, t_start: float, t_end: float, sample_fps: float) -> Window:
    """Frames between t_start and t_end seconds, keeping every (fps / sample_fps)-th frame."""
    pairs = list(iter_frames(path, t_start, t_end, sample_fps))
    return Window([f for _, f in pairs], [i for i, _ in pairs], info(path)[0])


def read_frame(path: str | Path, frame_no: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, bgr = cap.read()
    cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
