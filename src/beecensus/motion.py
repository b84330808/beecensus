"""Moving-blob candidates from frame differencing, for insects too blurred for SAM 3.

A honeybee in fast flight is a 30 px smear that SAM 3 scores below 0.1 for "bee". Field
cameras are fixed, so something that changed since a fifteenth of a second ago, and changes
again a fifteenth of a second later, is moving now; Polytrack pairs its detector with the same
idea. Blobs are only candidates: Muse Spark Open decides which are insects, as for SAM 3's hits.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from . import classify
from .detect import Tracklet
from .video import Window

GAP = 2  # frames between the three differenced frames (one sample at 15 fps)
THRESH = 15  # grey-level change that counts as movement
MIN_AREA, MAX_AREA = 80, 6000  # px: a 30 px bee, smeared up to a few body lengths
MAX_ELONGATION = 4.0  # leaf edges swaying make thin streaks
MAX_BLOBS = 10  # more at once is a gust or a light change moving whole plants; a frame then says nothing about insects
SCAN_FPS = 3  # cheap (CPU only), so denser than SAM 3's scan: a bee crossing the frame in 2 s is still sampled
SCORE = 0.2  # below SAM 3's scan threshold, so a SAM 3 close-up of the same visitor is shown to Spark first
LINK_PX = 100.0  # a bee covers up to ~60 px between samples at 15 fps
LINK_SAMPLES = 3
CACHE = Path(__file__).resolve().parents[2] / "cache"


def _grey(rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.int16)


def blobs(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> list[tuple[list[float], float]]:
    """Boxes (x0, y0, x1, y1) and scores of what is moving in grey frame b, given the grey frames before and after."""
    m = np.minimum(np.abs(b - a), np.abs(c - b))
    mask = cv2.morphologyEx((m > THRESH).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for x, y, w, h, area in st[1:n]:
        if MIN_AREA <= area <= MAX_AREA and max(w, h) <= MAX_ELONGATION * min(w, h):
            strength = float(m[y:y + h, x:x + w].max()) / 255
            out.append(([float(x), float(y), float(x + w), float(y + h)], SCORE * min(1.0, 2 * strength)))
    return out


def scan(path, t_start: float, t_end: float) -> tuple[dict[float, list], dict]:
    """Moving blobs SCAN_FPS times a second, plus a close-up of each for Spark; cached like the SAM 3 scan."""
    key = CACHE / f"{Path(path).stem}_{t_start:.0f}_{t_end:.0f}_motion_{SCAN_FPS}fps.json"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hits = {float(t): b for t, b in json.loads(key.read_text()).items()} if key.exists() else None
    step = max(1, int(round(fps / SCAN_FPS)))
    first, last = int(t_start * fps), int(t_end * fps)
    centres = range(first - first % step + step, last - GAP, step)
    want = {c + d: c for c in centres for d in (-GAP, 0, GAP)}
    if hits is not None:  # only the frames with blobs are needed again, for the close-ups
        want = {c: c for c in centres if 0 < len(hits.get(round(c / fps, 2), [])) <= MAX_BLOBS}
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, first))
    found, crops, buf = {}, {}, {}
    for i in range(max(0, first), last):
        if not cap.grab():
            break
        if i not in want:
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        c = want[i]
        t = round(c / fps, 2)
        if hits is not None:
            for b, _ in hits[t]:
                crops[(t, tuple(round(v) for v in b))] = classify.crop(rgb, b)
            continue
        buf[i] = (_grey(rgb), rgb if i == c else None)
        if i == c + GAP and all(k in buf for k in (c - GAP, c, c + GAP)):
            found[t] = blobs(buf[c - GAP][0], buf[c][0], buf[c + GAP][0])
            for b, _ in found[t]:
                crops[(t, tuple(round(v) for v in b))] = classify.crop(buf[c][1], b)
            buf = {k: v for k, v in buf.items() if k >= c + step - GAP}
    cap.release()
    if hits is None:
        hits = found
        CACHE.mkdir(exist_ok=True)
        key.write_text(json.dumps({str(t): b for t, b in hits.items()}))
    return {t: (b if len(b) <= MAX_BLOBS else []) for t, b in hits.items()}, crops


def tracklets(win: Window) -> list[Tracklet]:
    """Moving blobs in consecutive samples of a window, chained by distance (no appearance needed)."""
    out: list[Tracklet] = []
    active: list[Tracklet] = []
    g = [_grey(f) for f in win.frames[:2]]
    for k in range(1, len(win.frames) - 1):
        g.append(_grey(win.frames[k + 1]))
        f = win.index[k]
        found = blobs(g[0], g[1], g[2])
        for b, s in found if len(found) <= MAX_BLOBS else []:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            live = [t for t in active if f - t.frame[-1] <= LINK_SAMPLES * (win.index[1] - win.index[0]) and t.frame[-1] != f]
            near = min(live, key=lambda t: np.hypot(t.cx[-1] - cx, t.cy[-1] - cy), default=None)
            if near is None or np.hypot(near.cx[-1] - cx, near.cy[-1] - cy) > LINK_PX:
                near = Tracklet(tile=-1)
                active.append(near)
                out.append(near)
            near.frame.append(f)
            near.cx.append(cx)
            near.cy.append(cy)
            near.area.append(int((b[2] - b[0]) * (b[3] - b[1])))
            near.box.append([int(v) for v in b])
            near.score.append(s)
        g.pop(0)
    return out
