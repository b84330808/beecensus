"""Flowers and flower visits.

Flowers barely move, so SAM 3 (image mode) finds them on a few keyframes and positions are
clustered into stable flower ids. A visit is an insect's centre staying inside a flower's
circle long enough to land, not just flying over it.
"""

from dataclasses import dataclass

import numpy as np

from .detect import detect_boxes
from .tracks import Insect

MATCH_PX = 40.0  # same flower on two keyframes
VISIT_MIN_S = 0.3
VISIT_MERGE_S = 2.0  # leaving a flower for less than this (a hop off and back) is still the same visit
RADIUS_GAIN = 1.2  # landing on the petal edge counts
MAX_ASPECT = 1.8  # an open flower is round from above; long white shapes are pale leaves or labels


@dataclass
class Flower:
    id: int
    x: float
    y: float
    r: float
    seen: int  # keyframes it was found on


@dataclass
class Visit:
    insect: int
    flower: int
    start: int  # frame numbers
    end: int


def find_flowers(keyframes: list[np.ndarray], prompt: str = "white flower") -> list[Flower]:
    obs = []  # (x, y, r)
    for img in keyframes:
        for b, _ in detect_boxes(img, prompt, threshold=0.4, merge_px=MATCH_PX / 2):
            w, h = b[2] - b[0], b[3] - b[1]
            if max(w, h) > MAX_ASPECT * max(1.0, min(w, h)):
                continue
            obs.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) / 2))
    flowers: list[list[tuple]] = []
    for o in obs:
        for f in flowers:
            if np.hypot(o[0] - np.mean([p[0] for p in f]), o[1] - np.mean([p[1] for p in f])) < MATCH_PX:
                f.append(o)
                break
        else:
            flowers.append([o])
    out = [Flower(0, float(np.mean([p[0] for p in f])), float(np.mean([p[1] for p in f])), float(np.median([p[2] for p in f])), len(f))
           for f in flowers]
    out.sort(key=lambda f: (round(f.y / 200), f.x))  # reading order, so ids are easy to find on screen
    for k, f in enumerate(out):
        f.id = k + 1
    return out


def visits(insects: list[Insect], flowers: list[Flower], fps: float) -> list[Visit]:
    out = []
    for ins in insects:
        for fl in flowers:
            inside = np.hypot(ins.cx - fl.x, ins.cy - fl.y) <= fl.r * RADIUS_GAIN
            runs, cur = [], None
            for f, ok in zip(ins.frame, inside):
                if ok:
                    if cur and (f - cur[1]) / fps <= VISIT_MERGE_S:
                        cur[1] = f
                    else:
                        cur = [f, f]
                        runs.append(cur)
            for a, b in runs:
                if (b - a) / fps >= VISIT_MIN_S:
                    out.append(Visit(ins.id, fl.id, int(a), int(b)))
    out.sort(key=lambda v: v.start)
    return out
