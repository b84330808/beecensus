"""Turn SAM 3 tracklets into insects: drop static look-alikes, merge tile duplicates, bridge gaps.

A bee is often lost for a few frames (motion blur, crossing a tile seam, hidden under a
petal) and comes back as a new object id, so one insect arrives as several tracklets.
"""

from dataclasses import dataclass

import numpy as np

from .detect import Tracklet

STATIC_PX = 3.0  # a tracklet whose centre never moves more than this ...
STATIC_MIN_S = 8.0  # ... for this long is a bud, fruit or leaf spot; a bee on a flower moves on sooner.
# Shorter still things go to Spark, which rejects look-alikes.
DUP_PX = 40.0  # two tracklets this close on shared frames are the same insect seen from two tiles
GAP_S = 2.0  # longest disappearance bridged
SPEED_PX_S = 500.0  # how far an insect can plausibly travel per second while unseen
MIN_DETS = 3


@dataclass
class Insect:
    id: int
    frame: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    area: np.ndarray
    box: np.ndarray  # N x 4
    score: np.ndarray
    parts: int  # tracklets merged into this insect

    @property
    def start(self) -> int:
        return int(self.frame[0])

    @property
    def end(self) -> int:
        return int(self.frame[-1])


def is_static(t: Tracklet, fps: float = 30.0) -> bool:
    return (len(t.cx) >= MIN_DETS and (t.frame[-1] - t.frame[0]) / fps >= STATIC_MIN_S
            and np.std(t.cx) < STATIC_PX and np.std(t.cy) < STATIC_PX)


def _dup(a: Tracklet, b: Tracklet, tol: int) -> bool:
    fa = np.asarray(a.frame)
    d = []
    for f, x, y in zip(b.frame, b.cx, b.cy):
        k = np.argmin(np.abs(fa - f))
        if abs(fa[k] - f) <= tol:
            d.append(np.hypot(a.cx[k] - x, a.cy[k] - y))
    return len(d) > 0 and float(np.median(d)) < DUP_PX


def link(tracklets: list[Tracklet], fps: float, step: int) -> list[Insect]:
    ts = [t for t in tracklets if len(t.frame) >= MIN_DETS and not is_static(t, fps)]
    ts.sort(key=lambda t: t.frame[0])
    parent = list(range(len(ts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # same insect seen twice where tiles overlap
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if ts[j].frame[0] > ts[i].frame[-1]:
                break
            if _dup(ts[i], ts[j], step):
                parent[find(j)] = find(i)
    # bridge disappearances: each tracklet end continues into at most one later start
    groups: dict[int, list[int]] = {}
    for i in range(len(ts)):
        groups.setdefault(find(i), []).append(i)
    chains = [sorted(g, key=lambda i: ts[i].frame[0]) for g in groups.values()]
    ends = {c[-1]: c for c in chains}
    starts = sorted(chains, key=lambda c: ts[c[0]].frame[0])
    used = set()
    for c in starts:
        a = ts[c[0]]
        best, best_d = None, None
        for e, ce in ends.items():
            if ce is c or e in used:
                continue
            b = ts[e]
            gap = (a.frame[0] - b.frame[-1]) / fps
            if not (0 < gap <= GAP_S):
                continue
            d = np.hypot(a.cx[0] - b.cx[-1], a.cy[0] - b.cy[-1])
            if d <= DUP_PX + SPEED_PX_S * gap and (best_d is None or d < best_d):
                best, best_d = e, d
        if best is not None:
            used.add(best)
            parent[find(c[0])] = find(best)
    merged: dict[int, list[Tracklet]] = {}
    for i, t in enumerate(ts):
        merged.setdefault(find(i), []).append(t)
    insects = []
    for parts in merged.values():
        rows = {}
        for t in parts:
            for k, f in enumerate(t.frame):
                if f not in rows or t.score[k] > rows[f][5]:
                    rows[f] = (t.cx[k], t.cy[k], t.area[k], t.box[k], f, t.score[k])
        r = [rows[f] for f in sorted(rows)]
        insects.append(Insect(0, np.array([x[4] for x in r]), np.array([x[0] for x in r]), np.array([x[1] for x in r]),
                              np.array([x[2] for x in r]), np.array([x[3] for x in r]), np.array([x[5] for x in r]), len(parts)))
    insects.sort(key=lambda s: s.start)
    for k, s in enumerate(insects):
        s.id = k + 1
    return insects
