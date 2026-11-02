"""Zero-shot insect tracking with Meta SAM 3 (open weights, run locally).

Field cameras see insects at ~30 px in a 1920x1080 frame, which SAM 3 shrinks to 1008 px,
so each frame is split into overlapping tiles and every tile is tracked as its own video.
The default video thresholds are tuned for large objects: a bee crossing a frame in a
second barely overlaps itself between samples and gets dropped as a spurious tracklet, so
detection and new-object thresholds are lowered and sampling is 15 fps.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from .video import Window

MODEL_ID = "facebook/sam3"
CFG = {"score_threshold_detection": 0.3, "new_det_thresh": 0.4, "hotstart_unmatch_thresh": 12}


@dataclass
class Tracklet:
    """One SAM 3 object id inside one tile and one chunk; coordinates are full-frame pixels."""
    tile: int
    frame: list[int] = field(default_factory=list)
    cx: list[float] = field(default_factory=list)
    cy: list[float] = field(default_factory=list)
    area: list[int] = field(default_factory=list)
    box: list[list[int]] = field(default_factory=list)  # x0, y0, x1, y1
    score: list[float] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load():
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

    cfg = Sam3VideoConfig.from_pretrained(MODEL_ID)
    for k, v in CFG.items():
        setattr(cfg, k, v)
    model = Sam3VideoModel.from_pretrained(MODEL_ID, config=cfg, dtype=torch.bfloat16).to("cuda").eval()
    return model, Sam3VideoProcessor.from_pretrained(MODEL_ID)


def tiles(w: int, h: int, nx: int = 2, ny: int = 2, pad: int = 40) -> list[tuple[int, int, int, int]]:
    return [(max(0, i * w // nx - pad), max(0, j * h // ny - pad), min(w, (i + 1) * w // nx + pad), min(h, (j + 1) * h // ny + pad))
            for j in range(ny) for i in range(nx)]


@lru_cache(maxsize=1)
def _load_image():
    from transformers import Sam3Model, Sam3Processor

    return Sam3Model.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval(), Sam3Processor.from_pretrained(MODEL_ID)


def detect_boxes(img: np.ndarray, prompt: str, threshold: float = 0.3, grid: tuple[int, int] = (2, 2),
                 merge_px: float = 20.0) -> list[tuple[list[float], float]]:
    """Boxes (full-frame x0,y0,x1,y1) and scores for `prompt` in one RGB frame, tile by tile."""
    from PIL import Image

    model, proc = _load_image()
    h, w = img.shape[:2]
    out = []
    for x0, y0, x1, y1 in tiles(w, h, *grid):
        crop = Image.fromarray(img[y0:y1, x0:x1])
        inp = proc(images=crop, text=prompt, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            o = model(**inp)
        r = proc.post_process_object_detection(o, threshold=threshold, target_sizes=[crop.size[::-1]])[0]
        for b, s in zip(r["boxes"].float().tolist(), r["scores"].float().tolist()):
            out.append(([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], s))
    out.sort(key=lambda d: -d[1])  # tiles overlap: keep the best of near-identical boxes
    keep = []
    for b, s in out:
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if all(np.hypot(c[0] - (k[0] + k[2]) / 2, c[1] - (k[1] + k[3]) / 2) > merge_px for k, _ in keep):
            keep.append((b, s))
    return keep


def moving_hits(scan: dict[float, list], static_frac: float = 0.3, static_px: float = 15.0) -> list[tuple[float, list[float], float]]:
    """Scan hits (t, box, score) that are not fixed look-alikes.

    `scan` maps time (s) to boxes from a sparse pass of detect_boxes. A position that fires
    in a large share of samples is a bud or fruit, not a visitor.
    """
    pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, t, b, s) for t, bs in scan.items() for b, s in bs]
    n = max(1, len(scan))
    out = []
    for x, y, t, b, s in pts:
        seen = {p[2] for p in pts if abs(p[0] - x) < static_px and abs(p[1] - y) < static_px}
        if len(seen) / n < static_frac:
            out.append((t, b, s))
    return out


def windows(hits: list[tuple[float, list[float], float]], size: tuple[int, int], grid: tuple[int, int] = (2, 2),
            pad_s: float = 5.0, min_len_s: float = 6.0, tile_margin: float = 150.0) -> list[tuple[float, float, list[int]]]:
    """Seconds and tiles worth tracking around the given hits, padded and merged. Only tiles within
    reach of a hit are tracked, which skips most of the frame. The once-a-second scan sees a busy
    bee in only some seconds, so the padding is wide enough to join those sightings."""
    rects = tiles(*size, *grid)
    moving = []
    for t, b, *_ in hits:
        x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        near = {k for k, (x0, y0, x1, y1) in enumerate(rects)
                if x0 - tile_margin <= x <= x1 + tile_margin and y0 - tile_margin <= y <= y1 + tile_margin}
        moving.append((t, near))
    wins = []
    for t, near in sorted(moving, key=lambda m: m[0]):
        a, b = t - pad_s, t + pad_s
        if wins and a <= wins[-1][1]:
            wins[-1][1] = max(wins[-1][1], b)
            wins[-1][2] |= near
        else:
            wins.append([a, b, set(near)])
    out = []
    for a, b, near in wins:
        if b - a < min_len_s:
            c = (a + b) / 2
            a, b = c - min_len_s / 2, c + min_len_s / 2
        out.append((max(0.0, a), b, sorted(near)))
    return out


def active_windows(scan: dict[float, list], size: tuple[int, int], grid: tuple[int, int] = (2, 2), **kw) -> list[tuple[float, float, list[int]]]:
    return windows(moving_hits(scan), size, grid, **kw)


def cluster_hits(hits: list[tuple[float, list[float], float]], gap_s: float = 3.0, dist_px: float = 120.0) -> list[list[int]]:
    """Group hits that are plausibly one visitor (close in time and space); returns index lists."""
    order = sorted(range(len(hits)), key=lambda i: hits[i][0])
    groups: list[list[int]] = []
    for i in order:
        t, b, _ = hits[i]
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for g in reversed(groups):
            tj, bj, _ = hits[g[-1]]
            if t - tj > gap_s:
                continue
            if np.hypot(c[0] - (bj[0] + bj[2]) / 2, c[1] - (bj[1] + bj[3]) / 2) <= dist_px:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def track(win: Window, prompt: str = "bee", grid: tuple[int, int] = (2, 2), only: list[int] | None = None) -> list[Tracklet]:
    """Track every instance of `prompt` in each tile of the window (or just the tiles in `only`)."""
    model, proc = _load()
    h, w = win.frames[0].shape[:2]
    out: list[Tracklet] = []
    for ti, (x0, y0, x1, y1) in enumerate(tiles(w, h, *grid)):
        if only is not None and ti not in only:
            continue
        sub = [f[y0:y1, x0:x1] for f in win.frames]
        sess = proc.init_video_session(video=sub, inference_device="cuda", processing_device="cpu",
                                       video_storage_device="cpu", dtype=torch.bfloat16)
        proc.add_text_prompt(sess, prompt)
        per: dict[int, Tracklet] = {}
        with torch.inference_mode():
            for o in model.propagate_in_video_iterator(inference_session=sess):
                res = proc.postprocess_outputs(sess, o)
                scores = res.get("scores")
                for k, (oid, m) in enumerate(zip(res["object_ids"].tolist(), res["masks"])):
                    m = m.cpu().numpy().astype(bool)
                    if not m.any():
                        continue
                    ys, xs = np.nonzero(m)
                    t = per.setdefault(oid, Tracklet(tile=ti))
                    t.frame.append(win.index[o.frame_idx])
                    t.cx.append(float(xs.mean() + x0))
                    t.cy.append(float(ys.mean() + y0))
                    t.area.append(int(m.sum()))
                    t.box.append([int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0) + 1, int(ys.max() + y0) + 1])
                    t.score.append(float(scores[k]) if scores is not None else 1.0)
        out.extend(per.values())
        del sess
        torch.cuda.empty_cache()
    return out
