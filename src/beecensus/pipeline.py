"""Video in, insects + flowers + visits out, cached as JSON next to the run.

1. scan: SAM 3 image pass once a second finds where anything bee-like appears.
2. track: SAM 3 video tracking only around seconds with a moving candidate (insects are
   on screen a small share of a survey, so this skips most of the footage).
3. link tracklets into insects, find flowers, count visits.
4. Muse Spark Open looks at close-ups of each insect: FIT group, or reject it.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import classify, detect, flowers, tracks, video

SAMPLE_FPS = 15
SCAN_FPS = 1
CHUNK_S = 15  # SAM 3 keeps every frame of a session on the CPU; 15 s x 15 fps x 4 tiles fits
FLOWER_EVERY_S = 100  # Polytrack's interval; flowers barely move
CROPS_PER_INSECT = 3


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


CACHE = Path(__file__).resolve().parents[2] / "cache"


def scan(path, t_start: float, t_end: float, prompt: str) -> tuple[dict[float, list], dict]:
    """Boxes per scanned second, plus a close-up of each box for Spark.

    The scan is the slow half of a run, so its boxes are cached per video, range and prompt;
    a rerun only re-reads the frames that had hits to rebuild the close-ups.
    """
    key = CACHE / f"{Path(path).stem}_{t_start:.0f}_{t_end:.0f}_{prompt.replace(' ', '-')}_{SCAN_FPS}fps.json"
    hits, crops = {}, {}
    if key.exists():
        hits = {float(t): b for t, b in json.loads(key.read_text()).items()}
        fps = video.info(path)[0]
        for t, bs in hits.items():
            if bs:
                f = video.read_frame(path, int(round(t * fps)))
                for b, _ in bs:
                    crops[(t, tuple(round(v) for v in b))] = classify.crop(f, b)
        return hits, crops
    win = video.read_window(path, t_start, t_end, SCAN_FPS)
    for i, f in zip(win.index, win.frames):
        t = i / win.fps
        hits[t] = detect.detect_boxes(f, prompt)
        for b, _ in hits[t]:
            crops[(t, tuple(round(v) for v in b))] = classify.crop(f, b)
    CACHE.mkdir(exist_ok=True)
    key.write_text(json.dumps({str(t): b for t, b in hits.items()}))
    return hits, crops


def gate(moving: list, crops: dict, log=print) -> tuple[list, list[dict]]:
    """Muse Spark Open decides which moving scan hits are insects before any GPU time is spent tracking them.

    Hits close in time and space are one candidate visitor; Spark sees up to three of its close-ups.
    Wind-blown leaves and glare that slip past the static filter are dropped here.
    """
    groups = detect.cluster_hits(moving)
    jobs = []
    for g in groups:
        best = sorted(g, key=lambda i: -moving[i][2])[:CROPS_PER_INSECT]
        jobs.append([crops[(moving[i][0], tuple(round(v) for v in moving[i][1]))] for i in best])
    with ThreadPoolExecutor(4) as ex:
        verdicts = list(ex.map(lambda c: classify.classify(c, tag="gate"), jobs))
    kept, report = [], []
    for g, v in zip(groups, verdicts):
        ok = v["group"] != "not_insect"
        if ok:
            kept += [(*moving[i], v["group"]) for i in g]
        report.append({"t": [round(moving[i][0], 1) for i in g], "box": [round(v_) for v_ in moving[g[0]][1]],
                       "group": v["group"], "confidence": v["confidence"], "evidence": v["evidence"], "kept": ok})
    log(f"gate: {len(groups)} candidates, {sum(r['kept'] for r in report)} confirmed as insects by Spark")
    return kept, report


def best_frames(ins: tracks.Insect, fps: float, k: int = CROPS_PER_INSECT) -> list[int]:
    """Indices into the insect's samples where SAM 3 is surest it sees the insect, spread over the track.

    Not "slowest": after a bee leaves, the tracker can sit on the empty flower centre, and those
    still frames would show Spark a flower.
    """
    order = np.lexsort((-ins.area, -ins.score))
    picked = []
    for i in order:
        if all(abs(ins.frame[i] - ins.frame[j]) >= fps / 2 for j in picked):
            picked.append(int(i))
        if len(picked) == k:
            break
    return picked


def gate_confirmed(ins: tracks.Insect, gated: list, fps: float, px: float = 80.0, s: float = 1.0) -> str | None:
    """The group Spark gave a confirmed scan hit that lies on this track, if any."""
    for t, b, _, group in gated:
        f = t * fps
        k = np.argmin(np.abs(ins.frame - f))
        if abs(ins.frame[k] - f) <= s * fps and np.hypot(ins.cx[k] - (b[0] + b[2]) / 2, ins.cy[k] - (b[1] + b[3]) / 2) <= px:
            return group
    return None


def classify_insects(path, insects: list[tracks.Insect], fps: float) -> dict[int, dict]:
    jobs = []
    for ins in insects:
        crops = []
        for i in best_frames(ins, fps):
            fr = video.read_frame(path, int(ins.frame[i]))
            if fr is not None:
                crops.append(classify.crop(fr, ins.box[i]))
        jobs.append((ins.id, crops))
    with ThreadPoolExecutor(4) as ex:
        res = ex.map(lambda j: (j[0], classify.classify(j[1], tag="pipeline") if j[1] else
                                {"group": "unknown", "confidence": 0.0, "evidence": "no crop", "raw": ""}), jobs)
        return dict(res)


def run(path: str | Path, t_start: float = 0.0, t_end: float | None = None, prompt: str = "bee",
        flower_prompt: str = "white flower", out: str | Path | None = None, use_spark: bool = True, log=print) -> dict:
    fps, n, w, h = video.info(path)
    if t_end is None:
        t_end = n / fps if n else 600.0
    t0 = time.time()
    hits, crops = scan(path, t_start, t_end, prompt)
    moving = detect.moving_hits(hits)
    log(f"scan {t_start:.0f}-{t_end:.0f}s: {sum(map(len, hits.values()))} hits, {len(moving)} moving ({time.time() - t0:.0f}s)")
    gated, gate_report = gate(moving, crops, log) if use_spark else ([(*m, "unknown") for m in moving], [])
    wins = detect.windows(gated, (w, h))
    log(f"{len(wins)} windows "
        f"({sum(b - a for a, b, _ in wins):.0f}s, {sum((b - a) * len(k) for a, b, k in wins):.0f} tile-seconds to track) ({time.time() - t0:.0f}s)")
    tls = []
    for a, b, near in wins:
        for c in np.arange(a, b, CHUNK_S):
            win = video.read_window(path, c, min(c + CHUNK_S, b), SAMPLE_FPS)
            if win.frames:
                tls += detect.track(win, prompt, only=near)
        log(f"tracked {a:.0f}-{b:.0f}s: {len(tls)} tracklets ({time.time() - t0:.0f}s)")
    step = max(1, int(round(fps / SAMPLE_FPS)))
    insects = tracks.link(tls, fps, step)
    keys = [video.read_frame(path, int(t * fps)) for t in np.arange(t_start, t_end, FLOWER_EVERY_S)]
    fls = flowers.find_flowers([k for k in keys if k is not None], flower_prompt)
    labels = classify_insects(path, insects, fps) if use_spark else {}
    for i in insects:  # a track through a sighting Spark already confirmed stays, whatever its close-ups show
        g = gate_confirmed(i, gated, fps) if i.id in labels and labels[i.id]["group"] == "not_insect" else None
        if g:
            labels[i.id] = {**labels[i.id], "group": g, "overruled": "confirmed at the scan gate"}
    kept = [i for i in insects if labels.get(i.id, {}).get("group") != "not_insect"]
    vis = flowers.visits(kept, fls, fps)
    log(f"{len(insects)} tracks, {len(insects) - len(kept)} rejected by Spark, {len(fls)} flowers, {len(vis)} visits ({time.time() - t0:.0f}s)")
    res = {"video": str(path), "fps": fps, "size": [w, h], "t_start": t_start, "t_end": t_end, "prompt": prompt,
           "sample_fps": SAMPLE_FPS, "seconds": round(time.time() - t0, 1), "windows": [[a, b, k] for a, b, k in wins],
           "scan": {f"{t:.1f}": b for t, b in hits.items()}, "gate": gate_report,
           "tracklets": len(tls), "static_dropped": sum(tracks.is_static(t, fps) for t in tls),
           "insects": [{**asdict(i), "label": labels.get(i.id)} for i in insects],
           "flowers": [asdict(f) for f in fls], "visits": [asdict(v) for v in vis]}
    if out:
        Path(out).write_text(json.dumps(res, default=_jsonable), encoding="utf-8")
    return res
