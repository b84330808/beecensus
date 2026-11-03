"""Video in, insects + flowers + visits out, cached as JSON next to the run.

1. scan: SAM 3 image pass once a second finds where anything bee-like appears, and frame
   differencing three times a second finds what moves (bees in fast flight are too blurred for SAM 3).
   Muse Spark Open looks at each candidate and drops the ones that are not insects.
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

from . import classify, detect, flowers, motion, tracks, video

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
    fps = video.info(path)[0]
    part = key.with_name(key.stem + "_part.json")  # seconds scanned so far, so a crash mid-scan costs a minute, not the scan
    done = {float(t): b for t, b in _load(part).items()}
    for i, f in video.iter_frames(path, t_start, t_end, SCAN_FPS):  # one frame at a time: 10 min of 1080p is ~4 GB
        t = i / fps
        hits[t] = done[t] if t in done else detect.detect_boxes(f, prompt)
        for b, _ in hits[t]:
            crops[(t, tuple(round(v) for v in b))] = classify.crop(f, b)
        if t not in done and len(hits) % 60 == 0:
            _save(part, {str(t): b for t, b in hits.items()})
    CACHE.mkdir(exist_ok=True)
    key.write_text(json.dumps({str(t): b for t, b in hits.items()}))
    part.unlink(missing_ok=True)
    return hits, crops


def _load(key: Path | None) -> dict:
    return json.loads(key.read_text()) if key is not None and key.exists() else {}


def _save(key: Path | None, d: dict):
    """Write a resume cache atomically (a crash mid-write must not lose what was already done)."""
    if key is not None:
        CACHE.mkdir(exist_ok=True)
        tmp = key.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, default=_jsonable))
        tmp.replace(key)


def gate(moving: list, crops: dict, log=print, cache: Path | None = None) -> tuple[list, list[dict]]:
    """Muse Spark Open decides which moving scan hits are insects before any GPU time is spent tracking them.

    Hits close in time and space are one candidate visitor; Spark sees up to three of its close-ups.
    Wind-blown leaves and glare that slip past the static filter are dropped here. Verdicts are
    cached per candidate, so a rerun after a crash asks Spark only about the new ones.
    """
    groups = detect.cluster_hits(moving)
    done = _load(cache)
    names = [json.dumps(sorted([round(moving[i][0], 2), [round(v) for v in moving[i][1]]] for i in g)) for g in groups]
    jobs = []
    for g, name in zip(groups, names):
        if name not in done:
            best = sorted(g, key=lambda i: -moving[i][2])[:CROPS_PER_INSECT]
            jobs.append((name, [crops[(moving[i][0], tuple(round(v) for v in moving[i][1]))] for i in best]))
    with ThreadPoolExecutor(4) as ex:
        for name, v in ex.map(lambda j: (j[0], classify.classify(j[1], tag="gate")), jobs):
            done[name] = v
    _save(cache, done)
    verdicts = [done[n] for n in names]
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


def gate_confirmed(ins: tracks.Insect | detect.Tracklet, gated: list, fps: float, px: float = 80.0, s: float = 1.0) -> str | None:
    """The group Spark gave a confirmed scan hit that lies on this track, if any."""
    frame, cx, cy = np.asarray(ins.frame), np.asarray(ins.cx), np.asarray(ins.cy)
    for t, b, _, group in gated:
        f = t * fps
        k = np.argmin(np.abs(frame - f))
        if abs(frame[k] - f) <= s * fps and np.hypot(cx[k] - (b[0] + b[2]) / 2, cy[k] - (b[1] + b[3]) / 2) <= px:
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


def cache_key(path, t_start: float, t_end: float, prompt: str, what: str) -> Path:
    return CACHE / f"{Path(path).stem}_{t_start:.0f}_{t_end:.0f}_{prompt.replace(' ', '-')}_{what}.json"


def plan(path, t_start: float, t_end: float, prompt: str = "bee", use_spark: bool = True, log=print):
    """Scan, motion scan and Spark gate: (scan hits, moving blobs, gated candidates, gate report, windows to track)."""
    t0 = time.time()
    w, h = video.info(path)[2:]
    hits, crops = scan(path, t_start, t_end, prompt)
    moving = detect.moving_hits(hits)
    log(f"scan {t_start:.0f}-{t_end:.0f}s: {sum(map(len, hits.values()))} hits, {len(moving)} moving ({time.time() - t0:.0f}s)")
    mhits, mcrops = motion.scan(path, t_start, t_end)
    mmoving = detect.moving_hits(mhits)
    log(f"motion scan: {sum(map(len, mhits.values()))} moving blobs, {len(mmoving)} not fixed ({time.time() - t0:.0f}s)")
    moving, crops = moving + mmoving, {**crops, **mcrops}
    gated, gate_report = (gate(moving, crops, log, cache_key(path, t_start, t_end, prompt, "gate")) if use_spark
                          else ([(*m, "unknown") for m in moving], []))
    wins = detect.windows(gated, (w, h))
    log(f"{len(wins)} windows "
        f"({sum(b - a for a, b, _ in wins):.0f}s, {sum((b - a) * len(k) for a, b, k in wins):.0f} tile-seconds to track) ({time.time() - t0:.0f}s)")
    return hits, mhits, gated, gate_report, wins


def chunks(a: float, b: float, near: list[int]):
    """(cache name, start, end) of the CHUNK_S pieces a window is tracked in."""
    for c in np.arange(a, b, CHUNK_S):
        yield f"{c:.2f}-{min(c + CHUNK_S, b):.2f}:{near}", c, min(c + CHUNK_S, b)


def track_chunk(path, c: float, e: float, near: list[int], prompt: str = "bee") -> dict:
    """SAM 3 tracklets and moving-blob tracklets of one chunk, as stored in the track cache."""
    win = video.read_window(path, c, e, SAMPLE_FPS)
    return {"sam": [asdict(t) for t in detect.track(win, prompt, only=near)] if win.frames else [],
            "motion": [asdict(t) for t in motion.tracklets(win)] if win.frames else []}


def find_flowers(path, t_start: float, t_end: float, prompt: str = "white flower") -> list[flowers.Flower]:
    """Flowers from a keyframe every FLOWER_EVERY_S, cached like the scan."""
    key = cache_key(path, t_start, t_end, prompt, "flowers")
    if key.exists():
        return [flowers.Flower(**f) for f in json.loads(key.read_text())]
    fps = video.info(path)[0]
    keys = [video.read_frame(path, int(t * fps)) for t in np.arange(t_start, t_end, FLOWER_EVERY_S)]
    fls = flowers.find_flowers([k for k in keys if k is not None], prompt)
    _save(key, [asdict(f) for f in fls])
    return fls


def run(path: str | Path, t_start: float = 0.0, t_end: float | None = None, prompt: str = "bee",
        flower_prompt: str = "white flower", out: str | Path | None = None, use_spark: bool = True, log=print) -> dict:
    fps, n, w, h = video.info(path)
    if t_end is None:
        t_end = n / fps if n else 600.0
    t0 = time.time()
    hits, mhits, gated, gate_report, wins = plan(path, t_start, t_end, prompt, use_spark, log)
    tls = []
    track_cache = cache_key(path, t_start, t_end, prompt, f"tracks_{SAMPLE_FPS}fps")  # per chunk, so a crash costs one chunk, not the survey
    done = _load(track_cache)
    for a, b, near in wins:
        for name, c, e in chunks(a, b, near):
            if name not in done:
                done[name] = track_chunk(path, c, e, near, prompt)
                _save(track_cache, done)
            tls += [detect.Tracklet(**t) for t in done[name]["sam"]]
            # moving blobs fill in where SAM 3 loses a bee in fast flight; only along visitors Spark confirmed
            tls += [t for t in map(lambda d: detect.Tracklet(**d), done[name]["motion"])
                    if len(t.frame) >= tracks.MIN_DETS and gate_confirmed(t, gated, fps)]
        log(f"tracked {a:.0f}-{b:.0f}s: {len(tls)} tracklets ({time.time() - t0:.0f}s)")
    step = max(1, int(round(fps / SAMPLE_FPS)))
    insects = tracks.link(tls, fps, step)
    fls = find_flowers(path, t_start, t_end, flower_prompt)
    labels = classify_insects(path, insects, fps) if use_spark else {}
    for i in insects:  # a SAM 3 track through a sighting Spark already confirmed stays, whatever its close-ups show
        # (blobs alone are not enough: a bee shakes the leaves around it, and those blobs pass the same sighting)
        g = gate_confirmed(i, gated, fps) if i.sam and i.id in labels and labels[i.id]["group"] == "not_insect" else None
        if g:
            labels[i.id] = {**labels[i.id], "group": g, "overruled": "confirmed at the scan gate"}
    kept = [i for i in insects if labels.get(i.id, {}).get("group") != "not_insect"]
    vis = flowers.visits(kept, fls, fps)
    log(f"{len(insects)} tracks, {len(insects) - len(kept)} rejected by Spark, {len(fls)} flowers, {len(vis)} visits ({time.time() - t0:.0f}s)")
    res = {"video": str(path), "fps": fps, "size": [w, h], "t_start": t_start, "t_end": t_end, "prompt": prompt,
           "sample_fps": SAMPLE_FPS, "seconds": round(time.time() - t0, 1), "windows": [[a, b, k] for a, b, k in wins],
           "scan": {f"{t:.1f}": b for t, b in hits.items()}, "motion_blobs": sum(map(len, mhits.values())), "gate": gate_report,
           "tracklets": len(tls), "static_dropped": sum(tracks.is_static(t, fps) for t in tls),
           "insects": [{**asdict(i), "label": labels.get(i.id)} for i in insects],
           "flowers": [asdict(f) for f in fls], "visits": [asdict(v) for v in vis]}
    if out:
        Path(out).write_text(json.dumps(res, default=_jsonable), encoding="utf-8")
    return res
