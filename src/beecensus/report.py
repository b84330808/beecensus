"""Survey outputs: FIT Count table, pollination coverage map, evidence gallery, written summary."""

import csv
import io
from collections import Counter

import cv2
import numpy as np

from . import classify, muse, pipeline, tracks, video
from .render import NICE, group_of

FIT_ORDER = ["bumblebee", "honeybee", "solitary_bee", "wasp", "hoverfly", "other_fly", "butterfly_moth",
             "beetle", "small_insect", "other_insect", "unknown"]


def insects_kept(res: dict) -> list[dict]:
    return [i for i in res["insects"] if group_of(i) != "not_insect"]


def fit_table(res: dict) -> list[list]:
    """Rows: group, insects seen, flower visits (FIT Count records insects on flowers; we add all sightings)."""
    ins = insects_kept(res)
    g = {i["id"]: group_of(i) for i in ins}
    seen = Counter(g.values())
    vis = Counter(g[v["insect"]] for v in res["visits"] if v["insect"] in g)
    rows = [[NICE.get(k, k), seen.get(k, 0), vis.get(k, 0)] for k in FIT_ORDER if seen.get(k) or vis.get(k)]
    rows.append(["Total", sum(seen.values()), sum(vis.values())])
    return rows


def stats(res: dict) -> dict:
    fl = res["flowers"]
    visited = {v["flower"] for v in res["visits"]}
    minutes = (res["t_end"] - res["t_start"]) / 60
    return {"minutes": round(minutes, 1), "insects": len(insects_kept(res)),
            "rejected": len(res["insects"]) - len(insects_kept(res)), "visits": len(res["visits"]),
            "flowers": len(fl), "flowers_visited": len(visited),
            "coverage": round(len(visited) / len(fl), 2) if fl else 0.0,
            "visits_per_flower_hour": round(len(res["visits"]) / max(1, len(fl)) / max(minutes / 60, 1e-6), 2) if fl else 0.0}


def coverage_map(res: dict, width: int = 1280) -> np.ndarray:
    """A keyframe with every flower ringed by how often it was visited: red = none."""
    fps = res["fps"]
    img = video.read_frame(res["video"], int(res["t_start"] * fps))
    W, H = res["size"]
    s = width / W
    img = cv2.resize(img, (width, int(H * s)), interpolation=cv2.INTER_AREA)
    n = Counter(v["flower"] for v in res["visits"])
    for fl in res["flowers"]:
        k = n.get(fl["id"], 0)
        c = (230, 60, 60) if k == 0 else (250, 200, 40) if k == 1 else (60, 220, 90)  # RGB
        p = (int(fl["x"] * s), int(fl["y"] * s))
        cv2.circle(img, p, int(fl["r"] * s * 1.3), c, 4, cv2.LINE_AA)
        cv2.putText(img, str(k), (p[0] - 7, p[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, str(k), (p[0] - 7, p[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
    return img


def gallery(res: dict, limit: int = 24) -> list[tuple[np.ndarray, str]]:
    """One close-up per insect (Spark-rejected ones too, marked) with Spark's verdict."""
    out = []
    for i in res["insects"][:limit]:
        ins = tracks.Insect(i["id"], np.array(i["frame"]), np.array(i["cx"]), np.array(i["cy"]), np.array(i["area"]),
                            np.array(i["box"]), np.array(i["score"]), i["parts"])
        k = pipeline.best_frames(ins, res["fps"], 1)[0]  # landed and sharp, not the motion-blurred flight
        fr = video.read_frame(res["video"], int(i["frame"][k]))
        if fr is None:
            continue
        lab = i.get("label") or {}
        g = group_of(i)
        t = i["frame"][0] / res["fps"]
        cap = (f"#{i['id']} {t:.0f}s  {'REJECTED: ' if g == 'not_insect' else ''}{NICE.get(g, g)} "
               f"({lab.get('confidence', 0):.2f}) - {lab.get('evidence', '')}")
        out.append((classify.crop(fr, i["box"][k]), cap))
    return out


def fit_csv(res: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["insect_id", "group", "confidence", "first_seen_s", "last_seen_s", "flower_visits", "spark_evidence"])
    nv = Counter(v["insect"] for v in res["visits"])
    for i in res["insects"]:
        lab = i.get("label") or {}
        w.writerow([i["id"], group_of(i), lab.get("confidence"), round(i["frame"][0] / res["fps"], 1),
                    round(i["frame"][-1] / res["fps"], 1), nv.get(i["id"], 0), lab.get("evidence", "")])
    return buf.getvalue()


def summary(res: dict) -> str:
    """Muse Spark Open turns the numbers into a short note for the grower / survey coordinator."""
    st = stats(res)
    table = "\n".join(f"{r[0]}: {r[1]} insects, {r[2]} flower visits" for r in fit_table(res))
    prompt = f"""You are the field ecologist on a pollination survey. An automated camera survey of {st['minutes']} minutes over
{st['flowers']} open flowers produced:
{table}
Flowers visited at least once: {st['flowers_visited']} of {st['flowers']} ({st['coverage']:.0%}).
Objects the vision model flagged but you rejected as not insects: {st['rejected']}.
Write 3 short sentences for the grower: what pollinated the crop, how well-covered the flowers were, and one
practical next step. Plain words, no headings, no hype; say clearly when numbers are too small to conclude much."""
    return muse.chat([{"role": "user", "content": prompt}], max_tokens=3000, tag="summary").strip()
