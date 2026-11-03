"""Compare a BeeCensus run with Polytrack's tracks for the same Monash test video.

Polytrack is a trained detector+tracker (honeybee F 0.95 against human observers), so it is
the closest thing to an answer key the public dataset has per video. Track files are named
PDDHHMMSS0I.csv: P = monitoring point (= camera number), DD = day, HHMMSS = track start,
I = 0 honeybee, 1 hoverfly, 2 moth/butterfly, 3 wasp.
Usage: uv run python scripts/eval_polytrack.py out/T4.json
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("BEECENSUS_DATA", Path(__file__).resolve().parents[2] / "data" / "monash"))
TRACKS = DATA / "traj" / "Extracted_Tracks" / "Insect_Tracks_CSV"
KIND = ["honeybee", "hoverfly", "butterfly_moth", "wasp"]
R = 80.0  # px: a Polytrack point and one of ours this close within 0.1 s are the same insect (a bee in flight covers ~60 px in 0.1 s)
MIN_HITS = 3
SOURCE = {"T1": "cam_4_N_video_20210309_132604", "T2": "cam_8_N_video_20210310_133844", "T3": "cam_7_S_video_20210310_142405",
          "T4": "cam_3_N_video_20210317_142705", "T5": "cam_1_S_video_20210317_123004", "T6": "cam_2_S_video_20210308_112402",
          "T7": "cam_3_S_video_20210312_123203", "T8": "cam_1_N_video_20210315_132804", "T9": "cam_4_N_video_20210317_114802",
          "T10": "cam_4_N_video_20210317_120802"}


def polytrack_tracks(res: dict) -> list[dict]:
    name = Path(res["video"]).stem.split("_head")[0]
    m = re.search(r"cam_(\d+)_\w_video_2021\d\d(\d\d)_(\d\d)(\d\d)(\d\d)", SOURCE.get(name, name))
    if not m:
        raise SystemExit("video must be T1..T10 or keep its original cam_N_X_video_YYYYMMDD_HHMMSS name")
    cam, dd = m.group(1), m.group(2)
    t0 = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
    out = []
    for f in sorted(TRACKS.glob(f"{cam}{dd}*.csv")):
        tt = f.name[len(cam) + 2:len(cam) + 8]
        sec = int(tt[:2]) * 3600 + int(tt[2:4]) * 60 + int(tt[4:6]) - t0
        if not (res["t_start"] <= sec < res["t_end"]):
            continue
        rows = list(csv.reader(f.open()))[1:]
        a0 = int(rows[0][2])
        p = np.array([(sec * res["fps"] + int(r[2]) - a0, float(r[3]), float(r[4])) for r in rows if r[3]])
        out.append({"file": f.name, "kind": KIND[int(f.name[-5])], "sec": sec, "pts": p,
                    "flowers": sorted({r[7] for r in rows if r[7]})})
    return out


def compare(res: dict) -> dict:
    """Which Polytrack insects we found, which of ours Polytrack lacks, with the best frame offset."""
    pt = polytrack_tracks(res)
    ours = [i for i in res["insects"] if (i.get("label") or {}).get("group") != "not_insect"]
    by_frame = {}
    for i in ours:
        for f, x, y in zip(i["frame"], i["cx"], i["cy"]):
            by_frame.setdefault(int(f), []).append((x, y, i["id"]))
    step = max(1, int(round(res["fps"] / res.get("sample_fps", 15))))

    def match(p, delta):
        ids, hit = set(), 0
        for f, x, y in p:
            g = int(round((f + delta) / step)) * step
            c = [q for k in range(g - 2 * step, g + 3 * step, step) for q in by_frame.get(k, []) if (q[0] - x) ** 2 + (q[1] - y) ** 2 < R * R]
            hit += bool(c)
            ids.update(q[2] for q in c)
        return hit, ids

    delta = max(range(-60, 61, step), key=lambda d: sum(match(t["pts"], d)[0] for t in pt)) if pt else 0
    one_s = int(round(res["fps"]))
    rows, matched = [], set()
    for t in pt:
        # file names give each track's start only to the second, so align each track within one second of the video offset
        hit, ids = max((match(t["pts"], d) for d in range(delta - one_s, delta + one_s + 1, step)), key=lambda m: m[0])
        matched |= ids
        rows.append({"kind": t["kind"], "sec": t["sec"], "points": len(t["pts"]), "hits": hit, "found": hit >= MIN_HITS,
                     "our_ids": sorted(ids), "flowers": len(t["flowers"])})
    extra = [i for i in ours if i["id"] not in matched]
    return {"offset": delta, "polytrack": rows, "ours": len(ours),
            "rejected": sum(1 for i in res["insects"] if (i.get("label") or {}).get("group") == "not_insect"),
            "extra": [{"id": i["id"], "t0": round(i["frame"][0] / res["fps"], 1), "t1": round(i["frame"][-1] / res["fps"], 1),
                       "n": len(i["frame"]), **{k: (i.get("label") or {}).get(k) for k in ("group", "confidence", "evidence")}} for i in extra]}


def main(path: str):
    res = json.loads(Path(path).read_text())
    c = compare(res)
    print(f"{Path(res['video']).name}  {res['t_start']:.0f}-{res['t_end']:.0f}s  offset {c['offset']} frames")
    for r in c["polytrack"]:
        print(f"  Polytrack {r['kind']:14s} t={r['sec']:4d}s  {'FOUND' if r['found'] else 'missed':6s} {r['hits']}/{r['points']} pts  our ids {r['our_ids']}  flowers {r['flowers']}")
    print(f"Polytrack insects found: {sum(r['found'] for r in c['polytrack'])}/{len(c['polytrack'])}")
    print(f"our insects: {c['ours']} (+{c['rejected']} rejected by Spark); not in Polytrack: {len(c['extra'])}")
    for e in c["extra"]:
        print(f"  extra id {e['id']} t={e['t0']}-{e['t1']}s n={e['n']} {e['group']} {e['confidence']} {(e['evidence'] or '')[:60]}")
    print(f"visits: ours {len(res['visits'])}, Polytrack flowers-touched {sum(r['flowers'] for r in c['polytrack'])}")


if __name__ == "__main__":
    main(sys.argv[1])
