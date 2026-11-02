"""Roll up every out/T*.json against Polytrack: a markdown table for the README and numbers for the demo.

uv run python scripts/summarize_batch.py [--demo ../demo_bee/remotion/public/results.json]
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_polytrack import compare  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
runs = sorted((p for p in OUT.glob("T*.json") if re.fullmatch(r"T\d+", p.stem)), key=lambda p: int(p.stem[1:]))
lines = ["| Video | Honeybees found (Polytrack) | Other insects found (Polytrack) | Ours not in Polytrack | Spark gate rejected | Tracked | Visits | Flowers visited | Run time |",
         "|---|---|---|---|---|---|---|---|---|"]
tot = {"bee_found": 0, "bee_total": 0, "oth_found": 0, "oth_total": 0, "extra": 0, "gate_rej": 0, "gate_tot": 0,
       "track_s": 0.0, "span_s": 0.0, "secs": 0.0, "videos": 0}
for p in runs:
    res = json.loads(p.read_text())
    c = compare(res)
    bees = [r for r in c["polytrack"] if r["kind"] == "honeybee"]
    oth = [r for r in c["polytrack"] if r["kind"] != "honeybee"]
    gate = res.get("gate", [])
    rej = sum(not g["kept"] for g in gate)
    track_s = sum(b - a for a, b, _ in res.get("windows", []))
    visited = len({v["flower"] for v in res["visits"]})
    lines.append(f"| {p.stem} | {sum(r['found'] for r in bees)}/{len(bees)} | {sum(r['found'] for r in oth)}/{len(oth)} "
                 f"({', '.join(r['kind'] for r in oth) or '-'}) | {len(c['extra'])} | {rej}/{len(gate)} | {track_s:.0f}s | "
                 f"{len(res['visits'])} | {visited}/{len(res['flowers'])} | {res['seconds'] / 60:.0f} min |")
    tot["bee_found"] += sum(r["found"] for r in bees)
    tot["bee_total"] += len(bees)
    tot["oth_found"] += sum(r["found"] for r in oth)
    tot["oth_total"] += len(oth)
    tot["extra"] += len(c["extra"])
    tot["gate_rej"] += rej
    tot["gate_tot"] += len(gate)
    tot["track_s"] += track_s
    tot["span_s"] += res["t_end"] - res["t_start"]
    tot["secs"] += res["seconds"]
    tot["videos"] += 1
    for e in c["extra"]:
        print(f"  {p.stem} extra id {e['id']} {e['t0']}-{e['t1']}s {e['group']} {e['confidence']}: {(e['evidence'] or '')[:70]}")
lines.append(f"| **All {tot['videos']}** | **{tot['bee_found']}/{tot['bee_total']}** | {tot['oth_found']}/{tot['oth_total']} | {tot['extra']} | "
             f"{tot['gate_rej']}/{tot['gate_tot']} | {tot['track_s']:.0f}s of {tot['span_s']:.0f}s | | | {tot['secs'] / 60 / max(1, tot['videos']):.0f} min avg |")
print("\n".join(lines))
if "--demo" in sys.argv:
    d = Path(sys.argv[sys.argv.index("--demo") + 1])
    r = json.loads(d.read_text())
    r.update({"poly_found": tot["bee_found"], "poly_total": tot["bee_total"], "videos": tot["videos"],
              "gate_rejected": tot["gate_rej"], "gate_total": tot["gate_tot"],
              "track_share": round(100 * tot["track_s"] / max(1, tot["span_s"])),
              "minutes_per_survey": round(tot["secs"] / 60 / max(1, tot["videos"])), "extra": tot["extra"]})
    d.write_text(json.dumps(r, indent=1))
    print("demo numbers ->", d)
