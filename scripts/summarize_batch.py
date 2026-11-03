"""Roll up every out/T*.json against Polytrack: markdown tables for the README and numbers for the demo.

T2, T3, T9 and T10 are held out: nothing was tuned on them, so they are reported apart from the six
videos the pipeline was developed on.
uv run python scripts/summarize_batch.py [--out DIR] [--demo ../demo_bee/remotion/public/results.json]
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_polytrack import compare  # noqa: E402

OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(__file__).resolve().parents[1] / "out"
HELD_OUT = {"T2", "T3", "T9", "T10"}
HEAD = ["| Video | Honeybees found (Polytrack) | Other insects found (Polytrack) | Our insects | Ours not in Polytrack | Spark gate rejected | Tracked | Visits | Flowers visited |",
        "|---|---|---|---|---|---|---|---|---|"]
KEYS = ["bee_found", "bee_total", "oth_found", "oth_total", "ours", "extra", "gate_rej", "gate_tot", "track_s", "span_s", "videos"]


def table(runs: list[Path], label: str) -> tuple[list[str], dict]:
    lines, tot = list(HEAD), dict.fromkeys(KEYS, 0)
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
                     f"({', '.join(r['kind'] for r in oth) or '-'}) | {c['ours']} | {len(c['extra'])} | {rej}/{len(gate)} | {track_s:.0f}s | "
                     f"{len(res['visits'])} | {visited}/{len(res['flowers'])} |")
        for k, v in [("bee_found", sum(r["found"] for r in bees)), ("bee_total", len(bees)), ("oth_found", sum(r["found"] for r in oth)),
                     ("oth_total", len(oth)), ("ours", c["ours"]), ("extra", len(c["extra"])), ("gate_rej", rej), ("gate_tot", len(gate)),
                     ("track_s", track_s), ("span_s", res["t_end"] - res["t_start"]), ("videos", 1)]:
            tot[k] += v
        for e in c["extra"]:
            print(f"  {p.stem} extra id {e['id']} {e['t0']}-{e['t1']}s {e['group']} {e['confidence']}: {(e['evidence'] or '')[:70]}")
    lines.append(f"| **{label}** | **{tot['bee_found']}/{tot['bee_total']}** | {tot['oth_found']}/{tot['oth_total']} | {tot['ours']} | {tot['extra']} | "
                 f"{tot['gate_rej']}/{tot['gate_tot']} | {tot['track_s']:.0f}s of {tot['span_s']:.0f}s | | |")
    return lines, tot


runs = sorted((p for p in OUT.glob("T*.json") if re.fullmatch(r"T\d+", p.stem)), key=lambda p: int(p.stem[1:]))
tuned = [p for p in runs if p.stem not in HELD_OUT]
held = [p for p in runs if p.stem in HELD_OUT]
lines, tt = table(tuned, f"Development, {len(tuned)} videos")
print("\n".join(lines))
hl, th = table(held, f"Held out, {len(held)} videos")
if held:
    print("\n" + "\n".join(hl))
if "--demo" in sys.argv:
    d = Path(sys.argv[sys.argv.index("--demo") + 1])
    r = json.loads(d.read_text())
    al = {k: tt[k] + th[k] for k in KEYS}
    r.update({"poly_found": al["bee_found"], "poly_total": al["bee_total"], "videos": al["videos"],
              "held_found": th["bee_found"], "held_total": th["bee_total"], "held_videos": th["videos"],
              "gate_rejected": al["gate_rej"], "gate_total": al["gate_tot"],
              "track_share": round(100 * al["track_s"] / max(1, al["span_s"])),
              "ours": al["ours"], "extra": al["extra"]})
    r.pop("minutes_per_survey", None)  # GPU stages ran locally for some videos and on Modal for others
    d.write_text(json.dumps(r, indent=1))
    print("demo numbers ->", d)
