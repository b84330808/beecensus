"""Run the full census on a video and save JSON (+ rendered mp4 with --render).

uv run python scripts/run_survey.py <video> <out.json> [t_start] [t_end] [--render] [--title "..."]
"""

import json
import sys
from pathlib import Path

from beecensus import pipeline, render, report

args = [a for a in sys.argv[1:] if not a.startswith("--")]
video, out = args[0], Path(args[1])
t0 = float(args[2]) if len(args) > 2 else 0.0
t1 = float(args[3]) if len(args) > 3 else None
res = pipeline.run(video, t0, t1, out=out, log=lambda s: print(s, flush=True))
if "--title" in sys.argv:
    res["title"] = sys.argv[sys.argv.index("--title") + 1]
try:
    res["summary"] = report.summary(res)
    print(res["summary"])
except Exception as e:
    print("summary failed:", e)
out.write_text(json.dumps(res, default=pipeline._jsonable), encoding="utf-8")
if "--render" in sys.argv:
    render.render(res, out.with_suffix(".mp4"))
    print("rendered", out.with_suffix(".mp4"))
