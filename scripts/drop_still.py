"""Apply tracks.drop_still_spots to saved runs (no GPU), then recompute visits and the written summary.

Runs made before the rule existed report a still thing once per tracking window (T9: one pale-yellow
object, 11 times). uv run python scripts/drop_still.py out/T9.json [more.json ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from beecensus import flowers, pipeline, report, tracks

for p in sys.argv[1:]:
    res = json.loads(Path(p).read_text())
    ins = [tracks.Insect(i["id"], np.array(i["frame"]), np.array(i["cx"]), np.array(i["cy"]), np.array(i["area"]),
                         np.array(i["box"]), np.array(i["score"]), i["parts"], i.get("sam", True)) for i in res["insects"]]
    keep = {i.id for i in tracks.drop_still_spots(ins, res["fps"])}
    gone = [i["id"] for i in res["insects"] if i["id"] not in keep]
    if not gone:
        print(f"{p}: nothing still")
        continue
    insect = {i.id: i for i in ins}
    res["insects"] = [i for i in res["insects"] if i["id"] in keep]
    kept = [insect[i["id"]] for i in res["insects"] if (i.get("label") or {}).get("group") != "not_insect"]
    vis = flowers.visits(kept, [flowers.Flower(**f) for f in res["flowers"]], res["fps"])
    print(f"{p}: dropped still ids {gone}; visits {len(res['visits'])} -> {len(vis)}")
    res["visits"] = [vars(v) for v in vis]
    if "summary" in res:
        try:
            res["summary"] = report.summary(res)
        except Exception as e:
            print("summary failed:", e)
    Path(p).write_text(json.dumps(res, default=pipeline._jsonable), encoding="utf-8")
