"""Recompute visits for saved runs from their stored insects and flowers (no GPU), e.g. after changing visit rules.

uv run python scripts/revisit.py out/T8.json [more.json ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from beecensus import flowers, pipeline, tracks

for p in sys.argv[1:]:
    res = json.loads(Path(p).read_text())
    kept = [tracks.Insect(i["id"], np.array(i["frame"]), np.array(i["cx"]), np.array(i["cy"]), np.array(i["area"]),
                          np.array(i["box"]), np.array(i["score"]), i["parts"])
            for i in res["insects"] if (i.get("label") or {}).get("group") != "not_insect"]
    fls = [flowers.Flower(**f) for f in res["flowers"]]
    vis = flowers.visits(kept, fls, res["fps"])
    print(f"{p}: visits {len(res['visits'])} -> {len(vis)}")
    res["visits"] = [vars(v) for v in vis]
    Path(p).write_text(json.dumps(res, default=pipeline._jsonable), encoding="utf-8")
