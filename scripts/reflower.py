"""Recompute flowers and visits for saved runs (e.g. after changing the flower prompt), keeping the insects.

uv run python scripts/reflower.py out/T4.json [more.json ...] [--prompt "white flower"]
"""

import json
import sys
from pathlib import Path

import numpy as np

from beecensus import flowers, pipeline, tracks, video

prompt = sys.argv[sys.argv.index("--prompt") + 1] if "--prompt" in sys.argv else "white flower"
for p in [a for a in sys.argv[1:] if a.endswith(".json")]:
    res = json.loads(Path(p).read_text())
    fps = res["fps"]
    keys = [video.read_frame(res["video"], int(t * fps)) for t in np.arange(res["t_start"], res["t_end"], pipeline.FLOWER_EVERY_S)]
    fls = flowers.find_flowers([k for k in keys if k is not None], prompt)
    kept = [tracks.Insect(i["id"], np.array(i["frame"]), np.array(i["cx"]), np.array(i["cy"]), np.array(i["area"]),
                          np.array(i["box"]), np.array(i["score"]), i["parts"])
            for i in res["insects"] if (i.get("label") or {}).get("group") != "not_insect"]
    vis = flowers.visits(kept, fls, fps)
    print(f"{p}: flowers {len(res['flowers'])} -> {len(fls)}, visits {len(res['visits'])} -> {len(vis)}")
    res["flowers"] = [vars(f) for f in fls]
    res["visits"] = [vars(v) for v in vis]
    res["flower_prompt"] = prompt
    Path(p).write_text(json.dumps(res, default=pipeline._jsonable), encoding="utf-8")
