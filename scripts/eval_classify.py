"""Muse Spark Open as a filter + classifier on human-labelled insects and SAM 3 false positives.

Positives: every insect box in the Monash YOLO test images we hold (class 0 Hymenoptera =
honeybees + wasps, class 2 Syrphidae = hoverflies). Negatives: SAM 3 "bee" detections at
score >= 0.3 that match no labelled insect (buds, fruit, leaf spots).
Usage: uv run python scripts/eval_classify.py [n_negatives]
"""

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from beecensus import classify

DATA = Path(os.environ.get("BEECENSUS_DATA", Path(__file__).resolve().parents[2] / "data" / "monash"))
OUT = DATA / "spark_classify_eval.jsonl"
n_neg = int(sys.argv[1]) if len(sys.argv) > 1 else 80


def hit(g, b, pad=10):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return g[0] - pad <= cx <= g[2] + pad and g[1] - pad <= cy <= g[3] + pad


det = json.loads((DATA / "sam3_det_tile4.json").read_text())
items, negs = [], []
for r in det:
    ins = [(n, g) for n, g in r["gt"] if n != "Flower"]
    items += [(r["file"], n, g) for n, g in ins]
    negs += [(r["file"], "not_insect", b) for b, s in r["tile4:bee"] if s >= 0.3 and not any(hit(g, b) for _, g in ins)]
random.seed(0)
items += random.sample(negs, min(n_neg, len(negs)))
done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()


def one(it):
    f, label, box = it
    key = f"{f}:{[round(v) for v in box]}"
    if key in done:
        return None
    img = np.array(Image.open(DATA / "yolo_testing" / f).convert("RGB"))
    c = classify.classify([classify.crop(img, box)], tag="cls_eval")
    return {"key": key, "label": label, **{k: c[k] for k in ("group", "confidence", "evidence")}}


with ThreadPoolExecutor(4) as ex, OUT.open("a", encoding="utf-8") as fo:
    for r in ex.map(one, items):
        if r:
            fo.write(json.dumps(r) + "\n")
            fo.flush()

rows = [json.loads(l) for l in OUT.read_text().splitlines()]
ok = {"Hymenoptera": {"honeybee", "bumblebee", "solitary_bee", "wasp"}, "Syrphidae": {"hoverfly"}, "not_insect": {"not_insect"}}
for lab in ok:
    rs = [r for r in rows if r["label"] == lab]
    if not rs:
        continue
    right = sum(r["group"] in ok[lab] for r in rs)
    kept = sum(r["group"] != "not_insect" for r in rs)
    groups = {}
    for r in rs:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    print(f"{lab:12s} n={len(rs):3d} correct group {right}/{len(rs)} ({right/len(rs):.0%})  kept as insect {kept}/{len(rs)}  {groups}")
