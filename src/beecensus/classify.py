"""Muse Spark Open as the survey's entomologist: which FIT Count group is this tracked object?

SAM 3 is told "bee" and finds anything bee-sized, so it also returns buds, unripe fruit
and leaf spots. Spark sees a few close-up crops of each track and either names the insect
group or rejects it. When it is unsure it asks for sharper crops of the same track.
"""

import json
import re

import cv2
import numpy as np

from . import muse

GROUPS = ["honeybee", "bumblebee", "solitary_bee", "wasp", "hoverfly", "other_fly", "butterfly_moth",
          "beetle", "small_insect", "other_insect", "not_insect"]

PROMPT = """You review footage from a pollinator survey camera looking down on a crop (here: strawberry plants).
The {n} image(s) are close-ups of ONE tracked object at different moments; the object is at the centre of each crop.
Insects are small and often motion-blurred, so judge body shape, colour bands, wing shape and size relative to the flower.

Classify it into exactly one group of the UK Flower-Insect Timed (FIT) Count:
honeybee, bumblebee, solitary_bee, wasp, hoverfly, other_fly, butterfly_moth, beetle, small_insect (<3 mm), other_insect,
or not_insect (flower bud, unripe fruit, petal, leaf spot, debris, shadow).

Answer with JSON only: {{"group": "<one group>", "confidence": <0..1>, "evidence": "<a few words on what you see>"}}"""


def crop(frame: np.ndarray, box, min_side: int = 96, out_side: int = 224) -> np.ndarray:
    """Square crop around a detection, 3x its size so the flower and leaves give scale."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(min_side, 3 * max(x1 - x0, y1 - y0))
    h, w = frame.shape[:2]
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    c = frame[b:min(h, b + int(s)), a:min(w, a + int(s))]
    return cv2.resize(c, (out_side, out_side), interpolation=cv2.INTER_CUBIC)


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    g = str(d.get("group", "")).strip().lower()
    return {"group": g if g in GROUPS else "other_insect" if g else "unknown",
            "confidence": float(d.get("confidence", 0) or 0), "evidence": str(d.get("evidence", ""))[:200], "raw": text[:400]}


def classify(crops: list[np.ndarray], tag: str = "classify") -> dict:
    parts = [{"type": "text", "text": PROMPT.format(n=len(crops))}] + [muse.image_part(c, max_side=224, quality=90) for c in crops]
    # Spark reasons before answering; a long deliberation can run out of tokens, so retry once
    for max_tokens in (3000, 5000):
        r = _parse(muse.chat([{"role": "user", "content": parts}], max_tokens=max_tokens, tag=tag))
        if r["group"] != "unknown":
            break
    return r
