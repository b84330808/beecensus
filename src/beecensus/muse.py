"""Muse Spark Open (Meta open weights) through Baseten's OpenAI-compatible API.

The hackathon key has $100 of credit that has to last through the demo, so every call
is logged to data/muse_usage.jsonl and calls stop once the logged cost passes
MUSE_BUDGET_USD (default $60, leaving the rest for the demo itself).
"""

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://inference.baseten.co/v1"
MODEL = "meta-models/muse-spark-open-1026"  # the only model the hackathon key accepts
USAGE_LOG = ROOT.parent / "data" / "muse_usage.jsonl"  # shared with gravitycheck: one budget for the key
BUDGET_USD = float(os.environ.get("MUSE_BUDGET_USD", 60))
USD_PER_M_IN, USD_PER_M_OUT = 1.25, 4.25  # Baseten list price for Muse Spark Open


def _key() -> str:
    if os.environ.get("BASETEN_API_KEY"):
        return os.environ["BASETEN_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("BASETEN_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("BASETEN_API_KEY missing: put it in beecensus/.env")


def spent_usd() -> float:
    """Cost of every logged call so far, at list price (cached input billed as full)."""
    if not USAGE_LOG.exists():
        return 0.0
    rows = [json.loads(l) for l in USAGE_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sum(r["prompt_tokens"] * USD_PER_M_IN + r["completion_tokens"] * USD_PER_M_OUT for r in rows) / 1e6


def image_part(frame: np.ndarray, max_side: int = 768, quality: int = 80) -> dict:
    """An RGB frame as a chat image part; small JPEGs keep the token count down."""
    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}}


def chat(messages: list[dict], max_tokens: int = 2000, tag: str = "", **kw) -> str:
    """One chat completion; returns the answer text (reasoning, if any, is not returned)."""
    from openai import OpenAI

    spent = spent_usd()
    if spent > BUDGET_USD:
        raise RuntimeError(f"Muse budget reached (${spent:.2f} > ${BUDGET_USD:.0f}); raise MUSE_BUDGET_USD to go on")
    client = OpenAI(base_url=BASE_URL, api_key=_key())
    t = time.time()
    r = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens, **kw)
    u = r.usage
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": round(t), "tag": tag, "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens,
                            "seconds": round(time.time() - t, 1)}) + "\n")
    return r.choices[0].message.content or ""
