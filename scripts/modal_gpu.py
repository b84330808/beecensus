"""Run the SAM 3 stages of a census on Modal GPUs, for when the local GPU cannot be trusted.

Only SAM 3 goes to the cloud: the once-a-second scan, the 15 fps tracking chunks and the flower keyframes.
Muse Spark Open stays on this machine (its key and budget log never leave it), so the Spark gate runs here
between the scan and the tracking. Results land in the same cache/ files a local run writes; then
`scripts/run_survey.py` finishes each census here without touching the GPU.

Needs the SAM 3 weights and the videos in the Modal volume "beecensus-data" (/sam3, /videos/<name>.avi):
  modal volume put beecensus-data <huggingface snapshot of facebook/sam3> /sam3
  modal volume put beecensus-data ../data/monash/videos/T2.avi /videos/T2.avi
uv run --with modal modal run scripts/modal_gpu.py --videos T2,T10
"""

import json
from dataclasses import asdict
from pathlib import Path

import modal

GPU = "L40S"
T0, T1, PROMPT, FLOWER_PROMPT = 0.0, 600.0, "bee", "white flower"  # as data/monash/run_batch.cmd runs them
SCAN_PARTS = 10
VIDEOS = Path(__file__).resolve().parent.parent.parent / "data" / "monash" / "videos"  # local only

app = modal.App("beecensus")
vol = modal.Volume.from_name("beecensus-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("transformers==5.22.1", "accelerate==1.15.0", "numpy==2.4.6", "scipy==1.17.1",
                      "opencv-python-headless==5.0.0.93", "pillow==12.3.0")
         .env({"HF_HUB_OFFLINE": "1"})
         .add_local_python_source("beecensus"))
gpu_fn = app.function(gpu=GPU, image=image, volumes={"/data": vol}, timeout=3600, max_containers=10)


def _remote(name: str) -> str:
    from beecensus import detect, pipeline

    detect.MODEL_ID = "/data/sam3"
    pipeline.CACHE = Path("/tmp/cache")
    path = f"/data/videos/{name}.avi"
    if not Path(path).exists():  # an unreadable video would otherwise come back as empty chunks, cached as if done
        raise FileNotFoundError(f"{path} is not in the volume")
    return path


@gpu_fn
def scan_part(name: str, t0: float, t1: float) -> tuple[str, dict]:
    from beecensus import detect, pipeline, video

    path = _remote(name)
    fps = video.info(path)[0]
    return name, {str(i / fps): detect.detect_boxes(f, PROMPT) for i, f in video.iter_frames(path, t0, t1, pipeline.SCAN_FPS)}


@gpu_fn
def track(name: str, chunk: str, c: float, e: float, near: list[int]) -> tuple[str, str, dict]:
    from beecensus import pipeline

    return name, chunk, pipeline.track_chunk(_remote(name), c, e, near, PROMPT)


@gpu_fn
def find_flowers(name: str) -> tuple[str, list[dict]]:
    from beecensus import pipeline

    return name, [asdict(f) for f in pipeline.find_flowers(_remote(name), T0, T1, FLOWER_PROMPT)]


@app.local_entrypoint()
def check(name: str = "T8", start: float = 0.0, seconds: float = 10.0):
    """Scan the first seconds of a video in the cloud and compare with the local scan cache."""
    from beecensus import pipeline

    mine = json.loads((pipeline.CACHE / f"{name}_{T0:.0f}_{T1:.0f}_{PROMPT}_{pipeline.SCAN_FPS}fps.json").read_text())
    _, cloud = scan_part.remote(name, start, start + seconds)
    for t, boxes in cloud.items():
        print(t, "cloud", [[round(v) for v in b] + [round(s, 2)] for b, s in boxes],
              "local", [[round(v) for v in b] + [round(s, 2)] for b, s in mine[t]])


@app.local_entrypoint()
def main(videos: str):
    from beecensus import pipeline

    names = videos.split(",")
    local = {n: VIDEOS / f"{n}.avi" for n in names}
    scan_key = {n: pipeline.CACHE / f"{n}_{T0:.0f}_{T1:.0f}_{PROMPT}_{pipeline.SCAN_FPS}fps.json" for n in names}
    todo = [(n, k * T1 / SCAN_PARTS, (k + 1) * T1 / SCAN_PARTS) for n in names if not scan_key[n].exists() for k in range(SCAN_PARTS)]
    hits: dict[str, dict] = {}
    parts: dict[str, int] = {}
    for n, h in scan_part.starmap(todo, order_outputs=False) if todo else []:  # an empty map never returns
        hits.setdefault(n, {}).update(h)
        parts[n] = parts.get(n, 0) + 1
        if parts[n] == SCAN_PARTS:
            pipeline._save(scan_key[n], dict(sorted(hits[n].items(), key=lambda kv: float(kv[0]))))
            print(f"{n}: scan done", flush=True)
    jobs, done = [], {}
    for n in names:  # Spark gate, here
        print(f"== {n}", flush=True)
        _, _, _, _, wins = pipeline.plan(local[n], T0, T1, PROMPT, True, lambda s: print(s, flush=True))
        done[n] = pipeline._load(pipeline.cache_key(local[n], T0, T1, PROMPT, f"tracks_{pipeline.SAMPLE_FPS}fps"))
        jobs += [(n, name, c, e, near) for a, b, near in wins for name, c, e in pipeline.chunks(a, b, near) if name not in done[n]]
    print(f"{len(jobs)} chunks to track on {GPU}", flush=True)
    for k, (n, name, res) in enumerate(track.starmap(jobs, order_outputs=False) if jobs else []):
        done[n][name] = res
        pipeline._save(pipeline.cache_key(local[n], T0, T1, PROMPT, f"tracks_{pipeline.SAMPLE_FPS}fps"), done[n])
        print(f"tracked {k + 1}/{len(jobs)}: {n} {name}", flush=True)
    need = [n for n in names if not pipeline.cache_key(local[n], T0, T1, FLOWER_PROMPT, "flowers").exists()]
    for n, fls in find_flowers.map(need, order_outputs=False) if need else []:
        pipeline._save(pipeline.cache_key(local[n], T0, T1, FLOWER_PROMPT, "flowers"), fls)
        print(f"{n}: {len(fls)} flowers", flush=True)
    print("MODAL DONE", flush=True)
