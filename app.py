"""BeeCensus web app: pollinator survey video in, flower-visit census out.

uv run app.py  ->  http://127.0.0.1:7861
Examples replay saved runs (a 10-minute survey takes ~30 min on one GPU); uploads run live.
"""

import json
import tempfile
from pathlib import Path

import gradio as gr

from beecensus import pipeline, render, report

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"


def _examples() -> dict[str, Path]:
    return {json.loads(p.read_text())["title"]: p for p in sorted(EXAMPLES.glob("*.json"))} if EXAMPLES.exists() else {}


def _outputs(res: dict, video_path: Path, note: str):
    st = report.stats(res)
    head = (f"### {st['insects']} insects · {st['visits']} flower visits · {st['flowers_visited']}/{st['flowers']} flowers visited "
            f"({st['coverage']:.0%}) in {st['minutes']} min\n"
            f"Muse Spark Open rejected {st['rejected']} look-alikes (buds, unripe fruit) that SAM 3 flagged.\n\n{note}")
    csv_path = Path(tempfile.mkdtemp()) / "fit_count.csv"
    csv_path.write_text(report.fit_csv(res), encoding="utf-8")
    return (str(video_path), head, report.fit_table(res), report.coverage_map(res), report.gallery(res), str(csv_path))


def show_example(title: str):
    p = _examples()[title]
    res = json.loads(p.read_text())
    res["video"] = str((ROOT / res["video"]).resolve())  # stored relative to the repo; needs the Monash videos (tools/get_monash.sh)
    return _outputs(res, p.with_suffix(".mp4"), res.get("summary", ""))


def run_upload(path: str, t_start: float, t_end: float, prompt: str, flower_prompt: str, progress=gr.Progress()):
    if not path:
        raise gr.Error("Upload a video first")
    logs = []

    def log(s):
        logs.append(s)
        progress(0.1 + min(0.8, 0.05 * len(logs)), desc=s)

    res = pipeline.run(path, t_start, t_end or None, prompt=prompt, flower_prompt=flower_prompt, log=log)
    out = Path(tempfile.mkdtemp()) / "census.mp4"
    render.render(res, out)
    try:
        note = report.summary(res)
    except Exception as e:  # the survey is still useful without the written note
        note = f"(summary unavailable: {e})"
    return _outputs(res, out, note)


with gr.Blocks(title="BeeCensus") as demo:
    gr.Markdown("# BeeCensus\nPoint any camera at flowers. Get a pollinator census: who visited, how often, which flowers were missed. "
                "No training data: **Meta SAM 3** tracks every insect zero-shot, **Muse Spark Open** checks each one like a field ecologist.")
    with gr.Row():
        with gr.Column(scale=1):
            ex = gr.Dropdown(list(_examples()), value=next(iter(_examples()), None), label="Saved survey (Monash strawberry farm, CC BY 4.0)")
            ex_btn = gr.Button("Show saved survey", variant="primary")
            gr.Markdown("**Or run your own** (about 2–4 min of GPU per minute of footage with insects)")
            up = gr.Video(label="Survey video", sources=["upload"])
            with gr.Row():
                t0 = gr.Number(0, label="from (s)")
                t1 = gr.Number(60, label="to (s, 0 = end)")
            pr = gr.Textbox("bee", label="What to count (SAM 3 prompt)")
            fpr = gr.Textbox("white flower", label="Flowers (SAM 3 prompt)")
            run_btn = gr.Button("Run census")
        with gr.Column(scale=2):
            vid = gr.Video(label="Census", autoplay=True)
            head = gr.Markdown()
            table = gr.Dataframe(headers=["FIT group", "insects", "flower visits"], label="FIT Count")
    with gr.Row():
        cov = gr.Image(label="Pollination coverage: visits per flower (red = no visit)")
        gal = gr.Gallery(label="Evidence: every insect, with Muse Spark Open's verdict", columns=4, height=520)
    csv_out = gr.File(label="FIT Count export (CSV)")
    outs = [vid, head, table, cov, gal, csv_out]
    ex_btn.click(show_example, ex, outs)
    run_btn.click(run_upload, [up, t0, t1, pr, fpr], outs)

if __name__ == "__main__":
    demo.launch(server_port=7861)
