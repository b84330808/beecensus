# BeeCensus

**Point any camera at flowers. Get a pollinator census: who visited, how often, and which flowers were missed. No training data.**

Pollinator monitoring is still done by people staring at a flower patch with a clipboard. The UK Flower-Insect Timed (FIT) Count asks volunteers to watch a 50×50 cm patch for 10 minutes and tally every insect that lands. It works, but it does not scale: Ireland's national FIT Count received 1,323 validated counts from 235 volunteers in all of 2025, and a 2020 costing put a professional 800-site UK pollination monitoring network at £2.7M a year (Breeze et al., *Journal of Applied Ecology*). Growers who rent hives for pollination (around $200 a hive) have no easy way to see whether the bees are actually working the crop.

Computer vision can do the watching, but every published system so far trains its own detector on thousands of hand-labelled frames for one crop and one camera. BeeCensus does not train anything. It asks Meta's open-weight models to do what a field ecologist does:

1. **Meta SAM 3** (open weights, run locally) looks for "bee" in every frame and tracks each insect through the video, zero-shot.
2. **Muse Spark Open** (Meta open weights, served by Baseten) looks at close-ups of each tracked object and decides what it is, in FIT Count groups (honeybee, bumblebee, hoverfly, wasp, ...), or rejects it as a bud, unripe fruit or leaf spot. It then writes a plain-language note for the grower.
3. BeeCensus finds the flowers (SAM 3 again), decides when an insect has landed on one, and produces the census: a FIT Count table, a flower-visit log, a pollination coverage map (which flowers got no visit), and a CSV export.

## Results so far

All numbers are on public footage the models never saw: the Monash *Spatial Monitoring and Insect Behavioural Analysis Dataset* (strawberry polytunnels, 1920×1080, 30 fps, honeybees about 30 px across).

| Step | Test | Result |
|---|---|---|
| SAM 3 finds bees, zero-shot | 99 human-labelled test frames, 85 honeybees/wasps | **78/85 (92%)** at score 0.5, 82/85 (96%) at 0.3 |
| SAM 3 finds hoverflies (~20×12 px) | same frames, 16 hoverflies | 6–10/16: our weak spot |
| Muse Spark Open rejects look-alikes | 80 SAM 3 false alarms (buds, unripe fruit, leaf spots) | **80/80 rejected** |
| Muse Spark Open keeps real insects | the 101 labelled insects | 93/101 kept |
| Muse Spark Open names the group | same | bees/wasps 72%, hoverflies 31% |
| Whole pipeline vs Polytrack (a trained tracker, honeybee F-score 0.95) | full 10-minute videos T4, T8 (more running) | **10/12 honeybees found, 0 extra insects**; the 2 missed were on screen for 2.5 s and 0.4 s |
| Muse Spark Open as the scan gate | same videos | skipped 74 of 97 candidates (a plastic plot label, unripe fruit) before any tracking; no bee lost to it |

For reference, Polytrack trains YOLOv4 on 2,231 labelled honeybees from these same farms. BeeCensus was given the word "bee".

## How it works

```
video ──► scan: SAM 3 image detection once a second, 2×2 tiles
          (positions that fire in most seconds are buds or fruit, not visitors)
      ──► check: Muse Spark Open looks at each moving candidate; plot labels and unripe fruit are skipped
      ──► track: SAM 3 video tracking at 15 fps, only in the seconds and tiles with a confirmed candidate
      ──► link: merge duplicates where tiles overlap, bridge gaps of up to 2 s
      ──► flowers: SAM 3 "flower" every 100 s, clustered into stable flower ids
      ──► Muse Spark Open: 3 sharpest close-ups per insect → FIT group or "not an insect"
      ──► visits: insect centre inside a flower for ≥ 0.3 s
      ──► census: FIT Count table · visit log · coverage map · evidence gallery · CSV · written note
```

Why tiles: SAM 3 resizes its input to 1008 px, which would shrink a 30 px bee to 16 px. Tracking each quarter of the frame keeps bees at ~30 px. Why lowered video thresholds: SAM 3's video defaults suit large objects; a bee crossing the frame in a second barely overlaps itself between samples and was being discarded as a spurious tracklet.

## Run it

Needs an NVIDIA GPU (tested on an RTX 5060 Ti 16 GB) and a Baseten API key for Muse Spark Open in `.env` (`BASETEN_API_KEY=...`).

```bash
uv sync
uv run app.py                                   # web app on http://127.0.0.1:7861
uv run python scripts/run_survey.py video.mp4 out/survey.json 0 600 --render
uv run pytest
```

Evaluation data: `sh tools/get_monash.sh` downloads six Monash test videos, about 100 labelled test frames and the Polytrack tracks into `$BEECENSUS_DATA` (default `../data/monash`). It uses HTTP range requests, so only the needed parts of the 21 GB of archives are fetched. Evaluation scripts are in `scripts/`: `eval_detect_images.py` + `score_detect_images.py` (SAM 3 on labelled frames), `eval_classify.py` (Muse Spark Open on labelled insects and SAM 3 false alarms), `eval_polytrack.py` and `summarize_batch.py` (full videos vs Polytrack). A Polytrack insect counts as found when one of ours is within 80 px of it within 0.13 s for at least 3 of its points.

## Limitations

- Hoverflies and other insects under ~20 px are often missed or confused with bees. A closer camera fixes this better than any model change.
- A 10-minute 1080p survey takes roughly 20–35 minutes on one consumer GPU (the once-a-second SAM 3 scan is about 18 of them). Fine for daily surveys, not real time.
- "Visit" is geometric (insect centre inside the flower for 0.3 s), as in Polytrack; it does not know whether pollen was transferred.

## License

MIT (code). The Monash footage and labels are CC BY 4.0 and are not included; `tools/get_monash.sh` fetches them.

## Credits

- Footage and labels: Ratnayake, M.N., Amarathunga, D.C., Zaman, A., Dyer, A.G., Dorin, A. (2022). *Spatial Monitoring and Insect Behavioural Analysis Dataset*. Monash University. https://doi.org/10.26180/21533760 (CC BY 4.0). Polytrack: Ratnayake et al., *Spatial Monitoring and Insect Behavioural Analysis Using Computer Vision for Precision Pollination*, IJCV 2022.
- Models: Meta SAM 3 (`facebook/sam3`), Muse Spark Open (Meta, served by Baseten).
- Built for the Meta Global AI Developer Hackathon 2026.
