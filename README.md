# BeeCensus

**Point any camera at flowers. Get a pollinator census: who visited, how often, and which flowers were missed. No training data.**

Pollinator monitoring is still done by people staring at a flower patch with a clipboard. The UK Flower-Insect Timed (FIT) Count asks volunteers to watch a 50×50 cm patch for 10 minutes and tally every insect that lands. It works, but it does not scale: Ireland's national FIT Count received 1,323 validated counts from 235 volunteers in all of 2025, and a 2020 costing put a professional 800-site UK pollination monitoring network at £2.7M a year (Breeze et al., *Journal of Applied Ecology*). Growers who rent hives for pollination (around $200 a hive) have no easy way to see whether the bees are actually working the crop.

Computer vision can do the watching, but every published system so far trains its own detector on thousands of hand-labelled frames for one crop and one camera. BeeCensus does not train anything. It asks Meta's open-weight models to do what a field ecologist does:

1. **Meta SAM 3** (open weights, run locally) looks for "bee" in every frame and tracks each insect through the video, zero-shot.
2. **Muse Spark Open** (Meta open weights, served by Baseten) looks at close-ups of each tracked object and decides what it is, in FIT Count groups (honeybee, bumblebee, hoverfly, wasp, ...), or rejects it as a bud, unripe fruit or leaf spot. It then writes a plain-language note for the grower.
3. BeeCensus finds the flowers (SAM 3 again), decides when an insect has landed on one, and produces the census: a FIT Count table, a flower-visit log, a pollination coverage map (which flowers got no visit), and a CSV export.

## Results

All numbers are on public footage the models never saw: the Monash *Spatial Monitoring and Insect Behavioural Analysis Dataset* (strawberry polytunnels, 1920×1080, 30 fps, honeybees about 30 px across).

| Step | Test | Result |
|---|---|---|
| SAM 3 finds bees, zero-shot | 99 human-labelled test frames, 85 honeybees/wasps | **78/85 (92%)** at score 0.5, 82/85 (96%) at 0.3 |
| SAM 3 finds hoverflies (~20×12 px) | same frames, 16 hoverflies | 6–10/16: our weak spot |
| Muse Spark Open rejects look-alikes | 80 SAM 3 false alarms (buds, unripe fruit, leaf spots) | **80/80 rejected** |
| Muse Spark Open keeps real insects | the 101 labelled insects | 93/101 kept |
| Muse Spark Open names the group | same | bees/wasps 72%, hoverflies 31% |
| Whole pipeline vs Polytrack (a trained tracker, honeybee F-score 0.95) | ten full 10-minute videos | **21/23 honeybees found**: 19/21 on the six videos it was developed on, **2/2 on the four videos held back until the end** |
| Polytrack's other insects | same | 13/19: all 6 hoverflies (4 of them in held-out T10), 5/10 wasps, 2/3 moths. The 5 missed wasps were on Polytrack's own tracks for 2–23 points |
| Is what it reports real? | all 95 tracks it reported, each one checked | **86 (91%) are insects**: 78 on a Polytrack insect, 8 on insects Polytrack does not track (6 tracks of one black-and-orange beetle in T5, a moth in T7, a fly in T3). 4 are look-alikes (a dead leaf under a flower in T2, reported 3 times, and a flower); 5 are too blurred to tell |
| Muse Spark Open as the scan gate | same | skipped 1,556 of 1,834 candidates (a plastic plot label, unripe fruit, swaying leaves) before any tracking, so SAM 3 tracked a third of the footage |

For reference, Polytrack trains YOLOv4 on 2,231 labelled honeybees from these same farms. BeeCensus was given the word "bee".

Per video (`scripts/summarize_batch.py`). The six development videos:

| Video | Honeybees found (Polytrack) | Other insects found (Polytrack) | Our insects | Ours not in Polytrack | Spark gate rejected | Tracked | Visits | Flowers visited |
|---|---|---|---|---|---|---|---|---|
| T1 | 3/3 | 2/2 (hoverfly, hoverfly) | 17 | 0 | 75/94 | 127s | 16 | 9/20 |
| T4 | 6/7 | 0/1 (wasp) | 13 | 1 | 298/325 | 117s | 16 | 6/16 |
| T5 | 2/2 | 0/1 (wasp) | 8 | 6 | 74/85 | 145s | 3 | 3/16 |
| T6 | 0/0 | 5/5 (wasp, wasp, wasp, wasp, wasp) | 4 | 0 | 4/24 | 48s | 0 | 0/21 |
| T7 | 3/4 | 1/1 (butterfly_moth) | 15 | 1 | 86/113 | 156s | 26 | 7/38 |
| T8 | 5/5 | 1/3 (wasp, butterfly_moth, wasp) | 14 | 0 | 38/63 | 127s | 22 | 6/18 |
| **Development, 6 videos** | **19/21** | 9/13 | 71 | 8 | 575/704 | 721s of 3600s | | |

The four held-out videos (T2, T3, T9, T10), downloaded before tuning started and not looked at until the pipeline was frozen:

| Video | Honeybees found (Polytrack) | Other insects found (Polytrack) | Our insects | Ours not in Polytrack | Spark gate rejected | Tracked | Visits | Flowers visited |
|---|---|---|---|---|---|---|---|---|
| T2 | 1/1 | 0/0 (-) | 11 | 4 | 416/445 | 234s | 16 | 6/13 |
| T3 | 1/1 | 0/1 (wasp) | 6 | 3 | 205/230 | 277s | 16 | 9/26 |
| T9 | 0/0 | 0/1 (butterfly_moth) | 0 | 0 | 177/221 | 381s | 0 | 0/17 |
| T10 | 0/0 | 4/4 (hoverfly, hoverfly, hoverfly, hoverfly) | 7 | 2 | 183/234 | 350s | 5 | 5/21 |
| **Held out, 4 videos** | **2/2** | 4/6 | 24 | 9 | 981/1130 | 1242s of 2400s | | |

**One rule was added after seeing a held-out result.** In the first T9 run, one pale-yellow object (a resting butterfly or a yellowed leaf; Muse Spark Open could not decide either) sat still for all 10 minutes and was reported 11 times, once per tracking window: each window was shorter than the 8 s that marks a still thing as a bud. The rule now adds up still tracks at the same spot across windows (`tracks.drop_still_spots`). On the development videos it removes only tracks Spark had already rejected, so no number changes; on T9 it takes the tracks reported but not in Polytrack from 11 to 0. Runs made before the rule were updated with `scripts/drop_still.py` (no GPU needed). Nothing else that changes results was touched after the held-out runs (resume caches and the Modal path only change where the work runs).

**Where it ran.** T4, T7 and T9 ran start to finish on one RTX 5060 Ti (76–104 minutes each). Under hours of full load that desktop's GPU started dropping off the PCIe bus, so the SAM 3 stages of the other seven videos ran on Modal (`scripts/modal_gpu.py`, L40S): same code, same cache files, and a check on 22 seconds of T8 matched the local scan to within 2 px. Muse Spark Open calls and the rest of the census still ran locally.

## How it works

```
video ──► scan: SAM 3 image detection once a second, 2×2 tiles
          (positions that fire in most seconds are buds or fruit, not visitors)
          + frame differencing 3 times a second, for bees too blurred for SAM 3
      ──► check: Muse Spark Open looks at each moving candidate; plot labels and unripe fruit are skipped
      ──► track: SAM 3 video tracking at 15 fps, only in the seconds and tiles with a confirmed candidate
      ──► link: merge duplicates where tiles overlap, bridge gaps of up to 2 s,
          drop still things (no movement for 8 s in all, even across tracking windows)
      ──► flowers: SAM 3 "flower" every 100 s, clustered into stable flower ids
      ──► Muse Spark Open: 3 sharpest close-ups per insect → FIT group or "not an insect"
      ──► visits: insect centre inside a flower for ≥ 0.3 s
      ──► census: FIT Count table · visit log · coverage map · evidence gallery · CSV · written note
```

Why tiles: SAM 3 resizes its input to 1008 px, which would shrink a 30 px bee to 16 px. Tracking each quarter of the frame keeps bees at ~30 px. Why lowered video thresholds: SAM 3's video defaults suit large objects; a bee crossing the frame in a second barely overlaps itself between samples and was being discarded as a spurious tracklet.

## Run it

Needs an NVIDIA GPU (tested on an RTX 5060 Ti 16 GB) and a Baseten API key for Muse Spark Open in `.env` (`BASETEN_API_KEY=...`). Without a local GPU, `scripts/modal_gpu.py` runs the SAM 3 stages on Modal and leaves their results in `cache/`; `run_survey.py` then finishes the census locally.

```bash
uv sync
uv run app.py                                   # web app on http://127.0.0.1:7861
uv run python scripts/run_survey.py video.mp4 out/survey.json 0 600 --render
uv run pytest
```

Evaluation data: `sh tools/get_monash.sh` downloads ten Monash test videos, about 100 labelled test frames and the Polytrack tracks into `$BEECENSUS_DATA` (default `../data/monash`). It uses HTTP range requests, so only the needed parts of the 21 GB of archives are fetched. Evaluation scripts are in `scripts/`: `eval_detect_images.py` + `score_detect_images.py` (SAM 3 on labelled frames), `eval_classify.py` (Muse Spark Open on labelled insects and SAM 3 false alarms), `eval_polytrack.py` and `summarize_batch.py` (full videos vs Polytrack). A Polytrack insect counts as found when one of ours is within 80 px of it within 0.13 s for at least 3 of its points. Polytrack's file names give each track's start only to the second, so each track is aligned within one second of the video's best offset.

## Limitations

- Hoverflies and other insects under ~20 px are often missed or confused with bees. A closer camera fixes this better than any model change.
- A bee on screen for under 2 s can still be missed: the 2 Polytrack honeybees not found were visible for 0.4 s and 1.3 s.
- An insect that wanders off and comes back more than 2 s later is counted again (the T5 beetle became 6 tracks).
- Anything that does not move for 8 s is taken for a bud, so an insect resting that long is dropped too. The census counts visits, not insects sitting still.
- A 10-minute 1080p survey takes 76–104 minutes on one consumer GPU (the once-a-second SAM 3 scan is about 18 of them). Tracking splits into 15-second chunks, so it spreads over cloud GPUs: ten L40S did the SAM 3 work of seven videos in under half an hour. Fine for daily surveys, not real time.
- "Visit" is geometric (insect centre inside the flower for 0.3 s), as in Polytrack; it does not know whether pollen was transferred.

## License

MIT (code). The Monash footage and labels are CC BY 4.0 and are not included; `tools/get_monash.sh` fetches them.

## Credits

- Footage and labels: Ratnayake, M.N., Amarathunga, D.C., Zaman, A., Dyer, A.G., Dorin, A. (2022). *Spatial Monitoring and Insect Behavioural Analysis Dataset*. Monash University. https://doi.org/10.26180/21533760 (CC BY 4.0). Polytrack: Ratnayake et al., *Spatial Monitoring and Insect Behavioural Analysis Using Computer Vision for Precision Pollination*, IJCV 2022.
- Models: Meta SAM 3 (`facebook/sam3`), Muse Spark Open (Meta, served by Baseten).
- Built for the Meta Global AI Developer Hackathon 2026.
