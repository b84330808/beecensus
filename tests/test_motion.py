import numpy as np

from beecensus.motion import _grey as grey, blobs, tracklets
from beecensus.video import Window


def scene(bee_x=None, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), 90, np.uint8)
    img[200:260, 300:420] = 140  # a still flower
    if noise:
        img = np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    if bee_x is not None:
        img[100:124, bee_x:bee_x + 30] = 20  # a dark 30 px bee
    return img


def test_flying_bee_is_one_blob_at_its_current_place():
    a, b, c = (grey(scene(x)) for x in (100, 160, 220))
    found = blobs(a, b, c)
    assert len(found) == 1
    box, _ = found[0]
    assert abs((box[0] + box[2]) / 2 - 175) < 10 and abs((box[1] + box[3]) / 2 - 112) < 10


def test_still_scene_and_sensor_noise_give_nothing():
    a, b, c = (grey(scene(noise=4, seed=s)) for s in range(3))
    assert blobs(a, b, c) == []


def test_blobs_chain_into_one_tracklet():
    xs = range(60, 480, 40)
    win = Window([scene(x) for x in xs], list(range(0, 2 * len(xs), 2)), 30.0)
    ts = [t for t in tracklets(win) if len(t.frame) >= 3]
    assert len(ts) == 1 and len(ts[0].frame) == len(xs) - 2
