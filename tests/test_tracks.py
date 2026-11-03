import numpy as np

from beecensus.detect import Tracklet, active_windows, cluster_hits
from beecensus.flowers import Flower, visits
from beecensus.tracks import link

FPS, STEP = 30.0, 2


def tl(tile, frames, xs, ys):
    t = Tracklet(tile=tile)
    for f, x, y in zip(frames, xs, ys):
        t.frame.append(f)
        t.cx.append(float(x))
        t.cy.append(float(y))
        t.area.append(900)
        t.box.append([int(x) - 15, int(y) - 15, int(x) + 15, int(y) + 15])
        t.score.append(0.8)
    return t


def test_static_lookalike_dropped():
    bud = tl(0, range(0, 300, STEP), [500] * 150, [400] * 150)
    assert link([bud], FPS, STEP) == []


def test_still_thing_tracked_in_short_windows_dropped():
    # SAM 3 fires on a bud now and then: tracked 5 s at a time, never 8 s in one go
    pieces = [tl(0, range(f, f + 150, STEP), [500] * 75, [400] * 75) for f in range(0, 3000, 600)]
    bee = tl(0, range(3000, 3060, STEP), np.linspace(100, 200, 30), [300] * 30)
    ins = link(pieces + [bee], FPS, STEP)
    assert len(ins) == 1 and ins[0].start == 3000


def test_gap_bridged_into_one_insect():
    a = tl(0, range(0, 60, STEP), np.linspace(100, 200, 30), [300] * 30)
    b = tl(0, range(90, 150, STEP), np.linspace(230, 330, 30), [300] * 30)  # back after 1 s, 30 px on
    ins = link([a, b], FPS, STEP)
    assert len(ins) == 1 and ins[0].parts == 2


def test_far_reappearance_is_another_insect():
    a = tl(0, range(0, 60, STEP), np.linspace(100, 200, 30), [300] * 30)
    b = tl(0, range(90, 150, STEP), np.linspace(1500, 1600, 30), [900] * 30)  # 1400 px away after 1 s
    assert len(link([a, b], FPS, STEP)) == 2


def test_tile_overlap_duplicate_merged():
    a = tl(0, range(0, 60, STEP), np.linspace(930, 990, 30), [300] * 30)
    b = tl(1, range(0, 60, STEP), np.linspace(932, 992, 30), [302] * 30)
    assert len(link([a, b], FPS, STEP)) == 1


def test_visit_needs_landing_not_flyover():
    flower = Flower(1, 500.0, 500.0, 30.0, 3)
    land = tl(0, range(0, 90, STEP), [500] * 45, [500] * 45)  # sits 3 s
    fly = tl(0, range(200, 210, STEP), np.linspace(300, 700, 5), [500] * 5)  # crosses in 0.3 s at speed
    ins = link([land, fly], FPS, STEP)
    v = visits(ins, [flower], FPS)
    assert len(v) == 1 and v[0].flower == 1


def test_active_windows_ignore_fixed_positions():
    scan = {float(t): [([490, 390, 510, 410], 0.5)] for t in range(60)}  # a bud, every second
    scan[30.0] = scan[30.0] + [([1490, 790, 1520, 820], 0.6)]  # one moving visitor, bottom right
    w = active_windows(scan, (1920, 1080))
    assert len(w) == 1 and w[0][0] <= 30 <= w[0][1]
    assert w[0][2] == [3]  # only the bottom-right tile is tracked


def test_hits_cluster_by_visitor():
    hits = [(10.0, [100, 100, 130, 130], 0.5), (11.0, [140, 110, 170, 140], 0.6),  # one bee drifting
            (12.0, [1500, 900, 1530, 930], 0.5),  # another, far away
            (30.0, [110, 100, 140, 130], 0.5)]  # back at the first spot much later: new candidate
    assert sorted(map(sorted, cluster_hits(hits))) == [[0, 1], [2], [3]]
