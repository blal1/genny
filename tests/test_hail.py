import math

import numpy as np

from genny.contact import hail, hail_impulses


def test_hail_impulse_and_sizes():
    f, d, t = hail_impulses(4.0, 48000, rate=400.0, size=6.7e-3, decay=0.0, seed=1)
    assert abs(np.median(d) - 6.7e-3) < 0.6e-3
    v = 142.0 * np.sqrt(d)
    j = (1.9 * 870.0 * math.pi / 6.0 * d ** 3 * v).sum()
    assert abs(f.sum() / 48000 - j) / j < 0.02                       # Lane: impulse (1 + eta) m v per stone


def test_largest_stone_comes_early():
    # Kopp 2023: the largest stone of a dense shower falls at relative time 0.34 in median
    when = [t[np.argmax(d)] / 10.0 for s in range(200)
            for _, d, t in [hail_impulses(10.0, 8000, rate=50.0, decay=1.0, seed=s)]]
    assert 0.24 < np.median(when) < 0.36


def test_hail_renders():
    y = hail(sr=22050, dur=1.0, rate=60.0)
    assert np.isfinite(y).all() and 0.5 < np.max(np.abs(y)) <= 0.81
