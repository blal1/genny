import math

import numpy as np

from genny import nature as N


def test_drop_force_impulse_and_peak():
    sr = 2_000_000
    for d, v, (A, B, C) in ((2.0, 4.0, N.RAIN_FORCE["round"]), (5.0, 7.0, N.RAIN_FORCE["flat"])):
        f = N.drop_force(d, v, sr, pitch=0.0)
        D = d * 1e-3
        assert abs(f.sum() / sr - 998.0 * math.pi / 6.0 * D ** 3 * v) < 1e-9
        assert abs(np.argmax(f) / sr - C / B * D / v) < 2e-6
    assert N.RAIN_FORCE["round"] == (3.596, 3.333, 0.500)
    assert N.RAIN_FORCE["flat"] == (6.0413, 4.5860, 0.5274)


def test_drop_force_pitch_scales_normal_speed():
    sr = 2_000_000
    f0 = N.drop_force(2.0, 4.0, sr, pitch=0.0)
    f60 = N.drop_force(2.0, 4.0, sr, pitch=60.0)
    assert abs(f60.sum() / f0.sum() - 0.5) < 1e-6
    assert abs(f60.max() / f0.max() - 0.25) < 0.01


def test_rain_on_drops_option():
    y = N.rain_on(surface="toit", model="drops", intensity=0.5, dur=0.5, seed=1)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.1
    h = N.rain_on(surface="toit", intensity=0.5, dur=0.5, seed=1)
    assert not np.allclose(np.asarray(y).ravel()[: len(np.asarray(h).ravel())], np.asarray(h).ravel())
