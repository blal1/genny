import math

import numpy as np
import scipy.optimize as so

from genny import family_v as V, geo_u as G, psychoacoustics_p2 as P

SR = 48000


def _tone(f, level, dur=1.0):
    t = np.arange(int(dur * SR)) / SR
    return np.sqrt(2.0) * 2e-5 * 10.0 ** (level / 20.0) * np.sin(2 * np.pi * f * t)


def test_boom_matches_measured_dunes():
    for d_um, hz in ((160, 105.0), (210, 90.0), (270, 75.0), (340, 63.0)):
        assert abs(G.boom_hz(d_um) / hz - 1.0) < 0.15, d_um


def test_hms_calibration_same_as_standard_text():
    assert abs(P.hms_loudness(_tone(1000.0, 40.0), SR)[0] - 1.0) < 0.01


def test_cantilever_ratios_are_roots_of_cos_cosh():
    f = lambda x: math.cos(x) * math.cosh(x) + 1.0
    r = [so.brentq(f, a, b) for a, b in ((1.5, 2.2), (4.0, 5.0), (7.0, 8.5))]
    ratios = [(x / r[0]) ** 2 for x in r]
    assert abs(ratios[1] - V.FREE_REED_RATIOS[1]) < 0.01 and abs(ratios[2] - V.FREE_REED_RATIOS[2]) < 0.01


def test_woodworth_and_kuhn_formulas():
    a, c = 0.0875, V.C_SOUND
    assert abs(V.itd_woodworth(90.0) - a / c * (math.pi / 2 + 1.0)) < 1e-9
    assert abs(V.itd_kuhn(90.0) - 3.0 * a / c) < 1e-9


def test_violin_modes_in_published_bands():
    bands = {"A0": (270, 290), "CBR": (380, 420), "B1-": (430, 480), "B1+": (510, 580)}
    modes = {m[0]: m[1] for m in V.BODY_MODES["violin"]}
    for k, (lo, hi) in bands.items():
        assert lo <= modes[k] <= hi, (k, modes[k])


def test_elephant_trumpet_within_one_sd():
    from genny import fauna
    assert abs(np.mean(fauna.TRUMPET_HZ) - 474.0) < 70.0


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("ok", k)
