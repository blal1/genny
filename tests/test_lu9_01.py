import math

import numpy as np

from genny import creatures as cr
from genny import geo, matter, speech, viscous
from genny.physical import ambience as amb


def test_vot_tables():
    assert speech.VOICED_VOT == {"lab": 0.001, "alv": 0.005, "vel": 0.021}
    assert speech.ASPIRATED_VOT == {"lab": 0.058, "alv": 0.070, "vel": 0.080}
    assert speech.SPANISH_VOT == {"lab": 0.004, "alv": 0.009, "vel": 0.029}


def test_cavity_depth_slope():
    d, v = 3.2e-3, 1.5
    r = float(matter.cavity_depth(d, v))
    assert abs(r / math.sqrt(d ** 1.5 * v) - 0.4293) < 1e-3
    assert abs(float(matter.cavity_depth(2 * d, v)) / r - 2 ** 0.75) < 1e-9


def test_ice_floating_slows_group_speed():
    assert abs(math.sqrt(matter.ICE_WATER_LOAD) - 221.0 / 286.0) < 1e-12
    _, f0 = matter.ice_chirp(60.0, 0.3, return_freq=True)
    _, f1 = matter.ice_chirp(60.0, 0.3, return_freq=True, floating=True)
    assert abs(f1[0] / f0[0] - 1.0 / matter.ICE_WATER_LOAD) < 0.02


def test_strombolian_pulse_matches_fit():
    p = geo.strombolian_pulse()
    assert abs(len(p) / 2000.0 - 0.32) < 0.05
    assert p[0] > 0.0 and 8.0 < p.max() < 13.0 and -32.0 < p.min() < -27.0


def test_lava_slug_option():
    a = geo.lava(seed=1)
    assert np.array_equal(a, geo.lava(seed=1, slug=0.0))
    assert not np.array_equal(a, geo.lava(seed=1, slug=0.5))


def test_hill_turn():
    assert abs(amb.TURN_DEG - 16.3 * math.sqrt(math.pi / 2.0)) < 1e-9
    assert abs(amb.TURN_DEG * math.sqrt(2.0 / math.pi) - 16.3) < 1e-9


def test_burst_freq_measured():
    assert abs(viscous.burst_freq(0.0055, measured=True) - 42.1) < 0.5
    assert abs(viscous.burst_freq(0.022, measured=True) - 20.1) < 0.5
    assert abs(viscous.burst_freq(0.01) - 30.0) < 2.0


def test_tmg_locks_to_forcing():
    for rate in (5.0, 25.0):
        g = cr.tmg_pressure(4.0, rate)[4000:]
        sp = np.abs(np.fft.rfft(g - g.mean()))
        assert abs(np.fft.rfftfreq(len(g), 1.0 / 2000.0)[np.argmax(sp)] - rate) < 0.6
    y = cr.birdsong(gestures=5.0, model="tmg")
    assert y.size > 0 and np.isfinite(y).all()
