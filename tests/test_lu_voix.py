import math
from types import SimpleNamespace as NS

import numpy as np

from genny import compose as C


def _ctx():
    return NS(q=0.1, kpc=0.0, meter=4.0, chords=[{"root": 0.0}, {"root": 0.0}, {"root": 0.0}])


def _mk(ts, ps, dur=1.0):
    return [{"t": t, "dur": dur, "pitch": p, "vel": 0.5} for t, p in zip(ts, ps)]


def test_leap_duration_erratum_descending():
    a = _mk((0.0, 2.0, 4.0, 6.0, 8.0), (60.0, 62.0, 55.0, 53.0, 52.0))
    C._kth(a, _ctx(), {"leap_duration": 1.0})
    s = math.sqrt(7)
    assert np.allclose([x["t"] for x in a], [0.0, 2.0, 4.0 + 0.024 * s, 6.0 + 0.024 * s - 0.042 * s,
                                              8.0 + 0.024 * s - 0.042 * s])


def test_duration_contrast_leaves_long_notes():
    a = _mk((0.0, 8.0, 16.0, 24.0), (60.0,) * 4)
    C._kth(a, _ctx(), {"duration_contrast": 1.0})
    assert np.allclose([x["t"] for x in a], [0.0, 8.0, 16.0, 24.0])


def test_melodic_charge_smoothing():
    a = _mk((0.0, 2.0, 4.0), (67.0, 69.0, 67.0))
    C._kth(a, _ctx(), {"melodic_charge": 1.0})
    for x, db in zip(a, (0.75 * 0.6, 0.6, 0.55 * 0.6)):
        assert math.isclose(x["vel"], 0.5 * 10 ** (db / 20.0))


def test_melodic_and_harmonic_intonation():
    a = _mk((0.0, 2.0, 4.0), (64.0, 61.0, 62.0))
    C._kth(a, _ctx(), {"melodic_intonation": 1.0})
    assert np.allclose([x["pitch"] for x in a], [64.04, 61.07, 62.03])
    a = _mk((0.0, 2.0, 4.0), (60.0, 61.0, 62.0))
    C._kth(a, _ctx(), {"melodic_intonation": 1.0})
    assert math.isclose(a[1]["pitch"], 61.0 - 0.07)
    a = _mk((0.0, 0.0, 4.0), (64.0, 67.0, 60.0))
    C._kth(a, _ctx(), {"harmonic_intonation": 1.0})
    assert np.allclose([x["pitch"] for x in a], [64.0 - 0.14, 67.02, 60.0])


def test_beak_end_correction_fletcher():
    from genny.physical import voice as V
    assert math.isclose(V.beak_end_correction(3000.0, 0.01, 0.05), 0.0025 + 0.0075)


def test_choir_vib_arch_prame():
    from genny import choir
    assert choir.VIB_ARCH == (0.56, 0.73)
