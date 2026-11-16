import math

import numpy as np

from genny import cyber as C
from genny import road as R


def test_wet_road_law():
    w = C.wet_road_db(70.0)
    f = C._HN_F
    assert np.all(w[f < 1000.0] == 0.0)
    k = int(np.argmin(np.abs(f - 4000.0)))
    assert abs(w[k] - (15.0 * math.log10(f[k]) - 48.0)) < 1e-9
    assert np.all(C.wet_road_db(30.0) >= w) and np.all(C.wet_road_db(110.0) <= w)
    k = int(np.argmin(np.abs(f - 10000.0)))
    assert abs(C.wet_road_db(30.0)[k] - (15.0 * math.log10(f[k]) - 12.0 * math.log10(30.0 / 70.0) - 48.0)) < 1e-9


def test_tyre_wet_louder_than_dry():
    r = lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))
    assert r(R.tyre_roll(kind="tyre_wet", dur=0.5)) > r(R.tyre_roll(kind="tyre_road", dur=0.5))


def test_rocket_directivity_follows_sp8072_fig10():
    from genny import aero as A
    assert A.rocket_directivity_db(50.0) == 0.0
    assert -22.0 < A.rocket_directivity_db(180.0) < -18.0
    assert -14.0 < A.rocket_directivity_db(120.0) < -11.0
    assert A.rocket_directivity_db(45.0) > 0.0 > A.rocket_directivity_db(20.0)


def test_chowning_wind_index_holds_in_the_release():
    from genny.chiptune2 import chowning
    sr, f, dur = 48000, 200.0, 0.5
    y = np.asarray(chowning(f, dur, sr, voice="woodwind"), float).ravel()
    seg = y[int((dur + 0.05) * sr):int((dur + 0.09) * sr)]
    s = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    k = lambda h: s[int(round(h * f * len(seg) / sr))]
    assert k(1) > 0.3 * k(3)


def test_pass_by_wet_option():
    a = C.pass_by(dur=1.0, wet=0.0)
    b = C.pass_by(dur=1.0, wet=1.0)
    assert np.all(np.isfinite(b)) and a.shape == b.shape and not np.allclose(a, b)
