"""Lot 34 O2: Addendum livres, section 2 (lines 241-275). One check per changed row, on the rendered signal.

Run:  uv run --with pytest pytest tests/test_lot34_o2.py -q
"""
import math

import numpy as np

from genny import designed, friction, fdstring, instruments as I, plates, reverbs
from genny.physical import winds as W

SR = 44100


def _imp(n=SR):
    x = np.zeros(n)
    x[100] = 1.0
    return x


def _mag(y, f):
    S = np.abs(np.fft.rfft(y))
    return S[int(round(f * len(y) / SR))]


def test_directivity_table_membrane_and_player():
    x = _imp()
    for inst, lim in (("oboe", 500.0), ("clarinet", 700.0), ("bassoon", 250.0), ("violin", 400.0), ("tuba", 75.0)):
        y = reverbs.directivity(x, SR, inst, 180.0)
        assert abs(_mag(y, lim * 0.5) - 1.0) < 0.02                            # omni under the limit
        assert 0.2 < _mag(y, 16.0 * lim) / _mag(x, 16.0 * lim) < 0.3             # -12 dB at 16x the limit
    m = reverbs.directivity(x, SR, "membrane", 45.0, m=1)
    assert abs(20 * math.log10(_mag(m, 1000.0)) + 3.0) < 0.1                     # -3 dB at 45 degrees
    m2 = reverbs.directivity(x, SR, "membrane", 0.0, elevation=71.5, m=1)
    assert abs(20 * math.log10(_mag(m2, 1000.0)) + 10.0) < 0.2                   # -10 dB at 71.5 degrees
    p = reverbs.directivity(x, SR, "violin", 180.0, player=True)
    assert _mag(p, 8000.0) < 0.5 * _mag(reverbs.directivity(x, SR, "violin", 180.0), 8000.0)


def test_diffuser_curtain_tolerances():
    x = _imp(SR // 2)
    c = 343.2
    d = 0.1
    y = reverbs.reflection(x, SR, "diffuser", distance=3.0, depth=d, level=1.0, mix=1.0)
    echo = y - np.pad(x, (0, len(y) - len(x)))[:len(y)]
    f_quarter = c / (4.0 * d)                                                    # wavelength / 4 = depth
    f_half = c / (2.0 * d)
    assert _mag(echo, f_quarter) < 0.05 * _mag(echo, f_half)
    assert abs(reverbs.curtain_cutoff(10.0) - 850.0) < 1e-9
    z = reverbs.curtain(_imp(), SR, gap=10.0, absorption=0.75)
    assert abs(_mag(z, 6000.0) - 0.5) < 0.05 and abs(_mag(z, 100.0) - 1.0) < 0.05
    assert reverbs.t60_within_jnd(1.0, 1.03) and not reverbs.t60_within_jnd(1.0, 1.06) and reverbs.t60_within_jnd(0.1, 0.115)
    assert abs(reverbs.free_field_level(100.0, 10.0) - 69.0) < 0.01
    r = reverbs.room_metrics(np.exp(-np.arange(SR) / (0.2 * SR)) * np.random.default_rng(0).standard_normal(SR), SR)
    assert r["t20"] > 0


def test_rod_bank_options_and_chimes_driver():
    f = 220.0
    base = friction.rod_bank(f, 0.6, SR, 1.0, radiator="none")
    for kw in (dict(feed="tip"), dict(mount="rubber"), dict(damp="lead"), dict(body="wood"), dict(neighbours="random")):
        y = friction.rod_bank(f, 0.6, SR, 1.0, radiator="none", **kw)
        assert np.isfinite(y).all() and not np.allclose(y[:len(base)], base[:len(y)])
    tail = lambda y: float(np.sqrt(np.mean(y[int(0.4 * SR):int(0.6 * SR)] ** 2)))
    assert tail(friction.rod_bank(f, 0.6, SR, 1.0, radiator="none", damp="lead")) < 0.5 * tail(base)
    assert abs(friction.rod_length_for("fork", 100.0) ** 2 * 100.0 - 5.2) < 1e-9
    n = {d: designed.cage_hits(2, 20.0, np.random.default_rng(0), d).size for d in ("pendulum", "washer", "balls3", "wind")}
    assert abs(n["washer"] - 60) <= 1 and abs(n["balls3"] - 60) <= 1 and 15 < n["wind"] < 50
    y = designed.rod_cage(SR, 2, "none", 1.5, 0, "balls3")
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01


def test_water_instruments_and_lip():
    y = designed.suikinkutsu(SR, 1.0, 1.0, 3.0, 1500.0, 0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert _mag(y[:SR * 2], 520.0) > 10 * _mag(y[:SR * 2], 4000.0)               # the ring is at the pot's mode, the soil kills the highs
    g0, g1 = I.REGISTRY["glass"]["fn"](880.0, 0.8, SR, 1.0, 0.0), I.REGISTRY["glass"]["fn"](880.0, 0.8, SR, 1.0, 1.0)
    pk = lambda y: np.argmax(np.abs(np.fft.rfft(y[2000:2000 + 8192] * np.hanning(8192)))) * SR / 8192
    assert abs(pk(g1) / pk(g0) - 1.0 / math.sqrt(2.2)) < 0.03
    assert W.lip_two_resonance(350.0) == (96.0, 111.0) and W.lip_two_resonance(390.0) == (97.0, 117.0)
    assert abs(W.lip_volume_freq(9.0) - 500.0) < 1e-9 and abs(W.lip_volume_freq(12.0) - 460.0) < 1.0
    p = np.linspace(0.0, 1.0, 200)
    on = W.lip_sounding(p, 0.5)
    assert on[np.argmax(p > 0.5)] and W.lip_sounding(np.r_[0.6, 0.49, 0.47], 0.5)[1] and not W.lip_sounding(np.r_[0.6, 0.49, 0.47], 0.5)[2]
    clar = I.REGISTRY["slide_clarinet"]["fn"](220.0, 1.0, SR, 1.0, 12.0, 0.5)
    h = lambda s: np.argmax(np.abs(np.fft.rfft(s * np.hanning(len(s))))) * SR / len(s)
    assert h(clar[int(0.9 * SR):int(0.99 * SR)]) > 1.7 * h(clar[int(0.1 * SR):int(0.4 * SR)])


def test_fd_hard_friction_and_free_plates():
    y0 = fdstring.bowed_string_fd(220.0, 0.5, SR, 1.0, friction="smooth")
    y1 = fdstring.bowed_string_fd(220.0, 0.5, SR, 1.0, friction="hard", residual=0.3)
    assert np.isfinite(y1).all() and np.sqrt(np.mean(y1[8000:] ** 2)) > 0.01 and not np.allclose(y0[:len(y1)], y1[:len(y0)])
    b = fdstring.Bar(880.0, SR, ratio=4.0)
    assert b.os * SR <= 1.5e6
    a = plates.plate_crash(SR, 1.0, quality=0.5, dur=0.4, edges="supported")
    f = plates.plate_crash(SR, 1.0, quality=0.5, dur=0.4, edges="free")
    assert np.isfinite(f).all() and np.abs(f).max() > 0.01 and not np.allclose(a[:len(f)], f[:len(a)])
    g = plates.tam_tam(SR, 1.0, quality=0.5, dur=0.4, edges="free")
    assert np.isfinite(g).all()
    info = {}
    plates.nl_plate(SR, 1.0, 0.4, 1.0, 400.0, 12.0, (0.2, 0.3), 6.9, 0.005, "vk", 1.0, plates.BRONZE, 0.6, 0.5, 0.4, info=info, edges="free")
    assert "guard_hits" in info
