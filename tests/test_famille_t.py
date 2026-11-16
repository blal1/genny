"""Lot T: instruments du monde, mesures sur le signal rendu."""
import math

import numpy as np

from genny import instruments as I
from genny import world_t as W

SR = 44100


def _peak(y, lo, hi, sr=SR, t0=0.15, t1=None):
    x = y[int(t0 * sr):int((t1 or len(y) / sr) * sr)]
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
    fr = np.fft.rfftfreq(8 * len(x), 1 / sr)
    m = (fr > lo) & (fr < hi)
    return fr[m][np.argmax(sp[m])]


def _ok(y):
    assert np.isfinite(y).all() and np.abs(y).max() > 0.02
    assert abs(y[0]) < 0.05 and abs(y[-1]) < 0.01


def test_registered():
    for n in ("guzheng", "erhu", "morin_khuur", "oud", "qanun", "santur", "shakuhachi", "ney", "bansuri", "duduk", "bonang", "balafon"):
        assert n in I.REGISTRY


def test_string_physics_f0_tension_length():
    s = W.STRINGS["guzheng"]
    g = W.string_geometry(293.66, s)
    assert abs(W.string_freq(g["T"], g["mu"], g["L"]) - 293.66) < 1e-6
    assert abs(W.string_freq(4 * g["T"], g["mu"], g["L"]) / 293.66 - 2.0) < 1e-9
    assert abs(W.string_freq(g["T"], g["mu"], g["L"] / 2) / 293.66 - 2.0) < 1e-9
    ls = W.bridge_lengths()
    assert all(a > b for a, b in zip(ls, ls[1:]))
    y = W.guzheng(293.66, 1.2, SR)
    _ok(y)
    assert abs(_peak(y, 250, 340) / 293.66 - 1) < 0.01


def test_press_raises_pitch_by_tension():
    s = W.STRINGS["guzheng"]
    c = W.press_cents(220.0, 15.0, s)
    assert 20 < c < 400
    assert abs(W.press_cents(220.0, W.press_depth(220.0, c, s), s) - c) < 1e-6
    assert W.press_cents(220.0, 20.0, s) > c
    y = W.guzheng(220.0, 1.5, SR, press_mm=W.press_depth(220.0, 200.0, s), bend_at=0.2, bend_time=0.4)
    f0 = _peak(y, 190, 300, t0=0.1, t1=0.3)
    f1 = _peak(y, 190, 300, t0=1.0, t1=1.4)
    assert abs(1200 * math.log2(f1 / f0) - 200) < 25


def test_guzheng_glissando_monotone():
    c = W.cents_contour(40000, 44100, 0.9, bend=300.0, bend_at=0.1, bend_time=0.5)
    assert (np.diff(c) >= -1e-12).all() and abs(c[-1] - 300.0) < 1e-6
    y = W.guzheng(196.0, 1.2, SR, bend=300.0, bend_at=0.1, bend_time=0.7)
    fs = [_peak(y, 150, 300, t0=a, t1=a + 0.2) for a in (0.1, 0.4, 0.7, 0.95)]
    assert all(b > a - 0.3 for a, b in zip(fs, fs[1:])) and fs[-1] > fs[0] * 1.1


def test_decay_and_pluck_transient():
    y = W.guzheng(220.0, 0.5, SR, vel=1.0)
    from genny import filters as F
    lo = np.abs(F.bandpass(y, 220.0, SR, 15.0))
    hi = np.abs(F.bandpass(y, 1760.0, SR, 15.0))
    a, b = int(0.2 * SR), int(1.2 * SR)
    assert lo[b:b + 2000].mean() / lo[a:a + 2000].mean() > hi[b:b + 2000].mean() / hi[a:a + 2000].mean()
    assert np.argmax(np.abs(y[:int(0.05 * SR)])) < int(0.02 * SR)
    soft = W.guzheng(220.0, 0.4, SR, vel=0.2)
    hf = lambda x: (np.abs(np.fft.rfft(x[:1300])[int(1500 * 1300 / SR):]) ** 2).sum() / (np.abs(np.fft.rfft(x[:1300])) ** 2).sum()
    assert hf(y) > hf(soft)


def test_erhu_pitch_slide_skin():
    y = W.erhu(440.0, 0.8, SR)
    _ok(y)
    assert abs(_peak(y, 380, 500) / 440.0 - 1) < 0.02
    L = W.stopped_length(587.33, W.BOWED["erhu"])
    assert abs(L - 0.38 * 440.0 / 587.33) < 1e-9
    z = W.erhu(440.0, 1.2, SR, bend=200.0, bend_at=0.2, bend_time=0.6)
    fs = [_peak(z, 380, 540, t0=a, t1=a + 0.2) for a in (0.1, 0.5, 0.9)]
    assert fs[0] < fs[1] < fs[2] and abs(1200 * math.log2(fs[2] / fs[0]) - 200) < 60
    m1 = [f for f, _ in W.skin_modes(W.BOWED["erhu"], 1.0)]
    m2 = [f for f, _ in W.skin_modes(W.BOWED["erhu"], 4.0)]
    assert all(abs(b / a - 2.0) < 1e-9 for a, b in zip(m1, m2))
    assert np.isfinite(W.morin_khuur(110.0, 0.6, SR)).all()


def test_plucked_family_pitch():
    for fn, f in ((W.oud, 146.83), (W.qanun, 440.0), (W.santur, 392.0)):
        y = fn(f, 0.8, SR)
        _ok(y)
        assert abs(_peak(y, f * 0.93, f * 1.07, t0=0.1, t1=0.7) / f - 1) < 0.015
    assert W.qanun(440.0, 0.8, SR, mandal=50.0).shape[0] > 0


def test_blown_and_struck():
    for fn, f in ((W.shakuhachi, 293.66), (W.ney, 440.0), (W.bansuri, 392.0), (W.duduk, 262.0)):
        y = fn(f, 0.8, SR)
        _ok(y)
    y = W.bansuri(392.0, 1.4, SR, bend=-200.0, bend_at=0.3, bend_time=0.5)
    assert _peak(y, 300, 420, t0=1.0, t1=1.3) < _peak(y, 300, 420, t0=0.1, t1=0.3) * 0.95
    for fn, f in ((W.bonang, 440.0), (W.balafon, 262.0)):
        y = fn(f, 0.5, SR)
        _ok(y)
    a = W.balafon(262.0, 0.5, SR, buzz=0.0)
    b = W.balafon(262.0, 0.5, SR, buzz=1.0)
    hf = lambda x: np.sum(np.abs(np.fft.rfft(x))[int(1500 * len(x) / SR):])
    assert hf(b) > hf(a)
