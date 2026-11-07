import math

import numpy as np

from genny import contact as C


def test_wave_restitution_glass_and_steel():
    glass = (7e10, 0.25, 2600.0, 2e11, 1 / 3, 7800.0)
    for v, e in ((0.1, 0.999), (0.9, 0.996), (3.0, 0.993)):
        assert abs(C.wave_restitution(v, *glass) - e) < 6e-4
    for args, c0, coef in ((glass, math.sqrt(7e10 / 2600), 1.27), ((2e11, 1 / 3, 7800.0, 2e11, 1 / 3, 7800.0), 5000.0, 1.04)):
        lam = 1 - C.wave_restitution(1.0, *args) ** 2
        assert abs(lam / (1.0 / c0) ** 0.6 / coef - 1) < 0.04


def test_patch_average_is_the_contact_sinc():
    dx, a = 1e-5, 3e-4
    x = np.arange(100000) * dx
    for wl in (4e-3, 2e-3, 1e-3):
        k = 2 * np.pi / wl
        got = C.patch_average(np.sin(k * x), a, dx).std() / np.sin(k * x).std()
        assert abs(got - math.sin(k * a) / (k * a)) < 0.02
    assert C.patch_average(np.sin(2 * np.pi * x / (2 * a)), a, dx).std() < 0.03


def test_roll_patch_dulls_and_default_is_unchanged():
    def high(y, sr=48000):
        p = np.abs(np.fft.rfft(y)) ** 2
        return p[np.fft.rfftfreq(len(y), 1 / sr) > 1500].sum() / p.sum()
    base = C.roll(dur=0.5, speed=1.0, radius=0.03)
    assert np.array_equal(base, C.roll(dur=0.5, speed=1.0, radius=0.03, patch=0.0))
    assert high(C.roll(dur=0.5, speed=1.0, radius=0.03, patch=1.0)) < 0.7 * high(base)
