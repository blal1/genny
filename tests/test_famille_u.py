import math

import numpy as np

from genny import geo_u as U
from genny import matter as Ma
from genny.core import DEFAULT_SR

SR = Ma.SR


def _peak_hz(y, lo=20.0, hi=4000.0, sr=DEFAULT_SR):
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    k = np.where((f >= lo) & (f <= hi))[0]
    return f[k[np.argmax(sp[k])]]



def test_hertz_contact_time_scaling():
    t1 = U.hertz_contact_time(1.0, 2.0, 0.1)
    assert abs(U.hertz_contact_time(32.0, 2.0, 0.1) / t1 - 32.0 ** 0.4) < 1e-9
    assert abs(U.hertz_contact_time(1.0, 64.0, 0.1) / t1 - 32.0 ** -0.2) < 1e-9
    assert 1e-5 < t1 < 1e-2


def test_power_law_sizes_recovered():
    rng = np.random.default_rng(1)
    d = U.power_law_sizes(20000, 0.05, 5.0, 1.5, rng)
    assert d.min() >= 0.05 and d.max() <= 5.0
    assert abs(U.hill_exponent(d, 0.05) - 1.5) < 0.1


def test_boom_frequency_matches_formula_and_harmonics():
    for g in (150.0, 250.0, 400.0):
        y = U.dune_sound(mode="boom", grain=g, dur=5.0, seed=1)
        assert abs(_peak_hz(y) - U.boom_hz(g)) < 2.0
    assert abs(U.boom_hz(160.0) - 100.0) < 1.0
    y = U.dune_sound(mode="boom", grain=170.0, dur=5.0, seed=1)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / DEFAULT_SR)
    f0 = U.boom_hz(170.0)
    h2 = sp[(f > 1.97 * f0) & (f < 2.03 * f0)].max()
    assert h2 > 0.05 * sp.max()


def test_squeak_and_silent():
    ys = U.dune_sound(mode="squeak", grain=200.0, speed=0.2, dur=3.0, seed=2)
    assert abs(_peak_hz(ys, 300, 3000) - 1000.0) < 20.0
    yb = U.dune_sound(mode="boom", dur=3.0, seed=2)
    yq = U.dune_sound(mode="silent", dur=3.0, seed=2)
    p = lambda y: np.abs(np.fft.rfft(y * np.hanning(len(y)))).max() / np.sqrt(np.mean(np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2))
    assert p(yb) > 4.0 * p(yq)


def test_ice_flexural_dispersion():
    h, d = 0.05, 60.0
    y, f = Ma.ice_chirp(d, h, return_freq=True)
    D = Ma.ICE_E * h ** 3 / (12.0 * (1.0 - Ma.ICE_NU ** 2))
    a = math.sqrt(D / (Ma.ICE_RHO * h))
    t0 = math.sqrt((d * d / (8.0 * math.pi * a)) / f[0])
    for i in (len(f) // 5, len(f) // 2, 4 * len(f) // 5):
        t = t0 + i / SR
        cg = 2.0 * math.sqrt(a * 2.0 * math.pi * f[i])
        assert abs(d / cg - t) / t < 0.01
    seg = y[len(y) // 3:len(y) // 3 + 1200]
    assert abs(_peak_hz(seg, 50, 20000) - f[len(y) // 3 + 600]) / f[len(y) // 3 + 600] < 0.15


def test_ice_thickness_lowers_top_frequency():
    _, f1 = Ma.ice_chirp(60.0, 0.05, return_freq=True)
    _, f2 = Ma.ice_chirp(60.0, 0.30, return_freq=True)
    assert f1[0] > 3.0 * f2[0]


def test_brune_corner():
    assert abs(U.brune_corner(2.0) - 0.37 * 1957.0 / 2.0) < 1e-9
    assert U.brune_corner(1.0) > U.brune_corner(10.0)


def test_debris_power_cube_of_speed():
    n = SR * 4
    pw = lambda v: float(np.var(U.debris_impacts(n, v, 0.1, np.random.default_rng(3))))
    e = math.log(pw(8.0) / pw(4.0)) / math.log(2.0)
    assert abs(e - 3.0) < 0.35


def test_debris_bigger_grain_lowers_frequency():
    n = SR * 4
    c = lambda D: (lambda y: (np.fft.rfftfreq(n, 1 / SR) * np.abs(np.fft.rfft(y))).sum() / np.abs(np.fft.rfft(y)).sum())(
        U.debris_impacts(n, 5.0, D, np.random.default_rng(4)))
    assert c(0.4) < c(0.02)


def test_avalanche_cloud_corner_and_speed_law():
    n = SR * 8
    rng = np.random.default_rng(5)
    z = U.avalanche_cloud(n, 40.0, 100.0, rng, 1.0)
    sp = np.abs(np.fft.rfft(z)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / SR)
    lo = sp[(f > 20) & (f < 40)].mean()
    hi = sp[(f > 2000) & (f < 4000)].mean()
    slope = math.log(hi / lo) / math.log(3000.0 / 30.0)
    assert abs(slope + 5.0 / 3.0) < 0.3
    a = lambda u: math.sqrt(np.mean(U.avalanche_cloud(n, u, 100.0, np.random.default_rng(6)) ** 2))
    assert abs(a(40.0) / a(20.0) - 16.0) < 1.0


def test_avalanche_infrasound_band():
    y = U.avalanche_flow(dur=10.0, infra=1.0, seed=1)
    y0 = U.avalanche_flow(dur=10.0, infra=0.0, seed=1)
    band = lambda v: np.abs(np.fft.rfft(v))[(np.fft.rfftfreq(len(v), 1 / SR) > 2) & (np.fft.rfftfreq(len(v), 1 / SR) < 10)].sum()
    assert band(y) > 1.5 * band(y0)


def test_pour_rate_and_level_laws():
    assert abs(U.grain_rate(0.2, 0.2) * float(U.sphere_mass(0.2e-3, "sand")) - 0.2) < 1e-9
    n = SR * 2
    r = lambda q: float(np.std(U.pour_signal(n, q, 2.0, 0.5, "sand", 0.0, np.random.default_rng(7))))
    assert abs(r(0.4) / r(0.1) - 2.0) < 0.2


def test_all_sounds_render_finite():
    for fn, kw in ((U.rockfall_hertz, {}), (U.debris_flow_granular, {"dur": 6.0}), (U.avalanche_flow, {"dur": 6.0}),
                   (U.icequake_swarm, {"dur": 4.0}), (U.iceberg_fizz, {"dur": 3.0}), (U.granular_pour, {"wet": 0.5}),
                   (U.granular_pour, {"kind": "mud"}), (U.dune_sound, {"mode": "squeak"})):
        y = fn(**kw)
        assert np.all(np.isfinite(y)) and 0.05 < np.max(np.abs(y)) <= 1.0
        assert abs(y[0]) < 0.05 and abs(y[-1]) < 0.05
