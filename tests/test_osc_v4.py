"""Oscillators added from the deep-reading report (effects theme): residual tables, DPW, BLIT, sync, CZ, GBUZZ, noises."""
import numpy as np
from scipy.special import jv

from genny import analysis as A
from genny import osc
from genny.sfx import REGISTRY

SR = 48000


def test_four_point_polyblamp_table():
    d = np.array([0.0, 0.3, 0.7])
    r = lambda t: osc.residual(t, 3, "ramp")  # noqa: E731
    assert np.allclose(r(d - 2), d ** 5 / 120)
    assert np.allclose(r(d - 1), -d ** 5 / 40 + d ** 4 / 24 + d ** 3 / 12 + d ** 2 / 12 + d / 24 + 1 / 120)
    assert np.allclose(r(d), d ** 5 / 40 - d ** 4 / 12 + d ** 2 / 3 - d / 2 + 7 / 30)
    assert np.allclose(r(d + 1), -d ** 5 / 120 + d ** 4 / 24 - d ** 3 / 12 + d ** 2 / 12 - d / 24 + 1 / 120)
    assert abs(osc.residual(np.array([0.0]), 3, "ramp")[0] - 7 / 30) < 1e-12
    assert np.allclose(osc.residual(np.array([-0.5, 0.5]), 1, "ramp"), 0.5 ** 3 / 6)


def test_four_point_triangle_aliases_less_than_two_point():
    f = 3563.2
    two = A.alias_snr(osc.triangle(f, SR, SR, points=2), SR, f)
    four = A.alias_snr(osc.triangle(f, SR, SR), SR, f)
    assert four > two + 6


def test_saw_above_2k_uses_the_4_point_blep():
    f = 3563.2
    t = osc.phase(f, SR, SR) % 1.0
    two = A.alias_snr(2 * t - 1 - osc._polyblep(t, np.full(SR, f / SR)), SR, f)
    assert A.alias_snr(osc.saw(f, SR, SR), SR, f) > two + 3


def test_dpw_polynomials_match_table_one_and_three():
    x = np.linspace(-1, 0.999, 50)
    neg, pos = osc._dpw_poly(4, "saw")
    p = np.where(x < 0, neg(x), pos(x))
    assert np.allclose(p * 24 - (x ** 4 - 2 * x ** 2), (p * 24 - (x ** 4 - 2 * x ** 2))[0])
    neg, pos = osc._dpw_poly(4, "triangle")                   # Table III N = 4 with the corrected -2x^3
    xs = x[x >= 0]
    ref = xs ** 3 * np.abs(xs) - 2 * xs ** 3 + xs
    assert np.allclose(pos(xs), -ref / 12, atol=1e-12)
    assert np.allclose(neg(-xs), -pos(xs), atol=1e-12)          # odd: continuous through the wrap


def test_dpw_saw_amplitude_and_alias():
    f = 1046.5
    y = osc.dpw(f, SR, SR, order=3)
    assert abs(np.max(np.abs(y[1000:])) - 1.0) < 0.05
    assert A.alias_snr(y, SR, f) > A.alias_snr(osc.dpw(f, SR, SR, order=1), SR, f) + 15


def test_blit_has_m_harmonics_and_unit_mean():
    f = 1000.0
    y = osc.blit(f, SR, SR)
    assert abs(np.mean(y) - f / SR) < 1e-4
    h = np.abs(np.fft.rfft(y)) / (SR / 2)
    assert h[1000] > 0.9 * 2 * f / SR and h[23000] > 0.5 * h[1000]
    y5 = osc.blit(f, SR, SR, harmonics=5)
    h5 = np.abs(np.fft.rfft(y5))
    assert h5[6000] < 1e-6 * h5[5000]


def test_bspline_blit_droops_like_sinc4():
    f = 441.0
    y = osc.blit(f, SR, SR, kernel="bspline")
    h = np.abs(np.fft.rfft(y))
    k = 10584                                                 # 22.05 % of SR
    want = np.sinc(k / SR) ** 4
    assert abs(h[k] / h[441] - want / np.sinc(441 / SR) ** 4) < 0.02


def test_hard_sync_restarts_the_slave():
    f, r = 220.0, 2.5
    y = osc.hard_sync(f, SR, SR, ratio=r)
    h = np.abs(np.fft.rfft(y))
    assert np.argmax(h[100:2000]) + 100 in (440, 660)             # harmonics of the master only
    assert h[330] < 0.01 * h[440]
    naive = 2 * ((r * (osc.phase(f, SR, SR) % 1.0)) % 1.0) - 1
    assert A.alias_snr(y, SR, f) > A.alias_snr(naive, SR, f) + 10


def test_cz_phase_distortion_pieces():
    y = osc.phase_distortion(100.0, 480, SR, amount=0.0)
    assert np.allclose(y, np.cos(2 * np.pi * (osc.phase(100.0, 480, SR) % 1.0)))
    t = osc.phase(100.0, 480, SR) % 1.0
    z = osc.phase_distortion(100.0, 480, SR, amount=0.6)
    mid = 0.8
    assert np.allclose(z, np.cos(2 * np.pi * np.where(t < mid, 0.5 * t / mid, 0.5 + 0.5 * (t - mid) / (1 - mid))))


def test_gbuzz_matches_direct_sum():
    n, f = 2000, 200.0
    p = 2 * np.pi * osc.phase(f, n, SR)
    r = 0.7
    want = sum(r ** i * np.cos((2 + i) * p) for i in range(5)) * (1 - r) / (1 - r ** 5)
    assert np.allclose(osc.gbuzz(f, n, SR, nharm=5, lharm=2, mul=r), want, atol=1e-9)


def test_fm_sidebands_follow_bessel():
    f, I = 1000.0, 1.7
    h = np.abs(np.fft.rfft(osc.fm(f, SR, SR, ratio=0.25, index=I))) / (SR / 2)
    for k in range(4):
        assert abs(h[int(f + 250 * k)] - abs(jv(k, I))) < 0.01
    y2 = osc.fm(f, SR, SR, ratio=0.25, index=I, ratio2=0.5, index2=0.8)
    assert not np.allclose(y2, osc.fm(f, SR, SR, ratio=0.25, index=I))


def test_additive_drops_partials_above_nyquist_per_sample():
    n = SR
    f = np.linspace(1000, 10000, n)
    y = osc.additive(f, n, SR, partials=[(1, 1.0, 10.0), (3, 1.0, 10.0)])
    assert np.max(np.abs(y[:n // 4])) > 1.5                     # both partials live at first
    assert np.max(np.abs(y[-n // 10:])) < 1.05                  # the 3rd harmonic left at 8 kHz


def test_harmonic_bank_renormalises_under_nyquist():
    y = osc.harmonic(12000.0, 1.0, [[0.5, 0.5, 0.5]], SR, SR)   # 2nd and 3rd at or over 24 kHz
    assert np.allclose(y, np.sin(2 * np.pi * osc.phase(12000.0, SR, SR)))


def test_jp8000_offsets():
    assert osc.JP8000_CENTS == (-21.0, -14.0, -7.0, 0.0, 4.0, 8.0, 12.0)
    y = osc.supersaw(110.0, SR, SR, mode="jp8000")
    h = np.abs(np.fft.rfft(y))
    assert h[int(round(110 * 2 ** (-21 / 1200)))] > 0.1 * h[110]


def test_noises_clocked_and_held():
    y = osc.held_noise(4800, 100.0, SR)
    assert len(np.unique(y)) == 10 and np.all(y[:480] == y[0])               # 0.1 s at 100 Hz: 10 holds
    p = osc.pitched_noise(1000.0, SR, SR, interpolate=False)
    assert np.count_nonzero(np.diff(p)) <= 2000 + 1
    b = osc.pitched_noise(1000.0, SR, SR, color="brown")
    assert np.max(np.abs(b)) <= 1.0
    w = np.random.default_rng(0).uniform(-1, 1, 1 << 16)
    s = np.abs(np.fft.rfft(osc.pink_pk3(w))) ** 2
    lo, hi = s[100:200].mean(), s[1600:3200].mean()
    assert abs(10 * np.log10(lo / hi) - 12.0) < 2.0                # 4 octaves at -3 dB/octave


def test_gendyn_curves_bend_segments():
    a = osc.gendyn(100.0, 4800, SR, seed=1)
    b = osc.gendyn(100.0, 4800, SR, seed=1, curve=5.0)
    assert np.max(np.abs(a - b)) > 0.05 and np.max(np.abs(b)) <= 1.0 + 1e-9


def test_corner_clip_rounds_corners():
    f = 3563.2
    x = 2.0 * np.sin(2 * np.pi * f * np.arange(SR) / SR)
    plain = A.alias_snr(np.clip(x, -1, 1), SR, f)
    assert A.alias_snr(osc.corner_clip(x, 1.0), SR, f) > plain + 5
    full = osc.corner_clip(x, mode="full")
    assert A.alias_snr(full, SR, 2 * f) > A.alias_snr(np.abs(x), SR, 2 * f) + 5


def test_fm_machine_frequency_swings_by_work():
    y = REGISTRY["fm_machine"]["fn"](sr=SR, dur=1.0, size=200.0, work=0.0)
    h = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(h) - 200) <= 1
    z = REGISTRY["fm_machine"]["fn"](sr=SR, dur=1.0, size=200.0, work=80.0, speed=0.5, modfreq=30.0)
    hz = np.abs(np.fft.rfft(z))
    assert hz[200] < 0.9 * h[200]


def test_new_waves_dispatch():
    for w in ("jp8000", "cz", "gbuzz", "dpw_saw", "dpw_triangle", "dpw_square", "blit", "blit_saw", "blit_square",
              "sync", "pitched_noise", "held_noise"):
        y = osc.osc(w, 220.0, 4800, SR)
        assert y.shape == (4800,) and np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.05, w
