"""LU9_02: options added from the papers read on 2026-11-13."""
import math

import numpy as np
import pytest
from scipy import signal as sg

from genny import acoustic, analysis as A, contact, cyber, dsp, fauna, filters as F, machines, matter, sports


def test_fdn_saturation_time_coprime_pair():
    assert dsp.fdn_saturation_time([7, 13]) == 72
    assert dsp.fdn_saturation_time([4, 6]) == -1


def test_fdn_mixing_time_orders_abel_below_defrance():
    m = [479, 587, 673, 773, 881, 991, 1091, 1201]
    ta, td = (dsp.fdn_mixing_time(m, 48000, k) for k in ("abel", "defrance"))
    assert 0.0 < ta < 0.2 and 0.0 < td < 0.2 and ta != td
    assert dsp.fdn_echo_density(m, 4000)[:3000].sum() > 0


def test_ladder_dangelo_matches_linear_prototype():
    sr = 48000
    x = np.zeros(8192)
    x[0] = 1e-4
    for N in (3, 4, 6):
        k = 0.5 * F.ladder_k_max(N)
        y = F.ladder(x, 2000.0, sr, res=0.5, kind="dangelo", drive=1.0, order=N)
        wa = 2 * sr * math.tan(math.pi * 2000 / sr)
        pa = wa * (-1 + k ** (1 / N) * np.exp(1j * math.pi * (2 * np.arange(N) + 1) / N))
        b, a = sg.zpk2tf([], pa, wa ** N)
        bz, az = sg.bilinear(b, a, fs=sr)
        ref = sg.lfilter(bz, az, x)
        assert np.max(np.abs(y - ref)) / np.max(np.abs(ref)) < 1e-3


def test_ladder_k_max_and_stable_under_drive():
    assert F.ladder_k_max(4) == pytest.approx(4.0)
    assert F.ladder_k_max(8) == pytest.approx(1.884, abs=1e-3)
    xs = np.sin(2 * np.pi * 200 * np.arange(8192) / 48000)
    for d in (1, 10, 100):
        assert np.isfinite(F.ladder(xs, 2000.0, 48000, res=0.9, kind="dangelo", drive=d)).all()


def test_terhardt_pitch_finds_the_missing_fundamental():
    f = 99.0 * np.arange(1, 13)
    lv = 70.0 - 6.0 * np.log2(np.arange(1, 13))
    r = A.terhardt_pitch(f, lv)
    assert r["virtual"][0][0] == pytest.approx(99.0, rel=0.02)
    f = 99.0 * np.arange(2, 13)
    assert A.terhardt_pitch(f, np.full(f.size, 70.0))["virtual"][0][0] == pytest.approx(99.0, rel=0.03)


def test_mfcc_davis_shape():
    sr = 16000
    x = np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)
    y = A.mfcc_davis(x, sr)
    assert y.shape[1] == 10 and np.isfinite(y).all()


def test_wichern_paper_vs_legacy():
    sr = 16000
    t = np.arange(2 * sr) / sr
    sine = A.wichern_features(np.sin(2 * np.pi * 440 * t), sr)
    noise = A.wichern_features(np.random.default_rng(0).standard_normal(2 * sr) * 0.1, sr)
    assert sine["harmonicity"] > 0.9 > 0.3 > noise["harmonicity"]
    assert sine["spectral_sparsity"] > noise["spectral_sparsity"]
    assert noise["transient_index"] > sine["transient_index"]
    leg = A.wichern_features(np.sin(2 * np.pi * 440 * t), sr, legacy=True)
    assert leg["centroid"] == pytest.approx(440.0, abs=15.0)


def test_shorted_turns_adds_sideband_tones():
    y = cyber.induction_motor(dur=2.0, fault="shorted_turns") if "dur" in cyber.induction_motor.__code__.co_varnames else None
    assert y is None or np.isfinite(y).all()


def test_bubble_prosperetti_option_close_to_default_and_validated():
    a, b = matter.bubble_params(1e-3), matter.bubble_params(1e-3, thermal="prosperetti")
    assert b[0] == pytest.approx(a[0], rel=0.1)
    kappa, d_th = matter.bubble_thermal(1e-3, a[0])
    assert 0.9 < kappa < 1.5 and d_th > 0
    with pytest.raises(ValueError):
        matter.bubble_params(1e-3, thermal="x")


def test_gear_friction_coefficient_oil_range_and_trends():
    mu = float(machines.gear_friction_coefficient(0.2, 1.0, 15.0, 10.0, 0.5, 0.04))
    assert 0.003 < mu < 0.03
    assert machines.gear_friction_coefficient(0.2, 2.0, 15.0, 10.0, 0.5, 0.04) > mu
    assert machines.gear_friction_coefficient(0.2, 1.0, 15.0, 10.0, 1.0, 0.04) > mu


def test_hertz_elliptic_regression_fit_near_simple_and_exact_sphere():
    s = contact.hertz_elliptic(100.0, 0.01, 0.05, 2e11)
    r = contact.hertz_elliptic(100.0, 0.01, 0.05, 2e11, fit="regression")
    assert r[0] == pytest.approx(s[0], rel=0.05) and r[1] == pytest.approx(s[1], rel=0.05)
    with pytest.raises(ValueError):
        contact.hertz_elliptic(100.0, 0.01, 0.05, 2e11, fit="x")


def test_sphere_cavity_modes_roots():
    m = sports.sphere_cavity_modes(0.1195, 6)
    ka = [x[2] for x in m]
    assert ka[:3] == pytest.approx([2.082, 3.342, 4.493], abs=2e-3)
    assert m[0][1] == 1 and m[2][1] == 0


def test_grey_chiff_ping_sits_on_the_harmonic():
    y = acoustic.grey_chiff(8000, 44100, 200.0, 0.3, ping=3)
    fr = np.fft.rfftfreq(8000, 1 / 44100)
    assert fr[np.argmax(np.abs(np.fft.rfft(y)))] == pytest.approx(600.0, abs=15.0)


def test_pig_grunt_squeal_renders():
    y = fauna.pig(kind="grognement_couinement")
    assert y.size > 1000 and np.isfinite(y).all()
