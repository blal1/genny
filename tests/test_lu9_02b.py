"""LU9_02b: options added from the documents of group 02 read again in full (2026-11-15)."""
import math

import numpy as np
import pytest

from genny import acoustic, analysis as A, calibrate, dsp, friction, household, physics, tubes

SR = 44100


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def test_wineglass_mode_freq_matches_lowest_mode_and_orders_modes():
    r, h, H = 0.035, 0.0015, 0.035 / 1.2
    assert friction.wineglass_mode_freq(h, r, H) == pytest.approx(friction.wineglass_f0(h, r, H), rel=0.02)
    f = [friction.wineglass_mode_freq(h, r, H, 1.875, n) for n in (2, 3, 4, 5)]
    assert f == sorted(f)
    assert friction.wineglass_mode_freq(h, r, H, 4.694, 2) > f[0]
    with pytest.raises(ValueError):
        friction.wineglass_mode_freq(h, r, H, 1.875, 1)


def test_wineglass_torsion_far_above_the_elliptical_mode():
    assert friction.wineglass_torsion_freq(0.03) > 10000.0
    assert friction.wineglass_torsion_freq(0.03) > 5.0 * friction.wineglass_mode_freq(0.0015, 0.035, 0.03)


def test_filled_glass_french_partials():
    r = household.french_glass_ratios()
    assert r[0] == 1.0 and np.all(np.diff(r) > 0) and r[1] == pytest.approx(1.706, abs=0.01)
    y = household.filled_glass(SR, partials="french")
    assert y.size > 0 and np.isfinite(y).all()
    with pytest.raises(ValueError):
        household.filled_glass(SR, partials="nope")


def test_trumpet_mutes_pass_their_bands():
    t = np.arange(SR // 2) / SR
    tone = lambda f: np.sin(2 * np.pi * f * t)
    assert _rms(acoustic.trumpet_mute(tone(440.0), SR, "straight")) < 0.3 * _rms(tone(440.0))
    assert _rms(acoustic.trumpet_mute(tone(3000.0), SR, "straight")) > 0.8 * _rms(tone(3000.0))
    assert _rms(acoustic.trumpet_mute(tone(1000.0), SR, "cup")) > 0.9 * _rms(tone(1000.0))
    assert _rms(acoustic.trumpet_mute(tone(300.0), SR, "cup")) < 0.4 * _rms(tone(300.0))
    assert _rms(acoustic.trumpet_mute(tone(1700.0), SR, "harmon")) > 0.9 * _rms(tone(1700.0))
    with pytest.raises(ValueError):
        acoustic.trumpet_mute(tone(440.0), SR, "wah")


def test_trumpet_mute_kind_changes_the_sound_only_when_asked():
    a = acoustic.trumpet(440.0, 0.3, SR, mute=1.0)
    b = acoustic.trumpet(440.0, 0.3, SR, mute=1.0, mute_kind="straight")
    assert a.shape == b.shape and not np.allclose(a, b)
    assert np.allclose(acoustic.trumpet(440.0, 0.3, SR, mute_kind="straight"), acoustic.trumpet(440.0, 0.3, SR))


def test_mode_lock_captures_the_strong_low_partials_only():
    f, lock, w = dsp.mode_lock([100.0, 201.0, 298.0, 420.0], [1.0, 0.8, 0.5, 0.5])
    assert list(lock) == [True, True, False, False]
    assert f[1] == pytest.approx(2.0 * w) and f[3] == 420.0
    assert not dsp.mode_lock([100.0, 201.0, 298.0, 420.0], [1.0, 0.8, 0.5, 0.5], coupling=1e-4)[1].any()


def test_double_slope_reads_two_slopes_and_one():
    rng = np.random.default_rng(0)
    n = 3 * SR
    t = np.arange(n) / SR
    env = np.where(t < 0.25, 10 ** (-60 * t / 0.9 / 20), 10 ** (-(60 * 0.25 / 0.9) / 20) * 10 ** (-60 * (t - 0.25) / 2.1 / 20))
    d = calibrate.double_slope(env * rng.standard_normal(n), SR)
    assert d["t_early"] == pytest.approx(1.0, rel=0.15) and d["t_late"] == pytest.approx(2.08, rel=0.1) and d["ratio"] > 1.8
    one = calibrate.double_slope(10 ** (-60 * t / 1.5 / 20) * rng.standard_normal(n), SR)
    assert one["ratio"] == pytest.approx(1.0, rel=0.2)


def test_subglottal_pair_bounds_and_filter_dip():
    assert tubes.subglottal_pair(0.03) == (1350.0, 1425.0, 5.0)
    assert tubes.subglottal_pair(0.2) == (1350.0, 1575.0, 13.0)
    imp = np.zeros(SR)
    imp[0] = 1.0
    fq = np.fft.rfftfreq(SR, 1.0 / SR)
    db = lambda h, f0: 20 * np.log10(np.abs(np.fft.rfft(h))[np.argmin(np.abs(fq - f0))])
    h = tubes.subglottal_filter(imp, SR, 1350.0, 0.2)
    assert db(h, 1350.0) < -9.0 and abs(db(h, 300.0)) < 0.1 and abs(db(h, 4000.0)) < 0.1
    assert db(tubes.subglottal_filter(imp, SR, 1350.0, 0.03), 1350.0) > db(h, 1350.0)
    x = np.random.default_rng(0).standard_normal(2000)
    assert np.allclose(tubes.subglottal_filter(x, SR, open_phase=np.zeros(2000)), x)


def test_friedlander_peak_freq_and_blast_scaling():
    assert physics.friedlander_peak_freq(0.5e-3, 1.0) == pytest.approx(2000.0 / (2.0 * math.pi))
    with pytest.raises(ValueError):
        physics.friedlander_peak_freq(0.5e-3, 3.8)
    p = physics.muzzle_peak_angle([0.0, 90.0, 180.0], 112.0, 75.0)
    assert p[0] == pytest.approx(112.0) and p[2] == pytest.approx(75.0) and 75.0 < p[1] < 112.0
    assert physics.hopkinson_scale(8.0) == pytest.approx(2.0)


def test_goldstein_harmonicity_finds_the_fundamental():
    like, f0 = A.goldstein_harmonicity([200.0, 400.0, 600.0, 800.0])
    assert f0 == pytest.approx(200.0)
    assert A.goldstein_harmonicity([300.0]) == (0.0, 0.0)
    t = np.arange(SR // 2) / SR
    harm = sum(np.sin(2 * np.pi * 220.0 * k * t) / k for k in range(1, 8))
    noise = np.random.default_rng(0).standard_normal(t.size)
    hg = A.wichern_features(harm, SR, harmonicity="goldstein")["harmonicity"]
    ng = A.wichern_features(noise, SR, harmonicity="goldstein")["harmonicity"]
    assert hg > 10.0 * ng
    with pytest.raises(ValueError):
        A.wichern_features(harm, SR, harmonicity="x")
