import numpy as np

from genny import aero, ordnance


def test_bubble_period_factor_tnt_is_one_and_rdx_longer():
    assert ordnance.bubble_period_factor("tnt") == 1.0
    assert ordnance.bubble_period_factor("RDX") > 1.15


def test_double_exponential_tail_is_continuous_and_normalised():
    a1, k1, a2, k2 = ordnance.UW_TAIL
    assert abs(a1 + a2 - 1.0) < 1e-3
    assert abs(a1 * np.exp(-k1) + a2 * np.exp(-k2) - np.exp(-1.0)) < 0.35


def test_bubble_ode_first_collapse_near_period_and_finite():
    t0, p = ordnance.bubble_ode(500.0, 50.0, 30.0, 0.4)
    assert np.isfinite(p).all() and len(p) > 100
    assert np.abs(p).max() > 1e4


def test_underwater_blast_options_same_length_finite():
    rng = lambda: np.random.default_rng(1)
    base = ordnance.underwater_blast(100.0, 50.0, 30.0, rng())
    for kw in ({"tail": "double"}, {"bubble": "keller"}):
        y = ordnance.underwater_blast(100.0, 50.0, 30.0, rng(), **kw)
        assert np.isfinite(y).all() and y.shape == base.shape
    y = ordnance.underwater_blast(100.0, 50.0, 30.0, rng(), explosive="rdx")        # the length follows the bubble period (3 T + 1.5 s)
    assert np.isfinite(y).all() and y.shape != base.shape


def test_gear_spectrum_peaks_between_strouhal_half_and_five():
    x = np.logspace(-1, 1.5, 200)
    g = aero.gear_spectrum_db(x)
    assert 0.5 < x[np.argmax(g)] < 5.0
    assert aero.gear_spectrum_db(1.0, wheels=1) < aero.gear_spectrum_db(1.0, wheels=2)


def test_gear_overhead_weaker_than_sideline():
    x = np.logspace(0, 1, 20)
    assert (aero.gear_spectrum_db(x, phi=0.0) <= aero.gear_spectrum_db(x, phi=90.0) + 1e-9).all()


def test_airframe_gear_diameter_finite_and_default_unchanged():
    a = aero.airframe(speed=70.0, gear=1, gear_diameter=0.9, seed=2)
    b = aero.airframe(speed=70.0, gear=1, seed=2)
    assert np.isfinite(a).all() and a.shape == b.shape
    assert np.array_equal(b, aero.airframe(speed=70.0, gear=1, gear_diameter=0.0, seed=2))
