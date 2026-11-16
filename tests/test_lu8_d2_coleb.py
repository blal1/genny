import numpy as np

from genny import ordnance as O


def test_surface_period_factor_matches_small_charge_measurement():
    f = O.surface_period_factor(0.66 / 2.20462, 3.0 * 0.3048)
    assert 0.72 < f < 0.80  # measured 0.1475 s against 0.199 s in deep water: 0.74
    assert O.surface_period_factor(0.66 / 2.20462, 20.0 * 0.3048) > 0.9
    assert O.surface_period_factor(1000.0, 1.0) == 0.5


def _pulse_time(y, sr=None):
    sr = sr or O.P
    pk = int(np.argmax(np.abs(y[: int(0.01 * sr)])))
    rest = np.abs(y).copy()
    rest[: pk + int(0.01 * sr)] = 0.0
    return (int(np.argmax(rest)) - pk) / sr


def test_near_surface_shortens_first_pulse_and_default_unchanged():
    kw = dict(pulses=1)
    a = O.underwater_blast(0.3, 30.0, 1.0, np.random.default_rng(0), **kw)
    b = O.underwater_blast(0.3, 30.0, 1.0, np.random.default_rng(0), near_surface=True, **kw)
    c = O.underwater_blast(0.3, 30.0, 1.0, np.random.default_rng(0), near_surface=False, **kw)
    assert np.array_equal(a, c)
    assert _pulse_time(b) < _pulse_time(a)


def test_pulse_width_cap_is_22_percent_of_period():
    assert O.UW_PULSE_FRAC == 0.22
    y = O.underwater_blast(0.01, 300.0, 5.0, np.random.default_rng(0), pulses=1)
    assert np.isfinite(y).all() and np.abs(y).max() > 0

