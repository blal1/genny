import math

import numpy as np

from genny import calibrate, dsp, filters, matter, spectral
from genny.physical import impedance, particles

SR = 44100


def test_symclip_values():
    v = dsp._symclip(np.array([0.2, 0.5, 0.9, -0.2]))
    assert np.allclose(v, [0.4, 0.916667, 1.0, -0.4], atol=1e-5)


def test_g711_codes():
    assert abs(float(dsp.g711_mulaw_value(0x80)) - 32124 / 32768) < 1e-6
    assert float(dsp.g711_mulaw_value(0xFF)) == 0.0
    t = np.arange(SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 1000 * t)
    for code, val in ((dsp.g711_mulaw_code, dsp.g711_mulaw_value), (dsp.g711_alaw_code, dsp.g711_alaw_value)):
        y = val(code(x))
        snr = 10 * np.log10(np.sum(x ** 2) / np.sum((x - y) ** 2))
        assert snr > 33.0


def test_shelf1_gain():
    n = SR
    x = np.random.default_rng(0).standard_normal(n)
    y = filters.shelf1(x, "lowshelf", 500.0, SR, 12.0)
    X, Y = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(n, 1 / SR)
    lo = (f > 15) & (f < 30)
    hi = (f > 10000) & (f < 15000)
    assert abs(20 * np.log10(Y[lo].mean() / X[lo].mean()) - 12.0) < 0.5
    assert abs(20 * np.log10(Y[hi].mean() / X[hi].mean())) < 0.5


def test_scramble_keeps_magnitudes():
    x = np.random.default_rng(1).standard_normal(4096)
    y = spectral.scramble(x, SR, 3)
    assert y.shape == x.shape and np.all(np.isfinite(y))
    assert abs(np.abs(y).max() - np.abs(x).max()) < 1e-6
    assert not np.allclose(x, y)


def test_esprit_benchmark_sourced_error():
    r = calibrate.esprit_benchmark(n_modes=10, seed=0)
    assert r["f_err"] < 0.04 and r["d_err"] < 0.53


def test_rayleigh_modes_light_damping():
    f, d = calibrate.rayleigh_modes([200.0, 2000.0], 1.0, 1e-6)
    assert np.allclose(f, [200.0, 2000.0], rtol=1e-3)
    assert d[1] > d[0]


def test_blowhole_constants():
    c = networks_blowhole()
    assert abs(c["scatter"] + 0.0741) < 1e-3
    assert abs(c["th_coeff"] + 0.3043) < 1e-3


def networks_blowhole():
    from genny import networks
    return networks.blowhole_coefficients(22050)


def test_tube_loss():
    assert abs(float(impedance.tube_loss_alpha(440.0)) - 0.0887) < 1e-3
    assert abs(float(impedance.round_trip_loss(0.6, 440.0)) - 0.899) < 2e-3


def test_capacitance_pitch_is_sqrt_c():
    r = 1e-3
    for s in (3e-3, 2e-3):
        c = float(matter.bubble_capacitance(r, surface=s) / r)
        assert abs(float(matter.capacitance_pitch(r, surface=s)) - math.sqrt(c)) < 1e-9


def test_surfacing_bubble_film_adds_burst():
    a = particles.surfacing_bubble(np.random.default_rng(0))
    b = particles.surfacing_bubble(np.random.default_rng(0), film=1.0)
    assert len(b) >= len(a)
    assert np.all(np.isfinite(b))


def test_delta_th_value():
    assert abs(particles.DELTA_TH - 0.026) < 1e-9


def test_rain_options():
    from genny.physical import ambience
    base = ambience.rain(1.0, np.random.default_rng(0))
    again = ambience.rain(1.0, np.random.default_rng(0))
    assert np.array_equal(base, again)
    for kw in ({"wind": 5.0}, {"on_solids": 1.0}, {"on_leaves": 1.0}, {"drops": 0.5}, {"small_alpha": -1.0}):
        y = ambience.rain(1.0, np.random.default_rng(0), **kw)
        assert np.all(np.isfinite(y)) and len(y) == len(base)
    gone = ambience.rain(1.0, np.random.default_rng(0), wind=9.0)
    assert np.all(np.isfinite(gone))


def test_sword_swing_natural_flag():
    from genny.physical import modal
    a = modal.sword_swing("SWD1", np.random.default_rng(0))
    b = modal.sword_swing("SWD1", np.random.default_rng(0), natural=False)
    assert np.all(np.isfinite(b)) and len(a) == len(b)


def test_ui_gesture_kinds():
    from genny import designed
    for k in ("haptic", "liquid"):
        y = designed.ui_gesture(SR, kind=k)
        assert np.all(np.isfinite(y)) and np.abs(y).max() > 0


def test_loop_form_option():
    from genny import compose
    assert "loop_form" in compose._SHARED
