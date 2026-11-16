"""Lot 34 O3: Addendum livres, section 2 (lines 276-310): filters, delays, reverberation, synthesis, interactive composition. One check per row where code was added, and
a measurement for the rows found already done.

Run:  uv run --with pytest pytest tests/test_lot34_o3.py -q
"""
import math

import numpy as np
from scipy import signal as sg

from genny import compose, dsp, filters as F, graphsolver, spectral
from genny.physical import modal, space, winds

SR = 44100


def _mag_db(y, f, sr=SR):
    Y = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    return 20 * np.log10(Y[np.argmin(np.abs(fr - f))] + 1e-30)


def test_phase_splitter_is_90_degrees():
    assert dsp.phase_splitter_error(8, 100.0) < 1.0
    n = SR
    t = np.arange(n) / SR
    for f in (200.0, 1000.0, 5000.0, 15000.0):
        i, q = dsp.phase_splitter(np.sin(2 * np.pi * f * t), SR, 8, 100.0)
        s = slice(20000, 30000)
        ph = lambda y: np.angle(np.sum(y[s] * np.exp(-2j * np.pi * f * t[s])))   # noqa: E731
        d = (ph(q) - ph(i) + np.pi) % (2 * np.pi) - np.pi
        assert abs(abs(math.degrees(d)) - 90.0) < 1.5
        assert abs(np.std(q[s]) / np.std(i[s]) - 1.0) < 0.02


def test_plexiglass_wall():
    assert "plexiglass" in dsp.WALL_MATERIALS
    assert 20.0 < dsp.wall_rw("plexiglass") < 40.0


def test_shelf_q_default_unchanged_and_active():
    x = np.random.default_rng(0).standard_normal(8192)
    assert np.array_equal(F.highshelf(x, 3000.0, SR, 6.0), F.biquad(x, "highshelf", 3000.0, SR, 0.707, 6.0))
    assert not np.allclose(F.highshelf(x, 3000.0, SR, 6.0, q=1.41), F.highshelf(x, 3000.0, SR, 6.0))


def test_ground_bounce_gain():
    assert abs(float(space.ground_bounce_gain(5.0)) - 0.94) < 0.005
    assert abs(float(space.ground_bounce_gain(1.0)) - 0.486) < 0.002


def test_diode_ladder_poles_and_limits():
    p = F.diode_ladder_poles(0.0, 1.0 / (2 * math.pi))
    assert np.allclose(np.sort(-p.real), [0.07612, 0.61732, 1.38268, 1.92388], atol=1e-4)
    p = F.diode_ladder_poles(F.DIODE_K_MAX, 1.0 / (2 * math.pi))
    assert abs(p.real.max()) < 1e-6 and abs(abs(p.imag).max() - 1.0 / math.sqrt(2.0)) < 1e-6
    assert abs(F.LADDER_LIMITS["lp8"] - 1.884) < 0.001 and abs(F.ladder_k_max(8) - 1.884) < 0.001


def test_stinger_comma():
    assert abs(compose.PYTHAGOREAN_COMMA_ST - 0.2346) < 1e-3
    a = compose._stinger_events({"mood": "victory", "seed": 1})
    b = compose._stinger_events({"mood": "victory", "seed": 1, "comma": True})
    assert len(b) > len(a)


def test_dsf_rms_and_split_bands_flat():
    y = dsp.dsf(44100, 100.0)
    assert abs(float(np.sqrt(np.mean(y * y))) - 0.7071) < 0.01
    imp = np.zeros(1 << 15)
    imp[0] = 1.0
    h = sum(dsp.split_bands(imp, [300.0, 3000.0], SR))
    for f in (100.0, 300.0, 1000.0, 3000.0, 8000.0):
        assert abs(_mag_db(h, f)) < 0.02


def test_iir_families_and_cascade_levels():
    sos = dsp.iir_sos("lowpass", 1000.0, SR, 8, "cheby", ripple=29.0)
    w, h = sg.sosfreqz(sos, [1000.0], fs=SR)
    assert abs(20 * math.log10(abs(h[0])) + 0.035) < 0.02
    sos = dsp.iir_sos("lowpass", 1000.0, SR, 6, "cheby2", atten=60.0)
    assert abs(20 * math.log10(abs(sg.sosfreqz(sos, [1000.0], fs=SR)[1][0])) + 3.01) < 0.1
    imp = np.zeros(1 << 14)
    imp[0] = 1.0
    for order, want in ((1, -3.01), (2, -6.02), (4, -12.04)):
        assert abs(_mag_db(F.biquad(imp, "lowpass", 1000.0, SR, 0.7071, order=order), 1000.0) - want) < 0.3


def test_freqshift_folds_only_on_request():
    n = SR // 2
    x = np.sin(2 * np.pi * 21000.0 * np.arange(n) / SR)
    keep = spectral.freqshift(x, SR, 2000.0, fold=True)
    cut = spectral.freqshift(x, SR, 2000.0, fold=False)
    assert _mag_db(cut, 21100.0) < _mag_db(keep, 21100.0) - 20.0


def test_additive_table_and_thiran_range():
    t = modal.ADDITIVE_TABLES["delayed_bell"]
    assert t["attacks_ms"][:2] == (300, 700) and t["decay_ms"][0] == 9000 and t["delays_ms"][0] == 400 and t["delays_ms"][-1] == 0
    a = graphsolver.thiran_coeffs(2.0, 2)
    w = np.array([1e-3, 2e-3])
    x = np.exp(-1j * w)
    ph = np.unwrap(np.angle(np.polyval(a, x) / np.polyval(a[::-1], x)))   # numerator a_N + ... + a_0 z^-N over 1 + a_1 z^-1 + ...
    assert abs(-(ph[1] - ph[0]) / (w[1] - w[0]) - 2.0) < 0.01


def test_open_hole_shunt_has_radiation_resistance():
    z = winds.open_hole_shunt(1j * 0.3, 1.0, 1.0, 0.5, 1.0, 1.0, 0.3)
    assert z.real > 0.0
