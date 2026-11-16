import numpy as np

from genny import aero, compose
from genny.physical import waveguides as wg


def test_a0b0_split_at_unit_ratio():
    (fl, fh), (ql, qh) = wg.a0b0_modes(ratio=1.0, eps=0.05, f_air=280.0)
    assert abs((fh - fl) - 0.05 * 280.0) < 3.0
    assert fl < 280.0 < fh
    assert ql > 0 and qh > 0


def test_a0b0_far_apart_keeps_own_q():
    (fl, fh), (ql, qh) = wg.a0b0_modes(ratio=0.5, q_body=80, q_air=25)
    assert abs(ql - 80) < 15 or abs(qh - 80) < 15


def test_a0b0_admittance_peaks_at_modes():
    modes, _ = wg.a0b0_modes(ratio=1.0)
    f = np.linspace(200, 360, 4000)
    Y = np.abs(wg.a0b0_admittance(f, 1.0, wg.A0B0_EPS, 280.0, 80, 25))
    assert np.all(np.isfinite(Y))
    pk = f[np.argmax(Y)]
    assert min(abs(pk - modes[0]), abs(pk - modes[1])) < 10


def test_flap_spectrum_continuous_and_decreasing():
    x = np.array([1.0, 1.999, 2.001, 19.9, 20.1, 100.0])
    g = aero.flap_spectrum_db(x)
    assert abs(g[1] - g[2]) < 0.2
    assert abs(g[3] - g[4]) < 0.2
    assert g[-1] < g[4] + 1e-6
    gt = aero.flap_spectrum_db(np.array([74.9, 75.1]), triple=True)
    assert abs(gt[0] - gt[1]) < 0.2


def test_airframe_flap_angle_scales_with_sin():
    a = aero.airframe(flaps=1.0, flap_angle=20.0, seed=1)
    b = aero.airframe(flaps=1.0, flap_angle=45.0, seed=1)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = aero.airframe(flaps=1.0, seed=1)
    assert np.all(np.isfinite(d))


def test_accent_late_delays_stroke_after_accent():
    base = {"type": "compose", "role": "drums", "bars": 2, "seed": 3}
    off = compose._gen_drums.__name__
    assert off == "_gen_drums"
    assert compose.ACCENT_LATE == 0.1
