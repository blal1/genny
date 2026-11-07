import math

import numpy as np

from genny.piano import PianoHammerAction, PianoStringCourse, nonrigid_end_ratio
from genny.physical import footsteps as fs
from genny.physical.bow import bowed, mu_curve_table


def _vel(speed):
    return math.log(speed / 0.5) / math.log(8.0)


def _strike(note, speed, mass_factor=1.0):
    h = PianoHammerAction(note, sr=44100)
    h = PianoHammerAction(note, sr=44100, hammer_mass_kg=mass_factor * h.hammer_mass_kg)
    h.strike(_vel(speed))
    return h.last_contact_time_s, float(h.last_force_n.max())


def test_hammer_contact_and_peak_force_at_2_5_m_s():
    for note, t_want, f_want in ((96, 0.6e-3, 40.0), (60, 1.9e-3, 13.0), (36, 3.25e-3, 16.0)):
        t, f = _strike(note, 2.5)
        assert abs(t - t_want) < 0.25 * t_want, (note, t)
        assert abs(f - f_want) < 0.30 * f_want, (note, f)
    assert abs(_strike(60, 0.5)[1] - 2.0) < 0.6
    assert abs(_strike(60, 4.0)[1] - 22.0) < 4.4


def test_double_hammer_mass_lengthens_the_treble_contact_by_a_third():
    t1, _ = _strike(96, 2.5)
    t2, _ = _strike(96, 2.5, 2.0)
    assert 1.15 < t2 / t1 < 1.45


def test_felt_contact_shortens_with_hammer_speed():
    times = [_strike(60, v)[0] for v in (0.74, 1.00, 1.31)]
    assert times[0] > times[1] > times[2]
    for t, want in zip(times, (2.5e-3, 2.0e-3, 1.7e-3)):
        assert abs(t - want) < 0.20 * want, (t, want)


def test_nonrigid_bridge_shifts_the_note():
    assert nonrigid_end_ratio(262.0, 700.0, 0.62, 0.0) == 1.0
    assert nonrigid_end_ratio(262.0, 700.0, 0.62, 1e-3) < 1.0 < nonrigid_end_ratio(262.0, 700.0, 0.62, -1e-3)
    base = PianoStringCourse(60, sr=22050)
    assert PianoStringCourse(60, sr=22050, bridge_im_admittance=0.0).frequencies == base.frequencies
    soft = PianoStringCourse(60, sr=22050, bridge_im_admittance=1e-3)
    cents = 1200.0 * math.log2(soft.frequencies[0] / base.frequencies[0])
    assert -6.0 < cents < -1.0


def test_stick_squeal_rings_the_torsional_series():
    f = fs.stick_modes(1.2)
    assert 256.0 <= f[0] <= 284.0
    assert np.allclose(f[:4] / f[0], [1, 2, 3, 4], rtol=0.02)
    assert fs.stick_modes(0.3)[0] > 3.5 * f[0]
    y = fs.stick_squeal(1.2, 0.5)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.5
    s = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    fr = np.fft.rfftfreq(y.size, 1.0 / 48000)
    s[fr < 50] = 0
    assert abs(fr[np.argmax(s)] - f[0]) < 0.05 * f[0]
    assert np.max(np.abs(fs.stick_squeal(1.2, 0.3, temperature=0.0))) == 0.0


def test_static_bow_friction_curves():
    v, mu = mu_curve_table("exponential")
    assert abs(mu[0] - 1.2) < 1e-3 and abs(mu[-1] - 0.35) < 1e-3
    v, mu = mu_curve_table("hyperbolic")
    assert abs(mu[0] - 0.8) < 1e-3 and 0.3 < mu[-1] < 0.31
    for curve in ("exponential", "hyperbolic"):
        y = bowed(196.0, 0.3, 44100.0, curve=curve)
        assert np.isfinite(y).all() and np.sqrt(np.mean(y[4410:13230] ** 2)) > 0.1
