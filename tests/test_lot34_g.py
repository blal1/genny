"""Lot 34 G: spaces, propagation, spatialisation (values in research/SOURCES.md)."""
import math

import numpy as np

from genny import cyber, foley, friction, physics, reverbs, tubes
from genny.physical import modal
from genny.physical import space

SR = 44100


def _peak_hz(x, sr):
    return float(np.argmax(np.abs(np.fft.rfft(x))) * sr / len(x))


def test_pipe_engine_baldan_transmission_and_ignition_time():
    a = cyber.pipe_engine(SR, dur=0.4, rpm=2400, muffler_r=-0.5)
    b = cyber.pipe_engine(SR, dur=0.4, rpm=2400, muffler_r=-0.5, transmission="baldan")
    assert np.isfinite(b).all() and not np.allclose(a, b)
    c = cyber.pipe_engine(SR, dur=0.4, rpm=2400, ignition_time=0.05)
    assert np.isfinite(c).all() and not np.allclose(a, c)


def test_blast_wave_altitude_scaling():
    p0, t0, a0 = physics.blast_wave(1.0, 10.0)
    p1, t1, a1 = physics.blast_wave(1.0, 10.0, "tnt", 70.0, 5.0)
    assert a0 == a1
    assert p1 < p0 and t1 > t0
    assert abs(physics.blast_wave(1.0, 10.0, "tnt", 101.325, 15.0)[0] - p0) < 1e-9


def test_rotor_tip_doppler_spreads_the_blade_tone():
    r = physics.Rotor(rpm=9000.0, blades=2, radius_m=0.9, tip_doppler=True)
    y0 = physics.Rotor(rpm=9000.0, blades=2, radius_m=0.9).render(1.0, sr=SR, seed=1)
    y1 = r.render(1.0, sr=SR, seed=1)
    assert np.isfinite(y1).all() and not np.allclose(y0, y1)


def test_membrane_air_shift_lowers_axisymmetric_modes_most():
    d = physics.MembraneDrum(150.0, 0.3, air_shift=0.4)
    m0 = physics.MembraneDrum(150.0, 0.3)._modes(0.25, "tone")
    m1 = d._modes(0.25, "tone")
    r = np.asarray(m1.freqs) / np.asarray(m0.freqs)
    assert (r < 1.0).all() and r[0] < r[1] and r[3] < r[4]


def test_reed_temperature_raises_pitch_by_the_speed_of_sound():
    assert abs(tubes.air(30.0)["c"] / tubes.air(20.0)["c"] - 1.0169) < 1e-3
    x = np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    f = _peak_hz(tubes.warm(x, 35.0), SR)
    assert abs(1200 * math.log2(f / 440.0) - 43.0) < 6.0


def test_vowel_tube_wall_is_exposed():
    a = tubes.vowel_tube(110.0, 0.4, SR, wall=0.0)
    b = tubes.vowel_tube(110.0, 0.4, SR, wall=2.0)
    assert np.isfinite(a).all() and np.isfinite(b).all() and not np.allclose(a[: len(b)], b[: len(a)])


def test_spinning_blades_and_fork_doppler_peak_deviation():
    sr = SR
    x = np.sin(2 * np.pi * 3000.0 * np.arange(2 * sr) / sr)
    y = friction._spin_blades(x, sr, 2.0, doppler=True)
    assert np.isfinite(y).all()
    z = friction.tuning_fork(440.0, 0.5, sr, spin=2.0)
    assert np.isfinite(z).all() and len(z) > 0
    dev = 2 * math.pi * 2.0 * 0.3 / 343.0
    assert abs(dev - 0.011) < 0.001


def test_edge_track_brown_fallback_keeps_the_tone_where_the_fit_goes_negative():
    n = 2000
    mode, st, f, *_ = modal.edge_track(np.full(n, 30.0), np.full(n, 0.01), 0.0005)
    assert (st >= 0).all() and np.isfinite(f).all()


def test_room_src_beta_baroque_transition_and_modal_density():
    x = np.zeros(SR)
    x[0] = 1.0
    omni = reverbs.room(x, SR, mix=1.0, preset="hall", distance=8.0, src_beta=1.0)
    card = reverbs.room(x, SR, mix=1.0, preset="hall", distance=8.0, src_beta=0.5)
    assert np.sum(card[200:] ** 2) < np.sum(omni[200:] ** 2)
    t60 = reverbs.room_info("baroque")["t60"]
    assert 1.2 < float(np.mean(t60[2:4])) < 1.8
    y = reverbs.room(x, SR, mix=1.0, preset="hall", transition=5000)
    assert np.isfinite(y).all()
    v = 6000.0
    assert abs(reverbs.transition_time(v, 2000.0) - math.sqrt(2000.0 * v / (4 * math.pi * reverbs.C_AIR ** 3))) < 1e-9
    assert abs(space.modal_density(100.0, 100.0) - 4 * math.pi * 100.0 * 1e4 / space.C_AIR ** 3) < 1e-9
    assert reverbs.room_info("hall")["modes_per_hz_schroeder"] > 0


def test_radio_chatter_is_deterministic_and_hit_marker_has_both_tones():
    a, b = cyber.radio_chatter(SR, variant=3), cyber.radio_chatter(SR, variant=3)
    assert np.array_equal(a, b) and np.isfinite(a).all()
    assert len(cyber.radio_chatter(SR, variant=4)) > len(cyber.radio_chatter(SR, variant=0))
    h = foley.hit_marker(SR)
    assert abs(len(h) / SR - 0.05) < 0.01
    s = np.abs(np.fft.rfft(h, 1 << 15))
    f = np.fft.rfftfreq(1 << 15, 1.0 / SR)
    assert s[(f > 1800) & (f < 2000)].max() > 2.0 * s[(f > 2800) & (f < 2900)].max() > 0.0
