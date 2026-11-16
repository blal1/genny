"""Lot L-bis: Lecture profonde 3.3 lines 1034-1230 (gong triad threshold, plectrum junction, vinyl wow, ptr engine layouts, tonehole cutoff)."""
import math

import numpy as np

from genny.acoustic import harpsichord
from genny.graphsolver import PlectrumJunction
from genny.percussion import triad
from genny.physical.impedance import TONEHOLE_FC, clarinet
from genny.physical.winds import lattice_cutoff


SR = 44100


def _peak_hz(y, lo=100.0, hi=3000.0, sr=SR):
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 17))
    fr = np.fft.rfftfreq(1 << 17, 1 / sr)
    m = (fr > lo) & (fr < hi)
    return float(fr[m][np.argmax(sp[m])])


def test_triad_threshold_includes_the_detuning():
    # a3* = 2 w1 sqrt(4 mu^2 + d^2) / alpha, with alpha set so that a3* = threshold: below it the twins stay silent, above they grow
    f3, t60 = 113.0, 6.0
    n = 3 * SR
    low, _ = triad(f3, n, SR, 0.5 * 0.05, t60)
    high, _ = triad(f3, n, SR, 4.0 * 0.05, t60)
    assert np.max(np.abs(low)) < 0.02 * np.max(np.abs(high))
    # the detuned threshold is above the exact-resonance 4 mu w1 / alpha: alpha is smaller than before by the factor sqrt(1 + (d / 2 mu)^2)
    w3 = 2 * math.pi * f3
    mu = 6.91 / t60
    d = w3 * 0.001
    assert math.sqrt(4 * mu * mu + d * d) > 2 * mu


def test_plectrum_releases_at_f_max_and_misses_the_node():
    f0, beta = 261.63, 0.2
    j = PlectrumJunction(k=1.2e6, mu=0.05, f_max=0.5, feather_p=2.0, R=0.22, sr=SR)
    y, t_rel, f_pk = j.pluck(f0, 1.0, 1.0, beta)
    assert 0.002 < t_rel < 0.2 and abs(f_pk - 0.5) < 0.01                    # let go as the force reaches f_max
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    amp = [float(sp[np.argmin(abs(fr - f0 * m)) - 3:np.argmin(abs(fr - f0 * m)) + 4].max()) for m in range(1, 7)]
    assert amp[4] < 0.1 * max(amp)                                          # partial 5 sits on the plucking node (beta = 1/5)
    assert abs(_peak_hz(y, 100.0, 1200.0) / (2 * f0) - 1.0) < 0.01          # the bridge force emphasises partial 2
    j2 = PlectrumJunction(k=1.2e6, mu=0.05, f_max=1e-4, feather_p=2.0, R=0.22, sr=SR)
    y2, t2, _ = j2.pluck(f0, 0.5, 1.0, beta)
    assert t2 < 0.01 and np.max(np.abs(y2)) < 1e-3 * np.max(np.abs(y))      # a plectrum that lets go at once leaves no note


def test_plectrum_vel_changes_release_time():
    j = PlectrumJunction(k=1.2e6, mu=0.05, f_max=0.5, feather_p=2.0, R=0.22, sr=SR)
    _, t_soft, _ = j.pluck(261.63, 0.5, 0.0, 0.2)
    _, t_hard, _ = j.pluck(261.63, 0.5, 1.0, 0.2)
    assert t_soft > 2.0 * t_hard                                             # a slow jack takes longer to bend the quill to its limit


def test_harpsichord_plectrum_option_keeps_pitch_and_default():
    a = harpsichord(261.63, 0.6, SR, 1.0)
    b = harpsichord(261.63, 0.6, SR, 1.0, plectrum=1)
    assert np.isfinite(b).all() and a.shape == b.shape
    assert abs(_peak_hz(a, 100.0, 400.0) / 261.63 - 1.0) < 0.01 and abs(_peak_hz(b, 100.0, 400.0) / 261.63 - 1.0) < 0.01
    assert np.array_equal(a, harpsichord(261.63, 0.6, SR, 1.0, plectrum=0))


def test_vinyl_wow_warps_time_once_per_turn():
    from genny.sfx import vinyl as v
    f = getattr(v, "fn", v)
    flat = f(SR, 4.0, 14.0, 0.3, 0, 0.0, 0.0)
    wow = f(SR, 4.0, 14.0, 0.3, 0, 0.0, 0.005)
    assert flat.shape == wow.shape and not np.array_equal(flat, wow)
    turn = 100.0 / 3.0 / 60.0
    t = np.arange(len(flat)) / SR
    tau = t + 0.005 / (2 * math.pi * turn) * (1 - np.cos(2 * math.pi * turn * t))
    assert abs(np.max(np.gradient(tau, t)) - 1.005) < 1e-4 and abs(np.min(np.gradient(tau, t)) - 0.995) < 1e-4   # +-0.5 % of speed


def test_ptr_engine_layout_uses_firing_angles():
    from genny.engines import firing_offsets
    from genny.vehicle import car_engine
    f = getattr(car_engine, "fn", car_engine)
    base = f(SR, 1500, 8, 0.5, 0.25, 0.004, 1.0, 0.0, 7, 1.0, "ptr")
    cross = f(SR, 1500, 8, 0.5, 0.25, 0.004, 1.0, 0.0, 7, 1.0, "ptr", "auto", 0.0, 0.0, "filter", "v6_90")
    assert np.isfinite(cross).all() and cross.shape == base.shape and not np.allclose(cross, base)
    off = firing_offsets("v6_90")
    assert len(off) == 6 and abs(off[1] - 0.125) < 1e-9                      # 90 / 150 degree firing of a 90 degree V6, not the even 120


def test_tonehole_cutoff_at_2khz_does_not_squeak():
    assert TONEHOLE_FC == 1500.0
    assert 1200.0 < lattice_cutoff(7e-3, 3.5e-3, 0.03, 0.006) < 2200.0
    for f0 in (293.66, 659.26, 739.99):
        y = clarinet(f0, 0.7, SR, tonehole_fc=2100.0)
        assert abs(_peak_hz(y[int(0.15 * SR):int(0.6 * SR)]) / f0 - 1.0) < 0.01     # the reed does not lock onto its own 2.2 kHz


def test_fdn_mode_density_and_prime_powers():
    from genny.reverbs import fdn_lengths, prime_power_lengths
    M = fdn_lengths(48000, 6.0, 8, t60=2.0)
    assert M.sum() >= 0.15 * 2.0 * 48000 - 8                                  # t60 = 2 s at 48 kHz needs about 14400 samples of line
    P = prime_power_lengths(fdn_lengths(48000, 6.0, 8))
    assert all(math.gcd(int(a), int(b)) == 1 for i, a in enumerate(P) for b in P[i + 1:])


def test_longitudinal_coupling_is_tan_of_the_bridge_angle():
    from genny.physical.bank_piano import long_coupling
    assert abs(long_coupling(2.0) - math.tan(math.radians(2.0))) < 1e-12 and long_coupling(None) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
