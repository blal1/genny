"""LU9_02c: options added from the documents of group 02 read again in full, second pass (2026-11-15)."""
import math

import numpy as np

from genny import acoustic, analysis as A, contact, cyber, dsp, filters, machines, matter, sports
from genny.physical import nonverbal

SR = 44100


def test_fdn_polynomial_leading_term():
    m = [7, 13, 19]
    n = np.array([5000.0, 10000.0])
    lead = n ** 2 / (2.0 * math.prod(m))
    p = dsp.fdn_echo_density_polynomial(m, n)
    assert np.all(np.abs(p / lead - 1.0) < 0.01)


def test_fdn_degeneration_flags():
    assert dsp.fdn_degeneration([307, 557, 907, 1021])["order"] == 3
    d = dsp.fdn_degeneration([100, 150, 250])
    assert d["common_factor"] == 50 and not d["coprime"]
    assert dsp.fdn_degeneration([1000, 1010, 1020])["geometric_std"] < 1.2


def test_fdn_polack_mixing_and_delays():
    t = dsp.fdn_mixing_time([307, 557, 907, 1021, 1327, 1601, 1999, 2309], SR, kind="polack")
    assert 0.0 < t < 1.0
    d = dsp.fdn_delays_for_mixing(0.05, lines=8, sr=SR)
    assert d.size == 8 and dsp.fdn_degeneration(d)["coprime"]


def test_ladder_q_roundtrip_and_limits():
    for n, q in ((4, 3.0), (5, 5.0), (8, 10.0)):
        k = filters.ladder_k_from_q(n, q)
        assert 0.0 < k < 1.0 / math.cos(math.pi / n) ** n
    assert filters.ladder_pole_ratio(4, 0.0) == 1.0
    assert abs(filters.ladder_dc_gain(1.0) - 0.5) < 1e-12
    assert filters.ladder_cutoff_error(4, 4.0) < 1e-6


def test_ladder_pole_cutoff_changes_only_when_asked():
    x = np.random.default_rng(1).standard_normal(4000)
    a = filters.ladder(x, 2000.0, SR, 0.7, kind="dangelo")
    b = filters.ladder(x, 2000.0, SR, 0.7, kind="dangelo", pole_cutoff=True)
    assert a.shape == b.shape and not np.allclose(a, b)


def test_terhardt_noise_excluding_tonals():
    f = np.arange(0, 4000, 10.0)
    s = np.full(f.size, 10.0)
    s[100] = 70.0
    nz = A.tonal_noise_db(s, 10.0, [1000.0])
    assert nz.shape == (1,) and nz[0] < 40.0


def test_lfcc_and_lpc_cepstrum():
    y = np.random.default_rng(2).standard_normal(SR // 2)
    assert A.lfcc_davis(y, SR, 8).shape[1] == 8
    c = A.lpc_cepstrum([-0.9], 3)
    assert abs(c[0] + 0.9) < 1e-12 and abs(c[1] + 0.405) < 1e-9


def test_induction_motor_drive_lines_and_default():
    a = cyber.induction_motor(SR, 1.0, seed=0)
    b = cyber.induction_motor(SR, 1.0, drive_hz=3000.0, seed=0)
    assert a.shape == b.shape and not np.allclose(a, b)
    assert np.allclose(a, cyber.induction_motor(SR, 1.0, drive_hz=0.0, seed=0))


def test_bubble_thermal_full_matches_uniform_form():
    r0, w = 1.0e-3, 2 * math.pi * 3000.0
    u = matter.bubble_thermal(r0, w)
    f = matter.bubble_thermal(r0, w, full=True)
    assert np.allclose(np.asarray(f), np.asarray(u), rtol=0.02)


def test_hertz_elliptic_circular_is_exact():
    a, b, d, p0 = contact.hertz_elliptic(100.0, 0.01, 0.01, 1e11, fit="regression")
    a2, b2, d2, p02 = contact.hertz_elliptic(100.0, 0.01, 0.01, 1e11)
    assert abs(a / b - 1.0) < 1e-9 and abs(a / a2 - 1.0) < 1e-3


def test_sphere_cavity_first_modes():
    m = sports.sphere_cavity_modes(0.116, 24, 343.0)
    assert len(m) == 24
    hz = [x[0] for x in m]
    assert hz == sorted(hz) and abs(hz[0] - 980.0) < 3.0 and abs(hz[1] - 1573.0) < 3.0


def test_gear_rolling_force_and_efficiency():
    f = machines.gear_rolling_force(10.0, 2e5, 0.02, 2.3e11, 2.2e-8, 0.05)
    assert f > 0.0
    assert machines.gear_rolling_force(20.0, 2e5, 0.02, 2.3e11, 2.2e-8, 0.05) > f
    assert machines.gear_rolling_force(10.0, 2e5, 0.02, 2.3e11, 2.2e-8, 0.05, phi_t=0.9) < f
    eta = machines.gear_mesh_efficiency([5.0], [1.0], [1.0], [20.0], 10.0, 100.0)
    assert 0.9 < eta < 1.0


def test_laugh_calls_pattern_and_formants():
    s, ln = nonverbal.laugh_calls(5)
    assert ln[0] > ln[1] == ln[-1] and np.all(np.diff(s) > 0)
    gaps = s[1:] - (s[:-1] + ln[:-1])
    assert gaps[0] < gaps[-1] and abs(gaps.mean() - nonverbal.LAUGH_GAP_S) < 1e-9
    assert nonverbal.LAUGH_FORMANTS["male"][0] < nonverbal.LAUGH_FORMANTS["female"][0]


def test_flue_harmonics_between_the_two_mouths():
    lo, hi = acoustic.flue_harmonics(0.0), acoustic.flue_harmonics(1.0)
    mid = acoustic.flue_harmonics(0.5)
    assert lo[1] > hi[1] and lo[1] > mid[1] > hi[1] and mid[0] == 1.0
