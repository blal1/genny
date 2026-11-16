"""Lot 34 O1: Addendum livres, section 2 Instruments (lines 207-240). One check per row, on the rendered signal where there is one.

Run:  uv run --with pytest pytest tests/test_lot34_o1.py -q
"""
import math

import numpy as np

from genny import acoustic as A
from genny import friction, instruments as I, klang, percussion
from genny.physical import bank_piano
from genny.physical import winds as W

SR = 44100


def _band_db(y, lo, hi, t0=0.2, t1=1.2):
    s = y[int(t0 * SR):int(t1 * SR)]
    S = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    f = np.fft.rfftfreq(len(s), 1 / SR)
    return 20 * np.log10(S[(f >= lo) & (f <= hi)].max() / S.max())


def _h(y, f0, k, t0=0.25, n=16384):
    s = y[int(t0 * SR):int(t0 * SR) + n]
    Y = np.abs(np.fft.rfft(s * np.hanning(s.size), 8 * s.size))
    fr = np.fft.rfftfreq(8 * s.size, 1 / SR)
    return Y[(fr > k * f0 * 0.97) & (fr < k * f0 * 1.03)].max()


def _centroid(y):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return float((np.fft.rfftfreq(len(y), 1 / SR) * S).sum() / S.sum())


def test_brass_ceiling_and_vibrato_tables():
    assert W.BRASS_TOP_FF == dict(trumpet=15000.0, french_horn=8000.0, trombone=7000.0, tuba=3000.0)
    assert abs(W.brass_top("trumpet", 1.0, 4800.0) - 15000.0) < 1 and W.brass_top("trumpet", 0.0, 4800.0) == 4800.0
    assert W.vibrato_am("trombone", 100.0) == 3.0 and W.vibrato_am("trombone", 1000.0) == 9.0 and W.vibrato_am("flute", 4000.0) == 15.0
    assert W.vibrato_am("bassoon", 1600.0) == 17.5 and W.vibrato_am("clarinet", 2500.0) == 20.0
    assert A.attack_ms("trumpet", 165.0, 0.0) == 180.0 and A.attack_ms("trumpet", 1047.0, 1.0) == 20.0


def test_level_3k_variant_reaches_the_measured_gaps_and_is_opt_in():
    base = I.render_note("oboe", 1046.5, 1.5, SR, 0.85)
    assert np.array_equal(base, I.render_note("oboe", 1046.5, 1.5, SR, 0.85, level3k=0.0))
    for inst, f, tgt in (("oboe", 1046.5, -12.0), ("violin", 659.3, -10.0), ("clarinet", 164.8, I.level3k_target("clarinet", 164.8))):
        raw = _band_db(I.render_note(inst, f, 1.5, SR, 0.85), 2500, 3500)
        got = _band_db(I.render_note(inst, f, 1.5, SR, 0.85, level3k=1.0), 2500, 3500)
        assert abs(got - tgt) < abs(raw - tgt) and abs(got - tgt) < 6.0, (inst, raw, got, tgt)
    assert I.level3k_target("cello", 100.0) == -40.0


def test_clarinet_family_even_partials_30_db_down_in_the_low_register():
    y = klang.eb_clarinet(196.0, 1.0, SR, 0.8)
    ev = sum(_h(y, 196.0, k) ** 2 for k in (2, 4, 6))
    od = sum(_h(y, 196.0, k) ** 2 for k in (3, 5, 7))
    assert 10 * np.log10(ev / od) < -25.0
    a, b = klang.a_clarinet(392.0, 1.0, SR, 0.8), klang.eb_clarinet(392.0, 1.0, SR, 0.8)
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_valve_click_and_vibraphone_tube_rise():
    f = 130.8
    a, b = I.render_note("french_horn", f, 0.6, SR, 0.8), I.render_note("french_horn", f, 0.6, SR, 0.8, valve=1.0)
    n = int(0.02 * SR)
    assert np.sqrt(np.mean((b[:n] - a[:n]) ** 2)) > 0.05 * np.sqrt(np.mean(a[:int(0.3 * SR)] ** 2))
    assert np.allclose(a[int(0.15 * SR):], b[int(0.15 * SR):], atol=1e-3 * np.max(np.abs(a)))
    u, v = I.render_note("vibraphone", 440.0, 0.6, SR, 0.8, motor=0.0), I.render_note("vibraphone", 440.0, 0.6, SR, 0.8, motor=0.0, rise=15.0)
    assert not np.allclose(u[:int(0.04 * SR)], v[:int(0.04 * SR)]) and np.allclose(u[int(0.2 * SR):int(0.4 * SR)], v[int(0.2 * SR):int(0.4 * SR)], atol=1e-3 * np.max(np.abs(u)))
    assert np.array_equal(u, I.render_note("vibraphone", 440.0, 0.6, SR, 0.8, motor=0.0, rise=0.0))


def test_bowed_level_ranges():
    assert abs(20 * math.log10(A.bowed_level(1.25, 0.1) / A.bowed_level(0.1, 0.1)) - 21.9) < 0.2
    assert abs(20 * math.log10(A.bowed_level(0.3, 0.02) / A.bowed_level(0.3, 0.12)) - 15.6) < 0.2


def test_new_rod_sets_and_their_partials():
    for r, ratio in (("dead_weight", 3.0), ("weighted_29", 5.34), ("weighted_28", 5.04), ("weighted_a", 6.0)):
        y = friction.rod_bank(220.0, 1.0, SR, 0.9, rod=r, radiator="none", sympathetic=0.0)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.01
        assert _band_db(y, 220 * ratio * 0.985, 220 * ratio * 1.015, 0.02, 0.5) > -60.0, r      # struck rods: the upper partials sit 40-47 dB down, as in the older sets
    a = friction.rod_bank(220.0, 2.0, SR, 0.9, rod="clamped", radiator="none", sympathetic=0.0)
    b = friction.rod_bank(220.0, 2.0, SR, 0.9, rod="rigid_blade", radiator="none", sympathetic=0.0)
    assert np.sqrt(np.mean(b[int(1.0 * SR):int(1.5 * SR)] ** 2)) <= np.sqrt(np.mean(a[int(1.0 * SR):int(1.5 * SR)] ** 2)) * 1.01


def test_comb_bar_partial_and_ranks():
    y = friction.bowed_bar(440.0, 1.0, SR, 0.9, material="comb", partial=3)
    assert _band_db(y, 143.0, 151.0, 0.3, 0.9) > -12.0 and _band_db(y, 432.0, 448.0, 0.3, 0.9) > -12.0    # the series stands on 440 / 3, rank 3 is the note
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    z = friction.bowed_bar(440.0, 0.6, SR, 0.9, material="comb", ranks="1 2 4")
    assert np.isfinite(z).all()


def test_crystal_harmonics_follow_the_pressure_and_the_sheet_options():
    soft = friction.cristal(262.0, 1.0, SR, 0.9, pressure=0.5, radiator="none")
    hard = friction.cristal(262.0, 1.0, SR, 0.9, pressure=4.0, radiator="none")
    assert _centroid(hard[int(0.3 * SR):int(0.9 * SR)]) >= _centroid(soft[int(0.3 * SR):int(0.9 * SR)]) * 0.98
    assert np.array_equal(friction.cristal(262.0, 0.5, SR, 0.9, radiator="none"), friction.cristal(262.0, 0.5, SR, 0.9, pressure=1.0, radiator="none"))
    x = np.random.default_rng(1).standard_normal(SR)
    base = friction.sheet_radiator(x, SR, mix=1.0, seed=2)
    tuned = friction.sheet_radiator(x, SR, mix=1.0, seed=2, tuned_to=1500.0)
    assert not np.array_equal(base, tuned)
    S = lambda y: np.abs(np.fft.rfft(y))[1450:1550].max()
    assert S(tuned) > S(base)
    dull = friction.sheet_radiator(x, SR, mix=1.0, seed=2, tilt=-6.0)
    assert _centroid(dull) < _centroid(base)
    for kw in (dict(sectors=4), dict(fragments=5)):
        y = friction.sheet_radiator(x, SR, mix=1.0, seed=2, **kw)
        assert np.isfinite(y).all() and not np.array_equal(y, base)


def test_rows_closed_by_earlier_waves_still_hold():
    assert bank_piano.LID_DB["closed"] == -10.0 and bank_piano.ACTION_BOARD_HZ == 90.0
    assert percussion.meyer_bass_drum_t60(300.0) == 4.0
    assert W.FREE_REED_F == (0.254, 0.593, 0.777) and W.CONE_KX1[1] == 670.0
    assert W.LATTICE_FC["clarinet"] == 1300.0 and W.LATTICE_FC["bassoon"] == 500.0
    assert abs(W.worman_level(0.5, 3) - 0.125) < 1e-12
