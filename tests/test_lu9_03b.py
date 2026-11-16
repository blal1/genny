import math
from types import SimpleNamespace as NS

import numpy as np

from genny import compose as C
from genny import fdstring as FD
from genny import matter as MT
from genny import nature as NA
from genny import percussion as PC
from genny.physical import voice as VO
from genny.piano import (PianoHammerAction, critical_compliance, hard_hammer_contact_time,
                         heavy_hammer_energy_transfer, soft_hammer_critical_mode)


def _ev(ts, ps, dur=1.0):
    return [{"t": t, "dur": dur, "pitch": p, "vel": 0.5} for t, p in zip(ts, ps)]


def test_bar_t60_padouk_and_rosewood_fluid_term():
    assert math.isclose(FD.bar_t60("rosewood_bar", 440.0), 0.259, abs_tol=0.01)
    assert math.isclose(FD.bar_t60("rosewood_bar", 4000.0), 0.069, abs_tol=0.01)
    assert FD.bar_t60("padouk", 440.0) > FD.bar_t60("padouk", 2000.0)
    assert FD.bar_t60("rosewood_bar", 440.0) < FD.bar_t60("padouk", 440.0)


def test_hertz_contact_time_rubber_boxwood():
    r = FD.hertz_contact_time(*FD.MALLET_HERTZ["rubber"], 1.5)
    b = FD.hertz_contact_time(*FD.MALLET_HERTZ["boxwood"], 1.5)
    assert math.isclose(r, 0.62e-3, rel_tol=0.05) and math.isclose(b, 0.156e-3, rel_tol=0.05)
    assert FD.mallet_hardness("boxwood") > FD.mallet_hardness("rubber")


def test_todd_shape_keeps_length_fastest_at_turn():
    c = NS(q=0.1, meter=4.0, sections=[])
    a = _ev([float(i) for i in range(11)], (60.0,) * 11)
    C._kth(a, c, {"phrase_arch": 1.0, "phrase_shape": "todd"})
    t = np.array([e["t"] for e in a])
    d = np.diff(t)
    assert math.isclose(t[-1], 10.0) and d.argmin() == 7 and d[0] > d[-1] > d[7]


def test_kth_1983_rise_shortens_leap_lengthens():
    c = NS(q=0.1, meter=4.0, sections=[])
    a = _ev([float(i) for i in range(6)], (60.0, 62.0, 64.0, 60.0, 72.0, 60.0))
    C._kth(a, c, {"kth_1983": 1.0})
    d = np.diff([e["t"] for e in a])
    assert math.isclose(d[0], 0.97) and math.isclose(d[1], 0.97) and d[4] > 1.0
    assert a[3]["dur"] < 1.0
    b = _ev([float(i) for i in range(6)], (60.0, 62.0, 64.0, 60.0, 72.0, 60.0))
    C._kth(b, c, {})
    assert [e["t"] for e in b] == [float(i) for i in range(6)]


def test_gamelan_gong_modes_and_chladni():
    assert PC.GAMELAN_GONG_MODES[0][0] == (0, 1) and PC.GAMELAN_GONG_MODES[0][1] == 458.0
    assert math.isclose(PC.gamelan_chladni(8, 0), 47.26 * 64)
    assert math.isclose(PC.gamelan_chladni(4, 1), 48.70 * 36)
    assert abs(PC.gamelan_chladni(8, 0) - 3436.0) / 3436.0 < 0.15          # the law fits the n=0 family; the (m, 1) modes sit above it


def test_piano_voicing_and_strike_ratio_defaults_unchanged():
    a = PianoHammerAction(60)
    assert a.voicing is None and a.strike_ratio is None
    soft = PianoHammerAction(60, voicing=5)
    hard = PianoHammerAction(60, voicing=1)
    assert soft.felt_exponent == 3.3 and hard.felt_exponent == 2.3
    soft.strike(0.5)
    hard.strike(0.5)
    assert soft.last_contact_time_s > hard.last_contact_time_s
    near = PianoHammerAction(60, agraffe=True, strike_ratio=0.04)
    far = PianoHammerAction(60, agraffe=True, strike_ratio=0.24)
    assert far.agraffe_delay_s() > 5.0 * near.agraffe_delay_s()


def test_hall_hard_and_soft_hammer_closed_forms():
    tau = hard_hammer_contact_time(2.97e-3, 670.0, 0.0045, 0.12, 0.62)
    assert 1.0e-3 < tau < 4.0e-3
    c = math.sqrt(670.0 / 0.0045)
    assert hard_hammer_contact_time(1e-9, 670.0, 0.0045, 0.12, 0.62) < 2.0 * 0.12 * 0.62 / c * 1.01
    assert math.isclose(heavy_hammer_energy_transfer(1.0, 0.003), 1.0 - (1.0 / (1.0 + 0.001)) ** 2)
    c0 = critical_compliance(2.97e-3, 670.0, 0.0045)
    assert math.isclose(c0, 2.97e-3 / (16 * 0.0045 * 670.0))
    assert soft_hammer_critical_mode(2.97e-3, 0.0028, 4 * c0, 670.0, 0.0045) < soft_hammer_critical_mode(
        2.97e-3, 0.0028, c0, 670.0, 0.0045)


def test_titze_threshold_and_fold_f0():
    assert math.isclose(VO.phonation_threshold_pressure(0.04), 1.47, abs_tol=0.03)
    assert math.isclose(VO.phonation_threshold_pressure(0.16), 5.87, abs_tol=0.03)
    assert math.isclose(VO.phonation_threshold_pressure(0.04, 0.12), VO.phonation_threshold_pressure(0.04, 0.04) * 0.5)
    assert math.isclose(VO.fold_f0(2.0e5, 0.476), 103.0, abs_tol=0.5)


def test_fire_aie_peaks_and_tornado_peak():
    rng = np.random.default_rng(3)
    y = MT._crackles(48000, rng, 30.0, wood=0.0, peaks=MT.FIRE_AIE_PEAKS["spruce"])
    assert np.any(y)
    assert math.isclose(NA.tornado_peak_hz(), 45.0) and math.isclose(NA.tornado_peak_hz(67.5, 0.03), 150.0)
