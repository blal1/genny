import math

import numpy as np
import pytest

from genny import acoustic as A
from genny import strings as S
from genny import fdstring
from genny import piano as P
from genny.physical import piano_data as PD
from genny.physical import sbp
from genny.physical import winds as W

SR = 22050


def _peak_hz(y, sr=SR, lo=20.0, hi=4000.0):
    y = np.asarray(y, float)
    seg = y[len(y) // 4:]
    Y = np.abs(np.fft.rfft(seg * np.hanning(seg.size), 1 << 18))
    f = np.fft.rfftfreq(1 << 18, 1.0 / sr)
    m = (f > lo) & (f < hi)
    return float(f[m][np.argmax(Y[m])])


def _ok(y):
    y = np.asarray(y, float)
    assert y.size > 0 and np.all(np.isfinite(y)) and np.max(np.abs(y)) > 1e-4


# --- strings: losses, finger, pickups, tonestack, radiation -------------------

def test_worn_steel_decays_faster():
    # Table I on the A string: the worn string's internal losses (eta_F, eta_B) are higher but its air term eta_A
    # lower (0.6 against 1.2), so it decays faster only above ~700 Hz (partial 7 on)
    fk = 110.0 * np.arange(1, 20)
    assert np.all(A.woodhouse_tau(110.0, fk, "steel_worn")[6:] < A.woodhouse_tau(110.0, fk, "steel")[6:])


def test_plain_string_keeps_fundamental_t60():
    props = S.StringPhysicalProperties.from_frequency(196.0, material="nylon")
    fk = 196.0 * np.arange(1, 30)
    tau = A.plain_string_tau(fk, 3.0, props)
    assert tau[0] == pytest.approx(3.0 / 6.91, rel=1e-6)
    assert np.all(np.diff(tau) <= 0)


def test_finger_width_cutoff():
    c = 2 * 0.65 * 329.6
    fc = c / (math.pi * A.FINGER_WIDTH)
    assert fc == pytest.approx(9.09e3, rel=0.01)
    x = np.zeros(4096)
    x[0] = 1.0
    H = np.abs(np.fft.rfft(A.finger_width_filter(x, SR, c), 4096))
    f = np.fft.rfftfreq(4096, 1.0 / SR)
    i = np.argmin(np.abs(f - fc))
    assert H[i] == pytest.approx(math.exp(-1.0), rel=0.05)


def test_pickup_mix_out_of_phase_same_place_cancels():
    y = np.random.default_rng(0).standard_normal(4000)
    z = A.pickup_mix(y, SR, 110.0, 0.2, 0.2, (82.4, 110.0), phase="out")
    assert np.max(np.abs(z)) < 1e-9


def test_dispersive_comb_magnitude():
    x = np.zeros(8192)
    x[0] = 1.0
    B, pos, fo = 1e-4, 0.2, 110.0
    H = np.abs(np.fft.rfft(A.pickup_comb_dispersive(x, SR, fo, pos, (fo,), B), 8192)) ** 2
    f = np.fft.rfftfreq(8192, 1.0 / SR)
    t3 = 0.5 * pos / fo
    tau = 1.0 / np.sqrt(1.0 + B * (f / fo) ** 2)
    ref = 2.0 - 2.0 * np.cos(2 * np.pi * f * 2 * t3 * tau)
    assert np.allclose(H[10:2000], ref[10:2000], atol=0.05)


def test_tonestack_mid_scoop():
    from scipy.signal import freqz
    b, a = A.tonestack_coeffs(48000)
    w, h = freqz(b, a, [100.0, 500.0, 5000.0], fs=48000)
    g = 20 * np.log10(np.abs(h))
    assert g[1] < g[0] and g[1] < g[2]


def test_guitar_radiation_weights():
    f = np.array([e[0] for e in A.GUITAR_EFFICIENCY])
    eff = np.array([e[1] for e in A.GUITAR_EFFICIENCY])
    w = np.sqrt(eff / eff.mean())
    assert w[0] == pytest.approx(math.sqrt(40.8 / eff.mean()))
    assert f[0] == 200.0


# --- Rhodes, clavinet, harpsichord --------------------------------------------

def test_rhodes_contact_time():
    assert A.rhodes_contact_time(4.0) == pytest.approx(5.7e-4, rel=0.15)
    assert A.rhodes_contact_time(1.0) > A.rhodes_contact_time(4.0)


def test_tonebar_length():
    assert A.tonebar_length(293.7, coef=0.346) == pytest.approx(0.0821, rel=0.03)
    assert A.tonebar_length(293.7) == pytest.approx(0.090, rel=0.05)


def test_rhodes_ratios_and_damper():
    assert A.rhodes_ratios(43.65) == pytest.approx((7.11, 20.25))
    L = A.rhodes_tine_length(440.0)
    m = A.STEEL_RHO * math.pi * A.TINE_D ** 2 / 4.0 * L
    assert A.rhodes_damper_tau(440.0) == pytest.approx(max(2 * 0.2427 * m / A.RHODES_DAMPER_R, 2 / 440.0))


def test_beam_kirchhoff_shift():
    assert A.beam_kirchhoff_shift(1e-3, 4e-3) == pytest.approx(3.0 / 32.0)


def test_harpsichord_pluck_by_register():
    lo, hi = A.harpsichord_pluck_point(65.0), A.harpsichord_pluck_point(1200.0)
    assert {round(lo, 3), round(hi, 3)} == {0.12, 0.5}


@pytest.mark.parametrize("fn,f0", [(A.rhodes, 220.0), (A.clav, 220.0), (A.harpsichord, 220.0), (A.kantele, 196.0),
                                   (A.tanpura, 130.8), (A.tromba_marina, 130.8), (A.bowed_psaltery, 392.0),
                                   (A.resonator_guitar, 196.0), (A.whirl_string, 110.0), (A.guitar_chord, 110.0)])
def test_new_and_rewritten_instruments_pitch(fn, f0):
    y = fn(f0, 1.0, SR, 0.8)
    _ok(y)
    p = _peak_hz(y)
    k = round(p / f0)
    assert k >= 1 and abs(p / (k * f0) - 1.0) < 0.03


@pytest.mark.parametrize("fn", [A.string_scrape, A.zanza, A.baschet])
def test_new_noisy_instruments_render(fn):
    _ok(fn(220.0, 0.6, SR, 0.8))


def test_clav_pickup_c_is_difference():
    a = A.clav(220.0, 0.4, SR, 0.8, pickup="A")
    b = A.clav(220.0, 0.4, SR, 0.8, pickup="B")
    c = A.clav(220.0, 0.4, SR, 0.8, pickup="C")
    assert not np.allclose(a, c) and not np.allclose(b, c)


def test_bowed_level():
    assert 20 * math.log10(A.bowed_level(1.25, 0.1) / A.bowed_level(0.1, 0.1)) == pytest.approx(21.9, abs=0.2)
    assert 20 * math.log10(A.bowed_level(0.3, 0.02) / A.bowed_level(0.3, 0.12)) == pytest.approx(15.6, abs=0.2)


# --- piano, strings loss law ---------------------------------------------------

def test_piano_loss_and_inverse():
    b1, b2 = S.piano_loss_coeffs(261.6)
    assert b1 == pytest.approx(4.4e-3 * 261.6 - 4e-2) and b2 == pytest.approx(1e-6 * 261.6 + 1e-5)
    f = 100.0 * np.arange(1, 6) * (1 + 1e-4 * np.arange(1, 6) ** 2)
    al = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    D, mag, ph = S.loop_filter_from_partials(f, al)
    assert mag == pytest.approx(np.exp(-al * D))
    assert D == pytest.approx(1.0 / (f[1] - f[0]))


def test_nonlinear_regime():
    r = S.nonlinear_regime(2e4, 700.0)
    assert r["ratio"] == pytest.approx(math.sqrt(2e4 / 700.0))
    assert r["long_db"] == pytest.approx(20 * math.log10(0.25 * 2e4 * 1e-4 / 7.0))


def test_matched_tuning_and_octave_stretch():
    assert S.matched_tuning_cents(69) == pytest.approx(0.0)
    assert S.octave_stretch_cents(60, 72, 2, 1) > 0.0
    assert fdstring.tuning_curve_cents(440.0, "railsback") == pytest.approx(0.0)
    assert fdstring.tuning_curve_cents(27.5, "railsback") == pytest.approx(-17.0)


def test_wound_bare_ends_raise_B():
    assert S.wound_bare_ends_B(55.0, 1.0e-3, 0.02, 1.0, 6.2e-3, 3.0e-2) > 0.0


def test_felt_K_c4():
    assert PD.felt_K(122e6, 8e-3) == pytest.approx(PD.FELT_C4[2], rel=0.05)


def test_hammer_impulses_three_echoes():
    pulse = np.hanning(16)
    e = P.hammer_impulses(pulse, 50, 0.8, 48000)
    assert e.size == 16 + 3 * 50
    assert e[50:66].sum() < 0 < e[100:116].sum()
    assert P.ORTH_RADIATION == pytest.approx(10 ** (-0.5))


def test_mode_window():
    g = sbp.mode_window([100.0, 0.9 * 20000.0, 20000.0], 48000)
    assert g[0] == 1.0 and 0.0 < g[1] < 1.0 and g[2] == pytest.approx(0.0, abs=1e-12)


def test_bow_width_and_hair_change_output():
    from genny.physical import bow
    a = bow.bowed(196.0, 0.3, 22050)
    b = bow.bowed(196.0, 0.3, 22050, width=0.01, hair=True)
    assert not np.allclose(np.asarray(a, float)[:2000], np.asarray(b, float)[:2000])


# --- winds branchings (V4_VENTS requests) --------------------------------------

def test_attack_tables_woodwinds():
    assert A.attack_ms("flute", 262.0, 1.0) == pytest.approx(30.0)
    assert A.attack_ms("clarinet", 147.0, 0.0) == pytest.approx(100.0)
    assert A.attack_ms("bassoon", 523.0, 1.0) == pytest.approx(20.0)
    assert A.RELEASE_S["flute"] == 0.12


def test_brass_ceiling_rises_with_vel():
    assert W.brass_top("trumpet", 1.0, 4800.0) == pytest.approx(15000.0)
    assert W.brass_top("trumpet", 0.0, 4800.0) == pytest.approx(4800.0)


def _band_db(y, lo, hi, sr=SR):
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1.0 / sr)
    return 10 * np.log10(Y[(f >= lo) & (f < hi)].sum() + 1e-30)


def test_dynamic_timbre_soft_is_darker():
    y = np.random.default_rng(1).standard_normal(SR)
    soft = A.dynamic_timbre(y, SR, "trumpet", 0.0)
    loud = A.dynamic_timbre(y, SR, "trumpet", 1.0)
    assert _band_db(soft, 4000, 8000) < _band_db(loud, 4000, 8000) - 6.0


def test_band_vibrato_am_only_above_band():
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 5000 * t)
    z = A.band_vibrato_am(y, SR, "flute", 5.9)
    assert np.std(np.abs(z[SR // 2:])) > np.std(np.abs(y[SR // 2:]))


def test_level_3k_forte():
    y = A.oboe(1046.5, 1.0, SR, 1.0)
    Y = np.abs(np.fft.rfft(y[SR // 4:]))
    f = np.fft.rfftfreq(y[SR // 4:].size, 1.0 / SR)
    peak = 20 * np.log10(Y.max())
    at3k = 20 * np.log10(Y[(f > 2800) & (f < 3200)].max())
    assert at3k - peak >= W.LEVEL_3K_FORTE["general"] - 6.0


def test_brass_and_wind_models_render():
    _ok(A.trumpet(466.0, 0.5, SR, 0.8, model="lip2d"))
    _ok(A.trombone(233.0, 0.5, SR, 0.8, model="lip"))
    _ok(A.trumpet(466.0, 0.5, SR, 0.2, meyer=1))
    _ok(A.flute(523.0, 0.5, SR, 0.8, meyer=1, expert=1, vibrato=1.0))
    _ok(A.clarinet(196.0, 0.5, SR, 0.8, meyer=1, precursor=1.0))
    _ok(A.oboe(440.0, 0.5, SR, 0.8, even=-6.0, precursor=1.0))
    _ok(A.sax(220.0, 0.5, SR, 0.8, lattice="tenor"))


def test_brassy_length():
    assert W.brassy_length("trumpet") == pytest.approx(W.BRASSINESS["trumpet"] * 1.1 * 1.37)
