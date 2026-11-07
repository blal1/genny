"""Wind acoustics of `genny.physical.winds` and the options it feeds (waveguides, impedance, tubes, networks,
klang, instruments, interactions, voice). Values are the sourced ones of research/SOURCES.md (V4_VENTS)."""
import math

import numpy as np
import pytest

from genny import instruments as I
from genny import klang as K
from genny import networks as N
from genny import tubes as T
from genny.interactions import LipBoreInteraction
from genny.physical import impedance as Z
from genny.physical import voice as V
from genny.physical import waveguides as wg
from genny.physical import winds as W

SR = 48000


def _f0(y, hint):
    return wg.measure_f0(np.asarray(y, float), hint)


def _band_db(y, sr, f, width=0.03):
    S = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    fr = np.fft.rfftfreq(y.size, 1.0 / sr)
    m = (fr > f * (1 - width)) & (fr < f * (1 + width))
    return 20 * math.log10(S[m].max() + 1e-20)


# ------------------------------------------------------------------ air, holes, inharmonicity
def test_air_constants():
    a = W.air(20.0)
    assert abs(a["c"] - 343.2) < 0.3 and abs(a["rho"] - 1.204) < 0.003


def test_phase_velocity_slower_in_narrow_pipe():
    # c / (1 + 1.044 / rv), rv = R sqrt(omega rho / eta): a 2 mm staple is about 10 % slow at 60 Hz (rv ~ 10),
    # a low bassoon note in its ~6 mm body only up to 4 %
    a = W.air()
    rv = 2e-3 * math.sqrt(2 * math.pi * 60.0 * a["rho"] / a["eta"])
    assert float(W.phase_velocity(60.0, 2e-3)) / a["c"] == pytest.approx(1.0 / (1.0 + 1.044 / rv), rel=1e-9)
    assert 0.88 < float(W.phase_velocity(60.0, 2e-3)) / a["c"] < 0.92
    assert 0.96 < float(W.phase_velocity(60.0, 6e-3)) / a["c"] < 0.985


def test_lattice_cutoff_2100_hz():
    assert abs(W.lattice_cutoff(7.5e-3, 3e-3, 9e-3, 6e-3) - 2100.0) < 20.0


def test_hole_lengths():
    b, a, t = 3e-3, 7.5e-3, 3e-3
    assert W.chimney_length(b, a, t, "simple") == pytest.approx(t + b * (1.50 - 0.55 * 0.4))
    assert W.series_length(b, a, t) < 0.0 and W.series_length(b, a, t, open_=False) < 0.0
    assert W.matching_length(b, a) == pytest.approx(b * 0.4 * (1 + 0.207 * 0.4 ** 3) / 8)
    k = W.keefe_hole(a, b, t, 440.0)
    assert k["ta"] > 0 and k["te"] > t
    M = W.tonehole_tmatrix(k["Za"], k["Zs"])
    assert abs(np.linalg.det(M) - 1.0) < 1e-9


def test_finger_shortens_closed_hole():
    assert W.closed_hole_length(4e-3, 7.5e-3, 3e-3, finger_radius=8e-3) < W.closed_hole_length(4e-3, 7.5e-3, 3e-3)


def test_key_correction_grows_as_pad_closes():
    assert W.key_correction(4e-3, 1e-3, 3e-3, 6e-3) > W.key_correction(4e-3, 4e-3, 3e-3, 6e-3)


def test_tmm_first_resonance_of_closed_cylinder():
    f = W.first_resonance(0.6, 7.5e-3, [], kind="z")
    assert abs(f - W.air()["c"] / (4 * 0.6)) / f < 0.03


# ------------------------------------------------------------------ brass
def test_brassiness_and_shock():
    assert W.brassiness() > 0
    assert W.local_speed(10.0) > W.local_speed(0.0)


def test_trombone_mode3_and_slide():
    s = W.slide_for(229.5 / 4 * 4, 4)
    assert s is not None and abs(s) < 5.0


def test_lip_table_b_units():
    lp = W.lip_params(0.5)
    assert 2e-4 < lp["H0"] < 8e-4 and 8e-3 < lp["b"] < 2e-2
    assert lp["inv_mu1"] > 0 > lp["inv_mu2"]


def test_lip_threshold_outward_above_inward_below():
    lp = W.lip_params(0.5)
    fA = lp["fL1"]
    Pt, ft = W.lip_threshold(lp["fL1"] * 0.95, lp["QL1"], lp["inv_mu1"], lp["b"], lp["H0"], fA, 24.0, 20e6)
    assert 1000.0 < Pt < 8000.0 and ft > lp["fL1"] * 0.95
    Pt2, ft2 = W.lip_threshold(lp["fL2"], lp["QL2"], lp["inv_mu2"], lp["b"], lp["H0"], lp["fL2"], 24.0, 20e6)
    assert Pt2 > 0 and ft2 < lp["fL2"]


def test_lip_without_mouthpiece_does_not_buzz():
    assert np.max(np.abs(W.lip_buzz(0.2, mouthpiece=False))) < 1e-9
    assert np.std(W.lip_buzz(0.3)[SR // 10:]) > 0


def test_lip_hysteresis():
    p = np.r_[np.linspace(0, 1100, 100), np.full(10, 965.0), np.full(10, 950.0)]
    on = W.lip_sounding(p, 1000.0)
    assert on[105] and not on[-1]


def test_trombone_lip_pitch():
    y = W.trombone_lip(229.5, 0.5, SR)
    assert abs(1200 * math.log2(_f0(y[SR // 5:], 229.5) / 229.5)) < 50


def test_trumpet_lip2d_pitch():
    y = W.trumpet_lip2d(466.2, 0.5, SR)
    assert abs(1200 * math.log2(_f0(y[SR // 5:], 466.2) / 466.2)) < 30


def test_brass_top_follows_vel():
    assert W.brass_top("trumpet", 0.0, 4800.0) == 4800.0 and W.brass_top("trumpet", 1.0, 4800.0) == 15000.0


def test_brass_inward_resonance_changes_tone():
    a = wg.brass(220.0, 0.5, 0.6, 0.5)
    b = wg.brass(220.0, 0.5, 0.6, 0.5, inward=1.0)
    assert np.max(np.abs(a - b)) > 1e-3


# ------------------------------------------------------------------ reeds
def test_cubic_flow_matches_derivatives():
    zeta, gamma = 0.35, 0.39
    F = lambda p: zeta * (1 - gamma + p) * math.sqrt(gamma - p)
    h = 1e-4
    A, B, C = W.cubic_flow(zeta, gamma)
    assert A == pytest.approx((F(h) - F(-h)) / (2 * h), rel=1e-4)
    assert B == pytest.approx((F(h) - 2 * F(0) + F(-h)) / (2 * h * h), rel=1e-3)


def test_worman_rule_on_modal_clarinet():
    f = 196.0
    lv = []
    for g in (0.36, 0.40):
        y = W.modal_clarinet(f, 0.8, SR, gamma=g)[SR // 3:]
        lv.append((_band_db(y, SR, f), _band_db(y, SR, 3 * f)))
    # output is peak-normalised: Worman's P3 ~ P1^3 shows as H3 - H1 growing with the blowing
    assert lv[1][1] - lv[1][0] > lv[0][1] - lv[0][0] + 1.0
    assert W.worman_level(0.5, 3) == pytest.approx(0.125)


def test_modal_clarinet_plays_its_note():
    y = W.modal_clarinet(196.0, 0.6, SR)
    assert abs(1200 * math.log2(_f0(y[SR // 4:], 196.0) / 196.0)) < 30


def test_reed_recovery_and_aging():
    assert W.reed_recovery(0.0) == pytest.approx(0.93)
    assert W.reed_aging_force(30.0) == pytest.approx(-0.5)


def test_bend_rules():
    assert W.clarinet_bend_cents(4.5, 1.0) == 0.0
    assert W.clarinet_bend_cents(4.5, 2.0) > 0


def test_impedance_cone_bend_hardness_squeak():
    base = Z.clarinet(220.0, 0.5, SR)
    cone = Z.clarinet(220.0, 0.5, SR, bore="cone")
    assert abs(1200 * math.log2(_f0(cone[SR // 5:], 220.0) / 220.0)) < 50
    assert np.max(np.abs(base[:cone.size] - cone[:base.size])) > 1e-3
    hard = Z.clarinet(220.0, 0.5, SR, hardness=4.0)
    assert np.max(np.abs(base - hard)) > 1e-3
    sq = Z.clarinet(220.0, 0.5, SR, squeak=0.2)
    assert np.max(np.abs(base - sq)) > 1e-3
    bent = Z.clarinet(220.0, 0.5, SR, blow=0.9, lip_force=2.0, bend=True)
    assert _f0(bent[SR // 5:], 220.0) != pytest.approx(_f0(base[SR // 5:], 220.0), abs=0.1)


def test_reed_presets_and_contact_damping():
    p = T.reed_params(contact_damping=1e-3, **W.REED_PLAYED)
    assert p.size == 9 and p[8] == 1e-3 and p[3] == pytest.approx(14881.0)
    a = T.reed_tube(220.0, 0.4)
    b = T.reed_tube(220.0, 0.4, contact=2e-3)
    assert np.max(np.abs(a - b)) > 1e-4
    s = T.reed_cone(196.0, 0.4, reed="sax", mouthpiece=1.0)
    assert np.std(s) > 1e-3


def test_sax_equivalent_volume():
    assert W.sax_equivalent_volume(0.0) == pytest.approx(3.85e-6)


def test_buffet_clarinet_low_e():
    t, st = W.buffet_clarinet()
    f = t.resonances(400.0, hole_state=st)[0]
    assert 135.0 < f < 155.0


def test_playability():
    y = Z.clarinet(220.0, 0.5, SR, blow=0.8)
    assert W.playable(y[SR // 5:] * 4000.0, 4000.0 / 6, _f0(y[SR // 5:], 220.0), 220.0)


# ------------------------------------------------------------------ jets
def test_jet_controls_expert():
    c = W.jet_controls(261.63)
    assert c["Pm"] == pytest.approx(300.0) and W.jet_controls(523.25)["Uj"] / c["Uj"] == pytest.approx(1.35, abs=0.02)


def test_dynamic_threshold_343():
    assert W.dynamic_threshold(100.0, 311.5) == pytest.approx(343.0, abs=0.5)


def test_register_hysteresis():
    sr = 1000
    p = np.r_[np.linspace(0, 400, 2000), np.full(500, 400.0), np.full(1000, 280.0), np.full(1000, 200.0)]
    r = W.register_track(p, sr)
    assert r[2400] == 2 and r[3400] == 2 and r[-1] == 1


def test_jet_angle_offset_values():
    assert W.jet_angle_offset(50.0) == pytest.approx(-0.5)


def test_flute_offset_brings_even_harmonics():
    rng = np.random.default_rng(0)
    a = wg.recorder(440.0, None, 0.6, rng, noise_gain=0.0, vibrato_gain=0.0)
    b = wg.recorder(440.0, None, 0.6, np.random.default_rng(0), noise_gain=0.0, vibrato_gain=0.0, jet_offset=0.6)
    sa, sb = a[SR // 5:], b[SR // 5:]
    ea = _band_db(sa, SR, 880.0) - _band_db(sa, SR, 440.0)
    eb = _band_db(sb, SR, 880.0) - _band_db(sb, SR, 440.0)
    assert eb > ea


def test_flute_butter_and_tanh_options_play():
    # convection: the jet delay scales as 0.4/convection at a fixed breath; past ~0.46 the STK jet delay leaves the
    # fundamental's regime (the note breaks, as a too-fast jet does), so the pitch is checked just above 0.4
    for kw in (dict(jet_filter="butter"), dict(jet_shape="tanh"), dict(convection=0.45)):
        y = wg.recorder(440.0, None, 0.6, np.random.default_rng(0), noise_gain=0.0, vibrato_gain=0.0, **kw)
        assert abs(1200 * math.log2(_f0(y[SR // 5:], 440.0) / 440.0)) < 50


def test_recorder_auvray_and_register():
    y = wg.recorder(564.0, None, 0.5, np.random.default_rng(0), model="auvray")
    assert abs(1200 * math.log2(_f0(y[SR // 5:], 564.0) / 564.0)) < 60
    env = np.linspace(0.3, 1.2, int(0.8 * SR))
    z = wg.recorder(440.0, env, 0.8, np.random.default_rng(0), register=True, tune=False)
    assert _f0(z[-SR // 5:], 880.0) > 700.0


def test_auvray_wet_is_sharper():
    a = W.auvray_recorder(564.0, 0.5)
    b = W.auvray_recorder(564.0, 0.5, wet=True)
    assert _f0(b[SR // 5:], 564.0) > _f0(a[SR // 5:], 564.0)


# ------------------------------------------------------------------ waveguide clarinet
def test_pasp_clarinet_in_tune_and_legato():
    f = np.r_[np.full(SR // 2, 220.0), np.full(SR // 2, 247.0)]
    y = wg.clarinet(220.0, 0.6, 0.6, np.random.default_rng(0), noise_gain=0.0, vibrato_gain=0.0, reflection="pasp")
    assert abs(1200 * math.log2(_f0(y[SR // 5:], 220.0) / 220.0)) < 40
    a = wg.clarinet(f, 0.6, 1.0, np.random.default_rng(0), noise_gain=0.0, vibrato_gain=0.0)
    b = wg.clarinet(f, 0.6, 1.0, np.random.default_rng(0), noise_gain=0.0, vibrato_gain=0.0, legato=0.03)
    seg = slice(SR // 2, SR // 2 + int(0.03 * SR))
    assert np.max(np.abs(a[seg] - b[seg])) > 1e-4


# ------------------------------------------------------------------ networks, other modules
def test_tone_hole_impedance_open_below_closed():
    h = N.DynamicToneHole(openness=1.0)
    c = N.DynamicToneHole(openness=0.0)
    assert h.impedance(SR) < c.impedance(SR)
    assert h.series_length < 0


def test_lip_bore_area_from_table_b():
    r = LipBoreInteraction(lip_tension=0.5).solve(220.0, 0.2, sr=SR)
    lp = W.lip_params(0.5)
    assert r.metadata["lip_area"] == pytest.approx(lp["b"] * lp["H0"])


def test_glottal_noise_shift_moves_hump():
    _, a = V.lf_pulse(120.0, 0.6, 0.3, return_modulator=True)
    _, b = V.lf_pulse(120.0, 0.6, 0.3, return_modulator=True, noise_shift=0.35)
    assert np.max(np.abs(a - b)) > 1e-3


def test_bass_clarinet_low_even_under_30_db():
    y = K.bass_clarinet(73.42, 0.6, SR)[SR // 5:]
    assert _band_db(y, SR, 73.42) - _band_db(y, SR, 146.84) > 25.0


def test_a_clarinet_darker_at_1k():
    from genny.klang import _clarinet_body
    a = K.a_clarinet(220.0, 0.5, SR)
    b = _clarinet_body(220.0, 0.5, SR, 1.0, 1.0)
    n = min(a.size, b.size)
    assert _band_db(b[:n], SR, 1100.0) - _band_db(a[:n], SR, 1100.0) > 3.0


def test_new_wind_presets_render():
    for f, fn in ((58.27, K.contrabassoon), (110.0, K.heckelphone), (58.27, K.bass_trombone), (174.6, K.natural_horn),
                  (587.3, K.clarino), (392.0, K.eb_clarinet)):
        y = fn(f, 0.4, SR)
        assert abs(1200 * math.log2(_f0(y[SR // 10:], f) / f)) < 30


def test_organ_registration_and_click():
    a = I.organ(220.0, 0.3, SR, registration="888000000")
    b = I.organ(220.0, 0.3, SR, registration="008000000")
    assert _band_db(a, SR, 110.0) - _band_db(b, SR, 110.0) > 20.0
    c = I.organ(220.0, 0.3, SR, registration="888000000", key_click=1.0)
    assert np.max(np.abs(c[:48] - a[:48])) > 0


def test_brass_follow_cutoff():
    lo = I.brass(220.0, 0.4, SR, vel=0.3, follow=1)
    hi = I.brass(220.0, 0.4, SR, vel=1.0, follow=1)
    s = slice(SR // 5, SR * 3 // 10)
    assert _band_db(hi[s], SR, 2200.0) - _band_db(hi[s], SR, 220.0) > _band_db(lo[s], SR, 2200.0) - _band_db(lo[s], SR, 220.0)


def test_vibrato_am_and_omni_tables():
    assert W.vibrato_am("trombone", 3000.0) == 15.0 and W.vibrato_am("clarinet", 2500.0) == 20.0
    assert W.OMNI_BELOW["tuba"] == 75.0
