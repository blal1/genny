import math

import numpy as np

from genny import contact, friction, vehicle
from genny.physical import modal
from genny.physical.core import SR


def _band(y, sr, lo, hi):
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    s = np.abs(np.fft.rfft(y)) ** 2
    return float(s[(f >= lo) & (f < hi)].sum())


def test_ptr_thesis_constants():
    assert vehicle.PTR_NU == 0.9
    assert vehicle.PTR_JITTER == 2.0 / 720.0
    assert vehicle.ptr_envelope(0.0) == (7.6, 2.3)
    a, b = vehicle.ptr_envelope(1.0)
    assert math.isclose(a, 11.5) and math.isclose(b, 1.15)


def test_ptr_inharmonicity_is_a_bounded_gaussian_bump():
    assert not vehicle.ptr_inharmonicity(24, 0.0, 3).any()
    d = vehicle.ptr_inharmonicity(24, 1.0, 3)
    assert 0.0 < np.max(np.abs(d)) <= 0.08 + 1e-12


def test_car_engine_ptr_inharm_changes_the_render():
    a = vehicle.car_engine(sr=16000, model="ptr", dur=0.5, inharm=0.0)
    b = vehicle.car_engine(sr=16000, model="ptr", dur=0.5, inharm=1.0)
    assert not np.allclose(a, b)


def test_additive_table_delays_each_partial():
    t = {"ratios": (1.0,), "attacks_ms": (300,), "decay_ms": (9000,), "delays_ms": (400,), "amp_scale": 0.08, "phases": "aligned"}
    y = modal.additive_bell(t, 100.0, 1.0)
    assert np.all(y[:int(0.399 * SR)] == 0.0)
    assert np.max(np.abs(y[int(0.6 * SR):])) > 0.0
    w = modal.additive_bell("struck_wire", 100.0, 0.05)
    assert w[0] == 0.0
    assert modal.ADDITIVE_TABLES["delayed_bell"]["delays_ms"][0] == 400


def test_sum_tone_brings_the_sum_frequency():
    sr = 48000
    t = np.arange(sr // 2) / sr
    y = np.sin(2 * np.pi * 3000 * t) + np.sin(2 * np.pi * 3100 * t)
    q = friction.sum_tone(y, 3000.0, sr)
    assert _band(q, sr, 6050, 6150) > 10 * _band(q, sr, 50, 150)


def test_nail_length_is_the_cantilever_law():
    L = friction.nail_length(1000.0, 0.004)
    E, rho = friction.NAIL_STEEL
    f = 1.8751 ** 2 / (2 * math.pi * L * L) * 0.001 * math.sqrt(E / rho)
    assert math.isclose(f, 1000.0, rel_tol=1e-6)


def test_marloye_harp_sings_odd_harmonics():
    sr = 48000
    y = friction.marloye_harp(440.0, 0.6, sr=sr)
    assert _band(y, sr, 1300, 1340) > _band(y, sr, 860, 900)


def test_nail_violin_partial_ratio():
    assert friction.CLAMPED[1] == 6.267
    y = friction.nail_violin(440.0, 0.4, sr=48000)
    assert np.max(np.abs(y)) > 0


def test_uneven_decay_kills_the_highs_sooner():
    sr = 48000
    rng = np.random.default_rng(0)
    t = np.arange(sr) / sr
    y = rng.standard_normal(sr) * np.exp(-3 * t)
    assert np.allclose(contact.uneven_decay(y, sr, 0.0), y)
    z = contact.uneven_decay(y, sr, 1.0)
    late = slice(int(0.5 * sr), sr)
    assert _band(z[late], sr, 4000, 10000) < 0.1 * _band(y[late], sr, 4000, 10000)


from genny import cyber  # noqa: E402


def _line_db(y, sr, f, half=8.0):
    return 10 * np.log10(_band(y, sr, f - half, f + half) + 1e-30)


def test_gearbox_mount_mass_blocks_the_highs():
    assert abs(cyber.blocking_mass_fc(1.0) - 391.0) < 3.0
    sr = 48000
    a = cyber.gearbox(sr=sr, dur=1.0, rpm=1800.0, teeth=23, seed=1)
    b = cyber.gearbox(sr=sr, dur=1.0, rpm=1800.0, teeth=23, seed=1, mount_mass=1.0)
    drop = lambda f: _line_db(a, sr, f) - _line_db(b, sr, f)
    assert abs((drop(2070.0) - drop(690.0)) - 8.5) < 1.0
    assert np.allclose(a, cyber.gearbox(sr=sr, dur=1.0, rpm=1800.0, teeth=23, seed=1, mount_mass=0.0))


def test_bearing_sizes_limits_and_thrust():
    assert cyber.BEARING_SIZES["6205"] == (9, 7.94, 39.04)
    assert cyber.BEARING_LIMITS["6205"] == (18000.0, 8500.0)
    k = cyber.bearing_frequencies(60.0, 12, 5.556, 27.5, 90.0)
    assert math.isclose(k["bpfo"], k["bpfi"]) and math.isclose(k["bpfo"], 6.0)


def test_load_zone_stribeck():
    assert math.isclose(float(cyber.load_zone(0.0, 0.5)), 1.0)
    eps = 0.2 + 0.8 * 0.5
    psi = math.acos(1 - 2 * eps)
    assert float(cyber.load_zone(psi + 0.01, 0.5)) == 0.0
    assert math.isclose(float(cyber.load_zone(1.0, 0.0)), math.cos(1.0) ** 2)


def test_mirror_gain_law():
    f = 3000.0
    assert math.isclose(float(cyber.mirror_gain(f)), (math.pi * 0.09 * f / 343.0) ** 2)
    assert float(cyber.mirror_gain(500.0)) == 1.0


def test_bearing_new_states_render_and_rotating_outer_acts():
    for st in ("pitting", "crack", "spall", "polished", "cage"):
        y = cyber.bearing(sr=16000, dur=0.5, state=st)
        assert np.max(np.abs(y)) > 0
    a = cyber.bearing(sr=16000, dur=0.5, state="inner")
    b = cyber.bearing(sr=16000, dur=0.5, state="inner", rotating="outer")
    assert not np.allclose(a, b)


def test_bearing_defect_entry_and_exit_shock():
    sr = 48000
    a = cyber.bearing(sr=sr, dur=1.0, state="outer", severity=1.0, seed=2)
    b = cyber.bearing(sr=sr, dur=1.0, state="outer", severity=1.0, seed=2, defect=10.0)
    assert not np.allclose(a, b)


def test_transformer_joints_add_harmonics():
    b = np.sin(np.linspace(0, 2 * np.pi, 4000, endpoint=False))
    s0 = np.abs(np.fft.rfft(cyber.core_strain(b, 0.0)))
    s1 = np.abs(np.fft.rfft(cyber.core_strain(b, 1.0)))
    assert s1[4] / s1[2] > s0[4] / s0[2]


def test_gsm_bit_errors_and_channel():
    rng = np.random.default_rng(0)
    y = np.sin(np.linspace(0, 100, 20000))
    z = cyber.bit_errors(y, 0.0, 0.0, rng)
    assert np.max(np.abs(z - y)) < 1e-4
    z = cyber.bit_errors(y, 0.041, 0.13, rng)
    pk = np.max(np.abs(y))
    bits = lambda v: (np.round(v / pk * 32767).astype(np.int16).view(np.uint16)[:, None] >> np.arange(16)) & 1   # noqa: E731
    err = bits(z) != bits(y)
    assert abs(err[:, 8:].mean() - 0.041) < 0.003 and abs(err[:, :8].mean() - 0.13) < 0.005   # class I / class II BER
    assert cyber.GSM_CHANNEL["poor"][0] == 95 / 750


def test_pipe_engine_helpers():
    assert math.isclose(cyber.gas_speed(900.0), 601.6, rel_tol=2e-3)
    assert abs(cyber.plane_wave_cutoff(0.025) - 4020.0) < 5.0
    assert abs(cyber.side_branch_freq(0.1) - 857.5) < 1.0
    assert abs(cyber.bend_cents(0.025, 0.075) - 25.0) < 1.5
    assert cyber.bend_cents(0.025, 0.075, f=2100.0) == 0.0
    assert math.isclose(cyber.step_end_correction(0.0), 0.26164)
    ir = cyber.expansion_chamber_ir(2.0, 5)
    assert np.allclose(ir, [-1 / 3, -8 / 9, 8 / 27, -8 / 81, 8 / 243])
    assert math.isclose(float(cyber.twin_power_gain(1e-6)), 4.0, rel_tol=1e-6)
    assert math.isclose(cyber.helmholtz_neck(110.0, 3e-3, 1e-3), 0.6e-3 * 343 ** 2 / (3e-3 * (2 * math.pi * 110) ** 2))


def test_pipe_engine_defaults_unchanged_and_options_act():
    base = cyber.pipe_engine(sr=16000, dur=0.4, seed=3)
    assert np.allclose(base, cyber.pipe_engine(sr=16000, dur=0.4, seed=3, jitter=0.0, timing=None, cylinder=0.0, outlet=0.0))
    for kw in ({"jitter": 1.0}, {"timing": "-0.043,-0.004"}, {"cylinder": 0.2}, {"outlet": 2.0}, {"warp": 0.5},
               {"radiate": 1}, {"side": "110"}, {"expansion": 8.0}, {"roots": 1.0}, {"view": "cabin"}, {"extractor": 2.0}):
        assert not np.allclose(base, cyber.pipe_engine(sr=16000, dur=0.4, seed=3, **kw)), kw
    m = cyber.pipe_engine(sr=16000, dur=0.4, seed=3, units="m", header=0.5, pipe=2.0, gas_temp=900.0)
    assert np.max(np.abs(m)) > 0


def test_accessory_whine_frequencies():
    sr = cyber.SR
    rpm = np.full(sr, 3000.0)
    y = cyber.accessory_whine(rpm, 1.0, roots=1.0)
    f = np.fft.rfftfreq(sr, 1 / sr)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(y)))] - 600.0) < 2.0


def test_auto_load_follows_rpm_rate():
    r = np.concatenate([np.full(cyber.SR, 1000.0), np.linspace(1000, 4000, cyber.SR)])
    ld = cyber.auto_load(r)
    assert abs(ld[cyber.SR // 2] - 0.3) < 1e-6
    assert abs(ld[int(1.5 * cyber.SR)] - 1.0) < 1e-6


def test_ui_category_gates():
    assert cyber.ui_category(0.1) == "hover"
    assert cyber.ui_category(0.25, attack=0.01) == "tap"
    assert cyber.ui_category(0.5, attack=0.1, sweep_st=5) == "transition"
    assert cyber.ui_category(0.5, attack=0.1, sweep_st=-3, peak_hz=400, tonal=False) == "error"


def test_induction_motor_slot_line():
    sr = 48000
    y = cyber.induction_motor(sr=sr, dur=1.0, mains=50.0, poles=4, load=0.0, bars=28)
    fs = 28 * 2 * 50 * (1 - 0.005) / 4
    assert _line_db(y, sr, fs, 3) > _line_db(y, sr, fs + 40, 3) + 10


def test_pass_by_two_sources_differs():
    a = cyber.pass_by(sr=16000, dur=2.0, speed=50.0)
    b = cyber.pass_by(sr=16000, dur=2.0, speed=50.0, sources=2)
    assert a.shape == b.shape and not np.allclose(a, b)


def test_brake_kinds_and_default():
    a = friction.brake_squeal(sr=16000, dur=0.5)
    assert np.allclose(a, friction.brake_squeal(sr=16000, dur=0.5, kind="rub", hot_spots=0.0))
    g = friction.brake_squeal(sr=16000, dur=0.8, kind="groan")
    assert _band(g, 16000, 0, 500) > 20 * _band(g, 16000, 1000, 8000)
    for k in friction.BRAKE_KINDS:
        assert np.max(np.abs(friction.brake_squeal(sr=16000, dur=0.4, kind=k))) > 0


def test_contact_rolling_and_creep_laws():
    assert math.isclose(contact.rolling_resistance_peak(0.002, 0.1), 0.195 * 0.02)
    assert math.isclose(contact.rolling_loss_peak(1.0, 0.01), 183.0)
    assert math.isclose(contact.plastic_rolling_moment(1000.0, 1e6), 0.104)
    assert contact.creep_closed_form(0.0, 0.5, 100.0, 8e10, 1e-3) == 0.0
    full = 3 * 0.5 * 100.0 * (4 - 0.9) / (16 * 8e10 * 1e-6)
    assert math.isclose(abs(float(contact.creep_closed_form(50.0, 0.5, 100.0, 8e10, 1e-3))), full)


def test_valve_float_and_lift():
    assert math.isclose(float(contact.valve_lift(0.0)), 9.5 - 0.15)
    assert float(contact.valve_lift(1.2)) == 0.0
    assert 4000.0 < contact.valve_float_rpm() < 20000.0


def test_modal_engine_car1_and_valves():
    a = contact.modal_engine(sr=16000, dur=0.4)
    assert np.allclose(a, contact.modal_engine(sr=16000, dur=0.4, body="plate", valves=0.0))
    assert not np.allclose(a, contact.modal_engine(sr=16000, dur=0.4, body="car1"))
    assert not np.allclose(a, contact.modal_engine(sr=16000, dur=0.4, valves=1.0))
    assert contact.CAR1_BODY[0][3] == 900.0


def test_profile_plasticity_scales_with_sigma():
    x = np.arange(4000) * 1e-5
    h = 1e-6 * np.sin(2 * np.pi * x / 2e-4)
    p1 = contact.profile_plasticity(h, 1e-5, 1e11, 2e9)
    p2 = contact.profile_plasticity(4 * h, 1e-5, 1e11, 2e9)
    assert math.isclose(p2 / p1, 4.0, rel_tol=0.05)


from genny import designed, foley, machines, physics, road  # noqa: E402


def test_jackhammer_rod_and_box_fan():
    a = machines.jackhammer(sr=16000, dur=0.5)
    assert np.allclose(a, machines.jackhammer(sr=16000, dur=0.5, model="impact"))
    assert np.max(np.abs(machines.jackhammer(sr=16000, dur=0.5, model="rod"))) > 0
    assert np.max(np.abs(machines.hvac(sr=16000, dur=0.5, state="box_fan"))) > 0


def test_engine_rpm_starts_idles_and_limits():
    r = cyber.engine_rpm(np.zeros(3000))
    assert r[50] > 0 and r[-1] > 100
    r = cyber.engine_rpm(np.ones(20000), rpm0=1000.0)
    assert r.max() < 8000


def test_pipe_engine_new_options_neutral_and_active():
    base = cyber.pipe_engine(sr=16000, dur=0.4, seed=3)
    assert np.allclose(base, cyber.pipe_engine(sr=16000, dur=0.4, seed=3, crank=0.0, norm4=0, intake_gain=1.0,
                                               outlet_gain=1.0, ends="engine"))
    for kw in ({"crank": 0.5}, {"ends": "paper"}, {"intake_gain": 2.0}):
        assert not np.allclose(base, cyber.pipe_engine(sr=16000, dur=0.4, seed=3, **kw)), kw


def test_transformer_packet_and_grid():
    a = cyber.transformer(sr=16000, dur=1.0)
    assert np.allclose(a, cyber.transformer(sr=16000, dur=1.0, packet=0, grid=""))
    assert not np.allclose(a, cyber.transformer(sr=16000, dur=1.0, packet=3))
    f = cyber.grid_frequency(cyber.SR * 200, 50.0, 9.8, np.random.default_rng(0))
    inc = f[cyber.SR::cyber.SR] - f[:-cyber.SR:cyber.SR][: len(f[cyber.SR::cyber.SR])]
    assert 5e-3 < np.std(inc) < 15e-3


def test_polack_and_steepening():
    D = 0.03
    h = cyber.polack_ir(D, 48000)
    assert abs(np.argmax(h) / 48000 - 2 * D * D / 3) < 2e-4
    assert float(cyber.steepening(1e9)) == 0.62 and float(cyber.steepening(0.0)) == 1.0


def test_ruptures():
    assert math.isclose(contact.conrod_stress(0.5, 0.04, 6000, 0.14), 0.5 * 0.04 * (200 * math.pi) ** 2 * (1 + 0.04 / 0.14) / 1.6e-4)
    assert contact.valve_hits_piston(0.02, 0.01, 5e-5, 5e-3) is True
    assert math.isclose(float(contact.turbo_spin_down(100.0, 1.0)), 100 * math.exp(-3))
    assert float(contact.fatigue_damage(0.5, 1.0)) == 0.5 ** 9


def test_pd003_engine_and_classic_unchanged():
    y = vehicle.car_engine(sr=16000, dur=0.5, model="pd003")
    assert np.max(np.abs(y)) > 0
    assert np.max(np.abs(vehicle.car_engine(sr=16000, dur=0.5))) > 0


def test_helicopter_options_neutral():
    h = physics.Helicopter()
    a = h.render(0.3, sr=16000)
    assert np.allclose(a, physics.Helicopter(rotors=1, fuselage=0.0, gearbox=0.0, partials=0.0).render(0.3, sr=16000))
    for kw in ({"rotors": 4}, {"fuselage": 1.0}, {"gearbox": 1.0}, {"partials": 1.0}):
        assert not np.allclose(a, physics.Helicopter(**kw).render(0.3, sr=16000)), kw
    assert np.max(np.abs(physics.Rotor(style="farnell").render(0.3, sr=16000))) > 0


def test_car_wind_pillar_and_buffet():
    assert abs(road.cabin_helmholtz(1.0) - 343 / (2 * math.pi) * math.sqrt(0.056 / (2.7 * 0.05))) < 1e-9
    y = road.car_wind(sr=16000, dur=1.0, speed=180.0, view="outside")
    assert _band(y, 16000, 1567, 1767) > _band(y, 16000, 2367, 2567)


def test_alert_roughness_and_can_roll():
    a = designed.alert_tone(sr=16000)
    assert np.allclose(a, designed.alert_tone(sr=16000, rough=0.0))
    assert not np.allclose(a, designed.alert_tone(sr=16000, rough=1.0))
    c = designed.can_roll(sr=16000, dur=0.5)
    assert np.allclose(c, designed.can_roll(sr=16000, dur=0.5, pattern=1, bumps=0.0))
    assert not np.allclose(c, designed.can_roll(sr=16000, dur=0.5, pattern=2))


def test_foley_requests():
    assert foley.BLADES["broom"] == "BROOM" and foley.BLADES["wood_sword"] == "WOODSWD"
    assert np.max(np.abs(foley.sword(sr=16000, blade="broom"))) > 0
    f, t = foley.dipped(1000.0, 3.0, 1.0)
    assert math.isclose(f, 1000 / math.sqrt(1.5)) and math.isclose(t, 0.75)
    assert foley.dipped(1000.0, 3.0, 0.0) == (1000.0, 3.0)


def test_pnoise_switch_relay_servo_bell():
    s = foley.switch(sr=16000)
    assert np.allclose(s, foley.switch(sr=16000, noise="band"))
    assert not np.allclose(s, foley.switch(sr=16000, noise="pnoise"))
    assert np.max(np.abs(foley.relay_chatter(sr=16000, rate=100.0, count=4))) > 0
    a = foley.servo(sr=16000)
    assert not np.allclose(a, foley.servo(sr=16000, case=1))
    y = np.sin(np.linspace(0, 2000, 48000))
    assert np.allclose(foley.swung(y, 0.0), y)


def test_propeller_vortex_harmonics_default():
    a = modal.propeller_tone(2100, 3, 1.0, 200.0, 0.2, np.random.default_rng(1))
    b = modal.propeller_tone(2100, 3, 1.0, 200.0, 0.2, np.random.default_rng(1), vortex_harmonics=1, chord_tip=None)
    assert np.allclose(a, b)


def test_s100_position_folds():
    assert np.allclose(modal.resolve("sy_s100", position=0.25).gains, modal.resolve("sy_s100", position=0.75).gains)
    assert np.allclose(modal.resolve("sy_s100", position=1.0).gains, modal.resolve("sy_s100", position=0.0).gains)
