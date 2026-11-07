"""Machines and tools: every variant renders; the numbers of the spec are measured on the rendered signal."""
import math

import numpy as np
import pytest

from genny import foley as FO
from genny import machines as M
from genny.analysis import detect_clicks
from genny.sfx import REGISTRY

SR = 22050
HI = 48000


def spec(y, sr):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return np.fft.rfftfreq(len(y), 1 / sr), Y


def top(y, sr, lo, hi):
    f, Y = spec(y, sr)
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(Y[m])]


def line_db(y, sr, f0, width=0.01):
    """Peak within f0 +- width, against the median of the octave around it."""
    f, Y = spec(y, sr)
    near = (f >= f0 * (1 - width)) & (f <= f0 * (1 + width))
    around = (f >= f0 * 0.7) & (f <= f0 * 1.4)
    return 10 * np.log10(Y[near].max() / np.median(Y[around]))


def share(y, sr, lo, hi):
    f, Y = spec(y, sr)
    return Y[(f >= lo) & (f < hi)].sum() / Y.sum()


def centroid(y, sr):
    f, Y = spec(y, sr)
    return (f * Y).sum() / Y.sum()


def flatness(y, sr, lo, hi):
    f, Y = spec(y, sr)
    Y = Y[(f > lo) & (f < hi)] + 1e-20
    return np.exp(np.log(Y).mean()) / Y.mean()


def rms(y):
    return float(np.sqrt(np.mean(np.square(y))))


def env_top(y, sr, lo, hi):
    """Strongest modulation of the magnitude between lo and hi Hz."""
    e = np.abs(y) - np.mean(np.abs(y))
    return top(e, sr, lo, hi)


def env(y, sr, ms=10):
    w = max(1, int(ms * 1e-3 * sr))
    return np.convolve(np.abs(y), np.ones(w) / w, "same")


def peaks(y, sr, gap=0.1, floor=0.3):
    """Times of the attacks: envelope maxima above floor x the highest, gap s apart."""
    e = env(y, sr, 5)
    idx, last = [], -1e9
    for i in np.flatnonzero((e[1:-1] >= e[:-2]) & (e[1:-1] >= e[2:]) & (e[1:-1] > floor * e.max())) + 1:
        if i - last >= gap * sr:
            idx.append(i)
            last = i
    return np.array(idx) / sr


def env_period(y, sr, lo, hi):
    """Lag of the strongest repeat of the 5 ms envelope between lo and hi s, and how strong it is (0..1): the bed does not count as an attack."""
    e = env(y, sr, 5)
    e = e - np.convolve(e, np.ones(sr) / sr, "same")
    c = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)[:len(e)]
    a, b = int(lo * sr), int(hi * sr)
    i = a + np.argmax(c[a:b])
    return i / sr, c[i] / c[0]


# ---------------------------------------------------------------------------------- every variant
# Steady machines: no click anywhere.
SMOOTH = [(M.power_tool, dict(kind=k)) for k in ("ponceuse", "disqueuse", "scie circulaire", "tournevis", "soudeur")]
SMOOTH += [(M.elevator, dict(kind=k)) for k in ("montee", "carillon")]
SMOOTH += [(M.hvac, dict(state=s, dur=2.0)) for s in M.HVAC]
SMOOTH += [(M.medical, dict(kind=k)) for k in M.MEDICAL]
SMOOTH += [(M.gym, dict(kind=k)) for k in ("tapis", "velo")]
SMOOTH += [(M.pump, dict(kind=k)) for k in ("aquarium", "puits")]
SMOOTH += [(M.hard_drive, dict(state=s)) for s in ("idle", "spinup")]
SMOOTH += [(M.ride, dict(kind="manege", dur=5.0)), (M.escalator, {}), (M.conveyor, {}), (M.jackhammer, {}),
           (M.dental_drill, {}), (M.belt_squeal, {}), (M.savart_wheel, {})]
# Attacks are the sound (solenoids, stamps, ticks; detect_clicks is meant for smooth voices): finite, heard, clean ends.
PERCUSSIVE = [(M.power_tool, dict(kind="cloueuse")), (M.elevator, dict(kind="arret")), (M.elevator, dict(kind="porte")), (M.hard_drive, dict(state="seek")),
              (M.ride, dict(kind="montagnes russes")), (M.gym, dict(kind="poids")), (M.industrial, dict(kind="presse")), (M.industrial, dict(kind="marteau-pilon")),
              (M.sewing_machine, {}), (M.pump, dict(kind="air")), (M.industrial, dict(kind="usine"))]
PERCUSSIVE += [(M.slot_machine, dict(state=s)) for s in M.SLOT]


def _id(c):
    return f"{c[0].__name__}-{'-'.join(map(str, c[1].values()))}"


@pytest.mark.parametrize("case", SMOOTH, ids=_id)
def test_smooth_variant_renders_without_click(case):
    y = np.asarray(case[0](sr=SR, **case[1]))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert np.abs(y).max() <= 1.0
    assert len(detect_clicks(y)) == 0


@pytest.mark.parametrize("case", PERCUSSIVE, ids=_id)
def test_percussive_variant_renders_with_clean_ends(case):
    y = np.asarray(case[0](sr=SR, **case[1]))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


def test_catalog_and_kind_checks():
    for name in ("elevator", "hard_drive", "ride", "escalator", "gym", "jackhammer", "slot_machine", "dental_drill", "pump", "sewing_machine",
                 "power_tool", "hvac", "medical", "industrial", "conveyor", "belt_squeal", "savart_wheel"):
        assert name in REGISTRY
    for f in (M.elevator, M.ride, M.gym, M.pump, M.power_tool, M.medical, M.industrial):
        with pytest.raises(ValueError):
            f(sr=SR, kind="nope")
    for f in (M.hard_drive, M.slot_machine, M.hvac):
        with pytest.raises(ValueError):
            f(sr=SR, state="nope")
    assert np.array_equal(M.power_tool(sr=SR, seed=3), M.power_tool(sr=SR, seed=3))


# ---------------------------------------------------------------------------------- power tools
@pytest.mark.parametrize("rpm,teeth", [(5000.0, 24), (3000.0, 40), (9000.0, 12)])
def test_saw_blade_passes_at_rpm_teeth_over_60(rpm, teeth):
    y = M.power_tool(sr=HI, kind="scie circulaire", rpm=rpm, teeth=teeth, load=0.3, dur=1.0)
    bpf = rpm * teeth / 60.0
    assert M.saw_bpf(rpm, teeth) == bpf
    assert abs(top(y, HI, 0.7 * bpf, 1.3 * bpf) - bpf) < 3.0
    assert line_db(y, HI, bpf, 0.005) > 25


def test_tool_rpm_stays_in_the_spec_range():
    assert all(3000.0 <= r <= 11000.0 for r in M.TOOL_RPM.values())
    a = M.power_tool(sr=HI, kind="scie circulaire", rpm=50000.0, teeth=12, dur=0.5)
    assert abs(top(a, HI, 1500, 3000) - 11000.0 * 12 / 60.0) < 5.0


def test_sander_is_broad_and_load_brings_the_material():
    free, bite = M.power_tool(sr=SR, kind="ponceuse", load=0.0, dur=1.0), M.power_tool(sr=SR, kind="ponceuse", load=1.0, dur=1.0)
    assert flatness(bite, SR, 500, 8000) > 3 * flatness(free, SR, 500, 8000)   # motor lines, then a broad grind
    assert abs(env_top(bite, SR, 100, 200) - 150.0) < 3.0           # the orbit, 9000 rpm


def test_grinder_mesh_line_and_metal_band():
    y = M.power_tool(sr=HI, kind="disqueuse", rpm=11000.0, load=0.2, dur=1.0)
    assert line_db(y, HI, 11000.0 * 10 / 60.0) > 15                # bevel pinion, 10 teeth
    assert centroid(M.power_tool(sr=SR, kind="disqueuse", load=1.0, dur=1.0), SR) > centroid(M.power_tool(sr=SR, kind="disqueuse", load=0.0, dur=1.0), SR)


def test_nailer_fires_each_nail():
    y = M.power_tool(sr=SR, kind="cloueuse", shots=3)
    t = peaks(y, SR, gap=0.3, floor=0.5)
    assert len(t) == 3 and np.allclose(np.diff(t), 0.5, atol=0.02)
    assert len(y) / SR == pytest.approx(1.9, abs=0.01)


def test_screwdriver_clutch_clicks_after_65_percent():
    y = M.power_tool(sr=HI, kind="tournevis", dur=2.0)
    a, b = y[:int(1.2 * HI)], y[int(1.35 * HI):int(1.95 * HI)]
    assert abs(env_top(b, HI, 15, 60) - 30.0) < 2.0
    assert share(np.abs(b) - np.abs(b).mean(), HI, 25, 35) > 3 * share(np.abs(a) - np.abs(a).mean(), HI, 25, 35)


def test_welder_buzzes_at_twice_the_mains():
    assert abs(env_top(M.power_tool(sr=SR, kind="soudeur", dur=2.0), SR, 50, 200) - 100.0) < 2.0


@pytest.mark.parametrize("rate", [20.0, 22.5, 25.0])
def test_jackhammer_blows(rate):
    assert abs(env_top(M.jackhammer(sr=SR, rate=rate, dur=2.0), SR, 10, 40) - rate) < 0.7


def test_jackhammer_rate_held_to_the_spec():
    assert abs(env_top(M.jackhammer(sr=SR, rate=60.0, dur=2.0), SR, 10, 40) - 25.0) < 0.7


def test_dental_drill_whine_is_rpm_over_60():
    late = lambda **kw: M.dental_drill(sr=HI, dur=2.5, **kw)[int(1.5 * HI):int(2.3 * HI)]     # after the 0.15 s spool-up
    assert abs(top(late(), HI, 3000, 8000) - 5000.0) < 5.0
    assert abs(top(late(rpm=240000.0), HI, 3000, 8000) - 4000.0) < 5.0
    loaded = late(load=1.0)
    assert abs(top(loaded, HI, 3000, 8000) - 3500.0) < 10.0


# ---------------------------------------------------------------------------------- buildings
def test_lift_motor_comes_up_and_the_door_is_servo_driven():
    y = M.elevator(sr=SR, kind="montee", dur=3.0)
    assert rms(y[:int(0.3 * SR)]) < 0.6 * rms(y[int(2.0 * SR):int(2.8 * SR)])
    x = M.servo_env(10000)
    fm = FO._move(10000)
    assert np.argmax(x >= 0.9) < 0.5 * np.argmax(fm >= 0.9)        # sharper start
    assert np.flatnonzero(x > 0.1)[-1] > np.flatnonzero(fm > 0.1)[-1]   # sharper stop
    assert x.max() <= 1.0                                          # no overspin


def test_lift_stop_lands_on_the_brake():
    y = M.elevator(sr=SR, kind="arret", dur=3.0)
    assert 2.9 < np.argmax(np.abs(y)) / SR < 3.2
    assert rms(y[int(2.5 * SR):int(2.9 * SR)]) < rms(y[:int(0.5 * SR)])


def test_lift_door_stops_at_its_end():
    y = M.elevator(sr=SR, kind="porte")
    assert 1.15 < np.argmax(np.abs(y)) / SR < 1.3


def test_lift_chime_is_a_risset_bell_ringing_out():
    y = M.elevator(sr=SR, kind="carillon")
    assert rms(y[int(0.9 * SR):]) < 0.5 * rms(y[:int(0.3 * SR)])
    assert 200 < top(y, SR, 100, 3000) < 2000


@pytest.mark.parametrize("v,step,rate", [(0.5, 0.4, 1.25), (0.75, 0.4, 1.875)])
def test_escalator_steps(v, step, rate):
    y = M.escalator(sr=SR, speed=v, step=step, dur=8.0)
    t, r = env_period(y, SR, 0.6 / rate, 1.4 / rate)
    assert t == pytest.approx(1.0 / rate, rel=0.03) and r > 0.2


def test_hvac_blade_pass_and_compressor():
    fan_only, cool = M.hvac(sr=HI, state="fan", dur=2.0), M.hvac(sr=HI, state="cool", dur=2.0)
    assert line_db(fan_only, HI, 150.0) > 8                        # 1800 rpm x 5 / 60
    assert share(cool, HI, 95, 105) > 1.5 * share(fan_only, HI, 95, 105)   # the compressor's hum at 2 x 50 Hz


def test_hvac_rattle_adds_ticks():
    a, b = M.hvac(sr=SR, state="cool", dur=2.0), M.hvac(sr=SR, state="rattle", dur=2.0)
    assert share(b, SR, 3500, 6000) > 2 * share(a, SR, 3500, 6000)


def test_conveyor_splice_rate_and_gear_mesh():
    y = M.conveyor(sr=SR, speed=1.5, joint=0.5, dur=6.0)
    t, r = env_period(y, SR, 0.2, 0.47)
    assert t == pytest.approx(1.0 / 3.0, rel=0.03) and r > 0.2     # v / joint
    z = M.conveyor(sr=HI, dur=1.0)
    assert line_db(z, HI, 1450.0 * 20 / 60.0, 0.01) > 8            # drive pinion, 20 teeth


def test_hard_drive_spindle_is_120_hz():
    y = M.hard_drive(sr=HI, state="idle", dur=2.0)
    assert abs(top(y, HI, 90, 150) - 120.0) < 1.0
    assert abs(top(M.hard_drive(sr=HI, state="idle", rpm=5400.0, dur=2.0), HI, 70, 120) - 90.0) < 1.0


def test_hard_drive_click_lasts_2_ms():
    k = M.head_click(np.random.default_rng(0))
    assert len(k) == pytest.approx(0.002 * M.P, abs=1)
    y = M.hard_drive(sr=HI, state="seek", seeks=8.0, dur=1.0)
    i = np.argmax(np.abs(y))
    assert np.abs(y[i + int(0.003 * HI):i + int(0.006 * HI)]).max() < 0.25 * np.abs(y[i])


def test_hard_drive_spinup_rises():
    y = M.hard_drive(sr=HI, state="spinup", dur=2.0)
    assert abs(top(y[:HI // 4], HI, 10, 150) - 120.0) > 10.0
    assert abs(top(y[-HI // 2:-HI // 20], HI, 60, 150) - 120.0) < 3.0


# ---------------------------------------------------------------------------------- gym, rides, slots
def test_treadmill_footfalls_and_bike_cadence():
    y = M.gym(sr=SR, kind="tapis", rate=170.0, dur=5.0)
    assert abs(env_top(y, SR, 1.5, 5.0) - 170.0 / 60.0) < 0.15
    assert abs(env_top(M.gym(sr=SR, kind="velo", rate=70.0, dur=8.0), SR, 0.5, 3.0) - 70.0 / 60.0) < 0.1


def test_weights_land_after_they_lift():
    y = M.gym(sr=SR, kind="poids", rate=15.0, dur=5.0)
    t = peaks(y, SR, gap=1.0, floor=0.5)
    assert len(t) >= 1 and abs(t[0] - 0.55 * 4.0) < 0.1


def test_coaster_lift_then_drop():
    y = M.ride(sr=SR, kind="montagnes russes", dur=10.0)
    lift, drop = y[:int(2.9 * SR)], y[int(4.0 * SR):int(9.0 * SR)]
    assert abs(env_top(lift, SR, 2, 12) - M.LIFT_CLACKS) < 0.4
    assert rms(drop) > 1.5 * rms(lift)
    assert share(drop, SR, 1000, 5000) > 0.05                      # the screams


def test_carousel_turns():
    y = M.ride(sr=SR, kind="manege", dur=16.0)
    near = np.concatenate([y[int(1 * SR):int(3 * SR)], y[int(9 * SR):int(11 * SR)]])     # crowd facing us
    far = np.concatenate([y[int(5 * SR):int(7 * SR)], y[int(13 * SR):int(15 * SR)]])
    assert rms(near) > 1.1 * rms(far)


def test_slot_stop_one_thunk_per_reel():
    for r in (3, 5):
        y = M.slot_machine(sr=SR, state="stop", reels=r)
        t = peaks(y, SR, gap=0.2, floor=0.4)
        assert len(t) == r and np.allclose(np.diff(t), 0.35, atol=0.02)


def test_slot_play_runs_in_order():
    win, lose = M.slot_machine(sr=SR, state="play", win=1), M.slot_machine(sr=SR, state="play", win=0)
    tail = int((0.5 + 1.2 + 3 * 0.35 + 0.2) * SR)
    assert rms(win[tail:]) > 3 * rms(lose[tail:])


# ---------------------------------------------------------------------------------- pumps, sewing, medical
@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_aquarium_membrane_twice_the_mains(mains):
    y = M.pump(sr=HI, kind="aquarium", mains=mains, dur=2.0)
    assert abs(top(y, HI, 60, 200) - 2 * mains) < 1.0


@pytest.mark.parametrize("kind,rate", [("puits", 0.8), ("air", 4.0)])
def test_pump_strokes(kind, rate):
    # `piston` strikes twice a cycle: the strokes sit at 2 x rate, the cycle at rate
    y = M.pump(sr=SR, kind=kind, dur=8.0)
    e = env_top(y, SR, 0.5 * rate, 2.5 * rate)
    assert min(abs(e - rate), abs(e - 2 * rate)) < 0.2


def test_sewing_is_800_stitches_a_minute():
    y = M.sewing_machine(sr=SR, dur=3.0)
    assert abs(env_top(y, SR, 5, 22) - 800.0 / 60.0) < 0.5


@pytest.mark.parametrize("f", [440.0, 880.0])
def test_beep_has_five_harmonics(f):
    y = M.medical(sr=HI, kind="bip", freq=f)
    assert abs(top(y, HI, 0.8 * f, 1.2 * f) - f) < 6.0
    s, Y = spec(y, HI)
    for k in range(1, 6):
        assert Y[(s > k * f - 15) & (s < k * f + 15)].max() > 1e-2 * Y.max(), k
    assert Y[(s > 6.5 * f) & (s < 7.5 * f)].max() < 1e-4 * Y.max()


def test_beep_range_is_held():
    assert abs(top(M.medical(sr=HI, kind="bip", freq=4000.0), HI, 700, 1300) - 1000.0) < 6.0
    assert abs(top(M.medical(sr=HI, kind="bip", freq=20.0), HI, 100, 200) - 150.0) < 6.0


def test_ekg_rate_and_monitor_pattern():
    y = M.medical(sr=SR, kind="EKG", rate=60.0, dur=5.0)
    assert np.allclose(np.diff(peaks(y, SR, gap=0.5)), 1.0, atol=0.02)
    t = peaks(M.medical(sr=SR, kind="moniteur", dur=5.0), SR, gap=0.2)
    assert len(t) == 6 and np.allclose(np.diff(t[:3]), 0.3, atol=0.02)    # medium priority: 150 ms pulses 150 ms apart
    assert abs(t[3] - t[0] - 3.25) < 0.03                                   # 2.5 s after the burst


def test_alarm_pulse_rises_over_15_percent_of_its_length():
    y = np.abs(M._pulse(880.0, 0.15))
    e = int(0.15 * len(y))
    assert y[:e // 3].max() < 0.5 * y.max()                                 # 10 ms edges would be at full level here
    assert y[e:2 * e].max() > 0.8 * y.max()


def test_ventilator_breathes_at_its_rate():
    y = M.medical(sr=SR, kind="respirateur", rate=15.0, dur=8.0)
    a, b = rms(y[:int(1.5 * SR)]), rms(y[int(3.6 * SR):int(3.95 * SR)])
    assert a > 1.5 * b                                              # end of the cycle before the next breath at 4 s
    assert rms(y[int(4.0 * SR):int(5.5 * SR)]) > 1.5 * b


# ---------------------------------------------------------------------------------- industry
def test_press_rate_and_factory_period():
    assert abs(env_top(M.industrial(sr=SR, kind="presse", rate=1.0, dur=6.0), SR, 0.5, 2.0) - 1.0) < 0.1
    t, r = env_period(M.industrial(sr=SR, kind="usine", dur=10.0), SR, 0.8, 1.7)
    assert t == pytest.approx(1.25, rel=0.03) and r > 0.2


def test_factory_keeps_tooth_friction():
    z = M.industrial(sr=HI, kind="usine", dur=1.0)
    assert line_db(z, HI, 1450.0 * 23 / 60.0, 0.01) > 8


def test_crater_law_and_times():
    assert M.crater_radius(0.3, 0.45) == pytest.approx(0.9)
    a = M.forge_contact_time(200.0, 5.0, 0.3, 1.15e11)
    b = M.forge_contact_time(200.0, 5.0, 0.9, 1.15e11)
    assert b / a == pytest.approx(3.0 ** -0.2)                      # stiffer contact in the crater
    # equal steel spheres, R .1 m, 1 m/s: Love Tc = 5.6 (R^5 / (c0^4 V))^(1/5), chunk_30 line 36
    rho, e, nu, R = 7800.0, 2.1e11, 0.3, 0.1
    love = 5.6 * (R ** 5 / (e / rho) ** 2) ** 0.2
    hertz = M.forge_contact_time(4 / 3 * math.pi * R ** 3 * rho / 2, 1.0, R / 2, e / (2 * (1 - nu ** 2)))
    assert hertz == pytest.approx(love, rel=0.03)
    tc1, tc2 = M.forge_times()
    assert tc1 > tc2
    f = M._forge_force(M.P, [0.1, 0.5, 0.9])
    w = [np.sum(np.abs(f[int(t * M.P):int((t + 0.02) * M.P)]) > 1e-6) / M.P for t in (0.1, 0.5, 0.9)]
    assert w[0] == pytest.approx(tc1, abs=1e-4) and w[1] == pytest.approx(tc2, abs=1e-4) and w[2] == w[1]


def test_forge_first_blow_dull_next_clear_and_alike():
    y = M.industrial(sr=HI, kind="marteau-pilon", rate=0.8, dur=4.6)
    cut = [y[int(t * HI):int((t + 0.1) * HI)] for t in (0.1, 1.35, 2.6)]    # the blow itself
    a, b, c = (centroid(x, HI) for x in cut)
    assert b > 1.2 * a and abs(c - b) < 0.1 * b                     # clearer, then every reload the same


# ---------------------------------------------------------------------------------- tooth friction
def test_tooth_sliding_changes_sign_at_the_pitch_point():
    rpm, z1, z2 = 1800.0, 20, 40
    w = 2 * math.pi * rpm / 60.0
    assert M.sliding_speed(rpm, z1, z2, 0.0) == 0.0
    assert M.sliding_speed(rpm, z1, z2, 0.01) == pytest.approx((w + w / 2) * 0.01)
    assert M.sliding_speed(rpm, z1, z2, -0.01) == -M.sliding_speed(rpm, z1, z2, 0.01)


def test_tooth_friction_is_a_sawtooth_at_the_mesh():
    y = M.tooth_friction(1800.0, 20, M.P, load=1.0, mu=0.1)
    assert abs(top(y, M.P, 100, 2000) - 600.0) < 2.0
    f, Y = spec(y, M.P)
    h2 = Y[(f > 1190) & (f < 1210)].max()
    assert h2 / Y.max() == pytest.approx(0.25, abs=0.03)            # 1/k
    assert np.abs(np.mean(y)) < 1e-3                                # as much one way as the other
    assert 0.08 < np.abs(y).max() < 0.13                            # mu x load
    assert np.abs(M.tooth_friction(1800.0, 20, M.P, load=0.5, mu=0.1)).max() < 0.7 * np.abs(y).max()


# ---------------------------------------------------------------------------------- belt
def test_capstan_numbers():
    assert M.capstan_ratio(0.5, math.pi) == pytest.approx(4.81, abs=0.01)
    assert M.slip_torque(0.5, math.pi) == pytest.approx(0.656, abs=0.001)
    assert M.stribeck_speed(0.1, 0.1, 1000.0) == pytest.approx(0.041, abs=0.001)


def _pulley_share(y, sr=HI, f=2800.0):
    return rms(np.asarray(M._band(y, 0.9 * f, 1.1 * f, sr)))


def test_belt_is_silent_under_the_torque_threshold():
    mc = 0.656 * 2 * 0.06 * 150.0                                    # 11.8 N m
    below = M.belt_squeal(sr=HI, torque=0.95 * mc, rise=0.1, dur=1.0)
    above = M.belt_squeal(sr=HI, torque=1.3 * mc, rise=0.1, dur=1.0)
    assert 20 * np.log10(_pulley_share(above) / _pulley_share(below)) > 20
    assert abs(top(above, HI, 2000, 3500) - 2800.0) < 60.0           # on the pulley's mode


def test_belt_onset_follows_the_torque_rise():
    y = M.belt_squeal(sr=HI, torque=23.6, rise=1.0, dur=1.6)         # crosses 11.8 N m at 0.5 s
    e = env(np.asarray(M._band(y, 2500.0, 3100.0, HI)), HI, 10)
    floor = e[int(0.05 * HI):int(0.4 * HI)].max()
    assert np.argmax(e > 10 * floor) / HI == pytest.approx(0.5, abs=0.1)
    assert e[int(1.2 * HI)] > 100 * floor


def test_belt_threshold_moves_with_wrap_and_grip():
    # more wrap = more grip: the same torque no longer slips
    tight = M.belt_squeal(sr=HI, torque=15.0, wrap=1.5 * math.pi, rise=0.1, dur=0.8)
    loose = M.belt_squeal(sr=HI, torque=15.0, rise=0.1, dur=0.8)
    assert _pulley_share(loose) > 10 * _pulley_share(tight)


def test_oil_changes_the_stribeck_speed():
    assert not np.array_equal(M.belt_squeal(sr=SR, oil=0.1), M.belt_squeal(sr=SR, oil=5.0))


# ---------------------------------------------------------------------------------- Savart
def _pitch(y, sr, lo=60.0, hi=2000.0):
    """Period by autocorrelation: the first strong lag, so a harmonic near the reed does not read as the pitch."""
    y = y[len(y) // 4:len(y) // 4 + int(0.2 * sr)]
    c = np.correlate(y, y, "full")[len(y) - 1:]
    a, b = int(sr / hi), int(sr / lo)
    return sr / (a + np.argmax(c[a:b]))


@pytest.mark.parametrize("teeth,rpm", [(60, 600.0), (30, 400.0), (100, 180.0)])
@pytest.mark.parametrize("reed", [700.0, 1500.0, 3000.0])
def test_savart_pitch_is_imposed_not_the_reed(teeth, rpm, reed):
    f = teeth * rpm / 60.0
    y = M.savart_wheel(sr=HI, teeth=teeth, rpm=rpm, reed_f=reed, dur=0.6)
    assert line_db(y, HI, f, 0.02) > 10
    assert _pitch(y, HI, 0.6 * f, 1.5 * f) == pytest.approx(f, rel=0.03)


def test_savart_reed_sets_the_timbre_and_glide_moves_the_pitch():
    a, b = M.savart_wheel(sr=HI, reed_f=700.0, dur=0.5), M.savart_wheel(sr=HI, reed_f=3000.0, dur=0.5)
    assert centroid(b, HI) > 1.3 * centroid(a, HI)
    y = M.savart_wheel(sr=HI, rpm=300.0, rpm_end=1200.0, dur=2.0)
    assert line_db(y[:HI // 8], HI, 330.0, 0.1) > 5
    assert line_db(y[-HI // 8:], HI, 1140.0, 0.1) > 5
    assert not np.array_equal(M.savart_wheel(sr=SR, mount="cup", dur=0.5), M.savart_wheel(sr=SR, dur=0.5))
    with pytest.raises(ValueError):
        M.savart_wheel(sr=SR, mount="nope")
