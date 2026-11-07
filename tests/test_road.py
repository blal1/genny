"""The road: every variant renders; the numbers of the spec are measured on the rendered signal."""
import inspect

import numpy as np
import pytest

from genny import road as R
from genny.analysis import detect_clicks
from genny.cyber import _HN_AR, _HN_F, road_noise_levels
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


def rms(y):
    return float(np.sqrt(np.mean(np.square(y))))


def db(a, b):
    return 20 * np.log10(a / b)


def env_top(y, sr, lo, hi):
    """Strongest modulation of the magnitude between lo and hi Hz."""
    e = np.abs(y) - np.mean(np.abs(y))
    return top(e, sr, lo, hi)


# ---------------------------------------------------------------------------------- every variant
SMOOTH = [(R.avas, dict(rule="unece")), (R.avas, dict(rule="fmvss")), (R.ev_drive, {}), (R.hybrid_transition, {})]
SMOOTH += [(R.air_system, dict(kind=k)) for k in ("bus_kneel", "air_brake_apply", "air_brake_release", "air_dryer_purge", "parking_brake_spring")]
SMOOTH += [(R.tyre_roll, dict(kind=k, dur=1.0)) for k in ("tyre_road", "tyre_wet", "tyre_mud", "tyre_cobble", "tyre_burnout", "tyre_lockup", "road_cruise_interior")]
SMOOTH += [(R.car_door, dict(kind=k)) for k in ("car_door_close", "car_door_rear", "car_door_cheap", "trunk_slam", "hood_slam", "van_sliding_door")]
SMOOTH += [(R.air_horn, dict(kind=k)) for k in R.HORN_KINDS]
SMOOTH += [(R.klaxon, dict(kind=k)) for k in ("electric", "hand_crank")]
SMOOTH += [(R.backup_beeper, {}), (R.car_alarm, dict(seq="0")), (R.suspension, {})]
SMOOTH += [(R.ev, dict(kind=k, dur=1.0)) for k in ("voiture electrique", "trottinette", "voiturette")]
SMOOTH += [(R.interior, dict(kind=k, dur=1.0)) for k in R.INTERIORS]
# Attacks of their own (detect_clicks is meant for smooth voices): PhISEM grains, clicks and latches. The ends must still be clean.
GRAINY = [(R.tyre_roll, dict(kind=k, dur=1.0)) for k in ("tyre_gravel", "tyre_dirt", "tyre_sand", "tyre_snow")]
GRAINY += [(R.air_system, dict(kind="bus_door")), (R.car_door, dict(kind="car_door_open"))]
GRAINY += [(R.vehicle_mech, dict(kind=k)) for k in ("ceinture", "frein a main", "boite", "siege", "vitre")]


def _id(c):
    return f"{c[0].__name__}-{'-'.join(map(str, c[1].values()))}"


@pytest.mark.parametrize("case", SMOOTH, ids=_id)
def test_smooth_variant_renders_without_click(case):
    y = np.asarray(case[0](sr=SR, **case[1]))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert len(detect_clicks(y)) == 0


@pytest.mark.parametrize("case", GRAINY, ids=_id)
def test_percussive_variant_renders_with_clean_ends(case):
    y = np.asarray(case[0](sr=SR, **case[1]))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


def test_every_plan_call_is_in_the_catalog_and_names_are_checked():
    for name in ("avas", "ev_drive", "hybrid_transition", "air_system", "tyre_roll", "air_horn", "car_door", "klaxon", "vehicle_mech",
                 "suspension", "backup_beeper", "car_alarm", "ev", "interior"):
        assert name in REGISTRY
    for f in (R.ev, R.air_system, R.tyre_roll, R.car_door, R.air_horn, R.vehicle_mech, R.interior, R.klaxon):
        with pytest.raises(ValueError):
            f(sr=SR, kind="nope")
    with pytest.raises(ValueError):
        R.avas(sr=SR, rule="nope")


# ---------------------------------------------------------------------------------- electric drive
def test_ev_whine_contract_for_l08():
    assert list(inspect.signature(R.ev_whine).parameters)[:7] == ["speed", "poles", "slots", "fs", "load", "sr", "seed"]
    rpm = np.linspace(600.0, 3000.0, 4800)
    y = R.ev_whine(rpm, 4, 48, 2000.0, 0.5, 8000, 1)
    assert isinstance(y, np.ndarray) and y.shape == rpm.shape and np.isfinite(y).all() and np.abs(y).max() < 1.0
    assert np.array_equal(y, R.ev_whine(rpm, 4, 48, 2000.0, 0.5, 8000, 1))


def test_orders_24_48_72_96_at_6000_rpm():
    assert R.whine_orders(4, 48) == [24, 48, 72, 96]
    y = R.ev_whine(6000.0, 4, 48, 7500.0, 0.8, HI, 0, dur=0.5)
    for order in (24, 48, 72, 96):
        assert line_db(y, HI, order * 100.0) > 30, order
    assert abs(top(y, HI, 4500, 5100) - 4800.0) < 5.0          # 6000 rpm, order 48 = 4800 Hz


def test_pwm_sidebands_follow_k1_fs_k2_f0():
    y = R.ev_whine(3000.0, 4, 48, 6000.0, 0.8, HI, 0, dur=0.5)   # f0 = 200 Hz
    for f in (6000.0 - 400.0, 6000.0 + 400.0, 12000.0 + 200.0, 12000.0 - 200.0):
        assert line_db(y, HI, f, 0.002) > 20, f


def test_random_carrier_stays_in_7250_7750():
    y = R.ev_whine(300.0, 4, 48, 7500.0, 0.0, HI, 0, dur=1.0, fs_spread=250.0)   # f0 = 20 Hz: sidebands within 80 Hz
    assert share(y, HI, 7100, 7900) > 0.9 * share(y, HI, 6000, 9000)
    f, Y = spec(y, HI)
    spread = Y[(f > 7100) & (f < 7900)]
    assert np.sum(spread > 0.1 * spread.max()) > 20            # a band, not a line


def test_carrier_can_step_for_traction_drive():
    n = HI
    fs = np.where(np.arange(n) < n // 2, 500.0, 2000.0)
    y = R.ev_whine(100.0, 4, 48, fs, 0.0, HI, 0)
    a, b = y[:n // 2], y[n // 2:]
    assert share(a, HI, 400, 600) > 5 * share(a, HI, 1900, 2100) and share(b, HI, 1900, 2100) > 5 * share(b, HI, 400, 600)


def test_ev_drive_lines_at_60_kmh():
    y = R.ev_drive(sr=HI, speed=60.0, dur=1.0)
    rpm = float(R.motor_rpm(60.0))
    assert abs(rpm - 4342) < 5
    assert line_db(y, HI, 48 * rpm / 60.0) > 20
    assert line_db(y, HI, 17 * rpm / 60.0) > 15                # the reduction gear mesh, 17 teeth


def test_regen_changes_the_drive():
    assert not np.array_equal(R.ev_drive(sr=SR, dur=0.5, load=0.2, regen=1.0), R.ev_drive(sr=SR, dur=0.5, load=0.2))


def test_ev_scooter_whine_at_its_order():
    y = R.ev(sr=HI, kind="trottinette", dur=1.0)
    assert line_db(y, HI, 90.0 * float(R.motor_rpm(20.0, 1.0, 0.1)) / 60.0) > 20


# ---------------------------------------------------------------------------------- AVAS
@pytest.mark.parametrize("kmh", [5.0, 10.0, 15.0, 20.0])
def test_unece_pitch_rises_08_percent_per_kmh(kmh):
    y = R.avas(sr=HI, speed=kmh, style="hum", dur=1.0)
    assert abs(top(y, HI, 350, 600) / (400.0 * (1 + 0.008 * kmh)) - 1) < 0.003
    assert abs(top(y, HI, 1400, 2200) / (1600.0 * (1 + 0.008 * kmh)) - 1) < 0.003


def test_unece_levels():
    a = rms(R.avas(sr=SR, speed=10.0, style="hum"))
    b = rms(R.avas(sr=SR, speed=20.0, style="hum"))
    r = rms(R.avas(sr=SR, speed=6.0, reverse=1, style="hum"))
    assert abs(db(b, a) - 6.0) < 0.2                           # 50 -> 56 dB(A)
    assert abs(db(r, a) + 3.0) < 0.2                           # 47 dB(A) reversing
    assert 20 * np.log10(np.abs(R.avas(sr=SR, speed=20.0, style="hum")).max()) < -18.9    # 56 under the 75 maximum


def test_avas_glide_follows_speed():
    y = R.avas(sr=HI, speed=5.0, speed_end=20.0, style="hum", dur=2.0)
    a, b = top(y[:HI // 4], HI, 350, 600), top(y[-HI // 4:], HI, 350, 600)
    assert a < 425 and 450 < b < 470


def test_fmvss_no_glide_and_3_db_steps():
    ys = {v: R.avas(sr=HI, speed=v, rule="fmvss", style="hum", dur=1.0) for v in (5.0, 11.0, 21.0, 31.0)}
    for v, y in ys.items():
        assert abs(top(y, HI, 350, 600) - 400.0) < 2.0, v
    for a, b in ((5.0, 11.0), (11.0, 21.0), (21.0, 31.0)):
        assert abs(db(rms(ys[b]), rms(ys[a])) - 3.0) < 0.2


def test_hybrid_engine_catches_after_the_switch():
    y = R.hybrid_transition(sr=SR, at=1.0, dur=3.0)
    before, after = y[:int(0.9 * SR)], y[int(1.6 * SR):int(2.9 * SR)]
    assert rms(after) > rms(before)
    assert share(after, SR, 20, 400) > 3 * share(before, SR, 20, 400)
    assert abs(env_top(after, SR, 20, 60) - 40.0) < 2.0        # 4 cylinders at 1200 rpm


# ---------------------------------------------------------------------------------- tyres
def test_rolling_level_follows_harmonoise():
    a, b = R.tyre_roll(sr=SR, speed=70.0, tread=0.0), R.tyre_roll(sr=SR, speed=140.0, tread=0.0)
    expected = 10 * np.log10(np.sum(10 ** (road_noise_levels(140.0)[0] / 10)) / np.sum(10 ** (road_noise_levels(70.0)[0] / 10)))
    assert abs(db(rms(b), rms(a)) - expected) < 1.0


def test_rolling_bands_match_harmonoise_at_70():
    y = R.tyre_roll(sr=HI, speed=70.0, tread=0.0, dur=3.0)
    f, Y = spec(y, HI)
    i = np.flatnonzero((_HN_F > 100) & (_HN_F < 6000))
    lev = np.array([10 * np.log10(Y[(f >= _HN_F[k] / 2 ** (1 / 6)) & (f < _HN_F[k] * 2 ** (1 / 6))].sum()) for k in i])
    d = (lev - lev.mean()) - (_HN_AR[i] - _HN_AR[i].mean())
    assert np.abs(d).max() < 3.0


@pytest.mark.parametrize("kmh,lo,hi", [(70.0, 500.0, 1000.0), (140.0, 1000.0, 2000.0)])
def test_tread_line_band_scales_with_speed(kmh, lo, hi):
    y = R._tread_line(kmh, HI, np.random.default_rng(0))
    assert share(y, HI, lo, hi) > 0.5
    assert lo < top(y, HI, 200, 8000) < hi


def test_wet_adds_spray_above_2k():
    assert share(R.tyre_roll(sr=SR, kind="tyre_wet"), SR, 2000, 10000) > 1.3 * share(R.tyre_roll(sr=SR, kind="tyre_road"), SR, 2000, 10000)


def test_gravel_grains_at_cook_6460_hz():
    y = R._grains("BigRocks", 800.0, HI, np.random.default_rng(0))
    assert 6460 - 701 < top(y, HI, 2000, 12000) < 6460 + 701


def test_cobble_rate_is_v_over_l():
    y = R.tyre_roll(sr=SR, kind="tyre_cobble", speed=50.0, dur=3.0)
    assert abs(env_top(R._cobble(50.0, 3 * R.P, np.random.default_rng(0)), R.P, 60, 200) - 50.0 / 3.6 / 0.12) < 6.0
    assert np.isfinite(y).all()


def test_lockup_chops_at_15_hz():
    assert abs(env_top(R.tyre_roll(sr=SR, kind="tyre_lockup", dur=2.0), SR, 5, 40) - 15.0) < 1.0


@pytest.mark.parametrize("kmh", [30.0, 90.0])
def test_slip_relaxes_over_3a(kmh):
    step = R._slip_lag(np.ones(R.P), kmh)
    tau = np.searchsorted(step, 1 - np.exp(-1)) / R.P
    assert tau == pytest.approx(3 * R.CONTACT_A / (kmh / 3.6), rel=0.03)


def test_burnout_is_louder_and_longer_than_rolling():
    assert rms(R.tyre_roll(sr=SR, kind="tyre_burnout", dur=1.0)) > 2 * rms(R.tyre_roll(sr=SR, dur=1.0))


def test_cabin_takes_the_top_off():
    outside = R.tyre_roll(sr=SR, speed=90.0, dur=1.0)
    assert share(R.tyre_roll(sr=SR, kind="road_cruise_interior", speed=90.0, dur=1.0), SR, 1000, 10000) < 0.3 * share(outside, SR, 1000, 10000)
    assert share(R.interior(sr=SR, speed=90.0, dur=1.0), SR, 1000, 10000) < 0.3 * share(outside, SR, 1000, 10000)


# ---------------------------------------------------------------------------------- air
def test_dryer_is_125_db_over_the_brake():
    a = np.abs(R.air_system(sr=SR, kind="air_dryer_purge")).max()
    b = np.abs(R.air_system(sr=SR, kind="air_brake_apply")).max()
    assert abs(db(a, b) - 12.5) < 0.5                          # midpoints of 115-120 and 95-115 dB


def test_kneel_lasts_longer_than_brake_longer_than_purge():
    def t90(y):
        e = np.cumsum(y * y)
        return np.searchsorted(e, 0.9 * e[-1]) / SR
    k, b, d = (t90(R.air_system(sr=SR, kind=x)) for x in ("bus_kneel", "air_brake_apply", "air_dryer_purge"))
    assert k > 2 * b > 2 * d


def test_release_has_the_membrane_at_150_hz():
    y = R.air_system(sr=SR, kind="air_brake_release")
    assert abs(env_top(y, SR, 80, 300) - 150.0) < 5.0


def test_bus_door_stop_lands_within_the_stroke():
    y = R.air_system(sr=SR, kind="bus_door")
    e = np.abs(y)
    assert 0.9 < np.argmax(e[int(0.5 * SR):int(1.2 * SR)]) / SR + 0.5 < 1.2
    assert len(y) / SR == pytest.approx(R.DOOR_STROKE + 0.5, abs=0.01)


def test_parking_spring_lands_after_the_air():
    y = R.air_system(sr=SR, kind="parking_brake_spring")
    t = np.argmax(np.abs(y)) / SR
    assert 0.1 < t < 0.3


# ---------------------------------------------------------------------------------- doors
def test_front_door_energy_in_200_2000():
    assert share(R.car_door(sr=SR, kind="car_door_close"), SR, 200, 2000) > 0.7


def test_rear_door_energy_in_315_2500_and_higher():
    y = R.car_door(sr=SR, kind="car_door_rear")
    assert share(y, SR, 315, 2500) > 0.7
    assert centroid(y, SR) > centroid(R.car_door(sr=SR, kind="car_door_close"), SR)


def test_cheap_door_is_tinny():
    a, b = R.car_door(sr=SR, kind="car_door_close"), R.car_door(sr=SR, kind="car_door_cheap")
    assert share(b, SR, 500, 10000) > 0.7 > share(a, SR, 500, 10000)
    assert share(a, SR, 50, 500) > 2 * share(b, SR, 50, 500)


def test_quality_knob_moves_the_balance():
    tin, solid = R.car_door(sr=SR, quality=0.0), R.car_door(sr=SR, quality=1.0)
    assert share(solid, SR, 50, 500) > share(tin, SR, 50, 500)


def test_door_lasts_200_ms_with_two_hits_20_ms_apart():
    y = R.car_door(sr=HI, kind="car_door_close")
    e = np.cumsum(y * y)
    assert np.searchsorted(e, 0.99 * e[-1]) / HI < 0.2
    win = int(0.002 * HI)
    env = np.convolve(np.abs(y), np.ones(win) / win, "same")
    assert env[int(0.012 * HI):int(0.019 * HI)].min() < 0.5 * env[int(0.02 * HI):int(0.03 * HI)].max()   # the latch, a dip, the panel


def test_boot_and_bonnet_are_lower_than_a_door():
    door = centroid(R.car_door(sr=SR), SR)
    assert centroid(R.car_door(sr=SR, kind="trunk_slam"), SR) < door
    assert centroid(R.car_door(sr=SR, kind="hood_slam"), SR) < door


def test_van_door_runs_then_stops():
    y = R.car_door(sr=SR, kind="van_sliding_door")
    assert rms(y[int(0.2 * SR):int(0.7 * SR)]) > 0.01 and np.argmax(np.abs(y)) / SR > 0.85


# ---------------------------------------------------------------------------------- horns
@pytest.mark.parametrize("kind,tones", [("train_horn_k5la", (311, 370, 415, 494, 622)), ("truck_air_horn", (150, 180)), ("air_horn_car", (370, 415)),
                                        ("train_horn_euro", (370, 440)), ("vuvuzela", (235, 465))])
def test_horn_tones_are_in_the_signal(kind, tones):
    y = R.air_horn(sr=HI, kind=kind, dur=2.0)[HI // 4:-HI // 2]
    for f in tones:
        assert line_db(y, HI, f, 0.012) > 15, f


def test_freqs_override_truck_285_330():
    y = R.air_horn(sr=HI, kind="truck_air_horn", freqs="285/330", dur=1.5)[HI // 4:-HI // 2]
    assert line_db(y, HI, 285, 0.012) > 15 and line_db(y, HI, 330, 0.012) > 15


@pytest.mark.parametrize("pair,tones", [(0, (335, 400)), (1, (500, 412)), (2, (430, 355))])
def test_car_pairs(pair, tones):
    y = R.air_horn(sr=HI, kind="car_pairs", pair=pair, dur=1.0)
    for f in tones:
        assert line_db(y, HI, f, 0.005) > 15, f


def test_horn_rises_in_80_ms_and_sags_at_the_end():
    y = R.air_horn(sr=SR, kind="truck_air_horn", freqs="150", dur=1.0)
    win = int(0.01 * SR)
    e = np.convolve(np.abs(y), np.ones(win) / win, "same")
    steady = e[int(0.3 * SR):int(0.7 * SR)].mean()
    assert e[int(0.02 * SR)] < 0.85 * steady and e[int(0.12 * SR)] > 0.7 * steady
    assert e[int(0.97 * SR)] < 0.5 * steady


@pytest.mark.parametrize("kind", ["electric", "hand_crank"])
def test_klaxon_wheel_goes_100_to_500_and_back(kind):
    r = R.klaxon_rate(HI, kind)
    assert r.min() == pytest.approx(100.0, abs=30.0) and 450 < r.max() < 560
    assert r[len(r) // 2] > r[0] and r[len(r) // 2] > r[-1]


def test_klaxon_rendered_rate_at_the_top():
    y = R.klaxon(sr=HI, dur=1.2)
    a = y[int(0.5 * HI):int(0.7 * HI)]
    c = np.correlate(a, a, "full")[len(a) - 1:]
    lag = int(HI / 700) + np.argmax(c[int(HI / 700):int(HI / 350)])
    assert 430 < HI / lag < 560                                # impacts per second at the top of the sweep
    b = y[int(0.02 * HI):int(0.15 * HI)]
    c = np.correlate(b, b, "full")[len(b) - 1:]
    lag = int(HI / 450) + np.argmax(c[int(HI / 450):int(HI / 80)])
    assert HI / lag < 300                                      # slow at the start


def test_hand_crank_differs_and_follows_crank():
    assert R.klaxon_rate(HI, "hand_crank", 0.5).max() < R.klaxon_rate(HI, "hand_crank", 1.0).max()
    assert not np.array_equal(R.klaxon_rate(HI, "hand_crank"), R.klaxon_rate(HI, "electric"))


@pytest.mark.parametrize("freq", [1000.0, 1250.0])
def test_beeper_pitch_rate_duty(freq):
    y = R.backup_beeper(sr=HI, freq=freq, rate=1.0, duty=0.5, dur=4.0)
    assert abs(top(y, HI, 500, 3000) - freq) < 3.0
    assert abs(env_top(y, HI, 0.5, 3.0) - 1.0) < 0.3
    assert np.mean(np.abs(y) > 1e-3) == pytest.approx(0.5, abs=0.1)


def test_beeper_range_is_clamped_to_j994():
    assert abs(top(R.backup_beeper(sr=HI, freq=5000.0, dur=1.0), HI, 500, 4000) - 2800.0) < 5.0


def test_alarm_figures_length_and_band():
    assert len(R.car_alarm(sr=SR, seq="01", motif=3.0)) == pytest.approx(6 * SR, abs=2)
    for d in "012345":
        assert share(R.car_alarm(sr=SR, seq=d, motif=3.0), SR, 900, 3200) > 0.7, d


def test_load_changes_the_suspension():
    assert not np.array_equal(R.suspension(sr=SR, load=0.0), R.suspension(sr=SR, load=1.0))


def test_clamped_membrane_modes_are_leissa_ratios():
    lam2 = np.array([10.2158, 21.260, 34.877, 39.771, 51.030, 60.829])
    assert np.allclose(R.CLAMPED, lam2 / lam2[0], atol=2e-3)
    x = np.zeros(HI)
    x[10] = 1.0
    y = R._modes(x, 620.0 * np.array(R.CLAMPED), 0.3, np.ones(6))
    assert abs(top(y, HI, 2060.0, 2300.0) - 620.0 * 34.877 / 10.2158) < 3.0


def test_scooter_hub_motor_has_27_slots():
    assert R.EV["trottinette"][3:5] == (15, 27)
    y = R.ev(sr=HI, kind="trottinette", dur=1.0)
    assert line_db(y, HI, 27.0 * float(R.motor_rpm(20.0, 1.0, 0.1)) / 60.0) > 10


def _two_tone(sr=HI):
    t = np.arange(sr) / sr
    return np.sin(2 * np.pi * 100.0 * t) + np.sin(2 * np.pi * 1000.0 * t)


def test_air_speaker_radiates_the_flow_derivative_through_its_horn():
    y = R.air_speaker(_two_tone(), HI, depth=0.3, horn=300.0, hiss=0.0)
    f, Y = spec(y, HI)
    a = lambda f0: Y[(f > f0 - 3) & (f < f0 + 3)].max()
    expect = 20 * np.log10(10.0) + 10 * np.log10(1 + (300.0 / 100.0) ** 4) - 10 * np.log10(1 + (300.0 / 1000.0) ** 4)
    assert abs(10 * np.log10(a(1000.0) / a(100.0)) - expect) < 1.0


def test_air_speaker_valve_shuts_when_driven_hard():
    t = np.arange(HI) / HI
    x = np.sin(2 * np.pi * 500.0 * t)
    soft, hard = R.air_speaker(x, HI, depth=0.3, hiss=0.0), R.air_speaker(x, HI, depth=1.5, hiss=0.0)
    ratio = lambda y: (lambda F, Y: Y[(F > 995) & (F < 1005)].max() / Y[(F > 495) & (F < 505)].max())(*spec(y, HI))
    assert 10 * np.log10(ratio(soft)) < -40 and 10 * np.log10(ratio(hard)) > -15
