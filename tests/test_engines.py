"""genny.engines: the firing tables and their signature on the rendered signal, and every variant of the lot renders."""
import numpy as np
import pytest
from scipy.signal import hilbert

import genny.cyber as c
import genny.engines as e
import genny.vehicle as v
from genny.analysis import detect_clicks
from genny.sfx import REGISTRY

SR = c.SR
KINDS = [k for k in e.LAYOUTS if not k.startswith("diesel")]


def _spec(y):
    N = 1 << int(np.ceil(np.log2(len(y))) + 1)
    return np.abs(np.fft.rfft(y * np.hanning(len(y)), N)), np.fft.rfftfreq(N, 1.0 / SR)


def _db(sp, fr, f, bw=1.5):
    return 20 * np.log10(sp[(fr > f - bw) & (fr < f + bw)].max() + 1e-12)


def _band(sp, fr, a, b):
    return 10 * np.log10(np.sum(sp[(fr >= a) & (fr < b)] ** 2) + 1e-12)


def _comb(y, f):
    """How far the multiples of ``f`` stand above the points halfway between them (median of 6, dB)."""
    sp, fr = _spec(y)
    bw = max(1.0, f * 0.03)
    return float(np.median([_db(sp, fr, k * f, bw) - _db(sp, fr, (k + 0.5) * f, bw) for k in range(1, 7)]))


def _rms(y):
    return float(np.sqrt(np.mean(y * y)))


def _env_line(y, f, bw=1.5):
    """Level of an envelope line against the envelope's median spectrum."""
    env = np.abs(hilbert(y))
    sp, fr = _spec(env - env.mean())
    return _db(sp, fr, f, bw) - 20 * np.log10(np.median(sp[(fr > 1) & (fr < 4 * f)]) + 1e-12)


def _ends_clean(y):
    assert abs(y[0]) < 1e-6 and abs(y[-1]) < 1e-6
    # A real click is a jump across the range, not the corner of a pulse (detect_clicks is meant for smooth voices).
    assert np.abs(np.diff(y)).max() < np.abs(y).max()


# ---------------------------------------------------------------- firing tables

def test_v8_crossplane_fires_180_90_180_270_per_bank():
    b = e.bank_intervals("v8_crossplane")
    assert b["L"] == [180.0, 90.0, 180.0, 270.0]
    assert sorted(b["R"]) == [90.0, 180.0, 180.0, 270.0]
    assert np.diff(np.array(e.firing_offsets("v8_crossplane")) * 720.0).tolist() == [90.0] * 7


def test_flat_plane_alternates_banks_evenly():
    assert e.bank_intervals("v8_flatplane") == {"L": [180.0] * 4, "R": [180.0] * 4}


@pytest.mark.parametrize("kind", ["boxer_6", "v6_60"])
def test_six_cylinders_fire_240_per_bank(kind):
    assert e.bank_intervals(kind) == {"L": [240.0] * 3, "R": [240.0] * 3}


def test_intervals_from_the_spec():
    deg = lambda k: np.round(np.diff(np.r_[e.firing_offsets(k), 1.0]) * 720.0, 6).tolist()
    assert deg("vtwin_45") == [315.0, 405.0]
    assert deg("vtwin_90") == [270.0, 450.0]
    assert deg("twin_180") == [180.0, 540.0]
    assert deg("twin_270") == [270.0, 450.0]
    assert deg("twin_360") == [360.0, 360.0]
    assert deg("inline_4_crossplane") == [90.0, 180.0, 270.0, 180.0]
    assert deg("inline_5") == [144.0] * 5
    assert deg("v10") == [72.0] * 10
    assert deg("v12") == [60.0] * 12
    assert np.allclose(deg("v6_90"), [90, 150] * 3)


def test_ptr_routing_already_splits_the_banks_by_cylinder_number():
    # vehicle._ptr_engine: out[(c - 1) * banks // cyl], the correction the plan asks for: (c-1) // 4 and (c-1) // 3.
    def per_bank(cyl):
        order = v._FIRING[cyl]
        out = {0: [], 1: []}
        for slot, cy in enumerate(order):
            out[(cy - 1) * 2 // cyl].append(slot * 720.0 / cyl)
        return {b: [(t[(i + 1) % len(t)] - t[i]) % 720.0 for i in range(len(t))] for b, t in out.items()}
    p8 = per_bank(8)
    assert sorted(p8[0]) == [90.0, 180.0, 180.0, 270.0] and sorted(p8[1]) == [90.0, 180.0, 180.0, 270.0]
    assert per_bank(6) == {0: [240.0] * 3, 1: [240.0] * 3}


def test_unknown_layout_is_refused():
    with pytest.raises(ValueError):
        e.firing_offsets("v7")
    with pytest.raises(ValueError):
        e.engine_layout(sr=SR, kind="v7")
    with pytest.raises(ValueError):
        e.diesel_engine(sr=SR, kind="v8_crossplane")
    with pytest.raises(ValueError):
        e.forced_induction(sr=SR, kind="nitro")
    with pytest.raises(ValueError):
        e.engine_event(sr=SR, kind="sneeze")


# ---------------------------------------------------------------- the signal

@pytest.mark.parametrize("kind", KINDS)
def test_every_layout_renders(kind):
    y = e.engine_layout(sr=22050, kind=kind, dur=1.0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1 and len(y) == 22050
    _ends_clean(y)


@pytest.mark.parametrize("kind", ["boxer_4", "boxer_6", "inline_4", "inline_4_crossplane", "inline_5", "inline_6", "v6_60", "v6_90", "v8_crossplane",
                                  "v8_flatplane", "v10", "v12", "wankel"])
def test_many_cylinders_have_no_click(kind):
    assert len(detect_clicks(e.engine_layout(sr=SR, kind=kind, dur=1.0))) == 0


@pytest.mark.parametrize("kind, rpm, hz, lines", [("engine_single", 3000.0, 50.0, 2), ("boxer_4", 700.0, 46.667, 2), ("inline_5", 6000.0, 250.0, 5), ("v8_crossplane", 6000.0, 400.0, 8),
                                                  ("v8_flatplane", 9000.0, 600.0, 2), ("v10", 9000.0, 750.0, 2), ("v12", 8000.0, 800.0, 2), ("wankel", 9000.0, 300.0, 1), ("inline_6", 3000.0, 150.0, 6)])
def test_firing_rate_on_the_signal(kind, rpm, hz, lines):
    # A line at the firing rate (its double for the two slowest, under the 25 Hz high-pass), and nothing halfway to the next line of
    # the pattern, which repeats ``lines`` times per firing period at most: half-orders of banks and runners are real lines.
    sp, fr = _spec(e.engine_layout(sr=SR, kind=kind, rpm=rpm, dur=4.0))
    assert _db(sp, fr, hz) > _db(sp, fr, hz * (1.0 + 0.5 / lines)) + 10


@pytest.mark.parametrize("kind, null, strong", [("twin_180", 2, 3), ("twin_270", 4, 2), ("twin_360", 3, 2), ("vtwin_90", 4, 2), ("vtwin_45", 8, 2)])
def test_a_twin_cancels_exactly_the_harmonics_its_interval_predicts(kind, null, strong):
    # Two equal pulses at offset o sum to |1 + exp(2 pi i k o)| on harmonic k of rpm / 120: zero where k o is a half-integer.
    y = e.engine_layout(sr=SR, kind=kind, rpm=2400.0, dur=4.0)
    sp, fr = _spec(y)
    assert _db(sp, fr, strong * 20.0) > _db(sp, fr, null * 20.0) + 15


def test_lope_1_is_a_180_540_twin():
    a = c.pipe_engine(sr=SR, dur=4.0, rpm=2400.0, cylinders=2, lope=1.0, unequal=0.0)
    sp, fr = _spec(a)
    assert _db(sp, fr, 60.0) > _db(sp, fr, 40.0) + 15


def test_crossplane_rumbles_where_the_flat_plane_does_not():
    def odd(kind):
        sp, fr = _spec(e.engine_layout(sr=SR, kind=kind, rpm=1800.0, dur=4.0))
        return max(_db(sp, fr, k * 15.0) for k in (3, 5)) - _db(sp, fr, 120.0)
    assert odd("v8_crossplane") > odd("v8_flatplane") + 20


def test_unequal_boxer_runners_bring_the_odd_orders():
    even = dict(e.LAYOUTS["boxer_4"], header=(1.8, 1.8, 1.8, 1.8))
    def half(y):                                             # short, short, long, long repeats once per 720 degrees: odd multiples of rpm / 120
        sp, fr = _spec(y)
        return max(_db(sp, fr, k * 20.0) for k in (3, 5, 7, 9)) - max(_db(sp, fr, k * 80.0) for k in (1, 2, 3))
    boxer = e.engine_layout(sr=SR, kind="boxer_4", rpm=2400.0, dur=4.0)
    plain = c.pipe_engine(sr=SR, dur=4.0, rpm=2400.0, offsets=even["offsets"], headers=even["header"])
    assert half(boxer) > half(plain) + 15


def test_wankel_has_no_half_order():
    sp, fr = _spec(e.engine_layout(sr=SR, kind="wankel", rpm=9000.0, dur=4.0))
    assert _db(sp, fr, 300.0) > _db(sp, fr, 150.0) + 15


def test_two_stroke_fires_once_per_turn():
    y = e.two_stroke(sr=SR, rpm=7000.0, dur=4.0)
    sp, fr = _spec(y)
    low = (fr > 20) & (fr < 300)
    assert abs(fr[low][np.argmax(sp[low])] - 7000.0 / 60.0) < 1.5
    assert _db(sp, fr, 116.667) > _db(sp, fr, 58.333) + 25
    assert np.isfinite(y).all()
    _ends_clean(y)


def test_two_cylinder_two_stroke_doubles_the_rate():
    assert _comb(e.two_stroke(sr=SR, rpm=6000.0, cylinders=2, dur=4.0), 200.0) > 15


def test_expansion_chamber_returns_suction_then_a_push():
    lens, r, g = e.expansion_chamber(0.05)
    assert (r[:7] < 0).all() and (r[7:] > 0).all()          # diffuser widens, counter-cone narrows
    assert len(lens) == 14 and (g < 1).all() and (g > 0.99).all()
    # .2 m at sqrt(1.36 x 287 x 800) = 559 m/s is 17 samples at 48 kHz; the shorter chamber of a smaller bore keeps its lengths.
    assert lens[0] == round(0.2 / np.sqrt(1.36 * 287 * 800) * SR)


def test_chamber_changes_the_tone():
    bare = c.pipe_engine(sr=SR, rpm=7000.0, cylinders=1, cycle=2, header=0.3, pipe=0.5, unequal=0.0, dur=2.0)
    tuned = e.two_stroke(sr=SR, rpm=7000.0, load=0.5, dur=2.0)
    assert not np.allclose(bare[:len(tuned)], tuned, atol=1e-3)


# ---------------------------------------------------------------- diesel

def test_truck_diesel_at_1800_rpm_is_90_hz():
    y = e.diesel_engine(sr=SR, kind="diesel_i6_truck", rpm=1800.0, dur=4.0)
    sp, fr = _spec(y)
    low = (fr > 20) & (fr < 200)
    assert abs(fr[low][np.argmax(sp[low])] - 90.0) < 1.0
    assert _comb(y, 90.0) > 15


def test_truck_idle_600_is_30_hz():
    assert _comb(e.diesel_engine(sr=SR, kind="diesel_i6_truck", rpm=600.0, dur=4.0), 30.0) > 8


@pytest.mark.parametrize("kind, rpm", [("diesel_i4", 1000.0), ("diesel_i6_truck", 1800.0)])
@pytest.mark.parametrize("seed", [0, 1])
def test_diesel_detections_are_the_combustion(kind, rpm, seed):
    """A real transient, not masked: each detection falls within 1.5 ms of an ignition (measured .25 to 1.12 ms), clatter tau .8 ms at 6.2 kHz."""
    y = e.diesel_engine(sr=e.SR, kind=kind, rpm=rpm, dur=2.0, seed=seed)
    n = len(y)
    idx = e._events(np.full(n, rpm), e.LAYOUTS[kind]["offsets"], n) + int(e.ignition_delay_ms() * 1e-3 * e.SR)
    c = detect_clicks(y)
    j = np.searchsorted(idx, c, side="right") - 1
    assert np.all(j >= 0) and np.all((c - idx[j]) / e.SR < 1.5e-3)


def test_diesel_clatter_sits_in_its_band_and_follows_the_firing():
    y = e.diesel_engine(sr=SR, rpm=1000.0, dur=4.0, load=0.3)
    sp, fr = _spec(y)
    assert _band(sp, fr, 400.0, 2000.0) > _band(sp, fr, 4000.0, 8000.0) + 10      # 400-2000 Hz [DZ]
    a = _spec(e.diesel_engine(sr=SR, rpm=1000.0, dur=4.0, clatter=0.0))
    assert _band(*_spec(y), 1000.0, 2000.0) - _band(*_spec(y), 50.0, 400.0) > _band(*a, 1000.0, 2000.0) - _band(*a, 50.0, 400.0) + 10
    assert _env_line(_hp_band(y), 33.333) > 15


def _hp_band(y):
    return c._bp(y, 1500.0, 0.7)


def test_bigger_bore_lowers_the_clatter_modes():
    def top(b):
        x = np.zeros(SR)
        x[100] = 1.0
        sp, fr = _spec(e._ring(x, e._CLATTER, 0.086 / b))
        return fr[np.argmax(sp)]
    assert top(0.13) == pytest.approx(1700.0 * 0.086 / 0.13, abs=60.0)
    assert top(0.086) == pytest.approx(1700.0, abs=60.0)


def test_ignition_delay_formula():
    assert e.ignition_delay_ms(51.0, 100.0, 1000.0) == pytest.approx(0.44 * 100 ** -1.19 * np.exp(4.65))
    assert e.ignition_delay_ms(40.0) > e.ignition_delay_ms(60.0)


# ---------------------------------------------------------------- structure

def test_knock_frequency():
    assert e.knock_frequency(0.086) == pytest.approx(6474.0, abs=5.0)


def test_engine_mech_knock_rings_the_chamber():
    y = e.engine_mech(0.086, 2500.0, 0.8, 2.0, knock=1.0, valve=0.0, slap=0.0, fire=0.0)
    sp, fr = _spec(y)
    m = fr > 3000
    assert abs(fr[m][np.argmax(sp[m])] - 6474.0) < 200.0
    assert np.abs(y).max() == pytest.approx(0.8, abs=1e-6)


def test_two_valve_events_per_cylinder_per_cycle():
    assert _env_line(e.engine_mech(0.086, 1200.0, 1.0, 4.0, valve=1.0, slap=0.0, fire=0.0), 80.0) > 20      # 2 x 4 x 1200 / 120


def test_rod_knocks_twice_per_turn():
    assert _env_line(e.engine_mech(0.086, 1200.0, 1.0, 4.0, valve=0.0, slap=0.0, fire=0.0, rod=1.0), 40.0) > 20


def test_mech_scales_with_bore():
    def lo(bore):
        sp, fr = _spec(e.engine_mech(bore, 1500.0, 1.0, 2.0, valve=0.0, slap=0.0, rod=1.0, fire=0.0))
        m = fr < 600
        return fr[m][np.argmax(sp[m])]
    assert lo(0.172) < lo(0.086) * 0.65


def test_mech_bore_is_in_metres():
    with pytest.raises(ValueError):
        e.engine_mech(86.0, 1500.0)


# ---------------------------------------------------------------- forced induction

def test_turbo_spools_up_and_holds_its_boost():
    s = e.turbo_sim(3500.0, 1.0, 4.0)
    w = s["w"]
    assert w[int(0.5 * SR)] > 3 * w[int(0.01 * SR)]
    assert s["boost"][-1] > 0.5e5 and s["boost"].max() < 1.3e5      # the wastegate holds 1 bar
    assert s["surge"].max() == 0.0 and s["bov_flow"].max() == 0.0


def test_blow_off_opens_only_on_a_shut_throttle_with_boost():
    n = c._n(4.0)
    s = e.turbo_sim(3500.0, e._throttle_lift(n), 4.0)
    k = int(0.6 * n)
    assert s["bov_flow"][:k - 100].max() == 0.0 and s["bov_flow"][k:].max() > 0.0
    assert s["surge"].max() == 0.0


def test_no_blow_off_means_surge_cycles():
    n = c._n(4.0)
    s = e.turbo_sim(3500.0, e._throttle_lift(n), 4.0, bov=False)
    k = int(0.6 * n)
    assert s["surge"][:k - 100].max() == 0.0
    cycles = np.sum(np.diff((s["mdot"][k:] < 0).astype(int)) == 1)
    assert 5 < cycles < 200


def test_spool_whine_follows_the_shaft():
    y = e.forced_induction(sr=SR, kind="spool", dur=3.0)
    f = np.mean(e.turbo_sim(3500.0, 1.0, 3.0)["w"][-SR // 2:]) / (2 * np.pi)
    sp, fr = _spec(y[-SR // 2:])
    assert _db(sp, fr, f, 0.05 * f) > _band(sp, fr, 300.0, 600.0) - 25


def test_recirculated_valve_is_quieter_by_the_sources_ratio():
    # x 3 against x 14 = 13.4 dB, measured against the same whine before the throttle shuts.
    def bov_over_whine(mode):
        y = e.forced_induction(sr=SR, kind="bov", bov=mode, dur=3.0)
        k = int(0.6 * len(y))
        return 20 * np.log10(np.max(np.abs(y[k:k + SR // 4])) / _rms(y[k - SR // 2:k]))
    assert bov_over_whine("atmo") - bov_over_whine("recirc") > 8


def test_wastegate_leak_pulses_at_the_firing_rate():
    assert _env_line(e.forced_induction(sr=SR, kind="wastegate", rpm=3000.0, cylinders=4, dur=2.0), 100.0) > 15


def test_roots_whine_is_at_six_times_the_rotor():
    sp, fr = _spec(e.forced_induction(sr=SR, kind="roots", rpm=3000.0, dur=2.0))
    m = (fr > 100) & (fr < 2000)
    assert abs(fr[m][np.argmax(sp[m])] - 3000.0 / 60.0 * 2.1 * 6.0) < 2.0
    assert _db(sp, fr, 3000.0 / 60.0 * 2.1 * 31.0) > _db(sp, fr, 3000.0 / 60.0 * 2.1 * 31.5) + 20    # timing gears, 31 teeth


# ---------------------------------------------------------------- speed curves

def test_rpm_curve_torque_law():
    idle = e.rpm_curve(0.0, 3.0, rpm0=2000.0)
    assert idle[0] > idle[-1]
    assert abs(idle[-1] - 750.0) < 120.0                     # idle throttle .35 below 750
    full = e.rpm_curve(1.0, 3.0, rpm0=1000.0)
    assert np.argmax(full) > 0 and full.max() < 7700.0       # fuel cut above 7500
    start = e.rpm_curve(0.0, 0.3, start=True)
    assert start[int(0.05 * SR)] > 9.0 and start[-1] > 50.0  # the starter's +200 rpm/s, then the engine's own torque


def test_rpm_curve_accelerates_faster_than_it_falls():
    up = e.rpm_curve(1.0, 0.5, rpm0=2000.0)
    down = e.rpm_curve(0.0, 0.5, rpm0=up[-1])
    assert up[-1] - up[0] > down[0] - down[-1]


def test_gear_table():
    assert e.gear_rpm(1, 1.0) == pytest.approx(33.37 * 3.4, rel=2e-3)
    assert e.gear_rpm(6, 100.0) == pytest.approx(33.37 * 0.7 * 100.0, rel=2e-3)


def test_slew_is_limited():
    t = np.zeros(SR)
    t[SR // 4:] = 6000.0
    r = e._slew(t)
    assert np.max(np.diff(r)) * SR <= e.MAX_RPM_S * 1.01
    assert np.max(-np.diff(e._slew(6000.0 - t))) * SR <= e._FALL_RPM_S * 1.01


def test_limiter_bounces_between_limit_and_limit_less_120():
    r, g = e.limiter_curve(8000.0, 2.0)
    late = r[SR // 2:]
    assert late.max() < 8002.0 and late.min() > 7878.0
    assert 0.2 < g[SR // 2:].mean() < 0.8
    y = e.engine_event(sr=SR, kind="rev_limiter", limit=8000.0, dur=3.0)
    bounce = 1.0 / (120.0 / e.MAX_RPM_S + 120.0 / e._FALL_RPM_S)
    assert _env_line(y, bounce, 3.0) > 15


# ---------------------------------------------------------------- events

def test_ignition_gate_silences_the_combustion():
    n = c._n(2.0)
    y = c.pipe_engine(sr=SR, dur=2.0, rpm=2000.0, load=0.8, fire=np.where(np.arange(n) < n // 2, 0.0, 1.0))
    # The pistons' pumping remains, so the level hardly moves: the narrow ignition pulse is what goes from the top of the spectrum.
    a, b = _spec(y[2000:n // 2 - 2000]), _spec(y[n // 2 + 2000:-2000])
    assert _band(*b, 800.0, 6000.0) - _band(*b, 30.0, 300.0) > _band(*a, 800.0, 6000.0) - _band(*a, 30.0, 300.0) + 3


def test_afterfire_law():
    n = c._n(3.0)
    rng = np.random.default_rng(1)
    rr = np.full(n, 4000.0)
    pops, t = e.afterfire(rr, np.zeros(n), n, rng)
    assert 10 < len(t) < 400
    gaps = np.diff(t)
    assert gaps.std() / gaps.mean() > 0.5                    # not locked to the cycle
    assert len(e.afterfire(rr, np.ones(n), n, rng)[1]) == 0  # needs a shut throttle
    assert len(e.afterfire(np.full(n, 1200.0), np.zeros(n), n, rng)[1]) == 0   # and rpm over idle + 700
    assert np.isfinite(pops).all()


def test_starter_tones():
    n = SR
    rng = np.random.default_rng(0)
    y = e._starter(np.full(n, 250.0), np.ones(n), rng)
    sp, fr = _spec(y[SR // 4:])
    f_motor = 250.0 / 60.0 * 2 * np.pi * 15.0 * 12.0 / (2 * np.pi)
    f_mesh = 250.0 / 60.0 * 138.0
    assert _db(sp, fr, f_motor) > _db(sp, fr, f_motor * 1.1) + 15
    assert _db(sp, fr, f_mesh) > _db(sp, fr, f_mesh * 1.1) + 15


def test_start_goes_from_cranking_to_a_louder_idle():
    y = e.engine_event(sr=SR, kind="engine_start", layout="inline_4", seed=1)
    assert _rms(y[int(2.5 * SR):]) > 1.5 * _rms(y[:int(0.4 * SR)])
    cr = _spec(y[:int(0.4 * SR)])
    assert _band(*cr, 300.0, 3000.0) > _band(*cr, 8000.0, 20000.0)


def test_failed_start_has_three_sagging_attempts():
    y = e.engine_event(sr=SR, kind="engine_start_fail", dur=4.5)
    each = (4.5 - 0.7) / 3.0
    parts = [_rms(y[int(i * (each + 0.35) * SR):int((i * (each + 0.35) + each) * SR)]) for i in range(3)]
    gaps = [_rms(y[int((i * (each + 0.35) + each + 0.01) * SR):int(((i + 1) * (each + 0.35) - 0.01) * SR)]) for i in range(2)]
    assert parts[0] > parts[1] > parts[2] and max(gaps) < 0.01 * parts[0]


def test_engine_off_runs_down_in_two_seconds():
    y = e.engine_event(sr=SR, kind="engine_off", dur=2.5)
    assert _rms(y[int(2.05 * SR):]) < 1e-3
    assert _rms(y[:SR // 2]) > 2 * _rms(y[int(1.4 * SR):int(1.9 * SR)])


def test_stall_ends_on_a_shake_and_then_nothing():
    y = e.engine_event(sr=SR, kind="engine_stall", dur=2.5)
    assert _rms(y[-SR // 20:]) < 0.5 * _rms(y[:SR])


def test_free_rev_follows_the_script():
    y = e.engine_event(sr=SR, kind="free_rev", dur=9.0)
    assert len(y) == 9 * SR
    # Open from 5 to 8.2 s, closed before 2 s: pitch and level climb.
    a, b = y[int(0.5 * SR):int(1.5 * SR)], y[int(6.5 * SR):int(8.0 * SR)]
    sa, sb = _spec(a), _spec(b)
    cent = lambda s: float(np.sum(s[1] * s[0] ** 2) / np.sum(s[0] ** 2))
    assert cent(sb) > cent(sa)


@pytest.mark.parametrize("kind", ["downshift", "upshift"])
def test_shifts_have_a_clutch_and_a_change_of_note(kind):
    y = e.engine_event(sr=SR, kind=kind, gear=3, speed=60.0)
    assert np.isfinite(y).all() and len(y) == int(1.5 * SR)
    _ends_clean(y)
    a, b = y[:int(0.35 * SR)], y[int(0.9 * SR):]
    def fire(x):
        sp, fr = _spec(x)
        m = (fr > 40) & (fr < 300)
        return float(np.sum(fr[m] * sp[m] ** 2) / np.sum(sp[m] ** 2))
    assert (fire(b) > fire(a)) == (kind == "downshift")


def test_engine_bank_is_a_whole_number_of_cycles_and_seamless():
    y = e.engine_event(sr=SR, kind="engine_bank", rpm=3000.0, cycles=40)
    assert len(y) == 40 * 120 / 3000 * SR
    jump = abs(y[0] - y[-1])
    assert jump < 3.0 * np.percentile(np.abs(np.diff(y)), 99.9)
    y44 = e.engine_event(sr=44100, kind="engine_bank", rpm=3000.0, cycles=40)
    assert len(y44) == round(40 * 120 / 3000 * 44100)


def test_default_bank_length_is_at_least_3_2_s():
    y = e.engine_event(sr=SR, kind="engine_bank", rpm=2000.0)
    assert 3.2 <= len(y) / SR < 3.2 + 2 * 120.0 / 2000.0


@pytest.mark.parametrize("kind", e._EVENTS)
def test_every_event_renders(kind):
    y = e.engine_event(sr=22050, kind=kind, dur=1.5 if kind in ("free_rev", "engine_bank") else 0.0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1


@pytest.mark.parametrize("kind", ["engine_start", "engine_stall", "rev_limiter", "dfco_overrun", "downshift", "upshift", "valve_tick"])
def test_flowing_events_have_no_click(kind):
    assert len(detect_clicks(e.engine_event(sr=SR, kind=kind))) == 0


# Attacks of their own (detect_clicks is meant for smooth voices): knock, ticks, pops, the slow pulses of a run-down.
@pytest.mark.parametrize("kind", ["exhaust_crackle", "engine_off", "engine_knock", "valve_tick", "rod_knock", "engine_start_fail", "free_rev"])
def test_percussive_events_have_clean_ends(kind):
    y = e.engine_event(sr=SR, kind=kind, dur=2.0)
    _ends_clean(y)


def test_engine_knock_is_at_the_bores_frequency():
    sp, fr = _spec(e.engine_event(sr=SR, kind="engine_knock", bore=0.086, severity=1.0, dur=2.0))
    assert _band(sp, fr, 6000.0, 7000.0) > _band(sp, fr, 4000.0, 5000.0)


@pytest.mark.parametrize("fn, kw", [(e.diesel_engine, dict(kind="diesel_i4")), (e.diesel_engine, dict(kind="diesel_i6_truck")), (e.two_stroke, {}),
                                    (e.race_car, {}), (e.jalopy, {}), (e.forced_induction, dict(kind="spool")), (e.forced_induction, dict(kind="bov")),
                                    (e.forced_induction, dict(kind="flutter")), (e.forced_induction, dict(kind="wastegate")), (e.forced_induction, dict(kind="roots"))])
def test_the_rest_renders_clean(fn, kw):
    y = fn(sr=SR, dur=1.5, **kw)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1
    _ends_clean(y)


def test_flutter_pulses_after_the_lift():
    y = e.forced_induction(sr=SR, kind="flutter", dur=3.0)
    k = int(0.6 * len(y))
    def flicker(x):                                          # variation of the 5 ms level: the wheel stalls and recovers
        m = len(x) // 240
        r = np.sqrt(np.mean(c._hp(x, 3000.0)[:m * 240].reshape(m, 240) ** 2, axis=1))
        return r.std() / r.mean()
    assert flicker(y[k + SR // 20:k + SR // 2]) > 1.5 * flicker(y[k - SR // 2:k])


def test_race_car_pulls_through_its_gear_to_the_top():
    y = e.race_car(sr=SR, rpm=18000.0, gear=1, dur=3.0)
    sp, fr = _spec(y[-SR:])                                  # at the top: 10 x 18000 / 120 = 1500 Hz
    m = (fr > 500) & (fr < 3000)
    top = fr[m][np.argmax(sp[m])]                            # the banks' half-order is as strong as the firing line
    assert min(abs(top - 1500.0), abs(top - 750.0)) < 40.0
    assert _db(sp, fr, 1500.0, 30.0) > _db(sp, fr, 1125.0, 30.0) + 6


def test_jalopy_misfires():
    y = e.jalopy(sr=SR, rpm=900.0, rough=1.0, dur=4.0)
    z = e.jalopy(sr=SR, rpm=900.0, rough=0.0, backfire=0.0, dur=4.0)
    assert _comb(z, 30.0) > _comb(y, 30.0)


def test_rpm_curves_pass_through_the_presets():
    r = np.linspace(1000.0, 3000.0, 100)
    y = e.engine_layout(sr=SR, kind="v8_crossplane", rpm=r, dur=2.0)
    a, b = _spec(y[:SR // 4]), _spec(y[-SR // 4:])
    cent = lambda s: float(np.sum(s[1][s[1] < 1000] * s[0][s[1] < 1000] ** 2) / np.sum(s[0][s[1] < 1000] ** 2))
    assert cent(b) > cent(a)
    assert np.isfinite(e.two_stroke(sr=SR, rpm=r, dur=1.0)).all()
    assert np.isfinite(e.diesel_engine(sr=SR, rpm=r, dur=1.0)).all()


@pytest.mark.parametrize("name, params", [("engine_layout", dict(kind="v10")), ("diesel_engine", dict(kind="diesel_i4", rpm=1500)), ("two_stroke", dict(rpm=6000)),
                                          ("forced_induction", dict(kind="roots")), ("engine_event", dict(kind="engine_off")), ("race_car", {}), ("jalopy", {})])
def test_registered_like_any_sfx(name, params):
    assert name in REGISTRY
    y = REGISTRY[name]["fn"](sr=22050, **params)
    assert np.isfinite(y).all()
