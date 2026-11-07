"""Sport and glide: every variant renders clean, and the sourced numbers are measured on the signal. Nothing has been listened to."""
import math
import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

from genny import sports as S
from genny.analysis import detect_clicks
from genny.physical import footsteps as FS
from genny.sfx import REGISTRY

SR = 48000
RMS = lambda y: float(np.sqrt(np.mean(y * y)))
VARIANTS = ([(S.ball_bounce, dict(kind=k)) for k in S.BALLS] + [(S.snow, dict(kind=k)) for k in S.SNOW_KINDS] + [(S.sport, dict(kind=k)) for k in S.SPORT_KINDS]
            + [(S.skateboard, {}), (S.skateboard, dict(ollie=1)), (S.ice_skate, {}), (S.ice_skate, dict(stroke="stop")), (S.pea_whistle, {})] + [(S.swim, dict(stroke=k)) for k in S.STROKES])


def _lp(y, f):
    return sosfiltfilt(butter(4, f, "lowpass", fs=SR, output="sos"), y)


def _bp(y, a, b):
    return sosfiltfilt(butter(4, [a, b], "bandpass", fs=SR, output="sos"), y)


def _peak(y, lo, hi):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def _share(y, lo, hi):
    s = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(s[(f >= lo) & (f <= hi)].sum() / s.sum())


def _centroid(y):
    s = np.abs(np.fft.rfft(y)) ** 2
    return float(np.sum(np.fft.rfftfreq(len(y), 1 / SR) * s) / np.sum(s))


def _env(y, ms=5.0):
    k = int(ms / 1000 * SR)
    return np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))


def _period(y, lo, hi, ms=5.0):
    """Lag in s of the strongest autocorrelation of the envelope between lo and hi s."""
    e = _env(y, ms)
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    a, b = int(lo * SR), int(hi * SR)
    return (a + int(np.argmax(ac[a:b]))) / SR


@pytest.mark.parametrize("f,kw", VARIANTS, ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_variant_renders_without_a_click(f, kw, seed):
    y = f(sr=SR, seed=seed, **kw)
    assert np.isfinite(y).all() and 0.1 < np.abs(y).max() <= 1.0
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
    assert len(detect_clicks(y)) == 0


@pytest.mark.parametrize("f,kw", [(S.ball_bounce, dict(kind="basket", surface="concrete")), (S.ball_bounce, dict(kind="foot", surface="turf", roll_speed=2.0)), (S.ball_bounce, dict(kind="tennis", surface="clay")),
                                  (S.sport, dict(kind="batte", bat="aluminium")), (S.sport, dict(kind="billard", speed=4.0)), (S.pea_whistle, dict(rate=0.0, blasts=3)), (S.ice_skate, dict(crack=0.6)),
                                  (S.skateboard, dict(surface="asphalt")), (S.skateboard, dict(surface="wood", speed=8.0)), (S.snow, dict(kind="ski", temperature=-20.0)), (S.snow, dict(kind="luge", temperature=0.0))])
def test_options_stay_clean(f, kw):
    y = f(sr=SR, **kw)
    assert np.isfinite(y).all() and len(detect_clicks(y)) == 0


@pytest.mark.parametrize("f,kw", [(S.snow, dict(kind="boule")), (S.skateboard, {}), (S.skateboard, dict(ollie=1)), (S.ice_skate, {}), (S.ice_skate, dict(stroke="stop"))], ids=["boule", "skate", "ollie", "glide", "stop"])
def test_variants_that_clicked_on_other_seeds_stay_clean(f, kw):
    """Seeds where these fired: a lone pulse under the squeak, the corner of a linear fade, of a boxcar gate."""
    for seed in (4, 10, 11, 12):
        assert len(detect_clicks(f(sr=SR, seed=seed, **kw))) == 0, seed


def test_names_registered_and_bad_kinds_refused():
    assert {"ball_bounce", "snow", "ice_skate", "pea_whistle", "skateboard", "swim", "sport"} <= set(REGISTRY)
    for f, kw in ((S.ball_bounce, dict(kind="rugby")), (S.ball_bounce, dict(surface="ice")), (S.snow, dict(kind="yeti")), (S.sport, dict(kind="curling")), (S.sport, dict(kind="batte", bat="glass")),
                  (S.swim, dict(stroke="dog")), (S.ice_skate, dict(stroke="axel")), (S.skateboard, dict(surface="sand"))):
        with pytest.raises(ValueError):
            f(sr=SR, **kw)


# ball
def test_katz_numbers_from_his_laws():
    b = S.BALLS["basket"]
    w = S.ball_omega(b["M"], b["a"], b["P"])
    assert abs(w - 256) < 1.5 and abs(2 * w / (2 * np.pi) - 82) < 1 and abs(np.pi / w - 0.0123) < 0.0003      # "12 ms"
    assert abs(S.cavity_modes(b["a"])[0] - 997) < 1
    assert abs(S.thump_pa(b["a"], 4.43) - 0.89) < 0.01                      # 8.9 dyn/cm^2 at 3 m
    assert abs(S.ring_pa(b["M"], b["a"], b["P"], 4.43) - 0.177) < 0.015     # 1.87 dyn/cm^2 printed; 14 dB under


def test_russell_radius_gives_his_modes():
    f1, f2 = S.cavity_modes(0.116)
    assert abs(f1 - 970) / 970 < 0.015 and abs(f2 - 1580) / 1580 < 0.015     # his own theory: 979, 1572


def test_rendered_thump_is_one_cycle_over_12_ms_at_82_hz():
    raw, _ = S._ball_train("basket", 1.0, "turf", 0.0, 1.0, 0.0, 0)
    raw = raw[:int(0.06 * SR)]
    on = np.flatnonzero(raw)
    assert abs((on[-1] - on[0] + 1) / SR - 0.0123) < 0.0003                  # the pulse, before _fin's 12 Hz high-pass adds its slow tail
    y = S.ball_bounce(sr=SR, kind="basket", cavity=0.0, surface="turf", height=1.0)[:int(0.06 * SR)]
    a = on[0]
    z = np.flatnonzero(np.diff(np.sign(y[a + 8:a + int(0.0123 * SR) - 8])))
    half = (z[-1] - z[0]) / SR                                               # -cos crosses zero at T/4, 3T/4
    assert abs(1 / (2 * half) - 82) < 6


def test_rendered_ring_at_997_and_14_db_under_the_thump(monkeypatch):
    monkeypatch.setitem(S.BALL_MODES, "basket", ())                          # the ring alone, without the shell modes at 410-900 Hz
    y, t = S._ball_train("basket", 1.0, "turf", 1.0, 1.0, 0.0, 0)
    z, _ = S._ball_train("basket", 1.0, "turf", 0.0, 1.0, 0.0, 0)
    w = (y - z)[:int(0.2 * SR)]
    assert abs(20 * np.log10(np.abs(z[:int(0.2 * SR)]).max() / np.abs(w).max()) - 14) < 1.0
    assert abs(_peak(w, 700, 1300) - 997) < 25
    monkeypatch.undo()
    r = S.ball_bounce(sr=SR, kind="basket", surface="turf", ring_cycles=30.0)[:int(0.25 * SR)]      # held long enough for the line to stand
    assert abs(_peak(r, 950, 1100) - 997) < 6 and abs(_peak(r, 1450, 1750) - S.cavity_modes(0.114)[1]) < 12


def test_ring_rises_against_the_thump_on_small_bounces():
    """Thump as v^2, ring as v."""
    b = S.BALLS["basket"]
    q = lambda v: S.ring_pa(b["M"], b["a"], b["P"], v) / S.thump_pa(b["a"], v)
    assert abs(q(1.0) / q(4.0) - 4.0) < 1e-6
    def rendered(h):
        y = S.ball_bounce(sr=SR, kind="basket", height=h, surface="turf", cavity=1.0)[:int(0.025 * SR)]
        return np.abs(_bp(y, 1400, 2000)).max() / np.abs(_lp(y, 200)).max()
    assert rendered(0.1) > 2 * rendered(1.0)                                 # sqrt(10) = 3.16 by the law


def test_cavity_follows_the_size_of_the_ball():
    for k in ("tennis", "foot", "volley"):
        a = S.BALLS[k]["a"]
        y = S.ball_bounce(sr=SR, kind=k, surface="turf", ring_cycles=30.0)[:int(0.25 * SR)]
        f = S.cavity_modes(a)[0]
        assert abs(_peak(y, 0.9 * f, 1.1 * f) - f) < 0.012 * f, k


def test_volley_breathes_at_150_hz():
    y = S.ball_bounce(sr=SR, kind="volley", surface="turf", height=1.0)[:int(0.25 * SR)]
    assert abs(_peak(y, 60, 400) - 150) < 15
    z = S.ball_bounce(sr=SR, kind="volley", surface="turf", cavity=0.0)[:int(0.25 * SR)]
    assert abs(_peak(z, 60, 400) - 150) > 25


def test_bounce_train_is_geometric_with_the_restitution():
    y = S.ball_bounce(sr=SR, kind="basket", surface="concrete", height=1.0)
    e = S.BALLS["basket"]["e"]
    _, t = S._ball_train("basket", 1.0, "concrete", 1.0, 1.0, 0.0, 0)
    assert abs((t[2] - t[1]) / (t[1] - t[0]) - e) < 1e-6
    env = _env(_lp(y, 400))
    pk = [env[int(a * SR):int(a * SR) + int(0.03 * SR)].max() for a in t[:3]]
    assert 1.2 < pk[0] / pk[1] < 1 / e ** 2 * 1.1            # v^2 per hit
    assert abs(np.argmax(env[int(t[1] * SR) - 240:int(t[1] * SR) + 960]) - 240) < 700


def test_floors_are_heard_and_a_soft_one_is_dead():
    p = S.ball_bounce(sr=SR, surface="parquet")
    t = S.ball_bounce(sr=SR, surface="turf")
    w = int(0.05 * SR)                                                          # first contact only: the same thump on both
    rp, rt = S._ball_train("basket", 1.0, "parquet", 1.0, 1.0, 0.0, 0)[0][:w], S._ball_train("basket", 1.0, "turf", 1.0, 1.0, 0.0, 0)[0][:w]
    floor, A = rp - rt, S.thump_pa(0.114, np.sqrt(2 * 9.81))
    assert abs(np.abs(floor).max() / A - S.SURFACES["parquet"]["floor"]) < 0.02  # the plate is there at its level (only its own constant)
    assert _centroid(floor) < 400                                               # a 12 ms contact has nothing above about 1/T to give a plate: a low thud, no ring
    assert len(t) < len(p)                                                    # a dead floor stops the ball sooner


# rolling
def test_rolling_resistance_laws():
    assert abs(S.plastic_mu(0.73) - 0.076) < 0.001 and abs(S.plastic_mu(1.05) - 0.110) < 0.002
    assert abs(S.viscoelastic_mu(0.01, 0.1) - 0.0195) < 1e-9
    assert abs(S.strain_freq(2 * np.pi, 0.01) - 183) < 1e-6
    assert abs(S.rolling_time(5.0 / 7.0 * 0.1 * 9.81, 0.1) - 1) < 1e-9


def test_soft_floor_stops_a_rolling_ball_sooner():
    hard = S.ball_bounce(sr=SR, kind="foot", surface="parquet", roll_speed=0.5)
    soft = S.ball_bounce(sr=SR, kind="foot", surface="turf", roll_speed=0.5)
    b = S.BALLS["foot"]
    mu = S.viscoelastic_mu(S.ball_patch(b["M"], b["a"], b["P"]), b["a"])
    assert S.plastic_mu(0.73) > 2 * mu
    assert len(soft) < len(hard)


def test_hertz_time_scale():
    a = S.hertz_time(0.17, 0.17, 0.0286, 0.0286, S.E_STAR_RESIN, 2.0)
    assert 2e-4 < a < 8e-4
    assert abs(S.hertz_time(0.17, 0.17, 0.0286, 0.0286, S.E_STAR_RESIN, 2.0 * 32) / a - 0.5) < 1e-9     # V^(-1/5)


# pea whistle
def test_pea_whistle_carrier_and_am_at_its_rate():
    y = S.pea_whistle(sr=SR, rate=20.0, dur=2.0)
    assert 3050 <= _peak(y, 2000, 6000) <= 3500
    body = y[int(0.1 * SR):int(1.9 * SR)]
    e = np.abs(hilbert(_bp(body, 2900, 3800)))
    e = e - e.mean()
    s = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fr = np.fft.rfftfreq(len(e), 1 / SR)
    m = (fr > 5) & (fr < 80)
    assert abs(fr[m][np.argmax(s[m])] - 20) < 2


def test_pea_whistle_flips_between_two_frequencies_inside_the_band():
    y = S.pea_whistle(sr=SR, rate=20.0, dur=2.0)[int(0.1 * SR):int(1.9 * SR)]
    z = hilbert(_bp(y, 2900, 3700))
    f = np.diff(np.unwrap(np.angle(z))) * SR / (2 * np.pi)
    f = np.convolve(f, np.ones(48) / 48, mode="valid")
    lo, hi = np.quantile(f, 0.1), np.quantile(f, 0.9)
    assert 3050 <= lo < hi <= 3450 and hi - lo > 50
    hist = np.histogram(f, bins=[lo - 5, lo + 25, (lo + hi) / 2 - 15, (lo + hi) / 2 + 15, hi - 25, hi + 5])[0]
    assert hist[0] > 2 * hist[2] and hist[4] > 2 * hist[2]     # two states, not a sine sweep


def test_pea_whistle_has_harmonics_2_3_and_no_pea_no_am():
    y = S.pea_whistle(sr=SR, rate=20.0, dur=1.0)
    for h in (2, 3):
        assert RMS(_bp(y, 3400 * h * 0.97, 3400 * h * 1.03)) > 0.03 * RMS(y)
    def depth(rate):
        e = np.abs(hilbert(_bp(S.pea_whistle(sr=SR, rate=rate, dur=1.0)[int(0.1 * SR):int(0.9 * SR)], 3000, 3800)))
        e = _lp(e, 200.0)
        return np.std(e) / np.mean(e)
    assert depth(20.0) > 2 * depth(0.0)


# snow: the squeak spectrum, through the squeak
def test_squeak_follows_arabadzhi():
    rng = np.random.default_rng(0)
    n = 4 * SR
    fn = 0.5 + 0.45 * np.sin(2 * np.pi * 60 * np.arange(n) / SR)
    def tilt(T):
        y = S._squeak(fn, T, rng)
        return 20 * np.log10(RMS(_bp(y, 3560, 4490)) / RMS(_bp(y, 445, 561)))
    for T, want in ((-1.5, -13.5), (-15.0, -0.7)):               # table: 4 kHz minus 500 Hz
        i = list(FS.ARABADZHI_BANDS).index(4000.0), list(FS.ARABADZHI_BANDS).index(500.0)
        assert abs(FS.ARABADZHI_SQUEAK[T][i[0]] - FS.ARABADZHI_SQUEAK[T][i[1]] - want) < 0.05
        assert abs(tilt(T) - want) < 3.0, T


@pytest.mark.parametrize("kind", S.SNOW_KINDS)
def test_snow_is_brighter_cold_and_squeakless_at_zero(kind):
    hf = lambda y: _share(y, 3000, 9000)
    cold, warm = [S.snow(sr=SR, kind=kind, temperature=T, seed=1) for T in (-15.0, -1.5)]
    assert hf(cold) > hf(warm)
    assert np.all(S._squeak(np.ones(1000), 0.0, np.random.default_rng(0)) == 0)


def test_snow_kinds_have_their_gesture():
    n = S.snow(sr=SR, kind="ski", dur=5.6)
    assert abs(_period(n, 0.7, 2.0, 30.0) - S.TURN_S) < 0.1                 # ski: turn after turn
    b = S.snow(sr=SR, kind="boule")
    assert abs(_period(b, 0.3, 0.7, 20.0) - 0.5) < 0.05                     # three squeezes
    c = S.snow(sr=SR, kind="chute")
    assert _centroid(c[int(0.03 * SR):int(0.08 * SR)]) < _centroid(c[int(0.2 * SR):int(0.6 * SR)]) and np.argmax(_env(c, 10.0)) < 0.25 * SR
    l = S.snow(sr=SR, kind="luge", dur=4.0)
    assert _share(l, 40, 250) > 0.05 and _share(l, 900, 5000) > 0.1          # runners over rumble


# ice, skate, swim: timbre by knowledge, gestures measured
def test_skate_stroke_cadence_and_stop_is_brighter():
    y = S.ice_skate(sr=SR, rate=1.2, dur=5.0)
    assert abs(_period(y, 0.5, 1.2, 30.0) - 1 / 1.2) < 0.04
    assert _centroid(S.ice_skate(sr=SR, stroke="stop")) > _centroid(y)
    assert S.ice_skate(sr=SR, stroke="stop", crack=0.8).size > S.ice_skate(sr=SR, stroke="stop").size


def test_skateboard_joints_at_L_over_v_then_wheelbase():
    V, L = 4.0, 2.0
    y = S.skateboard(sr=SR, speed=V, joints=L, dur=6.0)
    assert abs(_period(y, 0.3, 0.75) - L / V) < 0.01
    assert abs(_period(y, 0.05, 0.13) - S.WHEELBASE / V) < 0.006
    z = S.skateboard(sr=SR, speed=V, joints=0.0, dur=6.0)
    assert np.max(_env(y)) / np.median(_env(y)) > 2 * np.max(_env(z)) / np.median(_env(z))


def test_ollie_goes_quiet_in_the_air_for_its_ballistic_time():
    d = 4.0
    y = S.skateboard(sr=SR, ollie=1, joints=0.0, dur=d)
    z = S.skateboard(sr=SR, joints=0.0, dur=d)
    t0, air = 0.45 * d, S.ollie_air()
    assert abs(air - 0.495) < 0.01
    mid = (int((t0 + 0.32 * air) * SR), int((t0 + 0.92 * air) * SR))
    ok = (int(0.5 * SR), int((t0 - 0.1) * SR))
    assert RMS(y[mid[0]:mid[1]]) / RMS(y[ok[0]:ok[1]]) < 0.5 * RMS(z[mid[0]:mid[1]]) / RMS(z[ok[0]:ok[1]])
    e = _env(y, 3.0)
    land = int((t0 + air) * SR)
    assert e[land:land + int(0.02 * SR)].max() > 3 * e[mid[0]:mid[1]].max() * 0.5


def test_swim_cadence_and_bubbles():
    for stroke, (cycle, arms, _) in S.STROKES.items():
        y = S.swim(sr=SR, stroke=stroke, dur=8.0)
        T = cycle / arms
        assert abs(_period(y, 0.6 * T, 1.4 * T, 30.0) - T) < 0.08, stroke
        assert _share(y, 200, 6000) > 0.9
    kicks = lambda s: _share(S.swim(sr=SR, stroke=s, dur=6.0), 2000, 6000)
    assert kicks("crawl") > kicks("breast")                                     # six kicks per arm cycle against none


# sport
def test_bowling_rolls_down_the_lane_then_the_pins():
    y = S.sport(sr=SR, kind="bowling", speed=8.0)
    t, end = S.bowling_roll(8.0)
    assert abs(t - 18.3 / ((8.0 + end) / 2)) < 0.01 and end < 8.0
    e = _env(_bp(y, 200, 4000), 3.0)
    roll = np.median(e[int(0.5 * SR):int((t - 0.3) * SR)])
    assert e[int(t * SR) - 240:int(t * SR) + int(0.03 * SR)].max() > 3 * roll
    lane = e[int((t + 0.1) * SR):int((t + 0.5) * SR)]
    assert np.sum((lane[1:] > 0.4 * lane.max()) & (lane[:-1] <= 0.4 * lane.max())) >= 3     # a clatter, not one blow
    assert len(y) == pytest.approx((t + 1.2) * SR, abs=2)


def test_bowling_takes_the_lane_time_of_its_speed():
    a, b = S.sport(sr=SR, kind="bowling", speed=5.0), S.sport(sr=SR, kind="bowling", speed=10.0)
    assert len(a) / SR - 1.2 > 1.8 * (len(b) / SR - 1.2)                        # 1.2 s: the pins, the same at any speed


def test_billard_click_is_brighter_at_higher_speed_by_hertz():
    def click(V):
        y = S.sport(sr=SR, kind="billard", speed=V)
        t = 0.02 + 1.0 / V
        return y[int(t * SR):int(t * SR) + int(0.02 * SR)]
    assert _centroid(click(4.0)) > _centroid(click(0.8))


def test_struck_sports_have_a_swing_then_a_blow():
    for kind in ("tennis", "golf", "batte"):
        y = S.sport(sr=SR, kind=kind)
        e = _env(y)
        k = int(np.argmax(e))
        assert k > 0.05 * SR and RMS(y[:k - 240]) > 0.005 * np.abs(y).max(), kind


def test_low_pass_on_the_bat_swing_takes_nothing_it_has():
    y = S.FO.sword(sr=SR, blade="bat", speed=0.83, seed=0)
    assert _share(y, 2000, 20000) < 0.01
    assert _share(S.FO.sword(sr=SR, blade="bat", speed=0.6, seed=0), 1000, 20000) < 0.01     # the racket's


def test_golf_uses_a_short_contact_and_bat_rings_longer_on_aluminium():
    g = S.sport(sr=SR, kind="golf")
    wood = S.sport(sr=SR, kind="batte", bat="wood")
    alu = S.sport(sr=SR, kind="batte", bat="aluminium")
    def decay(y):
        k = int(np.argmax(_env(y)))
        e = _env(y[k:], 5.0)
        return int(np.argmax(e < 0.1 * e.max()))
    assert decay(alu) > decay(wood)
    around = lambda y: y[max(int(np.argmax(_env(y))) - 240, 0):int(np.argmax(_env(y))) + 2400]
    assert _centroid(around(g)) > 2 * _centroid(around(wood))                   # .45 ms on steel against 1 ms on wood


def test_swim_kicks_per_arm_cycle():
    assert S.swim_rates("crawl") == pytest.approx((2 / 1.2, 6 / 1.2))           # six-beat kick: 5 a second, not 6
    assert S.swim_rates("butterfly")[1] == pytest.approx(2 / 1.4)
    assert S.swim_rates("breast")[1] == 0.0
    assert S.swim_rates("back", 2.0)[1] == pytest.approx(2 * 6 / 1.3)


def test_bowling_patch_is_hertz():
    from genny.contact import MATERIALS
    w = MATERIALS["wood"]
    e = 1 / ((1 - S.NU_RESIN ** 2) / S.E_RESIN + (1 - w.poisson ** 2) / w.young)
    a = (3 * 6.8 * 9.81 * 0.108 / (4 * e)) ** (1 / 3)
    assert S.BOWL["patch"] == pytest.approx(a) and 1.1e-3 < a < 1.4e-3
    t, end = S.bowling_roll(8.0)
    assert 8.0 - end == pytest.approx(8.0 - math.sqrt(64.0 - 2 * 5 / 7 * 0.195 * a / 0.108 * 9.81 * 18.3), rel=1e-6)
