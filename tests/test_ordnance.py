"""Heavy weapons: the claims of the spec, measured on the rendered signal, and every variant renders."""
import numpy as np
import pytest

from genny import analysis
from genny import ordnance as O
from genny.physics import blast_wave

SR = 48000
RENDERS = [
    *[(O.grenade, {"kind": k, "fuse": 0.2}) for k in ("frag", "throw_bounce", "stun", "smoke", "gl_pop")],
    (O.grenade, {"kind": "stun", "fuse": 0.2, "tinnitus": 1.0}), (O.grenade, {"kind": "throw_bounce", "fuse": 0.2, "detonate": True}),
    (O.molotov, {}), *[(O.mine, {"kind": k}) for k in ("at", "claymore", "bounding", "arm")],
    *[(O.launcher, {"kind": k}) for k in O._LAUNCH], (O.launcher, {"kind": "rpg", "target": 100.0}), (O.launcher, {"kind": "law", "loaded": False}),
    *[(O.missile, {"kind": k}) for k in ("javelin", "stinger")], (O.missile, {"kind": "javelin", "target": 200.0}), (O.missile, {"kind": "stinger", "distance": 100.0}),
    (O.rocket_salvo, {"count": 3}), (O.rocket_flyby, {}), *[(O.launcher_handling, {"action": a}) for a in ("load", "drop", "latch")],
    *[(O.mortar, {"kind": k}) for k in ("drop", "fire", "impact")], (O.howitzer, {}), (O.tank_gun, {}), (O.shell_incoming, {"dur": 0.5}),
    (O.shell_crack_overhead, {}), (O.aerial_bomb, {"charge": 5.0}), *[(O.flamethrower, {"kind": k, "dur": 0.5}) for k in ("burst", "ignite", "empty", "tail")],
    *[(O.torpedo, {"kind": k, "dur": 0.8}) for k in ("launch", "steam", "piston", "supercav", "hit")], (O.depth_charge, {}), (O.depth_charge, {"depth": 10.0, "sink": True}), (O.sea_mine, {}),
    *[(O.blast_sfx, {"kind": k}) for k in O.BLASTS],
]


def spectrum(y, lo=0.0, hi=SR / 2.0):
    s = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    keep = (f >= lo) & (f <= hi)
    return f[keep], s[keep]


def peak_hz(y, lo=0.0, hi=SR / 2.0):
    f, s = spectrum(y, lo, hi)
    return f[np.argmax(s)]


def env(y, ms=2.0):
    k = max(1, int(ms * 1e-3 * SR))
    return np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))


def onset(y, frac=0.2, start=0.0):
    i0 = int(start * SR)
    e = env(y)[i0:]
    return (i0 + int(np.argmax(e > frac * e.max()))) / SR


def centroid(y):
    f, s = spectrum(y)
    return float(np.sum(f * s * s) / np.sum(s * s))


@pytest.mark.parametrize("fn,kw", RENDERS, ids=lambda v: getattr(v, "__name__", str(v)))
def test_every_variant_renders_without_end_click(fn, kw):
    y = fn(sr=22050, seed=1, **kw)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1 and len(y) > 400
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
    # shocks are edges by design (detect_clicks's own docstring): the tail must be free of them
    tail = y[-int(0.05 * 22050):]
    assert len(analysis.detect_clicks(tail, window=min(441, len(tail) - 2))) == 0


def test_blast_is_not_silent_far_away():
    # physics.Explosion rendered nothing past dur * 343 m: this starts when the wave arrives
    assert np.abs(O.blast(1.0, 2000.0, np.random.default_rng(0))).max() > 0.0
    y = O.blast_sfx(sr=SR, charge=1.0, distance=600.0, ground=False)
    assert np.abs(y).max() > 0.5


def test_blast_keeps_its_level_in_pascals():
    rng = np.random.default_rng(0)
    assert np.abs(O.blast(0.239, 5.0, rng, fireball=0.0)).max() > 3.0 * np.abs(O.blast(0.239, 15.0, rng, fireball=0.0)).max()
    assert abs(blast_wave(0.239, 5.0)[0] - 10854) < 200       # spec table section 4


@pytest.mark.parametrize("w,r,td_ms,ps", [(0.0045, 1.5, 0.45, 9114), (0.239, 15.0, 2.63, 2323), (13.3, 50.0, 9.32, 2824), (0.93, 50.0, 5.97, 835), (93.5, 100.0, 18.28, 2652)])
def test_rendered_shock_has_the_table_duration_and_peak(w, r, td_ms, ps):
    y = O.blast(w, r, np.random.default_rng(0), height=30.0, fireball=0.0)
    zero = int(np.argmax(y < 0))
    assert abs(zero / SR * 1e3 - td_ms) < max(0.1 * td_ms, 2e3 / SR)
    assert abs(np.max(y) - ps) / ps < 0.1


def test_tail_scales_as_cube_root_of_charge():
    assert abs(O.rumble_time(8.0) / O.rumble_time(1.0) - 2.0) < 1e-9
    assert O.rumble_time(93.5) < 2.0                          # Explosion gave a 19 s decay for a Mk 82
    def tail(w):
        y = O._fireball(O.rumble_time(w), np.random.default_rng(0))
        return len(y) / SR
    assert abs(tail(8.0) - 0.1 - 2.0 * (tail(1.0) - 0.1)) < 1e-3


def test_fireball_sits_at_40_to_140_hz():
    y = O._fireball(2.0, np.random.default_rng(1))
    assert 30 < peak_hz(y, 10, 1000) < 150


def test_cos_noise_is_30_ms_and_near_6_khz_when_bright():
    y = O.cos_noise_sweep(1.0, np.random.default_rng(2))
    assert abs(len(y) / SR - 0.03) < 1e-4 and 3000 < centroid(y) < 9000


def test_ground_wave_leads_at_5c_and_splits_as_published():
    ev = O.ground_wave(10.0, 1000.0, np.random.default_rng(0))
    assert abs(ev[0][0] - 1000.0 * (1 / 343 - 1 / (5 * 343))) < 1e-9 and abs(ev[0][0] - 2.33) < 0.02   # 2.35 s at c = 340 in the book
    assert ev[0][0] > ev[1][0] > ev[2][0] > 0
    a = [np.abs(s).max() for _, s in ev]
    assert abs((a[1] / a[0]) ** 2 - 26.0 / 7.0) < 1e-6        # 7/26/67 % of the energy, both body waves on 1/R
    # on the signal: the thud is there before the air
    parts = O._boom(10.0, 1000.0, np.random.default_rng(0), fireball=0.0, ground=True)
    assert parts[0][1] == pytest.approx(ev[0][0], abs=1e-9)


def test_fragments_lead_the_blast():
    lead = [a for a, _ in O.fragments(40, 100.0, np.random.default_rng(1))]
    assert lead and all(0.0 < a < 100.0 * (1 / 343 - 1 / 1218) + 1e-9 for a in lead)


def test_no_fragments_past_230_m(monkeypatch):
    called = []
    monkeypatch.setattr(O, "fragments", lambda *a, **k: called.append(a) or [])
    O.grenade(sr=SR, distance=300.0, fuse=0.05)
    assert not called
    O.grenade(sr=SR, distance=200.0, fuse=0.05)
    assert called


def test_claymore_sheet_cracks_only_in_front_and_leads_by_the_cone():
    front = O.ball_sheet(30.0, 0.0, np.random.default_rng(0))
    assert len(front) > 300
    assert max(a for a, _ in front) == pytest.approx(30.0 * (1 / 343 - 1 / 1218), abs=0.004)
    assert O.ball_sheet(30.0, 180.0, np.random.default_rng(0)) == []
    ahead = O.mine(sr=SR, kind="claymore", azimuth=0.0, seed=1)
    behind = O.mine(sr=SR, kind="claymore", azimuth=180.0, seed=1)
    lead = 30.0 * (1 / 343 - 1 / 1218)                        # 63 ms: the clacker at 0, then balls, then the blast
    assert abs(onset(behind, 0.5, 0.04) - 0.05) < 0.003
    assert O._boom(0.91, 30.0, np.random.default_rng(0), fireball=0.0, ahead=front)[0][1] == pytest.approx(max(a for a, _ in front))
    assert abs(onset(ahead, 0.02, 0.04) - (0.05 + 0.0)) < 0.03                 # the first balls come straight after the clacker
    assert np.abs(ahead[int(0.052 * SR):int(0.105 * SR)]).max() > 0.01


def test_frag_timeline():
    y = O.grenade(sr=SR, kind="frag", fuse=4.5, distance=15.0, seed=2)
    t = onset(y, 0.2, 1.0)
    assert 4.9 < t < 5.1 and np.abs(y[:int(4.7 * SR)]).max() < 0.5 * np.abs(y).max()


def test_stun_shock_is_bare_and_its_fuse_its_own():
    y = O.grenade(sr=SR, kind="stun", distance=1.5, seed=3)
    assert 1.9 < onset(y, 0.3, 0.6) < 2.1                     # 0.5 + 1.5 s fuse
    late = y[int(2.3 * SR):]
    assert np.abs(late).max() < 0.05                          # no body, no debris
    ringing = O.grenade(sr=SR, kind="stun", distance=1.5, tinnitus=1.0, seed=3)
    assert 4000 <= peak_hz(ringing[int(2.5 * SR):], 3000, 9000) <= 8000


def test_gl_pop_tube_and_delay():
    y = O.grenade(sr=SR, kind="gl_pop", distance=76.0, seed=1)
    assert 240 < peak_hz(y[:int(0.2 * SR)], 100, 500) < 320   # 281 Hz
    assert abs(onset(y, 0.2, 0.2) - (1.0 + 76.0 / 343.0)) < 0.05


def test_launch_tube_factors_follow_farnell():
    b = np.random.default_rng(0).uniform(-1, 1, 48000)
    assert abs(peak_hz(O.launch_tube(1.27, True, b), 30, 200) - 67.5) < 6
    assert abs(peak_hz(O.launch_tube(0.93, False, b), 100, 600) - 366) < 25
    assert abs(peak_hz(O.launch_tube(0.305, True, b), 100, 600) - 281) < 20


def test_law_opens_as_the_rocket_leaves_and_empty_only_clicks():
    y = O.launcher(sr=SR, kind="law", seed=1)
    assert abs(peak_hz(y[int(0.1 * SR):int(0.6 * SR)], 250, 600) - 366) < 40
    empty = O.launcher(sr=SR, kind="law", loaded=False, seed=1)
    assert len(empty) < 0.2 * len(y) and len(empty) / SR < 0.2


def test_back_pressure_shape():
    e = env(O._back_pressure(np.random.default_rng(0)), 10)
    assert abs(len(e) / SR - 0.42) < 1e-3 and np.mean(e[:int(0.02 * SR)]) < np.mean(e[int(0.08 * SR):int(0.14 * SR)])
    assert np.max(np.abs(O._back_pressure(np.random.default_rng(0)))) <= 1.0


def test_rpg_second_ignition_and_burn():
    y = O.launcher(sr=SR, kind="rpg", seed=1)
    assert len(y) / SR > 1.7                                  # the motor burns out to 500 m
    only = O.moving(np.ones(4800), 295.0, 1.0, x0=10.0)
    assert abs(int(np.argmax(only > 0)) / SR - np.hypot(1, 10.06) / 343) < 2e-3


def test_moving_source_doppler_past_mach_1():
    t = np.arange(2 * SR) / SR
    y = O.moving(np.sin(2 * np.pi * 1000 * t), 745.0, 3.0)
    assert abs(peak_hz(y, 100, 2000) - 1000.0 / (1.0 + 745.0 / 343.0)) < 8


def test_stinger_falls_away_and_javelin_ignites_late():
    y = O.missile(sr=SR, kind="stinger", seed=1)
    a, b = y[int(0.35 * SR):int(0.6 * SR)], y[int(1.3 * SR):int(1.8 * SR)]
    assert centroid(b) < 0.8 * centroid(a) and np.sqrt(np.mean(b ** 2)) < np.sqrt(np.mean(a ** 2))
    j = O.missile(sr=SR, kind="javelin", delay=0.3, seed=1)
    assert np.abs(j[int(0.12 * SR):int(0.28 * SR)]).max() < 0.2 * np.abs(j).max()   # silence between eject and catch


def test_stinger_crack_leads_by_spec():
    y = O.missile(sr=SR, kind="stinger", distance=100.0, seed=1)
    assert abs(onset(y, 0.05, 0.03) - 0.157) < 0.02


def test_salvo_rate():
    y = O.rocket_salvo(sr=SR, count=2, rate=0.5, seed=1)
    assert 1.75 < onset(y, 0.3, 1.0) < 2.35                   # the second rocket, +-10 %


def test_flyby_drops_by_doppler():
    y = O.rocket_flyby(sr=SR, speed=200.0, closest=5.0, dur=3.0, seed=1)
    i = int(np.argmax(env(y, 20)))
    assert centroid(y[i + int(0.1 * SR):]) < 0.8 * centroid(y[max(0, i - int(0.6 * SR)):i - int(0.1 * SR)])


def test_handling_modes_and_drop_delays():
    y = O._tube_knock()
    f, s = spectrum(y)
    for m in (300.0, 826.0, 3200.0):
        k = (f > m * 0.9) & (f < m * 1.1)
        assert abs(f[k][np.argmax(s[k])] - m) < 5
    d = O.launcher_handling(sr=SR, action="drop", seed=1)
    e = env(d, 1.0)
    for t in (0.09, 0.11, 0.22):                              # 90, 20, 110 ms apart
        i = int(t * SR)
        assert e[i + 40:i + 200].max() > 1.05 * e[i - 100:i - 10].max() * 0.5
    assert not np.array_equal(d, O.launcher_handling(sr=SR, action="drop", loaded=False, seed=1))
    assert 4000 < peak_hz(O.launcher_handling(sr=SR, action="latch", seed=1)) < 6000


def test_gun_shock_lead_matches_spec():
    for v, lag in ((827.0, 0.171), (1750.0, 0.234), (745.0, 0.157)):
        assert abs(100.0 * (1 / 343 - 1 / v) - lag) < 0.002


@pytest.mark.parametrize("fn,lag", [(O.howitzer, 0.171), (O.tank_gun, 0.234)])
def test_muzzle_lag_on_the_signal(fn, lag):
    kw = {"breech": False} if fn is O.tank_gun else {}
    y = fn(sr=SR, azimuth=0.3, seed=1, **kw)
    assert abs(onset(y, 0.1, 0.03) - lag) < 0.015


def test_no_crack_past_75_degrees():
    assert all(len(p[0]) > 2000 for p in O._muzzle(6.0, 100.0, 80.0, 827.0, 155.0, 700.0, 5.08, np.random.default_rng(0)))


def test_overhead_crack_then_boom():
    y = O.shell_crack_overhead(sr=SR, miss=50.0, range=1000.0, seed=1)
    want = np.hypot(1000.0, 50.0) / 343.0 - (1000.0 / 827.0 + 50.0 * np.cos(np.arcsin(343 / 827)) / 343.0)
    assert abs(onset(y, 0.05, 0.2) - want) < 0.05


def test_cole_constants_against_the_13_g_test():
    w = 0.013
    rmax = 3.38 * (w / (2.0 + 10.1)) ** (1 / 3)
    t = 2.11 * w ** (1 / 3) / (2.0 + 10.1) ** (5 / 6)
    assert abs(rmax - 0.35) < 0.02 and abs(t - 0.062) < 0.004


def test_mortar_tube_is_quarter_wave():
    y = O.launch_tube(1.27, True, np.random.default_rng(0).uniform(-1, 1, 48000))
    f = np.fft.rfftfreq(len(y), 1 / 48000)
    spec = np.abs(np.fft.rfft(y))
    assert 60 < f[np.argmax(spec[:400])] < 75


def test_mortar_fire_and_drop():
    y = O.mortar(sr=SR, kind="fire", seed=1)
    assert 55 < peak_hz(y, 30, 120) < 80
    d = O.mortar(sr=SR, kind="drop", seed=1)
    assert 0.48 < onset(d, 0.5, 0.2) < 0.56                   # sqrt(2 L / g) = 0.51 s


def test_flamethrower_jet_peak():
    y = O.flamethrower(sr=SR, kind="burst", dur=2.0, seed=1)
    f, s = spectrum(y, 300, 4000)
    s = np.convolve(s, np.ones(801) / 801, mode="same")
    assert 600 < f[np.argmax(s)] < 1400                       # St U/D = 920 Hz


def test_shell_whistle_rises_one_octave():
    y = O.shell_incoming(sr=SR, dur=2.0, f0=900.0, seed=1)
    a, b = y[int(1.0 * SR):int(1.2 * SR)], y[int(1.75 * SR):int(1.95 * SR)]
    fa, fb = peak_hz(a, 300, 3000), peak_hz(b, 300, 3000)
    assert fb > fa and abs(fb - 900.0 * 2 ** 0.925) < 80


def test_torpedo_doppler_in_water():
    y = O.torpedo(sr=SR, kind="steam", dur=4.0, seed=1)
    a, b = peak_hz(y[:int(1.5 * SR)], 680, 760), peak_hz(y[-int(1.8 * SR):], 680, 760)
    want = 720.0 * (1500.0 / (1500.0 - 23.7) - 1500.0 / (1500.0 + 23.7))
    assert a > b and abs((a - b) - want) < 8


def test_piston_flow_above_engine():
    y = O.torpedo(sr=SR, kind="piston", dur=2.0, seed=1)
    f, s = spectrum(y)
    assert np.sum(s[(f > 400) & (f < 2000)] ** 2) > np.sum(s[(f > 30) & (f < 60)] ** 2)


def test_supercav_doppler_at_its_speed():
    n = 4 * SR
    t = np.arange(n) / SR
    y = O.moving(np.sin(2 * np.pi * 1000 * t), 103.0, 10.0, x0=-206.0, c=1500.0)
    a, b = peak_hz(y[:SR], 900, 1200), peak_hz(y[-SR:], 900, 1200)
    assert abs(a - 1000 / (1 - 103 / 1500)) < 5 and abs(b - 1000 / (1 + 103 / 1500)) < 5
    assert len(O.torpedo(sr=SR, kind="supercav", dur=2.0, seed=1)) / SR > 2.0


@pytest.mark.parametrize("w,d,r,T", [(118.0, 75.0, 50.0, 0.255)])
def test_underwater_periods_impulse_and_peak_on_signal(w, d, r, T):
    y = O.underwater_blast(w, r, d, np.random.default_rng(0))
    k = int(0.01 * SR)
    smooth = np.convolve(y, np.ones(k) / k, mode="same")
    q = int(0.2 * T * SR) // 2 + k
    t1 = T
    t2 = t1 + 0.34 ** (1 / 3) * T                                # 1.698 T (was 1.75 T)
    t3 = t2 + (0.34 * 0.54) ** (1 / 3) * T                       # 2.266 T (was 2.2525 T)
    for t, frac in ((t1, 1.0), (t2, 0.63), (t3, 0.4)):
        i = int(t * SR)
        assert abs((i - q + np.argmax(smooth[i - q:i + q])) / SR - t) < 0.025
    imp_shock = np.sum(y[:int(0.02 * SR)])
    imp_bubble = np.sum(y[int(T * SR) - q:int(T * SR) + q])
    assert 0.6 < imp_bubble / imp_shock < 2.1                 # Ar14: 0.84-2.1; low-pass and window loosen the bottom
    a = w ** (1 / 3)
    assert abs(O.UW_PM * 1e-6 * (a / r) ** 1.13 - 3.8) < 0.1  # spec: Pm 3.8 MPa
    lb, ft, psi = 0.45359237, 0.3048, 6894.757293             # TNT 2.16e4 psi, exponent 1.13 and K 4.36 in lb, ft converted to SI
    assert abs(O.UW_PM - 2.16e4 * psi * (ft / lb ** (1 / 3)) ** 1.13) < 0.01e6
    assert abs(O.UW_T - 4.36 / lb ** (1 / 3) * ft ** (5 / 6)) < 1e-3 and abs(O.UW_D - 33 * ft) < 1e-2
    assert O.UW_KEPT == (0.34, 0.54)
    assert abs(96.5e-3 * a * (a / r) ** -0.22 - 0.79) < 0.01  # spec: theta 0.79 ms


def test_mk14_period_at_10_m():
    assert abs(2.11 * 230.0 ** (1 / 3) / 20.1 ** (5 / 6) - 1.06) < 0.01
    assert abs(2.11 * 380.0 ** (1 / 3) / 20.1 ** (5 / 6) - 1.25) < 0.01
    y = O.torpedo(sr=SR, kind="hit", seed=1)
    assert len(y) / SR > 3.0 * 1.2


def test_surface_cuts_the_shock():
    a = O.underwater_blast(100.0, 100.0, 3.0, np.random.default_rng(0), pulses=1, surface=False)
    b = O.underwater_blast(100.0, 100.0, 3.0, np.random.default_rng(0), pulses=1, surface=True)
    i = int(2 * 3.0 / 1500.0 * SR) + 48
    assert np.abs(b[i:i + 96]).mean() != np.abs(a[i:i + 96]).mean() and np.sum(b[i:i + 96]) < np.sum(a[i:i + 96])


def test_depth_charge_sinks_17_s():
    y = O.depth_charge(sr=SR, depth=75.0, sink=True, seed=1)
    assert 16.5 < onset(y, 0.5, 5.0) < 17.5


def test_blast_kinds_differ_as_farnell_says():
    r = {k: O.blast_sfx(sr=SR, kind=k, charge=1.0, distance=30.0, ground=False, seed=1) for k in O.BLASTS}
    late = lambda y: np.sum(y[int(0.2 * SR):] ** 2) / np.sum(y ** 2)
    assert late(r["demolition"]) < 0.05 < late(r["grenade"]) < late(r["film"])
    neg = lambda w: O.blast(w[0], 30.0, np.random.default_rng(0), fireball=0.0, neg=w[1]).min()
    assert neg((1.0, 0.6)) < 3.0 * neg((1.0, 0.1))            # fuel-air: the deep suck


def test_earplug_curve():
    from genny.ordnance import earplug
    t = np.arange(SR) / SR
    for hz, db in ((125, -5.0), (1000, -10.0), (4000, -20.0)):
        y = earplug(np.sin(2 * np.pi * hz * t), SR)
        assert abs(20 * np.log10(np.abs(y).max()) - db) < 0.3
    assert earplug(np.zeros((100, 2)), SR).shape == (100, 2)


def test_unknown_kinds_are_refused():
    for fn in (O.grenade, O.mine, O.launcher, O.missile, O.mortar, O.flamethrower, O.torpedo, O.blast_sfx):
        with pytest.raises(ValueError):
            fn(sr=22050, kind="nope")
    with pytest.raises(ValueError):
        O.launcher_handling(sr=22050, action="nope")


def test_ground_waves_of_a_poisson_solid():
    assert abs(O.V_S / O.V_P - 3 ** -0.5) < 1e-9                 # nu = 0.25
    assert abs(O.V_R / O.V_S - 0.9194) < 1e-9                    # root of the Rayleigh equation at nu = 0.25
    assert O.GROUND_SHARE == (0.07, 0.26, 0.67)
