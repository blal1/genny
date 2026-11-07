"""Deep-read items of the Contact theme (reports/lecture_profonde/suivi/T_Contact.md): sourced numbers checked on the formula or the render."""
import math

import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from genny import contact as C
from genny.analysis import detect_clicks
from genny.designed import ROLL_PATTERN

SR = 44100


def _decay_rate(y, sr, f0, t1=0.05, t2=0.3, bw=0.1):
    """1/s of the band around f0, from its envelope at two times."""
    sos = butter(2, [f0 * (1 - bw), f0 * (1 + bw)], "bandpass", fs=sr, output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, y)))
    return math.log(e[int(t1 * sr)] / e[int(t2 * sr)]) / (t2 - t1)


def _centroid(y, sr=SR):
    f, p = welch(y, sr, nperseg=4096)
    return float(np.sum(f * p) / np.sum(p))


def _env_rate(y, sr=SR, lo=0.5, hi=30.0):
    h = int(sr * 0.005)
    e = np.convolve(np.abs(y), np.ones(h) / h, "same")[::h]
    e = e - e.mean()
    Y = np.abs(np.fft.rfft(e * np.hanning(len(e)), 16 * len(e)))
    f = np.fft.rfftfreq(16 * len(e), h / sr)
    m = (f > lo) & (f < hi)
    return float(f[m][np.argmax(Y[m])])


# ------------------------------------------------------------------- Hertz
def test_hertz_replaces_the_series_of_table_levels():
    es = C.e_star("steel", "wood")
    assert C._hertz_pair("steel", "wood", 0.01) == pytest.approx(4 / 3 * es * math.sqrt(0.01))
    # R 1 cm steel on wood: 7 x the old table series, so a contact 0.46 x as long
    assert C._hertz_pair("steel", "wood", 0.01) / C._series(C.MATERIALS["steel"].k, C.MATERIALS["wood"].k) == pytest.approx(7.2, rel=0.1)
    m = C._ball_mass("steel", 0.02)
    k = C._hertz_pair("steel", "wood", 0.01)
    r = C.ImpactInteractor(None, mass=m, k=k, mu=0.29).strike(1.0, 0.01, SR, oversample=16)
    assert r.contact_time == pytest.approx(C.hc_contact_time(m, k, 1.5, 0.29, 1.0), rel=0.03)


def test_hertz_contact_time_coefficient_and_inversion():
    """Chaigne & Kergomard 1.141-1.145: tau = 3.218 (m^2/(K^2 V))^(1/5); exact Hertz integral 3.2145."""
    m, k, v = 0.01, 1e9, 2.0
    tau = C.hc_contact_time(m, k, 1.5, 0.0, v)
    assert tau == pytest.approx(3.218 * (m * m / (k * k * v)) ** 0.2, rel=0.005)
    fmax = k * (5 * m * v * v / (4 * k)) ** 0.6
    assert C.hertz_fit(tau, fmax, m) == pytest.approx(k, rel=0.01)


def test_rigid_wall_case_of_papetti():
    """Papetti et al. 2011 case 2: k 1e9, mu 0.5, alpha 1.5, vin 1: the K method's v_out within 3 % (AM1 +2.551 %)."""
    r = C.ImpactInteractor(None, mass=0.01, k=1e9, alpha=1.5, mu=0.5).strike(1.0, 0.02, SR, oversample=16)
    assert r.restitution == pytest.approx(abs(C.hc_rebound(0.5, 1.0)), rel=0.03)


def test_traer_ten_point_nine_is_a_mass_ratio():
    """0.7 g and 7.6 g balls: 10.9 is the mass ratio the paper took, its own half-sine gives sqrt = 3.3; Hertz with R ~ m^(1/3) 2.2."""
    def t(m):
        r = (3 * m / (4 * math.pi * 7850)) ** (1 / 3)
        return C.hc_contact_time(m, C._hertz_pair("steel", "steel", r), 1.5, 0.0, math.sqrt(2 * C.G * 0.08))
    assert 7.6 / 0.7 == pytest.approx(10.9, rel=0.01)
    assert t(7.6e-3) / t(0.7e-3) == pytest.approx((7.6 / 0.7) ** (1 / 3), rel=0.01)


# ------------------------------------------------------------------- bodies and caps
def test_fixed_thickness_gives_one_over_l_squared_and_no_20_hz_floor():
    f = [C._make_body("bar", "wood", L, thickness=0.03).freqs[0] for L in (2.5, 3.0)]
    assert f[0] / f[1] == pytest.approx(1.44, rel=1e-6)
    assert f[1] == pytest.approx(22.3733 / (2 * math.pi * 9) * math.sqrt(11e9 / 650) * 0.03 / math.sqrt(12), rel=0.001)
    prop = [C._make_body("bar", "wood", L).freqs[0] for L in (2.5, 3.0)]
    assert prop[0] / prop[1] == pytest.approx(1.2, rel=1e-6)
    assert f[1] < 20.0                                                                 # was held at 20 Hz: 2.5 m and 3 m were identical


def test_impact_goes_past_50_kg_and_30_ms():
    a, b = (C.impact(SR, "steel", size=1.0, striker_mass=m, velocity=30.0, dur=0.5) for m in (50.0, 250.0))
    assert np.isfinite(b).all() and not np.array_equal(a, b)                           # both were clipped at 50 kg
    c = C.impact(SR, "steel", size=1.0, striker_mass=250.0, velocity=66.0, dur=0.5)
    assert np.isfinite(c).all() and _centroid(c) > _centroid(b)                        # and at 30 m/s; faster = brighter
    k = C._striker_k(C.MATERIALS["steel"], 0.6, 250.0)
    assert C.hc_contact_time(250.0, k, 1.5, 0.27, 66.0) > C.hc_contact_time(50.0, C._striker_k(C.MATERIALS["steel"], 0.6, 50.0), 1.5, 0.27, 30.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rumble_has_no_stray_clicks(seed):
    y = C.rumble(SR, dur=3.0, seed=seed)
    f, p = welch(y, SR, nperseg=4096)
    assert len(detect_clicks(y)) == 0
    assert 10 * math.log10(p[f > 1000].sum() / p.sum()) < -45


@pytest.mark.parametrize("v,L,kw,seed", [(3.0, 1.0, {"roughness": 0.1}, 0), (3.0, 1.0, {"roughness": 0.1}, 1), (3.0, 1.0, {}, 0), (1.0, 0.25, {}, 1)])
def test_roll_joints_carry_v_over_l(v, L, kw, seed):
    assert _env_rate(C.roll(SR, dur=4.0, speed=v, joints=L, seed=seed, **kw)) == pytest.approx(v / L, rel=0.035)


# ------------------------------------------------------------------- can
def test_roll_can_rings_the_measured_modes_at_their_levels():
    y = C.roll(SR, dur=3.0, speed=0.6, radius=0.033, body="can", surface="concrete", surface_size=3.0, seed=1)
    assert np.isfinite(y).all()
    f, p = welch(y, SR, nperseg=8192)
    def lev(f0):
        m = np.abs(f - f0) < 12
        return 10 * math.log10(p[m].max())
    bg = 10 * math.log10(np.median(p[(f > 500) & (f < 1500)]))
    for f0 in (359.0, 426.0, 1748.0):
        assert lev(f0) > bg + 3
    assert lev(426.0) - lev(359.0) == pytest.approx(20 * math.log10(0.513), abs=3.0)     # Farnell fig. 17.21; measured -3.7 dB against -5.8


def test_roll_can_level_grows_as_sqrt_speed_and_repeats_per_turn():
    n = 3 * SR
    r = []
    for v in (0.05, 0.2):                                                              # the same first hit, k = 5.6, turn 1/5.6
        t = (1 / 5.6) / (v / (2 * math.pi * 0.033))
        y = C._can_ring(np.zeros(n), np.full(n, v), 0.033, 0.015, SR)
        r.append(np.max(np.abs(y[int((t - 0.01) * SR):int((t + 0.03) * SR)])))
    assert r[1] / r[0] == pytest.approx(2.0, rel=0.15)                                 # lot_B03: amplitude in sqrt(speed)
    turn = 0.4 / (2 * math.pi * 0.033)
    n = 8 * SR
    y = C._can_ring(np.zeros(n), np.full(n, 0.4), 0.033, 0.015, SR)
    h = int(SR * 0.005)
    e = np.convolve(np.abs(y), np.ones(h) / h, "same")[::h]
    Y = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(len(e))))
    fr = np.fft.rfftfreq(len(e), h / SR)
    floor = np.median(Y[(fr > 0.5) & (fr < 30.0)])
    for k, _ in ROLL_PATTERN:                                                          # a line at k per turn for each term of the pattern
        assert Y[np.abs(fr - k * turn) < 0.2].max() > 2.0 * floor
    with pytest.raises(ValueError):
        C.roll(SR, dur=0.1, body="box")


# ------------------------------------------------------------------- damping laws
def test_new_laws_at_their_published_values():
    w = 2 * math.pi * 1000.0
    hyb = C.material_damping([10000.0], "ceramic", law="hybrid")[0]
    assert hyb == pytest.approx(20.7, abs=0.5)                                         # Sterling 2019: d = 20 with g3 = 1.027
    assert 0.5 * (3.9 + 5.2e-9 * (10 * w) ** 2) == pytest.approx(12.2, abs=0.5)        # and 12 with g3 = 1
    assert C.material_damping([1000.0], "wood", law="caughey")[0] == pytest.approx(0.5 * (277 - 2e-6 * w ** 2 + 4.8e-15 * w ** 4), rel=1e-9)
    assert C.material_damping([1000.0], "plastic2", law="power")[0] == pytest.approx(0.5 * 0.19 * w ** 0.74, rel=1e-9)
    assert C.material_damping([1000.0], "aluminium", law="sterling_rayleigh")[0] == pytest.approx(0.5 * (0.225 + 1.45e-6 * w ** 2), rel=1e-9)
    assert C.material_damping([1000.0], "wood", law="obrien")[0] == pytest.approx(0.5 * (50 + 8e-6 * w ** 2), rel=1e-9)
    assert C.material_damping([1000.0], "mdf", law="zheng")[0] == pytest.approx(0.5 * (35 + 5e-6 * w ** 2), rel=1e-9)
    with pytest.raises(ValueError):
        C.material_damping([1000.0], "bronze", law="hybrid")


def test_polycarbonate_rings_0_85_s_at_1_khz():
    d = C.material_damping([1000.0], "polycarbonate", law="objectfolder")[0]
    assert 6.91 / d == pytest.approx(0.85, abs=0.02)
    assert 6.91 / C.material_damping([1000.0], "plastic", law="objectfolder")[0] == pytest.approx(0.20, abs=0.02)
    assert C.get_material("plexiglass").name == "plastic"
    y = C.impact(SR, "polycarbonate", "bar", 0.3, freq=1000.0, law="objectfolder", damping=0.0, modes=1, dur=1.2)
    assert _decay_rate(y, SR, 1000.0) == pytest.approx(d, rel=0.03)


def test_objectfolder_law_takes_its_own_elastic_constants():
    """Glass 2600, 62 GPa: 4883 m/s against 5292, -7.7 % in frequency."""
    a = C._make_body("bar", "glass", 0.3, damping=0.0, law="objectfolder").freqs[0]
    b = C._make_body("bar", "glass", 0.3, damping=0.0).freqs[0]
    assert a / b == pytest.approx(4883 / 5292, rel=0.002)
    assert math.sqrt(62e9 / 2600) == pytest.approx(4883, abs=1)


def test_contact_damping_formula_and_rendered_effect():
    b = C._make_body("bar", "steel", 0.3, 0.28, seed=0, scatter=0.0, rest_force=20.0)
    free = C._make_body("bar", "steel", 0.3, 0.28, seed=0, scatter=0.0)
    assert (b.decays - free.decays) == pytest.approx(0.5 * 0.3 * 20.0 * free.phi ** 2 / free.mass, rel=1e-9)
    assert (b.decays > free.decays).all()
    with pytest.raises(ValueError):
        C._make_body("bar", "bronze", 0.3, rest_force=1.0)
    kw = dict(position=0.5, damping=0.0, dur=1.5, seed=1, modes=1, freq=600.0)
    long, short = C.impact(SR, "steel", size=0.3, **kw), C.impact(SR, "steel", size=0.3, rest_force=50.0, **kw)
    body = C._make_body("bar", "steel", 0.3, 0.5, modes=1, freq=600.0, damping=0.0, seed=1)
    added = 0.5 * 0.3 * 50.0 * body.phi[0] ** 2 / body.mass
    assert _decay_rate(short, SR, 600.0) - _decay_rate(long, SR, 600.0) == pytest.approx(added, rel=0.05)


def test_ice_relaxation():
    """Kuroiwa: peak 3 kHz at -10 C, 330 Hz at -31.5 C; at -10 C and 300 Hz the Debye term is 0.00175."""
    assert C.ice_loss([300.0], -10.0)[0] - 0.002 == pytest.approx(0.00175, abs=0.0001)
    f = np.geomspace(50, 20000, 2000)
    assert f[np.argmax(C.ice_loss(f, -10.0))] == pytest.approx(3000, rel=0.05)
    assert f[np.argmax(C.ice_loss(f, -31.5))] == pytest.approx(330, rel=0.1)
    cold, table = (C.impact(SR, "ice", "bar", 0.3, freq=300.0, modes=1, damping=0.0, dur=1.0, ice_temp=t) for t in (-10.0, 1.0))
    assert _decay_rate(cold, SR, 300.0) == pytest.approx(math.pi * 300.0 * (0.00175 + 0.002) + 15.0, rel=0.05)
    assert _decay_rate(table, SR, 300.0) > 1.2 * _decay_rate(cold, SR, 300.0)


def test_mount_at_the_node_spares_mode_one():
    a = C.impact(SR, "steel", "bar", 0.4, dur=1.5, damping=0.0)
    b = C.impact(SR, "steel", "bar", 0.4, dur=1.5, damping=0.0, mount=20.0)
    assert not np.array_equal(a, b)
    r = C._bar_roots(2, "free")
    s = C._bar_shape(r, 0.2242, "free") ** 2
    assert s[0] < 1e-3 and s[1] > 0.3


# ------------------------------------------------------------------- Johnson
def test_wave_speeds_against_johnson_table_11_1():
    s = C.MATERIALS["steel"]
    assert s.c_long == pytest.approx(5900, rel=0.05) and s.c_shear == pytest.approx(3200, rel=0.05) and s.c_rayleigh == pytest.approx(3000, rel=0.05)
    assert C.MATERIALS["glass"].c_shear == pytest.approx(3400, rel=0.05)
    nu = C.replace(s, poisson=0.25)
    assert nu.c_rayleigh / nu.c_shear == pytest.approx(0.92, abs=0.005)


def test_johnson_formulas_at_his_examples():
    es = 1.0 / (2 * (1 - 0.3 ** 2) / 200e9)
    assert C.yield_speed(7850.0, 1e9, es) == pytest.approx(0.14, rel=0.1)              # eq. 11.39: "about 0.14 m/s"
    m = C._ball_mass("steel", 0.01)
    assert C.plastic_contact_time(m, 5e-3, 1e9) == pytest.approx(1e-5, rel=0.1)        # "1e-4 to 1e-5 s"
    assert C.plastic_contact_time(m, 5e-3, 1e9, 0.5) / C.plastic_contact_time(m, 5e-3, 1e9) == pytest.approx(1.6)
    e = [C.plastic_restitution(m, v, 5e-3, 1e9, es) for v in (30.0, 60.0)]
    assert e[1] / e[0] == pytest.approx(2 ** -0.25, rel=1e-6)                          # e ~ V^-1/4
    assert C.plastic_restitution(m, 1.0, 5e-3, 1e9, es) == 1.0                         # the report's 0.60 at 1 m/s is wrong: formula 1.4, capped
    assert C.plastic_restitution(m, 29.0, 5e-3, 1e9, es) == pytest.approx(0.6, abs=0.02)
    assert C.sphere_yield_load(0.01, 1e9, es) == pytest.approx(math.pi ** 3 * 1e-4 * 1.6e9 ** 3 / (6 * es ** 2))
    a, p0 = C.line_contact(1e4, 0.01, es)
    assert p0 == pytest.approx(2 * 1e4 / (math.pi * a), rel=1e-9)
    assert C.cylinder_yield_load(0.01, 1e9, es) == pytest.approx(math.pi * 0.01 * (1.67e9) ** 2 / es)
    k, al = C.power_profile_k(1, 1 / (2 * 0.01), es)
    assert (k, al) == pytest.approx((4 / 3 * es * 0.1, 1.5))
    assert [C.power_profile_k(n, 1.0, es)[1] for n in (2, 3, 5)] == pytest.approx([1.25, 7 / 6, 1.1])
    assert C.viscoelastic_restitution(0.1) == pytest.approx(math.sqrt(1 - 0.1 * math.pi))
    pc, ac = C.jkr_pulloff(0.05, 0.01, 1e6)
    assert pc == pytest.approx(3 * math.pi * 0.05 * 0.01) and ac == pytest.approx((9 * math.pi * 0.05 * 1e-4 / 4e6) ** (1 / 3))
    assert C.winkler_stiffness(1.0, 1.0) == 1.7 and C.winkler_stiffness(1.0, 1.0, line=True) == 1.18
    assert C.plasticity_index(1e11, 1e9, 1e-6, 1e4) == pytest.approx(10.0)


def test_tip_exponents():
    """Flat: Tc = pi sqrt(m/k), independent of V (Johnson, P = 2 E* a d). Cone alpha 2: Tc ~ V^(-1/3). Ball: V^(-1/5)."""
    es, m = C.e_star("steel", "wood"), 0.1
    k = 2 * es * 0.01
    t = [C.hc_contact_time(m, k, 1.0, 1e-6, v) for v in (0.5, 5.0)]
    assert t[0] == pytest.approx(math.pi * math.sqrt(m / k), rel=0.01) and t[1] == pytest.approx(t[0], rel=0.01)
    c = C.cone_k(es, 60.0)
    assert c == pytest.approx(2 / math.pi * es * math.tan(math.radians(60)))
    t = [C.hc_contact_time(m, c, 2.0, 1e-6, v) for v in (1.0, 8.0)]
    assert t[0] / t[1] == pytest.approx(2.0, rel=0.01)
    r = {tp: C.impact(SR, "wood", velocity=3.0, dur=0.3, tip=tp) for tp in ("ball", "cone", "flat")}
    assert all(np.isfinite(v).all() for v in r.values())
    assert not np.array_equal(r["ball"], r["flat"]) and not np.array_equal(r["ball"], r["cone"])
    with pytest.raises(ValueError):
        C.impact(SR, tip="nail")


def test_plastic_bounce_loses_more_when_hit_hard():
    m = C._ball_mass("steel", 0.02)
    es = C.e_star("steel", "metal")
    of_v = lambda u: C.plastic_restitution(m, u, 0.01, 1e8, es)
    t, v = C.bounce_schedule(0.5, 20.0, 0.97, t_min=0.01, e_of_v=of_v)
    e = v[1:] / v[:-1]
    assert e[0] == pytest.approx(of_v(20.0), rel=1e-6) and e[1] > e[0]                 # e rises as the speed falls
    assert e[1] / e[0] == pytest.approx((v[1] / v[0]) ** -0.25, rel=0.001)
    el, pl = C.bounce_schedule(0.3, 3.0, 0.9)[1], C.bounce_schedule(0.3, 3.0, 0.9, e_of_v=of_v)[1]
    assert len(pl) < len(el)
    a, b = C.bounce(SR, velocity=5.0), C.bounce(SR, velocity=5.0, yield_stress=1e8)
    assert np.isfinite(b).all() and not np.array_equal(a, b)


def test_restrike_and_hit_spread():
    rng = np.random.default_rng(0)
    rows = np.ones((2, 20000))
    j = C._restrike(rows, 0.2, 0.0, np.ones(2), rng)
    assert np.array_equal(j[0], rows[0])
    assert np.std(j[1] ** 2) == pytest.approx(0.2, rel=0.03)                           # Traer 2019 sec. 2.3
    g = C.lloyd_gain(0.7, np.random.default_rng(1).random(200000))
    assert np.mean(g * g) == pytest.approx(1.0, rel=0.01)                              # E[lerp^2] = c^2
    assert np.std(C.lloyd_gain(0.9, np.random.default_rng(1).random(20000))) > np.std(C.lloyd_gain(0.2, np.random.default_rng(1).random(20000)))
    assert not np.array_equal(C.bounce(SR, restrike_jitter=0.2, seed=2), C.bounce(SR, seed=2))
    assert not np.array_equal(C.bounce(SR, hit_spread=1.0, seed=2), C.bounce(SR, seed=2))


# ------------------------------------------------------------------- force trains and scraping
def test_sharpness_narrows_the_pulse_as_the_power_of_the_cosine():
    L = 1001
    f = {xi: C._pulse_train(2000, np.array([1000]), np.array([1.0]), np.array([L]), xi) for xi in (1.0, 4.0)}
    assert f[1.0].max() == pytest.approx(1.0, abs=1e-5) and f[4.0].max() == pytest.approx(1.0, abs=1e-5)
    half = {xi: np.sum(v > 0.5) / L for xi, v in f.items()}
    assert half[1.0] == pytest.approx(0.5, abs=0.002)
    assert half[4.0] == pytest.approx(2 * math.acos(2 * 0.5 ** 0.25 - 1) / (2 * math.pi), abs=0.002)
    a, b = C.interaction(SR, dur=0.4, action=240.0, seed=3), C.interaction(SR, dur=0.4, action=240.0, seed=3, sharpness=4.0)
    assert _centroid(b) > _centroid(a)


def test_rolling_modulation_follows_three_v_over_s():
    """Conan eq. 17: nu_m = 3 V/S, per sample of the speed curve (already there before this batch)."""
    n = 4 * SR
    f = C.conan_force(n, SR, C.conan_weights(240.0), 0.3, 0.5, 2.0, seed=1)
    m = np.abs(f)
    e = np.convolve(m, np.ones(441) / 441, "same")[::441]
    Y = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(len(e)), 8 * len(e)))
    fr = np.fft.rfftfreq(8 * len(e), 441 / SR)
    b = (fr > 4) & (fr < 40)
    assert fr[b][np.argmax(Y[b])] == pytest.approx(12.0, rel=0.05)


def test_scrape_power_level_goes_as_sqrt_v_and_is_silent_at_rest():
    prof = np.random.default_rng(0).standard_normal(1 << 14)
    n = SR
    std = {(v, fn): C.scrape_force(prof, np.full(n, v), np.full(n, math.sqrt(v * fn)), 1e-4, SR).std() for v, fn in ((0.1, 1.0), (0.4, 1.0), (0.1, 4.0))}
    assert std[(0.4, 1.0)] / std[(0.1, 1.0)] == pytest.approx(2.0, rel=0.1)            # sqrt(v Fn), van den Doel & Pai sec. 4.2
    assert std[(0.1, 4.0)] / std[(0.1, 1.0)] == pytest.approx(2.0, rel=0.1)
    y = C.scrape(SR, dur=0.6, speed=[0.0, 0.0, 0.5, 0.5], level="power", seed=0)
    q = int(0.15 * SR)
    assert np.abs(y[:q // 2]).max() < 0.02 * np.abs(y).max()
    assert not np.array_equal(y, C.scrape(SR, dur=0.6, speed=[0.0, 0.0, 0.5, 0.5], seed=0))


def test_curvature_limit_and_force():
    """alpha 0.05 light .. 0.01 heavy (Agarwal eq. 5-7): a harder pressed scraper follows sharper curves, |S''| <= 1/alpha before smoothing."""
    z = C.surface_texture("sandpaper", 1 << 14, seed=1) * 2e-5
    light, heavy = (C.constrained_trajectory(z, 1e-4, N) for N in (0.0, 1.0))
    assert np.isfinite(light).all() and light.std() == pytest.approx(z.std(), rel=0.01)
    al = 0.05
    assert (np.tanh(al * np.array([1e9])) / al)[0] == pytest.approx(1 / al)
    assert not np.allclose(light, heavy)
    n = SR // 2
    v, N = np.full(n, 0.3), np.ones(n)
    a, b = (C.curvature_force(z, v, N, 1e-4, SR, m, beta1=0.0) for m in (0.01, 0.02))
    assert b == pytest.approx(2 * a, rel=1e-9)                                         # f = m S'' v^2
    v2 = C.curvature_force(z, 2 * v, N, 1e-4, SR, 0.01, beta1=0.0)
    assert v2[1:n // 2 - 1:1] == pytest.approx(4 * a[3:n - 2:2][:n // 2 - 2], rel=1e-6)      # same path at twice the speed: v^2
    y = C.scrape(SR, dur=0.4, model="curvature", seed=1)
    assert np.isfinite(y).all() and np.abs(y).max() == pytest.approx(0.7, rel=0.01)


def test_sonic_distance_and_dilation():
    a = C._make_body("bar", "steel", 0.3)
    assert C.sonic_distance(a, a) == 0.0
    b = C._make_body("bar", "steel", 0.3, 0.45)
    assert C.sonic_distance(a, b) > 0.0
    f = np.array([100.0, 300.0])
    assert C.spectral_dilation(f, 0.5, 0.1) == pytest.approx([0.5 * 100 * math.sqrt(1.1), 0.5 * 300 * math.sqrt(1.9)])
    y = C.impact(SR, "wood", dilate=0.85, dilate_r=0.05, dur=0.3)
    assert np.isfinite(y).all() and not np.array_equal(y, C.impact(SR, "wood", dur=0.3))


def test_new_materials_render_and_table_values():
    cast = C.MATERIALS["cast_iron"]
    assert cast.young / (2 * (1 + cast.poisson)) == pytest.approx(45e9)                # Johnson Table 5.1
    carb = C.MATERIALS["carbide"]
    assert carb.young / (2 * (1 + carb.poisson)) == pytest.approx(300e9, rel=0.001)
    for m in ("polycarbonate", "polystyrene", "mdf", "plastic2", "cast_iron", "carbide"):
        y = C.impact(SR, m, dur=0.2)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.1
