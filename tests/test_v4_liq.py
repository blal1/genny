"""V4_LIQ: open Liquides lines (bubbles, plumes, drops, pour, underwater bands, class metric)."""
import math

import numpy as np

import genny.matter as m
from genny import calibrate as C
from genny import nature as N
from genny.physical import ambience as A
from genny.physical import particles as P

SR = m.SR


def _peak_hz(y, lo=20.0, hi=20000.0):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    k = (f >= lo) & (f <= hi)
    return float(f[k][np.argmax(s[k])])


# ---------------------------------------------------------------- single bubble: amplitude, radiation, image

def test_xue_amplitude_goes_as_r_minus_3_and_moss_as_r():
    r = np.array([0.5e-3, 1e-3, 2e-3, 4e-3])
    xue = m.excitation_amp(r, "xue")
    moss = m.excitation_amp(r, "moss")
    slope_x = np.polyfit(np.log(r), np.log(xue), 1)[0]
    slope_m = np.polyfit(np.log(r), np.log(moss), 1)[0]
    assert abs(slope_x + 3.0) < 0.1 and abs(slope_m - 1.0) < 1e-9
    assert abs(float(m.excitation_amp(1e-3, "xue")) - 0.1) < 1e-12


def test_near_surface_damping_drops_the_radiation_term():
    r = 0.325e-3
    w0, b, _, d = m.bubble_params(r)
    _, b0, _, d0 = m.bubble_params(r, radiation=False)
    assert abs((float(d) - float(d0)) - float(w0) * r / m.C_WATER) < 1e-12
    assert float(b0) < float(b)
    assert abs(m._beta_of(float(w0), False, 1.0, 0.0) / float(b0) - 1.0) < 0.01


def test_image_dipole_gain_is_2_sin_kd():
    w0 = 2 * math.pi * 3000.0
    k = w0 / m.C_WATER
    for d in (0.002, 0.01, 0.05):
        assert abs(float(m.image_gain(w0, d)) - 2 * abs(math.sin(k * d))) < 1e-12
    assert float(m.image_gain(w0, 0.002)) < 0.06          # 2 kd: a shallow bubble is a weak dipole


def test_transfer_spread_and_drift_change_the_render():
    rng = lambda: np.random.default_rng(3)
    base = m.bubble_cloud(0.5, rng(), 40.0, 1e-3, 3e-3)
    same = m.bubble_cloud(0.5, rng(), 40.0, 1e-3, 3e-3, transfer_db=10.0)
    assert np.array_equal(base, same)
    wide = m.bubble_cloud(0.5, rng(), 40.0, 1e-3, 3e-3, transfer_db=20.0)
    drift = m.bubble_cloud(0.5, rng(), 40.0, 1e-3, 3e-3, drift=30.0)
    assert not np.array_equal(base, wide) and not np.array_equal(base, drift)


# ---------------------------------------------------------------- coupled bubbles, plumes

def test_coupled_bubbles_collective_bounds():
    r = np.full(5, 1e-3)
    f = m.coupled_frequencies(r, np.zeros((5, 3)))
    f0 = float(m.bubble_params(1e-3)[0]) / (2 * math.pi)
    assert abs(f[0] / f0 - math.sqrt(2.0 / (2.0 - 1.0 + 5))) < 1e-6
    assert np.allclose(f[1:] / f0, math.sqrt(2.0))
    assert abs(float(m.collective_shift(5)) - math.sqrt(2.0 / 6.0)) < 1e-12
    far = m.coupled_frequencies(np.array([1e-3, 1e-3]), np.array([[0, 0, 0], [1.0, 0, 0]]))
    assert np.allclose(far / f0, 1.0, atol=2e-3)


def test_two_bubbles_low_mode_follows_r_over_s():
    r, s = 1e-3, 5e-3
    f = m.coupled_frequencies(np.array([r, r]), np.array([[0, 0, 0], [s, 0, 0]]))
    f0 = float(m.bubble_params(r)[0]) / (2 * math.pi)
    M = 1.0 / math.sqrt(s * s / (r * r) + 4.0)
    assert abs(f[0] / f0 - 1.0 / math.sqrt(1.0 + M)) < 1e-9


def test_plume_collective_mode_and_void_fraction():
    assert abs(m.plume_frequency(0.75) - 6.3) < 0.1
    assert abs(m.plume_frequency(0.1) - 47.0) < 1.0
    assert m.PLUME_VOID == 0.34


def test_plume_radii_and_skin_depth():
    r = P.plume_radii(np.random.default_rng(0), 20000)
    assert r.min() >= 0.1e-3 and r.max() <= 5e-3
    small = r[(r > 0.2e-3) & (r < 1.0e-3)]
    h, e = np.histogram(np.log(small), 12)
    c = 0.5 * (e[1:] + e[:-1])
    assert abs(np.polyfit(c, np.log(h / np.diff(np.exp(e))), 1)[0] + 2.5) < 0.2
    assert abs(float(P.plume_skin_depth(100.0)) - 0.40) < 1e-9
    assert abs(float(P.plume_skin_depth(20000.0)) - 0.001) < 1e-9


def test_bubble_field_plume_attenuates_highs():
    a, ia = P.bubble_field(1.0, np.random.default_rng(1), rate=200.0, r_range=(0.3e-3, 3e-3), return_info=True)
    b = P.bubble_field(1.0, np.random.default_rng(1), rate=200.0, r_range=(0.3e-3, 3e-3), plume=0.05)
    assert np.sqrt(np.mean(b ** 2)) < 0.5 * np.sqrt(np.mean(a ** 2))


def test_micro_bubbles_rules():
    rng = np.random.default_rng(0)
    off, r, a = P.micro_bubbles(rng, 1.5e-3, 1.0)
    assert len(r) == 0
    counts = [len(P.micro_bubbles(rng, 4e-3, 1.0)[1]) for _ in range(400)]
    assert max(counts) <= 12 and 4.0 < np.mean(counts) < 8.0         # uniform on 0..3000 x 0.004
    off, r, a = P.micro_bubbles(np.random.default_rng(5), 5e-3, 2.0)
    assert np.all((off <= 0) & (off >= -0.1)) and np.all((r >= 1e-4) & (r <= 1e-3))
    assert np.allclose(a, 50.0 * 2.0 * r ** (1.0 / 3.0))


# ---------------------------------------------------------------- pop, forcing, boundaries

def test_bubble_pop_level_and_envelope():
    y = P.bubble_pop(1e-3, np.random.default_rng(0), life=0.01)
    pk = float(np.max(np.abs(y)))
    assert 0.001 <= pk <= 0.03
    assert np.max(np.abs(y[-max(2, y.size // 10):])) < 1e-3 * pk       # a 1 mm cap bursts in ~20 samples at 48 kHz


def test_fizz_pops_add_energy_only_when_asked():
    a = np.asarray(m.fizz(sr=SR, dur=0.6, seed=2))
    b = np.asarray(m.fizz(sr=SR, dur=0.6, seed=2, pops=1.0))
    assert not np.array_equal(a, b)


def test_pinch_forcing_is_t_squared_for_half_a_period():
    r = 1e-3
    w0 = float(m.bubble_params(r)[0])
    t = np.array([50e-6, 100e-6, 2 * math.pi / w0])
    f = m.forcing(t, r, "pinch", theta=math.radians(45.0))
    assert abs(f[1] / f[0] - 4.0) < 1e-9 and f[2] == 0.0
    eta = 1.0
    assert abs(f[0] + 9 * m.GAMMA * m.SIGMA_W * eta * math.sqrt(2) / (4 * m.RHO_W * r ** 3) * t[0] ** 2) < 1e-6 * abs(f[0])


def test_merge_time_and_forced_bubble_pitch():
    tl = m.merge_time(1e-3, 0.6)
    assert abs(tl - (0.6e-3) ** 2 / (m.MERGE_C ** 2 * math.sqrt(m.SIGMA_W * 1e-3 / m.RHO_W))) < 1e-15
    y = m.forced_bubble(2e-3, "pinch", np.random.default_rng(0))
    f0 = float(m.bubble_params(2e-3)[2]) / (2 * math.pi)
    assert abs(_peak_hz(y) / f0 - 1.0) < 0.03


def test_capacitance_pitch_follows_sqrt_c():
    assert abs(float(m.capacitance_pitch(3e-3, surface=3e-3 * 1.31 / (2 * 0.31))) - math.sqrt(1.31)) < 1e-9
    assert float(m.capacitance_pitch(3e-3, wall=0.01)) < 1.0
    assert float(m.capacitance_pitch(3e-3, surface=0.01)) > 1.0


def test_bubbles_flow_sets_the_rate():
    r = 2e-3
    rate = 0.5 / 60000.0 / (4.0 / 3.0 * math.pi * r ** 3)
    assert abs(rate - 249.0) < 1.0
    a = np.asarray(m.bubbles(sr=SR, dur=1.0, size=2.0, flow_lpm=0.5, seed=1))
    b = np.asarray(m.bubbles(sr=SR, dur=1.0, size=2.0, seed=1))
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------- drops

def test_terminal_speed_and_crater_frequency():
    assert abs(float(m.terminal_speed(2.0)) - (9.65 - 10.3 * math.exp(-1.2))) < 1e-12
    assert abs(float(m.crater_freq(2.4, speed=4.0)) - 4.0 / 1.2e-3) < 1e-6
    assert abs(m.CAVE_DROP_R - 3.25e-3) < 0.01e-3


def test_drip_harmonic_and_crater_are_off_by_default():
    a = np.asarray(m.drip(sr=SR, seed=4))
    b = np.asarray(m.drip(sr=SR, seed=4, harmonic=0.0, crater=0.0))
    c = np.asarray(m.drip(sr=SR, seed=4, harmonic=1.0))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_entrain_weight_kernel():
    tau = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.2])
    assert np.allclose(m.entrain_weight(tau), [1.0, 1 - 4 * 0.25 ** 3, 0.5, 4 * 0.25 ** 3, 0.0, 0.0])
    a = np.asarray(m.splash(sr=SR, seed=1))
    b = np.asarray(m.splash(sr=SR, seed=1, entrain="stochastic"))
    assert not np.array_equal(a, b)


def test_cave_drop_click_is_the_contact_time():
    D = 2 * m.CAVE_DROP_R * 1e3
    ms = D / float(m.terminal_speed(D))
    assert 0.6 < ms < 0.85


# ---------------------------------------------------------------- pour

def test_hot_pour_vapour_raises_the_air_pitch_and_bubble_size():
    assert m.air_speed_ratio(20.0) == 1.0
    x80 = m.vapour_fraction(80.0)
    assert abs(x80 - 0.474) < 0.005
    assert 1.17 < m.air_speed_ratio(80.0) < 1.22          # 411 / 344.5 m/s
    assert abs(m.vapour_bubble_scale(80.0) - ((1 - m.vapour_fraction(20.0)) / (1 - x80)) ** (1 / 3)) < 1e-12
    a = np.asarray(m.pour(sr=SR, dur=1.5, seed=0))
    b = np.asarray(m.pour(sr=SR, dur=1.5, seed=0, temperature=20.0, collective=0.0))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- surf, underwater

def test_surf_puff_gate_windows_are_50_ms():
    env = np.ones(SR * 2)
    env[-SR // 10:] = 0.0
    g = A.puff_gate(env, np.random.default_rng(0), 5.0)
    edges = np.flatnonzero(np.diff(np.concatenate([[0.0], g, [0.0]])))
    runs = (edges[1::2] - edges[::2]) / SR
    assert len(runs) and np.min(runs) >= 0.05 - 1e-9


def test_active_window_froude_scaling():
    T_lab = math.sqrt(2 * math.pi * 0.25 / 9.81)
    a, b = A.active_window(T_lab)
    assert abs(a - 0.3) < 1e-9 and abs(b - 0.8) < 1e-9
    a10, _ = A.active_window(10.0)
    assert abs(a10 / 0.3 - 10.0 / T_lab) < 1e-9


def test_surf_spectrum_slopes():
    db = m.surf_spectrum_db
    assert abs(float(db(1000.0))) < 1e-12 and abs(float(db(700.0)) + 5.0) < 1e-9
    assert abs(float(db(70.0)) - float(db(700.0)) - 40.0) < 1e-9
    assert abs(float(db(10000.0)) + 50.0) < 1e-9


def test_underwater_layers_off_by_default_and_bands():
    a = np.asarray(m.underwater_ambience(sr=SR, dur=2.0, seed=0))
    b = np.asarray(m.underwater_ambience(sr=SR, dur=2.0, seed=0, surf=0.0))
    assert np.array_equal(a, b)
    lay = m.sea_noise_bands(SR * 2, np.random.default_rng(0))
    assert 40.0 <= _peak_hz(lay["tide"], 10.0, 2000.0) <= 110.0
    assert _peak_hz(lay["ships"], 1.0, 2000.0) < 100.0


# ---------------------------------------------------------------- class metric, fire out

def test_allamanche_distance_zero_on_itself_and_grows():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR * 2) * 0.1
    y = np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR) * 0.1
    assert C.allamanche_distance(x, x, SR) < 1e-6
    assert C.allamanche_distance(x, y, SR) > 3.0
    assert C.CLASS_METRIC["rain"] == "allamanche"


def test_fire_out_details_default_on():
    a = np.asarray(N.fire_out(sr=SR, dur=5.0, detail=0.0))
    b = np.asarray(N.fire_out(sr=SR, dur=5.0))
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------- Meteo, Feu: wind atoms, stand density, sword groove, JWL, avalanche precursor

def test_wind_atom_shape_and_interpolation():
    f = np.array([1000.0, 1020.0, 1100.0])
    g = A.wind_atom_gain(f, 2.0, 1000.0, 60.0, 0.01)
    assert g[0] == 2.0 and g[1] == 2.0 and abs(g[2] - 2.0 * math.exp(-0.01 * 70.0)) < 1e-12
    assert A.wind_atom_params(0.0) == A.WIND_ATOM_PRESETS["whistle"]
    assert np.allclose(A.wind_atom_params(1.0), A.WIND_ATOM_PRESETS["rush"])
    w = np.full(SR, 0.8)
    y = A.wind_atoms(w, np.random.default_rng(0), 0.0)
    alpha, B, Ai, tau, Ci, Fi = A.WIND_ATOM_PRESETS["whistle"]
    s = np.abs(np.fft.rfft(y[SR // 2:] * np.hanning(SR - SR // 2))) ** 2
    f = np.fft.rfftfreq(SR - SR // 2, 1.0 / SR)
    k = (f >= 100.0) & (f <= 5000.0)
    centre = (f[k] * s[k]).sum() / s[k].sum()      # the band is flat on top: a noise periodogram's argmax wanders over it
    assert abs(centre - (Ci * Ai * 0.8 + Fi)) < 0.6 * B


def test_stand_density_index_reineke():
    assert abs(A.stand_density_index(500.0, 25.4) - 500.0) < 1e-9
    assert abs(A.stand_density_index(500.0, 50.8) - 500.0 * 2 ** 1.605) < 1e-9
    assert abs(A.stand_level_db(800.0) - 10 * math.log10(2.0)) < 1e-9


def test_diffuse_field_keeps_power_and_decorrelates():
    x = np.random.default_rng(1).standard_normal(SR)
    y = A.diffuse_field([x], np.random.default_rng(2), n_waves=8)
    assert y.shape == (SR, 2)
    p = np.mean(y ** 2, axis=0).sum()
    assert 0.6 < p / np.mean(x ** 2) < 1.4
    assert abs(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) < 0.9
    z = A.diffuse_field([x], np.random.default_rng(2), n_waves=8, t60=0.5)
    assert not np.array_equal(y, z)


def test_sword_groove_rossiter_modes_at_3_mm():
    from genny.physical import modal as MD
    fd, _ = MD.cavity_frequencies(24.0, 3.07e-3, 1.5e-3, 0.5)
    assert all(abs(a / b - 1) < 0.01 for a, b in zip(fd, (3213.0, 7497.0, 11780.0, 16064.0)))
    from genny import foley as FO
    a = np.asarray(FO.sword(sr=SR, seed=0))
    b = np.asarray(FO.sword(sr=SR, seed=0, groove=1.0))
    assert not np.array_equal(a, b)


def test_tnt_jwl_and_charge_radius():
    from genny import ordnance as O
    assert abs(O.charge_radius(23.0) - 0.150) < 0.002
    assert abs(O.charge_radius(960.0) - 0.520) < 0.003
    assert abs(O.charge_radius(18000.0) - 1.381) < 0.005
    j = O.TNT_JWL
    v = 2.0
    want = j["A"] * (1 - 0.3 / (4.15 * v)) * math.exp(-4.15 * v) + j["B"] * (1 - 0.3 / (0.95 * v)) * math.exp(-0.95 * v) + 0.3 * 1630 * 4.294e6 / v
    assert abs(O.jwl_pressure(v) / want - 1) < 1e-12


def test_snow_precursor_rate_rises_then_quiet():
    from genny import geo as G
    y = G.snow_precursor(10.0, 200.0, np.random.default_rng(0))
    n = len(y)
    e = lambda a, b: float(np.mean(y[int(a * n):int(b * n)] ** 2))
    assert e(0.6, 0.8) > 3 * e(0.0, 0.2)
    assert e(1 - G.PRECURSOR_QUIET + 0.02, 1.0) < 1e-3 * e(0.6, 0.8)
    a = np.asarray(G.snow_avalanche(sr=SR, dur=3.0, seed=0))
    b = np.asarray(G.snow_avalanche(sr=SR, dur=3.0, seed=0, precursor=2.0))
    assert len(b) == len(a) + 2 * SR


def test_pickup_window_aperture():
    from genny import reverbs as R
    h = R.pickup_window(200.0, "humbucker")
    assert abs(h.sum() - 1) < 1e-12 and len(h) % 2 == 1
    assert len(R.pickup_window(200.0, "single")) < len(h)
    a = np.asarray(R.feedback_guitar(110.0, 0.5, sr=SR, seed=0))
    b = np.asarray(R.feedback_guitar(110.0, 0.5, sr=SR, seed=0, aperture=0.0))
    assert np.array_equal(a, b)
    c = np.asarray(R.feedback_guitar(110.0, 0.5, sr=SR, seed=0, aperture=0.04))
    assert c.shape == a.shape and not np.allclose(a, c)


def test_swing_sources_measured_geometry():
    from genny.physical import modal_data as md
    assert set(md.SWING_SOURCES) == {"metal_sword", "wood_sword", "bat", "wood3", "broom"}
    for pts in md.SWING_SOURCES.values():
        r = [p[0] for p in pts]
        assert len(pts) == 8 and r == sorted(r) and all(0.001 < p[1] < 0.07 for p in pts)
    assert md.SWING_SOURCES["bat"][-1] == (0.804, 0.0659)
    assert md.SWING_CORRELATION["bat"] == 2.0 and md.SWING_CORRELATION_DEFAULT == 5.0
