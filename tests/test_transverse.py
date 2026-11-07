"""Needs left by the deep-reading themes in files they could not edit (reports/lecture_profonde/suivi/T_Transverse.md)."""
import math

import numpy as np
from scipy.signal import hilbert
from scipy.special import erf

from genny import aero as A
from genny import reverbs as R

SR = 16000


def _tone(f, dur):
    return np.sin(2 * np.pi * f * np.arange(int(dur * SR)) / SR)


def test_distance_turbulence_follows_rietdijk_log_amplitude_and_covariance():
    x = _tone(2000.0, 120.0)
    assert np.array_equal(R.distance(x, SR, metres=500, air=False, ref=500),
                          R.distance(x, SR, metres=500, air=False, ref=500, turbulence=0.0))
    y = R.distance(x, SR, metres=500, air=False, ref=500, turbulence=2.0, seed=3)
    la = np.log(np.abs(hilbert(y))[SR:-SR])
    sigma = float(A.scintillation_sigma(2000.0, 500.0))                 # with saturation, 2 kHz band centre
    assert abs(la.std() / sigma - 1.0) < 0.15
    la = la - la.mean()
    lag = int(A.TURB_L / 2.0 * SR)                                       # tau0 = L / v_perp
    rho = float(np.mean(la[:-lag] * la[lag:]) / np.mean(la * la))
    assert abs(rho - 0.5 * math.sqrt(math.pi) * erf(1.0)) < 0.15        # C(1) = Phi(1) = 0.747


def test_doppler_turbulence_wanders_faster_while_crossing_the_line_of_sight():
    x = _tone(2000.0, 20.0)
    still = R.doppler(x, SR, speed=60.0, closest=200.0)
    y = R.doppler(x, SR, speed=60.0, closest=200.0, turbulence=True, seed=1)
    assert y.shape == still.shape and np.all(np.isfinite(y))
    la = np.log(np.abs(hilbert(y.sum(1))) / (np.abs(hilbert(still.sum(1))) + 1e-12))
    b = la[:len(la) // 160 * 160].reshape(-1, 160).mean(1)              # 10 ms blocks
    rate = lambda u: np.mean(np.abs(np.diff(u))) / u.std()
    mid, ends = b[900:1100], np.concatenate([b[100:300], b[-300:-100]])
    assert rate(mid) > 1.4 * rate(ends)                                  # v_perp 60 m/s at the pass, ~23 m/s at 520 m


def test_rain_on_water_with_a_rate_is_the_drop_by_drop_model():
    from genny import matter as Ma
    from genny import nature as N
    a = np.asarray(N.rain_on(sr=48000, surface="eau", rate=5.0, dur=2.0, seed=2))
    b = np.asarray(Ma.rain_water(sr=48000, rate=5.0, dur=2.0, seed=2))
    m = min(len(a), len(b))
    assert np.corrcoef(a[2400:m - 4800], b[2400:m - 4800])[0, 1] > 0.999


def test_struck_bottle_with_water_drops_by_jundt_ratio():
    from genny import foley as FO
    from genny import matter as Ma

    def peak(y):
        f = np.fft.rfftfreq(len(y), 1 / 48000)
        p = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        return f[np.argmax(p * (f > 200))]
    for obj in ("bottle", "vase"):
        dry, full = (np.asarray(FO.clink(sr=48000, object=obj, liquid=h), float) for h in (0.0, 1.0))
        assert np.array_equal(dry, np.asarray(FO.clink(sr=48000, object=obj), float))
        assert abs(peak(full) / peak(dry) / float(Ma.glass_fill_ratio(1.0)) - 1.0) < 0.02


def test_gunshot_ground_reflection_lags_by_the_image_path():
    from genny import foley as FO
    from genny.physical import space
    d, h = 7.35, 1.6
    kw = dict(sr=48000, kind="rifle", distance=d, crack=0.0, echo=0.0, action=0.0, seed=4)
    dry = np.asarray(FO.gunshot(**kw), float)
    assert np.array_equal(dry, np.asarray(FO.gunshot(ground="none", **kw), float))
    wet = np.asarray(FO.gunshot(ground="hard", height=h, **kw), float)
    diff = wet[:len(dry)] - dry
    m = 1 << int(np.ceil(np.log2(2 * len(dry))))
    xc = np.fft.irfft(np.fft.rfft(diff, m) * np.conj(np.fft.rfft(dry, m)), m)[:480]
    lag = np.argmax(xc[24:]) + 24
    expect = (math.hypot(d, 2 * h) - d) / space.speed_of_sound(20.0)
    assert abs(lag / 48000 - expect) < 0.05e-3
    assert abs(lag / 48000 - 1.80e-3) / 1.80e-3 < 0.15                 # measured at about 1.6 m


def test_relay_contact_moves_its_operate_and_release_time_after_the_coil():
    from genny import foley as FO
    m = 1 << 16
    for st, ms in (("on", 14.0), ("off", 12.0)):
        a, b = (np.asarray(FO.switch(sr=48000, kind="relay", state=st, coil=c), float) for c in (0, 1))
        xc = np.fft.irfft(np.fft.rfft(b, m) * np.conj(np.fft.rfft(a, m)), m)
        assert abs(np.argmax(xc[:4800]) / 48.0 - ms) < 0.1
    on, off = (np.asarray(FO.switch(sr=48000, kind="relay", state=s), float) for s in ("on", "off"))
    assert not np.allclose(on, off)                                       # release bounces 7 ms, operate 5 ms


def test_doppler_tangent_pan_law_keeps_power_and_follows_chowning():
    x = _tone(500.0, 2.0)
    y = R.doppler(x, SR, speed=20.0, closest=5.0, air=False, law="tangent")
    y0 = R.doppler(x, SR, speed=20.0, closest=5.0, air=False)
    assert np.allclose((y ** 2).sum(1), (y0 ** 2).sum(1), atol=1e-9)
    _, r, xs, _ = R.flyby_path(len(x), SR, 20.0, 5.0, 0.5)
    th = (xs / r + 1.0) * np.pi / 4
    k = np.flatnonzero(np.abs(y[:len(th), 0]) > 1e-3)
    tn = np.tan(th[k] - np.pi / 4)
    assert np.allclose(y[k, 1] / y[k, 0], np.sqrt((1 + tn) / (1 - tn)), rtol=1e-6)


def test_reflected_blast_fits_meet_at_the_domain_split():
    from genny.physics import _blast_fit
    assert abs(_blast_fit("pr", 0.4) / 42.34e3 - 1) < 0.005                # MPa -> kPa, Z = 0.40-
    assert abs(_blast_fit("pr", 0.40001) / 42.73e3 - 1) < 0.005
    assert abs(_blast_fit("ir", 0.4) / 1875.41 - 1) < 0.005
    assert abs(_blast_fit("ir", 0.40001) / 1879.48 - 1) < 0.005


def test_reflected_blast_is_above_the_incident_one():
    from genny.physics import _blast_fit, blast_reflected, blast_wave
    for r in (2.0, 5.0, 20.0):
        pr, ir = blast_reflected(1.0, r)
        assert pr > blast_wave(1.0, r)[0] and ir > _blast_fit("is", r)    # always larger than the incident


def test_windows_break_only_above_the_glass_threshold():
    from genny.physics import Explosion, blast_damage, blast_wave
    assert blast_damage(999.0) == [] and blast_damage(1.0e3) == ["windows"]
    assert blast_damage(13e3) == ["windows", "light", "panels", "concrete"]
    near, far = 30.0, 400.0
    assert blast_wave(1.0, near)[0] >= 1e3 > blast_wave(1.0, far)[0]
    for d, broke in ((near, True), (far, False)):
        a = Explosion(1.0, d, 0.0, 0.0).render(1.0, sr=48000)
        b = Explosion(1.0, d, 0.0, 0.0, windows=1.0).render(1.0, sr=48000)
        assert (not np.array_equal(a, b)) is broke


def test_soil_presets_are_hess_delany_bazley_fits_and_order_the_first_dip():
    from genny.physical import space
    table = {"sand": 253000, "sharp_sand": 906000, "dry_loam": 370000, "wet_loam": 218000, "clay": 106000,
             "rolled_clay": 168000, "silt": 110000, "compacted_silt": 1045000}
    f = np.linspace(50.0, 5000.0, 4000)
    dips = []
    for name in ("clay", "sand", "sharp_sand", "compacted_silt"):
        sigma = R._ground_sigma(name)
        assert sigma == table[name] / 1000.0
        h = np.abs(space.ground_effect(f, 1.75, 0.44, 0.44, sigma))
        dips.append(f[np.argmin(h)])
    assert dips == sorted(dips)                                           # stiffer soil, higher first dip


def test_listening_test_sample_size_is_lindau_1071():
    from genny.calibrate import n_opt
    assert n_opt() == 1071
    assert n_opt(0.3583) < n_opt()


def test_wav_check_thresholds():
    from genny.analysis import wav_check
    sr = 44100
    t = np.arange(int(0.3 * sr)) / sr
    good = 0.8 * np.sin(2 * np.pi * 440 * t) * np.sin(np.pi * t / t[-1])
    assert wav_check(good, sr) == [] and wav_check(good, sr, "hit") == []
    assert any("peak" in p for p in wav_check(0.45 * good, sr))
    assert any("peak" in p for p in wav_check(np.clip(1.2 * good, -0.96, 0.96), sr))
    gap = good.copy()
    gap[3000:3000 + int(0.121 * sr)] = 0.0
    assert any("gap" in p for p in wav_check(gap, sr))
    assert any("DC" in p for p in wav_check(good + 0.011, sr))
    assert any("ends" in p for p in wav_check(np.concatenate([[0.03], good[1:]]), sr))
    assert wav_check(good, sr, "coin") == [] and any("for explosion" in p for p in wav_check(good, sr, "explosion"))
    assert wav_check(np.concatenate([good, np.zeros(int(0.1 * sr))]), sr, "coin") != []


def test_sway_force_tree_size_uses_moore_maguire_eq9():
    from genny.physical import ambience as amb
    w = np.full(20 * 48000 // 4, 0.6)
    a = amb.sway_force(w, np.random.default_rng(3), tree=(20.0, 15.0, False))
    b = amb.sway_force(w, np.random.default_rng(3), sway_hz=amb.tree_sway_hz(20.0, 15.0, False))
    assert abs(amb.tree_sway_hz(20.0, 15.0, False) - 0.40) < 0.005
    assert np.array_equal(a, b)
    assert not np.array_equal(a, amb.sway_force(w, np.random.default_rng(3), tree=(40.0, 15.0, False)))


def test_footstep_accel_and_incline_move_grf_toward_ball_farnell_20_6_3():
    from genny.physical import footsteps as S

    def amps(**kw):
        _, info = S.render_step("wood", 0, np.random.default_rng(4), return_info=True, **kw)
        h, _, b = info["gait"]["amps"]
        return h, b
    h0, b0 = amps()
    hu, bu = amps(incline=0.3)
    hd, bd = amps(accel=-0.3)
    assert bu / hu > b0 / h0 > bd / hd
    assert np.allclose(amps(accel=0.2, incline=0.1), (hu, bu), rtol=1e-12)
    assert np.array_equal(S.render_step("wood", 0, np.random.default_rng(4)),
                          S.render_step("wood", 0, np.random.default_rng(4), accel=0.0, incline=0.0))


def test_explosion_explosive_scales_energy_by_tnt_equivalent():
    from genny import sfx
    from genny.ordnance import RE
    assert (RE["petn"], RE["anfo"], RE["c4"]) == (1.66, 0.74, 1.34)
    a = sfx.explosion(sr=SR, dur=0.5, energy=1.66, distance=20.0, seed=1)
    b = sfx.explosion(sr=SR, dur=0.5, energy=1.0, distance=20.0, seed=1, explosive="petn")
    assert np.allclose(a, b)
    assert np.array_equal(sfx.explosion(sr=SR, dur=0.5, seed=1), sfx.explosion(sr=SR, dur=0.5, seed=1, explosive="tnt"))


def test_explosion_contact_burst_apparent_yield_kinney_ex_7_6():
    from genny import sfx
    from genny.physics import BURST_YIELD
    assert abs(78 * (BURST_YIELD["rigid"] * 1e6) ** (1 / 3) - 9800) < 50
    assert abs(78 * (BURST_YIELD["crater"] * 1e6) ** (1 / 3) - 8700) < 50
    a = sfx.explosion(sr=SR, dur=0.5, energy=1.5, distance=20.0, seed=1)
    b = sfx.explosion(sr=SR, dur=0.5, energy=1.0, distance=20.0, seed=1, burst="surface")
    assert np.allclose(a, b)


def test_pass_by_wind_fades_in_between_15_and_26_m_s_ice_simulator(monkeypatch):
    from genny import cyber as C
    real = C.O.pink
    def run(kmh, gain):
        monkeypatch.setattr(C.O, "pink", lambda n, s: real(n, s + gain))
        return C.pass_by(sr=SR, dur=1.0, speed=kmh, power="none", seed=2)
    assert np.allclose(run(50.0, 0), run(50.0, 7))
    assert not np.allclose(run(60.0, 0), run(60.0, 7))


def test_police_rc_siren_farnell_p5_log_rc_wave_clip_and_1500_hz_horn():
    from genny import foley as Fo
    u = np.linspace(0.0, 0.5, 101)[:-1]
    w = Fo._rc_wave(u)
    assert abs(w[0] + 1) < 1e-12 and abs(Fo._rc_wave(np.array([0.5]))[0] - 1) < 1e-12
    assert np.all(np.diff(w) > 0) and np.all(np.diff(w, 2) < 0)
    assert np.allclose(Fo._rc_wave(u + 0.5), -w)
    sr = 48000
    y = Fo.siren_wail(sr=sr, kind="police_rc", dur=4.0, echo=0.0)
    N = 4096
    fr = np.fft.rfftfreq(N, 1 / sr)
    k = (fr > 200) & (fr < 1300)
    pk = [fr[k][np.argmax(np.abs(np.fft.rfft(y[i:i + N] * np.hanning(N)))[k])] for i in range(0, len(y) - N - sr // 2, 1024)]
    assert abs(min(pk) - 300) < 2 * sr / N and abs(max(pk) - 1100) < 2 * sr / N


def _peak(y, sr, a, b, lo, hi):
    seg = y[int(a * sr):int(b * sr)]
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    fr = np.fft.rfftfreq(8 * len(seg), 1 / sr)
    k = (fr > lo) & (fr < hi)
    return fr[k][np.argmax(sp[k])]


def test_teleport_ring_and_shield_up_follow_retro_sfx_gen_ranges():
    from genny import foley as Fo
    from genny.analysis import wav_check
    sr = 48000
    for seed in range(3):
        y = Fo.teleport(sr=sr, dur=0.7, style="ring", seed=seed)
        f0 = _peak(y, sr, 0.05, 0.12, 100, 400)
        f1 = _peak(y, sr, 0.58, 0.65, 400, 2500)
        assert 180 * 4.5 ** 0.121 * 0.95 < f0 < 260 * 7 ** 0.121 * 1.05 and 4.5 ** 0.757 * 0.95 < f1 / f0 < 7 ** 0.757 * 1.05
        assert [p for p in wav_check(y, sr, "teleport") if "for teleport" in p] == []
        s = Fo.shield_up(sr=sr, seed=seed)
        assert 320 * 0.99 < _peak(s, sr, 0.3, 0.45, 250, 600) < 520 * 1.02
        assert [p for p in wav_check(s, sr, "shield") if "for shield" in p] == []
    assert np.array_equal(Fo.teleport(sr=sr, seed=1), Fo.teleport(sr=sr, seed=1, style="shepard"))


def test_baschet_tables_ruiz_i_carulla_annexes():
    from genny import friction as Fr
    for k, f in Fr.ORGUES_NOTE.items():
        assert f in Fr.BASCHET_MODES[f"orgues_{k}"]
    assert Fr.BASCHET_MODES["bell_prototype"][:2] == (106.0, 114.0)
    assert Fr.BASCHET_MODES["grande_flamme"] == (94.0, 188.0, 268.0, 360.0, 474.0, 532.0)
    assert all(list(v) == sorted(v) for v in Fr.BASCHET_MODES.values())


def test_layered_ground_hess_1990_eq_6_to_8_limits():
    from genny.physical import space as S
    f = np.array([100.0, 500.0, 2000.0])
    z1 = 0.436 * np.sqrt(906000.0 / f) * (1 + 1j)
    z2 = 0.436 * np.sqrt(20000.0 / f) * (1 + 1j)
    deep = S.layered_impedance(f, 906.0, 0.3, 10.0, 20.0, 0.5)
    thin = S.layered_impedance(f, 906.0, 0.3, 0.0, 20.0, 0.5)
    assert np.allclose(deep, z1) and np.allclose(thin, z2)
    assert np.allclose(S.layered_impedance(f, 300.0, 0.4, 0.05, 300.0, 0.4).real,
                       0.436 * np.sqrt(300000.0 / f))
    mid = S.layered_impedance(f, 906.0, 0.3, 0.01, 20.0, 0.5)
    assert not np.allclose(mid, z1) and not np.allclose(mid, z2)
    g = S.ground_effect(f, 20.0, layer=(906.0, 0.3, 0.01, 20.0, 0.5))
    assert g.shape == f.shape and not np.allclose(g, S.ground_effect(f, 20.0, sigma=906.0))


def test_bearing_contact_angle_and_creep_randall_antoni_and_skf_suffix_b(monkeypatch):
    from genny import cyber as C
    k40 = C.bearing_frequencies(60.0, 9, 7.94, 39.04, 40.0)
    assert abs(k40["bpfo"] - 3.799) < 0.001 and abs(k40["bpfi"] - 5.201) < 0.001
    seen = []
    real = C._impacts
    monkeypatch.setattr(C, "_impacts", lambda n, phase, *a, **kw: (seen.append(phase[-1]), real(n, phase, *a, **kw))[1])
    for kw in ({}, {"angle": 40.0}, {"creep": 1}):
        C.bearing(dur=0.5, rpm=1800.0, state="outer", **kw)
    base = seen[0]
    assert abs(seen[1] / base - k40["bpfo"] / C.bearing_frequencies(60.0, 9, 7.94, 39.04)["bpfo"]) < 1e-9
    assert 0.98 <= seen[2] / base <= 0.99


def test_wet_wood_obataya_minus_6_percent_e_over_rho_plus_24_percent_loss(monkeypatch):
    from genny import contact as K
    got = []
    real = K.ImpactInteractor
    monkeypatch.setattr(K, "ImpactInteractor", lambda body, **kw: (got.append((body.freqs.copy(), body.decays.copy())), real(body, **kw))[1])
    K.impact(material="wood", dur=0.05)
    K.impact(material="wood", dur=0.05, moisture=1.0)
    K.impact(material="steel", dur=0.05, moisture=1.0)
    K.impact(material="steel", dur=0.05)
    assert np.allclose(got[1][0] / got[0][0], math.sqrt(0.94)) and np.allclose(got[1][1] / got[0][1], 1.24)
    assert np.array_equal(got[2][0], got[3][0]) and np.array_equal(got[2][1], got[3][1])


def test_wet_wood_keeps_its_loss_through_a_material_angle(monkeypatch):
    from genny import contact as K
    got = []
    real = K.ImpactInteractor
    monkeypatch.setattr(K, "ImpactInteractor", lambda body, **kw: (got.append((body.freqs.copy(), body.decays.copy())), real(body, **kw))[1])
    K.impact(material="wood", dur=0.05, material_angle=120.0)
    K.impact(material="wood", dur=0.05, material_angle=120.0, moisture=1.0)
    assert np.allclose(got[1][0] / got[0][0], math.sqrt(0.94)) and np.allclose(got[1][1] / got[0][1], 1.24)


def test_reflected_blast_is_at_least_twice_the_incident_one():
    from genny.physics import _blast_fit, blast_reflected
    for z in (5.0, 15.0, 20.0, 30.0, 40.0):
        assert blast_reflected(1.0, z)[0] >= 2.0 * _blast_fit("ps", z) * 1e3 * (1 - 1e-12)


def test_emg_fit_sterling_2019_least_squares_overestimate_damping():
    from genny import calibrate as Cal
    rng = np.random.default_rng(1)
    f = np.sort(rng.uniform(200.0, 8000.0, 752))
    a, b = 35.0, 4.6e-7
    true = 0.5 * (a + b * (2 * np.pi * f) ** 2)
    c = true + rng.normal(0.0, 2.0, f.size) + rng.exponential(10.0, f.size)
    a1, b1 = Cal.rayleigh_fit(f, c)
    a2, b2, sg, eta = Cal.emg_fit(f, c)
    ls = 0.5 * (a1 + b1 * (2 * np.pi * f) ** 2)
    em = 0.5 * (a2 + b2 * (2 * np.pi * f) ** 2)
    assert np.mean(ls - true) > 8.0
    assert abs(np.mean(em - true)) < 1.0 and abs(sg - 2.0) < 0.5 and abs(eta - 0.1) < 0.02


# ---------------------------------------------------------------- phase 5: physics, klang, foley, contact (T_Transverse)
SR5 = 48000


def test_building_period_is_kinney_h_over_175_and_breadth_rule():
    from genny import physics as Ph
    assert abs(Ph.natural_period(175 * 0.3048) - 1.0) < 1e-12                     # T = H/175, H in ft
    assert abs(Ph.natural_period(100 * 0.3048, 25 * 0.3048) - 0.05 * 100 / 5.0) < 1e-12   # 0.05 H / sqrt(B)


def test_potential_collision_conserves_energy_to_machine_accuracy_torin():
    from genny import physics as Ph
    w, en = Ph.potential_collision(0.03, 2.0, 1e8, 1.5, sr=SR5)
    assert np.ptp(en[2:]) / en[2:].mean() < 1e-10
    assert abs(en[1] - 0.5 * 0.03 * 2.0 ** 2) < 1e-12                              # all of it was the mallet's
    assert np.abs(w).max() > 0 and np.isfinite(w).all()


def test_fireball_rises_in_100_ms_and_dies_over_8_s_farnell():
    from genny import physics as Ph
    y = Ph.fireball(sr=SR5, seed=1)
    assert abs(len(y) / SR5 - 8.1) < 0.01
    env = np.array([np.sqrt(np.mean(y[i:i + SR5 // 10] ** 2)) for i in range(0, len(y) - SR5 // 10, SR5 // 10)])
    assert np.argmax(env) <= 6 and env[-1] < 0.01 * env.max()                      # full body early, quartic tail
    sp = np.abs(np.fft.rfft(y[:SR5]))
    f = np.fft.rfftfreq(SR5, 1 / SR5)
    assert f[np.argmax(sp)] < 160.0                                                # low-passed 40..140 Hz


def test_whip_follows_burger_then_the_measured_supersonic_ramp():
    from genny import foley as Fo
    t, u = Fo.whip_speed(3.2)
    mu = 0.097 / 0.37e-3
    assert abs(u[0] * math.sqrt(mu + 1.0) - 744.0) < 1e-6                          # Vmax = U sqrt(M/m + 1)
    i = int(np.searchsorted(u, 343.0))
    assert abs(t[-1] - t[i] - 850e-6) < 2e-5 and abs(u[-1] - 744.0) < 1e-6
    x = 3.2 - np.concatenate([[0.0], np.cumsum(0.5 * (u[1:i] + u[:i - 1]) * np.diff(t[:i]))])
    v = u[0] * np.sqrt((mu + 1.0) / (mu * (3.2 + x) / 6.4 + 1.0))                  # V(x) of Buerger
    assert np.max(np.abs(v - u[:i]) / u[:i]) < 1e-3
    for s in (1.0, 2.0):                                                            # crack fired at the peak
        y = Fo.whip(sr=SR5, dur=0.08, size=s, echo=0.0)
        d = np.abs(np.diff(y, 2))
        assert abs(np.argmax(d > 0.3 * d.max()) / SR5 - Fo.whip_speed(3.2 * s)[0][-1]) < 1.5e-3


def test_square_tubes_measured_partials_and_rayleigh_chime_decay():
    from genny import foley as Fo
    from genny.analysis import detect_clicks
    y = Fo.wind_chime(587.4, 1.0, sr=SR5, tube="square")
    sp = np.abs(np.fft.rfft(y[:SR5] * np.hanning(SR5)))
    f = np.fft.rfftfreq(SR5, 1 / SR5)
    for r in (1.0, 2.67, 5.04, 8.0):
        b = (f > 587.4 * r * 0.97) & (f < 587.4 * r * 1.03)
        assert abs(f[b][np.argmax(sp[b])] / (587.4 * r) - 1.0) < 0.004
    assert abs(float(Fo.chime_decay(587.4, "rayleigh")) - 6.9078 / 0.68) < 0.2     # d = 0.68 /s, T60 ~ 10 s
    assert abs(float(Fo.chime_decay(1174.8, "rayleigh")) * 4 - float(Fo.chime_decay(587.4, "rayleigh"))) < 1e-6
    z = Fo.wind_chime(587.4, 3.0, sr=SR5, law="rayleigh")
    e = lambda a: np.sqrt(np.mean(z[int(a * SR5):int((a + 0.2) * SR5)] ** 2))     # noqa: E731
    assert abs(20 * np.log10(e(0.3) / e(2.3)) - 60 * 2.0 / 10.14) < 1.5            # 11.8 dB in 2 s
    assert len(detect_clicks(y)) == 0
    w = Fo.wind_chimes(sr=SR5, scale="measured", tubes=4, seed=2)
    sp = np.abs(np.fft.rfft(w[:SR5]))
    assert min(abs(np.fft.rfftfreq(SR5, 1 / SR5)[np.argmax(sp)] - np.array(Fo.CHIME_OBRIEN))) < 3.0


def test_pulse_dial_is_n_pulses_of_40_ms_every_100_ms_farnell():
    from genny import foley as Fo
    for d, k in ((7, 7), (0, 10)):
        y = Fo.pulse_dial(sr=SR5, digit=d, line=False)
        dy = np.diff(y)
        on = np.flatnonzero(dy > 0.3 * dy.max())
        off = np.flatnonzero(dy < 0.3 * dy.min())
        assert len(off) == k and len(on) >= k - 1                                 # the first rise is under the fade-in
        assert np.allclose(np.diff(off) / SR5, 0.1, atol=1e-3) and np.allclose((off[1:] - on[-(k - 1):]) / SR5, 0.04, atol=1e-3)
    z = Fo.pulse_dial(sr=SR5, digit=3)
    sp = np.abs(np.fft.rfft(z))
    f = np.fft.rfftfreq(len(z), 1 / SR5)
    assert abs(f[np.argmax(sp)] - 400.0) < 50.0 and np.isfinite(Fo.phone_tone(sr=SR5, line=True)).all()


def test_explosion_layers_follow_the_farnell_patch_timings():
    from genny import foley as Fo
    rng = np.random.default_rng(0)
    for spread in (0.0, 0.349, 1.0):
        ev = Fo.blast_times(spread, rng)
        assert abs(ev[0][0] - (100 * spread + 5) * 1e-3) < 1e-12
        assert all(0 <= b[0] - a[0] < max(0.3 * spread, 0.001) + 1e-12 for a, b in zip(ev, ev[1:]))
        assert 8 / 200 <= ev[0][1] <= 15 / 200 and 15 / 200 <= ev[1][1] <= 24 / 200 and 25 / 200 <= ev[2][1] <= 29 / 200
    y = Fo.explosion_layers(sr=SR5)
    assert np.isfinite(y).all() and abs(len(y) / SR5 - (0.5 * 1400 / 6300 + 8.0 + 0.8)) < 0.01
    env = [np.sqrt(np.mean(y[i:i + SR5 // 2] ** 2)) for i in range(0, len(y) - SR5 // 2, SR5 // 2)]
    assert env[-1] < 0.1 * max(env)


def test_scrape_stroke_normal_force_is_agarwal_eq_8_9():
    from genny import contact as C
    m, L, w, ta, mu = 0.02, 0.05, 2 * math.pi, math.tan(math.radians(30)), 0.6
    n = C.stroke_force(np.array([0.0, 0.5]), 1.0, L, 30.0, mu, m)
    a = w * w * m * L * ta / (1 - mu * ta)
    assert abs(n[0] - (m * 9.81 + a)) < 1e-12 and abs(n[1] - (m * 9.81 - a)) < 1e-12
    y = C.scrape(sr=SR5, dur=4.0, stroke=1.0, stroke_len=L)
    h = int(0.04 * SR5)
    seg = lambda c: np.concatenate([y[int((k + c) * SR5) - h:int((k + c) * SR5) + h] for k in range(4)])   # noqa: E731
    near = np.concatenate([seg(0.125), seg(0.875)])
    far = np.concatenate([seg(0.375), seg(0.625)])
    assert np.sqrt(np.mean(near ** 2) / np.mean(far ** 2)) > 1.1                  # pressing harder at the near end
    assert np.array_equal(C.scrape(sr=SR5, dur=0.3), C.scrape(sr=SR5, dur=0.3, stroke=0.0))


def test_modal_engine_misfire_per_cycle_and_stroke_gain_curves():
    from genny import contact as C
    n, rpm, p = SR5 * 8, 6000.0, 0.01
    cyc = np.floor(np.cumsum(np.full(n, rpm / 120.0)) / SR5).astype(int)
    f = C.engine_force(np.random.default_rng(0), n, SR5, rpm=rpm, intake=0.0, exhaust=0.0, misfire=p)
    ks = range(1, cyc[-1])
    miss = np.mean([np.all(f[cyc == k] == 0) for k in ks])
    pc = 1 - (1 - p) ** (120.0 / rpm / C.ENGINE_BLOCK)                  # R06: p per block -> p per cycle
    assert abs(miss - pc) < 4 * math.sqrt(pc * (1 - pc) / len(ks)) and miss > 0
    g = C.engine_force(np.random.default_rng(0), n, SR5, rpm=rpm, misfire=p)
    assert all(np.any(g[cyc == k] != 0) for k in ks)                   # intake and exhaust go on
    a = C.engine_force(np.random.default_rng(1), n, SR5, intake=0.5)
    b = C.engine_force(np.random.default_rng(1), n, SR5, intake=[0.5, 0.5])
    assert np.allclose(a, b)
    c = C.engine_force(np.random.default_rng(1), n, SR5, combustion=[0.0, 0.0])
    assert np.sum(c * c) < np.sum(a * a) and not np.allclose(a, c)
    assert np.isfinite(C.modal_engine(sr=SR5, misfire=0.02, exhaust=[0.2, 1.0])).all()


def test_free_disc_and_tube_fundamentals():
    from genny import contact as C
    # Ducceschi & Touze: a 0.4 m, h 1 mm steel 7.7 Hz with their omega 5.093 (nu = 0.38); steel here has nu = 0.3
    k = C.free_disc_omega((2, 0), 0.3) / 5.093
    assert abs(C.free_disc_freq(0.8, 1e-3, "steel") - 7.7 * k) < 0.05 * k
    assert abs(C.free_disc_freq(0.4, 0.5e-3, "steel") - 15.5 * k) < 0.1 * k
    f = C.fundamental("bar", 0.505, "aluminium", tube=(0.0125, 0.0115))
    assert abs(f / 585.8 - 1.0) < 0.03                                             # measured chime D3
    assert C.fundamental("bar", 0.5, "steel") == C.fundamental("bar", 0.5, "steel", tube=None)


def test_rolling_resistance_tabor_and_loss_tangent():
    from genny import contact as C
    assert abs(C.rolling_resistance(1e-3, 0.05, 0.1) - 2 * 0.26 * 1e-3 / (3 * math.pi * 0.05)) < 1e-15
    assert abs(C.rolling_resistance(1e-3, 0.05, 0.1, "sphere") - 0.26 * 3 / 16 * 1e-3 / 0.05) < 1e-15
    w = np.logspace(-2, 2, 4001)
    t = C.loss_tangent(w, 1.0, 1.0)
    assert abs(w[np.argmax(t)] - 1 / math.sqrt(2)) < 0.01 and abs(t.max() - 1 / (2 * math.sqrt(2))) < 1e-4
