"""Firearms: the numbers the module claims from its sources, measured on the signal, and every weapon and action renders."""
import numpy as np
import pytest

from genny import arms
from genny.analysis import detect_clicks
from genny.spec import render_spec

SR = 48000


def _env(y, ms=2.0):
    k = max(1, int(ms * 1e-3 * SR))
    return np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))


def _onset(y, frac=0.2, start=0.0):
    i0 = int(start * SR)
    e = _env(y)[i0:]
    return (i0 + int(np.argmax(e > frac * e.max()))) / SR


def _band(y, lo, hi):
    s = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    s[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(s, len(y))


def _centroid(u):
    s = np.abs(np.fft.rfft(u)) ** 2
    return np.sum(np.fft.rfftfreq(len(u), 1 / SR) * s) / np.sum(s)


def _period(y, lo, hi):
    """Lag in s of the strongest repetition of the envelope between lo and hi."""
    e = _env(y, 1.0)
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    a, b = int(lo * SR), int(hi * SR)
    return (a + int(np.argmax(ac[a:b]))) / SR


def test_n_wave_matches_spec_table():
    for cal, pa, us in (("308", 1207, 132), ("556", 996, 106)):
        d, l, v, _, _ = arms.CALIBERS[cal]
        p, t = arms.n_wave_pa(v, d, l, 1.0)
        assert abs(p - pa) / pa < 0.1 and abs(t * 1e6 - us) / us < 0.1, (cal, p, t)


def test_no_crack_when_subsonic_or_behind_the_cone():
    assert arms.n_wave_pa(300.0, 9.0, 15.0, 1.0) == (0.0, 0.0)
    assert arms.shock_arrival(300.0, 10.0, 1.0) is None
    assert arms.shock_arrival(900.0, -1.0, 1.0) is None
    assert arms.shock_arrival(900.0, 0.1, 5.0) is None
    assert arms.shock_arrival(900.0, 50.0, 5.0) > 50.0 / 900.0


def test_directivity_is_quieter_behind():
    assert arms._blast_gain(15.0, 0.0) == 1.0
    assert abs(20 * np.log10(arms._blast_gain(15.0, 180.0)) + 15.0) < 1e-9
    assert abs(20 * np.log10(arms._blast_gain(30.0, 90.0)) + 11.1) < 1e-9     # AR15 peaks: -11.1 dB at 90 of -30 behind
    assert abs(20 * np.log10(arms._blast_gain(30.0, 165.0)) + 26.8) < 1e-9


def test_drag_length_from_mass_and_area():
    assert 800.0 < arms.drag_length("556") < 900.0              # 4.0 g, 5.7 mm: 853 m, 948 -> 371 m/s at 800 m
    assert arms.drag_length("50bmg") > 1.8 * arms.drag_length("556")


def test_case_modes_match_spec():
    assert np.allclose(arms.shell_modes(9.93 / 2, 0.35)[:2], [6611, 18700], rtol=0.02)
    assert np.allclose(arms.shell_modes(11.9 / 2, 0.35)[:2], [4604, 13021], rtol=0.02)


def test_n_wave_slope_is_whithams_shape_free_rate():
    # Pmax and T together fall at 0.2 (1 - 1/M^2)^(1/2) / r atm per ms whatever the bullet
    for cal in ("556", "308", "50bmg"):
        d, l, v, _, _ = arms.CALIBERS[cal]
        for b in (1.0, 5.0, 30.0):
            p, t = arms.n_wave_pa(v, d, l, b)
            m = v / arms.C_AIR
            rate = 2 * p / arms.P0 / (t * 1e3)
            assert abs(rate / (0.2 * (1 - 1 / m ** 2) ** 0.5 / b) - 1.0) < 0.05 * (arms.C_AIR / 340.0) + 0.02


def test_shell_modes_rayleigh_and_love():
    E, rho, nu, R, h = 110.3e9, 8530.0, 0.31, 4.79e-3, 0.30e-3
    n = np.array([2.0, 3.0, 4.0])
    lam = h / R / np.sqrt(12) * n * (n * n - 1) / np.sqrt(1 + n * n)      # inextensional, h << R
    f = lam / (2 * np.pi * R) * np.sqrt(E / (rho * (1 - nu * nu)))
    fm = arms.shell_modes(4.79, 0.30, length_mm=44.7)
    assert len(fm) == 6 and np.allclose(fm[:3], f, rtol=0.01)
    q = (R / 44.7e-3) ** 2 / n ** 2
    assert np.allclose(fm[3:] / fm[:3], np.sqrt((1 + 24 * (1 - nu) * q) / (1 + 12 * q / (n * n + 1))), rtol=1e-9)
    assert 1.015 < fm[3] / fm[0] < 1.025                                   # the n = 2 pair of a 5.56 case, 2.0 % apart


def test_casing_physical_rings_on_both_mode_sets():
    from scipy.signal import welch
    y = arms.casing_physical(SR, "762", "concrete", 1.0, 1, 0.0, False, 3)
    f, p = welch(y, SR, nperseg=1 << 15)
    fm = arms.shell_modes(11.9 / 2, 0.35, length_mm=51.2)
    floor = np.median(p[(f > 3000) & (f < 6000)])
    for x in (fm[0], fm[3]):
        assert p[np.argmin(abs(f - x))] > 30 * floor


@pytest.mark.parametrize("weapon", arms.WEAPONS)
def test_every_weapon_renders(weapon):
    y = arms.firearm(render="pressure", sr=22050, weapon=weapon, shots=2, level="peak")     # relative level: see test_relative_level_follows_the_measured_peaks
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3 and len(detect_clicks(y[-int(0.05 * 22050):])) == 0


@pytest.mark.parametrize("action", [*arms.MECH, *arms.RELOADS])
def test_every_action_renders(action):
    y = arms.gun_mech(sr=22050, action=action)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3 and len(detect_clicks(y[-int(0.05 * 22050):])) == 0


def test_unknown_names_are_refused():
    with pytest.raises(ValueError):
        arms.firearm(render="pressure", sr=22050, weapon="phaser")
    with pytest.raises(ValueError):
        arms.gun_mech(sr=22050, action="phaser")
    with pytest.raises(ValueError):
        arms.gun_mech(sr=22050, action="mag_drop", floor="lava")


def test_garand_clip_ping_sits_at_2_to_3_khz():
    y = arms._clip_ping(np.random.default_rng(0))
    f = np.fft.rfftfreq(len(y), 1 / arms.P)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert 2000 < f < 3000


def test_bullet_slows_with_range():
    late = lambda y: np.sum(y[int(0.05 * 22050):] ** 2) / np.sum(y ** 2)
    near = arms.bullet_pass(render="pressure", sr=22050, caliber="556", miss=3.0, spin_tone=False)
    far = arms.bullet_pass(render="pressure", sr=22050, caliber="556", miss=3.0, range=2000.0, spin_tone=False)
    assert late(near) < 0.01 < 0.2 < late(far)      # a crack is over in ms; subsonic, it whizzes on


def test_suppressed_weapon_is_quieter_than_its_open_twin():
    a = arms._shot("smg_mp5sd", np.random.default_rng(0), 10.0, 30.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.07)
    b = arms._shot("smg_mp5", np.random.default_rng(0), 10.0, 30.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.07)
    assert np.sqrt(np.mean(a ** 2)) < 0.4 * np.sqrt(np.mean(b ** 2))




def test_first_pop_is_3_db_then_half():
    kw = dict(sr=SR, weapon="smg_mp5sd", shots=3, rate=4.0, action=0.0, echo=0.0, seed=2)
    a, b = arms.firearm(render="pressure", first_pop=3.0, **kw), arms.firearm(render="pressure", first_pop=0.0, **kw)
    k = lambda y, i: np.abs(y[int(i * 0.25 * SR) + 200:int((i + 0.8) * 0.25 * SR)]).max()
    r = [20 * np.log10(k(a, i) / k(b, i)) for i in range(3)]
    assert abs(r[0] - r[2] - 3.0) < 0.5 and abs(r[1] - r[2] - 1.5) < 0.5




def test_garand_pings_after_the_eighth_round_at_its_own_cadence():
    y = arms.firearm(render="pressure", sr=SR, weapon="m1_garand", shots=8, action=0.0, crack=0.0, echo=0.0, seed=3)
    assert 1.2 < _onset(y, 0.3, 0.5) < 1.5             # 40-50 rounds/min = 1.2-1.5 s
    gap = 1.0 / 0.75
    win = lambda k: y[int((k * gap + 0.05) * SR):int((k * gap + 0.25) * SR)]
    assert np.abs(_band(win(7), 2000, 5000)).max() > 2.0 * np.abs(_band(win(6), 2000, 5000)).max()


def test_bolt_cycle_is_700_ms_open_extract_close():
    last = max(t + max(e[1] for e in arms.MECH[a]) * 1e-3 for a, t in arms.RELOADS["bolt_cycle"])
    assert abs(last - 0.7) < 0.01
    e = _env(arms.gun_mech(sr=SR, action="bolt_cycle", seed=1), 2.0)
    for t in (0.0, 0.16, 0.2, 0.66):      # lift, stop, extraction, bolt home
        assert e[int(t * SR):int((t + 0.03) * SR)].max() > 0.02


def test_musket_reload_takes_at_least_20_s_with_its_steps():
    y = arms.gun_mech(sr=SR, action="reload_musket", seed=1)
    e = _env(y, 10.0)
    assert len(y) / SR >= 20.0 and e[int(20.0 * SR):int(20.2 * SR)].max() > 0.01
    for t in (1.5, 4.5, 6.5, 8.0, 10.0, 12.5, 13.3, 14.1, 16.0):      # powder, wad, ball, rod, rod, blows, rod out
        assert e[int(t * SR):int((t + 0.5) * SR)].max() > 0.002, t


def test_speed_keeps_the_gaps_inside_a_cluster():
    at = arms._cluster_times(arms.MECH["slide_rack"], 0.5)
    assert at[280] == 140.0 and at[290] == 150.0       # 10 ms stay 10 ms, only the cluster start moved
    assert arms._cluster_times(arms.MECH["slide_rack"], 1.0)[290] == 290.0


@pytest.mark.parametrize("floor", ["concrete", "wood", "metal", "tile", "dirt", "grass"])
def test_mag_drop_lands_400_ms_later_on_any_floor(floor):
    y = arms.gun_mech(sr=SR, action="mag_drop", floor=floor, mag="steel", seed=1)
    assert 0.38 < _onset(y, 0.3, 0.2) < 0.45


def test_soft_floor_hardly_lets_the_magazine_bounce():
    second = lambda f: (lambda h: np.abs(h[int(0.19 * arms.P):int(0.3 * arms.P)]).max() / np.abs(h).max())(arms._floor_hit("plastic", f, 1))
    assert second("grass") < 0.7 * second("concrete")      # restitution 0.1 against 0.5


def test_revolver_eject_six_cases_15_ms_apart():
    ev = arms._mech_extras("revolver_eject", "revolver", 1.0, 0, "concrete", "plastic", 6)
    assert len(ev) == 6 and np.allclose(np.diff([t for _, t, _ in ev]), 0.015)
    y = arms.gun_mech(sr=SR, action="revolver_eject", seed=1)
    assert abs(_onset(y, 0.3, 0.1) - 0.35) < 0.03      # 30 ms of slide, 0.32 s of fall


def test_jam_clear_two_blows_then_the_rack_then_the_case():
    e = _env(arms.gun_mech(sr=SR, action="jam_clear", seed=1), 2.0)
    for t in (0.3, 0.5, 0.8, 1.2):
        i = int(t * SR)
        assert e[i:i + int(0.05 * SR)].max() > 2.0 * e[i - int(0.02 * SR):i - int(0.002 * SR)].max(), t


@pytest.mark.parametrize("weapon,gap", [("lmg", 0.0706), ("hmg", 0.115)])
def test_links_at_the_weapons_rhythm(weapon, gap):
    y = arms.gun_mech(sr=SR, action="link_eject", weapon=weapon, rounds=12, seed=1)
    assert abs(_period(y[:int(12 * gap * SR)], 0.6 * gap, 1.5 * gap) - gap) < 0.004


def test_belt_rattle_is_chain_contacts_not_one_tick():
    y = arms.gun_mech(sr=SR, action="belt_rattle", seed=1)
    assert len(y) / SR >= 1.0 and np.count_nonzero(_env(y, 5.0) > 0.02) > 0.2 * SR


def test_spin_tone_is_at_speed_over_twist_at_minus_30_db():
    on = arms.bullet_pass(render="pressure", sr=SR, caliber="556", speed=915.0, miss=3.0, spin_tone=True, seed=1)
    off = arms.bullet_pass(render="pressure", sr=SR, caliber="556", speed=915.0, miss=3.0, spin_tone=False, seed=1)
    k = np.abs(np.fft.rfft(on[:int(0.015 * SR)], SR))
    f = np.fft.rfftfreq(SR, 1 / SR)
    m = (f > 3500) & (f < 7000)
    assert abs(f[m][np.argmax(k[m])] - 5083) < 120
    d = on - off * (np.abs(on).max() / np.abs(off).max())
    line = np.abs(_band(d, 4500, 5700)).max() / np.abs(on).max()
    assert 10 ** (-36 / 20) < line < 10 ** (-24 / 20)


def test_range_goes_from_crack_through_both_to_whizz():
    crack = lambda y: np.abs(y[:int(0.002 * SR)]).max()
    whiz = lambda y: np.sqrt(np.mean(y[int(0.1 * SR):int(0.4 * SR)] ** 2))
    near, mid, far = (arms.bullet_pass(render="pressure", sr=SR, caliber="556", range=r, spin_tone=False, seed=1) for r in (0.0, 800.0, 2000.0))
    assert crack(near) > 0.5 and whiz(near) < 1e-3
    assert crack(mid) > 0.2 and whiz(mid) > 1e-3       # 349 m/s: a thin N-wave and the whizz (P5.6)
    assert whiz(far) > 1e-3 and crack(far) < crack(near)


def test_a_burst_over_18_rounds_per_second_has_the_rate_as_its_pitch():
    y = arms.firearm(render="pressure", sr=SR, weapon="smg_mp5", shots=30, rate=25.0, echo=0.0, crack=0.0, seed=1)
    assert abs(_period(y, 0.02, 0.06) - 0.04) < 0.003


def test_minigun_rate_is_on_the_signal():
    y = arms.minigun(render="pressure", sr=SR, dur=1.0, rate=50.0, spin=0.1, seed=1)
    assert abs(_period(y[int(0.2 * SR):int(1.0 * SR)], 0.012, 0.03) - 0.02) < 0.002


def test_weight_lengthens_the_shot():
    e = lambda w: np.sum(arms.firearm(render="pressure", sr=SR, weapon="ar15", weight=w, crack=0.0, echo=0.0, action=0.0, seed=1)[int(0.02 * SR):] ** 2)
    assert e(100.0) > 1.5 * e(0.0)




def test_detonation_to_muzzle_is_barrel_over_speed():
    assert 0.3e-3 < arms.BARREL_MM["ar15"] * 1e-3 / 948.0 < 2e-3
    assert abs(arms.BARREL_MM["hmg_m2"] * 1e-3 / 930.0 - 1.23e-3) < 1e-4


def test_impact_force_length_depends_on_calibre_and_speed():
    a = arms._hit("steel", 0.2, 10.0, 40, 255.0, 11.5, np.random.default_rng(0))     # 90 us
    b = arms._hit("steel", 0.2, 10.0, 40, 948.0, 5.7, np.random.default_rng(0))      # 12 us: one sample
    assert np.isfinite(a).all() and _centroid(a) < _centroid(b)


def test_recipes_render_through_the_spec():
    y, sr = render_spec(dict(arms.RECIPES["musket_volley"]))
    y = y if y.ndim == 1 else y.mean(axis=1)
    k = int(0.002 * sr)
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))
    onsets = np.flatnonzero(np.diff((e > 0.2 * e.max()).astype(int)) == 1) / sr
    assert np.isfinite(y).all() and onsets[-1] - onsets[0] > 0.3      # spread 0.3-1 s
    dry = arms.firearm(render="pressure", sr=sr, weapon="pistol_9mm", distance=5.0, echo=0.0)
    wet, s2 = render_spec(dict(arms.RECIPES["gunshot_indoor"]))
    wet = wet if wet.ndim == 1 else wet.mean(axis=1)
    late = lambda u: np.sum(u[int(0.2 * s2):] ** 2) / np.sum(u ** 2)
    assert late(wet) > 5 * late(dry)


def _dry(weapon, **kw):
    return arms.firearm(render="pressure", sr=SR, weapon=weapon, azimuth=0.0, crack=0.0, action=0.0, seed=1, **kw)




def test_peak_level_is_the_old_contract_and_unknown_level_is_refused():
    for w in ("pistol_22", "sniper_50"):
        assert abs(np.abs(arms.firearm(render="pressure", sr=SR, weapon=w, level="peak")).max() - 0.9) < 1e-6
    with pytest.raises(ValueError):
        arms.firearm(render="pressure", sr=22050, level="loud")




def test_the_energy_of_a_shot_grows_with_its_charge():
    rms = [np.sqrt(np.mean(_dry(w) ** 2)) for w in ("pistol_22", "pistol_9mm", "ar15", "sniper_50")]
    assert rms == sorted(rms) and rms[3] > 30 * rms[0]


def test_a_longer_blast_is_darker():
    c = {w: _centroid(_dry(w)) for w in ("pistol_22", "ar15", "sniper_308", "sniper_50")}
    assert c["sniper_50"] < c["sniper_308"] < c["ar15"] < c["pistol_22"]


def test_friedlander_has_the_measured_length_and_a_longer_negative_phase():
    a = 1.5e-3 / 2.28                                  # as _body_blast sets it for a 1.5 ms blast
    y = arms._friedlander(a, int(40 * a * arms.P))
    e = np.cumsum(y ** 2) / np.sum(y ** 2)
    t93 = np.argmax(e > 0.93) / arms.P
    assert 0.5e-3 < t93 < 3e-3 and abs(t93 - 1.5e-3) < 1e-4      # 93 % of the energy: 0.5-3 ms, NIJ Fig. 6
    neg = np.flatnonzero(y < 0.01 * y.min())
    assert neg[-1] / arms.P - a > 2 * a                # "at least twice" (Kinney & Graham)


def test_echo_is_off_by_default():
    kw = dict(sr=22050, weapon="ar15", seed=1)
    assert np.array_equal(arms.firearm(render="pressure", **kw), arms.firearm(render="pressure", echo=0.0, **kw))
    assert not np.array_equal(arms.firearm(render="pressure", **kw), arms.firearm(render="pressure", echo=0.3, **kw))














def test_impact_peak_mode_keeps_the_old_level():
    assert abs(np.abs(arms.bullet_impact(sr=SR, caliber="22lr", level="peak")).max() - 0.85) < 1e-6


def test_impact_with_a_heavier_bullet_rings_lower_on_metal():
    a = arms.bullet_impact(sr=SR, material="metal", caliber="556", debris=0.0, seed=1)
    b = arms.bullet_impact(sr=SR, material="metal", caliber="50bmg", debris=0.0, seed=1)
    assert _centroid(b) < 0.92 * _centroid(a)












def test_casing_ricochet_soft_floor_hardly_rings_and_count_adds_cases():
    hard = arms.casing_ricochet(sr=SR, floor="concrete", seed=1)
    soft = arms.casing_ricochet(sr=SR, floor="grass", seed=1)
    assert len(soft) < 0.5 * len(hard) and _centroid(soft) < _centroid(hard)
    assert len(arms.casing_ricochet(sr=22050, caliber="105mm", seed=1)) < 8 * 22050
    one = arms.casing_ricochet(sr=22050, seed=2)
    many = arms.casing_ricochet(sr=22050, count=4, seed=2)
    assert np.isfinite(many).all() and len(many) >= len(one) and np.sqrt(np.mean(many ** 2)) > np.sqrt(np.mean(one ** 2))
    with pytest.raises(ValueError):
        arms.casing_ricochet(sr=22050, caliber="phaser")


def test_ricochet_is_deterministic_at_a_given_seed():
    assert np.array_equal(arms.ricochet(sr=22050, seed=5), arms.ricochet(sr=22050, seed=5))


def _pressure(weapon):
    y = arms._shot(weapon, np.random.default_rng(1), 3.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, 0.1)
    y = y / np.abs(y).max()
    return y[int(np.argmax(np.abs(y) > 0.05)):]








def test_pov_keeps_the_body():
    y = arms.firearm(render="pressure", sr=SR, weapon="ar15", pov=True, action=0.0, seed=1)
    e = np.cumsum(y ** 2) / np.sum(y ** 2)
    assert np.argmax(e > 0.93) / SR > 0.02


