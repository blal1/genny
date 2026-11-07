"""Bows, slings, siege engines, hand weapons: the numbers the module claims from its sources, measured on the signal, and every variant renders without a click."""
import math

import numpy as np
import pytest

from genny import melee as M
from genny.analysis import detect_clicks
from genny.foley import whip
from genny.physical import modal
from genny.sfx import REGISTRY

SR = 48000
C = 343.0

VARIANTS = [("crossbow", dict(kind="modern")), ("crossbow", dict(kind="arbalest", crank=3.0)), ("crossbow", dict(kind="reload", crank=3.0)),
            ("sling", dict(kind="whirl")), ("sling", dict(kind="release")), ("sling", dict(kind="whistling")), ("slingshot_weapon", {}),
            ("catapult", dict(kind="trebuchet")), ("catapult", dict(kind="mangonel")), ("catapult", dict(kind="ballista")), ("siege_impact", {}),
            *[("melee", dict(kind=k)) for k in M.MELEE_KINDS], ("melee", dict(kind="knife_slash", flat=1, target="wood")), ("melee", dict(kind="axe_chop", flat=1)),
            *[("knife", dict(action=a)) for a in M.KNIFE_ACTIONS],
            *[("armor", dict(kind=k, motion=m)) for k in M.ARMOR_KINDS for m in ("walk", "run", "shift")],
            *[("arrow", dict(kind=k)) for k in M.ARROW_KINDS], ("grab", dict(kind="arm")), ("grab", dict(kind="shirt")),
            *[("spear", dict(kind=k)) for k in M.SPEAR_KINDS], ("spear", dict(kind="parade", shield=1)),
            ("gouge", dict(mode="rubbing")), ("gouge", dict(mode="wear")), ("gouge", dict(mode="cutting"))]


def _env(y, ms=2.0):
    k = max(1, int(ms * 1e-3 * SR))
    return np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))


def _peak(y, lo, hi):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def _centroid(u):
    s = np.abs(np.fft.rfft(u)) ** 2
    return np.sum(np.fft.rfftfreq(len(u), 1 / SR) * s) / np.sum(s)


def _period(y, lo, hi, ms=5.0):
    e = _env(y, ms)
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    a, b = int(lo * SR), int(hi * SR)
    return (a + int(np.argmax(ac[a:b]))) / SR


def _onsets(y, frac=0.25, gap=0.03):
    e = _env(y, 1.0)
    above = e > frac * e.max()
    idx = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    out = []
    for i in idx:
        if not out or (i - out[-1]) / SR > gap:
            out.append(int(i))
    return np.array(out) / SR


@pytest.mark.parametrize("name,kw", VARIANTS, ids=lambda v: v if isinstance(v, str) else "-".join(map(str, v.values())))
def test_every_variant_renders_without_a_click_on_the_whole_signal(name, kw):
    for seed in (0, 1, 2):
        y = getattr(M, name)(sr=SR, seed=seed, **kw)
        assert np.isfinite(y).all() and 0.01 < np.abs(y).max() <= 1.2
        assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
        assert len(detect_clicks(y)) == 0, (name, kw, seed, detect_clicks(y)[:5] / SR)


def test_the_detector_still_sees_a_step_in_these_renders():
    """Otherwise the test above proves nothing: a step of 0.1 in a real render must fire."""
    for name, kw in (("melee", dict(kind="axe_chop")), ("armor", dict(kind="cuir"))):
        y = getattr(M, name)(sr=SR, **kw).copy()
        y[len(y) // 3:] += 0.1
        assert len(detect_clicks(y)) > 0


@pytest.mark.parametrize("name,kw", [("melee", dict(kind="axe_swing")), ("crossbow", dict(kind="modern")), ("armor", dict(kind="plates"))])
def test_default_rate_is_clean_too(name, kw):
    y = getattr(M, name)(**kw)
    assert np.isfinite(y).all() and len(detect_clicks(y)) == 0


def test_names_are_registered_and_unknown_kinds_refused():
    assert {"crossbow", "sling", "slingshot_weapon", "catapult", "siege_impact", "melee", "knife", "armor", "arrow", "grab", "spear", "gouge"} <= set(REGISTRY)
    for call in (lambda: M.melee(kind="halberd"), lambda: M.knife(action="peel"), lambda: M.arrow(kind="x"), lambda: M.armor(kind="scale"), lambda: M.armor(motion="fly"), lambda: M.spear(kind="x"), lambda: M.gouge(mode="x")):
        with pytest.raises(ValueError):
            call()


def test_foleys_blade_table_is_left_alone():
    from genny import foley
    assert foley.BLADES == {"sword": "SWD1", "sword2": "SWD2", "golf_club": "PGA", "bat": "MLB", "broom": "BROOM",
                            "wood_sword": "WOODSWD"}


# Johnson: wedge
@pytest.mark.parametrize("alpha,pm", [(30, 2.61), (40, 2.96), (45, 3.15), (50, 3.35), (60, 3.78), (70, 4.22), (80, 4.68), (90, 5.14)])
def test_wedge_pressure_table_6_18(alpha, pm):
    assert abs(M.wedge_pressure(alpha) - pm) < 0.015


def test_blade_force_follows_the_wedge():
    Y = M.yield_stress(3e7)
    assert Y == 1e7                                                     # Y = H / 3 of 6.29
    f = M.blade_force(30.0, 0.05, 0.01, Y)
    assert abs(f - 2.61 * (Y / 2) * 2 * 0.01 * math.tan(math.radians(30)) * 0.05) / f < 0.01
    assert M.blade_force(30.0, 0.05, 0.01, Y, mu=0.2) > f               # pm = pw (1 + mu cot alpha)
    narrow, wide = M.blade_contact(10.0, 1.0, 20.0, 0.05, Y), M.blade_contact(45.0, 1.0, 20.0, 0.05, Y)
    assert narrow[0] > wide[0] and narrow[1] < wide[1]                  # a sharp edge pushes less and for longer


@pytest.mark.parametrize("name", ["knife", "axe"])
def test_wedge_contact_falls_in_the_measured_ranges(name):
    """foley._TARGET_MS (van den Doel): flesh 10-50 ms, wood 1-3 ms. Computed from the wedge alone, not fitted to them."""
    assert 10.0 <= M.contact_ms(name, "flesh") <= 50.0
    assert 1.0 <= M.contact_ms(name, "wood") <= 3.0


def _rise(y):
    """s from the first sound to the top of the 1 ms envelope."""
    e = _env(y, 1.0)
    return (int(np.argmax(e)) - int(np.argmax(e > 0.02 * e.max()))) / SR


def test_contact_time_is_on_the_signal():
    flesh, wood = M.melee(sr=SR, kind="baton_hit", target="flesh"), M.melee(sr=SR, kind="baton_hit", target="wood")
    assert 0.005 <= _rise(flesh) <= 0.05 and _rise(wood) < 0.004


# Johnson: flat
def test_flat_contact_k_is_2_e_star_a():
    es = 1 / ((1 - 0.35 ** 2) / 11e9 + (1 - 0.09) / 200e9)
    assert abs(M.flat_contact(1.0, 0.03, "wood") - math.pi * math.sqrt(1 / (2 * es * 0.03)) * 1e3) < 1e-6
    assert 10.0 <= M.flat_contact(1.35, 0.035, "flesh") <= 50.0
    assert M.flat_contact(1.35, 0.035, "armour") < 0.3                  # van den Doel, steel on steel
    assert M.contact_ms("axe", "wood", 0.3, flat=True) == M.contact_ms("axe", "wood", 1.0, flat=True)     # independent of the speed


def test_flat_is_a_harder_blow_than_the_edge_on_the_signal():
    edge, flat = M.melee(sr=SR, kind="knife_slash", target="wood"), M.melee(sr=SR, kind="knife_slash", target="wood", flat=1)
    a, b = int(np.argmax(np.abs(edge))), int(np.argmax(np.abs(flat)))
    assert _centroid(flat[b:b + SR // 10]) > _centroid(edge[a:a + SR // 10])


# Johnson: ploughing
def test_gouge_q_over_p_is_cot_alpha_on_the_signal():
    a = M.gouge(sr=SR, alpha=20.0, mode="rubbing", seed=3)
    b = M.gouge(sr=SR, alpha=40.0, mode="rubbing", seed=3)
    assert abs(np.std(a) / np.std(b) - (1 / math.tan(math.radians(20))) / (1 / math.tan(math.radians(40)))) < 0.02
    assert abs(M.gouge_force(50.0, 30.0) - 50.0 / math.tan(math.radians(30))) < 1e-9


def test_gouge_three_modes():
    assert M.gouge_mode(30.0, 0.1) == "rubbing" and M.gouge_mode(80.0, 0.3) == "wear"      # mu > cot alpha: the wedge sticks
    assert M.gouge_force(50.0, 80.0, "wear") == 50.0                                        # Q / P towards 1
    crest = lambda y: (lambda e: e.max() / np.median(e[e > 0]))(_env(y[SR // 20:-SR // 20], 5.0))
    rub, wear, cut = (M.gouge(sr=SR, alpha=30.0, mode=m, dur=2.0, seed=1) for m in ("rubbing", "wear", "cutting"))
    assert crest(wear) > 2 * crest(rub)                                 # the prow goes in bursts
    assert crest(wear) > 2 * crest(cut)                                 # the chip does not


# K7 and the slings
def test_whirl_tone_and_rotation():
    y = M.sling(sr=SR, kind="whirl", dur=4.0)
    assert abs(40 / (2 * np.pi) - 6.37) < 0.01 and abs(_period(y, 0.08, 0.3) - 2 * np.pi / 40) < 0.012   # u / r = 6.4 turns/s
    assert abs(_peak(modal.aeolian_tone(40.0, 0.003, 2.0, np.random.default_rng(0), "loq"), 1000, 6000) - 2667) < 100
    z = M.whirl(40.0, 0.003, 40.0, 2.0, np.random.default_rng(0), depth=0.0)
    assert abs(_peak(z, 1000, 6000) - 2667) < 100
    k = SR // 25                                                        # with the rotation: the tone follows u, centred on St u / d
    f = np.array([_peak(y[i:i + k], 1000, 6000) for i in range(0, len(y) - k, k)])
    assert abs(np.median(f) - 2667) < 300 and np.ptp(f) > 800


def test_release_has_no_crack():
    y, w = M.sling(sr=SR, kind="release"), whip(sr=SR)
    rough = lambda z: np.abs(np.diff(z, 2)).max() / np.std(z)
    assert rough(y) < 0.2 * rough(w)
    t = y[int(0.17 * SR):int(0.4 * SR)]                                 # 28 g of lead is 16.8 mm: St u / d = 476 Hz, slowing 20 %
    assert 0.8 * 476 <= _peak(t, 200, 1000) <= 1.05 * 476


def test_whistling_is_the_cavity_mode_going_away():
    """UNRESOLVED GAP, reported in suivi/L03.md: the spec's 1600 Hz is St 0.2 on 5 mm, the cylinder law; the model it names (`cavity_tone`, L = D = W = 5 mm, u = 40) has its depth mode at 9.1 kHz.
    This pins what is rendered, the model's depth mode with the Doppler of a bullet going away; it does not confirm the pitch."""
    _, f = modal.cavity_frequencies(40.0, 0.005, 0.005, 0.005)
    f = float(f) * C / (C + 40.0)
    y = M.sling(sr=SR, kind="whistling", dur=2.0)
    assert abs(_peak(y[SR // 2:SR], 1000, 12000) - f) < 0.03 * f


# crossbow
def test_modern_bolt_tone_2356():
    assert abs(0.2 * 106 / 0.009 - 2356) < 1
    y = M.crossbow(sr=SR, kind="modern", distance=0.0)
    w = y[int(0.03 * SR):int(0.11 * SR)]                                # the bolt in its first 90 ms, slowing 20 % over 0.12 s; string 250 Hz below 2 kHz stops
    assert 0.8 * 2356 <= _peak(w, 1700, 3000) <= 1.03 * 2356


def test_string_stiffer_than_the_bow():
    y = M.crossbow(sr=SR, kind="modern", distance=0.0)
    assert abs(_peak(y[int(0.02 * SR):int(0.4 * SR)], 150, 400) - 250) < 12


def test_windlass_lasts_30_s_at_two_bolts_a_minute():
    y = M.crossbow(sr=SR, kind="arbalest", distance=0.0)
    e = _env(y, 5.0)
    assert 30.0 < len(y) / SR < 33.0
    assert e[:SR // 2].max() > 0 and np.quantile(e[28 * SR:29 * SR], 0.9) > 2 * np.quantile(e[SR:2 * SR], 0.9)      # tension rises, 76 to 340 kg
    assert np.std(y[int(30.2 * SR):int(30.5 * SR)]) > np.std(y[29 * SR:30 * SR])                                     # the release stands out
    assert abs(_period(y[:30 * SR], 0.1, 0.3) - 1 / 6) < 0.01


def test_reload_is_the_windlass_alone():
    y = M.crossbow(sr=SR, kind="reload")
    assert abs(len(y) / SR - 30.0) < 0.3


# slingshot, siege
def test_slingshot_pitch_and_flight():
    y = M.slingshot_weapon(sr=SR, draw=0.0, distance=10.0)
    assert abs(_peak(y[int(0.2 * SR):int(0.34 * SR)], 80, 300) - 140) < 15
    on = _onsets(y[int(0.3 * SR):], 0.3)
    assert abs(on[0] + 0.3 - 10 / 28.27) < 0.03


def test_launch_speed():
    assert abs(M.launch_speed(445.0) - 66.1) < 0.1 and abs(M.launch_speed(80.0) - 28.0) < 0.1


def test_trebuchet_arm_is_33_hz_and_the_counterweight_comes_last():
    assert abs(0.2 * 50 / M.ARM_D - 33.3) < 0.1
    y = M.catapult(sr=SR, kind="trebuchet")
    arm = y[int(2.8 * SR):int(3.3 * SR)]
    assert 25.0 <= _peak(arm, 15, 60) <= 35.0
    assert int(np.argmax(np.abs(y))) > int(3.2 * SR)                    # the thud, not the ropes
    assert np.std(y[int(0.2 * SR):int(1.4 * SR)]) > 0                   # ropes first


def test_mangonel_ropes_then_the_crossbar():
    y = M.catapult(sr=SR, kind="mangonel")
    assert abs(int(np.argmax(np.abs(y))) / SR - 1.15) < 0.1


@pytest.mark.parametrize("pitch", [80.0, 120.0])
def test_ballista_cord(pitch):
    y = M.catapult(sr=SR, kind="ballista", pitch=pitch)
    assert abs(_peak(y[int(0.6 * SR):int(1.2 * SR)], 60, 200) - pitch) < 0.12 * pitch


def test_siege_impact_takes_the_real_mass_and_speed():
    """`contact.impact` would clip them at 50 kg and 30 m/s."""
    light, heavy = M.siege_strike(60.0, 34.0), M.siege_strike(250.0, 66.0)
    assert np.max(np.abs(heavy)) > 1.5 * np.max(np.abs(light))
    a = M.siege_strike(250.0, 34.0)
    b = M.siege_strike(60.0, 34.0)
    assert _centroid(a) < _centroid(b)
    y = M.siege_impact(sr=SR)
    assert len(y) >= 2.5 * SR and np.std(y[SR:2 * SR]) > 0.02 * np.abs(y).max()    # debris after the blow


# swings
@pytest.mark.parametrize("kind,force,lo,hi", [("baton_swing", 0.8, 150.0, 260.0), ("axe_swing", 0.8, 150.0, 260.0)])
def test_swing_tones_follow_st_u_over_d(kind, force, lo, hi):
    """Baton: 35 m/s on 34 mm, 205 Hz. Axe: 30 m/s on 30 mm, 200 Hz. Doppler of the swing, up to 10 %."""
    y = M.melee(sr=SR, kind=kind, force=force)
    assert lo <= _peak(y, 40, 1000) <= hi


def test_sledgehammer_tone_is_50_hz():
    y = M.melee(sr=SR, kind="sledgehammer")
    at = int(np.argmax(np.abs(y)))
    assert 35.0 <= _peak(y[:at - SR // 100], 20, 150) <= 62.0


def test_baton_is_lower_than_a_thin_knife():
    k = M._air("knife", np.random.default_rng(0))
    b = M._air("baton", np.random.default_rng(0))
    assert _centroid(k) > 3 * _centroid(b)


def test_arc_shortens_the_swing():
    assert len(M._air("knife", np.random.default_rng(0), arc=0.5)) < 0.8 * len(M._air("knife", np.random.default_rng(0)))


def test_baton_extends_in_three_clicks():
    for seed in (0, 1, 2):
        e = _env(M.melee(sr=SR, kind="baton_extend", seed=seed), 1.0)
        k = int(0.002 * SR)                                             # a click is a jump of the envelope in 2 ms; the ring after it only falls and beats
        jump = (e[k:] - e[:-k]) > 0.3 * e.max()
        idx = np.flatnonzero(jump[1:] & ~jump[:-1]) + 1
        on = [i for j, i in enumerate(idx) if j == 0 or i - idx[j - 1] > 0.05 * SR]
        assert len(on) == 3, on
        assert all(0.075 <= g / SR <= 0.125 for g in np.diff(on))


def test_axe_chop_tears_then_pulls_free():
    y = M.melee(sr=SR, kind="axe_chop")
    at = int(np.argmax(np.abs(y)))
    late = y[at + int(0.6 * SR):at + int(1.0 * SR)]
    silent = y[at + int(0.55 * SR):at + int(0.62 * SR)]
    assert np.std(late) > 2 * np.std(silent)                            # the pull at .65 s
    assert np.std(y[at + SR // 50:at + SR // 10]) > 0.03                # fibres


def test_mace_and_sledge_are_contact_impact_with_the_mass():
    m = M.melee(sr=SR, kind="mace")
    s = M.melee(sr=SR, kind="sledgehammer")
    assert np.argmax(np.abs(m)) > 0 and np.argmax(np.abs(s)) > 0
    assert M.WEAPONS["mace"]["mass"] < M.WEAPONS["sledge"]["mass"]
    assert 3.6 <= M.WEAPONS["sledge"]["mass"] <= 9.0 and 1.0 <= M.WEAPONS["mace"]["mass"] <= 3.0


def test_throw_spins_then_lands():
    y = M.melee(sr=SR, kind="knife_throw")
    land = 0.6 + 6 / C
    assert abs(int(np.argmax(np.abs(y))) / SR - land) < 0.03
    flight = y[int(0.03 * SR):int(0.57 * SR)]
    assert 1 / 5.0 - 0.03 <= _period(flight, 0.15, 0.4, 10.0) <= 1 / 3.0 + 0.03


def test_flail_chain_before_and_after_the_head():
    y = M.melee(sr=SR, kind="flail")
    assert abs(int(np.argmax(np.abs(y))) / SR - 0.45) < 0.03
    assert np.std(y[int(0.05 * SR):int(0.4 * SR)]) > 0.01
    assert np.std(y[int(0.6 * SR):int(0.95 * SR)]) > 0.01


# knife, arrow, armour, grab, spear
def test_knife_draw_rings_at_the_end():
    y = M.knife(sr=SR, action="degainer")
    assert np.std(y[int(0.452 * SR):int(0.5 * SR)]) > np.std(y[int(0.05 * SR):int(0.15 * SR)])


def test_knife_cut_is_a_continuous_chip():
    y = M.knife(sr=SR, action="couper")
    e = _env(y[SR // 20:-SR // 20], 10.0)
    assert e.min() > 0.1 * e.max()


def test_arrow_flight_is_1750_hz():
    assert abs(0.2 * 70 / 0.008 - 1750) < 1e-6
    y = M.arrow(sr=SR, kind="vol", dur=1.0)
    w = y[int(0.03 * SR):int(0.2 * SR)]
    assert 0.85 * 1750 * C / (C + 70) <= _peak(w, 800, 3000) <= 1.05 * 1750


def test_arrow_stages_are_separate():
    assert len(M.arrow(sr=SR, kind="impact")) < SR
    for spec in ("vol seul", "sortie carquois"):
        assert M.arrow(sr=SR, kind=spec).size > 0                       # the plan's spelling
    q = M.arrow(sr=SR, kind="carquois")
    assert abs(int(np.argmax(_env(q))) / SR - 0.36) < 0.05              # the nock on the rim


def test_armour_keeps_step():
    for kind in ("plates", "maille"):
        y = M.armor(sr=SR, kind=kind, motion="walk", dur=4.0)
        assert abs(_period(y, 0.3, 0.8, 20.0) - 0.5) < 0.05, kind


def test_armour_kinds_differ_as_described():
    p, m, c = (M.armor(sr=SR, kind=k, dur=3.0) for k in M.ARMOR_KINDS)
    assert _centroid(c) < _centroid(p) and _centroid(c) < _centroid(m)  # leather is the dull one
    crest = lambda y: (lambda e: e.max() / np.median(e[e > 0]))(_env(y[SR // 10:-SR // 2], 5.0))
    assert crest(m) < crest(p)                                          # maille is a dense shake, plates separate contacts


def test_grab_slap_is_five_ms():
    y = M.grab(sr=SR, kind="arm")
    assert 0.001 <= _rise(y) <= 0.008
    z = M.grab(sr=SR, kind="shirt")
    assert _rise(z) > _rise(y)


def test_shaft_is_a_bar():
    assert abs(M.shaft_modes(2.0)[0] / M.shaft_modes(3.0)[0] - 2.25) < 0.01
    r = M.shaft_modes(2.5)
    assert abs(r[1] / r[0] - 2.757) < 0.03                              # free bar, 1 : 2.757
    y = M.spear(sr=SR, kind="hampe", length=2.0)[int(0.31 * SR):]
    z = M.spear(sr=SR, kind="hampe", length=3.0)[int(0.31 * SR):]
    assert abs(_peak(y, 30, 1000) / _peak(z, 30, 1000) - M.shaft_modes(2.0)[0] / M.shaft_modes(3.0)[0]) < 0.35


def test_thrust_lands_after_the_swoosh_and_the_shaft_shakes():
    y = M.spear(sr=SR, kind="poussee", target="wood")
    assert abs(int(np.argmax(np.abs(y))) / SR - 0.12) < 0.02
    assert np.std(y[int(0.0 * SR):int(0.1 * SR)]) > 0
    assert np.std(y[int(0.3 * SR):int(0.6 * SR)]) > 0.003


def test_parade_on_a_shield_is_shorter_than_on_the_shaft():
    s = M.spear(sr=SR, kind="parade", shield=1)
    h = M.spear(sr=SR, kind="parade")
    assert _centroid(s[:SR // 10]) > _centroid(h[:SR // 10])
