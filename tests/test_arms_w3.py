"""Arms rebuilt on ballistics and brass: pascals, no common normalisation, calibre in the impact, the case as an engine."""
import numpy as np
import pytest

from genny import arms, ballistics as B, foley
from genny.analysis import detect_clicks

SR = 48000


def _pa(y):
    return np.asarray(y) * arms.REF_PA / arms.REF_PEAK


def _spl(p):
    return 20 * np.log10(max(p, 1e-12) / 20e-6)


def _bare(w, **kw):
    kw = dict(render="pressure", sr=SR, weapon=w, azimuth=0.0, crack=0.0, action=0.0, seed=1) | kw
    return arms.firearm(**kw)


def test_presets_cover_the_old_table_and_cartridges():
    assert set(arms.WEAPONS) == set(arms.PRESETS) and len(arms.PRESETS) == 33
    assert all(p["cart"] in B.CARTRIDGES for p in arms.PRESETS.values())
    assert abs(arms.BARREL_MM["ar15"] - 508.0) < 1e-9


@pytest.mark.parametrize("weapon", arms.PRESETS)
def test_every_weapon_is_finite_and_clean(weapon):
    y = arms.firearm(render="pressure", sr=SR, weapon=weapon, shots=2, level="peak")
    assert np.isfinite(y).all() and abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
    assert len(detect_clicks(y[-int(0.05 * SR):])) == 0
    assert len(y) / SR < 6.0


def test_no_common_peak_two_weapons_keep_their_pressure():
    a, b = np.abs(_bare("sniper_50", distance=10.0, azimuth=30.0)).max(), np.abs(_bare("pistol_22", distance=10.0, azimuth=30.0)).max()
    assert 20 * np.log10(a / b) > 30.0


def test_blast_peak_matches_the_ballistics_level_at_3_m():
    y = _bare("ar15", distance=3.0)
    st = B.muzzle_state("556", 0.508)
    pk = B.blast_at(st, "556", 3.0, 0.0)[0]
    assert abs(_spl(np.abs(_pa(y)).max()) - _spl(pk)) < 1.5


def test_pov_levels_follow_the_ear_measurements():
    for w, db in (("sniper_308", 161.0), ("ar15", 160.0)):
        y = arms.firearm(render="pressure", sr=SR, weapon=w, pov=True, action=0.0, seed=1)
        assert abs(_spl(np.abs(_pa(y)).max()) - db) < 4.0, w


def test_nwave_of_the_5_56_at_3_m_and_none_outside_the_cone():
    d, l, v, _, _ = arms.CALIBERS["556"]
    w, ta = B.n_wave(v, d, l, 30.0, 3.0, arms.drag_length("556"), SR)
    assert ta is not None and abs(np.abs(w).max() - B.n_wave_pa(v, d, l, 3.0)[0]) < 25.0
    assert B.n_wave(v, d, l, 1.0, 30.0, arms.drag_length("556"), SR)[1] is None
    y = arms.firearm(render="pressure", sr=SR, weapon="smg_mp5sd", seed=1)
    assert np.isfinite(y).all()


def test_suppressor_lowers_the_blast_by_its_reduction_and_keeps_the_crack():
    open_, can = _bare("ar15", distance=3.0, suppressor=0.0), _bare("ar15", distance=3.0, suppressor=1.0)
    assert 20.0 < 20 * np.log10(np.abs(open_).max() / np.abs(can).max()) < 33.0
    c0 = arms.firearm(render="pressure", sr=SR, weapon="ar15", distance=10.0, azimuth=10.0, action=0.0, seed=1, ground="none")
    c1 = arms.firearm(render="pressure", sr=SR, weapon="ar15", distance=10.0, azimuth=10.0, action=0.0, seed=1, ground="none", suppressor=1.0, first_pop=0.0)
    k = lambda y: np.abs(y[:int(0.008 * SR)]).max()
    assert abs(20 * np.log10(k(c0) / k(c1))) < 1.0


def test_first_round_of_a_can_is_louder():
    kw = dict(sr=SR, weapon="smg_mp5sd", shots=3, rate=4.0, action=0.0, crack=0.0, echo=0.0, seed=2, render="pressure")
    a, b = arms.firearm(first_pop=3.0, **kw), arms.firearm(first_pop=0.0, **kw)
    k = lambda y, i: np.abs(y[int(i * 0.25 * SR) + 200:int((i + 0.8) * 0.25 * SR)]).max()
    r = [20 * np.log10(k(a, i) / k(b, i)) for i in range(3)]
    assert abs(r[0] - r[2] - 3.0) < 0.5


def test_recorder_camera_squashes_a_loud_shot_and_spares_a_quiet_one():
    q = lambda w, rec: arms.firearm(sr=SR, weapon=w, distance=50.0 if w == "pistol_22" else 10.0, recorder=rec, place="none", ground="none", crack=0.0, action=0.0, seed=1)
    cf = lambda y: np.abs(y).max() / np.sqrt(np.mean(y ** 2))
    assert cf(q("sniper_50", "camera")) < cf(q("sniper_50", "none")) / 1.5
    assert abs(20 * np.log10(np.abs(q("pistol_22", "camera")).max() / np.abs(q("pistol_22", "none")).max())) < 3.0


def test_barrel_changes_speed_and_level():
    short, long_ = _bare("ar15", distance=3.0, barrel=0.25), _bare("ar15", distance=3.0, barrel=0.6)
    assert B.muzzle_velocity("556", 0.25) < B.muzzle_velocity("556", 0.6)
    assert np.abs(short).max() != np.abs(long_).max()


def test_casing_default_is_the_556():
    assert np.array_equal(foley.casing(seed=2), foley.casing(caliber="556", seed=2))
    assert len(foley.casing(caliber="generic", seed=2)) != len(foley.casing(seed=2))


def test_casing_ricochet_rings_at_the_wall_modes_with_a_long_tail():
    y = arms.casing_ricochet(sr=SR, caliber="50bmg", spin=0.0, seed=1)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    s = np.abs(np.fft.rfft(y))
    m = (f > 1500) & (f < 4000)
    assert abs(f[m][np.argmax(s[m])] - 2700.0) < 150.0
    assert len(y) / SR > 0.5


def test_hit_default_path_is_the_ricochets_hit():
    a = arms._hit("steel", 0.2, 10.0, 60, 900.0, 7.0, np.random.default_rng(0))
    assert abs(np.max(np.abs(a)) - 1.0) < 1e-3


def test_impact_follows_the_impulse():
    pk = lambda m, c, **k: np.abs(arms.bullet_impact(sr=SR, material=m, caliber=c, distance=1.0, **k)).max()
    assert abs(20 * np.log10(pk("metal", "50bmg") / pk("metal", "556")) - 20.2) < 2.5
    assert abs(20 * np.log10(pk("metal", "50bmg", law="energy") / pk("metal", "556", law="energy")) - 10.0) < 2.5
    assert abs(20 * np.log10(pk("metal", "50bmg", gain_db=-6.0) / pk("metal", "50bmg")) + 6.0) < 0.1
    assert abs(pk("metal", "50bmg") - 0.85) < 0.02


def test_thin_metal_is_perforated_and_takes_less():
    j_thin = arms._transfer("metal_thin", "50bmg")[0]
    assert 12.0 < 20 * np.log10(j_thin / arms._transfer("metal_thin", "556")[0]) < 16.0
    assert arms._transfer("metal", "556")[0] == pytest.approx(0.004 * 948.0, rel=0.05)


def test_half_sine_sum_is_the_impulse():
    f = B.half_sine_force(2.0, 60e-6, 400, SR)
    assert abs(f.sum() / SR - 2.0) < 1e-5


def test_minigun_rate_and_rain_are_on_the_signal():
    y = arms.minigun(render="pressure", sr=SR, dur=1.0, rate=50.0, spin=0.1, seed=1)
    e = np.sqrt(np.convolve(y * y, np.ones(48) / 48, mode="same"))[int(0.2 * SR):int(1.0 * SR)]
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    assert abs((int(0.012 * SR) + int(np.argmax(ac[int(0.012 * SR):int(0.03 * SR)]))) / SR - 0.02) < 0.002
    assert len(y) / SR > 1.5 + 3.0 and np.isfinite(y).all()


def test_unknown_names_are_refused():
    with pytest.raises(ValueError):
        arms.firearm(sr=SR, weapon="phaser")
    with pytest.raises(ValueError):
        arms.bullet_impact(sr=SR, law="x")
    with pytest.raises(ValueError):
        B.recorder(np.zeros(4), "tape")
