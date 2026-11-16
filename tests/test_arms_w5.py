"""W5: default firearm render against descriptors measured on real recordings (open air, 10 m)."""
import inspect

import numpy as np
import pytest

from genny import arms, presets

SR = 44100
GEN = dict(centroid=(500, 1600), b_low=(-10, -1), b_1_4k=(-9, 3), b_4_16k=(-20, -7), crest=(8, 18), e50=(20, 200),
           e90=(100, 700), t20=(200, 1300), body=(40, 250))
HEAVY = {**GEN, "centroid": (250, 900), "b_low": (-4, 4), "b_1_4k": (-12, 0), "b_4_16k": (-24, -10), "crest": (6, 18)}
SUPP = {**GEN, "centroid": (300, 1500), "b_low": (-8, 10), "b_1_4k": (-9, 6), "b_4_16k": (-24, 5), "crest": (6, 18)}
REPS = {"pistol_9mm": GEN, "pistol_22": GEN, "sniper_22": GEN, "ar15": GEN, "ak47": GEN, "sniper_308": GEN, "shotgun_buck": GEN,
        "shotgun_slug": GEN, "revolver_357": GEN, "sniper_50": HEAVY, "smg_mp5sd": SUPP}


def profile(y, sr=SR, span=1.5):
    a = np.abs(y)
    k = int(a.argmax())
    lo = max(0, k - int(0.05 * sr))
    on = lo + int(np.flatnonzero(a[lo:k + 1] > 0.1 * a[k])[0])
    seg = y[on:on + int(span * sr)]
    e = np.cumsum(seg ** 2)
    e /= e[-1]
    nf = 1 << int(np.ceil(np.log2(len(seg) + 1)))
    spec = np.fft.rfft(seg, nf)
    p = np.abs(spec) ** 2
    f = np.fft.rfftfreq(nf, 1 / sr)
    band = [10 * np.log10(p[(f >= b0) & (f < min(b1, sr / 2))].sum() + 1e-30) for b0, b1 in ((20, 200), (200, 1000), (1000, 4000), (4000, 16000))]
    sel = (f >= 20) & (f < min(16000, sr / 2))
    fr = int(0.005 * sr)
    m = len(seg) // fr
    env = 10 * np.log10(np.mean(seg[:m * fr].reshape(m, fr) ** 2, axis=1) + 1e-30)
    i0 = int(env.argmax())
    spec[(f < 20) | (f > 300)] = 0.0
    low = np.fft.irfft(spec, nf)[:len(seg)]
    el = np.cumsum(low * low)
    return {"centroid": float((f[sel] * p[sel]).sum() / p[sel].sum()), "b_low": band[0] - band[1], "b_1_4k": band[2] - band[1],
            "b_4_16k": band[3] - band[1], "crest": 20 * np.log10(a[k] / np.sqrt(np.mean(seg[:int(0.1 * sr)] ** 2))),
            "e50": 1e3 * np.searchsorted(e, 0.5) / sr, "e90": 1e3 * np.searchsorted(e, 0.9) / sr,
            "t20": 5.0 * (np.flatnonzero(env[i0:] - env[i0] > -20)[-1] + 1), "body": 1e3 * np.searchsorted(el / el[-1], 0.5) / sr}


@pytest.mark.parametrize("weapon", sorted(REPS))
def test_default_render_in_measured_ranges(weapon):
    y = np.asarray(arms.firearm(sr=SR, weapon=weapon))
    assert np.all(np.isfinite(y)) and 0.0 < np.abs(y).max() <= 0.9
    d = profile(y)
    bad = {k: round(d[k], 1) for k, (lo, hi) in REPS[weapon].items() if not lo <= d[k] <= hi}
    assert not bad, bad


def test_relative_levels_follow_the_weapons():
    pk = {w: 20 * np.log10(np.abs(arms.firearm(sr=SR, weapon=w)).max()) for w in ("ar15", "pistol_9mm", "pistol_22", "smg_mp5sd", "shotgun_buck", "sniper_50")}
    rel = {w: v - pk["ar15"] for w, v in pk.items()}
    assert -15 <= rel["pistol_9mm"] <= -3 and -35 <= rel["pistol_22"] <= -18 and -40 <= rel["smg_mp5sd"] <= -15
    assert -6 <= rel["shotgun_buck"] <= 6 and 0 <= rel["sniper_50"] <= 12
    assert pk["smg_mp5sd"] > -60.0


def test_old_dry_render_is_kept():
    kw = dict(sr=SR, weapon="pistol_9mm", seed=2)
    dry = arms.firearm(env="none", **kw)
    assert np.array_equal(dry, arms.firearm(env="anechoic", **kw))
    wet = arms.firearm(**kw)
    assert len(wet) > len(dry) and profile(dry)["e90"] < 30.0 < profile(wet)["e90"]


def test_env_presets_toml_and_inline_are_identical():
    assert set(presets.list("weapon_env")) >= {"open_field", "forest", "urban", "range", "indoor"}
    kw = dict(sr=SR, weapon="ar15", seed=1)
    assert np.array_equal(arms.firearm(env="urban", **kw), arms.firearm(env={"extends": "urban"}, **kw))
    assert not np.array_equal(arms.firearm(env="urban", **kw), arms.firearm(env="indoor", **kw))
    with pytest.raises(ValueError):
        arms.firearm(env="nowhere", **kw)


def test_env_ir_direct_first_and_field_grows_with_distance():
    a, b = arms.env_ir("open_field", 10.0), arms.env_ir("open_field", 50.0)
    assert a[0] == 1.0 and b[0] == 1.0 and np.sum(b[1:] ** 2) > 10 * np.sum(a[1:] ** 2)
    assert np.array_equal(arms.env_ir("none", 10.0), np.ones(1))


def test_other_sounds_take_env_but_ricochet_does_not():
    for fn in (arms.minigun, arms.bullet_pass, arms.bullet_impact):
        assert "env" in inspect.signature(fn).parameters
    assert "env" not in inspect.signature(arms.ricochet).parameters
    for y in (arms.minigun(sr=SR, dur=0.5), arms.minigun(sr=SR, dur=0.5, fire=False), arms.bullet_pass(sr=SR), arms.bullet_impact(sr=SR)):
        y = np.asarray(y)
        assert y.size and np.all(np.isfinite(y)) and np.abs(y).max() > 1e-5
