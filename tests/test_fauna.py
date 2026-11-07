"""Animals: every variant renders without a click; the sourced numbers are measured on the rendered signal."""
import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from genny import creatures as C
from genny import fauna as A
from genny.analysis import detect_clicks
from genny.sfx import REGISTRY

SR = 22050
HI = 48000


def spec(y, sr):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return np.fft.rfftfreq(len(y), 1 / sr), Y


def top(y, sr, lo, hi):
    f, Y = spec(y, sr)
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(Y[m])]


def share(y, sr, lo, hi):
    f, Y = spec(y, sr)
    return Y[(f >= lo) & (f < hi)].sum() / Y.sum()


def centroid(y, sr):
    f, Y = spec(y, sr)
    return (f * Y).sum() / Y.sum()


def rms(y):
    return float(np.sqrt(np.mean(np.square(y))))


def env_top(y, sr, lo, hi):
    return top(np.abs(y) - np.mean(np.abs(y)), sr, lo, hi)


def f0_est(y, sr, lo, hi):
    """Lag of the strongest autocorrelation peak between lo and hi Hz."""
    y = y - y.mean()
    c = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)[:len(y)]
    a, b = int(sr / hi), int(sr / lo)
    return sr / (a + np.argmax(c[a:b]))


def span(y, sr, db=-30.0):
    e = np.convolve(np.abs(y), np.ones(int(0.005 * sr)) / int(0.005 * sr), "same")
    on = np.flatnonzero(e > e.max() * 10 ** (db / 20))
    return (on[-1] - on[0]) / sr


CASES = [(A.bird_kind, dict(kind=k)) for k in A.BIRD_KINDS] + [(A.chicken, dict(kind=k)) for k in A.CHICKEN_KINDS]
CASES += [(A.pig, dict(kind=k)) for k in A.PIG] + [(A.sheep, dict(kind=k)) for k in A.SHEEP] + [(A.horse, dict(kind=k)) for k in A.HORSE]
CASES += [(A.donkey, {})] + [(A.farm_bird, dict(kind=k)) for k in A.FARM_BIRDS] + [(A.farm_animal, dict(kind=k)) for k in A.FARM_KINDS]
CASES += [(A.wild_animal, dict(kind=k)) for k in A.WILD if k != "ours"] + [(A.primate, dict(kind=k)) for k in A.PRIMATES] + [(A.elephant, dict(kind=k)) for k in A.ELEPHANT]
CASES += [(A.whale, dict(kind="bosse")), (A.whale, dict(kind="dauphin"))] + [(A.rodent, dict(kind=k)) for k in A.RODENTS] + [(A.reptile, dict(kind="tortue"))]
CASES += [(A.slither, dict(kind=k)) for k in A.SLITHER] + [(A.swarm, dict(kind=k, count=3, dur=1.0)) for k in A.SWARM if k != "cigales"]


def _id(c):
    return f"{c[0].__name__}-{'-'.join(map(str, c[1].values()))}"


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_variant_renders_without_click(case):
    y = np.asarray(case[0](sr=SR, **case[1]))
    assert np.isfinite(y).all() and 0.01 < np.abs(y).max() <= 1.0
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
    assert len(detect_clicks(y)) == 0


def _clean(y):
    assert np.isfinite(y).all() and 0.01 < np.abs(y).max() <= 1.0
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


# Sounds with attacks of their own: every detection must be explained by where it falls, not waved through.
@pytest.mark.parametrize("call", ["chirp", "rattle", "whirr", "song"])
def test_cicada_detections_lie_on_the_bursts(call):
    n = 2 * A.P
    y = A.cicada_voice(n, call, np.random.default_rng(0))
    p = A.cicada_pulse(n, A.cicada_call_plan(n, call, np.random.default_rng(0)))
    c = detect_clicks(y)
    assert np.all(p[np.minimum(c, n - 1)] > 0.2)     # never in the gaps between bursts
    _clean(A.swarm(sr=SR, kind="cigales", count=3, call=call, dur=1.0))


def test_creature_detections_lie_on_the_footfalls():
    y, t = A.creature_track(6.0, 3, 0.5, "forest_floor", 0)
    long = len(C._step("forest_floor", 0, "walk", 1.0, 6.0, "barefoot", 0.5, 0)) / A.P
    c = detect_clicks(y) / A.P
    assert len(c) and all(np.any((u >= t - 0.01) & (u <= t + long)) for u in c)
    _clean(A.creature_steps(sr=SR, steps=3))


@pytest.mark.parametrize("f,kw,brick", [
    (A.wild_animal, dict(kind="ours"), dict(size=2.2, aggression=0.75, pitch=65.0, dur=1.6)),
    (A.reptile, dict(kind="crocodile"), dict(size=3.0, aggression=0.7, pitch=38.0, dur=2.0))])
def test_growl_detections_are_the_pulses_of_the_brick(f, kw, brick):
    """Below 50 Hz, or period-doubled, one growl pulse stands alone in the detector's 20 ms: the same times come from `growl` by itself, in sound, not in silence."""
    y = f(sr=SR, **kw)
    _clean(y)
    b = C.growl(sr=SR, **brick)
    c, cb = detect_clicks(y), detect_clicks(b)
    e = np.convolve(np.abs(y), np.ones(220) / 220, "same")
    assert np.all(e[c] > 0.03 * e.max())
    assert all(np.any(np.abs(cb - i) < 0.012 * SR) for i in c)


def test_rattle_detections_lie_in_the_rattle():
    """Narrow pulses of 7 kHz noise: within 20 ms each stands alone, d2 follows from level (arithmetic in L12.md). The hiss before it is clean."""
    y = A.reptile(sr=SR, kind="serpent")
    _clean(y)
    assert np.all(detect_clicks(y) / SR > 1.15)


def test_herd_detections_of_the_focus_lie_on_hooves():
    d = 3.0
    y = A.herd(2, d, "forest_floor", 0)
    pace = 4.0 * A.HERD_HZ * 60.0
    steps = int(d * 4.0 * A.HERD_HZ) + 1
    t = np.concatenate([C.footsteps_plan("gallop", pace, steps, 0.5, i) + 0.09 * i for i in range(A.FOCUS)])
    u = detect_clicks(y) / A.P
    assert all(np.any((v >= t - 0.01) & (v <= t + 0.2)) for v in u)
    _clean(A.creature_steps(sr=SR, kind="troupeau", count=10, dur=1.5))


def test_names_and_bad_kinds():
    fs = (A.bird_kind, A.chicken, A.pig, A.sheep, A.horse, A.farm_bird, A.farm_animal, A.wild_animal, A.primate, A.elephant, A.whale, A.rodent, A.reptile, A.slither, A.swarm, A.creature_steps)
    for f in fs:
        assert f.__name__ in REGISTRY
        with pytest.raises(ValueError):
            f(sr=SR, kind="nope")
    assert "donkey" in REGISTRY


# ---------------------------------------------------------------------------------- horse
def _segment(y, sr, a, b):
    return y[int(a * len(y)):int(b * len(y))]


def test_whinny_climax_has_both_sources_at_briefer_values():
    y = A.horse(sr=HI)
    c = _segment(y, HI, A.INTRO + 0.05, 0.75)
    assert top(c, HI, 1000, 2300) == pytest.approx(1543.26, abs=326.45)
    assert f0_est(c, HI, 250, 650) == pytest.approx(399.22, abs=99.39)


def test_whinny_intro_is_the_whistle_alone():
    y = A.horse(sr=HI)
    i = _segment(y, HI, 0.04, A.INTRO - 0.01)
    assert share(i, HI, 1200, 2100) > 0.9
    assert share(i, HI, 250, 650) < 0.01


def test_whinny_overlap_is_794_percent():
    y = A.whinny()
    n = int(A.INTRO * len(y))
    assert np.abs(y[:n - 64]).max() > 0 and A.INTRO == pytest.approx(0.206)
    win = int(0.02 * A.P)
    lp = sosfilt(butter(6, 800.0, "lowpass", fs=A.P, output="sos"), y)
    low = np.array([rms(lp[i:i + win]) for i in range(0, len(y) - win, win)])
    first = np.argmax(low > 0.1 * low.max()) * win / len(y)
    assert first == pytest.approx(A.INTRO, abs=0.07)


def test_whinny_sources_move_independently():
    f = A.horse(sr=HI, f0=350.0, g0=1700.0)
    g = A.horse(sr=HI, f0=450.0, g0=1700.0)
    assert abs(top(_segment(f, HI, 0.3, 0.75), HI, 1200, 2300) - top(_segment(g, HI, 0.3, 0.75), HI, 1200, 2300)) < 30
    assert f0_est(_segment(g, HI, 0.3, 0.75), HI, 250, 650) > f0_est(_segment(f, HI, 0.3, 0.75), HI, 250, 650)


def test_whinny_valence():
    assert len(A.horse(sr=SR, valence=-1.0)) > len(A.horse(sr=SR, valence=1.0))
    assert top(_segment(A.horse(sr=HI, valence=-1.0), HI, 0.03, 0.15), HI, 1000, 2300) > top(_segment(A.horse(sr=HI, valence=1.0), HI, 0.03, 0.15), HI, 1000, 2300)


def test_only_the_whinny_has_the_whistle():
    for k in ("ebrouement", "soufflement", "nicker"):
        assert share(A.horse(sr=HI, kind=k), HI, 1400, 1700) < 0.2, k


def test_snort_flutters_nicker_pulses():
    assert env_top(A.horse(sr=HI, kind="ebrouement"), HI, 20, 80) == pytest.approx(42.0, abs=3.0)
    assert env_top(A.horse(sr=HI, kind="nicker"), HI, 5, 30) == pytest.approx(13.0, abs=1.5)


# ---------------------------------------------------------------------------------- donkey, goat, sheep, pig
def test_bray_alternates_in_100_500():
    a = A.BRAY_S[0]
    y = A.donkey(sr=HI, cycles=1)
    hee, haw = y[int(0.05 * HI):int((a - 0.05) * HI)], y[int((a + 0.05) * HI):int((a + A.BRAY_S[1] - 0.1) * HI)]
    fh, fa = f0_est(hee, HI, 250, 560), f0_est(haw, HI, 90, 220)
    assert 300 <= fh <= 500 and 100 <= fa <= 190


def test_bray_length_follows_the_cycles():
    assert len(A.donkey(sr=SR, cycles=2)) == pytest.approx(2 * sum(A.BRAY_S) * SR, abs=0.1 * SR)


@pytest.mark.parametrize("age,hz", [(0.0, 573.0), (7.0, 467.0), (3.5, 520.0)])
def test_kid_f0_with_age(age, hz):
    y = A.farm_animal(sr=HI, kind="chevre", age=age)
    assert f0_est(y[int(0.1 * HI):int(0.7 * HI)], HI, 350, 700) == pytest.approx(hz, rel=0.02)


@pytest.mark.parametrize("kind", list(A.SHEEP))
def test_bleat_trill(kind):
    assert env_top(A.sheep(sr=HI, kind=kind), HI, 3, 15) == pytest.approx(A.BLEAT_HZ, abs=1.5)


def test_farm_table_plays_the_whole_table():
    assert len(A.farm_animal(sr=SR, kind="table")) / SR == pytest.approx(sum(p["dur"] for p in A.FARM.values()) + 0.5, abs=0.05)


@pytest.mark.parametrize("kind,s", [("grognement", 0.94), ("aboiement", 0.21)])
def test_pig_durations(kind, s):
    y = A.pig(sr=HI, kind=kind)
    assert len(y) / HI == pytest.approx(s, abs=0.01)
    assert span(y, HI) == pytest.approx(s, rel=0.2)


@pytest.mark.parametrize("seed", range(6))
def test_pig_squeal_q75_near_4_khz(seed):
    f, Y = spec(A.pig(sr=HI, kind="couinement", seed=seed), HI)
    assert f[np.searchsorted(np.cumsum(Y) / Y.sum(), 0.75)] == pytest.approx(4000.0, rel=0.1)
    assert centroid(A.pig(sr=HI, kind="grognement"), HI) < centroid(A.pig(sr=HI, kind="couinement"), HI)


# ---------------------------------------------------------------------------------- elephant, whales
def test_rumble_at_20_hz():
    y = A.elephant(sr=HI, kind="rumble")
    assert f0_est(y[HI // 2:-HI // 2], HI, 12, 40) == pytest.approx(20.0, abs=1.0)
    assert 12 <= A.RUMBLE_HZ <= 20


def test_combination_call_spans_over_four_octaves():
    y = A.elephant(sr=HI, kind="combine")
    n = len(y)
    a = f0_est(y[int(0.03 * n):int(0.25 * n)], HI, 12, 60)
    b = f0_est(y[int(0.88 * n):int(0.97 * n)], HI, 300, 600)
    assert a == pytest.approx(20.0, rel=0.2)
    assert b == pytest.approx(470.0, rel=0.05)
    assert np.log2(b / a) > 4.0


def test_blue_whale_call_follows_the_southern_ocean_figures():
    y = A.whale(sr=SR, kind="bleue", wet=0.0)
    assert len(y) / SR == pytest.approx(18.0, abs=0.1)
    assert top(y[int(1 * SR):int(8.5 * SR)], SR, 10, 40) == pytest.approx(27.0, abs=0.3)
    assert top(y[int(10.2 * SR):int(13 * SR)], SR, 10, 40) == pytest.approx(19.0, abs=0.6)
    assert top(y[int(16 * SR):int(17.2 * SR)], SR, 10, 40) == pytest.approx(18.0, abs=1.0)
    assert share(y, SR, 10, 40) > 0.9


def test_blue_whale_survives_the_water():
    assert top(A.whale(sr=SR, kind="bleue")[int(1 * SR):int(8.5 * SR)], SR, 10, 40) == pytest.approx(27.0, abs=0.3)


@pytest.mark.parametrize("wet", [0.0, 0.3])
def test_humpback_stays_in_40_4000(wet):
    y = A.whale(sr=HI, kind="bosse", wet=wet)
    assert share(y, HI, 40, 4000) > 0.99
    f, Y = spec(y, HI)
    p = np.cumsum(Y) / Y.sum()
    assert f[np.searchsorted(p, 0.5)] < f[np.searchsorted(p, 0.995)] and f[np.searchsorted(p, 0.995)] > 1000


def test_dolphin_whistles_sweep():
    y = A.whale(sr=HI, kind="dauphin")
    assert share(y, HI, 100, 24000) > 0.999
    assert share(y, HI, 5000, 16000) > 0.8
    assert top(y[:int(0.1 * HI)], HI, 4000, 16000) != top(y[int(0.5 * HI):int(0.6 * HI)], HI, 4000, 16000)


# ---------------------------------------------------------------------------------- birds: Farnell
def test_ring_modulator_fader():
    n = HI
    a, b = A._pulse(np.full(n, 8.0), 1.0), A._pulse(np.full(n, 5.0), 1.0)
    assert np.allclose(A.bronchi(n, 8.0, 5.0, 1.0, 1.0, 1.0), a + b)        # 1 = sum only
    assert np.allclose(A.bronchi(n, 8.0, 5.0, 1.0, 1.0, 0.0), a * b)        # 0 = difference only
    assert np.allclose(A.bronchi(n, 8.0, 5.0, 1.0, 1.0, 0.5), 0.5 * (a + b + a * b))


def test_rm_changes_the_syllable():
    s = lambda rm: (0.5, [(0, 800), (1, 800)], 1.0, 8.0, 5.0, rm, 150.0, 2.0, 1000, 2000, 3.0, 300, 0.0, 0.0)
    assert not np.allclose(A._syrinx(s(0.0), 0), A._syrinx(s(1.0), 0))


def test_eagle_is_brighter_than_heron():
    assert centroid(A.bird_kind(sr=HI, kind="rapaces"), HI) > centroid(A.bird_kind(sr=HI, kind="echassiers"), HI)


def test_chick_is_higher_than_hen():
    assert centroid(A.chicken(sr=HI, kind="poussin"), HI) > 2 * centroid(A.chicken(sr=HI, kind="poule"), HI)


# ---------------------------------------------------------------------------------- swarm: Farnell
def test_cricket_falls_over_the_whole_chirp_4500_to_3900():
    n = HI
    rng = np.random.default_rng(0)
    y = A.cricket_voice(n, 1.43, rng, bright=0.0)
    chirp_p = 0.017
    first, last = y[0:int(chirp_p * HI)], y[int(6 * chirp_p * HI):int(7 * chirp_p * HI)]
    fa, fb = top(first, HI, 3000, 6000), top(last, HI, 3000, 6000)
    assert fa > fb
    assert fa == pytest.approx(4500.0 * (1 - 0.133 * 0.5 / 7), rel=0.05)
    assert fb == pytest.approx(4500.0 * (1 - 0.133 * 6.5 / 7), rel=0.05)


def test_second_wing_offset_moves():
    d = A.wing_det(HI, np.random.default_rng(0))
    assert d.std() > 1e-4 and d.mean() == pytest.approx(1.004, abs=0.003)
    assert A.wing_det(HI, np.random.default_rng(0), True).max() > 1.02       # a mosquito turning


@pytest.mark.parametrize("kind,hz", [("abeilles", 230.0), ("mouches", 220.0)])
def test_swarm_wingbeat_at_the_voice(kind, hz):
    y = A._wings(A.SWARM[kind], HI, np.random.default_rng(0), hz, 0.0)
    assert top(y, HI, 150, 300) == pytest.approx(hz, abs=2.0)


def test_voices_are_decorrelated():
    n = HI // 2
    a, b = A.swarm_voices("abeilles", 4, n, 0), A.swarm_voices("abeilles", 16, n, 0)
    assert 20 * np.log10(rms(b) / rms(a)) == pytest.approx(6.0, abs=2.0)      # 4x the voices: +6 dB if incoherent


@pytest.mark.parametrize("call,hz", [("song", 25.0), ("whirr", 30.0), ("rattle", 8.0)])
def test_cicada_call_rates(call, hz):
    y = A.cicada_voice(3 * HI, call, np.random.default_rng(0))
    e = np.abs(y)
    assert top(e - e.mean(), HI, 2, 60) == pytest.approx(hz, abs=0.6)


def test_cicada_bands_and_notch():
    f, Y = spec(A.cicada_voice(2 * HI, "whirr", np.random.default_rng(0)), HI)
    band = lambda a, b: Y[(f >= a) & (f < b)].mean()
    assert band(5000, 6000) > 10 * band(6400, 6600)
    assert band(7000, 8000) > 10 * band(6400, 6600)
    assert band(5000, 6000) > band(7000, 8000)


def test_cicada_changes_call_every_5_s_at_random():
    plans = {tuple(A.cicada_call_plan(40 * A.P, "auto", np.random.default_rng(s))) for s in range(4)}
    assert len(plans) > 1 and all(len(p) == 9 and set(p) <= {0, 1, 2} for p in plans)


# ---------------------------------------------------------------------------------- steps and herd
def test_weight_goes_up_to_20_and_slows_the_steps():
    assert A.stride_pace(20.0) / A.stride_pace(1.0) == pytest.approx(20.0 ** -0.13)
    _, t = A.creature_track(20.0, 4, 0.0, "forest_floor", 0)
    assert np.mean(np.diff(t)) == pytest.approx(60.0 / A.stride_pace(20.0), rel=0.05)
    y20, y2 = A.creature_steps(sr=SR, weight=20.0, steps=3), A.creature_steps(sr=SR, weight=2.0, steps=3)
    assert centroid(y20, SR) < centroid(y2, SR)


def test_claws_add_the_top():
    a, b = A.creature_steps(sr=SR, steps=3, claws=0.0), A.creature_steps(sr=SR, steps=3, claws=1.0)
    assert share(b, SR, 2000, 6000) > share(a, SR, 2000, 6000)


def test_texture_rate_follows_the_herd():
    r = np.random.default_rng(0)
    assert len(A.herd_texture_times(2, 10.0, r)) == 0
    assert len(A.herd_texture_times(32, 10.0, r)) == pytest.approx(30 * 4 * A.HERD_HZ * 10, rel=0.1)


def test_herd_focus_gallops_at_the_stride_rate_and_texture_grows():
    few, many = A.creature_steps(sr=SR, kind="troupeau", count=2, dur=6.0), A.creature_steps(sr=SR, kind="troupeau", count=40, dur=6.0)
    assert env_top(few, SR, 0.8, 3.0) == pytest.approx(A.HERD_HZ, abs=0.3)
    e = lambda y: np.abs(y).std() / np.abs(y).mean()
    assert e(many) < e(few)                  # the many fill the gaps between the focus footfalls


def test_slither_swells_at_the_body_wave():
    assert env_top(A.slither(sr=SR, kind="sable", dur=4.0), SR, 0.5, 4.0) == pytest.approx(A.SLITHER_HZ, abs=0.3)


def test_bat_sweeps_are_brought_down_by_ten():
    y = A.rodent(sr=HI, kind="chauve_souris")
    assert share(y, HI, 1800, 4400) > 0.2
    assert env_top(y, HI, 5, 20) == pytest.approx(A.BAT_RATE, abs=1.0)
