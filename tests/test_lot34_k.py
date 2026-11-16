import math
import warnings

import numpy as np
import pytest

from genny import analysis, compose as C, contact, cyber
from genny.physical import nonverbal, space
from genny.reverbs import room_info

SR = 16000


def test_room_presets_stay_in_their_published_ranges():
    for name, vol, t_mid in (("concert_hall", (10000.0, 20000.0), (1.7, 2.4)), ("opera", (8000.0, 12000.0), (1.0, 1.7))):
        r = room_info(name)
        mid = float(np.mean(r["t60"][2:4]))
        assert vol[0] < r["volume"] < vol[1], (name, r["volume"])
        assert t_mid[0] < mid < t_mid[1], (name, mid)


def test_bearing_contact_angle_90_gives_half_ball_rate():
    f = cyber.bearing_frequencies(1800.0, balls=9, angle_deg=90.0)
    assert f["bpfo"] == pytest.approx(f["bpfi"], rel=1e-9)
    assert f["bpfo"] == pytest.approx(9 * 30.0 / 2.0, rel=1e-9)


def test_texture_synth_warns_when_the_target_is_not_reached_and_texturize_modes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2 * cyber.DEFAULT_SR)
    s = cyber.texture_stats(x, cyber.DEFAULT_SR)
    with pytest.warns(UserWarning, match="below 20 dB"):
        cyber.texture_synth(s, 0.5, iters=1, target_snr=1000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cyber.texture_synth(s, 0.5, iters=4, target_snr=-1000.0)
    g = cyber.texturize(x, cyber.DEFAULT_SR, length=1.0, mode="grain")
    assert g.shape == (cyber.DEFAULT_SR,) and np.all(np.isfinite(g))
    with pytest.raises(ValueError):
        cyber.texturize(x, cyber.DEFAULT_SR, mode="nope")


def test_equalized_noise_follows_the_32_erb_levels():
    fc = cyber.erb_centres(32)
    assert fc[0] == pytest.approx(50.0, rel=1e-6) and np.all(np.diff(fc) > 0) and fc[-1] < 12000.0
    lv = np.zeros(32)
    lv[20:24] = 15.0
    y = cyber.equalized_noise(1 << 18, np.random.default_rng(1), lv)
    X = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1.0 / cyber.SR)

    def band(k):
        e = 24.7 * (0.00437 * fc[k] + 1.0)
        return 10 * math.log10(X[(f > fc[k] - e / 2) & (f < fc[k] + e / 2)].sum())

    assert band(22) - band(10) == pytest.approx(15.0, abs=2.0)


def test_surface_texture_accepts_a_measured_profile():
    prof = np.sin(np.linspace(0, 40, 300)) + 0.3 * np.random.default_rng(2).standard_normal(300) + 5.0
    z = contact.surface_texture(prof, n=1024)
    assert z.shape == (1024,) and abs(z.mean()) < 0.2 and z.std() == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError):
        contact.surface_texture(np.ones(3))


def test_cycle_periods_resist_breath_noise():
    rng = np.random.default_rng(3)
    t0 = SR / 140.0
    per = t0 * (1.0 + 0.005 * rng.standard_normal(200))
    ph = 2 * np.pi * np.arange(per.size + 1)
    n = int(per.sum())
    x = np.interp(np.arange(n), np.concatenate([[0.0], np.cumsum(per)]), ph)
    x = np.sin(x) + 0.5 * np.sin(2 * x) + 0.1 * rng.standard_normal(n)
    p = analysis.cycle_periods(x, SR, 140.0)
    assert p.size > 150 and np.mean(p) * SR == pytest.approx(t0, rel=0.01)
    assert analysis.jitter_measures(p)["local"] < 1.2


def test_loop_gain_from_edt_reads_back_the_decay():
    sr = space.SR
    t = np.arange(int(1.5 * sr)) / sr
    ir = np.random.default_rng(4).standard_normal(t.size) * 10 ** (-3.0 * t / 0.9)
    d = space.loop_gain_from_edt(ir, 500.0, 2000.0, 0.056)
    assert d["t60"] == pytest.approx(0.9, rel=0.2)
    assert d["db_per_trip"] == pytest.approx(-60 * 0.056 / d["t60"], rel=1e-9)
    assert d["gain"] == pytest.approx(10 ** (d["db_per_trip"] / 20), rel=1e-9)


def test_laugh_presets_run_at_four_and_a_half_to_six_syllables_per_second():
    for sex in ("male", "female"):
        p = nonverbal.PRESETS[("laugh", sex)]
        assert 4.5 <= 1000.0 / (p["sylLen"] + p["pauseLen"]) <= 6.0
    assert nonverbal.PRESETS[("laugh", "male")]["pauseLen"] == 90
    assert nonverbal.PRESETS[("laugh", "female")]["pauseLen"] == 110


def test_humanize_offset_ms_shifts_every_onset_and_beta_is_accepted():
    L = dict(type="compose", role="bass", style="funk", key="E", bars=2, seed=3)
    a = C.generate(L, 120)
    b = C.generate(dict(L, humanize={"offset_ms": 40}), 120)
    d = np.array([y["t"] - x["t"] for x, y in zip(a, b)])
    assert len(a) == len(b) and np.all(d >= 0) and d.max() == pytest.approx(0.04 / 0.5, rel=1e-6)
    c = C.generate(dict(L, humanize={"time": 8, "beta": 1.0}), 120)
    assert len(c) == len(a) and any(abs(x["t"] - y["t"]) > 1e-6 for x, y in zip(a, c))
