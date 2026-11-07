"""genny.cyber and identify_machine: machines, engine, road, interface cues, effects, textures. Predictions + smoke."""
import numpy as np
import pytest
from scipy.signal import hilbert

import genny.cyber as c
from genny.sfx import REGISTRY, render_sfx

SR = c.SR


def _spec(y):
    N = 1 << int(np.ceil(np.log2(len(y))) + 1)
    return np.abs(np.fft.rfft(y * np.hanning(len(y)), N)), np.fft.rfftfreq(N, 1.0 / SR)


def _db(sp, fr, f, bw=1.5):
    return 20 * np.log10(sp[(fr > f - bw) & (fr < f + bw)].max() + 1e-12)


def _env_spec(y):
    e = np.abs(hilbert(y))
    return _spec(e - e.mean())


# ---------------------------------------------------------------- transformer

def test_transformer_hums_at_twice_the_mains():
    for mains in (50.0, 60.0):
        sp, fr = _spec(c.transformer(sr=SR, dur=4.0, mains=mains, load=0.0))
        ref = _db(sp, fr, 2 * mains)
        for k in (1, 3, 5):                                   # an even strain law has no odd multiples
            assert _db(sp, fr, k * mains) < ref - 60
        for k in (4, 6):
            assert ref - 45 < _db(sp, fr, k * mains) < ref + 6


def test_dc_bias_brings_back_the_odd_harmonics():
    sp, fr = _spec(c.transformer(sr=SR, dur=4.0, dc_bias=0.3))
    assert _db(sp, fr, 50.0) > _db(sp, fr, 100.0) - 20
    assert _db(sp, fr, 150.0) > _db(sp, fr, 100.0) - 25


def test_more_flux_is_richer():
    def hi(flux):
        sp, fr = _spec(c.transformer(sr=SR, dur=4.0, flux=flux, load=0.0))
        return _db(sp, fr, 400.0) - _db(sp, fr, 100.0)
    assert hi(1.2) > hi(0.5) + 10


# ---------------------------------------------------------------- gears

def test_gear_frequencies():
    g = c.gear_frequencies(1800, 23, 41)
    assert g["mesh"] == pytest.approx(690.0)
    assert g["shaft_out"] == pytest.approx(30.0 * 23 / 41)
    assert c.gear_frequencies(1800, 20, 40)["hunting"] == pytest.approx(600.0 * 20 / 800)


def test_gearbox_mesh_tone_and_sidebands():
    g = c.gear_frequencies(1800, 23, 41)
    sp, fr = _spec(c.gearbox(sr=SR, dur=4.0, wear=0.0, backlash=0.0, eccentricity=0.5))
    mesh = _db(sp, fr, g["mesh"])
    for side in (g["mesh"] - g["shaft_in"], g["mesh"] + g["shaft_in"], g["mesh"] + g["shaft_out"]):
        assert mesh - 40 < _db(sp, fr, side) < mesh
        assert _db(sp, fr, side) > _db(sp, fr, side + 7.0) + 10       # a line, not a noise floor
    flat, fr = _spec(c.gearbox(sr=SR, dur=4.0, wear=0.0, backlash=0.0, eccentricity=0.0))
    assert _db(flat, fr, g["mesh"] + g["shaft_in"]) < _db(flat, fr, g["mesh"]) - 60


# ---------------------------------------------------------------- bearings

def test_bearing_frequencies_cwru_6205():
    k = c.bearing_frequencies(60.0)                           # multiples of the shaft rate
    for name, value in (("bpfo", 3.5848), ("bpfi", 5.4152), ("bsf", 2.3567), ("ftf", 0.3983)):
        assert k[name] == pytest.approx(value, abs=5e-4)
    assert k["bpfo"] + k["bpfi"] == pytest.approx(9.0)         # their sum is the ball count


@pytest.mark.parametrize("state,key,mult", [("outer", "bpfo", 1), ("inner", "bpfi", 1), ("ball", "bsf", 2)])
def test_bearing_fault_shows_in_the_envelope_spectrum(state, key, mult):
    f = c.bearing_frequencies(1750.0)
    sp, fr = _env_spec(c.bearing(sr=SR, dur=4.0, state=state, severity=1.0))
    target = mult * f[key]
    floor = np.median(sp[(fr > 20) & (fr < 400)])
    assert sp[(fr > target * 0.985) & (fr < target * 1.015)].max() > 8 * floor


def test_inner_race_fault_is_modulated_at_the_shaft_rate():
    f = c.bearing_frequencies(1750.0)
    sp, fr = _env_spec(c.bearing(sr=SR, dur=4.0, state="inner", severity=1.0))
    out, _ = _env_spec(c.bearing(sr=SR, dur=4.0, state="outer", severity=1.0))
    band = (fr > f["shaft"] * 0.97) & (fr < f["shaft"] * 1.03)
    assert sp[band].max() > 5 * out[band].max()


def test_unknown_bearing_state():
    with pytest.raises(ValueError):
        c.bearing(state="melted")


# ---------------------------------------------------------------- smoke

@pytest.mark.parametrize("name", ["transformer", "gearbox", "bearing"])
def test_registered_and_safe(name):
    assert name in REGISTRY
    y = render_sfx(name, 44100, dur=0.5)
    assert np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) <= 1.0


@pytest.mark.parametrize("state", c.BEARING_STATES)
def test_bearing_states_render(state):
    y = c.bearing(sr=22050, dur=0.6, state=state, rpm=600, rpm_end=3000)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.05


# ---------------------------------------------------------------- engine, road

def test_pipe_engine_fires_at_cylinders_rpm_over_120():
    for cyl, rpm in ((4, 1800.0), (6, 2500.0)):
        sp, fr = _spec(c.pipe_engine(sr=SR, dur=4.0, rpm=rpm, cylinders=cyl, unequal=0.0))
        fire = cyl * rpm / 120.0
        low = (fr > 20) & (fr < 4 * fire)
        assert abs(fr[low][np.argmax(sp[low])] / fire - 1) < 0.01
        assert _db(sp, fr, fire) > _db(sp, fr, 1.5 * fire) + 20      # even firing: nothing between its harmonics


def test_lope_brings_out_the_half_firing_order():
    def half(lope):
        sp, fr = _spec(c.pipe_engine(sr=SR, dur=4.0, rpm=1800.0, cylinders=4, unequal=0.0, lope=lope))
        return _db(sp, fr, 30.0) - _db(sp, fr, 60.0)
    assert half(1.0) > half(0.0) + 20


def test_harmonoise_rolling_noise_grows_with_speed():
    r50, p50 = c.road_noise_levels(50.0)
    r100, p100 = c.road_noise_levels(100.0)
    k = int(np.argmin(np.abs(c._HN_F - 1000.0)))
    assert r100[k] - r50[k] == pytest.approx(29.1 * np.log10(2.0), abs=1e-9)      # bR at 1 kHz
    assert c.road_noise_levels(70.0)[0][k] == pytest.approx(90.6)
    j = int(np.argmin(np.abs(c._HN_F - 40.0)))
    assert c.road_noise_levels(70.0)[1][j] == pytest.approx(87.5)                   # aP at 40 Hz
    assert c.road_noise_levels(140.0)[1][j + 5] - c.road_noise_levels(70.0)[1][j + 5] == pytest.approx(0.0)   # 200 Hz: bP 0
    total = lambda L: 10 * np.log10(np.sum(10 ** (L / 10)))
    assert (total(p50) - total(r50)) - (total(p100) - total(r100)) > 7.0           # the tyres catch up with the engine


def test_third_octave_noise_matches_its_levels():
    rng = np.random.default_rng(0)
    lv = c.road_noise_levels(70.0)[0]
    y = c._third_octave_noise(SR * 8, rng, lv)
    P = np.abs(np.fft.rfft(y)) ** 2
    fr = np.fft.rfftfreq(len(y), 1.0 / SR)
    band = lambda f0: 10 * np.log10(P[(fr >= f0 * 2 ** (-1 / 6)) & (fr < f0 * 2 ** (1 / 6))].sum())
    for f0, ref in ((250.0, lv[11]), (4000.0, lv[23])):
        assert band(f0) - band(1000.0) == pytest.approx(ref - lv[17], abs=1.5)


def test_pass_by_pitch_drops_and_level_peaks_at_the_pass():
    y = c.pass_by(sr=SR, dur=4.0, speed=120.0, distance=5.0, power="electric", pan=1)
    assert y.ndim == 2 and c.pass_by(sr=SR, dur=1.0, pan=0).ndim == 1
    e = np.sqrt(np.mean(y.reshape(8, -1, 2) ** 2, axis=(1, 2)))
    assert 2 <= int(np.argmax(e)) <= 5 and e.max() > 3 * e[0]
    left, right = np.mean(y[:SR, 0] ** 2), np.mean(y[:SR, 1] ** 2)
    assert left > 2 * right                                                        # arrives from the left


def test_doppler_ground_reflection_is_a_comb():
    from genny.reverbs import doppler
    x = np.random.default_rng(1).standard_normal(SR * 2)
    dry = doppler(x, SR, speed=0.0, closest=4.0, pan=False, air=False)
    wet = doppler(x, SR, speed=0.0, closest=4.0, pan=False, air=False, ground=0.35, height=1.8)
    assert np.allclose(doppler(x, SR, speed=30.0), doppler(x, SR, speed=30.0, ground=0.0))
    delay = (np.hypot(4.0, 1.8) - 4.0) / 343.0                                    # extra path / c
    P = lambda s: np.abs(np.fft.rfft(s[:SR])) ** 2
    fr = np.fft.rfftfreq(SR, 1.0 / SR)
    gain = lambda f0: 10 * np.log10(P(wet)[(fr > f0 * 0.9) & (fr < f0 * 1.1)].sum() / P(dry)[(fr > f0 * 0.9) & (fr < f0 * 1.1)].sum())
    assert gain(1.0 / delay) < -2.0 and gain(0.5 / delay) > 1.5                    # inverted echo: notch at 1 / delay


# ---------------------------------------------------------------- interface cues

@pytest.mark.parametrize("kind", c.UI_KINDS)
def test_cyber_ui_is_short_and_never_piercing(kind):
    from genny.analysis import describe
    for seed in range(4):
        y = c.cyber_ui(44100, kind=kind, seed=seed)
        d = describe(y, 44100)
        assert 0.1 < d["seconds"] < 2.5 and np.isfinite(y).all()
        assert d["sharpness"] < 2.0 and d["above_5k"] < 0.03
    assert not np.array_equal(c.cyber_ui(44100, kind=kind, seed=0), c.cyber_ui(44100, kind=kind, seed=1))


# ---------------------------------------------------------------- effects

def _voice(dur=2.0):
    t = np.arange(int(dur * SR)) / SR
    return 0.5 * np.sin(2 * np.pi * 140 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.1 * np.sin(2 * np.pi * 1200 * t)


def test_packet_loss_rate_and_burst_length():
    x = np.ones(SR * 60)
    y = c.packet_loss(x, SR, loss=0.2, burst=3.0, packet_ms=20.0, conceal="mute", seed=3)
    lost = y.reshape(-1, int(0.02 * SR))[:, 0] == 0.0
    assert lost.mean() == pytest.approx(0.2, abs=0.04)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, lost.astype(int), 0])))[::2]
    assert runs.mean() == pytest.approx(3.0, abs=0.6)
    assert np.array_equal(c.packet_loss(x, SR, loss=0.0), x)
    with pytest.raises(ValueError):
        c.packet_loss(x, SR, conceal="pray")


def test_beat_repeat_keeps_length_and_repeats_slices():
    x = np.random.default_rng(0).standard_normal(SR * 2)
    L = int(0.125 * SR)
    y = c.beat_repeat(x, SR, grid=0.125, chance=1.0, repeats=2, decay=1.0, reverse=0.0)
    assert y.shape == x.shape
    mid = slice(200, L - 200)                                                      # away from the 2 ms edges
    assert np.allclose(y[L:2 * L][mid], y[:L][mid]) and np.allclose(y[:L][mid], x[:L][mid])
    assert np.array_equal(c.beat_repeat(x, SR, chance=0.0), x)


def test_paulstretch_length_spectrum_and_hold():
    x = _voice()
    y = c.paulstretch(x, SR, stretch=4.0, window=0.1)
    assert len(y) == 4 * len(x) and np.max(np.abs(y)) <= np.max(np.abs(x)) + 1e-9
    sp, fr = _spec(y)
    assert abs(fr[np.argmax(sp)] - 140.0) < 6.0                                    # the pitch is kept
    held = c.paulstretch(x, SR, hold=0.5, length=3.0)
    assert len(held) == 3 * SR
    e = np.sqrt(np.mean(held[SR // 2:5 * SR // 2].reshape(8, -1) ** 2, axis=1))
    assert e.max() / e.min() < 1.6                                                 # a held instant is steady


def test_waveset_and_formant_shift():
    x = _voice()
    y = c.waveset(x, SR, repeats=2)
    assert y.shape == x.shape and not np.allclose(y, x)
    assert len(c.waveset(x, SR, repeats=3, keep_length=False)) > 2.5 * len(x)
    t = np.arange(SR) / SR
    saw = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 60))
    from genny.fx import apply_chain
    vowel = apply_chain(saw, [{"type": "bandpass", "center": 1000.0, "q": 4.0}], SR)
    cen = lambda s: float(np.sum(np.fft.rfftfreq(len(s), 1 / SR) * np.abs(np.fft.rfft(s)) ** 2) / np.sum(np.abs(np.fft.rfft(s)) ** 2))
    up, down = c.formant_shift(vowel, SR, 7.0), c.formant_shift(vowel, SR, -7.0)
    assert cen(up) > 1.2 * cen(vowel) > 1.2 * 1.2 * cen(down)
    sp, fr = _spec(up)
    assert abs((fr[np.argmax(sp)] / 110.0) % 1.0 - 0.5) > 0.45                     # still harmonics of 110 Hz


def test_cyber_voice_runs_mono_and_stereo_at_the_input_level():
    x = _voice()
    y = c.cyber_voice(0.3 * x, SR, seed=1)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert np.sqrt(np.mean(y ** 2)) == pytest.approx(0.3 * np.sqrt(np.mean(x ** 2)), rel=0.01)
    assert np.max(np.abs(c.cyber_voice(1.6 * x, SR, seed=1))) <= 0.99 + 1e-9      # a hot input is not clipped
    assert c.cyber_voice(np.stack([x, x], axis=1), SR).shape == (len(x), 2)


# ---------------------------------------------------------------- textures

def test_texture_filters_reconstruct():
    n = c._tex_len(SR)
    H = c._tex_bank(n, SR)
    assert H.shape[0] == c.TEX_BANDS + 2 and np.allclose((H ** 2).sum(axis=0), 1.0)


def test_texture_synthesis_matches_the_statistics_it_was_given():
    rng = np.random.default_rng(5)
    n = SR * 4
    t = np.arange(n) / SR
    from genny import filters as F
    low = F.bandpass(rng.standard_normal(n), 300.0, SR, 2.0)
    high = F.bandpass(rng.standard_normal(n), 3000.0, SR, 2.0) * (0.5 + 0.5 * np.sin(2 * np.pi * 6.0 * t)) ** 2 * 0.2
    target = c.texture_stats(low + high, SR)
    y = c.texture_synth(target, 3.0, seed=1, iters=8)
    got = c.texture_stats(y, SR)
    lvl = lambda s: np.log10(s["quantiles"][:, c.TEX_Q // 2] + 1e-9)               # median envelope per band
    assert np.corrcoef(lvl(target), lvl(got))[0, 1] > 0.975
    band = int(np.argmax(target["modulation"][:, 3]))                              # the 4-8 Hz modulation band
    assert target["modulation"][band, 3] > 0.5
    assert got["modulation"][band, 3] == pytest.approx(target["modulation"][band, 3], abs=0.15)
    assert not np.allclose(y, c.texture_synth(target, 3.0, seed=2, iters=2))
    assert c.texturize(low + high, SR, length=2.0, iters=2).shape[0] == c._tex_len(2 * SR)


def test_texture_keeps_the_direction_of_sharp_onsets():
    # McDermott & Simoncelli's C2: abrupt onsets and slow decays have a phase that time reversal conjugates
    from genny import filters as F
    rng = np.random.default_rng(3)
    n = SR * 4
    env = np.zeros(n)
    for t0 in rng.uniform(0, n, 40).astype(int):
        env[t0:] += np.exp(-np.arange(n - t0) / (0.05 * SR))
    x = F.bandpass(rng.standard_normal(n), 2000.0, SR, 0.7) * env
    target, rev = c.texture_stats(x, SR), c.texture_stats(x[::-1].copy(), SR)
    assert np.allclose(rev["c2"], np.conj(target["c2"]), atol=0.05)
    got = c.texture_stats(c.texture_synth(target, 3.0, seed=1, iters=6), SR)
    assert np.abs(got["c2"] - target["c2"]).mean() < 0.25 * np.abs(rev["c2"] - target["c2"]).mean()
    assert np.abs(got["c1"] - target["c1"]).mean() < 0.03


# ---------------------------------------------------------------- identification

@pytest.mark.parametrize("name,kw,kind,expect", [
    ("transformer", dict(mains=50.0), "transformer", dict(mains=50.0)),
    ("transformer", dict(mains=60.0, dc_bias=0.3), "transformer", dict(mains=60.0)),
    ("gearbox", dict(), "gearbox", dict(rpm=1800.0, teeth=23, teeth2=41)),
    ("gearbox", dict(rpm=950.0, teeth=31, teeth2=40, housing=2200.0), "gearbox", dict(rpm=950.0, teeth=31, teeth2=40)),
    ("gearbox", dict(rpm=2400.0, teeth=17, teeth2=52, wear=0.6, backlash=0.6, load=0.3), "gearbox", dict(rpm=2400.0, teeth=17)),
    # one first-order sideband is 25 dB under the other here: the spacing must not be read as twice the shaft rate
    ("gearbox", dict(rpm=1500.0, teeth=19), "gearbox", dict(rpm=1500.0, teeth=19, teeth2=41)),
    # a damaged tooth knocks once per turn: an impulsive band, and not a bearing's cage
    ("gearbox", dict(rpm=1500.0, teeth=19, wear=0.8), "gearbox", dict(rpm=1500.0, teeth=19)),
    ("bearing", dict(state="outer", severity=1.0), "bearing", dict(rpm=1750.0, state="outer", balls=9)),
    ("bearing", dict(state="inner", severity=1.0), "bearing", dict(rpm=1750.0, state="inner")),
    # a ball defect shows as the cage rate and its multiples: that takes a few hundred cage turns
    ("bearing", dict(state="ball", severity=1.0, dur=20.0), "bearing", dict(rpm=1750.0, state="ball")),
    ("pipe_engine", dict(), "engine", dict(rpm=1800.0, cylinders=4)),
    ("tyre_squeal", dict(), "unknown", dict()),
])
def test_identify_machine_recovers_the_parameters_of_a_render(name, kw, kind, expect):
    from genny.identify import identify_machine
    r = identify_machine(render_sfx(name, 44100, **{"dur": 5.0, **kw}), 44100)
    assert r["kind"] == kind
    for key, value in expect.items():
        assert r["params"][key] == (pytest.approx(value, rel=0.002) if isinstance(value, float) else value)
    if kind != "unknown":
        assert np.isfinite(render_sfx(r["sfx"], 22050, dur=0.3, **r["params"])).all()      # the answer renders


# ---------------------------------------------------------------- smoke (everything this module registers)

@pytest.mark.parametrize("name", ["pipe_engine", "turbo", "tyre_squeal", "pass_by", "cyber_ui"])
def test_more_registered_and_safe(name):
    y = render_sfx(name, 44100, **({} if name == "cyber_ui" else {"dur": 0.6}))
    assert np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) <= 1.0


@pytest.mark.parametrize("name", ["packet_loss", "beat_repeat", "paulstretch", "waveset", "formant_shift", "cyber_voice", "texturize"])
def test_effects_registered(name):
    from genny.fx import REGISTRY as FX
    assert name in FX and FX[name]["desc"]


# ---------------------------------------------------------------- Randall 1982, Randall & Antoni 2011

def test_gear_load_raises_the_mesh_line_more_than_its_double():
    # Randall 1982, Fig. 3: +21 dB on the mesh line and +7 dB on its second harmonic from 10 % to full load.
    def lines(load):
        sp, fr = _spec(c.gearbox(sr=SR, dur=3.0, rpm=600.0, teeth=20, load=load, wear=0.0, eccentricity=0.0, backlash=0.0))
        return _db(sp, fr, 200.0), _db(sp, fr, 400.0)
    (a1, a2), (b1, b2) = lines(0.1), lines(1.0)
    assert 7.0 < (b1 - b2) - (a1 - a2) < 17.0      # 21 - 7 = 14 dB at the source, less once the housing is mixed in


def test_slip_adds_up_for_rolling_elements_and_not_for_teeth():
    # Randall & Antoni 2011, section 2.1: the spacing is the random variable (model 2), not the position.
    n, period = 20 * SR, 480
    phase = np.arange(n) / period
    def late(walk):
        hits = np.flatnonzero(np.diff(c._impacts(n, phase, np.random.default_rng(1), slip=0.02, width=2, walk=walk)) > 0)
        return np.abs(hits[-200:] - np.round(hits[-200:] / period) * period).mean() / period
    assert late(False) < 0.03 and late(True) > 0.1


def test_identify_names_a_ball_defect_by_its_cage_rate_and_ignores_mains_hum():
    from genny.identify import identify_machine
    rng = np.random.default_rng(0)
    t = np.arange(10 * 44100) / 44100
    hiss = rng.standard_normal(len(t))
    # A band that pulses at 100 Hz beside a 24 Hz shaft is a motor's hum (SUBF), not a race defect at order 4.17.
    hum = hiss * np.maximum(np.cos(2 * np.pi * 100.0 * t), 0.0) ** 8 + 0.5 * np.sin(2 * np.pi * 50.0 * t)
    assert identify_machine(hum, 44100, rpm=1440.0)["kind"] != "bearing"
    # Bursts once per cage turn (0.4 of the shaft rate), nothing at a ball-pass rate: the FSTF ball recordings.
    cage = hiss * np.maximum(np.cos(2 * np.pi * 0.4 * 20.0 * t), 0.0) ** 8 + 0.2 * np.sin(2 * np.pi * 20.0 * t)
    r = identify_machine(cage, 44100, rpm=1200.0)
    assert (r["kind"], r["params"]["state"]) == ("bearing", "ball") and abs(r["evidence"]["cage_order"] - 0.4) < 0.01


# ---------------------------------------------------------------- pipe_engine: firing, banks, cycles (L05)

def test_even_offsets_render_the_default_firing():
    a = c.pipe_engine(sr=SR, dur=0.5, rpm=2000.0, cylinders=4, unequal=0.0)
    b = c.pipe_engine(sr=SR, dur=0.5, rpm=2000.0, unequal=0.0, offsets=[0.0, 0.75, 0.5, 0.25])
    assert np.array_equal(a, b)


def test_offsets_are_firing_times_in_order():
    # Reversing the order of the same intervals is the time reversal of the firing: 180/540 is 540/180 rotated, so same lines.
    sp, fr = _spec(c.pipe_engine(sr=SR, dur=4.0, rpm=2400.0, offsets=[0.0, 0.25], unequal=0.0))
    assert _db(sp, fr, 60.0) > _db(sp, fr, 40.0) + 15


def test_lope_law_on_two_cylinders():
    for lope, off in ((1.0, 0.25), (0.5, 0.375), (0.25, 0.4375)):
        C = 2
        offs = (np.arange(C) / C - (np.arange(C) % 2) * 0.5 * lope / C) % 1.0
        assert offs[1] == pytest.approx(off)


def test_curve_of_constant_rpm_matches_the_number():
    a = c.pipe_engine(sr=SR, dur=0.5, rpm=2000.0)
    b = c.pipe_engine(sr=SR, dur=0.5, rpm=np.full(50, 2000.0))
    assert np.allclose(a, b, atol=1e-9)


def test_rpm_curve_moves_the_firing_rate():
    y = c.pipe_engine(sr=SR, dur=2.0, rpm=np.array([1200.0, 4800.0]), cylinders=4, unequal=0.0)
    sp, fr = _spec(y[-SR // 8:])
    m = (fr > 60) & (fr < 300)
    assert abs(fr[m][np.argmax(sp[m])] - 160.0) < 25.0


def test_falling_curve_can_backfire():
    n = c._n(1.0)
    fall = np.linspace(4000.0, 1500.0, n)
    assert not np.array_equal(c.pipe_engine(sr=SR, dur=1.0, rpm=fall, backfire=0.9), c.pipe_engine(sr=SR, dur=1.0, rpm=fall))
    rise = fall[::-1]
    assert np.array_equal(c.pipe_engine(sr=SR, dur=1.0, rpm=rise, backfire=0.9), c.pipe_engine(sr=SR, dur=1.0, rpm=rise))


def test_two_banks_keep_the_firing_rate():
    assert np.isfinite(c.pipe_engine(sr=SR, dur=0.5, cylinders=8, banks="LRLLRLRR")).all()
    assert np.isfinite(c.pipe_engine(sr=SR, dur=0.5, cylinders=8, banks=[0, 1] * 4)).all()
    with pytest.raises(ValueError):
        c.pipe_engine(sr=SR, dur=0.5, cylinders=8, banks="LRLR")
    with pytest.raises(ValueError):
        c.pipe_engine(sr=SR, dur=0.5, cylinders=4, banks="LLLL")


def test_headers_per_slot():
    a = c.pipe_engine(sr=SR, dur=0.5, cylinders=4, headers=[0.958, 0.333, 0.708, 0.958])
    assert np.isfinite(a).all() and not np.array_equal(a, c.pipe_engine(sr=SR, dur=0.5, cylinders=4))
    with pytest.raises(ValueError):
        c.pipe_engine(sr=SR, dur=0.5, cylinders=4, headers=[1.0, 2.0])


def test_cycles_set_the_firing_rate():
    for cycle, rate in ((4, 25.0), (2, 50.0), ("rotary", 100.0)):
        sp, fr = _spec(c.pipe_engine(sr=SR, dur=4.0, rpm=3000.0, cylinders=1, cycle=cycle, unequal=0.0))
        f = rate if cycle != "rotary" else 3 * 3000.0 / 180.0
        assert _db(sp, fr, f) > _db(sp, fr, 1.5 * f) + 15, cycle
    with pytest.raises(ValueError):
        c.pipe_engine(sr=SR, dur=0.5, cycle=3)


def test_sixteen_cylinders_are_allowed():
    fire = 16 * 2000.0 / 120.0
    sp, fr = _spec(c.pipe_engine(sr=SR, dur=2.0, rpm=2000.0, cylinders=16, unequal=0.0))
    assert _db(sp, fr, fire) > _db(sp, fr, 1.5 * fire) + 15


def test_tyre_lateral_force_is_the_brush_tyre():
    from genny import cyber as C
    F, mu, k = 4000.0, 1.0, 60e3
    a = 1e-4
    assert abs(float(C.tyre_lateral_force(a, F, mu, k)) / np.tan(a) - k) < 0.01 * k
    sl = np.arctan(3 * mu * F / k)
    assert abs(float(C.tyre_lateral_force(sl, F, mu, k)) - mu * F) < 1e-6
    assert abs(float(C.tyre_lateral_force(0.5, F, mu, k)) - mu * F) < 1e-6
    t = 0.5
    assert abs(float(C.tyre_lateral_force(np.arctan(t * 3 * mu * F / k), F, mu, k)) - 3 * mu * F * t * (1 - t + t * t / 3)) < 1e-6
    assert float(C.tyre_lateral_force(-0.05, F, mu, k)) == -float(C.tyre_lateral_force(0.05, F, mu, k))
