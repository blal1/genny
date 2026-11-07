"""Sounds added from the deep-reading report (reports/Lecture profonde genny 2026-10-29.md)."""
import numpy as np
import pytest

from genny.drums import REGISTRY as DRUMS
from genny.instruments import REGISTRY as INST

SR = 44100


def _peak_hz(y, lo, hi, sr=SR):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    k = (f > lo) & (f < hi)
    return f[k][np.argmax(S[k])]


def _band(y, lo, hi, sr=SR):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float(np.sum(S[(f > lo) & (f < hi)]))


def _call(reg, name):
    e = reg[name]
    return e.fn if hasattr(e, "fn") else e["fn"] if isinstance(e, dict) else e


def test_hat808_decay_and_highpass():
    hat = _call(DRUMS, "hat808")
    closed, opened = hat(SR), hat(SR, open=1)
    assert abs(len(closed) / SR - 0.09) < 0.002 and abs(len(opened) / SR - 0.51) < 0.002
    q = int(0.08 * SR)
    assert np.max(np.abs(closed[q - 200:q])) < 0.01 * np.max(np.abs(closed))   # -60 dB at Decay, near enough
    assert _band(opened, 5000, 20000) > 5 * _band(opened, 0, 2000)


def test_measured_bell_plays_the_written_note():
    from genny.foley import BELLS, bell_partial, church_bell
    assert BELLS["measured"] == "sy_bell4" and abs(bell_partial("sy_bell4") - 196.52) < 0.1
    assert abs(bell_partial("sy_bell4", "prime") / bell_partial("sy_bell4") - 2.0) < 0.02
    y = church_bell(220.0, 1.0, SR, type="measured")
    assert abs(_peak_hz(y, 150, 280) - 220.0) < 2.0
    short = church_bell(220.0, 1.0, SR, type="measured", decay=1.0)
    k = int(1.0 * SR)
    assert np.std(short[k:k + 4410]) < np.std(y[k:k + 4410])


def test_premier_timpani_has_the_measured_ratios():
    t = _call(INST, "timpani")
    y = t(147.0, 3.0, SR, model="premier", strike=0.75)
    late = y[int(0.5 * SR):]                                # (0, n) modes gone: the ringing (m, 1) family stays
    for f in (147.0, 245.0, 288.0, 355.0):
        assert abs(_peak_hz(late, f - 12, f + 12) - f) < 1.5
    assert abs(_peak_hz(t(98.0, 3.0, SR, model="premier")[int(0.5 * SR):], 85, 110) - 98.0) < 1.5


def test_padsynth_loops_and_spreads():
    p = _call(INST, "padsynth")
    y = p(220.0, 2.0, SR, attack=0.01)
    assert np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) <= 1.0
    mid = y[SR // 2:SR // 2 + 32768]
    assert abs(_peak_hz(mid, 180, 260) - 220.0) < 8.0
    S = np.abs(np.fft.rfft(mid * np.hanning(len(mid)))) ** 2
    f = np.fft.rfftfreq(len(mid), 1 / SR)
    wide = p(220.0, 2.0, SR, attack=0.01, bw=200.0)[SR // 2:SR // 2 + 32768]
    W = np.abs(np.fft.rfft(wide * np.hanning(len(wide)))) ** 2
    spread = lambda P: np.sqrt(np.sum(P[(f > 150) & (f < 300)] * (f[(f > 150) & (f < 300)] - 220) ** 2) / np.sum(P[(f > 150) & (f < 300)]))
    assert spread(W) > 1.5 * spread(S)


@pytest.mark.parametrize("name", ["marimba", "vibraphone"])
def test_tube_favours_the_fundamental(name):
    inst = _call(INST, name)
    dry = inst(440.0, 0.8, SR, resonator=0.0)
    wet = inst(440.0, 0.8, SR)
    ratio = lambda y: _band(y, 400, 480) / (_band(y, 1600, 1900) + 1e-30)
    assert ratio(wet) > ratio(dry)
    assert abs(_peak_hz(wet, 380, 500) - 440.0) < 4.0


def _ringing(kind, res, sr=SR):
    from genny.filters import ladder
    x = np.zeros(sr)
    x[0] = 0.01                                             # small: stays in the linear part of the tanh
    return ladder(x, 1000.0, sr, res, kind)


def test_moog_ladder_resonates_at_cutoff():
    assert abs(_peak_hz(_ringing("moog", 0.97), 300, 3000) - 1000.0) < 40.0


def test_diode_ladder_resonates_at_cutoff_over_sqrt2():
    assert abs(_peak_hz(_ringing("diode", 0.97), 300, 3000) - 1000.0 / np.sqrt(2.0)) < 40.0


@pytest.mark.parametrize("kind", ["moog", "diode"])
def test_ladder_self_oscillates_and_stays_bounded(kind):
    from genny.filters import ladder
    x = np.zeros(SR)
    x[0] = 0.01
    y = ladder(x, 800.0, SR, 1.05, kind)
    assert np.isfinite(y).all() and np.max(np.abs(y)) < 10.0
    assert np.std(y[-4410:]) > 10 * np.std(_ringing(kind, 0.5)[-4410:])
    white = np.random.default_rng(0).standard_normal(SR // 2)
    assert np.isfinite(ladder(white, np.linspace(50.0, 15000.0, SR // 2), SR, 1.0, kind)).all()


def test_diode_acid_renders():
    y = _call(INST, "acid")(110.0, 0.4, SR, model="diode", res=0.8)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.05


def test_dattorro_is_stereo_and_decays():
    from genny import fx
    d = fx.REGISTRY["dattorro"]["fn"]
    x = np.zeros(4410)
    x[0] = 1.0
    y = d(x, SR, mix=1.0, decay=0.5)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
    e = lambda a, b: float(np.mean(y[int(a * SR):int(b * SR)] ** 2))
    assert e(0.05, 0.25) > 10 * e(0.8, 1.0)
    assert abs(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) < 0.5        # two decorrelated outputs
    long = d(x, SR, mix=1.0, decay=0.9, tail=6.0)
    assert np.isfinite(long).all() and np.std(long[4 * SR:]) > 0.001 * np.std(long[:SR])
    assert 0.005 < np.argmax(np.abs(y[:, 0]) > 1e-6) / SR < 0.08 # first output after the diffusers and the shortest tap
    assert np.isfinite(d(x, 96000, mix=1.0)).all()


def test_tape_hysteresis_has_a_loop():
    from genny.dsp import _ja_kernel, ja_hysteresis
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 110 * t)
    y = _ja_kernel(x, 3.0, 380.0, 720.0, 380.0, 0.015, 0.25, 4)
    assert np.isfinite(y).all()
    rising = np.gradient(x) > 0
    i = (np.abs(x) < 0.05) & (t > 0.2)
    assert np.mean(y[i & rising]) < np.mean(y[i & ~rising]) - 1e-3   # the two branches cross zero at different levels
    assert np.isfinite(ja_hysteresis(np.stack([x, x], axis=1), SR, 2.0)).all()
    tape = __import__("genny.fx", fromlist=["REGISTRY"]).REGISTRY["tape"]["fn"]
    assert np.isfinite(tape(x, SR, model="hysteresis", hiss=-120)).all()


def test_priority_ducks_the_lower_layer():
    from genny.spec import render_layers
    pad = {"type": "synth", "inst": "sine", "notes": "A3", "dur": 2.0, "priority": 1}
    hit = {"type": "synth", "inst": "sine", "notes": "A5", "dur": 1.0, "at": 0.5, "priority": 3}
    y = render_layers([pad, hit], SR)
    a = y[int(0.1 * SR):int(0.4 * SR)]
    b = y[int(1.0 * SR):int(1.3 * SR)]
    ratio = _band(b, 200, 240) / len(b) / (_band(a, 200, 240) / len(a))
    assert abs(10 * np.log10(ratio) + 6.0) < 1.5                    # two steps = 6 dB
    assert np.array_equal(render_layers([pad], SR), render_layers([{k: v for k, v in pad.items() if k != "priority"}], SR))


def test_gverb_and_binaural():
    from genny import fx
    x = np.zeros(4410)
    x[0] = 1.0
    g = fx.REGISTRY["gverb"]["fn"](x, SR, revtime=2.0)
    assert g.shape[1] == 2 and np.isfinite(g).all()
    e1, e2 = np.std(g[int(0.1 * SR):int(0.3 * SR)]), np.std(g[int(1.6 * SR):int(1.8 * SR)])
    assert 10 < e1 / max(e2, 1e-12) < 3000                          # about -60 dB per 2 s
    b = fx.REGISTRY["binaural"]["fn"]
    n = np.random.default_rng(0).standard_normal(SR // 2)
    y = b(n, SR, azimuth=-90.0)
    assert np.std(y[:, 0]) > 1.2 * np.std(y[:, 1])                  # left ear nearer
    t = np.arange(8820) / SR
    z = b(np.sin(2 * np.pi * 300 * t), SR, azimuth=-90.0)[2000:]
    lag = np.argmax(np.correlate(z[:, 0], z[:, 1], "full")) - (len(z) - 1)
    assert -40 < lag < -8                                           # the left ear hears it first, about 0.6 ms
    assert abs(np.std(b(n, SR)[:, 0]) / np.std(b(n, SR)[:, 1]) - 1.0) < 0.01


def test_spring_hf_branch_is_quiet_and_stable():
    from genny import fx
    x = np.zeros(SR // 2)
    x[0] = 1.0
    s = fx.REGISTRY["spring_reverb"]["fn"]
    assert np.isfinite(s(x, SR, mix=1.0, hf=1.0)).all() and not np.array_equal(s(x, SR, mix=1.0, hf=1.0), s(x, SR, mix=1.0, hf=0.0))


def test_saxofony_speaks_in_tune():
    from genny.physical import waveguides as wg
    d = 1.5
    n = int(d * wg.SR)
    y = wg.saxofony(np.full(n, 220.0), np.full(n, 0.7), d, np.random.default_rng(0))
    late = y[n // 2:]
    assert np.isfinite(y).all() and np.std(late) > 1e-3
    assert abs(_peak_hz(late, 150, 300, wg.SR) - 220.0) < 8.0


def test_sax_stk_instrument():
    y = _call(INST, "sax")(220.0, 1.0, SR, model="stk")
    assert np.isfinite(y).all() and abs(_peak_hz(y[SR // 4:], 150, 300) - 220.0) < 8.0


def test_engine_knock_frequency_and_kinds():
    from genny.sfx import REGISTRY as S
    k = S["engine_knock"]["fn"]
    y = k(SR, severity=1.0)
    assert abs(_peak_hz(y, 3000, 10000) - 6470.0) < 300.0
    assert _peak_hz(k(SR, kind="rod"), 100, 1000) < 450 and np.isfinite(k(SR, kind="slap")).all()


def test_steepen_adds_harmonics_with_level():
    from genny.dsp import steepen
    x = np.sin(2 * np.pi * 350 * np.arange(SR) / SR)
    h2 = lambda y: _band(y, 680, 720) / _band(y, 330, 370)
    assert h2(steepen(x, SR, 2.8, 15000.0)) > 100 * h2(steepen(x, SR, 2.8, 500.0))
    assert np.isfinite(steepen(x, SR, 2.8, 15000.0)).all()


def test_hibbert_bells():
    from genny.foley import _hibbert_modes, church_bell
    f, t, g = _hibbert_modes("hibbert", 350.0, 1270.3, "nominal", 8.0)   # nominal 350 Hz x 1
    f2, _, _ = _hibbert_modes("hibbert", 350.0, 1240.0, "strike", 8.0)
    assert abs(1200 * np.log2(f[8] / f[4]) - (1.668 * 1270.3 + 0.106)) < 0.01
    assert f2[4] < 700.0
    for kind in ("hibbert", "old_english", "medieval", "big_ben"):
        y = church_bell(220.0, 0.5, SR, type=kind, tune="strike")
        assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.01


def test_budgie_reaches_its_trachea():
    from genny.sfx import REGISTRY as S
    y = S["bird_call"]["fn"](SR, freq=2125.0, species="budgie", two_voice=0.0, dur=0.2)
    assert 2000.0 < _peak_hz(y, 500, 12000) < 8000.0


def test_shell_stiffens_the_plate():
    from genny import plates as P
    f = np.zeros(4000)
    f[0] = 1.0
    d = dict(force=2.0, contact_ms=2.0, decay=8.0, damping=0.0, dur=0.3, size=0.4, thickness=1.0, accel=False, hp=20.0, level=0.8)
    flat = P.nl_plate(SR, 1.0, **d)
    shell = P.nl_plate(SR, 1.0, dome=0.5, **d)
    want = 3750.0 / 0.5 / (2 * np.pi)   # sqrt(E/rho)/Rs, 1195 Hz for short waves; measured 480 on the lowest mode, which the edge relieves
    a, b = _peak_hz(flat, 30, 4000), _peak_hz(shell, 30, 4000)
    assert b > a and b > 0.3 * want and np.isfinite(shell).all()
    y = _call(DRUMS, "shell_crash")(SR, vel=0.5, dur=0.3, quality=0.5)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.01


GEO = ("rockfall", "landslide", "debris_flow", "singing_dune", "mud_pot", "geyser", "lava", "glacier_calving", "sand_storm",
       "earthquake", "tree_fall", "snowfall", "snow_avalanche")


@pytest.mark.parametrize("name", GEO)
def test_geo_sfx_render(name):
    from genny.sfx import REGISTRY as S
    y = S[name]["fn"](SR, dur=4.0)
    assert np.isfinite(y).all() and 0.05 < np.max(np.abs(y)) <= 1.0 and abs(len(y) / SR - 4.0) < 0.5


def test_dune_sings_at_its_frequency_and_geyser_rings_at_minnaert():
    from genny.sfx import REGISTRY as S
    assert abs(_peak_hz(S["singing_dune"]["fn"](SR, freq=105.0, dur=4.0), 50, 400) - 105.0) < 5.0
    y = S["geyser"]["fn"](SR, dur=6.0, erupt=1.0)
    assert 40.0 < _peak_hz(y, 20, 300) < 120.0                       # 3.26 / 0.05 = 65 Hz


def test_farin_scaling():
    from genny.geo import boulder
    l1, f1 = boulder(8.0, 2.0)
    l2, f2 = boulder(8.0, 8.0)
    assert abs(l2 / l1 - 4.0 ** 1.3) < 1e-6 and f2 > f1
    assert boulder(64.0, 2.0)[1] < f1


def test_psg_tables_and_shared_load():
    from genny import retro as R
    for d in R.PSG_DACS.values():
        assert d[0] == 0.0 and abs(d[15] - 1.0) < 1e-9 and (np.diff(d) > 0).all()
    y = _call(INST, "psg")(220.0, 0.3, SR, dac="ym2149")
    assert abs(_peak_hz(y, 150, 300) - 220.0) < 6.0
    from genny.chiptune2 import stk_fm  # noqa: F401
    y3 = _call(INST, "psg3")(220.0, 0.3, SR)
    S = np.abs(np.fft.rfft(y3 * np.hanning(len(y3))))
    assert np.isfinite(y3).all() and S.max() > 0
    # a shared load squashes: three channels on are worth less than three times one
    ru, rd, res, off = R.PSG_CHIPS["ym2149"]
    g, l = 1 / res[15], 1 / 1000.0
    v = lambda m: (m / ru + m * g) / (3 / ru + 3 / rd + l + m * g + (3 - m) / res[0]) if False else (m * g) / (m * g + 3 / rd + l)
    assert v(3) < 3 * v(1)


def test_sfxr_link_round_trip_and_jumps():
    from genny import retro as R
    p = R.sfxr_params("laser", seed=3)
    raw = bytes([p["wave"]]) + np.array([p[k] for k in R.KNOBS], "<f4").tobytes()
    v, s = int.from_bytes(raw, "big"), ""
    while v:
        v, m = divmod(v, 58)
        s = R._B58[m] + s
    s = "1" * (len(raw) - len(raw.lstrip(b"\0"))) + s
    q = R.sfxr_link("https://sfxr.me/#" + s)
    assert q["wave"] == p["wave"] and all(abs(q[k] - p[k]) < 1e-6 for k in R.KNOBS)
    with pytest.raises(ValueError):
        R.sfxr_link("0OIl")
    assert abs(R.jump_knob(2.0) ** 2 * 0.9 - 0.5) < 1e-9
    kw = dict(wave="sine", hz=440.0, sustain=0.4, decay=0.0, soften=0.0, max_dur=0.3, seed=1)
    y0 = R.sfxr(SR, **kw)
    y1 = R.sfxr(SR, engine="bfxr", jump1=R.jump_knob(2.0), onset1=0.1, **kw)
    assert abs(_peak_hz(y0, 200, 1500) - 440.0) < 20.0 and abs(_peak_hz(y1[len(y1) // 2:], 200, 1500) - 880.0) < 40.0
    for name in ("laser", "explosion", "powerup", "hit", "jump", "blip", "pickup", "random"):
        y = R.sfxr(SR, preset=name, presets="bfxr", engine="bfxr", seed=2, mutate=1)
        assert np.isfinite(y).all()


def test_zzfx_mutate_and_bytebeat():
    from genny import retro as R
    rng = np.random.default_rng(1)
    a = R._zz_array(None, {}, None)
    b = R.zz_mutate(a, rng)
    assert (a != b).any() and b[R.ZZ_SLOTS.index("volume")] == a[R.ZZ_SLOTS.index("volume")] and b[R.ZZ_SLOTS.index("randomness")] == a[R.ZZ_SLOTS.index("randomness")]
    y = R.zzfx(SR, preset="pickup" if "pickup" in R.ZZFX_PRESETS else None, mutate=2, seed=4)
    assert np.isfinite(y).all()
    t = np.arange(40000, dtype=np.int64)
    assert all(((f(t) & 0xFFFF) >= 0).all() for f in R._BYTEBEATS)
    for i in range(7):
        y = R.bytebeat(SR, formula=i, dur=1.0)
        assert np.isfinite(y).all() and len(y) == SR


def test_stk_fm_voices_and_tables():
    from genny.chiptune2 import STK_FM_GAINS, STK_FM_SUS, _STK_FM
    assert STK_FM_GAINS[99] == 1.0 and abs(20 * np.log10(STK_FM_GAINS[98]) + 0.602) < 0.002 and abs(20 * np.log10(STK_FM_SUS[14]) + 3.01) < 0.002
    for v in _STK_FM:
        y = _call(INST, "stk_fm")(220.0, 0.6, SR, voice=v)
        assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.05
    y = _call(INST, "stk_fm")(220.0, 1.0, SR, voice="b3")
    assert abs(_peak_hz(y[SR // 4:], 150, 300) - 220.0) < 4.0
    y = _call(INST, "stk_fm")(220.0, 1.0, SR, voice="rhodey")
    e = np.abs(y)
    assert e[: SR // 4].max() > 1.5 * e[SR // 2 + SR // 4: SR].max()     # linear 1.5 s decay: about 0.4 by then


def test_rankine_hugoniot_kinney_example_and_strokes():
    from genny.physics import rankine_hugoniot
    ms, wind, tr = rankine_hugoniot(10.0 * 6894.76)
    assert abs(ms - 1.258) < 0.001 and abs(wind / 0.3048 - 436.0) < 6.0 and 1.1 < tr < 1.2
    from genny.sfx import REGISTRY as S
    y = S["lightning"]["fn"](SR, dur=2.0, distance=100.0, strokes=3)
    assert np.isfinite(y).all() and np.max(np.abs(y)) <= 1.0


def test_paulstretch_options():
    from genny.fx import REGISTRY as X
    x = np.sin(2 * np.pi * 440.0 * np.arange(SR) / SR) * 0.5
    ps = X["paulstretch"]["fn"]
    y = ps(x, SR, stretch=2.0, window=0.1, phase=0.0)
    assert abs(_peak_hz(y, 200, 1000) - 440.0) < 5.0
    y = ps(x, SR, stretch=2.0, window=0.1, continuity=0.3, loop=True)
    assert np.sqrt(np.mean(y[-SR // 4:] ** 2)) > 0.3 * np.sqrt(np.mean(y ** 2))
    sw = np.sin(2 * np.pi * (200.0 + 400.0 * np.arange(SR) / SR) * np.arange(SR) / SR)
    a = ps(sw, SR, hold=0.1, length=1.0, window=0.1, scan=0.0)
    b = ps(sw, SR, hold=0.1, length=1.0, window=0.1, scan=0.8)
    assert abs(_peak_hz(b[-SR // 4:], 100, 3000) - _peak_hz(a[-SR // 4:], 100, 3000)) > 100.0
    for r in ("subtle", "shimmer_deep"):
        y = ps(x, SR, stretch=2.0, window=0.1, recipe=r)
        assert np.isfinite(y).all() and np.max(np.abs(y)) <= 0.51


def test_serge_fold_and_soft_sync():
    from genny.fx import REGISTRY as X
    x = np.sin(2 * np.pi * 110.0 * np.arange(SR // 2) / SR) * 0.9
    y = X["serge_fold"]["fn"](x, SR, drive=1.0)
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert np.isfinite(y).all() and S[f > 1000].max() > 0.05 * S.max()
    small = X["serge_fold"]["fn"](x, SR, drive=0.03, stages=1)
    assert np.corrcoef(small, x)[0, 1] > 0.99                        # below the diodes: a wire
    a = _call(INST, "soft_sync")(220.0, 0.5, SR, ratio=2.0, window=0.0)
    assert abs(_peak_hz(a, 100, 1000) - 440.0) < 8.0                 # window 0: free running
    b = _call(INST, "soft_sync")(220.0, 0.5, SR, ratio=1.7, window=0.5)
    assert abs(_peak_hz(b, 100, 300) - 220.0) < 8.0                  # hard sync locks to the master


def test_french_sirens_footsteps_hummer_blast():
    from genny.sfx import REGISTRY as S
    from genny.physics import blast_ground
    fn = S["siren_wail"]["fn"]
    y = fn(SR, kind="pompiers", dur=4.0, echo=0.0)
    Y = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    i, j = np.argmin(np.abs(f - 435)), np.argmin(np.abs(f - 488))
    assert Y[i - 8: i + 8].max() > 3 * Y[np.argmin(np.abs(f - 460))] and Y[j - 8: j + 8].max() > 3 * Y[np.argmin(np.abs(f - 460))]
    b = np.abs(y[: int(4.0 * SR)])
    a = fn(SR, kind="ambulance_fr", dur=5.0, echo=0.0)
    w = int(1.3 * SR)
    assert np.max(np.abs(a[w: w + SR // 3])) < 0.05 * np.max(np.abs(a))
    for s in ("gravel", "snow", "wood", "dirt", "grass"):
        y = S["pd_footstep"]["fn"](SR, surface=s, steps=2)
        assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.05, s
    h = S["hummer"]["fn"](SR, spin=1.5, dur=2.0)
    p = _peak_hz(h, 100, 3000)
    assert abs(p / (343.0 / 1.6) - round(p / (343.0 / 1.6))) < 0.08
    assert 195.0 < _peak_hz(S["snow_squeal"]["fn"](SR, dur=1.0), 100, 400) < 205.0
    assert np.isfinite(S["blast_noise"]["fn"](SR)).all()
    assert abs(blast_ground(1.0, 3.0) / blast_ground(1.0, 2.999) - 1.0) < 0.05 and blast_ground(1.0, 1.0) > blast_ground(1.0, 10.0)


def test_chowning_jaw_harp_cuica():
    from genny.sfx import REGISTRY as S
    for v, h in (("clarinet", 3), ("woodwind", 3), ("bassoon", 5)):
        y = _call(INST, "chowning")(300.0, 0.8, SR, voice=v)
        assert np.isfinite(y).all()
        Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1 / SR)
        assert Y[np.argmin(np.abs(f - 300 * h))] > 0.1 * Y.max()     # carrier partial is there
    y = _call(INST, "jaw_harp")(110.0, 1.0, SR)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.05
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert Y[(f > 400) & (f < 2500)].sum() > Y[f < 200].sum()        # the moving mouth carries more than the drone
    y = S["cuica"]["fn"](SR)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.05
