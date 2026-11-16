"""Lot 34 O4: Addendum livres lines 311-355 (analysis, mastering, corrections) and the leftovers of O2. One check per changed row, on the rendered signal.

Run:  uv run --with pytest pytest tests/test_lot34_o4.py -q
"""
import math
import tempfile
from pathlib import Path

import numpy as np

from genny import analysis as AN, compose, designed, dsp, export, fdstring, fx, nature, plates, reverbs
from genny.core import read_wav, write_wav

SR = 44100


def _tone(f, n=SR, a=0.5):
    return a * np.sin(2 * np.pi * f * np.arange(n) / SR)


def _db(y, f):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return 20 * math.log10(S[int(round(f * len(y) / SR))] + 1e-12)


def test_free_crash_does_not_run_into_the_limit():
    info = {}
    y = plates.nl_plate(sr=SR, vel=1.0, force=400.0, quality=0.5, edges="free", info=info, dur=0.3)
    assert np.isfinite(y).all() and info["guard_hits"] < 100          # was 7500 per 0.5 s: rigid tilt drifted to the limit
    info = {}
    plates.nl_plate(sr=SR, vel=1.0, force=400.0, quality=0.5, edges="free", info=info, dur=0.3, ulim=1e9)
    assert info["guard_hits"] == 0 and info["peak_u"] < 50.0           # the limit is exposed, and the drift is gone


def test_hard_friction_picks_the_branch_of_the_previous_velocity():
    h = fdstring.bowed_string_fd(220.0, 0.4, SR, friction="hard", residual=0.2)
    s = fdstring.bowed_string_fd(220.0, 0.4, SR)
    assert np.isfinite(h).all() and np.abs(h).max() > 0.01 and not np.allclose(h[:len(s)], s[:len(h)])


def test_rhodes_fd_elastic_clamp():
    a = fdstring.rhodes_fd(262.0, 0.6, SR, clamp=60.0)
    b = fdstring.rhodes_fd(262.0, 0.6, SR, clamp=12.0, base=8.0)
    assert np.isfinite(a).all() and np.isfinite(b).all() and not np.allclose(a[:min(len(a), len(b))], b[:min(len(a), len(b))])
    S = np.abs(np.fft.rfft(a * np.hanning(len(a)), 1 << 17))
    f = np.fft.rfftfreq(1 << 17, 1 / SR)
    S[f < 100] = 0
    assert abs(f[int(np.argmax(S))] - 262.0) < 0.06 * 262.0              # retuned: the note stays the note
    bar = fdstring.Bar(262.0, SR, ("clamped", "free"), None, 32).clamp("left", 15 * 262.0, 8.0)
    assert bar.bcl == fdstring.FR


def test_rod_cage_order_and_big_fountain():
    r = designed.rod_cage(SR, pendulums=2, dur=3.0, driver="washer", ball="none")
    q = designed.rod_cage(SR, pendulums=2, dur=3.0, driver="washer", ball="none", order="random")
    assert np.isfinite(r).all() and not np.allclose(r[:min(len(r), len(q))], q[:min(len(r), len(q))])
    for j in (28, 56):
        y = nature.fountain(SR, jets=j, dur=2.0)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.1


def test_audience_absorption_line_257():
    assert [round(reverbs.audience_absorption(a)) for a in (100, 200, 300)] == [95, 140, 165]
    assert abs(reverbs.audience_absorption(200, 250.0) - 70.0) < 1e-6
    assert reverbs.audience_absorption(150) > 95 and reverbs.audience_absorption(0) == 0.0


def test_radio_rotator_off_by_default_and_expander_direction():
    x = _tone(100.0, SR) + 0.3 * _tone(2500.0, SR)
    a, b = fx.radio(x, SR), fx.radio(x, SR, rotator=True)
    assert np.isfinite(a).all() and not np.allclose(a, b)
    quiet = np.concatenate([_tone(1000.0, SR // 2, 0.9), _tone(1000.0, SR // 2, 0.05)])
    down = fx.expander(quiet, SR, thresh=0.3, slope=3.0, direction="down", attack=0.005, release=0.02)
    up = fx.expander(quiet, SR, thresh=0.3, slope=2.0, direction="up")
    r = lambda y: float(np.sqrt(np.mean(y[-SR // 4:] ** 2)))
    assert r(down) < 0.5 * r(quiet) and abs(r(up) - r(quiet)) < 0.5 * r(quiet) + 1e-6
    assert abs(fx.expander(quiet, SR, thresh_db=-6.0, direction="up")[:100] - fx.expander(quiet, SR, thresh=0.5, direction="up")[:100]).max() < 1e-9


def test_limiter_input_gain_and_duck_hold():
    x = _tone(440.0, SR, 0.3)
    lo, hi = fx.limiter(x, SR), fx.limiter(x, SR, input_db=12.0)
    assert np.abs(hi).max() <= 10 ** (-0.5 / 20) + 1e-3 and np.abs(hi).max() > 3 * np.abs(lo).max()
    key = np.concatenate([_tone(300.0, SR // 4, 0.8), np.zeros(SR // 4), _tone(300.0, SR // 2, 0.8)])
    music = _tone(800.0, key.size, 0.5)
    g0 = fx.duck(music, SR, key=key, release=0.02)
    g1 = fx.duck(music, SR, key=key, release=0.02, hold=0.3)
    gap = slice(SR // 4 + 4000, SR // 4 + 8000)
    assert np.abs(g1[gap]).max() < 0.8 * np.abs(g0[gap]).max()


def test_tape_bias_adds_even_harmonics():
    x = _tone(1000.0, SR, 0.5)
    kw = dict(wow=0.0, flutter=0.0, hiss=-120, bump=0, hf=0, drive=2.0)
    a = dsp.tape(x, SR, **kw)
    b = dsp.tape(x, SR, bias=0.3, **kw)
    assert _db(a, 2000.0) < _db(a, 1000.0) - 60.0 and _db(b, 2000.0) > _db(b, 1000.0) - 40.0


def test_duration_loudness_and_maxima_slope():
    off = [AN.duration_loudness_offset(d) for d in (0.02, 0.1, 0.2, 0.6)]
    assert [round(v, 2) for v in off] == [-7.0, -2.5, -1.0, 0.0]
    assert AN.duration_loudness_offset(0.005) < -7.0
    burst = _tone(1000.0, SR // 50, 0.5)
    assert AN.short_loudness(burst, SR) < AN.loudness(burst, SR, window=1.0 / 50) - 6.0
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 220.0 * t) * 10 ** (-20.0 * t / 20)
    assert abs(AN.maxima_slope(y, SR, 220.0) + 20.0) < 2.0
    assert AN.TRANSIENT_MS == (5.0, 50.0) and AN.ATTACK_AUDIBLE_DB == -10.0 and AN.ROUGHNESS_BARK == 0.25


def test_read_write_formats_and_silence():
    import soundfile as sf
    d = Path(tempfile.mkdtemp())
    x = _tone(440.0, 4410, 0.5)
    sf.write(str(d / "f.wav"), x, SR, subtype="FLOAT")
    sf.write(str(d / "a.aiff"), x, SR, subtype="PCM_24")
    for n in ("f.wav", "a.aiff"):
        y, sr = read_wav(d / n)
        assert sr == SR and np.abs(y - x).max() < 1e-4
    write_wav(d / "z.wav", np.zeros(2000), SR)
    z, _ = read_wav(d / "z.wav")
    assert not np.any(z)                                                # digital silence stays at code 0 under dither
    ints = AN.active_bits(np.round(x * 32767) / 32767)
    assert 12 <= ints <= 16 and AN.polarity(np.concatenate([np.zeros(5), [0.8, -0.5]])) == 1


def test_compressor_alias_floor():
    x = _tone(10000.0, SR, 0.9)
    y = fx.compressor(x, SR, threshold=-18, ratio=4.0, attack=0.0005, release=0.01)
    S = np.abs(np.fft.rfft(y[2000:2000 + 32768] * np.hanning(32768)))
    k = int(round(10000.0 * 32768 / SR))
    S[k - 40:k + 41] = 0
    assert 20 * math.log10(S.max() / (np.abs(np.fft.rfft(y[2000:2000 + 32768] * np.hanning(32768))).max())) < -55.0


def test_compose_floor_genre_table_and_fmod_script():
    out = compose.arrange({"style": "pop", "bars": 4, "intensity": 0.005, "seed": 1})
    assert sum(len([e for e in compose.generate(l, out["bpm"]) if e.get("pitch") is not None]) for l in out["layers"]) == 0
    assert all(s in compose.STYLES for v in compose.GENRE_STYLES.values() for s in v)
    assert compose.genre_styles("shooter")[0] == "rock" and "orchestral_heroic" in compose.genre_styles("shooter")
    assert compose.genre_styles("horror")[:1] == ("spooky",)
    js = export.fmod_import_script(["a/b.wav", "c.wav"], prefix="g_")
    assert js.count("importAudioFile") == 2 and "'g_b'" not in js and '"g_b"' in js and export.FMOD_PORT == 3663
