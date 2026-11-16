import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import jv

from genny import analysis as AN
from genny import fx as FX
from genny import instruments as INS
from genny import sfx as SFX
from genny import synth_s as S

SR = 44100
CAR_COUNT = {1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 1, 17: 1, 18: 1, 19: 3, 20: 3,
             21: 4, 22: 4, 23: 4, 24: 5, 25: 5, 26: 3, 27: 3, 28: 3, 29: 4, 30: 4, 31: 5, 32: 6}


def _spec(y, sr=SR):
    return np.abs(np.fft.rfft(y * np.hanning(len(y)))), np.fft.rfftfreq(len(y), 1 / sr)


def _peak_hz(y, sr=SR):
    sp, f = _spec(y, sr)
    k = int(np.argmax(sp))
    a, b, c = np.log(sp[k - 1] + 1e-12), np.log(sp[k] + 1e-12), np.log(sp[k + 1] + 1e-12)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / len(y)


def _harm(y, f0, k, sr=SR):
    n = len(y)
    t = np.arange(n) / sr
    return 2 * abs(np.sum(y * np.exp(-2j * np.pi * k * f0 * t))) / n


def _solo(coarse=1.0, ol=99, alg=32, **kw):
    ops = [S._op(l=(99, 99, 99, 0), ol=ol, coarse=coarse, **kw)] + [S._OFF] * 5
    return S.dx7_voice("SOLO", ops, alg=alg)


def test_algorithm_tables():
    for n, c in CAR_COUNT.items():
        car, mm, fr, fs = S.dx7_algorithm(n)
        assert int(car.sum()) == c, n
        assert np.all(np.triu(mm, 1) == mm), n
        assert 0 <= fr <= 5 and fs >= fr or (n in (4, 6)), n
        assert car[0] == 1.0, n
    assert len(S.VELOCITY_LUT) == 64 and len(S.EXP_SCALE) == 33 and len(S.LEVEL_LUT) == 20


def test_operator_frequencies():
    hz = 441.0
    for coarse, fine, want in ((1, 0, 441.0), (2, 0, 882.0), (0, 0, 220.5), (3, 50, 3 * 1.5 * 441.0)):
        y = S.render_dx7(_solo(coarse, fine=fine), hz, 1.0, SR)[SR // 4:SR // 4 + 22050]
        assert abs(_peak_hz(y) - want) < 1.0, (coarse, fine)
    y = S.render_dx7(_solo(2, mode=1, fine=50), hz, 1.0, SR)[SR // 4:SR // 4 + 22050]
    assert abs(_peak_hz(y) - 10 ** 2.5) < 1.0
    lo = S.render_dx7(_solo(1, det=0), hz, 1.0, SR)[SR // 4:SR // 4 + 32768]
    hi = S.render_dx7(_solo(1, det=14), hz, 1.0, SR)[SR // 4:SR // 4 + 32768]
    assert abs(_peak_hz(hi) / _peak_hz(lo) - 2 ** (14 * S.DET_CENTS / 1200)) < 2e-4


def test_fm_index_matches_bessel():
    ops = [S._op(ol=99, coarse=1), S._op(ol=91, coarse=1)] + [S._OFF] * 4
    y = S.render_dx7(S.dx7_voice("FM", ops, alg=1), 441.0, 1.0, SR)[SR // 2:SR // 2 + 22050]
    amp = 10 ** (32 * (S.dx7_out_level(91) - 127) * S.DX7_STEP_DB / 20)
    beta = S.DX7_MAX_INDEX * amp
    assert abs(beta - 2 * math.pi) < 0.01
    for h in range(1, 7):
        want = abs(jv(h - 1, beta) + (-1) ** h * jv(h + 1, beta))
        assert abs(_harm(y, 441.0, h) * math.sqrt(2.0) - want) < 0.01, h


def test_feedback_index():
    v = S.dx7_voice("FB", [S._OFF] * 5 + [S._op(ol=99, coarse=1)], alg=32, fb=7)
    cut = slice(SR // 2, SR // 2 + 22050)
    flat = S.render_dx7(v, 441.0, 1.0, SR, fb=0)[cut]
    fb = S.render_dx7(v, 441.0, 1.0, SR, fb=7)[cut]
    assert abs(_harm(flat, 441.0, 2)) < 1e-3
    assert _harm(fb, 441.0, 2) > 0.05 * _harm(fb, 441.0, 1) and _harm(fb, 441.0, 3) > 0.05 * _harm(fb, 441.0, 1)
    h = [sum(_harm(S.render_dx7(v, 441.0, 1.0, SR, fb=k)[cut], 441.0, m) ** 2 for m in range(2, 12)) for k in (1, 3, 5)]
    assert h[0] < h[1] < h[2]


def test_envelope_rates_and_levels():
    q = S.dx7_qrate(50)
    assert q == 32 and abs(S.dx7_rate_db_per_s(0) - 0.2819) < 1e-9 and abs(S.dx7_rate_db_per_s(4) - 2 * 0.2819) < 1e-9
    assert S.dx7_out_level(99) == 127 and S.dx7_out_level(20) == 48 and S.dx7_out_level(0) == 0
    low = 4096 - (127 - S.dx7_out_level(60)) * 32
    tgt = np.array([4096.0, low, low, low])
    eg = S._dx7_eg(4 * SR, float(SR), np.array([63, q, q, q], np.int64), tgt, 4 * SR)
    reach = int(np.argmax(eg <= low + 1e-9))
    want = (4096 - low) * S.DX7_STEP_DB / S.dx7_rate_db_per_s(q)
    assert abs(reach / SR - want) / want < 0.01
    up = S._dx7_eg(SR // 4, float(SR), np.array([40, 40, 40, 40], np.int64), np.array([4096.0, 4096.0, 4096.0, 272.0]), SR)
    assert up[0] >= 272 + S.DX7_ATTACK_JUMP and np.all(np.diff(up) >= 0)


def test_velocity_and_key_scaling():
    assert S.dx7_velocity(127, 0) == 0 and S.dx7_velocity(127, 7) > S.dx7_velocity(40, 7)
    assert S.dx7_key_scale(60, 39, 99, 99, 0, 0) == 0
    assert S.dx7_key_scale(84, 39, 0, 99, 0, 0) < 0 < S.dx7_key_scale(84, 39, 0, 99, 3, 3)
    assert S.dx7_key_scale(36, 39, 99, 0, 2, 0) > 0
    v1 = _solo(1, kvs=7)
    r = [np.sqrt(np.mean(S.render_dx7(v1, 441.0, 1.0, SR, vel=x)[SR // 4:SR // 2] ** 2)) for x in (1.0, 0.5)]
    steps = S.dx7_velocity(127, 7) - S.dx7_velocity(64, 7)
    assert abs(20 * math.log10(r[0] / r[1]) - steps * S.DX7_STEP_DB) < 0.2
    ks = _solo(1, rd=99, rc=0, bp=39)
    a = np.sqrt(np.mean(S.render_dx7(ks, 440.0 * 2, 1.0, SR)[SR // 4:SR // 2] ** 2))
    b = np.sqrt(np.mean(S.render_dx7(ks, 440.0 / 2, 1.0, SR)[SR // 4:SR // 2] ** 2))
    d = S.dx7_key_scale(81, 39, 0, 99, 0, 0) - S.dx7_key_scale(57, 39, 0, 99, 0, 0)
    assert abs(20 * math.log10(a / b) - 32 * d * S.DX7_STEP_DB) < 0.2


def test_sysex_roundtrip_and_checksum():
    ex = S.dx7_voice("EXTREME", [S._op((1, 2, 3, 4), (5, 6, 7, 8), 77, 31, 99, 14, 1, 99, 99, 98, 3, 2, 7, 3, 7) for _ in range(6)], alg=32,
                     fb=7, oks=0, pr=(1, 2, 3, 4), pl=(5, 6, 7, 8), lfo_speed=99, lfo_delay=98, pmd=97, amd=96, lfo_sync=0, lfo_wave=5, pms=7,
                     transpose=48)
    voices = list(S.DX7_PRESETS.values()) + [ex]
    raw = S.dx7_make_sysex(voices, channel=3)
    assert len(raw) == 4104 and raw[:6] == bytes([0xF0, 0x43, 3, 9, 0x20, 0x00]) and raw[-1] == 0xF7
    assert raw[-2] == (-sum(raw[6:-2])) & 0x7F
    back = S.dx7_parse_sysex(raw)
    assert len(back) == 32
    for a, b in zip(voices, back):
        assert a == b
    one = S.dx7_make_single(ex)
    assert len(one) == 163 and one[:6] == bytes([0xF0, 0x43, 0, 0, 1, 0x1B])
    assert S.dx7_parse_sysex(one)[0] == ex
    bad = bytearray(raw)
    bad[100] ^= 1
    with pytest.raises(Exception):
        S.dx7_parse_sysex(bytes(bad))
    assert S.dx7_parse_sysex(raw[6:-2])[2] == back[2]


def test_dx7_renders_all_algorithms_and_presets():
    for n in range(1, 33):
        ops = [S._op(ol=90, coarse=c) for c in (1, 1, 2, 1, 3, 1)]
        y = S.render_dx7(S.dx7_voice("A", ops, alg=n, fb=4), 220.0, 0.3, SR)
        assert np.all(np.isfinite(y)) and np.sqrt(np.mean(y ** 2)) > 0.01, n
    for name, v in S.DX7_PRESETS.items():
        y = INS.REGISTRY["dx7"]["fn"](220.0, 0.5, SR, 0.9, patch=name)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.01 and np.max(np.abs(y)) < 2.0, name
        assert len(AN.detect_clicks(y[:SR // 4:1] * 0 + y[SR // 10:SR // 10 + SR // 4])) < 5, name
    lfo = S.dx7_voice("L", [S._op(ol=99)] + [S._OFF] * 5, alg=32, pmd=99, pms=7, lfo_speed=60, amd=50)
    assert np.all(np.isfinite(S.render_dx7(lfo, 220.0, 0.5, SR)))


def test_buchla_fold_matches_theory():
    f0, A = 441.0, 2.5
    x = np.sin(2 * np.pi * f0 * np.arange(SR) / SR)
    y = FX.REGISTRY["buchla_fold"]["fn"](x, SR, timbre=A, cells=1, gain=1.0, symmetry=0.0, knee=0.0, tone=0.0, mix=1.0)
    m = np.arange(1, 801, 2)
    for k in (1, 3, 5, 7, 9):
        th = 16 / math.pi ** 2 * np.sum(((-1.0) ** ((m - 1) // 2)) * jv(k, m * math.pi * A / 2) / m ** 2)
        assert abs(_harm(y, f0, k) - abs(th)) < 0.01, k
    assert abs(_harm(y, f0, 2)) < 1e-3
    y2 = FX.REGISTRY["buchla_fold"]["fn"](x, SR, timbre=A, cells=3, gain=2.0, mix=1.0)
    assert np.all(np.isfinite(y2)) and np.max(np.abs(y2)) <= 1.01
    assert abs(_harm(y2, f0, 5)) > 0.01
    sym = FX.REGISTRY["buchla_fold"]["fn"](x, SR, timbre=A, cells=2, gain=1.5, symmetry=0.4, mix=1.0)
    assert _harm(sym, f0, 2) > 0.02


def test_lpg_time_constants_and_response():
    cv = np.zeros(SR)
    cv[2000:] = 1.0
    s, g = S.lpg_light(cv, SR, up=0.0025, thermal=0.15, weight=0.25)
    i = 2000 + int(0.0025 * SR)
    assert abs(s[i] - (1 - 1 / math.e)) < 0.01
    cv2 = np.ones(SR // 2)
    cv2[SR // 4:] = 0.0
    s2, g2 = S.lpg_light(np.concatenate([np.ones(SR), np.zeros(SR)]), SR, up=0.0025, dn_fast=0.035, dn_slow=0.4)
    ref = solve_ivp(lambda t, s_: -s_ / (0.035 + (0.4 - 0.035) * (1 - s_)), [0, 1.0], [s2[SR - 1]], t_eval=[0.05, 0.2, 0.5], rtol=1e-8)
    for t, r in zip((0.05, 0.2, 0.5), ref.y[0]):
        assert abs(s2[SR + int(t * SR)] - r) < 0.01
    assert g2[SR + int(0.5 * SR)] > s2[SR + int(0.5 * SR)]
    ron, roff, rl, c = 1.5e3, 1e7, 1e5, 10e-9
    gg = 0.5
    r = roff * (ron / roff) ** gg
    fc = 1.0 / (2 * math.pi * (r * rl / (r + rl)) * c)
    f = 300.0
    t = np.arange(3 * SR) / SR
    x = np.sin(2 * np.pi * f * t)
    y = S.lpg_process(x, np.full(3 * SR, 0.5), SR, "combo", 1)
    meas = np.sqrt(np.mean(y[2 * SR:] ** 2)) / np.sqrt(np.mean(x[2 * SR:] ** 2))
    want = rl / (r + rl) / math.sqrt(1 + (f / fc) ** 2)
    assert abs(meas / want - 1) < 0.03
    vca = S.lpg_process(x, np.full(3 * SR, 0.5), SR, "vca", 1)
    assert abs(np.sqrt(np.mean(vca[2 * SR:] ** 2)) / np.sqrt(0.5) / (rl / (r + rl)) - 1) < 0.01


def test_lpg_ping_and_effect():
    y = SFX.REGISTRY["lpg_ping"]["fn"](SR, dur=1.2, freq=220.0)
    assert np.all(np.isfinite(y))
    r = lambda a, b: np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2))
    assert r(0.0, 0.05) > 5 * r(0.5, 0.55) > 0
    sp, f = _spec(y[:int(0.1 * SR)])
    sp2, _ = _spec(y[int(0.3 * SR):int(0.4 * SR)])
    cen = lambda s_: np.sum(f * s_) / np.sum(s_)
    assert cen(sp) > 1.2 * cen(sp2)
    noise = np.random.default_rng(1).uniform(-1, 1, SR)
    w = FX.REGISTRY["lpg"]["fn"](noise, SR, cv="strike", rate=2.0)
    assert np.all(np.isfinite(w)) and np.sqrt(np.mean(w[:1000] ** 2)) > np.sqrt(np.mean(w[SR // 4:SR // 4 + 1000] ** 2))
    fo = FX.REGISTRY["lpg"]["fn"](noise * np.linspace(1, 0, SR), SR, cv="follow")
    assert np.all(np.isfinite(fo))


def test_serge_wave_multiplier():
    f0 = 441.0
    t = np.arange(SR) / SR
    sine = np.sin(2 * np.pi * f0 * t)
    fn = FX.REGISTRY["serge_wavemult"]["fn"]
    y = fn(sine, SR, stages=2, mode="cheb", blend=1.0)
    assert abs(_peak_hz(y) - 4 * f0) < 1.0
    assert _harm(y, f0, 4) > 0.9 and max(_harm(y, f0, k) for k in (1, 2, 3, 5, 6)) < 0.01
    tri = 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f0 * t))
    z = fn(tri, SR, stages=3, mode="abs", blend=1.0)
    assert abs(_peak_hz(z) - 8 * f0) < 1.0
    assert abs(_peak_hz(fn(sine, SR, stages=1, mode="abs")) - 2 * f0) < 1.0
    mid = fn(sine, SR, stages=2, mode="cheb", blend=0.75)
    assert _harm(mid, f0, 2) > 0.2 and _harm(mid, f0, 4) > 0.2 and _harm(mid, f0, 1) < 0.01


def test_serge_dusg():
    for curve in (0.0, 1.0, 0.5):
        gate = np.zeros(2 * SR)
        gate[0] = 1.0
        y = S.dusg(gate, SR, rise=0.01, fall=0.03, curve=curve, cycle=True)
        seg = y[SR // 4:SR // 4 + SR]
        assert abs(int(np.count_nonzero((seg[1:] >= 0.5) & (seg[:-1] < 0.5))) - 25) <= 1, curve
        assert y.min() >= 0.0 and y.max() <= 1.0 + 1e-9
    gate = np.zeros(SR)
    gate[100:200] = 1.0
    for curve in (0.0, 1.0):
        y = S.dusg(gate, SR, rise=0.02, fall=0.05, curve=curve)
        top = int(np.argmax(y >= 1.0 - 1e-9))
        assert abs((top - 100) / SR - 0.02) < 0.0005, curve
        assert y[-1] == 0.0
        end = int(np.argmax(y[top:] <= 0.0)) + top
        assert abs((end - top) / SR - 0.05) < 0.0008, curve
    sus = S.dusg(np.concatenate([np.zeros(10), np.ones(SR // 2), np.zeros(SR // 2)]), SR, rise=0.01, fall=0.02, sustain=True)
    assert abs(sus[SR // 4] - 1.0) < 1e-9 and sus[-1] == 0.0
    o = SFX.REGISTRY["serge_dusg"]["fn"](SR, dur=1.0)
    assert np.all(np.isfinite(o))
    assert np.all(np.isfinite(SFX.REGISTRY["serge_dusg"]["fn"](SR, dur=1.0, out="pitch", cycle=False)))


def test_serge_random():
    n = 4 * SR
    st = S.serge_random_cv(n, SR, rate=8.0, mode="stepped", seed=3)
    changes = int(np.count_nonzero(np.diff(st)))
    assert abs(changes - 8 * 4) <= 2
    assert st.min() >= -1 and st.max() <= 1
    sm = S.serge_random_cv(n, SR, rate=2.0, mode="smooth", lag=0.05, seed=3)
    held = S.serge_random_cv(n, SR, rate=2.0, mode="stepped", seed=3)
    k = int(np.nonzero(np.diff(held))[0][1]) + 1
    lag = int(0.05 * SR)
    frac = (sm[k + lag] - sm[k - 1]) / (held[k] - sm[k - 1])
    assert abs(frac - (1 - 1 / math.e)) < 0.02
    assert np.max(np.abs(np.diff(sm))) < np.max(np.abs(np.diff(held)))
    j = S.serge_random_cv(n, SR, rate=8.0, mode="stepped", jitter=0.5, seed=3)
    assert np.all(np.isfinite(j))
    for mode in ("stepped", "smooth"):
        for out in ("fm", "cv"):
            assert np.all(np.isfinite(SFX.REGISTRY["serge_random"]["fn"](SR, dur=1.0, mode=mode, out=out)))


def test_buchla_259_and_registry():
    y = SFX.REGISTRY["buchla_259"]["fn"](SR, dur=0.5)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.1
    for w in ("sine", "triangle", "square"):
        assert np.all(np.isfinite(SFX.REGISTRY["buchla_259"]["fn"](SR, dur=0.2, mod_wave=w, am=0.5)))
    for name in ("buchla_259", "lpg_ping", "serge_dusg", "serge_random"):
        assert name in SFX.REGISTRY
    for name in ("buchla_fold", "lpg", "serge_wavemult"):
        assert name in FX.REGISTRY
    assert "dx7" in INS.REGISTRY
