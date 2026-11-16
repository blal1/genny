"""Historic synthesizers: Yamaha DX7 (6 operators, 32 algorithms, 4-rate / 4-level envelopes, SysEx decoder), Buchla (series wavefolder,
vactrol low-pass gate, simplified 259) and Serge (wave multiplier, dual universal slope generator, smooth / stepped random).

Registers
    instruments  `dx7`
    effects      `buchla_fold`, `lpg`, `serge_wavemult`
    sfx          `buchla_259`, `lpg_ping`, `serge_dusg`, `serge_random`
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.signal import lfilter

from . import filters as F
from .core import DEFAULT_SR, samples
from .dsp import dc_blocker, oversampled
from .fx import effect
from .instruments import instrument
from .sfx import sfx
from .spec import SpecError

TWO_PI = 2.0 * math.pi


# ============================================================================ Yamaha DX7
DX7_STEP_DB = 0.0235
DX7_FULL = 4096
DX7_FLOOR = 3824
DX7_ATTACK_JUMP = 1700
DX7_MAX_INDEX = 4.0 * math.pi
LEVEL_LUT = (0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46)
EXP_SCALE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 16, 19, 23, 27, 33, 39, 47, 56, 66, 80, 94, 110, 126, 142, 158, 174, 190, 206,
             222, 238, 250)
VELOCITY_LUT = (0, 70, 86, 97, 106, 114, 121, 126, 132, 138, 142, 148, 152, 156, 160, 163, 166, 170, 173, 174, 178, 181, 184, 186, 189,
                190, 194, 196, 198, 200, 202, 205, 206, 209, 211, 214, 216, 218, 220, 222, 224, 225, 227, 229, 230, 232, 233, 235, 237,
                238, 240, 241, 242, 243, 244, 246, 246, 248, 249, 250, 251, 252, 253, 254)
DET_CENTS = 12606 * 1200.0 / 2 ** 24
PMS_TAB = (0, 10, 20, 33, 55, 92, 153, 255)
AMS_DB = tuple(18.0 * x / 16777216 for x in (0, 4342338, 7171437, 16777216))
PITCH_EG_LEVEL = (-128, -116, -104, -95, -85, -76, -68, -61, -56, -52, -49, -46, -43, -41, -39, -37, -35, -33, -32, -31, -30, -29, -28, -27,
                  -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0,
                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                  34, 35, 38, 40, 43, 46, 49, 53, 58, 65, 73, 82, 92, 103, 115, 127)
PITCH_EG_RATE = (1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 36, 37, 38, 39, 41, 42, 44, 46, 47, 49, 51, 53, 54, 56, 58, 60, 62, 64, 66, 68,
                 70, 72, 74, 76, 79, 82, 85, 88, 91, 94, 98, 102, 106, 110, 115, 120, 125, 130, 135, 141, 147, 153, 159, 165, 171, 178, 185,
                 193, 202, 211, 232, 243, 254, 255)

# (carriers, {operator: modulators}, (feedback receiver, feedback source)), operators numbered 1..6
_A = {
    1: ((1, 3), {1: (2,), 3: (4,), 4: (5,), 5: (6,)}, (6, 6)),
    2: ((1, 3), {1: (2,), 3: (4,), 4: (5,), 5: (6,)}, (2, 2)),
    3: ((1, 4), {1: (2,), 2: (3,), 4: (5,), 5: (6,)}, (6, 6)),
    4: ((1, 4), {1: (2,), 2: (3,), 4: (5,), 5: (6,)}, (6, 4)),
    5: ((1, 3, 5), {1: (2,), 3: (4,), 5: (6,)}, (6, 6)),
    6: ((1, 3, 5), {1: (2,), 3: (4,), 5: (6,)}, (6, 5)),
    7: ((1, 3), {1: (2,), 3: (4, 5), 5: (6,)}, (6, 6)),
    8: ((1, 3), {1: (2,), 3: (4, 5), 5: (6,)}, (4, 4)),
    9: ((1, 3), {1: (2,), 3: (4, 5), 5: (6,)}, (2, 2)),
    10: ((1, 4), {1: (2,), 2: (3,), 4: (5, 6)}, (3, 3)),
    11: ((1, 4), {1: (2,), 2: (3,), 4: (5, 6)}, (6, 6)),
    12: ((1, 3), {1: (2,), 3: (4, 5, 6)}, (2, 2)),
    13: ((1, 3), {1: (2,), 3: (4, 5, 6)}, (6, 6)),
    14: ((1, 3), {1: (2,), 3: (4,), 4: (5, 6)}, (6, 6)),
    15: ((1, 3), {1: (2,), 3: (4,), 4: (5, 6)}, (2, 2)),
    16: ((1,), {1: (2, 3, 5), 3: (4,), 5: (6,)}, (6, 6)),
    17: ((1,), {1: (2, 3, 5), 3: (4,), 5: (6,)}, (2, 2)),
    18: ((1,), {1: (2, 3, 4), 4: (5,), 5: (6,)}, (3, 3)),
    19: ((1, 4, 5), {1: (2,), 2: (3,), 4: (6,), 5: (6,)}, (6, 6)),
    20: ((1, 2, 4), {1: (3,), 2: (3,), 4: (5, 6)}, (3, 3)),
    21: ((1, 2, 4, 5), {1: (3,), 2: (3,), 4: (6,), 5: (6,)}, (3, 3)),
    22: ((1, 3, 4, 5), {1: (2,), 3: (6,), 4: (6,), 5: (6,)}, (6, 6)),
    23: ((1, 2, 4, 5), {2: (3,), 4: (6,), 5: (6,)}, (6, 6)),
    24: ((1, 2, 3, 4, 5), {3: (6,), 4: (6,), 5: (6,)}, (6, 6)),
    25: ((1, 2, 3, 4, 5), {4: (6,), 5: (6,)}, (6, 6)),
    26: ((1, 2, 4), {2: (3,), 4: (5, 6)}, (6, 6)),
    27: ((1, 2, 4), {2: (3,), 4: (5, 6)}, (3, 3)),
    28: ((1, 3, 6), {1: (2,), 3: (4,), 4: (5,)}, (5, 5)),
    29: ((1, 2, 3, 5), {3: (4,), 5: (6,)}, (6, 6)),
    30: ((1, 2, 3, 6), {3: (4,), 4: (5,)}, (5, 5)),
    31: ((1, 2, 3, 4, 5), {5: (6,)}, (6, 6)),
    32: ((1, 2, 3, 4, 5, 6), {}, (6, 6)),
}


def dx7_algorithm(n):
    """(carrier mask (6,), modulation matrix [receiver, source] (6, 6), feedback receiver index, feedback source index), 0-based."""
    if n not in _A:
        raise SpecError(f"dx7: algorithm {n} is not 1..32")
    car, mods, (fr, fs) = _A[n]
    c = np.zeros(6)
    c[[o - 1 for o in car]] = 1.0
    m = np.zeros((6, 6))
    for r, srcs in mods.items():
        for s in srcs:
            m[r - 1, s - 1] = 1.0
    return c, m, fr - 1, fs - 1


def dx7_out_level(v):
    """0..99 level -> 0..127 on the 0.75 dB scale."""
    v = int(v)
    return LEVEL_LUT[v] if v < 20 else 28 + v


def dx7_qrate(rate, note=60, rs=0):
    q = (int(rate) * 41) >> 6
    q += (int(rs) * min(31, max(0, int(note) // 3 - 7))) >> 3
    return min(63, q)


def dx7_rate_db_per_s(q):
    """Decay speed of a 0..63 quantized rate in dB / s: 0.2819 2^(q div 4) (1 + 0.25 (q mod 4))."""
    return 0.2819 * 2 ** (q >> 2) * (1 + 0.25 * (q & 3))


def dx7_key_scale(note, bp, ld, rd, lc, rc):
    """Keyboard level scaling in 0.75 dB units: offset = note - bp - 17, groups of 3 semitones ((offset + 1) // 3 to the right,
    (1 - offset) // 3 to the left); curves 0 -LIN, 1 -EXP, 2 +EXP, 3 +LIN; the exponential table is read with a shift of 15, the
    linear one of 12."""
    d = int(note) - int(bp) - 17
    depth, curve = (rd, rc) if d >= 0 else (ld, lc)
    g = (d + 1) // 3 if d >= 0 else (1 - d) // 3
    s = (g * int(depth) * 329) >> 12 if curve in (0, 3) else (EXP_SCALE[min(g, 32)] * int(depth) * 329) >> 15
    return -s if curve < 2 else s


def dx7_velocity(vel, sens):
    """Velocity 0..127 and sensitivity 0..7 -> level offset in 0.0235 dB steps (16 per unit)."""
    v = VELOCITY_LUT[min(127, max(0, int(vel))) >> 1] - 239
    return ((int(sens) * v + 7) >> 3) << 4


@nb.njit(cache=True)
def _dx7_eg(n, sr, q, tgt, off):
    out = np.empty(n)
    lv = tgt[3]
    stage = 0
    hold = False
    floor = DX7_FULL - DX7_FLOOR
    for i in range(n):
        if i == off:
            stage = 3
            hold = False
        while not hold and lv == tgt[stage]:
            if stage >= 2:
                hold = True
            else:
                stage += 1
        if not hold:
            base = 12.0 * 2.0 ** (q[stage] >> 2) * (1.0 + 0.25 * (q[stage] & 3)) / sr
            if tgt[stage] > lv:
                if lv < floor + DX7_ATTACK_JUMP:
                    lv = floor + DX7_ATTACK_JUMP
                lv += base * (2 + int((DX7_FULL - lv) / 256.0))
                if lv >= tgt[stage]:
                    lv = tgt[stage]
            else:
                lv -= base
                if lv <= tgt[stage]:
                    lv = tgt[stage]
        out[i] = lv
    return out


@nb.njit(cache=True)
def _dx7_core(n, inc, amp, modmask, carrier, fb_to, fb_from, fbk):
    out = np.zeros(n)
    ph = np.zeros(6)
    y = np.zeros(6)
    f0 = 0.0
    f1 = 0.0
    for t in range(n):
        for i in range(5, -1, -1):
            m = 0.0
            for j in range(i + 1, 6):
                if modmask[i, j] != 0.0:
                    m += y[j]
            m *= DX7_MAX_INDEX
            if i == fb_to:
                m += fbk * (f0 + f1)
            y[i] = amp[i, t] * math.sin(TWO_PI * ph[i] + m)
            ph[i] += inc[i, t]
            ph[i] -= math.floor(ph[i])
            out[t] += carrier[i] * y[i]
        f0 = f1
        f1 = y[fb_from]
    return out


def _op(r=(99, 99, 99, 99), l=(99, 99, 99, 0), ol=99, coarse=1, fine=0, det=7, mode=0, bp=39, ld=0, rd=0, lc=0, rc=0, rs=0, ams=0, kvs=0):
    return dict(r=list(r), l=list(l), ol=ol, coarse=coarse, fine=fine, det=det, mode=mode, bp=bp, ld=ld, rd=rd, lc=lc, rc=rc, rs=rs,
                ams=ams, kvs=kvs)


def dx7_voice(name="INIT VOICE", ops=None, alg=1, fb=0, oks=1, pr=(99, 99, 99, 99), pl=(50, 50, 50, 50), lfo_speed=35, lfo_delay=0,
              pmd=0, amd=0, lfo_sync=1, lfo_wave=0, pms=3, transpose=24):
    ops = list(ops) if ops else [_op(l=(99, 99, 99, 0), ol=99 if i == 0 else 0) for i in range(6)]
    return dict(name=name, ops=ops, alg=alg, fb=fb, oks=oks, pr=list(pr), pl=list(pl), lfo=dict(speed=lfo_speed, delay=lfo_delay, pmd=pmd,
                amd=amd, sync=lfo_sync, wave=lfo_wave, pms=pms), transpose=transpose)


_OFF = _op(ol=0, l=(0, 0, 0, 0))
DX7_PRESETS = {
    "epiano": dx7_voice("E.PIANO 1", [
        _op((95, 29, 20, 50), (99, 95, 0, 0), 99, 1), _op((95, 50, 35, 78), (99, 75, 0, 0), 58, 14), _op((95, 29, 20, 50), (99, 95, 0, 0), 99, 1),
        _op((95, 20, 20, 50), (99, 85, 0, 0), 82, 1), _OFF, _OFF], alg=5, fb=0),
    "bell": dx7_voice("TUB BELLS", [
        _op((95, 40, 35, 40), (99, 90, 20, 0), 99, 1), _op((95, 45, 35, 40), (99, 80, 10, 0), 80, 3, 50), _op((95, 30, 25, 40), (99, 85, 15, 0), 90, 4, 0),
        _op((95, 40, 30, 40), (99, 70, 0, 0), 70, 7, 0), _OFF, _OFF], alg=5, fb=3),
    "bass": dx7_voice("SYN BASS", [
        _op((99, 60, 40, 60), (99, 90, 85, 0), 99, 1), _op((99, 70, 40, 60), (99, 70, 20, 0), 86, 1), _op((99, 60, 40, 60), (99, 90, 85, 0), 99, 0),
        _OFF, _OFF, _OFF], alg=1, fb=0),
    "brass": dx7_voice("BRASS 1", [
        _op((60, 40, 40, 50), (99, 95, 90, 0), 99, 1), _op((55, 40, 40, 50), (99, 85, 75, 0), 80, 1), _OFF, _OFF, _OFF, _OFF], alg=1, fb=5),
}


def _lfo_wave(kind, n, sr, speed, seed=0):
    sr_ = 1 if speed == 0 else (165 * int(speed)) >> 6
    sr_ *= 11 if sr_ < 160 else 11 + ((sr_ - 160) >> 4)
    f = 25190424.0 * sr_ / 2 ** 32
    ph = (np.arange(n) * f / sr) % 1.0
    if kind == 0:
        return np.where(ph < 0.5, 4 * ph - 1, 3 - 4 * ph)
    if kind == 1:
        return 1 - 2 * ph
    if kind == 2:
        return 2 * ph - 1
    if kind == 3:
        return np.where(ph < 0.5, 1.0, -1.0)
    if kind == 4:
        return np.sin(TWO_PI * ph)
    k = np.floor(np.arange(n) * f * 2 / sr).astype(int)
    return np.random.default_rng(seed).uniform(-1, 1, k.max() + 2)[k]


def _lfo_delay(n, sr, delay):
    """LFO onset factor 0..1: silent for the first part, then a linear ramp (delay 0 = immediate)."""
    a = 99 - int(delay)
    if a == 99:
        return np.ones(n)
    a = (16 + (a & 15)) << (1 + (a >> 4))
    a2 = max(0x80, a & 0xFF80)
    t0 = 2.0 ** 31 / (25190424.0 * a)
    t1 = 2.0 ** 31 / (25190424.0 * a2)
    return np.clip((np.arange(n) / sr - t0) / t1, 0.0, 1.0)


def _pitch_eg(n, sr, pr, pl, off):
    """Pitch envelope in octaves: level table / 32, each segment a straight ramp at rate table / 21.3 octaves per second."""
    lv = [PITCH_EG_LEVEL[int(p)] / 32.0 for p in pl]

    def spd(k):
        return PITCH_EG_RATE[int(pr[k])] / 21.3

    kt, kv = [0.0], [lv[3]]
    for k in range(3):
        kt.append(kt[-1] + abs(lv[k] - kv[-1]) / spd(k))
        kv.append(lv[k])
    t = np.arange(n) / sr
    y = np.interp(t, kt, kv)
    t0 = off / sr
    cur = float(np.interp(t0, kt, kv))
    d = max(abs(lv[3] - cur) / spd(3), 1e-6)
    y[off:] = cur + (lv[3] - cur) * np.clip((t[off:] - t0) / d, 0.0, 1.0)
    return y


def render_dx7(voice, hz, dur, sr=DEFAULT_SR, vel=1.0, fb=None):
    """One note of a DX7 voice (a dict as `dx7_voice` / `dx7_parse_sysex` give). Returns mono, about [-1, 1]."""
    hz = float(hz)
    note = int(round(69 + 12 * math.log2(max(hz, 1.0) / 440.0)))
    v127 = int(round(float(np.clip(vel, 0.0, 1.0)) * 127))
    car, mm, fr, fs = dx7_algorithm(int(voice["alg"]))
    off = int(dur * sr)
    ops = voice["ops"]
    tails = []
    for o in ops:
        q4 = dx7_qrate(o["r"][3], note, o["rs"])
        drop = max(0.0, DX7_FULL - (127 - dx7_out_level(o["l"][2])) * 32 - (DX7_FULL - DX7_FLOOR)) * DX7_STEP_DB
        tails.append(drop / dx7_rate_db_per_s(q4))
    tail = float(np.clip(max(tails), 0.05, 6.0))
    n = off + int(tail * sr)
    inc = np.empty((6, n))
    amp = np.empty((6, n))
    ap = voice["lfo"]
    lfo = _lfo_wave(ap["wave"], n, sr, ap["speed"])
    t = np.arange(n) / sr
    fade = _lfo_delay(n, sr, ap["delay"])
    pm = 2.0 ** (lfo * fade * ((ap["pmd"] * 165) >> 6) * PMS_TAB[ap["pms"]] / 65536.0 +
                 _pitch_eg(n, sr, voice["pr"], voice["pl"], off))
    pm *= 2.0 ** ((voice["transpose"] - 24) / 12.0)
    am_db = AMS_DB
    for i, o in enumerate(ops):
        kls = dx7_key_scale(note, o["bp"], o["ld"], o["rd"], o["lc"], o["rc"])
        tg = np.array([max(DX7_FULL - DX7_FLOOR, DX7_FULL - (127 - dx7_out_level(L)) * 32) for L in o["l"]], np.float64)
        q = np.array([dx7_qrate(r, note, o["rs"]) for r in o["r"]], np.int64)
        lv = _dx7_eg(n, float(sr), q, tg, off)
        steps = (lv - DX7_FULL + 32 * (dx7_out_level(o["ol"]) + kls - 127) + dx7_velocity(v127, o["kvs"])).astype(np.float64)
        if o["ams"] and ap["amd"]:
            steps -= (0.5 + 0.5 * lfo) * fade * am_db[o["ams"]] * ap["amd"] / 99.0 / DX7_STEP_DB
        amp[i] = np.where(steps > -DX7_FLOOR, 10.0 ** (steps * DX7_STEP_DB / 20.0), 0.0)
        if o["mode"]:
            f = 10.0 ** ((o["coarse"] & 3) + o["fine"] / 100.0)
            if o["det"] > 7:
                f *= 2.0 ** (13457 * (o["det"] - 7) / 2 ** 24)
            inc[i] = f / sr
        else:
            r = (0.5 if o["coarse"] == 0 else o["coarse"]) * (1.0 + o["fine"] / 100.0) * 2.0 ** ((o["det"] - 7) * DET_CENTS / 1200.0)
            inc[i] = hz * r * pm / sr
    fbl = int(voice["fb"] if fb is None else fb)
    fbk = DX7_MAX_INDEX * 2.0 ** (fbl - 9) if fbl else 0.0
    y = _dx7_core(n, inc, amp, mm, car, fr, fs, fbk)
    return y / math.sqrt(max(car.sum(), 1.0))


# ---------------------------------------------------------------------------- SysEx
def dx7_checksum(data):
    """Two's complement of the sum of the data bytes, 7 bits."""
    return (-sum(int(b) for b in data)) & 0x7F


def _voice_from_155(p):
    ops = []
    for k in range(6):
        b = p[21 * k:21 * k + 21]
        ops.append(dict(r=list(b[0:4]), l=list(b[4:8]), bp=b[8], ld=b[9], rd=b[10], lc=b[11], rc=b[12], rs=b[13], ams=b[14], kvs=b[15],
                        ol=b[16], mode=b[17], coarse=b[18], fine=b[19], det=b[20]))
    ops.reverse()
    g = p[126:]
    return dict(name=bytes(g[19:29]).decode("ascii", "replace").rstrip(), ops=ops, pr=list(g[0:4]), pl=list(g[4:8]), alg=g[8] + 1, fb=g[9],
                oks=g[10], lfo=dict(speed=g[11], delay=g[12], pmd=g[13], amd=g[14], sync=g[15], wave=g[16], pms=g[17]), transpose=g[18])


def _voice_to_155(v):
    p = []
    for o in reversed(v["ops"]):
        p += o["r"] + o["l"] + [o["bp"], o["ld"], o["rd"], o["lc"], o["rc"], o["rs"], o["ams"], o["kvs"], o["ol"], o["mode"], o["coarse"],
                                o["fine"], o["det"]]
    lf = v["lfo"]
    p += v["pr"] + v["pl"] + [v["alg"] - 1, v["fb"], v["oks"], lf["speed"], lf["delay"], lf["pmd"], lf["amd"], lf["sync"], lf["wave"],
                             lf["pms"], v["transpose"]]
    p += list(v["name"].ljust(10)[:10].encode("ascii", "replace"))
    return p


def dx7_unpack_voice(b):
    """128 packed bytes (one voice of a 32-voice bulk dump) -> voice dict."""
    if len(b) != 128:
        raise SpecError("dx7: a packed voice is 128 bytes")
    p = []
    for k in range(6):
        o = b[17 * k:17 * k + 17]
        p += list(o[0:11]) + [o[11] & 3, o[11] >> 2 & 3, o[12] & 7, o[13] & 3, o[13] >> 2 & 7, o[14], o[15] & 1, o[15] >> 1 & 31, o[16],
                              o[12] >> 3 & 15]
    g = b[102:]
    p += list(g[0:8]) + [g[8] & 31, g[9] & 7, g[9] >> 3 & 1, g[10], g[11], g[12], g[13], g[14] & 1, g[14] >> 1 & 7, g[14] >> 4 & 7, g[15]]
    p += list(g[16:26])
    return _voice_from_155([int(x) for x in p])


def dx7_pack_voice(v):
    p = _voice_to_155(v)
    b = []
    for k in range(6):
        o = p[21 * k:21 * k + 21]
        b += o[0:11] + [o[11] | o[12] << 2, o[13] | o[20] << 3, o[14] | o[15] << 2, o[16], o[17] | o[18] << 1, o[19]]
    g = p[126:]
    b += g[0:8] + [g[8], g[9] | g[10] << 3, g[11], g[12], g[13], g[14], g[16] << 1 | g[15] | g[17] << 4, g[18]] + g[19:29]
    return bytes(b)


def dx7_parse_sysex(raw):
    """A DX7 SysEx file or message: 32-voice bulk dump (4096 data bytes), single voice (155 data bytes), or the bare data of either.
    Returns the list of voice dicts; the checksum is verified when the framing is present."""
    raw = bytes(raw)
    data, framed = raw, False
    if raw[:1] == b"\xf0":
        if len(raw) < 8 or raw[1] != 0x43 or raw[-1] != 0xF7:
            raise SpecError("dx7: not a Yamaha SysEx message")
        fmt, cnt = raw[3], (raw[4] << 7) | raw[5]
        data, ck = raw[6:-2], raw[-2]
        if (fmt, cnt) not in ((9, 4096), (0, 155)) or len(data) != cnt:
            raise SpecError("dx7: unsupported SysEx format or length")
        if dx7_checksum(data) != ck:
            raise SpecError("dx7: bad checksum")
        framed = True
    if len(data) == 4096:
        return [dx7_unpack_voice(data[128 * i:128 * i + 128]) for i in range(32)]
    if len(data) == 155:
        return [_voice_from_155(list(data))]
    if len(data) == 128 and not framed:
        return [dx7_unpack_voice(data)]
    raise SpecError("dx7: SysEx data must be 4096, 155 or 128 bytes")


def dx7_make_sysex(voices, channel=0):
    """Bulk dump of up to 32 voices (the rest initial voices), or a single-voice message for one voice and `single=True` semantics
    through `dx7_make_single`."""
    vs = list(voices)[:32] + [dx7_voice()] * (32 - len(list(voices)[:32]))
    data = b"".join(dx7_pack_voice(v) for v in vs)
    return bytes([0xF0, 0x43, channel & 15, 9, 0x20, 0x00]) + data + bytes([dx7_checksum(data), 0xF7])


def dx7_make_single(voice, channel=0):
    data = bytes(_voice_to_155(voice))
    return bytes([0xF0, 0x43, channel & 15, 0, 0x01, 0x1B]) + data + bytes([dx7_checksum(data), 0xF7])


def _load_voice(patch, sysex, index):
    if sysex:
        import os
        if not os.path.isfile(sysex) or os.path.getsize(sysex) > 1 << 20:
            raise SpecError("dx7: sysex must be a .syx file under 1 MB")
        with open(sysex, "rb") as fh:
            vs = dx7_parse_sysex(fh.read())
        return vs[int(np.clip(index, 0, len(vs) - 1))]
    if patch not in DX7_PRESETS:
        raise SpecError(f"dx7: patch {patch!r} is not one of {sorted(DX7_PRESETS)}")
    return DX7_PRESETS[patch]


@instrument("dx7", "Yamaha DX7: six sine operators, 32 algorithms, 4-rate / 4-level envelopes with the DX7 rate and level tables, key and velocity "
            "scaling, feedback, LFO, pitch EG; voices from a built-in preset or a SysEx file (32-voice dump or single voice).",
            family="synth", span=("C2", "C6"),
            patch=("epiano", "epiano | bell | bass | brass (ignored when `sysex` is given)"),
            sysex=("", "path of a .syx file (bulk dump of 32 voices or a single voice)"), voice=(0, "voice number in the SysEx file, 0..31"),
            feedback=(-1, "0..7 overrides the voice feedback, -1 keeps it"))
def dx7(freq, dur, sr=DEFAULT_SR, vel=1.0, patch="epiano", sysex="", voice=0, feedback=-1):
    v = _load_voice(patch, sysex, voice)
    y = render_dx7(v, freq, dur, sr, vel, None if feedback < 0 else int(feedback))
    return 0.5 * F.dc_block(y, sr)


# ============================================================================ Buchla
def _fold_cell(v, t, knee):
    u = v / t
    tri = (2 / np.pi) * np.arcsin(np.sin(0.5 * np.pi * u))
    sn = np.sin(0.5 * np.pi * u)
    return t * ((1.0 - knee) * tri + knee * sn)


def buchla_fold_curve(v, cells=3, gain=2.0, threshold=1.0, knee=0.0):
    """Series of folding cells: each multiplies by `gain` and reflects at +-threshold (knee 0 = straight reflection, 1 = sinusoidal)."""
    v = np.asarray(v, float)
    for _ in range(int(cells)):
        v = _fold_cell(float(gain) * v, float(threshold), float(knee))
    return v


@effect("buchla_fold", "Buchla-style timbre section: a series of folding cells, each with its gain, reflecting at +-1 (oversampled).",
        timbre=(1.0, "input gain before the first cell"), cells=(3, "1..8 cells in series"), gain=(2.0, "gain between cells"),
        symmetry=(0.0, "-1..1 offset before folding (even harmonics)"), knee=(0.0, "0 straight reflection .. 1 sinusoidal cell"),
        tone=(0.0, "post low-pass Hz (0 = off)"), mix=(1.0, "wet 0..1"))
def buchla_fold(x, sr=DEFAULT_SR, timbre=1.0, cells=3, gain=2.0, symmetry=0.0, knee=0.0, tone=0.0, mix=1.0):
    b = float(np.clip(symmetry, -1.0, 1.0))
    c = int(np.clip(cells, 1, 8))
    ref = buchla_fold_curve(b, c, gain, 1.0, knee)
    wet = oversampled(lambda v: buchla_fold_curve(float(timbre) * v + b, c, gain, 1.0, knee) - ref, np.asarray(x, float), 8)
    if b != 0:
        wet = dc_blocker(wet, sr)
    if tone and tone < 0.49 * sr:
        wet = F.lowpass(wet, tone, sr)
    return x * (1 - mix) + wet * mix


@nb.njit(cache=True)
def _lpg_light(u, sr, up, dn_fast, dn_slow, thermal, weight):
    n = u.size
    s_out = np.empty(n)
    g_out = np.empty(n)
    s = 0.0
    h = 0.0
    for i in range(n):
        tgt = u[i]
        tau = up if tgt > s else dn_fast + (dn_slow - dn_fast) * (1.0 - s)
        s += (tgt - s) / (tau * sr)
        h += (s - h) / (thermal * sr)
        s_out[i] = s
        g_out[i] = (1.0 - weight) * s + weight * h
    return s_out, g_out


@nb.njit(cache=True)
def _lpg_filter(x, g, sr, ron, roff, rload, c, mode, poles, floor):
    n = x.size
    y = np.empty(n)
    z1 = 0.0
    z2 = 0.0
    lr = math.log(roff / ron)
    for i in range(n):
        r = roff * math.exp(-lr * g[i])
        div = rload / (r + rload)
        if mode == 0:
            gain = div + floor
            fc = 1.0 / (TWO_PI * (r * rload / (r + rload)) * c)
        elif mode == 1:
            gain = 1.0
            fc = 1.0 / (TWO_PI * r * c)
        else:
            gain = div + floor
            fc = 0.45 * sr
        a = 1.0 - math.exp(-TWO_PI * min(fc, 0.45 * sr) / sr)
        z1 += a * (gain * x[i] - z1)
        if poles == 2:
            z2 += a * (z1 - z2)
            y[i] = z2
        else:
            y[i] = z1
    return y


_LPG_MODES = {"combo": 0, "lowpass": 1, "vca": 2}


def lpg_light(cv, sr=DEFAULT_SR, up=0.0025, dn_fast=0.035, dn_slow=0.4, thermal=0.15, weight=0.25, gamma=1.0):
    """Vactrol light and conductance (0..1) from a control voltage (0..1): the light rises with time constant `up`, falls with a time
    constant that grows from `dn_fast` (lit) to `dn_slow` (dark); the conductance adds a slow thermal memory (`thermal`, weight `weight`)."""
    u = np.clip(np.asarray(cv, float), 0.0, 1.0) ** float(gamma)
    return _lpg_light(np.ascontiguousarray(u), float(sr), float(up), float(dn_fast), float(dn_slow), float(thermal), float(weight))


def lpg_process(x, cv, sr=DEFAULT_SR, mode="combo", poles=1, ron=1.5e3, roff=1e7, rload=1e5, c=10e-9, floor_db=-80.0, **vactrol):
    """Low pass gate: conductance from `lpg_light`, series resistance R = roff (ron / roff)^g, load `rload` with shunt capacitor `c`:
    gain rload / (rload + R), corner 1 / (2 pi (R || rload) c) in `combo`; the corner at R alone with unit gain in `lowpass`; gain only in `vca`."""
    if mode not in _LPG_MODES:
        raise SpecError(f"lpg: mode {mode!r} is not one of {sorted(_LPG_MODES)}")
    _, g = lpg_light(cv, sr, **vactrol)
    return _lpg_filter(np.ascontiguousarray(x, np.float64), g, float(sr), float(ron), float(roff), float(rload), float(c),
                       _LPG_MODES[mode], int(poles), 10.0 ** (floor_db / 20.0))


@effect("lpg", "Buchla-style low pass gate (vactrol): the control is a periodic strike or the input envelope; the vactrol opens fast and "
        "closes slowly with a thermal tail, gating and darkening the sound together.",
        mode=("combo", "combo | lowpass | vca"), poles=(1, "1 | 2"), cv=("strike", "strike | follow"), rate=(2.0, "strikes per second (cv strike)"),
        width=(8.0, "strike width in ms"), up=(2.5, "vactrol rise time constant, ms"), down=(35.0, "fall time constant when lit, ms"),
        tail=(400.0, "fall time constant when dark, ms"), c_nf=(10.0, "filter capacitor nF"), mix=(1.0, "wet 0..1"))
def lpg(x, sr=DEFAULT_SR, mode="combo", poles=1, cv="strike", rate=2.0, width=8.0, up=2.5, down=35.0, tail=400.0, c_nf=10.0, mix=1.0):
    x = np.asarray(x, float)
    mono = x if x.ndim == 1 else x.mean(axis=1)
    if cv == "follow":
        u = np.clip(lfilter([0.01], [1.0, -0.99], np.abs(mono)) * 4.0, 0.0, 1.0)
    else:
        u = ((np.arange(mono.size) / sr * float(rate)) % 1.0 < float(width) * 1e-3 * float(rate)).astype(float)
    wet = lpg_process(mono, u, sr, mode, int(poles), c=float(c_nf) * 1e-9, up=up * 1e-3, dn_fast=down * 1e-3, dn_slow=tail * 1e-3)
    wet = wet if x.ndim == 1 else np.repeat(wet[:, None], x.shape[1], axis=1)
    return x * (1 - mix) + wet * mix


@sfx("lpg_ping", "Bongo ping of a vactrol low pass gate: a short strike opens the gate on an oscillator or a click, and the thermal tail closes it.",
     dur=(1.2, "s"), freq=(220.0, "Hz of the oscillator"), exc=("saw", "saw | square | sine | click"), strike=(5.0, "strike width in ms"),
     level=(1.0, "strike voltage 0..1"), up=(2.5, "rise time constant ms"), down=(35.0, "fall time constant when lit, ms"),
     tail=(400.0, "fall time constant when dark, ms"), mode=("combo", "combo | lowpass | vca"), poles=(2, "1 | 2"))
def lpg_ping(sr=DEFAULT_SR, dur=1.2, freq=220.0, exc="saw", strike=5.0, level=1.0, up=2.5, down=35.0, tail=400.0, mode="combo", poles=2):
    n = samples(float(np.clip(dur, 0.05, 20.0)), sr)
    t = np.arange(n) / sr
    ph = (t * float(freq)) % 1.0
    x = {"saw": 2 * ph - 1, "square": np.where(ph < 0.5, 1.0, -1.0), "sine": np.sin(TWO_PI * ph)}.get(exc)
    if x is None:
        x = np.random.default_rng(0).uniform(-1, 1, n) * (t < 0.002)
        x[0] = 1.0
    cv = np.where(t < float(strike) * 1e-3, float(level), 0.0)
    y = lpg_process(x, cv, sr, mode, int(poles), up=up * 1e-3, dn_fast=down * 1e-3, dn_slow=tail * 1e-3)
    return 0.8 * F.dc_block(y, sr)


@sfx("buchla_259", "Simplified Buchla 259 complex oscillator: a sine principal oscillator frequency- and amplitude-modulated by a second "
     "oscillator, then the series-cell timbre section and symmetry.",
     dur=(2.0, "s"), freq=(110.0, "Hz of the principal oscillator"), ratio=(1.5, "modulation oscillator / principal"),
     fm=(0.5, "modulation index of the principal oscillator (radians)"), am=(0.0, "amplitude modulation depth 0..1"),
     mod_wave=("sine", "sine | triangle | square"), timbre=(2.0, "input gain of the timbre section"), cells=(3, "folding cells 1..8"),
     gain=(1.6, "gain between cells"), symmetry=(0.0, "-1..1"), knee=(0.3, "0 straight .. 1 sinusoidal"))
def buchla_259(sr=DEFAULT_SR, dur=2.0, freq=110.0, ratio=1.5, fm=0.5, am=0.0, mod_wave="sine", timbre=2.0, cells=3, gain=1.6, symmetry=0.0,
               knee=0.3):
    n = samples(float(np.clip(dur, 0.05, 30.0)), sr)
    t = np.arange(n) / sr
    pm = (t * float(freq) * float(ratio)) % 1.0
    mod = {"sine": np.sin(TWO_PI * pm), "triangle": 1 - 4 * np.abs(pm - 0.5) + 0.0 * pm, "square": np.where(pm < 0.5, 1.0, -1.0)}.get(mod_wave)
    if mod is None:
        raise SpecError("buchla_259: mod_wave is sine | triangle | square")
    x = np.sin(TWO_PI * freq * t + float(fm) * mod) * (1.0 - float(am) + float(am) * (0.5 + 0.5 * mod))
    return 0.7 * buchla_fold(x, sr, timbre, cells, gain, symmetry, knee, 0.0, 1.0)


# ============================================================================ Serge
def wavemult_stage(v, mode="abs"):
    """One multiplier stage on a wave in [-1, 1]: `abs` full-wave rectifies and re-centres (2 |v| - 1, a triangle gives a triangle at
    twice the frequency), `cheb` is the Chebyshev polynomial T2 (2 v^2 - 1, a sine gives a sine at twice the frequency)."""
    v = np.asarray(v, float)
    return 2.0 * np.abs(v) - 1.0 if mode == "abs" else 2.0 * v * v - 1.0


@effect("serge_wavemult", "Serge-style wave multiplier: cascaded stages each doubling the frequency of the wave (full-wave fold or Chebyshev), "
        "with a blend between the stage outputs.",
        stages=(2, "0..6 doubling stages (output = 2^stages x the input frequency)"), mode=("abs", "abs | cheb"),
        blend=(1.0, "0..1: 1 = last stage only, 0 = the input; in between mixes the neighbouring stages"), mix=(1.0, "wet 0..1"))
def serge_wavemult(x, sr=DEFAULT_SR, stages=2, mode="abs", blend=1.0, mix=1.0):
    x = np.asarray(x, float)
    k = int(np.clip(stages, 0, 6))
    pk = float(np.max(np.abs(x))) + 1e-12

    def run(v):
        outs = [v / pk]
        for _ in range(k):
            outs.append(wavemult_stage(outs[-1], mode))
        pos = float(np.clip(blend, 0.0, 1.0)) * k
        lo = int(math.floor(pos))
        hi = min(lo + 1, k)
        return (1 - (pos - lo)) * outs[lo] + (pos - lo) * outs[hi]

    wet = oversampled(run, x, 4)
    wet = dc_blocker(wet, sr) * pk
    return x * (1 - mix) + wet * mix


@nb.njit(cache=True)
def _dusg(gate, sr, rise, fall, curve, cycle, sustain, top):
    n = gate.size
    y = np.empty(n)
    v = 0.0
    state = 0
    prev = 0.0
    t_ = 1.3
    tau_r = rise / math.log(t_ / (t_ - 1.0))
    tau_f = fall / math.log(t_ / (t_ - 1.0))
    for i in range(n):
        if (gate[i] > 0.5 and prev <= 0.5) or (cycle and state == 0 and v <= 0.0):
            state = 1
        prev = gate[i]
        if state == 1:
            lin = 1.0 / (rise * sr)
            ex = (t_ - v) * (1.0 - math.exp(-1.0 / (tau_r * sr)))
            v += (1.0 - curve) * lin + curve * ex
            if v >= top:
                v = top
                state = 2
        elif state == 2:
            if (not sustain) or gate[i] <= 0.5:
                state = 3
        elif state == 3:
            lin = 1.0 / (fall * sr)
            ex = (v + (t_ - 1.0)) * (1.0 - math.exp(-1.0 / (tau_f * sr)))
            v -= (1.0 - curve) * lin + curve * ex
            if v <= 0.0:
                v = 0.0
                state = 0
        y[i] = v
    return y


def dusg(gate, sr=DEFAULT_SR, rise=0.01, fall=0.1, curve=0.0, cycle=False, sustain=False, top=1.0):
    """Dual universal slope generator, one half: a gate (0/1 samples) starts the rise to `top`, then the fall to 0 (`sustain`: the
    fall waits for the gate to go low; `cycle`: it restarts by itself, an oscillator of period rise + fall). `curve` 0 linear .. 1 RC
    exponential (reaches the top and 0 in exactly the set times)."""
    return _dusg(np.ascontiguousarray(gate, np.float64), float(sr), float(max(rise, 1e-4)), float(max(fall, 1e-4)), float(np.clip(curve, 0.0, 1.0)),
                 bool(cycle), bool(sustain), float(top))


@sfx("serge_dusg", "Serge dual universal slope generator: a rise / fall slope (linear to exponential) that can cycle as an oscillator or "
     "be triggered; the slope is either heard directly or sweeps the pitch of a sine.",
     dur=(2.0, "s"), rise=(0.01, "rise time s"), fall=(0.1, "fall time s"), curve=(0.0, "0 linear .. 1 exponential"),
     cycle=(True, "free-running oscillator"), trig_hz=(2.0, "trigger rate when not cycling"), out=("slope", "slope | pitch"),
     freq=(220.0, "base Hz of the pitch output"), octaves=(3.0, "pitch sweep range in octaves"))
def serge_dusg(sr=DEFAULT_SR, dur=2.0, rise=0.01, fall=0.1, curve=0.0, cycle=True, trig_hz=2.0, out="slope", freq=220.0, octaves=3.0):
    n = samples(float(np.clip(dur, 0.05, 60.0)), sr)
    gate = (((np.arange(n) / sr * float(trig_hz)) % 1.0) < min(0.005 * float(trig_hz), 0.5)).astype(float)
    gate[0] = 1.0
    s = dusg(gate, sr, rise, fall, curve, cycle)
    if out == "pitch":
        f = float(freq) * 2.0 ** (float(octaves) * s)
        return 0.8 * np.sin(TWO_PI * np.cumsum(f) / sr)
    return 0.8 * dc_blocker(s * 2.0 - 1.0, sr, 10.0)


def serge_random_cv(n, sr=DEFAULT_SR, rate=8.0, mode="stepped", lag=0.05, jitter=0.0, seed=0):
    """Serge random voltage in [-1, 1]: a clock (`rate` Hz, `jitter` = relative spread of the intervals) samples white noise; `stepped`
    holds each value, `smooth` runs the held values through a one-pole slew of time constant `lag` s."""
    rng = np.random.default_rng(seed)
    m = int(n * rate / sr * (1 + 2 * abs(jitter))) + 4
    gaps = np.maximum(1.0 + jitter * rng.uniform(-1, 1, m), 0.05) * sr / rate
    idx = np.minimum(np.searchsorted(np.cumsum(gaps), np.arange(n), side="right"), m - 1)
    held = rng.uniform(-1, 1, m + 1)[idx]
    if mode == "stepped":
        return held
    if mode != "smooth":
        raise SpecError("serge_random: mode is stepped | smooth")
    a = math.exp(-1.0 / (max(float(lag), 1e-5) * sr))
    return lfilter([1 - a], [1.0, -a], held)


@sfx("serge_random", "Serge smooth / stepped random voltage generator: noise sampled by a clock, held (stepped) or slewed (smooth), "
     "heard as a pitch sequence (fm) or as the raw voltage.",
     dur=(4.0, "s"), rate=(6.0, "clock Hz"), mode=("stepped", "stepped | smooth"), lag=(0.05, "slew time constant s (smooth)"),
     jitter=(0.0, "relative clock jitter 0..1"), out=("fm", "fm | cv"), freq=(220.0, "Hz centre of the fm output"), octaves=(1.5, "pitch range"),
     seed=(0, "random seed"))
def serge_random(sr=DEFAULT_SR, dur=4.0, rate=6.0, mode="stepped", lag=0.05, jitter=0.0, out="fm", freq=220.0, octaves=1.5, seed=0):
    n = samples(float(np.clip(dur, 0.05, 120.0)), sr)
    cv = serge_random_cv(n, sr, float(np.clip(rate, 0.01, 2000.0)), mode, lag, float(np.clip(jitter, 0.0, 1.0)), int(seed))
    if out == "cv":
        return 0.9 * cv
    return 0.7 * np.sin(TWO_PI * np.cumsum(float(freq) * 2.0 ** (float(octaves) * cv)) / sr)
