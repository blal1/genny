"""Ports of the Klang ``procedural`` audio library, and what was left to integrate from it.

Source: ``synthgen/procedural`` (C++ sounds written for the Klang framework ``klang.h`` plus the
Python families built around the same kernels). Original code Copyright (c) 2025 Chris Nash,
licensed under the **Klang Open License 1.0** (Apache-2.0 plus an attribution clause for
interactive products: a game or installation that ships these sounds must show the Klang logo /
"Powered by Klang" in its credits; see ``genny/physical/KLANG_LICENSE.txt`` and
https://nash.audio/art/). This file is a derivative work: C++ translated to Python/numba, Python
recipes adapted to genny's registries. Every ported sound repeats the attribution in its docstring.

1. C++ ports (``procedural/vehicles/*.h``, klang semantics from ``klang/klang.h``):
   ``toy_boat`` (ToyBoatEngine), ``klang_car`` (FourStrokeEngine/Car and Mini), ``harrier``,
   ``bicycle``. They run at klang's own rate ``fs = 44100`` (klang.h l.1729; several constants in
   the headers are per-sample: ``rate*0.9999``, ``Control::smooth`` 0.999, ``DCF`` r 0.995) and
   are resampled at the boundary. klang's ``min``/``max`` return the type of their FIRST argument
   (klang.h l.269-272), so ``max(0, x)`` truncates ``x`` to an int; the ports keep that (see
   docs/klang.md "klang quirks kept").
   ``klang_rain`` is NOT a port: ``procedural/nature/Nature.h`` holds only ``#pragma once`` (the
   README lists Rain as work in progress). It is a small original model built from the same klang
   primitives; all its constants are marked UNSOURCED.
2. From the Python modules (48 kHz physical kernels in ``genny/physical``): ``ui_tick`` and
   ``ui_chime`` (interface.py), ``damage`` (impacts.py), ``sniff``, ``strain``, ``weight_shift``
   (body.py), ``hand_chime`` (charter.py); the orchestral voices the sampler map
   (instrument_map.py) has and genny lacked (``viola``, ``bassoon``, ``cor_anglais``,
   ``bass_clarinet``, ``piccolo``, ``recorder``); the ``sampler`` layer (sampler.py, for a
   user-supplied folder of recordings) and the ``articulate`` layer (the eight articulations of
   instrument_map.ARTICULATIONS played on any genny instrument).
3. ``coverage()`` / ``python -m genny.klang --qa``: the catalogue/coverage QA idea as a
   library-wide harness that renders every registered name and reports failures and levels.
"""
from __future__ import annotations

import math
import re
import sys
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numba
import numpy as np
from scipy.signal import lfilter, resample_poly

from . import filters as F
from . import instruments as I
from . import osc as O
from .core import DEFAULT_SR, mix, samples, silence
from .acoustic import _blown
from .instruments import _kt, _modes, instrument
from .notes import parse_sequence
from .physical import core as P
from .physical import modal as M
from .physical import particles as PT
from .physical import voice as V
from .sfx import sfx
from .spec import layer_type

KFS = 44100                      # klang::fs default sample rate (klang.h l.1729)
_W = 2.0 * math.pi / KFS         # klang fs.w
_Q0 = 1.0 / math.sqrt(2.0)       # Biquad::Filter::set(f) default Q = root2.inv


# ====================================================================== klang primitives (numba)
@numba.njit(cache=True)
def _u(st):
    """Uniform [0, 1): xorshift32 standing in for C ``rand()`` (deterministic per seed)."""
    s = st[0]
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    st[0] = s
    return s * 2.3283064365386963e-10


@numba.njit(cache=True)
def _bq(kind, f, q):
    """klang ``Filters::Biquad`` coefficients (klang.h l.6212-6357): kind 0 LPF, 1 HPF, 2 BPF
    (constant peak gain, the default). alpha uses Q clamped to >= 0.5. Returns b0,b1,b2,a1,a2."""
    w = f * _W
    c = math.cos(w)
    s = math.sin(w)
    a = s / (2.0 * max(q, 0.5))
    i0 = 1.0 / (1.0 + a)
    a1 = -2.0 * c * i0
    a2 = (1.0 - a) * i0
    if kind == 0:
        return (1.0 - c) * 0.5 * i0, (1.0 - c) * i0, (1.0 - c) * 0.5 * i0, a1, a2
    if kind == 1:
        return (1.0 + c) * 0.5 * i0, -(1.0 + c) * i0, (1.0 + c) * 0.5 * i0, a1, a2
    return a * i0, 0.0, -a * i0, a1, a2


@numba.njit(cache=True)
def _bt(c, Z, k, x):
    """One sample of the transposed direct form II biquad (klang.h l.6233), state row ``k``."""
    y = c[0] * x + Z[k, 0]
    Z[k, 0] = c[1] * x - c[3] * y + Z[k, 1]
    Z[k, 1] = c[2] * x - c[4] * y
    return y


@numba.njit(cache=True)
def _tap(buf, pos, size, d):
    """klang ``Delay<SIZE>::tap(float)`` (klang.h l.3860): linear-interpolated read ``d`` samples
    behind the last written sample."""
    r = (pos - 1) - d
    if r < 0.0:
        r += size
    i = int(r)
    fr = r - i
    j = (i + 1) % size
    return buf[i] + fr * (buf[j] - buf[i])


@numba.njit(cache=True)
def _pulse_avg(p, f, duty):
    """klang ``Fast::Pulse`` (OSM, klang.h l.5632-5843): the ideal +1 (phase < duty) / -1 pulse
    averaged over the step that ends at phase ``p`` (cycles), step ``f`` cycles."""
    if f <= 0.0:
        return 1.0 if p < duty else -1.0
    a = p - f
    hb = math.floor(p) * duty + min(p - math.floor(p), duty)
    ha = math.floor(a) * duty + min(a - math.floor(a), duty)
    return 2.0 * (hb - ha) / f - 1.0


def _fin(y, sr, src=KFS, n=None, peak=None, fade=0.005):
    """Resample ``src`` -> ``sr``, cut to ``n``, remove DC (8 Hz high-pass), fade both ends; set the peak to
    ``peak``, or (None) keep the model's own level and only scale down what would clip
    (procedural/render/render.cpp hard-clips to +-1 instead)."""
    y = np.asarray(y, dtype=np.float64)
    if int(sr) != int(src):
        g = math.gcd(int(sr), int(src))
        y = resample_poly(y, int(sr) // g, int(src) // g)
    if n:
        y = y[:n]
    r = math.exp(-2.0 * math.pi * 8.0 / sr)          # DC blocker: one-pole high-pass at 8 Hz
    y = lfilter([1.0, -1.0], [1.0, -r], y)
    k = max(2, min(len(y) // 2, int(fade * sr)))
    y[:k] *= np.linspace(0.0, 1.0, k)
    y[-k:] *= np.linspace(1.0, 0.0, k)
    p = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak is not None and p > 0:
        y *= peak / p
    elif p > 0.98:
        y *= 0.98 / p
    return y


def _ramp(a, b, n):
    return np.linspace(float(a), float(a) if float(b) < 0 else float(b), n)


def _kn(dur):
    return int(math.ceil(max(0.05, float(dur)) * KFS)) + 64


def _seed(seed):
    return int(np.random.default_rng(int(seed)).integers(1, 2 ** 32 - 1))


# ====================================================================== Toy Boat Engine (Motors.h)
@numba.njit(cache=True)
def _toy_boat_k(n, rate, brk, seed):
    st = np.array([seed], dtype=np.int64)
    out = np.zeros(n)
    Z = np.zeros((5, 2))
    c9 = _bq(2, rate, 15.0)            # bp_9_15
    c590 = _bq(2, 590.0, 4.0)          # bp_590_4
    cb0 = _bq(2, 470.0, 8.0)           # body[0..2]
    cb1 = _bq(2, 780.0, 9.0)
    cb2 = _bq(2, 1024.0, 10.0)
    e10 = math.exp(-10.0 * _W)         # OnePole: exp0 = exp(-f * fs.w)
    e30 = math.exp(-30.0 * _W)
    e1k = math.exp(-1000.0 * _W)
    e100 = math.exp(-100.0 * _W)
    ph = 0.0
    h10 = z10 = l30 = h1k = z1k = h100 = z100 = 0.0
    for i in range(n):
        if brk:                                         # noise >> bp_9_15: sputtering
            m = _bt(c9, Z, 0, 2.0 * _u(st) - 1.0)
        else:                                           # osc(9): regular pulse
            m = math.sin(2.0 * math.pi * ph)
            ph += rate / KFS
            ph -= math.floor(ph)
        x = min(1.0, max(0.0, m * 600.0))               # clip_0_1(mix * 600): exhaust valve
        h10 = 0.5 * (1.0 + e10) * (x - z10) + e10 * h10
        z10 = x
        l30 = (1.0 - e30) * h10 + e30 * l30
        w = 2.0 * _u(st) - 1.0                          # second read of the Noise object
        h1k = 0.5 * (1.0 + e1k) * (w - z1k) + e1k * h1k
        z1k = w
        m = l30 * _bt(c590, Z, 1, h1k)                  # formant: enveloped hip-1000 noise
        b = _bt(cb0, Z, 2, m) + _bt(cb1, Z, 3, m) + _bt(cb2, Z, 4, m)
        h100 = 0.5 * (1.0 + e100) * (b - z100) + e100 * h100
        z100 = b
        out[i] = h100 * 10.0
    return out


@sfx("toy_boat", "Toy boat engine (Klang procedural port of Farnell's putt-putt): a 9 Hz valve pulse gating formant noise through three body resonances.",
     rate=(9.0, "valve pulses per second (the header's constant: 9); the noise bursts come at twice this"),
     broken=(0, "1 = broken engine: the regular pulse is replaced by 9 Hz band-passed noise (sputtering)"),
     dur=(3.0, "s"), seed=(1, "noise seed"))
def toy_boat(sr=DEFAULT_SR, rate=9.0, broken=0, dur=3.0, seed=1):
    """Port of ``ToyBoatEngine`` (procedural/vehicles/Motors.h l.193-251; "A Toy Boat Engine", Farnell,
    Designing Sound p.511). Klang Open License 1.0, (c) 2025 Chris Nash.

    sine 9 Hz (or noise >> BPF 9 Hz Q15) -> clip01(x*600) -> one-pole HPF 10 -> one-pole LPF 30,
    times (noise -> one-pole HPF 1000 -> BPF 590 Q4), into BPF 470 Q8 + 780 Q9 + 1024 Q10 summed,
    one-pole HPF 100, x10. The valve signal is a square differentiated by the 10 Hz high-pass, so
    it has a bump on each edge: 2 x rate noise bursts per second.
    Added: ``rate`` (9 in the header; the C++ only exposes ``broken``)."""
    rate = float(np.clip(rate, 0.5, 60.0))
    y = _toy_boat_k(_kn(dur), rate, bool(int(broken)), _seed(seed))
    return _fin(y, sr, n=samples(dur, sr))


# ====================================================================== Basic Car Engine (Motors.h)
@numba.njit(cache=True)
def _car_k(speed, seed):
    st = np.array([seed], dtype=np.int64)
    n = speed.shape[0]
    out = np.zeros(n)
    SIZE = 3840
    a = np.zeros(SIZE + 1)
    b = np.zeros(SIZE + 1)
    pos = 0
    Z = np.zeros((3, 2))
    clp = _bq(0, 15.0, _Q0)
    chp = _bq(1, 100.0, _Q0)
    fb_last = -1.0
    cbp = _bq(2, 400.0, 0.5)
    ph = 0.0
    dcz = dco = 0.0
    k3 = 0.25 / math.tanh(3.0)
    for i in range(n):
        sp = speed[i]
        nl = _bt(clp, Z, 0, 2.0 * _u(st) - 1.0)         # noise >> lpf(15)
        b[pos] = nl * 30.0                              # n * 30 >> b
        a[pos] = nl * 0.5                               # n * 0.5 >> a
        pos += 1
        if pos == SIZE:
            pos = 0
        fb = 200.0 + sp * 400.0                         # bpf(200 + speed*400): one-arg set -> Q 0.7071
        if fb != fb_last:
            cbp = _bq(2, fb, _Q0)
            fb_last = fb
        n2 = 1.0 - (sp + 0.1) * _bt(cbp, Z, 1, 2.0 * _u(st) - 1.0) * 0.01
        i4 = ph * 4.0                                   # phasor(speed * 10) * 4
        ph += sp * 10.0 / KFS
        ph -= math.floor(ph)
        s = 22.0 - sp * 15.0
        ms = KFS / 250.0 * (0.99 + 0.01 * _u(st))       # fs / 250 * random(0.99, 1.0)
        o = 0.0
        for d in range(4):
            t = (d + 1) * 5.0 * ms
            phase = -(0.75 - d * 0.25) * n2
            mx = math.cos((_tap(a, pos, SIZE, t) + i4 + phase) * 2.0 * math.pi) * (_tap(b, pos, SIZE, t) + s)
            o += 1.0 / (mx * mx + 1.0)
        y = _bt(chp, Z, 2, o * sp * min(sp, 0.25))      # >> hpf(100)
        dco = y - dcz + 0.995 * dco                     # >> dc (DCF r = 0.995)
        dcz = y
        out[i] = k3 * math.tanh(dco * 3.0)              # Car: 0.25 * tanh(out * 3) / tanh(3)
    return out


def car_fundamental(rpm: float) -> float:
    """Pulse rate of the basic engine: the phasor runs at 10*rpm/7000 Hz, ``cos`` makes 4 cycles
    per phasor cycle, each cylinder pulses at both zero crossings and the four cylinders fall on
    two interleaved phases: 16 pulses per phasor cycle = rpm * 16 / 700 Hz."""
    return float(rpm) * 16.0 / 700.0


# ====================================================================== Advanced Car Engine: Mini
_MINI_F = np.array([86.1, 64.6, 43.1, 53.8, 99.6, 21.5, 110.4, 75.4, 175.0, 118.4, 131.9, 142.7, 166.9, 8.1, 185.7])
_MINI_DB = np.array([44.5, 43.8, 40.3, 37.0, 35.9, 35.4, 33.8, 31.5, 29.0, 28.5, 26.1, 24.8, 18.7, 18.4, 17.7])
_MINI_G = 10.0 ** ((_MINI_DB - 48.0) / 20.0)              # dB(gain - 48) -> Amplitude


def mini_rate(rpm: float, throttle: float = 0.0) -> float:
    """Settled ``rate`` of the Mini engine (power 1, rev 0): r - 0.05 r^2 + min(throttle, 0.707),
    r = rpm / 900. The loudest partial sits at 86.1 * rate Hz."""
    r = max(0.0, float(rpm) / 900.0)
    return r - 0.05 * r * r + min(float(throttle), 0.707)


@numba.njit(cache=True)
def _mini_k(rpm, thr, power, rev, done, seed, F_, G_):
    st = np.array([seed], dtype=np.int64)
    n = rpm.shape[0]
    out = np.zeros(n)
    Z = np.zeros((5, 2))
    e0 = _bq(2, 65.0, 3.0)                              # engine noise character: 4 BPF + shelf
    e1 = _bq(2, 1672.0, 3.0)
    e2 = _bq(2, 3316.0, 6.0)
    e3 = _bq(2, 9717.0, 6.0)
    g0 = 10.0 ** (14.2 / 20.0)
    g1 = 10.0 ** (10.3 / 20.0)
    g2 = 10.0 ** (6.9 / 20.0)
    g3 = 10.0 ** (1.1 / 20.0)
    shelf = 10.0 ** (-25.0 / 20.0)
    ph = np.zeros(15)
    SIZE = 512
    comb = np.zeros(SIZE + 1)
    cpos = 0
    rate = 0.0
    glp = 0.0
    lf = lq = -1.0
    cl = _bq(0, 5000.0, 5.0)
    e_slow = math.exp(-0.05 * _W)                       # throttle_lpf.set(0.05): start-up
    e_fast = math.exp(-0.5 * _W)                        # throttle_lpf.set(0.5) once rev has finished
    for i in range(n):
        pw = power[i]
        if pw == 0.0:                                   # skip processing when idle
            continue
        r = max(0.0, rpm[i] / 900.0)                    # 900 rpm = 1.0 (idle)
        r -= 0.05 * r * r                               # energy loss at high revs
        th = thr[i]
        new_rate = (r + rev[i] + min(th, 0.707)) * pw
        fin = done[i] != 0
        if (not fin) or new_rate > rate:                # rev up is slower than rev down
            rate = rate * 0.9999 + 0.0001 * new_rate
        else:
            rate = rate * 0.999 + 0.001 * new_rate
        gi = (th - 0.5) * 2.0                           # max(0, x): klang returns int -> truncation
        gi = float(int(gi)) if gi > 0.0 else 0.0
        e = e_fast if fin else e_slow
        glp = (1.0 - e) * gi + e * glp
        gas = glp * (1.0 - min(0.75, abs((rate * 0.125) ** 2)))
        g2_ = gas * gas
        w = 2.0 * _u(st) - 1.0
        en = w * shelf + _bt(e0, Z, 0, w) * g0 + _bt(e1, Z, 1, w) * g1 + _bt(e2, Z, 2, w) * g2 + _bt(e3, Z, 3, w) * g3
        tone = 0.0
        for p in range(15):                             # osc[p].set(f * rate * random(0.8, 1.2))
            tone += math.sin(2.0 * math.pi * ph[p]) * G_[p]
            ph[p] += F_[p] * rate * (0.8 + 0.4 * _u(st)) / KFS
            ph[p] -= math.floor(ph[p])
        tone = 1.5 * math.tanh(tone * (1.0 + g2_ * 0.25) / 1.5)      # softclip: exhaust rasp
        et = (1.0 - gas) + gas * tone
        m1 = 1.0 if ((7.5 - rate) ** 2 / 50.0 + 0.125) > 1.0 else 0.0     # min(1, x) as int
        et *= 0.5 + th * 0.5 + g2_ * 0.1 * m1
        am = tone * tone * tone * et * en
        fbk = _tap(comb, cpos, SIZE, (0.001 + 0.001 * _u(st)) * KFS)
        m2 = float(int(2.0 - rate)) if (2.0 - rate) > 0.0 else 0.0        # max(0, 2 - rate) as int
        am += gas * fbk * 0.99 * m2
        comb[cpos] = am
        cpos += 1
        if cpos == SIZE:
            cpos = 0
        f = min(5000.0 * (1.0 + (rate / 14.0) ** 2), 0.45 * KFS)          # guard: below Nyquist
        q = float(int(5.0 + g2_ * 5.0))                                   # max(1, x) as int
        if f != lf or q != lq:
            cl = _bq(0, f, q)
            lf = f
            lq = q
        tn = max(0.5, 1.0 - abs((rate * 0.25 - 0.75) ** 2)) * (1.0 - (rate - 1.0) * 0.01)
        o = tone * tn + en * (rate * 0.02) + _bt(cl, Z, 4, am) * 0.075 * (1.0 + g2_)
        out[i] = o * pw * 0.1                           # Mini: engine(...) * 0.1
    return out


def _mini_env(n, rng, off_at):
    """The starter and rev envelopes of ``Mini::Engine::set`` at ignition (klang Envelope = linear
    segments through the points, holding the last; ``finished()`` once the last point is reached),
    and ``starter.release(2)`` (linear to 0 over 2 s) at ``off_at`` seconds."""
    delay = rng.uniform(0.25, 0.75)
    t = np.arange(n) / KFS
    power = np.interp(t, [0.0, delay, delay + 0.25, delay + 0.5], [0.5, 1.0, 2.0, 1.0])
    rev = np.interp(t, [0.0, delay - 0.1, delay, delay + 0.125, delay + 0.25, delay + 5.0],
                    [0.0, 0.0, 1.0, rng.uniform(2.0, 5.0), 0.0, 0.0])
    done = (t >= delay + 5.0).astype(np.uint8)
    if off_at >= 0:
        k = min(n - 1, int(off_at * KFS))
        power[k:] = np.maximum(0.0, power[k] * (1.0 - (t[k:] - t[k]) / 2.0))
    return power, rev, done


@sfx("klang_car", "Car engine (Klang procedural ports): `basic` four-stroke pulse engine, or `mini`: a Mini resynthesised from 15 measured partials with exhaust rasp, throttle and ignition.",
     kind=("basic", "basic (FourStrokeEngine) | mini (Advanced Car Engine)"),
     rpm=(3000.0, "crank speed 0..7000 (basic is silent near 0; idle is about 900)"),
     rpm_end=(-1.0, "rpm at the end for a linear rev sweep; -1 = steady"),
     throttle=(0.3, "mini only, 0..1: adds up to 0.707 to the engine rate; exactly 1.0 = floored (over-rev rasp)"),
     ignition=(0, "mini only: 1 = switched on at t=0 (starter, rev blip, 0.75-1.25 s); 0 = already running"),
     off_at=(-1.0, "mini only: second at which the ignition is switched off (2 s run-down); -1 = never"),
     dur=(3.0, "s"), seed=(1, "noise / start-up seed"))
def klang_car(sr=DEFAULT_SR, kind="basic", rpm=3000.0, rpm_end=-1.0, throttle=0.3, ignition=0, off_at=-1.0, dur=3.0, seed=1):
    """Ports of ``Car``/``FourStrokeEngine`` and ``Mini`` (procedural/vehicles/Motors.h). Klang Open
    License 1.0, (c) 2025 Chris Nash.

    basic (l.253-331): speed = rpm/7000; phasor at 10*speed Hz; per cylinder d = 0..3,
    1 / (1 + (cos(2 pi (a(t) + 4 phasor + phase_d)) (b(t) + 22 - 15 speed))^2) with the 15 Hz
    noise read from two delay lines at (d+1)*20 ms; x speed x min(speed, 0.25) -> HPF 100 -> DC
    filter -> 0.25 tanh(3x)/tanh(3). Pulse rate = rpm * 16 / 700 Hz (``car_fundamental``).

    mini (l.26-191): 15 sines at ``partials[p].frequency * rate`` (each jittered +-20 % every
    sample), gains dB - 48; rate follows (rpm/900 - 0.05 (rpm/900)^2 + rev + min(throttle, 0.707))
    x power, slowly up (0.0001) and fast down (0.001); soft clip 1.5 tanh; engine noise = white
    x -25 dB shelf + BPF 65/1672/3316/9717 Hz; tone^3 x throttle term x noise through a 1-2 ms
    comb and a resonant LPF at 5000 (1 + (rate/14)^2) Hz. ``Gear`` is read but unused in the
    header and is not exposed. The loudest partial is 86.1 * ``mini_rate(rpm, throttle)`` Hz."""
    n = _kn(dur)
    r = np.clip(_ramp(rpm, rpm_end, n), 0.0, 7000.0)
    if kind == "basic":
        y = _car_k(r / 7000.0, _seed(seed))
    elif kind == "mini":
        rng = np.random.default_rng(int(seed))
        pre = 0 if int(ignition) else 6 * KFS           # start-up finishes at delay + 5 s <= 5.75 s
        power, rev, done = _mini_env(n + pre, rng, float(off_at) + pre / KFS if float(off_at) >= 0 else -1.0)
        thr = np.full(n + pre, float(np.clip(throttle, 0.0, 1.0)))
        y = _mini_k(np.concatenate([np.full(pre, r[0]), r]), thr, power, rev, done,
                    int(rng.integers(1, 2 ** 32 - 1)), _MINI_F, _MINI_G)[pre:]
    else:
        raise ValueError(f"klang_car kind {kind!r}: basic|mini")
    return _fin(y, sr, n=samples(dur, sr))


# ====================================================================== Jet (Harrier.h)
_TURB_F = np.array([3097.0, 4495.0, 5588.0, 7471.0, 11000.0])
_TURB_G = np.array([0.25, 0.25, 1.0, 0.4, 0.4])


def turbine_gain(speed: float) -> float:
    """``Turbine::set`` gain curve (Harrier.h l.230-240)."""
    if speed < 0.125:
        return speed * 8.0
    if speed < 0.25:
        return 1.0
    if speed < 0.75:
        return abs(0.5 - speed) * 2.0 + 0.5
    return 1.0 - (speed - 0.5)


@numba.njit(cache=True)
def _pdcos(f):
    """Pd's polynomial cosine (Harrier.h l.88): valid on [-pi/2, pi/2], 0 outside."""
    if f >= -0.5 * math.pi and f <= 0.5 * math.pi:
        g = f * f
        return ((g * g * g * (-1.0 / 720.0) + g * g * (1.0 / 24.0)) - g * 0.5) + 1.0
    return 0.0


@numba.njit(cache=True)
def _harrier_k(speed_c, alt, gain, echo_fb, seed, TF, TG):
    st = np.array([seed], dtype=np.int64)
    n = speed_c.shape[0]
    out = np.zeros(n)
    ph = np.zeros(5)
    Z = np.zeros((4, 2))
    chp = _bq(1, 120.0, _Q0)                            # Burn::hpf
    # pd::bpf(8000, 0.5): omega/q > 1 -> r = 0 -> a pure gain of 2 at 44.1 kHz
    om = 8000.0 * _W
    omr = min(1.0, om / 0.5)
    rb = 1.0 - omr
    b1c = 2.0 * _pdcos(om) * rb
    b2c = -rb * rb
    bg = 2.0 * omr * (omr + rb * om)
    bx1 = bx2 = 0.0
    re0 = im0 = re1 = im1 = 0.0
    lo = 0.0
    ESIZE = 192000
    echo = np.zeros(ESIZE + 1)
    epos = 0
    cw = _bq(2, 220.0, 3.0)
    wf = -1
    sm = speed_c[0]            # the port starts settled; the C++ smooths up from 0 (a 100 ms chirp)
    for i in range(n):
        sm = sm * 0.999 + 0.001 * speed_c[i]            # controls[0].smooth()
        sp = sm
        al = alt[i]
        # ---- Turbine: 5 partials at speed * f, clipped, shaped by speed
        t = 0.0
        for k in range(5):
            t += math.sin(2.0 * math.pi * ph[k]) * TG[k]
            ph[k] += sp * TF[k] / KFS
            ph[k] -= math.floor(ph[k])
        t = min(0.9, max(-0.9, t))
        if sp < 0.125:
            tg = sp * 8.0
        elif sp < 0.25:
            tg = 1.0
        elif sp < 0.75:
            tg = abs(0.5 - sp) * 2.0 + 0.5
        else:
            tg = 1.0 - (sp - 0.5)
        # ---- Burn: noise >> pd::bpf >> vcf0 >> hpf, overdriven, clipped, >> vcf1
        od = 30.0 if sp < 0.5 else 30.0 + (sp - 0.5) * 30.0
        od *= 1.0 + min(0.25, max(-0.5, sp * (5.0 - al) * 0.2))          # ground resistance
        od *= min(al, 2.0) * 0.5
        x = (2.0 * _u(st) - 1.0) + b1c * bx1 + b2c * bx2
        bx2 = bx1
        bx1 = x
        x *= bg
        f0 = max(0.001, sp * sp * 150.0)                # vcf0.set(speed^2 * 150, 1)
        o0 = f0 * _W
        r0 = max(0.0, 1.0 - o0)
        c0 = r0 * _pdcos(o0)
        s0 = r0 * math.sin(o0)
        nr = (2.0 - 2.0 / 3.0) * (1.0 - r0) * x + c0 * re0 - s0 * im0
        im0 = s0 * re0 + c0 * im0
        re0 = nr
        y = min(1.0, max(-1.0, _bt(chp, Z, 0, re0) * od)) * 0.1
        f1 = max(0.001, sp * 12000.0)                   # vcf1.set(speed * 12000, 0.6)
        o1 = f1 * _W
        r1 = max(0.0, 1.0 - o1 / 0.6)
        c1 = r1 * _pdcos(o1)
        s1 = r1 * math.sin(o1)
        nr = (2.0 - 2.0 / 2.6) * (1.0 - r1) * y + c1 * re1 - s1 * im1
        im1 = s1 * re1 + c1 * im1
        re1 = nr
        # ---- pd::lop(11000 (1 - speed/2))
        cf = min(1.0, max(0.0, 11000.0 * (1.0 - sp * 0.5) * _W))
        lo = cf * (t * tg * (0.03 * (1.0 - sp * 0.5)) + re1) + (1.0 - cf) * lo
        o = lo
        # ---- wind: all four min/max below return ints in klang
        mx = 500 if 500.0 > al / 10.0 else int(al / 10.0)
        v = 500.0 - mx + sp * 200.0
        fi = 10000 if 10000.0 < v else int(v)
        wn = 2.0 * _u(st) - 1.0
        if fi >= 1:
            if fi != wf:
                cw = _bq(2, float(fi), math.sqrt(2.0))
                wf = fi
            wv = _bt(cw, Z, 1, wn)
        else:                  # guard: the C++ filter is undriven at 0 Hz and unstable below it
            wv = 0.0
            Z[1, 0] = 0.0
            Z[1, 1] = 0.0
            wf = -1
        ws = max(0.0, sp - 0.6)
        ag = 200.0 if 200.0 < 0.5 * al else float(int(0.5 * al))
        ag *= 0.5 + sp * 3.0
        ag = float(int(ag)) if ag > 0.0 else 0.0
        o += _bt(_bq(0, 1000.0 - sp * 500.0, math.sqrt(2.0)), Z, 2, ws * ws * wv * ag)
        # ---- echo: Delay<192000>, read max(10, speed * fs) behind, through lpf2
        et = int(sp * KFS)
        if et < 10:
            et = 10
        o = _bt(_bq(0, 11000.0 - sp * 4000.0, _Q0), Z, 3, o + _tap(echo, epos, ESIZE, float(min(et, ESIZE))))
        # C++: out * max(0, speed * 0.75) -> int -> 0 for every speed <= 1: nothing is ever fed back
        echo[epos] = o * (echo_fb * sp * 0.75 if echo_fb > 0.0 else float(int(sp * 0.75)))
        epos += 1
        if epos == ESIZE:
            epos = 0
        out[i] = o * gain
    return out


@sfx("harrier", "Jet engine (Klang procedural port of the Harrier): turbine whine of five partials that follow the engine speed, an overdriven low-noise burn, and wind at speed and height.",
     speed=(0.5, "engine speed 0..1: turbine partials at speed x 3097/4495/5588/7471/11000 Hz, burn rumble at speed^2 x 150 Hz"),
     speed_end=(-1.0, "speed at the end for a spool-up / spool-down; -1 = steady"),
     altitude=(2.0, "0 = on the ground with no burn (turbine only, the C++ default); >= 2 = full burn; wind grows with min(200, altitude/2) above speed 0.6; usable up to about 5000"),
     gain=(0.5, "output gain 0..1 (the header's Gain dial)"),
     echo=(0.0, "0 = as the C++ behaves (its echo feedback truncates to 0); 1 = the evidently intended echo of period `speed` seconds, feedback 0.75 x speed"),
     dur=(3.0, "s"), seed=(1, "noise seed"))
def harrier(sr=DEFAULT_SR, speed=0.5, speed_end=-1.0, altitude=2.0, gain=0.5, echo=0.0, dur=3.0, seed=1):
    """Port of ``Harrier`` with ``Turbine``, ``Burn`` and the Pd-style ``lop``/``bpf``/``vcf``
    filters (procedural/vehicles/Harrier.h). Klang Open License 1.0, (c) 2025 Chris Nash.

    out = lop(turbine x 0.03 (1 - speed/2) + burn) + LPF(wind) [+ echo] -> LPF(11000 - 4000 speed),
    x gain. Burn = clip(HPF120(vcf(2 x noise, speed^2 x 150 Hz, Q 1)) x overdrive) x 0.1 through
    vcf(speed x 12000 Hz, Q 0.6); overdrive 30..45 x ground term x min(altitude, 2)/2.
    Deviations (all documented in docs/klang.md): the speed smoother starts settled; the wind
    band-pass is muted where the header's centre frequency is <= 0 Hz (altitude above
    5000 + 2000 speed, where the C++ filter is unstable and the output becomes NaN); what would
    exceed full scale (wind at height) is scaled down instead of hard-clipped."""
    n = _kn(dur)
    sp = np.clip(_ramp(speed, speed_end, n), 0.0, 1.0)
    alt = np.full(n, float(np.clip(altitude, 0.0, 100000.0)))
    y = _harrier_k(sp, alt, float(np.clip(gain, 0.0, 1.0)), float(np.clip(echo, 0.0, 1.0)), _seed(seed), _TURB_F, _TURB_G)
    return _fin(y, sr, n=samples(dur, sr))


# ====================================================================== Bicycle (Bicycle.h)
_WHEEL_DB = np.array([27.8, 33.7, 30.3, 28.8, 26.1, 22.0, 32.1, 24.6])
_WHEEL_G = 10.0 ** ((_WHEEL_DB - 38.0) / 20.0)             # dB(gain - 38) -> Amplitude


@numba.njit(cache=True)
def _bicycle_k(ped_c, wheel, pedal, seed, WG):
    st = np.array([seed], dtype=np.int64)
    n = ped_c.shape[0]
    out = np.zeros(n)
    Z = np.zeros((6, 2))
    c5887 = _bq(2, 5887.0, 1.2)        # Wheel::noise_bpf
    c12 = _bq(2, 12.0, _Q0)            # Wheel::osc_bpf.set(12)
    c2550 = _bq(2, 2550.0, 5.0)        # Chain::noise_bpf[0..2]
    c4250 = _bq(2, 4250.0, 15.0)
    c6500 = _bq(2, 6500.0, 25.0)
    cout = _bq(2, 11000.0, 1.0)        # Bicycle::bpf(11000, 1)
    wph = np.zeros(8)
    pph = 0.0                          # Pedal phasor
    cph = 0.0                          # Chain pressure sine
    uph = 0.0                          # Wheel pulse
    sm = ped_c[0]                      # the port starts settled (the C++ smooths up from 0)
    pedalling = ped_c[0]
    for i in range(n):
        sm = sm * 0.999 + 0.001 * ped_c[i]              # controls[0].smooth()
        pedalling += (sm - pedalling) * 0.001
        ws = wheel[i]
        ps = pedal[i] / 2.0
        # ---- Pedal: sin(phasor * 2 pi)^2, two thumps per crank cycle
        s = math.sin(pph * 2.0 * math.pi)
        energy = s * s
        pph += ps / KFS
        pph -= math.floor(pph)
        # ---- Wheel
        rate = ws / 8.8
        w = 2.0 * _u(st) - 1.0
        w = w * 0.5 + _bt(c5887, Z, 0, w) * 0.5
        tone = 0.0
        for p in range(8):                              # osc[p].set(8.8 * (p + 1) * rate)
            tone += math.sin(2.0 * math.pi * wph[p]) * WG[p]
            wph[p] += ws * (p + 1) / KFS
            wph[p] -= math.floor(wph[p])
        stp = ws / KFS
        flt = _bt(c12, Z, 1, _pulse_avg(uph, stp, 0.1))  # pulse(8.8 * rate, duty 0.1) >> osc_bpf
        uph += stp
        uph -= math.floor(uph)
        wn = tone ** 4 * min(0.01, rate * 0.005) * w + flt ** 3 * w * rate ** 3
        # ---- Chain (input = pedal energy): three reads of the Noise object
        cr = ps * 22.0
        nn = (_bt(c2550, Z, 2, 2.0 * _u(st) - 1.0) * (0.1 + energy * energy * 0.05)
              + _bt(c4250, Z, 3, 2.0 * _u(st) - 1.0) * 0.5 + _bt(c6500, Z, 4, 2.0 * _u(st) - 1.0))
        pr = math.sin(2.0 * math.pi * cph)              # pressure(rate * (0.9 + in * 0.2))
        cph += cr * (0.9 + energy * 0.2) / KFS
        cph -= math.floor(cph)
        c = (0.5 + energy * 0.25) * nn * nn * pr
        c = min(1.0, max(-1.0, c)) * 5.0                # saturate(out) * 5
        if cr < 10.0:
            c *= cr * 0.1
        cn = 0.7 * c + 0.25 * wn
        out[i] = _bt(cout, Z, 5, pedalling * cn + 0.25 * (1.0 - 0.75 * pedalling) * wn) * 0.6
    return out


@sfx("bicycle", "Bicycle (Klang procedural port): freewheel whirr and tick once per wheel turn, chain noise pulsed by the pedal strokes.",
     pedalling=(0.5, "0 = coasting (wheel only) .. 1 = pedalling hard (chain dominates)"),
     wheel_speed=(6.0, "wheel turns per second, 0..50: the whirr pattern and the tick repeat at this rate"),
     pedal_speed=(1.0, "pedal strokes per second, 0..4 (the crank turns at half this); the chain is modulated at 11 x this Hz"),
     dur=(3.0, "s"), seed=(1, "noise seed"))
def bicycle(sr=DEFAULT_SR, pedalling=0.5, wheel_speed=6.0, pedal_speed=1.0, dur=3.0, seed=1):
    """Port of ``Bicycle`` with ``Pedal``, ``Chain`` and ``Wheel`` (procedural/vehicles/Bicycle.h).
    Klang Open License 1.0, (c) 2025 Chris Nash.

    Wheel: 8 harmonics of wheel_speed (gains dB - 38), tone^4 x min(0.01, rate 0.005) x noise
    + (10 % pulse at wheel_speed through BPF 12 Hz)^3 x noise x rate^3, rate = wheel_speed / 8.8.
    Chain: (0.5 + e/4) x n^2 x sin at 22 x pedal_speed/2 x (0.9 + 0.2 e) Hz, saturated x5, where
    n = noise through BPF 2550/4250/6500 Hz and e = sin^2 of the crank phasor.
    out = BPF 11 kHz Q1 of (pedalling x (0.7 chain + 0.25 wheel) + 0.25 (1 - 0.75 pedalling) wheel)
    x 0.6. The commented-out frame resonances of the header are not ported. Above about
    wheel_speed 15 the rate^3 tick term exceeds full scale; the output is then scaled down."""
    n = _kn(dur)
    y = _bicycle_k(np.full(n, float(np.clip(pedalling, 0.0, 1.0))), np.full(n, float(np.clip(wheel_speed, 0.0, 50.0))),
                   np.full(n, float(np.clip(pedal_speed, 0.0, 4.0))), _seed(seed), _WHEEL_G)
    return _fin(y, sr, n=samples(dur, sr))


# ====================================================================== Rain (Nature.h is empty)
def rain_rate(intensity: float) -> float:
    """Drops per second heard individually. UNSOURCED: 2 + 600 intensity^2."""
    return 2.0 + 600.0 * float(np.clip(intensity, 0.0, 1.0)) ** 2


def rain_events(dur: float, intensity: float, seed: int = 1) -> np.ndarray:
    """Poisson drop onset times (s) for ``klang_rain``."""
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.uniform(0.0, float(dur), rng.poisson(rain_rate(intensity) * float(dur))))


@numba.njit(cache=True)
def _drops_k(n, t0, f, a, tau):
    out = np.zeros(n)
    for e in range(t0.shape[0]):
        i0 = int(t0[e] * KFS)
        w = f[e] * _W
        d = math.exp(-1.0 / (tau[e] * KFS))
        g = a[e]
        for k in range(min(n - i0, int(7.0 * tau[e] * KFS))):
            out[i0 + k] += g * math.sin(w * k)
            g *= d
    return out


@sfx("klang_rain", "Rain built from klang primitives (the procedural library's Rain is unwritten): Poisson drop ticks over a filtered-noise bed; both grow with intensity.",
     intensity=(0.5, "0 drizzle .. 1 downpour: 2 + 600 x intensity^2 audible drops per second, and the bed level"),
     hiss=(1.0, "level of the continuous bed (the countless far drops); 0 = only the individual drops"),
     tone=(1.0, "0.5 darker (soft ground) .. 2 brighter (hard surface): scales drop pitch and bed cutoff"),
     dur=(4.0, "s"), seed=(1, "seed"))
def klang_rain(sr=DEFAULT_SR, intensity=0.5, hiss=1.0, tone=1.0, dur=4.0, seed=1):
    """NOT a port: ``procedural/nature/Nature.h`` contains only ``#pragma once`` (README: "Rain
    (work-in-progress)"). Original model in the style of the library, built from klang's
    primitives (white noise, one-pole high-pass, biquad low-pass; klang.h Filters).
    UNSOURCED, all of it: drop = sine tick 1.5-7 kHz decaying in 1.5-5 ms with amplitude u^3;
    bed = noise -> one-pole HPF 900 Hz -> LPF 5.5 kHz at 0.02 + 0.1 intensity."""
    it = float(np.clip(intensity, 0.0, 1.0))
    tn = float(np.clip(tone, 0.25, 3.0))
    n = _kn(dur)
    rng = np.random.default_rng(int(seed))
    t0 = rain_events(n / KFS, it, int(seed))
    m = len(t0)
    y = _drops_k(n, t0, np.minimum(1500.0 * (7000.0 / 1500.0) ** rng.random(m) * tn, 0.4 * KFS),
                 0.05 + 0.3 * rng.random(m) ** 3, rng.uniform(0.0015, 0.005, m))
    if hiss > 0:
        e = math.exp(-900.0 * _W)
        bed = lfilter([0.5 * (1 + e), -0.5 * (1 + e)], [1.0, -e], rng.uniform(-1, 1, n))
        c = _bq(0, min(5500.0 * tn, 0.4 * KFS), _Q0)
        y += lfilter(c[:3], [1.0, c[3], c[4]], bed) * (0.02 + 0.1 * it) * float(hiss)
    return _fin(y, sr, n=samples(dur, sr))


# ====================================================================== Python families (48 kHz)
S = P.seconds


def _unit(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def _hip(x, fc):
    """Pd ``hip~``: x minus its one-pole low-pass."""
    return x - P.one_pole_lp(np.ascontiguousarray(x, dtype=np.float64), P.one_pole_coef(fc))


def _lop(x, fc):
    return P.one_pole_lp(np.ascontiguousarray(x, dtype=np.float64), P.one_pole_coef(fc))


def _strk(table, fs=1.0, ms=1.0, pos=0.0, **kw):
    return M.strike(table, fs, ms, pos, None, **kw)


def _panel(y, gin):
    """Farnell switch panel body: 50 ms delay fed back through bp(700, Q3) x 0.1, dry + body."""
    from . import foley as FO           # the numba kernels of the click models live there
    return y + FO._panel_k(np.ascontiguousarray(y, dtype=np.float64), S(0.05), 700.0, 3.0, 0.1, float(gin), float(P.SR))


def _paper_slide(dur, rng, res=0.5):
    """Paper friction: STK Sandpaper driven by the Farnell GRF hump as the hand's pressure."""
    x = np.arange(S(dur)) / max(S(dur) - 1, 1)
    return PT.shaker("Sandpaper", dur, rng, energy_curve=1.5 * 3.3333 * x * (1 - x) ** 2 * (1 + x),
                     resonance_scale=res, energy_mode="level")


def _cover_thud(ms=25.0, vel=1.0):
    """A soft cover landing: Hertz felt pulse into the Faust big square body."""
    force = np.zeros(S(0.3))
    p = P.felt_pulse(ms, 2.5) * vel
    force[:len(p)] = p
    return M.render_modes(force, M.resolve("body_squareBig", 1.0))


def _csmode(x, f, q):
    """Csound ``mode``: two-pole at f, bandwidth f/Q."""
    r = math.exp(-math.pi * f / (q * P.SR))
    th = 2 * math.pi * f / P.SR
    return P.biquad(np.ascontiguousarray(x, dtype=np.float64), math.sin(th), 0.0, 0.0, -2 * r * math.cos(th), r * r)


def _ui(kind, rng):
    from . import foley as FO
    if kind == "focus":                # stick.sy x0.3, 1 ms contact, panel body
        return _panel(P.fit(_strk("sy_stick", 0.3, 1.0), S(0.065)), 1.0)
    if kind == "back":                 # the tick lower, a second contact 10 ms later at half force
        y = np.zeros(S(0.075))
        P.place(y, _strk("sy_stick", 0.25, 1.2), 0.0)
        P.place(y, _strk("sy_stick", 0.25, 1.2, velocity=0.5), 0.010)
        return y
    if kind == "activate":             # 3cogs tock1 + shortping + muted wok.sy, bodyresonance 0.09
        n = S(0.13)
        nz = _hip(rng.uniform(-1, 1, n), 300.0)
        c = (FO._mclick(nz, [(6666, 40, 30), (7012, 40, 10), (1500, 35, 10)], 0.002, 0.33)
             + FO._mclick(nz, [(4535, 45, 70), (8325, 40, 45), (11023, 46, 11)], 0.020, 0.33)
             + FO._mclick(nz, [(3567, 40, 40), (4290, 20, 22), (6470, 20, 8)], 0.020, 0.33))
        c = _hip(c, 100.0)
        iron = P.fit(_strk("sy_wok", 1.0, 0.5, d_scale=15.0, dur=0.13), n)
        iron *= 0.25 * np.max(np.abs(c)) / (np.max(np.abs(iron)) + 1e-12)
        return FO._body(c + FO._shortping(n, 0.002, 0.02) + iron, 0.09)
    if kind == "refuse":               # computertower.sy, 35 ms contact, dscale 10 + clockhand clunk
        n = S(0.11)
        thud = P.fit(_strk("sy_computertower", 1.0, 35.0, d_scale=10.0), n)
        clunk = _lop(FO._mclick(_hip(rng.uniform(-1, 1, n), 200.0), FO.HAND, 0.0, 3.0), 800.0)
        return thud + clunk * 0.2 * np.max(np.abs(thud)) / (np.max(np.abs(clunk)) + 1e-12)
    if kind == "toggle":               # Farnell four-click panel switch
        nz = rng.uniform(-1, 1, S(0.11))
        return _panel(FO._switchclick(nz, 1, 50, 3000, 0.0) + FO._switchclick(nz, 1, 20, 4000, 0.0)
                      + FO._switchclick(nz, 1, 20, 5000, 0.010) + FO._switchclick(nz, 1, 50, 7000, 0.010), 10.0)
    if kind in ("page", "save", "load"):   # heavy page turn; book closing; book opening
        ln, s_at, s_d, s_r, s_g, c_at, c_ms, c_v, c_g = {
            "page": (0.2, 0.0, 0.18, 0.5, 1.0, 0.150, 12.0, 0.5, 0.3),
            "save": (0.42, 0.0, 0.15, 0.45, 0.6, 0.200, 25.0, 1.0, 1.0),
            "load": (0.3, 0.090, 0.18, 0.5, 0.7, 0.0, 20.0, 0.7, 0.7)}[kind]
        y = np.zeros(S(ln))
        P.place(y, s_g * _unit(_paper_slide(s_d, rng, s_r)), s_at)
        P.place(y, c_g * _unit(_cover_thud(c_ms, c_v)), c_at)
        return y
    if kind in ("dialog_open", "dialog_close"):
        opening = kind == "dialog_open"
        n = S(0.1 if opening else 0.12)
        nz = _hip(rng.uniform(-1, 1, n), 300.0)
        latch = FO._mclick(nz, FO.CATCH, 0.0, 0.33)
        if not opening:                # the end-stop clunk at the delay chain's 6 ms outlet
            latch = latch + FO._mclick(nz, FO.ENDSTOP, 0.006, 0.33)
        latch = _hip(latch * 2.0, 100.0)
        y = np.zeros(n)
        if opening:                    # catch releases, the lid knocks 12 ms later
            y += 0.5 * _unit(latch)
            P.place(y, _unit(P.fit(_strk("sy_stick", 0.35, 1.0, velocity=0.8), n)), 0.012)
        else:                          # the lid first, then catch and end-stop
            P.place(y, _unit(P.fit(_strk("sy_stick", 0.3, 1.2), n)), 0.0)
            P.place(y, 0.6 * _unit(latch), 0.012)
        return _panel(y, 1.0)
    if kind == "hint":                 # Csound "felt on wood": exciter 80/180 Hz, resonator 440/630 Hz
        shock = np.zeros(S(0.32))
        shock[0] = 1.0
        exc = np.clip(0.5 * (_csmode(shock, 80.0, 8.0) + _csmode(shock, 180.0, 3.0)), 0.0, 3.0)
        return exc + 0.5 * (_csmode(exc, 440.0, 60.0) + _csmode(exc, 630.0, 53.0))
    raise ValueError(f"ui_tick kind {kind!r}: {'|'.join(UI_KINDS)}")


UI_KINDS = ("focus", "back", "activate", "refuse", "toggle", "page", "save", "load", "dialog_open", "dialog_close", "hint")


@sfx("ui_tick", "Physical interface cues (wooden tick, iron click, dull refusal, panel switch, page and book, latch and lid, felt knock): material sounds instead of beeps.",
     kind=("focus", "focus (wood tick) | back (double tick) | activate (iron click) | refuse (dull thud) | toggle (panel switch) | page | save (book closes) | load (book opens) | dialog_open | dialog_close | hint (felt knock)"),
     seed=(0, "noise seed; the same seed always gives the same cue (a UI action must sound the same every time)"))
def ui_tick(sr=DEFAULT_SR, kind="focus", seed=0):
    """Port of the ``ui_*`` recipes of procedural/interface.py (Klang Open License 1.0, (c) 2025
    Chris Nash): modal strikes on the van den Doel measured objects (stick.sy wood, wok.sy iron,
    computertower.sy sheet-metal box), Farnell click clusters (mclick / clickfactory, switchclick,
    shortping) and bodies (bodyresonance~, switch panel), STK Sandpaper as paper friction, a
    Hertz felt pulse on the Faust big square body as a soft cover, Csound ``mode`` felt-on-wood.
    The click kernels are shared with ``genny.foley``. The game-specific ``c_*`` charter set
    (tuned bell motifs) is not ported."""
    return _fin(_ui(str(kind), np.random.default_rng(int(seed))), sr, src=P.SR, peak=0.6, fade=0.002)


@sfx("ui_chime", "Notification chimes from physical bells: Risset bell (discovery), a telephone bell pair (achievement), a low muted English bell (alert), a fingertip on a glass rim (spell).",
     kind=("discovery", "discovery | achievement | alert | spell"),
     pitch=(1.0, "pitch ratio applied to the cue (0.5 = an octave down)"), seed=(0, "seed"))
def ui_chime(sr=DEFAULT_SR, kind="discovery", pitch=1.0, seed=0):
    """Port of ``channel_discovery`` / ``channel_achievement`` / ``urgency_alert`` /
    ``channel_spell`` (procedural/interface.py; Klang Open License 1.0, (c) 2025 Chris Nash):
    Risset's additive bell at 523.25 Hz, 0.9 s; Farnell telephone bells 650 and 653 Hz struck
    60 ms apart; Faust englishBell x0.75 (t60 0.7 s, slope 3) hit by strikeModel(10, 3500, 0.25);
    STK BandedWG glass harmonica bowed at 880 Hz with a rise-hold-lift fingertip curve."""
    rng = np.random.default_rng(int(seed))
    k = float(np.clip(pitch, 0.25, 4.0))
    if kind == "discovery":
        y = M.additive_bell("risset_bell", 523.25 * k, 0.9, rng)
    elif kind == "achievement":
        y = np.zeros(S(0.6))
        P.place(y, M.farnell_bell(650.0 * k, 0.35, rng), 0.0)
        P.place(y, M.farnell_bell(653.0 * k, 0.35, rng, strength=0.8), 0.060)
    elif kind == "alert":
        y = M.render_modes(M.faust_strike(S(0.62), rng, cutoff=3500.0, sharpness=0.25),
                           M.resolve("englishBell", 0.75 * k, 0.0, t60=0.7, t60_slope=3.0))
    elif kind == "spell":
        from .physical.waveguides import banded
        t = np.arange(S(0.72)) / P.SR
        bow = 0.3 * (1.0 - (1.0 - np.clip(t / 0.15, 0, 1)) ** 2) * np.clip(1.0 - (t - 0.45) / 0.12, 0, 1) ** 2
        y = banded("glass_harmonica", 880.0 * k, 0.72, None, mode="bow", bow_env=bow, bow_pressure=0.5)
    else:
        raise ValueError(f"ui_chime kind {kind!r}: discovery|achievement|alert|spell")
    return _fin(y, sr, src=P.SR, peak=0.5, fade=0.004)


# ---------------------------------------------------------------------- impacts.py: damage
def _line(n):
    return 1.0 - np.arange(n) / max(n, 1)


def _shockwave(T=0.010):
    """Farnell ``pd shockwave``: 5 cycles sweeping 10/T Hz -> 0, hip 1."""
    v = _line(S(T))
    return _hip(np.concatenate([np.cos(2 * np.pi * 5.0 * v * v), np.zeros(S(0.02))]), 1.0)


def _nwave(trans_ms=12.0, height_ms=9.133):
    """Dilated N-wave with its ground reflection (explosive1.pd)."""
    a, b = S(0.001), S(trans_ms * 1e-3)
    shape = np.concatenate([np.linspace(0, 1, a, endpoint=False), np.linspace(1, -1, b, endpoint=False), np.linspace(-1, 0, a)])
    d = S(height_ms * 1e-3)
    n = len(shape) + d + S(0.03)
    refl = np.zeros(n)
    refl[d:d + len(shape)] = shape
    return _lop(P.fit(shape, n), 2000.0) + 1.5 * _lop(refl, 400.0)


def _onebeat(f, cycles, vol2, vol):
    """lovelyheart ``onebeat``: a phase-locked burst of ``cycles`` cycles at f with a squared fall."""
    n = S(cycles / f)
    r = np.arange(n) / n
    return (0.5 * np.cos(2 * np.pi * (r * cycles - 0.25)) + 0.5 * vol2 * np.cos(2 * np.pi * (r * 2 * cycles - 0.25))) * (1 - r) ** 2 * vol


def _murmur(gap, rng, vol=0.5):
    n = S(gap)
    a = max(int(0.3 * n), 1)
    env = np.concatenate([np.linspace(0, 1, a, endpoint=False), np.linspace(1, 0, n - a)])
    b = P.bandpass(rng.uniform(-1, 1, n), 50.0, 5.0)
    return _lop(b * b * env ** 2 * vol * 250.0, 500.0)


@numba.njit(cache=True)
def _comb3(x, d0, d1, d2, gain):
    """Three linearly interpolated delay taps (per-sample delays in samples) summed."""
    n = x.shape[0]
    y = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for d in (d0[i], d1[i], d2[i]):
            p = i - d
            if p >= 1.0:
                j = int(p)
                fr = p - j
                acc += (1.0 - fr) * x[j] + (fr * x[j + 1] if j + 1 < n else 0.0)
        y[i] = acc * gain[i]
    return y


def _damage(kind, rng):
    if kind == "physical":             # shockwave + barrel whump + v^8 burst + N-wave -> bpar8 -> tanh
        x = np.zeros(S(0.45))
        jit = 1.0 + rng.uniform(-0.03, 0.03)
        P.place(x, _shockwave(0.010 * jit), 0.0, 1.0)
        v = _line(S(0.040 * jit))
        P.place(x, np.cos(2 * np.pi * ((2 * v) ** 2 - 0.25)) * v, 0.0015, 0.7)       # pd barrel
        nb = S(0.0077)                                                               # pd noiseburst
        fm = _hip(P.bandpass(rng.uniform(-1, 1, nb), 3000.0, 2.0), 40.0)
        ph = np.cumsum(np.maximum(100.0 + 5000.0 * fm, 0.0)) / P.SR
        P.place(x, (4.0 * np.abs((ph % 1.0) - 0.5) - 1.0) * _line(nb) ** 8, 0.0, 0.5 * rng.uniform(0.8, 1.0))
        P.place(x, _nwave(12.0), 0.0, 0.6)
        centres = sorted(list(rng.uniform(200, 800, 4)) + list(rng.uniform(2200, 2800, 4)))
        body = sum(P.bandpass(x * 0.15, c * jit, 10.62) for c in centres)             # bpar8~
        return np.tanh(0.6 * _unit(x * 0.3 + body * 6.0) * 3.0) / math.tanh(1.8)
    if kind == "soul":                 # heartbeat + murmur under a fireball rumble and black noise
        dur = 1.7
        n = S(dur)
        period = 1.0 + rng.uniform(-0.03, 0.03)
        t2 = 0.35 * period
        heart = np.zeros(n)
        P.place(heart, _onebeat(63.0, 5, 0.0, 0.65), 0.0)
        P.place(heart, _onebeat(42.0, 3, 0.5, 1.0), t2)
        mur = np.zeros(n)
        P.place(mur, _murmur(t2, rng), 0.0)
        P.place(mur, _murmur(period - t2, rng), t2)
        heart = np.clip(_hip(heart + 0.25 * _unit(mur), 5.0), -1, 1)
        t = np.arange(n) / P.SR                                                       # pd fireball
        L = np.where(t < 0.1, t / 0.1, np.clip(1 - (t - 0.1) / 1.5, 0, 1))
        fc = 40.0 + 100.0 * L
        a = np.exp(-2 * np.pi * fc / P.SR)
        rum = _hip(P.tv_one_pole_lp(P.tv_one_pole_lp(P.tv_bandpass(rng.uniform(-1, 1, n), fc, np.ones(n)), a), a) * 10.0, 5.0) * _lop(L ** 4, 5.0)
        blk = _lop(_lop(rng.uniform(-1, 1, n), 80.0), 80.0)                           # black noise
        blk = _lop(_lop(np.clip(_hip(blk, 15.0) * 12.0, -1, 1), 80.0), 80.0)
        blk *= np.clip(np.arange(n) / S(0.1), 0, 1) * P.pow_decay(n, 2.0)
        return _unit(heart) + 0.45 * _unit(rum) + 0.3 * _unit(blk)
    if kind == "mental":               # boomer through the swept comb over a 6 Hz-spaced doom pipe
        dur = 1.6
        n = S(dur)
        f = 10.0 + 200.0 * (1.0 - 0.05)
        boom = _hip(_lop(_lop(P.bandpass(rng.uniform(-1, 1, n), f, 1.0), f), f) * 40.0, 1.0) * (0.35 + 0.5 * 0.05)
        na, nd = S(0.1), S(0.6 + 12.0 * 0.05)
        env = np.zeros(n)
        env[:na] = 1 - (1 - np.arange(na) / na) ** 2
        env[na:na + nd] = P.pow_decay(nd, 4.0)[:n - na]
        boom = boom * env
        v = np.clip(1.0 - np.arange(n) / n, 0.0, 1.0)                                 # combsweep
        s = 4.0 * (1.0 - v) + 1e-4
        ms = P.SR * 1e-3
        comb = _comb3(np.ascontiguousarray(boom), (10 + s) * ms, (10 + s + 13.7 * s) * ms, (10 + s + 27.4 * s) * ms, v * 0.33)
        ex = _lop(_lop(_hip(_hip(rng.uniform(-1, 1, n), 60.0), 60.0), 400.0), 400.0) * 100.0   # doom pipe
        ex = ex * ex * 0.01
        f0 = 90.0 * (1.0 + rng.uniform(-0.02, 0.02))
        pipe = sum(P.bandpass(ex, f0 + k * 6.0, f0 / 3.75) for k in range(3))
        pipe = np.clip(_unit(pipe) * 1.5, -0.9, 0.9)
        env2 = np.zeros(n)
        env2[:na] = 1 - (1 - np.arange(na) / na) ** 2
        env2[na:] = P.pow_decay(n - na, 3.0)
        return P.lowpass(_unit(boom) + 0.7 * _unit(comb) + 0.45 * pipe * env2 + 0.35 * _unit(P.fit(_shockwave(0.030), n)), 800.0)
    raise ValueError(f"damage kind {kind!r}: physical|soul|mental")


@sfx("damage", "Damage landing on a character: a physical blow, a `soul` hit (one heartbeat under a rumble) or a `mental` hit (pressure without pitch). Felt and low, never shrill.",
     kind=("physical", "physical | soul | mental"), seed=(0, "take: body resonances, noise, +-3 % force"))
def damage(sr=DEFAULT_SR, kind="physical", seed=0):
    """Port of ``impact_physical`` / ``impact_soul`` / ``impact_mental`` (procedural/impacts.py;
    Klang Open License 1.0, (c) 2025 Chris Nash), themselves ports of Farnell's patches:
    physical = shockwave chirp + barrel whump + v^8 noise burst + dilated N-wave with ground
    reflection through one bpar8 body bank (Q 10.62), tanh(0.6 x); soul = lovelyheart ``onebeat``
    63 Hz x5 and 42 Hz x3 with ``murmur``, under ``fireball`` rumble and black noise; mental =
    ``boomer`` (200 Hz) through ``combsweep`` (w 13.7) over three band-passes 6 Hz apart at 90 Hz
    (Farnell's doom tone), low-passed at 800 Hz."""
    return _fin(_damage(str(kind), np.random.default_rng(int(seed))), sr, src=P.SR, peak=0.8)


# ---------------------------------------------------------------------- body.py
@sfx("sniff", "Sniffing: short nasal inhale pulls (velum open, no voice).",
     pulls=(2, "number of pulls"), size=(1.0, "body scale: bigger = longer tract, lower"),
     pull=(0.09, "s per pull"), gap=(0.13, "s between pulls"), seed=(0, "seed"))
def sniff(sr=DEFAULT_SR, pulls=2, size=1.0, pull=0.09, gap=0.13, seed=0):
    """Port of ``body.sniff`` (procedural/body.py; Klang Open License 1.0, (c) 2025 Chris Nash):
    Pink Trombone with the velum open (nasal 1), glottis unvoiced, turbulence at a constriction
    at index 22 whose diameter narrows 0.7 -> 0.5 during each pull; Farnell articulation curve."""
    rng = np.random.default_rng(int(seed))
    k = int(np.clip(pulls, 1, 12))
    pull, gap = float(np.clip(pull, 0.04, 1.0)), float(np.clip(gap, 0.02, 1.0))
    dur = k * (pull + gap) + 0.1
    n = S(dur)
    act, env = np.zeros(n), np.zeros(n)
    for j in range(k):
        a, m = S(j * (pull + gap) * rng.uniform(0.92, 1.08)), S(pull)
        if a + m > n:
            break
        act[a:a + m] = 1.0
        env[a:a + m] = V.articulation_env(m, 0.3, 0.0) * (1.0 - 0.15 * j)
    y = V.pink_trombone(90.0, 0.1, 26.0, 3.2, [{"index": 22.0, "diameter": 0.7 - 0.2 * env, "active": act}],
                        tick_rate_scale=float(np.clip(1.0 / float(size), 0.25, 1.5)), dur=dur, rng=rng,
                        voiced_env=0.0, nasal_env=1.0)
    return _fin(y[:n] * np.minimum(env * 1.3, 1.0), sr, src=P.SR, peak=0.3)


@sfx("strain", "Straining effort: a pressed, held voiced push with throat friction (lifting, bracing, holding a door).",
     dur=(0.6, "s"), freq=(110.0, "voice pitch Hz (falls 6 % over the push)"), tenseness=(0.8, "0.5 lax .. 0.95 pressed"),
     size=(1.0, "body scale: bigger = lower formants"), seed=(0, "seed"))
def strain(sr=DEFAULT_SR, dur=0.6, freq=110.0, tenseness=0.8, size=1.0, seed=0):
    """Port of the synthesis branch of ``body.strain`` (procedural/body.py; Klang Open License 1.0,
    (c) 2025 Chris Nash): Pink Trombone, tenseness 0.8, constriction at index 10 with diameter
    0.5, tongue at the neutral rest shape (12.9, 2.43) so no vowel is named. The hybrid branch
    (LPC tract of a recorded strain) needs the recordings bank and is not ported."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.15, 4.0))
    n = S(dur)
    env = V.articulation_env(n, 0.35, 0.0)
    y = V.pink_trombone(float(freq) * np.linspace(1.0, 0.94, n), float(np.clip(tenseness, 0.1, 0.98)), 12.9, 2.43,
                        [{"index": 10.0, "diameter": 0.5, "active": 1.0}],
                        tick_rate_scale=float(np.clip(1.0 / float(size), 0.25, 1.5)), dur=dur, rng=rng,
                        voiced_env=(env > 0.05).astype(float), wobble=0.3)
    return _fin(y[:n] * env, sr, src=P.SR, peak=0.5)


@sfx("weight_shift", "Planted feet shifting under load: a scuff / shuffle on any granular ground (no footfall).",
     ground=("sand", "ground name, as for `footsteps` (sand, gravel, snow, leaves, ...)"),
     dur=(0.55, "s"), weight=(1.0, "load on the feet"), seed=(0, "seed"))
def weight_shift(sr=DEFAULT_SR, ground="sand", dur=0.55, weight=1.0, seed=0):
    """Port of ``body.weight_shift`` (procedural/body.py; Klang Open License 1.0, (c) 2025 Chris
    Nash): the Farnell ground-reaction-force polynomial with the heel : roll : ball weights
    3 : 1.5 : 1 drives the Visell granular ground texture (grains fire while the force rises)."""
    if ground not in PT.GROUNDS:
        raise ValueError(f"weight_shift ground {ground!r}: {'|'.join(PT.GROUNDS)}")
    n = S(float(np.clip(dur, 0.1, 4.0)))
    x = np.clip(np.arange(n) / n, 0, 0.75) / 0.75

    def p(u, N):
        return 1.5 * (1 - u) * (N * u - N * u ** 3)
    f = (p(np.clip(x, 0, 1 / 3) * 3, 10.0) + p((np.clip(x, 0.125, 0.875) - 0.125) / 0.75, 5.0)
         + p((np.clip(x, 2 / 3, 1) - 2 / 3) * 3, 3.3333)) * float(np.clip(weight, 0.2, 3.0))
    y = PT.granular_footstep_texture(f, str(ground), np.random.default_rng(int(seed)), tail=0.05)[:n]
    return _fin(y, sr, src=P.SR, peak=0.4)


# ---------------------------------------------------------------------- world.py: item handling
def _stroke(n, strokes):
    """Hand-stroke speed curve: half-cosine pushes (start s, length s, peak), Farnell's momentum push."""
    v = np.zeros(n)
    for at, ln, a in strokes:
        m = S(ln)
        P.place(v, np.sin(np.pi * np.arange(m) / m) * a, at)
    return v


def _creaks(force, rng, k=3, size=1.0):
    """Leather / rope / strap creak: k stick-slip sources at once through the door formants."""
    from .physical import ambience as AMB
    return sum(AMB.creak(np.clip(force * rng.uniform(0.85, 1.05), 0.0, 1.0), rng, material="pine", size=size) for _ in range(k))


def _rub(dur, e, rng, preset="Sandpaper", res=1.0):
    n = S(dur)
    c = np.zeros(n)
    c[:min(n, len(e))] = e[:n]
    return PT.shaker(preset, dur, rng, energy_curve=np.clip(c, 0.0, 1.0), resonance_scale=res, energy_mode="level")[:n]


def _cloth(dur, e, rng, level=1.0):
    """Cloth moved by a body: Sandpaper brush kept below 3 kHz plus a weak stick-slip band."""
    from .physical import ambience as AMB
    n = S(dur)
    c = np.zeros(n)
    c[:min(n, len(e))] = e[:n]
    fric = V._bp(AMB.stickslip_pulses(np.clip(0.2 + 0.7 * c, 0, 1), rng), 2500.0, 1.0)
    return (_unit(P.lowpass(_rub(dur, 0.9 * c, rng, "Sandpaper", 0.4), 3000.0)) + 0.25 * _unit(fric)) * level


def _handle(kind, rng):
    """world.item_pickup / item_equip: (buffer seconds, [(signal, at, gain), ...])."""
    from . import foley as FO
    from .physical import ambience as AMB
    j = lambda a: 1.0 + rng.uniform(-a, a)                                    # noqa: E731
    kn = lambda t, **kw: _unit(FO._knock(t, rng, **kw))                       # noqa: E731
    lift = lambda d, at=0.0, lv=0.25: _cloth(d, _stroke(S(d), [(at, 0.25 * j(0.1), 0.8)]), rng, lv)   # noqa: E731

    def drag(table, stroke, force, lam, q, mix_body, mix_dry, **kw):          # Cook scrape driving the object
        s = FO._scrape(_stroke(S(stroke[1] + 0.05), [stroke]), rng, lam, q) * math.sqrt(force)
        return mix_body * _unit(FO._drive(table, s, **kw)) + mix_dry * _unit(s)

    def creak(ln, peak, k, size):
        return _unit(_creaks(np.clip(_stroke(S(ln + 0.1), [(0.0, ln, peak)]), 0, 1), rng, k, size))

    def catch():                                                              # reload cluster 1: small catch
        return _unit(FO._mclick(AMB.hip(rng.uniform(-1, 1, S(0.08)), 300.0), FO.CATCH, 0.0, 0.33))
    L = []
    if kind == "weapon":       # blade tip drawn over stone, the grip closes, the hilt knocks the steel
        L += [(drag("sy_sword1", (0.0, 0.3, 0.6 * j(0.1)), 1.0, 6e-5, 3.0, 0.6, 0.3, d_scale=2.0), 0.02, 1.0),
              (kn("sy_sword1", d_scale=2.0, contact_ms=0.4, vel=0.5, micro=1), 0.30, 0.7), (creak(0.35, 0.75, 2, 0.35), 0.28, 0.35), (lift(0.4), 0.0, 1.0)]
        return 1.2, L
    if kind == "armor":        # plates knock together, a buckle catches, the straps creak
        t = 0.03
        for g in (1.0, 0.6, 0.35):
            L.append((kn("sy_computertower", f_scale=1.6 * j(0.02), d_scale=2.0, contact_ms=0.3, vel=g, micro=2), t, 0.8 * g))
            t += rng.uniform(0.06, 0.16)
        return 1.2, L + [(catch(), t + 0.05, 0.4), (creak(0.6, 0.85, 3, 0.4), 0.1, 0.4), (lift(0.6, lv=0.4), 0.0, 1.0)]
    if kind == "tool":         # wooden haft slides over stone, then the iron head knocks
        return 1.2, [(drag("sy_stick", (0.0, 0.25, 0.5), 1.0, 2e-4, 1.5, 0.5, 0.2, f_scale=0.3), 0.0, 1.0),
                     (kn("sy_wok", f_scale=0.6, d_scale=3.0, contact_ms=0.5, vel=0.7, micro=1), 0.24, 0.8),
                     (kn("sy_stick", f_scale=0.3, contact_ms=1.5, vel=0.7), 0.245, 0.4), (lift(0.4), 0.05, 1.0)]
    if kind == "food":         # a wrapped ration settles in the hand, crisp contents shift
        e = np.zeros(S(0.35))
        e[:S(0.05)] = 0.6
        return 1.2, [(kn("body_roundSmall", contact_ms=18.0, felt=True, dur=0.35), 0.02, 0.6),
                     (_unit(_rub(0.35, e, rng, "Crunch")), 0.04, 0.45), (lift(0.5, lv=0.6), 0.0, 1.0)]
    if kind == "glass":        # a small glass object: fingertip contact, then a light knock on stone
        return 1.2, [(kn("sy_calona0", f_scale=1.4 * j(0.02), contact_ms=0.25, vel=0.4), 0.12, 0.8),
                     (kn("sy_calona0", f_scale=1.4, contact_ms=6.0, vel=0.3, felt=True), 0.02, 0.35), (lift(0.4, lv=0.3), 0.0, 1.0)]
    if kind == "trinket":      # small metal lifted from a pile: tinkles on the bounce schedule
        t, fs = 0.0, 0.8 * j(0.02)
        for g, gap in ((0.416, 0.0), (0.257, rng.uniform(0.170, 0.344)), (0.05, rng.uniform(0.075, 0.149))):
            t += gap
            L.append((AMB.tinkle(rng, f_scale=fs, gain=g), t, 1.0))
        return 1.2, L + [(catch(), 0.02, 0.2), (lift(0.4, lv=0.2), 0.0, 1.0)]
    if kind == "bell":         # heavy bell metal: a short grind off the stone, ringing under the palm
        return 1.2, [(drag("sy_bell4", (0.0, 0.25, 0.4), 1.5, 1.5e-4, 2.0, 0.4, 0.15, d_scale=3.0), 0.0, 1.0),
                     (kn("sy_bell4", d_scale=3.0, contact_ms=4.0, vel=0.8, felt=True, dur=1.0), 0.2, 1.0), (lift(0.5, lv=0.3), 0.0, 1.0)]
    if kind == "draw":         # blade drawn from a leather scabbard, grip settles, the guard ticks
        return 1.5, [(drag("sy_sword1", (0.0, 0.5, 1.2 * j(0.1)), 0.8, 2.5e-4, 1.2, 0.55, 0.45, d_scale=1.5), 0.0, 1.0),
                     (kn("sy_sword1", d_scale=1.5, contact_ms=0.3, vel=0.6, micro=2), 0.5, 0.6), (creak(0.3, 0.8, 2, 0.35), 0.55, 0.3)]
    if kind == "shield":       # a wooden shield taken on the arm: forearm knock, strap creak, grip catch
        return 1.5, [(kn("body_squareBig", contact_ms=8.0, vel=0.8, felt=True), 0.05, 0.9), (creak(0.5, 0.9, 3, 0.4), 0.12, 0.45),
                     (_unit(AMB.clunk(rng, f_scale=j(0.02))), 0.5, 0.6), (lift(0.5, lv=0.3), 0.0, 1.0)]
    if kind == "don":          # armour put on: a long shrug of cloth, plates settling, two buckles
        L.append((_cloth(1.4, _stroke(S(1.4), [(0.0, 0.6, 0.9), (0.45, 0.5, 0.7)]), rng, 0.6), 0.0, 1.0))
        t, gap, g = 0.35, 0.22, 1.0
        for _ in range(3):     # Farnell bouncing: the intervals shrink and the hits fall
            L.append((kn("sy_computertower", f_scale=1.4 * j(0.02), d_scale=2.5, contact_ms=0.5, vel=g, micro=1), t, 0.7 * g))
            t, gap, g = t + gap, gap * 0.6, g * 0.6
        return 1.5, L + [(catch(), 1.05, 0.35), (catch(), 1.2 + rng.uniform(0, 0.05), 0.3)]
    if kind == "amulet":       # a pendant on a fine chain over the head, settling on cloth
        return 1.5, [(_unit(AMB.chain_shift(0.9, rng, pushes=1, f_scale=1.1 * j(0.02), contacts_per_s=60.0, clunk_p=0.0)), 0.0, 0.7),
                     (kn("sy_desklamp", f_scale=2.0, d_scale=2.0, contact_ms=3.0, vel=0.5, felt=True), 0.75, 0.4),
                     (_cloth(0.6, _stroke(S(0.6), [(0.0, 0.4, 0.5)]), rng, 0.2), 0.6, 1.0)]
    raise ValueError(f"handle kind {kind!r}: {'|'.join(HANDLE_KINDS)}")


HANDLE_KINDS = ("weapon", "armor", "tool", "food", "glass", "trinket", "bell", "draw", "shield", "don", "amulet")


@sfx("handle", "Picking up or equipping an object: the hand's sleeve, the object leaving the ground or its sheath, and its own body sounding once.",
     kind=("weapon", "pick up: weapon | armor | tool | food | glass | trinket | bell;  equip: draw (blade from a scabbard) | shield | don (armour) | amulet"),
     seed=(0, "take"))
def handle(sr=DEFAULT_SR, kind="weapon", seed=0):
    """Port of ``item_pickup`` and ``item_equip`` (procedural/world.py; Klang Open License 1.0,
    (c) 2025 Chris Nash): van den Doel measured bodies knocked with a (1 - cos) or Hertz felt
    contact, Cook's FoleyAutomatic scrape (fractal surface through a reson at v / lambda) driving
    the same bodies, Farnell stick-slip creaks (leather: several sources), click-factory catch,
    shell-casing tinkles, chain momentum, PhISEM Sandpaper/Crunch. Knock, drive and scrape are
    shared with ``genny.foley``. Not carried over: the per-table gain calibration (every layer is
    peak-normalised before mixing anyway)."""
    rng = np.random.default_rng(int(seed))
    dur, layers = _handle(str(kind), rng)
    y = np.zeros(S(dur))
    for x, at, g in layers:
        P.place(y, x, at, g)
    return _fin(P.highpass(y, 30.0), sr, src=P.SR, peak=0.7, fade=0.01)


@sfx("tide", "The tide turning: water starting to run through channels (rising) or dragging shingle back down the slope (falling).",
     kind=("rising", "rising | falling"), dur=(1.95, "s"), seed=(0, "seed"))
def tide(sr=DEFAULT_SR, kind="rising", dur=1.95, seed=0):
    """Port of ``tide_rising`` / ``tide_falling`` (procedural/world.py; Klang Open License 1.0,
    (c) 2025 Chris Nash): Reshetnikov's granular stream with its dynamics ramped 0.05 -> 0.55
    (or 0.6 -> 0.03), Moss bubbles whose rate follows the flow squared, and for the backwash
    PhISEM BigRocks + LittleRocks driven by the flow."""
    from .physical import ambience as AMB
    if kind not in ("rising", "falling"):
        raise ValueError(f"tide kind {kind!r}: rising|falling")
    rng = np.random.default_rng(int(seed))
    d = float(np.clip(dur, 0.5, 20.0))
    n = S(d)
    u = np.arange(n) / n
    if kind == "rising":
        dyn = 0.05 + 0.5 * np.clip(u / 0.82, 0, 1) ** 1.5
        e = dyn / 0.55
        b = PT.bubble_field(d, rng, rate=20.0 + 900.0 * e ** 2, alpha=2.0, r_range=(1e-3, 8e-3))
        w = AMB.stream(d, rng, d=dyn, instances=260)
        y = _unit(w.mean(axis=1) if w.ndim == 2 else w)[:n] + 0.4 * _unit(b)[:n]
    else:
        dyn = 0.6 - 0.57 * np.clip((u - 0.05) / 0.87, 0, 1) ** 0.6
        e = dyn / 0.6
        w = AMB.stream(d, rng, d=dyn, instances=260)
        peb = _rub(d, 0.9 * e, rng, "BigRocks") * 0.6 + _rub(d, 0.7 * e, rng, "LittleRocks") * 0.4
        b = PT.bubble_field(d, rng, rate=30.0 + 700.0 * e ** 2, alpha=2.0, r_range=(1e-3, 8e-3))
        y = _unit(P.lowpass((w.mean(axis=1) if w.ndim == 2 else w)[:n], 3500.0)) + 0.5 * _unit(peb) + 0.3 * _unit(b)[:n]
    return _fin(y, sr, src=P.SR, peak=0.6, fade=0.1)


@sfx("distant_bell", "A tower bell heard from far away: one toll carried through the air and returned by the buildings.",
     distance=(400.0, "metres: 1/r spreading and air absorption (the highs go first)"),
     size=(1.25, "bell size: bigger = lower (1 = the 301 mm model bell at 452 Hz)"),
     dur=(2.0, "s"), seed=(0, "strike position and noise"))
def distant_bell(sr=DEFAULT_SR, distance=400.0, size=1.25, dur=2.0, seed=0):
    """Port of ``week_turn`` (procedural/world.py; Klang Open License 1.0, (c) 2025 Chris Nash):
    Faust churchBell (mesh2faust FEM modes) struck with strikeModel noise, ``space.distance``
    (JOS air absorption + 1/r) and Farnell's outdoor building echoes (165, 121, 33 ms)."""
    from .physical import ambience as AMB
    from .physical import space as SP
    rng = np.random.default_rng(int(seed))
    d = float(np.clip(dur, 0.5, 12.0))
    x = M.strike("churchBell", 1.0 / float(np.clip(size, 0.25, 4.0)) * (1.0 + rng.uniform(-0.01, 0.01)), 1.0,
                 rng.uniform(0.0, 0.3), rng, exciter="faust", dur=d + 0.2)
    x = AMB.outdoor_echoes(SP.distance(x, float(np.clip(distance, 1.0, 5000.0)), ref=1.0))
    return _fin((x.mean(axis=1) if x.ndim == 2 else x)[:S(d)], sr, src=P.SR, peak=0.5, fade=0.3)


# ====================================================================== instruments
# The sampler map (procedural/instrument_map.py) lists the orchestral voices of the CC0 libraries.
# Those genny had no instrument for are added here on genny's own pitch-exact voicing helpers.
# UNSOURCED (all formant tables below): typical formant centres of each instrument (Meyer,
# "Acoustics and the Performance of Music"); none is in the research findings.
VIOLA_BODY = [(230, 50, 1.2), (370, 80, 1.5), (900, 300, 0.8), (2000, 450, 0.9)]


@instrument("viola", "Viola, bowed: darker and more nasal than a violin, a fifth lower (inner voices, melancholy solos).",
            family="bowed", span=("C3", "E6"), attack=(0.08, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def viola(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.08, vibrato=1.0):
    y = _blown(freq, dur, sr, vel, dict(tilt=1.7, formants=VIOLA_BODY, corner=1800.0, slope=3.0),
               dict(tilt=1.15, formants=VIOLA_BODY, corner=2600.0, slope=3.0),
               attack=attack, release=0.1, bloom=attack, vib=(5.5, 0.17 * vibrato, 0.14))
    n = y.shape[0]
    return (y + F.bandpass(O.white(n, seed=57), float(np.clip(freq * 4, 900, 2400)), sr, 1.2) * 0.03 * np.abs(y)) * 0.435


@instrument("bassoon", "Bassoon: dry, woody double reed; comic in the bass, plaintive in the tenor.",
            family="wind", span=("Bb1", "E5"))
def bassoon(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(480, 140, 2.4), (1200, 300, 0.9)]
    return _blown(freq, dur, sr, vel, dict(tilt=1.2, formants=body, corner=1600.0, slope=3.5),
                  dict(tilt=0.8, formants=body, corner=2300.0, slope=3.5),
                  attack=0.035, release=0.09, bloom=0.05, vib=(5.2, 0.07, 0.3)) * 0.43


@instrument("cor_anglais", "Cor anglais (English horn): a darker, rounder oboe a fifth lower (pastoral, mournful solos).",
            family="wind", span=("E3", "A5"))
def cor_anglais(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(950, 220, 2.4), (2300, 400, 0.6)]
    return _blown(freq, dur, sr, vel, dict(tilt=1.2, formants=body, corner=1900.0, slope=3.5),
                  dict(tilt=0.8, formants=body, corner=2600.0, slope=3.5),
                  attack=0.035, release=0.09, bloom=0.05, vib=(5.3, 0.1, 0.25)) * 0.42


@instrument("bass_clarinet", "Bass clarinet: hollow, dark odd-harmonic reed an octave below the clarinet.",
            family="wind", span=("D2", "F5"))
def bass_clarinet(freq, dur, sr=DEFAULT_SR, vel=1.0):
    even = _kt(freq, 0.06, 0.45, f_lo=130.0, f_hi=1000.0)
    return _blown(freq, dur, sr, vel, dict(tilt=1.6, even=even, corner=1100.0, slope=3.0),
                  dict(tilt=1.15, even=even, formants=[(1100, 400, 0.5)], corner=1800.0, slope=3.0),
                  attack=0.045, release=0.09, bloom=0.06, vib=(5.0, 0.03, 0.3)) * 0.44


def _fipple(freq, dur, sr, vel, breath, chiff, corner, seed, gain):
    y = _blown(freq, dur, sr, vel, dict(tilt=3.0, corner=corner, slope=3.0), dict(tilt=2.2, corner=corner * 1.3, slope=3.0),
               attack=0.03, release=0.07, bloom=0.03, vib=(5.0, 0.05, 0.3))
    n = y.shape[0]
    air = F.bandpass(O.white(n, seed=seed), float(np.clip(freq * 2, 900, 2600)), sr, 2.0)
    t = np.arange(n) / sr
    return (y + air * (float(breath) * 0.25 * np.abs(y) + float(chiff) * 0.2 * np.exp(-t / 0.02) * (0.5 + 0.5 * vel))) * gain


@instrument("piccolo", "Piccolo: the flute an octave up, jet-driven like it (marches, sparkle on top of a tutti; keep it soft).",
            family="wind", span=("D5", "C8"), breath=(0.15, "turbulence on the jet: 0 = none .. 0.4"), vibrato=(0.0, "0 = steady .. 1"))
def piccolo(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.15, vibrato=0.0):
    from .acoustic import _jet_voice
    return _jet_voice(freq, dur, sr, vel, breath, vibrato, top=6500.0, gain=1.0)


@instrument("recorder", "Recorder: a fipple flute, jet-driven, with the tongued chiff of each note as a short overshoot of the breath (early music, folk, school).",
            family="wind", span=("C4", "D7"), breath=(0.2, "turbulence on the jet: 0 = none .. 0.4"), chiff=(1.0, "tongued attack 0..2"),
            vibrato=(0.0, "0 = steady .. 1"))
def recorder(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.2, chiff=1.0, vibrato=0.0):
    """The STK jet model, not STK's Recorder (the Verge, Hirschberg and Causse model): that one is in
    genny.physical.waveguides.recorder(model="verge") and does not hold a written pitch. It blows every note at one
    pressure, and a scan of 40 pressures per note found at most a few narrow windows where the pipe speaks within
    60 cents of the note (one for C4, one for G4, none for C7), each at a different pressure."""
    from .acoustic import _jet_voice
    return _jet_voice(freq, dur, sr, vel, breath, vibrato, top=3600.0, gain=1.0, chiff=chiff)


@instrument("hand_chime", "Hand chime: a clapper-struck tuned tube, strong fundamental with its twelfth; softer and rounder than a handbell.",
            family="mallet", span=("G3", "C7"), decay=(1.2, "ring scale (seconds)"))
def hand_chime(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=1.2):
    """Port of ``charter.hand_chime`` (procedural/charter.py; Klang Open License 1.0, (c) 2025 Chris
    Nash): modes 1, 3 (the twelfth) and the untuned free-bar 5.404 at gains 1, 0.25, 0.05 with
    T60 2.2, 0.9, 0.35 x ``decay``, plus a 2 ms clapper click. UNSOURCED in the source too."""
    ring = max(dur, 1.6 * float(decay))
    n = samples(ring, sr)
    tau = float(decay) / 6.91                     # T60 -> time constant
    x = _modes(freq, n, sr, [(1.0, 1.0, 2.2 * tau), (3.0, 0.25 * (0.5 + 0.5 * vel), 0.9 * tau), (5.404, 0.05 * vel, 0.35 * tau)], corner=3200.0)
    click = F.bandpass(O.white(n, seed=60), float(np.clip(freq * 4, 1500, 3200)), sr, 2.0) * np.exp(-np.arange(n) / (0.002 * sr)) * 0.1 * vel
    env = np.minimum(1.0, np.arange(n) / (0.0015 * sr))
    k = samples(0.25, sr)
    off = samples(dur, sr)                        # hand damping after the note is released
    if off + k < n:
        env[off:] *= np.exp(-np.arange(n - off) / (0.08 * sr))
    return (x + click) * env * 0.49 * (0.6 + 0.4 * vel)


# ====================================================================== layers
ARTICULATIONS = ("sustain", "staccato", "pizz", "tremolo", "harmonic", "legato", "hit", "roll")


def _steps(layer):
    steps = layer.get("steps", layer.get("notes", "C4 E4 G4"))
    return parse_sequence(steps, float(layer.get("step", 0.25))) if isinstance(steps, str) else list(steps)


@layer_type("articulate")
def articulate_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    """A melodic line played with one articulation on any instrument:
    ``{"type": "articulate", "inst": "violin", "artic": "tremolo", "steps": "G4:1 A4:1"}``.

    The eight articulations are the sampler's (procedural/instrument_map.py ``ARTICULATIONS``),
    realised by how the notes are played instead of by a recording:
    sustain (as written), legato (each note overlaps the next by 8 %), staccato (35 % of the
    length, at least 60 ms), hit (60 ms: just the attack and the instrument's own ring), pizz
    (the `pizzicato` instrument at the same pitches), harmonic (an octave up, vel x 0.6),
    tremolo (re-struck ``rate`` times a second, alternating strong / weak), roll (the same with
    a crescendo over each note). ``rate``: strokes per second for tremolo / roll (default 12)."""
    art = str(layer.get("artic", "sustain"))
    if art not in ARTICULATIONS:
        raise ValueError(f"articulate: artic {art!r}; one of {', '.join(ARTICULATIONS)}")
    inst = "pizzicato" if art == "pizz" else layer.get("inst", "violin")
    params = dict(layer.get("params") or {})
    vel0 = float(layer.get("vel", 1.0))
    rate = float(np.clip(layer.get("rate", 12.0), 1.0, 40.0))
    k = 2.0 ** (float(layer.get("transpose", 0)) / 12.0) * (2.0 if art == "harmonic" else 1.0)
    parts, t = [], 0.0
    for st in _steps(layer):
        d = float(st["dur"]) * q
        v = vel0 * float(st.get("vel", 1.0)) * (0.6 if art == "harmonic" else 1.0)
        fr = [f * k for f in st["pitches"]]
        if fr and art in ("tremolo", "roll"):
            m = max(1, int(round(d * rate)))
            for j in range(m):
                g = (0.4 + 0.6 * (j + 1) / m) if art == "roll" else (1.0 if j % 2 == 0 else 0.8)
                parts.append((I.render_chord(inst, fr, d / m, sr, min(1.0, v * g), **params), t + j * d / m))
        elif fr:
            length = {"legato": d * 1.08, "staccato": max(0.06, d * 0.35), "hit": 0.06}.get(art, d)
            parts.append((I.render_chord(inst, fr, length, sr, v, **params), t))
        t += d
    return mix(parts, sr) if parts else silence(t or 0.1, sr)


# ---------------------------------------------------------------------- sampler layer
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"(?<![A-Za-z0-9#])([A-Ga-g])(#|s|b)?(-?\d)(?![0-9A-Za-z])")
_DYN = {"pppp": 1, "ppp": 2, "pp": 3, "p": 4, "mp": 5, "mf": 6, "f": 7, "ff": 8, "fff": 9,
        "soft": 3, "quiet": 3, "med": 6, "medium": 6, "loud": 8}
_AUDIO = (".wav", ".flac", ".aif", ".aiff", ".ogg")
DYN_RANGE_DB = 28.0              # sampler.py: loudness span from vel 1 to vel 0


def name_midi(stem: str) -> int | None:
    """MIDI note (C4 = 60) named in a file stem: ``A#3``, ``As3``, ``Bb2``, ``c4`` (sampler._name_midi)."""
    m = _NOTE_RE.search(stem.replace(" ", "_"))
    if not m:
        return None
    return 12 * (int(m.group(3)) + 1) + _PC[m.group(1).upper()] + (1 if m.group(2) in ("#", "s") else -1 if m.group(2) == "b" else 0)


def vel_rank(stem: str) -> int:
    """Velocity layer token of a file stem: ``v3``, ``vl2``, ``pp`` .. ``fff``, ``soft`` / ``loud``;
    0 when there is none (sampler._vel_rank). A note name such as ``F2`` is not the dynamic f."""
    for t in re.split(r"[_\- .]", stem):
        m = re.fullmatch(r"v[l]?(\d+)", t)
        if m:
            return int(m.group(1))
        if re.fullmatch(r"[A-Ga-g](#|s|b)?-?\d", t):
            continue
        m = re.fullmatch(r"(pppp|ppp|pp|p|mp|mf|fff|ff|f)\d*", t) or re.fullmatch(r"(soft|quiet|medium|med|loud)\d*", t.lower())
        if m:
            return _DYN[m.group(1)]
    return 0


@lru_cache(maxsize=32)
def sampler_zones(folder: str, pattern: str = "*", root: int = 60) -> tuple:
    """Zones of a sample folder: ((rank, ((midi, (path, ...)), ...)), ...), quiet layer first.
    Files without a note name are takes of the note ``root``."""
    d = Path(folder)
    if not d.is_dir():
        raise ValueError(f"sampler: no such folder {folder!r}")
    layers: dict = {}
    for p in sorted(d.glob(pattern)):
        if p.is_file() and p.suffix.lower() in _AUDIO:
            m = name_midi(p.stem)
            layers.setdefault(vel_rank(p.stem), {}).setdefault(root if m is None else m, []).append(str(p))
    if not layers:
        raise ValueError(f"sampler: no audio files ({', '.join(_AUDIO)}) match {pattern!r} in {folder!r}")
    return tuple((r, tuple((m, tuple(v)) for m, v in sorted(layers[r].items()))) for r in sorted(layers))


@lru_cache(maxsize=256)
def _sample(path: str):
    import soundfile as sf
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    return (x[:, 0] if x.shape[1] == 1 else x[:, :2]), int(fs)


def _repitch(x, sr_in, sr_out, semis):
    """Resample to ``sr_out`` with a pitch shift of ``semis`` semitones (sampler._repitch)."""
    fr = Fraction((sr_out / sr_in) * 2.0 ** (-semis / 12.0)).limit_denominator(640)
    return x if fr == 1 else resample_poly(x, fr.numerator, fr.denominator, axis=0)


@layer_type("sampler")
def sampler_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    """Play a folder of recordings as an instrument:
    ``{"type": "sampler", "dir": "samples/kalimba", "steps": "C4 E4 G4:0.5"}``.

    The mapping comes from the file names, as in procedural/sampler.py (Klang Open License 1.0,
    (c) 2025 Chris Nash): a note name (``C4``, ``F#3``, ``As2``, ``Bb1``; C4 = MIDI 60) gives the
    pitch, a dynamic token (``v1``.. / ``pp``..``fff`` / ``soft`` ``loud``) the velocity layer, and
    several files with the same note and layer are round-robin takes used in turn. A note picks
    the layer from its velocity (even split), the nearest recorded pitch in it, and is repitched
    by resampling. Level follows the sampler's velocity curve (-28 dB (1 - vel)^1.15).
    Keys: ``dir``, ``pattern`` (glob, default ``*``), ``root`` (note of files that name none,
    default C4), ``steps`` / ``step`` / ``legato`` / ``gap`` / ``transpose`` / ``vel`` as in a
    ``seq`` layer, ``release`` (s, fade after the note length; 0 = let every sample ring out),
    ``tune`` (cents), ``max_shift`` (semitones; farther notes raise an error instead of
    sounding wrong). Returns mono, or stereo if any file is stereo."""
    # ponytail: a note longer than its recording just ends with the recording; port
    # sampler._extend (cross-correlated loop splicing) if sustained libraries need holding.
    from .notes import note_to_midi
    root = layer.get("root", 60)
    zones = sampler_zones(str(layer["dir"]), str(layer.get("pattern", "*")), note_to_midi(root) if isinstance(root, str) else int(root))
    rel = float(layer.get("release", 0.08))
    legato = float(layer.get("legato", 1.0))
    gap = float(layer.get("gap", 0.0)) * q
    shift = float(layer.get("transpose", 0)) + float(layer.get("tune", 0.0)) / 100.0
    max_shift = layer.get("max_shift")
    vel0 = float(layer.get("vel", 1.0))
    rr: dict = {}                                  # round robin restarts with every render: deterministic
    parts, t = [], 0.0
    for st in _steps(layer):
        d = float(st["dur"]) * q
        v = float(np.clip(vel0 * float(st.get("vel", 1.0)), 0.0, 1.0))
        li = min(len(zones) - 1, int(v * len(zones)))
        notes = zones[li][1]
        for f in st["pitches"]:
            midi = 69.0 + 12.0 * math.log2(f / 440.0) + shift
            m, files = min(notes, key=lambda z: abs(z[0] - midi))
            if max_shift is not None and abs(midi - m) > float(max_shift):
                raise ValueError(f"sampler: note {midi:.1f} is {abs(midi - m):.1f} semitones from the nearest sample ({m})")
            c = rr.get((li, m), 0)
            rr[(li, m)] = c + 1
            x, fs = _sample(files[c % len(files)])
            y = np.array(_repitch(x, fs, sr, midi - m), dtype=np.float64)
            if rel > 0:
                on, rn = samples(d * legato, sr), samples(rel, sr)
                y = y[:on + rn]
                if len(y) > on:                    # sampler._release_curve: -60 dB, then exactly 0
                    u = np.linspace(0.0, 1.0, len(y) - on)
                    g = np.exp(-6.9 * u) * 0.5 * (1 + np.cos(np.pi * np.clip((u - 0.85) / 0.15, 0, 1)))
                    y[on:] *= g if y.ndim == 1 else g[:, None]
            k = min(len(y), max(2, int(0.0015 * sr)))
            fi = np.linspace(0.0, 1.0, k)
            y[:k] *= fi if y.ndim == 1 else fi[:, None]
            y[-k:] *= fi[::-1] if y.ndim == 1 else fi[::-1][:, None]
            parts.append((y * 10.0 ** (-DYN_RANGE_DB * (1.0 - v) ** 1.15 / 20.0), t))
        t += d + gap
    return mix(parts, sr) if parts else silence(t or 0.1, sr)


# ====================================================================== QA harness
def _qa_row(kind, name, fn, sr, deep=None):
    from .analysis import brightness, loudness, sharpness
    row = {"kind": kind, "name": name, "ok": False, "flags": []}
    t0 = time.perf_counter()
    try:
        y = np.asarray(fn(), dtype=np.float64)
    except Exception as e:  # one bad entry must not stop the catalogue (catalogue.render_id)
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        row["flags"].append("error")
        return row
    row["seconds"] = round(time.perf_counter() - t0, 3)
    row["dur"] = round(y.shape[0] / sr, 3)
    if y.size == 0 or not np.all(np.isfinite(y)):
        row["flags"].append("non-finite" if y.size else "empty")
        return row
    m = y.mean(axis=1) if y.ndim == 2 else y
    pk = float(np.max(np.abs(m)))
    row.update(peak=round(pk, 4), loud=round(loudness(m, sr), 1), sharp=round(float(sharpness(m, sr)), 2),
               above5k=round(float(brightness(m, sr)["above_5k"]), 3))
    if pk < 1e-4:
        row["flags"].append("silent")
    if pk > 1.5:
        row["flags"].append("hot")
    if kind != "effect":
        if abs(float(np.mean(m))) > 0.02 * max(pk, 1e-9) and abs(float(np.mean(m))) > 1e-3:
            row["flags"].append("dc")
        if abs(m[-1]) > 0.05 * max(pk, 1e-9) and abs(m[-1]) > 1e-3:
            row["flags"].append("ends-on-step")
        if row["sharp"] > 2.4 and kind == "instrument":
            row["flags"].append("piercing")
    if deep is not None:
        try:
            row["flags"] += deep(y)
        except Exception as e:
            row["flags"].append(f"deep-error:{type(e).__name__}")
    row["ok"] = not row["flags"]
    return row


def coverage(only=None, kinds=("sfx", "instrument", "drum", "effect"), sr: int = DEFAULT_SR, deep: bool = False) -> list[dict]:
    """Render every registered catalog entry with its default parameters and measure it.

    The library-wide version of procedural's catalogue / coverage / qa scripts (render every id,
    never stop on one failure, log a record per id). Returns one dict per entry: ``kind``,
    ``name``, ``ok``, ``flags`` (error, empty, non-finite, silent: peak < 1e-4, hot: peak > 1.5,
    dc, ends-on-step, piercing: instrument sharper than 2.4 acum), and when it rendered:
    ``peak``, ``loud`` (K-weighted dB of the loudest 0.4 s), ``sharp`` (acum), ``above5k``
    (share of energy above 5 kHz), ``dur``, ``seconds`` (render time).
    ``deep`` (sfx only, three more renders each): ``non-deterministic`` (two renders differ),
    ``seed-ignored`` (has a ``seed`` param but seed + 1 gives the same samples: the near-duplicate
    takes of qa_catalogue), ``fails-22050`` (not finite and bounded at 22050 Hz).
    ``only``: a name, a prefix ending in ``*``, or a list of those. Instruments play the middle
    of their range for 0.5 s at vel 0.9; effects process a 0.5 s test signal (220 Hz tone + click)."""
    from . import drums as D
    from . import fx as FX
    from . import sfx as SX
    from .notes import midi_to_freq, note_to_midi
    pats = [only] if isinstance(only, str) else list(only or [])

    def want(name):
        return not pats or any(name == p or (p.endswith("*") and name.startswith(p[:-1])) for p in pats)
    t = np.arange(samples(0.5, sr)) / sr
    probe = 0.5 * np.sin(2 * np.pi * 220.0 * t) * np.minimum(1.0, (0.5 - t) / 0.05)
    probe[:8] += 0.5
    rows = []
    if "sfx" in kinds:
        def deep_sfx(n):
            def check(y):
                f = [] if np.array_equal(y, SX.render_sfx(n, sr)) else ["non-deterministic"]
                sd = SX.REGISTRY[n]["params"].get("seed")
                if sd is not None and np.array_equal(y, SX.render_sfx(n, sr, seed=int(sd[0]) + 1)):
                    f.append("seed-ignored")
                z = np.asarray(SX.render_sfx(n, 22050))
                if not (np.all(np.isfinite(z)) and 1e-4 < np.max(np.abs(z)) < 1.5):
                    f.append("fails-22050")
                return f
            return check if deep else None
        rows += [_qa_row("sfx", n, lambda n=n: SX.render_sfx(n, sr), sr, deep_sfx(n)) for n in sorted(SX.REGISTRY) if want(n)]
    if "instrument" in kinds:
        for n in sorted(I.REGISTRY):
            if want(n):
                lo, hi = (note_to_midi(x) for x in I.REGISTRY[n]["range"].split("-"))
                rows.append(_qa_row("instrument", n, lambda n=n, f=midi_to_freq((lo + hi) // 2): I.render_note(n, f, 0.5, sr, 0.9), sr))
    if "drum" in kinds:
        rows += [_qa_row("drum", n, lambda n=n: D.render_drum(n, sr, 0.9), sr) for n in sorted(D.REGISTRY) if want(n)]
    if "effect" in kinds:
        rows += [_qa_row("effect", n, lambda n=n: FX.REGISTRY[n]["fn"](probe.copy(), sr), sr) for n in sorted(FX.REGISTRY) if want(n)]
    return rows


def report(rows: list[dict]) -> str:
    """Plain-text table of ``coverage()`` rows, failures first."""
    out = [f"{'kind':10s} {'name':22s} {'peak':>7s} {'loud':>6s} {'sharp':>5s} {'>5k':>5s} {'dur':>6s} {'sec':>6s}  flags"]
    for r in sorted(rows, key=lambda r: (r["ok"], r["kind"], r["name"])):
        if "peak" in r:
            out.append(f"{r['kind']:10s} {r['name']:22s} {r['peak']:7.3f} {r['loud']:6.1f} {r['sharp']:5.2f} {r['above5k']:5.2f} "
                       f"{r['dur']:6.2f} {r['seconds']:6.2f}  {' '.join(r['flags'])}")
        else:
            out.append(f"{r['kind']:10s} {r['name']:22s} {'-':>7s} {'-':>6s} {'-':>5s} {'-':>5s} {'-':>6s} {'-':>6s}  "
                       f"{' '.join(r['flags'])} {r.get('error', '')}")
    bad = sum(not r["ok"] for r in rows)
    out.append(f"{len(rows)} entries, {bad} flagged")
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="python -m genny.klang", description="Catalog QA: render every registered name and report failures and levels.")
    ap.add_argument("--qa", action="store_true", help="run the harness (the default action)")
    ap.add_argument("--only", nargs="*", default=[], help="names or prefixes ending in * (e.g. klang_* harrier)")
    ap.add_argument("--kind", nargs="*", default=["sfx", "instrument", "drum", "effect"], help="sfx instrument drum effect")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--deep", action="store_true", help="sfx: also check determinism, that the seed changes the sound, and 22050 Hz")
    ap.add_argument("--json", default="", help="also write the rows to this JSON file")
    a = ap.parse_args(argv)
    rows = coverage(a.only, tuple(a.kind), a.sr, a.deep)
    print(report(rows))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if any(not r["ok"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
