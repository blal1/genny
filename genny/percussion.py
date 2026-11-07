"""World, Latin, jazz and orchestral percussion, registered into the drum REGISTRY
(imported at the bottom of drums.py), so each is used like any other drum:
{"type": "drum", "kind": "conga", "params": {"slap": 1}} or in a "pattern" layer.

Hand drums are a few membrane modes with a small pitch drop plus a shaped noise slap; metal
percussion is a handful of inharmonic partials kept below ~5 kHz so it rings without stabbing.
"""
from __future__ import annotations

import math

import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, samples
from .drums import _dec, bridged_t, drum
from .env import perc


def _head(n, sr, f0, modes, bend=0.06, bend_tau=0.02, vel=1.0):
    """Drum-head modes [(ratio, amp, tau), ...] with a pitch drop of `bend` (fraction, at vel 1, scaled by vel^2)."""
    t = np.arange(n) / sr
    ph = np.cumsum(f0 * (1 + bend * float(np.clip(vel, 0.0, 1.0)) ** 2 * np.exp(-t / bend_tau)) / sr)
    y = np.zeros(n)
    for r, a, tau in modes:
        y += a * np.sin(2 * np.pi * r * ph) * np.exp(-t / tau)
    return y


def _slap(n, sr, center, q, tau, seed):
    return F.bandpass(O.white(n, seed=seed), center, sr, q) * _dec(n, tau, sr)


def _hand_drum(sr, vel, tune, slap, ring, seed):
    n = samples(ring * 4 + 0.05, sr)
    slap = float(np.clip(slap, 0.0, 1.0))
    tone = _head(n, sr, tune, [(1.0, 1.0, ring * (1 - 0.7 * slap)), (1.59, 0.3, ring * 0.4), (2.14, 0.15, ring * 0.25)], vel=vel)
    skin = _slap(n, sr, 1800 + 1000 * slap, 1.0, 0.004 + 0.006 * slap, seed) * (0.25 + 1.0 * slap)
    y = np.tanh((tone * (1 - 0.5 * slap) + skin) * 1.4)
    return y * perc(n / sr, 0.0005, sr, curve=1.0) * (0.5 + 0.5 * vel)


@drum("conga", "Conga: open tone, or a cracking slap (salsa, Afro-Cuban, funk).", tune=(220, "Hz (tumba 150, conga 220, quinto 300)"),
      slap=(0.0, "0 = open tone .. 1 = slap"))
def conga(sr=DEFAULT_SR, vel=1.0, tune=220, slap=0.0):
    return _hand_drum(sr, vel, tune, slap, 0.2, 71)


@drum("bongo", "Bongo: small, tight, high hand drum.", tune=(420, "Hz (hembra 320, macho 420)"), slap=(0.0, "0 = open .. 1 = slap"))
def bongo(sr=DEFAULT_SR, vel=1.0, tune=420, slap=0.0):
    return _hand_drum(sr, vel, tune, slap, 0.1, 72)


@drum("djembe", "Djembe: `stroke` bass (deep centre hit), tone (round edge hit) or slap (sharp crack).",
      tune=(330, "Hz of the tone stroke"), stroke=("tone", "bass|tone|slap"))
def djembe(sr=DEFAULT_SR, vel=1.0, tune=330, stroke="tone"):
    if stroke == "bass":
        n = samples(0.6, sr)
        y = _head(n, sr, tune * 0.24, [(1.0, 1.0, 0.16), (1.6, 0.25, 0.06)], bend=0.25, bend_tau=0.03, vel=vel)
        y = np.tanh((y + _slap(n, sr, 600, 0.8, 0.008, 73) * 0.3) * 1.6)
        return y * perc(0.6, 0.001, sr, curve=1.0) * (0.5 + 0.5 * vel)
    return _hand_drum(sr, vel, tune, 1.0 if stroke == "slap" else 0.1, 0.13, 74)


@drum("tabla", "Tabla: `stroke` na (ringing treble drum), ge (bass drum that bends upward) or ke (dry slap).",
      tune=(370, "Hz of the treble drum"), stroke=("na", "na|ge|ke"))
def tabla(sr=DEFAULT_SR, vel=1.0, tune=370, stroke="na"):
    if stroke == "ge":
        n = samples(0.8, sr)
        t = np.arange(n) / sr
        f = tune * 0.23 * (1 + 0.45 * (1 - np.exp(-t / 0.07)) * np.exp(-t / 0.5))   # the heel of the hand presses the head
        y = O.sine(f, n, sr) * _dec(n, 0.3, sr) + 0.25 * O.sine(f * 2, n, sr) * _dec(n, 0.12, sr)
        y += _slap(n, sr, 500, 0.8, 0.006, 75) * 0.3
        return np.tanh(y * 1.5) * perc(0.8, 0.001, sr, curve=1.0) * (0.5 + 0.5 * vel)
    if stroke == "ke":
        n = samples(0.12, sr)
        y = _slap(n, sr, 1300, 1.2, 0.01, 76) * 1.2 + O.sine(tune * 0.9, n, sr) * _dec(n, 0.012, sr) * 0.5
        return np.tanh(y * 1.4) * perc(0.12, 0.0005, sr, curve=1.0) * (0.5 + 0.5 * vel)
    n = samples(0.9, sr)
    # the loaded head makes the modes harmonic: that is the tabla's singing "na"
    y = _head(n, sr, tune, [(1.0, 0.6, 0.22), (2.0, 1.0, 0.28), (3.0, 0.5, 0.18), (4.0, 0.18, 0.08)], bend=0.01, vel=vel)
    y += _slap(n, sr, 2200, 1.5, 0.003, 77) * 0.4
    return np.tanh(y * 1.1) * perc(0.9, 0.0005, sr, curve=1.0) * 0.8 * (0.5 + 0.5 * vel)


@drum("cajon", "Cajon (box drum): `stroke` bass (boxy thump) or snare (wooden slap with wire buzz); an acoustic drum kit in one box.",
      stroke=("bass", "bass|snare"))
def cajon(sr=DEFAULT_SR, vel=1.0, stroke="bass"):
    n = samples(0.4, sr)
    if stroke == "snare":
        y = _head(n, sr, 210, [(1.0, 0.8, 0.05), (1.7, 0.4, 0.03)], bend=0.1, vel=vel)
        y += _slap(n, sr, 2400, 0.7, 0.045, 78) * 0.9 + _slap(n, sr, 900, 1.0, 0.01, 79) * 0.5
    else:
        y = _head(n, sr, 78, [(1.0, 1.0, 0.11), (2.3, 0.3, 0.04)], bend=0.3, bend_tau=0.02, vel=vel)
        y += _slap(n, sr, 500, 0.8, 0.008, 80) * 0.5
    return np.tanh(y * 1.5) * perc(0.4, 0.0005, sr, curve=1.0) * (0.5 + 0.5 * vel)


@drum("timbale", "Timbale: ringing metal-shell drum (salsa fills, mambo).", tune=(300, "Hz"))
def timbale(sr=DEFAULT_SR, vel=1.0, tune=300):
    n = samples(0.7, sr)
    y = _head(n, sr, tune, [(1.0, 1.0, 0.16), (1.59, 0.5, 0.11), (2.14, 0.3, 0.09), (2.65, 0.2, 0.07), (4.1, 0.12, 0.09)], bend=0.04, vel=vel)
    y += _slap(n, sr, 2600, 1.0, 0.004, 81) * 0.5
    return np.tanh(y * 1.2) * perc(0.7, 0.0005, sr, curve=1.0) * 0.8 * (0.5 + 0.5 * vel)


@drum("clave", "Claves: two hardwood sticks, the timekeeper of son and rumba.", tune=(1900, "Hz"))
def clave(sr=DEFAULT_SR, vel=1.0, tune=1900):
    n = samples(0.15, sr)
    y = O.sine(tune, n, sr) * _dec(n, 0.03, sr) + 0.15 * O.sine(tune * 2.7, n, sr) * _dec(n, 0.008, sr)
    y += _slap(n, sr, min(tune, 2500), 2.0, 0.002, 82) * 0.3
    return y * perc(0.15, 0.0005, sr, curve=1.0) * 0.7 * (0.5 + 0.5 * vel)


@drum("guiro", "Guiro: stick scraped along a ridged gourd (cha-cha, cumbia).", dur=(0.25, "seconds of scrape"), ridges=(28, "ticks in the scrape"))
def guiro(sr=DEFAULT_SR, vel=1.0, dur=0.25, ridges=28):
    n = samples(dur + 0.05, sr)
    y = np.zeros(n)
    rng = np.random.default_rng(83)
    tick = samples(0.006, sr)
    for i in range(int(ridges)):
        s = int((i / ridges) ** 0.85 * dur * sr)   # the stroke speeds up slightly
        e = min(n, s + tick)
        y[s:e] += rng.uniform(-1, 1, e - s) * np.exp(-np.arange(e - s) / (0.0015 * sr)) * (0.6 + 0.4 * rng.random())
    y = F.lowpass(F.bandpass(y, 2600, sr, 1.2), 5500, sr)
    return y * perc(dur + 0.05, 0.02, sr, curve=0.7) * 3.2 * (0.5 + 0.5 * vel)


def meyer_triangle_t60(f):
    return 30.0 * 400.0 / np.asarray(f, float)


TRIANGLE_BETA = (4.7300, 7.8532, 10.9956, 14.1372)   # free-free bar roots beta_n L; then (2n+1) pi / 2
TRIANGLE_SPLIT = (0.0, -0.021, 0.017)                 # the bends split each bar mode: two polarisations, three sides
TRIANGLE_CONTACT_MS = 0.14                            # steel beater on steel rod


def triangle_modes(f1=400.0, fmax=20000.0, strike=0.12):
    """Dense mode set of the triangle as a bent free-free rod: f_n = f1 (beta_n / beta_1)^2 for every bar mode,
    each split in three by the bends, amplitude = |mode shape at the strike point| x f |P(f)| (bar velocity
    falls as 1/f, the thin rod's dipole radiation rises as f^2, P = raised-cosine contact spectrum).
    Returns (freqs, amps) with amps peaking where the contact rolls the radiation off (about 6 kHz)."""
    fr, am = [], []
    tau = TRIANGLE_CONTACT_MS * 1e-3
    for n in range(1, 60):
        b = TRIANGLE_BETA[n - 1] if n <= 4 else (2 * n + 1) * np.pi / 2
        f = f1 * (b / TRIANGLE_BETA[0]) ** 2
        if f > fmax:
            break
        x = b * strike
        c = (np.cosh(b) - np.cos(b)) / (np.sinh(b) - np.sin(b))
        shape = abs(np.cosh(x) + np.cos(x) - c * (np.sinh(x) + np.sin(x))) / 2.0
        wt = 2 * np.pi * f * tau
        P = abs(np.sinc(wt / (2 * np.pi)) / (1.0 - (wt / (2 * np.pi)) ** 2 + 1e-9))
        for d in TRIANGLE_SPLIT:
            fr.append(f * (1.0 + d))
            am.append(shape * f * P)
    am = np.asarray(am)
    return np.asarray(fr), am / am.max()


@drum("triangle", "Triangle: a silvery ting whose partials ring for seconds (30 s at 400 Hz, halving per octave); "
      "`open` 0 gives the damped, short stroke; `axis` parallel = struck in its own plane: fewer, near-harmonic partials.",
      open=(1.0, "0 = damped .. 1 = ringing"), tune=(2100, "Hz of the lowest high partial (the 400 Hz partial follows)"),
      axis=("perpendicular", "perpendicular (normal stroke) | parallel (in the plane: 400, 1600, 2000 Hz)"),
      dur=(4.0, "s rendered when open; the low partial rings far longer and is faded out"))
def triangle(sr=DEFAULT_SR, vel=1.0, open=1.0, tune=2100, axis="perpendicular", dur=4.0):
    o = float(np.clip(open, 0.0, 1.0))
    k = float(tune) / 2100.0
    if str(axis) == "parallel":
        parts = ((400.0 * k, 1.0, 1.0), (1600.0 * k, 0.6, 0.6), (2000.0 * k, 0.5, 0.6))
    else:
        fr, am = triangle_modes(400.0 * k, 0.45 * sr)
        parts = tuple((f, a / float(np.sqrt(np.sum(am * am))), 1.0) for f, a in zip(fr, am))
    total = max(float(dur), 0.5) if o > 0.0 else 0.3
    n = samples(total, sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for f, a, d in parts:
        if f >= 0.45 * sr:
            continue
        t60 = meyer_triangle_t60(f) ** o * (6.91 * 0.08 * d / 3.0) ** (1.0 - o)
        y += a * np.sin(2 * np.pi * f * t) * np.exp(-6.91 * t / t60)
    env = np.ones(n)
    a4 = samples(0.004, sr)
    env[:a4] = np.linspace(0.0, 1.0, a4)
    _fade_out(env, sr, 0.3)
    return y * env * 0.25 * (0.5 + 0.5 * vel)


def meyer_bass_drum_t60(f):
    f =np.asarray(f, float)
    mid = 15.0 * (np.maximum(f, 50.0) / 50.0) ** (np.log(4.0 / 15.0) / np.log(4.0))
    return np.where(f <= 50.0, 15.0, np.where(f < 200.0, mid, np.where(f <= 400.0, 4.0, 4.0 * 400.0 / np.maximum(f, 1.0))))


@drum("bass_drum", "Orchestral bass drum (gran cassa): a dull boom near 100 Hz from many inharmonic membrane modes, "
      "close pairs that beat (the 'breathing'), a pitch that sinks up to 140 cents in the first second after a hard "
      "hit, and lows that ring for many seconds. `mallet` felt (round, dark) or rute (split-reed switch: short, hard).",
      tune=(50.0, "Hz of the (0,1) mode"), pos=(0.25, "strike point, 0 = centre (ring modes only, dull) .. 0.9 = near "
      "the rim (asymmetric modes, harsher)"), mallet=("felt", "felt | rute"), dur=(4.0, "s rendered (lows faded out)"))
def bass_drum(sr=DEFAULT_SR, vel=1.0, tune=50.0, pos=0.25, mallet="felt", dur=4.0):
    from scipy.special import jn_zeros, jv
    from scipy.signal import fftconvolve
    from .physical.modal import contact_time
    r = float(np.clip(pos, 0.0, 0.95))
    j01 = jn_zeros(0, 1)[0]
    n = samples(max(float(dur), 0.5), sr)
    t = np.arange(n) / sr
    v = float(np.clip(vel, 0.0, 1.0))
    bend = 1.0 + (2.0 ** (140.0 / 1200.0) - 1.0) * v * v * np.exp(-t / 0.25)
    y = np.zeros(n)
    for m in range(13):
        for j in jn_zeros(m, 6):
            f = float(tune) * j / j01
            if f > 0.4 * sr:
                continue
            g = jv(m, j * r) / abs(jv(m + 1, j)) / j
            ph = 2.0 * np.pi * np.cumsum(f * bend) / sr
            d = np.exp(-6.91 * t / float(meyer_bass_drum_t60(f)))
            if m == 0:
                y += g * np.sin(ph) * d
            else:
                y += 0.5 * g * (np.sin(ph) + np.sin(ph * 1.004 + 1.0)) * d
    ms = 1.0 if str(mallet) == "rute" else 6.0
    y = fftconvolve(y, _pulse(contact_time(ms, max(v, 0.05)), sr))[:n]
    _fade_out(y, sr, 0.4)
    return y / (np.max(np.abs(y)) + 1e-12) * 0.9 * (0.5 + 0.5 * vel)


def _pulse(ms, sr):
    """Raised-cosine contact force of `ms` milliseconds at `sr`, unit area."""
    k = max(2, int(round(ms * 1e-3 * sr)))
    p = 0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(k) + 1) / (k + 1)))
    return p / p.sum()


@drum("snap", "Finger snap (lo-fi, jazz, pop backbeats).")
def snap(sr=DEFAULT_SR, vel=1.0):
    n = samples(0.12, sr)
    y = _slap(n, sr, 2100, 1.4, 0.006, 85)
    y[samples(0.004, sr):] += 0.6 * _slap(n, sr, 1500, 2.0, 0.012, 86)[:n - samples(0.004, sr)]
    return np.tanh(y * 5.0) * perc(0.12, 0.0005, sr, curve=1.0) * 0.9 * (0.5 + 0.5 * vel)


@drum("brush", "Jazz brush on a snare: `swish` 0 is a soft tap, 1 a long stirred sweep.", swish=(0.0, "0 = tap .. 1 = sweep"), dur=(0.18, "seconds"))
def brush(sr=DEFAULT_SR, vel=1.0, swish=0.0, dur=0.18):
    swish = float(np.clip(swish, 0, 1))
    n = samples(dur * (1 + 2 * swish) + 0.05, sr)
    nz = F.lowpass(F.bandpass(O.pink(n, seed=87), 2200, sr, 0.6), 6000, sr)
    env = perc(n / sr, 0.004 + 0.12 * swish, sr, curve=2.0 - swish)
    tap = O.sine(190, n, sr) * _dec(n, 0.03, sr) * 0.4 * (1 - swish)
    return (nz * env * 2.0 + tap * 1.5) * (0.5 + 0.5 * vel)


@drum("snare808", "808 snare: two short tones under a soft noise burst (electro, hip-hop, synth-pop).", tune=(249.6, "Hz (low bridged-T; the high one follows at its circuit ratio)"), snappy=(0.6, "noise amount"))
def snare808(sr=DEFAULT_SR, vel=1.0, tune=249.6, snappy=0.6):
    n = samples(0.3, sr)
    (fl, _, tl), (fh, _, th) = bridged_t("snare_low"), bridged_t("snare_high")
    k = tune / fl
    tone = O.sine(tune, n, sr) * _dec(n, tl / k, sr) + 0.6 * O.sine(fh * k, n, sr) * _dec(n, th / k, sr)
    nz = F.lowpass(F.highpass(O.white(n, seed=88), 1200, sr), 6500, sr) * _dec(n, 0.07, sr) * snappy * 1.4
    return np.tanh((tone + nz) * 1.3) * perc(0.3, 0.0005, sr, curve=1.0) * (0.5 + 0.5 * vel)


@drum("sleigh", "Sleigh bells: a shake of many small bells (Christmas, winter levels).", decay=(0.35, "s"))
def sleigh(sr=DEFAULT_SR, vel=1.0, decay=0.35):
    n = samples(decay + 0.1, sr)
    rng = np.random.default_rng(89)
    y = np.zeros(n)
    for _ in range(9):   # each bell is hit at a slightly different moment
        s = int(rng.uniform(0, 0.06) * sr)
        f = rng.uniform(2300, 4800)
        m = n - s
        tt = np.arange(m) / sr
        y[s:] += (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 1.47 * tt)) * np.exp(-tt / (decay / 3)) * rng.uniform(0.5, 1.0)
    y = F.lowpass(y, 6500, sr)
    return y / 4.0 * perc(decay + 0.1, 0.001, sr, curve=1.0) * (0.5 + 0.5 * vel)

def triad(f3, n, sr, a3, t60, split=0.001, threshold=0.05, seed=1e-3):
    """Chaigne, Touze & Thomas, "Nonlinear vibrations and chaos in gongs and cymbals", Acoust. Sci. & Tech.
    26(5) (2005), eq. 9: a symmetric mode q3 at w3 struck to amplitude a3, quadratically coupled to twin modes
    q1, q2 at w3 / 2 (1 -+ split) (internal resonance 1:1:2), V = alpha / 2 (q1^2 + q2^2) q3. Below the parametric
    threshold a3* = 4 mu w1 / alpha the twins stay silent; above it they grow, after a delay, as the order-2
    subharmonics of their Fig. 4, and trade energy back and forth with q3. Returns (q1 + q2, q3). alpha
    (set so that a3* = ``threshold``: a full strike blooms after about 0.5 s, below 0.3 nothing), the seed (an
    off-centre strike gives the twins 1e-3 of the amplitude directly), the
    twin split and their damping (that of q3)."""
    w3 = 2.0 * np.pi * f3
    w1, w2 = 0.5 * w3 * (1.0 - split), 0.5 * w3 * (1.0 + split)
    mu = 6.91 / max(t60, 1e-3)
    al = 4.0 * mu * w1 / threshold
    return _triad(int(n), 1.0 / sr, w1, w2, w3, mu, al, float(a3) * w3, seed * float(a3))


import numba  # noqa: E402


@numba.njit(cache=True)
def _triad(n, dt, w1, w2, w3, mu, al, v3, s):
    out = np.zeros(n)
    par = np.zeros(n)
    q1, q2, q3 = s, -s, 0.0                                         # an off-centre strike: the twins get a little directly
    p1, p2, p3 = 0.0, 0.0, v3                                       # strike = initial velocity of q3
    for i in range(n):                                              # symplectic Euler
        p1 += dt * (-w1 * w1 * q1 - al * q1 * q3 - 2.0 * mu * p1)
        p2 += dt * (-w2 * w2 * q2 - al * q2 * q3 - 2.0 * mu * p2)
        p3 += dt * (-w3 * w3 * q3 - 0.5 * al * (q1 * q1 + q2 * q2) - 2.0 * mu * p3)
        q1 += dt * p1
        q2 += dt * p2
        q3 += dt * p3
        out[i] = q1 + q2
        par[i] = q3
    return out, par


BRONZE_SHELL = (8800.0, 1.24e11, 0.33)       # rho, E, nu of gong / cymbal bronze
GONG_GLIDE_CENTS = 80.0                      # pitch rise of a fortissimo gong stroke, cents (vel^2 law)
GONG_GLIDE_SETTLE = 4.0                      # s until the fortissimo pitch is stable (x vel)
AGUNG_REF = 90.0                             # Hz: tune the agung partials below are given for
AGUNG_PARTIALS = ((1.0, 1.0, 4.0, 1.0), (495.0 / AGUNG_REF, 0.35, 0.25, 0.5), (626.0 / AGUNG_REF, 0.3, 1.0 / 6.0, 0.5))
# (ratio, level, doublet split Hz, decay x `decay`): a split d beats as |cos(pi d t)|, silent at 1/(2d) s, then back


def agung(tune, dur, sr, vel=1.0, decay=6.0):
    """Balinese hanging gong: each partial a beating doublet; the fundamental pulses at the 4 Hz ombak, the 495 and
    626 Hz partials die away and come back (626 Hz at 3 s)."""
    n = samples(float(dur), sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for r, a, d, k in AGUNG_PARTIALS:
        f = float(tune) * r
        if f + d < 0.45 * sr:
            y += a * np.exp(-t / (k * float(decay))) * 0.5 * (np.sin(2 * np.pi * (f - d / 2) * t)
                                                              + np.sin(2 * np.pi * (f + d / 2) * t + 0.0))
    return y * np.minimum(1.0, t / 0.004) * float(np.clip(vel, 0.0, 1.0))


def shell_freqs(freqs, radius, material=BRONZE_SHELL):
    """Modes of a shallow spherical shell of radius of curvature `radius` (m) from those of the flat plate:
    w^2 = w_plate^2 + E / (rho R^2 (1 - nu^2)) (membrane stiffening, the same for every mode)."""
    rho, E, nu = material
    fr = math.sqrt(E / (rho * (1.0 - nu * nu))) / (2.0 * math.pi * radius) if radius and radius > 0 else 0.0
    return np.sqrt(np.asarray(freqs, float) ** 2 + fr * fr)


def pitch_glide(y, sr, cents, settle):
    """Warp `y` so its pitch starts `cents` low (negative: high) and reaches the nominal one with time constant
    settle / 3 (stable after `settle` s)."""
    n = len(y)
    t = np.arange(n) / sr
    rate = 2.0 ** (-cents * np.exp(-3.0 * t / max(settle, 1e-3)) / 1200.0)
    return np.interp(np.cumsum(rate) / sr, t, y, right=0.0)


# v0.6 physical migration: gong --------------------------------------------------
@drum("gong", "Physical modal gong/tam-tam with inharmonic resonances; a hard stroke starts flat and its pitch "
      "rises up to 80 cents over the first seconds (less and faster when soft).", tune=(90, "Hz"), decay=(6.0, "seconds"),
      cascade=(0.0, "0..1 level of the 1:1:2 internal-resonance twins (Chaigne & Touze): a hard strike blooms "
                    "a fifth-free subharmonic below the main groups a moment after the hit; soft strikes stay clean"),
      glide=(1.0, "0..1 amount of the rising pitch glide (1 = the measured 80 cents at full velocity)"),
      curvature=(0.0, "m radius of curvature of the gong face (0 = flat): a domed gong lifts every mode by the "
                      "shell stiffening, the low ones most"),
      preset=("", "'agung' = Balinese hanging gong: 4 Hz ombak on the fundamental, upper partials that fade "
                  "and swell back (626 Hz returns at 3 s)"),
      hardness=(0.0, "0..1 the fundamental as a hardening (cubic) mode: a hard stroke starts sharp and sinks to the nominal pitch as it dies"))
def gong(sr=DEFAULT_SR, vel=1.0, tune=90, decay=6.0, cascade=0.0, glide=1.0, curvature=0.0, preset="", hardness=0.0):
    from .physics import TunedPercussion
    if preset == "agung":
        y = agung(tune, max(float(decay), 4.0), sr, vel, decay)
    elif preset:
        raise ValueError("gong preset must be '' or 'agung'")
    elif curvature and curvature > 0.0:
        from .physical.modal_data import GONG
        n = samples(float(decay), sr)
        t = np.arange(n) / sr
        y = np.zeros(n)
        fs_ = shell_freqs(float(tune) * np.asarray(GONG["ratios"], float), float(curvature))
        for i, (f, at, de) in enumerate(zip(fs_, GONG["attack_ms"], GONG["decay_ms"])):
            if f >= 0.45 * sr:
                continue
            inst = f + GONG["fsk_depth_hz"] * (0.04 / (i + 1)) * np.sin(2 * np.pi * 0.37 * t + i)
            y += (np.minimum(1.0, t / max(at * 1e-3, 1e-4)) * np.exp(-t / max(de * 1e-3, 1e-4))
                  * np.sin(2 * np.pi * np.cumsum(inst) / sr + 0.7 * i) / (1 + 0.25 * i))
        y *= float(np.clip(vel, 0.0, 1.0))
    else:
        y = TunedPercussion("gong", decay=decay, hardness=.48).note(float(tune), float(decay), velocity=vel, sr=sr, seed=84)
    v = float(np.clip(vel, 0.0, 1.0))
    if glide > 0.0 and v > 0.0:
        y = pitch_glide(y, sr, float(np.clip(glide, 0.0, 2.0)) * GONG_GLIDE_CENTS * v * v, GONG_GLIDE_SETTLE * v)
    c = float(np.clip(cascade, 0.0, 1.0))
    if c > 0.0:
        from .physical.modal_data import GONG
        sub = np.zeros(len(y))
        for r in GONG["ratios"][:2]:                                 # the two deep groups
            tw, _ = triad(float(tune) * r, len(y), sr, float(np.clip(vel, 0.0, 1.0)), float(decay))
            sub += tw
        y = y + c * sub / (np.max(np.abs(sub)) + 1e-12) * np.max(np.abs(y)) * float(np.clip(vel, 0.0, 1.0))
    h = float(np.clip(hardness, 0.0, 1.0))
    if h > 0.0 and v > 0.0:
        w0 = 2 * math.pi * float(tune)
        u = cubic_mode(float(tune), float(decay) / 6.91, v, h * w0 * w0, len(y), sr)
        y = y + h * u / (np.max(np.abs(u)) + 1e-12) * np.max(np.abs(y))
    return np.tanh(y*1.15) * .8


@numba.njit(cache=True)
def _cubic_kernel(n, k, w0, sig, eps, u1):
    u = np.zeros(n)
    if n > 1:
        u[1] = u1
    for i in range(1, n - 1):
        q = 0.5 * k * k * eps * u[i] * u[i]
        u[i + 1] = (2.0 * u[i] - k * k * w0 * w0 * u[i] - u[i - 1] * (1.0 - sig * k + q)) / (1.0 + sig * k + q)
    return u


def cubic_mode(f0, tau, amp, eps, n, sr):
    """One mode u'' + 2 sig u' = -w0^2 u - eps u^3, sig = 1 / tau, by the energy-conserving scheme
    d_tt u + 2 sig d_t. u = -w0^2 u - eps u^2 mu_t. u (u^2 times the mean of u^{n+1} and u^{n-1}): for eps > 0 the
    discrete energy never grows, whatever the amplitude. Starts from rest with velocity amp w0."""
    w0 = 2 * math.pi * f0
    k = 1.0 / sr
    return _cubic_kernel(int(n), k, w0, 1.0 / max(tau, 1e-4), float(eps), amp * w0 * k)

# v0.7 physical membrane migration --------------------------------------------
@drum("conga", "Physical circular-membrane conga with position-dependent modes and contact slap.",
      tune=(220, "Hz"), slap=(0.0, "0=open tone .. 1=slap"))
def conga(sr=DEFAULT_SR, vel=1.0, tune=220, slap=0.0):
    from .physics import MembraneDrum
    s=float(np.clip(slap,0,1))
    return MembraneDrum(float(tune), .145, damping=1.3, air_coupling=.28).strike(
        .62, velocity=vel, position=.48+.38*s, stroke="slap" if s>.55 else "tone", sr=sr, seed=71)


@drum("bongo", "Physical small circular-membrane bongo.", tune=(420,"Hz"), slap=(0.0,"0=open .. 1=slap"))
def bongo(sr=DEFAULT_SR, vel=1.0, tune=420, slap=0.0):
    from .physics import MembraneDrum
    s=float(np.clip(slap,0,1))
    return MembraneDrum(float(tune), .085, damping=1.7, air_coupling=.17).strike(
        .38, velocity=vel, position=.52+.35*s, stroke="slap" if s>.55 else "tone", sr=sr, seed=72)


@drum("djembe", "Physical djembe membrane/cavity model: bass, tone or slap.",
      tune=(330,"Hz"), stroke=("tone","bass|tone|slap"))
def djembe(sr=DEFAULT_SR, vel=1.0, tune=330, stroke="tone"):
    from .physics import MembraneDrum
    st=str(stroke).lower()
    if st=="bass":
        return MembraneDrum(float(tune)*.24,.17,damping=.9,air_coupling=.55).strike(
            .72,velocity=vel,position=.06,stroke="bass",sr=sr,seed=73)
    return MembraneDrum(float(tune),.17,damping=1.25,air_coupling=.34).strike(
        .52,velocity=vel,position=.82 if st=="slap" else .58,stroke="slap" if st=="slap" else "tone",sr=sr,seed=74)


@drum("tabla", "Physical loaded-head tabla: harmonicized membrane with na/ge/ke strokes.",
      tune=(370,"Hz"), stroke=("na","na|ge|ke"))
def tabla(sr=DEFAULT_SR, vel=1.0, tune=370, stroke="na"):
    from .physics import MembraneDrum
    st=str(stroke).lower()
    if st=="ge":
        return MembraneDrum(float(tune)*.23,.145,damping=.75,harmonic_loading=.28,air_coupling=.48).strike(
            .85,velocity=vel,position=.12,stroke="bass",sr=sr,seed=75)
    if st=="ke":
        return MembraneDrum(float(tune)*.92,.095,damping=2.8,harmonic_loading=.72,air_coupling=.08).strike(
            .18,velocity=vel,position=.88,stroke="slap",sr=sr,seed=76)
    return MembraneDrum(float(tune),.105,damping=.82,harmonic_loading=.92,air_coupling=.12).strike(
        .95,velocity=vel,position=.72,stroke="tone",sr=sr,seed=77)


@drum("guiro", "Spatial-profile physical scrape: ridge spacing is converted to sound by speed.",
      dur=(0.25,"seconds"), ridges=(28,"ridge count"))
def guiro(sr=DEFAULT_SR, vel=1.0, dur=.25, ridges=28):
    from .physics import SurfaceContact
    spacing_mm=max(.25, 80.0/max(1,int(ridges)))
    speed=.08/max(float(dur),.03)
    return SurfaceContact(.74,spacing_mm,body_frequency=1850,body_decay=42).scrape(
        float(dur)+.05,speed=speed,force=.45+.75*vel,sr=sr,seed=83)


def _fade_out(x, sr, secs):
    """Raised-cosine fade (in place) reaching zero 5 ms before the end, so render_drum's 4 ms ramp meets silence."""
    z = samples(0.005, sr)
    m = min(len(x) // 4, samples(secs, sr))
    x[len(x) - z - m:len(x) - z] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(1, m + 1) / m))
    x[len(x) - z:] = 0.0


RHO_WATER = 1000.0


def _free_bar_shape(beta, x):
    c = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) - np.sin(beta))
    bx = beta * x
    return np.cosh(bx) + np.cos(bx) - c * (np.sinh(bx) + np.sin(bx))


def immersed_ratio(n, depth, ro=0.019, ri=0.0175, rho=2700.0, filled=True):
    """Frequency factor of free-free tube mode n (1 = lowest) dipped `depth` (0..1 of its length) into water:
    Rayleigh quotient with the water's added mass per length rho_w pi ro^2 (plus pi ri^2 when the bore floods)
    weighted by the mode shape squared over the wet part: f / f_dry = 1 / sqrt(1 + (m_a / m) int_wet phi^2 / int phi^2)."""
    b = TRIANGLE_BETA[n - 1] if n <= 4 else (2 * n + 1) * np.pi / 2
    x = np.linspace(0.0, 1.0, 801)
    phi2 = _free_bar_shape(b, x) ** 2
    d = float(np.clip(depth, 0.0, 1.0))
    wet = (x >= 1.0 - d) if d > 0.0 else np.zeros(x.size, bool)
    ratio = RHO_WATER * (ro * ro + (ri * ri if filled else 0.0)) / (rho * (ro * ro - ri * ri))
    share = float(np.sum(phi2 * wet) / np.sum(phi2))
    return 1.0 / math.sqrt(1.0 + ratio * share), ratio * share


@drum("water_tube", "Tubular bell lowered into water while it rings (Baschet water chime): the deeper it goes the lower "
      "the note, enough for microtones and glides, and the shorter it rings.",
      tune=(440.0, "Hz of the dry note (mode 2 of the tube)"), depth=(0.0, "0..1 of the length under water at the hit"),
      depth_end=(0.5, "0..1 under water at the end (glide)"), dur=(3.0, "s"), decay=(3.0, "dry T60 of the note, s"))
def water_tube(sr=DEFAULT_SR, vel=1.0, tune=440.0, depth=0.0, depth_end=0.5, dur=3.0, decay=3.0):
    n = samples(float(np.clip(dur, 0.1, 20.0)), sr)
    t = np.arange(n) / sr
    d = float(depth) + (float(depth_end) - float(depth)) * np.clip(t / max(float(dur), 1e-3), 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, 21)
    y = np.zeros(n)
    f1 = float(tune) / (TRIANGLE_BETA[1] / TRIANGLE_BETA[0]) ** 2
    for k in range(1, 7):
        b = TRIANGLE_BETA[k - 1] if k <= 4 else (2 * k + 1) * np.pi / 2
        f = f1 * (b / TRIANGLE_BETA[0]) ** 2
        if f >= 0.45 * sr:
            break
        tab = np.array([immersed_ratio(k, g) for g in grid])
        fr = f * np.interp(d, grid, tab[:, 0])
        load = np.interp(d, grid, tab[:, 1])
        rate = 6.91 / float(decay) * (1.0 + load)
        y += np.sin(2 * np.pi * np.cumsum(fr) / sr) * np.exp(-np.cumsum(rate) / sr) / k
    env = np.ones(n)
    a = samples(0.002, sr)
    env[:a] = np.linspace(0.0, 1.0, a)
    _fade_out(env, sr, 0.2)
    return y * env * 0.5 * (0.5 + 0.5 * vel)


SHELL_DEPTH = 0.35      # m: path from the batter head to a mic under the drum (shell depth + mic distance)
UNDER_LP = 1500.0       # Hz: the shell and the resonant head darken what the bottom mic hears


def double_mic(y, sr, under=0.0, flip=False, depth=SHELL_DEPTH):
    """Top + bottom microphones on one drum. The two faces of the head move in opposition (a dipole), so the bottom
    mic hears the head inverted, later by depth/c and darker; `under` 0..1 is its level in the mix. `flip` inverts it
    back, as engineers do: with flip the lows add up, without it they partly cancel (thin, hollow)."""
    u = float(np.clip(under, 0.0, 1.0))
    if u <= 0.0:
        return y
    d = int(round(float(depth) / 343.0 * sr))
    b = np.zeros_like(y)
    b[d:] = F.lowpass(y, UNDER_LP, sr)[:y.size - d]
    out = y + u * (b if flip else -b)
    return out * (np.max(np.abs(y)) / max(float(np.max(np.abs(out))), 1e-12))


SHELL_PRESETS = {   # kappa 1/s, q, centre, sigma0, sigma1, strike radius
    "cymbal": (50.0, 30.0, "clamped", 1.34, 0.003, 0.5),
    "flat": (50.0, 0.0, "clamped", 1.34, 0.003, 0.5),
    "deep": (50.0, 60.0, "clamped", 1.34, 0.003, 0.5),
    "gong": (5.0, 60.0, "free", 0.3, 0.001, 0.0),
}
SHELL_FORCE = 2.0e6     # scaled force density at vel 1: peak displacement of a few thicknesses (crash regime)
CYMBAL_TAPER = (1.0, 0.7, 0.25)     # centre thickness, rim thickness (relative), radius where the taper starts
SHELL_ZONES = {"bell": 0.12, "bow": 0.5, "edge": 0.92}


def cymbal_thickness(r, hc=CYMBAL_TAPER[0], he=CYMBAL_TAPER[1], rc=CYMBAL_TAPER[2]):
    """Thickness of a tapered cymbal at relative radius r: flat `hc` over the dome (r < rc), then linear to `he`."""
    r = np.clip(np.asarray(r, float), 0.0, 1.0)
    return np.where(r < rc, hc, hc + (he - hc) * (r - rc) / (1.0 - rc))


def zone_drive(r, taper=CYMBAL_TAPER):
    """Force factor that gives a uniform plate the local amplitude-to-thickness ratio of a tapered one struck at r:
    the point mobility of a plate goes as 1/h^2, so w/h goes as (hc/h(r))^3. Thin rim: crashes sooner."""
    return float((taper[0] / cymbal_thickness(r, *taper)) ** 3)


@drum("polar_shell", "Cymbal or gong as a shallow spherical shell on a polar grid (circular free rim, clamped or free "
      "centre, von Karman nonlinearity): `q` curvature empties the bottom of the spectrum, a hard hit crashes.",
      kind=("cymbal", "cymbal | flat | deep | gong"), q=(-1.0, "curvature (scaled; -1 = the kind's)"),
      dur=(2.0, "s"), pos=(-1.0, "strike radius 0..1 (-1 = the kind's)"), fs=(32000, "internal rate Hz"),
      rings=(16, "radial grid points"),
      zone=("", "bell | bow | edge: strike zone of a tapered, domed cymbal (sets pos; the thin rim crashes sooner, "
                "the bell pings); empty = `pos` on a uniform shell"))
def polar_shell(sr=DEFAULT_SR, vel=1.0, kind="cymbal", q=-1.0, dur=2.0, pos=-1.0, fs=32000, rings=16, zone=""):
    from .physical.fdgrid import polar_model, polar_simulate
    from .plates import _finish, _vel
    kap, q0, centre, s0, s1, r0 = SHELL_PRESETS.get(str(kind), SHELL_PRESETS["cymbal"])
    qq = q0 if float(q) < 0 else float(q)
    fs = int(np.clip(fs, 8000, 96000))
    mdl = polar_model(fs, "shell", int(np.clip(rings, 6, 32)), 2 * int(np.clip(rings, 6, 32)), "free", centre, kap,
                      0.0, qq, 0.2495, s0, s1)
    n = int(float(np.clip(dur, 0.1, 10.0)) * fs)
    m = max(2, int(0.003 * fs))
    f = np.zeros(n)
    rr = r0 if float(pos) < 0 else float(np.clip(pos, 0.0, 1.0))
    g = 1.0
    if zone:
        if zone not in SHELL_ZONES:
            raise ValueError(f"zone must be one of {sorted(SHELL_ZONES)}")
        rr = SHELL_ZONES[zone]
        g = zone_drive(rr)
    f[:m] = 0.5 * g * SHELL_FORCE * float(vel) ** 2 * (1.0 - np.cos(2.0 * np.pi * np.arange(m) / m))
    out = polar_simulate(mdl, f, (rr, 0.0), ((0.83, 0.37),))
    return _finish(_vel(out, fs)[:, 0], fs, sr, 0.8 * (0.5 + 0.5 * vel))


PERRIN_MODES = (   # lower frequency Hz, doublet split Hz, main partial (ring-inextensional or R=1 family)
    (292.72, 0.0, 1), (585.92, 0.0, 1), (692.94, 0.0, 1), (882.21, 0.32, 1), (1171.4, 0.6, 1), (1472.5, 0.3, 1),
    (1525.6, 0.0, 0), (1559.2, 0.6, 0), (1761.1, 3.2, 1), (1948.9, 0.0, 0), (2040.3, 0.0, 0), (2145.6, 1.2, 1),
    (2436.9, 4.3, 1), (2485.7, 0.0, 0), (2537.6, 2.7, 0), (2545.8, 4.6, 0), (2616.3, 1.5, 0), (2716.6, 1.5, 0),
    (2828.6, 3.9, 0), (2858.2, 0.3, 0), (2920.7, 0.9, 1), (3176.0, 7.1, 1), (3376.8, 0.0, 0), (3403.0, 0.5, 0),
    (3431.3, 2.1, 0), (3603.6, 3.5, 0), (3795.6, 2.0, 1), (3863.5, 3.5, 0), (3978.4, 3.6, 1), (4125.7, 1.1, 0),
    (4340.0, 0.0, 0), (4368.0, 33.0, 0), (4539.0, 4.0, 0), (4759.4, 5.9, 1), (4813.5, 0.0, 1), (5002.5, 3.0, 0),
    (5064.5, 11.5, 0), (5295.0, 3.0, 0), (5556.0, 7.0, 0), (5674.0, 2.0, 1), (5877.0, 1.0, 0), (6002.0, 4.0, 0),
    (6101.0, 4.0, 0), (6394.0, 8.0, 0), (6555.0, 4.0, 1), (6655.0, 2.0, 0), (6789.0, 28.0, 0), (6919.0, 2.0, 0),
    (7329.0, 0.0, 0), (7373.0, 4.0, 0), (7441.0, 7.0, 1), (7499.0, 0.0, 0), (7547.0, 3.0, 0), (7805.0, 7.0, 0),
    (7870.0, 4.0, 0), (8139.0, 3.0, 0), (8213.0, 23.0, 0), (8338.0, 5.0, 1), (8348.0, 15.0, 0), (8567.0, 6.0, 0),
    (8807.0, 5.0, 0), (9029.0, 5.0, 0), (9249.0, 5.0, 1),
)
BELL_HUM_T60 = 60.0     # s: T60 of the hum of a large bell; partial T60 falls as (f / hum)^-BELL_T60_EXP
BELL_T60_EXP = 1.1
BELL_AMP_EXP = 0.5      # partial amplitude (f / hum)^-0.5; the other mode families 6 dB under the main ones


def perrin_partials(tune=PERRIN_MODES[0][0]):
    """(freq Hz, amplitude, T60 s) of every measured mode, both halves of each doublet, scaled to the hum `tune`."""
    k = float(tune) / PERRIN_MODES[0][0]
    out = []
    for f, split, main in PERRIN_MODES:
        a = (f / PERRIN_MODES[0][0]) ** -BELL_AMP_EXP * (1.0 if main else 0.5)
        t60 = BELL_HUM_T60 * (f / PERRIN_MODES[0][0]) ** -BELL_T60_EXP
        for g in ((f,) if split == 0.0 else (f, f + split)):
            out.append((g * k, a / (1 if split == 0.0 else math.sqrt(2.0)), t60))
    return out


@drum("perrin_bell", "Church bell from the measured modes of one real bell (63 modes up to 9.2 kHz, the split doublets "
      "included: slow beats on the partials); `tune` = hum note.", tune=(292.72, "Hz of the hum"), dur=(8.0, "s"))
def perrin_bell(sr=DEFAULT_SR, vel=1.0, tune=292.72, dur=8.0):
    n = samples(float(np.clip(dur, 0.2, 60.0)), sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    rng = np.random.default_rng(31)
    bright = 0.4 + 0.6 * float(vel)
    for f, a, t60 in perrin_partials(tune):
        if f >= 0.45 * sr:
            continue
        y += a * bright ** (f / 2000.0) * np.exp(-6.91 * t / t60) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    env = np.ones(n)
    a0 = samples(0.001, sr)
    env[:a0] = np.linspace(0.0, 1.0, a0)
    _fade_out(env, sr, 0.3)
    return y * env * (0.8 / max(float(np.max(np.abs(y))), 1e-12)) * (0.5 + 0.5 * vel)


def bar_end_impact(speed, radius=0.005, ball=(7850.0, 2.0e11, 0.3), bar_area=math.pi * 0.01 ** 2,
                   bar=(7850.0, 2.0e11, 0.3), dt=1e-8, tmax=2e-3):
    """Sphere hitting the end of a long bar along its axis: Hertz contact in series with the bar end, which takes
    the force as a dashpot of impedance A rho c0 (a wave runs away and nothing comes back). Returns (t, force)
    over the contact; `speed` m/s, `ball` and `bar` = (rho, E, nu)."""
    rb, eb, nb = ball
    rr, er, nr = bar
    m = rb * 4.0 / 3.0 * math.pi * radius ** 3
    estar = 1.0 / ((1.0 - nb * nb) / eb + (1.0 - nr * nr) / er)
    kh = 4.0 / 3.0 * estar * math.sqrt(radius)
    z = bar_area * rr * math.sqrt(er / rr)
    xs, vs, ub = 0.0, float(speed), 0.0
    t, f = [], []
    for i in range(int(tmax / dt)):
        p = kh * max(xs - ub, 0.0) ** 1.5
        if i > 0 and p <= 0.0 and xs < ub:
            break
        vs -= p / m * dt
        xs += vs * dt
        ub += p / z * dt
        t.append(i * dt)
        f.append(p)
    return np.array(t), np.array(f)
