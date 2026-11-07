"""Electronic instruments: leads, plucks, pads, string machine and the basses of dance music.

Registered into the instrument REGISTRY (imported at the bottom of instruments.py). Filters are set
in Hz with a ceiling, not in harmonics, so these stay smooth in the top octaves (see the voicing
rules at the top of instruments.py).
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy import signal as _sig

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR
from .env import adsr, apply
from .fx import _per_channel, effect
from .instruments import _cut, _fm_cap, _harmonics, _n, _off, _roll, _spectrum, _vibrato, decay_env, instrument

FM_MEX_ATTACK = 0.002
FM_MEX_DECAY = 0.8
DRIFT_HZ = 0.5
ELLIP_SPEC = (8, 0.2, 90.0)
PACKET_GAINS = (0.5, 0.333, 0.25, 0.2, 0.1666)
PACKET_OUT = 0.4


def analog_drift(n: int, sr: int, cents: float, seed: int) -> np.ndarray:
    """Slow analog oscillator wander: a frequency multiplier whose log deviates by low-passed noise
    (second order, 0.5 Hz) of rms ``cents``."""
    if cents <= 0:
        return np.ones(n)
    w = np.random.default_rng(seed).standard_normal(n + sr)
    w = _sig.sosfilt(_sig.butter(2, DRIFT_HZ, fs=sr, output="sos"), w)[sr:]
    w /= np.sqrt(np.mean(w * w)) + 1e-12
    return 2.0 ** (cents * w / 1200.0)


def _drifted(f, n, sr, cents, freq):
    return f * analog_drift(n, sr, float(cents), int(freq * 16) + 7) if cents else f


@numba.njit(cache=True)
def _fb_fm(fc, fm, idx, fb, sr):
    """Two-operator FM with self-feedback on both: m = sin(phi_m + fb m[n-1]),
    y = sin(phi_c + idx m + fb y[n-1])."""
    n = len(fc)
    y = np.zeros(n)
    pc = 0.0
    pm = 0.0
    m1 = 0.0
    y1 = 0.0
    for i in range(n):
        m1 = math.sin(2.0 * math.pi * pm + fb * m1)
        y1 = math.sin(2.0 * math.pi * pc + idx[i] * m1 + fb * y1)
        y[i] = y1
        pc += fc[i] / sr
        pm += fm[i] / sr
        pc -= math.floor(pc)
        pm -= math.floor(pm)
    return y


def va_ladder_zpk(fc: float, sr: int, k: float, kind: str = "lp", poles: int = 4):
    """Linear zero-delay-feedback ladders and the Sallen-Key (Korg35) low-pass as exact digital filters
    (bilinear transform, cutoff prewarped), returned as SOS. One-pole stages G = 1/(1 + s) (lp) or s/(1 + s) (hp):
    lp: H = 1/((1 + s)^N + k), resonance at tan(pi/N) wc, oscillates at k = 1/cos(pi/N)^N (4 for N 4, 1.884 for N 8);
    hp: H = s^N/((1 + s)^N + k s^N), stable for k > -1; bp: alternating LP-HP stages with positive feedback,
    H = s^(N/2)/((1 + s)^N - k s^(N/2)), gain 2^(-N/2) at wc, oscillates at k = 2^(N/2) (4 for N 4, 16 for N 8), and for k < 0
    with N 8 two peaks at (sqrt 2 -+ 1) wc; korg35: H = 1/(s^2 + (2 - k) s + 1), oscillates at k = 2."""
    N = int(poles)
    if N not in (4, 8) and kind != "korg35":
        raise ValueError("poles must be 4 or 8")
    base = np.poly1d([1.0, 1.0]) ** N
    if kind == "lp":
        den, z, g = base + np.poly1d([k]), np.array([]), 1.0
    elif kind == "hp":
        den, z = base + np.poly1d([k] + [0.0] * N), np.zeros(N)
        g = 1.0 / den.coeffs[0]
    elif kind == "bp":
        den, z, g = base - np.poly1d([k] + [0.0] * (N // 2)), np.zeros(N // 2), 1.0
    elif kind == "korg35":
        den, z, g = np.poly1d([1.0, 2.0 - k, 1.0]), np.array([]), 1.0
    else:
        raise ValueError("kind must be lp, hp, bp or korg35")
    p = np.roots(den.coeffs / den.coeffs[0])
    wa = 2.0 * sr * math.tan(math.pi * min(fc, 0.45 * sr) / sr)
    zd, pd, gd = _sig.bilinear_zpk(z * wa, p * wa, g * wa ** (len(p) - len(z)), sr)
    return _sig.zpk2sos(zd, pd, gd)


@effect("va_ladder", "Virtual-analog filters as exact linear models: 4- and 8-pole low-, high- and band-pass ladders, "
        "the Korg35 (MS-20) low-pass, and a steep elliptic low-pass.",
        kind=("lp", "lp | hp | bp (LP-HP stages, positive feedback; bp 8 with k < 0 = two peaks) | korg35 | ellip"),
        poles=(4, "4 | 8 (lp, hp, bp)"), cutoff=(1000.0, "Hz"),
        k=(0.0, "feedback: lp 0..4 (8 poles 0..1.88), hp > -1, bp 0..2^(N/2) (4 or 16; 8 poles also down to -64), korg35 0..2; ellip ignores it"))
def va_ladder(x, sr=DEFAULT_SR, kind="lp", poles=4, cutoff=1000.0, k=0.0):
    if kind == "ellip":
        sos = _sig.ellip(ELLIP_SPEC[0], ELLIP_SPEC[1], ELLIP_SPEC[2], min(float(cutoff), 0.45 * sr), fs=sr, output="sos")
    else:
        sos = va_ladder_zpk(float(cutoff), sr, float(k), kind, int(poles))
    return _per_channel(lambda c: _sig.sosfilt(sos, c), np.asarray(x, float))


# ---------------------------------------------------------------- leads
@instrument("soft_lead", "Soft triangle lead with delayed vibrato: a clear melody voice that never bites (pop, chill, game themes).",
            family="synth", span=("C3", "C7"), vibrato=(0.15, "semitones"), cutoff=(2400, "Hz"),
            drift=(0.0, "cents rms of slow analog pitch wander (0 = none; 3-8 audible)"))
def soft_lead(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.15, cutoff=2400, drift=0.0):
    rel = 0.15
    n = _n(dur, sr, rel)
    f = _drifted(_vibrato(freq, n, sr, rate=5.5, depth=vibrato, delay=0.18, ramp=0.25), n, sr, drift, freq)
    x = O.triangle(f, n, sr) * 0.8 + O.sine(f, n, sr) * 0.4
    x = F.lowpass(x, max(cutoff, min(freq * 1.5, 4000.0)), sr, 0.8)
    return apply(x, adsr(dur, 0.012, 0.15, 0.8, rel, sr)) * 0.55 * (0.5 + 0.5 * vel)


@instrument("square_lead", "Rounded square lead whose filter closes after the attack (classic game melody, synth-pop).",
            family="synth", span=("C3", "C6"), vibrato=(0.1, "semitones"), width=(0.5, "pulse width 0.1..0.5"),
            drift=(0.0, "cents rms of slow analog pitch wander (0 = none; 3-8 audible)"))
def square_lead(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.1, width=0.5, drift=0.0):
    rel = 0.12
    n = _n(dur, sr, rel)
    f = _drifted(_vibrato(freq, n, sr, rate=5.5, depth=vibrato, delay=0.2, ramp=0.25), n, sr, drift, freq)
    t = np.arange(n) / sr
    fc = np.clip(freq * (2.5 + 3.5 * (0.4 + 0.6 * vel) * np.exp(-t / 0.12)), 900.0, 3600.0)
    x = F.biquad(O.pulse(f, n, sr, width=width) - (2 * width - 1), "lowpass", fc, sr, q=1.0)
    return apply(x, adsr(dur, 0.004, 0.2, 0.7, rel, sr)) * 0.45 * (0.5 + 0.5 * vel)


@instrument("synth_pluck", "Synth pluck: saw + pulse through a fast-closing filter (trance and synthwave arps, pop hooks).",
            family="synth", span=("C2", "C6"), decay=(0.3, "seconds"), bright=(1.0, "0.5 soft .. 2 sharp"))
def synth_pluck(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=0.3, bright=1.0):
    ring = dur + decay * 2.5
    n = _n(ring, sr)
    x = O.saw(freq, n, sr) * 0.6 + O.pulse(freq * 1.003, n, sr, width=0.5) * 0.4
    pk = _cut(freq, 9, 1800, 3600) * (0.5 + 0.5 * vel) * bright
    fc = np.minimum(max(freq * 2, 300.0) + pk * decay_env(n, decay * 0.35, sr), max(4200.0, freq * 1.5))
    x = F.biquad(x, "lowpass", fc, sr, q=1.4)
    env = (0.25 + 0.75 * decay_env(n, decay, sr)) * _off(n, dur, sr, decay * 0.5)
    env[:max(2, int(0.002 * sr))] *= np.linspace(0, 1, max(2, int(0.002 * sr)))
    return x * env * 0.5 * (0.5 + 0.5 * vel)


@instrument("theremin", "Theremin: eerie sliding sine with wide vibrato (sci-fi, spooky, retro B-movie).",
            family="synth", span=("C3", "C7"), vibrato=(0.25, "semitones"))
def theremin(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.25):
    rel = 0.2
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.2, depth=vibrato, delay=0.1, ramp=0.4, scoop=1.0)
    amps = np.array([1.0, 0.28, 0.1, 0.04]) * _roll(np.arange(1, 5) * freq, 2200.0) / float(_roll(freq, 2200.0))
    return apply(_harmonics(f, n, sr, amps), adsr(dur, 0.09, 0.1, 0.9, rel, sr)) * 0.6 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- pads / ensembles
@instrument("warm_pad", "Warm analog pad: dark, slow, breathing filter (ambient beds, ballads, lo-fi).",
            family="pad", span=("C2", "C5"), attack=(0.6, "s"), release=(1.2, "s"), cutoff=(900, "Hz at rest; opens ~1.6x mid-note"),
            drift=(0.0, "cents rms of slow analog pitch wander per oscillator (0 = none; 3-8 audible)"))
def warm_pad(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.6, release=1.2, cutoff=900, drift=0.0):
    n = _n(dur, sr, release)
    if drift:
        da = _drifted(np.ones(n), n, sr, drift, freq)
        db = _drifted(np.ones(n), n, sr, drift, freq + 1.0)
        x = O.saw(freq * 0.997 * da, n, sr) + O.saw(freq * 1.003 * db, n, sr) + 0.8 * O.triangle(freq * 0.5 * da, n, sr)
    else:
        x = O.saw(freq * 0.997, n, sr) + O.saw(freq * 1.003, n, sr) + 0.8 * O.triangle(freq * 0.5, n, sr)
    t = np.arange(n) / n
    base = max(cutoff, min(freq * 1.2, 2500.0))
    fc = base * (1 + 0.6 * np.sin(np.pi * t) * (0.5 + 0.5 * vel))
    x = F.biquad(x, "lowpass", fc, sr, q=0.7, order=2)
    return apply(x, adsr(dur, attack, 0.4, 0.85, release, sr)) * 0.3 * (0.6 + 0.4 * vel)


@instrument("glass_pad", "Glass pad: airy additive shimmer, each partial drifting on its own (new age, ice, space, dreamy menus).",
            family="pad", span=("C3", "C6"), attack=(0.7, "s"), release=(1.5, "s"))
def glass_pad(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.7, release=1.5):
    n = _n(dur, sr, release)
    t = np.arange(n) / sr
    rng = np.random.default_rng(int(freq) % 991 + 5)
    y = np.zeros(n)
    for h, g in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.3), (6, 0.12)):
        fh = freq * h
        if fh > 6000:
            continue
        a = g * float(_roll(fh, 2600.0)) / float(_roll(freq, 2600.0))
        drift = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.12, 0.4) * t + rng.uniform(0, 6.28))
        pair = np.sin(2 * np.pi * fh * 1.0012 * t) + np.sin(2 * np.pi * fh * 0.9988 * t + rng.uniform(0, 6.28))
        y += a * drift * pair * 0.5
    return apply(y, adsr(dur, attack, 0.3, 0.9, release, sr)) * 0.45 * (0.6 + 0.4 * vel)


@instrument("string_machine", "70s string machine (Solina): thin saws through a swirling ensemble chorus (disco, synthwave, prog, shoegaze).",
            family="pad", span=("C2", "C6"), attack=(0.09, "s"), release=(0.45, "s"))
def string_machine(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.09, release=0.45):
    n = _n(dur, sr, release)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for i in range(3):   # three chorus lines, 120 degrees apart, slow + fast LFO like the BBD ensemble
        ph = 2 * np.pi * i / 3
        f = freq * (1 + 0.0035 * np.sin(2 * np.pi * 0.63 * t + ph) + 0.0012 * np.sin(2 * np.pi * 6.1 * t + ph))
        x += O.saw(f, n, sr, phase0=i / 3) + 0.45 * O.saw(f * 2, n, sr, phase0=i / 5)
    x = F.lowpass(F.highpass(x / 3, 120, sr), max(2600.0, min(freq * 2, 3600.0)), sr, 0.7, order=2)
    return apply(x, adsr(dur, attack, 0.2, 0.9, release, sr)) * 0.45 * (0.6 + 0.4 * vel)


# ---------------------------------------------------------------- basses
_SAW9 = (1.0, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.12, 0.11)


@instrument("acid", "Acid bass line (303): saw or square through a squelchy resonant filter; high `vel` = accent (acid house, techno).",
            family="bass", span=("C1", "C4"), wave=("saw", "saw|square"), cutoff=(350, "Hz at rest"), env=(2000, "Hz the filter jumps by"),
            decay=(0.2, "s filter decay"), res=(0.7, "0..1 resonance (phase_sync: 0..4, above 1 saturates)"),
            model=("svf", "svf | diode: the TB-303 diode ladder | phase_sync: no filter, a sine at the cutoff restarted and "
                          "faded on every cycle rides on a 9-harmonic saw (Farnell's fake 303)"))
def acid(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="saw", cutoff=350, env=2000, decay=0.2, res=0.7, model="svf"):
    rel = 0.03
    n = _n(dur, sr, rel)
    fc = np.minimum(cutoff + env * vel ** 1.5 * decay_env(n, decay, sr), 2800.0)
    if model == "phase_sync":
        ph = (np.arange(n) * float(freq) / sr) % 1.0
        saw = sum(a * np.sin(2 * np.pi * (k + 1) * ph) for k, a in enumerate(_SAW9) if (k + 1) * freq < 0.45 * sr)
        ring = np.cos(2 * np.pi * ((ph * fc / float(freq)) % 1.0)) * np.minimum(float(np.clip(res, 0, 4)) * (1.0 - ph) ** 2, 1.0)
        x = F.dc_block(F.lowpass(np.tanh(1.5 * 0.5 * (saw + ring)) / 0.9, 4000, sr), sr)
        return apply(x, adsr(dur, 0.003, 0.15, 0.75, rel, sr)) * 0.4 * (0.5 + 0.5 * vel)
    x = O.square(freq, n, sr) if str(wave).startswith("sq") else O.saw(freq, n, sr)
    if model == "diode":                                     # k = 17 res loses (1 + k) of the level: give it back
        x = F.ladder(x, fc * 1.414, sr, res, "diode") * (1.0 + 17.0 * float(np.clip(res, 0, 1))) ** 0.5
    else:
        x = F.biquad(x, "lowpass", fc, sr, q=1.0 + 8.0 * float(np.clip(res, 0, 1)))
    x = F.dc_block(F.lowpass(np.tanh(1.5 * x) / 0.9, 4000, sr), sr)
    return apply(x, adsr(dur, 0.003, 0.15, 0.75, rel, sr)) * 0.4 * (0.5 + 0.5 * vel)


@instrument("reese", "Reese bass: detuned saws beating against each other under a dark filter (drum and bass, dubstep, dark techno).",
            family="bass", span=("C1", "C3"), cutoff=(600, "Hz"), detune=(0.18, "semitones"),
            drift=(0.0, "cents rms of slow analog pitch wander (0 = none; 3-8 audible)"))
def reese(freq, dur, sr=DEFAULT_SR, vel=1.0, cutoff=600, detune=0.18, drift=0.0):
    rel = 0.12
    n = _n(dur, sr, rel)
    d = 2 ** (detune / 12)
    f = _drifted(np.full(n, float(freq)), n, sr, drift, freq) if drift else freq
    x = O.saw(f, n, sr) + O.saw(f * d, n, sr, phase0=0.3) + O.saw(f / d, n, sr, phase0=0.7) + 0.8 * O.sine(f * 0.5, n, sr)
    t = np.arange(n) / sr
    fc = cutoff * (1 + 0.35 * np.sin(2 * np.pi * 0.35 * t))
    x = F.biquad(x / 2.5, "lowpass", fc, sr, q=1.2, order=2)
    return apply(np.tanh(1.4 * x), adsr(dur, 0.02, 0.0, 1.0, rel, sr)) * 0.55 * (0.5 + 0.5 * vel)


@instrument("fm_bass", "FM bass (DX-style): tight, rubbery, punchy attack (house, city pop, funk, 80s and 90s game music).",
            family="bass", span=("C1", "C4"), bite=(1.0, "0.3 round .. 2 hard"),
            ratio=(1.0, "modulator : carrier (1 round, 2 hollow and reedy, 3 nasal, 0.5 sub growl, 1.333 metallic)"),
            model=("classic", "classic | mex (one carrier, one modulator, index and level both falling as the square "
                              "of a ramp lasting 0.8 s x index^2: the longer the brighter)"),
            index=(0.6, "mex: modulation index in cycles at the attack (0.2 soft .. 0.8 hard), also the note length"),
            fb=(0.0, "classic: self-feedback of both operators in radians (0 = none, 1 = gritty, 1.5 = near noise)"))
def fm_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, bite=1.0, ratio=1.0, model="classic", index=0.6, fb=0.0):
    if model not in ("classic", "mex"):
        raise ValueError("fm_bass model must be classic or mex")
    r = float(np.clip(ratio, 0.1, 8.0))
    if model == "mex":
        I = float(np.clip(index, 0.05, 1.5))
        ring = max(float(dur), FM_MEX_DECAY * I * I) + FM_MEX_ATTACK
        n = _n(ring, sr)
        t = np.arange(n) / sr
        e = np.clip(np.minimum(t / FM_MEX_ATTACK, 1.0 - (t - FM_MEX_ATTACK) / (FM_MEX_DECAY * I * I)), 0.0, 1.0)
        e = e * e * _off(n, dur, sr, 0.03)
        x = np.cos(2 * np.pi * freq * t + 2 * np.pi * I * e * np.sin(2 * np.pi * freq * r * t)) * e
        return F.lowpass(x, 2500, sr) * 0.5 * (0.5 + 0.5 * vel)
    rel = 0.08
    n = _n(dur, sr, rel)
    idx = min((2.0 + 5.0 * vel) * bite, _fm_cap(freq, r, 2600.0) / 0.6) * decay_env(n, 0.11, sr) + 0.6
    if fb:
        fc = np.full(n, float(freq))
        x = _fb_fm(fc, fc * r, np.asarray(idx, float), float(np.clip(fb, 0.0, 2.0)), float(sr)) + 0.5 * O.sine(freq, n, sr)
    else:
        x = O.fm(freq, n, sr, ratio=r, index=idx) + 0.5 * O.sine(freq, n, sr)
    x = F.lowpass(x, 2500, sr)
    return apply(x, adsr(dur, 0.003, 0.2, 0.6, rel, sr)) * 0.5 * (0.5 + 0.5 * vel)


@instrument("bass808", "808 bass: a long, pitched kick-drum sub with a little drive so it reads on small speakers (trap, hip-hop, drill).",
            family="bass", span=("C1", "C3"), decay=(1.5, "seconds"), drive=(0.3, "0 clean sine .. 1 fuzzy"), punch=(0.5, "0..1 pitch drop at the onset"))
def bass808(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=1.5, drive=0.3, punch=0.5):
    ring = min(max(dur, 0.05) + 0.25, decay * 2.5)
    n = _n(ring, sr)
    f = freq * 2 ** (7.0 * punch * decay_env(n, 0.03, sr) / 12)
    g = 1 + 5 * drive
    x = np.tanh(O.sine(f, n, sr) * g) / np.tanh(g)
    x = F.lowpass(x, 1200, sr)
    env = decay_env(n, decay / 2.5, sr) * _off(n, dur, sr, 0.06)
    env[:max(2, int(0.002 * sr))] *= np.linspace(0, 1, max(2, int(0.002 * sr)))
    return x * env * 0.8 * (0.5 + 0.5 * vel)


def packet_gains(cutoff: float) -> np.ndarray:
    """Harmonic k = 1..5 gains of the Chebyshev saw (1/2, 1/3, 1/4, 1/5, 1/6): the fundamental stays, harmonic k
    is scaled by (2 cutoff)^(2^(k-2)) and clipped at its saw value."""
    a = 2.0 * float(np.clip(cutoff, 0.0, 1.0))
    s = np.array(PACKET_GAINS)
    return np.array([s[0]] + [min(s[k], s[k] * a ** (2 ** (k - 1))) for k in range(1, 5)])


@instrument("packet_pad", "Chebyshev packet pad: a five-harmonic saw whose upper harmonics close with `cutoff`, "
            "plus a windowed resonance packet restarted every cycle (rjlib s_spad).",
            family="pad", span=("C2", "C6"), cutoff=(0.5, "0 fundamental only .. 0.5 full saw; also sets the packet pitch"),
            res=(0.3, "0..1 packet resonance level"), attack=(0.3, "s"), release=(0.8, "s"))
def packet_pad(freq, dur, sr=DEFAULT_SR, vel=1.0, cutoff=0.5, res=0.3, attack=0.3, release=0.8):
    n = _n(dur, sr, release)
    ph = O.phase(freq, n, sr) % 1.0
    c = float(np.clip(cutoff, 0.0, 1.0))
    y = sum(g * np.cos(2 * np.pi * k * ph) for k, g in enumerate(packet_gains(c), 1) if k * freq < 0.45 * sr)
    m = 1.0 + 10.0 * c
    if m * freq < 0.45 * sr:
        y = y + float(np.clip(res, 0.0, 1.0)) * np.sin(2 * np.pi * ((ph * m) % 1.0)) * (1.0 - ph)
    return apply(y * PACKET_OUT, adsr(dur, attack, 0.3, 0.9, release, sr)) * (0.6 + 0.4 * vel)


@instrument("padsynth", "PADsynth: every harmonic widened into a band of random phases, one huge loop with a natural chorus (Nasca, ZynAddSubFX).",
            family="pad", span=("C2", "C6"), bw=(40.0, "cents: width of the first harmonic, the higher ones widen in proportion"),
            tilt=(1.0, "harmonic roll-off k^-tilt"), attack=(0.5, "s"), release=(1.2, "s"), seed=(0, "a different rendition"))
def padsynth(freq, dur, sr=DEFAULT_SR, vel=1.0, bw=40.0, tilt=1.0, attack=0.5, release=1.2, seed=0):
    """Nasca's algorithm as in Soundpipe padsynth.c: bw_Hz = (2^(bw/1200) - 1) f k; Gaussian profile cut at x^2 > 14.71;
    random phase per bin; IFFT of a 2^18 table, which loops by construction and has no alias."""
    N = 2 ** 18
    A = _spectrum(freq, sr, tilt=float(tilt), corner=2600.0)
    df = sr / N
    mag = np.zeros(N // 2 + 1)
    for k, a in enumerate(A, 1):
        half = max((2.0 ** (float(bw) / 1200.0) - 1.0) * freq * k / 2.0, df)   # x = (f - fk) / half
        lo, hi = int((freq * k - 3.84 * half) / df), int((freq * k + 3.84 * half) / df) + 2   # 3.84^2 = 14.7
        i = np.arange(max(lo, 1), min(hi, N // 2))
        mag[i] += a * np.exp(-(((i * df - freq * k) / half) ** 2)) * df / half
    rng = np.random.default_rng(int(seed) + int(freq * 8))
    table = np.fft.irfft(mag * np.exp(2j * np.pi * rng.random(mag.size)), N)
    table /= np.max(np.abs(table)) * 1.4142 + 1e-30
    n = _n(dur, sr, release)
    return apply(table[(np.arange(n) + rng.integers(N)) % N], adsr(dur, attack, 0.3, 0.9, release, sr)) * (0.6 + 0.4 * vel)
