"""Electronic instruments: leads, plucks, pads, string machine and the basses of dance music.

Registered into the instrument REGISTRY (imported at the bottom of instruments.py). Filters are set
in Hz with a ceiling, not in harmonics, so these stay smooth in the top octaves (see the voicing
rules at the top of instruments.py).
"""
from __future__ import annotations

import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR
from .env import adsr, apply
from .instruments import _cut, _fm_cap, _harmonics, _n, _off, _roll, _vibrato, decay_env, instrument


# ---------------------------------------------------------------- leads
@instrument("soft_lead", "Soft triangle lead with delayed vibrato: a clear melody voice that never bites (pop, chill, game themes).",
            family="synth", span=("C3", "C7"), vibrato=(0.15, "semitones"), cutoff=(2400, "Hz"))
def soft_lead(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.15, cutoff=2400):
    rel = 0.15
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.5, depth=vibrato, delay=0.18, ramp=0.25)
    x = O.triangle(f, n, sr) * 0.8 + O.sine(f, n, sr) * 0.4
    x = F.lowpass(x, max(cutoff, min(freq * 1.5, 4000.0)), sr, 0.8)
    return apply(x, adsr(dur, 0.012, 0.15, 0.8, rel, sr)) * 0.55 * (0.5 + 0.5 * vel)


@instrument("square_lead", "Rounded square lead whose filter closes after the attack (classic game melody, synth-pop).",
            family="synth", span=("C3", "C6"), vibrato=(0.1, "semitones"), width=(0.5, "pulse width 0.1..0.5"))
def square_lead(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.1, width=0.5):
    rel = 0.12
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.5, depth=vibrato, delay=0.2, ramp=0.25)
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
            family="pad", span=("C2", "C5"), attack=(0.6, "s"), release=(1.2, "s"), cutoff=(900, "Hz at rest; opens ~1.6x mid-note"))
def warm_pad(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.6, release=1.2, cutoff=900):
    n = _n(dur, sr, release)
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
@instrument("acid", "Acid bass line (303): saw or square through a squelchy resonant filter; high `vel` = accent (acid house, techno).",
            family="bass", span=("C1", "C4"), wave=("saw", "saw|square"), cutoff=(350, "Hz at rest"), env=(2000, "Hz the filter jumps by"),
            decay=(0.2, "s filter decay"), res=(0.7, "0..1 resonance"))
def acid(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="saw", cutoff=350, env=2000, decay=0.2, res=0.7):
    rel = 0.03
    n = _n(dur, sr, rel)
    x = O.square(freq, n, sr) if str(wave).startswith("sq") else O.saw(freq, n, sr)
    fc = np.minimum(cutoff + env * vel ** 1.5 * decay_env(n, decay, sr), 2800.0)
    x = F.biquad(x, "lowpass", fc, sr, q=1.0 + 8.0 * float(np.clip(res, 0, 1)))
    x = F.dc_block(F.lowpass(np.tanh(1.5 * x) / 0.9, 4000, sr), sr)
    return apply(x, adsr(dur, 0.003, 0.15, 0.75, rel, sr)) * 0.4 * (0.5 + 0.5 * vel)


@instrument("reese", "Reese bass: detuned saws beating against each other under a dark filter (drum and bass, dubstep, dark techno).",
            family="bass", span=("C1", "C3"), cutoff=(600, "Hz"), detune=(0.18, "semitones"))
def reese(freq, dur, sr=DEFAULT_SR, vel=1.0, cutoff=600, detune=0.18):
    rel = 0.12
    n = _n(dur, sr, rel)
    d = 2 ** (detune / 12)
    x = O.saw(freq, n, sr) + O.saw(freq * d, n, sr, phase0=0.3) + O.saw(freq / d, n, sr, phase0=0.7) + 0.8 * O.sine(freq * 0.5, n, sr)
    t = np.arange(n) / sr
    fc = cutoff * (1 + 0.35 * np.sin(2 * np.pi * 0.35 * t))
    x = F.biquad(x / 2.5, "lowpass", fc, sr, q=1.2, order=2)
    return apply(np.tanh(1.4 * x), adsr(dur, 0.02, 0.0, 1.0, rel, sr)) * 0.55 * (0.5 + 0.5 * vel)


@instrument("fm_bass", "FM bass (DX-style): tight, rubbery, punchy attack (house, city pop, funk, 80s and 90s game music).",
            family="bass", span=("C1", "C4"), bite=(1.0, "0.3 round .. 2 hard"))
def fm_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, bite=1.0):
    rel = 0.08
    n = _n(dur, sr, rel)
    idx = min((2.0 + 5.0 * vel) * bite, _fm_cap(freq, 1.0, 2600.0) / 0.6) * decay_env(n, 0.11, sr) + 0.6
    x = O.fm(freq, n, sr, ratio=1.0, index=idx) + 0.5 * O.sine(freq, n, sr)
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
