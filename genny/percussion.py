"""World, Latin, jazz and orchestral percussion, registered into the drum REGISTRY
(imported at the bottom of drums.py), so each is used like any other drum:
{"type": "drum", "kind": "conga", "params": {"slap": 1}} or in a "pattern" layer.

Hand drums are a few membrane modes with a small pitch drop plus a shaped noise slap; metal
percussion is a handful of inharmonic partials kept below ~5 kHz so it rings without stabbing.
"""
from __future__ import annotations

import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, samples
from .drums import _dec, drum
from .env import perc


def _head(n, sr, f0, modes, bend=0.06, bend_tau=0.02):
    """Drum-head modes [(ratio, amp, tau), ...] with a pitch drop of `bend` (fraction) at the onset."""
    t = np.arange(n) / sr
    ph = np.cumsum(f0 * (1 + bend * np.exp(-t / bend_tau)) / sr)
    y = np.zeros(n)
    for r, a, tau in modes:
        y += a * np.sin(2 * np.pi * r * ph) * np.exp(-t / tau)
    return y


def _slap(n, sr, center, q, tau, seed):
    return F.bandpass(O.white(n, seed=seed), center, sr, q) * _dec(n, tau, sr)


def _hand_drum(sr, vel, tune, slap, ring, seed):
    n = samples(ring * 4 + 0.05, sr)
    slap = float(np.clip(slap, 0.0, 1.0))
    tone = _head(n, sr, tune, [(1.0, 1.0, ring * (1 - 0.7 * slap)), (1.59, 0.3, ring * 0.4), (2.14, 0.15, ring * 0.25)])
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
        y = _head(n, sr, tune * 0.24, [(1.0, 1.0, 0.16), (1.6, 0.25, 0.06)], bend=0.25, bend_tau=0.03)
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
    y = _head(n, sr, tune, [(1.0, 0.6, 0.22), (2.0, 1.0, 0.28), (3.0, 0.5, 0.18), (4.0, 0.18, 0.08)], bend=0.01)
    y += _slap(n, sr, 2200, 1.5, 0.003, 77) * 0.4
    return np.tanh(y * 1.1) * perc(0.9, 0.0005, sr, curve=1.0) * 0.8 * (0.5 + 0.5 * vel)


@drum("cajon", "Cajon (box drum): `stroke` bass (boxy thump) or snare (wooden slap with wire buzz); an acoustic drum kit in one box.",
      stroke=("bass", "bass|snare"))
def cajon(sr=DEFAULT_SR, vel=1.0, stroke="bass"):
    n = samples(0.4, sr)
    if stroke == "snare":
        y = _head(n, sr, 210, [(1.0, 0.8, 0.05), (1.7, 0.4, 0.03)], bend=0.1)
        y += _slap(n, sr, 2400, 0.7, 0.045, 78) * 0.9 + _slap(n, sr, 900, 1.0, 0.01, 79) * 0.5
    else:
        y = _head(n, sr, 78, [(1.0, 1.0, 0.11), (2.3, 0.3, 0.04)], bend=0.3, bend_tau=0.02)
        y += _slap(n, sr, 500, 0.8, 0.008, 80) * 0.5
    return np.tanh(y * 1.5) * perc(0.4, 0.0005, sr, curve=1.0) * (0.5 + 0.5 * vel)


@drum("timbale", "Timbale: ringing metal-shell drum (salsa fills, mambo).", tune=(300, "Hz"))
def timbale(sr=DEFAULT_SR, vel=1.0, tune=300):
    n = samples(0.7, sr)
    y = _head(n, sr, tune, [(1.0, 1.0, 0.16), (1.59, 0.5, 0.11), (2.14, 0.3, 0.09), (2.65, 0.2, 0.07), (4.1, 0.12, 0.09)], bend=0.04)
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


@drum("triangle", "Triangle: a soft silvery ting; `open` 0 gives the damped, short stroke.", open=(1.0, "0 = damped .. 1 = ringing"), tune=(2100, "Hz"))
def triangle(sr=DEFAULT_SR, vel=1.0, open=1.0, tune=2100):
    ring = 0.08 + 1.1 * float(np.clip(open, 0, 1))
    n = samples(ring, sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for r, a, k in ((1.0, 1.0, 1.0), (1.52, 0.5, 0.8), (2.21, 0.35, 0.6), (2.9, 0.12, 0.35)):
        y += a * np.sin(2 * np.pi * tune * r * t) * np.exp(-t / (ring * k / 3))
    return y * perc(ring, 0.0005, sr, curve=1.0) * 0.25 * (0.5 + 0.5 * vel)


@drum("gong", "Gong / tam-tam: a low strike whose shimmer swells after the hit and takes seconds to die (drama, Asia, endings).",
      tune=(90, "Hz"), decay=(6.0, "seconds"))
def gong(sr=DEFAULT_SR, vel=1.0, tune=90, decay=6.0):
    n = samples(decay, sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    ratios = (1.0, 1.48, 2.11, 2.63, 3.17, 3.9, 4.72, 5.6, 6.8, 8.1, 9.7, 11.3, 13.6, 16.2)
    for k, r in enumerate(ratios):
        bloom = 1 - np.exp(-t / (0.01 + 0.05 * k))            # the high modes arrive late: energy creeps upward
        y += (1.0 / (1 + 0.35 * k)) * bloom * np.exp(-t / (decay / (2.5 + 0.5 * k))) * np.sin(2 * np.pi * tune * r * t + k)
    wash = F.bandpass(O.white(n, seed=84), 2400, sr, 0.7) * (1 - np.exp(-t / 0.4)) * np.exp(-t / (decay / 5)) * 0.08
    thud = O.sine(tune * 0.5, n, sr) * _dec(n, 0.15, sr) * 0.6
    return np.tanh((y / 2.5 + wash + thud) * 1.2) * perc(decay, 0.004, sr, curve=1.0) * (0.5 + 0.5 * vel)


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


@drum("snare808", "808 snare: two short tones under a soft noise burst (electro, hip-hop, synth-pop).", tune=(180, "Hz"), snappy=(0.6, "noise amount"))
def snare808(sr=DEFAULT_SR, vel=1.0, tune=180, snappy=0.6):
    n = samples(0.3, sr)
    tone = O.sine(tune, n, sr) * _dec(n, 0.06, sr) + 0.6 * O.sine(tune * 1.83, n, sr) * _dec(n, 0.04, sr)
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

# v0.6 physical migration: gong --------------------------------------------------
@drum("gong", "Physical modal gong/tam-tam with inharmonic resonances.", tune=(90, "Hz"), decay=(6.0, "seconds"))
def gong(sr=DEFAULT_SR, vel=1.0, tune=90, decay=6.0):
    from .physics import TunedPercussion
    y = TunedPercussion("gong", decay=decay, hardness=.48).note(float(tune), float(decay), velocity=vel, sr=sr, seed=84)
    return np.tanh(y*1.15) * .8

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
