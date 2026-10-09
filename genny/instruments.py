"""Synthesized instruments. Each takes (freq_hz, dur_seconds, sr, vel, **params) and returns a mono buffer.

The returned buffer may be longer than `dur` because the release tail is appended.

Voicing rules (why nothing here is piercing):
  * Brightness follows the keyboard down, not up. Upper partials are rolled off by their absolute
    frequency (`_roll`), FM indexes are capped by the note (`_fm_cap`) and filters key-track with a
    ceiling (`_cut`), so a voice gets purer as it climbs, like a real instrument, instead of
    carrying its low-register spectrum up into the 3-8 kHz range where the ear is most sensitive.
  * Nothing aliases: FM bandwidth and additive partials stay far below Nyquist.
  * `render_note` trims high notes (-1.5 dB per octave above C5) and level-matches every
    instrument (`_levels.TRIM`), so at equal `gain` they sound about equally loud.

More instruments live in acoustic.py and synths.py (imported at the bottom, same registry).
"""
from __future__ import annotations

import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, samples
from .env import adsr, apply, lfo, perc

REGISTRY: dict[str, dict] = {}

C4 = 261.63
C5 = 523.25


def instrument(name: str, desc: str, family: str = "synth", span: tuple[str, str] = ("C2", "C6"), **params):
    """Register an instrument. `span` is the register it is voiced for (it still plays outside it)."""
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "desc": desc, "params": params, "family": family, "range": f"{span[0]}-{span[1]}"}
        return fn
    return deco


# ---------------------------------------------------------------- voicing helpers
def _n(dur, sr, extra=0.0):
    return samples(dur + extra, sr)


def decay_env(n, tau, sr):
    return np.exp(-np.arange(n) / (max(tau, 1e-5) * sr))


def _kt(freq, lo, hi, f_lo=C4, f_hi=2093.0):
    """Key-track a value: `lo` at or below f_lo, `hi` at or above f_hi, log-interpolated between."""
    x = np.clip(np.log(max(freq, 1.0) / f_lo) / np.log(f_hi / f_lo), 0.0, 1.0)
    return lo + (hi - lo) * float(x)


def _roll(fh, corner=3000.0, slope=2.0):
    """Gain for a partial at `fh` Hz: flat below `corner`, then falling 6*slope dB/octave."""
    return 1.0 / (1.0 + (np.asarray(fh, dtype=np.float64) / corner) ** slope)


def _cut(freq, harm, lo, hi):
    """Filter cutoff that keeps `harm` harmonics of the note but stays inside [lo, hi] Hz."""
    return float(np.clip(freq * harm, lo, hi))


def _fm_cap(freq, ratio, top=5000.0):
    """Largest FM index that keeps the strong sidebands of `freq` (modulator = freq*ratio) under `top` Hz."""
    return max(0.0, (top / freq - 1.0) / max(ratio, 0.25)) * 0.6


def _off(n, dur, sr, rel=0.08):
    """Damper envelope: 1 while the note is held, exponential fall (time constant `rel`) after `dur`."""
    env = np.ones(n)
    off = samples(dur, sr)
    if off < n:
        env[off:] = np.exp(-np.arange(n - off) / (rel * sr))
    return env


def _modes(freq, n, sr, modes, corner=3200.0, slope=2.0, beat=0.0):
    """Struck-bar / bell partials: modes = [(ratio, amp, tau_seconds), ...].

    Every partial above the fundamental is rolled off by its absolute frequency, so the same recipe
    is rich at C4 and almost a pure tone at C7. `beat` (Hz) adds a detuned twin of each partial for
    the slow shimmer of real metal.
    """
    t = np.arange(n) / sr
    y = np.zeros(n)
    top = min(9000.0, 0.45 * sr)
    for ratio, amp, tau in modes:
        fh = freq * ratio
        if fh >= top:
            continue
        a = amp * (float(_roll(fh, corner, slope)) / float(_roll(freq, corner, slope)) if ratio > 1.01 else 1.0)
        if a < 1e-4:
            continue
        part = np.sin(2 * np.pi * fh * t)
        if beat:
            part = 0.6 * part + 0.4 * np.sin(2 * np.pi * (fh + beat * ratio ** 0.5) * t + 1.0)
        y += a * np.exp(-t / max(tau, 1e-3)) * part
    return y


def _spectrum(freq, sr, tilt=1.0, formants=(), corner=3000.0, slope=2.0, top=7000.0, even=1.0, maxh=48):
    """Harmonic amplitudes for `_harmonics`: source tilt k^-tilt, `even` scales harmonics 2, 4, 6...,
    `formants` = [(center_hz, width_hz, gain), ...] are fixed body resonances (gain 2 = +9.5 dB),
    then the roll-off above `corner`. Normalised to unit power."""
    K = max(1, int(min(maxh, min(top, 0.45 * sr) // freq)))
    k = np.arange(1, K + 1, dtype=np.float64)
    fk = k * freq
    a = k ** -tilt
    a[1::2] *= even
    if formants:
        body = np.ones(K)
        for fc, bw, g in formants:
            body += g * np.exp(-0.5 * ((fk - fc) / bw) ** 2)
        a *= body
    a *= _roll(fk, corner, slope)
    return a / np.sqrt(np.sum(a ** 2) + 1e-12)


def _harmonics(freq, n, sr, amps):
    """Band-limited harmonic stack; `freq` may be a per-sample array (vibrato, scoops)."""
    w = 2 * np.pi * O.phase(freq, n, sr)
    y = np.zeros(n)
    for k, a in enumerate(amps, 1):
        if a > 1e-4:
            y += a * np.sin(k * w)
    return y


def _vibrato(freq, n, sr, rate=5.5, depth=0.3, delay=0.15, ramp=0.25, scoop=0.0):
    """Per-sample frequency: vibrato (`depth` semitones) that fades in after `delay`, and an optional
    onset `scoop` (semitones the note starts flat by, gone in ~40 ms)."""
    t = np.arange(n) / sr
    onset = np.clip((t - delay) / max(ramp, 1e-3), 0.0, 1.0)
    semis = depth * onset * np.sin(2 * np.pi * rate * t)
    if scoop:
        semis = semis - scoop * np.exp(-t / 0.04)
    return freq * 2 ** (semis / 12.0)


def _soften(x, freq, sr, harm=6.0, lo=2000.0, hi=4500.0, bright=0.5):
    """Rounds a raw pulse/saw: 12 dB/oct lowpass that key-tracks. bright 0.5 = default, 1 = raw."""
    fc = _cut(freq, harm, lo, hi) * 2 ** ((bright - 0.5) * 4.0)
    if fc >= sr * 0.45:
        return x
    return F.lowpass(x, fc, sr, 0.707)


# ---------------------------------------------------------------- keys
def _piano(freq, dur, sr, vel, decay, bright, attack=0.003, hammer=0.1, thump=0.0, release=0.07):
    # upper strings ring shorter, as on the real thing
    ring = min(decay * _kt(freq, 1.0, 0.3, f_lo=C4, f_hi=4186.0), dur + 1.5)
    n = _n(ring, sr)
    t = np.arange(n) / sr
    b = bright * (0.45 + 0.55 * vel)                       # hammer hardness
    inharm = 0.00035 * (max(freq, 110.0) / C4) ** 1.4       # stiff-string stretch grows up the keyboard
    tilt = _kt(freq, 0.95, 1.7, f_lo=65.0, f_hi=1046.0) / max(b, 0.3) ** 0.4
    corner = 2400.0 * b
    top = min(6500.0, 0.45 * sr)
    r1 = float(_roll(freq, corner, 2.0))
    x = np.zeros(n)
    for h in range(1, 41):
        fh = freq * h * np.sqrt(1 + inharm * h * h)
        if fh > top and h > 1:
            break
        amp = h ** -tilt * (1.0 if h % 2 else 0.8) * float(_roll(fh, corner, 2.0)) / r1
        if amp < 2e-4:
            break
        tau = ring / (2.2 + 0.9 * (h - 1))
        # two strings a hair apart: the slow beating that keeps a piano tone alive
        x += amp * np.exp(-t / tau) * (np.sin(2 * np.pi * fh * t) + 0.6 * np.sin(2 * np.pi * fh * 1.0006 * t + h)) / 1.6
    knock = F.bandpass(O.white(n, seed=3), _cut(freq, 3, 400, 2200), sr, 0.7) * decay_env(n, 0.006, sr) * hammer * vel
    if thump:
        knock += F.lowpass(O.white(n, seed=4), 420, sr) * decay_env(n, 0.018, sr) * thump
    env = _off(n, dur, sr, release)
    a = samples(attack, sr)
    env[:a] *= np.linspace(0, 1, a)
    y = (x + knock) * env
    y = F.lowpass(y, max(800 + 4200 * b, min(freq * 2, 5000.0)), sr, 0.6)
    return y * (0.35 + 0.65 * vel)


@instrument("piano", "Acoustic piano: stretched partials, hammer knock, velocity-sensitive brightness; the top octaves thin out and ring shorter.",
            family="keys", span=("A0", "C7"), decay=(2.5, "seconds of ring"), bright=(1.0, "0.5 mellow .. 2 bright"))
def piano(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=2.5, bright=1.0):
    return _piano(freq, dur, sr, vel, decay, bright)


@instrument("felt_piano", "Soft felted upright (lo-fi / ambient / cinematic): muted hammers, slow attack, a little mechanical thump.",
            family="keys", span=("A1", "C6"), decay=(2.2, "seconds of ring"))
def felt_piano(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=2.2):
    return _piano(freq, dur, sr, vel, decay, 0.45, attack=0.012, hammer=0.03, thump=0.25, release=0.12)


@instrument("epiano", "Electric piano (Rhodes-style FM tine): round and warm, barks a little when hit hard.",
            family="keys", span=("C2", "C6"), tremolo=(0.2, "0..1 wobble"), decay=(1.8, "seconds"))
def epiano(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.2, decay=1.8):
    ring = min(decay, dur + 1.0)
    n = _n(ring, sr)
    cap = _fm_cap(freq, 1.0, 3200.0)
    idx = min(1.7 * vel, cap) * decay_env(n, 0.3, sr) + min(0.12, cap)
    tine = O.fm(freq, n, sr, ratio=1.0, index=idx)
    h = max(3, int(round(min(freq * 9, 3000.0) / freq)))    # the tine "ping" stays under 3 kHz on every key
    ping = O.sine(freq * h, n, sr) * decay_env(n, 0.035, sr) * 0.12 * vel * float(_roll(freq * h, 2600.0))
    env = decay_env(n, ring / 3, sr) * _off(n, dur, sr, 0.08)
    y = (tine + ping) * env
    if tremolo > 0:
        y *= 1 - tremolo * 0.5 * (1 + lfo(n / sr, 5.5, sr)[:n])
    return y * (0.4 + 0.6 * vel)


@instrument("organ", "Drawbar organ with rotary wobble; the top drawbars fold back an octave on high notes like a Hammond.",
            family="keys", span=("C2", "C6"), rotary=(0.3, "0..1"), upper=(1.0, "level of the upper drawbars: 0 = flute stops only .. 1.5 = full"))
def organ(freq, dur, sr=DEFAULT_SR, vel=1.0, rotary=0.3, upper=1.0):
    rel = 0.05
    n = _n(dur, sr, rel)
    x = np.zeros(n)
    for h, g in zip((1, 2, 3, 4, 6, 8), (1.0, 0.6, 0.4, 0.3, 0.2, 0.1)):
        fh = freq * h
        while fh > 4200.0 and fh > freq * 1.01:   # tonewheel fold-back
            fh /= 2
        x += g * (upper if h > 2 else 1.0) * float(_roll(fh, 3500.0) / _roll(freq, 3500.0)) * O.sine(fh, n, sr)
    x /= 2.6
    if rotary > 0:
        x *= 1 - rotary * 0.3 * (1 + lfo(n / sr, 6.0, sr)[:n]) / 2
    return apply(x, adsr(dur, 0.01, 0.0, 1.0, rel, sr)) * 0.7 * (0.6 + 0.4 * vel)


# ---------------------------------------------------------------- plucked strings
def _ks(freq, n, sr, t60=2.0, bright=3000.0, damp=0.5, pick=0.2, grit=0.3, seed=0):
    """Tuned Karplus-Strong string.

    The loop is delay + two-point lowpass + a first-order allpass for the fractional sample, so high
    notes are in tune (a plain integer delay is ~35 cents flat at C7). The excitation is one period
    built in the frequency domain: a plucked-string spectrum sin(k*pi*pick)/k, optionally roughened
    with noise (`grit`), band-limited at `bright` Hz, so every note of a run has the same timbre.

    t60: seconds for the fundamental to fall 60 dB; damp: 0.05 (wiry, highs sustain) .. 0.5 (soft).
    """
    from scipy.signal import lfilter
    P = sr / freq
    # the loop lowpass costs a fixed loss per period, i.e. far more per second on a high string:
    # ease it off up the neck or the top two octaves are just a tick
    d = float(np.clip(damp, 0.02, 0.5)) * _kt(freq, 1.0, 0.25)
    N = int(np.floor(P - d - 0.1))
    if N < 1:   # above ~sr/3: nothing left to model
        return O.sine(freq, n, sr) * decay_env(n, t60 / 6.9, sr)
    eta = P - d - N
    c = (1 - eta) / (1 + eta)
    w0 = 2 * np.pi * freq / sr
    loss = np.sqrt(1 - 2 * d * (1 - d) * (1 - np.cos(w0)))          # what the lowpass takes from the fundamental
    g = min(0.99995, float(np.exp(-6.91 / (freq * max(t60, 0.02)))) / loss)
    a = np.zeros(N + 3)
    a[0] = 1.0
    a[1] += c
    a[N] -= g * (1 - d) * c
    a[N + 1] -= g * ((1 - d) + d * c)
    a[N + 2] -= g * d
    L = max(4, int(round(P)))
    K = L // 2
    k = np.arange(1, K + 1, dtype=np.float64)
    rng = np.random.default_rng(1000 + seed + int(freq) % 997)
    mag = np.abs(np.sin(k * np.pi * pick)) / k
    if grit > 0:
        mag = (1 - grit) * mag + grit * rng.rayleigh(0.5, K) / np.sqrt(k)
    # scattered phases: the same spectrum as a clean pluck but spread over the period like a real
    # string after a few reflections, instead of one needle-sharp spike per cycle
    spec = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, K)) * _roll(k * freq, bright, 2.0)
    full = np.zeros(L // 2 + 1, dtype=complex)
    full[1:K + 1] = spec[:L // 2]
    burst = np.fft.irfft(full, L)
    burst /= np.max(np.abs(burst)) + 1e-12
    x = np.zeros(n)
    x[:min(L, n)] = burst[:min(L, n)]
    return lfilter([1.0, c], a, x)


def _string(freq, dur, sr, vel, t60, bright, damp, pick, grit, release=0.1, tail=1.0, seed=0, ring=None):
    """A plucked note: `_ks` + damper at note-off + fade, peak-normalised and velocity-scaled.
    `ring` forces the buffer length (seconds); by default it follows the string's own decay."""
    t60 = t60 * _kt(freq, 1.0, 0.45)
    ring = min(t60 * 0.9, dur + tail) if ring is None else ring
    n = _n(ring, sr)
    y = _ks(freq, n, sr, t60, bright * (0.5 + 0.5 * vel), damp, pick, grit, seed)
    y = y * _off(n, dur, sr, release) * perc(ring, 0.0005, sr, curve=1.0)
    y = F.dc_block(y, sr)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel)


@instrument("pluck", "Plucked string (Karplus-Strong), tuned on every key: generic guitar/harp-like pluck.",
            family="plucked", span=("C2", "C7"), damping=(0.5, "0 bright and wiry .. 1 soft and dull"), decay=(1.5, "seconds"))
def pluck(freq, dur, sr=DEFAULT_SR, vel=1.0, damping=0.5, decay=1.5):
    damping = float(np.clip(damping, 0.0, 1.0))
    return _string(freq, dur, sr, vel, t60=decay * 1.6, bright=1400 + 4200 * (1 - damping), damp=0.15 + 0.35 * damping,
                   pick=0.2, grit=0.35, ring=min(decay, dur + 1.0))


@instrument("harp", "Concert harp: soft finger pluck near the middle of the string, long ring.", family="plucked", span=("C2", "C7"))
def harp(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=4.0, bright=3000, damp=0.3, pick=0.38, grit=0.1, release=0.35, tail=2.0)
    return F.peak(y, 220, sr, 0.8, 2.5) * 0.9


@instrument("guitar", "Nylon-string classical guitar: warm, woody body.", family="plucked", span=("E2", "E5"))
def guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=3.0, bright=2600, damp=0.4, pick=0.24, grit=0.2, release=0.12, tail=1.4)
    y = F.peak(F.peak(y, 110, sr, 1.2, 3.0), 220, sr, 1.5, 2.0)
    return F.lowpass(y, 3500, sr, 0.7)


# ---------------------------------------------------------------- bass
@instrument("bass", "Fat analog synth bass with filter envelope.", family="bass", span=("C1", "C4"),
            cutoff=(1200, "Hz peak"), res=(1.5, "resonance"), sub=(0.5, "sine sub-oscillator 0..1"))
def bass(freq, dur, sr=DEFAULT_SR, vel=1.0, cutoff=1200, res=1.5, sub=0.5):
    rel = 0.15
    n = _n(dur, sr, rel)
    x = O.saw(freq, n, sr) * 0.6 + O.square(freq * 0.5, n, sr) * 0.3 + sub * O.sine(freq * 0.5, n, sr)
    fenv = 120 + cutoff * vel * decay_env(n, 0.25, sr)
    x = F.biquad(x, "lowpass", fenv, sr, q=res)
    env = adsr(dur, 0.005, 0.15, 0.7, rel, sr)
    return apply(x, env) * 0.6 * (0.5 + 0.5 * vel)


@instrument("sub", "Pure sine sub bass with soft click.", family="bass", span=("C1", "C3"))
def sub(freq, dur, sr=DEFAULT_SR, vel=1.0):
    n = _n(dur, sr, 0.1)
    x = O.sine(freq, n, sr) + 0.15 * O.sine(freq * 2, n, sr)
    return apply(x, adsr(dur, 0.01, 0.05, 0.9, 0.1, sr)) * 0.8 * (0.5 + 0.5 * vel)


@instrument("wobble", "Dubstep-style LFO filtered bass.", family="bass", span=("C1", "C3"), rate=(4.0, "wobble Hz"))
def wobble(freq, dur, sr=DEFAULT_SR, vel=1.0, rate=4.0):
    n = _n(dur, sr, 0.1)
    x = O.saw(freq, n, sr) + O.square(freq * 0.5, n, sr) * 0.5
    fc = 180 + 1900 * (0.5 + 0.5 * lfo(n / sr, rate, sr)[:n])
    x = F.biquad(x, "lowpass", fc, sr, q=2.8)
    return apply(x, adsr(dur, 0.01, 0.0, 1.0, 0.1, sr)) * 0.4 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- leads / pads / ensembles
@instrument("lead", "Supersaw lead: full and smooth by default; raise `cutoff` for bite.", family="synth", span=("C3", "C6"),
            detune=(0.3, "semitones spread"), cutoff=(3200, "Hz filter peak"))
def lead(freq, dur, sr=DEFAULT_SR, vel=1.0, detune=0.3, cutoff=3200):
    rel = 0.2
    n = _n(dur, sr, rel)
    x = O.supersaw(freq, n, sr, voices=5, detune=detune)
    pk = max(cutoff * (0.6 + 0.4 * vel), min(freq * 2.2, 4500.0))   # never closes below the 2nd harmonic
    fenv = pk * (0.45 + 0.55 * decay_env(n, 0.4, sr))
    x = F.biquad(x, "lowpass", fenv, sr, q=0.9, order=2)
    return apply(x, adsr(dur, 0.01, 0.2, 0.75, rel, sr)) * 0.5 * (0.5 + 0.5 * vel)


@instrument("pad", "Warm slow-attack synth pad (stereo when chorused).", family="pad", span=("C2", "C5"),
            attack=(0.4, "s"), release=(0.8, "s"), cutoff=(2000, "Hz"))
def pad(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.4, release=0.8, cutoff=2000):
    n = _n(dur, sr, release)
    x = O.supersaw(freq, n, sr, voices=7, detune=0.35) * 0.7 + O.sine(freq * 0.5, n, sr) * 0.3
    x = F.lowpass(x, max(cutoff, min(freq * 1.5, 3500.0)), sr, 0.7, order=2)
    return apply(x, adsr(dur, attack, 0.3, 0.8, release, sr)) * 0.45 * (0.6 + 0.4 * vel)


@instrument("strings", "String ensemble: detuned saws, slow attack, vibrato.", family="bowed", span=("C2", "C6"),
            attack=(0.25, "s"), release=(0.5, "s"))
def strings(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.25, release=0.5):
    n = _n(dur, sr, release)
    vib = _vibrato(freq, n, sr, rate=5.0, depth=0.07, delay=0.0, ramp=0.4)
    x = O.supersaw(vib, n, sr, voices=6, detune=0.25)
    x = F.lowpass(x, max(2600.0 * (0.7 + 0.3 * vel), min(freq * 1.8, 3800.0)), sr, 0.7, order=2)
    x = F.peak(x, 800, sr, 1.2, 3)
    return apply(x, adsr(dur, attack, 0.3, 0.85, release, sr, curve=1.5)) * 0.4 * (0.6 + 0.4 * vel)


@instrument("brass", "Synth brass section with an opening filter (fanfares, stabs).", family="brass", span=("C2", "C6"),
            attack=(0.06, "s"), bright=(1.0, "0.5 mellow .. 2 blaring"))
def brass(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.06, bright=1.0):
    rel = 0.25
    n = _n(dur, sr, rel)
    x = O.saw(freq, n, sr) * 0.7 + O.saw(freq * 1.005, n, sr) * 0.3
    t = np.arange(n) / sr
    pk = max(2600.0 * vel * bright, min(freq * 1.6, 3600.0))
    fenv = 350 + pk * (1 - np.exp(-t / 0.08)) * (0.6 + 0.4 * np.exp(-t / 0.5))
    x = F.biquad(x, "lowpass", fenv, sr, q=1.1, order=2)
    return apply(x, adsr(dur, attack, 0.1, 0.85, rel, sr)) * 0.55 * (0.5 + 0.5 * vel)


@instrument("choir", "Vowel-formant synth choir (\"aah\").", family="voice", span=("C2", "C6"), vowel=("a", "a|e|i|o|u"))
def choir(freq, dur, sr=DEFAULT_SR, vel=1.0, vowel="a"):
    formants = {"a": (800, 1150, 2900), "e": (400, 1600, 2700), "i": (350, 1700, 2700), "o": (450, 800, 2830), "u": (325, 700, 2530)}
    rel = 0.6
    n = _n(dur, sr, rel)
    src = O.supersaw(_vibrato(freq, n, sr, 5.2, 0.08, 0.2, 0.5), n, sr, voices=5, detune=0.2)
    y = np.zeros(n)
    for f, g in zip(formants.get(vowel, formants["a"]), (1.0, 0.6, 0.18)):
        y += g * F.bandpass(src, max(f, freq * 0.9), sr, 6.0)
    y = F.lowpass(y, 3600, sr, 0.7)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return apply(y, adsr(dur, 0.35, 0.3, 0.85, rel, sr)) * 0.5 * (0.6 + 0.4 * vel)


@instrument("pwm", "Pulse-width-modulated synth (retro/lush).", family="synth", span=("C2", "C6"), rate=(0.7, "LFO Hz"))
def pwm(freq, dur, sr=DEFAULT_SR, vel=1.0, rate=0.7):
    rel = 0.25
    n = _n(dur, sr, rel)
    # PWM = two saws with a moving phase offset
    ph = 0.5 + 0.4 * lfo(n / sr, rate, sr)[:n]
    t = O.phase(freq, n, sr) % 1.0
    dt = np.full(n, freq / sr)
    t2 = (t + ph) % 1.0
    a = 2.0 * t - 1.0 - O._polyblep(t, dt)
    b = 2.0 * t2 - 1.0 - O._polyblep(t2, dt)
    x = F.lowpass(a - b, _cut(freq, 6, 2200, 3800), sr, 0.7, order=2)
    return apply(x, adsr(dur, 0.02, 0.2, 0.8, rel, sr)) * 0.45 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- bells / mallets
@instrument("bell", "FM bell / chime (notifications, magic). The clang is capped by pitch, so high notes ring clean instead of clashing.",
            family="mallet", span=("C3", "C7"), decay=(2.0, "seconds"), ratio=(3.5, "modulator ratio; 2=softer, 5.04=glassy"),
            bright=(1.0, "0.4 soft chime .. 2 hard metallic strike"))
def bell(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=2.0, ratio=3.5, bright=1.0):
    ring = max(dur, decay)
    n = _n(ring, sr)
    idx_pk = min((0.8 + 1.2 * vel) * bright, _fm_cap(freq, ratio, 3200.0 * bright ** 0.5))
    idx = idx_pk * (0.2 + 0.8 * decay_env(n, ring / 7, sr))     # the strike is bright, the ring is mellow
    x = O.fm(freq, n, sr, ratio=ratio, index=idx)
    x += 0.25 * float(_roll(freq * 2.01, 2800.0)) * O.sine(freq * 2.01, n, sr) * decay_env(n, ring / 6, sr)
    return x * decay_env(n, ring / 3, sr) * perc(ring, 0.003, sr, curve=1.2) * 0.7 * (0.55 + 0.45 * vel)


@instrument("glass", "Glassy crystal tone (UI, sparkle): a singing wine glass, pure rather than sharp.", family="mallet", span=("C4", "C7"))
def glass(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 0.8)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 3), (2.32, 0.32, ring / 7), (4.25, 0.14, ring / 14)], corner=2400.0, beat=1.2)
    return x * perc(ring, 0.004, sr, curve=1.2) * 0.75 * (0.55 + 0.45 * vel)


@instrument("marimba", "Wooden mallet on a rosewood bar with resonator: round, woody, short.", family="mallet", span=("C2", "C7"),
            decay=(0.6, "seconds"))
def marimba(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=0.6):
    ring = max(dur, decay * _kt(freq, 1.3, 0.6, f_lo=130.0))
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.2), (4.0, 0.3 * vel, ring / 9), (9.2, 0.1 * vel, ring / 22)], corner=2600.0)
    click = F.bandpass(O.white(n, seed=11), _cut(freq, 3, 700, 2200), sr, 0.8) * decay_env(n, 0.004, sr) * 0.12 * vel
    return (x + click) * perc(ring, 0.001, sr, curve=1.0) * (0.6 + 0.4 * vel)


@instrument("kalimba", "Thumb piano: soft metal tine over a wooden box.", family="mallet", span=("C3", "C7"))
def kalimba(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.2)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 0.5), (2.0, 0.18, 0.3), (5.6, 0.3 * vel, 0.05)], corner=2800.0)
    tick = F.bandpass(O.white(n, seed=12), _cut(freq, 2, 500, 1800), sr, 1.0) * decay_env(n, 0.003, sr) * 0.08 * vel
    return (x + tick) * perc(ring, 0.001, sr, curve=1.0) * 0.8 * (0.6 + 0.4 * vel)


@instrument("vibraphone", "Vibraphone: soft mallets on metal bars with motor tremolo.", family="mallet", span=("F3", "F6"),
            motor=(4.5, "tremolo Hz (0 = motor off)"))
def vibraphone(freq, dur, sr=DEFAULT_SR, vel=1.0, motor=4.5):
    ring = max(dur, 1.8)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 0.9), (4.0, 0.25 * vel, 0.3), (10.0, 0.06 * vel, 0.06)], corner=2800.0)
    if motor > 0:
        x *= 1 - 0.3 * (1 + lfo(n / sr, motor, sr)[:n]) / 2
    return x * perc(ring, 0.002, sr, curve=1.0) * _off(n, dur + 0.6, sr, 0.25) * 0.8 * (0.6 + 0.4 * vel)


@instrument("music_box", "Tiny music-box comb tine: sweet and delicate, made to be played high.", family="mallet", span=("C5", "C8"))
def music_box(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.0)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 0.35), (3.0, 0.4, 0.14), (5.4, 0.2, 0.05)], corner=3000.0)
    return x * perc(ring, 0.001, sr, curve=1.0) * 0.7 * (0.6 + 0.4 * vel)


@instrument("steel_drum", "Caribbean steel pan.", family="mallet", span=("C3", "C6"))
def steel_drum(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.2)
    n = _n(ring, sr)
    x = O.fm(freq, n, sr, ratio=2.0, index=min(1.2 * vel + 0.2, _fm_cap(freq, 2.0, 4200.0)) * decay_env(n, 0.15, sr))
    x += 0.35 * float(_roll(freq * 2, 2600.0)) * O.fm(freq * 2.0, n, sr, ratio=1.5, index=min(0.5, _fm_cap(freq * 2, 1.5, 4200.0)) * decay_env(n, 0.1, sr))
    return x * decay_env(n, 0.45, sr) * perc(ring, 0.002, sr, curve=1.0) * 0.7 * (0.6 + 0.4 * vel)


# ---------------------------------------------------------------- winds
@instrument("flute", "Breathy concert flute (sine stack + breath + vibrato).", family="wind", span=("C4", "C7"), breath=(0.15, "noise amount"))
def flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.15):
    rel = 0.15
    n = _n(dur, sr, rel)
    vib = _vibrato(freq, n, sr, rate=5.5, depth=0.1, delay=0.1, ramp=0.3)
    amps = np.array([1.0, 0.25 * (0.6 + 0.4 * vel), 0.08 * vel]) * _roll(np.array([1, 2, 3]) * freq, 2600.0) / float(_roll(freq, 2600.0))
    x = _harmonics(vib, n, sr, amps)
    air = F.bandpass(O.white(n, seed=5), _cut(freq, 2, 700, 1900), sr, 2.0) * breath * _kt(freq, 1.0, 0.45)
    chiff = F.bandpass(O.white(n, seed=6), _cut(freq, 3, 900, 2400), sr, 1.5) * decay_env(n, 0.02, sr) * breath * 0.8
    return apply(x + air + chiff, adsr(dur, 0.07, 0.1, 0.85, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)


# ---------------------------------------------------------------- retro / raw
@instrument("chip", "Chiptune pulse wave (NES/GameBoy), rounded like it came out of a small console speaker.",
            family="retro", span=("C3", "C7"), width=(0.25, "pulse width 0.125|0.25|0.5"), vibrato=(0.0, "0..1"),
            bright=(0.5, "0 = dull .. 0.5 = default .. 1 = raw unfiltered pulse"))
def chip(freq, dur, sr=DEFAULT_SR, vel=1.0, width=0.25, vibrato=0.0, bright=0.5):
    rel = 0.02
    n = _n(dur, sr, rel)
    f = freq * (1 + 0.01 * vibrato * lfo(n / sr, 6.0, sr)[:n]) if vibrato else freq
    x = _soften(O.pulse(f, n, sr, width=width) - (2 * width - 1), freq, sr, harm=6, lo=2200, hi=3800, bright=bright)   # minus the pulse's DC
    return apply(x, adsr(dur, 0.002, 0.0, 1.0, rel, sr)) * 0.4 * (0.5 + 0.5 * vel)


@instrument("chiptri", "Chiptune triangle (NES bass/lead).", family="retro", span=("C1", "C6"))
def chiptri(freq, dur, sr=DEFAULT_SR, vel=1.0):
    n = _n(dur, sr, 0.02)
    x = O.triangle(freq, n, sr)
    x = np.round(x * 8) / 8  # 4-bit steps like the 2A03
    x = _soften(x, freq, sr, harm=10, lo=1800, hi=4200)
    return apply(x, adsr(dur, 0.002, 0.0, 1.0, 0.02, sr)) * 0.5 * (0.5 + 0.5 * vel)


@instrument("board", "80s pinball/arcade sound-board voice: DAC-gritty pulse with a pitch blip at each onset (bwip).",
            family="retro", span=("C3", "C6"),
            width=(0.35, "pulse width 0.1..0.5"), blip=(7.0, "semitones the onset drops from"), bits=(5, "DAC bit depth (0 = clean)"),
            vibrato=(0.0, "semitones of 9 Hz warble"))
def board(freq, dur, sr=DEFAULT_SR, vel=1.0, width=0.35, blip=7.0, bits=5, vibrato=0.0):
    rel = 0.03
    n = _n(dur, sr, rel)
    semis = blip * decay_env(n, 0.012, sr)
    if vibrato:
        semis = semis + vibrato * lfo(n / sr, 9.0, sr)[:n]
    x = O.pulse(freq * 2 ** (semis / 12), n, sr, width=width) - (2 * width - 1)
    if bits:
        q = 2 ** (int(bits) - 1)
        x = np.round(x * q) / q
    x = F.lowpass(x, _cut(freq, 6, 1800, 4200), sr)
    return apply(x, adsr(dur, 0.002, 0.08, 0.6, rel, sr)) * 0.45 * (0.5 + 0.5 * vel)


@instrument("sine", "Pure sine tone with soft envelope.", family="raw", span=("C2", "C7"))
def sine(freq, dur, sr=DEFAULT_SR, vel=1.0):
    n = _n(dur, sr, 0.05)
    return apply(O.sine(freq, n, sr), adsr(dur, 0.005, 0.0, 1.0, 0.05, sr)) * 0.7 * (0.5 + 0.5 * vel)


@instrument("square", "Square wave with envelope, corners rounded (for a truly raw one use `synth` with wave=square and a high cutoff).",
            family="raw", span=("C2", "C6"))
def square(freq, dur, sr=DEFAULT_SR, vel=1.0):
    n = _n(dur, sr, 0.05)
    x = _soften(O.square(freq, n, sr), freq, sr, harm=7, lo=2400, hi=3600)
    return apply(x, adsr(dur, 0.005, 0.0, 1.0, 0.05, sr)) * 0.4 * (0.5 + 0.5 * vel)


@instrument("saw", "Saw wave with envelope and a key-tracked lowpass.", family="raw", span=("C2", "C6"))
def saw(freq, dur, sr=DEFAULT_SR, vel=1.0):
    n = _n(dur, sr, 0.05)
    x = _soften(O.saw(freq, n, sr), freq, sr, harm=7, lo=2200, hi=3200)
    return apply(x, adsr(dur, 0.005, 0.0, 1.0, 0.05, sr)) * 0.45 * (0.5 + 0.5 * vel)


@instrument("synth", "Generic subtractive synth: pick wave, ADSR and filter (nothing is tamed here: what you set is what you get).",
            family="raw", span=("C1", "C7"),
            wave=("saw", "sine|saw|square|pulse|triangle|supersaw|fm|organ"), attack=(0.01, "s"), decay=(0.1, "s"),
            sustain=(0.7, "0..1"), release=(0.2, "s"), cutoff=(3200, "Hz"), res=(0.8, "resonance"),
            fenv=(0.0, "filter envelope amount Hz added at note start"), detune=(0.3, "supersaw spread"), width=(0.5, "pulse width"),
            ratio=(2.0, "fm ratio"), index=(1.0, "fm index"))
def synth(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="saw", attack=0.01, decay=0.1, sustain=0.7, release=0.2,
          cutoff=3200, res=0.8, fenv=0.0, detune=0.3, width=0.5, ratio=2.0, index=1.0):
    n = _n(dur, sr, release)
    x = O.osc(wave, freq, n, sr, detune=detune, width=width, ratio=ratio, index=index)
    if fenv:
        fc = cutoff + fenv * decay_env(n, max(decay, 0.05), sr)
        x = F.biquad(x, "lowpass", fc, sr, q=res)
    elif cutoff < sr / 2 - 100:
        x = F.lowpass(x, cutoff, sr, res)
    return apply(x, adsr(dur, attack, decay, sustain, release, sr)) * 0.5 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- rendering
def key_gain(freq: float) -> float:
    """Loudness trim for high notes: -1.5 dB per octave above C5, at most -5 dB.

    The ear is most sensitive around 2-5 kHz, so a line that climbs gets louder (and sharper)
    at equal amplitude. Real instruments and players pull the top back; so does this."""
    return float(10 ** (-min(5.0, 1.5 * max(0.0, np.log2(max(freq, 1.0) / C5))) / 20))


def render_note(name: str, freq: float, dur: float, sr: int = DEFAULT_SR, vel: float = 1.0, **params) -> np.ndarray:
    if name not in REGISTRY:
        raise ValueError(f"unknown instrument {name!r}; run `genny list instruments`")
    entry = REGISTRY[name]
    allowed = entry["params"]
    unknown = [k for k in params if k not in allowed]
    if unknown:
        raise ValueError(f"instrument {name!r} does not accept params {unknown}; allowed: {list(allowed)}")
    y = entry["fn"](freq, dur, sr, vel, **params)
    if y.ndim == 1 and y.shape[0] > 8:   # no note carries DC (blown and bowed models did: up to 18 % of peak); 5 Hz, far below any pitch
        from scipy.signal import lfilter
        y = lfilter([1.0, -1.0], [1.0, -(1.0 - 2 * np.pi * 5.0 / sr)], y)
    k = min(y.shape[0] // 2, samples(0.004, sr))   # no note ever ends on a step
    if k > 1:
        y = y.copy()
        y[-k:] *= np.linspace(1.0, 0.0, k)
    return y * (key_gain(freq) * TRIM.get(name, 1.0))


def render_chord(name: str, freqs: list[float], dur: float, sr: int = DEFAULT_SR, vel: float = 1.0, strum: float = 0.0, **params) -> np.ndarray:
    """Sum several notes; `strum` delays each successive note by that many seconds."""
    from .core import mix
    parts = [(render_note(name, f, dur, sr, vel, **params), i * strum) for i, f in enumerate(freqs)]
    y = mix(parts, sr)
    return y / np.sqrt(max(1, len(freqs)))


from . import acoustic  # noqa: E402,F401  (registers more instruments)
from . import synths  # noqa: E402,F401
from ._levels import TRIM  # noqa: E402

# -----------------------------------------------------------------------------
# Physical-engine compatibility override (v0.5)
@instrument("piano", "Physical acoustic piano: nonlinear felt hammer, stiff dispersive string and commuted soundboard.",
            family="keys", span=("A0", "C7"), decay=(2.5, "seconds of ring"), bright=(1.0, "0.5 mellow .. 2 bright"))
def piano(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=2.5, bright=1.0):
    from .physics import Piano as PhysicalPiano
    # Map the long-standing Genny controls onto the physical engine rather than
    # deleting them.  Higher ``bright`` means harder effective felt / more HF.
    b = float(np.clip(bright, 0.35, 2.0))
    model = PhysicalPiano(brightness=float(np.clip(0.33 + 0.34 * b, 0.2, 1.0)),
                          felt=float(np.clip(0.26 / b, 0.05, 0.45)),
                          inharmonicity=4e-4,
                          decay=float(np.clip(decay / 4.5, 0.25, 1.2)),
                          soundboard=0.28)
    y = model.note(freq, max(dur, min(decay, dur + 1.0)), velocity=vel, sr=sr, seed=3)
    # Keep Genny's historical top-octave taming; the physical model itself is
    # deliberately broader because it is also exposed directly via genny.Piano.
    y = F.lowpass(y, min(5800.0, max(1400.0, 1700.0 + 2800.0 * b)), sr, 0.7)
    return y * 0.72
