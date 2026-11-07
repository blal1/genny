"""Effects. Every effect is registered with its parameters so `genny list fx` is self-documenting.

Signature convention: fn(x, sr, **params) -> ndarray. Works on mono (n,) or stereo (n,2).
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter, resample_poly

from . import filters as F
from .core import DEFAULT_SR, db_to_gain, is_stereo, normalize as _normalize, read_wav, samples, to_mono, to_stereo, trim_tail as _trim_tail
from .env import lfo as _lfo

REGISTRY: dict[str, dict] = {}


def effect(name: str, desc: str, **params):
    """params: name=(default, help)"""
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "desc": desc, "params": params}
        return fn
    return deco


def _per_channel(fn, x):
    if x.ndim == 1:
        return fn(x)
    return np.stack([fn(x[:, c]) for c in range(x.shape[1])], axis=1)


def _comb(x, delay_samples: int, feedback: float, damp: float = 0.2):
    """Feedback comb with a one-pole lowpass in the loop (Freeverb style).
    y[n] = x[n] + fb * z[n-D],  z = lowpass(y). Processed in exact D-sized blocks so it is O(n)."""
    D = max(1, int(delay_samples))
    n = x.shape[0]
    y = np.empty(n)
    prev_z = np.zeros(D)
    zi = np.zeros(1)
    b, a = [1 - damp], [1, -damp]
    for s in range(0, n, D):
        e = min(s + D, n)
        blk = x[s:e] + feedback * prev_z[: e - s]
        y[s:e] = blk
        z, zi = lfilter(b, a, blk, zi=zi)
        if e - s == D:
            prev_z = z
        else:
            prev_z = np.concatenate([z, prev_z[e - s:]])
    return y


def _allpass(x, delay_samples: int, g: float = 0.5):
    """Schroeder allpass y[n] = -g x[n] + x[n-D] + g y[n-D], in exact D-sized blocks."""
    D = max(1, int(delay_samples))
    n = x.shape[0]
    y = np.empty(n)
    prev_x = np.zeros(D)
    prev_y = np.zeros(D)
    for s in range(0, n, D):
        e = min(s + D, n)
        k = e - s
        blk = -g * x[s:e] + prev_x[:k] + g * prev_y[:k]
        y[s:e] = blk
        if k == D:
            prev_x, prev_y = x[s:e], blk
        else:
            prev_x = np.concatenate([x[s:e], prev_x[k:]])
            prev_y = np.concatenate([blk, prev_y[k:]])
    return y


@effect("reverb", "Schroeder/Freeverb-style room reverb.",
        mix=(0.3, "wet amount 0..1"), size=(0.7, "room size 0..1"), damp=(0.4, "high-frequency damping 0..1"),
        predelay=(0.0, "seconds before reverb starts"), tail=(None, "seconds of tail to append (auto if omitted)"),
        width=(1.0, "stereo width 0..1 (stereo output when > 0)"),
        variant=("genny", "genny (comb lengths follow size) | freeverb (Jezar's exact structure: lengths fixed, damp 0.2 by default, his non-allpass 'allpass')"),
        freeze=(False, "Freeverb freeze: comb feedback 1 and no damping, the sound hangs for the whole tail (4 s if tail is omitted)"))
def reverb(x, sr=DEFAULT_SR, mix=0.3, size=0.7, damp=None, predelay=0.0, tail=None, width=1.0, variant="genny", freeze=False):
    """variant freeverb, PASP 3.6: f = 0.7 + 0.28 size, d = 0.2, combs at fixed lengths, series sections (-1 + (1+g) z^-N) / (1 - g z^-N), g = 0.5."""
    exact = variant == "freeverb"
    damp = (0.2 if exact else 0.4) if damp is None else float(damp)
    size = float(np.clip(size, 0.0, 1.0))
    fb = 0.7 + 0.28 * size
    if freeze:
        fb, damp = 1.0, 0.0
    if tail is None:
        tail = 4.0 if freeze else 0.4 + 2.6 * size
    n_extra = samples(tail, sr)
    combs = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
    aps = [225, 556, 441, 341]
    scale = sr / 44100.0

    def channel(sig, offset):
        sig = np.concatenate([sig, np.zeros(n_extra)])
        pd = samples(predelay, sr) if predelay > 0 else 0
        if pd:
            sig = np.concatenate([np.zeros(pd), sig])[: sig.shape[0]]
        wet = np.zeros_like(sig)
        for c in combs:
            wet += _comb(sig, int((c + offset) * scale * (1.0 if exact else 0.6 + 0.4 * size)), fb, damp)
        wet /= len(combs)
        for a in aps:
            N = int((a + offset) * scale)
            if exact:
                v = _comb(wet, N, 0.5, 0.0)
                wet = -wet + 1.5 * np.concatenate([np.zeros(N), v[:-N]])
            else:
                wet = _allpass(wet, N, 0.5)
        return wet

    dry = np.concatenate([x, np.zeros((n_extra,) + x.shape[1:])], axis=0)
    mono = to_mono(x)
    if width > 0:
        wl = channel(mono, 0)
        wr = channel(mono, 23)
        mid = (wl + wr) / 2
        side = (wl - wr) / 2 * width
        wet = np.stack([mid + side, mid - side], axis=1)
        dry = to_stereo(dry)
    else:
        wet = channel(mono, 0)
        if is_stereo(x):
            wet = to_stereo(wet)
    return dry * (1 - mix) + wet * mix * 0.8


@numba.njit(cache=True)
def _shaker_kernel(x, taps, ticks, targets, ramp):
    y = np.zeros_like(x)
    g = targets[:4].copy()
    step = np.zeros(4)
    left = np.zeros(4, dtype=np.int64)
    k = 4
    for n in range(x.shape[0]):
        if k < ticks.shape[0] and n >= ticks[k]:
            i = k % 4
            step[i] = (targets[k] - g[i]) / ramp
            left[i] = ramp
            k += 1
        acc = 0.0
        for i in range(4):
            if left[i] > 0:
                g[i] += step[i]
                left[i] -= 1
            if n >= taps[i]:
                acc += g[i] * x[n - taps[i]]
        y[n] = acc
    return y


@effect("shaker_comb", "Four-tap comb (0, 11, 17, 30 ms) whose tap gains wander at random: a steadily changing timbre (Puckette's shaker).",
        rate=(4.0, "gain changes per second (one tap at a time, in turn)"), ramp=(0.2, "seconds each gain takes to reach its new value"),
        seed=(0, "random gains"))
def shaker_comb(x, sr=DEFAULT_SR, rate=4.0, ramp=0.2, seed=0):
    rng = np.random.default_rng(int(seed))
    taps = np.array([0] + [samples(t, sr) for t in (0.011, 0.017, 0.030)], dtype=np.int64)

    def one(sig):
        n = sig.shape[0] + int(taps[-1])
        sig = np.concatenate([sig, np.zeros(int(taps[-1]))])
        per = max(1, samples(1.0 / float(np.clip(rate, 0.05, 200.0)), sr))
        ticks = np.concatenate([np.zeros(4, dtype=np.int64), np.arange(0, n, per, dtype=np.int64)])
        targets = rng.integers(0, 1000, ticks.shape[0]) * 2.0 / 1000.0 - 0.7
        return _shaker_kernel(np.ascontiguousarray(sig, dtype=np.float64), taps, ticks, targets, max(1, samples(float(ramp), sr)))
    return _per_channel(one, x)


@effect("delay", "Echo / feedback delay.",
        time=(0.25, "delay time in seconds"), feedback=(0.4, "0..0.95"), mix=(0.35, "wet amount"),
        pingpong=(False, "alternate left/right (stereo output)"), damp=(0.3, "lowpass in feedback loop 0..1"),
        tail=(None, "seconds of tail to append (auto if omitted)"))
def delay(x, sr=DEFAULT_SR, time=0.25, feedback=0.4, mix=0.35, pingpong=False, damp=0.3, tail=None):
    feedback = float(np.clip(feedback, 0, 0.95))
    if tail is None:
        tail = time * (1 + np.log(100) / max(-np.log(max(feedback, 0.01)), 0.05))
        tail = float(min(tail, 6.0))
    n_extra = samples(tail, sr)
    D = max(1, samples(time, sr))

    def one(sig):
        sig = np.concatenate([sig, np.zeros(n_extra)])
        wet = _comb(sig, D, feedback, damp) - sig  # remove the direct path from the comb
        return sig, wet

    if pingpong:
        mono = to_mono(x)
        dry, wet = one(mono)
        wet_r = np.concatenate([np.zeros(D), wet[:-D]])
        out = np.stack([dry + wet * mix, dry + wet_r * mix], axis=1)
        return out
    if x.ndim == 1:
        dry, wet = one(x)
        return dry * (1 - mix * 0.3) + wet * mix
    chans = [one(x[:, c]) for c in range(x.shape[1])]
    return np.stack([d * (1 - mix * 0.3) + w * mix for d, w in chans], axis=1)


def _mod_delay(sig, sr, base_ms, depth_ms, rate, shape="sine", phase0=0.0):
    n = sig.shape[0]
    d = (base_ms + depth_ms * (0.5 + 0.5 * _lfo(n / sr, rate, sr, shape, phase0=phase0)[:n])) * sr / 1000.0
    idx = np.arange(n) - d
    idx = np.clip(idx, 0, n - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = idx - i0
    return sig[i0] * (1 - frac) + sig[i1] * frac


@effect("chorus", "Thickening chorus (stereo output).",
        rate=(0.8, "LFO Hz"), depth=(3.0, "modulation depth ms"), mix=(0.5, "wet amount"), voices=(2, "1..4"))
def chorus(x, sr=DEFAULT_SR, rate=0.8, depth=3.0, mix=0.5, voices=2):
    mono = to_mono(x)
    l = np.zeros_like(mono)
    r = np.zeros_like(mono)
    for v in range(int(voices)):
        ph = v / max(1, voices)
        l += _mod_delay(mono, sr, 15 + 5 * v, depth, rate * (1 + 0.1 * v), phase0=ph)
        r += _mod_delay(mono, sr, 15 + 5 * v, depth, rate * (1 + 0.1 * v), phase0=ph + 0.25)
    l /= voices
    r /= voices
    dry = to_stereo(x)
    return dry * (1 - mix) + np.stack([l, r], axis=1) * mix


@effect("flanger", "Jet-like sweeping comb.", rate=(0.3, "LFO Hz"), depth=(2.0, "ms"), feedback=(0.5, "0..0.9"), mix=(0.5, "wet"))
def flanger(x, sr=DEFAULT_SR, rate=0.3, depth=2.0, feedback=0.5, mix=0.5):
    def one(sig):
        wet = _mod_delay(sig, sr, 1.0, depth, rate)
        # light feedback approximation: second pass
        wet2 = _mod_delay(wet, sr, 1.0, depth, rate)
        return sig * (1 - mix) + (wet + feedback * wet2) * mix
    return _per_channel(one, x)


@effect("phaser", "Sweeping allpass phaser.", rate=(0.5, "LFO Hz"), depth=(0.7, "0..1"), stages=(4, "allpass stages"), mix=(0.5, "wet"))
def phaser(x, sr=DEFAULT_SR, rate=0.5, depth=0.7, stages=4, mix=0.5):
    n = x.shape[0]
    center = 400 + 1600 * depth * (0.5 + 0.5 * _lfo(n / sr, rate, sr)[:n])

    def one(sig):
        wet = sig
        for _ in range(int(stages)):
            wet = F.biquad(wet, "allpass", center, sr, q=0.7, block=256)
        return sig * (1 - mix) + wet * mix
    return _per_channel(one, x)


@effect("vibrato", "Pitch wobble.", rate=(5.0, "Hz"), depth=(0.3, "semitones-ish (ms of modulation)"))
def vibrato(x, sr=DEFAULT_SR, rate=5.0, depth=0.3):
    return _per_channel(lambda s: _mod_delay(s, sr, 1.0, depth * 3.0, rate), x)


def _trem_lfo(n, sr, rate, shape, skew, duty, phase):
    """0..1 LFO; ``skew`` moves the crest of sine and triangle off the middle of the cycle, ``duty`` is the
    high share of the square."""
    p = (np.arange(n) / sr * float(np.clip(rate, 0.01, 20.0)) + float(phase)) % 1.0
    if shape == "square":
        return (p < float(np.clip(duty, 0.01, 0.99))).astype(float)
    k = 0.5 * (1.0 + float(np.clip(skew, -0.98, 0.98)))
    u = np.where(p < k, 0.5 * p / k, 0.5 + 0.5 * (p - k) / (1.0 - k))     # crest at p = k
    if shape == "triangle":
        return 1.0 - 2.0 * np.abs(u - 0.5)
    return 0.5 - 0.5 * np.cos(2 * np.pi * u)


@effect("tremolo", "Amplitude wobble.", rate=(6.0, "Hz, 0.01..20"), depth=(0.5, "0..1"), shape=("sine", "sine|triangle|square"),
        skew=(0.0, "-1..1: crest earlier (<0) or later (>0) in the cycle"), duty=(0.5, "square: share of the cycle at full level"),
        phase=(0.0, "0..1 start of the cycle"), spread=(0.0, "-1..1: the right channel runs this half-cycle ahead (stereo output)"))
def tremolo(x, sr=DEFAULT_SR, rate=6.0, depth=0.5, shape="sine", skew=0.0, duty=0.5, phase=0.0, spread=0.0):
    n = x.shape[0]
    if skew == 0.0 and duty == 0.5 and phase == 0.0 and spread == 0.0:
        g = 1 - depth * (0.5 + 0.5 * _lfo(n / sr, rate, sr, shape)[:n])
        return x * (g[:, None] if x.ndim == 2 else g)
    gl = 1 - depth * _trem_lfo(n, sr, rate, shape, skew, duty, phase)
    if spread == 0.0:
        return x * (gl[:, None] if x.ndim == 2 else gl)
    gr = 1 - depth * _trem_lfo(n, sr, rate, shape, skew, duty, phase + 0.5 * float(np.clip(spread, -1.0, 1.0)))
    st = to_stereo(x)
    return np.stack([st[:, 0] * gl, st[:, 1] * gr], axis=1)


def _adaa_tanh(x: np.ndarray) -> np.ndarray:
    """First-order antiderivative anti-aliasing of tanh: (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1]) with
    F = log cosh, or tanh of the midpoint when the step is tiny (Parker, Zavalishin & Le Bivic, DAFx-16;
    aanl.lib ADAA1). 5-8 dB less aliasing than plain tanh at drive 8-30, half a sample of delay."""
    x = np.asarray(x, float)
    xp = np.concatenate((np.zeros((1,) + x.shape[1:]), x[:-1]))
    d = x - xp
    ok = np.abs(d) > 1e-5
    y = np.tanh(0.5 * (x + xp))
    F = lambda v: np.logaddexp(v, -v) - math.log(2.0)        # noqa: E731
    y[ok] = (F(x[ok]) - F(xp[ok])) / d[ok]
    return y


@effect("distortion", "Waveshaping overdrive/fuzz.", drive=(4.0, "1..50"), mix=(1.0, "wet"), tone=(4000, "post lowpass Hz"),
        aa=("adaa1", "adaa1 | adaa2: antiderivative anti-aliasing order (2 aliases less, one sample of delay)"))
def distortion(x, sr=DEFAULT_SR, drive=4.0, mix=1.0, tone=4000, aa="adaa1"):
    if aa == "adaa2":
        from .dsp import _ANTI, adaa2
        wet = adaa2(np.asarray(x, float) * drive, np.tanh, *_ANTI["tanh"]) / np.tanh(min(drive, 20))
    else:
        wet = _adaa_tanh(x * drive) / np.tanh(min(drive, 20))
    if tone and tone < sr / 2:
        wet = F.lowpass(wet, tone, sr)
    return x * (1 - mix) + wet * mix


@effect("bitcrush", "Lo-fi bit depth and sample rate reduction.", bits=(8, "1..16"), rate=(11025, "effective sample rate Hz"),
        bits_end=(None, "bit depth reached at the end: a linear sweep, rounded to whole bits (jsfxr's bit crush sweep)"))
def bitcrush(x, sr=DEFAULT_SR, bits=8, rate=11025, bits_end=None):
    step = 2.0 ** (bits - 1)
    if bits_end is not None:
        step = 2.0 ** (np.clip(np.round(np.linspace(bits, bits_end, x.shape[0])), 1, 16) - 1)
        step = step[:, None] if x.ndim == 2 else step
    y = np.round(x * step) / step
    hold = max(1, int(round(sr / max(rate, 100))))
    if hold > 1:
        n = y.shape[0]
        idx = (np.arange(n) // hold) * hold
        y = y[idx]
    return y


@effect("lowpass", "Lowpass filter.", cutoff=(2000, "Hz"), q=(0.707, "resonance"), order=(1, "1=12dB/oct 2=24dB/oct"))
def lowpass(x, sr=DEFAULT_SR, cutoff=2000, q=0.707, order=1):
    return F.lowpass(x, cutoff, sr, q, order)


@effect("ladder", "Transistor or diode ladder low-pass: the Moog and TB-303 filters, saturating, self-oscillating at res 1.",
        cutoff=(1000, "Hz"), res=(0.5, "0..1 resonance"),
        kind=("moog", "moog|diode|lpf18 (3-pole, 18 dB/oct, Soundpipe)|huovilainen (2x oversampled, tanh per stage)"),
        drive=(1.0, "level into the saturation (lpf18: its distortion, 2 in Soundpipe)"),
        solver=("node", "moog: node (tanh of the linear estimate)|newton (delay-free loop)|adaa (antialiased tanh in the loop)"))
def ladder(x, sr=DEFAULT_SR, cutoff=1000, res=0.5, kind="moog", drive=1.0, solver="node"):
    return F.ladder(x, cutoff, sr, res, str(kind), drive, str(solver))


@effect("antisat_svf", "State-variable filter with a sinh antisaturator on its damping: past res 0.5 the damping turns negative and it self-oscillates at a level held by the antisaturator.",
        cutoff=(1000, "Hz"), res=(0.3, "0..1 (0.5 = undamped, above: self-oscillation)"),
        kind=("lowpass", "lowpass|bandpass|highpass"), a=(1.0, "antisaturator weight"), b=(1.0, "antisaturator knee"))
def antisat_svf(x, sr=DEFAULT_SR, cutoff=1000, res=0.3, kind="lowpass", a=1.0, b=1.0):
    return F.svf_antisat(x, str(kind), cutoff, sr, res, a, b)


@effect("random_allpass", "Cascade of second-order allpasses at random log-spaced frequencies (20-8000 Hz): same spectrum, scrambled phase (decorrelation, smeared transients).",
        n=(8, "sections"), seed=(0, "draw"), radius=(0.9, "pole radius (closer to 1: sharper phase jumps)"))
def random_allpass(x, sr=DEFAULT_SR, n=8, seed=0, radius=0.9):
    return F.random_allpass(x, sr, int(n), int(seed), radius)


@effect("highpass", "Highpass filter.", cutoff=(200, "Hz"), q=(0.707, "resonance"), order=(1, "1 or 2"))
def highpass(x, sr=DEFAULT_SR, cutoff=200, q=0.707, order=1):
    return F.highpass(x, cutoff, sr, q, order)


@effect("bandpass", "Bandpass filter.", center=(1000, "Hz"), q=(1.0, "narrowness"))
def bandpass(x, sr=DEFAULT_SR, center=1000, q=1.0):
    return F.bandpass(x, center, sr, q)


@effect("notch", "Notch filter.", center=(1000, "Hz"), q=(4.0, "narrowness"))
def notch(x, sr=DEFAULT_SR, center=1000, q=4.0):
    return F.notch(x, center, sr, q)


@effect("eq", "Peaking EQ band.", freq=(1000, "Hz"), gain=(6.0, "dB, negative to cut"), q=(1.0, "width"))
def eq(x, sr=DEFAULT_SR, freq=1000, gain=6.0, q=1.0):
    return F.peak(x, freq, sr, q, gain)


@effect("lowshelf", "Bass shelf.", freq=(200, "Hz"), gain=(6.0, "dB"))
def lowshelf(x, sr=DEFAULT_SR, freq=200, gain=6.0):
    return F.lowshelf(x, freq, sr, gain)


@effect("highshelf", "Treble shelf.", freq=(4000, "Hz"), gain=(6.0, "dB"))
def highshelf(x, sr=DEFAULT_SR, freq=4000, gain=6.0):
    return F.highshelf(x, freq, sr, gain)


def _envelope(x, sr, attack, release):
    """Attack / release follower of the absolute level (filters.follow); coefficient 1 - exp(-1 / (tau sr))."""
    a = 1.0 - math.exp(-1.0 / (max(attack, 1e-5) * sr))
    r = 1.0 - math.exp(-1.0 / (max(release, 1e-5) * sr))
    return F.follow(np.ascontiguousarray(np.abs(to_mono(x)), dtype=np.float64), a, r)


@effect("sweep", "Filter cutoff sweep over the sound, or following its level (`follow`: auto-wah).",
        kind=("lowpass", "lowpass|highpass|bandpass"), start=(300, "Hz"), end=(6000, "Hz"), q=(2.0, "resonance"),
        curve=("exp", "exp|lin"), follow=("", "'env' = the cutoff goes from start to end with the level of the sound instead of with time"),
        attack=(0.005, "follow: s"), release=(0.06, "follow: s"), sens=(1.0, "follow: level x sens = position 0..1"))
def sweep(x, sr=DEFAULT_SR, kind="lowpass", start=300, end=6000, q=2.0, curve="exp", follow="", attack=0.005, release=0.06, sens=1.0):
    """Follower times: DAFX 4.3 gives attack 1-30 ms, decay 30-100 ms; 5 and 60 ms picked there (B11)."""
    n = x.shape[0]
    if follow:
        u = np.clip(_envelope(x, sr, attack, release) * float(sens), 0.0, 1.0)
        return F.biquad(x, kind, start + (end - start) * u if curve == "lin" else start * (end / start) ** u, sr, q)
    if curve == "lin":
        fc = np.linspace(start, end, n)
    else:
        fc = start * (end / start) ** np.linspace(0, 1, n)
    return F.biquad(x, kind, fc, sr, q)


@numba.njit(cache=True)
def _wah_kernel(x, w, sr, M):
    """Faust vaeffects.lib crybaby (Julius Smith): Q = 2^(2(1 - w) + 1), fr = 450 2^(2.3 w), g = 0.1 4^w 44100 / sr,
    H = g (1 - z^-1) / (1 - 2 R cos(theta) z^-1 + R^2 z^-2), R = 1 - pi fr / (Q sr), theta = 2 pi fr / sr.
    Unit delays replaced by M samples: the M-fold wah of DAFX 2.4.1 (M band-passes spread over the spectrum, moving together)."""
    n = x.shape[0]
    y = np.zeros(n)
    k = 44100.0 / sr
    for i in range(n):
        q = 2.0 ** (2.0 * (1.0 - w[i]) + 1.0)
        fr = 450.0 * 2.0 ** (2.3 * w[i])
        R = 1.0 - math.pi * fr / (q * sr)
        v = x[i]
        if i >= M:
            v += 2.0 * R * math.cos(2.0 * math.pi * fr / sr) * y[i - M] - x[i - M]
        if i >= 2 * M:
            v -= R * R * y[i - 2 * M]
        y[i] = v
    return y * 0.1 * 4.0 ** w * k


@effect("wah", "Wah-wah: a resonant band-pass moved by a pedal position, an LFO or the level of the sound (auto-wah); folds > 1 gives the M-fold wah and bell effect.",
        mode=("lfo", "lfo | env (auto-wah) | pos (fixed pedal)"), rate=(0.5, "lfo: Hz"), pos=(0.5, "pos: pedal 0 (heel) .. 1 (toe)"),
        sens=(1.0, "env: level x sens = pedal position"), attack=(0.005, "env: s"), release=(0.06, "env: s"),
        folds=(1, "M: 1 = wah, 5-20 = M-fold wah, 100 = bell effect"), mix=(1.0, "wet 0..1"))
def wah(x, sr=DEFAULT_SR, mode="lfo", rate=0.5, pos=0.5, sens=1.0, attack=0.005, release=0.06, folds=1, mix=1.0):
    """DAFX Table 2.5: rate 0.5 Hz, M 5-20 and 100. Follower: Ch.4, attack 1-30 ms, decay 30-100 ms."""
    n = x.shape[0]
    if mode == "env":
        w = _envelope(x, sr, attack, release) * float(sens)
    elif mode == "pos":
        w = np.full(n, float(pos))
    elif mode == "lfo":
        w = 0.5 + 0.5 * _lfo(n / sr, rate, sr)[:n]
    else:
        raise ValueError("wah mode must be lfo|env|pos")
    w = np.ascontiguousarray(np.clip(w, 0.0, 1.0), dtype=np.float64)
    M = max(1, int(folds))
    return x * (1 - mix) + _per_channel(lambda s: _wah_kernel(np.ascontiguousarray(s, dtype=np.float64), w, float(sr), M), x) * mix


@effect("octaver", "Octave up and down on any sound, no pitch tracking: rectification for the octave up, a flip-flop on zero crossings for the octave down (best on single notes).",
        up=(0.5, "level of the octave up"), down=(0.5, "level of the octave down"), dry=(1.0, "level of the input"),
        tone=(1000.0, "low-pass on the octave down, Hz"))
def octaver(x, sr=DEFAULT_SR, up=0.5, down=0.5, dry=1.0, tone=1000.0):
    """DAFX 5.3.4: full-wave rectification gives even harmonics, a zero-crossing counter keeps one wave part out of two, low-pass leaves
    the fundamental. The flip-flop multiplies the signal, so the sign flips where it crosses zero, with no step. tone is a voicing choice."""
    def one(s):
        rising = (s[1:] > 0) & (s[:-1] <= 0)
        ff = 1.0 - 2.0 * (np.concatenate([[0], np.cumsum(rising)]) % 2)
        return dry * s + up * F.highpass(np.abs(s), 20.0, sr) + down * F.lowpass(s * ff, tone, sr)
    return _per_channel(one, np.asarray(x, float))


@effect("radio", "FM broadcast sound: phase rotator, AGC, 75 us pre-emphasis, clipper, de-emphasis, 15 kHz low-pass.",
        tau=(75e-6, "pre-emphasis time constant in s: 75e-6 USA, 50e-6 Europe"), drive=(1.0, "level into the clipper"))
def radio(x, sr=DEFAULT_SR, tau=75e-6, drive=1.0):
    """Katz, Mastering Audio, appendix 1: pre-emphasis +3 dB at 2.1 kHz, +17 dB at 15 kHz for 75 us, H = 1 + j 2 pi f tau; rotator of 4 poles
    at 200 Hz; AGC about 25 dB (threshold, ratio and times are voicing choices). The clipper is why bright sounds lose treble on the radio."""
    from .dsp import oversampled, steep_filter
    y = np.asarray(x, float)
    for _ in range(4):
        y = F.onepole(y, "allpass", 200.0, sr)
    y = compressor(y, sr, threshold=-25.0, ratio=4.0, attack=0.01, release=0.3)
    n = y.shape[0]
    Nf = n + 2 * sr // 10
    H = 1.0 + 2j * np.pi * np.fft.rfftfreq(Nf, 1.0 / sr) * float(tau)
    H = H if y.ndim == 1 else H[:, None]
    z = np.fft.irfft(np.fft.rfft(y, Nf, axis=0) * H, Nf, axis=0)
    z = oversampled(lambda v: np.clip(v, -1.0, 1.0), z * float(drive), 4)
    z = np.fft.irfft(np.fft.rfft(z, Nf, axis=0) / H, Nf, axis=0)[:n]
    return steep_filter(z, "lowpass", min(15000.0, 0.45 * sr), sr, 8, family="ellip")       # the appendix gives the 15 kHz, not the slope


@effect("muffle", "Sound heard through a wall / underwater / from another room.", amount=(0.7, "0..1, more = duller"))
def muffle(x, sr=DEFAULT_SR, amount=0.7):
    cutoff = 8000 * (1 - amount) ** 2 + 250
    y = F.lowpass(x, cutoff, sr, 0.9, order=2)
    return y * (1 - 0.4 * amount)


@effect("telephone", "Narrow band-limited radio/phone/walkie-talkie voice.", drive=(2.0, "saturation"), low=(300, "Hz"), high=(3400, "Hz"))
def telephone(x, sr=DEFAULT_SR, drive=2.0, low=300, high=3400):
    y = F.highpass(x, low, sr, order=2)
    y = F.lowpass(y, high, sr, order=2)
    return _adaa_tanh(y * drive) / np.tanh(drive)


@effect("underwater", "Submerged wobble + heavy lowpass.", amount=(0.8, "0..1"))
def underwater(x, sr=DEFAULT_SR, amount=0.8):
    y = F.lowpass(x, 300 + 1500 * (1 - amount), sr, 1.2, order=2)
    return _per_channel(lambda s: _mod_delay(s, sr, 5, 6 * amount, 0.9), y)


@effect("ringmod", "Ring modulation (metallic / robotic).", freq=(200, "carrier Hz"), mix=(1.0, "wet"),
        leak=(0.0, "carrier and input leaking through, as an unbalanced diode ring: wet = x c + leak (x + c); 0.5 = the classic ab + (a + b)/2"))
def ringmod(x, sr=DEFAULT_SR, freq=200, mix=1.0, leak=0.0):
    n = x.shape[0]
    car = np.sin(2 * np.pi * freq * np.arange(n) / sr)
    if x.ndim == 2:
        car = car[:, None]
    wet = x * car + float(leak) * (x + car) if leak else x * car
    return x * (1 - mix) + wet * mix


@numba.njit(cache=True)
def _decoupled_peak(xl, a_att, a_rel):
    """Smooth decoupled peak detector (Giannoulis, Massberg & Reiss, JAES 60(6) 2012, eq. 17): the release heads for the input, not for 0:
    y1 = max(x, aR y1 + (1 - aR) x); y = aA y + (1 - aA) y1."""
    y = np.empty_like(xl)
    y1 = 0.0
    yl = 0.0
    for n in range(xl.shape[0]):
        y1 = max(xl[n], a_rel * y1 + (1.0 - a_rel) * xl[n])
        yl = a_att * yl + (1.0 - a_att) * y1
        y[n] = yl
    return y


def _key_signal(key, n, sr):
    """Side-chain input: an array or a WAV path, mono, cut or padded to n samples."""
    if isinstance(key, (str, bytes)):
        key, ksr = read_wav(key)
        if ksr != sr:
            from .dsp import resample_sinc
            key = resample_sinc(key, ksr, sr)
    k = np.abs(to_mono(np.asarray(key, float)))[:n]
    return np.concatenate([k, np.zeros(n - k.shape[0])]) if k.shape[0] < n else k


# threshold, ratio, attack, release, auto makeup, mix. Katz, Mastering Audio ch. 10-11: mastering = ratio 1.5-3, threshold -20..-10, attack 100 ms,
# release 150-250 ms; invisible = ratio 1.01-1.1, threshold -30..-40 (its times taken from mastering); parallel (Hulse) = threshold -50,
# attack <= 1 ms, ratio 2.5, release 250-350 ms, compressed path 5-15 dB under the dry one: -10 dB is a mix of 0.316 / 1.316 = 0.24.
_COMP_PRESETS = {
    "mastering": (-15.0, 2.0, 0.1, 0.25, False, 1.0),
    "invisible": (-35.0, 1.1, 0.1, 0.25, False, 1.0),
    "parallel": (-50.0, 2.5, 0.001, 0.3, False, 0.24),
}


@effect("compressor", "Dynamics compressor (evens out level, adds punch): feed-forward, soft knee, smooth peak detector on the gain; side-chain and parallel mix.",
        threshold=(-18, "dB"), ratio=(4.0, "n:1"), attack=(0.005, "s"), release=(0.1, "s"), makeup=(0.0, "dB gain after (auto if 0 and auto_makeup)"),
        knee=(6.0, "dB width of the soft knee (0 = hard knee)"), mix=(1.0, "compressed amount against the dry sound 0..1: parallel compression"),
        auto_makeup=(True, "false: makeup is exactly what you set, 0 included"),
        key=(None, "side-chain: the level comes from this WAV file instead of the sound itself (ducking)"),
        preset=("", "mastering | invisible | parallel: sets threshold, ratio, attack, release, auto_makeup and mix"))
def compressor(x, sr=DEFAULT_SR, threshold=-18, ratio=4.0, attack=0.005, release=0.1, makeup=0.0, knee=6.0, mix=1.0, auto_makeup=True, key=None, preset=""):
    """Giannoulis, Massberg & Reiss 2012, the configuration of their conclusion: feed-forward, gain computer
    with a soft knee of width W (eq. 4) on the instantaneous level in dB, then the smooth decoupled peak
    detector (eq. 17) on the gain reduction, in the log domain. alpha = exp(-1 / (tau fs)) (eq. 7)."""
    if preset:
        threshold, ratio, attack, release, auto_makeup, mix = _COMP_PRESETS[preset]
    level = np.abs(to_mono(x)) if key is None else _key_signal(key, x.shape[0], sr)
    xg = 20 * np.log10(level + 1e-9)
    T, R, W = float(threshold), max(float(ratio), 1.0), max(float(knee), 0.0)
    over = xg - T
    yg = np.where(2 * over > W, T + over / R, xg)
    if W > 0:
        mid = np.abs(2 * over) <= W
        yg = np.where(mid, xg + (1 / R - 1) * (over + W / 2) ** 2 / (2 * W), yg)
    a_att = math.exp(-1 / (sr * max(attack, 1e-4)))
    a_rel = math.exp(-1 / (sr * max(release, 1e-3)))
    gr = _decoupled_peak(np.ascontiguousarray(xg - yg), a_att, a_rel)
    if makeup == 0.0 and auto_makeup and key is None:
        makeup = min(-T * (1 - 1 / R) * 0.5, 12)
    g = db_to_gain(makeup - gr)
    wet = x * (g[:, None] if x.ndim == 2 else g)
    return wet if mix >= 1.0 else x * (1.0 - mix) + wet * mix


@numba.njit(cache=True)
def _dyno_kernel(lv, thresh, above, below, at, rt):
    """ChucK Dyno (ugen_xxx.cpp, via lot R05): xd = xd (1 - rt) + at max(|in| - xd, 0), gain = (xd / thresh)^(slope - 1)."""
    g = np.empty_like(lv)
    xd = 0.0
    for i in range(lv.shape[0]):
        xd = xd * (1.0 - rt) + at * max(lv[i] - xd, 0.0)
        r = xd / thresh
        g[i] = r ** (above - 1.0) if xd > thresh else (r ** (below - 1.0) if xd > 0.0 else (1.0 if below == 1.0 else 0.0))
    return g


# slope above, slope below, threshold (linear), attack s, release s: ChucK Dyno presets, lot R05 (ugen_xxx.cpp:4297, :4484; not in this tree)
_DYNO = {"limit": (0.1, 1.0, 0.5, 0.005, 0.3), "compress": (0.5, 1.0, 0.5, 0.005, 0.5), "gate": (1.0, 1e8, 0.1, 0.011, 0.1),
         "expand": (2.0, 1.0, 0.5, 0.02, 0.4), "duck": (0.5, 1.0, 0.1, 0.01, 1.0)}


def _dyno(x, sr, preset, key=None, **over):
    above, below, thresh, attack, release = _DYNO[preset]
    above, below = over.get("above", above), over.get("below", below)
    thresh, attack, release = over.get("thresh", thresh), over.get("attack", attack), over.get("release", release)
    lv = np.abs(to_mono(x)) if key is None else _key_signal(key, x.shape[0], sr)
    g = _dyno_kernel(np.ascontiguousarray(lv, dtype=np.float64), float(thresh), float(above), float(below),
                     1.0 - math.exp(-1.0 / (attack * sr)), 1.0 - math.exp(-1.0 / (release * sr)))
    return x * (g[:, None] if x.ndim == 2 else g)


@effect("dyno", "Dynamics by a slope above and below one threshold (ChucK's Dyno): limit | compress | gate | expand | duck.",
        preset=("compress", "limit|compress|gate|expand|duck"), above=(None, "slope above the threshold"), below=(None, "slope below it"),
        thresh=(None, "threshold, linear"), attack=(None, "s"), release=(None, "s"), key=(None, "WAV file driving the gain (needed by duck)"))
def dyno(x, sr=DEFAULT_SR, preset="compress", above=None, below=None, thresh=None, attack=None, release=None, key=None):
    over = {k: v for k, v in dict(above=above, below=below, thresh=thresh, attack=attack, release=release).items() if v is not None}
    return _dyno(x, sr, preset, key, **over)


@effect("expander", "Downward expander: above `thresh` differences in level grow (ChucK's Dyno expand preset, slope 2 above 0.5).",
        thresh=(0.5, "threshold, linear"), slope=(2.0, "slope above it"), attack=(0.02, "s"), release=(0.4, "s"))
def expander(x, sr=DEFAULT_SR, thresh=0.5, slope=2.0, attack=0.02, release=0.4):
    return _dyno(x, sr, "expand", None, above=slope, thresh=thresh, attack=attack, release=release)


@effect("duck", "Duck this sound under another one (music under sfx): side-chain gain, ChucK's Dyno duck preset.",
        key=("", "WAV file that pushes this sound down"), thresh=(0.1, "level of the key where ducking starts, linear"), slope=(0.5, "slope above it"),
        attack=(0.01, "s"), release=(1.0, "s"))
def duck(x, sr=DEFAULT_SR, key="", thresh=0.1, slope=0.5, attack=0.01, release=1.0):
    if key is None or (isinstance(key, str) and not key):
        raise ValueError("duck needs key: the WAV file that pushes this sound down")
    return _dyno(x, sr, "duck", key, above=slope, thresh=thresh, attack=attack, release=release)


@numba.njit(cache=True)
def _gate_kernel(h, lt, ut, ht, att, rel):
    """DAFX M-file 5.4 (noisegt, after Bennett 1997): two thresholds, a hold, linear fades. One change: during the hold and the fade down the
    gain continues from where it is instead of jumping back to 1 (which clicks when the gate closes during a fade in)."""
    g = np.zeros_like(h)
    lc = uc = 0
    prev = 0.0
    for i in range(h.shape[0]):
        if h[i] <= lt or (h[i] < ut and lc > 0):
            lc += 1
            uc = 0
            if lc > ht:
                prev = 0.0 if lc > rel + ht else min(prev, 1.0 - (lc - ht) / rel)
        elif h[i] >= ut or (h[i] > lt and uc > 0):
            uc += 1
            prev = max(uc / att, prev) if prev < 1.0 else 1.0
            lc = 0
        else:
            lc = uc = 0
        g[i] = min(prev, 1.0)
    return g


@effect("gate", "Noise gate with hysteresis and hold: opens above open_db, closes below close_db once the hold has run out, with linear fades.",
        open_db=(-40.0, "dB relative to the peak of the sound (or the key): opens above it"), close_db=(-45.0, "closes below it, under open_db"),
        hold=(0.05, "s below the threshold before fading"), attack=(0.001, "fade in, s"), release=(0.1, "fade out, s"),
        detect=(0.002, "time constant of the level detector, s"), key=(None, "WAV file that opens the gate instead of the sound"))
def gate(x, sr=DEFAULT_SR, open_db=-40.0, close_db=-45.0, hold=0.05, attack=0.001, release=0.1, detect=0.002, key=None):
    """DAFX 5.2.3: envelope = two-pole lowpass of |x|, normalised to its maximum. Default values (the M-file has none)."""
    lv = np.abs(to_mono(x)) if key is None else _key_signal(key, x.shape[0], sr)
    a = math.exp(-1.0 / (float(detect) * sr))
    h = lfilter([(1.0 - a) ** 2], [1.0, -2.0 * a, a * a], lv)
    h = h / max(float(h.max()), 1e-12)
    ut, lt = db_to_gain(open_db), db_to_gain(min(close_db, open_db))
    g = _gate_kernel(np.ascontiguousarray(h, dtype=np.float64), float(lt), float(ut), samples(hold, sr), max(samples(attack, sr), 1), max(samples(release, sr), 1))
    return x * (g[:, None] if x.ndim == 2 else g)


@numba.njit(cache=True)
def _limiter_env(gr, a_fast, a_slow_up, a_slow):
    """Two detectors on the gain reduction in dB, the larger one wins. The fast one follows at once and lets go in `a_fast`: an isolated transient
    is released in about 2 ms. The slow one charges only under a sustained overload and lets go in `a_slow`."""
    y = np.empty_like(gr)
    f = s = 0.0
    for i in range(gr.shape[0]):
        f = max(gr[i], a_fast * f)
        s = a_slow_up * s + (1.0 - a_slow_up) * gr[i] if gr[i] > s else a_slow * s
        y[i] = max(f, s)
    return y


@effect("limiter", "Hard ceiling: look-ahead gain with a double release (fast for an isolated peak, slow under a sustained overload).",
        ceiling=(-0.5, "dB"), lookahead=(0.0015, "s"), release=(0.1, "s, under a sustained overload"),
        true_peak=(False, "true = the level includes the peaks between samples (4x oversampled)"))
def limiter(x, sr=DEFAULT_SR, ceiling=-0.5, lookahead=0.0015, release=0.1, true_peak=False):
    """Katz, Mastering Audio ch. 10: auto-release about 2 ms on a transient, 50-150 ms on sustained limiting; a look-ahead of 1.5 ms (lot B14).
    Charge time of the slow detector 80 ms: the duration threshold Katz reports for the Weiss DS1's double release.
    The gain is a minimum over +-L smoothed over +-L, so it never exceeds what any sample within reach needs: no clipping left to alias."""
    c = db_to_gain(ceiling)
    x = np.asarray(x, float)
    peak = np.abs(x) if x.ndim == 1 else np.abs(x).max(axis=1)
    if true_peak:
        over = np.abs(resample_poly(x, 4, 1, axis=0))
        over = over if over.ndim == 1 else over.max(axis=1)
        peak = np.maximum(peak, over[: 4 * peak.shape[0]].reshape(peak.shape[0], 4).max(axis=1))
    need = np.minimum(1.0, c / (peak + 1e-12))
    L = max(1, int(round(float(lookahead) * sr / 2.0)))
    hold = minimum_filter1d(need, 2 * L + 1, mode="nearest")
    gr = _limiter_env(np.ascontiguousarray(-np.log(hold)), math.exp(-1.0 / (0.002 * sr)), math.exp(-1.0 / (0.08 * sr)), math.exp(-1.0 / (max(float(release), 1e-3) * sr)))
    g = uniform_filter1d(np.exp(-gr), 2 * L + 1, mode="nearest")
    y = x * (g[:, None] if x.ndim == 2 else g)
    if np.max(np.abs(y)) > c:                                          # what is left to clip is clipped 4x oversampled
        from .dsp import oversampled
        y = oversampled(lambda v: np.clip(v, -c, c), y, 4)
    return np.clip(y, -c, c)


@effect("gain", "Volume change.", db=(0.0, "decibels"))
def gain(x, sr=DEFAULT_SR, db=0.0):
    return x * db_to_gain(db)


@effect("normalize", "Peak-normalize.", level=(-1.0, "target peak dB"))
def normalize(x, sr=DEFAULT_SR, level=-1.0):
    return _normalize(x, level)


def _itd_read(s, delay):
    """s delayed by `delay` samples, per sample, linear interpolation."""
    return np.interp(np.arange(s.shape[0]) - delay, np.arange(s.shape[0]), s, left=0.0)


@effect("pan", "Stereo position (produces stereo).", pos=(0.0, "-1 left .. 1 right"), itd=(0.0, "ms the far ear hears late at full position, up to 0.6"))
def pan(x, sr=DEFAULT_SR, pos=0.0, itd=0.0):
    """Interaural delay at most 0.6 ms: Fastl & Zwicker (lot B08)."""
    pos = float(np.clip(pos, -1, 1))
    theta = (pos + 1) * np.pi / 4
    l, r = np.cos(theta), np.sin(theta)
    mono = to_mono(x)
    d = float(np.clip(itd, 0.0, 0.6)) * 1e-3 * sr * pos
    return np.stack([_itd_read(mono, max(d, 0.0)) * l, _itd_read(mono, max(-d, 0.0)) * r], axis=1)


@effect("autopan", "Moving stereo position; `rear` walks the sound around the head, duller behind.", rate=(1.0, "Hz"), depth=(1.0, "0..1"),
        itd=(0.0, "ms the far ear hears late at full position, up to 0.6"), rear=(False, "low-pass that closes behind the listener"))
def autopan(x, sr=DEFAULT_SR, rate=1.0, depth=1.0, itd=0.0, rear=False):
    """rear: rjlib e_circlepan, a Q 1 filter at 1900 +- 2000 Hz (the lot notes the roles in that patch are uncertain)."""
    n = x.shape[0]
    pos = _lfo(n / sr, rate, sr)[:n] * depth
    theta = (pos + 1) * np.pi / 4
    mono = to_mono(x)
    if rear:
        front = _lfo(n / sr, rate, sr, phase0=0.25)[:n]
        mono = F.svf(mono, "lowpass", np.maximum(1900.0 + 2000.0 * front, 100.0), sr, 1.0)
    d = float(np.clip(itd, 0.0, 0.6)) * 1e-3 * sr * pos
    return np.stack([_itd_read(mono, np.maximum(d, 0.0)) * np.cos(theta), _itd_read(mono, np.maximum(-d, 0.0)) * np.sin(theta)], axis=1)


@effect("balance", "Pan a stereo sound without collapsing it: the channel on the losing side folds into the other.", pos=(0.0, "-1 left .. 1 right; 0 leaves it untouched"))
def balance(x, sr=DEFAULT_SR, pos=0.0):
    """Bole, Faust Tutorial 2: OutL = sqrt(min(1, 2 - 2c)) l + sqrt(max(0, 1 - 2c)) r, OutR = sqrt(max(0, 2c - 1)) l + sqrt(min(1, 2c)) r, c = (pos + 1) / 2."""
    y = to_stereo(np.asarray(x, float))
    c = (float(np.clip(pos, -1, 1)) + 1.0) / 2.0
    l, r = y[:, 0], y[:, 1]
    return np.stack([math.sqrt(min(1.0, 2.0 - 2.0 * c)) * l + math.sqrt(max(0.0, 1.0 - 2.0 * c)) * r,
                     math.sqrt(max(0.0, 2.0 * c - 1.0)) * l + math.sqrt(min(1.0, 2.0 * c)) * r], axis=1)


@effect("exp_delay", "Echoes that bunch up or spread out, like a bouncing ball: N taps at a power law of time and level.",
        length=(2.0, "s of the last tap"), taps=(20, "N"), durcurve=(1.0, "tap time = length (i/N)^durcurve; < 1 bunches them at the end"),
        ampcurve=(1.0, "tap level follows (i/N)^ampcurve, normalised to a sum of 1; 0 = all equal"), mix=(0.5, "wet"))
def exp_delay(x, sr=DEFAULT_SR, length=2.0, taps=20, durcurve=1.0, ampcurve=1.0, mix=0.5):
    """ChucK ExpDelay: t_i = L (i/N)^durcurve, a_i = (i/N)^ampcurve / sum, i = 0..N-1; N 20, L 2 s, curves 1."""
    N = max(int(taps), 1)
    u = np.arange(N) / N
    t = np.round(float(length) * u ** float(durcurve) * sr).astype(int)
    a = u ** float(ampcurve)
    a = a / max(float(a.sum()), 1e-12)
    n = x.shape[0]
    wet = np.zeros((n + int(t[-1]),) + x.shape[1:])
    for ti, ai in zip(t, a):
        wet[ti: ti + n] += ai * x
    dry = np.concatenate([x, np.zeros((int(t[-1]),) + x.shape[1:])], axis=0)
    return dry * (1 - mix) + wet * mix


@effect("width", "Stereo widening via short Haas delay (right copy 0-12 ms late: heard as one sound from the earlier, left side, not as an echo; the precedence holds for 1-30 ms, beyond it the copy splits off), or mid/side (produces stereo).", amount=(0.5, "0..1"),
        mode=("haas", "haas | ms: mid/side, amount 0 = mono, 0.5 = unchanged, 1 = side doubled; a mono input stays mono"))
def width(x, sr=DEFAULT_SR, amount=0.5, mode="haas"):
    if mode == "ms":
        st = to_stereo(x)
        m = 0.5 * (st[:, 0] + st[:, 1])
        sd = (st[:, 0] - st[:, 1]) * float(np.clip(amount, 0.0, 1.0))
        return np.stack([m + sd, m - sd], axis=1)
    if mode != "haas":
        raise ValueError("mode must be haas|ms")
    mono = to_mono(x)
    d = int(sr * 0.012 * amount)
    r = np.concatenate([np.zeros(d), mono[: mono.shape[0] - d]]) if d > 0 else mono
    return np.stack([mono, r], axis=1)


@effect("fade", "Fade in and/or out.", **{"in": (0.0, "seconds"), "out": (0.0, "seconds")})
def fade(x, sr=DEFAULT_SR, **kw):
    fi, fo = float(kw.get("in", 0.0)), float(kw.get("out", 0.0))
    y = x.copy()
    n = y.shape[0]
    if fi > 0:
        k = min(n, samples(fi, sr))
        ramp = np.linspace(0, 1, k)
        y[:k] *= ramp[:, None] if y.ndim == 2 else ramp
    if fo > 0:
        k = min(n, samples(fo, sr))
        ramp = np.linspace(1, 0, k)
        y[n - k:] *= ramp[:, None] if y.ndim == 2 else ramp
    return y


@effect("loop", "Make the file seamlessly loopable: the tail is crossfaded into the head and cut off, so `loop=true` playback has no click or gap.", crossfade=(0.05, "seconds of overlap"))
def loop(x, sr=DEFAULT_SR, crossfade=0.05):
    n = x.shape[0]
    k = min(samples(float(crossfade), sr), n // 2)
    if k <= 0:
        return x
    y = x[: n - k].copy()
    tail = x[n - k:]
    t = np.linspace(0, 1, k)
    fin, fout = np.sqrt(t), np.sqrt(1 - t)  # equal-power
    if y.ndim == 2:
        fin, fout = fin[:, None], fout[:, None]
    y[:k] = y[:k] * fin + tail * fout
    return y


@effect("reverse", "Play backwards.")
def reverse(x, sr=DEFAULT_SR):
    return x[::-1].copy()


@effect("speed", "Resample: >1 faster and higher pitch, <1 slower and lower (like a tape).", factor=(1.0, "ratio"))
def speed(x, sr=DEFAULT_SR, factor=1.0):
    n = x.shape[0]
    m = max(1, int(n / factor))
    t_out = np.linspace(0, n - 1, m)
    idx = np.arange(n)
    return _per_channel(lambda s: np.interp(t_out, idx, s), x)


@effect("pitch", "Pitch shift in semitones without changing length (granular, cheap).", semitones=(0.0, "+/- semitones"))
def pitch(x, sr=DEFAULT_SR, semitones=0.0):
    ratio = 2 ** (semitones / 12)
    if abs(ratio - 1) < 1e-6:
        return x
    grain = int(sr * 0.05)
    hop = grain // 2
    win = np.hanning(grain)
    n = x.shape[0]

    def one(sig):
        out = np.zeros(n + grain)
        norm = np.zeros(n + grain)
        for start in range(0, n, hop):
            src_len = int(grain * ratio)
            src = sig[start: start + src_len]
            if src.shape[0] < 2:
                break
            g = np.interp(np.linspace(0, src.shape[0] - 1, grain), np.arange(src.shape[0]), src)
            out[start: start + grain] += g * win
            norm[start: start + grain] += win
        return (out / np.maximum(norm, 1e-6))[:n]
    return _per_channel(one, x)


@effect("trim", "Remove silent tail.", threshold=(-70, "dB"))
def trim(x, sr=DEFAULT_SR, threshold=-70):
    return _trim_tail(x, threshold, sr=sr)


@effect("stutter", "Repeat a slice N times (glitch).", start=(0.0, "seconds"), length=(0.05, "seconds"), repeats=(4, "count"))
def stutter(x, sr=DEFAULT_SR, start=0.0, length=0.05, repeats=4):
    s = samples(start, sr)
    e = min(x.shape[0], s + samples(length, sr))
    seg = x[s:e]
    return np.concatenate([x[:s]] + [seg] * int(repeats) + [x[e:]], axis=0)


@effect("mono", "Collapse to mono.")
def mono(x, sr=DEFAULT_SR):
    return to_mono(x)


def apply_chain(x: np.ndarray, chain: list[dict], sr: int = DEFAULT_SR) -> np.ndarray:
    """chain: [{"type": "reverb", "mix": 0.3}, ...]"""
    for step in chain:
        step = dict(step)
        name = step.pop("type", None) or step.pop("fx", None)
        if name not in REGISTRY:
            raise ValueError(f"unknown effect {name!r}; run `genny list fx`")
        x = REGISTRY[name]["fn"](x, sr, **step)
    return x


def parse_fx_arg(text: str) -> dict:
    """'reverb:mix=0.3,size=0.8' or 'reverb' -> {'type': 'reverb', 'mix': 0.3, 'size': 0.8}"""
    name, _, rest = text.partition(":")
    out = {"type": name.strip()}
    if rest:
        for kv in rest.split(","):
            if not kv.strip():
                continue
            k, _, v = kv.partition("=")
            out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(v: str):
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


# ---------------------------------------------------------------- KasFilter, phase bashing, follower, echo audibility
@numba.njit(cache=True)
def _kas_kernel(x, inc, res, accent):
    n = x.shape[0]
    y = np.empty(n)
    ph = 0.0
    a = 0.0
    b = 0.0
    prev = 0.0
    for i in range(n):
        old = ph
        ph += inc[i]
        if ph >= 2.0 * math.pi:
            ph -= 2.0 * math.pi
            fr = ph / max(inc[i], 1e-12)
            xi = x[i] - fr * (x[i] - prev)
            b = xi + res * a
            if b > 1.0:
                b = 1.0
            elif b < -1.0:
                b = -1.0
        elif old < math.pi <= ph:
            fr = (ph - math.pi) / max(inc[i], 1e-12)
            a = x[i] - fr * (x[i] - prev)
        c = math.cos(ph)
        ac = abs(c)
        mix = 0.5 + 0.5 * c * (ac + accent * (1.0 - ac))
        y[i] = a * mix + b * (1.0 - mix)
        prev = x[i]
    return y


@effect("kasfilter", "Undersampling filter: the input is sampled twice per cycle of `freq` into two held values that cross-fade on a cosine, the second fed back with `res` (gritty, resonant, aliasing on purpose).",
        freq=(440.0, "Hz"), res=(0.85, "-1..1 feedback (negative = hollow)"), accent=(0.75, "0..1 sharpness of the cross-fade (0 = abrupt, 1 = plain cosine)"))
def kasfilter(x, sr=DEFAULT_SR, freq=440.0, res=0.85, accent=0.75):
    """Phase increment pi f / sr; A held at pi, B at the wrap, each by linear interpolation at the exact crossing;
    B = input + res A, kept in +-1; mix = 0.5 + 0.5 c (|c| + accent (1 - |c|)), c = cos(phase); out = A mix + B (1 - mix)."""
    f = np.asarray(freq, float)
    n = x.shape[0]
    inc = np.full(n, math.pi * float(f) / sr) if f.ndim == 0 else math.pi * np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f)), f) / sr
    r = float(np.clip(res, -1.0, 1.0))
    acc = float(np.clip(accent, 0.0, 1.0))
    return _per_channel(lambda s: _kas_kernel(np.ascontiguousarray(s, dtype=np.float64), inc, r, acc), x)


@effect("phase_bash", "Phase-bashed wavetables: each window of the sound becomes one cycle of a zero-phase wavetable (its spectrum, phases set so the cycle is centred), played at `pitch` and cross-faded window to window: the timbre of the source on a steady pitch, without vibrato.",
        pitch=(110.0, "Hz of the oscillator"), window=(2048, "samples per window (power of 2)"), mix=(1.0, "wet"))
def phase_bash(x, sr=DEFAULT_SR, pitch=110.0, window=2048, mix=1.0):
    """Each window (Hann) -> FFT -> |S[k]| (-1)^k -> IFFT: a real cycle, symmetric about its centre. Windows
    advance every N/4 samples; the oscillator reads the two tables around the current time and cross-fades them."""
    N = 1 << int(np.clip(round(math.log2(max(int(window), 64))), 6, 15))
    H = N // 4
    w = np.hanning(N)
    sign = (-1.0) ** np.arange(N // 2 + 1)

    def one(s):
        n = s.shape[0]
        pad = np.concatenate([s, np.zeros(N)])
        tabs = []
        for st in range(0, n, H):
            T = np.fft.irfft(np.abs(np.fft.rfft(pad[st:st + N] * w)) * sign, N)
            tabs.append(T)
        tabs = np.array(tabs)
        pos = np.arange(n) / H
        j = np.minimum(pos.astype(int), len(tabs) - 1)
        j1 = np.minimum(j + 1, len(tabs) - 1)
        fr = pos - j
        ph = (np.cumsum(np.full(n, float(np.clip(pitch, 10.0, sr / 4))) / sr) % 1.0) * N
        i0 = ph.astype(int) % N
        i1 = (i0 + 1) % N
        u = ph - np.floor(ph)
        ra = tabs[j, i0] * (1 - u) + tabs[j, i1] * u
        rb = tabs[j1, i0] * (1 - u) + tabs[j1, i1] * u
        y = ra * (1 - fr) + rb * fr
        g = np.sqrt(np.mean(s * s) / (np.mean(y * y) + 1e-20))
        return y * g

    wet = _per_channel(one, np.asarray(x, float))
    return x * (1 - mix) + wet * mix


def envelope_follower(x, sr=DEFAULT_SR, attack=0.01, release=0.1, mode="peak"):
    """Control signal: out += (in > out ? A : R) (in - out), A = 1 - exp(-1 / (sr attack)), R likewise; ``peak``
    follows |x|, ``rms`` is the square root of the follower run on x^2."""
    a = 1.0 - math.exp(-1.0 / (max(float(attack), 1e-5) * sr))
    r = 1.0 - math.exp(-1.0 / (max(float(release), 1e-5) * sr))
    m = to_mono(np.asarray(x, float))
    if mode == "rms":
        return np.sqrt(np.maximum(F.follow(np.ascontiguousarray(m * m), a, r), 0.0))
    return F.follow(np.ascontiguousarray(np.abs(m)), a, r)


ECHO_EC = {"speech": (2.0 / 3.0, 0.009, 1.0), "music": (1.0, 0.014, 1.8)}   # exponent n, window dtau s, critical EC


def echo_threshold_db(delay_ms: float) -> float:
    """Level (dB re the direct sound) above which a single reflection arriving ``delay_ms`` (ms) late is perceived
    at all (speech): dL = -0.575 t0 - 6, falling with the delay."""
    return -0.575 * float(delay_ms) - 6.0


def haas_delay_ms(syllables_per_s: float = 5.0) -> float:
    """Delay (ms) past which a reflection of speech reads as an echo: about 93 ms for slow speech, 43 ms for fast
    (linear between 3 and 9 syllables a second)."""
    return float(np.interp(float(syllables_per_s), (3.0, 9.0), (93.0, 43.0)))


def echo_criterion(ir, sr=DEFAULT_SR, signal="speech") -> tuple[float, bool]:
    """Dietsch-Kraak echo criterion on an impulse response: build-up time t_s(tau) = int t |p|^n / int |p|^n up to
    tau, EC = max of its slope over dtau; an echo is heard (by half of listeners) when EC > EC_crit. Returns (EC, echo)."""
    nexp, dtau, crit = ECHO_EC["music" if signal == "music" else "speech"]
    p = np.abs(to_mono(np.asarray(ir, float))) ** nexp
    i0 = int(np.argmax(p > 0.01 * p.max()))
    p = p[i0:]
    t = np.arange(len(p)) / sr
    ts = np.cumsum(t * p) / np.maximum(np.cumsum(p), 1e-20)
    k = max(1, int(round(dtau * sr)))
    ec = float(np.max((ts[k:] - ts[:-k]) / dtau)) if len(ts) > k else 0.0
    return ec, ec > crit
