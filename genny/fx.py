"""Effects. Every effect is registered with its parameters so `genny list fx` is self-documenting.

Signature convention: fn(x, sr, **params) -> ndarray. Works on mono (n,) or stereo (n,2).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from . import filters as F
from .core import DEFAULT_SR, db_to_gain, is_stereo, normalize as _normalize, samples, to_mono, to_stereo, trim_tail as _trim_tail
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
        width=(1.0, "stereo width 0..1 (stereo output when > 0)"))
def reverb(x, sr=DEFAULT_SR, mix=0.3, size=0.7, damp=0.4, predelay=0.0, tail=None, width=1.0):
    size = float(np.clip(size, 0.0, 1.0))
    fb = 0.7 + 0.28 * size
    if tail is None:
        tail = 0.4 + 2.6 * size
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
            wet += _comb(sig, int((c + offset) * scale * (0.6 + 0.4 * size)), fb, damp)
        wet /= len(combs)
        for a in aps:
            wet = _allpass(wet, int((a + offset) * scale), 0.5)
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


@effect("tremolo", "Amplitude wobble.", rate=(6.0, "Hz"), depth=(0.5, "0..1"), shape=("sine", "sine|triangle|square"))
def tremolo(x, sr=DEFAULT_SR, rate=6.0, depth=0.5, shape="sine"):
    n = x.shape[0]
    g = 1 - depth * (0.5 + 0.5 * _lfo(n / sr, rate, sr, shape)[:n])
    return x * (g[:, None] if x.ndim == 2 else g)


@effect("distortion", "Waveshaping overdrive/fuzz.", drive=(4.0, "1..50"), mix=(1.0, "wet"), tone=(4000, "post lowpass Hz"))
def distortion(x, sr=DEFAULT_SR, drive=4.0, mix=1.0, tone=4000):
    wet = np.tanh(x * drive) / np.tanh(min(drive, 20))
    if tone and tone < sr / 2:
        wet = F.lowpass(wet, tone, sr)
    return x * (1 - mix) + wet * mix


@effect("bitcrush", "Lo-fi bit depth and sample rate reduction.", bits=(8, "1..16"), rate=(11025, "effective sample rate Hz"))
def bitcrush(x, sr=DEFAULT_SR, bits=8, rate=11025):
    step = 2 ** (bits - 1)
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


@effect("sweep", "Filter cutoff sweep over the sound (auto-wah / opening filter).",
        kind=("lowpass", "lowpass|highpass|bandpass"), start=(300, "Hz"), end=(6000, "Hz"), q=(2.0, "resonance"),
        curve=("exp", "exp|lin"))
def sweep(x, sr=DEFAULT_SR, kind="lowpass", start=300, end=6000, q=2.0, curve="exp"):
    n = x.shape[0]
    if curve == "lin":
        fc = np.linspace(start, end, n)
    else:
        fc = start * (end / start) ** np.linspace(0, 1, n)
    return F.biquad(x, kind, fc, sr, q)


@effect("muffle", "Sound heard through a wall / underwater / from another room.", amount=(0.7, "0..1, more = duller"))
def muffle(x, sr=DEFAULT_SR, amount=0.7):
    cutoff = 8000 * (1 - amount) ** 2 + 250
    y = F.lowpass(x, cutoff, sr, 0.9, order=2)
    return y * (1 - 0.4 * amount)


@effect("telephone", "Narrow band-limited radio/phone/walkie-talkie voice.", drive=(2.0, "saturation"), low=(300, "Hz"), high=(3400, "Hz"))
def telephone(x, sr=DEFAULT_SR, drive=2.0, low=300, high=3400):
    y = F.highpass(x, low, sr, order=2)
    y = F.lowpass(y, high, sr, order=2)
    return np.tanh(y * drive) / np.tanh(drive)


@effect("underwater", "Submerged wobble + heavy lowpass.", amount=(0.8, "0..1"))
def underwater(x, sr=DEFAULT_SR, amount=0.8):
    y = F.lowpass(x, 300 + 1500 * (1 - amount), sr, 1.2, order=2)
    return _per_channel(lambda s: _mod_delay(s, sr, 5, 6 * amount, 0.9), y)


@effect("ringmod", "Ring modulation (metallic / robotic).", freq=(200, "carrier Hz"), mix=(1.0, "wet"))
def ringmod(x, sr=DEFAULT_SR, freq=200, mix=1.0):
    n = x.shape[0]
    car = np.sin(2 * np.pi * freq * np.arange(n) / sr)
    if x.ndim == 2:
        car = car[:, None]
    return x * (1 - mix) + x * car * mix


@effect("compressor", "Dynamics compressor (evens out level, adds punch).",
        threshold=(-18, "dB"), ratio=(4.0, "n:1"), attack=(0.005, "s"), release=(0.1, "s"), makeup=(0.0, "dB gain after (auto if 0)"))
def compressor(x, sr=DEFAULT_SR, threshold=-18, ratio=4.0, attack=0.005, release=0.1, makeup=0.0):
    mono = np.abs(to_mono(x)) + 1e-9
    a_coef = np.exp(-1 / (sr * max(attack, 1e-4)))
    r_coef = np.exp(-1 / (sr * max(release, 1e-3)))
    # two one-pole followers; take max of fast attack and slow release (cheap approximation)
    fast = lfilter([1 - a_coef], [1, -a_coef], mono)
    slow = lfilter([1 - r_coef], [1, -r_coef], mono)
    envl = np.maximum(fast, slow)
    lvl_db = 20 * np.log10(envl)
    over = np.maximum(lvl_db - threshold, 0)
    gain_db = -over * (1 - 1 / ratio)
    if makeup == 0.0:
        makeup = min(-threshold * (1 - 1 / ratio) * 0.5, 12)
    g = db_to_gain(gain_db + makeup)
    return x * (g[:, None] if x.ndim == 2 else g)


@effect("limiter", "Hard ceiling with lookahead smoothing.", ceiling=(-0.5, "dB"))
def limiter(x, sr=DEFAULT_SR, ceiling=-0.5):
    c = db_to_gain(ceiling)
    mono = np.abs(to_mono(x)) + 1e-9
    win = max(1, int(sr * 0.005))
    k = np.ones(win) / win
    env = np.maximum(np.convolve(mono, k, mode="same"), mono)
    g = np.minimum(1.0, c / env)
    return np.clip(x * (g[:, None] if x.ndim == 2 else g), -c, c)


@effect("gain", "Volume change.", db=(0.0, "decibels"))
def gain(x, sr=DEFAULT_SR, db=0.0):
    return x * db_to_gain(db)


@effect("normalize", "Peak-normalize.", level=(-1.0, "target peak dB"))
def normalize(x, sr=DEFAULT_SR, level=-1.0):
    return _normalize(x, level)


@effect("pan", "Stereo position (produces stereo).", pos=(0.0, "-1 left .. 1 right"))
def pan(x, sr=DEFAULT_SR, pos=0.0):
    pos = float(np.clip(pos, -1, 1))
    theta = (pos + 1) * np.pi / 4
    l, r = np.cos(theta), np.sin(theta)
    mono = to_mono(x)
    return np.stack([mono * l, mono * r], axis=1)


@effect("autopan", "Moving stereo position.", rate=(1.0, "Hz"), depth=(1.0, "0..1"))
def autopan(x, sr=DEFAULT_SR, rate=1.0, depth=1.0):
    n = x.shape[0]
    pos = _lfo(n / sr, rate, sr)[:n] * depth
    theta = (pos + 1) * np.pi / 4
    mono = to_mono(x)
    return np.stack([mono * np.cos(theta), mono * np.sin(theta)], axis=1)


@effect("width", "Stereo widening via short Haas delay (produces stereo).", amount=(0.5, "0..1"))
def width(x, sr=DEFAULT_SR, amount=0.5):
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
