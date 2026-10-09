"""Game sound-effect generators. Each takes (sr, **params) and returns a mono buffer."""
from __future__ import annotations

import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, mix, samples
from .env import adsr, apply, perc, lfo
from .notes import to_hz

REGISTRY: dict[str, dict] = {}


def sfx(name: str, desc: str, **params):
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "desc": desc, "params": params}
        return fn
    return deco


def _dec(n, tau, sr):
    return np.exp(-np.arange(n) / (tau * sr))


def _round(x, freq, sr, harm=5.0, lo=2000.0, hi=4000.0):
    """Rounds the corners of a square/saw so a beep is clear without being piercing: a lowpass that
    keeps `harm` harmonics of the tone but stays inside [lo, hi] Hz. `freq` may be a sweep array."""
    fc = np.clip(np.asarray(freq, dtype=np.float64) * harm, lo, hi)
    return F.biquad(x, "lowpass", float(fc) if fc.ndim == 0 else fc, sr, q=0.707)


@sfx("beep", "Simple UI beep.", freq=(880, "Hz or note name (110, '110hz' or 'A2')"), dur=(0.12, "s"), wave=("sine", "sine|square|triangle|saw|pulse"))
def beep(sr=DEFAULT_SR, freq=880, dur=0.12, wave="sine"):
    f = to_hz(freq)
    n = samples(dur + 0.02, sr)
    x = O.osc(wave, f, n, sr, width=0.3)
    if wave != "sine":
        x = _round(x, f, sr) * 0.6
    return apply(x, adsr(dur, 0.004, 0.0, 1.0, 0.02, sr)) * 0.8


@sfx("blip", "Very short high blip (cursor move, tick).", freq=(1400, "Hz or note name (110, '110hz' or 'A2')"), dur=(0.05, "s"))
def blip(sr=DEFAULT_SR, freq=1400, dur=0.05):
    f = to_hz(freq)
    n = samples(dur, sr)
    return _round(O.square(f, n, sr), f, sr, lo=2400, hi=4200) * perc(dur, 0.001, sr, curve=1.5) * 0.4


@sfx("click", "Mechanical click / tick.", tone=(3000, "Hz"))
def click(sr=DEFAULT_SR, tone=3000):
    n = samples(0.03, sr)
    x = F.bandpass(O.white(n, seed=21), tone, sr, 1.5) * _dec(n, 0.003, sr)
    x += O.sine(tone / 2, n, sr) * _dec(n, 0.002, sr) * 0.5
    return x * 1.5


@sfx("pop", "Soft bubble pop.", freq=(500, "Hz"))
def pop(sr=DEFAULT_SR, freq=500):
    n = samples(0.08, sr)
    f = freq * (1 + 2 * _dec(n, 0.01, sr))
    return O.sine(f, n, sr) * _dec(n, 0.02, sr) * perc(0.08, 0.0005, sr, curve=1.0) * 0.9


@sfx("coin", "Classic coin / pickup: two quick rising notes.", freq=("B5", "first note or Hz"), up=(5, "semitones jump"), wave=("square", "square|sine|triangle"))
def coin(sr=DEFAULT_SR, freq="B5", up=5, wave="square"):
    f1 = to_hz(freq)
    f2 = f1 * 2 ** (up / 12)
    n1 = samples(0.08, sr)
    n2 = samples(0.35, sr)
    a = O.osc(wave, f1, n1, sr) * 0.5
    b = O.osc(wave, f2, n2, sr) * 0.5 * adsr(0.15, 0.002, 0.1, 0.5, 0.2, sr)[:n2]
    y = np.concatenate([a, b])
    if wave == "sine":
        return y * 0.8
    return _round(y, np.concatenate([np.full(n1, f1), np.full(n2, f2)]), sr) * 0.8


@sfx("powerup", "Rising arpeggio power-up.", freq=("C5", "start note or Hz"), steps=(6, "count"), step=(3, "semitones per step"),
     rate=(0.05, "s per step"), wave=("square", "square|saw|sine"))
def powerup(sr=DEFAULT_SR, freq="C5", steps=6, step=3, rate=0.05, wave="square"):
    f0 = to_hz(freq)
    parts = []
    for i in range(int(steps)):
        f = f0 * 2 ** (i * step / 12)
        d = rate if i < steps - 1 else rate * 5
        n = samples(d, sr)
        x = O.osc(wave, f, n, sr)
        if wave != "sine":
            x = _round(x, f, sr)
        x = x * 0.45 * adsr(d, 0.002, 0.0, 1.0, 0.0, sr)[:n]
        if i == steps - 1:
            x *= _dec(n, d / 3, sr)
        parts.append((x, i * rate))
    return mix(parts, sr)


@sfx("powerdown", "Falling arpeggio power-down / lose life.", freq=("C6", "start note or Hz"), steps=(6, "count"), rate=(0.07, "s per step"))
def powerdown(sr=DEFAULT_SR, freq="C6", steps=6, rate=0.07):
    f0 = to_hz(freq)
    parts = []
    for i in range(int(steps)):
        f = f0 * 2 ** (-i * 3 / 12)
        n = samples(rate * (1 if i < steps - 1 else 4), sr)
        x = _round(O.square(f, n, sr), f, sr) * 0.4 * _dec(n, rate * 2, sr)
        parts.append((x, i * rate))
    return mix(parts, sr)


@sfx("laser", "Descending laser / pew.", start=(2400, "Hz"), end=(300, "Hz"), dur=(0.25, "s"), wave=("saw", "saw|square|sine"))
def laser(sr=DEFAULT_SR, start=2400, end=300, dur=0.25, wave="saw"):
    f = O.sweep(start, end, dur, sr)
    n = f.shape[0]
    x = O.osc(wave, f, n, sr)
    x = (x if wave == "sine" else _round(x, f, sr, harm=4, lo=1800, hi=4200)) * perc(dur, 0.002, sr, curve=1.5)
    return x * 0.6


@sfx("zap", "Electric zap / shock (noisy, buzzy).", dur=(0.2, "s"), freq=(120, "buzz Hz"))
def zap(sr=DEFAULT_SR, dur=0.2, freq=120):
    n = samples(dur, sr)
    f = freq * (1 + 6 * _dec(n, 0.03, sr))
    x = O.saw(f, n, sr) * O.white(n, seed=22) * 1.5
    x = F.bandpass(x, 1500, sr, 0.5)
    return np.tanh(x * 3) * perc(dur, 0.001, sr, curve=2.0) * 0.8


@sfx("hit", "Impact / hurt hit (noise burst + pitch drop).", dur=(0.25, "s"), tone=(200, "Hz"), crunch=(0.5, "0..1 distortion"))
def hit(sr=DEFAULT_SR, dur=0.25, tone=200, crunch=0.5):
    n = samples(dur, sr)
    f = tone * (1 + 3 * _dec(n, 0.02, sr))
    body = O.square(f, n, sr) * _dec(n, dur / 4, sr)
    nz = F.lowpass(O.white(n, seed=23), 3000, sr) * _dec(n, dur / 5, sr)
    y = np.tanh((body + nz) * (1 + 4 * crunch))
    return y * perc(dur, 0.001, sr, curve=1.0) * 0.8


@sfx("punch", "Meaty punch / body hit.", dur=(0.2, "s"))
def punch(sr=DEFAULT_SR, dur=0.2):
    n = samples(dur, sr)
    f = 80 * (1 + 5 * _dec(n, 0.02, sr))
    y = O.sine(f, n, sr) * _dec(n, dur / 4, sr)
    y += F.bandpass(O.brown(n, seed=24), 400, sr, 0.7) * _dec(n, 0.05, sr) * 1.2
    return np.tanh(y * 2.5) * perc(dur, 0.001, sr, curve=1.0) * 0.9


@sfx("explosion", "Explosion: brown noise with falling lowpass.", dur=(1.2, "s"), boom=(1.0, "low rumble amount"))
def explosion(sr=DEFAULT_SR, dur=1.2, boom=1.0):
    n = samples(dur, sr)
    nz = O.brown(n, seed=25) * 0.7 + O.white(n, seed=26) * 0.3
    fc = 200 + 6000 * _dec(n, dur / 6, sr)
    y = F.biquad(nz, "lowpass", fc, sr, q=0.9)
    rumble = O.sine(45 * (1 + 2 * _dec(n, 0.05, sr)), n, sr) * _dec(n, dur / 3, sr) * boom
    y = np.tanh((y * 2.0 + rumble) * 1.5)
    return y * perc(dur, 0.005, sr, curve=1.8)


@sfx("jump", "Platformer jump (rising sweep).", start=(300, "Hz"), end=(900, "Hz"), dur=(0.18, "s"))
def jump(sr=DEFAULT_SR, start=300, end=900, dur=0.18):
    f = O.sweep(start, end, dur, sr)
    n = f.shape[0]
    x = O.square(f, n, sr)
    return _round(x, f, sr, harm=5, lo=1800, hi=3800) * adsr(dur - 0.03, 0.005, 0.0, 1.0, 0.03, sr)[:n] * 0.45


@sfx("whoosh", "Air whoosh / swipe (filtered noise sweep).", dur=(0.5, "s"), low=(300, "Hz"), high=(4000, "Hz"), direction=("up", "up|down"))
def whoosh(sr=DEFAULT_SR, dur=0.5, low=300, high=4000, direction="up"):
    n = samples(dur, sr)
    fc = O.sweep(low, high, dur, sr) if direction == "up" else O.sweep(high, low, dur, sr)
    y = F.biquad(O.white(n, seed=27), "bandpass", fc, sr, q=1.2)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    return y * env * 1.5


@sfx("swoosh", "Faster, sharper swipe for UI page transitions.", dur=(0.25, "s"))
def swoosh(sr=DEFAULT_SR, dur=0.25):
    return whoosh(sr, dur=dur, low=800, high=6000, direction="down")


@sfx("alarm", "Alternating two-tone alarm / siren.", freq=(700, "low: Hz or note name"), freq2=(900, "high: Hz or note name"), rate=(4.0, "Hz alternation"), dur=(1.0, "s"), wave=("square", "square|sine|saw"))
def alarm(sr=DEFAULT_SR, freq=700, freq2=900, rate=4.0, dur=1.0, wave="square"):
    n = samples(dur, sr)
    gate = (lfo(dur, rate, sr, "square")[:n] > 0)
    f = np.where(gate, to_hz(freq), to_hz(freq2))
    x = O.osc(wave, f, n, sr)
    x = F.lowpass(x, 3400, sr)
    return x * adsr(dur - 0.02, 0.005, 0.0, 1.0, 0.02, sr)[:n] * 0.5


@sfx("siren", "Smooth rising/falling siren.", low=(500, "Hz"), high=(1000, "Hz"), rate=(0.7, "Hz"), dur=(2.0, "s"))
def siren(sr=DEFAULT_SR, low=500, high=1000, rate=0.7, dur=2.0):
    n = samples(dur, sr)
    f = low + (high - low) * (0.5 + 0.5 * lfo(dur, rate, sr, "triangle")[:n])
    x = F.lowpass(O.saw(f, n, sr), 3000, sr)
    return x * adsr(dur - 0.05, 0.02, 0.0, 1.0, 0.05, sr)[:n] * 0.5


@sfx("error", "Harsh error buzz (two low tones).", freq=(180, "Hz"), dur=(0.3, "s"))
def error(sr=DEFAULT_SR, freq=180, dur=0.3):
    n = samples(dur, sr)
    x = O.square(freq, n, sr) * 0.5 + O.square(freq * 1.06, n, sr) * 0.5
    x = F.lowpass(x, 2500, sr)
    gate = (lfo(dur, 2.0 / dur, sr, "square")[:n] > 0).astype(float)
    return x * adsr(dur - 0.02, 0.003, 0.0, 1.0, 0.02, sr)[:n] * (0.4 + 0.6 * gate) * 0.5


@sfx("success", "Bright rising two-note confirm.", freq=("E5", "first note or Hz"), up=(7, "semitones"), wave=("sine", "sine|triangle|square"))
def success(sr=DEFAULT_SR, freq="E5", up=7, wave="sine"):
    f1 = to_hz(freq)
    f2 = f1 * 2 ** (up / 12)
    d1, d2 = 0.1, 0.35
    a = O.osc(wave, f1, samples(d1 + 0.02, sr), sr)
    b = O.osc(wave, f2, samples(d2 + 0.1, sr), sr)
    if wave != "sine":
        a, b = _round(a, f1, sr), _round(b, f2, sr)
    a = a * adsr(d1, 0.005, 0.0, 1.0, 0.02, sr)
    b = b * adsr(d2, 0.005, 0.15, 0.6, 0.1, sr)
    return mix([(a, 0), (b, d1)], sr) * 0.7


@sfx("proximity", "Repeating proximity beep (faster = closer).", freq=(1000, "Hz"), rate=(4.0, "beeps per second"), dur=(1.0, "total s"), width=(0.4, "beep length as fraction of period"))
def proximity(sr=DEFAULT_SR, freq=1000, rate=4.0, dur=1.0, width=0.4):
    period = 1.0 / rate
    parts = []
    t = 0.0
    while t < dur - 1e-6:
        parts.append((beep(sr, freq=freq, dur=period * width), t))
        t += period
    return mix(parts, sr)


@sfx("radar", "Radar ping with echo tail.", freq=(1200, "Hz or note name (110, '110hz' or 'A2')"))
def radar(sr=DEFAULT_SR, freq=1200):
    n = samples(0.6, sr)
    f = to_hz(freq) * (1 + 0.3 * _dec(n, 0.05, sr))
    x = O.sine(f, n, sr) * _dec(n, 0.12, sr)
    return x * perc(0.6, 0.002, sr, curve=1.0) * 0.8


@sfx("riser", "Tension riser (noise + pitch climbing).", dur=(2.0, "s"), start=(100, "Hz"), end=(2000, "Hz"))
def riser(sr=DEFAULT_SR, dur=2.0, start=100, end=2000):
    n = samples(dur, sr)
    f = O.sweep(start, end, dur, sr)
    tone = O.saw(f, n, sr) * 0.4
    nz = F.biquad(O.white(n, seed=28), "bandpass", f * 4, sr, q=0.8)
    env = np.linspace(0, 1, n) ** 2
    y = (tone + nz) * env
    return F.highpass(y, 80, sr) * 0.8


@sfx("sweep_up", "Pure tone sweep upward.", start=(200, "Hz"), end=(2000, "Hz"), dur=(0.4, "s"), wave=("sine", "sine|saw|square"))
def sweep_up(sr=DEFAULT_SR, start=200, end=2000, dur=0.4, wave="sine"):
    f = O.sweep(start, end, dur, sr)
    n = f.shape[0]
    x = O.osc(wave, f, n, sr)
    if wave != "sine":
        x = _round(x, f, sr, harm=4, lo=1800, hi=4200)
    return x * adsr(dur - 0.05, 0.01, 0.0, 1.0, 0.05, sr)[:n] * 0.6


@sfx("sweep_down", "Pure tone sweep downward.", start=(2000, "Hz"), end=(200, "Hz"), dur=(0.4, "s"), wave=("sine", "sine|saw|square"))
def sweep_down(sr=DEFAULT_SR, start=2000, end=200, dur=0.4, wave="sine"):
    return sweep_up(sr, start, end, dur, wave)


@sfx("bubble", "Watery bubble/drop.", freq=(400, "Hz"))
def bubble(sr=DEFAULT_SR, freq=400):
    n = samples(0.15, sr)
    f = freq * (1 + 1.5 * np.linspace(0, 1, n) ** 2)
    return O.sine(f, n, sr) * _dec(n, 0.05, sr) * perc(0.15, 0.002, sr, curve=1.0) * 0.8


@sfx("glitch", "Digital glitch burst.", dur=(0.2, "s"))
def glitch(sr=DEFAULT_SR, dur=0.2):
    n = samples(dur, sr)
    rng = np.random.default_rng(29)
    y = np.zeros(n)
    seg = samples(0.012, sr)
    for s in range(0, n, seg):
        f = rng.choice([220, 440, 880, 1760, 3520]) * rng.uniform(0.9, 1.1)
        e = min(n, s + seg)
        y[s:e] = O.square(f, e - s, sr) * rng.uniform(0.2, 1.0)
    y = F.lowpass(np.round(y * 6) / 6, 5500, sr)
    return y * perc(dur, 0.001, sr, curve=0.8) * 0.5


@sfx("static", "Radio static / white noise burst.", dur=(0.5, "s"), color=("white", "white|pink|brown"))
def static(sr=DEFAULT_SR, dur=0.5, color="white"):
    n = samples(dur, sr)
    return O.noise(n, color, seed=30, sr=sr) * adsr(dur - 0.02, 0.005, 0.0, 1.0, 0.02, sr)[:n] * 0.5


@sfx("wind", "Wind gust.", dur=(3.0, "s"))
def wind(sr=DEFAULT_SR, dur=3.0):
    n = samples(dur, sr)
    fc = 300 + 700 * (0.5 + 0.5 * lfo(dur, 0.25, sr)[:n]) + 200 * lfo(dur, 1.3, sr, "random")[:n]
    y = F.biquad(O.pink(n, seed=31), "bandpass", fc, sr, q=0.7)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.7
    return y * env * 1.2


@sfx("thunder", "Distant thunder rumble.", dur=(2.5, "s"))
def thunder(sr=DEFAULT_SR, dur=2.5):
    n = samples(dur, sr)
    y = F.lowpass(O.brown(n, seed=32), 200, sr, order=2)
    env = _dec(n, dur / 3, sr) * (0.6 + 0.4 * (0.5 + 0.5 * lfo(dur, 3.0, sr, "random")[:n]))
    return np.tanh(y * 3 * env) * perc(dur, 0.05, sr, curve=1.0)


@sfx("footstep", "Footstep on hard floor.", tone=(120, "Hz thump"))
def footstep(sr=DEFAULT_SR, tone=120):
    n = samples(0.15, sr)
    thump = O.sine(tone * (1 + _dec(n, 0.01, sr)), n, sr) * _dec(n, 0.03, sr)
    scuff = F.bandpass(O.white(n, seed=33), 1500, sr, 0.6) * _dec(n, 0.02, sr) * 0.5
    return np.tanh((thump + scuff) * 1.5) * perc(0.15, 0.002, sr, curve=1.0) * 0.8


@sfx("door", "Door thud/close.")
def door(sr=DEFAULT_SR):
    n = samples(0.35, sr)
    thud = O.sine(70 * (1 + 2 * _dec(n, 0.02, sr)), n, sr) * _dec(n, 0.08, sr)
    latch = F.bandpass(O.white(n, seed=34), 2500, sr, 1.0) * _dec(n, 0.01, sr) * 0.6
    latch = np.concatenate([np.zeros(samples(0.04, sr)), latch])[:n]
    return np.tanh((thud + latch) * 2) * perc(0.35, 0.002, sr, curve=1.0) * 0.9


@sfx("engine", "Engine hum loop-able.", rpm=(60, "Hz base"), dur=(2.0, "s"))
def engine(sr=DEFAULT_SR, rpm=60, dur=2.0):
    n = samples(dur, sr)
    f = rpm * (1 + 0.02 * lfo(dur, 7.0, sr, "random")[:n])
    y = O.saw(f, n, sr) * 0.5 + O.pulse(f * 0.5, n, sr, width=0.3) * 0.5
    y = F.lowpass(y, 800, sr, 1.5)
    y += F.lowpass(O.brown(n, seed=35), 300, sr) * 0.3
    return np.tanh(y * 1.5) * adsr(dur - 0.05, 0.05, 0.0, 1.0, 0.05, sr)[:n] * 0.6


@sfx("magic", "Sparkly magic shimmer (random high bell notes).", dur=(0.8, "s"), density=(12, "notes"), freq=("C6", "base note or Hz"))
def magic(sr=DEFAULT_SR, dur=0.8, density=12, freq="C6"):
    from .instruments import render_note
    rng = np.random.default_rng(36)
    base = to_hz(freq)
    parts = []
    for i in range(int(density)):
        semi = rng.choice([0, 2, 4, 7, 9, 12, 14, 16])
        t = (i / density) * dur * 0.8 * rng.uniform(0.9, 1.1)
        parts.append((render_note("glass", base * 2 ** (semi / 12), 0.1, sr, vel=rng.uniform(0.4, 1.0)) * 0.5, t))
    return mix(parts, sr)


@sfx("heartbeat", "Heartbeat lub-dub.", rate=(1.0, "beats per second"), dur=(2.0, "s"))
def heartbeat(sr=DEFAULT_SR, rate=1.0, dur=2.0):
    parts = []
    t = 0.0
    while t < dur:
        n = samples(0.15, sr)
        lub = O.sine(55 * (1 + _dec(n, 0.03, sr)), n, sr) * _dec(n, 0.05, sr)
        parts.append((np.tanh(lub * 2), t))
        parts.append((np.tanh(lub * 1.5), t + 0.18))
        t += 1.0 / rate
    return F.lowpass(mix(parts, sr), 200, sr)


@sfx("tone", "Plain sustained tone with chosen wave (raw: nothing is rounded off).", freq=(440, "Hz or note name (110, '110hz' or 'A2')"), dur=(1.0, "s"),
     wave=("sine", "any osc wave"), seed=(0, "noise waves only: a different seed gives a different take"))
def tone(sr=DEFAULT_SR, freq=440, dur=1.0, wave="sine", seed=0):
    n = samples(dur, sr)
    x = O.osc(wave, to_hz(freq), n, sr, seed=seed)
    return x * adsr(dur - 0.02, 0.01, 0.0, 1.0, 0.02, sr)[:n] * 0.6


@sfx("noise", "Noise burst with envelope.", dur=(0.3, "s"), color=("white", "white|pink|brown"), attack=(0.005, "s"))
def noise(sr=DEFAULT_SR, dur=0.3, color="white", attack=0.005):
    n = samples(dur, sr)
    return O.noise(n, color, seed=37, sr=sr) * perc(dur, attack, sr) * 0.6


@sfx("typewriter", "Key press clack.")
def typewriter(sr=DEFAULT_SR):
    n = samples(0.06, sr)
    y = F.bandpass(O.white(n, seed=38), 2200, sr, 0.8) * _dec(n, 0.006, sr)
    y += O.sine(900, n, sr) * _dec(n, 0.004, sr) * 0.6
    return np.tanh(y * 3) * 0.8


@sfx("countdown", "Countdown tick: short low tone.", freq=(660, "Hz"))
def countdown(sr=DEFAULT_SR, freq=660):
    return beep(sr, freq=freq, dur=0.08, wave="square") * 0.8


@sfx("vinyl", "Vinyl record surface: crackle, pops and a little hiss and rumble (the bed under lo-fi music; loop it with \"loop\": true).",
     dur=(4.0, "s"), crackle=(14.0, "clicks per second"), hiss=(0.3, "0..1 surface noise"), seed=(0, "variation"))
def vinyl(sr=DEFAULT_SR, dur=4.0, crackle=14.0, hiss=0.3, seed=0):
    n = samples(dur, sr)
    rng = np.random.default_rng(40 + int(seed))
    y = np.zeros(n)
    for _ in range(int(dur * crackle)):
        s = int(rng.integers(0, n))
        size = rng.random() ** 3                       # mostly tiny ticks, the odd loud pop
        m = min(n - s, samples(0.0015 + 0.004 * size, sr))
        y[s:s + m] += rng.uniform(-1, 1, m) * np.exp(-np.arange(m) / (0.0006 * sr * (1 + 4 * size))) * (0.15 + 0.85 * size)
    y = F.lowpass(F.highpass(y, 500, sr), 4500, sr)
    y += F.lowpass(F.highpass(O.pink(n, seed=41 + int(seed)), 1500, sr), 6000, sr) * 0.03 * hiss
    y += F.lowpass(O.brown(n, seed=42 + int(seed), sr=sr), 60, sr) * 0.1 * hiss
    return y * 0.8


def _full_scale(y):
    """The ambience kernels are mastered for long beds and come back 10-25x full scale as one-shots."""
    peak = float(np.max(np.abs(y))) if np.size(y) else 0.0
    return y * (0.9 / peak) if peak > 1.0 else y


def render_sfx(name: str, sr: int = DEFAULT_SR, **params) -> np.ndarray:
    if name not in REGISTRY:
        raise ValueError(f"unknown sfx {name!r}; run `genny list sfx`")
    entry = REGISTRY[name]
    unknown = [k for k in params if k not in entry["params"]]
    if unknown:
        raise ValueError(f"sfx {name!r} does not accept params {unknown}; allowed: {list(entry['params'])}")
    return entry["fn"](sr, **params)


# Registers car_engine into REGISTRY (kept in its own module).
from . import vehicle  # noqa: E402,F401
# Pinball foley (solenoid, knocker, steel_ball, ...).
from . import pinball  # noqa: E402,F401

# -----------------------------------------------------------------------------
# Physical-engine SFX overrides (v0.5)
@sfx("footstep", "Physically driven footstep; compatibility `tone` selects hard-floor resonance.",
     tone=(120, "Hz thump"), ground=("wood", "wood|concrete|gravel|snow|sand|..."), variation=(0, "variation index"))
def footstep(sr=DEFAULT_SR, tone=120, ground="wood", variation=0):
    from .physics import Footstep
    try:
        y = Footstep(str(ground), int(variation)).render(sr=sr, seed=33 + int(variation))
    except Exception:
        y = Footstep("wood", int(variation)).render(sr=sr, seed=33 + int(variation))
    # Historical `tone` remains meaningful as a low resonant body emphasis.
    y = F.peak(y, float(np.clip(tone, 45, 300)), sr, 1.0, 2.0)
    return y * 0.78


@sfx("wind", "Procedural causal wind field using the physical ambience engine.", dur=(3.0, "s"), scale=(1.0, "strength"), seed=(0, "variation"))
def wind(sr=DEFAULT_SR, dur=3.0, scale=1.0, seed=0):
    from .physics import Environment
    y = Environment(seed=31 + int(seed)).wind(dur, sr=sr, scale=float(scale), sources=("static",))
    if np.asarray(y).ndim == 2:
        y = np.mean(y, axis=1)
    y = np.asarray(y, float).copy()
    k = min(len(y), max(2, int(round(0.01 * sr))))
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return y * 0.85


@sfx("thunder", "Segmented procedural thunder with distance-dependent propagation.", dur=(2.5, "s"), distance=(80.0, "m"), seed=(0, "variation"))
def thunder(sr=DEFAULT_SR, dur=2.5, distance=80.0, seed=0):
    from .physics import Environment
    y = Environment(seed=32 + int(seed)).thunder(max(float(dur), 0.2), distance=float(distance), sr=sr)
    if np.asarray(y).ndim == 2:
        y = np.mean(y, axis=1)
    y = np.asarray(y, float).copy()
    k = min(len(y), max(2, int(round(0.02 * sr))))
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return _full_scale(y)


@sfx("hit", "Physically resonant impact; material controls modal body and micro-collisions.",
     dur=(0.25, "s"), tone=(200, "Hz/body scale"), crunch=(0.5, "0..1 hardness"), material=("steel", "steel|glass|wood|stone|aluminum|rubber"))
def hit(sr=DEFAULT_SR, dur=0.25, tone=200, crunch=0.5, material="steel"):
    from .physics import ModalBody, MATERIALS
    mat = MATERIALS.get(str(material), MATERIALS["steel"])
    # standardBar has a stable, impact-like mode family. Scale it so the
    # historical tone parameter still moves the perceived body resonance.
    scale = float(np.clip(float(tone) / 200.0, 0.25, 4.0))
    body = ModalBody("sy_stick", material=mat, frequency_scale=scale,
                     decay_scale=1.35 - 0.65 * float(np.clip(crunch, 0, 1)),
                     gain_scale=1.0)
    y = body.strike(duration=max(0.06, float(dur)), contact_ms=1.8 - 1.5 * float(np.clip(crunch, 0, 1)),
                    velocity=0.55 + 0.75 * float(np.clip(crunch, 0, 1)), sr=sr,
                    micro_collisions=int(round(1 + 5 * float(np.clip(crunch, 0, 1)))), seed=23)
    return np.tanh(y * (1.2 + 1.6 * float(np.clip(crunch, 0, 1)))) * 0.78

@sfx("rain", "Physical/statistical rain field with Poisson drops and surface interactions.", dur=(3.0,"s"), intensity=(1.0,"strength"), seed=(0,"variation"))
def rain(sr=DEFAULT_SR, dur=3.0, intensity=1.0, seed=0):
    from .physics import Environment
    y=Environment(seed=100+int(seed)).rain(float(dur), float(intensity), sr=sr)
    return _full_scale(np.mean(y,axis=1) if np.asarray(y).ndim==2 else y)


@sfx("fire", "Procedural combustion texture with shared-state hiss/crackle/roar.", dur=(3.0,"s"), seed=(0,"variation"))
def fire(sr=DEFAULT_SR, dur=3.0, seed=0):
    from .physics import Environment
    y=Environment(seed=110+int(seed)).fire(float(dur), sr=sr)
    return np.mean(y,axis=1) if np.asarray(y).ndim==2 else y


@sfx("stream", "Procedural running water using stochastic cavities and bubble resonances.", dur=(3.0,"s"), seed=(0,"variation"))
def stream(sr=DEFAULT_SR, dur=3.0, seed=0):
    from .physics import Environment
    y=Environment(seed=120+int(seed)).stream(float(dur), sr=sr)
    return np.mean(y,axis=1) if np.asarray(y).ndim==2 else y


@sfx("pour", "Reduced physical pouring model: stream + bubbles + rising container cavity resonance.", dur=(2.0,"s"), flow=(1.0,"flow rate"), fill=(0.0,"initial fill 0..1"), seed=(0,"variation"))
def pour(sr=DEFAULT_SR, dur=2.0, flow=1.0, fill=0.0, seed=0):
    from .physics import Environment
    y=Environment(seed=130+int(seed)).pour(float(dur), sr=sr, flow=float(flow), fill=float(fill))
    return np.mean(y,axis=1) if np.asarray(y).ndim==2 else y


@sfx("engine", "Four-stroke combustion engine using firing pulses, exhaust resonances and intake noise.", rpm=(1500,"RPM"), dur=(2.0,"s"), cylinders=(4,"count"), load=(0.5,"0..1"), rough=(0.25,"0..1"))
def engine(sr=DEFAULT_SR, rpm=1500, dur=2.0, cylinders=4, load=.5, rough=.25):
    from .physics import CombustionEngine
    return CombustionEngine(int(cylinders), float(rpm), float(load), float(rough)).render(float(dur), sr=sr, seed=7)


@sfx("swoosh", "Distributed aeroacoustic swing using Reynolds/Strouhal vortex-shedding model.", dur=(0.25,"legacy duration; model duration follows preset"), speed=(1.0,"velocity scale"), diameter=(1.0,"diameter scale"), seed=(0,"variation"))
def swoosh(sr=DEFAULT_SR, dur=0.25, speed=1.0, diameter=1.0, seed=0):
    from .physics import AeroacousticSwing
    y=AeroacousticSwing("SWD1", velocity_scale=float(speed), diameter_scale=float(diameter)).render(sr=sr, seed=int(seed))
    target=max(1,int(round(float(dur)*sr)))
    return _full_scale(y[:target] if len(y)>=target else np.pad(y,(0,target-len(y))))

# v0.7 physical contact / rotor / destruction migration -----------------------
@sfx("door", "Physical stick-slip/creak door interaction.", dur=(1.0,"s"), speed=(0.5,"motion"), material=("wood","wood|metal"), seed=(0,"variation"))
def door(sr=DEFAULT_SR,dur=1.0,speed=.5,material="wood",seed=0):
    from .physics import Environment
    n=max(1,int(round(float(dur)*sr))); t=np.arange(n)/sr
    # hinge force rises and relaxes repeatedly as the door moves
    force=np.maximum(0,np.sin(2*np.pi*(1.3+2.4*abs(float(speed)))*t))*(.25+.75*abs(float(speed)))
    return Environment(seed=210+int(seed)).creak(force,material=str(material),sr=sr)


@sfx("roll", "Physical rolling contact over a rough spatial surface.", dur=(1.0,"s"), speed=(0.5,"m/s"), radius=(0.03,"m"), roughness=(0.5,"0..1"), seed=(0,"variation"))
def roll(sr=DEFAULT_SR,dur=1.0,speed=.5,radius=.03,roughness=.5,seed=0):
    from .physics import SurfaceContact
    return SurfaceContact(float(np.clip(roughness,0,1)),1.2,body_frequency=720,body_decay=16).roll(
        float(dur),speed=float(speed),radius_m=float(radius),force=1.0,sr=sr,seed=int(seed))


@sfx("scrape", "Physical spatial-profile scrape.", dur=(.8,"s"), speed=(.3,"m/s"), roughness=(.5,"0..1"), seed=(0,"variation"))
def scrape(sr=DEFAULT_SR,dur=.8,speed=.3,roughness=.5,seed=0):
    from .physics import SurfaceContact
    return SurfaceContact(float(np.clip(roughness,0,1)),.9,body_frequency=950,body_decay=23).scrape(
        float(dur),speed=float(speed),force=1.0,sr=sr,seed=int(seed))


@sfx("fan", "Physical rotor/fan using blade-passing frequency and tip-speed turbulence.", dur=(2.0,"s"), rpm=(1800,"RPM"), blades=(5,"count"), seed=(0,"variation"))
def fan(sr=DEFAULT_SR,dur=2.0,rpm=1800,blades=5,seed=0):
    from .physics import Rotor
    return Rotor(float(rpm),int(blades),.15,.48,.54).render(float(dur),sr=sr,seed=int(seed))


@sfx("propeller", "Physical propeller: low blade-passing pulse train plus aerodynamic turbulence.", dur=(2.0,"s"), rpm=(2100,"RPM"), blades=(3,"count"), seed=(0,"variation"))
def propeller(sr=DEFAULT_SR,dur=2.0,rpm=2100,blades=3,seed=0):
    from .physics import Rotor
    return Rotor(float(rpm),int(blades),1.05,.58,.72).render(float(dur),sr=sr,seed=int(seed))


@sfx("jet_engine", "Reduced physical multi-spool jet/turbofan model.", dur=(3.0,"s"), rpm=(9000,"fan/core RPM scale"), throttle=(.7,"0..1"), seed=(0,"variation"))
def jet_engine(sr=DEFAULT_SR,dur=3.0,rpm=9000,throttle=.7,seed=0):
    from .physics import JetEngine
    return JetEngine(float(rpm),24,36,float(throttle)).render(float(dur),sr=sr,seed=int(seed))


@sfx("helicopter", "Physical rotor helicopter: main rotor, tail rotor, turbine and blade-vortex slap.", dur=(3.0,"s"), rpm=(320,"main rotor RPM"), blades=(4,"main blade count"), load=(.55,"0..1"), seed=(0,"variation"))
def helicopter(sr=DEFAULT_SR,dur=3.0,rpm=320,blades=4,load=.55,seed=0):
    from .physics import Helicopter
    return Helicopter(float(rpm),int(blades),1600,4,float(load)).render(float(dur),sr=sr,seed=int(seed))


@sfx("shatter", "Physical fracture/destruction event with time-varying modal body.", dur=(2.0,"s"), material=("glass","glass|steel|wood|stone"), energy=(1.0,"strength"), seed=(0,"variation"))
def shatter(sr=DEFAULT_SR,dur=2.0,material="glass",energy=1.0,seed=0):
    from .physics import Destruction
    # Known research-table fallback is glass; other materials retain their
    # physical damping/hardness even when the geometry proxy is glass-like.
    return Destruction(str(material),"sy_vase").render(float(dur),energy=float(energy),sr=sr,seed=int(seed))

# v0.8 machine/contact/blast migration ---------------------------------------
@sfx("roll", "Bidirectionally coupled rolling contact over a rough surface.", dur=(1.0,"s"), speed=(0.5,"m/s"), radius=(0.03,"m"), roughness=(0.5,"0..1"), feedback=(0.18,"contact/resonator coupling"), seed=(0,"variation"))
def roll(sr=DEFAULT_SR,dur=1.0,speed=.5,radius=.03,roughness=.5,feedback=.18,seed=0):
    from .physics import AdvancedRollingContact
    return AdvancedRollingContact(float(np.clip(roughness,0,1)),1.2,720,16,float(feedback)).render(
        float(dur),speed=float(speed),radius_m=float(radius),normal_force=1.0,sr=sr,seed=int(seed))


@sfx("electric_motor", "Physical electric motor: rotor, commutation, imbalance, bearings and housing.", dur=(2.0,"s"), rpm=(6000,"RPM"), load=(0.5,"0..1"), startup=(0.0,"s"), seed=(0,"variation"))
def electric_motor(sr=DEFAULT_SR,dur=2.0,rpm=6000,load=.5,startup=0.0,seed=0):
    from .physics import ElectricMotor
    return ElectricMotor(rpm=float(rpm)).render(float(dur),load=float(load),startup=float(startup),sr=sr,seed=int(seed))


@sfx("gears", "Physical gear train with tooth-mesh frequency, shaft sidebands and backlash impacts.", dur=(2.0,"s"), rpm=(1800,"RPM"), driver_teeth=(20,"count"), driven_teeth=(40,"count"), backlash=(0.15,"0..1"), seed=(0,"variation"))
def gears(sr=DEFAULT_SR,dur=2.0,rpm=1800,driver_teeth=20,driven_teeth=40,backlash=.15,seed=0):
    from .physics import GearTrain
    return GearTrain(float(rpm),int(driver_teeth),int(driven_teeth),float(backlash)).render(
        float(dur),sr=sr,seed=int(seed))


@sfx("explosion", "Physical blast: N-wave shock, low-frequency expansion, debris and distance propagation.", dur=(1.5,"s"), energy=(1.0,"relative blast energy"), distance=(8.0,"m"), debris=(0.45,"0..1"), enclosure=(0.15,"0..1"), seed=(0,"variation"), boom=(None,"legacy name for energy"))
def explosion(sr=DEFAULT_SR,dur=1.5,energy=1.0,distance=8.0,debris=.45,enclosure=.15,seed=0,boom=None):
    # ``boom`` remains accepted for compatibility; map it to physical energy.
    if boom is not None: energy=float(boom)
    from .physics import Explosion
    return Explosion(float(energy),float(distance),float(debris),float(enclosure)).render(
        float(dur),sr=sr,seed=int(seed))
