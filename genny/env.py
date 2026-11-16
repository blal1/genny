"""Envelopes: ADSR, exponential decay, and generic breakpoint envelopes; segment envelopes with gate, release and
loop nodes; control-rate smoothing, sample and hold, triggers and slow random signals."""
from __future__ import annotations

import math

import numba
import numpy as np

from .core import DEFAULT_SR, samples


def adsr(dur: float, attack: float = 0.01, decay: float = 0.1, sustain: float = 0.7, release: float = 0.2,
         sr: int = DEFAULT_SR, curve: float = 2.0) -> np.ndarray:
    """ADSR where `dur` is the note-on length; the release is appended after it.

    Returned length = samples(dur + release). curve > 1 gives exponential-ish (natural) shapes.
    """
    n_on = samples(dur, sr)
    n_rel = samples(release, sr) if release > 0 else 0
    a = min(samples(attack, sr) if attack > 0 else 1, n_on)
    d = min(samples(decay, sr) if decay > 0 else 1, max(1, n_on - a))
    s = max(0, n_on - a - d)
    # concave rise 1 - (1 - x)^curve: same ease-in as x^(1/curve) but with a finite slope at the start, so
    # the first samples of a note do not step (x^(1/2) put 6.7 % of full level on sample 1 of a 5 ms attack)
    env_a = 1.0 - (1.0 - np.linspace(0, 1, a)) ** curve if a > 1 else np.ones(a)
    env_d = sustain + (1 - sustain) * (1 - np.linspace(0, 1, d)) ** curve if d > 1 else np.full(d, sustain)
    env_s = np.full(s, sustain)
    on = np.concatenate([env_a, env_d, env_s])[:n_on]
    if n_rel > 0:
        last = on[-1] if on.size else sustain
        rel = last * (1 - np.linspace(0, 1, n_rel)) ** curve
        return np.concatenate([on, rel])
    return on


def power_adsr(dur: float, attack: float = 0.01, decay: float = 0.1, sustain: float = 0.7, release: float = 0.2,
               sr: int = DEFAULT_SR, curves: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """ADSR with its own power curve per stage: each stage goes from start to target as start + (target - start)
    * x^c, x the linear 0..1 ramp; c < 1 is sharp, c > 1 soft, 1 linear. ``curves`` = (attack, decay, release).
    Same lengths as ``adsr``: samples(dur + release), the release starting at the note-on end."""
    ca, cd, cr = (max(float(c), 1e-6) for c in curves)
    n_on = samples(dur, sr)
    n_rel = samples(release, sr) if release > 0 else 0
    a = min(samples(attack, sr) if attack > 0 else 0, n_on)
    d = min(samples(decay, sr) if decay > 0 else 0, n_on - a)
    ramp = lambda m: np.arange(1, m + 1) / m
    on = np.concatenate([ramp(a) ** ca, 1.0 + (sustain - 1.0) * ramp(d) ** cd, np.full(n_on - a - d, float(sustain))])
    if n_rel == 0:
        return on
    last = on[-1] if on.size else 0.0
    return np.concatenate([on, last * (1.0 - ramp(n_rel) ** cr)])


WINDOWS = ("hann", "blackman", "parzen", "tukey", "welch", "poisson")


def _half_window(window: str, x: np.ndarray, a: float | None) -> np.ndarray:
    """Rising half of a window: x 0 at the edge, 1 at the centre (value 1)."""
    if window == "hann":
        return 0.5 - 0.5 * np.cos(np.pi * x)
    if window == "blackman":
        a = 0.16 if a is None else a
        return (1 - a) / 2 - 0.5 * np.cos(np.pi * x) + a / 2 * np.cos(2 * np.pi * x)
    if window == "parzen":
        u = 1.0 - x
        return np.where(u <= 0.5, 1 - 6 * u ** 2 + 6 * u ** 3, 2 * (1 - u) ** 3)
    if window == "tukey":
        a = 0.5 if a is None else a
        return np.where(x < a, 0.5 - 0.5 * np.cos(np.pi * np.minimum(x / max(a, 1e-9), 1.0)), 1.0)
    if window == "welch":
        return 1.0 - (1.0 - x) ** 2
    if window == "poisson":
        a = np.log(1000.0) if a is None else a
        return np.exp(-a * (1.0 - x))
    raise ValueError(f"unknown window {window!r}; use one of {WINDOWS}")


def win_env(dur: float, attack: float = 0.05, release: float = 0.05, sr: int = DEFAULT_SR, window: str = "hann",
            a: float | None = None) -> np.ndarray:
    """Attack / release envelope shaped by a window function: the attack is the window's rising half, held at 1
    until ``dur``, then the falling half over ``release``. ``a``: the window's coefficient (blackman alpha 0.16,
    tukey taper 0.5, poisson decay ln 1000 = -60 dB at the edge). Length samples(dur + release)."""
    n_on = samples(dur, sr)
    n_rel = samples(release, sr) if release > 0 else 0
    na = min(samples(attack, sr) if attack > 0 else 0, n_on)
    rise = _half_window(window, np.arange(na) / max(na, 1), a)
    fall = _half_window(window, 1.0 - np.arange(1, n_rel + 1) / max(n_rel, 1), a)
    return np.concatenate([rise, np.ones(n_on - na), fall])


def decay(dur: float, tau: float, sr: int = DEFAULT_SR, floor: float = 0.0) -> np.ndarray:
    """Exponential decay exp(-t/tau) over dur seconds."""
    t = np.arange(samples(dur, sr)) / sr
    e = np.exp(-t / max(tau, 1e-4))
    return np.maximum(e, floor)


def perc(dur: float, attack: float = 0.002, sr: int = DEFAULT_SR, curve: float = 3.0) -> np.ndarray:
    """Percussive: fast attack then curved decay to 0 over dur."""
    n = samples(dur, sr)
    a = min(max(1, samples(attack, sr)), n)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    rest = n - a
    if rest > 0:
        env[a:] = (1 - np.linspace(0, 1, rest)) ** curve
    return env


def breakpoints(points: list[tuple[float, float]], dur: float, sr: int = DEFAULT_SR) -> np.ndarray:
    """points: [(time_seconds, value), ...] linearly interpolated over dur."""
    n = samples(dur, sr)
    t = np.arange(n) / sr
    ts = [p[0] for p in points]
    vs = [p[1] for p in points]
    return np.interp(t, ts, vs)


def apply(sig: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Multiply, padding whichever is shorter with zeros / last value."""
    n = max(sig.shape[0], env.shape[0])
    if sig.shape[0] < n:
        sig = np.concatenate([sig, np.zeros(n - sig.shape[0])])
    if env.shape[0] < n:
        env = np.concatenate([env, np.zeros(n - env.shape[0])])
    return sig * env


def lfo(dur: float, rate: float, sr: int = DEFAULT_SR, shape: str = "sine", depth: float = 1.0, phase0: float = 0.0, seed: int = 7) -> np.ndarray:
    t = np.arange(samples(dur, sr)) / sr
    ph = (t * rate + phase0) % 1.0
    if shape == "sine":
        v = np.sin(2 * np.pi * ph)
    elif shape == "triangle":
        v = 4 * np.abs(ph - 0.5) - 1
    elif shape == "square":
        v = np.where(ph < 0.5, 1.0, -1.0)
    elif shape == "saw":
        v = 2 * ph - 1
    elif shape == "random":
        rng = np.random.default_rng(int(seed))
        steps = int(np.ceil(dur * rate)) + 1
        vals = rng.uniform(-1, 1, steps)
        v = vals[np.minimum((t * rate).astype(int), steps - 1)]
    else:
        raise ValueError(f"unknown lfo shape {shape}")
    return v * depth


# ---------------------------------------------------------------- segment envelopes (level, time, shape)
SEGMENT_SHAPES = ("step", "hold", "lin", "exp", "sine", "welch", "sqr", "cub")
LOG001 = math.log(0.001)


def segment(l0: float, l1: float, n: int, shape="lin") -> np.ndarray:
    """n samples from l0 (excluded) to l1 (reached on the last one). ``shape``: a name of SEGMENT_SHAPES or a number c,
    the curve l0 + (l1 - l0) (1 - e^(c u)) / (1 - e^c) (c < 0 fast start, c > 0 slow start, 0 = lin)."""
    if n <= 0:
        return np.zeros(0)
    u = np.arange(1, n + 1) / n
    if not isinstance(shape, str):
        c = float(shape)
        if abs(c) < 1e-3 or n == 1:
            return l0 + (l1 - l0) * u
        return l0 + (l1 - l0) * (1.0 - np.exp(c * u)) / (1.0 - math.exp(c))
    if shape == "step":
        return np.full(n, float(l1))
    if shape == "hold":
        y = np.full(n, float(l0))
        y[-1] = l1
        return y
    if shape == "lin" or n == 1:
        return l0 + (l1 - l0) * u
    if shape == "exp":
        a = l0 if abs(l0) > 1e-5 else math.copysign(1e-5, l1 if l1 else 1.0)
        b = l1 if abs(l1) > 1e-5 else math.copysign(1e-5, a)
        return a * (b / a) ** u if a * b > 0 else l0 + (l1 - l0) * u
    if shape == "sine":
        return l0 + (l1 - l0) * (0.5 - 0.5 * np.cos(np.pi * u))
    if shape == "welch":
        return l0 + (l1 - l0) * np.sin(0.5 * np.pi * u) if l1 >= l0 else l1 + (l0 - l1) * np.cos(0.5 * np.pi * u)
    if shape == "sqr":
        a, b = math.sqrt(max(l0, 0.0)), math.sqrt(max(l1, 0.0))
        return (a + (b - a) * u) ** 2
    if shape == "cub":
        a, b = np.cbrt(l0), np.cbrt(l1)
        return (a + (b - a) * u) ** 3
    raise ValueError(f"unknown segment shape {shape!r}; use a number or one of {SEGMENT_SHAPES}")


def _segments(start, levels, times, shapes, idx, sr):
    """Segments idx from the current level ``start``; lengths rounded on the running time so nothing drifts."""
    out, cur, t = [], float(start), 0.0
    for i in idx:
        n = int(round((t + times[i]) * sr)) - int(round(t * sr))
        t += times[i]
        seg = segment(cur, float(levels[i + 1]), n, shapes[i])
        if seg.size:
            cur = float(seg[-1])
        out.append(seg)
    return (np.concatenate(out) if out else np.zeros(0)), cur


def env(levels, times, shapes="lin", sr: int = DEFAULT_SR, gate: float | None = None, release_node: int | None = None,
        loop_node: int | None = None) -> np.ndarray:
    """Envelope through ``levels`` (N + 1 values) over ``times`` (N seconds), each segment with its shape (one for all or
    a list: names of SEGMENT_SHAPES or curve numbers). With ``release_node`` r and ``gate`` (s the note is held): the
    segments before node r play, then the level holds at node r, or the segments from ``loop_node`` to r loop, until the
    gate closes; the segments from r on then start from wherever the level is (release mid-attack is smooth)."""
    levels = [float(v) for v in levels]
    times = [max(float(t), 0.0) for t in times]
    N = len(times)
    if len(levels) != N + 1:
        raise ValueError("env needs one more level than times")
    shapes = list(shapes) if isinstance(shapes, (list, tuple)) else [shapes] * N
    if release_node is None or gate is None:
        return np.concatenate([[levels[0]], _segments(levels[0], levels, times, shapes, range(N), sr)[0]])
    r = int(np.clip(release_node, 0, N))
    G = max(0, int(round(float(gate) * sr)))
    pre, cur = _segments(levels[0], levels, times, shapes, range(r), sr)
    on = np.concatenate([[levels[0]], pre])
    if on.shape[0] >= G:
        on = on[:max(G, 1)]
    else:
        if loop_node is not None and 0 <= int(loop_node) < r:
            blk, _ = _segments(cur, levels, times, shapes, range(int(loop_node), r), sr)
        else:
            blk = np.zeros(0)
        need = G - on.shape[0]
        fill = np.resize(blk, need) if blk.size else np.full(need, cur)
        on = np.concatenate([on, fill])
    rel, _ = _segments(float(on[-1]), levels, times, shapes, range(r, N), sr)
    return np.concatenate([on, rel])


def env_adsr(attack=0.01, decay=0.3, sustain=0.5, release=1.0, gate=1.0, peak=1.0, curve=-4.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Attack, decay to ``sustain`` x ``peak``, hold while the gate is open, release from the current level; curve -4."""
    return env([0.0, peak, peak * sustain, 0.0], [attack, decay, release], curve, sr, gate=gate, release_node=2)


def env_perc(attack=0.01, release=1.0, level=1.0, curve=-4.0, sr: int = DEFAULT_SR) -> np.ndarray:
    return env([0.0, level, 0.0], [attack, release], curve, sr)


# ---------------------------------------------------------------- smoothing, slew, sample and hold, triggers
@numba.njit(cache=True)
def _lag_kernel(x, bu, bd):
    y = np.empty_like(x)
    s = x[0]
    for i in range(x.shape[0]):
        v = x[i]
        b = bu if v > s else bd
        s = v + b * (s - v)
        y[i] = s
    return y


def _lag_coef(t, sr):
    return math.exp(LOG001 / (float(t) * sr)) if t > 0 else 0.0


def lag(x, time=0.1, sr: int = DEFAULT_SR, stages: int = 1) -> np.ndarray:
    """One-pole smoothing that covers 60 dB of a step in ``time`` s; ``stages`` 2 or 3 = cascades (an S-shaped glide)."""
    return lag_ud(x, time, time, sr, stages)


def lag_ud(x, up=0.1, down=0.1, sr: int = DEFAULT_SR, stages: int = 1) -> np.ndarray:
    """Like ``lag``, with its own time for rising and for falling input; each stage compares its own input and state."""
    y = np.ascontiguousarray(x, dtype=np.float64)
    for _ in range(max(1, int(stages))):
        y = _lag_kernel(y, _lag_coef(up, sr), _lag_coef(down, sr))
    return y


@numba.njit(cache=True)
def _slew_kernel(x, up, dn):
    y = np.empty_like(x)
    s = x[0]
    for i in range(x.shape[0]):
        d = x[i] - s
        if d > up:
            d = up
        elif d < -dn:
            d = -dn
        s += d
        y[i] = s
    return y


def slew(x, up=1.0, down=1.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Slope limiter: the output moves at most ``up`` (rising) or ``down`` (falling) units per second."""
    return _slew_kernel(np.ascontiguousarray(x, dtype=np.float64), float(up) / sr, float(down) / sr)


@numba.njit(cache=True)
def _varlag_kernel(x, n):
    y = np.empty_like(x)
    s = x[0]
    tgt = x[0]
    step = 0.0
    left = 0
    for i in range(x.shape[0]):
        if x[i] != tgt:
            tgt = x[i]
            left = n
            step = (tgt - s) / n
        if left > 0:
            s += step
            left -= 1
            if left == 0:
                s = tgt
        y[i] = s
    return y


def varlag(x, time=0.1, sr: int = DEFAULT_SR) -> np.ndarray:
    """Each change of the (stepped) input starts a straight ramp of ``time`` s from the current value to the new one."""
    return _varlag_kernel(np.ascontiguousarray(x, dtype=np.float64), max(1, int(round(float(time) * sr))))


@numba.njit(cache=True)
def _latch_kernel(x, trig, wrap):
    y = np.empty_like(x)
    h = x[0] if wrap else 0.0
    for i in range(x.shape[0]):
        p = trig[i - 1] if i > 0 else (trig[0] if wrap else 0.0)
        if (wrap and trig[i] < p) or (not wrap and trig[i] > 0.0 and p <= 0.0):
            h = x[i]
        y[i] = h
    return y


def latch(x, trig, mode: str = "edge") -> np.ndarray:
    """Sample and hold: ``x`` is taken when ``trig`` crosses from <= 0 to > 0 (edge), or when the phase ``trig`` wraps
    (wrap: it falls, as a phasor does at the end of its cycle, so the held value changes exactly at the cycle start)."""
    return _latch_kernel(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(trig, dtype=np.float64), mode == "wrap")


@numba.njit(cache=True)
def _phasor_kernel(inc, reset, phase0):
    y = np.empty_like(inc)
    ph = phase0
    prev = 0.0
    for i in range(inc.shape[0]):
        if reset[i] > 0.0 and prev <= 0.0:
            ph = phase0
        prev = reset[i]
        y[i] = ph
        ph += inc[i]
        ph -= math.floor(ph)
    return y


def phasor(freq, dur: float, sr: int = DEFAULT_SR, reset=None, phase0: float = 0.0) -> np.ndarray:
    """Ramp 0 -> 1 at ``freq`` Hz (a number or one value per sample), sent back to ``phase0`` on each rising edge of
    ``reset``."""
    n = samples(dur, sr)
    inc = np.ascontiguousarray(np.broadcast_to(np.asarray(freq, float), (n,)) / sr)
    rs = np.zeros(n) if reset is None else np.ascontiguousarray(np.broadcast_to(np.asarray(reset, float), (n,)))
    return _phasor_kernel(inc, rs, float(phase0) % 1.0)


@numba.njit(cache=True)
def _schmitt_kernel(x, lo, hi, dead):
    y = np.zeros(x.shape[0])
    state = 0.0
    wait = 0
    for i in range(x.shape[0]):
        if wait > 0:
            wait -= 1
        elif state == 0.0 and x[i] > hi:
            state = 1.0
            wait = dead
        elif state == 1.0 and x[i] < lo:
            state = 0.0
            wait = dead
        y[i] = state
    return y


def schmitt(x, lo=0.4, hi=0.6, dead: float = 0.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Trigger with hysteresis: goes to 1 above ``hi``, back to 0 below ``lo``, and ignores the input for ``dead`` s after
    each change (no chatter on a noisy level)."""
    return _schmitt_kernel(np.ascontiguousarray(x, dtype=np.float64), float(lo), float(hi), int(round(float(dead) * sr)))


# ---------------------------------------------------------------- slow random signals and the cycle-by-cycle vibrato
@numba.njit(cache=True)
def _cnoise_kernel(inc, lo_k, hi_k, kind, seed):
    np.random.seed(seed)
    n = inc.shape[0]
    y = np.empty(n)
    a = 2.0 * np.random.random() - 1.0
    b = 2.0 * np.random.random() - 1.0
    c = 2.0 * np.random.random() - 1.0
    d = 2.0 * np.random.random() - 1.0
    k = lo_k + (hi_k - lo_k) * np.random.random()
    ph = 0.0
    for i in range(n):
        if kind == 0:
            v = b
        elif kind == 1:
            v = b + ph * (c - b)
        else:
            t = ph
            v = 0.5 * (2.0 * b + (c - a) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t * t + (3.0 * b - a - 3.0 * c + d) * t * t * t)
            v = min(max(v, -1.0), 1.0)
        y[i] = v
        ph += inc[i] * k
        while ph >= 1.0:
            ph -= 1.0
            a, b, c, d = b, c, d, 2.0 * np.random.random() - 1.0
            k = lo_k + (hi_k - lo_k) * np.random.random()
    return y


def control_noise(dur: float, freq=4.0, interp: str = "linear", freq_max: float | None = None, lo: float = -1.0, hi: float = 1.0,
                  seed: int = 0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Slow random signal between ``lo`` and ``hi``: a new random value ``freq`` times a second (a number or one value per
    sample), held (hold), joined by lines (linear) or by a smooth cubic (cubic, continuous slope). With ``freq_max`` each
    segment draws its own rate between freq and freq_max (an irregular, never periodic drift)."""
    n = samples(dur, sr)
    inc = np.ascontiguousarray(np.broadcast_to(np.asarray(freq, float), (n,)) / sr)
    lo_k, hi_k = 1.0, 1.0
    if freq_max is not None:
        base = float(np.mean(inc)) * sr
        lo_k, hi_k = 1.0, float(freq_max) / max(base, 1e-9)
    kind = {"hold": 0, "linear": 1, "cubic": 2}[interp]
    v = _cnoise_kernel(inc, lo_k, hi_k, kind, int(seed))
    return lo + (hi - lo) * (0.5 + 0.5 * v)


@numba.njit(cache=True)
def _vibrato_kernel(n, inc0, depth, rate_var, depth_var, d0, att, seed):
    np.random.seed(seed)
    y = np.ones(n)
    ph = -1.0
    inc = inc0
    A = depth
    B = depth
    lvl = 0.0
    for i in range(n):
        if i < d0:
            continue
        lvl = min(1.0, lvl + att)
        if ph < 1.0:
            v = A * (1.0 - ph * ph)
        else:
            z = ph - 2.0
            v = B * (z * z - 1.0)
        y[i] = 1.0 + lvl * v
        ph += inc
        if ph >= 3.0:
            ph -= 4.0
            inc = inc0 * (1.0 + rate_var * (2.0 * np.random.random() - 1.0))
            A = depth * (1.0 + depth_var * (2.0 * np.random.random() - 1.0))
            B = depth * (1.0 + depth_var * (2.0 * np.random.random() - 1.0))
    return y


def vibrato_ratio(dur: float, rate: float = 6.0, depth: float = 0.02, delay: float = 0.0, onset: float = 0.0, rate_var: float = 0.04,
                  depth_var: float = 0.1, sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
    """Frequency ratio of a sung or played vibrato (1 = no deviation): parabolic half-cycles of +-``depth``, every cycle
    with its own rate (+-``rate_var``) and each half with its own depth (+-``depth_var``), silent for ``delay`` s then
    fading in over ``onset`` s. Multiply a frequency by it, or play a sound at that rate."""
    n = samples(dur, sr)
    att = 1.0 / max(1.0, float(onset) * sr)
    return _vibrato_kernel(n, 4.0 * float(rate) / sr, float(depth), float(rate_var), float(depth_var), samples(delay, sr) if delay > 0 else 0, att, int(seed))
