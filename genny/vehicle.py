"""Vehicle sounds: a physically-inspired combustion engine.

Registered into the sfx REGISTRY (imported at the bottom of sfx.py), so it is
used like any other sfx: {"type": "sfx", "kind": "car_engine", "params": {...}}.

Model (4-stroke):
  firing rate = rpm / 60 * cylinders / 2 pulses per second.
  Each firing is a short exhaust pressure pulse with a fixed per-cylinder
  strength (that uneven lope is what makes an engine sound like an engine and
  not a buzzer) plus small per-event amplitude/timing jitter.
  The pulse train is shaped by the exhaust: a pipe comb (delay = pipe length),
  three low body resonances and a two-stage muffler lowpass that opens a little
  with throttle `load`. On top: a low exhaust burble and intake breath, both
  modulated by the firing. `bright` opens the muffler and adds a combustion
  rasp for sporty cars; at 0 nothing sits above ~2.5 kHz, so it never pierces.

Render loops at a few fixed rpm values and crossfade/pitch between them at
runtime (the standard "engine bank" technique) — one sample pitched across the
whole rev range sounds like a tape speeding up.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, samples
from .sfx import sfx


def _comb(x: np.ndarray, delay_s: float, fb: float, sr: int) -> np.ndarray:
    d = max(1, int(delay_s * sr))
    a = np.zeros(d + 1)
    a[0] = 1.0
    a[d] = -fb
    return lfilter([1.0], a, x)


@sfx(
    "car_engine",
    "Physically-modelled combustion engine loop (firing pulses, exhaust pipe, intake, valves).",
    rpm=(1500, "crank speed"),
    cylinders=(4, "cylinder count (4-stroke)"),
    load=(0.5, "throttle 0..1: fuller, more intake breath"),
    rough=(0.25, "0..1 uneven firing / jitter"),
    pipe=(0.004, "exhaust pipe delay in s (longer = lower, throatier)"),
    body=(1.0, "0..1 exhaust body resonance amount"),
    bright=(0.0, "0..1 open muffler / sporty rasp (0 = road car heard from the cabin)"),
    seed=(7, "per-engine character"),
    dur=(3.0, "s"),
    model=("classic", "classic | ptr (Doerfler 2026 pulse-train-resonator: shaped exhaust pulses per cylinder bank, "
                      "a Karplus-Strong manifold per bank and a shared tailpipe)"),
    exhaust=("auto", "ptr only: the exhaust the PTR paper learned, inline4 | v8 (low, resonant) | v8_metal (mid-range, "
                     "metallic); auto = inline4 below 6 cylinders, else v8"),
)
def car_engine(sr=DEFAULT_SR, rpm=1500, cylinders=4, load=0.5, rough=0.25, pipe=0.004, body=1.0, bright=0.0, seed=7, dur=3.0,
               model="classic", exhaust="auto"):
    if model == "ptr":
        return _ptr_engine(sr, float(rpm), max(1, int(cylinders)), float(np.clip(load, 0, 1)), float(np.clip(rough, 0, 1)),
                           float(pipe), float(np.clip(bright, 0, 1)), int(seed), float(dur), str(exhaust))
    rng = np.random.default_rng(int(seed))
    cyl = max(1, int(cylinders))
    rpm = float(rpm)
    load = float(np.clip(load, 0, 1))
    rough = float(np.clip(rough, 0, 1))
    n = samples(dur, sr)

    fire_rate = rpm / 60.0 * cyl / 2.0
    interval = 1.0 / fire_rate
    # Fixed per-cylinder strengths: the lope. Rough engines differ more.
    cyl_gain = 1.0 + rng.uniform(-1, 1, cyl) * (0.08 + 0.3 * rough)
    count = int(dur * fire_rate) + 4
    idx = np.arange(count)
    # Slow rpm wander (a real engine never holds exactly still)
    wander = 1.0 + 0.012 * np.sin(2 * np.pi * 0.37 * idx * interval + rng.uniform(0, 6.3))
    times = np.cumsum(np.full(count, interval) / wander) - interval
    times += rng.normal(0, 1, count) * interval * (0.004 + 0.03 * rough)
    amps = cyl_gain[idx % cyl] * (1.0 + rng.normal(0, 1, count) * (0.03 + 0.1 * rough))
    # Occasional weak firing on a rough engine
    miss = rng.random(count) < 0.02 * rough
    amps[miss] *= 0.35
    times = np.clip(times, 0, None)

    # Rounded exhaust pressure pulse. A razor-sharp pulse reads as a buzzy
    # click train; a real exhaust pulse has a soft ~1 ms rise and a few-ms
    # fall that shortens as the revs climb.
    rise = 0.0007
    tau = min(interval * 0.35, 0.0012 + 0.0025 * (1 - load) * (1500 / max(rpm, 600)))
    w = int(min(interval * 0.9, rise + tau * 6) * sr) + 2
    kk = np.arange(w) / sr
    shape = np.where(kk < rise, 0.5 - 0.5 * np.cos(np.pi * kk / rise), np.exp(-(kk - rise) / tau))
    trig = np.zeros(n + w)
    pos = (times * sr).astype(int)
    ok = pos < n
    np.add.at(trig, pos[ok], amps[ok])
    press = np.convolve(trig, shape)[:n]
    press = press - F.lowpass(press, 20, sr)  # remove the DC of a unipolar train

    # Exhaust: pipe comb + low body resonances, then the muffler (two
    # 2nd-order lowpasses) — what you hear of a car is mostly below 1 kHz.
    ex = _comb(press, float(pipe), 0.35 + 0.25 * body, sr)
    res = (F.bandpass(ex, 85, sr, 1.0) * 0.55
           + F.bandpass(ex, 170, sr, 1.4) * 0.8
           + F.bandpass(ex, 320, sr, 1.8) * 0.8
           + F.bandpass(ex, 650, sr, 2.0) * 0.5)
    ex = ex * (1 - 0.5 * body) + res * body * 1.5
    bright = float(np.clip(bright, 0, 1))
    muffler = (600 + 900 * load + 0.12 * rpm) * (1 + 1.5 * bright)
    ex = F.lowpass(ex, muffler, sr, 0.7, order=2)
    ex = F.lowpass(ex, 3200 + 3000 * bright, sr, 0.6, order=2)
    ex = ex / (np.sqrt(np.mean(ex ** 2)) + 1e-9) * 0.3

    # Firing envelope drives a low exhaust burble and the intake breath
    fenv = F.lowpass(np.convolve(trig, np.exp(-np.arange(int(0.006 * sr)) / (0.002 * sr)))[:n], fire_rate * 2, sr)
    fenv = np.clip(fenv / (np.max(fenv) + 1e-9), 0, None)
    burble = F.bandpass(O.pink(n, seed=int(seed) + 5), 260 + 120 * load, sr, 0.9) * fenv
    burble = F.lowpass(burble, 900, sr)
    burble = burble / (np.sqrt(np.mean(burble ** 2)) + 1e-9) * (0.03 + 0.05 * load)
    intake = F.lowpass(F.bandpass(O.pink(n, seed=int(seed) + 11), 500 + 350 * load, sr, 0.7), 1400 + 2500 * bright, sr) * (0.5 + 0.5 * fenv)
    intake = intake / (np.sqrt(np.mean(intake ** 2)) + 1e-9) * (0.012 + 0.03 * load)
    # Sporty rasp only when asked for
    rasp = F.bandpass(O.white(n, seed=int(seed) + 7), 1500, sr, 0.8) * fenv
    rasp = rasp / (np.sqrt(np.mean(rasp ** 2)) + 1e-9) * 0.05 * bright

    y = ex + burble + intake + rasp
    y = np.tanh(y * (1.1 + 0.4 * load)) * 0.85
    y = F.dc_block(y, sr)
    return y


# Doerfler et al., "Pulse-Train-Resonator", arXiv 2603.09391 (2026): firing orders of their engines and pulse shape
_FIRING = {4: (1, 3, 4, 2), 6: (1, 5, 3, 6, 2, 4), 8: (1, 5, 4, 8, 6, 3, 7, 2)}


def _ks(x, L, g, a, b):
    """Karplus-Strong exhaust resonator y[n] = x[n] + g (a y[n-L] + b y[n-L-1]) (eqs 8-9)."""
    den = np.zeros(L + 2)
    den[0], den[L], den[L + 1] = 1.0, -g * a, -g * b
    return lfilter([1.0], den, x)


# Resonators learned by PTR on its three datasets (best checkpoints of github.com/rdoerfler/ptr-model, CC BY-NC 4.0,
# read out in research/bibliographie_genny/data/doerfler_ptr/resonators.md): per resonator (L at 16 kHz, a1, a2) of
# y[n] = x[n] - a1 y[n-L] - a2 y[n-L-1], the loop gain folded in. Bank a, bank b, shared tailpipe.
PTR_EXHAUST = {
    "inline4": ((113, 0.0759, -0.0702), (77, 0.2168, -0.2148), (34, 0.0336, 0.1825)),
    "v8": ((91, 0.0489, -0.0460), (49, 0.2607, -0.2593), (40, 0.2873, 0.1838)),
    "v8_metal": ((61, -0.3678, -0.3696), (111, 0.0609, -0.0604), (34, 0.1349, 0.2394)),
}


def _ptr_engine(sr, rpm, cyl, load, rough, pipe, bright, seed, dur, exhaust="auto"):
    """PTR (Sec. III-IV): every cylinder fires once per engine cycle (f0 = rpm / 120) a bipolar pulse
    P = E(phi) sum_k a_k sin(k phi~), E = (1 - e^{-alpha phi}) e^{-beta phi} (eq. 4), a_k ~ e^{-0.5 k lambda} (eq. 6), phase
    bent as phi~ = 2 pi (phi / 2 pi)^nu by the hot leading edge (eq. 5); cylinders are summed per bank (V engines: two
    banks in firing order), each bank through its own Karplus-Strong manifold, then a shared tailpipe (Sec. IV-C4);
    turbulence modulates the pulse with throttle, g_thr = max(load, eps)^0.7 (eqs 1, 7).
    The manifolds and the tailpipe are the resonators PTR learned (``PTR_EXHAUST``), delays rescaled from 16 kHz;
    ``pipe`` / 0.004 scales the tailpipe length. UNSOURCED (learned per frame by the network, not stored as constants):
    alpha, beta, lambda, nu."""
    rng = np.random.default_rng(seed)
    n = samples(dur, sr)
    t = np.arange(n) / sr
    f0 = rpm / 120.0
    order = _FIRING.get(cyl, tuple(range(1, cyl + 1)))
    banks = 2 if cyl >= 6 else 1
    gthr = max(load, 1e-3) ** 0.7
    lam, nu = 0.35 - 0.2 * bright, 0.75
    alpha, beta = 40.0, 6.0 + 4.0 * (1.0 - load)
    K = 24
    ak = np.exp(-0.5 * np.arange(1, K + 1) * lam)
    out = [np.zeros(n) for _ in range(banks)]
    for slot, c in enumerate(order):
        phase = (f0 * t + slot / len(order) + rng.normal(0.0, 0.004 + 0.02 * rough)) % 1.0
        phi = 2 * np.pi * phase
        bent = 2 * np.pi * phase ** nu
        pulse = (1 - np.exp(-alpha * phi)) * np.exp(-beta * phi) * (np.sin(np.outer(bent, np.arange(1, K + 1))) @ ak)
        gain = 1.0 + rng.uniform(-1, 1) * (0.08 + 0.3 * rough)
        turb = 1.0 + 0.3 * gthr * F.lowpass(rng.standard_normal(n), 400.0, sr)
        out[(c - 1) % banks if banks > 1 else 0] += gain * pulse * turb
    if exhaust not in PTR_EXHAUST:
        exhaust = "inline4" if cyl < 6 else "v8"
    *manifolds, tail = PTR_EXHAUST[exhaust]
    y = np.zeros(n)
    for x, (L, a1, a2) in zip(out, manifolds):
        y += _ks(x, max(2, int(round(L * sr / 16000))), 1.0, -a1, -a2)
    L, a1, a2 = tail
    y = _ks(y, max(2, int(round(L * sr / 16000 * pipe / 0.004))), 1.0, -a1, -a2)
    y = F.lowpass(y, (700 + 1100 * load + 0.1 * rpm) * (1 + 1.5 * bright), sr, 0.7, order=2)
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.3
    return F.dc_block(np.tanh(y * (1.1 + 0.4 * load)) * 0.85, sr)
