"""Drum synthesis. Each takes (sr, vel, **params) and returns a mono buffer."""
from __future__ import annotations

import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, samples
from .env import perc

REGISTRY: dict[str, dict] = {}


def drum(name: str, desc: str, **params):
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "desc": desc, "params": params}
        return fn
    return deco


def _dec(n, tau, sr):
    return np.exp(-np.arange(n) / (tau * sr))


TR808_BRIDGED_T = {                                     # (R1 ohm, R2 ohm, C1 F, C2 F); toms and congas at R1 = 333.3 (pot 333.3-500)
    "bass_drum": (53.8e3, 1e6, 0.015e-6, 0.015e-6),
    "snare_low": (680.0, 820e3, 0.027e-6, 0.027e-6),
    "snare_high": (2.2e3, 1e6, 0.0068e-6, 0.0068e-6),
    "low_tom": (333.3, 820e3, 0.056e-6, 0.059e-6),
    "low_conga": (333.3, 820e3, 0.056e-6, 0.012e-6),
    "mid_tom": (333.3, 820e3, 0.0352e-6, 0.039e-6),
    "mid_conga": (333.3, 820e3, 0.027e-6, 0.012e-6),
    "high_tom": (333.3, 820e3, 0.027e-6, 0.0276e-6),
    "high_conga": (333.3, 820e3, 0.027e-6, 0.0056e-6),
    "rimshot": (5.6e3, 1e6, 0.0047e-6, 0.0047e-6),
    "clave": (1e3, 820e3, 0.0022e-6, 0.0022e-6),
}


def tom_r1(knob: float) -> float:
    """R1 (ohm) of a TR-808 tom / conga bridged-T at tuning knob position 0..1: (-x^2 + x + 2) / (x + 2) * 500."""
    x = float(np.clip(knob, 0.0, 1.0))
    return (-x * x + x + 2.0) / (x + 2.0) * 500.0


def bridged_t(voice: str, r1: float | None = None) -> tuple[float, float, float]:
    """Op-amp bridged-T resonator of a TR-808 voice: (fc Hz, Q, ring time constant s). fc = 1 / (2 pi sqrt(R1 R2 C1 C2)),
    Q = sqrt(R2 / R1) / (sqrt(C1 / C2) + sqrt(C2 / C1)), tau = Q / (pi fc). `r1` replaces R1 (tom / conga knob)."""
    R1, R2, C1, C2 = TR808_BRIDGED_T[voice]
    R1 = R1 if r1 is None else float(r1)
    fc = 1.0 / (2 * np.pi * np.sqrt(R1 * R2 * C1 * C2))
    q = np.sqrt(R2 / R1) / (np.sqrt(C1 / C2) + np.sqrt(C2 / C1))
    return float(fc), float(q), float(q / (np.pi * fc))


TR808_PULSE_SHAPER = (4.7e3, 100e3, 0.015e-6, 2.52e-9, 25.85e-3)     # R162, R163, C40, diode Is, VT
TR808_TRIGGER_MS, TR808_ACCENT_V = 1.0, (4.0, 14.0)
TR808_ENV_TAU = 1e6 * 0.033e-6                          # R161 C39: the envelope generator's fall
TR808_BD_ATTACK = (47e3 + 6.8e3) / 6.8e3               # R_eff from R165 + R166 to R166 while the envelope conducts


TR808_BD_FEEDBACK = dict(R164=47e3, R165=47e3, R166=6.8e3, R167=1e6, R169=47e3, R170=470e3, VR6=500e3,
                         C41=0.015e-6, C42=0.015e-6, C43=33e-6, R161=1e6)
TR808_BD_HZ = 49.5                                      # the loop's own ring: R161 || (R165 + R166) || R170
TR808_SIGH = (14.3150, -0.5560, 1.4765e-5)              # Q43 collector current fit: alpha 1/V, V0 V, m A/V


def sigh_current(v_comm, alpha: float = TR808_SIGH[0], v0: float = TR808_SIGH[1], m: float = TR808_SIGH[2]):
    """Current Q43 draws (A, <= 0) when the bridged-T common node swings below about a diode drop:
    i_C = -log(1 + exp(-alpha (V - V0))) m / alpha. It lowers R_eff, so the ring starts sharp and sighs down."""
    z = -alpha * (np.asarray(v_comm, float) - v0)
    return -np.logaddexp(0.0, z) * m / alpha
TR808_BD_OUTPUT = dict(R171=220.0, R172=10e3, VR5=10e3, C45=0.1e-6, VR4=100e3, C47=0.47e-6, R176=100e3, R177=82e3,
                       C49=0.47e-6, R174=100e3, C50=1e-6)


def _bd_deriv(x, vp, Rs, c41, c42, p):
    v41, v42, v43 = x
    O1 = vp - v41 - v42
    J = vp - v41
    vr6 = max(p["VR6"], 100.0)
    O2 = -(O1 / p["R164"] + v43 / vr6) / (1.0 / p["R169"] + 1.0 / vr6)
    d41 = -(v41 + v42) / (p["R167"] * c41)
    d42 = (c41 * d41 - J / Rs - (J - O2) / p["R170"]) / c42
    d43 = -(O2 + v43) / (vr6 * p["C43"])
    return np.array([d41, d42, d43])


def bd_resonator(vp: np.ndarray, sr: int = DEFAULT_SR, knob: float = 0.5, tune: float = TR808_BD_HZ,
                 attack: np.ndarray | None = None, sigh: bool = False) -> np.ndarray:
    """TR-808 bass drum bridged-T resonator with its decay feedback loop (ideal op-amps): C41, C42 and R165 + R166 to
    ground (R161 to the envelope node in parallel) form the bridged-T around the first op-amp (R167 feedback); the second
    op-amp inverts its output through R164 into R169 || (VR6 + C43) and returns it through R170, so the decay knob
    VR6 = knob x 500 k sets how much of the ring is fed back. `attack` (0..1 per sample) shorts R165 as the envelope
    generator conducts. `sigh`: Q43 draws `sigh_current` through R165 when the common node swings low, R_eff falls and
    the ring starts sharp, then settles. `tune` scales C41, C42 until the loop (knob included, attack and sigh off)
    rings at ``tune`` Hz; at TR808_BD_HZ the parts are the stock ones.
    Trapezoidal rule on the three capacitor voltages (the Q43 current one step behind); returns v_b.t.r. in volts."""
    p = dict(TR808_BD_FEEDBACK)
    p["VR6"] = float(np.clip(knob, 0.0, 1.0)) * TR808_BD_FEEDBACK["VR6"]
    tune = max(float(tune), 1.0)
    par = lambda r: r * p["R161"] / (r + p["R161"])
    sc = bridged_t("bass_drum")[0] / tune                         # f ~ 1/sqrt(C41 C42) = 1/sc
    eye = np.eye(3)
    for _ in range(3):                                            # the R170 loop pulls the ring off the bare bridged-T
        A = np.stack([_bd_deriv(e, 0.0, par(p["R165"] + p["R166"]), p["C41"] * sc, p["C42"] * sc, p) for e in eye], axis=1)
        sc *= float(np.max(np.abs(np.linalg.eigvals(A).imag))) / (2 * np.pi * tune)
    c41, c42 = p["C41"] * sc, p["C42"] * sc
    n, dt = len(vp), 1.0 / sr
    att = np.zeros(n) if attack is None else np.clip(np.asarray(attack, float), 0.0, 1.0)
    out = np.zeros(n)
    x = np.zeros(3)
    src = np.array([0.0, -dt / c42, 0.0])
    k_sigh = p["R165"] / (p["R165"] + p["R166"])
    cache = {}
    for i in range(n):
        Rs = par(p["R166"] + p["R165"] * (1.0 - att[i]))
        key = round(Rs)
        if key not in cache:
            A = np.stack([_bd_deriv(e, 0.0, Rs, c41, c42, p) for e in eye], axis=1)
            B = _bd_deriv(np.zeros(3), 1.0, Rs, c41, c42, p)
            cache[key] = (np.linalg.inv(eye - 0.5 * dt * A), eye + 0.5 * dt * A, B)
        Mi, Mp, B = cache[key]
        v0, v1 = (vp[i - 1] if i else 0.0), vp[i]
        rhs = Mp @ x + 0.5 * dt * B * (v0 + v1)
        if sigh:
            rhs = rhs + src * (1.0 - att[i]) * k_sigh * float(sigh_current(v0 - x[0]))
        x = Mi @ rhs
        out[i] = v1 - x[0] - x[1]
    return out


def bd_output_filter(x: np.ndarray, sr: int = DEFAULT_SR, tone: float = 0.5, level: float = 1.0) -> np.ndarray:
    """TR-808 bass drum tone stage, simplified to its four first-order sections (bilinear): tone lowpass
    (R171 + c R172 VR5 / (R172 + c VR5)) C45, level highpass VR4 C47 (gain l), highpass R176 C49 / (R176 + R177) C49,
    output highpass R174 C50."""
    from scipy.signal import bilinear, lfilter
    p = TR808_BD_OUTPUT
    c = float(np.clip(tone, 0.0, 1.0))
    lv = float(np.clip(level, 0.0, 1.0))
    stages = (([0.0, 1.0], [(p["R171"] + c * p["R172"] * p["VR5"] / (p["R172"] + c * p["VR5"])) * p["C45"], 1.0]),
              ([lv * p["VR4"] * p["C47"], 0.0], [p["VR4"] * p["C47"], 1.0]),
              ([p["R176"] * p["C49"], 0.0], [(p["R176"] + p["R177"]) * p["C49"], 1.0]),
              ([p["R174"] * p["C50"], 0.0], [p["R174"] * p["C50"], 1.0]))
    y = np.asarray(x, float)
    for b, a in stages:
        bz, az = bilinear(b, a, sr)
        y = lfilter(bz, az, y)
    return y


def pulse_shaper(vin: float = 4.0, sr: int = DEFAULT_SR, dur: float = 0.005, diode: bool = True) -> np.ndarray:
    """TR-808 bass drum pulse shaper: a ``TR808_TRIGGER_MS`` rectangular trigger of ``vin`` volts through C40 || R163 in
    series with R162 to ground, D53 across R162 (H = (R162 R163 C40 s + R162) / (R162 R163 C40 s + R162 + R163)).
    Backward Euler at 16 x the rate, Newton on the diode; returns v across R162 in volts, ``dur`` s long."""
    R162, R163, C40, Is, VT = TR808_PULSE_SHAPER
    os_ = 16
    dt = 1.0 / (sr * os_)
    n = int(round(dur * sr)) * os_
    on = int(round(TR808_TRIGGER_MS * 1e-3 / dt))
    out = np.zeros(n)
    u = v = 0.0
    g = C40 / dt + 1.0 / R162 + 1.0 / R163
    for i in range(n):
        vi = vin if i < on else 0.0
        v = (C40 * (vi - u) / dt + vi / R163) / g                  # the linear solution: exact while D53 is off
        if diode:
            v = max(v, -1.0)                                     # start a forward-biased D53 near its drop
        for _ in range(100):
            e = np.exp(min(-v / VT, 60.0)) if diode else 1.0
            F = C40 * (vi - v - u) / dt - v / R162 + (Is * (e - 1.0) if diode else 0.0) + (vi - v) / R163
            dF = -g - (Is / VT * e if diode else 0.0)
            step = F / dF
            v -= step
            if abs(step) < 1e-9:
                break
        u = vi - v
        out[i] = v
    return out.reshape(-1, os_).mean(axis=1)


@drum("kick808", "Long boomy 808 kick.", tune=(TR808_BD_HZ, "Hz (the bass drum bridged-T loop; 49.5 = stock parts)"),
      decay=(1.2, "s"),
      model=("classic", "classic: swept sine | circuit: the trigger through the pulse shaper into the bridged-T and its "
                        "decay feedback loop, retuned by the envelope generator during the attack, then the tone stage"),
      knob=(0.5, "circuit: decay knob 0..1 (VR6: how much of the ring the second op-amp feeds back)"),
      tone=(0.5, "circuit: tone knob 0..1 (the tone stage lowpass)"),
      sigh=(1, "circuit: 1 = the pitch sigh (Q43 leakage: the ring starts a few Hz sharp and settles over ~0.2 s); 0 = off"))
def kick808(sr=DEFAULT_SR, vel=1.0, tune=TR808_BD_HZ, decay=1.2, model="classic", knob=0.5, tone=0.5, sigh=1):
    n = samples(decay, sr)
    if model == "circuit":
        lo, hi = TR808_ACCENT_V
        p = pulse_shaper(lo + (hi - lo) * float(np.clip(vel, 0.0, 1.0)), sr)
        x = np.zeros(n)
        x[:min(len(p), n)] = p[:n]
        y = bd_output_filter(bd_resonator(x, sr, knob, tune, _dec(n, TR808_ENV_TAU, sr), bool(sigh)), sr, tone)
        y = y / (np.max(np.abs(y)) + 1e-12) * (0.6 + 0.4 * vel)
        return y * perc(decay, 0.0005, sr, curve=1.0)
    f = tune + tune * 4 * _dec(n, 0.02, sr)
    y = np.tanh(O.sine(f, n, sr) * 2.5 * _dec(n, decay / 2.5, sr))
    return y * perc(decay, 0.001, sr, curve=1.0) * (0.6 + 0.4 * vel)


CLAP_RJ_EVENTS = ((0.0, 1, 0.0), (0.0, 0, 34.0), (10.5, 1, 0.5), (11.0, 0, 34.0), (23.0, 1, 0.0), (23.0, 0, 34.0),
                  (31.0, 1, 0.0), (31.0, 0, 34.0), (36.0, 1, 0.0), (36.5, 0, 230.0), (65.0, 0, 500.0), (120.0, 0, 2000.0))
CLAP_RJ_BAND = (1000.0, 1.5, 800.0)


def vline(events, n: int, sr: int, peak: float = 1.0) -> np.ndarray:
    """Pd ``vline~``: (start ms, target, ramp ms) events; each ramp starts from the current value, a later event cuts it."""
    env = np.zeros(n)
    v = 0.0
    for i, (t0, tgt, ramp) in enumerate(events):
        tgt = tgt * peak
        s0 = min(int(round(t0 * sr / 1000.0)), n)
        s1 = min(int(round(events[i + 1][0] * sr / 1000.0)), n) if i + 1 < len(events) else n
        r = ramp * sr / 1000.0
        k = np.arange(s1 - s0 + 1)
        seg = np.full(len(k), float(tgt)) if r <= 0 else v + (tgt - v) * np.minimum(k / r, 1.0)
        env[s0:s1] = seg[:-1]
        v = float(seg[-1])
    return env


@drum("clap", "Hand clap (multi-burst noise).", decay=(0.3, "s"),
      model=("classic", "classic (four bursts 11 ms apart, band 1.5 kHz) | rj (five bursts at 0, 10.5, 23, 31, 36 ms with "
                        "34 ms ramps then a long tail, envelope^8, resonator 1 kHz Q 1.5, high-pass 800 Hz; `decay` unused)"))
def clap(sr=DEFAULT_SR, vel=1.0, decay=0.3, model="classic"):
    if model == "rj":
        n = samples(0.6, sr)
        fc, q, hp = CLAP_RJ_BAND
        e = vline(CLAP_RJ_EVENTS, n, sr, float(np.clip(vel, 0.0, 1.0)) ** 0.125)
        y = F.highpass(F.bandpass(O.white(n, seed=4), fc, sr, q) * e ** 8, hp, sr)
        return y / (np.max(np.abs(y)) + 1e-12) * (0.5 + 0.5 * vel)
    if model != "classic":
        raise ValueError("clap model must be classic or rj")
    n = samples(decay + 0.05, sr)
    nz = F.bandpass(O.white(n, seed=4), 1500, sr, 0.8)
    env = np.zeros(n)
    for i, t in enumerate((0.0, 0.011, 0.022, 0.033)):
        s = samples(t, sr)
        env[s:] += _dec(n - s, 0.006 if i < 3 else decay / 2.5, sr)
    return np.tanh(nz * env * 2.0) * (0.5 + 0.5 * vel)


@drum("hihat", "Closed hi-hat (metallic noise centred on `tone`).", decay=(0.08, "s"), tone=(8000, "Hz brightness (6000 dark .. 11000 crisp)"))
def hihat(sr=DEFAULT_SR, vel=1.0, decay=0.08, tone=8000):
    n = samples(decay + 0.02, sr)
    metal = sum(O.square(f, n, sr) for f in (296, 410, 522, 663, 781, 934)) / 6
    y = F.highpass(metal * 0.5 + O.white(n, seed=6) * 0.5, tone * 0.7, sr, order=2)
    y = F.lowpass(F.peak(y, tone, sr, 1.5, 3), min(tone * 1.4, 0.45 * sr), sr, 0.7)
    return y * _dec(n, decay / 3, sr) * perc(decay + 0.02, 0.0005, sr, curve=1.0) * 1.15 * (0.5 + 0.5 * vel)


@drum("openhat", "Open hi-hat.", decay=(0.4, "s"))
def openhat(sr=DEFAULT_SR, vel=1.0, decay=0.4):
    return hihat(sr, vel, decay=decay, tone=7500)


@drum("rim", "Rimshot / side stick.")
def rim(sr=DEFAULT_SR, vel=1.0):
    n = samples(0.08, sr)
    y = O.square(1800, n, sr) * _dec(n, 0.004, sr) + O.sine(450, n, sr) * _dec(n, 0.012, sr) * 0.8
    y += F.highpass(O.white(n, seed=9), 4000, sr) * _dec(n, 0.003, sr) * 0.5
    return np.tanh(y * 1.5) * 0.8 * (0.5 + 0.5 * vel)


@drum("cowbell", "808 cowbell.", decay=(0.3, "s"))
def cowbell(sr=DEFAULT_SR, vel=1.0, decay=0.3):
    n = samples(decay, sr)
    y = (O.pulse(540.0, n, sr, 0.4798) + O.pulse(800.0, n, sr, 0.4798)) * 0.5
    y = F.bandpass(y, 700, sr, 1.0) * _dec(n, decay / 4, sr)
    return np.tanh(y * 3) * 0.7 * (0.5 + 0.5 * vel)


@drum("crash", "Crash cymbal.", decay=(1.5, "s"))
def crash(sr=DEFAULT_SR, vel=1.0, decay=1.5):
    n = samples(decay, sr)
    metal = sum(O.square(f * 1.7, n, sr) for f in (296, 410, 522, 663, 781, 934)) / 6
    y = F.highpass(metal * 0.4 + O.white(n, seed=10) * 0.6, 3500, sr, order=2)
    y = F.lowpass(F.peak(y, 6000, sr, 0.8, 3), 9500, sr, 0.7)
    return y * _dec(n, decay / 3, sr) * perc(decay, 0.001, sr, curve=1.0) * 0.9 * (0.5 + 0.5 * vel)


@drum("ride", "Ride cymbal ping.", decay=(1.0, "s"))
def ride(sr=DEFAULT_SR, vel=1.0, decay=1.0):
    n = samples(decay, sr)
    ping = O.sine(2600, n, sr) * _dec(n, 0.12, sr) * 0.2 + O.fm(1800, n, sr, ratio=1.41, index=1.6) * _dec(n, decay / 3, sr) * 0.3
    wash = F.lowpass(F.highpass(O.white(n, seed=12), 5000, sr), 10000, sr) * _dec(n, decay / 2.5, sr) * 0.3
    return (ping + wash) * perc(decay, 0.001, sr, curve=1.0) * 0.8 * (0.5 + 0.5 * vel)


@drum("shaker", "Shaker / maraca.", decay=(0.12, "s"))
def shaker(sr=DEFAULT_SR, vel=1.0, decay=0.12):
    n = samples(decay, sr)
    y = F.bandpass(O.white(n, seed=13), 7000, sr, 1.5)
    env = perc(decay, 0.02, sr, curve=2.0)
    return y * env * 0.8 * (0.5 + 0.5 * vel)


@drum("tambourine", "Tambourine jingle.", decay=(0.25, "s"))
def tambourine(sr=DEFAULT_SR, vel=1.0, decay=0.25):
    n = samples(decay, sr)
    y = np.zeros(n)
    for i, f in enumerate((4100, 5200, 6100, 7300)):   # jingle pairs (plain sines: FM this high only aliased)
        y += (O.sine(f, n, sr) + 0.6 * O.sine(f * 1.31, n, sr)) * _dec(n, decay / (3 + i), sr) * 0.12
    y += F.lowpass(F.highpass(O.white(n, seed=14), 4500, sr), 10000, sr) * _dec(n, decay / 4, sr) * 0.4
    return y * perc(decay, 0.001, sr, curve=1.0) * 1.05 * (0.5 + 0.5 * vel)


@drum("woodblock", "Wood block / clave.", tune=(1100, "Hz"))
def woodblock(sr=DEFAULT_SR, vel=1.0, tune=1100):
    n = samples(0.12, sr)
    y = O.sine(tune, n, sr) * _dec(n, 0.02, sr) + O.sine(tune * 2.4, n, sr) * _dec(n, 0.01, sr) * 0.4
    return np.tanh(y * 1.5) * perc(0.12, 0.0005, sr, curve=1.0) * 0.8 * (0.5 + 0.5 * vel)


@drum("taiko", "Big deep taiko / cinematic drum.", tune=(60, "Hz"), decay=(0.9, "s"))
def taiko(sr=DEFAULT_SR, vel=1.0, tune=60, decay=0.9):
    n = samples(decay, sr)
    f = tune * (1 + 1.5 * _dec(n, 0.05, sr))
    body = O.sine(f, n, sr) * _dec(n, decay / 3, sr) + 0.3 * O.sine(f * 1.5, n, sr) * _dec(n, decay / 6, sr)
    skin = F.bandpass(O.brown(n, seed=15), 300, sr, 0.8) * _dec(n, 0.08, sr) * 0.8
    return np.tanh((body + skin) * 2.0) * perc(decay, 0.002, sr, curve=1.0) * (0.6 + 0.4 * vel)


@drum("zap_kick", "Electronic laser-kick hybrid.", decay=(0.3, "s"))
def zap_kick(sr=DEFAULT_SR, vel=1.0, decay=0.3):
    n = samples(decay, sr)
    f = 50 + 1500 * _dec(n, 0.03, sr)
    return np.tanh(O.sine(f, n, sr) * 2 * _dec(n, decay / 3, sr)) * perc(decay, 0.0005, sr, curve=1.0) * (0.6 + 0.4 * vel)


from . import percussion  # noqa: E402,F401  (registers more drums)


def render_drum(name: str, sr: int = DEFAULT_SR, vel: float = 1.0, **params) -> np.ndarray:
    if name not in REGISTRY:
        raise ValueError(f"unknown drum {name!r}; run `genny list drums`")
    entry = REGISTRY[name]
    unknown = [k for k in params if k not in entry["params"]]
    if unknown:
        raise ValueError(f"drum {name!r} does not accept params {unknown}; allowed: {list(entry['params'])}")
    y = entry["fn"](sr, vel, **params)
    k = min(y.shape[0] // 2, samples(0.004, sr))   # no drum ever ends on a step (gong and guiro did: a click)
    if k > 1:
        y = y.copy()
        y[-k:] *= np.linspace(1.0, 0.0, k)
    return y

# v0.6 physical migration: particle percussion ----------------------------------
@drum("shaker", "Physical PhISEM maraca/shaker: stochastic particle collisions in a resonant shell.", decay=(0.12, "s"))
def shaker(sr=DEFAULT_SR, vel=1.0, decay=0.12):
    from .physics import ParticleMaterial
    y = ParticleMaterial("Maraca", resonance_scale=1.0, damping=1.0).shake(max(.06,float(decay)), energy=.35+.65*vel, sr=sr, seed=13)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.45 + 0.4 * vel)   # particle peaks vary by 10 dB with the seed


@drum("tambourine", "Physical PhISEM tambourine/jingle particle model.", decay=(0.25, "s"))
def tambourine(sr=DEFAULT_SR, vel=1.0, decay=0.25):
    from .physics import ParticleMaterial
    y = ParticleMaterial("Tambourine").shake(max(.10,float(decay)), energy=.35+.65*vel, sr=sr, seed=14)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.45 + 0.4 * vel)


@drum("woodblock", "Physical modal wooden-bar/woodblock strike.", tune=(1100, "Hz"))
def woodblock(sr=DEFAULT_SR, vel=1.0, tune=1100):
    from .physics import TunedPercussion
    y = TunedPercussion("wood", hardness=.82).note(float(tune), .16, velocity=vel, sr=sr, seed=15)
    return y * .8

# v0.7 physical membrane / plate migration ------------------------------------
@drum("tom", "Physical circular-membrane tom with air-cavity coupling.", tune=(120,"Hz"), decay=(0.35,"s"),
      under=(0.0, "0..1 level of a second mic under the drum (head seen from below: inverted, later, darker)"),
      flip=(0, "1 = bottom mic polarity flipped back (full lows); 0 = raw, lows partly cancel"))
def tom(sr=DEFAULT_SR, vel=1.0, tune=120, decay=.35, under=0.0, flip=0):
    from .physics import MembraneDrum
    from .percussion import double_mic
    y = MembraneDrum(float(tune), .16, damping=1.15/max(float(decay),.08), air_coupling=.42).strike(
        max(.25,float(decay)+.12),velocity=vel,position=.26,stroke="tone",sr=sr,seed=8)
    return double_mic(y, sr, under, bool(flip))


@drum("taiko", "Physical large membrane/cavity taiko.", tune=(60,"Hz"), decay=(0.9,"s"))
def taiko(sr=DEFAULT_SR, vel=1.0, tune=60, decay=.9):
    from .physics import MembraneDrum
    return MembraneDrum(float(tune), .42, damping=.52/max(float(decay),.2), air_coupling=.72).strike(
        max(.55,float(decay)+.25),velocity=vel,position=.12,stroke="bass",sr=sr,seed=15)


@drum("crash", "Physical nonlinear cymbal plate / crash.", decay=(1.5,"s"))
def crash(sr=DEFAULT_SR, vel=1.0, decay=1.5):
    from .physics import CymbalPlate
    return CymbalPlate(205,1.0,damping=1.0,profile="crash").strike(max(.4,float(decay)),velocity=vel,sr=sr,seed=10)


@drum("ride", "Physical nonlinear ride cymbal with bell coupling.", decay=(1.0,"s"))
def ride(sr=DEFAULT_SR, vel=1.0, decay=1.0):
    from .physics import CymbalPlate
    return CymbalPlate(235,1.12,damping=.82,profile="ride").strike(max(.4,float(decay)),velocity=vel,bell=.65,sr=sr,seed=12)


@drum("hihat", "Physical short-decay cymbal pair model.", decay=(0.08,"s"), tone=(8000,"legacy brightness Hz"))
def hihat(sr=DEFAULT_SR, vel=1.0, decay=.08, tone=8000):
    from .physics import CymbalPlate
    fund=230+120*np.clip((float(tone)-6000)/5000,0,1)
    return CymbalPlate(fund,.65,damping=max(.8,.10/max(float(decay),.025)),profile="hihat").strike(
        max(.055,float(decay)+.02),velocity=vel,sr=sr,seed=6)


@drum("openhat", "Physical open hi-hat / cymbal pair.", decay=(0.4,"s"))
def openhat(sr=DEFAULT_SR, vel=1.0, decay=.4):
    from .physics import CymbalPlate
    return CymbalPlate(245,.68,damping=.85,profile="hihat").strike(max(.18,float(decay)),velocity=vel,sr=sr,seed=7)

# v0.8 physical kick/snare migration -----------------------------------------
def farnell_kick(sr=DEFAULT_SR, vel=1.0):
    """Phase-synchronised swept kick: y = line cos(2 pi (2.413 line)^2), line 1 -> 0 in 70 ms."""
    line = 1.0 - np.arange(samples(0.07, sr)) / (0.07 * sr)
    return line * np.cos(2.0 * np.pi * (2.413 * line) ** 2) * (0.6 + 0.4 * vel)


@drum("kick", "Physical kick: membrane, enclosed-air mode and beater compression.", tune=(55,"Hz"), decay=(0.4,"s"), punch=(1.0,"beater hardness"),
      model=("physical", "physical | farnell (70 ms phase-synchronised swept cosine, techno click; ignores tune/decay/punch)"))
def kick(sr=DEFAULT_SR, vel=1.0, tune=55, decay=.4, punch=1.0, model="physical"):
    if str(model) == "farnell":
        return farnell_kick(sr, vel)
    from .physics import KickDrum
    return KickDrum(float(tune), .28, float(decay), float(np.clip(punch,0,1)), .82).strike(
        max(.18,float(decay)+.12), velocity=vel, sr=sr, seed=1)


@drum("snare", "Physical snare: membrane/cavity plus acceleration-driven snare wires.", tune=(200,"Hz"),
      decay=(1.0,"s, T60 of the strongest partials"), snappy=(0.7,"wire amount"),
      under=(0.0, "0..1 level of a second mic under the drum (head seen from below: inverted, later, darker)"),
      flip=(0, "1 = bottom mic polarity flipped back (full lows); 0 = raw, lows partly cancel"))
def snare(sr=DEFAULT_SR, vel=1.0, tune=200, decay=1.0, snappy=.7, under=0.0, flip=0):
    from .physics import SnareDrum
    from .percussion import double_mic
    y = SnareDrum(float(tune), .17, float(decay), .72, 13).strike(
        max(.14,float(decay)), velocity=vel, snappy=float(snappy), sr=sr, seed=2)
    return double_mic(y, sr, under, bool(flip))


@drum("hat808", "TR-808 hi-hat: six detuned square oscillators through two high-passes, nothing else.",
      open=(0, "1 = open hat"), decay=(0.0, "s to -60 dB; 0 = 0.08 closed, 0.5 open"),
      detune=(0.0, "semitones on the 540 and 800 Hz oscillators, the two set by trimmers on the real board"))
def hat808(sr=DEFAULT_SR, vel=1.0, open=0, decay=0.0, detune=0.0):
    """Six 47.98 % pulses, the 7 kHz one-pole pair, decay exp(-6.9 t / Decay), gain 2.5."""
    from scipy.signal import lfilter
    d = float(decay) if decay > 0 else (0.5 if open else 0.08)
    n = samples(d + 0.01, sr)
    k = 2.0 ** (float(detune) / 12.0)
    x = sum(O.pulse(f, n, sr, 0.4798) for f in (205.3, 304.4, 369.6, 522.7, 540.0 * k, 800.0 * k)) / 6.0
    a = np.exp(-2.0 * np.pi * min(7000.0, 0.45 * sr) / sr)
    b = [(1.0 + a) / 2.0, -(1.0 + a) / 2.0]
    y = lfilter(b, [1.0, -a], lfilter(b, [1.0, -a], x))
    return 2.5 * y * np.exp(-6.9 * np.arange(n) / (d * sr)) * (0.5 + 0.5 * vel)


def _block_voice(sr, vel, voice, knob, dur, noise):
    """TR-808 tom / conga by blocks: a 1 ms trigger (4-14 V with the accent) rings the voice's bridged-T at the knob's R1;
    the toms add a pink-noise burst under the same envelope."""
    fc, _, tau = bridged_t(voice, tom_r1(knob))
    n = samples(dur, sr)
    lo, hi = TR808_ACCENT_V
    x = np.zeros(n)
    x[:max(1, samples(TR808_TRIGGER_MS * 1e-3, sr))] = lo + (hi - lo) * float(np.clip(vel, 0.0, 1.0))
    y = F.phasor_reson(x, fc, tau, sr)
    y = y / (np.max(np.abs(y)) + 1e-12)
    if noise > 0.0:
        y = y + noise * F.lowpass(O.pink(n, seed=808), 2.0 * fc, sr) * _dec(n, tau, sr)
    return y / (np.max(np.abs(y)) + 1e-12) * 0.9 * (0.5 + 0.5 * vel)


@drum("tom808", "TR-808 tom by blocks: trigger into the low/mid/high tom bridged-T, plus a little pink noise.",
      size=("mid", "low | mid | high"), knob=(0.5, "tuning knob 0..1 (R1 500 -> 333 ohm: higher)"),
      noise=(0.3, "pink-noise level"), dur=(0.8, "s"))
def tom808(sr=DEFAULT_SR, vel=1.0, size="mid", knob=0.5, noise=0.3, dur=0.8):
    return _block_voice(sr, vel, (size if size in ("low", "mid", "high") else "mid") + "_tom", knob, dur, float(noise))


@drum("conga808", "TR-808 conga by blocks: trigger into the low/mid/high conga bridged-T (no noise).",
      size=("mid", "low | mid | high"), knob=(0.5, "tuning knob 0..1"), dur=(0.5, "s"))
def conga808(sr=DEFAULT_SR, vel=1.0, size="mid", knob=0.5, dur=0.5):
    return _block_voice(sr, vel, (size if size in ("low", "mid", "high") else "mid") + "_conga", knob, dur, 0.0)
