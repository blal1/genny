# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Modal synthesis: struck objects, bells, wood, fracture, micro-collisions, swung blades.

Ports (research notes in ``out/research``):

* **Resonator bank** - van den Doel & Pai, "Modal synthesis for vibrating objects" (VdD-P)
  eq. 6 and ``ModalSonicObject.computeResonCoeff``: per mode
  ``v[m] = 2R cos(th) v[m-1] - R^2 v[m-2] + a R sin(th) F[m]``, ``R = exp(-d/SR)``,
  ``th = 2 pi f / SR`` (the one-sample force delay dropped as in the Java code). Research 06 §1.1.
  Faust tables use ``pm.modeFilter`` exactly (b = [1, 0, -1] * gain, r = 0.001^(1/(t60 SR)),
  bank divided by nModes). STK ModalBar uses ``BiQuad::setEqualGainZeroes`` (b = [1, 0, -1]).
* **Contact force** - VdD-P eq. 7 / ``BangForce``: ``0.5 (1 - cos(2 pi (i+1)/(N+1)))``;
  "hardness translates directly into the duration of the force, energy transfer into its
  magnitude". 0.01-10 ms bell, 0.1-50 ms bottle, ~10 ms typical, 50 ms thud. The pulse is fed
  **through** the modes; there is no separate click layer. Area = momentum (JOS PASP
  Ideal_String_Struck_Mass, research 10 §5.1), so a longer contact is darker, not quieter.
* **Loudness -> brightness** - JOS Force_Pulse_Synthesis (research 10 §5.3): louder contacts
  are taller and narrower pulses; Farnell rule 30 (research 01 §0).
* **Micro-collisions** - VdD-P p.5 / FoleyAutomatic: hard impacts contain "sequences of very
  fast contact separations and collisions", modelled as "a short burst of impulse trains at the
  dominant modal frequencies".
* **Fracture** - Zheng & James, "Rigid-body fracture sound with precomputed soundbanks"
  (research 06 §2): parent rings, its modes stop at the break (spectral discontinuity), debris
  pieces get fracture impulses (the "crack"), piece frequencies scale as omega / gamma with
  Rayleigh damping alpha + beta gamma^-2 omega^2, bounces with shrinking intervals re-excite
  the pieces (Farnell monotonicity rule, research 01 §2.6), ground slab 700 Hz - 4 kHz.
* **Aeolian sword** - Selfridge et al., SMC 2017 and the author's Pd patch
  ``physical-model-of-a-sword-sound`` (research 03 §4, 06 §4): Fey Strouhal St = St* + m/sqrt(Re),
  f = St u / d, bandwidth-derived Q (40-70), Goldstein dipole gain with u^6, harmonics 2f (drag)
  3f, 4f (drag), 5f with the patch's dB scaling, two cascaded ``vcf~`` per partial, 8 sources
  stepping 7 d inward from the tip, arm 0.35 m, trapezoid speed peaking at azimuth 180 deg,
  Doppler, equal-power pan, 4 presets verbatim (``modal_data.SWORD_PRESETS``).
* **Additive bells** - Risset (Pd D07 ``partial.pd``: 5 ms attack, quartic decay, amp x 0.1),
  Farnell telephone bell (``partialgroup.pd``, ``bellenv.pd`` squared line, striker = 10 ms
  quartic noise x 0.1), chinatown ``pd chinabell`` (ead~ 5 ms raised to the 4th power).
* **STK ModalBar** - ``Modal.cpp`` / ``ModalBar.cpp`` (research 07 §3): marmstk1 strike wave at
  rate 0.25*4^h, OnePole pole 1-amp, masterGain 0.1+1.8h, position-derived gains of modes 0-2,
  direct gain mix, 6 Hz vibrato for the vibraphone.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import modal_data as md
from .core import (SR, DEV_DIR, contact_pulse, decay_rate, one_pole_coef, one_pole_lp, pow_decay, rate_radius, seconds, soft_ends,
                   spectrum_peaks, tv_bandpass, write_wav)

LN1000 = float(np.log(1000.0))


# ==========================================================================================
# Resonator bank
# ==========================================================================================

@numba.njit(cache=True)
def _bank(x, b0, b2, a1, a2, stop, tails, kill, fade):
    """Sum of biquads y_k = b0 x + b2 x[n-2] - a1 y[n-1] - a2 y[n-2].

    ``stop[k]``: sample after which mode k is no longer computed (it has decayed away); it closes over the ``tails[k]`` samples before.
    ``kill``: sample at which every resonator state is destroyed (-1: never); the output of
    the killed modes ramps to zero over ``fade`` samples (fracture spectral discontinuity).
    """
    n = x.shape[0]
    y = np.zeros(n)
    for k in range(b0.shape[0]):
        x1 = 0.0
        x2 = 0.0
        y1 = 0.0
        y2 = 0.0
        end = min(n, stop[k])
        if kill >= 0:
            end = min(end, kill + fade)
        tail = stop[k] - tails[k]                       # the last stretch before the floor closes: a mode stopping dead is a step
        for i in range(end):
            xi = x[i]
            yi = b0[k] * xi + b2[k] * x2 - a1[k] * y1 - a2[k] * y2
            x2 = x1
            x1 = xi
            y2 = y1
            y1 = yi
            if kill >= 0 and i >= kill:
                g = 1.0 - (i - kill) / fade
                y[i] += yi * g * g
            elif i >= tail:
                g = math.cos(0.5 * math.pi * (i - tail + 1) / tails[k])
                y[i] += yi * g * g
            else:
                y[i] += yi
    return y


class Modes:
    """Resolved modes: frequencies (Hz), decay rates d (1/s), gains, and the filter form."""

    def __init__(self, freqs, decays, gains, form: str, norm: float = 1.0):
        self.freqs = np.asarray(freqs, float)
        self.decays = np.asarray(decays, float)
        self.gains = np.asarray(gains, float)
        self.form = form          # "vdd" | "zeros" (b = [g, 0, -g])
        self.norm = norm          # Faust ":> /(nModes)"

    def coefficients(self):
        f, d, g = self.freqs, self.decays, self.gains
        r = np.exp(-d / SR)
        th = 2 * np.pi * f / SR
        a1 = -2 * r * np.cos(th)
        a2 = r * r
        if self.form == "vdd":
            b0 = g * r * np.sin(th)
            b2 = np.zeros_like(b0)
        else:
            b0 = g / self.norm
            b2 = -b0
        return b0, b2, a1, a2

    def keep(self, mask):
        return Modes(self.freqs[mask], self.decays[mask], self.gains[mask], self.form, self.norm)


def _interp_rows(rows, position, grid=None) -> np.ndarray:
    """Gain vector at a fractional strike position in [0, 1] across the measured points. With ``grid`` = (ny, nx),
    the rows are points measured on a surface (row-major) and ``position`` = (u, v) in [0, 1]^2 is bilinear."""
    rows = np.asarray(rows, float)
    if grid is not None and np.ndim(position) == 1:
        ny, nx = int(grid[0]), int(grid[1])
        R = rows.reshape(ny, nx, -1)
        u = float(np.clip(position[0], 0.0, 1.0)) * (nx - 1)
        v = float(np.clip(position[1], 0.0, 1.0)) * (ny - 1)
        i, j = min(int(u), max(nx - 2, 0)), min(int(v), max(ny - 2, 0))
        a, b = u - i, v - j
        i1, j1 = min(i + 1, nx - 1), min(j + 1, ny - 1)
        return ((1 - a) * (1 - b) * R[j, i] + a * (1 - b) * R[j, i1] + (1 - a) * b * R[j1, i] + a * b * R[j1, i1])
    if len(rows) == 1:
        return rows[0]
    p = np.clip(position, 0.0, 1.0) * (len(rows) - 1)
    i = int(np.floor(p))
    j = min(i + 1, len(rows) - 1)
    w = p - i
    return (1 - w) * rows[i] + w * rows[j]


def faust_t60s(freqs, fmax, t60, ratio, slope):
    """Faust bell/marimba T60 law: t60 * (1 - f/fmax * ratio) ** slope (clipped at 0)."""
    return t60 * np.clip(1.0 - np.asarray(freqs) / fmax * ratio, 0.0, None) ** slope


def resolve(table: dict | str, f_scale: float = 1.0, position: float = 0.0, *, d_scale: float = 1.0,
            a_scale: float = 1.0, f0: float | None = None, t60: float | None = None,
            t60_slope: float | None = None, t60_ratio: float | None = None) -> Modes:
    """Turn a ``modal_data`` table into ``Modes`` at SR.

    ``f_scale`` multiplies every frequency (VdD fscale; Z-J size scaling omega/gamma),
    ``d_scale`` every decay rate, ``a_scale`` every gain. ``position`` in [0, 1] selects /
    interpolates the measured strike points. ``f0`` is the base frequency for ratio tables.
    """
    t = md.TABLES[table] if isinstance(table, str) else table
    kind = t["kind"]
    if kind == "vdd":
        f = np.asarray(t["freqs"]) * t["fscale"] * f_scale
        d = np.asarray(t["decays"]) * t["dscale"] * d_scale
        if t is md.SY_S100:                                  # points x = 0.025 .. 0.5 of a string: fold past the middle
            position = 2.0 * min(float(position), 1.0 - float(position))
        g = _interp_rows(t["gains"], position, t.get("grid")) * t["ascale"] * a_scale
        m = Modes(f, d, g, "vdd")
        return m.keep((f > 0) & (f < 0.45 * SR))            # VdD code: skip fn >= 0.45 sr
    if kind == "faust":
        n = len(t["gains"][0])
        g = _interp_rows(t["gains"], position, t.get("grid")) * a_scale
        if "ratios" in t:                                    # marimbaBarModel(freq, ...)
            base = (f0 or 65.4) * f_scale
            r = np.asarray(t["ratios"])
            f = base * r
            t60s = faust_t60s(r, t["fmax_ratio"], t60 or t["t60"], t60_ratio or t["t60_ratio"],
                              t60_slope or t["t60_slope"])
        elif "fmax" in t:                                    # bells
            f = np.asarray(t["freqs"]) * f_scale
            t60s = faust_t60s(f, t["fmax"] * f_scale, t60 or t["t60"], t60_ratio or t["t60_ratio"],
                              t60_slope or t["t60_slope"])
        else:                                                # modeInterpRes bodies (measured T60)
            f = np.asarray(t["freqs"]) * f_scale
            t60s = np.asarray(t["t60"]) * (t60 or 1.0)
        with np.errstate(divide="ignore"):
            d = np.where(t60s > 0, LN1000 / np.maximum(t60s, 1e-9), np.inf) * d_scale
        m = Modes(f, d, g, "zeros", norm=n)
        return m.keep(np.isfinite(d) & (f < SR / 2 - 1) & (g != 0))
    if kind == "stk_modal":
        base = f0 or 440.0
        ratios = np.asarray(t["ratios"], float)
        f = np.where(ratios > 0, ratios * base * f_scale, -ratios)   # negative ratio = fixed Hz
        f = np.array([_stk_fold(x) for x in f])
        radii = np.array([rate_radius(r, 44100) for r in t["radii_44k"]])
        d = -np.log(radii) * SR * d_scale
        g = np.array(t["gains"][0], float)
        p = (t["strike_position"] if position is None else position) * np.pi   # ModalBar::setStrikePosition
        g[0] = 0.12 * np.sin(p)
        g[1] = -0.03 * np.sin(0.05 + 3.9 * p)
        g[2] = 0.11 * np.sin(-0.05 + 11 * p)
        return Modes(f, d, g * a_scale, "zeros")
    raise ValueError(f"table kind {kind!r} is not a modal body")


def _stk_fold(f: float) -> float:
    """Modal.cpp:90-101: a mode above Nyquist is halved until it fits."""
    while f >= SR / 2:
        f *= 0.5
    return f


def render_modes(force: np.ndarray, modes: Modes, *, kill_at: int = -1, fade_ms: float = 1.0,
                 floor_db: float = -100.0) -> np.ndarray:
    """Drive the resonator bank with ``force`` (per-sample force signal)."""
    force = np.ascontiguousarray(force, dtype=np.float64)
    if len(modes.freqs) == 0:
        return np.zeros_like(force)
    b0, b2, a1, a2 = modes.coefficients()
    nz = np.nonzero(force)[0]
    last = int(nz[-1]) if len(nz) else 0
    ring = np.log(10 ** (-floor_db / 20)) / np.maximum(modes.decays, 1e-6)
    stop = (last + ring * SR + 2).astype(np.int64)
    fade = max(int(fade_ms * 1e-3 * SR), 1)
    return _bank(force, b0, b2, a1, a2, stop, np.maximum((0.1 * ring * SR).astype(np.int64), 1), int(kill_at), fade)


def ring_time(modes: Modes, db: float = 60.0) -> float:
    """Time for the gain-weighted mode sum to fall ``db`` below its start."""
    if len(modes.freqs) == 0:
        return 0.0
    g = np.abs(modes.gains) / modes.norm
    t = np.linspace(0, 60, 6001)
    env = (g[:, None] * np.exp(-modes.decays[:, None] * t[None, :])).sum(axis=0)
    below = np.nonzero(env < env[0] * 10 ** (-db / 20))[0]
    return float(t[below[0]]) if len(below) else 60.0


def tension_glide(modes: Modes, n: int, alpha: float) -> np.ndarray:
    """Free decay of string ``modes`` (mode m at wavenumber m pi, m = f / f_min) under Kirchhoff-Carrier tension:
    every frequency scales by sqrt(1 + alpha^2 / 2 ||u_x||^2), ||u_x||^2 averaged over a cycle = sum (m pi A_m)^2 / 4."""
    t = np.arange(int(n)) / SR
    k = np.pi * modes.freqs / modes.freqs.min()
    G = np.ones(t.size)
    for km, g, d in zip(k, modes.gains, modes.decays):
        G += float(alpha) ** 2 / 8.0 * (km * g) ** 2 * np.exp(-2.0 * d * t)
    s = np.sqrt(G) / SR
    y = np.zeros(t.size)
    for f, g, d in zip(modes.freqs, modes.gains, modes.decays):
        y += g * np.exp(-d * t) * np.sin(2.0 * np.pi * f * np.cumsum(s))
    return y


# ==========================================================================================
# Excitation
# ==========================================================================================

MICRO_WINDOW_MS = 15.0


def bang(ms: float, magnitude: float = 1.0) -> np.ndarray:
    """VdD ``BangForce``: 0.5 (1 - cos(2 pi (i+1)/(N+1))), scaled to area ``magnitude``."""
    return contact_pulse(ms) * magnitude


def contact_time(contact_ms: float, velocity: float) -> float:
    """Loudness -> brightness: louder contacts are shorter (JOS force-pulse synthesis).

    # Exponent -1/5: Hertz impact duration Tc = 2.87 (m^2 / (R E*^2 V))^(1/5) (Johnson 1985
    # eq. 11.24, p. 353; VdD-P cites it without the formula); sets how fast the pulse narrows.
    """
    return contact_ms * max(velocity, 1e-3) ** -0.2


def faust_strike(n: int, rng: np.random.Generator, cutoff: float = 7000.0, sharpness: float = 0.25,
                 gain: float = 1.0, hp: float = 10.0) -> np.ndarray:
    """Faust ``pm.strikeModel(HP, LP, sharpness, gain)``: white noise -> HP2 -> LP2 x en.ar(att, att),
    att = 0.002 * sharpness (research 08 §1.3)."""
    from scipy.signal import butter, sosfilt
    na = max(1, int(0.002 * sharpness * SR))
    env = np.concatenate([np.arange(1, na + 1) / na, 1 - np.arange(1, na + 1) / na])
    x = rng.uniform(-1, 1, len(env))
    x = sosfilt(butter(2, hp, "high", fs=SR, output="sos"), x)
    x = sosfilt(butter(2, min(cutoff, 0.45 * SR), "low", fs=SR, output="sos"), x)
    out = np.zeros(n)
    m = min(n, len(env))
    out[:m] = (x * env * gain)[:m]
    return out


def micro_collisions(force: np.ndarray, modes: Modes, count: int, contact_ms: float, velocity: float,
                     rng: np.random.Generator, start: int = 0) -> None:
    """Add FoleyAutomatic micro-collisions in place: a short train of re-contacts, each spaced by
    the period of one of the four lowest modes, the whole train within ``MICRO_WINDOW_MS`` of the
    first contact (VdD-P p.5)."""
    f = np.sort(modes.freqs[modes.freqs > 0])[:4]
    if count <= 0 or len(f) == 0:
        return
    end = min(start + int(MICRO_WINDOW_MS * 1e-3 * SR), len(force))
    t = start + len(bang(contact_ms))
    amp = velocity
    for _ in range(count):
        t += int(round(SR / f[rng.integers(len(f))]))
        amp *= rng.uniform(0.35, 0.7)
        p = bang(contact_ms * 0.5, amp)
        if t + len(p) > end:
            break
        force[t:t + len(p)] += p


# ==========================================================================================
# strike()
# ==========================================================================================

def strike(table: dict | str, f_scale: float = 1.0, contact_ms: float = 1.0, position: float = 0.0,
           rng: np.random.Generator | None = None, *, velocity: float = 1.0, dur: float | None = None,
           d_scale: float = 1.0, a_scale: float = 1.0, f0: float | None = None, micro: int = 0,
           exciter: str = "contact", t60: float | None = None, t60_slope: float | None = None,
           max_dur: float = 12.0, lead: float = 0.0) -> np.ndarray:
    """One strike of a measured / FEM object, returned as a mono float array at SR.

    ``contact_ms`` is the (1 - cos) contact duration at velocity 1 (VdD: 0.01-10 ms bell,
    0.1-50 ms bottle, 1-3 ms wood tick, 30-50 ms thud); ``velocity`` scales the momentum and
    shortens the contact (brighter when louder). ``exciter="faust"`` uses Faust strikeModel
    noise instead (cutoff 7000, sharpness 0.25). ``micro`` adds FoleyAutomatic re-contacts.
    The sound ends by its own decay: ``dur`` defaults to the -60 dB ring time (capped at
    ``max_dur``, where a quartic fade over the last 20 % is applied instead of a cut).
    """
    rng = rng or np.random.default_rng(0)
    modes = resolve(table, f_scale, position, d_scale=d_scale, a_scale=a_scale, f0=f0, t60=t60,
                    t60_slope=t60_slope)
    T = contact_time(contact_ms, velocity)
    natural = ring_time(modes) + T * 1e-3
    length = dur if dur is not None else min(natural, max_dur)
    n = seconds(length + lead) + 1
    force = np.zeros(n)
    i0 = seconds(lead)
    if exciter == "faust":
        force[i0:] += faust_strike(n - i0, rng, gain=velocity)
    else:
        p = bang(T, velocity)
        force[i0:i0 + len(p)] += p[: n - i0]
    micro_collisions(force, modes, micro, T, velocity, rng, start=i0)
    y = render_modes(force, modes)
    if dur is None and natural > max_dur:
        nf = int(0.2 * len(y))
        y[-nf:] *= pow_decay(nf, 4.0)
    elif dur is None:                                       # the buffer stops at -60 dB, a step: its last tenth closes
        y *= soft_ends(len(y), 1, max(len(y) // 10, 1))
    return y


# ==========================================================================================
# STK ModalBar
# ==========================================================================================

def stk_modal_bar(preset: str, f0: float, amp: float = 0.8, *, hardness: float | None = None,
                  position: float | None = None, dur: float | None = None) -> np.ndarray:
    """STK ModalBar note (Modal.cpp tick / strike, ModalBar.cpp setPreset) at SR.

    Strike wave marmstk1 (22.05 kHz file) read at 0.25*4^h file samples per output sample of
    a 44.1 kHz STK run, converted to SR (same duration); OnePole pole = 1 - amp; masterGain
    0.1 + 1.8 h; 4 equal-gain-zero BiQuads; direct gain mix; vibraphone 6 Hz AM x 0.2.
    """
    t = md.TABLES[f"stk_{preset.lower()}"]
    h = t["stick_hardness"] if hardness is None else hardness
    wave = np.asarray(md.STK_MARMSTK1["samples_int16"], float) / 32768.0
    rate = 0.25 * 4.0 ** h * 44100.0 / SR
    pos = np.arange(0, len(wave) - 1, rate)
    exc = np.interp(pos, np.arange(len(wave)), wave) * amp
    # OnePole setPole(1 - amp): b0 = 1 - pole = amp, a1 = -pole
    pole = 1.0 - amp
    y = np.empty_like(exc)
    s = 0.0
    for i, v in enumerate(exc):
        s = amp * v + pole * s
        y[i] = s
    exc = y * (0.1 + 1.8 * h)
    modes = resolve(t, 1.0, t["strike_position"] if position is None else position, f0=f0)
    length = dur if dur is not None else ring_time(modes) + 0.05
    force = np.zeros(seconds(length))
    force[: len(exc)] = exc[: len(force)]
    out = render_modes(force, modes)
    dg = t["direct_gain"]
    out = out * (1 - dg) + dg * force
    if t["vibrato_gain"]:
        out *= 1 + t["vibrato_gain"] * np.sin(2 * np.pi * 6.0 * np.arange(len(out)) / SR)
    return out


# ==========================================================================================
# Additive bells (free-running partials with power-law envelopes)
# ==========================================================================================

def _ead4(n: int, attack_ms: float, decay_s: float) -> np.ndarray:
    """Linear attack/decay line raised to the 4th power (Pd partial.pd / chinabell ead~ x ead~ ^2)."""
    t = np.arange(n) / SR
    a = attack_ms * 1e-3
    line = np.where(t < a, t / a, np.clip(1 - (t - a) / max(decay_s, 1e-6), 0, 1))
    return line ** 4


# Per-partial attack, decay and onset delay (ms): the highs leave first, the fundamental blooms after 400 ms.
ADDITIVE_TABLES = {
    "delayed_bell": {"ratios": (1.0, 2.06, 2.44, 4.06, 5.52, 6.14, 8.1, 11.2, 13.9, 17.0),
                     "attacks_ms": (300, 700, 400, 200, 37, 31, 20, 15, 5, 2),
                     "decay_ms": (9000, 8000, 7000, 6000, 5000, 4000, 3000, 1500, 1000, 500),
                     "delays_ms": (400, 100, 32, 20, 17, 12, 7, 5, 2, 0), "amp_scale": 0.08, "phases": "free"},
    "struck_wire": {"ratios": (1.0, 2.01, 4.02, 6.06, 8.12, 10.25, 12.5, 14.8, 17.0, 19.02),
                    "attacks_ms": (300, 700, 400, 200, 37, 31, 20, 15, 5, 2),
                    "decay_ms": (9000, 8000, 7000, 6000, 5000, 3200, 1600, 800, 600, 300),
                    "delays_ms": (400, 100, 32, 20, 17, 12, 7, 5, 2, 0), "amp_scale": 0.01, "phases": "aligned"},
}


def additive_bell(table: dict | str, f0: float, dur: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Risset (``risset_bell``) or chinatown (``chinabell``) additive bell.

    Risset: partial k at ratio*f0 + detune, amplitude amp*0.1, quartic envelope with 5 ms attack
    and decay over dur*rel_duration. Chinabell: ratio*f0, amp, ead~(5 ms, decay_ms)^4 (``dur``
    unused except as the render length). Phases free-run (random), as Farnell allows for bells.
    A table with ``delays_ms`` (``ADDITIVE_TABLES``: ``delayed_bell``, ``struck_wire``) gives each partial its own
    attack, decay and onset delay; ``phases`` aligned starts every partial at phase 0.
    """
    rng = rng or np.random.default_rng(0)
    t = (ADDITIVE_TABLES.get(table) or md.TABLES[table]) if isinstance(table, str) else table
    n = seconds(dur)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    if "delays_ms" in t:
        aligned = t.get("phases") == "aligned"
        for r, a_ms, d_ms, w_ms in zip(t["ratios"], t["attacks_ms"], t["decay_ms"], t["delays_ms"]):
            k = min(int(w_ms * 1e-3 * SR), n)
            env = np.zeros(n)
            env[k:] = _ead4(n - k, a_ms, d_ms * 1e-3)
            ph = 0.0 if aligned else rng.uniform(0, 2 * np.pi)
            out += env * t["amp_scale"] * np.sin(2 * np.pi * r * f0 * tt + ph)
        return out
    if "rel_durations" in t:
        for a, rd, r, dt in zip(t["amps"], t["rel_durations"], t["ratios"], t["detune_hz"]):
            env = _ead4(n, t["attack_ms"], dur * rd) * a * t["amp_scale"]
            out += env * np.sin(2 * np.pi * (r * f0 + dt) * tt + rng.uniform(0, 2 * np.pi))
    else:
        for r, dms, a in zip(t["ratios"], t["decay_ms"], t["amps"]):
            env = _ead4(n, t["attack_ms"], dms * 1e-3) * a
            out += env * np.sin(2 * np.pi * r * f0 * tt + rng.uniform(0, 2 * np.pi))
    return out


def farnell_bell(f0: float, decay: float, rng: np.random.Generator, strength: float = 1.0,
                 dur: float | None = None, striker: bool = True) -> np.ndarray:
    """Farnell telephone bell (ratios.pd groups, partialgroup.pd x 0.333, bellenv squared line of
    ``decay_scale * decay`` seconds; striker = 10 ms noise, quartic, x 0.1). DS p.375-378."""
    groups = md.FARNELL_PHONE_BELL["groups"]
    length = dur if dur is not None else 1.2 * decay + 0.05
    n = seconds(length)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for g in groups:
        T = g["decay_scale"] * decay
        env = np.clip(1 - tt / T, 0, 1) ** 2
        s = sum(a * strength * np.cos(2 * np.pi * f0 * r * tt + rng.uniform(0, 2 * np.pi))
                for r, a in zip(g["ratios"], g["amps"]))
        out += env * s * md.FARNELL_PHONE_BELL["group_gain"]
    if striker:
        ns = seconds(0.010)
        out[:ns] += rng.uniform(-1, 1, ns) * pow_decay(ns, 4.0) * 0.1
    return out


# ==========================================================================================
# Fracture (Zheng & James)
# ==========================================================================================

def _rayleigh_fit(modes: Modes) -> tuple[float, float]:
    """Least-squares non-negative (alpha, beta) with d = (alpha + beta w^2) / 2 (Z-J eq. 14-15)."""
    w = 2 * np.pi * modes.freqs
    A = np.stack([np.full_like(w, 0.5), 0.5 * w * w], axis=1)
    from scipy.optimize import nnls
    coef, _ = nnls(A, modes.decays)
    return float(coef[0]), float(coef[1])


def _scaled_piece(parent: Modes, gamma: float, alpha: float, beta: float) -> Modes:
    """Uniform scale by gamma (< 1 smaller): f -> f / gamma, d from the Rayleigh law at the new
    frequency (alpha + beta gamma^-2 w^2) / 2 (Z-J §5.1 eq. 9)."""
    f = parent.freqs / gamma
    w = 2 * np.pi * parent.freqs
    d = 0.5 * (alpha + beta * w * w / (gamma * gamma))
    m = Modes(f, d, parent.gains.copy(), parent.form, parent.norm)
    return m.keep(f < 0.45 * SR)


def _ground_slab(rng: np.random.Generator, n_modes: int = 1000) -> Modes:
    """Concrete slab 9 x 9 x 0.9 m, 1000 modes in 700 Hz - 4 kHz (Z-J §2.3).

    # Z-J give only the band and mode count; frequencies are drawn uniformly, gains
    # Gaussian / sqrt(N), and the decay rate 60 /s (a dull concrete thud) is chosen here.
    """
    f = rng.uniform(700.0, 4000.0, n_modes)
    g = rng.normal(0, 1, n_modes) / np.sqrt(n_modes)
    d = np.full(n_modes, 60.0)
    return Modes(f, d, g, "vdd")


FRACTURE_ETA = 0.8          # at most this share of the strain energy goes into new crack surface


def fracture_budget(mass: float, velocity: float, size: float, toughness: float,
                    strain_share: float = 0.5) -> tuple[int, float]:
    """Energy budget of a brittle break: E_S = strain_share m v^2 / 2, crack area A_F = eta E_S / G_c; a body of
    side `size` cut into n equal cells has 3 size^2 (n^(1/3) - 1) of new surface, so n = (1 + A_F / 3 size^2)^3.
    Returns (pieces, kinetic energy released to the debris E_S - G_c A_F)."""
    es = float(strain_share) * 0.5 * mass * velocity * velocity
    af = FRACTURE_ETA * es / max(float(toughness), 1e-9)
    n = int(np.clip(round((1.0 + af / (3.0 * size * size)) ** 3), 2, 64))
    return n, es - toughness * af


def ellipsoid_axes(inertia, mass: float) -> np.ndarray:
    """Semi-axes of the solid ellipsoid with the same principal moments: a^2 = 5 (I2 + I3 - I1) / (2 m), ..."""
    i1, i2, i3 = (float(v) for v in inertia)
    s = 5.0 / (2.0 * mass)
    return np.sqrt(np.maximum(s * np.array([i2 + i3 - i1, i1 + i3 - i2, i1 + i2 - i3]), 0.0))


def ellipsoid_gamma(piece_axes, parent_axes) -> float:
    """Size ratio of a debris proxy to its parent: for the same shape family the modes go as 1/size, so gamma =
    omega_parent / omega_piece is the ratio of the geometric-mean semi-axes."""
    return float(np.prod(piece_axes) / np.prod(parent_axes)) ** (1.0 / 3.0)


def fracture(table: dict | str, rng: np.random.Generator, *, f_scale: float = 1.0, toughness: float = 80.0,
             t_break: float = 0.004, n_pieces: int | None = None, contact_ms: float = 0.2,
             velocity: float = 1.0, fall_height: float = 0.8, restitution: float = 0.45,
             ground: bool = True, ground_gain: float = 0.3, position: float = 0.0,
             crack_gain: float = 1.0, dur: float | None = None, f0: float | None = None,
             mass: float | None = None, size: float = 0.1) -> np.ndarray:
    """Brittle fracture as a time-varying modal model (Zheng & James, research 06 §2.4).

    (a)-(b) impact ``bang(contact_ms)`` makes the parent ring at its base frequency;
    (c) at ``t_break`` the parent's resonators are destroyed and every debris piece gets a
    fracture impulse (0.05 ms: the crack runs on a 20 kHz time scale) proportional to its
    share of the released strain energy -> the "crack"; pieces share the parent's mode ratios
    with f / gamma, gamma = (volume fraction)^(1/3), Rayleigh-rescaled damping;
    (d)-(f) each piece falls ``fall_height`` and bounces with shrinking intervals
    (restitution), re-exciting its own modes and the ground slab.
    Lower ``toughness`` (G_c, J/m^2; glass 80, ceramic 200 in Z-J Table 1) -> more, smaller,
    higher pieces.
    ``mass`` (kg) and ``size`` (m) switch on the energy budget: the piece count comes from
    `fracture_budget`, every piece leaves with v = sqrt(2 dE / m) (impulse ~ its volume) and its modes are
    scaled by gamma^-3/2 (mass-normalised shapes of a smaller body).
    """
    parent = resolve(table, f_scale, position, f0=f0)
    alpha, beta = _rayleigh_fit(parent)
    budget = mass is not None and mass > 0
    if budget:
        n_auto, d_ek = fracture_budget(float(mass), float(velocity), float(size), float(toughness))
        v_debris = math.sqrt(2.0 * max(d_ek, 0.0) / float(mass))
        if n_pieces is None:
            n_pieces = n_auto
    if n_pieces is None:
        # Z-J give only the trend (lower G_c -> more debris); count = 6 at glass G_c.
        n_pieces = int(np.clip(round(6 * 80.0 / max(toughness, 1.0)), 2, 24))
    # volume fractions from a flat Dirichlet (Voronoi pieces seeded by strain energy
    # are not simulated here); the largest piece keeps the most energy share.
    vol = np.sort(rng.dirichlet(np.ones(n_pieces)))[::-1]
    g_grav = 9.81
    v_imp = np.sqrt(2 * g_grav * fall_height)
    # bounce schedules
    events = []  # (piece, time_s, velocity)
    t_end = t_break
    for i in range(n_pieces):
        t = t_break + np.sqrt(2 * fall_height / g_grav) * rng.uniform(0.9, 1.1)
        v = v_imp * rng.uniform(0.8, 1.0)
        while v > 0.05:
            events.append((i, t, v))
            dt = 2 * v / g_grav                     # ballistic interval, shrinks with v (monotonic)
            if dt < 0.002:
                break
            t += dt
            v *= restitution
        t_end = max(t_end, t)
    length = dur if dur is not None else t_end + 0.6
    n = seconds(length)
    # parent
    fp = np.zeros(n)
    pb = bang(contact_time(contact_ms, velocity), velocity)
    fp[: len(pb)] += pb
    kb = seconds(t_break)
    out = render_modes(fp, parent, kill_at=kb, fade_ms=0.5)
    # pieces
    crack = bang(0.05)
    fground = np.zeros(n)
    for i in range(n_pieces):
        gamma = vol[i] ** (1.0 / 3.0)
        piece = _scaled_piece(parent, gamma, alpha, beta)
        fpi = np.zeros(n)
        if budget:
            piece = Modes(piece.freqs, piece.decays, piece.gains * gamma ** -1.5, piece.form, piece.norm)
            c = crack * crack_gain * v_debris * vol[i]
        else:
            # energy share proportional to volume -> impulse amplitude sqrt(vol)
            c = crack * crack_gain * velocity * np.sqrt(vol[i])
        fpi[kb:kb + len(c)] += c[: n - kb]
        for (j, t, v) in events:
            if j != i:
                continue
            k = seconds(t)
            if k >= n:
                continue
            # bounce contact 0.1 ms at 1 m/s, amplitude ~ v * sqrt(volume)
            p = bang(contact_time(0.1, v), v / v_imp * np.sqrt(vol[i]) * 0.5)
            m = min(len(p), n - k)
            fpi[k:k + m] += p[:m]
            fground[k:k + m] += p[:m]
        out += render_modes(fpi, piece)
    if ground:
        out += ground_gain * render_modes(fground, _ground_slab(rng))
    return out


# ==========================================================================================
# Aeolian sword swing (Selfridge et al. 2017 + Pd patch)
# ==========================================================================================

RHO_AIR = 1.225          # freqCalcCyl4HiFi.pd
MU_AIR = 1.81e-5
C_SOUND = 343.56         # GoldfreqGainSword: c^3 = 4.05513e7
ARM = 0.35               # SwordCorInterp arm length (m)
_SWING_BASE = {k: md.SWORD_PRESETS["presets"]["SWD1"][k]
               for k in ("Xpos", "Ypos", "Zpos", "PreAzim", "PostAzim", "PreElev", "PostElev", "Gain")}
SWING_PRESETS = {**md.SWORD_PRESETS["presets"],
                 "BROOM": {**_SWING_BASE, "Len": 1.3, "TopSpeed": 20.0, "HiltThick": 0.027,
                           "TipThick": 0.027, "Source": "broom"},
                 "WOODSWD": {**_SWING_BASE, "Len": 0.607, "TopSpeed": 30.0, "HiltThick": 0.0117,
                             "TipThick": 0.0095, "Source": "wood_sword"}}
P_REF = 2e-5             # Dipoles2017D1 dB reference


def fey_strouhal(re: np.ndarray) -> np.ndarray:
    """St = St* + m / sqrt(Re), coefficients by Re range (newStrou.pd), clamped >= 0."""
    t = md.FEY_STROUHAL
    re = np.asarray(re, float)
    idx = np.searchsorted(np.asarray(t["re_bounds"]), re, side="right")
    idx = np.clip(idx, 0, len(t["st_star"]) - 1)
    st = np.asarray(t["st_star"])[idx] + np.asarray(t["m"])[idx] / np.sqrt(np.maximum(re, 1e-9))
    return np.maximum(st, 0.0)


def strouhal_q(re: np.ndarray) -> np.ndarray:
    """StBandwidth: df/f [%] = 4.624e-5 Re + 0.9797 (Re < 193260) else
    1.227e-10 Re^2 - 8.553e-5 Re + 16.5; Q = 1 / max(df/f/100, 0.005)."""
    re = np.asarray(re, float)
    pct = np.where(re < 193260, 4.624e-5 * re + 0.9797, 1.227e-10 * re * re - 8.553e-5 * re + 16.5)
    return 1.0 / np.maximum(pct / 100.0, 0.005)


def _db_scale(g: np.ndarray, k: float, natural: bool) -> np.ndarray:
    """Dipoles2017D1 harmonic gain: 2e-5 * 10 ** (k * log(|g| / 2e-5)).

    The 3f branch uses ``log~ 10`` (base 10); the 4f and 5f branches use ``log~`` with no
    argument, which in Pd is the natural log - ported as wired. |g| is used so a negative
    directivity flips phase instead of producing log(negative) (Pd would output 0 there).
    """
    ratio = np.maximum(np.abs(g), 1e-30) / P_REF
    lg = np.log(ratio) if natural else np.log10(ratio)
    return uniform_filter1d(P_REF * 10.0 ** (k * lg), seconds(0.005))   # |g|^k has an infinite slope where the directivity crosses 0 (measured: 5f, baton, 0.072 s). 5 ms


def _cascade_bp(x, f, q):
    """Two cascaded vcf~ at the same centre and Q (Dipoles2017D1)."""
    return tv_bandpass(tv_bandpass(x, f, q), f, q)


CORR_LENGTH_D = 3.0
WAKE_GAMMA = 0.2
WAKE_POLE = 0.99


def exp_sigmoid(x, exponent: float = 10.0, max_value: float = 2.0, threshold: float = 1e-7):
    """Exponentiated sigmoid for positive gains: max_value * sigmoid(x) ** ln(exponent) + threshold."""
    return max_value * _sig(np.asarray(x, float)) ** math.log(exponent) + threshold


def goldstein_intensity(u, d: float, st, *, length: float | None = None, theta: float = 90.0, phi: float = 0.0,
                        r: float = 1.0, kappa: float = 1.0):
    """Lift-dipole intensity (W/m^2) of a cylinder of diameter ``d`` at speed ``u``:
    sqrt(2 pi) kappa^2 St^2 l b rho u^6 sin^2 th cos^2 ph / (32 c^3 r^2 (1 - M cos th)^4)
    x exp(-0.5 (2 pi M St l / d)^2 sin^2 th sin^2 ph), b = d, l = correlation length (``CORR_LENGTH_D`` d)."""
    u = np.asarray(u, float)
    l = CORR_LENGTH_D * d if length is None else float(length)
    th, ph = math.radians(theta), math.radians(phi)
    M = u / C_SOUND
    I = (math.sqrt(2 * math.pi) * kappa ** 2 * st ** 2 * l * d * RHO_AIR * u ** 6 * math.sin(th) ** 2 * math.cos(ph) ** 2
         / (32 * C_SOUND ** 3 * r ** 2 * (1 - M * math.cos(th)) ** 4))
    return I * np.exp(-0.5 * (2 * math.pi * M * st * l / d) ** 2 * math.sin(th) ** 2 * math.sin(ph) ** 2)


def wake_intensity(u, d: float, st, *, length: float | None = None, theta: float = 90.0, phi: float = 0.0,
                   r: float = 1.0, kappa: float = 1.0, B: float = 1.0, gamma: float = WAKE_GAMMA):
    """Lateral-quadrupole wake intensity: Gamma sqrt(2 pi) kappa^2 St^2 l b rho u^8 / (16 pi^2 c^5 (1 - M cos(pi - th))^5 r^2)
    x [1 + B cos^4 th - (B + 3)/4 sin^2(2 th) sin^2 ph], B in [-1, 3]."""
    u = np.asarray(u, float)
    l = CORR_LENGTH_D * d if length is None else float(length)
    th, ph = math.radians(theta), math.radians(phi)
    B = float(np.clip(B, -1.0, 3.0))
    M = u / C_SOUND
    dirn = 1 + B * math.cos(th) ** 4 - (B + 3) / 4 * math.sin(2 * th) ** 2 * math.sin(ph) ** 2
    return (gamma * math.sqrt(2 * math.pi) * kappa ** 2 * st ** 2 * l * d * RHO_AIR * u ** 8
            / (16 * math.pi ** 2 * C_SOUND ** 5 * (1 - M * math.cos(math.pi - th)) ** 5 * r ** 2) * max(dirn, 0.0))


def wake_noise(u, d: float, dur: float, rng: np.random.Generator, **kw) -> np.ndarray:
    """Wake noise in Pa: brown noise 1 / (1 - 0.99 z^-1) through two one-pole high-passes at the shedding
    frequency (little energy below it, falling above), scaled to sqrt(I_w rho c)."""
    from scipy.signal import lfilter
    n = seconds(dur)
    u = np.broadcast_to(np.asarray(u, float), (n,)).copy()
    st = fey_strouhal(RHO_AIR * u * d / MU_AIR)
    fl = float(np.clip(np.median(st * u / d), 1.0, 0.45 * SR))
    x = lfilter([1.0], [1.0, -WAKE_POLE], rng.standard_normal(n))
    a = math.exp(-2 * math.pi * fl / SR)
    for _ in range(2):
        x = lfilter([(1 + a) / 2, -(1 + a) / 2], [1.0, -a], x)
    x /= np.sqrt(np.mean(x * x)) + 1e-30
    return x * np.sqrt(wake_intensity(u, d, st, **kw) * RHO_AIR * C_SOUND)


def aeolian_tone(u: np.ndarray | float, d: float, dur: float, rng: np.random.Generator,
                 model: str = "patch", *, physical: bool = False, wake: bool = False, theta: float = 90.0,
                 phi: float = 0.0, r: float = 1.0, length: float | None = None) -> np.ndarray:
    """One compact cylinder source at speed ``u`` (m/s, scalar or per-sample), diameter ``d`` (m).

    Lift fundamental only (unit gain): noise through two cascaded band-passes at St(Re) u / d
    with the bandwidth-derived Q. ``model="loq"`` fixes St = 0.2 and Q = 10 (Selfridge LoQ).
    ``physical``: the tone is scaled to the dipole pressure sqrt(I rho c) in Pa (``goldstein_intensity``
    at elevation ``theta``, azimuth ``phi``, distance ``r``); ``wake`` adds ``wake_noise`` on the same scale.
    """
    n = seconds(dur)
    u = np.broadcast_to(np.asarray(u, float), (n,)).copy()
    re = RHO_AIR * u * d / MU_AIR
    st = np.full(n, 0.2) if model == "loq" else fey_strouhal(re)
    q = np.full(n, 10.0) if model == "loq" else strouhal_q(re)
    f = np.clip(st * u / d, 1.0, 0.45 * SR)
    y = _cascade_bp(rng.standard_normal(n), f, q)
    if not (physical or wake):
        return y
    kw = dict(length=length, theta=theta, phi=phi, r=r)
    y = y / (np.sqrt(np.mean(y * y)) + 1e-30) * np.sqrt(goldstein_intensity(u, d, st, **kw) * RHO_AIR * C_SOUND)
    if wake:
        y = y + wake_noise(u, d, dur, rng, **kw)
    return y


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _bp(x, f, q):
    """Band-passed noise, not normalised: its power grows with the bandwidth f / Q, as in the Pd patches."""
    return tv_bandpass(x, np.ascontiguousarray(f, dtype=float), np.ascontiguousarray(q, dtype=float))


def cavity_frequencies(u, length: float, depth: float, width: float):
    """(Rossiter modes 1..4, depth-mode monopole) in Hz: DAFx-17 eqs 4 and 9."""
    M = np.asarray(u, float) / C_SOUND
    fd = [(lam - 0.25) / (1 / 0.57 + M / np.sqrt(1 + 0.2 * M * M)) * u / length for lam in range(1, 5)]
    return fd, C_SOUND / (4.0 * (depth + math.sqrt(math.pi * width * length / 4.0)))


def cavity_tone(u, length: float, depth: float, width: float, dur: float, rng: np.random.Generator, *,
                elevation: float = 90.0, distance: float = 1.0) -> np.ndarray:
    """Air flowing over an open cavity (Selfridge, Reiss & Avital, DAFx-17): four Rossiter dipoles
    (Heller's compressible form, alpha 0.25, K 0.57, eq. 4) and the depth-mode monopole c / 4(d + zeta)
    (eq. 9), each band-passed noise at Q = 87.715 - 5.296 ln Re_L (eq. 22, 2..90) scaled by Howe's
    far-field gains (eqs 20-21) with the shear-layer thickness of eqs 10-15. The dipoles fade out when
    the cavity closes (L/d 8..11, sigmoid). ``elevation`` is phi in degrees (90 = overhead)."""
    n = seconds(dur)
    u = np.maximum(np.broadcast_to(np.asarray(u, float), (n,)), 0.05)
    L, d, b = max(float(length), 1e-3), max(float(depth), 1e-3), max(float(width), 1e-3)
    M = u / C_SOUND
    re = RHO_AIR * u * L / MU_AIR
    q = np.clip(87.715 - 5.296 * np.log(re), 2.0, 90.0)
    turb = _sig((re - 25000.0) / 2500.0)                        # laminar -> turbulent at Re_L 25000
    theta0 = L / (9.39 * L / d + 36.732)                        # eq. 10
    d0 = (2.69 + (1.29 - 2.69) * turb) * theta0                 # eq. 11, H laminar 2.69, turbulent 1.29
    x = 0.75 * L
    dc = (1 - turb) * np.sqrt(x * L / re) + turb * x / (6.0 * math.sqrt(8.0))   # eqs 12-13, sigma 6
    dstar = dc / 8.0 + d0                                       # eqs 14-15, n = 7
    A = b * L
    zeta = math.sqrt(math.pi * A / 4.0)
    cphi = math.cos(math.radians(float(elevation)))

    def common(f):
        w = 2 * np.pi * f
        s = w * dstar / u
        return M ** 2 * s ** (5 / 3) / (max(float(distance), 0.1) * (1 + M * cphi) ** 2 * (s ** 2 + 0.12 ** 2) ** 1.5)

    fds, fm = cavity_frequencies(u, L, d, b)
    fm = np.full(n, fm)
    k = 2 * np.pi * fm / C_SOUND
    e = k * k * A / (2 * np.pi)
    X, Y = np.cos(k * (d + zeta)) * np.cosh(e), np.sin(k * (d + zeta)) * np.sinh(e)
    S = 1.02 * np.sin(k * d) / (X * X + Y * Y)                 # C2 = 1.02 (Howe)
    gm = common(fm) * np.abs((X * S) ** 2 + (Y * S) ** 2 + 2 * Y * S * (cphi - M))
    y = gm * _bp(rng.standard_normal(n), np.minimum(fm, 0.45 * SR), q)
    open_ = _sig((9.5 - L / d) * 3.0)                           # closed above L/d 8..11
    for fd in fds:
        gd = common(fd) * (cphi - M) ** 2
        y += open_ * gd * _bp(rng.standard_normal(n), np.minimum(fd, 0.45 * SR), q)
    return y


# Edge tone: Selfridge, Reiss & Avital, AES 144 (2018) paper 9956 and thesis ch. 4.4. The paper prints neither
# the mode tree nor the full-precision fit; both are ported from the authors' EdgeToneD5.pd
# (research/bibliographie_genny/data/selfridge_edgetone).
# St_k = C1 + C2 Re_d + C3 h/d + C4 Re_d^2 + C5 Re_d h/d + C6 (h/d)^2 (eq. 12, Table 2; BestFreq*D2), modes 1..5
_EDGE_FIT = ((0.119379, 3.21351e-05, -0.0128848, 6.09391e-09, -3.83339e-06, 0.000436755),
             (0.180982, 0.000103822, -0.0110952, -9.40805e-09, -8.54037e-06, 0.000199891),
             (0.187383, 0.000129937, -0.00708081, -1.32888e-08, -7.47187e-06, 8.64753e-05),
             (0.187624, 0.000181775, -0.00509238, -2.13317e-08, -8.20091e-06, 4.72372e-05),
             (1.00213, -0.000129432, -0.0763784, 6.84169e-09, 7.57556e-06, 0.00162923))
# Vaik et al. 2014: St = (C1 - C2 / Re_d)(d / h - C3) (eq. 6, thesis Table 4.7): top-hat modes 1..3, and mode 1
# co-existing with mode 2 or 3 for either profile
_EDGE_VAIK = ((0.4837, 12.31, 0.005461), (1.066, 27.11, 0.004157), (1.884, 19.96, 0.001261))
_EDGE_VAIK_CO = {"parabolic": (0.5029, 6.6451, 0.01417), "top_hat": (0.4167, 0.2292, 0.01426)}
# Weka trees (DecisionTree9_Inc5, ModeOneOption). Node ("Re" | "Red" | "Y", threshold, <=, >) on Re_h, Re_d or the
# jet thickness at the wedge Y (m); ("du" | "dh", rising, falling, steady) on the airspeed / distance direction
_EDGE_MODE = ('Re', 4796.79, ('Y', 0.004336, ('Re', 3044.77, 1, ('Y', 0.003633, 1, ('dh', 1, ('Y', 0.003799, 1, 2), 1))), ('Re', 2639.5, ('Y', 0.007688, 1, 2), ('Y', 0.005309, 2, ('Re', 3837.43, 2, ('du', 2, 3, 3))))), ('Y', 0.004897, ('Re', 19230.8, ('Y', 0.003927, ('Y', 0.00314, 1, ('Re', 6632.6, ('dh', 1, 2, ('du', ('Y', 0.003719, 1, 2), 2, 2)), 2)), ('Re', 7461.67, 2, ('Re', 12182.3, ('du', 2, 3, ('dh', ('Y', 0.004652, 2, ('Y', 0.004714, 3, 2)), ('Y', 0.004143, 2, 3), ('Y', 0.004512, 2, 3))), 3))), ('Re', 27071.8, 3, ('Y', 0.003469, 4, 5))), ('Re', 13691.6, ('Y', 0.007442, ('du', ('Re', 10422.7, ('Y', 0.005309, ('Re', 5712.15, 2, 3), 2), 3), 3, ('dh', ('Re', 6490.38, ('Y', 0.005531, 2, 3), 3), ('Y', 0.005856, 3, ('Re', 6928.69, 3, ('Re', 9310.0, ('Y', 0.00647, 3, 4), 4))), ('Re', 5329.77, ('Re', 5075.97, 3, 2), 3))), ('Re', 9475.14, ('Y', 0.013589, ('Y', 0.008574, 3, ('Re', 6429.56, 3, 4)), 4), 4)), ('Y', 0.006894, ('dh', ('Y', 0.006122, 3, 4), 4, ('du', 3, 4, ('Y', 0.005856, 3, 4))), 5))))  # noqa: E501
_EDGE_CO = {2: ('du', 0, 1, ('Red', 355.318, 1, 0)),
            3: ('Red', 882.541, ('du', ('Red', 636.526, 1, 0), 0, ('Red', 388.616, ('Red', 125.207, 0, 1), 0)), 0)}


def _tree(t, v: dict):
    while isinstance(t, tuple):
        if t[0] in ("du", "dh"):
            t = t[1] if v[t[0]] > 0 else t[2] if v[t[0]] < 0 else t[3]
        else:
            t = t[2] if v[t[0]] <= t[1] else t[3]
    return t


def _direction(x: np.ndarray, tol: float) -> np.ndarray:
    """+1 / -1 / 0: x minus its 1 Hz one-pole lag (Pd lop~ 1) beyond +-tol (lengthDirection, AirspeedDirection)."""
    lag = one_pole_lp(x - x[0], one_pole_coef(1.0)) + x[0]
    return np.where(x - lag > tol, 1, np.where(x - lag < -tol, -1, 0))


def edge_track(u, h, delta: float, profile: str = "parabolic"):
    """Per-sample edge-tone state for jet speed ``u`` (m/s), nozzle-to-wedge distance ``h`` and nozzle exit
    width ``delta`` (m): (mode 1..5, St and f of that mode, f of a co-existing mode 1 or 0, Q, h/d gate 0..1).

    Mode: the tree on Re_h, the jet thickness at the wedge Y = 2 xi + d (laminar xi = (1e4 h)^(2/3) / 1e4,
    turbulent 0.115 h, sigmoid at Re_h 25000, ShearThickJet) and the u / h directions, re-read every 20 ms as
    the patch's metro, which is what gives the hysteresis; a top-hat jet stops at mode 3 (modeLim). Q is the
    mean of the cavity-tone law on Re_h and 137.571 - 12.1602 ln Re_d (eqs 13-14, natural log as log~, 2..90).
    The tone exists for 2 < h/d < 61 (sigmoids, range)."""
    n = np.size(u) if np.ndim(u) else np.size(h)
    u = np.maximum(np.broadcast_to(np.asarray(u, float), (n,)), 0.05).copy()
    h = np.maximum(np.broadcast_to(np.asarray(h, float), (n,)), 1e-5).copy()
    d = max(float(delta), 1e-5)
    re_h, re_d, ratio = RHO_AIR * u * h / MU_AIR, RHO_AIR * u * d / MU_AIR, h / d
    lam = 1.0 / (1.0 + np.exp(np.clip(80.0 * (re_h / 25000.0 - 1.0), -60.0, 60.0)))
    Y = lam * (1e4 * h) ** (2 / 3) / 5000.0 + (1.0 - lam) * 0.23 * h + d
    du, dh = _direction(u, 0.01), _direction(h, 1e-4)
    top = profile == "top_hat"
    blk = max(int(0.02 * SR), 1)
    mode, co = np.empty(n, int), np.zeros(n, bool)
    for i in range(0, n, blk):
        v = {"Re": re_h[i], "Red": re_d[i], "Y": Y[i], "du": du[i], "dh": dh[i]}
        m = _tree(_EDGE_MODE, v)
        m = min(m, 3) if top else m
        mode[i:i + blk] = m
        co[i:i + blk] = m in _EDGE_CO and _tree(_EDGE_CO[m], v) == 1
    if top:
        c = np.asarray(_EDGE_VAIK)[mode - 1]
        st = (c[:, 0] - c[:, 1] / re_d) * (1.0 / ratio - c[:, 2])
    else:
        c = np.asarray(_EDGE_FIT)[mode - 1]
        st = c[:, 0] + c[:, 1] * re_d + c[:, 2] * ratio + c[:, 3] * re_d ** 2 + c[:, 4] * re_d * ratio + c[:, 5] * ratio ** 2
    st = np.maximum(st, 0.0)                                       # the fit goes negative in places: max~ 0
    c1, c2, c3 = _EDGE_VAIK_CO["top_hat" if top else "parabolic"]
    st_co = np.where(co, np.maximum((c1 - c2 / re_d) * (1.0 / ratio - c3), 0.0), 0.0)
    q = np.clip(0.5 * (87.715 - 5.296 * np.log(re_h) + 137.571 - 12.1602 * np.log(re_d)), 2.0, 90.0)
    gate = _sig(10.0 * (ratio - 2.0)) * _sig(10.0 * (61.0 - ratio))
    return mode, st, np.clip(st * u / d, 0.0, 20000.0), st_co, np.clip(st_co * u / d, 0.0, 20000.0), q, gate


def edge_tone(u, h, delta: float, dur: float, rng: np.random.Generator, *, profile: str = "parabolic",
              elevation: float = 0.0, distance: float = 1.0) -> np.ndarray:
    """Planar jet on a wedge in pascals (Selfridge et al. 2018): per mode, noise through two cascaded band-passes
    at f_k (Dipoles: two vcf~) with the edge-tone Q, times Powell's far-field pressure
    |P| = (alpha / 2) rho St u^3 cos(theta) / (r c), alpha = 4 (eq. 7). Each mode's band is gated in and out
    with the 20 ms ramps of the patch's line~; a co-existing mode 1 sounds alongside modes 2 and 3."""
    n = seconds(dur)
    u = np.maximum(np.broadcast_to(np.asarray(u, float), (n,)), 0.05)
    mode, st, f, st_co, f_co, q, gate = edge_track(u, h, delta, profile)
    p0 = 2.0 * RHO_AIR * u ** 3 * abs(math.cos(math.radians(float(elevation)))) / (max(float(distance), 0.05) * C_SOUND) * gate
    ramp = np.ones(max(int(0.02 * SR), 1)) / max(int(0.02 * SR), 1)
    y = np.zeros(n)
    for k in range(1, 6):
        on = np.convolve((mode == k).astype(float), ramp)[:n]
        if on.any():
            y += on * p0 * st * _cascade_bp(rng.standard_normal(n), np.clip(f, 20.0, 0.45 * SR), q)
    if st_co.any():
        on = np.convolve((st_co > 0).astype(float), ramp)[:n]
        y += on * p0 * st_co * _cascade_bp(rng.standard_normal(n), np.clip(f_co, 20.0, 0.45 * SR), q)
    return y


# Selfridge, Moffat & Reiss, Audio Mostly 2017, Table 1 (blade count from the aircraft): top speed m/s, rpm,
# engine hp, blade radius m, blades
AIRCRAFT = {"c130": (164.0, 1020.0, 4590.0, 2.06, 4), "b17": (144.0, 2500.0, 1200.0, 1.63, 3),
            "tiger_moth": (48.0, 2100.0, 130.0, 0.996, 2), "yak52": (79.0, 1860.0, 360.0, 1.209, 2),
            "cessna340": (125.0, 2500.0, 310.0, 0.96, 3), "p51": (193.0, 1280.0, 1490.0, 1.70, 4)}
_ABSORB = ((90, 0.0), (180, 0.2), (355, 0.6), (710, 1.0), (1400, 1.8), (2800, 3.3), (5600, 6.0), (11200, 11.4))


def propeller_tone(rpm, blades: int, radius: float, power_hp: float, dur: float, rng: np.random.Generator, *,
                   chord: float = 0.15, angle: float = 110.0, distance: float = 30.0, vortex_harmonics: int = 1,
                   chord_tip: float | None = None) -> np.ndarray:
    """Propeller in pascals at ``distance`` m, ``angle`` degrees from the flight direction (Selfridge et al.
    2017). Loading noise: the ten-step Hamilton Standard estimate (eqs 3-11, dB re 20 uPa) for harmonics
    1..10 of the blade-passing frequency, each band-passed noise; atmospheric absorption from Table 2.
    Vortex noise: seven compact Aeolian sources per blade at St 0.85 on the chord (eq. 13), Q from the
    cylinder bandwidth law, intensity of eq. 2 at the source's own speed, p = sqrt(I rho c). ``vortex_harmonics`` up to 3
    adds 0.85 N u / chord at 1 / N^2; ``chord_tip``: the chord tapers linearly from ``chord`` at the root."""
    n = seconds(dur)
    rpm = np.maximum(np.broadcast_to(np.asarray(rpm, float), (n,)), 1.0)
    B, R, D = max(int(blades), 1), max(float(radius), 0.05), 2.0 * max(float(radius), 0.05)
    th = float(np.clip(angle, 0.0, 180.0))
    r = max(float(distance), 1.0)
    mt = np.pi * D * rpm / 60.0 / C_SOUND
    L = (15.11 * np.log10(max(float(power_hp), 1.0)) + 83.57 + 20 * np.log10(4.0 / B) + 40 * np.log10(4.72 / D)
         + (25.12 * mt - 33.40) * np.log10(0.305 / D) + (34.37 * mt - 36.88)
         + max(-5.3e-3 * th * th + 1.19 * th - 62.32, -20.0) - 20 * np.log10(3.375 * r - 1.0))
    bpf = B * rpm / 60.0
    y = np.zeros(n)
    for N in range(1, 11):
        f = N * bpf
        if f[0] > 0.45 * SR:
            break
        absorb = np.interp(f, [a for a, _ in _ABSORB], [v for _, v in _ABSORB], right=20.0) * r / 304.8
        hn = 26.0 * np.exp(-(-0.7 * mt + 0.79) * N) - 22.0
        g = P_REF * 10 ** ((L - hn - absorb) / 20.0)
        # Q 75 for the loading harmonics, "set perceptually" (Selfridge 2019 PhD §5.3.4, pp. 144-145)
        y += g * _bp(rng.standard_normal(n), f, np.full(n, 75.0))
    span = R / 7.0
    s2 = math.sin(math.radians(th)) ** 2
    c0 = chord
    for i in range(7):
        chord = c0 if chord_tip is None else c0 + (float(chord_tip) - c0) * (i + 0.5) / 7.0
        u = 2 * np.pi * rpm / 60.0 * (i + 0.5) * span
        M = u / C_SOUND
        re = RHO_AIR * u * chord / MU_AIR
        # eq. 2 as printed drops the source's diameter x length; restored as chord x span (AES 141 Aeolian tone form)
        inten = math.sqrt(2 * math.pi) * 0.85 ** 2 * chord * span * RHO_AIR * u ** 6 * s2 / (
            32 * C_SOUND ** 3 * r * r * (1 - M * math.cos(math.radians(th))) ** 4)
        p = np.sqrt(inten * RHO_AIR * C_SOUND)
        for _ in range(B):
            # ponytail: the cylinder bandwidth law diverges past Re ~2e5 (Q -> 0.1); held at its Re 193260 value (Q 10)
            y += p * _bp(rng.standard_normal(n), np.minimum(0.85 * u / chord, 0.45 * SR), strouhal_q(np.minimum(re, 193260.0)))
            for N in range(2, int(np.clip(vortex_harmonics, 1, 3)) + 1):
                y += p / N ** 2 * _bp(rng.standard_normal(n), np.minimum(0.85 * N * u / chord, 0.45 * SR),
                                      strouhal_q(np.minimum(re, 193260.0)))
    return y


LOCK_HYST = 1.1


def lock_state(V, lo: float = 0.8, hi: float = 1.27, hyst: float = LOCK_HYST) -> np.ndarray:
    """Lock-in with hysteresis: a string locks when the reduced velocity V enters [lo, hi] and lets go only
    once V leaves the wider window [lo / hyst, hi hyst]; returns 0/1 per sample."""
    V = np.asarray(V, float)
    s = np.zeros(len(V))
    on = False
    for i, v in enumerate(V):
        on = (lo / hyst <= v <= hi * hyst) if on else (lo <= v <= hi)
        s[i] = on
    return s


def aeolian_harp(u, notes, dur: float, rng: np.random.Generator, *, diameter: float = 5e-4,
                 mass_damping: float = 0.05, index: float = 1.0, hysteresis: bool = False) -> np.ndarray:
    """Strings in the wind (Selfridge, Moffat, Reiss & Avital, ICSV24 2017). Each string sheds an Aeolian
    tone at St(Re) u / d; when that frequency nears string harmonic eta (eq. 9) and the reduced velocity
    V = u / (2 pi d f_eta) (eq. 3) lies in the lock-in range 0.8..1.27, the shedding locks to f_eta and the
    string sings: an FM tone with carrier f_eta and modulator f_(eta+1) - f_eta (eqs 8, 13-14), scaled by
    the Govardhan-Williamson peak amplitude (eq. 7) for mass-damping alpha = ``mass_damping`` * eta (eq. 12).
    ``notes`` are the strings' fundamentals (Hz).
    ``hysteresis``: lock-in follows ``lock_state`` (a sweeping wind holds the string past the window edges)
    instead of the smooth window. Lambda, the window steepness and the default mass_damping are voicing choices."""
    n = seconds(dur)
    u = np.maximum(np.broadcast_to(np.asarray(u, float), (n,)), 0.05)
    d = max(float(diameter), 1e-5)
    re = RHO_AIR * u * d / MU_AIR
    fs = fey_strouhal(re) * u / d
    a_re = np.where(re > 500.0, np.log10(0.41 * re ** 0.36), 0.63)
    y = np.zeros(n)
    for f1 in np.atleast_1d(np.asarray(notes, float)):
        eta = np.maximum(np.round(fs / f1), 1.0)
        feta = eta * f1
        V = u / (2 * np.pi * d * feta)
        if hysteresis:
            lock = uniform_filter1d(lock_state(V), seconds(0.01), mode="nearest")
        else:
            lock = _sig((V - 0.8) / 0.02) * _sig((1.27 - V) / 0.02)
        al = np.minimum(mass_damping * eta, 1.87)               # past the vertex the fit no longer holds
        coef = np.where(re > 500.0, 1 - 1.12 * al + 0.30 * al * al, 1 - 1.29 * al + 0.59 * al * al)
        # amax and lock jump where eta rounds to the next harmonic: a step in the singing tone (measured d2 -41 dB). 10 ms:
        sung = uniform_filter1d(np.maximum(coef, 0.0) * a_re * lock, seconds(0.01), mode="nearest")
        fv = lock * feta + (1 - lock) * fs
        fdm = feta + f1 - fv
        ph_d = 2 * np.pi * np.cumsum(fdm) / SR
        ph_v = 2 * np.pi * np.cumsum(fv) / SR
        sing = np.cos(ph_v + index * np.sin(ph_d)) + 0.1 * np.cos(ph_d + index * np.sin(ph_d))
        g = _cascade_bp(rng.standard_normal(n), np.clip(fs, 1.0, 0.45 * SR), strouhal_q(re))
        g /= np.sqrt(np.mean(g * g)) + 1e-30
        y += g + 3.0 * sung * sing                               # Lambda = 3 against a unit-RMS tone
    return y


def _swing_kinematics(p: dict, lead: float, tail: float):
    """Trapezoid tip speed peaking at azimuth 180 deg (SwordCorInterp): returns t, azimuth (deg),
    elevation (deg), tip speed (m/s), swing direction sign."""
    top = p["TopSpeed"]
    rt = p["Len"] + ARM
    pre, post = p["PreAzim"], p["PostAzim"]
    s1 = abs(180.0 - pre) * np.pi / 180 * rt
    s2 = abs(post - 180.0) * np.pi / 180 * rt
    t1 = s1 / (top / 2)
    t2 = s2 / (top / 2)
    n_lead, n1, n2, n_tail = seconds(lead), max(seconds(t1), 1), max(seconds(t2), 1), seconds(tail)
    tau1 = np.arange(n1) / n1
    tau2 = np.arange(n2) / n2
    az = np.concatenate([np.full(n_lead, pre), pre + (180 - pre) * tau1 ** 2,
                         180 + (post - 180) * (1 - (1 - tau2) ** 2), np.full(n_tail, post)])
    u = np.concatenate([np.zeros(n_lead), top * tau1, top * (1 - tau2), np.zeros(n_tail)])
    # the patch moves elevation pre -> mid -> post; mid is not a preset field, so the
    # elevation is interpolated linearly with swing progress.
    prog = np.clip((az - pre) / (post - pre), 0, 1) if post != pre else np.zeros_like(az)
    el = p["PreElev"] + (p["PostElev"] - p["PreElev"]) * prog
    sign = 1.0 if post >= pre else -1.0
    return az, el, u, sign


def sword_swing(preset: str | dict = "SWD1", rng: np.random.Generator | None = None, *,
                model: str = "patch", doppler: bool = True, n_sources: int = 8,
                top_speed: float | None = None, lead: float = 0.05, tail: float = 0.15,
                partials: tuple[int, ...] = (1, 2, 3, 4, 5)) -> np.ndarray:
    """Selfridge Aeolian swing, stereo (n, 2), from a ``SWORD_PRESETS`` preset (SWD1, SWD2, PGA, MLB), or BROOM, WOODSWD (``SWING_PRESETS``).

    Per source k (8, stepping 7 d inward from the tip, d tapering linearly hilt -> tip):
    u_k = u_tip * (arm + pos_k) / (arm + Len); Re = rho u d / mu; St (Fey) or 0.2 (LoQ);
    f = St u / d (x Doppler 1/(1 - M cos theta)); Q from StBandwidth (or 10 for LoQ);
    lift gain G_L = sqrt(2 pi) rho St^2 (7 d) Len u^6 D_L / (32 c^3 r^2 (1 - M cos theta)^4),
    drag gain G_D = 0.1 x the same with D_D; partials f (G_L), 2f (G_D), 3f (dB x 0.6),
    4f (drag, dB x 0.125), 5f (dB x 0.1), each through two cascaded band-passes of one white
    noise per source; equal-power pan; output x preset Gain.
    Directivities: D_L = n . (a x v), D_D = n . v with n the unit vector source -> listener, a the
    blade axis and v the motion direction - the vector form of Goldstein's sin(theta) cos(phi)
    dipole pattern that the patch evaluates from elevation and azimuth.
    """
    rng = rng or np.random.default_rng(0)
    p = dict(SWING_PRESETS[preset]) if isinstance(preset, str) else dict(preset)
    if top_speed is not None:
        p["TopSpeed"] = top_speed
    az, el, u_tip, sign = _swing_kinematics(p, lead, tail)
    n = len(az)
    L = p["Len"]
    hilt, tip = p["HiltThick"], p["TipThick"]
    # axis convention - shoulder at the origin, x right, y front, z up; listener at the
    # preset (Xpos, Ypos, Zpos); azimuth 0 = behind, 180 = in front; elevation from vertical.
    listener = np.array([p["Xpos"], p["Ypos"], p["Zpos"]])
    phi = np.radians(az)
    eps = np.radians(el)
    a = np.stack([np.sin(eps) * np.sin(phi), -np.sin(eps) * np.cos(phi), np.cos(eps)], axis=1)
    v_dir = np.stack([np.sin(eps) * np.cos(phi), np.sin(eps) * np.sin(phi), np.zeros(n)], axis=1) * sign
    v_dir /= np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), 1e-9)
    lift_dir = np.cross(a, v_dir)
    measured = md.SWING_SOURCES.get(p.get("Source", ""))
    if measured:
        span = np.array([s[0] for s in measured])
        span = span * L / span[-1]
        dia = np.array([s[1] for s in measured])
        corr = md.SWING_CORRELATION.get(p["Source"], md.SWING_CORRELATION_DEFAULT)

        def thick(pos):
            return float(np.interp(pos, span, dia))
    else:
        corr = 7.0

        def thick(pos):
            return hilt + (tip - hilt) * pos / L
    positions = [L]
    for _ in range(n_sources - 1):
        pos = positions[-1]
        positions.append(pos - corr * thick(pos))
    out = np.zeros((n, 2))
    for pos in positions:
        if pos <= 0:
            break
        d = thick(pos)
        radius = ARM + pos
        u = u_tip * radius / (ARM + L)
        src = a * radius
        rel = listener[None, :] - src
        dist = np.maximum(np.linalg.norm(rel, axis=1), 0.3)   # 0.3 m floor on distance
        nvec = rel / dist[:, None]
        cos_t = np.sum(nvec * v_dir, axis=1)
        mach = u / C_SOUND
        conv = np.maximum(1 - mach * cos_t, 0.2)
        re = RHO_AIR * u * d / MU_AIR
        if model == "loq":
            st = np.full(n, 0.2)
            q = np.full(n, 10.0)
        else:
            st = fey_strouhal(re)
            q = strouhal_q(re)
        f = st * u / d
        if doppler:
            f = f / conv
        dl = np.sum(nvec * lift_dir, axis=1)
        dd = cos_t
        # 7 d as the authors' patch has it: its Norberg corrLen (44.67 Re^-0.245) reaches an inlet that nothing reads
        common = 2.50663 * RHO_AIR * st ** 2 * (corr * d) * L * u ** 6 / (32 * C_SOUND ** 3 * dist ** 2 * conv ** 4)
        gl = common * dl
        gd = 0.1 * common * dd
        noise = rng.standard_normal(n)
        sig = np.zeros(n)
        live = np.sin(0.5 * np.pi * np.clip(u / 2.0, 0.0, 1.0)) ** 2     # the dB-scaled partials go as u^.6: a hard cut at rest is a step. 2 m/s
        for mult, gain in ((1, gl), (2, gd), (3, _db_scale(gl, 0.6, False)), (4, _db_scale(gd, 0.125, True)),
                           (5, _db_scale(gl, 0.1, True))):
            if mult not in partials:
                continue
            fc = np.clip(mult * f, 20.0, 0.45 * SR)
            sig += _cascade_bp(noise, fc, np.maximum(q, 0.5)) * gain * live
        # equal-power pan from the source's lateral offset (fcpan)
        pan = np.clip(-rel[:, 0] / dist, -1, 1)
        ang = (pan + 1) * np.pi / 4
        out[:, 0] += sig * np.cos(ang)
        out[:, 1] += sig * np.sin(ang)
    return out * p["Gain"]


def swing_peak_tone(preset: str | dict = "SWD1", model: str = "patch") -> float:
    """Tip lift-tone frequency at top speed, St(Re) u / d (no Doppler) - for checks."""
    p = SWING_PRESETS[preset] if isinstance(preset, str) else preset
    u, d = p["TopSpeed"], p["TipThick"]
    re = RHO_AIR * u * d / MU_AIR
    st = 0.2 if model == "loq" else float(fey_strouhal(np.array([re]))[0])
    return st * u / d


# ==========================================================================================
# Self-test
# ==========================================================================================

def _peak_near(x, f, span=8.0):
    pk = spectrum_peaks(x, 400, fmin=f - span, fmax=f + span, zero_pad=8)
    return sorted(pk, key=lambda p: -p[1])[:1]


if __name__ == "__main__":
    out_dir = DEV_DIR / "modal"
    rng = np.random.default_rng(7)
    ok = True

    def check(name, cond, msg):
        global ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")

    # 1. French bell (Faust FEM), position 0: the 1808.84 / 1820.07 Hz pair must both appear
    y = strike("frenchBell", 1.0, 0.3, 0.0, rng, dur=6.0)
    write_wav(out_dir / "bell_french.wav", 0.9 * y / np.max(np.abs(y)))
    seg = y[seconds(0.05):]
    pk = spectrum_peaks(seg, 300, fmin=1790, fmax=1835, zero_pad=4)
    near = sorted([p for p in pk if p[1] > max(q[1] for q in pk) - 30], key=lambda p: p[0])
    fs_ = [round(p[0], 1) for p in near]
    has_a = any(abs(f - 1808.84) < 1.5 for f in fs_)
    has_b = any(abs(f - 1820.07) < 1.5 for f in fs_)
    check("bell pair", has_a and has_b, f"peaks in 1790-1835 Hz: {fs_}")
    pk2 = [round(p[0], 2) for p in spectrum_peaks(seg, 60, fmin=430, fmax=450, zero_pad=8)[:4]]
    check("bell hum doublet", any(abs(f - 439.08) < 0.6 for f in pk2) and any(abs(f - 440.31) < 0.6 for f in pk2),
          f"peaks 430-450 Hz: {sorted(pk2)} (table 439.077 / 440.305)")
    check("bell finite", np.all(np.isfinite(y)), f"peak {np.max(np.abs(y)):.3g}")
    # bell4.sy measured doublets at 782.0/784.6/787.1
    y4 = strike("sy_bell4", 1.0, 0.1, 0.0, rng, dur=8.0)
    write_wav(out_dir / "bell4_sy.wav", 0.9 * y4 / np.max(np.abs(y4)))
    pk4 = sorted(round(p[0], 1) for p in spectrum_peaks(y4, 20, fmin=775, fmax=795, zero_pad=8)[:3])
    check("bell4 triplet", sum(any(abs(a - b) < 0.8 for a in pk4) for b in (782.0, 784.6, 787.1)) >= 2,
          f"peaks 775-795 Hz: {pk4} (table 782.0 / 784.6 / 787.1)")

    # 2. Wood tick (stick.sy), 2 ms contact at fscale 1 and 0.4: decay 130-320 /s
    for fsc in (1.0, 0.4):
        w = strike("sy_stick", fsc, 2.0, 0.0, rng)
        write_wav(out_dir / f"wood_tick_f{fsc}.wav", 0.9 * w / np.max(np.abs(w)))
        d = decay_rate(w, span_db=40)
        check(f"wood decay fscale {fsc}", 130 <= d <= 320,
              f"fitted d = {d:.0f} /s (T60 {LN1000 / d * 1000:.0f} ms), length {len(w) / SR * 1000:.0f} ms")
    # brightness with velocity: soft vs hard centroid
    def centroid(x):
        X = np.abs(np.fft.rfft(x))
        fr = np.fft.rfftfreq(len(x), 1 / SR)
        return float((X * fr).sum() / X.sum())
    soft = strike("sy_wok", 1.0, 3.0, 0.0, rng, velocity=0.1, dur=0.5)
    hard = strike("sy_wok", 1.0, 3.0, 0.0, rng, velocity=1.0, dur=0.5)
    check("velocity brightness", centroid(hard) > centroid(soft),
          f"centroid soft {centroid(soft):.0f} Hz < hard {centroid(hard):.0f} Hz")
    thud = strike("sy_computertower", 1.0, 40.0, 0.0, rng, dur=0.6)
    write_wav(out_dir / "thud_tower_40ms.wav", 0.9 * thud / np.max(np.abs(thud)))
    check("thud dark", centroid(thud) < 400, f"40 ms contact on computertower: centroid {centroid(thud):.0f} Hz")
    clang = strike("sy_sword1", 1.0, 0.1, 0.0, rng, micro=4, dur=1.5)
    write_wav(out_dir / "sword_clang_micro.wav", 0.9 * clang / np.max(np.abs(clang)))
    mb = stk_modal_bar("Wood1", 600.0, 0.8)
    write_wav(out_dir / "stk_wood1.wav", 0.9 * mb / np.max(np.abs(mb)))
    check("stk wood1", np.all(np.isfinite(mb)) and len(mb) < SR * 0.3, f"length {len(mb) / SR * 1000:.0f} ms")
    rb = additive_bell("risset_bell", 523.25, 4.0, rng)
    write_wav(out_dir / "risset_bell.wav", 0.9 * rb / np.max(np.abs(rb)))
    fb = farnell_bell(650.0, 2.0, rng) + np.pad(farnell_bell(653.0, 2.0, rng), (0, 0))
    write_wav(out_dir / "farnell_bell.wav", 0.9 * fb / np.max(np.abs(fb)))
    cb = additive_bell("chinabell", 880.0, 6.0, rng)
    write_wav(out_dir / "chinabell.wav", 0.9 * cb / np.max(np.abs(cb)))

    # 3. Fracture: parent ring, then crack (spectral discontinuity) on a glass bottle
    tb = 0.08
    fr = fracture("sy_calona0", rng, t_break=tb, toughness=50.0, contact_ms=0.2)
    write_wav(out_dir / "fracture_bottle.wav", 0.9 * fr / np.max(np.abs(fr)))
    fr_def = fracture("sy_vase", rng, toughness=120.0)
    write_wav(out_dir / "fracture_vase_default.wav", 0.9 * fr_def / np.max(np.abs(fr_def)))
    pre = fr[seconds(0.01):seconds(tb - 0.005)]
    post = fr[seconds(tb + 0.003):seconds(tb + 0.06)]
    parent_f = 2628.14
    band = lambda x, lo, hi: float(np.sum(np.abs(np.fft.rfft(x * np.hanning(len(x)))[
        int(lo * len(x) / SR):int(hi * len(x) / SR)]) ** 2))
    pre_ratio = band(pre, parent_f - 60, parent_f + 60) / band(pre, 20, 20000)
    post_ratio = band(post, parent_f - 60, parent_f + 60) / band(post, 20, 20000)
    lvl = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    jump = lvl(fr[seconds(tb):seconds(tb + 0.01)]) - lvl(fr[seconds(tb - 0.02):seconds(tb - 0.005)])
    check("fracture parent ring", pre_ratio > 0.5, f"parent 2628 Hz band holds {pre_ratio * 100:.0f} % of energy before the break")
    check("fracture discontinuity", post_ratio < 0.1, f"after the break the parent band holds {post_ratio * 100:.1f} %")
    check("fracture crack", jump > 6, f"crack transient {jump:+.1f} dB over the late parent ring")
    cen_pre, cen_post = centroid(pre), centroid(post)
    check("fracture pitch up", cen_post > cen_pre, f"centroid {cen_pre:.0f} Hz -> {cen_post:.0f} Hz")

    # 4. Aeolian tone and sword swing: peak at St u / d ~ 0.2 u / d
    u, dd = 26.0, 0.008
    tone = aeolian_tone(u, dd, 2.0, rng)
    f_pk = spectrum_peaks(tone, 1, fmin=100, zero_pad=1)[0][0]
    expect = 0.2 * u / dd
    check("aeolian tone", abs(f_pk / expect - 1) < 0.06, f"u={u} d={dd}: peak {f_pk:.0f} Hz vs 0.2u/d = {expect:.0f} Hz")
    for pre_name in ("SWD1", "SWD2", "PGA", "MLB"):
        p = md.SWORD_PRESETS["presets"][pre_name]
        az, _, u_tip, _ = _swing_kinematics(p, 0.05, 0.15)
        i_top = int(np.argmax(u_tip))                   # azimuth 180: top speed
        f02 = 0.2 * p["TopSpeed"] / p["TipThick"]
        fst = swing_peak_tone(pre_name)
        sw = sword_swing(pre_name, rng)
        write_wav(out_dir / f"swing_{pre_name}.wav", 0.9 * sw / np.max(np.abs(sw)))
        write_wav(out_dir / f"swing_{pre_name}_loq.wav",
                  0.9 * (s := sword_swing(pre_name, rng, model="loq")) / np.max(np.abs(s)))
        def lift_peak(sig, tol):
            """Strongest spectral peak within tol of 0.2u/d, and its level re the overall maximum."""
            half = seconds(max(0.015, 4.0 / f02))           # at least 8 periods of the tone
            w = sig[i_top - half:i_top + half]
            pk = spectrum_peaks(w, 40, fmin=40, zero_pad=8)
            top_db = pk[0][1]
            near = [q for q in pk if abs(q[0] / f02 - 1) < tol]
            return (near[0][0], near[0][1] - top_db, pk[0][0]) if near else (float("nan"), -99.0, pk[0][0])
        tip_only = sword_swing(pre_name, rng, n_sources=1, doppler=False, partials=(1,)).sum(axis=1)
        f_tip, rel_tip, strongest_tip = lift_peak(tip_only, 0.1)
        mono = sw.sum(axis=1)
        f_all, rel_all, strongest_all = lift_peak(mono, 0.2)
        env = np.convolve(mono ** 2, np.ones(seconds(0.005)), "same")
        t_loud = (int(np.argmax(env)) - i_top) / SR * 1000
        check(f"swing {pre_name}", rel_tip > -12 and rel_all > -12 and np.all(np.isfinite(sw)),
              f"at top speed: tip source lift peak {f_tip:.0f} Hz ({rel_tip:+.1f} dB re strongest {strongest_tip:.0f} Hz), "
              f"all 8 sources {f_all:.0f} Hz ({rel_all:+.1f} dB re {strongest_all:.0f} Hz); 0.2u/d = {f02:.0f} Hz, "
              f"St(Re)u/d = {fst:.0f} Hz; loudest {t_loud:+.0f} ms from top speed; length {len(sw) / SR * 1000:.0f} ms")
    print("modal self-test", "OK" if ok else "FAILED")


REN_CZ, REN_CD = 5.0, 100.0


def ren_features(f, d):
    """Perceptual feature point of a mode (f Hz, damping d 1/s): (cz Z(f), cd / d), Z(f) = 6 asinh(f / 600) Bark."""
    f, d = np.asarray(f, float), np.asarray(d, float)
    return np.stack([REN_CZ * 6.0 * np.arcsinh(f / 600.0), REN_CD / d], axis=-1)


def ren_match_score(D):
    """Point-to-point match k(D): 1 - D / 2 up to D = 1, then 0.5 / D up to 5, then 0."""
    D = np.asarray(D, float)
    return np.where(D <= 1.0, 1.0 - 0.5 * D, np.where(D <= 5.0, 0.5 / np.maximum(D, 1e-12), 0.0))


def ren_match_ratio(ref, est, weights=None):
    """How well the estimated modes ``est`` [(f, d), ...] cover the reference ``ref``: sum_i w_i R_i / sum_i w_i,
    R_i = sum_j u_ij k_ij / sum_j u_ij, u_ij = 1 where k_ij > 0. 1 = every reference mode matched exactly."""
    a, b = ren_features(*np.asarray(ref, float).T), ren_features(*np.asarray(est, float).T)
    k = ren_match_score(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1))
    u = k > 0
    r = np.where(u.any(axis=1), (k * u).sum(axis=1) / np.maximum(u.sum(axis=1), 1), 0.0)
    w = np.ones(len(r)) if weights is None else np.asarray(weights, float)
    return float((w * r).sum() / w.sum())
