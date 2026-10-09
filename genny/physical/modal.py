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

import numba
import numpy as np

from . import modal_data as md
from .core import (SR, DEV_DIR, contact_pulse, decay_rate, pow_decay, rate_radius, seconds, spectrum_peaks,
                   tv_bandpass, write_wav)

LN1000 = float(np.log(1000.0))


# ==========================================================================================
# Resonator bank
# ==========================================================================================

@numba.njit(cache=True)
def _bank(x, b0, b2, a1, a2, stop, kill, fade):
    """Sum of biquads y_k = b0 x + b2 x[n-2] - a1 y[n-1] - a2 y[n-2].

    ``stop[k]``: sample after which mode k is no longer computed (it has decayed away).
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


def _interp_rows(rows, position: float) -> np.ndarray:
    """Gain vector at a fractional strike position in [0, 1] across the measured points."""
    rows = np.asarray(rows, float)
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
        g = _interp_rows(t["gains"], position) * t["ascale"] * a_scale
        m = Modes(f, d, g, "vdd")
        return m.keep((f > 0) & (f < 0.45 * SR))            # VdD code: skip fn >= 0.45 sr
    if kind == "faust":
        n = len(t["gains"][0])
        g = _interp_rows(t["gains"], position) * a_scale
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
    return _bank(force, b0, b2, a1, a2, stop, int(kill_at), fade)


def ring_time(modes: Modes, db: float = 60.0) -> float:
    """Time for the gain-weighted mode sum to fall ``db`` below its start."""
    if len(modes.freqs) == 0:
        return 0.0
    g = np.abs(modes.gains) / modes.norm
    t = np.linspace(0, 60, 6001)
    env = (g[:, None] * np.exp(-modes.decays[:, None] * t[None, :])).sum(axis=0)
    below = np.nonzero(env < env[0] * 10 ** (-db / 20))[0]
    return float(t[below[0]]) if len(below) else 60.0


# ==========================================================================================
# Excitation
# ==========================================================================================

def bang(ms: float, magnitude: float = 1.0) -> np.ndarray:
    """VdD ``BangForce``: 0.5 (1 - cos(2 pi (i+1)/(N+1))), scaled to area ``magnitude``."""
    return contact_pulse(ms) * magnitude


def contact_time(contact_ms: float, velocity: float) -> float:
    """Loudness -> brightness: louder contacts are shorter (JOS force-pulse synthesis).

    # UNSOURCED: exponent -1/5 is the Hertz sphere-on-plane law (VdD-P cites Hertz/Johnson 1985
    # but gives no formula); it only sets how fast the pulse narrows with velocity.
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
    """Add FoleyAutomatic micro-collisions in place: a short train of re-contacts spaced by the
    periods of the dominant mode (VdD-P p.5).

    # UNSOURCED: the re-contact spacing (1-3 periods of the dominant mode), the amplitude ratio
    # (0.35-0.7 per re-contact) and the halved contact time are not given by the source.
    """
    if count <= 0 or len(modes.freqs) == 0:
        return
    dom = modes.freqs[int(np.argmax(np.abs(modes.gains)))]
    t = start + len(bang(contact_ms))
    amp = velocity
    for _ in range(count):
        t += int(round(rng.integers(1, 4) * SR / dom))
        amp *= rng.uniform(0.35, 0.7)
        p = bang(contact_ms * 0.5, amp)
        if t + len(p) >= len(force):
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


def additive_bell(table: dict | str, f0: float, dur: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Risset (``risset_bell``) or chinatown (``chinabell``) additive bell.

    Risset: partial k at ratio*f0 + detune, amplitude amp*0.1, quartic envelope with 5 ms attack
    and decay over dur*rel_duration. Chinabell: ratio*f0, amp, ead~(5 ms, decay_ms)^4 (``dur``
    unused except as the render length). Phases free-run (random), as Farnell allows for bells.
    """
    rng = rng or np.random.default_rng(0)
    t = md.TABLES[table] if isinstance(table, str) else table
    n = seconds(dur)
    tt = np.arange(n) / SR
    out = np.zeros(n)
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

    # UNSOURCED: Z-J give only the band and mode count; frequencies are drawn uniformly, gains
    # Gaussian / sqrt(N), and the decay rate 60 /s (a dull concrete thud) is chosen here.
    """
    f = rng.uniform(700.0, 4000.0, n_modes)
    g = rng.normal(0, 1, n_modes) / np.sqrt(n_modes)
    d = np.full(n_modes, 60.0)
    return Modes(f, d, g, "vdd")


def fracture(table: dict | str, rng: np.random.Generator, *, f_scale: float = 1.0, toughness: float = 80.0,
             t_break: float = 0.004, n_pieces: int | None = None, contact_ms: float = 0.2,
             velocity: float = 1.0, fall_height: float = 0.8, restitution: float = 0.45,
             ground: bool = True, ground_gain: float = 0.3, position: float = 0.0,
             crack_gain: float = 1.0, dur: float | None = None, f0: float | None = None) -> np.ndarray:
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
    """
    parent = resolve(table, f_scale, position, f0=f0)
    alpha, beta = _rayleigh_fit(parent)
    if n_pieces is None:
        # UNSOURCED: Z-J give only the trend (lower G_c -> more debris); count = 6 at glass G_c.
        n_pieces = int(np.clip(round(6 * 80.0 / max(toughness, 1.0)), 2, 24))
    # UNSOURCED: volume fractions from a flat Dirichlet (Voronoi pieces seeded by strain energy
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
        # UNSOURCED: energy share proportional to volume -> impulse amplitude sqrt(vol)
        c = crack * crack_gain * velocity * np.sqrt(vol[i])
        fpi[kb:kb + len(c)] += c[: n - kb]
        for (j, t, v) in events:
            if j != i:
                continue
            k = seconds(t)
            if k >= n:
                continue
            # UNSOURCED: bounce contact 0.1 ms at 1 m/s, amplitude ~ v * sqrt(volume)
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
    return P_REF * 10.0 ** (k * lg)


def _cascade_bp(x, f, q):
    """Two cascaded vcf~ at the same centre and Q (Dipoles2017D1)."""
    return tv_bandpass(tv_bandpass(x, f, q), f, q)


def aeolian_tone(u: np.ndarray | float, d: float, dur: float, rng: np.random.Generator,
                 model: str = "patch") -> np.ndarray:
    """One compact cylinder source at speed ``u`` (m/s, scalar or per-sample), diameter ``d`` (m).

    Lift fundamental only (unit gain): noise through two cascaded band-passes at St(Re) u / d
    with the bandwidth-derived Q. ``model="loq"`` fixes St = 0.2 and Q = 10 (Selfridge LoQ).
    """
    n = seconds(dur)
    u = np.broadcast_to(np.asarray(u, float), (n,)).copy()
    re = RHO_AIR * u * d / MU_AIR
    st = np.full(n, 0.2) if model == "loq" else fey_strouhal(re)
    q = np.full(n, 10.0) if model == "loq" else strouhal_q(re)
    f = np.clip(st * u / d, 1.0, 0.45 * SR)
    return _cascade_bp(rng.standard_normal(n), f, q)


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
    # UNSOURCED: the patch moves elevation pre -> mid -> post; mid is not a preset field, so the
    # elevation is interpolated linearly with swing progress.
    prog = np.clip((az - pre) / (post - pre), 0, 1) if post != pre else np.zeros_like(az)
    el = p["PreElev"] + (p["PostElev"] - p["PreElev"]) * prog
    sign = 1.0 if post >= pre else -1.0
    return az, el, u, sign


def sword_swing(preset: str | dict = "SWD1", rng: np.random.Generator | None = None, *,
                model: str = "patch", doppler: bool = True, n_sources: int = 8,
                top_speed: float | None = None, lead: float = 0.05, tail: float = 0.15,
                partials: tuple[int, ...] = (1, 2, 3, 4, 5)) -> np.ndarray:
    """Selfridge Aeolian swing, stereo (n, 2), from a ``SWORD_PRESETS`` preset (SWD1, SWD2, PGA, MLB).

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
    p = dict(md.SWORD_PRESETS["presets"][preset]) if isinstance(preset, str) else dict(preset)
    if top_speed is not None:
        p["TopSpeed"] = top_speed
    az, el, u_tip, sign = _swing_kinematics(p, lead, tail)
    n = len(az)
    L = p["Len"]
    hilt, tip = p["HiltThick"], p["TipThick"]
    # UNSOURCED: axis convention - shoulder at the origin, x right, y front, z up; listener at the
    # preset (Xpos, Ypos, Zpos); azimuth 0 = behind, 180 = in front; elevation from vertical.
    listener = np.array([p["Xpos"], p["Ypos"], p["Zpos"]])
    phi = np.radians(az)
    eps = np.radians(el)
    a = np.stack([np.sin(eps) * np.sin(phi), -np.sin(eps) * np.cos(phi), np.cos(eps)], axis=1)
    v_dir = np.stack([np.sin(eps) * np.cos(phi), np.sin(eps) * np.sin(phi), np.zeros(n)], axis=1) * sign
    v_dir /= np.maximum(np.linalg.norm(v_dir, axis=1, keepdims=True), 1e-9)
    lift_dir = np.cross(a, v_dir)
    positions = [L]
    for _ in range(n_sources - 1):
        pos = positions[-1]
        dk = hilt + (tip - hilt) * pos / L
        positions.append(pos - 7 * dk)
    out = np.zeros((n, 2))
    for pos in positions:
        if pos <= 0:
            break
        d = hilt + (tip - hilt) * pos / L
        radius = ARM + pos
        u = u_tip * radius / (ARM + L)
        src = a * radius
        rel = listener[None, :] - src
        dist = np.maximum(np.linalg.norm(rel, axis=1), 0.3)   # UNSOURCED: 0.3 m floor on distance
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
        common = 2.50663 * RHO_AIR * st ** 2 * (7 * d) * L * u ** 6 / (32 * C_SOUND ** 3 * dist ** 2 * conv ** 4)
        gl = common * dl
        gd = 0.1 * common * dd
        noise = rng.standard_normal(n)
        sig = np.zeros(n)
        live = u > 0.05
        for mult, gain in ((1, gl), (2, gd), (3, _db_scale(gl, 0.6, False)), (4, _db_scale(gd, 0.125, True)),
                           (5, _db_scale(gl, 0.1, True))):
            if mult not in partials:
                continue
            fc = np.clip(mult * f, 20.0, 0.45 * SR)
            gain = np.where(live, gain, 0.0)
            sig += _cascade_bp(noise, fc, np.maximum(q, 0.5)) * gain
        # equal-power pan from the source's lateral offset (fcpan)
        pan = np.clip(-rel[:, 0] / dist, -1, 1)
        ang = (pan + 1) * np.pi / 4
        out[:, 0] += sig * np.cos(ang)
        out[:, 1] += sig * np.sin(ang)
    return out * p["Gain"]


def swing_peak_tone(preset: str | dict = "SWD1", model: str = "patch") -> float:
    """Tip lift-tone frequency at top speed, St(Re) u / d (no Doppler) - for checks."""
    p = md.SWORD_PRESETS["presets"][preset] if isinstance(preset, str) else preset
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
