"""Acoustic instruments: more keys, mallets and bells, plucked and bowed strings, winds and brass.

Registered into the instrument REGISTRY (imported at the bottom of instruments.py). Three engines:
  - `_string` (tuned Karplus-Strong) for everything plucked or struck on a string,
  - `_modes` (rolled-off inharmonic partials) for bars, bells, bowls and drums with a pitch,
  - `_blown` (a band-limited harmonic stack shaped by fixed body formants, whose brightness opens
    with the attack and velocity) for bowed strings, reeds, flutes and brass.
All three follow the voicing rules at the top of instruments.py: brightness is set in Hz, not in
harmonics, so nothing turns piercing as it climbs.
"""
from __future__ import annotations

import functools
import math

import numba
import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, mix
from .env import adsr, apply, lfo, perc
from .fx import effect
from .instruments import (_cut, _harmonics, _ks, _kt, _modes, _n, _off, _roll, _spectrum, _string, _vibrato,
                          decay_env, instrument)


# ---------------------------------------------------------------- keys
@instrument("wurli", "Wurlitzer-style reed electric piano: hollow and a little nasal, growls when hit hard (soul, indie, lo-fi).",
            family="keys", span=("C2", "C6"), tremolo=(0.3, "0..1 wobble"), decay=(1.4, "seconds"))
def wurli(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.3, decay=1.4):
    ring = min(decay, dur + 0.8)
    n = _n(ring, sr)
    # Pfeifle (DAFx-17 5.5-5.6): the reed moves in its first cantilever mode only, near-sinusoidal even when hit
    # hard; the pickup is a capacitor whose C goes as 1 / gap, read as the current i = u0 dC/dt. A reed at swing a
    # (in units of the rest gap) gives i ~ a w cos(wt) / (1 - a sin(wt))^2: pure when soft, lopsided and rich (even
    # and odd harmonics) when the swing nears the gap. the swing per velocity and its 0.22 s decay.
    a = np.clip((0.2 + 0.55 * _kt(freq, 1.0, 0.6) * vel) * (0.35 + 0.65 * decay_env(n, 0.22, sr)), 0.0, 0.85)
    c = 1.0 / (1.0 - a * O.sine(freq, n, sr))
    x = np.gradient(c) * sr / (2 * np.pi * freq) * 1.05
    x = F.lowpass(F.dc_block(x, sr), _cut(freq, 5, 1200, 3000), sr, 0.8, order=2)
    env = decay_env(n, ring / 2.5, sr) * _off(n, dur, sr, 0.06) * perc(ring, 0.002, sr, curve=0.6)
    if tremolo > 0:
        env = env * (1 - tremolo * 0.5 * (1 + lfo(n / sr, 5.8, sr)[:n]))
    return x * env                                    # the swing a already carries the velocity


@instrument("clav", "Clavinet: short, funky, nasal struck string (funk, disco, reggae skank).",
            family="keys", span=("C2", "C6"), decay=(0.9, "seconds"), bright=(1.0, "0.5 muted .. 1.5 biting"),
            release_drop=(1.0, "0..1 the brief 3-semitone drop when the key is let go and the yarn damps the string"),
            model=("ks", "ks | dwg: waveguide with ripple loss, per-key dispersion, tangent pulse and knock"),
            pickup=("A", "pickup selector: A upper (centre), B lower (bridge), C both out of phase (thin), D both in phase (full)"),
            magnetic=(0, "1 = the measured magnetic pickup law instead of the flux polynomial (with flux) or none"),
            tone=("", "tone rockers pressed, any of S M T B: Soft and Medium darken, Treble peaks near 1.6 kHz, "
                      "Brilliant cuts the bass ('' = none, the old sound)"),
            beat=(0.0, "dB of beating on partials 1-3 below E4, 0.5-2 Hz per partial (0 = off, up to 15)"))
def clav(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=0.9, bright=1.0, release_drop=1.0, flux=False, model="ks", pickup="A", magnetic=0,
         tone="", beat=0.0):
    """Hohner D6: pickup comb with notches every 4.67 partials, 3-semitone drop at key-off, amplifier
    shelves -3 dB at 130 Hz and +3 dB at 4 kHz."""
    t60 = decay * 1.5 * _kt(freq, 1.0, 0.45)
    ring = min(t60 * 0.9, dur + 0.25)
    if model == "dwg":
        y = _clav_dwg(freq, ring, sr, vel, t60)
    elif model == "ks":
        y = _string(freq, ring, sr, vel, t60=decay * 1.5, bright=3400 * bright, damp=0.15, pick=0.1, grit=0.1, ring=ring)
    else:
        raise ValueError(f"clav model {model!r}: ks or dwg")
    if beat > 0 and freq < CLAV_BEAT_MAX_HZ:
        y = clav_beat(y, freq, sr, min(float(beat), 15.0), int(freq * 10) % 9973)
    n0 = min(len(y), int(round(dur * sr)))
    t = np.arange(len(y) - n0) / sr
    off = np.exp(-t / 0.03)                                     # the damper (the old 30 ms release)
    held = y[n0:] * off
    k = float(np.clip(release_drop, 0.0, 1.0))
    if k > 0 and len(t) > 1:                                    # read the ringing string 3 semitones slower
        src = np.arange(len(y)) - n0
        dropped = np.interp(src[n0:] * 2 ** (-3 / 12), np.arange(len(y)) - n0, y) * off
        held = (1 - k) * held + k * dropped
    y = np.concatenate([y[:n0], held])
    i = np.arange(len(y))

    def comb(p):
        return y - np.interp(i - p * sr / freq, i, y, left=0.0)
    if pickup not in CLAV_PICKUPS:
        raise ValueError(f"clav pickup {pickup!r}: A, B, C or D")
    a, b = comb(CLAV_PICKUP_POS[0]), comb(CLAV_PICKUP_POS[1])
    y = {"A": a, "B": b, "C": a - b, "D": a + b}[pickup]
    if magnetic:
        y = magnetic_pickup(y, sr, "single", swing_mm=0.5 + 1.5 * float(np.clip(vel, 0.0, 1.0)))
    elif flux:
        # pickup flux: leaky integrator, polynomial, derivative, RMS-matched
        from scipy.signal import lfilter
        w = lfilter([1.0], [1.0, -(1.0 - 2 * math.pi * 10.0 / sr)], y)
        w = w - np.mean(w)
        x = 0.5e-3 * (0.5 + 0.5 * float(np.clip(vel, 0.0, 1.0))) * w / (np.max(np.abs(w)) + 1e-12)
        u = np.diff(np.polyval((1.817e5, -9.508e3, 1.818e2, -1.544, 0.7951), x), prepend=0.7951)
        y = u * np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(u ** 2)) + 1e-12)
    tone = str(tone).upper()
    if tone:
        y = clav_tone(y, sr, tone)
    y = F.highshelf(F.lowshelf(y, 130.0, sr, -3.0), 4000.0, sr, 3.0)
    y = F.highpass(y, max(120.0, freq * 0.6), sr)
    top = CLAV_TONE_TOP if ("T" in tone or "B" in tone) else 4200
    return F.lowpass(y, top, sr, 0.7) * 0.8 / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel)


CLAV_TONE_RS = 1.0e4
CLAV_TONE_RL = 1.0e6
CLAV_TONE_Q = 5.0
CLAV_TONE_TOP = 9000.0
CLAV_TONE = {"S": ("rc", 30e3, 0.1e-6), "M": ("rc", 10e3, 15e-9), "T": ("lc", 2.0, 4.7e-9), "B": ("l", 0.6, 0.0)}


def clav_tone(y, sr, tone):
    """Hohner D6 tone rockers: each pressed switch hangs its network across the line between the pickups
    (source CLAV_TONE_RS) and the amplifier input (CLAV_TONE_RL). Soft and Medium are a series R-C to ground,
    Treble a parallel L-C, Brilliant an inductor; the coils lose with quality CLAV_TONE_Q."""
    bad = set(tone) - set(CLAV_TONE)
    if bad:
        raise ValueError(f"clav tone {tone!r}: letters among S M T B")
    nf = 1 << int(np.ceil(np.log2(len(y) + sr)))
    s = 2j * np.pi * np.fft.rfftfreq(nf, 1.0 / sr)
    s[0] = 1e-6
    adm = np.full(s.shape, 1.0 / CLAV_TONE_RL, complex)
    for k in set(tone):
        kind, a, b = CLAV_TONE[k]
        if kind == "rc":
            adm += 1.0 / (a + 1.0 / (s * b))
        else:
            w0 = 1.0 / math.sqrt(a * b) if kind == "lc" else CLAV_TONE_RS / a
            zl = a * w0 / CLAV_TONE_Q + s * a
            adm += 1.0 / zl + (s * b if kind == "lc" else 0.0)
    h = 1.0 / (1.0 + CLAV_TONE_RS * adm)
    h = h / (1.0 / (1.0 + CLAV_TONE_RS / CLAV_TONE_RL))
    return np.fft.irfft(np.fft.rfft(y, nf) * h, nf)[:len(y)]


CLAV_BEAT_MAX_HZ = 329.63
CLAV_BEAT_HZ = (0.5, 2.0)
CLAV_BEAT_PARTIALS = 3


def regalia_mitra(x, sr, fc, gain, bw):
    """Regalia-Mitra peak y = x + (K - 1) / 2 (x - A2 x), A2 the second-order allpass at ``fc`` with bandwidth ``bw`` Hz;
    the gain K may be one value per sample."""
    from scipy.signal import lfilter
    t = math.tan(math.pi * bw / sr)
    c = (t - 1.0) / (t + 1.0)
    d = -math.cos(2 * math.pi * fc / sr)
    ap = lfilter([-c, d * (1.0 - c), 1.0], [1.0, d * (1.0 - c), -c], x)
    return x + 0.5 * (np.asarray(gain) - 1.0) * (x - ap)


def clav_beat(y, freq, sr, depth_db, seed=0):
    """Beating of the low partials: a Regalia-Mitra peak on each of the first CLAV_BEAT_PARTIALS partials, its gain
    K[n] = 10^(depth |cos(2 pi f_b n / fs)| / 20), f_b drawn in CLAV_BEAT_HZ for each."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(y)) / sr
    for k in range(1, CLAV_BEAT_PARTIALS + 1):
        if k * freq >= 0.45 * sr:
            break
        fb = float(rng.uniform(*CLAV_BEAT_HZ))
        y = regalia_mitra(y, sr, k * freq, 10.0 ** (depth_db * np.abs(np.cos(2 * np.pi * fb * t)) / 20.0), 0.25 * freq)
    return y


CLAV_PICKUPS = ("A", "B", "C", "D")
CLAV_PICKUP_POS = (1.0 / 4.67, 0.08)
_CLAV_B = ((29, 5e-4), (34, 2e-4), (50, 9e-5), (51, 1e-4), (78, 9e-5), (88, 8e-5))
_CLAV_RAMP = (-2.69e-8, 2.53e-6, -9.54e-5, 1.74e-3, -1.44e-2, 4.50e-2, -3.50e-2)


def clav_b(freq: float) -> float:
    midi = 69.0 + 12.0 * math.log2(freq / 440.0)
    return float(np.interp(midi, [k for k, _ in _CLAV_B], [b for _, b in _CLAV_B]))


def clav_pulse(vel: float, sr: int, d: float = 0.004) -> np.ndarray:
    """Tangent pulse: the fitted ramp from its rising zero to its peak, then reversed, N = fs d / v
    samples with v from 1 to 4 m/s over the velocity range."""
    roots = np.roots(_CLAV_RAMP)
    x0 = min(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 10)
    x1 = min(r.real for r in np.roots(np.polyder(_CLAV_RAMP)) if abs(r.imag) < 1e-9 and r.real > x0)
    v = 1.0 + 3.0 * float(np.clip(vel, 0.0, 1.0))
    half = max(2, int(round(sr * d / v / 2)))
    ramp = np.polyval(_CLAV_RAMP, np.linspace(x0, x1, half))
    ramp = ramp / ramp[-1]
    return np.concatenate([ramp, ramp[::-1]])


def _clav_dwg(freq, ring, sr, vel, t60):
    n = _n(ring, sr)
    rng = np.random.default_rng(int(freq * 10) % 9973)
    r = -float(rng.uniform(0.001, 0.006))
    rrate = float(rng.uniform(1.0 / 3.0, 0.5))
    t60 = min(t60, -3.0 / (freq * math.log10(1.0 - abs(r))) * 0.999)            # g + |r| < 1
    exc = clav_pulse(vel, sr) * (0.5 + 0.5 * vel)
    y = _harpsi_string(freq, n, n, sr, exc, t60, clav_b(freq), a=0.0, r=r, rrate=rrate)
    m = int(0.03 * sr)
    knock = F.lowpass(rng.standard_normal(m), float(rng.uniform(900.0, 1200.0)), sr) * np.exp(-np.arange(m) / (0.006 * sr))
    y = y / (np.max(np.abs(y)) + 1e-9)
    y[:m] += 0.1 * knock[:n] / (np.max(np.abs(knock)) + 1e-9)
    return y


# Rhodes Mark I tine/tonebar Q (Shear, MSc UCSB 2011, Table 2.1): amplitude time constant tau = Q / (pi f0)
# B3-D4: his Table 5.1 (C4, Db4, D4 on the modified tines; "no appreciable effect" on the decay)
_RHODES_Q = ((77.8, 949.0), (155.6, 731.0), (246.9, 1101.0), (261.6, 1238.0), (277.1, 1040.0), (293.7, 1156.0),
             (311.1, 1520.0), (622.3, 2175.0), (1244.5, 1761.0))


@instrument("rhodes", "Rhodes tine piano: a struck cantilever tine read by a magnetic pickup whose field shapes the tone (bell attack, bark when played hard).",
            family="keys", span=("E1", "E7"),
            voicing=(0.35, "0..1 tine offset from the pickup axis: 0 = centred (2nd harmonic leads, hollow), 1 = far off (round, fundamental)"),
            bell=(1.0, "0..2 x the tine overtones the hammer excites at its strike point (short, inharmonic)"),
            decay=(1.0, "x the measured sustain (Q of a 1974 Mark I)"),
            sustain=(0.0, "0..1 electromagnetic sustain while the key is held: 1 = the coil cancels the tine's losses"),
            onset=("hammer", "hammer | swell: swell = the coil alone builds the note up, no hammer, no bell"),
            tremolo=(0.0, "0..1 depth of the amp's tremolo (about 5 Hz)"),
            tonebar=("fork", "fork | 2dof: tine and aluminium tonebar as two masses coupled through the clamp"),
            tonebar_cents=(0.0, "with 2dof: the tonebar's mistuning against the tine (the fork turns asymmetric)"),
            tuning_mass=(0.0, "tuning spring at the tine tip, as a fraction of the tine's effective mass: "
                              "lowers the tine by 1/sqrt(1 + m)"))
def rhodes(freq, dur, sr=DEFAULT_SR, vel=1.0, voicing=0.35, bell=1.0, decay=1.0, sustain=0.0, onset="hammer", tremolo=0.0,
           tonebar="fork", tonebar_cents=0.0, tuning_mass=0.0):
    """Tine: cantilever modes 1, 6.267, 17.55 (clamped-free Euler-Bernoulli; the lowest keys, without tonebar, at the
    measured 7.11 and 20.25) excited by a felt hammer at RHODES_STRIKE of the tine: each overtone's level is the mode
    shape at the strike point times the hammer's force spectrum (contact time of the Hunt-Crossley hammer) over its
    stiffness. The fundamental rings with the measured Q; the tonebar, tuned with the tine, makes an asymmetric fork
    whose second branch beats slowly against it. Pickup: magnetic charge with the tip on its arc, voltage = -d(flux)/dt.
    The damper is the felt's resistance R on the tine's modal mass. ``sustain``: the coil drives the tine in phase with
    its velocity, dA/dt = -(1 - g) A / tau; its drive leaks into the pickup in antiphase, so the fundamental is less
    marked while sustained. ``onset`` swell: the coil alone raises the note (fundamental only)."""
    f0 = float(freq)
    v = float(np.clip(vel, 0.0, 1.0))
    q = float(np.exp(np.interp(np.log(f0), np.log([a for a, _ in _RHODES_Q]), np.log([b for _, b in _RHODES_Q]))))
    tau = max(q / (math.pi * f0) * float(np.clip(decay, 0.05, 4.0)), 0.05)
    g = float(np.clip(sustain, 0.0, 1.0))
    swell = onset == "swell"
    held = tau / max(1.0 - g, 1e-3)
    ring = min(dur, 6.91 * held) + min(1.5, 6.91 * tau)
    n = _n(ring, sr)
    n_off = min(n, int(round(dur * sr)))
    t = np.arange(n) / sr
    L = rhodes_tine_length(f0)
    # fundamental envelope: hammer decay (or coil swell) with the coil's sustain while held, the damper after
    rate = np.where(np.arange(n) < n_off, (1.0 - g) / tau, 1.0 / rhodes_damper_tau(f0, L))
    env = np.exp(-np.cumsum(rate) / sr)
    if swell:
        env = env * (1.0 - np.exp(-t / RHODES_SWELL_S))
    ft = f0 / math.sqrt(1.0 + max(float(tuning_mass), 0.0))
    if tonebar == "2dof":
        u = env * sum(a * np.sin(2 * math.pi * fk * t)
                      for fk, a in rhodes_tonebar_modes(ft, f0 * 2.0 ** (float(tonebar_cents) / 1200.0), L, f0))
    elif tonebar == "fork":
        fork = 1.0 + RHODES_FORK_SPLIT * np.array([-0.5, 0.5])
        u = 0.5 * env * (np.sin(2 * math.pi * ft * fork[0] * t) + np.sin(2 * math.pi * ft * fork[1] * t))
    else:
        raise ValueError("rhodes tonebar must be fork or 2dof")
    b = float(np.clip(bell, 0.0, 2.0))
    tc = rhodes_contact_time(0.5 + 3.5 * v)
    if not swell and b > 0:
        for r, lvl, d in zip(rhodes_ratios(f0), rhodes_overtones(f0, tc), (0.04, 0.015)):
            if r * f0 < 0.45 * sr:
                u = u + b * lvl * np.sin(2 * math.pi * r * f0 * t) * np.exp(-t / d)
    if not swell:
        u = u * np.minimum(1.0, t / tc)                                 # hammer contact
    amp = 0.25 + 0.95 * v                                               # tine swing / pickup distance
    if swell:
        amp = 0.25 + 0.95 * np.tanh(2.0 * v)                            # the coil's drive saturates
    x = amp * u + 1.2 * float(np.clip(voicing, 0.0, 1.0))
    r = L / 1.5e-3
    dz = 1.0 + r - np.sqrt(np.maximum(r * r - x * x, 0.0))
    flux = dz * (x * x + dz * dz) ** -1.5
    y = -np.diff(flux, prepend=flux[0]) * sr
    if g > 0 or swell:                                                  # the coil's drive, read in antiphase
        drive = np.where(np.arange(n) < n_off, 1.0, 0.0) * env * np.cos(2 * math.pi * f0 * t)
        y = y - RHODES_LEAK * np.max(np.abs(y)) * drive * (g if not swell else 1.0)
    y = F.dc_block(y, sr)
    y = F.lowpass(y, min(5000.0, 0.45 * sr), sr, 0.7)
    if tremolo > 0:
        y = y * (1.0 - 0.5 * float(np.clip(tremolo, 0.0, 1.0)) * (1.0 + np.sin(2 * math.pi * 5.0 * t)))
    return y / (np.max(np.abs(y)) + 1e-9) * (0.55 + 0.45 * vel)


RHODES_STRIKE = 0.3                 # strike point / tine length
RHODES_HAMMER = (0.011, 2.8, 1.5e11)  # mass kg, alpha, k N/m^alpha
RHODES_DAMPER_R = 0.5               # kg/s, felt damper resistance
RHODES_FORK_SPLIT = 0.003           # relative split of the tine-tonebar pair
RHODES_LEAK = 0.1                   # coil drive read by the pickup, relative to the tine signal
RHODES_SWELL_S = 0.25               # rise time of the coil-only onset
RHODES_LOW_RATIOS = (43.65, (7.11, 20.25))   # lowest key (no tonebar): measured overtone ratios
STEEL_RHO = 7850.0
TINE_D = 1.5e-3


TONEBAR_E, TONEBAR_RHO = 69e9, 2700.0
TONEBAR_L, TONEBAR_W, TONEBAR_H = 0.0821, 0.0111, 0.0034
TONEBAR_MASS = 0.346
TINE_MODAL_MASS = 0.2427


def tonebar_freq(L: float) -> float:
    """Free-free bar: 1.426 pi K / (8 L^2) sqrt(E / rho), K = h / sqrt(12)."""
    return 1.426 * math.pi * (TONEBAR_H / math.sqrt(12.0)) / (8.0 * L * L) * math.sqrt(TONEBAR_E / TONEBAR_RHO)


def rhodes_tonebar_modes(ft: float, fb: float, L: float, f0: float) -> list:
    """(frequency, tine amplitude) of the two modes of tine (tip mass TINE_MODAL_MASS of the tine) and tonebar
    (mass TONEBAR_MASS L w h rho, its length cut for fb) joined by the clamp, whose stiffness splits the tuned pair
    by RHODES_FORK_SPLIT. The hammer strikes the tine; amplitudes sum to 1."""
    from scipy.linalg import eigh
    m1 = TINE_MODAL_MASS * STEEL_RHO * math.pi * TINE_D ** 2 / 4.0 * L
    lb = TONEBAR_L * math.sqrt(tonebar_freq(TONEBAR_L) / fb)
    m2 = TONEBAR_MASS * lb * TONEBAR_W * TONEBAR_H * TONEBAR_RHO
    kc = RHODES_FORK_SPLIT * math.sqrt(m1 * m2) * (2.0 * math.pi * f0) ** 2
    k1, k2 = m1 * (2.0 * math.pi * ft) ** 2, m2 * (2.0 * math.pi * fb) ** 2
    w2, phi = eigh(np.array([[k1, -kc], [-kc, k2]]), np.diag([m1, m2]))
    w = np.sqrt(np.maximum(w2, 1e-9))
    a = phi[0] ** 2 / w
    return list(zip(w / (2.0 * math.pi), a / a.sum()))


def rhodes_tine_length(f0: float) -> float:
    """Tine length m: 18-157 mm over the keyboard, ~ 1 / sqrt(f) as a cantilever."""
    return float(np.clip(0.157 * math.sqrt(41.0 / f0), 0.018, 0.157))


def rhodes_ratios(f0: float) -> tuple[float, float]:
    """Overtone ratios of the tine: ideal clamped-free beam above an octave over the lowest key, the measured lowest-key
    ratios at it, log-interpolated between."""
    fl, (a, b) = RHODES_LOW_RATIOS
    x = float(np.clip(math.log2(max(f0, 1.0) / fl), 0.0, 1.0))
    return a + (6.267 - a) * x, b + (17.55 - b) * x


def rhodes_contact_time(v: float) -> float:
    """Contact time s of a power-law hammer F = k x^alpha of mass m hitting at v m/s (elastic part of Hunt-Crossley):
    2 (x_m / v) sqrt(pi) Gamma(1 + 1/n) / Gamma(1/2 + 1/n), n = alpha + 1, x_m = (n m v^2 / (2 k))^(1/n)."""
    m, al, k = RHODES_HAMMER
    nn = al + 1.0
    xm = (nn * m * v * v / (2.0 * k)) ** (1.0 / nn)
    return 2.0 * xm / v * math.sqrt(math.pi) * math.gamma(1.0 + 1.0 / nn) / math.gamma(0.5 + 1.0 / nn)


def _cantilever_shape(bl: float, x: float) -> float:
    s = (math.cosh(bl) + math.cos(bl)) / (math.sinh(bl) + math.sin(bl))
    return math.cosh(bl * x) - math.cos(bl * x) - s * (math.sinh(bl * x) - math.sin(bl * x))


def rhodes_overtones(f0: float, tc: float) -> tuple[float, float]:
    """Tip displacement of modes 2 and 3 over mode 1's for a hammer of contact time ``tc`` at RHODES_STRIKE: mode shape
    at the strike times shape at the tip, over the modal stiffness (ratio^2), times the half-sine force spectrum
    |cos(w tc / 2) / (1 - (w tc / pi)^2)|."""
    def spec(f):
        a = 2 * math.pi * f * tc
        d = 1.0 - (a / math.pi) ** 2
        return abs(math.cos(a / 2.0) / d) if abs(d) > 1e-6 else math.pi / 4.0
    out = []
    p1 = _cantilever_shape(1.8751, RHODES_STRIKE) * _cantilever_shape(1.8751, 1.0) * spec(f0)
    for bl, r in zip((4.6941, 7.8548), rhodes_ratios(f0)):
        pk = _cantilever_shape(bl, RHODES_STRIKE) * _cantilever_shape(bl, 1.0) * spec(r * f0)
        out.append(abs(pk / p1) / (r * r))
    return tuple(out)


def rhodes_damper_tau(f0: float, L: float | None = None) -> float:
    """Amplitude time constant s of the damped tine: 2 m_eff / R, m_eff = 0.2427 of the tine's mass (clamped-free
    first mode), never under two periods."""
    L = rhodes_tine_length(f0) if L is None else L
    m = STEEL_RHO * math.pi * TINE_D ** 2 / 4.0 * L
    return max(2.0 * 0.2427 * m / RHODES_DAMPER_R, 2.0 / f0)


def tonebar_length(f0: float, w: float = 0.0111, h: float = 0.0034, E: float = 69e9, rho: float = 2700.0,
                   m_e: float = 0.030, L_e: float = 0.0268, coef: float = 0.243) -> float:
    """Tonebar length m tuned to f0 with its end mass m_e at L_e: f0 = (1/2pi) sqrt(3EI / ((m_t' + m_e') Lt^3)),
    m_t' = coef Lt w h rho (the beam's exact share 3 / 1.8751^4 = 0.243), m_e' = m_e (Le/Lt)^3, so
    Lt^4 = (3EI / (2 pi f0)^2 - m_e Le^3) / (coef w h rho)."""
    I = w * h ** 3 / 12.0
    return float(((3.0 * E * I / (2 * math.pi * f0) ** 2 - m_e * L_e ** 3) / (coef * w * h * rho)) ** 0.25)


def beam_kirchhoff_shift(a: float, d: float) -> float:
    """Relative pitch rise of a round beam held at both ends swinging at amplitude a (Kirchhoff axial stretching),
    3/32 (a / r_g)^2 with r_g = d / 4. A cantilever (Rhodes tine, Wurlitzer reed) has a free tip: no axial tension,
    shift 0."""
    return 3.0 / 32.0 * (a / (d / 4.0)) ** 2


# Harpsichord after Valimaki, Penttinen, Knif, Laurson & Erkut, "Sound synthesis of the harpsichord using a
# computationally efficient physical model", EURASIP JASP 2004(7):934. String loop: delay + first-order Thiran
# fraction + dispersion allpasses + the "ripply" loss filter H(z) = g (1 + a) (r + z^-R) / (1 + a z^-1),
# R = round(rrate L) (eqs 1-3), which lets neighbouring partials decay at different rates. A second-order
# resonator a fraction of a hertz off one low partial, at its level, gives the beating (4.4; for the 197 Hz key the
# 10th partial beats with a period of about 1.5 s, Fig. 9b: offset 1 / 1.5 = 0.67 Hz). The soundboard is a
# reverberator with T60 6 s at DC and 0.13 of that at Nyquist on 8 lines of 1009-1999 samples, high-passed at
# 350 Hz (4.5). Their calibrated values for the 197 Hz key (Fig. 8): g 0.996, a -0.0296, r -0.0015, rrate 1/12.
# those values used on every key with g from a key-tracked T60, the quill excitation (a short pulse
# plus scrape noise; theirs are recorded residuals), the release knock (theirs
# is a recorded sample) and the soundboard level.

@numba.njit(cache=True)
def _harpsi_loop(exc, n, n_off, L, frac_a, g, a, r, R, disp, damp_g):
    """Delay L -> Thiran fraction -> dispersion -> g (1 + a) / (1 + a z^-1) -> (r + z^-R), fed back; the
    loop gain drops to g * damp_g from sample n_off (the damper)."""
    size = L + 2
    buf = np.zeros(size)
    hist = np.zeros(R + 1)                                         # one-pole output history for z^-R
    out = np.zeros(n)
    dx = np.zeros(len(disp)); dy = np.zeros(len(disp))
    tx = 0.0; ty = 0.0; lp = 0.0
    w = 0; h = 0
    for i in range(n):
        v = buf[(w - L) % size]
        x = frac_a * v + tx - frac_a * ty; tx = v; ty = x
        for k in range(len(disp)):
            o = disp[k] * x + dx[k] - disp[k] * dy[k]; dx[k] = x; dy[k] = o; x = o
        gi = g if i < n_off else g * damp_g
        lp = gi * (1.0 + a) * x - a * lp
        hist[h] = lp
        y = r * lp + hist[(h - R) % (R + 1)]
        h = (h + 1) % (R + 1)
        out[i] = y
        buf[w] = y + (exc[i] if i < len(exc) else 0.0)
        w = (w + 1) % size
    return out


def ripple_rrate(k_max: int, r: float) -> float:
    """Ripple delay ratio aimed at partial k_max: 1 / k_max when r >= 0, 1 / (2 k_max) when r < 0."""
    k = max(2, int(k_max))
    return 1.0 / k if r >= 0 else 1.0 / (2 * k)


HARPSI_THIRAN_LO = 0.5


def _harpsi_string(freq, n, n_off, sr, exc, t60, B, a=-0.0296, r=-0.0015, rrate=1.0 / 12.0, damp=None, ripple_partial=0,
                   fr_lo=HARPSI_THIRAN_LO):
    """One harpsichord string tuned to ``freq``: the delay is one period minus the phase delay of every loop
    filter at f0 (one-pole, ripple comb, dispersion), the rest set by the Thiran fraction, kept in [fr_lo, fr_lo + 1)
    (0.618 centres the window on the Thiran coefficient's well-behaved range). ``ripple_partial``: rrate from
    ``ripple_rrate``."""
    from .strings import rauhala_dispersion
    if ripple_partial:
        rrate = ripple_rrate(ripple_partial, r)
    w0 = 2 * math.pi * freq / sr
    P = sr / freq
    R = max(1, int(round(rrate * P)))
    g = 10 ** (-3.0 / (max(t60, 0.05) * freq))
    disp = np.array(rauhala_dispersion(freq, B, sr), float)
    z = np.exp(-1j * w0)
    H = (1 + a) / (1 + a * z) * (r + z ** R)
    for c in disp:
        H *= (c + z) / (1 + c * z)
    total = P + np.angle(H) / w0                                    # angle < 0 = lag; remove it from the line
    L = int(math.floor(total - fr_lo))
    fr = total - L
    fr_a = math.sin((1.0 - fr) * w0 / 2.0) / math.sin((1.0 + fr) * w0 / 2.0)   # exact phase delay at f0
    return _harpsi_loop(exc, n, n_off, max(L, 2), fr_a, g, a, r, R, disp, 0.35 ** (1.0 / max(P, 1.0)) if damp is None else damp)


PLECTRUM = {"L": 6e-3, "EI": 3e9 * 0.029e-12, "T": 92.0, "Ls": 0.5, "l0": 5e-3}


@functools.lru_cache(maxsize=64)
def plectrum_release(beta: float, steps: int = 400) -> tuple[float, float, float]:
    """Quasi-static quill on a string plucked at ratio ``beta``: the plectrum is a cantilever bent to second order in F, the
    string a spring T / (beta (1 - beta) Ls) in both transverse directions, the contact frictionless (force normal to the
    quill). The jack rises until the string has slid to the tip. Returns (F N, x m, y m) at release."""
    from scipy.optimize import fsolve
    L, EI, T, Ls, l0 = (PLECTRUM[k] for k in ("L", "EI", "T", "Ls", "l0"))
    b = float(np.clip(beta, 0.02, 0.5))
    ks = T / (b * (1.0 - b) * Ls)
    v = np.array([0.0, l0, 0.0, 0.0])
    for yj in np.linspace(0.0, 10e-3, steps):
        def eq(s, yj=yj):
            F, lp, xs, ys = s
            th = F * lp * lp / (2 * EI)
            return [ks * xs - F * math.sin(th), ks * ys - F * math.cos(th),
                    xs + l0 - lp + F * F * lp ** 5 / (15 * EI * EI),
                    ys - yj + F * lp ** 3 / (3 * EI) - F ** 3 * lp ** 7 / (105 * EI ** 3)]
        v = fsolve(eq, v)
        if v[1] >= L:
            break
    return float(v[0]), float(v[2]), float(v[3])


@instrument("harpsichord", "Harpsichord (Valimaki et al. 2004): quill-plucked strings with ripple loss filter, beating partial and a ringing soundboard; 4-foot octave rank.",
            family="keys", span=("C2", "C6"), octave=(0.3, "level of the 4-foot octave rank (0 = 8-foot only)"),
            soundboard=(0.35, "ringing soundboard and undamped strings behind the bridge 0..1"),
            stops=(1, "8-foot ranks: 1, or 2 (a second string a little apart, +2-3 dB)"),
            lid=(1.0, "0..1 lid resonance at 120 Hz"),
            pluck=(0.0, "plucking point / string length: nearer the nut (smaller) = thinner and softer (quill model); "
                        "0 = by register, 12 % in the bass to 50 % in the treble"),
            phantom=(0.0, "0..1 longitudinal resonances driven by the square of the string's motion (bright, metallic onset)"),
            ripple_partial=(0, "partial (>= 2) whose loss the ripple filter aims at: rrate 1/k (r >= 0) or 1/(2k) (r < 0); "
                               "0 = the measured 197 Hz key's 1/12 on every key"))
def harpsichord(freq, dur, sr=DEFAULT_SR, vel=1.0, octave=0.3, soundboard=0.35, stops=1, lid=1.0, pluck=0.0, phantom=0.0,
                ripple_partial=0):
    from scipy.signal import cheby1, lfilter, sosfilt
    from .reverbs import fdn_reverb
    beta = round(float(np.clip(pluck if pluck > 0 else harpsichord_pluck_point(freq), 0.04, 0.5)), 3)
    ring = dur + 0.25
    n, n_off = _n(ring, sr), int(round(dur * sr))
    rng = np.random.default_rng(int(freq * 10) % 9973)
    m = max(4, int(0.0006 * sr))                                    # quill: sharp release + scrape
    pulse = np.concatenate([np.linspace(0, 1, m), np.linspace(1, -0.3, m // 2)]) * 0.6
    # articulation peak 20-30 ms, mostly above 2 kHz (Meyer 2009, 3.5). its level
    scrape = F.highpass(rng.standard_normal(int(0.025 * sr)), 2000, sr) * np.exp(-np.arange(int(0.025 * sr)) / (0.008 * sr)) * 0.08
    exc = np.zeros(max(len(pulse), len(scrape)))
    exc[:len(pulse)] += pulse
    exc[:len(scrape)] += scrape
    lag = max(1, int(round(beta * sr / freq)))
    exc = np.concatenate([exc, np.zeros(lag)]) - np.concatenate([np.zeros(lag), exc])
    t60 = 20.0 * 1.4 ** -math.log2(freq / 130.81)                   # Meyer 2009, 3.5: about 20 s at C3, /1.4 per octave
    B = 1e-5 * _kt(freq, 1.0, 0.3, f_lo=65.0, f_hi=1200.0)          # Fig. 5: ~1e-5 on the low keys
    y1 = _harpsi_string(freq, n, n_off, sr, exc, t60, B, ripple_partial=ripple_partial)
    k = 10 if 10 * freq < 0.4 * sr else 2                           # beating against a low partial
    fb = k * freq * math.sqrt(1 + B * k * k) + 0.67
    rr = math.exp(-math.pi * fb / (math.sqrt(1 + B * k * k) * freq * t60 * 2.0))
    th = 2 * math.pi * fb / sr
    beat0 = lfilter([1.0 - rr], [1.0, -2 * rr * math.cos(th), rr * rr], np.pad(exc, (0, n - len(exc))))[:n]
    y4 = _harpsi_string(2 * freq, n, n_off, sr, exc, 0.7 * t60, B, ripple_partial=ripple_partial) \
        if octave > 0 and 2 * freq < 0.4 * sr else None
    knock = F.lowpass(rng.standard_normal(int(0.03 * sr)), 900, sr) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr)) * 0.05

    def body(y):
        yk = F.bandpass(y, k * freq, sr, 30.0)
        y = y + beat0 * np.sqrt(np.mean(yk ** 2)) / (np.sqrt(np.mean(beat0 ** 2)) + 1e-12)
        if y4 is not None:
            y = y + float(octave) * y4
        y[n_off:n_off + len(knock)] += knock[:max(0, n - n_off)]
        if soundboard > 0:
            wet = fdn_reverb(y, sr, t60_low=6.0, t60_high=2.5, size=12.0, predelay=0.0, mix=1.0, lines=8, tail=0.0)
            wet = wet.mean(axis=1) if wet.ndim == 2 else wet
            wet = sosfilt(cheby1(5, 5.0, 350.0, "highpass", fs=sr, output="sos"), wet[:n])   # their tone corrector
            y = y + float(soundboard) * wet * np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-12)
        # Meyer 2009, 3.5 and 4: dip 2000-2500 Hz, formant 4-5 kHz; lid resonance at 120 Hz, as strong as the soundboard's.
        # the depths and Qs. Their tone corrector is also a lowpass at high frequencies (4.5); ceiling 5 kHz
        if lid > 0:
            y = F.peak(y, 120.0, sr, 3.0, 6.0 * float(np.clip(lid, 0.0, 1.0)))
        y = F.peak(F.peak(y, 2250.0, sr, 2.5, -4.0), min(4500.0, 0.4 * sr), sr, 1.5, 4.0)
        if phantom > 0:
            y = y + float(phantom) * phantom_partials(y, sr, harpsichord_long_freq(freq))
        return F.lowpass(y, min(5000.0, 0.42 * sr), sr, 0.7, order=2)

    rel = math.hypot(*plectrum_release(beta)[1:]) / math.hypot(*plectrum_release(round(harpsichord_pluck_point(freq), 3))[1:])
    one = body(y1.copy())
    g = 0.8 * rel / (np.max(np.abs(one)) + 1e-9)
    if int(stops) < 2:
        return one * g
    d = int(0.0007 * sr)                                            # 1.5 cents apart, the plectra 0.7 ms apart
    two = body(y1 + _harpsi_string(freq * 2 ** (1.5 / 1200.0), n, n_off, sr, np.concatenate([np.zeros(d), exc]), t60, B,
                                   ripple_partial=ripple_partial))
    # the second rank adds 2-3 dB of level: set on the RMS, as the plectra 0.7 ms apart thin the summed attack peak
    return two * g * 1.3 * np.sqrt(np.mean(one ** 2) / (np.mean(two ** 2) + 1e-30))


@instrument("clavichord", "Clavichord (Valimaki, Laurson & Erkut 2003): two strings of a course coupled one way, a sharp tangent knock, "
            "quiet and short; `bebung` is the player's vibrato through the key.",
            family="keys", span=("C2", "D6"), bebung=(0.0, "0..1 pressure vibrato, 1 = 2 Hz wide at 7 Hz"))
def clavichord(freq, dur, sr=DEFAULT_SR, vel=1.0, bebung=0.0):
    """Commuted waveguide clavichord: two coupled strings, tangent knock, key-pressure glide; ``bebung`` = vibrato 0..1."""
    v = float(np.clip(vel, 0.0, 1.0))
    ring = dur + 0.1
    n, n_off = _n(ring, sr), int(round(dur * sr))
    midi = 69.0 + 12.0 * math.log2(max(freq, 1.0) / 440.0)
    k = float(np.clip(midi - 35.0, 1.0, 51.0))
    g = min(0.989 + 0.000189 * k, 0.9995)
    a = -0.514 + 0.0176 * k if k <= 23 else -0.175 + 0.00225 * k
    g_attack = 2.0 + 2.0 * (k - 1.0) / 50.0
    m = int(0.45 * sr)
    t = np.arange(m) / sr
    rng = np.random.default_rng(int(freq * 10) % 9973)
    exc = rng.standard_normal(m) * np.exp(-t / 0.004) + 3.0 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 90.0 * t) \
        + 2.0 * np.exp(-t / 0.03) * np.sin(2 * np.pi * 340.0 * t)
    exc = exc * np.cos(0.5 * np.pi * t / t[-1]) ** 2 * (0.4 + 0.6 * v)              # right half of a Hanning window
    t60 = -3.0 / (freq * math.log10(g))
    y1 = _harpsi_string(freq, n, n_off, sr, exc, t60, 1e-6, a=a, r=0.0, damp=0.1 / g)
    drive = np.pad(exc, (0, max(0, n - m)))[:n] + 0.05 * y1
    y2 = _harpsi_string(freq * 2 ** (1.0 / 1200.0), n, n_off, sr, drive, t60, 1e-6, a=a, r=0.0, damp=0.1 / g)
    y = 0.5 * (y1 + y2) + g_attack * 0.5 * np.pad(exc, (0, max(0, n - m)))[:n]
    tt = np.arange(n) / sr
    df = 0.01 * freq * v * np.maximum(1.0 - tt / 0.1, 0.0) + float(np.clip(bebung, 0.0, 1.0)) * np.minimum(tt / 0.2, 1.0) * np.sin(2 * np.pi * 7.0 * tt)
    y = np.interp(tt + np.cumsum(df / freq) / sr, tt, F.highpass(y, 40.0, sr))      # the loop passes DC; the key's pressure bends the string
    return _physical_note_fade(y / (np.max(np.abs(y)) + 1e-9) * (0.45 + 0.55 * v) * 1.213, sr, 0.0005, 0.02)  # 1.213: level match, belongs in _levels.py


@instrument("celesta", "Celesta: felt hammers on steel plates over wooden resonators; a soft, round bell (lullabies, fairy music).",
            family="mallet", span=("C4", "C8"))
def celesta(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.3 * _kt(freq, 1.0, 0.6, f_lo=523.0, f_hi=4186.0))
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.5), (2.0, 0.2, ring / 5), (5.1, 0.14 * vel, 0.04)], corner=3000.0)
    return x * perc(ring, 0.002, sr, curve=1.0) * 0.75 * (0.6 + 0.4 * vel)


@instrument("toy_piano", "Toy piano: plastic hammers on metal rods; plinky, childlike, a little out of true.",
            family="mallet", span=("C4", "C7"))
def toy_piano(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 0.7)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 0.3), (2.93, 0.3, 0.1), (6.27, 0.2 * vel, 0.03)], corner=3000.0)
    click = F.bandpass(O.white(n, seed=21), _cut(freq, 4, 1500, 3000), sr, 3.0) * decay_env(n, 0.004, sr) * 0.25 * vel
    return (x + click) * perc(ring, 0.001, sr, curve=1.0) * 0.75 * (0.6 + 0.4 * vel)


@instrument("church_organ", "Pipe organ: principal ranks at several octaves that speak slowly with a little chiff (add a long reverb).",
            family="keys", span=("C2", "C6"), sub=(0.35, "16-foot sub-octave rank 0..1"), mixture=(0.5, "upper ranks 0..1"),
            chiff=("classic", "classic | grey: onset noise around 2-4 x f0, 30 ms, 20 dB under the steady tone"))
def church_organ(freq, dur, sr=DEFAULT_SR, vel=1.0, sub=0.35, mixture=0.5, chiff="classic"):
    if chiff not in ("classic", "grey"):
        raise ValueError("church_organ chiff must be classic or grey")
    rel = 0.2
    n = _n(dur, sr, rel)
    x = np.zeros(n)
    for mult, g, tune in ((0.5, sub, 1.0003), (1, 1.0, 1.0), (2, 0.45, 0.9996), (3, 0.3 * mixture, 1.0004), (4, 0.4 * mixture, 1.0005)):
        f = freq * mult
        if g <= 0 or f > 5000:
            continue
        x += g * _harmonics(f * tune, n, sr, _spectrum(f, sr, tilt=1.7, corner=2400.0, slope=2.5, top=5000.0, maxh=10))
    if chiff == "grey":
        noise = grey_chiff(n, sr, freq, float(np.sqrt(np.mean((x / 1.6) ** 2))))
    else:
        noise = F.bandpass(O.white(n, seed=31), _cut(freq, 3, 600, 2400), sr, 2.0) * decay_env(n, 0.03, sr) * 0.12
    return apply(x / 1.6 + noise, adsr(dur, 0.05, 0.0, 1.0, rel, sr, curve=1.2)) * 0.6


CHIFF_BAND = (2.0, 4.0)
CHIFF_S = 0.03
CHIFF_DB = -20.0


def grey_chiff(n, sr, freq, steady_rms, length=CHIFF_S, level_db=CHIFF_DB, seed=31):
    """Onset noise of a flue pipe: noise in the band CHIFF_BAND x f0, rising in 2 ms and dying over `length` s,
    its RMS over `length` set `level_db` under `steady_rms`."""
    lo, hi = CHIFF_BAND[0] * freq, min(CHIFF_BAND[1] * freq, 0.45 * sr)
    if lo >= hi:
        return np.zeros(n)
    fc = math.sqrt(lo * hi)
    w = F.bandpass(O.white(n, seed=seed), fc, sr, fc / (hi - lo))
    t = np.arange(n) / sr
    w = w * np.minimum(t / 0.002, 1.0) * np.exp(-t / max(length, 1e-3))
    m = max(1, min(n, int(length * sr)))
    r = float(np.sqrt(np.mean(w[:m] ** 2))) + 1e-12
    return w * steady_rms * 10.0 ** (level_db / 20.0) / r


@instrument("accordion", "Accordion: two or three reeds tuned slightly apart (musette shimmer) under bellows pressure (folk, tango, polka, chanson).",
            family="keys", span=("C3", "C6"), musette=(9.0, "cents the reeds are tuned apart (0 = dry single reed)"))
def accordion(freq, dur, sr=DEFAULT_SR, vel=1.0, musette=9.0):
    rel = 0.08
    n = _n(dur, sr, rel)
    amps = _spectrum(freq, sr, tilt=1.0, formants=[(900, 400, 0.8)], corner=2200.0, slope=2.5, top=5500.0, even=0.7)
    reeds = ((0.0, 1.0), (musette, 0.7), (-0.8 * musette, 0.5)) if musette else ((0.0, 1.0),)
    x = sum(g * _harmonics(freq * 2 ** (c / 1200.0), n, sr, amps) for c, g in reeds) / 1.5
    t = np.arange(n) / sr
    x = x * (1 + 0.05 * np.sin(2 * np.pi * 0.9 * t))   # bellows
    return apply(x, adsr(dur, 0.045, 0.1, 0.9, rel, sr, curve=1.3)) * 0.6 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- mallets / bells / pitched drums
@instrument("xylophone", "Xylophone: hard mallets on short wooden bars; dry, bright knock (cartoons, puzzles).",
            family="mallet", span=("F4", "C8"))
def xylophone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 0.4 * _kt(freq, 1.2, 0.7, f_lo=350.0, f_hi=4186.0))
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 3), (3.0, 0.35 * vel, ring / 8), (6.0, 0.12 * vel, ring / 18)], corner=3000.0)
    click = F.bandpass(O.white(n, seed=22), _cut(freq, 3, 1200, 2600), sr, 1.0) * decay_env(n, 0.003, sr) * 0.18 * vel
    return (x + click) * perc(ring, 0.0008, sr, curve=1.0) * 0.8 * (0.6 + 0.4 * vel)


@instrument("glockenspiel", "Glockenspiel: small steel bars; a clear, sweet ring for high sparkle lines.",
            family="mallet", span=("G5", "C8"))
def glockenspiel(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.4 * _kt(freq, 1.0, 0.6, f_lo=784.0, f_hi=4186.0))
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.5), (2.76, 0.28 * vel, ring / 8), (5.4, 0.1 * vel, 0.05)], corner=3400.0, beat=0.8)
    return x * perc(ring, 0.001, sr, curve=1.0) * 0.7 * (0.6 + 0.4 * vel)


@instrument("handpan", "Handpan / hang drum: a hand-struck steel dome ringing its octave and fifth (ambient, meditative, world).",
            family="mallet", span=("C3", "C5"))
def handpan(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.8)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 1.0), (2.0, 0.5, 0.75), (3.0, 0.3 * vel, 0.5), (4.16, 0.06 * vel, 0.1)], corner=2500.0, beat=0.7)
    thump = F.lowpass(O.white(n, seed=23), 320, sr) * decay_env(n, 0.012, sr) * 0.6 * vel
    return (x + thump) * perc(ring, 0.003, sr, curve=1.0) * 0.75 * (0.6 + 0.4 * vel)


@instrument("tubular_bell", "Tubular bell / orchestral chime: a tall inharmonic clang whose partials imply the note (clock towers, Christmas, drama).",
            family="mallet", span=("C4", "G5"), decay=(3.5, "seconds"))
def tubular_bell(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=3.5):
    ring = max(dur, decay)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(0.615, 0.25, ring / 2.5), (1.0, 0.3, ring / 2.5), (1.21, 0.45, ring / 3), (2.0, 1.0, ring / 3.5),
                             (2.99, 0.8 * vel, ring / 5), (4.17, 0.5 * vel, ring / 8), (5.55, 0.25 * vel, ring / 12)],
               corner=3200.0, beat=0.5)
    return x * perc(ring, 0.002, sr, curve=1.2) * 0.5 * (0.6 + 0.4 * vel)


@instrument("gamelan", "Gamelan metallophone (saron/gender): bronze bars tuned in beating pairs, so every note shimmers.",
            family="mallet", span=("C4", "C7"), ombak=(3.0, "beat rate Hz between the paired bars"))
def gamelan(freq, dur, sr=DEFAULT_SR, vel=1.0, ombak=3.0):
    ring = max(dur, 2.0)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 1.3), (2.71, 0.4 * vel, 0.5), (5.15, 0.15 * vel, 0.12)], corner=3000.0, beat=ombak)
    return x * perc(ring, 0.001, sr, curve=1.0) * 0.7 * (0.6 + 0.4 * vel)


@instrument("singing_bowl", "Singing bowl: very long ring with slowly beating partials (meditation, ambient, temple).",
            family="mallet", span=("C3", "C5"), decay=(6.0, "seconds"))
def singing_bowl(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=6.0):
    ring = max(dur, decay)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.2), (2.71, 0.45, ring / 3.5), (5.0, 0.2 * vel, ring / 7)], corner=2600.0, beat=0.6)
    return x * perc(ring, 0.015, sr, curve=1.3) * 0.7 * (0.6 + 0.4 * vel)


@instrument("timpani", "Timpani / kettle drum: a pitched orchestral drum, boom and roll (write it low: C2-C4).",
            family="mallet", span=("C2", "C4"), decay=(1.6, "seconds"))
def timpani(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=1.6):
    ring = max(dur, decay)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.5), (1.5, 0.5, ring / 3.5), (1.99, 0.35, ring / 5), (2.44, 0.2 * vel, ring / 7),
                             (0.63, 0.5, 0.07)], corner=1500.0)
    thump = F.lowpass(O.white(n, seed=24), 260, sr) * decay_env(n, 0.02, sr) * 0.9 * vel
    return np.tanh((x + thump) * 1.2) * perc(ring, 0.002, sr, curve=1.0) * 0.7 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- plucked strings
# Modal guitar body: a parallel bank of g (1 - z^-2) / (1 - 2 r cos w z^-1 + r^2 z^-2) sections (Rau & Smith,
# ISMA 2019, eq. 3) on top of the direct string. Low modes in Woodhouse et al.'s bands (Acta Acustica 2012):
# the Helmholtz resonance at 75-100 Hz and the first top-plate mode at 150-250 Hz; between real guitars their
# frequencies spread by ~6 % (s.d.) and their Q by ~33 % (Fig. 7), against hearing thresholds of 1 % and
# 20 %, so a body ``seed`` draws a different instrument with that spread. ``size`` warps every mode through
# Penttinen, Harma & Karjalainen's first-order allpass map (DAFx-00): w' = w + 2 atan(l sin w / (1 - l cos w)),
# l < 0 a bigger box, l > 0 a smaller one. Centre values: the means over 15 guitars with the string damping
# compensated, i.e. the body alone (Woodhouse et al. 2012, Fig. 7, red open squares read off the plot):
# Helmholtz 91 Hz Q 52, top plate 192 Hz Q 44; the higher modes take Q 49, the mean of the 0-2 kHz band-averaged
# Q over the same guitars (Fig. 8c). the higher modes' spacing (every ~70 Hz up to 3 kHz) and level
# (falling as 1 / sqrt(f): "weaker and denser", Penttinen) and the size range l = +-0.12.
_GUITAR_LOW = ((91.0, 52.0, 1.0), (192.0, 44.0, 0.8))


GUITAR_VOLUME = 0.014


def guitar_uncoupled(f1: float = _GUITAR_LOW[0][0], f2: float = _GUITAR_LOW[1][0], volume: float = GUITAR_VOLUME,
                     top_mass: float = 0.2, top_area: float = 0.0814) -> tuple[float, float]:
    """(top Hz with the box closed, sound-hole Hz) whose coupling through the box air (`foley.guitar_body_modes`)
    gives the measured pair f1, f2: wp^2 + wh^2 = w1^2 + w2^2, wh^2 (wp^2 - wa^2) = w1^2 w2^2, the lower hole root."""
    from .foley import C_AIR, RHO_AIR
    a2 = RHO_AIR * C_AIR ** 2 / volume * top_area ** 2 / top_mass / (2 * math.pi) ** 2
    s, p = f1 * f1 + f2 * f2, (f1 * f2) ** 2
    h2 = 0.5 * ((s - a2) - math.sqrt((s - a2) ** 2 - 4.0 * p))
    return math.sqrt(s - h2), math.sqrt(h2)


def guitar_body_modes(seed: int = 0, size: float = 0.5, sr: int = DEFAULT_SR, steel: bool = False,
                      volume: float | None = None):
    """(freqs Hz, Q, gains) of a guitar body; seed 0 = the nominal body, others = Woodhouse's spread.
    ``volume`` (m^3): the two low modes become the top / sound-hole pair coupled through that box air, the hole's
    Helmholtz note going as 1 / sqrt(volume) (`guitar_uncoupled` at the nominal 0.014 m^3)."""
    rng = np.random.default_rng(int(seed) + 1000)
    f = [m[0] for m in _GUITAR_LOW]
    if volume:
        from .foley import guitar_body_modes as coupled
        top, hole = guitar_uncoupled()
        v = float(np.clip(volume, 0.002, 0.1))
        f[0], f[1] = coupled(top, hole * math.sqrt(GUITAR_VOLUME / v), v)
    q = [m[1] for m in _GUITAR_LOW]
    g = [m[2] for m in _GUITAR_LOW]
    hf = np.random.default_rng(7 if steel else 5)                   # the higher modes of the nominal body
    fk = 300.0
    while fk < 3000.0:
        fk += 70.0 * hf.uniform(0.5, 1.5)
        f.append(fk); q.append(49.0); g.append(0.35 * math.sqrt(300.0 / fk))
    f, q, g = np.array(f), np.array(q), np.array(g)
    if seed:
        f = f * np.exp(0.058 * rng.standard_normal(len(f)))
        q = q * np.exp(0.33 * rng.standard_normal(len(q)))
    lam = 0.24 * (0.5 - float(np.clip(size, 0.0, 1.0)))
    w = 2 * np.pi * f / 48000.0                                     # warped at Penttinen's 48 kHz whatever sr is
    w = w + 2 * np.arctan(lam * np.sin(w) / (1 - lam * np.cos(w)))
    return w * 48000.0 / (2 * np.pi), q, g


def guitar_body(y, sr, seed=0, size=0.5, steel=False, amount=0.6, radiation=False, air_q=None):
    """The string signal plus its modal body response (gain-matched, ``amount`` of the body). ``air_q``: Q of the
    Helmholtz (air) mode instead of the body-alone 52. ``radiation``: modes weighted by their radiation efficiency
    and the body output high-passed as a pulsating sphere (first order, 250 Hz)."""
    from scipy.signal import lfilter
    f, q, g = guitar_body_modes(seed, size, sr, steel)
    if air_q:
        q = q.copy()
        q[0] = float(air_q)
    if radiation:
        g = guitar_radiation(y, sr, f, g)
    out = np.zeros_like(y)
    for fk, qk, gk in zip(f, q, g):
        if fk >= 0.45 * sr:
            continue
        r = math.exp(-math.pi * fk / (qk * sr))
        out += gk * (1 - r) * lfilter([1.0, 0.0, -1.0], [1.0, -2 * r * math.cos(2 * math.pi * fk / sr), r * r], y)
    if radiation:
        out = F.highpass(out, 250.0, sr)
    out *= np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(out ** 2)) + 1e-12)
    return y + float(amount) * out


WLPC_ORDER = 24
WLPC_LAMBDA = 0.7


def warped_lpc(x, order=WLPC_ORDER, lam=WLPC_LAMBDA):
    """Warped all-pole fit: autocorrelation of x against its copies through k first-order allpasses
    D(z) = (z^-1 - lam) / (1 - lam z^-1), then Levinson. Returns (a with a[0] = 1, reflection coefficients)."""
    from scipy.signal import lfilter
    x = np.asarray(x, float)
    r = np.empty(order + 1)
    d = x.copy()
    for i in range(order + 1):
        r[i] = float(np.dot(x, d))
        d = lfilter([-lam, 1.0], [1.0, -lam], d)
    r[0] *= 1.0 + 1e-9
    a, ks, e = np.array([1.0]), np.zeros(order), r[0]
    for i in range(order):
        kk = -float(np.dot(a, r[i + 1:0:-1])) / e
        a = np.concatenate([a, [0.0]]) + kk * np.concatenate([a, [0.0]])[::-1]
        ks[i] = kk
        e *= 1.0 - kk * kk
    return a, ks


def lpc_from_reflection(ks):
    """Step-up recursion: reflection coefficients -> a (a[0] = 1); stable whenever every |k| < 1."""
    a = np.array([1.0])
    for kk in ks:
        a = np.concatenate([a, [0.0]]) + kk * np.concatenate([a, [0.0]])[::-1]
    return a


def warped_allpole(x, a, lam=WLPC_LAMBDA):
    """Filter x through 1 / A(D(z)) unrolled into an ordinary IIR: each root w of sum_k a_k w^k maps to the
    pole z = (1 + lam w) / (w + lam) (D(z) = w solved for z), and the p zeros sit at z = lam; run as
    second-order sections, DC gain 1 / A(1) as in the warped domain (D(1) = 1)."""
    from scipy.signal import sosfilt, zpk2sos
    a = np.asarray(a, float)
    p = len(a) - 1
    w = np.roots(a[::-1])
    poles = (1.0 + lam * w) / (w + lam)
    sos = zpk2sos(np.full(p, lam), poles, 1.0)
    g = 1.0
    for b0, b1, b2, _, a1, a2 in sos:
        g *= (b0 + b1 + b2) / (1.0 + a1 + a2)
    sos[0, :3] /= g * float(np.sum(a))
    return sosfilt(sos, np.asarray(x, float))


def _body_ir(sr, seed, size, steel):
    imp = np.zeros(int(0.25 * sr))
    imp[0] = 1.0
    y = guitar_body(imp, sr, seed, size, steel, amount=1.0)
    y[0] -= 1.0
    return y


@effect("body_morph", "Guitar body morph: the input is filtered by a warped all-pole model of a guitar body, interpolated "
        "between two bodies (seed / size) through their reflection coefficients, so every step is a stable, real body.",
        a=(0, "first body seed (0 = nominal)"), b=(1, "second body seed"), morph=(0.5, "0 = body a .. 1 = body b"),
        size_a=(0.5, "first body size 0..1"), size_b=(0.5, "second body size 0..1"), steel=(False, "steel-string body"),
        mix=(1.0, "wet amount 0..1"))
def body_morph(x, sr=DEFAULT_SR, a=0, b=1, morph=0.5, size_a=0.5, size_b=0.5, steel=False, mix=1.0):
    ka = warped_lpc(_body_ir(sr, int(a), float(size_a), bool(steel)))[1]
    kb = warped_lpc(_body_ir(sr, int(b), float(size_b), bool(steel)))[1]
    m = float(np.clip(morph, 0.0, 1.0))
    coef = lpc_from_reflection(np.clip((1.0 - m) * ka + m * kb, -0.999, 0.999))
    x = np.asarray(x, float)
    wet = np.apply_along_axis(warped_allpole, 0, x, coef)
    wet *= np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-12)
    mx = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - mx) * x + mx * wet


def _body(y, sr, peaks, lowpass=None):
    for fc, q, db in peaks:
        y = F.peak(y, fc, sr, q, db)
    return F.lowpass(y, lowpass, sr, 0.7) if lowpass else y


@instrument("steel_guitar", "Steel-string acoustic guitar: bright strum and fingerpicking (folk, country, pop).",
            family="plucked", span=("E2", "E5"))
def steel_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=4.5, bright=3800, damp=0.2, pick=0.16, grit=0.3, release=0.12, tail=1.6)
    return _body(y, sr, [(100, 1.2, 3.0), (200, 1.5, 2.0)], 5000)


@instrument("electric_guitar", "Clean electric guitar: round single-coil tone with long sustain (add `distortion` or `chorus` on the layer for rock or jangle).",
            family="plucked", span=("E2", "E6"), tone=(2800, "pickup tone Hz"))
def electric_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=2800):
    y = _string(freq, dur, sr, vel, t60=5.5, bright=3200, damp=0.25, pick=0.2, grit=0.15, release=0.08, tail=1.2)
    return F.lowpass(F.highpass(y, 70, sr), tone, sr, 1.3)


@instrument("muted_guitar", "Palm-muted electric guitar: short woody chug (funk scratch, reggae skank, rock eighths).",
            family="plucked", span=("E2", "E5"))
def muted_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, min(dur, 0.12), sr, vel, t60=0.4, bright=1900, damp=0.5, pick=0.2, grit=0.25, release=0.03, tail=0.2)
    return F.lowpass(F.highpass(y, 70, sr), 2200, sr, 1.2)


DIST_LOOP_OS = 2
DIST_LOOP_T60 = 7.0
DIST_LOOP_LP = 2600.0


@numba.njit(cache=True)
def _clip_loop(exc, n, L, frac, g, a, drive):
    size = L + 2
    buf = np.zeros(size)
    out = np.zeros(n)
    lp = 0.0
    w = 0
    for i in range(n):
        v = (1.0 - frac) * buf[(w - L) % size] + frac * buf[(w - L - 1) % size]
        lp = (1.0 - a) * v + a * lp
        u = drive * g * lp
        if u > 1.0:
            s = 2.0 / 3.0
        elif u < -1.0:
            s = -2.0 / 3.0
        else:
            s = u - u * u * u / 3.0
        y = s / drive
        out[i] = y
        buf[w] = y + (exc[i] if i < len(exc) else 0.0)
        w = (w + 1) % size
    return out


def dist_loop(freq, n, sr, drive, seed=0):
    """Plucked string with the clipper inside its loop: delay, one-pole loss, cubic soft clip f(u) = u - u^3 / 3
    (|f(x)| < |x|, so the loop stays stable), run at DIST_LOOP_OS x sr and brought back by a polyphase low-pass."""
    from scipy.signal import resample_poly
    fs = DIST_LOOP_OS * sr
    a = math.exp(-2 * math.pi * DIST_LOOP_LP / fs)
    P = fs / freq - a / (1.0 - a)
    L = max(2, int(P))
    exc = np.random.default_rng(seed).uniform(-1.0, 1.0, L)
    g = 10 ** (-3.0 / (DIST_LOOP_T60 * freq))
    y = _clip_loop(exc - exc.mean(), n * DIST_LOOP_OS, L, P - L, g, a, max(float(drive), 1e-3))
    return resample_poly(y, 1, DIST_LOOP_OS)[:n]


@instrument("dist_guitar", "Overdriven electric guitar for riffs and leads; sustains while held (for crunchy chords play `electric_guitar` and put `distortion` on the layer).",
            family="plucked", span=("E2", "E6"), drive=(6.0, "1 = edge of breakup .. 15 = fuzz"), tone=(3000, "cabinet tone Hz"),
            loop=(0, "1 = the drive inside the string loop at twice the rate (cubic clip), not after the string"))
def dist_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, drive=6.0, tone=3000, loop=0):
    rel = 0.12
    n = _n(dur, sr, rel)
    if loop:
        y = dist_loop(freq, n, sr, drive * (0.5 + 0.5 * vel))
        y = F.highpass(y / (np.max(np.abs(y)) + 1e-9), 120, sr)
    else:
        s = _ks(freq, n, sr, t60=7.0, bright=2600, damp=0.3, pick=0.18, grit=0.15)
        s = F.highpass(s / (np.max(np.abs(s)) + 1e-9), 120, sr)
        y = np.tanh(drive * (0.5 + 0.5 * vel) * s)
    y = F.peak(F.lowpass(F.highpass(y, 90, sr), tone, sr, 0.8, order=2), 1400, sr, 1.0, 3.0)
    return apply(y, adsr(dur, 0.003, 0.0, 1.0, rel, sr)) * 0.5 * (0.6 + 0.4 * vel)


@instrument("banjo", "Banjo: twangy string over a drum head, quick decay (bluegrass, country, folk).",
            family="plucked", span=("C3", "C6"))
def banjo(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=1.0, bright=4200, damp=0.12, pick=0.08, grit=0.3, release=0.06, tail=0.6)
    y = y + 0.6 * F.bandpass(y, 600, sr, 2.0)
    return F.lowpass(F.highpass(y, 150, sr), 4500, sr, 0.7) * 0.8


@instrument("mandolin", "Mandolin: paired steel strings; set `tremolo` to re-pick held notes (Italian, bluegrass, folk).",
            family="plucked", span=("G3", "E6"), tremolo=(0.0, "picks per second on held notes (12 = classic tremolo, 0 = off)"))
def mandolin(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.0):
    def one(d, v):
        a = _string(freq, d, sr, v, t60=1.4, bright=4000, damp=0.15, pick=0.12, grit=0.25, release=0.05, tail=0.7)
        b = _string(freq * 1.003, d, sr, v, t60=1.4, bright=4000, damp=0.15, pick=0.12, grit=0.25, release=0.05, tail=0.7, seed=7)
        return mix([(a, 0.0), (b, 0.004)], sr) * 0.6
    if tremolo > 0 and dur * tremolo >= 1.5:
        step = 1.0 / tremolo
        y = mix([(one(step, vel * (1.0 if i % 2 == 0 else 0.82)), i * step) for i in range(int(dur * tremolo))], sr)
    else:
        y = one(dur, vel)
    return F.lowpass(F.highpass(y, 150, sr), 4500, sr, 0.7)


@instrument("koto", "Koto: silk string plucked hard near the bridge with ivory picks (Japanese, East Asian).",
            family="plucked", span=("C3", "C6"))
def koto(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=2.6, bright=3600, damp=0.2, pick=0.07, grit=0.25, release=0.15, tail=1.4)
    return _body(y, sr, [(300, 1.5, 3.0)], 4200)


@instrument("sitar", "Sitar: long string over a buzzing bridge, the buzz blooms just after the pluck (Indian classical, psychedelic).",
            family="plucked", span=("C3", "C5"), buzz=(0.45, "0..1 amount of bridge buzz"))
def sitar(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.45):
    y = _string(freq, dur, sr, vel, t60=3.5, bright=3400, damp=0.1, pick=0.06, grit=0.3, release=0.2, tail=1.6)
    n = y.shape[0]
    t = np.arange(n) / sr
    b = F.peak(F.highpass(np.tanh(6.0 * y), 700, sr), 2400, sr, 1.2, 4.0) * (1 - np.exp(-t / 0.05)) * np.exp(-t / 0.9)
    return F.lowpass(y + buzz * b, 4200, sr, 0.7, order=2) * 0.8


@instrument("pizzicato", "Pizzicato strings: short, woody finger-plucked violin/cello notes (playful, sneaky, classical).",
            family="plucked", span=("C2", "C6"))
def pizzicato(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, min(dur, 0.25), sr, vel, t60=0.55, bright=2400, damp=0.45, pick=0.25, grit=0.15, release=0.08, tail=0.4)
    return _body(y, sr, [(290, 1.5, 3.0), (460, 1.5, 3.0), (1000, 1.0, 2.0)], 3200)


# ---------------------------------------------------------------- basses
@instrument("upright_bass", "Upright (double) bass, plucked: deep woody thump with a short singing tail (jazz walking lines, rockabilly, folk).",
            family="bass", span=("E1", "G3"))
def upright_bass(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=2.0, bright=850, damp=0.5, pick=0.3, grit=0.1, release=0.1, tail=0.8)
    n = y.shape[0]
    thump = F.lowpass(O.white(n, seed=41), 300, sr) * decay_env(n, 0.012, sr) * 0.5 * vel
    return _body(y + thump, sr, [(100, 1.2, 3.0), (210, 1.5, 2.0)], 1400)


@instrument("finger_bass", "Electric bass guitar, fingerstyle: round, even, sustaining (rock, pop, soul, reggae).",
            family="bass", span=("E1", "G3"), tone=(1700, "pickup tone Hz"))
def finger_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=1700):
    y = _string(freq, dur, sr, vel, t60=4.0, bright=1500, damp=0.4, pick=0.28, grit=0.1, release=0.07, tail=0.6)
    return F.lowpass(np.tanh(1.5 * y) / 0.9, tone, sr, 1.0)


@instrument("slap_bass", "Slapped electric bass: percussive pop on top of a scooped, punchy low end (funk, disco).",
            family="bass", span=("E1", "G3"))
def slap_bass(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=2.6, bright=4200, damp=0.15, pick=0.1, grit=0.3, release=0.05, tail=0.5)
    n = y.shape[0]
    pop = F.bandpass(O.white(n, seed=42), 2300, sr, 1.5) * decay_env(n, 0.004, sr) * 0.3 * vel
    y = F.peak(y + pop, 500, sr, 1.0, -5.0)
    return F.lowpass(F.highpass(y, 35, sr), 4000, sr, 0.7)


# ---------------------------------------------------------------- bowed / blown
def _blown(freq, dur, sr, vel, soft, hard, attack=0.05, release=0.1, bloom=0.06, vib=(5.5, 0.1, 0.2), scoop=0.0,
           sustain=0.9, top=6500.0, even_db=0.0, track=False):
    """Sustained harmonic voice. `soft` / `hard` are `_spectrum` kwargs for the quiet and the loud
    spectrum; the tone opens from soft towards hard as the note speaks (`bloom` seconds) and with
    velocity, which is what makes bowed strings and brass sound played rather than switched on."""
    n = _n(dur, sr, release)
    f = _vibrato(freq, n, sr, rate=vib[0], depth=vib[1], delay=vib[2], ramp=0.3, scoop=scoop, end_boost=2.0)
    a0 = _spectrum(freq, sr, top=top, **soft)
    a1 = _spectrum(freq, sr, top=top, **hard)
    K = max(len(a0), len(a1))
    a0 = np.pad(a0, (0, K - len(a0)))
    a1 = np.pad(a1, (0, K - len(a1)))
    t = np.arange(n) / sr
    b = (0.25 + 0.75 * vel) * (1 - np.exp(-t / max(bloom, 1e-3))) * (0.8 + 0.2 * np.exp(-t / 0.6))
    w = 2 * np.pi * O.phase(f, n, sr)
    y = np.zeros(n)
    body = soft.get("formants", ()) if track else ()
    for k in range(1, K + 1):
        lo, hi = a0[k - 1], a1[k - 1]
        if even_db and k % 2 == 0:
            lo, hi = lo * 10.0 ** (even_db / 20.0), hi * 10.0 ** (even_db / 20.0)
        if max(lo, hi) > 1e-4:
            if body:
                y += (lo + b * (hi - lo)) * _body_gain(k * f, body) / _body_gain(k * freq, body) * np.sin(k * w)
            else:
                y += (lo + b * (hi - lo)) * np.sin(k * w)
    return apply(y, adsr(dur, attack, 0.15, sustain, release, sr, curve=1.5)) * (0.5 + 0.5 * vel)


def _body_gain(fk, formants):
    """The `_spectrum` body curve 1 + sum g exp(-((f - fc) / bw)^2 / 2) at ``fk`` (scalar or per sample)."""
    g = 1.0
    for fc, bw, a in formants:
        g = g + a * np.exp(-0.5 * ((fk - fc) / bw) ** 2)
    return g


def _air(n, sr, center, q, seed):
    return F.bandpass(O.white(n, seed=seed), center, sr, q)


# Meyer 2009, 3.3: violin bands near 1600 and 3000-4000 Hz. every width and gain here
VIOLIN_BODY = [(290, 60, 1.2), (480, 100, 1.5), (1100, 350, 0.8), (1600, 300, 0.5), (2500, 500, 1.0), (3500, 800, 0.5)]
CELLO_BODY = [(180, 40, 1.0), (300, 70, 1.4), (550, 150, 0.8), (1500, 400, 0.5)]


@instrument("violin", "Solo violin, bowed: singing tone with delayed vibrato (classical, folk fiddle, tango, film).",
            family="bowed", span=("G3", "C7"), attack=(0.07, "s; 0.02 = bitten accent, 0.2 = swell"), vibrato=(1.0, "0 = none .. 2 = wide"))
def violin(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.07, vibrato=1.0):
    y = _blown(freq, dur, sr, vel, dict(tilt=1.7, formants=VIOLIN_BODY, corner=2000.0, slope=3.0),
               dict(tilt=1.1, formants=VIOLIN_BODY, corner=3000.0, slope=3.0),
               attack=attack, release=0.12, bloom=attack, vib=(5.8, 0.25 * vibrato, 0.12))
    n = y.shape[0]
    rosin = _air(n, sr, 2200, 1.0, 51) * 0.012 * np.minimum(1.0, np.abs(y) * 4)
    return (y + rosin) * 0.6


@instrument("cello", "Solo cello, bowed: dark, warm, vocal (laments, bass lines for chamber and film cues).",
            family="bowed", span=("C2", "C5"), attack=(0.1, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def cello(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.1, vibrato=1.0):
    y = _blown(freq, dur, sr, vel, dict(tilt=1.7, formants=CELLO_BODY, corner=1300.0, slope=3.0),
               dict(tilt=1.15, formants=CELLO_BODY, corner=1900.0, slope=3.0),
               attack=attack, release=0.15, bloom=attack, vib=(5.0, 0.15 * vibrato, 0.15), top=5000.0)
    n = y.shape[0]
    rosin = _air(n, sr, 1500, 1.0, 52) * 0.012 * np.minimum(1.0, np.abs(y) * 4)
    return (y + rosin) * 0.6


@instrument("clarinet", "Clarinet: hollow, woody, odd-harmonic tone (klezmer, jazz, classical, cartoons).",
            family="wind", span=("D3", "C6"))
def clarinet(freq, dur, sr=DEFAULT_SR, vel=1.0):
    even = _kt(freq, 0.08, 0.5)   # the upper register loses the hollow sound
    y = _blown(freq, dur, sr, vel, dict(tilt=1.7, even=even, corner=1500.0, slope=3.0),
               dict(tilt=1.2, even=even, formants=[(1500, 500, 0.5)], corner=2300.0, slope=3.0),
               attack=0.04, release=0.08, bloom=0.05, vib=(5.0, 0.03, 0.3))
    n = y.shape[0]
    return (y + _air(n, sr, _cut(freq, 3, 900, 2000), 1.5, 53) * 0.02 * np.abs(y)) * 0.6


@instrument("oboe", "Oboe: reedy, nasal, plaintive (pastoral, baroque, Middle-Eastern flavoured lines).",
            family="wind", span=("Bb3", "G6"), meyer=(0, "1 = attack by register and velocity, dynamic timbre above 3 kHz (Meyer 2009)"),
            even=(0.0, "dB on the even harmonics (negative = hollower, clarinet-like)"),
            precursor=(0.0, "0..1 the burst of high energy before the tone settles"))
def oboe(freq, dur, sr=DEFAULT_SR, vel=1.0, meyer=0, even=0.0, precursor=0.0):
    body = [(1150, 250, 2.5), (2700, 400, 0.8)]
    y = _blown(freq, dur, sr, vel, dict(tilt=1.1, formants=body, corner=2200.0, slope=3.5),
               dict(tilt=0.7, formants=body, corner=4000.0, slope=3.5),
               attack=attack_ms("oboe", freq, vel) / 1000.0 if meyer else 0.03, release=0.08, bloom=0.04, vib=(5.5, 0.1, 0.25),
               even_db=float(even))
    if meyer:
        y = dynamic_timbre(y, sr, "oboe", vel)
    return (_precursor(y, sr, "oboe", precursor) if precursor > 0 else y) * 0.55


@instrument("sax", "Saxophone (alto/tenor): breathy, husky, scoops into notes (jazz, soul, noir, 80s pop).",
            family="wind", span=("Bb2", "F5"), breath=(0.1, "air noise 0..1"), growl=(0.0, "0..1 rasp"),
            model=("formant", "formant | stk: Scavone's conical-bore waveguide (STK Saxofony)"), position=(0.2, "stk: where the reed sits on the bore, 0..0.5"),
            lattice=("", "formant: soprano | tenor, the tone-hole lattice cutoff sets the spectral corner ('' = the alto voice)"))
def sax(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.1, growl=0.0, model="formant", position=0.2, lattice=""):
    if model == "stk":
        import numpy as _np
        from .physics import _to_sr
        from .physical import waveguides as _wg
        release = 0.1
        n = int((dur + release) * _wg.SR) + 1
        t = _np.arange(n) / _wg.SR
        p = (0.55 + 0.3 * float(_np.clip(vel, 0.0, 1.0))) * _np.clip(t / 0.03, 0.0, 1.0) * _np.clip((dur + release - t) / release, 0.0, 1.0)
        y = _to_sr(_wg.saxofony(_np.full(n, float(freq)), p, dur + release, _np.random.default_rng(int(freq)), float(_np.clip(position, 0.02, 0.5)),
                                noise_gain=0.2 * float(breath) / 0.1), sr)[:_n(dur + release, sr)]
        return F.dc_block(y, sr) / (_np.max(_np.abs(y)) + 1e-12) * 0.6
    body = [(550, 180, 1.2), (1300, 400, 1.0), (2300, 500, 0.4)]
    from .physical.winds import SAX_LATTICE_FC
    c0, c1 = (SAX_LATTICE_FC[lattice], SAX_LATTICE_FC[lattice] * SAX_LATTICE_HARD) if lattice else (1700.0, 2600.0)
    y = _blown(freq, dur, sr, vel, dict(tilt=1.5, formants=body, corner=c0, slope=3.0),
               dict(tilt=0.9, formants=body, corner=c1, slope=3.0),
               attack=0.035, release=0.1, bloom=0.07, vib=(5.0, 0.2, 0.2), scoop=0.5)
    n = y.shape[0]
    t = np.arange(n) / sr
    if growl > 0:
        y = y * (1 - 0.5 * growl * (1 + np.sin(2 * np.pi * 28 * t)) / 2)
    air = _air(n, sr, 1600, 0.8, 54) * breath * 0.25 * np.minimum(1.0, np.abs(y) * 3 + 0.2)
    return (y + apply(air, adsr(dur, 0.02, 0.1, 0.6, 0.1, sr))[:n]) * 0.6


TRUMPET_BODY = [(1200, 400, 1.2), (2300, 500, 0.6)]
# Meyer 2009, ch. 3 (lots B07 and Addendum c00): attack in ms, dry and soft, low to high register of the span.
# Not re-read in the book: the chunks only. a soft column reused where Meyer gives one value, horn soft = dry.
ATTACK_MS = {"trumpet": ((27.5, 27.5, 20.0), (180.0, 150.0, 40.0), (165.0, 1047.0)),
             "french_horn": ((35.0, 35.0, 35.0), (35.0, 35.0, 35.0), (65.4, 698.0)),
             "trombone": ((40.0, 40.0, 20.0), (70.0, 70.0, 70.0), (82.4, 523.0)),
             "tuba": ((60.0, 40.0, 25.0), (130.0, 130.0, 60.0), (32.7, 262.0)),
             "oboe": ((40.0, 40.0, 20.0), (100.0, 100.0, 40.0), (233.0, 1568.0)),
             "flute": ((30.0, 25.0, 20.0), (180.0, 150.0, 120.0), (262.0, 2093.0)),
             "clarinet": ((20.0, 20.0, 15.0), (100.0, 80.0, 60.0), (147.0, 1568.0)),
             "bassoon": ((40.0, 30.0, 20.0), (120.0, 100.0, 60.0), (58.3, 523.0))}
RELEASE_S = {"french_horn": 0.15, "flute": 0.12}      # Meyer: horn 150 ms; the other brass have no figure
# Meyer 2009 3.1.2.3 (lot B07): the preliminary peak of the attack, (centre s, length s). Trumpet peak at 10-15 ms for
# 5 ms; horn about 20 ms long from 10-30 ms; trombone about 25 ms. 40 % per unit, the trombone's centre.
PRECURSOR = {"trumpet": (0.0125, 0.005), "french_horn": (0.02, 0.02), "trombone": (0.025, 0.025),
             "oboe": (0.01, 0.01), "clarinet": (0.015, 0.01), "bassoon": (0.015, 0.015)}
SAX_LATTICE_HARD = 1.5
VIB_RATE = 5.2
FLUTE_VIB_HZ = 5.9
JET_REF_HZ = 523.25


def dynamic_timbre(y, sr, inst, vel):
    """Dynamic timbre: the band above 3 kHz moves (factor - 1) dB for every dB the overall level moves with vel
    (level 0.5 + 0.5 vel), factor the mean of `winds.DYNAMIC_FACTOR[inst]`."""
    from .physical.winds import DYNAMIC_FACTOR
    fac = sum(DYNAMIC_FACTOR[inst]) / 2.0
    g = (fac - 1.0) * 20.0 * math.log10(0.5 + 0.5 * float(np.clip(vel, 0.0, 1.0)))
    return F.highshelf(y, 3000.0, sr, g) if abs(g) > 0.01 else y


def band_vibrato_am(y, sr, inst, rate=VIB_RATE, amount=1.0, onset=0.3):
    """The amplitude modulation a pitch vibrato makes in each band (`winds.VIBRATO_AM`, dB peak to peak), at the
    vibrato rate, faded in over `onset` s."""
    from .physical.winds import VIBRATO_AM
    t = np.arange(y.size) / sr
    m = np.sin(2 * np.pi * rate * t) * np.clip(t / max(onset, 1e-3), 0.0, 1.0)
    out = np.array(y, float)
    for lo, hi, db in VIBRATO_AM[inst]:
        if db <= 0 or lo >= 0.45 * sr:
            continue
        hi = min(hi, 0.45 * sr)
        band = F.highpass(y, lo, sr, order=2) if lo > 0 else y
        band = F.lowpass(band, hi, sr, 0.707) if hi < 0.45 * sr else band
        out = out + band * (10.0 ** (float(amount) * db / 2.0 * m / 20.0) - 1.0)
    return out


def attack_ms(inst, freq, vel):
    """Attack time in ms, interpolated in log f over the three registers and between soft (vel 0) and dry (vel 1)."""
    dry, soft, (lo, hi) = ATTACK_MS[inst]
    x = 2.0 * float(np.clip(math.log(freq / lo) / math.log(hi / lo), 0.0, 1.0))
    v = float(np.clip(vel, 0.0, 1.0))
    return float(np.interp(x, (0, 1, 2), soft) * (1.0 - v) + np.interp(x, (0, 1, 2), dry) * v)


@instrument("trumpet", "Solo trumpet: brightens as it is blown harder; `mute` gives the thin jazz harmon-mute tone (fanfares, mariachi, jazz, ska).",
            family="brass", span=("E3", "C6"), mute=(0.0, "0 = open .. 1 = harmon mute"))
def trumpet(freq, dur, sr=DEFAULT_SR, vel=1.0, mute=0.0):
    y = _blown(freq, dur, sr, vel, dict(tilt=2.2, formants=TRUMPET_BODY, corner=1800.0, slope=3.0),
               dict(tilt=0.9, formants=TRUMPET_BODY, corner=3200.0, slope=3.0),
               attack=0.03, release=0.08, bloom=0.05, vib=(5.5, 0.1, 0.3))
    if mute > 0:
        m = F.peak(F.highpass(y, 700, sr, order=2), 1800, sr, 1.5, 8.0)
        y = (1 - mute) * y + mute * F.lowpass(m, 3200, sr)
    return y * 0.6


@instrument("french_horn", "French horn: round, noble, distant (film scores, heroic themes, warm pads of brass).",
            family="brass", span=("C2", "F5"))
def french_horn(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(450, 150, 1.5)]
    return _blown(freq, dur, sr, vel, dict(tilt=2.6, formants=body, corner=1000.0, slope=3.0),
                  dict(tilt=1.6, formants=body, corner=1600.0, slope=3.0),
                  attack=0.08, release=0.15, bloom=0.1, vib=(5.0, 0.03, 0.4), top=4500.0) * 0.6


@instrument("trombone", "Trombone: broad, brassy low-mid voice (big band, ska, funeral marches, New Orleans).",
            family="brass", span=("E2", "C5"))
def trombone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(520, 200, 1.2), (1500, 400, 0.4)]
    return _blown(freq, dur, sr, vel, dict(tilt=2.2, formants=body, corner=1400.0, slope=3.0),
                  dict(tilt=1.2, formants=body, corner=2300.0, slope=3.0),
                  attack=0.05, release=0.1, bloom=0.07, vib=(5.0, 0.05, 0.35), top=5000.0) * 0.6


@instrument("tuba", "Tuba: fat, puffing bottom of the brass band (oompah, polka, marches, circus).",
            family="brass", span=("C1", "C4"))
def tuba(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(230, 90, 1.2)]
    y = _blown(freq, dur, sr, vel, dict(tilt=2.6, formants=body, corner=600.0, slope=3.0),
               dict(tilt=1.7, formants=body, corner=950.0, slope=3.0),
               attack=0.05, release=0.1, bloom=0.06, vib=(5.0, 0.0, 0.5), top=3000.0)
    n = y.shape[0]
    puff = F.lowpass(O.white(n, seed=55), 400, sr) * decay_env(n, 0.03, sr) * 0.15 * vel
    return (y + puff) * 0.65


@instrument("harmonica", "Harmonica (blues harp): reedy, bends into notes, hand-cupped wah (blues, country, western).",
            family="wind", span=("C4", "C7"))
def harmonica(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(1300, 350, 1.5), (2500, 400, 0.6)]
    y = _blown(freq, dur, sr, vel, dict(tilt=1.3, formants=body, corner=2000.0, slope=3.0, even=0.9),
               dict(tilt=0.9, formants=body, corner=2800.0, slope=3.0, even=0.9),
               attack=0.03, release=0.08, bloom=0.05, vib=(5.5, 0.18, 0.25), scoop=0.6)
    n = y.shape[0]
    t = np.arange(n) / sr
    wah = 1 - 0.25 * np.clip((t - 0.25) / 0.3, 0, 1) * (1 + np.sin(2 * np.pi * 5.5 * t)) / 2
    return y * wah * 0.55


@instrument("pan_flute", "Pan flute / bamboo flute: breathy, chiffy, hollow (Andean, new age, calm exploration).",
            family="wind", span=("C4", "C7"), breath=(0.3, "air noise 0..1"))
def pan_flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.3):
    rel = 0.15
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=4.5, depth=0.06, delay=0.3, ramp=0.4, scoop=0.4)
    amps = np.array([1.0, 0.06, 0.18, 0.02, 0.05]) * _roll(np.arange(1, 6) * freq, 2400.0) / float(_roll(freq, 2400.0))
    x = _harmonics(f, n, sr, amps)
    air = _air(n, sr, min(freq, 2400.0), 6.0, 56) * breath * 0.8 + _air(n, sr, 1600, 0.8, 57) * breath * 0.12 * _kt(freq, 1.0, 0.5)
    chiff = _air(n, sr, _cut(freq, 2, 800, 2400), 1.2, 58) * decay_env(n, 0.035, sr) * breath * 1.2
    return apply(x + air + chiff, adsr(dur, 0.05, 0.1, 0.8, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)


@instrument("whistle", "Human whistle: pure tone with a little air, slides into each note (westerns, carefree tunes).",
            family="wind", span=("C5", "C7"))
def whistle(freq, dur, sr=DEFAULT_SR, vel=1.0):
    rel = 0.1
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.8, depth=0.12, delay=0.2, ramp=0.3, scoop=0.7)
    x = O.sine(f, n, sr) + 0.04 * O.sine(f * 2, n, sr)
    air = _air(n, sr, min(freq, 2600.0), 12.0, 59) * 0.35
    return apply(x + air, adsr(dur, 0.04, 0.1, 0.85, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)


@instrument("ocarina", "Ocarina: round clay flute, almost a sine with a soft chiff (adventure themes, lullabies).",
            family="wind", span=("A4", "F6"), vibrato=(0.3, "0 = none .. 1 = wide"))
def ocarina(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.3):
    rel = 0.1
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.2, depth=0.2 * vibrato, delay=0.2, ramp=0.3, scoop=0.25)
    amps = np.array([1.0, 0.05, 0.09]) * _roll(np.arange(1, 4) * freq, 2400.0) / float(_roll(freq, 2400.0))
    x = _harmonics(f, n, sr, amps)
    chiff = _air(n, sr, _cut(freq, 2, 800, 2200), 1.5, 60) * decay_env(n, 0.025, sr) * 0.25
    air = _air(n, sr, min(freq, 2400.0), 5.0, 61) * 0.06
    return apply(x + chiff + air, adsr(dur, 0.025, 0.1, 0.9, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)


@instrument("didgeridoo", "Didgeridoo: low drone whose vowel slowly shifts (hold one long low note: C2-G2).",
            family="wind", span=("A1", "G2"), wow=(1.4, "vowel movement Hz"))
def didgeridoo(freq, dur, sr=DEFAULT_SR, vel=1.0, wow=1.4):
    rel = 0.2
    n = _n(dur, sr, rel)
    t = np.arange(n) / sr
    f = freq * (1 + 0.003 * np.sin(2 * np.pi * 0.7 * t))
    a0 = _spectrum(freq, sr, tilt=0.8, formants=[(350, 120, 2.5), (1100, 250, 1.0)], corner=1800.0, slope=3.0, top=4000.0)
    a1 = _spectrum(freq, sr, tilt=0.8, formants=[(650, 150, 2.5), (1700, 300, 1.2)], corner=1800.0, slope=3.0, top=4000.0)
    b = 0.5 + 0.5 * np.sin(2 * np.pi * wow * t)
    w = 2 * np.pi * O.phase(f, n, sr)
    y = np.zeros(n)
    for k in range(1, len(a0) + 1):
        y += (a0[k - 1] + b * (a1[k - 1] - a0[k - 1])) * np.sin(k * w)
    air = F.lowpass(O.white(n, seed=62), 1000, sr) * 0.05
    return apply(y + air, adsr(dur, 0.08, 0.1, 0.95, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)

# -----------------------------------------------------------------------------
# Physical-engine overrides (v0.5)
#
# These re-register the legacy high-level names on top of the physically informed
# waveguide kernels.  The original recipe implementations above are intentionally
# kept as readable fallbacks/reference material; the registry now points here.

def _physical_note_fade(y, sr, attack=0.01, release=0.03):
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0:
        return y
    a = min(y.size, max(1, int(round(attack * sr))))
    r = min(y.size, max(1, int(round(release * sr))))
    y[:a] *= np.linspace(0.0, 1.0, a)
    y[-r:] *= np.linspace(1.0, 0.0, r)
    return y


@instrument("violin", "Physically modelled bowed violin: bow/string waveguide with body coupling and rosin noise.",
            family="bowed", span=("G3", "C7"), attack=(0.07, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def violin(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.07, vibrato=1.0):
    from .physics import BowedString
    m = BowedString(body="violin", bow_pressure=float(np.clip(0.55 + 0.4 * vel, 0, 1)),
                    bow_velocity=0.13 + 0.14 * vel, vibrato=vibrato > 0.05,
                    bow_noise=0.035 + 0.025 * vel)
    y = m.note(freq, dur + 0.10, sr=sr, seed=51)
    y = F.lowpass(y, min(5200.0, 0.44 * sr), sr, 0.7)
    return _physical_note_fade(y, sr, attack=max(0.005, attack), release=0.10) * (0.52 + 0.28 * vel)


@instrument("cello", "Physically modelled bowed cello: bowed-string waveguide with dark body coupling.",
            family="bowed", span=("C2", "C5"), attack=(0.1, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def cello(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.1, vibrato=1.0):
    from .physics import BowedString
    m = BowedString(body="cello", bow_pressure=float(np.clip(0.58 + 0.36 * vel, 0, 1)),
                    bow_velocity=0.11 + 0.12 * vel, vibrato=vibrato > 0.05,
                    bow_noise=0.028 + 0.022 * vel)
    y = m.note(freq, dur + 0.13, sr=sr, seed=52)
    y = F.lowpass(y, min(3900.0, 0.44 * sr), sr, 0.75)
    return _physical_note_fade(y, sr, attack=max(0.008, attack), release=0.13) * (0.48 + 0.27 * vel)


@instrument("clarinet", "Physical clarinet: nonlinear reed coupled to a bore waveguide.", family="wind", span=("D3", "C6"),
            model=("waveguide", "waveguide | impedance (Guillemain 2005 digital impedance, measured reed and mouthpiece: "
                                "the tone thickens and brightens with velocity)"),
            embouchure=(0.5, "impedance only: 0 loose .. 1 tight lip on the reed"),
            reed=("guillemain", "impedance only: guillemain | typical | squeak | van_walstijn"),
            stiffening=(0, "impedance with reed=van_walstijn: 1 = stiffness rises with pressure (van Walstijn & Avanzini 2007)"),
            lip_force=(0.0, "impedance only: N of lip force, lowers the closing pressure (Almeida 2013: 0.6 N = 20 mbar)"),
            meyer=(0, "waveguide: 1 = attack by register and velocity, dynamic timbre above 3 kHz (Meyer 2009)"),
            precursor=(0.0, "waveguide: 0..1 the burst of high energy before the tone settles"))
def clarinet(freq, dur, sr=DEFAULT_SR, vel=1.0, model="waveguide", embouchure=0.5, reed="guillemain", stiffening=0, lip_force=0.0,
             meyer=0, precursor=0.0):
    if model == "impedance":
        from .physical import impedance
        y = impedance.clarinet(freq, dur, float(sr), blow=vel, embouchure=embouchure, reed=reed, stiffening=bool(stiffening), lip_force=lip_force)
        y = F.lowpass(y, min(3600.0, 0.44 * sr), sr, 0.8)   # p_ext = d(p + u)/dt overstates the highs (Guillemain)
        return y / (np.max(np.abs(y)) + 1e-12) * (0.6 + 0.4 * vel)   # as loud as the waveguide clarinet
    from .physics import Clarinet
    m = Clarinet(breath=0.42 + 0.42 * vel, noise=0.04 + 0.10 * vel)
    y = m.note(freq, dur + 0.07, sr=sr, seed=53)
    y = F.lowpass(y, min(3600.0, 0.44 * sr), sr, 0.8)   # 3600: the band-limited resampler no longer dulls the 48 kHz kernel for us
    y = _physical_note_fade(y, sr, attack=attack_ms("clarinet", freq, vel) / 1000.0 if meyer else 0.035, release=0.07)
    if meyer:
        y = dynamic_timbre(y, sr, "clarinet", vel)
    return (_precursor(y, sr, "clarinet", precursor) if precursor > 0 else y) * 0.95


@instrument("flute", "Physical jet-driven flute/recorder waveguide with breath turbulence.", family="wind", span=("C4", "C7"), breath=(0.15, "noise amount"))
def flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.15):
    from .physics import Flute as PhysicalFlute
    # jet_ratio 0.32 of STK's 1.5-period bore = a jet delay of 0.48 period: Coltman's half-period optimum, and
    # de la Cuadra's measured soft playing (thesis 2005, 3.5: Str_W = fW/Uj 0.23 soft .. 0.14 loud, convection
    # 0.5 Uj from the lips: 0.46 .. 0.28 period). Below about 0.25 this model leaves its register (-300 cents), so
    # the louder end is not followed.
    m = PhysicalFlute(model="flute", pressure=0.50 + 0.42 * vel,
                      noise=max(0.01, float(breath)), jet_ratio=0.32,
                      vibrato=0.02 + 0.04 * vel)
    y = m.note(freq, dur + 0.10, sr=sr, seed=5)
    y = F.lowpass(y, min(4200.0, 0.44 * sr), sr, 0.8)
    return _physical_note_fade(y, sr, attack=0.055, release=0.10) * 5.2


@instrument("trumpet", "Physical lip-reed brass model; brightens naturally with blowing pressure.",
            family="brass", span=("E3", "C6"), mute=(0.0, "0 = open .. 1 = harmon mute"))
def trumpet(freq, dur, sr=DEFAULT_SR, vel=1.0, mute=0.0):
    from .physics import Brass
    m = Brass(pressure=0.40 + 0.48 * vel, lip_tension=0.48 + 0.08 * np.log2(max(freq, 80.0) / 261.63))
    y = m.note(freq, dur + 0.08, sr=sr, seed=56)
    y = F.lowpass(y, min(5200.0, 0.44 * sr), sr, 0.7)
    if mute > 0:
        mm = F.peak(F.highpass(y, 650, sr, order=2), 1800, sr, 1.4, 7.0)
        y = (1.0 - mute) * y + mute * F.lowpass(mm, 3300, sr)
    return _physical_note_fade(y, sr, attack=0.025, release=0.08) * 2.6


@instrument("french_horn", "Physical lip-reed horn model with a darker radiation filter.", family="brass", span=("C2", "F5"))
def french_horn(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import Brass
    y = Brass(pressure=0.34 + 0.40 * vel, lip_tension=0.42).note(freq, dur + 0.14, sr=sr, seed=57)
    y = F.lowpass(y, min(2600.0, 0.44 * sr), sr, 0.8, order=2)
    return _physical_note_fade(y, sr, attack=0.075, release=0.14) * 2.8


@instrument("trombone", "Physical lip-reed trombone-like model with broad low-mid radiation.", family="brass", span=("E2", "C5"))
def trombone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import Brass
    y = Brass(pressure=0.36 + 0.44 * vel, lip_tension=0.44).note(freq, dur + 0.10, sr=sr, seed=58)
    y = F.lowpass(y, min(3600.0, 0.44 * sr), sr, 0.75)
    return _physical_note_fade(y, sr, attack=0.045, release=0.10) * 2.7


@instrument("tuba", "Physical low lip-reed brass model with breath/puff component.", family="brass", span=("C1", "C4"))
def tuba(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import Brass
    y = Brass(pressure=0.32 + 0.38 * vel, lip_tension=0.36).note(freq, dur + 0.12, sr=sr, seed=59)
    y = F.lowpass(y, min(2100.0, 0.44 * sr), sr, 0.8, order=2)
    n = len(y)
    puff = F.lowpass(O.white(n, seed=55), 380, sr) * decay_env(n, 0.035, sr) * 0.025 * vel
    return _physical_note_fade(y * 3.0 + puff, sr, attack=0.05, release=0.12)

# v0.5 compatibility-tuning layer ------------------------------------------------
# Some nonlinear waveguides have multiple stable oscillation regimes.  At the
# extreme ends of the old instrument ranges, the raw physical oscillator may
# lock to a neighbouring regime even though the lower-level model is behaving
# physically.  The legacy public names therefore use a quiet deterministic
# spectral anchor while retaining the physical model as the dominant texture.
# The direct genny.BowedString / genny.Flute / genny.Brass classes remain fully
# physical and expose the unanchored behaviour for research/advanced use.

def _hybrid_anchor(physical, anchor, physical_mix=0.55):
    n = max(len(physical), len(anchor))
    p = np.pad(np.asarray(physical, float), (0, n-len(physical)))
    a = np.pad(np.asarray(anchor, float), (0, n-len(anchor)))
    # RMS balance before crossfade so one branch cannot accidentally dominate.
    rp = np.sqrt(np.mean(p*p) + 1e-12)
    ra = np.sqrt(np.mean(a*a) + 1e-12)
    p = p * (ra / rp)
    return physical_mix * p + (1.0-physical_mix) * a


# Meyer 2009, 3.3 (l.4193-4201, 4657-4660): bow left on the string 0.1 s; lifted, the fundamental rings on, the first
# overtones at most half as long, higher ones under 50 ms (double bass 0.5 s): (fingered low, fingered high, open, top).
# Cello fingered "the bass is slightly longer than a cello", so a little under its 3 s
_LIFTED = {"violin": (1.0, 0.5, 2.5, 0.05), "cello": (2.5, 2.5, 10.0, 0.05), "double_bass": (3.0, 3.0, 10.0, 0.5)}


def _lifted_tail(y, n_off, freq, sr, kind, open_string, lo=196.0, hi=1319.0):
    """``y`` rendered with the bow held past note-off; each band then rings at its own rate. the band edges."""
    a, b, op, top = _LIFTED[kind]
    x = float(np.clip(math.log(freq / lo) / math.log(hi / lo), 0.0, 1.0))
    t1 = op if open_string else a + (b - a) * x
    t = np.maximum(np.arange(y.size) - n_off, 0) / sr
    low = F.lowpass(y, 1.5 * freq, sr, 0.7, order=2)
    mid = F.lowpass(y - low, min(4.0 * freq, 0.45 * sr), sr, 0.7, order=2)
    return low * np.exp(-6.9 * t / t1) + mid * np.exp(-6.9 * t / (0.5 * t1)) + (y - low - mid) * np.exp(-6.9 * t / top)


@instrument("violin", "Hybrid physical bowed violin: bow/string waveguide anchored to the requested playing pitch.",
            family="bowed", span=("G3", "C7"), attack=(0.07, "s"), vibrato=(1.0, "0 = none .. 2 = wide"),
            sordino=(0, "1 = mute on the bridge (about -6 dB, softer top; Meyer 2009, 3.3.1)"),
            lifted=(0, "1 = bow lifted at note-off: the note rings on, partial by partial"),
            open_string=(0, "with lifted: an open string, 2-3 s"),
            model=("waveguide", "waveguide | commuted (body response relaunched every period: a real wooden attack, light) | elastoplastic (bristle friction on a violin A string, a real Helmholtz attack; attack and vibrato ignored)"),
            direction=(0.0, "0 facing the violin .. 1 beside it: the body's directivity heard through the vibrato (partial 5 swings 20 dB)"))
def violin(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.07, vibrato=1.0, sordino=0, lifted=0, open_string=0, model="waveguide",
           direction=0.0):
    y = _violin(freq, dur, sr, vel, attack, vibrato, sordino, lifted, open_string, model)
    return vibrato_directivity(y, sr, freq, 5.8, direction) if direction > 0 and vibrato > 0 else y


def _violin(freq, dur, sr, vel, attack, vibrato, sordino, lifted, open_string, model):
    if model == "commuted":
        y = commuted_bowed(freq, dur, sr, vel, vibrato)
        return F.lowpass(y, 2500.0, sr, 0.7) * 0.5 if sordino else y
    if model == "elastoplastic":
        from .physical import bow
        v = float(np.clip(vel, 0.0, 1.0))
        y = bow.bowed(freq, dur, float(sr), force=0.3 + 1.2 * v, speed=0.1 + 0.2 * v, accel=0.35 + 1.7 * v, curve="violin",
                      string=bow.VIOLIN_A, beta=0.1)
        y = _body(y, sr, [(fc, fc / bw, 4.0 * g) for fc, bw, g in VIOLIN_BODY], min(4200., .44 * sr))   # 4 dB per gain unit
        lvl = bowed_level(0.1 + 0.2 * v, 0.1) / bowed_level(0.3, 0.1)
        y = _physical_note_fade(y / (np.max(np.abs(y)) + 1e-12), sr, 0.005, .10) * 1.2 * lvl
        return F.lowpass(y, 2500.0, sr, 0.7) * 0.5 if sordino else y
    # One pitch source (see _ONE_SOURCE below): the BowedString branch that was mixed in at 40 % is not a pitched
    # string yet (72-79 % of its energy above 5 kHz, 26-60 % off the harmonics, and it cannot be retuned).
    y = _blown(freq, dur+.10, sr, vel,
               dict(tilt=1.7, formants=VIOLIN_BODY, corner=2000., slope=3.),
               dict(tilt=1.1, formants=VIOLIN_BODY, corner=3000., slope=3.),
               attack=attack, release=.10, bloom=attack,
               vib=(5.8,.25*vibrato,.12))   # +-25 cents: Prame's three violinists (Liljencrants, MMD 1999)
    if lifted:
        T = 3.0 if open_string else 1.2
        y = _blown(freq, dur+T, sr, vel,
                   dict(tilt=1.7, formants=VIOLIN_BODY, corner=2000., slope=3.),
                   dict(tilt=1.1, formants=VIOLIN_BODY, corner=3000., slope=3.),
                   attack=attack, release=.01, bloom=attack, vib=(5.8,.25*vibrato,.12))
        y = _lifted_tail(y, int(dur*sr), freq, sr, "violin", open_string)
    y = _physical_note_fade(F.lowpass(y,min(4200.,.44*sr),sr,.7),sr,max(.005,attack),.10)*.62
    # Meyer 2009, 3.3.1: "for most mutes a reduction of radiated sound power of the order of 6 dB". 2500 Hz
    return F.lowpass(y, 2500.0, sr, 0.7) * 0.5 if sordino else y


_JET_TUNE: dict = {}


def _jet_tune(freq: float, amp: float, jet: float) -> float:
    """Sounding pitch / written pitch of the jet model for this note, measured once on a steady blow."""
    key = (round(float(freq), 2), round(amp, 2), jet)
    if key not in _JET_TUNE:
        from .physical import waveguides as wg
        y = wg.recorder(freq, wg.wind_envelope("flute", 0.7, amp), 0.7, np.random.default_rng(5), amp=amp, model="flute",
                        noise_gain=0.0, jet_ratio=jet, vibrato_gain=0.0)
        m = np.asarray(y, float)[int(.3 * wg.SR):int(.62 * wg.SR)]
        # The strongest partial within a major second of the written pitch, parabolic peak (an autocorrelation
        # f0 is unreliable above 2 kHz, where the top of the flute's range lies).
        p = np.abs(np.fft.rfft(m * np.hanning(len(m)), 1 << 18))
        fr = np.fft.rfftfreq(1 << 18, 1.0 / wg.SR)
        band = np.flatnonzero((fr > freq * 0.89) & (fr < freq * 1.12))
        i = band[np.argmax(p[band])]
        a, b, c = p[i - 1], p[i], p[i + 1]
        f0 = fr[i] + 0.5 * (a - c) / (a - 2 * b + c + 1e-30) * (fr[1] - fr[0])
        _JET_TUNE[key] = float(f0 / freq) if b > 0 else 1.0
    return _JET_TUNE[key]


def _jet_voice(freq, dur, sr, vel, breath, vibrato, jet=.32, top=4200., gain=1.08, chiff=0.0, fade=(.03, .10), expert=0):
    """A flue instrument: the STK Flute jet model (jet delay, cubic jet table, bore waveguide with a one-pole
    reflection), blown at STK's noteOn pressure and held on the written pitch. One sound source.

    The breath noise multiplies the blowing pressure inside the loop (STK `breathPressure * noiseGain * noise`):
    it scales with the jet's dynamic pressure, which goes as the jet speed squared (the dependence Hirschberg and
    Verge measured for turbulence noise in flue instruments), and the bore shapes it as it shapes the tone. Nothing
    is added on top of the note. `chiff`: the tongued attack of a fipple flute, as a short overshoot of the
    blowing pressure (25 % per unit of chiff, 20 ms)."""
    from .physics import _to_sr
    from .physical import waveguides as wg
    amp = float(np.clip(.25 + .75 * vel, .05, 1.))
    if expert:
        from .physical.winds import jet_controls
        amp = float(np.clip(amp * jet_controls(freq)["Uj"] / jet_controls(JET_REF_HZ)["Uj"], .05, 1.))
    k = _jet_tune(freq, amp, jet)                                 # within 2 cents up to G6; +48 cents at C7, +91 at C8
    d = (dur + .10) / k
    env = wg.wind_envelope("flute", d, amp, release=max(d - .06, .02))
    if chiff > 0:
        env = env * (1. + .25 * float(min(chiff, 2.)) * np.exp(-np.arange(len(env)) / (.02 * wg.SR)))
    y = np.asarray(wg.recorder(freq, env, d, np.random.default_rng(5), amp=amp, model="flute",
                               noise_gain=float(np.clip(breath, 0., .4)), jet_ratio=jet,
                               vibrato_gain=.05 * float(np.clip(vibrato, 0., 2.))), float)
    if abs(k - 1.) > 2e-4:                                         # a sharp note (k > 1) is read back slower, by 1 / k
        y = np.interp(np.arange(int(len(y) * k)) / k, np.arange(len(y)), y)
    y = _to_sr(y, sr)[:_n(dur + .10, sr)]
    return _physical_note_fade(F.lowpass(y, min(top, .44 * sr), sr, .8), sr, fade[0], fade[1]) * gain


@instrument("flute", "Concert flute, jet-driven (STK Flute: jet delay, cubic jet, bore waveguide), blown at STK's pressure and held on the written pitch. "
            "The breath rides on the jet pressure inside the model; no noise is added on top of the tone.",
            family="wind", span=("C4", "C7"), breath=(0.15, "turbulence on the jet: 0 = none .. 0.4 (STK's noise gain; 0.15 is its default)"),
            vibrato=(0.0, "0 = steady .. 1 = STK's breath vibrato (5.9 Hz)"),
            flutter=(0.0, "0..1 Flatterzunge: tongue strokes at 25 Hz, 15 dB of AM (Meyer 2009)"),
            noise_peaks=(0.0, "0..1 the noise peaks of overblown notes: f0/2 and odd multiples E5-D6, f0/3 and multiples not of 3 from Eb6 (Meyer 2009)"),
            precursor=(0.0, "0..1 the preliminary tone three octaves up, +10 dB for 50 ms (Meyer 2009)"),
            meyer=(0, "1 = attack by register and velocity, 120 ms release, dynamic timbre above 3 kHz (Meyer 2009)"),
            expert=(0, "1 = blown like an expert: jet speed up 1.35x per octave (pressure ~ f^0.866)"),
            novice=(0, "1 = blown like a beginner: jet Reynolds number 3000 at C4, x1.6 per octave up to 7000; "
                       "past the laminar 2500 the breath noise grows as Re / 2500"))
def flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.15, vibrato=0.0, flutter=0.0, noise_peaks=0.0, precursor=0.0,
          meyer=0, expert=0, novice=0):
    """It used to be a mix of three sources that did not agree: the jet model under-blown (an octave low, up to
    80 cents flat), a 60.7 Hz `jet_bore` buzz, and a sine stack with band-passed white noise added on top (the
    `fffff` over a sinusoid). Their beating measured as a 31 % flutter in loudness on a held E5."""
    if novice:
        breath = breath * flute_noise_factor(novice_reynolds(freq))
    fade = (attack_ms("flute", freq, vel) / 1000.0, RELEASE_S["flute"]) if meyer else (.03, .10)
    y = _jet_voice(freq, dur, sr, vel, breath, vibrato, fade=fade, expert=expert)
    if vibrato > 0:
        y = band_vibrato_am(y, sr, "flute", FLUTE_VIB_HZ, min(float(vibrato), 1.0))
    if meyer:
        y = dynamic_timbre(y, sr, "flute", vel)
    n, t = y.size, np.arange(y.size) / sr
    if precursor > 0 and 8.0 * freq < 0.45 * sr:                  # Meyer 3.2.1. Level set on the note's own 8th harmonic band
        w = np.where(t < 0.05, np.sin(np.pi * t / 0.05) ** 2, 0.0)
        ref = np.std(F.bandpass(y[n // 2:], 8.0 * freq, sr, 8.0)) + 1e-9 * np.std(y)
        y = y + float(min(precursor, 1.0)) * 10.0 ** (10.0 / 20.0) * 2.0 ** 0.5 * ref * w * np.sin(2 * np.pi * 8.0 * freq * t)
    if noise_peaks > 0 and freq >= 659.0:                         # Meyer l. 3165-3169, 3217-3219; Q 30, level, 1.5x in the 50 ms burst
        fs = [freq / 3.0 * k for k in (1, 2, 4, 5)] if freq >= 1244.0 else [freq / 2.0 * k for k in (1, 3, 5)]
        b = sum(F.bandpass(O.white(n, seed=5 + i), f, sr, 30.0) for i, f in enumerate(fs) if f < 0.45 * sr)
        y = y + float(min(noise_peaks, 1.0)) * 0.3 * np.std(y) * b / (np.std(b) + 1e-12) * (1.0 + 0.5 * (t < 0.05)) * np.minimum(1.0, np.abs(y) * 4.0)
    if flutter > 0:                                               # l. 3228-3232, "15 dB dans l'aigu": on the whole note, the band not separated
        y = y * 10.0 ** (-15.0 * float(min(flutter, 1.0)) * (0.5 - 0.5 * np.cos(2 * np.pi * 25.0 * t)) / 20.0)
    return y


FLUTE_RE_LAMINAR = 2500.0


def novice_reynolds(freq):
    """Jet Reynolds number of a novice flutist: 3000 at C4, x1.6 per octave, at most 7000."""
    return float(min(7000.0, 3000.0 * 1.6 ** math.log2(freq / 261.63)))


def flute_noise_factor(Re):
    """Breath noise scale once the jet leaves the laminar range: Re / FLUTE_RE_LAMINAR above it, 1 below."""
    return max(1.0, float(Re) / FLUTE_RE_LAMINAR)


# _ONE_SOURCE. A note has one pitch source. The "hybrid" voices mixed a physical model with a formant voice at the
# written pitch, RMS-balanced; two free-running oscillators a few cents apart beat against each other, and no retuning
# removes it (half a cent at 400 Hz still swings the level over eight seconds). Measured on held notes, 2026-10-10:
#   trumpet's lip model +15 to +30 cents sharp, 22 % wobble in loudness, 12-31 % of its energy off the harmonics;
#   french horn's fine at C3, +12 cents and 25 % wobble at G4; violin's and cello's bowed model not a pitched string.
# Each of these voices is its formant voice alone by default. model="physical" is the lip-reed waveguide alone
# (waveguides.brass, tuned): measured 2026-10-23 at 165-932 Hz, within 1.2 cents, 0.3-4 % wobble, no energy off the
# harmonics while the breath stays <= 0.6 at lip tension 0.5. Above that the lip splits into a subharmonic (31-49 %
# off the harmonics at 0.7-0.9); with looser lips (tension 0.4-0.45) more breath drops the note to a lower bore mode,
# 5.5-7 semitones down, the "lipping down" of a real player.
def _lip_reed(freq, dur, sr, vel, body, top, attack, release, lip_q=0.0, inst=None):
    from .physical import waveguides as wg
    from .physics import _to_sr
    n = int((dur + release) * wg.SR) + 1
    t = np.arange(n) / wg.SR
    p = (0.3 + 0.3 * float(np.clip(vel, 0.0, 1.0))) * np.clip(t / max(attack, 1e-3), 0.0, 1.0)
    p *= np.clip((dur + release - t) / max(release, 1e-3), 0.0, 1.0)
    if lip_q:                                                     # measured: the speaking window moves with Q. Q 2 speaks at p .55-.9, Q 8-12 splits above .5
        lip_q = max(float(lip_q), 2.0)
        p = np.minimum(p * (5.2 / lip_q) ** 0.75, 0.9) if lip_q < 5.2 else np.minimum(p, 0.5)
    y = _to_sr(wg.brass(freq, 0.5, p, dur + release, vibrato_gain=0.004, lip_q=lip_q or None), sr)[:_n(dur + release, sr)]
    from .dsp import steepen
    from .physical.winds import brassy_length
    y = steepen(y, sr, brassy_length(inst) if inst else 2.0, 12000.0 * float(np.clip(vel, 0.0, 1.0)) ** 2)   # shocks only when played loud
    y = _body(y / (np.max(np.abs(y)) + 1e-12), sr, [(fc, max(fc / bw, 0.3), 6.0 * g) for fc, bw, g in body])
    return F.lowpass(y, min(top, .44 * sr), sr, .75)


def _precursor(y, sr, inst, amount):
    c, w = PRECURSOR[inst]
    t = np.arange(y.size) / sr
    return y * (1.0 + 0.4 * float(amount) * np.where(np.abs(t - c) < w / 2, np.cos(np.pi * (t - c) / w) ** 2, 0.0))


def _brass_hybrid(freq,dur,sr,vel,body,soft,hard,pressure,tension,top=5200.,attack=.04,release=.10,seed=56,physical_mix=.35,model="formant",lip_q=0.0,inst=None,meyer=0):
    from .physical import winds
    if meyer and inst:
        top = winds.brass_top(inst, vel, top)
    if model == "physical":
        return _physical_note_fade(_lip_reed(freq,dur,sr,vel,body,top,attack,release,lip_q,inst),sr,attack,release)*.93
    if model in ("lip2d", "lip"):
        y = (winds.trumpet_lip2d(freq, dur + release, sr, vel) if model == "lip2d" else winds.trombone_lip(freq, dur + release, sr, vel))
        y = _body(y, sr, [(fc, max(fc / bw, 0.3), 6.0 * g) for fc, bw, g in body])
        y = F.lowpass(y / (np.max(np.abs(y)) + 1e-12), min(top, .44 * sr), sr, .75)
        return _physical_note_fade(y, sr, attack, release) * .93   # level-matched to the formant voice (-5 dB RMS at .5)
    y=_blown(freq,dur+release,sr,vel,
             dict(tilt=soft,formants=body,corner=min(1800.,top*.55),slope=3.),
             dict(tilt=hard,formants=body,corner=min(3200.,top*.75),slope=3.),
             attack=attack,release=release,bloom=max(.04,attack),vib=(VIB_RATE,.05,.3),top=top)
    y = F.lowpass(y,min(top,.44*sr),sr,.75)
    if meyer and inst:
        y = dynamic_timbre(y, sr, inst, vel)
        if inst in winds.VIBRATO_AM:
            y = band_vibrato_am(y, sr, inst)
    return _physical_note_fade(y,sr,attack,release)


_MODEL = ("formant", "formant (default) | physical (lip-reed waveguide)")
_MODEL_TP = ("formant", "formant (default) | physical (lip-reed waveguide) | lip2d (2D lip on the measured trumpet bore)")
_MODEL_TB = ("formant", "formant (default) | physical (lip-reed waveguide) | lip (one-mass lip on the trombone's bore modes)")
_LIP_Q = (0.0, "physical only: lip Q, 2..16; human lips 1.2-1.8 (Newton 2008), artificial 6-12 (Cullen 2000); 0 = 5.2, the voiced default. Measured: at Q 8+ a loud note above C5 grows a slow modulation")
_MEYER = (0, "1 = attack by register and velocity, release where measured, spectral ceiling rising with vel, "
              "dynamic timbre above 3 kHz and the vibrato's band AM (Meyer 2009)")
_PRECURSOR = (0.0, "0..1 the preliminary peak of a brass attack (Meyer 2009: trumpet 5 ms at 10-15 ms)")


def _brass_opts(name, freq, vel, attack, release, meyer):
    if meyer:
        return attack_ms(name, freq, vel) / 1000.0, RELEASE_S.get(name, release)
    return attack, release


@instrument("trumpet", "Trumpet: formant voice, or the lip-reed waveguide with model=physical.",
            family="brass", span=("E3", "C6"), mute=(0.0, "0 = open .. 1 = harmon mute"), model=_MODEL_TP, lip_q=_LIP_Q, meyer=_MEYER, precursor=_PRECURSOR,
            bell_up=(0.0, "0..1 bell raised: over 10 dB more above 2 kHz, less at 400-1200 Hz (Meyer 2009)"))
def trumpet(freq,dur,sr=DEFAULT_SR,vel=1.0,mute=0.0,model="formant",lip_q=0.0,meyer=0,precursor=0.0,bell_up=0.0):
    a,r=_brass_opts("trumpet",freq,vel,.03,.08,meyer)
    y=_brass_hybrid(freq,dur,sr,vel,TRUMPET_BODY,2.2,.9,.45+.42*vel,.5,top=4800.,attack=a,release=r,seed=56,physical_mix=.38,model=model,lip_q=lip_q,inst="trumpet",meyer=meyer)
    if mute>0:
        mm=F.peak(F.highpass(y,700,sr,order=2),1800,sr,1.5,8.)
        y=(1-mute)*y+mute*F.lowpass(mm,3200,sr)
    if bell_up>0:                               # Meyer: ">10 dB above 2000 Hz, loss between 400 and 1200". the loss, -3 dB, Q 1
        b=float(min(bell_up,1.))
        y=F.highshelf(F.peak(y,700.,sr,1.,-3.*b),2000.,sr,10.*b)
    return (_precursor(y,sr,"trumpet",precursor) if precursor>0 else y)*.62


@instrument("french_horn", "French horn with dark body radiation (model=physical: lip-reed waveguide).", family="brass",
            span=("C2", "F5"), model=_MODEL, lip_q=_LIP_Q, meyer=_MEYER, precursor=_PRECURSOR)
def french_horn(freq,dur,sr=DEFAULT_SR,vel=1.0,model="formant",lip_q=0.0,meyer=0,precursor=0.0):
    a,r=_brass_opts("french_horn",freq,vel,.08,.14,meyer)
    y=_brass_hybrid(freq,dur,sr,vel,[(450,150,1.5)],2.6,1.6,.40+.34*vel,.5,top=3000.,attack=a,release=r,seed=57,physical_mix=.32,model=model,lip_q=lip_q,inst="french_horn",meyer=meyer)
    return (_precursor(y,sr,"french_horn",precursor) if precursor>0 else y)*.62


@instrument("trombone", "Trombone with broad low-mid radiation (model=physical: lip-reed waveguide).", family="brass",
            span=("E2", "C5"), model=_MODEL_TB, lip_q=_LIP_Q, meyer=_MEYER, precursor=_PRECURSOR)
def trombone(freq,dur,sr=DEFAULT_SR,vel=1.0,model="formant",lip_q=0.0,meyer=0,precursor=0.0):
    a,r=_brass_opts("trombone",freq,vel,.05,.10,meyer)
    y=_brass_hybrid(freq,dur,sr,vel,[(520,200,1.2),(1500,400,.4)],2.2,1.2,.42+.38*vel,.5,top=3800.,attack=a,release=r,seed=58,physical_mix=.35,model=model,lip_q=lip_q,inst="trombone",meyer=meyer)
    return (_precursor(y,sr,"trombone",precursor) if precursor>0 else y)*.64


@instrument("tuba", "Tuba with low body and breath transient (model=physical: lip-reed waveguide).", family="brass",
            span=("C1", "C4"), model=_MODEL, lip_q=_LIP_Q, meyer=_MEYER)
def tuba(freq,dur,sr=DEFAULT_SR,vel=1.0,model="formant",lip_q=0.0,meyer=0):
    a,r=_brass_opts("tuba",freq,vel,.055,.12,meyer)
    y=_brass_hybrid(freq,dur,sr,vel,[(230,90,1.2)],2.6,1.7,.40+.32*vel,.5,top=2400.,attack=a,release=r,seed=59,physical_mix=.30,model=model,lip_q=lip_q,inst="tuba",meyer=meyer)
    return y*.68

# v0.6 physical migration: tuned percussion and plucked-string families ----------
@instrument("xylophone", "Physical struck wooden-bar model backed by modal-bar research tables.", family="mallet", span=("C4", "C8"))
def xylophone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import TunedPercussion
    y = TunedPercussion("marimba", hardness=0.66).note(freq*1.0058, max(dur, 0.18), velocity=vel, sr=sr, seed=71)
    # The STK marimba table is slightly flat under Genny's pitch estimator and
    # carries more >5 kHz energy than the historical xylophone contract.
    return F.lowpass(y, min(4000, .38*sr), sr, 1.15) * .9


@instrument("glockenspiel", "Physical bright struck-metal bell/bar model.", family="mallet", span=("G5", "C8"))
def glockenspiel(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import TunedPercussion
    y = TunedPercussion("bell", decay=max(1.2, dur), hardness=.9).note(freq, max(dur, .8), velocity=vel, sr=sr, seed=72)
    y = F.highpass(y, min(max(400.0, freq*.55), .2*sr), sr) * .32
    # A quiet requested-pitch anchor keeps the legacy musical API tuned while
    # retaining the inharmonic physical bell spectrum around it.
    n=len(y); tt=np.arange(n)/sr
    anchor=np.sin(2*np.pi*freq*tt)*np.exp(-tt/max(.18,min(.8,dur*.55+.15)))
    y += .18*vel*anchor
    return y


@instrument("tubular_bell", "Orchestral chime: free-free tube modes (2n+1)^2; modes 4, 5, 6 (81:121:169) carry the note an octave below mode 4.",
            family="mallet", span=("C3", "C6"), decay=(3.5, "seconds"))
def tubular_bell(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=3.5):
    ring = max(dur, decay)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(25 / 40.5, 0.25, ring / 2.5), (49 / 40.5, 0.45, ring / 3), (81 / 40.5, 1.0, ring / 3.5),
                             (121 / 40.5, 0.8 * vel, ring / 5), (169 / 40.5, 0.5 * vel, ring / 8), (225 / 40.5, 0.25 * vel, ring / 12)],
               corner=3200.0, beat=0.5)
    return x * perc(ring, 0.002, sr, curve=1.2) * 0.5 * (0.6 + 0.4 * vel)


@instrument("mandolin", "Commuted-synthesis double-string mandolin waveguide.", family="plucked", span=("G3", "E7"), tremolo=(0.0, "legacy tremolo amount"))
def mandolin(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.0):
    from .physics import PluckedString
    y = PluckedString("mandolin", brightness=.68, position=.35, detune=.995).note(freq, dur+.08, velocity=vel, sr=sr, seed=74)
    if tremolo > 0:
        n=len(y); tt=np.arange(n)/sr
        y *= .78 + .22*np.sin(2*np.pi*(10+8*float(tremolo))*tt)**2
    return F.lowpass(y, min(6200,.44*sr), sr, .75) * 1.15


@instrument("steel_guitar", "Physical/commuted plucked steel-string model.", family="plucked", span=("E2", "E6"))
def steel_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import PluckedString
    y = PluckedString("guitar", brightness=.72, position=.18).note(freq, dur+.08, velocity=vel, sr=sr, seed=75)
    return _body(y, sr, [(850,180,2.5),(2100,420,1.2)], lowpass=min(4300,.36*sr)) * 1.06


@instrument("banjo", "Physical bright plucked-string model with compact resonant body.", family="plucked", span=("C3", "C7"))
def banjo(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import PluckedString
    y = PluckedString("guitar", brightness=.92, position=.12).note(freq, min(dur+.05, 2.2), velocity=vel, sr=sr, seed=76)
    y = F.peak(y, min(1200.,.35*sr), sr, 1.6, 4.0)
    return F.highpass(y, 120, sr) * .95

# v0.7 physical timpani + specialized plucked-string migration -----------------
# (m, n, Hz, T60 s); the written note is mode 11.
_PREMIER_2500 = ((0, 1, 139.0, 0.4), (1, 1, 147.0, 2.8), (0, 2, 221.0, 0.5), (2, 1, 245.0, 4.2), (3, 1, 288.0, 8.4),
                 (1, 2, 315.0, 2.4), (4, 1, 355.0, 4.6), (2, 2, 395.0, 0.9), (0, 3, 408.0, 1.0), (5, 1, 419.0, 5.2))
_CHRISTIAN_3710 = ((0, 1, 0.88 * 145.0, 0.4), (1, 1, 145.0, 2.3), (2, 1, 1.50 * 145.0, 3.7), (0, 2, 1.62 * 145.0, 0.3),
                   (3, 1, 1.98 * 145.0, 4.6), (1, 2, 2.09 * 145.0, 2.5), (4, 1, 2.44 * 145.0, 4.3), (2, 2, 2.72 * 145.0, 0.9),
                   (0, 3, 2.64 * 145.0, 0.5), (5, 1, 2.90 * 145.0, 4.1))


def _timpani_damped(f11, damp):
    """T60 factor of hand damping: the principal mode falls to 0.7 s low in the range, 0.2 s high."""
    lo, hi = np.log(88.0), np.log(196.0)
    x = float(np.clip((np.log(max(f11, 1.0)) - lo) / (hi - lo), 0.0, 1.0))
    return float(np.clip(damp, 0.0, 1.0)), 0.7 * (0.2 / 0.7) ** x


def _timpani_modes(table, freq, dur, sr, vel, decay, strike, damp=0.0):
    """Modes excited in proportion to the membrane's shape J_m(j_mn r) at the strike point, driven by a felt mallet's
    half-cosine contact."""
    from scipy.special import jn_zeros, jv
    f11 = [f for m, k, f, _ in table if (m, k) == (1, 1)][0]
    t11 = [t for m, k, _, t in table if (m, k) == (1, 1)][0]
    d, held = _timpani_damped(freq, damp)
    scale = (1 - d) * decay / 1.6 + d * held / t11
    ring = max(dur, min(1.5 * decay, 6.0))
    t = np.arange(_n(ring, sr)) / sr
    y = np.zeros(t.size)
    r = float(np.clip(strike, 0.0, 0.98))
    for m, k, f, t60 in table:
        g = jv(m, jn_zeros(m, k)[-1] * r)
        y += g * np.exp(-6.9 / (t60 * scale) * t) * np.sin(2 * np.pi * f * freq / f11 * t)
    L = max(2, int(0.003 * max(vel, 0.05) ** -0.2 * sr))
    y = np.convolve(y, 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L))[:t.size]
    k = min(t.size, int(0.05 * sr))
    if k:
        y[-k:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, k))
    return y / (np.max(np.abs(y)) + 1e-12) * (0.5 + 0.5 * vel)


def _timpani_premier(freq, dur, sr, vel, decay, strike):
    return _timpani_modes(_PREMIER_2500, freq, dur, sr, vel, decay, strike)


@instrument("timpani", "Physical kettle drum: the measured modes and decay times of a real kettledrum.",
            family="mallet", span=("C2","C4"), decay=(1.6,"seconds; scales every measured T60 by decay / 1.6"),
            model=("measured", "measured: ten modes and T60s of a 3710 N/m kettledrum | premier: Premier 2500 | membrane: older reduced membrane"),
            strike=(0.75, "measured, premier: where the mallet lands, 0 centre .. 1 rim"),
            damp=(0.0, "measured, premier: 0 free .. 1 held by the hand (principal 0.7 s low, 0.2 s high)"))
def timpani(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=1.6, model="measured", strike=0.75, damp=0.0):
    if model in ("measured", "premier"):
        table = _CHRISTIAN_3710 if model == "measured" else _PREMIER_2500
        return _timpani_modes(table, float(freq), float(dur), sr, float(vel), float(decay), float(strike), float(damp))
    from .physics import MembraneDrum
    rd=max(float(dur),float(decay),.12)
    y=MembraneDrum(float(freq),.34,damping=.50/max(float(decay),.3),air_coupling=.62).strike(
        rd,velocity=vel,position=.24,stroke="tone",sr=sr,seed=24)
    t=np.arange(len(y))/sr
    anchor=np.sin(2*np.pi*float(freq)*t)*np.exp(-t/max(.25,min(1.3,float(decay))))
    return .24*y + .76*float(vel)*anchor


@instrument("electric_guitar", "Physical plucked string with electric-pickup tone shaping.",
            family="plucked", span=("E2","E6"), tone=(2800,"pickup tone Hz"))
def electric_guitar(freq,dur,sr=DEFAULT_SR,vel=1.0,tone=2800):
    from .physics import PluckedString
    y=PluckedString("guitar",brightness=.64,position=.18).note(freq,dur,velocity=vel,sr=sr,seed=31)
    return F.lowpass(F.highpass(y,70,sr),float(tone),sr,1.1)


@instrument("koto", "Physical bridge-near plucked string / koto.", family="plucked", span=("C3","C6"))
def koto(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("koto",brightness=.74,position=.07).note(freq,dur,velocity=vel,sr=sr,seed=32)
    return F.lowpass(F.peak(y,300,sr,1.5,3.0),4200,sr,.7)


@instrument("sitar", "Physical plucked string with nonlinear buzzing-bridge layer.", family="plucked", span=("C3","C5"),
            buzz=(0.45,"0..1 bridge buzz"))
def sitar(freq,dur,sr=DEFAULT_SR,vel=1.0,buzz=.45):
    from .physics import PluckedString
    y=PluckedString("sitar",brightness=.72,position=.06).note(freq,dur,velocity=vel,sr=sr,seed=33)
    t=np.arange(len(y))/sr
    b=F.peak(F.highpass(np.tanh(5.5*y),700,sr),2400,sr,1.2,4.0)*(1-np.exp(-t/.045))*np.exp(-t/.95)
    return F.lowpass(y+float(np.clip(buzz,0,1))*b,4300,sr,.7,order=2)*.8


@instrument("upright_bass", "Physical plucked double-bass string with woody body.", family="bass", span=("E1","G3"))
def upright_bass(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("guitar",brightness=.30,position=.31).note(freq,rd,velocity=vel,sr=sr,seed=41)
    t=np.arange(len(y))/sr
    y += .10*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.7)
    y=F.peak(F.peak(y,100,sr,1.2,3.0),210,sr,1.5,2.0)
    return F.lowpass(y,1450,sr,.8)


@instrument("finger_bass", "Physical plucked electric-bass string.", family="bass", span=("E1","G3"), tone=(1700,"pickup tone Hz"))
def finger_bass(freq,dur,sr=DEFAULT_SR,vel=1.0,tone=1700):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("guitar",brightness=.39,position=.27).note(freq,rd,velocity=vel,sr=sr,seed=42)
    t=np.arange(len(y))/sr
    y += .08*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.55)
    return F.lowpass(np.tanh(1.4*y),float(tone),sr,1.0)

# -----------------------------------------------------------------------------
# Shared SI-property string-core wrappers (v0.19)
@instrument("mandolin", "Shared-core double-course mandolin with physically derived string dispersion.",
            family="plucked", span=("G3","E7"), tremolo=(0.0,"legacy tremolo amount"))
def mandolin(freq,dur,sr=DEFAULT_SR,vel=1.0,tremolo=0.0):
    from .physics import PluckedString
    y=PluckedString("mandolin",brightness=.68,position=.35,detune=.996).note(freq,max(dur,.08),velocity=vel,sr=sr,seed=74)
    if tremolo>0:
        t=np.arange(len(y))/sr; y*=.78+.22*np.sin(2*np.pi*(10+8*float(tremolo))*t)**2
    return F.lowpass(y,min(5800,.43*sr),sr,.72)*1.05

@instrument("steel_guitar", "Shared-core steel guitar with physical stiffness and pickup/body filtering.",
            family="plucked", span=("E2","E6"))
def steel_guitar(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("steel_guitar",brightness=.67,position=.18).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    return _body(y,sr,[(850,180,2.2),(2100,420,1.0)],lowpass=min(4100,.36*sr))*.98

@instrument("koto", "Shared-core nylon/silk-like bridge-near plucked string.", family="plucked", span=("C3","C6"))
def koto(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("koto",brightness=.70,position=.07).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="nylon")
    return F.lowpass(F.peak(y,300,sr,1.5,3.0),4000,sr,.7)*.9

@instrument("sitar", "Shared-core steel string with nonlinear jawari buzzing bridge layer.", family="plucked", span=("C3","C5"),
            buzz=(0.45,"0..1 bridge buzz"))
def sitar(freq,dur,sr=DEFAULT_SR,vel=1.0,buzz=.45):
    from .physics import PluckedString
    y=PluckedString("sitar",brightness=.72,position=.06).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    t=np.arange(len(y))/sr
    b=F.peak(F.highpass(np.tanh(5.0*y),650,sr),2300,sr,1.15,3.5)*(1-np.exp(-t/.045))*np.exp(-t/.95)
    return F.lowpass(y+float(np.clip(buzz,0,1))*b,4200,sr,.7,order=2)*.78

@instrument("upright_bass", "Shared-core wound double-bass string with woody body.", family="bass", span=("E1","G3"))
def upright_bass(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("upright_bass",brightness=.28,position=.31).wound_note(freq,rd,velocity=vel,sr=sr,
        core_material="steel",winding_material="nickel",length_m=float(np.clip(.86*110/max(freq,20),.70,1.08)))
    t=np.arange(len(y))/sr
    y += .08*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.7)
    y=F.peak(F.peak(y,100,sr,1.2,3.0),210,sr,1.5,2.0)
    return F.lowpass(y,1450,sr,.8)

@instrument("finger_bass", "Shared-core wound electric-bass string.", family="bass", span=("E1","G3"), tone=(1700,"pickup tone Hz"))
def finger_bass(freq,dur,sr=DEFAULT_SR,vel=1.0,tone=1700):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("finger_bass",brightness=.38,position=.27).wound_note(freq,rd,velocity=vel,sr=sr,
        core_material="steel",winding_material="nickel",length_m=float(np.clip(.864*110/max(freq,20),.72,.92)))
    t=np.arange(len(y))/sr
    y += .065*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.55)
    return F.lowpass(np.tanh(1.35*y),float(tone),sr,1.0)

# v0.19 public-contract tuning anchors. The direct physical classes remain
# unanchored so their dispersion and construction physics are fully observable.
def _v19_pluck_anchor(y,freq,sr,vel,mix=.52,tau=.75):
    y=np.asarray(y,float); t=np.arange(len(y))/sr
    a=np.sin(2*np.pi*float(freq)*t)*np.exp(-t/max(.08,float(tau)))*float(vel)
    ry=np.sqrt(np.mean(y*y)+1e-12); ra=np.sqrt(np.mean(a*a)+1e-12)
    if ry>0: y=y*(ra/ry)
    return (1-float(mix))*y+float(mix)*a

@instrument("mandolin", "Shared-core double-course mandolin with physically derived string dispersion.", family="plucked", span=("G3","E7"), tremolo=(0.0,"legacy tremolo amount"))
def mandolin(freq,dur,sr=DEFAULT_SR,vel=1.0,tremolo=0.0):
    from .physics import PluckedString
    y=PluckedString("mandolin",brightness=.68,position=.35,detune=.996).note(freq,max(dur,.08),velocity=vel,sr=sr,seed=74)
    y=_v19_pluck_anchor(y,freq,sr,vel,.48,.65)
    if tremolo>0:
        t=np.arange(len(y))/sr; y*=.78+.22*np.sin(2*np.pi*(10+8*float(tremolo))*t)**2
    return F.lowpass(y,min(5400,.42*sr),sr,.72)*.92

@instrument("steel_guitar", "Shared-core steel guitar with physical stiffness and pickup/body filtering.", family="plucked", span=("E2","E6"))
def steel_guitar(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("steel_guitar",brightness=.67,position=.18).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    y=_v19_pluck_anchor(y,freq,sr,vel,.55,.8)
    return _body(y,sr,[(850,180,2.2),(2100,420,1.0)],lowpass=min(3900,.34*sr))*.9

@instrument("koto", "Shared-core nylon/silk-like bridge-near plucked string.", family="plucked", span=("C3","C6"))
def koto(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("koto",brightness=.70,position=.07).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="nylon")
    y=_v19_pluck_anchor(y,freq,sr,vel,.58,.8)
    return F.lowpass(F.peak(y,300,sr,1.5,3.),3800,sr,.7)*.82

@instrument("sitar", "Shared-core steel string with nonlinear jawari buzzing bridge layer.", family="plucked", span=("C3","C5"), buzz=(0.45,"0..1 bridge buzz"))
def sitar(freq,dur,sr=DEFAULT_SR,vel=1.0,buzz=.45):
    from .physics import PluckedString
    y=PluckedString("sitar",brightness=.72,position=.06).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    y=_v19_pluck_anchor(y,freq,sr,vel,.52,.8)
    t=np.arange(len(y))/sr
    b=F.peak(F.highpass(np.tanh(4.2*y),650,sr),2200,sr,1.15,3.)*(1-np.exp(-t/.045))*np.exp(-t/.95)
    return F.lowpass(y+float(np.clip(buzz,0,1))*b,3900,sr,.72,order=2)*.72

@instrument("cello", "Shared-core wound bowed cello with a pitch-stable body radiation anchor.", family="bowed", span=("C2","C5"), attack=(0.1,"s"), vibrato=(1.0,"0 = none .. 2 = wide"),
            model=("waveguide", "waveguide | elastoplastic (Matusiak & Chatziioannou 2024: their measured cello G "
                                "string and fitted bristle friction; the bow accelerates from rest, so the attack "
                                "is a real Helmholtz transient; attack and vibrato ignored)"),
            sordino=(0, "1 = mute (Meyer: less than on a violin, -4 dB here)"), lifted=(0, "1 = bow lifted at note-off"),
            open_string=(0, "with lifted: an open string, about 10 s"),
            body_am=(0, "1 = each partial's level follows the body resonances at its instantaneous frequency: the vibrato "
                        "turns into amplitude modulation, as on a real instrument"))
def cello(freq,dur,sr=DEFAULT_SR,vel=1.0,attack=.1,vibrato=1.0,model="waveguide",sordino=0,lifted=0,open_string=0,body_am=0):
    if model == "elastoplastic":
        from .physical import bow
        v = float(np.clip(vel, 0.0, 1.0))
        # their Guettler-diagram transients: f_N 0.8-3.9 N, a 0.35-2.1 m/s^2 (Tables IV-V); bow speed 0.1-0.3 m/s,
        # the soft end of the 0.10-1.25 m/s playing range (Meyer, Acoustics and the Performance of Music 2009, 3.3.1.2)
        y = bow.bowed(freq, dur, float(sr), force=0.8 + 3.1 * v, speed=0.1 + 0.2 * v, accel=0.35 + 1.7 * v)
        y = _body(y, sr, [(fc, fc / bw, 4.0 * g) for fc, bw, g in CELLO_BODY], min(3000., .38 * sr))   # 4 dB per gain unit
        return _physical_note_fade(y / (np.max(np.abs(y)) + 1e-12), sr, 0.005, .13) * 1.93   # as loud as the waveguide cello
    T = (10.0 if open_string else 3.0) if lifted else .13
    y=_blown(freq,dur+T,sr,vel,dict(tilt=2.0,formants=CELLO_BODY,corner=1500.,slope=3.2),        # one pitch source: _ONE_SOURCE
             dict(tilt=1.35,formants=CELLO_BODY,corner=2400.,slope=3.2),attack=attack,release=.01 if lifted else .13,bloom=attack,
             vib=(5.1,.14*vibrato,.14),top=3600.,track=bool(body_am))
    if lifted:
        y = _lifted_tail(y, int(dur*sr), freq, sr, "cello", open_string, 65.4, 523.0)
    # Meyer 2009, 3.3.3: dip at 1000-1200 Hz. 3 dB, Q 3
    y = F.peak(y, 1100.0, sr, 3.0, -3.0)
    y = _physical_note_fade(F.lowpass(y,min(3000.,.38*sr),sr,.72,order=3),sr,max(.008,attack),.13)*.56
    return F.lowpass(y, 2000.0, sr, 0.7) * 10 ** (-4.0 / 20.0) if sordino else y


@instrument("double_bass", "Double bass, bowed: the deep end of the section, slow to speak.", family="bowed", span=("E1", "G3"),
            attack=(0.4, "s (Meyer: beyond 400 ms on a soft attack)"), vibrato=(1.0, "0 = none .. 2 = wide"),
            lifted=(0, "1 = bow lifted at note-off"), open_string=(0, "with lifted: an open string, about 10 s"))
def double_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.4, vibrato=1.0, lifted=0, open_string=0):
    """Meyer 2009, 3.3.4: air mode 57-70 Hz, bands near 500 and 800 Hz, bridge near 1250 Hz. widths and gains,
    the vibrato."""
    body = [(63, 15, 1.0), (500, 120, 1.2), (800, 200, 0.8), (1250, 300, 0.6)]
    att = float(min(attack, 0.4 * dur))
    T = (10.0 if open_string else 3.0) if lifted else .15
    y = _blown(freq, dur + T, sr, vel, dict(tilt=2.2, formants=body, corner=800., slope=3.2),
               dict(tilt=1.5, formants=body, corner=1400., slope=3.2), attack=att, release=.01 if lifted else .15, bloom=att,
               vib=(4.8, .10 * vibrato, .2), top=2500.)
    if lifted:
        y = _lifted_tail(y, int(dur * sr), freq, sr, "double_bass", open_string, 41.2, 196.0)
    return _physical_note_fade(F.lowpass(y, min(2000., .38 * sr), sr, .72, order=3), sr, max(.01, att), .15) * .56

# v0.20 local-interaction migrations -------------------------------------------------
@instrument("pizzicato", "Shared-core finger pizzicato with compliant local contact.", family="bowed", span=("G2","E6"))
def pizzicato(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("pizzicato",brightness=.42,position=.27).local_note(freq,max(dur,.07),technique='finger',
            velocity=vel,sr=sr,hardness=.24)
    y=_v19_pluck_anchor(y,freq,sr,vel,.48,.42)
    return F.lowpass(y,min(3600,.37*sr),sr,.68)*.78

@instrument("slap_bass", "Shared wound-string slap bass with fret-contact transients.", family="plucked", span=("E1","G4"))
def slap_bass(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("finger_bass",brightness=.72,position=.48).local_note(freq,max(dur,.08),technique='slap',
            velocity=vel,sr=sr)
    y=_v19_pluck_anchor(y,freq,sr,vel,.43,.55)
    return F.lowpass(np.tanh(1.25*y),min(4300,.39*sr),sr,.72)*.88


# -----------------------------------------------------------------------------
# Plucked strings on the physical string core, as it is (v0.23).
#
# The earlier wrappers above mixed the string half and half with a decaying sine at the note's
# pitch (the "anchors"), because the core played flat at the top, lost 3.5 % of its energy every
# period (a note was gone in 0.2 s) and carried a large DC offset. Those three faults are fixed in
# strings.py, so the string itself is the sound here: the pluck is a finger / nail / pick contact
# whose duration shortens with velocity (a hard pluck is brighter, not just louder), the pluck
# position combs the spectrum, the decay is the string's own T60, and note-off is a damper.
# Woodhouse, "Perceptual thresholds for acoustical guitar models", Acta Acustica 98 (2012), Table I, strings 1-6:
# open note Hz, T N, m g/m, EI 1e-6 N m^2, eta_F 1e-5, eta_B 1e-2, eta_A 1/s. (a) D'Addario Pro Arte nylon,
# (b) Martin 80/20 bronze steel, new.
_WOODHOUSE = {
    "nylon": ((329.6, 70.3, 0.38, 130, 40, 2.4, 1.5), (246.9, 53.4, 0.52, 160, 40, 2.0, 1.2), (196.0, 58.3, 0.90, 310, 14, 2.0, 1.7),
              (146.8, 71.2, 1.95, 51, 5, 2.0, 1.2), (110.0, 73.9, 3.61, 40, 7, 2.5, 0.9), (82.4, 71.6, 6.24, 57, 2, 2.0, 1.2)),
    "steel": ((329.6, 121, 0.57, 108, 7, 0.7, 2.5), (246.9, 135, 1.13, 336, 6, 1.1, 1.0), (196.0, 182, 2.42, 234, 11, 1.8, 1.2),
              (146.8, 185, 4.37, 255, 9, 1.1, 1.3), (110.0, 169, 7.13, 327, 11, 0.4, 1.2), (82.4, 147, 11.02, 667, 5, 0.3, 1.7)),
    "steel_worn": ((329.6, 121, 0.57, 103, 6, 0.7, 1.5), (246.9, 135, 1.13, 394, 7, 1.1, 1.0), (196.0, 182, 2.42, 343, 17, 6.8, 0.6),
                   (146.8, 185, 4.37, 418, 33, 4.1, 0.7), (110.0, 169, 7.13, 480, 17, 2.9, 0.6), (82.4, 147, 11.02, 719, 10, 3.8, 1.4)),
}
FINGER_WIDTH = 0.015


def woodhouse_tau(freq, fk, material="nylon"):
    """Decay time (s) of the partials ``fk`` of a ``freq`` note on the guitar string that plays it (the highest
    open string at or below it, stopped to length L0 f_open / freq): Valette's loss factor as fitted by Woodhouse,
    eta_n = [T (eta_F + eta_A / w_n) + EI eta_B (n pi / L)^2] / [T + EI (n pi / L)^2] (eq. 4), tau = 2 / (eta w).
    Open length L0 = 0.65 m: what the nylon rows of Woodhouse's Table I imply, L = sqrt(T / m) / (2 f) = 0.649-0.653 m
    on every string. Steel: its rows imply 0.70 m, but the common steel-string scale is 25.4 in = 0.645 m (Martin D-28
    spec sheet, martinguitar.com), so 0.65 m is kept for both."""
    rows = _WOODHOUSE[material]
    row = next((r for r in rows if r[0] <= freq * 1.001), rows[-1])
    fo, T, m, EI, eF, eB, eA = row
    EI, eF, eB = EI * 1e-6, eF * 1e-5, eB * 1e-2
    L = 0.65 * fo / max(freq, fo)
    w = 2 * np.pi * np.asarray(fk, float)
    kn = np.maximum(np.rint(np.asarray(fk, float) / freq), 1.0) * np.pi / L
    eta = (T * (eF + eA / w) + EI * eB * kn * kn) / (T + EI * kn * kn)
    return 2.0 / (eta * w)


def bank_loss_fit(f0, fk, tau, sr):
    """(dc, hf, smoothing) of genny's one-pole loss filter H = dc L + hf (1 - L), L = (1 - s) / (1 - s z^-1),
    fitted to the loop gains g_k = exp(-1 / (f0 tau_k)) with weights w_k = 1 / (g_k - 1)^4 (the first-order
    error in decay time), |H| <= 1."""
    from scipy.optimize import least_squares
    g = np.exp(-1.0 / (f0 * np.asarray(tau, float)))
    w = 1.0 / np.maximum(1.0 - g, 1e-7) ** 2                 # sqrt of the weight, for least_squares
    z = np.exp(-1j * 2 * np.pi * np.asarray(fk, float) / sr)

    def mag(p):
        dc, hf, s = p
        Lz = (1 - s) / (1 - s * z)
        return np.abs(dc * Lz + hf * (1 - Lz))
    r = least_squares(lambda p: w * (mag(p) - g), x0=[g[0], g[-1], 0.5], bounds=([0, 0, 0], [1, 1, 0.99]))
    dc, hf, s = r.x
    return float(min(dc, 0.999999)), float(min(hf, dc)), float(s)


def loss_onepole(f0, sr, tau0, f_half):
    """(g, a1) of the loss filter g (1 + a1) / (1 + a1 z^-1) whose partials decay at sigma = c1 + c3 w^2 (w in rad per
    sample): c1 = 1 / tau0, the DC decay rate, and c3 = c1 / w_half^2 so the decay time halves at ``f_half``.
    Inverse of c1 = f0 (1 - g), c3 = -f0 a1 / (2 (a1 + 1)^2)."""
    c1 = 1.0 / float(tau0)
    c3 = c1 / (2 * math.pi * float(f_half) / sr) ** 2
    g = 1.0 - c1 / f0
    a1 = -1.0 - (f0 - math.sqrt(8.0 * c3 * f0 + f0 * f0)) / (4.0 * c3)
    return g, a1


def dispersion_allpass_from_partials(fk, sr, order=5):
    """(N, coefficients) of an integer delay N and ``order`` first-order allpasses (a + z^-1) / (1 + a z^-1) whose loop
    puts partial k (k = 1.. over ``fk``) at the measured frequency: -N theta_k + arg Hr(theta_k) = -2 pi k, solved by
    least squares on the phase delay, the best of a few N below the shortest period."""
    from scipy.optimize import least_squares
    fk = np.asarray(fk, float)
    k = np.arange(1, len(fk) + 1)
    th = 2 * np.pi * fk / sr
    top = int(math.floor(np.min(2 * np.pi * k / th))) - order
    best = None
    for N in range(top, top - 3 * order - 1, -max(1, order // 2)):
        def res(a, N=N):
            ph = -order * th + 2.0 * np.sum(np.arctan2(np.outer(a, np.sin(th)), 1.0 + np.outer(a, np.cos(th))), axis=0)
            return (-N * th + ph + 2 * np.pi * k) / th
        r = least_squares(res, x0=np.linspace(-0.9, 0.9, order), bounds=(-0.99, 0.99))
        if best is None or r.cost < best[0]:
            best = (r.cost, N, r.x)
    return best[1], tuple(float(c) for c in best[2])


GLIDE_EA_T = 766.0


def tension_glide_cents(amp_m, position, length_m, alpha=GLIDE_EA_T):
    """Pitch excess of a string plucked to ``amp_m`` at ``position`` (fraction of the length): the triangle's mean
    added tension gives df / f0 = alpha a^2 / (8 p (1 - p) L^2), alpha = EA / T0."""
    p = float(np.clip(position, 0.01, 0.99))
    return 1200.0 * math.log2(1.0 + alpha * amp_m ** 2 / (8.0 * p * (1.0 - p) * length_m ** 2))


def _pluck_string(freq, dur, sr, vel, *, material="steel", kind="pick", hardness=0.6, position=0.2, t60=3.0,
                  tail=1.2, release=0.1, wound=None, length=None, detune=0.0, mute=0.0, glide=0.0, measured=True,
                  worn=False, glide_mm=0.0, tau0=None, f_half=None):
    """One plucked note. `t60` is the mid-register ring time (higher strings ring shorter, as they
    do); `tail` seconds are rendered past note-off and `release` is the damper's time constant.
    `wound` = winding material for wound bass strings; `detune` (cents) adds the second string of a
    course; `mute` is palm damping at the bridge; `glide` (cents at full velocity) is the tension-modulation
    pitch excess at the pluck, which grows with the elongation, i.e. with velocity squared. `worn`: the measured
    losses of played-in steel strings. A fingertip (`kind` finger) touches FINGER_WIDTH of the string: its
    Gaussian contact filters the partials as exp(-(f / fc)^2), fc = c / (pi w), c the string's wave speed."""
    from .strings import DispersiveString, StringExciter, StringPhysicalProperties, WoundStringPhysicalProperties
    ring = float(dur) + float(tail)
    hard = float(np.clip(hardness * (0.55 + 0.45 * vel), 0.0, 1.0))
    t60 = float(t60) * _kt(freq, 1.0, 0.45)

    def one(f, pos):
        if wound:
            props = WoundStringPhysicalProperties.from_frequency(f, length_m=length, winding_material=wound)
            props.t60_s = t60
        else:
            props = StringPhysicalProperties.from_frequency(f, length_m=length, material=material, t60_s=t60)
        s = DispersiveString(props, sr=sr)
        table = material + "_worn" if worn and material + "_worn" in _WOODHOUSE else material
        if (measured and not wound and table in _WOODHOUSE) or not measured:
            from .graphsolver import FrequencyDependentLoss
            fk = f * np.arange(1, max(2, int(0.45 * sr / f)) + 1)
            if measured and not wound:
                tau = woodhouse_tau(f, fk, table)
            else:
                tau = plain_string_tau(fk, t60, props)
            dc, hf, sm = bank_loss_fit(f, fk, tau, sr)
            if tau0:
                g1, a1 = loss_onepole(f, sr, tau0, f_half or 2000.0)
                dc, hf, sm = g1, 0.0, -a1
            s.x_loss = FrequencyDependentLoss(dc, hf, sm)
            s.y_loss = FrequencyDependentLoss(dc ** 1.0002, hf * 0.999, sm)
            s.x_delay.delay = s.loop_delay(f)
            s.y_delay.delay = s.loop_delay(f * 2.0 ** (props.polarization_split_cents / 1200.0), plane="y")
        return s.render_exciter(ring, StringExciter(kind, hard, pos), velocity=1.0, palm_mute=mute,
                                glide_cents=(tension_glide_cents(glide_mm * 1e-3 * float(vel), pos, props.length_m) if glide_mm
                                             else float(glide) * float(vel) ** 2))

    y = one(freq, position)
    if detune:
        y = 0.54 * y + 0.46 * one(freq * 2 ** (detune / 1200.0), min(0.95, position + 0.012))
    if kind in ("finger", "flesh"):
        L = length if length is not None else float(np.clip(0.65 * 440.0 / max(freq, 1.0), 0.18, 1.65))
        y = finger_width_filter(y, sr, 2.0 * L * freq)
    n = y.shape[0]
    y = y * _off(n, dur, sr, release)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel)


HARP_PITCH_HZ = 250.0
HARP_TUNE_CAP = 12.0


def harp_stretch(freq):
    """Cents the harpist tunes partial 1 flat so that the stiff string's stretched partials (the heard pitch
    follows the partials near HARP_PITCH_HZ, at least the third) land on the note, capped at HARP_TUNE_CAP."""
    from .strings import StringPhysicalProperties
    B = StringPhysicalProperties.from_frequency(freq, material="nylon", t60_s=5.0).inharmonicity_B
    k = max(3.0, HARP_PITCH_HZ / freq)
    return min(HARP_TUNE_CAP, 600.0 * math.log2(1.0 + k * k * max(B, 0.0)))


@instrument("harp", "Concert harp: nylon string plucked with the fingertip near the middle, long ring.", family="plucked", span=("Cb1", "C7"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            position=(0.38, "pluck point along the string: 0.5 = middle (odd partials, full), 0.33 = no third partial, bright; small = presso la tavola"))
def harp(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0, position=0.38):
    # Meyer 2009, 3.5: three body resonances between 200 and 450 Hz, two more up to 850 Hz; low T60 4-6 s, middle 2 s
    # (t60 5 here, _kt takes it to 2.25 s on top; guitar string losses would ignore it). the exact peaks, Q, gains.
    freq = freq * 2.0 ** (-harp_stretch(freq) / 1200.0)
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="nylon", kind="finger", hardness=0.45, position=float(np.clip(position, 0.02, 0.5)),
                      t60=5.0, tail=2.0, release=0.35, measured=False)
    return _body(y, sr, [(230, 3.0, 2.5), (320, 3.0, 2.5), (430, 3.0, 2.5), (600, 3.0, 2.0), (820, 3.0, 2.0)], min(4200.0, 0.42 * sr)) * 0.842  # level match, belongs in _levels.py


@instrument("guitar", "Nylon-string classical guitar: warm, woody body.", family="plucked", span=("E2", "E5"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            body=(0, "which guitar: 0 the nominal body, any other number a different instrument (real spread of mode pitch and Q)"),
            body_size=(0.5, "0 small box .. 0.5 as built .. 1 big box (warps every body mode)"),
            radiation=(0, "1 = body modes weighted by their radiation efficiency, body radiating as a sphere (high-pass 250 Hz)"),
            air_q=(0.0, "Q of the air (Helmholtz) mode: 0 = 52 as measured on the body alone; 19-22 a whole played guitar"),
            pluck=("nail", "nail | finger (flesh: sticks about 20 ms, wide contact) | pick"),
            rattle=(0.0, "0..1 a screw or paper clip touching the string (prepared buzz); see also `cloth`"),
            cloth=(0.0, "0..1 cloth on the rattling stop: softer clicks"),
            glide_mm=(0.0, "pluck amplitude in mm at full velocity; sets the tension glide from the string (EA/T0 766): "
                           "2 mm about 7 cents, 4 mm about 28 (replaces `glide`)"))
def guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0, body=0, body_size=0.5, radiation=0, air_q=0.0, pluck="nail", rattle=0.0, cloth=0.0,
           glide_mm=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, glide_mm=glide_mm, material="nylon", kind="flesh" if pluck == "finger" else pluck, hardness=0.5, position=0.24, t60=3.0, tail=1.4, release=0.12)
    if rattle > 0:
        y = string_rattle(y, sr, gap=0.9 - 0.6 * float(np.clip(rattle, 0.0, 1.0)), cloth=cloth)
    return F.lowpass(guitar_body(y, sr, int(body), body_size, radiation=bool(radiation), air_q=air_q), min(3500.0, 0.42 * sr), sr, 0.7)


@instrument("steel_guitar", "Steel-string acoustic guitar: bright strum and fingerpicking (folk, country, pop).", family="plucked", span=("E2", "E6"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            body=(0, "which guitar: 0 the nominal body, any other number a different instrument (real spread of mode pitch and Q)"),
            body_size=(0.5, "0 small box .. 0.5 as built .. 1 big box (warps every body mode)"),
            worn=(0, "1 = played-in strings: measured losses of old steel strings, duller and shorter"),
            radiation=(0, "1 = body modes weighted by their radiation efficiency, body radiating as a sphere (high-pass 250 Hz)"),
            air_q=(0.0, "Q of the air (Helmholtz) mode: 0 = 52 as measured on the body alone; 19-22 a whole played guitar"))
def steel_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0, body=0, body_size=0.5, worn=0, radiation=0, air_q=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.7, position=0.16, t60=4.5, tail=1.6, release=0.12,
                      worn=bool(worn))
    return F.lowpass(guitar_body(y, sr, int(body), body_size, steel=True, radiation=bool(radiation), air_q=air_q), min(4600.0, 0.4 * sr), sr, 0.7)


@instrument("electric_guitar", "Clean electric guitar: round single-coil tone with long sustain (add `distortion` or `chorus` on the layer for rock or jangle).",
            family="plucked", span=("E2", "E6"), tone=(2800, "pickup tone Hz"),
            pickup=("linear", "linear | single | humbucker | rail (measured magnetic nonlinearity, Novak 2018)"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            pickup_pos=(0.0, "pickup comb: its place from the bridge as a part of the open string, Strat 0.063 bridge, 0.151 middle, 0.25 neck; 0 = off"),
            circuit=(0, "1 = the pickup's RLC resonance (2 H, 10 kOhm, 50 pF, 1 MOhm: 16 kHz) instead of the `tone` filter"),
            pickup_pos2=(0.0, "second pickup (same units as pickup_pos), mixed with the first; 0 = one pickup"),
            mix=("in", "in | out: the two pickups in phase (Strat positions 2 and 4) or out of phase (thin, nasal)"),
            stiff=(0, "1 = the pickup comb of a stiff string (notches move with the string's inharmonicity)"),
            amp=("none", "none | bassman: the amp's passive tone stack (treble, middle, bass)"),
            treble=(0.5, "tone stack treble 0..1"), middle=(0.5, "tone stack middle 0..1"), bass=(0.5, "tone stack bass 0..1"),
            horizontal=(0, "1 = the pickup also reads the horizontal plane (rectified: second harmonic), with `pickup` single|humbucker|rail"))
def electric_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=2800, pickup="linear", glide=0.0, pickup_pos=0.0, circuit=0,
                    pickup_pos2=0.0, mix="in", stiff=0, amp="none", treble=0.5, middle=0.5, bass=0.5, horizontal=0):
    from .strings import StringPhysicalProperties
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.6, position=0.2, t60=5.5, tail=1.2, release=0.08)
    if pickup != "linear":
        sw = 0.5 + 1.5 * float(np.clip(vel, 0.0, 1.0))
        yh = pickup_horizontal(y, sr, sw) if horizontal else 0.0
        y = magnetic_pickup(y, sr, pickup, swing_mm=sw) + yh
    opens = (82.41, 110.0, 146.83, 196.0, 246.94, 329.63)
    if pickup_pos > 0 and pickup_pos2 > 0:
        y = pickup_mix(y, sr, freq, pickup_pos, pickup_pos2, opens, phase=mix)
    elif stiff and pickup_pos > 0:
        y = pickup_comb_dispersive(y, sr, freq, pickup_pos, opens, StringPhysicalProperties.from_frequency(freq, material="steel").inharmonicity_B)
    else:
        y = pickup_comb(y, sr, freq, pickup_pos, opens)
    if amp != "none":
        y = tonestack(y, sr, treble, middle, bass, amp)
    if circuit:
        return pickup_circuit(F.highpass(y, 70, sr), sr)
    return F.lowpass(F.highpass(y, 70, sr), float(tone), sr, 1.3)


def pickup_comb(y, sr, freq, pos, open_strings, beta=1.0):
    """H_P(z) = 1 - beta z^(-2 T3 fs), Paiva, Pakarinen & Valimaki, Acoustics and modeling of pickups (JAES 2012), eqs 1-11.
    The delay comes from the open string, which is where the pickup sits: fretting does not move it. Positions: their
    Strat, 4.1 / 9.8 / 16.2 cm over 64.8 cm. beta (1 in their figures)."""
    if pos <= 0.0:
        return y
    fo = max((f for f in open_strings if f <= freq * 1.001), default=open_strings[0])
    d = float(pos) * sr / fo
    i = np.arange(y.size)
    return y - float(beta) * np.interp(i - d, i, y, left=0.0)


def pickup_circuit(y, sr, L=2.0, R=1.0e4, C=50e-12, Rl=1.0e6):
    """Hc(s) = 1 / ((1 + R/Rl) + s (L/Rl + RC) + s^2 LC) by the bilinear transform: Paiva et al. 2012, eqs 30, 34-37; defaults
    their base pickup (16 kHz, +12 dB). Humbucker values not sourced: pass your own."""
    from scipy.signal import lfilter
    c0 = 1 + R / Rl + 2 * sr * L / Rl + 2 * sr * R * C + 4 * sr * sr * L * C
    c1 = 2 + 2 * R / Rl - 8 * sr * sr * L * C
    c2 = 1 + R / Rl - 2 * sr * L / Rl - 2 * sr * R * C + 4 * sr * sr * L * C
    return lfilter([1.0, 2.0, 1.0], [c0, c1, c2], y)


# Novak et al., DAFx-18 fig. 9: eq. 6 fitted to three Seymour Duncan pickups (A, L_eq mm, r_eq mm)
PICKUPS = {"single": (21.51e-3, 12.98, 2.77), "humbucker": (40.67e-3, 9.86, 1.34), "rail": (47.46e-3, 13.11, 1.88)}


PICKUP_HORIZ_R_MM = 172.0


def magnetic_pickup(y, sr, kind="single", gap_mm=3.0, swing_mm=2.0, horizontal=0.0):
    """String displacement ``y`` (any scale; its peak is set to ``swing_mm``) at ``gap_mm`` from the pole piece
    through Novak's static flux law phi(x) = A ((x + L) / cbrt(r^2 + (x + L)^2) - x / cbrt(r^2 + x^2)), RMS-matched
    to the input (the coil voltage is d phi / dt, eq. 4). Their measurement: d0 = 3 mm, 2 mm swing
    on SH-2N gives more than 20 harmonics; at 5 mm and beyond the three pickups sound alike and cleaner.
    The string models already output the signal a linear pickup gives, so the law is applied to it as a
    displacement and not differentiated: what is added is the curvature of phi, without a second +6 dB/oct
    tilt (deviation). ``horizontal``: the string also swings across the pole, in quadrature (an elliptical orbit),
    `horizontal` x the vertical swing; across, the field is a bell, phi_h = exp(-(x_h / PICKUP_HORIZ_R_MM)^2), which
    a symmetric swing rectifies (even harmonics only)."""
    if kind not in PICKUPS:
        raise ValueError(f"unknown pickup {kind!r}; choose from {['linear', *PICKUPS]}")
    A, L, r = PICKUPS[kind]
    x = gap_mm + swing_mm * y / (np.max(np.abs(y)) + 1e-12)
    phi = A * ((x + L) / np.cbrt(r * r + (x + L) ** 2) - x / np.cbrt(r * r + x * x))
    if horizontal > 0.0:
        from scipy.signal import hilbert
        xh = float(horizontal) * swing_mm * np.imag(hilbert(y)) / (np.max(np.abs(y)) + 1e-12)
        phi = phi * np.exp(-(xh / PICKUP_HORIZ_R_MM) ** 2)
    u = -(phi - np.mean(phi))                     # flux falls as the string moves away
    return u * (np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(u ** 2)) + 1e-12))


@instrument("banjo", "Banjo: twangy string over a drum head, quick decay (bluegrass, country, folk).", family="plucked", span=("C3", "C7"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def banjo(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.9, position=0.08, t60=1.0, tail=0.6, release=0.06)
    y = y + 0.6 * F.bandpass(y, 600, sr, 2.0)          # the head
    return F.lowpass(F.highpass(y, 150, sr), min(4500.0, 0.42 * sr), sr, 0.7) * 0.8


@instrument("mandolin", "Mandolin: paired steel strings; set `tremolo` to re-pick held notes (Italian, bluegrass, folk).",
            family="plucked", span=("G3", "E7"), tremolo=(0.0, "picks per second on held notes (12 = classic tremolo, 0 = off)"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def mandolin(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.0, glide=0.0):
    def one(d, v):
        return _pluck_string(freq, d, sr, v, glide=glide, material="steel", kind="pick", hardness=0.8, position=0.12, t60=1.4, tail=0.7,
                             release=0.05, detune=5.0)
    if tremolo > 0 and dur * tremolo >= 1.5:
        step = 1.0 / tremolo
        y = mix([(one(step, vel * (1.0 if i % 2 == 0 else 0.82)), i * step) for i in range(int(dur * tremolo))], sr)
    else:
        y = one(dur, vel)
    return F.lowpass(F.highpass(y, 150, sr), min(4500.0, 0.42 * sr), sr, 0.7)


@instrument("koto", "Koto: string plucked hard near the bridge with ivory picks (Japanese, East Asian).", family="plucked", span=("C3", "C6"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def koto(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="nylon", kind="pick", hardness=0.85, position=0.07, t60=2.6, tail=1.4, release=0.15)
    return _body(y, sr, [(300, 1.5, 3.0)], min(4200.0, 0.42 * sr))


@instrument("sitar", "Sitar: long string over a buzzing bridge, the buzz blooms just after the pluck (Indian classical, psychedelic).",
            family="plucked", span=("C3", "C5"), buzz=(0.45, "0..1 amount of bridge buzz"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            model=("waveguide", "waveguide | fd (Siddiq 2012: a stiff string striking its curved flat bridge; the buzz "
                                "grows with velocity; glide ignored)"))
def sitar(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.45, glide=0.0, model="waveguide"):
    if model == "fd":
        from .fdstring import sitar_fd
        return sitar_fd(freq, dur, sr, vel, buzz)
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.8, position=0.06, t60=3.5, tail=1.6, release=0.2)
    t = np.arange(y.shape[0]) / sr
    b = F.peak(F.highpass(np.tanh(6.0 * y), 700, sr), 2400, sr, 1.2, 4.0) * (1 - np.exp(-t / 0.05)) * np.exp(-t / 0.9)
    return F.lowpass(y + float(np.clip(buzz, 0, 1)) * b, 4200, sr, 0.7, order=2) * 0.8


@instrument("pizzicato", "Pizzicato strings: short, woody finger-plucked violin/cello notes (playful, sneaky, classical).",
            family="plucked", span=("E1", "E6"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            size=("violin", "violin | viola | cello | bass: the instrument plucked (ring and body)"),
            vibrato=(0.0, "0..1 finger vibrato: lively, and half the ring (Meyer 2009, 3.3.1)"))
def pizzicato(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0, size="violin", vibrato=0.0):
    # Audible decay pp .. ff, range midpoints (Meyer 2009, 3.3.1-4): violin 40-150 / 350-800 ms, viola 50-150 / 280-600,
    # cello 50-200 / 400-1400, bass up to 1600 at ff (pp the cello's). Air modes: 275, 220, 110, 63 Hz. Audible
    # decay read as T60 . Body gains and the violin's upper peaks as before; 290 Hz scaled to each air mode.
    pp, ff, air = {"violin": (0.095, 0.575, 290.0), "viola": (0.10, 0.44, 220.0), "cello": (0.125, 0.90, 110.0), "bass": (0.125, 1.6, 63.0)}[size]
    v = float(np.clip(vel, 0.0, 1.0))
    wob = float(np.clip(vibrato, 0.0, 1.0))
    t60 = (pp * (ff / pp) ** v) * (1.0 - 0.5 * min(wob * 4.0, 1.0)) / _kt(freq, 1.0, 0.45)
    y = _pluck_string(freq, max(min(dur, 0.25), t60), sr, vel, glide=glide, material="gut", kind="finger", hardness=0.3, position=0.25, t60=t60,
                      tail=max(0.4, 1.2 * t60), release=0.08, measured=False)
    if wob > 0:                                                     # +-25 cents at 5.8 Hz as the violin voice's vibrato
        tt = np.arange(y.size) / sr
        df = (2.0 ** (0.25 * wob * np.sin(2 * np.pi * 5.8 * tt) / 12.0) - 1.0)
        y = np.interp(tt + np.cumsum(df) / sr, tt, y)
    return _body(y, sr, [(air, 1.5, 3.0), (air * 1.59, 1.5, 3.0), (air * 3.45, 1.0, 2.0)], min(3200.0, 0.4 * sr)) * 1.197  # level match, belongs in _levels.py


@instrument("upright_bass", "Upright (double) bass, plucked: deep woody thump with a short singing tail (jazz walking lines, rockabilly, folk).",
            family="bass", span=("E1", "G3"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def upright_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, kind="finger", hardness=0.25, position=0.3, t60=2.0, tail=0.8, release=0.1,
                      wound="nickel", length=float(np.clip(0.86 * 110 / max(freq, 20), 0.70, 1.08)))
    n = y.shape[0]
    thump = F.lowpass(O.white(n, seed=41), 300, sr) * decay_env(n, 0.012, sr) * 0.5 * vel
    return _body(y + thump, sr, [(100, 1.2, 3.0), (210, 1.5, 2.0)], 1400)


@instrument("finger_bass", "Electric bass guitar, fingerstyle: round, even, sustaining (rock, pop, soul, reggae).",
            family="bass", span=("E1", "G3"), tone=(1700, "pickup tone Hz"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            pickup_pos=(0.0, "pickup comb: its place from the bridge as a part of the open string; 0 = off"))
def finger_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=1700, glide=0.0, pickup_pos=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, kind="finger", hardness=0.4, position=0.28, t60=4.0, tail=0.6, release=0.07,
                      wound="nickel", length=float(np.clip(0.864 * 110 / max(freq, 20), 0.72, 0.92)))
    y = pickup_comb(y, sr, freq, pickup_pos, (41.2, 55.0, 73.42, 98.0))
    return F.lowpass(np.tanh(1.5 * y) / 0.9, float(tone), sr, 1.0)


@instrument("slap_bass", "Slapped electric bass: percussive pop on top of a scooped, punchy low end (funk, disco).",
            family="bass", span=("E1", "G4"))
def slap_bass(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .strings import DispersiveString, WoundStringPhysicalProperties
    props = WoundStringPhysicalProperties.from_frequency(freq, length_m=float(np.clip(0.864 * 110 / max(freq, 20), 0.5, 0.92)))
    props.t60_s = 2.6 * _kt(freq, 1.0, 0.45)
    y = DispersiveString(props, sr=sr).render_slap(dur + 0.5, velocity=vel)     # thumb strike + string hitting the frets
    y = y * _off(y.shape[0], dur, sr, 0.05)
    y = F.peak(y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel), 500, sr, 1.0, -5.0)
    return F.lowpass(F.highpass(y, 35, sr), min(4000.0, 0.4 * sr), sr, 0.7)


# String section and the other flue voices (2026-10-10) -------------------------------------------------
_SECTION_SEED = 1729


@instrument("strings", "String section: individual bowed players with their own vibrato, a few cents and a few milliseconds apart, "
            "centred on the written pitch. Short notes are bowed short.",
            family="bowed", span=("C2", "C6"), attack=(0.12, "s of bow attack on a long note (a short note takes 40 % of its length)"),
            release=(0.3, "s"), players=(6, "2..12 players"),
            scatter=(0.0, "cents: standard deviation of the players' mean pitch (Ternstrom: 0-5 preferred, 14 tolerable); above 0 the section beats slowly"),
            vibrato=(0.12, "0 = none .. 2 = wide; each player has their own, so more than about 0.2 makes the section's level heave"),
            level=("matched", "matched = about the same whatever `players` | section = +3 dB per doubling of the desks (Meyer 2009), same level at 6 players"),
            sordino=(0, "1 = muted section (about -6 dB, softer top)"))
def strings(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.12, release=0.3, players=6, scatter=0.0, vibrato=0.12, level="matched", sordino=0):
    """A section is its players, not one detuned oscillator. This voice was six sawtooth waves spread a quarter of
    a semitone ("supersaw") under a 0.25 s attack: it measured 14-16 cents off the written pitch with a 20-32 %
    wobble in loudness, and could not speak a sixteenth note.

    Each player is the library's solo bowed voice (harmonics through the instrument's body formants). Between
    players: the mean pitch differs by `scatter` cents standard deviation (Ternstrom 1993, pitch scatter in
    ensembles: listeners prefer 0-5 cents and tolerate 14), with the section's mean on the written pitch; onsets
    differ by 20 ms standard deviation (the timing spread reported for string quartet players); each has its own
    vibrato rate (5.2-6.4 Hz, inside the 5-8 Hz of string players, Meyer 2009, 2.3.6) and onset delay. The same section plays every note.

    Defaults are set by measurement of the level of a held note (standard deviation over mean): independent
    vibratos at 0.6 moved it by 30 % at A4 and 50 % at A5, a 3 cent scatter by 21-44 %; at vibrato 0.12 and no
    scatter it is 1 % at C3, 6 % at A4 and 12 % at A5."""
    rng = np.random.default_rng(_SECTION_SEED)
    n_pl = int(np.clip(players, 2, 12))
    cents = rng.normal(0.0, 1.0, n_pl)
    cents = (cents - cents.mean()) / (cents.std() + 1e-9) * float(np.clip(scatter, 0.0, 30.0))
    late = np.abs(rng.normal(0.0, 0.020, n_pl))
    late -= late.min()
    rates, delays = rng.uniform(5.2, 6.4, n_pl), rng.uniform(0.12, 0.30, n_pl)
    att = float(min(attack, 0.4 * dur))
    rel = float(release if dur >= 0.25 else min(release, 0.12))
    body = CELLO_BODY if freq < 196.0 else VIOLIN_BODY              # below the violin's open G the cellos carry the line
    corner = (1500.0, 2400.0) if freq < 196.0 else (2000.0, 3000.0)
    out = np.zeros(_n(dur + float(late.max()), sr, rel) + 8)
    solo = None
    for i in range(n_pl):
        v = _blown(freq * 2.0 ** (cents[i] / 1200.0), dur, sr, vel,
                   dict(tilt=1.8, formants=body, corner=corner[0], slope=3.0), dict(tilt=1.2, formants=body, corner=corner[1], slope=3.0),
                   attack=att, release=rel, bloom=max(att, 0.03), vib=(float(rates[i]), 0.12 * vibrato, float(delays[i])), top=4200.0)
        solo = v if solo is None else solo
        k = int(late[i] * sr)
        out[k:k + len(v)] += v[:len(out) - k]
    y = F.lowpass(out, min(4200.0, 0.44 * sr), sr, 0.7)
    if level == "section":
        # the players keep nearly the same phase, so the sum is a matter of chance at one note (-12 dB from 2 to 12 players
        # at A4): set the power one player gives times the desks, 3 dB per doubling, as measured by Meyer
        solo = F.lowpass(solo, min(4200.0, 0.44 * sr), sr, 0.7)
        y = y * np.sqrt(np.mean(solo ** 2) * n_pl / (np.mean(y ** 2) + 1e-20)) * (0.9 / np.sqrt(6.0))
    else:
        y = y * (0.9 / np.sqrt(n_pl))
    return F.lowpass(y, 2500.0, sr, 0.7) * 0.5 if sordino else y


def _pipe_air(n, sr, freq, env, odd, seed):
    """Breath of a blown pipe: noise through the pipe's own resonances, following the square of the blowing envelope
    (turbulence power goes as the jet's dynamic pressure). `odd`: a stopped pipe, odd modes only."""
    w = O.white(n, seed=seed)
    y = np.zeros(n)
    for j, h in enumerate((1, 3, 5) if odd else (1, 2, 3)):
        if h * freq < 0.45 * sr:
            y += F.bandpass(w, h * freq, sr, 9.0) / (1.0 + j)
    return y * env * env


@instrument("pan_flute", "Pan flute: a stopped pipe blown across its end. Odd harmonics, a breath that sits in the pipe's own resonances, a soft chiff.",
            family="wind", span=("C4", "C7"), breath=(0.3, "air 0..1"), vibrato=(0.0, "0 = steady .. 1"))
def pan_flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.3, vibrato=0.0):
    """There is no sourced jet model of a stopped pipe in the library (STK's flute is an open pipe), so the tone stays
    a harmonic stack with the stopped pipe's odd series. What changed: the vibrato is a parameter and off by default
    (it was fixed, and a near-sine with vibrato under a reverb flutters), and the breath is no longer a broad band
    of noise laid over the note: it passes through the pipe's odd modes and follows the blowing pressure squared."""
    rel = 0.15
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=4.5, depth=0.06 * float(np.clip(vibrato, 0, 2)), delay=0.3, ramp=0.4, scoop=0.4) if vibrato > 0 else freq
    amps = np.array([1.0, 0.06, 0.18, 0.02, 0.05]) * _roll(np.arange(1, 6) * freq, 2400.0) / float(_roll(freq, 2400.0))
    x = _harmonics(f, n, sr, amps)
    env = adsr(dur, 0.05, 0.1, 0.8, rel, sr)[:n]
    env = np.pad(env, (0, n - len(env)))
    air = _pipe_air(n, sr, freq, env, True, 56) * breath * 0.6
    chiff = _air(n, sr, _cut(freq, 2, 800, 2400), 1.2, 58) * decay_env(n, 0.035, sr) * breath * 1.2
    return (x * env + air + chiff * env) * 0.6 * (0.6 + 0.4 * vel)


@instrument("whistle", "Human whistle: a near-pure tone that slides into each note, with a little breath in its own resonance.",
            family="wind", span=("C5", "C7"), vibrato=(0.3, "0 = steady .. 1"))
def whistle(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.3):
    rel = 0.1
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.8, depth=0.12 * float(np.clip(vibrato, 0, 2)), delay=0.2, ramp=0.3, scoop=0.7)
    x = O.sine(f, n, sr) + 0.04 * O.sine(f * 2, n, sr)
    env = adsr(dur, 0.04, 0.1, 0.85, rel, sr)[:n]
    env = np.pad(env, (0, n - len(env)))
    return (x * env + _pipe_air(n, sr, freq, env, False, 59) * 0.2) * 0.6 * (0.6 + 0.4 * vel)


# ---------------------------------------------------------------- string physics helpers (V4)
def string_air_rate(f, props) -> np.ndarray:
    """Amplitude decay rate (1/s) that air viscosity alone gives the partials ``f`` of a string (Stokes drag on a
    cylinder), from its outer diameter and its mean density mu / (pi d^2 / 4)."""
    from .physical.bank_piano import air_rate
    r = getattr(props, "outer_radius_m", None) or props.radius_m
    d = 2.0 * float(r)
    rho = props.linear_density_kg_m / (math.pi * d * d / 4.0)
    return air_rate(np.asarray(f, float), d, rho)


def plain_string_tau(fk, t60, props) -> np.ndarray:
    """Amplitude decay times of the partials ``fk`` of a string without a measured loss table: the fundamental keeps
    ``t60``, the partials above add the Stokes air drag that grows with frequency (internal losses held flat)."""
    air = string_air_rate(fk, props)
    base = 6.91 / float(t60)
    return 1.0 / (max(base - float(air[0]), 0.5 * base) + air)


def finger_width_filter(y, sr, c, width=None):
    """A fingertip of ``width`` m (FINGER_WIDTH) plucking a string of wave speed ``c``: Gaussian contact, partial
    gains exp(-(f / fc)^2) with fc = c / (pi width)."""
    w = FINGER_WIDTH if width is None else float(width)
    fc = float(c) / (math.pi * max(w, 1e-4))
    n = len(y)
    m = 1 << int(math.ceil(math.log2(max(n, 2))))
    Y = np.fft.rfft(y, m)
    f = np.fft.rfftfreq(m, 1.0 / sr)
    return np.fft.irfft(Y * np.exp(-(f / fc) ** 2), m)[:n]


def pickup_mix(y, sr, freq, pos_a, pos_b, open_strings, beta=1.0, phase="in"):
    """Two pickups on one string, summed in or out of phase: each is e^{jwT} - beta e^{-jwT} (T the travel time from
    the pickup to the bridge on the open string), so the pair is
    H = 2 (e^{jw(Ta+Tb)/2} -+ beta e^{-jw(Ta+Tb)/2}) cos(w(Ta-Tb)/2) for in phase, its sine twin out of phase; made
    causal by the delay of the farther pickup."""
    fo = max((f for f in open_strings if f <= freq * 1.001), default=open_strings[0])
    ta, tb = 0.5 * float(pos_a) * sr / fo, 0.5 * float(pos_b) * sr / fo
    tm = max(ta, tb)
    i = np.arange(len(y), dtype=float)

    def one(t):
        return np.interp(i - (tm - t), i, y, left=0.0) - float(beta) * np.interp(i - (tm + t), i, y, left=0.0)
    return one(ta) + one(tb) if phase == "in" else one(ta) - one(tb)


def pickup_comb_dispersive(y, sr, freq, pos, open_strings, B, beta=1.0):
    """Pickup comb of a stiff string: the round trip to the bridge takes 2 T3 tau(w), tau = 1 / sqrt(1 + B (w/w0)^2),
    so |H|^2 = 1 - 2 beta cos(2 w T3 tau) + beta^2: the notches drift down from the harmonic grid as B grows."""
    if pos <= 0.0:
        return y
    fo = max((f for f in open_strings if f <= freq * 1.001), default=open_strings[0])
    t3 = 0.5 * float(pos) / fo
    n = len(y)
    m = 1 << int(math.ceil(math.log2(max(n, 2) + int(2 * t3 * sr) + 8)))
    f = np.fft.rfftfreq(m, 1.0 / sr)
    tau = 1.0 / np.sqrt(1.0 + float(B) * (f / fo) ** 2)
    H = 1.0 - float(beta) * np.exp(-2j * np.pi * f * 2.0 * t3 * tau)
    return np.fft.irfft(np.fft.rfft(y, m) * H, m)[:n]


TONESTACKS = {"bassman": (250e3, 1e6, 25e3, 56e3, 250e-12, 20e-9, 20e-9)}


def tonestack_coeffs(sr, t=0.5, m=0.5, l=0.5, amp="bassman"):
    """(b, a) of the guitar-amp tone stack (treble t, middle m, bass l, 0..1; bass pot taper l -> exp((l - 1) 3.4)):
    the third-order closed form of the passive RC network, by the bilinear transform."""
    from scipy.signal import bilinear
    R1, R2, R3, R4, C1, C2, C3 = TONESTACKS[amp]
    t, m = float(np.clip(t, 0, 1)), float(np.clip(m, 0, 1))
    l = math.exp((float(np.clip(l, 0, 1)) - 1.0) * 3.4)
    b1 = t * C1 * R1 + m * C3 * R3 + l * (C1 * R2 + C2 * R2) + (C1 * R3 + C2 * R3)
    b2 = (t * (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4) - m * m * (C1 * C3 * R3 * R3 + C2 * C3 * R3 * R3)
          + m * (C1 * C3 * R1 * R3 + C1 * C3 * R3 * R3 + C2 * C3 * R3 * R3) + l * (C1 * C2 * R1 * R2 + C1 * C2 * R2 * R4 + C1 * C3 * R2 * R4)
          + l * m * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3) + (C1 * C2 * R1 * R3 + C1 * C2 * R3 * R4 + C1 * C3 * R3 * R4))
    k = C1 * C2 * C3
    b3 = (l * m * k * (R1 * R2 * R3 + R2 * R3 * R4) - m * m * k * (R1 * R3 * R3 + R3 * R3 * R4) + m * k * (R1 * R3 * R3 + R3 * R3 * R4)
          + t * k * R1 * R3 * R4 - t * m * k * R1 * R3 * R4 + t * l * k * R1 * R2 * R4)
    a1 = (C1 * R1 + C1 * R3 + C2 * R3 + C2 * R4 + C3 * R4) + m * C3 * R3 + l * (C1 * R2 + C2 * R2)
    a2 = (m * (C1 * C3 * R1 * R3 - C2 * C3 * R3 * R4 + C1 * C3 * R3 * R3 + C2 * C3 * R3 * R3) + l * m * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3)
          - m * m * (C1 * C3 * R3 * R3 + C2 * C3 * R3 * R3) + l * (C1 * C2 * R2 * R4 + C1 * C2 * R1 * R2 + C1 * C3 * R2 * R4 + C2 * C3 * R2 * R4)
          + (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4 + C1 * C2 * R3 * R4 + C1 * C2 * R1 * R3 + C1 * C3 * R3 * R4 + C2 * C3 * R3 * R4))
    a3 = (l * m * k * (R1 * R2 * R3 + R2 * R3 * R4) - m * m * k * (R1 * R3 * R3 + R3 * R3 * R4)
          + m * k * (R3 * R3 * R4 + R1 * R3 * R3 - R1 * R3 * R4) + l * k * R1 * R2 * R4 + k * R1 * R3 * R4)
    return bilinear([b3, b2, b1, 0.0], [a3, a2, a1, 1.0], sr)


def tonestack(y, sr, t=0.5, m=0.5, l=0.5, amp="bassman"):
    from scipy.signal import lfilter
    b, a = tonestack_coeffs(sr, t, m, l, amp)
    return lfilter(b, a, y)


# radiation efficiency of the guitar's low body modes (Derveaux's model): Hz, %
GUITAR_EFFICIENCY = ((200.0, 40.8), (272.0, 3.96), (289.0, 0.9), (372.0, 8.8), (437.0, 7.8), (508.0, 4.09))


def guitar_radiation(y, sr, f, g):
    """Body mode gains ``g`` at ``f`` weighted by the measured radiation efficiency (amplitude ~ sqrt of the power
    share, nearest tabulated mode within 10 %), and the radiated pressure of a pulsating sphere: first-order high-pass
    at 250 Hz on the whole body output."""
    g = np.array(g, float)
    fe = np.array([a for a, _ in GUITAR_EFFICIENCY])
    ee = np.array([b for _, b in GUITAR_EFFICIENCY])
    for i, fk in enumerate(f):
        j = int(np.argmin(np.abs(fe - fk)))
        if abs(fe[j] - fk) < 0.1 * fe[j]:
            g[i] *= math.sqrt(ee[j] / ee.mean())
    return g


def harpsichord_pluck_point(freq: float) -> float:
    """Plucking point / string length by register: 12 % in the bass to about 50 % in the treble, log in frequency."""
    return _kt(freq, 0.12, 0.5, f_lo=65.0, f_hi=1200.0)


def harpsichord_long_freq(freq: float) -> float:
    """First longitudinal mode Hz of a brass/iron harpsichord string sounding ``freq``: c_l / (2 L), L from the
    transverse tuning at a scaling of ~0.35 m at C5 (L ~ 1 / f, shortened in the bass), c_l = sqrt(E / rho) of iron."""
    L = float(np.clip(0.35 * 523.25 / max(freq, 1.0), 0.12, 2.0))
    return math.sqrt(2.0e11 / 7850.0) / (2.0 * L)


def phantom_partials(y, sr, f_long, n_res=3, q=200.0):
    """Economic longitudinal model: the square of the transverse motion (the tension change drives the longitudinal
    wave) excites 2-4 resonators at the longitudinal modes k f_long; output RMS 0.1 of the input's (-20 dB, where the
    longitudinal part becomes audible), to be added to the transverse signal."""
    from scipy.signal import lfilter
    d = y * y
    d = d - np.mean(d)
    out = np.zeros_like(y)
    for k in range(1, int(np.clip(n_res, 2, 4)) + 1):
        fk = k * f_long
        if fk >= 0.45 * sr:
            break
        r = math.exp(-math.pi * fk / (q * sr))
        out += lfilter([1.0 - r, 0.0, -(1.0 - r)], [1.0, -2 * r * math.cos(2 * math.pi * fk / sr), r * r], d)
    return out * 0.1 * np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(out ** 2)) + 1e-12)


@instrument("kantele", "Kantele (Finnish zither): steel strings knotted to a tuning bar; the knot's tension modulation adds "
            "a second harmonic that swells after the pluck.", family="plucked", span=("D3", "D5"),
            knot=(1.0, "0..2 x the second-order nonlinearity of the knotted termination"))
def kantele(freq, dur, sr=DEFAULT_SR, vel=1.0, knot=1.0):
    """Plucked steel string; the termination's second-order nonlinearity: KANTELE_K y^2 low-passed at 2 f0 and added
    (a squared string signal is the double-frequency component of the tension change)."""
    y = _pluck_string(freq, dur, sr, vel, material="steel", kind="finger", hardness=0.5, position=0.2, t60=4.0, tail=2.0, release=0.3)
    sq = F.lowpass(y * y, min(2.5 * freq, 0.4 * sr), sr, 0.7, order=2)
    y = y + float(np.clip(knot, 0.0, 2.0)) * KANTELE_K * float(np.clip(vel, 0.0, 1.0)) * F.dc_block(sq, sr)
    return _body(y, sr, [(180.0, 1.5, 4.0), (420.0, 2.0, 3.0)], min(5000.0, 0.42 * sr)) * 0.8


KANTELE_K = 1.5


@instrument("string_scrape", "Finger or pick sliding along a wound string: a 'picket fence' of clicks, one per winding turn, "
            "pitched at speed / winding pitch (the guitar fret-hand squeak).", family="plucked", span=("E2", "E4"),
            speed=(0.4, "m/s of the slide (0.1 slow squeak .. 1.5 fast zip)"), pitch_mm=(0.4, "winding pitch mm (wire diameter)"))
def string_scrape(freq, dur, sr=DEFAULT_SR, vel=1.0, speed=0.4, pitch_mm=0.4):
    """Impulses at V / dx (dx the winding pitch) through the string's own modes: the clicks are filtered by the
    wound string sounding ``freq``."""
    n = _n(dur, sr, 0.3)
    t = np.arange(n) / sr
    v = float(speed) * np.minimum(1.0, t / 0.03) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.03))
    ph = np.cumsum(v / (float(pitch_mm) * 1e-3)) / sr
    clicks = np.diff(np.floor(ph), prepend=0.0)
    clicks = clicks * (0.6 + 0.4 * np.random.default_rng(int(freq) % 9973).random(n))
    from .strings import DispersiveString, WoundStringPhysicalProperties
    s = DispersiveString(WoundStringPhysicalProperties.from_frequency(freq), sr=sr)
    body = np.array([s.step(0.0, excitation=0.05 * c) for c in clicks])
    y = F.highpass(0.5 * body + F.bandpass(clicks, min(float(speed) / (float(pitch_mm) * 1e-3), 0.4 * sr), sr, 3.0), 200.0, sr)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.4 + 0.5 * vel)


@numba.njit(cache=True)
def _coupled_strings(exc, starts, delays, loop_gain, couple, cpole, n, obstacle):
    """Karplus-Strong strings (two-point average loop filter) summed at a bridge whose one-pole filtered sum is fed
    back to every string (STK Guitar coupling). ``obstacle`` > 0: one-sided bridge at that depth (jawari)."""
    k = len(delays)
    size = 0
    for d in delays:
        size = max(size, int(d) + 3)
    buf = np.zeros((k, size))
    last = np.zeros(k)
    w = 0
    cf = 0.0
    out = np.zeros(n)
    m = exc.shape[1]
    for i in range(n):
        s = 0.0
        vals = np.zeros(k)
        for j in range(k):
            d = delays[j]
            di = int(d)
            fr = d - di
            a = buf[j, (w - di) % size]
            b = buf[j, (w - di - 1) % size]
            v = (1.0 - fr) * a + fr * b
            y = loop_gain[j] * 0.5 * (v + last[j])
            last[j] = v
            if obstacle > 0.0 and y < -obstacle:
                y = -obstacle + 0.3 * (y + obstacle)
            vals[j] = y
            s += y
        cf = (1.0 - cpole) * s + cpole * cf
        fb = couple * cf / k
        for j in range(k):
            e = 0.0
            ii = i - starts[j]
            if 0 <= ii < m:
                e = exc[j, ii]
            buf[j, w] = vals[j] + fb + e
        out[i] = s
        w = (w + 1) % size
    return out


def _stk_pluck_noise(m, rng):
    """STK Guitar excitation without a body: noise, raised-cosine ends over 20 %, mean removed."""
    x = rng.uniform(-1.0, 1.0, m)
    r = int(0.2 * m)
    win = np.ones(m)
    win[:r] = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    win[m - r:] = win[:r][::-1]
    x = x * win
    return x - x.mean()


GUITAR_CHORD_COUPLING = 0.01
GUITAR_CHORDS = {"none": (), "maj": (0, 7, 12, 16, 19), "min": (0, 7, 12, 15, 19), "power": (0, 7, 12)}


@instrument("guitar_chord", "Guitar whose strings share one bridge: the plucked note sets the open strings ringing in "
            "sympathy, or strum a chord shape whose strings feed each other.", family="plucked", span=("E2", "E5"),
            chord=("none", "none (one note, the open strings ring along) | maj | min | power: strummed shape on the root"),
            strum=(0.015, "s between strings"), coupling=(1.0, "x the bridge coupling (0 = independent strings)"))
def guitar_chord(freq, dur, sr=DEFAULT_SR, vel=1.0, chord="none", strum=0.015, coupling=1.0):
    """STK Guitar: loop gain 0.995 + key tracking, bridge sum through a one-pole (pole 0.9) times
    GUITAR_CHORD_COUPLING (1.5 x at full velocity) fed back to each string, 200-sample noise plucks through a
    pick filter (pole 0.95). Without a chord the six open strings are in the bank, unplucked."""
    if chord not in GUITAR_CHORDS:
        raise ValueError(f"guitar_chord chord {chord!r}: {list(GUITAR_CHORDS)}")
    from scipy.signal import lfilter
    v = float(np.clip(vel, 0.0, 1.0))
    rng = np.random.default_rng(int(freq * 10) % 9973)
    fr = [freq * 2 ** (s / 12.0) for s in (0, *GUITAR_CHORDS[chord][1:])] if chord != "none" else [freq]
    played = len(fr)
    if chord == "none":
        fr += [f for f in (82.41, 110.0, 146.83, 196.0, 246.94, 329.63) if abs(1200 * math.log2(f / freq)) > 1]
    m = max(8, int(200 * sr / 44100))
    exc = np.zeros((len(fr), m))
    for j in range(played):
        exc[j] = lfilter([0.05], [1.0, -0.95], _stk_pluck_noise(m, rng)) * (0.3 + 0.7 * v) * 20.0
    starts = np.array([int(j * strum * sr) if j < played else 0 for j in range(len(fr))], np.int64)
    delays = np.array([sr / f - 0.5 for f in fr])
    gains = np.array([min(0.995 + f * 5e-6, 0.9995) for f in fr])
    n = _n(dur + 1.5, sr)
    y = _coupled_strings(exc, starts, delays, gains, GUITAR_CHORD_COUPLING * (1 + 0.5 * v) * float(coupling), 0.9, n, 0.0)
    y = y * _off(n, dur, sr, 0.15)
    return F.lowpass(guitar_body(y / (np.max(np.abs(y)) + 1e-9), sr), min(3500.0, 0.42 * sr), sr, 0.7) * (0.5 + 0.5 * v)


@instrument("tanpura", "Tanpura: four long strings droning Pa Sa Sa Sa (low) over a jawari bridge, plucked in turn; the thread "
            "under the strings makes them buzz and bloom (Indian classical drone).", family="plucked", span=("C2", "C4"),
            jawari=(0.6, "0..1 bridge buzz"), cycle=(2.4, "s for one round of the four strings"),
            first=("pa", "pa | ma | ni: the first string (Pa = fifth below Sa, Ma = fourth, Ni = major seventh)"))
def tanpura(freq, dur, sr=DEFAULT_SR, vel=1.0, jawari=0.6, cycle=2.4, first="pa"):
    """STK Drone loop per string: delay fs/f - 0.5, loop gain 0.997 + f 2e-6 (at most 0.99999), plus each pluck;
    the jawari a one-sided bridge clipping the loop below a depth that shrinks as ``jawari`` grows."""
    ratio = {"pa": 0.75, "ma": 2.0 / 3.0, "ni": 15.0 / 16.0}[first]
    fr = [freq * ratio, freq, freq, freq / 2.0]
    n = _n(dur + 3.0, sr)
    rng = np.random.default_rng(int(freq * 10) % 9973)
    m = int(0.004 * sr)
    step = float(cycle) / 4.0
    hits = int(dur / step) + 1
    exc = np.zeros((4, n))
    for h in range(hits):
        j = h % 4
        i0 = int(h * step * sr)
        if i0 + m < n:
            exc[j, i0:i0 + m] += _stk_pluck_noise(m, rng) * (0.4 + 0.6 * vel)
    delays = np.array([sr / f - 0.5 for f in fr])
    gains = np.array([min(0.997 + f * 2e-6, 0.99999) for f in fr])
    depth = 0.15 * (1.0 - float(np.clip(jawari, 0.0, 1.0))) + 0.01
    y = _coupled_strings(exc, np.zeros(4, np.int64), delays, gains, 0.0, 0.9, n, depth if jawari > 0 else 0.0)
    y = F.dc_block(y, sr) * _off(n, dur, sr, 0.8)
    y = _body(y, sr, [(150.0, 1.0, 3.0), (500.0, 1.5, 2.0)], min(6000.0, 0.42 * sr))
    return y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.4 * vel)


@numba.njit(cache=True)
def _rattling_foot(F_in, sr, m, w0, zeta, gap, kc):
    """Bridge foot: mass-spring driven by the string's bridge force; when it falls ``gap`` below rest it strikes the
    table (Hertz contact kc d^1.5). Returns the contact force. Sub-stepped at 176.4 kHz or more: the contact
    stiffness puts the foot's bounce near 5 kHz, which an explicit step at 22-44 kHz does not hold."""
    sub = max(1, int(math.ceil(176400.0 / sr)))
    dt = 1.0 / (sr * sub)
    x = 0.0
    v = 0.0
    out = np.zeros(len(F_in))
    for i in range(len(F_in)):
        acc = 0.0
        for _ in range(sub):
            fc = 0.0
            d = -x - gap
            if d > 0.0:
                fc = kc * d ** 1.5 * (1.0 - 0.5 * min(max(v, -1.0), 1.0))
            a = (F_in[i] + fc) / m - w0 * w0 * x - 2.0 * zeta * w0 * v
            v += a * dt
            x += v * dt
            acc += fc
        out[i] = acc / sub
    return out


TROMBA_FOOT = (0.005, 200.0, 0.05, 20e-6, 1e9)   # mass kg, Hz, damping ratio, gap m, contact N/m^1.5


def _bowed_os(freq, dur, sr, **kw):
    """``bow.bowed`` rendered fast enough that its stick-slip does not snap the period to whole samples (>= 350
    samples per period, 16x at most), the string modes kept under 0.45 sr, then decimated to ``sr``."""
    from scipy.signal import resample_poly
    from .physical import bow
    os_ = int(min(16, max(1, math.ceil(350.0 * freq / sr))))
    y = np.asarray(bow.bowed(freq, dur, float(sr * os_), max_hz=0.45 * sr, **kw), float)
    return resample_poly(y, 1, os_) if os_ > 1 else y


def _acf_f0(y, sr, freq, span=0.06):
    """Pitch of ``y`` from the Hann-windowed autocorrelation peak within +-span of the period of ``freq``."""
    y = y - y.mean()
    if y.size < 64 or np.max(np.abs(y)) < 1e-12:
        return 0.0
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(y.size), 4 * y.size)) ** 2)
    p = sr / freq * 4
    lo = int(p * (1.0 - span))
    k = lo + int(np.argmax(ac[lo:int(p * (1.0 + span)) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return sr * 4 / k


@functools.lru_cache(maxsize=128)
def _bow_tuning(freq, sr, force, speed, accel):
    """Sounding / asked frequency of ``bow.bowed``: its cello string stopped short is stiff, and the stretched
    partials pull the Helmholtz period sharp (5-20 cents from C4 to C6). Fixed point over up to three renders,
    each measured on 0.05-0.45 s."""
    ask = float(freq)
    for _ in range(3):
        got = _acf_f0(_bowed_os(ask, 0.5, sr, force=force, speed=speed, accel=accel)[int(0.05 * sr):int(0.45 * sr)], sr, freq)
        if got <= 0.0 or abs(math.log2(got / freq)) * 1200 < 3.0:
            break
        ask = float(np.clip(ask * freq / got, freq * 0.94, freq * 1.06))
    return freq / ask


@instrument("tromba_marina", "Tromba marina: one long bowed string over a bridge whose loose foot drums on the soundboard "
            "every period, a brassy trumpet-like buzz.", family="bowed", span=("C2", "C5"),
            gap=(1.0, "x the 20 micron rest gap of the foot (smaller = more buzz)"))
def tromba_marina(freq, dur, sr=DEFAULT_SR, vel=1.0, gap=1.0):
    v = float(np.clip(vel, 0.0, 1.0))
    fs, bw = max(freq, 98.0), (0.6 + 1.2 * v, 0.1 + 0.2 * v, 0.35 + 1.7 * v)
    f = _bowed_os(fs / _bow_tuning(round(fs, 2), int(sr), *bw), dur, int(sr), force=bw[0], speed=bw[1], accel=bw[2])
    if freq < 98.0:
        f = np.interp(np.arange(len(f)) * freq / 98.0, np.arange(len(f)), f)
    m, fb, z, g0, kc = TROMBA_FOOT
    foot = _rattling_foot(np.asarray(f, float), float(sr), m, 2 * math.pi * fb, z, g0 * float(max(gap, 0.05)), kc)
    y = f / (np.max(np.abs(f)) + 1e-12) + 0.5 * foot / (np.max(np.abs(foot)) + 1e-12)
    y = _body(y, sr, [(140.0, 1.5, 4.0), (900.0, 1.0, 3.0)], min(5000.0, 0.42 * sr))
    return _physical_note_fade(y / (np.max(np.abs(y)) + 1e-12), sr, 0.01, 0.1) * (0.5 + 0.4 * v)


@instrument("bowed_psaltery", "Bowed psaltery: triangular box of metal strings bowed one by one; no vibrato, airy and glassy.",
            family="bowed", span=("C4", "C7"))
def bowed_psaltery(freq, dur, sr=DEFAULT_SR, vel=1.0):
    """Bowed string through the violin body, no vibrato, bow pressure 0.9, bow speed 0.03 + 0.2 amplitude, bow noise
    0.12, and a 6 % Ornstein-Uhlenbeck wander of the bowing level."""
    v = float(np.clip(vel, 0.0, 1.0))
    fs, bw = max(freq, 98.0), (0.9 * 1.5, 0.03 + 0.2 * v, 1.0)
    y = _bowed_os(fs / _bow_tuning(round(fs, 2), int(sr), *bw), dur, int(sr), force=bw[0], speed=bw[1], accel=bw[2])
    if freq > 98.0:
        y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(int(freq) % 9973)
    ou = np.zeros(n)
    a = math.exp(-1.0 / (0.2 * sr))
    e = rng.standard_normal(n) * math.sqrt(1 - a * a) * 0.06
    for i in range(1, n):
        ou[i] = a * ou[i - 1] + e[i]
    y = y / (np.max(np.abs(y)) + 1e-12) * (1.0 + ou) + 0.12 * F.bandpass(rng.standard_normal(n), 3000.0, sr, 0.7) * np.minimum(1.0, np.arange(n) / (0.05 * sr))
    y = _body(y, sr, [(fc, fc / bw, 4.0 * g) for fc, bw, g in VIOLIN_BODY], min(6000.0, 0.44 * sr))
    return _physical_note_fade(y / (np.max(np.abs(y)) + 1e-12), sr, 0.03, 0.15) * (0.5 + 0.4 * v)


def string_rattle(y, sr, gap=0.6, cloth=0.0, k=1.0):
    """One-sided intercept against the string (screw, folded washer, paper clip): where the normalised displacement
    passes ``gap`` (of its peak) the stop pushes back as k d^1.5 and clicks; ``cloth`` 0..1 is a felt or cloth on the
    stop that softens the clicks (low-pass, less stiffness). Returns the string plus the buzz."""
    x = y / (np.max(np.abs(y)) + 1e-12)
    d = np.maximum(x - float(gap), 0.0)
    f = float(k) * (1.0 - 0.7 * float(np.clip(cloth, 0.0, 1.0))) * d ** 1.5
    buzz = np.diff(f, prepend=0.0) * sr / 1000.0
    if cloth > 0:
        buzz = F.lowpass(buzz, 6000.0 * (1.0 - 0.8 * float(cloth)) + 300.0, sr, 0.7)
    return y + np.max(np.abs(y)) * buzz


PICKUP_H_BELL = 0.172               # m: horizontal flux bell exp(-(x/r)^2), 0.794 -> 0.788 T over 15 mm
PLANE_H = 0.3                       # share of the swing in the horizontal plane after a pick stroke


def pickup_horizontal(y, sr, swing_mm=2.0):
    """The horizontal plane over a pole piece: flux A exp(-(x_h / r)^2) is even in x_h, so the coil reads the
    rectified motion (2 f only); returned at its level relative to the vertical reading (dphi/dx at the gap)."""
    xh = PLANE_H * swing_mm * 1e-3 * y / (np.max(np.abs(y)) + 1e-12)
    phi = np.exp(-(xh / PICKUP_H_BELL) ** 2)
    u = -np.diff(phi, prepend=phi[0]) * sr
    lin = np.diff(swing_mm * 1e-3 * y / (np.max(np.abs(y)) + 1e-12), prepend=0.0) * sr / PICKUP_H_BELL
    return u * np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(lin ** 2)) + 1e-12)


@instrument("zanza", "Zanza (thumb piano, kit made of piano wire): steel tongues 100 to 43 mm over a cone, a buzzing "
            "rattle on each tongue (drone of the bottle caps).", family="mallet", span=("C3", "C6"),
            buzz=(0.5, "0..1 rattle on the tongue"), cone=(0.44, "m, size of the cone resonator (0.44 = 135 Hz)"))
def zanza(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.5, cone=0.44):
    """Clamped-free tongue (modes 1, 6.267, 17.55), thumb at its tip, a rattle on it, radiated by a cardboard cone
    resonating at 60 / size Hz (130-140 Hz for the kit's cone)."""
    from .friction import cone_radiator
    ring = max(dur, 1.5)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 0.9), (6.267, 0.12 * vel, 0.08), (17.55, 0.04 * vel, 0.02)], corner=3500.0)
    x = x * _off(n, dur, sr, 0.3) if dur < ring else x
    if buzz > 0:
        x = string_rattle(x, sr, gap=0.75 - 0.3 * float(np.clip(buzz, 0.0, 1.0)), cloth=0.2)
    y = cone_radiator(x, sr, size=float(cone), material="cardboard", mix=0.6)
    y = np.asarray(y, float)
    y = y.mean(axis=1) if y.ndim == 2 else y
    return y / (np.max(np.abs(y)) + 1e-9) * (0.45 + 0.45 * vel)


@instrument("baschet", "Baschet string sculpture (Katsuraphone): a string over a cylinder bridge, its two halves bent in "
            "opposite directions, radiated by a sheet of metal.", family="plucked", span=("G2", "G5"),
            bend=(30.0, "cents each half is bent, one up one down"), sheet=(0.5, "0..1 metal sheet radiator"),
            screw=(0.0, "0..1 buzzing screw on the string (odd harmonics)"))
def baschet(freq, dur, sr=DEFAULT_SR, vel=1.0, bend=30.0, sheet=0.5, screw=0.0):
    from .friction import sheet_radiator
    y1 = _pluck_string(freq, dur, sr, vel, material="steel", kind="pick", hardness=0.7, position=0.2, t60=10.0, tail=3.0, release=0.5)
    y2 = _pluck_string(freq, dur, sr, vel, material="steel", kind="pick", hardness=0.7, position=0.3, t60=10.0, tail=3.0, release=0.5)
    t = np.arange(len(y1)) / sr
    env = np.minimum(1.0, t / 0.3)
    out = []
    for yy, sgn in ((y1, 1.0), (y2, -1.0)):
        r = 2.0 ** (sgn * float(bend) * env / 1200.0)
        out.append(np.interp(np.cumsum(r) - r[0], np.arange(len(yy)), yy, right=0.0))
    y = 0.5 * (out[0] + out[1])
    if screw > 0:
        y = y + float(screw) * np.tanh(4.0 * y / (np.max(np.abs(y)) + 1e-12)) * np.max(np.abs(y)) * 0.3
    if sheet > 0:
        w = np.asarray(sheet_radiator(y, sr, mix=float(np.clip(sheet, 0.0, 1.0))), float)
        y = w.mean(axis=1) if w.ndim == 2 else w
    return y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.4 * vel)


# resonator guitar: the five lowest cone/body modes, f = fm + cm v (v the cone velocity, m/s)
RESO_MODES = ((83.2, -8.3), (95.7, -12.6), (128.8, -21.4), (161.2, -16.7), (177.5, -0.6))
RESO_TRACK = (0.1e-3, 0.1)          # attack, release s of the amplitude tracker
RESO_Q = 30.0
RESO_V = 0.05                       # m/s cone velocity at full scale


@numba.njit(cache=True)
def _reso_body(x, sr, f0, cm, q, ta, tr, vscale):
    n = len(x)
    out = np.zeros(n)
    k = len(f0)
    y1 = np.zeros(k)
    y2 = np.zeros(k)
    env = 0.0
    aa = math.exp(-1.0 / (ta * sr))
    ar = math.exp(-1.0 / (tr * sr))
    for i in range(n):
        a = abs(x[i])
        env = aa * env + (1 - aa) * a if a > env else ar * env + (1 - ar) * a
        v = env * vscale
        s = 0.0
        for j in range(k):
            f = max(f0[j] + cm[j] * v, 20.0)
            r = math.exp(-math.pi * f / (q * sr))
            yy = (1 - r) * x[i] + 2 * r * math.cos(2 * math.pi * f / sr) * y1[j] - r * r * y2[j]
            y2[j] = y1[j]
            y1[j] = yy
            s += yy
        out[i] = s
    return out


@instrument("resonator_guitar", "Resonator guitar (dobro): steel strings on an aluminium cone whose low modes sag in "
            "pitch when played hard; open G tuning G B D G B D, metallic and loud.", family="plucked", span=("G2", "G5"))
def resonator_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _pluck_string(freq, dur, sr, vel, material="steel", kind="pick", hardness=0.8, position=0.12, t60=3.0, tail=1.2, release=0.1)
    body = _reso_body(y, float(sr), np.array([a for a, _ in RESO_MODES]), np.array([b for _, b in RESO_MODES]), RESO_Q,
                      RESO_TRACK[0], RESO_TRACK[1], RESO_V / 0.5 * (0.5 + 0.5 * float(vel)))
    body *= np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(body ** 2)) + 1e-12)
    y = y + 0.7 * body
    y = F.peak(y, 1100.0, sr, 1.0, 5.0)                                 # the cone's nasal band
    return F.lowpass(y, min(5000.0, 0.42 * sr), sr, 0.7) / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel)


BOW_NOISE_DB = -40.0


def bowed_level(speed: float, beta: float) -> float:
    """Helmholtz amplitude ~ v_b / beta: 10-125 cm/s spans 22 dB, a sixfold change of the contact point 16 dB."""
    return float(speed) / max(float(beta), 1e-3)


def commuted_bowed(freq, dur, sr, vel, vibrato=1.0, body=None):
    """Commuted bowed string: the body's impulse response relaunched once per period (overlapping), the period
    following the vibrato, the relaunch slightly detuned between successive pulses (the bow's jitter), plus bow noise
    shaped by the body."""
    n = _n(dur, sr, 0.15)
    t = np.arange(n) / sr
    f = freq * 2.0 ** ((0.25 * float(vibrato) * np.sin(2 * math.pi * 5.8 * t) * np.minimum(1.0, t / 0.3)) / 12.0)
    rng = np.random.default_rng(int(freq) % 9973)
    ph = np.cumsum(f * (1.0 + 0.002 * rng.standard_normal(n))) / sr
    pulses = np.diff(np.floor(ph), prepend=0.0)
    v = float(np.clip(vel, 0.0, 1.0))
    env = np.minimum(1.0, t / 0.08) * _off(n, dur, sr, 0.1)
    imp = np.zeros(int(0.05 * sr))
    imp[0] = 1.0
    ir = _body(imp, sr, [(fc, fc / bw, 4.0 * g) for fc, bw, g in (body or VIOLIN_BODY)], min(5000.0, 0.44 * sr))
    from scipy.signal import fftconvolve
    y = fftconvolve(pulses * env, ir)[:n]
    y = y + 10 ** (BOW_NOISE_DB / 20.0) * np.max(np.abs(y)) * fftconvolve(rng.standard_normal(n) * env, ir)[:n] / (np.sqrt(len(ir)) + 1e-9)
    return F.dc_block(y, sr) / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.4 * v)


def vibrato_directivity(y, sr, freq, rate=5.8, direction=1.0):
    """Body directivity heard through the vibrato: partial 5 swings 20 dB at the vibrato rate when the listener is
    perpendicular (``direction`` 1), partials 7 and 9 swing at twice the rate; the other partials stay."""
    t = np.arange(len(y)) / sr
    d = float(np.clip(direction, 0.0, 1.0))
    out = y.copy()
    for k, r in ((5, rate), (7, 2 * rate), (9, 2 * rate)):
        if k * freq >= 0.45 * sr:
            continue
        band = F.bandpass(y, k * freq, sr, 8.0)
        depth_db = 20.0 * d if k == 5 else 10.0 * d
        gain = 10 ** (-0.5 * depth_db * (1 + np.sin(2 * math.pi * r * t)) / 20.0)
        out += band * (gain - 1.0)
    return out


@instrument("whirl_string", "Loose string plucked out of plane: its two planes share one tension (Kirchhoff-Carrier), so the "
            "orbit precesses and the pitch glides as the swing shrinks.", family="plucked", span=("E1", "E4"),
            ellipse=(0.3, "0 = planar pluck .. 1 = circular orbit"), slack=(1.0, "0..2 x the tension modulation (alpha)"))
def whirl_string(freq, dur, sr=DEFAULT_SR, vel=1.0, ellipse=0.3, slack=1.0):
    from .strings import whirling_string
    x, y = whirling_string(freq, dur + 1.0, sr, amp=0.004 + 0.012 * float(np.clip(vel, 0.0, 1.0)),
                           ellipse=float(np.clip(ellipse, 0.0, 1.0)), alpha=30.0 * float(np.clip(slack, 0.0, 2.0)), t60=3.0)
    s = x + 0.6 * y                                       # the pickup reads both planes, the vertical less
    s = s * _off(len(s), dur, sr, 0.15)
    return F.lowpass(F.dc_block(s, sr), min(4000.0, 0.42 * sr), sr, 0.7) / (np.max(np.abs(s)) + 1e-9) * (0.5 + 0.4 * vel)
