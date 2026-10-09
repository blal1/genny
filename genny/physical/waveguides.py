# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Digital-waveguide instruments (numba, offline, 48 kHz).

Every model is a port of a named source; constants are copied verbatim and converted to 48 kHz
with the ``core.rate_*`` helpers where they were tuned per sample at another rate. Anything
that has no source is marked ``# UNSOURCED:`` in the code and listed in ``UNSOURCED`` below.

Sources (research notes in ``synthgen/out/research``; originals under ``synthgen/stk/stk-master``):

* STK 5 (Cook & Scavone): ``DelayL`` / ``DelayA`` (DelayL.h, DelayA.h/.cpp), ``OnePole``,
  ``OneZero``, ``BiQuad``, ``PoleZero::setBlockZero``, ``ADSR``, ``Envelope``, ``BowTable.h``,
  ``ReedTable.h``, ``JetTable.h``, ``Twang.cpp``, ``Mandolin.cpp`` (+ rawwaves/mand1..12.raw),
  ``Bowed.cpp`` (bow table, beta 0.127236, Maestre 6-section body), ``BandedWG.cpp``,
  ``Brass.cpp``, ``Clarinet.cpp``, ``Flute.cpp``, ``Recorder.cpp``
  (research 07 §2, §4, §5, §6, §7, §17).
* Faust physmodels.lib ``violinModel`` / ``brassModel`` (research 08 §2.5, §3): same bow table
  and lip model; lip frequency law ``f * 4**(2*tension - 1)`` is identical to STK CC2.
* J.O. Smith PASP (research 10 §3, §4, §5): extended Karplus-Strong filters (pick direction,
  pick position comb, dynamic-level low-pass R_L = exp(-pi L T), brightness/sustain loop FIR
  g0 = exp(-6.91 P / S)), two polarisations, commuted body (air mode ~100 Hz, Q 10, >= 100 ms
  of body; the residual replaced by enveloped noise with a contracting low-pass), bowed-string
  vibrato (~6 Hz, ~1 %), mass/felt hammer (f = Q0 x^p, Q0 = 183 e^(0.045 n), p = 3.7 + 0.015 n,
  hysteresis alpha = 248 + 1.83 n - 0.055 n^2 us, m = 11.074 - 0.074 n + 0.0001 n^2 g,
  momentum transfer into (m s + 2R), force -> velocity f / 2R), piano unison detuning.
* Karplus & Strong 1983 (research 06 §6): two-level random initial table.
* Csound ``wgbow`` example (research 11 §C1): vibrato 0 for 0.5 s then a 1 s fade-in, depth 0.01.
* sc3 ``MdaPiano`` (sonic-pi, research 11 §C6, MdaUGens.cpp 194-212): decay law
  ``exp(-0.6 + 0.033 note - L)`` (note >= 44), muffle low-pass
  ``l = 50 + muffle^2 160 + muffvel (vel - 64)`` clamped to [55 + 0.25 note, 210], ``ff = l^2/fs``.
  SC synthetic piano detune [-0.05, 0, 0.04] semitones (research 11 §C5).
* Cook, Real Sound Synthesis §7.4-7.5 (research 04): stiffness by first-order allpasses
  (a < 0: shorter delay at high frequency), f_n = n f0 sqrt(1 + B n^2); chattering bridge = a
  point that reflects both travelling waves when its displacement exceeds a threshold;
  hurdy-gurdy wheel = constant bow.
* rjlib ``s_hstr`` (research 09 A3.4) and Cook §7.2: bow noise (low-passed noise on the bow).

Public API (all return mono float64 arrays at ``SR``; envelopes may be scalars, arrays of any
length -- resampled per sample by linear interpolation -- or length-n arrays):

    pluck(f0, dur, rng, position, brightness, body=...)      extended Karplus-Strong
    mandolin(f0, dur, rng, ...)                              STK Mandolin (commuted body files)
    bowed_string(f0_env, bow_pressure_env, bow_velocity_env, dur, rng, body=...)   STK Bowed
    banded(preset, f0, dur, rng, mode='strike'|'bow', ...)   STK BandedWG
    brass(f0_env, lip_tension, pressure_env, dur, rng)       STK Brass
    recorder(f0_env, breath_env, dur, rng, model='flute'|'verge')   STK Flute jet / STK Recorder
    clarinet(f0_env, breath_env, dur, rng)                   STK Clarinet
    hurdy_gurdy(f0, dur, rng)                                bowed strings + chattering bridge
    felt_piano(f0, velocity, dur, rng)                       felt hammer + stiff unison strings
    stk_adsr(...), bow_envelope(...), wind_envelope(...)     STK control envelopes at 48 kHz
    commuted_body_ir(kind, rng)                              body impulse responses
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy import signal

from .core import DEV_DIR, SR, decay_rate, rate_radius, rate_samples, spectrum_peaks, write_wav

STK_FS = 44100.0  # STK default sample rate (research 07 §0)
_RAWWAVES = Path(__file__).resolve().parent.parent / "stk" / "stk-master" / "rawwaves"

UNSOURCED = {
    "pluck.two_planes": "JOS gives two polarisations with the vertical decaying faster and a few "
                        "cents apart but no numbers: H plane +0.8 cent, sustain x1.6, 35 % of the "
                        "excitation.",
    "pluck.level_bandwidth": "dynamic-level low-pass bandwidth L = 1000 + 7000*velocity Hz follows "
                             "the research-10 sketch ('L ~ 1-8 kHz by velocity'), not a measured law.",
    "body.synthetic": "synthetic commuted bodies: noise T60 (0.08-0.2 s), contracting low-pass "
                      "8 kHz -> 400 Hz, air-mode weight 4, and air-mode frequencies for "
                      "violin/cello/bass (275/100/60 Hz, general acoustics); guitar 100 Hz, Q 10, "
                      ">= 100 ms is JOS.",
    "bowed.body_scale": "cello/bass/hurdy-gurdy bodies are the Maestre violin body with poles and "
                        "zeros moved down in frequency (Q kept) by 0.47 (cello/violin length), "
                        "0.22 (bass/violin air mode ~60/275 Hz) and 0.8.",
    "bowed.bow_noise": "bow-noise depth 0.08 and its 2 kHz two-pole low-pass (rjlib knob at full "
                       "scale) are chosen, the sources give the structure only.",
    "brass.normalisation": "breath expressed relative to the lip filter's DC gain, lip bandwidth "
                           "scaled with f (constant Q, = STK's at 220 Hz) and DC-blocker cut-off "
                           "scaled below 220 Hz: without these STK Brass only speaks in a narrow, "
                           "pitch-dependent pressure window.",
    "hurdy_gurdy": "string tunings (trompette f0, bourdon f0/2, mouche f0*3/4), chien position "
                   "(8 % of the bow-bridge segment), one-sided contact threshold (0.6 x running "
                   "RMS displacement), integrator leak 0.995, wheel speed wobble (0.4 %, 1.1 Hz), "
                   "wheel spin-up 0.25 s and the string mix are all chosen; only the mechanism "
                   "(Cook §7.5 chattering bridge, wheel = constant bow) is sourced.",
    "felt_piano": "string wave impedance R = 2.2 kg/s per side with 3 strings loading the "
                  "hammer, hammer speed 0.4 + 4.6 v m/s, strike point 1/8, felt softening of Q0 "
                  "(x (1 - 0.9 felt)), soundboard noise tail (T60 100 ms, 6 kHz -> 300 Hz, gain "
                  "0.3), loop brightness 0.35, inharmonicity B = 4e-4 (Cook gives ~0.004), and "
                  "the output scale are chosen.",
    "banded.exact_pitch": "exact_pitch=True (fractional delay minus the lastOut sample, loop gain "
                          "capped at 0.99995) fixes STK's integer delay lengths; not STK behaviour.",
    "tuning": "STK leaves Brass (lip pull), Bowed, Flute and the chien string off pitch by a few to "
              "~180 cents; `tune=True` renders once, measures, and rescales the frequency (cached "
              "calibration, not a model change). STK Twang/Mandolin/Plucked are 1 sample flat "
              "(lastOut read not compensated); corrected.",
}


# ==========================================================================================
# Envelope / control helpers
# ==========================================================================================

def _n(dur: float) -> int:
    return max(int(round(dur * SR)), 1)


def _as_env(x, n: int) -> np.ndarray:
    """Scalar -> constant; array of length != n -> linearly resampled to n samples."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape[0] == n:
        return a.copy()
    if a.shape[0] == 1:
        return np.full(n, float(a[0]))
    src = np.linspace(0.0, 1.0, a.shape[0])
    return np.interp(np.linspace(0.0, 1.0, n), src, a)


def _rate(r: float) -> float:
    """STK per-sample linear ramp rate tuned at 44.1 kHz -> 48 kHz (research 07 §0)."""
    return r * STK_FS / SR


def stk_adsr(n: int, attack_rate: float, decay_rate: float, sustain: float,
             release_rate: float, note_off: int | None = None, target: float = 1.0) -> np.ndarray:
    """STK ``ADSR::tick`` (ADSR.h) with per-sample rates given at 44.1 kHz (converted here).

    Linear segments: attack to ``target``, decay to ``sustain``, release to 0 after ``note_off``.
    """
    ar, dr, rr = _rate(attack_rate), _rate(decay_rate), _rate(release_rate)
    env = np.empty(n)
    v = 0.0
    state = 0  # 0 attack, 1 decay, 2 sustain, 3 release
    off = n if note_off is None else int(note_off)
    for i in range(n):
        if i == off:
            state = 3
        if state == 0:
            v += ar
            if v >= target:
                v = target
                state = 1
        elif state == 1:
            if v > sustain:
                v -= dr
                if v <= sustain:
                    v, state = sustain, 2
            else:
                v += dr
                if v >= sustain:
                    v, state = sustain, 2
        elif state == 3:
            v -= rr
            if v <= 0.0:
                v = 0.0
        env[i] = v
    return env


def bow_envelope(dur: float, amp: float = 0.8, release: float | None = None,
                 release_amp: float = 0.5) -> np.ndarray:
    """Bow velocity of STK ``Bowed::noteOn/noteOff`` (Bowed.cpp): maxVelocity = 0.03 + 0.2 amp,
    ADSR setAllTimes(0.02, 0.005, 0.9, 0.01) with attack rate amp*0.001 per sample and release
    rate (1 - release_amp)*0.005 per sample (44.1 kHz rates)."""
    n = _n(dur)
    off = None if release is None else int(release * SR)
    decay_rate = (1.0 - 0.9) / (0.005 * STK_FS)
    env = stk_adsr(n, amp * 0.001, decay_rate, 0.9, (1.0 - release_amp) * 0.005, off)
    return (0.03 + 0.2 * amp) * env


def wind_envelope(model: str, dur: float, amp: float = 0.8, release: float | None = None) -> np.ndarray:
    """Breath/pressure control of the STK wind ``noteOn/noteOff`` (research 07 §7, §17).

    * ``brass``   : p = amp*ADSR(0.005, 0.001, 1.0, 0.010), attack amp*0.001, release amp*0.005
                    (Brass.cpp; p is the normalised breath of ``brass``).
    * ``flute``   : (1.1 + 0.2 amp)/0.8 * ADSR(0.005, 0.01, 0.8, 0.010), attack/release amp*0.02.
    * ``clarinet``: Envelope target 0.55 + 0.3 amp, rate amp*0.005; off rate amp*0.01 (Clarinet.cpp).
    * ``recorder``: maxPressure = 35 (1.1 + 0.2 amp), default ADSR (decay 0.001, sustain 0.5),
                    attack amp*0.02, release amp*0.02 (Recorder.cpp).
    """
    n = _n(dur)
    off = None if release is None else int(release * SR)
    if model == "brass":
        return amp * stk_adsr(n, amp * 0.001, 1.0 / (0.001 * STK_FS) * 0.0, 1.0, amp * 0.005, off)
    if model == "clarinet":
        tgt = 0.55 + 0.3 * amp
        env = np.empty(n)
        r_on, r_off = _rate(amp * 0.005), _rate(amp * 0.01)
        v = 0.0
        for i in range(n):
            if off is not None and i >= off:
                v = max(v - r_off, 0.0)
            else:
                v = min(v + r_on, tgt)
            env[i] = v
        return env
    if model == "recorder":
        return 35.0 * (1.1 + 0.2 * amp) * stk_adsr(n, amp * 0.02, 0.001, 0.5, amp * 0.02, off)
    if model == "flute":
        # Flute.cpp: maxPressure = (1.1 + 0.2 amp)/0.8, ADSR(0.005, 0.01, 0.8, 0.010), attack amp*0.02
        return (1.1 + 0.2 * amp) / 0.8 * stk_adsr(n, amp * 0.02, (1 - 0.8) / (0.01 * STK_FS), 0.8,
                                                    amp * 0.02, off)
    raise ValueError(model)


def delayed_vibrato(n: int, rate: float = 6.12723, depth: float = 0.01,
                    delay: float = 0.5, fade: float = 1.0) -> np.ndarray:
    """Csound wgbow.csd vibrato: none for 0.5 s, then a 1 s linear fade-in to ``depth``
    (research 11 §C1); STK Bowed vibrato rate 6.12723 Hz (Bowed.cpp)."""
    t = np.arange(n) / SR
    amt = np.clip((t - delay) / fade, 0.0, 1.0) * depth
    return amt * np.sin(2 * np.pi * rate * t)


# ==========================================================================================
# Numba primitives (STK semantics)
# ==========================================================================================

@njit(cache=True)
def _dl_read(buf, w, d):
    """STK DelayL output right after writing ``buf[w]``: x[n - d], linear interpolation."""
    L = buf.shape[0]
    pos = w - d
    while pos < 0.0:
        pos += L
    i = int(pos)
    a = pos - i
    i0 = i % L
    i1 = i0 + 1
    if i1 == L:
        i1 = 0
    return buf[i0] * (1.0 - a) + buf[i1] * a


@njit(cache=True)
def _da_read(buf, w, d, st):
    """STK DelayA output right after writing ``buf[w]`` (DelayA.cpp setDelay / nextOut).

    Integer part M = floor(d - 0.5), allpass part alpha = d - M in [0.5, 1.5),
    coeff = (1 - alpha)/(1 + alpha); y = c x[n-M] + x[n-M-1] - c y[n-1].
    ``st[0]`` = last output, ``st[1]`` = previous allpass input.
    """
    L = buf.shape[0]
    m = math.floor(d - 0.5)
    if m < 0:
        m = 0
    alpha = d - m
    c = (1.0 - alpha) / (1.0 + alpha)
    x = buf[(w - m) % L]
    y = c * x + st[1] - c * st[0]
    st[0] = y
    st[1] = x
    return y


@njit(cache=True)
def _bowtable(x, offset, slope):
    """STK BowTable.h: clip((|(x + offset) slope| + 0.75)^-4, 0.01, 0.98)."""
    s = abs((x + offset) * slope) + 0.75
    r = s ** -4.0
    if r < 0.01:
        r = 0.01
    if r > 0.98:
        r = 0.98
    return r


# ==========================================================================================
# Plucked strings: extended Karplus-Strong (JOS) and STK Twang / Mandolin
# ==========================================================================================

@njit(cache=True)
def _eks_kernel(exc, n, D, h0, h1):
    """Loop: DelayA (tuning allpass) + symmetric 3-tap FIR [h1, h0, h1] (1 sample delay)."""
    L = int(D) + 8
    buf = np.zeros(L)
    st = np.zeros(2)
    out = np.empty(n)
    x1 = 0.0
    x2 = 0.0
    w = 0
    ne = exc.shape[0]
    for i in range(n):
        lo = st[0]
        v = h1 * lo + h0 * x1 + h1 * x2
        x2 = x1
        x1 = lo
        e = exc[i] if i < ne else 0.0
        buf[w] = e + v
        out[i] = _da_read(buf, w, D, st)
        w += 1
        if w == L:
            w = 0
    return out


def _one_pole(x, p):
    """H = (1 - p)/(1 - p z^-1) (JOS pick-direction / dynamic-level low-pass)."""
    return signal.lfilter([1.0 - p], [1.0, -p], x)


def _load_mand(mic: int) -> np.ndarray | None:
    """STK rawwaves/mand{mic+1}.raw: headerless 16-bit big-endian mono at 22050 Hz
    (FileRead.cpp), normalised to peak 1 by FileWvIn (doNormalize default), resampled to SR."""
    path = _RAWWAVES / f"mand{mic + 1}.raw"
    if not path.exists():
        return None
    x = np.fromfile(path, dtype=">i2").astype(np.float64) / 32768.0
    x /= max(np.max(np.abs(x)), 1e-12)
    return signal.resample_poly(x, 320, 147)  # 22050 * 320/147 = 48000


def _tv_lowpass_noise(n, f_start, f_end, rng):
    """White noise through a one-pole low-pass whose cutoff glides exponentially (per sample)."""
    fc = f_start * (f_end / f_start) ** (np.arange(n) / max(n - 1, 1))
    a = np.exp(-2 * np.pi * fc / SR)
    x = rng.uniform(-1.0, 1.0, n)
    y = np.empty(n)
    s = 0.0
    for i in range(n):
        s = (1.0 - a[i]) * x[i] + a[i] * s
        y[i] = s
    return y


_BODY_AIR = {  # air (Helmholtz) mode Hz, Q, body T60 s
    "guitar": (100.0, 10.0, 0.10),   # JOS Body_Factoring_Example: ~100 Hz, Q 10 used, >=100 ms
    "violin": (275.0, 10.0, 0.08),   # UNSOURCED: violin A0 ~275 Hz (general acoustics)
    "cello": (100.0, 10.0, 0.15),    # UNSOURCED: cello A0 ~100 Hz
    "bass": (60.0, 10.0, 0.20),      # UNSOURCED: double-bass A0 ~60 Hz
}


def commuted_body_ir(kind: str, rng: np.random.Generator, mic: int = 0) -> np.ndarray:
    """Body impulse response used as the string excitation (commuted synthesis, JOS §3.5).

    ``mandolin``: the recorded STK body responses (mand1..12.raw). Other kinds follow JOS'
    factored body: the residual is enveloped noise through a low-pass whose bandwidth shrinks
    (PASP/Approximating_Shortened_Excitations_Noise) plus the air mode as a resonator
    (PASP/Body_Factoring_Example). Unit energy.
    """
    if kind == "mandolin":
        ir = _load_mand(mic)
        if ir is not None:
            return ir / np.sqrt(np.sum(ir ** 2))
        kind = "guitar"  # files absent -> synthetic commuted body
    f_air, q, t60 = _BODY_AIR[kind]
    n = int(3 * t60 * SR)
    t = np.arange(n) / SR
    # UNSOURCED: 8 kHz -> 400 Hz contraction over the IR; attack 0.5 ms.
    resid = _tv_lowpass_noise(n, 8000.0, 400.0, rng) * np.exp(-6.91 * t / t60)
    resid *= np.minimum(t / 0.0005, 1.0)
    r = math.exp(-math.pi * f_air / q / SR)
    b0 = 0.5 - 0.5 * r * r  # STK setResonance(normalize=true)
    air = signal.lfilter([b0, 0.0, -b0], [1.0, -2 * r * math.cos(2 * math.pi * f_air / SR), r * r], resid)
    ir = resid + 4.0 * air  # UNSOURCED: air-mode weight
    return ir / np.sqrt(np.sum(ir ** 2))


def pluck(f0: float, dur: float, rng: np.random.Generator, position: float = 0.2,
          brightness: float = 0.5, body: str = "guitar", velocity: float = 0.8,
          sustain: float | None = None, direction: float = 0.0, two_planes: bool = True,
          mic: int = 0) -> np.ndarray:
    """Extended Karplus-Strong string (JOS PASP/Extended_Karplus_Strong_Algorithm, research 10 §3).

    * excitation: one period of two-level random +-1 (Karplus & Strong 1983), pick-direction
      low-pass ``(1-p)/(1-p z^-1)``, dynamic-level low-pass ``R_L = exp(-pi L T)``, pick-position
      comb ``1 - z^-floor(beta N + 1/2)``, then convolved with a body IR (commuted synthesis)
      when ``body`` is 'guitar' | 'mandolin' | 'violin' | 'cello' | 'bass' ('none' = bare).
    * loop: tuning allpass (STK DelayA) + FIR ``g0 [(1-B)/4, (1+B)/2, (1-B)/4]``,
      ``g0 = exp(-6.91 P / S)`` so the sustain S (s to -60 dB) is pitch independent.
      Default S = T60 of STK Twang's loop gain 0.995 + f*5e-6 per period (Twang.cpp).
    * two polarisations (JOS §3.4): a second, slightly detuned, longer-sustaining plane.
    """
    n = _n(dur)
    N = SR / f0
    if sustain is None:
        g = min(0.995 + f0 * 0.000005, 0.99999)
        sustain = 6.91 / (f0 * -math.log(g))
    B = float(np.clip(brightness, 0.0, 1.0))
    ne = int(N)
    exc = rng.choice(np.array([-1.0, 1.0]), ne)
    exc = _one_pole(exc, direction)
    L_bw = 1000.0 + 7000.0 * velocity  # UNSOURCED (research-10 sketch): level bandwidth
    exc = _one_pole(exc, math.exp(-math.pi * L_bw / SR))
    m = int(math.floor(position * N + 0.5))
    if 0 < m < ne:
        exc[m:] -= exc[:-m].copy()
    exc *= velocity
    if body != "none":
        exc = np.convolve(exc, commuted_body_ir(body, rng, mic))
    exc = np.pad(exc, (0, max(0, n - exc.shape[0])))[:max(n, exc.shape[0])]

    def plane(f, S, e):
        P = 1.0 / f
        g0 = math.exp(-6.91 * P / S)
        h0, h1 = g0 * (1 + B) / 2, g0 * (1 - B) / 4
        # loop = DelayA D + 1 sample (lastOut read) + 1 sample (FIR centre)
        return _eks_kernel(e, n, SR / f - 2.0, h0, h1)

    if not two_planes:
        return 0.4 * plane(f0, sustain, exc)  # UNSOURCED: output scale (peak < 1 at velocity 1)
    # UNSOURCED: plane detune (+0.8 cent), sustain ratio 1.6, excitation share 35 %.
    v = plane(f0, sustain, 0.65 * exc)
    h = plane(f0 * 2 ** (0.8 / 1200), 1.6 * sustain, 0.35 * exc)
    return 0.4 * (v + h)


@njit(cache=True)
def _twang_kernel(exc, n, D, comb_d, gain):
    """STK Twang::tick: s = DelayA(in + Fir[0.5,0.5]*gain(s_last)); out = 0.5 (s - DelayL_comb(s))."""
    L = int(D) + 8
    buf = np.zeros(L)
    Lc = int(comb_d) + 8
    cbuf = np.zeros(Lc)
    st = np.zeros(2)
    out = np.empty(n)
    fx1 = 0.0
    w = 0
    cw = 0
    ne = exc.shape[0]
    for i in range(n):
        lo = st[0]
        f = gain * (0.5 * lo + 0.5 * fx1)
        fx1 = lo
        e = exc[i] if i < ne else 0.0
        buf[w] = e + f
        y = _da_read(buf, w, D, st)
        w += 1
        if w == L:
            w = 0
        cbuf[cw] = y
        c = _dl_read(cbuf, cw, comb_d)
        cw += 1
        if cw == Lc:
            cw = 0
        out[i] = 0.5 * (y - c)
    return out


def mandolin(f0: float, dur: float, rng: np.random.Generator, velocity: float = 0.8,
             position: float = 0.4, mic: int = 0, detune: float = 0.995,
             body_size: float = 1.0, sustain: float | None = None) -> np.ndarray:
    """STK Mandolin (Mandolin.cpp, research 07 §4.4): two Twang strings at f and f*0.995, both
    excited by the recorded body response ``mand{mic+1}.raw`` (commuted synthesis) scaled by
    the pluck amplitude; out = 0.2 (s0 + s1). ``body_size`` = CC2 playback-rate factor,
    ``sustain`` in [0,1] = CC11 (loop gain 0.97 + 0.03 v). Without the raw files a synthetic
    commuted body (JOS) is used instead.
    """
    n = _n(dur)
    ir = _load_mand(mic)
    if ir is None:
        ir = commuted_body_ir("guitar", rng) * 0.3
    if body_size != 1.0:  # FileWvIn rate = size * 22050/fs -> time-scale by 1/size
        m = max(int(len(ir) / body_size), 2)
        ir = np.interp(np.arange(m) * body_size, np.arange(len(ir)), ir)
    exc = ir * velocity
    loop_gain = 0.995 if sustain is None else 0.97 + 0.03 * sustain
    out = np.zeros(n)
    for f in (f0, f0 * detune):
        # STK uses fs/f - 0.5 (Fir phase delay) but its loop also reads delayLine.lastOut(), one
        # more sample: STK Mandolin is ~1 sample flat (-14 cents at 392 Hz). Compensated here,
        # as STK itself does in Clarinet/Flute ("- 1.0").
        D = SR / f - 0.5 - 1.0
        gain = loop_gain + f * 0.000005
        if gain >= 1.0:
            gain = 0.99999
        out += _twang_kernel(exc, n, D, 0.5 * position * D, gain)
    return 0.2 * out


# ==========================================================================================
# Bowed string: STK Bowed (+ Faust violinModel table), bow noise, delayed vibrato, bodies
# ==========================================================================================

@njit(cache=True)
def _bowed_kernel(f0, slope, down, bowvel, vib, beta, pole, fs, maxlen):
    """STK Bowed::tick (Bowed.h) with per-sample setFrequency / bow position."""
    n = f0.shape[0]
    L = maxlen + 8
    neck = np.zeros(L)
    bridge = np.zeros(L)
    out = np.empty(n)
    neck_last = 0.0
    bridge_last = 0.0
    sf = 0.0
    b0 = (1.0 - pole) * 0.95  # OnePole setPole(p) b0 = 1-p, setGain(0.95)
    w = 0
    for i in range(n):
        base = fs / f0[i] - 4.0
        if base <= 0.0:
            base = 0.3
        bd = base * beta
        nd = base * (1.0 - beta) + base * vib[i]
        sf = b0 * bridge_last + pole * sf
        bridge_refl = -sf
        nut_refl = -neck_last
        dv = bowvel[i] - (bridge_refl + nut_refl)
        nv = 0.0
        if down[i] > 0.0:
            nv = dv * _bowtable(dv, 0.001, slope[i])
        neck[w] = bridge_refl + nv
        bridge[w] = nut_refl + nv
        neck_last = _dl_read(neck, w, nd)
        bridge_last = _dl_read(bridge, w, bd)
        out[i] = bridge_last
        w += 1
        if w == L:
            w = 0
    return out


# Esteban Maestre body filter, six SOS rows [b0, b1, b2, a1, a2] at 44.1 kHz (Bowed.cpp 63-68).
_MAESTRE_BODY = np.array([
    [1.0, 1.5667, 0.3133, -0.5509, -0.3925],
    [1.0, -1.9537, 0.9542, -1.6357, 0.8697],
    [1.0, -1.6683, 0.8852, -1.7674, 0.8735],
    [1.0, -1.8585, 0.9653, -1.8498, 0.9516],
    [1.0, -1.9299, 0.9621, -1.9354, 0.9590],
    [1.0, -1.9800, 0.9888, -1.9867, 0.9923],
])
# UNSOURCED: cello = body-length ratio (35.5/75.5 cm); bass = air-mode ratio (~60/275 Hz; the
# length ratio 0.32 leaves the A1 fundamental 44 dB under its 3rd harmonic); hurdy-gurdy 0.8.
_BODY_SCALE = {"violin": 1.0, "cello": 0.47, "bass": 0.22, "gurdy": 0.8}


def _map_root(z: complex, k: float) -> complex:
    """Move a pole/zero from angle th, radius r to angle k*th, radius r**k (rate_radius form):
    same Q, frequencies scaled by k. Negative real roots (Nyquist-side tilt) are kept."""
    if abs(z.imag) < 1e-12:
        return z if z.real <= 0 else complex(abs(z.real) ** k, 0.0)
    r, th = abs(z), math.atan2(z.imag, z.real)
    return r ** k * complex(math.cos(k * th), math.sin(k * th))


def body_sos(kind: str = "violin") -> np.ndarray:
    """Maestre violin body (STK Bowed) converted from 44.1 kHz to SR, optionally scaled down in
    frequency for larger bodies; peak gain kept equal to the original. sos rows for sosfilt."""
    k = _BODY_SCALE[kind] * STK_FS / SR
    rows = []
    for b0, b1, b2, a1, a2 in _MAESTRE_BODY:
        zs = [_map_root(z, k) for z in np.roots([b0, b1, b2])]
        ps = [_map_root(p, k) for p in np.roots([1.0, a1, a2])]
        b = np.real(np.poly(zs)) * b0
        a = np.real(np.poly(ps))
        rows.append(np.concatenate([b, a]))
    sos = np.array(rows)
    orig = np.column_stack([_MAESTRE_BODY[:, :3], np.ones(6), _MAESTRE_BODY[:, 3:]])
    _, h0 = signal.sosfreqz(orig, 4096)
    _, h1 = signal.sosfreqz(sos, 4096)
    sos[0, :3] *= np.max(np.abs(h0)) / np.max(np.abs(h1))
    return sos


def _bow_noise(n, rng, depth):
    """Low-passed noise on the bow velocity (Cook §7.2 'noise in the bow junction'; rjlib s_hstr
    noise -> lop~ x2). UNSOURCED: 2 kHz cutoff, depth."""
    a = math.exp(-2 * math.pi * 2000.0 / SR)
    x = signal.lfilter([1 - a], [1, -a], rng.uniform(-1, 1, n))
    x = signal.lfilter([1 - a], [1, -a], x)
    return depth * x / max(np.std(x), 1e-12) * 0.33


def bowed_string(f0_env, bow_pressure_env, bow_velocity_env, dur: float,
                 rng: np.random.Generator, body: str = "cello", position: float = 0.127236,
                 vibrato: bool = True, bow_noise: float = 0.08, tune: bool = True) -> np.ndarray:
    """STK Bowed (Bowed.cpp/.h, research 07 §5; Faust violinModel research 08 §2.5).

    ``bow_pressure_env`` in [0, 1] maps to the bow-table slope 5 - 4 p (CC2); p <= 0 lifts the
    bow. ``bow_velocity_env`` is the bow velocity in STK units (maxVelocity*ADSR, see
    ``bow_envelope``). Bow position beta default 0.127236. Delays: base = fs/f - 4,
    bridge = base*beta, neck = base*(1-beta) (+ base*vibrato). String filter OnePole pole
    0.75 - 0.2*22050/fs, gain 0.95. Output = 0.1248 * body(bridge wave). ``body`` =
    'violin' | 'cello' | 'bass' | 'none'. Vibrato: Csound wgbow delayed fade-in, 6.12723 Hz, 1 %.
    Measured: pressure must grow with bow speed (Schelleng minimum force). With STK's default
    slope 3 (p = 0.5) and bow_envelope(amp=0.8) cello/bass notes fall into double slip (sounding
    the octave); p >= 0.8 gives Helmholtz motion from 55 to 440 Hz. ``tune`` removes the
    remaining few-cent offset of the fs/f - 4 delay rule (cached calibration).
    """
    n = _n(dur)
    f0 = _as_env(f0_env, n)
    p = np.clip(_as_env(bow_pressure_env, n), 0.0, 1.0)
    vel = _as_env(bow_velocity_env, n)
    if bow_noise > 0.0:
        vel = vel * (1.0 + _bow_noise(n, rng, bow_noise))
    vib = delayed_vibrato(n) if vibrato else np.zeros(n)
    slope = 5.0 - 4.0 * p
    down = (p > 0.0).astype(np.float64)
    pole = 0.75 - 0.2 * 22050.0 / SR
    ratio = 1.0
    if tune:
        ratio = _bowed_tuning(float(np.median(f0)), float(np.median(p[p > 0])) if np.any(p > 0) else 0.5,
                              float(np.max(vel)), position, pole)
    fr = f0 * ratio
    maxlen = int(SR / np.min(fr)) + 16
    y = _bowed_kernel(fr, slope, down, vel, vib, position, pole, float(SR), maxlen)
    if body != "none":
        y = signal.sosfilt(body_sos(body), y)
    return 0.1248 * y


_TUNE_CACHE: dict = {}


def _bowed_tuning(f, p, v, beta, pole) -> float:
    key = ("bowed", round(f, 3), round(p, 3), round(v, 4), round(beta, 4))
    if key not in _TUNE_CACHE:
        n = int(0.7 * SR)
        fr = np.full(n, f)
        env = np.minimum(np.arange(n) / (0.05 * SR), 1.0) * v
        y = _bowed_kernel(fr, np.full(n, 5.0 - 4.0 * p), np.ones(n), env, np.zeros(n), beta, pole,
                          float(SR), int(SR / f) + 16)
        fm = measure_f0(y[int(0.3 * SR):], f)
        _TUNE_CACHE[key] = f / fm if fm > 0 and abs(math.log2(f / fm)) < 0.1 else 1.0
    return _TUNE_CACHE[key]


# ==========================================================================================
# Hurdy-gurdy: wheel-bowed strings + chattering (chien) bridge
# ==========================================================================================

@njit(cache=True)
def _bowed_chien_kernel(f0, slope, bowvel, beta, pole, fs, maxlen, chien, thr_mult, leak):
    """STK Bowed loop whose bridge segment is split at a chattering contact point (Cook §7.5):
    while the point's displacement exceeds a threshold it reflects both travelling waves
    (the string is briefly shortened), otherwise it passes them."""
    n = f0.shape[0]
    L = maxlen + 8
    neck = np.zeros(L)
    a1 = np.zeros(L)  # bow -> chien
    bb = np.zeros(L)  # chien -> bridge -> chien (round trip)
    a2 = np.zeros(L)  # chien -> bow
    out = np.empty(n)
    neck_last = 0.0
    a1_last = 0.0
    bb_last = 0.0
    a2_last = 0.0
    sf = 0.0
    b0 = (1.0 - pole) * 0.95
    disp = 0.0
    ms = 1e-8
    w = 0
    for i in range(n):
        base = fs / f0[i] - 4.0
        if base <= 0.0:
            base = 0.3
        bd = base * beta
        blen = bd * chien
        alen = 0.5 * (bd - blen)
        nd = base * (1.0 - beta)
        # chien junction
        sf = b0 * bb_last + pole * sf
        from_bridge = -sf
        from_bow = a1_last
        disp = leak * disp + from_bow + from_bridge
        ms = 0.999 * ms + 0.001 * disp * disp
        thr = thr_mult * math.sqrt(ms)
        if disp > thr:
            b_in = -from_bridge
            a2_in = -from_bow
        else:
            b_in = from_bow
            a2_in = from_bridge
        # bow junction
        bridge_refl = a2_last
        nut_refl = -neck_last
        dv = bowvel[i] - (bridge_refl + nut_refl)
        nv = dv * _bowtable(dv, 0.001, slope[i])
        neck[w] = bridge_refl + nv
        a1[w] = nut_refl + nv
        bb[w] = b_in
        a2[w] = a2_in
        neck_last = _dl_read(neck, w, nd)
        a1_last = _dl_read(a1, w, alen)
        bb_last = _dl_read(bb, w, blen)
        a2_last = _dl_read(a2, w, alen)
        out[i] = bb_last
        w += 1
        if w == L:
            w = 0
    return out


def hurdy_gurdy(f0: float, dur: float, rng: np.random.Generator, amp: float = 0.7,
                buzz: float = 0.6, melody=None, wheel_attack: float = 0.25, pressure: float = 0.8,
                strings: tuple = ("trompette", "bourdon", "mouche")) -> np.ndarray:
    """Hurdy-gurdy drone: wheel-bowed STK Bowed strings (Cook: wheel = constant bow) --
    ``trompette`` at f0 over a chattering chien bridge (Cook §7.5 nonlinear scattering bridge),
    ``bourdon`` at f0/2, ``mouche`` at 3/4 f0 -- plus an optional chanterelle following
    ``melody`` (an f0 envelope, delayed vibrato). ``buzz`` = contact threshold in running-RMS
    units of the chien-point displacement (lower = more buzz). ``pressure`` -> bow-table slope
    5 - 4 p (0.8 keeps Helmholtz motion at the wheel speed; lower drops into double slip).

    UNSOURCED: tunings, chien position, threshold law, wheel wobble, mix (see ``UNSOURCED``).
    """
    n = _n(dur)
    t = np.arange(n) / SR
    pole = 0.75 - 0.2 * 22050.0 / SR
    slope = np.full(n, 5.0 - 4.0 * pressure)
    # Wheel: constant bow velocity (Cook); STK maxVelocity 0.03 + 0.2 amp; smooth spin-up.
    wheel = (0.03 + 0.2 * amp) * (1.0 - (1.0 - np.clip(t / wheel_attack, 0, 1)) ** 2)
    wheel *= 1.0 + 0.004 * np.sin(2 * np.pi * 1.1 * t + rng.uniform(0, 2 * np.pi))  # UNSOURCED
    wheel *= 1.0 + _bow_noise(n, rng, 0.05)
    vmax = float(wheel.max())
    zero = np.zeros(n)
    out = np.zeros(n)
    if "trompette" in strings:
        chien = 0.08  # UNSOURCED: chien point at 8 % of the bow-bridge segment
        key = ("chien", round(f0, 3), round(pressure, 3), round(vmax, 4), buzz)
        if key not in _TUNE_CACHE:  # calibrate the shortened-string pitch shift
            c_ = 1.0
            for _ in range(4):
                m = int(1.0 * SR)
                env = np.minimum(np.arange(m) / (0.05 * SR), 1.0) * vmax
                y = _bowed_chien_kernel(np.full(m, f0 * c_), np.full(m, slope[0]), env, 0.127236, pole,
                                        float(SR), int(SR / f0) + 16, chien, buzz, 0.995)
                mf = measure_f0(y[int(0.4 * SR):], f0)
                if mf <= 0 or abs(math.log2(mf / f0)) > 0.1:
                    c_ = _bowed_tuning(f0, pressure, vmax, 0.127236, pole)
                    break
                if abs(1200 * math.log2(mf / f0)) < 0.5:
                    break
                c_ *= f0 / mf
            _TUNE_CACHE[key] = c_
        out += _bowed_chien_kernel(np.full(n, f0 * _TUNE_CACHE[key]), slope, wheel, 0.127236,
                                   pole, float(SR), int(SR / f0) + 16, chien, buzz, 0.995)
    for name, fr, g in (("bourdon", f0 / 2, 0.8), ("mouche", f0 * 0.75, 0.5)):
        if name in strings:
            r = _bowed_tuning(fr, pressure, vmax, 0.127236, pole)
            out += g * _bowed_kernel(np.full(n, fr * r), slope, np.ones(n), wheel, zero,
                                     0.127236, pole, float(SR), int(SR / fr) + 16)
    if melody is not None:
        mel = _as_env(melody, n)
        r = _bowed_tuning(float(np.median(mel)), pressure, vmax, 0.127236, pole)
        out += 0.8 * _bowed_kernel(mel * r, slope, np.ones(n), wheel,
                                   delayed_vibrato(n, depth=0.004), 0.127236, pole, float(SR),
                                   int(SR / mel.min()) + 16)
    return 0.1 * signal.sosfilt(body_sos("gurdy"), out)  # UNSOURCED mix level (STK 0.1248)


# ==========================================================================================
# STK BandedWG: struck / bowed bars, glass harmonica, Tibetan bowl
# ==========================================================================================

BANDED_PRESETS = {  # BandedWG.cpp setPreset, verbatim
    "uniform_bar": dict(modes=[1.0, 2.756, 5.404, 8.933],
                        basegains=[0.9 ** (i + 1) for i in range(4)], excitation=[1.0] * 4),
    "tuned_bar": dict(modes=[1.0, 4.0198391420, 10.7184986595, 18.0697050938],
                      basegains=[0.999 ** (i + 1) for i in range(4)], excitation=[1.0] * 4),
    "glass_harmonica": dict(modes=[1.0, 2.32, 4.25, 6.63, 9.38],
                            basegains=[0.999 ** (i + 1) for i in range(5)], excitation=[1.0] * 5),
    "tibetan_bowl": dict(
        modes=[0.996108344, 1.0038916562, 2.979178, 2.99329767, 5.704452, 5.704452, 8.9982,
               9.01549726, 12.83303, 12.807382, 17.2808219, 21.97602739726],
        basegains=[0.999925960128219, 0.999925960128219, 0.999982774366897, 0.999982774366897,
                   1.0, 1.0, 1.0, 1.0, 0.999965497558225, 0.999965497558225,
                   0.9999999999999999999965497558225, 0.999999999999999965497558225],
        excitation=[11.900357 / 10, 11.900357 / 10, 10.914886 / 10, 10.914886 / 10,
                    42.995041 / 10, 42.995041 / 10, 40.063034 / 10, 40.063034 / 10,
                    7.063034 / 10, 7.063034 / 10, 57.063034 / 10, 57.063034 / 10]),
}


@njit(cache=True)
def _banded_kernel(n, D, gains, b0s, a1s, a2s, bow, bowvel, slope, base_gain, integ,
                   pre_val, pre_cnt, frac):
    """BandedWG::tick: per mode a delay loop through a normalised 2-pole band-pass."""
    nm = D.shape[0]
    L = int(np.max(D)) + 8
    bufs = np.zeros((nm, L))
    ws = np.zeros(nm, dtype=np.int64)
    last = np.zeros(nm)
    x1 = np.zeros(nm)
    x2 = np.zeros(nm)
    y1 = np.zeros(nm)
    y2 = np.zeros(nm)
    # pluck(): write excitation_k*amp/nModes into delay k int(len_k/len_min) times
    for k in range(nm):
        for j in range(pre_cnt[k]):
            w = ws[k]
            bufs[k, w] = pre_val[k]
            if frac:
                last[k] = _dl_read(bufs[k], w, D[k])
            else:
                last[k] = bufs[k, (w - int(D[k])) % L]
            ws[k] = (w + 1) % L
    out = np.empty(n)
    vin = 0.0
    for i in range(n):
        x = 0.0
        if bow:
            vin = integ * vin
            for k in range(nm):
                vin += base_gain * last[k]
            x = bowvel[i] - vin
            x = x * _bowtable(x, 0.0, slope)
            x = x / nm
        acc = 0.0
        for k in range(nm):
            u = x + gains[k] * last[k]
            y = b0s[k] * u - b0s[k] * x2[k] - a1s[k] * y1[k] - a2s[k] * y2[k]
            x2[k] = x1[k]
            x1[k] = u
            y2[k] = y1[k]
            y1[k] = y
            w = ws[k]
            bufs[k, w] = y
            if frac:
                last[k] = _dl_read(bufs[k], w, D[k])
            else:
                last[k] = bufs[k, (w - int(D[k])) % L]
            ws[k] = (w + 1) % L
            acc += y
        out[i] = 4.0 * acc
    return out


def banded(preset: str, f0: float, dur: float, rng: np.random.Generator | None = None,
           mode: str = "strike", velocity: float = 0.8, bow_pressure: float | None = None,
           bow_env=None, sustain: float | None = None, integration: float = 0.0,
           position: float | None = None, exact_pitch: bool | None = None,
           tune: bool = True) -> np.ndarray:
    """STK BandedWG (Essl & Cook; BandedWG.cpp, research 07 §6, 11 §B2).

    One band per mode: delay of int(fs/(f*mode_k)) samples (modes with length <= 2 dropped,
    f clamped to 1568 Hz) around a band-pass ``setResonance(f*mode_k, 1 - 32 pi/fs, normalize)``.
    ``mode='strike'``: pluck() pre-loads each delay; ``'bow'``: BowTable (slope 3.0, or
    10 - 9 p for ``bow_pressure`` p = CC2) against the summed delay outputs, bow velocity =
    ADSR*(0.03 + 0.1 amp) with attack amp*0.001 (or ``bow_env``). ``sustain`` v -> CC1
    gains = basegains*(0.9 + 0.1 v). ``position`` -> Csound wgbowedbar strike gains for the
    first 4 modes. ``exact_pitch`` (default: True when bowed) uses fractional delays
    (UNSOURCED fix of STK's integer lengths, gain cap 0.99995); struck notes stay STK-exact.
    Deterministic; ``rng`` unused (kept for a uniform signature).
    """
    pr = BANDED_PRESETS[preset]
    n = _n(dur)
    if exact_pitch is None:  # bowed notes lock to the loop period -> tune them; struck = STK
        exact_pitch = mode == "bow"
    if mode == "bow" and exact_pitch and tune:  # the bow pulls the pitch sharp: calibrate
        key = ("banded", preset, round(f0, 3), velocity, bow_pressure, integration)
        if key not in _TUNE_CACHE:
            c_ = 1.0
            for _ in range(4):
                y = banded(preset, f0 * c_, 2.5, None, "bow", velocity, bow_pressure, None,
                           sustain, integration, position, True, False)
                # the bowed pitch drifts while it builds; inharmonic -> measure the partial
                mf = measure_partial(y[int(1.0 * SR):], f0)
                if mf <= 0 or abs(math.log2(mf / f0)) > 0.1:
                    c_ = 1.0
                    break
                if abs(1200 * math.log2(mf / f0)) < 0.5:
                    break
                c_ *= f0 / mf
            _TUNE_CACHE[key] = c_
        f0 = f0 * _TUNE_CACHE[key]
    f = min(f0, 1568.0)
    base = SR / f
    modes, bg, ex = pr["modes"], np.array(pr["basegains"]), np.array(pr["excitation"], float)
    D, keep = [], 0
    for m in modes:
        # STK: integer length int(fs/(f m)); the loop adds 1 sample (lastOut). exact_pitch
        # compensates that sample and keeps the fraction.
        ln = base / m - 1.0 if exact_pitch else float(int(base / m))
        if ln <= 2.0:
            break
        D.append(ln)
        keep += 1
    D = np.array(D)
    modes = np.array(modes[:keep])
    gains = bg[:keep].copy()
    if sustain is not None:
        gains *= 0.9 + 0.1 * sustain
    if exact_pitch:
        gains = np.minimum(gains, 0.99995)
    ex = ex[:keep].copy()
    if position is not None:  # Csound bowedbar.c strike-position gains
        pg = [abs(math.sin(position * math.pi / 2)), abs(math.sin(position * math.pi)) * 0.9,
              abs(math.sin(position * math.pi * 1.5)) * 0.81, abs(math.sin(2 * position * math.pi)) * 0.729]
        for k in range(min(4, keep)):
            ex[k] *= pg[k]
    r = 1.0 - math.pi * 32.0 / SR
    b0s = np.full(keep, 0.5 - 0.5 * r * r)
    a1s = -2.0 * r * np.cos(2 * np.pi * f * modes / SR)
    a2s = np.full(keep, r * r)
    bow = mode == "bow"
    if bow:
        if bow_env is None:
            decay_rate = (1.0 - 0.9) / (0.005 * STK_FS)
            bow_env = (0.03 + 0.1 * velocity) * stk_adsr(n, velocity * 0.001, decay_rate, 0.9, 0.0)
        bowvel = _as_env(bow_env, n)
        slope = 3.0 if bow_pressure is None else 10.0 - 9.0 * bow_pressure
        pre_val, pre_cnt = np.zeros(keep), np.zeros(keep, dtype=np.int64)
    else:
        bowvel, slope = np.zeros(n), 3.0
        pre_val = ex * velocity / keep
        pre_cnt = np.array([int(d / D[-1]) for d in D], dtype=np.int64)
    return _banded_kernel(n, D, gains, b0s, a1s, a2s, bow, bowvel, slope, 0.999, integration,
                          pre_val, pre_cnt, exact_pitch)


# ==========================================================================================
# Winds: STK Brass, Clarinet, Recorder
# ==========================================================================================

@njit(cache=True)
def _brass_kernel(f0, lipf, breath, scale, r, dcp, fs, maxlen, tune):
    """STK Brass::tick (Brass.h): lip = BiQuad resonance (gain 0.03), squared, clipped at 1;
    bore = DelayA of 2 fs/f + 3 samples; DC blocker PoleZero(dcp). ``breath`` is multiplied by
    ``scale`` (see ``brass``) and the output divided by it."""
    n = f0.shape[0]
    L = maxlen + 8
    buf = np.zeros(L)
    st = np.zeros(2)
    out = np.empty(n)
    ly1 = 0.0
    ly2 = 0.0
    dx1 = 0.0
    dy1 = 0.0
    w = 0
    for i in range(n):
        D = (fs / f0[i] * 2.0 + 3.0) * tune
        a1 = -2.0 * r * math.cos(2.0 * math.pi * lipf[i] / fs)
        a2 = r * r
        mouth = 0.3 * breath[i] * scale[i]
        bore = 0.85 * st[0]
        y = 0.03 * (mouth - bore) - a1 * ly1 - a2 * ly2
        ly2 = ly1
        ly1 = y
        dp = y * y
        if dp > 1.0:
            dp = 1.0
        s = dp * mouth + (1.0 - dp) * bore
        dc = s - dx1 + dcp * dy1
        dx1 = s
        dy1 = dc
        buf[w] = dc
        out[i] = _da_read(buf, w, D, st) / scale[i]
        w += 1
        if w == L:
            w = 0
    return out


def _brass_constants(f_ref: float) -> tuple[float, float]:
    """Lip pole radius and DC-blocker pole. STK: 0.997 / 0.99 at 44.1 kHz (rate_radius), i.e. a
    fixed ~44 Hz lip bandwidth. UNSOURCED: the lip bandwidth is scaled with f so the lip keeps
    the Q it has at 220 Hz at every pitch, and the DC blocker cut-off is scaled down below
    220 Hz. Measured: with STK's fixed radius low notes never speak and notes above ~330 Hz
    turn quasi-periodic (subharmonic) for p >= 0.7; with constant Q every note 110-880 Hz speaks
    cleanly for p 0.3-0.8."""
    r0 = rate_radius(0.997, STK_FS)
    d0 = rate_radius(0.99, STK_FS)
    k = f_ref / 220.0
    return 1.0 - (1.0 - r0) * k, 1.0 - (1.0 - d0) * min(k, 1.0)


def brass(f0_env, lip_tension, pressure_env, dur: float, rng: np.random.Generator | None = None,
          vibrato_gain: float = 0.0, vibrato_freq: float = 6.137, tune: bool = True) -> np.ndarray:
    """STK Brass lip-reed (Brass.cpp, research 07 §7.1; Faust brassModel research 08 §3).

    Lip resonance at f * 4**(2 tension - 1) (STK CC2 = Faust brassLipsTable), radius 0.997, lip
    gain 0.03, output area = lip^2 clipped at 1, bore DelayA 2 fs/f + 3, DC blocker 0.99.

    ``pressure_env`` p (0..1) is the breath in units of the lip's static opening: the STK
    breath is p / (0.3 G_dc(f_lip)), G_dc = lip-filter DC gain. UNSOURCED normalisation, found by
    measurement: STK's unnormalised lip filter has G_dc ~ 1/f^2 (36 at 220 Hz, 140 at 110 Hz), so
    at STK's own noteOn pressures (0.5-1.0) the lip is always saturated open and the loop dies;
    it self-oscillates only while the static lip value 0.3 G_dc breath is ~0.3-1. The output is
    scaled back by the same factor. STK vibrato (6.137 Hz) is added to p. ``tune`` rescales bore
    and lip frequency together to cancel the lip's pitch pull (measured +90..+180 cents at
    tension 0.5; one cached calibration render per pitch/pressure).
    Deterministic; ``rng`` unused.
    """
    n = _n(dur)
    f0 = _as_env(f0_env, n)
    lipf = f0 * 4.0 ** (2.0 * _as_env(lip_tension, n) - 1.0)
    p = _as_env(pressure_env, n)
    if vibrato_gain > 0:
        p = p + vibrato_gain * np.sin(2 * np.pi * vibrato_freq * np.arange(n) / SR)
    fm = float(np.median(f0))
    r, dcp = _brass_constants(fm)
    c = 1.0
    if tune:  # global frequency factor on bore and lip together (pitch ~ c); iterate
        lm = float(np.median(lipf))
        pm = float(np.median(p[p > 0.05])) if np.any(p > 0.05) else 0.5
        key = ("brass", round(fm, 3), round(lm / fm, 4), round(pm, 3))
        if key not in _TUNE_CACHE:
            c_ = 1.0
            for _ in range(6):
                m = int(0.8 * SR)
                sc = 1.0 / (0.3 * 0.03 / (1 - 2 * r * math.cos(2 * math.pi * lm * c_ / SR) + r * r))
                y = _brass_kernel(np.full(m, fm * c_), np.full(m, lm * c_),
                                  np.minimum(np.arange(m) / (0.03 * SR), 1.0) * pm, np.full(m, sc),
                                  r, dcp, float(SR), int(2 * SR / (fm * c_ * 0.8)) + 16, 1.0)
                mf = measure_f0(y[int(0.4 * SR):], fm)
                if mf <= 0 or abs(math.log2(mf / fm)) > 0.25:
                    c_ = 1.0
                    break
                if abs(1200 * math.log2(mf / fm)) < 0.5:
                    break
                c_ *= fm / mf
            _TUNE_CACHE[key] = c_
        c = _TUNE_CACHE[key]
    f0 = f0 * c
    lipf = lipf * c
    th = 2 * np.pi * lipf / SR
    scale = 1.0 / (0.3 * 0.03 / (1.0 - 2.0 * r * np.cos(th) + r * r))
    maxlen = int(2 * SR / f0.min() + 16)
    return _brass_kernel(f0, lipf, p, scale, r, dcp, float(SR), maxlen, 1.0)


@njit(cache=True)
def _clarinet_kernel(f0, bp, noise, vib, slope, fs, maxlen):
    """STK Clarinet::tick: pd = -0.95 OneZero(delay) - bp; delay <- bp + pd*Reed(pd)."""
    n = f0.shape[0]
    L = maxlen + 8
    buf = np.zeros(L)
    out = np.empty(n)
    last = 0.0
    fx1 = 0.0
    w = 0
    for i in range(n):
        D = 0.5 * fs / f0[i] - 0.5 - 1.0  # OneZero [0.5,0.5] phase delay 0.5
        b = bp[i]
        b += b * noise[i]
        b += b * vib[i]
        fo = 0.5 * last + 0.5 * fx1
        fx1 = last
        pd = -0.95 * fo - b
        refl = 0.7 + slope * pd
        if refl > 1.0:
            refl = 1.0
        if refl < -1.0:
            refl = -1.0
        buf[w] = b + pd * refl
        last = _dl_read(buf, w, D)
        out[i] = last
        w += 1
        if w == L:
            w = 0
    return out


def clarinet(f0_env, breath_env, dur: float, rng: np.random.Generator, amp: float = 0.8,
             reed_stiffness: float | None = None, noise_gain: float = 0.2,
             vibrato_gain: float = 0.1, vibrato_freq: float = 5.735) -> np.ndarray:
    """STK Clarinet (Clarinet.cpp, research 07 §7.4): reed table offset 0.7, slope -0.3
    (or -0.44 + 0.26 v for ``reed_stiffness`` v, CC2 / Faust clarinetModel); delay
    0.5 fs/f - 1.5; noise 0.2 and vibrato 0.1 at 5.735 Hz multiply the breath; output
    gain amp + 0.001. ``breath_env`` = Envelope value (see ``wind_envelope('clarinet')``).
    """
    n = _n(dur)
    f0 = _as_env(f0_env, n)
    bp = _as_env(breath_env, n)
    noise = noise_gain * rng.uniform(-1.0, 1.0, n)
    vib = vibrato_gain * np.sin(2 * np.pi * vibrato_freq * np.arange(n) / SR)
    slope = -0.3 if reed_stiffness is None else -0.44 + 0.26 * reed_stiffness
    y = _clarinet_kernel(f0, bp, noise, vib, slope, float(SR), int(0.5 * SR / f0.min()) + 8)
    return (amp + 0.001) * y


@njit(cache=True)
def _recorder_kernel(f0, pf, noise, fs, noise_gain, tb0, ta1, ta2, psi, maxlen, jetmax, tune):
    """STK Recorder::tick (Recorder.cpp), verbatim physics (Verge 1997 / Bredholt 2023)."""
    rho = 1.2041
    c0 = 343.21
    lc = 0.02
    h = 0.001
    H = 0.02
    W = 4 * h
    Sp = H * H
    Sm = W * H
    din = 0.0030
    dout = 0.0063
    dm = din + dout
    dd = 0.0035
    rp = math.sqrt(Sp / math.pi)
    b = 0.4 * h
    b2 = Sp / (rho * c0)
    T = 1.0 / fs
    b1 = rho / (4.0 * math.pi * c0 * T * T)
    b3 = dm * Sp / (T * Sm * c0)
    b4 = rho * dout / (Sm * T)
    A = rp * rp / (4 * c0 * c0 * T * T)
    B = 0.82 * rp / (c0 * T)
    a0 = A - B - 1.0  # Iir normalises by a[0]
    rb0 = (1 + A - B) / a0
    rb1 = (B - 2 * A) / a0
    rb2 = A / a0
    ra1 = (B - 2 * A) / a0
    ra2 = A / a0
    vb = np.array([0.83820223947141, -0.16888603248373, -0.64759781930259, 0.07424498608506])
    va = np.array([1.0, -0.33623476246554, -0.71257915055968, 0.14508304017256])
    vin_x = np.zeros(4)
    vin_y = np.zeros(4)
    vout_x = np.zeros(4)
    vout_y = np.zeros(4)
    rx1 = 0.0
    rx2 = 0.0
    ry1 = 0.0
    ry2 = 0.0
    jx1 = 0.0
    jx2 = 0.0
    jy1 = 0.0
    jy2 = 0.0
    ty1 = 0.0
    ty2 = 0.0
    L = maxlen + 8
    pinb = np.zeros(L)
    poutb = np.zeros(L)
    Lj = jetmax + 8
    jetb = np.zeros(Lj)
    jetD = min(200.0, float(jetmax))
    pin_last = 0.0
    pout_last = 0.0
    jet_last = 0.0
    pin = 0.0
    pinm1 = 0.0
    pinm2 = 0.0
    pout = 0.0
    poutm1 = 0.0
    poutm2 = 0.0
    Uj = 0.0
    Qj = 0.0
    Qjm1 = 0.0
    Qjm2 = 0.0
    Q1 = 0.0
    Qp = 0.0
    pm = 0.0
    n = f0.shape[0]
    out = np.empty(n)
    w = 0
    wj = 0
    for i in range(n):
        M = (fs / f0[i] - 4.0 - 3.0) * tune
        pinm2 = pinm1
        pinm1 = pin
        pin = pin_last
        poutm2 = poutm1
        poutm1 = pout
        poutL = pout_last
        # visco-thermal filters (3rd order Iir)
        for k in range(3, 0, -1):
            vin_x[k] = vin_x[k - 1]
            vin_y[k] = vin_y[k - 1]
            vout_x[k] = vout_x[k - 1]
            vout_y[k] = vout_y[k - 1]
        vin_x[0] = pin
        vout_x[0] = poutL
        yi = vb[0] * vin_x[0] + vb[1] * vin_x[1] + vb[2] * vin_x[2] + vb[3] * vin_x[3] \
            - va[1] * vin_y[1] - va[2] * vin_y[2] - va[3] * vin_y[3]
        yo = vb[0] * vout_x[0] + vb[1] * vout_x[1] + vb[2] * vout_x[2] + vb[3] * vout_x[3] \
            - va[1] * vout_y[1] - va[2] * vout_y[2] - va[3] * vout_y[3]
        vin_y[0] = yi
        vout_y[0] = yo
        pin = yi
        poutL = yo
        p_f = pf[i]
        Ujm1 = Uj
        Uj = Ujm1 + T / (rho * lc) * (p_f - pm - 0.5 * rho * Ujm1 * Ujm1)
        Qjm2 = Qjm1
        Qjm1 = Qj
        Qj = h * H * Uj
        Ujs = max(math.sqrt(2 * max(p_f, 0.0) / rho), 0.1)
        fcj = 0.36 / W * Ujs
        gj = 0.002004 * math.exp(-0.06046 * Ujs)
        rj = 0.95 - Ujs * 0.015
        b0j = gj * (1 - rj * rj) / 2
        a1j = -2 * rj * math.cos(2 * math.pi * fcj * T)
        a2j = rj * rj
        eta = b0j * jet_last - b0j * jx2 - a1j * jy1 - a2j * jy2
        jx2 = jx1
        jx1 = jet_last
        jy2 = jy1
        jy1 = eta
        Q1m1 = Q1
        Q1 = b * H * Uj * (1 + math.tanh(eta / (psi * b)))
        pjd = -rho * dd / Sm * (Q1 - Q1m1) / T
        sg = 0.0
        if Qp < 0:
            sg = -1.0
        elif Qp > 0:
            sg = 1.0
        pa = -0.5 * rho * (Qp / (0.6 * Sm)) * (Qp / (0.6 * Sm)) * sg
        tin = noise_gain * noise[i] * 0.5 * rho * Uj * Uj
        pt = tb0 * tin - ta1 * ty1 - ta2 * ty2
        ty2 = ty1
        ty1 = pt
        dp = pjd + pa + pt
        pout = ((b3 - b1 * b2 - 1) * pin + (2 * b1 * b2 - b3) * (pinm1 - poutm1)
                + b1 * b2 * (poutm2 - pinm2) - b1 * (Qj - 2 * Qjm1 + Qjm2)
                + b4 * (Qj - Qjm1) + dp) / (1 - b1 * b2 + b3)
        Qpm1 = Qp
        Qp = Sp / (rho * c0) * (pout - pin)
        pm = pout + pin - dp + rho * din / Sm * (Qp - Qpm1) / T
        Q1d = Q1 - 0.5 * b * H * Uj
        Vac = 2.0 / math.pi * Qp / Sm - 0.38 * Q1d / Sm
        jetb[wj] = Vac
        jet_last = _dl_read(jetb, wj, jetD)
        wj += 1
        if wj == Lj:
            wj = 0
        jetD = min(W / (0.6 * Ujs * T), float(jetmax))
        # radiation filter
        pin_L = rb0 * poutL + rb1 * rx1 + rb2 * rx2 - ra1 * ry1 - ra2 * ry2
        rx2 = rx1
        rx1 = poutL
        ry2 = ry1
        ry1 = pin_L
        poutb[w] = pout
        pinb[w] = pin_L
        pout_last = _dl_read(poutb, w, M)
        pin_last = _dl_read(pinb, w, M)
        w += 1
        if w == L:
            w = 0
        out[i] = pout + pin
    return out


@njit(cache=True)
def _flute_kernel(f0, bp, noise, vib, pole, dcp, jet_ratio, end_refl, jet_refl, fs, maxlen, tune):
    """STK Flute::tick (Flute.h, Flute.cpp): jet delay + cubic JetTable + DC block after the
    jet (GPS 2020) + bore DelayL with one-pole reflection loss; overblown ('lastF = 0.66666 f')."""
    n = f0.shape[0]
    L = maxlen + 8
    bore = np.zeros(L)
    jet = np.zeros(L)
    out = np.empty(n)
    bore_last = 0.0
    fl = 0.0
    dx1 = 0.0
    dy1 = 0.0
    w = 0
    b0 = 1.0 - pole
    for i in range(n):
        lf = f0[i] * 0.66666
        wv = 2.0 * math.pi * lf / fs
        # OnePole phase delay at lastF: H = b0 / (1 - p e^-jw)
        ph = -math.atan2(pole * math.sin(wv), 1.0 - pole * math.cos(wv))
        D = (fs / lf - (-ph / wv) - 1.0) * tune
        b = bp[i]
        b += b * (noise[i] + vib[i])
        fl = b0 * bore_last + pole * fl
        temp = -fl
        pd = b - jet_refl * temp
        jet[w] = pd
        pd = _dl_read(jet, w, D * jet_ratio)
        jt = pd * (pd * pd - 1.0)
        if jt > 1.0:
            jt = 1.0
        if jt < -1.0:
            jt = -1.0
        dc = jt - dx1 + dcp * dy1
        dx1 = jt
        dy1 = dc
        pd = dc + end_refl * temp
        bore[w] = pd
        bore_last = _dl_read(bore, w, D)
        out[i] = 0.3 * bore_last
        w += 1
        if w == L:
            w = 0
    return out


def _flute(f0, bp, rng, amp, noise_gain, vibrato_gain, jet_ratio, tune):
    n = f0.shape[0]
    pole = 0.7 - 0.1 * 22050.0 / SR
    dcp = rate_radius(0.99, STK_FS)
    noise = noise_gain * rng.uniform(-1.0, 1.0, n)
    vib = vibrato_gain * np.sin(2 * np.pi * 5.925 * np.arange(n) / SR)
    maxlen = int(SR / (0.66666 * f0.min())) + 8
    k = 1.0
    if tune:
        fm = float(np.median(f0))
        pm = float(np.median(bp[bp > 0])) if np.any(bp > 0) else 1.0
        key = ("flute", round(fm, 3), round(pm, 3), jet_ratio)
        if key not in _TUNE_CACHE:
            m = int(0.8 * SR)
            y = _flute_kernel(np.full(m, fm), np.minimum(np.arange(m) / (0.03 * SR), 1.0) * pm,
                              np.zeros(m), np.zeros(m), pole, dcp, jet_ratio, 0.5, 0.5, float(SR),
                              maxlen, 1.0)
            mf = measure_f0(y[int(0.4 * SR):], fm)
            _TUNE_CACHE[key] = mf / fm if mf > 0 and abs(math.log2(mf / fm)) < 0.1 else 1.0
        k = _TUNE_CACHE[key]
    return (amp + 0.001) * _flute_kernel(f0, bp, noise, vib, pole, dcp, jet_ratio, 0.5, 0.5,
                                         float(SR), int(maxlen * 1.2), k)


def recorder(f0_env, breath_env, dur: float, rng: np.random.Generator, amp: float = 0.8,
             model: str = "flute", noise_gain: float | None = None, breath_cutoff: float = 500.0,
             softness: float = 1.0, jet_ratio: float = 0.32, vibrato_gain: float = 0.05,
             tune: bool = True) -> np.ndarray:
    """Breathy end-blown flute. ``breath_env=None`` -> the STK noteOn envelope of the model.

    ``model='verge'``: STK Recorder (Recorder.cpp, research 07 §17; Verge/Hirschberg/Causse 1997,
    Bredholt 2023), ported sample-exact (checked against a literal transcription of the C++).
    ``breath_env`` = blowing pressure pf (``wind_envelope('recorder')`` = STK noteOn
    35 (1.1 + 0.2 amp) ADSR). Turbulence noise 0.2 * 0.5 rho Uj^2 through the 500 Hz Q 0.99
    low-pass; ``softness`` psi; output amp/40; visco filter as given; jet-delay cap 200 samples
    scaled to 48 kHz. Measured: the model does not lock to the requested pitch -- it jumps between
    pipe modes (e.g. 1341 Hz for 523 Hz, regardless of pipe length, at STK's pressures), so it is
    kept only as an option for unstable, overblown 'wrong-shape breath' sounds.

    ``model='flute'`` (default): STK Flute jet model (Flute.cpp, research 07 §7.2, Csound
    wgflute): JetTable x^3 - x, jet ratio 0.32, end/jet reflection 0.5, noise 0.15
    (multiplicative turbulence; raise to 0.3-0.4 for shakuhachi-like breath), vibrato 0.05 at
    5.925 Hz, one-pole reflection 0.7 - 0.1*22050/fs, overblown bore fs/(0.66666 f).
    ``breath_env`` = maxPressure*ADSR (``wind_envelope('flute')``). Output amp + 0.001.
    """
    if model == "flute":
        n = _n(dur)
        f0 = _as_env(f0_env, n)
        if breath_env is None:
            breath_env = wind_envelope("flute", dur, amp)
        bp = _as_env(breath_env, n)
        return _flute(f0, bp, rng, amp, 0.15 if noise_gain is None else noise_gain, vibrato_gain,
                      jet_ratio, tune)
    if breath_env is None:
        breath_env = wind_envelope("recorder", dur, amp)
    noise_gain = 0.2 if noise_gain is None else noise_gain
    n = _n(dur)
    f0 = _as_env(f0_env, n)
    pf = _as_env(breath_env, n)
    noise = rng.uniform(-1.0, 1.0, n)
    rq = 2.0 * math.sin(math.pi * breath_cutoff / SR)
    q = 1.0 - rq * 0.99
    tb0, ta1, ta2 = rq * rq, rq * rq - q - 1.0, q
    jetmax = int(round(rate_samples(200.0, STK_FS)))
    maxlen = int(SR / f0.min()) + 8
    k = 1.0
    if tune:
        fm = float(np.median(f0))
        key = ("rec", round(fm, 3), round(float(pf.max()), 2), noise_gain, breath_cutoff, softness)
        if key not in _TUNE_CACHE:
            m = int(0.8 * SR)
            env = np.minimum(np.arange(m) / (0.05 * SR), 1.0) * float(np.median(pf[pf > 0]) if np.any(pf > 0) else 1.0)
            y = _recorder_kernel(np.full(m, fm), env, np.zeros(m), float(SR), noise_gain, tb0, ta1,
                                 ta2, softness, maxlen, jetmax, 1.0)
            mf = measure_f0(y[int(0.4 * SR):], fm)
            _TUNE_CACHE[key] = mf / fm if mf > 0 and abs(math.log2(mf / fm)) < 0.2 else 1.0
        k = _TUNE_CACHE[key]
    y = _recorder_kernel(f0, pf, noise, float(SR), noise_gain, tb0, ta1, ta2, softness,
                         int(maxlen * 1.3), jetmax, k)
    return (amp / 40.0) * y


# ==========================================================================================
# Felt piano: JOS felt hammer + unison stiff strings (Cook dispersion) + MdaPiano laws
# ==========================================================================================

@njit(cache=True)
def _hammer_force(n_max, fs, m, Q0, p, alpha, v0, R2, sub):
    """Mass on a hysteretic nonlinear felt spring against the string's (m s + 2R) load
    (JOS PASP/Ideal_String_Struck_Mass, Nonlinear_Spring_Model, Including_Hysteresis):
    f = Q0 [x^p + alpha d(x^p)/dt] (x in mm, f in N); string point moves at f/2R."""
    F = np.zeros(n_max)
    dt = 1.0 / (fs * sub)
    yh = 0.0
    vh = v0
    ys = 0.0
    xp_prev = 0.0
    started = False
    for i in range(n_max):
        acc = 0.0
        for s in range(sub):
            x = (yh - ys) * 1000.0
            xp = x ** p if x > 0.0 else 0.0
            f = Q0 * (xp + alpha * (xp - xp_prev) / dt)
            xp_prev = xp
            if f < 0.0:
                f = 0.0
            vh -= f / m * dt
            yh += vh * dt
            ys += f / R2 * dt
            acc += f
        F[i] = acc / sub
        if F[i] > 0.0:
            started = True
        elif started and yh < ys:
            break
    return F


@njit(cache=True)
def _stiff_string_kernel(exc, n, D, K, a, h0, h1):
    """Stiff string loop: DelayA + K first-order allpasses (a < 0: high partials travel faster,
    Cook §7.4 / research 04) + JOS brightness/sustain FIR [h1, h0, h1]."""
    L = int(D) + 8
    buf = np.zeros(L)
    st = np.zeros(2)
    out = np.empty(n)
    ax1 = np.zeros(8)
    ay1 = np.zeros(8)
    f1 = 0.0
    f2 = 0.0
    w = 0
    ne = exc.shape[0]
    for i in range(n):
        x = st[0]
        for k in range(K):
            y = a * x + ax1[k] - a * ay1[k]
            ax1[k] = x
            ay1[k] = y
            x = y
        v = h1 * x + h0 * f1 + h1 * f2
        f2 = f1
        f1 = x
        e = exc[i] if i < ne else 0.0
        buf[w] = e + v
        out[i] = _da_read(buf, w, D, st)
        w += 1
        if w == L:
            w = 0
    return out


def _ap_lag(a: float, w: np.ndarray) -> np.ndarray:
    """Phase lag (rad, unwrapped, >= 0) of H = (a + z^-1)/(1 + a z^-1) at w."""
    z = np.exp(-1j * w)
    return -np.unwrap(np.angle((a + z) / (1 + a * z)))


def dispersion_design(f0: float, B: float, n_fit: int = 12, K: int = 4) -> tuple[float, float, int]:
    """Fit K first-order allpasses (coefficient a) and the loop delay D so the loop modes follow
    the stiff-string law f_n = n f0 sqrt(1 + B n^2) / sqrt(1 + B) (JOS PASP/Stiff_String, Cook
    §7.4), fundamental exactly f0. Loop = DelayA D + 1 (lastOut) + 1 (FIR centre) + K allpasses.
    Returns (D, a, K)."""
    N = SR / f0
    K = int(max(0, min(K, (N - 6) // 4)))
    w1 = 2 * np.pi * f0 / SR
    nmax = max(2, min(n_fit, int(0.45 * SR / f0 / math.sqrt(1 + B * n_fit ** 2)) or 2))
    ns = np.arange(1, nmax + 1)
    target = ns * f0 * np.sqrt(1 + B * ns ** 2) / math.sqrt(1 + B)
    grid = np.linspace(1e-4, np.pi, 8192)
    best = (N - 2.0, 0.0, 1e18)
    if K == 0 or B <= 0:
        return N - 2.0 - K, 0.0, K
    for a in np.linspace(-0.9, 0.0, 181):
        lag = _ap_lag(a, grid)
        lag1 = float(np.interp(w1, grid, lag))
        D = (2 * np.pi - K * lag1) / w1 - 2.0
        if D < 1.0:
            continue
        theta = grid * (D + 2.0) + K * lag
        wn = np.interp(2 * np.pi * ns, theta, grid)
        err = np.sum((1200 * np.log2(wn * SR / (2 * np.pi) / target)) ** 2)
        if err < best[2]:
            best = (D, a, err)
    return best[0], best[1], K


def felt_piano(f0: float, velocity: float, dur: float, rng: np.random.Generator,
               felt: float = 0.8, brightness: float = 0.35, inharmonicity: float = 4e-4,
               decay: float = 0.5, soundboard: float = 0.3) -> np.ndarray:
    """Felt piano note (JOS PASP hammer + commuted soundboard, research 10 §5; stiffness
    Cook §7.4; sc3 MdaPiano decay/muffle laws, research 11 §C6).

    1. Hammer: key n = 12 log2(f0/27.5) + 1; Q0 = 183 e^(0.045 n) N/mm^p, p = 3.7 + 0.015 n,
       alpha = 248 + 1.83 n - 0.055 n^2 us, m = 11.074 - 0.074 n + 0.0001 n^2 g, integrated
       against the strings' 2R load (JOS 'mass into (m s + 2R)') -> force pulse; the string
       gets velocity f/2R. ``felt`` softens Q0 (moderator felt; UNSOURCED amount).
    2. Excitation = pulse + ``soundboard`` * pulse convolved with a noise tail whose low-pass
       contracts (JOS 'soundboard tap', UNSOURCED numbers), then the strike-position comb at
       1/8 of the string (UNSOURCED position).
    3. Three unison strings detuned [-0.05, 0, 0.04] semitones (SC synthetic piano). Each loop:
       DelayA + K first-order allpasses fitted to f_n = n f0 sqrt(1 + B n^2) (B =
       ``inharmonicity``, UNSOURCED default 4e-4; Cook gives ~0.004 for piano strings) + JOS
       loop FIR with sustain S = 6.91 / exp(-0.6 + 0.033 note - L) (MdaPiano decay law,
       note >= 44, L from ``decay``) and brightness B.
    4. MdaPiano muffle low-pass: l = 50 + felt^2 160 + 3.2 (vel - 64), clamp [55 + 0.25 note,
       210], ff = l^2/fs (sonic-pi muffle 0.8, velmuff 0.8 -> muffvel 3.2).
    """
    n = _n(dur)
    key = 12.0 * math.log2(f0 / 27.5) + 1.0
    note = key + 20.0  # MIDI
    Q0 = 183.0 * math.exp(0.045 * key) * (1.0 - 0.9 * felt)  # UNSOURCED: felt softening
    p = 3.7 + 0.015 * key
    alpha = (248.0 + 1.83 * key - 0.055 * key * key) * 1e-6
    m = (11.074 - 0.074 * key + 0.0001 * key * key) * 1e-3
    R = 2.2  # UNSOURCED: steel piano string wave impedance per side (kg/s)
    R2 = 3 * 2 * R  # three unison strings, both sides
    v0 = 0.4 + 4.6 * float(np.clip(velocity, 0.0, 1.0))  # UNSOURCED: hammer speed m/s
    F = _hammer_force(int(0.05 * SR), float(SR), m, Q0, p, alpha, v0, R2, 16)
    F = np.trim_zeros(F, "b")
    vinj = F / (2 * R)
    nt = int(0.3 * SR)  # soundboard tap (JOS PASP/Excitation_Synthesis). UNSOURCED numbers.
    tail = _tv_lowpass_noise(nt, 6000.0, 300.0, rng) * np.exp(-6.91 * np.arange(nt) / SR / 0.1)
    tail /= np.sqrt(np.sum(tail ** 2))
    sb = soundboard * np.convolve(vinj, tail) * math.sqrt(SR / 1000.0)
    exc = np.pad(vinj, (0, len(sb) - len(vinj))) + sb
    N = SR / f0
    mm = int(math.floor(N / 8.0 + 0.5))  # UNSOURCED: hammer at 1/8 of the string
    if mm > 0:
        exc[mm:] -= exc[:-mm].copy()
    nd = max(note, 44.0)
    Ld = 2.0 * decay
    if Ld < 1.0:
        Ld += 0.25 - 0.5 * decay
    S = 6.91 / math.exp(-0.6 + 0.033 * nd - Ld)  # MdaPiano decay law -> T60 in s
    Bb = brightness
    out = np.zeros(n)
    for dsemi in (-0.05, 0.0, 0.04):
        f = f0 * 2.0 ** (dsemi / 12.0)
        g0 = math.exp(-6.91 / f / S)
        h0, h1 = g0 * (1 + Bb) / 2, g0 * (1 - Bb) / 4
        D, a, K = dispersion_design(f, inharmonicity)
        out += _stiff_string_kernel(exc, n, D, K, a, h0, h1)
    vel = 127.0 * velocity
    lm = 50.0 + felt * felt * 160.0 + 0.8 * 0.8 * 5.0 * (vel - 64.0)
    lm = min(max(lm, 55.0 + 0.25 * note), 210.0)
    ff = lm * lm / SR
    # MdaPiano: f0 += ff (x + x1 - f0)  ->  y[n] = (1 - ff) y[n-1] + ff (x[n] + x[n-1])
    out = signal.lfilter([ff, ff], [1.0, -(1.0 - ff)], out)
    return 0.005 * out  # UNSOURCED: output scale (velocity dynamics kept, peak < 1)


# ==========================================================================================
# Measurement (self-test)
# ==========================================================================================

def _acf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    nfft = 1 << int(math.ceil(math.log2(2 * len(x))))
    X = np.fft.rfft(x * np.hanning(len(x)), nfft)
    r = np.fft.irfft(np.abs(X) ** 2)[: len(x)]
    return r / max(r[0], 1e-300)


def measure_f0(x: np.ndarray, f_hint: float | None = None, fmin: float = 25.0,
               fmax: float = 2000.0, return_quality: bool = False):
    """Autocorrelation f0 (Hann window, normalised ACF, parabolic peak interpolation).

    Candidates are the ACF peaks >= 0.9 of the largest one. Without ``f_hint`` the smallest lag
    wins (avoids sub-octave errors); with a hint the candidate closest to it in log frequency
    wins (a missing-fundamental bass or a quasi-periodic brass tone still reports its period).
    ``return_quality`` also returns r(T)/r_max of the chosen lag (1.0 = the strongest period).
    """
    x = np.asarray(x, float)
    if np.max(np.abs(x - np.mean(x))) < 1e-9:
        return (0.0, 0.0) if return_quality else 0.0
    r = _acf(x)
    lo, hi = int(SR / fmax), min(int(SR / fmin), len(r) - 2)
    seg = r[lo:hi]
    pk = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if not pk:
        return (0.0, 0.0) if return_quality else 0.0
    best = max(seg[i] for i in pk)
    cand = [i for i in pk if seg[i] >= 0.9 * best]
    if f_hint:
        i = min(cand, key=lambda j: abs(math.log(SR / (lo + j) / f_hint)))
    else:
        i = cand[0]
    a, b, c = seg[i - 1], seg[i], seg[i + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    f = SR / (lo + i + d)
    return (f, float(seg[i] / best)) if return_quality else f


def measure_partial(x: np.ndarray, f: float, cents: float = 50.0) -> float:
    """Frequency of the partial near ``f``: power-weighted centroid of the zero-padded Hann
    spectrum within +-``cents`` (for inharmonic sounds -- bars, glass, stiff strings -- where an
    autocorrelation 'f0' is a compromise between stretched partials)."""
    x = np.asarray(x, float) - np.mean(x)
    nf = 1 << max(int(math.ceil(math.log2(len(x) * 8))), 16)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), nf)) ** 2
    fr = np.fft.rfftfreq(nf, 1 / SR)
    band = np.abs(1200 * np.log2(np.maximum(fr, 1e-9) / f)) < cents
    if not np.any(band) or np.sum(X[band]) <= 0:
        return 0.0
    k = np.argmax(np.where(band, X, 0.0))  # centroid around the strongest bin of the band
    sel = band & (np.abs(fr - fr[k]) < 3 * SR / len(x))
    return float(np.sum(fr[sel] * X[sel]) / np.sum(X[sel]))


def _cents(f, ref):
    return 1200.0 * math.log2(f / ref) if f > 0 else float("nan")


def _harmonicity(x: np.ndarray, f0: float, nh: int = 30) -> tuple[float, float]:
    """(fraction of spectral energy within +-15 cents... +-3 % of harmonics k f0, ACF peak)."""
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    mask = np.zeros_like(fr, bool)
    for k in range(1, nh + 1):
        mask |= np.abs(fr - k * f0) < 0.03 * f0
    band = (fr > 0.5 * f0) & (fr < (nh + 0.5) * f0)
    frac = X[mask & band].sum() / X[band].sum()
    lag = int(round(SR / f0))
    ac = np.dot(x[:-lag], x[lag:]) / math.sqrt(np.dot(x[:-lag], x[:-lag]) * np.dot(x[lag:], x[lag:]))
    return float(frac), float(ac)


if __name__ == "__main__":
    import time

    out_dir = DEV_DIR / "waveguides"
    rng = np.random.default_rng(7)
    results = []

    def check(name, y, f_target, seg=(0.3, 0.9), note=""):
        finite = bool(np.all(np.isfinite(y)))
        peak = float(np.max(np.abs(y))) if finite else float("nan")
        a, b = int(seg[0] * SR), int(seg[1] * SR)
        fm, qual = measure_f0(y[a:b], f_target, return_quality=True) if (f_target and finite) else (0.0, 0.0)
        note = (f"r(T)/rmax {qual:.3f}  " if f_target else "") + note
        c = _cents(fm, f_target) if f_target else float("nan")
        stable = finite and peak < 10.0
        write_wav(out_dir / f"{name}.wav", y / max(peak, 1e-9) * 0.5 if finite else np.zeros(10))
        ok = stable and (not f_target or abs(c) <= 5.0)
        results.append((name, ok))
        ftxt = f"target {f_target:8.2f} measured {fm:8.2f} Hz {c:+7.2f} c" if f_target else " " * 46
        print(f"{name:28s} {ftxt}  peak {peak:8.4f}  finite {finite}  {'PASS' if ok else 'FAIL'}  {note}")
        return fm

    def helmholtz(name, y, f, seg=(1.0, 3.0)):
        x = y[int(seg[0] * SR):int(seg[1] * SR)]
        x = x - x.mean()
        frac, ac = _harmonicity(x, f)
        X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        fr = np.fft.rfftfreq(len(x), 1 / SR)
        h = [X[(np.abs(fr - k * f) < 0.03 * f)].max() for k in (1, 2, 3)]
        rms = np.sqrt(np.convolve(x ** 2, np.ones(4800) / 4800, "valid")[::4800])
        cv = float(np.std(rms) / np.mean(rms))
        ok = frac > 0.95 and ac > 0.9 and cv < 0.1 and 20 * np.log10(max(h) / h[0]) < 40.0
        if not name.startswith("trompette"):  # the chien string buzzes on purpose: info only
            results.append((f"helmholtz_{name}", ok))
        verdict = ("info (buzz on purpose)" if name.startswith("trompette")
                   else ("PASS" if ok else "FAIL"))
        print(f"   Helmholtz {name}: harmonic energy {frac * 100:.1f} %, ACF(T0) {ac:.3f}, "
              f"RMS CV {cv * 100:.1f} %, H1-H2 {20 * np.log10(h[0] / h[1]):+.1f} dB, "
              f"H1-H3 {20 * np.log10(h[0] / h[2]):+.1f} dB  {verdict}")

    t0 = time.time()
    print("-- plucked")
    check("pluck_guitar_G3", pluck(196.0, 3.0, rng, position=0.2, body="guitar"), 196.0, (0.2, 0.8))
    check("pluck_bass_G1_pizz", pluck(49.0, 3.0, rng, body="bass", brightness=0.3), 49.0, (0.3, 1.2))
    check("pluck_bare_A4", pluck(440.0, 2.0, rng, body="none"), 440.0, (0.2, 0.8))
    have = _load_mand(0) is not None
    check("mandolin_G4", mandolin(392.0, 2.5, rng), 392.0, (0.15, 0.6),
          note=f"mand1.raw {'loaded' if have else 'ABSENT (synthetic body)'}")

    print("-- bowed (STK Bowed, pressure 0.8)")
    for body, f in (("violin", 440.0), ("cello", 130.81), ("bass", 55.0)):
        y = bowed_string(f, 0.8, bow_envelope(4.0, 0.8, release=3.5), 4.0, rng, body=body, vibrato=False)
        check(f"bowed_{body}", y, f, (1.0, 3.0))
        helmholtz(body, y, f)
    y = bowed_string(130.81, 0.8, bow_envelope(5.0, 0.8, release=4.5), 5.0, rng, body="cello")
    check("bowed_cello_vibrato", y, 130.81, (0.1, 0.45), note="pitch before the vibrato onset")
    y = bowed_string(130.81, 0.5, bow_envelope(3.0, 0.8, release=2.5), 3.0, rng, body="cello", vibrato=False)
    fm = measure_f0(y[SR:int(2.5 * SR)])
    print(f"   (pressure 0.5 at the same bow speed: measured {fm:.2f} Hz = double-slip octave, "
          "Schelleng minimum bow force; not a failure of the port)")

    print("-- banded waveguides")
    check("banded_tuned_bar_bow", banded("tuned_bar", 220.0, 3.0, rng, mode="bow"), 220.0, (1.0, 2.5))
    y = banded("glass_harmonica", 523.25, 3.0, rng, mode="bow")
    check("banded_glass_bow", y, 0.0, note="inharmonic: pass on the mode-1 partial below")
    fp = measure_partial(y[SR:int(2.5 * SR)], 523.25)
    fa = measure_f0(y[SR:int(2.5 * SR)], 523.25)
    results.append(("banded_glass_bow_partial", abs(_cents(fp, 523.25)) <= 5.0))
    print(f"   glass mode-1 partial {fp:.2f} Hz ({_cents(fp, 523.25):+.2f} c) "
          f"{'PASS' if abs(_cents(fp, 523.25)) <= 5 else 'FAIL'}; ACF 'f0' {fa:.2f} Hz "
          f"({_cents(fa, 523.25):+.2f} c) mixes the inharmonic modes 1, 2.32, 4.25 ...")
    y = banded("tuned_bar", 220.0, 3.0, rng, mode="bow", exact_pitch=False)
    fm = measure_f0(y[SR:int(2.5 * SR)])
    print(f"   STK integer delays (exact_pitch=False): tuned bar bowed at 220 Hz -> {fm:.2f} Hz "
          f"({_cents(fm, 220.0):+.1f} c)")
    for pre, f in (("tibetan_bowl", 180.0), ("uniform_bar", 440.0)):
        y = banded(pre, f, 6.0, rng, mode="strike")
        check(f"banded_{pre}_strike", y, 0.0)
        pk = spectrum_peaks(y[: 2 * SR], 4, 50.0, 8000.0)
        print(f"   {pre} strongest peaks (Hz): {[round(float(q[0]), 1) for q in pk]}")

    print("-- winds")
    for f in (110.0, 220.0, 440.0):
        y = brass(f, 0.5, wind_envelope("brass", 2.5, 0.8, release=2.2), 2.5, rng)
        check(f"brass_{f:.0f}", y, f, (0.6, 2.0))
    for f in (523.25, 784.0):
        y = recorder(f, None, 2.5, rng)
        check(f"flute_{f:.0f}", y, f, (0.6, 2.0))
    y = recorder(523.25, wind_envelope("recorder", 2.5, 0.8, release=2.2), 2.5, rng, model="verge", tune=False)
    check("recorder_verge_523", y, 0.0)
    fm = measure_f0(y[int(0.6 * SR):2 * SR])
    print(f"   Verge/STK Recorder asked for 523.25 Hz plays {fm:.1f} Hz (pipe-mode jump; see docstring)")
    check("clarinet_D3", clarinet(146.83, wind_envelope("clarinet", 2.5, 0.8, release=2.2), 2.5, rng),
          146.83, (0.5, 2.0))

    print("-- hurdy-gurdy")
    y = hurdy_gurdy(196.0, 4.0, rng, strings=("trompette",))
    check("hurdy_trompette_G3", y, 196.0, (1.0, 3.5), note="chien string alone")
    helmholtz("trompette_chien", y, 196.0, (1.0, 3.5))
    y2 = hurdy_gurdy(196.0, 4.0, rng, strings=("trompette",), buzz=1e9)
    X1 = np.abs(np.fft.rfft(y[SR:3 * SR])); X2 = np.abs(np.fft.rfft(y2[SR:3 * SR]))
    fr = np.fft.rfftfreq(2 * SR, 1 / SR)
    hi = fr > 3000
    print(f"   chien buzz: energy above 3 kHz {10 * np.log10(np.sum(X1[hi] ** 2) / np.sum(X1 ** 2)):+.1f} dB "
          f"vs {10 * np.log10(np.sum(X2[hi] ** 2) / np.sum(X2 ** 2)):+.1f} dB with the chien disabled")
    check("hurdy_gurdy_G3_full", hurdy_gurdy(196.0, 6.0, rng, melody=[392.0, 392.0, 440.0, 392.0]),
          0.0, note="trompette + bourdon + mouche + chanterelle")

    print("-- felt piano (pass = fundamental partial within 5 c; ACF pitch shows the stretch)")
    for f, v in ((65.41, 0.6), (261.63, 0.3), (261.63, 0.9), (880.0, 0.5)):
        y = felt_piano(f, v, 4.0, rng)
        x = y[int(0.1 * SR):int(1.5 * SR)]
        nf = 1 << 22
        X = np.abs(np.fft.rfft(x * np.hanning(len(x)), nf)) ** 2
        fr = np.fft.rfftfreq(nf, 1 / SR)
        band = np.abs(1200 * np.log2(np.maximum(fr, 1e-9) / f)) < 25.0  # the 3 unison strings
        f1 = float(np.sum(fr[band] * X[band]) / np.sum(X[band]))
        c1 = _cents(f1, f)
        fa = measure_f0(y[int(0.1 * SR):int(0.8 * SR)], f)
        finite = bool(np.all(np.isfinite(y)))
        ok = finite and abs(c1) <= 5.0 and float(np.max(np.abs(y))) < 10
        results.append((f"felt_piano_{f:.0f}_v{v}", ok))
        write_wav(out_dir / f"felt_piano_{f:.0f}_v{v}.wav", y / np.max(np.abs(y)) * 0.5)
        print(f"felt_piano_{f:.0f}_v{v:<18} target {f:8.2f} partial-1 {f1:8.2f} Hz {c1:+6.2f} c; "
              f"ACF {fa:8.2f} Hz {_cents(fa, f):+6.2f} c  peak {np.max(np.abs(y)):.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
        pk = spectrum_peaks(y[int(0.05 * SR):SR], 6, 0.8 * f, 7.5 * f)
        ratios = sorted(round(float(q[0]) / f, 3) for q in pk)
        print(f"   partial ratios {ratios}; T60 (fit over first 30 dB) {6.91 / decay_rate(y, 0.05):.2f} s")
    F_soft = _hammer_force(int(0.05 * SR), float(SR), 9.9e-3, 183 * math.exp(0.045 * 40) * 0.28,
                           4.3, 2.6e-4, 0.4 + 4.6 * 0.2, 13.2, 16)
    F_hard = _hammer_force(int(0.05 * SR), float(SR), 9.9e-3, 183 * math.exp(0.045 * 40) * 0.28,
                           4.3, 2.6e-4, 0.4 + 4.6 * 0.9, 13.2, 16)
    for lab, F in (("soft", F_soft), ("hard", F_hard)):
        nz = np.nonzero(F > 0.02 * F.max())[0]
        print(f"   hammer {lab}: peak force {F.max():.1f} N, contact {len(nz) / SR * 1000:.2f} ms")
    print(f"total {time.time() - t0:.1f} s")
    bad = [r[0] for r in results if not r[1]]
    print("FAILURES:", bad if bad else "none")
