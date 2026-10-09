# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Ambience family: act/library beds (seamless loops) and place detail one-shots.

Every emitter is a port of a named source (research notes in ``out/research``):

* **Wind** - Farnell, *Designing Sound* Practical 18 and the decoded patch
  ``PureData_examples/11-wind/main-wind4_mod.pd`` (research 02 §13, 05 §8.1): windspeed =
  base + square-law gust (own noise, lop~ 0.5 x2, x50, x((b+0.5)^2-0.125)) + squall (own noise,
  lop~ 3 x2, x20, x((max(b,0.4)-0.4)*8)^2), clip 0..1, read by every emitter through its own
  propagation delay (0/100/300/500/900/3000 ms): static bp~ 800 Q1 x (w+0.2) with a moving rzero~
  notch, two doorway howls inside speed windows (bp~ 400/200 Q40 AM'd by 30+200a / 20+100a Hz),
  two Q60 wire whistles (600+400w, 1000+1000w Hz, amplitude (w+0.12)^2 and w^2), leaves
  (thresholded noise, 3 s lag). Base control: the patch's 0.1 Hz osc~, or (``control="walk"``) the
  "slow random walk" shared control of Farnell AM07 (research 09 D2.3) / rjlib ``c_drunk`` (A7).
* **Creak** - Farnell Practical 9 stick-slip (research 02 §4, 05 §2.3): force-dependent
  metronome (1-f)60+3 ms + U 6(1-f), threshold f > 0.3, pulse sqrt(sqrt(x)+0.1) falling over
  sqrt(x) ms then squared (x = ms since last slip / 100), door formants 62.5/125/250/395/560/790 Hz
  (Q 1,1,2,2,3,3) + 0.2 dry, panel of 8 delay-feedback elements 4.52...16 ms fb 0.05, hip~ 125, x0.2.
  Iron: the same stick-slip train drives the measured sheet-metal box (modalpaper computertower.sy,
  research 06 App. A) through ``modal.render_modes``.
* **Fire** - Farnell Practical 11 + thesis p.316-325 (research 02 §6, 05 §8.2): one shared noise
  per generator; hiss = HP'd noise x (lop~1 noise)^4, crackles triggered when env~ of lop~1 noise
  sits in the 50-51 dB window (square-law 0-30 ms bursts through bp~ Q1 at 1.5-16.5 kHz / 100-1000 Hz),
  lapping = bp~ 30 Q5 x100, hip~ 25 x2, clip +-0.9, x0.6; mix 0.2/0.3/0.6; big fire = 4 generators
  with own noises through bp 600 Q0.2 / 1200 Q0.6 / 2600 Q0.4 / hip 1000, x0.2.
* **Running water** - Reshetnikov ``03-water/abs/water.pd`` (research 02 §8): per instance a metro
  of 10 + U 68(1-d^2) ms, centre 300 + 15700 r^5 Hz glided over 100-160 ms, Q 25 + 20(1-r),
  grain (1-r)^(12d+1) falling to 0 in 1 ms then squared, x(1-d)^2, vcf~, pan p in [0,0.5).
  Farnell's own flowing-water voice (thesis T p.336-342, research 05 §8.3): bilinear-exponential
  targets every 6 ms x1600 + 800 Hz, slew 2.689 ms, amplitude lop~10(positive difference^2).
* **Rain** - Farnell Practical 15 (research 02 §10, 05 §8.4): Poisson-timed parabolic clicks
  (width 0.1-12 ms, amplitude sqrt(w)+0.1, generators (50,12,0.15) (60,12,0.2) (70,12,0.4) ...,
  sum x0.1, hip 900), pink comfort noise (rjlib s_pinknoise) with a 300 Hz band, drops on water =
  Moss bubbles with the rain radius law alpha = 2.9 (research 06 §3.3).
* **Thunder** - Farnell ``thunder/main-thunder4.pd`` decoded (research 02 §12.2): strike pattern
  accumulator, 4 round-robin strike voices (T = 200(1.4-v)^5 ms, bp 100+1200v and half, Q3), rumble
  (two independent noises, lop at a density line 2000->0 over 9 s, phasor 3000x+1, samphold,
  rpole 0.99, parabolic pulse train, hip 100, 8 s decay), afterimage (stuttered noise, bp 333 Q4,
  6 s), deep noise (lop 80 x2, hip 15, x12 clip, lop 80 x2, 1 s delay, 2 s rise, 20 s fall),
  travelling distance filter (3 lop at (1-d)6000+50, comb 2+3d ms), 24-tap echo box (12 s).
* **Surf** - Moss et al. "Sounding liquids" (research 06 §3): Minnaert bubbles with the breaking-wave
  radius power law alpha 1.5-3.3, formation rate following u^2 kappa (the wave crest), plus
  non-spherical (tube) bubbles radiating at 2 f_n near the radial mode (eq. 8-12); Farnell: a stormy
  sea is close to white noise (thesis T p.338); line source -3 dB per doubling (research 01 §5).
* **Drips / bubbles** - ``particles.drip`` (STK water drop + Farnell impact click) as quantised
  (relaxation) flow, ``particles.surfacing_bubble`` scheduled by Farnell's prime selector (15 ms
  counter mod 200, primes 29 37 47 67 89 113 157 197, 50 % culling; research 05 §8.3).
* **Chains / keys** - Farnell shell-casing tinkle ``06-guns/abs/shellcasings.pd`` (research 03 §5.3:
  noise x e^4 (0-39 ms) -> 3 bp~ Q800 at f0 4.5-5.5 kHz, f0+700+U100, +300+U100, x100 clip +-0.9,
  bounces 75 / +170-344 / +75-149 ms with gains 0.416 0.257 0.05), reload click-factory end-stop
  cluster (345 Q22 70 ms, 256 Q3 45 ms, 11023 Q46 11 ms), rolling-can momentum (500 ms half-cosine
  pushes into a low-pass, amplitude root law; research 02 §3, 05 §8.7); keys = Farnell ``pd tamb``
  jingles (bp Q217 at 1700/3149/2330/1511 Hz, decays d 2d 4d 8d, ring-mod 6300 Hz x0.6, second hit
  3 sep ms later; research 09 C8.2).
* **Void / pressure** - sonic-pi ``dark_ambience`` (atmos.clj, research 11 D3): pink noise x0.005 ->
  Ringz at note 52, +12, +24 semitones (ring 0.2 s), mix [s1 s1 s2 s3] -> big reverb -> tanh ->
  RLPF 4186 Hz rq 0.3. zita_rev1 stands in for GVerb (space.py has no GVerb).
* **Hum** - Farnell Practical 16 (research 02 §11): phasors 99.8 + 100.2 Hz, summed, -1, clip.
* **Space** - ``space.zita_rev1`` (caves), Farnell's outdoor building echoes from the siren
  practical (165/121/33 ms summed and recirculated x0.1, echo x0.05, direct x0.2; research 05 §8.6),
  ``space.distance``-style spreading and air absorption (applied circularly for loops).

Loops (doc 07 §5.2): 120 s, seamless. Two techniques, both exact: (1) *circular filtering* - every
excitation is built n-periodic, the whole chain runs over two periods and the second is kept
(the steady state of a periodic input through stable filters is periodic); (2) *folding* - event
tails that run past the loop end are added back at its start. Beds are stereo, mono-safe
(independent noises per side for broadband layers, point emitters panned, no polarity tricks).
Details are 6 mono one-shots per place, 2-10 s.

Everything is 48 kHz, deterministic (explicit rng), per-sample numba filters (Pd formulas from
``d_filter.c`` / ``d_osc.h``: lop~, hip~, bp~, vcf~, rzero~, rpole~).
"""

from __future__ import annotations

import math
import re
import subprocess
import zlib
from pathlib import Path

import numba
import numpy as np

from . import analysis as A
from . import modal as M
from . import particles as P
from . import space as S
from .core import (SR, DEV_DIR, fit, highpass, lowpass, master, pan_gains, pow_decay, seconds,
                   spectrum_peaks, tv_lowpass, write_wav)

FAMILY_PREFIXES: tuple[str, ...] = ("amb.act.", "amb.place.", "amb.library.")

BED_SECONDS = 120.0          # doc 07 §5.2 / direction.toml: 90-180 s seamless loop
N_DETAILS = 6                # direction.toml [audio.classes.ambience_detail]
PD_FS = 44100.0              # the Pd patches were tuned at 44.1 kHz
# Low-passed noise std scales with sqrt(fc / fs): x sqrt(48000/44100) keeps the Pd patch's control
# statistics (thresholds, square laws) identical at 48 kHz (PLAN rule 2, rate conversion).
PDN = math.sqrt(SR / PD_FS)
_PI_PD = 3.14159             # Pd's own constant in d_filter.c

OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "samples_v2"
MANIFEST = Path(r"C:\Users\bilal\Documents\SS\game\assets\production\manifest.toml")


# ==========================================================================================
# Pd filters, per-sample (d_filter.c, d_osc.h)
# ==========================================================================================

@numba.njit(cache=True)
def _lop_k(x, f):
    """Pd lop~ with a per-sample cutoff array: coef = f 2pi/sr clipped to [0,1]; y += coef (x - y)."""
    n = x.shape[0]
    y = np.empty(n)
    last = 0.0
    for i in range(n):
        c = f[i] * (2.0 * _PI_PD) / 48000.0
        if c < 0.0:
            c = 0.0
        elif c > 1.0:
            c = 1.0
        last = c * x[i] + (1.0 - c) * last
        y[i] = last
    return y


@numba.njit(cache=True)
def _hip_k(x, f):
    """Pd hip~ (normalised form, Pd >= 0.44)."""
    coef = 1.0 - f * (2.0 * _PI_PD) / 48000.0
    if coef < 0.0:
        coef = 0.0
    n = x.shape[0]
    y = np.empty(n)
    if coef >= 1.0:
        for i in range(n):
            y[i] = x[i]
        return y
    normal = 0.5 * (1.0 + coef)
    last = 0.0
    for i in range(n):
        new = x[i] + coef * last
        y[i] = normal * (new - last)
        last = new
    return y


@numba.njit(cache=True)
def _qcos(f):
    if -0.5 * 3.14159 <= f <= 0.5 * 3.14159:
        g = f * f
        return ((g * g * g * (-1.0 / 720.0) + g * g * (1.0 / 24.0)) - g * 0.5) + 1.0
    return 0.0


@numba.njit(cache=True)
def _bp_k(x, f, q):
    """Pd bp~ (sigbp_docoef / sigbp_perform) with per-sample centre and Q."""
    n = x.shape[0]
    y = np.empty(n)
    last = 0.0
    prev = 0.0
    for i in range(n):
        ff = f[i]
        if ff < 0.001:
            ff = 10.0
        qq = q[i] if q[i] > 0.0 else 0.0
        omega = ff * (2.0 * _PI_PD) / 48000.0
        omr = 1.0 if qq < 0.001 else omega / qq
        if omr > 1.0:
            omr = 1.0
        r = 1.0 - omr
        c1 = 2.0 * _qcos(omega) * r
        c2 = -r * r
        gain = 2.0 * omr * (omr + r * omega)
        out = x[i] + c1 * last + c2 * prev
        y[i] = gain * out
        prev = last
        last = out
    return y


@numba.njit(cache=True)
def _vcf_k(x, f, q):
    """Pd vcf~ (SIGVCFPERF): complex one-pole resonator, band-pass (real) output."""
    n = x.shape[0]
    y = np.empty(n)
    re_ = 0.0
    im = 0.0
    isr = 6.28318 / 48000.0
    for i in range(n):
        qq = q[i]
        qinv = 1.0 / qq if qq > 0.0 else 0.0
        amp = 2.0 - 2.0 / (qq + 2.0)
        cf = f[i] * isr
        if cf < 0.0:
            cf = 0.0
        r = 1.0 - cf * qinv if qinv > 0.0 else 0.0
        if r < 0.0:
            r = 0.0
        omr = 1.0 - r
        cr = r * math.cos(cf)
        ci = r * math.sin(cf)
        re2 = re_
        re_ = amp * omr * x[i] + cr * re2 - ci * im
        im = ci * re2 + cr * im
        y[i] = re_
    return y


@numba.njit(cache=True)
def _rzero_k(x, c):
    """Pd rzero~ with a signal coefficient: y[n] = x[n] - c[n] x[n-1]."""
    n = x.shape[0]
    y = np.empty(n)
    last = 0.0
    for i in range(n):
        y[i] = x[i] - c[i] * last
        last = x[i]
    return y


@numba.njit(cache=True)
def _rpole_k(x, a):
    """Pd rpole~: y[n] = x[n] + a y[n-1]."""
    n = x.shape[0]
    y = np.empty(n)
    last = 0.0
    for i in range(n):
        last = x[i] + a * last
        y[i] = last
    return y


@numba.njit(cache=True)
def _dfb_k(x, d, fb):
    """Farnell ``dfbef`` delay-feedback element: w = x + fb * w[n-d]; output w[n-d]."""
    n = x.shape[0]
    buf = np.zeros(d)
    y = np.empty(n)
    j = 0
    for i in range(n):
        out = buf[j]
        buf[j] = x[i] + fb * out
        y[i] = out
        j += 1
        if j >= d:
            j = 0
    return y


def _arr(v, n: int) -> np.ndarray:
    if np.ndim(v) == 0:
        return np.full(n, float(v))
    return np.ascontiguousarray(v, dtype=np.float64)


def lop(x, f):
    x = np.ascontiguousarray(x, dtype=np.float64)
    return _lop_k(x, _arr(f, len(x)))


def hip(x, f: float):
    return _hip_k(np.ascontiguousarray(x, dtype=np.float64), float(f))


def bp(x, f, q):
    x = np.ascontiguousarray(x, dtype=np.float64)
    return _bp_k(x, _arr(f, len(x)), _arr(q, len(x)))


def vcf(x, f, q):
    x = np.ascontiguousarray(x, dtype=np.float64)
    return _vcf_k(x, _arr(f, len(x)), _arr(q, len(x)))


def rzero(x, c):
    x = np.ascontiguousarray(x, dtype=np.float64)
    return _rzero_k(x, _arr(c, len(x)))


def pd_pan(p: float) -> float:
    """Pd equal-power pan p in [0,1] (L = cos(pi p/2)) -> core pan in [-1, 1] (same law)."""
    return 2.0 * float(p) - 1.0


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """rjlib ``s_pinknoise`` (Kellet/RBJ 3-pole filter, research 09 A1.1), x0.17."""
    from scipy.signal import lfilter
    b = np.poly([0.984436, 0.833923, 0.0756836])
    a = np.poly([0.995728, 0.947906, 0.535675])
    return 0.17 * lfilter(b, a, rng.uniform(-1, 1, n))


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-30))


def _norm(x) -> np.ndarray:
    """Unit-RMS copy (for scene mixing; levels are then set in dB by the scene table)."""
    return np.asarray(x) / _rms(x)


def _db(g: float) -> float:
    return 10 ** (g / 20)


# ==========================================================================================
# Working domain: one-shot, or seamless loop by circular filtering / folding
# ==========================================================================================

class Dom:
    """Rendering domain of ``dur`` seconds.

    loop=False: plain one-shot of n samples.
    loop=True: excitations are n-periodic; ``tile`` repeats them over m = 2n samples, the chain
    runs over m and ``out`` keeps the second period (exact steady state -> seamless).
    ``fold`` adds event tails that overrun the period back onto its start.
    """

    def __init__(self, dur: float, loop: bool):
        self.dur = float(dur)
        self.loop = bool(loop)
        self.n = seconds(dur)
        self.m = 2 * self.n if loop else self.n

    def tile(self, x):
        x = np.asarray(x)
        if not self.loop:
            return fit(x, self.n)
        return np.concatenate([x[: self.n], x[: self.n]], axis=0)

    def noise(self, rng):
        return self.tile(rng.uniform(-1.0, 1.0, self.n))

    def pink(self, rng):
        """rjlib s_pinknoise over the domain (filtering the tiled white noise keeps it periodic)."""
        from scipy.signal import lfilter
        b = np.poly([0.984436, 0.833923, 0.0756836])
        a = np.poly([0.995728, 0.947906, 0.535675])
        return 0.17 * lfilter(b, a, self.noise(rng))

    def fold(self, y):
        y = np.asarray(y)
        n = self.n
        if not self.loop:
            return fit(y, n)
        out = fit(y[:n].copy(), n)
        k = n
        while k < len(y):
            seg = y[k:k + n]
            out[: len(seg)] += seg
            k += n
        return out

    def out(self, y):
        return y[self.n:self.m] if self.loop else y[: self.n]

    def cyc(self, f: float) -> float:
        """Loop mode: nearest frequency with a whole number of cycles per period (>= 1)."""
        if not self.loop:
            return f
        k = max(1, round(f * self.dur))
        return k / self.dur

    def lfo(self, f: float, phase: float = 0.0) -> np.ndarray:
        t = np.arange(self.m) / SR
        return np.cos(2 * np.pi * self.cyc(f) * t + phase)

    def delay(self, x, ms: float):
        d = int(round(ms * 1e-3 * SR))
        if d <= 0:
            return x
        if self.loop:
            return np.roll(x, d, axis=0)
        return np.concatenate([np.full((d,) + x.shape[1:], x[0]), x[:-d]], axis=0)

    def osc(self, f_m: np.ndarray) -> np.ndarray:
        """osc~ with a frequency signal. Loop mode: phase advance per period rounded to whole
        cycles (a < 1/dur Hz shift) so the oscillator itself is n-periodic."""
        f_m = np.asarray(f_m, float)
        if not self.loop:
            return np.cos(2 * np.pi * np.cumsum(f_m) / SR)
        f2 = f_m[self.n:self.m] / SR
        tot = float(np.sum(f2))
        corr = (round(tot) - tot) / self.n
        ph = np.cumsum(f2 + corr)
        return self.tile(np.cos(2 * np.pi * ph))


def _place(buf: np.ndarray, sig: np.ndarray, i: int, gain: float = 1.0, pan: float | None = None):
    """Add sig into buf at sample i (clipped to buf). pan None + stereo buf = centre."""
    if i >= len(buf) or i < 0:
        return
    s = np.asarray(sig)[: len(buf) - i] * gain
    k = len(s)
    if buf.ndim == 1:
        buf[i:i + k] += s if s.ndim == 1 else s.mean(axis=1)
        return
    if s.ndim == 1:
        gl, gr = pan_gains(0.0 if pan is None else pan)
        buf[i:i + k, 0] += s * gl
        buf[i:i + k, 1] += s * gr
    else:
        buf[i:i + k] += s


def _circ_air(x: np.ndarray, r: float, *, ref: float = 1.0, line: bool = False, circular: bool = True):
    """space.distance, but circular (period = len) so a loop stays seamless: 1/r or 1/sqrt(r)
    spreading and zero-phase JOS air absorption over (r - ref) m (research 01 §5, 10 §2.5)."""
    if not circular:
        return S.distance(x, r, ref=ref, line_source=line)
    x = np.asarray(x, float)
    g = math.sqrt(ref / r) if line else ref / r
    n = x.shape[0]
    fr = np.fft.rfftfreq(n, 1 / SR)
    mag = 10 ** (-S.air_db_per_m(fr) * max(r - ref, 0.0) / 20)
    X = np.fft.rfft(x, axis=0)
    X = X * (mag if x.ndim == 1 else mag[:, None])
    return np.fft.irfft(X, n, axis=0) * g


# ==========================================================================================
# Wind (Farnell Practical 18, main-wind4_mod.pd)
# ==========================================================================================

# read delay (ms) and pan (Pd 0..1) per source; pans: book/thesis values (research 02 §13, 05 §8.1)
WIND_SOURCES = {
    "static": dict(delay=0.0, pan=0.5),
    "howl1": dict(delay=100.0, pan=0.91),
    "howl2": dict(delay=300.0, pan=0.03),
    "wire1": dict(delay=500.0, pan=0.28),
    "wire2": dict(delay=900.0, pan=0.68),
    "leaves": dict(delay=3000.0, pan=0.5),
}


def windspeed(dom: Dom, rng: np.random.Generator, *, control: str = "walk", scale: float = 1.0,
              offset: float = 0.0) -> np.ndarray:
    """Farnell ``windspeed`` over the working domain (length dom.m), clipped 0..1.

    control="pd": base = 0.25 (1 + osc~ 0.1 Hz), exactly as the patch (a 10 s swell).
    control="walk": base = the patch's 0..0.5 range driven by Farnell's AM07 "slow random walk"
      shared control (research 09 D2.3; rjlib c_drunk reflected walk, A7).
      # UNSOURCED: walk step (1 s knots, +-0.12 per step, lop~ 0.1 Hz smoothing); neither source
      # gives the walk rate for wind.
    scale / offset: # UNSOURCED scene intensity (calm places < 1), applied to the base only.
    Gust and squall are the patch's, each from its own noise (Farnell: independence is required).
    """
    m = dom.m
    if control == "pd":
        base = 0.25 * (1.0 + dom.lfo(0.1))
    else:
        k = max(4, int(round(dom.dur)))
        steps = rng.uniform(-0.12, 0.12, k)
        walk = np.empty(k)
        v = rng.uniform(0.15, 0.35)
        for i in range(k):                               # c_drunk: reflect at the bounds
            v += steps[i]
            if v < 0.0:
                v = -v
            if v > 0.5:
                v = 1.0 - v
            walk[i] = v
        knots_t = np.arange(k + 1) * (dom.n / k)
        kn = np.append(walk, walk[0] if dom.loop else walk[-1])
        base_n = np.interp(np.arange(dom.n), knots_t, kn)
        base = lop(dom.tile(base_n), 0.1)
    base = np.clip(base * scale + offset, 0.0, 1.0)
    if control != "pd" and A.enabled("wind"):
        return _windspeed_measured(dom, rng, base, scale)
    ng, ns = dom.noise(rng), dom.noise(rng)
    gust = lop(lop(ng, 0.5), 0.5)
    gust = (gust - gust.mean()) * 50.0 * PDN * ((base + 0.5) ** 2 - 0.125)
    squall = lop(lop(ns, 3.0), 3.0)
    squall = (squall - squall.mean()) * 20.0 * PDN * ((np.maximum(base, 0.4) - 0.4) * 8.0) ** 2
    return np.clip(base + gust + squall, 0.0, 1.0)


def wind_kind(scale: float) -> str:
    """Which recorded gust statistics a scene follows. # UNSOURCED: the 1.05 split."""
    return "gale" if scale >= 1.05 else "calm"


def _windspeed_measured(dom: Dom, rng: np.random.Generator, base: np.ndarray, scale: float) -> np.ndarray:
    """Hybrid windspeed (SFX_HYBRID.md): Farnell's base + square-law gust + squall structure, but
    the gust *shape* is the modulation spectrum measured on a CC0 wind recording of the scene's
    kind (``analysis.gust_stats``: log-envelope spectrum, depth in dB), synthesised circularly.
    Mapping: the static layer's amplitude is (w + 0.2) (Farnell), so the gust is applied as a dB
    deviation of that amplitude - the bed's level then fluctuates with the recorded depth - and
    scaled by Farnell's square law ((b + 0.5)^2 - 0.125) normalised at b = 0.25 ("steady at low
    speed, strong gusts above half speed"). The squall stays the patch's.
    UNSOURCED: the dB mapping and the normalisation point."""
    stats = A.wind_stats(wind_kind(scale))
    name, st = stats[int(rng.integers(len(stats)))]
    _WIND_USED.append(name)
    g_db = A.gust_control(st, dom.dur, rng, loop=dom.loop)[: dom.n]
    g_db = dom.tile(fit(g_db, dom.n))
    sq_law = ((base + 0.5) ** 2 - 0.125) / 0.4375
    w = (base + 0.2) * 10 ** (g_db * sq_law / 20.0) - 0.2
    ns = dom.noise(rng)
    squall = lop(lop(ns, 3.0), 3.0)
    squall = (squall - squall.mean()) * 20.0 * PDN * ((np.maximum(base, 0.4) - 0.4) * 8.0) ** 2
    return np.clip(w + squall, 0.0, 1.0)


_WIND_USED: list = []
_WIND_TEX_GAIN = 1.0          # UNSOURCED: recorded grains vs the Farnell static rush


def _recorded_wind(dom: Dom, wd: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    """Hybrid wind texture: two independent level-matched grain clouds (L, R: uncorrelated,
    mono-safe) of one CC0 wind recording of the scene's kind, the grains chosen where the
    recording's level matches the windspeed's rank (``analysis.recorded_texture``; Farnell §9.7
    granular "textures such as wind"), circular for loops. Returns (dom.m, 2)."""
    w_out = dom.out(wd) if dom.loop else wd
    pick = float(rng.random())
    q = A.WIND_KINDS[kind]
    chans = [A.recorded_texture("wind", q, pick, w_out, rng, loop=dom.loop, grain_s=(0.12, 0.35),
                                overlap=3.0, used=_WIND_USED) for _ in (0, 1)]
    y = np.stack(chans, axis=1)
    y = dom.tile(y) if dom.loop else y
    # Farnell (research 02 §13): the rush brightens with speed. The recorded grains carry the
    # recordist's microphone brightness, so a speed-driven low-pass sets it procedurally
    # (circular: runs over the tiled domain). # UNSOURCED: 900 + 7000 w Hz.
    fc = np.clip(900.0 + 7000.0 * wd, 300.0, 16000.0)
    q = np.full(len(fc), 0.707)
    return np.stack([tv_lowpass(np.ascontiguousarray(y[:, c]), fc, q) for c in (0, 1)], axis=1)


def wind_scene(dur: float, rng: np.random.Generator, *, loop: bool = False,
               sources=("static", "howl1", "howl2", "wire1", "wire2", "leaves"),
               w: np.ndarray | None = None, control: str = "walk", scale: float = 1.0,
               offset: float = 0.0, wire_scale: float = 1.0, return_speed: bool = False,
               kind: str | None = None):
    """Farnell's wind scene (main-wind4_mod.pd): causally delayed emitters reading one windspeed.

    Returns stereo (n, 2). ``w`` may be a windspeed already made over ``Dom(dur, loop).m``;
    ``return_speed`` also returns the windspeed over the output period (for causally driven
    creaks, research 02 §13 / §4). ``wire_scale`` multiplies the wire centres (thinner or thicker
    obstruction, f ~ v / d; # UNSOURCED per-scene value).
    The static layer uses an independent noise per channel (a spread, mono-safe source).
    """
    dom = Dom(dur, loop)
    if w is None:
        w = windspeed(dom, rng, control=control, scale=scale, offset=offset)
    y = np.zeros((dom.m, 2))
    for name in sources:
        src = WIND_SOURCES[name]
        wd = dom.delay(w, src["delay"])
        if name == "static":
            for ch in (0, 1):
                s = bp(dom.noise(rng), 800.0, 1.0) * (wd + 0.2)
                s = rzero(s, np.clip(0.6 * (wd + 0.2), 0.0, 0.99)) * 0.2
                y[:, ch] += s
                static_rms = _rms(s)
            continue
        if name in ("howl1", "howl2"):
            lo, hi, lp, f0, df, fc = ((0.35, 0.6, 0.5, 30.0, 200.0, 400.0) if name == "howl1"
                                      else (0.25, 0.5, 0.1, 20.0, 100.0, 200.0))
            x = (np.clip(wd, lo, hi) - lo) * 2.0
            a = lop(np.cos(2 * np.pi * (x - 0.25)), lp)
            s = bp(dom.noise(rng), fc, 40.0) * a * dom.osc(f0 + df * a) * 2.0
        elif name == "wire1":
            s = vcf(dom.noise(rng), (600.0 + 400.0 * wd) * wire_scale, 60.0) * (wd + 0.12) ** 2 * 1.2
        elif name == "wire2":
            s = vcf(dom.noise(rng), (1000.0 + 1000.0 * wd) * wire_scale, 60.0) * wd ** 2 * 2.0
        else:  # leaves
            L = lop(wd + 0.3, 0.07)
            T = 1.0 - 0.4 * L
            s = (np.maximum(dom.noise(rng), T) - T) * T
            s = lop(hip(s, 200.0), 4000.0) * (L - 0.2)
        gl, gr = pan_gains(pd_pan(src["pan"]))
        y[:, 0] += s * gl
        y[:, 1] += s * gr
    if A.enabled("wind") and "static" in sources:
        # Hybrid: recorded wind grains under the same windspeed, level set to the static rush
        # (# UNSOURCED: 1:1 balance; the Farnell emitters keep their own balances).
        tex = _recorded_wind(dom, w, rng, kind or wind_kind(scale))
        tex = tex / (_rms(dom.out(tex)) + 1e-12) * static_rms * _WIND_TEX_GAIN
        y = y * np.array([1.0, 1.0]) + tex
    y = dom.out(y) * 0.45
    if return_speed:
        return y, dom.out(w)
    return y


# ==========================================================================================
# Creak (Farnell Practical 9: stick-slip -> wood formants -> panel)
# ==========================================================================================

DOOR_FORMANTS = ((62.5, 1.0), (125.0, 1.0), (250.0, 2.0), (395.0, 2.0), (560.0, 3.0), (790.0, 3.0))
PANEL_MS = (4.52, 5.06, 6.27, 8.0, 5.48, 7.14, 10.12, 16.0)


def stickslip_pulses(force: np.ndarray, rng: np.random.Generator, *, periodic: bool = False,
                     return_times: bool = False):
    """Farnell's stick-slip generator (research 02 §4, Fig 32.4) on a per-sample force in [0, 1].

    The force goes through a 100 ms lag (Pd ``line``; here a 100 ms moving average). While
    f > 0.3 a metronome ticks every (1-f)60 + 3 ms + U(0, 6(1-f)) ms; each tick is a slip whose
    pulse jumps to sqrt(sqrt(x) + 0.1) and falls to 0 over sqrt(x) ms (x = min(ms since the last
    slip, 100) / 100), then is squared. periodic=True wraps the pulses around the array end.
    """
    n = len(force)
    k = max(1, int(0.1 * SR))
    f = np.convolve(np.concatenate([force[-k:] if periodic else np.full(k, force[0]), force]),
                    np.ones(k) / k, mode="valid")[1:n + 1]
    out = np.zeros(n + 64)
    t = 0.0
    last = -1e9
    times = []
    fs_ms = SR / 1000.0
    while t < n:
        i = int(t)
        fi = f[i]
        if fi <= 0.3:
            t += 0.005 * SR                    # metro off: poll the force every 5 ms
            continue
        since = (t - last) / fs_ms if last > -1e8 else 100.0
        x = min(since, 100.0) / 100.0
        a = math.sqrt(math.sqrt(x) + 0.1)
        dn = max(2, int(math.sqrt(x) * fs_ms))
        pulse = (a * (1.0 - np.arange(dn) / dn)) ** 2
        out[i:i + dn] += pulse[: len(out) - i]
        times.append(i)
        last = t
        t += ((1.0 - fi) * 60.0 + 3.0 + rng.random() * 6.0 * (1.0 - fi)) * fs_ms
    if periodic:
        out[:64] += out[n:]
    out = out[:n]
    return (out, np.array(times)) if return_times else out


def creak(force: np.ndarray, rng: np.random.Generator, *, loop: bool = False, material: str = "wood",
          size: float = 1.0) -> np.ndarray:
    """Creak from a per-sample force curve (mono). material:
    "wood" - Farnell's door: formant bank + panel resonator (research 02 §4).
    "pine" - the same with every formant/panel frequency divided by ``size`` (Farnell: "for
             different sized doors ... recalculate the resonator and formant characteristics";
             # UNSOURCED: the size factor per object).
    "iron" - the stick-slip train through the measured sheet-metal box (computertower.sy)
             scaled by 1/size (# UNSOURCED mapping: Farnell says metal needs a different body).
    loop=True treats the force as one period of a loop (seamless).
    """
    n = len(force)
    pulses = stickslip_pulses(np.asarray(force, float), rng, periodic=loop)
    dom = Dom(n / SR, loop)
    x = dom.tile(pulses)
    if material == "iron":
        modes = M.resolve("sy_computertower", f_scale=1.0 / size)
        y = M.render_modes(x, modes)
        y = hip(y, 125.0)
        return dom.out(y) / (np.max(np.abs(y)) + 1e-12) * 0.5
    fs = 1.0 / size if material == "pine" else 1.0
    form = 0.2 * x
    for f, q in DOOR_FORMANTS:
        form = form + bp(x, f * fs, q)
    panel = np.zeros_like(form)
    for ms in PANEL_MS:
        panel += _dfb_k(form, max(1, int(round(ms * size * SR / 1000.0))) if material == "pine"
                        else max(1, int(round(ms * SR / 1000.0))), 0.05)
    y = hip(panel, 125.0 * fs) * 0.2
    return dom.out(y)


def sway_force(w: np.ndarray, rng: np.random.Generator, *, sway_hz: float = 0.35, lag_ms: float = 3000.0,
               gain: float = 1.6, loop: bool = False) -> np.ndarray:
    """Force on a tree / wagon from the (delayed, inertial) windspeed.

    Research 02 §13/§4: "drive the force from the wind speed (delayed like the leaves) to get
    causally linked creaking pines" - leaves use a 3 s read delay and lop~ 0.07.
    # UNSOURCED: the sway itself (narrow-band noise at ``sway_hz``, Q 3) and ``gain``; the sources
    # give no tree-sway model, only the lag.
    """
    dom = Dom(len(w) / SR, loop)
    wm = dom.tile(w)
    L = lop(dom.delay(wm, lag_ms), 0.07)
    sw = bp(dom.noise(rng), sway_hz, 3.0)
    sw = sw / (np.std(sw[dom.m // 4:]) + 1e-12)
    f = np.clip(L * gain * (0.55 + 0.35 * sw), 0.0, 1.0)
    return dom.out(f)


# ==========================================================================================
# Fire (Farnell Practical 11 / thesis 8.2)
# ==========================================================================================

def _env_db(x: np.ndarray, win: int = 1024, hop: int = 512) -> np.ndarray:
    """Pd env~ (sigenv): Hann-weighted mean square over ``win`` samples every ``hop``, in dB with
    100 = 1.0 RMS (rmstodb)."""
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    w = w / w.sum()
    x2 = np.square(x)
    nfr = (len(x) - win) // hop + 1
    pw = np.lib.stride_tricks.sliding_window_view(x2, win)[::hop][:nfr] @ w
    return np.maximum(0.0, 100.0 + 10.0 * np.log10(pw + 1e-20)), np.arange(nfr) * hop + win


def fire(dur: float, rng: np.random.Generator, *, loop: bool = False, units: int = 1,
         wood_crackle: float = 0.5, crackle_lp: float = 1.0, return_info: bool = False):
    """Farnell fire generator(s), stereo.

    units=1..3: generators with their own noises, panned slightly apart (a low fire).
    units=4: Farnell's big fire (each generator through bp 600 Q0.2 / bp 1200 Q0.6 / bp 2600 Q0.4
      / hip 1000, x0.2).
    wood_crackle: fraction of crackles centred 100-1000 Hz ("good for burning wood") instead of the
      patch's 1.5-16.5 kHz. # UNSOURCED mixture ratio.
    crackle_lp: cutoff (Hz) of the slow noise whose env~ window fires crackles (patch: 1 Hz;
      Farnell: density is set by this cutoff).
    """
    dom = Dom(dur, loop)
    n = dom.n
    y = np.zeros((dom.m, 2))
    big = units >= 4
    pans = [0.0] if units == 1 else list(np.linspace(-0.35, 0.35, units))
    n_crackles = 0
    for u in range(units):
        w_n = rng.uniform(-1, 1, n)                  # ONE shared noise per generator
        w = dom.tile(w_n)
        # hiss: noise x (lop~1 noise)^4 x10, hip~ 1000
        lp1 = lop(w, 1.0) * PDN
        sd = np.std(lp1[dom.m - n:]) + 1e-12
        # UNSOURCED: pre-gain 1/(3 sigma). The published pre/post gains (600/10 in the patch,
        # 2.0/2000 in the text) give levels 60 dB apart at any rate; this puts a 3-sigma excursion
        # of the modulator at the published x10 make-up, keeping Farnell's 4th-power expansion.
        hiss = hip(w * (lp1 / (3.0 * sd)) ** 4 * 10.0, 1000.0)
        # crackles: env~ of lop~1 noise inside the 50-51 dB window
        lpc = lop(w, crackle_lp) * PDN
        db, pos = _env_db(lpc[dom.m - n:] if dom.loop else lpc)
        # Pd env~ reads 54 dB for this signal at 44.1 kHz: the 50-51 dB window is in the patch's scale
        hits = pos[(db >= 50.0) & (db < 51.0)]
        exc = np.zeros(n + seconds(0.05))
        cr = np.zeros(n + seconds(0.05))
        for p_ in hits:
            T = max(2, int(rng.random() * 0.030 * SR))
            env = (1.0 - np.arange(T) / T) ** 2
            i0 = int(p_) % n
            seg = w_n[(i0 + np.arange(T)) % n] * env
            if rng.random() < wood_crackle:
                fc = 100.0 + rng.random() * 900.0
            else:
                fc = 1500.0 + 500.0 * rng.integers(0, 31)
            burst = bp(np.concatenate([seg, np.zeros(seconds(0.02))]), fc, 1.0)
            _place(cr, burst, i0)
        n_crackles += len(hits)
        crackle = dom.tile(dom.fold(cr))
        # lapping: bp~ 30 Q5 x100 -> hip~ 25 x2 -> clip +-0.9 -> x0.6
        lap = hip(hip(bp(w, 30.0, 5.0) * 100.0, 25.0), 25.0)
        lap = np.clip(lap, -0.9, 0.9) * 0.6
        s = crackle * 0.2 + hiss * 0.3 + lap * 0.6
        if big:
            s = [lambda v: bp(v, 600.0, 0.2), lambda v: bp(v, 1200.0, 0.6),
                 lambda v: bp(v, 2600.0, 0.4), lambda v: hip(v, 1000.0)][u % 4](s) * 0.2
        gl, gr = pan_gains(pans[u % len(pans)] if not big else [-0.4, 0.4, -0.15, 0.15][u % 4])
        y[:, 0] += s * gl
        y[:, 1] += s * gr
    out = dom.out(y)
    if return_info:
        return out, dict(crackles_per_s=n_crackles / dur / units)
    return out


# ==========================================================================================
# Running water (Reshetnikov water.pd; Farnell flowing-water voice)
# ==========================================================================================

@numba.njit(cache=True)
def _water_kernel(n, tail, n_inst, dyn, seed, loop):
    """Reshetnikov ``water.pd`` x n_inst over n samples (+ tail to fold). dyn = per-sample d."""
    np.random.seed(seed)
    L = n + tail
    out = np.zeros((L, 2))
    fs_ms = 48000.0 / 1000.0
    maxt = int(n / (10.0 * fs_ms)) + 4
    tt = np.zeros(maxt, np.int64)
    tgt = np.zeros(maxt)
    glide = np.zeros(maxt, np.int64)
    qs = np.zeros(maxt)
    amps = np.zeros(maxt)
    cf = np.zeros(L)
    qa = np.zeros(L)
    ex = np.zeros(L)
    for k in range(n_inst):
        # tick list over one period
        t = np.random.random() * 78.0 * fs_ms
        c = 0
        while t < n and c < maxt:
            i = int(t)
            d = dyn[i]
            r = np.random.random()
            tt[c] = i
            tgt[c] = 300.0 + 15700.0 * r ** 5
            glide[c] = int((100.0 + np.random.random() * 60.0) * fs_ms)
            qs[c] = 25.0 + 20.0 * (1.0 - r)
            amps[c] = (1.0 - r) ** (12.0 * d + 1.0) * (1.0 - d) ** 2
            c += 1
            t += (10.0 + np.random.random() * 68.0 * (1.0 - d * d)) * fs_ms
        if c == 0:
            continue
        # centre path (line~ glides) and Q per tick, over [0, L)
        prev = tgt[c - 1] if loop else tgt[0]
        q0 = qs[c - 1] if loop else qs[0]
        for i in range(0, tt[0]):
            cf[i] = prev
            qa[i] = q0
        for j in range(c):
            a0 = tt[j]
            end = tt[j + 1] if j + 1 < c else L
            fr = cf[a0 - 1] if a0 > 0 else prev
            g = glide[j]
            for i in range(a0, end):
                u = (i - a0) / g
                if u > 1.0:
                    u = 1.0
                cf[i] = fr + (tgt[j] - fr) * u
                qa[i] = qs[j]
        # excitation: noise x (amp (1 - t/1ms))^2
        for i in range(L):
            ex[i] = 0.0
        g1 = int(fs_ms)
        for j in range(c):
            a = amps[j]
            for q in range(g1):
                idx = tt[j] + q
                if idx < L:
                    v = a * (1.0 - q / g1)
                    ex[idx] += (np.random.random() * 2.0 - 1.0) * v * v
        # vcf~
        re_ = 0.0
        im = 0.0
        p = np.random.random() * 0.5
        gl = p + 0.25
        gr = 0.75 - p
        isr = 6.28318 / 48000.0
        for i in range(L):
            qq = qa[i]
            qinv = 1.0 / qq
            amp = 2.0 - 2.0 / (qq + 2.0)
            w = cf[i] * isr
            rr = 1.0 - w * qinv
            if rr < 0.0:
                rr = 0.0
            omr = 1.0 - rr
            cr = rr * math.cos(w)
            ci = rr * math.sin(w)
            re2 = re_
            re_ = amp * omr * ex[i] + cr * re2 - ci * im
            im = ci * re2 + cr * im
            out[i, 0] += re_ * gl
            out[i, 1] += re_ * gr
    return out


def reshetnikov_d(dom: Dom) -> np.ndarray:
    """The patch's automation d = 0.125 + 0.125 sin(2 pi 0.04 t) + 0.25 + 0.25 sin(2 pi 0.3 t)
    (loop mode: whole cycles per period)."""
    t = np.arange(dom.n) / SR
    return (0.125 + 0.125 * np.sin(2 * np.pi * dom.cyc(0.04) * t)
            + 0.25 + 0.25 * np.sin(2 * np.pi * dom.cyc(0.3) * t))


def stream(dur: float, rng: np.random.Generator, *, loop: bool = False, d=None, instances: int = 420,
           ice: bool = False, voices: int = 0) -> np.ndarray:
    """Running water, stereo: Reshetnikov's granular stream (+ optional Farnell water voices).

    d: dynamics 0.001-0.9 (scalar or per-sample over dur); None = the patch's automation.
    instances: 420 in the patch (lower = sparser, cheaper; a distant stream needs fewer).
    ice: water under an ice lid. # UNSOURCED: Farnell says under ice the stream "should also be
      low-passed" with no numbers; a 2nd-order low-pass at 900 Hz plus the lid's 1 kHz dip is used.
    voices: number of Farnell flowing-water voices added (thesis: "use 3-4 instances").
    """
    dom = Dom(dur, loop)
    n = dom.n
    dd = reshetnikov_d(dom) if d is None else np.clip(fit(_arr(d, n) if np.ndim(d) else np.full(n, d), n),
                                                       0.001, 0.9)
    tail = seconds(0.4)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    y = _water_kernel(n, tail, int(instances), np.ascontiguousarray(dd), seed, dom.loop)
    y = np.stack([dom.fold(y[:, 0]), dom.fold(y[:, 1])], axis=1) if dom.loop else y[:n]
    for v in range(voices):
        wv = farnell_water_voice(dur, rng, loop=loop, centre=800.0 * (0.8 + 0.1 * v))
        gl, gr = pan_gains(rng.uniform(-0.6, 0.6))
        y[:, 0] += wv * gl * 0.05
        y[:, 1] += wv * gr * 0.05
    y = np.stack([hip(dom.tile(y[:, c]), 5.0) for c in (0, 1)], axis=1)   # patch: hip~ 5
    if ice:
        y = np.stack([lowpass(lowpass(y[:, c], 900.0), 900.0) for c in (0, 1)], axis=1)
    return dom.out(y)


def _bilexp(rng: np.random.Generator, k: int) -> np.ndarray:
    """Farnell's bilinear exponential random (thesis T p.339, research 05 §8.3)."""
    r = rng.integers(0, 8192, k)
    v = np.exp((r % 4096) / 4096.0 * 9.0) / 23000.0
    return np.where(r > 4096, v, -v)


def farnell_water_voice(dur: float, rng: np.random.Generator, *, loop: bool = False, centre: float = 800.0,
                        mult: float = 1600.0, tick_ms: float = 6.0, slew_ms: float = 2.689) -> np.ndarray:
    """Farnell flowing-water voice (thesis T p.336-342): every 6 ms target = bilexp x1600 + 800 Hz,
    linear slew 2.689 ms, sine whose amplitude = lop~10(positive first difference squared)."""
    dom = Dom(dur, loop)
    n = dom.n
    step = seconds(tick_ms / 1000.0)
    k = n // step + 1
    tg = _bilexp(rng, k) * mult + centre
    sl = max(1, seconds(slew_ms / 1000.0))
    f = np.empty(k * step)
    prev = tg[-1] if loop else tg[0]
    ramp = np.minimum(np.arange(step) / sl, 1.0)
    for j in range(k):
        f[j * step:(j + 1) * step] = prev + (tg[j] - prev) * ramp
        prev = tg[j]
    f = f[:n]
    fm = dom.tile(f)
    d = np.diff(fm, prepend=fm[-1] if loop else fm[0])
    a = lop(np.clip(d, 0.0, 1.0) ** 2, 10.0)
    y = a * dom.osc(fm)
    return dom.out(y)


# ==========================================================================================
# Rain (Farnell Practical 15)
# ==========================================================================================

RAIN_GENERATORS = ((50.0, 12.0, 0.15), (60.0, 12.0, 0.2), (70.0, 12.0, 0.4))   # (rate ms, spread, base)


def rain(dur: float, rng: np.random.Generator, *, loop: bool = False, intensity: float = 1.0,
         n_generators: int = 12, on_water: float = 0.5, comfort: float = 0.5) -> np.ndarray:
    """Rain bed, stereo: Poisson parabolic clicks (10-15 generators, the thesis's three settings
    repeated with rates scaled by 1/intensity), pink comfort noise + a 300 Hz band, drops on water
    as Moss bubbles (alpha 2.9). # UNSOURCED: the relative layer levels ``on_water``/``comfort``.
    Farnell's ``drops`` spike layer is omitted (its figure-only parameters are ambiguous)."""
    dom = Dom(dur, loop)
    n = dom.n
    y = np.zeros((n + seconds(0.05), 2))
    for g in range(n_generators):
        rate, spread, base = RAIN_GENERATORS[g % 3]
        scale = rate / 1000.0 / max(intensity, 1e-3)
        pan = rng.uniform(-0.8, 0.8)
        gl, gr = pan_gains(pan)
        t = rng.random() * scale
        while t < dur:
            wms = base + spread * rng.random()
            m = max(2, int(wms * 1e-3 * SR))
            xx = np.linspace(0, 1, m)
            amp = (math.sqrt(wms) + 0.1) * abs(rng.normal())       # Gaussian intensity (Farnell)
            pulse = (1.0 - 4.0 * (xx - 0.5) ** 2) * amp
            i = int(t * SR)
            y[i:i + m, 0] += pulse * gl
            y[i:i + m, 1] += pulse * gr
            t += -math.log(1.0 - rng.random()) * scale
    clicks = np.stack([dom.fold(y[:, 0]), dom.fold(y[:, 1])], axis=1) * 0.1
    clicks = np.stack([hip(dom.tile(clicks[:, c]), 900.0) for c in (0, 1)], axis=1)
    out = _norm(clicks)
    if on_water > 0:
        bw = []
        for c in (0, 1):
            b = P.bubble_field(dur + 0.3, rng, rate=400.0 * intensity, alpha=2.9, r_range=(0.25e-3, 1.5e-3))
            bw.append(dom.tile(dom.fold(b)))
        bw = np.stack(bw, axis=1)
        out = out + _norm(bw) * on_water
    if comfort > 0:
        cn = []
        for c in (0, 1):
            cn.append(dom.pink(rng) + 0.5 * bp(dom.noise(rng), 300.0, 1.0))
        cn = np.stack(cn, axis=1)
        out = out + _norm(cn) * comfort
    return dom.out(out)


# ==========================================================================================
# Thunder (Farnell main-thunder4.pd)
# ==========================================================================================

def thunder(rng: np.random.Generator, *, dur: float = 24.0, distance: float = 0.0,
            return_info: bool = False) -> np.ndarray:
    """Farnell's 8-layer thunder, stereo. distance in [0, 1] is the start of the patch's travelling
    distance ramp d (0 -> 1 over 12 s); distance > 0.6 drops the close strike clatter (a far storm)."""
    n = seconds(dur)
    y = np.zeros((n, 2))
    # 1. strike pattern
    events = []
    x, t, c = 1.0, 0.0, 0
    while x < 100.0:
        t += x
        events.append((t, c % 4, (100.0 - x) / 100.0))
        x += rng.integers(0, 100) / 10.0
        c += 1
    # 2. strike sounds (0.3)
    strike = np.zeros(n)
    if distance <= 0.6:
        for t_ms, _, v in events:
            T = max(2, seconds(200.0 * (1.4 - v) ** 5 / 1000.0))
            e = 1.0 - np.arange(T) / T
            s = rng.uniform(-1, 1, T) * e
            s = bp(s, 100.0 + 1200.0 * v, 3.0) + bp(s, (100.0 + 1200.0 * v) / 2, 3.0)
            _place(strike, s, seconds(t_ms / 1000.0))
    # 3. rumble
    nA, nB = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    tt = np.arange(n) / SR
    density = np.clip(2000.0 * (1.0 - tt / 9.0), 0.0, None)
    A = lop(lop(nA, density), density)
    B = lop(lop(nB, density), density)
    fph = np.maximum(B, 0.0) * 3000.0 + 1.0
    ph = np.cumsum(fph) / SR % 1.0
    wrap = np.flatnonzero(np.diff(ph) < 0) + 1
    held = np.zeros(n)
    last, v0 = 0, 0.0
    for wv in wrap:
        held[last:wv] = v0
        v0 = A[wv]
        last = wv
    held[last:] = v0
    walk = _rpole_k(held, 0.99) * 0.04
    ax = np.abs(walk)
    # parabolic shaper: half-circle excursions per phasor cycle, height |x| (research 02 §12.2)
    rumble = hip(ax * 0.5 * (1.0 - 4.0 * (ph - 0.5) ** 2), 100.0)
    renv = np.where(tt < 0.05, tt / 0.05, np.clip(1.0 - (tt - 0.05) / 8.0, 0.0, 1.0))
    rumble = rumble * renv * 0.3
    # 4. afterimage (after 200 ms, x0.9)
    after = np.zeros(n)
    k0 = seconds(0.2)
    m = n - k0
    ta = np.arange(m) / SR
    cut = 30.0 * np.clip(1.0 - ta / 0.6, 0.0, 1.0) + 3.0
    g = lop(rng.uniform(-1, 1, m), cut) * 80.0 * PDN * rng.uniform(-1, 1, m)
    g = bp(np.clip(g, -1.0, 1.0), 333.0, 4.0) * np.clip(1.0 - ta / 6.0, 0.0, 1.0) * 0.9
    after[k0:] = g
    # 5. deep noise
    dn = lop(lop(rng.uniform(-1, 1, n), 80.0), 80.0)
    dn = lop(lop(np.clip(hip(dn, 15.0) * 12.0 * PDN, -1, 1), 80.0), 80.0)
    de = np.where(tt < 1.0, 0.0, np.where(tt < 3.0, (tt - 1.0) / 2.0, np.clip(1.0 - (tt - 3.0) / 20.0, 0, 1)))
    deep = dn * de ** 2 * 0.5
    # 6. distance filter on rumble + deep
    d = np.clip(distance + (1.0 - distance) * tt / 12.0, 0.0, 1.0)
    cutd = (1.0 - d) * 6000.0 + 50.0
    rd = lop(lop(lop(rumble + deep, cutd), cutd), cutd)
    dl = ((2.0 + 3.0 * d) * 1e-3 * SR).astype(int)
    idx = np.clip(np.arange(n) - dl, 0, n - 1)
    rd = rd + 0.5 * rd[idx]
    # 7. echo box (24 taps up to 12 s)
    for _ in range(24):
        D = rng.uniform(0.0, 12.0)
        gg = (1.0 - D / 12.0) * rng.uniform(0.5, 1.0)
        tap = bp(rd, 400.0 + 400.0 * gg, 2.0) * gg
        _place(y, tap, seconds(D), 1.0, pd_pan(gg))
    # 8. mix
    direct = (strike * 0.3 + after) if distance <= 0.6 else after * (1.0 - distance)
    y[:, 0] += direct + rd
    y[:, 1] += direct + rd
    fade = seconds(1.0)
    y[-fade:] *= pow_decay(fade, 2.0)[:, None]
    if return_info:
        return y, dict(strikes=len(events), strike_span_ms=events[-1][0])
    return y


# ==========================================================================================
# Surf (Moss bubbles + Farnell)
# ==========================================================================================

SIGMA_W, RHO_W, MU_W = 0.0728, 998.0, 1.0e-3   # water: surface tension N/m, density, dyn. viscosity


def tube_bubble(r: float, rng: np.random.Generator, n_modes: int = 8) -> np.ndarray:
    """Moss §3.2 non-spherical bubble: zonal modes f_n^2 = (n-1)(n+1)(n+2) sigma / (4 pi^2 rho r^3),
    radiating at 2 f_n, weight (n-1)(n+2)(4n-1)/(2n+1) * w_n^2 / sqrt((4w_n^2-w_b^2)^2+(4 b_n w_n)^2),
    viscous decay b_n = (n+2)(2n+1) mu / (rho r^2); the 8 strongest kept (Table I: 8-10 suffice),
    plus the radial Minnaert mode.
    # UNSOURCED: equal shape amplitudes c_n, the radial:shape balance 1:0.5, and the decay floor at
    # the radial mode's beta0 (Moss prints no radiation damping for shape modes)."""
    fb = P.MINNAERT_FR / r
    b0 = math.pi * fb * (P.DELTA_RAD + P.DELTA_TH)
    wb = 2 * math.pi * fb
    ns = np.arange(2, 400)
    fn = np.sqrt((ns - 1) * (ns + 1) * (ns + 2) * SIGMA_W / (RHO_W * r ** 3)) / (2 * math.pi)
    wn = 2 * math.pi * fn
    bn = (ns + 2) * (2 * ns + 1) * MU_W / (RHO_W * r * r)
    W = (ns - 1) * (ns + 2) * (4 * ns - 1) / (2 * ns + 1) * wn ** 2 / np.sqrt(
        (4 * wn ** 2 - wb ** 2) ** 2 + (4 * bn * wn) ** 2)
    ok = 2 * fn < 0.45 * SR
    order = np.argsort(np.where(ok, W, 0))[::-1][:n_modes]
    L = seconds(min(6.91 / b0, 1.5))
    t = np.arange(L) / SR
    y = np.sin(wb * (t + P.MOSS_XI * b0 * t * t / 2)) * np.exp(-b0 * t)
    Wk = W[order] / (W[order].max() + 1e-30)
    shape = np.zeros(L)
    for k, j in enumerate(order):
        shape += Wk[k] * np.cos(2 * wn[j] * t + rng.uniform(0, 2 * np.pi)) * np.exp(-max(bn[j], b0) * t)
    shape /= (np.max(np.abs(shape)) + 1e-12)
    return (y + 0.5 * shape) * r * 1e3


def _wave_env(n: int, rng: np.random.Generator, period=(8.0, 13.0), loop: bool = True, single: bool = False):
    """Wave crest envelopes and their start times over one period.
    # UNSOURCED: swell period 8-13 s, crest rise 1-2 s (square law), wash decay 3-6 s (quartic);
    # no source gives a surf timing model."""
    env = np.zeros(n + seconds(8.0))
    starts = []
    t = 0.3 if single else rng.uniform(0.0, period[0])
    dur = n / SR
    while t < dur - (1.0 if loop else 3.0):
        tr, td = rng.uniform(1.0, 2.0), rng.uniform(3.0, 6.0)
        a = rng.uniform(0.6, 1.0)
        up = (np.arange(seconds(tr)) / seconds(tr)) ** 2
        down = pow_decay(seconds(td), 4.0)
        e = np.concatenate([up, down]) * a
        i = seconds(t)
        env[i:i + len(e)] = np.maximum(env[i:i + len(e)], e[: len(env) - i])
        starts.append((i, seconds(tr), a))
        if single:
            break
        t += rng.uniform(*period)
    if loop:
        env[: len(env) - n] = np.maximum(env[: len(env) - n], env[n:])
    return env[:n], starts


_SEA_USED: list = []
_SEA_MIX = (0.7, 0.4, 1.5)      # UNSOURCED: bubbles, noise wash, recorded wash (RMS-normalised)


def surf(dur: float, rng: np.random.Generator, *, loop: bool = False, distance: float = 120.0,
         period=(8.0, 13.0), bubble_rate: float = 3000.0, single: bool = False,
         state: str | None = None) -> np.ndarray:
    """Breaking surf on a shore, stereo, as a line source ``distance`` m away.

    Per wave: bubble formation rate ~ crest^2 (Moss Gamma = u^2 kappa), radius law alpha drawn in
    the breaking-wave range 1.5-3.3 per wave, r 0.3-20 mm; at each break a few large tube bubbles
    (Moss §3.2); the broadband wash is noise shaped by the crest (Farnell: a stormy sea is close to
    white noise), brighter at the break. # UNSOURCED: bubble_rate, the wash low-pass law
    (400 + 5000 c Hz) and the wash:bubble balance.

    Hybrid (SFX_HYBRID.md, ``state`` "calm" or "storm"): the crest timing comes from wave-cycle
    statistics measured on CC0 surf recordings of that state (``analysis.wave_stats``: crest
    intervals, rise and decay times, crest-level spread, crest-trough depth), and the wash is a
    level-matched grain cloud of such a recording under that procedural timing (Farnell §9.7:
    granular "textures such as water"; the grains of a crest come from the recording's crests),
    two independent clouds for L/R; Moss bubbles stay on top. # UNSOURCED: layer balance.
    """
    dom = Dom(dur, loop)
    n = dom.n
    hybrid = A.enabled("sea")
    kind = state or "calm"
    name = None
    if hybrid and not single:
        stats = A.sea_stats_screened(kind)
        name, st = stats[int(rng.integers(len(stats)))]
        _SEA_USED.append(name)
        env, starts = A.wave_control(st, n, rng, loop=loop)
    else:
        env, starts = _wave_env(n, rng, period, loop, single)
    pick = float(rng.random())
    ys = []
    for ch in (0, 1):
        rate = bubble_rate * np.maximum(env - 0.15, 0.0) ** 2
        b = np.zeros(n + seconds(10.5))
        for (i0, tr, a) in starts:
            seg_len = min(n - i0, seconds(9.0)) if not loop else seconds(9.0)
            rr = np.zeros(seg_len)
            k = np.arange(seg_len)
            idx = (i0 + k) % n if loop else np.clip(i0 + k, 0, n - 1)
            rr[:] = rate[idx]
            alpha = rng.uniform(1.5, 3.3)
            bb = P.bubble_field(seg_len / SR, rng, rate=rr, alpha=alpha, r_range=(0.3e-3, 20e-3))
            _place(b, bb, i0)
            for _ in range(int(rng.integers(2, 5))):              # large tube bubbles at the break
                tb = tube_bubble(rng.uniform(5e-3, 40e-3), rng)
                eps = P._powerlaw(rng, 1, 0.01, 0.1, -2.0)[0]          # Moss eps in [0.01, 0.1]
                _place(b, tb * eps, i0 + tr + int(rng.uniform(0, 0.8) * SR))
        bub = dom.tile(dom.fold(b))
        wash_c = dom.tile(env)
        wash = tv_lowpass(dom.noise(rng) * wash_c ** 2, 400.0 + 5000.0 * wash_c, np.full(dom.m, 0.707))
        bub_n = bub / (_rms(bub[dom.m - n:]) + 1e-12)
        wash_n = wash / (_rms(wash[dom.m - n:]) + 1e-12)
        if hybrid:
            rec = A.recorded_texture("sea", A.SEA_KINDS[kind], pick, env, rng, loop=loop,
                                     grain_s=(0.15, 0.4), overlap=3.0, used=_SEA_USED,
                                     file=name if not single else None)
            rec = rec * (0.45 + 0.55 * env / (env.max() + 1e-12))   # crest emphasis (UNSOURCED)
            rec = dom.tile(rec) if loop else rec
            ys.append(_SEA_MIX[0] * bub_n + _SEA_MIX[1] * wash_n + _SEA_MIX[2] * rec / (_rms(rec) + 1e-12))
        else:
            ys.append(bub_n + 1.2 * wash_n)
    y = dom.out(np.stack(ys, axis=1))
    return _circ_air(y, distance, line=True, circular=loop)


# ==========================================================================================
# Drips, bubbles, cave
# ==========================================================================================

PRIMES = (29, 37, 47, 67, 89, 113, 157, 197)


def bubble_burst(dur: float, rng: np.random.Generator, *, size: float = 0.5) -> np.ndarray:
    """Farnell's surfacing-bubble pattern (thesis T p.331-334): a 15 ms counter mod 200 fires on
    the primes 29..197 with 50 % random culling; each fire = particles.surfacing_bubble, the size
    slightly varied (round-robin voices never cut each other off: bubbles are summed)."""
    n = seconds(dur)
    y = np.zeros(n + seconds(0.3))
    step = seconds(0.015)
    for k in range(int(dur / 0.015)):
        if (k % 200) in PRIMES and rng.random() < 0.5:
            b = P.surfacing_bubble(rng, dim=np.clip(size + rng.uniform(-0.15, 0.15), 0.05, 1.0))
            _place(y, b, k * step)
    return y[:n]


def cave_drips(dur: float, rng: np.random.Generator, *, loop: bool = False, sources: int = 5,
               reverb: str | None = "cave", wet: float = 0.6) -> np.ndarray:
    """Drips in a cave, stereo. Each source is a quantised flow (a relaxation oscillator: surface
    tension vs weight, thesis T p.80-81), so it drips nearly periodically with jitter; each drop is
    ``particles.drip`` (STK waterDrop + Farnell impact click). Space: zita_rev1 "cave".
    # UNSOURCED: drip periods 1.2-7 s, jitter +-12 %, and the wet level."""
    dom = Dom(dur, loop)
    n = dom.n
    y = np.zeros((n + seconds(0.5), 2))
    for s in range(sources):
        per = rng.uniform(1.2, 7.0)
        f0 = rng.uniform(450.0, 1200.0)
        pan = rng.uniform(-0.8, 0.8)
        g = rng.uniform(0.3, 1.0)
        t = rng.uniform(0, per)
        while t < dur:
            d = P.drip(rng, f0=f0 * rng.uniform(0.97, 1.03), dur=0.35)
            _place(y, d, seconds(t), g * rng.uniform(0.8, 1.0), pan)
            t += per * rng.uniform(0.88, 1.12)
    dry = np.stack([dom.fold(y[:, 0]), dom.fold(y[:, 1])], axis=1)
    if reverb is None:
        return dry
    wetsig = S.zita_rev1(dom.tile(dry), **S.ZITA_PRESETS[reverb])
    return dom.out(dom.tile(dry) + wet * wetsig / (_rms(wetsig) + 1e-12) * _rms(dry))


# ==========================================================================================
# Chains, keys
# ==========================================================================================

def tinkle(rng: np.random.Generator, *, f_scale: float = 1.0, gain: float = 1.0) -> np.ndarray:
    """One shell-casing / link contact (shellcasings.pd): noise x e^4 (decay 0-39 ms) -> 3 bp~ Q800
    at f0 = 4500 + U1000, f1 = f0 + 700 + U100, f2 = f1 + 300 + U100 -> sum x100 -> clip +-0.9."""
    d = max(8, int(rng.integers(0, 40) * 1e-3 * SR))
    e = (1.0 - np.arange(d) / d) ** 4
    x = np.concatenate([rng.uniform(-1, 1, d) * e, np.zeros(seconds(0.35))])
    f0 = (4500.0 + rng.random() * 1000.0) * f_scale
    f1 = f0 + (700.0 + rng.random() * 100.0) * f_scale
    f2 = f1 + (300.0 + rng.random() * 100.0) * f_scale
    s = (bp(x, f0, 800.0) + bp(x, f1, 800.0) + bp(x, f2, 800.0)) * 100.0
    return np.clip(s, -0.9, 0.9) * gain


def clunk(rng: np.random.Generator, *, f_scale: float = 1.0, gain: float = 1.0) -> np.ndarray:
    """Reload click-factory cluster 2, the end-stop clunk (research 03 §5.3): noise hip~ 300 ->
    345 Hz Q22 (70 ms), 256 Hz Q3 (45 ms), 11023 Hz Q46 (11 ms); each band x (1 in 1 ms, 0 over d)^2
    (filters before the envelope, Farnell mclick), sum x0.33."""
    n = seconds(0.12)
    x = hip(rng.uniform(-1, 1, n), 300.0) * gain
    y = np.zeros(n)
    for f, q, dms in ((345.0, 22.0, 70.0), (256.0, 3.0, 45.0), (11023.0, 46.0, 11.0)):
        a = seconds(0.001)
        dd = seconds(dms / 1000.0)
        env = np.concatenate([np.linspace(0, 1, a), 1.0 - np.arange(dd) / dd, np.zeros(max(0, n - a - dd))])[:n]
        y += bp(x, min(f * f_scale, 0.45 * SR), q) * env ** 2
    return y * 0.33


def chain_shift(dur: float, rng: np.random.Generator, *, pushes: int = 2, f_scale: float = 0.8,
                contacts_per_s: float = 40.0, settle: bool = True, clunk_p: float = 0.25) -> np.ndarray:
    """A chain moved by a body (mono). Momentum model of Farnell's rolling can (research 02 §3,
    05 §8.7): 500 ms half-cosine pushes into a low-pass give the speed; link contacts are fired
    at a rate ~ speed, amplitude ~ sqrt(speed) (root law: the chain saturates); each contact is a
    shell-casing tinkle, some also a click-factory clunk (link on link). The chain comes to rest
    with the shell-casing bounce pattern (75, +170-344, +75-149 ms; gains 0.416, 0.257, 0.05).
    # UNSOURCED: contacts_per_s, f_scale (heavier links than a casing: lower), clunk_p, lop 1.5 Hz."""
    n = seconds(dur)
    push = np.zeros(n)
    hc = np.sin(np.pi * np.arange(seconds(0.5)) / seconds(0.5))
    for k in range(pushes):
        at = rng.uniform(0.05, max(0.1, dur * 0.55))
        _place(push, hc * rng.uniform(0.5, 1.0), seconds(at))
    speed = lop(lop(push, 1.5), 1.5)
    speed = speed / (speed.max() + 1e-12)
    y = np.zeros(n + seconds(0.5))
    t = 0.0
    last = 0
    while t < dur:
        s = speed[min(int(t * SR), n - 1)]
        if s > 0.05:
            a = math.sqrt(s) * rng.uniform(0.5, 1.0)
            _place(y, tinkle(rng, f_scale=f_scale, gain=a), seconds(t))
            if rng.random() < clunk_p * s:
                _place(y, clunk(rng, f_scale=f_scale, gain=a), seconds(t))
            last = seconds(t)
            t += -math.log(1.0 - rng.random()) / (contacts_per_s * s)
        else:
            t += 0.01
    if settle:
        t0 = last + seconds(0.075)
        for g, gap in ((0.416, 0.0), (0.257, rng.uniform(0.170, 0.344)), (0.05, rng.uniform(0.075, 0.149))):
            t0 += seconds(gap)
            _place(y, tinkle(rng, f_scale=f_scale, gain=g), t0)
    return y[:n]


def keys_jingle(rng: np.random.Generator, *, hits: int = 3, sep_ms: float = 10.0, d_ms: float = 40.0,
                q: float = 217.0) -> np.ndarray:
    """Farnell ``pd tamb`` jingles (research 09 C8.2) for a ring of keys: noise -> bp~ Q217 at
    1700 (decay d), 3149 (2d), 2330 (4d), 1511 (8d) Hz -> ring-mod by 6300 Hz -> x0.6; each hit is
    a pair 3*sep ms apart. # UNSOURCED: several hits 60-250 ms apart for a hanging bunch."""
    n = seconds(0.6 + 0.25 * hits)
    y = np.zeros(n + seconds(0.6))
    t = 0
    for _ in range(hits):
        for off in (0, 3 * sep_ms):
            m = seconds(8 * d_ms / 1000.0) + seconds(0.05)
            x = rng.uniform(-1, 1, m) * rng.uniform(0.5, 1.0)
            s = np.zeros(m)
            for f, k in ((1700.0, 1), (3149.0, 2), (2330.0, 4), (1511.0, 8)):
                dd = seconds(k * d_ms / 1000.0)
                env = np.clip(1.0 - np.arange(m) / dd, 0.0, 1.0)
                s += bp(x, f, q) * env
            s = s * np.cos(2 * np.pi * 6300.0 * np.arange(m) / SR) * 0.6
            _place(y, s, t + seconds(off / 1000.0))
        t += seconds(rng.uniform(0.06, 0.25))
    return y[:n]


# ==========================================================================================
# Void pressure (sonic-pi dark_ambience), hum (Farnell), outdoor echoes (Farnell siren)
# ==========================================================================================

def _ringz(x, f, decay):
    """SuperCollider Ringz: R = exp(ln 0.001 / (decay SR)), y0 = x + 2R cos(w) y1 - R^2 y2,
    out = 0.5 (y0 - y2)."""
    R = math.exp(math.log(0.001) / (decay * SR))
    w = 2 * math.pi * f / SR
    from scipy.signal import lfilter
    return lfilter([0.5, 0.0, -0.5], [1.0, -2 * R * math.cos(w), R * R], x)


def dark_ambience(dur: float, rng: np.random.Generator, *, loop: bool = False, note: float = 52.0,
                  ring: float = 0.2, reverb_time: float = 100.0) -> np.ndarray:
    """sonic-pi dark_ambience (atmos.clj defaults: note 52, detunes +12/+24, ring 0.2, room 70,
    reverb_time 100, cutoff 110, res 0.7 -> rq 0.3, pink noise x0.005), stereo.
    GVerb is not ported (space.py): zita_rev1 with t60 = reverb_time stands in, mixed as GVerb's
    defaults dry 1 / tail 0.5. # UNSOURCED: that substitution."""
    dom = Dom(dur, loop)
    src = dom.pink(rng) * 0.005
    f1 = 440.0 * 2 ** ((note - 69) / 12)
    s1 = _ringz(src, f1, ring)
    s2 = _ringz(src, f1 * 2.0, ring)
    s3 = _ringz(src, f1 * 4.0, ring)
    mix = s1 + s1 + s2 + s3
    wet = S.zita_rev1(mix, rdel=60.0, f1=200.0, f2=6000.0, t60dc=reverb_time, t60m=reverb_time)
    y = np.stack([mix, mix], axis=1) + 0.5 * wet
    y = np.tanh(y)
    fc = 440.0 * 2 ** ((110 - 69) / 12)
    y = np.stack([lowpass(y[:, c], fc, q=1 / 0.3) for c in (0, 1)], axis=1)
    return dom.out(y)


def hum(dur: float, rng: np.random.Generator, *, loop: bool = False) -> np.ndarray:
    """Farnell Practical 16 hum: two phasors at 99.8 and 100.2 Hz (0.4 Hz beat), summed, -1, hard
    clip; mono."""
    dom = Dom(dur, loop)
    t = np.arange(dom.m) / SR
    ph = (dom.cyc(99.8) * t) % 1.0 + (dom.cyc(100.2) * t) % 1.0 - 1.0
    y = np.clip(ph, -1.0, 1.0)
    y = dom.out(y)
    # Band-limit the naive phasors (they alias, and their hard edges are re-coded unevenly by
    # Vorbis at the loop wrap, which made a seam): zero-phase, circular, 4th-order 4 kHz
    # magnitude, so the loop stays exactly periodic. A mains hum heard through a room has no
    # content above that anyway. # UNSOURCED: the 4 kHz corner.
    if loop:
        Y = np.fft.rfft(y)
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        y = np.fft.irfft(Y / np.sqrt(1.0 + (fr / 4000.0) ** 8), len(y))
    else:
        y = lowpass(lowpass(y, 4000.0), 4000.0)
    return y


def outdoor_echoes(x: np.ndarray, *, loop: bool = False) -> np.ndarray:
    """Farnell's building echoes (siren practical, thesis T p.274): delays 165, 121, 33 ms summed and
    recirculated x0.1, echo tap x0.05, direct x0.2 (per channel)."""
    dom = Dom(len(x) / SR, loop)
    out = []
    xs = x if x.ndim == 2 else x[:, None]
    for c in range(xs.shape[1]):
        v = dom.tile(xs[:, c])
        d = [seconds(ms / 1000.0) for ms in (165.0, 121.0, 33.0)]
        m = len(v)
        buf = np.zeros(m)
        fb = np.zeros(m)
        for i in range(0, m, min(d)):            # block recursion: blocks shorter than the shortest delay
            j = min(m, i + min(d))
            s = np.zeros(j - i)
            for dd in d:
                a, b = i - dd, j - dd
                if b > 0:
                    s[max(0, -a):] += (v[max(a, 0):b] + fb[max(a, 0):b])
            buf[i:j] = s
            fb[i:j] = 0.1 * s
        out.append(dom.out(v * 0.2 + buf * 0.05))
    y = np.stack(out, axis=1)
    return y if x.ndim == 2 else y[:, 0]


# ==========================================================================================
# Beds
# ==========================================================================================

def _stereo(x):
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def _mix(layers):
    """layers: list of (stereo signal, dB). Each layer is normalised to unit RMS, then set in dB
    (# UNSOURCED: scene mix levels are design choices; the emitters inside each layer keep the
    sources' own balances)."""
    out = None
    for sig, db in layers:
        s = _stereo(sig)
        s = s / _rms(s) * _db(db)
        out = s if out is None else out + s
    return out


def bed_caravan(rng, dur=BED_SECONDS):
    """A frozen mountain road at night where a prison caravan has halted.
    Wind (static rush over rock, wagon rigging/chains as the two wires, the slatted wagons as the
    200 Hz cavity howl), wagon timber creaks and faint iron creaks both driven by the same gusts
    (causal link, research 02 §13 / §4)."""
    w_dom = Dom(dur, True)
    w = windspeed(w_dom, rng, control="walk", scale=1.0)
    wind, wout = wind_scene(dur, rng, loop=True, sources=("static", "howl2", "wire1", "wire2"), w=w,
                            return_speed=True)
    f1 = sway_force(wout, rng, sway_hz=0.45, lag_ms=300.0, gain=1.5, loop=True)
    f2 = sway_force(wout, rng, sway_hz=0.3, lag_ms=600.0, gain=1.4, loop=True)
    wood = creak(f1, rng, loop=True, material="wood")
    iron = creak(f2, rng, loop=True, material="iron", size=0.8)
    wood = _circ_air(np.stack([wood * 0.8, wood * 0.55], axis=1), 6.0)
    iron = _circ_air(np.stack([iron * 0.45, iron * 0.8], axis=1), 9.0)
    return _mix([(wind, 0.0), (wood, -13.0), (iron, -21.0)])


def bed_wilds(rng, dur=BED_SECONDS):
    """Black mountain slopes, dead pines, bone fields, ash-streaked snow, a frozen stream.
    Wind (rush over rock; the bare pine branches as the Q60 wire whistles - research 02 §13 note:
    "bare branches: use the wire whistles plus stick-slip creaks"; rock gaps as the 400 Hz howl;
    no leaves: the pines are dead), two pines creaking from the 3 s-lagged wind, and the frozen
    stream under ice 40 m away."""
    w = windspeed(Dom(dur, True), rng, control="walk", scale=1.1)
    wind, wout = wind_scene(dur, rng, loop=True, sources=("static", "howl1", "wire1", "wire2"), w=w,
                            return_speed=True, wire_scale=1.15)
    pines = []
    for k, (hz, size, pan) in enumerate(((0.28, 1.6, -0.6), (0.37, 1.3, 0.55))):
        f = sway_force(wout, rng, sway_hz=hz, lag_ms=3000.0 + 700.0 * k, gain=1.5, loop=True)
        c = creak(f, rng, loop=True, material="pine", size=size)
        gl, gr = pan_gains(pan)
        pines.append(_circ_air(np.stack([c * gl, c * gr], axis=1), 15.0 + 10.0 * k))
    water = stream(dur, rng, loop=True, d=0.28, instances=140, ice=True)
    water = _circ_air(water, 40.0)
    return _mix([(wind, 0.0), (pines[0], -15.0), (pines[1], -17.0), (water, -14.0)])


def bed_shore(rng, dur=BED_SECONDS):
    """Crimson coral labyrinth over black mud beside a lightless black sea.
    Distant surf line (line source, 150 m), black water lapping the mud near by (Reshetnikov grains
    whose dynamics follow each wave 2.5 s later - the wash reaching the shore), sparse mud bubbles,
    and wind through the coral: rush + the two cavity howls (coral holes are cavities)."""
    dom = Dom(dur, True)
    n = dom.n
    sea = surf(dur, rng, loop=True, distance=150.0)
    env, _ = _wave_env(n, np.random.default_rng(int(rng.integers(1 << 30))), (8.0, 13.0), True)
    lap_d = np.clip(0.02 + 0.22 * np.roll(env, seconds(2.5)), 0.001, 0.9)
    lap = stream(dur, rng, loop=True, d=lap_d, instances=90)
    lap = lap * np.roll(env, seconds(2.5))[:, None] ** 0.5
    mud = np.zeros(n + seconds(0.5))
    t = rng.uniform(0, 4)
    while t < dur:
        _place(mud, bubble_burst(rng.uniform(1.0, 2.5), rng, size=rng.uniform(0.55, 0.85)), seconds(t))
        t += rng.uniform(5.0, 14.0)
    mud = dom.fold(mud)
    mud = _circ_air(np.stack([mud * 0.8, mud * 0.6], axis=1), 5.0)
    wind = wind_scene(dur, rng, loop=True, sources=("static", "howl1", "howl2"), control="walk", scale=0.85)
    extra = []
    if A.enabled("sea"):     # AUDIO_CHARTER §8.7: room tone; the Dark Sea's Eb1 tide swell
        extra = [(room_tone(dur, rng, loop=True), -22.0), (tide_swell(dur, rng, loop=True), -10.0)]
    return _mix([(sea, 0.0), (lap, -8.0), (mud, -20.0), (wind, -7.0)] + extra)


def bed_awakening(rng, dur=BED_SECONDS):
    """A lightless void where a soul takes shape: only sonic-pi's dark_ambience pressure (tuned
    noise resonances in a near-infinite reverb, research 11 D3)."""
    return dark_ambience(dur, rng, loop=True)


def bed_city(rng, dur=BED_SECONDS):
    """A colossal ruined grey-stone city on a crater rim, a castle on a hill, shanties below.
    Wind over stone with the two doorway howls (broken arches and gates), distant shanty cook-fires
    (Farnell fire, 2 generators, 70 m), a far wooden creak of the shanties, all through Farnell's
    building echoes (stone reverberation)."""
    wind = wind_scene(dur, rng, loop=True, sources=("static", "howl1", "howl2", "wire1"), control="walk",
                      scale=0.9)
    fires = _circ_air(fire(dur, rng, loop=True, units=2, wood_crackle=0.7), 70.0)
    wv = windspeed(Dom(dur, True), rng, control="walk", scale=0.9)
    f = sway_force(Dom(dur, True).out(wv), rng, sway_hz=0.4, lag_ms=1000.0, gain=1.4, loop=True)
    shanty = creak(f, rng, loop=True, material="wood")
    shanty = _circ_air(np.stack([shanty * 0.5, shanty * 0.85], axis=1), 35.0)
    y = _mix([(wind, 0.0), (fires, -12.0), (shanty, -22.0)])
    return outdoor_echoes(y, loop=True) / 0.2


TIDE_HZ = 38.89   # Eb1: AUDIO_CHARTER §8.7 "the Tide sub swell tuned to Eb1 (38.9 Hz)", the b2 of D


def tide_swell(dur: float, rng: np.random.Generator, *, loop: bool = True) -> np.ndarray:
    """The Dark Sea's Tide sub swell (AUDIO_CHARTER §8.7, §3.4 M4 "the same b2 as a filtered sub
    swell"): a narrow noise band at Eb1 (Pd bp~ Q 12) with its sine core and the 2nd partial at
    -12 dB, swelling like M4 (rise 5, crest 6, fall 4 beats) once per 20-40 s wave group,
    circular. Mono in both channels (sub). # UNSOURCED: Q, group period, the sine:noise mix."""
    dom = Dom(dur, loop)
    nz = bp(dom.noise(rng), TIDE_HZ, 12.0)
    nz = nz / (_rms(nz) + 1e-12)
    nz2 = bp(dom.noise(rng), 2 * TIDE_HZ, 12.0)
    nz2 = nz2 / (_rms(nz2) + 1e-12) * 0.25
    core = dom.osc(np.full(dom.m, TIDE_HZ)) * 0.7
    sw = (0.5 - 0.5 * dom.lfo(1.0 / rng.uniform(20.0, 40.0), rng.uniform(0, 2 * np.pi))) ** 2
    y = dom.out((nz + nz2 + core) * sw)
    return np.stack([y, y], axis=1)


def room_tone(dur: float, rng: np.random.Generator, *, loop: bool = True) -> np.ndarray:
    """The place's air (AUDIO_CHARTER §8.7 bed = far field + mid field + room tone): rjlib pink
    noise through two lop~ 300, independent per channel (spread, mono-safe), with a 0.05 Hz
    breathing of +-15 %. # UNSOURCED: corner and drift."""
    dom = Dom(dur, loop)
    ch = []
    for k in (0, 1):
        x = lop(lop(dom.pink(rng), 300.0), 300.0) * (1.0 + 0.15 * dom.lfo(0.05, k * 1.7))
        ch.append(dom.out(x))
    return np.stack(ch, axis=1)


def bed_generic(rng, text: str, dur=BED_SECONDS):
    """Keyword recipe for acts/library beds without a hand-built scene. Each keyword group adds the
    sourced emitter that the description calls for. # UNSOURCED: the keyword mapping and levels."""
    t = text.lower()
    has = lambda *ws: any(re.search(r"\b" + w, t) for w in ws)
    layers = []
    indoor = has("interior", "hall", "room", "pod", "institution", "lounge", "parlour", "gymnasium")
    if has("sea", "ocean", "surf", "shore", "tide", "harbour", "cliff", "whirlpool", "wave", "shipwreck"):
        layers.append((surf(dur, rng, loop=True, distance=90.0 if has("cliff", "crash") else 140.0,
                            state="storm" if has("crash", "storm", "heavy swell", "rough", "cliff") else "calm"), 0.0))
    if has("river", "stream", "waterfall", "cascade", "rapids", "flooded", "fast cold"):
        layers.append((stream(dur, rng, loop=True, d=0.45 if has("fast", "waterfall") else 0.3,
                              instances=200, ice=has("frozen", "ice")), -3.0))
    if has("cave", "cavern", "tunnel", "catacomb", "mine", "crypt", "underground", "flooded", "temple interior",
           "dripping"):
        layers.append((cave_drips(dur, rng, loop=True, sources=6), -6.0))
    if has("storm", "rain", "swamp"):
        layers.append((rain(dur, rng, loop=True, intensity=1.0 if has("storm", "rain") else 0.3), -4.0))
    if has("fire", "torch", "hearth", "smoke", "camp", "lava", "fissure", "burn", "battle", "war"):
        layers.append((_circ_air(fire(dur, rng, loop=True, units=4 if has("war", "battle", "burn") else 2),
                                 40.0), -10.0))
    if has("pod", "institution", "hum", "spelltech", "modern", "floodlight", "city street", "debrief",
           "lounge", "towers and lit"):
        h = hum(dur, rng, loop=True)
        layers.append((_circ_air(np.stack([h, h], axis=1), 8.0), -14.0))
    if has("void", "sky thick", "night sky", "sun", "pool", "still") and not layers:
        layers.append((dark_ambience(dur, rng, loop=True), 0.0))
    if not indoor or not layers:
        howls = ("static", "howl1", "howl2") if has("ruin", "gate", "arch", "temple", "wall", "tower", "castle",
                                                     "coral", "stone", "bone") else ("static",)
        wires = ("wire1", "wire2") if has("mountain", "peak", "snow", "pass", "glacier", "pine", "forest", "tree",
                                          "cage", "web", "chain") else ()
        leaves = ("leaves",) if has("jungle", "grass", "heather", "forest", "tree", "garden", "meadow") and \
            not has("dead", "leafless") else ()
        wind = wind_scene(dur, rng, loop=True, sources=howls + wires + leaves, control="walk",
                          scale=1.1 if has("storm", "wind", "peak", "cliff") else 0.8)
        layers.append((wind, -2.0 if layers else 0.0))
    if has("cage", "chain") and not indoor:
        wv = windspeed(Dom(dur, True), rng)
        f = sway_force(Dom(dur, True).out(wv), rng, sway_hz=0.4, lag_ms=500.0, gain=1.5, loop=True)
        c = creak(f, rng, loop=True, material="iron")
        layers.append((_circ_air(np.stack([c * 0.7, c * 0.7], axis=1), 20.0), -18.0))
    if A.enabled("sea"):
        # AUDIO_CHARTER §8.7: the room tone under every bed; the Dark Sea carries the Eb1 tide
        layers.append((room_tone(dur, rng, loop=True), -22.0))
        if has("black water", "dark sea", "black sea") and has("sea", "shore", "water", "tide", "wave"):
            layers.append((tide_swell(dur, rng, loop=True), -10.0))
    y = _mix(layers)
    if has("stone", "ruin", "castle", "city", "town", "wall", "temple", "quarry", "arena", "colosseum"):
        y = outdoor_echoes(y, loop=True) / 0.2
    return y


ACT_BEDS = {"caravan": bed_caravan, "wilds": bed_wilds, "shore": bed_shore, "awakening": bed_awakening,
            "city": bed_city}


BED_MODIFIERS = ("night", "dawn", "high_tide", "dark")        # AUDIO_CHARTER §8.7 variants


def render_bed(row: dict, rng: np.random.Generator, dur: float = BED_SECONDS) -> np.ndarray:
    """A bed; with the charter on, proposed variant ids ``<bed id>.<modifier>`` (modifiers in
    ``BED_MODIFIERS``) render the same place under ``bed_modifier``."""
    from . import charter
    parts = row["id"].split(".")
    mods = [p for p in parts[2:] if p in BED_MODIFIERS] if charter.enabled() else []
    base_id = ".".join(p for p in parts if p not in mods)
    name = base_id.split(".")[-1]
    if base_id.startswith("amb.act.") and name in ACT_BEDS:
        y = ACT_BEDS[name](rng, dur)
    else:
        y = bed_generic(rng, _place_text(row), dur)
    for m in mods:
        y = bed_modifier(y, m, rng, dur)
    return y


def bed_modifier(y: np.ndarray, mod: str, rng: np.random.Generator, dur: float) -> np.ndarray:
    """AUDIO_CHARTER §8.7, all circular (the loop stays seamless):
    * night: low-pass the bed at 4 kHz (circular FFT, 4th-order magnitude), the mid field swaps
      species of sound - wind in (a quiet Farnell static + wire layer under the bed, -10 dB);
    * dawn: a single long swell of the far field over the loop (+-3 dB raised cosine);
    * high_tide: deep water (the bed's own low band doubled below 250 Hz) and the Tide sub swell
      at Eb1 (38.89 Hz, the b2 of D), -8 dB;
    * dark: the "bed -4 dB, details +2 dB" rule is a mix level - mastering to -26 LUFS would undo
      it, so the variant carries the offset as a gain the game applies (see the report); the
      content itself gets duller (low-pass 6 kHz) so a dark place also sounds enclosed.
    The -2 dB of night and the -4 dB of dark are mixer gains (direction.toml), not baked in."""
    n = y.shape[0]
    Y = np.fft.rfft(y, axis=0)
    fr = np.fft.rfftfreq(n, 1 / SR)
    lp = lambda fc: 1.0 / np.sqrt(1.0 + (fr / fc) ** 8)
    if mod == "night":
        y = np.fft.irfft(Y * lp(4000.0)[:, None], n, axis=0)
        wind = wind_scene(dur, rng, loop=True, sources=("static", "wire1"), control="walk", scale=0.8)
        y = y + wind / (_rms(wind) + 1e-12) * _rms(y) * _db(-10.0)
    elif mod == "dawn":
        t = np.arange(n) / n
        g = _db(-3.0) + (_db(3.0) - _db(-3.0)) * (0.5 - 0.5 * np.cos(2 * np.pi * (t - rng.uniform(0, 1))))
        y = y * g[:, None]
    elif mod == "high_tide":
        low = np.fft.irfft(Y * (fr < 250.0)[:, None], n, axis=0)
        tide = tide_swell(dur, rng, loop=True)
        y = y + low + tide / (_rms(tide) + 1e-12) * _rms(y) * _db(-8.0)
    elif mod == "dark":
        y = np.fft.irfft(Y * lp(6000.0)[:, None], n, axis=0)
    return y


# ==========================================================================================
# Detail one-shots (6 per place)
# ==========================================================================================

def _shape(y: np.ndarray, dur: float, fade_in: float = 0.02, fade_out: float = 0.4) -> np.ndarray:
    """Fit to dur with a short power-law fade-in and a quartic fade-out (no clicks)."""
    y = fit(np.asarray(y, float), seconds(dur))
    a = seconds(fade_in)
    if a > 1:
        y[:a] *= (np.arange(a) / a) ** 2
    b = min(seconds(fade_out), len(y) // 2)
    if b > 1:
        y[-b:] *= pow_decay(b, 4.0)
    return y


def _mono(y):
    return y.mean(axis=1) if np.ndim(y) == 2 else y


def _in_cave(y: np.ndarray, wet: float = 0.5, preset: str = "cave") -> np.ndarray:
    tail = seconds(3.0)
    x = np.concatenate([y, np.zeros(tail)])
    r = _mono(S.zita_rev1(x, **S.ZITA_PRESETS[preset]))
    return x + wet * r / (_rms(r) + 1e-12) * _rms(x)


def d_fire_flare(rng):
    """Wood spits then an upsurge (Farnell exercise: smoulder -> crackle + lap -> flare): one fire
    generator whose crackle density rises (crackle_lp 1 -> 4 Hz) under a rise-and-fall envelope.
    # UNSOURCED: the flare envelope (0.6 s rise, quartic 3 s fall)."""
    dur = rng.uniform(4.0, 6.5)
    y = _mono(fire(dur, rng, units=1, crackle_lp=3.0, wood_crackle=0.6))
    n = len(y)
    env = np.concatenate([(np.arange(seconds(0.6)) / seconds(0.6)) ** 2, np.ones(seconds(0.8))])
    env = fit(np.concatenate([env, pow_decay(max(1, n - len(env)), 4.0)]), n)
    return _shape(y * (0.35 + 0.65 * env), dur)


def d_fire_pop(rng):
    """Popping / spitting wood: a cluster of Farnell crackles (square-law noise bursts 0-30 ms through
    bp~ Q1, 100-1000 Hz and 1.5-16.5 kHz) at a low fire's crackle density. # UNSOURCED: cluster
    timing (3-9 crackles within 1.5 s)."""
    dur = rng.uniform(2.0, 3.5)
    n = seconds(dur)
    y = np.zeros(n + seconds(0.1))
    base = _mono(fire(dur, rng, units=1)) * 0.25
    t = rng.uniform(0.2, 0.5)
    for _ in range(int(rng.integers(3, 10))):
        T = max(2, int(rng.random() * 0.030 * SR))
        env = (1.0 - np.arange(T) / T) ** 2
        fc = 100.0 + rng.random() * 900.0 if rng.random() < 0.5 else 1500.0 + 500.0 * rng.integers(0, 31)
        s = bp(np.concatenate([rng.uniform(-1, 1, T) * env, np.zeros(seconds(0.02))]), fc, 1.0)
        _place(y, s / (np.max(np.abs(s)) + 1e-12), seconds(t), rng.uniform(0.3, 1.0))
        t += rng.exponential(0.12)
    return _shape(y[:n] + base, dur)


def d_log_settle(rng):
    """Fuel settling (Farnell's 'clattering'): a burnt log drops - measured wood stick modes
    (modalpaper stick.sy, lower by size) struck with shrinking bounce intervals (monotonicity,
    research 01 §2.6), then a crackle upsurge. # UNSOURCED: log size factor 0.35-0.5, bounce law."""
    dur = rng.uniform(3.5, 5.5)
    n = seconds(dur)
    y = np.zeros(n + seconds(1.0))
    fs = rng.uniform(0.35, 0.5)
    t, gap, v = 0.15, rng.uniform(0.09, 0.14), 1.0
    for _ in range(int(rng.integers(3, 6))):
        s = M.strike("sy_stick", f_scale=fs, contact_ms=4.0, rng=rng, velocity=v, max_dur=0.8)
        _place(y, s, seconds(t))
        t += gap
        gap *= rng.uniform(0.55, 0.75)
        v *= rng.uniform(0.35, 0.6)
    y = y / (np.max(np.abs(y)) + 1e-12)
    fl = _mono(fire(dur, rng, units=1, crackle_lp=3.0))
    env = np.clip((np.arange(n) / SR - 0.3) / 0.8, 0, 1) ** 2 * fit(pow_decay(n, 2.0), n)
    return _shape(y[:n] + 0.6 * fl / (_rms(fl) + 1e-12) * _rms(y[:n]) * env, dur)


def d_chain_shift(rng, heavy=True):
    dur = rng.uniform(2.5, 5.0)
    return _shape(chain_shift(dur, rng, pushes=int(rng.integers(1, 3)), f_scale=0.7 if heavy else 1.0), dur)


def d_chain_drag(rng):
    """A longer drag: many pushes, faster contacts."""
    dur = rng.uniform(4.0, 7.0)
    return _shape(chain_shift(dur, rng, pushes=int(rng.integers(3, 5)), contacts_per_s=70.0, f_scale=0.65), dur)


def d_keys(rng):
    """Keys on the warden's chain: tamb jingles + a light chain movement."""
    k = keys_jingle(rng, hits=int(rng.integers(2, 5)))
    c = chain_shift(len(k) / SR + 0.8, rng, pushes=1, f_scale=0.9, contacts_per_s=25.0)
    y = fit(np.concatenate([np.zeros(seconds(0.1)), k]), len(c)) + 0.5 * c / (np.max(np.abs(c)) + 1e-12) \
        * np.max(np.abs(k))
    return _shape(y, max(2.0, len(y) / SR))


def _gesture_force(dur: float, rng, peak=(0.6, 0.95)) -> np.ndarray:
    """A load that rises and falls (a body shifting weight). # UNSOURCED gesture shape."""
    n = seconds(dur)
    t = np.arange(n) / n
    c = rng.uniform(0.35, 0.6)
    w = rng.uniform(0.18, 0.3)
    return np.clip(rng.uniform(*peak) * np.exp(-0.5 * ((t - c) / w) ** 2), 0, 1)


def d_iron_creak(rng):
    dur = rng.uniform(2.5, 4.5)
    y = creak(_gesture_force(dur, rng), rng, material="iron", size=rng.uniform(0.7, 1.0))
    return _shape(y, dur)


def d_wood_creak(rng, size=1.0):
    dur = rng.uniform(2.5, 4.5)
    mat = "pine" if size != 1.0 else "wood"
    y = creak(_gesture_force(dur, rng), rng, material=mat, size=size)
    return _shape(y, dur)


def d_breath(rng):
    """Sleepers breathing in the cold, a few metres off: slow whispered-vowel breaths
    (voice.breath: Cook whispered formants + Farnell articulation). # UNSOURCED: sleep breathing
    timing (exhale 1.4-2 s, inhale 1.1-1.5 s, pause 0.8-1.6 s)."""
    from . import voice as V
    y = []
    total = 0.0
    while total < rng.uniform(5.0, 7.5):
        ins = V.breath(rng.uniform(1.1, 1.5), "in", rng, formant_scale=rng.uniform(0.95, 1.1))
        out = V.breath(rng.uniform(1.4, 2.0), "out", rng, formant_scale=rng.uniform(0.95, 1.1))
        gap = np.zeros(seconds(rng.uniform(0.8, 1.6)))
        seg = np.concatenate([ins * 0.7, np.zeros(seconds(0.15)), out, gap])
        y.append(seg)
        total += len(seg) / SR
    y = np.concatenate(y)
    y = _mono(S.distance(y, rng.uniform(3.0, 6.0)))
    return _shape(y, min(10.0, len(y) / SR))


def d_drip(rng, cave=True):
    """A few drops from one or two quantised sources (particles.drip) into water, in a cave."""
    dur = rng.uniform(2.5, 5.0)
    n = seconds(dur)
    y = np.zeros(n + seconds(0.4))
    for s in range(int(rng.integers(1, 3))):
        per = rng.uniform(0.5, 1.6)
        f0 = rng.uniform(450.0, 1200.0)
        t = rng.uniform(0.05, 0.4)
        g = rng.uniform(0.5, 1.0)
        while t < dur - 1.0:
            _place(y, P.drip(rng, f0=f0 * rng.uniform(0.97, 1.03)), seconds(t), g)
            t += per * rng.uniform(0.9, 1.1)
    y = y[:n]
    if cave:
        y = _in_cave(y, 0.7)
    return _shape(y, min(10.0, len(y) / SR))


def d_gurgle(rng, ice=False):
    """A surge in running water: Reshetnikov grains (all 420 instances) at a fixed dynamics under a
    swell envelope. (In the patch a larger d makes the stream denser but quieter: (1-d)^2.)
    # UNSOURCED: d range 0.2-0.4 and the sin^0.5 swell."""
    dur = rng.uniform(3.0, 5.5)
    n = seconds(dur)
    tt = np.arange(n) / n
    y = _mono(stream(dur, rng, d=rng.uniform(0.2, 0.4), instances=420, ice=ice, voices=0 if ice else 2))
    return _shape(y * np.sin(np.pi * tt) ** 0.5, dur, fade_in=0.3, fade_out=0.8)


def d_bubbles(rng, under_ice=False):
    """A burst of surfacing bubbles. The prime selector fires at most 8 times per 3 s counter
    cycle and culls half, so a short burst can come out empty: the burst is redrawn (up to 6
    times) until it holds at least 3 bubbles, and runs 3-4.5 s. # UNSOURCED: the minimum."""
    dur = rng.uniform(3.0, 4.5)
    for _ in range(6):
        y = bubble_burst(dur, rng, size=rng.uniform(0.4, 0.8))
        env = np.abs(y) > 0.05 * (np.max(np.abs(y)) + 1e-12)
        onsets = int(np.sum(np.diff(env.astype(int)) > 0))
        if np.max(np.abs(y)) > 0 and onsets >= 3:
            break
    if under_ice:
        y = lowpass(lowpass(y, 900.0), 900.0)
    return _shape(y, dur)


def d_pebbles(rng, cave=False):
    """A trickle of small stones (STK BigRocks/LittleRocks PhISEM, research 07/11) - a short energy
    injection that dies at the preset's system decay."""
    dur = rng.uniform(2.0, 3.5)
    n = seconds(dur)
    e = np.zeros(n)
    k = seconds(rng.uniform(0.3, 0.8))
    e[:k] = 0.02 * pow_decay(k, 2.0)
    name = "BigRocks" if rng.random() < 0.6 else "LittleRocks"
    y = P.shaker(name, dur, rng, energy_curve=e)
    if cave:
        y = _in_cave(y, 0.6)
    return _shape(y, min(10.0, len(y) / SR))


def d_sea_surge(rng, tunnel=True):
    """One wave's wash heard down a tunnel: a single surf wave (Moss bubbles + wash) far off, in a
    stone space."""
    dur = rng.uniform(5.0, 7.0)
    y = _mono(surf(dur, rng, single=True, distance=60.0))
    if tunnel:
        y = lowpass(y, 1500.0)
        y = _in_cave(y, 0.8, "stone_hall")
    return _shape(y, min(10.0, len(y) / SR), fade_in=0.5, fade_out=1.0)


def d_wave_slap(rng):
    dur = rng.uniform(3.0, 5.0)
    y = _mono(surf(dur, rng, single=True, distance=15.0))
    return _shape(y, dur, fade_in=0.3, fade_out=0.8)


def d_thunder_far(rng):
    y = _mono(thunder(rng, dur=10.0, distance=0.8))
    return _shape(y, 10.0, fade_in=0.2, fade_out=2.0)


def d_gust_howl(rng):
    """One gust through a gap: the wind scene with a single rise/fall of windspeed through the howl
    windows. # UNSOURCED: gust gesture."""
    dur = rng.uniform(4.0, 7.0)
    dom = Dom(dur, False)
    t = np.arange(dom.m) / dom.m
    w = np.clip(0.2 + 0.5 * np.sin(np.pi * t) ** 2 + 0.05 * lop(dom.noise(rng), 2.0) * 30, 0, 1)
    y = _mono(wind_scene(dur, rng, sources=("static", "howl1", "howl2"), w=w))
    return _shape(y, dur, fade_in=0.4, fade_out=1.0)


DETAIL_KINDS = {
    "fire_flare": d_fire_flare, "fire_pop": d_fire_pop, "log_settle": d_log_settle,
    "chain_shift": d_chain_shift, "chain_light": lambda r: d_chain_shift(r, heavy=False),
    "chain_drag": d_chain_drag, "keys": d_keys, "iron_creak": d_iron_creak, "wood_creak": d_wood_creak,
    "beam_creak": lambda r: d_wood_creak(r, size=1.4), "breath": d_breath,
    "drip_cave": d_drip, "drip": lambda r: d_drip(r, cave=False), "gurgle": d_gurgle,
    "gurgle_ice": lambda r: d_gurgle(r, ice=True), "bubbles": d_bubbles,
    "bubbles_ice": lambda r: d_bubbles(r, under_ice=True), "pebbles": d_pebbles,
    "pebbles_cave": lambda r: d_pebbles(r, cave=True), "sea_surge": d_sea_surge, "wave_slap": d_wave_slap,
    "thunder_far": d_thunder_far, "gust_howl": d_gust_howl,
}

# Hand-read palettes for places whose description names the emitters precisely.
PLACE_PALETTES = {
    # "A low fire ... It wears a warden's chain across its shoulders ... The keys to the wagon hang from it."
    "campfire": ("fire_flare", "log_settle", "fire_pop", "chain_shift", "keys", "fire_pop"),
    # "A row of chained sleepers, breathing in the cold."
    "cage_line": ("chain_shift", "breath", "iron_creak", "chain_light", "wood_creak", "breath"),
    # "Black ice over running water, loud enough to cover a footstep."
    "frozen_stream": ("gurgle_ice", "bubbles_ice", "gurgle_ice", "drip", "gurgle", "bubbles_ice"),
    # "Water-smoothed stone a hundred metres down, black puddles and salt air: at night the sea fills all of this."
    "flooded_tunnels": ("drip_cave", "sea_surge", "pebbles_cave", "drip_cave", "gurgle", "sea_surge"),
}

_KEYWORDS = (
    (("fire", "flame", "hearth", "torch", "ember", "smoke", "fire pit", "firepit"), ("fire_pop", "fire_flare",
                                                                                      "log_settle")),
    (("chain", "irons", "shackle", "collar", "chained", "manacle"), ("chain_shift", "chain_drag")),
    (("key",), ("keys",)),
    (("cage", "iron", "gate", "bars", "ring"), ("iron_creak",)),
    (("wagon", "cart", "timber", "plank", "boat", "ship", "hut", "shant", "lodge", "door", "beam", "wood"),
     ("wood_creak", "beam_creak")),
    (("sleep", "breath", "sleeper"), ("breath",)),
    (("cave", "tunnel", "cavern", "catacomb", "mine", "cellar", "crypt", "underground", "hollow", "shaft"),
     ("drip_cave", "pebbles_cave")),
    (("frozen", "ice"), ("gurgle_ice", "bubbles_ice")),
    (("stream", "river", "water", "pool", "puddle", "spring", "falls", "waterfall", "lake", "flood"),
     ("gurgle", "drip", "bubbles")),
    (("sea", "shore", "tide", "surf", "wave", "coast", "beach", "strait", "ocean"), ("wave_slap", "sea_surge")),
    (("mud", "swamp", "bog", "marsh"), ("bubbles",)),
    (("storm", "thunder", "lightning"), ("thunder_far",)),
    (("stone", "rock", "rubble", "quarry", "bone", "rockfall", "ruin", "cliff"), ("pebbles",)),
    (("wind", "peak", "summit", "ridge", "pass", "mountain", "arch", "ledge", "chasm", "crevice"), ("gust_howl",)),
)


def _place_text(row: dict) -> str:
    p = row.get("prompt", "")
    return p.split("Place:")[-1] if "Place:" in p else p


def place_palette(row: dict) -> tuple[str, ...]:
    name = row["id"].split(".")[-1]
    if name in PLACE_PALETTES:
        return PLACE_PALETTES[name]
    t = _place_text(row).lower()
    kinds: list[str] = []
    for words, ks in _KEYWORDS:
        if any(re.search(r"\b" + w, t) for w in words):
            kinds.extend(k for k in ks if k not in kinds)
    # Widen to 6 different kinds (QA 2026-10-02: 480 places had only 3 kinds, so their six details
    # repeated): first the siblings of the kinds the text named, then a pool chosen by the kind of
    # place (indoor / cave / water / outdoor). Never a creature voice or footsteps (docs/07 §5.2).
    # UNSOURCED: the sibling and pool tables.
    indoor = any(re.search(r"\b" + w, t) for w in ("hall\\b", "inside", "interior", "chamber", "house", "hut\\b", "inn\\b",
                                                   "tavern", "temple", "lodge", "tower", "castle", "library",
                                                   "pod", "lounge", "shop", "barracks", "cell", "wagon"))
    cave = any(k.endswith("_cave") for k in kinds)
    water = any(k in kinds for k in ("gurgle", "gurgle_ice", "drip", "bubbles", "wave_slap", "sea_surge"))
    for k in list(kinds):
        for s in SIBLINGS.get(k, ()):
            if s not in kinds:
                kinds.append(s)
    pool = (POOL_CAVE if cave else POOL_INDOOR if indoor else POOL_WATER if water else POOL_OUTDOOR)
    for k in pool + POOL_OUTDOOR:
        if len(set(kinds)) >= N_DETAILS:
            break
        if k not in kinds:
            kinds.append(k)
    out = [kinds[i % len(kinds)] for i in range(N_DETAILS)]
    return tuple(out)


SIBLINGS = {
    "chain_shift": ("chain_light", "chain_drag"), "chain_light": ("chain_shift",), "chain_drag": ("chain_shift",),
    "wood_creak": ("beam_creak",), "beam_creak": ("wood_creak",), "drip": ("drip_cave",),
    "drip_cave": ("drip", "pebbles_cave"), "pebbles": ("pebbles_cave",), "pebbles_cave": ("pebbles",),
    "gurgle": ("gurgle_ice", "bubbles"), "gurgle_ice": ("gurgle", "bubbles_ice"), "bubbles": ("bubbles_ice",),
    "bubbles_ice": ("bubbles",), "fire_pop": ("fire_flare", "log_settle"), "fire_flare": ("fire_pop",),
    "log_settle": ("fire_pop",), "wave_slap": ("sea_surge",), "sea_surge": ("wave_slap",),
    "iron_creak": ("chain_light",), "keys": ("chain_light",),
}
POOL_INDOOR = ("beam_creak", "wood_creak", "iron_creak", "drip_cave", "pebbles_cave", "log_settle")
POOL_CAVE = ("drip_cave", "pebbles_cave", "gurgle", "bubbles", "drip", "gust_howl")
POOL_WATER = ("drip", "gurgle", "bubbles", "wave_slap", "pebbles", "gust_howl")
POOL_OUTDOOR = ("gust_howl", "pebbles", "wood_creak", "drip", "beam_creak", "iron_creak", "pebbles_cave")


def render_detail(row: dict, index: int, rng: np.random.Generator) -> np.ndarray:
    kind = place_palette(row)[index % N_DETAILS]
    y = DETAIL_KINDS[kind](rng)
    y = np.asarray(y, float)
    if len(y) < seconds(2.0):
        y = fit(y, seconds(2.0))
    return y[: seconds(10.0)]


# ==========================================================================================
# Family entry point
# ==========================================================================================

def render(row: dict, variation: int, rng: np.random.Generator) -> np.ndarray:
    """Beds (amb.act.* / amb.library.*): stereo (n, 2), 120 s seamless loop, variation ignored.
    Places (amb.place.*): ``variation`` = detail index 0..5 (detail_1..6), mono 2-10 s.
    Not mastered (the render driver masters: bed -26 LUFS with loop=True, detail -30 LUFS)."""
    rid = row["id"]
    if rid.startswith("amb.place."):
        return render_detail(row, int(variation), rng)
    return render_bed(row, rng)


def master_loop(y: np.ndarray, cls: str = "bed") -> np.ndarray:
    """core.master for a loop without a seam: the 20 Hz high-pass inside master() is warmed up on
    the loop's own last second, which is then dropped."""
    k = seconds(1.0)
    z = master(np.concatenate([y[-k:], y], axis=0), cls, loop=True)
    return z[k:]


# ==========================================================================================
# Self-test
# ==========================================================================================

SELFTEST_BEDS = ("amb.act.shore", "amb.act.wilds", "amb.act.caravan", "amb.act.awakening", "amb.act.city")
SELFTEST_PLACES = ("amb.place.campfire", "amb.place.cage_line", "amb.place.frozen_stream",
                   "amb.place.flooded_tunnels")


def _rows(ids) -> dict:
    import tomllib
    with open(MANIFEST, "rb") as fh:
        m = tomllib.load(fh)
    return {a["id"]: a for a in m["asset"] if a["id"] in ids}


def _lufs(y) -> float:
    import pyloudnorm
    meter = pyloudnorm.Meter(SR, block_size=min(0.4, len(y) / SR * 0.99))
    return float(meter.integrated_loudness(y))


def _ogg(wav: Path) -> Path:
    ogg = wav.with_suffix(".ogg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libvorbis", "-q:a", "6",
                    str(ogg)], check=True)
    return ogg


def _band_peaks(x, lo, hi, k=3):
    seg = x[: seconds(min(20.0, len(x) / SR))]
    return [round(f) for f, _ in spectrum_peaks(seg, k, fmin=lo, fmax=hi, zero_pad=1)]


def _bed_report(name: str, y: np.ndarray) -> str:
    n = len(y)
    mono = y.mean(axis=1)
    corr = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1])
    loss = _lufs(y) - _lufs(np.stack([mono, mono], axis=1))
    d = np.abs(np.diff(y, axis=0))
    seam = float(np.max(np.abs(y[0] - y[-1])) / (np.percentile(d, 99.9) + 1e-12))
    w = seconds(0.25)
    rms_end, rms_start = _rms(y[-w:]), _rms(y[:w])
    peak = 20 * np.log10(np.max(np.abs(y)) + 1e-12)
    cen = float(np.sum(np.abs(np.fft.rfft(mono)) * np.fft.rfftfreq(n, 1 / SR)) /
                np.sum(np.abs(np.fft.rfft(mono))))
    return (f"{name}: {n / SR:.1f}s peak {peak:.1f} dBFS  LUFS {_lufs(y):.1f}  L/R corr {corr:+.2f}  "
            f"mono-fold {loss:+.1f} dB  seam jump/p99.9 {seam:.2f}  seam RMS end/start "
            f"{20 * np.log10(rms_end / rms_start):+.1f} dB  centroid {cen:.0f} Hz")


def _component_checks(rng) -> list[str]:
    """Measured numbers vs the sources' numbers, written to out/dev/ambience/."""
    dev = DEV_DIR / "ambience"
    lines = []
    # wind wires: Q60 whistles at 600+400w and 1000+1000w
    dom = Dom(20.0, False)
    for const in (0.2, 0.8):
        w = np.full(dom.m, const)
        y = _mono(wind_scene(20.0, rng, sources=("wire1",), w=w))
        y2 = _mono(wind_scene(20.0, rng, sources=("wire2",), w=w))
        lines.append(f"wind wire1 w={const}: peak {_band_peaks(y, 300, 3000, 1)} Hz (source {600 + 400 * const:.0f})"
                     f"; wire2 {_band_peaks(y2, 300, 4000, 1)} Hz (source {1000 + 1000 * const:.0f})")
    wv = windspeed(Dom(120.0, True), rng)
    lines.append(f"windspeed walk: mean {wv.mean():.2f} min {wv.min():.2f} max {wv.max():.2f} (patch range 0..1)")
    # fire
    y, info = fire(30.0, rng, units=1, return_info=True)
    write_wav(dev / "fire_unit.wav", master(y, "detail"))
    lap = lop(y.mean(axis=1), 200.0)
    lines.append(f"fire: crackles {info['crackles_per_s']:.1f}/s (window 50-51 dB of lop~1 noise); "
                 f"lapping peak {_band_peaks(lap, 10, 200, 1)} Hz (source bp~ 30 Q5)")
    # creak
    f = _gesture_force(4.0, rng)
    pulses, times = stickslip_pulses(f, rng, return_times=True)
    per = np.diff(times) / SR * 1000
    c = creak(f, rng)
    write_wav(dev / "creak_wood.wav", master(c, "sfx"))
    imp = np.zeros(seconds(1.0))
    imp[0] = 1.0
    ir = 0.2 * imp
    for fq, q in DOOR_FORMANTS:
        ir = ir + bp(imp, fq, q)
    lines.append(f"creak: {len(times)} slips, period {per.min():.1f}-{per.max():.1f} ms (source 3-63 ms + jitter)"
                 f"; formant bank IR peaks {sorted(_band_peaks(ir, 40, 1000, 6))} Hz "
                 f"(source 62.5 125 250 395 560 790, Q 1-3 so neighbours merge)")
    # stream
    s = stream(6.0, rng, d=0.3, instances=420)
    write_wav(dev / "stream.wav", master(s, "detail"))
    lines.append(f"stream d=0.3: peaks {_band_peaks(s.mean(axis=1), 200, 16000, 5)} Hz (centres 300+15700 r^5)")
    # chain
    t1 = tinkle(rng, f_scale=1.0)
    lines.append(f"tinkle: peaks {sorted(_band_peaks(t1, 3000, 8000, 3))} Hz (source f0 4.5-5.5k, +700-800, +300-400)")
    # thunder
    th, ti = thunder(rng, return_info=True)
    write_wav(dev / "thunder.wav", master(th, "sfx"))
    lines.append(f"thunder: {ti['strikes']} strike events over {ti['strike_span_ms']:.0f} ms (source ~20 over ~1 s)")
    # tube bubble
    tb = tube_bubble(0.02, rng)
    lines.append(f"tube bubble r=20 mm: peaks {_band_peaks(tb, 50, 2000, 3)} Hz (Minnaert 3.29/r = {3.29 / 0.02:.0f})")
    return lines


def selftest():
    import time
    rng0 = np.random.default_rng(20260926)
    print("component checks:")
    for ln in _component_checks(np.random.default_rng(1)):
        print("  " + ln)
    rows = _rows(set(SELFTEST_BEDS) | set(SELFTEST_PLACES))
    for rid in SELFTEST_BEDS:
        row = rows[rid]
        t0 = time.time()
        rng = np.random.default_rng(zlib.crc32(rid.encode()))
        y = render(row, 0, rng)
        assert np.all(np.isfinite(y)), rid
        y = master_loop(y, "bed")
        rel = Path(row["path"]).relative_to("assets/audio")
        wav = write_wav((OUT_DIR / rel).with_suffix(".wav"), y)
        _ogg(wav)
        print(_bed_report(str(rel), y) + f"  [{time.time() - t0:.0f}s]")
    for rid in SELFTEST_PLACES:
        row = rows[rid]
        pal = place_palette(row)
        rel = Path(row["path"]).relative_to("assets/audio")
        for i in range(N_DETAILS):
            rng = np.random.default_rng(zlib.crc32(rid.encode()) + i)
            y = render(row, i, rng)
            assert np.all(np.isfinite(y)), (rid, i)
            y = master(y, "detail")
            wav = write_wav(OUT_DIR / rel / f"detail_{i + 1}.wav", y)
            _ogg(wav)
            pk = spectrum_peaks(y, 2, fmin=40)
            print(f"{rel}/detail_{i + 1} [{pal[i]}]: {len(y) / SR:.1f}s peak "
                  f"{20 * np.log10(np.max(np.abs(y)) + 1e-12):.1f} dBFS LUFS {_lufs(y):.1f} "
                  f"main peaks {[round(f) for f, _ in pk]} Hz")
    del rng0


if __name__ == "__main__":
    selftest()
