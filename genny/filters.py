"""Filters (RBJ biquads through scipy.lfilter). Static cutoff, or a per-sample cutoff: a TPT state-variable
filter for lowpass / highpass / bandpass / notch, block-wise RBJ for the others."""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .core import DEFAULT_SR


def _rbj(kind: str, fc: float, sr: int, q: float = 0.707, gain_db: float = 0.0):
    fc = float(np.clip(fc, 10.0, sr * 0.49))
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * max(q, 0.05))
    A = 10 ** (gain_db / 40)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1, -2 * cw, 1]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    elif kind == "allpass":
        b = [1 - alpha, -2 * cw, 1 + alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"unknown filter kind {kind}")
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def _per_channel(fn, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return fn(x)
    return np.stack([fn(x[:, c]) for c in range(x.shape[1])], axis=1)


# Zavalishin 4.23 and following: AP = x - 4R BP, resonance = LP - HP, band shelf = x + K 2R BP, K = 10^(GdB/20) - 1.
_SVF_MODES = {"lowpass": 0, "highpass": 1, "bandpass": 2, "notch": 3, "allpass": 4, "resonance": 5, "bandshelf": 6,
              "lowshelf": 7, "highshelf": 8, "tilt": 9}
_SWEPT_SVF = ("lowpass", "highpass", "bandpass", "notch", "allpass")    # biquad's "peak" is the RBJ EQ band, not LP - HP


@numba.njit(cache=True)
def _svf_kernel(x, fc, sr, r, mode, k=0.0):
    """Zavalishin's TPT 2-pole SVF ("The Art of VA Filter Design" 2.1, eq. 4.14 and its code), g = tan(pi fc/fs)
    per sample. Same response as the RBJ biquad at a fixed cutoff, but the states are the integrators, so a
    swept cutoff neither zippers nor blows up."""
    y = np.empty_like(x)
    s1 = 0.0
    s2 = 0.0
    fp = -1.0
    g = 0.0
    for n in range(x.shape[0]):
        if fc[n] != fp:
            fp = fc[n]
            f = min(max(fp, 10.0), 0.49 * sr)
            g = math.tan(math.pi * f / sr)
            if mode == 7:
                g = g / math.sqrt(k)
            elif mode >= 8:
                g = g * math.sqrt(k)
        hp = (x[n] - (2.0 * r + g) * s1 - s2) / (1.0 + 2.0 * r * g + g * g)
        v1 = g * hp
        bp = v1 + s1
        s1 = bp + v1
        v2 = g * bp
        lp = v2 + s2
        s2 = lp + v2
        if mode == 0:
            y[n] = lp
        elif mode == 1:
            y[n] = hp
        elif mode == 2:
            y[n] = 2.0 * r * bp           # unit gain at the centre, as the RBJ band-pass
        elif mode == 3:
            y[n] = x[n] - 2.0 * r * bp
        elif mode == 4:
            y[n] = x[n] - 4.0 * r * bp
        elif mode == 5:
            y[n] = lp - hp
        elif mode == 6:
            y[n] = x[n] + k * 2.0 * r * bp
        elif mode == 7:
            y[n] = k * k * lp + k * 2.0 * r * bp + hp
        elif mode == 8:
            y[n] = lp + k * 2.0 * r * bp + k * k * hp
        else:
            y[n] = lp / k + 2.0 * r * bp + k * hp
    return y


@numba.njit(cache=True)
def _logcosh(a):
    b = abs(a)
    return b + math.log1p(math.exp(-2.0 * b)) - math.log(2.0)


@numba.njit(cache=True)
def _moog_kernel(x, fc, sr, k, drive, nl=0):
    """Zavalishin's TPT 4-pole ladder (VA Filter Design ch. 5): u = (x - k S) / (1 + k G^4), tanh at the feedback node.
    Self-oscillates at k = 4. nl 0: tanh of the linear node estimate; nl 1: the node solved by Newton,
    v + k(G^4 tanh(d v) + S) = x; nl 2: the same delay-free loop with the antiderivative form of tanh (ADAA1 between
    the previous and the current node value), solved by Newton on its derivative."""
    y = np.empty_like(x)
    s = np.zeros(4)
    prev = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.45 * sr) / sr)
        G = g / (1.0 + g)
        c = 1.0 / (1.0 + g)
        S = G * G * G * s[0] * c + G * G * s[1] * c + G * s[2] * c + s[3] * c
        G4 = G * G * G * G
        v = (x[n] - k * S) / (1.0 + k * G4)
        if nl == 0:
            u = math.tanh(drive * v)
        elif nl == 1:
            for _ in range(16):
                t = math.tanh(drive * v)
                dv = (v + k * (G4 * t + S) - x[n]) / (1.0 + k * G4 * drive * (1.0 - t * t))
                v -= dv
                if abs(dv) < 1e-10:
                    break
            u = math.tanh(drive * v)
        else:
            for _ in range(16):
                a = drive * v
                e = a - prev
                if abs(e) > 1e-6:
                    u = (_logcosh(a) - _logcosh(prev)) / e
                    du = (math.tanh(a) * e - (_logcosh(a) - _logcosh(prev))) / (e * e)
                else:
                    u = math.tanh(0.5 * (a + prev))
                    du = 0.5 * (1.0 - u * u)
                dv = (v + k * (G4 * u + S) - x[n]) / (1.0 + k * G4 * drive * du)
                v -= dv
                if abs(dv) < 1e-10:
                    break
            a = drive * v
            e = a - prev
            u = (_logcosh(a) - _logcosh(prev)) / e if abs(e) > 1e-6 else math.tanh(0.5 * (a + prev))
            prev = a
        for i in range(4):
            v = (u - s[i]) * G
            u = v + s[i]
            s[i] = u + v
        y[n] = u
    return y


def ladder_k_max(order: int = 4) -> float:
    """Loop gain at which an ``order``-stage ladder 1 / ((1 + s)^N + k) goes unstable: sec(pi/N)^N (4 for N = 4;
    no limit below N = 3, 4 is returned)."""
    return 4.0 if order < 3 else 1.0 / math.cos(math.pi / order) ** order


def ladder_poles(order: int, k: float, fc: float) -> np.ndarray:
    """Poles (rad/s) of the N-stage ladder 1 / ((1 + s / w)^N + k), w = 2 pi fc: the circle of radius w k^(1/N)
    centred on -w, at angles pi (2 i + 1) / N."""
    w = 2.0 * math.pi * float(fc)
    i = np.arange(order)
    return w * (-1.0 + float(k) ** (1.0 / order) * np.exp(1j * math.pi * (2 * i + 1) / order))


def ladder_resonance(order: int, k: float, fc: float) -> tuple[float, float]:
    """(Hz, Q) of the leading pole pair of ``ladder_poles`` (the pair nearest the imaginary axis, angle pi / N); k = 0 gives (fc, 0.5)."""
    p = ladder_poles(order, k, fc)
    p = p[np.argmax(np.where(p.imag > 0, p.real, -1e30))] if order > 1 else p[0]
    return abs(p) / (2.0 * math.pi), abs(p) / (-2.0 * p.real) if p.real < 0 else float("inf")


def ladder_pole_ratio(order: int, k: float) -> float:
    """Ratio of the leading-pole frequency to the natural cut-off of the N-stage ladder: A0(k) = sqrt(1 + k^(2/N)
    - 2 k^(1/N) cos(pi / N)); 1 at k = 0."""
    r = float(k) ** (1.0 / order)
    return math.sqrt(max(1.0 + r * r - 2.0 * r * math.cos(math.pi / order), 0.0))


def ladder_k_from_q(order: int, q: float) -> float:
    """Loop gain k that gives the leading poles of the N-stage ladder (N >= 2) the quality factor ``q`` (>= 1/2);
    q -> infinity gives ``ladder_k_max``."""
    c, sn = math.cos(math.pi / order), math.sin(math.pi / order)
    q2 = 4.0 * q * q
    return ((((q2 - 1.0) * c - math.sqrt(q2 - 1.0) * sn) / (q2 * c * c - 1.0)) ** order) if q2 * c * c != 1.0 else float("nan")


DIODE_K_MAX = 17.0     # the diode ladder 1 / (8 (1 + s)^4 - 8 (1 + s)^2 + 1 + k) self-oscillates at k = 17, resonance at fc / sqrt(2)


def diode_ladder_poles(k: float, fc: float) -> np.ndarray:
    """Poles (rad/s) of the 4-stage diode ladder, the roots of 8 (1 + s/w)^4 - 8 (1 + s/w)^2 + 1 + k, w = 2 pi fc;
    at k = 0: -w (0.07612, 0.61732, 1.38268, 1.92388)."""
    u = np.roots([8.0, 0.0, -8.0, 0.0, 1.0 + float(k)])
    return 2.0 * math.pi * float(fc) * (u - 1.0)


LADDER_LIMITS = {"lp4": 4.0, "lp8": 1.0 / math.cos(math.pi / 8) ** 8, "bp4": 4.0, "bp4_stable": 16.0, "bp8": 16.0, "bp8_negative": -64.0, "diode": DIODE_K_MAX}


def ladder_dc_gain(k: float) -> float:
    """DC gain of the ladder, 1 / (1 + k) (the same for every number of stages)."""
    return 1.0 / (1.0 + float(k))


def ladder_cutoff_error(order: int, k: float) -> float:
    """Relative error |1 - 1 / A0(k)| between the natural cut-off and the leading-pole frequency; its maximum over k is
    1 for N <= 6 and csc(pi / N) - 1 above (N = 4: zero at k = k_max)."""
    return abs(1.0 - 1.0 / ladder_pole_ratio(order, k))


@numba.njit(cache=True)
def _dangelo_kernel(x, fc, sr, k, drive, order):
    """Explicit N-stage ladder, one tanh per stage and no delay in the loops: each stage w = p0 [(1 + g) w' - 2 g tanh w'
    + g (u + u')], p0 = 1 / (1 + g), exact in small signal for the trapezoidal ladder; the global loop is solved from
    the chain's free response, u = (x - k w_free) / (1 + k (g / (1 + g))^N). Output w_N / drive."""
    y = np.empty_like(x)
    w = np.zeros(order)
    t = np.zeros(order)
    xp = np.zeros(order)
    wf = np.zeros(order)
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.45 * sr) / sr)
        p0 = 1.0 / (1.0 + g)
        gam = g * p0
        u = 0.0
        for i in range(order):
            wf[i] = p0 * ((1.0 + g) * w[i] - 2.0 * g * t[i] + g * (u + xp[i]))
            u = math.tanh(wf[i])
        a = (drive * x[n] - k * wf[order - 1]) / (1.0 + k * gam ** order)
        u = math.tanh(a)
        for i in range(order):
            wi = p0 * ((1.0 + g) * w[i] - 2.0 * g * t[i] + g * (u + xp[i]))
            xp[i] = u
            w[i] = wi
            t[i] = math.tanh(wi)
            u = t[i]
        y[n] = w[order - 1] / drive
    return y


HUOVILAINEN_THERMAL = 0.000025
HUOVILAINEN_SCALE = 32768.0


@numba.njit(cache=True)
def _huovilainen_kernel(x, fc, sr, res, scale, th):
    """Huovilainen's Moog ladder as in Soundpipe moogladder: 2x oversampled, four tanh stages with thermal voltage
    th, cutoff and resonance tuned by the fcr / acr polynomials, output the half-sample average of the last stage."""
    y = np.empty_like(x)
    delay = np.zeros(6)
    tstg = np.zeros(3)
    stg = np.zeros(4)
    for n in range(x.shape[0]):
        f = min(max(fc[n], 10.0), 0.45 * sr) / sr
        f2 = f * f
        fcr = 1.8730 * f2 * f + 0.4955 * f2 - 0.6490 * f + 0.9988
        acr = -3.9364 * f2 + 1.8409 * f + 0.9968
        tune = (1.0 - math.exp(-(2.0 * math.pi) * (0.5 * f) * fcr)) / th
        res4 = 4.0 * res * acr
        xin = x[n] * scale
        for _ in range(2):
            u = xin - res4 * delay[5]
            stg[0] = delay[0] + tune * (math.tanh(u * th) - tstg[0])
            delay[0] = stg[0]
            for j in range(1, 4):
                tstg[j - 1] = math.tanh(stg[j - 1] * th)
                stg[j] = delay[j] + tune * (tstg[j - 1] - (tstg[j] if j != 3 else math.tanh(delay[j] * th)))
                delay[j] = stg[j]
            delay[5] = (stg[3] + delay[4]) * 0.5
            delay[4] = stg[3]
        y[n] = delay[5] / scale
    return y


LEVIEN_SQUARINGS = 10


@numba.njit(cache=True)
def _levien_kernel(x, fc, sr, k, drive, nl, squarings):
    """Levien's matrix ladder: the Euler step of y0' = a(x - k y3 - y0), yi' = a(y(i-1) - yi) over 1/2^N sample,
    raised to 2^N by N squarings (x held over the sample). nl: y += B tanh(x - k y3) + C tanh(y), C = A - I with
    k B added to its last column, so the small-signal case is the linear one."""
    y = np.empty_like(x)
    s = np.zeros(4)
    t = np.zeros(4)
    A = np.zeros((4, 4))
    B = np.zeros(4)
    C = np.zeros((4, 4))
    last = -1.0
    for n in range(x.shape[0]):
        f = min(max(fc[n], 10.0), 0.49 * sr)
        if f != last:
            a = 2.0 * math.pi * f / sr / (1 << squarings)
            M = np.eye(5)
            for i in range(4):
                M[i + 1, i] = a
                M[i + 1, i + 1] = 1.0 - a
            M[1, 4] = -k * a
            for _ in range(squarings):
                M = M @ M
            B[:] = M[1:, 0]
            A[:, :] = M[1:, 1:]
            C[:, :] = A - np.eye(4)
            C[:, 3] += k * B
            last = f
        xin = drive * x[n]
        if nl:
            for i in range(4):
                t[i] = math.tanh(s[i])
            u = math.tanh(xin - k * s[3])
            s = s + B * u + C @ t
        else:
            s = A @ s + B * xin
        y[n] = s[3] / drive
    return y


@numba.njit(cache=True)
def _diode_solve(g, k, u, s1, s2, s3, s4):
    """One trapezoidal step of the chain with feedback k, node input u: rows d y1 - g y2 + g k y4 = s1 + g u;
    -h y1 + d y2 - h y3 = s2; -h y2 + d y3 - h y4 = s3; -h y3 + d y4 = s4."""
    h = 0.5 * g
    d = 1.0 + g
    A1, B1, C1 = g / d, -g * k / d, (s1 + g * u) / d
    D = d - h * A1
    A2, B2, C2 = h / D, h * B1 / D, (s2 + h * C1) / D
    D = d - h * A2
    B3, C3 = h * (1.0 + B2) / D, (s3 + h * C2) / D
    y4 = (s4 + h * C3) / (d - h * B3)
    y3 = B3 * y4 + C3
    y2 = A2 * y3 + B2 * y4 + C2
    return A1 * y2 + B1 * y4 + C1, y2, y3, y4


@numba.njit(cache=True)
def _diode_kernel(x, fc, sr, k, drive):
    """Zavalishin's diode ladder (5.18): y1' = w((u + y2) - y1), y2' = w((y1 + y3)/2 - y2), y3' = w((y2 + y4)/2 - y3),
    y4' = w(y3/2 - y4), u = in - k y4, by the trapezoidal rule. H = 1/(8(1+s)^4 - 8(1+s)^2 + 1 + k): self-oscillates
    at k = 17, at w = wc/sqrt(2). The linear solve finds the feedback, then tanh at the node bounds the loop."""
    out = np.empty_like(x)
    s1 = s2 = s3 = s4 = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.45 * sr) / sr)
        y4 = _diode_solve(g, k, x[n], s1, s2, s3, s4)[3]
        y1, y2, y3, y4 = _diode_solve(g, 0.0, math.tanh(drive * (x[n] - k * y4)), s1, s2, s3, s4)
        s1, s2, s3, s4 = 2.0 * y1 - s1, 2.0 * y2 - s2, 2.0 * y3 - s3, 2.0 * y4 - s4
        out[n] = y4
    return out


@numba.njit(cache=True)
def follow(x, attack, release):
    """One-pole attack / release smoother; attack and release are per-sample coefficients 1 - exp(-1/(tau sr))."""
    y = np.empty_like(x)
    z = 0.0
    for n in range(x.shape[0]):
        z += (attack if x[n] > z else release) * (x[n] - z)
        y[n] = z
    return y


@numba.njit(cache=True)
def _lpf18_kernel(x, fc, sr, res, dist):
    """Comajuncosas' 3-pole LPF18 as in Soundpipe: polynomial-tuned one-pole stages, tanh feedback, tanh output with distortion."""
    y = np.empty_like(x)
    ay1 = 0.0
    ay2 = 0.0
    aout = 0.0
    lastin = 0.0
    for n in range(x.shape[0]):
        kfcn = 2.0 * min(max(fc[n], 10.0), 0.45 * sr) / sr
        kp = ((-2.7528 * kfcn + 3.0429) * kfcn + 1.718) * kfcn - 0.9984
        kp1 = kp + 1.0
        kp1h = 0.5 * kp1
        kres = res * (((-2.7079 * kp1 + 10.963) * kp1 - 14.934) * kp1 + 8.4974)
        value = 1.0 + dist * (1.5 + 2.0 * res * (1.0 - kfcn))
        ax1 = lastin
        ay11 = ay1
        ay31 = ay2
        lastin = x[n] - math.tanh(kres * aout)
        ay1 = kp1h * (lastin + ax1) - kp * ay1
        ay2 = kp1h * (ay1 + ay11) - kp * ay2
        aout = kp1h * (ay2 + ay31) - kp * aout
        y[n] = math.tanh(aout * value)
    return y


def faust_q_to_k(kind: str, q: float, norm_freq: float = 0.0) -> float:
    """Faust ve.* mapping of a Q (0.7071..25, 10 for korg35) to the loop gain: moog 4(Q - .7071)/(25 - .7071),
    diode (17 - n^10 9.7)(Q - .7071)/(25 - .7071) with n the normalised cutoff 0..1, korg35 2(Q - .7071)/(10 - .7071)."""
    q = float(q) - 0.7071
    if kind == "moog":
        return 4.0 * q / (25.0 - 0.7071)
    if kind == "diode":
        return (17.0 - float(norm_freq) ** 10 * 9.7) * q / (25.0 - 0.7071)
    if kind == "korg35":
        return 2.0 * q / (10.0 - 0.7071)
    raise ValueError("kind must be moog|diode|korg35")


def ladder(x: np.ndarray, cutoff, sr: int = DEFAULT_SR, res: float = 0.5, kind: str = "moog", drive: float = 1.0,
           solver: str = "node", q: float | None = None, order: int = 4, pole_cutoff: bool = False) -> np.ndarray:
    """Transistor (``moog``) or diode (``diode``, TB-303 type) ladder low-pass; ``res`` 1 is the edge of self-oscillation
    (k = 4 and 17). ``cutoff`` is a number or per-sample; ``drive`` is the level into the tanh.
    ``lpf18``: the 3-pole LPF18 (18 dB/oct), ``res`` 0..1 as in Soundpipe, ``drive`` its distortion (default there 2).
    ``huovilainen``: Huovilainen's 2x-oversampled tanh-per-stage Moog (Soundpipe moogladder), ``res`` 0..1, ``drive``
    the input level against the thermal voltage (1 = full scale 32768, as Csound).
    ``dangelo``: explicit ``order``-stage ladder (D'Angelo & Valimaki), one tanh per stage, ``res`` 0..1 of ``ladder_k_max(order)``,
    ``drive`` the level into the tanh, output / drive; ``pole_cutoff`` makes ``cutoff`` the frequency of the leading poles
    (``ladder_pole_ratio``) instead of the natural cut-off, so the resonance stays where it is set whatever ``res``.
    ``levien_linear``: Levien's exact sample-and-hold matrix ladder (tuning exact up to Nyquist, no oversampling);
    ``levien``: the same matrix with tanh on the input and on each stage; ``drive`` scales into the tanh, output / drive.
    ``solver`` (moog): ``node`` tanh of the linear node estimate, ``newton`` the delay-free loop solved by Newton,
    ``adaa`` the same loop with an antiderivative-antialiased tanh. ``q`` (moog, diode): a Faust Q 0.7071..25 in place
    of ``res`` (``faust_q_to_k``); the diode then pre-saturates its input as Faust does (x 1.5, cubic)."""
    if kind == "lpf18":
        def one18(sig):
            fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
            return _lpf18_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                                 float(np.clip(res, 0.0, 1.0)), float(max(drive, 0.0)))
        return _per_channel(one18, np.asarray(x, float))
    if kind == "huovilainen":
        def oneh(sig):
            fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
            return _huovilainen_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                                       float(np.clip(res, 0.0, 1.0)), HUOVILAINEN_SCALE * float(max(drive, 1e-6)),
                                       HUOVILAINEN_THERMAL)
        return _per_channel(oneh, np.asarray(x, float))
    if kind == "dangelo":
        n_st = int(np.clip(order, 1, 16))
        kd = ladder_k_max(n_st) * float(np.clip(res, 0.0, 1.05))
        ratio = ladder_pole_ratio(n_st, kd) if pole_cutoff and n_st >= 2 and kd > 0.0 else 1.0
        def oned(sig):
            fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64) / max(ratio, 1e-3), sig.shape)
            return _dangelo_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                                   kd, float(max(drive, 1e-6)), n_st)
        return _per_channel(oned, np.asarray(x, float))
    if kind in ("levien", "levien_linear"):
        kl = 4.0 * float(np.clip(res, 0.0, 1.05))
        def onel(sig):
            fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
            return _levien_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                                  kl, float(max(drive, 1e-6)), kind == "levien", LEVIEN_SQUARINGS)
        return _per_channel(onel, np.asarray(x, float))
    if kind not in ("moog", "diode"):
        raise ValueError("kind must be moog|diode|lpf18|huovilainen|levien|levien_linear|dangelo")
    if solver not in ("node", "newton", "adaa"):
        raise ValueError("solver must be node|newton|adaa")
    top = 17.0 if kind == "diode" else 4.0
    if q is None:
        k = top * float(np.clip(res, 0.0, 1.05))
    else:
        k = faust_q_to_k(kind, q, float(np.clip(np.mean(np.asarray(cutoff, dtype=np.float64)) / (0.5 * sr), 0.0, 1.0)))
    x = np.asarray(x, float)
    if q is not None and kind == "diode":
        v = np.clip(1.5 * x, -1.0, 1.0)
        x = v - v ** 3 / 3.0
    nl = {"node": 0, "newton": 1, "adaa": 2}[solver]
    def one(sig):
        fc = np.ascontiguousarray(np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape))
        sig = np.ascontiguousarray(sig, dtype=np.float64)
        if kind == "moog":
            return _moog_kernel(sig, fc, float(sr), k, float(drive), nl)
        return _diode_kernel(sig, fc, float(sr), k, float(drive))
    return _per_channel(one, x)


def svf_peak_gain(q: float) -> float:
    """Peak gain of the 2-pole lowpass/highpass: Q / sqrt(1 - 1/(4 Q^2)) above Q = 1/sqrt 2, else 1."""
    q = max(float(q), 0.05)
    return q / math.sqrt(1.0 - 0.25 / (q * q)) if q > 0.7072 else 1.0


def svf(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR, q: float = 0.707, gain_db: float = 0.0,
        level: str = "none") -> np.ndarray:
    """TPT state-variable filter (lowpass, highpass, bandpass, notch, allpass, resonance = LP - HP, bandshelf with
    `gain_db` at the centre; lowshelf M^2 LP + M 2R BP + HP with g / sqrt M, highshelf LP + M 2R BP + M^2 HP with
    g sqrt M, tilt LP / M + 2R BP + M HP with g sqrt M: -gain_db/2 below, +gain_db/2 above, M = 10^(gain_db/40)) with a scalar or
    per-sample cutoff, so a shelf or a tilt can be swept. level="peak": lowpass/highpass divided by their resonant
    peak, so raising Q does not raise the level."""
    if level == "peak" and kind in ("lowpass", "highpass"):
        return svf(x, kind, cutoff, sr, q, gain_db) / svf_peak_gain(q)
    k = 10.0 ** (float(gain_db) / 40.0) if kind in ("lowshelf", "highshelf", "tilt") else 10.0 ** (float(gain_db) / 20.0) - 1.0
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        if fc.shape[0] != sig.shape[0]:
            fc = np.interp(np.linspace(0, 1, sig.shape[0]), np.linspace(0, 1, fc.shape[0]), fc)
        return _svf_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                           0.5 / max(float(q), 0.05), _SVF_MODES[kind], k)
    return _per_channel(one, np.asarray(x, float))


def svf_band(x: np.ndarray, kind: str, center: float, bw_oct: float, sr: int = DEFAULT_SR, gain_db: float = 0.0) -> np.ndarray:
    """SVF whose -3 dB edges sit exactly at center 2^(+-bw_oct/2), also near Nyquist. Zavalishin 4.19-4.20: R = sinh(D ln2 / 2);
    prewarping: g = sqrt(mu(hi) mu(lo)), D' = log2(mu(hi) / mu(lo)), mu(f) = tan(pi f / sr)."""
    hi = min(center * 2.0 ** (bw_oct / 2.0), 0.49 * sr)
    lo = max(center * 2.0 ** (-bw_oct / 2.0), 1.0)
    mh, ml = math.tan(math.pi * hi / sr), math.tan(math.pi * lo / sr)
    r = math.sinh(math.log2(mh / ml) * math.log(2.0) / 2.0)
    return svf(x, kind, math.atan(math.sqrt(mh * ml)) * sr / math.pi, sr, 0.5 / r, gain_db)


@numba.njit(cache=True)
def _onepole_kernel(x, fc, sr, mode):
    """Zavalishin's TPT one-pole: v = (x - s) G, lp = v + s, s = lp + v, G = g / (1 + g); hp = x - lp, ap = lp - hp."""
    y = np.empty_like(x)
    s = 0.0
    fp = -1.0
    g = 0.0
    for n in range(x.shape[0]):
        if fc[n] != fp:
            fp = fc[n]
            g = math.tan(math.pi * min(max(fp, 1.0), 0.49 * sr) / sr)
        v = (x[n] - s) * g / (1.0 + g)
        lp = v + s
        s = lp + v
        y[n] = lp if mode == 0 else (x[n] - lp if mode == 1 else 2.0 * lp - x[n])
    return y


def onepole(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR) -> np.ndarray:
    """TPT one-pole lowpass, highpass or allpass (phase -90 degrees at the cutoff), scalar or per-sample cutoff."""
    mode = {"lowpass": 0, "highpass": 1, "allpass": 2}[kind]
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        return _onepole_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr), mode)
    return _per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _resonz_kernel(x, fc, q, sr, norm):
    """R = exp(-pi (fc / Q) / sr), H = g (1 - z^-2) / (1 - 2 R cos(theta) z^-1 + R^2 z^-2) (rjlib e_resonzq)."""
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    fp = -1.0
    R = g = c = 0.0
    for n in range(x.shape[0]):
        if fc[n] != fp:
            fp = fc[n]
            f = min(max(fp, 1.0), 0.49 * sr)
            R = math.exp(-math.pi * (f / q) / sr)
            g = 1.0 if norm == 0 else ((1.0 - R * R) / 2.0 if norm == 1 else math.sqrt((1.0 - R * R) / 2.0))
            c = math.cos(2.0 * math.pi * f / sr)
        v = g * (x[n] - x2) + 2.0 * R * c * y1 - R * R * y2
        x2, x1 = x1, x[n]
        y2, y1 = y1, v
        y[n] = v
    return y


def resonz(x: np.ndarray, center, sr: int = DEFAULT_SR, q: float = 5.0, norm: str = "power") -> np.ndarray:
    """Two-pole resonator with zeros at DC and Nyquist. norm: none | peak (unit gain at the centre, level on noise falls with Q) |
    power (level on white noise stays put, for a steady level while sweeping; rjlib e_resonzq)."""
    m = {"none": 0, "peak": 1, "power": 2}[norm]
    def one(sig):
        fc = np.broadcast_to(np.asarray(center, dtype=np.float64), sig.shape)
        return _resonz_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), max(float(q), 0.05), float(sr), m)
    return _per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _phasor_kernel(u, f, tau, sr):
    x = y = 0.0
    out = np.empty_like(u)
    for n in range(u.shape[0]):
        r = math.exp(-1.0 / (tau[n] * sr)) if tau[n] > 0.0 else 1.0
        th = 2.0 * math.pi * f[n] / sr
        x1, y1 = r * math.cos(th), r * math.sin(th)
        x, y = x1 * x - y1 * y + u[n], y1 * x + x1 * y
        out[n] = y
    return out


def phasor_reson(x: np.ndarray, freq, tau, sr: int = DEFAULT_SR) -> np.ndarray:
    """Two-pole resonator as a rotating complex state z(n+1) = r e^(j theta) z(n) + u(n), output Im z: H = r sin(theta) z^-2 /
    (1 - 2 r cos(theta) z^-1 + r^2 z^-2), r = exp(-1 / (tau sr)). ``freq`` (Hz) and ``tau`` (s to fall to 1/e; 0 or inf = no
    decay) may be per-sample: changing them leaves the ringing amplitude untouched, so very high Q can be swept without clicks."""
    def one(sig):
        f = np.broadcast_to(np.asarray(freq, dtype=np.float64), sig.shape)
        t = np.nan_to_num(np.broadcast_to(np.asarray(tau, dtype=np.float64), sig.shape), posinf=0.0)
        return _phasor_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(f), np.ascontiguousarray(t), float(sr))
    return _per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _korg35_kernel(x, fc, sr, res, sat):
    """Soundpipe modules/wpkorg35.c (Pirkle): G = g / (1 + g), alpha = 1 / (1 - res G + res G^2), feedback through a one-pole
    lowpass and highpass, tanh(sat u) in the loop."""
    y = np.empty_like(x)
    z1 = z2 = zh = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.45 * sr) / sr)
        G = g / (1.0 + g)
        b2 = (res - res * G) / (1.0 + g)
        bh = -1.0 / (1.0 + g)
        a = 1.0 / (1.0 - res * G + res * G * G)
        v = (x[n] - z1) * G
        y1 = v + z1
        z1 = y1 + v
        u = a * (y1 + zh * bh + z2 * b2)
        if sat > 0.0:
            u = math.tanh(sat * u)
        v = (u - z2) * G
        o = v + z2
        z2 = o + v
        v = (o * res - zh) * G
        zh = v + (v + zh)
        y[n] = o
    return y


@numba.njit(cache=True)
def _korg35_hp_kernel(x, fc, sr, res, sat):
    """Korg 35 high-pass: one-pole HP on the input, loop u = alpha (y1 - G zh / (1 + g) + zl / (1 + g)), y = res tanh(sat u),
    y feeds a one-pole HP then a one-pole LP whose states are the feedback; output y / res."""
    y = np.empty_like(x)
    z1 = zh = zl = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.45 * sr) / sr)
        G = g / (1.0 + g)
        a = 1.0 / (1.0 - res * G + res * G * G)
        v = (x[n] - z1) * G
        lp = v + z1
        z1 = lp + v
        y1 = x[n] - lp
        u = a * (y1 - zh * G / (1.0 + g) + zl / (1.0 + g))
        if sat > 0.0:
            u = math.tanh(sat * u)
        o = res * u
        v = (o - zh) * G
        lp = v + zh
        zh = lp + v
        h = o - lp
        v = (h - zl) * G
        lp = v + zl
        zl = lp + v
        y[n] = o / res
    return y


def korg35(x: np.ndarray, cutoff, sr: int = DEFAULT_SR, res: float = 1.0, sat: float = 1.0, kind: str = "lowpass") -> np.ndarray:
    """Korg 35 (MS-20) low-pass or high-pass, scalar or per-sample cutoff; `res` 0..2 (2 is the edge of self-oscillation,
    Soundpipe default 1)."""
    kern = _korg35_hp_kernel if kind == "highpass" else _korg35_kernel
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        return kern(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                    float(np.clip(res, 1e-3, 1.999)), float(sat))
    return _per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _ra_kernel(x, fc, sr, bw, v0):
    """DAFX 2.3.2: y = x + (H0/2)(x - A2(x)), A2 = (-c + d(1-c) z^-1 + z^-2) / (1 + d(1-c) z^-1 - c z^-2), d = -cos(2 pi fc / sr),
    H0 = V0 - 1, c = (t - 1)/(t + 1) to boost and (V0 t - 1)/(V0 t + 1) to cut, t = tan(pi bw / sr). The state is the allpass's own,
    so moving fc has no transient."""
    y = np.empty_like(x)
    h0 = v0 - 1.0
    t = math.tan(math.pi * bw / sr)
    c = (t - 1.0) / (t + 1.0) if v0 >= 1.0 else (v0 * t - 1.0) / (v0 * t + 1.0)
    s1 = s2 = 0.0
    for n in range(x.shape[0]):
        d = -math.cos(2.0 * math.pi * min(max(fc[n], 10.0), 0.49 * sr) / sr)
        e = d * (1.0 - c)
        ap = -c * x[n] + s1
        s1 = e * x[n] - e * ap + s2
        s2 = x[n] + c * ap
        y[n] = x[n] + 0.5 * h0 * (x[n] - ap)
    return y


def peak_allpass(x: np.ndarray, center, sr: int = DEFAULT_SR, bw: float = 200.0, gain_db: float = 6.0) -> np.ndarray:
    """Regalia-Mitra peak filter built on an allpass: gain_db at `center` (scalar or per-sample), unity at DC and Nyquist."""
    def one(sig):
        fc = np.broadcast_to(np.asarray(center, dtype=np.float64), sig.shape)
        return _ra_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr), float(bw), 10.0 ** (float(gain_db) / 20.0))
    return _per_channel(one, np.asarray(x, float))


def matched_coeffs(kind: str, fc: float, sr: int = DEFAULT_SR, q: float = 0.707, gain_db: float = 0.0):
    """Vicanek, Matched Second Order Digital Filters: poles from impulse invariance (eq. 11-12), so no peak narrowing near Nyquist;
    zeros matched to the analog prototype at the cutoff (lowpass eq. 30-34, highpass 36, bandpass 39-41, peak 44-45 with 29)."""
    f = float(np.clip(fc, 1.0, 0.495 * sr))
    w = 2.0 * math.pi * f / sr
    G = 10.0 ** (gain_db / 20.0)
    Qd = max(q, 0.05) * (math.sqrt(G) if kind == "peak" else 1.0)
    d = 1.0 / (2.0 * Qd)
    a1 = -2.0 * math.exp(-d * w) * (math.cos(math.sqrt(1.0 - d * d) * w) if d <= 1.0 else math.cosh(math.sqrt(d * d - 1.0) * w))
    a2 = math.exp(-2.0 * d * w)
    p1 = math.sin(w / 2.0) ** 2
    p0 = 1.0 - p1
    p2 = 4.0 * p0 * p1
    A0, A1, A2 = (1.0 + a1 + a2) ** 2, (1.0 - a1 + a2) ** 2, -4.0 * a2
    D = A0 * p0 + A1 * p1 + A2 * p2
    D2 = -A0 + A1 + 4.0 * (p0 - p1) * A2
    if kind == "lowpass":
        B1 = max((D * q * q - A0 * p0) / p1, 0.0)
        b0 = 0.5 * (1.0 + a1 + a2 + math.sqrt(B1))
        b = [b0, 1.0 + a1 + a2 - b0, 0.0]
    elif kind == "highpass":
        b0 = math.sqrt(D) / (4.0 * p1) * q
        b = [b0, -2.0 * b0, b0]
    elif kind == "bandpass":
        B2 = (D - D2 * p1) / (4.0 * p1 * p1)
        B1 = max(D2 + 4.0 * (p1 - p0) * B2, 0.0)
        b1 = -0.5 * math.sqrt(B1)
        b0 = 0.5 * (math.sqrt(max(B2 + b1 * b1, 0.0)) - b1)
        b = [b0, b1, -b0 - b1]
    elif kind == "peak":
        R1, R2 = D * G * G, D2 * G * G
        B2 = (R1 - R2 * p1 - A0) / (4.0 * p1 * p1)
        B1 = max(R2 + A0 + 4.0 * (p1 - p0) * B2, 0.0)
        W = 0.5 * (math.sqrt(A0) + math.sqrt(B1))
        b0 = 0.5 * (W + math.sqrt(max(W * W + B2, 0.0)))
        b = [b0, 0.5 * (math.sqrt(A0) - math.sqrt(B1)), -B2 / (4.0 * b0)]
    else:
        raise ValueError("matched filters: lowpass|highpass|bandpass|peak")
    return np.array(b), np.array([1.0, a1, a2])


def matched(x: np.ndarray, kind: str, fc: float, sr: int = DEFAULT_SR, q: float = 0.707, gain_db: float = 0.0) -> np.ndarray:
    """Biquad matched to its analog prototype: unlike the RBJ forms, the response does not cramp near Nyquist."""
    b, a = matched_coeffs(kind, fc, sr, q, gain_db)
    return lfilter(b, a, x, axis=0)


def biquad(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR, q: float = 0.707, gain_db: float = 0.0,
           order: int = 1, block: int = 64) -> np.ndarray:
    """Apply an RBJ biquad. `cutoff` may be a scalar or a per-sample array: a modulated lowpass /
    highpass / bandpass / notch runs as a TPT state-variable filter (`svf`), the other kinds block-wise.

    order = number of cascaded biquads (1 = 12 dB/oct, 2 = 24 dB/oct). The cascade is not retuned: each section keeps its gain at
    the cutoff, so order 1/2/4 are -3/-6/-12 dB there and a resonance doubles in dB with each section (Zavalishin c27).
    """
    if not np.isscalar(cutoff) and kind in _SWEPT_SVF:
        y = x
        for _ in range(max(1, order)):
            y = svf(y, kind, cutoff, sr, q)
        return y
    def one(sig: np.ndarray) -> np.ndarray:
        y = sig
        for _ in range(max(1, order)):
            if np.isscalar(cutoff):
                b, a = _rbj(kind, float(cutoff), sr, q, gain_db)
                y = lfilter(b, a, y)
            else:
                fc = np.asarray(cutoff, dtype=np.float64)
                if fc.shape[0] != y.shape[0]:
                    fc = np.interp(np.linspace(0, 1, y.shape[0]), np.linspace(0, 1, fc.shape[0]), fc)
                out = np.empty_like(y)
                zi = None
                for s in range(0, y.shape[0], block):
                    e = min(s + block, y.shape[0])
                    b, a = _rbj(kind, float(fc[s:e].mean()), sr, q, gain_db)
                    if zi is None:
                        zi = lfilter_zi(b, a) * y[0]
                    out[s:e], zi = lfilter(b, a, y[s:e], zi=zi)
                y = out
        return y

    return _per_channel(one, x)


def lowpass(x, cutoff, sr=DEFAULT_SR, q=0.707, order=1):
    return biquad(x, "lowpass", cutoff, sr, q, order=order)


def highpass(x, cutoff, sr=DEFAULT_SR, q=0.707, order=1):
    return biquad(x, "highpass", cutoff, sr, q, order=order)


def bandpass(x, center, sr=DEFAULT_SR, q=1.0, order=1):
    return biquad(x, "bandpass", center, sr, q, order=order)


def notch(x, center, sr=DEFAULT_SR, q=4.0):
    return biquad(x, "notch", center, sr, q)


def peak(x, center, sr=DEFAULT_SR, q=1.0, gain_db=6.0):
    return biquad(x, "peak", center, sr, q, gain_db)


def lowshelf(x, cutoff, sr=DEFAULT_SR, gain_db=6.0, q=0.707):
    """Low shelf; `q` above 0.707 makes the shelf overshoot before it levels, below 0.707 it is gentler."""
    return biquad(x, "lowshelf", cutoff, sr, q, gain_db)


def highshelf(x, cutoff, sr=DEFAULT_SR, gain_db=6.0, q=0.707):
    """High shelf; `q` as for `lowshelf`."""
    return biquad(x, "highshelf", cutoff, sr, q, gain_db)


def dc_block(x, sr=DEFAULT_SR):
    return biquad(x, "highpass", 20.0, sr, 0.707)


def matched_shelf(x: np.ndarray, kind: str, fc: float, sr: int = DEFAULT_SR, gain_db: float = 6.0) -> np.ndarray:
    """First-order shelf matched to its analog prototype (s + G wc)/(s + wc) (low) or (G s + wc)/(s + wc) (high):
    pole by impulse invariance exp(-wc T), zero and gain set so the DC and Nyquist magnitudes equal the analog ones,
    so the shelf does not cramp near Nyquist."""
    G = 10.0 ** (float(gain_db) / 20.0)
    wc = 2.0 * math.pi * float(np.clip(fc, 1.0, 0.49 * sr))
    W = math.pi * sr
    p = math.exp(-wc / sr)
    if kind == "lowshelf":
        h0, hn = G, math.sqrt((W * W + G * G * wc * wc) / (W * W + wc * wc))
    elif kind == "highshelf":
        h0, hn = 1.0, math.sqrt((G * G * W * W + wc * wc) / (W * W + wc * wc))
    else:
        raise ValueError("kind must be lowshelf|highshelf")
    s0, d0 = h0 * (1.0 - p), hn * (1.0 + p)
    return lfilter([0.5 * (s0 + d0), 0.5 * (s0 - d0)], [1.0, -p], x, axis=0)


def shelf1(x: np.ndarray, kind: str, fc: float, sr: int = DEFAULT_SR, gain_db: float = 6.0) -> np.ndarray:
    """First-order plateau from the one-pole design rules: H = (1 - q z^-1) / (1 - p z^-1) with d = 2 pi fc / sr,
    p = 1 - d / sqrt(g), q = 1 - d sqrt(g), so the DC gain is exactly g and the gain far above fc is 1 (lowshelf);
    highshelf is the same plateau with gain 1 / g scaled by g. Cheaper and less warped near Nyquist than the bilinear
    shelves, but the corner is only accurate for fc well under sr / 4."""
    g = 10.0 ** (float(gain_db) / 20.0)
    d = 2.0 * math.pi * float(np.clip(fc, 1.0, 0.25 * sr)) / sr
    if kind == "lowshelf":
        k = g
    elif kind == "highshelf":
        k = 1.0 / g
    else:
        raise ValueError("kind must be lowshelf|highshelf")
    p, q = 1.0 - d / math.sqrt(k), 1.0 - d * math.sqrt(k)
    y = lfilter([1.0, -q], [1.0, -p], x, axis=0)
    return y if kind == "lowshelf" else y * g


@numba.njit(cache=True)
def _antisat_kernel(x, fc, sr, r, a, b):
    """TPT SVF with the antisaturator A(v) = (1 - a/b) v + a sinh(v/b) on the damping path: the feedback is
    2 D(bp), D(v) = (R - 1) v + A(v), so the small-signal damping is R (may be negative) and the sinh excess damps
    large swings. bp (1 + g^2) + 2 g D(bp) = s1 + g (x - s2), solved by Newton per sample."""
    y = np.empty((x.shape[0], 3))
    s1 = 0.0
    s2 = 0.0
    bp = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.49 * sr) / sr)
        rhs = s1 + g * (x[n] - s2)
        for _ in range(50):
            D = (r - a / b) * bp + a * math.sinh(bp / b)
            dD = (r - a / b) + (a / b) * math.cosh(bp / b)
            d = (bp * (1.0 + g * g) + 2.0 * g * D - rhs) / ((1.0 + g * g) + 2.0 * g * dD)
            bp -= d
            if abs(d) < 1e-12:
                break
        D = (r - a / b) * bp + a * math.sinh(bp / b)
        lp = g * bp + s2
        y[n, 0] = lp
        y[n, 1] = bp
        y[n, 2] = x[n] - 2.0 * D - lp
        s1 = 2.0 * bp - s1
        s2 = 2.0 * lp - s2
    return y


def svf_antisat(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR, res: float = 0.5, a: float = 1.0,
                b: float = 1.0) -> np.ndarray:
    """Antisaturated SVF (lowpass, bandpass, highpass), R = 1 - 2 res: res > 0.5 makes R negative and the filter
    self-oscillates at a level held by the antisaturator A(v) = (1 - a/b) v + a sinh(v/b)."""
    col = {"lowpass": 0, "bandpass": 1, "highpass": 2}[kind]
    def one(sig):
        fc = np.ascontiguousarray(np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape))
        return _antisat_kernel(np.ascontiguousarray(sig, dtype=np.float64), fc, float(sr), 1.0 - 2.0 * float(res),
                               float(a), float(max(b, 1e-6)))[:, col]
    return _per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _atanh_onepole_kernel(x, fc, sr):
    """ZDF one-pole with the antisaturator atanh in its feedback, y = g(x - atanh y) + s: with y = tanh u it becomes
    tanh u = (g x + s) - g u, solved by Newton on u."""
    y = np.empty_like(x)
    s = 0.0
    u = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.49 * sr) / sr)
        c = g * x[n] + s
        for _ in range(50):
            t = math.tanh(u)
            d = (t + g * u - c) / (1.0 - t * t + g)
            u -= d
            if abs(d) < 1e-12:
                break
        yn = math.tanh(u)
        s = 2.0 * yn - s
        y[n] = yn
    return y


def onepole_atanh(x: np.ndarray, cutoff, sr: int = DEFAULT_SR) -> np.ndarray:
    """One-pole low-pass whose output stays inside (-1, 1): the feedback goes through atanh (zero-delay solved)."""
    def one(sig):
        fc = np.ascontiguousarray(np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape))
        return _atanh_onepole_kernel(np.ascontiguousarray(sig, dtype=np.float64), fc, float(sr))
    return _per_channel(one, np.asarray(x, float))


def stretched_eq(x: np.ndarray, sr: int = DEFAULT_SR, center: float = 183.0, bw: float = 146.0, gain_db: float = 6.0,
                 stretch: int = 5) -> np.ndarray:
    """Stretched second-order peak: designed at sr / K, then every unit delay becomes z^-K, so the band repeats every
    sr / K (defaults: F_peak 183 Hz, -3 dB width 146 Hz, K = 5)."""
    K = max(int(stretch), 1)
    b, a = _rbj("peak", center, sr / K, q=center / max(bw, 1e-3), gain_db=gain_db)
    bs, az = np.zeros(2 * K + 1), np.zeros(2 * K + 1)
    bs[::K], az[::K] = b, a
    return lfilter(bs, az, x, axis=0)


def prewarp_bounded(fc, sr: int = DEFAULT_SR, fmax: float = 16000.0, mode: str = "min"):
    """Bounded cutoff prewarping, returned as the integrator gain g: ``min`` prewarps at wp = min(wc, wmax),
    g = tan(wp T/2) wc / wp; ``linear`` keeps tan(wc T/2) up to wmax and continues it above with the slope
    1 + tan^2(wmax T/2), so g and its derivative stay continuous."""
    mu = np.pi * np.asarray(fc, dtype=np.float64) / sr
    mm = math.pi * min(float(fmax), 0.49 * sr) / sr
    if mode == "min":
        mp = np.minimum(mu, mm)
        return np.tan(mp) * mu / np.maximum(mp, 1e-12)
    if mode == "linear":
        t = math.tan(mm)
        return np.where(mu <= mm, np.tan(np.minimum(mu, mm)), t + (mu - mm) * (1.0 + t * t))
    raise ValueError("mode must be min|linear")


def random_allpass(x: np.ndarray, sr: int = DEFAULT_SR, n: int = 8, seed: int = 0, radius: float = 0.9) -> np.ndarray:
    """Cascade of n second-order allpasses with poles at radius e^(+-jw), w drawn log-uniform between 20 and 8000 Hz:
    a = [1, -2 Re z0, |z0|^2], b = [|z0|^2, -2 Re z0, 1]. Same magnitude, scrambled phase."""
    rng = np.random.default_rng(seed)
    y = np.asarray(x, float)
    for f in np.exp(rng.uniform(math.log(20.0), math.log(8000.0), int(n))):
        z0 = radius * np.exp(2j * np.pi * f / sr)
        a = np.array([1.0, -2.0 * z0.real, abs(z0) ** 2])
        y = lfilter(a[::-1], a, y, axis=0)
    return y


def sallen_key(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR, k: float = 1.0) -> np.ndarray:
    """Linear Sallen-Key low-pass or band-pass (MS-20 type): H_LP = 1/(s^2 + (2 - k)s + 1), H_BP = s/(...), the SVF
    with 2R = 2 - k. k -> 2 self-oscillates; k is held below 1.99 here."""
    if kind not in ("lowpass", "bandpass"):
        raise ValueError("sallen_key kind must be lowpass|bandpass")
    return svf(x, kind, cutoff, sr, q=1.0 / (2.0 - min(float(k), 1.99)))


def tsk_allpass_q(k: float) -> float:
    """Q of the TSK all-pass made from a Sallen-Key core with feedback k: R = (1 - k)/(1 + k), Q = 1/(2R)."""
    r = (1.0 - float(k)) / (1.0 + float(k))
    return 1.0 / (2.0 * max(r, 1e-6))


@numba.njit(cache=True)
def _nl_onepole_kernel(x, fc, sr, mode):
    out = np.empty_like(x)
    y1 = y2 = y3 = y4 = 0.0
    for n in range(x.shape[0]):
        h = 2.0 * math.pi * min(max(fc[n], 1.0), 0.2 * sr) / (4.0 * sr)
        for _ in range(4):
            if mode == 0:
                k1 = math.tanh(x[n]) - math.tanh(y1)
                y1 += h * (math.tanh(x[n]) - math.tanh(y1 + 0.5 * h * k1))
            elif mode == 1:
                k1 = math.tanh(x[n] - y1)
                y1 += h * math.tanh(x[n] - y1 - 0.5 * h * k1)
            else:
                a = math.tanh(x[n]) - math.tanh(y1 - y2)
                b = 0.5 * (math.tanh(y1 - y2) - math.tanh(y2 - y3))
                c = 0.5 * (math.tanh(y2 - y3) - math.tanh(y3 - y4))
                d = 0.5 * (math.tanh(y3 - y4) - math.tanh(y4))
                m1, m2, m3, m4 = y1 + 0.5 * h * a, y2 + 0.5 * h * b, y3 + 0.5 * h * c, y4 + 0.5 * h * d
                y1 += h * (math.tanh(x[n]) - math.tanh(m1 - m2))
                y2 += h * 0.5 * (math.tanh(m1 - m2) - math.tanh(m2 - m3))
                y3 += h * 0.5 * (math.tanh(m2 - m3) - math.tanh(m3 - m4))
                y4 += h * 0.5 * (math.tanh(m3 - m4) - math.tanh(m4))
        out[n] = y1 if mode < 2 else y4
    return out


ZDF_BISECT_ITERS = 12


@numba.njit(cache=True)
def _zdf_onepole_kernel(x, fc, sr, hard, iters):
    out = np.empty_like(x)
    s = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 1.0), 0.45 * sr) / sr)
        if hard:
            y = (g * x[n] + s) / (1.0 + g)
            e = x[n] - y
            if e > 1.0:
                y = g + s
            elif e < -1.0:
                y = s - g
        else:
            lo, hi = s - g, s + g
            for _ in range(iters):
                y = 0.5 * (lo + hi)
                if y - g * math.tanh(x[n] - y) - s > 0.0:
                    hi = y
                else:
                    lo = y
            y = 0.5 * (lo + hi)
        u = (y - s) / g
        s = y + g * u
        out[n] = y
    return out


def onepole_nl(x: np.ndarray, cutoff, sr: int = DEFAULT_SR, kind: str = "transistor") -> np.ndarray:
    """Nonlinear one-pole low-pass: transistor y' = wc (tanh x - tanh y), ota y' = wc tanh(x - y), diode the 4-stage
    diode cascade y1' = wc (tanh x - tanh(y1 - y2)), yn' = wc/2 (tanh(y(n-1) - yn) - tanh(yn - y(n+1))), y5 = 0.
    Midpoint rule, 4 sub-steps per sample, cutoff capped at 0.2 sr.
    Zero-delay-feedback (trapezoidal, solved each sample, cutoff up to 0.45 sr): ``zdf_hard`` y' = wc clip(x - y)
    with a hard clipper at +-1, solved exactly (the linear solution, replaced by the saturated one when |x - y| > 1);
    ``zdf_ota`` y' = wc tanh(x - y), solved by bisection on [s - g, s + g] (``ZDF_BISECT_ITERS`` steps)."""
    if kind in ("zdf_hard", "zdf_ota"):
        def zone(sig):
            fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
            return _zdf_onepole_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                                       kind == "zdf_hard", ZDF_BISECT_ITERS)
        return _per_channel(zone, np.asarray(x, float))
    mode = {"transistor": 0, "ota": 1, "diode": 2}[kind]
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        return _nl_onepole_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr), mode)
    return _per_channel(one, np.asarray(x, float))


LADDER_MIX = {"lp4": (0, 0, 0, 0, 1), "lp2": (0, 0, 1, 0, 0), "lp1": (0, 1, 0, 0, 0), "hp4": (1, -4, 6, -4, 1),
              "hp2": (1, -2, 1, 0, 0), "hp1": (1, -1, 0, 0, 0), "bp4": (0, 0, 1, -2, 1), "bp2": (0, 1, -1, 0, 0),
              "notch": (1, -4, 8, -8, 4), "ap3": (1, -6, 12, -8, 0)}


@numba.njit(cache=True)
def _ladder_mix_kernel(x, fc, sr, k, drive, m):
    out = np.empty_like(x)
    s1 = s2 = s3 = s4 = 0.0
    for n in range(x.shape[0]):
        g = math.tan(math.pi * min(max(fc[n], 10.0), 0.45 * sr) / sr)
        G = g / (1.0 + g)
        S = (G * G * G * s1 + G * G * s2 + G * s3 + s4) / (1.0 + g)
        u = math.tanh(drive * (x[n] - k * S) / (1.0 + k * G ** 4)) / drive
        v = (u - s1) * G
        y1 = v + s1
        s1 = y1 + v
        v = (y1 - s2) * G
        y2 = v + s2
        s2 = y2 + v
        v = (y2 - s3) * G
        y3 = v + s3
        s3 = y3 + v
        v = (y3 - s4) * G
        y4 = v + s4
        s4 = y4 + v
        out[n] = m[0] * u + m[1] * y1 + m[2] * y2 + m[3] * y3 + m[4] * y4
    return out


def ladder_mix(x: np.ndarray, cutoff, sr: int = DEFAULT_SR, res: float = 0.5, mode="hp4", drive: float = 1.0) -> np.ndarray:
    """Linear TPT 4-pole ladder (k = 4 res, tanh on the loop input) whose output is a mix of the stage outputs
    y0..y4, Hn = (1+s)^(4-n) / (k + (1+s)^4): ``mode`` a LADDER_MIX name or five weights (hp4 = 1,-4,6,-4,1;
    bp4 = 0,0,1,-2,1; ...)."""
    m = np.asarray(LADDER_MIX[mode] if isinstance(mode, str) else mode, dtype=np.float64)
    if m.shape != (5,):
        raise ValueError("ladder_mix mode: a LADDER_MIX name or 5 weights")
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        return _ladder_mix_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                                  4.0 * float(np.clip(res, 0.0, 1.2)), float(max(drive, 1e-3)), m)
    return _per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _tbvcf_kernel(x, fco, res, dist, asym):
    out = np.empty_like(x)
    ih = 0.001
    y = y1 = y2 = 0.0
    q1 = res / (1.0 + math.sqrt(dist))
    for n in range(x.shape[0]):
        fco1 = (max(fco[n], 1.0) * 260.0 / (1.0 + 0.5 * q1)) ** 0.58
        q = q1 * fco1 * fco1 * 0.0005
        fc = fco1 / 8.0
        fdbk = q * y / (1.0 + math.exp(min(-3.0 * y, 50.0)) * asym)
        y1 += ih * ((x[n] - y1) * fc - fdbk)
        d = -2.0 * y
        ad = (d * d * d + y2) * 100.0 * dist
        y2 += ih * ((y1 - y2) * fc + ad)
        y += ih * ((y2 - y) * fc)
        out[n] = y * fc / 1000.0 * (1.0 + q1) * 3.2
    return out


def tbvcf(x: np.ndarray, cutoff=500.0, res: float = 0.8, dist: float = 2.0, asym: float = 0.5) -> np.ndarray:
    """Pinkston's TB-303 VCF (Csound tbvcf): fixed step 0.001, so the response does not depend on the sample rate;
    ``res`` 0..2, ``dist`` the diode distortion, ``asym`` the feedback asymmetry."""
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        return _tbvcf_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc),
                             float(np.clip(res, 0.0, 2.0)), float(max(dist, 0.0)), float(max(asym, 0.0)))
    return _per_channel(one, np.asarray(x, float))


FOUR_STAGE_K = 14.445


def onepole4(x: np.ndarray, cutoff: float, sr: int = DEFAULT_SR) -> np.ndarray:
    """Four cascaded one-pole low-passes as one recursion: a0 = (1-z)^4, b = 4z, -6z^2, 4z^3, -z^4, z = exp(-14.445
    fc/sr) (the 2 pi of one stage, widened so the cascade is -3 dB at fc)."""
    z = math.exp(-FOUR_STAGE_K * float(cutoff) / sr)
    return lfilter([(1.0 - z) ** 4], [1.0, -4.0 * z, 6.0 * z * z, -4.0 * z ** 3, z ** 4], np.asarray(x, float), axis=0)


def guard(x: np.ndarray, limit: float | None = None) -> np.ndarray:
    """Non-finite samples to 0; with `limit`, clip to +-limit. Leaves a finite buffer untouched."""
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -limit, limit) if limit is not None else x


def biquad_stable(a) -> bool:
    """Stability triangle of 1 + a1 z^-1 + a2 z^-2: |a2| < 1 and |a1| < 1 + a2."""
    a1, a2 = float(a[1]) / float(a[0]), float(a[2]) / float(a[0])
    return abs(a2) < 1.0 and abs(a1) < 1.0 + a2


LOG001 = math.log(0.001)


def _t60_pole(t, sr, unit):
    t = np.asarray(t, dtype=np.float64)
    k = LOG001 if unit == "t60" else -1.0
    with np.errstate(divide="ignore"):
        return np.where(np.abs(t) > 0.0, np.exp(k / (np.abs(t) * sr)), 0.0)


@numba.njit(cache=True)
def _ring_kernel(x, w, R):
    y = np.empty_like(x)
    y1 = y2 = 0.0
    for n in range(x.shape[0]):
        y0 = x[n] + 2.0 * R[n] * math.cos(w[n]) * y1 - R[n] * R[n] * y2
        y[n] = 0.5 * (y0 - y2)
        y2, y1 = y1, y0
    return y


def ringz(x: np.ndarray, freq, decay, sr: int = DEFAULT_SR, unit: str = "t60") -> np.ndarray:
    """Two-pole resonator set by its ring time: R = exp(ln 0.001 / (decay sr)) (unit="tau": exp(-1 / (decay sr))),
    y0 = x + 2R cos(w) y1 - R^2 y2, out = (y0 - y2) / 2. freq and decay scalar or per-sample; the gain grows with decay."""
    def one(sig):
        w = 2.0 * np.pi * np.clip(np.broadcast_to(np.asarray(freq, dtype=np.float64), sig.shape), 0.0, 0.5 * sr) / sr
        R = np.broadcast_to(_t60_pole(decay, sr, unit), sig.shape)
        return _ring_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(w), np.ascontiguousarray(R))
    return _per_channel(one, np.asarray(x, float))


def formlet(x: np.ndarray, freq, attack: float = 0.005, decay: float = 0.05, sr: int = DEFAULT_SR, unit: str = "t60") -> np.ndarray:
    """FOF-like resonator: ringz(decay) - ringz(attack), halved, so an impulse gives a sine with a smooth attack.
    unit="tau" (1/e times) is the Csound fofilter form."""
    return 0.5 * (ringz(x, freq, decay, sr, unit) - ringz(x, freq, attack, sr, unit))


def fofilter(x: np.ndarray, freq, attack: float = 0.003, decay: float = 0.02, sr: int = DEFAULT_SR) -> np.ndarray:
    return formlet(x, freq, attack, decay, sr, unit="tau")


def decay(x: np.ndarray, time, sr: int = DEFAULT_SR) -> np.ndarray:
    """Exponential decay of each input impulse to -60 dB in `time`: y = x + b y[n-1], b = exp(ln 0.001 / (time sr))."""
    b = float(_t60_pole(time, sr, "t60"))
    return lfilter([1.0], [1.0, -b], np.asarray(x, float), axis=0)


def decay2(x: np.ndarray, attack: float = 0.01, time: float = 1.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """decay(time) - decay(attack): impulses become smooth attack-decay envelopes."""
    return decay(x, time, sr) - decay(x, attack, sr)


@numba.njit(cache=True)
def _slop_kernel(x, k, n_dn, k_dn, n_up, k_up):
    y = np.empty_like(x)
    s = 0.0
    for i in range(x.shape[0]):
        d = x[i] - s
        if d > n_up:
            s += k * n_up + k_up * (d - n_up)
        elif d < -n_dn:
            s += -k * n_dn + k_dn * (d + n_dn)
        else:
            s += k * d
        y[i] = s
    return y


def slop(x: np.ndarray, freq: float = 1000.0, max_down: float = 1e9, down_freq: float = 0.0, max_up: float = 1e9,
         up_freq: float = 0.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Slew-limiting low-pass: one-pole at `freq` while |x - y| stays inside [-max_down, max_up], beyond it the excess
    follows a one-pole at down_freq / up_freq (0 = hard slew limit). Coefficients min(1, 2 pi f / sr)."""
    c = lambda f: min(1.0, 2.0 * math.pi * max(float(f), 0.0) / sr)
    def one(sig):
        return _slop_kernel(np.ascontiguousarray(sig, dtype=np.float64), c(freq), float(max_down), c(down_freq),
                            float(max_up), c(up_freq))
    return _per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _bob_deriv(y, u, wc, k, sat, d):
    t0 = sat * math.tanh((u - k * y[3]) / sat)
    t1 = sat * math.tanh(y[0] / sat)
    t2 = sat * math.tanh(y[1] / sat)
    t3 = sat * math.tanh(y[2] / sat)
    t4 = sat * math.tanh(y[3] / sat)
    d[0] = wc * (t0 - t1)
    d[1] = wc * (t1 - t2)
    d[2] = wc * (t2 - t3)
    d[3] = wc * (t3 - t4)


@numba.njit(cache=True)
def _bob_kernel(x, fc, sr, k, sat, os):
    out = np.empty_like(x)
    y = np.zeros(4)
    k1, k2, k3, k4, tmp = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    h = 1.0 / (sr * os)
    for n in range(x.shape[0]):
        wc = 2.0 * math.pi * min(max(fc[n], 1.0), 0.45 * sr)
        for _ in range(os):
            _bob_deriv(y, x[n], wc, k, sat, k1)
            for i in range(4):
                tmp[i] = y[i] + 0.5 * h * k1[i]
            _bob_deriv(tmp, x[n], wc, k, sat, k2)
            for i in range(4):
                tmp[i] = y[i] + 0.5 * h * k2[i]
            _bob_deriv(tmp, x[n], wc, k, sat, k3)
            for i in range(4):
                tmp[i] = y[i] + h * k3[i]
            _bob_deriv(tmp, x[n], wc, k, sat, k4)
            for i in range(4):
                y[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
            if not math.isfinite(y[3]):
                y[:] = 0.0
        out[n] = y[3]
    return out


def bob(x: np.ndarray, cutoff, sr: int = DEFAULT_SR, res: float = 0.5, sat: float = 3.0, oversample: int = 2) -> np.ndarray:
    """Moog ladder as its ODE, yi' = wc (S(y(i-1)) - S(yi)), y0 = x - 4 res y4, S(v) = sat tanh(v / sat), integrated by RK4
    with `oversample` steps per sample. res 1 is the edge of self-oscillation."""
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        return _bob_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                           4.0 * float(np.clip(res, 0.0, 1.2)), float(max(sat, 1e-3)), max(int(oversample), 1))
    return _per_channel(one, np.asarray(x, float))


def cascade(x: np.ndarray, kind: str, cutoff, order: int = 4, design: str = "butter", ripple: float = 1.0,
            sr: int = DEFAULT_SR) -> np.ndarray:
    """Butterworth or Chebyshev I low/high-pass of any order as SVF sections (+ one one-pole if odd), so the cutoff can
    be per-sample. Analog poles p = -sinh(v) sin(t) + j cosh(v) cos(t), t = (2i - 1) pi / 2N (Butterworth: sinh = cosh = 1):
    section frequency |p| (1/|p| for high-pass), Q = |p| / (2 sinh(v) sin t). Each section is prewarped on its own."""
    if kind not in ("lowpass", "highpass"):
        raise ValueError("cascade kind must be lowpass|highpass")
    N = max(int(order), 1)
    if design == "butter":
        sh, ch, gain = 1.0, 1.0, 1.0
    elif design == "cheby1":
        eps = math.sqrt(10.0 ** (float(ripple) / 10.0) - 1.0)
        v = math.asinh(1.0 / eps) / N
        sh, ch = math.sinh(v), math.cosh(v)
        gain = 1.0 / math.sqrt(1.0 + eps * eps) if N % 2 == 0 else 1.0
    else:
        raise ValueError("design must be butter|cheby1")
    fc = np.asarray(cutoff, dtype=np.float64)
    y = np.asarray(x, float)
    for i in range(1, N // 2 + 1):
        t = (2 * i - 1) * math.pi / (2 * N)
        re, im = sh * math.sin(t), ch * math.cos(t)
        w0 = math.hypot(re, im)
        y = svf(y, kind, fc * (w0 if kind == "lowpass" else 1.0 / w0), sr, w0 / (2.0 * re))
    if N % 2:
        y = onepole(y, kind, fc * (sh if kind == "lowpass" else 1.0 / sh), sr)
    return y * gain
