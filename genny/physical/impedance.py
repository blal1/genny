"""Clarinet by digital impedance: Guillemain, Kergomard & Voinier, "Real-time synthesis of clarinet-like
instruments using digital impedance models", JASA 118(1), 483-494 (2005).

Dimensionless mouthpiece pressure p, flow u and reed displacement x:
- bore: cylinder, input impedance Ze = i tan(kL) with Kirchhoff losses k = w/c - i^(3/2) (alpha c / 2) sqrt(w),
  alpha = 2 / (R c^(3/2)) (sqrt(lv) + (Cp/Cv - 1) sqrt(lt)); the round trip exp(-2ikL) is a delay D = 2 fe L / c and
  a one-pole b0 / (1 - a1 z^-1) whose modulus is exact at the first two impedance peaks (eqs 14-16), giving
  p(n) = u(n) - a1 u(n-1) - b0 u(n-D) + a1 p(n-1) - b0 p(n-D)  (eq. 17);
- reed: one mode, x'' / wr^2 + qr x' / wr + x = p, discretised with centred differences so that x(n) depends on
  p(n-1) only (eq. 19);
- flow: u = H(1 - gamma + x) zeta (1 - gamma + x) sign(gamma - p) sqrt|gamma - p| (eq. 8), solved explicitly each
  sample (eqs 26-30);
- radiated pressure p_ext = d/dt (p + u) (eq. 13), as a first difference.

Parameters:
- reed f_r = 2205 Hz, q_r = 0.3: Guillemain's own simulation (Sec. IV E). Chatziioannou & van Walstijn's fitted reed
  (k 8.66e6 Pa/m, m 0.05 kg/m^2, g 3000 1/s; Acta Acust. 2012, Table 1) gives 2095 Hz and 0.23, which locks onto
  the twelfth on E5-F#5 at soft blowing;
- zeta = Zc w H sqrt(2 / rho P_M) and gamma = p_mouth / P_M with Dalmont, Gilbert & Ollivier's measured clarinet
  mouthpiece (JASA 114, 2003, Table 1: H 0.4-1 mm, P_M 40-100 mbar, w 12-18 mm; P_M about 80 mbar loose,
  60 mbar tight) and the blowing pressures of Almeida et al. (JASA 134, 2013: notes die below about 3 kPa,
  players reach 6.5-7.5 kPa).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

C, RHO = 340.0, 1.2
LV, LT, CPCV = 4e-8, 5.6e-8, 1.4          # Guillemain Sec. II A (m K s)
BORE_R = 7e-3                              # clarinet bore radius (Guillemain)
REED_F, REED_Q = 2205.0, 0.3               # Guillemain Sec. IV E (Chatziioannou: 2095 Hz, 0.23 squeaks at E5-F#5)
# Open tone holes: above the lattice cutoff the bore resonances are "very slightly marked" and the cutoff hardly
# depends on the note (Benade, in Chaigne & Kergomard 2016, 7.8); about 1.5 kHz on a clarinet (Lefebvre 2010).
# A plain cylinder keeps its high peaks and the reed locks onto the one nearest f_r (a squeak on every note above
# E4); one pole at the cutoff in the loop stands for the lattice. UNSOURCED: the one-pole shape.
TONEHOLE_FC = 1500.0


def loss_filter(L: float, sr: float, R: float = BORE_R) -> tuple[float, float]:
    """(a1, b0) of the one-pole round-trip loss, exact at the first two peaks of a cylinder (eqs 15-16)."""
    alpha = 2.0 / (R * C ** 1.5) * (math.sqrt(LV) + (CPCV - 1.0) * math.sqrt(LT))
    w1, w2 = math.pi * C / (2 * L), 3 * math.pi * C / (2 * L)
    F1 = math.exp(-2.0 * alpha * C * math.sqrt(w1 / 2.0) * L)
    F2 = math.exp(-2.0 * alpha * C * math.sqrt(w2 / 2.0) * L)
    c1, c2 = math.cos(w1 / sr), math.cos(w2 / sr)
    A12, F12 = F1 * c1 - F2 * c2, F1 - F2
    r = A12 - math.sqrt(max(A12 * A12 - F12 * F12, 0.0))
    return r / F12, math.sqrt(2.0 * F1 * F2 * (c1 - c2) * r) / F12


def reed_coefs(sr: float, fr: float = REED_F, qr: float = REED_Q) -> tuple[float, float, float]:
    """(b1a, a1a, a2a) of x(n) = b1a p(n-1) + a1a x(n-1) + a2a x(n-2) (eq. 19)."""
    wr = 2 * math.pi * fr
    k2, k1 = sr * sr / (wr * wr), sr * qr / (2 * wr)
    a0 = k2 + k1
    return 1.0 / a0, (2 * k2 - 1.0) / a0, (k1 - k2) / a0


def mouthpiece(embouchure: float) -> tuple[float, float]:
    """(zeta, P_M in Pa) from embouchure 0 = loose .. 1 = tight, inside Dalmont's Table 1: reed opening
    H 1.0 -> 0.4 mm, P_M 80 -> 60 mbar, effective width 15 mm."""
    e = float(np.clip(embouchure, 0.0, 1.0))
    H, PM, w = (1.0 - 0.6 * e) * 1e-3, (80.0 - 20.0 * e) * 100.0, 15e-3
    Zc = RHO * C / (math.pi * BORE_R ** 2)
    return Zc * w * H * math.sqrt(2.0 / (RHO * PM)), PM


@njit(cache=True)
def _kernel(gamma, zeta, D, a1, b0, ac, b1a, a1a, a2a):
    """Ze = (1 - G) / (1 + G) with G = b0 z^-D / (1 - a1 z^-1) (eq. 12) gives p = u - G (p + u): eq. 17 with the
    loop filter written once, so ``ac`` can add the tone-hole cutoff pole (1 - ac) / (1 - ac z^-1) to it."""
    n = gamma.shape[0]
    p = np.zeros(n)
    u = np.zeros(n)
    s = np.zeros(n)
    x1 = 0.0
    x2 = 0.0
    w = 0.0
    y = 0.0
    di = int(D)
    fr = D - di
    for i in range(n):
        j = i - di
        sd = (1.0 - fr) * s[j] + fr * s[j - 1] if j >= 1 else 0.0   # linear fractional delay
        w = b0 * sd + a1 * w
        y = (1.0 - ac) * w + ac * y
        pp = p[i - 1] if i >= 1 else 0.0
        x = b1a * pp + a1a * x1 + a2a * x2
        x2 = x1
        x1 = x
        V = -y
        o = 1.0 - gamma[i] + x
        W = zeta[i] * o if o > 0.0 else 0.0
        d = gamma[i] - V
        sg = 1.0 if d >= 0.0 else -1.0
        u[i] = 0.5 * sg * (-W * W + W * math.sqrt(W * W + 4.0 * abs(d)))
        p[i] = u[i] + V
        s[i] = p[i] + u[i]
    return p, u


def _f0(y: np.ndarray, sr: float, f_guess: float) -> float:
    y = y - y.mean()
    r = np.correlate(y, y, "full")[y.size - 1:]
    lo, hi = int(sr / (f_guess * 1.25)), int(sr / (f_guess * 0.8)) + 1     # +-25 %: the bore corrections stay under 20 %
    k = lo + int(np.argmax(r[lo:hi]))
    a, b, c = r[k - 1], r[k], r[k + 1]
    return sr / (k + 0.5 * (a - c) / (a - 2 * b + c + 1e-30))


def _twelfth(p: np.ndarray, sr: float, freq: float) -> bool:
    """True when the note has gone to the twelfth (3 f, the clarion regime): almost nothing left at f."""
    S = np.abs(np.fft.rfft(p * np.hanning(p.size), 1 << 15))
    f = np.fft.rfftfreq(1 << 15, 1.0 / sr)
    return S[(f > 0.8 * freq) & (f < 1.2 * freq)].max() < 0.1 * S.max()


def clarinet(freq: float, dur: float, sr: float, *, blow: float = 0.6, embouchure: float = 0.5, attack: float = 0.01,
             release: float = 0.06, seed: int = 0) -> np.ndarray:
    """One note: gamma = p_mouth / P_M = 0.5 + 0.3 blow (above the threshold P_M / 3, where the model squeaks,
    up to a beating reed; 3-6 kPa in Dalmont's mouthpiece, inside Almeida's range), tuned to ``freq`` by three
    trial runs that correct the bore length (the playing frequency sits below c / 4L: reed, losses, lattice).
    Around C5 a slow attack can settle on the twelfth (the bore's third peak): the note is then started again
    with a sharper attack, as a player would tongue it."""
    for _ in range(3):
        y, p = _note(freq, dur, sr, blow, embouchure, attack, release, seed)
        if not _twelfth(p[int(0.2 * sr):int(0.4 * sr)], sr, freq):
            break
        attack *= 0.5
    return y


def _note(freq, dur, sr, blow, embouchure, attack, release, seed):
    zeta0, PM = mouthpiece(embouchure)
    g0 = 0.5 + 0.3 * float(np.clip(blow, 0.0, 1.0))
    b1a, a1a, a2a = reed_coefs(sr)
    ac = math.exp(-2 * math.pi * TONEHOLE_FC / sr)
    rng = np.random.default_rng(seed)

    def run(D, n, env_g, env_z):
        a1, b0 = loss_filter(D * C / (2 * sr), sr)
        return _kernel(env_g, env_z, D, a1, b0, ac, b1a, a1a, a2a)

    n = int((dur + release) * sr)
    nc = max(n, int(0.4 * sr))
    t = np.arange(nc) / sr
    # Guillemain Fig. 14: an abrupt rise of gamma, steady blowing, then gamma and zeta falling linearly to zero.
    # UNSOURCED: the rise and fall times, and 0.5 % breath turbulence on gamma.
    rise = np.clip(t / max(attack, 1e-3), 0.0, 1.0)
    fall = np.clip((dur + release - t) / max(release, 1e-3), 0.0, 1.0)
    turb = 1.0 + 0.005 * np.convolve(rng.standard_normal(nc), np.ones(48) / 48.0, "same")
    # tuning: the same attack held for 0.4 s, period measured over 0.2-0.4 s, bore length corrected, three times
    D = sr / (2.0 * freq)
    for _ in range(3):
        p, _u = run(D, nc, g0 * rise * turb, np.full(nc, zeta0))
        seg = p[int(0.2 * sr):int(0.4 * sr)]
        if np.std(seg) < 1e-6:
            break
        D *= _f0(seg, sr, freq) / freq
    p, u = run(D, nc, g0 * rise * fall * turb, zeta0 * fall)
    return np.diff(p + u, prepend=0.0)[:n], p
