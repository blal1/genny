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
# E4); one pole at the cutoff in the loop stands for the lattice.
TONEHOLE_FC = 1500.0
CONE = dict(R=4e-3, angle=2.0)             # reed-end radius (m) and full apex angle (deg) of the conical bore
SR_DP = np.array([0.0, 1550.0, 5100.0, 8000.0])
SR_FIG = np.array([0.78e-4, 0.55e-4, 0.27e-4, 0.55e-4])
SQUEAK_REED = (1100.0, 0.1)                # f_r, q_r of the squeak gesture: the reed's own frequency, lower bound
REED_HARDNESS_REF = 2.5
# Reeds: (f_r Hz, q_r, opening H m or None, P_M Pa or None); None keeps ``mouthpiece(embouchure)``.
REEDS = {
    "guillemain": (REED_F, REED_Q, None, None),
    "typical": (2500.0, 0.2, None, None),                      # Guillemain Sec. III: "in the clarinet model"
    "squeak": (1850.0, 0.2, None, None),                       # Guillemain Sec. IV: the performance that squeaks
    # van Walstijn & Avanzini 2007: f_r 1890-1910 Hz, g 2750-3100 1/s, q = g / w_r; excursion about 0.4 mm;
    # Ka 0.75e7 N/m^3 at the 1550 Pa knee. P_M = Ka H0.
    "van_walstijn": (1900.0, 2900.0 / (2 * math.pi * 1900.0), 0.4e-3, 0.75e7 * 0.4e-3),
}
KA0 = 0.75e7
KA_DP = np.array([0.0, 1550.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0])
KA_FIG = np.array([7.27e6, 7.5e6, 8.5e6, 10.1e6, 11.7e6, 13.2e6, 15.8e6, 18.3e6, 20.8e6])


def cone_peaks(L: float, xe: float) -> tuple[float, float]:
    """First two resonance angular frequencies of a truncated cone with its apex xe before the reed end:
    c (12 pi L + 9 pi^2 xe + 16 L) / (4 L (4 L + 3 pi xe + 4 xe)) and c (28 pi L + 49 pi^2 xe + 16 L) /
    (4 L (4 L + 7 pi xe + 4 xe))."""
    w1 = C * (12 * math.pi * L + 9 * math.pi ** 2 * xe + 16 * L) / (4 * L * (4 * L + 3 * math.pi * xe + 4 * xe))
    w2 = C * (28 * math.pi * L + 49 * math.pi ** 2 * xe + 16 * L) / (4 * L * (4 * L + 7 * math.pi * xe + 4 * xe))
    return w1, w2


def loss_filter(L: float, sr: float, R: float = BORE_R, xe: float | None = None) -> tuple[float, float]:
    """(a1, b0) of the one-pole round-trip loss, exact at the first two peaks of a cylinder (eqs 15-16), or of a
    cone with apex distance ``xe`` (mean radius R (1 + 5 L / 12 xe) for the losses)."""
    if xe is None:
        w1, w2 = math.pi * C / (2 * L), 3 * math.pi * C / (2 * L)
    else:
        w1, w2 = cone_peaks(L, xe)
        R = R * (1.0 + 5.0 * L / (12.0 * xe))
    alpha = 2.0 / (R * C ** 1.5) * (math.sqrt(LV) + (CPCV - 1.0) * math.sqrt(LT))
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


def mouthpiece(embouchure: float, H: float | None = None, PM: float | None = None,
               lip_force: float = 0.0) -> tuple[float, float]:
    """(zeta, P_M in Pa) from embouchure 0 = loose .. 1 = tight, inside Dalmont's Table 1: reed opening
    H 1.0 -> 0.4 mm, P_M 80 -> 60 mbar, effective width 15 mm. ``H``, ``PM`` take a reed's own. ``lip_force`` (N) lowers
    P_M along the extinction line of Almeida et al. 2013, dF/dP about -3 cm^2: 0.6 N is 20 mbar."""
    e = float(np.clip(embouchure, 0.0, 1.0))
    H = (1.0 - 0.6 * e) * 1e-3 if H is None else H
    PM = (80.0 - 20.0 * e) * 100.0 if PM is None else PM
    PM, w = max(PM - float(lip_force) / 3e-4, 500.0), 15e-3     # the 5 mbar floor
    Zc = RHO * C / (math.pi * BORE_R ** 2)
    return Zc * w * H * math.sqrt(2.0 / (RHO * PM)), PM


@njit(cache=True)
def _kernel_ka(gamma, zeta, D, a1, b0, ac, PM, fr, g, sr, knee, kflow):
    """``_kernel`` with the stiffness depending on pressure (van Walstijn & Avanzini 2007, eqs 7-11): the closure z of
    the reed in units of H is z(n) = b1 dp(n-1) - a1r z(n-1) - a2r z(n-2), a2r = exp(-g T), a1r = -2 exp(-g T / 2)
    cos(w_r T), b1 = (1 + a1r + a2r) KA0 / Ka, w_r^2 = w_0^2 Ka / KA0 - g^2 / 4, recomputed each sample. ``knee``
    rescales the pressure axis of Ka(dp) (1 = the measured reed). ``kflow`` = Zc H / P_M > 0 adds the flow the reed
    sweeps, Sr(dp) H dz/dt, with the effective area Sr of the same measurement."""
    n = gamma.shape[0]
    p = np.zeros(n)
    u = np.zeros(n)
    s = np.zeros(n)
    z1 = 0.0
    z2 = 0.0
    w = 0.0
    y = 0.0
    di = int(D)
    fd = D - di
    T = 1.0 / sr
    w02 = (2 * math.pi * fr) ** 2
    a2r = math.exp(-g * T)
    eg = math.exp(-0.5 * g * T)
    for i in range(n):
        j = i - di
        sd = (1.0 - fd) * s[j] + fd * s[j - 1] if j >= 1 else 0.0
        w = b0 * sd + a1 * w
        y = (1.0 - ac) * w + ac * y
        pp = p[i - 1] if i >= 1 else 0.0
        dp = gamma[i] - pp
        adp = abs(dp) * PM
        rk = np.interp(adp * knee, KA_DP, KA_FIG) / KA0
        wr = math.sqrt(max(w02 * rk - 0.25 * g * g, 1.0))
        a1r = -2.0 * eg * math.cos(wr * T)
        z = (1.0 + a1r + a2r) / rk * dp - a1r * z1 - a2r * z2
        ur = kflow * np.interp(adp, SR_DP, SR_FIG) * (z - z1) * sr if kflow > 0.0 else 0.0
        z2 = z1
        z1 = z
        V = -y + ur
        o = 1.0 - z
        W = zeta[i] * o if o > 0.0 else 0.0
        d = gamma[i] - V
        sg = 1.0 if d >= 0.0 else -1.0
        u[i] = 0.5 * sg * (-W * W + W * math.sqrt(W * W + 4.0 * abs(d)))
        p[i] = u[i] + V
        u[i] += ur
        s[i] = p[i] + u[i]
    return p, u


@njit(cache=True)
def _kernel_cone(gamma, zeta, D, a1, b0, ac, b1a, a1a, a2a, K):
    """``_kernel`` on a truncated cone: its admittance is the cylinder's plus the apex inertance c / (j w xe),
    discretised bilinearly, so u = Gp p + y + ui(n-1) + K p(n-1) with K = c / (2 fe xe), Gp = 1 + K, and the
    cylinder branch carries ucyl = p + y into the loop."""
    n = gamma.shape[0]
    p = np.zeros(n)
    u = np.zeros(n)
    s = np.zeros(n)
    x1 = 0.0
    x2 = 0.0
    w = 0.0
    y = 0.0
    ui = 0.0
    di = int(D)
    fr = D - di
    Gp = 1.0 + K
    for i in range(n):
        j = i - di
        sd = (1.0 - fr) * s[j] + fr * s[j - 1] if j >= 1 else 0.0
        w = b0 * sd + a1 * w
        y = (1.0 - ac) * w + ac * y
        pp = p[i - 1] if i >= 1 else 0.0
        x = b1a * pp + a1a * x1 + a2a * x2
        x2 = x1
        x1 = x
        V = -(y + ui + K * pp) / Gp
        o = 1.0 - gamma[i] + x
        W = zeta[i] * o / Gp if o > 0.0 else 0.0
        d = gamma[i] - V
        sg = 1.0 if d >= 0.0 else -1.0
        up = 0.5 * sg * (-W * W + W * math.sqrt(W * W + 4.0 * abs(d)))
        p[i] = up + V
        u[i] = Gp * up
        uc = p[i] + y
        ui = u[i] - uc
        s[i] = p[i] + uc
    return p, u


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
             release: float = 0.06, seed: int = 0, reed: str = "guillemain", stiffening: bool = False,
             lip_force: float = 0.0, bore: str = "cylinder", cone_angle: float = CONE["angle"],
             tonehole_fc: float = TONEHOLE_FC, hardness: float | None = None, bite: float = 0.0,
             reed_age: float = 0.0, wet: bool = False, bend: bool = False, squeak: float | None = None,
             reed_flow: bool = False) -> np.ndarray:
    """One note: gamma = p_mouth / P_M = 0.5 + 0.3 blow (above the threshold P_M / 3, where the model squeaks,
    up to a beating reed; 3-6 kPa in Dalmont's mouthpiece, inside Almeida's range), tuned to ``freq`` by three
    trial runs that correct the bore length (the playing frequency sits below c / 4L: reed, losses, lattice).
    Around C5 a slow attack can settle on the twelfth (the bore's third peak): the note is then started again
    with a sharper attack, as a player would tongue it. ``reed`` is a key of ``REEDS``; "squeak" keeps the note where
    it lands. ``stiffening`` = Ka(dp), van Walstijn's stiffness (``_kernel_ka``; on another reed its knee is moved
    to P_M / 2, where the reed starts touching the lay). ``lip_force``: see ``mouthpiece``.

    ``bore`` "cone": a truncated cone of apex angle ``cone_angle`` (deg) from a 4 mm reed end (saxophone, oboe,
    bassoon), overblowing to the octave. ``tonehole_fc``: the lattice cutoff (``winds.lattice_cutoff`` gives it from
    the hole geometry). ``hardness`` (reed strength, 2.5 = as voiced): P_M and f_r rise with it. ``bite`` -1..1:
    long bite (> 0, lip further from the tip) needs 3-4 times the lip force and squeaks more; short bite (< 0) is
    lower, duller and less stable. ``reed_age`` (minutes played): a softening reed, as 0.5 N more lip force after
    30 min. ``wet``: the soaked reed's lower resonance. ``bend``: the pitch follows blowing pressure and lip force
    (``winds.clarinet_bend_cents``). ``squeak`` (s): a squeak at that time, the reed locking near its own frequency
    for 120 ms. ``reed_flow``: with ``stiffening``, the flow the moving reed sweeps."""
    if reed not in REEDS:
        raise ValueError(f"unknown reed {reed!r}; use one of {sorted(REEDS)}")
    if bore not in ("cylinder", "cone"):
        raise ValueError("bore is 'cylinder' or 'cone'")
    opts = dict(bore=bore, cone_angle=cone_angle, tonehole_fc=tonehole_fc, hardness=hardness, bite=bite,
                reed_age=reed_age, wet=wet, reed_flow=reed_flow)
    for _ in range(3):
        y, p = _note(freq, dur, sr, blow, embouchure, attack, release, seed, reed, stiffening, lip_force, **opts)
        if reed == "squeak" or bore == "cone" or not _twelfth(p[int(0.2 * sr):int(0.4 * sr)], sr, freq):
            break
        attack *= 0.5
    if squeak is not None and 0.0 <= squeak < dur:
        ys, _ = _note(freq, dur, sr, blow, embouchure, attack, release, seed, reed, stiffening, lip_force,
                      squeak_reed=True, **opts)
        t = np.arange(y.size) / sr
        w = np.clip(1.0 - np.abs(t - squeak - 0.06) / 0.06, 0.0, 1.0) ** 0.5
        y = (1.0 - w) * y + w * ys * (np.std(y) / (np.std(ys) + 1e-12))
    if bend:
        from .winds import clarinet_bend_cents
        _, PM = mouthpiece(embouchure, REEDS[reed][2], REEDS[reed][3], lip_force)
        k = 2.0 ** (clarinet_bend_cents((0.5 + 0.3 * float(np.clip(blow, 0.0, 1.0))) * PM / 1000.0, 1.0 + lip_force) / 1200.0)
        y = np.interp(np.arange(y.size) * k, np.arange(y.size), y, right=0.0)
    return y


def _note(freq, dur, sr, blow, embouchure, attack, release, seed, reed="guillemain", stiffening=False, lip_force=0.0,
          bore="cylinder", cone_angle=CONE["angle"], tonehole_fc=TONEHOLE_FC, hardness=None, bite=0.0, reed_age=0.0,
          wet=False, reed_flow=False, squeak_reed=False):
    from .winds import REED_WET_RATIO, reed_aging_force
    fr, qr, H, PM_r = REEDS[reed]
    if squeak_reed:
        fr, qr = SQUEAK_REED
    if wet:
        fr *= REED_WET_RATIO
    b = float(np.clip(bite, -1.0, 1.0))
    if b > 0:
        qr *= 1.0 - 0.4 * b
    elif b < 0:
        qr *= 1.0 - 0.5 * b
        fr *= 1.0 + 0.1 * b
    force = (float(lip_force) - reed_aging_force(reed_age)) / (1.0 + 2.5 * max(b, 0.0))
    zeta0, PM = mouthpiece(embouchure, H, PM_r, force)
    if hardness is not None:
        s = float(hardness) / REED_HARDNESS_REF
        fr *= math.sqrt(s)
        zeta0 /= math.sqrt(s)
        PM *= s
    g0 = 0.5 + 0.3 * float(np.clip(blow, 0.0, 1.0))
    b1a, a1a, a2a = reed_coefs(sr, fr, qr)
    ac = math.exp(-2 * math.pi * tonehole_fc / sr)
    rng = np.random.default_rng(seed)
    cone = bore == "cone"
    xe = CONE["R"] / math.sin(math.radians(cone_angle) / 2.0)
    knee = 1.0 if reed == "van_walstijn" else 1550.0 / (0.5 * PM)
    kflow = (RHO * C / (math.pi * BORE_R ** 2)) * (H or 0.4e-3) / PM if reed_flow else 0.0

    def run(D, n, env_g, env_z):
        L = D * C / (2 * sr)
        if cone:
            a1, b0 = loss_filter(L, sr, CONE["R"], xe)
            return _kernel_cone(env_g, env_z, D, a1, b0, ac, b1a, a1a, a2a, C / (2.0 * sr * xe))
        a1, b0 = loss_filter(L, sr)
        if stiffening:
            return _kernel_ka(env_g, env_z, D, a1, b0, ac, PM, fr, qr * 2 * math.pi * fr, sr, knee, kflow)
        return _kernel(env_g, env_z, D, a1, b0, ac, b1a, a1a, a2a)

    n = int((dur + release) * sr)
    nc = max(n, int(0.4 * sr))
    t = np.arange(nc) / sr
    # Guillemain Fig. 14: an abrupt rise of gamma, steady blowing, then gamma and zeta falling linearly to zero.
    # the rise and fall times, and 0.5 % breath turbulence on gamma.
    rise = np.clip(t / max(attack, 1e-3), 0.0, 1.0)
    fall = np.clip((dur + release - t) / max(release, 1e-3), 0.0, 1.0)
    turb = 1.0 + 0.005 * np.convolve(rng.standard_normal(nc), np.ones(48) / 48.0, "same")
    # tuning: the same attack held for 0.4 s, period measured over 0.2-0.4 s, bore length corrected, three times
    D = sr / (2.0 * freq)
    if cone:                                   # first guess: the cone length whose first peak is the note
        lo, hi = 0.02, 3.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cone_peaks(mid, xe)[0] > 2 * math.pi * freq else (lo, mid)
        D = 2.0 * sr * 0.5 * (lo + hi) / C
    for _ in range(3):
        p, _u = run(D, nc, g0 * rise * turb, np.full(nc, zeta0))
        seg = p[int(0.2 * sr):int(0.4 * sr)]
        if np.std(seg) < 1e-6:
            break
        D *= _f0(seg, sr, freq) / freq
    p, u = run(D, nc, g0 * rise * fall * turb, zeta0 * fall)
    return np.diff(p + u, prepend=0.0)[:n], p
