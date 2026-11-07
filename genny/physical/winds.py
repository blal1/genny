"""Wind-instrument acoustics: air and bore losses, tone holes, inharmonicity of a woodwind bore, brass bores and
lips, flue jets, reed laws, and three self-oscillating models: a modal clarinet, a one-mass lip on modal bore
resonances, and a two-dimensional lip on a modal trumpet bore.

Units are SI. Pressures in Pa, flows in m^3/s, impedances in Pa s/m^3 unless a function says it is normalised.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .core import SR

RHO, C0, GAMMA, PR = 1.2, 343.0, 1.4, 0.71
ETA = 1.846e-5
NU = 1.5e-5
P_ATM = 101325.0


# ------------------------------------------------------------------------------------------------ air and bore
def air(t_c: float = 20.0) -> dict:
    """Air at ``t_c`` deg C: c = 331.45 sqrt(T/T0), rho = 1.2929 T0/T, eta = 1.708e-5 (1 + 0.0029 t), gamma 1.402,
    Pr 0.71, viscous length lv 4e-8 m."""
    T0 = 273.15
    T = T0 + float(t_c)
    return dict(c=331.45 * math.sqrt(T / T0), rho=1.2929 * T0 / T, eta=1.708e-5 * (1.0 + 0.0029 * t_c),
                gamma=1.402, Pr=0.71, lv=4e-8)


def phase_velocity(f, radius: float, t_c: float = 20.0):
    """Phase velocity in a narrow pipe, c / (1 + alpha1 / rv), alpha1 1.044, rv = R sqrt(omega rho / eta): up to
    4 % slower on a low bassoon note."""
    a = air(t_c)
    rv = radius * np.sqrt(2 * np.pi * np.asarray(f, float) * a["rho"] / a["eta"])
    return a["c"] / (1.0 + 1.044 / np.maximum(rv, 1e-9))


def bore_wavenumber(f, radius: float, t_c: float = 20.0):
    """Propagation constant Gamma = j omega / c + (1 + j) 3e-5 sqrt(f) / R (1/m)."""
    f = np.asarray(f, float)
    return 1j * 2 * np.pi * f / air(t_c)["c"] + (1 + 1j) * 3e-5 * np.sqrt(f) / radius


def visco_alpha(f, radius: float, nu: float = NU, gamma: float = GAMMA, pr: float = PR):
    """Boundary-layer attenuation alpha = sqrt(2 nu omega) / (c a) [1 + (gamma - 1) / sqrt(Pr)] (1/m)."""
    w = 2 * np.pi * np.asarray(f, float)
    return np.sqrt(2 * nu * w) / (C0 * radius) * (1.0 + (gamma - 1.0) / math.sqrt(pr))


def end_nonlinear_impedance(v0: float, zc: float, cd: float = 13.0 / 9.0) -> float:
    """Quadratic loss at an open end or hole, p_s = cd rho v|v| / 2, as a resistance (4 cd / 3 pi) M Zc for a
    sinusoidal velocity of amplitude v0 (M = v0 / c); valid above about 10 m/s."""
    return 4.0 * cd / (3.0 * math.pi) * abs(v0) / C0 * zc


# ------------------------------------------------------------------------------------------------ tone holes
def lattice_cutoff(a: float, b: float, s: float, t: float, c: float = C0) -> float:
    """Open tone-hole lattice cutoff fc = (c / 2 pi) (b / a) / sqrt(2 s t): bore radius a, hole radius b, half the
    hole spacing s, effective hole height t. a 7.5, b 3, s 9, t 6 mm: 2.10 kHz."""
    return c / (2 * math.pi) * (b / a) / math.sqrt(2.0 * s * t)


def _poly(d, cs):
    return sum(c * d ** i for i, c in enumerate(cs))


def chimney_length(b: float, a: float, t: float, kind: str = "chimney") -> float:
    """Effective length te of an open hole, te / b = t / b + [1 + f g] h (delta = b / a). ``kind`` "chimney"
    (a hole with a chimney), "thick" (drilled in a thick wall: recorder, chalumeau, wooden flute) or "simple"
    (te = t + b (1.50 - 0.55 delta))."""
    d = b / a
    if kind == "simple":
        return t + b * (1.50 - 0.55 * d)
    if kind == "thick":
        f = _poly(d, (-0.044, 0.269, -1.519, 2.332, -1.897, 0.560))
        g = 1.0 - math.tanh(0.788 * t / b)
        h = _poly(d, (1.643, -0.684, 0.182, -0.394, 0.295, -0.063))
    else:
        f = _poly(d, (0.095, -0.422, 1.168, -1.808, 1.398, -0.416))
        g = 1.0 - math.tanh(0.778 * t / b)
        h = _poly(d, (1.435, 0.030, -1.566, 2.138, -1.614, 0.502))
    return t + b * (1.0 + f * g) * h


def inner_length(b: float, a: float) -> float:
    """Inner length correction ti = b (0.822 - 0.095 d - 1.566 d^2 + 2.138 d^3 - 1.640 d^4 + 0.502 d^5)."""
    return b * _poly(b / a, (0.822, -0.095, -1.566, 2.138, -1.640, 0.502))


def matching_length(b: float, a: float) -> float:
    """Matching volume length tm = b d (1 + 0.207 d^3) / 8."""
    d = b / a
    return b * d * (1.0 + 0.207 * d ** 3) / 8.0


def series_length(b: float, a: float, t: float, open_: bool = True, thick: bool = False) -> float:
    """Series (negative) length ta of a hole, ta = -b d^4 f g, for a chimney or a thick-wall hole, open or closed."""
    d = b / a
    x = t / b
    if thick:
        f, g = ((1.0 + (0.261 - 0.022 * d) * (1.0 - math.tanh(2.364 * x)), 0.302 - 0.010 * d - 0.006 * d * d) if open_
                else (1.0 - (0.956 - 0.104 * d) * (1.0 - math.tanh(2.390 * x)), 0.299 - 0.018 * d + 0.006 * d * d))
    else:
        f, g = ((1.0 + (0.333 - 0.138 * d) * (1.0 - math.tanh(2.666 * x)), 0.307 - 0.022 * d - 0.002 * d * d) if open_
                else (1.0 - (0.923 - 0.363 * d) * (1.0 - math.tanh(2.385 * x)), 0.302 - 0.019 * d + 0.003 * d * d))
    return -b * d ** 4 * f * g


def series_length_ls(b: float, a: float, t: float, open_: bool = True) -> float:
    """Series length from the finite-element fit for t of the order of b: open ta / (b d^2) = -0.35 + 0.06
    tanh(2.7 t/b), closed -0.12 - 0.17 tanh(2.4 t/b)."""
    d = b / a
    x = t / b
    k = -0.35 + 0.06 * math.tanh(2.7 * x) if open_ else -0.12 - 0.17 * math.tanh(2.4 * x)
    return k * b * d * d


def dubos_hole(r: float, R: float, h: float, f: float, open_: bool = True, c: float = C0) -> dict:
    """Side hole of radius r and chimney h on a bore of radius R at f Hz: series length la, matching hm, inner hs,
    radiation hr, total h' and the shunt impedance Zs (Pa s/m^3)."""
    d = r / R
    k = 2 * math.pi * f / c
    hm = r * d / 8.0 * (1.0 + 0.207 * d ** 3)
    hs = r * _poly(d, (0.82, -0.193, -1.09, 1.27, -0.71))
    hr = 0.82 * r / ((1.0 + (0.77 * k * r) ** 2) / (1.0 + 0.77 * k * r))
    x = 1.84 * (h + hm) / r
    tt = math.tanh(x) if open_ else 1.0 / math.tanh(max(x, 1e-9))
    la = -r * d * d / (1.78 * tt + 0.940 + 0.540 * d + 0.285 * d * d)
    zh = RHO * c / (math.pi * r * r)
    if open_:
        hp = h + hm + hs + hr
        zs = 1j * zh * k * hp
    else:
        hp = h + hm
        zs = zh * (-1j / math.tan(max(k * hp, 1e-9)) + 1j * k * hs)
    return dict(la=la, hm=hm, hs=hs, hr=hr, h_eff=hp, Zs=zs)


def keefe_hole(a: float, b: float, t: float, f: float, open_: bool = True, rc: float = 5e-4,
               t_c: float = 26.85, inner: bool = False) -> dict:
    """Single tone hole (bore radius a, hole radius b, wall t, edge radius rc) at f Hz: series length ta (negative),
    matching height th, effective length te, open-hole loss xi_e, series and shunt impedances Za, Zs.
    `inner` (closed hole only): adds the inner inertance k th (0.25 (b/th)^2 + 0.58 (b/a)^2 - pi b / (4 th)) to cot(k th)."""
    A = air(t_c)
    c, rho, eta = A["c"], A["rho"], A["eta"]
    w = 2 * math.pi * f
    k = w / c
    q = b / a
    den = (math.tanh(1.84 * t / b) if open_ else 1.0 / math.tanh(max(1.84 * t / b, 1e-9))) + 0.62 * q * q + 0.64 * q
    ta = 0.47 * b * q ** 4 / den
    th = t + b * b / (8.0 * a) * (1.0 + 0.172 * q * q)
    kt = math.tan(k * th)
    te = (kt / k + b * (1.40 - 0.58 * q * q)) / (1.0 - 0.61 * k * b * kt)
    rv = b * math.sqrt(w * rho / eta)
    alpha = w / c * (1.045 / rv + 1.080 / rv ** 2 + 0.750 / rv ** 3)
    dv = math.sqrt(2 * eta / (rho * w))
    xi = 0.25 * (k * b) ** 2 + alpha * th + 0.25 * k * dv * math.log(2 * b / rc)
    R0 = rho * c / (math.pi * a * a)
    za = R0 / (q * q) * (-1j * k * ta)
    ci = k * th * (0.25 * (b / th) ** 2 + 0.58 * q * q - math.pi * b / (4.0 * th)) if inner else 0.0
    zs = R0 / (q * q) * (1j * k * te + xi) if open_ else -1j * R0 / (q * q) * (1.0 / math.tan(max(k * th, 1e-9)) + ci)
    return dict(ta=ta, th=th, te=te, xi=xi, Za=za, Zs=zs)


def tonehole_tmatrix(Za: complex, Zs: complex) -> np.ndarray:
    """Symmetric T-matrix of a hole: [[1 + Za/2Zs, Za (1 + Za/4Zs)], [1/Zs, 1 + Za/2Zs]]."""
    e = 1.0 + Za / (2.0 * Zs)
    return np.array([[e, Za * (1.0 + Za / (4.0 * Zs))], [1.0 / Zs, e]], complex)


def key_correction(b: float, h: float, w: float, r_pad: float) -> float:
    """Extra radiation length of an open hole under a pad held at height h (hole radius b, wall w, pad radius
    r_pad): b / [3.5 (h/b)^0.8 (h/b + 3 w/b)^-0.4 + 30 (h/r_pad)^2.6]; it grows without bound as the pad closes."""
    hb = max(h, 1e-9) / b
    return b / (3.5 * hb ** 0.8 * (hb + 3.0 * w / b) ** -0.4 + 30.0 * (max(h, 1e-9) / r_pad) ** 2.6)


def closed_hole_length(b: float, a: float, t: float, thick: bool = False, finger_radius: float = 0.0) -> float:
    """Length of the closed side branch, Zs(c) = -j Z0h cot(k te): te = t + tm, or for a thick wall
    t + tm - (b^2 / 8 (a + t)) (1 + 0.207 (b / (a + t))^3); a finger of radius R removes
    t_f = (pi/6) hf (3 + (hf/b)^2), hf = R - sqrt(R^2 - b^2)."""
    te = t + matching_length(b, a)
    if thick:
        q = b / (a + t)
        te -= b * b / (8.0 * (a + t)) * (1.0 + 0.207 * q ** 3)
    if finger_radius > b:
        hf = finger_radius - math.sqrt(finger_radius ** 2 - b * b)
        te -= math.pi / 6.0 * hf * (3.0 + (hf / b) ** 2)
    return max(te, 0.0)


def hole_nonlinear_resistance(v_hole: float, b: float, a: float, c: float = C0) -> tuple[float, float]:
    """(Re Za, Re Zh) of a sharp-edged hole at acoustic velocity v_hole: Ka Mh Zc and Kh Mh Zc, Mh = v / c,
    Ka = 0.4 / delta^2, Kh = 0.5 (the hole's own Zc for Zh, the bore's for Za)."""
    M = abs(v_hole) / c
    d = b / a
    return 0.4 / (d * d) * M * RHO * c / (math.pi * a * a), 0.5 * M * RHO * c / (math.pi * b * b)


def open_hole_shunt(s: float, xi: float, xi_e: float, S_T: float, phi: float, gamma: float, kb: float) -> complex:
    """Normalised shunt of a partly open hole (opening phi): (s xi_e / (phi gamma S_T) + R) in parallel with
    gamma / ((1 - phi) s xi S_T), s = j omega in scheme units; R = (kb)^2 / 4 / S_T, the low-frequency radiation
    resistance of the hole mouth."""
    R = 0.25 * kb * kb / S_T
    z1 = s * xi_e / (max(phi, 1e-6) * gamma * S_T) + R
    if phi >= 1.0:
        return z1
    z2 = gamma / ((1.0 - phi) * s * xi * S_T)
    return z1 * z2 / (z1 + z2)


def cents(ih: float) -> float:
    """Inharmonicity IH (relative deviation of a resonance ratio from its harmonic value) in cents."""
    return ih / 5.78e-4


def ih_register_hole(Sh: float, S: float, l: float, h_eff: float, k2: float) -> float:
    return 2.0 / (3 * math.pi) * (Sh * l / (S * h_eff)) * math.cos(k2 * l) ** 2 / (k2 * l)


def ih_open_hole(r: float, R: float, h_eff: float, l_d: float, k1: float) -> tuple[float, float]:
    """(length correction dl, IH) of the first open hole, l_d the length downstream."""
    q = r * r / (R * R * h_eff)
    dl = 1.0 / (q + 1.0 / (l_d * (1.0 + (k1 * l_d) ** 2 / 3.0)))
    return dl, -16.0 / (3 * math.pi) * k1 ** 3 * l_d ** 3 / (1.0 + q * l_d) ** 2


def ih_closed_hole(v: float, S: float, la_c: float, k1: float, l: float) -> float:
    return 4.0 / math.pi * (v / S - la_c) * k1 * math.sin(2 * k1 * l) ** 2 * math.cos(2 * k1 * l)


def ih_section(alpha: float, k1: float, l: float) -> float:
    return 4.0 / (3 * math.pi) * (alpha - 1.0) * math.sin(2 * k1 * l) ** 3


def ih_cone(l_c: float, x1: float, k1: float, l: float) -> float:
    return -8.0 / (3 * math.pi) * (l_c / x1) * math.sin(2 * k1 * l) ** 3


def ih_dispersion(g1: float, g3: float) -> float:
    return (g1 - g3) / (1.0 - g1)


def ih_temperature(t_out: float, t_in: float) -> float:
    """t in kelvin."""
    return -0.0422 * (t_out - t_in) / t_out


def bell_length_correction(k: float, L_h: float, h: float = 0.085) -> float:
    return -L_h + math.atan(k * h * math.tan(L_h / h)) / k


def _cyl_tmatrix(L: float, a: float, f: float, t_c: float = 20.0) -> np.ndarray:
    G = complex(bore_wavenumber(f, a, t_c))
    zc = RHO * air(t_c)["c"] / (math.pi * a * a)
    ch, sh = np.cosh(G * L), np.sinh(G * L)
    return np.array([[ch, zc * sh], [sh / zc, ch]], complex)


def tmm_impedance(length: float, a: float, holes, freqs, opens=None, t_c: float = 20.0) -> np.ndarray:
    """Input impedance of a cylinder of radius a with side holes [(x from the input, b, t), ...] (open = ``opens``,
    default all open), unflanged end; transfer matrices with Keefe's hole."""
    freqs = np.atleast_1d(np.asarray(freqs, float))
    holes = sorted(holes)
    opens = [True] * len(holes) if opens is None else list(opens)
    zc = RHO * air(t_c)["c"] / (math.pi * a * a)
    out = np.empty(freqs.size, complex)
    for i, f in enumerate(freqs):
        k = 2 * math.pi * f / air(t_c)["c"]
        zl = zc * (0.25 * (k * a) ** 2 + 1j * k * 0.6133 * a)
        M = np.eye(2, dtype=complex)
        x0 = 0.0
        for (x, b, t), o in zip(holes, opens):
            M = M @ _cyl_tmatrix(x - x0, a, f, t_c)
            hh = keefe_hole(a, b, t, f, o)
            M = M @ tonehole_tmatrix(hh["Za"], hh["Zs"])
            x0 = x
        M = M @ _cyl_tmatrix(max(length - x0, 1e-6), a, f, t_c)
        out[i] = (M[0, 0] * zl + M[0, 1]) / (M[1, 0] * zl + M[1, 1])
    return out


def first_resonance(length, a, holes, opens=None, fmin=50.0, fmax=3000.0, kind="z") -> float:
    """Lowest |Z| peak (kind "z", a reed end) or |Y| peak (kind "y", a flue end)."""
    f = np.arange(fmin, fmax, 1.0)
    z = np.abs(tmm_impedance(length, a, holes, f, opens))
    m = z if kind == "z" else 1.0 / z
    i = np.nonzero((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]))[0] + 1
    return float(f[i[0]]) if i.size else 0.0


def optimise_holes(length, a, holes, targets, kind="z", fingerings=None):
    """Hole positions minimising sum (1200 log2(f_i / f_target_i))^2, L-BFGS-B, positions kept in order inside
    the bore. ``fingerings[i]`` is the open/closed list for target i (default: target i opens holes >= i)."""
    from scipy.optimize import minimize
    n = len(holes)
    fing = fingerings or [[j >= i for j in range(n)] for i in range(len(targets))]

    def err(xs):
        hs = [(x, b, t) for x, (_, b, t) in zip(xs, holes)]
        e = 0.0
        for fd, o in zip(targets, fing):
            f = first_resonance(length, a, hs, o, kind=kind)
            e += (1200.0 * math.log2(max(f, 1.0) / fd)) ** 2
        return e

    x0 = np.array([h[0] for h in holes])
    r = minimize(err, x0, method="L-BFGS-B", bounds=[(0.05 * length, 0.98 * length)] * n)
    return [(x, b, t) for x, (_, b, t) in zip(r.x, holes)], float(r.fun)


# ------------------------------------------------------------------------------------------------ brass bores
BRASSINESS = {
    "trombone_french": 0.81, "trombone_german": 0.76, "bass_trombone": 0.70, "trombone_american": 0.67,
    "bass_trumpet": 0.61, "baritone": 0.54, "horn_double_bb": 0.51, "saxhorn_bass": 0.45, "euphonium": 0.40,
    "wagner_tuba": 0.36, "ophicleide": 0.30, "trombone_5717": 0.70, "cornophone": 0.43, "kaiserbaryton": 0.37,
}
# Bb trumpet bore, (x m, inner diameter m): leadpipe, cylinder, bell.
TRUMPET_BORE = ((0.0, 9.0e-3), (0.24, 11.66e-3), (0.95, 11.66e-3), (1.15, 14e-3), (1.25, 20e-3), (1.31, 35e-3),
                (1.35, 60e-3), (1.37, 123e-3))
BRASS_LENGTH = {"trumpet": 1.37, "french_horn": 3.74, "trombone": 2.75, "tuba": 5.49}


def brassiness(profile=TRUMPET_BORE, L_ecl: float | None = None) -> float:
    """B = sum (l_n / L_ecl) (2 D0 / (D_n + D_n+1)), L_ecl = 1.1 L by default."""
    x = np.array([p[0] for p in profile])
    D = np.array([p[1] for p in profile])
    L_ecl = 1.1 * (x[-1] - x[0]) if L_ecl is None else L_ecl
    return float(np.sum(np.diff(x) / L_ecl * 2.0 * D[0] / (D[:-1] + D[1:])))


BRASSINESS["trumpet"] = round(brassiness(), 3)
BRASSINESS.update(french_horn=BRASSINESS["horn_double_bb"], trombone=BRASSINESS["trombone_american"],
                  tuba=BRASSINESS["wagner_tuba"])


def brassy_length(inst: str) -> float:
    """Length of plain nonlinear propagation that gives the instrument's shock: B L_ecl."""
    return BRASSINESS[inst] * 1.1 * BRASS_LENGTH[inst]


def shock_delay(p, B: float, L_ecl: float, c0: float = C0, rho: float = RHO):
    """Amplitude-dependent arrival advance of a pressure p at the bell, from dx/dt = c0 + (gamma+1)/2 u and
    u(x) = u0 D0 / D(x): -(gamma+1) / (2 c0^2) B L_ecl p / (rho c0) seconds."""
    return -(GAMMA + 1.0) / (2.0 * c0 * c0) * B * L_ecl * np.asarray(p, float) / (rho * c0)


def local_speed(u, c0: float = C0):
    """dx/dt = c0 + (gamma + 1) u / 2."""
    return c0 + 0.5 * (GAMMA + 1.0) * np.asarray(u, float)


# King tenor trombone, slide cm: (f Hz, Z MOhm, Q) of modes 4, 5, 6.
TROMBONE_MODES = np.array([
    [0, 229.5, 20, 24.0, 288.5, 22, 31.5, 343.0, 19, 29.5], [3, 225.0, 18, 26.5, 284.5, 20, 28.5, 335.0, 17, 23.5],
    [6, 220.5, 20, 26.5, 279.0, 21, 25.0, 329.0, 20, 29.0], [9, 216.0, 20, 26.5, 273.5, 21, 25.0, 322.5, 19, 26.0],
    [12, 212.0, 19, 26.0, 268.0, 21, 25.0, 317.0, 20, 33.0], [15, 208.0, 20, 25.0, 263.0, 21, 28.0, 311.0, 20, 33.5],
    [18, 205.0, 19, 23.5, 257.5, 19, 24.0, 306.5, 18, 33.5], [21, 201.5, 20, 24.0, 253.0, 20, 24.5, 302.0, 19, 26.5],
    [24, 197.0, 21, 25.0, 248.0, 21, 27.0, 297.0, 20, 26.5], [27, 194.0, 20, 25.0, 243.0, 19, 25.5, 292.0, 20, 28.5],
    [30, 190.5, 19, 23.5, 239.0, 18, 26.5, 288.5, 21, 33.5], [33, 187.5, 13, 24.0, 234.5, 14, 27.5, 284.5, 15, 27.5],
    [36, 184.0, 14, 22.5, 230.5, 14, 29.5, 280.5, 15, 28.0], [39, 180.5, 14, 21.5, 226.5, 15, 32.0, 276.0, 15, 28.5],
    [42, 177.5, 15, 25.5, 222.5, 13, 28.0, 271.0, 15, 28.5], [45, 174.5, 15, 22.0, 219.5, 13, 28.0, 267.0, 15, 27.5],
    [48, 171.0, 15, 22.0, 215.5, 13, 31.0, 262.0, 16, 26.5], [51, 168.5, 15, 22.5, 213.0, 12, 24.5, 257.5, 15, 28.0],
    [54, 165.5, 16, 23.0, 209.5, 13, 27.5, 253.5, 15, 27.5], [57, 163.0, 15, 20.5, 206.5, 12, 27.5, 249.5, 15, 28.0],
    [60, 160.5, 15, 23.0, 204.0, 12, 28.5, 245.5, 14, 27.5], [63, 158.0, 15, 22.0, 201.0, 12, 27.5, 241.5, 14, 27.0],
])
TROMBONE_MODE3 = (170.5, 21.0, 19.0)
TROMBONE_MOUTHPIECE = dict(volume=12.7e-6, f=518.0, Q=7.1)


def trombone_modes(slide_cm: float = 0.0) -> list[tuple[int, float, float, float]]:
    """[(n, f Hz, Z Pa s/m^3, Q)] of modes 2-8 at a slide extension (0-63 cm): modes 4-6 measured, the others on
    the same harmonic series (mode 3 measured at 0 cm), Z 20 MOhm and Q 24 where not measured."""
    s = float(np.clip(slide_cm, 0.0, 63.0))
    T = TROMBONE_MODES
    m = {}
    for j, n in enumerate((4, 5, 6)):
        m[n] = tuple(float(np.interp(s, T[:, 0], T[:, 1 + 3 * j + i])) for i in range(3))
    base = m[4][0] / 4.0
    out = []
    for n in range(2, 9):
        if n in m:
            f, z, q = m[n]
        elif n == 3:
            f, z, q = TROMBONE_MODE3[0] * m[4][0] / T[0, 1], TROMBONE_MODE3[1], TROMBONE_MODE3[2]
        else:
            f, z, q = n * base, 20.0, 24.0
        out.append((n, f, z * 1e6, q))
    return out


def slide_for(freq: float, n: int) -> float | None:
    """Slide extension (cm) at which mode n (4, 5 or 6) sounds ``freq``; None when out of reach."""
    j = {4: 1, 5: 4, 6: 7}[n]
    f = TROMBONE_MODES[:, j]
    if not f[-1] <= freq <= f[0]:
        return None
    return float(np.interp(freq, f[::-1], TROMBONE_MODES[::-1, 0]))


# Bb trumpet input impedance peaks (f Hz, |Zin| / Zc).
TRUMPET_ZIN = ((87.0, 43.7), (232.0, 35.2), (341.0, 46.4), (457.0, 55.7), (572.0, 48.1), (686.0, 51.6),
               (800.0, 44.5), (913.0, 28.5))
TRUMPET_ZIN_Q = 25.0
# f_peak, f_lip, f_sound, p0 Pa of the stable oscillations on modes II-VI.
TRUMPET_OPTIMAL = ((232.0, 250.0, 236.0, 2000.0), (341.0, 370.0, 342.0, 2500.0), (457.0, 490.0, 453.0, 3000.0),
                   (572.0, 580.0, 563.0, 3500.0), (686.0, 680.0, 673.0, 4000.0))
LIP2D = dict(c=340.0, rho=1.2, S_cup=2.3e-4, b=7.0e-3, d=2.0e-3, joint=(0.0, 4.0e-3), x_eq=1.0e-3, y_eq=2.0e-6,
             Q_open=3.0, Q_closed=0.5, contact=3.0)


def bore_mode_impedance(f, fA: float, QA: float, ZA: float):
    """Z = ZA / (1 + 2j QA (f - fA) / fA)."""
    f = np.asarray(f, float)
    return ZA / (1.0 + 2j * QA * (f - fA) / fA)


# ------------------------------------------------------------------------------------------------ lips
# Artificial trombone lips, embouchures 0 (loosest) .. 19: H0 m, b m, fL1, QL1, 1/muL1, fL2, QL2, 1/muL2 (m^2/kg).
LIP_TABLE_B = np.array([
    [0.70, 18, 182.0, 9.5, 0.044, 223.0, 9.2, -0.065], [0.68, 17, 182.0, 10.0, 0.041, 225.5, 10.5, -0.059],
    [0.65, 16, 186.0, 10.5, 0.042, 228.0, 9.8, -0.066], [0.63, 15, 189.5, 10.5, 0.042, 230.5, 8.0, -0.084],
    [0.61, 14, 192.5, 11.5, 0.039, 234.2, 10.2, -0.069], [0.58, 13, 196.5, 10.0, 0.044, 238.0, 12.5, -0.058],
    [0.56, 13, 202.0, 8.5, 0.053, 241.2, 11.8, -0.065], [0.53, 12, 204.5, 7.0, 0.063, 244.5, 11.0, -0.073],
    [0.51, 12, 209.5, 7.5, 0.064, 247.0, 10.8, -0.073], [0.48, 12, 213.0, 8.5, 0.059, 249.5, 10.5, -0.074],
    [0.46, 11, 216.0, 9.5, 0.056, 252.8, 13.2, -0.059], [0.44, 11, 219.0, 9.5, 0.059, 256.0, 16.0, -0.048],
    [0.41, 11, 221.5, 10.5, 0.056, 259.0, 15.8, -0.049], [0.39, 10, 222.0, 11.0, 0.057, 262.0, 15.5, -0.050],
    [0.37, 10, 225.0, 12.0, 0.056, 264.8, 14.8, -0.052], [0.34, 10, 227.5, 12.0, 0.059, 267.5, 14.0, -0.053],
    [0.32, 9, 229.5, 11.5, 0.064, 270.5, 13.0, -0.059], [0.30, 9, 232.0, 11.5, 0.067, 273.5, 12.0, -0.067],
    [0.27, 9, 234.0, 11.5, 0.070, 275.8, 12.2, -0.068], [0.25, 9, 237.0, 12.0, 0.072, 278.0, 12.5, -0.069],
])
LIP_TABLE_B[:, 0] *= 1e-3
LIP_TABLE_B[:, 1] *= 1e-3
# Slack / medium / tight: b, H0 (m); modes (fL, QL, 1/muL), None where not measured.
LIP_TABLE_1 = {
    "slack": dict(b=15e-3, H0=0.63e-3, modes=((189.0, 10.5, 0.07), (234.0, 11.0, -0.13), (338.5, 8.0, -0.30))),
    "medium": dict(b=12e-3, H0=0.53e-3, modes=((203.5, 6.0, 0.11), None, (362.0, 4.0, -0.26))),
    "tight": dict(b=11e-3, H0=0.44e-3, modes=((222.0, 9.0, 0.09), (259.5, 10.5, -0.11), (394.0, 13.0, -0.14))),
}
# Ranges measured on artificial lips, and the two earlier models they are compared with.
LIP_RANGES = {
    "cullen": dict(b=(9e-3, 18e-3), H0=(0.25e-3, 0.70e-3), fL1=(182, 237), QL1=(6, 12), inv_mu1=(0.04, 0.11),
                   fL2=(225, 278), QL2=(8, 16), inv_mu2=(-0.13, -0.05)),
    "elliot_bowsher": dict(b=(13e-3, 13e-3), H0=(0.30e-3, 0.45e-3), Q=(0.5, 0.5), inv_mu1=(0.3, 0.6),
                           inv_mu2=(-0.7, -0.5)),
    "saneyoshi": dict(b=(8.8e-3, 10.5e-3), H0=(0.07e-3, 0.17e-3), QL2=(5, 5), inv_mu2=(-0.27, -0.19)),
}
LIP_Q_HUMAN = (1.2, 1.8)
LIP_Q_LATEX = (8.0, 10.0)
LIP_Q_GLYCERINE = (4.0, 6.0)


def lip_params(embouchure: float = 0.5) -> dict:
    """Table B interpolated, embouchure 0 = loosest (index 0) .. 1 = tightest (index 19)."""
    i = float(np.clip(embouchure, 0.0, 1.0)) * 19.0
    r = [float(np.interp(i, np.arange(20), LIP_TABLE_B[:, j])) for j in range(8)]
    return dict(H0=r[0], b=r[1], fL1=r[2], QL1=r[3], inv_mu1=r[4], fL2=r[5], QL2=r[6], inv_mu2=r[7])


def lip_response(f, fL: float, QL: float, inv_mu: float):
    """Mechanical response h / p (m/Pa) of a one-mass lip: (1/mu) / (wL^2 - w^2 + j w wL / QL)."""
    w = 2 * np.pi * np.asarray(f, float)
    wL = 2 * np.pi * fL
    return inv_mu / (wL * wL - w * w + 1j * w * wL / QL)


def lip_matrix(Pm: float, fL, QL, inv_mu, b, H0, fA, QA, ZA, rho: float = RHO) -> np.ndarray:
    """State matrix [[0, I], [-K, -C]] of the one-mass lip coupled to one bore mode, linearised at mouth
    pressure Pm: state (h, psi, h', psi'), p = psi'."""
    wL, wA = 2 * math.pi * fL, 2 * math.pi * fA
    H = H0 + inv_mu * Pm / (wL * wL)
    U = math.sqrt(2.0 * Pm / rho)
    g = ZA * wA / QA
    K = np.array([[wL * wL, 0.0], [-g * b * U, wA * wA]])
    C = np.array([[wL / QL, inv_mu], [0.0, wA / QA + g * b * max(H, 0.0) / (rho * U)]])
    return np.block([[np.zeros((2, 2)), np.eye(2)], [-K, -C]])


def lip_threshold(fL, QL, inv_mu, b, H0, fA, QA, ZA, p_max: float = 20000.0, step: float = 10.0) -> tuple[float, float]:
    """(threshold mouth pressure Pa, frequency Hz at threshold): the first Pm whose linearised system has an
    eigenvalue with a positive real part. (0, 0) if none below ``p_max``."""
    for Pm in np.arange(step, p_max + step, step):
        lam = np.linalg.eigvals(lip_matrix(Pm, fL, QL, inv_mu, b, H0, fA, QA, ZA))
        k = int(np.argmax(lam.real))
        if lam[k].real > 0.0:
            return float(Pm), abs(float(lam[k].imag)) / (2 * math.pi)
    return 0.0, 0.0


@njit(cache=True)
def _lip1_kernel(pm, fs_int, os, wL, QL, inv_mu, b, H0, fm, zm, qm):
    n = pm.shape[0]
    dt = 1.0 / fs_int
    nm = fm.shape[0]
    psi = np.zeros(nm)
    dpsi = np.zeros(nm)
    h = 0.0
    v = 0.0
    out = np.zeros(n)
    for i in range(n):
        P = pm[i]
        for _ in range(os):
            p = 0.0
            for j in range(nm):
                p += dpsi[j]
            o = H0 + h
            if o < 0.0:
                o = 0.0
            d = P - p
            U = b * o * math.sqrt(2.0 * abs(d) / 1.2)
            if d < 0.0:
                U = -U
            a = inv_mu * (P - p) - wL / QL * v - wL * wL * h
            v += a * dt
            h += v * dt
            for j in range(nm):
                w = 2.0 * math.pi * fm[j]
                acc = zm[j] * w / qm[j] * U - w / qm[j] * dpsi[j] - w * w * psi[j]
                dpsi[j] += acc * dt
                psi[j] += dpsi[j] * dt
        p = 0.0
        for j in range(nm):
            p += dpsi[j]
        out[i] = p
    return out


def lip_one_mass(pm, fL: float, QL: float, inv_mu: float, b: float, H0: float, modes, sr: int = SR,
                 os: int = 8) -> np.ndarray:
    """Mouthpiece pressure (Pa) of a one-mass lip, h'' + (wL/QL) h' + wL^2 h = (1/muL) (Pm - p), with quasi-steady
    flow U = b (H0 + h) sqrt(2 |Pm - p| / rho), on bore modes [(f, Z, Q), ...] driven as
    psi'' + (w/Q) psi' + w^2 psi = (Z w / Q) U, p = sum psi'. ``pm`` per sample (Pa)."""
    pm = np.ascontiguousarray(np.asarray(pm, float))
    m = np.asarray([(f, z, q) for f, z, q in modes], float).reshape(-1, 3)
    return _lip1_kernel(pm, float(sr * os), int(os), 2 * math.pi * fL, float(QL), float(inv_mu), float(b), float(H0),
                        np.ascontiguousarray(m[:, 0]), np.ascontiguousarray(m[:, 1]), np.ascontiguousarray(m[:, 2]))


def mouthpiece_mode(volume: float = TROMBONE_MOUTHPIECE["volume"], f: float = TROMBONE_MOUTHPIECE["f"],
                    Q: float = TROMBONE_MOUTHPIECE["Q"]) -> tuple[float, float, float]:
    """The mouthpiece alone as one Helmholtz mode (f, Z, Q), Z = Q rho c^2 / (omega V)."""
    return f, Q * RHO * C0 * C0 / (2 * math.pi * f * volume), Q


def lip_buzz(dur: float, sr: int = SR, pressure: float = 3000.0, embouchure: float = 0.5,
             mouthpiece: bool = True) -> np.ndarray:
    """Buzz of lips on a mouthpiece without the instrument (the mouthpiece's Helmholtz mode as the only bore
    mode). With ``mouthpiece=False`` the lip has nothing to couple to and stays still: a single mass on a steady
    pressure does not oscillate."""
    lp = lip_params(embouchure)
    n = int(dur * sr)
    t = np.arange(n) / sr
    pm = pressure * np.clip(t / 0.02, 0.0, 1.0) * np.clip((dur - t) / 0.03, 0.0, 1.0)
    modes = [mouthpiece_mode()] if mouthpiece else [(500.0, 0.0, 7.1)]
    return lip_one_mass(pm, lp["fL1"], lp["QL1"], lp["inv_mu1"], lp["b"], lp["H0"], modes, sr)


def trombone_lip(freq: float, dur: float, sr: int = SR, vel: float = 0.8, embouchure: float = 0.5,
                 inward: bool = False) -> np.ndarray:
    """A trombone note on the one-mass lip: the slide and mode that sound ``freq`` (modes 4-6 measured; others
    on the harmonic series), lip Q and 1/mu from Table B (outward lip, or the inward one), lip frequency set
    under the note for an outward lip and above it for an inward one, mouth pressure 1.2-2 times the linear
    threshold. The lip pulls the note off the bore peak (an outward lip sharp, an inward one flat): one tuning
    pass moves the slide, all bore modes and the lip by the measured ratio. Radiated pressure (first difference
    of the mouthpiece pressure), peak 1."""
    from .waveguides import measure_f0
    lp = lip_params(embouchure)
    best = None
    for n in (4, 5, 6):
        s = slide_for(freq, n)
        if s is not None:
            best = (n, s)
            break
    if best is None:
        n0 = min(range(2, 9), key=lambda k: abs(math.log(freq / (k * 229.5 / 4.0))))
        modes = [(f * freq / (n0 * 229.5 / 4.0), z, q) for _, f, z, q in trombone_modes(0.0)]
        n, fA, ZA, QA = n0, freq, 20e6, 24.0
    else:
        n, s = best
        ms = trombone_modes(s)
        modes = [(f, z, q) for _, f, z, q in ms]
        _, fA, ZA, QA = next(m for m in ms if m[0] == n)
    if inward:
        fL, QL, im = fA * lp["fL2"] / lp["fL1"], lp["QL2"], lp["inv_mu2"]
    else:
        fL, QL, im = fA * 0.95, lp["QL1"], lp["inv_mu1"]
    Pt, _ = lip_threshold(fL, QL, im, lp["b"], lp["H0"], fA, QA, ZA)
    Pt = Pt or 2000.0
    k = int(dur * sr)
    t = np.arange(k) / sr
    pm = Pt * (1.2 + 0.8 * float(np.clip(vel, 0.0, 1.0))) * np.clip(t / 0.03, 0.0, 1.0) * np.clip((dur - t) / 0.05, 0.0, 1.0)
    p = lip_one_mass(pm, fL, QL, im, lp["b"], lp["H0"], modes, sr)
    seg = p[int(0.3 * k):int(0.8 * k)]
    mf = measure_f0(seg, freq) if seg.size > 256 and np.std(seg) > 1e-6 else 0.0
    if mf > 0 and abs(math.log2(mf / freq)) < 0.2:
        r = freq / mf
        p = lip_one_mass(pm, fL * r, QL, im, lp["b"], lp["H0"], [(f * r, z, q) for f, z, q in modes], sr)
    y = np.diff(p, prepend=0.0)
    return y / (np.max(np.abs(y)) + 1e-12)


@njit(cache=True)
def _lip2d_kernel(p0, fs_int, os, fl, m, k, Qo, Qc, kc, b, d, xj, yj, xe, ye, scup, rho, fm, zm, qm):
    n = p0.shape[0]
    dt = 1.0 / fs_int
    nm = fm.shape[0]
    psi = np.zeros(nm)
    dpsi = np.zeros(nm)
    x = xe
    y = ye
    vx = 0.0
    vy = 0.0
    U = 0.0
    out = np.zeros(n)
    sl = np.zeros(n)
    for i in range(n):
        P0 = p0[i]
        for _ in range(os):
            p = 0.0
            for j in range(nm):
                p += dpsi[j]
            dx = x - xj
            dy = y - yj
            S = 2.0 * b * y
            if S > 1e-10:
                plip = p - rho * U * abs(U) / (scup * S) * (1.0 - S / scup)
                dU = S / (rho * d) * (P0 - plip - 0.5 * rho * U * abs(U) / (S * S))
                U += dU * dt
            else:
                S = 0.0
                U = 0.0
                plip = p
            if y > 0.0:
                Q = Qo
                ky = k
            else:
                Q = Qc
                ky = kc * k
            damp = 0.5 * math.sqrt(m * k) / Q
            fx = -0.5 * k * (x - xe) + b * (P0 - p) * (-dy) - damp * vx
            fy = -0.5 * ky * (y - ye) + b * (P0 - p) * dx + b * d * plip - damp * vy
            vx += fx / (0.5 * m) * dt
            vy += fy / (0.5 * m) * dt
            x += vx * dt
            y += vy * dt
            Ul = b * (dx * vy - dy * vx)
            Ut = U + Ul
            for j in range(nm):
                w = 2.0 * math.pi * fm[j]
                acc = zm[j] * w / qm[j] * Ut - w / qm[j] * dpsi[j] - w * w * psi[j]
                dpsi[j] += acc * dt
                psi[j] += dpsi[j] * dt
        p = 0.0
        for j in range(nm):
            p += dpsi[j]
        out[i] = p
        sl[i] = S
    return out, sl


def lip2d(p0, f_lip: float, modes=TRUMPET_ZIN, Q_modes: float = TRUMPET_ZIN_Q, sr: int = SR, os: int = 8,
          y_eq: float | None = None, return_area: bool = False):
    """Two-dimensional lip (a swinging door with a hinge): (m/2) xi'' = -(sqrt(mk)/2Q) xi' - (k/2)(xi - xi_eq)
    + b (p0 - p) R90(xi - xi_joint) + b d p_lip e_y, m = 1.5 / ((2 pi)^2 f_lip), k = 1.5 f_lip, Q 3 open and 0.5
    with the lips in contact (stiffness 3k there); flow p0 - p_lip = (rho/2)(U/S)^2 + (rho d / S) U',
    p_lip - p = -rho U^2 / (S_cup S) (1 - S / S_cup), S = 2 b y; plus the flow swept by the lip
    b (xi - xi_joint) x xi'. The bore is the modal sum of ``modes`` [(f, |Zin| / Zc)] with Zc = rho c / S_cup.
    Returns the mouthpiece pressure (Pa), and the lip opening S (m^2) with ``return_area``."""
    P = LIP2D
    p0 = np.ascontiguousarray(np.asarray(p0, float))
    zc = P["rho"] * P["c"] / P["S_cup"]
    fm = np.array([f for f, _ in modes], float)
    zm = np.array([z * zc for _, z in modes], float)
    qm = np.full(fm.size, float(Q_modes))
    m = 1.5 / ((2 * math.pi) ** 2 * f_lip)
    k = 1.5 * f_lip
    p, S = _lip2d_kernel(p0, float(sr * os), int(os), float(f_lip), m, k, P["Q_open"], P["Q_closed"], P["contact"],
                         P["b"], P["d"], P["joint"][0], P["joint"][1], P["x_eq"],
                         P["y_eq"] if y_eq is None else float(y_eq), P["S_cup"], P["rho"], fm, zm, qm)
    return (p, S) if return_area else p


def lip_ratio(f: float) -> float:
    """f_lip / f_sound at the stable optimum, interpolated over modes II-VI (1.06-1.08 low, 1.01 high)."""
    fs = [o[2] for o in TRUMPET_OPTIMAL]
    r = [o[1] / o[2] for o in TRUMPET_OPTIMAL]
    return float(np.interp(f, fs, r))


def trumpet_lip2d(freq: float, dur: float, sr: int = SR, vel: float = 0.8, tune: bool = True) -> np.ndarray:
    """A trumpet note on the 2D lip: the bore is Table II transposed so that its nearest mode (II-VIII) sits on
    ``freq`` (the valves), the lip at the optimal f_lip / f_sound, blowing pressure that of the mode in
    Table III scaled by ``vel`` (2-5 kPa); one tuning pass corrects the transposition. Radiated pressure
    (first difference), peak 1."""
    from .waveguides import measure_f0
    peaks = [f for f, _ in TRUMPET_ZIN[1:]]
    j = int(np.argmin([abs(math.log(freq / f)) for f in peaks]))
    p_mode = float(np.interp(peaks[j], [o[0] for o in TRUMPET_OPTIMAL], [o[3] for o in TRUMPET_OPTIMAL]))
    p_mode *= 0.8 + 0.4 * float(np.clip(vel, 0.0, 1.0))
    k = 1.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.clip(t / 0.025, 0.0, 1.0) * np.clip((dur - t) / 0.05, 0.0, 1.0)
    for _ in range(2 if tune else 1):
        sc = freq / peaks[j] * k
        modes = [(f * sc, z) for f, z in TRUMPET_ZIN]
        p = lip2d(p_mode * env, lip_ratio(peaks[j]) * peaks[j] * sc, modes)
        if not tune:
            break
        seg = p[int(0.3 * n):int(0.8 * n)]
        mf = measure_f0(seg, freq) if seg.size > 256 and np.std(seg) > 1e-6 else 0.0
        if mf <= 0 or abs(math.log2(mf / freq)) > 0.2:
            break
        k *= freq / mf
    y = np.diff(p, prepend=0.0)
    if sr != SR:
        y = np.interp(np.arange(int(dur * sr)) * SR / sr, np.arange(n), y)
    return y / (np.max(np.abs(y)) + 1e-12)


# ------------------------------------------------------------------------------------------------ reeds
REED_ALONE = dict(dry=(1890.0, 2440.0), wet=(1530.0, 1890.0), Q1=(30.0, 50.0), K1=(2.5e7, 6.7e7), rho=(450.0, 520.0))
REED_WET_RATIO = 0.82
# Clarinet reed measured in play (impedance form and Bilbao form).
REED_PLAYED = dict(k=1.24e7, S=1.77e-4, ym=2.55e-4, pm=1671.0, lam=0.012, m=0.056, g=453.0, kc=3.7e11, fr=2368.0,
                   qr=0.030, Mr=0.056 * 1.77e-4, omega0=14881.0, sigma0=226.5, H0=2.55e-4, Sr=1.77e-4, alpha=2.0,
                   omega1=1190.0)
# Tenor saxophone reed + lip (Bilbao form for `tubes.reed_params(**SAX_REED)`).
SAX_REED = dict(Mr=0.269 * 1.77e-4, omega0=4224.0, sigma0=1858.5, Sr=1.77e-4)
# Alto saxophone (Meyer n. 8 mouthpiece): reed mass per area, f_r, damping, mouthpiece volume, missing cone volume.
SAX_ALTO = dict(mu_r=0.0475, fr=1500.0, gr_over_wr=0.6, Vm=8.1e-6, Vmissing=9.75e-6)
SAX_ALTO_HARMONICS = (138.6, 280.2, 424.6, 573.5)
SAX_SOPRANO_PEAKS = (207.0, 411.0, 606.0, 824.0, 1063.0, 1305.0, 1561.0, 1801.0, 2038.0)
SAX_GEOMETRY = dict(soprano=dict(L=0.710, half_angle=1.74), tenor=dict(L=1.490, half_angle=1.52),
                    oboe=dict(half_angle=0.71), bassoon=dict(half_angle=0.41), mouthpiece_cm3=2.25,
                    reed_volume_cm3=(1.2, 1.9), missing_cone_cm3=3.35, truncation=9.2e-3)
SAX_LATTICE_FC = dict(soprano=1340.0, tenor=760.0)
MOUTHPIECE_FACING = dict(length=19.0e-3, opening=1.05e-3)


def sax_equivalent_volume(f) -> np.ndarray:
    """Mouthpiece equivalent volume Ve = 3.85 + 7.24e-7 f^2 + 2.33e-12 f^4 cm^3, returned in m^3."""
    f = np.asarray(f, float)
    return (3.85 + 7.24e-7 * f ** 2 + 2.33e-12 * f ** 4) * 1e-6


def reed_response(f, mu_r: float, fr: float, gr: float):
    """D = (wr^2 - w^2 - j w gr) / (mu_r [(wr^2 - w^2)^2 + w^2 gr^2]) (m/Pa)."""
    w = 2 * np.pi * np.asarray(f, float)
    wr = 2 * np.pi * fr
    return (wr * wr - w * w - 1j * w * gr) / (mu_r * ((wr * wr - w * w) ** 2 + w * w * gr * gr))


def reed_recovery(t) -> np.ndarray:
    """Opening of a reed after a long squeeze, H(t)/H0: 93 % at once, 97 % with tau 8 s, the rest with tau 2900 s."""
    t = np.asarray(t, float)
    return 1.0 - 0.04 * np.exp(-t / 8.0) - 0.03 * np.exp(-t / 2900.0)


REED_HYSTERESIS = 0.04


def reed_aging_force(minutes: float) -> float:
    """Shift (N, negative) of the lines of equal frequency in the (pressure, lip force) plane after ``minutes`` of
    playing: viscoelastic constant 15 min, -0.5 N after 30 min."""
    return -0.5 * (1.0 - math.exp(-float(minutes) / 15.0)) / (1.0 - math.exp(-2.0))


def reed_flow_fit(q, dp, dpdt, u) -> tuple[float, float, float]:
    """(c1, c2, c3) of u = c1 q |dp|^1.5 + c2 q |dp|^0.5 + c3 dp' by least squares (c1 = -lambda / k,
    c2 = y_m lambda, c3 = -S / k)."""
    q, dp, dpdt, u = (np.asarray(v, float) for v in (q, dp, dpdt, u))
    A = np.stack([q * np.abs(dp) ** 1.5, q * np.abs(dp) ** 0.5, dpdt], axis=1)
    c, *_ = np.linalg.lstsq(A, u, rcond=None)
    return float(c[0]), float(c[1]), float(c[2])


def playable(y, p_mouth: float, f0: float, f_res: float) -> bool:
    """Playable when the steady amplitude is at least p_mouth / 6 and the note within 200 cents of the first
    resonance."""
    a = 0.5 * (np.max(y) - np.min(y))
    return bool(a >= p_mouth / 6.0 and abs(1200.0 * math.log2(f0 / f_res)) <= 200.0)


def clarinet_bend_cents(p_kpa: float, force_n: float, p_ref: float = 4.5, f_ref: float = 1.0) -> float:
    """Pitch deviation (cents) of a clarinet note with blowing pressure (kPa) and lip force (N) around (p_ref,
    f_ref): with a firm lip the note rises with P and F; with a light lip it first falls with P, then rises."""
    dp = float(p_kpa) - p_ref
    df = float(force_n) - f_ref
    return 6.0 * df * dp + 3.0 * dp * dp + 10.0 * df


# Buffet Crampon clarinet: bore (x mm from the reed tip, diameter mm), holes (n, l mm, r mm, h mm).
BUFFET_BORE = ((0, 0), (89, 14.8), (117.5, 14.6), (117.5, 15.6), (152.5, 14.5), (492.5, 14.5), (526, 15.1),
               (565, 16.9), (578.2, 18.4), (598, 22.2), (664, 60))
BUFFET_HOLES = ((24, 152.7, 1.4, 13.0), (23, 166.9, 2.35, 7.1), (22, 193.4, 2.8, 7.1), (21, 202.9, 2.7, 7.1),
                (20, 213.2, 2.55, 7.2), (19, 230.4, 2.4, 6.4), (18, 238.9, 3.9, 10.8), (17, 241.3, 2.35, 6.8),
                (16, 252.2, 2.45, 8.8), (15, 270.5, 2.3, 6.9), (14, 284.3, 3.3, 8.8), (13, 287.1, 2.7, 6.8),
                (12, 289.3, 2.85, 6.9), (11, 308.2, 3.95, 7.4), (10, 318.8, 2.4, 7.2), (9, 348.9, 3.55, 6.5),
                (8, 364.8, 4.05, 8.9), (7, 369.5, 3.95, 6.1), (6, 391.0, 3.85, 8.9), (5, 413.1, 4.85, 8.9),
                (4, 445.6, 5.4, 6.0), (3, 472.9, 6.1, 5.0), (2, 504.9, 5.6, 5.2), (1, 543.7, 5.8, 4.4))
BUFFET_MOUTHPIECE = dict(length=77e-3, volume=11e-6, reed_correction=10e-3, hole_shift=-12e-3, register_ratio=0.2041,
                         L=0.585, radius=7.5e-3, register=(12.5e-3, 1.55e-3), barrel=(62.74e-3, 36.76e-3, 7.17e-3))


def buffet_clarinet(open_from: int = 0, register: bool = False, **kw):
    """`tubes.WebsterTube` of the Buffet Crampon clarinet from the mouthpiece (closed reed end) to the bell, with
    its 24 holes; holes 1..open_from open (counted from the bell end: open_from 0 = all closed, low E), the
    register hole (24) open with ``register``. Returns (tube, hole_state)."""
    from ..tubes import WebsterTube
    L = (BUFFET_BORE[-1][0] - BUFFET_BORE[0][0]) * 1e-3
    xs = np.array([p[0] for p in BUFFET_BORE]) * 1e-3
    ds = np.array([max(p[1], 14.8) for p in BUFFET_BORE]) * 1e-3
    r0 = 0.5 * ds[1]
    area = [(x / L, (d / ds[1]) ** 2) for x, d in zip(xs, ds)]
    holes, state = [], []
    for n, l, r, h in BUFFET_HOLES:
        x = (l * 1e-3 + BUFFET_MOUTHPIECE["hole_shift"]) / L
        holes.append((min(max(x, 0.02), 0.98), r * 1e-3, h * 1e-3))
        state.append(1.0 if (n == 24 and register) or (n != 24 and n <= open_from) else 0.0)
    return WebsterTube(L, area=area, radius_m=r0, left="closed", right="unflanged", loss=1.0, holes=holes, **kw), state


# ------------------------------------------------------------------------------------------------ flue jets
RECORDER_ZENON = dict(L=0.265, f1=564.0, omega=(3547.0, 3547.0 * 2.023, 3547.0 * 3.066), eps=(0.0397, 0.0318, 0.0285),
                      Y=(1.38e-3, 1.21e-3, 9.81e-4), c_jd=0.4, alpha_h=0.3, c_dv=0.4, rho=1.2, alpha_vc=0.6,
                      y0=0.1e-3, W=4e-3, H=12e-3, h=1e-3, u0_range=(1.0, 56.0), end_correction=0.6 * 9.5e-3)
RECORDER_YRT304 = dict(R=12.36e-3, duct_L=72e-3, duct_H=14.7e-3, h=(1.95e-3, 1.09e-3), chamfer=(45.0, 0.7e-3),
                       W=4.83e-3, bevel=14.8)
JET_LINEAR = dict(g=0.145, alpha_p=2 / math.pi, alpha_j=0.38, Md_over_M=0.46, Md_coef=0.88)
JET_STROUHAL = dict(h=(0.02, 0.05), lips=0.03, W=(0.15, 0.25), W_over_h=(3.0, 12.0), W_over_h_opt=8.0,
                    convection=(0.3, 0.5))
CONVECTION_RANGE = (0.4, 0.92)


def jet_controls(freq: float, f_ref: float = 261.63) -> dict:
    """An expert flutist's blowing: pressure x (f/f_ref)^0.866 from 300 Pa at f_ref (jet speed x 1.35 per octave),
    lip-to-edge distance W 8 -> 4.5 mm from D4 to A5, Uj = sqrt(2 Pm / rho)."""
    Pm = 300.0 * (freq / f_ref) ** 0.866
    lo, hi = math.log(293.66), math.log(880.0)
    W = 8e-3 + (4.5e-3 - 8e-3) * float(np.clip((math.log(freq) - lo) / (hi - lo), 0.0, 1.0))
    Uj = math.sqrt(2.0 * Pm / RHO)
    return dict(Pm=Pm, W=W, Uj=Uj, St_W=freq * W / Uj)


def jet_trajectory(Re: float) -> dict:
    """Operating line of flutists: Re + 250 W/h = 3500 (1000 <= Re <= 2500, 4 <= W/h <= 10), and
    Str_h = 0.2 Str_W - 0.01 (0.07 <= Str_W <= 0.3)."""
    Re = float(np.clip(Re, 1000.0, 2500.0))
    return dict(Re=Re, W_over_h=(3500.0 - Re) / 250.0)


def strouhal_h(St_W: float) -> float:
    """Str_h = 0.2 Str_W - 0.01 on the flutists' operating line."""
    return 0.2 * float(np.clip(St_W, 0.07, 0.3)) - 0.01


def reynolds(Uj: float, h: float, nu: float = NU) -> float:
    return Uj * h / nu


def jet_delay(W: float, Uj: float, convection: float = 0.4) -> float:
    """Jet travel time W / (convection Uj)."""
    return W / (convection * Uj)


def optimal_jet_velocity(freq: float, W: float, m: int = 0, c_jd: float = 0.4) -> float:
    """u0 = f W / St_opt, St_opt = (4m + 1) c_jd / 4 (m = 0 first regime)."""
    return freq * W / ((4 * m + 1) / 4.0 * c_jd)


def jet_regime(freq: float, W: float, Uj: float, h: float | None = None, c_jd: float = 0.4) -> dict:
    """Jet regime: Wf/Uj < 0.3 jet drive, > 0.5 discrete vortices; amplification exp(2 pi W f / Uj); the
    hydrodynamic wavelength ratio lambda_h / h = (c_jd / St) W / h (critical near 9.2-10.4)."""
    St = W * freq / Uj
    r = dict(St=St, amplification=math.exp(2 * math.pi * St),
             mode="jet-drive" if St < 0.3 else ("vortex" if St > 0.5 else "transition"))
    if h:
        r["lambda_h_over_h"] = c_jd / St * W / h
    return r


def jet_transfer(omega, U0: float, W: float, d: float = 0.0, h: float = 1e-3, convection: float = 0.4,
                 g: float = JET_LINEAR["g"], mu: float | None = None):
    """Linear jet: J(omega) = -g (U0 / W) (1 / i omega) (1 - e^(mu W') e^(-i omega W' / cp)), W' = W + d,
    cp = convection U0, mu = 0.3 / h by default."""
    w = np.asarray(omega, float)
    Wp = W + d
    mu = 0.3 / h if mu is None else mu
    cp = convection * U0
    return -g * (U0 / W) / (1j * w) * (1.0 - np.exp(mu * Wp) * np.exp(-1j * w * Wp / cp))


def bickley_flux(h: float, U0: float, H: float) -> float:
    """Jet flux Qj = (2/3) h U0 H of the Bickley profile U = U0 sech^2(y / b), b = 2h/5."""
    return 2.0 / 3.0 * h * U0 * H


def jet_angle_offset(angle_deg: float) -> float:
    """Effective jet offset y/h at the labium for a jet angle: 0.1, -0.5, -0.7 at 39, 50, 65 deg."""
    return float(np.interp(angle_deg, (39.0, 50.0, 65.0), (0.1, -0.5, -0.7)))


def even_odd_db(offset_over_h: float) -> float:
    """SPL2 - SPL3 (dB) against the jet offset y/h: -1 dB at 0, 28 dB at 1.2."""
    return float(np.interp(abs(offset_over_h), (0.0, 1.2), (-1.0, 28.0)))


def radiated_power(p, r: float, rho: float = RHO, c: float = C0) -> float:
    """Upper bound of the radiated power (W) from a pressure record at distance r: mean p^2 4 pi r^2 / (rho c)."""
    p = np.asarray(p, float)
    return float(np.mean(p * p) * 4 * math.pi * r * r / (rho * c))


def jet_power(Sm: float, Uj, Pm) -> float:
    """Pneumatic power of the jet, mean Sm Uj Pm (W)."""
    return float(np.mean(Sm * np.asarray(Uj, float) * np.asarray(Pm, float)))


def source_power(Q: float, k: float, kind: str = "monopole", d: float = 0.0, rho: float = RHO, c: float = C0) -> float:
    """Power of a compact source of volume velocity amplitude Q: monopole Q^2 rho c k^2 / (8 pi) (omega^2),
    dipole of separation d Q^2 rho c k^4 d^2 / (6 pi) (omega^4), quadrupole one power of (kd)^2 further (omega^6)."""
    if kind == "monopole":
        return Q * Q * rho * c * k * k / (8 * math.pi)
    if kind == "dipole":
        return Q * Q * rho * c * k ** 4 * d * d / (6 * math.pi)
    return Q * Q * rho * c * k ** 6 * d ** 4 / (60 * math.pi)


def pipe_tone(f_edge: float, pipe_modes) -> float:
    """An edge tone next to a pipe locks onto the pipe mode nearest to it (the pipe tone)."""
    return float(min(pipe_modes, key=lambda f: abs(math.log(f / f_edge))))


# Register changes of a recorder (alto, G4 fingering): static thresholds and the dynamic delay.
REGISTER_G4 = dict(p12=311.5, p21=259.0, alpha=0.85, p_floquet=229.0, p_cross=224.0)
REGISTER_F4 = dict(p12=294.3, p21=259.0 * 294.3 / 311.5, alpha=0.85)
REGISTER_EFFECTS = dict(artificial_up=0.45, artificial_down=-0.16, model_up=0.155, model_down=-0.18,
                        flutist_hysteresis=(1.69, 3.80), novice_hysteresis=(0.27, 2.33))


def dynamic_threshold(a: float, p_stat: float, alpha: float = 0.85) -> float:
    """Threshold reached on a linear ramp of a Pa/s: Pstat + (3 a / (2 alpha))^(2/3)."""
    return p_stat + (3.0 * abs(a) / (2.0 * alpha)) ** (2.0 / 3.0)


def register_track(p, sr: float, p12: float = REGISTER_G4["p12"], p21: float = REGISTER_G4["p21"],
                   alpha: float = 0.85, skill: float = 0.0) -> np.ndarray:
    """Register (1 or 2) along a blowing pressure p (Pa, per sample). A change happens when
    zeta = integral alpha sqrt(|p - Pstat|) dt past the static threshold reaches 1, so a fast crescendo overshoots
    the static threshold by (3 a / 2 alpha)^(2/3). ``skill`` 0..1 widens the hysteresis as a flutist does,
    by 2.7 times at 1 (measured 169-380 % wider)."""
    p = np.asarray(p, float)
    w = (p12 - p21) * (1.0 + 2.7 * float(np.clip(skill, 0.0, 1.0)))
    up, dn = p21 + w, p21
    reg = np.ones(p.size, np.int64)
    r, z, dt = 1, 0.0, 1.0 / sr
    for i, x in enumerate(p):
        if r == 1:
            z = z + alpha * math.sqrt(x - up) * dt if x > up else 0.0
            if z >= 1.0:
                r, z = 2, 0.0
        else:
            z = z + alpha * math.sqrt(dn - x) * dt if x < dn else 0.0
            if z >= 1.0:
                r, z = 1, 0.0
        reg[i] = r
    return reg


@njit(cache=True)
def _auvray_kernel(u0, fs_int, os, wn, en, yn, W, h, H, cjd, alpha_h, y_off, avc_in, avc_out, rho, noise, kimp):
    n = u0.shape[0]
    dt = 1.0 / fs_int
    nm = wn.shape[0]
    v = np.zeros(nm)
    dv = np.zeros(nm)
    L = int(0.05 * fs_int) + 8
    hist = np.zeros(L)
    w = 0
    out = np.zeros(n)
    th_last = 0.0
    dp_last = 0.0
    b = 0.4 * h
    dd = 4.0 * math.sqrt(2.0 * h * W) / math.pi
    t_on = -1.0
    t = 0.0
    for i in range(n):
        U = u0[i]
        for _ in range(os):
            vac = 0.0
            for j in range(nm):
                vac += v[j]
            hist[w] = vac
            if U > 0.05:
                if t_on < 0.0:
                    t_on = t
                cp = cjd * U
                tau = W / cp
                dly = tau * fs_int
                if dly > L - 3:
                    dly = L - 3.0
                di = int(dly)
                fr = dly - di
                i0 = w - di
                if i0 < 0:
                    i0 += L
                i1 = i0 - 1
                if i1 < 0:
                    i1 += L
                vd = (1.0 - fr) * hist[i0] + fr * hist[i1]
                eta = math.exp(alpha_h / h * W) * h * vd / U
                th = math.tanh((eta - y_off) / b)
                djd = rho * dd * b * U / W * (th - th_last) / dt
                th_last = th
                ptr = 0.0
                if kimp > 0.0:
                    tt = t - t_on - tau
                    ptr = kimp * 0.5 * rho * U * U * math.exp(-0.5 * (tt / (0.25 * tau)) ** 2)
            else:
                djd = 0.0
                ptr = 0.0
                th_last = 0.0
            a = avc_in if vac >= 0.0 else avc_out
            dlos = -0.5 * rho * vac * abs(vac) / (a * a)
            dp = djd + dlos + ptr + noise[i] * 0.5 * rho * U * U
            ddp = (dp - dp_last) / dt
            dp_last = dp
            for j in range(nm):
                acc = yn[j] * ddp - en[j] * wn[j] * dv[j] - wn[j] * wn[j] * v[j]
                dv[j] += acc * dt
                v[j] += dv[j] * dt
            w += 1
            if w == L:
                w = 0
            t += dt
        s = 0.0
        for j in range(nm):
            s += v[j]
        out[i] = s
    return out


def auvray_recorder(freq: float, dur: float, sr: int = SR, u0: float | None = None, attack: float = 0.03,
                    release: float = 0.05, labium_offset: float | None = None, edge: str = "sharp",
                    stopped: bool = False, wet: bool = False, attack_impulse: float = 0.0, noise: float = 0.0,
                    seed: int = 0, os: int = 6) -> np.ndarray:
    """Flue instrument as three modes of the window admittance Y = sum j w Y_n / (w_n^2 - w^2 + j eps_n w_n w)
    (the Zen-On recorder's, scaled to ``freq``) driven by the jet: eta(W, t) = e^(alpha_i W) h v_ac(t - W/cp) / u0,
    alpha_i = 0.3 / h, cp = 0.4 u0; jet drive rho delta_d b u0 / W d/dt tanh((eta - y_off) / b), delta_d =
    4 sqrt(2 h W) / pi, b = 2h/5; vena-contracta loss -rho v|v| / (2 alpha_vc^2) (0.6; 0.6 in, 1.0 out for a
    ``edge="thick"`` edge). u0 defaults to the optimum of the first regime, 9 f W. Turbulent above Re = u0 h / nu
    of 2500 (the convection speed halves by 3000). ``stopped``: odd modes of a closed pipe; ``wet``: a wet bore,
    +1.5 % on the modes and peaks 9 % higher; ``attack_impulse`` K: a pressure pulse K rho u0^2 / 2 when the jet
    first reaches the edge; ``labium_offset`` in units of h (default 0.1 mm). Radiated pressure (d v_ac / dt), peak 1."""
    Z = RECORDER_ZENON
    W, h, H = Z["W"], Z["h"], Z["H"]
    k = freq / Z["f1"]
    if stopped:
        wn = np.array([Z["omega"][0] * k * (2 * i + 1) for i in range(3)])
    else:
        wn = np.array(Z["omega"]) * k
    en = np.array(Z["eps"], float)
    yn = np.array(Z["Y"], float)
    if wet:
        wn = wn * 1.015
        en = en / 1.09
    U0 = 9.0 * freq * W if u0 is None else float(u0)
    Re = reynolds(U0, h)
    cjd = Z["c_jd"] * (1.0 - 0.5 * float(np.clip((Re - 2500.0) / 500.0, 0.0, 1.0)))
    yoff = Z["y0"] if labium_offset is None else float(labium_offset) * h
    n = int((dur + release) * sr)
    t = np.arange(n) / sr
    env = np.clip(t / max(attack, 1e-3), 0.0, 1.0) * np.clip((dur + release - t) / max(release, 1e-3), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    turb = float(noise) + 0.05 * float(np.clip((Re - 2500.0) / 500.0, 0.0, 1.0))
    nz = turb * np.convolve(rng.standard_normal(n), np.ones(4) / 4.0, "same")
    avo = 1.0 if edge == "thick" else Z["alpha_vc"]
    v = _auvray_kernel(np.ascontiguousarray(U0 * env), float(sr * os), int(os), wn, en, yn, W, h, H, cjd, Z["alpha_h"],
                       yoff, Z["alpha_vc"], avo, Z["rho"], nz, float(attack_impulse))
    y = np.diff(v, prepend=0.0)
    return y / (np.max(np.abs(y)) + 1e-12)


# ------------------------------------------------------------------------------------------------ modal clarinet
def cubic_flow(zeta: float, gamma: float) -> tuple[float, float, float]:
    """(A, B, C) of the flow u = F(p) = zeta (1 - gamma + p) sqrt(gamma - p) expanded to third order at p = 0."""
    sg = math.sqrt(gamma)
    return (zeta * (3 * gamma - 1) / (2 * sg), -zeta * (3 * gamma + 1) / (8 * gamma * sg),
            -zeta * (gamma + 1) / (16 * gamma ** 2 * sg))


@njit(cache=True)
def _modal_clar_kernel(gam, zeta, fs_int, os, wn, yn, c_l):
    n = gam.shape[0]
    dt = 1.0 / fs_int
    nm = wn.shape[0]
    p = np.zeros(nm)
    dp = np.zeros(nm)
    p[0] = 1e-6                                                       # P = 0 is an equilibrium: a seed starts it
    out = np.zeros(n)
    for i in range(n):
        g = gam[i]
        sg = math.sqrt(max(g, 1e-6))
        A = zeta * (3 * g - 1) / (2 * sg)
        B = -zeta * (3 * g + 1) / (8 * g * sg) if g > 1e-6 else 0.0
        C = -zeta * (g + 1) / (16 * g * g * sg) if g > 1e-6 else 0.0
        for _ in range(os):
            P = 0.0
            dP = 0.0
            for j in range(nm):
                P += p[j]
                dP += dp[j]
            drive = 2.0 * c_l * (A + 2 * B * P + 3 * C * P * P) * dP
            for j in range(nm):
                acc = drive - 2.0 * yn[j] * c_l * dp[j] - wn[j] * wn[j] * p[j]
                dp[j] += acc * dt
                p[j] += dp[j] * dt
                if p[j] > 4.0:
                    p[j] = 4.0
                elif p[j] < -4.0:
                    p[j] = -4.0
        s = 0.0
        for j in range(nm):
            s += p[j]
        out[i] = s
    return out


def modal_clarinet(freq: float, dur: float, sr: int = SR, n_modes: int = 6, zeta: float = 0.35, gamma: float = 0.39,
                   eta: float = 0.02, c: float = 340.0, attack: float = 0.02, release: float = 0.05,
                   os: int = 4) -> np.ndarray:
    """Clarinet as ``n_modes`` resonators of a cylinder (l = c / 4f): p_n'' + 2 Y_n (c/l) p_n' + w_n^2 p_n =
    (2c/l)[A + 2B P + 3C P^2] P', P = sum p_n, w_j = (2j - 1) 2 pi c / 4l, Y_j = 1.3 eta sqrt(2j - 1), with the
    reed's cubic flow (A, B, C from zeta, gamma). A non-beating reed. Radiated pressure (first difference), peak 1."""
    l = c / (4.0 * freq)
    j = np.arange(1, n_modes + 1)
    wn = (2 * j - 1) * 2 * math.pi * c / (4 * l)
    yn = 1.3 * eta * np.sqrt(2 * j - 1)
    n = int((dur + release) * sr)
    t = np.arange(n) / sr
    g = gamma * np.clip(t / max(attack, 1e-3), 0.0, 1.0) * np.clip((dur + release - t) / max(release, 1e-3), 0.0, 1.0)
    p = _modal_clar_kernel(np.ascontiguousarray(g), float(zeta), float(sr * os), int(os), wn, yn, c / l)
    y = np.diff(p, prepend=0.0)
    return y / (np.max(np.abs(y)) + 1e-12)


# ------------------------------------------------------------------------------------------------ thresholds, directivity
# Lip reed: the note dies when the mouth pressure falls to these fractions of its onset threshold (three embouchures).
LIP_HYSTERESIS = (0.927, 0.964, 0.981)


def lip_sounding(p, p_threshold: float, ratio: float = LIP_HYSTERESIS[1]) -> np.ndarray:
    """Whether a lip reed sounds along a mouth pressure p: on above p_threshold, off only below ratio x p_threshold."""
    p = np.asarray(p, float)
    on = np.zeros(p.size, bool)
    s = False
    for i, x in enumerate(p):
        s = x >= p_threshold if not s else x >= ratio * p_threshold
        on[i] = s
    return on


# Frequency (Hz) below which an instrument radiates omnidirectionally.
OMNI_BELOW = dict(trumpet=500.0, oboe=500.0, clarinet=700.0, bassoon=250.0, violin=400.0, french_horn=100.0, tuba=75.0)


# Highest partial region reached at fortissimo (Hz); tuba: 3 kHz.
BRASS_TOP_FF = dict(trumpet=15000.0, french_horn=8000.0, trombone=7000.0, tuba=3000.0)


def brass_top(inst: str, vel: float, top_pp: float) -> float:
    """Spectral ceiling of a brass note: top_pp at vel 0 rising log-linearly to BRASS_TOP_FF at vel 1."""
    hi = BRASS_TOP_FF[inst]
    v = float(np.clip(vel, 0.0, 1.0))
    return float(top_pp * (hi / top_pp) ** v) if hi > top_pp else float(top_pp)


# Amplitude modulation (dB peak to peak) that a pitch vibrato makes in each band: (band low Hz, band high Hz, dB).
VIBRATO_AM = dict(trombone=((0.0, 500.0, 3.0), (500.0, 2000.0, 9.0), (2000.0, 20000.0, 15.0)),
                  flute=((0.0, 3000.0, 0.0), (3000.0, 20000.0, 15.0)),
                  bassoon=((0.0, 1400.0, 0.0), (1400.0, 1800.0, 17.5), (1800.0, 20000.0, 0.0)),
                  clarinet=((0.0, 2000.0, 0.0), (2000.0, 3000.0, 20.0), (3000.0, 20000.0, 0.0)))


def vibrato_am(inst: str, f: float) -> float:
    """Vibrato amplitude modulation (dB peak to peak) of the band holding f."""
    for lo, hi, db in VIBRATO_AM[inst]:
        if lo <= f < hi:
            return db
    return 0.0


# dB gained at 3 kHz per dB of overall level (dynamic timbre), (low, high).
DYNAMIC_FACTOR = dict(french_horn=(1.5, 3.0), trumpet=(2.2, 4.5), trombone=(2.1, 2.9), tuba=(1.5, 2.0), flute=(1.5, 2.0),
                      oboe=(1.7, 1.9), clarinet=(2.0, 2.5), bassoon=(0.6, 1.5), viola=(1.1, 1.1), cello=(1.1, 1.1),
                      double_bass=(0.6, 0.8), piano=(2.0, 2.0))
# Level at 3 kHz under the spectral maximum at forte (dB): general 20, low instruments 30-50, high oboe 12, high violin < 10.
LEVEL_3K_FORTE = dict(general=-20.0, low=(-50.0, -30.0), oboe_high=-12.0, violin_high=-10.0)


# Odd-harmonic reed instruments below beating: P_n proportional to P_1^n (Worman).
def worman_level(p1, n: int, k: float = 1.0):
    """Amplitude of harmonic n for a fundamental amplitude p1 in the non-beating regime, k p1^n."""
    return k * np.asarray(p1, float) ** int(n)


# Free reed: the coefficients F1-F3 of its first three modes, .254, .593, .777; soprano sax full cone angle 0.07 rad; truncated-cone mouthpiece condition k x1 = sqrt 3
# (670 Hz); tone-hole lattice cutoffs of the clarinet and the bassoon.
FREE_REED_F = (0.254, 0.593, 0.777)
SOPRANO_CONE_ANGLE = 0.07
CONE_KX1 = (math.sqrt(3.0), 670.0)
LATTICE_FC = dict(clarinet=1300.0, bassoon=500.0, soprano_sax=1340.0, tenor_sax=760.0)


def map_limits(zeta: float, lam: float) -> dict:
    """Limits of the clarinet iterated map (dimensionless pressure gamma, embouchure zeta, loss lam):
    chi = (1/9)(sqrt(3 + 1/zeta^2) - 1/zeta), A = chi (zeta + 1/(3 zeta)) - 1/18, f_max = gamma/2 + A,
    beating threshold gamma_b = (1 - 2 lam A)/(1 + lam), negative flow if lam > 1/(1 + 2A),
    gamma_i2 = 2 lam^2 A / (1 - lam^2)."""
    chi = (math.sqrt(3.0 + 1.0 / zeta ** 2) - 1.0 / zeta) / 9.0
    A = chi * (zeta + 1.0 / (3.0 * zeta)) - 1.0 / 18.0
    return {"A": A, "fmax_offset": A, "gamma_beat": (1.0 - 2.0 * lam * A) / (1.0 + lam),
            "negative_flow": lam > 1.0 / (1.0 + 2.0 * A), "gamma_i2": 2.0 * lam * lam * A / (1.0 - lam * lam)}


def _reed_flow(p, gamma, zeta):
    d = gamma - p
    if 1.0 - d <= 0.0:
        return 0.0
    return zeta * (1.0 - d) * math.copysign(math.sqrt(abs(d)), d)


def iterated_map(gamma: float, zeta: float, lam: float = 0.93, n: int = 200) -> np.ndarray:
    """Outgoing pressure waves p+_n of the lossy cylindrical clarinet as an iterated map (one step per 2l/c):
    the reflected wave is p- = -lam p+_{n-1}, and p+ solves p+ - p- = u(p+ + p-) with the reed flow
    u(p) = zeta (1 - gamma + p) sqrt(gamma - p) (0 when the reed is shut)."""
    from scipy.optimize import brentq
    out = np.empty(n)
    pp = 0.0
    for i in range(n):
        pm = -lam * pp
        g = lambda x: x - pm - _reed_flow(x + pm, gamma, zeta)   # noqa: E731
        pp = brentq(g, -4.0, 4.0)
        out[i] = pp
    return out


def channel_flow(dp: float, w: float, H: float, L: float, rho: float = RHO, nu: float = NU) -> float:
    """Volume flow (m^3/s) through a slit w x H, length L: the smaller of the Poiseuille flow
    w H^3 dp / (12 nu rho L) (viscous, near closure) and the Bernoulli flow w H sqrt(2 dp / rho)."""
    dp = max(float(dp), 0.0)
    return min(w * H ** 3 * dp / (12.0 * nu * rho * L), w * H * math.sqrt(2.0 * dp / rho))
