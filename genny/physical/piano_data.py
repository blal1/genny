"""Steinway D string and hammer parameters, note by note (C1-B7), from Chabassier & Durufle, "Physical
parameters for piano modeling", INRIA RT-0425 (2012), appendix A (wrapped strings, real lengths;
hal-00688679v2). Strings are homogenised steel: E = 2.02e11 Pa (their section 3.2), so the stiffness is
B = pi^3 E d^4 / (64 T L^2). That B is smallest at the bass/tenor break (5e-5 near C2) and climbs to 8e-3
at C7, unlike a single power of f.
"""
from __future__ import annotations

import math

import numpy as np

E_STEEL = 2.02e11

CHABASSIER_D = (  # midi, f0 Hz, L m, d mm, rho kg/m3, T0 N, felt p, hammer g, K_H N/m^p
    (24, 32.78, 2.007, 1.48, 57787, 1722, 2.31, 10.95, 7.16e+07),
    (25, 34.73, 1.997, 1.505, 52229, 1788, 2.305, 10.89, 8.091e+07),
    (26, 36.79, 1.981, 1.506, 47494, 1798, 2.3, 10.83, 9.143e+07),
    (27, 38.98, 1.965, 1.492, 43195, 1773, 2.295, 10.76, 1.033e+08),
    (28, 41.3, 1.938, 1.46, 39737, 1703, 2.291, 10.7, 1.168e+08),
    (29, 43.75, 1.911, 1.419, 36571, 1618, 2.287, 10.64, 1.319e+08),
    (30, 46.35, 1.879, 1.37, 33851, 1515, 2.284, 10.58, 1.491e+08),
    (31, 49.11, 1.842, 1.316, 31521, 1404, 2.282, 10.51, 1.685e+08),
    (32, 52.03, 1.805, 1.262, 29376, 1297, 2.28, 10.45, 1.904e+08),
    (33, 55.13, 1.762, 1.207, 27588, 1190, 2.278, 10.39, 2.152e+08),
    (34, 58.4, 1.709, 1.148, 26243, 1083, 2.277, 10.33, 2.431e+08),
    (35, 61.88, 1.655, 1.096, 25043, 991, 2.276, 10.26, 2.748e+08),
    (36, 65.56, 1.602, 1.051, 23919, 915, 2.276, 10.2, 3.105e+08),
    (37, 69.45, 1.548, 1.012, 22925, 853, 2.276, 10.14, 3.509e+08),
    (38, 73.58, 1.495, 0.982, 21997, 807, 2.277, 10.08, 3.965e+08),
    (39, 77.96, 1.442, 0.96, 21160, 774, 2.278, 10.02, 4.481e+08),
    (40, 82.59, 1.378, 0.937, 20738, 741, 2.28, 9.95, 5.063e+08),
    (41, 87.51, 1.837, 1.117, 7887, 799, 2.282, 9.89, 5.722e+08),
    (42, 92.71, 1.757, 1.108, 7734, 791, 2.285, 9.83, 6.466e+08),
    (43, 98.22, 1.66, 1.091, 7772, 773, 2.288, 9.77, 7.307e+08),
    (44, 104.06, 1.591, 1.095, 7589, 783, 2.292, 9.7, 8.257e+08),
    (45, 110.25, 1.482, 1.071, 7850, 754, 2.296, 9.64, 9.331e+08),
    (46, 116.81, 1.403, 1.067, 7850, 754, 2.301, 9.58, 1.054e+09),
    (47, 123.75, 1.329, 1.065, 7850, 756, 2.306, 9.52, 1.192e+09),
    (48, 131.11, 1.259, 1.063, 7850, 759, 2.312, 9.45, 1.347e+09),
    (49, 138.9, 1.192, 1.061, 7850, 762, 2.318, 9.39, 1.522e+09),
    (50, 147.17, 1.129, 1.059, 7850, 764, 2.325, 9.33, 1.72e+09),
    (51, 155.91, 1.07, 1.057, 7850, 766, 2.332, 9.27, 1.943e+09),
    (52, 165.19, 1.013, 1.053, 7850, 767, 2.339, 9.2, 2.196e+09),
    (53, 175.01, 0.96, 1.049, 7850, 766, 2.347, 9.14, 2.481e+09),
    (54, 185.41, 0.909, 1.045, 7850, 765, 2.356, 9.08, 2.804e+09),
    (55, 196.45, 0.861, 1.039, 7850, 763, 2.365, 9.02, 3.169e+09),
    (56, 208.12, 0.816, 1.033, 7850, 759, 2.375, 8.96, 3.581e+09),
    (57, 220.49, 0.773, 1.027, 7850, 755, 2.385, 8.89, 4.047e+09),
    (58, 233.62, 0.732, 1.02, 7850, 751, 2.395, 8.83, 4.573e+09),
    (59, 247.52, 0.694, 1.013, 7850, 746, 2.406, 8.77, 5.168e+09),
    (60, 262.22, 0.657, 1.006, 7850, 741, 2.418, 8.71, 5.84e+09),
    (61, 277.81, 0.622, 0.999, 7850, 735, 2.43, 8.64, 6.599e+09),
    (62, 294.34, 0.59, 0.991, 7850, 730, 2.442, 8.58, 7.457e+09),
    (63, 311.82, 0.559, 0.984, 7850, 725, 2.455, 8.52, 8.427e+09),
    (64, 330.36, 0.529, 0.977, 7850, 720, 2.468, 8.46, 9.523e+09),
    (65, 350.05, 0.501, 0.97, 7850, 715, 2.482, 8.39, 1.076e+10),
    (66, 370.8, 0.475, 0.964, 7850, 711, 2.497, 8.33, 1.216e+10),
    (67, 392.87, 0.45, 0.958, 7850, 707, 2.512, 8.27, 1.374e+10),
    (68, 416.26, 0.426, 0.952, 7850, 704, 2.527, 8.21, 1.553e+10),
    (69, 441.0, 0.404, 0.947, 7850, 701, 2.543, 8.14, 1.755e+10),
    (70, 467.19, 0.383, 0.941, 7850, 699, 2.559, 8.08, 1.983e+10),
    (71, 494.96, 0.363, 0.937, 7850, 697, 2.576, 8.02, 2.241e+10),
    (72, 524.48, 0.344, 0.932, 7850, 696, 2.593, 7.96, 2.532e+10),
    (73, 555.6, 0.326, 0.928, 7850, 695, 2.611, 7.9, 2.862e+10),
    (74, 588.66, 0.308, 0.924, 7850, 694, 2.629, 7.83, 3.234e+10),
    (75, 623.69, 0.292, 0.921, 7850, 694, 2.648, 7.77, 3.655e+10),
    (76, 660.69, 0.277, 0.917, 7850, 694, 2.668, 7.71, 4.13e+10),
    (77, 699.91, 0.262, 0.914, 7850, 694, 2.687, 7.65, 4.667e+10),
    (78, 741.65, 0.249, 0.91, 7850, 695, 2.708, 7.58, 5.274e+10),
    (79, 785.63, 0.236, 0.907, 7850, 695, 2.728, 7.52, 5.96e+10),
    (80, 832.54, 0.223, 0.904, 7850, 696, 2.749, 7.46, 6.735e+10),
    (81, 882.06, 0.211, 0.901, 7850, 697, 2.771, 7.4, 7.611e+10),
    (82, 934.6, 0.2, 0.898, 7850, 697, 2.793, 7.33, 8.6e+10),
    (83, 990.2, 0.19, 0.894, 7850, 697, 2.816, 7.27, 9.719e+10),
    (84, 1048.83, 0.18, 0.891, 7850, 697, 2.839, 7.21, 1.098e+11),
    (85, 1111.06, 0.171, 0.887, 7850, 697, 2.863, 7.15, 1.241e+11),
    (86, 1177.66, 0.162, 0.883, 7850, 697, 2.887, 7.09, 1.402e+11),
    (87, 1247.25, 0.153, 0.879, 7850, 696, 2.912, 7.02, 1.585e+11),
    (88, 1321.31, 0.145, 0.875, 7850, 695, 2.937, 6.96, 1.791e+11),
    (89, 1399.97, 0.138, 0.87, 7850, 693, 2.962, 6.9, 2.024e+11),
    (90, 1483.31, 0.13, 0.866, 7850, 691, 2.988, 6.84, 2.287e+11),
    (91, 1571.35, 0.124, 0.86, 7850, 689, 3.015, 6.77, 2.585e+11),
    (92, 1665.41, 0.117, 0.855, 7850, 686, 3.042, 6.71, 2.921e+11),
    (93, 1764.21, 0.111, 0.85, 7850, 683, 3.069, 6.65, 3.301e+11),
    (94, 1869.22, 0.105, 0.844, 7850, 679, 3.097, 6.59, 3.73e+11),
    (95, 1980.57, 0.1, 0.837, 7850, 674, 3.126, 6.52, 4.215e+11),
    (96, 2098.32, 0.095, 0.831, 7850, 670, 3.155, 6.46, 4.763e+11),
    (97, 2222.39, 0.09, 0.824, 7850, 664, 3.184, 6.4, 5.382e+11),
    (98, 2355.34, 0.085, 0.817, 7850, 659, 3.214, 6.34, 6.082e+11),
    (99, 2494.63, 0.081, 0.81, 7850, 652, 3.245, 6.27, 6.873e+11),
    (100, 2643.17, 0.076, 0.802, 7850, 646, 3.276, 6.21, 7.767e+11),
    (101, 2801.35, 0.072, 0.795, 7850, 639, 3.307, 6.15, 8.777e+11),
    (102, 2965.15, 0.069, 0.787, 7850, 631, 3.339, 6.09, 9.919e+11),
    (103, 3142.89, 0.065, 0.778, 7850, 623, 3.371, 6.03, 1.121e+12),
    (104, 3330.76, 0.062, 0.77, 7850, 615, 3.404, 5.96, 1.267e+12),
    (105, 3528.57, 0.058, 0.761, 7850, 606, 3.438, 5.9, 1.431e+12),
    (106, 3735.94, 0.055, 0.752, 7850, 598, 3.471, 5.84, 1.617e+12),
    (107, 3959.63, 0.052, 0.743, 7850, 588, 3.506, 5.78, 1.828e+12),
)

_MIDI = np.array([r[0] for r in CHABASSIER_D], float)
_B = np.array([math.pi ** 3 * E_STEEL * (r[3] * 1e-3) ** 4 / (64.0 * r[5] * r[2] ** 2) for r in CHABASSIER_D])


def steinway_B(midi) -> np.ndarray | float:
    """Inharmonicity coefficient of the Steinway D string at MIDI note ``midi`` (log-interpolated; held at
    the C1 / B7 values outside the table)."""
    out = np.exp(np.interp(np.asarray(midi, float), _MIDI, np.log(_B)))
    return float(out) if np.ndim(out) == 0 else out


def steinway_string(midi: int) -> dict:
    """Length (m), diameter (m), density (kg/m3) and tension (N) of the nearest tabulated string."""
    r = CHABASSIER_D[int(np.argmin(np.abs(_MIDI - float(midi))))]
    return {"length": r[2], "diameter": r[3] * 1e-3, "density": r[4], "tension": r[5]}


# Rauhala & Valimaki, "Dispersion modeling in waveguide piano synthesis using tunable allpass filters",
# DAFx-06, Table 2: closed-form design of a cascade of M identical first-order allpasses (eqs 3-8)
_RAUHALA = dict(k1=-0.00179, k2=-0.0233, k3=-2.93, m1=0.0126, m2=0.0606, m3=-0.00825, m4=1.97)


def _ap_phase(w, a):
    """Phase lag (rad, positive) of (a + z^-1) / (1 + a z^-1) at w rad/sample."""
    return -np.angle((a + np.exp(-1j * w)) / (1.0 + a * np.exp(-1j * w)))


def rauhala_coefficient(f0: float, B: float, M: int = 8) -> float:
    """a1 of each of M identical first-order allpasses: D = exp(Cd - Ikey kd), a1 = (1 - D)/(1 + D) (eqs 3-8;
    Ikey = piano key number, 1 = A0). The fit covers B up to about 1e-3."""
    p = _RAUHALA
    ikey = math.log(f0 * 2 ** (1 / 12) / 27.5) / math.log(2 ** (1 / 12))
    lb, lm = math.log(max(B, 1e-9)), math.log(max(int(M), 1))
    kd = math.exp(p["k1"] * lb * lb + p["k2"] * lb + p["k3"])
    cd = math.exp((p["m1"] * lm + p["m2"]) * lb + p["m3"] * lm + p["m4"])
    d = math.exp(cd - ikey * kd)
    return (1.0 - d) / (1.0 + d)


def _fit(f0, B, sr, M):
    """(a1, delay, rms phase residual) of M identical first-order allpasses for the partials under 0.4 sr."""
    w0 = 2 * math.pi * f0 / sr
    n = np.arange(1, 17)
    wn = w0 * n * np.sqrt(1 + B * n * n) / math.sqrt(1 + B)
    keep = wn < 0.8 * math.pi
    n, wn = n[keep], wn[keep]

    def err(a):                                           # loop phase residual at the target partials
        N = (2 * math.pi - M * _ap_phase(w0, a)) / w0
        return float(np.sum((N * wn + M * _ap_phase(wn, a) - 2 * math.pi * n) ** 2))
    if B <= 1e-3 and abs(sr - 44100) < 1 and M == 8:
        a = rauhala_coefficient(f0, B, M)
    elif len(n) > 1:
        from scipy.optimize import minimize_scalar
        a = float(minimize_scalar(err, bounds=(-0.98, 0.0), method="bounded").x)
    else:
        a = 0.0
    a = float(np.clip(a, -0.98, 0.98))
    return a, M * float(_ap_phase(w0, a)) / w0, math.sqrt(err(a) / max(len(n), 1))


def dispersion_allpass(f0: float, B: float, sr: float, M: int | None = None) -> tuple[float, float, int]:
    """(a1, delay, M) for a stiff string of fundamental ``f0`` and inharmonicity ``B`` in a waveguide at ``sr``:
    M identical first-order allpasses whose a1 puts the loop's partials on n f0 sqrt(1 + B n^2) / sqrt(1 + B),
    and the phase delay (samples) they add at f0, to take off the delay line. Rauhala's closed form for M = 8,
    B <= 1e-3 (its fitted range) at 44.1 kHz; otherwise a1 is fitted to the partials under 0.4 sr. With M None,
    the M of 1, 2, 4, 8, 12, 16, 24, 32 that fits best while leaving at least 3 samples of plain delay (the treble needs
    more sections: 8 leave C6 up to 38 cents off, 32 within 5). ``B`` <= 0 gives (0, 0, 0)."""
    if B <= 0 or f0 <= 0:
        return 0.0, 0.0, 0
    period = sr / f0
    if M is not None:
        a, d, _ = _fit(f0, B, sr, int(M))
        return a, d, int(M)
    best = None
    for m in (1, 2, 4, 8, 12, 16, 24, 32):
        a, d, e = _fit(f0, B, sr, m)
        if period - d >= 3.0 and (best is None or e < best[3]):
            best = (a, d, m, e)
    if best is None:
        return 0.0, 0.0, 0
    return best[0], best[1], best[2]
