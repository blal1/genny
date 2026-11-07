"""Story, "A parametric model of the vocal tract area function for vowel and consonant simulation",
JASA 117(5), 3231-3254 (2005): 44 sections from glottis (1) to lips (44), 0.396825 cm each (17.46 cm, note 5).

Tier I, the vowel substrate (eq. 3):  V(i) = (pi/4) [Omega(i) + q1 phi1(i) + q2 phi2(i)]^2   (cm^2)
Tier II, a consonant constriction (eqs 5-9), multiplied into it: A(i) = V(i) C(i),
  C(i) = 1 - m_c d_c exp(-ln 16 ((i - l_c) / r_c{b,f})^2),  d_c = 1 - a_c / V(l_c),
  r_cf = s_c r_c / (1 + s_c), r_cb = r_c / (1 + s_c), C = max(C, 0).
"""
from __future__ import annotations

import functools
import math

import numba
import numpy as np

DX_CM = 0.396825
LENGTH_M = 44 * DX_CM / 100.0

# Table III: neutral diameter Omega(i) and the two modes phi1(i), phi2(i)
OMEGA = np.array([0.636, 0.561, 0.561, 0.550, 0.598, 0.895, 1.187, 1.417, 1.380, 1.273, 1.340, 1.399, 1.433, 1.506,
                  1.493, 1.473, 1.499, 1.529, 1.567, 1.601, 1.591, 1.547, 1.570, 1.546, 1.532, 1.496, 1.429, 1.425,
                  1.496, 1.608, 1.668, 1.757, 1.842, 1.983, 2.073, 2.123, 2.194, 2.175, 2.009, 1.785, 1.675, 1.539,
                  1.405, 1.312])
PHI1 = np.array([0.018, 0.001, -0.013, -0.025, -0.036, -0.048, -0.062, -0.076, -0.093, -0.111, -0.130, -0.149, -0.167,
                 -0.183, -0.196, -0.204, -0.207, -0.203, -0.193, -0.175, -0.151, -0.119, -0.082, -0.041, 0.004, 0.051,
                 0.097, 0.141, 0.181, 0.214, 0.240, 0.257, 0.264, 0.260, 0.246, 0.224, 0.194, 0.159, 0.122, 0.087,
                 0.057, 0.038, 0.034, 0.048])
PHI2 = np.array([-0.013, -0.007, -0.029, -0.059, -0.088, -0.108, -0.120, -0.123, -0.118, -0.107, -0.092, -0.075, -0.056,
                 -0.035, -0.014, 0.008, 0.032, 0.057, 0.084, 0.111, 0.138, 0.164, 0.188, 0.206, 0.218, 0.221, 0.214,
                 0.195, 0.164, 0.121, 0.070, 0.013, -0.046, -0.100, -0.143, -0.167, -0.165, -0.132, -0.066, 0.031,
                 0.148, 0.264, 0.346, 0.338])

# Story, JASA 123(1), 327-335 (2008), Table I: measured (MRI, 2002) area functions of one adult male, 44 sections
# glottis -> lips, cm^2, and the section length Delta (cm) of each vowel
MEASURED = {
    "i": (0.384, (0.51, 0.59, 0.62, 0.72, 1.24, 2.30, 3.30, 3.59, 3.22, 2.86, 3.00, 3.61, 4.39, 4.95, 5.17, 5.16, 5.18, 5.26, 5.20, 5.02, 4.71, 4.13, 3.43, 2.83, 2.32, 1.83, 1.46, 1.23, 1.08, 0.94, 0.80, 0.67, 0.55, 0.46, 0.40, 0.36, 0.35, 0.35, 0.38, 0.51, 0.74, 0.92, 0.96, 0.91)),
    "ɪ": (0.376, (0.28, 0.21, 0.21, 0.30, 0.47, 0.71, 1.12, 1.48, 1.35, 1.05, 0.92, 0.92, 1.19, 1.94, 2.83, 3.31, 3.48, 3.60, 3.64, 3.49, 3.20, 2.90, 2.59, 2.21, 1.87, 1.54, 1.20, 0.92, 0.74, 0.59, 0.52, 0.54, 0.59, 0.65, 0.71, 0.67, 0.61, 0.57, 0.50, 0.48, 0.54, 0.73, 0.93, 0.82)),
    "e": (0.386, (0.29, 0.26, 0.30, 0.40, 0.55, 0.74, 0.99, 1.09, 0.90, 0.69, 0.77, 1.31, 2.13, 2.74, 3.03, 3.23, 3.33, 3.27, 3.09, 2.84, 2.66, 2.46, 2.14, 1.79, 1.44, 1.17, 1.00, 0.88, 0.80, 0.81, 0.85, 0.84, 0.86, 0.96, 1.18, 1.35, 1.48, 1.62, 1.49, 1.29, 1.24, 1.17, 1.04, 0.95)),
    "ɛ": (0.393, (0.37, 0.30, 0.24, 0.23, 0.29, 0.41, 0.58, 0.82, 0.97, 0.82, 0.62, 0.54, 0.54, 0.68, 1.09, 1.62, 2.03, 2.35, 2.51, 2.39, 2.22, 2.13, 2.00, 1.78, 1.58, 1.43, 1.31, 1.23, 1.24, 1.38, 1.61, 1.82, 1.96, 2.01, 2.00, 1.95, 1.77, 1.48, 1.30, 1.21, 1.10, 0.99, 0.91, 0.78)),
    "æ": (0.366, (0.31, 0.21, 0.18, 0.23, 0.33, 0.50, 0.78, 0.96, 0.85, 0.63, 0.46, 0.36, 0.33, 0.46, 0.73, 1.00, 1.30, 1.66, 1.97, 2.06, 2.03, 2.01, 1.89, 1.66, 1.49, 1.42, 1.37, 1.34, 1.41, 1.58, 1.82, 2.19, 2.63, 2.97, 3.17, 3.40, 3.56, 3.57, 3.58, 3.44, 3.15, 3.38, 3.99, 4.17)),
    "ʌ": (0.39, (0.23, 0.34, 0.47, 0.60, 0.77, 1.06, 1.26, 1.09, 0.80, 0.65, 0.56, 0.47, 0.37, 0.24, 0.17, 0.18, 0.23, 0.27, 0.28, 0.29, 0.33, 0.52, 0.97, 1.50, 1.91, 2.23, 2.65, 3.29, 4.13, 5.00, 5.77, 6.33, 6.61, 6.63, 6.45, 6.04, 5.39, 4.42, 3.29, 2.37, 1.74, 1.36, 1.17, 0.99)),
    "ɑ": (0.388, (0.56, 0.62, 0.66, 0.78, 0.97, 1.16, 1.12, 0.82, 0.55, 0.45, 0.37, 0.29, 0.21, 0.15, 0.16, 0.25, 0.34, 0.43, 0.54, 0.61, 0.67, 0.98, 1.76, 2.75, 3.52, 4.08, 4.74, 5.61, 6.60, 7.61, 8.48, 9.06, 9.29, 9.26, 9.06, 8.64, 7.91, 6.98, 6.02, 5.13, 4.55, 4.52, 4.71, 4.72)),
    "ɔ": (0.395, (0.27, 0.43, 0.54, 0.67, 0.83, 0.92, 0.89, 0.73, 0.55, 0.44, 0.37, 0.28, 0.18, 0.14, 0.15, 0.14, 0.13, 0.14, 0.18, 0.20, 0.23, 0.49, 1.03, 1.58, 2.06, 2.61, 3.35, 4.34, 5.51, 6.70, 7.75, 8.63, 9.29, 9.59, 9.42, 8.78, 7.82, 6.50, 4.95, 3.47, 2.15, 1.38, 1.11, 0.90)),
    "o": (0.417, (0.38, 0.45, 0.57, 0.77, 1.31, 1.92, 1.74, 1.11, 0.75, 0.59, 0.57, 0.68, 0.73, 0.67, 0.58, 0.49, 0.44, 0.42, 0.49, 0.53, 0.38, 0.30, 0.45, 0.61, 0.71, 0.79, 0.86, 1.01, 1.41, 2.09, 3.00, 4.10, 5.16, 6.22, 7.34, 8.15, 8.61, 8.37, 6.76, 4.37, 2.30, 1.06, 0.58, 0.47)),
    "ʊ": (0.44, (0.37, 0.38, 0.49, 0.62, 0.85, 1.28, 1.62, 1.47, 1.04, 0.81, 1.03, 1.44, 1.49, 1.28, 1.06, 0.85, 0.69, 0.56, 0.42, 0.27, 0.27, 0.41, 0.51, 0.49, 0.47, 0.50, 0.58, 0.80, 1.19, 1.62, 2.27, 3.24, 4.16, 5.00, 5.70, 6.11, 6.21, 6.29, 6.24, 4.91, 2.61, 1.09, 0.63, 0.59)),
    "u": (0.445, (0.54, 0.61, 0.66, 0.75, 1.13, 1.99, 2.83, 2.90, 2.52, 2.40, 2.83, 3.56, 3.99, 3.89, 3.50, 3.04, 2.64, 2.44, 2.31, 2.07, 1.80, 1.52, 1.14, 0.74, 0.42, 0.22, 0.14, 0.20, 0.47, 0.89, 1.15, 1.42, 2.17, 3.04, 3.69, 4.70, 5.74, 5.41, 3.82, 2.34, 1.35, 0.65, 0.29, 0.16)),
}

MEASURED_FEMALE = {
    "i": (0.36, 0.50, 0.82, 1.00, 1.45, 3.90, 3.33, 2.84, 2.75, 3.84, 4.30, 4.38, 4.16, 3.22, 2.50, 2.25, 2.19, 2.27, 1.78, 1.25, 0.74, 0.28, 0.18, 0.14, 0.19, 0.28, 0.48, 0.75, 0.51, 0.71, 1.00, 0.74, 1.24, 1.51, 1.65, 1.59),
    "ɪ": (0.21, 0.14, 0.40, 0.51, 0.82, 0.72, 0.88, 1.01, 0.91, 1.55, 1.70, 1.86, 1.36, 1.14, 1.67, 1.93, 1.79, 1.57, 1.46, 1.22, 1.28, 1.22, 1.31, 1.23, 1.13, 1.20, 0.98, 1.47, 1.70, 0.97, 0.86, 1.20, 1.18, 1.34),
    "ɛ": (0.18, 0.15, 0.55, 0.92, 0.49, 0.53, 0.50, 0.46, 1.35, 1.16, 1.54, 1.31, 1.03, 1.38, 1.67, 1.45, 1.26, 1.22, 1.29, 1.27, 1.28, 1.45, 1.83, 1.46, 1.26, 1.25, 1.77, 1.60, 1.52, 1.80, 2.14, 1.59),
    "æ": (0.26, 0.19, 0.70, 1.23, 1.14, 1.03, 1.10, 1.27, 1.78, 1.94, 1.80, 2.16, 2.78, 3.01, 3.00, 2.98, 2.92, 3.27, 3.80, 3.79, 4.03, 3.46, 3.06, 2.75, 3.89, 2.91, 3.22, 3.82, 3.90, 3.43),
    "ʌ": (0.18, 0.29, 0.51, 0.94, 0.43, 0.37, 0.42, 0.27, 0.51, 0.60, 1.00, 0.48, 0.21, 0.28, 0.62, 0.84, 0.92, 1.15, 1.47, 2.00, 2.32, 2.71, 2.64, 2.74, 2.26, 2.14, 1.96, 2.02, 1.83, 1.25, 1.35, 1.90, 1.70, 1.69),
    "ɑ": (0.20, 0.42, 0.46, 1.17, 1.55, 0.99, 0.38, 0.18, 0.97, 1.17, 1.24, 1.32, 1.08, 1.18, 1.38, 2.04, 2.06, 2.36, 2.78, 2.86, 3.50, 3.54, 3.84, 4.31, 5.16, 5.19, 4.39, 3.20, 2.89, 2.13, 1.95, 1.38, 1.29, 2.81),
    "ɔ": (0.82, 1.11, 0.95, 0.70, 0.44, 0.45, 0.38, 0.36, 0.56, 1.01, 0.95, 0.79, 0.68, 0.33, 0.88, 2.06, 2.41, 2.47, 3.10, 4.01, 4.43, 5.15, 5.61, 6.07, 6.62, 7.15, 8.05, 8.38, 8.13, 8.10, 8.64, 6.02, 3.32, 2.26, 1.94, 1.58),
    "o": (0.17, 0.18, 0.81, 0.65, 0.61, 0.46, 0.44, 0.61, 0.67, 0.97, 1.19, 0.66, 0.37, 0.46, 1.33, 2.00, 1.81, 1.77, 2.24, 2.43, 2.87, 4.24, 4.63, 4.95, 4.92, 4.76, 5.34, 5.43, 4.75, 4.26, 2.97, 1.99, 1.33, 1.45, 1.60, 1.39),
    "ʊ": (0.07, 0.10, 0.07, 0.05, 0.18, 0.43, 0.75, 0.47, 0.29, 0.11, 0.32, 0.58, 1.11, 0.84, 0.17, 0.39, 0.81, 1.03, 1.02, 1.04, 1.42, 1.93, 2.10, 2.21, 2.53, 2.82, 2.80, 2.46, 2.32, 2.46, 2.09, 2.57, 1.76, 1.37, 1.09, 1.16),
    "u": (0.22, 0.48, 0.81, 0.90, 1.20, 3.13, 2.71, 1.75, 2.01, 2.20, 2.16, 2.56, 2.61, 2.34, 2.05, 1.21, 0.80, 0.74, 1.36, 1.18, 1.05, 0.78, 0.78, 1.17, 1.47, 2.00, 2.50, 2.92, 3.10, 2.98, 3.40, 3.61, 2.97, 2.74, 1.79, 0.87, 0.44, 0.18),
}

# Table IV (2005): (q1, q2) per vowel. Its reconstructions put /i/ at F1 108 Hz and /ʌ/ at 201 Hz (checked with a
# lossless chain-matrix model), so the vowels themselves come from MEASURED; the modes stay for the neutral tract.
VOWEL_Q = {"i": (-5.176, -2.556), "ɪ": (-1.092, 0.677), "ɛ": (2.561, 3.849), "æ": (3.471, 1.729), "ʌ": (0.102, -3.565),
           "ɑ": (0.981, 1.043), "ɔ": (0.331, 2.315), "ʊ": (0.128, 1.356), "o": (-0.493, -1.910), "u": (-2.685, -2.065)}
ALIASES = {"iy": "i", "ih": "ɪ", "eh": "ɛ", "ey": "e", "ae": "æ", "ah": "ʌ", "aa": "ɑ", "a": "ɑ", "ao": "ɔ", "uh": "ʊ",
           "ow": "o", "uw": "u"}

# Table I: consonants (q1, q2, l_c, a_c cm^2, r_c, s_c, m_c, a_np cm^2); q1, q2 are the vowel context of the fit
CONSONANTS = {"p": (-1.5, 0.5, 44, 0.0, 4, 1.0, 1.1, 0.0), "t": (-1.5, 2.0, 39, 0.0, 6, 1.1, 1.3, 0.0),
              "k": (0.0, 0.0, 30, 0.0, 10, 1.5, 1.2, 0.0), "m": (0.0, -3.0, 44, 0.0, 4, 1.0, 1.1, 1.04),
              "n": (-0.5, 0.4, 38, 0.0, 11, 3.0, 1.3, 1.09), "ŋ": (3.0, -1.0, 31, 0.0, 8, 1.5, 2.0, 1.26),
              "f": (-3.0, -2.0, 43, 0.1, 4, 1.0, 1.0, 0.0), "θ": (1.0, 2.0, 42, 0.05, 9, 0.3, 1.0, 0.0),
              "s": (0.5, 3.5, 41, 0.05, 9, 0.6, 1.0, 0.0), "ʃ": (-4.0, 3.5, 36, 0.05, 11, 0.4, 1.0, 0.0)}
CONS_ALIASES = {"ng": "ŋ", "th": "θ", "sh": "ʃ", "b": "p", "d": "t", "g": "k", "v": "f", "z": "s", "dh": "θ", "zh": "ʃ"}


# Story 2005, Table II (q1, q2, l_c, a_c, r_c), s = m = 1: liquids as two superposed constrictions, A = V C1 C2.
# F3 goes 2.3 -> 1.75 kHz for /ɹ/b and 2.7 -> 3 kHz for /l/b.
LIQUIDS = {"ɹa": (0.0, 2.0, (12.0, 35.0), (1.0, 0.6), (4.0, 4.0)), "la": (3.0, 0.0, (12.0, 40.0), (3.5, 1.0), (4.0, 4.0)),
           "ɹb": (0.0, -1.0, (12.0, 32.0), (0.4, 0.1), (4.0, 6.0)), "lb": (2.0, -2.0, (12.0, 40.0), (3.0, 0.1), (4.0, 4.0))}

# Warm humid air of the tract (cgs), wall and loss constants.
C_CMS, RHO, MU = 35000.0, 0.00114, 1.86e-4
WALL_R, WALL_FW, WALL_FT, WALL_Q = 408.0, 15.0, 200.0, 4.0


def vowel_key(v: str) -> str:
    v = str(v).strip("/").lower()
    v = ALIASES.get(v, v)
    if v == "neutral":
        return v
    if v not in MEASURED:
        raise ValueError(f"unknown vowel {v!r}; choose from {sorted(MEASURED) + sorted(ALIASES)}")
    return v


def substrate(q1: float, q2: float) -> np.ndarray:
    """Tier I vowel area function (cm^2), eq. 3."""
    return np.pi / 4.0 * (OMEGA + q1 * PHI1 + q2 * PHI2) ** 2


def vowel_area(v: str, speaker: str = "male") -> np.ndarray:
    """Measured area function (cm^2) of vowel ``v``, glottis to lips: male 44 sections, female 30-38 sections of
    DX_CM (no /e/, no neutral). ``neutral`` is the mean of the male areas."""
    k = vowel_key(v)
    if speaker == "female":
        if k not in MEASURED_FEMALE:
            raise ValueError(f"no female area function for {v!r}; choose from {sorted(MEASURED_FEMALE)}")
        return np.array(MEASURED_FEMALE[k], float)
    if k == "neutral":
        return np.mean([a for _, a in MEASURED.values()], axis=0)
    return np.array(MEASURED[k][1], float)


def vowel_length(v: str, speaker: str = "male") -> float:
    """Tract length (m) of vowel ``v``: male 44 Delta, female sections x DX_CM."""
    if speaker == "female":
        return vowel_area(v, "female").size * DX_CM / 100.0
    k = vowel_key(v)
    if k == "neutral":
        return 44 * float(np.mean([d for d, _ in MEASURED.values()])) / 100.0
    return 44 * MEASURED[k][0] / 100.0


def min_jerk(t, T: float, u0: float = 0.0, uf: float = 1.0):
    """Minimum-jerk move of a tract parameter from u0 to uf over T s (Story 2005, eq. 14)."""
    x = np.clip(np.asarray(t, float) / T, 0.0, 1.0)
    return u0 + (uf - u0) * (10.0 * x ** 3 - 15.0 * x ** 4 + 6.0 * x ** 5)


def _line(w):
    alpha = np.sqrt(w * WALL_Q)
    beta = w * (2 * np.pi * WALL_FT) ** 2 / ((w + WALL_R) * w + (2 * np.pi * WALL_FW) ** 2) + alpha
    gamma = np.sqrt((WALL_R + w) / (beta + w))
    return gamma, gamma * (beta + w)


def _zrad(a_cm2, w):
    a = max(float(a_cm2), 1e-3)
    L = 8.0 * math.sqrt(a / math.pi) * RHO / (3.0 * math.pi * a)
    R = 128.0 * RHO * C_CMS / (9.0 * math.pi ** 2 * a)
    return w * L * R / (R + w * L)


def _march(A, dx, w, P, U, out, branches=(), keep=False):
    """Walk the chain from its far end (P, U, out) to its near end; ``branches`` load the junctions."""
    gamma, sigma = _line(w)
    A = np.asarray(A, float)
    dx = np.broadcast_to(np.asarray(dx, float), A.shape)
    by_at = {}
    for b in branches:
        by_at.setdefault(int(b["at"]), []).append(b)
    trace = []
    for i in range(A.size - 1, -1, -1):
        ch, sh = np.cosh(sigma * dx[i] / C_CMS), np.sinh(sigma * dx[i] / C_CMS)
        z = RHO * C_CMS / max(A[i], 1e-3)
        P, U = ch * P + z * gamma * sh * U, sh / (z * gamma) * P + ch * U
        if keep:
            trace.append((P, U))
        for b in by_at.get(i, ()):
            Y, T = branch_load(b, w)
            U = U + Y * P
            out = out + T * P
    return P, U, out, trace[::-1]


def branch_load(b, w):
    """(admittance U/P at its mouth, radiated flow per mouth pressure) of a side branch: ``area`` (cm^2 per
    section, mouth first), ``dx`` (cm), ``end`` closed | radiating | helmholtz (``volume`` cm^3, ``neck_area`` cm^2,
    ``neck_len`` cm), optional nested ``branches``."""
    end = b.get("end", "closed")
    if end == "helmholtz":
        an = float(b["neck_area"])
        ln = float(b["neck_len"]) + 1.7 * math.sqrt(an / math.pi)
        M, Cv = RHO * ln / an, float(b["volume"]) / (RHO * C_CMS ** 2)
        Rn = 8.0 * math.pi * MU * ln / an ** 2
        return 1.0 / (w * M + Rn + 1.0 / (w * Cv)), np.zeros_like(w)
    A = np.asarray(b["area"], float)
    one, zero = np.ones_like(w), np.zeros_like(w)
    if end == "radiating":
        P, U, out, _ = _march(A, b.get("dx", DX_CM), w, _zrad(A[-1], w), one, one, b.get("branches", ()))
    else:
        P, U, out, _ = _march(A, b.get("dx", DX_CM), w, one, zero, zero, b.get("branches", ()))
    return U / P, out / P


def tract_response(A, f, dx_cm=DX_CM, branches=(), source=None) -> np.ndarray:
    """|U_radiated / U_source| of the area function ``A`` (cm^2) at ``f`` (Hz > 0): chain of lossy transmission-line
    sections with yielding walls, viscous and thermal losses. Radiation: piston in a baffle,
    Z = j w L R / (R + j w L), L = 8 a rho / (3 pi A), R = 128 rho c / (9 pi^2 A).

    ``dx_cm`` one length or one per section (``section_lengths``). ``branches``: side branches (``branch_load``)
    at junction ``at`` (sections counted from the glottis: piriform 4, velum 22); radiating ones (the nose) add their
    flow. ``source``: None = ideal flow source (rigid closure at the glottis), else a callable w -> impedance of the
    glottis plus the subglottal system (``source_impedance``): the result is then U_out / (U_g + P_g / Z_s)."""
    w = 2j * np.pi * np.asarray(f, float)
    A = np.asarray(A, float)
    P, U, out, _ = _march(A, dx_cm, w, _zrad(A[-1], w), np.ones_like(w), np.ones_like(w), branches)
    if source is not None:
        U = U + P / source(w)
    return np.abs(out / U)


def formants(A, dx_cm=DX_CM, fmax: float = 5000.0, df: float = 2.0, branches=(), source=None) -> np.ndarray:
    """Formants (Hz) of the area function: the peaks of ``tract_response``."""
    f = np.arange(50.0, fmax, df)
    h = tract_response(A, f, dx_cm, branches, source)
    return f[1:-1][(h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:])]


def formant_error(F, target) -> np.ndarray:
    """Percent error of each formant re ``target``: 100 (F - target) / target (Story 2008, Table II)."""
    target = np.asarray(target, float)
    return 100.0 * (np.asarray(F, float)[:target.size] - target) / target


def liquid_area(name: str) -> np.ndarray:
    """A = V C1 C2 of Story 2005 Table II; ``name`` in ``LIQUIDS``."""
    q1, q2, lc, ac, rc = LIQUIDS[name]
    V = substrate(q1, q2)
    A = V.copy()
    for l_, a_, r_ in zip(lc, ac, rc):
        A = A * constriction(V, l_, a_, r_, 1.0, 1.0)
    return A


def corner_weights(v: str) -> tuple[float, float]:
    """(a1, a2): x_v = x_a + a1 (x_i - x_a) + a2 (x_u - x_a), least squares on the areas, with 0 <= a1, 0 <= a2,
    a1 + a2 <= 1 (Birkholz 2011, eqs 2-4)."""
    return _corner_weights(vowel_area(v))


def _corner_weights(x: np.ndarray) -> tuple[float, float]:
    xa, xi, xu = vowel_area("ɑ"), vowel_area("i"), vowel_area("u")
    a = np.linalg.lstsq(np.stack([xi - xa, xu - xa], axis=1), x - xa, rcond=None)[0]
    a = np.maximum(a, 0.0)
    return tuple(float(z) for z in (a / max(1.0, float(a.sum()))))


def coarticulated_area(c: str, v) -> np.ndarray:
    """Closure of consonant ``c`` in the context of vowel ``v`` (name or area function): the three corner closures
    weighted as the vowel is, x_c = x_c(a) + a1 (x_c(i) - x_c(a)) + a2 (x_c(u) - x_c(a)) (Birkholz 2011, eq. 5)."""
    a1, a2 = _corner_weights(vowel_area(v) if isinstance(v, str) else np.asarray(v, float))
    ca, ci, cu = (consonant_area(c, vowel_area(k)) for k in ("ɑ", "i", "u"))
    return ca + a1 * (ci - ca) + a2 * (cu - ca)


def constriction(V: np.ndarray, lc: float, ac: float, rc: float, sc: float, mc: float) -> np.ndarray:
    """Tier II superposition C(i), eqs 5-9, on the vowel V (cm^2); i, l_c, r_c in section units."""
    i = np.arange(1, V.size + 1, dtype=float)
    vl = float(np.interp(lc, i, V))
    dc = 1.0 - ac / max(vl, 1e-9)
    r = np.where(i < lc, rc / (1.0 + sc), sc * rc / (1.0 + sc))
    return np.maximum(1.0 - mc * dc * np.exp(-math.log(16.0) * ((i - lc) / r) ** 2), 0.0)


def consonant_area(c: str, V: np.ndarray, mc: float | None = None) -> np.ndarray:
    """A = V C for consonant ``c`` from Table I over the vowel ``V`` (the coarticulated closure)."""
    key = CONS_ALIASES.get(str(c).lower(), str(c).lower())
    if key not in CONSONANTS:
        raise ValueError(f"unknown consonant {c!r}; choose from {sorted(CONSONANTS) + sorted(CONS_ALIASES)}")
    _, _, lc, ac, rc, sc, m, _ = CONSONANTS[key]
    return V * constriction(V, lc, ac, rc, sc, m if mc is None else mc)


def breakpoints(A: np.ndarray, floor_cm2: float = 0.02):
    """(x, S) pairs on x in [0, 1] (section centres) for ``tubes.WebsterTube``; closures floored."""
    A = np.maximum(np.asarray(A, float), floor_cm2)
    x = (np.arange(A.size) + 0.5) / A.size
    return tuple(zip(np.concatenate([[0.0], x, [1.0]]), np.concatenate([[A[0]], A, [A[-1]]])))


# ------------------------------------------------------------------------------------------ measured fricatives
# (Delta cm, areas cm^2 glottis -> lips, 44 sections)
FRICATIVES_MEASURED = {
    "f": (0.361, (0.37, 0.32, 0.65, 1.22, 1.46, 1.43, 1.34, 1.32, 1.84, 2.55, 2.21, 2.16, 2.09, 1.82, 2.01, 1.95, 1.91, 1.77, 1.88, 2.40, 2.04, 1.87, 2.13, 2.05, 1.65, 1.05, 0.64, 0.52, 0.45, 0.45, 0.29, 0.24, 0.16, 0.21, 1.04, 1.88, 2.27, 2.00, 1.93, 1.89, 1.31, 0.83, 0.33, 0.11)),
    "θ": (0.356, (0.39, 0.40, 0.76, 0.71, 1.10, 1.50, 1.34, 1.23, 1.15, 1.34, 1.72, 1.65, 1.52, 1.50, 1.32, 1.42, 1.49, 1.49, 1.94, 2.41, 2.29, 2.37, 2.99, 3.09, 2.98, 2.75, 2.82, 2.96, 2.95, 3.13, 3.24, 3.13, 3.36, 3.43, 3.21, 3.08, 2.82, 2.39, 1.75, 0.90, 0.09, 0.00, 0.09, 0.22)),
    "s": (0.353, (0.47, 0.47, 0.54, 0.68, 0.94, 2.23, 2.18, 2.39, 2.17, 1.95, 2.12, 2.19, 2.61, 3.08, 2.51, 2.32, 2.70, 2.65, 3.21, 3.71, 3.72, 4.14, 4.58, 4.63, 4.32, 4.38, 4.21, 4.52, 4.53, 4.40, 3.98, 3.78, 3.52, 3.20, 2.77, 2.16, 1.53, 1.03, 0.60, 0.26, 0.11, 0.15, 0.24, 0.06)),
    "ʃ": (0.377, (0.49, 0.49, 0.63, 0.64, 0.83, 2.26, 2.51, 2.53, 2.77, 2.85, 3.11, 3.42, 3.31, 3.69, 3.54, 3.74, 4.18, 4.56, 4.98, 5.23, 5.18, 5.80, 5.67, 4.82, 4.39, 3.82, 2.89, 2.48, 1.81, 1.38, 0.98, 0.57, 0.45, 0.34, 0.03, 0.00, 0.00, 0.00, 0.34, 2.13, 1.96, 1.24, 0.76, 0.33)),
}


def fricative_area(c: str) -> tuple[np.ndarray, float]:
    """(areas cm^2, section length cm) of a measured fricative f th s sh (v dh z zh: same place)."""
    key = CONS_ALIASES.get(str(c).lower(), str(c).lower())
    if key not in FRICATIVES_MEASURED:
        raise ValueError(f"no measured fricative {c!r}; choose from {sorted(FRICATIVES_MEASURED)}")
    d, a = FRICATIVES_MEASURED[key]
    return np.array(a, float), d


# ------------------------------------------------------------------------------------------ section lengths
def section_lengths(n: int = 44, delta: float = DX_CM, pg: float = 0.0, pm: float = 0.0, lg: float = 1.0,
                    lm: float = 44.0, rg: float = 8.0, rm: float = 8.0) -> np.ndarray:
    """Length (cm) of each section: L(i) = Delta a(i) b(i), a = 1 + pg e_g(i) / (Delta sum e_g), e_g =
    exp(-K ((i - lg) / (2 rg))^2), K = 2 ln 10000, b the same at the lips. ``pg`` < 0 shortens the larynx end,
    ``pm`` > 0 protrudes the lips (cm added to the tract)."""
    K = 2.0 * math.log(10000.0)
    i = np.arange(1, n + 1, dtype=float)
    eg = np.exp(-K * ((i - lg) / (2.0 * rg)) ** 2)
    em = np.exp(-K * ((i - lm) / (2.0 * rm)) ** 2)
    return delta * (1.0 + pg * eg / (delta * eg.sum())) * (1.0 + pm * em / (delta * em.sum()))


def uniform(A, lengths, dx: float) -> tuple[np.ndarray, float]:
    """Area function on non-uniform sections resampled to equal sections close to ``dx`` (cm):
    (areas, section length)."""
    A = np.asarray(A, float)
    lengths = np.broadcast_to(np.asarray(lengths, float), A.shape)
    total = float(lengths.sum())
    n = max(2, int(round(total / dx)))
    xc = np.cumsum(lengths) - 0.5 * lengths
    return np.interp((np.arange(n) + 0.5) * total / n, xc, A), total / n


# ------------------------------------------------------------------------------------------ side branches
PIRIFORM_AT, PIRIFORM_SECTIONS, PIRIFORM_VOLUME = 4, 3, 0.5       # junction, sections of DX_CM, cm^3 (both sides)
VELUM_AT = 22                                                    # nasal port junction, sections from the glottis
NASAL_LEN_CM, NOSTRILS_CM2, NASAL_PEAK_CM2 = 11.5, 1.2, 3.5
SINUSES = ({"at": 17, "end": "helmholtz", "volume": 30.0, "neck_area": 0.2, "neck_len": 0.5},   # maxillary pair
           {"at": 12, "end": "helmholtz", "volume": 8.0, "neck_area": 0.08, "neck_len": 0.4},   # sphenoid
           {"at": 17, "end": "helmholtz", "volume": 6.0, "neck_area": 0.05, "neck_len": 1.0})   # frontal
SUBLINGUAL = (13.1, 0.375, (1.5, 0.5))                           # cm from the glottis, section length, areas
LATERAL = (22, 32, 0.04, 0.28)                                   # sections, lateral passage area range cm^2
VOWEL_MIN_CM2, LATERAL_MIN_CM2, CONSTRICTION_MIN_CM2 = 0.25, 0.20, 0.15


def nasal_area(port_cm2: float, dx: float = DX_CM) -> np.ndarray:
    """Nasal tract from the velopharyngeal port (first section = ``port_cm2``) to the two nostrils, summed."""
    n = max(4, int(round(NASAL_LEN_CM / dx)))
    a = np.interp(np.linspace(0.0, 1.0, n), (0.0, 0.3, 0.75, 1.0), (0.6, NASAL_PEAK_CM2, NASAL_PEAK_CM2, NOSTRILS_CM2))
    a[0] = max(float(port_cm2), 1e-4)
    return a


def nasal_branch(port_cm2: float, sinuses: bool = True) -> dict:
    """The nose as a radiating side branch at the velum (``tract_response(branches=...)``)."""
    return {"at": VELUM_AT, "area": nasal_area(port_cm2), "end": "radiating",
            "branches": SINUSES if sinuses else ()}


def piriform_branch(volume_cm3: float = PIRIFORM_VOLUME) -> dict:
    """Both piriform sinuses as one closed side branch just above the glottis."""
    L = PIRIFORM_SECTIONS * DX_CM
    return {"at": PIRIFORM_AT, "area": np.full(PIRIFORM_SECTIONS, float(volume_cm3) / L), "end": "closed"}


def liquid(name: str) -> tuple[np.ndarray, list]:
    """(areas, branches) of /ɹ/ (``ɹa``, ``ɹb``) with its sublingual cavity, or /l/ (``la``, ``lb``) with the
    lateral passages summed into the main channel."""
    A = liquid_area(name)
    if name.startswith("ɹ"):
        x, d, a = SUBLINGUAL
        return A, [{"at": int(round(x / DX_CM)), "area": np.array(a), "dx": d, "end": "closed"}]
    i0, i1, a0, a1 = LATERAL
    A = A.copy()
    A[i0 - 1:i1] += np.linspace(a0, a1, i1 - i0 + 1)
    return A, []


# ------------------------------------------------------------------------------------------ glottis and trachea
GLOTTIS_LEN_CM, GLOTTIS_THICK_CM, KC = 1.4, 0.3, 1.38
TRACHEA_LEN_CM, TRACHEA_TOP_CM2, TRACHEA_CM2, R_LUNG = 14.7, 0.3, 1.5, -0.7
JET_SPREAD = math.tan(math.radians(4.0))


def trachea_area(dx: float = DX_CM) -> np.ndarray:
    """Subglottal tube, glottis first: 0.3 cm^2 under the glottis widening to 1.5 cm^2 over three sections."""
    n = max(4, int(round(TRACHEA_LEN_CM / dx)))
    a = np.full(n, TRACHEA_CM2)
    a[:3] = np.linspace(TRACHEA_TOP_CM2, TRACHEA_CM2, 4)[:3]
    return a


def glottal_resistance(ag_cm2: float) -> tuple[float, float, float]:
    """(viscous R = 12 mu lg^2 Xg / Ag^3, kinetic k with dP = k |U| U, k = kc rho / (2 Ag^2), inertance
    rho Xg / Ag), cgs."""
    a = max(float(ag_cm2), 1e-6)
    return (12.0 * MU * GLOTTIS_LEN_CM ** 2 * GLOTTIS_THICK_CM / a ** 3, KC * RHO / (2.0 * a * a),
            RHO * GLOTTIS_THICK_CM / a)


def jet_recovery(ag_cm2: float, u_cm3s: float, a_tract_cm2: float, kind: str = "turbulent") -> float:
    """Pressure recovered by the glottal jet as a share of rho U^2 / (2 Ag^2): 2 r (1 - r), r = Ag / A_R, A_R the
    reattachment area. none | uniform (A_R = Ag) | laminar (A_R = Ag + 21.8 W (U^2/Ag)^(-1/3) nu^(2/3) dx^(2/3)) |
    turbulent (A_R = (h + 2 tan 4 deg dx) W), dx = glottal thickness, capped by the tract."""
    a = max(float(ag_cm2), 1e-6)
    if kind in ("none", "uniform"):
        return 0.0
    W, dx, nu = GLOTTIS_LEN_CM, GLOTTIS_THICK_CM, MU / RHO
    if kind == "laminar":
        ar = a + 21.8 * W * max(u_cm3s * u_cm3s / a, 1e-9) ** (-1.0 / 3.0) * nu ** (2.0 / 3.0) * dx ** (2.0 / 3.0)
    elif kind == "turbulent":
        ar = (a / W + 2.0 * JET_SPREAD * dx) * W
    else:
        raise ValueError(f"jet {kind!r}: none, uniform, laminar or turbulent")
    r = a / min(max(ar, a), max(float(a_tract_cm2), a))
    return 2.0 * r * (1.0 - r)


def source_impedance(ag_cm2: float = 0.08, u_cm3s: float = 200.0, subglottal: bool = True):
    """w -> small-signal impedance of the glottis (R + 2 k |U0| + j w L) in series with the subglottal input
    impedance, for ``tract_response(source=...)``."""
    R, k, L = glottal_resistance(ag_cm2)
    T = trachea_area()

    def z(w):
        zg = R + 2.0 * k * abs(u_cm3s) + w * L
        if not subglottal:
            return zg
        zl = (1.0 + R_LUNG) / (1.0 - R_LUNG) * RHO * C_CMS / T[-1]
        P, U, _, _ = _march(T, DX_CM, w, zl * np.ones_like(w), np.ones_like(w), np.zeros_like(w))
        return zg + P / U
    return z


# ------------------------------------------------------------------------------------------ sensitivity, deflection
def sensitivity(A, F, dx: float = DX_CM) -> np.ndarray:
    """Acoustic sensitivity S_j(i) = (KE - PE) / TE of each section to each frequency F_j (rows), KE = rho dx
    |U|^2 / (2 A), PE = A dx |P|^2 / (2 rho c^2): dF_j / F_j = sum_i S_j(i) dA_i / A_i. Lossless hard-walled tube,
    closed at the glottis, P = 0 at the lips, taken at its resonance nearest each F_j; P and U at mid-section."""
    A = np.maximum(np.asarray(A, float), 1e-3)
    z = RHO * C_CMS / A

    def walk(f, keep=False):
        k = 2.0 * np.pi * np.asarray(f, float) / C_CMS * 0.5 * dx
        c, s = np.cos(k), np.sin(k)
        P, U = np.zeros_like(k), np.ones_like(k)          # P imaginary part, U real part
        mid = []
        for i in range(A.size - 1, -1, -1):
            for half in (0, 1):
                P, U = c * P + z[i] * s * U, -s / z[i] * P + c * U
                if keep and half == 0:
                    mid.append((P, U))
        return U, mid[::-1]

    rows = []
    for fj in np.atleast_1d(F):
        f = np.arange(0.75 * float(fj), 1.25 * float(fj), 0.5)
        ug, _ = walk(f)
        cr = np.nonzero(np.sign(ug[:-1]) != np.sign(ug[1:]))[0]
        if cr.size:
            i = cr[np.argmin(np.abs(f[cr] - fj))]
            fj = f[i] - ug[i] * (f[i + 1] - f[i]) / (ug[i + 1] - ug[i])
        _, mid = walk(np.array([fj]), keep=True)
        P = np.array([p[0] for p, _ in mid])
        U = np.array([u[0] for _, u in mid])
        ke = 0.5 * RHO * dx * U * U / A
        pe = 0.5 * A * dx * P * P / (RHO * C_CMS ** 2)
        rows.append((ke - pe) / (ke + pe).sum())
    return np.array(rows)


DEFLECTION = {"b": (-1.0, -1.0, -1.0), "d": (-1.0, 1.0, 1.0), "g": (-1.0, 1.0, -1.0)}
I_EPI, R_EPI = 5, 6


def deflection_constriction(V, deltas, mu: float = 1.0, E: float = 1.0) -> np.ndarray:
    """C(i) = 1 - mu E y(i) / min y, y = g(i) sum_j delta_j S_j(i) over the first three formants of V; g = 0 below
    the epiglottis (i < 5), else 1 - exp(-ln 16 ((i - 5) / 6)^2); C >= 0. ``deltas`` a name of ``DEFLECTION``
    (b d g; p t k the same places) or three signs."""
    if isinstance(deltas, str):
        deltas = DEFLECTION[{"p": "b", "t": "d", "k": "g"}.get(deltas, deltas)]
    V = np.asarray(V, float)
    y = np.asarray(deltas, float) @ sensitivity(V, formants(V)[:3])
    i = np.arange(1, V.size + 1, dtype=float)
    y = y * np.where(i < I_EPI, 0.0, 1.0 - np.exp(-math.log(16.0) * ((i - I_EPI) / R_EPI) ** 2))
    return np.maximum(1.0 - float(mu) * float(E) * y / min(float(y.min()), -1e-12), 0.0)


# ------------------------------------------------------------------------------------------ (q1, q2) from formants
@functools.lru_cache(maxsize=2)
def _q_grid(n: int = 80):
    pts, fs = [], []
    for a in np.linspace(-6.2, 4.2, n):
        for b in np.linspace(-4.3, 4.6, n):
            V = substrate(a, b)
            if V.min() < 0.05:
                continue
            F = formants(V, fmax=3200.0, df=5.0)
            if F.size >= 2:
                pts.append((a, b))
                fs.append(F[:2])
    return np.array(pts), np.log(np.array(fs))


def q_from_formants(F1, F2):
    """(q1, q2) whose substrate has the nearest (F1, F2) in log frequency, on an 80 x 80 grid (arrays allowed)."""
    pts, lf = _q_grid()
    target = np.log(np.stack(np.broadcast_arrays(np.asarray(F1, float), np.asarray(F2, float)), axis=-1))
    d = ((target[..., None, :] - lf) ** 2).sum(-1)
    q = pts[np.argmin(d, axis=-1)]
    return q[..., 0], q[..., 1]


# ------------------------------------------------------------------------------------------ other speakers
@functools.lru_cache(maxsize=4)
def speaker_modes(speaker: str = "male"):
    """(Omega, phi1, phi2, q_min (2,), q_max (2,), mean length m) of a speaker on 44 sections. Male: the published
    modes and their vowel coefficients; female: principal components of the measured diameters."""
    if speaker == "male":
        q = np.array(list(VOWEL_Q.values()))
        return OMEGA, PHI1, PHI2, q.min(0), q.max(0), LENGTH_M
    if speaker != "female":
        raise ValueError("speaker: male or female")
    x = np.linspace(0.0, 1.0, OMEGA.size)
    D = np.array([np.interp(x, np.linspace(0.0, 1.0, len(a)), np.sqrt(4.0 * np.asarray(a) / np.pi))
                  for a in MEASURED_FEMALE.values()])
    om = D.mean(0)
    _, _, vt = np.linalg.svd(D - om, full_matrices=False)
    p1 = vt[0] * (np.sign(vt[0] @ PHI1) or 1.0)
    p2 = vt[1] * (np.sign(vt[1] @ PHI2) or 1.0)
    q = (D - om) @ np.stack([p1, p2], 1)
    L = float(np.mean([len(a) for a in MEASURED_FEMALE.values()])) * DX_CM / 100.0
    return om, p1, p2, q.min(0), q.max(0), L


def transfer(q1: float, q2: float, src: str = "male", dst: str = "female") -> tuple[np.ndarray, float]:
    """Vowel (q1, q2) of ``src`` on ``dst``'s tract: N_k = (q_k - q_kmin) / r_qk, p_k = N_k r_pk + p_kmin,
    V = (pi/4) [Omega + p1 phi1 + p2 phi2]^2. Returns (areas cm^2, tract length m)."""
    _, _, _, qa, qb, _ = speaker_modes(src)
    om, f1, f2, pa, pb, L = speaker_modes(dst)
    N = (np.array([q1, q2], float) - qa) / (qb - qa)
    p = N * (pb - pa) + pa
    return np.pi / 4.0 * np.maximum(om + p[0] * f1 + p[1] * f2, 0.0) ** 2, L


FEMALE_NATURAL = {"i": (489.0, 2387.0, 3526.0), "ɑ": (961.0, 1488.0, 2779.0), "ɔ": (817.0, 1264.0, 2574.0),
                  "u": (533.0, 1152.0, 3122.0)}


# ------------------------------------------------------------------------------------------ time-domain tract
NOISE_K, RE_CRIT, ASPIRATION_CM = 2e-6, 1800.0, 1.5
VISC_CM, WALL_LOSS, A_MIN_CM2 = 4e-4, 5e-4, 1e-5


@numba.njit(cache=True)
def _rad_coefs(a, fs):
    z0 = RHO * C_CMS / a
    L = 8.0 * math.sqrt(a / math.pi) * RHO / (3.0 * math.pi * a)
    R = 128.0 * RHO * C_CMS / (9.0 * math.pi ** 2 * a)
    K = 2.0 * fs
    a0 = K * L * (R + z0) + z0 * R
    return ((K * L * (R - z0) - z0 * R) / a0, (-K * L * (R - z0) - z0 * R) / a0,
            (-K * L * (R + z0) + z0 * R) / a0)


@numba.njit(cache=True)
def _kl_kernel(shapes, kidx, kfrac, port, ag, psub, noise, trach, nose, pir_a, pir_n, nose_at, pir_at, asp_at,
               r_lung, fs, nb, jet):
    """Kelly-Lochbaum tract on power-normalised waves (w = p sqrt(A / rho c)), one tick per section: oral tract,
    trachea, nose and piriform branch (3- and 4-port junctions), pressure-driven glottis with viscous, kinetic and
    inertive resistance, lips and nostrils radiating through a first-order piston load, turbulence noise above
    the critical Reynolds number."""
    m = kidx.size
    N = shapes.shape[1]
    NT = trach.size
    NN = nose.size
    zc = RHO * C_CMS
    sz = math.sqrt(zc)
    R = np.zeros(N)
    L = np.zeros(N)
    nR = np.zeros(N)
    nL = np.zeros(N)
    TU = np.zeros(NT)
    TD = np.zeros(NT)
    nTU = np.zeros(NT)
    nTD = np.zeros(NT)
    NR = np.zeros(max(NN, 1))
    NLw = np.zeros(max(NN, 1))
    nNR = np.zeros(max(NN, 1))
    nNL = np.zeros(max(NN, 1))
    PR = np.zeros(max(pir_n, 1))
    PL = np.zeros(max(pir_n, 1))
    nPR = np.zeros(max(pir_n, 1))
    nPL = np.zeros(max(pir_n, 1))
    A = np.zeros(N)
    Ap = np.zeros(N)
    sA = np.zeros(N)
    dxs = C_CMS / fs
    out = np.zeros(m)
    ug = np.zeros(m)
    U = 0.0
    lx = 0.0
    ly = 0.0
    nx = 0.0
    ny = 0.0
    uprev = 0.0
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    pb0, pb1, pa1 = _rad_coefs(nose[NN - 1] if NN > 0 else 1.0, fs)
    nu = MU / RHO
    for k in range(m):
        j = kidx[k]
        fr = kfrac[k]
        for i in range(N):
            Ap[i] = A[i]
            d0 = math.sqrt(shapes[j, i])
            d1 = math.sqrt(shapes[j + 1, i])
            a = (d0 + fr * (d1 - d0)) ** 2
            A[i] = a if a > A_MIN_CM2 else A_MIN_CM2
            sA[i] = math.sqrt(A[i])
            if k == 0:
                Ap[i] = A[i]
        ap = port[k] if port[k] > 1e-4 else 1e-4
        # ---- glottis
        At = trach[0]
        pt = TU[0] * sz / math.sqrt(At)
        po = L[0] * sz / sA[0]
        a = ag[k]
        if a < 1e-6:
            U = 0.0
        else:
            Rv = 12.0 * MU * GLOTTIS_LEN_CM ** 2 * GLOTTIS_THICK_CM / (a * a * a)
            kb = KC * RHO / (2.0 * a * a)
            if jet > 0:
                W = GLOTTIS_LEN_CM
                if jet == 2:
                    ar = a + 21.8 * W * max(U * U / a, 1e-9) ** (-1.0 / 3.0) * nu ** (2.0 / 3.0) * GLOTTIS_THICK_CM ** (2.0 / 3.0)
                else:
                    ar = (a / W + 2.0 * JET_SPREAD * GLOTTIS_THICK_CM) * W
                ar = min(max(ar, a), max(A[0], a))
                r = a / ar
                kb -= RHO / (2.0 * a * a) * 2.0 * r * (1.0 - r)
            Lg = RHO * GLOTTIS_THICK_CM / a
            B = Rv + zc / At + zc / A[0] + Lg * fs
            D = 2.0 * pt - 2.0 * po + Lg * fs * U
            if kb < 1e-12:
                U = D / B
            else:
                U = math.copysign((-B + math.sqrt(B * B + 4.0 * kb * abs(D))) / (2.0 * kb), D)
        ug[k] = U
        nR[0] = (po + zc / A[0] * U) * sA[0] / sz
        nTD[0] = (pt - zc / At * U) * math.sqrt(At) / sz
        # ---- turbulence: aspiration above the glottis, frication past the narrowest oral section
        w0 = noise[k]
        wf = nb[0] * w0 + nb[1] * x1 + nb[2] * x2 - nb[3] * y1 - nb[4] * y2
        x2 = x1
        x1 = w0
        y2 = y1
        y1 = wf
        ps_asp = 0.0
        if a > 1e-6:
            re = 2.0 * abs(U) / (GLOTTIS_LEN_CM * nu)
            if re > RE_CRIT:
                ps_asp = NOISE_K * (re * re - RE_CRIT * RE_CRIT) * wf
        imin = asp_at + 1
        for i in range(asp_at + 1, N):
            if A[i] < A[imin]:
                imin = i
        ps_fr = 0.0
        jfr = imin + 1 if imin + 1 < N else imin       # a constriction at the lips sounds at its own inlet
        if A[imin] < VOWEL_MIN_CM2:
            re = abs(U) * math.sqrt(4.0 / math.pi) / (nu * sA[imin])
            if re > RE_CRIT:
                ps_fr = NOISE_K * (re * re - RE_CRIT * RE_CRIT) * wf
        # ---- oral junctions (2-port, or 3/4-port at the velum and the piriform sinuses)
        for i in range(1, N):
            ia = R[i - 1]
            ib = L[i]
            num = ia * sA[i - 1] + ib * sA[i]
            den = A[i - 1] + A[i]
            pn_ = 0.0
            pp_ = 0.0
            if i == nose_at:
                pn_ = NLw[0]
                num += pn_ * math.sqrt(ap)
                den += ap
            if i == pir_at:
                pp_ = PL[0]
                num += pp_ * math.sqrt(pir_a)
                den += pir_a
            PJ = 2.0 * num / den
            nR[i] = PJ * sA[i] - ib
            nL[i - 1] = PJ * sA[i - 1] - ia
            if i == nose_at:
                nNR[0] = PJ * math.sqrt(ap) - pn_
            if i == pir_at:
                nPR[0] = PJ * math.sqrt(pir_a) - pp_
            if i == asp_at and ps_asp != 0.0:
                nR[i] += 0.5 * ps_asp * sA[i] / sz
                nL[i - 1] -= 0.5 * ps_asp * sA[i - 1] / sz
            if i == jfr and ps_fr != 0.0:
                nR[i] += 0.5 * ps_fr * sA[i] / sz
                nL[i - 1] -= 0.5 * ps_fr * sA[i - 1] / sz
            # moving walls: a section that widens draws the flow -X dA/dt into its left junction
            dP = -dxs * (A[i] - Ap[i]) * fs * zc / (A[i - 1] + A[i])
            nR[i] += dP * sA[i] / sz
            nL[i - 1] += dP * sA[i - 1] / sz
        # ---- lips
        al = A[N - 1]
        b0, b1, a1 = _rad_coefs(al, fs)
        pr = R[N - 1] * sz / sA[N - 1]
        pl = b0 * pr + b1 * lx - a1 * ly
        lx = pr
        ly = pl
        nL[N - 1] = pl * sA[N - 1] / sz
        ulip = (pr - pl) * al / zc
        # ---- nose: puff at the port while the velum shuts, junctions, nostrils
        unose = 0.0
        if NN > 0:
            if port[k] > 1e-4 and port[k] < VOWEL_MIN_CM2 and k > 0 and port[k] < port[k - 1]:
                up = abs(U) * ap / (ap + A[nose_at])
                re = up * math.sqrt(4.0 / math.pi) / (nu * math.sqrt(ap))
                if re > RE_CRIT:
                    nNR[0] += 0.5 * NOISE_K * (re * re - RE_CRIT * RE_CRIT) * wf * math.sqrt(ap) / sz
            for i in range(1, NN):
                a0_ = ap if i == 1 else nose[i - 1]
                PJ = 2.0 * (NR[i - 1] * math.sqrt(a0_) + NLw[i] * math.sqrt(nose[i])) / (a0_ + nose[i])
                nNR[i] = PJ * math.sqrt(nose[i]) - NLw[i]
                nNL[i - 1] = PJ * math.sqrt(a0_) - NR[i - 1]
            an = nose[NN - 1]
            pr = NR[NN - 1] * sz / math.sqrt(an)
            pl = pb0 * pr + pb1 * nx - pa1 * ny
            nx = pr
            ny = pl
            nNL[NN - 1] = pl * math.sqrt(an) / sz
            unose = (pr - pl) * an / zc
        # ---- piriform: closed far end
        for i in range(1, pir_n):
            nPR[i] = PR[i - 1]
            nPL[i - 1] = PL[i]
        if pir_n > 0:
            nPL[pir_n - 1] = PR[pir_n - 1]
        # ---- trachea, lungs at the bottom
        for i in range(1, NT):
            PJ = 2.0 * (TD[i - 1] * math.sqrt(trach[i - 1]) + TU[i] * math.sqrt(trach[i])) / (trach[i - 1] + trach[i])
            nTD[i] = PJ * math.sqrt(trach[i]) - TU[i]
            nTU[i - 1] = PJ * math.sqrt(trach[i - 1]) - TD[i - 1]
        ab = trach[NT - 1]
        pd = TD[NT - 1] * sz / math.sqrt(ab)
        nTU[NT - 1] = (r_lung * pd + (1.0 - r_lung) * 0.5 * psub[k]) * math.sqrt(ab) / sz
        # ---- one tick of travel, with losses
        for i in range(N):
            g = 1.0 - VISC_CM / math.sqrt(A[i] / math.pi) - WALL_LOSS - 2.0 * math.pi * MU * dxs / (zc * A[i])
            g = g if g > 0.9 else 0.9
            R[i] = nR[i] * g
            L[i] = nL[i] * g
        for i in range(NT):
            TU[i] = nTU[i]
            TD[i] = nTD[i]
        for i in range(NN):
            NR[i] = nNR[i] * (1.0 - WALL_LOSS)
            NLw[i] = nNL[i] * (1.0 - WALL_LOSS)
        for i in range(pir_n):
            PR[i] = nPR[i]
            PL[i] = nPL[i]
        utot = ulip + unose
        out[k] = (utot - uprev) * fs
        uprev = utot
    return out, ug


def glottal_area(f0, fs: float, amax: float = 0.08, oq: float = 0.6, sq: float = 2.4, leak: float = 0.0):
    """Glottal area (cm^2) per sample for an f0 track: cosine opening over OQ SQ / (1 + SQ) of the period, quarter
    cosine closing over OQ / (1 + SQ), closed the rest; ``leak`` a gap always open."""
    ph = np.cumsum(np.atleast_1d(np.asarray(f0, float))) / fs % 1.0
    to, tc = oq * sq / (1.0 + sq), oq / (1.0 + sq)
    a = np.where(ph < to, 0.5 * (1.0 - np.cos(np.pi * ph / to)),
                 np.where(ph < to + tc, np.cos(0.5 * np.pi * (ph - to) / tc), 0.0))
    return amax * a + leak


def _shape(s, speaker):
    if isinstance(s, str):
        if s.lower() in ("f", "v", "th", "dh", "s", "z", "sh", "zh", "θ", "ʃ"):
            a, d = fricative_area(s)
            return a, a.size * d
        return vowel_area(s, speaker), vowel_length(s, speaker) * 100.0
    a = np.asarray(s, float)
    return a, a.size * DX_CM


def _track(v, m):
    v = np.atleast_1d(np.asarray(v, float))
    return np.interp(np.linspace(0.0, 1.0, m), np.linspace(0.0, 1.0, v.size), v)


def kl_voice(shapes, f0=110.0, dur: float = 0.3, *, times=None, sr: int = 44100, port=0.0, speaker: str = "male",
             piriform: float = PIRIFORM_VOLUME, subglottal: bool = True, p_sub: float = 6000.0, ramp: float = 0.02,
             amax: float = 0.08, oq: float = 0.6, sq: float = 2.4, leak: float = 0.0, jet: str = "none",
             noise: bool = True, noise_cut: float = 6000.0, seed: int = 0) -> np.ndarray:
    """Time-domain voice through the area functions ``shapes`` (vowel or fricative names, or arrays in cm^2),
    reached at ``times`` (s) and moved between by minimum-jerk trajectories (in diameter). ``f0`` Hz, an array per
    output sample or (start, end) for a line. ``port`` velum opening cm^2 (scalar or per sample). Lungs at ``p_sub``
    dyn/cm^2 reached by a cosine ramp of ``ramp`` s. ``jet`` none | uniform | laminar | turbulent: pressure
    recovered past the glottis. Radiated pressure (d/dt of lip + nostril flow), peak 0.9."""
    from scipy.signal import butter, resample_poly
    fs = 2.0 * sr
    dx = C_CMS / fs
    shapes = [shapes] if isinstance(shapes, (str, np.ndarray)) else list(shapes)
    raw = [_shape(s, speaker) for s in shapes]
    N = max(8, int(round(float(np.mean([L for _, L in raw])) / dx)))
    S = np.array([np.interp((np.arange(N) + 0.5) / N, (np.arange(a.size) + 0.5) / a.size, a) for a, _ in raw])
    S = np.ascontiguousarray(np.vstack([S, S[-1:]]))
    m = int(round(dur * fs))
    t = np.arange(m) / fs
    K = len(raw)
    times = np.linspace(0.0, 0.6 * dur, K) if times is None else np.asarray(times, float)
    kidx = np.clip(np.searchsorted(times, t, side="right") - 1, 0, K - 1).astype(np.int64)
    nxt = np.minimum(kidx + 1, K - 1)
    x = np.clip((t - times[kidx]) / np.maximum(times[nxt] - times[kidx], 1e-9), 0.0, 1.0)
    kfrac = np.where(kidx < K - 1, min_jerk(x, 1.0), 0.0)
    f0 = _track((np.linspace(f0[0], f0[1], m) if isinstance(f0, tuple) else f0), m)
    ag = glottal_area(f0, fs, amax, oq, sq, leak)
    ps = float(p_sub) * np.where(t < ramp, 0.5 * (1.0 - np.cos(np.pi * t / max(ramp, 1e-9))), 1.0)
    pt = _track(port, m)
    nose = nasal_area(1.0, dx) if np.any(pt > 1e-4) else np.zeros(0)
    trach = trachea_area(dx) if subglottal else np.full(1, 1e4)
    pir_n = PIRIFORM_SECTIONS if piriform > 0 else 0
    pir_a = float(piriform) / (PIRIFORM_SECTIONS * DX_CM) if piriform > 0 else 1.0
    wn = np.random.default_rng(seed).uniform(-1.0, 1.0, m) if noise else np.zeros(m)
    b, a = butter(2, min(noise_cut, 0.45 * fs) / (0.5 * fs))
    jets = {"none": 0, "uniform": 0, "laminar": 2, "turbulent": 3}
    if jet not in jets:
        raise ValueError(f"jet {jet!r}: none, uniform, laminar or turbulent")
    y, _ = _kl_kernel(S, kidx, kfrac, pt, ag, ps, wn, trach, nose, pir_a, pir_n,
                      int(round(VELUM_AT * DX_CM / dx)) if nose.size else -1,
                      int(round(PIRIFORM_AT * DX_CM / dx)) if pir_n else -1,
                      int(round(ASPIRATION_CM / dx)), R_LUNG if subglottal else -1.0, fs,
                      np.array([b[0], b[1], b[2], a[1], a[2]]), jets[jet])
    y = resample_poly(y, 1, 2)[:int(round(dur * sr))]
    return 0.9 * y / (np.max(np.abs(y)) + 1e-12)


def protocol(vowel: str, speaker: str = "male", dur: float = 0.3, **kw) -> np.ndarray:
    """Synthetic vowel of the identification protocol: trachea 0.3 -> 1.5 cm^2, lungs 0 -> 6000 dyn/cm^2 in 20 ms,
    glottal area peak 0.08 cm^2, SQ 2.4, OQ 0.6, f0 falling from +25 % to -25 % of 110 Hz (male) or 220 Hz
    (female)."""
    f = 110.0 if speaker == "male" else 220.0
    return kl_voice(vowel, (1.25 * f, 0.75 * f), dur, speaker=speaker, p_sub=6000.0, ramp=0.02, amax=0.08,
                    oq=0.6, sq=2.4, **kw)
