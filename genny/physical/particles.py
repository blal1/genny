# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Particle / stochastic-event synthesis: PhISEM, PhOLIES, granular footsteps, bubbles, drips.

Ports (see ``out/research`` notes 04, 06, 07 section 1, 11 Part A, 02 section 7, 05 section 5 and 8):

* ``shaker``: STK ``Shakers`` (Perry Cook, STK 4.x), ``stk/stk-master/src/Shakers.cpp`` and
  ``include/Shakers.h:150-301``, ported sample for sample into one numba kernel, all 23 presets
  verbatim (``STK_PRESETS``, each row carries its source line numbers). The optional
  ``excitation="noise"`` variant is the older noise-excited engine of Cook's book (Example 13.3,
  research 04 section 2.2) and Csound ``Opcodes/phisem.c`` (water drops = ``wuter``, phisem.c 955-1124).
* ``COOK_MEASURED``: Cook's measured walking gravels (book Ch.15 / "Modeling Bill's Gait", research 04
  section 2.4), usable as extra ``shaker`` presets.
* ``granular_footstep_texture``: Visell et al. aggregate-ground model (research 06 section 7.3):
  grain rate ``lambda = A u (1 + tanh(B u)) / 2`` with ``u = dF_L/dt`` (loading only), grain energies
  from a power law ``p(E) ~ E**gamma`` summing to a fixed step energy, each grain a resonant impact
  through the PhISEM resonators of the ground's preset (``GROUNDS``). Farnell's coherence rule
  (research 05 section 1 and 5.3: density *and* tone follow the foot pressure) moves the grain
  resonance with the force.
* ``bubble_field``: Moss et al. "Sounding liquids" section 3 (research 06 section 3): Minnaert
  bubbles, ``f0 * r = 3.29 m/s``, radiative damping, rising chirp xi = 0.1, amplitude eps * r,
  radius power law ``r**-alpha``.
* ``surfacing_bubble``: Farnell Practical 12, decoded ``single-bubble.pd`` (research 02 section 7).
* ``drip``: one STK water-drop voice (Shakers.h:199-228) preceded by Farnell's parabolic impact
  pulse (rain on water: impact click then bubble whistle, research 02 section 10).

Everything runs at 48 kHz. Constants tuned at another rate are converted with the ``rate_*``
helpers (PLAN rule 2): per-sample decays and pole radii ``x ** (fs_src / SR)``, per-sample event
probabilities ``p * fs_src / SR``.

Output levels are the raw levels of the ported engines (STK does not normalise either); loudness is
set downstream by ``core.master``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numba
import numpy as np

try:  # PLAN: import SR and rate helpers from core when it exists
    from .core import SR, rate_decay, rate_prob, rate_radius, rate_samples
except ImportError:  # TODO use core
    SR = 48000

    def rate_decay(d: float, fs_src: float) -> float:
        return d ** (fs_src / SR)

    def rate_radius(r: float, fs_src: float) -> float:
        return r ** (fs_src / SR)

    def rate_prob(p: float, fs_src: float) -> float:
        return p * fs_src / SR

    def rate_samples(n: float, fs_src: float) -> float:
        return n * SR / fs_src


STK_FS = 44100.0  # STK default sample rate (Stk.h), research 07 section 0

# Engine kinds
_NORMAL, _WATER, _RATCHET, _ANGKLUNG = 0, 1, 2, 3

# Shakers.h:196-197
MIN_ENERGY = 0.001
WATER_FREQ_SWEEP = 1.0001
# Shakers.cpp:663
MAX_SHAKE = 1.0


# --------------------------------------------------------------------------------------------
# Preset table: STK Shakers.cpp, verbatim, numbers at 44.1 kHz
# --------------------------------------------------------------------------------------------

def _p(kind, sound_decay, system_decay, gain, n_objects, freqs, radii, gains, vary_on, vary,
       decay_scale, eq, source, ratchet_delta=0.0, fs_src=STK_FS):
    return dict(kind=kind, sound_decay=sound_decay, system_decay=system_decay, gain=gain,
                n_objects=n_objects, freqs=tuple(freqs), radii=tuple(radii), gains=tuple(gains),
                vary_on=tuple(vary_on), vary=vary, decay_scale=decay_scale, eq=tuple(eq),
                ratchet_delta=ratchet_delta, fs_src=fs_src, source=source)


_MUG_F = (2123, 4518, 8856, 10753)          # Shakers.cpp:203
_MUG_R = (0.997, 0.997, 0.997, 0.997)       # Shakers.cpp:204
_MUG_G = (1.0, 0.8, 0.6, 0.4)               # Shakers.cpp:205


def _mug(coin_f, coin_r, coin_g, src):
    # Shakers.cpp:478-547 (mug: 197-205, coin rows as cited)
    return _p(_NORMAL, 0.97, 0.9995, 0.8, 3, _MUG_F + tuple(coin_f), _MUG_R + tuple(coin_r),
              _MUG_G + tuple(coin_g), (False,) * (4 + len(coin_f)), 0.0, 0.95, (1.0, 0.0, -1.0), src)


STK_PRESETS: dict[str, dict] = {
    # 0
    "Maraca": _p(_NORMAL, 0.95, 0.999, 4.0, 25, (3200,), (0.96,), (1.0,), (False,), 0.0, 0.97,
                 (1.0, -1.0, 0.0), "Shakers.cpp:56-64, 631-650"),
    # 1
    "Cabasa": _p(_NORMAL, 0.96, 0.997, 8.0, 512, (3000,), (0.7,), (1.0,), (False,), 0.0, 0.97,
                 (1.0, -1.0, 0.0), "Shakers.cpp:66-74, 263-281"),
    # 2
    "Sekere": _p(_NORMAL, 0.96, 0.999, 4.0, 64, (5500,), (0.6,), (1.0,), (False,), 0.0, 0.94,
                 (1.0, 0.0, -1.0), "Shakers.cpp:76-84, 282-300"),
    # 3 (fixed shell + 2 moving cymbal resonances)
    "Tambourine": _p(_NORMAL, 0.95, 0.9985, 1.0, 32, (2300, 5600, 8100), (0.96, 0.99, 0.99),
                     (0.1, 0.8, 1.0), (False, True, True), 0.05, 0.95, (1.0, 0.0, -1.0),
                     "Shakers.cpp:96-104, 301-321"),
    # 4
    "SleighBells": _p(_NORMAL, 0.97, 0.9994, 1.0, 32, (2500, 5300, 6500, 8300, 9800), (0.99,) * 5,
                      (1.0, 1.0, 1.0, 0.5, 0.3), (True,) * 5, 0.03, 0.9, (1.0, 0.0, -1.0),
                      "Shakers.cpp:106-114, 322-341"),
    # 5
    "BambooChimes": _p(_NORMAL, 0.9, 0.9999, 0.4, 1.2, (2800, 0.8 * 2800.0, 1.2 * 2800.0),
                       (0.995,) * 3, (1.0,) * 3, (True,) * 3, 0.2, 0.7, (1.0, 0.0, 0.0),
                       "Shakers.cpp:86-94, 342-361"),
    # 6
    "Sandpaper": _p(_NORMAL, 0.999, 0.999, 0.5, 128, (4500,), (0.6,), (1.0,), (False,), 0.0, 0.97,
                    (1.0, 0.0, -1.0), "Shakers.cpp:116-124, 362-380"),
    # 7 (Helmholtz + 4 metal resonances)
    "CokeCan": _p(_NORMAL, 0.97, 0.999, 0.5, 48, (370, 1025, 1424, 2149, 3596),
                  (0.99, 0.992, 0.992, 0.992, 0.992), (1.0, 1.8, 1.8, 1.8, 1.8), (False,) * 5, 0.0,
                  0.95, (1.0, 0.0, -1.0), "Shakers.cpp:126-134, 381-399"),
    # 8 Stix1 (PhOLIES, "walking on brittle sticks", phisem.c:40)
    "Sticks": _p(_NORMAL, 0.96, 0.998, 6.0, 2, (5500,), (0.6,), (1.0,), (False,), 0.0, 0.96,
                 (1.0, 0.0, -1.0), "Shakers.cpp:177-185, 400-418"),
    # 9 Crunch1 (PhOLIES, "like new fallen snow, or not", phisem.c:41)
    "Crunch": _p(_NORMAL, 0.95, 0.99806, 4.0, 7, (800,), (0.95,), (1.0,), (False,), 0.0, 0.96,
                 (1.0, -1.0, 0.0), "Shakers.cpp:187-195, 419-437"),
    # 10
    "BigRocks": _p(_NORMAL, 0.98, 0.9965, 4.0, 23, (6460,), (0.932,), (1.0,), (True,), 0.11, 0.95,
                   (1.0, 0.0, -1.0), "Shakers.cpp:232-240, 438-457"),
    # 11
    "LittleRocks": _p(_NORMAL, 0.98, 0.99586, 4.0, 1600, (9000,), (0.843,), (1.0,), (True,), 0.18,
                      0.95, (1.0, 0.0, -1.0), "Shakers.cpp:242-250, 458-477"),
    # 12-18
    "NextMug": _mug((), (), (), "Shakers.cpp:197-205, 478-498"),
    "PennyMug": _mug((11000, 5200, 3835), (0.999,) * 3, (1.0, 0.8, 0.5),
                     "Shakers.cpp:197-210, 478-507"),
    "NickelMug": _mug((5583, 9255, 9805), (0.9992,) * 3, (1.0, 0.8, 0.5),
                      "Shakers.cpp:197-205, 212-214, 508-515"),
    "DimeMug": _mug((4450, 4974, 9945), (0.9993,) * 3, (1.0, 0.8, 0.5),
                    "Shakers.cpp:197-205, 216-218, 516-523"),
    "QuarterMug": _mug((1708, 8863, 9045), (0.9995,) * 3, (1.0, 0.8, 0.5),
                       "Shakers.cpp:197-205, 220-222, 524-531"),
    "FrancMug": _mug((5583, 11010, 1917), (0.9995,) * 3, (0.7, 0.4, 0.3),
                     "Shakers.cpp:197-205, 224-226, 532-539"),
    "PesoMug": _mug((7250, 8150, 10060), (0.9996,) * 3, (1.0, 1.2, 0.7),
                    "Shakers.cpp:197-205, 228-230, 540-547"),
    # 19, 20 ratchets (no system decay)
    "Guiro": _p(_RATCHET, 0.95, 1.0, 0.4, 128, (2500, 4000), (0.97, 0.97), (1.0, 1.0), (False, False),
                0.0, 0.0, (1.0, 0.0, -1.0), "Shakers.cpp:146-153, 549-569", ratchet_delta=0.0001),
    "Wrench": _p(_RATCHET, 0.95, 1.0, 0.4, 128, (3200, 8000), (0.99, 0.992), (1.0, 1.0), (False, False),
                 0.0, 0.0, (1.0, 0.0, -1.0), "Shakers.cpp:155-162, 570-590", ratchet_delta=0.00015),
    # 21 (Shakers.h:199-228 for the drop logic)
    "WaterDrops": _p(_WATER, 0.95, 0.996, 1.0, 10, (450, 600, 750), (0.9985,) * 3, (1.0,) * 3,
                     (False,) * 3, 0.0, 0.8, (-1.0, 0.0, 1.0), "Shakers.cpp:164-172, 591-611; Shakers.h:199-228"),
    # 22 Tuned bamboo chimes (angklung)
    "TunedBambooChimes": _p(_ANGKLUNG, 0.95, 0.9999, 0.5, 1.2,
                            (1046.6, 1174.8, 1397.0, 1568, 1760, 2093.3, 2350), (0.996,) * 7, (1.0,) * 7,
                            (False,) * 7, 0.0, 0.7, (1.0, 0.0, -1.0), "Shakers.cpp:136-144, 612-630"),
}
STK_PRESET_ORDER = list(STK_PRESETS)  # index == STK instrument number (Shakers.cpp:22-45)

# Cook's measured walking gravels (book p.197-199, Gait p.4-5; research 04 section 2.4).
# N is in the units of the 22.05 kHz synthesis model the estimator was calibrated on (fs_src 22050);
# alpha_c 0.95 / alpha_s 0.999 are Cook Example 13.3 (22.05 kHz); EQ is Example 13.3's one-zero
# y[n]-y[n-1]. The recording rate behind r is not stated: research 04 says assume 44.1 kHz, so r is
# converted from 44.1 kHz separately (radius_fs). vary = std/mean of the LPC centre frequency, the same
# reading STK used for BigRocks (0.11 ~ 701/6460).
COOK_MEASURED: dict[str, dict] = {
    "CookLargeGravel": dict(_p(_NORMAL, 0.95, 0.999, 4.0, 23.22, (6460,), (0.932,), (1.0,), (True,),
                               701.0 / 6460.0, 0.95, (1.0, -1.0, 0.0),
                               "Cook 2002 p.197-199 / Gait Table: 6460+-701 Hz, r 0.932, N 23.22",
                               fs_src=22050.0), radius_fs=STK_FS),
    "CookSmallGravel": dict(_p(_NORMAL, 0.95, 0.999, 4.0, 1068.0, (12670,), (0.843,), (1.0,), (True,),
                               3264.0 / 12670.0, 0.95, (1.0, -1.0, 0.0),
                               "Cook 2002 p.197-199 / Gait Table: 12670+-3264 Hz, r 0.843, N 1068",
                               fs_src=22050.0), radius_fs=STK_FS),
}
# UNSOURCED: baseGain 4.0 for the Cook gravels (the book uses its own log4 gain formula); taken from
# the STK rock presets so levels match them.


def preset(name: str) -> dict:
    if name in STK_PRESETS:
        return STK_PRESETS[name]
    if name in COOK_MEASURED:
        return COOK_MEASURED[name]
    raise KeyError(f"unknown PhISEM preset {name!r}; known: {STK_PRESET_ORDER + list(COOK_MEASURED)}")


# --------------------------------------------------------------------------------------------
# The engine (numba). One kernel covers STK tick (Shakers.h:230-301), waterDrop (199-228),
# the ratchet branch, the angklung branch, the noise-excited variant (Cook Ex.13.3 / phisem.c)
# and externally scheduled events (granular footsteps).
# --------------------------------------------------------------------------------------------

@numba.njit(cache=True)
def _phisem_kernel(kind, noise_exc, ctrl, ctrl_level, use_ext, ext_amp, fscale,
                   Ds, Dn, p_event, G, base_f, radii, fgains, vary_on, vary,
                   eq0, eq1, eq2, sweep, ratchet_delta, ratchet_c, min_energy, max_shake,
                   water_j_max, seed):
    np.random.seed(seed)
    n = ctrl.shape[0]
    m = base_f.shape[0]
    two_pi_over_sr = 2.0 * np.pi / 48000.0
    out = np.zeros(n)
    a1 = np.empty(m)
    a2 = np.empty(m)
    y1 = np.zeros(m)
    y2 = np.zeros(m)
    gain = fgains.copy()
    tf = base_f.copy()
    for i in range(m):  # setResonance, Shakers.h:150-154
        a1[i] = -2.0 * radii[i] * np.cos(two_pi_over_sr * base_f[i])
        a2[i] = radii[i] * radii[i]
    if kind == 1:
        for i in range(m):
            gain[i] = G if noise_exc else 0.0   # phisem.c wuterset: gains = log(N)*G/N; STK: 0 until a drop
    E = 0.0
    L = 0.0
    s1 = 0.0
    s2 = 0.0
    for t in range(n):
        inp = 0.0
        itube = 0
        c = ctrl[t]
        if kind == 2:
            # ---------------- ratchet (Guiro / Wrench), Shakers.h:234-248 ----------------
            if c > 0.0:   # ratchetCount > 0 ; c = scrape speed (ratchetDelta = base * count)
                E -= ratchet_delta * c + ratchet_c * E
                if E < 0.0:
                    E = 1.0
                if np.random.random() < p_event:
                    L += E * E
                inp = L * (2.0 * np.random.random() - 1.0) * E
        else:
            if use_ext:
                ev = ext_amp[t]
                if ev > 0.0:
                    if kind == 1:
                        L = ev
                    else:
                        L += ev
                    fs_now = fscale[t]
                    if kind == 1:
                        j = np.random.randint(0, 3)
                        u = 2.0 * np.random.random() - 1.0
                        if j == 0 and gain[0] == 0.0:
                            tf[0] = base_f[1] * fs_now * (0.75 + 0.25 * u)
                            gain[0] = abs(2.0 * np.random.random() - 1.0)
                        elif j == 1 and gain[1] == 0.0:
                            tf[1] = base_f[1] * fs_now * (1.0 + 0.25 * u)
                            gain[1] = abs(2.0 * np.random.random() - 1.0)
                        elif gain[2] == 0.0:
                            tf[2] = base_f[1] * fs_now * (1.25 + 0.25 * u)
                            gain[2] = abs(2.0 * np.random.random() - 1.0)
                    else:
                        for i in range(m):
                            f = base_f[i] * fs_now
                            if vary_on[i]:
                                f *= 1.0 + vary * (2.0 * np.random.random() - 1.0)
                            a1[i] = -2.0 * radii[i] * np.cos(two_pi_over_sr * f)
                        if kind == 3:
                            itube = np.random.randint(0, 7)
                    if not noise_exc:
                        inp = L
                if noise_exc:
                    inp = L * (2.0 * np.random.random() - 1.0)
                elif kind == 1:
                    inp = L
            else:
                # energy input between ticks: noteOn / CC2 (Shakers.cpp:671-672, 700-701)
                if ctrl_level:
                    if c > E:
                        E = c
                else:
                    E += c
                if E > max_shake:
                    E = max_shake
                if E >= min_energy:            # Shakers.h:250
                    E *= Ds                    # Shakers.h:253
                    if kind == 1:
                        if noise_exc:
                            # phisem.c wuter 1060-1080
                            if np.random.random() < p_event:
                                L = E
                                j = np.random.randint(0, water_j_max + 1)
                                u = 2.0 * np.random.random() - 1.0
                                if j == 0:
                                    tf[0] = base_f[1] * (0.75 + 0.25 * u)
                                    gain[0] = abs(2.0 * np.random.random() - 1.0)
                                elif j == 1:
                                    tf[1] = base_f[1] * (1.0 + 0.25 * u)
                                    gain[1] = abs(2.0 * np.random.random() - 1.0)
                                else:
                                    tf[2] = base_f[1] * (1.25 + 0.25 * u)
                                    gain[2] = abs(2.0 * np.random.random() - 1.0)
                            inp = L * (2.0 * np.random.random() - 1.0)
                        else:
                            # Shakers.h:201-216
                            if np.random.random() < p_event:
                                L = E
                                j = np.random.randint(0, 3)
                                u = 2.0 * np.random.random() - 1.0
                                if j == 0 and gain[0] == 0.0:
                                    tf[0] = base_f[1] * (0.75 + 0.25 * u)
                                    gain[0] = abs(2.0 * np.random.random() - 1.0)
                                elif j == 1 and gain[1] == 0.0:
                                    tf[1] = base_f[1] * (1.0 + 0.25 * u)
                                    gain[1] = abs(2.0 * np.random.random() - 1.0)
                                elif gain[2] == 0.0:
                                    tf[2] = base_f[1] * (1.25 + 0.25 * u)
                                    gain[2] = abs(2.0 * np.random.random() - 1.0)
                            inp = L                 # Shakers.h:258
                    else:
                        if np.random.random() < p_event:   # randomFloat(1024) < nObjects
                            L += E                          # Shakers.h:262
                            if not noise_exc:
                                inp = L                     # Shakers.h:263
                            for i in range(m):              # Shakers.h:265-270
                                if vary_on[i]:
                                    f = base_f[i] * (1.0 + vary * (2.0 * np.random.random() - 1.0))
                                    a1[i] = -2.0 * radii[i] * np.cos(two_pi_over_sr * f)
                            if kind == 3:
                                itube = np.random.randint(0, 7)   # Shakers.h:271
                        if noise_exc:
                            inp = L * (2.0 * np.random.random() - 1.0)   # Cook Ex.13.3
                # else: STK returns 0 here; we keep ringing the resonators with zero input (no click)
        if kind == 1:
            # sweep the drop voices, Shakers.h:219-227 / phisem.c 1082-1099
            # (kept running below MIN_ENERGY too so a voice never freezes)
            for i in range(3):
                gain[i] *= radii[i]
                if gain[i] > 0.001:
                    tf[i] *= sweep
                    a1[i] = -2.0 * radii[i] * np.cos(two_pi_over_sr * tf[i])
                else:
                    gain[i] = 0.0
        L *= Dn                                 # Shakers.h:277
        # resonators, Shakers.h:156-163 and 279-292
        acc = 0.0
        for i in range(m):
            if kind == 1:
                if noise_exc:
                    x = inp * gain[i]
                else:
                    x = inp * gain[i] * G
            elif kind == 3:
                x = inp * gain[i] * G if i == itube else 0.0
            else:
                x = inp * gain[i] * G
            y = x - a1[i] * y1[i] - a2[i] * y2[i]
            y2[i] = y1[i]
            y1[i] = y
            if kind == 1 and noise_exc:
                acc += gain[i] * y          # phisem.c 1111-1121: data = sum gains_j * y_j
            else:
                acc += y
        if kind == 1 and noise_exc:
            acc *= 4.0 * 0.005              # phisem.c 1123-1127 (EQ taps -1, 0, 1 below)
        # tickEqualize, Shakers.h:172-179
        o = eq0 * acc + eq1 * s1 + eq2 * s2
        s2 = s1
        s1 = acc
        out[t] = o
    return out


def _engine_params(p: dict, n_objects, resonance_scale, damping):
    """Convert one preset to 48 kHz engine constants (PLAN rule 2)."""
    fs = p["fs_src"]
    rfs = p.get("radius_fs", fs)
    base_decay = p["system_decay"]
    if damping is not None and p["kind"] != _RATCHET:
        # CC4 system decay, Shakers.cpp:705
        base_decay = base_decay + 2.0 * (float(damping) - 0.5) * p["decay_scale"] * (1.0 - base_decay)
    N = float(p["n_objects"] if n_objects is None else n_objects)
    # currentGain_ = log(nObjects) * baseGain / nObjects (Shakers.cpp:657, 709)
    G = math.log(N) * p["gain"] / N
    if p["kind"] == _WATER:
        prob_src = N / 32767.0   # randomInt(32767) < nObjects, Shakers.h:201
    else:
        prob_src = N / 1024.0    # randomFloat(1024) < nObjects, Shakers.h:243, 261
    return dict(
        Ds=rate_decay(base_decay, fs),
        Dn=rate_decay(p["sound_decay"], fs),
        p_event=min(rate_prob(prob_src, fs), 1.0),
        G=G,
        base_f=np.asarray(p["freqs"], float) * float(resonance_scale),
        radii=np.array([rate_radius(r, rfs) for r in p["radii"]]),
        fgains=np.asarray(p["gains"], float),
        vary_on=np.asarray(p["vary_on"], np.bool_),
        vary=float(p["vary"]),
        eq=p["eq"],
        sweep=WATER_FREQ_SWEEP ** (fs / SR),
        ratchet_delta=rate_prob(p["ratchet_delta"], fs),         # linear per-sample ramp -> x fs/SR
        ratchet_c=1.0 - rate_decay(1.0 - 0.002, fs),             # the 0.002*E term is a decay
    )


def cook_shakes(dur: float, period: float = 5050 / 22050.0, shake_len: float = 0.05,
                amount: float = 1.0) -> np.ndarray:
    """Per-sample energy injection for repeated shakes (Cook Example 13.3, research 04 section 2.2).

    One shake is a raised cosine over 50 ms, repeated every 5050 samples at 22.05 kHz (0.229 s).
    UNSOURCED: each shake injects ``amount`` (default MAX_SHAKE) in total; Cook's book has its own
    unclamped energy scale.
    """
    n = int(round(dur * SR))
    env = np.zeros(n)
    m = max(2, int(round(shake_len * SR)))
    shape = 1.0 - np.cos(2 * np.pi * np.arange(m) / m)
    shape *= amount / shape.sum()
    step = int(round(period * SR))
    for s in range(0, n, step):
        k = min(m, n - s)
        env[s:s + k] += shape[:k]
    return env


def shaker(preset_name: str, dur: float, rng: np.random.Generator,
           energy_curve: np.ndarray | None = None, n_objects: float | None = None,
           resonance_scale: float = 1.0, *, energy_mode: str = "inject", damping: float | None = None,
           excitation: str = "impulse") -> np.ndarray:
    """Render one STK Shakers / PhISEM instrument (Shakers.h:230-301) at 48 kHz.

    preset_name: a key of ``STK_PRESETS`` (23 STK instruments) or ``COOK_MEASURED``.
    energy_curve: per-sample control of the system energy (length >= dur*SR, else zero-padded).
        ``energy_mode="inject"``: added to the energy every sample, like noteOn/CC2 (E += x, clamp 1),
        Cook Ex.13.3 drives it the same way. ``energy_mode="level"``: the curve is the energy the
        system is pushed up to (E = max(E, x)); it then falls at the preset's system decay.
        UNSOURCED convenience mapping, handy for a foot-pressure curve in [0, 1].
        For Guiro / Wrench the curve is the scrape speed (ratchetDelta = base * speed; STK CC2 sets
        ratchetCount = |controller change|, Shakers.cpp:693-697); 0 = no scraping.
        None: repeated Cook shakes (``cook_shakes``); ratchets scrape at base speed throughout.
    n_objects: STK CC11, the object count N (collision probability N/1024 per 44.1 kHz sample and
        gain log(N)*baseGain/N).
    resonance_scale: multiplies every resonance frequency (STK CC1 does x 4**(v-0.5)).
    damping: STK CC4 value in [0, 1]; system decay = base + 2(v-0.5)*decayScale*(1-base).
    excitation: "impulse" = STK 4.x (input is the sound level on collision samples only);
        "noise" = Cook's book Ex.13.3 / Csound phisem.c (input = level * white noise every sample;
        water drops follow phisem.c ``wuter``).
    """
    p = preset(preset_name)
    n = int(round(dur * SR))
    kind = p["kind"]
    if energy_curve is None:
        ctrl = np.ones(n) if kind == _RATCHET else cook_shakes(dur)
        level = False
    else:
        ctrl = np.zeros(n)
        e = np.asarray(energy_curve, float)[:n]
        ctrl[: len(e)] = e
        level = energy_mode == "level"
        if energy_mode not in ("inject", "level"):
            raise ValueError("energy_mode must be 'inject' or 'level'")
    if excitation not in ("impulse", "noise"):
        raise ValueError("excitation must be 'impulse' or 'noise'")
    noise = excitation == "noise"
    q = _engine_params(p, n_objects, resonance_scale, damping)
    p_event = q["p_event"]
    water_j_max = 2
    if kind == _WATER and noise:
        # phisem.c: my_random(32767) = rand % 32768 -> p = N/32768; my_random(3) -> 0..3 (j=3 -> voice 2)
        N = float(p["n_objects"] if n_objects is None else n_objects)
        p_event = rate_prob(N / 32768.0, p["fs_src"])
        water_j_max = 3
    seed = int(rng.integers(0, 2**31 - 1))
    dummy = np.zeros(1)
    return _phisem_kernel(kind, noise, ctrl, level, False, dummy, dummy,
                          q["Ds"], q["Dn"], p_event, q["G"], q["base_f"], q["radii"], q["fgains"],
                          q["vary_on"], q["vary"], q["eq"][0], q["eq"][1], q["eq"][2], q["sweep"],
                          q["ratchet_delta"], q["ratchet_c"], MIN_ENERGY, MAX_SHAKE, water_j_max, seed)


# --------------------------------------------------------------------------------------------
# Granular footsteps (Visell et al. + Farnell), driven by a force curve
# --------------------------------------------------------------------------------------------

# Ground -> PhISEM layers. The mapping table is research 11 Part A3 ("the mapping is inference; the
# numbers are Cook's"), plus the PhOLIES header comments (phisem.c:38-45) and Cook's measured gravel.
GROUNDS: dict[str, dict] = {
    "snow": dict(layers=("Crunch",), source="phisem.c:41 'Crunch1 (like new fallen snow)'; research 11 A3"),
    "sand": dict(layers=("Sandpaper",), source="research 11 A3 table: sand -> Sandpaper"),
    "gravel": dict(layers=("BigRocks",), source="Cook measured large gravel 6460+-701 Hz, r .932, N 23 "
                                                "(research 04 section 2.4) = STK BigRocks"),
    "pebbles": dict(layers=("LittleRocks",), source="research 11 A3 table: pebbles -> Little rocks"),
    "leaves": dict(layers=("Sekere",), source="research 11 A3 table: leaves, dry grass -> Stix1 / Sekere"),
    "grass": dict(layers=("Sekere",), source="research 11 A3 table: leaves, dry grass -> Stix1 / Sekere"),
    "twigs": dict(layers=("Sticks",), source="phisem.c:40 'Stix1 (walking on brittle sticks)'"),
    "shallow_water": dict(layers=("WaterDrops", "Maraca"),
                          source="research 11 A3 table: shallow water -> Water drops + Maraca-like splash"),
    "coral": dict(layers=("BigRocks",), source="research 11 A3 table: coral, bone -> Big rocks "
                                               "(lower freq / higher R is inference, not applied)"),
    "bone": dict(layers=("BigRocks",), source="research 11 A3 table: coral, bone -> Big rocks "
                                              "(lower freq / higher R is inference, not applied)"),
    # UNSOURCED: no source has an ash preset (research 11 A3); nearest sourced texture = Crunch.
    "ash": dict(layers=("Crunch",), source="UNSOURCED: nearest sourced preset Crunch", unsourced=True),
    # UNSOURCED: Farnell thesis 5.3 says use bubbles for mud but gives no mud numbers; nearest sourced
    # PhISEM preset = WaterDrops (a bubble model).
    "mud": dict(layers=("WaterDrops",), source="UNSOURCED: nearest sourced preset WaterDrops "
                                               "(Farnell T 5.3: mud -> bubbles)", unsourced=True),
}

# Visell: crumpling exponent -1.6 < gamma < -1.3 (Sethna & Dahmen); per-ground values not published.
VISELL_GAMMA = -1.45   # UNSOURCED midpoint of the published crumpling range
VISELL_B = 20.0        # UNSOURCED: B * u with u normalised to its maximum; (1+tanh)/2 is then ~ a step
GRAIN_E_MIN = 1e-3     # UNSOURCED: lower cut of the power law (60 dB energy span)
FORCE_LP_HZ = 300.0    # Visell: f_L = force below ~300 Hz


def _lowpass_force(force: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(2, FORCE_LP_HZ / (SR / 2), output="sos")
    return sosfilt(sos, force)


def _powerlaw(rng, n, lo, hi, gamma):
    """Samples of p(x) ~ x**gamma on [lo, hi] (inverse CDF)."""
    u = rng.random(n)
    if abs(gamma + 1.0) < 1e-9:
        return lo * (hi / lo) ** u
    a = gamma + 1.0
    return (lo ** a + u * (hi ** a - lo ** a)) ** (1.0 / a)


def granular_footstep_texture(force: np.ndarray, ground: str, rng: np.random.Generator, *,
                              energy: float | None = None, density: float = 1.0,
                              gamma: float = VISELL_GAMMA, tone_coherence: bool = True,
                              resonance_scale: float = 1.0, excitation: str = "impulse",
                              tail: float = 0.1, return_info: bool = False):
    """Aggregate-ground footstep texture from a ground-reaction-force curve (Visell section 3.3.2).

    force: per-sample GRF at 48 kHz (any units; e.g. body weights). Only its < 300 Hz part is used.
    ground: key of ``GROUNDS``.
    Grain times: non-homogeneous Poisson, ``lambda(t) = A u (1+tanh(B u))/2``, ``u = dF_L/dt`` - grains
      fire only while the foot is loading, in proportion to how fast the force rises (Visell eq. 1-2).
      A is set so the mean rate over the loading phase equals the preset's PhISEM collision rate
      (N/1024 per 44.1 kHz sample) times ``density`` (UNSOURCED bridge: Visell publishes no A).
    Grain energies: ``p(E) ~ E**gamma``, scaled so they sum to the step energy (``energy``, default
      max(force)); each grain adds sqrt(E_i) to the PhISEM sound level (Visell energy budget).
    Grains: resonant impacts through the preset's resonators, retuned per grain (STK vary factor),
      and, with ``tone_coherence``, moved by the force: f * 2**(Fn**2 - 0.5) (Farnell: grain density
      and tone rise with the GRF, gravel range x2 = 600..1200 Hz with f ~ GRF**2; centring the x2 range
      on the preset frequency is UNSOURCED).
    Returns len(force) + tail seconds of audio (and an info dict when ``return_info``).
    """
    if ground not in GROUNDS:
        raise KeyError(f"unknown ground {ground!r}; known: {list(GROUNDS)}")
    g = GROUNDS[ground]
    f = np.asarray(force, float)
    n_tail = int(round(tail * SR))
    fl = _lowpass_force(np.concatenate([f, np.zeros(n_tail)]))
    fmax = float(np.max(np.abs(fl))) or 1.0
    fn = np.clip(fl / fmax, 0.0, None)
    u = np.gradient(fn) * SR                         # normalised force rate, 1/s
    umax = float(np.max(u))
    n = len(fn)
    E_step = float(np.max(f)) if energy is None else float(energy)
    layers = g["layers"]
    out = np.zeros(n)
    info = dict(ground=ground, layers=layers, source=g["source"], unsourced=g.get("unsourced", False),
                grains={}, grains_while_unloading=0)
    if umax <= 0:
        return (out, info) if return_info else out
    lam0 = u * (1.0 + np.tanh(VISELL_B * u / umax)) / 2.0
    lam0 = np.clip(lam0, 0.0, None)
    loading = u > 0.02 * umax
    t_load = loading.sum() / SR
    integ = lam0.sum() / SR
    fscale = 2.0 ** (fn ** 2 - 0.5) if tone_coherence else np.ones(n)
    noise = excitation == "noise"
    for name in layers:
        p = preset(name)
        q = _engine_params(p, None, resonance_scale, None)
        rate_N = q["p_event"] * SR                    # PhISEM collisions per second at 48 kHz
        A = rate_N * density * t_load / integ
        prob = np.clip(A * lam0 / SR, 0.0, 1.0)
        ev = rng.random(n) < prob
        k = int(ev.sum())
        amps = np.zeros(n)
        if k:
            e = _powerlaw(rng, k, GRAIN_E_MIN, 1.0, gamma)
            e *= (E_step / len(layers)) / e.sum()     # UNSOURCED: equal energy split between layers
            amps[ev] = np.sqrt(e)
        info["grains"][name] = k
        info["grains_while_unloading"] += int((ev & (u <= 0)).sum())
        seed = int(rng.integers(0, 2**31 - 1))
        y = _phisem_kernel(p["kind"], noise, np.zeros(n), False, True, amps, fscale,
                           q["Ds"], q["Dn"], q["p_event"], 1.0, q["base_f"], q["radii"], q["fgains"],
                           q["vary_on"], q["vary"], q["eq"][0], q["eq"][1], q["eq"][2], q["sweep"],
                           q["ratchet_delta"], q["ratchet_c"], MIN_ENERGY, MAX_SHAKE, 2, seed)
        out += y
    return (out, info) if return_info else out


# --------------------------------------------------------------------------------------------
# Bubbles
# --------------------------------------------------------------------------------------------

MINNAERT_FR = 3.29        # f0 * r0 for air in water, m/s (Moss eq. 1, derived in research 06 3.1)
DELTA_RAD = 0.0139        # radiative damping, water (derived in research 06 3.1)
DELTA_TH = 0.02           # UNSOURCED: thermal damping, G_th not given by Moss; research 06 sketch value
MOSS_XI = 0.1             # rising-chirp constant (Moss, after van den Doel 2005)


@numba.njit(cache=True)
def _bubble_sum(n, starts, f0s, betas, amps, xi):
    out = np.zeros(n)
    inv_sr = 1.0 / 48000.0
    for k in range(starts.shape[0]):
        s = starts[k]
        f0 = f0s[k]
        b0 = betas[k]
        a = amps[k]
        m = int(6.91 / b0 * 48000.0) + 1           # to -60 dB
        if s + m > n:
            m = n - s
        for i in range(m):
            t = i * inv_sr
            ph = 2.0 * np.pi * f0 * (t + xi * b0 * t * t * 0.5)   # f(t) = f0 (1 + xi b0 t)
            out[s + i] += a * np.sin(ph) * np.exp(-b0 * t)
    return out


def bubble_field(dur: float, rng: np.random.Generator, rate: float | np.ndarray = 50.0,
                 alpha: float = 2.9, r_range: tuple[float, float] = (0.5e-3, 5e-3),
                 eps: tuple[float, float, float] = (0.01, 0.1, 2.0), *, xi: float = MOSS_XI,
                 delta_th: float = DELTA_TH, return_info: bool = False):
    """Moss et al. bubble ensemble (research 06 section 3.1, 3.3).

    rate: bubbles per second (Poisson), a number or a per-sample envelope (Moss: formation follows
      Gamma = u**2 kappa, i.e. bursts at crests / fast flow - pass that envelope here).
    alpha: radius power law p(r) ~ r**-alpha (surf 1.5-3.3, rain 2.9), truncated to r_range (metres).
    eps: (lo, hi, mu) initial displacement, power law g(eps) ~ eps**-mu on [lo, hi] (Moss: [0.01, 0.1],
      most below 10 %; mu = 2 is UNSOURCED, research 06 sketch).
    Each bubble: p(t) = eps r sin(2 pi f0 (t + xi b0 t^2/2)) exp(-b0 t), f0 = 3.29/r,
      b0 = pi f0 (delta_rad + delta_th). Amplitude is eps * r with r in millimetres (scale choice).
    Onsets are spread at random sample positions (Moss: avoid in-phase artefacts).
    """
    n = int(round(dur * SR))
    if np.ndim(rate) == 0:
        lam = np.full(n, float(rate))
    else:
        lam = np.zeros(n)
        r_ = np.asarray(rate, float)[:n]
        lam[: len(r_)] = r_
    starts = np.flatnonzero(rng.random(n) < np.clip(lam / SR, 0, 1)).astype(np.int64)
    k = len(starts)
    r = _powerlaw(rng, k, r_range[0], r_range[1], -alpha)
    e = _powerlaw(rng, k, eps[0], eps[1], -eps[2])
    f0 = MINNAERT_FR / r
    beta = np.pi * f0 * (DELTA_RAD + delta_th)
    amp = e * r * 1e3
    keep = f0 < 0.45 * SR
    y = _bubble_sum(n, starts[keep], f0[keep], beta[keep], amp[keep], float(xi))
    if return_info:
        return y, dict(n=int(keep.sum()), f0=f0[keep], r=r[keep])
    return y


def _onepole_lp_hz(x: np.ndarray, hz: float) -> np.ndarray:
    """Pd ``lop~``: y += k (x - y), k = 2 pi f / SR."""
    from scipy.signal import lfilter
    k = min(2 * np.pi * hz / SR, 1.0)
    return lfilter([k], [1.0, -(1.0 - k)], x)


def surfacing_bubble(rng: np.random.Generator, dim: float = 0.5, dur: float = 0.09,
                     fmin: float = 60.0, fmax: float = 300.0, sweep: float = 2000.0, vol: float = 1.0,
                     mydim: float = 1.0, mydur: float = 1.0) -> np.ndarray:
    """Farnell surfacing bubble, decoded ``generatore_di_bolle/single-bubble.pd`` (research 02 section 7).

    D = dim (1 - mydim U[0,1)); dur' = dur (1 + mydur U[-.5,.5)); T = D dur'
    f(t) = fmin + (1 - D) fmax + sweep (1 - (1 - t/T)**4)
    a(t) = lop~12((1 - t/T)**8) (D + 0.1) vol
    Defaults are the patch defaults (dur 90 ms, range 2000 Hz, fmin 60, fmax 300, dim 0.5).
    The lop~ tail is rendered for 4 time constants after T with the pitch held.
    """
    D = dim * (1.0 - mydim * rng.random())
    D = max(D, 1e-3)
    d = dur * (1.0 + mydur * (rng.random() - 0.5))
    T = D * d
    m = max(int(T * SR), 8)
    tail = int(4 * SR / (2 * np.pi * 12.0))
    t = np.arange(m + tail) / SR
    x = np.clip(t / T, 0.0, 1.0)
    l = 1.0 - x
    f = fmin + (1.0 - D) * fmax + sweep * (1.0 - l ** 4)
    a = _onepole_lp_hz(l ** 8, 12.0) * (D + 0.1) * vol
    return a * np.sin(2 * np.pi * np.cumsum(f) / SR)


def drip(rng: np.random.Generator, f0: float | None = None, *, voice: int | None = None,
         click: bool = True, click_ms: float | None = None, click_level: float = 0.3,
         dur: float = 0.35) -> np.ndarray:
    """One water drop: STK waterDrop voice (Shakers.h:199-228) after a Farnell impact pulse.

    The voice: level L = 1 on the drop sample, decaying by soundDecay 0.95; the resonator's input gain
    |u| decays by r = 0.9985 per sample while its frequency rises x1.0001 per sample (44.1 kHz values,
    converted); r is the WaterDrops radius. f0 None: STK's choice - voice j (random 0..2) starts at
    600 * (0.75 | 1.0 | 1.25 + 0.25 u).
    click: Farnell rain on water (research 02 section 10): impact click, then the bubble whistle; the
      click is a parabolic pulse 1-4(x-.5)^2 of width 0.1 + 12 U ms with amplitude sqrt(w) + 0.1
      (Poisson-click rain). UNSOURCED: click_level (Moss: the impact is far quieter than the bubble).
    """
    p = STK_PRESETS["WaterDrops"]
    q = _engine_params(p, None, 1.0, None)
    n = int(round(dur * SR))
    if voice is None:
        voice = int(rng.integers(0, 3))
    u = rng.uniform(-1, 1)
    centre = (0.75, 1.0, 1.25)[voice]
    f_start = p["freqs"][1] * (centre + 0.25 * u) if f0 is None else float(f0)
    g0 = abs(rng.uniform(-1, 1)) if f0 is None else 1.0
    r = q["radii"][voice]
    Dn, sweep = q["Dn"], q["sweep"]
    y = _drip_voice(n, f_start, g0, r, Dn, sweep, q["G"])
    # EQ (-1, 0, 1), Shakers.cpp:610
    y = -y + np.concatenate([np.zeros(2), y[:-2]])
    if click:
        w = (0.1 + 12.0 * rng.random()) * 1e-3 if click_ms is None else click_ms * 1e-3
        mlen = max(2, int(w * SR))
        xx = np.linspace(0, 1, mlen)
        pulse = (1 - 4 * (xx - 0.5) ** 2) * (np.sqrt(w) + 0.1)
        pulse *= click_level * np.max(np.abs(y)) / np.max(pulse)
        y = np.concatenate([pulse, y])[:n]
    return y


@numba.njit(cache=True)
def _drip_voice(n, f_start, g0, r, Dn, sweep, G):
    out = np.zeros(n)
    tw = 2.0 * np.pi / 48000.0
    a2 = r * r
    f = f_start
    a1 = -2.0 * r * np.cos(tw * f)
    gain = g0
    L = 1.0
    y1 = 0.0
    y2 = 0.0
    for t in range(n):
        gain *= r
        if gain > 0.001:
            f *= sweep
            a1 = -2.0 * r * np.cos(tw * f)
        else:
            gain = 0.0
        inp = L
        L *= Dn
        y = inp * gain * G - a1 * y1 - a2 * y2
        y2 = y1
        y1 = y
        out[t] = y
    return out


# --------------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------------

def _farnell_grf(dur: float, n_heel=3.3333, n_roll=3.3333, n_ball=3.3333) -> np.ndarray:
    """One foot contact, Farnell GRF polynomial (research 03 2.4-2.5, 05 5.2)."""
    x = np.linspace(0, 1, int(dur * SR))
    x = np.minimum(x / 0.75, 1.0)

    def poly(v, N):
        v = np.clip(v, 0, 1)
        return 1.5 * N * v * (1 - v) ** 2 * (1 + v)
    h = poly(np.clip(x, 0, 1 / 3) * 3, n_heel)
    r = poly((np.clip(x, .125, .875) - .125) / .75, n_roll)
    b = poly((np.clip(x, 2 / 3, 1) - 2 / 3) * 3, n_ball)
    g = h + r + b
    return g / g.max()


def _spectrum(y):
    from scipy.signal import welch
    f, P = welch(y, SR, nperseg=4096)
    return f, P


def _analyse(y):
    from scipy.signal import find_peaks
    f, P = _spectrum(y)
    band = f > 100
    cen = float((f[band] * P[band]).sum() / max(P[band].sum(), 1e-300))
    Ps = np.convolve(P, np.ones(3) / 3, mode="same")
    main = float(f[band][np.argmax(Ps[band])])
    db = 10 * np.log10(Ps + 1e-30)
    pk, prop = find_peaks(db, prominence=6)
    pk = [i for i in pk if f[i] > 100]
    pk = sorted(pk, key=lambda i: -db[i])[:5]
    return cen, main, sorted(float(f[i]) for i in pk)


def _theory_peak(p: dict):
    """Peak of |EQ(z)| * |1/(1 + a1 z^-1 + a2 z^-2)| for a one-resonance preset at 48 kHz."""
    if len(p["freqs"]) != 1:
        return None
    q = _engine_params(p, None, 1.0, None)
    f = np.linspace(50, SR / 2 - 50, 20000)
    z = np.exp(-1j * 2 * np.pi * f / SR)
    r, f0 = q["radii"][0], q["base_f"][0]
    den = 1 - 2 * r * np.cos(2 * np.pi * f0 / SR) * z + r * r * z * z
    eq = q["eq"][0] + q["eq"][1] * z + q["eq"][2] * z * z
    return float(f[np.argmax(np.abs(eq / den))])


def _write(path: Path, y: np.ndarray):
    import soundfile as sf
    pk = float(np.max(np.abs(y))) or 1.0
    sf.write(str(path), (y / pk * 10 ** (-3 / 20)).astype(np.float32), SR, subtype="PCM_24")


def _inst_freq(y, t0, t1):
    """Dominant frequency in [t0, t1) s by zero-padded FFT peak."""
    seg = y[int(t0 * SR): int(t1 * SR)]
    seg = seg * np.hanning(len(seg))
    nfft = 1 << 18
    S = np.abs(np.fft.rfft(seg, nfft))
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    S[fr < 100] = 0
    return float(fr[np.argmax(S)])


def _selftest():
    out_dir = Path(__file__).resolve().parent.parent / "out" / "dev" / "particles"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    ok = True
    print(f"SR {SR}; writing to {out_dir}")
    print(f"{'preset':<18}{'raw peak':>10}{'centroid':>10}{'main pk':>9}  {'source f (Hz)':<34} peaks found (Hz)")
    names = STK_PRESET_ORDER + list(COOK_MEASURED)
    for name in names:
        p = preset(name)
        if p["kind"] == _WATER:
            y = shaker(name, 2.0, rng, energy_curve=np.ones(2 * SR), energy_mode="level")
        else:
            y = shaker(name, 2.0, rng)
        finite = bool(np.all(np.isfinite(y)))
        nz = bool(np.max(np.abs(y)) > 0)
        ok &= finite and nz
        cen, main, peaks = _analyse(y)
        src = ", ".join(f"{v:g}" for v in p["freqs"])
        near = min(abs(main / fs - 1) for fs in np.asarray(p["freqs"]))
        flag = ""
        if near >= 0.15:
            th = _theory_peak(p)
            if th is not None and abs(main / th - 1) < 0.2:
                # broad (r <= 0.7) poles: the EQ zeros move the peak and its top is flat
                flag = f"  (~ analytic STK res+EQ peak {th:.0f} Hz, broad pole)"
            elif p["kind"] == _WATER:
                flag = "  (drops sweep upward; see centroid)"
            else:
                flag = "  <-- main peak off source list"
        if not finite or not nz:
            flag += "  FAIL (nan/silent)"
        print(f"{name:<18}{np.max(np.abs(y)):>10.3g}{cen:>10.0f}{main:>9.0f}  {src:<34} "
              f"{[round(v) for v in peaks]}{flag}")
        _write(out_dir / f"shaker_{name}.wav", y)

    # noise-excited variants (Cook book / phisem.c) for the footstep presets and water
    print("\nnoise-excited variant (Cook Ex.13.3 / phisem.c):")
    for name in ("Maraca", "Crunch", "BigRocks", "WaterDrops"):
        if name == "WaterDrops":
            y = shaker(name, 2.0, rng, energy_curve=np.ones(2 * SR), energy_mode="level", excitation="noise")
        else:
            y = shaker(name, 2.0, rng, excitation="noise")
        cen, main, peaks = _analyse(y)
        ok &= bool(np.all(np.isfinite(y)))
        print(f"  {name:<14} centroid {cen:6.0f} Hz  main {main:6.0f} Hz  peak {np.max(np.abs(y)):.3g}")
        _write(out_dir / f"shaker_noise_{name}.wav", y)

    # water-drop voices: start frequency and upward sweep (source: 450/600/750 centres, x1.0001/sample)
    print("\nwater-drop voices (STK: centres 450/600/750 Hz, sweep 1.0001/sample @44.1k = x e^(4.41 t)):")
    for v in range(3):
        y = drip(rng, voice=v, click=False)
        # the actual start frequency: re-derive with the same draw by using a fixed f0 at the centre
        yc = drip(rng, f0=600 * (0.75, 1.0, 1.25)[v], voice=v, click=False)
        fa = _inst_freq(yc, 0.0, 0.02)
        fb = _inst_freq(yc, 0.04, 0.06)
        expect = math.exp(0.0001 * 44100 * 0.04)
        rising = fb > fa
        ok &= rising
        print(f"  voice {v}: centre {600 * (0.75, 1.0, 1.25)[v]:.0f} Hz, f(0-20ms) {fa:6.0f} Hz, "
              f"f(40-60ms) {fb:6.0f} Hz, ratio {fb / fa:.3f} (expected ~{expect:.3f}) rising={rising}")
        _write(out_dir / f"drip_voice{v}.wav", np.concatenate([y, yc]))
    yd = np.concatenate([drip(rng) for _ in range(6)])
    _write(out_dir / "drips.wav", yd)

    # granular footsteps
    print("\ngranular footsteps (Visell rate ~ dF/dt while loading; Farnell GRF 0.6 s):")
    grf = _farnell_grf(0.6)
    for ground, g in GROUNDS.items():
        y, info = granular_footstep_texture(grf, ground, rng, return_info=True)
        cen, main, peaks = _analyse(y)
        ok &= bool(np.all(np.isfinite(y))) and info["grains_while_unloading"] == 0
        srcf = "/".join(f"{preset(l)['freqs'][0]:g}" for l in g["layers"])
        tag = " UNSOURCED" if g.get("unsourced") else ""
        print(f"  {ground:<14} layers {'+'.join(g['layers']):<20} grains {info['grains']} "
              f"unloading {info['grains_while_unloading']}  centroid {cen:6.0f}  main {main:6.0f} "
              f"(preset f {srcf}){tag}")
        steps = np.concatenate([granular_footstep_texture(grf * (0.8 + 0.4 * rng.random()), ground, rng)
                                for _ in range(3)])
        _write(out_dir / f"footstep_{ground}.wav", steps)

    # bubbles
    print("\nMoss bubble field (rain alpha 2.9, r 0.5-5 mm, 80 /s):")
    y, info = bubble_field(2.0, rng, rate=80.0, alpha=2.9, r_range=(0.5e-3, 5e-3), return_info=True)
    cen, main, peaks = _analyse(y)
    ok &= bool(np.all(np.isfinite(y)))
    print(f"  bubbles {info['n']}, median f0 {np.median(info['f0']):.0f} Hz (f0*r = {MINNAERT_FR}), "
          f"centroid {cen:.0f} Hz, main {main:.0f} Hz, 1 mm -> {MINNAERT_FR / 1e-3:.0f} Hz (Farnell: ~3 kHz)")
    _write(out_dir / "bubble_field_rain.wav", y)
    y, info = bubble_field(2.0, rng, rate=40.0, alpha=1.5, r_range=(1e-3, 3e-2), return_info=True)
    _write(out_dir / "bubble_field_surf.wav", y)
    print(f"  surf-like alpha 1.5, r 1-30 mm: bubbles {info['n']}, median f0 {np.median(info['f0']):.0f} Hz")

    print("\nFarnell surfacing bubble (f = fmin + (1-D) fmax + 2000 (1-(1-t/T)^4)):")
    for dim in (0.2, 0.5, 0.9):
        yb = surfacing_bubble(rng, dim=dim, mydim=0.0, mydur=0.0)
        T = dim * 0.09
        fa = _inst_freq(yb, 0.0, T * 0.3)
        fb = _inst_freq(yb, T * 0.8, T + 0.01)
        f_start = 60 + (1 - dim) * 300
        print(f"  dim {dim}: T {T * 1e3:.0f} ms, f early {fa:6.0f} Hz, f late {fb:6.0f} Hz "
              f"(formula start {f_start:.0f}, end {f_start + 2000:.0f})")
        ok &= fb > fa and bool(np.all(np.isfinite(yb)))
    ys = np.concatenate([surfacing_bubble(rng) for _ in range(12)])
    _write(out_dir / "surfacing_bubbles.wav", ys)

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
