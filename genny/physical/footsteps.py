# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Footsteps: one footstep on a named ground, driven by a ground-reaction-force (GRF) curve.

Manifest ids ``sfx.foley.step.<ground>`` ("Subject: one footstep on <ground>").

Ports (research notes in ``out/research``):

* **GRF curve** - Farnell, "Marching onwards: procedural synthetic footsteps" (PdCon 2007) and
  *Designing Sound* Practical 26 (research 03 §2.3-2.5, 05 §5.2):
  ``p(x) = 1.5 (1 - x) (N x - N x^3)``, N = 3.3333, per phase; the foot phase is compressed to
  3/4 of the contact (``clip 0..0.75 x 1.3333``) and split into heel ``clip 0..1/3 x 3``,
  roll/outstep ``clip 0.125..0.875 - 0.125 x 1.3333`` and ball ``clip 2/3..1 - 2/3 x 3``.
  As printed, N multiplies the whole pulse (``(1-x)(Nx - Nx^3) = N x (1-x)^2 (1+x)``), so the per
  phase N are the phase amplitudes; we set them from the PD phase ratios **3:2:3 walking** and
  **1:2:3 slow uphill / difficult (snow)**. Acceleration term of the patch: ``ballf = a + 0.5``,
  ``heelf = (1 - a) 1.7``, applied relative to a = 0 (heel x (1 - a), ball x (1 + 2a)).
* **Two-foot phasor** - Farnell fig. 49.4 / PD §4: right foot = phasor + 0.5, each foot's phase
  ``min(p, c)/c``. Walking overlap 10-20 % of the cycle (thesis T p.443): the trailing foot is
  still on the ground (its ball push-off) when the leading heel lands; both feet drive their own
  texture generator ("at least two overlapping sound generators", research 03 §2.1).
* **Pace** - Visell et al.: 75-125 steps/min, ~0.5 s (female) to ~0.8 s (male) between steps;
  expressive tempo irregularity is more natural than a constant pace (research 06 §7.1).
* **Per-step variation** - Cook, "Modeling Bill's Gait" / *Real Sound Synthesis* Ch.15
  (research 04 §3): every step is a prototype envelope plus per-breakpoint deviation, and the two
  feet differ (left/right asymmetry). Here the deviation is applied to the GRF phase amplitudes,
  split points and pace.
* **Aggregate grounds** - Visell (research 06 §7.3): grains fire only while the foot loads,
  ``lambda = A u (1 + tanh(B u))/2``, ``u = dF_L/dt``; grain energies ``p(E) ~ E^gamma`` summing
  to the step energy; each grain a resonant impact through Cook's PhISEM resonators (STK
  ``Shakers.cpp`` presets and Cook's measured gravels, ``particles.py``). Farnell coherence: grain
  tone follows the pressure (research 05 §1, §5.3). Blends of 2-3 texture synths (Farnell T 5.3).
* **Hard grounds** - Visell §3.3.1 source-filter (heel strike and toe slap are two impacts into
  the floor and/or shoe modal response); van den Doel & Pai contact pulse ``(1 - cos)``, hardness =
  contact duration (research 06 §1.3); Farnell: stone/concrete = square of the rate of work
  envelopes a noise burst into the surface filter; resonant surfaces: "contact time increases
  damping, so we can integrate the GRF" - the pressing foot adds damping in proportion to the GRF
  and the surface rings on after the lift (research 03 §2.1, 05 §5.3). FoleyAutomatic
  micro-collisions for glassy/metal contacts (research 06 §1.3).
* **Soft thud** - van den Doel recipe "dull low thud": ``stick.sy`` at fscale ~0.1, bang
  T = 30-50 ms (research 06 §1.5); heel impact energy sits in 100-500 Hz (Visell / Pastore).
  Plastic grounds (mud, ash, sand) do not ring, they thud (Farnell rule 6, research 01).
* **Snow squeak** - Farnell: snow deforms asymmetrically and squeaks (research 03 §2.1); built with
  Farnell's efficient stick-slip (thesis discovery: a pulse train whose amplitudes are
  proportional to the square root of the pulse separation, research 05 §0.4) exciting the snow's
  own Crunch resonance.
* **Water / mud** - Moss bubble ensembles with formation rate ~ u^2 (research 06 §3.3), STK water
  drops after the lift, Farnell surfacing bubbles for the mud suction (``particles.py``).

Measured tables used for the hard grounds (``modal_data``), and why:

=============  ===========================  ===============================================
ground         table                        reason
=============  ===========================  ===============================================
wood           Faust ``body_squareBig``     measured wooden body (modeInterpRes); a plank
                                            floor is a large square wooden body
metal_web      modalpaper ``wok.sy`` +      iron, measured; STK mapping "metal web: Coke can"
               STK CokeCan grains           (research 07 table) for the rattle; Cook: grains
                                            into the web modes plus the modal thump
ice            modalpaper ``calona0.sy``    glass bottle: the only measured brittle glassy
                                            body; frozen things ring (Farnell)
crystal        ``calona0.sy`` (2 bodies)    glassy crystal pieces
stone,         STK ModalBar ``Clump``       research 07 mapping "stone/marble/...: ModalBar
concrete,                                   Wood1/Wood2/Clump + short Shakers grit layer";
cobblestone                                 Clump's clustered ratios 1-1.73 match Farnell's
                                            "stone: many clustered resonances"
hard_floor     STK ModalBar ``Wood2``       same research 07 mapping; short dead knock
marble         modalpaper ``vase.sy``       measured ceramic: homogeneous brittle material
               + ``Clump`` slab             gives purer tones (Farnell T p.58)
bone           modalpaper ``stick.sy``      nearest measured stiff dry rod (UNSOURCED)
coral          modalpaper ``vase.sy``       nearest measured brittle calcareous body
                                            (UNSOURCED), heavily damped (porous)
shoe heel      modalpaper ``stick.sy``      Visell: "the resonator is the floor and/or the
                                            shoe sole"; stacked leather/wood heel (UNSOURCED)
=============  ===========================  ===============================================

Everything marked ``# UNSOURCED`` is a mapping or a number no source gives (layer mix levels,
table choices for bone/coral/ash/salt/mud/wet grounds, f0 of the ModalBar floors, the foot
damping amount, the squeak pulse rate, bubble/drip counts).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numba
import numpy as np

from . import modal
from . import particles as pt
from .core import SR, highpass, lowpass, one_pole_coef, one_pole_lp, reson, master, pow_decay, seconds, spectrum_peaks, write_wav

FAMILY_PREFIXES: tuple[str, ...] = ("sfx.foley.step.",)

# ==========================================================================================
# GRF (Farnell PdCon 2007 / Practical 26)
# ==========================================================================================

GRF_N = 3.3333                              # PD §4.2 default N
PHASE_RATIOS = {                            # PD §4: heel : roll : ball amplitude ratios
    "walk": (3.0, 2.0, 3.0),
    "run": (3.0, 1.0, 3.0),
    "difficult": (1.0, 2.0, 3.0),           # "slow uphill / difficult (snow)"
}
SPLITS = ((0.0, 1.0 / 3.0), (0.125, 0.875), (2.0 / 3.0, 1.0))   # PD §4 curve generator
COMPRESS = 0.75                             # "compress X to 3/4 of the step"
WALK_OVERLAP = (0.10, 0.20)                 # thesis T p.443: walk overlap 10-20 % of the cycle
STEP_INTERVAL = (0.48, 0.80)                # Visell: 75-125 steps/min


def grf_poly(x: np.ndarray) -> np.ndarray:
    """Farnell's GRF pulse with N factored out: 1.5 x (1-x)^2 (1+x); peak 0.3027 at x = 0.390
    (x N = 3.3333: 1.009, research 03 §2.4).

    ``1.5 (1-x)(N x^3 - N x)`` is <= 0 on [0,1]; its magnitude is N * this function.
    """
    x = np.clip(x, 0.0, 1.0)
    return 1.5 * x * (1.0 - x) ** 2 * (1.0 + x)


def foot_grf(n: int, contact: float, amps=(3.0, 2.0, 3.0), splits=SPLITS, offset: float = 0.0):
    """One foot's GRF components over ``n`` samples.

    The foot touches down at ``offset`` seconds; its GRF lasts ``contact`` seconds (the 3/4
    compressed part of its phase). Returns (heel, roll, ball) arrays; the GRF is their sum.
    Amplitudes are the per-phase N (PD phase ratios).
    """
    t = np.arange(n) / SR - offset
    x = np.clip(t / contact, 0.0, 1.0)
    x[t < 0] = 0.0
    x[t > contact] = 1.0
    comps = []
    for (lo, hi), a in zip(splits, amps):
        comps.append(a * GRF_N * grf_poly((np.clip(x, lo, hi) - lo) / (hi - lo)))
    return comps


# ==========================================================================================
# Gait: pace, overlap, foot, per-step deviation (Visell pace; Cook deviation and asymmetry)
# ==========================================================================================

def _gait(variation: int, rng: np.random.Generator, ratios: str) -> dict:
    """Pace and GRF shape of one natural step. Variation = which foot and a slightly different pace.

    # UNSOURCED: the deviation sizes (pace +-3 %, phase amplitudes +-12 %, split points +-0.02,
    # acceleration term +-0.08) and the left-foot asymmetry (heel x0.92, ball x1.06, pace x0.97)
    # are not given by Cook or Farnell (Cook prints shapes only); they sit inside Visell's pace range.
    """
    base = {0: (0.50, "R", 0.0), 1: (0.48, "L", -0.05), 2: (0.53, "R", 0.08)}[variation % 3]
    T, foot, accel = base
    T *= 1.0 + rng.uniform(-0.03, 0.03)
    T = float(np.clip(T, STEP_INTERVAL[0] - 0.03, STEP_INTERVAL[1]))
    ov = rng.uniform(WALK_OVERLAP[0], 0.14)           # low half of 10-20 %: trailing ball ends near heel strike
    accel += rng.uniform(-0.08, 0.08)
    h, r, b = PHASE_RATIOS[ratios]
    dev = 1.0 + rng.uniform(-0.12, 0.12, 3)
    h, r, b = h * dev[0] * (1 - accel), r * dev[1], b * dev[2] * (1 + 2 * accel)   # patch heelf/ballf
    if foot == "L":
        h, b, T = h * 0.92, b * 1.06, T * 0.97
    sj = rng.uniform(-0.02, 0.02, 3)
    splits = ((0.0, 1 / 3 + sj[0]), (0.125 + sj[1], 0.875), (2 / 3 + sj[2], 1.0))
    stride = 2.0 * T
    contact = (0.5 + ov) * stride                      # = COMPRESS * c * stride with c = (0.5+ov)/0.75
    return dict(T=T, stride=stride, overlap=ov, contact=contact, amps=(h, r, b), splits=splits,
                foot=foot, accel=accel, c=(0.5 + ov) / COMPRESS)


# ==========================================================================================
# Private helpers
# ==========================================================================================

@numba.njit(cache=True)
def _tv_bank(x, f, d, g, vdd, norm, extra):
    """Modal bank (van den Doel reson / Faust modeFilter forms) with a per-sample extra decay
    rate ``extra`` (1/s) shared by all modes: the pressing foot's damping."""
    n = x.shape[0]
    y = np.zeros(n)
    for k in range(f.shape[0]):
        th = 2.0 * np.pi * f[k] / 48000.0
        c = np.cos(th)
        s = np.sin(th)
        x1 = 0.0
        x2 = 0.0
        y1 = 0.0
        y2 = 0.0
        for i in range(n):
            R = np.exp(-(d[k] + extra[i]) / 48000.0)
            a1 = -2.0 * R * c
            a2 = R * R
            if vdd:
                yi = g[k] * R * s * x[i] - a1 * y1 - a2 * y2
            else:
                yi = g[k] / norm * (x[i] - x2) - a1 * y1 - a2 * y2
            x2 = x1
            x1 = x[i]
            y2 = y1
            y1 = yi
            y[i] += yi
    return y


def _render_tv(force: np.ndarray, modes: modal.Modes, extra: np.ndarray) -> np.ndarray:
    return _tv_bank(np.ascontiguousarray(force, np.float64), modes.freqs, modes.decays, modes.gains,
                    modes.form == "vdd", float(modes.norm), np.ascontiguousarray(extra, np.float64))


def _norm(y: np.ndarray) -> np.ndarray:
    p = float(np.max(np.abs(y)))
    return y / p if p > 0 else y


def _db(v: float) -> float:
    return 10.0 ** (v / 20.0)


def _grain_events(force: np.ndarray, rate_per_s: float, density: float, gamma: float,
                  energy: float, rng: np.random.Generator):
    """Visell grain times and energies (research 06 §7.3), same law as
    ``particles.granular_footstep_texture``: returns (amps per sample, normalised force)."""
    fl = pt._lowpass_force(force)
    fmax = float(np.max(np.abs(fl))) or 1.0
    fn = np.clip(fl / fmax, 0.0, None)
    u = np.gradient(fn) * SR
    umax = float(np.max(u))
    n = len(fn)
    amps = np.zeros(n)
    if umax <= 0:
        return amps, fn, u
    lam0 = np.clip(u * (1.0 + np.tanh(pt.VISELL_B * u / umax)) / 2.0, 0.0, None)
    t_load = (u > 0.02 * umax).sum() / SR
    A = rate_per_s * density * t_load / (lam0.sum() / SR)
    ev = rng.random(n) < np.clip(A * lam0 / SR, 0.0, 1.0)
    k = int(ev.sum())
    if k:
        e = pt._powerlaw(rng, k, pt.GRAIN_E_MIN, 1.0, gamma)
        e *= energy / e.sum()
        amps[ev] = np.sqrt(e)
    return amps, fn, u


def _texture(force: np.ndarray, preset_name: str, rng: np.random.Generator, *, rscale: float = 1.0,
             density: float = 1.0, gamma: float = pt.VISELL_GAMMA, info: dict | None = None):
    """PhISEM grains of any preset driven by a GRF (Visell law + Farnell tone coherence).

    Private extension of ``particles.granular_footstep_texture``, which only accepts its own
    ``GROUNDS`` keys; the grain law, energy budget, per-grain retune and tone coherence
    ``f * 2**(Fn^2 - 0.5)`` are the library's.
    """
    p = pt.preset(preset_name)
    q = pt._engine_params(p, None, rscale, None)
    amps, fn, u = _grain_events(force, q["p_event"] * SR, density, gamma, float(np.max(force)), rng)
    if info is not None:
        gr = info.setdefault("grains", {})
        gr[preset_name] = gr.get(preset_name, 0) + int((amps > 0).sum())
        info["unloading"] = info.get("unloading", 0) + int(((amps > 0) & (u <= 0)).sum())
    fscale = 2.0 ** (fn ** 2 - 0.5)
    seed = int(rng.integers(0, 2**31 - 1))
    return pt._phisem_kernel(p["kind"], False, np.zeros(len(fn)), False, True, amps, fscale,
                             q["Ds"], q["Dn"], q["p_event"], 1.0, q["base_f"], q["radii"], q["fgains"],
                             q["vary_on"], q["vary"], q["eq"][0], q["eq"][1], q["eq"][2], q["sweep"],
                             q["ratchet_delta"], q["ratchet_c"], pt.MIN_ENERGY, pt.MAX_SHAKE, 2, seed)


def _modal_grains(force: np.ndarray, spec: dict, rng: np.random.Generator) -> np.ndarray:
    """Cook's stochastic-resonance PhISEM for bone/coral clatter (research 04 §2.5 "coins in a
    mug": per collision a small object with its own mode set): the loudest Visell grains each
    strike a measured modal body at a random size (f_scale)."""
    amps, fn, u = _grain_events(force, spec["rate"], 1.0, pt.VISELL_GAMMA, 1.0, rng)
    idx = np.flatnonzero(amps)
    if len(idx) == 0:
        return np.zeros(len(force))
    top = idx[np.argsort(-amps[idx])[: spec["top"]]]
    out = np.zeros(len(force))
    for i in top:
        fs = rng.uniform(*spec["fs"])
        m = modal.resolve(spec["table"], fs, rng.random(), d_scale=spec["d_scale"])
        ring = int(min(modal.ring_time(m, 50.0), 0.25) * SR) + 64
        e = np.zeros(ring)
        pulse = modal.bang(spec["contact_ms"], amps[i])
        e[: len(pulse)] = pulse[:ring]
        y = modal.render_modes(e, m)
        k = min(ring, len(out) - i)
        out[i:i + k] += y[:k]
    return out


@numba.njit(cache=True)
def _stick_slip_pulses(n, fn, dfn, thresh, rate_lo, rate_hi, jitter, seed):
    """Farnell efficient stick-slip: pulses whose amplitude ~ sqrt(separation) (research 05 §0.4),
    firing while the pressure is above ``thresh`` and rising (compaction)."""
    np.random.seed(seed)
    out = np.zeros(n)
    t = 0.0
    last = -1
    i = 0
    while i < n:
        if fn[i] > thresh and dfn[i] > 0.0:
            rate = rate_lo + (rate_hi - rate_lo) * (fn[i] - thresh) / (1.0 - thresh)
            sep = (1.0 / rate) * (1.0 + jitter * (2.0 * np.random.random() - 1.0))
            if last >= 0:
                out[i] = np.sqrt(sep) * (fn[i] - thresh) / (1.0 - thresh)
            last = i
            i += max(1, int(sep * 48000.0))
        else:
            last = -1
            i += 1
    return out


def _ramp_in(n: int, ms: float) -> np.ndarray:
    m = max(2, seconds(ms * 1e-3))
    r = np.ones(n)
    k = min(m, n)
    r[:k] = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / m)
    return r


# ==========================================================================================
# Ground specs
# ==========================================================================================
# layer mix levels (dB, each layer peak-normalised first) are UNSOURCED throughout.
# grains: (PhISEM preset, dB, resonance_scale, density)

def _floor(table, db, heel_ms, toe_ms, *, f_scale=1.0, f0=None, d_scale=1.0, foot_damp=60.0,
           micro=0, bodies=1, body_ratio=1.0, contact_spread=0.0):
    return dict(table=table, db=db, heel_ms=heel_ms, toe_ms=toe_ms, f_scale=f_scale, f0=f0,
                d_scale=d_scale, foot_damp=foot_damp, micro=micro, bodies=bodies, body_ratio=body_ratio,
                contact_spread=contact_spread)


THUD = dict(table="sy_stick", f_scale=0.1)       # VdD "dull low thud": stick.sy fscale ~0.1, T 30-50 ms
SHOE = dict(table="sy_stick", f_scale=0.7, d_scale=1.5, heel_ms=2.5, toe_ms=5.0)   # UNSOURCED (see doc)

GROUND_SPECS: dict[str, dict] = {
    # ---------------- aggregate / soft grounds ----------------
    "snow": dict(friction=-9.0, ratios="difficult", thud=(45.0, -4.0), grains=[("Crunch", 0.0, 1.0, 1.0)], squeak=-20.0,
                 src="PhOLIES Crunch1 'like new fallen snow' 800 Hz r.95 N7 (phisem.c:41); PD 1:2:3 snow; "
                     "Farnell snow squeak"),
    "sand": dict(soft_lp=3000.0, friction=-6.0, ratios="difficult", thud=(50.0, -3.0), grains=[("Sandpaper", 0.0, 1.0, 1.0)],
                 src="research 11 A3 sand -> Sandpaper 4500 Hz r.6 N128; plastic -> thud (Farnell rule 6)",
                 unsourced="1:2:3 'difficult' ratios extended from snow to sand"),
    "wet_sand": dict(soft_lp=2000.0, friction=-9.0, ratios="difficult", thud=(45.0, 0.0), grains=[("Sandpaper", -6.0, 0.55, 0.5)],
                     bubbles=dict(peak_rate=120.0, r=(1.0e-3, 4.0e-3), alpha=2.9, delta_th=0.08, db=-14.0),
                     src="Sandpaper grains + Moss bubbles",
                     unsourced="wet sand: Sandpaper resonance x0.55, density 0.5, a few damped bubbles"),
    "ash": dict(soft_lp=2000.0, friction=-8.0, ratios="walk", thud=(50.0, 0.0), grains=[("Sandpaper", -4.0, 0.5, 0.7), ("Crunch", -8.0, 0.7, 0.6)],
                src="ash is plastic: thud, no ring (Farnell rule 6, research 01)",
                unsourced="no source has an ash preset: Sandpaper x0.5 + Crunch x0.7 (research 11: "
                          "'Sandpaper or Cabasa with the resonator lowered')"),
    "mud": dict(soft_lp=2500.0, friction=-8.0, ratios="difficult", thud=(50.0, 0.0), grains=[("WaterDrops", -10.0, 0.7, 0.5)],
                bubbles=dict(peak_rate=900.0, r=(2.0e-3, 1.2e-2), alpha=2.9, delta_th=0.12, db=-2.0),
                suction=(1, 3, -6.0),
                src="Farnell T 5.3: mud -> bubbles (Moss); surfacing bubbles (Farnell) for the suction",
                unsourced="mud bubble radii 2-12 mm, thermal/viscous damping 0.12, rates, suction count"),
    "shallow_water": dict(friction=-10.0, ratios="walk", thud=(35.0, -8.0), grains=[("WaterDrops", -3.0, 1.0, 1.0),
                                                                 ("Maraca", -9.0, 1.0, 0.6)],
                          bubbles=dict(peak_rate=900.0, r=(0.8e-3, 6.0e-3), alpha=2.9, delta_th=0.02, db=0.0),
                          drips=(3, 6, -9.0),
                          src="research 11 A3: Water drops + Maraca splash; Moss rain alpha 2.9; STK drops",
                          unsourced="splash bubble rate ~ u^2 peak 900/s, drip count 3-6"),
    "leaves": dict(friction=-9.0, ratios="walk", thud=(30.0, -8.0), grains=[("Sekere", 0.0, 1.0, 1.0), ("Sticks", -8.0, 1.0, 0.8)],
                   src="research 11 A3 leaves -> Stix1/Sekere 5500 Hz r.6"),
    "grass": dict(soft_lp=4500.0, friction=-8.0, ratios="walk", thud=(35.0, -2.0), grains=[("Sekere", -4.0, 0.8, 0.45)],
                  src="research 11 A3 dry grass -> Sekere",
                  unsourced="grass is softer than leaves: Sekere x0.8, density 0.45"),
    "forest_floor": dict(friction=-9.0, ratios="walk", thud=(35.0, -2.0), grains=[("Sekere", -3.0, 0.9, 0.6), ("Sticks", -6.0, 1.0, 0.7),
                                                                 ("Crunch", -10.0, 0.8, 0.5)],
                         src="Farnell exercise: blend earth + grass + twigs (research 03 §2.6; T 5.3 blend 2-3 synths)",
                         unsourced="layer levels"),
    "gravel": dict(friction=-9.0, ratios="walk", thud=(30.0, -6.0), grains=[("CookLargeGravel", 0.0, 1.0, 1.0),
                                                          ("CookSmallGravel", -9.0, 1.0, 0.5)],
                   src="Cook measured gravel: large 6460 Hz r.932 N23, small 12670 Hz r.843 N1068"),
    "pebbles": dict(friction=-10.0, ratios="walk", thud=(25.0, -6.0), grains=[("BigRocks", 0.0, 0.7, 0.6)],
                    modal_grains=dict(table="stk_clump", fs=(1.0, 1.0), rate=40.0, top=6, d_scale=6.0,
                                      contact_ms=0.4, db=-10.0, f0=(1800.0, 3200.0)),
                    src="STK BigRocks; Cook: bigger stones -> lower resonance, fewer collisions",
                    unsourced="pebbles = BigRocks x0.7, density 0.6; stone-clack Clump grains 1.8-3.2 kHz"),
    "salt": dict(friction=-9.0, ratios="walk", thud=(30.0, -8.0), grains=[("Crunch", -2.0, 1.6, 1.0), ("LittleRocks", -4.0, 1.0, 0.5)],
                 src="Cook: salt shaker is a maraca-type PhISEM (research 04 §2.5)",
                 unsourced="salt crust = Crunch x1.6 + LittleRocks blend"),
    # ---------------- hard / resonant grounds ----------------
    "stone": dict(friction=-34.0, ratios="walk", floor=_floor("stk_clump", 0.0, 4.0, 7.0, f0=300.0, d_scale=2.0),
                  shoe=-4.0, scuff=-18.0, grains=[("Sandpaper", -18.0, 1.0, 0.4)],
                  src="research 07: stone -> ModalBar Clump + short Shakers grit layer",
                  unsourced="Clump f0 300 Hz, d x2 (slab bedded in ground)"),
    "concrete": dict(friction=-32.0, ratios="walk", floor=_floor("stk_clump", 0.0, 4.0, 7.0, f0=230.0, d_scale=3.0),
                     shoe=-3.0, scuff=-14.0, grains=[("Sandpaper", -14.0, 1.0, 0.5)],
                     src="Farnell: concrete = rate-of-work squared envelopes noise into the surface filter",
                     unsourced="Clump f0 230 Hz, d x3 (monolithic, dead)"),
    "cobblestone": dict(friction=-32.0, ratios="walk", floor=_floor("stk_clump", 0.0, 3.5, 6.0, f0=420.0, d_scale=2.5,
                                                    contact_spread=0.15),
                        shoe=-4.0, scuff=-16.0, grains=[("Sandpaper", -16.0, 1.0, 0.4), ("BigRocks", -20.0, 0.8, 0.2)],
                        src="research 07 Clump + grit; each contact on a different stone",
                        unsourced="Clump f0 420 Hz +-15 % per contact"),
    "wet_stone": dict(friction=-26.0, ratios="walk", floor=_floor("stk_clump", 0.0, 5.0, 8.0, f0=300.0, d_scale=3.0),
                      shoe=-8.0, scuff=-20.0, grains=[("WaterDrops", -10.0, 1.2, 0.5)],
                      bubbles=dict(peak_rate=250.0, r=(0.5e-3, 2.5e-3), alpha=2.9, delta_th=0.02, db=-12.0),
                      src="stone floor + a water film: STK drops and Moss bubbles",
                      unsourced="water film = longer, softer contact; small bubbles"),
    "marble": dict(ratios="walk", floor=_floor("sy_vase", 0.0, 2.5, 5.0, d_scale=2.0, foot_damp=40.0),
                   floor2=_floor("stk_clump", -6.0, 3.0, 6.0, f0=340.0, d_scale=1.5),
                   shoe=-6.0, scuff=-22.0,
                   src="modalpaper vase.sy (measured ceramic, homogeneous) + Clump slab",
                   unsourced="vase for marble; d x2"),
    "hard_floor": dict(friction=-36.0, ratios="walk", floor=_floor("stk_wood2", 0.0, 4.0, 7.0, f0=200.0),
                       shoe=-3.0, scuff=-16.0, grains=[("Sandpaper", -18.0, 1.0, 0.3)],
                       src="research 07: hard grounds -> ModalBar Wood1/Wood2/Clump",
                       unsourced="hard floor = Wood2 at 200 Hz (dead knock)"),
    "wood": dict(ratios="walk", floor=_floor("body_squareBig", 0.0, 3.0, 6.0, foot_damp=30.0),
                 shoe=-6.0, scuff=-18.0,
                 src="Faust modeInterpRes squareBig (measured wooden body); Farnell wooden floorboards"),
    "metal_web": dict(friction=-30.0, ratios="walk", floor=_floor("sy_wok", 0.0, 2.0, 4.0, foot_damp=90.0, micro=3),
                      shoe=-8.0, scuff=-16.0, grains=[("CokeCan", -8.0, 1.0, 0.5)],
                      src="modalpaper wok.sy (iron, measured) damped by the foot (Farnell integrate GRF); "
                          "STK CokeCan rattle (research 07); Cook grains into the web modes",
                      unsourced="foot damping 90 /s at full GRF"),
    "ice": dict(friction=-24.0, ratios="walk", floor=_floor("sy_calona0", 0.0, 2.0, 4.0, f_scale=0.35, foot_damp=40.0, micro=2),
                floor2=_floor("stk_clump", -6.0, 4.0, 7.0, f0=260.0, d_scale=1.0),
                shoe=-8.0, grains=[("Crunch", -12.0, 1.4, 0.5)],
                src="modalpaper calona0.sy glass; frozen things ring (Farnell rule 6); frost crunch",
                unsourced="calona0 f x0.35 for a large ice sheet"),
    "crystal": dict(friction=-34.0, ratios="walk", floor=_floor("sy_calona0", 0.0, 1.0, 2.5, foot_damp=40.0, micro=3, bodies=2,
                                                body_ratio=1.37),
                    shoe=-10.0, grains=[("LittleRocks", -16.0, 1.0, 0.2)],
                    src="modalpaper calona0.sy glass, two bodies, FoleyAutomatic micro-collisions",
                    unsourced="second body at 1.37 x"),
    "bone": dict(friction=-14.0, ratios="walk", thud=(20.0, -8.0), floor=_floor("sy_stick", -3.0, 1.5, 3.0, f_scale=0.5, foot_damp=40.0),
                 grains=[("BigRocks", -8.0, 0.55, 0.6)],
                 modal_grains=dict(table="sy_stick", fs=(0.35, 0.8), rate=60.0, top=10, d_scale=0.8,
                                   contact_ms=0.6, db=-2.0),
                 src="research 11: bone -> Big rocks with lower freq (inference); Cook coins-in-mug per grain",
                 unsourced="stick.sy as bone; BigRocks x0.55; bone sizes f x0.35-0.8"),
    "coral": dict(friction=-12.0, ratios="walk", thud=(25.0, -8.0), floor=_floor("sy_vase", -6.0, 1.5, 3.0, f_scale=1.4, d_scale=6.0,
                                                              foot_damp=40.0),
                  grains=[("BigRocks", 0.0, 0.8, 0.8)],
                  modal_grains=dict(table="sy_vase", fs=(1.6, 2.6), rate=80.0, top=10, d_scale=8.0,
                                    contact_ms=0.4, db=-6.0),
                  src="research 11: coral -> Big rocks; Farnell crushing model driven by GRF",
                  unsourced="vase.sy as coral, heavily damped (porous); BigRocks x0.8"),
}


# ==========================================================================================
# Rendering
# ==========================================================================================

def _floor_layer(spec: dict, comps_lead, n, rng, foot_press):
    """Heel strike + toe slap contact pulses (and micro-collisions) into the floor modes,
    damped by the pressing foot. Toe slap = onset of the roll phase (inference)."""
    fl = spec
    out = np.zeros(n)
    heel, roll, _ = comps_lead
    dh = np.clip(np.gradient(heel) * SR, 0, None)
    dr = np.clip(np.gradient(roll) * SR, 0, None)
    ih = int(np.argmax(heel > 1e-6 * heel.max())) if heel.max() > 0 else 0
    ir = int(np.argmax(roll > 1e-6 * roll.max())) if roll.max() > 0 else 0
    ref = dh.max() or 1.0
    contacts = [(ih, 1.0, fl["heel_ms"]), (ir, float(dr.max() / ref), fl["toe_ms"])]
    for body in range(fl["bodies"]):
        fs_body = fl["f_scale"] * (fl["body_ratio"] ** body) * (1.0 + rng.uniform(-0.02, 0.02))
        for i0, vel, ms in contacts:
            # each contact at its own strike position (and, on cobbles, its own stone)
            fs = fs_body * (1.0 + rng.uniform(-fl["contact_spread"], fl["contact_spread"]))
            m = modal.resolve(fl["table"], fs, rng.uniform(0.2, 0.6), d_scale=fl["d_scale"], f0=fl["f0"])
            T = modal.contact_time(ms, vel)
            p = modal.bang(T, vel)
            f1 = np.zeros(n)
            f1[i0:i0 + len(p)] += p[: n - i0]
            modal.micro_collisions(f1, m, fl["micro"], T, vel, rng, start=i0)
            out += _render_tv(f1, m, foot_press * fl["foot_damp"])
    return out


def _scuff(F_all: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Farnell stone/concrete: (rate of work)^2 envelopes a noise burst (research 05 §5.3)."""
    dF = np.clip(np.gradient(F_all) * SR, 0, None)
    env = (dF / (dF.max() or 1.0)) ** 2
    return env * rng.uniform(-1, 1, len(F_all))


def render_step(ground: str, variation: int, rng: np.random.Generator, *, return_info: bool = False):
    """One footstep on ``ground`` (mono, 48 kHz, not mastered)."""
    if ground not in GROUND_SPECS:
        raise KeyError(f"unknown ground {ground!r}; known: {list(GROUND_SPECS)}")
    sp = GROUND_SPECS[ground]
    g = _gait(variation, rng, sp["ratios"])
    pre = 0.002
    tail = 0.06
    dur = min(pre + g["contact"] + tail, 0.72)
    n = seconds(dur)
    lead = foot_grf(n, g["contact"], g["amps"], g["splits"], offset=pre)
    trail = foot_grf(n, g["contact"], g["amps"], g["splits"], offset=pre - g["T"])
    F_lead = lead[0] + lead[1] + lead[2]
    F_trail = trail[0] + trail[1] + trail[2]
    F_all = F_lead + F_trail
    fmax = float(F_all.max()) or 1.0
    info = dict(ground=ground, gait=g)
    layers: list[tuple[str, np.ndarray, float]] = []

    # --- aggregate grains: one generator per foot (polyphony), trailing foot fades in from the file start
    for name, db, rs, dens in sp.get("grains", []):
        y = _texture(F_lead, name, rng, rscale=rs, density=dens, info=info)
        y += _texture(F_trail, name, rng, rscale=rs, density=dens, info=info) * _ramp_in(n, 8.0)
        layers.append((f"grains:{name}", y, db))

    # --- friction / crushing driven by the GRF directly (Farnell T 5.3), Cook's PhISEM friction
    # engine (research 04 §2.6: "feed PhISEM a control envelope from force x speed"): fills the
    # roll phase, where Visell's loading-only grains are silent. Same preset as the first grain layer.
    if "grains" in sp and sp.get("friction") is not None:
        name, _, rs, _ = sp["grains"][0]
        ctrl = np.clip(F_lead / fmax, 0.0, 1.0)
        # UNSOURCED: control level 0.6 of the full shake energy
        y = pt.shaker(name, dur, rng, energy_curve=0.6 * ctrl, energy_mode="level", resonance_scale=rs)
        layers.append((f"friction:{name}", y[:n], sp["friction"]))

    # --- soft heel thud (VdD dull thud recipe) at heel strike and toe slap
    if "thud" in sp:
        ms, db = sp["thud"]
        m = modal.resolve(THUD["table"], THUD["f_scale"] * (1 + rng.uniform(-0.02, 0.02)))
        force = np.zeros(n)
        heel, roll, _ = lead
        i0 = seconds(pre)
        dr = np.clip(np.gradient(roll), 0, None).max() / (np.clip(np.gradient(heel), 0, None).max() or 1)
        ir = int(np.argmax(roll > 0))
        for i, v, c in ((i0, 1.0, ms), (ir, float(dr), ms * 1.3)):
            p = modal.bang(modal.contact_time(c, v), v)
            force[i:i + len(p)] += p[: n - i]
        layers.append(("thud", modal.render_modes(force, m), db))

    # --- hard floor: contacts into the floor modes, damped by the foot
    press = np.clip(F_all / fmax, 0, None)     # foot pressure 0..1; extra decay = press * foot_damp (1/s)
    for key in ("floor", "floor2"):
        if key in sp:
            y = _floor_layer(sp[key], lead, n, rng, press)
            layers.append((key, y, sp[key]["db"]))
    if "shoe" in sp:
        m = modal.resolve(SHOE["table"], SHOE["f_scale"] * (1 + rng.uniform(-0.02, 0.02)),
                          d_scale=SHOE["d_scale"])
        force = np.zeros(n)
        heel, roll, _ = lead
        i0 = seconds(pre)
        ir = int(np.argmax(roll > 0))
        dr = np.clip(np.gradient(roll), 0, None).max() / (np.clip(np.gradient(heel), 0, None).max() or 1)
        for i, v, c in ((i0, 1.0, SHOE["heel_ms"]), (ir, float(dr), SHOE["toe_ms"])):
            p = modal.bang(modal.contact_time(c, v), v)
            force[i:i + len(p)] += p[: n - i]
        layers.append(("shoe", modal.render_modes(force, m), sp["shoe"]))
    if "scuff" in sp and "floor" in sp:
        fl = sp["floor"]
        m = modal.resolve(fl["table"], fl["f_scale"], 0.4, d_scale=fl["d_scale"], f0=fl["f0"])
        layers.append(("scuff", _render_tv(_scuff(F_all, rng), m, press * fl["foot_damp"]), sp["scuff"]))

    # --- stochastic-resonance clatter (bone, coral, pebbles)
    if "modal_grains" in sp:
        mg = dict(sp["modal_grains"])
        if "f0" in mg:   # ModalBar table: pick a pitch per stone
            f0s = mg["f0"]
            out = np.zeros(n)
            amps, fn, u = _grain_events(F_lead, mg["rate"], 1.0, pt.VISELL_GAMMA, 1.0, rng)
            idx = np.flatnonzero(amps)
            for i in idx[np.argsort(-amps[idx])[: mg["top"]]]:
                m = modal.resolve(mg["table"], 1.0, rng.uniform(0.2, 0.6), d_scale=mg["d_scale"],
                                  f0=rng.uniform(*f0s))
                e = np.zeros(seconds(0.15))
                p = modal.bang(mg["contact_ms"], amps[i])
                e[: len(p)] = p
                y = modal.render_modes(e, m)
                k = min(len(y), n - i)
                out[i:i + k] += y[:k]
            layers.append(("clatter", out, mg["db"]))
        else:
            layers.append(("clatter", _modal_grains(F_lead, mg, rng), mg["db"]))

    # --- snow squeak: stick-slip into the Crunch resonance (800 Hz, r .95 @44.1k)
    if "squeak" in sp:
        fl_ = pt._lowpass_force(F_lead)
        fn = np.clip(fl_ / (fl_.max() or 1), 0, None)
        dfn = np.gradient(fn)
        # UNSOURCED: threshold 0.45 of peak GRF, pulse rate 350-1100 Hz, 30 % jitter
        pulses = _stick_slip_pulses(n, fn, dfn, 0.45 + rng.uniform(-0.05, 0.05), 350.0, 1100.0, 0.30,
                                    int(rng.integers(0, 2**31 - 1)))
        cr = pt.preset("Crunch")
        r = pt.rate_radius(cr["radii"][0], cr["fs_src"])
        y = reson(pulses, cr["freqs"][0] * (1 + rng.uniform(-0.03, 0.03)), r)
        y = highpass(y, 400.0)
        layers.append(("squeak", y, sp["squeak"]))

    # --- bubbles (Moss: formation rate ~ u^2), drips after the lift, mud suction
    if "bubbles" in sp:
        b = sp["bubbles"]
        dF = np.clip(np.gradient(pt._lowpass_force(F_lead)) * SR, 0, None)
        rate = b["peak_rate"] * (dF / (dF.max() or 1.0)) ** 2
        y = pt.bubble_field(dur, rng, rate=rate, alpha=b["alpha"], r_range=b["r"], delta_th=b["delta_th"])
        layers.append(("bubbles", y[:n], b["db"]))
    if "drips" in sp:
        lo, hi, db = sp["drips"]
        y = np.zeros(n)
        t_lift = pre + g["contact"] * 0.75
        for _ in range(int(rng.integers(lo, hi + 1))):
            d = pt.drip(rng, dur=0.2)
            i = seconds(t_lift + rng.uniform(-0.05, 0.12))
            if i < n:
                k = min(len(d), n - i)
                y[i:i + k] += d[:k] * rng.uniform(0.3, 1.0)
        layers.append(("drips", y, db))
    if "suction" in sp:
        lo, hi, db = sp["suction"]
        y = np.zeros(n)
        t_lift = pre + g["contact"] * 0.8
        for _ in range(int(rng.integers(lo, hi + 1))):
            s = pt.surfacing_bubble(rng, dim=rng.uniform(0.6, 0.9), fmin=60.0, fmax=200.0, sweep=600.0)
            i = seconds(t_lift + rng.uniform(-0.04, 0.06))
            if i < n:
                k = min(len(s), n - i)
                y[i:i + k] += s[:k]
        layers.append(("suction", y, db))

    mix = np.zeros(n)
    soft = sp.get("soft_lp")
    for name, y, db in layers:
        if soft and (name.startswith("grains") or name.startswith("friction")):
            # Farnell T 5.3: a brightness/damping control approximates the ground's hardness
            # (softer ground = HF decays faster). UNSOURCED cut-off per ground.
            y = one_pole_lp(np.ascontiguousarray(y), one_pole_coef(soft))
        mix += _norm(y[:n]) * _db(db)
    nf = seconds(0.04)
    mix[-nf:] *= pow_decay(nf, 4.0)
    mix *= _ramp_in(n, 1.0)
    info["layers"] = [l[0] for l in layers]
    info["F"] = F_all
    info["F_lead"] = F_lead
    return (mix, info) if return_info else mix


# ==========================================================================================
# AUDIO_CHARTER §8.6: heel + roll + toe, jitter, the chain layer, equipment layers
# ==========================================================================================

EQUIPMENT = ("cloth", "leather", "mail", "plate")


def _heel_index(mix: np.ndarray) -> int:
    e = np.abs(mix)
    return int(np.argmax(e > 0.3 * e.max())) if e.max() > 0 else 0


def _varispeed(y: np.ndarray, ratio: float) -> np.ndarray:
    n = int(len(y) / ratio)
    return np.interp(np.arange(n) * ratio, np.arange(len(y)), y)


def charter_step(ground: str, variation: int, rng: np.random.Generator, *, chained: bool = False,
                 equipment: tuple[str, ...] = ("cloth",)) -> np.ndarray:
    """One step per AUDIO_CHARTER §8.6.

    * heel (the step's own transient) + roll (the ground's body, as rendered) + **toe**: the heel
      transient (first 25 ms from strike) re-struck 60-120 ms later, lighter (-8 dB) and 3 % higher
      (the smaller contact of the ball);
    * per-take jitter: +-2 dB gain and +-3 % pitch (varispeed of the whole step);
    * **chain layer** (``chained``): a chain clink on the gait's accent - the heel strike - lagging
      10-30 ms as the links swing (Farnell shell-casing tinkle / chain push, ambience.chain_shift);
      removed when the collar opens (the unchained render is the same step without it);
    * **equipment**: cloth (always, quiet: STK Sandpaper friction under the step's force, -24 dB),
      leather (a small stick-slip creak into the measured vase modes, -22 dB), mail (5-9 small
      tinkles scattered over 40 ms after the heel, -16 dB), plate (a measured-steel clank at the
      heel, -14 dB). UNSOURCED: all levels and timings."""
    from . import ambience as amb
    from . import body
    mix = render_step(ground, variation, rng)
    n0 = len(mix)
    hi = _heel_index(mix)
    heel = mix[hi:hi + seconds(0.025)].copy()
    heel *= pow_decay(len(heel), 2.0)
    toe_t = hi + seconds(rng.uniform(0.06, 0.12))
    toe = _varispeed(heel, 1.03) * _db(-8.0)
    y = np.concatenate([mix, np.zeros(seconds(0.35))])
    y[toe_t:toe_t + len(toe)] += toe[: len(y) - toe_t]
    t_heel = hi / SR
    if chained:
        c = amb.chain_shift(0.35, rng, pushes=1, f_scale=rng.uniform(0.75, 0.9))
        c = _norm(np.asarray(c, float)) * _db(-9.0) * float(np.max(np.abs(mix)))
        i = hi + seconds(rng.uniform(0.01, 0.03))
        y[i:i + len(c)] += c[: len(y) - i]
    pk = float(np.max(np.abs(mix)))
    force = np.zeros(len(y))
    force[:n0] = np.abs(mix) / (pk + 1e-12)
    force = pt._lowpass_force(force) if hasattr(pt, "_lowpass_force") else force
    if "cloth" in equipment:
        cl = pt.shaker("Sandpaper", len(y) / SR, rng, energy_curve=0.5 * np.clip(force, 0, 1), energy_mode="level",
                       resonance_scale=0.4)[: len(y)]
        y[: len(cl)] += _norm(lowpass(cl, 5000.0)) * _db(-24.0) * pk
    if "leather" in equipment:
        m = seconds(0.18)
        e = np.sin(np.pi * np.arange(m) / m)
        ss = body._stick_slip(0.3 + 0.5 * e, rng)
        lz = modal.render_modes(ss, modal.resolve("sy_vase", 0.7))
        i = hi + seconds(0.02)
        y[i:i + len(lz)] += (_norm(lz) * _db(-22.0) * pk)[: len(y) - i]
    if "mail" in equipment:
        for _ in range(int(rng.integers(5, 10))):
            tk = amb.tinkle(rng, f_scale=rng.uniform(1.1, 1.6), gain=1.0)
            i = hi + seconds(rng.uniform(0.0, 0.04))
            y[i:i + len(tk)] += (_norm(tk) * _db(-16.0) * pk * rng.uniform(0.4, 1.0))[: len(y) - i]
    if "plate" in equipment:
        cl = modal.strike("sy_sword2", rng.uniform(0.35, 0.45), 3.0, rng.uniform(0, 1), rng, velocity=0.4,
                          dur=0.3)
        i = hi + seconds(0.004)
        y[i:i + len(cl)] += (_norm(cl) * _db(-14.0) * pk)[: len(y) - i]
    y = _varispeed(y, 1.0 + rng.uniform(-0.03, 0.03)) * _db(rng.uniform(-2.0, 2.0))
    nz = np.flatnonzero(np.abs(y) > 1e-4 * (np.max(np.abs(y)) + 1e-12))
    y = y[: (nz[-1] + seconds(0.02)) if len(nz) else len(y)]
    nf = min(seconds(0.03), len(y) // 4)
    y[-nf:] *= pow_decay(nf, 3.0)
    return y


def render(row: dict, variation: int, rng: np.random.Generator) -> np.ndarray:
    """Family entry point: ``sfx.foley.step.<ground>`` and, with the charter on, the variants
    ``sfx.foley.step.<ground>.chained`` / ``.<equipment>`` / ``.chained.<equipment>`` (proposed
    manifest ids, AUDIO_CHARTER §8.6)."""
    from . import charter
    rid = row["id"]
    if not rid.startswith(FAMILY_PREFIXES[0]):
        raise KeyError(rid)
    parts = rid[len(FAMILY_PREFIXES[0]):].split(".")
    ground, mods = parts[0], parts[1:]
    if not charter.enabled():
        return render_step(ground, variation, rng)
    eq = ("cloth",) + tuple(m for m in mods if m in EQUIPMENT and m != "cloth")
    return charter_step(ground, variation, rng, chained="chained" in mods, equipment=eq)


# ==========================================================================================
# Self-test
# ==========================================================================================

MANIFEST = Path(r"C:\Users\bilal\Documents\SS\game\assets\production\manifest.toml")
OUT = Path(__file__).resolve().parent.parent / "out" / "samples_v2"


def _source_freqs(sp: dict) -> list[float]:
    fs = []
    for name, _, rs, _ in sp.get("grains", []):
        fs.append(pt.preset(name)["freqs"][0] * rs)
    for key in ("floor", "floor2"):
        if key in sp:
            fl = sp[key]
            m = modal.resolve(fl["table"], fl["f_scale"], 0.4, d_scale=fl["d_scale"], f0=fl["f0"])
            top = np.argsort(-np.abs(m.gains))[:2]
            fs += [float(v) for v in m.freqs[top]]
    return fs


def _lufs(x: np.ndarray) -> float:
    import pyloudnorm
    meter = pyloudnorm.Meter(SR, block_size=min(0.4, len(x) / SR * 0.99))
    return float(meter.integrated_loudness(x))


def _selftest() -> bool:
    import tomllib
    with open(MANIFEST, "rb") as fh:
        rows = [a for a in tomllib.load(fh)["asset"] if a["id"].startswith(FAMILY_PREFIXES[0])]
    ok = True
    print(f"{len(rows)} footstep ids; writing to {OUT}")
    print(f"{'file':<28}{'dur':>6}{'peak':>7}{'LUFS':>7}{'heel/ball ms':>14}  {'main peaks (Hz)':<30} source f (Hz) / layers")
    for row in rows:
        ground = row["id"].split(".")[-1]
        rel = Path(row["path"]).relative_to("assets/audio")
        for v in range(3):
            seed = zlib.crc32(f"{row['id']}#{v}".encode())
            rng = np.random.default_rng(seed)
            y, info = render_step(ground, v, rng, return_info=True)
            finite = bool(np.all(np.isfinite(y)))
            y = master(y, "sfx")
            name = rel.with_name(rel.stem + ("" if v == 0 else f"_v{v + 1}") + rel.suffix)
            write_wav(OUT / name, y)
            pk = 20 * np.log10(np.max(np.abs(y)) + 1e-12)
            lu = _lufs(y)
            # heel and ball timing from the GRF
            F = info["F_lead"]
            hb = f"{np.argmax(F[:seconds(0.15)]) / SR * 1e3:4.0f}/{(np.argmax(F[seconds(0.2):]) + seconds(0.2)) / SR * 1e3:4.0f}"
            peaks = [round(f) for f, _ in spectrum_peaks(y, 4, fmin=60.0)]
            unl = info.get("unloading", 0)
            # a transient step is peak-limited by the class ceiling (core.master soft knee) before
            # it reaches -20 LUFS; LUFS is reported, the ceiling is the hard check
            good = finite and pk <= -2.9 and -32.0 < lu <= -19.0 and unl == 0 and 0.35 <= len(y) / SR <= 0.75
            ok &= good
            src = ", ".join(f"{f:.0f}" for f in _source_freqs(GROUND_SPECS[ground]))
            flag = "" if good else "  <-- CHECK"
            print(f"{str(name.name):<28}{len(y) / SR:6.2f}{pk:7.1f}{lu:7.1f}{hb:>14}  {str(peaks):<30} "
                  f"[{src}] {'+'.join(info['layers'])} T={info['gait']['T']:.2f}s {info['gait']['foot']}"
                  f" grains={sum(info.get('grains', {}).values())}{flag}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
