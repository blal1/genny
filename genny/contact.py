"""Sounds of solid objects interacting: impact, bounce, break, crumple, roll, scrape.

Architecture (Rocchesso & Fontana eds., *The Sounding Object*, 2003, ch. 8-9; findings g04 §4-5,
g05 §0-4): a **modal resonator** x a **contact interaction** x a **temporal pattern**.

* Resonators - `ShapeBody`: analytic modal bodies whose mode gains are the mode shape sampled at the
  strike point (van den Doel & Pai, "The Sounds of Physical Shapes", 1996, eq. 10-11; findings g06
  A2-A6, B4), damped by the internal-friction law ``d_n = pi f_n tan(phi)`` plus an external loss
  (g06 A5, B1; g04 §5.1), rendered by a bilinear state-space bank that exposes displacement and
  velocity at the contact point (Sounding Object §8.2.3; g04 §4.3).
* Interaction - `ImpactInteractor`: Hunt-Crossley contact ``f = k x^a (1 + mu v)`` between a striker
  mass and the bank, solved per sample with the K method (Sounding Object §8.3.1 and appendix 8.A;
  g04 §4.4-4.5).
* Patterns - `bounce_schedule`, `break_schedule`, `crumple_events`, `rolling_offset`
  (Sounding Object §9.2-9.5; g05 §1-4), `fractal_noise` (§8.3.3; g04 §4.7), scrape = surface profile
  read at speed x normal force (van den Doel thesis ch. 6; g06 B6), avalanche and 4-stroke force
  loops into a modal body (g06 B7), box / sphere cavity with Sabine decay (Sounding Object §6.1;
  g04 §2.1).

Units are SI throughout (m, kg, s, N); every kernel derives its coefficients from the sample rate.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numba
import numpy as np
from scipy import special
from scipy.ndimage import grey_dilation, uniform_filter1d
from scipy.optimize import brentq
from scipy.signal import lfilter, resample_poly

from . import filters as F
from .analysis import _ear_weight
from .core import DEFAULT_SR, declick, samples
from .fx import effect
from .physical import modal as _modal
from .physical.core import tv_one_pole_lp
from .physics import MATERIALS as _PM
from .sfx import sfx

G = 9.81
TWO_PI = 2.0 * math.pi


# ==========================================================================================
# Materials
# ==========================================================================================

@dataclass(frozen=True)
class ContactMaterial:
    """Material as heard: decay law + contact law + what sets the pitch.

    ``fd`` is van den Doel's "material constant" f/d (thesis pp. 62-65; g06 B1): under the
    internal-friction model ``d = pi f tan(phi)`` it is one number per material,
    ``tan(phi) = 1/(pi fd)``, and the quality factor of every mode is ``q0 = pi fd``
    (Sounding Object §6.2: q0 = pi f t_e; g04 §2.3). ``scatter`` is the measured relative spread
    of f/d between the modes of one object. ``ext`` is a frequency-independent external loss in
    1/s (tau_0 of g06 A5). ``k`` (N/m^1.5) and ``mu`` (s/m) are the Hunt-Crossley stiffness and
    viscoelastic characteristic of a contact with this material (g04 §4.4). ``rayleigh`` is the
    sourced (alpha 1/s, beta s) pair, ``d = (alpha + beta w^2)/2`` (g07 §1.7, §4.1-4.2).
    """
    name: str
    fd: float
    scatter: float
    ext: float
    k: float
    mu: float
    density: float
    young: float
    poisson: float
    rayleigh: tuple | None = None

    @property
    def tan_phi(self) -> float:
        return 1.0 / (math.pi * self.fd)

    @property
    def q0(self) -> float:
        return math.pi * self.fd

    @property
    def c(self) -> float:
        """Bar wave speed sqrt(E/rho), m/s."""
        return math.sqrt(self.young / self.density)

    @property
    def c_shear(self) -> float:
        """Shear wave speed c2 = sqrt(G/rho), G = E/(2(1+nu)) (Johnson 1985, eqs 11.1-11.7, Table 11.1)."""
        return math.sqrt(self.young / (2.0 * (1.0 + self.poisson)) / self.density)

    @property
    def c_long(self) -> float:
        """Longitudinal wave speed c1 = sqrt(2(1-nu)G/((1-2nu)rho)) (Johnson, §11.1)."""
        return self.c_shear * math.sqrt(2.0 * (1.0 - self.poisson) / (1.0 - 2.0 * self.poisson))

    @property
    def c_rayleigh(self) -> float:
        """Rayleigh wave speed, Viktorov's c3 = c2 (0.87 + 1.12 nu)/(1 + nu): 0.92 c2 at nu = 0.25, Johnson's 0.919."""
        return self.c_shear * (0.87 + 1.12 * self.poisson) / (1.0 + self.poisson)


def _m(name, fd, scatter, ext, k, mu, rho, young, nu, rayleigh=None):
    p = _PM.get("aluminum" if name == "aluminium" else name)     # reuse genny.physics elastic constants
    if p is not None:
        rho, young, nu = p.density, p.young_modulus, p.poisson
    return ContactMaterial(name, fd, scatter, ext, k, mu, rho, young, nu, rayleigh)


# Contact stiffness: the five log-spaced levels of the hardness experiment, 5e6 (rubber mallet),
# 3.3e7, 2.24e8, 1.495e9, 1e10 (steel mallet) (Sounding Object §5.3; g04 §1.4).
MATERIALS: dict[str, ContactMaterial] = {m.name: m for m in (
    # q0 = 5..20 for rubber (g04 §2.3) -> fd = 10/pi; k = rubber mallet; mu = the book's reference 0.6
    _m("rubber", 3.2, 0.5, 0.0, 5e6, 0.6, 1100, 0.01e9, 0.49),
    # REN plastic plate alpha 52.6, beta 8.78e-7 (g07 §4.2): d(1 kHz) from beta = 17.3/s -> fd 58; ext = alpha/2
    _m("plastic", 58.0, 0.5, 26.3, 3.3e7, 0.4, 1190, 3.2e9, 0.35, (52.627, 8.7753e-7)),
    # wooden hockey stick f/d = 35 +- 22 (g06 B1); REN wood plate (g07 §4.2), ext = alpha/2;
    # mu from the bouncer tables: steel ball on wood, interval ratio 0.70-0.76 at ~1.4 m/s
    # (g05 §1, Table 13.5) with e = 1 - (2/3) mu v
    _m("wood", 35.0, 0.63, 1.07, 2.24e8, 0.29, 650, 11e9, 0.35, (2.1364, 3.0828e-6)),
    # q0 = 150..1500 for glass (g04 §2.3), geometric middle 470 -> fd 150; REN glass bowl, ext = alpha/2
    _m("glass", 150.0, 0.55, 9.15, 1.495e9, 0.1, 2500, 70e9, 0.22, (18.301, 1.4342e-7)),
    # REN porcelain plate: d(1 kHz) = 1.66/s -> fd 600; ext = OB ceramic alpha2/2 (g07 §1.7)
    _m("ceramic", 600.0, 0.55, 5.0, 1.495e9, 0.1, 2400, 70e9, 0.2, (3.7388e-2, 8.4142e-8)),
    # metal vase f/d = 477 +- 262 (g06 B1); REN metal plate, ext = alpha/2
    _m("metal", 477.0, 0.55, 3.15, 1e10, 0.08, 7850, 200e9, 0.3, (6.3035, 2.1160e-8)),
    # metal sword f/d = 1005 +- 949 (g06 B1); q0 = 1500..5000 for steel/aluminium (g04 §2.3)
    _m("steel", 1005.0, 0.94, 3.15, 1e10, 0.08, 7850, 200e9, 0.3, (6.3035, 2.1160e-8)),
    # q0 = 5000 "aluminium" top of the scale (g04 §2.3) -> fd 1590; OB chime alpha2 = 0, alpha1 = 1e-7
    _m("aluminium", 1590.0, 0.55, 0.0, 1e10, 0.08, 2700, 69e9, 0.33, (0.0, 1e-7)),
    # church bell f/d read off the plot: 800-3500 (g06 B1)
    _m("bronze", 2000.0, 0.55, 0.0, 1e10, 0.08, 8800, 110e9, 0.34),
    # ice: polycrystalline ice bars in flexure, tan delta at the proton-relaxation peak 0.0085-0.0095 (Kuroiwa
    # 1964, Contrib. Inst. Low Temp. Sci. A18, §III Fig. 3); its Arrhenius fit (tau0 6.9e-16 s, 13.1 kcal/mol)
    # puts the peak at ~3 kHz at -10 C, so audio-band eta = 0.009 -> fd = 1/(pi eta) = 35.4.
    _m("ice", 35.4, 0.55, 15.0, 1.495e9, 0.2, 917, 9e9, 0.33),
    # stone: Barre and Quincy granite spheres, Q ~ 200 at 11-200 kHz (Birch 1975, JGR 80(5) 756) -> fd = Q/pi.
    _m("stone", 63.7, 0.55, 20.0, 1.495e9, 0.15, 2600, 50e9, 0.25),
    # concrete loss factor 0.02-0.05 (COMSOL "Damping in structural dynamics" table), order 1e-2 for dense
    # concrete (NPTEL 112104040 lecture 9.6, after Cremer & Heckl): eta = 0.02 -> fd = 1/(pi eta) = 15.9.
    _m("concrete", 15.9, 0.55, 40.0, 1.495e9, 0.2, 2400, 30e9, 0.2),
    # fd from beta at 1 kHz, ext = alpha/2, like plastic above. rho, E, nu, alpha, beta: Gao et al. 2022 Table 1 (polycarbonate),
    # Zheng & James 2011 Table 1 (polystyrene, MDF); Sterling & Lin 2016 Table 1 (Plastic 2: E, a1, a2; rho, nu = plastic's).
    # k and mu of all four are plastic's (MDF: wood's).
    _m("polycarbonate", 126.7, 0.5, 0.25, 3.3e7, 0.4, 1190, 2.4e9, 0.37, (0.5, 4e-7)),
    _m("polystyrene", 63.3, 0.5, 15.0, 3.3e7, 0.4, 1050, 3.5e9, 0.34, (30.0, 8e-7)),
    _m("mdf", 10.1, 0.63, 17.5, 2.24e8, 0.29, 615, 4e9, 0.32, (35.0, 5e-6)),
    _m("plastic2", 50.7, 0.5, 29.0, 3.3e7, 0.4, 1190, 2.4e9, 0.35, (58.0, 1e-6)),
    # G, nu of Johnson Table 5.1 (E = 2 G (1 + nu)). rho; damping, k, mu are metal's.
    _m("cast_iron", 477.0, 0.55, 3.15, 1e10, 0.08, 7200, 112.5e9, 0.25, (6.3035, 2.1160e-8)),
    _m("carbide", 477.0, 0.55, 3.15, 1e10, 0.08, 15000, 732e9, 0.22, (6.3035, 2.1160e-8)),
    _m("copper", 477.0, 0.55, 3.15, 1e10, 0.08, 8940, 123.4e9, 0.34, (6.3035, 2.1160e-8)),
    _m("acrylic", 58.0, 0.5, 26.3, 3.3e7, 0.4, 1200, 2.4e9, 0.37, (52.627, 8.7753e-7)),
)}
_ALIAS = {"aluminum": "aluminium", "porcelain": "ceramic", "iron": "metal", "plexiglass": "plastic"}

# Rayleigh pairs (alpha 1/s, beta s), d = (alpha + beta w^2)/2, of the ObjectFolder 2.0 audio simulation
# (Gao et al., CVPR 2022, supplementary Table 1; validated against recordings of the real objects).
# Glass rings 2.8 s at 1 kHz here against 0.58 s with Ren's glass bowl: a free glass, not one on a table.
OBJECTFOLDER_RAYLEIGH = {"ceramic": (6.0, 1e-7), "glass": (1.0, 1e-7), "wood": (60.0, 2e-6), "plastic": (30.0, 1e-6),
                        "metal": (5.0, 1e-7), "steel": (5.0, 3e-8), "polycarbonate": (0.5, 4e-7)}
# The same Table 1's density, E, nu: `law="objectfolder"` takes them with its pairs, so pitch and decay come from one object.
OBJECTFOLDER_ELASTIC = {"ceramic": (2700.0, 72e9, 0.19), "glass": (2600.0, 62e9, 0.20), "wood": (750.0, 11e9, 0.25),
                        "plastic": (1070.0, 1.4e9, 0.35), "metal": (8000.0, 210e9, 0.28),
                        "polycarbonate": (1190.0, 2.4e9, 0.37), "steel": (7850.0, 200e9, 0.29)}
# (alpha, beta) of O'Brien et al. 2002 Table 1: wood bowl (a2 50, a1 8e-6), ceramic rabbit (10, 1e-6), plastic shelf (50, 1e-6)
OBRIEN_RAYLEIGH = {"wood": (50.0, 8e-6), "ceramic": (10.0, 1e-6), "plastic": (50.0, 1e-6), "aluminium": (0.0, 1e-7)}
# (alpha, beta) and the contact-damping gamma (s/m) of Zheng & James 2011 Table 1 (C = alpha M + beta K + gamma G, eq. 19)
ZHENG_RAYLEIGH = {"ceramic": (6.0, 1e-7), "polystyrene": (30.0, 8e-7), "steel": (5.0, 3e-8), "mdf": (35.0, 5e-6), "wood": (60.0, 2e-6)}
CONTACT_GAMMA = {"ceramic": 3e-2, "polystyrene": 4e-4, "steel": 3e-1, "mdf": 9e-3, "wood": 5e-4}
# Sterling & Lin 2016 Table 1, the same objects as STERLING_POWER: Rayleigh (a1, a2), c = a1 + a2 w^2 (eq. 12);
# Caughey (eta1, eta2, eta3), c = eta1 + eta2 w^2 + eta3 w^4 (eq. 13). All d = c/2. Fitted inside the objects' bands.
STERLING_RAYLEIGH = {"plastic": (125.0, 8e-6), "plastic2": (58.0, 1e-6), "ceramic": (189.0, 1.5e-8), "wood": (35.0, 4.6e-7),
                     "aluminium": (0.225, 1.45e-6)}
STERLING_CAUGHEY = {"plastic": (280.0, -3.6e-7, 2.0e-15), "plastic2": (85.0, 6.6e-7, 5.6e-14), "ceramic": (420.0, -2.4e-6, 3.8e-15),
                    "wood": (277.0, -2.0e-6, 4.8e-15), "aluminium": (9.7, -3.4e-6, 5.8e-13)}
# (g1, g2, g3), c = g1 + g2 w^(2 g3) (Sterling, Rewkowski, Klatzky & Lin 2019, eq. 1-2 and Table 1: porcelain plate, travertine
# tile, wood block, steel wrench, plastic bowl, glass tile). Porcelain at 10 kHz: d = 20 (d = 12 with g3 = 1).
STERLING_HYBRID = {"ceramic": (3.9, 5.2e-9, 1.027), "stone": (1.3, 2.5e-8, 1.001), "wood": (39.0, 2.1e-7, 0.978),
                   "steel": (2.4, 5.5e-8, 1.011), "plastic": (34.83, 4.1e-7, 0.95), "glass": (1.9, 1.5e-7, 0.974)}
# Power-law damping c = mu1 w^(2 mu2), d = c/2, fitted to recordings of real objects (Sterling & Lin, I3D 2016,
# "Interactive modal sound synthesis using generalized proportional damping", Table 1: plastic 1 = rigid bowl,
# porcelain, wood block, aluminium). Listeners rated it below Rayleigh and Caughey (sec. 5.3.2, Table 2); kept for its compact form.
STERLING_POWER = {"plastic": (1.13, 0.30), "plastic2": (0.19, 0.37), "ceramic": (163.0, 0.01), "wood": (6.7, 0.18), "aluminium": (0.02, 0.445)}
# Aramaki, Besson, Kronland-Martinet & Ystad, IEEE TASLP 19(2) 2011: damping law alpha(w) = exp(aG + aR w) (eq. 4),
# reference sounds of the "material space" (Table II): (aG, aR) per category
ARAMAKI_REFS = {"glass": (2.5, 1.5e-4), "metal": (0.6, 2e-4), "wood": (3.0, 4e-4)}
DAMPING_LAWS = ("friction", "rayleigh", "objectfolder", "power", "obrien", "zheng", "sterling_rayleigh", "caughey", "hybrid")
_LAW_TABLES = {"objectfolder": OBJECTFOLDER_RAYLEIGH, "power": STERLING_POWER, "obrien": OBRIEN_RAYLEIGH, "zheng": ZHENG_RAYLEIGH,
               "sterling_rayleigh": STERLING_RAYLEIGH, "caughey": STERLING_CAUGHEY, "hybrid": STERLING_HYBRID}


def rolling_resistance(a: float, radius: float, tan_delta: float, shape: str = "cylinder") -> float:
    """Coefficient of rolling resistance, hysteresis theory, with the lost fraction alpha = 2.6 tan(delta) (peak of
    the viscoelastic solution): cylinder 2 alpha a/(3 pi R), ``a`` = contact half-width; sphere 3 alpha a/(16 R),
    ``a`` = contact radius."""
    alpha = 2.6 * float(tan_delta)
    if shape == "sphere":
        return 3.0 * alpha * float(a) / (16.0 * float(radius))
    if shape != "cylinder":
        raise ValueError("shape must be cylinder or sphere")
    return 2.0 * alpha * float(a) / (3.0 * math.pi * float(radius))


def loss_tangent(w, relax_t: float, beta: float = 1.0):
    """tan(delta)(w) = beta w T/(1 + (1 + beta) w^2 T^2) of the standard viscoelastic solid."""
    wt = np.asarray(w, float) * relax_t
    return beta * wt / (1.0 + (1.0 + beta) * wt * wt)


def ice_loss(freqs, temp_c: float = -10.0) -> np.ndarray:
    """Loss factor of ice, the proton relaxation: 0.009 x 2 w tau/(1 + (w tau)^2) + 0.002, tau = 6.9e-16 exp(54810/(R T))
    (Kuroiwa 1964, eqs 1-2: tau0, Q = 13.1 kcal/mol; peak at 3 kHz at -10 C). The floor is read off his Fig. 6, approximate;
    the rise near 0 C (grain boundaries) is not modelled."""
    wt = TWO_PI * np.asarray(freqs, float) * 6.9e-16 * math.exp(54810.0 / (8.314 * (273.15 + float(temp_c))))
    return 0.009 * 2.0 * wt / (1.0 + wt * wt) + 0.002


def aramaki_point(angle_deg: float, radius: float = 1.0) -> tuple[float, float]:
    """(aG, aR) at a point of Aramaki et al.'s material disk: glass at 0 degrees, metal at 120, wood at 240 on
    the rim; along the rim only two neighbours are mixed (eqs 20-21); inside, (1 - r) P_C + r P(theta) with C the
    mean of the three (eq. 22)."""
    th = math.radians(float(angle_deg)) % (2 * math.pi)

    def T(t):
        t %= 2 * math.pi
        if t < 2 * math.pi / 3:
            return 1.0 - 3 * t / (2 * math.pi)
        if t <= 4 * math.pi / 3:
            return 0.0
        return 3 * t / (2 * math.pi) - 2.0
    g, m, w = ARAMAKI_REFS["glass"], ARAMAKI_REFS["metal"], ARAMAKI_REFS["wood"]
    tg, tm, tw = T(th), T(th - 2 * math.pi / 3), T(th - 4 * math.pi / 3)
    rim = [tg * g[i] + tm * m[i] + tw * w[i] for i in (0, 1)]
    r = float(np.clip(radius, 0.0, 1.0))
    c = [(g[i] + m[i] + w[i]) / 3.0 for i in (0, 1)]
    return (1 - r) * c[0] + r * rim[0], (1 - r) * c[1] + r * rim[1]


WOOD_SPECIES = {   # specific gravity, specific modulus E'/gamma (GPa), tan delta (1e-3): air-dry heartwood, axial
    "spruce": (0.50, 24.4, 8.4), "sitka_spruce": (0.47, 32.2, 5.9), "fir": (0.54, 30.0, 6.7),
    "scots_pine": (0.48, 17.9, 10.6), "caribbean_pine": (0.36, 9.0, 13.2), "yew": (0.64, 11.1, 10.0),
    "red_cedar": (0.39, 19.5, 7.2), "sycamore": (0.64, 10.6, 16.1), "maple": (0.72, 10.5, 11.5),
    "boxwood": (0.93, 9.6, 15.3), "hornbeam": (0.74, 20.8, 9.5), "walnut": (0.58, 12.7, 8.0),
    "cherry": (0.62, 17.4, 10.0), "pear": (0.69, 9.9, 16.7), "strawberry_tree": (0.94, 12.0, 18.7),
    "madagascar_rosewood": (0.96, 20.1, 4.6), "african_blackwood": (1.31, 15.7, 5.8),
    "brazilian_rosewood": (0.81, 20.5, 7.0), "cocobolo": (0.95, 11.8, 8.6), "padauk": (0.77, 18.3, 5.2),
    "kosso": (0.83, 19.3, 5.6), "burma_padauk": (0.75, 12.9, 6.8), "pernambuco": (1.01, 21.5, 4.4),
    "snakewood": (1.31, 22.5, 5.4), "aniba": (1.20, 29.1, 3.9), "ebony": (1.18, 16.8, 9.1),
    "mahogany": (0.48, 20.7, 6.4), "khaya": (0.85, 9.2, 12.0), "spanish_cedar": (0.48, 21.8, 7.8),
    "kapok": (0.22, 10.1, 14.0), "courbaril": (0.77, 28.4, 5.3), "swartzia": (1.21, 26.3, 4.3),
}
ONO_NORIMOTO = {"standard": (1.23, 0.68), "all": (1.322, 0.692), "softwood": (1.222, 0.639),
                "hardwood": (1.231, 0.677), "hardwood_extractive": (1.474, 0.620), "hardwood30": (1.34, 0.62)}
WOOD_SHIFT = {"normal": 1.0, "compression": 1.30, "extractive": 0.67}


def wood_eta(specific_modulus: float, group: str = "standard", kind: str = "normal") -> float:
    """Loss factor of air-dry wood along the grain from its specific modulus E'/gamma (GPa):
    tan delta = 10^-A (E'/gamma)^-B, (A, B) per ``group``; ``kind`` compression wood x1.3, rich in extractives x0.67."""
    A, B = ONO_NORIMOTO[group]
    return 10.0 ** -A * float(specific_modulus) ** -B * WOOD_SHIFT[kind]


def wood_species(name: str) -> ContactMaterial:
    """The wood ContactMaterial of a species (`material="wood:<species>"`): density = 1000 gamma, E = gamma E'/gamma, fd = 1/(pi tan delta)."""
    g, sm, td = WOOD_SPECIES[name]
    return replace(MATERIALS["wood"], density=1000.0 * g, young=g * sm * 1e9, fd=1.0 / (math.pi * td * 1e-3))


def get_material(name) -> ContactMaterial:
    if isinstance(name, ContactMaterial):
        return name
    s = str(name).lower()
    if s.startswith("wood:"):
        if s[5:] not in WOOD_SPECIES:
            raise ValueError(f"unknown wood {s[5:]!r}; choose from {sorted(WOOD_SPECIES)}")
        return wood_species(s[5:])
    key = _ALIAS.get(s, s)
    if key not in MATERIALS:
        raise ValueError(f"unknown material {name!r}; choose from {sorted(MATERIALS)}")
    return MATERIALS[key]


def aramaki_damping(freqs, angle: float = 240.0, radius: float = 1.0) -> np.ndarray:
    """Mode decay rates (1/s) alpha(w) = exp(aG + aR w), w = 2 pi f, at ``angle``/``radius`` of the material disk
    (`aramaki_point`). At 1 kHz: metal 6.4/s (T60 1.1 s), glass 31/s (0.22 s), wood 247/s (28 ms)."""
    ag, ar = aramaki_point(angle, radius)
    return np.exp(ag + ar * TWO_PI * np.asarray(freqs, float))


def material_damping(freqs, mat="steel", *, law: str = "friction", ext: float | None = None,
                     scatter: float | None = None, seed: int = 0) -> np.ndarray:
    """Decay rate d_n (1/s, amplitude envelope exp(-d t); T60 = 6.91/d) of each mode.

    ``law="friction"``: ``d_n = pi f_n tan(phi) / J_n + ext`` - the Wildes-Richards internal
    friction law (g06 A5 eq. 12-13; g04 §5.1) with the frequency-independent external loss, and
    per-mode log-normal scatter ``J_n`` of relative spread ``scatter`` ("draw each mode's damping
    as d = f/R with R scattered 50-100 %", g06 B1). ``law="rayleigh"``:
    ``d_n = (alpha + beta (2 pi f_n)^2)/2`` (O'Brien eq. 9, Ren eq. 6; g07 §1.1, §4.1).
    ``law="objectfolder"``: the same with the ObjectFolder 2.0 pairs. ``law="power"``:
    ``d_n = mu1 (2 pi f_n)^(2 mu2) / 2`` (Sterling & Lin 2016). These two carry the losses of the
    measured objects; ``ext`` (extra support loss) is added only when given. ``rayleigh`` ignores ``ext``.
    """
    mat = get_material(mat)
    f = np.asarray(freqs, float)
    if law not in DAMPING_LAWS:
        raise ValueError(f"unknown damping law {law!r}; choose from {DAMPING_LAWS}")
    if law != "friction":
        table = {mat.name: mat.rayleigh} if law == "rayleigh" and mat.rayleigh else _LAW_TABLES.get(law, {})
        if mat.name not in table:
            raise ValueError(f"no sourced {law} damping for {mat.name}; materials with one: {sorted(table)}")
        p = table[mat.name]
        w = TWO_PI * f
        if law == "power":
            d = 0.5 * p[0] * w ** (2 * p[1])
        elif law == "caughey":
            d = 0.5 * (p[0] + p[1] * w ** 2 + p[2] * w ** 4)
        elif law == "hybrid":
            d = 0.5 * (p[0] + p[1] * w ** (2 * p[2]))
        else:
            d = 0.5 * (p[0] + p[1] * w ** 2)
        return d + (0.0 if ext is None or law == "rayleigh" else float(ext))
    cv = mat.scatter if scatter is None else float(scatter)
    j = np.ones_like(f)
    if cv > 0:
        s = math.sqrt(math.log1p(cv * cv))
        j = np.exp(np.random.default_rng(seed).normal(-0.5 * s * s, s, f.shape))   # mean 1
    return math.pi * f * mat.tan_phi / j + (mat.ext if ext is None else float(ext))


# ==========================================================================================
# Analytic shapes (g06 A6, B4; formulas: Morse, "Vibration and Sound"; Fletcher & Rossing)
# ==========================================================================================

SHAPES = ("string", "bar", "rect_membrane", "rect_plate", "circ_membrane", "circ_plate", "circ_plate_free", "l_membrane")
BOUNDARIES = ("free", "clamped", "supported", "clamped_clamped")     # bars; "clamped" = clamped-free


def _bar_roots(n: int, boundary: str) -> np.ndarray:
    """Roots b_n of cos b cosh b = 1 (free-free, clamped-clamped) or = -1 (clamped-free)."""
    if boundary == "supported":
        return np.pi * np.arange(1, n + 1)
    sgn, off = (1.0, 0.5) if boundary == "clamped" else (-1.0, 1.5)
    return np.array([brentq(lambda b: math.cos(b) + sgn / math.cosh(b), (i + off) * np.pi - 0.45,
                            (i + off) * np.pi + 0.45) for i in range(n)])


def _bar_shape(b: np.ndarray, x: float, boundary: str) -> np.ndarray:
    """Euler-Bernoulli mode shapes at x in [0, 1], unit mean square. Written with
    cosh(bx) - s sinh(bx) = exp(-bx) + (1 - s) sinh(bx) so high modes do not cancel."""
    if boundary == "supported":
        return math.sqrt(2.0) * np.sin(b * x)
    if boundary == "clamped":                                # clamped at 0, free at 1
        oms = (-np.exp(-b) + np.sin(b) - np.cos(b)) / (np.sinh(b) + np.sin(b))
    else:
        oms = (-np.exp(-b) + np.cos(b) - np.sin(b)) / (np.sinh(b) - np.sin(b))
    s = 1.0 - oms
    hyp = np.exp(-b * x) + oms * np.sinh(b * x)
    if boundary == "free":
        return hyp + np.cos(b * x) - s * np.sin(b * x)
    return hyp - np.cos(b * x) + s * np.sin(b * x)


def _plate_g(k, m):
    return special.jv(m, k) * special.ive(m + 1, k) + special.ive(m, k) * special.jv(m + 1, k)


def _plate_radial(m, k, r):
    """Clamped circular plate radial shape J_m(kr) - (J_m(k)/I_m(k)) I_m(kr), scaled Bessel I."""
    return special.jv(m, k * r) - special.jv(m, k) * special.ive(m, k * r) / special.ive(m, k) * np.exp(k * (r - 1.0))


def _free_disc_terms(k, m, nu):
    """Free-edge Kirchhoff disc, w = (A J_m(kr) + B I_m(kr) e^-k) cos m theta at r = 1: radial moment and
    effective shear rows (mj, mi), (vj, vi)."""
    J, Jp, Jpp = special.jv(m, k), special.jvp(m, k, 1), special.jvp(m, k, 2)
    e = np.exp(-k)
    I, Ip, Ipp = special.ive(m, k), special.ivp(m, k, 1) * e, special.ivp(m, k, 2) * e
    mj = k * k * Jpp + nu * (k * Jp - m * m * J)
    mi = k * k * Ipp + nu * (k * Ip - m * m * I)
    vj = -k ** 3 * Jp - (1 - nu) * m * m * (k * Jp - J)
    vi = k ** 3 * Ip - (1 - nu) * m * m * (k * Ip - I)
    return mj, mi, vj, vi


def _free_disc_g(k, m, nu):
    mj, mi, vj, vi = _free_disc_terms(k, m, nu)
    return mj * vi - mi * vj


def _free_disc_radial(m, k, r, nu=0.3):
    mj, mi, _, _ = _free_disc_terms(k, m, nu)
    return special.jv(m, k * r) - mj / mi * special.ive(m, k * r) * np.exp(k * (r - 1.0))


def _free_disc_roots(m, nu, kmax):
    grid = np.arange(0.5, kmax + 0.02, 0.01)
    v = _free_disc_g(grid, m, nu)
    idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    return [brentq(_free_disc_g, grid[i], grid[i + 1], args=(m, nu)) for i in idx]


@lru_cache(maxsize=64)
def free_disc_omega(mode=(2, 0), nu: float = 0.3) -> float:
    """Dimensionless omega = k^2 of mode (m nodal diameters, n nodal circles) of a free-edged circular plate,
    omega = 2 pi f a^2 sqrt(rho h / D), from the free-edge frequency equation (zero radial moment and Kirchhoff shear)."""
    m, n = int(mode[0]), int(mode[1])
    i = n - 1 if m < 2 else n                               # (0, 0) and (1, 0) are rigid-body motions (k = 0)
    if i < 0:
        raise ValueError(f"mode {mode} is a rigid-body motion of a free disc")
    ks = _free_disc_roots(m, float(nu), 6.0 + 1.2 * m + math.pi * (n + 1.5))
    return float(ks[i] ** 2)


def airy_disc_roots(l: int, n: int = 6) -> np.ndarray:
    """First ``n`` roots x of J_{l-1}(x) I_l(x) - I_{l-1}(x) J_l(x) = 0: the in-plane (Airy stress) modes Psi(l, i)
    of a free-edged disc (written J_l I_{l+1} + I_l J_{l+1} by the Bessel recurrences)."""
    kmax = 6.0 + 1.2 * l + math.pi * (n + 1.5)
    grid = np.arange(0.5, kmax, 0.01)
    v = _plate_g(grid, l)
    idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0][:n]
    return np.array([brentq(_plate_g, grid[i], grid[i + 1], args=(l,)) for i in idx])


def vk_disc_couplings(k: int) -> tuple[int, ...]:
    """Orders l of the Airy modes Psi(l, i) that a transverse disc mode with k nodal diameters drives
    through the von Karman term: (0, n) only l = 0; (k, n) l = 0 and 2k."""
    return (0,) if int(k) == 0 else (0, 2 * int(k))


_L_LAMBDA = (9.6397238440219, 15.19725192, 2 * math.pi ** 2, 29.5214811, 31.9126360, 41.4745099, 44.948488,
             5 * math.pi ** 2, 5 * math.pi ** 2, 56.709610, 65.376535, 71.057755)


@lru_cache(maxsize=4)
def _l_modes(n_modes: int = 48, n: int = 64):
    """L-shaped membrane (three unit squares, fixed edge) on [-1, 1]^2 minus the quadrant x, y > 0: eigenvalues
    of -Laplacian and mode shapes (unit mean square) on a 5-point grid, h = 1/n. The tabulated eigenvalues
    replace the grid's where known; the rest are scaled by the grid's error on the last tabulated one."""
    from scipy.sparse import diags, identity, kron
    from scipy.sparse.linalg import eigsh
    G, h = 2 * n - 1, 1.0 / n
    x = -1.0 + h * np.arange(1, 2 * n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    keep = ~((X > -1e-12) & (Y > -1e-12))
    D = diags([-np.ones(G - 1), 2.0 * np.ones(G), -np.ones(G - 1)], [-1, 0, 1]) / h ** 2
    k = np.flatnonzero(keep.ravel())
    A = (kron(D, identity(G)) + kron(identity(G), D)).tocsr()[k][:, k]
    lam, V = eigsh(A, k=max(len(_L_LAMBDA), n_modes), sigma=0.0, which="LM")
    o = np.argsort(lam)
    lam, V = lam[o], V[:, o]
    t = len(_L_LAMBDA)
    lam = np.concatenate([_L_LAMBDA, lam[t:] * _L_LAMBDA[-1] / lam[t - 1]])
    shapes = np.zeros((len(lam), 2 * n + 1, 2 * n + 1))
    full = np.zeros((len(lam), G * G))
    full[:, k] = V.T
    shapes[:, 1:-1, 1:-1] = full.reshape(len(lam), G, G)
    shapes *= np.sqrt(len(k))
    return lam, shapes


@lru_cache(maxsize=64)
def _shape_table(shape: str, boundary: str, aspect: float, n_modes: int, rmax: float):
    """(ratios to the lowest mode, descriptor arrays), lowest ``n_modes`` modes with ratio <= rmax."""
    n_modes = int(max(1, min(n_modes, 400)))
    if shape == "string":
        desc = {"n": np.arange(1, n_modes + 1, dtype=float)}
        ratio = desc["n"].copy()
    elif shape == "bar":
        b = _bar_roots(min(n_modes, 200), boundary)
        desc, ratio = {"b": b}, (b / b[0]) ** 2
    elif shape in ("rect_membrane", "rect_plate"):
        r2 = float(aspect) ** 2
        qmax = rmax * rmax if shape == "rect_membrane" else rmax
        top = int(min(80, math.ceil(math.sqrt(qmax * (1 + r2)) / min(1.0, float(aspect))) + 1))
        mm, nn = np.meshgrid(np.arange(1, top + 1), np.arange(1, top + 1), indexing="ij")
        q = (mm.ravel() ** 2 + r2 * nn.ravel() ** 2) / (1.0 + r2)
        desc = {"m": mm.ravel().astype(float), "n": nn.ravel().astype(float)}
        ratio = np.sqrt(q) if shape == "rect_membrane" else q
    elif shape == "circ_membrane":
        j01 = special.jn_zeros(0, 1)[0]
        ms, js = [], []
        for m in range(0, 60):
            z = special.jn_zeros(m, 30)
            if z[0] / j01 > rmax and m > 0:
                break
            ms += [m] * len(z)
            js += list(z)
        ms, js = np.array(ms, float), np.array(js)
        # mean square over the disc of J_m(j r) cos(m theta): J_{m+1}(j)^2 (x 1/2 for m > 0)
        norm = special.jv(ms + 1, js) ** 2 * np.where(ms == 0, 1.0, 0.5)
        desc, ratio = {"m": ms, "j": js, "norm": norm}, js / j01
    elif shape == "circ_plate":
        k01 = brentq(_plate_g, 3.0, 3.4, args=(0,))
        kmax = min(math.sqrt(rmax) * k01 + 0.1, 1.3 * math.pi * math.sqrt(n_modes) + 6.0, 70.0)
        grid = np.arange(0.5, kmax + 0.06, 0.05)
        ms, ks = [], []
        for m in range(0, 80):
            v = _plate_g(grid, m)
            idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
            if len(idx) == 0:
                break
            ms += [m] * len(idx)
            ks += [brentq(_plate_g, grid[i], grid[i + 1], args=(m,)) for i in idx]
        ms, ks = np.array(ms, float), np.array(ks)
        rr = (np.arange(600) + 0.5) / 600.0
        norm = np.array([2.0 * np.mean(_plate_radial(m, k, rr) ** 2 * rr) for m, k in zip(ms, ks)])
        norm *= np.where(ms == 0, 1.0, 0.5)
        desc, ratio = {"m": ms, "k": ks, "norm": norm}, (ks / k01) ** 2
    elif shape == "circ_plate_free":                        # nu = 0.3 for the ratios; f1 takes the material's
        k20 = math.sqrt(free_disc_omega((2, 0), 0.3))
        kmax = min(math.sqrt(rmax) * k20 + 0.1, 1.3 * math.pi * math.sqrt(n_modes) + 6.0, 70.0)
        ms, ks = [], []
        for m in range(0, int(kmax) + 3):                   # (m, 0) lies above k = m: no root past that
            r = [k for k in _free_disc_roots(m, 0.3, kmax) if k <= kmax]
            ms += [m] * len(r)
            ks += r
        ms, ks = np.array(ms, float), np.array(ks)
        rr = (np.arange(600) + 0.5) / 600.0
        norm = np.array([2.0 * np.mean(_free_disc_radial(m, k, rr) ** 2 * rr) for m, k in zip(ms, ks)])
        norm *= np.where(ms == 0, 1.0, 0.5)
        desc, ratio = {"m": ms, "k": ks, "norm": norm}, (ks / k20) ** 2
    elif shape == "l_membrane":
        lam = _l_modes()[0]
        desc, ratio = {"k": np.arange(len(lam), dtype=float)}, np.sqrt(lam / lam[0])
    else:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    order = np.argsort(ratio, kind="stable")
    order = order[ratio[order] <= rmax * (1 + 1e-9)][:n_modes]
    if len(order) == 0:
        order = np.argsort(ratio, kind="stable")[:1]
    return ratio[order] / ratio[order[0]], {k: v[order] for k, v in desc.items()}


def _thickness(shape: str, size: float) -> float:
    # geometric proportions of a "typical" object (bar 4 %, plate 2 % of its length).
    return (0.04 if shape == "bar" else 0.02) * size


FREE_DISC_OMEGA = {(2, 0): 5.093, (0, 1): 9.175, (3, 0): 11.90, (50, 0): 2687.9, (0, 18): 3196.8, (24, 8): 3352.1}   # the frequency equation gives these at nu = 0.38


def free_disc_freq(diameter: float, thickness: float, mat="steel", mode=(2, 0)) -> float:
    """Frequency (Hz) of mode (k, n) of a free-edged circular plate: omega_kn sqrt(D/(rho h))/(2 pi a^2),
    D = E h^3/(12 (1 - nu^2)), omega_kn from `free_disc_omega` at the material's Poisson ratio."""
    mat = get_material(mat)
    a = 0.5 * float(diameter)
    w = free_disc_omega(tuple(int(v) for v in mode), round(float(mat.poisson), 6))
    return w * float(thickness) * mat.c / math.sqrt(12.0 * (1.0 - mat.poisson ** 2)) / (TWO_PI * a * a)


def cut_length(freq: float, mat="steel", *, boundary: str = "free", thickness: float | None = None,
               tube: tuple[float, float] | None = None) -> float:
    """Length (m) of a bar or tube whose lowest bending mode is ``freq``: the inverse of `fundamental` for a bar,
    L = sqrt(b1^2 c r_g / (2 pi f)), r_g = h / sqrt(12) (solid) or sqrt(ro^2 + ri^2) / 2 (tube)."""
    mat = get_material(mat)
    b1 = _bar_roots(1, boundary)[0]
    rg = math.hypot(*tube) / 2.0 if tube else float(thickness if thickness else 0.01) / math.sqrt(12.0)
    return math.sqrt(b1 * b1 * mat.c * rg / (TWO_PI * float(freq)))


def fundamental(shape: str, size: float, mat="steel", *, boundary: str = "free", aspect: float = 1.0,
                thickness: float | None = None, wave_speed: float = 120.0, tube: tuple[float, float] | None = None) -> float:
    """Lowest mode frequency (Hz) of a shape of length / diameter ``size`` (m).

    Bending bodies (bar, plates) follow from the material: bar ``b1^2/(2 pi L^2) sqrt(E/rho) h/sqrt(12)``,
    simply supported plate ``(pi/2) h c_p (1/Lx^2 + 1/Ly^2)``, clamped circular plate
    ``k01^2/(2 pi a^2) h c_p`` with ``c_p = sqrt(E/(12 rho (1 - nu^2)))`` (Fletcher & Rossing ch. 2-3).
    Tension bodies (string, membranes) need a transverse ``wave_speed`` (m/s).
    """
    mat = get_material(mat)
    L = max(float(size), 1e-3)
    h = thickness if thickness else _thickness(shape, L)
    cp = mat.c / math.sqrt(12.0 * (1.0 - mat.poisson ** 2))
    if shape == "bar":
        b1 = _bar_roots(1, boundary)[0]
        if tube:
            return b1 * b1 / (TWO_PI * L * L) * mat.c * math.hypot(*tube) / 2.0
        return b1 * b1 / (TWO_PI * L * L) * mat.c * h / math.sqrt(12.0)
    if shape == "rect_plate":
        return 0.5 * math.pi * h * cp * (1.0 / L ** 2 + (aspect / L) ** 2)
    if shape == "circ_plate":
        return 3.19622 ** 2 / (TWO_PI * (0.5 * L) ** 2) * h * cp
    if shape == "circ_plate_free":
        return free_disc_freq(L, h, mat, (2, 0))
    if shape == "string":
        return wave_speed / (2.0 * L)                       # default speed: tension is free
    if shape == "rect_membrane":
        return 0.5 * wave_speed * math.sqrt(1.0 / L ** 2 + (aspect / L) ** 2)
    if shape == "circ_membrane":
        return wave_speed * 2.404826 / (TWO_PI * 0.5 * L)
    if shape == "l_membrane":                               # size = the outer side, two unit squares
        return wave_speed * math.sqrt(_L_LAMBDA[0]) / (TWO_PI * 0.5 * L)
    raise ValueError(f"unknown shape {shape!r}")


def body_mass(shape: str, size: float, mat="steel", *, aspect: float = 1.0, thickness: float | None = None) -> float:
    """Mass (kg) of the shape, used as the modal mass of every (unit mean-square) mode."""
    mat = get_material(mat)
    L = max(float(size), 1e-3)
    h = thickness if thickness else _thickness(shape, L)
    if shape == "bar":
        return mat.density * L * 2.5 * h * h                # width 2.5 x thickness
    if shape == "rect_plate":
        return mat.density * L * L / aspect * h
    if shape in ("circ_plate", "circ_plate_free"):
        return mat.density * math.pi * (0.5 * L) ** 2 * h
    if shape == "string":
        return 0.005 * L
    area = L * L / aspect if shape == "rect_membrane" else 0.75 * L * L if shape == "l_membrane" else math.pi * (0.5 * L) ** 2
    return 0.262 * area                                     # timpani head 0.262 kg/m^2 (Christian et al. 1984)


def critical_bandwidth(f):
    """Critical bandwidth in Hz: 25 + 75 (1 + 1.4 f^2)^0.69, f in kHz."""
    fk = np.asarray(f, float) / 1000.0
    return 25.0 + 75.0 * (1.0 + 1.4 * fk * fk) ** 0.69


def classify_material(a_g: float, a_r: float) -> str:
    """Perceived material (metal, wood or glass) of a decay law alpha(w) = exp(a_g + a_r w), from the three
    quadratic class boundaries (a_r in 1e-4 s units inside; positive side = metal over glass, wood over metal,
    wood over glass)."""
    r, g = float(a_r) * 1e4, float(a_g)
    b1 = 109.69 - 48.44 * r - 48.30 * g + 5.33 * r * r + 11.33 * r * g + 3.37 * g * g
    b2 = -41.18 + 1.50 * r + 17.75 * g + 0.27 * r * r - 0.09 * r * g - 0.20 * g * g
    b3 = 68.51 - 46.94 * r - 30.55 * g + 5.60 * r * r + 11.24 * r * g + 3.17 * g * g
    if b2 > 0 and b3 > 0:
        return "wood"
    return "metal" if b1 > 0 and b2 <= 0 else "glass"


def am_sidebands(body: "ModalObject", index: float, fmax: float = 20000.0) -> "ModalObject":
    """The body with every mode amplitude-modulated at a quarter of its critical band: (1 + I cos wn t) cos wm t
    = cos wm t + I/2 cos (wm +- wn) t, i.e. two side modes of the same decay at I/2 the shape."""
    fn = 0.25 * critical_bandwidth(body.freqs)
    keep = body.freqs + fn < fmax
    f = np.concatenate([body.freqs, (body.freqs - fn)[keep], (body.freqs + fn)[keep]])
    d = np.concatenate([body.decays, body.decays[keep], body.decays[keep]])
    phi = np.concatenate([body.phi, 0.5 * index * body.phi[keep], 0.5 * index * body.phi[keep]])
    aw = np.concatenate([body.aw0, body.aw0[keep], body.aw0[keep]])
    out = ModalObject(f, d, phi, mass=body.mass, aw0=aw, name=body.name + "+am")
    out.radiator_radius = body.radiator_radius
    if body.multipole is not None:
        out.multipole = np.concatenate([body.multipole, body.multipole[keep], body.multipole[keep]])
    return out


def bark(f):
    """Critical-band rate (Bark) of a frequency in Hz."""
    f = np.asarray(f, float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def masked_modes(freqs, energy) -> np.ndarray:
    """True for each mode lying under the masking threshold of a stronger one. Spreading over the Bark
    distance dz = z_mode - z_masker: 15.81 + 7.5 (dz + 0.474) - 17.5 sqrt(1 + (dz + 0.474)^2) dB, minus the
    tonal-masker offset 14.5 + z_masker dB."""
    z = bark(freqs)
    lv = 10.0 * np.log10(np.asarray(energy, float) + 1e-300)
    dz = z[:, None] - z[None, :]                          # row: masked mode, column: masker
    x = dz + 0.474
    thr = lv[None, :] + 15.81 + 7.5 * x - 17.5 * np.sqrt(1.0 + x * x) - (14.5 + z[None, :])
    np.fill_diagonal(thr, -np.inf)
    return np.any(lv[:, None] < thr, axis=1)


# ==========================================================================================
# Modal objects
# ==========================================================================================

class ModalObject:
    """A set of modes: ``freqs`` (Hz), ``decays`` (1/s), ``phi`` (mode shape at the contact point,
    unit mean square over the body) and one modal ``mass`` (kg).

    Each mode obeys ``x'' + 2 d x' + w^2 x = phi F / mass`` (Sounding Object eq. 8.1-8.2); the
    displacement of the contact point is ``sum phi x`` (eq. 8.4). The heard signal is
    ``sum aw x`` with ``aw = aw0 (f/f1)^(2 bright)``: ``bright = 0`` is the displacement, van den
    Doel's amplitudes ``a_n = Psi_n(p)/w_n`` (eq. 10-11; g06 A3-A4); ``0.5`` the velocity, i.e. the
    frequency-weighted radiation of O'Brien et al. (g07 §1.5) and the thesis's brighter ``u + v``
    tap (g06 B3); ``1`` the acceleration.
    """

    def __init__(self, freqs, decays, phi, *, mass: float = 1.0, aw0=None, name: str = "modes"):
        self.freqs = np.ascontiguousarray(freqs, dtype=np.float64)
        self.decays = np.ascontiguousarray(decays, dtype=np.float64)
        self.phi = np.ascontiguousarray(phi, dtype=np.float64)
        self.aw0 = np.ones_like(self.freqs) if aw0 is None else np.ascontiguousarray(aw0, dtype=np.float64)
        self.mass = float(mass)
        self.name = name
        self._desc: dict = {}
        self.radiator_radius: float | None = None   # m; set to roll off what a small body cannot radiate
        self.multipole: np.ndarray | None = None     # per mode, share of |phi| that is net volume flow

    @classmethod
    def from_table(cls, table, f_scale: float = 1.0, position: float = 0.0, mass: float = 1.0) -> "ModalObject":
        """A measured ``genny.physical.modal_data`` table (e.g. ``"sy_vase"``) as a modal object.
        Measured gains are taken as ``a_n = phi_n^2 / w_n`` (drive and pick-up at the same point)."""
        m = _modal.resolve(table, f_scale=f_scale, position=position)
        amp = np.abs(m.gains) / m.norm
        phi = np.sqrt(amp * TWO_PI * m.freqs)
        phi /= math.sqrt(float(np.mean(phi ** 2))) + 1e-30
        return cls(m.freqs, m.decays, phi, mass=mass, aw0=phi, name=str(table))

    def row(self, position=None) -> np.ndarray:
        return self.phi

    def audio_weights(self, bright: float = 0.5) -> np.ndarray:
        w = self.aw0 * (self.freqs / self.freqs.min()) ** (2.0 * float(np.clip(bright, 0.0, 1.0)))
        if self.radiator_radius:
            # a body of radius a radiates like a pulsating sphere: |p| ~ ka / sqrt(1 + (ka)^2), rising 6 dB/oct
            # below ka = 1 and flat above (Chaigne & Kergomard, Acoustics of Musical Instruments, eq. 12.5)
            ka = TWO_PI * self.freqs * self.radiator_radius / 343.2
            w = w * ka / np.sqrt(1.0 + ka * ka)
            if getattr(self, "multipole", None) is not None:
                q = self.multipole
                w = w * (q + (1.0 - q) * np.minimum(1.0, ka / math.sqrt(3.0)))
        return w

    def gains(self, position=None, bright: float = 0.0) -> np.ndarray:
        """van den Doel's heard amplitudes ``a_n = Psi_n(p) / (w_n Q(p))``, ``Q = sqrt(sum Psi^2)``
        (eq. 11; g06 A4): every strike point delivers the same energy."""
        p = self.row(position)
        q = math.sqrt(float(np.sum(p * p))) + 1e-30
        return p * self.audio_weights(bright) / (TWO_PI * self.freqs * q)

    def rank(self, position=None, bright: float = 0.5, energy: bool = False) -> np.ndarray:
        """Mode indices, most important first: ear-weighted mode energy ``(a_n f_n)^2 S(f_n)``
        (sonic-distance energy of g06 A7 with genny's ear weighting as S; "add modes in decreasing
        order of importance", g06 B3/B9). ``energy``: times the time integral of the decaying mode,
        E_m = 1 / (4 d_n), so a heavily damped mode weighs less."""
        e = (self.gains(position, bright) * self.freqs * _ear_weight(self.freqs)) ** 2
        if energy:
            e = e / (4.0 * np.maximum(self.decays, 1e-6))
        return np.argsort(-e, kind="stable")

    def subset(self, idx) -> "ModalObject":
        idx = np.sort(np.asarray(idx))
        o = copy.copy(self)
        for k in ("freqs", "decays", "phi", "aw0"):
            setattr(o, k, np.ascontiguousarray(getattr(self, k)[idx]))
        o._desc = {k: v[idx] for k, v in self._desc.items()}
        return o

    def pruned(self, keep: int | None = None, floor_db: float | None = None, bright: float = 0.5,
               masking: bool = False) -> "ModalObject":
        """Keep the ``keep`` most important modes and / or those within ``floor_db`` of the strongest
        (drop a mode whose log-energy is below the audible floor, g06 A7/B9). ``masking``: also drop a
        mode hidden under a stronger neighbour (spreading function over Bark distance, tonal masker
        offset 14.5 + z dB)."""
        order = self.rank(bright=bright)
        e = (self.gains(bright=bright) * self.freqs * _ear_weight(self.freqs)) ** 2
        if floor_db is not None:
            order = order[10 * np.log10(e[order] / (e[order[0]] + 1e-300) + 1e-300) >= -abs(floor_db)]
        if masking and order.size > 1:
            order = order[~masked_modes(self.freqs[order], e[order])]
        if keep is not None:
            order = order[:max(1, int(keep))]
        return self.subset(order)

    def t60(self, bright: float = 0.5, within_db: float = 30.0) -> float:
        """Longest T60 among the modes within ``within_db`` of the strongest one."""
        a = np.abs(self.gains(bright=bright))
        ok = a >= a.max() * 10 ** (-within_db / 20)
        return float(6.91 / max(self.decays[ok].min(), 1e-3))

    def as_modes(self, bright: float = 0.0) -> "_modal.Modes":
        """The same body for ``genny.physical.modal.render_modes`` ("vdd" form)."""
        return _modal.Modes(self.freqs, self.decays, self.gains(bright=bright), "vdd")

    def ring(self, force, sr: int = DEFAULT_SR, bright: float = 0.5):
        """Drive the body with a force signal: ``(audio, displacement, velocity)`` at the contact point."""
        return modal_bank(force, self, sr, bright)

    def strike(self, velocity: float = 1.5, *, mass: float = 0.02, k: float | None = None, mu: float | None = None,
               alpha: float = 1.5, dur: float = 1.0, sr: int = DEFAULT_SR, bright: float = 0.5) -> np.ndarray:
        """One Hunt-Crossley strike by a point mass (see `ImpactInteractor`)."""
        return ImpactInteractor(self, mass=mass, k=1.495e9 if k is None else k, mu=0.1 if mu is None else mu,
                                alpha=alpha, bright=bright).strike(velocity, dur, sr).audio


class ShapeBody(ModalObject):
    """Analytic modal body with true strike-position gains (van den Doel & Pai 1996; g06 A2-A6, B4).

    ``shape``: ``string`` (sin, harmonic), ``bar`` (Euler-Bernoulli; ``boundary`` ``free`` |
    ``clamped`` (clamped-free) | ``supported`` | ``clamped_clamped``), ``rect_membrane``,
    ``rect_plate`` (simply supported; ``aspect`` = Lx/Ly), ``circ_membrane`` (Bessel zeros),
    ``circ_plate`` (clamped edge), ``circ_plate_free`` (free edge: lids, coins, cymbals). ``position``: x in 0..1 (string, bar), (x, y) (rectangles) or r
    / (r, theta) with 0 = centre (circles). ``n_modes`` / ``fmax`` truncate the series (the
    authors' fix for the divergent delta strike, g06 A3). Damping: `material_damping`.
    ``excitation="plucked"`` adds the extra 1/n of a displacement-initialised body (g06 A6).
    """

    def __init__(self, shape: str = "bar", f1: float = 440.0, *, material="steel", boundary: str = "free",
                 aspect: float = 1.0, position=0.3, n_modes: int = 24, fmax: float = 16000.0, mass: float = 1.0,
                 ext: float | None = None, scatter: float | None = None, law: str = "friction",
                 excitation: str = "struck", seed: int = 0):
        if boundary not in BOUNDARIES:
            raise ValueError(f"unknown boundary {boundary!r}; choose from {BOUNDARIES}")
        f1 = float(np.clip(f1, 5.0, 20000.0))
        ratio, desc = _shape_table(shape, boundary, round(float(aspect), 6), int(n_modes),
                                   round(max(float(fmax) / f1, 1.0), 6))
        self.shape, self.boundary, self.aspect, self.f1 = shape, boundary, float(aspect), f1
        self.material = get_material(material)
        freqs = f1 * ratio
        decays = material_damping(freqs, self.material, law=law, ext=ext, scatter=scatter, seed=seed)
        super().__init__(freqs, decays, np.zeros_like(freqs), mass=mass,
                         aw0=1.0 / ratio if excitation == "plucked" else None, name=shape)
        self._desc = dict(desc)
        self.position = position
        self.phi = self.row(position)

    @classmethod
    def from_size(cls, shape: str, size: float, material="steel", *, thickness: float | None = None,
                  wave_speed: float = 120.0, **kw) -> "ShapeBody":
        """Pitch and mass from physical size (m) and material (see `fundamental`, `body_mass`)."""
        f1 = fundamental(shape, size, material, boundary=kw.get("boundary", "free"), aspect=kw.get("aspect", 1.0),
                         thickness=thickness, wave_speed=wave_speed)
        mass = body_mass(shape, size, material, aspect=kw.get("aspect", 1.0), thickness=thickness)
        return cls(shape, f1, material=material, mass=mass, **kw)

    def residual_compliance(self, n_total: int = 400) -> float:
        """Static compliance (m/N) at the contact point of the modes the bank drops, sum phi^2 / (m w^2) over
        the next modes up to ``n_total``: below them the dropped modes act as a spring in series with the bank."""
        full = ShapeBody(self.shape, self.f1, material=self.material, boundary=self.boundary, aspect=self.aspect,
                         position=self.position, n_modes=n_total, fmax=self.f1 * 1e6, mass=self.mass)
        top = float(np.max(self.freqs)) * (1.0 + 1e-9)
        up = full.freqs > top
        return float(np.sum(full.phi[up] ** 2 / (self.mass * (TWO_PI * full.freqs[up]) ** 2)))

    def monopole_share(self) -> np.ndarray:
        """Per mode |int phi dS| / int |phi| dS: 1 for a mode that only breathes, 0 when its parts cancel."""
        if self.shape == "l_membrane":
            sh = _l_modes()[1][self._desc["k"].astype(int)]
            return np.abs(sh.sum(axis=(1, 2))) / (np.abs(sh).sum(axis=(1, 2)) + 1e-30)
        u = (np.arange(64) + 0.5) / 64
        if self.shape in ("string", "bar"):
            P, wt = np.array([self.row(a) for a in u]), np.ones(u.size)
        elif self.shape.startswith("circ"):
            th = (np.arange(32) + 0.5) / 32 * TWO_PI
            P = np.array([self.row((r, t)) for r in u for t in th])
            wt = np.repeat(u, th.size)
        else:
            P = np.array([self.row((a, b)) for a in u for b in u])
            wt = np.ones(P.shape[0])
        return np.abs(wt @ P) / (wt @ np.abs(P) + 1e-30)

    def row(self, position=None) -> np.ndarray:
        """Mode shapes (unit mean square over the body) sampled at ``position``."""
        p = self.position if position is None else position
        a, b = (float(p[0]), float(p[1])) if np.ndim(p) else (float(p), None)
        d = self._desc
        if self.shape == "string":
            return math.sqrt(2.0) * np.sin(np.pi * d["n"] * a)
        if self.shape == "bar":
            return _bar_shape(d["b"], float(np.clip(a, 0.0, 1.0)), self.boundary)
        if self.shape in ("rect_membrane", "rect_plate"):
            return 2.0 * np.sin(np.pi * d["m"] * a) * np.sin(np.pi * d["n"] * (a if b is None else b))
        if self.shape == "l_membrane":                       # (x, y) over the outer square; the cut corner is x, y > 0.5
            from scipy.interpolate import RegularGridInterpolator
            sh = _l_modes()[1]
            g = np.linspace(0.0, 1.0, sh.shape[1])
            pt = (float(np.clip(a, 0, 1)), float(np.clip(a if b is None else b, 0, 1)))
            return np.array([RegularGridInterpolator((g, g), sh[int(i)])(np.array([pt]))[0] for i in d["k"]])
        r = float(np.clip(a, 0.0, 1.0))
        ang = np.cos(d["m"] * (0.0 if b is None else b))
        if self.shape == "circ_membrane":
            rad = special.jv(d["m"], d["j"] * r)
        elif self.shape == "circ_plate_free":
            rad = np.array([_free_disc_radial(m, k, r) for m, k in zip(d["m"], d["k"])])
        else:
            rad = _plate_radial(d["m"], d["k"], r)
        return rad * ang / np.sqrt(d["norm"])


def _concat(bodies, bright):
    """Several bodies touching at one contact point as one bank. The force acts with opposite sign
    on the two sides of a contact, but with y = -x for the far side both obey the same equations."""
    f = np.concatenate([b.freqs for b in bodies])
    d = np.concatenate([b.decays for b in bodies])
    invm = np.concatenate([np.full(len(b.freqs), 1.0 / b.mass) for b in bodies])
    aw = np.concatenate([b.audio_weights(bright) * g for b, g in zip(bodies, _levels(bodies, bright))])
    return f, d, invm, aw


def _levels(bodies, bright):
    """Per-body output gain so each body's unit-impulse response has the same peak order."""
    out = []
    for b in bodies:
        a = b.phi * b.audio_weights(bright) / (TWO_PI * b.freqs * b.mass)
        out.append(1.0 / (math.sqrt(float(np.sum(a * a))) + 1e-30))
    m = max(out)
    return [o / m for o in out]


# ==========================================================================================
# State-space modal bank (bilinear transform; Sounding Object §8.2.3 eq. 8.5-8.7; g04 §4.3)
# ==========================================================================================

def _prewarp(freqs, decays, fs):
    """Analog (w, d) that land the bilinear-transformed mode on f and d: w = 2 fs tan(pi f/fs)
    (frequency warping) and d (1 + tan^2), since the transform scales the pole radius by cos^2."""
    t = np.tan(np.pi * np.asarray(freqs, float) / fs)
    return 2.0 * fs * t, np.asarray(decays, float) * (1.0 + t * t)


@numba.njit(cache=True)
def _bank_kernel(force, fs, w, d, phi, invm, aw):
    """x(n) = A x(n-1) + b (y(n) + y(n-1)) per mode, state [x, x'] (g04 §4.3):
    D = fs^2 + d fs + w^2/4, A = [[D - w^2/2, fs], [-fs w^2, 2 fs^2 - D]]/D, b = c [1, 2 fs]/(4 D)."""
    n = force.shape[0]
    audio = np.zeros(n)
    u = np.zeros(n)
    ud = np.zeros(n)
    for j in range(w.shape[0]):
        ww = w[j] * w[j]
        D = fs * fs + d[j] * fs + 0.25 * ww
        a11 = (D - 0.5 * ww) / D
        a12 = fs / D
        a21 = -fs * ww / D
        a22 = (2.0 * fs * fs - D) / D
        c = phi[j] * invm[j]
        bx = 0.25 * c / D
        bv = 0.5 * fs * c / D
        x = 0.0
        v = 0.0
        fp = 0.0
        for i in range(n):
            s = force[i] + fp
            fp = force[i]
            nx = a11 * x + a12 * v + bx * s
            v = a21 * x + a22 * v + bv * s
            x = nx
            audio[i] += aw[j] * x
            u[i] += phi[j] * x
            ud[i] += phi[j] * v
    return audio, u, ud


def modal_bank(force, body: ModalObject, sr: int = DEFAULT_SR, bright: float = 0.5, residual: bool = False):
    """Linear state-space bank: force (N) at the body's contact point -> ``(audio, displacement,
    velocity)``; the last two are the motion of the contact point itself (what an interaction
    needs to be two-way coupled). ``residual``: add the quasi-static motion of the truncated modes
    (`ShapeBody.residual_compliance` times the force) to the contact-point motion."""
    force = np.ascontiguousarray(force, dtype=np.float64)
    ok = body.freqs < 0.45 * sr
    w, d = _prewarp(body.freqs[ok], body.decays[ok], float(sr))
    audio, u, ud = _bank_kernel(force, float(sr), w, d, np.ascontiguousarray(body.phi[ok]),
                                np.full(int(ok.sum()), 1.0 / body.mass), np.ascontiguousarray(body.audio_weights(bright)[ok]))
    if residual and isinstance(body, ShapeBody):
        c = body.residual_compliance()
        u = u + c * force
        ud = ud + c * np.gradient(force) * float(sr)
    return audio, u, ud


def gammatone_bank(force, body: ModalObject, sr: int = DEFAULT_SR, bright: float = 0.5) -> np.ndarray:
    """Audio of `modal_bank` with every mode's impulse response a gammatone of order 2, t exp(-d t) sin(w t),
    instead of exp(-d t) sin(w t): z-transform r sin(th) z^-1 (1 - r^2 z^-2)/(1 - 2 r cos(th) z^-1 + r^2 z^-2)^2.
    Scaled by d e so its envelope peaks (at t = 1/d) at the one-pole mode's amplitude phi/(m w)."""
    force = np.asarray(force, dtype=np.float64)
    ok = np.flatnonzero(body.freqs < 0.45 * sr)
    aw = body.audio_weights(bright)
    out = np.zeros_like(force)
    for j in ok:
        f, d = float(body.freqs[j]), float(body.decays[j])
        r, th = math.exp(-d / sr), TWO_PI * f / sr
        a1 = np.array([1.0, -2.0 * r * math.cos(th), r * r])
        b = np.array([0.0, r * math.sin(th), 0.0, -r ** 3 * math.sin(th)])
        g = body.phi[j] / (body.mass * TWO_PI * f) * d * math.e / (sr * sr)
        out += aw[j] * g * lfilter(b, np.convolve(a1, a1), force)
    return out


def moving_bank(force, body: ModalObject, sr: int = DEFAULT_SR, path=None, bright: float = 0.5, variation: float = 0.0,
                seed: int = 0, block: int = 512) -> np.ndarray:
    """Audio of `modal_bank` with a contact that moves (``path``: per-sample position 0..1, the body's own second
    coordinate kept) and/or mode levels that wander along it: every ``block`` samples each mode's input gain is its
    shape at the contact (`ShapeBody.row`) times sqrt(max(1 + e, 0)), e Gaussian of std ``variation`` in power;
    gains are interpolated between blocks, the modes keep ringing across."""
    force = np.ascontiguousarray(force, dtype=np.float64)
    n = len(force)
    ok = np.flatnonzero(body.freqs < 0.45 * sr)
    w, d = _prewarp(body.freqs[ok], body.decays[ok], float(sr))
    aw = body.audio_weights(bright)[ok]
    centres = np.arange(0, n + block, block)
    rows = np.empty((len(centres), len(ok)))
    p0 = getattr(body, "position", None)
    for i, c in enumerate(centres):
        if path is not None and isinstance(body, ShapeBody):
            x = float(path[min(c, n - 1)])
            pos = (x, float(p0[1])) if np.ndim(p0) else x
            rows[i] = body.row(pos)[ok]
        else:
            rows[i] = body.phi[ok]
    if variation > 0:
        e = np.random.default_rng(int(seed) + 4242).normal(0.0, variation, rows.shape)
        rows = rows * np.sqrt(np.maximum(1.0 + e, 0.0))
    t = np.arange(n)
    out = np.zeros(n)
    for j in range(len(ok)):
        g = np.interp(t, centres, rows[:, j])
        out += _bank_kernel(force * g, float(sr), w[j:j + 1], d[j:j + 1], np.ones(1), np.array([1.0 / body.mass]),
                            aw[j:j + 1].copy())[0]
    return out


def resonant_prefilter(force, body: ModalObject, sr: int = DEFAULT_SR, depth: float = 1.0) -> np.ndarray:
    """An imposed contact force reinforced near the body's modes, as the two-way contact does by itself:
    one unit-peak resonator per mode (its f and decay d, |H| ~ 1/sqrt((w - w_k)^2 + d^2) near w_k), summed,
    brought to the force's RMS and mixed in by ``depth`` (0 = the force unchanged)."""
    force = np.asarray(force, float)
    depth = float(np.clip(depth, 0.0, 1.0))
    if depth <= 0.0:
        return force
    ok = body.freqs < 0.45 * sr
    acc = np.zeros_like(force)
    for f, d in zip(body.freqs[ok], body.decays[ok]):
        r = math.exp(-float(d) / sr)
        g = 0.5 * (1.0 - r * r)
        acc += lfilter([g, 0.0, -g], [1.0, -2.0 * r * math.cos(TWO_PI * float(f) / sr), r * r], force)
    acc *= math.sqrt(float(np.mean(force * force)) / (float(np.mean(acc * acc)) + 1e-30))
    return (1.0 - depth) * force + depth * acc


# ==========================================================================================
# Hunt-Crossley impact (Sounding Object §8.3.1 eq. 8.11-8.14, 8.27, appendix 8.A; g04 §4.4-4.5)
# ==========================================================================================

def hc_rebound(mu: float, v_in: float) -> float:
    """Velocity (< 0, i.e. separating) at which a mass leaves a rigid wall it hit at ``v_in``:
    the root of ``-mu (v - v_in) + ln((1 + mu v)/(1 + mu v_in)) = 0`` (phase-plane eq. 8.13).
    Independent of m, k and alpha; ``|v_out| -> 1/mu`` for hard hits."""
    mu = max(float(mu), 1e-4 / max(v_in, 1e-9))
    ui = 1.0 + mu * v_in
    u = brentq(lambda q: -(q - ui) + math.log(q / ui), 1e-300, 1.0 - 1e-15)
    return (u - 1.0) / mu


HC_SERIES = (1.0, 1.0, 2.0 / 3.0, 2.0 / 9.0, 14.0 / 135.0)


def hc_rebound_closed(mu: float, v_in: float) -> float:
    """Closed form of `hc_rebound`: v_out = -(1/mu) [1 - (sum_j b_j (mu v_in)^j) exp(-2 mu v_in)], b = 1, 1, 2/3, 2/9,
    14/135 (within about 2 % of the exact root)."""
    x = float(mu) * float(v_in)
    return -(1.0 - sum(b * x ** j for j, b in enumerate(HC_SERIES)) * math.exp(-2.0 * x)) / float(mu)


def directional_gains(axis_gains, direction) -> np.ndarray:
    """Mode gains for a unit force along `direction`: k1 a1 + k2 a2 + k3 a3, with a_i the gains of a unit force along
    axis i (shape (3, modes)) and k the unit direction."""
    k = np.asarray(direction, float)
    k = k / (np.linalg.norm(k) + 1e-30)
    return k @ np.asarray(axis_gains, float)


def hc_contact_time(mass: float, k: float, alpha: float, mu: float, v_in: float) -> float:
    """Contact time t0 of a mass on a rigid wall (eq. 8.27; g04 §4.4):

        t0 = (m/k)^(1/(a+1)) (mu^2/(a+1))^(a/(a+1)) *
             integral_{v_out}^{v_in} dv / ((1 + mu v) [-mu (v - v_in) + ln((1 + mu v)/(1 + mu v_in))]^(a/(a+1)))

    It depends only on v_in, mu and m/k; for mu -> 0 it is the Hertz law
    t0 ~ (m/k)^(1/(a+1)) v_in^(-(a-1)/(a+1)).
    """
    mu = max(float(mu), 1e-4 / max(v_in, 1e-9))
    p = alpha / (alpha + 1.0)
    ui = 1.0 + mu * v_in
    uo = 1.0 + mu * hc_rebound(mu, v_in)
    s = (np.arange(20000) + 0.5) / 20000.0                   # midpoint rule on a warped axis:
    a = 8.0                                                  # removes the end-point singularities
    den = s ** a + (1.0 - s) ** a
    lo, hi = (ui - uo) * s ** a / den, -(ui - uo) * (1.0 - s) ** a / den     # u - u_out, u - u_in
    u = uo + lo
    du = (ui - uo) * a * (s * (1.0 - s)) ** (a - 1.0) / den ** 2
    # B(u_out) = B(u_in) = 0, so measure from the nearer end (no cancellation)
    B = np.maximum(np.where(s < 0.5, -lo + np.log1p(lo / uo), -hi + np.log1p(hi / ui)), 1e-300)
    integral = float(np.mean(du / (mu * u * B ** p)))
    return (mass / k) ** (1.0 / (alpha + 1.0)) * (mu * mu / (alpha + 1.0)) ** p * integral


@numba.njit(cache=True)
def _impact_kernel(n, fs, w, d, phi, invm, aw, k, lam, alpha, ev_i, ev_v, ev_p, ev_m, off, load, y0, v0, m0, act0):
    """Striker mass + modal bank coupled by f = k x^a + lam x^a v (x > 0), K method (g04 §4.5).

    Geometry: striker height y (up +), contact-point displacement u = sum phi x, surface offset
    ``off`` (rolling), compression x = off + u - y; +f on the striker, -f on the bank; ``load`` is
    a downward force on the striker (its weight). Event e (re)launches the striker at sample
    ev_i[e] touching the surface with approach speed ev_v[e], mass ev_m[e], mode-shape row ev_p[e].
    The bank is never reset: a hit on a ringing body differs from a hit on a silent one.
    """
    N = w.shape[0]
    P = phi.shape[0]
    a11 = np.empty(N)
    a12 = np.empty(N)
    a21 = np.empty(N)
    a22 = np.empty(N)
    bx = np.empty(N)
    bv = np.empty(N)
    for j in range(N):
        ww = w[j] * w[j]
        D = fs * fs + d[j] * fs + 0.25 * ww
        a11[j] = (D - 0.5 * ww) / D
        a12[j] = fs / D
        a21[j] = -fs * ww / D
        a22[j] = (2.0 * fs * fs - D) / D
        bx[j] = 0.25 / D
        bv[j] = 0.5 * fs / D
    S1 = np.zeros(P)
    S2 = np.zeros(P)
    for p in range(P):
        for j in range(N):
            c = phi[p, j] * phi[p, j] * invm[j]
            S1[p] += c * bx[j]
            S2[p] += c * bv[j]
    x = np.zeros(N)
    xd = np.zeros(N)
    audio = np.zeros(n)
    force = np.zeros(n)
    hy = np.zeros(n)
    hv = np.zeros(n)
    du = np.zeros(n)
    active = act0
    y = y0
    vy = v0
    mh = m0
    row = 0
    hbx = 0.25 / (mh * fs * fs)
    hbv = 0.5 / (mh * fs)
    f_prev = 0.0
    ld_prev = load[0] if n > 0 else 0.0
    off_prev = off[0] if n > 0 else 0.0
    e = 0
    ne = ev_i.shape[0]
    umax = 0.0
    for i in range(n):
        while e < ne and ev_i[e] <= i:
            row = ev_p[e]
            mh = ev_m[e]
            hbx = 0.25 / (mh * fs * fs)
            hbv = 0.5 / (mh * fs)
            u0 = 0.0
            ud0 = 0.0
            for j in range(N):
                u0 += phi[row, j] * x[j]
                ud0 += phi[row, j] * xd[j]
            y = u0 + off[i]
            off_prev = off[i]
            vy = ud0 - ev_v[e]
            active = True
            f_prev = 0.0
            ld_prev = load[i]
            e += 1
        u = 0.0
        ud = 0.0
        for j in range(N):
            s = -f_prev * phi[row, j] * invm[j]
            nx = a11[j] * x[j] + a12[j] * xd[j] + bx[j] * s
            xd[j] = a21[j] * x[j] + a22[j] * xd[j] + bv[j] * s
            x[j] = nx
            u += phi[row, j] * nx
            ud += phi[row, j] * xd[j]
        f = 0.0
        if active:
            sh = f_prev - load[i] - ld_prev
            ny = y + vy / fs + hbx * sh
            nvy = vy + hbv * sh
            xt = off[i] + u - ny
            vt = (off[i] - off_prev) * fs + ud - nvy
            if xt > 0.0:
                K1 = -(hbx + S1[row])
                K2 = -(hbv + S2[row])
                hi = k * xt ** alpha + lam * xt ** alpha * vt
                if hi > 0.0:
                    lo = 0.0                                  # g(h) = f(xt + K1 h, vt + K2 h) - h is
                    h = min(f_prev, hi) if f_prev > 0.0 else hi   # decreasing: root in (0, hi]
                    tol = 1e-12 * hi + 1e-300
                    for _ in range(80):
                        xc = xt + K1 * h
                        vc = vt + K2 * h
                        if xc > 0.0:
                            pw = xc ** alpha
                            g = k * pw + lam * pw * vc - h
                            dg = alpha * xc ** (alpha - 1.0) * (k + lam * vc) * K1 + lam * pw * K2 - 1.0
                        else:
                            g = -h
                            dg = -1.0
                        if g > 0.0:
                            lo = h
                        else:
                            hi = h
                        hn = h - g / dg
                        if not (hn > lo and hn < hi):
                            hn = 0.5 * (lo + hi)
                        if abs(hn - h) < tol or hi - lo < tol:
                            h = hn
                            break
                        h = hn
                    f = max(h, 0.0)
            y = ny + hbx * f
            vy = nvy + hbv * f
            if f > 0.0:
                u = 0.0
                ud = 0.0
                for j in range(N):
                    c = phi[row, j] * invm[j] * f
                    x[j] -= bx[j] * c
                    xd[j] -= bv[j] * c
                    u += phi[row, j] * x[j]
                    ud += phi[row, j] * xd[j]
            ld_prev = load[i]
            if abs(u) > umax:
                umax = abs(u)
            if f == 0.0 and load[i] == 0.0 and vy > ud and (y - u - off[i]) > 1e-6 + 20.0 * umax:
                active = False                                # flown away for good: park the striker
        off_prev = off[i]
        f_prev = f
        a = 0.0
        for j in range(N):
            a += aw[j] * x[j]
        audio[i] = a
        force[i] = f
        hy[i] = y
        hv[i] = vy
        du[i] = u
    return audio, force, hy, hv, du


@dataclass
class ImpactResult:
    audio: np.ndarray        # heard signal at ``sr``
    force: np.ndarray        # contact force (N) at the internal rate ``fs``
    pos: np.ndarray          # striker height (m), internal rate
    vel: np.ndarray          # striker velocity (m/s), internal rate
    disp: np.ndarray         # contact-point displacement of the body (m), internal rate
    fs: float
    sr: int

    def contacts(self) -> list[tuple[int, int]]:
        """(start, end) sample ranges (internal rate) of every uninterrupted contact."""
        on = np.concatenate([[False], self.force > 0.0, [False]])
        edge = np.diff(on.astype(np.int8))
        return list(zip(np.nonzero(edge == 1)[0].tolist(), np.nonzero(edge == -1)[0].tolist()))

    @property
    def contact_time(self) -> float:
        c = self.contacts()
        return (c[0][1] - c[0][0]) / self.fs if c else 0.0

    @property
    def restitution(self) -> float:
        """|v_out| / |v_in| of the striker across the first contact."""
        c = self.contacts()
        if not c:
            return 0.0
        return float(abs(self.vel[min(c[0][1], len(self.vel) - 1)]) / (abs(self.vel[max(c[0][0] - 1, 0)]) + 1e-30))


class ImpactInteractor:
    """Striker mass <-> modal body through a Hunt-Crossley contact, integrated per sample.

    ``f(x, v) = k x^alpha (1 + mu v)`` for compression x > 0 (eq. 8.11). ``k`` (N/m^alpha) is the
    hardness: 5e6 rubber mallet ... 1e10 steel mallet; ``alpha`` the contact geometry (1.5 two
    spheres, 2.8 felt-like / sticky); ``mu`` (s/m) the dissipation, restitution
    ``e ~ 1 - (2/3) mu v`` for soft hits and ``e -> 1/(mu v)`` for hard ones. ``body`` is a
    `ModalObject`, a list of them (both sides of the contact) or ``None`` (rigid wall).
    What it gives that a stored force pulse cannot (g04 §4.4): velocity-dependent brightness
    (shorter contact when harder / faster), micro-bounces on hard contacts, a different force every
    time a ringing body is hit again, heavy-striker damping of the low mode.
    """

    def __init__(self, body=None, *, mass: float = 0.01, k: float = 1.495e9, alpha: float = 1.5,
                 mu: float = 0.1, gravity: float = 0.0, bright: float = 0.5):
        self.bodies = [] if body is None else (list(body) if isinstance(body, (list, tuple)) else [body])
        self.mass = float(max(mass, 1e-7))
        self.k = float(np.clip(k, 1.0, 1e13))
        self.alpha = float(np.clip(alpha, 1.0, 4.0))
        self.mu = float(np.clip(mu, 0.0, 50.0))
        self.gravity = float(gravity)
        self.bright = bright

    def contact_time(self, velocity: float) -> float:
        """Rigid-wall contact time of this striker (eq. 8.27)."""
        return hc_contact_time(self.mass, self.k, self.alpha, self.mu, velocity)

    def run(self, dur: float, sr: int = DEFAULT_SR, *, events=None, rows=None, y0: float = 0.0, v0: float = 0.0,
            active: bool = False, offset=None, load=None, oversample: int | None = None) -> ImpactResult:
        """``events``: (times s, approach speeds m/s[, row indices[, striker masses kg]]).
        ``rows``: (P, n_modes) mode-shape rows the events index (default: the bodies' own strike
        points). ``offset`` / ``load``: surface offset curve (m) and downward force on the striker
        (N) at the internal rate ``sr * oversample``."""
        t, v, pr, pm = (list(events) + [None, None])[:4] if events is not None else ([], [], None, None)
        t = np.atleast_1d(np.asarray(t, float))
        v = np.atleast_1d(np.asarray(v, float))
        pm = np.full(len(t), self.mass) if pm is None else np.maximum(np.asarray(pm, float), 1e-7)
        pr = np.zeros(len(t), np.int64) if pr is None else np.asarray(pr, np.int64)
        if oversample is None:        # keep the shortest contact at least ~10 internal samples long
            vmax = max(float(v.max()) if len(v) else 0.0, abs(v0), math.sqrt(2 * self.gravity * max(y0, 0.0)), 0.05)
            tc = hc_contact_time(float(pm.min()) if len(pm) else self.mass, self.k, self.alpha, self.mu, vmax)
            oversample = int(min(16, 2 ** max(0, math.ceil(math.log2(max(10.0 / (tc * sr), 1.0))))))
        fs = float(sr * oversample)
        n = samples(dur, sr) * oversample
        if self.bodies:
            f, d, invm, aw = _concat(self.bodies, self.bright)
            phi = np.concatenate([b.phi for b in self.bodies])[None, :] if rows is None else np.atleast_2d(rows)
        else:
            f = d = invm = aw = np.zeros(0)
            phi = np.zeros((1, 0))
        ok = f < 0.45 * sr
        w, da = _prewarp(f[ok], d[ok], fs)
        off = np.zeros(n) if offset is None else np.ascontiguousarray(offset, dtype=np.float64)
        ld = np.full(n, self.mass * self.gravity) if load is None else np.ascontiguousarray(load, dtype=np.float64)
        out = _impact_kernel(n, fs, w, da, np.ascontiguousarray(phi[:, ok], dtype=np.float64),
                             np.ascontiguousarray(invm[ok]), np.ascontiguousarray(aw[ok]), self.k,
                             self.mu * self.k, self.alpha, np.round(t * fs).astype(np.int64), v,
                             np.clip(pr, 0, phi.shape[0] - 1), pm, off, ld, float(y0), float(v0), self.mass, bool(active))
        audio = out[0] if oversample == 1 else resample_poly(out[0], 1, oversample)[: n // oversample]
        return ImpactResult(audio, out[1], out[2], out[3], out[4], fs, sr)

    def strike(self, velocity: float = 1.0, dur: float = 1.0, sr: int = DEFAULT_SR, **kw) -> ImpactResult:
        """One hit at t = 0 with approach speed ``velocity`` (m/s)."""
        return self.run(dur, sr, events=([0.0], [velocity]), **kw)

    def drop(self, height: float = 0.1, dur: float = 1.0, sr: int = DEFAULT_SR, **kw) -> ImpactResult:
        """Release the striker from ``height`` (m) under gravity: it bounces by itself and ends in
        permanent contact (Sounding Object §9.2, Fig. 9.7)."""
        if self.gravity <= 0:
            self.gravity = G
        return self.run(dur, sr, y0=height, v0=0.0, active=True, **kw)


def _oversample_for(tc: float, sr: int) -> int:
    return int(min(16, 2 ** max(0, math.ceil(math.log2(max(8.0 / (tc * sr), 1.0))))))


# ==========================================================================================
# Surface texture and temporal patterns
# ==========================================================================================

def fractal_noise(n: int, beta: float = 1.0, sr: float = DEFAULT_SR, *, f_low: float = 20.0,
                  poles_per_decade: float = 4.0, seed: int = 0) -> np.ndarray:
    """1/f^beta noise (unit RMS): the fractal surface profile of Sounding Object §8.3.3
    (eq. 8.22-8.26; g04 §4.7, g05 §9.5). beta = 2H + 1: 0 white (very rough), 1 pink, 2 Brownian
    (smooth). White noise through a cascade of first-order pole/zero sections
    ``f_p,i = f_p,i-1 10^(1/h)``, ``f_0,i = f_p,i 10^(beta/(2h))``,
    ``H_i(z) = (1 - e^(-2 pi f_0 T) z^-1)/(1 - e^(-2 pi f_p T) z^-1)`` (minus signs per eq.
    8.24-8.25; the printed 8.26 shows "+"). Average slope -10 beta dB/decade above ``f_low``."""
    beta = float(np.clip(beta, 0.0, 3.0))
    h = float(np.clip(poles_per_decade, 1.0, 12.0))
    f_low = float(np.clip(f_low, 1e-9 * sr, 0.2 * sr))
    x = np.random.default_rng(seed).standard_normal(int(n))
    if beta > 0:
        for i in range(int(math.floor(h * math.log10(0.45 * sr / f_low))) + 1):
            fp = f_low * 10 ** (i / h)
            f0 = fp * 10 ** (beta / (2 * h))
            if f0 >= 0.5 * sr:
                break
            x = lfilter([1.0, -math.exp(-TWO_PI * f0 / sr)], [1.0, -math.exp(-TWO_PI * fp / sr)], x)
    x = x - x.mean()
    return x / (x.std() + 1e-30)


RESTITUTION = {"steel": 0.75, "metal": 0.75, "aluminium": 0.75, "copper": 0.75, "glass": 0.8, "wood": 0.5, "stone": 0.45,
               "plastic": 0.55, "plastic2": 0.55, "polycarbonate": 0.55, "polystyrene": 0.55, "acrylic": 0.55, "rubber": 0.85,
               "ceramic": 0.65, "ice": 0.4, "mdf": 0.5}


def material_restitution(name) -> float:
    """Typical coefficient of restitution of a ball of this material on a hard floor (0.6 if unknown)."""
    key = str(name).split(":")[0]
    return RESTITUTION.get(_ALIAS.get(key, key), 0.6)


def bounce_schedule(first_interval: float = 0.3, velocity: float = 1.0, factor: float = 0.74,
                    vel_factor: float | None = None, *, dev_t: float = 0.0, dev_v: float = 0.0,
                    regularize: float = 0.0, t_min: float = 0.004, n_max: int = 200, seed: int = 0, e_of_v=None):
    """The "bouncer" (Sounding Object §9.2, eq. 9.3-9.5; g05 §1): ``(times, velocities)``.

    Maximal intervals and velocities follow one geometric law ``t_n = factor^n t_0``,
    ``v_n = vel_factor^n v_0`` (for a sphere both factors are the restitution sqrt(C); 0.70-0.76
    for a steel ball on wood, Table 13.5). ``dev_t`` / ``dev_v`` in 0..1 are random deviations
    *below* those maxima (non-spherical objects are irregular but bounded by the spherical
    envelope). ``regularize`` > 0 makes the deviations die out as the bounces shrink (the falling
    coin: random start, regular end). The series stops when the interval falls under ``t_min``
    (the "buzz" guard / terminating bang). ``e_of_v(v)``: a speed-dependent restitution that caps both factors at each bounce.
    """
    rng = np.random.default_rng(seed)
    factor = float(np.clip(factor, 0.05, 0.98))
    velf = factor if vel_factor is None else float(np.clip(vel_factor, 0.05, 1.0))
    t, T, V = 0.0, float(first_interval), float(velocity)
    times, vels = [], []
    for _ in range(int(n_max)):
        k = (T / first_interval) ** regularize if regularize > 0 else 1.0
        times.append(t)
        vels.append(V * (1.0 - dev_v * k * rng.random()))
        if T < t_min:
            break
        t += T * (1.0 - dev_t * k * rng.random())
        T *= factor if e_of_v is None else min(factor, e_of_v(V))
        V *= velf if e_of_v is None else min(velf, e_of_v(V))
    return np.array(times), np.array(vels)


def break_schedule(dur: float = 1.2, *, first: float = 0.004, growth: float = 1.25, dev: float = 0.8,
                   velocity: float = 1.0, vel_factor: float = 0.85, trains: int = 1, seed: int = 0):
    """The "dropper" (Sounding Object §9.3; g05 §2): ``(times, velocities, train index)``, sorted.

    A bouncer with time factor > 1 and high randomness: fragments do not bounce, they "nod" -
    intervals *grow*, so the event density starts massive and falls quickly. Several ``trains``
    with different growth and velocity factors = several damped pulse trains with different
    damping, the cue for breaking as opposed to bouncing (Warren & Verbrugge 1984; g04 §1.1).
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in range(max(1, int(trains))):
        g = growth * (1.0 + 0.25 * (rng.random() - 0.5)) if trains > 1 else growth
        vf = float(np.clip(vel_factor * (1.0 + 0.2 * (rng.random() - 0.5)), 0.3, 0.98)) if trains > 1 else vel_factor
        t = rng.random() * 0.03 if trains > 1 else 0.0
        T, V = first * (0.5 + rng.random() if trains > 1 else 1.0), velocity * (0.5 + 0.5 * rng.random() if trains > 1 else 1.0)
        out.append((t, V, k))
        while V > 1e-3 * velocity and len(out) < 20000:
            t += T * (1.0 - dev * rng.random())
            V *= vf
            if t >= dur:
                break
            out.append((t, V * (1.0 - dev * rng.random()), k))
            T *= max(g, 1.0001)
    out.sort()
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2].astype(np.int64)


def crumple_energy_floor(gamma: float, e_max: float = 1.0) -> float:
    """Lowest event energy m for which P(E) = E^gamma integrates to 1 on [m, e_max]
    (eq. 9.12-9.13; g05 §4.1): m = (M^(g+1) - (g+1))^(1/(g+1))."""
    g1 = gamma + 1.0
    return (e_max ** g1 - g1) ** (1.0 / g1)


def crumple_events(e_tot: float = 0.03, gamma: float = -1.3, rate: float = 20.0, *, e_unit: float = 7.5e-4,
                   e_min: float | None = None, e_max: float = 1.0, d0: float = 1.0, seed: int = 0,
                   n_max: int = 200000) -> dict:
    """Crumpling event process (Sounding Object §9.5, eq. 9.8-9.15; g05 §4).

    Event energies follow the power law ``P(E) ~ E^gamma`` on ``[e_min, e_max]`` (gamma -1.3...-1.6
    measured for paper; -1.15 soft ... -1.5 hard crushing of a can), gaps are Poisson with
    density ``rate`` (1/s), and the process stops when the budget ``e_tot`` (0.001-0.1 = size of
    the object) is spent. Each event buckles one facet in two: a random point splits a segment of
    length ``d0``; ``left`` / ``right`` are the distances to the nearest existing marks and share
    the event energy in proportion (eq. 9.14). ``cut`` is the closing low-pass of the crushed can,
    1400 -> 500 Hz as the energy is spent (eq. 9.15).
    Returns arrays ``t, energy, left, right, cut`` (one entry per event).
    """
    rng = np.random.default_rng(seed)
    gamma = float(np.clip(gamma, -3.0, -1.01))
    g1 = gamma + 1.0
    m = crumple_energy_floor(gamma, e_max) if e_min is None else float(e_min)
    marks = [0.0, float(d0)]
    spent, t = 0.0, 0.0
    T, E, L, R, C = [], [], [], [], []
    # e_unit maps the normalised energies of (9.12) onto the e_tot budget (the book
    # gives the ranges of both but not their ratio); 7.5e-4 makes e_tot = 0.03 about 60 events.
    while spent < e_tot and len(T) < n_max:
        t += rng.exponential(1.0 / rate)
        e = (m ** g1 + rng.random() * (e_max ** g1 - m ** g1)) ** (1.0 / g1)       # inverse CDF
        spent += e * e_unit
        p = rng.uniform(0.0, d0)
        j = int(np.searchsorted(marks, p))
        left, right = p - marks[j - 1], marks[j] - p
        marks.insert(j, p)
        T.append(t)
        E.append(e)
        L.append(left)
        R.append(right)
        C.append(500.0 + max(e_tot - spent, 0.0) / e_tot * 900.0)
    return {"t": np.array(T), "energy": np.array(E), "left": np.array(L), "right": np.array(R), "cut": np.array(C)}


def micro_impact_force(z: np.ndarray, speed, mass: float, depth: float = 1.0, decay: float = 1.0) -> tuple[np.ndarray, int]:
    """Energy-counter rolling force, for cross-checking `roll`: the load is -m g; where the ground rises under a
    ball that is not in flight it gets a bump dz depth (m v^2 / 2) / sqrt(decay) and flies for 2 bump, the flight
    store losing m g each sample. Returns (force, number of micro-impacts)."""
    z = np.asarray(z, float)
    v = np.broadcast_to(np.asarray(speed, float), z.shape)
    out = np.full(z.shape, -mass * G)
    flight, hits = 0.0, 0
    dec = math.sqrt(max(float(decay), 1e-12))
    for i in range(1, z.size):
        dz = z[i] - z[i - 1]
        if dz > 0 and flight <= 0.0:
            bump = dz * depth * 0.5 * mass * v[i] * v[i] / dec
            flight = 2.0 * bump
            out[i] += bump
            hits += 1
        else:
            flight = max(0.0, flight - mass * G)
    return out, hits


def rolling_offset(profile: np.ndarray, radius: float, dx: float) -> np.ndarray:
    """The "rolling filter" (Sounding Object eq. 9.6-9.7; g05 §3.1): height of the lowest point of
    a ball of ``radius`` rolling over ``profile`` (m, sampled every ``dx`` m),
    ``offset(x) = max_q [ s(q) + sqrt(r^2 - (q - x)^2) ] - r`` - a grey-scale dilation by a
    circular arc. The ball touches the surface only at isolated points, so the curve is smooth
    arcs joined at edges; a bigger ball bridges more and gives a smoother, lower curve."""
    s = np.asarray(profile, float)
    r = max(float(radius), dx)
    half = int(min(math.sqrt(2.0 * r * (np.ptp(s) + 1e-12)) / dx + 2, r / dx, 4000, len(s) // 2))
    q = np.arange(-half, half + 1) * dx
    cap = np.sqrt(np.maximum(r * r - q * q, 0.0)) - r
    return grey_dilation(s, structure=cap, mode="wrap")


def rolling_surface(length: float, dx: float = 1e-4, roughness: float = 0.5, *, joints: float = 0.0,
                    seed: int = 0) -> np.ndarray:
    """Surface height (m) along the rolling path: white noise through a single second-order
    band-pass whose steepness is the microscopic roughness (Sounding Object §9.4; g05 §3.2: the
    strongly band-limited surface "became much more convincing" than 1/f^beta noise), plus
    periodic grooves every ``joints`` m (floor joints, §9.4 macroscopic features)."""
    n = int(np.clip(length / dx + 16, 2048, 1 << 21))
    roughness = float(np.clip(roughness, 0.0, 1.0))
    s = F.bandpass(np.random.default_rng(seed).standard_normal(n), 600.0, 1.0 / dx, q=4.0 - 3.4 * roughness)
    s *= 1e-6 * 10 ** (1.5 * roughness) / (s.std() + 1e-30)
    if joints > 0:
        x = (np.arange(n) * dx) % joints
        w = 0.004                                            # 4 mm wide, 0.5 mm deep groove
        s -= 5e-4 * np.where(x < w, 0.5 * (1 - np.cos(TWO_PI * x / w)), 0.0)
        # a 50 um lip behind it. A ball hopping on a rough floor flies over a bare groove (v/L lost in the envelope at 3 m/s)
        s += 5e-5 * np.where((x >= w) & (x < 2 * w), 0.5 * (1 - np.cos(TWO_PI * (x - w) / w)), 0.0)
    return s


_TEXTURES = {      # beta, periodic share, period (grains), spike share. Names from van den Doel's
    "white": (0.0, 0.0, 0, 0.0),       # texture set (thesis Fig. 7.11; g06 B6): white, 1/f^.5 and
    "pink_half": (0.5, 0.0, 0, 0.0),   # 1/f are specified there, "grid" is "a periodic profile";
    "pink": (1.0, 0.0, 0, 0.0),
    "brown": (2.0, 0.0, 0, 0.0),       # (gritty, wood, metal, plastic, sandpaper).
    "gritty": (0.3, 0.0, 0, 0.6),
    "sandpaper": (0.0, 0.0, 0, 0.25),
    "grid": (2.0, 1.0, 12, 0.0),
    "wood": (1.2, 0.5, 24, 0.0),
    "metal": (1.6, 0.35, 6, 0.0),
    "plastic": (2.24, 0.0, 0, 0.0),
}


def surface_texture(texture="pink", n: int = 1 << 15, seed: int = 0) -> np.ndarray:
    """Looped surface height profile (unit RMS), one sample per grain: a named texture or a number
    = the spectral exponent beta (0 white = very rough ... 1 = smoother, g06 B6)."""
    if isinstance(texture, str):
        if texture not in _TEXTURES:
            raise ValueError(f"unknown texture {texture!r}; choose from {sorted(_TEXTURES)} or give beta")
        beta, per, period, spikes = _TEXTURES[texture]
    else:
        beta, per, period, spikes = float(texture), 0.0, 0, 0.0
    rng = np.random.default_rng(seed + 7)
    p = fractal_noise(n, beta, 1.0, f_low=2.0 / n, seed=seed)
    if per > 0:
        ph = (np.arange(n) / period + 0.15 * np.cumsum(rng.standard_normal(n)) / math.sqrt(n)) % 1.0
        ridge = np.where(ph < 0.25, 0.5 * (1 - np.cos(TWO_PI * ph / 0.25)), 0.0)
        p = (1 - per) * p + per * (ridge - ridge.mean()) / (ridge.std() + 1e-30)
    if spikes > 0:
        s = np.where(rng.random(n) < 0.02, rng.exponential(1.0, n), 0.0)
        p = (1 - spikes) * p + spikes * (s - s.mean()) / (s.std() + 1e-30)
    return p / (p.std() + 1e-30)


def meso_force(z: np.ndarray, speed, mass: float, dx: float, sr: int) -> np.ndarray:
    """Scraper on a height map of pixels dx wide: crossing from pixel i-1 to i, the normal momentum m v (slope) of a
    body that follows the surface changes, J = m v (s_i - s_(i-1)), delivered as one impulse in that sample."""
    z = np.asarray(z, float)
    s = np.diff(np.concatenate([z, z[:1]])) / dx
    v = np.asarray(speed, float)
    x = np.cumsum(v) / sr
    p = np.floor(x / dx).astype(np.int64)
    out = np.zeros(v.shape[0])
    k = np.nonzero(np.diff(p) != 0)[0] + 1
    i = p[k] % s.size
    out[k] = mass * v[k] * (s[i] - s[(i - 1) % s.size]) * sr
    return out


def scrape_force(profile: np.ndarray, speed, force, dx: float, sr: int) -> np.ndarray:
    """Force of a scraper that follows a looped surface profile exactly (van den Doel thesis
    ch. 6; g06 B6): read the profile at ``speed`` (m/s per sample) and scale by the normal
    ``force``. ``dx`` (m) is the grain: the spacing of the profile samples."""
    pos = np.cumsum(np.asarray(speed, float)) / (sr * dx) % len(profile)
    i = pos.astype(np.int64)
    fr = pos - i
    return np.asarray(force, float) * ((1 - fr) * profile[i] + fr * profile[(i + 1) % len(profile)])


def stroke_force(t, rate: float, length: float, angle_deg: float = 30.0, mu: float = 0.6, mass: float = 0.02) -> np.ndarray:
    """Normal force (N) of a held cylinder scraped back and forth, x = L cos(w t): the push along it follows the
    acceleration, so N = m g + w^2 m L tan(a)/(1 - mu tan(a)) cos(w t), Nmax at the near end, Nmin at the far end."""
    w = TWO_PI * rate
    ta = math.tan(math.radians(angle_deg))
    if mu * ta >= 1.0:
        raise ValueError("stroke: mu x tan(angle) must stay below 1 (the scraper would jam)")
    return np.maximum(mass * 9.81 + w * w * mass * length * ta / (1.0 - mu * ta) * np.cos(w * np.asarray(t, float)), 0.0)


def contact_gate(z: np.ndarray, speed, normal, dx: float, sr: int, mass: float) -> np.ndarray:
    """1 where a scraper of ``mass`` pressed by ``normal`` (N) plus gravity follows the looped profile z (m), 0 where
    the surface falls away faster than it can be pulled down: z'' v^2 < -(N/m + g). A lost contact lasts until the
    profile comes back up to the falling scraper (free flight under N/m + g)."""
    n = max(np.size(speed), np.size(normal))
    v = np.broadcast_to(np.asarray(speed, float), (n,))
    N = np.broadcast_to(np.asarray(normal, float), (n,))
    pos = np.cumsum(v) / (sr * dx) % len(z)
    zi = np.interp(pos, np.arange(len(z)), z, period=len(z))
    g_eff = N / mass + G
    out = np.ones(n)
    h, vy, flying = 0.0, 0.0, False
    dt = 1.0 / sr
    dz = np.gradient(zi) * sr                               # surface vertical speed under the scraper
    acc = np.gradient(dz) * sr
    for i in range(n):
        if not flying:
            if acc[i] < -g_eff[i]:
                flying, h, vy = True, zi[i], dz[i]
        if flying:
            vy -= g_eff[i] * dt
            h += vy * dt
            if h <= zi[i]:
                flying = False
            else:
                out[i] = 0.0
    return out


def constrained_trajectory(z: np.ndarray, dx: float, normal=0.5, *, n_min: float = 0.0, n_max: float = 1.0, zeta: float = 0.95,
                           a_max: float = 0.05, a_min: float = 0.01) -> np.ndarray:
    """Path S of a finite-sized scraper over a height profile z: S'' = tanh(alpha z'')/alpha, alpha(N) = (1 - nu) a_max + nu a_min,
    nu = ((N - Nmin)/(Nmax - Nmin))^zeta (Agarwal, Cusimano, Traer & McDermott 2021, eqs 4-7). Curvature in the paper's units is
    per sample of the path; then a 5-sample moving Gaussian (their mean half-window at 44.1 kHz)."""
    nu = float(np.clip((float(normal) - n_min) / max(n_max - n_min, 1e-9), 0.0, 1.0)) ** zeta
    al = (1.0 - nu) * a_max + nu * a_min
    scale = float(np.max(np.abs(z))) + 1e-30                 # tanh acts on z'' relative to the deepest point. this normalisation
    c = np.tanh(al * np.gradient(np.gradient(z / scale))) / al
    s = np.cumsum(np.cumsum(c - c.mean()))
    s -= np.polyval(np.polyfit(np.arange(len(s)), s, 1), np.arange(len(s)))   # the double sum drifts
    w = np.exp(-0.5 * (np.arange(-5, 6) / 2.5) ** 2)
    return np.convolve(s, w / w.sum(), "same") * scale * (z.std() / (s.std() * scale + 1e-30))


def curvature_force(z: np.ndarray, speed, normal, dx: float, sr: int, mass: float, *, beta1: float = 0.05, beta2: float = 1.0) -> np.ndarray:
    """f = m S'' v^2 + beta1 (v S')^beta2 at the scraper's position, S the constrained path (Agarwal et al. 2021 eqs 1-3: beta1 0.05, beta2 1).
    The looped profile is read like `scrape_force`; the normal force scales both, as the Coulomb part of their horizontal force does.
    Traer et al. 2019 eqs 6-8 has the same form and calls its constants unknown."""
    v, N = np.asarray(speed, float), np.asarray(normal, float)
    s = constrained_trajectory(z, dx, float(np.mean(N)) / max(float(np.max(N)), 1e-9))
    s1, s2 = np.gradient(s, dx), np.gradient(np.gradient(s, dx), dx)
    pos = np.cumsum(v) / (sr * dx) % len(s)
    return N * (mass * np.interp(pos, np.arange(len(s)), s2, period=len(s)) * v * v
                + beta1 * np.sign(sl := np.interp(pos, np.arange(len(s)), s1, period=len(s)) * v) * np.abs(sl) ** beta2)


LOD_REF = 2.0


def lod_gains(gains, ratio, dratio, distance: float, ref: float = LOD_REF):
    """Level of detail by distance: each gain row keeps its K = ceil(J min(1, ref / distance)) most important
    modes, ranked by ear-weighted energy (g f)^2 S(f) / (4 d) on the ratio axis; the others are zeroed."""
    g = np.array(gains, float, copy=True)
    rows = np.atleast_2d(g)
    J = rows.shape[1]
    K = max(1, int(math.ceil(J * min(1.0, ref / max(float(distance), 1e-6)))))
    if K >= J:
        return g
    w = (ratio * _ear_weight(ratio * 1000.0)) ** 2 / (4.0 * np.maximum(dratio, 1e-9))
    for row in rows:
        row[np.argsort(-(row * row) * w, kind="stable")[K:]] = 0.0
    return g


def hertz_tau(m1: float, r1: float, e1: float, nu1: float, v: float, *, m2: float = math.inf, r2: float = math.inf,
              e2: float = math.inf, nu2: float = 0.0) -> float:
    """Hertz contact time tau = 2.87 (m^2 / (r E*^2 v))^(1/5), 1/m = 1/m1 + 1/m2, 1/r = 1/r1 + 1/r2,
    1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2."""
    m = 1.0 / (1.0 / m1 + 1.0 / m2)
    r = 1.0 / (1.0 / r1 + 1.0 / r2)
    es = 1.0 / ((1 - nu1 ** 2) / e1 + (1 - nu2 ** 2) / e2)
    return 2.87 * (m * m / (r * es * es * max(float(v), 1e-6))) ** 0.2


def acceleration_noise(tau: float, dv: float, radius: float, sr: int = DEFAULT_SR, *, r: float = 1.0,
                       rho: float = 1.2, c: float = 343.0) -> np.ndarray:
    """Acceleration noise of a rigid sphere (radius a) whose velocity changes by ``dv`` under a half-sine force
    sin(pi t / tau): A = (pi dv / 2 tau) sin(pi t / tau), far-field compact dipole p = rho a^3 / (2 r c) dA/dt (Pa, on axis)."""
    n = max(4, int(math.ceil(tau * sr)) + 2)
    t = np.arange(n) / sr
    on = t < tau
    dA = np.where(on, (math.pi * dv / (2 * tau)) * (math.pi / tau) * np.cos(math.pi * t / tau), 0.0)
    return rho * radius ** 3 / (2 * r * c) * dA


@numba.njit(cache=True)
def _event_bank(n, sr, ev_i, ev_a, ev_f, ev_d, ev_x, ev_tc, ev_r, ratio, dratio, gains):
    """Many small impacts, each on its own resonator: event e sends a (1 - cos) force pulse of
    ev_tc[e] samples and area ev_a[e] (van den Doel eq. 6.1; g06 B5) into modes
    f = ev_f[e] ratio[j], d = ev_d[e] dratio[j] + ev_x[e], gains[ev_r[e], j] (resonator: g06 B3)."""
    out = np.zeros(n)
    for e in range(ev_i.shape[0]):
        i0 = ev_i[e]
        tc = max(ev_tc[e], 1)
        if i0 >= n or i0 < 0:
            continue
        for j in range(ratio.shape[0]):
            f = ev_f[e] * ratio[j]
            g = gains[ev_r[e], j] * ev_a[e]
            if f >= 0.45 * sr or g == 0.0:
                continue
            dd = ev_d[e] * dratio[j] + ev_x[e]
            R = math.exp(-dd / sr)
            th = 2.0 * math.pi * f / sr
            a1 = 2.0 * R * math.cos(th)
            a2 = -R * R
            b = g * R * math.sin(th)
            end = min(n, i0 + tc + int(9.2 * sr / max(dd, 1e-3)))
            y1 = 0.0
            y2 = 0.0
            tail = max((end - i0) // 8, 1)
            for i in range(i0, end):
                kk = i - i0
                y = a1 * y1 + a2 * y2
                if kk < tc:
                    y += b * (1.0 - math.cos(2.0 * math.pi * (kk + 1) / (tc + 1))) / (tc + 1)
                y2 = y1
                y1 = y
                w = end - i                                  # closes before its stop at -80 dB, a step when nothing else sounds
                out[i] += y if w > tail else y * math.sin(0.5 * math.pi * w / tail) ** 2
    return out


# ==========================================================================================
# Helpers shared by the registered sounds
# ==========================================================================================

def _curve(v, n: int) -> np.ndarray:
    """Number -> constant; list of breakpoints -> evenly spaced curve over the sound."""
    a = np.atleast_1d(np.asarray(v, dtype=float))
    if a.size == 1:
        return np.full(n, float(a[0]))
    return np.interp(np.linspace(0, a.size - 1, n), np.arange(a.size), a)


def _out(y, sr: int, peak: float = 0.8, fade_ms: float = 8.0) -> np.ndarray:
    """DC removed, short fades, peak set; never NaN."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) < 8:
        y = np.pad(y, (0, 8 - len(y)))
    y = F.dc_block(y, sr)                                    # no y - mean: that offset is a step at both ends
    p = float(np.max(np.abs(y)))
    if p < 1e-30:
        return y
    y = declick(y / p, sr, fade_ms)                          # declick looks at the level: normalise first
    return y * (peak / (float(np.max(np.abs(y))) + 1e-30))


def hertz_k(e1: float, nu1: float, e2: float, nu2: float, radius: float) -> float:
    """Hertz contact stiffness k (N/m^1.5, F = k d^1.5) of a sphere of radius ``radius`` on a flat:
    k = 4/3 E* sqrt(R), 1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2 (Johnson, Contact Mechanics 1985, eqs 4.18-4.23)."""
    e_star = 1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)
    return 4.0 / 3.0 * e_star * math.sqrt(max(float(radius), 1e-6))


def _striker_k(mat, hardness: float, mass: float) -> float:
    """Hertz k of a spherical striker on ``mat``: hardness 0..1 runs from a rubber mallet (E 0.01 GPa, nu 0.49,
    1100 kg/m3) to steel (200 GPa, 0.30, 7850 kg/m3), log-interpolated; radius from the mass and density."""
    h = float(np.clip(hardness, 0.0, 1.0))
    e = 0.01e9 * (200e9 / 0.01e9) ** h
    nu = 0.49 + (0.30 - 0.49) * h
    rho = 1100.0 * (7850.0 / 1100.0) ** h
    r = (3.0 * max(float(mass), 1e-6) / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return hertz_k(e, nu, mat.young, mat.poisson, r)


def e_star(a, b) -> float:
    """Contact modulus of two materials: 1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2 (Johnson eq. 4.22)."""
    a, b = get_material(a), get_material(b)
    return 1.0 / ((1.0 - a.poisson ** 2) / a.young + (1.0 - b.poisson ** 2) / b.young)


def _hertz_pair(a, b, radius: float) -> float:
    a, b = get_material(a), get_material(b)
    return hertz_k(a.young, a.poisson, b.young, b.poisson, radius)


def power_profile_k(n: float, A: float, es: float) -> tuple[float, float]:
    """(k, alpha) of F = k d^alpha for a punch z = A r^(2n), integer n (Steuermann, Johnson eqs 5.20-5.22):
    alpha = (2n + 1)/2n, k = 4n/(2n + 1) E* (A C)^(-1/2n), C = 2.4..2n / 1.3..(2n-1). n = 1, A = 1/2R is Hertz."""
    n = int(n)
    C = float(np.prod([2.0 * i / (2.0 * i - 1.0) for i in range(1, n + 1)]))
    return 4.0 * n / (2.0 * n + 1.0) * es * (A * C) ** (-0.5 / n), (2.0 * n + 1.0) / (2.0 * n)


def cone_k(es: float, half_angle_deg: float) -> float:
    """F = k d^2 of a rigid cone: k = (2/pi) E* tan(alpha), alpha the semi-angle (Love 1939; Johnson eq. 5.14)."""
    return 2.0 / math.pi * es * math.tan(math.radians(half_angle_deg))


def line_contact(p: float, r: float, es: float) -> tuple[float, float]:
    """Half-width a = sqrt(4 P R/(pi E*)) and p0 = sqrt(P E*/(pi R)), P per unit length (Johnson §4.3)."""
    return math.sqrt(4.0 * p * r / (math.pi * es)), math.sqrt(p * es / (math.pi * r))


def sphere_yield_load(r: float, y: float, es: float) -> float:
    """First yield of a sphere: Py = pi^3 R^2 (1.60 Y)^3/(6 E*^2) (Johnson eq. 6.10, Tresca)."""
    return math.pi ** 3 * r * r * (1.60 * y) ** 3 / (6.0 * es * es)


def cylinder_yield_load(r: float, y: float, es: float) -> float:
    """First yield of a cylinder, per unit length: Py = pi R (1.67 Y)^2/E* (Johnson eq. 6.7, Tresca)."""
    return math.pi * r * (1.67 * y) ** 2 / es


def yield_speed(rho: float, y: float, es: float) -> float:
    """Impact speed of first yield, rho Vy^2/Y = 26 (Y/E*)^4 (Johnson eq. 11.39). Hard steel on mild steel, Y 1 GPa: 0.14 m/s."""
    return math.sqrt(26.0 * (y / es) ** 4 * y / rho)


def plastic_restitution(mass: float, v: float, r: float, y: float, es: float) -> float:
    """e = 3.8 (Y/E*)^(1/2) (m V^2/(2 Y R^3))^(-1/8), fully plastic, p_d = 3 Y (Johnson eq. 11.45), capped at 1: it
    exceeds 1 until the indent is fully plastic. The steel ball at Y 1 GPa reads 1.4 at 1 m/s, 0.6 near 29 m/s."""
    return min(1.0, 3.8 * math.sqrt(y / es) * (0.5 * mass * v * v / (y * r ** 3)) ** -0.125)


def plastic_contact_time(mass: float, r: float, y: float, e: float = 0.0) -> float:
    """t_p + t' = (pi m/(8 R p_d))^(1/2) (1 + 1.2 e), p_d = 3 Y, independent of V (Johnson eqs 11.46-11.47)."""
    return math.sqrt(math.pi * mass / (8.0 * r * 3.0 * y)) * (1.0 + 1.2 * e)


def viscoelastic_restitution(tan_delta: float) -> float:
    """e = (1 - pi tan(phi))^(1/2), tan(phi) at w ~ pi/Tc (Johnson eq. 11.48)."""
    return math.sqrt(max(1.0 - math.pi * tan_delta, 0.0))


def jkr_pulloff(gamma: float, r: float, es: float) -> tuple[float, float]:
    """Adhesion: pull-off force 3 pi gamma R and contact radius there (9 pi gamma R^2/(4 E*))^(1/3), the minimum of `jkr_load`."""
    return 3.0 * math.pi * gamma * r, (9.0 * math.pi * gamma * r * r / (4.0 * es)) ** (1.0 / 3.0)


def winkler_stiffness(es: float, a: float, line: bool = False) -> float:
    """K/h matched to Hertz: 1.70 E*/a at a point, 1.18 E*/a on a line (Johnson §4.4, after eqs 4.56, 4.59)."""
    return (1.18 if line else 1.70) * es / a


def winkler_roller(p: float, r: float, k: float) -> tuple[float, float]:
    """A cylinder of radius r lying on a Winkler foundation (pressure k x sinking, Pa/m) under load p per unit
    length: p = 2 k b^3 / (3 r), sinking d = b^2 / (2 r). Returns (half-width b, sinking d): a log, a tube or a
    roller resting on soft ground or a plank."""
    b = (1.5 * p * r / k) ** (1.0 / 3.0)
    return b, b * b / (2.0 * r)


def tangential_stiffness(g: float, a: float, nu: float) -> float:
    """Tangential stiffness of a circular contact of radius a before slip: 8 G a / (2 - nu)."""
    return 8.0 * g * a / (2.0 - nu)


def torsional_stiffness(g: float, a: float) -> float:
    """Twisting stiffness of a bonded circular contact: 16 G a^3 / 3."""
    return 16.0 * g * a ** 3 / 3.0


def relative_radius(r1: float, r2: float) -> float:
    """1 / (1/r1 + 1/r2) with a concave surface given a negative radius (a ball in a cup: a larger, conforming
    contact)."""
    c = 1.0 / r1 + 1.0 / r2
    if c <= 0:
        raise ValueError("a concave radius must be larger than the convex one")
    return 1.0 / c


def crater_radius(a: float, r: float, p: float, es: float) -> float:
    """Radius of curvature rho of the dent left by a plastic indentation of radius a under load p: 4 a^3 (1/r - 1/rho)
    = 3 p / E*. Later blows of the same load on it are elastic (the dent conforms)."""
    c = 1.0 / r - 3.0 * p / (4.0 * a ** 3 * es)
    return math.inf if c <= 0 else 1.0 / c


def mud_contact_radius(r: float, eta: float, p0: float, t) -> np.ndarray:
    """Contact radius of a sphere pushed with force p0 into a viscous layer (mud): a^3 = 3 r p0 t / (16 eta)."""
    return np.cbrt(3.0 * r * p0 * np.asarray(t, float) / (16.0 * eta))


def plasticity_index(es: float, hardness: float, sigma: float, kappa: float) -> float:
    """psi = (E*/H) sqrt(sigma kappa), sigma the asperity height spread, kappa their curvature (Greenwood-Williamson; Johnson §13.4)."""
    return es / hardness * math.sqrt(sigma * kappa)


def hertz_elliptic(p: float, r1: float, r2: float, es: float) -> tuple[float, float, float, float]:
    """Contact of two bodies with principal relative radii ``r1``, ``r2`` (gap z = x^2/2r1 + y^2/2r2) under load p:
    (a, b, delta, p0), a and b the semi-axes, a along the larger radius, delta the approach, p0 = 3P/(2 pi a b).
    With al = Ry/Rx: ellipticity k = al^(2/pi), elliptic integrals E = 1 + (pi/2 - 1)/al, F = pi/2 + (pi/2 - 1) ln al
    (exact Hertz at al = 1)."""
    rx, ry = min(r1, r2), max(r1, r2)
    R = 1.0 / (1.0 / rx + 1.0 / ry)
    al = ry / rx
    k = al ** (2.0 / math.pi)
    E = 1.0 + (0.5 * math.pi - 1.0) / al
    Fi = 0.5 * math.pi + (0.5 * math.pi - 1.0) * math.log(al)
    ep = 2.0 * es
    a = (6.0 * k * k * E * p * R / (math.pi * ep)) ** (1.0 / 3.0)
    b = (6.0 * E * p * R / (math.pi * k * ep)) ** (1.0 / 3.0)
    d = Fi * (9.0 / (2.0 * E * R) * (p / (math.pi * k * ep)) ** 2) ** (1.0 / 3.0)
    return a, b, d, 3.0 * p / (2.0 * math.pi * a * b)


def hertz_k_elliptic(es: float, r1: float, r2: float) -> float:
    """k of F = k d^1.5 for a non-spherical (crossed cylinders, edge, barrel) contact, from `hertz_elliptic`."""
    return hertz_elliptic(1.0, r1, r2, es)[2] ** -1.5


def line_contact_shear(p: float, r: float, es: float) -> tuple[float, float]:
    """Largest subsurface shear of a line contact, 0.30 p0, and its depth 0.78 a."""
    a, p0 = line_contact(p, r, es)
    return 0.30 * p0, 0.78 * a


def cylinder_between_plates(p: float, r: float, e: float, nu: float) -> float:
    """Approach of the two flats squeezing a cylinder (load p per unit length, same material):
    2 P (1 - nu^2)/(pi E) (2 ln(4R/a) - 1), a the Hertz half-width; nearly linear in P."""
    es = e / (2.0 * (1.0 - nu * nu))
    a = line_contact(p, r, es)[0]
    return 2.0 * p * (1.0 - nu * nu) / (math.pi * e) * (2.0 * math.log(4.0 * r / a) - 1.0)


def hertz_edge_tension(p0: float, nu: float) -> float:
    """Radial tension at the rim of a Hertz circle, (1 - 2 nu) p0/3: where glass starts its ring cracks."""
    return (1.0 - 2.0 * nu) * p0 / 3.0


def jkr_load(a: float, gamma: float, r: float, es: float) -> float:
    """Load for contact radius a with adhesion gamma (J/m^2): P = 4E* a^3/(3R) - sqrt(16 pi gamma E* a^3).
    Under load control it snaps off at -3 pi gamma R; under displacement control it holds to -5/9 of that."""
    return 4.0 * es * a ** 3 / (3.0 * r) - math.sqrt(16.0 * math.pi * gamma * es * a ** 3)


def elastoplastic_mean_pressure(a: float, r: float, y: float, es: float) -> float:
    """Mean pressure under a sphere at contact radius a: Hertz 4E*a/(3 pi R) until the first yield (1.07 Y),
    then the expanding-cavity law (2/3) Y (1 + ln(E* a/(3 Y R))) once it exceeds 1.07 Y, capped at the full plastic 3 Y."""
    x = es * a / (y * r)
    pe = 4.0 * x / (3.0 * math.pi)
    if pe <= 1.07:
        return pe * y
    return y * min(3.0, max(1.07, 2.0 / 3.0 * (1.0 + math.log(x / 3.0))))


def elastoplastic_contact(delta: float, delta_max: float, r: float, y: float, es: float) -> tuple[float, float]:
    """Force at approach ``delta`` after a largest approach ``delta_max`` (loading if delta >= delta_max), and the
    crater depth left. Loading: P = pi a^2 pm(a), a^2 = R delta. Unloading is elastic, Hertz on the recovered depth
    3 Pmax/(4 E* amax): P = Pmax ((delta - d_res)/(delta_max - d_res))^1.5, d_res the crater."""
    dm = max(delta_max, delta, 0.0)
    am = math.sqrt(r * dm)
    pmax = math.pi * am * am * elastoplastic_mean_pressure(am, r, y, es)
    rec = min(dm, 3.0 * pmax / (4.0 * es * am)) if am > 0 else 0.0
    res = dm - rec
    if delta >= dm:
        return pmax, res
    return (pmax * (max(delta - res, 0.0) / rec) ** 1.5 if rec > 0 else 0.0), res


def matthews_alpha(n: float) -> float:
    """Exponent of F = k d^alpha for a sphere into a power-law hardening solid, strain ~ stress^n:
    pm ~ (a/R)^(1/n), d ~ a^2, so alpha = 1 + 1/(2n): 1.5 elastic (n = 1), 1 fully plastic (n -> inf)."""
    return 1.0 + 0.5 / max(float(n), 1.0)


def matthews_indentation(a: float, r: float, n: float, sigma0: float, eps0: float) -> tuple[float, float]:
    """(delta, P) of a sphere at contact radius a in eps = eps0 (sigma/sigma0)^n: delta = (2n/(2n+1))^(2(n-1)) a^2/R,
    P = pi a^2 pm, pm = 3 sigma0 (eps_R/eps0)^(1/n), representative strain eps_R = 0.2 a/R."""
    n = max(float(n), 1.0)
    d = (2.0 * n / (2.0 * n + 1.0)) ** (2.0 * (n - 1.0)) * a * a / r
    return d, math.pi * a * a * 3.0 * sigma0 * (0.2 * a / r / eps0) ** (1.0 / n)


def layer_k(e: float, nu: float, thickness: float, r: float, bonded: bool = False) -> tuple[float, float]:
    """(k, alpha) of F = k d^alpha for a sphere on a thin layer (felt, rubber pad, paint, coating) on a rigid base.
    Compressible, frictionless: P = (2/3) E a^3/((1 - nu^2) R b), d = a^2/2R; bonded x (1 - nu)^2/(1 - 2 nu).
    Bonded and incompressible (nu >= 0.499): P = 2 E a^5/(45 R b^3), d = a^2/6R, alpha 2.5."""
    b = thickness
    if bonded and nu >= 0.499:
        return 2.0 * e * (6.0 * r) ** 2.5 / (45.0 * r * b ** 3), 2.5
    k = 2.0 / 3.0 * e * (2.0 * r) ** 1.5 / ((1.0 - nu * nu) * r * b)
    if bonded:
        k *= (1.0 - nu) ** 2 / (1.0 - 2.0 * nu)
    return k, 1.5


def wavy_contact(p: float, es: float, amp: float, wavelength: float, two_d: bool = False) -> float:
    """Fraction of a wavy surface (amplitude amp, wavelength l) in contact under mean pressure p. 1-D: p* = pi E* amp/l,
    2a/l = (2/pi) asin(sqrt(p/p*)); 2-D: p* = 2 pi E* amp/l, pi a^2/l^2 = pi (3p/(8 pi p*))^(2/3) at light load,
    1 - (3/(2 pi))(1 - p/p*) near full contact, the two joined linearly between p/p* 0.3 and 0.8. 1 when p >= p*."""
    ps = (2.0 if two_d else 1.0) * math.pi * es * amp / wavelength
    q = min(max(p / ps, 0.0), 1.0)
    if not two_d:
        return 2.0 / math.pi * math.asin(math.sqrt(q))
    light = lambda s: math.pi * (3.0 * s / (8.0 * math.pi)) ** (2.0 / 3.0)
    full = lambda s: 1.0 - 1.5 / math.pi * (1.0 - s)
    if q <= 0.3:
        return light(q)
    if q >= 0.8:
        return full(q)
    return light(0.3) + (q - 0.3) / 0.5 * (full(0.8) - light(0.3))


def gw_contact(p: float, es: float, sigma_s: float, kappa_s: float) -> tuple[float, float]:
    """Rough flat (exponential asperity heights): real-area fraction sqrt(pi) (sigma kappa)^(-1/2) p/E*, and the
    mean real pressure 0.56 E* (sigma kappa)^(1/2), independent of load. sigma from Ra: sqrt(pi/2) Ra (Gaussian)."""
    sk = sigma_s * kappa_s
    return min(math.sqrt(math.pi) * p / (es * math.sqrt(sk)), 1.0), 0.56 * es * math.sqrt(sk)


def rough_sphere_alpha(sigma_s: float, r: float, p: float, es: float) -> float:
    """Roughness parameter of a rough sphere, sigma_s/delta0 = sigma_s (16 R E*^2/(9 P^2))^(1/3): below 0.05 it is Hertz."""
    return sigma_s * (16.0 * r * es * es / (9.0 * p * p)) ** (1.0 / 3.0)


def rough_hertz_k(k: float, alpha: float) -> float:
    """Hertz k softened by roughness: the approach of a rough sphere grows by (1 + 2 alpha) over the Hertz one at the
    same load (the asperity layer is compressed first), so k / (1 + 2 alpha)^1.5; unchanged below alpha 0.05."""
    return k if alpha < 0.05 else k / (1.0 + 2.0 * alpha) ** 1.5


def mindlin(q: float, p: float, mu: float, a: float, g1: float, nu1: float, g2: float, nu2: float) -> tuple[float, float, float]:
    """Partial slip of two spheres under a tangential force q < mu p: (c/a, tangential displacement, stiffness).
    c/a = (1 - Q/mu P)^(1/3); d = (3 mu P/16a) ((2 - nu1)/G1 + (2 - nu2)/G2) (1 - (1 - Q/mu P)^(2/3));
    k_t = 8 G* a (1 - Q/mu P)^(1/3), 1/G* = (2 - nu1)/G1 + (2 - nu2)/G2."""
    s = (2.0 - nu1) / g1 + (2.0 - nu2) / g2
    u = max(1.0 - min(abs(q) / (mu * p), 1.0), 0.0)
    return u ** (1.0 / 3.0), 3.0 * mu * p / (16.0 * a) * s * (1.0 - u ** (2.0 / 3.0)), 8.0 * a / s * u ** (1.0 / 3.0)


def mindlin_loss(q_amp: float, p: float, mu: float, a: float, g1: float, nu1: float, g2: float, nu2: float) -> float:
    """Energy lost per cycle of a tangential force +-q_amp (micro-slip, fretting):
    (9 mu^2 P^2/10a) s (1 - (1 - Q/mu P)^(5/3) - (5Q/6 mu P)(1 + (1 - Q/mu P)^(2/3))), s = (2 - nu1)/G1 + (2 - nu2)/G2."""
    s = (2.0 - nu1) / g1 + (2.0 - nu2) / g2
    x = min(abs(q_amp) / (mu * p), 1.0)
    u = 1.0 - x
    return 9.0 * mu * mu * p * p / (10.0 * a) * s * (1.0 - u ** (5.0 / 3.0) - 5.0 * x / 6.0 * (1.0 + u ** (2.0 / 3.0)))


def mindlin_torque(p: float, mu: float, a: float) -> float:
    """Torque to make a Hertz circle spin (full slip): 3 pi mu P a/16."""
    return 3.0 * math.pi * mu * p * a / 16.0


def sliding_traction(mu: float, p0: float) -> tuple[float, bool]:
    """A sliding Hertz contact: the tension at the trailing edge, 2 mu p0, and whether yield starts at the surface
    (mu > 0.30, von Mises): above it a scratch tears instead of gliding."""
    return 2.0 * mu * p0, mu > 0.30


def oblique_impact(vx: float, vz: float, mu: float, nu: float = 0.3, gyration: float = 0.4, steps: int = 4000) -> dict:
    """A sphere hitting a flat obliquely (tangential vx, normal speed vz), elastic, same material on both sides:
    Hertz normal spring, incremental Mindlin tangential spring kappa x the normal tangent stiffness, Coulomb cap
    mu x normal force. kappa = 2(1 - nu)/(2 - nu), chi = kappa (1 + 1/gyration)/2 (gyration = k^2/R^2, 0.4 solid),
    psi = kappa vx/(mu vz). Returns kappa, chi, psi, the regime (stick if psi <= 1, gross slip if psi >= 4 chi - 1,
    partial slip between) and the tangential velocity of the contact point after the impact, ``vt_out`` (negative:
    the spin turned it back)."""
    kap = 2.0 * (1.0 - nu) / (2.0 - nu)
    chi = kap * (1.0 + 1.0 / gyration) / 2.0
    psi = kap * abs(vx) / (mu * abs(vz) + 1e-30)
    regime = "stick" if psi <= 1.0 else ("gross_slip" if psi >= 4.0 * chi - 1.0 else "partial_slip")
    mstar = gyration / (1.0 + gyration)
    vz = abs(vz)
    scale = vz ** -0.2                                   # m = K = 1: the contact lasts about 3.2 vz^(-1/5)
    dt = 4.0 * scale / steps
    z, vn, vt, ft = 0.0, vz, float(vx), 0.0
    for i in range(steps):
        fn = z ** 1.5 if z > 0 else 0.0
        kt = kap * 1.5 * math.sqrt(z) if z > 0 else 0.0
        cap = mu * fn
        ft = min(max(ft - kt * vt * dt, -cap), cap)
        vn -= fn * dt
        z += vn * dt
        vt += ft / mstar * dt
        if i > 0 and z <= 0.0:
            break
    return {"kappa": kap, "chi": chi, "psi": psi, "regime": regime, "vt_out": vt, "vn_out": vn}


def rayleigh_speed(c2: float, nu: float) -> float:
    """Rayleigh surface-wave speed c3 = alpha c2: (2 - alpha^2)^4 = 16 (1 - alpha^2)(1 - alpha^2 c2^2/c1^2);
    alpha 0.919 at nu 0.25, 0.955 at nu 0.5."""
    k2 = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))       # (c2/c1)^2
    al = brentq(lambda a: (2.0 - a * a) ** 4 - 16.0 * (1.0 - a * a) * (1.0 - a * a * k2), 0.5, 0.9999)
    return al * c2


def halfspace_radiated_power(p_amp: float, omega: float, g: float, c2: float) -> float:
    """Mean power an oscillating point force P* cos(wt) sends into an elastic half-space as waves (nu 0.25):
    W = 0.074 P*^2 w^2/(G c2), 7 % P waves, 26 % S, 67 % Rayleigh."""
    return 0.074 * p_amp * p_amp * omega * omega / (g * c2)


def johnson_regime(rho: float, v: float, sigma_s: float) -> str:
    """Damage number rho V^2 / sigma_s: below 1e-6 elastic, to 1e-3 elasto-plastic, to 1e-1 fully plastic, beyond
    that the target flows."""
    j = rho * v * v / sigma_s
    return "elastic" if j < 1e-6 else "elastoplastic" if j < 1e-3 else "plastic" if j < 1e-1 else "hypervelocity"


def impact_regime(rho: float, v: float, y: float, es: float) -> str:
    """'elastic' below the first-yield speed, 'plastic' up to rho V^2/Y = 1, 'hypervelocity' beyond (the target flows)."""
    if v < yield_speed(rho, y, es):
        return "elastic"
    return "plastic" if rho * v * v / y < 1.0 else "hypervelocity"


def moving_contact_ratio(v: float, c1: float, c2: float, nu: float) -> float:
    """a(V)/a(0) of a load sliding at V on an elastic half-space: {beta1 M2^2/((1 - nu) N)}^(1/2),
    M2 = V/c2, beta_i = (1 - (V/c_i)^2)^(1/2), N = (2 - M2^2)^2 - 4 beta1 beta2; it diverges at the Rayleigh speed.
    Above it (up to c2) there is no steady solution: nan."""
    if v <= 0:
        return 1.0
    m2 = v / c2
    b1, b2 = math.sqrt(max(1.0 - (v / c1) ** 2, 0.0)), math.sqrt(max(1.0 - m2 * m2, 0.0))
    N = (2.0 - m2 * m2) ** 2 - 4.0 * b1 * b2
    if N >= 0:
        return float("nan")
    return math.sqrt(b1 * m2 * m2 / ((1.0 - nu) * -N))


def viscoelastic_contact_radius(t, p0: float, r: float, g1: float, g2: float, eta: float) -> np.ndarray:
    """Radius of a sphere pressed by a step load p0 into a standard linear solid (spring g1 in series with g2 || eta):
    a^3(t) = (3/8) R P0 Phi(t), creep Phi = 1/g1 + (1/g2)(1 - exp(-t/T1)), T1 = eta/g2. It grows from the glassy to
    the rubbery radius: flesh, gel, rubber settling."""
    t = np.asarray(t, float)
    phi = 1.0 / g1 + (1.0 - np.exp(-np.maximum(t, 0.0) * g2 / eta)) / g2
    return (3.0 / 8.0 * r * p0 * phi) ** (1.0 / 3.0)


def hunter_restitution(tc: float, relax_t: float) -> float:
    """Restitution of a Maxwell-damped elastic impact, 1 - (4/9) Tc/T (T the relaxation time, Tc << T)."""
    return max(1.0 - 4.0 / 9.0 * tc / relax_t, 0.0)


_WAVE_BETA = ((0.25, 1.0 / 3.0), (0.5374, 0.415))          # Poisson's ratio -> beta of the radiated power


def wave_restitution(v: float, e_body: float, nu_body: float, rho_body: float, e_ball: float, nu_ball: float,
                     rho_ball: float) -> float:
    """Restitution of a small elastic sphere on a massive elastic body when the only loss is the elastic waves sent
    into the body: e = (1 - lam)^(1/2), lam = 2 tau (4 pi/3)^(-1/5) rho1^(-1/5) g^(-6/5) V^(3/5) / (rho C0^3),
    g = (1 - nu^2)/E + (1 - nu1^2)/E1, C0 = (E/rho)^(1/2) of the body,
    tau = 1.068^5 (1 + nu) ((1 - nu^2)/(1 - 2 nu))^(1/2) (16/15)^(6/5) beta(nu). Independent of the radius and close
    to 1 (steel on steel 0.997 at 1 m/s): what a metal impact loses is internal damping, not waves."""
    nu = float(np.clip(nu_body, 0.25, 1.0 / 3.0))
    beta = float(np.interp(nu, *_WAVE_BETA))
    tau = 1.068 ** 5 * (1.0 + nu) * math.sqrt((1.0 - nu * nu) / (1.0 - 2.0 * nu)) * (16.0 / 15.0) ** 1.2 * beta
    g = (1.0 - nu_body ** 2) / e_body + (1.0 - nu_ball ** 2) / e_ball
    c0 = math.sqrt(e_body / rho_body)
    lam = 2.0 * tau * (4.0 * math.pi / 3.0) ** -0.2 * rho_ball ** -0.2 * g ** -1.2 * abs(v) ** 0.6 / (rho_body * c0 ** 3)
    return math.sqrt(max(1.0 - lam, 0.0))


def patch_average(profile: np.ndarray, a: float, dx: float) -> np.ndarray:
    """Mean height of a looped profile (m, every ``dx`` m) over a contact of half-length ``a`` centred on each point:
    a wavelength L is scaled by sin(k a)/(k a), k = 2 pi/L, gone at L = 2a; rolled at speed U, a low-pass whose first
    zero is U/(2a)."""
    n = int(round(2.0 * a / dx)) // 2 * 2 + 1
    return np.asarray(profile, float) if n < 3 else uniform_filter1d(np.asarray(profile, float), n, mode="wrap")


def rolling_microslip(a: float, r: float, *, beta: float = 0.0, mu: float = 0.0, groove: float = 0.0) -> float:
    """Rolling friction coefficient from micro-slip: dissimilar elastic materials 1.5e-4 |beta| a/R (beta Dundurs'
    second constant: perspex/steel 0.19, glass/steel 0.21, duralumin/steel 0.12); a ball in a conforming groove
    (contact half-length groove = b) 0.08 mu (a/R)^2 (b/a)^2. The two add."""
    out = 1.5e-4 * abs(beta) * a / r
    if groove > 0 and mu > 0:
        out += 0.08 * mu * (a / r) ** 2 * (groove / a) ** 2
    return out


def hertz_fit(tau: float, fmax: float, mass: float) -> float:
    """K from a measured contact time and peak force (Chaigne & Kergomard eq. 1.145): K = (35.41/tau^3) sqrt(m^3/Fmax)."""
    return 35.41 / tau ** 3 * math.sqrt(mass ** 3 / fmax)


def sonic_distance(a: "ModalObject", b: "ModalObject", floor_db: float = 60.0) -> float:
    """van den Doel's distance between two equal-length mode sets: sum S(f) [H(log(E1/E0)) - H(log(E2/E0))]^2, E = (a f)^2,
    H(x) = max(x, 0), S genny's ear weighting, E0 ``floor_db`` under the larger set's top (thesis §4.6)."""
    ea, eb = (a.gains() * a.freqs) ** 2, (b.gains() * b.freqs) ** 2
    e0 = max(float(ea.max()), float(eb.max())) * 10 ** (-abs(floor_db) / 10)
    ha, hb = (np.maximum(np.log(np.maximum(e, 1e-300) / e0), 0.0) for e in (ea, eb))
    return float(np.sum(_ear_weight(a.freqs) * (ha - hb) ** 2))


def lloyd_gain(v: float, xi) -> np.ndarray:
    """Per-hit gain of a mode, g = lerp(1, xi, alpha(v))/c(v), alpha = bias(v, 0.9), c = sqrt(1 - alpha + alpha^2/3), xi uniform 0..1:
    unit mean square, more spread the harder the hit (Lloyd, Raghuvanshi & Govindaraju 2011). bias(x, b) = x/((1/b - 2)(1 - x) + 1)."""
    v = float(np.clip(v, 0.0, 1.0))
    al = v / ((1.0 / 0.9 - 2.0) * (1.0 - v) + 1.0)
    return (1.0 + (np.asarray(xi, float) - 1.0) * al) / math.sqrt(1.0 - al + al * al / 3.0)


def spectral_dilation(freqs, sg: float, sr_: float) -> np.ndarray:
    """f = SG f sqrt(1 + SR (f/f1)^2), with f1 the lowest (Aramaki et al. 2011). Metal SG 0.5, SR 0.1; wood 0.85, 0.05; glass 2.4, 0.2.
    No Tukey window: every mode is dilated."""
    f = np.asarray(freqs, float)
    return float(sg) * f * np.sqrt(np.maximum(1.0 + float(sr_) * (f / f.min()) ** 2, 1e-6))


def _series(*ks) -> float:
    """Two compliant surfaces in contact: the softer one dominates (Hertz: 1/E* = sum (1 - nu^2)/E)."""
    return 1.0 / sum(1.0 / k for k in ks)


def _split_shape(shape: str):
    for b in ("clamped_clamped", "clamped", "supported", "free"):
        if shape == f"bar_{b}":
            return "bar", b
    return shape, "free"


REALIMPACT_FREQS = {    # Hz, lowest measured partials of real objects struck in an anechoic room
    "ceramic_bowl": (830.0, 2112.0, 3805.0), "glass": (1949.0, 4753.0, 9608.0), "wood_plate": (1127.0, 2067.0, 3147.0),
    "iron_pan": (473.0, 1103.0, 1246.0), "plastic_bowl": (2319.0, 3740.0, 10825.0), "steel_pot": (1646.0, 1888.0, 2379.0),
    "polycarbonate_cup": (1159.0, 1342.0, 1573.0),
}


def realimpact_body(name: str, mat, size: float = 0.3) -> ModalObject:
    """A measured object (`REALIMPACT_FREQS`) as a modal body: its partials at their measured pitch (size does not
    scale them), decays f/fd of ``mat``, equal mode shapes, the mass of a ``size`` sphere shell of ``mat``."""
    mat = get_material(mat)
    f = np.array(REALIMPACT_FREQS[name])
    b = ModalObject(f, f / mat.fd + mat.ext, np.ones(len(f)), mass=mat.density * 4.0 * math.pi * (0.5 * size) ** 2 * 0.003)
    b.radiator_radius = 0.5 * size
    return b


def directivity_gains(n: int, azimuth: float, seed: int = 0) -> np.ndarray:
    """Level of each of n modes heard from ``azimuth`` (degrees) instead of averaged over directions: mode j radiates
    in o_j = ceil(sqrt(j + 1)) lobe pairs (lobes grow as the square root of the mode count of a 2-D body), at a
    seeded orientation; power gain 0.25 + 1.5 cos^2(o_j phi + psi_j), mean 1, nulls at -6 dB (diffraction fills them)."""
    rng = np.random.default_rng(int(seed) + 9001)
    o = np.ceil(np.sqrt(np.arange(n) + 1.0))
    psi = rng.uniform(0.0, math.pi, n)
    return np.sqrt(0.25 + 1.5 * np.cos(o * math.radians(azimuth) + psi) ** 2)


DOUBLET_SPLIT = 0.01


def sample_modes(material, n: int = 24, f1: float = 300.0, *, shape: str = "rect_plate", seed: int = 0,
                 tries: int = 200) -> ModalObject:
    """A new object of ``material``: n modes drawn from a multivariate Gaussian on (log amplitude, log decay, log
    frequency step) around the material's analytic ``shape`` (its steps and fd), std 0.35 / material scatter / 0.25,
    log decay following log frequency (d = f/fd); draws whose mean spacing leaves +-10 % of the reference are rejected.
    A degenerate pair of the reference (square plate) is split by 1 % before the draw."""
    mat = get_material(material)
    ref = ShapeBody(shape, float(f1), material=mat, n_modes=n)
    lf = np.log(ref.freqs)
    steps = np.diff(lf)
    steps = np.where(steps > 1e-9, steps, DOUBLET_SPLIT)
    rng = np.random.default_rng(seed)
    target = float(np.mean(np.diff(ref.freqs)))
    sig = np.array([0.35, max(float(mat.scatter), 0.05), 0.25])
    for _ in range(tries):
        z = rng.standard_normal((len(steps), 3)) * sig
        f = ref.freqs[0] * np.exp(np.concatenate([[0.0], np.cumsum(steps * np.exp(z[:, 2]))]))
        if abs(float(np.mean(np.diff(f))) / target - 1.0) <= 0.10:
            break
    za = np.concatenate([[0.0], z[:, 0]])
    zd = np.concatenate([[0.0], z[:, 1]])
    return ModalObject(f, f / mat.fd * np.exp(zd) + mat.ext, ref.phi * np.exp(za), mass=ref.mass, name=f"{mat.name}_sampled")


def effective_mass(mass: float, inertia, r, n) -> float:
    """Mass a contact at offset r (m, from the centre of mass) feels along the normal n of a free rigid body:
    1/m_eff = 1/M + (r x n)^T I^-1 (r x n), I the 3x3 inertia tensor (kg m^2). Off-centre hits feel lighter."""
    c = np.cross(np.asarray(r, float), np.asarray(n, float) / np.linalg.norm(n))
    return 1.0 / (1.0 / mass + float(c @ np.linalg.solve(np.asarray(inertia, float), c)))


def _make_body(shape, mat, size, position=0.3, position2=0.37, *, freq=0.0, modes=24, damping=0.0, seed=0,
               fmax=12000.0, scatter=None, aspect=1.3, law="friction", thickness=None, rest_force=0.0):
    """A catalog body: an analytic shape (pitch and mass from size and material) or a measured
    ``modal_data`` table name (frequencies scaled by 0.3 / size). ``thickness`` (m) replaces the proportional one of
    `_thickness`, with which a bar goes as 1/L instead of 1/L^2. ``rest_force`` (N): resting on something, each mode gains
    d = gamma F phi^2/(2 m), a normal contact (Zheng & James 2011 eqs 16, 19-20, with C = 2 d); gamma from CONTACT_GAMMA."""
    mat = get_material(mat)
    if law == "objectfolder" and mat.name in OBJECTFOLDER_ELASTIC:
        rho, ym, nu = OBJECTFOLDER_ELASTIC[mat.name]
        mat = replace(mat, density=rho, young=ym, poisson=nu)
    size = float(np.clip(size, 0.005, 20.0))
    shape, boundary = _split_shape(str(shape))
    modes = int(np.clip(modes, 1, 200))
    if shape in SHAPES:
        asp = aspect if shape.startswith("rect") else 1.0
        f1 = float(freq) if freq and freq > 0 else fundamental(shape, size, mat, boundary=boundary, aspect=asp, thickness=thickness)
        # every edge that is held is a node of all modes: keep the strike point just inside it
        lo = 0.0 if shape.startswith("circ") else 0.02
        pos = float(np.clip(position, lo, 0.98))
        if shape.startswith("rect") or shape == "l_membrane":
            pos = (pos, float(np.clip(position2, 0.02, 0.98)))
        # f1 may be inaudible, its partials are not: 5 Hz is only ShapeBody's own guard
        body = ShapeBody(shape, float(np.clip(f1, 5.0, 8000.0)), material=mat, boundary=boundary, aspect=asp,
                         position=pos, n_modes=modes, fmax=fmax, mass=body_mass(shape, size, mat, aspect=asp, thickness=thickness),
                         ext=(mat.ext if law == "friction" else 0.0) + max(float(damping), 0.0), scatter=scatter,
                         law=law, seed=seed)
        body.radiator_radius = 0.5 * size
        if rest_force > 0:
            body.decays = body.decays + _rest_damping(body, mat, rest_force)
        return body
    if shape.startswith("ri_") and shape[3:] in REALIMPACT_FREQS:
        b = realimpact_body(shape[3:], mat, size)
        if freq and freq > 0:
            b.freqs = b.freqs * (float(freq) / b.freqs.min())
        b.decays = b.decays + max(float(damping), 0.0)
        return b
    if shape not in _modal.md.TABLES:
        raise ValueError(f"unknown body {shape!r}: use one of {SHAPES} (bar_clamped, bar_supported, "
                         f"bar_clamped_clamped) or a modal table such as 'sy_vase'")
    b = ModalObject.from_table(shape, f_scale=0.3 / size, position=float(np.clip(position, 0, 1)))
    if freq and freq > 0:
        b.freqs = b.freqs * (float(freq) / b.freqs.min())
    b.decays = b.decays + max(float(damping), 0.0)
    b.radiator_radius = 0.5 * size
    return b.pruned(keep=modes) if len(b.freqs) > modes else b


def _rest_damping(body, mat, force):
    g = CONTACT_GAMMA.get(mat.name)
    if g is None:
        raise ValueError(f"no sourced contact damping for {mat.name}; materials with one: {sorted(CONTACT_GAMMA)}")
    return 0.5 * g * float(force) * body.phi ** 2 / body.mass


def _ball_mass(mat, diameter: float) -> float:
    return get_material(mat).density * math.pi / 6.0 * float(diameter) ** 3


def _hertz_samples(mass, k, alpha, mu, v, sr):
    """Contact length in samples for each speed v (one eq. 8.27 integral, Hertz-scaled in v)."""
    t1 = hc_contact_time(mass, k, alpha, mu, 1.0)
    return np.maximum(np.round(t1 * np.maximum(v, 1e-3) ** (-(alpha - 1) / (alpha + 1)) * sr), 2).astype(np.int64)


def _tail(body, cap: float = 4.0) -> float:
    return float(np.clip(body.t60(), 0.12, cap))


# ==========================================================================================
# Cavity (Sounding Object §6.1 eq. 6.1-6.5; g04 §2.1)
# ==========================================================================================

_SABINE_F = np.array([0.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 11025.0])
_SABINE_A = np.array([0.19, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 1.00])     # "smooth wood-like enclosure"
C_AIR = 350.0     # reproduces the book's 463.8 Hz lowest partial of a d = 0.5 m sphere


@lru_cache(maxsize=32)
def cavity_modes(shape: str = "box", size: float = 0.5, wall: float = 1.0, fmax: float = 5000.0,
                 per_band: int = 10, seed: int = 0, gamma: float | None = None):
    """``(freqs, decays, gains)`` of a hollow box (cube of edge ``size``) or the sphere of equal
    volume (pitch follows volume, §6.1).

    Box: ``f(l,m,n) = (c/2) sqrt((l/X)^2 + (m/Y)^2 + (n/Z)^2)`` (eq. 6.1). Sphere:
    ``f(n,s) = c z_ns/(2 pi a)``, z_ns the roots of the derivative of the spherical Bessel
    function j_n (eq. 6.2). Decay from Sabine, ``T60 = 0.163 V/(alpha(f) A)`` (eq. 6.3), with the
    wood-like absorption table (eq. 6.4-6.5) scaled by ``wall`` (< 1 = harder walls). Mode
    amplitudes are random and the f^2 growth of modal density is stopped at ``per_band`` modes
    per third octave ("where single modes can no longer be discriminated").
    ``gamma``: superquadric |x|^g + |y|^g + |z|^g = 1 between the sphere (g = 2) and the box (g -> inf): the sorted
    modes of both are interpolated in log frequency (decays linearly) with weight w = 2 / g on the sphere.
    """
    if gamma is not None:
        w = float(np.clip(2.0 / max(float(gamma), 2.0), 0.0, 1.0))
        fb, db, gb = cavity_modes("box", size, wall, fmax, per_band, seed)
        fs, ds, _ = cavity_modes("sphere", size, wall, fmax, per_band, seed)
        k = min(len(fb), len(fs))
        f = np.exp((1 - w) * np.log(fb[:k]) + w * np.log(fs[:k]))
        return f, (1 - w) * db[:k] + w * ds[:k], gb[:k]
    size = float(np.clip(size, 0.03, 20.0))
    if shape == "sphere":
        a = size * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        vol, area = 4.0 / 3.0 * math.pi * a ** 3, 4.0 * math.pi * a * a
        zmax = TWO_PI * a * fmax / C_AIR
        grid = np.arange(0.3, zmax + 0.05, 0.02)
        z = []
        for nn in range(0, int(zmax) + 3):
            v = special.spherical_jn(nn, grid, derivative=True)
            idx = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
            z += [brentq(lambda q: special.spherical_jn(nn, q, derivative=True), grid[i], grid[i + 1]) for i in idx]
        f = np.sort(np.array(z)) * C_AIR / (TWO_PI * a)
    elif shape == "box":
        vol, area = size ** 3, 6.0 * size * size
        top = int(2.0 * size * fmax / C_AIR) + 1
        g = np.arange(0, top + 1)
        q = (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2).ravel()
        f = C_AIR / (2.0 * size) * np.sqrt(np.unique(q[q > 0]).astype(float))
    else:
        raise ValueError("cavity shape must be 'box' or 'sphere'")
    f = f[f <= fmax]
    rng = np.random.default_rng(seed)
    band = np.floor(3.0 * np.log2(f / f[0]) + 1e-9).astype(int)
    keep = np.concatenate([rng.permutation(np.nonzero(band == b)[0])[:per_band] for b in np.unique(band)])
    f = np.sort(f[keep])
    t60 = 0.163 * vol / (np.interp(f, _SABINE_F, _SABINE_A) * max(float(wall), 0.02) * area)
    gains = rng.uniform(0.3, 1.0, len(f)) * rng.choice([-1.0, 1.0], len(f))
    gains[0] = 1.0
    return f, 6.91 / t60, gains


def _cavity(x, sr, shape="box", size=0.5, wall=1.0, seed=0):
    f, d, g = cavity_modes(shape, round(float(size), 4), round(float(wall), 4), min(5000.0, 0.4 * sr), 10, int(seed))
    body = ModalObject(f, d, np.ones_like(f), aw0=g * TWO_PI * f / (TWO_PI * f[0]))    # equal-amplitude sines
    return modal_bank(x, body, sr, bright=0.0)[0]


def _wet(x, wet, mix):
    """Wet signal level-matched (RMS) to the dry one, then mixed."""
    wet = wet * (np.sqrt(np.mean(x * x)) / (np.sqrt(np.mean(wet * wet)) + 1e-30))
    mix = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - mix) * x + mix * wet


def _fx_apply(x, sr, tail, fn, mix):
    x = np.asarray(x, dtype=np.float64)
    pad = samples(max(float(tail), 0.0), sr) if tail and tail > 0 else 0

    def one(ch):
        ch = np.concatenate([ch, np.zeros(pad)])
        if not np.any(ch):
            return ch
        y = _wet(ch, np.nan_to_num(fn(np.ascontiguousarray(ch))), mix)
        if pad:
            y[-min(pad, samples(0.01, sr)):] *= np.linspace(1.0, 0.0, min(pad, samples(0.01, sr)))
        return y
    return one(x) if x.ndim == 1 else np.stack([one(x[:, c]) for c in range(x.shape[1])], axis=1)


@effect("resonate", "Re-body: the signal is the force driving a modal body (plate, bar, membrane or a measured object).",
        body=("rect_plate", "string|bar|bar_clamped|bar_supported|rect_membrane|rect_plate|circ_membrane|circ_plate|circ_plate_free|l_membrane, or a modal table (sy_vase, sy_wok, churchBell...)"),
        material=("steel", "rubber|plastic|wood|glass|ceramic|metal|steel|aluminium|bronze|ice|stone|concrete: sets ring time (and pitch with size)"),
        size=(0.4, "m: length / diameter of the body; bigger = lower"),
        position=(0.3, "0..1 where the force enters (bars: 0 end .. 0.5 middle; circles: 0 centre .. 1 rim)"),
        mix=(1.0, "0..1 wet"), bright=(0.5, "0 dark (displacement pickup) .. 0.5 velocity .. 1 bright (acceleration)"),
        modes=(32, "number of modes (5-10 already reads as an object)"),
        damping=(0.0, "extra external damping 1/s (0 free ... 30 held in the hand)"),
        tail=(1.0, "s of ring-out appended"))
def resonate(x, sr=DEFAULT_SR, body="rect_plate", material="steel", size=0.4, position=0.3, mix=1.0, bright=0.5,
             modes=32, damping=0.0, tail=1.0):
    """Any signal as the force on a modal body ("audio signal processor", g06 B6/B11)."""
    b = _make_body(body, material, size, position, modes=modes, damping=damping, fmax=min(12000.0, 0.45 * sr))
    return _fx_apply(x, sr, tail, lambda ch: modal_bank(ch, b, sr, bright)[0], mix)


@effect("cavity", "Hollow box / sphere resonance: the sound of being inside a small enclosure (Sabine decay).",
        shape=("box", "box|sphere (sphere of the same volume: a little brighter)"),
        size=(0.5, "m: edge of the cube (0.3-1 m); pitch follows the volume"),
        wall=(1.0, "wall absorption x: 1 wood, < 1 harder (marble) = longer ring, > 1 padded"),
        mix=(0.6, "0..1 wet"), seed=(0, "which modes are excited"), tail=(0.5, "s of ring-out appended"))
def cavity(x, sr=DEFAULT_SR, shape="box", size=0.5, wall=1.0, mix=0.6, seed=0, tail=0.5):
    return _fx_apply(x, sr, tail, lambda ch: _cavity(ch, sr, shape, size, wall, seed), mix)


# ==========================================================================================
# Registered sounds
# ==========================================================================================

_MAT_HELP = "rubber|plastic|plastic2|polycarbonate|polystyrene|mdf|wood|wood:<species> (spruce, maple, padauk, ebony...)|glass|ceramic|metal|steel|cast_iron|carbide|aluminium|bronze|copper|acrylic|ice|stone|concrete"
_SHAPE_HELP = "string|bar|bar_clamped|bar_supported|rect_membrane|rect_plate|circ_membrane|circ_plate|circ_plate_free|l_membrane|ri_<object> (measured: ceramic_bowl, glass, wood_plate, iron_pan, plastic_bowl, steel_pot, polycarbonate_cup)"


@sfx("impact", "One object struck by another: analytic modal body + Hunt-Crossley contact (harder / faster = shorter contact = brighter).",
     material=("wood", _MAT_HELP + ": ring time, pitch and contact hardness of the struck body"),
     shape=("bar", _SHAPE_HELP + " (or a modal table name such as sy_vase)"),
     size=(0.3, "m: length / diameter of the struck body; bigger = lower"),
     position=(0.28, "0..1 strike point (bars: 0 end .. 0.5 middle; circles: 0 centre .. 1 rim); near an edge = brighter"),
     position2=(0.37, "0..1 second coordinate of the strike point on rectangles"),
     striker_mass=(0.02, "kg: heavier = louder, longer contact, darker"),
     hardness=(0.6, "0..1 striker hardness, rubber mallet .. steel ball (Hertz stiffness from both materials and the striker size)"),
     velocity=(1.5, "m/s strike speed: louder and brighter"),
     hollow=(0.0, "0..1 share of a hollow air-cavity resonance of the same size"),
     damping=(2.0, "external damping 1/s: 0 freely hanging .. 30 lying on a table / held"),
     bright=(0.5, "0 dark .. 1 bright radiation"), freq=(0.0, "Hz: lowest mode, 0 = from size and material"),
     modes=(24, "number of modes"), dur=(0.0, "s, 0 = until it has rung out (max 4 s)"), seed=(0, "variation of the mode decays"),
     law=("friction", "mode decay law: friction (default) | rayleigh (Ren) | objectfolder (free objects, glass rings 2.8 s at 1 kHz; takes that table's "
                      "density and E too) | power, sterling_rayleigh, caughey, hybrid (Sterling, fitted to recordings) | obrien | zheng; "
                      "all but the first two exist for some materials only, and `damping` is added on top"),
     thickness=(0.0, "m: of the bar or plate; 0 = in proportion to size (a bar then goes 1/L, a fixed thickness 1/L^2)"),
     tip=("ball", "shape of the striker: ball (Hertz) | cone (contact grows faster, sharper at high speed) | flat (contact time independent of speed)"),
     tip_size=(0.0, "m: flat face radius; 0 = the striker's own radius"), tip_angle=(60.0, "degrees: cone semi-angle"),
     mount=(0.0, "1/s: a free bar hung at its nodes 22.4 % from the ends, scale of the suspension loss: mode 1 keeps ringing, the others lose it"),
     rest_force=(0.0, "N: resting on a support, each mode damped by what it does at the contact point (ceramic, polystyrene, steel, mdf, wood)"),
     ice_temp=(1.0, "degrees C, for material ice below 0: decays from the proton relaxation of ice at that temperature, the lows ring longer in the cold"),
     dilate=(0.0, "0 = off; the SG of a spectral dilation f -> SG f sqrt(1 + SR (f/f1)^2), metal 0.5, wood 0.85, glass 2.4"),
     dilate_r=(0.0, "the SR of that dilation: metal 0.1, wood 0.05, glass 0.2"),
     material_angle=(-1.0, "degrees on Aramaki's perceived-material disk, overriding the decays: 0 glass, 120 metal, 240 wood, "
                           "in between = a continuous blend (-1 = off)"),
     material_radius=(1.0, "0..1 distance from the disk centre (0 = the average of glass, metal and wood)"),
     radiation=("sphere", "sphere: every mode radiates as a pulsating body | modes: only each mode's net volume flow is a monopole, "
                          "the rest a dipole, weaker by ka / sqrt 3 while the body is small against the wavelength"),
     moisture=(0.0, "0..1 wood only: water content from 6 to 12 %, E/rho down 6 % and the loss up 24 % at 1"),
     azimuth=(-1.0, "degrees: where the listener stands around the object; each mode's level follows its lobes (-1 = averaged over all directions)"),
     rough=(0.0, "0..1 perceived roughness: every mode amplitude-modulated at a quarter of its critical band (the most dissonant rate), "
                 "this the modulation index"),
     freq_factor=(1.0, "x on every mode frequency: 0.7 with a heavier striker reads as a bigger object, 4 as a small ceramic one"),
     decay_factor=(1.0, "x on every decay rate (the ring time divides by it): 50 turns a ringing metal into a wood-like knock"),
     uneven=(0.0, "0..1 uneven decay, as real wood rings: a slow random swell on the tail and the highs dying sooner (0.5 for wood; metal decays smoothly)"),
     accel=(0.0, "0..1 acceleration noise of the striker: the click of its velocity reversal over the Hertz contact time (a steel ball's modes are above hearing, this is what is heard of it)"))
def impact(sr=DEFAULT_SR, material="wood", shape="bar", size=0.3, position=0.28, position2=0.37, striker_mass=0.02,
           hardness=0.6, velocity=1.5, hollow=0.0, damping=2.0, bright=0.5, freq=0.0, modes=24, dur=0.0, seed=0, law="friction",
           material_angle=-1.0, material_radius=1.0, thickness=0.0, tip="ball", tip_size=0.0, tip_angle=60.0, mount=0.0, rest_force=0.0,
           ice_temp=1.0, dilate=0.0, dilate_r=0.0, radiation="sphere", moisture=0.0, azimuth=-1.0, rough=0.0,
           freq_factor=1.0, decay_factor=1.0, uneven=0.0, accel=0.0):
    mat = get_material(material)
    body = _make_body(shape, mat, size, position, position2, freq=freq, modes=modes, damping=damping, seed=seed,
                      fmax=min(12000.0, 0.45 * sr), law=str(law), thickness=float(thickness) if thickness and thickness > 0 else None,
                      rest_force=float(rest_force))
    if radiation == "modes" and isinstance(body, ShapeBody):
        body.multipole = body.monopole_share()
    if azimuth is not None and float(azimuth) >= 0:
        body.aw0 = body.aw0 * directivity_gains(len(body.freqs), float(azimuth), seed)
    if mat.name == "ice" and float(ice_temp) < 0:
        body.decays = math.pi * body.freqs * ice_loss(body.freqs, max(float(ice_temp), -100.0)) + mat.ext + max(float(damping), 0.0)
    if mount > 0 and isinstance(body, ShapeBody) and body.shape == "bar" and body.boundary == "free":
        # theory 0.2242 L (Rossing; lot B38): the loss goes as the shape there squared, none for mode 1. the scale
        body.decays = body.decays + float(mount) * _bar_shape(body._desc["b"], 0.2242, "free") ** 2
    if dilate > 0:
        body.freqs = np.minimum(spectral_dilation(body.freqs, dilate, dilate_r), 0.44 * sr)
    if material_angle is not None and float(material_angle) >= 0:
        body.decays = aramaki_damping(body.freqs, float(material_angle), float(material_radius)) + max(float(damping), 0.0)
    if moisture > 0 and mat.name == "wood":
        w = float(np.clip(moisture, 0.0, 1.0))
        body.freqs = body.freqs * math.sqrt(1.0 - 0.06 * w)
        body.decays = body.decays * (1.0 + 0.24 * w)
    velocity = float(np.clip(velocity, 0.01, 300.0))             # guards; the sound needed 250 kg at 66 m/s (lot L03)
    hardness = float(np.clip(hardness, 0.0, 1.0))
    m_s = float(np.clip(striker_mass, 1e-5, 2000.0))
    k = _striker_k(mat, hardness, m_s)
    alpha = 1.5
    if tip != "ball":                                        # E* of the same striker: k_ball = 4/3 E* sqrt(R), with R = R(m_s)
        h_rho = 1100.0 * (7850.0 / 1100.0) ** hardness
        r = (3.0 * m_s / (4.0 * math.pi * h_rho)) ** (1.0 / 3.0)
        es = 0.75 * k / math.sqrt(r)
        if tip == "cone":
            k, alpha = cone_k(es, float(np.clip(tip_angle, 5.0, 85.0))), 2.0
        elif tip == "flat":
            k, alpha = 2.0 * es * (float(tip_size) if tip_size and tip_size > 0 else r), 1.0       # P = 2 E* a d (Johnson eq. 5.1)
        else:
            raise ValueError("tip must be ball, cone or flat")
    if freq_factor != 1.0:
        body.freqs = np.minimum(body.freqs * float(np.clip(freq_factor, 0.05, 20.0)), 0.44 * sr)
    if decay_factor != 1.0:
        body.decays = body.decays * float(np.clip(decay_factor, 0.01, 100.0))
    if rough > 0:
        body = am_sidebands(body, float(np.clip(rough, 0.0, 1.0)), 0.44 * sr)
    it = ImpactInteractor(body, mass=m_s, k=k, alpha=alpha, mu=max(mat.mu, 0.05 + 0.55 * (1.0 - hardness)), bright=bright)
    T = float(np.clip(dur, 0.03, 12.0)) if dur and dur > 0 else _tail(body) + 0.02
    y = it.strike(velocity, T, sr).audio
    if accel > 0:
        h_rho = 1100.0 * (7850.0 / 1100.0) ** hardness
        r_s = (3.0 * m_s / (4.0 * math.pi * h_rho)) ** (1.0 / 3.0)
        es = 0.75 * _striker_k(mat, hardness, m_s) / math.sqrt(r_s)
        tau = 2.87 * (m_s * m_s / (r_s * es * es * velocity)) ** 0.2
        a = acceleration_noise(tau, velocity * (1.0 + material_restitution(mat.name)), r_s, sr)
        a = a[:len(y)] / (np.max(np.abs(a)) + 1e-30)
        y[:len(a)] += float(np.clip(accel, 0.0, 1.0)) * float(np.max(np.abs(y))) * a
    if hollow > 0:
        h = float(np.clip(hollow, 0.0, 1.0))
        y = _wet(y, _cavity(y, sr, "box", max(0.6 * float(size), 0.05), 1.0, seed), h)
    if uneven > 0:
        y = uneven_decay(y, sr, float(np.clip(uneven, 0.0, 1.0)), seed)
    return _out(y, sr, 0.85 * float(np.clip((velocity / 3.0) ** 0.5, 0.25, 1.0)))


UNEVEN_RATE = (2.0, 8.0)      # Hz band of the slow random swell on a wooden tail
UNEVEN_HIGH = (2000.0, 25.0)  # Hz above which the tail loses an extra decay rate (1/s) at full `uneven`


def uneven_decay(y, sr, amount, seed=0):
    """Non-smooth decay of wood: the tail swells and sags at random 2-8 Hz (depth ``amount``, growing in 20 ms), and the
    band above 2 kHz loses 25 ``amount`` /s more."""
    rng = np.random.default_rng(int(seed) + 4242)
    n = len(y)
    t = np.arange(n) / sr
    lf = F.bandpass(rng.standard_normal(n), math.sqrt(UNEVEN_RATE[0] * UNEVEN_RATE[1]), sr, 0.7)
    lf = lf / (np.max(np.abs(lf)) + 1e-12)
    am = 1.0 + amount * 0.6 * lf * (1.0 - np.exp(-t / 0.02))
    lo = F.lowpass(y, UNEVEN_HIGH[0], sr, order=2)
    return (lo + (y - lo) * np.exp(-amount * UNEVEN_HIGH[1] * t)) * am      # the two bands sum back to y


@sfx("bounce", "An object dropped on a surface bouncing to rest: geometric bounce train, each hit a physical impact on the still-ringing bodies.",
     material=("steel", _MAT_HELP + ": the bouncing object"),
     size=(0.02, "m: diameter / length of the object; heavier = longer contacts, darker"),
     shape=("ball", "ball (rigid: only the surface rings) | bar | circ_plate (lid, coin) | rect_plate ..."),
     surface=("wood", _MAT_HELP + ": what it lands on (dominates the heard material)"),
     surface_size=(0.6, "m: size of the surface plate; bigger = lower"),
     first_interval=(0.3, "s between the first two impacts (= drop height: h = g t^2/8)"),
     restitution=(0.74, "ratio of successive intervals and speeds, 0.3 dead .. 0.9 lively (0.70-0.76 steel ball on wood); 0 = the typical value of the material"),
     irregularity=(0.0, "0..1 non-roundness: intervals and hits randomly shorter / weaker, strike point changes per bounce"),
     velocity=(0.0, "m/s first impact speed; 0 = from the interval, g t/(2 restitution)"),
     yield_stress=(0.0, "Pa: dynamic yield of the softer body (lead 1e8 .. hard steel 1e9), 0 = elastic: a hard hit dents and bounces less, e falling as v^-1/4"),
     restrike_jitter=(0.0, "share of the mean modal power thrown on every mode at each later hit; 0.2 is the measured value"),
     hit_spread=(0.0, "0..1 Lloyd's per-hit mode gains, wider the harder the hit"),
     bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def bounce(sr=DEFAULT_SR, material="steel", size=0.02, shape="ball", surface="wood", surface_size=0.6,
           first_interval=0.3, restitution=0.74, irregularity=0.0, velocity=0.0, yield_stress=0.0, restrike_jitter=0.0, hit_spread=0.0,
           bright=0.5, seed=0):
    y, _ = _bounce_render(sr, material, size, shape, surface, surface_size, first_interval, restitution,
                          irregularity, velocity, bright, seed, yield_stress=yield_stress, jitter=restrike_jitter, spread=hit_spread)
    return _out(y, sr, 0.85)


def _bounce_bodies(sr, mat, size, shape, surface, surface_size, seed):
    smat = get_material(surface)
    bodies = [_make_body("rect_plate", smat, surface_size, 0.37, 0.41, modes=28, damping=6.0, seed=seed,
                         fmax=min(10000.0, 0.45 * sr))]
    if shape != "ball":
        ob = _make_body(shape, mat, size, 0.3, 0.37, modes=16, damping=4.0, seed=seed + 1, fmax=min(12000.0, 0.45 * sr))
        if ob.freqs.min() < 0.4 * sr:
            bodies.append(ob)
        mass = ob.mass
    else:
        mass = _ball_mass(mat, size)
    return bodies, float(np.clip(mass, 1e-5, 50.0)), smat


def _restrike(rows, jitter, spread, v_rel, rng):
    """Each later hit: mode powers + N(0, jitter x their mean) (Traer et al. 2019 sec. 2.3: 20 %), and Lloyd's gains."""
    out = rows.copy()
    for r in range(1, len(out)):
        p = out[r] ** 2
        if jitter > 0:
            p = np.maximum(p + rng.normal(0.0, float(jitter) * p.mean(), p.shape), 0.0)
        g = lloyd_gain(float(spread) * v_rel[min(r, len(v_rel) - 1)], rng.random(p.shape)) if spread > 0 else 1.0
        out[r] = np.sign(out[r]) * np.sqrt(p) * g
    return out


def _rows(bodies, n_rows, irregularity, rng):
    """Mode-shape rows for successive hits: the object turns, so its strike point (and a little
    the landing point) changes per bounce (g05 §1: "of strong perceptual significance")."""
    rows = []
    for r in range(n_rows):
        parts = []
        for b in bodies:
            if r == 0 or irregularity <= 0 or not isinstance(b, ShapeBody):
                parts.append(b.phi)
            elif b.shape.startswith("rect"):
                j = 0.08 * irregularity
                parts.append(b.row((float(np.clip(b.position[0] + rng.uniform(-j, j), 0.05, 0.95)),
                                    float(np.clip(b.position[1] + rng.uniform(-j, j), 0.05, 0.95)))))
            else:
                parts.append(b.row(float(np.clip(b.position + irregularity * rng.uniform(-0.3, 0.6), 0.02, 0.98))))
        rows.append(np.concatenate(parts))
    return np.array(rows)


def _bounce_render(sr, mat, size, shape, surface, surface_size, first_interval, restitution, irregularity,
                   velocity, bright, seed, extra=0.0, stage=None, yield_stress=0.0, jitter=0.0, spread=0.0):
    mat = get_material(mat)
    rng = np.random.default_rng(seed)
    irr = float(np.clip(irregularity, 0.0, 1.0))
    e = float(np.clip(restitution if restitution > 0 else material_restitution(mat.name), 0.05, 0.97))
    t0 = float(np.clip(first_interval, 0.01, 3.0))
    v0 = float(velocity) if velocity and velocity > 0 else G * t0 / (2.0 * e)
    bodies, mass, smat = _bounce_bodies(sr, mat, float(np.clip(size, 0.002, 2.0)), shape, surface, surface_size, seed)
    # Hertz, sphere on a flat (Johnson eqs 4.17-4.24). for an object that is not a ball: half its size as the radius
    radius = 0.5 * float(np.clip(size, 0.002, 2.0))
    of_v = None
    if yield_stress and yield_stress > 0:                    # events relaunch the striker: mu does not count this loss twice
        es = e_star(mat, smat)
        of_v = lambda u: plastic_restitution(mass, u, radius, float(yield_stress), es)
    t, v = bounce_schedule(t0, v0, e, dev_t=0.6 * irr, dev_v=0.6 * irr, t_min=0.012, seed=seed, e_of_v=of_v)
    many = irr > 0 or jitter > 0 or spread > 0
    rows = _rows(bodies, min(len(t), 12) if many else 1, irr, rng)
    if jitter > 0 or spread > 0:
        rows = _restrike(rows, jitter, spread, v / max(v0, 1e-9), rng)
    pr = np.arange(len(t)) % len(rows)
    pm = mass * (1.0 - 0.5 * irr * rng.random(len(t)))          # effective mass varies per impact (g05 §1)
    it = ImpactInteractor(bodies, mass=mass, k=_hertz_pair(mat, smat, radius), mu=max(mat.mu, smat.mu), bright=bright)
    T = float(t[-1]) + extra + min(max(_tail(b, 2.5) for b in bodies), 2.5) + 0.05
    kw = stage(it, T, t[-1], mass) if stage else {}
    if stage:
        t, v, pr, pm = (np.append(t, t[-1] + 0.012), np.append(v, 0.02), np.append(pr, 0), np.append(pm, mass))
    return it.run(T, sr, events=(t, v, pr, pm), rows=rows, **kw).audio, t


def _roll_stage(n, fs, speed, radius, mass, roughness, joints, dents, asymmetry, seed, patch_a=0.0):
    """Offset curve and normal load of a rolling ball (Sounding Object §9.4; g05 §3): band-limited
    surface -> rolling filter -> read at the ball's position (all surface frequencies scale with
    speed, amplitude constant); a dent recurs once per revolution (f = v/(2 pi r)); an eccentric ball (centre of
    mass r = 0.1 asymmetry R off its centre) rolls with its contact at x = R theta - r sin(theta), its centre of mass
    rises and falls by r cos(theta) (an offset the contact sees). The external load stays m g: following that offset, the
    ball's own inertia makes the contact force m (g + r theta'^2 cos(theta)); adding it to the load would count it twice."""
    asym = float(np.clip(asymmetry, 0.0, 1.0))
    ecc = 0.1 * asym * radius
    theta = np.cumsum(speed) / (fs * radius)
    pos = radius * theta - ecc * np.sin(theta)
    dx = 1e-4
    off = rolling_offset(rolling_surface(float(pos[-1]) + 0.05, dx, roughness, joints=joints, seed=seed), radius, dx)
    if patch_a > 0:
        off = patch_average(off, patch_a, dx)
    off = np.interp(pos / dx, np.arange(len(off)), off, period=len(off))
    if dents > 0:
        w = 0.25                                             # dent 0.25 rad wide, up to 50 um deep
        th = (theta + math.pi) % TWO_PI - math.pi
        off = off - float(np.clip(dents, 0, 1)) * 5e-5 * np.where(np.abs(th) < w, np.cos(0.5 * math.pi * th / w) ** 2, 0.0)
    if ecc > 0:
        off = off + ecc * (1.0 - np.cos(theta))
    return off, np.full(len(theta), mass * G)


@sfx("drop", "A ball dropped on a surface: bounces until it stays in contact, then rolls away and comes to rest.",
     material=("steel", _MAT_HELP + ": the ball"), size=(0.03, "m: ball diameter"),
     surface=("wood", _MAT_HELP + ": the floor / table"), surface_size=(0.8, "m: size of the surface plate"),
     height=(0.25, "m drop height (sets impact speed and first bounce interval)"),
     restitution=(0.6, "bounce interval ratio 0.3..0.9; 0 = the typical value of the ball material"), roll=(1.5, "s of rolling after the last bounce (0 = none)"),
     roll_speed=(0.4, "m/s rolling speed right after the bounces; slows to rest"),
     roughness=(0.5, "0..1 surface roughness while rolling"), irregularity=(0.0, "0..1 uneven bounces"),
     bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def drop(sr=DEFAULT_SR, material="steel", size=0.03, surface="wood", surface_size=0.8, height=0.25, restitution=0.6,
         roll=1.5, roll_speed=0.4, roughness=0.5, irregularity=0.0, bright=0.5, seed=0):
    e = float(np.clip(restitution if restitution > 0 else material_restitution(material), 0.05, 0.95))
    v0 = math.sqrt(2.0 * G * float(np.clip(height, 0.005, 20.0)))        # v = sqrt(2 g h), t = 2 e v/g
    roll = float(np.clip(roll, 0.0, 20.0))
    radius = 0.5 * float(np.clip(size, 0.002, 2.0))

    def stage(it, T, t_end, mass):
        if roll <= 0:
            return {}
        os_ = _oversample_for(it.contact_time(v0), sr)
        fs, n = sr * os_, samples(T, sr) * os_
        i0 = int((t_end + 0.012) * fs)
        tt = np.clip((np.arange(n) - i0) / (roll * fs), 0.0, 1.0)
        speed = np.where(np.arange(n) >= i0, float(np.clip(roll_speed, 0.0, 20.0)) * (1.0 - tt) ** 1.5, 0.0)
        off, load = _roll_stage(n, fs, speed, radius, mass, roughness, 0.0, 0.0, 0.3 * irregularity, seed)
        off = np.where(np.arange(n) >= i0, off - off[min(i0, n - 1)], 0.0)
        load = np.where(np.arange(n) >= i0, load, 0.0)
        return {"offset": off, "load": load, "oversample": os_}
    y, _ = _bounce_render(sr, material, size, "ball", surface, surface_size, 2.0 * e * v0 / G, e, irregularity, v0,
                          bright, seed, extra=roll, stage=stage)
    return _out(y, sr, 0.85)


@sfx("roll", "Ball rolling on a surface: surface profile -> rolling filter (ball radius) -> micro-impacts with the ball's inertia -> surface modes.",
     dur=(1.0, "s"), speed=(0.5, "m/s; a number, or a list of breakpoints = speed curve over the sound (e.g. [1.2, 0.6, 0])"),
     radius=(0.03, "m ball radius: bigger = smoother, lower, heavier (small balls sound least real)"),
     roughness=(0.5, "0..1 microscopic surface roughness"),
     material=("steel", _MAT_HELP + ": the ball"), surface=("wood", _MAT_HELP + ": the surface (dominates the timbre)"),
     surface_size=(1.0, "m: size of the surface plate"),
     joints=(0.0, "m: spacing of floor joints / grooves the ball crosses (0 = none)"),
     dents=(0.0, "0..1 a dent on the ball, heard once per revolution"),
     asymmetry=(0.0, "0..1 off-centre centre of mass, 0.1 x asymmetry x radius: contact point, height and load swing once per turn"),
     feedback=(0.18, "legacy, ignored: the contact is now physically two-way coupled"),
     body=("", "'' a solid ball, only the surface rings | can: an empty can with its four measured modes"),
     patch=(0.0, "0 = off; 1 = the surface is felt as its mean height over the Hertz contact patch (half-length "
                 "x patch), which dulls wavelengths shorter than the patch: a low-pass at speed / (2 a)"),
     bright=(0.4, "0 dark .. 1 bright"), seed=(0, "variation"))
def roll(sr=DEFAULT_SR, dur=1.0, speed=0.5, radius=0.03, roughness=0.5, material="steel", surface="wood",
         surface_size=1.0, joints=0.0, dents=0.0, asymmetry=0.0, feedback=0.18, body="", patch=0.0, bright=0.4, seed=0):
    mat, smat = get_material(material), get_material(surface)
    radius = float(np.clip(radius, 0.002, 1.0))
    dur = float(np.clip(dur, 0.03, 120.0))
    mass = 0.015 if body == "can" else float(np.clip(_ball_mass(mat, 2.0 * radius), 1e-5, 500.0))   # 15 g empty can
    k = _hertz_pair(mat, smat, radius)
    plate = _make_body("rect_plate", smat, surface_size, 0.37, 0.41, modes=28, damping=6.0, seed=seed,
                      fmax=min(8000.0, 0.45 * sr))
    os_ = 2
    fs, n = sr * os_, samples(dur, sr) * os_
    v = np.clip(_curve(speed, n), 0.0, 50.0)
    x_static = (mass * G / k) ** (1.0 / 1.5)                  # start resting on the surface
    patch_a = float(np.clip(patch, 0.0, 10.0)) * math.sqrt(radius * x_static)     # Hertz a^2 = R d
    off, load = _roll_stage(n, fs, v, radius, mass, roughness, float(max(joints, 0.0)), dents, asymmetry, seed, patch_a)
    it = ImpactInteractor(plate, mass=mass, k=k, mu=max(mat.mu, smat.mu), bright=bright)
    if body not in ("", "can"):
        raise ValueError("roll body must be '' or can")
    res = it.run(dur, sr, y0=off[0] - x_static, active=True, offset=off, load=load, oversample=os_)
    if body == "can":
        ring = _can_ring(res.force[::os_][:samples(dur, sr)], v[::os_][:samples(dur, sr)], radius, mass, sr)
        return _out(ring / (np.max(np.abs(ring)) + 1e-30) + 0.1 * res.audio / (np.max(np.abs(res.audio)) + 1e-30), sr, 0.75, 15.0)   # floor at -20 dB
    return _out(res.audio, sr, 0.75, 15.0)


def _can_ring(force, v, radius, mass, sr):
    """The can's own modes driven by the contact force plus one pattern per turn, amplitude in sqrt(speed), clipped at 0.6
    (Farnell, Designing Sound, Practical 8, §17.20-17.24: 359, 426, 1748, 3150 Hz at 1, .513, .192; `designed.can_roll` for the rest)."""
    from .designed import CAN_LEVELS, CAN_MODES, roll_pattern     # designed imports this module
    rev = np.cumsum(v) / (sr * TWO_PI * radius)
    # 150 Hz high-pass as in can_roll; the fluctuation of the contact force, 0.05 rms under the pattern
    drive = F.highpass(roll_pattern(rev) + 0.05 * (force - force.mean()) / (force.std() + 1e-30), 150.0, sr) * np.sqrt(v / (TWO_PI * radius))
    f = np.array(CAN_MODES)
    body = ModalObject(f, np.full(4, 6.91 / 0.08), np.ones(4), aw0=np.array(CAN_LEVELS) * TWO_PI * f / (TWO_PI * f[0]))   # T60 80 ms
    return modal_bank(np.clip(drive, -0.6, 0.6), body, sr, 0.0)[0]


@sfx("scrape", "Scraping / sliding: a surface profile read at the sliding speed x normal force, driving any modal body.",
     dur=(0.8, "s"), speed=(0.3, "m/s; number or list of breakpoints (a moving profile is what makes it expressive)"),
     force=(1.0, "normal force 0..1; number or list of breakpoints"),
     roughness=(0.5, "0..1 (1 = white, very rough; 0 = smooth); used when texture is empty"),
     texture=("", "'' | white|pink_half|pink|brown|gritty|sandpaper|grid|wood|metal|plastic"),
     grain=(0.1, "mm: size of the surface grain; smaller or faster = brighter"),
     body=("rect_plate", _SHAPE_HELP + " or a modal table (sy_vase, sy_sword1...)"),
     material=("wood", _MAT_HELP + ": the scraped body"), size=(0.4, "m: size of the body"),
     position=(0.3, "0..1 where it is scraped; a list of breakpoints = the scraper moves along the body (each mode follows its shape there)"),
     variation=(0.0, "0..1 random change of each mode's level along the path (0.2 = measured spread of 20 % in power); 0 = none"),
     level=("force", "force: the profile x normal force (van den Doel), speed changes only the pitch of the grain | power: x sqrt(speed x force), silent at rest (van den Doel & Pai; friction power)"),
     model=("profile", "profile | curvature: the path of a scraper of real mass over the profile, its inertia gives the force (Agarwal et al. 2021) | "
                       "meso: one impulse per surface pixel crossed, the change of the scraper's normal momentum (a height map)"),
     scraper_mass=(0.02, "kg, model curvature"), height=(2e-5, "m rms of the profile, model curvature"),
     bright=(0.5, "0 dark .. 1 bright"),
     stroke=(0.0, "Hz of a back-and-forth stroke by hand (0 = off): speed and normal force follow the gesture, pressing hardest at the near end"),
     stroke_len=(0.1, "m, half the stroke"), stroke_angle=(30.0, "degrees of the held scraper from the surface"),
     mu=(0.6, "friction coefficient, stroke only"),
     gate=(0, "1 = the scraper leaves the surface where the profile falls away faster than force/mass can pull it down (fast scrapes chatter); uses height, scraper_mass, force in N"),
     seed=(0, "variation"))
def scrape(sr=DEFAULT_SR, dur=0.8, speed=0.3, force=1.0, roughness=0.5, texture="", grain=0.1, body="rect_plate",
           material="wood", size=0.4, position=0.3, level="force", model="profile", scraper_mass=0.02, height=2e-5, bright=0.5,
           stroke=0.0, stroke_len=0.1, stroke_angle=30.0, mu=0.6, gate=0, variation=0.0, seed=0):
    n = samples(float(np.clip(dur, 0.03, 120.0)), sr)
    moving = isinstance(position, (list, tuple, np.ndarray))
    b = _make_body(body, material, size, float(np.asarray(position, float).ravel()[0]) if moving else position, modes=28, damping=8.0,
                   seed=seed, fmax=min(10000.0, 0.45 * sr))
    prof = surface_texture(texture if texture else 2.0 * (1.0 - float(np.clip(roughness, 0.0, 1.0))), seed=seed)
    fade = np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(1.0, 0.005 * sr))
    v = np.clip(_curve(speed, n), 0.0, 50.0)
    fn = np.clip(_curve(force, n), 0.0, 10.0)
    if stroke > 0:
        w = TWO_PI * float(np.clip(stroke, 0.05, 20.0))
        t = np.arange(n) / sr
        L = float(np.clip(stroke_len, 0.005, 1.0))
        v = w * L * np.abs(np.sin(w * t))
        fn = fn * stroke_force(t, float(stroke), L, float(stroke_angle), float(mu), float(scraper_mass)) / (float(scraper_mass) * 9.81)
    dx = float(np.clip(grain, 0.005, 20.0)) * 1e-3
    if model == "curvature":
        f = curvature_force(prof * float(height), v, fn, dx, sr, float(scraper_mass)) * fade
    elif model == "profile":
        f = scrape_force(prof, v, (np.sqrt(v * fn) if level == "power" else fn) * fade, dx, sr)
    elif model == "meso":
        f = meso_force(prof * float(height), v, float(scraper_mass), dx, sr) * fade
    else:
        raise ValueError("scrape model must be profile, curvature or meso")
    if gate:
        f = f * contact_gate(prof * float(height), v, fn, dx, sr, float(scraper_mass))
    if moving or variation > 0:
        path = np.clip(_curve(list(position), n), 0.0, 1.0) if moving else None
        return _out(moving_bank(f, b, sr, path, bright, float(variation), seed), sr, 0.7, 15.0)
    return _out(modal_bank(f, b, sr, bright)[0], sr, 0.7, 15.0)


# ---------------------------------------------------------------- continuous interactions (Conan)
# Conan et al., "Navigating in a space of synthesized interaction-sounds: rubbing, scratching and rolling"
# (DAFx-13, eqs 3-9) and "A synthesis model with intuitive control capabilities for rolling sounds" (IEEE TASLP
# 22(8):1260, 2014, eqs 6-9, 16-17, Table I). Every action is a train of raised-cosine impacts whose amplitudes A
# and intervals dT come from one shared white process W through inverse CDFs and 1-pole 1-zero ARMA filters:
# rubbing = one impact per sample (white noise), scratching = sparse impacts with exponential intervals, rolling =
# strongly correlated Gaussian series whose contact time grows with the ball (t0 = 7.88e-4 S A^-0.29 s) plus an
# amplitude modulation at 3 V / S Hz. The three parameter sets are mixed linearly on Conan's circle (eqs 6-9),
# and the train is low-passed at a cutoff that follows the sliding speed (van den Doel) before driving the body.
_CONAN_ROLL = {  # Table I, roughness 0 -> 1: sigma, a1, b1, mean; dT in s
    "A": ((0.04, 0.04), (-0.97, -0.93), (0.07, 0.32), (0.43, 0.27)),
    "dT": ((0.19e-3, 0.85e-3), (-0.97, -0.93), (-0.34, 0.35), (3.1e-3, 6.4e-3)),
}
CONAN_SCRATCH_DT = 0.012
CONAN_RUB_SIGMA = 0.43


def conan_weights(angle: float = 0.0, radius: float = 1.0) -> np.ndarray:
    """(rub, scratch, roll) weights at ``angle`` degrees (0 rub, 120 scratch, 240 roll) and ``radius`` 0..1
    (0 = the centre, an equal mix): T(theta) of DAFx-13 eq. 7 around the rim, eq. 8 inwards."""
    def T(t):
        t = t % TWO_PI
        return 1.0 - 3.0 * t / TWO_PI if t <= TWO_PI / 3 else (3.0 * t / TWO_PI - 2.0 if t >= 2 * TWO_PI / 3 else 0.0)
    th = math.radians(float(angle))
    rim = np.array([T(th), T(th - TWO_PI / 3), T(th - 2 * TWO_PI / 3)])
    r = float(np.clip(radius, 0.0, 1.0))
    return (1.0 - r) / 3.0 + r * rim


@numba.njit(cache=True)
def _pulse_train(n, idx, amp, length, xi=1.0):
    """Raised-cosine impacts (TASLP eq. 9), each centred over its own length in samples; ``xi`` sharpens them,
    f = (F/2^xi)(1 - cos(2 pi n/N))^xi (Conan et al. DAFx-13 / Agarwal 2021 eqs 4-5; Chadwick et al. 2012)."""
    f = np.zeros(n)
    for e in range(len(idx)):
        L = length[e]
        s = idx[e] - L // 2
        for k in range(L):
            j = s + k
            if 0 <= j < n:
                f[j] += amp[e] * (0.5 * (1.0 + math.cos(TWO_PI * (k + 0.5 - 0.5 * L) / L))) ** xi
    return f


def conan_force(n: int, sr: int, weights, roughness: float = 0.5, size: float = 0.5, speed: float = 0.3,
                seed: int = 0, sharpness: float = 1.0, radius: float = 0.0) -> np.ndarray:
    """The interaction force (before the speed low-pass) for (rub, scratch, roll) ``weights``. roughness 0..1 is
    Table I's rho, size the normalised ball radius S (0.1..1), speed in m/s (number or per-sample) sets the rolling
    modulation: once per turn, v/(2 pi radius), when the ball's ``radius`` (m) is given, else 3 V/S."""
    w = np.asarray(weights, float) / max(float(np.sum(weights)), 1e-12)
    rho = float(np.clip(roughness, 0.0, 1.0))
    S = float(np.clip(size, 0.1, 1.0))
    roll = {k: [lo + rho * (hi - lo) for lo, hi in v] for k, v in _CONAN_ROLL.items()}
    sa, a1a, b1a, ma = roll["A"]
    sd, a1d, b1d, md = roll["dT"]
    mean_dt = w[0] / sr + w[1] * CONAN_SCRATCH_DT + w[2] * md
    m = int(n / sr / mean_dt * 1.3) + 64
    rng = np.random.default_rng(seed)
    W = np.clip(rng.random(m), 1e-9, 1 - 1e-9)
    g = special.ndtri(W)                                     # one W drives both series (TASLP Fig. 5)
    expo = -np.log1p(-W) - 1.0                               # centred unit exponential
    cA = lfilter([1.0, w[2] * b1a], [1.0, w[2] * a1a], (w[0] + w[1]) * CONAN_RUB_SIGMA * g + w[2] * sa * g)
    cD = lfilter([1.0, w[2] * b1d], [1.0, w[2] * a1d], w[1] * CONAN_SCRATCH_DT * expo + w[2] * sd * g)
    A = w[2] * ma + cA
    dT = np.maximum(mean_dt + cD, 1.0 / sr)
    t = np.cumsum(dT) - dT[0]
    keep = t < n / sr
    t, A = t[keep], A[keep]
    t0 = (w[0] + w[1]) / sr + w[2] * 7.88e-4 * S * np.abs(np.maximum(np.abs(A), 1e-3)) ** (-0.29 * w[2])
    L = np.maximum(np.round(t0 * sr), 1).astype(np.int64)
    f = _pulse_train(n, np.round(t * sr).astype(np.int64), A, L, float(sharpness))
    v = np.broadcast_to(np.asarray(speed, float), (n,))
    nu = v / (TWO_PI * radius) if radius > 0 else 3.0 * v / S
    return f * (1.0 + 0.3 * w[2] * np.sin(TWO_PI * np.cumsum(nu) / sr))


@sfx("interaction", "Rubbing, scratching or rolling, and anything between: one impact-train model (Conan) morphing on a circle, driving any body.",
     dur=(1.0, "s"), action=(0.0, "degrees on Conan's circle: 0 rub, 120 scratch, 240 roll; between = a blend"),
     blend=(1.0, "0..1 distance from the centre: 1 = the pure action, 0 = an equal mix of all three"),
     speed=(0.3, "m/s; number or list of breakpoints: brightness (low-pass) and the rolling wobble"),
     roughness=(0.5, "0 smooth .. 1 rough surface (Conan's Table I)"),
     ball=(0.5, "0.1..1 size of the rolling object: longer contacts, slower wobble"),
     body=("rect_plate", _SHAPE_HELP + " or a modal table (sy_vase, sy_sword1...)"),
     material=("wood", _MAT_HELP + ": the surface"), size=(0.4, "m: size of the surface"),
     position=(0.3, "0..1 where it is touched"), sharpness=(1.0, "1 raised cosine .. 4 narrower, clicking impacts"),
     prefilter=(0.0, "0..1 reinforce the force near the body's modes (what a two-way contact does; 0 = off)"),
     radius=(0.0, "m: the rolling ball's radius; its wobble is then once per turn, v/(2 pi radius) (0 = Conan's 3 V/ball)"),
     gamma=(1, "1 | 2: each mode rings as t exp(-t/tau) sin (a gammatone, softer onset) instead of exp(-t/tau) sin"),
     bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def interaction(sr=DEFAULT_SR, dur=1.0, action=0.0, blend=1.0, speed=0.3, roughness=0.5, ball=0.5, body="rect_plate",
                material="wood", size=0.4, position=0.3, sharpness=1.0, prefilter=0.0, radius=0.0, gamma=1, bright=0.5, seed=0):
    n = samples(float(np.clip(dur, 0.03, 120.0)), sr)
    v = np.clip(_curve(speed, n), 0.0, 50.0)
    f = conan_force(n, sr, conan_weights(action, blend), roughness, ball, v, seed, sharpness, float(max(radius, 0.0)))
    f = F.svf(f, "lowpass", np.clip(15000.0 * v, 150.0, 0.45 * sr), sr) * np.clip(v / 0.02, 0.0, 1.0)
    b = _make_body(body, material, size, position, modes=28, damping=8.0, seed=seed, fmax=min(10000.0, 0.45 * sr))
    f = resonant_prefilter(f, b, sr, prefilter)
    y = gammatone_bank(f, b, sr, bright) if int(gamma) == 2 else modal_bank(f, b, sr, bright)[0]
    return _out(y, sr, 0.7, 15.0)


@sfx("smash", "Something brittle breaking on a hard floor: impact + noise burst + dense, random, thinning cloud of fragment impacts.",
     material=("glass", "glass|ceramic|ice|wood|plastic|stone... (metal reads as 'not breaking')"),
     size=(0.25, "m: size of the object; bigger = lower, heavier fragments"),
     shape=("rect_plate", "shape of the object before it breaks: rect_plate|circ_plate|bar"),
     energy=(1.0, "0.2..2 how hard it hits: louder, more fragments flying longer"),
     fragments=(8, "number of fragments (independent impact trains)"),
     dur=(1.6, "s until the last fragment settles"), burst=(0.5, "0..1 level of the 50 ms rupture noise"),
     bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"),
     distance=(0.0, "m from the listener: farther fragments keep fewer modes, the most important first (0 = all modes)"))
def smash(sr=DEFAULT_SR, material="glass", size=0.25, shape="rect_plate", energy=1.0, fragments=8, dur=1.6,
          burst=0.5, bright=0.5, seed=0, distance=0.0):
    mat = get_material(material)
    rng = np.random.default_rng(seed)
    energy = float(np.clip(energy, 0.05, 4.0))
    dur = float(np.clip(dur, 0.1, 30.0))
    n = samples(dur + 0.4, sr)
    parent = _make_body(shape, mat, size, 0.31, 0.43, modes=20, damping=4.0, seed=seed, fmax=min(12000.0, 0.45 * sr))
    k = _hertz_pair(mat, "concrete", 0.5 * float(size))      # for a plate: half its size as the radius of what lands
    # 1. the object hits the floor (its own mass against a rigid floor: extreme mass ratio, g05 §2)
    it = ImpactInteractor(parent, mass=float(np.clip(parent.mass, 1e-4, 50.0)), k=k, mu=max(mat.mu, 0.2), bright=bright)
    t_break = 0.012                                         # the parent rings 12 ms, then is gone
    hit = it.strike(3.0 * energy ** 0.5, t_break + 0.02, sr).audio
    hit[-samples(0.02, sr):] *= np.linspace(1.0, 0.0, samples(0.02, sr)) ** 2
    out = np.zeros(n)
    out[:len(hit)] += hit / (np.max(np.abs(hit)) + 1e-30)
    # 2. 50 ms noise burst on the attack (Warren & Verbrugge; g04 §1.1, g05 §2)
    nb = samples(0.05, sr)
    nz = F.highpass(rng.standard_normal(nb), 900.0, sr) * np.exp(-np.arange(nb) / (0.012 * sr))
    nz = F.lowpass(nz, 3000.0 + 5000.0 * bright, sr) * np.hanning(2 * nb)[nb:]
    out[:nb] += float(np.clip(burst, 0.0, 1.0)) * 0.8 * nz / (np.max(np.abs(nz)) + 1e-30)
    # 3. fragments: one decelerating, highly random train each, with its own size (pitch) and damping
    nf = int(np.clip(fragments, 1, 60))
    t, v, tr = break_schedule(dur, first=0.004 / energy ** 0.5, growth=1.25, dev=0.8, velocity=energy ** 0.5,
                              vel_factor=0.85, trains=nf, seed=seed)
    frac = rng.uniform(0.2, 0.7, nf)
    f1 = np.minimum(parent.freqs.min() / frac ** 2, 3000.0 + 3000.0 * bright)    # plate: f ~ h/L^2
    ratio = parent.freqs / parent.freqs.min()
    dratio = material_damping(ratio, mat, ext=0.0, seed=seed) / max(mat.tan_phi * math.pi, 1e-9)   # d/f1 per mode
    g = np.array([parent.gains(rng.uniform(0.1, 0.9, 2) if isinstance(parent, ShapeBody) and parent.shape.startswith("rect")
                               else float(rng.uniform(0.05, 0.95)), bright) for _ in range(16)])
    g /= np.max(np.abs(g)) + 1e-30
    tc = np.empty(len(t), np.int64)
    for j in range(nf):
        sel = tr == j
        tc[sel] = _hertz_samples(parent.mass * frac[j] ** 2, _hertz_pair(mat, "concrete", 0.5 * float(size) * frac[j]), 1.5, max(mat.mu, 0.2), v[sel], sr)
    if distance and distance > 0:
        g = lod_gains(g, ratio, dratio, float(distance))
    cloud = _event_bank(n, float(sr), np.round((t + t_break) * sr).astype(np.int64), v * frac[tr],
                        f1[tr], f1[tr] * mat.tan_phi * math.pi, np.full(len(t), mat.ext + 25.0), tc,
                        rng.integers(0, len(g), len(t)), ratio, dratio, g)
    out += 0.9 * cloud / (np.max(np.abs(cloud)) + 1e-30)
    return _out(out, sr, 0.85)


_CRUMPLE = {
    # f_min, f_max (Hz), tau_min, tau_max (s, 1/e time of a facet), closing low-pass, events x, rate x,
    # gamma at energy 0, gamma slope with energy, lowest event energy (None = normalised floor)
    "can": (700.0, 3600.0, 0.004, 0.030, True, 1.0, 1.0, -1.15, -0.35, None),
    "bottle": (300.0, 2200.0, 0.006, 0.050, True, 1.0, 1.0, -1.15, -0.35, None),
    "paper": (1000.0, 1000.0, 0.001, 0.001, False, 6.0, 8.0, -1.30, -0.29, 1e-3),
    "foil": (1800.0, 6000.0, 0.002, 0.012, False, 4.0, 6.0, -1.15, -0.35, None),
    "bag": (900.0, 4200.0, 0.0006, 0.004, False, 6.0, 10.0, -1.01, -0.19, 1e-4),
}


def _power_mean(gamma: float, m: float) -> float:
    """Mean of P(E) ~ E^gamma on [m, 1]."""
    g1, g2 = gamma + 1.0, gamma + 2.0
    return (1.0 - m ** g2) / g2 * g1 / (1.0 - m ** g1)


@sfx("crumple", "Crumpling / crushing: stochastic impact bursts with power-law energies; facets shrink, so pitch drifts up as it is crushed.",
     kind=("can", "can|paper|bottle|foil|bag"),
     energy=(0.5, "0..1 force of the crushing: 0 soft and even (gamma -1.15) .. 1 hard, peaky (gamma -1.5)"),
     density=(0.5, "0..1 event density = softness of the material: stiff (1 event/s) .. soft (50 events/s)"),
     size=(0.5, "0..1 size of the object = energy budget (0.001 .. 0.1): how long it goes on"),
     dur=(0.0, "s cap, 0 = until the energy budget is spent"), bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def crumple(sr=DEFAULT_SR, kind="can", energy=0.5, density=0.5, size=0.5, dur=0.0, bright=0.5, seed=0):
    if kind not in _CRUMPLE:
        raise ValueError(f"unknown crumple kind {kind!r}; choose from {sorted(_CRUMPLE)}")
    fmin, fmax, tmin, tmax, close, nx, rx, g0, gs, e_min = _CRUMPLE[kind]
    ring = fmin == fmax
    fmax = min(fmax * (0.6 + 0.8 * float(np.clip(bright, 0, 1))), 0.4 * sr)
    fmin = fmax if ring else fmin
    rng = np.random.default_rng(seed)
    gamma = max(g0 + gs * float(np.clip(energy, 0.0, 1.0)), -3.0)
    rate = rx * 10 ** (math.log10(50.0) * float(np.clip(density, 0.0, 1.0)))
    e_tot = nx * 10 ** (-3.0 + 2.0 * float(np.clip(size, 0.0, 1.0)))
    e_unit = 7.5e-4
    if e_min is not None:
        g = float(np.clip(gamma, -3.0, -1.01))
        e_unit *= _power_mean(g, crumple_energy_floor(g)) / _power_mean(g, e_min)
    ev = crumple_events(e_tot, gamma, rate, e_unit=e_unit, e_min=e_min, seed=seed, n_max=6000)
    if dur and dur > 0:
        keep = ev["t"] < float(dur)
        ev = {k: a[keep] for k, a in ev.items()} if keep.any() else {k: a[:1] for k, a in ev.items()}
    t = np.repeat(ev["t"], 2) + 0.01
    span = ev["left"] + ev["right"]
    part = np.stack([ev["left"] / span, ev["right"] / span], axis=1).ravel()       # energy share (9.14)
    facet = np.stack([ev["left"], ev["right"]], axis=1).ravel()                    # length / D0
    f = fmax - facet * (fmax - fmin)                         # small facet = high pitch
    tau = tmin + facet * (tmax - tmin)                       # ... and short decay (prose of §9.5; g05 §4.2)
    amp = np.sqrt(np.repeat(ev["energy"], 2) * part)
    facet_body = ShapeBody("rect_plate", 100.0, aspect=1.3, n_modes=5, scatter=0.0, ext=0.0)
    fr = facet_body.freqs / facet_body.freqs[0]
    g = np.array([facet_body.gains(tuple(rng.uniform(0.15, 0.85, 2)), bright) for _ in range(16)])
    g /= np.max(np.abs(g)) + 1e-30
    n = samples(float(t[-1]) + 6.91 * tmax + 0.05, sr)
    y = _event_bank(n, float(sr), np.round(t * sr).astype(np.int64), amp, f, 1.0 / tau, np.zeros(len(t)),
                    np.maximum(np.round(sr * (0.00015 + 0.0005 * facet)), 2).astype(np.int64),   # contact length
                    rng.integers(0, len(g), len(t)), fr, fr, g)
    if close:                                                # closing first-order low-pass 1400 -> 500 Hz (9.15)
        cut = np.interp(np.arange(n) / sr, ev["t"] + 0.01, ev["cut"], left=1400.0)
        y = tv_one_pole_lp(y, np.exp(-TWO_PI * cut / sr))
    return _out(y, sr, 0.8)


def _rumble(sr, dur, fmin, fmax, damping, modes, rise, hold, fade, seed, prop):
    """Enveloped white noise into N random low modes (van den Doel thesis, avalanche; g06 B7)."""
    dur = float(np.clip(dur, 0.1, 120.0))
    n = samples(dur, sr)
    rng = np.random.default_rng(seed)
    lo = float(np.clip(min(fmin, fmax), 10.0, 0.4 * sr))
    hi = float(np.clip(max(fmin, fmax), lo + 1e-3, 0.4 * sr))
    f = np.sort(rng.uniform(lo, hi, int(np.clip(modes, 1, 200))))
    body = ModalObject(f, np.full(len(f), float(np.clip(damping, 0.02, 500.0))), np.ones(len(f)),
                       aw0=rng.uniform(0.3, 1.0, len(f)) * f / f[0])
    a, b, c = (dur * p if not (v and v > 0) else float(v) for v, p in zip((rise, hold, fade), prop))
    a = min(a, dur)
    b, c = min(max(b, a), dur), min(max(c, max(b, a) + 1e-3), dur)
    env = np.interp(np.arange(n) / sr, [0.0, a, b, c], [0.0, 1.0, 1.0, 0.0], right=0.0)
    return _out(modal_bank(rng.uniform(-1, 1, n) * env, body, sr, 0.0)[0], sr, 0.8, 30.0)


# Hail (Lane et al., "A hail size distribution impact transducer", JASA 2006 / arXiv 1408.4702; Kopp et al.,
# AMT 16:3487, 2023): a stone of diameter D falls at v = 142 sqrt(D) (Lane; Kopp's drag balance with
# rho_ice 870, cw 0.5 gives 138) and hits with the ramp F(t) = F0 (1 - t/tau), tau = xi D / v, carrying the
# impulse (1 + eta) m v (Lane eq. 1-3: eta = 0.9, xi = 0.1 for hard hail, ~1 for a raindrop: the short ramp
# is why hail has far more energy above 10 kHz than rain). Sizes are exponential (Kopp: median 6.7 mm over
# 80 Swiss sensors); with a rate falling as 1 - t/dur the largest stone lands at relative time 0.29 in
# median (Kopp: 0.34 for dense events).
HAIL_RHO, HAIL_ETA, HAIL_XI = 870.0, 0.9, 0.1


def hail_impulses(dur, sr, rate=80.0, size=6.7e-3, decay=1.0, hardness=1.0, seed=0):
    """Force (N) of a hail shower on one surface: Poisson stones, exponential diameters (median ``size`` m),
    each a Lane triangle of area (1 + eta) m v. Returns ``(force, diameters, times)``."""
    rng = np.random.default_rng(seed)
    n = samples(dur, sr)
    decay = float(np.clip(decay, 0.0, 1.0))
    # thinning: candidates at the peak rate, kept with probability 1 - decay t/dur
    t = np.sort(rng.uniform(0.0, dur, rng.poisson(max(rate, 0.0) * dur)))
    t = t[rng.uniform(size=len(t)) < 1.0 - decay * t / dur]
    d = np.clip(rng.exponential(size / math.log(2.0), len(t)), 1e-3, 0.15)
    v = 142.0 * np.sqrt(d)
    j = (1.0 + HAIL_ETA) * HAIL_RHO * math.pi / 6.0 * d ** 3 * v
    tau = HAIL_XI / float(np.clip(hardness, 0.05, 1.0)) * d / v
    force = np.zeros(n)
    for ti, ji, ta in zip(t, j, tau):
        m = max(1, int(round(ta * sr)))
        ramp = 1.0 - np.arange(m) / m
        i0 = int(ti * sr)
        seg = force[i0:i0 + m]
        seg += (ji * sr / ramp.sum() * ramp)[:len(seg)]
    return force, d, t


@sfx("hail", "Hail on a surface: Poisson stones, exponential sizes, each a short force ramp (Lane 2006) ringing the body; the shower thins out over dur.",
     dur=(6.0, "s"), rate=(80.0, "stones/s hitting the surface at the start"),
     size=(6.7, "mm median stone diameter (Kopp 2023: 6.7; 20-30 = severe)"),
     decay=(1.0, "0 steady .. 1 the rate falls to zero by the end (measured showers are front-loaded)"),
     hardness=(1.0, "0..1: 1 hard hail (contact 0.1 D/v), lower = graupel / slush, longer and darker"),
     body=("rect_plate", _SHAPE_HELP + " or a modal table"), material=("metal", _MAT_HELP + ": tin roof, car (metal), "
     "conservatory (glass), deck (wood)"), surface_size=(1.2, "m"), bright=(0.6, "0 dark .. 1 bright"), seed=(0, "variation"))
def hail(sr=DEFAULT_SR, dur=6.0, rate=80.0, size=6.7, decay=1.0, hardness=1.0, body="rect_plate", material="metal",
         surface_size=1.2, bright=0.6, seed=0):
    dur = float(np.clip(dur, 0.1, 120.0))
    size = float(np.clip(size, 1.0, 100.0)) * 1e-3
    y = np.zeros(samples(dur, sr))
    # stones land all over the surface: three strike points, each with its share of the shower
    for k, (p1, p2) in enumerate(((0.31, 0.37), (0.53, 0.71), (0.77, 0.23))):
        f, _, _ = hail_impulses(dur, sr, float(rate) / 3.0, size, decay, hardness, seed * 7 + k)
        b = _make_body(body, material, surface_size, p1, p2, modes=200, seed=seed + k, fmax=min(16000.0, 0.45 * sr))
        y += modal_bank(f, b, sr, bright)[0]
    return _out(y, sr, 0.8, 10.0)


@sfx("avalanche", "Avalanche / rock slide rumble: enveloped noise ringing a cluster of random low modes; never the same twice (seed).",
     dur=(10.0, "s until all sound has ended"), fmin=(100.0, "Hz lowest rumbling frequency"),
     fmax=(160.0, "Hz highest (raise it for wind-like or eerie tones)"), damping=(0.87, "1/s decay rate of the modes"),
     modes=(25, "number of random modes"), rise=(0.0, "s to full level (0 = 20 % of dur)"),
     hold=(0.0, "s when the fade starts (0 = 30 % of dur)"), fade=(0.0, "s when the slide itself has ended (0 = 60 % of dur)"),
     seed=(0, "which modes"))
def avalanche(sr=DEFAULT_SR, dur=10.0, fmin=100.0, fmax=160.0, damping=0.87, modes=25, rise=0.0, hold=0.0, fade=0.0, seed=0):
    return _rumble(sr, dur, fmin, fmax, damping, modes, rise, hold, fade, seed, (0.20, 0.30, 0.60))


@sfx("rumble", "Low rumble (earthquake, heavy machinery, distant collapse): enveloped noise into random low modes.",
     dur=(4.0, "s"), fmin=(35.0, "Hz"), fmax=(90.0, "Hz"), damping=(3.0, "1/s decay rate of the modes: lower = more tonal, slower"),
     modes=(16, "number of random modes"), rise=(0.0, "s to full level (0 = 10 % of dur)"),
     hold=(0.0, "s when the fade starts (0 = 70 % of dur)"), fade=(0.0, "s when the drive has ended (0 = 90 % of dur)"),
     seed=(0, "which modes"))
def rumble(sr=DEFAULT_SR, dur=4.0, fmin=35.0, fmax=90.0, damping=3.0, modes=16, rise=0.0, hold=0.0, fade=0.0, seed=0):
    return _rumble(sr, dur, fmin, fmax, damping, modes, rise, hold, fade, seed, (0.10, 0.70, 0.90))


ENGINE_BLOCK = 128.0 / 44100.0          # s per call of the force loop that may misfire

# car1 engine body: Hz (50 and 100 n^2), decay 1/s (not monotone), unit gains
CAR1_BODY = ((50.0, 100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0, 4900.0, 6400.0, 8100.0, 10000.0),
             (8.16104, 20.4026, 40.8052, 106.09352, 150.97924, 204.026, 412.13252, 334.60264, 497.82344, 500.0, 500.0))

# Valve train: lift mm, cam duration crank deg, lash mm, profile exponent, valve mass kg, spring N/mm, preload N,
# rebound ratio, exhaust valve heavier by
VALVE = dict(lift=9.5, duration=250.0, lash=0.15, shape=2.0, mass=0.11, k=38.0, preload=280.0, rebound=0.28, exhaust=1.3)


def valve_lift(x, lift=VALVE["lift"], lash=VALVE["lash"], shape=VALVE["shape"]):
    """Valve lift (mm) at x = 2 d / duration, d the crank angle from the cam nose: lift (1 - x^2)^shape - lash, 0 outside."""
    x = np.asarray(x, float)
    return np.clip(np.where(np.abs(x) < 1.0, lift * np.clip(1.0 - x * x, 0.0, None) ** shape, 0.0) - lash, 0.0, None)


def valve_float_rpm(lift=VALVE["lift"], duration=VALVE["duration"], shape=VALVE["shape"], mass=VALVE["mass"],
                    k=VALVE["k"], preload=VALVE["preload"]):
    """Crank rpm above which the spring can no longer hold the valve on the cam nose: the cam's deceleration there,
    2 shape lift (2 / D)^2 w^2 (D the duration in rad of crank, w the crank speed), equals (preload + k lift) / mass."""
    D = math.radians(float(duration))
    decel_per_w2 = 2.0 * float(shape) * float(lift) * 1e-3 * (2.0 / D) ** 2
    w = math.sqrt((float(preload) + float(k) * 1e3 * float(lift) * 1e-3) / float(mass) / decel_per_w2)
    return w * 60.0 / (2 * math.pi)


RUPTURE = dict(rod_area=1.6e-4, rod_bore=0.086, rod_strength=620e6, valve_gap=2.2e-3, valve_k=0.55, turbo_rate=3.0,
               fatigue_exp=9.0)


def conrod_stress(mass, crank, rpm, rod, bore=RUPTURE["rod_bore"]):
    """Peak tensile stress in the rod at TDC, sigma = m r w^2 (1 + r / l) / A, its section A = 1.6e-4 (B / 0.086)^2 m^2."""
    w = 2 * math.pi * float(rpm) / 60.0
    area = RUPTURE["rod_area"] * (float(bore) / RUPTURE["rod_bore"]) ** 2
    return float(mass) * float(crank) * w * w * (1.0 + float(crank) / float(rod)) / area


def conrod_breaks(mass, crank, rpm, rod, bore=RUPTURE["rod_bore"]):
    return conrod_stress(mass, crank, rpm, rod, bore) > RUPTURE["rod_strength"]


def valve_hits_piston(lift, piston_gap, clearance_volume, piston_area):
    """Valve meets piston when its lift passes the piston's distance plus 0.55 Vc / A plus 2.2 mm (all in m)."""
    return float(lift) > float(piston_gap) + RUPTURE["valve_k"] * float(clearance_volume) / float(piston_area) + RUPTURE["valve_gap"]


def turbo_spin_down(w0, t):
    """Turbo shaft after the oil fails: w0 exp(-3 t)."""
    return float(w0) * np.exp(-RUPTURE["turbo_rate"] * np.asarray(t, float))


def fatigue_damage(p, pmax):
    """Damage per cycle relative to the rated peak pressure: (p / pmax)^9."""
    return (np.asarray(p, float) / float(pmax)) ** RUPTURE["fatigue_exp"]


def valvetrain_force(n, sr, rpm, cylinders=1, seed=0):
    """Seat clicks of each cylinder's valves, exhaust closing just after TDC of the overlap (cycle 0.02), intake 40 crank
    degrees after BDC (0.306); the exhaust valve 1.3 times heavier. Past ``valve_float_rpm`` each seating bounces once,
    0.28 of its speed, a millisecond later, and the timing scatters."""
    rng = np.random.default_rng(int(seed) + 61)
    r = np.clip(_curve(rpm, n), 30.0, 30000.0)
    phase = np.cumsum(r / 120.0) / sr
    nc = int(np.clip(cylinders, 1, 16))
    out = np.zeros(n + int(0.004 * sr))
    click = np.exp(-np.arange(int(0.003 * sr)) / (0.0004 * sr)) * np.sin(2 * np.pi * 3000.0 * np.arange(int(0.003 * sr)) / sr)
    fl = valve_float_rpm()
    for c in range(nc):
        for at, g in ((0.02, VALVE["exhaust"]), (0.306, 1.0)):
            cyc = phase + c / nc - at
            for i in np.nonzero(np.diff(np.floor(cyc)) > 0)[0]:
                over = r[i] > fl
                j = i + (int(rng.normal(0.0, 0.0005) * sr) if over else 0)
                v = g * (r[i] / 3000.0)
                for jj, a in ((j, v), (j + int(0.001 * sr), VALVE["rebound"] * v if over else 0.0)):
                    if 0 <= jj < n and a > 0:
                        out[jj:jj + len(click)] += a * click
    return out[:n]


def creep_closed_form(q, mu, p, g, a, nu=0.3):
    """Longitudinal creep of a sphere (Hertz circle radius ``a``) under tangential force ``q`` below sliding (|q| < mu p):
    xi = -(3 mu p (4 - 3 nu) / (16 G a^2)) (1 - (1 - q / (mu p))^(1/3))."""
    s = np.clip(np.asarray(q, float) / (float(mu) * float(p)), -1.0, 1.0)
    return -3.0 * mu * p * (4.0 - 3.0 * nu) / (16.0 * g * a * a) * np.sign(s) * (1.0 - (1.0 - np.abs(s)) ** (1.0 / 3.0))


SPIN_CREEP_MAX = 2.0                    # psi R / (mu c) at which spin (camber) saturates the side force


def rolling_loss_peak(v, a):
    """Angular frequency (rad/s) at which a viscoelastic cylinder of contact half-width ``a`` rolling at ``v`` loses most:
    1.83 v / a; with beta = 1 the peak resistance is mu_R = 0.195 a / R (`rolling_resistance_peak`)."""
    return 1.83 * float(v) / float(a)


def rolling_resistance_peak(a, radius):
    """Largest rolling resistance coefficient of the standard viscoelastic solid (beta = 1): 0.195 a / R."""
    return 0.195 * float(a) / float(radius)


PLASTIC_ROLL = (0.104, 0.076, 0.047)    # M0 k / P^2: first pass, and the two steady-state bounds


def plastic_rolling_moment(p, k, case=0):
    """Resisting moment per unit length of a rigid cylinder rolling plastically on a surface of shear yield ``k``,
    load ``p`` per unit length: M0 = c p^2 / k, c from ``PLASTIC_ROLL`` (a soft ground, a cannonball on turf)."""
    return PLASTIC_ROLL[int(case)] * float(p) ** 2 / float(k)


def profile_plasticity(height, dx, es, hardness):
    """Plasticity index of a measured profile: sigma = rms height, kappa = mean curvature of its summits (second
    difference over the sampling step ``dx``), psi = (E*/H) sqrt(sigma kappa). sigma grows as sqrt of the profile length
    and kappa with a finer step (10 um against 100 um): give the step the contact sees."""
    h = np.asarray(height, float) - np.mean(height)
    curv = -(h[2:] - 2 * h[1:-1] + h[:-2]) / float(dx) ** 2
    peaks = (h[1:-1] > h[:-2]) & (h[1:-1] > h[2:])
    kappa = float(np.mean(curv[peaks])) if peaks.any() else float(np.mean(np.abs(curv)))
    return plasticity_index(es, hardness, float(np.std(h)), kappa)


def engine_force(rng, n: int, sr: int, rpm=1200, cylinders=1, intake=0.5, combustion=1.0, exhaust=0.7, cyl_spread=0.35,
                 misfire=0.0, seed=0) -> np.ndarray:
    """Looped four-stroke force of `modal_engine`. The stroke gains are curves. ``misfire`` is the probability per
    ``ENGINE_BLOCK`` s; a cylinder's cycle misfires with p_cycle = 1 - (1 - p)^(cycle / block), and then its combustion
    stroke gives no force (intake and exhaust go on)."""
    L = 4096
    q = L // 4
    tq = np.arange(q) / q
    tabs = np.zeros((3, L))
    tabs[0, :q] = rng.uniform(-1, 1, q) * np.exp(-0.5 * ((tq - 0.5) / 0.18) ** 2)                     # Gaussian bell
    comb = fractal_noise(q, 1.0, 1.0, f_low=2.0 / q, seed=seed + 1)                                      # 1/f burst
    tabs[1, 2 * q:3 * q] = comb / (np.max(np.abs(comb)) + 1e-30) * np.minimum(tq / 0.03, 1.0) * np.exp(-tq / 0.25)
    tabs[2, 3 * q:] = rng.uniform(-1, 1, q) * np.exp(-tq / 0.12)                                         # fast decay
    g = [np.maximum(_curve(v, n), 0.0) for v in (intake, combustion, exhaust)]
    d = np.clip(_curve(rpm, n), 30.0, 30000.0) / 120.0
    p = float(np.clip(misfire, 0.0, 1.0))
    phase = np.cumsum(d) / sr
    p_cycle = 1.0 - (1.0 - p) ** (1.0 / (d * ENGINE_BLOCK))
    nc = int(np.clip(cylinders, 1, 16))
    gains = 1.0 + float(np.clip(cyl_spread, 0.0, 1.0)) * rng.uniform(-0.8, 0.8, nc) if nc > 1 else np.ones(1)
    force = np.zeros(n)
    u = np.random.default_rng(seed + 2).random((nc, int(phase[-1]) + 3))
    for c in range(nc):
        x = (phase + c / nc) % 1.0 * L
        fire = u[c, np.floor(phase + c / nc).astype(np.int64)] >= p_cycle if p > 0 else 1.0
        for k in range(3):
            gk = g[k] * fire if k == 1 else g[k]
            force += gains[c] * gk * np.interp(x, np.arange(L + 1), np.append(tabs[k], tabs[k, 0]))
    return force


@sfx("modal_engine","Four-stroke engine as a looped force (intake noise, silence, combustion 1/f burst, exhaust noise) driving a low plate body: motorcycle, mower, chainsaw.",
     rpm=(1200, "crank RPM; number or list of breakpoints (one cycle = 2 revolutions)"), dur=(2.0, "s"),
     cylinders=(1, "count, evenly spaced over the cycle"), intake=(0.5, "gain of the intake stroke"),
     combustion=(1.0, "gain of the combustion stroke"), exhaust=(0.7, "gain of the exhaust stroke"),
     cyl_spread=(0.35, "0..1 gain differences between cylinders (needed for a convincing multi-cylinder)"),
     fan=(0.15, "level of the pitched background (fan / rotating parts)"),
     body_freq=(50.0, "Hz lowest body resonance: 50 motorcycle .. 300+ with high damping = lawn mower"),
     damping=(5.56, "1/s external damping of the body"), modes=(13, "body modes (about 7 suffice)"),
     bright=(0.5, "0 dark .. 1 bright (0.5 = velocity tap, the thesis's choice for engines)"),
     misfire=(0.0, "0..0.1 probability per 128-sample block (at 44.1 kHz), converted per cycle: a misfiring cycle has no combustion stroke"),
     pick=("lowest", "lowest: the lowest body modes | rank: the most audible modes up to 8060 Hz (the thesis's rectplate span)"),
     body=("plate", "plate (the rectangular plate above) | car1 (a car engine body: 50 Hz and 100 n^2 Hz, n = 1..10)"),
     valves=(0.0, "0..1 valve train clatter: each valve seating, and bouncing once past the valve-float rpm"),
     seed=(0, "variation"))
def modal_engine(sr=DEFAULT_SR, rpm=1200, dur=2.0, cylinders=1, intake=0.5, combustion=1.0, exhaust=0.7,
                 cyl_spread=0.35, fan=0.15, body_freq=50.0, damping=5.56, modes=13, bright=0.5, misfire=0.0, pick="lowest",
                 seed=0, body="plate", valves=0.0):
    """A looped four-stroke force driving a rectangular plate (aspect 7.0528, 13 modes, base 50 Hz) or the car1 body.
    ``intake``, ``combustion``, ``exhaust`` take breakpoint lists (a load that changes while driving)."""
    if body not in ("plate", "car1"):
        raise ValueError("modal_engine body must be plate or car1")
    rng = np.random.default_rng(seed)
    n = samples(float(np.clip(dur, 0.05, 120.0)), sr)
    force = engine_force(rng, n, sr, rpm, cylinders, intake, combustion, exhaust, cyl_spread, misfire, seed)
    if fan > 0:        # "a short noisy note of a 70 cm pipe": narrow noise at c/(2 x 0.7 m) = 245 Hz
        force += float(fan) * 4.0 * F.bandpass(rng.uniform(-1, 1, n), 245.0, sr, q=12.0)
    if valves > 0:
        force += float(np.clip(valves, 0.0, 1.0)) * valvetrain_force(n, sr, rpm, cylinders, seed)
    if body == "car1":
        f = np.array(CAR1_BODY[0])
        ok = f < 0.45 * sr
        car = ModalObject(f[ok], np.array(CAR1_BODY[1])[ok] + float(np.clip(damping, 0.0, 2000.0)), np.ones(int(ok.sum())))
        return _out(modal_bank(force, car, sr, bright)[0], sr, 0.8, 20.0)
    keep = int(np.clip(modes, 1, 60))
    body = ShapeBody("rect_plate", float(np.clip(body_freq, 15.0, 2000.0)), material="metal", aspect=7.0528,
                     position=(0.31, 0.43), n_modes=400 if pick == "rank" else keep, fmax=min(8060.745, 0.45 * sr),
                     ext=float(np.clip(damping, 0.0, 2000.0)), seed=seed)
    if pick == "rank":
        body = body.pruned(keep=keep, bright=bright)
    return _out(modal_bank(force, body, sr, bright)[0], sr, 0.8, 20.0)
