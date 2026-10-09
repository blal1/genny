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
from dataclasses import dataclass
from functools import lru_cache

import numba
import numpy as np
from scipy import special
from scipy.ndimage import grey_dilation
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


def _m(name, fd, scatter, ext, k, mu, rho, young, nu, rayleigh=None):
    p = _PM.get("aluminum" if name == "aluminium" else name)     # reuse genny.physics elastic constants
    if p is not None:
        rho, young, nu = p.density, p.young_modulus, p.poisson
    return ContactMaterial(name, fd, scatter, ext, k, mu, rho, young, nu, rayleigh)


# Contact stiffness: the five log-spaced levels of the hardness experiment, 5e6 (rubber mallet),
# 3.3e7, 2.24e8, 1.495e9, 1e10 (steel mallet) (Sounding Object §5.3; g04 §1.4).
# UNSOURCED: which of the three middle levels belongs to plastic / wood / glass (only the ends are named).
# UNSOURCED: mu per material except wood (the book prints only mu = 0.6 for a felt-like contact).
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
    # UNSOURCED: ice, stone, concrete have no damping data in the findings; minimal glass-like guesses.
    _m("ice", 90.0, 0.55, 15.0, 1.495e9, 0.2, 917, 9e9, 0.33),
    _m("stone", 120.0, 0.55, 20.0, 1.495e9, 0.15, 2600, 50e9, 0.25),
    _m("concrete", 60.0, 0.55, 40.0, 1.495e9, 0.2, 2400, 30e9, 0.2),
)}
_ALIAS = {"aluminum": "aluminium", "porcelain": "ceramic", "iron": "metal", "plexiglass": "plastic"}
K_SOFT, K_HARD = 5e6, 1e10          # rubber mallet .. steel mallet (g04 §1.4)


def get_material(name) -> ContactMaterial:
    if isinstance(name, ContactMaterial):
        return name
    key = _ALIAS.get(str(name).lower(), str(name).lower())
    if key not in MATERIALS:
        raise ValueError(f"unknown material {name!r}; choose from {sorted(MATERIALS)}")
    return MATERIALS[key]


def material_damping(freqs, mat="steel", *, law: str = "friction", ext: float | None = None,
                     scatter: float | None = None, seed: int = 0) -> np.ndarray:
    """Decay rate d_n (1/s, amplitude envelope exp(-d t); T60 = 6.91/d) of each mode.

    ``law="friction"``: ``d_n = pi f_n tan(phi) / J_n + ext`` - the Wildes-Richards internal
    friction law (g06 A5 eq. 12-13; g04 §5.1) with the frequency-independent external loss, and
    per-mode log-normal scatter ``J_n`` of relative spread ``scatter`` ("draw each mode's damping
    as d = f/R with R scattered 50-100 %", g06 B1). ``law="rayleigh"``:
    ``d_n = (alpha + beta (2 pi f_n)^2)/2`` (O'Brien eq. 9, Ren eq. 6; g07 §1.1, §4.1).
    """
    mat = get_material(mat)
    f = np.asarray(freqs, float)
    if law == "rayleigh":
        if mat.rayleigh is None:
            raise ValueError(f"no sourced Rayleigh pair for {mat.name}")
        a, b = mat.rayleigh
        return 0.5 * (a + b * (TWO_PI * f) ** 2)
    cv = mat.scatter if scatter is None else float(scatter)
    j = np.ones_like(f)
    if cv > 0:
        s = math.sqrt(math.log1p(cv * cv))
        j = np.exp(np.random.default_rng(seed).normal(-0.5 * s * s, s, f.shape))   # mean 1
    return math.pi * f * mat.tan_phi / j + (mat.ext if ext is None else float(ext))


# ==========================================================================================
# Analytic shapes (g06 A6, B4; formulas: Morse, "Vibration and Sound"; Fletcher & Rossing)
# ==========================================================================================

SHAPES = ("string", "bar", "rect_membrane", "rect_plate", "circ_membrane", "circ_plate")
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
    else:
        raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")
    order = np.argsort(ratio, kind="stable")
    order = order[ratio[order] <= rmax * (1 + 1e-9)][:n_modes]
    if len(order) == 0:
        order = np.argsort(ratio, kind="stable")[:1]
    return ratio[order] / ratio[order[0]], {k: v[order] for k, v in desc.items()}


def _thickness(shape: str, size: float) -> float:
    # UNSOURCED: geometric proportions of a "typical" object (bar 4 %, plate 2 % of its length).
    return (0.04 if shape == "bar" else 0.02) * size


def fundamental(shape: str, size: float, mat="steel", *, boundary: str = "free", aspect: float = 1.0,
                thickness: float | None = None, wave_speed: float = 120.0) -> float:
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
        return b1 * b1 / (TWO_PI * L * L) * mat.c * h / math.sqrt(12.0)
    if shape == "rect_plate":
        return 0.5 * math.pi * h * cp * (1.0 / L ** 2 + (aspect / L) ** 2)
    if shape == "circ_plate":
        return 3.19622 ** 2 / (TWO_PI * (0.5 * L) ** 2) * h * cp
    if shape == "string":
        return wave_speed / (2.0 * L)                       # UNSOURCED default speed: tension is free
    if shape == "rect_membrane":
        return 0.5 * wave_speed * math.sqrt(1.0 / L ** 2 + (aspect / L) ** 2)
    if shape == "circ_membrane":
        return wave_speed * 2.404826 / (TWO_PI * 0.5 * L)
    raise ValueError(f"unknown shape {shape!r}")


def body_mass(shape: str, size: float, mat="steel", *, aspect: float = 1.0, thickness: float | None = None) -> float:
    """Mass (kg) of the shape, used as the modal mass of every (unit mean-square) mode."""
    mat = get_material(mat)
    L = max(float(size), 1e-3)
    h = thickness if thickness else _thickness(shape, L)
    if shape == "bar":
        return mat.density * L * 2.5 * h * h                # UNSOURCED: width 2.5 x thickness
    if shape == "rect_plate":
        return mat.density * L * L / aspect * h
    if shape == "circ_plate":
        return mat.density * math.pi * (0.5 * L) ** 2 * h
    if shape == "string":
        return 0.005 * L                                    # UNSOURCED: 5 g/m string
    area = L * L / aspect if shape == "rect_membrane" else math.pi * (0.5 * L) ** 2
    return 0.3 * area                                       # UNSOURCED: 0.3 kg/m^2 membrane


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
        return self.aw0 * (self.freqs / self.freqs.min()) ** (2.0 * float(np.clip(bright, 0.0, 1.0)))

    def gains(self, position=None, bright: float = 0.0) -> np.ndarray:
        """van den Doel's heard amplitudes ``a_n = Psi_n(p) / (w_n Q(p))``, ``Q = sqrt(sum Psi^2)``
        (eq. 11; g06 A4): every strike point delivers the same energy."""
        p = self.row(position)
        q = math.sqrt(float(np.sum(p * p))) + 1e-30
        return p * self.audio_weights(bright) / (TWO_PI * self.freqs * q)

    def rank(self, position=None, bright: float = 0.5) -> np.ndarray:
        """Mode indices, most important first: ear-weighted mode energy ``(a_n f_n)^2 S(f_n)``
        (sonic-distance energy of g06 A7 with genny's ear weighting as S; "add modes in decreasing
        order of importance", g06 B3/B9)."""
        e = (self.gains(position, bright) * self.freqs * _ear_weight(self.freqs)) ** 2
        return np.argsort(-e, kind="stable")

    def subset(self, idx) -> "ModalObject":
        idx = np.sort(np.asarray(idx))
        o = copy.copy(self)
        for k in ("freqs", "decays", "phi", "aw0"):
            setattr(o, k, np.ascontiguousarray(getattr(self, k)[idx]))
        o._desc = {k: v[idx] for k, v in self._desc.items()}
        return o

    def pruned(self, keep: int | None = None, floor_db: float | None = None, bright: float = 0.5) -> "ModalObject":
        """Keep the ``keep`` most important modes and / or those within ``floor_db`` of the strongest
        (drop a mode whose log-energy is below the audible floor, g06 A7/B9)."""
        order = self.rank(bright=bright)
        if floor_db is not None:
            e = (self.gains(bright=bright) * self.freqs * _ear_weight(self.freqs)) ** 2
            order = order[10 * np.log10(e[order] / (e[order[0]] + 1e-300) + 1e-300) >= -abs(floor_db)]
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
    ``circ_plate`` (clamped edge). ``position``: x in 0..1 (string, bar), (x, y) (rectangles) or r
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
        r = float(np.clip(a, 0.0, 1.0))
        ang = np.cos(d["m"] * (0.0 if b is None else b))
        rad = special.jv(d["m"], d["j"] * r) if self.shape == "circ_membrane" else _plate_radial(d["m"], d["k"], r)
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


def modal_bank(force, body: ModalObject, sr: int = DEFAULT_SR, bright: float = 0.5):
    """Linear state-space bank: force (N) at the body's contact point -> ``(audio, displacement,
    velocity)``; the last two are the motion of the contact point itself (what an interaction
    needs to be two-way coupled)."""
    force = np.ascontiguousarray(force, dtype=np.float64)
    ok = body.freqs < 0.45 * sr
    w, d = _prewarp(body.freqs[ok], body.decays[ok], float(sr))
    return _bank_kernel(force, float(sr), w, d, np.ascontiguousarray(body.phi[ok]),
                        np.full(int(ok.sum()), 1.0 / body.mass), np.ascontiguousarray(body.audio_weights(bright)[ok]))


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


def bounce_schedule(first_interval: float = 0.3, velocity: float = 1.0, factor: float = 0.74,
                    vel_factor: float | None = None, *, dev_t: float = 0.0, dev_v: float = 0.0,
                    regularize: float = 0.0, t_min: float = 0.004, n_max: int = 200, seed: int = 0):
    """The "bouncer" (Sounding Object §9.2, eq. 9.3-9.5; g05 §1): ``(times, velocities)``.

    Maximal intervals and velocities follow one geometric law ``t_n = factor^n t_0``,
    ``v_n = vel_factor^n v_0`` (for a sphere both factors are the restitution sqrt(C); 0.70-0.76
    for a steel ball on wood, Table 13.5). ``dev_t`` / ``dev_v`` in 0..1 are random deviations
    *below* those maxima (non-spherical objects are irregular but bounded by the spherical
    envelope). ``regularize`` > 0 makes the deviations die out as the bounces shrink (the falling
    coin: random start, regular end). The series stops when the interval falls under ``t_min``
    (the "buzz" guard / terminating bang).
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
        T *= factor
        V *= velf
    return np.array(times), np.array(vels)


def break_schedule(dur: float = 1.2, *, first: float = 0.004, growth: float = 1.25, dev: float = 0.8,
                   velocity: float = 1.0, vel_factor: float = 0.85, trains: int = 1, seed: int = 0):
    """The "dropper" (Sounding Object §9.3; g05 §2): ``(times, velocities, train index)``, sorted.

    A bouncer with time factor > 1 and high randomness: fragments do not bounce, they "nod" -
    intervals *grow*, so the event density starts massive and falls quickly. Several ``trains``
    with different growth and velocity factors = several damped pulse trains with different
    damping, the cue for breaking as opposed to bouncing (Warren & Verbrugge 1984; g04 §1.1).
    """
    # UNSOURCED: the book gives the rule but no numbers for first / growth / dev / vel_factor.
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
    # UNSOURCED: e_unit maps the normalised energies of (9.12) onto the e_tot budget (the book
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
    # UNSOURCED: band centre (600 cycles/m), Q range and height (1-30 um rms) - the book prints no values.
    s = F.bandpass(np.random.default_rng(seed).standard_normal(n), 600.0, 1.0 / dx, q=4.0 - 3.4 * roughness)
    s *= 1e-6 * 10 ** (1.5 * roughness) / (s.std() + 1e-30)
    if joints > 0:
        x = (np.arange(n) * dx) % joints
        w = 0.004                                            # UNSOURCED: 4 mm wide, 0.5 mm deep groove
        s -= 5e-4 * np.where(x < w, 0.5 * (1 - np.cos(TWO_PI * x / w)), 0.0)
    return s


_TEXTURES = {      # beta, periodic share, period (grains), spike share. Names from van den Doel's
    "white": (0.0, 0.0, 0, 0.0),       # texture set (thesis Fig. 7.11; g06 B6): white, 1/f^.5 and
    "pink_half": (0.5, 0.0, 0, 0.0),   # 1/f are specified there, "grid" is "a periodic profile";
    "pink": (1.0, 0.0, 0, 0.0),        # UNSOURCED: the recipes for the recorded ones
    "brown": (2.0, 0.0, 0, 0.0),       # (gritty, wood, metal, plastic, sandpaper).
    "gritty": (0.3, 0.0, 0, 0.6),
    "sandpaper": (0.0, 0.0, 0, 0.25),
    "grid": (2.0, 1.0, 12, 0.0),
    "wood": (1.2, 0.5, 24, 0.0),
    "metal": (1.6, 0.35, 6, 0.0),
    "plastic": (1.4, 0.0, 0, 0.0),
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


def scrape_force(profile: np.ndarray, speed, force, dx: float, sr: int) -> np.ndarray:
    """Force of a scraper that follows a looped surface profile exactly (van den Doel thesis
    ch. 6; g06 B6): read the profile at ``speed`` (m/s per sample) and scale by the normal
    ``force``. ``dx`` (m) is the grain: the spacing of the profile samples."""
    pos = np.cumsum(np.asarray(speed, float)) / (sr * dx) % len(profile)
    i = pos.astype(np.int64)
    fr = pos - i
    return np.asarray(force, float) * ((1 - fr) * profile[i] + fr * profile[(i + 1) % len(profile)])


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
            for i in range(i0, end):
                kk = i - i0
                y = a1 * y1 + a2 * y2
                if kk < tc:
                    y += b * (1.0 - math.cos(2.0 * math.pi * (kk + 1) / (tc + 1))) / (tc + 1)
                y2 = y1
                y1 = y
                out[i] += y
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
    y = F.dc_block(y - y.mean(), sr)
    p = float(np.max(np.abs(y)))
    if p < 1e-30:
        return y
    y = declick(y / p, sr, fade_ms)                          # declick looks at the level: normalise first
    return y * (peak / (float(np.max(np.abs(y))) + 1e-30))


def _k_of(hardness: float) -> float:
    """0..1 -> log-spaced contact stiffness over the sourced range, rubber mallet 5e6 ... steel 1e10."""
    return K_SOFT * (K_HARD / K_SOFT) ** float(np.clip(hardness, 0.0, 1.0))


def _series(*ks) -> float:
    """Two compliant surfaces in contact: the softer one dominates (Hertz: 1/E* = sum (1 - nu^2)/E)."""
    return 1.0 / sum(1.0 / k for k in ks)


def _split_shape(shape: str):
    for b in ("clamped_clamped", "clamped", "supported", "free"):
        if shape == f"bar_{b}":
            return "bar", b
    return shape, "free"


def _make_body(shape, mat, size, position=0.3, position2=0.37, *, freq=0.0, modes=24, damping=0.0, seed=0,
               fmax=12000.0, scatter=None, aspect=1.3):
    """A catalog body: an analytic shape (pitch and mass from size and material) or a measured
    ``modal_data`` table name (frequencies scaled by 0.3 / size)."""
    mat = get_material(mat)
    size = float(np.clip(size, 0.005, 20.0))
    shape, boundary = _split_shape(str(shape))
    modes = int(np.clip(modes, 1, 200))
    if shape in SHAPES:
        asp = aspect if shape.startswith("rect") else 1.0
        f1 = float(freq) if freq and freq > 0 else fundamental(shape, size, mat, boundary=boundary, aspect=asp)
        # every edge that is held is a node of all modes: keep the strike point just inside it
        lo = 0.0 if shape.startswith("circ") else 0.02
        pos = float(np.clip(position, lo, 0.98))
        if shape.startswith("rect"):
            pos = (pos, float(np.clip(position2, 0.02, 0.98)))
        return ShapeBody(shape, float(np.clip(f1, 20.0, 8000.0)), material=mat, boundary=boundary, aspect=asp,
                         position=pos, n_modes=modes, fmax=fmax, mass=body_mass(shape, size, mat, aspect=asp),
                         ext=mat.ext + max(float(damping), 0.0), scatter=scatter, seed=seed)
    if shape not in _modal.md.TABLES:
        raise ValueError(f"unknown body {shape!r}: use one of {SHAPES} (bar_clamped, bar_supported, "
                         f"bar_clamped_clamped) or a modal table such as 'sy_vase'")
    # UNSOURCED: measured tables carry no size; 0.3 m is taken as the size they were measured at.
    b = ModalObject.from_table(shape, f_scale=0.3 / size, position=float(np.clip(position, 0, 1)))
    if freq and freq > 0:
        b.freqs = b.freqs * (float(freq) / b.freqs.min())
    b.decays = b.decays + max(float(damping), 0.0)
    return b.pruned(keep=modes) if len(b.freqs) > modes else b


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
                 per_band: int = 10, seed: int = 0):
    """``(freqs, decays, gains)`` of a hollow box (cube of edge ``size``) or the sphere of equal
    volume (pitch follows volume, §6.1).

    Box: ``f(l,m,n) = (c/2) sqrt((l/X)^2 + (m/Y)^2 + (n/Z)^2)`` (eq. 6.1). Sphere:
    ``f(n,s) = c z_ns/(2 pi a)``, z_ns the roots of the derivative of the spherical Bessel
    function j_n (eq. 6.2). Decay from Sabine, ``T60 = 0.163 V/(alpha(f) A)`` (eq. 6.3), with the
    wood-like absorption table (eq. 6.4-6.5) scaled by ``wall`` (< 1 = harder walls). Mode
    amplitudes are random and the f^2 growth of modal density is stopped at ``per_band`` modes
    per third octave ("where single modes can no longer be discriminated").
    """
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
    band = np.floor(3.0 * np.log2(f / f[0]) + 1e-9).astype(int)        # UNSOURCED: 10 modes per third octave
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
        body=("rect_plate", "string|bar|bar_clamped|bar_supported|rect_membrane|rect_plate|circ_membrane|circ_plate, or a modal table (sy_vase, sy_wok, churchBell...)"),
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

_MAT_HELP = "rubber|plastic|wood|glass|ceramic|metal|steel|aluminium|bronze|ice|stone|concrete"
_SHAPE_HELP = "string|bar|bar_clamped|bar_supported|rect_membrane|rect_plate|circ_membrane|circ_plate"


@sfx("impact", "One object struck by another: analytic modal body + Hunt-Crossley contact (harder / faster = shorter contact = brighter).",
     material=("wood", _MAT_HELP + ": ring time, pitch and contact hardness of the struck body"),
     shape=("bar", _SHAPE_HELP + " (or a modal table name such as sy_vase)"),
     size=(0.3, "m: length / diameter of the struck body; bigger = lower"),
     position=(0.28, "0..1 strike point (bars: 0 end .. 0.5 middle; circles: 0 centre .. 1 rim); near an edge = brighter"),
     position2=(0.37, "0..1 second coordinate of the strike point on rectangles"),
     striker_mass=(0.02, "kg: heavier = louder, longer contact, darker"),
     hardness=(0.6, "0..1 striker hardness, rubber mallet .. steel (contact stiffness 5e6 .. 1e10 N/m^1.5)"),
     velocity=(1.5, "m/s strike speed: louder and brighter"),
     hollow=(0.0, "0..1 share of a hollow air-cavity resonance of the same size"),
     damping=(2.0, "external damping 1/s: 0 freely hanging .. 30 lying on a table / held"),
     bright=(0.5, "0 dark .. 1 bright radiation"), freq=(0.0, "Hz: lowest mode, 0 = from size and material"),
     modes=(24, "number of modes"), dur=(0.0, "s, 0 = until it has rung out (max 4 s)"), seed=(0, "variation of the mode decays"))
def impact(sr=DEFAULT_SR, material="wood", shape="bar", size=0.3, position=0.28, position2=0.37, striker_mass=0.02,
           hardness=0.6, velocity=1.5, hollow=0.0, damping=2.0, bright=0.5, freq=0.0, modes=24, dur=0.0, seed=0):
    mat = get_material(material)
    body = _make_body(shape, mat, size, position, position2, freq=freq, modes=modes, damping=damping, seed=seed,
                      fmax=min(12000.0, 0.45 * sr))
    velocity = float(np.clip(velocity, 0.01, 30.0))
    hardness = float(np.clip(hardness, 0.0, 1.0))
    # UNSOURCED: striker dissipation vs. hardness (soft mallets lose more); the body's own mu is sourced/table.
    it = ImpactInteractor(body, mass=float(np.clip(striker_mass, 1e-5, 50.0)), k=_series(mat.k, _k_of(hardness)),
                          mu=max(mat.mu, 0.05 + 0.55 * (1.0 - hardness)), bright=bright)
    T = float(np.clip(dur, 0.03, 12.0)) if dur and dur > 0 else _tail(body) + 0.02
    y = it.strike(velocity, T, sr).audio
    if hollow > 0:
        h = float(np.clip(hollow, 0.0, 1.0))
        y = _wet(y, _cavity(y, sr, "box", max(0.6 * float(size), 0.05), 1.0, seed), h)
    return _out(y, sr, 0.85 * float(np.clip((velocity / 3.0) ** 0.5, 0.25, 1.0)))


@sfx("bounce", "An object dropped on a surface bouncing to rest: geometric bounce train, each hit a physical impact on the still-ringing bodies.",
     material=("steel", _MAT_HELP + ": the bouncing object"),
     size=(0.02, "m: diameter / length of the object; heavier = longer contacts, darker"),
     shape=("ball", "ball (rigid: only the surface rings) | bar | circ_plate (lid, coin) | rect_plate ..."),
     surface=("wood", _MAT_HELP + ": what it lands on (dominates the heard material)"),
     surface_size=(0.6, "m: size of the surface plate; bigger = lower"),
     first_interval=(0.3, "s between the first two impacts (= drop height: h = g t^2/8)"),
     restitution=(0.74, "ratio of successive intervals and speeds, 0.3 dead .. 0.9 lively (0.70-0.76 steel ball on wood)"),
     irregularity=(0.0, "0..1 non-roundness: intervals and hits randomly shorter / weaker, strike point changes per bounce"),
     velocity=(0.0, "m/s first impact speed; 0 = from the interval, g t/(2 restitution)"),
     bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def bounce(sr=DEFAULT_SR, material="steel", size=0.02, shape="ball", surface="wood", surface_size=0.6,
           first_interval=0.3, restitution=0.74, irregularity=0.0, velocity=0.0, bright=0.5, seed=0):
    y, _ = _bounce_render(sr, material, size, shape, surface, surface_size, first_interval, restitution,
                          irregularity, velocity, bright, seed)
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
                   velocity, bright, seed, extra=0.0, stage=None):
    mat = get_material(mat)
    rng = np.random.default_rng(seed)
    irr = float(np.clip(irregularity, 0.0, 1.0))
    e = float(np.clip(restitution, 0.05, 0.97))
    t0 = float(np.clip(first_interval, 0.01, 3.0))
    v0 = float(velocity) if velocity and velocity > 0 else G * t0 / (2.0 * e)
    bodies, mass, smat = _bounce_bodies(sr, mat, float(np.clip(size, 0.002, 2.0)), shape, surface, surface_size, seed)
    t, v = bounce_schedule(t0, v0, e, dev_t=0.6 * irr, dev_v=0.6 * irr, t_min=0.012, seed=seed)
    rows = _rows(bodies, min(len(t), 12) if irr > 0 else 1, irr, rng)
    pr = np.arange(len(t)) % len(rows)
    pm = mass * (1.0 - 0.5 * irr * rng.random(len(t)))          # effective mass varies per impact (g05 §1)
    it = ImpactInteractor(bodies, mass=mass, k=_series(mat.k, smat.k), mu=max(mat.mu, smat.mu), bright=bright)
    T = float(t[-1]) + extra + min(max(_tail(b, 2.5) for b in bodies), 2.5) + 0.05
    kw = stage(it, T, t[-1], mass) if stage else {}
    if stage:
        t, v, pr, pm = (np.append(t, t[-1] + 0.012), np.append(v, 0.02), np.append(pr, 0), np.append(pm, mass))
    return it.run(T, sr, events=(t, v, pr, pm), rows=rows, **kw).audio, t


def _roll_stage(n, fs, speed, radius, mass, roughness, joints, dents, asymmetry, seed):
    """Offset curve and normal load of a rolling ball (Sounding Object §9.4; g05 §3): band-limited
    surface -> rolling filter -> read at the ball's position (all surface frequencies scale with
    speed, amplitude constant); a dent recurs once per revolution (f = v/(2 pi r)); a non-round
    ball modulates contact speed and normal force at the rotation rate."""
    asym = float(np.clip(asymmetry, 0.0, 1.0))
    theta = np.cumsum(speed) / (fs * radius)
    pos = np.cumsum(speed * (1.0 + 0.3 * asym * np.cos(theta))) / fs
    dx = 1e-4
    off = rolling_offset(rolling_surface(float(pos[-1]) + 0.05, dx, roughness, joints=joints, seed=seed), radius, dx)
    off = np.interp(pos / dx, np.arange(len(off)), off, period=len(off))
    if dents > 0:
        w = 0.25                                             # UNSOURCED: dent 0.25 rad wide, up to 50 um deep
        th = (theta + math.pi) % TWO_PI - math.pi
        off = off - float(np.clip(dents, 0, 1)) * 5e-5 * np.where(np.abs(th) < w, np.cos(0.5 * math.pi * th / w) ** 2, 0.0)
    return off, mass * G * (1.0 + 0.6 * asym * np.sin(theta))


@sfx("drop", "A ball dropped on a surface: bounces until it stays in contact, then rolls away and comes to rest.",
     material=("steel", _MAT_HELP + ": the ball"), size=(0.03, "m: ball diameter"),
     surface=("wood", _MAT_HELP + ": the floor / table"), surface_size=(0.8, "m: size of the surface plate"),
     height=(0.25, "m drop height (sets impact speed and first bounce interval)"),
     restitution=(0.6, "bounce interval ratio 0.3..0.9"), roll=(1.5, "s of rolling after the last bounce (0 = none)"),
     roll_speed=(0.4, "m/s rolling speed right after the bounces; slows to rest"),
     roughness=(0.5, "0..1 surface roughness while rolling"), irregularity=(0.0, "0..1 uneven bounces"),
     bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def drop(sr=DEFAULT_SR, material="steel", size=0.03, surface="wood", surface_size=0.8, height=0.25, restitution=0.6,
         roll=1.5, roll_speed=0.4, roughness=0.5, irregularity=0.0, bright=0.5, seed=0):
    e = float(np.clip(restitution, 0.05, 0.95))
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
     asymmetry=(0.0, "0..1 non-round ball: speed and load wobble at the rotation rate"),
     feedback=(0.18, "legacy, ignored: the contact is now physically two-way coupled"),
     bright=(0.4, "0 dark .. 1 bright"), seed=(0, "variation"))
def roll(sr=DEFAULT_SR, dur=1.0, speed=0.5, radius=0.03, roughness=0.5, material="steel", surface="wood",
         surface_size=1.0, joints=0.0, dents=0.0, asymmetry=0.0, feedback=0.18, bright=0.4, seed=0):
    mat, smat = get_material(material), get_material(surface)
    radius = float(np.clip(radius, 0.002, 1.0))
    dur = float(np.clip(dur, 0.03, 120.0))
    mass = float(np.clip(_ball_mass(mat, 2.0 * radius), 1e-5, 500.0))
    k = _series(mat.k, smat.k)
    body = _make_body("rect_plate", smat, surface_size, 0.37, 0.41, modes=28, damping=6.0, seed=seed,
                      fmax=min(8000.0, 0.45 * sr))
    os_ = 2
    fs, n = sr * os_, samples(dur, sr) * os_
    v = np.clip(_curve(speed, n), 0.0, 50.0)
    off, load = _roll_stage(n, fs, v, radius, mass, roughness, float(max(joints, 0.0)), dents, asymmetry, seed)
    it = ImpactInteractor(body, mass=mass, k=k, mu=max(mat.mu, smat.mu), bright=bright)
    x_static = (mass * G / k) ** (1.0 / it.alpha)             # start resting on the surface
    y = it.run(dur, sr, y0=off[0] - x_static, active=True, offset=off, load=load, oversample=os_).audio
    return _out(y, sr, 0.75, 15.0)


@sfx("scrape", "Scraping / sliding: a surface profile read at the sliding speed x normal force, driving any modal body.",
     dur=(0.8, "s"), speed=(0.3, "m/s; number or list of breakpoints (a moving profile is what makes it expressive)"),
     force=(1.0, "normal force 0..1; number or list of breakpoints"),
     roughness=(0.5, "0..1 (1 = white, very rough; 0 = smooth); used when texture is empty"),
     texture=("", "'' | white|pink_half|pink|brown|gritty|sandpaper|grid|wood|metal|plastic"),
     grain=(0.1, "mm: size of the surface grain; smaller or faster = brighter"),
     body=("rect_plate", _SHAPE_HELP + " or a modal table (sy_vase, sy_sword1...)"),
     material=("wood", _MAT_HELP + ": the scraped body"), size=(0.4, "m: size of the body"),
     position=(0.3, "0..1 where it is scraped"), bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def scrape(sr=DEFAULT_SR, dur=0.8, speed=0.3, force=1.0, roughness=0.5, texture="", grain=0.1, body="rect_plate",
           material="wood", size=0.4, position=0.3, bright=0.5, seed=0):
    n = samples(float(np.clip(dur, 0.03, 120.0)), sr)
    # UNSOURCED: 8 1/s of extra damping while the scraper touches ("adjust the damping during contact", g06 B5).
    b = _make_body(body, material, size, position, modes=28, damping=8.0, seed=seed, fmax=min(10000.0, 0.45 * sr))
    prof = surface_texture(texture if texture else 2.0 * (1.0 - float(np.clip(roughness, 0.0, 1.0))), seed=seed)
    fade = np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(1.0, 0.005 * sr))
    f = scrape_force(prof, np.clip(_curve(speed, n), 0.0, 50.0), np.clip(_curve(force, n), 0.0, 10.0) * fade,
                     float(np.clip(grain, 0.005, 20.0)) * 1e-3, sr)
    return _out(modal_bank(f, b, sr, bright)[0], sr, 0.7, 15.0)


@sfx("smash", "Something brittle breaking on a hard floor: impact + noise burst + dense, random, thinning cloud of fragment impacts.",
     material=("glass", "glass|ceramic|ice|wood|plastic|stone... (metal reads as 'not breaking')"),
     size=(0.25, "m: size of the object; bigger = lower, heavier fragments"),
     shape=("rect_plate", "shape of the object before it breaks: rect_plate|circ_plate|bar"),
     energy=(1.0, "0.2..2 how hard it hits: louder, more fragments flying longer"),
     fragments=(8, "number of fragments (independent impact trains)"),
     dur=(1.6, "s until the last fragment settles"), burst=(0.5, "0..1 level of the 50 ms rupture noise"),
     bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def smash(sr=DEFAULT_SR, material="glass", size=0.25, shape="rect_plate", energy=1.0, fragments=8, dur=1.6,
          burst=0.5, bright=0.5, seed=0):
    mat = get_material(material)
    rng = np.random.default_rng(seed)
    energy = float(np.clip(energy, 0.05, 4.0))
    dur = float(np.clip(dur, 0.1, 30.0))
    n = samples(dur + 0.4, sr)
    parent = _make_body(shape, mat, size, 0.31, 0.43, modes=20, damping=4.0, seed=seed, fmax=min(12000.0, 0.45 * sr))
    k = _series(mat.k, MATERIALS["concrete"].k)
    # 1. the object hits the floor (its own mass against a rigid floor: extreme mass ratio, g05 §2)
    it = ImpactInteractor(parent, mass=float(np.clip(parent.mass, 1e-4, 50.0)), k=k, mu=max(mat.mu, 0.2), bright=bright)
    t_break = 0.012                                         # UNSOURCED: the parent rings 12 ms, then is gone
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
    frac = rng.uniform(0.2, 0.7, nf)                        # UNSOURCED: fragment size 20-70 % of the parent
    f1 = np.minimum(parent.freqs.min() / frac ** 2, 3000.0 + 3000.0 * bright)    # plate: f ~ h/L^2
    ratio = parent.freqs / parent.freqs.min()
    dratio = material_damping(ratio, mat, ext=0.0, seed=seed) / max(mat.tan_phi * math.pi, 1e-9)   # d/f1 per mode
    g = np.array([parent.gains(rng.uniform(0.1, 0.9, 2) if isinstance(parent, ShapeBody) and parent.shape.startswith("rect")
                               else float(rng.uniform(0.05, 0.95)), bright) for _ in range(16)])
    g /= np.max(np.abs(g)) + 1e-30
    tc = np.empty(len(t), np.int64)
    for j in range(nf):
        sel = tr == j
        tc[sel] = _hertz_samples(parent.mass * frac[j] ** 2, k, 1.5, max(mat.mu, 0.2), v[sel], sr)
    cloud = _event_bank(n, float(sr), np.round((t + t_break) * sr).astype(np.int64), v * frac[tr],
                        f1[tr], f1[tr] * mat.tan_phi * math.pi, np.full(len(t), mat.ext + 25.0), tc,
                        rng.integers(0, len(g), len(t)), ratio, dratio, g)
    out += 0.9 * cloud / (np.max(np.abs(cloud)) + 1e-30)
    return _out(out, sr, 0.85)


_CRUMPLE = {
    # f_min, f_max (Hz), tau_min, tau_max (s, 1/e time of a facet), closing low-pass, events x, rate x
    # Sourced for the can: the 500-1400 Hz closing low-pass and the E_tot / gamma / density ranges
    # (g05 §4.3-4.4). UNSOURCED: every f / tau range and the paper / bottle / foil / bag scalings
    # ("no values for f_MIN, f_MAX, tau_MIN, tau_MAX are printed"; "paper / bottle provided s is reshaped").
    "can": (700.0, 3600.0, 0.004, 0.030, True, 1.0, 1.0),
    "bottle": (300.0, 2200.0, 0.006, 0.050, True, 1.0, 1.0),
    "paper": (1200.0, 5200.0, 0.0008, 0.005, False, 6.0, 8.0),
    "foil": (1800.0, 6000.0, 0.002, 0.012, False, 4.0, 6.0),
    "bag": (900.0, 4200.0, 0.0006, 0.004, False, 6.0, 10.0),
}


@sfx("crumple", "Crumpling / crushing: stochastic impact bursts with power-law energies; facets shrink, so pitch drifts up as it is crushed.",
     kind=("can", "can|paper|bottle|foil|bag"),
     energy=(0.5, "0..1 force of the crushing: 0 soft and even (gamma -1.15) .. 1 hard, peaky (gamma -1.5)"),
     density=(0.5, "0..1 event density = softness of the material: stiff (1 event/s) .. soft (50 events/s)"),
     size=(0.5, "0..1 size of the object = energy budget (0.001 .. 0.1): how long it goes on"),
     dur=(0.0, "s cap, 0 = until the energy budget is spent"), bright=(0.5, "0 dark .. 1 bright"), seed=(0, "variation"))
def crumple(sr=DEFAULT_SR, kind="can", energy=0.5, density=0.5, size=0.5, dur=0.0, bright=0.5, seed=0):
    if kind not in _CRUMPLE:
        raise ValueError(f"unknown crumple kind {kind!r}; choose from {sorted(_CRUMPLE)}")
    fmin, fmax, tmin, tmax, close, nx, rx = _CRUMPLE[kind]
    fmax = min(fmax * (0.6 + 0.8 * float(np.clip(bright, 0, 1))), 0.4 * sr)
    rng = np.random.default_rng(seed)
    gamma = -1.15 - 0.35 * float(np.clip(energy, 0.0, 1.0))
    rate = rx * 10 ** (math.log10(50.0) * float(np.clip(density, 0.0, 1.0)))
    e_tot = nx * 10 ** (-3.0 + 2.0 * float(np.clip(size, 0.0, 1.0)))
    ev = crumple_events(e_tot, gamma, rate, seed=seed, n_max=6000)
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
                    np.maximum(np.round(sr * (0.00015 + 0.0005 * facet)), 2).astype(np.int64),   # UNSOURCED contact length
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


@sfx("avalanche", "Avalanche / rock slide rumble: enveloped noise ringing a cluster of random low modes; never the same twice (seed).",
     dur=(10.0, "s until all sound has ended"), fmin=(100.0, "Hz lowest rumbling frequency"),
     fmax=(150.0, "Hz highest (raise it for wind-like or eerie tones)"), damping=(0.87, "1/s decay rate of the modes"),
     modes=(25, "number of random modes"), rise=(0.0, "s to full level (0 = 12 % of dur)"),
     hold=(0.0, "s when the fade starts (0 = 30 % of dur)"), fade=(0.0, "s when the slide itself has ended (0 = 60 % of dur)"),
     seed=(0, "which modes"))
def avalanche(sr=DEFAULT_SR, dur=10.0, fmin=100.0, fmax=150.0, damping=0.87, modes=25, rise=0.0, hold=0.0, fade=0.0, seed=0):
    return _rumble(sr, dur, fmin, fmax, damping, modes, rise, hold, fade, seed, (0.12, 0.30, 0.60))


@sfx("rumble", "Low rumble (earthquake, heavy machinery, distant collapse): enveloped noise into random low modes.",
     dur=(4.0, "s"), fmin=(35.0, "Hz"), fmax=(90.0, "Hz"), damping=(3.0, "1/s decay rate of the modes: lower = more tonal, slower"),
     modes=(16, "number of random modes"), rise=(0.0, "s to full level (0 = 10 % of dur)"),
     hold=(0.0, "s when the fade starts (0 = 70 % of dur)"), fade=(0.0, "s when the drive has ended (0 = 90 % of dur)"),
     seed=(0, "which modes"))
def rumble(sr=DEFAULT_SR, dur=4.0, fmin=35.0, fmax=90.0, damping=3.0, modes=16, rise=0.0, hold=0.0, fade=0.0, seed=0):
    return _rumble(sr, dur, fmin, fmax, damping, modes, rise, hold, fade, seed, (0.10, 0.70, 0.90))


@sfx("modal_engine", "Four-stroke engine as a looped force (intake noise, silence, combustion 1/f burst, exhaust noise) driving a low plate body: motorcycle, mower, chainsaw.",
     rpm=(1200, "crank RPM; number or list of breakpoints (one cycle = 2 revolutions)"), dur=(2.0, "s"),
     cylinders=(1, "count, evenly spaced over the cycle"), intake=(0.5, "gain of the intake stroke"),
     combustion=(1.0, "gain of the combustion stroke"), exhaust=(0.7, "gain of the exhaust stroke"),
     cyl_spread=(0.35, "0..1 gain differences between cylinders (needed for a convincing multi-cylinder)"),
     fan=(0.15, "level of the pitched background (fan / rotating parts)"),
     body_freq=(50.0, "Hz lowest body resonance: 50 motorcycle .. 300+ with high damping = lawn mower"),
     damping=(5.56, "1/s external damping of the body"), modes=(13, "body modes (about 7 suffice)"),
     bright=(0.5, "0 dark .. 1 bright (0.5 = velocity tap, the thesis's choice for engines)"), seed=(0, "variation"))
def modal_engine(sr=DEFAULT_SR, rpm=1200, dur=2.0, cylinders=1, intake=0.5, combustion=1.0, exhaust=0.7,
                 cyl_spread=0.35, fan=0.15, body_freq=50.0, damping=5.56, modes=13, bright=0.5, seed=0):
    """van den Doel thesis Fig. 6.1 and model file `rectplate` (aspect 7.0528, 13 modes, base 50 Hz); g06 B7."""
    rng = np.random.default_rng(seed)
    n = samples(float(np.clip(dur, 0.05, 120.0)), sr)
    L = 4096
    q = L // 4
    tq = np.arange(q) / q
    cyc = np.zeros(L)
    cyc[:q] = max(float(intake), 0.0) * rng.uniform(-1, 1, q) * np.exp(-0.5 * ((tq - 0.5) / 0.18) ** 2)   # Gaussian bell
    comb = fractal_noise(q, 1.0, 1.0, f_low=2.0 / q, seed=seed + 1)                                      # 1/f burst
    cyc[2 * q:3 * q] = max(float(combustion), 0.0) * comb / (np.max(np.abs(comb)) + 1e-30) * np.minimum(tq / 0.03, 1.0) * np.exp(-tq / 0.25)
    cyc[3 * q:] = max(float(exhaust), 0.0) * rng.uniform(-1, 1, q) * np.exp(-tq / 0.12)                   # fast decay
    phase = np.cumsum(np.clip(_curve(rpm, n), 30.0, 30000.0) / 120.0) / sr
    nc = int(np.clip(cylinders, 1, 16))
    gains = 1.0 + float(np.clip(cyl_spread, 0.0, 1.0)) * rng.uniform(-0.8, 0.8, nc) if nc > 1 else np.ones(1)
    force = np.zeros(n)
    for c in range(nc):
        force += gains[c] * np.interp((phase + c / nc) % 1.0 * L, np.arange(L + 1), np.append(cyc, cyc[0]))
    if fan > 0:        # "a short noisy note of a 70 cm pipe": narrow noise at c/(2 x 0.7 m) = 245 Hz
        force += float(fan) * 4.0 * F.bandpass(rng.uniform(-1, 1, n), 245.0, sr, q=12.0)
    body = ShapeBody("rect_plate", float(np.clip(body_freq, 15.0, 2000.0)), material="metal", aspect=7.0528,
                     position=(0.31, 0.43), n_modes=int(np.clip(modes, 1, 60)), fmax=min(8060.745, 0.45 * sr),
                     ext=float(np.clip(damping, 0.0, 2000.0)), seed=seed)
    return _out(modal_bank(force, body, sr, bright)[0], sr, 0.8, 20.0)
