"""Interior ballistics, muzzle blast, bullet shock wave, suppressor, ground and recording chain, in absolute pascals.

Pure functions, no effect registration. A cartridge is a row of ``CARTRIDGES``; a weapon is a cartridge and a barrel length.
Nothing here is normalised to a peak: every amplitude is a pressure that a microphone would read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt

from .physical import ambience as amb
from .physical import space
from . import presets
from .physics import blast_wave
from .presets import F

RATE = 48000
GAMMA, R_GAS, K_DIR = 1.25, 340.0, 5.17
P0 = 101325.0
TNT_J = 4.184e6
C_AIR = space.C_AIR
DISCHARGE = 0.58
JET_SPEED = 1100.0
BLAST_SHAPE_DB = (0.0, 0.1, -0.6, -0.9, -2.6, -4.0, -11.1, -12.2, -16.1, -19.3, -23.6, -26.8, -30.0)
BRAKE_A_DB = 10.0
SUPPRESSOR_DEFAULT_DB = 25.0
SUPPRESSOR_T0_CAP = 10e-3
EJECTION_PORT_MAX_DB = 14.0
PRECURSOR_GAIN, PRECURSOR_T0, PRECURSOR_LEAD = 0.12, 0.3, 6.0
JET_RMS, JET_Q, JET_TAUS = 0.03, 0.7, 5.0
CAN_TONE_DB = -12.0
REF_DISTANCE = 3.0
NEAR_EXPONENT, FAR_EXPONENT, NEAR_DIAMETERS = 0.87, 0.75, 1000.0


@dataclass(frozen=True)
class Cartridge:
    d_mm: float
    l_mm: float
    m_g: float
    v_ref: float
    l_ref: float
    v_c_cm3: float
    m_c_g: float
    p_max_mpa: float
    trim_db: float = 0.0
    supp_db: float | None = None
    bore_mm: float = 0.0
    pellets: int = 1
    b: float = 0.88
    t_flame: float = 2800.0


_CART_KEYS = frozenset(Cartridge.__dataclass_fields__)
CARTRIDGES: dict[str, Cartridge] = presets.schema("cartridge", {
    "d_mm": F(float, lo=1.0, hi=60.0, unit="mm"), "l_mm": F(float, lo=1.0, hi=300.0, unit="mm"), "m_g": F(float, lo=0.1, hi=500.0, unit="g"),
    "v_ref": F(float, lo=50.0, hi=2000.0, unit="m/s"), "l_ref": F(float, lo=0.02, hi=3.0, unit="m"),
    "v_c_cm3": F(float, lo=0.05, hi=200.0, unit="cm3"), "m_c_g": F(float, lo=0.01, hi=200.0, unit="g"),
    "p_max_mpa": F(float, lo=1.0, hi=1000.0, unit="MPa"), "trim_db": F(float, lo=-40.0, hi=40.0, unit="dB", required=False),
    "supp_db": F(float, lo=0.0, hi=60.0, unit="dB", required=False), "bore_mm": F(float, lo=0.0, hi=60.0, unit="mm", required=False),
    "pellets": F(int, lo=1, hi=64, required=False), "b": F(float, lo=0.1, hi=2.0, required=False),
    "t_flame": F(float, lo=300.0, hi=5000.0, unit="K", required=False),
    "legacy_peak_pa": F(float, lo=1.0, hi=1e5, unit="Pa", required=False), "legacy_dir_db": F(float, lo=0.0, hi=60.0, unit="dB", required=False),
    "legacy_e_j": F(float, lo=1.0, hi=1e6, unit="J", required=False),
}, shape=lambda d: Cartridge(**{k: v for k, v in d.items() if k in _CART_KEYS}))


def _area(c: Cartridge) -> float:
    return math.pi * ((c.bore_mm or c.d_mm) * 1e-3) ** 2 / 4.0


def _phi(c: Cartridge, length: float) -> float:
    vc = c.v_c_cm3 * 1e-6
    return 1.0 - (vc / (vc + _area(c) * length)) ** (GAMMA - 1.0)


def muzzle_velocity(cart: str, barrel_m: float) -> float:
    """Speed at the muzzle for a barrel length, from the adiabatic expansion of the burnt gas, calibrated on ``v_ref`` at ``l_ref``."""
    c = CARTRIDGES[cart]
    return c.v_ref * math.sqrt(_phi(c, barrel_m) / _phi(c, c.l_ref))


def muzzle_state(cart: str, barrel_m: float) -> dict:
    """Gas state when the bullet leaves: ``v``, ``p_m`` (Pa), ``u_m`` (J), ``e_b`` (J), ``w_tnt`` (kg), ``r0`` (m),
    ``tau_b`` (s, time to empty the barrel), ``f_jet`` (Hz), ``t_gas`` (K), ``e_gas`` (J, kinetic energy of the gas)."""
    c = CARTRIDGES[cart]
    m, mc, a = c.m_g * 1e-3, c.m_c_g * 1e-3, _area(c)
    v = muzzle_velocity(cart, barrel_m)
    u0 = (0.5 * m * c.v_ref ** 2 + mc * c.v_ref ** 2 / 6.0) / _phi(c, c.l_ref)
    vc = c.v_c_cm3 * 1e-6
    vol = vc + a * barrel_m
    um = u0 * (1.0 - _phi(c, barrel_m))
    pm = (GAMMA - 1.0) * um / vol
    tm = c.t_flame * (vc / vol) ** (GAMMA - 1.0)
    eg = mc * v * v / 6.0
    eb = um + eg
    cg = math.sqrt(GAMMA * R_GAS * tm)
    dj = (c.bore_mm or c.d_mm) * 1e-3 * (pm / P0) ** 0.4
    return {"cart": cart, "barrel": barrel_m, "v": v, "p_m": pm, "u_m": um, "e_b": eb, "e_gas": eg, "w_tnt": eb / TNT_J,
            "r0": (eb / P0) ** (1.0 / 3.0), "tau_b": vol / (a * DISCHARGE * cg), "f_jet": 0.2 * JET_SPEED / dj, "t_gas": tm}


def _dir_gain(azimuth: float, a_db: float) -> float:
    s = float(np.interp(azimuth, np.arange(0.0, 181.0, 15.0), BLAST_SHAPE_DB))
    return 10.0 ** (s * (a_db / 30.0) / 20.0)


def blast_at(state: dict, cart: str, r: float, azimuth: float, a_db: float = 30.0, k_t0: float = 1.0, trim: bool = True) -> tuple[float, float, float]:
    """``(peak Pa, T0 s, b)`` of the Friedlander muzzle wave at ``r`` m and ``azimuth`` degrees from the bore axis.
    Up to 3 m the free-air blast table at the muzzle energy; beyond, the weak-shock law p r sqrt(ln r) = const with the
    positive phase growing as sqrt(ln r). ``a_db``: front-to-back spread of the directivity; ``k_t0``: scale of T0;
    ``trim``: the per-cartridge correction on the axis level."""
    c = CARTRIDGES[cart]
    r = max(float(r), 0.05)
    ps, t0, _ = blast_wave(state["w_tnt"], min(r, REF_DISTANCE))
    if r > REF_DISTANCE:
        r0 = state["r0"]
        lr, l3 = math.log(r / r0), math.log(REF_DISTANCE / r0)
        ps *= (REF_DISTANCE / r) * math.sqrt(l3 / lr)
        t0 *= math.sqrt(lr / l3)
    peak = K_DIR * ps * (10.0 ** (c.trim_db / 20.0) if trim else 1.0) * _dir_gain(azimuth, a_db)
    return peak, t0 * k_t0, c.b


def friedlander(peak: float, t0: float, b: float, n: int | None, rate: int = RATE) -> np.ndarray:
    """p = peak (1 - t/t0) exp(-b t/t0), first sample at the peak. ``n`` None: up to t0 (1 + 8/b)."""
    if n is None:
        n = int(t0 * (1.0 + 8.0 / b) * rate) + 1
    u = np.arange(n) / (rate * t0)
    return peak * (1.0 - u) * np.exp(-b * u)


def precursor(state: dict, cart: str, r: float, azimuth: float, barrel_m: float, v: float, rate: int = RATE) -> tuple[np.ndarray, float]:
    """The column of air pushed out ahead of the bullet: ``(signal Pa, lead s)``; the pulse arrives ``lead`` before the main blast."""
    if azimuth >= 90.0:
        return np.zeros(1), 0.0
    pk, t0, b = blast_at(state, cart, r, 0.0)
    amp = PRECURSOR_GAIN * pk * math.cos(math.radians(azimuth)) ** 2
    return friedlander(amp, PRECURSOR_T0 * t0, b, None, rate), barrel_m / (PRECURSOR_LEAD * v)


def gas_jet(state: dict, peak: float, rng: np.random.Generator, rate: int = RATE) -> np.ndarray:
    """Turbulent jet leaving the barrel: band noise at the jet Strouhal peak, e^(-t/tau_b) over 5 tau_b, rms 3 % of ``peak``."""
    tau = state["tau_b"]
    n = max(int(JET_TAUS * tau * rate), 16)
    w = amb.bp(rng.uniform(-1.0, 1.0, n), min(state["f_jet"], 0.45 * rate), JET_Q) * np.exp(-np.arange(n) / (rate * tau))
    return w * (JET_RMS * peak / math.sqrt(float(np.mean(w * w)) + 1e-30))


def brake_pulses(peak: float, t0: float, b: float, rate: int = RATE) -> list[tuple[np.ndarray, float]]:
    """Two port pulses of a muzzle brake: ``[(signal, delay s), ...]`` at +0.15 and +0.30 ms, gains 0.4 and 0.25."""
    return [(friedlander(g * peak, t0, b, None, rate), d) for g, d in ((0.4, 0.15e-3), (0.25, 0.30e-3))]


def brake_boost_db(azimuth: float) -> float:
    """A brake sends 3 dB more sideways and back (90 to 165 degrees); use with ``a_db=BRAKE_A_DB``."""
    return 3.0 if 90.0 <= azimuth <= 165.0 else 0.0


def flash_pulse(peak: float, t0: float, b: float, amount: float, rate: int = RATE) -> tuple[np.ndarray, float]:
    """Secondary flash: a second Friedlander, 2 ms later, 0.5 ``amount`` of the peak, twice as long."""
    return friedlander(0.5 * amount * peak, 2.0 * t0, b, None, rate), 2e-3


def suppress(peak: float, t0: float, cart: str, reduction_db: float | None = None, ejection_port: bool = False) -> tuple[float, float]:
    """Peak and T0 after a suppressor: the peak drops by D, T0 grows as 10^(D/40) (half the impulse is kept), capped at 10 ms.
    ``ejection_port``: gas leaves through the open action, D limited to 14 dB."""
    c = CARTRIDGES[cart]
    d = reduction_db if reduction_db is not None else (c.supp_db if c.supp_db is not None else SUPPRESSOR_DEFAULT_DB)
    if ejection_port:
        d = min(d, EJECTION_PORT_MAX_DB)
    return peak * 10.0 ** (-d / 20.0), min(t0 * 10.0 ** (d / 40.0), SUPPRESSOR_T0_CAP)


def can_warmup_db(k: int) -> float:
    """Extra level of the first (+3 dB) and second (+1.5 dB) shot through a cold suppressor."""
    return (3.0, 1.5)[k] if 0 <= k < 2 else 0.0


def helmholtz_freq(area: float, volume: float, l_eff: float, t_gas: float = 1200.0) -> float:
    """Frequency of the last suppressor chamber seen as a resonator: c/(2 pi) sqrt(A / (V L))."""
    return math.sqrt(GAMMA * R_GAS * t_gas) / (2.0 * math.pi) * math.sqrt(area / (volume * l_eff))


def can_tone(peak: float, rng: np.random.Generator, rate: int = RATE, f: float = 1100.0, q: float = 4.0) -> np.ndarray:
    """Ring of the suppressor chamber after the shot, its peak 12 dB under ``peak`` (the treated blast peak)."""
    n = int(8.0 * q / (math.pi * f) * rate)
    w = amb.bp(np.append(rng.uniform(-1.0, 1.0, 8), np.zeros(n - 8)), f, q)
    return w * (10.0 ** (CAN_TONE_DB / 20.0) * peak / (float(np.max(np.abs(w))) + 1e-30))


def ground(x: np.ndarray, r: float, h_s: float, h_l: float, ground: str, rate: int = RATE, temp_c: float = 20.0) -> np.ndarray:
    """Direct plus ground-reflected path for a source and a listener ``r`` m apart at heights ``h_s`` and ``h_l``
    (``ground``: a name of ``reverbs`` or a flow resistivity; "none" returns ``x``). Returns ``x`` padded by 50 ms."""
    x = np.asarray(x, float)
    if str(ground) == "none":
        return x
    from .reverbs import _ground_sigma
    nf = 1 << int(np.ceil(np.log2(x.shape[0] + int(0.05 * rate))))
    h = space.ground_effect(np.fft.rfftfreq(nf, 1.0 / rate), float(r), float(h_s), float(h_l), _ground_sigma(ground), temp_c)
    return np.fft.irfft(np.fft.rfft(x, nf) * h, nf)[:x.shape[0] + int(0.05 * rate)]


def air_absorb(x: np.ndarray, r: float, rate: int = RATE, ref: float = REF_DISTANCE, rh: float = 50.0, temp_c: float = 20.0) -> np.ndarray:
    """Absorption of the air between ``ref`` and ``r`` metres only (minimum phase, no 1/r: the pressures already carry the distance)."""
    x = np.asarray(x, float)
    extra = max(float(r) - ref, 0.0)
    if extra <= 0.0:
        return x
    nf = 1 << int(np.ceil(np.log2(x.shape[0] + int(0.02 * rate))))
    lm = -space.air_db_per_m(np.fft.rfftfreq(nf, 1.0 / rate), rh, temp_c) * extra * math.log(10.0) / 20.0
    cep = np.fft.irfft(lm, nf)
    cep[1:nf // 2] *= 2.0
    cep[nf // 2 + 1:] = 0.0
    return np.fft.irfft(np.fft.rfft(x, nf) * np.exp(np.fft.rfft(cep)), nf)[:x.shape[0]]


RECORDERS = {"none": (None, None, None), "field": (5.0, None, 172.0), "camera": (80.0, None, 135.0), "phone": (200.0, 7000.0, 120.0)}


def recorder(x_pa: np.ndarray, kind: str, clip_db: float | None = None, rate: int = RATE) -> np.ndarray:
    """Pressure as a recording chain gives it back, still in pascals: ``none`` (bare), ``field`` (measurement microphone, high-pass 5 Hz),
    ``camera`` (high-pass 80 Hz, soft clip at ``clip_db`` dB SPL, coupling-capacitor lock, limiter 1 ms / 300 ms), ``phone``
    (200 Hz to 7 kHz, same clip and limiter). Nothing is scaled to the file's peak."""
    x = np.asarray(x_pa, float)
    if kind not in RECORDERS:
        raise ValueError(f"recorder must be one of {', '.join(RECORDERS)}")
    hp, lp, clip = RECORDERS[kind]
    if hp is None:
        return x.copy()
    clip = clip if clip_db is None else float(clip_db)
    y = sosfilt(butter(2 if kind == "phone" else 1, hp, "high", fs=rate, output="sos"), x)
    if lp is not None:
        y = sosfilt(butter(2, lp, "low", fs=rate, output="sos"), y)
    pc = 20e-6 * 10.0 ** (clip / 20.0)
    z = pc * np.tanh(y / pc)
    if kind == "field":
        return z
    over = np.where(np.abs(y) > pc, pc * np.tanh((y - z) / pc), 0.0)
    idx = np.flatnonzero(over)
    if idx.size:
        a = math.exp(-1.0 / (0.03 * rate))
        s, cur, last = np.zeros_like(z), 0.0, 0
        for k, i in enumerate(idx):
            cur *= a ** (i - last)
            if abs(0.3 * over[i]) > abs(cur):
                cur = 0.3 * over[i]
            last = i
            j = idx[k + 1] if k + 1 < idx.size else z.shape[0]
            s[i:j] = cur * a ** np.arange(j - i)
        z = z - s
    # ponytail: running maximum through a^-k, fine to about 10 s at 48 kHz; chunk it for longer takes
    a_r = math.exp(-1.0 / (0.3 * rate))
    k = np.arange(z.shape[0])
    env = np.maximum.accumulate(np.abs(z) * a_r ** -k) * a_r ** k
    g = np.minimum(1.0, 0.5 * pc / np.maximum(env, 1e-30))
    a_a = math.exp(-1.0 / (0.001 * rate))
    b_, a_ = [1.0 - a_a], [1.0, -a_a]
    return z * lfilter(b_, a_, g, zi=lfilter_zi(b_, a_) * g[0])[0]


def half_sine_force(impulse: float, t_c: float, n: int, rate: int = RATE) -> np.ndarray:
    """Force (N) of a half-sine contact of impulse ``impulse`` (N s) lasting ``t_c`` s, each sample the mean over its interval
    (spectrum J cos(pi f t_c) / (1 - (2 f t_c)^2), first zero at 1.5 / t_c); the samples add up to ``impulse * rate`` whatever ``t_c``."""
    cum = 0.5 * impulse * (1.0 - np.cos(math.pi * np.clip(np.arange(n + 1) / (rate * t_c), 0.0, 1.0)))
    return np.diff(cum) * rate


def n_wave_pa(v: float, d_mm: float, l_mm: float, miss: float, c: float = C_AIR, near: bool = True) -> tuple[float, float]:
    """``(peak Pa, duration s)`` of the N-wave of a supersonic bullet ``miss`` m from its path; ``(0, 0)`` when subsonic.
    Closer than 1000 diameters the peak follows miss^-0.87 from its value there (``near=False``: the far-field law throughout)."""
    m = v / c
    if m <= 1.0:
        return 0.0, 0.0
    d, l, b, k = d_mm * 1e-3, l_mm * 1e-3, max(float(miss), 0.05), m * m - 1.0

    def far(bb):
        return 0.53 * P0 * k ** 0.125 * d / (bb ** FAR_EXPONENT * l ** 0.25)
    b1 = NEAR_DIAMETERS * d
    p = far(b1) * (b1 / b) ** NEAR_EXPONENT if (near and b < b1) else far(b)
    return p, 1.82 * m * b ** 0.25 * d / (c * k ** 0.375 * l ** 0.25)


def emission_speed(v0: float, x: float, y: float, lam: float, c: float = C_AIR) -> tuple[float, float]:
    """Speed and position ``(v_e, x_e)`` of the bullet where it emitted the cone that reaches a listener ``x`` m down the path and ``y`` m off it,
    for a speed decaying as v0 e^(-x/lam). ``(0, 0)`` when the cone never gets there or the bullet is subsonic by then."""
    v = v0
    for _ in range(4):
        if v <= c:
            return 0.0, 0.0
        v = v0 * math.exp(-max(x - y * math.tan(math.asin(c / v)), 0.0) / lam)
    if v <= c:
        return 0.0, 0.0
    xe = x - y * math.tan(math.asin(c / v))
    return (v, xe) if xe >= 0.0 else (0.0, 0.0)


def n_wave_shape(dur: float, rate: int = RATE) -> np.ndarray:
    """Rise in one sample, linear down to -1, back in one sample."""
    return np.append(np.linspace(1.0, -1.0, max(2, int(round(dur * rate)))), 0.0)


def n_wave(v0: float, d_mm: float, l_mm: float, x: float, y: float, lam: float, rate: int = RATE, c: float = C_AIR) -> tuple[np.ndarray, float | None]:
    """N-wave at a listener ``x`` m along the path and ``y`` m off it, with the bullet slowing down: ``(signal Pa, arrival s after the shot)``,
    ``(zeros(1), None)`` outside the cone."""
    ve, xe = emission_speed(v0, x, y, lam, c)
    if ve <= c:
        return np.zeros(1), None
    p, dur = n_wave_pa(ve, d_mm, l_mm, y, c)
    return p * n_wave_shape(dur, rate), lam / v0 * math.expm1(xe / lam) + y / (c * math.cos(math.asin(c / ve)))
