"""Aircraft and rockets: spec reports/sons/aeronefs_eau.md sections 3-5, every number below comes from it unless marked.

- Sonic boom: Carlson, NASA TP-1122, 1978, eqs 1-2 (worked problem 1: 74 Pa, 0.17 s, reproduced by test_aero).
- Propellers: `physical.modal.propeller_tone` (Selfridge 2017) with `cyber.pipe_engine`.
- Rotors: Farnell, Designing Sound, Practical "Helicopter"; line numbers of research/root_text/content.txt.
- Jet: Lighthill U^8 and U^3 (Eldred, NASA SP-8072, 1971). Tam, Golebiowski & Seiner 1996, eqs 5-6 for the two
  similarity spectra, their peak Strouhal per angle from Tam et al. 2008, Fig 7(b).
- Rockets: Eldred SP-8072 (efficiency, St 0.02, 50 degrees, Fig 5 spectrum) and Kellison & Gee, JASA EL 3, 2023.
- Drone: DJI Phantom II measurements, AIAA 2016-2873.
Mixing weights between the parts of a machine are estimated throughout, and nothing here was compared with a recording.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy.signal import fftconvolve

from . import cyber, foley, klang, plates
from . import matter as Ma
from .core import DEFAULT_SR
from .physical import modal
from .reverbs import distance as _distance
from .reverbs import doppler, occlude
from .sfx import sfx

SR = Ma.SR
C = Ma.C_AIR
P0 = 101325.0


def _end(y, sr, peak=0.8, fin=0.01, fout=0.05, dc=True):
    """Like `matter._fin` but keeps the infrasound (spec: do not filter under 20 Hz): only DC goes, below 3 Hz.
    ``peak=None`` keeps the model's level, so that a sourced law stays measurable; only clipping is scaled down."""
    y = np.where(np.isfinite(y), y, 0.0)
    if dc:
        y = Ma._hp(y, 3.0)
    sr = int(sr)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        y = resample_poly(y, sr // g, SR // g)
    m = float(np.max(np.abs(y)))
    if peak is not None and m > 1e-12:
        y = y * (peak / m)
    elif m > 0.98:
        y = y * (0.98 / m)
    a, b = min(max(2, int(fin * sr)), len(y) // 2), min(max(2, int(fout * sr)), len(y) // 2)
    y[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    y[-b:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(b) / b)
    return y


def _rms1(x):
    return x / (math.sqrt(float(np.mean(x * x))) + 1e-12)


def _pass(y, speed, closest, ground=0.35):
    """Straight pass: `reverbs.doppler`, ground comb 0.35 (F:22496-22497), mono."""
    return doppler(y, SR, speed=float(speed), closest=float(closest), pan=False, ground=float(ground), height=3.0)


# ======================================================================== sonic boom
def isa_atmosphere(h: float):
    """(pressure Pa, speed of sound m/s) at altitude ``h`` m. Standard atmosphere to 20 km, constants not in the spec:
    estimated here, but they reproduce Carlson's worked problem."""
    h = float(h)
    if h <= 11000.0:
        t = 288.15 - 0.0065 * h
        return P0 * (t / 288.15) ** 5.2559, 20.05 * math.sqrt(t)
    return 22632.0 * math.exp(-(h - 11000.0) / 6341.6), 20.05 * math.sqrt(216.65)


def carlson_boom(mach, altitude, length, ks=0.087, kp=1.10, kt=0.85, kr=1.8):
    """Carlson 1978 eqs 1-2: (peak overpressure Pa, duration s). KR 1.8-2.0, ground reflection."""
    pv, av = isa_atmosphere(altitude)
    k = mach * mach - 1.0
    if k <= 0:
        return 0.0, 0.0
    dp = kp * kr * math.sqrt(pv * P0) * k ** 0.125 * altitude ** -0.75 * length ** 0.75 * ks
    dt = kt * 3.42 / av * mach / k ** 0.375 * altitude ** 0.25 * length ** 0.75 * ks
    return dp, dt


PA_FULL = 200.0     # Pa at digital full scale, so that the level of a boom stays readable in the file


@sfx("sonic_boom", "Sonic boom: the N-wave at the ground from Carlson's simplified prediction. Heard indoors with interior=1 (through the glass, panes rattling).",
     mach=(1.99, "flight Mach number > 1"), altitude=(14480.0, "m"), length=(32.7, "m aircraft length"),
     ks=(0.087, "Carlson shape factor"), kp=(1.10, "pressure factor"), kt=(0.85, "duration factor"),
     rise_ms=(3.0, "rise time, 1-5 ms measured, about 3.7 x longer in moderate turbulence"),
     interior=(0, "1 = heard inside a room"), seed=(0, "variation"))
def sonic_boom(sr=DEFAULT_SR, mach=1.99, altitude=14480.0, length=32.7, ks=0.087, kp=1.10, kt=0.85, rise_ms=3.0, interior=0, seed=0):
    dp, dt = carlson_boom(max(float(mach), 1.001), max(float(altitude), 100.0), max(float(length), 1.0), ks, kp, kt)
    n = int((dt + 0.6 + (1.2 if interior else 0.0)) * SR)
    m = int(dt * SR)
    y = np.zeros(n)
    y[int(0.02 * SR):int(0.02 * SR) + m] = np.linspace(1.0, -1.0, m)
    h = np.hanning(max(8, int(float(np.clip(rise_ms, 0.2, 30.0)) * 1e-3 * SR)))
    y = np.convolve(y, h / h.sum(), mode="same") * dp / PA_FULL
    if interior:
        # Report: occlude + modal panes, all estimated. Pane 4 mm, 5 % leak round the frame.
        rng = Ma._rng(seed, 901)
        y = occlude(y, SR, material="glass", thickness=0.004, leak=0.05, edge=300.0)
        for s in (0.5, 0.8):
            pane = plates.glass_pane(sr=SR, size=s * rng.uniform(0.9, 1.1), thickness=4.0, decay=0.6, vel=0.5, hardness=0.3)
            pane = Ma._lp(pane, 3000.0, 2)
            pane[:96] *= np.hanning(192)[:96]
            pane[-480:] *= np.hanning(960)[480:]
            Ma._place(y, pane, int((dt * rng.uniform(0.2, 0.9) + 0.02) * SR), 0.15 * dp / PA_FULL)
    return _end(y, sr, peak=None, fin=0.002, fout=0.05, dc=False)      # an N-wave has no DC, and a high-pass tilts it


# ======================================================================== turbine partials
# Farnell's snapshot (Hz, level), Designing Sound fig 47.3, content.txt:22029-22071: the spectrum shifts linearly with speed.
TURBINE_14 = ((739, 1.0), (5464, 0.633), (2464, 0.346), (1641, 0.327), (1220, 0.283), (10948, 0.268), (448, 0.253),
              (1032, 0.230), (3673, 0.211), (2186, 0.201), (191, 0.193), (6105, 0.173), (4145, 0.172), (7110, 0.153))
TURBINE_5 = tuple(zip(klang._TURB_F.tolist(), klang._TURB_G.tolist()))


def _partials(table, speed, n, rng, fm=None):
    """Independent phases and a little clipping (Farnell: both matter). ``speed`` may be an array; ``fm`` scales frequency."""
    s = np.broadcast_to(np.asarray(speed, float), (n,))
    if fm is not None:
        s = s * fm
    y = np.zeros(n)
    for f, g in table:
        ph = np.cumsum(f * s) / SR + rng.random()
        y += g * np.sin(2.0 * np.pi * ph) * (f * s < 0.45 * SR)
    return 0.6 * np.tanh(0.8 * _rms1(y))


def _forced_flame(speed, n, rng):
    """Farnell fig 47.4: noise, band gently off round 8 kHz, a sweep v^2 x 150 Hz over a 120 Hz high-pass, x 100 and clipped,
    then a filter at v x 12 kHz, roll-off above 11 kHz."""
    v = np.broadcast_to(np.asarray(speed, float), (n,))
    x = Ma._bp(rng.uniform(-1, 1, n), 8000.0, 0.5)
    x = Ma._hp(Ma._tvbp(x, np.maximum(v * v * 150.0, 5.0), 1.0), 120.0)
    x = np.clip(x * 100.0, -1.0, 1.0)
    return Ma._lp(Ma._tvbp(x, np.clip(v * 12000.0, 20.0, 0.45 * SR), 0.6), 11000.0)


@sfx("jet_cycle", "Jet start-up and shut-down: spark igniter in the combustor, the turbine's 14 partials sliding with the speed, exhaust from hiss to roar.",
     phase=("cycle", "start | stop | cycle"), speed=(0.8, "peak speed 0..1 (1000..50000 rpm scale in Farnell)"),
     dur=(14.0, "s"), ignite=(1, "1 = igniter ticks in the start"), seed=(0, "variation"))
def jet_cycle(sr=DEFAULT_SR, phase="cycle", speed=0.8, dur=14.0, ignite=1, seed=0):
    rng = Ma._rng(seed, 902)
    n = Ma._n(dur)
    top = float(np.clip(speed, 0.05, 1.0))
    t = np.arange(n) / n
    # the shape of the gesture (Farnell: warm-up after ignition, then very slowly up; speed slugged by a low-pass).
    up = 0.25 * top * np.minimum(t / 0.25, 1.0) + 0.75 * top * np.clip((t - 0.3) / 0.7, 0.0, 1.0) ** 2
    if str(phase) == "start":
        s = up
    elif str(phase) == "stop":
        s = top * np.exp(-3.0 * t) * (1.0 - t)
    elif str(phase) == "cycle":
        s = np.where(t < 0.6, np.interp(t, np.linspace(0, 0.6, n), up), top * np.exp(-5.0 * (t - 0.6)) * (1.0 - (t - 0.6) / 0.4))
    else:
        raise ValueError(f"unknown phase {phase!r}; choose from start, stop, cycle")
    s = Ma._lp(s, 2.0)
    # Farnell: the turbine dominates at low speed and is quickly overwhelmed by the noise.
    y = _partials(TURBINE_14, s, n, rng) * np.minimum(s * 8.0, 1.0) * 0.5 + _forced_flame(s, n, rng) * 0.6 * s
    if ignite and str(phase) != "stop":
        # c12 (Farnell): the igniter alone gives the impulse response of the combustor. No value there: 4 ticks/s,
        # cavity modes and levels estimated.
        m = int(0.06 * SR)
        k = np.arange(m) / SR
        tick = (1.0 - np.exp(-k / 0.001)) * sum(g * np.exp(-k / 0.012) * np.sin(2.0 * np.pi * f * k) for f, g in ((420.0, 1.0), (1130.0, 0.7), (2300.0, 0.4)))
        tick[-48:] *= np.linspace(1.0, 0.0, 48)
        for at in np.arange(0.5, min(3.5, 0.3 * dur), 0.25):
            Ma._place(y, tick, int(at * SR), 0.5 * rng.uniform(0.6, 1.0))
    return _end(y, sr)


# ======================================================================== jet mixing noise
def jet_power(u, c=C):
    """Lighthill: acoustic power ~ U^8 below Mach 1, ~ U^3 above (SP-8072), joined at U = c. Relative units."""
    u = np.asarray(u, float) / c
    return np.where(u < 1.0, u ** 8, u ** 3)


def crackle(n, rng, drive=1.0, fc=800.0):
    """Non-linear steepening of noise, the mechanism named for jet crackle: an exponential skews the pressure, then each
    sample moves ahead in proportion to it, which turns the fronts into shocks (Earnshaw). Criteria, from a search extract
    only: pressure skewness > 0.4 (Ffowcs Williams), derivative skewness > 5 (BYU). fc, drive, shift estimated."""
    g = _rms1(Ma._lp(rng.standard_normal(n), fc, 2))
    p = np.expm1(0.8 * float(drive) * g)
    pos = np.maximum.accumulate(np.arange(n) - 40.0 * float(drive) * p)
    y = np.interp(np.arange(n), pos, p)
    return _rms1(Ma._lp(y - y.mean(), 9000.0))


def tam_spectra(x):
    """Similarity spectra of the two jet-noise components in dB, at x = f / f_peak: (10 log F, 10 log G); F(1) = G(1) = 1."""
    x = np.maximum(np.asarray(x, float), 1e-9)
    lx = np.log10(x)
    big = np.select([x >= 2.5, x >= 1.0, x >= 0.5],
                    [5.64174 - 27.7472 * lx, (1.06617 - 45.29940 * lx + 21.40972 * lx ** 2) * lx, -38.19338 * lx ** 2 - 16.91175 * lx ** 3],
                    2.53895 + 18.4 * lx)
    l10, l3 = lx - 1.0, np.log10(20.0 * x / 3.0)
    fine = np.select([x >= 30.0, x >= 10.0, x >= 1.0, x >= 0.15, x >= 0.05],
                     [29.77786 - 38.16739 * lx, -11.8 - (27.2523 + 0.8091863 * l10 + 14.851964 * l10 ** 2) * l10,
                      -(8.1476823 + 3.6523177 * lx) * lx ** 2, (-1.0550362 + 4.9774046 * lx) * lx ** 2,
                      -3.5 + (11.874876 + 2.1202444 * l3 + 7.5211814 * l3 ** 2) * l3],
                     9.9 + 14.91126 * lx)
    return big, fine


def jet_peak_strouhal(angle):
    """Peak Strouhal numbers (large scale, fine scale) of a hot Mach 1 jet, ``angle`` degrees from the jet axis downstream."""
    th = 180.0 - float(np.clip(angle, 0.0, 180.0))
    large = float(np.interp(th, (110.0, 120.0, 130.0, 140.0, 150.0), (0.31, 0.29, 0.21, 0.19, 0.17)))
    fine = float(np.interp(th, (50.0, 120.0), (0.18, 0.28)))
    return large, fine


def _shaped(x, fpeak, which):
    """White noise ``x`` given the spectrum ``which`` (0 large, 1 fine) peaking at ``fpeak`` Hz, unit RMS."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g = 10.0 ** (tam_spectra(np.maximum(f, 0.1) / max(float(fpeak), 1.0))[which] / 20.0)
    return _rms1(np.fft.irfft(np.fft.rfft(x) * g, n))


def _jet_mix(u, d, n, rng, angle):
    """Two components on their similarity spectra, each at its own peak Strouhal number for this angle; ``u`` is the jet speed,
    or (dry, reheat, on) to cross-fade two states sharing the same noise."""
    sl, sf = jet_peak_strouhal(angle)
    th = math.radians(float(np.clip(angle, 0.0, 180.0)))
    wl = max(math.cos(th), 0.0) ** 2
    zl, zf = rng.standard_normal(n), rng.standard_normal(n)

    def mix(v):
        return wl * _shaped(zl, sl * v / float(d), 0) + (1.0 - wl) * (0.25 + 0.75 * math.sin(th)) * _shaped(zf, sf * v / float(d), 1)
    if not isinstance(u, tuple):
        return mix(float(u))
    dry, wet, on = u
    return (1.0 - on) * mix(dry) + on * mix(wet) if dry != wet else mix(wet)


@sfx("jet_afterburner", "Military jet in reheat: mixing noise in two components, shock crackle and the flame, with the step when the afterburner lights.",
     dur=(5.0, "s"), velocity=(1100.0, "m/s jet speed in reheat"), diameter=(1.0, "m nozzle: St 0.2 peak"),
     dry=(0.65, "dry jet speed as a fraction of velocity"), ignite=(0.3, "fraction of dur when the reheat lights; < 0 = always on"),
     crackle_gain=(0.5, "0..1"), angle=(30.0, "degrees from the jet axis downstream: low, directional roar near 30, hissier to the side"),
     seed=(0, "variation"))
def jet_afterburner(sr=DEFAULT_SR, dur=5.0, velocity=1100.0, diameter=1.0, dry=0.65, ignite=0.3, crackle_gain=0.5, angle=30.0, seed=0):
    rng = Ma._rng(seed, 903)
    n = Ma._n(dur)
    uj = float(np.clip(velocity, 30.0, 3000.0))
    if ignite < 0:
        on = np.ones(n)
    else:
        i, w = int(float(ignite) * n), int(0.3 * SR)
        on = np.clip((np.arange(n) - i) / max(w, 1), 0.0, 1.0)
    u = uj * (float(dry) + (1.0 - float(dry)) * on)
    amp = np.sqrt(jet_power(u) / jet_power(1100.0))
    y = 0.2 * _jet_mix((uj * float(dry), uj, on), diameter, n, rng, angle) * amp
    g = float(np.clip(crackle_gain, 0.0, 1.0))
    if g > 0 and uj > C:
        y = y + 0.15 * g * crackle(n, rng) * on * amp
    y = y + 0.06 * Ma._fit(Ma.flame(sr=SR, kind="blowtorch", dur=n / SR, seed=seed), n) * on * amp
    return _end(y, sr, peak=None)


# ======================================================================== turbofan
# CFM56 (Wikipedia, read): N1 rpm, N2 rpm, fan diameter m, blades. 7B: 24 blades. 5B: 36 blades (estimated). N2 of the 7B estimated.
TURBOFANS = {"cfm56_7b": (5175.0, 14460.0, 1.55, 24), "cfm56_5b": (5000.0, 14460.0, 1.84, 36)}
BPR = 5.0       # bypass ratio; the jet falls as (1 + BPR)^-1


def tip_mach(rpm, diameter, c=C):
    return math.pi * float(diameter) * float(rpm) / 60.0 / c


def _turbofan_layers(engine, n1, rpm, n, rng):
    """(fan, jet). Fan: rotor-stator lines at the BPF and, once the tip is supersonic, the buzz-saw: harmonics of the shaft
    with amplitudes drawn at random, standing for blade-to-blade differences."""
    if engine not in TURBOFANS:
        raise ValueError(f"unknown engine {engine!r}; choose from {sorted(TURBOFANS)}")
    r1, r2, d, z = TURBOFANS[engine]
    r = (float(rpm) if rpm > 0 else r1) * float(n1)
    f = r / 60.0
    t = np.arange(n) / SR
    fan = np.zeros(n)
    for k, g in ((1, 1.0), (2, 0.45), (3, 0.2)):
        if k * z * f < 0.45 * SR:
            fan += g * np.sin(2.0 * np.pi * k * z * f * t + rng.uniform(0, 6.28)) * (1.0 + 0.05 * Ma._lfn(rng, n, 6.0))
    fan += 0.2 * _rms1(Ma._bp(rng.standard_normal(n), min(z * f, 0.4 * SR), 3.0))
    m = tip_mach(r, d)
    if m > 1.0:
        b = min((m - 1.0) / 0.2, 1.0)
        for k in range(1, int(min(0.45 * SR / f, 2.5 * z)) + 1):
            if k % z:
                a = b * 0.5 * np.exp(rng.normal(-0.7, 0.7)) * math.exp(-k / (1.5 * z))
                fan += a * np.sin(2.0 * np.pi * k * f * t + rng.uniform(0, 6.28))
    fan += 0.08 * _partials(TURBINE_5, r2 / 14460.0 * float(n1) * 0.6, n, rng)
    uj = 300.0 * float(n1)
    jet = Ma._tvbp(rng.standard_normal(n), np.full(n, max(Ma.STROUHAL * uj / (0.6 * d), 20.0)), 0.5)
    return _rms1(fan), _rms1(jet) * float(np.sqrt(jet_power(uj) / jet_power(300.0))) / (1.0 + BPR)


@sfx("turbofan", "Civil turbofan: fan tone at the blade rate, buzz-saw harmonics of the shaft once the tip is supersonic, a bypass-reduced jet.",
     engine=("cfm56_7b", "|".join(TURBOFANS)), n1=(1.0, "fan speed as a fraction of the table's"), rpm=(0.0, "fan rpm at n1 = 1 (0 = from the table)"),
     dur=(3.0, "s"), seed=(0, "variation: the blade-to-blade table"))
def turbofan(sr=DEFAULT_SR, engine="cfm56_7b", n1=1.0, rpm=0.0, dur=3.0, seed=0):
    n = Ma._n(dur)
    fan, jet = _turbofan_layers(str(engine), float(np.clip(n1, 0.1, 1.2)), float(rpm), n, Ma._rng(seed, 904))
    return _end(0.6 * fan + 1.2 * jet, sr)


@sfx("flyby_jet", "Jet passing: engine + Doppler + ground comb, the fan ahead of it and the jet roar behind.",
     engine=("turbofan", "turbofan | afterburner"), speed=(250.0, "m/s"), closest=(150.0, "m"), rpm=(0.0, "fan rpm (0 = CFM56-7B)"),
     dur=(8.0, "s"), ground=(0.35, "0..1 reflection strength"), seed=(0, "variation"))
def flyby_jet(sr=DEFAULT_SR, engine="turbofan", speed=250.0, closest=150.0, rpm=0.0, dur=8.0, ground=0.35, seed=0):
    rng = Ma._rng(seed, 905)
    n = Ma._n(dur)
    if str(engine) == "afterburner":
        front = _partials(TURBINE_5, 0.5, n, rng)
        rear = jet_afterburner(sr=SR, dur=n / SR, ignite=-1.0, angle=30.0, seed=seed)[:n]
        rear = _rms1(Ma._fit(rear, n))
    else:
        front, rear = _turbofan_layers("cfm56_7b", 1.0, float(rpm), n, rng)
        rear = _rms1(rear)
    # Report: downstream of the jet on departure. Estimated: this crossfade, over the time it takes to fly `closest`.
    w = 1.0 / (1.0 + np.exp(-(np.arange(n) / SR - 0.5 * dur) * float(speed) / max(float(closest), 1.0)))
    y = (1.0 - 0.85 * w) * 0.6 * front + (0.25 + 0.75 * w) * rear
    return _end(_pass(y, speed, closest, ground), sr)


# ======================================================================== propellers + engines
# prop rpm, blades, radius m, hp, engine rpm (0 = turbine), cylinders, header ms, unequal
# r985: 9 cyl, 450 hp at 2300 rpm, direct drive (WF); 2 blades, radius, header, unequal estimated.
# p51: V-1650-7 (ML) and modal.AIRCRAFT; header, unequal estimated. cessna172: O-320 (O320); r 0.95, header, unequal estimated.
# atr72: 6 blades, PW127 2750 hp (WF); 1200 rpm, radius estimated.
PROP_AIRCRAFT = {"r985": (2300.0, 2, 1.35, 450.0, 2300.0, 9, 0.5, 0.3), "p51": (1280.0, 4, 1.70, 1490.0, 3000.0, 12, 1.2, 0.1),
                 "cessna172": (2700.0, 2, 0.95, 160.0, 2700.0, 4, 1.6, 0.15), "atr72": (1200.0, 6, 1.96, 2750.0, 0.0, 0, 0.0, 0.0)}
TURBOPROP_SPEED = 0.3   # Farnell's five turbine partials x this


def _prop_layer(prop_rpm, blades, radius, hp, n, rng, angle=110.0, dist=30.0):
    return _rms1(modal.propeller_tone(prop_rpm, blades, radius, hp, n / SR, rng, angle=angle, distance=dist)[:n])


@sfx("prop_engine", "Propeller aircraft with its engine on one speed: Hamilton Standard harmonics + vortex noise, plus radial, V12, flat-four or turbine.",
     aircraft=("cessna172", "|".join(PROP_AIRCRAFT)), rpm=(0.0, "engine rpm (0 = from the table); the propeller follows through the reduction"),
     engine=(0.5, "0 propeller only .. 1 engine only"), angle=(110.0, "degrees from the flight direction"), distance=(30.0, "m"),
     speed=(0.0, "m/s: > 0 makes it pass the listener"), closest=(40.0, "m at the pass"), dur=(3.0, "s"), seed=(0, "variation"))
def prop_engine(sr=DEFAULT_SR, aircraft="cessna172", rpm=0.0, engine=0.5, angle=110.0, distance=30.0, speed=0.0, closest=40.0, dur=3.0, seed=0):
    if aircraft not in PROP_AIRCRAFT:
        raise ValueError(f"unknown aircraft {aircraft!r}; choose from {sorted(PROP_AIRCRAFT)}")
    rng = Ma._rng(seed, 906)
    n = Ma._n(dur)
    pr, bl, rad, hp, er, cyl, hdr, uneq = PROP_AIRCRAFT[aircraft]
    k = (float(rpm) / (er or pr)) if rpm > 0 else 1.0
    y = (1.0 - float(engine)) * _prop_layer(pr * k, bl, rad, hp, n, rng, angle, distance)
    if cyl:
        e = cyber.pipe_engine(sr=SR, dur=n / SR, rpm=er * k, cylinders=cyl, load=0.8, header=hdr, unequal=uneq, seed=seed)
    else:
        e = _partials(TURBINE_5, TURBOPROP_SPEED * k, n, rng) * 0.3      # "partiels turbine a faible gain"
    y = y + float(engine) * _rms1(Ma._fit(e, n)) * (1.0 if cyl else 0.3)
    return _end(_pass(y, speed, closest) if speed > 0 else y, sr)


@sfx("airship", "Airship: three small engines a little apart, so they beat, slow propellers, soft wind and creaking frame.",
     rpm=(2700.0, "engine rpm"), spread=(0.005, "rpm difference between engines"), dur=(5.0, "s"), seed=(0, "variation"))
def airship(sr=DEFAULT_SR, rpm=2700.0, spread=0.005, dur=5.0, seed=0):
    """Zeppelin NT: Lycoming IO-360, about 147 kW = 197 hp (ZEP). 3 engines, 2700 rpm, 2 blades, 0.5 %, radius 0.9 m."""
    rng = Ma._rng(seed, 907)
    n = Ma._n(dur)
    y = np.zeros(n)
    for i, k in enumerate((1.0 - spread, 1.0, 1.0 + spread)):
        r = float(rpm) * k
        y += _prop_layer(r, 2, 0.9, 197.0, n, rng, 110.0, 60.0)
        y += 0.7 * _rms1(Ma._fit(cyber.pipe_engine(sr=SR, dur=n / SR, rpm=r, cylinders=4, load=0.4, seed=seed + i), n))
    y = _rms1(y)
    y += 0.12 * _rms1(Ma._fit(Ma.gust(sr=SR, dur=n / SR, strength=0.2, seed=seed), n))
    y += 0.08 * Ma._fit(foley.creak(sr=SR, material="rope", force="sway", dur=n / SR, seed=seed), n)
    return _end(y, sr)


# ======================================================================== airframe
def airframe_peak(u, delta):
    """TBL-TE peak: St = f delta* / U, St1 = 0.02 M^-0.6 (BPM, NASA RP-1218). U = 30 m/s, delta* 5 mm: about 518 Hz."""
    return 0.02 * (float(u) / C) ** -0.6 * float(u) / float(delta)


def _tbl(u, delta, n, rng):
    return _rms1(Ma._tvbp(rng.standard_normal(n), np.full(n, min(airframe_peak(u, delta), 0.45 * SR)), 1.0))


@sfx("airframe", "Airframe noise of a gliding aircraft: trailing-edge turbulence at U^5, plus flaps and landing gear.",
     speed=(70.0, "m/s"), delta=(0.005, "m boundary-layer displacement thickness"), flaps=(0.0, "0..1"), gear=(0, "1 = down"),
     dur=(3.0, "s"), seed=(0, "variation"))
def airframe(sr=DEFAULT_SR, speed=70.0, delta=0.005, flaps=0.0, gear=0, dur=3.0, seed=0):
    """U^5 (BPM, FK). Gear and flaps dominate (FK); their bands and +dB are estimated, FAA-RD-77-29 not read."""
    rng = Ma._rng(seed, 908)
    n = Ma._n(dur)
    u = float(np.clip(speed, 5.0, 300.0))
    y = _tbl(u, delta, n, rng)
    if flaps > 0:
        y = y + float(flaps) * 10 ** (6.0 / 20.0) * _rms1(Ma._bp(rng.standard_normal(n), min(0.45 * SR, airframe_peak(u, delta) * 2.5), 0.8))
    if gear:
        y = y + 10 ** (10.0 / 20.0) * _rms1(Ma._bp(rng.standard_normal(n), Ma.STROUHAL * u / 0.2, 0.7))
    return _end(0.3 * y * (u / 100.0) ** 2.5, sr, peak=None)


@sfx("glider", "Glider: no engine, only airframe noise, wind over the wings and a faint canopy vent.",
     speed=(30.0, "m/s"), delta=(0.005, "m displacement thickness"), dur=(4.0, "s"), seed=(0, "variation"))
def glider(sr=DEFAULT_SR, speed=30.0, delta=0.005, dur=4.0, seed=0):
    rng = Ma._rng(seed, 909)
    n = Ma._n(dur)
    u = float(np.clip(speed, 5.0, 100.0))
    y = _tbl(u, delta, n, rng)
    y = y + 0.15 * _rms1(Ma._fit(Ma.gust(sr=SR, dur=n / SR, strength=0.3, seed=seed), n))
    y = y + 0.05 * _rms1(Ma._bp(rng.standard_normal(n), Ma.STROUHAL * u / 0.02, 10.0))
    return _end(y, sr)


# ======================================================================== rockets
def rocket_pwl(thrust, ue, eta=0.005):
    """Sound power level dB re 1 pW = efficiency x mechanical power F Ue / 2. eta 0.5 % (0.2-1.0 %, SP-8072); SLS 0.33 %, 202.4 dB (KG)."""
    return 10.0 * math.log10(float(eta) * 0.5 * float(thrust) * float(ue) / 1e-12)


def rocket_peak(diameter, ue):
    """St = f De / Ue = 0.02 (SP-8072)."""
    return 0.02 * float(ue) / float(diameter)


# Normalised sound-power spectrum of chemical rockets, (St = f de / Ue, dB per unit St): peak 0.02
ROCKET_SPECTRUM = ((0.002, -0.5), (0.005, 7.5), (0.01, 10.3), (0.02, 11.2), (0.05, 9.3), (0.1, 6.6), (0.2, 1.5),
                   (0.5, -7.2), (1.0, -13.0), (2.0, -18.9), (5.0, -26.6))


def rocket_spectrum_db(st):
    """``ROCKET_SPECTRUM`` at Strouhal ``st``, straight in log St between the points and extended along the end slopes."""
    s, d = np.log10([p[0] for p in ROCKET_SPECTRUM]), np.array([p[1] for p in ROCKET_SPECTRUM])
    x = np.log10(np.maximum(np.asarray(st, float), 1e-9))
    lo, hi = (d[1] - d[0]) / (s[1] - s[0]), (d[-1] - d[-2]) / (s[-1] - s[-2])
    return np.where(x < s[0], d[0] + lo * (x - s[0]), np.where(x > s[-1], d[-1] + hi * (x - s[-1]), np.interp(x, s, d)))


def rocket_spectrum_noise(n, fp, rng):
    """Unit-RMS noise on ``ROCKET_SPECTRUM``, its 0.02 peak moved to ``fp`` Hz."""
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g = 10.0 ** (rocket_spectrum_db(0.02 * np.maximum(f, 0.1) / float(fp)) / 20.0)
    return _rms1(np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * g, n))


TURB_SIGMA_MU2, TURB_L = 3e-6, 1.1
TURB_BANDS = (250.0, 1000.0, 4000.0)


def turbulence_sequence(dur: float, tau0: float, rng) -> tuple:
    """Unit-variance Gaussian sequence with the normalised covariance C(tau) = Phi(tau/tau0) / (tau/tau0),
    Phi(x) = (sqrt(pi)/2) erf(x), sampled at fs = 5 / tau0. Returns (sequence, fs)."""
    from scipy.special import erf
    fs = 5.0 / tau0
    m = int(math.ceil(dur * fs)) + 2
    lag = np.arange(-8 * 5, 8 * 5 + 1) / fs                     # +-8 tau0
    x = np.abs(lag) / tau0
    c = np.where(x > 0, 0.5 * math.sqrt(math.pi) * erf(x) / np.maximum(x, 1e-12), 1.0)
    H = np.sqrt(np.maximum(np.real(np.fft.fft(np.fft.ifftshift(c))), 0.0))
    h = np.real(np.fft.fftshift(np.fft.ifft(H)))
    s = np.convolve(rng.standard_normal(m + len(h)), h, "valid")[:m]
    return s / (np.std(s) + 1e-12), fs


def scintillation_sigma(f, d: float, sigma_mu2: float = TURB_SIGMA_MU2, L: float = TURB_L) -> np.ndarray:
    """Standard deviation of the log-amplitude chi at f Hz after d m, Gaussian turbulence, with saturation:
    variance (sqrt(pi)/2) sigma_mu^2 k^2 d L over 1 + d / r_s, r_s = 1 / (2 sigma_mu^2 k^2 L)."""
    k2 = (2 * math.pi * np.asarray(f, float) / C) ** 2
    var = 0.5 * math.sqrt(math.pi) * sigma_mu2 * k2 * d * L
    return np.sqrt(var / (1.0 + d * 2 * sigma_mu2 * k2 * L))


def scintillate(y, d: float, v_perp: float, rng, sigma_mu2: float = TURB_SIGMA_MU2, L: float = TURB_L):
    """Propagation through turbulence: y(t - tau(t)) e^chi(t), chi = sigma s - sigma^2, one sequence for gain and phase, the gain per band
    (sigma_chi grows with f), the delay the same for every f. ``v_perp`` m/s of the turbulence across the path."""
    n = len(y)
    tau0 = L / max(float(v_perp), 1e-3)
    s, fs = turbulence_sequence(n / SR, tau0, rng)
    s = np.interp(np.arange(n) / SR, np.arange(len(s)) / fs, s)
    delay = math.sqrt(0.5 * math.sqrt(math.pi) * sigma_mu2 * d * L) / C * s
    t = np.arange(n) / SR
    x = np.interp(t - delay, t, y)
    out = np.zeros(n)
    edges = (0.0,) + TURB_BANDS + (0.5 * SR,)
    rest = x
    from scipy.signal import butter, sosfiltfilt
    for i, fc in enumerate(TURB_BANDS + (None,)):
        band = rest if fc is None else sosfiltfilt(butter(2, fc, fs=SR, output="sos"), rest)
        rest = rest - band if fc is not None else rest
        centre = math.sqrt(max(edges[i], 62.5) * edges[i + 1])
        sig = float(scintillation_sigma(centre, d, sigma_mu2, L))
        out += band * np.exp(sig * s - sig * sig)
    return out


# peak Hz (0 = from St 0.02), De m, Ue m/s, crackle. heavy: 10-20 Hz measured (KG, BYU extract). small: De, Ue and crackle exponent estimated.
ROCKETS = {"heavy": (15.0, 8.3, 2580.0, 1.0), "small": (0.0, 0.3, 2000.0, 0.6)}


@sfx("space_rocket", "Rocket motor from far: broad two-pole roar with the Strouhal peak of its exhaust, crackle, high-frequency dispersion. phase=liftoff: ignition ramp, maximum, departure.",
     size=("heavy", "heavy | small"), diameter=(0.0, "m exit diameter (0 = from size)"), phase=("steady", "steady | liftoff"),
     distance=(1500.0, "m"), angle=(50.0, "degrees from the jet axis: loudest at 50"), dur=(8.0, "s (liftoff: use 25)"),
     turbulence=(0.0, "m/s wind across the path: scintillation (0 = still air)"), seed=(0, "variation"))
def space_rocket(sr=DEFAULT_SR, size="heavy", diameter=0.0, phase="steady", distance=1500.0, angle=50.0, dur=8.0,
                 turbulence=0.0, seed=0):
    if size not in ROCKETS:
        raise ValueError(f"unknown size {size!r}; choose from {sorted(ROCKETS)}")
    rng = Ma._rng(seed, 910)
    n = Ma._n(dur)
    fp, de, ue, cr = ROCKETS[size]
    if diameter > 0:
        fp, de = 0.0, float(diameter)
    fp = fp or rocket_peak(de, ue)
    low = rocket_spectrum_noise(n, fp, rng)
    hf = _rms1(Ma._hp(rng.standard_normal(n), 500.0))                                      # KG: strong dispersion above 500 Hz
    y = low + 0.12 * hf + 0.3 * cr * crackle(n, rng, drive=cr)
    y = _distance(y, SR, metres=float(np.clip(distance, 1.0, 20000.0)))[:n] * float(np.clip(distance, 1.0, 20000.0))
    # 50 degrees (SP:236); width estimated.
    y = y * math.exp(-(((float(angle) - 50.0) / 60.0) ** 2))
    if str(phase) == "liftoff":
        # 3 s ramp estimated; max OASPL T+8..18 s for SLS (KG:190); no Doppler over the first 5 s (report); levels estimated.
        t = np.arange(n) / SR
        env = np.interp(t, [0.0, 3.0, 8.0, 18.0, 30.0], [0.0, 0.5, 1.0, 1.0, 0.2])
        far = Ma._lp(Ma._lp(y, 400.0), 400.0)
        w = np.clip((t - 18.0) / 12.0, 0.0, 1.0)
        y = env * ((1.0 - w) * y + w * far)
    elif str(phase) != "steady":
        raise ValueError(f"unknown phase {phase!r}; choose from steady, liftoff")
    if turbulence > 0:
        y = scintillate(y, float(np.clip(distance, 1.0, 20000.0)), float(turbulence), Ma._rng(seed, 911))
    return _end(0.2 * y, sr, peak=None, fin=0.1, fout=0.3)


# ======================================================================== rotors
def rotor_pulse(rate, n, symmetry=0.5, duty=0.5, phase=0.0):
    """Farnell fig 19.55: an asymmetric triangle read through the first 1/8 of a cosine, in 0..1. ``symmetry`` moves the apex:
    half-parabola one way, half-parabola the other (phase distortion). ``rate`` pulses per second, scalar or array."""
    x = (np.cumsum(np.broadcast_to(np.asarray(rate, float), (n,))) / SR + float(phase)) % 1.0
    x = x / float(np.clip(duty, 0.05, 1.0))
    b = float(np.clip(symmetry, 0.02, 0.98))
    tri = np.where(x < b, x / b, np.maximum(1.0 - x, 0.0) / (1.0 - b)) * (x < 1.0)
    return (np.cos(2.0 * np.pi * (1.0 - tri) / 8.0) - math.cos(math.pi / 4.0)) / (1.0 - math.cos(math.pi / 4.0))


def _mdel(x, sig, base_ms, depth_s, lop_hz, gain):
    """Farnell fig 19.57, `mdel`: x in parallel with a low-passed copy read ``base + 25 ms + depth x signature`` late."""
    t = np.arange(len(x), dtype=float)
    late = t - (float(base_ms + 25.0) * 1e-3 + depth_s * sig) * SR
    return x + gain * Ma._lp(np.interp(late, t, x, left=0.0), lop_hz)


def _flatnoise(f, q, n, rng):
    """Farnell fig 19.56: level compensation for Q and for frequency, content.txt:22680-22683."""
    y = Ma._tvbp(rng.standard_normal(n), f, q)
    return y * (0.1 + math.sqrt(q) * 0.6) * np.log(np.sqrt(np.maximum(5.0 - f * f * 1.1e-7, 1.0)))


def _main_rotor(bpf, n, rng, symmetry, shift, basef=120.0, devf=220.0, reson=1.0, phase=0.0, blade_hz=200.0):
    """Fig 19.58: pulse + edge noise swept by the pulse, through mdel 10/400/0.7, 10/400/0.7, 20/400/0.5 whose delays the pulse
    moves at 1.1, 0.9, 1.2 x the period.
    Returns (chop, signature, blade ring). Blade modes 200 Hz x 1, 2, 2.7, 6.2 (F:22288)."""
    p = rotor_pulse(bpf, n, symmetry, phase=phase)
    edge = _flatnoise(basef + devf * p, reson, n, rng) * p
    y = _rms1(Ma._hp(p, 5.0)) + 0.6 * _rms1(edge)
    t0 = 1.0 / bpf
    for sc, base, g in ((1.1, 10.0, 0.7), (0.9, 10.0, 0.7), (1.2, 20.0, 0.5)):
        y = _mdel(y, p, base, float(shift) * sc * t0 / 3.0, 400.0, g)
    kick = np.abs(np.gradient(p))
    ring = sum(g * Ma._bp(kick, blade_hz * k, 12.0) for k, g in ((1.0, 1.0), (2.0, 0.6), (2.7, 0.5), (6.2, 0.25)))
    return _rms1(y), p, _rms1(ring)


def _slap_train(bpf, n, sharp, phase=0.0):
    """Blade slap: short pulses, flat up to about harmonic 40 of the BPF at sharp = 1."""
    x = (np.arange(n) * bpf / SR + phase) % 1.0
    w = 1.0 / (10.0 + 30.0 * float(sharp))
    return np.where(x < w, 0.5 - 0.5 * np.cos(2.0 * np.pi * x / w), 0.0)


def _tail(rate, n, rng):
    """Fig 19.60: fixed-width pulse + noise swept around 250 Hz (100 + 500 x pulse, Q 12). No BVI on a tail (RS:233)."""
    p = rotor_pulse(rate, n, 0.5)
    noise = Ma._tvbp(rng.standard_normal(n), 100.0 + 500.0 * p, 12.0)
    return _rms1(0.3 * Ma._hp(p, 20.0) / (np.std(p) + 1e-9) + 0.15 * noise / (np.std(noise) + 1e-9) * 1.5 * p)


@numba.njit(cache=True)
def _piston_ring(x, noise, d, fb):
    """Fig 19.54: 4 delays in a circle with a non-linear junction: a part of a cosine (x 0.125, -0.25), clipped at +-0.5."""
    n = len(x)
    bufs = [np.zeros(d[i]) for i in range(4)]
    y = np.zeros(n)
    for t in range(n):
        a, b, c, e = bufs[0][t % d[0]], bufs[1][t % d[1]], bufs[2][t % d[2]], bufs[3][t % d[3]]
        j = math.cos(2.0 * math.pi * (0.125 * (a + c + noise[t]) - 0.25))
        j = max(-0.5, min(0.5, j))
        bufs[0][t % d[0]] = x[t] + fb * e
        bufs[1][t % d[1]] = a + 0.5 * j
        bufs[2][t % d[2]] = b - 0.5 * j
        bufs[3][t % d[3]] = c
        y[t] = 0.7 * b + 0.3 * e
    return y


def _piston(rate, n, rng):
    """Fig 19.54: shark fin 14x^3 - 14x^2, shaper 1/(1 + kx^2) (book prints 1 - kx^2, the patch has +: chunk_09); k estimated;
    delays 8, 9, 3, 6 ms, hip 200 x 2, lop 200, bp 500 Q 5; junction noise bp 100 Q 2."""
    x = (np.arange(n) * rate / SR) % 1.0
    fin = 14.0 * x ** 3 - 14.0 * x ** 2
    pulse = 1.0 / (1.0 + 8.0 * fin * fin)
    pulse = pulse - pulse.mean()
    d = np.array([int(k * 1e-3 * SR) for k in (8, 9, 3, 6)], dtype=np.int64)
    noise = 0.3 * Ma._bp(rng.standard_normal(n), 100.0, 2.0)
    y = _piston_ring(np.ascontiguousarray(pulse), np.ascontiguousarray(noise), d, 0.25)
    y = Ma._hp(Ma._hp(y, 200.0), 200.0)
    return _rms1(Ma._lp(y, 200.0) + Ma._bp(y, 500.0, 5.0))


def tail_rpm_law(rpm, load):
    """chunk_13's proposal on Farnell (3-5 x the main rotor, rises with lift, top of range 2000): rpm (4 + load), capped."""
    return min(float(rpm) * (4.0 + float(load)), 2000.0)


def tip_mach_rotor(rpm, radius, forward=0.0, c=C):
    """Advancing tip Mach. Tips rarely above 300 m/s (F)."""
    return (2.0 * math.pi * float(rpm) / 60.0 * float(radius) + float(forward)) / c


# rpm, blades, radius, drive ("t" turbine or "p" piston), engine rpm, cabin dB (TM:328, 0 = none measured), kind.
# uh60: 258 (TM), 4 (WF), 8.18 (estimated), 20900 (TM), 115. oh58: 347.5 and 35350 (TM); blades, radius estimated. r44: 400 (POH); 2, radius, engine estimated.
# ch47: 243 (TM), 3, 18.29 m (F), 118, tandem; engine rpm estimated. uh1b: 2 blades, 15 m (F); rpm, engine rpm estimated. s55: piston, 16.15 m (F); rest estimated.
# ah1: 120 dB (TM), rest estimated. "": Farnell's 300 rpm, 2 blades.
ROTORCRAFT = {"": (300.0, 2, 7.0, "t", 20000.0, 0.0, "single"), "uh60": (258.0, 4, 8.18, "t", 20900.0, 115.0, "single"),
              "oh58": (347.5, 2, 5.4, "t", 35350.0, 0.0, "single"), "r44": (400.0, 2, 5.0, "p", 2700.0, 0.0, "single"),
              "ch47": (243.0, 3, 9.145, "t", 15000.0, 118.0, "tandem"), "uh1b": (324.0, 2, 7.5, "t", 6600.0, 0.0, "single"),
              "s55": (230.0, 3, 8.075, "p", 2400.0, 0.0, "single"), "ah1": (324.0, 2, 6.7, "t", 6600.0, 120.0, "single")}
GEAR_RATIOS = ((1.45, 0.33), (2.81, 0.2), (9.41, 0.14))     # Farnell fig 19.61, content.txt:22818-22830 (his text: 1.5, 2, 10, "fairly arbitrary")
TANDEM_DELAY = 18.0 / C                                     # F 18.29 m, report rounds to 18 m: 52 ms
_STATES = {"hover": (0.5, 0.3), "cruise": (0.5, 0.4), "bank": (0.25, 0.7), "descent": (0.12, 1.0), "fast": (0.25, 0.5)}


def _shock_warp(y, drive, amount):
    """Steepen ``y`` in time along the smooth ``drive``: a sawtooth-like skewed wave (F:22177) for the advancing tip.
    ``amount`` 1 stops just short of overturning, so the front is steep but stays continuous."""
    t = np.arange(len(y), dtype=float)
    return np.interp(t, t - float(amount) * 0.9 / (np.max(np.gradient(drive)) + 1e-12) * drive, y)


def _fuselage(y, n):
    """Metal sheet as a convolution filter (LP:2046, F: thin sheet, "similar to a large car body"). Size, thickness estimated."""
    h = plates.metal_plate(sr=SR, size=1.2, thickness=1.5, decay=0.5, vel=0.5, hardness=0.3)
    return _rms1(fftconvolve(y, _rms1(h) / len(h), mode="full")[:n])


@sfx("rotorcraft", "Helicopter built on Farnell's rotor pulse: main rotor with moving combs, tail rotor, turbine or piston, gear whine, slap by flight state and by where you stand.",
     craft=("", "'' (300 rpm) | " + " | ".join(k for k in ROTORCRAFT if k)), rpm=(0.0, "main rotor rpm (0 = from craft)"), blades=(0, "main blades (0 = from craft; 4 alone picks uh60)"),
     radius=(0.0, "m (0 = from craft)"), kind=("", "single | tandem | coaxial ('' = from craft)"), tail_rpm=(0.0, "0 = rpm (4 + load), at most 2000"), tail_blades=(4, "2..16"),
     engine=("", "turbine | piston ('' = from craft)"), gearbox=(0, "1 = three gear whines"), interior=(0, "1 = inside the fuselage"), state=("hover", "|".join(_STATES)),
     angle=(30.0, "0 ahead in the rotor plane, 30 slap peak, 90 under it, 180 behind"), symmetry=(-1.0, "pulse apex 0..1, -1 = from state and angle"), shift=(0.5, "0 swoosh .. 1 slap"),
     load=(0.55, "0..1"), far=(0.0, "0 near .. 1 far: Farnell's distance filter"), speed=(0.0, "m/s: > 0 passes the listener"), closest=(60.0, "m"), dur=(3.0, "s"), seed=(0, "variation"))
def rotorcraft(sr=DEFAULT_SR, craft="", rpm=0.0, blades=0, radius=0.0, kind="", tail_rpm=0.0, tail_blades=4, engine="", gearbox=0, interior=0, state="hover",
               angle=30.0, symmetry=-1.0, shift=0.5, load=0.55, far=0.0, speed=0.0, closest=60.0, dur=3.0, seed=0):
    if not craft and int(blades) == 4:
        craft = "uh60"
    if craft not in ROTORCRAFT:
        raise ValueError(f"unknown craft {craft!r}; choose from {sorted(ROTORCRAFT)}")
    if state not in _STATES:
        raise ValueError(f"unknown state {state!r}; choose from {sorted(_STATES)}")
    r0, b0, rad, drive, erpm, cabin_db, kind0 = ROTORCRAFT[craft]
    kind = str(kind) or kind0
    if kind not in ("single", "tandem", "coaxial"):
        raise ValueError(f"unknown kind {kind!r}; choose from single, tandem, coaxial")
    rng = Ma._rng(seed, 911)
    n = Ma._n(dur)
    r = float(rpm) if rpm > 0 else r0
    z = int(blades) if blades > 0 else b0
    rad = float(radius) if radius > 0 else rad
    ld = float(np.clip(load, 0.0, 1.0))
    bpf = r * z / 60.0
    ang = float(np.clip(angle, 0.0, 180.0))
    th = math.radians(ang)
    sym0, slap0 = _STATES[state]
    # Pulse symmetry exchanges slap and whoosh, midpoint = directly below (F content.txt:22779-22782).
    sym = float(symmetry) if symmetry >= 0 else float(np.clip(0.5 + (sym0 - 0.65) * abs(math.cos(th)), 0.02, 0.98))

    # Position (F:22187-22201); only the places are sourced, these curves are estimated.
    g_chop = 0.12 + 0.88 * math.cos(th) ** 2
    g_slap = math.exp(-(((ang - 30.0) / 18.0) ** 2))                                    # 30 degrees below the plane
    g_eng = 0.25 + 0.75 * math.sin(th) ** 2 + 0.4 * max(-math.cos(th), 0.0)             # under it; behind: exhaust
    g_down = 0.1 + 0.9 * math.sin(th) ** 2
    g_tail = 0.25 + 0.75 * (1.0 - math.cos(th)) / 2.0                                   # behind

    chop, sig, ring = _main_rotor(bpf, n, rng, sym, shift)
    slap = _rms1(Ma._hp(_slap_train(bpf, n, slap0), 100.0))
    if kind == "tandem":
        c2, s2, r2 = _main_rotor(bpf, n, rng, sym, shift, phase=0.5)
        late = lambda a: np.interp(np.arange(n) - TANDEM_DELAY * SR, np.arange(n, dtype=float), a, left=0.0)
        chop, ring = _rms1(chop + late(c2)), _rms1(ring + late(r2))
        slap = _rms1(slap + late(Ma._hp(_slap_train(bpf, n, slap0, 0.5), 100.0)))
    elif kind == "coaxial":
        c2, s2, r2 = _main_rotor(bpf, n, rng, sym, shift, phase=0.5)
        chop, ring = _rms1(chop + c2), _rms1(ring + r2)
        slap = _rms1(slap + Ma._hp(_slap_train(bpf, n, slap0, 0.5), 100.0))
    if state == "fast":
        # Past Mach 0.85 at the advancing tip; forward 80 m/s, both estimated. Not on the tail (RS:233).
        m = tip_mach_rotor(r, rad, 80.0)
        if m > 0.85:
            chop = _shock_warp(chop, sig, min((m - 0.85) / 0.05, 1.0))
    y = g_chop * chop + 0.3 * slap0 * g_chop * ring + slap0 * 1.5 * g_slap * slap

    fm = 1.0 + 0.004 * (sig - sig.mean())                                               # engine pulses with the rotor, Doppler (F:22178-22180)
    e_hz = erpm / 60.0
    if (engine or ("piston" if drive == "p" else "turbine")) == "piston":
        # 4-stroke firing of an unstated cylinder count: e_hz x 2 stands for a four-cylinder.
        eng = _piston(e_hz * 2.0, n, rng)
    elif (engine or "turbine") == "turbine":
        eng = _partials(TURBINE_5, 0.35, n, rng, fm)
    else:
        raise ValueError(f"unknown engine {engine!r}; choose from turbine, piston")
    y = y + 0.35 * g_eng * _rms1(eng)
    if int(gearbox):
        ph = np.cumsum(e_hz * fm) / SR
        whine = sum(g * np.sin(2.0 * np.pi * k * ph) for k, g in GEAR_RATIOS if k * e_hz < 0.45 * SR)
        y = y + 0.25 * g_eng * _rms1(np.asarray(whine))
    y = y + 0.3 * g_down * (0.4 + ld) * _rms1(Ma._bp(rng.standard_normal(n), 900.0, 0.4)) * (0.8 + 0.2 * sig)     # downdraft

    if kind == "single":
        tr = float(tail_rpm) if tail_rpm > 0 else tail_rpm_law(r, ld)
        y = y + 0.3 * g_tail * _tail(tr * int(np.clip(tail_blades, 2, 16)) / 60.0, n, rng)

    f = float(np.clip(far, 0.0, 1.0))
    if f > 0:
        # Fig 19.62: 3 x lop 100; ground reflection x 0.5, delay (1 - d) x 4000 + 50 ms in a 1000 ms line, d = 1 - far. Blend estimated.
        low = Ma._lp(Ma._lp(Ma._lp(y, 100.0), 100.0), 100.0)
        d = min(f * 4000.0 + 50.0, 1000.0) * 1e-3
        wet = low + 0.5 * np.interp(np.arange(n) - d * SR, np.arange(n, dtype=float), low, left=0.0)
        y = (1.0 - f) * y + f * _rms1(wet) * math.sqrt(float(np.mean(y * y)))
    if int(interior):
        y = _rms1(_fuselage(y, n)) * (0.35 * 10 ** (((cabin_db or 115.0) - 120.0) / 20.0))     # cabin levels re AH-1S (TM:328)
        return _end(_pass(y, speed, closest) if speed > 0 else y, sr, peak=None)
    return _end(_pass(y, speed, closest) if speed > 0 else y, sr)


# ======================================================================== drones
# Phantom II (BU): shaft 64 Hz 19 dB over the floor, BPF 128 Hz 45 dB, harmonics 1-4 alike, then 32-38 dB; motor 34 dB at 832 and 2240 Hz.
DRONE_RPM, DRONE_BLADES, DRONE_DB = 3840.0, 2, {"shaft": 19.0, "bpf": 45.0, "high": 35.0, "motor": 34.0}
HUMP_DB, HUMP_BAND = 10.0, (600.0, 6000.0)


def _drone_rotor(rpm_t, blades, n, rng):
    """One rotor over a floor of level 1: lines at the shaft rate and its blade harmonics, no tanh."""
    y = np.zeros(n)
    base = np.cumsum(rpm_t / 60.0) / SR
    amp = lambda db: 10 ** (db / 20.0)
    y += amp(DRONE_DB["shaft"]) * np.sin(2.0 * np.pi * base + rng.uniform(0, 6.28))
    f0 = float(rpm_t.mean()) / 60.0
    for k in range(1, 13):
        if k * blades * f0 >= 0.45 * SR:
            break
        y += amp(DRONE_DB["bpf"] if k <= 4 else DRONE_DB["high"]) * np.sin(2.0 * np.pi * k * blades * base + rng.uniform(0, 6.28))
    for f in (832.0, 2240.0):
        y += amp(DRONE_DB["motor"]) * np.sin(2.0 * np.pi * np.cumsum(rpm_t / DRONE_RPM * f) / SR + rng.uniform(0, 6.28))
    return y


def _maneuver_rpm(rpm, n, rng, on):
    """Hover, translate, brake: +-30 % in 50 ms ramps, per rotor; a new demand every 0.8 s."""
    if not on:
        return np.full(n, float(rpm))
    k = int(n / (0.8 * SR)) + 2
    steps = np.repeat(rng.uniform(-0.3, 0.3, k), int(0.8 * SR))[:n]
    w = np.hanning(int(0.05 * SR))
    return float(rpm) * (1.0 + np.convolve(steps, w / w.sum(), mode="same"))


@sfx("drone", "Multirotor drone: blade-rate comb of each rotor, slightly apart so they beat, the broad hump of the blades and two motor tones.",
     n=(4, "rotors: 4, 6, 8"), rpm=(3840.0, "hover rpm"), blades=(2, "per rotor"), detune=(0.01, "rotor-to-rotor spread"),
     maneuver=(0, "1 = per-rotor rpm changes"), dur=(3.0, "s"), seed=(0, "variation"))
def drone(sr=DEFAULT_SR, n=4, rpm=3840.0, blades=2, detune=0.01, maneuver=0, dur=3.0, seed=0):
    rng = Ma._rng(seed, 912)
    m = Ma._n(dur)
    y = np.zeros(m)
    for _ in range(int(np.clip(n, 1, 12))):
        r = float(rpm) * (1.0 + float(detune) * rng.uniform(-1.0, 1.0))
        y += _drone_rotor(_maneuver_rpm(r, m, rng, int(maneuver)), int(blades), m, rng)
    # Floor of one rotor per FFT bin, then the hump 10 dB over 600-6000 Hz.
    from scipy.signal import butter, sosfilt
    w = rng.standard_normal(m) * math.sqrt(int(np.clip(n, 1, 12)))
    sos = butter(2, HUMP_BAND, "band", fs=SR, output="sos")
    y += w + (10 ** (HUMP_DB / 20.0) - 1.0) * sosfilt(sos, w)
    return _end(y, sr)


# ======================================================================== cabin
def cabin_wall(y):
    """30 dB above 500 Hz: a complementary split at 500 Hz, the top turned down."""
    lo = Ma._lp(y, 500.0, 2)
    return lo + 10 ** (-30.0 / 20.0) * (y - lo)


@sfx("cabin", "Inside an airliner: the engines through the fuselage wall, and the air-conditioning.",
     kind=("jet", "jet | turboprop"), air=(0.4, "0..1 air-conditioning level"), dur=(4.0, "s"), seed=(0, "variation"))
def cabin(sr=DEFAULT_SR, kind="jet", air=0.4, dur=4.0, seed=0):
    rng = Ma._rng(seed, 913)
    n = Ma._n(dur)
    if str(kind) == "jet":
        fan, jet = _turbofan_layers("cfm56_7b", 0.9, 0.0, n, rng)
        bed = 0.6 * fan + 1.2 * jet
    elif str(kind) == "turboprop":
        bed = _prop_layer(1200.0, 6, 1.96, 2750.0, n, rng)
    else:
        raise ValueError(f"unknown kind {kind!r}; choose from jet, turboprop")
    y = _rms1(cabin_wall(bed))
    y = y + float(np.clip(air, 0.0, 1.0)) * _rms1(Ma._bp(rng.standard_normal(n), 1800.0, 0.5))
    return _end(y, sr)
