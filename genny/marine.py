"""Ships and boats: radiated ship noise, slow two-stroke, horn and fog signals, bell, outboard, jet ski, sailboat, submarine, diaphone, lapping water.

Sources of every constant: research/SOURCES.md, section marine.
Air-water transmission T = 0.11 %, -29.5 dB, from the impedances 415 and 1.48e6. Every airborne level is estimated: see tests/test_marine.py for what was measured.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, firwin2, resample_poly

from .core import DEFAULT_SR
from .cyber import pipe_engine
from .engines import _even, _events, _pulses, engine_mech, two_stroke
from .fauna import _put
from .foley import P, _fin, _n, church_bell, clink, cloth, creak, sonar
from .foley import aeolian_harp as _aeolian_harp
from .matter import _lfn, _lp, _unit, air_leak, bubbles, gurgle, splash, surf
from .plates import metal_plate
from .reverbs import distance as _distance
from .road import _band, _fit, _kind, _modes
from .sfx import sfx

SHIP_REF_DB = 225.0                                         # dB re 1 uPa^2/Hz at 1 m where the digital density is 1 / Hz: keeps the speed law audible
SUB_REF_DB = 175.0                                         # the same for a submarine, which radiates 50-60 dB less
BOAT_REF_DB = 185.0                                         # broadband dB at digital RMS 1, small craft under water
HORN_REF_DB = 160.0                                         # dB at 1 m at digital RMS 1: 143 dB in the annex's band is RMS .14


def _out(y, sr, level=None, fade=0.05):
    """As `foley._fin`, without its 12 Hz high-pass, which would take the blade rate and the slow engine's firing away."""
    y = np.asarray(y, float)
    y = y - y.mean()
    if sr != P:
        y = resample_poly(y, int(sr), P)
    k = min(len(y) // 2, int(fade * sr))
    w = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / max(k, 1))
    y[:k] *= w
    y[len(y) - k:] *= w[::-1]
    pk = float(np.max(np.abs(y))) + 1e-12
    return y * (level / pk) if level is not None else (y * 1.2 / pk if pk > 1.2 else y)


def _shaped(n, rng, db, ref):
    """Gaussian noise through a FIR of power density ``db(f)`` dB, 0 dB of ``ref`` being a density of 1 / Hz."""
    f = np.linspace(0.0, P / 2.0, 2 ** 15 + 1)
    g = 10.0 ** ((db(np.maximum(f, 1.0)) - ref) / 20.0) * np.sqrt(P / 2.0)
    g[f < 2.0] = 0.0
    h = firwin2(2 ** 14 + 1, f, g, fs=P)
    return fftconvolve(rng.standard_normal(n + len(h) - 1), h, mode="valid")


def _lines(n, f, amps, rng):
    """Harmonics of f (constant), one amplitude per harmonic."""
    t = np.arange(n) / P
    y = np.zeros(n)
    for k, a in enumerate(amps, 1):
        if a > 0 and k * f < 0.45 * P:
            y += a * np.sin(2.0 * np.pi * k * f * t + rng.uniform(0.0, 2.0 * np.pi))
    return y


def _db_amp(db, ref):
    """Amplitude of a sine whose power is ``db`` over ``ref`` (density in 1 Hz: a line and a density are both read in dB re 1 uPa^2/Hz)."""
    return np.sqrt(2.0) * 10.0 ** ((db - ref) / 20.0)


# =============================================================================================
# RADIATED SHIP NOISE
# =============================================================================================
AIR_WATER_DB = 10.0 * np.log10(4.0 * 415.0 * 1.48e6 / (415.0 + 1.48e6) ** 2)   # -29.5 dB: applied as a gain, so a normalize that follows erases it
LENGTH_REF = 300.0 * 0.3048                                 # m
SHIPS = {"cargo": dict(model="wales", vref=14.0, rpm=100.0, blades=5, tons=30000.0, cb=0.65),
         "tanker": dict(model="ross", vref=14.0, rpm=80.0, blades=4, tons=30000.0, cb=0.78)}
VC, WITTEKIND_C = 9.0, (125.0, 0.35, -8e-3, 6e-5, -2e-7, 2.2e-10)   # kn; SL1 c0-c5, under 400 Hz
WITTEKIND_DREF = 10000.0                                    # t
WITTEKIND_SL2 = ((25.0, 40.0, 60.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 10000.0),
                 (140.0, 150.0, 158.0, 164.0, 166.5, 167.0, 166.5, 165.5, 163.5, 160.5, 157.0, 153.0, 152.0))
WITTEKIND_SL2_V, WITTEKIND_SL2_T = 14.0 / 9.0, 33000.0     # the reference ship: V / Vc, t


def wales_heitmeyer(f):
    """S(f) = 230 - 10 log f^3.594 + 10 log (1 + f^2/340^2)^.917, flat under 30 Hz."""
    f = np.maximum(np.asarray(f, float), 30.0)
    return 230.0 - 35.94 * np.log10(f) + 9.17 * np.log10(1.0 + (f / 340.0) ** 2)


def ross(f, speed, tons):
    """S = 154 - 20 log f + 60 log(V / 10 kn) + 9 log DT, f > 100 Hz, DT < 30000 t; flat under 100 Hz, where the formula stops."""
    return 154.0 - 20.0 * np.log10(np.maximum(np.asarray(f, float), 100.0)) + 60.0 * np.log10(speed / 10.0) + 9.0 * np.log10(min(float(tons), 30000.0))


def randi(f, speed, length):
    """RANDI 3.1 ship source level, dB re 1 uPa^2/Hz at 1 m: the average ship (12 kn, 300 ft) scaled by speed and length."""
    f = np.maximum(np.asarray(f, float), 1.0)
    lf = np.log10(f)
    lso = np.where(f < 500.0, -10.0 * np.log10(10.0 ** (-1.06 * lf - 14.34) + 10.0 ** (3.32 * lf - 21.425)), 173.2 - 18.0 * lf)
    df = np.where(f <= 28.4, 8.1, np.where(f <= 191.6, 22.3 - 9.77 * lf, 0.0))
    ft = float(length) / 0.3048
    return lso + 60.0 * np.log10(speed / 12.0) + 20.0 * np.log10(ft / 300.0) + df * ft ** 1.15 / 3643.0 + 3.0


def ship_level(f, model="wales", speed=14.0, vref=14.0, length=LENGTH_REF, tons=30000.0):
    """dB re 1 uPa^2/Hz at 1 m. Wales-Heitmeyer scaled by 60 log(V / Vref) and 20 log(l / 300 ft); sonic: +8 dB; ross: `ross`."""
    if model == "randi":
        return randi(f, speed, length)
    if model == "ross":
        return ross(f, speed, tons)
    s = wales_heitmeyer(f) + 60.0 * np.log10(speed / vref) + 20.0 * np.log10(length / LENGTH_REF)
    return s + 8.0 if model == "sonic" else s


def wittekind_sl1(f):
    """SL1 alone, third-octave band level, polynomial in f. It falls to 45 dB at 300 Hz and goes negative at 400 Hz: floored at 40 dB."""
    f = np.minimum(np.asarray(f, float), 400.0)
    return np.maximum(sum(c * f ** k for k, c in enumerate(WITTEKIND_C)), 40.0)


def wittekind_a(speed, vc, cb):
    """Speed and hull fullness term of SL1."""
    return 80.0 * np.log10(speed / vc * 4.0 * cb)


def wittekind_b(tons):
    """Displacement term of SL1 and SL2."""
    return 10.0 * np.log10((tons / WITTEKIND_DREF) ** (2.0 / 3.0))


def wittekind_sl2(f, speed, vc, tons):
    """High-frequency cavitation, third-octave band level: the reference ship's curve, moved by 60 log V/Vc and B."""
    f = np.clip(np.asarray(f, float), WITTEKIND_SL2[0][0], WITTEKIND_SL2[0][-1])
    ref = np.interp(np.log10(f), np.log10(WITTEKIND_SL2[0]), WITTEKIND_SL2[1])
    return ref + 60.0 * np.log10(speed / vc / WITTEKIND_SL2_V) + wittekind_b(tons) - wittekind_b(WITTEKIND_SL2_T)


def wittekind_density(f, speed, vc, cb, tons):
    """dB re 1 uPa^2/Hz at 1 m above inception: SL1 (with A, B) and SL2 summed, third-octave levels over their bandwidth 0.23 f."""
    f = np.maximum(np.asarray(f, float), 1.0)
    s = 10.0 ** ((wittekind_sl1(f) + wittekind_a(speed, vc, cb) + wittekind_b(tons)) / 10.0) + 10.0 ** (wittekind_sl2(f, speed, vc, tons) / 10.0)
    return 10.0 * np.log10(s) - 10.0 * np.log10(0.2316 * f)


def _cavitation(speed, vc, bpf, n, rng, dur, seed, cb=0.65, tons=30000.0):
    """Above Vc: the Wittekind spectrum, the noise amplitude-modulated at the blade rate, lines on the blade harmonics, crackle of bubbles."""
    if speed <= vc:
        return np.zeros(n)
    level = lambda f: wittekind_density(f, speed, vc, cb, tons)
    am = 1.0 + 0.8 * np.sin(2.0 * np.pi * bpf * np.arange(n) / P)
    y = _shaped(n, rng, level, SHIP_REF_DB) * am
    amps = []
    for k in range(1, int(400.0 / bpf) + 1):
        f = k * bpf
        amps.append(_db_amp(float(level(f)) + (10.0 if 50.0 <= f <= 150.0 else -10.0), SHIP_REF_DB))
    crackle = bubbles(sr=P, dur=dur, rate=300.0, size=1.0, underwater=1.0, seed=seed)
    return y + _lines(n, bpf, amps, rng) + 0.3 * float(np.std(y)) * _unit(_fit(crackle, n)) * am


def _radiated(k, model, speed, length, tons, rpm, blades, cavitation, vc, n, rng, dur, seed):
    vref = SHIPS[k]["vref"]
    level = lambda f: ship_level(f, model, speed, vref, length, tons)
    bpf = rpm * speed / vref * blades / 60.0                # fixed pitch: constant advance ratio, rpm with speed
    # Lines 10 dB over the noise at the fundamental, falling as 1/k. The BPF comb stays with every model.
    amps = [_db_amp(float(level(bpf)) + 10.0, SHIP_REF_DB) / h for h in range(1, 13)]
    y = _shaped(n, rng, level, SHIP_REF_DB) + _lines(n, bpf, amps, rng)
    return (y + _cavitation(speed, vc, bpf, n, rng, dur, seed, SHIPS[k]["cb"], tons)) if cavitation else y


@sfx("ship", "A ship's radiated noise. Under water: Wales-Heitmeyer (cargo) or Ross (tanker) spectrum in absolute dB, speed law 60 log V, blade-rate comb, cavitation above 9 kn. In air: the same 29.5 dB down under funnel and wake (estimated). Levels are kept, do not normalize to compare them.",
     kind=("cargo", "cargo | tanker"), listen=("under", "under | air"), model=("", "wales | sonic | ross | randi; empty = the kind's"), speed=(-1.0, "knots, 2..30; -1 = 14"),
     length=(LENGTH_REF, "m: +20 log(l / 300 ft), wales and sonic"), tons=(-1.0, "t displacement, ross, up to 30000"), rpm=(-1.0, "shaft at the kind's reference speed"), blades=(0, "propeller blades; 0 = the kind's"),
     cavitation=(0, "1 = cavitates above vc"), vc=(VC, "knots, inception"), engine=(1.0, "air: level of funnel and wake"), dur=(6.0, "s"), seed=(0, "variation"))
def ship(sr=DEFAULT_SR, kind="cargo", listen="under", model="", speed=-1.0, length=LENGTH_REF, tons=-1.0, rpm=-1.0, blades=0, cavitation=0, vc=VC, engine=1.0, dur=6.0, seed=0):
    k = _kind(kind, SHIPS, "ship")
    spec = SHIPS[k]
    m = str(model) or spec["model"]
    if m not in ("wales", "sonic", "ross", "randi"):
        raise ValueError(f"ship model {model!r}: use wales | sonic | ross | randi")
    if listen not in ("under", "air"):
        raise ValueError(f"ship listen {listen!r}: use under | air")
    V = spec["vref"] if speed < 0 else float(np.clip(speed, 2.0, 30.0))
    dur = float(np.clip(dur, 0.5, 120.0))
    n = _n(dur)
    rng = np.random.default_rng(int(seed))
    shaft = spec["rpm"] if rpm <= 0 else float(rpm)
    y = _radiated(k, m, V, float(length), spec["tons"] if tons <= 0 else float(tons), shaft, int(blades) or spec["blades"], cavitation, float(vc), n, rng, dur, seed)
    if listen == "air":
        y = y * 10.0 ** (AIR_WATER_DB / 20.0)
        if engine > 0:                                      # estimated: engine at .05, wake at .02 growing as 60 log V
            s = V / spec["vref"]
            wake = _fit(surf(sr=P, dur=max(dur, 10.0), distance=30.0, period_min=3.0, period_max=6.0, bubble_rate=1500.0, seed=seed), n)
            y = y + float(engine) * (0.05 * _fit(_slow_engine(shaft * s, 6, 2, 0.6, 1.0, dur, seed), n) + 0.02 * s ** 3 * wake)
    return _out(y, sr)


# =============================================================================================
# SLOW MARINE DIESEL
# =============================================================================================
HULL_MODES = (18.0, 33.0, 61.0)                             # Hz, local modes of decks and engine seating


def _slow_engine(rpm, cylinders, strokes, load, hull, dur, seed):
    """Waveguide engine with even firing, big head and block, and the firing felt through the hull: each firing a 40 ms pressure pulse with its hull modes."""
    n = _n(dur)
    r = np.full(n, float(rpm))
    off = _even(int(cylinders))
    per = 60.0 if int(strokes) == 2 else 120.0
    # Already running: one cycle is thrown away, as the first port opening into empty pipes is a transient (measured, seed 1: the file's peak, and a click, at .052 s).
    lead = _n(per / float(rpm))
    y = pipe_engine(sr=P, dur=(n + lead) / P, rpm=np.full(n + lead, float(rpm)), load=load, cycle=int(strokes), offsets=list(off), header=12.0, pipe=40.0, unequal=0.0, mic=0.15, seed=seed)[lead:]
    idx = _events(r, off, n, per=per)
    m = _n(0.04)
    x = np.zeros(n)
    np.add.at(x, idx, 1.0)
    pressure = fftconvolve(x, 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(m) + 0.5) / m))[:n]
    ring = _modes(_pulses(idx, n), HULL_MODES, 0.5, (1.0, 0.6, 0.3))
    # engine_mech's cycle is 720 degrees: a 2-stroke fires each slot twice in it.
    mech = _fit(engine_mech(0.6, r, 1.0, dur, tuple((i + j) / 2.0 for j in (0, 1) for i in off) if int(strokes) == 2 else off, seed=seed), n)
    return _fit(_unit(y), n) + 0.4 * _unit(mech) +float(hull) * (2.0 * _unit(pressure) + _unit(ring))


@sfx("ship_engine", "Slow marine diesel: a two-stroke cargo engine, six cylinders at 100 rpm firing at 10 Hz, felt as a pulse through the hull.",
     rpm=(100.0, "crankshaft, 30.."), cylinders=(6, "1..12"), strokes=(2, "2 | 4"), load=(0.6, "0..1"), hull=(1.0, "level of the pulse in the hull"), dur=(4.0, "s"), seed=(0, "variation"))
def ship_engine(sr=DEFAULT_SR, rpm=100.0, cylinders=6, strokes=2, load=0.6, hull=1.0, dur=4.0, seed=0):
    if int(strokes) not in (2, 4):
        raise ValueError(f"ship_engine strokes {strokes!r}: use 2 | 4")
    y = _slow_engine(max(float(rpm), 30.0), int(np.clip(cylinders, 1, 12)), int(strokes), float(load), float(hull), float(np.clip(dur, 0.5, 60.0)), seed)
    return _out(y, sr, 0.7)


# =============================================================================================
# HORN, FOG SIGNALS, BELL
# =============================================================================================
PROLONGED, SHORT = 5.0, 1.0                                 # s: prolonged 4-6, short about 1
FOG_INTERVAL, BELL_INTERVAL = 120.0, 60.0                   # s, the most allowed
SIGNALS = {"underway": (PROLONGED,), "stopped": (PROLONGED, PROLONGED), "sail": (PROLONGED, SHORT, SHORT)}
BLAST_GAP = 2.0                                             # s between blasts
SECOND_TONE = 1.12                                          # two-tone horn: a whole tone


def horn_spec(length):
    """Fundamental band (Hz), dB at 1 m, range in nmi by ship length in m."""
    if length >= 200.0:
        return (70.0, 200.0), 143.0, 2.0
    if length >= 75.0:
        return (130.0, 350.0), 138.0, 1.5
    if length >= 20.0:
        return (250.0, 700.0), 130.0, 1.0
    return (250.0, 700.0), 120.0, 0.5


def horn_band_rms(y):
    """RMS of the strongest third-octave inside 180-700 Hz, the band the annex's level is given in."""
    s = np.fft.rfft(y)
    fr = np.fft.rfftfreq(len(y), 1.0 / P)
    return max(float(np.sqrt(2.0 * np.sum(np.abs(s[(fr >= c / 2 ** (1 / 6)) & (fr < c * 2 ** (1 / 6))]) ** 2))) / len(y) for c in (200.0, 250.0, 315.0, 400.0, 500.0, 630.0))


def _horn_tone(f, dur, rng):
    """Two pulsed sources into a horn, unit in the annex's band: horn low-pass at 4 kHz, .2 % flutter, the air's rise and fall of .25 and .35 s, the second tone at -3 dB.
    Not `tubes.reed_tube`: measured on it, its closed bore puts the fundamental 14-20 dB under the third harmonic, and at 418 Hz, pressure .6, 69 dB under."""
    n = _n(dur)
    flutter = 1.0 + 0.002 * _lfn(rng, n, 3.0)
    y = np.zeros(n)
    for k, g in ((1.0, 1.0), (SECOND_TONE, 0.7)):
        y += g * np.maximum(np.sin(2.0 * np.pi * np.cumsum(k * f * flutter) / P + rng.uniform(0.0, 6.28)), 0.0) ** 3
    y = _lp(y - y.mean(), 4000.0, 2)
    a, b = min(_n(0.25), n // 3), min(_n(0.35), n // 3)
    unit = horn_band_rms(y[a:n - b]) + 1e-12
    y[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    y[n - b:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(b) + 1.0) / b)
    return y / unit


def _blasts(signal, f, seed):
    """One sequence of blasts."""
    t, parts = 0.0, []
    rng = np.random.default_rng(int(seed))
    tones = {d: _horn_tone(f, d, rng) for d in set(signal)}
    for d in signal:
        parts.append((t, tones[d]))
        t += d + BLAST_GAP
    y = np.zeros(_n(t - BLAST_GAP + 0.05))
    for t, x in parts:
        y[_n(t):_n(t) + len(x)] += x
    return y


def _place(y, metres, coast):
    """Spreading and air (`reverbs.distance`), and the echo from a coast `coast` m off, back past the ship: 2 coast / c later, 1 / 2 coast down, reflection .5."""
    d = _distance(y, sr=P, metres=float(metres), reverb=0.0) if metres > 1.0 else y
    if coast <= 0:
        return d
    late = _distance(y, sr=P, metres=2.0 * float(coast) + float(metres), reverb=0.0)
    k = _n(2.0 * float(coast) / 343.0)
    out = np.zeros(len(d) + k)
    out[:len(d)] = d
    out[k:k + len(late)] += 0.5 * late
    return out


def _horn_out(y, length, metres, coast, sr):
    db = horn_spec(float(length))[1]
    return _out(_place(y * 10.0 ** ((db - HORN_REF_DB) / 20.0), metres, coast), sr, None, fade=0.02)


def _horn_freq(length, freq):
    lo, hi = horn_spec(float(length))[0]
    return float(np.sqrt(lo * hi)) if freq <= 0 else float(freq)   # the geometric middle of the band


@sfx("ship_horn", "Ship's whistle by COLREG annex III: two reed pipes, fundamental by ship length (70-200 Hz at 200 m and over), 143 dB at 1 m down to 120. Distance, echo off the coast. Levels are kept.",
     length=(250.0, "ship's length, m: sets the band and the level"), freq=(0.0, "fundamental, Hz; 0 = the middle of the band"), signal=("prolonged", "prolonged (5 s) | short (1 s) | underway | stopped | sail"),
     distance=(1.0, "m to the listener"), coast=(0.0, "m to a coast that echoes it; 0 = none"), dur=(0.0, "s of a single blast; 0 = the signal's"), seed=(0, "variation"))
def ship_horn(sr=DEFAULT_SR, length=250.0, freq=0.0, signal="prolonged", distance=1.0, coast=0.0, dur=0.0, seed=0):
    table = {**SIGNALS, "prolonged": (PROLONGED,), "short": (SHORT,)}
    seq = table[_kind(signal, table, "ship_horn signal")]
    if dur > 0 and len(seq) == 1:
        seq = (float(np.clip(dur, 0.8, 30.0)),)
    return _horn_out(_blasts(seq, _horn_freq(length, freq), int(seed)), length, distance, coast, sr)


BELL_HUM = 440.0                                            # Hz at the 300 mm mouth; f ~ thickness / mouth^2 ~ 1 / mouth for bells of one profile
BELL_RATE = 4.0                                             # strokes per second, "rapid"
BELL_RING = 5.0                                             # s of ringing


def _bell_rms(y):
    return float(np.std(y[: min(len(y), _n(BELL_RING))]))


@sfx("ship_bell", "Ship's bell: rapid strokes for 5 s (rule 35), mouth of 300 mm or more, 110 dB at 1 m (annex III), on the church bell model. Level kept against ship_horn.",
     mouth=(300.0, "mm, 300 or more for ships of 20 m and over"), hammer=(0.03, "hammer / bell mass, 0.03 or more"), ring=(BELL_RING, "s of strokes"), rate=(BELL_RATE, "strokes per second"), seed=(0, "variation"))
def ship_bell(sr=DEFAULT_SR, mouth=300.0, hammer=0.03, ring=BELL_RING, rate=BELL_RATE, seed=0):
    if float(mouth) < 300.0 or float(hammer) < 0.03:
        raise ValueError("ship_bell: annex III wants a mouth of 300 mm or more and a hammer of 3 % of the bell's mass or more")
    rng = np.random.default_rng(int(seed))
    ring, rate = float(np.clip(ring, 0.5, 30.0)), float(np.clip(rate, 0.5, 12.0))
    tone = church_bell(BELL_HUM * 300.0 / float(mouth), 0.3, sr=P, vel=0.8, decay=2.5, bright=min(1.0 + 10.0 * (float(hammer) - 0.03), 2.5))  # a heavier hammer, a harder blow
    y = np.zeros(_n(ring) + len(tone))
    for t in np.arange(0.0, ring - 0.5 / rate, 1.0 / rate):
        _put(y, tone, float(t) + rng.uniform(0.0, 0.015), rng.uniform(0.85, 1.0))
    y = y / (_bell_rms(y) + 1e-12) * 10.0 ** ((110.0 - HORN_REF_DB) / 20.0)
    return _fin(y, sr, None, fade=0.002, tail=0.05)


@sfx("fog_signal", "COLREG rule 35 sound signals in fog: underway one prolonged blast, stopped two, sailing one prolonged and two short, each at most 2 minutes apart; anchored: the bell for 5 s at most a minute apart.",
     signal=("underway", "underway | stopped | sail | anchored"), length=(250.0, "ship's length, m"), repeats=(1, "times the signal is given"), interval=(-1.0, "s between signals, -1 or more than the rule's 120 (bell 60) = the rule's"),
     distance=(1.0, "m to the listener"), coast=(0.0, "m to a coast that echoes it"), seed=(0, "variation"))
def fog_signal(sr=DEFAULT_SR, signal="underway", length=250.0, repeats=1, interval=-1.0, distance=1.0, coast=0.0, seed=0):
    s = _kind(signal, (*SIGNALS, "anchored"), "fog_signal")
    top = BELL_INTERVAL if s == "anchored" else FOG_INTERVAL
    every = top if interval <= 0 else min(float(interval), top)
    if s == "anchored":
        one = ship_bell(sr=P, seed=seed)
    else:
        one = _horn_out(_blasts(SIGNALS[s], _horn_freq(length, 0.0), int(seed)), length, distance, coast, P)
    r = max(int(repeats), 1)
    every = max(every, len(one) / P + 0.5 if r > 1 else every)
    y = np.zeros(int(every * P) * (r - 1) + len(one))
    for i in range(r):
        y[int(every * P) * i:int(every * P) * i + len(one)] += one
    return _out(y, sr, None, fade=0.02)


DIAPHONE_F, GRUNT_S = 240.0, 0.8                            # Hz in 180-300; s


@sfx("foghorn", "Diaphone foghorn: an air siren, its rotor chopping the air at the note, and at the end the pressure falls and the note drops an octave in the grunt.",
     kind=("diaphone", "diaphone"), freq=(DIAPHONE_F, "Hz, 180-300"), dur=(4.0, "s of the blast, grunt included"), grunt=(GRUNT_S, "s of the final fall"), distance=(1.0, "m to the listener"), seed=(0, "variation"))
def foghorn(sr=DEFAULT_SR, kind="diaphone", freq=DIAPHONE_F, dur=4.0, grunt=GRUNT_S, distance=1.0, seed=0):
    """180-300 Hz, final fall of one octave. The note follows the falling pressure: the rotor is air-driven."""
    _kind(kind, ("diaphone",), "foghorn")
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 1.0, 30.0))
    g = float(np.clip(grunt, 0.1, 0.6 * dur))
    n, m = _n(dur), _n(g)
    press = np.minimum(np.arange(n) / _n(0.25), 1.0)
    fall = np.zeros(n)
    fall[n - m:] = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    f = float(freq) * 2.0 ** -fall
    amp = press * (1.0 - 0.6 * fall)
    amp[n - _n(0.15):] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(_n(0.15)) / _n(0.15))
    pulse = np.maximum(np.sin(2.0 * np.pi * np.cumsum(f) / P), 0.0) ** 2
    y = _lp(_unit(pulse - pulse.mean()), 2500.0, 2) * amp + 0.05 * _band(rng.standard_normal(n), 300.0, 3000.0) * amp
    return _fin(_place(_unit(y) * 0.2, float(distance), 0.0), sr, None, fade=0.005, tail=0.05)


# =============================================================================================
# SMALL CRAFT
# =============================================================================================
OUT_GEAR, OUT_BLADES = 2.0, 3                               # crank / propeller
OUT_CYL = {4: 6, 2: 3}                                      # 6 cylinders 4 strokes, 3 cylinders 2 strokes, both 275 Hz at 5500 rpm
ER16 = ((3.0, 134.0), (16.0, 171.0))                        # m/s, dB under water
OUT_SKIRT = 20.0                                            # dB per decade under 70 Hz (surface dipole) and over 400 Hz (cavitation)
SLAP_RATE = 2.0                                             # hull blows / s: wave encounters at planing speed in a short chop


def boat_level(speed):
    """Broadband dB under water, 134 at 3 m/s to 171 at 16 m/s, a power law in speed between."""
    (a, x), (b, z) = ER16
    return x + (z - x) * np.log(np.clip(speed, 1.0, 25.0) / a) / np.log(b / a)


def _wake(speed, n, dur, seed):
    """Light surf and small bubbles, unit RMS."""
    w = _fit(surf(sr=P, dur=max(dur, 10.0), distance=25.0, period_min=1.0, period_max=2.0, bubble_rate=800.0 + 150.0 * float(speed), seed=seed), n)
    b = _fit(bubbles(sr=P, dur=dur, rate=60.0, size=3.0, seed=seed), n)
    return _unit(_unit(w) + 0.4 * _unit(b))


def _hull_slaps(rate, n, rng, seed):
    """Very short splashes on the hull's modes, at about ``rate`` per second, unit peak."""
    y = np.zeros(n)
    t = rng.uniform(0.05, 0.3)
    i = 0
    while t < n / P - 0.25:
        _put(y, splash(sr=P, size=0.25, dur=0.2, droplets=0, seed=int(seed) + i), t, rng.uniform(0.6, 1.0))
        t += rng.uniform(0.6, 1.4) / rate
        i += 1
    hull = _modes(_lp(y, 500.0), (95.0, 170.0, 310.0), 0.12, (1.0, 0.7, 0.3))
    return y + hull / (np.max(np.abs(hull)) + 1e-12)


@sfx("outboard", "Outboard motor: exhaust through the hub under water (low-pass), propeller tone, light wake. Six cylinders 4-stroke or three 2-stroke, both 275 Hz at 5500 rpm. planing adds the hull's blows. listen=under: 70-400 Hz peak, 134 to 171 dB from 3 to 16 m/s.",
     strokes=(4, "4 (6 cyl) | 2 (3 cyl, `two_stroke`)"), rpm=(5500.0, "crankshaft, 800-5500"), load=(0.7, "0..1"), speed=(10.0, "m/s"), planing=(0, "1 = on the plane: hull slamming"), slap=(SLAP_RATE, "planing: blows per second"), listen=("air", "air | under"), dur=(4.0, "s"), seed=(0, "variation"))
def outboard(sr=DEFAULT_SR, strokes=4, rpm=5500.0, load=0.7, speed=10.0, planing=0, slap=SLAP_RATE, listen="air", dur=4.0, seed=0):
    if int(strokes) not in OUT_CYL:
        raise ValueError(f"outboard strokes {strokes!r}: use 4 | 2")
    if listen not in ("air", "under"):
        raise ValueError(f"outboard listen {listen!r}: use air | under")
    dur = float(np.clip(dur, 0.5, 60.0))
    n = _n(dur)
    rng = np.random.default_rng(int(seed))
    bpf = float(rpm) / OUT_GEAR * OUT_BLADES / 60.0
    prop = _lines(n, bpf, (1.0, 0.5, 0.25), rng)
    if listen == "under":                                   # pressure spectrum flat in 70-400 Hz, skirts of OUT_SKIRT
        rolloff = lambda f: -OUT_SKIRT * np.log10(np.maximum(70.0 / f, 1.0)) - OUT_SKIRT * np.log10(np.maximum(f / 400.0, 1.0))
        y = _unit(_shaped(n, rng, rolloff, 0.0) + 0.002 * prop)
        return _out(y * 10.0 ** ((boat_level(float(speed)) - BOAT_REF_DB) / 20.0), sr)
    C = OUT_CYL[int(strokes)]
    r = np.full(n, float(np.clip(rpm, 300.0, 9000.0)))
    if int(strokes) == 2:
        eng = two_stroke(sr=P, rpm=r, cylinders=C, load=load, dur=dur, seed=seed)
    else:
        eng = pipe_engine(sr=P, dur=dur, rpm=r, cylinders=C, load=load, cycle=4, unequal=0.0, seed=seed)
    y = _unit(_lp(_fit(eng, n), 900.0, 2)) + 0.15 * prop + 0.2 * min(float(speed) / 10.0, 2.0) * _wake(speed, n, dur, seed)   # immersion low-pass
    if planing:
        y = y + 3.0 * min(float(speed) / 10.0, 2.0) * _hull_slaps(float(np.clip(slap, 0.5, 6.0)), n, rng, seed)
    return _fin(y, sr, 0.7, fade=0.01, tail=0.05)


JET_RPM, JET_FREE = 7000.0, 8500.0                          # rpm; free-revving out of the water, at the limiter
JET_BLADES = 3                                              # impeller on the crankshaft: 350 Hz at 7000 rpm
JET_PCT = (122.0, 137.0, 149.0)                             # dB under water, 95th / 50th / 5th percentiles


def _jet_jump(n, start=0.35, length=0.3):
    """0 in the water, 1 in the air, with a .12 s edge."""
    a, b, e = int(start * n), int((start + length) * n), _n(0.12)
    g = np.zeros(n)
    g[a:b] = 1.0
    return np.clip(np.convolve(g, np.ones(e) / e, mode="same"), 0.0, 1.0), b


@sfx("jet_ski", "Jet ski: 3-cylinder 4-stroke, 175 Hz at 7000 rpm; 3-blade impeller, 350 Hz; the jet of water. jump=1: out of the water the engine unloads and free-revs, the jet stops, the hull lands. listen=under: 100 Hz-10 kHz, levels as measured (122/137/149 dB).",
     rpm=(JET_RPM, "crankshaft"), load=(0.8, "0..1"), jump=(0, "1 = a jump in the middle of the render"), listen=("air", "air | under"), dur=(4.0, "s"), seed=(0, "variation"))
def jet_ski(sr=DEFAULT_SR, rpm=JET_RPM, load=0.8, jump=0, listen="air", dur=4.0, seed=0):
    if listen not in ("air", "under"):
        raise ValueError(f"jet_ski listen {listen!r}: use air | under")
    dur = float(np.clip(dur, 0.5, 60.0))
    n = _n(dur)
    rng = np.random.default_rng(int(seed))
    f_imp = float(rpm) * JET_BLADES / 60.0
    if listen == "under":
        band = lambda f: -120.0 * np.log10(np.maximum(100.0 / f, 1.0)) - 120.0 * np.log10(np.maximum(f / 9000.0, 1.0))   # skirts set so the power stays in 100 Hz-10 kHz (from 10 kHz at -60 dB per decade: 14 % out, measured)
        x = _shaped(n, rng, band, 0.0)
        y = _unit(x) + 0.5 * _lines(n, f_imp, (1.0, 0.5, 0.25, 0.12), rng)
        e = _lfn(rng, n, 0.7)
        q = np.interp(e, np.sort(e), np.linspace(0.0, 1.0, n))      # rank: its percentiles are then those of the table, the smooth curve keeps no click
        db = np.interp(q, (0.05, 0.5, 0.95), JET_PCT)
        return _out(_unit(y) * 10.0 ** ((db - BOAT_REF_DB) / 20.0), sr)
    air, end = _jet_jump(n) if jump else (np.zeros(n), n)
    r = float(rpm) + (JET_FREE - JET_RPM) * float(rpm) / JET_RPM * air
    ld = float(np.clip(load, 0.0, 1.0)) * (1.0 - 0.85 * air)
    eng = _unit(_fit(pipe_engine(sr=P, dur=dur, rpm=r, cylinders=3, load=ld, cycle=4, unequal=0.0, seed=seed), n))
    wet = 1.0 - air
    imp = np.zeros(n)
    ph = 2.0 * np.pi * np.cumsum(r * JET_BLADES / 60.0) / P
    for k, g in ((1, 1.0), (2, 0.5), (3, 0.25)):
        imp += g * np.sin(k * ph)
    jet = _unit(_fit(air_leak(sr=P, dur=dur, pressure=0.8, orifice=8.0, seed=seed), n))   # direct noise from exhaust and jet
    y = eng + 0.25 * imp * wet + 0.5 * jet * wet
    if jump:
        _put(y, splash(sr=P, size=0.6, dur=0.6, droplets=4, seed=seed), end / P, 8.0)   # standing over the engine's peak
    return _fin(y, sr, 0.7, fade=0.01, tail=0.05)


# =============================================================================================
# SAILBOAT
# =============================================================================================
SHROUD_M, SAIL_WIND = 0.005, 10.0                           # 5 mm shroud at 10 m/s: St .2 gives 400 Hz
FLAP_RATE, HALYARD_RATE = 3.0, 2.5                          # per second: a luffing sail, a slack halyard


@sfx("sailboat", "Under sail: shroud singing in the wind (5 mm at 10 m/s, 400 Hz), a sail flapping, a halyard against the mast, water on the hull.",
     wind=(SAIL_WIND, "m/s"), shroud=(5.0, "mm"), flap=(FLAP_RATE, "sail flaps / s"), halyard=(HALYARD_RATE, "halyard hits / s"), dur=(8.0, "s"), seed=(0, "variation"))
def sailboat(sr=DEFAULT_SR, wind=SAIL_WIND, shroud=5.0, flap=FLAP_RATE, halyard=HALYARD_RATE, dur=8.0, seed=0):
    """The string is the harmonic the shedding locks to: with a low fundamental, `foley.aeolian_harp` jumps eta as the gusts move, and amax and the lock step with it (measured: .027 to .096 in one sample)."""
    dur = float(np.clip(dur, 1.0, 60.0))
    n = _n(dur)
    rng = np.random.default_rng(int(seed))
    harp = _fit(_aeolian_harp(sr=P, wind=wind, gust=0.2, notes=f"{0.2 * float(wind) / (float(shroud) * 1e-3)}hz", diameter=float(shroud), dur=dur, seed=seed), n)
    sail = _fit(cloth(sr=P, kind="flap", dur=dur, rate=float(flap), weight=0.8, seed=seed), n)
    y = np.zeros(n)
    t = rng.exponential(1.0 / float(halyard))
    tick = clink(sr=P, object="bell", size=1.5, force=0.5, seed=seed)
    while t < dur - 0.3:
        _put(y, tick, t, rng.uniform(0.4, 1.0))
        t += 0.08 + rng.exponential(1.0 / float(halyard))
    hull = _fit(surf(sr=P, dur=max(dur, 10.0), distance=15.0, period_min=2.0, period_max=4.0, bubble_rate=800.0, seed=seed), n)
    return _fin(0.5 * _unit(harp) + 0.5 * _unit(sail) + 0.3 * y + 0.4 * _unit(hull), sr, 0.7, fade=0.05, tail=0.1)


# =============================================================================================
# SUBMARINE
# =============================================================================================
SUB_RNS = {"electric": 100.0, "nuclear": 110.0}             # dB at 1 kHz, 4 kn; the self-noise at its own sonar (80, 90) is not used here
SUB_VREF, SUB_RPM, SUB_BLADES = 8.0, 60.0, 7                # 60 rpm x 7 = 7 Hz at 8 kn: a skewed 7-blade propeller
SUB_MACHINES = (50.0, 100.0, 150.0, 300.0)                  # Hz: the mains and their harmonics, pumps and generators


def _sub_outside(kind, speed, n, rng):
    """Machines (their continuum at RNS and their lines) dominate slowly; flow and blade comb, 60 log V as a ship and equal to them at 8 kn, dominate above. No cavitation, submerged.
    -20 log f slope, lines 25 dB over the continuum, comb 10 dB."""
    base = lambda f: SUB_RNS[kind] - 20.0 * np.log10(np.maximum(f, 20.0) / 1000.0)
    machines = _shaped(n, rng, base, SUB_REF_DB)
    lines = sum(_lines(n, f, (_db_amp(float(base(f)) + 25.0, SUB_REF_DB), 0.2 * _db_amp(float(base(f)) + 25.0, SUB_REF_DB)), rng) for f in SUB_MACHINES)
    bpf = SUB_RPM * speed / SUB_VREF * SUB_BLADES / 60.0
    comb = _lines(n, bpf, [_db_amp(float(base(max(bpf, 20.0))) + 10.0, SUB_REF_DB) / k for k in range(1, 9)], rng)
    flow = _shaped(n, rng, base, SUB_REF_DB) + comb
    return machines + lines, flow * np.std(machines + lines) / np.std(flow) * (speed / SUB_VREF) ** 3


@sfx("submarine", "Submarine. view=out: machine tonals and a soft continuum below 8 kn, flow and propulsion above, no cavitation; 100 or 110 dB at 1 kHz at 4 kn. view=in: machines, sonar ping, pressure creaks in the hull, a ballast blow.",
     view=("out", "out | in"), kind=("electric", "electric | nuclear"), speed=(4.0, "knots"), dur=(10.0, "s"), seed=(0, "variation"))
def submarine(sr=DEFAULT_SR, view="out", kind="electric", speed=4.0, dur=10.0, seed=0):
    k = _kind(kind, SUB_RNS, "submarine")
    if view not in ("out", "in"):
        raise ValueError(f"submarine view {view!r}: use out | in")
    dur = float(np.clip(dur, 1.0, 120.0))
    n = _n(dur)
    rng = np.random.default_rng(int(seed))
    V = float(np.clip(speed, 1.0, 30.0))
    if view == "out":
        return _out(sum(_sub_outside(k, V, n, rng)), sr)
    # A scene, no model of its own: timing and levels are design choices.
    hum = _unit(_lines(n, 50.0, (1.0, 0.5, 0.3, 0.15), rng)) + 0.15 * _unit(_lp(rng.standard_normal(n), 300.0))
    y = 0.08 * hum
    _put(y, sonar(sr=P, freq=1500, range_m=750.0, seed=seed), 0.1 * dur, 0.5)
    _put(y, creak(sr=P, material="iron", force="ramp", level=0.8, dur=min(1.5, 0.2 * dur), seed=seed), 0.4 * dur, 0.5)
    _put(y, metal_plate(sr=P, size=1.0, thickness=4.0, decay=3.0, vel=0.3, bright=0.2), 0.42 * dur, 0.15)
    blow = min(2.0, 0.25 * dur)
    _put(y, air_leak(sr=P, dur=blow, pressure=0.7, orifice=20.0, seed=seed), 0.65 * dur, 0.3)
    _put(y, bubbles(sr=P, dur=blow, rate=150.0, size=5.0, underwater=1.0, seed=seed), 0.65 * dur, 0.5)
    return _fin(y, sr, 0.7, fade=0.02, tail=0.05)


# =============================================================================================
# LAPPING WATER
# =============================================================================================
LAPS = {"clapot": dict(period=(1.2, 2.2), band=(120.0, 900.0), modes=(95.0, 170.0, 310.0), hollow=0.9, bubbles=40.0),       # against a hull
        "quai": dict(period=(1.8, 3.0), band=(250.0, 2500.0), modes=(), hollow=0.0, bubbles=80.0),                           # on stone, gurgling back
        "baignade": dict(period=(1.0, 1.8), band=(400.0, 3500.0), modes=(), hollow=0.0, bubbles=25.0)}                       # around a swimmer
# Periods inside 1-3 s, a short wind chop.


@sfx("lap", "Water lapping: a slow beat of small waves with bubbles. clapot against a hull, with its hollow modes; quai on stone, gurgling back; baignade around a swimmer, small splashes.",
     kind=("clapot", "clapot | quai | baignade"), dur=(8.0, "s"), level=(0.7, "peak"),
     plume=(0.0, "m: radius of the bubble plume each wave traps (0 = none; 0.1 = a 47 Hz thump, 0.75 = 6 Hz)"),
     seed=(0, "variation"))
def lap(sr=DEFAULT_SR, kind="clapot", dur=8.0, level=0.7, plume=0.0, seed=0):
    k = _kind(kind, LAPS, "lap")
    s = LAPS[k]
    dur = float(np.clip(dur, 1.0, 60.0))
    n = _n(dur)
    rng = np.random.default_rng(int(seed))
    noise = _band(rng.standard_normal(n), *s["band"])
    env = np.zeros(n)
    t = rng.uniform(0.0, 0.3)
    shape = np.arange(_n(0.8)) / P
    shape = (1.0 - np.exp(-shape / 0.05)) * np.exp(-shape / 0.2)
    shape[-_n(0.05):] *= np.linspace(1.0, 0.0, _n(0.05))
    times = []
    while t < dur - 0.2:
        times.append(t)
        _put(env, shape, t, rng.uniform(0.7, 1.0))
        t += rng.uniform(*s["period"])
    y = _unit(noise * env)
    swell = _lp(rng.standard_normal(n), 60.0, 2) * env
    y = y + 0.4 * _unit(swell) + 0.25 * _unit(_fit(bubbles(sr=P, dur=dur, rate=s["bubbles"], size=3.0, seed=seed), n)) * (0.2 + env / (env.max() + 1e-12))
    if s["hollow"]:
        hull = _modes(_lp(noise * env, 500.0), s["modes"], 0.2, (1.0, 0.7, 0.3))
        y = y + s["hollow"] * hull / (np.max(np.abs(hull)) + 1e-12)
    if k == "quai":
        y = y + 0.5 * _unit(_fit(gurgle(sr=P, dur=dur, rate=1.0, size=0.4, seed=seed), n))
    if k == "baignade":
        for i, t in enumerate(times[::2]):
            _put(y, splash(sr=P, size=0.15, dur=0.5, droplets=2, seed=int(seed) + i), t, 0.6)
    if plume > 0:
        y = y + PLUME_LEVEL * _unit(plume_tone(n, times, float(np.clip(plume, 0.05, 1.0))))
    return _fin(y, sr, float(level), fade=0.02, tail=0.1)


PLUME_Q, PLUME_LEVEL = 3.0, 0.5


def plume_tone(n, times, radius):
    """Collective oscillation of the plume each wave entrains: a damped sine at ``plume_f0(radius)``, Q 3, from each wave."""
    f0 = plume_f0(radius)
    m = min(n, _n(PLUME_Q / f0 * 3.0) + 1)
    tt = np.arange(m) / P
    ring = np.exp(-np.pi * f0 * tt / PLUME_Q) * np.sin(2.0 * np.pi * f0 * tt)
    out = np.zeros(n)
    for t in times:
        _put(out, ring, t + 0.05, 1.0)
    return out


def plume_f0(radius, beta=0.34, p0=101325.0, rho=1000.0):
    """Collective oscillation of a bubble plume (Minnaert law for a cloud of void fraction beta, radius m):
    f0 = sqrt(3 p0 / (rho beta)) / (2 pi R). 0.75 m -> 6.3 Hz, 0.1 m -> 47 Hz."""
    return float(np.sqrt(3.0 * p0 / (rho * beta)) / (2.0 * np.pi * float(radius)))
