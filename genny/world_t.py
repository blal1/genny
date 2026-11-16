"""World instruments (lot T): guzheng, erhu and morin khuur, oud, qanun, santur, shakuhachi, ney, bansuri, duduk, bonang, balafon.

Engines (data in the dictionaries below, nothing per instrument in the code):
    plucked   modal string with a tension contour (pressed or bent string), loss law, contact filter, courses, body bank
    bowed     Bilbao bow on a string-mode bank with a pitch contour, bridge on a modelled membrane (skin)
    blown     jet flute / double reed of the library, bent by a time warp of the note
    struck    modal kettle gong and wooden bar with a resonator and a buzzing membrane
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import jn_zeros, jv

from . import filters as F
from . import osc as O
from .acoustic import _body, _jet_voice
from .core import DEFAULT_SR, samples
from .friction import FS as FSB, _autotune, _bow_force, _spectral_peak, _gate, _to_sr, bowed_modal, sympathetic
from .instruments import instrument
from .notes import to_freq

MATERIALS = {"steel": (2.0e11, 7850.0), "nylon": (3.0e9, 1140.0), "gut": (4.0e9, 1300.0), "horsehair": (5.0e9, 1100.0)}

# ------------------------------------------------------------------------------------------------ data: plucked
STRINGS = {
    "guzheng": dict(material="steel", tension=80.0, L_ref=1.15, f_ref=73.4, L_exp=0.45, press_side=0.35, t60=3.5, t60_hi=0.35,
                    position=0.11, exc="pluck", contact=0.0012, courses=1, spread=0.0, noise=0.05, release=0.5,
                    ring=("D3 A3 D4 A4", 0.08), body=[(95, 2.5, 4.0), (180, 3.0, 3.0), (340, 3.5, 3.0), (620, 4.0, 2.0), (1100, 4.0, 1.5)], top=6500.0),
    "oud": dict(material="nylon", tension=45.0, open=("C2", "F2", "A2", "D3", "G3", "C4"), scale=0.61, wound_below=150.0, press_side=0.3,
                t60=1.8, t60_hi=0.25, position=0.17, exc="pluck", contact=0.0006, courses=2, spread=4.0, noise=0.08, release=0.25,
                ring=("", 0.0), body=[(140, 3.0, 5.0), (260, 3.5, 3.0), (430, 4.0, 3.0), (800, 4.0, 2.0)], top=5200.0),
    "qanun": dict(material="nylon", tension=28.0, L_ref=0.70, f_ref=220.0, L_exp=0.5, press_side=0.2, t60=2.4, t60_hi=0.3,
                  position=0.08, exc="pluck", contact=0.0004, courses=3, spread=2.0, noise=0.07, release=0.4,
                  ring=("", 0.0), body=[(180, 3.0, 4.0), (350, 3.5, 3.5), (700, 4.0, 3.0), (1400, 4.0, 3.0), (2600, 3.0, 2.0)], top=7000.0),
    "santur": dict(material="steel", tension=60.0, L_ref=0.65, f_ref=220.0, L_exp=0.5, press_side=0.2, t60=2.0, t60_hi=0.25,
                   position=0.2, exc="strike", contact=0.0008, courses=4, spread=1.5, noise=0.06, release=0.4,
                   ring=("", 0.0), body=[(150, 3.0, 4.0), (300, 3.5, 3.0), (620, 4.0, 3.0), (1250, 4.0, 2.0)], top=7500.0),
}

# ------------------------------------------------------------------------------------------------ data: bowed
BOWED = {
    "erhu": dict(open=("D4", "A4"), L_open=0.38, bow_dist=0.05, B=3e-5, t60=2.5, t60_hi=0.4, force=1.0, speed=0.25,
                 skin_radius=0.0273, skin_tension=730.0, skin_density=0.4, skin_loss=0.1, bridge_r=0.3, skin_gain=9.0, cavity=(260.0, 4.0, 4.0), top=7500.0),
    "gaohu": dict(open=("G4", "D5"), L_open=0.34, bow_dist=0.045, B=3e-5, t60=2.0, t60_hi=0.35, force=1.0, speed=0.25,
                  skin_radius=0.0235, skin_tension=730.0, skin_density=0.4, skin_loss=0.1, bridge_r=0.3, skin_gain=9.0, cavity=(330.0, 4.0, 4.0), top=8000.0),
    "zhonghu": dict(open=("G3", "D4"), L_open=0.44, bow_dist=0.055, B=3e-5, t60=2.5, t60_hi=0.4, force=1.0, speed=0.25,
                    skin_radius=0.0330, skin_tension=730.0, skin_density=0.4, skin_loss=0.1, bridge_r=0.3, skin_gain=9.0, cavity=(200.0, 4.0, 4.0), top=7000.0),
    "yehu": dict(open=("G4", "D5"), L_open=0.36, bow_dist=0.05, B=3e-5, t60=1.6, t60_hi=0.3, force=1.1, speed=0.25,
                 skin_radius=0.0450, skin_tension=420.0, skin_density=0.9, skin_loss=0.14, bridge_r=0.3, skin_gain=8.0, cavity=(300.0, 3.0, 3.0), top=6500.0),
    "morin_khuur": dict(open=("F2", "C3"), L_open=0.60, bow_dist=0.06, B=1e-5, t60=1.4, t60_hi=0.25, force=1.0, speed=0.3,
                        skin_radius=0.0600, skin_tension=250.0, skin_density=0.45, skin_loss=0.12, bridge_r=0.35, skin_gain=8.0, cavity=(140.0, 3.0, 4.0), top=6000.0),
}
SKIN_MODES = ((0, 1), (1, 1), (2, 1), (0, 2), (3, 1), (1, 2))

# ------------------------------------------------------------------------------------------------ data: blown
BLOWN = {
    "shakuhachi": dict(kind="jet", jet=0.32, breath=0.30, top=4200.0, gain=1.0, span=("D4", "D6")),
    "ney": dict(kind="jet", jet=0.30, breath=0.20, top=4600.0, gain=1.0, span=("D4", "D6")),
    "bansuri": dict(kind="jet", jet=0.32, breath=0.12, top=4800.0, gain=1.0, span=("G3", "G6")),
    "duduk": dict(kind="reed", cone=0.15, embouchure=1.2, breath=0.05, bright=0.8, span=("A3", "A5")),
}

# ------------------------------------------------------------------------------------------------ data: struck
BONANG = {"barung": dict(ratios=(1.0, 1.57, 2.29, 3.12, 4.07), taus=(1.6, 1.0, 0.7, 0.4, 0.25), amps=(1.0, 0.7, 0.5, 0.3, 0.2), ombak=0.0),
          "panerus": dict(ratios=(1.0, 1.55, 2.25, 3.05, 4.0), taus=(1.1, 0.7, 0.45, 0.3, 0.2), amps=(1.0, 0.7, 0.5, 0.3, 0.2), ombak=0.0)}
BALAFON = dict(tuning="xylophone", gourd_q=9.0, mirliton_hz=900.0, body=[(120, 2.0, 2.0)])

GUZHENG_TUNING = "D2 E2 F#2 A2 B2 D3 E3 F#3 A3 B3 D4 E4 F#4 A4 B4 D5 E5 F#5 A5 B5 D6"

_CONTOUR = dict(bend=(0.0, "cents the pitch moves over the note (+ up): a pressed, bent or slid note"),
                bend_at=(0.5, "0..1 fraction of the note at which the bend starts"),
                bend_time=(0.25, "s the bend takes (smooth, monotone)"),
                scoop=(0.0, "cents under the written pitch at the start, rising into it"),
                scoop_time=(0.08, "s the scoop takes"),
                vib=(0.0, "vibrato depth in cents (peak)"), vib_hz=(5.5, "vibrato rate, Hz"), vib_delay=(0.3, "s before the vibrato is at full depth"))


# ------------------------------------------------------------------------------------------------ pitch contour
def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def cents_contour(n, rate, dur, bend=0.0, bend_at=0.5, bend_time=0.25, scoop=0.0, scoop_time=0.08, vib=0.0, vib_hz=5.5, vib_delay=0.3):
    t = np.arange(n) / float(rate)
    c = np.zeros(n)
    if scoop:
        c -= scoop * (1.0 - _smooth(t / max(scoop_time, 1e-3)))
    if bend:
        c += bend * _smooth((t - bend_at * dur) / max(bend_time, 1e-3))
    if vib:
        c += vib * np.sin(2.0 * np.pi * vib_hz * t) * _smooth((t - 0.1) / max(vib_delay, 1e-3))
    return c


def _ratio(n, rate, dur, **kw):
    return 2.0 ** (cents_contour(n, rate, dur, **kw) / 1200.0)


# ------------------------------------------------------------------------------------------------ plucked string physics
def string_geometry(freq, spec):
    """Length, tension, linear density, diameter, axial stiffness EA and inharmonicity B of the string that sounds `freq`."""
    E, rho = MATERIALS[spec["material"]]
    if spec.get("open"):
        opens = [to_freq(x) for x in spec["open"]]
        fo = max([o for o in opens if o <= freq * 1.001], default=min(opens))
        L = spec["scale"] * fo / freq
    else:
        fo = freq
        L = spec["L_ref"] * (spec["f_ref"] / freq) ** spec["L_exp"]
    T = float(spec["tension"])
    mu = T / (2.0 * L * freq) ** 2
    d = math.sqrt(4.0 * mu / (rho * math.pi))
    B = math.pi ** 2 * E * (math.pi * d ** 4 / 64.0) / (T * L * L)
    if fo < spec.get("wound_below", 0.0):
        B *= 0.05
    return dict(L=L, T=T, mu=mu, d=d, EA=E * mu / rho, B=min(B, 2e-3), f_open=fo)


def string_freq(T, mu, L):
    return math.sqrt(T / mu) / (2.0 * L)


def press_cents(freq, depth_mm, spec):
    """Pitch rise (cents) of a string pressed `depth_mm` on the side behind the bridge: the stretch d^2 / 2b raises the tension by EA dL / L."""
    g = string_geometry(freq, spec)
    b, d = spec["press_side"], depth_mm * 1e-3
    dL = math.sqrt(b * b + d * d) - b
    r2 = 1.0 + g["EA"] * dL / (g["L"] + b) / g["T"]
    return 1200.0 * math.log2(math.sqrt(r2))


def press_depth(freq, cents, spec):
    """Depth (mm) of the press that bends the note by `cents`."""
    g = string_geometry(freq, spec)
    b = spec["press_side"]
    dL = g["T"] * (2.0 ** (cents / 600.0) - 1.0) * (g["L"] + b) / g["EA"]
    return 1e3 * math.sqrt(2.0 * b * dL + dL * dL)


def bridge_lengths(tuning=GUZHENG_TUNING, spec=None):
    """Sounding length (m) of every string of a zheng tuning."""
    spec = spec or STRINGS["guzheng"]
    return [string_geometry(to_freq(n), spec)["L"] for n in tuning.split()]


def _modal_string(freq, n, sr, dur, vel, spec, ratio, position, B, t60, t60_hi, release, rng_seed=0):
    nm = int(min(140, 0.45 * sr / (freq * 1.2) / 1.0))
    k = np.arange(1, max(nm, 2) + 1)
    fk = freq * k * np.sqrt(1.0 + B * k * k) / math.sqrt(1.0 + B)
    keep = fk * float(ratio.max()) < 0.45 * sr
    k, fk = k[keep], fk[keep]
    fhi = max(4000.0, 3.0 * freq)
    t60k = np.maximum(t60 * (t60_hi / t60) ** (np.log(np.maximum(fk, freq) / freq) / math.log(fhi / freq)), 0.02)
    sig = 6.91 / t60k
    tau = spec["contact"] * (2.0 - 1.4 * vel)
    a = np.sin(k * np.pi * position) / k * np.exp(-0.5 * (np.pi * fk * tau) ** 2)
    ph0 = 2.0 * np.pi * np.cumsum(ratio) / sr
    t = np.arange(n) / sr
    y = np.zeros(n)
    for i in range(k.size):
        if abs(a[i]) < 1e-5:
            continue
        y += a[i] * np.exp(-sig[i] * t) * np.sin(ph0 * fk[i])
    if dur < n / sr:
        y *= np.where(t > dur, np.exp(-(t - dur) / release), 1.0)
    return y


def _plucked(freq, dur, sr, vel, spec, position=None, ratio_kw=None, cents_extra=0.0, release=None, t60=None, tension=None):
    spec = dict(spec)
    if tension is not None:
        spec["tension"] = float(tension)
    g = string_geometry(freq, spec)
    t60 = float(spec["t60"] if t60 is None else t60)
    tail = min(t60, 2.0)
    n = samples(dur + tail, sr)
    ratio = _ratio(n, sr, dur, **(ratio_kw or {})) * 2.0 ** (cents_extra / 1200.0)
    pos = float(np.clip(spec["position"] if position is None else position, 0.03, 0.5))
    nc = int(spec["courses"])
    rel = float(spec["release"] if release is None else release)
    y = np.zeros(n)
    for c in range(nc):
        det = spec["spread"] * (c - 0.5 * (nc - 1)) * (1.0 if nc > 1 else 0.0)
        r = ratio * 2.0 ** (det / 1200.0)
        yc = _modal_string(freq, n, sr, dur, vel, spec, r, pos, g["B"], t60, spec["t60_hi"], rel)
        sh = samples(0.0006 * c, sr)
        y += np.concatenate((np.zeros(sh), yc[:n - sh]))
    y /= max(np.max(np.abs(y)), 1e-12)
    click = F.bandpass(O.white(n, seed=11), 3200.0, sr, 1.0) * np.exp(-np.arange(n) / (0.004 * sr)) * spec["noise"] * (0.1 + 0.9 * vel * vel)
    y = y + click
    ring_tuning, ring_mix = spec.get("ring", ("", 0.0))
    return y, ring_tuning, ring_mix


def _finish_plucked(y, sr, spec, vel, ring_tuning, ring_mix, ring):
    mix = float(np.clip(ring, 0.0, 1.0)) if ring is not None else ring_mix
    if ring_tuning and mix > 0:
        wet = sympathetic(y, sr, tuning=ring_tuning, kind="string", decay=2.5, mix=mix) - y
        y = y + F.lowpass(wet, 1600.0, sr, 0.7, order=2)
    y = _body(y, sr, spec["body"], min(spec["top"], 0.42 * sr))
    y = y / max(np.max(np.abs(y)), 1e-12) * 0.8 * (0.35 + 0.65 * vel)
    return np.concatenate((y, np.zeros(samples(0.006, sr))))



@instrument("guzheng", "Guzheng: 21 steel-core strings over movable bridges on a long paulownia board; the left hand presses behind the bridge, "
            "raising the tension (slides, rou xian vibrato), the right hand plucks with plectra.", family="plucked", span=("D2", "D6"),
            position=(0.11, "pluck point, 0..0.5 of the sounding length from the bridge end"),
            press_mm=(0.0, "left-hand press depth in mm behind the bridge: the tension rise sets the pitch rise (overrides `bend`)"),
            tension=(80.0, "string tension in N: sets diameter, stiffness (inharmonicity) and the press response"),
            t60=(3.5, "ring time of the fundamental, s"),
            ring=(0.08, "0..1 sympathetic ring of the open strings"), **_CONTOUR)
def guzheng(freq, dur, sr=DEFAULT_SR, vel=1.0, position=0.11, press_mm=0.0, tension=80.0, t60=3.5, ring=0.08, bend=0.0, bend_at=0.5,
            bend_time=0.25, scoop=0.0, scoop_time=0.08, vib=0.0, vib_hz=5.5, vib_delay=0.3):
    spec = STRINGS["guzheng"]
    if press_mm:
        bend = bend + press_cents(freq, press_mm, dict(spec, tension=tension))
    kw = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y, rt, rm = _plucked(freq, dur, sr, vel, spec, position, kw, t60=t60, tension=tension)
    return _finish_plucked(y, sr, spec, vel, rt, rm, ring)


@instrument("oud", "Oud: fretless short-neck lute, five double courses and a single bass, nylon strings plucked with a long plectrum over a deep bowl.",
            family="plucked", span=("C2", "C5"), position=(0.17, "pluck point, 0..0.5 of the sounding length from the bridge"),
            spread=(4.0, "cents between the two strings of a course"), **_CONTOUR)
def oud(freq, dur, sr=DEFAULT_SR, vel=1.0, position=0.17, spread=4.0, bend=0.0, bend_at=0.5, bend_time=0.25, scoop=0.0, scoop_time=0.08,
        vib=0.0, vib_hz=5.5, vib_delay=0.3):
    spec = dict(STRINGS["oud"], spread=spread)
    kw = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y, rt, rm = _plucked(freq, dur, sr, vel, spec, position, kw)
    return _finish_plucked(y, sr, spec, vel, rt, rm, None)


@instrument("qanun", "Qanun: trapezoidal zither, three nylon strings per note plucked with plectra on the index fingers, the bridge on a fish-skin patch; "
            "mandal levers retune a note by microtones.", family="plucked", span=("A2", "E6"),
            position=(0.08, "pluck point, 0..0.5 of the sounding length from the bridge"),
            mandal=(0.0, "cents the lever raises the note (shortens the string): 25 = a quarter tone"), **_CONTOUR)
def qanun(freq, dur, sr=DEFAULT_SR, vel=1.0, position=0.08, mandal=0.0, bend=0.0, bend_at=0.5, bend_time=0.25, scoop=0.0, scoop_time=0.08,
          vib=0.0, vib_hz=5.5, vib_delay=0.3):
    spec = STRINGS["qanun"]
    kw = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y, rt, rm = _plucked(freq, dur, sr, vel, spec, position, kw, cents_extra=mandal)
    return _finish_plucked(y, sr, spec, vel, rt, rm, None)


@instrument("santur", "Santur: trapezoidal hammered dulcimer, courses of four steel strings struck by two light wooden mallets.",
            family="plucked", span=("D3", "D6"), position=(0.2, "strike point, 0..0.5 of the sounding length from the bridge"),
            contact=(0.0008, "mallet contact time at vel 0.5, s: shorter = brighter"))
def santur(freq, dur, sr=DEFAULT_SR, vel=1.0, position=0.2, contact=0.0008):
    spec = dict(STRINGS["santur"], contact=contact)
    y, rt, rm = _plucked(freq, dur, sr, vel, spec, position)
    return _finish_plucked(y, sr, spec, vel, rt, rm, None)


# ------------------------------------------------------------------------------------------------ bowed string and skin
def skin_modes(spec, tension_scale=1.0):
    """Frequencies (Hz) and bridge coupling of the circular skin: f = alpha / (2 pi a) sqrt(Ts / sigma)."""
    a = spec["skin_radius"]
    c = math.sqrt(spec["skin_tension"] * tension_scale / spec["skin_density"])
    out = []
    for m, nth in SKIN_MODES:
        al = float(jn_zeros(m, nth)[-1])
        out.append((al * c / (2.0 * math.pi * a), abs(float(jv(m, al * spec["bridge_r"])))))
    return out


def _skin(y, sr, spec, tension_scale):
    out = 0.5 * y
    q = 1.0 / spec["skin_loss"]
    for f, g in skin_modes(spec, tension_scale):
        if f < 0.42 * sr:
            out = out + g * (10.0 ** (spec["skin_gain"] / 20.0)) * 0.12 * F.bandpass(y, f, sr, q)
    cf, cq, cdb = spec["cavity"]
    return F.peak(out, cf, sr, cq, cdb)


def stopped_length(freq, spec):
    fo = max([to_freq(o) for o in spec["open"] if to_freq(o) <= freq * 1.001], default=min(to_freq(o) for o in spec["open"]))
    return spec["L_open"] * fo / freq


def _bowed(freq, dur, sr, vel, spec, force, speed, bow_dist, skin_tension, contour, tail=0.3):
    L = stopped_length(freq, spec)
    xb = float(np.clip((spec["bow_dist"] if bow_dist is None else bow_dist) / L, 0.03, 0.45))
    k = np.arange(1, 25)
    ratios = k * np.sqrt(1.0 + spec["B"] * k * k)
    fk = freq * ratios
    fhi = max(4000.0, 3.0 * freq)
    t60k = np.maximum(spec["t60"] * (spec["t60_hi"] / spec["t60"]) ** (np.log(fk / freq) / math.log(fhi / freq)), 0.02)
    xi = 6.91 / (2.0 * np.pi * fk * t60k)
    w = np.sin(k * np.pi * xb) ** 2
    w = w / np.max(w)
    vb = float(np.clip(speed, 0.05, 0.6)) * (0.4 + 0.6 * float(np.clip(vel, 0.0, 1.5)))
    fb = _bow_force(spec["force"] * force, freq, vb, w[0])
    n = int((dur + tail) * FSB)
    n_on = int(dur * FSB)
    r = _ratio(n, FSB, dur, **contour)

    def pilot(s, t):
        m = int(t * FSB)
        return bowed_modal(fk * s, xi, w, fb * _gate(m, m, 0.03, 0.03), vb, m / FSB)["out"]

    def play(sc):
        return bowed_modal(fk, xi, w, fb * _gate(n, n_on, 0.04, 0.06), vb, n / FSB, pitch=sc * r)["out"]

    scale = _autotune(pilot, freq * float(r[0]))
    y = play(scale)
    a, b = int(0.05 * FSB), min(int(0.45 * FSB), n_on)
    best = (np.inf, y)
    for _ in range(4):                      # the pitch of a bowed string hops by tens of cents: tune on the note itself and keep the closest
        f = _spectral_peak(y[a:b], freq * float(r[0]), span=0.25) if b - a > 2048 else 0.0
        if f <= 0.0:
            break
        err = abs(np.log2(f / (freq * float(r[0]))))
        if err < best[0]:
            best = (err, y)
        if err < 0.0025:
            break
        scale *= freq * float(r[0]) / f
        y = play(scale)
    else:
        f = _spectral_peak(y[a:b], freq * float(r[0]), span=0.25)
        if f > 0.0 and abs(np.log2(f / (freq * float(r[0])))) < best[0]:
            best = (0.0, y)
    y = best[1] if np.isfinite(best[0]) else y
    y = _to_sr(y, sr)
    y = F.dc_block(y, sr)
    y = _skin(y, sr, spec, skin_tension)
    y = F.lowpass(y, min(spec["top"], 0.42 * sr, float(np.clip(6.0 * freq, 2500.0, 4500.0))), sr, 0.7, order=2)
    y = y / max(np.max(np.abs(y)), 1e-12) * 0.8 * (0.5 + 0.5 * min(vel, 1.0))
    m = min(samples(0.004, sr), y.size // 2)
    y[:m] *= np.sin(np.linspace(0, np.pi / 2, m)) ** 2
    y[-m:] *= np.cos(np.linspace(0, np.pi / 2, m)) ** 2
    return y


_BOW = dict(force=(1.0, "bow force, relative (0.2-2.5)"), speed=(0.25, "bow speed, m/s (0.05-0.6)"),
            bow_dist=(0.0, "bow point from the bridge in m (0 = the instrument's own)"),
            skin_tension=(1.0, "skin tension, relative: the skin modes follow its square root"))


@instrument("erhu", "Erhu family: two steel strings with the horsehair bow between them, no fingerboard, the bridge on a python skin over a hexagonal "
            "sound box; every note can slide.", family="bowed", span=("D4", "D7"),
            kind=("erhu", "erhu | gaohu | zhonghu | yehu (coconut shell)"), **_BOW, **_CONTOUR)
def erhu(freq, dur, sr=DEFAULT_SR, vel=1.0, kind="erhu", force=1.0, speed=0.25, bow_dist=0.0, skin_tension=1.0, bend=0.0, bend_at=0.5,
         bend_time=0.25, scoop=0.0, scoop_time=0.08, vib=0.0, vib_hz=5.5, vib_delay=0.3):
    if kind not in BOWED or kind == "morin_khuur":
        raise ValueError("erhu kind must be erhu, gaohu, zhonghu or yehu")
    c = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    return _bowed(freq, dur, sr, vel, BOWED[kind], force, speed, bow_dist or None, skin_tension, c)


@instrument("morin_khuur", "Morin khuur: Mongolian horse-head fiddle, two horsehair strings bowed from the side, trapezoidal skin-faced body.",
            family="bowed", span=("F2", "F5"), **_BOW, **_CONTOUR)
def morin_khuur(freq, dur, sr=DEFAULT_SR, vel=1.0, force=1.0, speed=0.3, bow_dist=0.0, skin_tension=1.0, bend=0.0, bend_at=0.5, bend_time=0.25,
                scoop=0.0, scoop_time=0.08, vib=0.0, vib_hz=5.5, vib_delay=0.3):
    c = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    return _bowed(freq, dur, sr, vel, BOWED["morin_khuur"], force, speed, bow_dist or None, skin_tension, c)


# ------------------------------------------------------------------------------------------------ blown
def _warp(y, r):
    pos = np.cumsum(r) - r[0]
    return np.interp(pos, np.arange(y.size), y)


def _blown(name, freq, dur, sr, vel, breath, c, **extra):
    s = BLOWN[name]
    if s["kind"] == "jet":
        y = _jet_voice(freq, dur, sr, vel, breath if breath is not None else s["breath"], 0.0, jet=s["jet"], top=s["top"], gain=s["gain"])
    else:
        from .friction import _spectral_peak
        from .tubes import _reed_voice

        def voice(f, d):
            return _reed_voice(f, d, sr, vel, extra.get("cone", s["cone"]), 1.0, s["embouchure"], breath if breath is not None else s["breath"],
                               0.0, 0.03, 0.5, 0.0, 0.0, s["bright"], 0)
        pilot = voice(freq, min(dur, 0.4))                                  # the loose embouchure flattens the reed: pilot, then retune
        f = _spectral_peak(pilot[int(0.1 * sr):], freq, sr)
        y = voice(freq * float(np.clip(freq / f, 0.97, 1.03)) if f > 0 else freq, dur)
    if any(c.values()):
        y = _warp(y, _ratio(y.size, sr, dur, **c))
    return y


@instrument("shakuhachi", "Shakuhachi: end-blown bamboo flute with a notched rim, a very breathy jet; head tilt bends the pitch (meri) and shakes it.",
            family="wind", span=("D4", "D6"), breath=(0.30, "turbulence on the jet, 0..0.4"), **_CONTOUR)
def shakuhachi(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.30, bend=0.0, bend_at=0.5, bend_time=0.25, scoop=0.0, scoop_time=0.08, vib=0.0,
               vib_hz=5.5, vib_delay=0.3):
    c = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    return _blown("shakuhachi", freq, dur, sr, vel, breath, c)


@instrument("ney", "Ney: end-blown reed flute held at the side, a rim-blown jet with a soft, airy edge; pitch is bent by the lips.",
            family="wind", span=("D4", "D6"), breath=(0.20, "turbulence on the jet, 0..0.4"), **_CONTOUR)
def ney(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.20, bend=0.0, bend_at=0.5, bend_time=0.25, scoop=0.0, scoop_time=0.08, vib=0.0,
        vib_hz=5.5, vib_delay=0.3):
    c = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    return _blown("ney", freq, dur, sr, vel, breath, c)


@instrument("bansuri", "Bansuri: transverse bamboo flute, a clean jet over six holes, with meend slides.", family="wind", span=("G3", "G6"),
            breath=(0.12, "turbulence on the jet, 0..0.4"), **_CONTOUR)
def bansuri(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.12, bend=0.0, bend_at=0.5, bend_time=0.25, scoop=0.0, scoop_time=0.08, vib=0.0,
            vib_hz=5.5, vib_delay=0.3):
    c = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    return _blown("bansuri", freq, dur, sr, vel, breath, c)


@instrument("duduk", "Duduk: Armenian apricot-wood double reed, a wide reed on a nearly cylindrical bore: warm, nasal, with slow slides.",
            family="wind", span=("A3", "A5"), breath=(0.05, "turbulence on the mouth pressure, 0..0.5"),
            cone=(0.15, "0..1 bore taper (0 = cylinder, odd harmonics)"), **_CONTOUR)
def duduk(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.05, cone=0.15, bend=0.0, bend_at=0.5, bend_time=0.25, scoop=0.0, scoop_time=0.08, vib=0.0,
          vib_hz=5.5, vib_delay=0.3):
    c = dict(bend=bend, bend_at=bend_at, bend_time=bend_time, scoop=scoop, scoop_time=scoop_time, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    return _blown("duduk", freq, dur, sr, vel, breath, c, cone=cone)


# ------------------------------------------------------------------------------------------------ struck
def _hit_modes(freqs, taus, amps, dur_tail, sr, vel, contact):
    n = samples(dur_tail, sr)
    t = np.arange(n) / sr
    tc = contact * (1.4 - 0.8 * vel)
    y = np.zeros(n)
    for f, tau, a in zip(freqs, taus, amps):
        if f < 0.45 * sr:
            y += a * np.exp(-0.5 * (np.pi * f * tc) ** 2) * np.exp(-t / tau) * np.sin(2.0 * np.pi * f * t)
    return y


@instrument("bonang", "Bonang: gamelan kettle gong on cords, struck on its boss with a padded stick; a few inharmonic modes, the note beating with its pair.",
            family="mallet", span=("C4", "C6"), kind=("barung", "barung | panerus"),
            ombak=(3.0, "Hz of beating between the two kettles of a pair (0 = single)"),
            contact=(0.0015, "stick contact time at vel 0.5, s"))
def bonang(freq, dur, sr=DEFAULT_SR, vel=1.0, kind="barung", ombak=3.0, contact=0.0015):
    s = BONANG[kind]
    f = freq * np.array(s["ratios"]) - 0.5 * ombak * np.array(s["ratios"])     # the pair beats around the written note, not above it
    ring = max(dur, 0.2) + 1.5
    y = _hit_modes(f, s["taus"], s["amps"], ring, sr, vel, contact)
    if ombak:
        y = 0.5 * (y + _hit_modes(f + ombak * np.array(s["ratios"]), s["taus"], s["amps"], ring, sr, vel, contact))
    y = y + 0.02 * F.bandpass(O.white(y.size, seed=3), 2500.0, sr, 1.0) * np.exp(-np.arange(y.size) / (0.006 * sr))
    y = y / max(np.max(np.abs(y)), 1e-12) * 0.8 * (0.35 + 0.65 * vel)
    return np.concatenate((y, np.zeros(samples(0.006, sr))))


@instrument("balafon", "Balafon: wooden bars on a frame, each over a tuned gourd whose hole is covered with a spider-silk membrane that buzzes when played loud.",
            family="mallet", span=("C3", "C6"), buzz=(0.5, "0..1 mirliton buzz"), gourd=(1.0, "0..1 level of the gourd resonance"),
            mallet=(0.5, "mallet hardness 0..1"))
def balafon(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.5, gourd=1.0, mallet=0.5):
    from .fdstring import fd_bar
    b = BALAFON
    y = fd_bar(freq, dur, sr, vel, material="wood", tuning=b["tuning"], mallet=mallet)
    r = F.bandpass(y, freq, sr, b["gourd_q"])
    y = y + float(gourd) * 2.0 * r
    sc = max(np.max(np.abs(y)), 1e-12)
    yn = y / sc
    mir = F.highpass(np.tanh(6.0 * r / max(np.max(np.abs(r)), 1e-12)) ** 3, b["mirliton_hz"], sr)
    y = yn + float(np.clip(buzz, 0.0, 1.0)) * 0.5 * mir * (0.3 + 0.7 * vel)
    y = _body(y, sr, b["body"], min(6500.0, 0.42 * sr))
    return y / max(np.max(np.abs(y)), 1e-12) * 0.8 * (0.35 + 0.65 * vel)
