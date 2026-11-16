"""Instruments the catalogue lacked (Plan sons, L19): a voice for each, built on the library's physical kernels.

Sourced: the chanter and drones of the bagpipes (BAG, PET in taxonomie_universelle.md), the construction and playing
of waterphone, berimbau, shamisen and talking drum (Ruiz i Carulla, Annexos, root_text/02.MRiC_ANNEXOS.txt
l.13327, 14208, 14399, 14566). The books only name calliope and flexatone (a-composers-guide l.8533, 8331), so those
two voices rest on their construction (research/SOURCES.md).
"""
from __future__ import annotations

import numpy as np

from . import filters as F
from . import osc as O
from .acoustic import _body, _lip_reed, _pluck_string, banjo, guitar_body, strings
from .contact import resonate
from .core import DEFAULT_SR, samples
from .dsp import tape
from .fdstring import fd_string
from .friction import FS as BOW_FS, _BAR_MATERIAL, _bow_force, _edges, _friction_note, _gate, _tone, CLAMPED, beam_shape, bowed_modal, sympathetic
from .instruments import _modes, _n, _off, _vibrato, decay_env, instrument
from .notes import to_freq as note_to_freq
from .physics import MembraneDrum
from .tubes import C_AIR, FS, UNFLANGED, WebsterTube, _edge, _finish, _noise, _pipe, _through, reed_cone, reed_tube

# Output gains that put each voice near -12 dB K-weighted at vel 0.9 (same contract as friction._LEVEL).
_LEVEL = {"ukulele": 0.63, "bagpipes": 0.94, "mellotron": 0.5, "zither": 0.75, "shofar": 1.6, "traditional_horn": 0.82,
          "slide_whistle": 0.82, "waterphone": 0.385, "talking_drum": 0.305, "berimbau": 1.2, "shamisen": 0.82, "calliope": 0.89,
          "flexatone": 0.43, "risset_arpeggio": 0.202, "gendyn": 0.56}


def tension_ratio(squeeze, k):
    """f / f0 of a membrane or plate whose tension grows by k*squeeze: f = f0 sqrt(1 + k squeeze), since f ~ sqrt(T)."""
    return np.sqrt(1.0 + k * np.clip(squeeze, 0.0, 1.0))


def _gesture(n, sr, dur, amount, curve):
    """Hand gesture over a note: hold | rise | fall | wave (up to `amount` at mid-note and back)."""
    if curve not in ("hold", "rise", "fall", "wave"):
        raise ValueError("curve must be hold, rise, fall or wave")
    x = np.clip(np.arange(n) / (max(dur, 1e-3) * sr), 0.0, 1.0)
    s = x * x * (3.0 - 2.0 * x)
    return float(amount) * {"hold": np.ones(n), "rise": s, "fall": 1.0 - s, "wave": 0.5 - 0.5 * np.cos(2 * np.pi * x)}[curve]


def _settle(y, sr):
    """A few ms of silence after a tail: render_note's 4 ms end ramp then falls on nothing. On a faded tail its corner read as a click
    (measured at -60 dB, 4 ms before the end of zither and horn notes)."""
    return np.concatenate((y, np.zeros(samples(0.006, sr))))


# ---------------------------------------------------------------------------------------- plucked
UKULELE_OPEN = ("G4", "C4", "E4", "A4")     # GCEA, re-entrant (taxonomie_universelle.md:381: usage, from memory)
UKULELE_SCALE_M = 0.345                     # soprano-concert scale length


@instrument("ukulele", "Ukulele: nylon strings on a small box, re-entrant GCEA; the note is stopped on the nearest open string.",
            family="plucked", span=("C4", "C6"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings"))
def ukulele(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    below = [f for f in map(note_to_freq, UKULELE_OPEN) if f <= freq * 1.001]
    length = UKULELE_SCALE_M * max(below, default=freq) / freq          # stopped length: f ~ 1 / L
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="nylon", kind="finger", hardness=0.5, position=0.2, t60=1.6,
                      tail=0.7, release=0.08, length=length)            # pluck, ring
    y = F.lowpass(guitar_body(y, sr, 0, 0.0), min(3800.0, 0.42 * sr), sr, 0.7)
    return _settle(y * _edge(y.shape[0], 1, 0.1 * sr), sr) * _LEVEL["ukulele"]


ZITHER_COURSE = "C3 G3 C4 G4"               # open accompaniment strings that ring along


@instrument("zither", "Zither: koto's hard pick near the bridge on a course of steel strings over a box, the open strings ringing along.",
            family="plucked", span=("C3", "C6"), ring=(0.15, "0..1 level of the open strings that ring along"),
            glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings"))
def zither(freq, dur, sr=DEFAULT_SR, vel=1.0, ring=0.15, glide=0.0):
    # (taxonomie_universelle.md:399) koto's pluck, steel, course 2 cents apart, box peak
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.85, position=0.07, t60=3.0,
                      tail=1.4, release=0.15, detune=2.0)
    # The bank answers every period of the string with a burst up to 9 kHz (measured: one click per period at C3):
    # open strings struck through a bridge do not carry that, so only its low part is kept.
    wet = sympathetic(y, sr, tuning=ZITHER_COURSE, kind="string", decay=2.5, mix=float(np.clip(ring, 0.0, 1.0))) - y
    y = y + F.lowpass(wet, 1600.0, sr, 0.7, order=2)
    y = _body(y, sr, [(260, 1.5, 3.0)], min(4200.0, 0.42 * sr))
    return _settle(y * _edge(y.shape[0], 1, 0.1 * sr), sr) * _LEVEL["zither"]


SHAMISEN_SAWARI_BELOW = "G3"                # second open string; below it the note is on the buzzing low string


@instrument("shamisen", "Shamisen: banjo-like string over a skin with a large plectrum; the low string alone buzzes on its altered nut (sawari).",
            family="plucked", span=("D3", "D6"), sawari=(0.6, "0..1 buzz of the low string (Annexos l.14410: a less firm contact at the nut)"),
            skin=(0.5, "0..1 slap of the plectrum on the skin"))
def shamisen(freq, dur, sr=DEFAULT_SR, vel=1.0, sawari=0.6, skin=0.5):
    y = banjo(freq, dur, sr, vel)
    n = y.shape[0]
    if freq < note_to_freq(SHAMISEN_SAWARI_BELOW):
        # Same buzz layer as acoustic.sitar, blooming after the pluck. Not fdstring.sitar_fd, the brick the plan names: measured at D3,
        # its bridge leaves under 2 % of the energy above 8 f0 where the free string has 83 %, so it damps rather than buzzes.
        t = np.arange(n) / sr
        b = F.lowpass(F.peak(F.highpass(np.tanh(6.0 * y), 700, sr), 2400, sr, 1.2, 4.0), 2600.0, sr, 0.7, order=2) * (1 - np.exp(-t / 0.05)) * np.exp(-t / 0.9)   # its own lowpass: raw, the pick pulses read as one click per period at D3
        y = F.lowpass(y + float(np.clip(sawari, 0.0, 1.0)) * b, min(4200.0, 0.42 * sr), sr, 0.7, order=2)
    a = samples(0.001, sr)
    slap = F.bandpass(O.white(n, seed=19), 700.0, sr, 1.2) * decay_env(n, 0.012, sr) * float(np.clip(skin, 0.0, 1.0)) * vel
    slap[:a] *= np.linspace(0.0, 1.0, a)
    return (y + 0.25 * slap) * _LEVEL["shamisen"]                       # slap band and level


# ---------------------------------------------------------------------------------------- blown, reeds
BAG_CHANTER_A = 481.0       # chanter low A: 476-486 Hz today, 441 in 1885 (BAG, research extract, not read in full)


@instrument("bagpipes", "Bagpipes: conical double-reed chanter over three cylinder-reed drones on A (two tenors, one bass).",
            family="wind", span=("G4", "A5"), drones=(1.0, "0..1 level of the drones"),
            chanter_a=(BAG_CHANTER_A, "Hz of the chanter's A, which the drones sit on (476-486 today, 441 in 1885)"))
def bagpipes(freq, dur, sr=DEFAULT_SR, vel=1.0, drones=1.0, chanter_a=BAG_CHANTER_A):
    y = reed_cone(freq, dur, sr, vel, tongue=0.3)
    g = float(np.clip(drones, 0.0, 1.0))
    if g > 0.0:
        # Drones on A (PET); tenors one octave and bass two octaves below the chanter's A, their levels
        # (the levels keep the melody's pitch readable over them: at twice these an autocorrelation reads C#5 20 cents off)
        for f, a in ((chanter_a / 2.0, 0.15), (chanter_a / 2.0, 0.15), (chanter_a / 4.0, 0.2)):
            d = reed_tube(f, dur, sr, vel, tongue=0.3)
            k = min(y.shape[0], d.shape[0])
            y[:k] += g * a * d[:k]
    return y * _LEVEL["bagpipes"]


# ---------------------------------------------------------------------------------------- lip reeds
_SHOFAR_L = 0.42            # (rg shofar: nothing) length and throat of a ram's horn


def _shofar_area(x):
    """Curved, flattened horn: a cone with a waist where it bends. """
    return (1.0 + 2.2 * x) ** 2 * (1.0 - 0.6 * np.exp(-((x - 0.4) / 0.1) ** 2))


def _horn(freq, dur, sr, vel, body, top, attack, release, harm=6.0):
    """Lip-reed waveguide (acoustic._lip_reed), key-tracked 12 dB/oct so that the kernel's once-per-period corner (its output
    area is clipped at 1: measured on wg.brass alone, one click per period below C4) does not reach the output, and raised-cosine edges."""
    y = _lip_reed(freq, dur, sr, vel, body, top, attack, release)
    y = F.lowpass(y, float(np.clip(harm * freq, 900.0, min(top, 0.42 * sr))), sr, 0.7, order=2)
    y = F.highpass(y, 20.0, sr, order=2)        # the kernel's DC: faded later, it leaves a tail in render_note's blocker that its end ramp clicks on
    return _settle(y * _edge(y.shape[0], attack * sr, release * sr), sr) * 0.55


def _shofar_tube():
    return WebsterTube(_SHOFAR_L, _shofar_area, radius_m=0.006, left="closed", right="unflanged", loss=2.0)


@instrument("shofar", "Shofar: lip-reed through a short irregular horn, raw and nasal.", family="brass", span=("G3", "G5"))
def shofar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _horn(freq, dur, sr, vel, [(1100, 500, 0.5)], 4200.0, 0.03, 0.07)
    t = _shofar_tube()
    y = _through(y, sr, lambda u: t.run(u)["u_out"], 0.7)               # the horn rings past the fade: close it again
    return _settle(y * _edge(y.shape[0], 1, 0.03 * sr), sr) * _LEVEL["shofar"]


TRAD_HORNS = {          # (taxonomie_universelle.md:270) body peaks (Hz, width, gain), ceiling, attack, release
    "conch": ([(650, 280, 1.4)], 2400.0, 0.06, 0.10),
    "alphorn": ([(320, 120, 1.0), (850, 300, 0.5)], 2800.0, 0.12, 0.18),
    "war_horn": ([(900, 400, 0.8), (1900, 500, 0.5)], 4200.0, 0.03, 0.08)}
TRAD_HARM = {"conch": 6.0, "alphorn": 6.0, "war_horn": 12.0}     # harmonics kept; the war horn is the raw one


@instrument("traditional_horn", "Natural horns on the lip-reed waveguide: conch shell, alphorn or war horn.", family="brass",
            span=("C3", "C5"), kind=("alphorn", "conch | alphorn | war_horn"))
def traditional_horn(freq, dur, sr=DEFAULT_SR, vel=1.0, kind="alphorn"):
    if kind not in TRAD_HORNS:
        raise ValueError(f"kind must be one of {sorted(TRAD_HORNS)}")
    body, top, attack, release = TRAD_HORNS[kind]
    return _horn(freq, dur, sr, vel, body, top, attack, release, TRAD_HARM[kind]) * _LEVEL["traditional_horn"]


# ---------------------------------------------------------------------------------------- tape
@instrument("mellotron", "Mellotron: a few bowed players read from tape, with flutter and a head that runs out of tape.",
            family="keys", span=("G2", "F5"), flutter=(0.2, "tape speed wobble, percent"), wow=(0.1, "slow drift, percent"),
            tape_s=(8.0, "seconds of tape behind each key; the note stops there"), seed=(0, "flutter and hiss"))
def mellotron(freq, dur, sr=DEFAULT_SR, vel=1.0, flutter=0.2, wow=0.1, tape_s=8.0, seed=0):
    # (taxonomie_universelle.md:394: no source read) 3 players, range, 8 s of tape, flutter depth
    held = min(float(dur), float(tape_s))
    y = strings(freq, held, sr, vel, attack=0.08, release=0.12, players=3)
    return tape(y, sr, wow=wow, flutter=flutter, drive=1.2, bump=1.5, hf=9000, hiss=-62, seed=int(seed)) * _LEVEL["mellotron"]


# ---------------------------------------------------------------------------------------- pipes
SLIDE_RADIUS_M = 0.008


def slide_length(freq):
    """Length of a stopped pipe, end correction included: f = c / (4 (L + 0.6133 a)), Bilbao (9.11)."""
    return C_AIR / (4.0 * np.asarray(freq, float)) - UNFLANGED * SLIDE_RADIUS_M


@instrument("slide_whistle", "Slide whistle: stopped pipe whose piston travels at a steady speed, so the pitch follows 1/L; odd harmonics and a little air.",
            family="wind", span=("C5", "C7"), sweep=(0.0, "semitones the note starts away from the written pitch (+ above, - below)"),
            slide=(0.0, "seconds of piston travel; 0 = the whole note"), breath=(0.15, "0..1 air noise in the pipe's mode"), seed=(0, "noise"))
def slide_whistle(freq, dur, sr=DEFAULT_SR, vel=1.0, sweep=0.0, slide=0.0, breath=0.15, seed=0):
    rel = 0.06
    n = _n(dur, sr, rel)
    t = np.arange(n) / sr
    L1 = float(slide_length(freq))
    L0 = float(slide_length(freq * 2.0 ** (float(sweep) / 12.0)))
    x = np.clip(t / max(float(slide) if slide > 0.0 else float(dur), 1e-3), 0.0, 1.0)
    f = C_AIR / (4.0 * (L0 + (L1 - L0) * x + UNFLANGED * SLIDE_RADIUS_M))
    w = 2 * np.pi * O.phase(f, n, sr)
    y = np.sin(w) + 0.05 * np.sin(3 * w) + 0.012 * np.sin(5 * w)        # stopped pipe: odd series. their levels
    air = F.lowpass(np.random.default_rng(int(seed)).standard_normal(n), 60.0, sr, order=2) * 12.0   # narrow band around the mode
    y = y * (1.0 + float(np.clip(breath, 0.0, 1.0)) * air)
    return y * _edge(n, 0.015 * sr, rel * sr) * (0.4 + 0.6 * vel) * 0.5 * _LEVEL["slide_whistle"]


STEAM_R, STEAM_T = 461.5, 414.0     # J/kg/K of water vapour, K of saturated steam at 40 psi gauge
STEAM_SAG = 0.045                   # fraction of the boiler's pressure lost for each further open whistle
STEAM_DRAW = 0.03                   # df/f for a relative pressure change of 1


def steam_speed(warm=1.0, temp=STEAM_T):
    """Speed of sound in the steam: sqrt(gamma R T), gamma 1.13 in wet steam up to 1.30 in dry steam."""
    return float(np.sqrt((1.13 + 0.17 * float(np.clip(warm, 0.0, 1.0))) * STEAM_R * temp))


def calliope_diameter(freq):
    """Bore of the whistle for this note: 100 mm at C3 to 40 mm at G5, on a log scale."""
    x = np.clip(np.log2(float(freq) / 130.81) / np.log2(783.99 / 130.81), 0.0, 1.0)
    return float(0.10 * (0.04 / 0.10) ** x)


def calliope_pitch(freq, warm=1.0, pressure=1.0, sag=0):
    """Sounding pitch of a whistle tuned for `freq` with hot steam at the nominal pressure, and the relative pressure it gets."""
    p = float(np.clip(pressure, 0.2, 2.0)) * (1.0 + STEAM_SAG) / (1.0 + STEAM_SAG * (1.0 + max(int(sag), 0)))
    return float(freq) * steam_speed(warm) / steam_speed(1.0) * (1.0 + STEAM_DRAW * (p - 1.0)), p


@instrument("calliope", "Calliope: steam whistles, each a short stopped pipe blown by steam at 40 psi, so sharper than in air; cold steam is flat and spits, a chord of open whistles draws the boiler down.",
            family="keys", span=("C3", "G5"), steam=(0.25, "0..1 hiss of the steam in the pipe"), waver=(0.06, "semitones of pressure waver"),
            diameter=(0.0, "whistle bore, m (0 = by the note, 100 mm at C3 to 40 mm at G5)"), pressure=(1.0, "boiler pressure relative to nominal"),
            warm=(1.0, "0 cold wet steam (flat, spitting) .. 1 hot dry steam"), sag=(0, "other whistles open at the same time: the pressure sags"),
            overblow=(0.0, "relative pressure above which the whistle jumps to its 3rd harmonic (0 = never, about 1.5 for a hard blast)"), seed=(0, "noise"))
def calliope(freq, dur, sr=DEFAULT_SR, vel=1.0, steam=0.25, waver=0.06, diameter=0.0, pressure=1.0, warm=1.0, sag=0, overblow=0.0, seed=0):
    f0, p = calliope_pitch(freq, warm, pressure, sag)
    d = calliope_diameter(freq) if diameter <= 0 else float(np.clip(diameter, 0.01, 0.2))
    tb = _pipe(round(f0, 4), d / 2.0, 1.0, round(steam_speed(warm), 1))
    n = int((dur + 0.5) * FS)
    rng = np.random.default_rng(int(seed))
    f = _vibrato(f0, n, FS, rate=6.5, depth=float(waver), delay=0.05, ramp=0.1)
    if overblow > 0.0 and p >= float(overblow) and 3.0 * f0 < 0.4 * FS:
        f = f * (1.0 + 2.0 * np.clip((np.arange(n) / FS - 0.01) / 0.04, 0.0, 1.0))
    w = 2 * np.pi * O.phase(f, n, FS)
    u = np.sin(w) + 0.5 * np.sin(2 * w) / 2.0 ** 1.5 + 0.5 * np.sin(3 * w) / 3.0 ** 1.5
    hiss = float(np.clip(steam, 0.0, 1.0)) * p ** 2 * _noise(n, int(seed))
    cold = 1.0 - float(np.clip(warm, 0.0, 1.0))
    for t in rng.uniform(0.0, dur, rng.poisson(10.0 * cold * dur)):
        k, m = int(t * FS), int(0.006 * FS)
        m = min(m, n - k)
        hiss[k:k + m] += cold * rng.uniform(0.5, 1.5) * rng.standard_normal(m) * np.hanning(2 * m + 2)[m + 1:2 * m + 1]
    env = np.zeros(n)
    hold = int(dur * FS) + 1
    env[:hold] = _edge(hold, 0.015 * FS, 0.03 * FS)
    chuff = 0.6 * float(steam) * _noise(n, int(seed) + 1, 1500.0) * np.exp(-np.arange(n) / (0.02 * FS))
    out = np.asarray(tb.run((u + hiss) * env + chuff)["p_out"], float)
    k = min(int(0.2 * FS), out.size)
    out[-k:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(k) + 1) / k)       # the low pipes are still ringing at the end of the buffer
    y = _finish(out, sr, tone=3600.0, peak=0.5 * (0.35 + 0.65 * min(float(vel), 1.2)) * p ** 0.5)
    return y * _LEVEL["calliope"]


# ---------------------------------------------------------------------------------------- bowed and struck metal
WATERPHONE_RODS = (1.0, 0.94, 1.07, 0.89, 1.13)     # lengths of the neighbouring rods, relative; f ~ 1 / L^2
WATERPHONE_FORMANTS = ((380.0, 2.5, 1.0), (1050.0, 3.0, 0.8), (2300.0, 3.5, 0.5))   # vessel (Hz, Q, gain)
WATERPHONE_K = 1.0          # f = f0 sqrt(1 + k slosh) (Plan sons L19), k calibrated
WATERPHONE_RATE = 0.7       # Hz of the rocked vessel


def vessel_formants(n, sr, slosh, rate=WATERPHONE_RATE):
    """Formant ratio over time: the water moves as the vessel rocks (Annexos l.13330-13332)."""
    s = 0.5 - 0.5 * np.cos(2 * np.pi * float(rate) * np.arange(n) / sr)
    return tension_ratio(float(np.clip(slosh, 0.0, 1.0)) * s, WATERPHONE_K)


@instrument("waterphone", "Waterphone: unequal rods welded around a thin metal vessel holding water, bowed or struck; moving the water sweeps the formants.",
            family="bowed", span=("F3", "F6"), slosh=(0.5, "0..1 how much the water moves the vessel's formants"),
            action=("bowed", "bowed | struck"), neighbours=(0.25, "0..1 the other rods ringing along"), bright=(1.0, "0.5 mellow .. 2 bright"))
def waterphone(freq, dur, sr=DEFAULT_SR, vel=1.0, slosh=0.5, action="bowed", neighbours=0.25, bright=1.0):
    ratios = np.array(CLAMPED)
    if action == "bowed":
        w = beam_shape("clamped", 0.97) ** 2
        vb = 0.2 * (0.4 + 0.6 * float(np.clip(vel, 0.0, 1.5)))
        fb = _bow_force(1.0, freq, vb, w[0])
        xi = _BAR_MATERIAL["steel"][1] * ratios

        def run(scale, t, n_on):
            k = int(t * BOW_FS)
            return bowed_modal(freq * scale * ratios, xi, w, fb * _gate(k, n_on, min(0.05, 0.4 * dur), 0.05), vb, k / BOW_FS)["out"]

        y = _tone(_friction_note(run, freq, dur, sr, 0.6, 0.2), freq, sr, bright)
    elif action == "struck":
        n = _n(dur, sr, 1.5)
        y = _modes(freq, n, sr, [(float(r), a, 1.2 / r ** 0.5) for r, a in zip(ratios, (1.0, 0.3, 0.1, 0.04))]) * vel
    else:
        raise ValueError("action must be bowed or struck")
    n = y.shape[0]
    t = np.arange(n) / sr
    g = float(np.clip(neighbours, 0.0, 1.0)) * np.max(np.abs(y))
    for i, L in enumerate(WATERPHONE_RODS[1:]):
        f = freq / L ** 2
        if f < 0.45 * sr:
            y = y + g * 0.5 * np.sin(2 * np.pi * f * t + i) * decay_env(n, 0.7, sr) * (1.0 - np.exp(-t / 0.02))
    y = resonate(y, sr, body="circ_plate", material="steel", size=0.25, mix=0.2, modes=12, tail=0.0)
    n = y.shape[0]
    sweep = vessel_formants(n, sr, slosh)
    out = 0.55 * y
    for fc, q, a in WATERPHONE_FORMANTS:
        out = out + a * F.svf(y, "bandpass", np.minimum(fc * sweep, 0.45 * sr), sr, q)
    return _edges(out, sr) * _LEVEL["waterphone"]


# ---------------------------------------------------------------------------------------- struck bow and heads
BERIMBAU_GOURD_HZ = 420.0   # open gourd's air resonance (a 15 cm gourd)


def gourd_freq(opening):
    """Helmholtz resonance f ~ sqrt(A): closing the mouth against the body lowers it (Annexos l.14213: "wah-wah")."""
    return BERIMBAU_GOURD_HZ * np.sqrt(np.clip(opening, 0.04, 1.0))


@instrument("berimbau", "Berimbau: steel string on a bow struck with a stick, heard through an open gourd the player closes against the body (wah); a stone on the string bends the pitch up.",
            family="plucked", span=("A2", "C5"), gourd=(0.8, "0 closed against the body .. 1 open"),
            wah=(0.0, "Hz the gourd is moved at (0 = held; swings between closed and `gourd`)"), press=(0.0, "semitones the stone raises the string (Annexos l.14215)"))
def berimbau(freq, dur, sr=DEFAULT_SR, vel=1.0, gourd=0.8, wah=0.0, press=0.0):
    f = float(freq) * 2.0 ** (float(press) / 12.0)
    # strike point, stiffness, ring
    y = fd_string(f, dur, sr, vel, excitation="strike", position=0.11, B=1e-4, t60_lo=1.6, t60_hi=0.5, pickup=0.3)
    n = y.shape[0]
    op = np.full(n, float(np.clip(gourd, 0.0, 1.0)))
    if wah > 0.0:
        op = op * (0.5 - 0.5 * np.cos(2 * np.pi * float(wah) * np.arange(n) / sr))
    # The string and the bow alone barely reach the air (Annexos l.14211); the air gain follows the opening. levels
    out = 0.35 * y + (0.3 + 1.4 * op) * F.svf(y, "bandpass", gourd_freq(op), sr, 3.0)
    return out * _LEVEL["berimbau"]


TALKING_K = 3.0             # f = f0 sqrt(1 + k squeeze); k calibrated to an octave at full squeeze: (chunk_39.md:23 "k a calibrer")


@instrument("talking_drum", "Talking drum: hourglass drum whose two heads share their cords; the arm squeezes the cords and the pitch slides, f = f0 sqrt(1 + k squeeze).",
            family="percussion", span=("G2", "G4"), squeeze=(0.0, "0 relaxed .. 1 full arm on the cords (an octave)"),
            curve=("hold", "hold | rise | fall | wave: the arm's gesture over the note"), ring=(0.35, "seconds of ring"))
def talking_drum(freq, dur, sr=DEFAULT_SR, vel=1.0, squeeze=0.0, curve="hold", ring=0.35):
    n = _n(dur, sr, 0.5)
    f = float(freq) * tension_ratio(_gesture(n, sr, dur, squeeze, curve), TALKING_K)
    ph = np.cumsum(f) / sr
    t = np.arange(n) / sr
    y = np.zeros(n)
    ratios = MembraneDrum._RATIOS[:4]                      # ideal circular membrane
    tau = max(float(ring), 0.02)
    for r, a, d in zip(ratios, (1.0, 0.35, 0.2, 0.1), (1.0, 0.45, 0.3, 0.2)):   # levels and relative decays
        fac = np.exp(-t / (tau * d))
        y += a * fac * (np.sin(2 * np.pi * r * ph) + 0.3 * np.sin(2 * np.pi * r * 1.004 * ph))   # the far head, 7 cents off
    slap = F.bandpass(O.white(n, seed=23), 900.0, sr, 0.9) * decay_env(n, 0.008, sr) * (0.15 + 0.35 * vel)
    y = y + slap
    y = F.peak(y, 180.0, sr, 1.2, 3.0) * _off(n, dur, sr, 0.12)                 # shell.
    y *= _edge(n, 0.002 * sr, 0.1 * sr)                                         # no corner at the strike or in the damped tail
    return y * (0.35 + 0.65 * vel) * _LEVEL["talking_drum"]


FLEXATONE_K = 3.0           # f = f_min (1 + k s): the thumb's force bends the blade, the pitch follows the curvature
FLEXATONE_PLATE = (1.0, 2.3, 3.9)          # trapped modes of the S-curved blade, relative to the first
FLEXATONE_TAU = (0.35, 0.2, 0.12)          # s, decay of each mode
FLEXATONE_BALLS = (8.0, 9.0)               # Hz, the two wooden balls on their wire springs
FLEXATONE_GAIN = 0.26


def _shake(n, sr, rate, rng):
    """The two balls driven by the shaken frame: (phase of the frame in turns, [(sample, strength)] of the strikes). The frame swings at rate/2, its pace wandering a few percent; the ball on one side strikes when it reaches the blade, the other ball on the other side."""
    step = max(int(sr // 4000), 1)
    dt = step / sr
    wander = F.lowpass(rng.standard_normal(n), 1.0, sr)
    wander = 0.04 * wander / (np.std(wander) + 1e-12)
    fs = 0.5 * float(rate) * (1.0 + wander)
    turns = np.cumsum(fs) / sr
    ws = 2 * np.pi * 0.5 * float(rate)
    hits, gap = [], 1.0
    for sign, fb in zip((1.0, -1.0), FLEXATONE_BALLS):
        wb, z = 2 * np.pi * fb, 0.15
        amp = 2.5 * gap * np.hypot(wb * wb - ws * ws, 2 * z * wb * ws)
        x = v = 0.0
        last = -n
        for i in range(0, n, step):
            a = amp * np.cos(2 * np.pi * turns[i])
            v += dt * (-wb * wb * x - 2 * z * wb * v - a)
            x += dt * v
            if sign * x >= gap and sign * v > 0.0 and i - last > 0.4 * sr / (0.5 * float(rate)):
                last = i
                hits.append((i, min(abs(v) / (2.5 * gap * ws), 1.5)))
                x, v = sign * gap, -0.4 * v
    return turns, hits


@instrument("flexatone", "Flexatone: shaken frame, two wooden balls on springs strike a curved steel blade in turn; the thumb bends the blade and the pitch slides up to two octaves.",
            family="percussion", span=("C4", "C6"), rate=(11.0, "strikes per second (two balls)"), squeeze=(0.0, "0..1 thumb on the blade (0 = lowest note, 1 = two octaves up)"),
            curve=("hold", "hold | rise | fall | wave: the thumb's gesture over the note"), vibrato=(10.0, "cents of pitch swing at the shake, synchronous with the strikes"),
            seed=(0, "timing of the shake"))
def flexatone(freq, dur, sr=DEFAULT_SR, vel=1.0, rate=11.0, squeeze=0.0, curve="hold", vibrato=10.0, seed=0):
    n = _n(dur, sr, 0.5)
    rng = np.random.default_rng(int(seed))
    turns, hits_l = _shake(n, sr, rate, rng)
    f = float(freq) * (1.0 + FLEXATONE_K * _gesture(n, sr, dur, squeeze, curve)) * 2.0 ** (float(vibrato) / 1200.0 * np.sin(2 * np.pi * turns))
    ph = 2 * np.pi * np.cumsum(f) / sr
    hits = np.zeros(n)
    for k, a in hits_l:
        if k < samples(dur, sr):
            hits[k] += a * (0.35 + 0.65 * vel)
    from scipy.signal import lfilter
    hits = F.lowpass(hits, min(2000.0, 0.42 * sr), sr)             # the ball's contact, about 0.3 ms
    y = np.zeros(n)
    for r, a, tau in zip(FLEXATONE_PLATE, (1.0, 0.35, 0.15), FLEXATONE_TAU):
        if r * float(freq) * (1.0 + FLEXATONE_K) < 0.42 * sr:
            # One phase per mode, fed by every strike: a mode already ringing keeps its phase.
            y += a * F.lowpass(lfilter([1.0], [1.0, -np.exp(-1.0 / (tau * sr))], hits), 1200.0, sr) * np.sin(r * ph)
    y = F.lowpass(y, min(5000.0, 0.42 * sr), sr, 0.7) * _edge(n, 0.001 * sr, 0.05 * sr)
    return y * FLEXATONE_GAIN * _LEVEL["flexatone"]


_RISSET_PARTIALS = ((1, 1.0), (5, 0.7), (6, 0.7), (7, 0.7), (8, 0.7), (9, 0.7), (10, 0.7))


@instrument("risset_arpeggio", "Risset's arpeggio: nine copies of one harmonic tone a few hundredths of a hertz apart; their beating walks an arpeggio up the harmonics. Hold long notes.",
            family="synth", span=("C2", "C4"), delta=(0.03, "Hz between neighbouring copies: the arpeggio's speed"))
def risset_arpeggio(freq, dur, sr=DEFAULT_SR, vel=1.0, delta=0.03):
    rel = 0.5
    n = _n(dur, sr, rel)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for k in range(-4, 5):
        f = float(freq) + k * float(delta)
        for h, a in _RISSET_PARTIALS:
            if h * f < 0.45 * sr:
                y += a * np.sin(2 * np.pi * h * f * t)
    env = np.interp(t, [0.0, 0.02, max(dur, 0.02), max(dur, 0.02) + rel], [0.0, 1.0, 1.0, 0.0])
    return y * env / 9.0 * (0.5 + 0.5 * vel) * _LEVEL["risset_arpeggio"]


@instrument("gendyn", "Xenakis' dynamic stochastic synthesis: a waveform of breakpoints whose amplitudes and spacings drift by "
            "second-order random walks between reflecting walls; with `wander` the pitch drifts too.",
            family="synth", span=("C2", "C5"), points=(12, "breakpoints per cycle"),
            amp_step=(0.5, "largest amplitude step per cycle (0..1)"), wander=(0.0, "segment lengths walk within +- this share: the pitch wanders"),
            seed=(0, "walk"))
def gendyn(freq, dur, sr=DEFAULT_SR, vel=1.0, points=12, amp_step=0.5, wander=0.0, seed=0):
    rel = 0.05
    n = _n(dur, sr, rel)
    y = O.gendyn(float(freq), n, sr, int(points), float(np.clip(amp_step, 0.0, 1.0)), float(np.clip(wander, 0.0, 0.9)), int(seed))
    y = F.lowpass(y - np.mean(y), min(3000.0, 8.0 * float(freq)), sr, order=2)
    t = np.arange(n) / sr
    env = np.interp(t, [0.0, 0.01, max(dur, 0.01), max(dur, 0.01) + rel], [0.0, 1.0, 1.0, 0.0])
    return y * env * (0.5 + 0.5 * vel) * _LEVEL["gendyn"]


from . import world_t  # noqa: E402,F401  (guzheng, erhu, oud, qanun, santur, winds, bonang, balafon)


# ---------------------------------------------------------------------------------------- electronic
TRAUTONIUM_FORMANTS = ((500.0, 4.0), (1500.0, 6.0))


@instrument("trautonium", "Mixtur-Trautonium: a sawtooth with its subharmonics f/2, f/3 ... f/n mixed in, through two fixed formant filters; the finger's pressure is the volume.",
            family="synth", span=("C2", "C6"),
            subs=(4, "0..6 subharmonic dividers mixed under the note"), fade=(0.6, "0..1 level of each subharmonic against the one before"),
            formant=(1.0, "0..1 share of the formant filters in the sound"), pressure=(0.5, "0..1 finger pressure on the ribbon: a softer touch has a slower attack"))
def trautonium(freq, dur, sr=DEFAULT_SR, vel=1.0, subs=4, fade=0.6, formant=1.0, pressure=0.5):
    rel = 0.12
    n = _n(dur, sr, rel)
    k = int(np.clip(subs, 0, 6))
    x = np.zeros(n)
    for d in range(1, k + 2):
        f = freq / d
        if f >= 20.0:
            x += (float(fade) ** (d - 1)) * O.saw(f, n, sr)
    x /= max(1.0, float(sum(float(fade) ** i for i in range(k + 1))))
    fm = float(np.clip(formant, 0.0, 1.0))
    if fm > 0:
        form = sum(F.bandpass(x, fc, sr, q) for fc, q in TRAUTONIUM_FORMANTS)
        x = (1.0 - fm) * x + fm * form * 0.6
    att = 0.02 + 0.2 * (1.0 - float(np.clip(pressure, 0.0, 1.0)))
    t = np.arange(n) / sr
    env = np.minimum(t / att, 1.0) * np.where(t < dur, 1.0, np.exp(-(t - dur) / (rel / 4.0)))
    return x * env * 0.55 * (0.5 + 0.5 * vel)


MARTENOT_DIFFUSERS = ("principal", "palme", "metallique")
MARTENOT_WAVES = {"sine": (1.0, 0.0, 0.0), "triangle": (0.2, 0.8, 0.0), "square": (0.0, 0.2, 0.8), "saw": (0.1, 0.0, 0.0)}


@instrument("ondes_martenot", "Ondes Martenot: the beat of two radio-frequency oscillators, a near-sine wave with its timbre switch, the ribbon's vibrato, and one of three diffusers.",
            family="synth", span=("C2", "C7"),
            wave=("sine", "sine | triangle | square | saw: the timbre selector"),
            diffuser=("principal", "principal (the plain speaker) | palme (a lyre-shaped set of strings ringing in sympathy) | metallique (a gong behind the cone)"),
            vib=(0.25, "ribbon vibrato depth in semitones"), vib_hz=(5.5, "Hz"))
def ondes_martenot(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="sine", diffuser="principal", vib=0.25, vib_hz=5.5):
    if wave not in MARTENOT_WAVES or diffuser not in MARTENOT_DIFFUSERS:
        raise ValueError("ondes_martenot wave is sine|triangle|square|saw, diffuser principal|palme|metallique")
    rel = 0.1
    n = _n(dur, sr, rel)
    t = np.arange(n) / sr
    f = freq * 2.0 ** (float(vib) * np.sin(2 * np.pi * float(vib_hz) * t) * np.minimum(t / 0.4, 1.0) / 12.0)
    ph = np.cumsum(f) / sr
    s, tr, sq = MARTENOT_WAVES[wave]
    x = s * np.sin(2 * np.pi * ph) + tr * (2.0 * np.abs(2.0 * (ph % 1.0) - 1.0) - 1.0) + sq * np.sign(np.sin(2 * np.pi * ph)) * 0.6
    if wave == "saw":
        x = 0.8 * (2.0 * (ph % 1.0) - 1.0) + 0.2 * np.sin(2 * np.pi * ph)
    if diffuser == "palme":
        x = x + 0.25 * _modes(freq, n, sr, [(1.0, 1.0, 1.5), (2.0, 0.5, 1.2), (3.0, 0.3, 1.0), (4.0, 0.2, 0.8)], corner=4000.0)
    elif diffuser == "metallique":
        x = x + 0.2 * _modes(freq, n, sr, [(1.0, 0.6, 1.0), (2.76, 1.0, 1.4), (5.4, 0.7, 1.0), (8.93, 0.4, 0.7)], corner=5000.0)
    env = np.minimum(t / 0.015, 1.0) * np.where(t < dur, 1.0, np.exp(-(t - dur) / (rel / 4.0)))
    return x * env * 0.55 * (0.5 + 0.5 * vel)
