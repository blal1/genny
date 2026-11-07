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
    return F.lowpass(guitar_body(y, sr, 0, 0.0), min(3800.0, 0.42 * sr), sr, 0.7) * _LEVEL["ukulele"]


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


@instrument("calliope", "Calliope: steam whistles, each a short stopped pipe blown hard, with hiss and an unsteady pressure.",
            family="keys", span=("C4", "C6"), steam=(0.25, "0..1 hiss of the steam in the pipe"), waver=(0.06, "semitones of pressure waver"),
            diameter=(0.05, "whistle diameter, m"), seed=(0, "noise"))
def calliope(freq, dur, sr=DEFAULT_SR, vel=1.0, steam=0.25, waver=0.06, diameter=0.05, seed=0):
    # Named only in the sources (a-composers-guide l.8533); pipe = tubes._pipe, drive, hiss, waver, range
    tb = _pipe(round(float(freq), 4), float(np.clip(diameter, 0.01, 0.2)) / 2.0, 1.0)
    n = int((dur + 0.25) * FS)
    f = _vibrato(float(freq), n, FS, rate=6.5, depth=float(waver), delay=0.05, ramp=0.1)
    w = 2 * np.pi * O.phase(f, n, FS)
    u = np.sin(w) + 0.5 * np.sin(2 * w) / 2.0 ** 1.5 + 0.5 * np.sin(3 * w) / 3.0 ** 1.5
    hiss = float(np.clip(steam, 0.0, 1.0)) * _noise(n, int(seed))
    env = np.zeros(n)
    hold = int(dur * FS) + 1
    env[:hold] = _edge(hold, 0.02 * FS, 0.03 * FS)
    y = _finish(tb.run((u + hiss) * env)["p_out"], sr, tone=3600.0, peak=0.5 * (0.35 + 0.65 * min(float(vel), 1.2)))
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


FLEXATONE_K = 3.0           # same law as the talking drum, the thumb bends the plate
FLEXATONE_PLATE = (1.0, 2.5, 4.0, 5.0)     # simply supported square plate, (m^2 + n^2) / 2 (friction._plate_modes)


@instrument("flexatone", "Flexatone: shaken frame, two wooden balls on springs strike a thin steel plate in turn; the thumb bends the plate and the pitch slides.",
            family="percussion", span=("C4", "C6"), rate=(11.0, "strikes per second"), squeeze=(0.0, "0..1 thumb on the plate"),
            curve=("hold", "hold | rise | fall | wave: the thumb's gesture over the note"), seed=(0, "timing of the shake"))
def flexatone(freq, dur, sr=DEFAULT_SR, vel=1.0, rate=11.0, squeeze=0.0, curve="hold", seed=0):
    # Named only in the sources (a-composers-guide l.8331; Annexos l.13000 "shaken and indirectly struck"). all levels, times
    n = _n(dur, sr, 0.3)
    f = float(freq) * tension_ratio(_gesture(n, sr, dur, squeeze, curve), FLEXATONE_K)
    ph = 2 * np.pi * np.cumsum(f) / sr
    rng = np.random.default_rng(int(seed))
    hits = np.zeros(n)
    step = 1.0 / max(float(rate), 0.5)
    for i in range(int(dur / step) + 1):
        k = int((i * step + rng.uniform(-0.08, 0.08) * step + step * 0.1) * sr)
        if 0 <= k < samples(dur, sr):
            hits[k] += (1.0 if i % 2 == 0 else 0.75)
    from scipy.signal import lfilter
    y = np.zeros(n)
    for r, a, tau in zip(FLEXATONE_PLATE, (1.0, 0.3, 0.14, 0.07), (0.09, 0.05, 0.03, 0.02)):
        if r * freq < 0.42 * sr:
            # One phase per mode, fed by every strike: a mode already ringing keeps its phase. ponytail: real strikes add at their own phase (a comb a few cents wide); add per-strike phases if it sounds too clean.
            env = lfilter([1.0], [1.0, -np.exp(-1.0 / (tau * sr))], hits)
            env = F.lowpass(env, 900.0, sr)             # the ball's contact, about 1 ms
            y += a * env * np.sin(r * ph)
    y = F.lowpass(y, min(4000.0, 0.42 * sr), sr, 0.7) * _edge(n, 0.001 * sr, 0.05 * sr)
    return y * (0.35 + 0.65 * vel) * _LEVEL["flexatone"]


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
