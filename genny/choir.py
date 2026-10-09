# Portions ported from procedural/choir.py and procedural/voice.py (procedural/Klang code
# (c) 2025 Chris Nash, Klang Open License 1.0; see THIRD_PARTY_KLANG_LICENSE.txt): the per-singer
# kernel layout (LF wavetable source -> nasal pole-zero -> formant cascade, parallel frication),
# the keyframe-track articulation scheme and the singer roster idea. Adapted for genny: any sample
# rate, published per-voice-type formant tables, lyrics via genny.speech, styles, layer types.
"""Sung voice and choir: individual singers, each one glottal source and one moving vocal tract
rendered continuously across a phrase, summed into sections. Not a pad: nothing here is a chorus
effect on one voice.

Model and sources (research notes are in ``synthgen/out/research``)
-------------------------------------------------------------------
* **Source** - Liljencrants-Fant glottal flow derivative in Fant's one-parameter Rd form
  (Fant, Liljencrants & Lin 1985; Fant 1995; coefficients = ``genny.physical.voice._lf_coefs``,
  the Pink Trombone glottis, research 08 §5.4). Rd 0.5 pressed .. 1 modal .. 2.7 breathy. One
  wavetable per Rd, band-limited per pitch (mip-maps: no aliasing at any sample rate), every table
  normalised on its first harmonic so that effort raises the upper partials rather than the
  fundamental (Sundberg 1987 ch.4, as in procedural/choir.py). The flow derivative already holds
  the lip radiation (JOS: glottis -12 dB/oct, lips +6 dB/oct; research 10 §8.1, g13 §12.3).
* **Aspiration** - noise through the same tract (Cook p.67-69: whispered and breathy sound keeps
  the formant peaks, the "fuzz" between harmonics; research 04 §9.1), gated by the glottal flow of
  the current period (Klatt & Klatt 1990). Turbulence falls 6 dB/oct above 3 kHz (Stevens 1998).
* **Jitter, shimmer, flutter, drift, vibrato** - per-period period/amplitude scatter (0.3-0.6 %,
  3-5 %: normal-voice ranges, Baken & Orlikoff 2000); Klatt's flutter (12.7 + 7.1 + 4.7 Hz, Klatt &
  Klatt 1990); slow intonation drift (research 12 §8 default: a few cents); vibrato delayed and
  faded in (research 12 §8; rate 5-6.5 Hz, Prame 1994; extent by style, Prame 1997 solo +-34..123
  cents; STK SingWave/Modulate is the simpler ancestor, research 07 §8).
* **Tract** - cascade of two-pole resonators with unity DC gain (Klatt 1980; Cook Fig 8.4; JOS: the
  tract is all-pole in series, a parallel bank equals it only with partial-fraction numerators,
  research g14 item 4 - so the cascade is used and the levels of the formants follow from their
  frequencies and bandwidths). F1-F5 and B1-B5 per vowel and voice type from the Csound-manual
  table (Faust ``physmodels.lib formantValues`` = ``genny.physical.voice.formant_table``; research
  08, 11 C8): soprano, alto, countertenor, tenor, bass. Baritone, mezzo and treble are blends /
  scalings of those rows (marked UNSOURCED). Above F5 the neutral tube's next resonances
  (2k-1)c/4L (Fant 1960) are kept in the cascade with wide bandwidths, instead of Klatt's fixed
  higher-pole correction. (procedural/choir.py adds a +12..28 dB correction FIR above 5 kHz on top;
  measured here it puts 6 % of an alto's energy above 12 kHz, so it is not ported.)
* **Spectral tilt** - Klatt's TL (Klatt 1980; Klatt & Klatt 1990: dB down at 3 kHz, 0-24): a one-
  pole low-pass on the voiced source that opens with effort, so pp -> ff changes the balance of the
  upper partials (Sundberg 1987: the singer's formant gains about 1.5 dB per dB of level).
* **Wall loss** - a yielding tract wall broadens the low formants (Bilbao NSS §9.2 eq. 9.18-9.19,
  research g02 §16: "without it the sound rings unnaturally as components approach a peak"): B1
  gets a term that grows as F1 falls.
* **Formant tuning** - F1 follows f0 once f0 passes it, F2 is bent with pitch (Faust
  ``autobendFreq`` from Olsen's chant-lib = ``genny.physical.voice._autobend``; Sundberg 1987;
  Joliveau, Smith & Wolfe 2004).
* **Singer's formant** - the published male rows already cluster F3-F5 at 2.4-3.3 kHz (Sundberg
  1974: 2.5 kHz bass, 2.7 baritone, 2.9 tenor). ``ring`` moves F4/F5 between the neutral-tube
  positions (0, untrained) and the table (1, trained); choirs use less than soloists (Rossing,
  Sundberg & Ternstrom 1986).
* **Nasals / hum** - Klatt's nasal pole-zero pair (pole 270 Hz; zero per place from
  ``genny.speech.PH[...]['fnz']``) mixed in by a nasality weight; closed-mouth hum uses the STK
  ``mmm`` / ``nnn`` formant rows (research 07 §8).
* **Consonants and text** - phoneme inventory, loci, coarticulation factors, frication and burst
  spectra and levels are ``genny.speech``'s (Klatt-style; Peterson & Barney vowels). Formants glide
  between targets (locus equations), they are never cross-faded. Text -> phonemes is
  ``genny.speech`` for English and Spanish plus a small ecclesiastical-Latin rule set here.
  Consonants are placed before the beat so the vowel lands on it (choral diction).
* **Ensemble** - every singer has its own tract length (Fitch & Giedd 1999), formant
  idiosyncrasy, pitch offset (choir scatter about 10 cents, Ternstrom & Sundberg 1988), vibrato
  rate / depth / phase, timing lag, breathiness, level, pan and random stream; uncorrelated voices
  add as sqrt(N) (Farnell, research 01 §7.5).
* Moving-tract note: the resonators are direct-form two-poles updated every ~0.7 ms on slowly
  gliding formants; research g13 §13.3 (normalised Kelly-Lochbaum junctions) is the remedy for
  area-function tracts, which this formant model does not use (``genny.tubes.vowel_tube`` /
  ``genny.physical.voice.pink_trombone`` are the area-function voices).

Numbers without a source are marked ``# UNSOURCED``.

Public API: ``sing_phrase`` (notes + lyrics -> stereo), ``render_singers`` (the individual
streams), ``voice_formants``, ``tuned_formants``, ``parse_lyrics``, ``parse_dynamics``; catalog
instruments ``voice vocal_choir hum throat_singing falsetto boys_choir chant``; layer types
``sing`` and ``satb``.
"""
from __future__ import annotations

import math
import re
import zlib
from functools import lru_cache

import numba
import numpy as np
from scipy.signal import butter, filtfilt, lfilter

from . import filters as FL
from . import speech as SP
from .analysis import loudness
from .core import DEFAULT_SR
from .instruments import instrument
from .notes import midi_to_freq, parse_pitch_list, parse_sequence
from .physical.voice import STK_PHONEMES, _autobend, formant_table, lf_waveform
from .spec import SpecError, layer_type

# ==============================================================================================
# Tables
# ==============================================================================================

# voice types. tab: blend of Csound/Faust table rows; tscale: factor on the row's formants;
# sc: tract factor for genny.speech's adult-male consonant/glide formants (speech.VOICES uses 1.16
# female, 1.28 child); lo/hi: comfortable range (MIDI); brk: top of the full (chest/mixed) voice, above
# it a man goes into falsetto; sf: centre of the
# singer's formant (Sundberg 1974).
VOICES = {
    "bass": dict(tab={"bass": 1.0}, tscale=1.0, sc=0.94, lo=40, hi=64, brk=64, male=True, sf=2500.0),
    # UNSOURCED: baritone = mean of the bass and tenor rows (no baritone row in the table)
    "baritone": dict(tab={"bass": 0.5, "tenor": 0.5}, tscale=1.0, sc=0.97, lo=43, hi=67, brk=67, male=True, sf=2700.0),
    "tenor": dict(tab={"tenor": 1.0}, tscale=1.0, sc=1.0, lo=48, hi=72, brk=72, male=True, sf=2900.0),
    "countertenor": dict(tab={"countertenor": 1.0}, tscale=1.0, sc=1.04, lo=55, hi=77, brk=None, male=True, sf=3000.0),
    "alto": dict(tab={"alto": 1.0}, tscale=1.0, sc=1.12, lo=53, hi=77, brk=None, male=False, sf=3100.0),
    # UNSOURCED: mezzo = mean of the alto and soprano rows
    "mezzo": dict(tab={"alto": 0.5, "soprano": 0.5}, tscale=1.0, sc=1.14, lo=57, hi=81, brk=None, male=False, sf=3100.0),
    "soprano": dict(tab={"soprano": 1.0}, tscale=1.0, sc=1.16, lo=60, hi=84, brk=None, male=False, sf=3200.0),
    # UNSOURCED: treble = soprano row x 1.10 (Peterson & Barney 1952 children's formants sit 10-20 %
    # above women's; speech.VOICES child/female shift 1.28/1.16 = 1.10)
    "treble": dict(tab={"soprano": 1.0}, tscale=1.10, sc=1.28, lo=60, hi=81, brk=None, male=False, sf=3400.0),
}
_ALIAS = {"child": "treble", "boy": "treble", "boys": "treble", "girl": "treble", "contralto": "alto",
          "counter": "countertenor", "mezzo-soprano": "mezzo", "sopranos": "soprano", "altos": "alto",
          "tenors": "tenor", "basses": "bass", "baritones": "baritone", "mezzos": "mezzo", "trebles": "treble"}
# breaks / ranges above: UNSOURCED exact notes (standard choral ranges and passaggio regions)

# sections: list of voice types the singers cycle through; "oct" = men an octave under the women
SECTIONS = {
    "sopranos": ("soprano",), "altos": ("alto",), "tenors": ("tenor",), "basses": ("bass",),
    "baritones": ("baritone",), "mezzos": ("mezzo",), "boys": ("treble",), "children": ("treble",),
    "countertenors": ("countertenor",),
    "men": ("tenor", "bass", "baritone"), "women": ("soprano", "alto", "mezzo"),
    "mixed": ("soprano", "tenor", "alto", "bass"), "satb": ("soprano", "tenor", "alto", "bass"),
}

# styles = presets of the model's parameters. vib: vibrato extent of a soloist (+- cents, peak);
# choir: factor on it for section singers; rate Hz; delay s before the vibrato fades in; ring:
# singer's-formant amount; rd: voice-quality offset (negative = pressed/bright); breath 0..1;
# scatter: pitch scatter between singers (cents sd); lag: onset spread (s sd); scoop: onset scoop
# from below (cents); attack s; f1: factor on F1 (open "belt"/shout raises it).
# UNSOURCED: every number in this table is a style choice inside the sourced ranges of the module
# docstring (vibrato 5-6.5 Hz and +-30..100 cents; scatter <= 14 cents; lag 10-40 ms).
STYLES = {
    "classical": dict(vib=60.0, choir=0.5, rate=5.7, delay=0.28, ring=0.9, rd=0.0, breath=0.10, scatter=7.0,
                      lag=0.018, scoop=14.0, attack=0.07, f1=1.0, drift=5.0),
    "gregorian": dict(vib=7.0, choir=0.7, rate=5.0, delay=0.6, ring=0.45, rd=0.15, breath=0.12, scatter=6.0,
                      lag=0.022, scoop=8.0, attack=0.09, f1=1.0, drift=4.0),
    "gospel": dict(vib=75.0, choir=0.65, rate=5.4, delay=0.35, ring=0.6, rd=-0.25, breath=0.10, scatter=9.0,
                   lag=0.020, scoop=30.0, attack=0.05, f1=1.05, drift=6.0),
    "pop": dict(vib=32.0, choir=0.5, rate=5.2, delay=0.55, ring=0.15, rd=0.30, breath=0.35, scatter=6.0,
                lag=0.015, scoop=18.0, attack=0.06, f1=1.0, drift=4.0),
    "children": dict(vib=12.0, choir=0.6, rate=5.6, delay=0.5, ring=0.2, rd=0.20, breath=0.22, scatter=11.0,
                     lag=0.030, scoop=14.0, attack=0.07, f1=1.0, drift=7.0),
    "epic": dict(vib=45.0, choir=0.6, rate=5.9, delay=0.25, ring=1.15, rd=-0.35, breath=0.05, scatter=8.0,
                 lag=0.014, scoop=10.0, attack=0.045, f1=1.04, drift=5.0),
    "throat": dict(vib=0.0, choir=1.0, rate=5.0, delay=1.0, ring=0.5, rd=-0.55, breath=0.02, scatter=4.0,
                   lag=0.030, scoop=6.0, attack=0.12, f1=1.0, drift=3.0),
    "bulgarian": dict(vib=0.0, choir=1.0, rate=5.5, delay=1.0, ring=0.55, rd=-0.50, breath=0.03, scatter=4.0,
                      lag=0.010, scoop=5.0, attack=0.035, f1=1.10, drift=3.0),
}
STYLES["cathedral"] = STYLES["classical"]
STYLES["chant"] = STYLES["gregorian"]
STYLES["trailer"] = STYLES["epic"]
STYLES["overtone"] = STYLES["throat"]

C_SOUND = 350.0             # m/s, warm humid tract air (procedural/choir.py)
TUBE_F1 = C_SOUND / (4 * 0.175)   # 500 Hz: first resonance of a 17.5 cm neutral tube (Fant 1960)
NF = 20                     # resonators in the cascade: F1-F5 + the tube poles up to ~0.45 sr
                            # (procedural/choir.py N_POLES: five alone lose 60-100 dB above 5 kHz)
FNP = SP.FNP                # nasal pole 270 Hz (Klatt 1980, genny.speech)

# aspiration levels (noise into the cascade re the first harmonic, H1 = 1, unit-RMS noise).
# ASP_VOICED is set by measurement on this kernel: the harmonics-to-noise ratio in 1-4 kHz of a
# solo tenor / bass note is 18 dB at mf, 21-24 dB at ff, 6-8 dB at pp (CC0 sung notes: 20-34 dB,
# procedural/choir.py) and about 0 dB at breath = 1. (procedural/choir.py's 0.33 measures -3 dB at
# mf here: a hoarse whisper.) The consonant levels are procedural/choir.py's (broadband level re
# the vowel).
ASP_VOICED = 0.015
CASP_VOT = 0.35             # voice-onset aspiration about vowel -10 dB
CASP_H = 0.45               # /h/ about vowel -8 dB
CASP_LEAD = 0.08            # breathy onset of a wordless entry (UNSOURCED: lowered from 0.14)
CASP_EXHALE = 0.03          # exhale after the release (UNSOURCED: lowered from 0.08, which measured -18 dB re the vowel)
CASP_INHALE = 0.20          # UNSOURCED: audible breath intake before a phrase (`breaths`)
ASP_TILT_HZ = 3000.0        # Stevens 1998: turbulence source falls above a few kHz (UNSOURCED: two poles;
                            # with one, a breathy "pop" section measured 9 % of its energy above 5 kHz)
VOWEL_RMS = 2.9             # kernel output RMS of an mf vowel at level 1 (measured: tenor G3 "ah")
CONS_DB = 3.0               # UNSOURCED: sung consonants are over-articulated re speech levels
_CAL = 0.036                # UNSOURCED: output scale so an ff section peaks under full scale

# sung vowels as blends of the table's five cardinal vowels. Choral diction modifies English
# vowels toward the cardinals; weights chosen so F1/F2 match genny.speech.PH (Peterson & Barney).
# UNSOURCED: the blend weights. Diphthongs: (first target, second target).
_VT = {
    "A": ({"a": 1.0},), "E": ({"e": 1.0},), "I": ({"i": 1.0},), "O": ({"o": 1.0},), "U": ({"u": 1.0},),
    "IY": ({"i": 1.0},), "IH": ({"i": 0.6, "e": 0.4},), "EH": ({"e": 0.7, "a": 0.3},),
    "AE": ({"a": 0.55, "e": 0.45},), "AA": ({"a": 1.0},), "AO": ({"o": 0.6, "a": 0.4},),
    "UH": ({"u": 0.6, "o": 0.4},), "UW": ({"u": 1.0},), "AH": ({"a": 0.8, "e": 0.2},),
    "AX": ({"a": 0.3, "e": 0.4, "o": 0.3},), "IX": ({"i": 0.5, "e": 0.5},),
    "ER": ({"a": 0.3, "e": 0.4, "o": 0.3},), "AXR": ({"a": 0.3, "e": 0.4, "o": 0.3},),
    "EY": ({"e": 1.0}, {"e": 0.4, "i": 0.6}), "AY": ({"a": 1.0}, {"i": 0.7, "e": 0.3}),
    "OY": ({"o": 1.0}, {"i": 0.7, "e": 0.3}), "AW": ({"a": 1.0}, {"u": 0.6, "o": 0.4}),
    "OW": ({"o": 1.0}, {"u": 0.7, "o": 0.3}),
}
# wordless syllables and hums (whole-token match in any language)
_SPECIAL = {"a": "A", "ah": "A", "aa": "A", "aah": "A", "ahh": "A", "e": "E", "eh": "E", "ee": "I", "eee": "I",
            "i": "I", "o": "O", "oh": "O", "ohh": "O", "oo": "U", "ooh": "U", "u": "U", "uh": "AX"}
_HUMS = {"mm": "M", "m": "M", "mmm": "M", "hm": "M", "hmm": "M", "hum": "M", "nn": "N", "n": "N", "ng": "NG"}
# closed-mouth formants: STK Phonemes.cpp rows mmm / nnn (genny.physical.voice.STK_PHONEMES) plus
# two tube poles; bandwidths of a damped nasal murmur from procedural/choir.py HUM.
_HUM_B = (90.0, 280.0, 260.0, 320.0, 380.0)

_KIND = {SP.STOP: "stop", SP.FRIC: "fric", SP.ASP: "asp", SP.NAS: "nasal", SP.LIQ: "liquid",
         SP.GLI: "glide", SP.TAP: "tap", SP.TRILL: "trill"}
_AFFR = {"CH": ("T", "SH"), "JH": ("D", "ZH")}
# stop internals (s): burst length as genny.speech (_durations), short-lag VOT for sung diction
# (Lisker & Abramson 1964 via procedural/choir.py)
_VOT = {"P": 0.022, "T": 0.028, "K": 0.034, "B": 0.006, "D": 0.008, "G": 0.012}
_BURST = {"lab": 0.006, "alv": 0.009, "vel": 0.015}
# formant transition times into / out of a consonant (Liberman et al. 1956: stops fast, glides slow)
_TRANS = {"stop": 0.045, "fric": 0.035, "asp": 0.0, "nasal": 0.040, "liquid": 0.060, "tap": 0.030,
          "trill": 0.030, "glide": 0.090}
# frication spectrum of the velar fricative (Spanish jota) - not in speech.FRIC_SPEC (it uses the
# cascade there); values from procedural/choir.py FRIC_SPEC["vel"]
_FRIC_X = ([(1700.0, 600.0, 0.9), (3000.0, 1200.0, 0.4), (5000.0, 2500.0, 0.15)], 0.08)
_DYN = {"ppp": 0.10, "pp": 0.20, "p": 0.33, "mp": 0.45, "mf": 0.58, "f": 0.72, "ff": 0.86, "fff": 1.0}
# UNSOURCED: effort value of each dynamic mark (0..1), and LEVEL_DB dB of gain per unit effort. The
# source itself gains about 16 dB from pp to ff (pressed pulse, open tilt), so pp -> ff measures
# about 22 dB in all, a trained singer's range.
LEVEL_DB = 6.0
# UNSOURCED: Rd of a trained voice at mf. Fant 1995 gives about 1.0 for modal speech; singers close
# more abruptly, and at 1.0 the harmonics above 1 kHz sink into the aspiration (measured HNR 10 dB).
RD_MF = 0.8


def _voice_name(v: str) -> str:
    v = str(v).lower().strip()
    v = _ALIAS.get(v, v)
    if v not in VOICES:
        raise ValueError(f"unknown voice type {v!r}; use one of {', '.join(VOICES)}")
    return v


def _auto_voice(midi: float) -> str:
    # UNSOURCED: split points between voice types (as genny.physical.voice._voice_type_for)
    return "bass" if midi < 50 else "tenor" if midi < 60 else "alto" if midi < 69 else "soprano"


@lru_cache(maxsize=None)
def _cardinals(voice: str) -> dict:
    """vowel -> (F[5] Hz, B[5] Hz, G[5] linear) for a voice type, from the Csound/Faust table."""
    V = VOICES[voice]
    out = {}
    for vw in "aeiou":
        F = np.zeros(5)
        B = np.zeros(5)
        G = np.zeros(5)
        for row, w in V["tab"].items():
            t = formant_table(row, vw)
            F += w * t["freqs"]
            B += w * t["bws"]
            G += w * t["gains"]
        out[vw] = (F * V["tscale"], B, G)
    return out


def voice_formants(voice: str, vowel: str = "a") -> dict:
    """Published five-formant row used for `voice` singing `vowel` (a e i o u):
    {"freqs": Hz, "bws": Hz, "gains_db": dB re F1}. The amplitudes are what the cascade should
    (and does, see tests) roughly produce; they are not applied as gains."""
    F, B, G = _cardinals(_voice_name(voice))[vowel]
    return {"freqs": F.copy(), "bws": B.copy(), "gains_db": 20 * np.log10(G)}


def tuned_formants(F, f0, voice: str = "soprano") -> np.ndarray:
    """Formant tuning (Faust `autobendFreq`): F1 raised to f0 when f0 > F1, F2 bent with pitch.
    F: (..., >=2) Hz; f0 scalar or per-row."""
    F = np.atleast_2d(np.asarray(F, float)).copy()
    f0 = np.broadcast_to(np.asarray(f0, float), (F.shape[0],))
    V = VOICES[_voice_name(voice)]
    row = voice if voice in ("alto", "bass", "countertenor", "soprano", "tenor") else ("tenor" if V["male"] else "soprano")
    return _autobend(F, f0, row)


@lru_cache(maxsize=None)
def _upper_poles(voice: str):
    """(neutral-tube resonances of the voice, its poles above F5, mean F5). The poles above F5 are
    the tube's 6th and higher resonances whatever the vowel or `ring`: were they re-seated when F5
    moves, a vowel change would glide fifteen resonators by 1 kHz (measured: a burst of noise)."""
    s = VOICES[voice]["sc"]
    neutral = np.array([(2 * k - 1) * TUBE_F1 * s for k in range(1, NF + 1)])
    return neutral, neutral[5:], float(np.mean([c[0][4] for c in _cardinals(voice).values()]))


def _slots(F5, B5, voice: str, ring: float):
    """The NF cascade poles (and bandwidths) of a five-formant target. `ring` 1 = the published
    row (F3-F5 of the male rows are the singer's-formant cluster, leaving a gap above it as in a
    real trained voice), 0 = F4/F5 at the neutral-tube positions (speech-like), > 1 pulls F3-F5
    further onto the cluster centre."""
    V = VOICES[voice]
    neutral, hi, _ = _upper_poles(voice)
    r1 = np.concatenate([F5, hi])
    n4 = max(neutral[3], F5[2] * 1.15)
    r0 = np.concatenate([F5[:3], [n4, max(neutral[4], n4 * 1.15)], hi])
    wide = lambda f: np.maximum(200.0, 0.08 * f)      # UNSOURCED: bandwidth of the upper tube poles
    b1 = np.concatenate([B5, wide(hi)])
    b0 = np.concatenate([B5[:3], wide(r0[3:])])
    r = float(np.clip(ring, 0.0, 1.6))
    w = min(r, 1.0)
    F = r0 + w * (r1 - r0)
    B = b0 + w * (b1 - b0)
    if r > 1.0:
        for k, off in zip((2, 3, 4), (-300.0, 0.0, 350.0)):     # cluster spacing: procedural/choir.py
            F[k] += (r - 1.0) * (V["sf"] + off - F[k])
    return F, B


def _vowel_FB(voice: str, mix: dict, ring: float, f1: float = 1.0) -> np.ndarray:
    card = _cardinals(voice)
    F5 = sum(w * card[v][0] for v, w in mix.items())
    B5 = sum(w * card[v][1] for v, w in mix.items())
    F5 = F5.copy()
    F5[0] *= f1
    F, B = _slots(F5, B5, voice, ring)
    return np.concatenate([F, B])


def _hum_FB(voice: str, which: str) -> np.ndarray:
    s = VOICES[voice]["sc"]
    row = STK_PHONEMES["mmm" if which == "M" else "nnn"][1]
    _, hi, f5 = _upper_poles(voice)
    f = np.concatenate([np.array([r[0] for r in row]) * s, [max(f5, 3600.0 * s)], hi])
    b = np.concatenate([_HUM_B, np.maximum(400.0, 0.08 * hi)])
    return np.concatenate([f, b])


def _cons_FB(ph: str, vFB: np.ndarray, sc: float) -> np.ndarray:
    """Formant targets during a consonant from the adjacent vowel: genny.speech's loci and
    coarticulation factors (speech._targets), scaled to the singer's tract."""
    info = SP.PH[ph]
    kind = _KIND[info["k"]]
    FB = vFB.copy()
    if kind == "asp":
        return FB
    if kind in ("liquid", "glide"):
        FB[:3] = np.array(info["f"], float) * sc
        return FB
    place = info["place"]
    if place == "vel":
        f2 = float(np.clip(vFB[1] + 150.0 * sc, 1350.0 * sc, 2400.0 * sc))
        f = [250.0 * sc, f2, max(f2 + 450.0 * sc, 2200.0 * sc)]
    else:
        loc = SP.LOCUS[place]
        kk = SP.COART.get(info["k"], 0.4)
        f = [loc[0] * sc, loc[1] * sc + kk * (vFB[1] - loc[1] * sc), loc[2] * sc + kk * 0.6 * (vFB[2] - loc[2] * sc)]
    if kind == "nasal":
        f[0] = 280.0 * sc
        FB[NF] = 110.0
    if kind in ("tap", "trill"):
        f[0] = 300.0 * sc
        f[1] = 1700.0 * sc + 0.5 * (vFB[1] - 1700.0 * sc)
    FB[0] = min(vFB[0], f[0])
    FB[1] = f[1]
    FB[2] = max(f[2], f[1] + 200.0)
    return FB


def _fric_vec(spec, level_db: float, sr: int) -> np.ndarray:
    """(f, bw, g) x 3 + flat as a 10-vector whose branch output for unit-RMS noise has the RMS of
    a vowel + level_db (band power of a unit-peak band-pass in white noise = pi bw / sr)."""
    bands, flat = spec
    top = 0.45 * sr
    v = np.zeros(10)
    p = flat * flat
    for k, (f, bw, g) in enumerate(bands[:3]):
        f = min(float(f), top)
        v[3 * k:3 * k + 3] = (f, bw, g)
        p += g * g * math.pi * bw / sr
    lin = VOWEL_RMS * 10 ** ((level_db + CONS_DB) / 20.0) / math.sqrt(p + 1e-12)
    v[2] *= lin
    v[5] *= lin
    v[8] *= lin
    v[9] = flat * lin
    return v


# ==============================================================================================
# Text: lyrics -> syllables (onset consonants, vowel nucleus, coda)
# ==============================================================================================

def _g2p_la(word: str) -> list[str]:
    """Ecclesiastical (Italianate) Latin spelling -> phonemes: pure vowels a e i o u, ae/oe = e,
    c/g soft before e i (tch / dj), gn = ny, ti + vowel = tsi, h silent, qu = kw, x = ks.
    Liber Usualis pronunciation rules; r is a tap."""
    w = re.sub(r"[^a-zæœ]", "", word.lower()).replace("æ", "ae").replace("œ", "oe")
    out: list[str] = []
    n = len(w)
    V = "aeiouy"
    i = 0

    def soft(j):                       # the letter at j starts a front vowel (e, i, y, ae, oe)
        return j < n and (w[j] in "eiy" or (w[j] in "ao" and j + 1 < n and w[j + 1] == "e"))

    while i < n:
        c = w[i]
        nx = w[i + 1] if i + 1 < n else ""
        nx2 = w[i + 2] if i + 2 < n else ""
        if c in "ao" and nx == "e":
            out.append("E")
            i += 2
            continue
        if c == "i" and nx and nx in V and (i == 0 or w[i - 1] in V) and nx != "i":
            out.append("Y")            # consonantal i: Iesu, eius, alleluia
        elif c in "aeiou":
            out.append(c.upper())
        elif c == "y":
            out.append("I")
        elif c == "c":
            if nx == "h":
                out.append("K")
                i += 1
            elif soft(i + 1):
                out += ["T", "SH"]
            elif nx == "c" and soft(i + 2):
                out += ["T", "SH"]
                i += 1
            else:
                out.append("K")
        elif c == "g":
            if nx == "n":
                out += ["N", "Y"]
                i += 1
            elif soft(i + 1):
                out += ["D", "ZH"]
            else:
                out.append("G")
        elif c == "h":
            pass
        elif c == "j":
            out.append("Y")
        elif c == "q":
            out.append("K")
            if nx == "u":
                out.append("W")
                i += 1
        elif c == "p" and nx == "h":
            out.append("F")
            i += 1
        elif c == "t":
            if nx == "h":
                out.append("T")
                i += 1
            elif nx == "i" and nx2 and nx2 in V and (i == 0 or w[i - 1] not in "stx"):
                out += ["T", "S"]
            else:
                out.append("T")
        elif c == "x":
            if nx == "c" and soft(i + 2):                # excelsis: ek-shel-sis
                out += ["K", "SH"]
                i += 1
            else:
                out += ["K", "S"]
        elif c == "z":
            out += ["D", "Z"]
        elif c == "s":
            if nx == "c" and soft(i + 2):
                out.append("SH")
                i += 1
            else:
                out.append("S")
        elif c == "r":
            out.append("RR" if (i == 0 or nx == "r") else "DX")
            if nx == "r":
                i += 1
        elif c == "v":
            out.append("V")
        elif c == "w":
            out.append("W")
        elif c in "bdfklmnp":
            out.append(c.upper())
            if nx == c:
                i += 1
        i += 1
    return out


def _phones(word: str, lang: str) -> list[str]:
    if lang == "la":
        ph = _g2p_la(word)
    elif lang == "es":
        ph = [p for p, _ in SP._g2p_es(word)]
    elif lang == "en":
        low = word.lower()
        ph = []
        for wd in (SP._word_en(word, False) if (low in SP.DICT_EN or len(low) > 1) else [{"ph": SP._lts_en(low)}]):
            ph += [p for p, _ in wd.get("ph", [])]
    else:
        raise ValueError(f"lang must be en, es or la, got {lang!r}")
    out = []
    for p in ph:
        out += list(_AFFR.get(p, (p,)))
    return out


def _is_v(p: str) -> bool:
    return SP.PH[p]["k"] in SP.VOWELISH


def _one_syllable(ph: list[str]) -> dict:
    vi = [k for k, p in enumerate(ph) if _is_v(p)]
    if not vi:
        nas = [p for p in ph if p in ("M", "N", "NG")]
        if nas and len(nas) == len(ph):
            return {"onset": [], "nucl": [], "coda": [], "hum": nas[0]}
        return {"onset": list(ph), "nucl": ["AX"], "coda": [], "hum": None}
    a, b = vi[0], vi[-1]
    nucl = [p for p in ph[a:b + 1] if _is_v(p)]
    return {"onset": ph[:a], "nucl": [nucl[0]] + ([nucl[-1]] if len(nucl) > 1 else []), "coda": ph[b + 1:], "hum": None}


def _syllabify(ph: list[str]) -> list[dict]:
    """Split a word's phonemes at its vowels with maximal onsets: one consonant between vowels
    opens the next syllable; of a cluster the last one does (obstruent + liquid/glide stay
    together), the rest close the previous syllable."""
    vi = [k for k, p in enumerate(ph) if _is_v(p)]
    if len(vi) <= 1:
        return [_one_syllable(ph)]
    cuts = []
    for v0, v1 in zip(vi, vi[1:]):
        gap = v1 - v0 - 1
        take = min(gap, 1)
        if gap >= 2 and _KIND.get(SP.PH[ph[v1 - 1]]["k"]) in ("liquid", "glide", "tap", "trill") \
                and _KIND.get(SP.PH[ph[v1 - 2]]["k"]) in ("stop", "fric"):
            take = 2
        cuts.append(v1 - take)
    out, start = [], 0
    for c in cuts + [len(ph)]:
        out.append(_one_syllable(ph[start:c]))
        start = c
    return out


def _special(tok: str):
    low = tok.lower()
    if low in _HUMS:
        return {"onset": [], "nucl": [], "coda": [], "hum": _HUMS[low]}
    if low in _SPECIAL:
        return {"onset": [], "nucl": [_SPECIAL[low]], "coda": [], "hum": None}
    return None


def parse_lyrics(lyrics, lang: str = "la") -> list:
    """Lyrics -> one entry per note: a syllable dict {"onset", "nucl", "coda", "hum"} or "_"
    (melisma: keep the previous vowel).

    Words are split into syllables at their vowels; hyphens ("A-ve Ma-ri-a") mark the split
    explicitly when it differs. "ah" "oh" "oo" "ee" "eh" are plain vowels, "mm" "nn" "ng" closed-
    mouth hums, "/T AA/" is one syllable of raw ARPAbet (genny.speech phonemes), "_" or "~"
    holds the previous vowel."""
    if lyrics is None:
        return []
    out: list = []
    for tok in (re.findall(r"/[^/]*/|\S+", lyrics) if isinstance(lyrics, str) else list(lyrics)):
        if tok in ("_", "~"):
            out.append("_")
            continue
        if tok.startswith("/"):
            ph = []
            for w in SP._parse_raw(tok.strip("/")):
                for p, _ in w.get("ph", []):
                    ph += list(_AFFR.get(p, (p,)))
            out.append(_one_syllable(ph))
            continue
        parts = [re.sub(r"[^\w'æœ]", "", p) for p in tok.split("-")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if len(parts) == 1 and _special(parts[0]):
            out.append(_special(parts[0]))
            continue
        syl = _syllabify(_phones("".join(parts), lang))
        if len(parts) > 1 and len(syl) != len(parts):
            syl = [_special(p) or _one_syllable(_phones(p, lang) if lang != "en" else
                                                [q for x in SP._lts_en(p) for q in _AFFR.get(x[0], (x[0],))])
                   for p in parts]
        out += syl
    return out


def parse_dynamics(text, total: float):
    """"mp<f>p" -> effort curve over `total` seconds as (times, values): the marks (ppp pp p mp mf
    f ff fff, or a number 0..1) are spaced evenly; "<" / ">" between two marks is a hairpin
    (ramp), a space or comma a sudden change at the later mark."""
    if text is None or text == "":
        text = "mf"
    if isinstance(text, (int, float)):
        return np.array([0.0, total]), np.array([float(text)] * 2)
    toks = re.findall(r"ppp|pp|p|mp|mf|fff|ff|f|[<>]|\d*\.?\d+", str(text).lower())
    marks, hair = [], []
    pend = None
    for t in toks:
        if t in "<>":
            pend = t
            continue
        marks.append(_DYN[t] if t in _DYN else float(t))
        hair.append(pend)
        pend = None
    if not marks:
        marks, hair = [_DYN["mf"]], [None]
    if pend:                                            # trailing hairpin: one mark louder / softer
        marks.append(float(np.clip(marks[-1] + (0.14 if pend == "<" else -0.14), 0.05, 1.0)))
        hair.append(pend)
    if len(marks) == 1:
        return np.array([0.0, total]), np.array([marks[0]] * 2)
    pos = np.linspace(0.0, total, len(marks))
    ts, vs = [0.0], [marks[0]]
    for j in range(1, len(marks)):
        if hair[j] is None:                             # subito
            ts.append(pos[j] - 1e-4)
            vs.append(marks[j - 1])
        ts.append(pos[j])
        vs.append(marks[j])
    return np.array(ts), np.array(vs)


# ==============================================================================================
# Glottal source tables (LF, Rd form), mip-mapped - procedural/choir.py _tables
# ==============================================================================================

RD_GRID = np.round(np.arange(0.5, 2.71, 0.1), 2)
TAB_N = 2048
LEVEL_H = np.array([512, 362, 256, 181, 128, 90, 64, 45, 32, 23, 16, 11, 8, 5, 3, 2, 1], dtype=np.float64)


@lru_cache(maxsize=1)
def _tables():
    tabs = np.zeros((len(RD_GRID), len(LEVEL_H), TAB_N))
    flows = np.zeros((len(RD_GRID), TAB_N))
    for r, rd in enumerate(RD_GRID):
        w = lf_waveform(float(rd), TAB_N)
        w = w - w.mean()
        S = np.fft.rfft(w)
        S *= (TAB_N / 2) / (abs(S[1]) + 1e-12)          # first harmonic amplitude 1 for every Rd
        for lv, H in enumerate(LEVEL_H):
            s2 = S.copy()
            s2[int(H) + 1:] = 0.0
            tabs[r, lv] = np.fft.irfft(s2, TAB_N)
        fl = np.cumsum(w)
        fl -= fl.min()
        flows[r] = fl / (fl.mean() + 1e-12)             # mean 1: the gate keeps the noise power
    return tabs, flows


# ==============================================================================================
# Kernel: LF source + gated aspiration -> nasal pole-zero -> formant cascade; parallel frication
# ==============================================================================================

@numba.njit(cache=True)
def _kernel(f0, amp, rdi, tl, sub, asp, fric, na, nf, tabs, flows, lvl_h, F, B, NZ, NW, FS, blk, rj, rs,
            ph0, gate_depth, sr):
    n = len(f0)
    nrd = tabs.shape[0]
    nlv = tabs.shape[1]
    L = tabs.shape[2]
    out = np.zeros(n)
    outf = np.zeros(n)
    nfm = F.shape[1]
    y1 = np.zeros(nfm)
    y2 = np.zeros(nfm)
    ca = np.ones(nfm)
    cb = np.zeros(nfm)
    cc = np.zeros(nfm)
    za = 1.0
    zb = 0.0
    zc = 0.0
    x1 = 0.0
    x2 = 0.0
    ny1 = 0.0
    ny2 = 0.0
    pb = 0.0
    pc = 0.0
    nw = 0.0
    fa = np.zeros(3)
    fb = np.zeros(3)
    fc = np.zeros(3)
    fg = np.zeros(3)
    fy1 = np.zeros(3)
    fy2 = np.zeros(3)
    fx1 = 0.0
    fx2 = 0.0
    gflat = 0.0
    glp = 0.0
    ph = ph0
    per = 0
    jf = 1.0
    sh = 1.0
    sg = 1.0
    lv = nlv - 1
    r0 = 0
    rw = 0.0
    nr = len(rj)
    two_pi = 2.0 * math.pi
    top = 0.45 * sr
    first = True
    for i in range(n):
        if i % blk == 0:
            b = i // blk
            for k in range(nfm):
                fk = F[b, k]
                if fk >= top:                          # pole past the band: identity section
                    ca[k] = 1.0
                    cb[k] = 0.0
                    cc[k] = 0.0
                    continue
                r = math.exp(-math.pi * B[b, k] / sr)
                cc[k] = -r * r
                cb[k] = 2.0 * r * math.cos(two_pi * fk / sr)
                ca[k] = 1.0 - cb[k] - cc[k]            # Klatt 1980 resonator, unity gain at DC
            # Klatt nasal pole-zero pair: zero NZ (bw 180), pole 270 Hz (bw 100), unity at DC
            r = math.exp(-math.pi * 180.0 / sr)
            z1c = -2.0 * r * math.cos(two_pi * min(NZ[b], top) / sr)
            z2c = r * r
            rp = math.exp(-math.pi * 100.0 / sr)
            pb = 2.0 * rp * math.cos(two_pi * 270.0 / sr)
            pc = -rp * rp
            kz = (1.0 - pb - pc) / (1.0 + z1c + z2c)
            za = kz
            zb = kz * z1c
            zc = kz * z2c
            nw = NW[b]
            for k in range(3):
                bw = FS[b, 3 * k + 1]
                fk = FS[b, 3 * k]
                if bw < 1.0 or fk < 1.0:
                    fa[k] = 0.0
                    fb[k] = 0.0
                    fc[k] = 0.0
                    fg[k] = 0.0
                    continue
                if fk > top:
                    fk = top
                r = math.exp(-math.pi * bw / sr)
                fb[k] = 2.0 * r * math.cos(two_pi * fk / sr)
                fc[k] = -r * r
                fa[k] = 0.5 * (1.0 - r * r)            # zeros at +-1: unit peak gain (STK BiQuad)
                fg[k] = FS[b, 3 * k + 2]
            gflat = FS[b, 9]
        f = f0[i] * jf
        ph += f / sr
        if ph >= 1.0 or first:
            if not first:
                ph -= 1.0
                per += 1
                jf = 1.0 + rj[per % nr]
                sh = 1.0 + rs[per % nr]
            first = False
            sg = 1.0 - sub[i] if (per % 2) == 1 else 1.0
            hmax = top / max(f0[i], 20.0)
            lv = 0
            while lv < nlv - 1 and lvl_h[lv] > hmax:
                lv += 1
            x = rdi[i]
            r0 = int(x)
            if r0 >= nrd - 1:
                r0 = nrd - 2
                rw = 1.0
            else:
                rw = x - r0
        pos = ph * L
        j = int(pos)
        if j >= L:
            j = L - 1
        fr = pos - j
        j2 = j + 1
        if j2 >= L:
            j2 = 0
        v0 = tabs[r0, lv, j] + fr * (tabs[r0, lv, j2] - tabs[r0, lv, j])
        v1 = tabs[r0 + 1, lv, j] + fr * (tabs[r0 + 1, lv, j2] - tabs[r0 + 1, lv, j])
        glp += tl[i] * ((1.0 - rw) * v0 + rw * v1 - glp)     # spectral tilt (Klatt TL)
        g = glp
        gate = 1.0
        if amp[i] > 1e-5:
            fl = (1.0 - rw) * flows[r0, j] + rw * flows[r0 + 1, j]
            gate = 1.0 - gate_depth + gate_depth * fl
        s = amp[i] * sh * sg * g + asp[i] * na[i] * gate
        yn = za * s + zb * x1 + zc * x2 + pb * ny1 + pc * ny2
        ny2 = ny1
        ny1 = yn
        x2 = x1
        x1 = s
        xx = (1.0 - nw) * s + nw * yn
        # highest pole first: the moving formants (F1-F5) come last, so the small steps their
        # coefficient updates inject are not amplified by the static upper poles (each has a gain
        # of 30 dB or more near Nyquist; in the other order a formant glide measured as a noise
        # burst three times the signal)
        for k in range(nfm - 1, -1, -1):
            y = ca[k] * xx + cb[k] * y1[k] + cc[k] * y2[k]
            y2[k] = y1[k]
            y1[k] = y
            xx = y
        e = fric[i] * nf[i] * gate
        acc = gflat * e
        for k in range(3):
            y = fa[k] * (e - fx2) + fb[k] * fy1[k] + fc[k] * fy2[k]
            fy2[k] = fy1[k]
            fy1[k] = y
            acc += fg[k] * y
        fx2 = fx1
        fx1 = e
        out[i] = xx
        outf[i] = acc
    return out, outf


# ==============================================================================================
# Control tracks
# ==============================================================================================

def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _lp(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """Zero-phase 2nd-order low-pass (control smoothing)."""
    if x.shape[0] < 16:
        return x
    b, a = butter(2, min(0.99, fc / (fs / 2)))
    return filtfilt(b, a, x, axis=0, padlen=min(x.shape[0] - 1, 9))


class _Tracks:
    """Keyframe collector: (time, value) pairs per track, interpolated at block times."""

    def __init__(self):
        self.k: dict[str, list] = {}

    def add(self, name, t, v):
        self.k.setdefault(name, []).append((float(t), v))

    def curve(self, name, tb, default=0.0):
        pts = self.k.get(name)
        if not pts:
            return np.full(len(tb), default)
        pts = sorted(pts, key=lambda p: p[0])
        t = np.array([p[0] for p in pts]) + np.arange(len(pts)) * 1e-7      # strictly increasing
        v = np.array([p[1] for p in pts], float)
        if v.ndim == 1:
            return np.interp(tb, t, v)
        return np.stack([np.interp(tb, t, v[:, j]) for j in range(v.shape[1])], axis=1)


def _cdur(ph: str) -> float:
    # sung consonant length: genny.speech's inherent duration. UNSOURCED: the 0.85 factor.
    return 0.85e-3 * SP.PH[ph]["dur"][0]


def _phone(tr: _Tracks, ph: str, t0: float, t1: float, FB: np.ndarray, sr: int) -> None:
    """Keyframes of one consonant between t0 and t1 (after procedural/choir.py _phone)."""
    info = SP.PH[ph]
    kind = _KIND[info["k"]]
    voiced = bool(info.get("v", True))
    place = info.get("place", "")
    r = min(0.008, (t1 - t0) / 4)
    if kind == "stop":
        burst = min(_BURST[place], (t1 - t0) * 0.2)
        vot = min(_VOT[ph], (t1 - t0) * 0.4)
        rel = t1 - vot - burst                                   # release instant
        fb = FB.copy()
        if voiced:
            fb[0] *= 0.7
        tr.add("FB", t0 + r, fb)
        tr.add("FB", rel, fb)
        tr.add("amp", t0 + r, 0.07 if voiced else 0.0)           # voice bar in a voiced closure
        tr.add("amp", rel - 0.002, 0.07 if voiced else 0.0)
        tr.add("amp", t1, 0.3 if voiced else 0.0)
        if place == "vel":                                       # compact burst at the F2 onset
            spec = ([(FB[1] * 1.08, 350.0, 1.0), (3200.0, 1200.0, 0.35), (5000.0, 3000.0, 0.1)], 0.05)
        else:
            spec = SP.BURST_SPEC[place]
        v = _fric_vec(spec, SP.BURST_DB[place] + 8.0, sr)        # +8 dB: a 10 ms burst re its mean level
        tr.add("fspec", t0, v)
        tr.add("fspec", t1 + 0.01, v)
        tr.add("fric", rel - 0.001, 0.0)
        tr.add("fric", rel + 0.0015, 1.0)
        tr.add("fric", rel + burst, 0.25)
        tr.add("fric", t1, 0.0)
        tr.add("casp", rel + burst * 0.5, 0.0)
        tr.add("casp", rel + burst, 0.0 if voiced else CASP_VOT)
        tr.add("casp", t1, 0.0 if voiced else 0.7 * CASP_VOT)
        tr.add("casp", t1 + 0.015, 0.0)
        return
    tr.add("FB", t0 + r, FB)
    tr.add("FB", t1 - r, FB)
    if kind == "fric":
        spec = SP.FRIC_SPEC.get(ph, _FRIC_X)
        v = _fric_vec(spec, SP.FRIC_DB.get(ph, -15), sr)
        tr.add("fspec", t0, v)
        tr.add("fspec", t1, v)
        e = min(0.02, (t1 - t0) / 3)
        tr.add("fric", t0, 0.0)
        tr.add("fric", t0 + e, 1.0)
        tr.add("fric", t1 - e, 1.0)
        tr.add("fric", t1, 0.0)
        tr.add("amp", t0 + r, 0.45 if voiced else 0.0)
        tr.add("amp", t1 - r, 0.45 if voiced else 0.0)
        tr.add("casp", t1 - r, 0.0 if voiced else 0.2 * CASP_VOT)
        tr.add("casp", t1 + 0.01, 0.0)
    elif kind == "asp":
        tr.add("amp", t0 + r, 0.0)
        tr.add("amp", t1 - 0.004, 0.0)
        tr.add("casp", t0, 0.0)
        tr.add("casp", t0 + min(0.02, (t1 - t0) / 3), CASP_H)
        tr.add("casp", t1, 0.9 * CASP_H)
        tr.add("casp", t1 + 0.02, 0.0)
    elif kind == "nasal":
        tr.add("nw", t0 - 0.002, 0.0)
        tr.add("nw", t0 + 0.006, 1.0)
        tr.add("nw", t1 - 0.006, 1.0)
        tr.add("nw", t1 + 0.002, 0.0)
        tr.add("nzf", t0 - 0.02, float(info.get("fnz", 1000)))
        tr.add("nzf", t1 + 0.02, float(info.get("fnz", 1000)))
        tr.add("amp", t0 + r, 0.3)                               # murmur about vowel -6 dB
        tr.add("amp", t1 - r, 0.3)
    elif kind == "liquid":
        tr.add("amp", t0 + r, 0.7)
        tr.add("amp", t1 - r, 0.7)
    elif kind == "tap":
        mid = 0.5 * (t0 + t1)
        tr.add("amp", t0, 0.9)
        tr.add("amp", mid, 0.2)
        tr.add("amp", t1, 0.9)
    elif kind == "trill":                                        # three tongue-tip closures
        for j in range(3):
            a = t0 + (t1 - t0) * j / 3
            tr.add("amp", a, 0.9)
            tr.add("amp", a + (t1 - t0) / 6, 0.2)
        tr.add("amp", t1, 0.9)
    elif kind == "glide":
        tr.add("amp", t0 + r, 0.8)
        tr.add("amp", t1 - r, 0.85)


# ==============================================================================================
# Singers
# ==============================================================================================

def _roster(types, singers: int, st: dict, seed: int, solo: bool, width: float, pan0: float = 0.0) -> list[dict]:
    """The section's singers; each a different person. UNSOURCED: the spreads of the per-singer
    traits where the docstring gives only a range."""
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, zlib.crc32("|".join(types).encode()), singers])
    out = []
    for i in range(singers):
        vt = types[i % len(types)]
        out.append(dict(
            voice=vt, idx=i,
            vtl=1.0 if solo else float(np.clip(rng.normal(1.0, 0.03), 0.93, 1.07)),   # formant shift (tract length)
            fdev=1.0 + np.clip(rng.normal(0, 0.025 if not solo else 0.012, 5), -0.06, 0.06),
            detune=0.0 if solo else float(np.clip(rng.normal(0, st["scatter"]), -2.2 * st["scatter"], 2.2 * st["scatter"])),
            vib_rate=float(st["rate"] + (0.0 if solo else rng.uniform(-0.45, 0.45))),
            vib_depth=float(st["vib"] * (1.0 if solo else st["choir"] * rng.uniform(0.7, 1.3))),
            vib_delay=float(st["delay"] * (1.0 if solo else rng.uniform(0.75, 1.5))),
            vib_fade=float(rng.uniform(0.3, 0.5)),
            # period / amplitude scatter of a sustained sung vowel: low end of the normal ranges (jitter
            # < 1 %, shimmer < 4 %, Baken & Orlikoff 2000). UNSOURCED exact values: 0.4 % / 4 % measured
            # 16 dB HNR at 1-4 kHz on this kernel (hoarse); these give about 28 dB.
            jitter=float(rng.uniform(0.001, 0.002)), shimmer=float(rng.uniform(0.01, 0.02)),
            rd_off=0.0 if solo else float(rng.normal(0, 0.12)), asp0=1.0 if solo else float(rng.uniform(0.7, 1.3)),
            level_db=0.0 if solo else float(rng.normal(0, 1.2)),
            lag=0.0 if solo else float(np.clip(rng.normal(0, st["lag"]), -2 * st["lag"], 2 * st["lag"])),
            scoop=float(st["scoop"] * rng.uniform(0.6, 1.4)), glide=float(rng.uniform(0.07, 0.13)),
            drift=float(st["drift"] * rng.uniform(0.6, 1.3)), pan=pan0,
            seed=[int(seed) & 0x7FFFFFFF, i, 9173],
        ))
    if not solo and singers > 1:
        d = np.array([s["detune"] for s in out])
        d -= d.mean()                                  # the section as a whole is in tune
        pos = np.linspace(-width, width, singers) + rng.normal(0, 0.03, singers)
        rng.shuffle(pos)
        for s, dd, p in zip(out, d, pos):
            s["detune"] = float(dd)
            s["pan"] = float(np.clip(pan0 + p, -1, 1))
    return out


def _render_singer(sg: dict, notes: list[dict], P: dict, n: int, sr: int, cons_gain: float) -> np.ndarray:
    rng = np.random.default_rng(sg["seed"])
    blk = P["blk"]
    fsb = sr / blk
    tb = np.arange(n // blk + 2) * blk / sr
    nb = len(tb)
    vt = sg["voice"]
    V = VOICES[vt]
    st = P["style"]
    sc = V["sc"] * sg["vtl"]
    solo = P["solo"]
    stac = P["artic"] == "staccato"
    marc = P["artic"] == "marcato"
    tr = _Tracks()
    N = len(notes)
    # ---- per-singer pitch of every note (mixed sections: men an octave under the women) ----
    midi = []
    for nt in notes:
        m = nt["midi"]
        if P["octaves"]:
            if V["male"] and m >= 60:
                m -= 12
            elif not V["male"] and m < 55:
                m += 12
        midi.append(m)
    fal = []
    for m in midi:                                      # falsetto share of each note
        if P["falsetto"] is True:
            fal.append(1.0)
        elif P["falsetto"] is False or not V["male"] or V["brk"] is None:
            fal.append(0.0)
        else:
            fal.append(float(_smoothstep((m - (V["brk"] - 1.0)) / 3.0)))
    # ---- timing: consonants before the beat, the vowel on it --------------------------------
    cs, vo, ve, kc = [0.0] * N, [0.0] * N, [0.0] * N, [1.0] * N
    Rn = [0.0] * N
    for i, nt in enumerate(notes):
        sy = nt["syl"]
        beat = nt["t"] + sg["lag"]
        if not solo:
            beat += float(np.clip(rng.normal(0, 0.006), -0.015, 0.015)) if sy["onset"] else \
                (0.0 if nt["jp"] else P["spread_t"] * rng.uniform(0, 1) ** 1.3)
        tc = sum(_cdur(c) for c in sy["onset"]) + sum(_cdur(c) for c in sy["coda"])
        k = min(1.0, 0.45 * nt["d"] / tc) if tc > 0 else 1.0     # short notes: shorter consonants
        Tc = k * sum(_cdur(c) for c in sy["onset"])
        floor = (vo[i - 1] + 0.04) if i else 0.0
        c0 = beat - Tc
        if c0 < floor:                                  # no room before the beat: squeeze, then delay
            k2 = max(0.6, (beat - floor) / Tc) if Tc > 0 else 1.0
            k *= k2
            Tc *= k2
            c0 = max(floor, beat - Tc)
        kc[i] = k
        cs[i] = c0
        vo[i] = c0 + Tc
    for i, nt in enumerate(notes):
        sy = nt["syl"]
        Tco = kc[i] * sum(_cdur(c) for c in sy["coda"])
        sound = nt["d"] * (0.5 if stac else 1.0)
        Rn[i] = min(P["release"] * (0.5 if stac else 1.0), 0.4 * nt["d"] + 0.02) if (P["exact"] or stac) else P["release"]
        if nt["jn"]:
            end = cs[i + 1]
        else:
            end = nt["t"] + sg["lag"] + sound - (Rn[i] if P["exact"] else 0.0)
        ve[i] = max(vo[i] + 0.03, end - Tco)
    # ---- phones: formants, amplitudes, frication ---------------------------------------------
    hold = None
    for i, nt in enumerate(notes):
        sy = nt["syl"]
        ring = P["ring"] * (1.0 - 0.7 * fal[i])
        a, b = vo[i], ve[i]
        if sy["hum"]:
            FBh = _hum_FB(vt, sy["hum"])
            nzf = float(SP.PH[sy["hum"]].get("fnz", 1000)) * V["sc"]
            for t_ in (a, b):
                tr.add("FB", t_, FBh)
                tr.add("nzf", t_, nzf)
                tr.add("amp", t_, 0.62)
            tr.add("nw", a - 0.01, 1.0)
            tr.add("nw", b + 0.01, 1.0)
            hold = FBh
            continue
        tg = []
        for v in sy["nucl"]:
            for mix in _VT[v]:
                fb = _vowel_FB(vt, mix, ring, st["f1"] * P["f1"])
                dv = np.random.default_rng(sg["seed"] + [zlib.crc32(v.encode())]).normal(0, 0.02, 2)
                fb[:2] *= 1.0 + (0.0 if solo else 1.0) * np.clip(dv, -0.05, 0.05)   # this singer's vowel
                if v in ("ER", "AXR"):
                    fb[2] *= 0.72                        # r-colouring: F3 lowered (speech.PH ER)
                tg.append(fb)
        first, last = tg[0], tg[-1]
        hold = last
        t = cs[i]
        tin = 0.03 if (nt["jp"] and nt["slur_in"]) else 0.0
        for c in sy["onset"]:
            d = kc[i] * _cdur(c)
            _phone(tr, c, t, t + d, _cons_FB(c, first, sc), sr)
            t += d
            tin = _TRANS[_KIND[SP.PH[c]["k"]]]
        tout = _TRANS[_KIND[SP.PH[sy["coda"][0]]["k"]]] if sy["coda"] else (0.04 if nt["jn"] else 0.0)
        ta = min(a + tin, 0.5 * (a + b))
        tz = max(b - tout, ta + 1e-3)
        tr.add("FB", ta, first)
        if len(tg) > 1:
            tr.add("FB", ta + 0.55 * (tz - ta), first)
        tr.add("FB", tz, last)
        if nt["jp"] and not nt["slur_in"] and not sy["onset"] and not notes[i - 1]["syl"]["coda"]:
            tr.add("amp", a - 0.05, 1.0)                # re-articulated note: a short glottal dip
            tr.add("amp", a - 0.012, 0.15)
            tr.add("amp", a + 0.03, 1.0)
        else:
            tr.add("amp", a + (0.012 if (sy["onset"] or nt["jp"]) else 0.0), 1.0)
        tr.add("amp", b - 0.004, 1.0)
        tr.add("nw", a, 0.0)
        tr.add("nw", b, 0.0)
        t = ve[i]
        for c in sy["coda"]:
            d = kc[i] * _cdur(c)
            _phone(tr, c, t, t + d, _cons_FB(c, last, sc), sr)
            t += d
    # ---- block-rate curves -------------------------------------------------------------------
    FB = tr.curve("FB", tb) if "FB" in tr.k else np.tile(hold, (nb, 1))
    FB = _lp(FB, 28.0, fsb)                             # formant trajectories, not steps
    phon = np.clip(tr.curve("amp", tb), 0.0, 1.0)
    casp = np.clip(tr.curve("casp", tb), 0.0, None) * cons_gain
    fric = np.clip(tr.curve("fric", tb), 0.0, None) * cons_gain
    nw = np.clip(_lp(np.clip(tr.curve("nw", tb), 0.0, 1.0), 60.0, fsb), 0.0, 1.0)
    nz = tr.curve("nzf", tb, 1000.0)
    fsp = tr.curve("fspec", tb) if "fspec" in tr.k else np.zeros((nb, 10))
    # ---- groups of joined notes: envelope, level, voice quality, pitch gestures --------------
    env = np.zeros(nb)
    rd = np.full(nb, RD_MF)
    cents = np.zeros(nb)
    vib_env = np.zeros(nb)
    extra = np.zeros(nb)                                # breath noise: lead-in, exhale, inhale
    dyn_t, dyn_v = P["dyn"]
    eff = np.interp(tb, dyn_t, dyn_v) * P["effort"]
    vel = np.interp(tb, [x for i in range(N) for x in (vo[i], ve[i])], [nt["vel"] for nt in notes for _ in (0, 1)])
    eff = np.clip(eff * vel, 0.02, 1.25)
    falt = np.interp(tb, [x for i in range(N) for x in (vo[i], ve[i])], [f for f in fal for _ in (0, 1)])
    mt = np.interp(tb, [x for i in range(N) for x in (vo[i], ve[i])], [m for m in midi for _ in (0, 1)])
    groups = []
    for i, nt in enumerate(notes):
        if nt["jp"] and groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    prev_end = -1e9
    for g in groups:
        i0, i1 = g[0], g[-1]
        on0 = notes[i0]["syl"]["onset"]
        t_on = cs[i0]
        t_voice = vo[i0] if on0 and not SP.PH[on0[0]].get("v", True) else t_on
        t_end = ve[i1] + kc[i1] * sum(_cdur(c) for c in notes[i1]["syl"]["coda"])
        e0 = float(np.interp(t_voice, tb, eff))
        A = 0.03 if (on0 or stac) else st["attack"] * rng.uniform(0.8, 1.25) * (1.35 - 0.6 * e0)
        A = min(A, 0.6 * notes[i0]["d"] + 0.008)
        R = Rn[i1] * (rng.uniform(0.8, 1.0) if P["exact"] else rng.uniform(0.85, 1.15))
        x = (tb - t_voice) / A
        att = np.where(tb >= t_voice, 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1)), 0.0)
        y = (tb - t_end) / max(R, 1e-3)
        rel = np.where(tb <= t_end, 1.0, np.clip(1 - y, 0, 1) ** 2 * np.exp(-2.5 * np.clip(y, 0, 1)))
        seg = (tb >= min(t_on, t_voice) - 0.2) & (tb <= t_end + R + 0.4)
        env = np.where(seg, np.maximum(env, att * rel), env)
        # voice quality: breathy onset and release (procedural/choir.py), marcato = pressed accent
        rdg = 0.5 * np.exp(-np.clip(tb - t_voice, 0, None) / 0.08) * (tb >= t_voice - 0.1) \
            + 0.7 * _smoothstep((tb - t_end) / max(R, 0.05))
        rd = np.where(seg, RD_MF + rdg, rd)
        if not on0:
            lead = rng.uniform(0.04, 0.09)
            extra += np.interp(tb, [t_voice - lead, t_voice - 0.01, t_voice + 0.06], [0, CASP_LEAD, 0], 0, 0)
        tail = rng.uniform(0.12, 0.3)
        if not P["exact"]:
            extra += np.interp(tb, [t_end + 0.3 * R, t_end + R, t_end + R + tail], [0, CASP_EXHALE, 0], 0, 0)
        gap = t_on - max(prev_end, 0.0)
        if P["breaths"] > 0 and gap >= 0.3:             # breath intake before the phrase
            d_in = min(0.55, 0.7 * gap) * rng.uniform(0.75, 1.0)
            t1 = t_on - rng.uniform(0.03, 0.07)
            extra += P["breaths"] * CASP_INHALE * np.interp(
                tb, [t1 - d_in, t1 - 0.4 * d_in, t1 - 0.08 * d_in, t1], [0, 1.0, 0.8, 0], 0, 0)
        prev_end = t_end + R
        if not stac:
            vib_env = np.where(seg & (tb >= t_on - 0.2),
                               _smoothstep((tb - (t_voice + sg["vib_delay"])) / sg["vib_fade"]), vib_env)
        scp = sg["scoop"] * rng.uniform(0.6, 1.3) * (0.5 if stac else 1.0)
        cents += np.where((tb >= t_on - 0.2) & (tb <= t_end + R),
                          -scp * np.exp(-np.clip(tb - t_voice, 0, None) / rng.uniform(0.03, 0.06)), 0.0)
        cents += -rng.uniform(5, 25) * _smoothstep((tb - t_end) / max(R, 0.05)) * seg     # falling release
    if marc:
        for i in range(N):
            acc = np.exp(-np.clip(tb - vo[i], 0, None) / 0.12) * (tb >= vo[i] - 0.01)
            eff = eff + 0.22 * acc
    # pitch targets: portamento between slurred notes, quick steps otherwise
    tgt = np.full(nb, midi[0] * 100.0)
    for i in range(N - 1):
        dm = (midi[i + 1] - midi[i]) * 100.0
        if dm == 0.0:
            continue
        if notes[i]["jn"] and notes[i]["slur_out"]:
            B0 = 0.5 * (ve[i] + vo[i + 1]) + float(rng.normal(0, 0.008))
            gl = float(np.clip(max(sg["glide"], vo[i + 1] - ve[i]) * (1 + abs(dm) / 1200), 0.05, 0.3)) * P["porta"]
            gl = max(gl, 0.02)
            tgt += dm * _smoothstep((tb - (B0 - gl / 2)) / gl)
            ov = np.sign(dm) * min(14.0, 0.10 * abs(dm)) * rng.uniform(0.3, 1.0)
            x = np.clip(tb - (B0 + gl / 2), 0, None) / 0.06
            tgt += ov * x * np.exp(1 - x)
            vib_env *= 1 - 0.6 * np.exp(-0.5 * ((tb - B0) / 0.08) ** 2)
        else:
            tgt += dm * _smoothstep((tb - (cs[i + 1] - 0.02)) / 0.025)
    cents += tgt + sg["detune"] + P["cents"]
    dr = _lp(np.cumsum(rng.normal(0, 1, nb)), 0.4, fsb)             # slow intonation drift
    dr = dr - dr[0]
    dr = dr / (np.max(np.abs(dr)) + 1e-9) * sg["drift"] * np.clip(tb / 1.5, 0, 1)
    fl = sum(np.sin(2 * np.pi * f * tb + rng.uniform(0, 6.3)) for f in (12.7, 7.1, 4.7)) * rng.uniform(0.6, 1.2)
    vr = sg["vib_rate"] * (1 + 0.03 * _lp(rng.normal(0, 1, nb), 0.5, fsb) * 8) if P["vib_rate"] is None else \
        np.full(nb, float(P["vib_rate"]) + (0.0 if solo else sg["vib_rate"] - st["rate"]) * 0.5)
    vph = np.cumsum(2 * np.pi * vr / fsb) + rng.uniform(0, 6.3)
    vd = (sg["vib_depth"] if P["vibrato"] is None else
          float(P["vibrato"]) * (1.0 if solo else sg["vib_depth"] / max(st["vib"] * st["choir"], 1e-6) if st["vib"] > 0 else 1.0))
    vd = vd * (1 + 0.1 * np.clip(_lp(rng.normal(0, 1, nb), 0.3, fsb) * 10, -1.5, 1.5)) * (1.0 - 0.6 * falt)
    cents += dr + fl + vd * vib_env * np.sin(vph)
    f0b = np.clip(440.0 * 2 ** ((cents / 100.0 - 69) / 12), 25.0, min(4000.0, 0.2 * sr))
    # voice quality (Rd): effort, pitch, breath, falsetto
    # UNSOURCED: slopes. Direction after Sundberg 1987 (loud = pressed/flat tilt, soft and high =
    # steeper tilt) and Fant 1995 (Rd about 1 modal, > 2 breathy).
    rdv = rd + st["rd"] + sg["rd_off"] - 1.5 * (eff - 0.6) + 0.7 * P["breath"] \
        + 0.4 * np.clip((mt - (V["lo"] + 7.0)) / 12.0, 0.0, 2.5)
    rdv = rdv + falt * 0.85 * (2.3 - rdv)
    rdv = np.clip(rdv, 0.5, 2.7)
    lev = 10 ** ((-LEVEL_DB * (1.0 - np.clip(eff, 0, 1.2)) + sg["level_db"] - 4.0 * falt) / 20.0)
    # ---- formants of this singer -------------------------------------------------------------
    Fs = FB[:, :NF] * sg["vtl"]
    Bs = FB[:, NF:].copy()
    Fs[:, :5] *= sg["fdev"][None, :]
    wander = np.stack([_lp(rng.normal(0, 1, nb), 0.8, fsb) for _ in range(5)], axis=1)
    Fs[:, :5] *= 1 + 0.01 * wander / (wander.std(axis=0, keepdims=True) + 1e-9)
    if P["overtone"] is not None and sg["idx"] % 3 != 1:            # sygyt-like: F2 locked on a harmonic
        hh = _lp(np.interp(tb, P["overtone"][0], P["overtone"][1]), 5.0, fsb)
        Fs[:, 1] = hh * f0b
        Fs[:, 2] = Fs[:, 1] * 1.13
        Fs[:, 0] = np.minimum(Fs[:, 0], 0.6 * Fs[:, 1])
        Bs[:, 1] = 28.0                                              # narrow: picks one harmonic
        Bs[:, 2] = 60.0
    elif P["tuning"]:
        Fs[:, :5] = tuned_formants(Fs[:, :5], f0b, vt) * (1 - nw[:, None]) + Fs[:, :5] * nw[:, None]
    for k in range(1, NF):                                           # keep the poles in order
        Fs[:, k] = np.maximum(Fs[:, k], Fs[:, k - 1] * 1.04 + 20.0)
    # yielding wall (Bilbao 9.18-9.19): low formants broadened. UNSOURCED: 50 Hz at F1 = 250 Hz
    Bs[:, 0] += 50.0 * (250.0 / np.maximum(Fs[:, 0], 120.0)) ** 2 + 25.0 * np.clip(rdv - 1.0, 0, None)
    Bs = np.clip(Bs, 30.0, 2000.0)
    Fs = np.clip(Fs, 60.0, 0.49 * sr)
    # ---- per-sample signals and kernel --------------------------------------------------------
    ts = np.arange(n) / sr
    up = lambda v: np.interp(ts, tb, v)
    voiced = 0.0 if P["whisper"] else 1.0
    amp = up(phon * env * lev) * voiced
    rds = up(rdv)
    # Klatt TL: dB down at 3 kHz (0-24). UNSOURCED: 36 dB per unit effort below f, +6 dB at breath 1
    tl_db = np.clip(36.0 * (0.72 - eff), 0.0, 24.0) + 6.0 * P["breath"] + 8.0 * falt
    fc = 3000.0 / np.sqrt(np.maximum(10 ** (tl_db / 10.0) - 1.0, 1e-3))
    tla = np.clip(1.0 - np.exp(-2 * np.pi * np.minimum(fc, 0.4 * sr) / sr), 0.0, 1.0)
    tla = np.where(tl_db < 0.3, 1.0, tla)
    # the breath noise loses half of the tilt (in dB): soft singing is breathy, not hissy
    asp_rel = 10 ** (-0.5 * up(np.clip(36.0 * (0.72 - eff), 0.0, 24.0)) / 20.0) * ASP_VOICED * sg["asp0"] * (0.25 + 0.75 * (rds - 0.5) / 2.2) * (1.0 + 2.5 * P["breath"]) * (1 + 1.5 * up(falt))
    asp = asp_rel * amp + up((casp + extra) * lev)
    if P["whisper"]:
        asp = asp + 0.9 * up(phon * env * lev)                       # Cook: whisper = noise through the vowel
    fr = up(fric * lev)
    a = math.exp(-2 * math.pi * min(ASP_TILT_HZ, 0.4 * sr) / sr)
    na = lfilter([1 - a], [1, -a], lfilter([1 - a], [1, -a], rng.standard_normal(n)))   # -12 dB/oct above 3 kHz
    na /= np.sqrt(np.mean(na * na)) + 1e-12
    tabs, flows = _tables()
    Pn = int(n * 2200 / sr) + 64
    rj = np.clip(rng.normal(0, sg["jitter"], Pn), -3 * sg["jitter"], 3 * sg["jitter"])
    rs = np.clip(rng.normal(0, sg["shimmer"], Pn), -0.2, 0.2)
    y, yf = _kernel(up(f0b), amp, (rds - 0.5) / 0.1, up(tla), np.full(n, float(P["sub"]) if sg["idx"] % 3 == 1 or solo else 0.0),
                    asp, fr, na, rng.standard_normal(n), tabs, flows, LEVEL_H,
                    np.ascontiguousarray(Fs), np.ascontiguousarray(Bs), np.ascontiguousarray(nz),
                    np.ascontiguousarray(nw), np.ascontiguousarray(fsp), blk, rj, rs, float(rng.uniform(0, 1)), 0.5, float(sr))
    if sr > 26000:                                                   # frication falls off above ~10 kHz
        a = math.exp(-2 * math.pi * 10000.0 / sr)
        yf = lfilter([1 - a], [1, -a], yf)
    return y + yf


# ==============================================================================================
# Phrase render
# ==============================================================================================

def _note_list(notes, lyrics, lang: str, legato, default_vowel: str = "ah", detached: bool = False) -> list[dict]:
    """notes: [(start_s, midi, dur_s, vel)] sorted -> note dicts with syllables and joins."""
    syl = parse_lyrics(lyrics if lyrics not in (None, "") else default_vowel, lang)
    single = len(syl) == 1 and syl[0] != "_"
    out = []
    prev = None
    for i, (t, m, d, v) in enumerate(notes):
        s = syl[0] if single else (syl[i] if i < len(syl) else "_")
        if s == "_":
            if prev is None:
                s = _special(default_vowel) or _special("ah")
            else:                                       # melisma: keep the vowel; the coda moves to the end
                s = {"onset": [], "nucl": prev["nucl"][-1:], "coda": prev["coda"], "hum": prev["hum"]}
                out[-1]["syl"] = dict(prev, coda=[])
        elif single and i > 0 and legato:               # one vowel sung legato: no new consonant
            s = dict(s, onset=[], coda=[])
        prev = s
        out.append({"t": float(t), "midi": float(m), "d": max(float(d), 0.02), "vel": float(v), "syl": dict(s),
                    "jp": False, "jn": False, "slur_in": False, "slur_out": False})
    for a, b in zip(out, out[1:]):
        if b["t"] - (a["t"] + a["d"]) <= 0.03 and not detached:
            a["jn"] = b["jp"] = True
            a["slur_out"] = b["slur_in"] = bool(legato)
    return out


def _params(sr: int, style: str, solo: bool, **kw) -> dict:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; use one of {', '.join(sorted(STYLES))}")
    st = STYLES[style]
    artic = kw.get("articulation") or "normal"
    if artic not in ("normal", "legato", "staccato", "marcato"):
        raise ValueError("articulation must be normal, legato, staccato or marcato")
    mode = kw.get("mode") or "sing"
    if mode not in ("sing", "whisper", "shout", "falsetto"):
        raise ValueError("mode must be sing, whisper, shout or falsetto")
    shout = mode == "shout"
    ring = kw.get("ring")
    if ring is None:
        ring = st["ring"] * (1.0 if solo else 0.65)     # choirs use a weaker cluster than soloists
    breath = kw.get("breath")
    return dict(
        style=st, solo=solo, blk=max(8, int(round(sr / 1500.0))), artic=artic,
        whisper=mode == "whisper", falsetto=True if mode == "falsetto" else kw.get("falsetto"),
        ring=float(ring), breath=float(np.clip(st["breath"] if breath is None else breath, 0.0, 1.0)),
        effort=float(kw.get("effort", 1.0)) * (1.35 if shout else 1.0),
        f1=1.08 if shout else 1.0,                      # UNSOURCED size: shouting raises F1 (Lienard & Di Benedetto 1999)
        vibrato=kw.get("vibrato"), vib_rate=kw.get("vib_rate"), tuning=bool(kw.get("tuning", True)),
        release=float(kw.get("release", 0.16)), exact=bool(kw.get("exact", False)),
        breaths=float(kw.get("breaths", 0.0)), spread_t=float(kw.get("timing", 2.0 * st["lag"])),
        porta=float(kw.get("portamento", 1.0)), cents=float(kw.get("cents", 0.0)),
        overtone=kw.get("overtone"), sub=float(np.clip(kw.get("sub", 0.0), 0.0, 0.6)),
        octaves=bool(kw.get("octaves", False)), dyn=kw.get("dyn"),
    )


def render_singers(notes, sr: int = DEFAULT_SR, *, voice=None, section=None, singers=None, lyrics=None,
                   lang: str = "la", legato: bool = True, dynamics="mf", style: str = "classical", seed: int = 0,
                   spread: float = 0.6, pan: float = 0.0, total: float | None = None, **kw):
    """The individual singers of a phrase: ([(mono stream, pan)], n_samples).

    notes: [(start_s, midi, dur_s, vel)]. `voice` = one voice type (solo unless `singers` > 1),
    `section` = a SECTIONS key. Every stream is one person: own source, tract and random stream."""
    notes = sorted(((float(a), float(b), float(c), float(d)) for a, b, c, d in notes), key=lambda x: x[0])
    if not notes:
        raise ValueError("no notes to sing")
    if section is not None and str(section) not in ("auto", ""):
        sec = str(section).lower()
        types = SECTIONS.get(sec) or (_voice_name(sec),)
        nsing = 6 if singers is None else int(singers)
    else:
        mean_m = float(np.mean([m for _, m, _, _ in notes]))
        types = (_voice_name(voice) if voice not in (None, "auto", "") else _auto_voice(mean_m),)
        nsing = (1 if section is None else 6) if singers is None else int(singers)
    nsing = int(np.clip(nsing, 1, 48))
    solo = nsing == 1
    exact = bool(kw.get("exact", False))
    end = max(t + d for t, _, d, _ in notes)
    if total is None:
        total = end if exact else end + float(kw.get("release", 0.16)) * 1.2 + 0.06
    n = max(8, int(round(total * sr)))
    P = _params(sr, style, solo, **kw)
    P["dyn"] = parse_dynamics(dynamics, end)
    P["octaves"] = P["octaves"] or (section is not None and str(section).lower() in ("mixed", "satb"))
    nl = _note_list(notes, lyrics, lang, (legato or P["artic"] == "legato") and P["artic"] != "staccato",
                    detached=P["artic"] == "staccato")
    ros = _roster(types, nsing, P["style"], seed, solo, float(np.clip(spread, 0.0, 1.0)), float(pan))
    # consonant noise at full level on the three tightest singers only, so a section's "s" is one
    # consonant and not a smear (procedural/choir.py CONS_VOICES; UNSOURCED: 3 and the 0.3)
    tight = set(np.argsort([abs(s["lag"]) for s in ros])[:3].tolist())
    out = []
    for i, sg in enumerate(ros):
        cg = 1.0 if nsing <= 3 else (1.3 if i in tight else 0.3)
        out.append((_render_singer(sg, [dict(x, syl=dict(x["syl"])) for x in nl], P, n, sr, cg), sg["pan"]))
    return out, n


def sing_phrase(notes, sr: int = DEFAULT_SR, *, stereo: bool = True, bright: float = 1.0, **kw) -> np.ndarray:
    """Render a phrase: every singer continuously across it, summed (stereo (n, 2) or mono).

    Keywords as `render_singers`, plus: articulation normal|legato|staccato|marcato, mode
    sing|whisper|shout|falsetto, vibrato (+- cents), vib_rate (Hz), breath 0..1, ring 0..1.5,
    effort (x), breaths 0..1 (intake noise), exact (release inside the last note: exact length),
    total (buffer seconds), overtone ((times, harmonic numbers)), sub 0..0.6 (kargyraa)."""
    streams, n = render_singers(notes, sr, **kw)
    k = 1.0 / math.sqrt(len(streams))
    if stereo:
        y = np.zeros((n, 2))
        for s, p in streams:
            th = (float(np.clip(p, -1, 1)) + 1) * math.pi / 4
            y[:, 0] += math.cos(th) * s
            y[:, 1] += math.sin(th) * s
        y *= k * math.sqrt(2.0) if len(streams) == 1 else k * 1.2
    else:
        y = sum(s for s, _ in streams) * k
    y = FL.highpass(y, 55.0, sr) * _CAL
    if abs(bright - 1.0) > 1e-3:
        y = FL.highshelf(y, 3000.0, sr, gain_db=float(np.clip(9.0 * math.log2(max(bright, 0.05)), -18.0, 12.0)))
    m = min(n // 2, int(0.004 * sr))
    if m > 1:
        w = np.linspace(0.0, 1.0, m)
        y[:m] = y[:m] * (w[:, None] if y.ndim == 2 else w)
        y[-m:] = y[-m:] * (w[::-1, None] if y.ndim == 2 else w[::-1])
    pk = float(np.max(np.abs(y))) if n else 0.0
    if pk > 0.98:
        y *= 0.98 / pk
    return y


# ==============================================================================================
# Catalog instruments (mono: one note = one sung note of a soloist or a section)
# ==============================================================================================

def _note(freq, dur, sr, vel, *, vowel="ah", level_db=0.0, **kw) -> np.ndarray:
    midi = 69.0 + 12.0 * math.log2(max(float(freq), 20.0) / 440.0)
    vel = float(np.clip(vel, 0.02, 1.0))
    eff = 0.15 + 0.62 * vel                              # UNSOURCED: velocity -> effort (0.9 = f)
    dur = max(float(dur), 0.02)
    y = sing_phrase([(0.0, midi, dur, 1.0)], sr, stereo=False, lyrics=vowel, dynamics=eff, **kw)
    target = -12.0 + level_db + 20.0 * math.log10(vel / 0.9)       # contract: -12 dB K-weighted at vel 0.9
    y = y * 10 ** ((target - loudness(y, sr, window=0.15)) / 20.0)
    pk = float(np.max(np.abs(y)))
    return y * (1.4 / pk) if pk > 1.4 else y


_VIB_HELP = "vibrato extent, +- cents peak (-1 = the style's own: classical solo 60, choir 30; 0 = straight tone)"
_STYLE_HELP = "classical|cathedral|gregorian|gospel|pop|children|epic|throat|bulgarian"


def _v(x):
    return None if x is None or float(x) < 0 else float(x)


@instrument("voice", "Solo singer: LF glottal source through a five-formant vocal tract; sings a vowel or a syllable.",
            family="voice", span=("C2", "C6"),
            voice=("auto", "auto (by pitch)|bass|baritone|tenor|countertenor|alto|mezzo|soprano|treble"),
            vowel=("ah", "vowel or syllable: ah eh ee oh oo, mm (hum), or a syllable such as la, ta, sanc; /T AA/ = raw phonemes"),
            vibrato=(-1.0, _VIB_HELP), vib_rate=(-1.0, "vibrato rate Hz, 4.5..7 (-1 = style)"),
            breath=(-1.0, "breathiness 0 (clean) .. 1 (airy); -1 = style"),
            effort=(1.0, "vocal effort x velocity: 0.5 soft/dark .. 1.4 pressed/bright (changes the source spectrum)"),
            ring=(-1.0, "singer's formant 0 (untrained) .. 1 (operatic) .. 1.5; -1 = style"),
            mode=("sing", "sing|whisper|shout|falsetto"), style=("classical", _STYLE_HELP),
            lang=("la", "how a syllable is read: la (Latin vowels)|en|es"),
            bright=(1.0, "0.5 dark .. 2 bright (shelf above 3 kHz)"), seed=(0, "which singer / take"))
def voice(freq, dur, sr=DEFAULT_SR, vel=1.0, voice="auto", vowel="ah", vibrato=-1.0, vib_rate=-1.0, breath=-1.0,
          effort=1.0, ring=-1.0, mode="sing", style="classical", lang="la", bright=1.0, seed=0):
    return _note(freq, dur, sr, vel, vowel=vowel, voice=voice, singers=1, vibrato=_v(vibrato), vib_rate=_v(vib_rate),
                 breath=_v(breath), effort=effort, ring=_v(ring), mode=mode, style=style, lang=lang, bright=bright,
                 seed=seed)


@instrument("vocal_choir", "Choir section of individual singers (own tract, pitch, vibrato, timing each): sopranos .. basses, boys, mixed.",
            family="voice", span=("C2", "C6"),
            section=("auto", "auto (by pitch)|sopranos|altos|tenors|basses|baritones|mezzos|boys|men|women|mixed (men an octave under the women)"),
            singers=(8, "number of singers 1..48 (1 = solo)"),
            vowel=("ah", "vowel or syllable: ah eh ee oh oo, mm (hum), la, ta, ..."),
            spread=(1.0, "how different the singers are: scales pitch scatter and timing spread (0 = tight, 2 = loose)"),
            vibrato=(-1.0, _VIB_HELP), breath=(-1.0, "breathiness 0..1; -1 = style"),
            style=("classical", _STYLE_HELP), lang=("la", "la|en|es"),
            bright=(1.0, "0.5 dark .. 2 bright"), seed=(0, "which choir / take"))
def vocal_choir(freq, dur, sr=DEFAULT_SR, vel=1.0, section="auto", singers=8, vowel="ah", spread=1.0, vibrato=-1.0,
                breath=-1.0, style="classical", lang="la", bright=1.0, seed=0):
    name = _scaled_style(style, float(np.clip(spread, 0.0, 3.0)))
    return _note(freq, dur, sr, vel, vowel=vowel, section=section, singers=singers, vibrato=_v(vibrato),
                 breath=_v(breath), style=name, lang=lang, bright=bright, seed=seed)


def _scaled_style(style: str, k: float) -> str:
    """Register a copy of a style with scatter and lag scaled by k (the `spread` control)."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; use one of {', '.join(sorted(STYLES))}")
    if abs(k - 1.0) < 1e-6:
        return style
    name = f"{style}*{k:.3f}"
    if name not in STYLES:
        STYLES[name] = dict(STYLES[style], scatter=STYLES[style]["scatter"] * k, lag=STYLES[style]["lag"] * k)
    return name


@instrument("hum", "Closed-mouth humming (mm): nasal murmur, soft and dark; solo or a section.",
            family="voice", span=("C2", "C5"),
            singers=(6, "number of singers (1 = solo)"), section=("auto", "auto|sopranos|altos|tenors|basses|men|women|mixed"),
            consonant=("m", "m (lips closed)|n (tongue, a little brighter)"),
            vibrato=(-1.0, _VIB_HELP), breath=(-1.0, "breathiness 0..1; -1 = style"),
            style=("classical", _STYLE_HELP), seed=(0, "take"))
def hum(freq, dur, sr=DEFAULT_SR, vel=1.0, singers=6, section="auto", consonant="m", vibrato=-1.0, breath=-1.0,
        style="classical", seed=0):
    return _note(freq, dur, sr, vel, vowel="nn" if str(consonant).lower().startswith("n") else "mm", section=section,
                 singers=singers, vibrato=_v(vibrato), breath=_v(breath), style=style, seed=seed)


@instrument("throat_singing", "Overtone / throat singing: pressed low drone with a narrow formant locked on one harmonic (sygyt), optional subharmonic (kargyraa).",
            family="voice", span=("C2", "C3"),
            harmonic=(8.0, "which harmonic of the drone the whistle sits on, 5..14"),
            harmonic2=(-1.0, "glide to this harmonic over the note (-1 = stay)"),
            sub=(0.0, "kargyraa: strength of the octave-below period doubling 0..0.6"),
            singers=(1, "number of singers"), seed=(0, "take"))
def throat_singing(freq, dur, sr=DEFAULT_SR, vel=1.0, harmonic=8.0, harmonic2=-1.0, sub=0.0, singers=1, seed=0):
    h1 = float(np.clip(harmonic, 3.0, 20.0))
    h2 = h1 if harmonic2 is None or float(harmonic2) <= 0 else float(np.clip(harmonic2, 3.0, 20.0))
    return _note(freq, dur, sr, vel, vowel="oh", voice="bass", singers=singers, style="throat", sub=sub,
                 overtone=(np.array([0.0, max(dur, 0.05)]), np.array([h1, h2])), seed=seed)


@instrument("falsetto", "Male falsetto / head voice: breathy, flute-like, almost no upper partials.",
            family="voice", span=("A3", "E5"),
            voice=("tenor", "tenor|baritone|bass|countertenor"), vowel=("oo", "vowel or syllable"),
            singers=(1, "number of singers"), vibrato=(-1.0, _VIB_HELP), breath=(-1.0, "breathiness 0..1; -1 = style"),
            style=("classical", _STYLE_HELP), seed=(0, "take"))
def falsetto(freq, dur, sr=DEFAULT_SR, vel=1.0, voice="tenor", vowel="oo", singers=1, vibrato=-1.0, breath=-1.0,
             style="classical", seed=0):
    return _note(freq, dur, sr, vel, vowel=vowel, voice=voice, singers=singers, vibrato=_v(vibrato), breath=_v(breath),
                 style=style, mode="falsetto", seed=seed)


@instrument("boys_choir", "Boys' / children's choir: treble voices, short tracts, nearly straight tone, a little breath.",
            family="voice", span=("C4", "G5"),
            singers=(8, "number of singers"), vowel=("ah", "vowel or syllable"),
            vibrato=(-1.0, _VIB_HELP), breath=(-1.0, "breathiness 0..1; -1 = style"), seed=(0, "take"))
def boys_choir(freq, dur, sr=DEFAULT_SR, vel=1.0, singers=8, vowel="ah", vibrato=-1.0, breath=-1.0, seed=0):
    return _note(freq, dur, sr, vel, vowel=vowel, section="boys", singers=singers, vibrato=_v(vibrato),
                 breath=_v(breath), style="children", seed=seed)


@instrument("chant", "Gregorian chant: unison men (tenors, baritones, basses), straight tone, soft onsets.",
            family="voice", span=("A2", "D4"),
            singers=(8, "number of monks"), vowel=("oh", "vowel or syllable"),
            vibrato=(-1.0, _VIB_HELP), breath=(-1.0, "breathiness 0..1; -1 = style"), seed=(0, "take"))
def chant(freq, dur, sr=DEFAULT_SR, vel=1.0, singers=8, vowel="oh", vibrato=-1.0, breath=-1.0, seed=0):
    return _note(freq, dur, sr, vel, vowel=vowel, section="men", singers=singers, vibrato=_v(vibrato),
                 breath=_v(breath), style="gregorian", seed=seed)


# ==============================================================================================
# Layer types: sing, satb
# ==============================================================================================

_COMMON = {"type", "at", "gain", "fx", "repeat", "every", "bpm"}
_SING_KEYS = {"voice", "section", "singers", "steps", "notes", "step", "lyrics", "lang", "legato", "articulation",
              "dynamics", "style", "seed", "transpose", "vibrato", "vib_rate", "breath", "ring", "effort", "mode",
              "bright", "spread", "pan", "breaths", "lead", "tail", "vel", "overtones", "sub", "portamento", "timing"}
_SATB_KEYS = (_SING_KEYS - {"voice", "section", "steps", "notes", "pan", "overtones", "sub"}) | {
    "chords", "parts", "soprano", "alto", "tenor", "bass", "balance"}


def _steps(layer: dict, steps, q: float):
    """Step notation (as `seq`) -> ([(start_s, midi, dur_s, vel)], total_s). Durations in beats
    when a tempo is in effect. A chord sings its first note."""
    default = float(layer.get("step", 0.25))
    if isinstance(steps, str):
        st = parse_sequence(steps, default)
    else:
        st = []
        for s in steps:
            if isinstance(s, str):
                st.extend(parse_sequence(s, default))
            else:
                p = s.get("notes", s.get("note", s.get("pitches", [])))
                st.append({"pitches": parse_pitch_list(p) if p not in ([], None, "-") else [],
                           "dur": float(s.get("dur", default)), "vel": float(s.get("vel", 1.0))})
    tr = float(layer.get("transpose", 0))
    vel = float(layer.get("vel", 1.0))
    notes, t = [], 0.0
    for s in st:
        d = s["dur"] * q
        if s["pitches"]:
            notes.append((t, 69.0 + 12.0 * math.log2(s["pitches"][0] / 440.0) + tr, d, vel * s["vel"]))
        t += d
    return notes, t


def _sing_kw(layer: dict, q: float) -> dict:
    kw = dict(lyrics=layer.get("lyrics"), lang=layer.get("lang", "la"), legato=bool(layer.get("legato", True)),
              articulation=layer.get("articulation"), dynamics=layer.get("dynamics", "mf"),
              style=layer.get("style", "classical"), seed=int(layer.get("seed", 0)), mode=layer.get("mode"),
              bright=float(layer.get("bright", 1.0)), spread=float(layer.get("spread", 0.6)),
              breaths=float(layer.get("breaths", 0.0)), effort=float(layer.get("effort", 1.0)),
              portamento=float(layer.get("portamento", 1.0)))
    for k in ("vibrato", "vib_rate", "breath", "ring", "timing"):
        if layer.get(k) is not None:
            kw[k] = float(layer[k])
    return kw


def _check(layer: dict, allowed: set, name: str) -> None:
    bad = [k for k in layer if k not in allowed and k not in _COMMON]
    if bad:
        raise SpecError(f"{name} layer does not accept {bad}; allowed: {sorted(allowed)}")


def _sing_part(layer: dict, steps, sr: int, q: float, **over) -> np.ndarray:
    notes, total = _steps(layer, steps, q)
    lead = max(0.0, float(layer.get("lead", 0.0)))
    tail = max(0.0, float(layer.get("tail", 0.0)))
    if not notes:
        return np.zeros((max(1, int(round((lead + total + tail) * sr))), 2))
    kw = _sing_kw(layer, q)
    kw.update(over)
    if layer.get("overtones") is not None:
        hs = [float(h) for h in re.split(r"[\s,]+", str(layer["overtones"]).strip()) if h] \
            if isinstance(layer["overtones"], str) else [float(h) for h in layer["overtones"]]
        kw["overtone"] = (lead + np.linspace(0.0, total, len(hs) + 1)[:-1] + 0.5 * total / len(hs), np.array(hs))
        kw["sub"] = float(layer.get("sub", 0.0))
        kw.setdefault("style", "throat")
    n = int(round((lead + total + tail) * sr))
    return sing_phrase([(t + lead, m, d, v) for t, m, d, v in notes], sr, total=n / sr, exact=tail <= 0.0,
                       release=0.16 if tail <= 0.0 else min(0.3, max(0.12, tail)), **kw)[:n]


@layer_type("sing")
def sing_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    """{"type": "sing", "voice": "soprano" | "section": "tenors", "steps": "C4:1 D4:1 E4:2",
    "lyrics": "A-ve Ma-ri-a", "singers": 8, "legato": true, "dynamics": "mp<f>p", "style": "cathedral",
    "seed": 1}. One glottal source and one moving tract per singer across the whole phrase; stereo.
    The buffer is exactly `lead` + the steps' length (+ `tail` seconds): beat 1 is at `lead`."""
    _check(layer, _SING_KEYS, "sing")
    steps = layer.get("steps", layer.get("notes"))
    if steps is None:
        raise SpecError('sing layer needs "steps" (e.g. "C4:1 D4:1 E4:2")')
    try:
        return _sing_part(layer, steps, sr, q, voice=layer.get("voice"), section=layer.get("section"),
                          singers=layer.get("singers"), pan=float(layer.get("pan", 0.0)))
    except ValueError as e:
        raise SpecError(str(e)) from e


_SATB = (("soprano", "sopranos", 60, 79, -0.45), ("alto", "altos", 55, 72, 0.45),
         ("tenor", "tenors", 48, 67, -0.15), ("bass", "basses", 40, 60, 0.15))   # name, section, range, stage position


def voice_lead(chords: list[list[float]]) -> list[list[float]]:
    """Four-part voicing of a chord progression (each chord = MIDI notes, root first): bass on the
    root, the upper voices on the nearest chord tones that cover the chord without crossing.
    Returns [soprano, alto, tenor, bass] lines. UNSOURCED: the cost weights (common-practice rules:
    least motion, complete chords, no crossing)."""
    import itertools
    lines = [[], [], [], []]
    prev = [72.0, 65.0, 58.0, 48.0]
    for ch in chords:
        pcs = [int(round(m)) % 12 for m in ch]
        root = pcs[0]
        cand = []
        for (_, _, lo, hi, _), p, pool in zip(_SATB, prev, (pcs, pcs, pcs, [root])):
            c = [m for m in range(lo, hi + 1) if m % 12 in pool]
            cand.append(sorted(c, key=lambda m: abs(m - p))[:5])
        best, cost_best = None, 1e18
        for s, a, t, b in itertools.product(*cand):
            if not (s > a > t > b):
                continue
            cost = abs(s - prev[0]) * 1.2 + abs(a - prev[1]) + abs(t - prev[2]) + abs(b - prev[3]) * 0.6
            cost += 6.0 * len(set(pcs) - {s % 12, a % 12, t % 12, b % 12})
            cost += 3.0 * max(0, (s - a) - 12) + 3.0 * max(0, (a - t) - 12)
            if cost < cost_best:
                best, cost_best = (s, a, t, b), cost
        if best is None:
            best = tuple(c[0] for c in cand)
        for k in range(4):
            lines[k].append(float(best[k]))
        prev = list(best)
    return lines


@layer_type("satb")
def satb_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    """Four `sing` parts with shared lyrics. Either {"chords": "C4:maj:2 F4:maj:2 G4:dom7:2 C4:maj:2"}
    (voiced automatically, root in the bass) or the four lines: {"soprano": steps, "alto": steps,
    "tenor": steps, "bass": steps} (or "parts": {...}). Other keys as `sing`; `singers` is per
    section; "balance": [s, a, t, b] gains."""
    _check(layer, _SATB_KEYS, "satb")
    parts = dict(layer.get("parts") or {})
    for name, *_ in _SATB:
        if layer.get(name) is not None:
            parts[name] = layer[name]
    if not parts:
        if layer.get("chords") is None:
            raise SpecError('satb layer needs "chords" or the parts "soprano" / "alto" / "tenor" / "bass"')
        default = float(layer.get("step", 1.0))
        st = parse_sequence(layer["chords"], default) if isinstance(layer["chords"], str) else layer["chords"]
        idx = [i for i, s in enumerate(st) if s["pitches"]]
        lines = voice_lead([[69 + 12 * math.log2(f / 440.0) for f in st[i]["pitches"]] for i in idx])
        for k, (name, *_) in enumerate(_SATB):
            seq, j = [], 0
            for i, s in enumerate(st):
                if s["pitches"]:
                    seq.append({"note": f"{midi_to_freq(lines[k][j]):.6f}hz", "dur": s["dur"], "vel": s["vel"]})
                    j += 1
                else:
                    seq.append({"note": "-", "dur": s["dur"]})
            parts[name] = seq
    bal = list(layer.get("balance") or [1.0, 0.9, 0.9, 1.0])
    width = float(layer.get("spread", 0.6))
    sub = {k: v for k, v in layer.items() if k not in ("chords", "parts", "soprano", "alto", "tenor", "bass", "balance")}
    out = None
    try:
        for k, (name, section, _, _, pos) in enumerate(_SATB):
            if name not in parts:
                continue
            y = _sing_part(dict(sub, spread=0.35 * width), parts[name], sr, q, section=section,
                           singers=layer.get("singers", 6), pan=pos * width / 0.6,
                           seed=int(layer.get("seed", 0)) * 4 + k) * float(bal[k])
            if out is None:
                out = y
            else:
                m = max(len(out), len(y))
                out = np.pad(out, ((0, m - len(out)), (0, 0))) + np.pad(y, ((0, m - len(y)), (0, 0)))
    except ValueError as e:
        raise SpecError(str(e)) from e
    out = out * 0.6
    pk = float(np.max(np.abs(out)))
    return out * (0.98 / pk) if pk > 0.98 else out

