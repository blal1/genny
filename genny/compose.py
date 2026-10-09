"""Deterministic algorithmic composition and performance tools: music from a seed instead of
hand-written notes.

Layer types: `compose` (one generated part), `arrangement` (a whole piece from a style name),
`euclid`, `scatter`, `arp`, `stinger`, `adaptive`.  Instrument `ks_string`, drum `ks_drum`.
Helpers: `generate(layer)` -> note events, `to_spec(layer)` -> ordinary seq/pattern layers,
`harmony(layer)`, `arrange(layer)`, `automix(layers)`, `euclid(k, n)`, `voice_lead(...)`.

Everything a generated layer plays is first expanded to plain `seq` / `pattern` layers
(`to_spec`) and rendered by `genny.spec`; there is one code path, so the editable export sounds
exactly like the generator.

Sources (findings files in out/research):
  g15 s1   Farnell, Composition 000/001/002 + MEX: counter/modulo/select masks, half-chance gates,
           chord-slot roles, tempo-locked delays, the unresolved chord lists, arp pool, 303 tuples;
           course drafts: Zen-game random walk, Markov chains; Farnell 2007: Collatz melodies.
  g15 s3-4 Jaffe & Smith 1983 / Karplus & Strong 1983 (CMJ 7(2)): `ks_string`, `ks_drum`.
  g15 s6   Perez-Gonzalez & Reiss, DAFx-07 automatic panner: `automix`.
  g15 s7-8 Farnell LOD (`silence_p`), iMuse patent US 5,315,057: `adaptive`.
  12-music Phillips 2014 (loops, stingers, vertical layering, horizontal re-sequencing);
           Ariza, MIT 21M.380 (Beta velocities, Markov, 1/f, tempo drift); Strudel `euclid`;
           FMOD Scatterer.
  g08 s4.3 Colvig/Harrison tunings: Ptolemy syntonic diatonic, the "7-11" pentatonic.
`TempoMap`-style drift, the sine velocity contour and the motif operators follow
procedural/music.py by Chris Nash (Klang Open License 1.0, see THIRD_PARTY_NOTICES.md).
"""
from __future__ import annotations

import json
import math
import re
import zlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from scipy.signal import lfilter, resample_poly

from . import filters as F
from . import fx as FX
from . import notes as N
from . import spec as SP
from .core import DEFAULT_SR, mix, pad_to, samples, to_mono, to_stereo, write_wav
from .drums import REGISTRY as DRUMS
from .drums import drum
from .env import perc
from .instruments import _kt, _off, instrument
from .spec import SpecError, layer_type

# ============================================================================ theory: scales
_NEW_SCALES = {
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "hijaz": [0, 1, 4, 5, 7, 8, 10],          # D Eb F# G A Bb C (AGENTS.md, middle_eastern)
    "in": [0, 1, 5, 7, 8],                    # D Eb G A Bb (AGENTS.md, japanese)
    "hirajoshi": [0, 2, 3, 7, 8],
    "pelog": [0, 1, 3, 7, 8],                 # UNSOURCED: usual 12-TET stand-in for 5-tone pelog
    "slendro": [0, 2, 5, 7, 9],               # UNSOURCED: usual 12-TET stand-in for slendro
    "octatonic": [0, 2, 3, 5, 6, 8, 9, 11],   # whole-half; skip-one chords are diminished sevenths
    "octatonic_hw": [0, 1, 3, 4, 6, 7, 9, 10],
    "bebop": [0, 2, 4, 5, 7, 9, 10, 11],      # bebop dominant
    "bebop_major": [0, 2, 4, 5, 7, 8, 9, 11],
}
for _k, _v in _NEW_SCALES.items():            # also visible to `genny list scales` and notes.scale()
    N.SCALES.setdefault(_k, _v)

# scales given as frequency ratios to the tonic (not 12-TET)
RATIO_SCALES = {
    "ptolemy": [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8],      # syntonic diatonic, g08 s4.3
    "seven_eleven": [1, 7 / 6, 4 / 3, 3 / 2, 11 / 6],                # harmonics 6:7:8:9:11:12, g08 s4.3
    "slendro_equal": [2 ** (i / 5) for i in range(5)],               # UNSOURCED: 5-tone equal idealisation
}
RATIO_SCALES["7-11"] = RATIO_SCALES["seven_eleven"]
# UNSOURCED: the usual 5-limit chromatic, used by "tuning": "just" to retune any 12-TET scale
JUST12 = [1, 16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3, 45 / 32, 3 / 2, 8 / 5, 5 / 3, 9 / 5, 15 / 8]


def _ratio(v) -> float:
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        return float(a) / float(b)
    return float(v)


def scale_offsets(scale="major", tuning: str = "equal") -> list[float]:
    """Scale -> ascending offsets from the tonic in (possibly fractional) semitones, first = 0.

    `scale`: a name (`genny list scales`, or ptolemy / seven_eleven ("7-11") / slendro_equal);
    a list starting with 0 = semitones; a list starting with 1 (or of "a/b" strings) = frequency
    ratios; {"ratios": [...]}, {"cents": [...]} or {"semitones": [...]}.
    `tuning="just"` retunes the 12-TET degrees to 5-limit ratios (JUST12)."""
    if isinstance(scale, dict):
        if "ratios" in scale:
            offs = [12 * math.log2(_ratio(r)) for r in scale["ratios"]]
        elif "cents" in scale:
            offs = [float(c) / 100 for c in scale["cents"]]
        else:
            offs = [float(s) for s in scale["semitones"]]
    elif isinstance(scale, (list, tuple)):
        if any(isinstance(v, str) for v in scale) or _ratio(scale[0]) == 1.0:
            offs = [12 * math.log2(_ratio(r)) for r in scale]
        else:
            offs = [float(s) for s in scale]
    elif scale in RATIO_SCALES:
        offs = [12 * math.log2(r) for r in RATIO_SCALES[scale]]
    elif scale in N.SCALES:
        offs = [float(s) for s in N.SCALES[scale]]
    else:
        raise SpecError(f"unknown scale {scale!r}; known: {sorted(N.SCALES) + sorted(RATIO_SCALES)}")
    if tuning == "just":
        offs = [12 * math.log2(JUST12[int(o)]) if float(o).is_integer() else o for o in offs]
    elif tuning != "equal":
        raise SpecError('tuning must be "equal" or "just"')
    offs = sorted({round(o % 12, 6) for o in offs})
    if not offs or offs[0] != 0:
        raise SpecError("a scale must contain its tonic (offset 0 / ratio 1)")
    return offs


def _key_pc(key) -> int:
    key = str(key).strip()
    return N.note_to_midi(key if key[-1].isdigit() else key + "4") % 12


def _pitches(kpc: float, pcs, lo: float, hi: float) -> list[float]:
    """Every pitch (MIDI number, may be fractional) with one of the pitch classes `pcs` in [lo, hi]."""
    out = {round(kpc + 12 * o + x, 6) for o in range(-1, 11) for x in pcs}
    return sorted(p for p in out if lo - 1e-6 <= p <= hi + 1e-6)


def _near(cands, x):
    return min(cands, key=lambda p: (abs(p - x), p))


def quantize(pitch: float, key="C", scale="major") -> float:
    """Nearest pitch of the scale (MIDI numbers)."""
    return _near(_pitches(_key_pc(key), scale_offsets(scale), pitch - 12, pitch + 12), pitch)


# ============================================================================ theory: chords
_RN = re.compile(r"^([b#]?)([iv]+|[IV]+)(maj7|m7|7|9|5)?$")
_DEG = {"i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6}
_EXT = {None: 0, "maj7": 4, "m7": 4, "7": 4, "9": 5, "5": 2}


def chord_pcs(degree: int, offs: list[float], size: int = 3) -> list[float]:
    """Diatonic chord on a scale degree (0-based): stacked scale thirds, root first; size 2 = root + fifth."""
    L = len(offs)
    idx = [0, 4] if size == 2 else [2 * i for i in range(size)]
    out = []
    for i in idx:
        pc = offs[(degree + i) % L]
        if pc not in out:
            out.append(pc)
    return out


def parse_progression(text: str, offs: list[float], size: int = 3) -> list[dict]:
    """"i VII VI7 V" -> chords. A numeral names a scale degree; the chord is built from the scale itself
    (so every part stays in the scale); the case of the numeral is not used. Suffix 7/maj7/m7 = four
    notes, 9 = five, 5 = root + fifth. `|` is ignored."""
    out = []
    for tok in str(text).replace("|", " ").split():
        m = _RN.match(tok)
        if not m:
            raise SpecError(f"bad chord {tok!r} in progression (roman numerals i..vii, optional 7/maj7/9/5)")
        if m.group(1):
            raise SpecError(f"{tok!r}: altered degrees are not supported; pick the mode that contains the chord "
                            "(mixolydian for bVII, harmonic_minor for a major V in minor)")
        d = _DEG[m.group(2).lower()] % len(offs)
        pcs = chord_pcs(d, offs, _EXT[m.group(3)] or size)
        out.append({"roman": tok, "root": pcs[0], "pcs": pcs})
    return out


def voice_lead(prev, pcs, key_pc: float = 0, lo: float = 48, hi: float = 72) -> list[float]:
    """Nearest-inversion voice leading: the close-position inversion of `pcs` (offsets from the key)
    inside [lo, hi] that moves least from the previous voicing `prev` (or sits mid-range)."""
    cands = []
    for r in range(len(pcs)):
        rot = list(pcs[r:]) + list(pcs[:r])
        for o in range(-1, 10):
            v = [key_pc + 12 * o + rot[0]]
            for pc in rot[1:]:
                p = key_pc + 12 * o + pc
                while p <= v[-1]:
                    p += 12
                v.append(p)
            if v[0] >= lo - 1e-6:
                cands.append(v)
    inside = [v for v in cands if v[-1] <= hi + 1e-6]
    cands = inside or [min(cands, key=lambda v: v[0])]
    ref = list(prev) if prev else [(lo + hi) / 2.0]

    def cost(v):
        return (sum(min(abs(p - q) for q in ref) for p in v) + sum(min(abs(p - q) for p in v) for q in ref), v[0])
    return [round(p, 6) for p in min(cands, key=cost)]


def functional_progression(rng, n: int = 4) -> str:
    """Tonic -> subdominant -> dominant walk that ends on a dominant (turnaround).
    # UNSOURCED: transition weights are textbook functional harmony, not from the findings files."""
    cls = {"T": ["I", "vi", "iii"], "S": ["IV", "ii"], "D": ["V", "vii"]}
    nxt = {"T": (["T", "S", "D"], [0.2, 0.5, 0.3]), "S": (["T", "S", "D"], [0.3, 0.1, 0.6]), "D": (["T", "S", "D"], [0.8, 0.1, 0.1])}
    out, cur = ["I"], "T"
    for i in range(1, n):
        cur = "D" if (i == n - 1 and n >= 4) else str(rng.choice(nxt[cur][0], p=nxt[cur][1]))
        out.append(str(rng.choice(cls[cur], p=[0.7, 0.3] if len(cls[cur]) == 2 else [0.6, 0.3, 0.1])))
    return " ".join(out)


# Farnell C002 "Moon over Pokesdown station": three 8-slot chord lists in tension with no cadence
# (g15 s1.5), 8 bars each. All notes lie in D mixolydian.
POKESDOWN = [[50, 52, 66, 69, 76, 79, 83, 86], [50, 48, 64, 67, 76, 79, 81, 86], [52, 48, 64, 67, 76, 79, 81, 83]]


# ============================================================================ theory: rhythm
def euclid(k: int, n: int, rotate: int = 0) -> list[bool]:
    """Euclidean rhythm E(k, n) (Bjorklund / Toussaint; Strudel `euclid`, 12-music s3.5):
    k onsets spread as evenly as possible over n steps. E(3,8) = x--x--x-, E(5,8) = x-xx-xx-.
    `rotate` > 0 moves the pattern later by that many steps."""
    n = max(1, int(n))
    k = max(0, min(int(k), n))
    if k == 0:
        return [False] * n
    a, b = [[True] for _ in range(k)], [[False] for _ in range(n - k)]
    while len(b) > 1:
        m = min(len(a), len(b))
        a, b = [a[i] + b[i] for i in range(m)], (a[m:] or b[m:])
    pat = [x for g in a + b for x in g]
    r = int(rotate) % n
    return pat[-r:] + pat[:-r] if r else pat


def steps_str(pat) -> str:
    return "".join("x" if p else "-" for p in pat)


CLAVES = {                                    # 16 steps per bar; tresillo/cinquillo are 8
    "son_3_2": "x--x--x---x-x---", "son_2_3": "--x-x---x--x--x-",
    "rumba_3_2": "x--x---x--x-x---", "rumba_2_3": "--x-x---x--x---x",
    "bossa": "x--x--x---x--x--", "tresillo": "x--x--x-", "cinquillo": "x-xx-xx-",
    "bell_6_8": "x-x-xx-x-x-x",
}
_LEVEL = {"x": 0.8, "X": 1.0, "o": 0.4, "?": 0.8}     # hit, accent, ghost, half-chance hit


def mask_hits(mask: str, rng=None, gate_steps: dict | None = None) -> list[tuple[int, float]]:
    """Step string -> [(step, velocity)]. x hit, X accent, o ghost, ? = a hit with probability 1/2
    (Farnell's `random 2 -> sel 1` gate, g15 s1.3); `gate_steps` {step: probability} gates more."""
    out = []
    for i, ch in enumerate(c for c in str(mask) if c not in " |"):
        if ch not in _LEVEL:
            continue
        p = 0.5 if ch == "?" else 1.0
        p *= float((gate_steps or {}).get(str(i), (gate_steps or {}).get(i, 1.0)))
        if p >= 1.0 or (rng is not None and rng.random() < p):
            out.append((i, _LEVEL[ch]))
    return out


def voss(n: int, rng, generators: int = 4) -> np.ndarray:
    """1/f ("pink") sequence in 0..1, Voss-McCartney: generator g is re-rolled every 2^g steps and
    the generators are summed (MIT 21M.380: gamma about 1 for melodic walks, 12-music s3.5)."""
    vals = rng.random(generators)
    out = np.empty(n)
    for i in range(n):
        for g in range(generators):
            if i % (1 << g) == 0:
                vals[g] = rng.random()
        out[i] = vals.sum() / generators
    return out


# ============================================================================ theory: motifs
def transpose(motif, steps: int):
    """Motif = [(scale-step offset, beats), ...]. Diatonic transposition."""
    return [(o + steps, d) for o, d in motif]


def invert(motif):
    """Mirror the contour about the first note."""
    return [(2 * motif[0][0] - o, d) for o, d in motif]


def retrograde(motif):
    return list(reversed(motif))


def augment(motif, factor: float = 2.0):
    """Stretch the rhythm (factor 0.5 = diminution)."""
    return [(o, d * factor) for o, d in motif]


# ============================================================================ styles
def _st(bpm, scale, progs, kit=None, bass=None, chords=None, lead=None, pad=None, arp=None, counter=None,
        swing=0.0, meter=4, grid=0.25, ext=3, dens=0.5, delay=(), dist=0.0):
    return NS(bpm=bpm, scale=scale, progs=progs, kit=kit, bass=bass, chords=chords, lead=lead, pad=pad, arp=arp,
              counter=counter, swing=swing, meter=meter, grid=grid, ext=ext, dens=dens, delay=delay, dist=dist)


_K4, _BB, _H8, _H16 = "x---x---x---x---", "----x-------x---", "x-x-x-x-x-x-x-x-", "xoxoxoxoxoxoxoxo"
_CR = "X" + "-" * 63        # a pattern longer than a bar runs across bars: once every 4 bars
_T = 1 / 3

# The style palette of AGENTS.md "Music" / examples/library/styles.json as data: tempo range, scale,
# progressions (roman numerals, one chord per bar), drum grid (x X o ?), bass (inst, pattern),
# harmony (inst, comping), lead / pad / arp / counter instruments, swing, meter and grid.
STYLES = {
    "lofi": _st((70, 85), "dorian", ["i7 iv7 VII7 III7", "ii7 v7 i7 i7"],
                {"kick": "x-----x---x-----", "snap": _BB, "brush": "----o-------o---", "hihat": _H8},
                ("upright_bass", "root_fifth"), ("epiano", "sustain"), "felt_piano", swing=0.18, ext=4, dens=0.4),
    "jazz_swing": _st((120, 180), "major", ["ii7 V7 I7 vi7", "I7 vi7 ii7 V7"],
                      {"ride": "x--x-xx--x-x", "hihat": "---x-----x--", "brush": "x--o--x--o--"},
                      ("upright_bass", "walking"), ("piano", "stabs"), "trumpet", counter="sax", grid=_T, ext=4),
    "bossa_nova": _st((120, 140), "major", ["I7 vi7 ii7 V7", "I7 ii7 V7 I7"],
                      {"rim": CLAVES["bossa"], "shaker": _H16, "kick": "x--x----x--x----"},
                      ("upright_bass", "root_fifth"), ("guitar", "synco"), "flute", ext=4),
    "blues_shuffle": _st((90, 120), "mixolydian", ["I I I I IV IV I I V IV I V"],
                         {"kick": "x-----x-----", "snare": "---x-----x--", "hihat": "x-xx-xx-xx-x"},
                         ("finger_bass", "walking"), ("electric_guitar", "eighths"), "harmonica", grid=_T, ext=2),
    "soul_ballad": _st((60, 80), "major", ["I vi IV V", "I iii IV V"],
                       {"kick": "x-------x-x-----", "snap": _BB, "hihat": _H8},
                       ("finger_bass", "root_fifth"), ("wurli", "half"), "sax", pad="strings", ext=4),
    "funk": _st((95, 115), "dorian", ["i7 i7 IV7 i7", "i7 IV7"],
                {"kick": "X--x------X--x--", "snare": "----X--o-o--X--o", "hihat": "x-x-x-x-x-x-x-xx", "openhat": "--------------x-"},
                ("slap_bass", "syncopated"), ("clav", "synco"), "brass", ext=4),
    "disco": _st((115, 125), "minor", ["i VI III VII", "i iv VII III"],
                 {"kick": _K4, "clap": _BB, "openhat": "--x---x---x---x-", "hihat": "xx-xxx-xxx-xxx-x"},
                 ("finger_bass", "octave"), ("string_machine", "sustain"), "violin"),
    "reggae": _st((70, 80), "major", ["I IV", "I V vi IV"],
                  {"kick": "--------x-------", "rim": "--------x-------", "hihat": _H8, "shaker": _H16},
                  ("finger_bass", "syncopated"), ("organ", "offbeat"), "trombone", swing=0.2),
    "latin_salsa": _st((90, 100), "harmonic_minor", ["i iv V i", "i VI V V"],
                       {"clave": CLAVES["son_2_3"], "conga": "--o---xx--o---xx", "cowbell": "x---x-x-x---x-x-",
                        "timbale": "------x-------x-", "guiro": "x-xxx-xxx-xxx-xx"},
                       ("upright_bass", "tumbao"), ("piano", "montuno"), "trumpet"),
    "reggaeton": _st((90, 98), "minor", ["i VI III VII", "i iv VI V"],
                     {"kick": _K4, "snare808": "---x--x----x--x-", "hihat": _H8},
                     ("bass808", "long"), ("warm_pad", "sustain"), "marimba"),
    "rock": _st((110, 150), "mixolydian", ["I VII IV I", "I IV V IV"],
                {"kick": "x-----x-x-----x-", "snare": _BB, "hihat": _H8, "crash": _CR},
                ("finger_bass", "eighths"), ("electric_guitar", "eighths"), "dist_guitar", ext=2, dist=8.0),
    "metal": _st((150, 190), "phrygian", ["i II i VII", "i VI VII i"],
                 {"kick": "xxxxxxxxxxxxxxxx", "snare": "--------x-------", "hihat": _H8, "crash": _CR},
                 ("finger_bass", "sixteenths"), ("electric_guitar", "gallop"), None, ext=2, dist=16.0),
    "pop": _st((100, 120), "major", ["I V vi IV", "vi IV I V"],
               {"kick": "x-----x-x-------", "clap": _BB, "hihat": _H8},
               ("finger_bass", "eighths"), ("piano", "half"), "soft_lead", arp="synth_pluck"),
    "synthwave": _st((90, 110), "minor", ["i VI III VII", "i VII VI VII"],
                     {"kick": _K4, "snare808": _BB, "hihat": _H8},
                     ("bass", "eighths"), ("string_machine", "sustain"), "lead", arp="synth_pluck", delay=("arp",)),
    "house": _st((120, 128), "dorian", ["i VII VI VII", "i iv"],
                 {"kick": _K4, "clap": _BB, "openhat": "--x---x---x---x-", "shaker": _H16},
                 ("fm_bass", "offbeat"), ("piano", "synco"), None, pad="warm_pad"),
    "acid_techno": _st((128, 136), "phrygian", ["i", "i i i VII"],
                       {"kick": _K4, "hihat": "--x---x---x---x-", "clap": _BB, "rim": "---x--x---?x--x-"},
                       ("acid", "acid"), None, None),
    "drum_and_bass": _st((165, 175), "minor", ["i VI", "i III VII VI"],
                         {"kick": "x---------x-----", "snare": "----x--o-o--x---", "hihat": _H8, "shaker": _H16},
                         ("reese", "long"), ("glass_pad", "sustain"), "bell", delay=("lead",), dens=0.3),
    "trap": _st((130, 150), "minor", ["i VI", "i iv"],
                {"kick": "x------x--x-----", "clap": "--------x-------", "hihat": "x-x-x-x-x-x-xxxx"},
                ("bass808", "long"), ("warm_pad", "sustain"), "bell", dens=0.3),
    "chiptune": _st((130, 160), "major", ["I IV V I", "I V vi IV"],
                    {"kick": "x-----x-x---x---", "snare": _BB, "hihat": _H8},
                    ("chiptri", "octave"), None, "chip", arp="chip"),
    "orchestral_heroic": _st((90, 120), "mixolydian", ["I VII IV I", "I v IV I"],
                             {"taiko": "x-----x-x-------", "crash": _CR},
                             ("cello", "long"), ("strings", "sustain"), "french_horn", counter="trumpet"),
    "baroque": _st((90, 110), "harmonic_minor", ["i iv VII III VI ii V i"], None,
                   ("cello", "walking"), ("harpsichord", "broken"), "oboe", counter="violin"),
    "ragtime": _st((90, 105), "major", ["I vi ii V", "I I V V"], None,
                   ("piano", "root_fifth"), ("piano", "backbeat"), "piano", dens=0.7),
    "lullaby": _st((70, 90), "major", ["I IV I V"], None, None, ("celesta", "broken"), "music_box",
                   pad="glass_pad", meter=3, grid=0.5, dens=0.4),
    "spooky": _st((80, 100), "octatonic", ["i ii i iii"], None, None, ("church_organ", "sustain"), "theremin",
                  arp="celesta", dens=0.35),
    "christmas": _st((100, 120), "major", ["I IV V I", "I vi IV V"], {"sleigh": _H8},
                     ("upright_bass", "root_fifth"), ("strings", "sustain"), "glockenspiel", counter="french_horn"),
    "celtic_jig": _st((110, 125), "dorian", ["i VII i VII", "i i VII i"], {"djembe": "x-ox-o"},
                      None, ("harp", "half"), "violin", counter="whistle", meter=2, grid=_T, dens=0.9),
    "bluegrass": _st((110, 130), "major", ["I IV I V", "I I IV V"], None,
                     ("upright_bass", "root_fifth"), ("steel_guitar", "backbeat"), "violin", arp="banjo"),
    "polka_oompah": _st((110, 125), "major", ["I I V V V V I I"],
                        {"kick": "x-------x-------", "snare": _BB, "crash": _CR},
                        ("tuba", "root_fifth"), ("accordion", "backbeat"), "clarinet"),
    "tango": _st((110, 125), "harmonic_minor", ["i iv V i", "i VI V V"], None,
                 ("upright_bass", "root_fifth"), ("accordion", "eighths"), "violin"),
    "andean": _st((90, 110), "minor", ["i III VII i", "i VII VI i"],
                  {"taiko": "x-----x-x-----x-", "shaker": _H8},
                  ("guitar", "root_fifth"), ("mandolin", "eighths"), "pan_flute"),
    "middle_eastern": _st((90, 105), "hijaz", ["i II i vii"],
                          {"djembe:stroke=bass": "x-------x-------", "djembe:stroke=slap": "--x---x-----x---", "tambourine": _H8},
                          None, ("strings", "sustain"), "oboe", arp="pluck", ext=2),
    "indian_raga": _st((80, 100), "mixolydian", ["I"],
                       {"tabla:stroke=ge": "x-----x---x-----", "tabla:stroke=na": "--x-x---x---x-x-", "tabla:stroke=ke": "---x-x---x-x---x"},
                       None, ("sitar", "sustain"), "sitar", ext=2),
    "japanese": _st((70, 90), "in", ["i iv i v"], {"taiko": "x---------x-----", "woodblock": "------x-------?-"},
                    None, None, "pan_flute", arp="koto", dens=0.3),
    "gamelan": _st((90, 100), "pelog", ["i iii i iv"], {"gong": _CR},
                   ("handpan", "long"), None, "gamelan", arp="gamelan"),
    "ambient": _st((50, 70), "lydian", ["I II", "I iii"], None, None, ("warm_pad", "sustain"), "handpan",
                   pad="glass_pad", dens=0.2),
}

# drum balance inside one kit: the "Mix by gain" priors of AGENTS.md / styles.json
DRUM_GAIN = {"kick": 1.0, "kick808": 1.0, "snare": 0.7, "snare808": 0.7, "clap": 0.6, "snap": 0.6, "rim": 0.45,
             "hihat": 0.25, "openhat": 0.25, "shaker": 0.18, "ride": 0.3, "crash": 0.4, "brush": 0.3, "tom": 0.5,
             "taiko": 0.6, "clave": 0.4, "conga": 0.4, "cowbell": 0.3, "timbale": 0.3, "guiro": 0.15,
             "tambourine": 0.2, "sleigh": 0.2, "djembe": 0.6, "tabla": 0.45, "woodblock": 0.25, "gong": 0.6}
_HATS = {"hihat", "shaker", "ride", "tambourine", "guiro", "sleigh", "brush"}
_ROLES = {"melody": ((60, 84), "soft_lead"), "bass": ((28, 55), "finger_bass"), "chords": ((48, 72), "piano"),
          "pad": ((48, 72), "warm_pad"), "arp": ((60, 84), "synth_pluck"), "counter": ((55, 79), "strings"),
          "drums": ((0, 127), None)}
# rhythm tables for a 4-beat bar: (beat, duration in beats[, tone]); scaled and snapped to the style grid
_BASS = {
    "root_fifth": [(0, 1.5, "R"), (2, 1.5, "5")],
    "octave": [(i * 0.5, 0.4, "R8"[i % 2]) for i in range(8)],
    "eighths": [(i * 0.5, 0.45, "R") for i in range(8)],
    "sixteenths": [(i * 0.25, 0.2, "R") for i in range(16)],
    "offbeat": [(i + 0.5, 0.4, "R") for i in range(4)],
    "long": [(0, 3.8, "R")],
    "tumbao": [(1.5, 1.0, "5"), (3.0, 1.0, "N")],            # anticipates the next chord's root on beat 4
}
_COMP = {
    "sustain": [(0, 4.0)], "half": [(0, 1.9), (2, 1.9)], "backbeat": [(1, 0.5), (3, 0.5)],
    "offbeat": [(i + 0.5, 0.3) for i in range(4)], "eighths": [(i * 0.5, 0.45) for i in range(8)],
    "synco": [(0, 0.4), (0.75, 0.4), (1.5, 0.4), (2.5, 0.4), (3.25, 0.4)],
    "montuno": [(b, 0.25) for b in (0, 0.5, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.5)],
    "gallop": [(b + o, 0.2) for b in range(4) for o in (0, 0.5, 0.75)],
}


def _style(name, required=False):
    if not name:
        if required:
            raise SpecError(f"a style is needed; one of {sorted(STYLES)}")
        return None
    if name not in STYLES:
        raise SpecError(f"unknown style {name!r}; one of {sorted(STYLES)}")
    return STYLES[name]


def _rng(seed, *tags):
    return np.random.default_rng([int(seed) & 0x7FFFFFFF] + [zlib.crc32(str(t).encode()) for t in tags])


def style_bpm(style: str, seed: int = 0) -> int:
    """The tempo an `arrangement` uses when no `bpm` is set: a seeded pick in the style's range."""
    lo, hi = _style(style, True).bpm
    return int(lo + _rng(seed, "bpm").integers(0, hi - lo + 1))


# ============================================================================ context: key, form, harmony
_RANGE = re.compile(r"^\s*([A-Ga-g][#b]?-?\d+)\s*-\s*([A-Ga-g][#b]?-?\d+)\s*$")


def _context(layer: dict, bpm=None) -> NS:
    role = {"lead": "melody"}.get(layer.get("role", "melody"), layer.get("role", "melody"))
    if role not in _ROLES:
        raise SpecError(f"compose role {role!r}: one of {sorted(_ROLES)} (or lead)")
    st = _style(layer.get("style"))
    bpm = layer.get("bpm") or bpm
    c = NS(role=role, style=st, seed=int(layer.get("seed", 0)), q=60.0 / float(bpm) if bpm else 1.0,
           meter=float(layer.get("meter", st.meter if st else 4)), grid=float(layer.get("grid", st.grid if st else 0.25)),
           bars=int(layer.get("bars", 4)), kpc=_key_pc(layer.get("key", "C")),
           density=float(np.clip(layer.get("density", st.dens if st else 0.5), 0.0, 1.0)))
    if c.bars < 1 or c.grid <= 0 or c.meter < c.grid:
        raise SpecError("compose needs bars >= 1 and 0 < grid <= meter")
    c.S = max(1, int(round(c.meter / c.grid)))                   # steps per bar
    c.spb = max(1, int(round(1.0 / c.grid)))                     # steps per beat
    c.offs = scale_offsets(layer.get("scale") or (st.scale if st else "major"), layer.get("tuning", "equal"))
    c.ext = int(layer.get("chord_size", st.ext if st else 3))
    rg = layer.get("range")
    if rg:
        m = _RANGE.match(str(rg))
        if not m:
            raise SpecError(f'bad range {rg!r}: write it like "C4-C6"')
        c.lo, c.hi = sorted(float(N.note_to_midi(g)) for g in m.groups())
    else:
        c.lo, c.hi = _ROLES[role][0]
    # form: "A A B A", optional intro / outro; equal bars per section, the remainder goes to the last
    names = layer.get("sections", "A")
    names = names.split() if isinstance(names, str) else list(names)
    if not names or len(names) > c.bars:
        raise SpecError("sections: need at least one, and at least one bar each")
    first = next((n for n in names if n not in ("intro", "outro")), "A")
    per, c.sections, b0 = c.bars // len(names), [], 0
    for i, n in enumerate(names):
        nb = per if i < len(names) - 1 else c.bars - b0
        c.sections.append(NS(name=n, mat=first if n in ("intro", "outro") else n, bar0=b0, nbars=nb, index=i))
        b0 += nb
    _harmonise(c, layer)
    # intensity curve (vertical layering): one value per section, or any list spread over the bars
    inten = layer.get("intensity")
    if inten is None:
        c.intensity = [0.35 if s.name in ("intro", "outro") else 1.0 for s in c.sections for _ in range(s.nbars)]
    else:
        inten = [float(v) for v in (inten if isinstance(inten, (list, tuple)) else [inten])]
        if len(inten) == len(c.sections):
            c.intensity = [inten[s.index] for s in c.sections for _ in range(s.nbars)]
        else:
            c.intensity = [inten[min(len(inten) - 1, b * len(inten) // c.bars)] for b in range(c.bars)]
    return c


def _harmonise(c: NS, layer: dict) -> None:
    """One chord per bar for the whole piece. Depends only on seed/key/scale/style/form, never on the
    role, so every part of an arrangement gets the same harmony."""
    prog, hr = layer.get("progression", "auto"), layer.get("harmonic_rhythm")
    explicit = layer.get("chords")
    if prog == "pokesdown":
        explicit, hr = POKESDOWN, hr or 8
    hr = max(1, int(hr or 1))
    mats = list(dict.fromkeys(s.mat for s in c.sections))
    table = {}
    if explicit:
        chords = []
        for ch in explicit:
            mid = [69 + 12 * math.log2(f / 440.0) for f in N.parse_pitch_list(ch)]
            pcs = list(dict.fromkeys(round((m - c.kpc) % 12, 6) for m in mid))
            chords.append({"roman": "", "root": pcs[0], "pcs": pcs, "slots": [round(m, 6) for m in mid]})
        table = {m: chords for m in mats}
    else:
        pick = int(_rng(c.seed, "harmony").integers(0, 1 << 16))
        for i, m in enumerate(mats):
            text = prog.get(m, prog.get(mats[0], "auto")) if isinstance(prog, dict) else prog
            if text == "auto":
                if c.style:
                    text = c.style.progs[(pick + i) % len(c.style.progs)]
                else:
                    text = functional_progression(_rng(c.seed, "harmony", m), 4)
            table[m] = parse_progression(text, c.offs, c.ext)
    c.chords = []
    for s in c.sections:
        s.chords = [table[s.mat][(b // hr) % len(table[s.mat])] for b in range(s.nbars)]
        if s.name == "outro" and not explicit:
            s.chords[-1] = parse_progression("I", c.offs, c.ext)[0]      # the outro comes home
        c.chords += s.chords


def harmony(layer: dict, bpm=None) -> list[dict]:
    """The chord of every bar: [{"bar", "roman", "root", "pcs"}], root and pcs as pitch classes 0..12
    (C = 0; fractional in non-12-TET tunings). Identical for every part that shares seed/key/scale/style/form."""
    if layer.get("type") == "arrangement":
        layer = arrange(layer, bpm)["layers"][0]
    c = _context(layer, bpm)
    return [{"bar": b, "roman": ch["roman"], "root": round((c.kpc + ch["root"]) % 12, 6),
             "pcs": sorted(round((c.kpc + p) % 12, 6) for p in ch["pcs"])} for b, ch in enumerate(c.chords)]


def _weight(c, s):
    """Metric weight of a step. # UNSOURCED: bar start 1, beats 0.8, half beats 0.5, the rest 0.12."""
    if s == 0:
        return 1.0
    if s % c.spb == 0:
        return 0.8
    if (c.spb % 2 == 0 and s % c.spb == c.spb // 2) or (c.spb == 3 and s % 3 == 2):
        return 0.5
    return 0.12


def _strong(c, s):
    return s % (c.S // 2 if (c.S % 2 == 0 and c.meter != 3) else c.S) == 0


def _contour(n, rng, lo=0.75, hi=1.0, period=14):
    """Velocity swell: a sine over 14 events (MIT 21M.380 p.54) plus a Beta(0.3, 0.3) spread
    (after procedural/music.py `_contour`)."""
    c = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / period + rng.uniform(0, 2 * np.pi))
    return lo + (hi - lo) * (0.75 * c + 0.25 * rng.beta(0.3, 0.3, n))


def _ev(t, dur, pitch, vel, **kw):
    return {"t": t, "dur": dur, "pitch": pitch, "vel": vel, **kw}


def _tones(c, ch, lo=None, hi=None):
    return _pitches(c.kpc, ch["pcs"], c.lo if lo is None else lo, c.hi if hi is None else hi)


def _reflect(j, n):
    while j < 0 or j >= n:
        j = -j if j < 0 else 2 * (n - 1) - j
    return j


# ============================================================================ generators: melody
def _parse_motif(c, motif, default_dur):
    """"D4 F4 G4" | [0, 2, 3] (scale steps) | [[0, 0.5], [2, 1]] -> [(scale-step offset, beats)]."""
    allp = _pitches(c.kpc, c.offs, 0, 127)
    if isinstance(motif, str):
        idx = [allp.index(_near(allp, 69 + 12 * math.log2(f / 440.0))) for f in N.parse_pitch_list(motif)]
        return [(i - idx[0], default_dur) for i in idx]
    return [(int(m[0]), float(m[1])) if isinstance(m, (list, tuple)) else (int(m), default_dur) for m in motif]


def _bar_rhythm(c, rng, dens):
    return [s for s in range(c.S) if rng.random() < min(1.0, _weight(c, s) * 2 * dens)] or [0]


def _gen_melody(c, sec, rng, L):
    """Constrained random walk over the scale (the Zen-game rules of g15 s1.9: note length, phrase
    length then rest, chord-tone probability, small-interval probability, register tendency), or a
    Markov / 1/f / Collatz pitch source under the same constraints: chord tones on the strong beats,
    an arch contour per phrase, every phrase cadencing on a chord tone and the section on the root."""
    mode = L.get("mode", "walk")
    if mode in ("motif", "call_response"):
        return _gen_motif(c, sec, rng, L, mode)
    P = _pitches(c.kpc, c.offs, c.lo, c.hi)
    if len(P) < 3:
        raise SpecError("range too narrow for the scale")
    ph = max(1, min(int(L.get("phrase", 2)), sec.nbars))
    snap = bool(L.get("snap", True))
    p_near, p_chord = float(L.get("p_near", 0.75)), float(L.get("p_chord", 0.3))     # UNSOURCED: Zen rules give no numbers
    gate = float(L.get("gate", 0.92))
    shape = {"arch": lambda x: 0.3 + 0.5 * math.sin(math.pi * x), "rise": lambda x: 0.2 + 0.6 * x,
             "fall": lambda x: 0.8 - 0.6 * x, "flat": lambda x: 0.5}.get(L.get("contour", "arch"))
    if shape is None:
        raise SpecError("contour: arch|rise|fall|flat")
    base = [_bar_rhythm(c, rng, c.density) for _ in range(ph)]
    onsets = []
    for b in range(sec.nbars):
        k, pb = divmod(b, ph)
        if pb == 0:                                     # odd phrases keep the first bar's rhythm and vary the last
            phr = [list(r) for r in base]
            if k % 2 and ph > 1:
                phr[-1] = _bar_rhythm(c, rng, c.density)
        steps = phr[pb]
        if pb == ph - 1 or b == sec.nbars - 1:          # breathe: rest on the last beat of a phrase
            steps = [s for s in steps if s < c.S - c.spb] or [0]
        onsets += [(b, s) for s in steps]
    n = len(onsets)
    # pitch sources ---------------------------------------------------------------
    mid = P.index(_near(P, (c.lo + c.hi) / 2))
    if mode == "markov":
        order = int(np.clip(L.get("order", 1), 1, 2))
        mot = [o for o, _ in (_parse_motif(c, L["motif"], 0.5) if L.get("motif") is not None
                              else [(int(v), 0.5) for v in np.cumsum(rng.integers(-2, 3, 8))])]
        table = {}
        for i in range(len(mot)):                       # trained cyclically so no context is a dead end
            table.setdefault(tuple(mot[(i + j) % len(mot)] for j in range(order)), []).append(mot[(i + order) % len(mot)])
        seq = list(mot[:order])
        while len(seq) < n:
            seq.append(int(rng.choice(table[tuple(seq[-order:])])))
        raw = [P[_reflect(mid + o, len(P))] for o in seq]
    elif mode == "voss":
        raw = [P[int(round(v * (len(P) - 1)))] for v in voss(n, rng)]
    elif mode == "collatz":                             # hailstone numbers coerced to the range and scale
        a0 = a = max(2, int(L.get("start", 7 + c.seed % 90)))
        raw = []
        for _ in range(n):
            raw.append(P[a % len(P)])
            a = a // 2 if a % 2 == 0 else 3 * a + 1
            if a == 1:
                a0 += 1
                a = a0
    elif mode == "walk":
        raw = None
    else:
        raise SpecError("melody mode: walk|markov|voss|collatz|motif|call_response")
    vel = _contour(n, rng)
    out, cur = [], None
    for i, (b, s) in enumerate(onsets):
        ch = sec.chords[b]
        T = _tones(c, ch) or P
        here = b * c.S + s
        nxt = onsets[i + 1][0] * c.S + onsets[i + 1][1] if i + 1 < n else sec.nbars * c.S
        phrase_end = i + 1 == n or onsets[i + 1][0] // ph != b // ph
        target = c.lo + (c.hi - c.lo) * shape(((b % ph) * c.S + s) / (ph * c.S))
        if raw is not None:
            prop = raw[i]
        elif cur is None:
            prop = _near(P, target)
        else:
            j = P.index(cur)
            toward = 1 if target > cur else -1
            d = toward if rng.random() < 0.7 else -toward
            prop = P[_reflect(j + d * (1 if rng.random() < p_near else int(rng.integers(2, 5))), len(P))]
        if i + 1 == n and sec.index == len(c.sections) - 1:
            prop = _near([p for p in T if abs((p - c.kpc - ch["root"]) % 12) < 1e-6] or T, prop)   # home: the root
        elif phrase_end or (snap and _strong(c, s)) or (raw is None and rng.random() < p_chord):
            prop = _near([p for p in T if p != cur] or T, prop)       # a chord tone, but keep the line moving
        cur = prop
        dur = min((nxt - here) * c.grid, c.meter if phrase_end else 2.0) * gate
        out.append(_ev(sec.bar0 * c.meter + here * c.grid, dur, cur, float(vel[i]) * (0.75 + 0.25 * _weight(c, s))))
    return out


def _gen_motif(c, sec, rng, L, mode):
    """Motif development: a seed motif restated per unit with transpose / invert / retrograde /
    augment (operators after procedural/music.py `_motif_op`), or call and response (the answer is
    the call moved a step and brought home to the chord root)."""
    P = _pitches(c.kpc, c.offs, c.lo, c.hi)
    unit = 1 if mode == "call_response" else max(1, min(int(L.get("phrase", 2)), sec.nbars))
    if L.get("motif") is not None:
        motif = _parse_motif(c, L["motif"], 2 * c.grid)
    else:
        steps = _bar_rhythm(c, rng, max(c.density, 0.4))
        steps = [s for s in steps if s < c.S - c.spb] or [0]
        offs = np.concatenate([[0], np.cumsum(rng.choice([-2, -1, 1, 2], len(steps) - 1))]) if len(steps) > 1 else [0]
        motif = [(int(o), (b - a) * c.grid) for o, a, b in zip(offs, steps, steps[1:] + [c.S - c.spb])]
    ops = L.get("develop") or (["state", "answer"] if mode == "call_response" else ["state", "transpose", "invert", "retrograde", "augment"])
    out = []
    for u in range(0, sec.nbars, unit):
        op = ops[(u // unit) % len(ops)]
        m = {"state": motif, "transpose": transpose(motif, int(rng.choice([-2, -1, 1, 2]))), "invert": invert(motif),
             "retrograde": retrograde(motif), "augment": augment(motif, 2.0), "diminish": augment(motif, 0.5),
             "answer": transpose(motif, int(rng.choice([-2, -1, 1])))}.get(op)
        if m is None:
            raise SpecError(f"develop op {op!r}: state|transpose|invert|retrograde|augment|diminish|answer")
        ch0 = sec.chords[u]
        base = P.index(_near(P, _near(_tones(c, ch0) or P, (c.lo + c.hi) / 2)))
        t, end = 0.0, min(unit, sec.nbars - u) * c.meter
        for k, (o, d) in enumerate(m):
            if t >= end - 1e-9:
                break
            b = min(sec.nbars - 1, u + int(t // c.meter))
            ch = sec.chords[b]
            p = P[_reflect(base + o - m[0][0] if op != "transpose" and op != "answer" else base + o, len(P))]
            last = k == len(m) - 1 or t + d >= end - 1e-9
            if last:
                T = _tones(c, ch) or P
                if op == "answer" or u + unit >= sec.nbars:
                    T = [x for x in T if abs((x - c.kpc - ch["root"]) % 12) < 1e-6] or T
                p = _near(T, p)
            elif L.get("snap") and _strong(c, int(round((t % c.meter) / c.grid))):
                p = _near(_tones(c, ch) or P, p)
            out.append(_ev((sec.bar0 + u) * c.meter + t, min(d, end - t) * 0.92, p, 0.85 if k else 0.95))
            t += d
    return out


def _gen_counter(c, sec, rng, L):
    """Counter-line: slow guide tones (the chord's third or top note) entering a beat late, so it
    sits in the gaps of the lead (Phillips: layers "jump into the gaps", 12-music s2.1)."""
    out, cur = [], (c.lo + c.hi) / 2
    for b, ch in enumerate(sec.chords):
        guide = [ch["pcs"][1 % len(ch["pcs"])], ch["pcs"][-1]]
        cur = _near(_pitches(c.kpc, guide, c.lo, c.hi) or _tones(c, ch), cur)
        off = min(1.0, c.meter / 2)
        out.append(_ev((sec.bar0 + b) * c.meter + off, (c.meter - off) * 0.95, cur, 0.7))
    return out


# ============================================================================ generators: bass
def _root(c, ch, x):
    return _near(_pitches(c.kpc, [ch["root"]], c.lo, c.hi) or _pitches(c.kpc, [ch["root"]], c.lo - 12, c.hi + 12), x)


def _snap_table(c, table):
    """Scale a 4-beat table to the meter and snap positions to the grid (on a triplet grid an
    off-beat eighth lands on the last triplet: the shuffle comes for free)."""
    out, seen = [], set()
    for row in table:
        pos = math.floor(row[0] * c.meter / 4 / c.grid + 0.5) * c.grid
        if pos < c.meter - 1e-9 and round(pos, 6) not in seen:
            seen.add(round(pos, 6))
            out.append((pos, row[1] * c.meter / 4) + tuple(row[2:]))
    return out


def _gen_bass(c, sec, rng, L):
    pat = L.get("pattern") or (c.style.bass[1] if c.style and c.style.bass else "root_fifth")
    anchor = c.lo + 8
    out = []
    nxt_ch = lambda b: sec.chords[(b + 1) % sec.nbars]                           # noqa: E731

    def tone(ch, nx, t, ref):
        r = _root(c, ch, ref)
        if t == "R":
            return r
        if t == "N":
            return _root(c, nx, r)
        if t == "8":
            return r + 12 if r + 12 <= c.hi else r
        pc = ch["pcs"][{"5": 2, "3": 1, "7": -1}[t] % len(ch["pcs"])]
        above = [p for p in _pitches(c.kpc, [pc], c.lo, c.hi) if p > r]
        return above[0] if above and above[0] - r < 12 else _near(_pitches(c.kpc, [pc], c.lo, c.hi) or [r], r)

    if pat == "walking":
        # quarter notes: chord root on beat 1, chord and scale tones between, an approach tone (a half
        # step from the next root: chromatic, flagged "approach") on the last beat
        P = _pitches(c.kpc, c.offs, c.lo, c.hi)
        p_app = float(L.get("approach", 0.5))
        m, prev = max(1, int(round(c.meter))), anchor
        for b, ch in enumerate(sec.chords):
            r = _root(c, ch, prev)
            nr = _root(c, nxt_ch(b), r)
            line = [r]
            ap, chrom = None, False
            if m >= 2:
                if rng.random() < p_app:
                    ap = nr + (-1 if (rng.random() < 0.6 and nr - 1 >= c.lo) or nr + 1 > c.hi else 1)
                    chrom = not any(abs(ap - p) < 1e-6 for p in P)
                else:
                    ap = _near([p for p in P if abs(p - nr) > 1e-6], nr + (0.1 if r > nr else -0.1))
            bend = (4.0 if rng.random() < 0.5 else -4.0) if ap is not None and abs(ap - r) < 3 else 0.0
            for j in range(1, m - 1):
                ideal = r + (ap - r) * j / (m - 1) + bend * math.sin(math.pi * j / (m - 1))
                pool = (_tones(c, ch) if j % 2 else P) or P
                line.append(_near([p for p in pool if abs(p - line[-1]) > 1e-6] or pool, ideal))
            if ap is not None:
                line.append(ap)
            for j, p in enumerate(line):
                out.append(_ev((sec.bar0 + b) * c.meter + j, 0.9, p, 0.9 if j == 0 else 0.75,
                               **({"approach": True} if chrom and j == m - 1 else {})))
            prev = line[-1]
    elif pat in ("syncopated", "acid"):
        S = c.S
        if pat == "syncopated":
            # funk line: a Euclidean mask with "the one" always on the root, ghost notes between
            mask = euclid(3 + int(round(c.density * 5)), S, int(rng.integers(S)))
            mask[0] = True
            cells = [None] * S
            for s in range(S):
                if mask[s]:
                    ghost = s and rng.random() < 0.25
                    cells[s] = (str(rng.choice(["R", "R", "R", "8", "5", "7"])) if s else "R",
                                0.5 if ghost else float(rng.integers(1, 3)), 0.5 if ghost else (1.0 if s == 0 else 0.8), False)
        else:
            # 303 line (Farnell MEX note tuple, g15 s1.8): note, duration in steps, slide, accent
            cells = [None] * S
            for s in range(S):
                if s == 0 or rng.random() < 0.45 + 0.5 * c.density:
                    slide = rng.random() < 0.2
                    cells[s] = (str(rng.choice(["R", "R", "R", "8", "8", "3", "7"])) if s else "R",
                                1.0 if slide else 0.5, 1.0 if rng.random() < 0.3 else 0.65, bool(slide))
        for b, ch in enumerate(sec.chords):
            for s, cell in enumerate(cells):
                if cell is None:
                    continue
                t, d, v, slide = cell
                p = tone(ch, nxt_ch(b), t, anchor)
                e = _ev((sec.bar0 + b) * c.meter + s * c.grid, d * c.grid, p, v)
                if pat == "acid":
                    e.update(slide=slide, accent=v >= 1.0, tuple303=[round(p, 3), 2 if slide else 1, int(slide), v])
                out.append(e)
    elif pat in _BASS:
        for b, ch in enumerate(sec.chords):
            for pos, d, t in _snap_table(c, _BASS[pat]):
                out.append(_ev((sec.bar0 + b) * c.meter + pos, d, tone(ch, nxt_ch(b), t, anchor), 0.9 if pos == 0 else 0.78))
    else:
        raise SpecError(f"bass pattern {pat!r}: one of {sorted(_BASS) + ['acid', 'syncopated', 'walking']}")
    return out


# ============================================================================ generators: harmony
def _gen_chords(c, sec, rng, L, pad=False):
    comp = "sustain" if pad else (L.get("pattern") or (c.style.chords[1] if c.style and c.style.chords else "sustain"))
    if comp not in _COMP and comp not in ("stabs", "broken"):
        raise SpecError(f"chords pattern {comp!r}: one of {sorted(_COMP) + ['broken', 'stabs']}")
    out, prev = [], None
    for b, ch in enumerate(sec.chords):
        v = voice_lead(prev, ch["pcs"], c.kpc, c.lo, c.hi)
        prev, t0 = v, (sec.bar0 + b) * c.meter
        if comp == "broken":                            # Alberti-style broken chord on half beats
            order = [0, len(v) - 1, len(v) // 2, len(v) - 1]
            for i in range(int(round(c.meter / (2 * c.grid) if c.spb % 2 == 0 else c.meter / c.grid))):
                step = 2 * c.grid if c.spb % 2 == 0 else c.grid
                out.append(_ev(t0 + i * step, step * 0.95, v[order[i % 4]], 0.8 if i % 4 == 0 else 0.65))
            continue
        if comp == "stabs":                             # sparse comping: one or two off-beat stabs a bar
            cand = _snap_table(c, [(p, 0.5) for p in (0, 1.5, 2.5, 3.5)])
            rows = [cand[i] for i in sorted(rng.choice(len(cand), size=min(len(cand), int(rng.integers(1, 3))), replace=False))]
        else:
            rows = _snap_table(c, _COMP[comp])
        for pos, d in rows:
            for p in v:
                out.append(_ev(t0 + pos, min(d, c.meter - pos) * (1.0 if pad or comp == "sustain" else 0.9), p,
                               0.6 if pad else (0.8 if pos == 0 else 0.7)))
    return out


def _slots(c, ch):
    """Farnell's 8-slot chord list (g15 s1.5): slots 1-2 bass pair, 3-4 middle, 5-6 upper, 7-8 top."""
    if ch.get("slots"):
        s = list(ch["slots"])
        return (s * 8)[:8] if len(s) < 8 else s[:8]
    r = _near(_pitches(c.kpc, [ch["root"]], 36, 52), 43)
    low2 = _near([p for p in _pitches(c.kpc, [ch["pcs"][2 % len(ch["pcs"])]], 36, 64) if p > r] or [r + 7], r + 7)
    return [r, low2] + _pitches(c.kpc, ch["pcs"], 60, 96)[:6]


def _arp_order(tones, mode, n, rng, pattern=None):
    k = len(tones)
    if mode == "up":
        idx = [i % k for i in range(n)]
    elif mode == "down":
        idx = [k - 1 - i % k for i in range(n)]
    elif mode == "updown":
        cyc = list(range(k)) + list(range(k - 2, 0, -1)) or [0]
        idx = [cyc[i % len(cyc)] for i in range(n)]
    elif mode == "random":
        idx = [int(rng.integers(k)) for _ in range(n)]
    elif mode == "pattern":
        pat = pattern or [0, 2, 1, 2]
        idx = [int(pat[i % len(pat)]) % k for i in range(n)]
    else:
        raise SpecError("arp mode: up|down|updown|random|pattern|pool")
    return [tones[i] for i in idx]


def _gen_arp(c, sec, rng, L):
    mode, rate = L.get("mode", "up"), float(L.get("rate", c.grid))
    gate, octs = float(L.get("gate", 0.8)), int(L.get("octaves", 2))
    out = []
    for b, ch in enumerate(sec.chords):
        n = int(round(c.meter / rate))
        t0 = (sec.bar0 + b) * c.meter
        if mode == "pool":
            # Farnell C002 arp: the 8 chord slots + slots 6-8 an octave up = 11 notes, `random 12`
            # -> one draw in twelve is a rest (g15 s1.3)
            s = _slots(c, ch)
            pool = s + [p + 12 for p in s[5:8]]
            for i in range(n):
                r = int(rng.integers(12))
                if r < 11:
                    out.append(_ev(t0 + i * rate, rate * gate, pool[r], 0.75))
            continue
        tones = _pitches(c.kpc, ch["pcs"], c.lo, min(c.hi, c.lo + 12 * octs)) or _tones(c, ch)
        for i, p in enumerate(_arp_order(tones, mode, n, rng, L.get("pattern"))):
            out.append(_ev(t0 + i * rate, rate * gate, p, 0.8 if i % max(1, c.spb) == 0 else 0.65))
    return out


def _gen_slots(c, sec, rng, L):
    """Farnell's sequencer written out (g15 s1.10): a step mask (`select` list) with gated steps, each
    hit reading a fixed slot of the current 8-note chord list; optional sub-octave and rest rolls."""
    slots = [int(s) for s in L.get("slots", [1, 2])]
    mask = L.get("mask", "x-------x-------")
    flip, sub, rest = int(L.get("flip", 12)), float(L.get("sub_octave_p", 0.0)), float(L.get("rest_p", 0.0))
    steps = [ch for ch in str(mask) if ch not in " |"]
    out = []
    for b, ch in enumerate(sec.chords):
        s8 = _slots(c, ch)
        for s in range(c.S):
            g = b * c.S + s
            hit = mask_hits(steps[g % len(steps)], rng, {"0": (L.get("gate_steps") or {}).get(str(g % len(steps)), 1.0)})
            if not hit or rng.random() < rest:
                continue
            p = s8[(slots[(g // flip) % len(slots)] - 1) % 8]
            if rng.random() < sub:
                p -= 12
            out.append(_ev((sec.bar0 + b) * c.meter + s * c.grid, c.grid * 0.9, p, hit[0][1]))
    return out


# ============================================================================ generators: drums
def _gen_drums(c, sec, rng, L):
    kit = L.get("kit") or (c.style.kit if c.style and c.style.kit else {"kick": "x-----x-x-------", "snare": _BB, "hihat": _H8})
    if isinstance(kit, str):
        kit = _style(kit, True).kit or {}
    fe = int(L.get("fill_every", 4))
    names = [k.split(":")[0] for k in kit]
    fill_a = next((k for k in ("snare", "snare808", "clap", "snap", "rim") if k in names), names[0] if names else "snare")
    fill_b = "tom" if fill_a.startswith(("snare", "clap", "snap")) else fill_a
    out = []
    for b in range(sec.nbars):
        zone = c.S
        if fe > 0 and (b + 1) % fe == 0 and c.S >= 8 and kit:           # a fill at the end of every N-th bar
            zone = c.S - (c.S // 2 if (b + 1) % (2 * fe) == 0 else c.S // 4)
        t0 = (sec.bar0 + b) * c.meter
        for key, pat in kit.items():
            steps = [ch for ch in str(pat) if ch not in " |"]
            name = key.split(":")[0]
            for s in range(c.S):
                ch = steps[(b * c.S + s) % len(steps)]
                if ch not in _LEVEL or (ch == "?" and rng.random() < 0.5):
                    continue
                if s >= zone and not name.startswith("kick"):
                    continue
                if name in _HATS and s % c.spb and (rng.random() > 0.5 + c.density or (ch == "o" and c.density < 0.35)):
                    continue
                out.append(_ev(t0 + s * c.grid, 0.0, None, _LEVEL[ch], drum=key))
        for s in range(zone, c.S):
            if rng.random() < 0.75:
                x = (s - zone) / max(1, c.S - zone - 1)
                out.append(_ev(t0 + s * c.grid, 0.0, None, 0.5 + 0.5 * x, drum=fill_b if x > 0.5 else fill_a, fill=True))
    return out


_GEN = {"melody": _gen_melody, "bass": _gen_bass, "chords": _gen_chords, "arp": _gen_arp, "drums": _gen_drums,
        "counter": _gen_counter, "pad": lambda c, s, r, L: _gen_chords(c, s, r, L, pad=True)}


# ============================================================================ performance
def _perform(ev, c, L, tag):
    """Swing, humanise (timing / velocity), tempo drift, clamp to the piece length, sort."""
    total = c.bars * c.meter
    sw = float(L.get("swing", c.style.swing if c.style else 0.0))
    sg = float(L.get("swing_grid", c.grid))
    if sw:                                              # same meaning as `pattern`: every other step is late
        for e in ev:
            k = e["t"] / sg
            if abs(k - round(k)) < 1e-6 and int(round(k)) % 2:
                e["t"] += sw * sg
    hum = L.get("humanize", 0)
    hum = hum if isinstance(hum, dict) else {"time": 10.0 * float(hum), "vel": 0.13 * float(hum)}
    # UNSOURCED default (12-music s5): onset jitter sigma 5-10 ms. Velocity spread: Beta(0.2, 0.2)
    # over 110..127 of 127 (= 0.13 of the range), MIT 21M.380 p.56.
    sig, vs = float(hum.get("time", 0.0)) / 1000.0 / c.q, float(hum.get("vel", 0.0))
    if sig > 0 or vs > 0:
        rng = _rng(c.seed, tag, "perform")
        jit = {}
        for e in sorted(ev, key=lambda e: e["t"]):
            if sig > 0:
                if e["t"] not in jit:                   # notes of one chord move together
                    jit[e["t"]] = float(rng.normal(0, sig))
                e["_j"] = jit[e["t"]]
            if vs > 0:
                e["vel"] *= 1.0 - vs * float(rng.beta(0.2, 0.2))
        for e in ev:
            e["t"] = max(0.0, e["t"] + e.pop("_j", 0.0))
    a = float(L.get("drift", 0.0))
    if a:
        # periodic tempo drift, +-a around the tempo with a whole number of ~20 s cycles so the piece
        # keeps its exact length (MIT 21M.380 p.58 "sine 20 s, 115-125 BPM"; after procedural TempoMap)
        k = max(1, int(round(total * c.q / 20.0)))
        w = 2 * math.pi * k / total
        phi = float(L.get("drift_phase", _rng(c.seed, "drift").uniform(0, 2 * math.pi)))
        f = lambda b: b + a / w * (math.sin(w * b + phi) - math.sin(phi))      # noqa: E731
        for e in ev:
            t1 = f(e["t"] + e["dur"])
            e["t"] = f(e["t"])
            e["dur"] = t1 - e["t"]
    out = []
    for e in ev:
        if e["t"] >= total - 1e-9:
            continue
        e["t"] = round(max(0.0, e["t"]), 6)
        e["dur"] = round(max(0.0, min(e["dur"], total - e["t"])), 6)
        e["vel"] = round(float(np.clip(e["vel"], 0.05, 1.0)), 4)
        if e["pitch"] is not None:
            e["pitch"] = round(float(e["pitch"]), 6)
        out.append(e)
    return sorted(out, key=lambda e: (e["t"], e.get("drum", ""), e["pitch"] or 0.0))


def _beat_q(layer, bpm):
    bpm = layer.get("bpm") or bpm
    return 60.0 / float(bpm) if bpm else 1.0


def generate(layer: dict, bpm: float | None = None) -> list[dict]:
    """Note events of a generator layer (`compose`, `euclid`, `arp`, `stinger`), ready to export.

    Each event: {"t": onset in beats, "dur": beats, "pitch": MIDI number (fractional in a non-12-TET
    tuning; None for drums), "vel": 0..1}, drums add "drum" (kind), a chromatic approach tone adds
    "approach": true, acid notes add "slide"/"accent"/"tuple303". Same layer -> same events.
    `bpm` (or the layer's own "bpm") only matters for `humanize` (ms) and `drift`."""
    kind = layer.get("type", "compose")
    if kind == "euclid":
        return _euclid_events(layer, bpm)
    if kind == "arp":
        return _arp_events(layer, bpm)
    if kind == "stinger":
        return _stinger_events(layer, bpm)
    if kind != "compose":
        raise SpecError(f"generate() handles compose|euclid|arp|stinger layers, not {kind!r}")
    c = _context(layer, bpm)
    gen = _gen_slots if (layer.get("mode") == "slots" or "slots" in layer) and c.role != "drums" else _GEN[c.role]
    ev = []
    for sec in c.sections:
        # a section's material depends on its letter, so "A" comes back note for note
        ev += gen(c, sec, _rng(c.seed, c.role, sec.mat, layer.get("name", "")), layer)
    thr = float(layer.get("threshold", 0.0))
    kept = []
    for e in ev:                                        # vertical layering: mute the bars under the threshold
        lvl = c.intensity[min(c.bars - 1, int(e["t"] // c.meter + 1e-9))]
        if lvl + 1e-9 >= thr:
            e["vel"] *= 0.8 + 0.2 * min(1.0, lvl)
            kept.append(e)
    return _perform(kept, c, layer, c.role)


# ============================================================================ events -> ordinary layers
def _tok(p: float):
    return N.midi_to_name(int(round(p))) if abs(p - round(p)) < 1e-6 else f"{N.midi_to_freq(p):.4f}hz"


def _layers_from_events(ev, layer, inst=None) -> list[dict]:
    out = []
    notes = [e for e in ev if e["pitch"] is not None]
    if notes:
        groups = []                                     # notes sharing an onset form a chord step
        for e in notes:
            if groups and groups[-1][0] == e["t"]:
                groups[-1][1].append(e)
            else:
                groups.append((e["t"], [e]))
        steps, cursor = [], 0.0
        for i, (t, es) in enumerate(groups):
            if t > cursor + 1e-9:
                steps.append({"note": "-", "dur": round(t - cursor, 6)})
                cursor += steps[-1]["dur"]
            room = groups[i + 1][0] - cursor if i + 1 < len(groups) else max(e["dur"] for e in es)
            d = round(max(1e-4, min(max(e["dur"] for e in es), room)), 6)
            st = {"notes": [_tok(e["pitch"]) for e in es]} if len(es) > 1 else {"note": _tok(es[0]["pitch"])}
            steps.append({**st, "dur": d, "vel": round(float(np.mean([e["vel"] for e in es])), 4)})
            cursor += d
        seq = {"type": "seq", "inst": inst or layer.get("inst") or "pluck", "steps": steps}
        for k in ("params", "strum", "legato"):
            if k in layer:
                seq[k] = layer[k]
        out.append(seq)
    groups = {}
    for e in ev:
        if e["pitch"] is None:
            groups.setdefault((e["drum"], round(round(e["vel"] / 0.05) * 0.05, 2)), []).append(e["t"])
    kp = layer.get("kit_params") or {}
    for (key, v), hits in groups.items():               # one `pattern` per drum and velocity level
        name, _, ptxt = key.partition(":")
        params = {a: FX._coerce(b) for a, b in (kv.split("=") for kv in ptxt.split(",") if kv)}
        params.update(kp.get(name, {}))
        if layer.get("type") == "euclid":
            params.update(layer.get("params") or {})
        pl = {"type": "pattern", "kind": name, "hits": hits, "vel": v, "gain": round(v * DRUM_GAIN.get(name, 0.4), 4)}
        if params:
            pl["params"] = params
        out.append(pl)
    return out


_COMMON = ("at", "gain", "fx", "repeat", "every", "bpm")


def to_spec(layer: dict, bpm: float | None = None, sr: int = DEFAULT_SR) -> dict:
    """Convert a generator layer into ordinary layers the user can edit: a `group` of `seq` layers
    (notes) and `pattern` layers (drum hits, one per drum and velocity level), carrying the layer's
    own at / gain / fx / repeat. It renders to the same samples as the generator layer itself.
    Pass the spec's `bpm` when the layer has none of its own and uses `humanize`, `drift`, or is an
    arrangement. `scatter` becomes a group of placed children; `arrangement` a group of its parts
    with their gains, pans (`automix`, analysed at `sr`) and master fx."""
    kind = layer.get("type")
    if kind in ("compose", "euclid", "arp", "stinger"):
        dflt = MOODS.get(layer.get("mood", "victory"), MOODS["victory"])[1] if kind == "stinger" else (
            _ROLES.get({"lead": "melody"}.get(layer.get("role", "melody"), layer.get("role", "melody")), (0, None))[1] if kind == "compose" else None)
        inner = _layers_from_events(generate(layer, bpm), layer, layer.get("inst") or dflt)
    elif kind == "scatter":
        inner = _scatter_layers(layer)
    elif kind == "arrangement":
        a = arrange(layer, bpm)
        parts, _ = _automix(a["layers"], sr, a["bpm"], float(layer.get("width", 0.6))) if layer.get("automix", True) else (a["layers"], None)
        out = {"type": "group", "bpm": a["bpm"], "layers": [to_spec(p, a["bpm"], sr) for p in parts], "fx": a["fx"] + list(layer.get("fx") or [])}
        out.update({k: layer[k] for k in ("at", "gain", "repeat", "every") if k in layer})
        return out
    else:
        return dict(layer)
    return {"type": "group", "layers": inner, **{k: layer[k] for k in _COMMON if k in layer}}


def _bpm(q: float):
    """A bpm with 60/bpm == q exactly (layer functions are handed q, the renderer wants bpm)."""
    if q == 1.0:
        return None                                     # no tempo: time units are seconds
    b = 60.0 / q
    for c in (b, np.nextafter(b, 0), np.nextafter(b, 1e9), np.nextafter(np.nextafter(b, 0), 0), np.nextafter(np.nextafter(b, 1e9), 1e9)):
        if 60.0 / float(c) == q:
            return float(c)
    return b


def _render_generated(layer, sr, q):
    bpm = _bpm(q)
    return SP.render_layers(to_spec(layer, bpm, sr)["layers"], sr, bpm)


layer_type("compose", "One generated part (role melody|bass|chords|arp|drums|pad|counter) from key, scale, progression, style and seed.")(_render_generated)
layer_type("euclid", "Euclidean drum pattern E(k, n) with rotation: hits spread as evenly as possible over the steps.")(_render_generated)
layer_type("arp", "Arpeggiate given chords (up, down, updown, random or a pattern).")(_render_generated)
layer_type("stinger", "Generated jingle by mood: victory, defeat, mystery, danger, pickup, unlock.")(_render_generated)
layer_type("scatter", "Random placement of child layers (count or rate, time / pitch / gain / pan jitter): ambiences, debris.")(_render_generated)


# ============================================================================ euclid / arp / stinger layers
def _euclid_events(layer, bpm=None):
    """{"type": "euclid", "kind": "hihat", "k": 5, "n": 16, "rotate": 0, "bars": 2, "step": 0.25}"""
    n, step = int(layer.get("n", 16)), float(layer.get("step", 0.25))
    pat = euclid(int(layer.get("k", 4)), n, int(layer.get("rotate", 0)))
    acc = [ch for ch in str(layer.get("accent", "")) if ch not in " |"]
    prob, rng = float(layer.get("prob", 1.0)), _rng(layer.get("seed", 0), "euclid")
    pitch = 69 + 12 * math.log2(N.to_freq(layer["note"]) / 440.0) if layer.get("inst") else None
    ev = []
    for i in range(n * max(1, int(layer.get("bars", 1)))):
        if pat[i % n] and (prob >= 1.0 or rng.random() < prob):
            v = _LEVEL.get(acc[i % len(acc)], 0.8) if acc else 0.8
            ev.append(_ev(i * step, step * 0.9, pitch, v, **({} if pitch is not None else {"drum": layer.get("kind", "kick")})))
    c = NS(bars=1, meter=n * step * max(1, int(layer.get("bars", 1))), grid=step, style=None, seed=int(layer.get("seed", 0)), q=_beat_q(layer, bpm))
    return _perform(ev, c, layer, "euclid")


def _arp_events(layer, bpm=None):
    """{"type": "arp", "inst": "harp", "chords": "C4:maj A3:min", "mode": "updown", "rate": 0.25, "beats": 4}"""
    chords = layer.get("chords", "C4:maj")
    chords = chords.split() if isinstance(chords, str) else chords
    q = _beat_q(layer, bpm)
    rate = float(layer.get("rate", 0.25 if q != 1.0 else 0.125))
    per = float(layer.get("beats", 4 if q != 1.0 else 1))
    octs, gate, mode = int(layer.get("octaves", 1)), float(layer.get("gate", 0.8)), layer.get("mode", "up")
    rng, ev, n = _rng(layer.get("seed", 0), "arp"), [], max(1, int(round(per / rate)))
    for ci, ch in enumerate(chords):
        base = sorted(69 + 12 * math.log2(f / 440.0) for f in N.parse_pitch_list(ch))
        tones = [round(p + 12 * o, 6) for o in range(max(1, octs)) for p in base]
        for i, p in enumerate(_arp_order(tones, mode, n, rng, layer.get("pattern"))):
            ev.append(_ev(ci * per + i * rate, rate * gate, p, 0.85 if i == 0 else 0.7))
    c = NS(bars=1, meter=per * len(chords), grid=rate, style=None, seed=int(layer.get("seed", 0)), q=q)
    return _perform(ev, c, layer, "arp")


# mood -> (scale, instrument, default length in notes, tonic MIDI octave base, velocity)
# Shapes follow AGENTS.md "Design recipes" (rising = positive, falling = negative; win = major triad up
# to the octave chord; game over = descending minor) and Phillips (a stinger with no known key is
# cluster-like and unmetred, 12-music s2.3).
MOODS = {"victory": ("major", "brass", 5, 60, 0.9), "defeat": ("minor", "piano", 4, 60, 0.75),
         "mystery": ("whole_tone", "vibraphone", 5, 60, 0.6), "danger": ("locrian", "brass", 4, 48, 0.95),
         "pickup": ("pentatonic", "marimba", 3, 72, 0.8), "unlock": ("lydian", "bell", 4, 72, 0.7)}


def _stinger_events(layer, bpm=None):
    mood = layer.get("mood", "victory")
    if mood not in MOODS:
        raise SpecError(f"stinger mood {mood!r}: one of {sorted(MOODS)}")
    sc, _, n0, base, vel = MOODS[mood]
    offs = scale_offsets(layer.get("scale", sc))
    L, n = len(offs), max(2, int(layer.get("length", n0)))
    q = _beat_q(layer, bpm)
    step = float(layer.get("step", 0.25 if q != 1.0 else (0.22 if mood == "defeat" else 0.11)))
    rng = _rng(layer.get("seed", 0), "stinger", mood)
    tonic = _key_pc(layer.get("key", "C")) + base + 12 * int(layer.get("octave", 0))
    P = lambda i: tonic + 12 * (i // L) + offs[i % L]                             # noqa: E731
    v = int(rng.integers(0, 3))
    if mood == "victory":                               # up the tonic triad, landing on the octave chord
        seq = [[0, 2, 4][k % 3] + L * (k // 3 + 1) for k in range(1 - n, 0)]      # ...sol do mi sol | DO
        final = [[L, L + 2, L + 4], [4, L, L + 2], [L, L + 4, 2 * L]][v]
    elif mood == "defeat":                              # down by step to the tonic, last note long
        seq = [n - 1 - i + (1 if (i == 0 and v == 1) else 0) for i in range(n - 1)]
        final = [0]
    elif mood == "mystery":                             # wandering whole tones ending on an unresolved augmented chord
        seq = [int(x) for x in np.cumsum(np.concatenate([[v], rng.choice([-1, 1, 2], n - 2)]))]
        final = [seq[-1] + 1, seq[-1] + 3, seq[-1] + 5]
    elif mood == "danger":                              # root against the flat second, then a tritone cluster
        seq = [(i + v) % 2 for i in range(n - 1)]
        final = [0, 1, 4]
    elif mood == "pickup":                              # a quick rising figure
        up = [0, 2, 3, 5, 7, 8, 10]
        seq = [up[(i + v) % len(up)] for i in range(n - 1)]
        final = [up[(n - 1 + v) % len(up)] + (L if (n - 1 + v) >= len(up) else 0)]
    else:                                               # unlock: the raised fourth turning into the octave
        cyc = [[0, 4, 3, 4], [0, 2, 3, 4], [4, 3, 4, 6]][v]
        seq = [cyc[i % 4] + L * (i // 4) for i in range(n - 1)]
        final = [L, L + 4]
    ev = [_ev(i * step, step * 0.95, P(s), vel * (0.8 + 0.2 * (i + 1) / n)) for i, s in enumerate(seq)]
    ev += [_ev((n - 1) * step, step * float(layer.get("hold", 4)), P(s), vel) for s in final]
    c = NS(bars=1, meter=(n - 1 + float(layer.get("hold", 4))) * step, grid=step, style=None, seed=int(layer.get("seed", 0)), q=q)
    return _perform(ev, c, layer, "stinger")


# ============================================================================ scatter layer
def _scatter_layers(layer: dict) -> list[dict]:
    """Random placement of child layers (FMOD Scatterer, 12-music s6.4: interval window 500-1000 ms,
    random volume / pitch windows; Xenakis density fields). Placement modes: `interval` [min, max]
    between onsets, `rate` (events per time unit, Poisson: exponential waits, as Cook's PhISM
    collisions), or `count` uniform in the window. `silence_p` leaves slots empty (Farnell's
    "silence as an option", g15 s7)."""
    kids = layer.get("layers") or []
    if not kids:
        raise SpecError("scatter needs child `layers`")
    rng = np.random.default_rng(int(layer.get("seed", 0)))
    dur = float(layer.get("dur", 8.0))
    if "interval" in layer:
        lo, hi = (float(v) for v in layer["interval"])
        if lo <= 0 or hi < lo:
            raise SpecError("scatter interval: [min, max] with 0 < min <= max")
        times, t = [], float(rng.uniform(0, hi))
        while t < dur:
            times.append(t)
            t += float(rng.uniform(lo, hi))
    elif "rate" in layer:
        rate = float(layer["rate"])
        if rate <= 0:
            raise SpecError("scatter rate must be > 0")
        times, t = [], float(rng.exponential(1.0 / rate))
        while t < dur and len(times) < 100000:
            times.append(t)
            t += float(rng.exponential(1.0 / rate))
    else:
        times = sorted(float(x) for x in rng.uniform(0, dur, max(0, int(layer.get("count", 8)))))
    win = lambda k, d: tuple(float(v) for v in (layer.get(k) if isinstance(layer.get(k), (list, tuple)) else (d if layer.get(k) is None else (-abs(layer[k]), abs(layer[k])))))   # noqa: E731
    jt, pw, gw, nw = float(layer.get("time_jitter", 0.0)), win("pitch", (0, 0)), win("gain_db", (0, 0)), win("pan", (0, 0))
    sil = float(layer.get("silence_p", 0.0))
    from . import sfx as S
    out = []
    for t in times:
        kid = dict(kids[int(rng.integers(len(kids)))])
        st, g, p, j = float(rng.uniform(*pw)), float(rng.uniform(*gw)), float(rng.uniform(*nw)), float(rng.uniform(-jt, jt))
        new_seed = int(rng.integers(1 << 30))
        if rng.random() < sil:
            continue
        reg = {"sfx": S.REGISTRY, "drum": DRUMS}.get(kid.get("type"))
        if layer.get("reseed") and reg and "seed" in reg.get(kid.get("kind"), {}).get("params", {}):
            kid["params"] = dict(kid.get("params") or {}, seed=new_seed)     # a different take for every event
        chain = ([{"type": "speed", "factor": round(2 ** (st / 12), 6)}] if st else []) + ([{"type": "pan", "pos": round(p, 4)}] if nw != (0, 0) else [])
        out.append({"type": "group", "layers": [kid], "at": round(max(0.0, t + j), 6), "gain": round(10 ** (g / 20), 5), "fx": chain})
    return out


# ============================================================================ automix (automatic panner)
def pan_steps(R: int) -> list[int]:
    """Perez-Gonzalez & Reiss panning steps for R sources sharing a band, in priority order:
    PS(i) = round((i-1) * 127 / (R-1)) (eq. 3), the most central position to the first priority, then
    alternating sides, extremes last (Table 1). MIDI pan 0..127, centre 64.
    (Table 1 prints 43/84 for R = 4 where the equation gives 42/85.)"""
    if R <= 1:
        return [64]
    pos = [int(math.floor(i * 127 / (R - 1) + 0.5)) for i in range(R)]
    return sorted(pos, key=lambda p: (abs(p - 63.5), p))


def band_class(x: np.ndarray, sr: int) -> int | None:
    """Dominant octave band of a signal: 100 ms frames, frames more than 60 dB under the peak are
    gated out, each surviving frame votes for its strongest band, the band with most votes wins
    (decisions accumulate). Band k spans 20*2^k .. 20*2^(k+1) Hz. Returns None for silence.

    Deviation from the paper (its filter-bank equations are garbled in the source text, and bands
    whose count equals the channel count cannot classify a 4-part mix): fixed octave bands from
    Fmin = 20 Hz, measured by FFT energy, and a gate relative to the signal's own peak."""
    m = to_mono(np.asarray(x, dtype=np.float64))
    w = max(64, int(0.1 * sr))
    peak = float(np.max(np.abs(m))) if m.size else 0.0
    if peak < 1e-9:
        return None
    m = pad_to(m, w)
    frames = m[:(m.shape[0] // w) * w].reshape(-1, w)
    keep = np.max(np.abs(frames), axis=1) > peak * 1e-3
    spec = np.abs(np.fft.rfft(frames[keep] * np.hanning(w), axis=1)) ** 2
    f = np.fft.rfftfreq(w, 1.0 / sr)
    band = np.floor(np.log2(np.maximum(f, 1e-9) / 20.0)).astype(int)
    votes = np.zeros(12)
    for row in spec:
        e = np.bincount(band[(band >= 0) & (band < 12)], weights=row[(band >= 0) & (band < 12)], minlength=12)
        votes[int(np.argmax(e))] += 1
    return int(np.argmax(votes))


def automix_pan(signals: list[np.ndarray], sr: int, center_below: float = 200.0, width: float = 1.0) -> list[float]:
    """Automatic panner of Perez-Gonzalez & Reiss (DAFx-07; g15 s6.1). Priority = list order.
    Rule 1: a source whose dominant band ends below `center_below` Hz stays centred. Rule 2-3: sources
    sharing a dominant band take the equidistant `pan_steps` in priority order, so similar spectra end
    up far apart (less masking). Returns positions -1 (left) .. 1 (right) for the constant-power `pan`
    fx (theta = PS/127 * pi/2). `width` scales the spread (1 = the paper)."""
    classes = [band_class(s, sr) for s in signals]
    pos = [0.0] * len(signals)
    for k in set(c for c in classes if c is not None):
        members = [i for i, c in enumerate(classes) if c == k]
        if 20.0 * 2 ** (k + 1) < center_below:
            continue
        for i, ps in zip(members, pan_steps(len(members))):
            pos[i] = 0.0 if ps == 64 else round((ps / 127.0 * 2 - 1) * float(width), 4)
    return pos


def _automix(layers, sr, bpm, width=1.0, center_below=200.0):
    sigs = [SP.render_layer(dict(l, at=0), sr, bpm)[0] for l in layers]
    out = []
    for l, p in zip(layers, automix_pan(sigs, sr, center_below, width)):
        out.append(dict(l, fx=list(l.get("fx") or []) + [{"type": "pan", "pos": p}]) if p else dict(l))
    return out, sigs


def automix(layers: list[dict], sr: int = DEFAULT_SR, bpm: float | None = None, width: float = 1.0,
            center_below: float = 200.0) -> list[dict]:
    """Set the pan of each layer from what it sounds like (see `automix_pan`): renders every layer
    once, returns copies with a `pan` fx appended where the position is off centre. Gains are left
    alone: the sources give no numbers for automatic gain or unmasking EQ (g15 s6.2), so
    `arrangement` uses the fixed "Mix by gain" priors of AGENTS.md instead."""
    return _automix(layers, sr, bpm, width, center_below)[0]


# ============================================================================ arrangement
# part, role, gain (AGENTS.md "Mix by gain"), intensity threshold
# UNSOURCED thresholds: low / medium / high intensity grouping (inFAMOUS 2, 12-music s2.1) as numbers
_PARTS = [("drums", "drums", 1.0, 0.4), ("bass", "bass", 0.85, 0.2), ("chords", "chords", 0.45, 0.0),
          ("pad", "pad", 0.3, 0.0), ("lead", "melody", 0.7, 0.5), ("arp", "arp", 0.35, 0.7), ("counter", "counter", 0.45, 0.85)]
_SHARED = ("key", "scale", "tuning", "bars", "sections", "progression", "chords", "harmonic_rhythm", "seed", "meter",
           "grid", "swing", "humanize", "drift", "intensity", "density", "chord_size")


def tempo_delay(bpm: float, k: float = 3, feedback: float = 0.3, mix: float = 0.25) -> dict:
    """Tempo-locked echo (Farnell C002, g15 s1.1: every time-based parameter is a multiple of the
    beat period): time = k eighth notes. Arp 3 x / 0.3, bass 2 x / 0.6, slap-back 0.5 x / 0.05."""
    return {"type": "delay", "time": round(k * 30.0 / float(bpm), 6), "feedback": min(0.9, feedback), "mix": mix}


def arrange(layer: dict, bpm: float | None = None) -> dict:
    """An `arrangement` layer -> {"bpm", "beats", "layers": [compose parts], "fx": master chain}.
    The parts share key, scale, form, progression and seed, so the harmony is the same in all of
    them; instruments, grooves and tempo come from the style (`STYLES`). No pans yet: see `automix`."""
    name = layer.get("style", "pop")
    st = _style(name, True)
    seed = int(layer.get("seed", 0))
    bpm = float(layer.get("bpm") or bpm or style_bpm(name, seed))
    shared = {k: layer[k] for k in _SHARED if k in layer}
    shared.update(style=name, seed=seed, bars=int(layer.get("bars", 8)))
    if layer.get("drift"):
        shared["drift_phase"] = layer.get("drift_phase", float(_rng(seed, "drift").uniform(0, 2 * math.pi)))
    over = layer.get("parts") or {}
    bad = [k for k in over if k not in [p[0] for p in _PARTS]]
    if bad:
        raise SpecError(f"arrangement parts {bad}: names are {[p[0] for p in _PARTS]}")
    inst = {"drums": "kit" if st.kit else None, "bass": st.bass and st.bass[0], "chords": st.chords and st.chords[0],
            "pad": st.pad, "lead": st.lead, "arp": st.arp, "counter": st.counter}
    parts = []
    for pname, role, gain, thr in _PARTS:
        ov = over.get(pname)
        if ov is False or (inst[pname] is None and not isinstance(ov, dict)):
            continue
        p = {"type": "compose", "role": role, "name": pname, **shared, "threshold": thr, "gain": gain}
        if role != "drums":
            p["inst"] = inst[pname] or _ROLES[role][1]
        fx = []
        if pname == "chords" and st.dist:               # a distorted layer sits at full scale: turn it down
            fx.append({"type": "distortion", "drive": st.dist})
            p["gain"] = 0.3
        if pname in st.delay:
            fx.append(tempo_delay(bpm))
        if name == "chiptune" and pname == "drums":
            fx.append({"type": "bitcrush", "bits": 6, "rate": 11025})
        if fx:
            p["fx"] = fx
        if isinstance(ov, dict):
            p.update(ov)
        parts.append(p)
    c = _context(parts[0] if parts else dict(shared, type="compose"), bpm)
    master = [] if layer.get("master", True) is False else [{"type": "reverb", "mix": 0.15, "size": 0.5}, {"type": "compressor", "threshold": -16, "ratio": 2.5}]
    return {"bpm": bpm, "beats": c.bars * c.meter, "layers": parts, "fx": master}


def _stems(layer, sr, q):
    """Render an arrangement part by part: ([(name, stereo signal placed and panned)], n_exact, plan)."""
    bpm = _bpm(q) if (q != 1.0 or layer.get("bpm")) else None
    a = arrange(layer, bpm)
    bpm = a["bpm"]
    if layer.get("automix", True):
        parts, sigs = _automix(a["layers"], sr, bpm, float(layer.get("width", 0.6)))     # UNSOURCED width: the paper's full spread hard-pans pairs
        sigs = [FX.apply_chain(s, [f for f in p.get("fx", [])[-1:] if f["type"] == "pan"], sr) for p, s in zip(parts, sigs)]
    else:
        parts = a["layers"]
        sigs = [SP.render_layer(p, sr, bpm)[0] for p in parts]
    n = int(round(a["beats"] * 60.0 / bpm * sr))
    return [(p["name"], to_stereo(s)) for p, s in zip(parts, sigs)], n, a


def _fit(y, n, sr, layer):
    """Length policy: natural ring-out (default), `tail` seconds after the last beat, or `fold`
    (the ring-out laid back over the start: an exact, seamless musical loop like "loop": "fold")."""
    if layer.get("fold"):
        out = pad_to(y[:n].copy(), n)
        for s in range(n, y.shape[0], n):
            seg = y[s:s + n]
            out[:seg.shape[0]] += seg
        return out
    if layer.get("tail") is not None:
        m = n + int(round(float(layer["tail"]) * sr))
        y = pad_to(y, m)[:m].copy()
        k = min(m // 2, samples(0.005, sr))
        y[-k:] *= np.linspace(1.0, 0.0, k)[:, None] if y.ndim == 2 else np.linspace(1.0, 0.0, k)
    return y


@layer_type("arrangement", "A whole piece from one description: style, key, bars, sections, seed (drums, bass, harmony, lead on one progression).")
def _render_arrangement(layer, sr, q):
    stems, n, a = _stems(layer, sr, q)
    y = _fit(FX.apply_chain(mix([(s, 0.0) for _, s in stems], sr), a["fx"], sr), n, sr, layer)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y * (0.98 / peak) if peak > 0.98 else y      # one gain for the whole piece, never a clipped sum


# ============================================================================ adaptive music
def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name)) or "x"


@layer_type("adaptive")
def _render_adaptive(layer, sr, q):
    """iMuse-style state graph kept to what a WAV renderer can honestly do (g15 s8; Phillips, vertical
    layering and horizontal re-sequencing, 12-music s2): every state is an arrangement from the same
    seed / key / tempo (one shared score), rendered as an exact folded loop; with `out_stems` each
    state, each part of each state and each transition stinger is written as its own WAV next to a
    `graph.json` (states, loop lengths, transitions quantised to the bar). All files share one gain,
    so the intensity differences between states survive. Returns the audible demo: the states in
    `order`, transition stingers on the boundaries."""
    states = layer.get("states") or {}
    if not states:
        raise SpecError("adaptive needs `states`: {name: {arrangement overrides}}")
    base = {k: v for k, v in layer.items() if k not in ("type", "states", "transitions", "order", "out_stems", "at", "gain", "fx", "repeat", "every")}
    rendered, graph = {}, {"states": {}, "transitions": []}
    for name, over in states.items():
        cfg = {**base, **(over or {}), "type": "arrangement", "fold": True}
        cfg.pop("tail", None)
        stems, n, a = _stems(cfg, sr, q)
        full = _fit(FX.apply_chain(mix([(s, 0.0) for _, s in stems], sr), a["fx"], sr), n, sr, cfg)
        rendered[name] = (to_stereo(full), [(pn, _fit(s, n, sr, cfg)) for pn, s in stems], a)
        graph["states"][name] = {"file": f"{_safe(name)}.wav", "beats": a["beats"], "bpm": a["bpm"], "loop": True,
                                 "stems": [f"{_safe(name)}__{_safe(pn)}.wav" for pn, _ in stems]}
    key = base.get("key", "C")
    stingers = {}
    for tr in layer.get("transitions") or []:
        a, b = tr["from"], tr["to"]
        if a not in states or b not in states:
            raise SpecError(f"transition {a!r} -> {b!r}: unknown state")
        g = {"from": a, "to": b, "quantize": tr.get("quantize", "bar")}
        if tr.get("stinger"):
            st = {"type": "stinger", "mood": tr["stinger"], "key": key, "seed": int(base.get("seed", 0)),
                  **{k: tr[k] for k in ("inst", "length", "octave", "scale") if k in tr}}
            stingers[(a, b)] = to_stereo(SP.render_layer(st, sr, rendered[b][2]["bpm"])[0]) * float(tr.get("gain", 0.7))
            g["stinger"] = f"stinger_{_safe(a)}_to_{_safe(b)}.wav"
        graph["transitions"].append(g)
    order = layer.get("order") or list(states)
    parts, t = [], 0.0
    for i, name in enumerate(order):
        if name not in rendered:
            raise SpecError(f"order: unknown state {name!r}")
        if i and (order[i - 1], name) in stingers:
            parts.append((stingers[(order[i - 1], name)], t))
        parts.append((rendered[name][0], t))
        t += rendered[name][0].shape[0] / sr
    demo = mix(parts, sr)
    peak = max([float(np.max(np.abs(f))) for f, _, _ in rendered.values()] + [float(np.max(np.abs(demo))), 1e-9])
    g = min(1.0, 0.89 / peak)                           # ONE gain for the whole set (Phillips / 12-music W3)
    if layer.get("out_stems"):
        out = Path(str(layer["out_stems"]))
        for name, (full, stems, _) in rendered.items():
            write_wav(out / f"{_safe(name)}.wav", full * g, sr)
            for pn, s in stems:
                write_wav(out / f"{_safe(name)}__{_safe(pn)}.wav", s * g, sr)
        for (a, b), s in stingers.items():
            write_wav(out / f"stinger_{_safe(a)}_to_{_safe(b)}.wav", s * g, sr)
        (out / "graph.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
    return demo * g


# ============================================================================ Karplus-Strong family
_KS_FM = math.sqrt(N.midi_to_freq(N.note_to_midi("E2")) * N.midi_to_freq(N.note_to_midi("E6")))   # geometric mean of the span


def js_design(freq: float, sr: int, t60: float, stretch: float = 0.0, loss: float = 1.0):
    """Loop design of the extended Karplus-Strong string (Jaffe & Smith 1983; g15 s3).
    Returns (N, C, S, rho): integer delay, tuning-allpass coefficient, stretch weight, loss factor.

      loop filter      H_s(z) = (1-S) + S z^-1, gain sqrt(1 - 2S(1-S)(1 - cos wT)) (eq. 21);
                       S = 1/2 is the plain string, S -> 0 removes the frequency-dependent loss
                       ("decay stretching"). `stretch` >= 1 is the factor by which the decay time of
                       the partials grows: the loss per period is ~ S(1-S)(1 - cos wT), so
                       4 S (1-S) = 1/stretch. (Karplus & Strong's d = 1/(2 stretch) only gives that
                       factor for their probabilistic version; checked numerically.)
      loss             rho <= 1 scales every partial alike ("decay shortening", eq. 17).
      t60              S (when `stretch` is 0 = automatic) and rho are solved so the fundamental falls
                       60 dB in t60 seconds: rho * G_s(f1) = 0.001^(1/(f1 t60)). A string that would ring
                       longer on its own is shortened with rho; one that would die sooner is stretched.
      tuning           N + P_s + P_c = fs/f1 with P_s the phase delay of H_s at f1 (eq. 22) and the
                       allpass (C + z^-1)/(1 + C z^-1) supplying P_c in [eps, 1+eps):
                       C = sin((wT - wT P_c)/2) / sin((wT + wT P_c)/2) (eq. 16).
    Stability: rho <= 0.99999 and |H_s| <= 1, so the loop gain is below 1 at every frequency."""
    w = 2 * math.pi * freq / sr
    gt = 10 ** (-3.0 / (freq * max(t60, 0.02)))
    if stretch and stretch > 0:
        S = (1 - math.sqrt(1 - 1 / max(1.0, float(stretch)))) / 2       # 4S(1-S) = 1/stretch
    else:
        K = (1 - gt * gt) / (1 - math.cos(w))
        S = 0.5 if K >= 0.5 else (1 - math.sqrt(1 - 2 * K)) / 2
    S = float(np.clip(S, 1e-5, 0.5))
    gs = math.sqrt(1 - 2 * S * (1 - S) * (1 - math.cos(w)))
    rho = min(0.99999, gt / gs, float(np.clip(loss, 0.5, 1.0)))
    ps = math.atan2(S * math.sin(w), (1 - S) + S * math.cos(w)) / w
    P = sr / freq
    Nd = int(math.floor(P - ps - 0.1))
    pc = P - Nd - ps
    C = math.sin((w - w * pc) / 2) / math.sin((w + w * pc) / 2)
    return Nd, C, S, rho


def dynamics_R(freq: float, sr: int, level_hz: float, f_ref: float = _KS_FM) -> float:
    """Jaffe-Smith dynamics filter y = (1-R) x + R y[-1] with loudness equalised across pitch: R is
    chosen so the gain at the note's fundamental equals the gain a filter of bandwidth `level_hz`
    (R_L = exp(-2 pi L T)) has at the reference pitch `f_ref` (g15 s3 extension 3; the closed form is
    re-derived here: (1-G^2) R^2 - 2 (1 - G^2 cos w1) R + (1-G^2) = 0, root inside the unit circle)."""
    rl = math.exp(-2 * math.pi * level_hz / sr)
    wm, w1 = 2 * math.pi * f_ref / sr, 2 * math.pi * freq / sr
    g2 = (1 - rl) ** 2 / (1 - 2 * rl * math.cos(wm) + rl * rl)
    if g2 >= 1 - 1e-12:
        return 0.0
    bq = 1 - g2 * math.cos(w1)
    disc = bq * bq - (1 - g2) ** 2
    return float(np.clip((bq - math.sqrt(max(disc, 0.0))) / (1 - g2), 0.0, 0.9999))


def js_string(freq, n, sr, t60=3.0, stretch=0.0, loss=1.0, pick=0.2, level_hz=1200.0, bright=3500.0, seed=0):
    """n samples of the extended Karplus-Strong string: two-level (+-1) random burst of one period
    (Karplus & Strong), dynamics lowpass, `bright` ceiling, pick-position comb x[n] - x[n - pick*N],
    then the loop 1/(1 - rho H_s(z) H_c(z) z^-N) run as one recursive filter."""
    Nd, C, S, rho = js_design(freq, sr, t60, stretch, loss)
    if Nd < 1:                                          # above ~sr/3 there is no loop left to model
        return np.sin(2 * np.pi * freq * np.arange(n) / sr) * np.exp(-6.91 * np.arange(n) / (sr * max(t60, 0.02)))
    rng = np.random.default_rng(2000 + int(seed) + int(freq) % 997)
    L = max(2, int(round(sr / freq)))
    x = np.zeros(2 * L + 64)
    x[:L] = rng.integers(0, 2, L) * 2.0 - 1.0
    x[:L] -= x[:L].mean()                               # the loop passes DC with gain rho: do not feed it any
    R = dynamics_R(freq, sr, level_hz)
    x = lfilter([1 - R], [1, -R], x)
    x = F.lowpass(x, min(float(bright), 0.45 * sr), sr)
    m = int(float(np.clip(pick, 0.0, 0.5)) * L)
    if m >= 1:
        x[m:] = x[m:] - x[:-m]
    a = np.zeros(Nd + 3)
    a[0] = 1.0
    a[1] += C
    a[Nd] -= rho * (1 - S) * C
    a[Nd + 1] -= rho * ((1 - S) + S * C)
    a[Nd + 2] -= rho * S
    xin = np.zeros(max(n, 1))
    k = min(len(x), n)
    xin[:k] = x[:k]
    return lfilter([1.0, C], a, xin)


@instrument("ks_string", "Extended Karplus-Strong string (Jaffe-Smith): decay stretching and shortening, pick position, "
            "pitch-equalised dynamics filter, optional sympathetic second string; in tune on every key.",
            family="plucked", span=("E2", "E6"),
            decay=(3.0, "seconds for the fundamental to fall 60 dB at the bottom of the range (shorter up the neck)"),
            stretch=(0.0, "decay-stretch factor >= 1: the upper partials ring this many times longer (1 = plain string, highs die first; 8 = wiry); 0 = automatic from `decay`"),
            loss=(1.0, "loop gain per period rho, 0.9..1: below 1 damps every partial alike (palm mute)"),
            pick=(0.2, "pick position 0..0.5 as a fraction of the string: 0.5 removes even harmonics, 0 = off"),
            dyn=(2000.0, "Hz: bandwidth of the dynamics filter at full velocity (brightness of a hard pluck)"),
            bright=(3500.0, "Hz: ceiling on the pluck spectrum; raise to open it up"),
            sympathetic=(0.0, "0..0.5: how hard the note drives an undamped second string"),
            symp_interval=(-12.0, "semitones from the note to the sympathetic string"),
            seed=(0, "int: pluck noise"))
def ks_string(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=3.0, stretch=0.0, loss=1.0, pick=0.2, dyn=2000.0, bright=3500.0,
              sympathetic=0.0, symp_interval=-12.0, seed=0):
    t60 = max(0.05, float(decay)) * _kt(freq, 1.0, 0.45)
    ring = max(0.05, min(t60 * 0.9, dur + 1.0))
    n = samples(ring, sr)
    level = max(80.0, float(dyn) * (0.25 + 0.75 * float(np.clip(vel, 0.0, 1.0)) ** 2))   # UNSOURCED velocity -> bandwidth map
    y = js_string(freq, n, sr, t60, stretch, loss, pick, level, bright, seed)
    g = float(np.clip(sympathetic, 0.0, 0.5))
    if g > 0:                                           # Jaffe-Smith sympathetic string: a second loop fed by the first
        f2 = freq * 2 ** (float(symp_interval) / 12)
        if 20.0 < f2 < sr / 3:
            Nd, C, S, rho = js_design(f2, sr, t60, 1.0, 0.9995)      # its own loss is mandatory (rho < 1)
            a = np.zeros(Nd + 3)
            a[0] = 1.0
            a[1] += C
            a[Nd] -= rho * (1 - S) * C
            a[Nd + 1] -= rho * ((1 - S) + S * C)
            a[Nd + 2] -= rho * S
            y = y + g * (lfilter([1.0, C], a, y) - y) * 0.5
    y = y * _off(n, dur, sr, 0.1) * perc(ring, 0.0005, sr, curve=1.0)
    y = F.dc_block(y, sr)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel) * _KS_TRIM


_KS_TRIM = 0.85           # level match: about -12 dB K-weighted at vel 0.9 (measured in tests/test_compose.py)
_KS_FI = 22050           # internal rate of the drum kernel


def ks_drum_raw(n: int, p: int, blend: float = 0.5, stretch: float = 1.0, seed: int = 0) -> np.ndarray:
    """Karplus-Strong drum (CMJ 7(2):43-55; g15 s4): Y[t] = +(Y[t-p] + Y[t-p-1])/2 with probability
    `blend`, minus that otherwise. blend 1 = the plucked string, 1/2 = a drum (rms halves every 2p+1
    samples), 0 = an octave lower with odd harmonics only. Decay stretching: with probability
    1 - 1/stretch the sample is copied instead of averaged. The table starts as two-level noise.
    Vectorised a period at a time (a sample only depends on samples at least p back)."""
    p = max(2, int(p))
    rng = np.random.default_rng(int(seed))
    y = np.zeros(max(n, p + 2))
    y[:p + 1] = rng.integers(0, 2, p + 1) * 2.0 - 1.0
    for s in range(p + 1, y.shape[0], p):
        j = np.arange(s, min(s + p, y.shape[0]))
        v = 0.5 * (y[j - p] + y[j - p - 1])
        if stretch > 1.0:
            v = np.where(rng.random(j.size) < 1.0 - 1.0 / stretch, y[j - p], v)
        y[j] = np.where(rng.random(j.size) < blend, v, -v)
    return y[:n]


@drum("ks_drum", "Karplus-Strong drum: a noise loop whose sign flips at random; snare-like at blend 0.5, a plucked string at 1, a hollow 'plucked bottle' at 0.",
      decay=(0.25, "seconds to fall 60 dB (sets the loop length: 0.2-0.4 snare, 0.03 brushed tom)"),
      blend=(0.5, "0..1 probability of keeping the sign: 0.5 drum, 1 string, 0 octave-down odd harmonics"),
      stretch=(1.0, ">= 1: decay stretching; on a drum more 'snare'"),
      tone=(6000.0, "Hz lowpass on the output"),
      seed=(0, "int: the random sign sequence"))
def ks_drum(sr=DEFAULT_SR, vel=1.0, decay=0.25, blend=0.5, stretch=1.0, tone=6000.0, seed=0):
    decay, stretch = float(np.clip(decay, 0.01, 4.0)), max(1.0, float(stretch))
    # rms ~ 2^(-t/(2p+1)) -> T60 = (60/6.02) (2p+1)/fs, times the stretch factor
    p = int(np.clip(round((decay / stretch * _KS_FI / 9.966 - 1) / 2), 2, 4000))
    n = int((decay * 1.3 + 0.03) * _KS_FI)
    y = ks_drum_raw(n, p, float(np.clip(blend, 0.0, 1.0)), stretch, seed)
    if sr != _KS_FI:
        g = math.gcd(int(sr), _KS_FI)
        y = resample_poly(y, int(sr) // g, _KS_FI // g)
    y = F.lowpass(y, min(float(tone) * (0.5 + 0.5 * vel), 0.45 * sr), sr)
    y = F.dc_block(y, sr) * perc(y.shape[0] / sr, 0.0005, sr, curve=1.0)
    y[-min(len(y) // 2, samples(0.01, sr)):] *= np.linspace(1, 0, min(len(y) // 2, samples(0.01, sr)))
    return y / (np.max(np.abs(y)) + 1e-9) * 0.8 * (0.6 + 0.4 * vel)


DRUM_GAIN["ks_drum"] = 0.6
