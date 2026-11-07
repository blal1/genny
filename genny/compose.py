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
    "pelog": [0, 1, 3, 7, 8],                 # pelog degrees 1 2 3 5 6 rounded: saron at 0/124/256/668/788 c (Rahn 2016, YorkSpace hdl 10315/32503, ensemble 1)
    "slendro": [0, 2, 5, 7, 9],               # slendro from degree 5 rounded: 0/241/502/699/945 c (Surjodiningrat et al. 1972 Table 1 p. 37, Kyahi Madumurti saron, via Rahn 2022 hdl 10315/39544)
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
    "slendro_equal": [2 ** (i / 5) for i in range(5)],               # octave in 5 equal parts (Ruiz i Carulla, Baschet annexes, citing Sethares p. 306)
}
RATIO_SCALES["7-11"] = RATIO_SCALES["seven_eleven"]
RATIO_SCALES["ikeda_18"] = [2 ** (i / 18) for i in range(18)]
RATIO_SCALES["pythagorean"] = sorted(1.5 ** i / 2 ** math.floor(i * math.log2(1.5)) for i in range(12))
# 5-limit chromatic, used by "tuning": "just" to retune any 12-TET scale: every ratio is in Truax,
# Handbook for Acoustic Ecology, "Just tuning" table (semitone 16:15 ... augmented fourth 45:32, minor seventh 9:5, major seventh 15:8)
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
    """Tonic -> subdominant -> dominant walk that ends on a dominant (turnaround)."""
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


_SIEVE_TOK = re.compile(r"\s*(?:(\d+)@(\d+)|([|&()-]))")


def sieve(expr: str, n: int | None = None):
    """Xenakis sieve: residue classes M@shift joined by | (union), & (intersection), - (difference),
    with parentheses. Returns (steps of 0..n-1 in the sieve, period = lcm of the moduli)."""
    toks, pos, mods = [], 0, []
    expr = str(expr)
    while pos < len(expr.rstrip()):
        m = _SIEVE_TOK.match(expr, pos)
        if not m:
            raise SpecError(f"sieve {expr!r}: write residue classes like 3@0 | 4@1 & (5@2 - 7@0)")
        if m.group(1):
            mods.append((int(m.group(1)), int(m.group(2))))
            toks.append(("c", mods[-1]))
        else:
            toks.append(("o", m.group(3)))
        pos = m.end()
    if not mods or any(mm < 1 for mm, _ in mods):
        raise SpecError(f"sieve {expr!r}: needs at least one class with modulus >= 1")
    period = math.lcm(*[mm for mm, _ in mods])
    n = period if n is None else int(n)

    def expr_(i):                                       # union and difference bind looser than intersection
        s, i = term(i)
        while i < len(toks) and toks[i] in (("o", "|"), ("o", "-")):
            op = toks[i][1]
            t, i = term(i + 1)
            s = s | t if op == "|" else s - t
        return s, i

    def term(i):
        s, i = atom(i)
        while i < len(toks) and toks[i] == ("o", "&"):
            t, i = atom(i + 1)
            s = s & t
        return s, i

    def atom(i):
        if i >= len(toks):
            raise SpecError(f"sieve {expr!r}: incomplete")
        kind, v = toks[i]
        if kind == "c":
            mm, sh = v
            return {x for x in range(n) if x % mm == sh % mm}, i + 1
        if v == "(":
            s, i = expr_(i + 1)
            if i >= len(toks) or toks[i] != ("o", ")"):
                raise SpecError(f"sieve {expr!r}: missing )")
            return s, i + 1
        raise SpecError(f"sieve {expr!r}: unexpected {v!r}")

    s, i = expr_(0)
    if i != len(toks):
        raise SpecError(f"sieve {expr!r}: unexpected {toks[i][1]!r}")
    return sorted(s), period


def tone_row_matrix(row) -> list[list[int]]:
    """12 x 12 matrix of a twelve-tone row: row 0 is P0 (transposed to start on 0), column 0 is I0
    (I = (12 - n) % 12 from the first note); read rows right to left for R, columns bottom-up for RI."""
    row = [int(p) % 12 for p in row]
    if sorted(row) != list(range(12)):
        raise SpecError("a twelve-tone row uses each pitch class 0..11 once")
    p0 = [(p - row[0]) % 12 for p in row]
    return [[(p + (12 - q) % 12) % 12 for p in p0] for q in p0]


def urn(values, n: int, rng) -> list:
    """n draws without replacement, the urn refilled once empty: no value comes back before all the others."""
    vals, out = list(values), []
    if not vals:
        raise SpecError("urn needs at least one value")
    while len(out) < n:
        out += [vals[i] for i in rng.permutation(len(vals))]
    return out[:n]


def select(values, n: int, rng, method: str = "alea", weights=None, group=None) -> list:
    """Koenig's selection principles: series (urn), alea (random choice), ratio (weighted choice),
    group (a random choice repeated `group` times, an int or [lo, hi]), sequence (in order, cycling)."""
    vals = list(values)
    if not vals:
        raise SpecError("select needs at least one value")
    if method == "series":
        return urn(vals, n, rng)
    if method == "alea":
        return [vals[int(i)] for i in rng.integers(0, len(vals), n)]
    if method == "ratio":
        w = np.asarray(weights if weights is not None else [], float)
        if w.shape != (len(vals),) or np.any(w < 0) or w.sum() <= 0:
            raise SpecError("ratio needs one weight >= 0 per value")
        return [vals[int(i)] for i in rng.choice(len(vals), n, p=w / w.sum())]
    if method == "group":
        if group is None:
            raise SpecError("group needs `group`: repetitions, an int or [lo, hi]")
        lo, hi = (group, group) if np.isscalar(group) else group
        out = []
        while len(out) < n:
            out += [vals[int(rng.integers(0, len(vals)))]] * int(rng.integers(int(lo), int(hi) + 1))
        return out[:n]
    if method == "sequence":
        return [vals[i % len(vals)] for i in range(n)]
    raise SpecError("select method: series|alea|ratio|group|sequence")


def _curve(c, x):
    """A constant, or breakpoints [[x, value], ...] with x in 0..1, read at x."""
    if np.isscalar(c):
        return np.full(np.shape(x), float(c))
    pts = np.asarray(c, float)
    return np.interp(x, pts[:, 0], pts[:, 1])


def tendency(lo, hi, n: int, rng) -> np.ndarray:
    """Tendency mask: n uniform draws between two boundaries that move over the span (constants or breakpoints)."""
    x = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(n)
    a, b = _curve(lo, x), _curve(hi, x)
    return np.minimum(a, b) + rng.random(n) * np.abs(b - a)


def elementary_ca(rule: int, width: int, rows: int, init="center") -> np.ndarray:
    """One-dimensional two-state cellular automaton, neighbourhood 3, Wolfram rule number, wrapping row."""
    r = int(rule) % 256
    if init == "center":
        row = np.zeros(width, int)
        row[width // 2] = 1
    else:
        row = np.array([int(ch) for ch in str(init)] * (width // max(1, len(str(init))) + 1), int)[:width]
    out = [row]
    for _ in range(rows - 1):
        k = 4 * np.roll(row, 1) + 2 * row + np.roll(row, -1)
        row = (r >> k) & 1
        out.append(row)
    return np.array(out)


_LSYS = re.compile(r"(\w+)\{([^}]*)\}")


def lsystem(grammar: str, steps: int, rng=None):
    """'a{3}b{-2} @ a{ab} b{a} @ a': symbol values, rewrite rules (alternatives `x|y`, weights `x=3|y=2`), axiom.
    Returns the values of the string after `steps` rewrites."""
    parts = [p.strip() for p in str(grammar).split("@")]
    if len(parts) != 3:
        raise SpecError("lsystem grammar: 'values @ rules @ axiom'")
    vals = {k: float(v) for k, v in _LSYS.findall(parts[0])}
    rules = {}
    for src, dst in _LSYS.findall(parts[1]):
        alts = [a.split("=") for a in dst.split("|")]
        rules[src] = ([a[0] for a in alts], np.array([float(a[1]) if len(a) > 1 else 1.0 for a in alts]))
    s = parts[2]
    syms = sorted(vals, key=len, reverse=True)
    for _ in range(int(steps)):
        out, i = [], 0
        while i < len(s):
            src = next((r for r in sorted(rules, key=len, reverse=True) if s.startswith(r, i)), None)
            if src is None:
                out.append(s[i])
                i += 1
                continue
            alts, w = rules[src]
            j = 0 if len(alts) == 1 or rng is None else int(rng.choice(len(alts), p=w / w.sum()))
            out.append(alts[j])
            i += len(src)
        s = "".join(out)
    res, i = [], 0
    while i < len(s):
        sym = next((k for k in syms if s.startswith(k, i)), None)
        if sym is None:
            raise SpecError(f"lsystem: symbol at {s[i:i + 4]!r} has no value")
        res.append(vals[sym])
        i += len(sym)
    return res


GA_FIT = ((4, 3, 1), (4, 3, 1), (4, 1, 0), (2, 1, 1), (4, 1, 0), (3, 2, 1), (3, 1, 0))


def ga_fitness(ch, fit) -> float:
    """Distance in beats of a rhythm (divisor, multiplier, note 1 / rest 0) to the target: 0 = the target."""
    dur = lambda a: a[1] / a[0]                                                  # noqa: E731
    tot = lambda r, s: sum(dur(a) for a in r if s is None or a[2] == s)          # noqa: E731
    same_v = sum(abs(dur(a) - dur(b)) < 1e-9 for a, b in zip(ch, fit))
    same_a = sum(abs(dur(a) - dur(b)) < 1e-9 and a[2] == b[2] for a, b in zip(ch, fit))
    unit = tot(fit, None) / len(fit)
    return (1.5 * abs(tot(ch, 1) - tot(fit, 1)) + 1.5 * abs(tot(ch, 0) - tot(fit, 0)) + 2.33 * abs(tot(ch, None) - tot(fit, None))
            + 1.0 * (len(fit) - same_a) * unit + 0.66 * (len(fit) - same_v) * unit)


def _ga_mutate(ch, rng):
    ch = [list(a) for a in ch]
    kind = int(rng.integers(5))
    i = int(rng.integers(len(ch)))
    a = ch[i]
    if kind == 0:
        f = int(rng.choice((2, 3)))
        if rng.random() < 0.5 and a[0] % f == 0 and a[1] % f == 0:
            a[0], a[1] = a[0] // f, a[1] // f
        else:
            a[0], a[1] = a[0] * f, a[1] * f
    elif kind in (1, 2):
        j = kind - 1
        a[j] = a[j] + 1 if a[j] == 1 or rng.random() < 0.5 else a[j] - 1
    elif kind == 3:
        a[2] = 1 - a[2]
    else:
        lo, hi = sorted(int(x) for x in rng.integers(len(ch), size=2))
        ch[lo:hi + 1] = ch[lo:hi + 1][::-1]
    return [tuple(x) for x in ch]


_RIFF_TXT = """
Eb4 D4 A4 F4 E4 C5 A4 A4 | F4 A4 Eb5 D5 E4 A4 C5 A4 | Ab4 A4 H G5 H Eb5 C5 E5 | Ab4 A4 B4 C5 Eb5 E5 Ab5 A5
A4 Bb4 B4 C5 Db5 D5 Eb5 E5 | A4 Bb4 B4 C5 E5 Eb5 D5 C5 | A4 B4 C5 A4 B4 C5 D5 B4 | A4 B4 C5 D5 Eb5 E5 Eb5 C5
A4 C5 D5 Eb5 Gb5 Ab5 A5 C6 | A4 C5 Eb5 B4 D5 F5 Eb5 C5 | A4 C5 E5 G5 B5 A5 G5 E5 | A4 C5 E5 A5 G5 Eb5 C5 A4
B4 A4 B4 C5 B4 A4 B4 C5 | B4 A4 B4 C5 B4 C5 B4 A4 | B4 A4 B4 C5 D5 C5 D5 Eb5 | C5 Ab4 A4 G5 F5 Gb5 Eb5 E5
C5 D5 C5 B4 C5 B4 A4 H | C5 D5 Eb5 C5 D5 Eb5 F5 D5 | D5 C5 A4 C5 E5 Eb5 D5 C5 | D5 C5 D5 Eb5 D5 C5 D5 Eb5
D5 Eb5 E5 F5 Gb5 G5 Ab5 A5 | D5 Eb5 G5 Eb5 D5 C5 B4 C5 | D5 Eb5 A5 D5 H C5 A4 E4 | D5 E5 G5 E5 C5 H D5 A5
Eb5 D5 Eb5 D5 H C5 A4 H | Eb5 D5 Eb5 F5 Eb5 D5 C5 B4 | Eb5 E5 D5 C5 B4 A4 Ab4 A4 | Eb5 E5 A5 C5 B4 E5 A4 A4
Eb5 Gb5 E5 A4 B4 D5 C5 E4 | E5 A4 C5 Ab4 B4 G4 Gb4 E4 | E5 Eb5 D5 C5 B4 C5 D5 F5 | G5 E5 D5 B4 Eb5 H C5 A4
G5 E5 D5 Gb5 C5 H A4 H | Ab5 A5 Ab5 A5 Ab5 A5 Ab5 A5 | A5 E5 C5 G4 C5 E5 A5 A5 | A5 E5 C5 A4 G5 Eb5 C5 A4
A5 B5 G5 E5 F5 Gb5 G5 Ab5 | B5 C6 A5 E5 G5 B5 A5 H | B5 D6 C6 E5 Ab5 B5 A5 C5 | C6 B5 A5 G5 Gb5 E5 Eb5 C5
"""
_PC = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
RIFFS = [[None if n == "H" else 12 * (int(n[-1]) + 1) + _PC[n[:-1]] for n in r.split()]
         for r in _RIFF_TXT.replace("\n", " | ").split("|") if r.strip()]
RIFF_TRIES = 3
RIFF_BIV = (28, 0, 7, 0, 14, 0, 7, 4)


def riff_energy(i: int, nbars: int) -> float:
    """Chance (%) that a riff note is played: high at the start, lowest in the middle third, rising to the end."""
    if 3 * i < nbars:
        return 100 - 90 * i / nbars
    if 3 * i > 2 * nbars:
        return 40 + 90 * i / nbars
    return 70.0


def riffology(nbars: int, rng, meter: float = 4.0, transpose: float = 0.0, vel: float = 0.5) -> list:
    """Riffs of 8 notes strung end to end: each bar plays 1, 2 or 4 riffs (a random walk of the speed), each riff
    the best of three draws, the one starting nearest the last note (a repeat counts as a tritone); a note is
    skipped when energy + beat weight < U(0, 100), and then becomes a rest or a hold."""
    out, last, tempo, t = [], None, 1, 0.0
    for b in range(nbars):
        if tempo > rng.integers(3):
            tempo -= 1
        elif tempo < rng.integers(3):
            tempo += 1
        rpb = 1 << tempo
        d = meter / (8 * rpb)
        en = riff_energy(b, nbars)
        for _ in range(rpb):
            best, bd = None, 999
            for _ in range(RIFF_TRIES):
                r = RIFFS[int(rng.integers(len(RIFFS)))]
                if last is None:
                    best = r
                    break
                dn = abs(last - (r[0] if r[0] is not None else last)) or 6
                if dn < bd:
                    best, bd = r, dn
            for i, n in enumerate(best):
                if n is not None and en + RIFF_BIV[i] < rng.integers(100):
                    n = "rest" if rng.random() < 0.5 else None
                if n is None:
                    if out and out[-1]["t"] + out[-1]["dur"] >= t - 1e-9:
                        out[-1]["dur"] += d
                elif n != "rest":
                    out.append(_ev(t, d, n + transpose, vel))
                    last = n
                t += d
    return out


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    while n > 0x7F:
        n >>= 7
        out.append(0x80 | (n & 0x7F))
    return bytes(reversed(out))


def write_midi(events, path, bpm: float = 120.0, ppq: int = 480, channel: int = 0) -> bytes:
    """Standard MIDI File, format 0, one track: events {"t", "dur"} in beats, "pitch" MIDI, "vel" 0..1.
    Writes ``path`` when given and returns the bytes."""
    msgs = []
    for e in events:
        k, v = int(round(e["pitch"])), max(1, min(127, int(round(127 * float(e.get("vel", 0.5))))))
        on, off = int(round(e["t"] * ppq)), int(round((e["t"] + e["dur"]) * ppq))
        msgs += [(off, 0, bytes((0x80 | channel, k, 0))), (on, 1, bytes((0x90 | channel, k, v)))]
    us = int(round(60e6 / bpm))
    trk, now = bytearray(b"\x00\xff\x51\x03" + us.to_bytes(3, "big")), 0
    for tick, _, m in sorted(msgs, key=lambda m: m[:2]):
        trk += _vlq(tick - now) + m
        now = tick
    trk += b"\x00\xff\x2f\x00"
    data = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + ppq.to_bytes(2, "big")
    data += b"MTrk" + len(trk).to_bytes(4, "big") + bytes(trk)
    if path:
        with open(path, "wb") as f:
            f.write(data)
    return data


def ga_rhythm(fit, gens: int, rng, n: int = 20, pc: float = 0.8, pm: float = 0.18, elite: float = 0.16) -> list:
    """Best rhythm of each generation of a genetic algorithm that evolves random rhythms toward `fit`."""
    fit = [tuple(int(v) for v in a) for a in fit]
    pop = [[fit[int(i)] for i in rng.integers(len(fit), size=len(fit))] for _ in range(n)]
    best = []
    for _ in range(int(gens)):
        f = np.array([ga_fitness(ch, fit) for ch in pop])
        order = np.argsort(f, kind="stable")
        best.append(pop[order[0]])
        w = f.max() - f + 1e-9
        w = w / w.sum()
        new = [pop[i] for i in order[:int(round(elite * n))]]
        while len(new) < n:
            a, b = (pop[int(i)] for i in rng.choice(n, size=2, p=w))
            if rng.random() < pc and len(fit) > 2:
                i, j = sorted(int(x) for x in rng.choice(np.arange(1, len(fit)), size=2, replace=False))
                a, b = a[:i] + b[i:j] + a[j:], b[:i] + a[i:j] + b[j:]
            new += [_ga_mutate(x, rng) if rng.random() < pm else list(x) for x in (a, b)]
        pop = new[:n]
    return best


def monte_carlo(n: int, size: int, rng, min_step: int = 1, max_step: int = 3, start: int = 0) -> list[int]:
    """Indices into a scale of `size` pitches: each next one drawn uniformly among those whose distance
    to the previous one lies in [min_step, max_step] (rejection sampling without the retries)."""
    if not 0 <= min_step <= max_step or min_step >= size:
        raise SpecError("montecarlo needs 0 <= min_step <= max_step < scale size")
    out = [int(np.clip(start, 0, size - 1))]
    while len(out) < n:
        ok = [j for j in range(size) if min_step <= abs(j - out[-1]) <= max_step]
        if not ok:
            raise SpecError(f"montecarlo: no scale degree lies {min_step}..{max_step} steps from degree {out[-1]} of {size}")
        out.append(ok[int(rng.integers(0, len(ok)))])
    return out[:n]


MOOD_POINTS = {"calm": (0.6, -0.6, 0.05), "fun": (0.8, 0.5, 0.1), "fastpaced": (0.3, 0.9, 0.4),
               "anticipation": (0.0, 0.3, 0.7), "threat": (-0.7, 0.4, 0.8), "horror": (-0.9, 0.2, 0.9),
               "sad": (-0.6, -0.5, 0.2), "triumphant": (0.9, 0.7, 0.15)}


def mood_params(valence: float, arousal: float, tension: float) -> dict:
    """Mood (valence -1..1, arousal -1..1, tension 0..1) -> tempo, root (MIDI), scale, brightness (Hz),
    detune between voices (semitones), decorative events per minute, heart rate (bpm)."""
    v, a, T = float(valence), float(arousal), float(tension)
    if not (-1 <= v <= 1 and -1 <= a <= 1 and 0 <= T <= 1):
        raise SpecError("mood: valence and arousal in -1..1, tension in 0..1")
    scale = "chromatic" if T > 0.75 else "major" if v > 0.3 else "minor" if v > -0.3 else "phrygian"
    return {"bpm": 72 + 48 * a, "root": 45 + 7 * v - 8 * T, "scale": scale,
            "brightness": 600 + 2600 * min(1.0, max(0.0, 0.5 + 0.35 * v + 0.25 * a)),
            "detune": 0.03 + 0.35 * T + 0.15 * max(0.0, -v), "events_per_min": 2 + 10 * max(0.0, a) + 4 * T,
            "heart_bpm": 60 + 30 * max(0.0, a) + 40 * T}


def tension_step(tension: int) -> int:
    """One update of a decaying tension 0..100: minus (tension + 20) // 10 from 2 up, minus 1 at 1."""
    t = int(tension)
    if t >= 2:
        return max(0, t - (t + 20) // 10)
    return t - 1 if t >= 1 else t


def condition_met(cond: dict, value) -> bool:
    """{"type": "equal|greater|less|range", "value": v or [lo, hi]} against a game value."""
    kind, v = str(cond.get("type", "equal")).lower(), cond.get("value")
    if kind == "equal":
        return value == v
    if kind == "greater":
        return value > v
    if kind == "less":
        return value < v
    if kind == "range":
        return v[0] <= value <= v[1]
    raise SpecError("condition type: equal|greater|less|range")


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


def _pink_white(n: int, rng, white: float = 0.5) -> np.ndarray:
    """Zero-mean, unit-sd mix of 1/f (Voss, enough generators to span the sequence) and white noise;
    ``white`` is the white share of the variance."""
    if n < 1:
        return np.zeros(0)
    p = voss(n, rng, generators=max(4, int(np.ceil(np.log2(max(n, 2))))))
    p = (p - p.mean()) / (p.std() + 1e-12) if n > 1 else np.zeros(1)
    w = rng.standard_normal(n)
    return np.sqrt(1.0 - white) * p + np.sqrt(white) * w


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


def gamma_noise(n: int, gamma: float, rng) -> np.ndarray:
    """1/f^gamma sequence in 0..1 (gamma 0 white, 1 pink, 2 brown), from `contact.fractal_noise` run at
    one sample per step with its lowest pole at 1/n."""
    from .contact import fractal_noise
    if n < 2:
        return np.full(max(n, 0), 0.5)
    x = fractal_noise(n, gamma, 1.0, f_low=1.0 / n, seed=int(rng.integers(2**31)))
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


HENNIG_BETA, HENNIG_MS = 1.0, 15.0
ARMA_HIHAT_AR = (-0.5, 0.48, 0.19, 0.03)
ARMA_HIHAT_MA = 0.93


def arma_hihat(n: int, rng) -> np.ndarray:
    """Unit-sd timing deviations from the ARIMA(4,0,1) fitted to a drummer's 16th-note hi-hat
    (AR -0.5, 0.48, 0.19, 0.03; MA 0.93); 200 burn-in steps are dropped."""
    e = rng.standard_normal(n + 200)
    x = lfilter([1.0, ARMA_HIHAT_MA], np.r_[1.0, -np.asarray(ARMA_HIHAT_AR)], e)[200:]
    return (x - x.mean()) / (x.std() + 1e-12) if n > 1 else np.zeros(n)


def poisson_grains(dur: float, density: float, rng, grain=(0.06, 0.12)) -> list[tuple[float, float]]:
    """Stochastic grain cloud: exponential inter-onset times (Poisson, `density` grains per second),
    each grain lasting uniform in `grain` seconds. Returns (onset, length) pairs in seconds."""
    out, t = [], 0.0
    lam = max(float(density), 1e-6)
    while True:
        t += rng.exponential(1.0 / lam)
        if t >= dur:
            return out
        out.append((t, float(rng.uniform(*grain))))


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
    """Metric weight of a step: bar start and half bar 1, beats 0.8, half beats 0.5, the rest 0.12.
    Order after Colley & Dean 2019 (PLoS ONE e0216088): drummers accent 16ths 1 and 9 > 3, 5, 7... > the rest."""
    if s == 0 or (_strong(c, s) and c.meter == 4):
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
    if mode == "riff":
        tr = (c.kpc - 9) % 12
        tr = tr - 12 if tr > 6 else tr
        tr += 12 * round((0.5 * (c.lo + c.hi) - 72 - tr) / 12)
        t0 = sec.bar0 * c.meter
        return [{**e, "t": e["t"] + t0, "vel": float(L.get("vel", 0.5))}
                for e in riffology(sec.nbars, rng, c.meter, tr)]
    P = _pitches(c.kpc, c.offs, c.lo, c.hi)
    if len(P) < 3:
        raise SpecError("range too narrow for the scale")
    if mode == "ga":
        t, end, out, j = sec.bar0 * c.meter, (sec.bar0 + sec.nbars) * c.meter, [], len(P) // 2
        for ch in ga_rhythm(L.get("fit", GA_FIT), int(L.get("generations", 25)), rng):
            for dv, ml, state in ch:
                d = ml / dv
                if t + d > end + 1e-9:
                    return out
                if state:
                    j = _reflect(j + int(rng.integers(-2, 3)), len(P))
                    out.append(_ev(t, d * float(L.get("gate", 0.92)), P[j], 0.7))
                t += d
        return out
    ph = max(1, min(int(L.get("phrase", 2)), sec.nbars))
    snap = bool(L.get("snap", True))
    p_near, p_chord = float(L.get("p_near", 0.75)), float(L.get("p_chord", 0.3))
    p_repeat = float(L.get("p_repeat", 0.0))
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
        ford = float(np.clip(L.get("order", 1), 0, 4))
        o_lo, frac = int(math.floor(ford)), ford - math.floor(ford)
        mot = [o for o, _ in (_parse_motif(c, L["motif"], 0.5) if L.get("motif") is not None
                              else [(int(v), 0.5) for v in np.cumsum(rng.integers(-2, 3, 8))])]
        tables = {}
        for order in {o_lo, min(4, o_lo + 1)}:
            table = tables[order] = {}
            for i in range(len(mot)):                   # trained cyclically so no context is a dead end
                table.setdefault(tuple(mot[(i + j) % len(mot)] for j in range(order)), []).append(mot[(i + order) % len(mot)])
        seq = list(mot[:o_lo + (1 if frac > 0 else 0)])
        while len(seq) < n:
            order = o_lo + 1 if frac > 0 and rng.random() < frac else o_lo      # fractional order: the two neighbours
            seq.append(int(rng.choice(tables[order][tuple(seq[len(seq) - order:])])))
        raw = [P[_reflect(mid + o, len(P))] for o in seq[:n]]
    elif mode == "montecarlo":
        idx = monte_carlo(n, len(P), rng, int(L.get("min_step", 1)), int(L.get("max_step", 3)), mid)
        raw = [P[i] for i in idx]
    elif mode == "ca":
        width = int(L.get("cells", 20))
        rows = elementary_ca(int(L.get("rule", 90)), width, max(2, n), L.get("init", "center"))
        act = [int(i) for r in rows for i in np.flatnonzero(r)] or [0]
        raw = [P[_reflect(mid - width // 2 + a, len(P))] for a in act[:n]]
    elif mode == "lsystem":
        g = L.get("grammar", "a{3}b{-2} @ a{ab} b{a} @ a")
        k, vals = 1, []
        while len(vals) < n and k < 40:
            vals = lsystem(g, k, rng)
            k += 1
        steps = (list(vals) * (n // max(1, len(vals)) + 1))[:n]
        raw, j = [], mid
        for s in steps:                                 # values are intervals in scale steps
            j = _reflect(j + int(round(s)), len(P))
            raw.append(P[j])
    elif mode == "koenig":
        raw = select(P, n, rng, L.get("select", "series"), L.get("weights"), L.get("group"))
    elif mode == "tendency":
        lo, hi = L.get("mask_lo", c.lo), L.get("mask_hi", c.hi)
        raw = [_near(P, x) for x in tendency(lo, hi, n, rng)]
    elif mode == "logistic":
        r = float(L.get("r", 3.9))
        if not 0.0 < r <= 4.0:
            raise SpecError("logistic r: 0 < r <= 4")
        x, raw = float(L.get("x0", rng.uniform(0.05, 0.95))), []
        for _ in range(n):
            raw.append(P[int(round(x * (len(P) - 1)))])
            x = r * x * (1.0 - x)
    elif mode == "row":
        M = tone_row_matrix(L.get("row") or list(rng.permutation(12)))
        forms = [M[0], M[0][::-1], [r[0] for r in M], [r[0] for r in M][::-1]]       # P0, R0, I0, RI0
        pcs = [pc for f in forms for pc in f]
        raw = []
        for i in range(n):
            pc = (pcs[i % len(pcs)] + c.kpc) % 12
            same = [p for p in P if abs((p - pc) % 12) < 1e-6]
            raw.append(_near(same or P, raw[-1] if raw else P[mid]))
    elif mode == "voss":
        g = L.get("gamma")
        seq = voss(n, rng) if g is None else gamma_noise(n, float(g), rng)
        raw = [P[int(round(v * (len(P) - 1)))] for v in seq]
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
        raise SpecError("melody mode: ga|riff|walk|markov|voss|collatz|logistic|row|montecarlo|ca|lsystem|koenig|tendency|motif|call_response")
    vel = _contour(n, rng)
    out, cur = [], None
    for i, (b, s) in enumerate(onsets):
        ch = sec.chords[b]
        T = _tones(c, ch) or P
        here = b * c.S + s
        nxt = onsets[i + 1][0] * c.S + onsets[i + 1][1] if i + 1 < n else sec.nbars * c.S
        phrase_end = i + 1 == n or onsets[i + 1][0] // ph != b // ph
        target = c.lo + (c.hi - c.lo) * shape(((b % ph) * c.S + s) / (ph * c.S))
        rep = None
        if raw is not None:
            prop = raw[i]
        elif cur is None:
            prop = _near(P, target)
        elif p_repeat and rng.random() < p_repeat:
            prop = rep = cur
        else:
            j = P.index(cur)
            toward = 1 if target > cur else -1
            d = toward if rng.random() < 0.7 else -toward
            prop = P[_reflect(j + d * (1 if rng.random() < p_near else int(rng.integers(2, 5))), len(P))]
        if i + 1 == n and sec.index == len(c.sections) - 1:
            prop = _near([p for p in T if abs((p - c.kpc - ch["root"]) % 12) < 1e-6] or T, prop)   # home: the root
        elif phrase_end or (snap and _strong(c, s)) or (raw is None and rng.random() < p_chord):
            if rep is None or rep not in T:
                prop = _near([p for p in T if p != cur] or T, prop)   # a chord tone, but keep the line moving
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
    lv = dict(_LEVEL)
    if L.get("accent") is not None:                     # accent X : hit x ratio; ghosts keep their ratio to x
        r = float(L["accent"])
        if r < 1:
            raise SpecError("accent >= 1 (X : x)")
        lv = {"X": 1.0, "x": 1.0 / r, "?": 1.0 / r, "o": _LEVEL["o"] / _LEVEL["x"] / r}
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
                out.append(_ev(t0 + s * c.grid, 0.0, None, lv[ch], drum=key))
        for s in range(zone, c.S):
            if rng.random() < 0.75:
                x = (s - zone) / max(1, c.S - zone - 1)
                out.append(_ev(t0 + s * c.grid, 0.0, None, 0.5 + 0.5 * x, drum=fill_b if x > 0.5 else fill_a, fill=True))
    dip = L.get("accent_dip")
    if dip:
        dip = ACCENT_DIP_DB if dip is True else float(dip)
        by = {}
        for e in sorted(out, key=lambda e: e["t"]):
            by.setdefault(e["drum"], []).append(e)
        for seq in by.values():
            for a, b in zip(seq, seq[1:]):
                if b["vel"] >= lv["X"] - 1e-9 > a["vel"]:
                    a["vel"] *= 10 ** (-dip / 20.0)
    return out


_GEN = {"melody": _gen_melody, "bass": _gen_bass, "chords": _gen_chords, "arp": _gen_arp, "drums": _gen_drums,
        "counter": _gen_counter, "pad": lambda c, s, r, L: _gen_chords(c, s, r, L, pad=True)}


# ============================================================================ performance
def swing_auto(pair_s: float) -> float:
    """`swing` for a long-short pair lasting `pair_s` seconds: the short note stays at 100 ms, the
    ratio between 1:1 and 3.5:1 (2:1 at 200 bpm eighths, 1:1 at 300)."""
    r = float(np.clip((pair_s - 0.1) / 0.1, 1.0, 3.5))
    return (r - 1.0) / (r + 1.0)


def soloist_delay(bpm: float) -> float:
    """Jazz soloist's lag behind the ride cymbal on the beats, in ms (the off-beats stay together)."""
    return max(0.0, 107.2 - 0.306 * float(bpm))


PHRASE_LENGTHEN_NOTES, PHRASE_LENGTHEN_LAST = 3, 0.20
NMA_MS = {"musician": 30.0, "novice": 50.0}
ENSEMBLE_SD = (0.03, 30.0, 50.0)
ACCENT_DIP_DB = 2.0
EMOTIONS = {
    "angry": {"rules": {"duration_contrast": 1.0, "high_loud": 1.0, "leap_articulation": 1.0, "repetition": 1.0,
                        "phrase_arch": -1.0}, "tempo": 1.15, "level_db": 3.0, "articulation": 0.75},
    "happy": {"rules": {"duration_contrast": 1.0, "leap_articulation": 1.0, "repetition": 1.0, "faster_uphill": 1.0},
              "tempo": 1.15, "level_db": 3.0, "articulation": 0.75},
    "sad": {"rules": {"duration_contrast": -1.0, "phrase_arch": 1.0, "kth_phrase": 1.0},
            "tempo": 0.85, "level_db": -3.0, "articulation": 1.0},
    "tender": {"rules": {"duration_contrast": -1.0, "phrase_arch": 1.0, "kth_phrase": 1.0, "high_loud": -1.0},
               "tempo": 0.85, "level_db": -3.0, "articulation": 1.0},
}


def ensemble_sigma_ms(ioi_ms: float) -> float:
    """Spread (sd, ms) of the onsets of an ensemble: 3 % of the IOI, held within 30-50 ms."""
    r, lo, hi = ENSEMBLE_SD
    return float(np.clip(r * float(ioi_ms), lo, hi))


ELLIS_DELAY = ((90.0, 40.0), (120.0, 47.0), (150.0, 53.0), (180.0, 54.0), (210.0, 63.0))
FEEL_BASS_MS = 10.0
RIDE_RATIO_REF = (132.0, 2.4)
RIDE_STRAIGHT_BPM = 300.0


def ellis_delay(bpm: float) -> float:
    """Soloist's lag on the beats in ms, measured over 90-210 bpm (held flat outside)."""
    b, d = zip(*ELLIS_DELAY)
    return float(np.interp(float(bpm), b, d))


def ride_ratio(bpm: float) -> float:
    """Long:short ratio of the ride cymbal's swung eighths: 2.4:1 at 132 bpm, straight (1:1) at 300,
    linear in between and beyond, capped at 3.5:1."""
    b0, r0 = RIDE_RATIO_REF
    return float(np.clip(r0 + (1.0 - r0) * (float(bpm) - b0) / (RIDE_STRAIGHT_BPM - b0), 1.0, 3.5))


def ritard_map(rit, total: float, meter: float = 4.0, end: float | None = None):
    """Final ritardando over the `beats` (default one bar) before `end`, the onset of the last note
    (default `total`): tempo v(x) = [1 + (v_end^q - 1) x]^(1/q), x from 0 to 1 across the span, then
    v_end. Returns the nominal-beat -> played-beat map, or None. `rit`: a number of beats or
    {"beats", "v_end", "q"}."""
    if not rit:
        return None
    rit = rit if isinstance(rit, dict) else {"beats": rit}
    end = float(rit.get("end", total if end is None else end))
    span = float(rit.get("beats", meter))
    if rit.get("model") == "kth":
        span = min(span, end)
        if span <= 0:
            return None
        s0 = end - span
        return lambda b: b if b <= s0 else (s0 + span * 2.0 * (math.sqrt(1.1) - math.sqrt(1.1 - min((b - s0) / span, 1.0)))
                                            + max(0.0, b - end) / math.sqrt(0.1))
    q, v_end = float(rit.get("q", 3.0)), float(rit.get("v_end", 0.3))
    if span <= 0 or q <= 0 or not 0 < v_end <= 1:
        raise SpecError("ritard needs beats > 0, q > 0 and 0 < v_end <= 1")
    span = min(span, end)
    if span <= 0:
        return None
    total = end
    s0, k = total - span, v_end ** q - 1.0

    def f(b):
        if b <= s0:
            return b
        x = min((b - s0) / span, 1.0)
        if abs(k) < 1e-12:
            u = x
        elif abs(q - 1.0) < 1e-12:
            u = math.log1p(k * x) / k
        else:
            u = q / ((q - 1.0) * k) * ((1.0 + k * x) ** ((q - 1.0) / q) - 1.0)
        return s0 + span * u + max(0.0, b - total) / v_end
    return f


def _ioi_warp(ev, q, fn):
    """Re-time the onsets (chords together) so each inter-onset interval in seconds becomes fn(ioi)."""
    on = sorted({e["t"] for e in ev})
    if len(on) < 2:
        return
    new = np.concatenate(([on[0]], on[0] + np.cumsum(np.maximum(fn(np.diff(on) * q), 0.0) / q)))
    m = lambda b: float(np.interp(b, on, new)) if b <= on[-1] else new[-1] + b - on[-1]   # noqa: E731
    for e in ev:
        t1 = m(e["t"] + e["dur"])
        e["t"] = m(e["t"])
        e["dur"] = max(0.0, t1 - e["t"])


def _kth(ev, c, L):
    """KTH rules on the score, chords as one tone (top pitch): high loud, high sharp, double duration, duration
    contrast, faster uphill, articulation of repetition."""
    k = float(L.get("high_loud", 0.0))
    if k:
        for e in ev:
            if e["pitch"] is not None:
                e["vel"] *= 10 ** ((e["pitch"] - 60.0) / 4.0 * k / 20.0)
    written = {id(e): e["pitch"] for e in ev}
    k = float(L.get("high_sharp", 0.0))
    if k:
        for e in ev:
            if e["pitch"] is not None:
                e["pitch"] += k * 0.04 * (e["pitch"] - 60.0) / 12.0
    on = sorted({e["t"] for e in ev})
    if len(on) < 2:
        return
    grp = [[e for e in ev if e["t"] == t] for t in on]
    top = [max((written[id(e)] for e in g if e["pitch"] is not None), default=None) for g in grp]
    ioi = 1000.0 * c.q * np.diff(on)
    last = 1000.0 * c.q * max(e["dur"] for e in grp[-1])
    d = np.zeros(ioi.size)
    dbl = set()
    k = float(L.get("double_duration", 0.0))
    if k:
        nxt = np.append(ioi[1:], last)
        for i in range(1, ioi.size):
            if ioi[i] < 1000.0 and abs(2.0 * ioi[i] - ioi[i - 1]) < 1e-6 and nxt[i] > ioi[i]:
                x = 0.12 * ioi[i] * k
                d[i] += x
                d[i - 1] -= x
                dbl |= {i - 1, i}
    k = float(L.get("duration_contrast", 0.0))
    if k:
        for i in range(ioi.size):
            if i in dbl:
                continue
            x = k * float(np.interp(ioi[i], *_DDC_DDR))
            d[i] += max(x, min(0.0, 100.0 - ioi[i]))
            for e in grp[i]:
                e["vel"] *= 10 ** (k * float(np.interp(ioi[i], *_DDC_DL)) / 20.0)
    k = float(L.get("faster_uphill", 0.0))
    if k:
        for i in range(ioi.size):
            if top[i] is not None and top[i + 1] is not None and top[i + 1] > top[i]:
                d[i] -= 2.0 * k
    k = float(L.get("inegales", 0.0))
    if k:
        i = 0
        while i < ioi.size - 1:
            if abs(ioi[i] - ioi[i + 1]) < 1e-6:
                pos = on[i] * 1000.0 * c.q / ioi[i]
                x = 0.22 * ioi[i] * k
                if abs(pos - round(pos)) < 1e-6 and round(pos) % 2 == 0:
                    d[i] += x
                    d[i + 1] -= x
                    i += 2
                    continue
                if i == 0 or abs(ioi[i - 1] - ioi[i]) > 1e-6:
                    d[i] -= x
            i += 1
    k = float(L.get("leap_duration", 0.0))
    if k:
        step = lambda a, b: a is not None and b is not None and abs(b - a) <= 2.0 + 1e-6   # noqa: E731
        for i in range(1, ioi.size):
            if top[i] is None or top[i + 1] is None or top[i - 1] is None:
                continue
            dn = top[i + 1] - top[i]
            if abs(dn) < 3.0 - 1e-6 or not step(top[i - 1], top[i]) or (i + 2 < len(on) and not step(top[i + 1], top[i + 2])):
                continue
            s = math.sqrt(abs(dn)) * k
            d[i] += (-4.2 if dn > 0 else 2.4) * s
            if i + 1 < ioi.size:
                d[i + 1] += (4.2 if dn > 0 else -4.2) * s
    k = float(L.get("harmonic_charge", 0.0))
    chords = getattr(c, "chords", None)
    if k and chords and len(on) > 1:
        ci = [min(int(t // c.meter), len(chords) - 1) for t in on]
        charge = []
        for j in ci:
            p = chords[j].get("pcs") or []
            charge.append(max(0.0, 2.0 * (_CMEL[int(round(p[0])) % 12] / 2 + _CMEL[int(round(p[1])) % 12] / 3
                                          + _CMEL[int(round(p[2])) % 12] / 6) - 3.0) if len(p) >= 3 else 0.0)
        first = [0] + [i for i in range(1, len(on)) if ci[i] != ci[i - 1] and chords[ci[i]].get("pcs") != chords[ci[i - 1]].get("pcs")]
        sec = [t * c.q for t in on]
        bx, by = [], []
        for a, b in zip(first, first[1:] + [None]):
            la = 1.5 * math.sqrt(charge[a]) * k
            if b is None:
                bx += [sec[a]]
                by += [la]
                break
            lb = 1.5 * math.sqrt(charge[b]) * k
            bx += [sec[a]] + ([max(sec[a], sec[b] - 1.9)] if lb >= la else [])
            by += [la] + ([la] if lb >= la else [])
        dl = np.interp(sec, bx, by)
        for i, g in enumerate(grp):
            for e in g:
                e["vel"] *= 10 ** (dl[i] / 20.0)
            if i < ioi.size:
                d[i] += ioi[i] * (math.sqrt(1.0 + 0.018 * dl[i]) - 1.0) + 2.0 * math.sqrt(charge[i]) * k
                if i in first:
                    d[i] += 8.0 * math.sqrt(charge[i]) * k
    k = float(L.get("melodic_charge", 0.0))
    chords = getattr(c, "chords", None)
    root_at = lambda i: c.kpc + chords[min(int(on[i] // c.meter), len(chords) - 1)]["root"]   # noqa: E731
    if k and chords:
        cms = [None if top[i] is None else _CMEL[int(round(top[i] - root_at(i))) % 12] for i in range(len(grp))]
        dl0 = [None if x is None else 0.2 * x * k for x in cms]
        dl, dur = list(dl0), np.append(ioi, last)
        for n in range(1, len(grp)):
            m = n - 1
            if (dl0[n] is None or dl0[m] is None or not 1.0 - 1e-6 <= abs(top[n] - top[m]) <= 2.0 + 1e-6
                    or abs(dur[m] - dur[n]) > 1e-6 or dur[m] >= 500.0 or not dl0[m] < 0.5 * dl0[n]):
                continue
            dl[m] = 0.75 * dl0[n]
            if n + 1 < len(grp) and dl0[n + 1] is not None and dl0[n + 1] < 0.5 * dl0[n]:
                dl[n + 1] = 0.55 * dl0[n]
        for i, g in enumerate(grp):
            if cms[i] is None:
                continue
            for e in g:
                e["vel"] *= 10 ** (dl[i] / 20.0)
            if i < ioi.size:
                d[i] += ioi[i] * 2.0 / 3.0 * cms[i] * k / 100.0
    k = float(L.get("melodic_intonation", 0.0))
    if k and chords:
        for i, g in enumerate(grp):
            if top[i] is None:
                continue
            ic = int(round(top[i] - root_at(i))) % 12
            lead = (0 < i < len(grp) - 1 and top[i - 1] is not None and top[i + 1] is not None
                    and abs(top[i] - top[i - 1] - 1.0) > 1e-6 and abs(top[i + 1] - top[i] - 1.0) < 1e-6)
            cents = -_MEL_INT[ic] if lead and ic in (1, 3, 8) else _MEL_INT[ic]
            for e in g:
                if written[id(e)] == top[i]:
                    e["pitch"] += cents * k / 100.0
    k = float(L.get("harmonic_intonation", 0.0))
    if k and chords:
        for i, g in enumerate(grp):
            for e in g:
                if written[id(e)] is not None:
                    e["pitch"] += _HARM_INT[int(round(written[id(e)] - root_at(i))) % 12] * k / 100.0
    k = float(L.get("chromatic_charge", 0.0))
    tones = [i for i in range(len(grp)) if top[i] is not None]
    if k and len(tones) > 1:
        # chromatic charge of each tone from the interval to the next, averaged over five tones (centred)
        cc = np.array([((32 - abs(top[j] - top[i])) % 12) - 6 for i, j in zip(tones, tones[1:])] + [np.nan])
        for m, i in enumerate(tones):
            w = cc[max(0, m - 2):m + 3]
            cm = float(np.nanmean(w)) if np.isfinite(w).any() else 0.0
            for e in grp[i]:
                e["vel"] *= 10 ** (1.35 * cm * k / 20.0)
            if i < ioi.size:
                d[i] += ioi[i] * 0.009 * cm * k / 100.0
    k = float(L.get("kth_phrase", 0.0))
    ends = []
    if k:
        ends = [i for i in range(len(on) - 1) for sct in getattr(c, "sections", [])
                if on[i] < (sct.bar0 + sct.nbars) * c.meter <= on[i + 1]]
        for i in ends:
            d[i] += 40.0 * k
    k = float(L.get("phrase_arch", 0.0))
    if k:
        bounds = sorted({0.0, *(s.bar0 * c.meter for s in getattr(c, "sections", [])), on[-1] + 1e-9})
        for a, b in zip(bounds, bounds[1:]):
            idx = [i for i in range(ioi.size) if a <= on[i] < b]
            if len(idx) < 2:
                continue
            x = (np.array([on[i] for i in idx]) - a) / (b - a)
            p = float(L.get("phrase_turn", 0.7))
            m = 0.1 * k * ((x - p) / np.where(x < p, p, 1.0 - p)) ** 2
            m -= np.average(m, weights=ioi[idx])
            for j, i in enumerate(idx):
                d[i] += ioi[i] * m[j]
                for e in grp[i]:
                    e["vel"] /= 1.0 + m[j]
    k = float(L.get("phrase_lengthen", 0.0))
    if k:
        pe = sorted({i for i in range(len(on) - 1) for sct in getattr(c, "sections", [])
                     if on[i] < (sct.bar0 + sct.nbars) * c.meter <= on[i + 1]} | {ioi.size - 1})
        for i in pe:
            for m in range(1, PHRASE_LENGTHEN_NOTES + 1):
                j = i - PHRASE_LENGTHEN_NOTES + m
                if j >= 0:
                    d[j] += ioi[j] * PHRASE_LENGTHEN_LAST * k * (m / PHRASE_LENGTHEN_NOTES) ** 2
    if d.any():
        _ioi_warp(ev, c.q, lambda x: x + d / 1000.0)
    k = float(L.get("repetition", 0.0))
    if k:
        for i in range(len(on) - 1):
            if top[i] is not None and top[i] == top[i + 1]:
                gap = grp[i + 1][0]["t"] - grp[i][0]["t"] - 0.035 * k / c.q
                for e in grp[i]:
                    e["dur"] = max(0.0, min(e["dur"], gap))
    k = float(L.get("leap_articulation", 0.0))
    if k:
        borders = {s.bar0 * c.meter for s in getattr(c, "sections", [])}
        for i in range(ioi.size):
            if top[i] is None or top[i + 1] is None or ioi[i] < 100.0 or any(on[i] < b <= on[i + 1] for b in borders):
                continue
            dn = min(abs(top[i + 1] - top[i]), 9.0)
            if dn < 3.0 - 1e-6:
                continue
            dro = 8.0 * dn * (0.3 * (ioi[i] - 100.0) / 600.0 + 0.3) * k
            gap = grp[i + 1][0]["t"] - grp[i][0]["t"] - dro / 1000.0 / c.q
            for e in grp[i]:
                e["dur"] = max(0.0, min(e["dur"], gap))
    k = float(L.get("kth_phrase", 0.0))
    if k:
        for i in ends:
            gap = grp[i + 1][0]["t"] - grp[i][0]["t"] - 0.08 * k / c.q
            for e in grp[i]:
                e["dur"] = max(0.0, min(e["dur"], gap))
        for e in grp[-1]:
            e["dur"] += 0.08 * k / c.q


def _rmd(ev, c, L):
    """Power-law change of expressive timing: IOI' = e^a IOI^b, total kept; IOIs longer than `rmd_unit`
    beats are cut in equal units first."""
    b = L.get("rmd")
    if b is None:
        return
    if b == "auto":
        ref = L.get("rmd_ref_bpm") or (sum(c.style.bpm) / 2.0 if getattr(c, "style", None) else 60.0 / c.q)
        b = float(np.clip(1.0 + 0.8 * (float(ref) * c.q / 60.0 - 1.0), 0.8, 1.2))
    b, u = float(b), float(L.get("rmd_unit", 0.5))

    def f(x):
        m = np.maximum(np.round(x / (u * c.q)), 1.0)
        y = m * (x / m) ** b
        return y * x.sum() / y.sum()
    _ioi_warp(ev, c.q, f)


_CMEL = (0.0, 6.5, 2.0, 4.5, 4.0, 2.5, 6.0, 1.0, 5.5, 3.0, 3.5, 5.0)
_MEL_INT = (0.0, -7.0, 3.0, -4.0, 4.0, 2.0, 10.0, 1.0, -6.0, 4.0, -4.0, 9.0)
_HARM_INT = (0.0, 5.0, 4.0, 16.0, -14.0, -2.0, -10.0, 2.0, 14.0, -16.0, 18.0, -12.0)
_DDC_DDR = ([30.0, 200.0, 400.0, 600.0], [0.0, -16.5, -10.5, 0.0])
_DDC_DL = ([30.0, 200.0, 400.0, 600.0], [0.0, -0.825, -0.525, 0.0])


def sync_asynchrony(ioi, rng, sigma: float, alpha: float = 0.6, beta: float = 0.0) -> np.ndarray:
    """Asynchronies a(n) of a player keeping time with the score's inter-onset intervals `ioi`: timekeeper
    T(n) = T(n-1) - beta (T(n-1) - IOI(n-1)) plus gaussian noise of sd `sigma`, phase correction
    a(n+1) = a(n) - alpha a(n) + T(n) - IOI(n)."""
    ioi = np.asarray(ioi, float)
    a = np.zeros(ioi.size)
    T = float(ioi[0]) if ioi.size else 0.0
    for n in range(ioi.size - 1):
        if n:
            T -= beta * (T - ioi[n - 1])
        a[n + 1] = a[n] - alpha * a[n] + (T + sigma * rng.standard_normal()) - ioi[n]
    return a


def _perform(ev, c, L, tag):
    """KTH rules, swing, humanise (timing / velocity), tempo drift, clamp to the piece length, sort."""
    total = c.bars * c.meter
    for e in ev:
        e["_t0"], e["_t1"] = e["t"], e["t"] + e["dur"]
    emo = L.get("emotion")
    if emo:
        if emo not in EMOTIONS:
            raise SpecError(f"emotion must be one of {sorted(EMOTIONS)}")
        L = {**EMOTIONS[emo]["rules"], **L}
    _kth(ev, c, L)
    if emo:
        g, art = 10 ** (EMOTIONS[emo]["level_db"] / 20.0), EMOTIONS[emo]["articulation"]
        for e in ev:
            e["vel"] *= g
            e["dur"] *= art
    sw = L.get("swing", c.style.swing if c.style else 0.0)
    sg = float(L.get("swing_grid", c.grid))
    if sw == "auto":
        sw = swing_auto(2 * sg * c.q)
    sw = float(sw)
    if sw:                                              # same meaning as `pattern`: every other step is late
        for e in ev:
            k = e["t"] / sg
            if abs(k - round(k)) < 1e-6 and int(round(k)) % 2:
                e["t"] += sw * sg
    hum = L.get("humanize", 0)
    # humanize 1: onset sigma 0.98 % of a beat (about 6 ms of onset jitter on a 16th hi-hat), 6 ms at 96 bpm,
    # 10 ms at 60. A dict gives "time" in ms instead. Velocity spread 13 %.
    hum = hum if isinstance(hum, dict) else {"time": 1000.0 * 0.0098 * c.q * float(hum), "vel": 0.13 * float(hum)}
    sig = float(hum.get("time", HENNIG_MS if hum.get("model") == "hennig" else 0.0)) / 1000.0 / c.q
    vs = float(hum.get("vel", 0.0))
    if sig > 0 or vs > 0:
        rng = _rng(c.seed, tag, "perform")
        order = sorted(ev, key=lambda e: e["t"])
        onsets = sorted({e["t"] for e in order})        # notes of one chord move together
        # Deviations are 1/f, which listeners prefer to white jitter at the same sigma, plus an equal white part:
        # real drumming is long-range correlated over phrases and anticorrelated hit to hit. model "hennig":
        # pure 1/f^beta, one series shared by every layer of the seed (the whole band drifts together).
        sync = L.get("sync")
        if sync and sig > 0:
            sync = sync if isinstance(sync, dict) else {}
            tdev = dict(zip(onsets, sync_asynchrony(np.diff(onsets, append=onsets[-1] + (onsets[-1] - onsets[-2] if len(onsets) > 1 else 1.0)),
                                                    rng, sig, float(sync.get("alpha", 0.6)), float(sync.get("beta", 0.0)))))
        elif hum.get("model") == "hennig" and sig > 0:
            g = gamma_noise(len(onsets), float(hum.get("beta", HENNIG_BETA)), _rng(c.seed, "hennig"))
            tdev = dict(zip(onsets, sig * (g - g.mean()) / (g.std() + 1e-12)))
        elif hum.get("model") == "arma" and sig > 0:
            tdev = dict(zip(onsets, sig * arma_hihat(len(onsets), rng)))
        else:
            tdev = dict(zip(onsets, sig * _pink_white(len(onsets), rng))) if sig > 0 else {}
        # velocity: 1/f too, mapped to 0..1 like the old Beta draw
        vdev = _pink_white(len(order), rng, white=0.3) if vs > 0 else None
        for i, e in enumerate(order):
            if vdev is not None:
                e["vel"] *= 1.0 - vs * float(np.clip(0.5 + 0.25 * vdev[i], 0.0, 1.0))
        for e in ev:
            e["t"] = max(0.0, e["t"] + tdev.get(e["t"], 0.0))
    if L.get("ensemble"):
        rng_e = _rng(c.seed, tag, "ensemble")
        on_ = sorted({e["t"] for e in ev})
        nxt = dict(zip(on_, np.diff(on_, append=on_[-1] + 1.0)))
        for e in ev:
            sd = ensemble_sigma_ms(1000.0 * c.q * nxt[e["t"]])
            e["t"] = max(0.0, e["t"] + rng_e.normal(0.0, sd) / 1000.0 / c.q)
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
    p = float(L.get("drift_trend", 0.0))
    if p:
        # monotone tempo change, (1 + p) x faster at the end, renormalised so the piece keeps its length
        if p <= -1:
            raise SpecError("drift_trend > -1")
        f = lambda b: total * math.log1p(p * min(b, total) / total) / math.log1p(p) + max(0.0, b - total) / (1 + p)   # noqa: E731
        for e in ev:
            t1 = f(e["t"] + e["dur"])
            e["t"] = f(e["t"])
            e["dur"] = t1 - e["t"]
    _rmd(ev, c, L)
    s = L.get("expression")
    if s is not None:
        # every timing deviation from the score scaled by s (0 = deadpan, 1 = as performed)
        s = float(s)
        for e in ev:
            t1 = e["_t1"] + s * (e["t"] + e["dur"] - e["_t1"])
            e["t"] = e["_t0"] + s * (e["t"] - e["_t0"])
            e["dur"] = max(0.0, t1 - e["t"])
    ph = L.get("phase")
    if ph:
        # Reich phasing: every repetition of the `cycle` beats lands `shift` beats later than the last
        ph = ph if isinstance(ph, dict) else {"cycle": float(ph)}
        cyc, sh = float(ph.get("cycle", c.meter)), float(ph.get("shift", 1.0 / 16.0))
        for e in ev:
            e["t"] += sh * int(e["_t0"] // cyc + 1e-9)
    feel = L.get("feel")
    if feel not in (None, False, "soloist", "bass", "ride"):
        raise SpecError(f"feel must be soloist, bass or ride, not {feel!r}")
    bpm = 60.0 / c.q
    if feel == "ride":
        r = ride_ratio(bpm)
        for e in ev:
            if abs(e["_t0"] - math.floor(e["_t0"]) - 0.5) < 1e-6:
                e["t"] += r / (1.0 + r) - 0.5
    for e in ev:
        e["_on"] = abs(e["_t0"] - round(e["_t0"])) < 1e-6
        del e["_t0"], e["_t1"]
    ev = _off_echo(ev, L, total, c.q)
    out = []
    last = max((round(max(0.0, e["t"]), 6) for e in ev if e["t"] < total - 1e-9), default=total)
    rit = ritard_map(L.get("ritard"), total, c.meter, last)
    shift = float(L.get("offset_ms", 0.0)) / 1000.0 / c.q
    late = float(L.get("downbeat_delay_ms", ellis_delay(bpm) if feel == "soloist" else 0.0)) / 1000.0 / c.q
    if feel == "bass":
        shift += FEEL_BASS_MS / 1000.0 / c.q
    nma = L.get("nma")
    if nma:
        shift -= (NMA_MS.get(nma, 0.0) if isinstance(nma, str) else (NMA_MS["musician"] if nma is True else float(nma))) / 1000.0 / c.q
    for e in ev:
        on_beat = e.pop("_on", False)
        if e["t"] >= total - 1e-9:
            continue
        if on_beat and late:
            e["t"] += late
        t0 = round(max(0.0, e["t"]), 6)
        d = max(0.0, min(e["dur"], total - t0))
        if rit:
            t0, d = rit(t0), rit(t0 + d) - rit(t0)
        e["t"] = round(t0 + shift, 6)
        e["dur"] = round(d, 6)
        e["vel"] = round(float(np.clip(e["vel"], 0.05, 1.0)), 4)
        if e["pitch"] is not None:
            e["pitch"] = round(float(e["pitch"]), 6)
        out.append(e)
    return sorted(out, key=lambda e: (e["t"], e.get("drum", ""), e["pitch"] or 0.0))


def _off_echo(ev, L, total, q=1.0):
    """`off` {"time" beats, "steps" semitones, "vel"}: adds a shifted, transposed copy of every event.
    `echo` {"n", "time" beats, "decay"}: adds n delayed repeats, each `decay` times softer.
    `min_ioi` (ms): per drum (or for the pitched part), drops onsets closer than that to the last kept one."""
    off, echo = L.get("off"), L.get("echo")
    src = list(ev)
    if off:
        off = off if isinstance(off, dict) else {"time": float(off)}
        dt, st, gv = float(off.get("time", 0.25)), float(off.get("steps", 0.0)), float(off.get("vel", 0.8))
        for e in src:
            d = dict(e, t=e["t"] + dt, vel=e["vel"] * gv)
            if d["pitch"] is not None:
                d["pitch"] = d["pitch"] + st
            ev.append(d)
    if echo:
        echo = echo if isinstance(echo, dict) else {"n": int(echo)}
        n, dt, g = int(echo.get("n", 2)), float(echo.get("time", 0.5)), float(echo.get("decay", 0.5))
        for e in src:
            for k in range(1, n + 1):
                ev.append(dict(e, t=e["t"] + k * dt, vel=e["vel"] * g ** k))
    m = float(L.get("min_ioi", 0.0)) / 1000.0 / q
    if m > 0:
        last, keep = {}, []
        for e in sorted(ev, key=lambda e: e["t"]):
            k = e.get("drum")
            t0 = last.get(k)
            if t0 is None or abs(e["t"] - t0) < 1e-6 or e["t"] - t0 >= m:
                if t0 is None or e["t"] - t0 >= 1e-6:
                    last[k] = e["t"]
                keep.append(e)
        ev = keep
    return [e for e in ev if e["t"] < total - 1e-9]


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
    owner, gi = None, None
    if layer.get("group") is not None:
        # interchange: the parts of one group never sound together; bar by bar the group's urn picks one
        gn = int(layer.get("group_size", 2))
        gi = int(layer.get("group_index", 0))
        if not 0 <= gi < gn:
            raise SpecError("group_index must be in 0 .. group_size - 1")
        owner = urn(range(gn), c.bars, _rng(c.seed, "group", str(layer["group"])))
    kept = []
    for e in ev:                                        # vertical layering: mute the bars under the threshold
        lvl = c.intensity[min(c.bars - 1, int(e["t"] // c.meter + 1e-9))]
        if lvl > 0 and lvl + 1e-9 >= thr and (owner is None or owner[min(c.bars - 1, int(e["t"] // c.meter + 1e-9))] == gi):
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
        mp = stinger_mood(layer) if kind == "stinger" else None
        if mp:
            layer = {**layer, "fx": list(layer.get("fx") or []) + [{"type": "lowpass", "cutoff": round(mp["brightness"], 3)}]}
        mkey = _MAP_FIGURE[mp["scale"]] if mp else layer.get("mood", "victory")
        dflt = MOODS.get(mkey, MOODS["victory"])[1] if kind == "stinger" else (
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
    if layer.get("sieve"):
        on = set(sieve(layer["sieve"], n)[0])
        pat = [i in on for i in range(n)]
    else:
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
         "pickup": ("pentatonic", "marimba", 3, 72, 0.8), "unlock": ("lydian", "bell", 4, 72, 0.7),
         "alarm": ("chromatic", "brass", 5, 60, 0.9), "settle": ("chromatic", "strings", 5, 60, 0.7)}
_MAP_FIGURE = {"major": "victory", "minor": "defeat", "phrygian": "danger", "chromatic": "alarm"}


def stinger_mood(layer) -> dict | None:
    """The (valence, arousal, tension) map of a numeric or preset mood, None for a named shape."""
    m = layer.get("mood", "victory")
    if isinstance(m, (list, tuple)):
        return mood_params(*m)
    return mood_params(*MOOD_POINTS[m]) if m in MOOD_POINTS else None


def _stinger_events(layer, bpm=None):
    mood = layer.get("mood", "victory")
    mp = stinger_mood(layer)
    if mp:
        mood = _MAP_FIGURE[mp["scale"]]
    elif mood not in MOODS:
        raise SpecError(f"stinger mood {mood!r}: one of {sorted(MOODS)}, {sorted(MOOD_POINTS)} or [valence, arousal, tension]")
    sc, _, n0, base, vel = MOODS[mood]
    offs = scale_offsets(layer.get("scale", mp["scale"] if mp else sc))
    L, n = len(offs), max(2, int(layer.get("length", n0)))
    q = _beat_q(layer, bpm)
    step = float(layer.get("step", 0.25 if q != 1.0 else (15.0 / mp["bpm"] if mp else 0.22 if mood == "defeat" else 0.11)))
    rng = _rng(layer.get("seed", 0), "stinger", mood)
    tonic = (mp["root"] if mp and "key" not in layer else _key_pc(layer.get("key", "C")) + base) + 12 * int(layer.get("octave", 0))
    P = lambda i: tonic + 12 * (i // L) + offs[i % L]                             # noqa: E731
    v = int(rng.integers(0, 3))
    if mood in ("alarm", "settle"):                     # a chromatic cluster, attacks off any grid, held;
        hold = float(layer.get("dur", 2.0)) / q         # settle enters from the top down (falling action)
        on = np.sort(rng.uniform(0.0, (n - 1) * step, n))
        ev = [_ev(float(t), hold - float(t), P(n - 1 - i if mood == "settle" else i), vel) for i, t in enumerate(on)]
        c = NS(bars=1, meter=hold, grid=step, style=None, seed=int(layer.get("seed", 0)), q=q)
        return _perform(ev, c, layer, "stinger")
    if mood == "victory":                               # up the tonic triad, landing on the octave chord
        seq = [[0, 2, 4][k % 3] + L * (k // 3 + 1) for k in range(1 - n, 0)]      # ...sol do mi sol | DO
        final = [[L, L + 2, L + 4], [4, L, L + 2], [L, L + 4, 2 * L]][v]
    elif mood == "defeat":                              # down by step, then a falling minor third to the tonic
        seq = [n - i + (1 if (i == 0 and v == 1 and n > 2) else 0) for i in range(n - 1)]
        final = [0]
    elif mood == "mystery":                             # wandering whole tones ending on an unresolved augmented chord
        seq = [int(x) for x in np.cumsum(np.concatenate([[v], rng.choice([-1, 1, 2], n - 2)]))]
        final = [seq[-1] + 1, seq[-1] + 3, seq[-1] + 5]
    elif mood == "danger":                              # root against the flat second, then a tritone cluster
        seq = ([0, 1] + [(i + v) % 2 for i in range(n - 3)])[:n - 1]
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
    if mp:                                              # a second voice detuned against the final chord
        ev += [_ev((n - 1) * step, step * float(layer.get("hold", 4)), P(s) + mp["detune"], vel) for s in final]
    c = NS(bars=1, meter=(n - 1 + float(layer.get("hold", 4))) * step, grid=step, style=None, seed=int(layer.get("seed", 0)), q=q)
    return _perform(ev, c, layer, "stinger")


# ============================================================================ scatter layer
def _scatter_layers(layer: dict) -> list[dict]:
    """Random placement of child layers (FMOD Scatterer, 12-music s6.4: interval window 500-1000 ms,
    random volume / pitch windows; Xenakis density fields). Placement modes: `interval` [min, max]
    between onsets, `rate` (events per time unit, Poisson: exponential waits, as Cook's PhISM
    collisions), or `count` uniform in the window. `silence_p` leaves slots empty (Farnell's
    "silence as an option", g15 s7). Child choice: `pick` "random" (default) or "shuffle" (each child
    once before any repeats), `weights` one number per child, random picks only."""
    kids = layer.get("layers") or []
    if not kids:
        raise SpecError("scatter needs child `layers`")
    pick, weights = layer.get("pick", "random"), layer.get("weights")
    if pick not in ("random", "shuffle"):
        raise SpecError("scatter pick: random or shuffle")
    if weights is not None:
        if pick == "shuffle":
            raise SpecError("scatter weights only apply to pick=random")
        weights = np.asarray(weights, float)
        if weights.shape != (len(kids),) or not np.all(weights >= 0) or weights.sum() <= 0:
            raise SpecError("scatter weights: one number >= 0 per child, not all zero")
        weights = weights / weights.sum()
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
    out, bag = [], []
    for t in times:
        if pick == "shuffle":
            bag = bag or [int(i) for i in rng.permutation(len(kids))]
            k = bag.pop()
        else:
            k = int(rng.choice(len(kids), p=weights)) if weights is not None else int(rng.integers(len(kids)))
        kid = dict(kids[k])
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
_PARTS = [("drums", "drums", 1.0, 0.4), ("bass", "bass", 0.85, 0.2), ("chords", "chords", 0.45, 0.0),
          ("pad", "pad", 0.3, 0.0), ("lead", "melody", 0.7, 0.5), ("arp", "arp", 0.35, 0.7), ("counter", "counter", 0.45, 0.85)]
_SHARED = ("key", "scale", "tuning", "bars", "sections", "progression", "chords", "harmonic_rhythm", "seed", "meter",
           "grid", "swing", "humanize", "drift", "intensity", "density", "chord_size", "ritard", "high_loud",
           "high_sharp", "double_duration", "duration_contrast", "faster_uphill", "repetition", "rmd", "rmd_unit",
           "rmd_ref_bpm", "inegales", "leap_duration", "melodic_charge", "kth_phrase", "chromatic_charge", "drift_trend",
           "expression", "harmonic_charge", "leap_articulation", "phrase_arch", "phrase_turn", "melodic_intonation",
           "harmonic_intonation")


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
    lead = layer.get("melody_lead", False)
    if lead == "jazz" or (lead is True and name.startswith("jazz")):
        for p in parts:
            if p["name"] == "lead":
                p["downbeat_delay_ms"] = soloist_delay(bpm)
    elif lead and any(p["name"] == "lead" for p in parts):
        ms = 20.0 if lead is True else float(lead)
        for p in parts:
            if p["name"] != "lead":
                p["offset_ms"] = p.get("offset_ms", 0.0) + ms
    c = _context(parts[0] if parts else dict(shared, type="compose"), bpm)
    rit = None
    if shared.get("ritard"):
        # one map for every part: the span ends on the last onset of the whole arrangement
        last = max((e["t"] for p in parts for e in generate({**p, "ritard": None, "offset_ms": 0.0}, bpm)), default=None)
        r = shared["ritard"]
        r = {**(r if isinstance(r, dict) else {"beats": r}), "end": last if last is not None else c.bars * c.meter}
        for p in parts:
            p["ritard"] = r
        rit = ritard_map(r, c.bars * c.meter, c.meter)
    master = [] if layer.get("master", True) is False else [{"type": "reverb", "mix": 0.15, "size": 0.5}, {"type": "compressor", "threshold": -16, "ratio": 2.5}]
    return {"bpm": bpm, "beats": rit(c.bars * c.meter) if rit else c.bars * c.meter, "layers": parts, "fx": master,
            "meter": c.meter, "bars": c.bars}


def _stems(layer, sr, q):
    """Render an arrangement part by part: ([(name, stereo signal placed and panned)], n_exact, plan)."""
    bpm = _bpm(q) if (q != 1.0 or layer.get("bpm")) else None
    a = arrange(layer, bpm)
    bpm = a["bpm"]
    if layer.get("automix", True):
        parts, sigs = _automix(a["layers"], sr, bpm, float(layer.get("width", 0.6)))
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


def xfade_times(layer, states, a, b):
    """(fade in of b, fade out of a) in seconds when b takes over from a: the fade in is b's own `xfade_in`,
    else a's, else the piece's; the fade out b's `xfade_out`, else a's, else the piece's (0: a stops at once)."""
    sa, sb = states.get(a) or {}, states.get(b) or {}
    fin = next((float(v) for v in (sb.get("xfade_in"), sa.get("xfade_in"), layer.get("xfade_in")) if v is not None), 0.0)
    fout = next((float(v) for v in (sb.get("xfade_out"), sa.get("xfade_out"), layer.get("xfade_out")) if v is not None), 0.0)
    return fin, fout


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
    own = ("type", "states", "transitions", "order", "out_stems", "at", "gain", "fx", "repeat", "every", "xfade_in", "xfade_out", "tails", "first")
    base = {k: v for k, v in layer.items() if k not in own}
    rendered, graph, tails, firsts = {}, {"states": {}, "transitions": []}, {}, {}
    for name, over in states.items():
        over = over or {}
        once = bool(over.get("oneshot"))
        cfg = {k: v for k, v in {**base, **over, "type": "arrangement", "fold": not once}.items()
               if k not in ("condition", "xfade_in", "xfade_out")}
        if not once:
            cfg.pop("tail", None)
        stems, n, a = _stems(cfg, sr, q)
        y = FX.apply_chain(mix([(s, 0.0) for _, s in stems], sr), a["fx"], sr)
        full = _fit(y, n, sr, cfg)
        rendered[name] = (to_stereo(full), [(pn, _fit(s, n, sr, cfg)) for pn, s in stems], a, n)
        bars = int(a["bars"])
        graph["states"][name] = {"file": f"{_safe(name)}.wav", "beats": a["beats"], "bpm": a["bpm"], "loop": not once,
                                 "meter": a["meter"], "bars": bars, "samples": n,
                                 "markers": [int(round(b * n / bars)) for b in range(bars)],
                                 "stems": [f"{_safe(name)}__{_safe(pn)}.wav" for pn, _ in stems]}
        for k in ("xfade_in", "xfade_out"):
            if over.get(k) is not None:
                graph["states"][name][k] = float(over[k])
        if over.get("condition") is not None:
            cond = dict(over["condition"])
            condition_met(cond, cond["value"][0] if isinstance(cond.get("value"), (list, tuple)) else cond.get("value"))
            graph["states"][name]["condition"] = cond
        if not once and layer.get("tails") and y.shape[0] > n:
            tails[name] = to_stereo(y[n:])                  # the ring-out past the loop, played when the state is left
            graph["states"][name]["tail"] = f"{_safe(name)}__tail.wav"
        if not once and layer.get("first"):              # the first pass, without the ring-out folded onto its start
            firsts[name] = to_stereo(pad_to(y[:n].copy(), n))
            graph["states"][name]["first"] = f"{_safe(name)}__first.wav"
        if once and over.get("next"):
            if over["next"] not in states:
                raise SpecError(f"state {name!r}: next {over['next']!r} is not a state")
            graph["states"][name]["next"] = over["next"]
    key = base.get("key", "C")
    stingers = {}
    for tr in layer.get("transitions") or []:
        a, b = tr["from"], tr["to"]
        if a not in states or b not in states:
            raise SpecError(f"transition {a!r} -> {b!r}: unknown state")
        g = {"from": a, "to": b, "quantize": tr.get("quantize", "bar")}
        if tr.get("layer"):                             # any layer in place of a mood stinger (e.g. two kicks)
            stingers[(a, b)] = to_stereo(SP.render_layer(tr["layer"], sr, rendered[b][2]["bpm"])[0]) * float(tr.get("gain", 0.7))
            g["stinger"] = f"stinger_{_safe(a)}_to_{_safe(b)}.wav"
        elif tr.get("stinger"):
            st = {"type": "stinger", "mood": tr["stinger"], "key": (states[b] or {}).get("key", key), "seed": int(base.get("seed", 0)),
                  **{k: tr[k] for k in ("inst", "length", "octave", "scale") if k in tr}}
            stingers[(a, b)] = to_stereo(SP.render_layer(st, sr, rendered[b][2]["bpm"])[0]) * float(tr.get("gain", 0.7))
            g["stinger"] = f"stinger_{_safe(a)}_to_{_safe(b)}.wav"
        if tr.get("level") is not None:
            lvl = tr["level"]
            a_db, b_db = (float(lvl[0]), float(lvl[1])) if isinstance(lvl, (list, tuple)) else (0.0, float(lvl))
            g["level_db"] = {"from": a_db, "to": b_db, "curve": "linear_db"}
        if tr.get("pause"):
            g["pause_s"] = (1.0 if tr["pause"] is True else float(tr["pause"])) * 60.0 / float(rendered[b][2]["bpm"] or 60.0)
        graph["transitions"].append(g)
    order = layer.get("order") or list(states)
    parts, t, bar_clock, prev = [], 0.0, 0, None
    for i, item in enumerate(order):
        name, nb = (item.get("state"), item.get("bars")) if isinstance(item, dict) else (item, None)
        if name not in rendered:
            raise SpecError(f"order: unknown state {name!r}")
        full, _, a, n = rendered[name]
        bars = int(a["bars"])
        spb = n / bars
        if nb is None:
            seg, nb = full, bars
        else:                                           # a few bars, entered in phase with the running bar count
            nb = int(nb)
            idx = (np.arange(int(round(nb * spb))) + int(round((bar_clock % bars) * spb))) % n
            seg = full[idx]
        if prev is not None:
            if (prev, name) in stingers:
                parts.append((stingers[(prev, name)], t))
            fin, fout = xfade_times(layer, states, prev, name)
            if fin:
                seg = seg.copy()
                k = min(seg.shape[0], int(round(fin * sr)))
                seg[:k] *= np.linspace(0.0, 1.0, k, endpoint=False)[:, None]
            if fout:                                    # the state left keeps playing its loop while it fades
                k = int(round(fout * sr))
                pf, pn = rendered[prev][0], rendered[prev][3]
                ramp = np.linspace(1.0, 0.0, k, endpoint=False)[:, None]
                parts.append((pf[np.arange(k) % pn] * ramp, t))
            elif prev in tails:
                parts.append((tails[prev], t))
        parts.append((seg, t))
        t += (n if isinstance(item, str) else seg.shape[0]) / sr
        bar_clock += nb
        prev = name
    demo = mix(parts, sr)
    peak = max([float(np.max(np.abs(f))) for f, _, _, _ in rendered.values()] + [float(np.max(np.abs(demo))), 1e-9])
    g = min(1.0, 0.89 / peak)                           # ONE gain for the whole set (Phillips / 12-music W3)
    if layer.get("out_stems"):
        out = Path(str(layer["out_stems"]))
        for name, (full, stems, _, _) in rendered.items():
            write_wav(out / f"{_safe(name)}.wav", full * g, sr)
            for pn, s in stems:
                write_wav(out / f"{_safe(name)}__{_safe(pn)}.wav", s * g, sr)
        for (a, b), s in stingers.items():
            write_wav(out / f"stinger_{_safe(a)}_to_{_safe(b)}.wav", s * g, sr)
        for name, s in tails.items():
            write_wav(out / f"{_safe(name)}__tail.wav", s * g, sr)
        for name, s in firsts.items():
            write_wav(out / f"{_safe(name)}__first.wav", s * g, sr)
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
    level = max(80.0, float(dyn) * (0.25 + 0.75 * float(np.clip(vel, 0.0, 1.0)) ** 2))
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


def ks_drum_raw(n: int, p: int, blend: float = 0.5, stretch: float = 1.0, seed: int = 0, loop: str = "avg",
                copies: int = 1) -> np.ndarray:
    """Karplus-Strong drum (CMJ 7(2):43-55; g15 s4): Y[t] = +(Y[t-p] + Y[t-p-1])/2 with probability
    `blend`, minus that otherwise. blend 1 = the plucked string, 1/2 = a drum (rms halves every 2p+1
    samples), 0 = an octave lower with odd harmonics only. Decay stretching: with probability
    1 - 1/stretch the sample is copied instead of averaged. The table starts as two-level noise.
    Vectorised a period at a time (a sample only depends on samples at least p back).
    `loop`: "avg" the two-point average, "121" (Y[t-p-1] + 2Y[t-p] + Y[t-p+1])/4 (pitch fs/p), "siegel"
    (Y[t-p] + Y[t-1])/2 (softer, nylon-like). `copies` k fills the table with k copies of one noise
    burst, so the note sounds k times higher with the same decay."""
    p = max(2, int(p))
    rng = np.random.default_rng(int(seed))
    y = np.zeros(max(n, p + 2))
    k = max(1, int(copies))
    y[:p + 1] = np.tile(rng.integers(0, 2, -(-(p + 1) // k)) * 2.0 - 1.0, k)[:p + 1]
    if loop == "siegel":
        for t in range(p + 1, y.shape[0]):
            v = 0.5 * (y[t - p] + y[t - 1])
            if stretch > 1.0 and rng.random() < 1.0 - 1.0 / stretch:
                v = y[t - p]
            y[t] = v if rng.random() < blend else -v
        return y[:n]
    step = p - 1 if loop == "121" else p
    for s in range(p + 1, y.shape[0], step):
        j = np.arange(s, min(s + step, y.shape[0]))
        if loop == "121":
            v = 0.25 * (y[j - p - 1] + 2.0 * y[j - p] + y[j - p + 1])
        else:
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
