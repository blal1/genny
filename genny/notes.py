"""Note names, frequencies, scales, chords and the step-sequence mini language."""
from __future__ import annotations

import re

NOTE_INDEX = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
    "whole_tone": [0, 2, 4, 6, 8, 10],
}

CHORDS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "add9": [0, 4, 7, 14],
    "power": [0, 7],
    "maj9": [0, 4, 7, 11, 14],
    "min9": [0, 3, 7, 10, 14],
    "oct": [0, 12],
}

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_freq(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12.0)


def midi_to_name(m: int) -> str:
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


def note_to_midi(name: str) -> int:
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad note name: {name!r} (expected like C4, F#3, Bb2)")
    letter, acc, octave = m.groups()
    semi = NOTE_INDEX[letter.upper()] + (1 if acc == "#" else -1 if acc == "b" else 0)
    return (int(octave) + 1) * 12 + semi


def to_hz(value) -> float:
    """A frequency parameter (`freq`, `freq2`, speech `pitch`): a number is always Hz.

    Accepts 110, 110.0, "110", "110hz" (all 110 Hz) or a note name such as "A2".
    Unlike note lists (`to_freq`), a bare number is never read as a MIDI note, so
    `{"freq": 55}` is a 55 Hz hum, not MIDI 55 (G3, 196 Hz).
    """
    if isinstance(value, bool):
        raise ValueError("frequency cannot be a bool")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.lower().endswith("hz"):
        return float(s[:-2])
    if _NOTE_RE.match(s):
        return midi_to_freq(note_to_midi(s))
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"cannot interpret frequency {value!r} (use Hz like 110 or '110hz', or a note like 'A2')") from e


def to_freq(value) -> float:
    """A note in a note list (`notes`, `steps`): 'C4', 'F#3', a MIDI int 0..127, '440hz', or a float in Hz.

    Frequency *parameters* use `to_hz`, where a bare number is always Hz.
    """
    if isinstance(value, bool):
        raise ValueError("pitch cannot be a bool")
    if isinstance(value, int):
        return midi_to_freq(value) if 0 <= value <= 127 else float(value)
    if isinstance(value, float):
        return value
    s = str(value).strip()
    if s.lower().endswith("hz"):
        return float(s[:-2])
    if _NOTE_RE.match(s):
        return midi_to_freq(note_to_midi(s))
    try:
        f = float(s)
    except ValueError as e:
        raise ValueError(f"cannot interpret pitch {value!r}") from e
    return midi_to_freq(f) if f.is_integer() and 0 <= f <= 127 else f


def chord(root: str, kind: str = "maj") -> list[str]:
    base = note_to_midi(root)
    return [midi_to_name(base + i) for i in CHORDS[kind]]


def scale(root: str, kind: str = "major", octaves: int = 1) -> list[str]:
    base = note_to_midi(root)
    out = []
    for o in range(octaves):
        out += [midi_to_name(base + 12 * o + i) for i in SCALES[kind]]
    out.append(midi_to_name(base + 12 * octaves))
    return out


def parse_pitch_list(value) -> list[float]:
    """'C4 E4 G4' | 'C4,E4' | ['C4', 440.0] | 'C4:maj' (named chord) -> list of Hz."""
    if isinstance(value, (list, tuple)):
        return [to_freq(i) for i in value]
    s = str(value).strip()
    if ":" in s:
        root, kind = s.split(":", 1)
        if kind in CHORDS:
            return [to_freq(n) for n in chord(root, kind)]
    items = [p for p in re.split(r"[\s,]+", s) if p]
    return [to_freq(i) for i in items]


def parse_sequence(text: str, default_dur: float = 0.25) -> list[dict]:
    """Step notation for melodies. Tokens separated by whitespace:

      C4            note, default duration
      C4:0.5        note, duration in seconds
      [C4,E4,G4]:1  chord
      C4:maj:0.5    named chord (see CHORDS)
      -:0.25        rest
      C4:0.5@0.6    velocity 0..1 after '@'
      |             bar line, ignored (for readability)

    Returns list of {"pitches": [hz...], "dur": float, "vel": float}; pitches empty for a rest.
    """
    steps = []
    for tok in text.split():
        if tok == "|":   # cosmetic bar line
            continue
        vel = 1.0
        if "@" in tok:
            tok, v = tok.rsplit("@", 1)
            vel = float(v)
        dur = default_dur
        if tok.startswith("["):
            close = tok.index("]")
            pitches = parse_pitch_list(tok[1:close])
            rest = tok[close + 1:]
            if rest.startswith(":"):
                dur = float(rest[1:])
        else:
            parts = tok.split(":")
            if parts[0] in ("-", "_", "r", "R"):
                pitches = []
                if len(parts) > 1:
                    dur = float(parts[1])
            elif len(parts) >= 2 and parts[1] in CHORDS:
                pitches = parse_pitch_list(parts[0] + ":" + parts[1])
                if len(parts) > 2:
                    dur = float(parts[2])
            else:
                pitches = parse_pitch_list(parts[0])
                if len(parts) > 1:
                    dur = float(parts[1])
        steps.append({"pitches": pitches, "dur": dur, "vel": vel})
    return steps
