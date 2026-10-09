"""Builds examples/library/styles.json: one short loop (4-8 bars) per music style, and renders them
to sounds/styles/<style>.wav. Each entry is a plain genny spec, so the JSON doubles as a cookbook:
which instruments, registers, rhythms and mix levels make a style read as that style.

Run:  uv run python examples/build_styles.py [--no-render] [--only lofi jazz ...]

Each spec sets "bpm", so every time in it is in beats (see AGENTS.md "Music"):
  Song.line(inst, "C4:1 E4:.5 [C4,E4,G4]:2 -:1")   durations in beats, "|" is a cosmetic bar line
  Song.drum(kind, "x---x---x---x---")              one bar of 16ths, repeated; X accent, x normal, o ghost
Mix levels follow AGENTS.md "Music": kick 1, snare 0.8, hats 0.2-0.35, bass 0.8, chords 0.35-0.5, lead 0.7-0.8.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "examples" / "library"

ROOM = {"type": "reverb", "mix": 0.14, "size": 0.45, "damp": 0.5}
HALL = {"type": "reverb", "mix": 0.25, "size": 0.8, "damp": 0.5}
GLUE = {"type": "compressor", "threshold": -16, "ratio": 2.5}


class Song:
    """Collects layers for one loop. Everything is in beats: the spec carries `"bpm"`, so genny reads
    step durations, `at` and pattern steps as beats (see AGENTS.md "Music")."""

    def __init__(self, name, bpm, bars=4, beats=4, swing=0.0, fx=None, tail=1.5, about=""):
        self.name, self.bpm, self.bars, self.beats, self.swing = name, bpm, bars, beats, swing
        self.layers: list[dict] = []
        self.fx = fx if fx is not None else [ROOM, GLUE]
        self.tail = tail
        self.about = about

    @staticmethod
    def _tidy(text: str) -> str:
        text = re.sub(r"(\d*\.\d{4})\d+", r"\1", text)          # 0.6666666666 -> 0.6666
        return " ".join(text.split())

    def line(self, inst, text, gain=1.0, vel=1.0, at=0.0, legato=1.0, params=None, fx=None, strum=0.0):
        """A melodic / chord part. Tokens are genny step notation with durations in beats; a token
        without a duration lasts one beat."""
        layer = {"type": "seq", "inst": inst, "steps": self._tidy(text), "step": 1, "gain": gain}
        if vel != 1.0:
            layer["vel"] = vel
        if at:
            layer["at"] = at
        if legato != 1.0:
            layer["legato"] = legato
        if strum:
            layer["strum"] = strum
        if params:
            layer["params"] = params
        if fx:
            layer["fx"] = fx
        self.layers.append(layer)
        return self

    def drum(self, kind, pattern, gain=1.0, vel=1.0, div=4, params=None, fx=None, swing=None):
        """`pattern` is one bar (or several) at `div` steps per beat; it repeats to fill the song.
        X = accent, x = normal, o = ghost note."""
        swing = self.swing if swing is None else swing
        steps = pattern.replace(" ", "")
        count = len(steps.replace("|", ""))
        total = self.bars * self.beats * div
        if set(steps) & set("xo"):
            gain = round(gain * 0.7, 3)        # leave headroom for the accents (X plays 1.4x)
        else:
            steps = steps.replace("X", "x")
        layer = {"type": "pattern", "kind": kind, "steps": steps, "gain": gain}
        if total % count:
            raise ValueError(f"{self.name}/{kind}: pattern of {count} steps does not divide {total}")
        if total // count > 1:
            layer["bars"] = total // count
        if div != 4:
            layer["step"] = round(1 / div, 4)
        if swing:
            layer["swing"] = swing
        if vel != 1.0:
            layer["vel"] = vel
        if params:
            layer["params"] = params
        if fx:
            layer["fx"] = fx
        self.layers.append(layer)
        return self

    def add(self, layer: dict, at=0.0):
        layer = dict(layer)
        if at:
            layer["at"] = at
        self.layers.append(layer)
        return self

    def spec(self) -> dict:
        return {"out": f"{self.name}.wav", "about": self.about, "bpm": self.bpm, "beats": self.bars * self.beats,
                "tail": self.tail, "layers": self.layers, "fx": self.fx, "normalize": -2.0}


SONGS: list[Song] = []


def song(*a, **k) -> Song:
    s = Song(*a, **k)
    SONGS.append(s)
    return s


# ============================================================ chill / jazz
s = song("lofi", 76, swing=0.18, about="Lo-fi hip-hop: felted keys, lazy swung drums, upright bass, vinyl crackle, everything lowpassed.",
         fx=[{"type": "lowpass", "cutoff": 5200}, {"type": "vibrato", "rate": 0.6, "depth": 0.12}, ROOM, GLUE])
s.line("epiano", "[D3,A3,C4,E4,F4]:4 | [G2,F3,A3,B3,E4]:4 | [C3,G3,B3,D4,E4]:4 | [A2,G3,B3,C#4,E4]:4", gain=0.5, vel=0.6, strum=0.03)
s.line("upright_bass", "D2:1.5 D2:.5 -:1 A2:1 | G2:1.5 G2:.5 -:1 D2:1 | C2:1.5 C2:.5 -:1 G2:1 | A2:1.5 A2:.5 -:.5 E2:.5 A2:1", gain=0.85)
s.line("felt_piano", "-:2 A4:.5 C5:.5 E5:1 | D5:1.5 B4:.5 -:2 | -:1 G4:.5 B4:.5 E5:2 | C#5:1 E5:.5 A4:1.5 -:1", gain=0.7, vel=0.7)
s.drum("kick", "X-----x---X-----", gain=0.9, params={"punch": 0.4, "decay": 0.3})
s.drum("snap", "----X-------X---", gain=0.6)
s.drum("brush", "----x-------x---", gain=0.5)
s.drum("hihat", "x-x-x-x-x-x-x-xo", gain=0.31, params={"tone": 6500})
s.add({"type": "sfx", "kind": "vinyl", "params": {"dur": 13.5}, "gain": 0.18})

s = song("jazz_swing", 138, swing=0.0, about="Swing trio + horn: ride in triplets, walking upright bass, sparse piano comping, muted trumpet.")
T = 2 / 3   # swung eighth: long-short
s.line("upright_bass", "D2 F2 A2 C3 | B2 A2 G2 F2 | E2 G2 C3 B2 | A2 C#3 E3 G2", gain=0.9)
s.line("piano", f"-:{1 + T} [F3,A3,C4,E4]:{1 / 3 + 1} -:1 | -:{T} [F3,A3,B3,E4]:{1 / 3 + 2} -:1 | -:{1 + T} [E3,G3,B3,D4]:{1 / 3 + 1} -:1 | -:{T} [G3,B3,C#4,E4]:{1 / 3 + 1} -:2",
       gain=0.6, vel=0.6, legato=0.6)
s.line("trumpet", f"-:1 A4:{T} C5:{1 / 3} E5:{T} D5:{1 / 3} C5:1 | B4:{T} A4:{1 / 3} G4:1 -:2 | -:1 G4:{T} B4:{1 / 3} D5:{T} E5:{1 / 3} C5:1 | C#5:{T} E5:{1 / 3} A4:2 -:1",
       gain=0.7, vel=0.75, params={"mute": 0.8})
s.drum("ride", "X--x-xX--x-x", div=3, gain=0.45)            # ding, ding-a-ding (triplet grid)
s.drum("hihat", "---x-----x--", div=3, gain=0.35, params={"decay": 0.04})
s.drum("brush", "x--x--x--x--", div=3, gain=0.3, params={"swish": 0.6})
s.drum("kick", "o-----o-----", div=3, gain=0.5, params={"punch": 0.2, "decay": 0.25})

s = song("bossa_nova", 132, about="Bossa nova: nylon guitar comping, bossa clave on the rim, soft kick, shaker, flute melody.")
s.line("guitar", "[D3,A3,C4,F4]:1.5 [D3,A3,C4,F4]:1 [D3,A3,C4,F4]:1.5 | [G3,B3,D4,F4]:1.5 [G3,B3,D4,F4]:1 [G3,B3,D4,F4]:1.5 | [C3,G3,B3,E4]:1.5 [C3,G3,B3,E4]:1 [C3,G3,B3,E4]:1.5 | [A2,G3,C#4,E4]:1.5 [A2,G3,C#4,E4]:1 [A2,G3,C#4,E4]:1.5",
       gain=0.6, vel=0.7, legato=0.8, strum=0.012)
s.line("upright_bass", "D2:1.5 A2:.5 D2:1.5 A2:.5 | G2:1.5 D3:.5 G2:1.5 D3:.5 | C2:1.5 G2:.5 C2:1.5 G2:.5 | A2:1.5 E2:.5 A2:1.5 E2:.5", gain=0.8)
s.line("flute", "-:1 A5:1 G5:.5 F5:.5 E5:1 | D5:2 -:1 F5:1 | E5:1.5 D5:.5 C5:1 B4:1 | C#5:3 -:1", gain=0.65, vel=0.7)
s.drum("rim", "X--X--X---X--X--", gain=0.35)
s.drum("shaker", "xxXxxxXxxxXxxxXx", gain=0.28)
s.drum("kick", "x--x----x--x----", gain=0.55, params={"punch": 0.2, "decay": 0.25})

s = song("blues_shuffle", 104, about="12-bar style shuffle: boogie guitar figure, walking bass, harmonica lead, shuffled drums.")
B = lambda r, f, s6: f"[{r},{f}]:{T} [{r},{f}]:{1 / 3} [{r},{s6}]:{T} [{r},{s6}]:{1 / 3} "   # the boogie 5th-6th rock  # noqa: E731
s.line("electric_guitar", B("A2", "E3", "F#3") * 4 + B("D3", "A3", "B3") * 2 + B("A2", "E3", "F#3") * 2, gain=0.6, legato=0.8)
s.line("finger_bass", "A1 C#2 E2 F#2 | G2 F#2 E2 C#2 | D2 F#2 A2 B2 | A1 C#2 E2 E2", gain=0.85)
s.line("harmonica", f"-:2 E5:{T} G5:{1 / 3} A5:1 | G5:{T} E5:{1 / 3} D5:1 C5:{T} A4:{1 / 3} -:1 | -:1 F#5:{T} A5:{1 / 3} C6:1 A5:1 | G5:{T} E5:{1 / 3} C5:{T} A4:{1 / 3} A4:2", gain=0.7)
s.drum("kick", "X-----X-----", div=3, gain=0.9)
s.drum("snare", "---X-----X--", div=3, gain=0.7)
s.drum("hihat", "x-xx-xx-xx-x", div=3, gain=0.35)

s = song("soul_ballad", 72, about="Slow soul: Wurlitzer chords, round electric bass, sax melody, finger snaps, a pad of strings.")
s.line("wurli", "[F3,A3,C4,E4]:4 | [D3,F3,A3,C4]:4 | [Bb2,D3,F3,A3]:4 | [C3,E3,G3,Bb3]:4", gain=0.5, vel=0.65, strum=0.02)
s.line("strings", "[A4,C5]:4 | [A4,C5]:4 | [F4,A4]:4 | [G4,Bb4]:4", gain=0.28, vel=0.6)
s.line("finger_bass", "F2:1.5 F2:.5 -:1 C2:.5 E2:.5 | D2:1.5 D2:.5 -:1 A1:1 | Bb1:1.5 Bb1:.5 -:1 F2:1 | C2:1.5 C2:.5 E2:1 G2:1", gain=0.85)
s.line("sax", "-:1 C5:.5 A4:.5 C5:1.5 D5:.5 | F5:2 E5:.5 D5:.5 C5:1 | D5:1.5 C5:.5 A4:2 | G4:1 Bb4:1 C5:2", gain=0.75, vel=0.8)
s.drum("kick", "X---------x-----", gain=0.8, params={"punch": 0.4})
s.drum("snap", "----X-------X---", gain=0.6)
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.25, params={"tone": 6500})

# ============================================================ groove
s = song("funk", 104, about="Funk: slap bass on the one, clavinet 16ths, muted guitar scratch, brass stabs, tight drums with ghost notes.")
s.line("slap_bass", "E2:.5 -:.25 E2:.25 -:.25 G2:.25 A2:.5 -:.5 E3:.25 -:.25 D3:.5 E3:.5 | E2:.5 -:.25 E2:.25 -:.25 G2:.25 A2:.5 -:.5 B2:.25 -:.25 D3:.25 B2:.25 A2:.5", gain=0.9)
s.line("slap_bass", "E2:.5 -:.25 E2:.25 -:.25 G2:.25 A2:.5 -:.5 E3:.25 -:.25 D3:.5 E3:.5 | A2:.5 -:.25 A2:.25 -:.25 C3:.25 D3:.5 B2:.5 -:.5 B2:.25 D3:.25 E3:.5", gain=0.9, at=8)
s.line("clav", ("-:.5 [G3,B3,D4]:.25 -:.25 [G3,B3,D4]:.25 -:.5 [G3,B3,D4]:.25 -:.5 [A3,C#4,E4]:.25 -:.25 [G3,B3,D4]:.5 -:.5 | " * 4), gain=0.45, legato=0.7)
s.line("muted_guitar", ("E3:.25 E3:.25 -:.25 E3:.25 " * 16), gain=0.35, vel=0.7)
s.line("brass", "-:3 [E4,G#4,B4,D5]:.25 -:.75 | -:4 | -:3 [E4,G#4,B4,D5]:.25 -:.75 | -:2.5 [D4,F#4,A4]:.5 [E4,G#4,B4]:1", gain=0.5, legato=0.8)
s.drum("kick", "X--x------X--x--", gain=1.0)
s.drum("snare", "----X--o-o--X--o", gain=0.8, params={"decay": 0.16})
s.drum("hihat", "x-x-x-x-x-x-x-xx", gain=0.35, params={"decay": 0.045})
s.drum("openhat", "--------------x-", gain=0.28, params={"decay": 0.15})

s = song("disco", 120, about="Disco: four-on-the-floor, open hat on every off-beat, octave bass, string machine, clav chick.")
s.line("finger_bass", ("C2:.5 C3:.5 " * 4 + "A1:.5 A2:.5 " * 4 + "F1:.5 F2:.5 " * 4 + "G1:.5 G2:.5 " * 4), gain=0.85, legato=0.8)
s.line("string_machine", "[G4,C5,E5]:4 | [A4,C5,E5]:4 | [A4,C5,F5]:4 | [G4,B4,D5]:4", gain=0.42)
s.line("clav", ("-:.5 [E4,G4,C5]:.25 -:.25 " * 4 + "-:.5 [E4,A4,C5]:.25 -:.25 " * 4 + "-:.5 [F4,A4,C5]:.25 -:.25 " * 4 + "-:.5 [D4,G4,B4]:.25 -:.25 " * 4), gain=0.4)
s.line("violin", "G5:1.5 E5:.5 C5:2 | A5:1.5 E5:.5 C5:2 | A5:1 G5:1 F5:1 C5:1 | D5:3 -:1", gain=0.5, vel=0.8)
s.drum("kick", "X---X---X---X---", gain=1.0)
s.drum("clap", "----X-------X---", gain=0.6)
s.drum("openhat", "--x---x---x---x-", gain=0.35, params={"decay": 0.18})
s.drum("hihat", "x---x---x---x---", gain=0.28)

s = song("reggae", 76, about="Roots reggae: one-drop drums (kick and rim on beat 3), organ and guitar skank on the off-beats, melodic bass with space.")
s.line("finger_bass", "A1:.75 -:.25 A1:.5 C2:.5 E2:1 -:1 | D2:.75 -:.25 D2:.5 F2:.5 A2:1 -:1 | A1:.75 -:.25 A1:.5 C2:.5 E2:1 -:1 | E2:.75 -:.25 E2:.5 G2:.5 B1:1 E2:1", gain=0.95, params={"tone": 1100})
s.line("organ", ("-:.5 [A3,C4,E4]:.25 -:.25 " * 4 + "-:.5 [A3,D4,F4]:.25 -:.25 " * 4 + "-:.5 [A3,C4,E4]:.25 -:.25 " * 4 + "-:.5 [G#3,B3,E4]:.25 -:.25 " * 4), gain=0.4, params={"upper": 0.5})
s.line("muted_guitar", ("-:.5 [A3,C4,E4]:.5 " * 4 + "-:.5 [A3,D4,F4]:.5 " * 4 + "-:.5 [A3,C4,E4]:.5 " * 4 + "-:.5 [G#3,B3,E4]:.5 " * 4), gain=0.4)
s.line("trombone", "-:4 | -:2 A3:.5 C4:.5 D4:1 | E4:3 -:1 | D4:.5 C4:.5 B3:1 E3:2", gain=0.55)
s.drum("kick", "--------X-------", gain=1.0)
s.drum("rim", "--------X-------", gain=0.6)
s.drum("hihat", "x-xxx-xxx-xxx-xx", gain=0.31, swing=0.3)
s.drum("shaker", "--x---x---x---x-", gain=0.25)

s = song("latin_salsa", 96, about="Salsa: 2-3 son clave, conga tumbao, piano montuno, anticipated bass, cowbell, trumpet line.")
s.line("piano", ("[C4,C5]:.5 E4:.25 [G4,C5]:.5 E4:.25 [G4,C5]:.5 E4:.5 [G4,C5]:.5 E4:.25 [A4,C5]:.75 | [B3,B4]:.5 D4:.25 [G4,B4]:.5 D4:.25 [G4,B4]:.5 D4:.5 [F4,B4]:.5 D4:.25 [G4,B4]:.75 | " * 2), gain=0.5, vel=0.7, legato=0.8)
s.line("upright_bass", ("-:1.5 G2:1 C3:1.5 | -:1.5 D3:1 G2:1.5 | " * 2), gain=0.9)
s.line("trumpet", "-:8 | E5:.5 G5:.5 -:.5 E5:.5 D5:1 C5:1 | D5:.5 F5:.5 -:.5 D5:.5 B4:2", gain=0.65)
s.drum("clave", "--X-X---X--X--X-", gain=0.4)
s.drum("conga", "--------------xx", gain=0.6)                               # open tones lead into the bar
s.drum("conga", "x--x--x---x-----", gain=0.45, params={"slap": 1.0})
s.drum("cowbell", "X---X---X---X---", gain=0.3)
s.drum("timbale", "------------x-xx", gain=0.4)
s.drum("guiro", "x---x-x-x---x-x-", gain=0.2, params={"dur": 0.12, "ridges": 14})

s = song("reggaeton", 94, about="Reggaeton: the dembow (kick on every beat, snare on the 'a' of 1 and the 'and' of 2), sub bass, plucked hook.")
s.line("bass808", "A1:1.5 A1:.5 -:1 E2:1 | F1:1.5 F1:.5 -:1 C2:1 | G1:1.5 G1:.5 -:1 D2:1 | E1:1.5 E1:.5 E2:1 G#1:1", gain=0.85, params={"decay": 0.8})
s.line("marimba", "A4:.5 C5:.5 E5:.25 C5:.75 A4:.5 E5:.5 -:1 | A4:.5 C5:.5 F5:.25 C5:.75 A4:.5 F5:.5 -:1 | B4:.5 D5:.5 G5:.25 D5:.75 B4:.5 G5:.5 -:1 | B4:.5 E5:.5 G#5:.25 E5:.75 D5:.5 B4:.5 -:1", gain=0.6)
s.line("warm_pad", "[A3,C4,E4]:4 | [A3,C4,F4]:4 | [B3,D4,G4]:4 | [B3,E4,G#4]:4", gain=0.3, params={"attack": 0.2})
s.drum("kick", "X---X---X---X---", gain=1.0)
s.drum("snare808", "---X--X----X--X-", gain=0.7)
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.28)

# ============================================================ rock / pop
s = song("rock", 132, about="Rock: overdriven power chords (clean guitar + distortion on the layer), eighth-note bass, big backbeat, lead guitar.")
s.line("electric_guitar", "[E2,B2,E3]:1.5 [E2,B2,E3]:.5 -:.5 [E2,B2,E3]:.5 [G2,D3,G3]:1 | [A2,E3,A3]:1.5 [A2,E3,A3]:.5 -:.5 [A2,E3,A3]:.5 [G2,D3,G3]:1 | [E2,B2,E3]:1.5 [E2,B2,E3]:.5 -:.5 [E2,B2,E3]:.5 [G2,D3,G3]:1 | [C3,G3,C4]:2 [D3,A3,D4]:2",
       gain=0.3, fx=[{"type": "distortion", "drive": 9, "tone": 3200}])
s.line("finger_bass", ("E2:.5 " * 6 + "G2:.5 G2:.5 " + "A2:.5 " * 6 + "G2:.5 G2:.5 " + "E2:.5 " * 6 + "G2:.5 G2:.5 " + "C2:.5 " * 4 + "D2:.5 " * 4), gain=0.8, legato=0.9)
s.line("dist_guitar", "-:8 | B4:1 D5:.5 E5:1.5 G5:1 | E5:.5 D5:.5 B4:1 A4:.5 B4:1.5", gain=0.5, params={"drive": 5})
s.drum("kick", "X-----x-X-x-----", gain=1.0)
s.drum("snare", "----X-------X---", gain=0.85)
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.35)
s.drum("crash", "X---------------" + "-" * 48, gain=0.49)

s = song("metal", 170, about="Metal: palm-muted gallops with heavy distortion, double kick, half-time snare, low power chords.")
s.line("electric_guitar", ("E2:.5 E2:.25 E2:.25 " * 3 + "[G2,D3]:.5 [F#2,C#3]:.5 " + "E2:.5 E2:.25 E2:.25 " * 3 + "[A2,E3]:.5 [Bb2,F3]:.5 ") * 2,
       gain=0.3, legato=0.85, params={"tone": 2400}, fx=[{"type": "distortion", "drive": 16, "tone": 3000}])
s.line("finger_bass", ("E1:.5 E1:.25 E1:.25 " * 3 + "G1:.5 F#1:.5 " + "E1:.5 E1:.25 E1:.25 " * 3 + "A1:.5 Bb1:.5 ") * 2, gain=0.75,
       fx=[{"type": "distortion", "drive": 3, "tone": 1500}, {"type": "gain", "db": -6}])
s.drum("kick", "XxxxXxxxXxxxXxxx", gain=0.9, params={"decay": 0.2, "punch": 1.0})
s.drum("snare", "--------X-------", gain=0.9)
s.drum("hihat", "x---x---x---x---", gain=0.31)
s.drum("crash", "X---------------" + "-" * 16, gain=0.42)

s = song("pop", 112, about="Modern pop: piano chords, round bass, simple beat with claps, a soft lead hook and a plucked counter-line.")
s.line("piano", "[C4,E4,G4]:1.5 [C4,E4,G4]:2.5 | [G3,B3,D4]:1.5 [G3,B3,D4]:2.5 | [A3,C4,E4]:1.5 [A3,C4,E4]:2.5 | [F3,A3,C4]:1.5 [F3,A3,C4]:2.5", gain=0.5, vel=0.7)
s.line("finger_bass", "C2:1.5 C2:1.5 C2:1 | G1:1.5 G1:1.5 G1:1 | A1:1.5 A1:1.5 A1:1 | F1:1.5 F1:1.5 G1:1", gain=0.85)
s.line("soft_lead", "E5:1 G5:.5 E5:.5 D5:1 C5:1 | D5:1 G5:.5 D5:.5 B4:2 | C5:1 E5:.5 C5:.5 A4:1 C5:1 | A4:.5 C5:.5 F5:1 E5:1 D5:1", gain=0.7)
s.line("synth_pluck", ("C5:.25 G4:.25 E5:.25 G4:.25 " * 4 + "B4:.25 G4:.25 D5:.25 G4:.25 " * 4 + "C5:.25 A4:.25 E5:.25 A4:.25 " * 4 + "C5:.25 A4:.25 F5:.25 A4:.25 " * 4), gain=0.3, vel=0.7)
s.drum("kick", "X-----x---X-----", gain=1.0)
s.drum("clap", "----X-------X---", gain=0.6)
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.31)

# ============================================================ electronic
s = song("synthwave", 100, about="Synthwave: pulsing eighth-note bass, string machine chords, plucked arpeggio, gated-reverb snare, soaring lead.",
         fx=[{"type": "reverb", "mix": 0.2, "size": 0.7, "damp": 0.5}, GLUE])
s.line("bass", ("A1:.5 " * 8 + "F1:.5 " * 8 + "C2:.5 " * 8 + "G1:.5 " * 8), gain=0.8, legato=0.75, params={"cutoff": 900})
s.line("string_machine", "[A3,C4,E4]:4 | [A3,C4,F4]:4 | [G3,C4,E4]:4 | [G3,B3,D4]:4", gain=0.45)
s.line("synth_pluck", ("A4:.25 C5:.25 E5:.25 A5:.25 " * 4 + "A4:.25 C5:.25 F5:.25 A5:.25 " * 4 + "G4:.25 C5:.25 E5:.25 G5:.25 " * 4 + "G4:.25 B4:.25 D5:.25 G5:.25 " * 4), gain=0.35,
       fx=[{"type": "delay", "time": 0.225, "feedback": 0.3, "mix": 0.25}])
s.line("lead", "-:2 E5:1 A5:1 | G5:1.5 F5:.5 C5:2 | E5:1 G5:1 C6:1.5 B5:.5 | G5:3 -:1", gain=0.6, fx=[{"type": "chorus", "mix": 0.3}])
s.drum("kick", "X---X---X---X---", gain=1.0)
s.drum("snare808", "----X-------X---", gain=0.8, fx=[{"type": "reverb", "mix": 0.4, "size": 0.6}])
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.28)
s.drum("tom", "-" * 56 + "x-x-x-xx", gain=0.5, params={"tune": 140})

s = song("house", 124, about="House: four-on-the-floor, off-beat open hat, claps, rubbery FM bass on the off-beats, piano chord stabs.")
s.line("fm_bass", ("-:.5 C2:.5 " * 3 + "-:.25 C2:.25 Eb2:.5 " + "-:.5 Ab1:.5 " * 3 + "-:.25 Ab1:.25 Bb1:.5 ") * 2, gain=0.85, legato=0.8)
s.line("piano", ("[C4,Eb4,G4,Bb4]:.75 [C4,Eb4,G4,Bb4]:.75 -:.5 [C4,Eb4,G4,Bb4]:.5 -:.5 [C4,Eb4,G4,Bb4]:.5 -:.5 | [C4,Eb4,Ab4]:.75 [C4,Eb4,Ab4]:.75 -:.5 [D4,F4,Bb4]:.5 -:.5 [D4,F4,Bb4]:.5 -:.5 | " * 2), gain=0.5, vel=0.75, legato=0.7)
s.line("warm_pad", "[G4,Bb4,Eb5]:8 | [Ab4,C5,Eb5]:8", gain=0.25)
s.drum("kick", "X---X---X---X---", gain=1.0)
s.drum("clap", "----X-------X---", gain=0.6)
s.drum("openhat", "--x---x---x---x-", gain=0.35, params={"decay": 0.14})
s.drum("shaker", "xxxxxxxxxxxxxxxx", gain=0.17)

s = song("acid_techno", 132, about="Acid techno: a 16th-note 303 line whose accents open the filter, relentless kick, off-beat hats, rim.")
s.line("acid", ("A1:.25@1 A1:.25@.5 A2:.25@.6 A1:.25@.5 C2:.25@1 A1:.25@.5 A1:.25@.6 G2:.25@1 A1:.25@.5 A1:.25@.5 E2:.25@1 A1:.25@.5 D2:.25@.7 A1:.25@.5 C2:.25@1 B1:.25@.6 | " * 4),
       gain=0.8, legato=0.8, params={"res": 0.8, "env": 2200, "cutoff": 300, "decay": 0.16})
s.drum("kick", "X---X---X---X---", gain=1.0, params={"decay": 0.3})
s.drum("hihat", "--x---x---x---x-", gain=0.35)
s.drum("clap", "----X-------X---", gain=0.5)
s.drum("rim", "-------x--x-----", gain=0.3)

s = song("drum_and_bass", 172, about="Drum and bass: two-step breakbeat with ghost snares, long reese bass notes, a cold pad and bell stabs.")
s.line("reese", "F1:3 -:.5 F1:.5 | Ab1:3 -:1 | F1:3 -:.5 F1:.5 | Eb1:2 C1:2", gain=0.85, params={"cutoff": 700})
s.line("glass_pad", "[C4,F4,Ab4]:8 | [C4,Eb4,G4]:8", gain=0.35)
s.line("bell", "-:2.5 C5:.5 -:1 | -:1.5 Ab4:.5 -:2 | -:2.5 C5:.5 -:1 | -:1.5 G4:.5 Bb4:.5 -:1.5", gain=0.35, vel=0.6,
       fx=[{"type": "delay", "time": 0.26, "feedback": 0.4, "mix": 0.35}])
s.drum("kick", "X---------X-----", gain=1.0, params={"decay": 0.25})
s.drum("snare", "----X--o-o--X---", gain=0.85, params={"decay": 0.18})
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.31)
s.drum("shaker", "-x-x-x-x-x-x-x-x", gain=0.17)

s = song("trap", 140, about="Trap: half-time clap on beat 3, long sliding 808 bass, hi-hat 16ths with fast rolls, a dark sparse bell.")
s.line("bass808", "G1:3 -:1 | G1:1.5 Bb1:2.5 | Eb1:3 -:1 | F1:2 D2:2", gain=0.9, params={"decay": 1.4, "drive": 0.45})
s.line("bell", "G4:1 Bb4:.5 D4:1.5 -:1 | G4:1 Bb4:.5 A4:1.5 -:1 | Eb4:1 G4:.5 Bb3:1.5 -:1 | F4:1 A4:.5 C5:.5 Bb4:2", gain=0.45, vel=0.55, params={"bright": 0.6, "decay": 1.2})
s.line("warm_pad", "[G3,Bb3,D4]:8 | [Eb3,G3,Bb3]:4 [F3,A3,C4]:4", gain=0.25)
s.drum("kick", "X---------X--x--|X-----x---------", gain=1.0)
s.drum("clap", "--------X-------", gain=0.7)
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.31)
s.drum("hihat", "------------------------xxxxxxxx|--------------------------------", div=8, gain=0.22, params={"decay": 0.03})

s = song("chiptune", 150, about="8-bit chiptune: pulse lead, fast arpeggio standing in for chords, triangle bass, noise drums.",
         fx=[GLUE])
s.line("chip", "E5:.5 E5:.5 -:.5 E5:.5 -:.5 C5:.5 E5:1 | G5:1 -:1 G4:1 -:1 | C5:1 -:.5 G4:1 -:.5 E4:1 | -:.5 A4:1 B4:1 Bb4:.5 A4:1", gain=0.6, legato=0.85, params={"width": 0.25})
s.line("chip", ("C4:.125 E4:.125 G4:.125 C5:.125 " * 8 + "G3:.125 B3:.125 D4:.125 G4:.125 " * 8 + "A3:.125 C4:.125 E4:.125 A4:.125 " * 8 + "F3:.125 A3:.125 C4:.125 F4:.125 " * 8),
       gain=0.25, params={"width": 0.125})
s.line("chiptri", ("C3:.5 C3:.5 G2:.5 C3:.5 " * 2 + "G2:.5 G2:.5 D3:.5 G2:.5 " * 2 + "A2:.5 A2:.5 E3:.5 A2:.5 " * 2 + "F2:.5 F2:.5 C3:.5 G2:.5 " * 2), gain=0.7, legato=0.8)
s.drum("kick", "X-------X-x-----", gain=0.8, params={"tune": 70, "decay": 0.12}, fx=[{"type": "bitcrush", "bits": 6, "rate": 11025}])
s.drum("snare", "----X-------X---", gain=0.5, params={"decay": 0.1}, fx=[{"type": "bitcrush", "bits": 5, "rate": 11025}, {"type": "lowpass", "cutoff": 5000}])
s.drum("hihat", "x-x-x-x-x-x-x-x-", gain=0.17, params={"decay": 0.03, "tone": 6500})

# ============================================================ orchestral / classical
s = song("orchestral_heroic", 108, about="Heroic orchestral: horn theme over string chords, trumpet answer, cello and timpani underneath, taiko and cymbal accents.",
         fx=[HALL, GLUE], tail=2.5)
s.line("strings", "[C4,E4,G4]:4 | [A3,C4,F4]:4 | [G3,C4,E4]:4 | [G3,B3,D4]:4", gain=0.5, params={"attack": 0.12})
s.line("cello", "C2:4 | F2:4 | C2:2 E2:2 | G2:4", gain=0.6)
s.line("french_horn", "G3:1.5 C4:.5 E4:2 | F4:1.5 E4:.5 C4:2 | E4:1 G4:1 C5:1.5 B4:.5 | G4:3 -:1", gain=0.8)
s.line("trumpet", "-:8 | -:2 G4:.5 C5:.5 E5:1 | D5:.75 D5:.25 D5:1 G5:2", gain=0.55)
s.line("timpani", "C2:1 -:2 G2:1 | F2:1 -:3 | C2:1 -:2 C2:.5 C2:.5 | G2:.5 G2:.5 G2:.5 G2:.5 G2:2", gain=0.6)
s.drum("taiko", "X---------------|----------------|X---------------|X-------X-------", gain=0.6)
s.drum("crash", "X---------------" + "-" * 32 + "X---------------", gain=0.35)
s.line("harp", "C4:.125 E4:.125 G4:.125 C5:.125 E5:.125 G5:.125 C6:.25", gain=0.4, at=15)

s = song("baroque", 100, about="Baroque: harpsichord broken chords, a walking cello bass line and an oboe melody with ornaments; no drums.",
         fx=[{"type": "reverb", "mix": 0.2, "size": 0.6}])
s.line("harpsichord", ("D4:.25 F4:.25 A4:.25 F4:.25 " * 4 + "C#4:.25 E4:.25 A4:.25 E4:.25 " * 4 + "D4:.25 F4:.25 Bb4:.25 F4:.25 " * 4 + "C#4:.25 E4:.25 A4:.25 G4:.25 " * 2 + "[D4,F4,A4]:2"), gain=0.6)
s.line("cello", "D3:.5 E3:.5 F3:.5 D3:.5 A3:.5 G3:.5 F3:.5 E3:.5 | A2:.5 B2:.5 C#3:.5 A2:.5 E3:.5 D3:.5 C#3:.5 A2:.5 | Bb2:.5 C3:.5 D3:.5 Bb2:.5 G3:.5 F3:.5 E3:.5 D3:.5 | A2:1 A3:1 D3:2", gain=0.6, params={"attack": 0.04}, legato=0.9)
s.line("oboe", "A5:1 F5:.5 D5:.5 A5:.75 Bb5:.125 A5:.125 G5:.5 F5:.5 | E5:1 A5:1 G5:.5 F5:.5 E5:1 | D5:.5 F5:.5 Bb5:1 A5:.5 G5:.5 F5:.5 E5:.5 | E5:.75 D5:.25 C#5:1 D5:2", gain=0.65)

s = song("ragtime", 96, about="Ragtime piano: left hand strides (bass note, chord), right hand plays a syncopated tune. One instrument.",
         fx=[ROOM])
s.line("piano", "C2:.5 [E3,G3,C4]:.5 G2:.5 [E3,G3,C4]:.5 C2:.5 [E3,G3,C4]:.5 G2:.5 [E3,G3,C4]:.5 | F2:.5 [F3,A3,C4]:.5 C2:.5 [F3,A3,C4]:.5 F2:.5 [F3,A3,C4]:.5 C2:.5 [F3,A3,C4]:.5 | G2:.5 [F3,G3,B3]:.5 D2:.5 [F3,G3,B3]:.5 G2:.5 [F3,G3,B3]:.5 D2:.5 [F3,G3,B3]:.5 | C2:.5 [E3,G3,C4]:.5 G2:.5 [E3,G3,C4]:.5 [C2,C3]:1 -:1",
       gain=0.6, vel=0.7, legato=0.7)
s.line("piano", "E5:.25 C5:.25 E5:.5 C5:.25 E5:.25 G5:.5 E5:.25 C5:.25 -:.5 E5:1 | F5:.25 C5:.25 F5:.5 C5:.25 F5:.25 A5:.5 F5:.25 C5:.25 -:.5 A5:1 | G5:.25 D5:.25 G5:.5 D5:.25 F5:.25 B5:.5 G5:.25 F5:.25 D5:.5 B4:1 | C5:.25 E5:.25 G5:.5 C6:.5 G5:.5 [E5,C6]:1 -:1",
       gain=0.8, vel=0.9)

s = song("lullaby", 84, beats=3, bars=8, about="Music-box lullaby in 3/4: music box melody, celesta broken chords, a soft glass pad; no drums.",
         fx=[{"type": "reverb", "mix": 0.25, "size": 0.6}])
s.line("music_box", "E6:1 G6:1 E6:1 | D6:2 C6:1 | D6:1 F6:1 D6:1 | C6:2 -:1 | E6:1 G6:1 C7:1 | B6:2 A6:1 | G6:1 F6:1 D6:1 | C6:3", gain=0.7)
s.line("celesta", ("C5:1 E5:1 G5:1 | B4:1 D5:1 G5:1 | B4:1 D5:1 F5:1 | C5:1 E5:1 G5:1 | " * 2), gain=0.4, vel=0.6)
s.line("glass_pad", "[C4,G4]:6 | [G3,D4]:3 [C4,E4]:3 | [C4,G4]:6 | [G3,F4]:3 [C4,E4]:3", gain=0.25)

s = song("spooky", 92, about="Spooky: theremin tune over diminished pipe-organ chords, a celesta ostinato, a tolling bell and low timpani.",
         fx=[HALL], tail=3)
s.line("church_organ", "[D3,F3,G#3,B3]:4 | [C#3,E3,G3,Bb3]:4 | [D3,F3,A3]:4 | [C#3,E3,A3]:4", gain=0.45)
s.line("celesta", ("D5:.5 F5:.5 G#5:.5 F5:.5 " * 2 + "C#5:.5 E5:.5 G5:.5 E5:.5 " * 2 + "D5:.5 F5:.5 A5:.5 F5:.5 " * 2 + "C#5:.5 E5:.5 A5:.5 E5:.5 " * 2), gain=0.35)
s.line("theremin", "-:2 D5:2 | C#5:1 E5:1 G5:2 | F5:1.5 E5:.5 D5:2 | C#5:3 -:1", gain=0.65, params={"vibrato": 0.35})
s.line("tubular_bell", "D4:4 | -:4 | D4:4 | -:4", gain=0.4)
s.line("timpani", "D2:1 -:3 | -:3 D2:.5 D2:.5 | D2:1 -:3 | A2:1 -:1 A2:.5 A2:.5 A2:1", gain=0.5)

s = song("christmas", 112, about="Christmas: sleigh bells on every eighth, glockenspiel tune, warm strings and horns, chimes on the downbeats.",
         fx=[{"type": "reverb", "mix": 0.2, "size": 0.6}, GLUE])
s.line("glockenspiel", "E6:.5 E6:.5 E6:1 G6:.5 C6:.5 D6:.5 E6:.5 | F6:.5 F6:.5 F6:.75 F6:.25 E6:.5 E6:.5 E6:1 | D6:.5 D6:.5 E6:.5 D6:.5 G6:2 | E6:.5 D6:.5 C6:1 C6:2", gain=0.55)
s.line("strings", "[G4,C5,E5]:4 | [A4,C5,F5]:2 [G4,C5,E5]:2 | [F4,A4,D5]:2 [G4,B4,D5]:2 | [G4,C5,E5]:4", gain=0.4)
s.line("french_horn", "C4:4 | F4:2 E4:2 | D4:2 G3:2 | C4:4", gain=0.4)
s.line("upright_bass", "C2:1 G2:1 C2:1 G2:1 | F2:1 C2:1 C2:1 G2:1 | D2:1 A2:1 G2:1 D2:1 | C2:1 G2:1 C2:2", gain=0.7)
s.line("tubular_bell", "C4:4 | -:4 | -:4 | C4:4", gain=0.35)
s.drum("sleigh", "x-x-x-x-x-x-x-x-", gain=0.3, params={"decay": 0.18})

# ============================================================ folk / world
s = song("celtic_jig", 116, beats=3, bars=8, swing=0.0, about="Celtic jig in 6/8: fiddle tune doubled by a whistle, harp chords, bodhran pulse (each beat here is a dotted quarter).",
         fx=[ROOM, GLUE])
J = 1 / 3    # an eighth note of the jig
s.line("violin", f"D5:{J} E5:{J} F#5:{J} A5:{J} F#5:{J} D5:{J} B4:{J} D5:{J} F#5:{J} | E5:{J} F#5:{J} G5:{J} B5:{J} G5:{J} E5:{J} C#5:{J} E5:{J} A5:{J} | "
                 f"D5:{J} E5:{J} F#5:{J} A5:{J} F#5:{J} D5:{J} B5:{J} A5:{J} F#5:{J} | E5:{J} G5:{J} F#5:{J} E5:{J} D5:{J} C#5:{J} D5:1 | " * 2,
       gain=0.7, params={"attack": 0.025, "vibrato": 0.3}, legato=0.9)
s.line("whistle", f"-:12 | D6:{J} E6:{J} F#6:{J} A6:{J} F#6:{J} D6:{J} B5:{J} D6:{J} F#6:{J} | E6:{J} F#6:{J} G6:{J} B6:{J} G6:{J} E6:{J} C#6:{J} E6:{J} A6:{J} | "
                  f"D6:{J} E6:{J} F#6:{J} A6:{J} F#6:{J} D6:{J} B6:{J} A6:{J} F#6:{J} | E6:{J} G6:{J} F#6:{J} E6:{J} D6:{J} C#6:{J} D6:1", gain=0.35)
s.line("harp", ("[D3,A3,D4,F#4]:3 | [A2,E3,A3,C#4]:3 | [D3,A3,D4,F#4]:2 [B2,F#3,B3]:1 | [A2,E3,A3]:2 [D3,A3,D4]:1 | " * 2), gain=0.5, strum=0.02)
s.drum("djembe", "X-xx-xX-x", div=3, gain=0.5, params={"stroke": "bass", "tune": 420})
s.drum("djembe", "-x--x--x-", div=3, gain=0.3, params={"stroke": "tone", "tune": 260})

s = song("bluegrass", 124, about="Bluegrass: banjo roll, boom-chick guitar and mandolin chop, upright bass on one and three, fiddle on top.")
s.line("banjo", ("G3:.25 B3:.25 D4:.25 G4:.25 B3:.25 D4:.25 G4:.25 D4:.25 " * 4 + "C4:.25 E4:.25 G4:.25 C5:.25 E4:.25 G4:.25 C5:.25 G4:.25 " * 2 +
                 "D4:.25 F#4:.25 A4:.25 D5:.25 F#4:.25 A4:.25 D5:.25 A4:.25 " + "G3:.25 B3:.25 D4:.25 G4:.25 B3:.25 D4:.25 G4:.5"), gain=0.6, vel=0.8)
s.line("upright_bass", "G2:1 -:1 D2:1 -:1 | G2:1 -:1 D2:1 -:1 | C2:1 -:1 G2:1 -:1 | D2:1 -:1 G2:2", gain=0.9)
s.line("steel_guitar", ("-:1 [G3,B3,D4,G4]:.5 -:.5 " * 4 + "-:1 [G3,C4,E4,G4]:.5 -:.5 " * 2 + "-:1 [F#3,A3,D4]:.5 -:.5 -:1 [G3,B3,D4,G4]:1"), gain=0.45, strum=0.012, legato=0.6)
s.line("mandolin", ("-:1 [B4,D5,G5]:.25 -:.75 " * 4 + "-:1 [C5,E5,G5]:.25 -:.75 " * 2 + "-:1 [A4,D5,F#5]:.25 -:.75 -:2"), gain=0.5)
s.line("violin", "-:8 | G5:.5 E5:.5 G5:1 E5:.5 D5:.5 C5:1 | D5:.5 F#5:.5 A5:.5 F#5:.5 G5:2", gain=0.55, params={"attack": 0.03, "vibrato": 0.3})

s = song("polka_oompah", 120, about="Oompah / polka: tuba on the beat, accordion chords on the off-beat, clarinet tune, snare and cymbal.")
s.line("tuba", "F2:1 C2:1 F2:1 C2:1 | F2:1 C2:1 F2:1 A2:1 | Bb2:1 F2:1 C3:1 G2:1 | F2:1 C2:1 F2:2", gain=0.9, legato=0.6)
s.line("accordion", ("-:.5 [A3,C4,F4]:.5 " * 8 + "-:.5 [Bb3,D4,F4]:.5 " * 2 + "-:.5 [Bb3,C4,E4]:.5 " * 2 + "-:.5 [A3,C4,F4]:.5 " * 2 + "[A3,C4,F4]:1 -:1"), gain=0.45, legato=0.6)
s.line("clarinet", "C5:.5 A4:.5 F5:1 E5:.5 D5:.5 C5:1 | A4:.5 C5:.5 F5:.5 A5:.5 G5:1 F5:1 | D5:.5 F5:.5 Bb5:1 G5:.5 E5:.5 C5:1 | F5:.5 E5:.5 G5:.5 E5:.5 F5:2", gain=0.7, legato=0.85)
s.drum("kick", "X-------X-------", gain=0.7, params={"punch": 0.4})
s.drum("snare", "----x-------x---", gain=0.55, params={"decay": 0.12})
s.drum("crash", "-" * 56 + "X-------", gain=0.35, params={"decay": 0.6})

s = song("tango", 116, about="Tango: staccato accordion (bandoneon) and piano marking four, upright bass, a dramatic violin line in a minor key.")
s.line("accordion", "[A3,C4,E4]:.5 -:.5 [A3,C4,E4]:.5 -:.5 [A3,C4,E4]:.5 -:.5 [A3,C4,E4]:.25 [B3,D4,E4]:.25 [C4,E4,A4]:.5 | [G#3,B3,E4]:.5 -:.5 [G#3,B3,E4]:.5 -:.5 [G#3,B3,E4]:.5 -:.5 [G#3,B3,E4]:.5 [E3,G#3,D4]:.5 | "
                    "[A3,D4,F4]:.5 -:.5 [A3,D4,F4]:.5 -:.5 [A3,C4,E4]:.5 -:.5 [A3,C4,E4]:.5 -:.5 | [G#3,B3,E4]:.5 -:.5 [G#3,B3,D4]:.5 -:.5 [A3,C4,E4]:1 -:1", gain=0.55, params={"musette": 4})
s.line("upright_bass", "A2:1 -:1 E2:1 A2:.5 B2:.5 | E2:1 -:1 B2:1 E2:1 | D2:1 -:1 A2:1 -:1 | E2:1 E2:1 A2:2", gain=0.85)
s.line("violin", "E5:1.5 A5:.5 C6:1 B5:.5 A5:.5 | G#5:1.5 B5:.5 E5:2 | F5:1 A5:.5 D6:.5 C6:1 A5:.5 E5:.5 | D5:.5 E5:.5 G#5:1 A5:2", gain=0.7, params={"vibrato": 1.4, "attack": 0.04})
s.line("piano", "[A1,A2]:.5 -:1.5 [E2,E3]:.5 -:1.5 | [E2,E3]:.5 -:1.5 [B1,B2]:.5 -:1.5 | [D2,D3]:.5 -:1.5 [A1,A2]:.5 -:1.5 | [E2,E3]:.5 -:.5 [E2,E3]:.5 -:.5 [A1,A2]:1 -:1", gain=0.4)

s = song("andean", 100, about="Andean folk: pan flute melody, tremolo-picked charango (mandolin), nylon guitar, a deep bombo drum.")
s.line("pan_flute", "E5:1 G5:.5 A5:.5 B5:1 A5:.5 G5:.5 | E5:1.5 D5:.5 E5:2 | G5:1 A5:.5 B5:.5 D6:1 B5:.5 A5:.5 | G5:.5 E5:.5 D5:1 E5:2", gain=0.75)
s.line("mandolin", "[E4,G4,B4]:2 [E4,G4,B4]:2 | [D4,G4,B4]:2 [E4,G4,B4]:2 | [C4,E4,G4]:2 [D4,F#4,A4]:2 | [B3,D4,G4]:2 [E4,G4,B4]:2", gain=0.4, params={"tremolo": 12})
s.line("guitar", ("E2:.5 [E3,G3,B3]:.25 [E3,G3,B3]:.25 " * 4 + "D2:.5 [D3,G3,B3]:.25 [D3,G3,B3]:.25 " * 2 + "E2:.5 [E3,G3,B3]:.25 [E3,G3,B3]:.25 " * 2 +
                  "C2:.5 [E3,G3,C4]:.25 [E3,G3,C4]:.25 " * 2 + "D2:.5 [D3,F#3,A3]:.25 [D3,F#3,A3]:.25 " * 2 + "G2:.5 [D3,G3,B3]:.25 [D3,G3,B3]:.25 " * 2 + "E2:.5 [E3,G3,B3]:.5 E2:1"), gain=0.5)
s.drum("taiko", "X-----x-X-------", gain=0.5, params={"tune": 75, "decay": 0.4})
s.drum("shaker", "x-xxx-xxx-xxx-xx", gain=0.21)

s = song("middle_eastern", 96, about="Middle-Eastern: Hijaz-scale oboe (zurna) line with ornaments, plucked oud-style riff, dum-tek hand drum, string drone.",
         fx=[ROOM, GLUE])
s.line("oboe", "D5:1 Eb5:.25 F#5:.25 G5:.5 A5:1 G5:.25 F#5:.25 Eb5:.5 | D5:1.5 Eb5:.25 D5:.25 C5:.5 D5:1.5 | G5:.5 A5:.5 Bb5:.5 A5:.25 G5:.25 F#5:1 Eb5:.5 D5:.5 | Eb5:.5 F#5:.5 Eb5:.5 D5:.5 D5:2", gain=0.7)
s.line("pluck", ("D3:.5 D3:.25 A3:.25 D4:.5 A3:.5 Eb4:.5 D4:.5 A3:.5 D3:.5 | " * 4), gain=0.5, params={"damping": 0.6, "decay": 0.9})
s.line("strings", "[D3,A3]:16", gain=0.25)
s.drum("djembe", "X--X--X-X--X----", gain=0.8, params={"stroke": "bass", "tune": 360})    # dum
s.drum("djembe", "--x--x-x--x-x-x-", gain=0.5, params={"stroke": "slap", "tune": 420})     # tek
s.drum("tambourine", "----x-------x---", gain=0.28)

s = song("indian_raga", 88, bars=4, about="Indian classical flavour: sitar phrases in raga Yaman over a drone, tabla keeping a 16-beat cycle.",
         fx=[ROOM, GLUE], tail=3)
s.line("sitar", "[C3,G3,C4]:8 [C3,G3,C4]:8", gain=0.45, params={"buzz": 0.6}, strum=0.08)
s.line("sitar", "E4:1 F#4:.5 G4:.5 B4:1 A4:.5 G4:.5 | F#4:.5 E4:.5 D4:1 E4:2 | G4:.5 B4:.5 C5:1 D5:.5 C5:.5 B4:.5 A4:.5 | G4:.5 F#4:.5 E4:.5 D4:.5 C4:2", gain=0.75)
s.drum("tabla", "X---x---X-----x-", gain=0.7, params={"stroke": "ge"})
s.drum("tabla", "--x-x-xx--x-x-xx", gain=0.55, params={"stroke": "na"})
s.drum("tabla", "-x---x---x-x-x--", gain=0.35, params={"stroke": "ke"})

s = song("japanese", 80, about="Japanese: koto arpeggios in the In scale (D Eb G A Bb), a breathy bamboo flute, sparse taiko.",
         fx=[{"type": "reverb", "mix": 0.22, "size": 0.7}], tail=3)
s.line("koto", "D4:.5 Eb4:.5 G4:.5 A4:.5 Bb4:1 A4:1 | G4:.5 Eb4:.5 D4:1 [D3,A3]:2 | D4:.5 G4:.5 A4:.5 Bb4:.5 D5:1 Bb4:1 | A4:.5 G4:.5 Eb4:.5 D4:.5 [D3,A3,D4]:2", gain=0.7)
s.line("pan_flute", "-:4 | -:2 A5:2 | Bb5:1.5 A5:.5 G5:2 | Eb5:1 D5:3", gain=0.55, params={"breath": 0.5})
s.drum("taiko", "X---------------|----------------|X-----------x---|X---------------", gain=0.6)
s.drum("woodblock", "------------x---|------------x-x-", gain=0.25, params={"tune": 900})

s = song("gamelan", 96, about="Gamelan: interlocking metallophone patterns on a five-note scale, a slower core melody, a gong marking the cycle.",
         fx=[{"type": "reverb", "mix": 0.22, "size": 0.7}], tail=4)
s.line("gamelan", ("C5:.25 -:.25 E5:.25 -:.25 F5:.25 -:.25 E5:.25 -:.25 " * 8), gain=0.5)
s.line("gamelan", ("-:.25 G5:.25 -:.25 B5:.25 -:.25 G5:.25 -:.25 C6:.25 " * 8), gain=0.45, vel=0.8)
s.line("gamelan", "C4:2 E4:2 | F4:2 G4:2 | B4:2 G4:2 | F4:2 E4:2", gain=0.6, params={"ombak": 5})
s.line("handpan", "C3:4 | F3:4 | G3:4 | C3:4", gain=0.4)
s.drum("gong", "X" + "-" * 63, gain=0.6, params={"tune": 65, "decay": 5})

s = song("ambient", 60, bars=4, about="Ambient: slow warm pad and glass pad, sparse handpan notes, a singing bowl, long reverb; no pulse.",
         fx=[{"type": "reverb", "mix": 0.4, "size": 0.9, "damp": 0.5}], tail=5)
s.line("warm_pad", "[D3,A3,F4]:8 | [Bb2,F3,D4]:8", gain=0.5, params={"attack": 1.5, "release": 2.5})
s.line("glass_pad", "[A4,D5]:8 | [F4,Bb4,D5]:8", gain=0.3, params={"attack": 2.0, "release": 3.0})
s.line("handpan", "-:1 D4:1.5 A4:1.5 F4:2 -:2 | -:1 Bb3:1.5 F4:1.5 D4:2 A3:2", gain=0.5, vel=0.7)
s.line("singing_bowl", "D3:8 | -:8", gain=0.35)


def dump(batch: dict) -> str:
    """Compact, greppable JSON: one layer per line, so one style is a dozen lines an agent can read."""
    j = lambda v: json.dumps(v, ensure_ascii=False)   # noqa: E731
    out = ["{", f' "out_dir": {j(batch["out_dir"])},', ' "sounds": [']
    for i, sp in enumerate(batch["sounds"]):
        head = ", ".join(f"{j(k)}: {j(sp[k])}" for k in ("out", "about", "bpm", "beats", "tail"))
        out.append("  {" + head + ",")
        out.append('   "layers": [')
        out += [f"    {j(layer)}{',' if n < len(sp['layers']) - 1 else ''}" for n, layer in enumerate(sp["layers"])]
        out.append("   ],")
        out.append(f'   "fx": {j(sp["fx"])}, "normalize": {j(sp["normalize"])}' + "}" + ("," if i < len(batch["sounds"]) - 1 else ""))
    out += [" ]", "}"]
    return "\n".join(out) + "\n"


def main():
    render = "--no-render" not in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    batch = {"out_dir": "sounds/styles", "sounds": [s.spec() for s in SONGS]}
    text = dump(batch)
    assert json.loads(text) == batch
    # the cookbook lives in the examples and next to the skill (the skill reads the style it needs)
    for path in (LIB / "styles.json", ROOT / "skill" / "genny" / "styles.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(SONGS)} styles)")
    if render:
        todo = {"out_dir": "sounds/styles", "sounds": [s.spec() for s in SONGS if not only or s.name in only]}
        subprocess.run([sys.executable, "-m", "genny.cli", "render", "-", "--quiet"], input=json.dumps(todo), text=True, cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
