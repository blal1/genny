"""Builds examples/library/<category>.json batch files (25 sounds each) and renders them to sounds/<category>/.

Run:  uv run python examples/build_library.py [--no-render]
Each entry is a plain genny spec, so the JSON files double as reference examples.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "examples" / "library"


# ---------- tiny helpers so the specs below stay readable ----------
def L(type_, **kw):
    d = {"type": type_}
    d.update(kw)
    return d


def synth(inst, notes, dur=0.4, **kw):
    return L("synth", inst=inst, notes=notes, dur=dur, **kw)


def seq(inst, steps, **kw):
    return L("seq", inst=inst, steps=steps, **kw)


def drum(kind, **kw):
    return L("drum", kind=kind, **kw)


def sfx(kind, params=None, **kw):
    return L("sfx", kind=kind, params=params or {}, **kw)


def fx(type_, **kw):
    d = {"type": type_}
    d.update(kw)
    return d


def sound(name, layers, fx_chain=None, **kw):
    spec = {"out": f"{name}.wav", "layers": layers}
    if fx_chain:
        spec["fx"] = fx_chain
    spec.update(kw)
    return spec


REV_S = fx("reverb", mix=0.18, size=0.35)
REV_M = fx("reverb", mix=0.28, size=0.6)
REV_L = fx("reverb", mix=0.35, size=0.85, damp=0.5)

# ---------- 1. notifications ----------
notifications = [
    sound("ding_glass", [synth("glass", "E6", 0.25)], [REV_S]),
    sound("ding_bell_soft", [synth("bell", "C6", 0.3, vel=0.6, params={"ratio": 2.0, "decay": 1.2})], [REV_S]),
    sound("chime_two_up", [seq("bell", "G5:0.12 C6:0.4", params={"decay": 1.0})], [REV_S]),
    sound("chime_two_down", [seq("bell", "C6:0.12 G5:0.4", params={"decay": 1.0})], [REV_S]),
    sound("tri_tone", [seq("glass", "E6:0.1 G#6:0.1 B6:0.35")], [REV_S]),
    sound("marimba_tap", [synth("marimba", "A5", 0.2)], [REV_S]),
    sound("marimba_double", [seq("marimba", "D5:0.09 A5:0.3")], [REV_S]),
    sound("kalimba_message", [seq("kalimba", "E5:0.1 G5:0.1 B5:0.3", vel=0.8)], [REV_S]),
    sound("music_box_hello", [seq("music_box", "C6:0.12 E6:0.12 G6:0.12 E6:0.4")], [REV_M]),
    sound("epiano_mail", [seq("epiano", "F4:0.15 A4:0.35", vel=0.7)], [REV_S]),
    sound("pop_soft", [sfx("pop", {"freq": 600})], [REV_S]),
    sound("pop_double", [sfx("pop", {"freq": 500}), sfx("pop", {"freq": 750}, at=0.09)], [REV_S]),
    sound("blip_sine", [sfx("beep", {"freq": "A6", "dur": 0.06})], [REV_S]),
    sound("success_rise", [sfx("success", {"freq": "E5", "up": 7})], [REV_S]),
    sound("success_bright", [sfx("success", {"freq": "A5", "up": 5, "wave": "triangle"})], [REV_S]),
    sound("mention_triad", [seq("glass", "C6:0.08 E6:0.08 G6:0.3", vel=0.9)], [REV_S]),
    sound("reminder_three", [seq("bell", "E5:0.18 E5:0.18 A5:0.5", params={"decay": 1.2}, vel=0.7)], [REV_M]),
    sound("gentle_pad_swell", [synth("pad", "C5:maj", 0.5, params={"attack": 0.25, "release": 0.5}, gain=0.8)], [fx("lowpass", cutoff=3000), REV_M]),
    sound("achievement_sparkle", [seq("glass", "C6:0.07 E6:0.07 G6:0.07 C7:0.4"), sfx("magic", {"dur": 0.5, "density": 6}, gain=0.4, at=0.2)], [REV_M]),
    sound("alert_double_beep", [sfx("beep", {"freq": 1046, "dur": 0.08, "wave": "triangle"}, repeat=2, every=0.14)], [REV_S]),
    sound("warning_low", [seq("sine", "A4:0.12 F4:0.3", vel=0.8)], [fx("lowpass", cutoff=2500), REV_S]),
    sound("low_battery", [seq("chip", "E5:0.1 C5:0.1 A4:0.25", params={"width": 0.5})], [fx("lowpass", cutoff=4000), REV_S]),
    sound("unlock_glass_up", [seq("glass", "A5:0.06 C#6:0.06 E6:0.06 A6:0.3")], [REV_S]),
    sound("lock_glass_down", [seq("glass", "A6:0.06 E6:0.06 C#6:0.06 A5:0.3")], [REV_S]),
    sound("bubble_message", [sfx("bubble", {"freq": 450}), sfx("bubble", {"freq": 650}, at=0.1)], [REV_S]),
]

# ---------- 2. game cues ----------
cues = [
    sound("beep_sine", [sfx("beep", {"freq": 880, "dur": 0.1})]),
    sound("beep_square", [sfx("beep", {"freq": 660, "dur": 0.1, "wave": "square"})]),
    sound("blip_cursor", [sfx("blip", {"freq": 1400, "dur": 0.04})]),
    sound("blip_cursor_low", [sfx("blip", {"freq": 900, "dur": 0.05})]),
    sound("select_confirm", [seq("chip", "C6:0.05 G6:0.12", params={"width": 0.25}, legato=0.9)]),
    sound("cancel_back", [seq("chip", "G5:0.05 C5:0.12", params={"width": 0.25}, legato=0.9)]),
    sound("click_ui", [sfx("click", {"tone": 3500})]),
    sound("tick_soft", [sfx("click", {"tone": 2000}, gain=0.6)]),
    sound("wall_proximity_far", [sfx("proximity", {"freq": 900, "rate": 2, "dur": 1.5, "width": 0.25})]),
    sound("wall_proximity_mid", [sfx("proximity", {"freq": 1000, "rate": 5, "dur": 1.2, "width": 0.35})]),
    sound("wall_proximity_close", [sfx("proximity", {"freq": 1150, "rate": 12, "dur": 1.0, "width": 0.5})]),
    sound("radar_ping", [sfx("radar", {"freq": 1300})], [fx("delay", time=0.3, feedback=0.35, mix=0.3)]),
    sound("sonar_deep", [sfx("radar", {"freq": 700})], [fx("underwater", amount=0.5), REV_L]),
    sound("laser_pew", [sfx("laser", {"start": 2200, "end": 300, "dur": 0.22})]),
    sound("laser_short", [sfx("laser", {"start": 3000, "end": 800, "dur": 0.1, "wave": "square"})]),
    sound("jump", [sfx("jump", {"start": 280, "end": 900, "dur": 0.18})]),
    sound("coin_pickup", [sfx("coin", {"freq": "B5", "up": 5})]),
    sound("item_pickup", [seq("chip", "E6:0.05 G6:0.05 B6:0.05 E7:0.2", params={"width": 0.5}, legato=0.9)]),
    sound("hit_hurt", [sfx("hit", {"dur": 0.22, "tone": 220, "crunch": 0.6})]),
    sound("punch_impact", [sfx("punch"), drum("kick", gain=0.5, params={"tune": 50})]),
    sound("explosion_small", [sfx("explosion", {"dur": 0.7, "boom": 0.6})]),
    sound("explosion_big", [sfx("explosion", {"dur": 1.6, "boom": 1.2}), drum("taiko", gain=0.6)], [REV_M]),
    sound("footstep_stone", [sfx("footstep", {"tone": 110})]),
    sound("door_close", [sfx("door")], [REV_S]),
    sound("low_health_heartbeat", [sfx("heartbeat", {"rate": 1.3, "dur": 2.3})], [fx("lowpass", cutoff=400)]),
]

# ---------- 3. melodic stingers ----------
stingers = [
    sound("discovery_harp", [seq("harp", "C5:0.09 E5:0.09 G5:0.09 C6:0.09 E6:0.5")], [REV_M]),
    sound("quest_accepted", [seq("brass", "G4:0.12 C5:0.12 E5:0.45"), synth("strings", "C4:maj", 0.7, gain=0.5)], [REV_M]),
    sound("secret_found", [seq("music_box", "E6:0.1 G6:0.1 B6:0.1 D7:0.1 C7:0.5")], [fx("delay", time=0.2, feedback=0.3, mix=0.25), REV_M]),
    sound("mystery_minor", [seq("epiano", "A4:0.2 C5:0.2 E5:0.2 G#5:0.6", vel=0.7), synth("pad", "A3:min", 1.0, gain=0.4)], [REV_L]),
    sound("heroic_brass", [seq("brass", "C4:0.15 G4:0.15 C5:0.15 E5:0.15 G5:0.6"), drum("taiko", gain=0.6)], [REV_M]),
    sound("playful_marimba", [seq("marimba", "C5:0.1 E5:0.1 G5:0.1 E5:0.1 C6:0.3")], [REV_S]),
    sound("sad_piano", [seq("piano", "E5:0.3 D5:0.3 C5:0.3 B4:0.8", vel=0.6)], [REV_L]),
    sound("chip_level_up", [seq("chip", "C5:0.07 E5:0.07 G5:0.07 C6:0.07 E6:0.07 G6:0.3", params={"width": 0.25}, legato=0.9)], [fx("bitcrush", bits=8, rate=22050)]),
    sound("chip_fanfare", [seq("chip", "G5:0.1 G5:0.1 G5:0.1 C6:0.4", params={"width": 0.5}), seq("chiptri", "G3:0.3 C4:0.4", gain=0.8)]),
    sound("bell_arpeggio", [seq("bell", "D5:0.1 F#5:0.1 A5:0.1 D6:0.1 F#6:0.6", params={"decay": 1.5})], [REV_M]),
    sound("glass_shimmer", [seq("glass", "C6:0.08 G6:0.08 E7:0.08 C7:0.5"), sfx("magic", {"dur": 0.6, "density": 8}, gain=0.3)], [REV_M]),
    sound("kalimba_calm", [seq("kalimba", "G4:0.2 B4:0.2 D5:0.2 G5:0.6", vel=0.8)], [REV_M]),
    sound("strings_swell", [synth("strings", "F4:maj7", 1.2, params={"attack": 0.5})], [REV_L]),
    sound("flute_motif", [seq("flute", "E5:0.2 G5:0.2 A5:0.2 G5:0.6")], [REV_M]),
    sound("organ_church", [seq("organ", "C4:maj:0.4 F4:maj:0.4 C4:maj:0.8")], [REV_L]),
    sound("steel_drum_island", [seq("steel_drum", "C5:0.12 E5:0.12 G5:0.12 A5:0.12 G5:0.4")], [REV_M]),
    sound("guitar_strum", [synth("guitar", "E3 B3 E4 G#4 B4 E5", 1.2, strum=0.03)], [REV_M]),
    sound("pluck_question", [seq("pluck", "C5:0.15 E5:0.15 G5:0.15 B5:0.5")], [fx("delay", time=0.25, feedback=0.3, mix=0.3)]),
    sound("vibraphone_jazz", [seq("vibraphone", "D5:0.15 F5:0.15 A5:0.15 C6:0.15 E6:0.6")], [REV_M]),
    sound("lead_synth_riff", [seq("lead", "A4:0.1 C5:0.1 E5:0.1 A5:0.4", params={"cutoff": 5000})], [fx("delay", time=0.15, feedback=0.35, mix=0.25), REV_S]),
    sound("pwm_retro_lush", [seq("pwm", "F4:0.2 A4:0.2 C5:0.2 F5:0.7")], [fx("chorus", mix=0.5), REV_M]),
    sound("choir_reveal", [synth("choir", "D4:min", 1.4, params={"vowel": "o"}), synth("bell", "D6", 0.3, gain=0.4, at=0.6)], [REV_L]),
    sound("brass_alert_minor", [seq("brass", "E4:0.12 G4:0.12 B4:0.5"), drum("kick", gain=0.5)], [REV_M]),
    sound("harp_gliss_down", [seq("harp", "C7:0.05 A6:0.05 F6:0.05 D6:0.05 B5:0.05 G5:0.05 E5:0.05 C5:0.5")], [REV_M]),
    sound("epiano_soft_intro", [seq("epiano", "G4:0.25 B4:0.25 D5:0.25 F#5:0.8", vel=0.6)], [fx("chorus", mix=0.35), REV_M]),
]

# ---------- 4. game over ----------
gameover = [
    sound("descend_piano", [seq("piano", "G5:0.25 Eb5:0.25 C5:0.25 G4:0.9", vel=0.7)], [REV_L]),
    sound("descend_chip", [seq("chip", "C6:0.12 B5:0.12 Bb5:0.12 A5:0.12 Ab5:0.12 G5:0.5", params={"width": 0.5})], [fx("bitcrush", bits=8, rate=16000)]),
    sound("powerdown_classic", [sfx("powerdown", {"freq": "C6", "steps": 7, "rate": 0.08})]),
    sound("powerdown_slow", [sfx("powerdown", {"freq": "A5", "steps": 5, "rate": 0.14})], [fx("lowpass", cutoff=3000), REV_M]),
    sound("minor_chord_pad", [synth("pad", "C4:min", 1.2, params={"attack": 0.05, "release": 1.0})], [REV_L]),
    sound("dark_organ", [seq("organ", "C4:min:0.5 Ab3:maj:0.9")], [fx("lowpass", cutoff=1800), REV_L]),
    sound("taiko_doom", [drum("taiko", params={"tune": 50, "decay": 1.4}), drum("taiko", at=0.6, params={"tune": 45, "decay": 1.8})], [REV_L]),
    sound("kick808_death", [drum("kick808", params={"tune": 40, "decay": 1.5}), synth("sub", "C2", 1.0, gain=0.6)], [fx("lowpass", cutoff=800)]),
    sound("muffled_end", [seq("piano", "E5:0.3 C5:0.3 A4:1.0", vel=0.6)], [fx("muffle", amount=0.7), REV_L]),
    sound("underwater_drown", [seq("bell", "A5:0.3 F5:0.3 D5:0.8", params={"decay": 1.5})], [fx("underwater", amount=0.8), REV_L]),
    sound("sad_strings", [seq("strings", "E4:0.5 D4:0.5 C4:1.2", params={"attack": 0.15})], [REV_L]),
    sound("sad_flute", [seq("flute", "A4:0.4 G4:0.4 E4:1.0")], [REV_L]),
    sound("chip_death_jingle", [seq("chip", "E5:0.1 E5:0.1 -:0.1 E5:0.1 C5:0.1 E5:0.1 G5:0.3 -:0.2 G4:0.5", params={"width": 0.25}, legato=0.85), seq("chiptri", "C3:0.6 -:0.2 G2:0.5", gain=0.7)]),
    sound("glitch_crash", [sfx("glitch", {"dur": 0.4}), sfx("static", {"dur": 0.5, "color": "pink"}, at=0.3, gain=0.6)], [fx("bitcrush", bits=6), fx("lowpass", cutoff=5000)]),
    sound("heartbeat_stop", [sfx("heartbeat", {"rate": 1.0, "dur": 2.1}), synth("sub", "C2", 0.8, at=2.2, gain=0.5)], [fx("lowpass", cutoff=500), REV_M]),
    sound("error_fail", [sfx("error", {"freq": 160, "dur": 0.5})], [REV_S]),
    sound("slow_bell_toll", [synth("bell", "C4", 1.5, params={"ratio": 2.0, "decay": 3.0}), synth("bell", "C4", 1.5, at=1.2, params={"ratio": 2.0, "decay": 3.0}, vel=0.8)], [REV_L]),
    sound("tape_stop", [seq("lead", "C5:0.2 E5:0.2 G5:0.6")], [fx("speed", factor=0.6), fx("lowpass", cutoff=3000), REV_M]),
    sound("reversed_riser", [sfx("riser", {"dur": 1.5, "start": 1500, "end": 100})], [fx("lowpass", cutoff=2500), REV_M]),
    sound("bass_thud_minor", [seq("bass", "A2:0.3 F2:0.3 D2:0.8", params={"cutoff": 700}), drum("kick", gain=0.6)], [REV_S]),
    sound("choir_lament", [synth("choir", "A3:min", 1.6, params={"vowel": "u"})], [fx("lowpass", cutoff=2000), REV_L]),
    sound("epiano_defeat", [seq("epiano", "F5:0.3 D5:0.3 Bb4:0.3 A4:1.0", vel=0.55)], [fx("tremolo", rate=4, depth=0.4), REV_L]),
    sound("distorted_hit_end", [sfx("hit", {"dur": 0.5, "tone": 120, "crunch": 0.9}), drum("crash", gain=0.5)], [fx("distortion", drive=3, tone=3000), REV_M]),
    sound("marimba_wilt", [seq("marimba", "G5:0.15 F5:0.15 Eb5:0.15 D5:0.15 C5:0.5", vel=0.6)], [fx("lowpass", cutoff=2500), REV_M]),
    sound("thunder_end", [sfx("thunder", {"dur": 2.5}), synth("pad", "D3:min", 1.5, gain=0.5)], [REV_L]),
]

# ---------- 5. win ----------
win = [
    sound("fanfare_brass", [seq("brass", "C4:0.12 E4:0.12 G4:0.12 C5:0.5"), synth("brass", "C5:maj", 0.5, at=0.36, gain=0.6), drum("crash", gain=0.4)], [REV_M]),
    sound("fanfare_brass_long", [seq("brass", "G4:0.15 G4:0.15 G4:0.15 C5:0.3 E5:0.3 G5:0.7"), drum("taiko", gain=0.6), drum("crash", at=0.9, gain=0.4)], [REV_M]),
    sound("chip_victory", [seq("chip", "C5:0.08 E5:0.08 G5:0.08 C6:0.08 E6:0.08 G6:0.08 C7:0.45", params={"width": 0.25}, legato=0.9), seq("chiptri", "C3:0.25 G3:0.25 C4:0.5", gain=0.7)]),
    sound("chip_victory_bitcrushed", [seq("chip", "E5:0.1 G5:0.1 C6:0.1 E6:0.1 G6:0.4", params={"width": 0.5}), drum("kick", gain=0.5)], [fx("bitcrush", bits=7, rate=16000)]),
    sound("bell_triumph", [seq("bell", "C5:0.1 E5:0.1 G5:0.1 C6:0.7", params={"decay": 2.0}), synth("pad", "C4:maj", 1.0, gain=0.4)], [REV_L]),
    sound("glass_win_sparkle", [seq("glass", "G5:0.07 B5:0.07 D6:0.07 G6:0.07 B6:0.07 D7:0.4"), sfx("magic", {"dur": 0.7, "density": 10}, gain=0.4, at=0.2)], [REV_M]),
    sound("harp_win_gliss", [seq("harp", "C4:0.05 E4:0.05 G4:0.05 C5:0.05 E5:0.05 G5:0.05 C6:0.05 E6:0.05 G6:0.6")], [REV_M]),
    sound("piano_win", [seq("piano", "C5:0.15 E5:0.15 G5:0.15 C6:maj:0.8", vel=0.9)], [REV_M]),
    sound("marimba_win", [seq("marimba", "C5:0.08 E5:0.08 G5:0.08 C6:0.08 G5:0.08 C6:0.4"), drum("shaker", gain=0.4, repeat=4, every=0.12)], [REV_S]),
    sound("orchestral_hit", [synth("strings", "C4:maj", 0.6, params={"attack": 0.01}), synth("brass", "C5:maj", 0.6, gain=0.6), drum("taiko", gain=0.8), drum("crash", gain=0.5)], [REV_L]),
    sound("lead_synth_win", [seq("lead", "E5:0.1 G5:0.1 B5:0.1 E6:0.5", params={"cutoff": 7000}), seq("bass", "E2:0.4 E2:0.4", gain=0.7), drum("kick", gain=0.6, repeat=2, every=0.4)], [fx("delay", time=0.2, feedback=0.3, mix=0.2), REV_S]),
    sound("organ_win", [seq("organ", "G4:maj:0.25 C5:maj:0.25 G4:maj:0.25 C5:maj:0.8")], [REV_M]),
    sound("steel_drum_win", [seq("steel_drum", "C5:0.1 E5:0.1 G5:0.1 C6:0.1 E6:0.5"), drum("shaker", gain=0.3, repeat=6, every=0.1)], [REV_M]),
    sound("choir_glory", [synth("choir", "C4:maj", 1.6, params={"vowel": "a"}), seq("bell", "C6:0.2 G6:0.6", gain=0.5, at=0.4)], [REV_L]),
    sound("music_box_win", [seq("music_box", "C6:0.1 E6:0.1 G6:0.1 C7:0.1 E7:0.5")], [REV_M]),
    sound("coin_shower", [sfx("coin", {"freq": "B5", "up": 5}, repeat=5, every=0.09)], [REV_S]),
    sound("powerup_win", [sfx("powerup", {"freq": "C5", "steps": 8, "step": 2, "rate": 0.06})], [REV_S]),
    sound("kalimba_win", [seq("kalimba", "G4:0.1 B4:0.1 D5:0.1 G5:0.1 B5:0.5")], [REV_M]),
    sound("vibraphone_win", [seq("vibraphone", "F5:0.12 A5:0.12 C6:0.12 F6:0.6")], [REV_M]),
    sound("pluck_win_delay", [seq("pluck", "D5:0.1 F#5:0.1 A5:0.1 D6:0.5")], [fx("delay", time=0.18, feedback=0.4, mix=0.35, pingpong=True), REV_S]),
    sound("epiano_win_soft", [seq("epiano", "A4:0.15 C#5:0.15 E5:0.15 A5:0.7", vel=0.7)], [fx("chorus", mix=0.4), REV_M]),
    sound("flute_win", [seq("flute", "G5:0.12 A5:0.12 B5:0.12 D6:0.6"), synth("strings", "G4:maj", 0.9, gain=0.5)], [REV_M]),
    sound("drum_roll_finish", [L("pattern", kind="snare", steps="xxxxxxxxxxxxxxxx", step=0.045, vel=0.7), drum("crash", at=0.75, gain=0.7), synth("brass", "C5:maj", 0.6, at=0.75, gain=0.7)], [REV_M]),
    sound("pwm_win_lush", [seq("pwm", "C5:0.15 E5:0.15 G5:0.15 C6:0.7")], [fx("chorus", mix=0.5), REV_M]),
    sound("guitar_win_strum", [synth("guitar", "D3 A3 D4 F#4 A4 D5", 1.0, strum=0.025), synth("guitar", "D3 A3 D4 F#4 A4 D5", 1.2, at=0.3, strum=0.025, vel=0.8)], [REV_M]),
]

CATEGORIES = {
    "notifications": notifications,
    "game_cues": cues,
    "melodic_stingers": stingers,
    "game_over": gameover,
    "win": win,
}


def main():
    LIB.mkdir(parents=True, exist_ok=True)
    files = []
    for cat, sounds in CATEGORIES.items():
        assert len(sounds) == 25, f"{cat} has {len(sounds)} sounds, expected 25"
        names = [s["out"] for s in sounds]
        assert len(set(names)) == 25, f"duplicate names in {cat}"
        batch = {"out_dir": f"sounds/{cat}", "sounds": sounds}
        path = LIB / f"{cat}.json"
        path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        files.append(path)
        print(f"wrote {path.relative_to(ROOT)} ({len(sounds)} sounds)")
    if "--no-render" in sys.argv:
        return
    cmd = [sys.executable, "-m", "genny.cli", "render", "--quiet", *map(str, files)]
    print("rendering...")
    r = subprocess.run(cmd, cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
