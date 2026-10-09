---
name: genny
description: Generate game sounds and music as .wav files with the genny CLI from a free-form request, e.g. "/genny 10 notification sounds", "/genny bell + hit", "/genny a 4 bar bossa nova loop", "/genny cat meows and is thrown in a blender". Use whenever the user wants sound effects, UI sounds, jingles, stingers, music loops in any style, drums, instrument samples, short spoken words/letters/numbers, or any audio for a game or app.
---

# genny — make sounds from a one-line request

`genny` synthesizes sounds and music offline (129 instruments, 48 drums, 234 sound effects from physical models, sung and spoken voice, 96 effects, generated music) into WAV files from JSON specs. You are the sound designer: turn the request into specs, render them, report the files. Do not ask clarifying questions; make reasonable choices and say what you chose.

## 0. Check the tool

```
genny --version
```
If it is missing: `git clone https://github.com/blal1/genny`, then `python install.py --tool` in that folder (installs the `genny` command with uv and refreshes this skill; `AGENTS.md` in the repo is the full manual). If `genny list instruments` has no `violin`, the install is an old copy: `git pull` in the clone and run `python install.py --tool` again. If uv fails with `os error 448` ("untrusted mount point") on a managed Python, add `--python "C:\Program Files\Python314\python.exe"` to use the system Python instead.

## 1. Read the manual (once per session)

Read `reference.md` next to this file. It documents the spec format, note/step notation, tempo (`"bpm"`: write music in beats), every layer type, instrument registers and design recipes. For music, `styles.json` next to it holds a complete short loop for each of 35 styles (lofi, jazz_swing, bossa_nova, funk, rock, synthwave, trap, orchestral_heroic, celtic_jig, indian_raga, japanese...): open only the entry you need (search `"out": "<style>.wav"`). Then run `genny list --json` and keep the output in mind: it is the exact catalog of instruments, drums, sfx, effects and their parameters. Unknown names or params are rejected, so stick to the catalog.

## 2. Interpret the request

- **"N <category> sounds"** → N distinct sounds in that category. Vary instrument, pitch, contour (rising/falling), length and effects; no near-duplicates. Default count is 1 for a singular request, 5 for a plural one without a number.
- **"A + B"** → one sound that layers A and B (e.g. `bell + hit` = a bell note layer plus a `hit` sfx layer, timed together). Render 2–3 variations unless told otherwise.
- **Imaginative / descriptive requests** ("cat meows and is thrown in a blender", "haunted elevator ding") → decompose the scene into stages on the timeline and build each stage from primitives:
  - spoken words / letters / numbers / announcer callouts ("G", "fuel low", "level up!", "tres"): a `speech` layer (`{"type": "speech", "text": "fuel low", "params": {"voice": "female"}}`, `lang: "es"` for Spanish; character voices `robot`, `android`, `synth`, `bad_robot`, `evil_robot`, `monster`, `giant`, `alien`, `ghost`; set `pitch` in Hz, e.g. `90`); `genny say "text" --show` prints the phonemes; see the Speech section of `reference.md`
  - voices/animals: `flute`/`choir`/`synth` with `vibrato`, `tone` with pitch sweeps via `sweep_up`/`sweep_down`, `pitch`, `ringmod` for weirdness
  - pinball/arcade machines, switches, coils: `solenoid`, `flipper`, `pop_bumper`, `slingshot`, `knocker`, `steel_ball`, `ball_roll`, `spinner`, `spring`, plus `chirp`/`board` for sound-board bleeps (no bells)
  - vehicles: `car_engine` (realistic, rpm/load/cylinders; render an rpm bank and crossfade at runtime — see reference)
  - machines: `engine`, `tone` with `saw`, `distortion`, `ringmod`, `tremolo`, `bitcrush`
  - impacts/violence: `hit`, `punch`, `explosion`, `glitch`, `stutter`, `speed` (tape stop), `reverse`
  - space/place: `reverb` size, `muffle`, `telephone`, `underwater`, `delay`
  Use `at` offsets so the story reads left to right. Aim for 1–4 s total.
- **Music** ("a jazz loop", "8 bars of synthwave", "boss theme", "menu music"): set `"bpm"` and write in beats; pick the instruments, rhythm and mix levels from the style palette in `reference.md` ("Music"), starting from the nearest entry of `styles.json`. Write your own notes: change key, progression and melody, keep the style's instrumentation and groove. `"beats"` fixes the length; add `"loop": "fold"` when the game will loop it. Default 4 bars (8 for slow styles), 2-4 variations only if asked.
- **Style words**: "retro/8-bit" → `chip`, `chiptri`, `bitcrush`; "cinematic" → `taiko`, `timpani`, `strings`, `french_horn`, `choir`, big `reverb`; "soft/subtle" → `celesta`, `glass`, `marimba`, `felt_piano`, low `vel`, small reverb; "warm/acoustic" → `guitar`, `upright_bass`, `felt_piano`, `cello`; "lo-fi" → `epiano`, `felt_piano`, `vinyl`, `lowpass`; "spooky" → `theremin`, `church_organ`, `tubular_bell`; a country or region → its row in the style palette (`koto`, `sitar` + `tabla`, `pan_flute`, `accordion`, `banjo`...); "harsh" (only when asked) → `distortion`, raw `synth` square, `alarm`, `error`.

## 3. Write a batch and render it

Put everything in one batch file under `sounds/<slug>/` in the current working directory (or wherever the user said), with descriptive file names:

```json
{
  "out_dir": "sounds/notifications",
  "sounds": [
    {"out": "chime_two_up.wav", "layers": [{"type": "seq", "inst": "bell", "steps": "G5:0.12 C6:0.4"}],
     "fx": [{"type": "reverb", "mix": 0.2, "size": 0.4}]}
  ]
}
```

```
genny render sounds/notifications/batch.json --quiet
```

Read the final `N rendered, M failed` line. Fix every `ERROR in spec #i` (usually a bad param name; check `genny list`) and re-render. Keep the batch JSON next to the WAVs so the user can tweak and re-run.

## 4. Report

List the files with one short phrase each describing the sound and how to vary it (which param to change). Mention `genny play <file>` to audition and `genny render <batch.json>` to re-render after edits. Keep the report short.

## Quality rules

- Notifications/UI: 0.05–0.6 s, pitched C5–C6 (C7 only with `glass`/`celesta`/`glockenspiel`/`music_box`, `vel` ≤ 0.7), normalize -1 dB. Game-over/win stingers: 1–4 s. Never exceed ~6 s unless asked (music loops: as many bars as asked).
- **Never piercing.** Write each instrument in its `range` (`genny list instruments`), keep hats/cymbals/shakers at `gain` 0.25–0.4, and do not reach for the raw `synth` with a high `cutoff` or for `highshelf` boosts unless the request is for something harsh. After rendering, run `genny info` on anything tonal: `sharp` above ~2.0 or `above5k` above 3 % means it bites: lower the register, the `vel`, the `bright`/`cutoff`, or swap the instrument, and re-render.
- Layer 2–4 things (body + transient + tail) rather than one raw oscillator.
- Add a small `reverb` to anything melodic; leave pure UI clicks/blips dry.
- Pitch: `freq`/`freq2`/speech `pitch` numbers are Hz (`"freq": 55` = 55 Hz). In `notes`/`steps` a bare integer is a MIDI note, so write Hz there as `"110hz"`.
- Do not ask questions before rendering. Render first, then offer variations.

## Sounds for games (runtime cues)

When the sounds feed a game engine that loops/positions them (the user's audio games):
- **Loops that mark a place must pulse fast**: ≤ ~0.5 s between onsets (a spoken beacon letter back to back, a cone "bloop bloop" every 0.4 s). A player can't steer toward something heard every 1.5 s.
- **Several of one kind will overlap**: plan per-instance pitch offsets (render variants or pitch at runtime, a few semitones apart) so they don't merge into one voice.
- **Positional cues**: mono (reverb/chorus/pan make stereo — add `{"type": "mono"}`), with a bright layer above ~2 kHz (`static` white → `highpass` 1800 → `bandpass` ~3000) so binaural can place them.
- **Never piercing**: the player's own continuous bed (engine, wind, wheels) keeps nothing above ~2.5 kHz (`car_engine` with `bright` 0); harsh/raspy timbre reads as "too loud" long before level does. UI/pickups: marimba/kalimba/celesta ≤ C6.
- **Per-pitch sample sets** (the game's own sequencer re-pitches them): one spec per reference pitch, identical apart from the note, no tremolo/vibrato/detuned layers baked in (re-pitching changes their speed and they drift against the beat). The renderer's loudness ceiling keeps the top notes of a set from screaming; do not disable it.
- **Objects must sound like the object** (a fuel can = metal tonk + slosh + glug, not a drone) — foley from shaped noise, tones only for abstract signals.
- **Informational speech** (scores, countdowns, menus) usually belongs to the game's screen reader; render speech only for voices that must come *from somewhere* in the world or are part of the sound design.
