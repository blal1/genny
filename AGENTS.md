# genny — agent guide

`genny` is a CLI that synthesizes game sounds (instruments, drums, sound effects, effects chains) into `.wav` files. It is deterministic, has no external audio dependencies, and is meant to be driven by an AI agent that writes JSON specs.

**Read this file top to bottom once, then use `genny list --json` for the exact, current catalog of names and parameters. The catalog is the source of truth; this file explains how to combine things.**

## Setup / invocation

```
python install.py --tool                     # installs the `genny` command (uv) and the Claude Code skill
uv tool install --editable <path-to-genny>   # just the command
# or, from the repo:
uv run genny ...
```

Every command writes a WAV and prints `wrote <path> (<seconds>, mono|stereo, <sr> Hz)`. Non-zero exit on error with `error: ...` on stderr. Unknown instrument/effect/param names are rejected with the allowed list, so failures are cheap to fix.

## Two ways to talk to genny

1. **Quick one-shots** on the command line (`synth`, `seq`, `drum`, `sfx`, `say`, `fx`).
2. **JSON specs** via `genny render` — the main interface for anything layered or for batches. Prefer this when making more than one sound.

### One-shots

```
genny synth <inst> "<notes>" [--dur S] [--vel 0..1] [--strum S] [-p k=v ...] [--fx NAME:k=v,k=v ...] -o out.wav
genny seq   <inst> "<steps>" [--step S] [--legato F] [--gap S] [--transpose N] [--bpm N] [-p ...] [--fx ...] -o out.wav
genny drum  <kind> [-p k=v ...] [--fx ...] -o out.wav
genny sfx   <kind> [-p k=v ...] [--fx ...] -o out.wav
genny say   "<text>" [--phonemes] [--show] [-p k=v ...] [--fx ...] -o out.wav   # speech (see "Speech")
genny fx    in.wav --fx NAME[:k=v,...] [--fx ...] -o out.wav      # process an existing file
genny render spec.json [more.json ...] [--out-dir DIR] [--quiet]   # or:  genny render --json '{...}'   or:  ... | genny render -
genny list [instruments|drums|sfx|fx|speech|phonemes|words|waves|scales|chords] [--json]   # instruments: family + register
genny info file.wav ...      # duration, peak/rms, loudness, sharpness, energy above 5 kHz (--json)
genny play file.wav ...      # audition through the system player
genny example                # prints a full example spec
```

Examples:

```
genny sfx beep -p freq=880 -p dur=0.1 -o ui/beep.wav
genny sfx proximity -p rate=8 -p dur=1 -o ui/wall_close.wav
genny synth bell "C6" --dur 0.3 -p ratio=5.04 --fx reverb:mix=0.3 -o notif/ding.wav
genny seq brass "C4:0.15 E4:0.15 G4:0.15 C5:0.6" --fx reverb:mix=0.25,size=0.7 -o win/fanfare.wav
genny seq violin "D5:.5 E5:.5 F#5:1 A5:2" --bpm 100 --fx reverb:mix=0.2 -o tune.wav
genny drum kick -p tune=48 -p decay=0.6 -o drums/kick.wav
genny fx voice.wav --fx telephone --fx reverb:mix=0.1 -o voice_radio.wav
```

## Pitch notation

Two kinds of pitch input:

- **Frequency parameters** (`freq`, `freq2` on sfx such as `tone`/`beep`/`alarm`, speech `pitch`): **a number is always Hz** — `"freq": 55` is a 55 Hz hum. `"110hz"`, `"110"` and a note name (`"A2"`) also work.
- **Notes** (`notes` in synth layers, `steps` in seq layers, `genny synth`/`seq` arguments): `C4`, `F#3`, `Bb2`, a MIDI number `60`, or Hz `440hz` / `440.0`. Here a bare integer 0–127 is a MIDI note (musical context); write Hz with the suffix.

Multiple notes: `"C4 E4 G4"` or `"C4,E4,G4"`. Named chord: `"C4:maj"` (chords: `genny list chords` → maj min dim aug sus2 sus4 maj7 min7 dom7 add9 power maj9 min9 oct).

## Step notation (melodies / stingers)

Whitespace-separated tokens for `seq`:

| token | meaning |
|---|---|
| `C5` | note with default step duration (`--step`, default 0.25 s) |
| `C5:0.12` | note lasting 0.12 s |
| `[C5,E5,G5]:0.5` | chord for 0.5 s |
| `C5:maj:0.5` | named chord for 0.5 s |
| `-:0.1` | rest for 0.1 s |
| `C5:0.2@0.6` | velocity 0.6 (softer, usually darker) |
| `\|` | bar line, ignored (readability) |

Instruments add their own release/ring after each note, so notes overlap naturally. `legato` < 1 shortens the sounding part of each step (staccato); `gap` adds silence between steps. With a tempo (`"bpm"`, see "Music") the durations are beats instead of seconds.

## JSON spec format

```jsonc
{
  "out": "sounds/win/fanfare_01.wav",   // output path (relative to --out-dir if given)
  "sr": 44100,                          // optional
  "duration": 1.5,                      // optional hard length (pads or cuts)
  "layers": [ ...layer objects... ],    // mixed together on one timeline
  "fx": [ {"type": "reverb", "mix": 0.2}, {"type": "compressor"} ],   // master chain, in order
  "bpm": 120,                           // optional tempo: musical times in the layers are then beats (see "Music")
  "normalize": -1.0,                    // peak dB after fx; null to skip
  "max_loudness": -6.0,                 // ear-weighted ceiling applied after normalize; null to disable
  "gain": 1.0,
  "trim": true,                         // drop silent tail
  "declick": true,
  "loop": false                         // true or crossfade seconds: seamless loop; "fold": musical loop (see "Music")
}
```

### Layer objects

Common keys for every layer: `"at"` (start seconds, default 0), `"gain"` (linear, default 1), `"fx"` (effects chain applied to just this layer), `"repeat"` + `"every"` (repeat the layer N times every X seconds).

```jsonc
{"type": "synth", "inst": "bell", "notes": "C5 E5 G5", "dur": 0.4, "vel": 0.8, "strum": 0.03, "params": {"ratio": 3.5}}
{"type": "seq",   "inst": "pluck", "steps": "C4:0.1 E4:0.1 G4:0.3", "step": 0.25, "legato": 1.0, "gap": 0, "transpose": 0, "vel": 1, "params": {}}
{"type": "seq",   "inst": "piano", "steps": [{"note": "C4", "dur": 0.2}, {"notes": ["E4", "G4"], "dur": 0.4, "vel": 0.7}, {"note": "-", "dur": 0.1}]}
{"type": "drum",  "kind": "kick", "vel": 1.0, "params": {"tune": 50}}
{"type": "pattern", "kind": "hihat", "steps": "x-x-xXx-", "step": 0.125}      // x hit, X accent, o ghost; or "hits": [0, 0.5, 0.75]
{"type": "sfx",   "kind": "whoosh", "params": {"dur": 0.4, "direction": "down"}}
{"type": "speech", "text": "fuel low", "params": {"voice": "female"}}          // spoken words (see "Speech")
{"type": "file",  "path": "existing.wav"}                                      // import a sample
{"type": "silence", "dur": 0.5}
{"type": "group", "layers": [ ... ], "fx": [ ... ]}                              // sub-mix with its own fx
```

`params` must only contain keys the instrument/drum/sfx/speech declares (see `genny list --json`). Anything else is an error.

### Batches

```jsonc
{
  "out_dir": "sounds/notifications",     // prefixed to each relative "out"
  "fx": [{"type": "compressor"}],         // defaults inherited by sounds that don't set their own (also sr, normalize, max_loudness, bpm)
  "sounds": [ { ...spec... }, { ...spec... } ]
}
```
A top-level JSON array of specs also works. `genny render batch.json` renders all, reports `N rendered, M failed`, and continues past errors unless `--strict`.

## Catalog overview (run `genny list` for parameters)

**Instruments** (`synth`/`seq` layers). Each note = `inst(freq, dur)`, returns note plus release. `genny list instruments` shows every one with its family, the register it is voiced for (`range`) and its params.
- Keys: `piano`, `felt_piano` (soft, lo-fi), `epiano` (Rhodes), `wurli`, `clav`, `harpsichord`, `organ` (drawbars; `upper`), `church_organ`, `accordion` (`musette`), `celesta`, `toy_piano`
- Mallets and bells: `marimba`, `xylophone`, `vibraphone`, `glockenspiel`, `kalimba`, `music_box`, `steel_drum`, `handpan`, `bell` (`bright`, `ratio`), `glass`, `tubular_bell`, `gamelan`, `singing_bowl`, `timpani`
- Plucked: `pluck`, `harp`, `guitar` (nylon), `steel_guitar`, `electric_guitar` (clean), `muted_guitar`, `dist_guitar`, `banjo`, `mandolin` (`tremolo`), `koto`, `sitar`, `pizzicato`
- Bass: `bass` (analog synth), `sub`, `wobble`, `upright_bass`, `finger_bass`, `slap_bass`, `acid` (303), `reese`, `fm_bass`, `bass808`
- Bowed: `strings` (ensemble), `violin`, `cello`
- Winds: `flute`, `pan_flute`, `clarinet`, `oboe`, `sax`, `harmonica`, `whistle`, `ocarina`, `didgeridoo`
- Brass: `brass` (synth section), `trumpet` (`mute`), `french_horn`, `trombone`, `tuba`
- Synth leads and pads: `lead` (supersaw), `soft_lead`, `square_lead`, `synth_pluck`, `theremin`, `pad`, `warm_pad`, `glass_pad`, `string_machine`, `choir`, `pwm`
- Retro: `chip` (pulse; `width` 0.125/0.25/0.5, `bright`), `chiptri`, `board` (80s pinball/arcade sound-board voice: gritty pulse with an onset pitch blip)
- Raw: `sine`, `square`, `saw`, `synth` (generic and untamed: `wave`, ADSR, `cutoff`, `res`, `fenv`, fm `ratio`/`index`)

**Drums**: `kick`, `kick808`, `snare`, `snare808`, `clap`, `snap`, `brush` (jazz; `swish`), `hihat`, `openhat`, `tom`, `rim`, `cowbell`, `crash`, `ride`, `shaker`, `tambourine`, `woodblock`, `taiko`, `zap_kick`; hand and world percussion: `conga` (`slap`), `bongo`, `djembe` (`stroke` bass|tone|slap), `tabla` (`stroke` na|ge|ke), `cajon` (`stroke` bass|snare), `timbale`, `clave`, `guiro`, `triangle`, `gong`, `sleigh`. (`timpani` is an instrument: it takes notes.)

**SFX**: `beep`, `blip`, `click`, `pop`, `coin`, `powerup`, `powerdown`, `laser`, `zap`, `hit`, `punch`, `explosion`, `jump`, `whoosh`, `swoosh`, `alarm`, `siren`, `error`, `success`, `proximity`, `radar`, `riser`, `sweep_up`, `sweep_down`, `bubble`, `glitch`, `static`, `wind`, `thunder`, `footstep`, `door`, `engine`, `car_engine`, `magic`, `heartbeat`, `tone`, `noise`, `vinyl` (record crackle), `typewriter`, `countdown`; pinball/mechanical: `solenoid`, `flipper`, `pop_bumper`, `slingshot`, `knocker`, `steel_ball`, `ball_roll`, `spinner`, `spring`, `chirp` (see "Pinball / mechanical foley").

**Speech**: `speech` layer / `genny say` — formant voice for letters, numbers and short game words; voices `male`, `female`, `child`, `whisper`, plus character voices `robot`, `android`, `synth`, `bad_robot`, `evil_robot`, `monster`, `giant`, `alien`, `ghost`; English + Spanish (`lang=es`). See "Speech" below.

**Effects** (`"fx"` chains; also `--fx name:k=v,k=v`):
- Space: `reverb` (mix,size,damp,predelay,width), `delay` (time,feedback,mix,pingpong), `chorus`, `flanger`, `phaser`, `width`, `pan`, `autopan`
- Tone: `lowpass`, `highpass`, `bandpass`, `notch`, `eq`, `lowshelf`, `highshelf`, `sweep` (moving filter), `muffle` (through a wall), `telephone`, `underwater`
- Character: `distortion`, `bitcrush`, `ringmod`, `vibrato`, `tremolo`, `stutter`, `pitch` (semitones, same length), `speed` (tape speed), `reverse`
- Dynamics/utility: `compressor`, `limiter`, `gain`, `normalize`, `fade` (in,out), `trim`, `mono`

`reverb`, `chorus`, `pan`, `autopan`, `width`, and `delay pingpong` produce stereo files; everything else keeps the channel count.

## Extended catalog (physical models)

The names above are the core set. The modules below add the rest: 129 instruments, 48 drums, 234 sfx, 96 effects
and 13 extra layer types in all. `genny list` (or `genny list --json`, `genny list layers`) prints every name with
its parameters; `docs/catalog.md` is the same list as a page, `docs/physics.md` says which physical model is behind
each sound, and each module has a page with recipes.

- **contact** (`docs/contact.md`) — sfx: `roll`, `scrape`, `impact`, `bounce`, `drop`, `smash`, `crumple`, `avalanche`, `rumble`, `modal_engine`; fx: `resonate`, `cavity`
- **friction** (`docs/friction.md`) — instruments: `cristal`, `rubbed_glass`, `bowed_bar`, `bowed_sheet`, `rod_bank`, `whistling_blades`, `tuning_fork`, `coil_spring`; sfx: `stick_slip`, `squeak`, `brake_squeal`, `rub`; fx: `sheet_radiator`, `cone_radiator`, `sympathetic`
- **matter** (`docs/matter.md`) — sfx: `fire`, `pour`, `drip`, `drops`, `splash`, `babble`, `bubbles`, `boil`, `sizzle`, `fizz`, `gurgle`, `drain`, `surf`, `cave_drips`, `waterfall`, `underwater_ambience`, `steam`, `air_leak`, `spray`, `kettle`, `balloon`, `gust`, `flame`, `match`, `spark`, `arc`, `mains_hum`, `tesla`, `neon`, `lightning`, `ice_crack`, `ice_cubes`, `freeze`
- **foley** (`docs/foley.md`) — instruments: `church_bell`, `handbell`, `china_bell`, `glass_harmonica`, `bowed_bowl`, `hurdy_gurdy`, `wind_chime`; drums: `cabasa`, `sekere`, `maraca`, `bamboo_chimes`, `sandpaper`, `sleigh_bells`, `agogo`, `anvil`, `brake_drum`, `wind_chimes`; sfx: `hit`, `clang`, `clink`, `thud`, `knock`, `shake`, `chain`, `keys`, `cloth`, `creak`, `zipper`, `velcro`, `paper`, `coin_spin`, `dice`, `cork_pop`, `boing`, `ratchet`, `drawer`, `book_drop`, `gunshot`, `reload`, `casing`, `sword`, `clash`, `stab`, `slash`, `bow_shot`, `whip`, `shield_bash`, `clock`, `switch`, `button`, `keyboard_typing`, `mouse_click`, `camera_shutter`, `servo`, `printer`, `drill`, `saw`, `hammering`, `ratchet_wrench`, `piston`, `steam_engine`, `train`, `sonar`, `dtmf`, `phone_tone`, `phone_bell`, `siren_wail`, `car_horn`, `bike_bell`, `buzzer`, `shepard`, `transporter`, `robot_babble`, `red_alert`, `dark_drone`, `force_field`, `teleport`, `energy_blade`, `warp`, `charge_up`, `scanner`
- **creatures** (`docs/creatures.md`) — instruments: `whistling`; sfx: `heartbeat`, `roar`, `growl`, `bark`, `meow`, `purr`, `moo`, `howl`, `monster`, `animal`, `cricket`, `cicada`, `fly`, `mosquito`, `bee`, `night_insects`, `wings`, `birdsong`, `bird_call`, `dawn_chorus`, `frog`, `rattlesnake`, `hiss`, `breath`, `grunt`, `cough`, `sneeze`, `snore`, `gulp`, `eat`, `handclap`, `applause`, `crowd_murmur`, `whistle_human`, `footsteps`
- **choir** (`docs/choir.md`) — instruments: `voice`, `vocal_choir`, `hum`, `throat_singing`, `falsetto`, `boys_choir`, `chant`; layers: `sing`, `satb`
- **fdstring** (`docs/fdstring.md`) — instruments: `fd_string`, `fd_piano`, `prepared_piano`, `slack_string`, `fd_bar`, `bowed_string_fd`; sfx: `twang`; fx: `spring_reverb`
- **plates** (`docs/plates.md`) — drums: `fd_drum`, `plate_crash`, `plate_ride`, `plate_china`, `plate_gong`, `tam_tam`, `thunder_sheet`; sfx: `metal_plate`, `glass_pane`, `wood_panel`, `plate_bow`; fx: `plate_reverb`, `room2d`
- **tubes** (`docs/tubes.md`) — instruments: `reed_tube`, `reed_cone`, `vowel_tube`, `piston_pipe`, `pvc_pipe`, `bottle`; sfx: `pipe_blow`; fx: `formant_tract`, `tube`
- **reverbs** (`docs/reverbs.md`) — instruments: `feedback_guitar`; fx: `chorus`, `flanger`, `phaser`, `jcrev`, `satrev`, `fdn_reverb`, `early_reflections`, `room`, `zita`, `distance`, `doppler`, `occlude`, `leslie`, `tape_delay`, `shimmer`, `gated_reverb`, `reverse_reverb`
- **spectral** (`docs/spectral.md`) — fx: `timestretch`, `pitch_shift`, `harmonizer`, `vocoder`, `cross_synth`, `spectral_gate`, `sines_only`, `noise_only`, `freeze`, `spectral_blur`, `freqshift`, `robotize`, `whisperize`, `spectral_tilt`; layers: `resynth`
- **dsp** (`docs/dsp.md`) — sfx: `noise_bed`; fx: `convolve`, `fir_eq`, `steep_lowpass`, `steep_highpass`, `steep_bandpass`, `overdrive`, `wavefold`, `mulaw`, `dither`, `codec`, `lofi`, `smooth`, `tape`, `wall`, `dc_block`
- **retro** (`docs/retro.md`) — instruments: `sfxr_voice`, `zzfx_voice`, `psg`; sfx: `sfxr`, `zzfx`; layers: `song`
- **compose** (`docs/compose.md`) — instruments: `ks_string`; drums: `ks_drum`; layers: `compose`, `euclid`, `arp`, `stinger`, `scatter`, `arrangement`, `adaptive`
- **cyber** (`docs/cyber.md`) — sfx: `transformer`, `gearbox`, `bearing`, `pipe_engine`, `turbo`, `tyre_squeal`, `pass_by`, `cyber_ui` (kind: scan, confirm, deny, hack, decrypt, upload, warning, target_lock, implant, neural_link); effects: `packet_loss`, `beat_repeat`, `paulstretch`, `waveset`, `formant_shift`, `cyber_voice`, `texturize`; CLI: `genny identify file.wav`
- **klang** (`docs/klang.md`) — instruments: `viola`, `bassoon`, `cor_anglais`, `bass_clarinet`, `piccolo`, `recorder`, `hand_chime`; sfx: `toy_boat`, `klang_car`, `harrier`, `bicycle`, `klang_rain`, `ui_tick`, `ui_chime`, `damage`, `sniff`, `strain`, `weight_shift`, `handle`, `tide`, `distant_bell`; layers: `articulate`, `sampler`

Extra layer types are used like the built-in ones (`"type": "compose"`, `"sing"`, ...) and take the common layer keys
(`at`, `gain`, `fx`, `repeat`/`every`). The ones you will reach for most:

```jsonc
{"type": "sing", "section": "sopranos", "singers": 8, "steps": "C5:1 D5:1 E5:2", "lyrics": "A-ve Ma-ri-a", "style": "cathedral"}
{"type": "satb", "chords": "C4:maj:2 F4:maj:2 G4:dom7:2 C4:maj:2", "lyrics": "A-men A-men", "singers": 4}
{"type": "compose", "role": "melody", "inst": "flute", "key": "D", "scale": "dorian", "bars": 8, "seed": 7}
{"type": "arrangement", "style": "funk", "key": "E", "bars": 8, "seed": 3}
{"type": "euclid", "kind": "conga", "k": 5, "n": 8, "bars": 4}
{"type": "stinger", "mood": "victory", "inst": "trumpet", "seed": 1}
{"type": "song", "rows": 4, "instruments": {"lead": {"inst": "psg"}, "kick": {"drum": "kick"}}, "patterns": {"A": {"lead": "C-4 . E-4 . G-4 . C-5 .", "kick": "x . . . x . . ."}}, "order": ["A", "A"]}
```

Choosing by physics rather than by name: a struck, dropped, rolled, scraped or broken object is `impact`, `bounce`,
`drop`, `roll`, `scrape`, `smash` with `material` and `size`; anything rubbed or squeaking is in `friction`; water,
steam, fire, sparks and ice are in `matter`; a room is the `room` effect with a `preset` or its dimensions; a sound
heard from far away, moving past, or through a wall is `distance`, `doppler`, `occlude` / `wall`.

Changed from earlier versions: `pour` takes `fill_start` / `fill_end` (was `fill`); `heartbeat` is a lub-dub model
with `rate` and `dur`; `flanger`, `phaser`, `chorus`, `roll`, `scrape`, `hit`, `fire` are new models that still accept
their old parameters; plucked strings (`guitar`, `harp`, `banjo`, basses...) now ring for their natural decay after
the note ends, so write shorter notes or lower `gain` if a line smears.

## Instruments: register, brightness, loudness

The instruments are voiced so they do not pierce. Brightness is set in Hz, not in harmonics: upper partials roll off above about 3 kHz, FM clang is capped by the note and filters key-track with a ceiling. A voice is rich in its low register and gets purer as it climbs, like a real instrument, so a melodic layer rarely needs a `lowpass` to be bearable. What is left to you:

- **Write in the instrument's register.** The catalog gives a `range` for each (`piano` A0-C7, `upright_bass` E1-G3, `glockenspiel` G5-C8). Bass lines E1-G3, chords and pads C3-C5, melodies C4-C6. Above C6 use the voices made for it (`glockenspiel`, `celesta`, `music_box`, `glass`, `xylophone`, `whistle`) and keep them soft (`vel` 0.5-0.7, low `gain`): a clean tone at 2-4 kHz is where the ear is most sensitive.
- **Brightness controls**: `vel` (softer = darker on most voices), `bright` (`piano`, `bell`, `brass`, `chip`, `clav`, `synth_pluck`), `cutoff` (`lead`, `pad`, `soft_lead`, `warm_pad`, `bass`, `acid`, `reese`), `tone` (`electric_guitar`, `finger_bass`, `dist_guitar`, `hihat`). Anything else: a `highshelf` (+3 dB at 3000 for air) or `lowpass` in the layer `fx`.
- **Raw building blocks are not tamed**: the `synth` instrument and the `tone` sfx give exactly the wave and cutoff you ask for. Keep `cutoff` at or below about 4x the note (and under 3500) unless harshness is the point. The tonal sfx (`beep`, `blip`, `coin`, `powerup`, `laser`, `jump`, `alarm`...) round their square/saw waves themselves.
- **Levels are matched**: at the same `gain` and `vel` every instrument is about as loud as a kick or snare, so `gain` is the mix (see "Music"). High notes are trimmed 1.5 dB per octave above C5.
- **Loudness ceiling**: after peak normalisation, a sound that is still louder *to the ear* than `"max_loudness"` (default -6, ear-weighted dB) is turned down to it. Only held tones in the 1-5 kHz range ever reach it (a full-scale 3 kHz tone is 8 dB over); drums, noise, bass and short blips never do. `"max_loudness": null` on a spec or batch disables it.
- **Check without ears**: `genny info file.wav` prints `loud` (K-weighted level of the loudest 0.4 s), `ear` (ear-weighted level of the loudest 0.2 s: what the ceiling limits), `sharp` (sharpness in acum) and `above5k` (share of energy above 5 kHz). For anything with a pitch: `sharp` up to about 1.6 is soft, 1.6-2.0 bright, above 2.0 it bites (a pure tone already reads 1.0 at C6 and 1.5 at C7); `above5k` over 3 % is harsh. Hats, cymbals and shakers are 2.5-3.5 by nature: keep them quiet instead.

## Music (tempo, loops, styles)

Set `"bpm"` on a spec (or on a batch, a group or one layer) and every musical time in it is in **beats** instead of seconds: `at`, `dur`, the durations in `steps`, `step`, `gap`, `hits`, `every`. Performance and sound-design times (`strum`, instrument/sfx `params`, fx times, `tail`, `duration`) stay in seconds.

```jsonc
{
  "out": "funk_loop.wav", "bpm": 104,
  "beats": 16,                // hard length: 4 bars of 4/4
  "tail": 1.5,                // seconds of ring-out kept after the last beat (ignored with "loop": "fold")
  "layers": [
    {"type": "seq", "inst": "slap_bass", "step": 1, "steps": "E2:.5 -:.25 E2:.25 -:.25 G2:.25 A2:.5 -:.5 E3:.25 -:.25 D3:.5 E3:.5 | E2:1 -:1 G2:1 A2:1", "gain": 0.9},
    {"type": "seq", "inst": "clav", "step": 1, "steps": "-:.5 [G3,B3,D4]:.25 -:1.25 [A3,C#4,E4]:.25 -:1.75", "gain": 0.45, "at": 8},
    {"type": "pattern", "kind": "kick",  "steps": "X--x------X--x--", "bars": 4, "gain": 0.7},
    {"type": "pattern", "kind": "snare", "steps": "----X--o-o--X--o", "bars": 4, "gain": 0.55},
    {"type": "pattern", "kind": "hihat", "steps": "x-x-x-x-x-x-x-xx", "bars": 4, "gain": 0.25, "swing": 0.15}
  ],
  "fx": [{"type": "reverb", "mix": 0.14, "size": 0.45}, {"type": "compressor", "threshold": -16, "ratio": 2.5}]
}
```

- **`steps` for notes**: durations are beats (`C4:1` a quarter, `:.5` an eighth, `:.25` a sixteenth, `:.3333` a triplet eighth); a token without a duration lasts `step` beats (default 0.25, so set `"step": 1` for quarter-note lines). `|` is a bar line and is ignored. One-shot: `genny seq sax "C4:1 E4:.5 G4:1.5" --bpm 96`.
- **`pattern` for drums**: `steps` is a grid, by default sixteenths with a tempo (`"step": 0.25` beats; `0.3333` for triplets, `0.125` for 32nd rolls). `x` = hit, `X` = accent (1.4x), `o` = ghost note (0.4x), anything else = rest; spaces and `|` are ignored. `"bars": N` plays the written pattern N times. `"swing": 0..0.6` delays every other step by that fraction of a step (0.15 lazy, 0.33 shuffle; or write true triplets with `"step": 0.3333`).
- **Length and loops**: `"beats"` fixes the length. `"loop": "fold"` makes a sample-exact musical loop: whatever rings past the last beat (release tails, reverb) is laid back over the start, which is what the next repetition would sound like, so the file loops with no gap, click or missing tail. (`"loop": true` is the crossfade loop for drones and engines, not for music.)
- **Mix by `gain`** (instruments are level-matched): kick 1.0, snare/clap 0.6-0.85, hats/shaker/ride 0.25-0.4, crash 0.3-0.5, bass 0.8-0.9, chords and pads 0.35-0.5, arpeggios 0.3-0.4, lead 0.6-0.8, counter-melody 0.4-0.5. A `distortion` on a layer brings it to full scale: drop that layer to 0.25-0.35. Master: a small `reverb` (mix 0.12-0.25) and a gentle `compressor` (threshold -16, ratio 2.5), `normalize` -2.
- **Per-note samples for a game's own sequencer**: render one file per reference pitch every 6-12 semitones (the engine re-pitches by a few semitones), all with the same spec apart from the note. Do not bake LFOs, tremolo or detuned pairs into them (re-pitching changes their speed: a wobble that drifts against the beat); add movement at runtime or keep it very slow.

**Style palette.** `styles.json` (next to the skill's `SKILL.md`; in the repo `examples/library/styles.json`, written by `examples/build_styles.py`) holds a complete 4-8 bar loop for each of these, a dozen lines each: find `"out": "<style>.wav"` and read that entry before writing your own.

| style | bpm | drums | bass | harmony | lead and colour |
|---|---|---|---|---|---|
| `lofi` | 70-85 | soft `kick`, `snap` + `brush` backbeat, dark `hihat`, swing 0.18 | `upright_bass` | `epiano` 9th chords | `felt_piano`, `vinyl` sfx bed, master `lowpass` 5000 |
| `jazz_swing` | 120-180 | `ride` in triplets, `hihat` on 2 and 4, `brush` | walking `upright_bass` | sparse `piano` stabs | `trumpet` (`mute`), `sax`, `vibraphone` |
| `bossa_nova` | 120-140 | `rim` bossa clave, `shaker` 16ths, soft `kick` | `upright_bass` root-fifth | `guitar` syncopated chords | `flute` |
| `blues_shuffle` | 90-120 | triplet grid: `kick`, `snare`, `hihat` | walking `finger_bass` | `electric_guitar` boogie 5th-6th | `harmonica` |
| `soul_ballad` | 60-80 | `kick`, `snap`, quiet `hihat` | `finger_bass` | `wurli`, `strings` pad | `sax` |
| `funk` | 95-115 | syncopated `kick`, ghosted `snare`, 16th `hihat` | `slap_bass` | `clav`, `muted_guitar` scratch | `brass` stabs |
| `disco` | 115-125 | four-on-the-floor, off-beat `openhat`, `clap` | octave `finger_bass` | `string_machine`, `clav` chick | `violin` line |
| `reggae` | 70-80 | one drop: `kick` + `rim` on beat 3, swung `hihat` | spacious `finger_bass` (`tone` 1100) | `organ` + `muted_guitar` off-beat skank | `trombone` |
| `latin_salsa` | 90-100 | `clave` 2-3, `conga` tumbao, `cowbell`, `timbale`, `guiro` | anticipated `upright_bass` | `piano` montuno | `trumpet` |
| `reggaeton` | 90-98 | dembow: `kick` every beat, `snare808` on "a" of 1 and "and" of 2 | `bass808` | `warm_pad` | `marimba` hook |
| `rock` | 110-150 | `kick`/`snare` backbeat, 8th `hihat`, `crash` | `finger_bass` eighths | `electric_guitar` power chords + `distortion` fx | `dist_guitar` |
| `metal` | 150-190 | 16th double `kick`, half-time `snare` | `finger_bass` + light `distortion` | `electric_guitar` gallops + `distortion` drive 16 | |
| `pop` | 100-120 | `kick`, `clap`, 8th `hihat` | `finger_bass` | `piano` chords | `soft_lead`, `synth_pluck` arp |
| `synthwave` | 90-110 | four-on-the-floor, `snare808` with reverb, `tom` fills | `bass` eighths | `string_machine` | `lead` + chorus, `synth_pluck` arp with delay |
| `house` | 120-128 | four-on-the-floor, `clap`, off-beat `openhat` | off-beat `fm_bass` | `piano` stabs, `warm_pad` | |
| `acid_techno` | 128-136 | four-on-the-floor, off-beat `hihat`, `clap`, `rim` | `acid` 16ths, accents by `@vel` | | |
| `drum_and_bass` | 165-175 | two-step `kick`, `snare` on 2 and 4 with ghosts | long `reese` notes | `glass_pad` | `bell` stabs with delay |
| `trap` | 130-150 | half-time `clap` on 3, `hihat` 16ths with 32nd rolls | long `bass808` | `warm_pad` | dark sparse `bell` (`bright` 0.6) |
| `chiptune` | 130-160 | bitcrushed `kick`/`snare`, short `hihat` | `chiptri` | fast `chip` arpeggio (`width` 0.125) | `chip` (`width` 0.25) |
| `orchestral_heroic` | 90-120 | `timpani`, `taiko`, `crash` accents | `cello` | `strings` chords | `french_horn` theme, `trumpet` answer, `harp` gliss |
| `baroque` | 90-110 | none | `cello` walking line | `harpsichord` broken chords | `oboe`, `violin` |
| `ragtime` | 90-105 | none | `piano` left-hand stride | | `piano` syncopated right hand |
| `lullaby` | 70-90 (3/4) | none | | `celesta` broken chords, `glass_pad` | `music_box` |
| `spooky` | 80-100 | `timpani` | | `church_organ` diminished chords, `celesta` ostinato | `theremin`, `tubular_bell` |
| `christmas` | 100-120 | `sleigh` eighths | `upright_bass` | `strings`, `french_horn` | `glockenspiel`, `tubular_bell` |
| `celtic_jig` | 110-125 (6/8) | `djembe` as bodhran | | `harp` chords | `violin` fiddle (`attack` 0.025), `whistle` |
| `bluegrass` | 110-130 | none (the `mandolin` chop is the snare) | `upright_bass` on 1 and 3 | `steel_guitar` boom-chick | `banjo` roll, `violin` |
| `polka_oompah` | 110-125 | `kick`, `snare` off-beats, `crash` | `tuba` | `accordion` off-beats | `clarinet` |
| `tango` | 110-125 | none | `upright_bass` | staccato `accordion`, `piano` | `violin` (`vibrato` 1.4) |
| `andean` | 90-110 | low `taiko` as bombo, `shaker` | `guitar` bass notes | `guitar`, `mandolin` (`tremolo` 12) | `pan_flute` |
| `middle_eastern` | 90-105 | `djembe` bass/slap (dum-tek), `tambourine` | | `strings` drone, `pluck` riff | `oboe` in Hijaz (D Eb F# G A Bb C) |
| `indian_raga` | 80-100 | `tabla` ge/na/ke | | `sitar` drone (root + fifth) | `sitar` |
| `japanese` | 70-90 | sparse `taiko`, `woodblock` | | `koto` arpeggios in the In scale (D Eb G A Bb) | `pan_flute` (`breath` 0.5) |
| `gamelan` | 90-100 | `gong` on the cycle | `handpan` | interlocking `gamelan` patterns | `gamelan` core melody |
| `ambient` | 50-70 | none | | `warm_pad`, `glass_pad` | `handpan`, `singing_bowl`, reverb mix 0.4 |

## Speech (spoken letters, numbers, short words)

A built-in Klatt-style formant synthesizer (no models, no downloads) for short spoken cues: letter names, numbers, and short game words/phrases ("fuel low", "level up", "game over"). English by default; Spanish with `lang=es`.

```
genny say "G" -o say/g.wav
genny say "fuel low" -p voice=female -o say/fuel_low.wav
genny say "G" -p lang=es -o say/g_es.wav                      # "ge" /xe/
genny say "level up!" --show -o say/level_up.wav               # --show prints the phonemes it used
genny say "JH IY1" --phonemes -o say/g_raw.wav                 # raw ARPAbet
```

Layer (all common layer keys work: `at`, `gain`, `fx`, `repeat`/`every`):

```jsonc
{"type": "speech", "text": "fuel low", "params": {"voice": "female", "rate": 1.1}}
{"type": "speech", "text": "S", "params": {"lang": "es"}}          // "ese"
{"type": "speech", "phonemes": "F Y UW1 AH0 L | L OW1"}            // raw ARPAbet, "|" = word break, "_" = short pause
{"type": "speech", "text": "go /G OW1/ now"}                       // /slashes/ inside text = raw phonemes
```

Params (`genny list speech`):

| param | default | meaning |
|---|---|---|
| `voice` | `male` | `male`, `female`, `child`, `whisper` (unvoiced); robots/creatures: `robot` (monotone + ring-mod), `android` (clean, snapped pitch, faint ring + chorus), `synth` (vocoder-style saw carrier, semitone-snapped), `bad_robot` (bit-crushed, stutters/dropouts/pitch hiccups), `evil_robot` (very low, octave-down sub, slow ring, overdrive), `monster` (huge tract, growl, distortion), `giant` (big slow natural), `alien` (tiny tract, high, warbling ring), `ghost` (swimming whisper) |
| `pitch` | from voice | base F0 in Hz or a note name (male 115, female 205, child 280, robot 100, evil_robot 62, monster 58, alien 240); every preset still takes `pitch`/`formant_shift`/`rate` overrides |
| `rate` | 1.0 | speed multiplier |
| `formant_shift` | from voice | vocal-tract size: 0.85 = big/deep, 1.16 female, 1.28 child |
| `breath` | from voice | breathiness 0..1 |
| `vibrato` | 0 | depth in semitones |
| `intonation` | `auto` | `statement` (falling), `question` (rising), `exclaim`, `flat`; `auto` follows a final `?`/`!` |
| `emphasis` | 1.0 | pitch-accent strength |
| `lang` | `en` | `en` or `es` |
| `gap` | 0 | extra seconds between words (countdowns, spelling) |
| `seed` | 0 | tiny jitter/noise variation between takes |

Text handling: a lone letter (or an all-caps token of 2–5 letters not in the dictionary, e.g. `GPS`) is spoken as letter names; digits become number words (`21` → "twenty one"; Spanish "veintiuno"); `,` `.` `!` `?` split phrases with a pause. English words come from a built-in dictionary (`genny list words`: numbers, letters and ~250 common game words — go, ready, level, up, fuel, low, bomb, shield, magnet, turbo, time, game, over, perfect, combo, bonus, score, win, lose, danger, warning, coin, lap, left, right...) with a rough letter-to-sound fallback for anything else — check odd words with `--show` and pass phonemes if it guesses wrong. Spanish uses spelling rules (Castilian: `z`/`ce`/`ci` = TH, `j`/`ge` = X, `r` tap / `rr` trill, stress from accents and the penultimate rule), so any Spanish word works.

Phonemes (`genny list phonemes`), ARPAbet with stress digit on vowels (1 primary, 2 secondary, 0 unstressed):
- vowels `IY IH EH AE AA AO UH UW AH AX IX ER AXR`, diphthongs `EY AY OY AW OW`, Spanish pure vowels `A E I O U`
- stops `P B T D K G`, fricatives `F V TH DH S Z SH ZH HH X` (X = Spanish jota), affricates `CH JH`
- nasals `M N NG`, liquids/glides `L R W Y`, flap/tap `DX`, trill `RR`

Tips:
- Output is dry mono; `normalize: -3` is a good level. A touch of `reverb` (mix 0.08–0.15, size 0.3) makes it sit in a mix; `telephone` (+ `distortion` light) = radio/announcer; `voice: robot`/`android`/`synth` = computer, `bad_robot` = broken machine, `evil_robot`/`monster` = villain/boss, `alien`/`ghost` = otherworldly (character voices are mono and dry — add fx on top); `pitch` + `formant_shift` 0.85 = giant/boss voice.
- Keep utterances short (1–4 words). It is intelligible but clearly synthetic; single letters and numbers are the strongest, long sentences the weakest.
- Weakest sounds: `TH`/`DH`/`F`/`V` (quiet, easily confused), `NG`, and the letter-to-sound fallback for unknown English words.

## Pinball / mechanical foley

Real machines are mechanism, not music: build them from these instead of `bell`/`glass`/`marimba` (tonal mallets make a table sound like a bell tree). Each is a short noise transient + inharmonic steel modes that die in tens of ms + a wooden-cabinet thump.

- `solenoid` — any coil or relay: `size` 0 (relay/switch tick) .. 1 (big coil), `metal` ring, `thump` cabinet thud, `bounce` armature rebound (the "ka-chak"). Use size 0 for menu ticks and switch hits.
- `flipper` (`force`, `buzz` AC coil hum), `pop_bumper` (`tone` 0.6..1.6 tells bumpers apart), `slingshot` (`tone`), `knocker` (the replay/extra-ball BANG; `size`, `rattle`).
- `steel_ball` — ball impact, `surface` metal (rail/post clank) | wood (tock) | rubber (thup) | plastic (target/ramp clack), `force` 0..1.
- `ball_roll` — steel ball on a wooden playfield, `speed` 0..1; loop it with `"loop": 0.3`.
- `spinner` — one flap tick per pass (`spins` > 1 = a decelerating whirr); `spring` — plunger, `action` pull (creak tick) | release (rod slam + boing), `tension`.
- Solid-state sound board (80s pinball/arcade speaker): `chirp` sfx (pitch sweep with `warble` LFO, `steps` for stepped bip-bips, `bits` DAC grit) and the `board` instrument (gritty pulse with a pitch blip on every note) for score blips and jingles; `chiptri` makes the bass. Recipe: foley layer + one short `chirp`/`board` blip = a scoring hit (e.g. `pop_bumper` + square `chirp` 980→330 Hz 0.07 s `bits` 5 at gain 0.3).

## Design recipes (what tends to sound right)

- **UI / notification**: short (0.1–0.6 s), C5–C6 (up to C7 only with `glass`/`celesta`/`glockenspiel`/`music_box` at `vel` ≤ 0.7), `bell`/`glass`/`marimba`/`kalimba`/`celesta`, a small `reverb` (mix 0.15–0.3, size 0.3–0.5). Two rising notes = positive, two falling = negative. Keep `normalize` at -1 to -3 dB.
- **Wall proximity / radar**: `sfx proximity` with `rate` 2 (far) → 12 (close); or a `beep` layer with `repeat`/`every`.
- **Stingers**: `seq` 3–6 notes, 0.08–0.15 s each, ending on a longer note or chord; layer a `pad`/`strings` chord underneath at `gain` 0.4–0.6; master `reverb` mix 0.2–0.35. For a flavour, borrow the lead and harmony instruments of a style from the palette in "Music" (`koto` + `pan_flute`, `banjo` + `violin`, `harpsichord` + `oboe`...).
- **Win / fanfare**: `brass`, `trumpet` or `lead` arpeggio up a major triad ending on the octave chord, `french_horn`/`strings` underneath; add `crash`, `timpani` or `taiko` at `at: 0`; `chip`+`bitcrush` for retro.
- **Game over**: descending minor line (`piano`, `felt_piano`, `cello`, `chip`, `organ`), slow `pad`/`warm_pad` minor chord, `powerdown` sfx, `taiko`/`kick808`/`gong`; darken with `lowpass` 1500–3000 or `muffle`; long `reverb` (size 0.8).
- **Impacts**: `hit`/`punch`/`explosion` + `kick` layer for weight; `distortion` for crunch; `lowpass` sweep for distance.
- **Layering rule of thumb**: 2–4 layers; one body (low), one transient (click/noise), one tail (reverb/pad). Sum gains ≈ 1–1.5, normalize handles the rest.
- **Variations**: change `vel`, `transpose`, `params.decay`, effect `mix`, or the note set. Same spec with a different `seed`-free variation still renders identically, so name files by intent (`ui_confirm_soft.wav`).

## Typical agent workflow

1. `genny list --json` once, cache it.
2. Write one batch JSON per folder/category with descriptive `out` names.
3. `genny render batch.json --quiet`; read the `N rendered, M failed` line and fix any `ERROR in spec #i` lines.
4. `genny info` on outputs if you need to verify durations/levels. `genny play` to audition (blocks until done).

## Vehicle engines (`car_engine`)

`{"type": "sfx", "kind": "car_engine", "params": {"rpm": 3000, "load": 0.6}}` is a physically-modelled combustion engine: per-cylinder firing pulses (4-stroke, fixed per-cylinder strength = the lope), exhaust pipe comb + body resonances + muffler lowpass that opens with `load`, a low exhaust burble and intake breath modulated by the firing. Params: `rpm` (crank speed), `cylinders` (4), `load` 0..1 (throttle: brighter, more intake), `rough` 0..1 (uneven firing, jitter, misfires), `pipe` (exhaust delay s, longer = throatier), `body` 0..1, `bright` 0..1 (0 = road car from the cabin, nothing above ~2.5 kHz; raise for an open sporty exhaust with combustion rasp), `seed` (engine character), `dur`. Keep `bright` at 0 for the player's own car: sharp/raspy engine timbres read as piercing and "too loud" long before their level is.

- **Engine bank for a player car:** render 4–6 loops at fixed rpm (e.g. 1100, 2000, 3000, 4200, 5600; `"loop": 0.3`, `duration` 3.3) and at runtime equal-power crossfade the two nearest, each played at `rate = rpm / renderedRpm`. One sample pitched across the whole range sounds like a tape speeding up.
- **Ignition / stall:** chain short `car_engine` layers — cranking is `rpm` 220–300 with `rough` 0.9 and a starter-motor `synth` saw with tremolo; a catch is a burst at 2300–2600 then settling to idle; a stall is 1100 → 700 → 420 with fades.
- The generic `engine` sfx is a simple saw hum; prefer `car_engine` for anything the player drives.

## Loops (for runtime engines that loop a sample)

Set `"loop": true` (or `"loop": 0.08` for a custom crossfade in seconds) at the top level of a spec. The last `crossfade` seconds are blended into the start with an equal-power crossfade and cut off, so playing the file with `loop=true` in a game engine has no click or gap. `trim`/`declick` are skipped for loop specs (they would break the seam). Use a `duration` long enough to hide the repetition (2–6 s for drones, wind, engines). The same thing is available as an effect: `{"type": "loop", "crossfade": 0.05}`. For music use `"loop": "fold"` with `"bpm"` + `"beats"` instead (see "Music"): the ring-out is folded onto the start and the length is exact.
