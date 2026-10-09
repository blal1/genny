# Retro generators (`genny/retro.py`)

| type | name | what it is |
|---|---|---|
| sfx | `sfxr` | The sfxr engine (jsfxr-faithful): 22 knobs, seeded presets, mutate; `engine: "bfxr"` adds waves, harmonics, crush, compression |
| sfx | `zzfx` | The ZzFX 1.3.2 sound function: paste a ZzFX array, use named slots, or a seeded designer preset |
| instrument | `sfxr_voice` | The raw sfxr oscillator as a pitched voice (C2-C6) |
| instrument | `zzfx_voice` | A ZzFX array played at the note's pitch (C2-C6) |
| instrument | `psg` | AY-3-8910 sound-chip voice: clock-divider pitch, 16-step DAC, LFSR noise, hardware envelope (C2-C6) |
| layer type | `song` | Tracker layer: patterns of cells per channel, order list, ZzFXM import |

sfxr and ZzFX are defined at 44100 Hz, so both always run there and are resampled to the spec's `sr`.
Everything is deterministic: the same params and `seed` give the same samples.

## sfx `sfxr`

Order of precedence: defaults, then `preset` (drawn from `seed`), then `mutate` rounds, then any knob you
write. A knob you write is never changed by the preset or by mutate.

| param | default | meaning |
|---|---|---|
| `preset` | `none` | `pickup` (`coin`), `laser` (`shoot`), `explosion`, `powerup`, `hit` (`hurt`), `jump`, `blip` (`select`), `synth`, `tone` (440 Hz sine, 1 s), `click`, `random` |
| `seed` | 0 | int or string. A different seed is a different sound of the same kind |
| `mutate` | 0 | rounds of the sfxr "mutate": each knob has a 50% chance of moving by up to 0.05 |
| `wave` | preset's, else `square` | `square`, `saw`, `sine`, `noise`; with `engine: "bfxr"` also `triangle`, `breaker`, `tan`, `whistle`, `bitnoise` |
| `engine` | `sfxr` | `sfxr` or `bfxr` |
| `level` | 0.8 | peak level of the result |
| `max_dur` | 3.0 | longest sound in seconds (cut with a 20 ms fade) |
| `soften` | 1.0 | 0 = raw sfxr, 1 = top end rounded (12 dB/oct low-pass at 4 kHz) |
| `hz` | - | base pitch in Hz or a note name; overrides `freq` |

The 22 knobs (0..1, the signed ones -1..1):

| knob | default | meaning |
|---|---|---|
| `attack`, `sustain`, `decay` | 0, 0.3, 0.4 | envelope stage lengths: v² x 2.27 s each |
| `punch` | 0 | the sustain starts up to 3x louder and falls back |
| `freq` | 0.3 | base pitch: Hz = 3528 x (v² + 0.001). 0.3517 = 440 Hz, 1 = 3532 Hz |
| `freq_limit` | 0 | same scale as `freq`; the sound ends when the pitch falls to it |
| `slide`, `dslide` | 0 | pitch slide (above 0 = up) and its acceleration (signed) |
| `vib_depth`, `vib_speed` | 0 | vibrato depth and speed (v² x 70 Hz) |
| `arp_mod`, `arp_speed` | 0 | one pitch jump (signed: 0.745 = up an octave, -0.316 = down an octave) after (1-v)² x 0.45 s |
| `duty`, `duty_sweep` | 0 | square: 0 = 50% .. 1 = thin pulse; saw: 0 = triangle, 1 = pure saw (set for you when you write `wave: "saw"`) |
| `repeat` | 0 | retrigger of pitch, slide, duty and jump every (1-v)² x 0.45 s |
| `flange_offset`, `flange_sweep` | 0 | flanger delay (v² x 23 ms) and its sweep (signed) |
| `lpf`, `lpf_sweep`, `resonance` | 1, 0, 0 | resonant low-pass: cutoff (1 = off), sweep (signed), resonance |
| `hpf`, `hpf_sweep` | 0 | high-pass cutoff and sweep (signed) |

`engine: "bfxr"` extras: `harmonics` (0..10 overtones), `harmonics_falloff` (0..1), `crush` (sample-and-hold
rate reduction 0..1), `crush_sweep` (-1..1), `compression` (0..1, lifts the quiet parts).

```json
{"type": "sfx", "kind": "sfxr", "params": {"preset": "coin", "seed": 7}}
{"type": "sfx", "kind": "sfxr", "params": {"preset": "explosion", "seed": "boss", "mutate": 2, "decay": 0.6}}
{"type": "sfx", "kind": "sfxr", "params": {"wave": "saw", "hz": "A3", "slide": -0.25, "sustain": 0.2, "decay": 0.3, "lpf": 0.6, "resonance": 0.7}}
{"type": "sfx", "kind": "sfxr", "params": {"engine": "bfxr", "wave": "whistle", "preset": "powerup", "seed": 3, "harmonics": 3, "crush": 0.3}}
{"type": "sfx", "kind": "sfxr", "params": {"preset": "laser", "seed": 12, "soften": 0}}
```

## sfx `zzfx`

Order of precedence: ZzFX defaults, then `preset`, then `p`, then named slots.

| param | default | meaning |
|---|---|---|
| `p` | - | the ZzFX array, up to 21 slots, `null` where the JavaScript has a hole |
| `preset` | - | `random`, `pickup`, `powerup`, `jump`, `shoot`, `blip`, `hit`, `explosion` (the ZzFX designer's generators) |
| `seed` | 0 | preset draw and the `randomness` pitch jitter |
| `level`, `max_dur`, `soften` | 0.8, 5.0, 1.0 | as for `sfxr` |

Named slots, in array order: `volume` 1, `randomness` 0.05, `frequency` 220 Hz, `attack` 0 s, `sustain` 0 s,
`release` 0.1 s, `shape` 0 (0 sine, 1 triangle, 2 saw, 3 tan, 4 noise, 5 square), `shape_curve` 1 (waveform
power; for the square: duty x 2), `slide` 0, `delta_slide` 0, `pitch_jump` 0 Hz, `pitch_jump_time` 0 s,
`repeat_time` 0 s (resets pitch and re-arms the jump, so each repeat steps the pitch again; also the tremolo
period), `noise` 0, `modulation` 0 Hz, `bit_crush` 0 (hold period x 100 samples), `delay` 0 s,
`sustain_volume` 1, `decay` 0 s, `tremolo` 0, `filter` 0 Hz (above 0 high-pass, below 0 low-pass).

```json
{"type": "sfx", "kind": "zzfx", "params": {"p": [null, null, 925, 0.04, 0.3, 0.6, 1, 0.3, null, 6.27, -184, 0.09, 0.17]}}
{"type": "sfx", "kind": "zzfx", "params": {"preset": "powerup", "seed": 4}}
{"type": "sfx", "kind": "zzfx", "params": {"frequency": 300, "shape": 5, "shape_curve": 0.5, "sustain": 0.1, "release": 0.2, "slide": 6, "filter": -1500}}
```

## instrument `sfxr_voice`

The sfxr oscillator on a note: the pitch comes from the note, the sustain stage from its length.
Harsher than `chip`; `bright` 1 is the raw wave.

`wave` (`square`; any of the nine waves), `duty` 0, `duty_sweep` 0 (PWM), `attack` 0.03, `decay` 0.1 (the
release), `punch` 0, `vib_depth` 0, `vib_speed` 0.5, `slide` 0, `arp_mod` 0, `arp_speed` 0.6, `lpf` 1,
`resonance` 0, `hpf` 0, `harmonics` 0, `harmonics_falloff` 0.5, `bright` 0.5, `seed` 0. Knob scales as in `sfxr`.

```json
{"type": "seq", "inst": "sfxr_voice", "steps": "C4 E4 G4 C5", "params": {"duty": 0.5, "vib_depth": 0.15}}
{"type": "seq", "inst": "sfxr_voice", "steps": "C2:0.5 G2:0.5", "params": {"wave": "saw", "lpf": 0.45, "resonance": 0.7}}
```

## instrument `zzfx_voice`

A ZzFX array as an instrument. Slot 2 (frequency) follows the note and `randomness` is forced to 0, so it is
in tune unless the array itself slides or jumps. The `volume` slot (0..3) scales the level.

`p` (the array), `hold` (true: the sustain slot follows the note length; false: the array's own envelope, cut
at the end of the note, as ZzFXM plays it), `shape`, `shape_curve`, `attack`, `decay`, `sustain_volume`,
`release`, `bright` 0.5.

```json
{"type": "seq", "inst": "zzfx_voice", "steps": "A3 C4 E4", "params": {"shape": 1, "release": 0.3}}
{"type": "seq", "inst": "zzfx_voice", "steps": "C3:1 G3:1", "params": {"p": [1, 0, 220, 0.01, 0, 0.2, 2, 1.5, null, null, null, null, null, null, 4]}}
```

## instrument `psg` (AY-3-8910)

What the chip does, and what each param maps to:

- **Pitch grid.** The tone is the clock divided by 16 and then by a 12-bit period TP: f = clock / (16 x TP).
  The voice picks the nearest TP, so high notes land slightly off like on the real machine: under 2 cents at
  C6, 13 cents at 2 kHz with the 1.7734 MHz clock. `clock` moves the grid (1773400 ZX Spectrum, 2000000
  Atari ST, 1789772 MSX).
- **Volume.** `level` 0..15 on a 16-step logarithmic DAC, 3 dB per step.
- **Noise.** `noise` 0 = the tone channel, 1 = a noise channel, in between both mixed. The noise is a 17-bit
  shift register clocked at clock / (16 x `noise_period`), `noise_period` 1 (hiss) .. 31 (rumble).
- **Hardware envelope.** `shape` 0..15 replaces `level` with the chip's 16-step ramp generator; `env_hz` is
  ramps per second. `shape` -1 = off.

| shape | pattern |
|---|---|
| 0-3, 9 | falls once, then silent |
| 4-7, 15 | rises once, then silent |
| 8 | falling sawtooth, repeating |
| 10 | triangle, starting down |
| 11 | falls once, then jumps to full and holds |
| 12 | rising sawtooth, repeating |
| 13 | rises once and holds |
| 14 | triangle, starting up |

Other params: `bright` 0.5 (1 = raw square), `seed` (noise start state).

```json
{"type": "seq", "inst": "psg", "steps": "C4 E4 G4 C5", "step": 0.12}
{"type": "seq", "inst": "psg", "steps": "C5:0.3 G4:0.3", "params": {"shape": 9, "env_hz": 5}}
{"type": "seq", "inst": "psg", "steps": "C2:0.5 C2:0.5", "params": {"shape": 10, "env_hz": 65.4, "bright": 0.8}}
{"type": "seq", "inst": "psg", "steps": "C4:0.1 -:0.15 C4:0.1", "params": {"noise": 1, "noise_period": 4, "shape": 9, "env_hz": 12}}
```

The third recipe is the "buzz bass": the envelope runs at the note's own frequency and becomes the waveform.
The fourth is a noise hi-hat.

## layer type `song`

A tracker. Each pattern holds one row string per channel; `order` lists the patterns to play.

| key | default | meaning |
|---|---|---|
| `rows` | 4 | rows per beat (4 = a row is a sixteenth). Without a `bpm` a beat is one second |
| `instruments` | - | channel name to instrument (below). A channel that is not listed is looked up in the catalog by its own name |
| `patterns` | required | `{name: {channel: "cells"}}`. A pattern is as long as its longest channel |
| `order` | all patterns | pattern names in playing order; repeat a name to repeat the pattern |
| `pan` | - | `{channel: -1..1}`; any pan makes the layer stereo |
| `tail` | 0 | seconds of ring-out kept after the last row |
| `format`, `data` | - | `"zzfxm"` + a raw ZzFXM song array `[instruments, patterns, sequence, bpm]` |

Cells, separated by spaces (`|` is ignored):

| cell | meaning |
|---|---|
| `C-4`, `C#4`, `Bb3` | note (tracker or genny spelling). It sounds until the next note or note-off on that channel |
| `.` | nothing: the current note keeps ringing |
| `^^` | note off |
| `C-4@0.6` | per-cell volume 0..1 |
| `x`, `X`, `o` | drum channels: hit, accent (1.4x), ghost (0.4x) |

Instrument forms: `"kick"` (a catalog name: instrument or drum), `{"inst": "psg", "params": {...}}`,
`{"drum": "hihat", "params": {...}}`, `{"sfxr": {<sfxr_voice params>}}`, `{"zzfx": [<ZzFX array>]}`.
Each takes optional `"gain"` and `"pan"`.

The layer is exactly `total rows x row length` long (plus `tail`), so it lines up with `beats` and
`"loop": "fold"`.

```json
{"bpm": 120, "layers": [{"type": "song", "rows": 4,
  "instruments": {
    "lead": {"zzfx": [1, 0, 220, 0.01, 0.1, 0.2, 1, 1.5], "pan": -0.4},
    "bass": {"sfxr": {"wave": "triangle"}, "gain": 0.8},
    "arp":  {"inst": "psg", "params": {"shape": 9, "env_hz": 6}, "pan": 0.4},
    "hat":  {"drum": "hihat", "gain": 0.3}},
  "patterns": {
    "A": {"lead": "C-5 . E-5 . | G-5 . ^^ . | C-6@0.5 . . . | G-5 . . .",
          "bass": "C-2 . . . . . . . G-2 . . . . . . .",
          "arp":  "C-4 E-4 G-4 C-5 C-4 E-4 G-4 C-5 C-4 E-4 G-4 C-5 C-4 E-4 G-4 C-5",
          "kick": "x . . . x . . . x . . . x . X .",
          "hat":  ". . x . . . x . . . x . . . x o"},
    "B": {"lead": "F#4 . . . ^^ . . .", "kick": "x . . . x . . ."}},
  "order": ["A", "A", "B", "A"]}]}
```

ZzFXM import (`null` where the JavaScript array has a hole):

```json
{"layers": [{"type": "song", "format": "zzfxm", "data": [
  [[1, 0, 220, null, 0.05, 0.1, 1], [2, 0, 440, null, null, 0.05, 2]],
  [[[0, -1, 13, null, null, null, 25, null, -1, null], [1, 1, null, null, 20.5, null, null, null, 20, null]]],
  [0, 0], 150]}]}
```

A ZzFXM channel is `[instrument, pan, note, note, ...]`. Note `N.A`: pitch = the instrument's frequency x
2^((N-12)/12), volume = 1 - A; `-1` = note off. A row lasts `floor(floor(44100 / bpm x 60) / 4)` samples at
44100 Hz, as in `zzfxm.js`, so the length matches the original player.

## Differences from the originals

- `sfxr`: an envelope stage of length 0 gives silence for its one sample where jsfxr gives NaN. Mutated and
  hand-written knobs are clamped to their range; preset draws are left as jsfxr makes them.
- `engine: "bfxr"` is the sfxr loop plus Bfxr's waves, harmonics, crush and compression. Bfxr's bugs are not
  reproduced: triangle and breaker have no DC offset, `crush` 0 is transparent, `bitnoise` follows `freq`
  like every other wave, harmonics are normalised instead of clipped. Bfxr's two pitch jumps, its own preset
  generators and its three sampled waves (rasp, fmsyn, voice) are not included.
- `zzfx`: sample-exact with `ZzFX.js` when the `noise` slot is 0. With noise, the first 1553 samples match and
  the rest is the same noise statistically (the original hashes `sin(i^5)`, whose value past 2^53 depends on
  the JavaScript engine's `pow`).
- `song` import: voices are rendered by `zzfx_voice` with the ZzFX 1.3.2 generator, not ZzFXM's embedded
  variant, so songs sound the same but are not sample-identical. Volume-only cells (`0.A`) are ignored.
- `psg`: one voice of the chip, not three channels sharing one envelope. The DAC table (3 dB per step), the
  noise register's taps and the default clock are not in the datasheet.
