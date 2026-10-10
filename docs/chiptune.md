# Sound chips (`genny/chiptune.py`)

Each chip is modelled on its own, from its hardware documentation, not as one generic "8-bit" oscillator:
the NES 2A03, the Game Boy and the Yamaha YM2612.

| type | name | what it is |
|---|---|---|
| layer type | `chip` | A whole chip, hardware-accurate: one voice per channel, the chip's mixer and output filters |
| instrument | `nes_pulse` | NES 2A03 pulse channel: four duties, 4-bit volume, hardware envelope and sweep (C2-C7) |
| instrument | `nes_triangle` | NES 2A03 triangle: 32-step 4-bit staircase, no volume control (C1-C6) |
| instrument | `nes_noise` | NES 2A03 noise: 15-bit LFSR at 16 fixed rates, long and short modes (C3-C8) |
| instrument | `gb_pulse` | Game Boy pulse channel: four duties, hardware volume envelope, frequency sweep (C2-C7) |
| instrument | `gb_wave` | Game Boy wave channel: 32 four-bit samples of wave RAM (C1-C6) |
| instrument | `gb_noise` | Game Boy noise: 15-bit or 7-bit LFSR, hardware envelope (C3-C8) |
| instrument | `ym2612` | Yamaha YM2612 (Mega Drive) FM channel: four operators, eight algorithms (C1-C7) |

The sfxr and ZzFX effect generators and the AY-3-8910 voice are in [`retro.md`](retro.md).

## Two modes

| mode | how | what you get |
|---|---|---|
| `hardware_accurate` | the `chip` layer | One note at a time per channel. Pitches are the chip's timer values (high notes are a few cents off, as on the machine). The channels meet in the chip's own mixer and leave through its filters. |
| `retro_stylized` | the instruments, in any `seq`, `song` or `compose` layer | One channel on its own, as many notes at once as you write, played back in tune and rounded like the other retro voices. `"mode": "hardware_accurate"` in `params` gives the raw channel instead. |

## Layer `chip`

```json
{"type": "chip", "system": "nes", "channels": {
  "pulse1":   {"steps": "D5:1 F#5:.5 A5:1 G5:.5 F#5:1", "duty": 1, "vol": [15, 13, 11, 10, 9, 8]},
  "pulse2":   {"steps": "[D4,F#4,A4]:1 -:1 [D4,F#4,A4]:1 -:1", "vol": [8, 7, 6, 5, 5, 4]},
  "triangle": {"steps": "D3:1 A2:1 D3:1 A2:1", "gate": 0.8},
  "noise":   [{"steps": "C8 C8 C8 C8 C8 C8 C8 C8", "step": 0.5, "vol": [8, 5, 3, 1, 0]},
              {"steps": "- C6 - C6", "step": 1, "decay": 1}]}}
```

| key | default | meaning |
|---|---|---|
| `system` | `nes` | `nes` (channels `pulse1`, `pulse2`, `triangle`, `noise`), `gb` (`pulse1`, `pulse2`, `wave`, `noise`) or `ym2612` (`fm1` .. `fm6`, see below) |
| `channels` | - | channel name -> a lane, or a list of lanes sharing the channel (the latest note cuts the one before, as when a driver shares the noise channel between its drums) |
| `region` | `ntsc` | NES: `ntsc` (CPU 1789773 Hz) or `pal` (1662607 Hz); changes the pitch grid, the noise rates and the envelope clock |
| `filters` | `nes` | NES output: `nes` (high-pass 90 Hz, high-pass 440 Hz, low-pass 14 kHz), `famicom` (high-pass 37 Hz), `none` |
| `dmc` | 0 | NES: the level 0..127 the DMC channel rests at. A high level makes the triangle and the noise quieter (the mixer is not linear) |
| `wave` | `triangle` | Game Boy wave RAM: 32 values 0..15, 32 hex digits, or `triangle`, `saw`, `square`, `pulse25`, `sine`, `organ`, `bass` |
| `model` | `dmg` | Game Boy output high-pass: `dmg` or `cgb` (stronger) |
| `mode` | `hardware_accurate` | `retro_stylized` drops the NES filters and the Game Boy panning |

A lane is `steps` (the usual step notation; `step`, `vel`, `transpose` as in a `seq` layer) plus a patch. A chord in
`steps` is played as a 60 Hz arpeggio, the way one channel plays a chord. A Game Boy lane may carry
`"pan": "L" | "R"` (the console pans each channel hard left, hard right or centre).

### Patch

| key | channels | meaning |
|---|---|---|
| `volume` | pulse, noise | 0..15 when no envelope runs (default 15) |
| `duty` | pulse | 0..3 = 12.5, 25, 50, 75 % (default 2), or a table of them |
| `decay`, `loop` | NES pulse, noise | hardware envelope: the volume falls 15 -> 0, one step every (decay + 1)/240 s; `loop` starts it again |
| `env` | Game Boy pulse, noise | hardware envelope: -1..-7 falls one step every n/64 s from `volume`, 1..7 rises |
| `sweep` | pulse | NES `[period 0..7, shift 1..7, negate 0 or 1]`; Game Boy pulse 1 only `[pace 1..7, step 1..7, 1 up or -1 down]`. A sweep that leaves the range silences the channel, as on the chip |
| `linear` | NES triangle | the note stops after this many 1/240 s (a short triangle thump) |
| `level` | Game Boy wave | 1 = 100 %, 2 = 50 %, 3 = 25 % |
| `period`, `short` | noise | NES period index 0..15, Game Boy `[shift, divider]`; -1 takes it from the note (high note = hiss, low note = rumble). `short` is the buzzing pitched mode |
| `gate` | all | fraction of the step the note sounds for |
| `vol` | pulse, noise | driver table: volumes 0..15, one per 1/60 s, the last one held |
| `arp` | all | driver table: semitone offsets, one per 1/60 s, cycling |
| `pitch` | all | driver table: semitone offsets, one per 1/60 s, the last one held (a drum: `[12, 7, 3, 0]`) |
| `vib`, `vib_hz`, `vib_delay` | pulse, triangle, wave | vibrato depth in semitones, rate, delay; it moves the timer once per 1/60 s |

The driver tables are what a game's music routine does on top of the chip, once per video frame; they are genny's
own convention, not hardware.

## What is modelled

NES 2A03 (NESdev wiki): pulse period `fCPU / (16 (t + 1))` with the 8-step duty sequences and silence below
`t = 8`; triangle `fCPU / (32 (t + 1))` over its 32-step sequence, holding its last value when it stops; noise with
its 16 periods and both feedback taps (bit 1, or bit 6 for the 93-step mode); the envelope and sweep units clocked by
the frame counter's quarter and half frames, pulse 1 sweeping with the ones' complement; the mixer's
`pulse_out` and `tnd_out` formula; the output filters.

Game Boy (Pan Docs): pulse `131072 / (2048 - period)` Hz and wave `65536 / (2048 - period)` Hz; the duty table;
wave RAM read at index 1 first after a trigger, the level shifting the digital value; noise clocked at
`262144 / (divider x 2^shift)` Hz with the XNOR feedback and the 7-bit mode; volume envelope at 64 Hz and
frequency sweep at 128 Hz with its overflow check; DACs mapping 0..15 to +1..-1; hard panning; the output
capacitor's high-pass.

## Yamaha YM2612

```json
{"type": "chip", "system": "ym2612", "lfo": 3, "channels": {
  "fm1": {"steps": "E4:0.5 G4:0.5 C5:1", "patch": "lead", "pan": "L"},
  "fm2": {"steps": "C3:1 G2:1", "patch": "bass"},
  "fm3": {"steps": "C4:2", "patch": {"alg": 4, "fb": 3, "ops": [
      {"tl": 30, "mul": 2, "dr": 10, "sl": 3}, {"dr": 6, "sl": 2},
      {"tl": 40, "mul": 1}, {"dt": 2}]}}}}
```

Six channels `fm1` .. `fm6`, one note at a time each (spread a chord over several channels). A lane takes `steps`,
`step`, `vel`, `transpose`, `gate` (default 0.95), `pan` (`L`, `R` or centre) and `patch`. Layer keys: `lfo`
(rate 0 slowest .. 7 fastest, absent = off), `tail` (seconds after the last note, default 0.5), `clock`
(default 7670453 Hz; the chip runs at clock / 144 = 53267 Hz) and `mode`.

A patch is a name (`bass`, `bell`, `brass`, `clav`, `epiano`, `lead`, `organ`, `pluck`, `sine`, `strings`; they are
genny's own starting points) or the chip's own parameters:

| key | range | meaning |
|---|---|---|
| `alg` | 0..7 | how the four operators are wired. 0: 1>2>3>4. 1: (1+2)>3>4. 2: (1+(2>3))>4. 3: ((1>2)+3)>4. 4: (1>2)+(3>4). 5: 1>(2, 3, 4). 6: (1>2)+3+4. 7: 1+2+3+4 |
| `fb` | 0..7 | operator 1 feeding itself: 0 a sine, 7 close to noise |
| `pms`, `ams` | 0..7, 0..3 | how far the LFO bends the pitch, and how deep it dips operators that have `am: 1` |
| `ops` | 4 | the operators, each `{ar, dr, sr, rr, sl, tl, ks, mul, dt, am}` |

| operator key | range | meaning |
|---|---|---|
| `ar` | 0..31 | attack rate (31 = at once) |
| `dr` | 0..31 | decay rate down to the sustain level (0 = no decay) |
| `sr` | 0..31 | second decay, while the key is held (0 = holds) |
| `rr` | 0..15 | release rate |
| `sl` | 0..15 | sustain level, 3 dB a step below full (15 = silence) |
| `tl` | 0..127 | total level, 0.75 dB a step of attenuation. On a modulator this is the brightness |
| `ks` | 0..3 | key scaling: higher notes run their envelope faster |
| `mul` | 0..15 | frequency multiple (0 = half) |
| `dt` | 0..7 | detune: 1..3 sharp, 5..7 flat |
| `am` | 0, 1 | the LFO's tremolo reaches this operator |

Velocity is added to the total level of the operators that reach the output, as a sound driver does.
`hardware_accurate` (the layer's default) keeps each channel's 9-bit output and the step the YM2612's DAC makes
across zero; `retro_stylized` (the instrument's default) sums the operators at full resolution.

The FM core is a port of ymfm (Aaron Giles, BSD 3-Clause): log-sine and power tables, envelope generator with its
increment table and key scaling, phase generator with detune, the eight algorithms, the LFO. Not ported: SSG-EG
envelopes, channel 3's multi-frequency mode, CSM, channel 6's PCM DAC.

Not modelled: the NES DMC sample channel (only its resting level), length counters, the Game Boy's power-on and
"zombie" quirks.

Examples: `examples/chiptune/nes_overworld.json`, `examples/chiptune/gb_cave.json`, `examples/chiptune/md_stage.json`.
