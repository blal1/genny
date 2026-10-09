# Reverbs, rooms, propagation and delay-modulation effects (`genny/reverbs.py`)

Everything here is an effect for an `"fx"` chain (`{"type": "<name>", ...}` or `--fx name:k=v,k=v`), plus one
instrument. All of them take mono or stereo input, run at any sample rate and are deterministic.

Sources: J.O. Smith, *Physical Audio Signal Processing* (PASP) for the Schroeder reverbs, the FDN, delay-line
interpolation, flanger, phaser, chorus, Leslie, Doppler and the electric-guitar feedback chain; Farnell,
*Designing Sound* Table 5.1 for wall absorption; Takala & Hahn, *Sound Rendering* for the microphone pattern
and occlusion; Puckette for the pitch shifter. Numbers the sources do not give are marked `# UNSOURCED` in the
code (Leslie speeds and sizes, room preset geometry, material densities, tape wow/flutter rates).

Wet level convention: every reverb's impulse response is normalised to about unit energy, so `mix` means the
same thing whatever the decay time (a steady sound is as loud wet as dry).

## Catalog

| name | type | what it is | channels out |
|---|---|---|---|
| `jcrev` | fx | Chowning's JCRev (1972) Schroeder reverb; `variant=samson` = Samson-box version | stereo |
| `satrev` | fx | Chowning's SATREV (1971): small, bright, vintage | as input |
| `fdn_reverb` | fx | Jot feedback delay network, 8/16 lines, separate low and high decay times | stereo |
| `room` | fx | physical room: preset or dimensions + materials (early reflections + late reverb) | stereo |
| `zita` | fx | zita-rev1 (the existing `physical/space.py` kernel) | stereo |
| `early_reflections` | fx | first wall/floor/ceiling echoes of a box room, no tail | stereo (or as input) |
| `distance` | fx | 1/r level, air absorption, reverb send | as input (stereo with `reverb`) |
| `doppler` | fx | fly-by with true Doppler shift, swell, air, pan | stereo (mono with `pan=false`) |
| `occlude` | fx | sound from behind a wall / door / window | as input |
| `leslie` | fx | rotary speaker: horn + drum, slow / fast / brake | stereo |
| `flanger` | fx | swept comb with real feedback (replaces the old one) | as input (stereo with `stereo`) |
| `phaser` | fx | 4/6/8-stage first-order allpass phaser (replaces the old one) | as input (stereo with `stereo`) |
| `chorus` | fx | multi-voice chorus on interpolated delays (replaces the old one) | stereo |
| `tape_delay` | fx | tape echo / ping-pong with filtered feedback, wow, flutter, saturation | stereo (as input if `pingpong=false`) |
| `shimmer` | fx | reverb regenerated through an octave-up shifter | stereo |
| `gated_reverb` | fx | big reverb cut off abruptly (80s drums) | stereo |
| `reverse_reverb` | fx | reverb that swells in before the sound | stereo |
| `feedback_guitar` | instrument | overdriven guitar in front of its amp: sustains and blooms into feedback | mono |

The old `flanger`, `phaser` and `chorus` parameter names (`rate`, `depth`, `feedback`, `mix`, `stages`, `voices`)
keep their meaning. `vibrato` (interpolated-delay pitch vibrato) is available as the library function
`genny.reverbs.vibrato`; the registered `vibrato` effect in `fx.py` is unchanged.

## Reverbs

### `jcrev`
Three series allpasses (347, 113, 37 samples, g 0.7) into four parallel feedback combs (1687, 1601, 2053, 2251;
g 0.773, 0.802, 0.753, 0.733) and a mixing matrix: the lengths CCRMA tuned by ear at 25 kHz. They are scaled by
`sr/25000` and moved to the nearest unused prime so they stay mutually prime (44.1 kHz: 613, 199, 67 / 2971,
2819, 3623, 3967). `variant=samson`: allpasses 1051, 337, 113, combs 4799, 4999, 5399, 5801 and two
decorrelating output delays (the figure draws feed-forward combs, JOS's text and STK use feedback combs: feedback
is used).

| param | default | meaning |
|---|---|---|
| `mix` | 0.3 | wet amount 0..1 |
| `variant` | `mus10` | `mus10` or `samson` |
| `t60` | none | decay time in seconds; omit for the published comb gains (about 2 s, samson 4.5 s) |
| `damp` | 0 | extra high-frequency damping in the comb loops 0..0.95 |
| `width` | 1 | stereo width 0..1 |
| `predelay` | 0 | seconds |
| `tail` | auto | seconds of tail appended |

### `satrev`
Four parallel combs (901, 778, 1011, 1123; g 0.805, 0.827, 0.783, 0.764) into three allpasses (125, 42, 12).
Params: `mix` 0.3, `t60` (none = published, about 1.2 s), `damp` 0, `invert_right` false (true = the published
two-output form, right = -left: wide, but cancels in mono), `predelay` 0, `tail` auto.

### `fdn_reverb`
A lossless prototype (N delay lines fed back through an orthogonal matrix: all the energy circulates, nothing is
lost) plus one damping filter per line. The filter is a one-pole low-pass solved so the loop loses exactly
60 dB in `t60_low` seconds at 250 Hz and in `t60_high` seconds at 4 kHz. Line lengths are mutually prime,
spread exponentially around the mean free path `size`.

| param | default | meaning |
|---|---|---|
| `t60_low` | 2.2 | decay time at 250 Hz, seconds |
| `t60_high` | 1.1 | decay time at 4 kHz, seconds (clamped to at most `t60_low`) |
| `size` | 8 | mean free path in metres = 4 x volume / surface: 1 booth, 4 room, 8 hall, 20 cathedral, 50 canyon. Small `size` with a long decay rings like a small hard room |
| `predelay` | 0.02 | seconds before the reverb starts |
| `diffusion` | 0.7 | input allpass diffusion 0..1 (0 = grainy early echoes) |
| `mix` | 0.3 | wet amount |
| `width` | 1 | stereo width |
| `lines` | 16 | 8 or 16 delay lines |
| `matrix` | `householder` | `householder` (N = 16: Jot's nested form) or `hadamard` |
| `tonal` | 1 | tonal correction 0..1: gives back the highs that the faster HF decay takes out of the tail |
| `tail` | auto | seconds appended (1.2 x `t60_low`, max 10) |

### `zita`
`physical/space.py` `zita_rev1` at its native 48 kHz, resampled at the boundary. Params: `mix` 0.3, `preset`
(`default`, `instrument`, `exterior_dry`, `stone_hall`, `cave`), `t60_low` (decay below `xover`, s), `t60_mid`
(mid-band decay, s), `xover` (Hz, 200), `damp` (Hz where the decay time has halved, 6000), `predelay` (s),
`width` 1, `tail` auto.

### `shimmer`, `gated_reverb`, `reverse_reverb`
- `shimmer`: `mix` 0.35, `shimmer` 0.5 (octave-up regeneration), `t60` 3, `size` 10, `tone` 4500 (low-pass on the
  shifted signal, Hz), `tail`. The feedback through the shifter is unrolled into three generations.
- `gated_reverb`: `mix` 0.5, `gate` 0.25 (s the reverb stays open after the sound drops below the threshold),
  `release` 0.03, `threshold` -30 (dB below the input peak), `t60` 2.5, `size` 9, `boost` 6 (dB).
- `reverse_reverb`: `mix` 0.5, `t60` 1.5 (length of the swell; the output is this much longer at the front),
  `size` 8, `dark` 0.6.

## Rooms

### `room`
Direct sound + early reflections from a shoebox image-source model + late reverberation.

- Decay per band: Sabine, `T60 = 0.161 V / (sum of surface x absorption + 4 m V)`, with Farnell's absorption
  table per material and the air term from PASP's air-absorption table.
- Early reflections: every mirror image of the source up to third order, delayed by its extra path, attenuated
  by 1/r and by `sqrt(1 - absorption)` per bounce and band, picked up by two opposite cardioids.
- Late reverb: the FDN with `t60_low` / `t60_high` = the Sabine values at 250 Hz / 4 kHz and `size` = the mean
  free path 4V/S, starting one mean free path after the direct sound.
- Balance: reverberant / direct energy = `(distance / critical distance)^2`, so moving the listener away makes
  the room louder relative to the source. `mix` = 1 is what that listener hears; lower `mix` adds dry signal.

| param | default | meaning |
|---|---|---|
| `preset` | `hall` | `closet` `studio` `bedroom` `bathroom` `hall` `church` `cave` `forest` `street` `canyon` |
| `lx`, `ly`, `lz` | preset | width, length (the direction you look in), height in metres |
| `walls`, `floor`, `ceiling` | preset | `carpet concrete marble wood brick glass plaster fabric metal people water tile rock foliage soil open` |
| `furnish` | none | m2 of soft absorbers. Presets bring their own; a room with explicit dimensions or materials is bare unless this is set |
| `distance` | preset | source to listener, metres |
| `mix` | 0.5 | 0 dry .. 1 the sound at the listening position |
| `model` | `fdn` | late engine: `fdn`, `zita`, `puckette` (Puckette G08, one decay time), `farnell` (Farnell's tight prime-delay room, fixed about 0.5 s) |
| `width` | 1 | stereo width of the reverberation |
| `predelay` | 0 | extra seconds before the late reverb |
| `humidity` | 50 | relative humidity %, 40..70 |
| `tail` | auto | seconds appended (1.2 x longest decay, max 10) |

Preset decay times (Sabine at 500-1000 Hz / measured on the rendered impulse response):
closet 0.18 / 0.19 s, studio 0.43 / 0.38, bedroom 0.54 / 0.42, bathroom 0.93 / 0.78, forest 1.44 / 0.88,
street 1.74 / 1.04, hall 2.11 / 1.61, canyon 5.41 / 2.39, church 6.37 / 4.47, cave 6.91 / 6.12.
(The broadband measurement is shorter because the highs decay faster.)

### `early_reflections`
Only the image-source taps. Params: `mix` 0.5 (1 = direct + reflections at their physical level), `order` 3
(1..4), `stereo` true, `humidity`, and the room params above (`preset`, `lx`, `ly`, `lz`, `walls`, `floor`,
`ceiling`, `distance`, `furnish`).

## Propagation

### `distance`
| param | default | meaning |
|---|---|---|
| `metres` | 10 | distance to the source; the input is the sound at `ref` metres |
| `air` | true | air absorption (5.6 dB/km at 1 kHz, 90 dB/km at 4 kHz at 50 % humidity, f^2 beyond) |
| `humidity` | 50 | %, 40..70; drier air absorbs more highs |
| `reverb` | 0 | reverb send 0..1: the reverberant level stays put while the direct sound drops as 1/r |
| `reverb_time` | 1.5 | seconds |
| `ref` | 1 | metres where gain = 1 (closer than this is capped at +6 dB) |
| `line_source` | false | true = 1/sqrt(r) (road, river) |
| `delay` | false | true = prepend the travel time (r - ref) / 345 s |

### `doppler`
The source moves on a straight line past the listener; each output sample reads the input at its emission time
(solved exactly), so the pitch is `c / (c - v)` approaching and `c / (c + v)` leaving, with c = 345 m/s.
The path is centred on the sound's own duration (`pass_at`).

| param | default | meaning |
|---|---|---|
| `speed` | 20 | m/s (20 = 72 km/h car, 60 racing car, 250 jet; clamped below 0.9 c) |
| `closest` | 5 | closest distance, metres (small = sudden pass, large = slow swell) |
| `pass_at` | 0.5 | when it passes, as a fraction of the duration |
| `direction` | `lr` | `lr` or `rl` |
| `pan` | true | stereo, panned by the source direction (constant power) |
| `air` | true | air absorption that opens as it approaches |
| `humidity` | 50 | % |
| `ref` | `closest` | distance where gain = 1 |

### `occlude`
Transmitted path: mass law, `TL = 20 log10(f x surface density) - 47 dB` (6 dB more loss per octave and per
doubling of thickness). Diffracted path: a low-pass at `edge` scaled by `leak`.
Params: `material` (`concrete brick marble rock tile wood glass plaster metal fabric carpet water soil foliage
people`), `thickness` 0.1 m, `leak` 0.15 (0 = sealed, 1 = a screen you can nearly see past), `edge` 600 Hz.

## Modulation and delay

### `leslie`
Soft-clip overdrive, 800 Hz crossover, then a horn and a drum rotor. The horn is a source on a circle read
through a delay `distance(t) / c` (Doppler vibrato of depth `radius x angular speed / c`, 2.3 % on fast) with
1/r and directivity tremolo and a moving low-pass, plus one cabinet-wall image with the opposite shift; the drum
is tremolo plus a moving low-pass. Two microphones `spread` degrees apart give the stereo motion.

| param | default | meaning |
|---|---|---|
| `speed` | `fast` | `slow` (0.8 Hz), `fast` (6.7 Hz) or `brake` |
| `start` | none | speed at the start; if different the rotors accelerate (horn about 1 s, drum about 4 s) |
| `drive` | 0.3 | amplifier overdrive 0..1 |
| `mic` | 0.6 | microphone distance, metres 0.3..3 (close = deep tremolo) |
| `spread` | 90 | angle between the microphones, degrees |
| `mix` | 1 | wet amount |
| `crossover` | 800 | Hz |

### `flanger`
`y = x + g x[n - M(n)]` with regeneration from the delay output back to its input. Notches sit at odd multiples
of `1 / (2 delay)`, spaced `1 / delay` apart, and move with the LFO.
Params: `rate` 0.3 Hz, `depth` 2 (sweep width ms: delay moves between `delay` and `delay + depth`),
`feedback` 0.5 (-0.95..0.95), `mix` 0.5, `delay` 1 ms, `invert` false (true = notch at DC, thin),
`shape` `sine` | `triangle` | `exp`, `phase` 0, `stereo` 0 (LFO offset between channels, cycles).

### `phaser`
Chain of first-order allpasses with break frequencies `freq x spacing^k` (default 100, 200, 400, 800 Hz), swept
together every sample; `stages` sections give `stages / 2` notches, unevenly spaced.
Params: `rate` 0.5, `depth` 0.7, `stages` 4 | 6 | 8, `mix` 0.5, `feedback` 0 (-0.9..0.9), `freq` 100,
`spacing` 2, `sweep` 3 (octaves at depth 1), `phase` 0, `stereo` 0.

### `chorus`
Params: `rate` 0.8, `depth` 3 ms, `mix` 0.5, `voices` 2 (1..8), `delay` 15 ms, `spacing` 5 ms per extra voice,
`spread` 0.5 (pan spread), `feedback` 0 (0..0.7).

### `tape_delay`
Params: `time` 0.3 s, `feedback` 0.45, `mix` 0.35, `pingpong` true, `tone` 3500 (loop low-pass, Hz), `lowcut` 120
(loop high-pass, Hz), `wow` 0.3, `flutter` 0.2, `drive` 0.3, `tail` auto.

## Instrument: `feedback_guitar`
Span E2-E5, family plucked. A plucked string feeds a pickup (comb at the pickup position), a bass-heavy
pre-gain, a cubic soft clipper `x - x^3/3` with an offset for even harmonics, and a treble-restoring post-gain;
the clipped signal also travels through the air (`distance / 345` s) back into the string. Above a feedback of
about 0.3 the loop gain exceeds the string's losses and the note sustains; the air delay decides which harmonic
takes over.

| param | default | meaning |
|---|---|---|
| `drive` | 6 | pre-distortion gain 1..30 |
| `feedback` | 0.5 | acoustic feedback 0..1 (0 = plain overdriven pluck) |
| `pickup` | 0.18 | pickup position, fraction of the string from the bridge 0.05..0.5 |
| `distance` | 1.2 | amp-to-guitar distance, metres (nudged by under a quarter wavelength so the howl is in phase) |
| `tone` | 2400 | cabinet low-pass, Hz |
| `asym` | 0.15 | clipper asymmetry 0..0.6 |
| `seed` | 0 | pluck noise seed |

## Recipes

```jsonc
// piano in a stone church
{"out": "church_piano.wav",
 "layers": [{"type": "synth", "inst": "piano", "notes": "C4 E4 G4", "dur": 0.8}],
 "fx": [{"type": "room", "preset": "church", "mix": 0.45}]}

// footstep at the far end of a bare concrete corridor (explicit dimensions and materials)
{"out": "corridor_step.wav",
 "layers": [{"type": "sfx", "kind": "footstep"}],
 "fx": [{"type": "room", "lx": 3, "ly": 12, "lz": 2.6, "walls": "concrete", "floor": "concrete",
         "ceiling": "concrete", "distance": 8, "mix": 0.8}]}

// car passing at 126 km/h, 6 m away
{"out": "car_pass.wav", "trim": false,
 "layers": [{"type": "sfx", "kind": "car_engine", "params": {"rpm": 4200, "load": 0.8, "dur": 4.0}}],
 "fx": [{"type": "doppler", "speed": 35, "closest": 6}, {"type": "distance", "metres": 3, "reverb": 0.15}]}

// organ through a Leslie spinning up from slow to fast
{"out": "organ_leslie.wav",
 "layers": [{"type": "synth", "inst": "organ", "notes": "C3:maj", "dur": 3.0}],
 "fx": [{"type": "leslie", "speed": "fast", "start": "slow", "drive": 0.4}]}

// sustained feedback guitar riff with tape echo
{"out": "feedback_riff.wav",
 "layers": [{"type": "seq", "inst": "feedback_guitar", "steps": "E2:2 G2:1 A2:3", "step": 1,
             "params": {"feedback": 0.8, "drive": 10}}],
 "fx": [{"type": "tape_delay", "time": 0.35, "feedback": 0.4, "mix": 0.25},
        {"type": "fdn_reverb", "t60_low": 1.8, "t60_high": 0.9, "size": 6, "mix": 0.2}]}
```

More one-liners (all rendered while writing this page):
- voice behind a door: `[{"type": "occlude", "material": "wood", "thickness": 0.04, "leak": 0.1}, {"type": "room", "preset": "bedroom", "mix": 0.4}]`
- distant blast in a canyon: `[{"type": "distance", "metres": 400}, {"type": "room", "preset": "canyon", "mix": 0.7}]`
- 80s snare: `[{"type": "gated_reverb", "gate": 0.3, "mix": 0.6}]`
- jet whoosh on noise: `[{"type": "flanger", "rate": 0.2, "depth": 6, "feedback": 0.8, "shape": "exp"}]`
- shimmering pad: `[{"type": "phaser", "stages": 6, "rate": 0.25, "feedback": 0.6, "stereo": 0.25}, {"type": "shimmer", "mix": 0.4}]`
- ghost bell: `[{"type": "reverse_reverb", "t60": 1.2}, {"type": "jcrev", "mix": 0.2}]`

## What the main parameters mean physically
- **T60** (`t60`, `t60_low`, `t60_high`): seconds for the reverberation to fall by 60 dB. Rooms always lose highs
  faster (air and soft surfaces), so `t60_high` < `t60_low`; equal values sound metallic and artificial.
- **Mean free path** (`size`): average distance sound travels between two wall bounces, 4V/S. It sets how far
  apart the echoes are, i.e. how big the room sounds, independently of how long it rings.
- **Absorption / material**: fraction of the energy a surface does not return, per frequency band. Carpet and
  fabric absorb highs, wood panels and glass absorb lows, marble, tile and concrete absorb almost nothing.
- **Critical distance**: where direct and reverberant sound are equally loud; `room`'s `distance` is measured
  against it (closet 0.25 m, hall 3 m, canyon 24 m).
- **Doppler**: the pitch ratio depends only on the speed along the line to the listener, so `closest` sets how
  abruptly the pitch drops at the pass, `speed` how far it drops.
- **Mass law**: a wall twice as heavy (or a frequency twice as high) gets through 6 dB weaker; that is why only
  the bass of the neighbours' music arrives.
