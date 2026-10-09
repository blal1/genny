# Acoustic tubes (`genny/tubes.py`)

Air columns solved as physics: one finite-difference kernel for Webster's horn equation (any bore shape,
optionally changing in time), a passive radiating end, wall and boundary-layer loss, side holes that open
continuously, and a real reed (mass, spring, damper, collision with the mouthpiece, Bernoulli flow). Sources:
Bilbao, *Numerical Sound Synthesis* ch. 9 and listing A.10; Smith, *PASP* (Keefe wall loss, cones); STK
BlowBotl; the Baschet piston pipes.

Everything runs at 44.1 kHz internally and is resampled to the output rate, so pitch and reed behaviour are
the same at 22.05, 44.1 and 48 kHz.

## Instruments

| name | what it is | span |
|---|---|---|
| `reed_tube` | clarinet-like cylinder + beating reed: hollow, odd harmonics | D3-G5 |
| `reed_cone` | sax/oboe-like conical bore + the same reed: even harmonics appear | D3-C6 |
| `bottle` | blown bottle / jug (STK BlowBotl): hooty, breath noise that pulses with the tone | C3-C6 |
| `vowel_tube` | sung vowel: glottal pulses through a vocal tract, gliding between two vowels | E2-C5 |
| `piston_pipe` | Baschet "windless organ pipe": a piston drives a quarter-wave pipe, deep and slow to speak | C2-C4 |
| `pvc_pipe` | slapped plastic tube (thongophone / boomwhacker-like) | C2-C5 |

### `reed_tube`, `reed_cone`

The tube length is solved per note from the scheme's own resonance, including the open-end correction and the
reed (whose motion is worth 9.6 mm of extra tube); the remaining few cents are measured and compensated.

| param | default | meaning |
|---|---|---|
| `cone` (`reed_cone` only) | 1.0 | 0..1 taper. 0 = cylinder (odd harmonics), 1 = end radius twice the reed end. Steeper cones jump to the octave, as on a real, badly voiced instrument, so the range stops there |
| `pressure` | 1.0 | mouth pressure scale, 0.7..1.25. Below about 0.65 the reed does not speak (the oscillation threshold is real). Higher = the reed beats harder against the mouthpiece: louder, brighter |
| `embouchure` | 1.0 | reed opening, 0.7..1.5. Tight lips (< 1) soften the tone and sharpen it a little; loose lips are louder and a little flat |
| `breath` | 0.04 | 0..0.5 turbulence on the mouth pressure (enters after the attack) |
| `vibrato` | 0.0 | 0..1 breath-pressure vibrato, 5 Hz |
| `attack` | 0.005 | s, pressure rise time |
| `tongue` | 0.5 | 0..1 tongued start (short pressure overshoot). 0 = breath attack: low notes then take 100 ms or more to speak |
| `register` | 0.0 | 0..1 register hole. Open, the cylinder overblows a twelfth (3x), like a clarinet's speaker key |
| `hole` | 0.0 | 0..1 tonehole at 80 % of the bore: bends the note up (about a semitone at 0.5); fully open the cylinder may flip register. 0 keeps the note in tune |
| `bright` | 1.0 | 0.25..4 output lowpass scale |
| `seed` | 0 | noise seed |

`vel` is mouth pressure too (plus level): soft notes are rounder, loud ones buzz. Pitch stays within about
2 cents from vel 0.3 to 1.0.

### `bottle`

| param | default | meaning |
|---|---|---|
| `noise` | 0.12 | 0..1.5 breath noise. 0.5 is the STK default (very airy; the pitch then wanders 10 cents or more with the turbulence), 0.75 and up = creature breath |
| `vibrato` | 0.0 | 0..1 breath vibrato |
| `bright` | 1.0 | 0.25..4 output lowpass scale |
| `seed` | 0 | noise seed |

Works far below its span (30-120 Hz = a huge vessel).

### `vowel_tube`

| param | default | meaning |
|---|---|---|
| `vowel` | "a" | a, e, i, o, u: tract shape at the start |
| `vowel2` | "" | vowel to glide to; empty = hold the first |
| `morph` | 1.0 | 0..1 how far the tract travels toward `vowel2` during the note |
| `breath` | 0.08 | 0..1 aspiration noise at the glottis |
| `vibrato` | 0.4 | 0..2 pitch vibrato (1 = +-3 %, 5 Hz, comes in after 0.3 s) |
| `size` | 1.0 | tract length scale: 1 = 17 cm adult, 0.8 small and bright, 1.4 giant. Formants scale as 1/size, pitch does not move |
| `tense` | 0.2 | 0..0.8 glottal closed fraction: 0 soft, 0.6 pressed and buzzy |
| `seed` | 0 | noise seed |

Formants (F1/F2, Hz): a 649/1085, e 406/1912, i 245/2109, o 460/987, u 277/793. /a/ and /e/ are Bilbao's
tables (Fant); /i/, /o/, /u/ are this module's own shapes.

### `piston_pipe`

| param | default | meaning |
|---|---|---|
| `diameter` | 0.2 | pipe diameter, 0.12..0.36 m. Wider = speaks faster, rings shorter |
| `rod` | 0.35 | 0..1 harmonic content of the driving rod (the pipe passes the odd harmonics) |
| `detune` | 0.0 | cents the pipe is off the driven note. The pipe only amplifies when they match: 500 cents off is about 30 dB weaker |

### `pvc_pipe`

| param | default | meaning |
|---|---|---|
| `radius` | 0 | pipe radius in m (0.004..0.12); 0 = length/16, the same proportions on every note. Narrow = longer, purer ring |
| `hardness` | 0.5 | 0 soft palm (thud) .. 1 hard paddle (click, more overtones) |
| `loss` | 1.5 | wall loss 0.5..10: 1 = smooth rigid pipe, higher = shorter ring |
| `seed` | 0 | slap noise seed |

`dur` is how long the paddle stays on the end; after it the column is let go.

## Sound effect

`pipe_blow`: breath across the mouth of a pipe; noise coloured by the pipe's resonances, never a steady tone.

| param | default | meaning |
|---|---|---|
| `length_m` | 0.3 | pipe length, 0.03..5 m |
| `radius_m` | 0.012 | pipe radius |
| `stopped` | true | far end closed (panpipe, bottle neck: odd quarter-wave series) or open (half-wave series) |
| `dur` | 1.2 | s |
| `breath` | 0.7 | 0..1 gustiness |
| `seed` | 0 | noise seed |

## Effects

`tube`: the sound enters one end of a duct and is heard at the other.

| param | default | meaning |
|---|---|---|
| `length_m` | 1.0 | 0.03..30 m. Resonances at c/4L x 1, 3, 5.. (closed-open) or c/2L x 1, 2, 3.. (open-open); also the arrival delay. A 30 m duct renders in about 0.3 s per second of audio |
| `radius_m` | 0.05 | 0.002..1 m. Narrow = more wall loss, sharper resonances. One-dimensional model: right below roughly 100/radius Hz |
| `ends` | "closed-open" | `closed-open` (source at a closed end: well, bottle, stopped pipe), `open-open` (vent, culvert, tunnel), `closed-closed` (heard inside a sealed duct or tank) |
| `loss` | 1.0 | 0..20: 1 = smooth rigid wall, 3-10 = rough, lined or leaky |
| `mix` | 1.0 | wet/dry |

`formant_tract`: puts a vowel's vocal-tract resonances on any input.

| param | default | meaning |
|---|---|---|
| `vowel` | "a" | a, e, i, o, u |
| `shift` | 1.0 | 0.5..2 formant scale (> 1 = shorter tract, higher formants) |
| `mix` | 1.0 | wet/dry |

Both keep the input's length and level (the wet signal is matched to the dry RMS) and accept mono or stereo.

## Recipes

Clarinet phrase:
```json
{"bpm": 96, "layers": [{"type": "seq", "inst": "reed_tube", "steps": "D3:1 F3:.5 A3:.5 D4:1 C4:.5 A3:.5 F3:2",
  "params": {"breath": 0.08, "vibrato": 0.3}}], "fx": [{"type": "reverb", "mix": 0.18, "size": 0.6}]}
```

Pushed sax note, then a clarinet overblown through its register hole:
```json
{"layers": [
  {"type": "synth", "inst": "reed_cone", "notes": "G3", "dur": 1.5, "vel": 1.0,
   "params": {"pressure": 1.2, "embouchure": 1.3, "breath": 0.15, "bright": 1.5}},
  {"type": "synth", "inst": "reed_tube", "notes": "F3", "dur": 1.0, "at": 1.7, "params": {"register": 1.0}}]}
```

Two voices gliding between vowels:
```json
{"layers": [
  {"type": "synth", "inst": "vowel_tube", "notes": "A2", "dur": 2.5,
   "params": {"vowel": "a", "vowel2": "i", "vibrato": 0.8, "breath": 0.15}},
  {"type": "synth", "inst": "vowel_tube", "notes": "E3", "dur": 2.5, "gain": 0.7,
   "params": {"vowel": "o", "vowel2": "u", "size": 0.85}}]}
```

Jug band: slapped pipes, bottle, a piston pipe drone:
```json
{"bpm": 110, "layers": [
  {"type": "seq", "inst": "pvc_pipe", "steps": "C2:.5 C2:.5 G2:.5 C3:.5 Bb2:.5 G2:.5 C2:1", "params": {"hardness": 0.7}},
  {"type": "seq", "inst": "bottle", "steps": "C4:1 Eb4:1 G4:2", "gain": 0.6, "params": {"noise": 0.3}},
  {"type": "synth", "inst": "piston_pipe", "notes": "C2", "dur": 4, "gain": 0.8, "params": {"diameter": 0.3}}]}
```

A voice down a ventilation duct, with air moving in it; and a sawtooth made to say "oo":
```json
{"layers": [
  {"type": "speech", "text": "is anybody down there",
   "fx": [{"type": "tube", "length_m": 6.0, "radius_m": 0.15, "ends": "open-open", "loss": 3.0, "mix": 0.85}]},
  {"type": "sfx", "kind": "pipe_blow", "at": 0.2, "gain": 0.3,
   "params": {"length_m": 2.0, "radius_m": 0.05, "stopped": false, "dur": 2.5}}]}
```
```json
{"layers": [{"type": "synth", "inst": "saw", "notes": "A1", "dur": 2.0,
  "fx": [{"type": "formant_tract", "vowel": "u", "shift": 0.8}]}]}
```

## For code: `WebsterTube`

```python
from genny.tubes import WebsterTube, reed_params
t = WebsterTube(0.664, area=1.0, radius_m=0.0074, left="reed", right="unflanged", loss=1.0,
                holes=[(0.8, 0.003, 0.005)])            # (position 0..1, hole radius m, chimney height m)
out = t.run(pm=mouth_pressure, reed=reed_params(), hole_state=[openness])   # p_out, u_out, p_in, y, p_pick
t.resonances(2000)                                       # impedance peaks of the scheme itself, Hz
```

`area` is a function of x in [0, 1], a list of [x, S] pairs, samples, or a number; `area2` plus
`run(morph=...)` moves the bore in time. Ends: `closed` (driven by `u_in`), `open` (ideal), `flanged`,
`unflanged`, `reed`. `loss` = boundary-layer loss (1 = smooth pipe), `wall` = yielding vocal-tract walls.
Stable for any bore. `python -m genny.tubes` re-measures the reed tuning and level tables.

Mouth pressure is in units of rho c^2: the reed starts to sound at 0.012, beats against the mouthpiece above
about 0.017, and is held shut at 0.035.
