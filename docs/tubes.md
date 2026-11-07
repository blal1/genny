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
| `mesh_vowel` | sung vowel through a 2-D waveguide mesh of the measured tract (as wide as the tract: the cross modes add upper formant detail); `vowel` a e i o u, `vibrato` 0.4 | C2-C5 |
| `box_bar` | metal bar over a stopped box resonator: in tune the box takes the bar's energy (loud, short); `detune` (20 cents between box and bar) lets the bar ring on, quieter; `decay` 4 s | C3-C6 |

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
| `reed` | `default` | reed preset: `default` (Bilbao), `played` (clarinet reed measured in play), `sax` (tenor sax reed + lip) |
| `contact` | 0.0 | s, Hunt-Crossley damping of the reed on the lay (0 = elastic contact) |
| `mouthpiece` | 0.0 | 0..1 share of the saxophone mouthpiece equivalent volume added at the reed, pitch kept |
| `seed` | 0 | noise seed |

`reed_params(contact_damping=)` returns the nine non-dimensional reed constants.

`vel` is mouth pressure too (plus level): soft notes are rounder, loud ones buzz. Pitch stays within about
2 cents from vel 0.3 to 1.0.

Since 1.2.1 a slow `attack` no longer flips the cylinder to the twelfth: the fundamental stays (third harmonic
within 6 dB of the first). The default attack sounds as before.

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
| `vowel` | "a" | i ih eh ae ah aa ao uh ow uw (MRI area functions; a e o u = aa eh ow uw): tract shape at the start |
| `vowel2` | "" | vowel to glide to, along the tract's modes; empty = hold the first |
| `morph` | 1.0 | 0..1 how far the tract travels toward `vowel2` during the note |
| `consonant` | "" | p t k m n ng f th s sh (b d g v z...: their place): the note starts from that constriction and releases into the vowel. The constriction is one event (`tubes.event`), held 50 ms and released over 60 ms by a raised cosine; voiceless ones start voicing 40 ms after the release, with aspiration in the gap |
| `source` | bilbao | glottal flow: `bilbao` (squared-sine pulse shaped by `tense`) or `rosenberg` (Rosenberg C pulse: opening 40 % of the period, closing 16 %; `tense` has no effect) |
| `glide` | smooth | path to `vowel2`: `smooth` (cubic) or `minjerk` (minimum-jerk, the speech-motor profile) |
| `model` | webster | `webster`, or `kl`: Kelly-Lochbaum tract (`story.kl_voice`) with subglottal system and nasal branch |
| `nasal` | 0 | velum opening, cm² (0..2); above 0 uses `model` kl |
| `breath` | 0.08 | 0..1 aspiration noise at the glottis |
| `vibrato` | 0.4 | 0..2 pitch vibrato (1 = +-3 %, 5 Hz, comes in after 0.3 s) |
| `size` | 1.0 | tract length scale: 1 = 17 cm adult, 0.8 small and bright, 1.4 giant. Formants scale as 1/size, pitch does not move |
| `tense` | 0.2 | 0..0.8 glottal closed fraction: 0 soft, 0.6 pressed and buzzy |
| `seed` | 0 | noise seed |

The vowels are Story's (2005) MRI area functions; a glide between two vowels follows his modes.
`tubes.event(t, n01, npk, n02, mu)`: a constriction event, 0 before n01, rising by a raised cosine to mu at npk and
back to 0 at n02; `tubes.events(t, [(n01, npk, n02, mu), ...])` sums several, held to 0..1.

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

`wah_pipe`: a long pipe blown by a fan through a rubber flap that swells. The pressure ramp brings in the
fundamental, then the partials one by one (partial n speaks once the pressure passes n / N of full scale); a
half-stopped end beats its open and stopped series. `length_m` 6.1 (0.5..12), `diameter` 0.254 m, `dur` 4 s, `ramp`
2.5 s (time for the flap pressure to reach full), `pressure` 1 (0..1: how many partials sound), `stop` 0 (0 open ..
1 stopped), `seed`.

## Effects

`tube`: the sound enters one end of a duct and is heard at the other.

| param | default | meaning |
|---|---|---|
| `length_m` | 1.0 | 0.03..30 m. Resonances at c/4L x 1, 3, 5.. (closed-open) or c/2L x 1, 2, 3.. (open-open); also the arrival delay. A 30 m duct renders in about 0.3 s per second of audio |
| `radius_m` | 0.05 | 0.002..1 m. Narrow = more wall loss, sharper resonances. One-dimensional model: right below roughly 100/radius Hz |
| `ends` | "closed-open" | `closed-open` (source at a closed end: well, bottle, stopped pipe), `open-open` (vent, culvert, tunnel), `closed-closed` (heard inside a sealed duct or tank) |
| `loss` | 1.0 | 0..20: 1 = smooth rigid wall, 3-10 = rough, lined or leaky |
| `mix` | 1.0 | wet/dry |
| `steepen` | 0 | kPa peak pressure in the duct: loud pulses steepen into shocks along the tube (Burgers characteristics, `steepen_through`, `cyber.steepening`) |
| `silencer` | 0 | ms: absorptive silencer at the outlet, Polack's diffusive response peaking at this time, D² = 1.5 × the peak (`silence_through`, `cyber.polack_ir`; 1-10) |

### `tube_mesh` — heard down a long square duct (3-D)

A 3-D wave mesh of a square duct (default 9.5 x 0.45 x 0.45 m, 74 mm cells at 8 kHz). The source sits at one end
on the axis; both ends absorb completely, the four side walls absorb `absorption` of the energy at each reflection.
The output is stereo: the right ear on the axis, the left ear 15 cm off it, at `distance` from the source.
The level is referred to the closest point (0.42 m), so moving away lowers the level less than open air does
(less than 6 dB per doubling) while the reverberant part grows: a distance cue that also works for unfamiliar sounds.
Listeners judged distance in such a tube as a compressed power of the real one (exponent 0.61 on headphones,
0.53 on loudspeakers) and overestimate close sources.

Params: `distance` (m, 1.68), `length_m` (9.5), `width_m` (0.45), `absorption` (0.2), `tail` (s, 0.3), `mix` (1).
Unlike `tube` and `formant_tract`, it appends `tail` seconds of ring-out.

    {"type": "tube_mesh", "distance": 4.75}

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

Helpers: `radiation_impedance(ka, model)` (normalised pipe-end radiation impedance, rational fits `rayleigh1`,
`rayleigh2` (flanged) or `levine2`), `end_reflection(ka, end)` (pressure reflection of a pipe end),
`end_reflection_filter(radius_m, sr, end)` (its biquad for a waveguide termination), `box_length(freq)` (stopped
resonator length L = c/(4f) − 0.6 r) and `bar_resonator` (struck bar and resonator as two coupled oscillators).

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
`end_wall` (outer / inner radius of an unflanged end, >= 1) sets that end's mass correction; `causse_zr(ka)` is the
radiation impedance of that end and `bessel_bore(x0, nu)` a Bessel-horn area; no instrument uses them yet.
`WebsterTube(hole_loss=True)` adds the series resistance of each open hole (viscous and radiation loss,
`hole_rate(b, l_eff, f)`: r = (2/b)√(πfη/ρ) + c(kb)²/(4 l_eff)); `run()` then also returns `u_holes`.
`hole_kind` sets an open side hole's length: `lumped` (default, one external correction b (1.4 − 0.58 (b/a)²)),
`unflanged` or `flanged` (chimney + matching volume + inner correction + radiation 0.6133 b or 0.8216 b).
`hole_length_corrections(b, a, kind)` returns those terms (t_m, t_i, t_r, sum) in m.
`jet_receptivity(f, l, U0, R)` gives a flue jet's gain exp(0.4 ω l/U0) and phase π/2 + ω l/(0.5 U0) from the flue
to the edge, plus a flag where f R/U0 >= 1 (a rounded edge).

Mouth pressure is in units of rho c^2: the reed starts to sound at 0.012, beats against the mouthpiece above
about 0.017, and is held shut at 0.035.
