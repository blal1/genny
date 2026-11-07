# Friction and the Baschet family (`genny/friction.py`)

Sounds made by rubbing (stick-slip), and the sound-sculpture instruments of the Baschet brothers: a
vibrating rod, a heavy collector that mixes the rods, and a radiator (folded steel sheet, cone or balloon)
that colours what reaches the air.

Three friction solvers share one modal bank; all run at 48 kHz internally and are resampled to the
requested rate.

| solver (Python) | model | source |
|---|---|---|
| `friction_modal_resonator` | hyperbolic friction law, exact modal recurrence, stick/slip state machine | Couineaux, Ablitzer, Gautier, "Minimal physical model of the cristal Baschet", Acta Acustica 2023 |
| `elastoplastic_friction` | elasto-plastic bristle (LuGre-family) friction, K method + Newton | Rocchesso & Fontana, *The Sounding Object*, §8.3.2 |
| `bowed_modal` | soft bow characteristic, implicit solve with uniqueness bound | Bilbao, *Numerical Sound Synthesis*, §4.3.1, §7.4, Problem 7.17 |

Helpers: `fn_min` (minimum normal force), `rise_time`, `rod_frequency` / `rod_length` (clamped rod,
f ~ 1/L², length x 0.7071 per octave), `beam_shape`, `struck_rod`, `dax_free_length`.
`thoret_friction(f0, dur, sr, eps=1, wander=0.1, cutoff=10, seed=0, diameter=None, speed=None)`: additive friction
source, sum of sin(k phi)/k on a fundamental that wanders (`eps` 1 squeaks and creaks, 0 a steady tone); with
`diameter` and `speed` it beats at v/D, like a singing glass.
More helpers: `wineglass_f0` (lowest (2, 0) mode of a wine glass), `akay_friction` (friction coefficient smooth
through v = 0), `stick_slip_number` (G = (F0 / k) / (V0 / ω0): the larger, the longer the sticking phase),
`coulomb_envelope` (a mode damped by dry friction loses 4 μN / k per cycle: a straight-line decay),
`pulley_rod_freq` (rod on two counter-rotating pulleys, Ω = √(μ g / a)), `collector_loss` (T60 factor of a rod
in a collector lighter than 3 times its mass), `struck_rod(sustain=)`; `friction_modal_resonator(pitch=)` and
`bowed_modal(pitch=)` take a per-sample factor on every mode frequency (a bend while the voice sings).

Baschet sculptures have five elements: energy source, oscillator, modifier, radiator, support. Solids are 1e3 to
1e5 times stiffer acoustically than air, so a radiator must carry the vibration out.

## Instruments

Self-oscillating voices are tuned automatically: friction pulls the pitch away from the mode (a heavier
bow or finger flattens it), so a short pilot run measures the sounding pitch and the modes are rescaled.

### `cristal` — Cristal Baschet (C2-C6)
A wet finger rubs a glass rod; the rod drives a clamped steel shaft by stick-slip. Near-sine, no attack,
sustained as long as the finger moves.

| param | default | meaning |
|---|---|---|
| `pressure` | 1.0 | finger normal force in newtons (0-6). Below about 0.1 N the rod stays silent; more force speaks faster |
| `finger_speed` | 0.15 | finger speed along the rod, m/s (0.02-0.5). The rod's velocity follows it, so this is the loudness |
| `radiator` | `sheet` | `sheet` (ringing, saturates when loud), `cone` (dry, clean), `balloon` (dry, mid-heavy) |
| `bright` | 1.0 | 0.5 mellow .. 2 bright: friction harmonics and lowpass |
| `wetness` | 1.0 | 1 = well-wetted finger (the paper's friction law); lower = more slippery, slower to speak |
| `bend` | 0 | semitones (−2..2) the note glides to over its duration while the rod keeps singing |
| `modes` | 1 | 1: the playing mode only; 4: also the resonator's modes 1, 4 and 8 (low thumps and a close mode beating) |

Low notes speak slowly (the gesture starts slow and accelerates as the rod locks in); that is the
instrument. Each note's resonator is given a mobility proportional to its pitch, as the makers
individualise the weights per register; with one fixed resonator the minimum force would rise with pitch. Harmonics grow with `pressure` (unchanged at 1 N).

### `rubbed_glass` — wine glass / glass harmonica (C4-C7)
| param | default | meaning |
|---|---|---|
| `pressure` | 0.3 | finger force, N (0.05-3) |
| `speed` | 0.1 | finger speed, m/s (0.02-0.25). The glass only sings inside a window: too fast and it stops |
| `roughness` | 0.2 | 0 smooth .. 1 gritty (noise in the friction force) |
| `bright` | 1.0 | 0.5 .. 2 |
| `seed` | 0 | roughness noise |

### `bowed_bar` — bowed free bar (F3-F6)
| param | default | meaning |
|---|---|---|
| `material` | `aluminium` | `aluminium` (measured Baschet blade partials 1 : 2.71 : 4.87 : 8.67), `aluminium_b` (second measured Ikedaphone blade, no partial near 2.7), `iron` (Gamelinus blade), `steel`, `glass`, `wood` (theory 1 : 2.756 : 5.404 : 8.933; wood is the most damped), `comb` (a rubbed bar locking onto a harmonic series n f, the bow point choosing f) |
| `force` | 1.0 | bow force, relative (0.2-2.5): more = faster attack, harder tone |
| `position` | 0.03 | bow point from the bar end, fraction of length (0-0.15) |
| `speed` | 0.2 | bow speed, m/s (0.05-0.5): loudness |
| `bright` | 1.0 | 0.5 .. 2 |

### `bowed_sheet` — bowed steel sheet, steel-cello-like (C2-C5)
| param | default | meaning |
|---|---|---|
| `force`, `speed`, `bright` | 1.0, 0.2, 1.0 | as `bowed_bar` |
| `thickness` | 0.5 | sheet thickness, mm (0.3-0.8): thinner saturates more |
| `ring` | 0.5 | 0 dry bowed tone .. 1 mostly the sheet's own ringing modes |
| `glide`, `glide_time` | 0, 0.3 | semitones (−12..12) the sheet starts away from the note and slides onto it, and the slide time in s |
| `seed` | 0 | which sheet modes |

### `rod_bank` — Baschet percussion (G2-G5)
A struck clamped rod; its vibration crosses the collector, wakes the other seven rods of the set
(pitches from the Kit Barres lengths 880-610 mm by f ~ 1/L²) and leaves through the radiator.

| param | default | meaning |
|---|---|---|
| `rod` | `kit` | `kit` (measured partials of a 750 mm rod in its collector, with beating pairs), `clamped` (1 : 6.27 : 17.55 : 34.39), `weighted` (intermediate weights: 1 : 6 : 18 : 34), `mushroom` (end weight: short, one clear tone), `square` (square tube 1 : 2.67 : 5.04 : 8, very long), `piano` (Baschet piano rod: octave and major sixth), `bronto` (Brontosaure bar), `sad` (SAD bundle bar O), `fixed` (bolted blade: dry, short, vague pitch), `paired` (centre-clamped blade: the other half answers (180/160)² higher) |
| `strike` | `normal` | `normal`, or `tangential`: a grazing hit along the rod, only the fundamental, a pure whistle |
| `hardness` | 0.5 | 0 soft heavy rubber (4 ms contact) .. 1 hard wood (0.3 ms) |
| `decay` | 3.0 | ring time, s (0.1-12) |
| `sympathetic` | 0.35 | 0-1: how much the neighbouring rods ring along |
| `collector` | 4.0 | collector mass / rod mass (0.5-20): below 3 the node at the clamp does not form and the rod rings short and blurred |
| `radiator` | `sheet` | `sheet`, `cone`, `balloon`, `metal`, `none` |
| `seed` | 0 | sheet modes |

### `whistling_blades` — struck on the narrow edge (F#6-E8)
Two high partials per blade. Along the set the interval between them shrinks from an augmented fourth to
a crossing at the middle blade (about 2.8 kHz, where they beat at 5 Hz) and opens again to a major third.

| param | default | meaning |
|---|---|---|
| `decay` | 3.0 | ring time, s |
| `second` | 0.2 | level of the second partial (0-1). Real blades are nearer 0.5-1; that blurs the pitch |
| `spin` | 0 | rotation of the rack, turns/s (0-8): "waa-waa" tremolo |
| `hardness` | 0.7 | 0 soft .. 1 hard stick |
| `sumtone` | 0 | 0..1 the ear's sum tone: the two partials also heard at their sum (3000 + 3100 → about 6100 Hz) |

### `nail_violin` — bowed nails (C4-C7)
Iron nails clamped in a wooden soundbox and bowed; each nail rings its cantilever modes (1 : 6.27 : 17.5), the
box colours it.

| param | default | meaning |
|---|---|---|
| `force` | 1.0 | bow force, relative (0.2-2.5) |
| `speed` | 0.2 | bow speed, m/s (0.05-0.5) |
| `bow` | 0.9 | bow point along the nail from the board, 0..1 of its length |
| `bright` | 1.0 | 0.5 mellow .. 2 bright |

### `marloye_harp` — rubbed rods (C4-C7)
Rods standing in a soundbox, rubbed lengthwise with rosined fingers: they sing their longitudinal modes (odd
harmonics 1 : 3 : 5 : 7).

| param | default | meaning |
|---|---|---|
| `force` | 1.0 | finger pressure, relative (0.2-2.5) |
| `speed` | 0.15 | finger speed, m/s (0.05-0.5) |
| `material` | `steel` | `steel` or `wood` (damper, softer) |
| `bright` | 1.0 | 0.5 mellow .. 2 bright |

### `tuning_fork` — fork with balloon or cone radiator (G3-D7)
| param | default | meaning |
|---|---|---|
| `radiator` | `balloon` | `balloon`, `cone`, `none` |
| `pressure` | 0.7 | 0-1: balloon pressed on the fork: louder, the note is not shortened |
| `swell` | 0 | seconds over which the pressure is applied after the strike (crescendo) |
| `hardness` | 0.25 | 0 thick felt (pure tone) .. 1 bare hammer (clang partial at 6.27 x) |
| `decay` | 6.0 | ring time, s |

### `coil_spring` — struck or scraped spring (C2-C5)
A train of chirped echoes (high partials travel faster) around a sustained tone.

| param | default | meaning |
|---|---|---|
| `excite` | `strike` | `strike` or `scrape` (granular, rumbling; the grains are noise, so the pitch is only approximate) |
| `dispersion` | 0.5 | 0 nearly harmonic echoes .. 1 strong chirps |
| `tone` | 0.7 | 0 blurred pitch .. 1 the note clearly on top (below about 0.5 the pitch is no longer exact) |
| `decay` | 4.0 | ring time, s |
| `hardness` | 0.5 | 0 soft mallet .. 1 hard stick (laser-like chirp) |
| `wire`, `coil`, `turns` | 0.1 mm, 2.5 mm, 112 | wire radius, helix radius and number of turns: they set the echo round trip 4 L R / (r c) (default about 35 ms) |
| `seed` | 0 | scrape grains |

### `musical_saw` — bowed bent saw blade (C4-C6)
A bowed flexible steel blade (musical saw, Baschet flexible disc): bending it sets the pitch, so it slides onto its
note and sings with a wide knee vibrato; near-harmonic when bowed (Bilbao bow on modes whose frequency moves),
radiated by a cone.

| param | default | meaning |
|---|---|---|
| `glide`, `glide_time` | −2, 0.25 | semitones (−12..12) the note starts away from its pitch and bends onto it, and the bend time in s |
| `vibrato`, `rate` | 0.25, 5.5 | vibrato depth in semitones (0-1, the player's knee shaking the blade) and rate in Hz (3-8) |
| `force`, `speed` | 1.0, 0.2 | bow force, relative (0.2-2.5), and bow speed, m/s (0.05-0.5) |
| `radiator` | `cone` | `cone`, `sheet`, `none` |
| `bright` | 1.0 | 0.5 mellow .. 2 bright |

### `daxophone` — bowed wooden tongue (C3-C6)
A wooden tongue clamped at its foot and bowed; a wooden "dax" pressed on it sets the free length, and with it the
pitch (f ~ 1/L², `dax_free_length(freq)`; tongue 330 x 30 x 5 mm). The bow point weights the cantilever modes: near
the dax the tone toggles to the second mode (x6.27), as players describe.

| param | default | meaning |
|---|---|---|
| `bow` | 0.8 | bow point from the dax toward the free tip, 0..1 of the free length: nearer the dax the tone toggles up |
| `force` | 1.0 | bow force, relative (0.2-2.5) |
| `speed` | 0.2 | bow speed, m/s (0.05-0.5) |
| `bright` | 1.0 | 0.5 mellow .. 2 bright |

## Sound effects

| name | params (default) | notes |
|---|---|---|
| `squeak` | `surface` hinge\|shoe\|chalk (hinge), `pressure` (1), `speed` (1), `pitch` (1), `dur` (0.6), `model` (physical), `seed` | friction locking onto one resonance; a hinge is two such pairs. `model=additive`: an additive friction source (harmonics 1/k on a wandering fundamental) through the same modes; `physical` (bristle friction on the modes) is the default, unchanged |
| `brake_squeal` | `speed` m/s (1.5), `pressure` N (4), `pitch` Hz (1900), `dur` (1.6), `kind` (rub), `hot_spots` (0), `seed` | rubbing noise while fast, squeal in the last part of the stop. `kind`: `rub` (default), `squeal_hf` (5-10 nodal diameters, 5-15 kHz, with locked harmonics), `squeal_lf` (1-4 nodal diameters, 1-4 kHz), `squeak` (short squeal), `groan` (10-30 Hz stick-slip at low speed), `judder` (whole multiples of the wheel rate), `wire_brush` (squeal randomly modulated), `squelch` (squeal amplitude-modulated). `hot_spots` 0..1: thermo-elastic hot spots, the grip pulses about 10 times per disc turn |
| `rub` | `surface` balloon\|glass\|wood (balloon), `pressure` (1), `speed` (1), `strokes` (3), `dur` (1.2), `seed` | hand moving back and forth |
| `stick_slip` | `mu_s` (0.975), `mu_d` (0.197), `stiffness` N/m (2e4), `speed` m/s (0.01), `pressure` N (8), `freq` (420), `dur` (1.5), `seed` | generic creak: slips repeat about every 2 (mu_s - mu_d) pressure / (stiffness x speed) seconds; slow = cracks, fast = groan |

Physical meaning of the friction controls: `mu_s` is the grip before sliding, `mu_d` the grip while
sliding; the bigger the gap, the more violent each slip. `pressure` scales both. `speed` decides whether
the contact sticks at all: stick-slip needs a speed where friction still falls as sliding gets faster.

## Effects

| name | params (default) | notes |
|---|---|---|
| `sheet_radiator` | `thickness` mm (0.5), `size` m (1.5), `drive` (1), `decay` s (1.5), `mix` (0.5), `bright` (1), `metal` (""), `mute` (0), `seed` | steel-sheet modes with uneven gains; quiet input passes almost clean, loud input saturates and wakes low "thunder" modes. Thinner = earlier saturation, more strident; thicker = fuller. `metal` galvanized or stainless: modes up to about 7 / 15 kHz ("" = up to 4500 Hz × bright). `mute` 0..1: a foam clamp pressed on the sheet by a pedal (shorter ring, later saturation) |
| `cone_radiator` | `size` m (0.6), `material` cardboard\|metal\|balloon, `mix` (1), `seed` | no tail (cardboard, balloon) or a short bright ring (metal); low cut = 60/size Hz |
| `sympathetic` | `tuning` ("D3 A3 D4 F#4 A4", or `random`), `kind` string or rod, `decay` s (3), `mix` (0.3), `spread` (2 octaves), `seed` | strings (harmonic) or free-end rods ("whiskers", 1 : 6.27 : 17.55 : 34.39) that ring when the input hits their partials; `tuning="random"`: five fundamentals drawn log-uniform over `spread` octaves above 110 Hz |
| `force_pitch` | `note` (A3, Hz, or `nearest`), `floor` Hz (40), `mix` (1) | forces a voiced input onto one note (or the nearest semitone): f0 tracked (`pitch_clean`) and each moment shifted by the ratio to the target; frames under `floor` or unvoiced are left alone |

All three accept mono or stereo.

## Recipes

```jsonc
// Cristal melody through its steel sheet
{"out": "cristal.wav", "bpm": 60, "layers": [
  {"type": "seq", "inst": "cristal", "step": 1, "steps": "D4:2 F4:1 A4:1 G4:2 D4:2",
   "params": {"pressure": 1.2, "finger_speed": 0.18, "radiator": "sheet"}}],
 "fx": [{"type": "reverb", "mix": 0.2, "size": 0.7}]}

// Baschet percussion: rods ringing into each other, extra whiskers on top
{"out": "rods.wav", "layers": [
  {"type": "seq", "inst": "rod_bank", "steps": "G3:.4 B3:.4 D4:.4 G3:.8", "params": {"rod": "kit", "hardness": 0.7, "sympathetic": 0.6},
   "fx": [{"type": "sympathetic", "kind": "rod", "tuning": "G4 D5 G5", "mix": 0.25}]}]}

// Kawakamiphone: struck rods with free-end rod whiskers tuned to the instrument
{"out": "kawakamiphone.wav", "layers": [
  {"type": "seq", "inst": "rod_bank", "steps": "C4:.5 E4:.5 G4:1", "fx": [{"type": "sympathetic", "kind": "rod", "tuning": "48 65 90 138", "mix": 0.3}]}]}

// Door: slow creak, then a hinge squeak
{"out": "door.wav", "layers": [
  {"type": "sfx", "kind": "stick_slip", "params": {"speed": 0.006, "stiffness": 30000, "pressure": 10, "freq": 380, "dur": 1.2}},
  {"type": "sfx", "kind": "squeak", "at": 1.0, "gain": 0.6, "params": {"surface": "hinge", "dur": 0.7, "pitch": 0.9}}]}

// Any sound through a thin steel sheet: thunder
{"out": "thunder_sheet.wav", "layers": [
  {"type": "drum", "kind": "taiko", "fx": [{"type": "sheet_radiator", "thickness": 0.3, "size": 3, "drive": 4, "decay": 5, "mix": 0.9}]}]}

// Wine-glass chord and a train braking
{"out": "glass.wav", "layers": [{"type": "synth", "inst": "rubbed_glass", "notes": "C5 E5 G5", "dur": 2.5, "strum": 0.4, "params": {"speed": 0.08}}]}
{"out": "brake.wav", "layers": [{"type": "sfx", "kind": "brake_squeal", "params": {"speed": 3, "pressure": 6, "pitch": 2300, "dur": 2.5}}]}
```

## What is not from a source

Chosen without a direct source (each justified in `research/SOURCES.md`): all bristle parameters except the book's mu_s, mu_d, v_s; the resonance
frequencies of the sfx bodies (the book tunes them by ear too); the sheet's nonlinearity (the thesis
describes it only in words); partial levels of the rod tables (only frequencies are given); the coil
spring's mode law (Bilbao's §7.9 helical-spring equations are not used here); `cristal`'s
`wetness` mapping, pitch-proportional mobility and harmonic mix. The whistling-blade beat uses the quoted
4-6 Hz; the thesis also prints a measured pair 2870/2917 Hz (47 Hz apart) that contradicts it.
