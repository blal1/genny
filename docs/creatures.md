# Creatures: animals, insects, birds, body, crowds, footsteps (`genny/creatures.py`)

35 sound effects and one instrument for living things. Every sfx is dry mono, peak-normalised, and takes a
`seed` (same params give the same samples). Kernels run at 48 kHz and are resampled to the spec's `sr`.

Sung voice and choirs are not here: see `genny/choir.py`.

Numbers come from Farnell (*Designing Sound*, Practicals 26-29 and his Pd patches), Cook, Visell, Smyth & Smith,
Dolbear, Weissler and the LovelyHeart patch. Where no source gives a number the code line is marked `# UNSOURCED`;
the big unsourced groups are named below.

## The one control that runs through everything: `size`

`size` is a body scale. Fundamental frequency is `pitch / size` and the vocal tract is `size` times longer, so every
resonance drops by the same factor (a quarter-wave tube resonates at `c / 4l`: 500 Hz for a 0.17 m human tract).
Measured: `animal` at size 0.5 / 1 / 2 gives 400 / 200 / 100 Hz; its tract modes move from 490 and 2487 Hz to 264
and 1225 Hz; `frog` at size 2 has half the spectral centroid (1592 to 793 Hz).

## Mammals (Farnell's animal generator)

Source: a "flapping cord" waveshaper. Tract: a parallel band-pass comb at the tube modes plus three vowel formants.
`aggression` / `roughness` raise the ripple, narrow the pulses and add noise ("narrow pulses with lots of ripple and
some noise give a harsh, gritty, snarling excitation"), plus cycle jitter and period doubling.

| name | params (default) | notes |
|---|---|---|
| `roar` | `size` (1), `aggression` (0.7), `dur` (2.5), `seed` | lion arc 30 -> 240 -> 120 Hz divided by size (Farnell) |
| `growl` | `size` (1), `aggression` (0.6), `pitch` (70 Hz), `dur` (1.5), `seed` | wobbling low pulses, period doubling |
| `bark` | `size` (`medium`: small, medium, large), `count` (2), `aggression` (0.6), `pitch` (0 = from size), `seed` | dog |
| `meow` | `size` (1), `pitch` (550 Hz), `dur` (0.9), `seed` | cat near 500 Hz (Farnell), vowel path i-ae-A-u |
| `purr` | `rate` (25 pulses/s), `size` (1), `dur` (3), `seed` | glottal pulses through the tract, breathing cycle |
| `moo` | `size` (1), `pitch` (115 Hz), `dur` (1.8), `seed` | closed-mouth "mm" opening to "oo" |
| `howl` | `size` (1), `pitch` (420 Hz), `dur` (3), `seed` | wolf |
| `monster` | `size` (2.5), `aggression` (0.8), `pitch` (220 Hz at size 1), `dur` (2), `seed` | subharmonics, double tract, drifting vowels |
| `animal` | `size` (1), `pitch` (200), `contour` (`arch`: flat, rise, fall, arch, dip, wobble), `excursion` (0.3), `roughness` (0.3), `vowels` (`A-u`), `dur` (0.8), `seed` | the generic generator |

`vowels` is a path of Farnell's vowel names joined by `-`: `i:` meet, `I` ship, `E` pet, `ae` cat, `^` love,
`u` root, `U` hook, `@` about, `A` father.

Unsourced: every preset number except the lion arc and the cat's 500 Hz (dog pitches, tract lengths, call shapes).

## Insects

| name | params (default) | notes |
|---|---|---|
| `cricket` | `species` (`field`: field, tree, bell), `temp` (18 degC), `rate` (0 = from temp), `bright` (0.5), `dur` (3), `seed` | chirp rate from temperature |
| `cicada` | `rate` (120 Hz clicks), `tone` (0.85), `dur` (3), `seed` | Farnell `circada2`; `tone` 1 = 5-8 kHz |
| `fly` | `freq` (220 Hz), `wobble` (0.5), `motion` (0.5), `proximity` (0.7), `tone` (2000 Hz), `dur` (2), `seed` | Farnell's measured harmonic table |
| `mosquito` | same, `freq` 600, `tone` 4200 | 3 % wing pulse through a Q 12 band (Farnell's patch uses 8000) |
| `bee` | same, `freq` 230, `tone` 2600 | Farnell's patch uses 6000 |
| `night_insects` | `density` (0.5), `temp` (20), `bright` (0.5), `dur` (6), `seed` | gated crickets + "critters" + "chirper" bed |
| `wings` | `kind` (`bird`: bird, bat, insect), `rate` (0 = 6 / 11 / 35 flaps/s), `size` (1), `dur` (2), `seed` | fan pulse into swept air noise |

- **Dolbear's law**: chirps per minute = `7.2 T - 32` with T in degC (T_F = 50 + (N - 40)/4). Measured in 12 s:
  14 / 23 / 32 chirps at 14 / 20 / 26 degC (law: 13.8 / 22.4 / 31.0). The `bell` cricket runs 2.52 times faster.
- The field cricket is Farnell's recording: 7 pulses per chirp, 17 ms apart, amplitude rising, 4.5 kHz falling to
  3.9 kHz in each pulse, two wings at 4.4 and 4.6 kHz.
- `freq` is the wingbeat fundamental (measured 220.4, 601.2, 230.5 Hz for 220, 600, 230). `wobble` 1 is Farnell's
  "as much as 20 percent" warble, common to both wings. `motion` circles the insect around the listener (Doppler
  from the radial speed, level as 1/distance). `proximity` sets the distance and the top end.
- Insects are high by nature. Defaults sit lower than the patches; raise `tone` / `bright` for realism and keep
  the layer quiet.

## Birds

`birdsong` builds a song from the hierarchy element -> syllable -> phrase -> song and plays it on Farnell's FM
syrinx: a pulse oscillator `1/(1 + (w cos)^2)` (small `w` = pure whistle, large = harmonic call), FM from a bronchial
pulse oscillator (trills), feedback and noise FM for the rough non-songbirds, two tracheal band-passes, a beak
high-pass.

| name | params (default) |
|---|---|
| `birdsong` | `species` (`robin`), `repeats` (1), `syllables` (0 = preset), `pitch` (1), `tempo` (1), `two_voice` (0), `bright` (1), `seed` |
| `bird_call` | `freq` (450 Hz, 150-900), `freq_end` (0 = no glide), `pressure` (3 kPa, 1.5-6), `attack` (0.3), `two_voice` (5 Hz), `species` (`raven`: oil_bird, raven, budgie), `dur` (0.5) |
| `dawn_chorus` | `birds` (6), `dur` (8), `bright` (1), `seed` |

Species: `canary` (three trill phrases), `robin` (varied warble), `blackbird` (slow flutes then a twitter),
`sparrow` (cheeps), `owl` (four hoots), `crow` (caws), `gull`, `duck` (quacks slowing and fading), `rooster`
(four syllables), `cuckoo` (667 then 530 Hz), `pigeon` (purring coos), `woodpecker` (12-16 strikes on a wooden body).

**All preset numbers are unsourced tuning.** Kept from the sources: songbird syllables 5-300 ms, songbird pitch
mostly 3-5 kHz, tracheal Q 2-5, and the raven-type trachea resonances 1200 / 2200 Hz used for the crow.
`creatures.birdsong_plan(species, seed, ...)` returns the syllable list; `creatures.bird_range(species)` the pitch
range. Measured: the rendered syllable count equals the plan for every tested species.

`bird_call` is the physical model (Smyth & Smith): air-sac `pressure` drives two membrane valves into a trachea
waveguide. `freq` is the membrane frequency, i.e. the tension; the call follows it (294.9 / 444.7 / 694.3 Hz for
300 / 450 / 700). It is harmonically rich and raven-like, not a whistle. `two_voice` detunes the second valve.

## Frogs and reptiles

| name | params (default) | notes |
|---|---|---|
| `frog` | `kind` (`croak`: croak, ribbit, trill), `size` (1), `calls` (3), `gap` (0.5 s), `seed` | Farnell's three frog patches |
| `rattlesnake` | `rate` (20 pulses/s), `tone` (0.85), `dur` (1.5), `seed` | Farnell `rattler` |
| `hiss` | `kind` (`snake`: snake, cat), `size` (1), `dur` (0 = from kind), `seed` | noise through fricative formants (mapping unsourced) |

## Body

| name | params (default) | notes |
|---|---|---|
| `breath` | `kind` (`out`: in, out, pant, gasp, sigh), `effort` (0.5), `size` (1), `dur` (0 = from kind), `seed` | noise through whispered-vowel formants (Cook) |
| `grunt` | `pitch` (110 Hz), `effort` (0.5), `size` (1), `dur` (0.3), `seed` | LF glottal pulse through a neutral tract |
| `cough` | `count` (2), `effort` (0.7), `size` (1), `seed` | unsourced recipe |
| `sneeze` | `effort` (0.8), `size` (1), `seed` | unsourced recipe |
| `snore` | `rate` (14 breaths/min), `pitch` (45 Hz flutter), `size` (1), `dur` (8), `seed` | unsourced recipe |
| `gulp` | `size` (1), `seed` | unsourced recipe |
| `eat` | `kind` (`crunch`: crunch, chew, slurp), `bites` (4), `rate` (1.7 /s), `seed` | PhISEM "Crunch" grains driven by the jaw |
| `heartbeat` | `rate` (1.0 beats/s), `dur` (2), `murmur` (0), `tone` (1), `dub_delay` (0 = physiological), `seed` | **replaces** the old `heartbeat` |
| `handclap` | `cup` (0.3), `seed` | one clap; cupped = lower |
| `applause` | `crowd` (30), `enthusiasm` (0.6), `dur` (4), `seed` | |
| `crowd_murmur` | `people` (20), `mood` (0.5), `distance` (0.5), `dur` (5), `seed` | gated talking-pulse voices |
| `whistle_human` | `kind` (`wolf`: wolf, up, down, note, taxi), `freq` (1800 Hz or a note), `breath` (0.3), `speed` (1), `seed` | |

- **Heartbeat**: S1 "lub" is exactly 5 cycles of 63 Hz, S2 "dub" 3 cycles of 42 Hz plus its octave, both starting
  at phase zero with a squared decay (LovelyHeart). The S1-S2 gap is the ejection time, `0.453 - 0.0017 bpm`
  seconds (Weissler's regression plus 0.04 s, the added constant is unsourced): 0.351 s at 60 bpm, 0.249 s at 120.
  `rate` keeps its old meaning (beats per second). `tone` scales both thump pitches.
- **Applause**: each person has a clap rate (2 /s polite to 4.5 /s wild, unsourced), a hand resonance and a
  level; total claps = crowd x rate x dur (measured 557 for a predicted 560). Above 64 people the rest of the crowd
  is a Poisson grain bed with the same spectrum (Farnell: beyond about 20 claps/s, blur).

## Footsteps

`footsteps` sequences the 24 ground kernels. Every step is synthesised again from its own ground-reaction-force
curve, so no two are the same samples (the looped identical step is what listeners call "mechanical").

| param | default | meaning |
|---|---|---|
| `ground` | `wood` | snow, sand, wet_sand, ash, mud, shallow_water, leaves, grass, forest_floor, gravel, pebbles, salt, stone, concrete, cobblestone, wet_stone, marble, hard_floor, wood, metal_web, ice, crystal, bone, coral |
| `gait` | `walk` | walk, run, sneak, limp, stairs, trot, gallop |
| `pace` | 0 | footfalls per minute; 0 = walk 105, run 170, sneak 66, limp 84, stairs 96, trot 312, gallop 440 |
| `steps` | 0 | footfalls; 0 = 8 (16 for trot / gallop) |
| `weight` | 1 | 0.4-3: heavier is lower, longer, bassier |
| `shoe` | `default` | default, barefoot, boot, heel, hoof |
| `variation` | 0.5 | timing and level irregularity |
| `seed` | 0 | |

Walking is legato (the feet overlap), running staccato (a silent flight phase), sneaking soft-onset and dull,
a limp alternates short and long gaps (0.72 : 1.28). `trot` is two beats of diagonal hoof pairs, `gallop` four
beats then a suspension; both default to the `hoof` shoe. Measured: 8 steps at 100 /min give 8 onsets 0.602 s apart.
Unsourced: the gait defaults other than walk / run, the quadruped beat fractions, the shoe and weight shaping.

## Instrument

| name | family | range | params |
|---|---|---|---|
| `whistling` | winds | C5-C7 | `vibrato` (0.15 semitones), `breath` (0.3), `seed` |

A near-pure tone with breath noise around the pitch. In tune within 2 cents, about -12 dB at vel 0.9.

## Recipes

```jsonc
// Night in a meadow, 24 degC, a frog pond nearby (loopable)
{"out": "amb/night_meadow.wav", "duration": 8, "loop": 0.5, "layers": [
  {"type": "sfx", "kind": "night_insects", "params": {"dur": 8.5, "temp": 24, "density": 0.6}, "gain": 0.5},
  {"type": "sfx", "kind": "frog", "params": {"kind": "ribbit", "calls": 6, "gap": 0.9, "size": 1.3}, "at": 1.0, "gain": 0.35,
   "fx": [{"type": "lowpass", "cutoff": 2500}]},
  {"type": "sfx", "kind": "birdsong", "params": {"species": "owl"}, "at": 3.5, "gain": 0.3}],
 "fx": [{"type": "reverb", "mix": 0.25, "size": 0.7}]}

// A horse gallops past on cobblestones
{"out": "foley/gallop.wav", "layers": [
  {"type": "sfx", "kind": "footsteps", "params": {"ground": "cobblestone", "gait": "gallop", "steps": 24, "weight": 2.2}}],
 "fx": [{"type": "reverb", "mix": 0.12, "size": 0.4}]}

// Boss monster: roar with a growl underneath, big and far
{"out": "creature/boss_roar.wav", "layers": [
  {"type": "sfx", "kind": "monster", "params": {"size": 4, "aggression": 1.0, "dur": 2.6}},
  {"type": "sfx", "kind": "roar", "params": {"size": 2.5, "dur": 2.6}, "gain": 0.7},
  {"type": "sfx", "kind": "growl", "params": {"size": 3, "dur": 2.0}, "at": 2.2, "gain": 0.5}],
 "fx": [{"type": "reverb", "mix": 0.3, "size": 0.85}]}

// Tense: someone creeping on gravel, heart racing
{"out": "tension/sneak.wav", "layers": [
  {"type": "sfx", "kind": "footsteps", "params": {"ground": "gravel", "gait": "sneak", "steps": 6}},
  {"type": "sfx", "kind": "heartbeat", "params": {"rate": 1.9, "dur": 5.5}, "gain": 0.8},
  {"type": "sfx", "kind": "breath", "params": {"kind": "pant", "effort": 0.3, "dur": 5}, "gain": 0.25}]}

// Win: a small crowd claps and someone whistles
{"out": "win/crowd.wav", "layers": [
  {"type": "sfx", "kind": "applause", "params": {"crowd": 200, "enthusiasm": 0.9, "dur": 4}},
  {"type": "sfx", "kind": "whistle_human", "params": {"kind": "taxi", "freq": 2200}, "at": 0.8, "gain": 0.3}],
 "fx": [{"type": "reverb", "mix": 0.2, "size": 0.6}, {"type": "fade", "out": 1.2}]}
```
