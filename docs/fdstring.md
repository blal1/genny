# Finite-difference strings and bars (`genny/fdstring.py`)

Grid models from Bilbao, *Numerical Sound Synthesis* (2009), ch. 6-8: the string or bar is a row of points
advanced sample by sample, so things can touch it at a point (hammer, bow, rattle, rubber, mallet) and react
to what it is doing at that instant. That is what these voices add over genny's waveguide and modal ones:
tone that changes with velocity, buzzes that start only when the string moves far enough, pitch that sags
after a hard pluck.

Registered names: instruments `fd_string`, `fd_piano`, `prepared_piano`, `slack_string`, `fd_bar`,
`bowed_string_fd`; sfx `twang`; fx `spring_reverb`.

## Instruments

All take `bright` (default 1; 0.5 mellow .. 2 open). It scales the output ceiling, which is about 6 partials
of the note kept inside 1.6-4.2 kHz.

### `fd_string` (strings, E1-C6)
Stiff string, plucked, struck or bowed.

| param | default | meaning |
|---|---|---|
| `excitation` | `pluck` | `pluck` (shaped displacement, released), `strike` (shaped velocity), `bow` |
| `position` | 0.18 | excitation point, 0..1 along the string. 0.5 removes the even partials (hollow) |
| `B` | 1e-4 | stiffness at C4: partial n sits at n·f0·sqrt(1 + B·n²). 0 = ideal string, 1e-3 = clangy. Scaled by (f/C4)² over the keyboard |
| `t60_lo` | 4.0 | seconds for the note itself to fall 60 dB |
| `t60_hi` | 0.6 | the same at max(4 kHz, 3x the note). Close to `t60_lo` = bright all the way; small = bright attack that mellows |
| `pickup` | 0.3 | readout point (a second one sits 0.43 further along; the two are mixed) |

Velocity widens the pluck (soft = wide = dull) as well as lowering the level.

### `fd_piano` (keys, A0-C7)
A felt hammer with its own mass flies into 1-3 strings and is thrown back by them. On C4 contact lasts about
1.3 ms at `vel` 0 and 0.8 ms at `vel` 1, so loud notes are brighter, not just louder.

| param | default | meaning |
|---|---|---|
| `strings` | 0 | strings per note, 1-3; 0 = by register (1 up to C#2, 2 up to F3, 3 above) |
| `detune` | 3.0 | total spread of the course, cents: slow beating and a faster early decay |
| `hammer_mass` | 1.0 | scale on hammer mass (1 = 0.75x the string's mass at C4). Heavier = longer contact, darker, more thump |
| `hardness` | 1.0 | felt stiffness scale, 0.2-5. Harder = shorter contact, brighter |
| `position` | 0.12 | strike point |
| `B` | 1e-4 | inharmonicity at C4, scaled by (f/C4)² |
| `decay` | 6.0 | seconds of ring at C4 (longer in the bass, shorter in the treble) |
| `t60_hi` | 1.0 | seconds of ring at max(4 kHz, 3x the note) |

### `prepared_piano` (keys, C2-C6)
One hammered string with an object on it.

| param | default | meaning |
|---|---|---|
| `preparation` | `rattle` | `rattle`: light loose piece, buzzes when the string swings past its gap (stays in tune). `rubber`: wedge, damps and mutes. `bolt`: heavy loose piece, clanks, detunes and splits partials. `screw`: stiff spring, inharmonic and gong-like, more so when hit hard |
| `position` | 0.3 | where the object sits |
| `amount` | 1.0 | 0 (plain string) .. 3 |
| `decay` | 4.0 | seconds of ring |

Only `rattle` is held to pitch; the other three move the partials on purpose. Rattles need a firm hit:
below about `vel` 0.4 the string does not reach them.

### `slack_string` (strings, E1-E5)
A loose string whose tension rises while it is stretched: a hard pluck starts sharp and settles onto the
note, a soft one does not move.

| param | default | meaning |
|---|---|---|
| `glide` | 0.7 | semitones sharp at the first instant of a full-velocity pluck (scales with `vel`²). Up to about 1 the note still reads in tune; 3-6 is a rubber-band boing |
| `position` | 0.8 | pluck point |
| `decay` | 1.2 | seconds of ring |
| `pickup` | 0.3 | readout point |

### `fd_bar` (mallet, C3-C7)
A bar hit by a mallet with mass and a stiffening tip.

| param | default | meaning |
|---|---|---|
| `material` | `wood` | ring time: `wood` (0.7 s at A4), `plastic` (0.25 s), `glass` (1.6 s), `metal` (4 s) |
| `boundary` | `free` | `free`: partials 1 : 2.76 : 5.40 (glockenspiel, chime). `clamped_free`: 1 : 6.27 : 17.5 (tine, ruler, kalimba). `clamped`: 1 : 2.76 : 5.40, thuddier. `supported`: 1 : 4 : 9 |
| `tuning` | `none` | undercut arch on a free bar: `xylophone` puts partial 2 at 3x, `marimba` at 4x (voiced for C3-C6; it gets quiet above) |
| `mallet` | 0.5 | 0 soft yarn (contact about 2.7 ms at A4, round) .. 1 hard (0.45 ms, click) |
| `position` | 0.5 | strike point; the middle favours the fundamental, 0.1-0.3 brings out the upper partials |
| `pickup` | 0.42 | readout point |
| `decay` | 1.0 | ring-time scale |
| `quality` | 1 | grid of 20 / 32 / 48 points: finer = upper partials closer to their true ratios, slower |

### `bowed_string_fd` (strings, G2-C6)
Bowed stiff string; the bow point can move during the note.

| param | default | meaning |
|---|---|---|
| `force` | 1.0 | bow pressure scale 0.05-4: more = harder edge, quicker start, slightly flat |
| `speed` | 1.0 | bow speed scale 0.1-4 |
| `position` | 0.12 | bow point from the bridge; small = glassy, 0.2+ = flutey |
| `travel` | 0.03 | how far the bow point drifts over the note: partials swell and fade as it passes their nodes |
| `B` | 5e-5 | inharmonicity at C4 |
| `decay` | 2.5 | seconds of free ring after the bow lifts |
| `pickup` | 0.3 | readout point |

## SFX and FX

**`twang`** (sfx): `kind` `band` (rubber band: starts up to 4 semitones sharp and sags) or `ruler`
(clamped ruler flicked on a desk edge, slapping the desk); `freq` 140 Hz settled pitch; `dur` 0.9 s;
`amount` 1 (0-2: glide depth, or how hard the ruler slaps).

**`spring_reverb`** (fx): chirping spring-tank echoes. `mix` 0.3, `decay` 2 s, `delay` 0.045 s per trip
along the spring, `cutoff` 4000 Hz (top of the chirps), `density` 60 (10-200: more = longer chirps). This is
a dispersive allpass-chain loop, **not** the book's helical-spring model, and its constants have no source.

## Recipes

```jsonc
// dark upright piano line
{"type": "seq", "inst": "fd_piano", "steps": "C3:1 G3:1 [E4,G4,C5]:2", "step": 1,
 "params": {"hardness": 0.6, "detune": 5, "decay": 4}, "fx": [{"type": "reverb", "mix": 0.15}]}

// prepared-piano gamelan: screws low, a rattle on top
{"type": "seq", "inst": "prepared_piano", "steps": "D3:.5 A3:.5 F3:.5 C4:.5", "vel": 0.9,
 "params": {"preparation": "screw", "position": 0.4, "amount": 1.5}}
{"type": "synth", "inst": "prepared_piano", "notes": "D5", "dur": 1, "vel": 1.0, "params": {"preparation": "rattle"}}

// cartoon boing
{"type": "sfx", "kind": "twang", "params": {"kind": "band", "freq": 110, "amount": 1.6, "dur": 1.2},
 "fx": [{"type": "spring_reverb", "mix": 0.35, "decay": 1.5}]}

// glockenspiel-like and marimba-like bars
{"type": "seq", "inst": "fd_bar", "steps": "C6 E6 G6 C7", "params": {"material": "metal", "mallet": 0.9, "position": 0.3}}
{"type": "seq", "inst": "fd_bar", "steps": "C4 E4 G4 C5", "params": {"tuning": "marimba", "mallet": 0.2}}

// slow bowed line, bow sliding toward the fingerboard
{"type": "seq", "inst": "bowed_string_fd", "steps": "D4:2 F4:2 A4:4", "step": 1, "bpm": 80,
 "params": {"travel": 0.15, "force": 0.7}}
```

## Python

```python
from genny.fdstring import StiffString, Bar
s = StiffString(110.0, 44100, B=1e-3, t60=(110, 8.0, 4000, 2.0), n_strings=3, detune=5.0)
s.hammer(pos=0.12, mass=0.25, w=3000.0, alpha=2.5, v=2.0).rattle(0.3, w=8000.0, eps=1e-3, mass_ratio=100.0)
stereo = s.render(2.0, pickups=(0.3, 0.73))      # (n, 2): one pickup per channel
b = Bar(440.0, 44100, bc=("free", "free"), ratio=4.0)      # arch cut so that partial 2 = 4x
```
`pluck`, `strike`, `hammer`, `spring` (spring / cubic spring / damper), `rattle`, `bow`, `run(n, energy=True)`
(returns the discrete energy as well) and `partial(n)` are the whole interface. Displacements are in units
of the string length.

## What the model does, and its limits

- **Tuning.** The grid is sized by the stability rule, then the wave speed is corrected so the *scheme's*
  first partial lands on the note (0.00 cents measured). Stiffness stretches the upper partials, which pulls
  the heard pitch sharp, so `fd_string`, `fd_piano` and `bowed_string_fd` are tuned a few cents flat
  (3, 2 and 5 cents at default settings) to read on pitch. Raising `B` well above the default makes the note
  read sharp, as on a real stiff string.
- **Decay.** `t60_lo` / `t60_hi` are real 60 dB times. The book prints T60 = 6 ln10 / sigma but its own
  listing halves the loss terms; this module uses sigma = 3 ln10 / T60 with the full terms, which is the same.
- **Bars** run at up to 1.5 MHz internally on a fixed grid, because the explicit bar scheme at audio rate
  puts the third partial of a tuned bar up to a semitone flat. At the top of the range the grid is coarsened
  to hold that rate.
- **Free bars** are hung from an ideal suspension: the part of the mallet blow that would only push or spin
  the whole bar is removed.
- **Contacts** (hammer, rattle) are solved one grid point at a time, and their stiffness is clamped to what
  the time step resolves. Contact make/break is not exactly energy-conserving: a linear-felt hammer gains
  0.2 % of its energy over one strike, the default alpha = 2.5 felt 0.01 %.
- **Tension modulation and point objects are not solved together**: `slack_string` and `twang` have no
  hammer or preparation.
- Not implemented: the helical-spring reverb scheme, the 0.7026 implicit bar scheme, phantom partials
  (longitudinal string motion), whirling strings, coupled bars, `wire_snap`.
