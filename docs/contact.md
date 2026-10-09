# Contact sounds: solid objects hitting, bouncing, breaking, crumpling, rolling, scraping

Module `genny/contact.py`. Everything here is built from three parts, as in *The Sounding Object*
(Rocchesso & Fontana, 2003) and van den Doel's modal work:

1. **a resonator**: a modal body whose pitch comes from its shape, size and material, and whose mode
   levels come from where it is touched;
2. **an interaction**: a physical contact (a striker mass with a non-linear contact spring), or a force
   signal (a surface profile, noise, an engine cycle);
3. **a temporal pattern**: one hit, an accelerating bounce train, a thinning cloud of fragments, a
   stochastic crumple burst, continuous rolling.

What tells the listener *what happened* is mostly the pattern (bouncing vs breaking vs rolling). What
tells them *what it is made of* is the decay time; *how big* is the pitch; *how hard it was hit* is the
attack brightness and the level.

## Sound effects (`"type": "sfx"`)

| kind | what it is |
|---|---|
| `impact` | one object struck by another |
| `bounce` | an object dropped on a surface, bouncing to rest |
| `drop` | a ball: bounce, then roll away, then settle |
| `smash` | something brittle breaking on a hard floor |
| `crumple` | crushing a can, crumpling paper, a bottle, foil, a bag |
| `roll` | a ball rolling on a surface (replaces the old `roll`) |
| `scrape` | scraping / sliding along a surface (replaces the old `scrape`) |
| `rumble` | low rumble: earthquake, heavy machinery, distant collapse |
| `avalanche` | rock slide / avalanche rumble with a long tail |
| `modal_engine` | four-stroke engine as a force loop on a low plate body |

Materials everywhere: `rubber`, `plastic`, `wood`, `glass`, `ceramic`, `metal`, `steel`, `aluminium`,
`bronze`, `ice`, `stone`, `concrete`. A material sets three things at once: how long it rings, how
stiff the object is (pitch for a given size) and how hard its surface is in a contact.

Body shapes: `string`, `bar` (free ends), `bar_clamped` (one end held, like a ruler on a table edge),
`bar_supported`, `bar_clamped_clamped`, `rect_membrane`, `rect_plate`, `circ_membrane`, `circ_plate`.
Where a `shape` / `body` is accepted you can also name a measured object from the modal tables:
`sy_vase`, `sy_wok`, `sy_sword1`, `sy_stick`, `sy_desklamp`, `sy_computertower`, `churchBell`...

### `impact`

| param | default | meaning |
|---|---|---|
| `material` | `wood` | the struck body: ring time, stiffness, surface hardness |
| `shape` | `bar` | body shape (list above) or a modal table name |
| `size` | 0.3 | m, length or diameter. Bigger = lower. Bars and plates: pitch falls with the square of the size |
| `position` | 0.28 | 0..1 strike point. Bars: 0 = end, 0.5 = middle. Circles: 0 = centre, 1 = rim. Near a held edge = brighter, thinner |
| `position2` | 0.37 | 0..1 second coordinate on rectangles |
| `striker_mass` | 0.02 | kg. Heavier = longer contact = darker, and it can mute the lowest mode |
| `hardness` | 0.6 | 0..1 striker hardness, rubber mallet to steel. Harder = shorter contact = brighter attack |
| `velocity` | 1.5 | m/s. Faster = louder and brighter (the contact gets shorter) |
| `hollow` | 0.0 | 0..1 share of an air-cavity resonance of the object's size |
| `damping` | 2.0 | 1/s external loss: 0 = hanging freely, 30 = lying on a table or held in the hand |
| `bright` | 0.5 | 0 = dark (displacement), 0.5 = velocity, 1 = bright (acceleration) |
| `freq` | 0 | Hz, lowest mode; 0 = computed from size and material |
| `modes` | 24 | number of modes |
| `dur` | 0 | s; 0 = until it has rung out (at most 4 s) |
| `seed` | 0 | variation of the individual mode decays |

A node of a mode at the strike point removes that mode: a bar struck in the middle loses every second
partial; a drum struck dead centre keeps only its symmetric modes.

```json
{"type": "sfx", "kind": "impact", "params": {"material": "glass", "shape": "rect_plate", "size": 0.2, "hardness": 0.9, "velocity": 2}}
{"type": "sfx", "kind": "impact", "params": {"material": "wood", "shape": "bar", "size": 0.35, "hardness": 0.2, "striker_mass": 0.05}}
{"type": "sfx", "kind": "impact", "params": {"material": "steel", "shape": "circ_plate", "size": 0.4, "position": 0.7, "damping": 0, "dur": 3}}
{"type": "sfx", "kind": "impact", "params": {"material": "plastic", "shape": "rect_plate", "size": 0.3, "hollow": 0.6}}
{"type": "sfx", "kind": "impact", "params": {"shape": "sy_vase", "material": "ceramic", "hardness": 0.8}}
```

### `bounce`

| param | default | meaning |
|---|---|---|
| `material` | `steel` | the bouncing object |
| `size` | 0.02 | m, diameter / length of the object. Heavier objects make longer, darker contacts |
| `shape` | `ball` | `ball` (rigid: only the surface rings), or a ringing object: `circ_plate` (lid), `bar` (rod), `rect_plate` |
| `surface` | `wood` | what it lands on. This dominates the material you hear |
| `surface_size` | 0.6 | m, size of the surface plate |
| `first_interval` | 0.3 | s between the first two impacts. It is the drop height: h = g t² / 8 |
| `restitution` | 0.74 | ratio of successive intervals and impact speeds. 0.70-0.76 is a steel ball on wood; 0.4 is dead, 0.9 lively |
| `irregularity` | 0 | 0..1 non-roundness: intervals and hits randomly shorter and weaker, strike point changes per bounce |
| `velocity` | 0 | m/s first impact speed; 0 = derived from the interval |
| `bright`, `seed` | 0.5, 0 | |

The train stops before the intervals reach buzz rate. Every hit lands on bodies that are still ringing
from the previous one, so no two hits are the same.

```json
{"type": "sfx", "kind": "bounce", "params": {"material": "steel", "size": 0.012, "surface": "wood", "first_interval": 0.25}}
{"type": "sfx", "kind": "bounce", "params": {"material": "rubber", "size": 0.06, "surface": "concrete", "restitution": 0.85, "first_interval": 0.5}}
{"type": "sfx", "kind": "bounce", "params": {"shape": "circ_plate", "material": "steel", "size": 0.18, "surface": "stone", "irregularity": 0.7, "restitution": 0.6}}
```

### `drop`

A ball released from `height`: the bounces follow from the height and `restitution`, then it stays in
contact, rolls off at `roll_speed` and slows to rest over `roll` seconds.

| param | default | meaning |
|---|---|---|
| `material`, `size` | `steel`, 0.03 | the ball and its diameter (m) |
| `surface`, `surface_size` | `wood`, 0.8 | the floor / table |
| `height` | 0.25 | m. Impact speed sqrt(2 g h), first interval 2 e sqrt(2h/g) |
| `restitution` | 0.6 | bounce interval ratio |
| `roll` | 1.5 | s of rolling after the last bounce (0 = none) |
| `roll_speed` | 0.4 | m/s when rolling starts |
| `roughness` | 0.5 | 0..1 surface roughness while rolling |
| `irregularity`, `bright`, `seed` | 0, 0.5, 0 | |

```json
{"type": "sfx", "kind": "drop", "params": {"material": "glass", "size": 0.016, "surface": "wood", "height": 0.4, "roll": 2.5}}
{"type": "sfx", "kind": "drop", "params": {"material": "steel", "size": 0.05, "surface": "concrete", "height": 0.1, "restitution": 0.45}}
```

### `smash`

The object hits the floor, rings for an instant and is gone; a 50 ms noise burst marks the rupture;
then each fragment makes its own train of impacts whose intervals *grow* (fragments do not bounce,
they flop), dense at first and thinning quickly. Several independent trains with different rates are
what makes it read as breaking and not as bouncing.

| param | default | meaning |
|---|---|---|
| `material` | `glass` | `glass`, `ceramic`, `ice`, `wood`, `plastic`, `stone`... Metals read as "not breaking" |
| `size` | 0.25 | m. Bigger = lower, heavier, duller fragments; small objects tinkle |
| `shape` | `rect_plate` | shape before breaking |
| `energy` | 1.0 | 0.2..2: harder hit, denser cloud |
| `fragments` | 8 | number of independent fragment trains |
| `dur` | 1.6 | s until the last fragment settles |
| `burst` | 0.5 | 0..1 level of the rupture noise |
| `bright`, `seed` | 0.5, 0 | |

```json
{"type": "sfx", "kind": "smash", "params": {"material": "glass", "size": 0.12, "fragments": 14, "energy": 1.4}}
{"type": "sfx", "kind": "smash", "params": {"material": "ceramic", "shape": "circ_plate", "size": 0.25, "dur": 2.2}}
{"type": "sfx", "kind": "smash", "params": {"material": "ice", "size": 0.4, "fragments": 5, "burst": 0.8, "bright": 0.3}}
```

### `crumple`

A stochastic burst of small impacts. Each event is one facet of the surface buckling in two; the
facets get smaller as the object is crushed, so the clicks drift upward in pitch and get shorter.

| param | default | meaning |
|---|---|---|
| `kind` | `can` | `can`, `paper`, `bottle`, `foil`, `bag` |
| `energy` | 0.5 | 0..1 force of the crushing. 0 = soft, even; 1 = hard, a few big cracks among many small ones |
| `density` | 0.5 | 0..1 events per second = softness of the material (stiff 1/s .. soft 50/s; paper and bags are scaled up) |
| `size` | 0.5 | 0..1 size of the object = energy budget: how long it goes on |
| `dur` | 0 | s cap; 0 = until the budget is spent |
| `bright`, `seed` | 0.5, 0 | |

`can` and `bottle` close progressively (a low-pass sliding from 1400 to 500 Hz as they collapse).

```json
{"type": "sfx", "kind": "crumple", "params": {"kind": "can", "energy": 0.8, "density": 0.7, "size": 0.6}}
{"type": "sfx", "kind": "crumple", "params": {"kind": "paper", "energy": 0.3, "density": 0.8, "dur": 1.5}}
{"type": "sfx", "kind": "crumple", "params": {"kind": "bottle", "energy": 0.6, "density": 0.4, "size": 0.7}}
```

### `roll`

Rolling is a stream of tiny impacts, not friction: the surface profile is smoothed by the ball's own
curvature (a big ball bridges the small pits), and the ball, with its mass, hops on what is left.
Loudness and brightness follow the speed by themselves.

| param | default | meaning |
|---|---|---|
| `dur` | 1.0 | s |
| `speed` | 0.5 | m/s. A number, or a list of breakpoints spread over the sound, e.g. `[1.2, 0.6, 0]` |
| `radius` | 0.03 | m. Bigger = smoother, lower, heavier. Medium and large balls sound most real |
| `roughness` | 0.5 | 0..1 microscopic surface roughness |
| `material` | `steel` | the ball |
| `surface`, `surface_size` | `wood`, 1.0 | the surface (dominates the timbre) and its size in m |
| `joints` | 0 | m, spacing of floor joints / grooves (0 = none): a tick each time, rate follows speed |
| `dents` | 0 | 0..1 a dent on the ball, heard once per revolution |
| `asymmetry` | 0 | 0..1 non-round ball: speed and load wobble at the rotation rate |
| `feedback` | 0.18 | accepted for old specs, ignored |
| `bright`, `seed` | 0.4, 0 | |

```json
{"type": "sfx", "kind": "roll", "params": {"dur": 3, "speed": [1.0, 0.7, 0.3, 0], "radius": 0.03, "surface": "wood", "joints": 0.12}}
{"type": "sfx", "kind": "roll", "params": {"dur": 2, "speed": 0.8, "radius": 0.008, "material": "glass", "surface": "stone", "roughness": 0.8}}
{"type": "sfx", "kind": "roll", "params": {"dur": 4, "speed": [0.2, 1.5, 0.2], "radius": 0.1, "material": "stone", "surface": "concrete", "asymmetry": 0.6, "dents": 0.5}}
```

### `scrape`

The scraper follows a surface profile; the profile is read at the sliding speed and scaled by the
normal force, and that force drives the body. A steady scrape is hard to recognise: give `speed` or
`force` a shape.

| param | default | meaning |
|---|---|---|
| `dur` | 0.8 | s |
| `speed` | 0.3 | m/s, number or list of breakpoints. Faster = brighter (and periodic textures rise in pitch) |
| `force` | 1.0 | 0..1 normal force, number or list |
| `texture` | `""` | `white`, `pink_half`, `pink`, `brown`, `gritty`, `sandpaper`, `grid`, `wood`, `metal`, `plastic`; empty = use `roughness` |
| `roughness` | 0.5 | 0..1 when no texture is named: 1 = white (very rough), 0 = smooth |
| `grain` | 0.1 | mm, size of the surface grain. Smaller = brighter |
| `body` | `rect_plate` | what is scraped: a shape or a modal table |
| `material`, `size` | `wood`, 0.4 | of the scraped body |
| `position`, `bright`, `seed` | 0.3, 0.5, 0 | |

```json
{"type": "sfx", "kind": "scrape", "params": {"dur": 1.2, "speed": [0.05, 0.6, 0.1], "force": [0.4, 1, 0.3], "texture": "wood", "material": "wood"}}
{"type": "sfx", "kind": "scrape", "params": {"body": "sy_sword1", "material": "steel", "texture": "metal", "speed": [0.2, 1.5], "grain": 0.05, "dur": 0.6}}
{"type": "sfx", "kind": "scrape", "params": {"body": "rect_plate", "material": "stone", "size": 1.0, "texture": "gritty", "speed": 0.15, "grain": 0.5, "dur": 2}}
```

### `rumble` and `avalanche`

White noise with a rise / hold / fade envelope rings a cluster of randomly placed low modes. Each
`seed` gives another cluster.

| param | `avalanche` | `rumble` | meaning |
|---|---|---|---|
| `dur` | 10 | 4 | s until all sound has ended |
| `fmin`, `fmax` | 100, 150 | 35, 90 | Hz band of the modes. Raise `fmax` (1000+) for wind-like or eerie tones |
| `damping` | 0.87 | 3.0 | 1/s decay rate of the modes: lower = more tonal, slower to move |
| `modes` | 25 | 16 | number of modes |
| `rise` | 0 (12 %) | 0 (10 %) | s to full level |
| `hold` | 0 (30 %) | 0 (70 %) | s when the fade starts |
| `fade` | 0 (60 %) | 0 (90 %) | s when the drive has stopped; the modes ring on until `dur` |
| `seed` | 0 | 0 | |

```json
{"type": "sfx", "kind": "avalanche", "params": {"dur": 12, "seed": 3}}
{"type": "sfx", "kind": "rumble", "params": {"dur": 6, "fmin": 25, "fmax": 60, "damping": 1.5}}
{"type": "sfx", "kind": "avalanche", "params": {"dur": 8, "fmin": 200, "fmax": 1800, "modes": 40, "damping": 2}}
```

### `modal_engine`

One engine cycle is a force with four strokes: intake (a bell of noise), compression (silence),
combustion (a burst of 1/f noise), exhaust (noise dying fast). It loops at half the crank speed and
shakes a long, low plate.

| param | default | meaning |
|---|---|---|
| `rpm` | 1200 | crank RPM, number or list of breakpoints |
| `dur` | 2.0 | s |
| `cylinders` | 1 | evenly spaced over the cycle |
| `intake`, `combustion`, `exhaust` | 0.5, 1.0, 0.7 | gains of the three strokes (more `intake` = open throttle hiss; more `combustion` = load) |
| `cyl_spread` | 0.35 | 0..1 level differences between cylinders; at 0 a multi-cylinder engine sounds wrong |
| `fan` | 0.15 | level of a pitched background (fan, rotating parts) |
| `body_freq` | 50 | Hz, lowest body resonance. 50 = motorcycle; 300+ with high `damping` = lawn mower, chainsaw |
| `damping` | 5.56 | 1/s external damping of the body |
| `modes` | 13 | body modes |
| `bright`, `seed` | 0.5, 0 | |

```json
{"type": "sfx", "kind": "modal_engine", "params": {"rpm": 1100, "cylinders": 1, "dur": 3}}
{"type": "sfx", "kind": "modal_engine", "params": {"rpm": [1500, 5000, 3000], "cylinders": 4, "cyl_spread": 0.5, "dur": 4}}
{"type": "sfx", "kind": "modal_engine", "params": {"rpm": 3200, "body_freq": 350, "damping": 120, "intake": 0.8, "fan": 0.3}}
```

## Effects (`"fx"`)

### `resonate`

The layer becomes the force on a modal body: a voice through a steel plate, footsteps through a
wooden box, a guitar through a bell. With `mix` 1 only the body is heard.

| param | default | meaning |
|---|---|---|
| `body` | `rect_plate` | a shape or a modal table (`sy_vase`, `sy_wok`, `churchBell`...) |
| `material` | `steel` | ring time (and pitch with `size`) |
| `size` | 0.4 | m; for modal tables 0.3 = as measured, bigger = lower |
| `position` | 0.3 | 0..1 where the force enters |
| `mix` | 1.0 | 0..1 wet (level-matched to the dry signal) |
| `bright` | 0.5 | 0 dark .. 1 bright |
| `modes` | 32 | number of modes |
| `damping` | 0 | 1/s extra external damping |
| `tail` | 1.0 | s of ring-out appended |

```json
{"type": "resonate", "body": "circ_plate", "material": "bronze", "size": 0.5, "mix": 0.7}
{"type": "resonate", "body": "sy_vase", "mix": 0.5, "tail": 0.5}
{"type": "resonate", "body": "rect_membrane", "material": "rubber", "size": 0.3, "bright": 0.2}
```

### `cavity`

The resonances of a small hollow enclosure (a box, a barrel, a helmet, the inside of an object), with
each mode decaying according to the wall absorption.

| param | default | meaning |
|---|---|---|
| `shape` | `box` | `box` (cube) or `sphere` of the same volume (a little brighter, longer) |
| `size` | 0.5 | m, edge of the cube. Lowest resonance 175 / size Hz for the box |
| `wall` | 1.0 | absorption: 1 = wood, below 1 = harder walls (marble) and a longer ring, above 1 = padded |
| `mix` | 0.6 | 0..1 wet |
| `seed` | 0 | which modes are excited |
| `tail` | 0.5 | s of ring-out appended |

```json
{"type": "cavity", "shape": "box", "size": 0.4, "mix": 0.5}
{"type": "cavity", "shape": "sphere", "size": 0.25, "wall": 0.3, "mix": 0.8}
```

## What the main parameters mean physically

- **Material** is a decay law. Every mode loses energy at a rate proportional to its own frequency, so
  each rings for about the same number of cycles: roughly 10 for rubber, 110 for wood, 470 for glass,
  1500-3000 for metals, 5000 for aluminium. High partials therefore always die first, and a larger
  object of the same material rings longer in seconds. On top of that comes one frequency-independent
  loss (`damping`): the mounting, the hand, the table. Damp glass enough and it turns into plastic.
- **Size** scales all mode frequencies together: 1/size² for bars and plates, 1/size for strings,
  membranes and air cavities.
- **Strike position** changes no frequency, only the mix: each mode is as loud as its mode shape is
  large at that point.
- **Hardness and striker mass** set the contact time, roughly (mass / stiffness)^0.4. The contact
  time is a low-pass on the excitation: 0.1 ms excites the whole audio band, 1 ms little above 2 kHz.
  A harder or lighter striker gives a shorter contact and a brighter attack. A striker that is heavy
  compared with the body stays long enough to mute the lowest mode, and a very hard one can chatter
  (several micro-contacts in one hit).
- **Velocity** shortens the contact a little (speed^-0.2) and loses more energy in the contact: the
  rebound is 1 - (2/3) μ v of the impact speed for gentle hits.
- **Restitution** in `bounce` / `drop` is the ratio between one bounce interval and the next, and
  between successive impact speeds.

## Library use

```python
from genny import contact as C

body = C.ShapeBody.from_size("rect_plate", 0.3, "glass", aspect=1.4, position=(0.3, 0.4), n_modes=30)
body.freqs, body.decays, body.gains()                 # Hz, 1/s, heard amplitudes at this strike point
body = body.pruned(keep=12)                           # keep the 12 most audible modes

hit = C.ImpactInteractor(body, mass=0.01, k=1.5e9, alpha=1.5, mu=0.1).strike(velocity=2.0, dur=1.0, sr=44100)
hit.audio, hit.force, hit.contact_time, hit.restitution, hit.contacts()

audio, disp, vel = C.modal_bank(force_signal, body, sr=44100)      # any force; contact-point motion exposed

t, v = C.bounce_schedule(0.3, 2.0, 0.74, dev_t=0.3, dev_v=0.3)     # bouncer
t, v, train = C.break_schedule(1.5, trains=8)                      # dropper
ev = C.crumple_events(0.03, gamma=-1.3, rate=20)                   # t, energy, left, right, cut
noise = C.fractal_noise(44100, beta=1.81)                          # 1/f^beta surface texture
offset = C.rolling_offset(C.rolling_surface(1.0), radius=0.03, dx=1e-4)
C.hc_contact_time(0.01, 1.5e9, 1.5, 0.1, 2.0)                      # contact time of a mass on a rigid wall
C.MATERIALS, C.material_damping(freqs, "wood")
```
