# Foley: objects, weapons, machines, signals, sci-fi (`genny/foley.py`)

Struck, shaken, rubbed and fired things, built from physical models: measured modal objects, particle (PhISEM) shakers, stick-slip friction, Farnell's click and gun layers, finite-element bells, banded waveguides. Every sound is mono, deterministic (`seed` gives another take) and renders at any sample rate.

Parts of the module are ports of the Klang `procedural/` code (© 2025 Chris Nash, Klang Open License 1.0, see `KLANG_LICENSE.txt`): a product that ships these sounds interactively carries the Klang attribution.

## What the main parameters mean physically

- **`contact_ms` (hardness).** How long the two things touch. 0.05 ms is steel on steel (bright, rings everything), 1 to 3 ms is wood or a knuckle, 10 to 50 ms is a soft thud where the pulse itself removes the highs. Louder hits are shorter, so they are brighter.
- **`size`.** Scales every resonance: 2 is twice as big and an octave lower. Use it before reaching for a filter.
- **`position`.** Where the object is struck, across the measured strike points (the bottle has 34, the bells 7). It changes which modes ring, not their pitch.
- **`damping`, `decay`.** How fast the modes die: a hand on the blade, a muted bell.
- **`object`.** The thing itself, from measurements: `stick` (wood), `wok` (iron), `sword`, `sword2`, `bottle` (glass), `vase` (ceramic), `metal_box` (sheet metal), `lamp`, `bell`, `church_bell`, and the wooden bodies `block`, `box`, `crate`, `tub`.
- **`energy` (shakers).** How hard the container is shaken. More energy is louder and keeps the particles colliding for longer.
- **`force` / `level` (friction).** The force on a creaking joint. Nothing slips below 0.3; above it the slips come faster as the force grows, so the creak rises in pitch.
- **`distance` (gunshot).** Metres to the shooter. The sound falls as 1/r, the air removes the highs, and the muzzle blast arrives `d (1/c - 1/v)` after the bullet's supersonic crack (0.26 s at 150 m for a rifle).
- **`mount` (switches).** The panel a switch sits on colours it more than the switch does: `plastic`, `wood`, `metal`.

## Recipes

A door knock, then the door creaking open:

```json
{"out": "door_knock_open.wav", "layers": [
  {"type": "sfx", "kind": "knock", "params": {"count": 3, "pace": 4.5, "material": "wood", "size": 1.4}},
  {"type": "sfx", "kind": "creak", "at": 1.2, "params": {"material": "door", "force": "pull", "level": 0.75, "dur": 1.4}}
], "fx": [{"type": "reverb", "mix": 0.12, "size": 0.3}]}
```

A distant rifle shot with its casing landing nearby:

```json
{"out": "rifle_far.wav", "layers": [
  {"type": "sfx", "kind": "gunshot", "params": {"kind": "rifle", "distance": 200, "echo": 0.6}},
  {"type": "sfx", "kind": "casing", "at": 0.9, "gain": 0.3, "params": {"count": 1}}
]}
```

Sword fight: swing, clash, cut.

```json
{"out": "sword_fight.wav", "layers": [
  {"type": "sfx", "kind": "sword", "params": {"blade": "sword", "speed": 1.1}},
  {"type": "sfx", "kind": "clash", "at": 0.28, "params": {"force": 0.9, "damping": 2.5}},
  {"type": "sfx", "kind": "slash", "at": 0.9, "params": {"target": "armour", "force": 0.8}}
]}
```

Dialling a number on an old phone line, then it rings:

```json
{"out": "phone_call.wav", "layers": [
  {"type": "sfx", "kind": "phone_tone", "params": {"kind": "dial", "region": "uk", "dur": 1.0}},
  {"type": "sfx", "kind": "dtmf", "at": 1.1, "params": {"digits": "02079460018", "tone_ms": 90, "gap_ms": 70}},
  {"type": "sfx", "kind": "phone_tone", "at": 3.2, "params": {"kind": "ringback", "region": "uk", "dur": 3.0}}
], "fx": [{"type": "telephone"}]}
```

A grandfather clock striking three, and a tolling bell (the written note is the bell's hum; the strike note is heard about an octave above):

```json
[{"out": "clock_three.wav", "layers": [{"type": "sfx", "kind": "clock", "params": {"size": "grandfather", "chime": 3, "dur": 8}}]},
 {"out": "bell_toll.wav", "bpm": 30, "layers": [
   {"type": "seq", "inst": "church_bell", "step": 1, "steps": "C3 C3 C3", "params": {"type": "english", "decay": 12, "position": 0.1}}],
  "fx": [{"type": "reverb", "mix": 0.3, "size": 0.85}]}]
```

An endless riser as a seamless loop (`dur` is a whole number of cycles of `1 / rate` seconds, so no crossfade is needed):

```json
{"out": "endless_rise.wav", "loop": 0.0, "trim": false, "layers": [{"type": "sfx", "kind": "shepard", "params": {"direction": "up", "rate": 0.25, "dur": 8}}]}
```

## Sound effects (`{"type": "sfx", "kind": ...}`)

## Objects and materials

### `hit`

Physically resonant impact on a measured object: contact duration is the hardness, the modes are the object.

| param | default | meaning |
|---|---|---|
| `dur` | `0.25` | s |
| `tone` | `200` | Hz/body scale (200 = as measured) |
| `crunch` | `0.5` | 0..1 hardness: shorter contact, more re-contacts |
| `material` | `steel` | steel / glass / wood / stone / aluminum / rubber: picks the object when object=auto |
| `object` | `stick` | auto / stick / wok / sword / sword2 / bottle / vase / metal_box / lamp / bell / church_bell / crate / box / block / tub / stone |
| `position` | `0.0` | 0..1 strike point across the measured points |
| `contact_ms` | `0.0` | contact duration ms, 0.05 (steel on steel) .. 50 (dull thud); 0 = from crunch |
| `size` | `1.0` | object size: 2 = twice as big, an octave lower |
| `seed` | `0` | variation |

### `clang`

Metal struck by metal: measured iron wok, sword, sheet-metal box or lamp with re-contact chatter.

| param | default | meaning |
|---|---|---|
| `object` | `wok` | wok / sword / sword2 / metal_box / lamp / bell |
| `size` | `1.0` | size (bigger = lower) |
| `force` | `0.8` | 0..1 |
| `damping` | `1.5` | decay-rate scale: 1 free ringing .. 15 hand-muted |
| `position` | `0.0` | 0..1 strike point |
| `dur` | `2.0` | max s |
| `seed` | `0` | variation |

### `clink`

Glass or small-metal clink: the measured bottle (34 strike points), a vase, or a coin tinkle.

| param | default | meaning |
|---|---|---|
| `object` | `bottle` | bottle / vase / coin / bell |
| `size` | `1.25` | size (bigger = lower; 1.25 = a wine glass from the bottle table) |
| `force` | `0.7` | 0..1 |
| `position` | `0.5` | 0..1 strike point |
| `seed` | `0` | variation |

### `thud`

Dull soft impact with no ring: a long contact on a box or a soft mass landing.

| param | default | meaning |
|---|---|---|
| `body` | `box` | box (sheet-metal/hollow) / crate (wood) / sack (felt contact on a big body) |
| `contact_ms` | `15.0` | 5..50 ms: longer = softer, lower (35 = felt rather than heard) |
| `size` | `1.0` | size (bigger = lower) |
| `force` | `0.8` | 0..1 |
| `seed` | `0` | variation |

### `knock`

Knocking on a door or panel: a sequence of knuckle contacts on a measured body.

| param | default | meaning |
|---|---|---|
| `count` | `3` | knocks |
| `pace` | `4.5` | knocks per second |
| `material` | `wood` | wood / hollow / metal / glass |
| `size` | `1.0` | panel size (bigger = lower) |
| `force` | `0.7` | 0..1 |
| `seed` | `0` | variation |

### `shake`

Any of the 25 PhISEM particle presets shaken: coins in a mug, gravel, sleigh bells, bamboo chimes, sandpaper, a soda can, a ratchet...

| param | default | meaning |
|---|---|---|
| `preset` | `maraca` | maraca / cabasa / sekere / tambourine / sleigh_bells / bamboo_chimes / sandpaper / soda_can / sticks / crunch / rocks / pebbles / mug / pennies / nickels / dimes / coins / francs / pesos / guiro / ratchet / water_drops / angklung / gravel / fine_gravel |
| `energy` | `0.7` | 0..1 shake energy: louder and longer-lived |
| `rate` | `4.0` | shakes per second (0 = one shake that dies away); ratchets: strokes per second |
| `dur` | `1.5` | s |
| `size` | `1.0` | container/object size: bigger = lower resonances |
| `objects` | `0` | particle count (0 = the preset's own) |
| `seed` | `0` | variation |

### `chain`

A chain moved by a body: link contacts fired at a rate that follows the momentum, then settling.

| param | default | meaning |
|---|---|---|
| `dur` | `1.5` | s |
| `pushes` | `2` | movements |
| `weight` | `0.5` | 0 light jewellery chain .. 1 heavy iron |
| `density` | `40.0` | link contacts per second at full speed |
| `settle` | `1` | 1 = come to rest with the three-bounce pattern |
| `seed` | `0` | variation |

### `keys`

A bunch of keys jingling (Farnell tambourine jingles: four Q217 bands ring-modulated at 6.3 kHz).

| param | default | meaning |
|---|---|---|
| `hits` | `3` | jingles |
| `decay` | `40.0` | ms base decay of the brightest band |
| `spread` | `10.0` | ms between the two contacts of one jingle / 3 |
| `seed` | `0` | variation |

### `cloth`

Cloth: rustle of a moving garment, a tear, or a flag/cape flapping.

| param | default | meaning |
|---|---|---|
| `kind` | `rustle` | rustle / tear / flap |
| `dur` | `1.0` | s |
| `rate` | `3.0` | movements (rustle) or flaps per second |
| `weight` | `0.5` | 0 silk .. 1 canvas/sacking: lower, rougher |
| `seed` | `0` | variation |

### `creak`

Stick-slip creak of wood, a door, a floorboard, rope, leather or iron under a force curve.

| param | default | meaning |
|---|---|---|
| `material` | `wood` | wood / door / floorboard / rope / leather / iron |
| `force` | `ramp` | force curve: ramp / pull / sway / constant |
| `level` | `0.8` | 0.3..1 peak force: more force = faster slips, higher creak (nothing slips below 0.3) |
| `dur` | `1.2` | s |
| `size` | `1.0` | object size (bigger = lower formants) |
| `seed` | `0` | variation |

### `zipper`

Zipper: slider teeth clicking at a rate that follows the pull speed.

| param | default | meaning |
|---|---|---|
| `dur` | `0.5` | s |
| `length` | `0.25` | m of zip |
| `pitch_mm` | `2.5` | tooth spacing mm |
| `direction` | `up` | up / down |
| `seed` | `0` | variation |

### `velcro`

Velcro peeled apart: a dense crackle of hooks letting go.

| param | default | meaning |
|---|---|---|
| `dur` | `0.6` | s |
| `speed` | `0.6` | 0..1 peel speed |
| `seed` | `0` | variation |

### `paper`

Paper: rustle, tear, or a page turn.

| param | default | meaning |
|---|---|---|
| `kind` | `page` | rustle / tear / page |
| `dur` | `0.5` | s (rustle, tear) |
| `weight` | `0.5` | 0 tissue .. 1 heavy paper/card |
| `seed` | `0` | variation |

### `coin_spin`

A coin spinning down on a table: the wobble accelerates, then stops dead.

| param | default | meaning |
|---|---|---|
| `coin` | `quarter` | penny / nickel / dime / quarter |
| `dur` | `1.8` | s until it lies flat |
| `start` | `9.0` | wobble rate at the start, Hz |
| `seed` | `0` | variation |

### `dice`

Dice thrown on a table: each die bounces with shrinking intervals and rattles to rest.

| param | default | meaning |
|---|---|---|
| `count` | `2` | dice |
| `surface` | `wood` | wood / felt |
| `force` | `0.7` | 0..1 throw strength |
| `seed` | `0` | variation |

### `cork_pop`

A cork pulled from a bottle: the pressure release rings the bottle's Helmholtz resonance and the neck.

| param | default | meaning |
|---|---|---|
| `volume_ml` | `750.0` | air volume of the bottle, ml |
| `neck_mm` | `18.5` | neck inner diameter, mm |
| `neck_len_mm` | `80.0` | neck length, mm |
| `fizz` | `0.0` | 0..1 sparkling hiss after the pop |
| `seed` | `0` | variation |

### `boing`

Twanged ruler / springy bar: clamped-bar modes with the pitch dropping in the attack.

| param | default | meaning |
|---|---|---|
| `freq` | `426` | Hz or note name (the book's figure value) |
| `decay` | `1.5` | s |
| `bend` | `6.0` | Hz the pitch starts above the final pitch |
| `bright` | `1.0` | 0.3 dull .. 3 all overtones |
| `seed` | `0` | variation |

### `ratchet`

Ratchet / pawl clicks (Farnell click-factory ratchet cluster) at a steady or slowing rate.

| param | default | meaning |
|---|---|---|
| `clicks` | `10` | count |
| `rate` | `30.0` | clicks per second |
| `slow` | `0.0` | 0..1: the rate falls towards the end |
| `size` | `1.0` | mechanism size (bigger = lower) |
| `mount` | `metal` | plastic / wood / metal |
| `seed` | `0` | variation |

### `drawer`

A wooden drawer sliding open or shut, ending on its stop.

| param | default | meaning |
|---|---|---|
| `direction` | `close` | open / close |
| `dur` | `0.5` | s of travel |
| `size` | `1.0` | drawer size (bigger = lower) |
| `force` | `0.6` | 0..1: speed of the slide and weight of the stop |
| `seed` | `0` | variation |

### `book_drop`

A book dropped flat: the cover's soft slap, the pages, and one small rebound.

| param | default | meaning |
|---|---|---|
| `size` | `1.0` | book size (bigger = lower, heavier) |
| `height` | `0.5` | 0..1 drop height / force |
| `seed` | `0` | variation |

## Weapons

### `gunshot`

Gunshot: detonation chirp, muzzle burst and gas through the weapon body, mechanical action, supersonic crack, distance and echoes.

| param | default | meaning |
|---|---|---|
| `kind` | `rifle` | rifle / pistol / smg / shotgun / silenced / cannon |
| `distance` | `10.0` | m to the shooter: 1/r, air absorption, and the blast lags the crack by d(1/c - 1/v) |
| `shots` | `1` | rounds |
| `rate` | `0.0` | rounds per second (0 = the weapon's own) |
| `action` | `0.6` | 0..1 mechanical recycling (lost beyond 100 m) |
| `crack` | `1.0` | 0..1 supersonic N-wave of the passing bullet (none for subsonic rounds) |
| `echo` | `0.3` | 0..1 street / treeline echoes |
| `seed` | `0` | variation |

### `reload`

Weapon reload / cocking: metal slide sweep, ratchet click clusters and the end-stop clunk through the weapon body.

| param | default | meaning |
|---|---|---|
| `kind` | `rifle` | rifle / pistol / smg / shotgun / silenced / cannon |
| `speed` | `1.0` | 0.5 slow and deliberate .. 2 fast |
| `slide` | `1.0` | 0..1 slide friction level |
| `seed` | `0` | variation |

### `casing`

Spent shell casings (or coins, small brass) bouncing on a hard floor.

| param | default | meaning |
|---|---|---|
| `count` | `1` | casings |
| `size` | `1.0` | bigger = lower (1 = rifle brass at 4.5-5.5 kHz) |
| `spread` | `0.25` | s over which they land |
| `seed` | `0` | variation |

### `sword`

A blade or club swung through the air: Aeolian tones of eight sources along the blade (Selfridge presets).

| param | default | meaning |
|---|---|---|
| `blade` | `sword` | sword / sword2 / golf_club / bat (thin sword, thicker sword, golf club, baseball bat) |
| `speed` | `1.0` | tip-speed scale (1 = the preset, e.g. 36 m/s) |
| `thickness` | `1.0` | diameter scale (thicker = lower) |
| `model` | `loq` | loq (St 0.2, Q 10: rated best)  /  patch (Fey Strouhal, bandwidth Q) |
| `seed` | `0` | variation |

### `clash`

Blade on blade: both measured swords ring from one shared steel-on-steel contact.

| param | default | meaning |
|---|---|---|
| `force` | `0.8` | 0..1 |
| `size` | `1.0` | blade size (bigger = lower) |
| `damping` | `2.0` | decay-rate scale (hands damp the blades) |
| `seed` | `0` | variation |

### `stab`

A blade point driven into flesh, armour or wood (no swing).

| param | default | meaning |
|---|---|---|
| `target` | `flesh` | flesh / armour / wood |
| `force` | `0.7` | 0..1 |
| `seed` | `0` | variation |

### `slash`

A sword cut: the swing through the air, then the edge landing on flesh, armour or wood.

| param | default | meaning |
|---|---|---|
| `swing` | `0.5` | 0..1 level of the air swoosh before the cut |
| `target` | `flesh` | flesh / armour / wood |
| `force` | `0.7` | 0..1 |
| `seed` | `0` | variation |

### `bow_shot`

Bow and arrow: the draw creaks, the string twangs, the shaft whistles away and thunks into the target.

| param | default | meaning |
|---|---|---|
| `draw` | `0.35` | s of draw creak (0 = none) |
| `distance` | `20.0` | m to the target: the thunk comes distance / 70 m/s later (0 = no impact) |
| `pitch` | `120.0` | bowstring Hz (90-160) |
| `target` | `wood` | wood / flesh / armour |
| `seed` | `0` | variation |

### `whip`

Whip crack: the thong accelerates past the speed of sound and its tip makes a small sonic boom.

| param | default | meaning |
|---|---|---|
| `dur` | `0.35` | s of swing before the crack |
| `size` | `1.0` | whip length scale: longer N-wave, lower whoosh |
| `echo` | `0.2` | 0..1 outdoor echoes |
| `seed` | `0` | variation |

### `shield_bash`

A shield slammed into something: a heavy long contact on a wooden or iron shield with its boss and rattle.

| param | default | meaning |
|---|---|---|
| `material` | `wood` | wood (round shield with an iron boss) / metal |
| `force` | `0.8` | 0..1 |
| `size` | `1.0` | shield size (bigger = lower) |
| `seed` | `0` | variation |

## Machines

### `clock`

Clockwork tick-tock: escapement micro-clicks in a resonant case, from a wristwatch to a grandfather clock, with an optional chime.

| param | default | meaning |
|---|---|---|
| `size` | `mantle` | watch / mantle / grandfather |
| `rate` | `0.0` | ticks per second (0 = the size's own: 5, 4, 1) |
| `dur` | `4.0` | s |
| `pendulum` | `0.0` | pendulum length in m; > 0 sets the rate from T = 2 pi sqrt(L/g) (0.994 m = one tick per second) |
| `chime` | `0` | bell strokes struck at the start (0 = none) |
| `seed` | `0` | variation |

### `switch`

Mechanical switch: every click is one mechanical event, coloured by what it is mounted on.

| param | default | meaning |
|---|---|---|
| `kind` | `toggle` | toggle / rocker / push / relay / slide / rotary |
| `mount` | `plastic` | plastic / wood / metal: the panel it sits on |
| `state` | `on` | on / off (latching push button sounds different each way) |
| `seed` | `0` | variation |

### `button`

Momentary push button: one sprung click with a tiny metallic ping, and a softer release.

| param | default | meaning |
|---|---|---|
| `release` | `1` | 1 = add the release click |
| `hold` | `0.12` | s the button is held |
| `mount` | `plastic` | plastic / wood / metal |
| `seed` | `0` | variation |

### `keyboard_typing`

Typing on a computer keyboard: plastic key clicks with bottom-out and release, human timing, the odd space bar.

| param | default | meaning |
|---|---|---|
| `keys` | `12` | key presses |
| `speed` | `6.0` | keys per second |
| `seed` | `0` | variation |

### `mouse_click`

Computer mouse click: the microswitch going down and coming back up.

| param | default | meaning |
|---|---|---|
| `double` | `0` | 1 = double click |
| `hold` | `0.09` | s between press and release |
| `seed` | `0` | variation |

### `camera_shutter`

SLR camera shutter: mirror slap, shutter opening and closing after the exposure, optional film wind.

| param | default | meaning |
|---|---|---|
| `exposure` | `0.0167` | s the shutter stays open (1/60 = 0.0167) |
| `wind` | `0.0` | s of motor wind afterwards (0 = none) |
| `seed` | `0` | variation |

### `servo`

Servo / stepper moves: a small motor whining up to speed and stopping, in a sequence of moves.

| param | default | meaning |
|---|---|---|
| `moves` | `2` | moves |
| `dur` | `0.35` | s per move |
| `speed` | `180.0` | rotor Hz at full speed (pitch) |
| `gap` | `0.12` | s between moves |
| `seed` | `0` | variation |

### `printer`

Desktop printer: the carriage stepping back and forth with the head buzzing, and a paper feed after each line.

| param | default | meaning |
|---|---|---|
| `lines` | `3` | printed lines |
| `speed` | `1.0` | 0.5 slow .. 2 fast |
| `seed` | `0` | variation |

### `drill`

Electric drill: motor spin-up, load-dependent speed, and the bit cutting.

| param | default | meaning |
|---|---|---|
| `dur` | `2.0` | s |
| `rpm` | `9000.0` | motor RPM unloaded |
| `load` | `0.4` | 0..1 pressure on the bit: slower, rougher |
| `material` | `wood` | none / wood / metal: what the bit cuts |
| `startup` | `0.25` | s to reach speed |
| `seed` | `0` | variation |

### `saw`

Hand saw cutting wood: push and pull strokes, the teeth passing at a rate that follows the stroke speed.

| param | default | meaning |
|---|---|---|
| `strokes` | `4` | push-pull cycles |
| `rate` | `1.6` | cycles per second |
| `length` | `0.35` | m of blade travel per stroke |
| `tpi` | `8.0` | teeth per inch |
| `seed` | `0` | variation |

### `hammering`

Hammering a nail into wood: the nail's ring climbs with every blow as it sinks.

| param | default | meaning |
|---|---|---|
| `strikes` | `12` | blows |
| `pace` | `2.2` | blows per second |
| `start_freq` | `200.0` | Hz of the nail before the first blow |
| `sink` | `0.6` | 0..0.9 fraction of the nail driven in by the last blow |
| `seed` | `0` | variation |

### `ratchet_wrench`

Socket wrench: clicking back on the return swing, quiet on the drive stroke.

| param | default | meaning |
|---|---|---|
| `strokes` | `3` | swings |
| `rate` | `1.6` | swings per second |
| `clicks` | `9` | pawl clicks per return swing |
| `size` | `1.0` | tool size (bigger = lower) |
| `seed` | `0` | variation |

### `piston`

Pneumatic piston strokes: a hiss of air, the rod travelling, and the end-stop clunk.

| param | default | meaning |
|---|---|---|
| `cycles` | `3` | out-and-back cycles |
| `rate` | `1.2` | cycles per second |
| `pressure` | `0.6` | 0..1 air pressure: louder, brighter hiss |
| `seed` | `0` | variation |

### `steam_engine`

Steam locomotive: exhaust chuffs, four to a wheel turn, with the valve gear clanking, speeding up or slowing down.

| param | default | meaning |
|---|---|---|
| `dur` | `4.0` | s |
| `speed` | `3.0` | chuffs per second at the start |
| `accel` | `0.0` | chuffs per second gained per second (negative = slowing) |
| `cylinders` | `2` | double-acting cylinders: 2 x cylinders chuffs per wheel turn |
| `seed` | `0` | variation |

### `train`

Riding a train: wheels clacking over rail joints in the rhythm set by the bogie spacing and the speed, with an optional horn.

| param | default | meaning |
|---|---|---|
| `dur` | `6.0` | s |
| `speed` | `60.0` | km/h |
| `rail_len` | `25.0` | m between rail joints |
| `wheelbase` | `2.5` | m between the two axles of a bogie |
| `bogie_spacing` | `16.0` | m between the bogies of a car |
| `horn` | `0.0` | s of horn at the start (0 = none) |
| `seed` | `0` | variation |

### `sonar`

Active sonar ping with its echo coming back from a target.

| param | default | meaning |
|---|---|---|
| `freq` | `1500` | Hz or note name |
| `range_m` | `750.0` | m to the target: the echo returns 2 x range / c later (0 = no echo) |
| `c` | `1500.0` | sound speed m/s (sea water ~1500) |
| `ring` | `0.6` | s decay of the ping in the water |
| `seed` | `0` | variation |

## Signals

### `dtmf`

Telephone keypad tones (DTMF): each key is one row tone plus one column tone.

| param | default | meaning |
|---|---|---|
| `digits` | `5551234` | keys to dial: 0-9 * # A-D (anything else is a pause of one tone + gap) |
| `tone_ms` | `80.0` | ms per tone (the standard asks for >= 40) |
| `gap_ms` | `80.0` | ms of silence between tones |
| `twist` | `0.0` | dB the column (high) tone is above the row tone |

### `phone_tone`

Telephone network tones: dial tone, busy signal or ringback, with each region's frequencies and cadence.

| param | default | meaning |
|---|---|---|
| `kind` | `dial` | dial / busy / ringback |
| `region` | `us` | us / uk / eu |
| `dur` | `0.0` | s (0 = 2 s of dial tone, two busy cycles, or one ringback cycle) |

### `phone_bell`

Old electromechanical telephone ringer: a hammer rattling between two slightly detuned bells inside a Bakelite case.

| param | default | meaning |
|---|---|---|
| `freq` | `650` | Hz or note name of the first bell |
| `detune` | `3.0` | Hz the second bell is higher (beating) |
| `decay` | `2.0` | s base decay of a bell |
| `rate` | `28.6` | hammer strikes per second (35 ms apart in the patch) |
| `bursts` | `2` | rings |
| `ring` | `0.875` | s per ring (25 strikes) |
| `gap` | `0.325` | s between rings |
| `bright` | `1.0` | 0.3 dull .. 3 all hammer partials |
| `seed` | `0` | variation |

### `siren_wail`

Emergency siren heard in a street: police wail, two-tone ambulance or a wind-up air-raid siren, with building echoes.

| param | default | meaning |
|---|---|---|
| `kind` | `police` | police / ambulance / air_raid |
| `dur` | `5.0` | s |
| `rate` | `0.0` | cycles per second (0 = the kind's own) |
| `echo` | `0.5` | 0..1 building echoes (delays 165, 121, 33 ms) |
| `bright` | `1.0` | 0.5 distant .. 2 close |
| `seed` | `0` | variation |

### `car_horn`

Car horn: two diaphragm horns a third apart.

| param | default | meaning |
|---|---|---|
| `dur` | `0.7` | s |
| `freq` | `420` | Hz or note name of the low horn |
| `freq2` | `500` | Hz or note name of the high horn |
| `bright` | `1.0` | 0.5 muffled .. 2 harsh |

### `bike_bell`

Bicycle bell: ding-ding, the striker catching the dome on the way out and back.

| param | default | meaning |
|---|---|---|
| `freq` | `1300` | Hz or note name |
| `strikes` | `2` | dings |
| `gap` | `0.16` | s between dings |
| `decay` | `1.2` | s |
| `bright` | `1.0` | 0.3 dull .. 3 bright |
| `seed` | `0` | variation |

### `buzzer`

Electromechanical buzzer (door entry, game-show wrong answer): an armature rattling at twice the mains frequency against a metal plate.

| param | default | meaning |
|---|---|---|
| `dur` | `0.6` | s |
| `freq` | `100` | Hz or note name (100 = 50 Hz mains, 120 = 60 Hz) |
| `harsh` | `0.6` | 0..1 plate rattle |

## Sci-fi and game

### `shepard`

Shepard tone: a pitch that seems to rise (or fall) forever. Loops exactly when dur is a whole number of cycles.

| param | default | meaning |
|---|---|---|
| `direction` | `up` | up / down |
| `rate` | `0.125` | octaves per second (one cycle = 1 / rate s) |
| `dur` | `8.0` | s |
| `freq` | `C4` | centre of the spectral bell: Hz or note name |
| `voices` | `10` | octave-spaced sines |
| `dropoff` | `10.0` | width of the Gaussian spectral envelope (higher = narrower) |

### `transporter`

Sci-fi transporter beam: a shimmering whole-tone cluster of narrow noise bands that swells and dissolves upward.

| param | default | meaning |
|---|---|---|
| `dur` | `3.0` | s |
| `freq` | `A4` | lowest band: Hz or note name |
| `bands` | `10` | whole-tone steps stacked |
| `seed` | `0` | variation |

### `robot_babble`

Droid chatter: a string of random FM bleeps, whistles and warbles.

| param | default | meaning |
|---|---|---|
| `dur` | `1.5` | s |
| `rate` | `11.0` | syllables per second |
| `low` | `300.0` | Hz lowest carrier |
| `high` | `2800.0` | Hz highest carrier |
| `seed` | `0` | variation |

### `red_alert`

Starship red alert: a rising klaxon whoop repeated.

| param | default | meaning |
|---|---|---|
| `whoops` | `3` | repeats |
| `low` | `440` | start: Hz or note name |
| `high` | `880` | end: Hz or note name |
| `rate` | `1.3` | whoops per second |

### `dark_drone`

Dark resonant drone: pink noise ringing three octave-stacked resonators inside a very long reverb.

| param | default | meaning |
|---|---|---|
| `dur` | `6.0` | s |
| `note` | `52.0` | MIDI note of the lowest resonator (52 = E3) |
| `ring` | `0.2` | s ring time of the resonators |
| `reverb_time` | `100.0` | s reverb T60 |
| `seed` | `0` | variation |

### `force_field`

Energy barrier hum: a beating mains-like buzz with a slow comb shimmer.

| param | default | meaning |
|---|---|---|
| `dur` | `3.0` | s |
| `freq` | `100` | Hz or note name |
| `shimmer` | `0.5` | 0..1 moving comb / high band |
| `seed` | `0` | variation |

### `teleport`

Teleport zap: a fast endless-rise glissando with a noise rush, ending in a thump (or reversed for arriving).

| param | default | meaning |
|---|---|---|
| `dur` | `0.8` | s |
| `direction` | `out` | out (dematerialise)  /  in (arrive) |
| `seed` | `0` | variation |

### `energy_blade`

Energy sword: a steady beating hum that swells and bends in pitch as the blade swings past.

| param | default | meaning |
|---|---|---|
| `dur` | `2.0` | s |
| `freq` | `92` | hum: Hz or note name |
| `swings` | `1` | swings |
| `speed` | `25.0` | m/s peak blade speed: Doppler shift c / (c - v) |
| `buzz` | `0.4` | 0..1 upper buzz |
| `seed` | `0` | variation |

### `warp`

Warp / hyperspace jump: engines winding up faster and faster, then the jump.

| param | default | meaning |
|---|---|---|
| `dur` | `3.0` | s |
| `start` | `40.0` | Hz |
| `end` | `1400.0` | Hz at the jump |
| `seed` | `0` | variation |

### `charge_up`

Weapon / ability charging: a tone climbing with a tremolo that speeds up until it is ready.

| param | default | meaning |
|---|---|---|
| `dur` | `1.5` | s |
| `start` | `120.0` | Hz |
| `end` | `1600.0` | Hz |
| `ready` | `1` | 1 = end on a short ready blip |

### `scanner`

Scanner / tricorder: a repeating swept warble.

| param | default | meaning |
|---|---|---|
| `dur` | `2.0` | s |
| `rate` | `2.0` | sweeps per second |
| `low` | `600.0` | Hz |
| `high` | `1800.0` | Hz |
| `warble` | `18.0` | Hz of the fast flutter |

## Drums (`{"type": "drum", "kind": ...}` or `pattern`)

### `cabasa`

Cabasa (PhISEM: 512 beads on a 3 kHz gourd): a short dry scratch.

| param | default | meaning |
|---|---|---|
| `size` | `1.0` | instrument size (bigger = lower) |
| `seed` | `0` | variation |

### `sekere`

Sekere / shekere (PhISEM: 64 beads, 5.5 kHz): a bright net-on-gourd shake.

| param | default | meaning |
|---|---|---|
| `size` | `1.0` | instrument size (bigger = lower) |
| `seed` | `0` | variation |

### `maraca`

Maraca (PhISEM: 25 beans in a 3.2 kHz shell), one shake.

| param | default | meaning |
|---|---|---|
| `size` | `1.0` | instrument size (bigger = lower) |
| `seed` | `0` | variation |

### `bamboo_chimes`

Bamboo wind chimes (PhISEM: sparse knocks on three tubes near 2.8 kHz), one stir.

| param | default | meaning |
|---|---|---|
| `size` | `1.0` | instrument size (bigger = lower) |
| `seed` | `0` | variation |

### `sandpaper`

Sandpaper blocks (PhISEM: 128 grains, 4.5 kHz), one rub.

| param | default | meaning |
|---|---|---|
| `size` | `1.0` | instrument size (bigger = lower) |
| `seed` | `0` | variation |

### `sleigh_bells`

Sleigh bells (PhISEM: 32 pellets ringing five bell modes 2.5-9.8 kHz), one shake that rings on.

| param | default | meaning |
|---|---|---|
| `size` | `1.0` | instrument size (bigger = lower) |
| `seed` | `0` | variation |

### `agogo`

Agogo bell (STK ModalBar preset: modes 1, 4.08, 6.669 and a fixed 3725 Hz).

| param | default | meaning |
|---|---|---|
| `tune` | `660.0` | Hz of the bell (the high bell is about a third above the low one) |
| `hardness` | `0.6` | 0 soft beater .. 1 hard |
| `decay` | `1.0` | s |

### `anvil`

Anvil struck with a hammer: measured steel (sword1.sy) under a 0.15 ms contact with re-contact chatter.

| param | default | meaning |
|---|---|---|
| `tune` | `1.0` | pitch scale (2 = an octave up) |
| `decay` | `1.0` | ring-time scale |
| `seed` | `0` | variation |

### `brake_drum`

Brake drum: the measured iron wok struck hard, a dry clanging pitch.

| param | default | meaning |
|---|---|---|
| `tune` | `1.0` | pitch scale (2 = an octave up) |
| `decay` | `1.0` | ring-time scale |
| `seed` | `0` | variation |

### `wind_chimes`

Metal wind chimes stirred: several tuned tubes struck at random, ringing over each other.

| param | default | meaning |
|---|---|---|
| `tune` | `880.0` | Hz of the lowest tube |
| `tubes` | `7` | strikes in the stir |
| `spread` | `1.0` | s over which they are struck |
| `decay` | `3.0` | s ring |
| `seed` | `0` | variation |

## Instruments (`synth` / `seq` layers)

### `church_bell` — mallet, range C3-C5

Cast bell from a finite-element model (50 modes with split doublets): church, English, French, German, Russian or standard profile. The written note is the hum, the lowest and longest partial; the strike note is heard about an octave above.

| param | default | meaning |
|---|---|---|
| `type` | `church` | church / english / french / german / russian / standard |
| `position` | `0.0` | 0 lip .. 1 crown: where the clapper strikes (7 measured points) |
| `decay` | `8.0` | s T60 of the hum |
| `bright` | `1.0` | 0.4 soft clapper, distant .. 2.5 hard and close |
| `tune` | `hum` | hum / prime: which partial plays the written note |

### `handbell` — mallet, range G4-G6

Small bright handbell (Farnell's bell partial groups): hammer clang that settles to a pure tone, with a second bell a hair sharp for the beat.

| param | default | meaning |
|---|---|---|
| `decay` | `2.5` | s base decay |
| `detune` | `0.3` | % the second bell is sharp (0 = one bell) |
| `bright` | `1.0` | 0.3 dull .. 3 all hammer partials |

### `china_bell` — mallet, range C3-C5

Chinese temple bell (Farnell's chinatown bell): six inharmonic partials with very long quartic decays.

| param | default | meaning |
|---|---|---|
| `decay` | `1.0` | ring-time scale (1 = 15-22 s partial decays) |
| `bright` | `1.0` | 0.4 dark .. 2.5 bright |

### `glass_harmonica` — bowed, range C4-G6

Glass harmonica: a rubbed glass bowl (banded waveguide, modes 1, 2.32, 4.25, 6.63, 9.38) that swells slowly and rings on.

| param | default | meaning |
|---|---|---|
| `mode` | `bow` | bow (rubbed rim)  /  strike (tapped glass) |
| `pressure` | `0.5` | 0..1 finger pressure: more upper modes |
| `ring` | `1.5` | s of ring kept after the note |

### `bowed_bowl` — bowed, range C3-C5

Tibetan singing bowl rubbed with a wooden puja (banded waveguide: twelve modes in beating pairs), or struck.

| param | default | meaning |
|---|---|---|
| `mode` | `bow` | bow (rubbed rim)  /  strike |
| `pressure` | `0.5` | 0..1 puja pressure: more upper modes |
| `ring` | `3.0` | s of ring kept after the note |

### `hurdy_gurdy` — bowed, range G2-G5

Hurdy-gurdy: a string bowed by a rosined wheel over a buzzing bridge (the trompette), with an optional drone an octave below.

| param | default | meaning |
|---|---|---|
| `buzz` | `0.6` | 0.2 constant rattle .. 1.5 clean: the buzzing bridge's threshold |
| `pressure` | `0.8` | 0.3..1 wheel pressure |
| `drone` | `0.0` | 0..1 level of the bourdon an octave below |
| `attack` | `0.06` | s for the wheel to reach speed |

### `wind_chime` — mallet, range C5-C7

One tube of a metal wind chime: a struck free-hanging bar with its inharmonic overtones ringing for seconds.

| param | default | meaning |
|---|---|---|
| `decay` | `3.0` | s T60 of the fundamental |
| `bright` | `1.0` | 0.4 soft striker .. 2.5 hard |

## Notes and limits

- **`hit` is compatible.** With only its old parameters (`dur`, `tone`, `crunch`, `material`) it renders the same sound as before. `object` selects a measured object; `object: "auto"` lets `material` choose one (steel = sword, glass = bottle, wood = stick, stone = lowered ceramic, aluminum = sheet-metal box, rubber = a long soft contact).
- **`shake` presets**: `maraca cabasa sekere tambourine sleigh_bells bamboo_chimes sandpaper soda_can sticks crunch rocks pebbles mug pennies nickels dimes coins francs pesos guiro ratchet water_drops angklung gravel fine_gravel`. `guiro` and `ratchet` are scraped, not shaken: `rate` is strokes per second and `energy` the scrape speed. Shakers are bright by nature (most energy above 5 kHz): keep them low in a mix.
- **Bells are tuned on a stated partial.** `church_bell` puts the hum (the lowest, longest partial, a split pair a few cents apart) on the written note; `tune: "prime"` puts the next pair there instead. `china_bell`, `handbell` and `wind_chime` put their ratio-1 partial on the note. `church_bell` and `china_bell` are inharmonic, so a pitch detector that assumes harmonics reads them off the note; the partial itself is exact.
- **`glass_harmonica` and `bowed_bowl` swell slowly**, like the real things: for short notes use `mode: "strike"`. Their fundamental cannot go above 1568 Hz (G6).
- **`hurdy_gurdy`** needs about 0.1 s to speak; notes shorter than that are held for 0.12 s.
- **Gunshots are loud by nature and are already saturated** (the model includes the clipping of the recording chain). Do not add `distortion`; use `distance` and `echo` for placement and a `reverb` for rooms.
- **Not sourced from measurements** (built from the nearest physical model and marked `UNSOURCED` in the code): cloth, paper, velcro, zipper, steam and pistons, the siren oscillators, car horn, and all of the sci-fi set except `shepard` and `dark_drone`. Phone tones and DTMF use the published telephone standards.
