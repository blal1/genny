# matter — sounds by state of matter

`genny/matter.py` registers 33 sound effects for liquids, gases, fire, electricity and ice. Solids
(impacts, rolling, breaking, friction) live in `contact` and `friction`.

Every effect takes `seed` (same parameters and seed give identical samples), works at any sample
rate (rendered at 48 kHz, resampled at the output), is mono, DC-free and starts and ends at zero.

## Pick a sound by physics

| State | Phenomenon | sfx | Governing formula | Source |
|---|---|---|---|---|
| Liquid | One air bubble ringing | `bubbles` | Minnaert `f = (1/2πr)·√(3γP₀/ρ)`, so `f·r = 3.28 m/s`; decay `β = ω₀δ/√(δ²+4)`, `δ = δ_rad + δ_vis + δ_th` | Harmonic Fluids App. A |
| Liquid | Bubble nearing the surface ("bloop") | `bubbles`, `drip`, `babble` | `Δω = 72.95·ω_d·exp(−1.24(φ/φ₀−1))·Δφ` per sample, `φ₀` = −8…−25 mm; rise speed `(2/3)√(g·r)` | Harmonic Fluids eq. 6; Farnell |
| Liquid | Drop into a pool | `drip`, `drops` | 4–5 bubbles per drop, 500 Hz–4 kHz; drop timing is a relaxation oscillator (near-periodic) | Harmonic Fluids Table 1 |
| Liquid | Object into water | `splash` | 127 bubbles concentrated in the event, 300 Hz–6 kHz; crown droplets return after `t = 2√(2h/g)` | Harmonic Fluids Table 1 |
| Liquid | Filling a vessel | `pour` | Cup: quarter wave `f = c / (4(L_air + 0.61a))`, odd harmonics. Bottle: Helmholtz `f = (c/2π)·√(A/(V·L′))`. Both rise as `L_air` falls. 1580 bubbles/s | Farnell Practical 14 |
| Liquid | Brook over a step | `babble` | 3100 bubbles/s, 300 Hz–5 kHz, two-band radii, chirping | Harmonic Fluids Table 1 |
| Liquid | Large wobbling bubbles | `gurgle`, `drain` | Shape modes `f_n² = (n−1)(n+1)(n+2)σ/(4π²ρr³)` radiating at `2f_n` | Moss, Sounding Liquids |
| Liquid | Breaking waves | `surf` | Bubble rate follows the crest (`Γ = u²κ`), radii `p(r) ∝ r^−α`, α 1.5–3.3 | Moss |
| Liquid | Many falls at once | `waterfall`, `underwater_ambience`, `cave_drips` | Dense bubbles tend to white noise; far away the detail is replaced by noise | Farnell |
| Liquid → gas | Boiling | `boil` | Bubbles in bursts whose rate decays (relaxation); rumble peaks before the boil | Farnell Practical 12 (law of the rumble unsourced) |
| Liquid → gas | Water on a hot surface | `sizzle` | Poisson micro-explosions, log-normal sizes | unsourced model |
| Gas out of liquid | Carbonation | `fizz` | Poisson rate `r(t) = r₀·2^(−t/t½)`; each bubble by the Minnaert and damping laws | rate unsourced |
| Gas | Jet from a hole | `steam`, `air_leak`, `spray` | Strouhal peak `f_p = St·U/D`, `St = 0.2`; power `∝ U⁸` (`∝ U⁶` near solid edges) | Selfridge (St); power law unsourced in the findings |
| Gas | Jet locking to a cavity | `kettle` | Edge tone `f_e = St·U/gap` pulled onto the cavity mode when within 30 % | unsourced model |
| Gas | Flapping neck | `balloon` | Relaxation oscillator; Helmholtz cavity rising as volume falls | Farnell (mechanism) |
| Gas | Wind on objects | `gust` | Wires: tone `∝ U/d`, amplitude `∝ U²`. Openings howl only inside a speed window. Leaves lag by seconds | Farnell Practical 18 |
| Plasma (flame) | Combustion roar | `flame`, `fire`, `match` | `p(t) = dI/dt` (I = flux through the flame front); above 180 Hz power `∝ f^−α`, α 2.5–3.5, modulated by `|p|` | Animating Fire with Sound |
| Plasma (flame) | Crackle, hiss | `fire`, `flame` | Crackle: noise × (linear decay 0–30 ms)² through a band-pass. Hiss: noise × (slow noise)⁴ | Farnell Practical 11 |
| Plasma (electric) | Spark | `spark`, `tesla` | Sweep 7 kHz → 20 Hz in 10 ms + noise × (decay)⁴ through the spark formant (480, 720, 4600, 7200 Hz) | Farnell Practical 16 |
| Plasma (electric) | Arc, mains | `arc`, `mains_hum`, `neon` | Conduction each half cycle: buzz at `2·f_mains`; hum = `f_mains` plus harmonics, 2nd strongest | Farnell Practical 16 |
| Plasma (electric) | Lightning | `lightning` | Crack within 500 m; thunder layers darken with distance | Farnell Practical 17 |
| Solid ↔ liquid | Ice sheet cracking | `ice_crack`, `freeze` | Flexural dispersion `ω = k²·a`, `a = h·√(E/(12ρ(1−ν²)))`: arrival frequency `f(t) = d²/(8π·a·t²)` — highs first | textbook plate theory (not in the findings) |
| Solid in liquid | Ice in a glass | `ice_cubes` | Contact pulses into a glass modal body; liquid loads and damps it | modal tables; loading unsourced |

Handy numbers: bubble radius 10 mm rings at 330 Hz for 325 ms; 5 mm at 660 Hz for 143 ms; 1 mm at
3.3 kHz for 19 ms; 0.5 mm at 6.6 kHz for 8 ms. A 15 cm glass sounds 500 Hz empty and 2.4 kHz at
90 % full. 5 cm ice chirps down from 3 kHz; 30 cm ice from 500 Hz.

## Liquids

| sfx | Parameters (default) | Physical meaning |
|---|---|---|
| `drip` | `size` (0.4), `bubbles` (4.7), `secondary` (0.5), `click` (0.15), `seed` | `size` 0..1 sets the mean bubble radius 0.9–4.5 mm (3.6 kHz–730 Hz). `secondary` is the chance of smaller rebound drops. `click` is the level of the impact itself, which is quiet in reality. |
| `drops` | `dur` (4), `rate` (2), `rate_end` (-1), `jitter` (0.15), `size` (0.4), `seed` | `rate` drops per second (0.05–40). `rate_end` ramps it. `jitter` 0 = dripping tap, 1 = random. |
| `splash` | `size` (0.5), `dur` (1.5), `droplets` (6), `seed` | `size` sets the cavity bubble (6–20 mm) and bubble count; `droplets` fall back 0.2–0.6 s later. |
| `pour` | `dur` (4), `flow` (1), `fill_start` (0.05), `fill_end` (0.9), `height` (0.15 m), `radius` (0.035 m), `neck` (0), `neck_len` (0.05 m), `material` (glass), `resonance` (0.6), `seed` | `flow` 1 = 1580 bubbles/s. Fill levels are fractions of `height`. `neck` 0 = open cup; a neck radius makes it a bottle (much lower pitch). `material` glass, metal, ceramic, plastic, paper sets how sharp the air resonance is. `resonance` is how clearly the rising pitch reads. |
| `babble` | `dur` (4), `flow` (1), `size` (1), `turbulence` (0.5), `seed` | `flow` 1 = 3100 bubbles/s. `size` scales every radius: larger sounds like deeper, slower water. |
| `bubbles` | `dur` (2), `rate` (12), `size` (2 mm), `spread` (0.4), `cluster` (0.5), `underwater` (0), `rise` (1), `seed` | `size` is the mean radius in mm. `underwater` 1 = deep, muffled, steady pitch. `rise` multiplies the rise speed, so the chirp. |
| `boil` | `dur` (4), `intensity` (0.8), `ramp` (0), `size` (0.5), `seed` | `intensity` 0.2 simmer, 0.6 loudest rumble, 1 rolling boil. `ramp` 1 heats from cold to `intensity` over the take. |
| `sizzle` | `dur` (3), `intensity` (0.7), `spit` (6), `decay` (0), `hiss` (0.4), `bright` (0.5), `seed` | `intensity` density of micro-explosions. `spit` larger pops per second. `decay` seconds to boil away. |
| `fizz` | `dur` (3), `rate` (900), `half_life` (1.5), `size` (0.7 mm), `bright` (0.35), `seed` | Bubble rate halves every `half_life` seconds. Smaller `size` is higher and shorter. |
| `gurgle` | `dur` (2.5), `rate` (3.5), `size` (0.5), `seed` | Glugs per second; `size` 0..1 = 6–20 mm bubbles (550–165 Hz). |
| `drain` | `dur` (4), `size` (0.5), `seed` | Basin size lowers the suction pitch. |
| `surf` | `dur` (12), `distance` (120 m), `period_min` (8), `period_max` (13), `bubble_rate` (3000), `seed` | Under 10 s renders one wave. |
| `cave_drips` | `dur` (6), `sources` (5), `wet` (0.6), `seed` | Dripping points and cave reverb level. |
| `waterfall` | `dur` (4), `height` (6 m), `distance` (8 m), `seed` | Height adds rumble; distance removes bubble detail and top end. |
| `underwater_ambience` | `dur` (6), `depth` (5 m), `bubbles` (4), `seed` | Deeper is darker. `bubbles` per second, 0 for none. |

## Gases

| sfx | Parameters (default) | Physical meaning |
|---|---|---|
| `steam` | `dur` (2.5), `pressure` (0.6), `orifice` (4 mm), `wet` (0.4), `tone` (9000), `seed` | `pressure` 0..1 = exit speed 40–343 m/s. Hiss pitch is `0.2·U/D`: doubling the hole halves it. `wet` adds sputter. |
| `air_leak` | `dur` (2), `pressure` (0.5), `orifice` (2 mm), `burst` (0), `whistle` (0), `tone` (10000), `seed` | `burst` > 0 turns it into a pneumatic release whose pressure decays with that time constant. `whistle` adds a narrow tone at the Strouhal frequency. |
| `spray` | `dur` (0.8), `pressure` (0.7), `nozzle` (0.5 mm), `can` (900), `tone` (9000), `seed` | Aerosol can with valve ticks and a hollow can resonance. |
| `kettle` | `dur` (5), `tone` (1800), `onset` (2.5), `breath` (0.4), `rumble` (0.3), `seed` | `tone` is the whistle cavity; `onset` the seconds for steam pressure to build. |
| `balloon` | `dur` (2), `size` (0.5), `flutter` (0.7), `seed` | `flutter` 0 = hiss, 1 = raspberry. |
| `gust` | `dur` (4), `through` (gap), `strength` (0.5), `tone` (1), `seed` | `through` gap, wire, leaves, canyon, doorway. Howls sound only for wind speeds about 0.25–0.6: a stronger gust howls on the way up and down and goes quiet at its peak. `tone` scales the wire pitch. |

## Fire and electricity

| sfx | Parameters (default) | Physical meaning |
|---|---|---|
| `flame` | `kind` (campfire), `dur` (3), `rate` (1), `seed` | `kind`: candle (silent until a draught), torch (ruffles with each swing, `rate` swings/s), campfire, bonfire, gas_burner (steady, α 2.5), blowtorch (strong jet), fireball (one thump and roar), flamethrower (on, hold, burn-off). |
| `fire` | `dur` (4), `size` (0.4), `crackle` (12), `wet` (0.2), `wind` (0.2), `seed` | `size` 0..1 = 1–4 independent flame fronts and more roar. `crackle` per second. `wet` wood hisses more and pops lower. `wind` ruffles the flame: a steady flame is quiet, a ruffled one roars. |
| `match` | `dur` (1.6), `strike` (0.12), `rough` (0.6), `seed` | Scrape, flare, small flame. |
| `spark` | `size` (0.3), `tone` (1), `seed` | `size` 0 = static tick, 1 = heavy contactor. `tone` < 1 sounds bigger or enclosed. |
| `arc` | `dur` (2), `mains` (50), `stability` (0.7), `tone` (1), `seed` | Buzz at twice `mains`. `stability` is the fraction of time the arc holds. |
| `mains_hum` | `dur` (3), `mains` (50), `harmonics` (0.3), `beat` (0.4), `resonance` (0.4), `seed` | Fundamental is exactly `mains`. `harmonics` 0 = soft hum, 1 = transformer buzz. `beat` is the slow throb in Hz. |
| `tesla` | `dur` (1), `rate` (120), `chaos` (0.4), `sweep` (0), `tone` (1), `seed` | `rate` discharges per second (the pitch of the buzz); `sweep` octaves of glide. |
| `neon` | `dur` (3), `mains` (50), `flicker` (0.5), `seed` | `flicker` 0 = steady tube, 1 = failing tube. |
| `lightning` | `dur` (6), `distance` (150 m), `delay` (0), `seed` | Under 500 m there is a sharp crack; farther is rumble only. `delay` 1 inserts the real travel time. |

## Ice

| sfx | Parameters (default) | Physical meaning |
|---|---|---|
| `ice_crack` | `dur` (1.5), `thickness` (0.05 m), `distance` (60 m), `cracks` (3), `brittle` (0.6), `seed` | Thin ice gives a high laser-like "pew", thick ice a low boom. Farther cracks chirp longer. `brittle` is the air-borne snap that follows. |
| `ice_cubes` | `dur` (1.5), `cubes` (3), `glass` (1), `liquid` (0.6), `energy` (0.7), `seed` | `glass` scales the glass pitch. `liquid` lowers and damps it. `energy` is how hard it is swirled. |
| `freeze` | `dur` (4), `rate` (8), `creak` (0.5), `pings` (2), `seed` | Ticks accelerate to `rate` per second; `creak` is stick-slip groaning; `pings` are distant ice chirps. |

## Recipes

Filling a glass, then a bottle:

```json
{"out": "pour_glass.wav", "layers": [{"type": "sfx", "kind": "pour",
  "params": {"dur": 5, "fill_start": 0.0, "fill_end": 0.95, "height": 0.14, "radius": 0.035}}]}
```
```json
{"out": "pour_bottle.wav", "layers": [{"type": "sfx", "kind": "pour",
  "params": {"dur": 6, "height": 0.25, "radius": 0.04, "neck": 0.01, "neck_len": 0.07, "flow": 0.6}}]}
```

Thaw (melt drips speeding up over cracking ice):

```json
{"out": "thaw.wav", "layers": [
  {"type": "sfx", "kind": "drops", "params": {"dur": 8, "rate": 0.4, "rate_end": 5, "jitter": 0.5, "size": 0.3}},
  {"type": "sfx", "kind": "freeze", "params": {"dur": 8, "rate": 3, "creak": 0.3, "pings": 3}, "gain": 0.5}]}
```

Kettle from cold to whistle:

```json
{"out": "kettle.wav", "layers": [
  {"type": "sfx", "kind": "boil", "params": {"dur": 8, "ramp": 1, "intensity": 0.9, "size": 0.2}, "gain": 0.6},
  {"type": "sfx", "kind": "kettle", "params": {"dur": 4, "onset": 2.5, "tone": 2000}, "at": 5}]}
```

Campfire in the rain, with a far strike:

```json
{"out": "storm_camp.wav", "duration": 10, "layers": [
  {"type": "sfx", "kind": "fire", "params": {"dur": 10, "size": 0.5, "crackle": 18, "wet": 0.7, "wind": 0.5}},
  {"type": "sfx", "kind": "gust", "params": {"through": "leaves", "dur": 8, "strength": 0.8}, "gain": 0.5},
  {"type": "sfx", "kind": "lightning", "params": {"distance": 1500, "dur": 7}, "at": 3, "gain": 0.8}]}
```

Faulty machine room (evaporation hiss, hum, arcing):

```json
{"out": "machine_room.wav", "layers": [
  {"type": "sfx", "kind": "mains_hum", "params": {"dur": 6, "mains": 60, "harmonics": 0.6}, "gain": 0.5},
  {"type": "sfx", "kind": "steam", "params": {"dur": 6, "pressure": 0.3, "orifice": 8, "wet": 0.2}, "gain": 0.3},
  {"type": "sfx", "kind": "arc", "params": {"dur": 1.2, "mains": 60, "stability": 0.4}, "at": 2.5, "gain": 0.6}]}
```

Other transitions come from parameters: water hitting a pan and boiling away is
`sizzle` with `decay` 1.5; a fresh soda is `fizz` with `rate` 3000, `half_life` 0.8; a pneumatic
door is `air_leak` with `burst` 0.12, `orifice` 8.

## What is and is not sourced

Sourced and checked by `tests/test_matter.py`: the bubble frequency, damping and pitch-rise laws;
bubble counts and frequency ranges per scene; the vessel resonance formulas; the fire chain
(`p = dI/dt`, 30 Hz high-pass, 180 Hz hand-over, `f^−α`); Strouhal scaling; the Farnell crackle,
hiss, snap, spark formant, hum, wind and thunder recipes.

Assumptions, marked `# UNSOURCED` in the code: the flame flux `I(t)` (the paper takes it from a
fluid simulation; here it is smooth noise shaped per flame kind), the jet power law, the kettle,
balloon, drain, sizzle and boil-rumble models, every ice constant, bubble start depths and the
per-bubble gain spread.

Two deviations from the findings' reference code, both measured: the bubble is integrated with an
exact amplitude/phase recursion instead of the midpoint rule (which at 48 kHz was 2.9 % sharp at
3.3 kHz and halved the damping), and the fire extension gain is capped at three times its median
(the printed per-window formula produced isolated loud windows).
