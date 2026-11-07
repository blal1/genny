# Arms: firearms and heavy weapons (`genny/arms.py`, `genny/ordnance.py`)

`gunshot` stays the quick stylised shot. `firearm` is a named weapon in physical units: distance, direction and the
bullet's own crack change what you hear. Parameters: `genny list sfx`.

## Recipes

A burst heard from far away, with the rounds spread by hand:

```json
{"out": "burst_far.wav", "layers": [
  {"type": "sfx", "kind": "firearm", "params": {"weapon": "ak47", "shots": 6, "distance": 300, "azimuth": 10}},
  {"type": "sfx", "kind": "firearm", "at": 1.4, "params": {"weapon": "ak47", "shots": 3, "distance": 300, "azimuth": 10, "seed": 1}}
], "fx": [{"type": "distance", "metres": 50}]}
```

A shot indoors: dry weapon, then the room.

```json
{"out": "pistol_room.wav", "layers": [{"type": "sfx", "kind": "firearm", "params": {"weapon": "pistol_9mm", "distance": 5, "echo": 0}}],
 "fx": [{"type": "room", "preset": "hall"}]}
```

## Levels and what you hear

- `firearm` and `minigun` default to `level="relative"`: weapons keep their measured differences (a .22 is about 30 dB
  under a .50), so turn `normalize` off in a spec that compares them; `level="peak"` puts every file at 0.9.
  `bullet_impact` takes `level` too.
- `firearm`, `minigun` and `bullet_pass` default to `render="heard"`: the shot as a recording or the ear gets it. The
  blast takes Farnell's layers (chirp, barrel, noise burst) at 0.6 of its peak and 3 ms x energy^(1/3) of extra
  weight, goes through a microphone overloaded at 150 dB (tanh, drive 30) and then through the place (`place`, a
  `room` preset at mix 0.35, `street` by default; `none` when the spec adds its own room). Without it a shot played
  back is a click of a few ms. `render="pressure"` is the bare pressure at the ear: the measured Friedlander wave (0.3
  to 2.6 ms of blast by calibre, a 5 ms x energy^(1/3) gas tail), short and dry, for measurements or your own chain.
  `echo` is 0 by default. `pov=true` (the shooter's ears) keeps the 300 ms ringing body, `ring` sets its length,
  `weight` lengthens the blast.
- `gun_mech action=` also plays `reload_musket`, `jam_clear`, `revolver_eject`; `floor`, `mag`, `rounds` change the
  dropped magazine and the casings. `casing_ricochet` is a case bouncing away.
- `ordnance`: `blast` (`kind`) is the bare explosion core; the effect `earplug` is the dulled hearing after a blast
  (`grenade tinnitus=` adds the ring). `explosion explosive=`, `burst=air|surface|rigid|crater`.
  Python: `ordnance.TNT_JWL` (JWL constants of TNT: A 371.2 GPa, B 3.231 GPa, R1 4.15, R2 0.95, ω 0.30, ρ 1630 kg/m³,
  D 6930 m/s, P_CJ 21 GPa), `jwl_pressure(v, e)` (pressure of the detonation products at relative volume v) and
  `charge_radius(w_kg)` (radius of a bare spherical charge, (3W / 4πρ)^(1/3)).
- Melee and energy weapons live in the `melee` and `energy_arms` modules (see `catalog.md`).

## What is measured and what is not

- Measured: muzzle peaks and directivity of the main calibres, the N-wave formula, suppressor losses, case modes against
  Farnell's range, blast levels and durations (Kinney & Graham), fills and speeds.
- Estimates (from physics or known orders of magnitude, listed in `research/SOURCES.md` as `connaissance`): nearly all
  handling times, bullet lengths, several calibre levels, the timing inside every heavy-weapon sequence, shell whistles.
- Measured or derived: underwater peak, bubble period and the energy each bubble keeps (Cole 1948, Tables 7.3, 8.1, 8.2);
  case modes include the finite-length (Love) shell modes; off-axis muzzle levels follow the measured AR15 shape
  (-11 dB at 90 degrees, -27 dB at 165 degrees for A = 30); a passing bullet slows with range by its drag length
  2m / (rho Cd A), Cd 0.3; the minigun's motor is about 100 dB and its bolts 120 dB peak at 1 m.
- Aircraft (`genny.aero`): `jet_afterburner` and `flyby_jet` are shaped by Tam's two similarity spectra (large scale
  downstream, fine scale to the side), with the peak Strouhal per angle from Tam et al. 2008; `space_rocket`'s low band
  follows the SP-8072 normalised rocket spectrum (peak Strouhal 0.02).
- None of it has been compared by ear with recordings. Sources: `reports/sons/armes_feu.md`, `rechargement_douilles.md`,
  `armes_lourdes_autres.md` in the parent repository.
