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
- (Superseded for the shot itself by "The v4 shot" below: the tanh chain and Farnell texture described here are gone; `place` and `echo` still act on the mix.)
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

## Added in 1.3.0

- `space_rocket`: tabulated directivity (maximum +0.2 dB near 45-47 degrees, -5.7 dB under 20, -7.7 dB at 90, -13 dB at
  120, -20.6 dB at 180, relative to 50 degrees; `aero.ROCKET_DI`, `aero.rocket_directivity_db(angle)`).
## The v4 shot: weapon presets, recorder, cases (families W1-W4)

`firearm` and `minigun` were rebuilt on physical units, so the sound is not the chain described above under "Levels and
what you hear" for `render="heard"`: the shot is now the Friedlander blast of the cartridge in pascals
(`ballistics.blast_at`, with a precursor, muzzle brake, flash, a gas jet, the suppressor tone and `barrel_ring`), then air
absorption, the ground, the `place` and a `recorder`. There is no fixed tanh, no 3 ms Farnell texture and no echo inside the
shot (`echo` acts on the finished mix). A .22 comes out 30 dB under a 5.56 with `level="relative"` (pressure over the
peak of a .50 at 3 m); `level="peak"` puts every file at 0.9.

| Parameter | Meaning |
|---|---|
| `weapon` | a name (33 shipped) or an inline dict `{"extends": "ar15", "barrel": 0.4}` |
| `recorder` | `none`, `field` (default), `camera`, `phone`: what recorded the shot; `clip_db` sets its clipping |
| `barrel` | barrel length in m, overrides the preset; `barrel_ring` rings the barrel (default 0) |
| `flash`, `ground`, `treeline` | muzzle flash, ground type, forest edge in the path |
| `k_t0` | short (Beck, Kingery) or long (Maher) positive duration of the blast |
| `brass` | the empty case ringing on the floor (`brass.render_case`) |
| `suppressor` | loss D of the cartridge, T0 times 10^(D/40) (10 ms at most), the chamber's tone; the first shot is 3 dB louder |

- `bullet_pass`: the same near-field law (b^-0.87), `recorder`, `level` (`peak` by default; `relative` puts a 5.56 at 3 m
  35 dB under the .50). The N-wave starts at an emission point (`ballistics.n_wave`); buckshot is nine waves +-0.2 ms.
- `bullet_impact`: modes scale with the impulse (`law="impulse"`: m v (1 + e), or the Thomson / Lambert-Jonas
  perforation for sheet metal, wood and glass; `law="energy"` the old square root), debris with the square root of the
  dissipated energy, flesh with the impulse; `gain_db`; the plate size no longer depends on the bullet. A .50 on steel at
  1 m is the level reference (0.85).
- `casing_ricochet` calls `brass.render_case`: same calibre names plus `ring`, `lively`, `rattle_law` (`v` or `sqrt_v`),
  `spin_depth`, `wobble`, `air`, `wall_scale`; `.45`, `20mm` and `105mm` have a local geometry. `casing` `caliber=` also accepts any cartridge name of the `cartridge` family (`9x19`, `762x51`, `357`, a user
  cartridge), resolved by `case_of`; the case of a user cartridge comes from `case_of` + `case_dim`. `casing` without a
  calibre is the 5.56 case (`caliber="generic"` is the old model), `size` scales the geometry. The case rolls 3-9 s.
- `gun_mech finish= spread=`; `firearm treeline=`; the minigun's `whine_ratio`.
- `arms.ricochet` is frozen and renders as before (its `water_*` drift is fixed: `tube_bubble(delta_th=None)` and
  `particles.DELTA_TH_FROZEN`).

## The venue field `env` (weapon W5)

A shot heard in the open is mostly the field of the place (ground, relief, vegetation, facades), not the 0.5 ms blast. The
v4 shot had none, so every weapon was a click (E50 1-8 ms, crest about 25 dB, where recordings give E50 40-55 ms and crest
9.5-11.4 dB). `firearm`, `minigun`, `bullet_pass` and `bullet_impact` now take `env=` (not `ricochet`, not the cases):

- `env`: a name of the `weapon_env` presets (`open_field` by default, `forest`, `urban`, `range`, `indoor`), an inline dict
  (`{"extends": "urban", "t60": 2.0}`) or a user file; `anechoic` or `none` gives the dry render of before (bit for bit for
  the unsuppressed weapons; with `place="street"` the whole v4 render). The response is `arms.env_ir(env, r, seed, azimuth)`:
  the direct sound, a dense early body (`body_*`), a ground and relief bass (`boom_*`), discrete echoes and a late tail in
  two bands (`t60` under 2 kHz, `t60_hf` above). Energies are given in their band relative to the direct sound at 10 m and
  30 degrees; they grow as r^2 and as the directivity moves the direct sound away (azimuth), capped at +20 dB. It is applied
  before the recorder, which clips the mix like a real microphone. `bullet_pass` computes the field at max(miss, 10 m).
- `place` is `none` by default now (it was `street`): the venue is `env`; `place` stays available.
- `level="relative"` with a field refers to the 5.56 at 10 m at 0.5 (`REF_PEAK_ENV`) instead of the .50 at 3 m at 0.7 (about
  +20 dB in peak; the differences between weapons are unchanged); `bullet_impact` with its field is capped at 0.89.
- Suppressors: gas hiss leaving the suppressor (`_can_hiss`, 1.5 kHz band, 12 ms); the MP5SD goes from a 4 ms dull thump to a
  "pfft", and the dry render of suppressed weapons changes too.
- Reference ranges (B&G and Zenodo recordings, otherwise `connaissance`): see `genny/research/ref_armes/`. `forest`,
  `urban`, `range` and `indoor` are set by knowledge, without a recording.
- `level` stays a `firearm` / `minigun` / `bullet_impact` / `bullet_pass` parameter of its own. To mix weapons with the other
  sfx use the generic `out_level` (`level` as a layer key, `genny sfx --level`): `"match"` puts a weapon at the level of the
  family table `_sfx_levels.py`; the default keeps the absolute pressure.

## Adding a weapon or a cartridge

A weapon is a TOML table, not code. Put `weapon.toml` (family `weapon`) in the user folder (`genny presets path`) and
write:

```toml
schema = "genny.preset/1"
family = "weapon"

[my_carbine]
extends = "ar15"
barrel = 0.4          # m
rate = 11.0           # shots per second
```

Fields: `cart` (the cartridge), `barrel` (m), `rate` (shots/s), `action`, `body`, `muzzle`, `wood`, `clip`, `cadence`,
`lock`, `suppressed`. A cartridge (family `cartridge`) has `d_mm` and the powder and ballistics fields; `legacy_*`
fields are optional. A cartridge you add works in `bullet_pass caliber=`, `bullet_impact caliber=`, `firearm` and
`casing`. A case is three tables: `case` (geometry), `case_dim` (length, diameter, wall in mm) and `case_of`
(`case = "<case>"`, which case a cartridge uses). `genny presets list weapon|cartridge|case|case_dim|case_extra|case_of|gun_legacy`
shows what is shipped; see `docs/presets.md`.

Other changes: `physics.blast_wave(source="tnt"|"point")` (Table XI A or B of Kinney; `altitude` through `pressure_kpa`
and `temp_c`), `Explosion(secondary=)` and `explosion(model="farnell"|"physical", secondary=)`: the second Friedlander
shock below Z 0.4 m/kg^(1/3). `Rotor.tip_doppler`, `MembraneDrum.air_shift`, `rocket_motor(fall)`,
`ground_wave(vp, vs, vr)`, aero rotors `xv15`, `fl282`, `twin`.
