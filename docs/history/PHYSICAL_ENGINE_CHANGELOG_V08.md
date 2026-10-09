# Genny 0.8.0 — Physical graph, drums, machines and blast

## Physical ports and graph

Added `genny.ports` with SI-unit aware conjugate ports:

- mechanical: force [N] / velocity [m/s]
- acoustic: pressure [Pa] / volume flow [m^3/s]
- `PassiveCoupler` for bounded bidirectional scattering
- `PhysicalGraph` for explicit connection/diagnostics
- reduced mechanical/acoustic impedance helpers

This is the first infrastructure step toward connecting exciters and resonators
bidirectionally rather than treating all synthesis as a one-way DSP chain.

## New physical models

- `KickDrum`: circular membrane, air coupling, beater compression and contact
  transient.
- `SnareDrum`: membrane/cavity plus acceleration-driven stochastic wire bank.
- `AdvancedRollingContact`: surface profile and resonator-to-contact feedback.
- `ElectricMotor`: rotor, commutation, imbalance force, bearing texture and
  housing modes.
- `GearTrain`: tooth-mesh frequency, shaft sidebands, backlash impacts and
  housing modes.
- `Explosion`: near-field N-wave, distance-dependent smoothing/delay, expansion
  rumble, sparse debris and simple enclosure reflections.

## Public registry migration

The legacy public names `kick`, `snare`, `roll` and `explosion` now point to the
new physical wrappers. New SFX are `electric_motor` and `gears`.

## Compatibility

Existing parameters remain accepted where possible. `explosion(..., boom=x)`
is mapped to physical blast energy for backward compatibility.

## Validation

Targeted suite: 20 tests passed, including v0.4–v0.8 physical models, public
wrappers and the historical spec renderer. New v0.8 models are tested at
22.05/44.1/48 kHz where applicable.
