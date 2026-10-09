# Genny 0.7 — membranes, plates, contact and rotor acoustics

Genny 0.7 continues the migration from one-off DSP recipes to shared physical models.

## New shared physical models

- `MembraneDrum`: reduced circular-membrane modal model using Bessel-mode ratios, strike position, contact hardness, cavity/shell coupling and optional harmonic loading for tabla-like heads.
- `CymbalPlate`: dense inharmonic thin-plate renderer with velocity-dependent nonlinear shimmer.
- `SurfaceContact`: spatial-profile scrape and rolling contact. Ridge/texture spacing is converted into temporal frequency by `f = v / lambda`; rolling additionally uses `f_rot = v/(2*pi*R)`.
- `Rotor`: generic fan/propeller rotor with `f_BPF = blades * RPM / 60`, tip-speed-dependent turbulence and source directivity.
- `JetEngine`: reduced multi-spool turbofan model combining rotor tones, correlated combustion rumble and broadband jet mixing noise.
- `Helicopter`: main rotor + tail rotor + turbine + blade-vortex interaction/slap.
- `Destruction`: high-level time-varying modal fracture facade with a safe proxy fallback.

## Legacy API migrated

Drums/percussion now routed through the shared physical engines:

- conga, bongo, djembe, tabla
- tom, taiko, timpani
- crash, ride, hi-hat/open hi-hat
- guiro (spatial-profile scrape)

Plucked instruments migrated further:

- electric guitar
- koto
- sitar (physical string plus buzzing-bridge nonlinearity)
- upright bass
- finger bass

New/updated SFX:

- `door` (stick-slip/creak)
- `roll`
- `scrape`
- `fan`
- `propeller`
- `jet_engine`
- `helicopter`
- `shatter`

## Validation

The v0.7 targeted suite covers 22.05, 44.1 and 48 kHz render paths and the migrated public registries. At packaging time:

```
16 passed
```

for the v0.7/v0.6/unified/physics/spec targeted suite.
