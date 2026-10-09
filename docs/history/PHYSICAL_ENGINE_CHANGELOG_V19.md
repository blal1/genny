# Genny 0.19.0 — Shared String Migration and Wound Strings

## New physical construction model
- `WoundStringPhysicalProperties` models a load-bearing core plus helical winding.
- Winding mass includes helix path length and material density.
- Effective bending rigidity combines the core with a partially coupled annular winding.
- New string materials: nickel, copper, phosphor bronze.
- Physical quantities exposed: outer radius, helix factor, linear density, tension, characteristic impedance, effective EI, inharmonicity B and longitudinal mode.

## Common renderer migration
- `DispersiveString` now renders both plucked and bowed excitation.
- `render_bow()` uses the same two-polarisation string state plus stateful stick/slip hysteresis.
- `BowedString` now uses the shared SI-property string core. Cello defaults to a wound construction.
- `PluckedString.note()` now uses the common physical renderer; mandolin uses two detuned physical strings.
- `PluckedString.wound_note()` exposes the wound-string path directly.

## Public instrument migration
The public guitar, harp, mandolin, steel guitar, koto, sitar, upright bass, finger bass, violin and cello paths now source their string motion from the shared physical core. Body/pickup/jawari radiation remain instrument-specific post-processing.

Public musical wrappers retain small pitch anchors where needed so the registry contract remains tuned. Direct `DispersiveString` rendering is not pitch-anchored and exposes its physical dispersion directly.

## Validation
- Targeted v0.6→v0.19 regression suite: 76 passed.
- New construction/render tests run at 22050, 44100 and 48000 Hz.
- Affected public wrappers were separately checked for short-note audibility, tuning and >5 kHz energy.
