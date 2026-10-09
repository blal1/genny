# Genny 0.13.0 — Multi-hole winds, coupled strings and cavities

## Added

- `DynamicToneHole`: continuously variable side-hole opening with reduced acoustic impedance.
- `MultiHoleBore`: chain of bidirectional bore segments connected through passive three-port scattering junctions.
- `MultiHoleClarinet`: reed-driven multi-hole bore with static or sample-by-sample dynamic fingering.
- `MultiHoleFlute`: jet-driven multi-hole bore with hydrodynamic delay and dynamic fingering.
- `CoupledStringBank`: multiple detuned strings sharing a bridge state for piano unisons, courses and sympathetic coupling.
- `AcousticCavityNetwork`: lumped compliant cavity with multiple inertive ports.

## Numerical model

The wind network uses one-way fractional delays for each bore section. At each tone-hole location the left bore, right bore and side branch are joined by a lossless impedance-weighted scattering junction:

`pJ = 2 * sum(p_i / Z_i) / sum(1 / Z_i)`

`p_i- = pJ - p_i+`

A closed tone hole is represented by a very high branch impedance; opening the hole continuously lowers the branch impedance, avoiding a hard topology switch.

## Compatibility

- Public version is now `0.13.0`.
- Historical v0.12 version assertion was relaxed to accept later compatible releases.
- Existing v0.6-v0.12 APIs remain available.

## Validation

The targeted physical/compatibility suite covering v0.6 through v0.13 passes: **46 tests**.
