# Genny 0.9.0 — Unified Interaction Ports

Version 0.9 introduces a common physical interaction interface above Genny's
specialized nonlinear waveguide solvers.

## New interaction layer

`genny.interactions` adds:

- `InteractionResult`
- `HammerStringInteraction`
- `BowStringInteraction`
- `ReedBoreInteraction`
- `LipBoreInteraction`
- `SyrinxTracheaInteraction`

Every interaction returns the rendered audio together with two typed
`PhysicalPort` traces and diagnostic metadata. Mechanical ports carry force [N]
and velocity [m/s]; acoustic ports carry pressure [Pa] and volume flow [m^3/s].

The mature 48 kHz waveguide kernels remain the high-rate nonlinear solvers in
this release. The new port layer standardizes their external physical interface
so the internal solvers can be replaced incrementally without changing the
public model API.

## Public models migrated

The following facades now construct and solve the shared interaction objects:

- `Piano` -> `HammerStringInteraction`
- `BowedString` -> `BowStringInteraction`
- `Clarinet` -> `ReedBoreInteraction`
- `Brass` -> `LipBoreInteraction`
- `BirdSyrinx` -> `SyrinxTracheaInteraction`

Each facade exposes `.interaction()` for advanced access to the physical-port
solver.

## Physical diagnostics

- Felt hammer diagnostics expose the nonlinear exponent and stiffness derived
  from key/frequency.
- Reed, lip and syrinx interactions expose reduced characteristic acoustic
  impedance.
- Ports expose instantaneous power (`effort * flow`) and integrated energy.
- Passive couplers are checked by tests using their orthogonal scattering norm.

## Validation

The targeted regression suite passes 25 tests covering versions 0.3–0.9,
including 22.05, 44.1 and 48 kHz rendering paths where applicable.
