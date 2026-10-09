# Genny 0.10.0 — Common sample-by-sample physical solver

This release adds the first shared sample-by-sample numerical core used by the
instrument interaction layer.

## New common solver

`genny.graphsolver` adds:

- `FractionalDelay`
- `CommonPhysicalSolver`
- `hammer_string()`
- `bow_string()`
- `reed_bore()`
- `lip_bore()`
- `syrinx_trachea()`

The five solvers run sample by sample and exchange conjugate physical variables
through `PhysicalPort` objects. They include propagation delay, passive losses,
reflection and reduced nonlinear contact/flow laws.

## Hybrid migration

The v0.9 interaction facades now use the common solver for their physical port
traces and feedback behavior while retaining the mature specialized kernels as
part of the audible renderer. The audio is currently a calibrated hybrid. This
keeps timbral compatibility while the common solver is progressively refined.

Migrated interactions:

- Hammer ↔ String
- Bow ↔ String
- Reed ↔ Bore
- Lips ↔ Brass bore
- Syrinx ↔ Trachea

The `InteractionResult.metadata["solver"]` field identifies these as
`hybrid-common/...`.

## Numerical notes

The new core is deliberately reduced-order rather than a claim of complete
FEM/FDTD accuracy. It provides one common causal feedback engine that can be
made progressively more sophisticated without changing the public model API.
