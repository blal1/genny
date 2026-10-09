# Genny 0.11.0 — waveguide core upgrade

## New shared waveguide primitives

- Corrected the fractional-delay read geometry used by the v0.10 common graph.
- Added cubic 4-point Lagrange fractional delay interpolation.
- Added first-order Thiran allpass delay for magnitude-preserving fractional group delay.
- Added passive frequency-dependent loop loss so high frequencies can decay faster than lows.
- Added allpass-based stiff-string dispersion sections.
- Added stateful stick/slip hysteresis for bowed/creaking contacts.
- Added N-port acoustic scattering junctions using branch impedances.
- Added reduced continuously variable tone-hole scattering.
- Added passive bell/open-end radiation approximation.
- Added hydrodynamic jet-delay model (`tau = L / U_conv`).

## Common solver migration

- `hammer_string` now uses cubic fractional delay, frequency-dependent losses and stiff-string dispersion.
- `bow_string` now uses stateful stick/slip hysteresis rather than a memoryless friction curve.
- `reed_bore` now uses a Thiran delay, frequency-dependent losses, a reduced tone hole and bell radiation.
- `lip_bore` now uses a Thiran delay, frequency-dependent losses and bell radiation.
- `syrinx_trachea` now uses a Thiran delay and frequency-dependent tracheal losses.
- Added `jet_bore`, a shared flute/edge-tone loop using a physical hydrodynamic jet delay.
- `genny.Flute` now blends the common jet/bore solver into the mature specialized renderer during migration.

## Legacy wrapper cleanup found by stricter tests

While validating the new core, the old all-instrument tests exposed several legacy wrappers that had drifted from the historic pitch/brightness contract. The xylophone and bell wrappers received conservative pitch/radiation anchors and HF damping. The full strict catalog still exposes additional legacy issues (currently including timpani tuning and marginal steel-guitar sharpness), so v0.11 does **not** claim that the entire historical `test_instruments.py` suite passes yet.

## Validation

The focused compatibility/physical suite passes 39 tests across v0.6-v0.11, including spec rendering, unified wrappers, ports/interactions and the new waveguide primitives.
