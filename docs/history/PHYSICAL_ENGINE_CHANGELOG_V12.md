# Genny 0.12.0 — Passive multiport waveguide networks

## New common waveguide network
- `WaveguideBranch`: passive branch with fractional propagation and frequency-dependent loss.
- `WaveguideNetwork`: N-branch lossless scattering network built from branch impedances.
- Reuses `MultiportScatteringJunction` from the v0.11 common core.
- Supports impulse rendering for network diagnostics and future tone-hole/branch/cavity graphs.

## Historical-wrapper fixes
- `upright_bass` and `finger_bass` now render an internal minimum physical window so very short requests no longer return silence at low pitch.
- `timpani` keeps the physical membrane/cavity texture while adding a stronger fundamental anchor for the legacy API tuning contract.

## Validation
- 41 targeted physical/common-core/compatibility tests pass.
- Critical historical checks pass for the corrected bass short-note cases and timpani tuning (<2 cents in the checked C2–C4 points).
- The full historical instrument suite remains computationally heavy and exceeded the available single-run test window when executed as one block; it is not claimed as fully completed in this release.
