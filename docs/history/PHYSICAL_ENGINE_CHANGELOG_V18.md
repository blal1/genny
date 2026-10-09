# Genny 0.18 — Shared Physical String Core

## Added

- `StringMaterial` and `STRING_MATERIALS` for steel/music wire, nylon, gut and bronze.
- `StringPhysicalProperties` with explicit SI geometry/material quantities and derived:
  - cross-sectional area and linear density,
  - tension from requested fundamental,
  - flexural rigidity `E I`,
  - characteristic impedance `sqrt(T mu)`,
  - stiff-string inharmonicity coefficient `B`,
  - longitudinal wave speed and longitudinal fundamental,
  - stretched partial-frequency prediction.
- `DispersiveString`, a stateful two-polarization reduced waveguide with:
  - Thiran fractional delays,
  - T60-derived loop loss,
  - stiffness-derived dispersion allpasses,
  - transverse polarization exchange,
  - weak longitudinal mode,
  - bridge-velocity coupling.
- `PluckedString.physical_note()` as a backward-compatible route into the shared string core.

## Piano integration

`PianoStringCourse` now creates a physical specification for every unison string. The
course geometry scales across the keyboard and the derived inharmonicity drives the
existing dispersive sections. Public piano APIs remain unchanged.

The realtime piano still uses dimensionless/scaled bridge coupling internally for
numerical robustness. `physical_bridge_impedance_n_s_m` exposes the SI string-side
impedance without pretending that every legacy bridge state is already unit preserving.

## Validation

- v0.18 string tests pass at 22.05, 44.1 and 48 kHz.
- targeted v0.6–v0.18 regression suite: 72 passed.
- package/tests/examples compile successfully.
