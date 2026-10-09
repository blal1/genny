# Genny 0.22.0 — Continuous String Performance

## Added

- `HertzFretContact`: nonlinear reduced contact law `F = k * delta^(3/2)` with dissipative impact damping.
- `StringGesture`: time-bounded continuous control lane for bend, vibrato, slide/fret position, palm mute, fret contact, and harmonic touch.
- `StringPerformance`: one stateful renderer that combines several gestures inside the same dispersive-string simulation.
- `PluckedString.performance(...)`: high-level facade accepting `StringGesture` objects or dictionaries.

## Physical changes

- Fret/slide motion changes effective vibrating length continuously.
- Bend/vibrato changes tension continuously, with `T/T0 = (f/f0)^2` for the tension contribution.
- Effective inharmonicity is tracked as `B ~ 1 / (T L^2)` in diagnostics.
- Palm mute changes loop loss rather than post-render envelope only.
- Fret collision is injected back into the travelling-wave state using a nonlinear Hertz-like contact force.
- Multiple gesture lanes can overlap in one render.

## Diagnostics

`StringPerformance.last_diagnostics` exposes final frequency, vibrating length, tension ratio, inharmonicity, maximum contact force, minimum length, and maximum frequency.

## Validation

- 5 dedicated v0.22 tests.
- 93 targeted regression tests passed across v0.6 through v0.22 plus unified wrappers, physics, and spec tests.
- Full package, tests, and examples compile successfully.

This remains a reduced-order realtime model. Fret/string collision and finger geometry are not a full 3-D FEM/contact solve.
