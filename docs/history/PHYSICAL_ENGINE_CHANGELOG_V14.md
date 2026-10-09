# Genny 0.14.0 — Sympathetic Piano Network

## Added

- `BridgeImpedance`: reduced mechanical mass-spring-damper bridge in SI units.
- `ModalSoundboard`: one shared damped modal soundboard driven by bridge force.
- `SympatheticPiano`: unison and unstruck sympathetic strings coupled through the same bridge and soundboard.
- `Piano.sympathetic_note(...)`: high-level entry point that leaves the legacy-calibrated `Piano.note(...)` unchanged.

## Physical model

The network uses a common bridge state rather than summing independent string renders. String velocity-like states generate bridge force, the bridge drives a shared modal soundboard, and a bounded fraction of bridge/soundboard velocity is scattered back to every string. Consequently unstruck strings can acquire energy from struck strings.

The bridge follows a reduced mechanical model:

`m x'' + c x' + k x = F`

with state energy

`E = 1/2 m v^2 + 1/2 k x^2`.

The soundboard modes are stable second-order recurrences with radii derived from per-mode T60 values.

## Validation

- Explicit test that sympathetic strings receive energy only when coupling is enabled.
- Soundboard impulse-decay test.
- Bridge finite-energy test.
- 22.05, 44.1 and 48 kHz render tests.
- Targeted compatibility suite v0.6-v0.14: 50 tests passed.
