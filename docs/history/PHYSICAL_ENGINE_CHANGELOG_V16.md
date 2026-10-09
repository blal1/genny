# Genny 0.16.0 - Piano action and pedals

- Added true sostenuto capture semantics to `PolyphonicPiano`.
- Added una-corda/soft-pedal state. Treble/mid courses strike one fewer string and use a softer hammer excitation.
- Added continuous per-note damper contact via `set_damper(note, amount)`.
- Added keyboard-dependent hammer strike position and a reduced-order position comb in the excitation.
- Added note-dependent duplex/aliquot stretch in the sympathetic bank.
- Extended `render_events()` with `sostenuto`, `una_corda`/`soft`, and `damper`.
- Fixed voice retirement so sostenuto-held notes cannot be collected prematurely.
- Preserved all v0.15 public APIs.
