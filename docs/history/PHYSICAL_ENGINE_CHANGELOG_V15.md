# Genny 0.15 — Stateful Polyphonic Piano

## Added

- `FrequencyDependentBridge`: reduced bridge with low/high-frequency force split.
- `PianoStringCourse`: 1/2/3-string course depending on keyboard register.
- `SympatheticResonatorBank`: keyboard-wide passive resonance bank plus undamped aliquot modes.
- `PolyphonicPiano`: stateful 88-key reduced physical piano with shared bridge and modal soundboard.
- Sample-accurate `note_on`, `note_off`, `sustain_pedal`, and `render_events` API.
- `Piano.polyphonic()` convenience constructor.

## Physical behaviour

- Notes share one bridge and one soundboard rather than being rendered independently.
- Note-off changes damper loss instead of hard-gating the signal.
- Sustain pedal keeps released courses and sympathetic resonators undamped.
- Aliquot modes remain open and can receive energy through bridge motion.
- Bass notes automatically use fewer unison strings than the treble.
- Per-block rendering preserves state and relative dynamics; it is not independently normalized unless clipping would occur.

This remains a reduced-order realtime model, not an FEM piano.
