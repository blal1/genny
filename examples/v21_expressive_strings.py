"""Genny 0.21 expressive string gestures."""
import genny

SR = 48000
guitar = genny.PluckedString("guitar", position=0.18)

bend = guitar.expressive_note(196.0, 1.0, technique="bend", cents=200, sr=SR)
vibrato = guitar.expressive_note(196.0, 1.0, technique="vibrato", depth_cents=22, rate_hz=5.4, sr=SR)
slide = guitar.expressive_note(196.0, 1.2, technique="slide", end_frequency=293.66,
                               roughness=0.55, pressure=0.6, sr=SR)
hammer = guitar.expressive_note(196.0, 0.8, technique="hammer_on", fret=5, sr=SR)
pull = guitar.expressive_note(196.0, 0.8, technique="pull_off", fret=3, sr=SR)
tap = guitar.expressive_note(196.0, 0.8, technique="tapping", fret=12, sr=SR)
dead = guitar.expressive_note(196.0, 0.6, technique="dead_note", sr=SR)
pinch = guitar.expressive_note(196.0, 1.0, technique="pinch_harmonic", harmonic=4, sr=SR)
buzz = guitar.expressive_note(196.0, 1.0, technique="fret_buzz_feedback", fret=3, sr=SR)

violin = genny.BowedString(body="violin")
bowed_vibrato = violin.vibrato_note(220.0, 1.2, depth_cents=17, rate_hz=5.2, sr=SR)
