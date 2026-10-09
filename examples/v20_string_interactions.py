"""Genny 0.20 local string-interaction examples."""
import genny

sr = 48000
props = genny.WoundStringPhysicalProperties.from_frequency(82.41, length_m=0.648)

# Hard pick.
string = genny.DispersiveString(props, sr=sr)
picked = string.render_exciter(1.0, genny.StringExciter("pick", hardness=0.9, position=0.18), velocity=0.8)

# Fretted note with buzz.
string = genny.DispersiveString(props, sr=sr)
buzz = string.render_fret_buzz(5, 1.0, velocity=0.9, buzz=0.75)

# Palm mute.
string = genny.DispersiveString(props, sr=sr)
muted = string.render_exciter(1.0, genny.StringExciter("pick", 0.8, 0.15), velocity=0.85, palm_mute=0.8)

# Continuous slide: the delay itself moves sample-by-sample.
string = genny.DispersiveString(props, sr=sr)
slide = string.render_glissando(1.2, 82.41, 123.47, velocity=0.75)

# Natural harmonic.
string = genny.DispersiveString(props, sr=sr)
harmonic = string.render_natural_harmonic(3, 1.0, velocity=0.7)

# Slap / pop.
string = genny.DispersiveString(props, sr=sr)
slap = string.render_slap(0.8, velocity=0.95)
string = genny.DispersiveString(props, sr=sr)
pop = string.render_slap(0.8, velocity=0.95, pop=True)
