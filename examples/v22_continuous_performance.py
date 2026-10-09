import genny

string = genny.PluckedString("guitar")

gestures = [
    genny.StringGesture("bend", 0.15, 0.75, 0.0, 180.0),
    genny.StringGesture("vibrato", 0.35, 1.20, 12.0, rate_hz=5.4),
    genny.StringGesture("slide", 0.70, 1.15, 0.0, 3.0),
    genny.StringGesture("fret_contact", 0.50, 1.30, 0.3, 0.9),
    genny.StringGesture("palm_mute", 1.05, 1.50, 0.0, 0.8),
]

audio = string.performance(196.0, 1.5, gestures=gestures, sr=48000)
print(string.last_performance_diagnostics)
print(audio.shape)
