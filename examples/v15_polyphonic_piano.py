import genny

p = genny.PolyphonicPiano(sr=48000)
events = [
    {"time": 0.00, "type": "note_on", "note": 60, "velocity": .85},
    {"time": 0.00, "type": "note_on", "note": 64, "velocity": .75},
    {"time": 0.00, "type": "note_on", "note": 67, "velocity": .80},
    {"time": 0.35, "type": "sustain", "down": True},
    {"time": 0.55, "type": "note_off", "note": 60},
    {"time": 0.55, "type": "note_off", "note": 64},
    {"time": 0.55, "type": "note_off", "note": 67},
    {"time": 1.60, "type": "sustain", "down": False},
]
audio = p.render_events(events, 2.4)
print(audio.shape, audio.min(), audio.max())
