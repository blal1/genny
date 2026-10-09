"""Small examples of the Genny 0.3 physical API."""
from pathlib import Path
import genny
from genny.core import write_wav

OUT = Path("out_physical")

write_wav(OUT / "string.wav", genny.StringInstrument(body="guitar").pluck(220, 2.0))
write_wav(OUT / "piano.wav", genny.Piano().note(261.63, 3.0, velocity=.8))
write_wav(OUT / "maraca.wav", genny.ParticleMaterial("Maraca").shake(2.0))
write_wav(OUT / "bubbles.wav", genny.BubblePopulation(rate=80).render(3.0))
write_wav(OUT / "voice.wav", genny.VocalTract().phonate(120, 1.5))
write_wav(OUT / "bird.wav", genny.BirdSyrinx().call((650, 657), 1.5))
write_wav(OUT / "rain.wav", genny.Environment(seed=7).rain(4.0, intensity=.8))
