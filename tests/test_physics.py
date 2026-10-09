"""Smoke tests for the physically informed Genny v0.3 API."""
import numpy as np
import genny


def finite_audio(x):
    x = np.asarray(x)
    assert x.size > 0
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) > 1e-8


def test_physical_models_smoke():
    finite_audio(genny.ModalBody("standardBell", genny.MATERIALS["glass"]).strike(duration=0.08))
    finite_audio(genny.ParticleMaterial("Maraca").shake(0.08))
    finite_audio(genny.BubblePopulation(rate=12).render(0.08))
    finite_audio(genny.StringInstrument(body="none").pluck(220.0, 0.08))
    finite_audio(genny.Piano().note(220.0, 0.08))
    finite_audio(genny.Clarinet().note(220.0, 0.08))
    finite_audio(genny.Brass().note(220.0, 0.08))
    finite_audio(genny.VocalTract().phonate(120.0, 0.08))
    finite_audio(genny.BirdSyrinx().call(500.0, 0.05))


def test_environment_and_acoustics():
    finite_audio(genny.Environment(seed=1).rain(0.08))
    x = np.zeros(4800)
    x[0] = 1.0
    y = genny.Propagation().apply(x, 3.0, sr=48000, delay=False)
    finite_audio(y)
    t60 = genny.Room(volume=50, surfaces={"plaster": 50, "wood": 20}).t60_bands()
    assert t60.shape == (6,)
    assert np.all(np.isfinite(t60)) and np.all(t60 > 0)


def test_v04_extended_models():
    finite_audio(genny.BarBody(genny.MATERIALS["steel"], length_m=0.35).strike(duration=0.08))
    finite_audio(genny.BowedString(body="violin").note(220.0, 0.08))
    finite_audio(genny.Flute().note(440.0, 0.08))
    finite_audio(genny.Footstep("wood").render(seed=2))
    finite_audio(genny.AeroacousticSwing().render(seed=3))
    finite_audio(genny.Environment(seed=4).wind(0.08))
    finite_audio(genny.Environment(seed=4).cave_drips(0.15, sources=1))


def test_moving_propagation():
    sr = 48000
    t = np.arange(int(0.12 * sr)) / sr
    x = np.sin(2 * np.pi * 440 * t)
    d = np.linspace(12.0, 3.0, len(x))
    y = genny.MovingPropagation().apply_trajectory(x, d, sr=sr, block=512)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) > 1e-8
