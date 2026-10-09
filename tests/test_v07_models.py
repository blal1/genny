import numpy as np
import pytest

from genny.physics import MembraneDrum, CymbalPlate, SurfaceContact, Rotor, JetEngine, Helicopter, Destruction
from genny.drums import render_drum
from genny.sfx import REGISTRY as SFX
from genny.instruments import render_note


def sane(x):
    x=np.asarray(x)
    assert x.size > 64
    assert np.isfinite(x).all()
    assert np.max(np.abs(x)) > 1e-6
    assert np.max(np.abs(x)) <= 1.25


@pytest.mark.parametrize('sr',[22050,44100,48000])
def test_v07_core_models(sr):
    sane(MembraneDrum(180).strike(.12,sr=sr))
    sane(CymbalPlate().strike(.12,sr=sr))
    sane(SurfaceContact().scrape(.12,sr=sr))
    sane(SurfaceContact().roll(.12,sr=sr))
    sane(Rotor().render(.12,sr=sr))
    sane(JetEngine().render(.12,sr=sr))
    sane(Helicopter().render(.12,sr=sr))
    sane(Destruction().render(.12,sr=sr))


def test_v07_legacy_migrations():
    for name in ('conga','bongo','djembe','tabla','tom','taiko','crash','ride','hihat','guiro'):
        sane(render_drum(name,sr=22050,vel=.7))
    for name,f in [('timpani',90),('electric_guitar',220),('koto',330),('sitar',220),('upright_bass',55),('finger_bass',82)]:
        sane(render_note(name,f,.14,sr=22050,vel=.7))
    for name in ('door','roll','scrape','fan','propeller','jet_engine','helicopter','shatter'):
        sane(SFX[name]['fn'](22050,dur=.14))


def test_rotor_formulae():
    r=Rotor(rpm=1800,blades=5,radius_m=.2)
    assert r.rotation_hz == pytest.approx(30.0)
    assert r.blade_passing_hz == pytest.approx(150.0)
    assert r.tip_speed == pytest.approx(2*np.pi*30*.2)
