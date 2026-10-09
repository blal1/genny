import numpy as np
from genny.instruments import render_note
from genny.drums import render_drum
from genny.sfx import render_sfx


def ok(x):
    x=np.asarray(x); assert x.size>100; assert np.isfinite(x).all(); assert np.max(np.abs(x))>1e-9


def test_migrated_instruments():
    for name,f in [('xylophone',440),('glockenspiel',880),('tubular_bell',330),('mandolin',330),('steel_guitar',220),('banjo',330)]:
        ok(render_note(name,f,.25,sr=22050,vel=.7))


def test_migrated_drums():
    for name in ['shaker','tambourine','woodblock','gong']:
        ok(render_drum(name,sr=22050,vel=.7))


def test_environment_and_vehicle_sfx():
    for name,p in [('rain',dict(dur=.2)),('fire',dict(dur=.2)),('stream',dict(dur=.2)),('pour',dict(dur=.2)),('engine',dict(dur=.2,rpm=1800)),('swoosh',dict(dur=.15))]:
        ok(render_sfx(name,sr=22050,**p))
