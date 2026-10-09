import numpy as np
import genny


def _ok(x):
    x=np.asarray(x)
    assert x.size > 100
    assert np.isfinite(x).all()
    assert np.max(np.abs(x)) > 1e-8


def test_tuned_percussion_and_plucks():
    _ok(genny.TunedPercussion('marimba').note(440, .25, sr=22050))
    _ok(genny.TunedPercussion('bell').note(440, .25, sr=22050))
    _ok(genny.PluckedString('guitar').note(220, .25, sr=22050))
    _ok(genny.PluckedString('mandolin').note(330, .25, sr=22050))


def test_engine_and_pour():
    eng=genny.CombustionEngine(cylinders=4, rpm=1800)
    assert abs(eng.firing_rate_hz-60.0) < 1e-9
    _ok(eng.render(.25, sr=22050))
    _ok(genny.Environment(seed=4).pour(.25, sr=22050, flow=.8, fill=.2))
