import numpy as np
import genny
from genny.graphsolver import hammer_string, bow_string, reed_bore


def _ok(r, n):
    assert len(r.audio)==n
    assert np.isfinite(r.audio).all()
    assert np.max(np.abs(r.audio)) <= 1.000001
    assert np.isfinite(r.drive.power).all()
    assert np.isfinite(r.response.power).all()


def test_common_hammer_string():
    sr=24000; r=hammer_string(440, .15, sr=sr)
    _ok(r, int(.15*sr))
    assert r.metadata['solver']=='common-graph'


def test_common_bow_string():
    sr=24000; r=bow_string(220, .15, sr=sr)
    _ok(r, int(.15*sr))


def test_common_reed_bore():
    sr=24000; r=reed_bore(220, .15, sr=sr)
    _ok(r, int(.15*sr))


def test_public_exports():
    assert genny.CommonPhysicalSolver is not None

def test_common_lip_and_syrinx():
    from genny.graphsolver import lip_bore, syrinx_trachea
    _ok(lip_bore(220,.1,sr=24000),2400)
    _ok(syrinx_trachea((700,710),.1,sr=24000),2400)

def test_hybrid_interactions_report_common_solver():
    for obj, args in [
        (genny.HammerStringInteraction(), (440,.08)),
        (genny.BowStringInteraction(), (220,.08)),
        (genny.ReedBoreInteraction(), (220,.08)),
        (genny.LipBoreInteraction(), (220,.08)),
        (genny.SyrinxTracheaInteraction(), ((700,710),.08)),
    ]:
        r=obj.solve(*args,sr=24000)
        assert 'hybrid-common' in r.metadata['solver']
        assert np.isfinite(r.audio).all()
