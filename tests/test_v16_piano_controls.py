import numpy as np
import genny


def rms(x):
    x=np.asarray(x,float)
    return float(np.sqrt(np.mean(x*x)))


def test_una_corda_reduces_energy_and_struck_strings():
    p=genny.PolyphonicPiano(sr=22050)
    p.note_on(72, 0.85)
    normal=p.render(0.12)
    p2=genny.PolyphonicPiano(sr=22050)
    p2.una_corda(True)
    p2.note_on(72, 0.85)
    soft=p2.render(0.12)
    assert np.all(np.isfinite(soft))
    assert rms(soft) < rms(normal)
    assert np.count_nonzero(p2.courses[72]._hammer_gains) < len(p2.courses[72].lines)


def test_sostenuto_captures_only_current_notes():
    p=genny.PolyphonicPiano(sr=22050)
    p.note_on(60, .8)
    p.sostenuto_pedal(True)
    p.note_on(64, .8)
    p.note_off(60)
    p.note_off(64)
    assert p.courses[60].sostenuto_held
    assert not p.courses[64].sostenuto_held
    tail=p.render(.08)
    assert np.max(np.abs(tail)) > 0
    p.sostenuto_pedal(False)
    assert not p.courses[60].sostenuto_held


def test_partial_damper_is_between_open_and_closed():
    def tail(amount):
        p=genny.PolyphonicPiano(sr=22050)
        p.note_on(55,.8)
        p.render(.08)
        p.note_off(55)
        p.set_damper(55, amount)
        return rms(p.render(.15))
    opened=tail(0.0)
    partial=tail(0.5)
    closed=tail(1.0)
    assert opened > partial > closed


def test_hammer_geometry_varies_with_key():
    p=genny.PolyphonicPiano(sr=22050)
    lo=p._course(30)
    hi=p._course(90)
    assert lo.hammer_position_ratio > hi.hammer_position_ratio
    assert 0 < hi.hammer_position_ratio < .2


def test_render_events_supports_all_three_pedals_and_damper():
    p=genny.PolyphonicPiano(sr=22050)
    y=p.render_events([
        {'time':0.00,'type':'note_on','note':60,'velocity':.8},
        {'time':0.02,'type':'sostenuto','down':True},
        {'time':0.03,'type':'note_off','note':60},
        {'time':0.04,'type':'una_corda','down':True},
        {'time':0.05,'type':'note_on','note':72,'velocity':.7},
        {'time':0.06,'type':'damper','note':72,'amount':.4},
        {'time':0.08,'type':'sustain','down':True},
    ], .12)
    assert y.shape == (int(round(.12*22050)),)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) > 0
