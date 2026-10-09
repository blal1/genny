import numpy as np
import genny


def test_polyphonic_chord_is_finite_and_nonzero():
    p=genny.PolyphonicPiano(sr=12000)
    p.note_on(60,.8); p.note_on(64,.7); p.note_on(67,.75)
    y=p.render(.08)
    assert len(y)==960
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y))>1e-5
    assert len(p.courses)==3


def test_sustain_changes_release_decay():
    # Compare identical notes after note-off, with and without sustain.
    dry=genny.PolyphonicPiano(sr=12000)
    dry.note_on(60,.8); dry.render(.05); dry.note_off(60)
    yd=dry.render(.08)

    sus=genny.PolyphonicPiano(sr=12000)
    sus.note_on(60,.8); sus.render(.05); sus.sustain_pedal(True); sus.note_off(60)
    ys=sus.render(.08)
    # Pedal path should retain materially more tail energy.
    assert np.sqrt(np.mean(ys*ys)) > np.sqrt(np.mean(yd*yd))*1.05


def test_render_events_and_aliquots():
    p=genny.PolyphonicPiano(sr=12000)
    events=[
        {'time':0.0,'type':'note_on','note':60,'velocity':.8},
        {'time':0.02,'type':'sustain','down':True},
        {'time':0.04,'type':'note_off','note':60},
        {'time':0.06,'type':'note_on','note':67,'velocity':.7},
    ]
    y=p.render_events(events,.12)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y))>1e-6
    assert len(p.sympathetic.freq) > p.sympathetic.base_count


def test_piano_facade_polyphonic():
    p=genny.Piano().polyphonic(sr=12000)
    assert isinstance(p,genny.PolyphonicPiano)
    p.note_on(69,.7)
    y=p.render(.04)
    assert np.max(np.abs(y))>0


def test_frequency_dependent_bridge_is_finite():
    b=genny.FrequencyDependentBridge(sr=12000)
    vals=[b.step_split(1.0 if i==0 else 0.0) for i in range(200)]
    arr=np.asarray(vals)
    assert np.all(np.isfinite(arr))
    assert b.energy_j >= 0.0
