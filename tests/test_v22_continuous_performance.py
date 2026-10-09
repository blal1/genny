import numpy as np
import genny


def test_hertz_contact_is_superlinear_and_zero_outside_contact():
    c=genny.HertzFretContact(max_force_n=1e6)
    assert c.force_n(0.0)==0.0
    f1=c.force_n(1e-5)
    f2=c.force_n(4e-5)
    assert f1>0 and f2>4*f1


def test_combined_performance_is_finite_and_changes_geometry():
    p=genny.StringPhysicalProperties.from_frequency(110.0,material='steel')
    perf=genny.StringPerformance(p,sr=22050)
    y=perf.render(.22,gestures=[
        genny.StringGesture('bend',.02,.18,0,160),
        genny.StringGesture('vibrato',.04,.18,10,rate_hz=5.5),
        genny.StringGesture('slide',.06,.16,0,2),
        genny.StringGesture('palm_mute',.14,.22,.2,.8),
        genny.StringGesture('fret_contact',.03,.22,.5,1.0),
    ])
    assert len(y)==round(.22*22050)
    assert np.isfinite(y).all() and np.max(np.abs(y))>1e-5
    d=perf.last_diagnostics
    assert d['min_length_m'] < p.length_m
    assert d['max_frequency_hz'] > p.frequency_hz
    assert d['final_inharmonicity_B'] > 0


def test_bend_uses_tension_squared_frequency_relation():
    p=genny.StringPhysicalProperties.from_frequency(220.0)
    perf=genny.StringPerformance(p,sr=22050)
    perf.render(.12,gestures=[genny.StringGesture('bend',0,.12,0,120)])
    target=2**(2*120/1200)
    assert abs(perf.last_diagnostics['final_tension_ratio']-target)<.01


def test_contact_changes_same_loop_output():
    p=genny.StringPhysicalProperties.from_frequency(146.83)
    a=genny.StringPerformance(p,sr=22050).render(.25)
    perf=genny.StringPerformance(p,sr=22050)
    b=perf.render(.25,gestures=[genny.StringGesture('fret_contact',0,.25,1,1)])
    assert np.isfinite(b).all()
    assert np.linalg.norm(a-b)>1e-4


def test_high_level_plucked_performance_accepts_dict_gestures():
    inst=genny.PluckedString('guitar')
    y=inst.performance(196.0,.15,sr=22050,gestures=[
        {'kind':'bend','start_s':.02,'end_s':.12,'value':0,'end_value':80},
        {'kind':'vibrato','start_s':.04,'end_s':.14,'value':8,'rate_hz':5.0},
    ])
    assert len(y)==round(.15*22050)
    assert np.isfinite(y).all() and np.max(np.abs(y))>1e-5
    assert inst.last_performance_diagnostics['max_frequency_hz']>196.0
