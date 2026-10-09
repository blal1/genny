import numpy as np
import genny


def rms(x):
    x=np.asarray(x,float)
    return float(np.sqrt(np.mean(x*x)))


def test_hammer_action_has_force_contact_and_rebound():
    h=genny.PianoHammerAction(60,sr=22050)
    p=h.strike(.85)
    assert len(p)>2
    assert np.all(np.isfinite(p))
    assert np.max(h.last_force_n)>0
    assert h.last_contact_time_s>0
    assert h.last_rebound_velocity_m_s<=0


def test_two_polarizations_and_longitudinal_mode_receive_energy():
    c=genny.PianoStringCourse(60,sr=22050,coupling=.07)
    c.note_on(.9)
    for _ in range(3000):
        incoming,_=c.read_force()
        c.write(incoming,0.02)
    assert c.energy_proxy()>0
    assert c.polarization_energy()>0
    assert c.longitudinal_energy()>0


def test_note_aware_bridge_returns_different_local_mobility():
    b=genny.FrequencyDependentBridge(sr=22050)
    _,_,local=b.step_note_forces({30:1.0,90:1.0})
    assert set(local)=={30,90}
    assert local[30] != local[90]
    assert np.isfinite(local[30]) and np.isfinite(local[90])


def test_polyphonic_piano_v17_stable_and_exposes_new_state():
    for sr in (22050,44100,48000):
        p=genny.PolyphonicPiano(sr=sr)
        p.note_on(48,.8)
        p.note_on(72,.7)
        y=p.render(.08)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y))>0
        assert p.courses[48].polarization_energy()>0
        assert p.courses[72].hammer_action.last_contact_time_s>0
        assert hasattr(p.bridge,'last_note_velocities')


def test_soft_pedal_hammer_contact_is_lower_energy():
    h1=genny.PianoHammerAction(72,sr=22050)
    a=h1.strike(.8,soft=False)
    f1=np.sum(h1.last_force_n**2)
    h2=genny.PianoHammerAction(72,sr=22050)
    b=h2.strike(.8,soft=True)
    f2=np.sum(h2.last_force_n**2)
    assert f2 < f1
    assert rms(b) < rms(a)
