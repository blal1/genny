import numpy as np
import genny


def finite_nonzero(x):
    x=np.asarray(x,float)
    return x.size and np.all(np.isfinite(x)) and np.max(np.abs(x))>1e-8


def test_bridge_energy_finite():
    b=genny.BridgeImpedance(sr=22050)
    for i in range(500):
        b.step(1.0 if i<20 else 0.0)
    assert np.isfinite(b.energy_j) and b.energy_j>=0


def test_soundboard_impulse_decays_and_is_finite():
    s=genny.ModalSoundboard(sr=22050)
    y=[]
    for i in range(5000):
        r,_=s.step(1.0 if i==0 else 0.0); y.append(r)
    y=np.asarray(y)
    assert finite_nonzero(y)
    assert np.max(np.abs(y[-500:])) < np.max(np.abs(y[:1000]))


def test_sympathetic_strings_receive_energy():
    p=genny.SympatheticPiano(220.0, sr=22050, coupling=.08)
    y,stems=p.render(.35,velocity=.9,return_stems=True)
    assert finite_nonzero(y)
    assert np.all(np.isfinite(stems))
    # first 3 are struck unisons; later strings must receive bridge energy.
    sym=np.sqrt(np.mean(stems[3:]**2,axis=1))
    assert np.max(sym)>1e-7
    assert p.last_soundboard_energy>0


def test_sympathetic_coupling_changes_unstruck_energy():
    p0=genny.SympatheticPiano(220.0, sr=22050, coupling=0.0)
    _,s0=p0.render(.25,return_stems=True)
    p1=genny.SympatheticPiano(220.0, sr=22050, coupling=.09)
    _,s1=p1.render(.25,return_stems=True)
    e0=float(np.mean(s0[3:]**2)); e1=float(np.mean(s1[3:]**2))
    assert e1 > e0 + 1e-12


def test_sympathetic_piano_multisr():
    for sr in (22050,44100,48000):
        p=genny.SympatheticPiano(330.0,sr=sr,coupling=.06)
        y=p.render(.08)
        assert len(y)==round(.08*sr)
        assert finite_nonzero(y)
        assert np.max(np.abs(y))<=.900001


def test_piano_high_level_sympathetic_api():
    p=genny.Piano()
    y,stems=p.sympathetic_note(220,.12,sr=22050,coupling=.07,return_stems=True)
    assert finite_nonzero(y)
    assert stems.shape[1]==len(y)
    assert stems.shape[0] >= 4
