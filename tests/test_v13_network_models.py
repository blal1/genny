import numpy as np
import genny


def finite_nonzero(x):
    x=np.asarray(x,float)
    return len(x)>0 and np.all(np.isfinite(x)) and np.max(np.abs(x))>1e-7


def test_multihole_bore_dynamic_fingering():
    sr=22050; n=int(.15*sr)
    bore=genny.MultiHoleBore(length_m=.55,sr=sr)
    drive=np.zeros(n); drive[:8]=.1
    fing=np.zeros((n,len(bore.holes)))
    fing[n//2:, -2:]=1.0
    y,p=bore.render(drive,fing)
    assert finite_nonzero(y)
    assert np.all(np.isfinite(p))


def test_multihole_clarinet():
    c=genny.MultiHoleClarinet(220,sr=22050)
    f=np.zeros(len(c.bore.holes)); f[-1]=.7
    y,a,b=c.note(.12,fingering=f,seed=2)
    assert finite_nonzero(y)
    assert a.effort_unit=='Pa' and b.flow_unit=='m^3/s'


def test_multihole_flute():
    f=genny.MultiHoleFlute(440,sr=22050)
    fing=np.zeros(len(f.bore.holes)); fing[-1]=1.0
    y,a,b=f.note(.12,fingering=fing)
    assert finite_nonzero(y)
    assert np.all(np.isfinite(a.power))


def test_coupled_strings():
    s=genny.CoupledStringBank((220,220.4,219.6),sr=22050,coupling=.1)
    y=s.render(.15)
    assert finite_nonzero(y)


def test_acoustic_cavity():
    c=genny.AcousticCavityNetwork(volume_m3=.03,sr=22050)
    y=c.render_impulse(2000)
    assert finite_nonzero(y)
    assert np.max(np.abs(y))<=.900001
