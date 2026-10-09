import math
import numpy as np
import genny
from genny.graphsolver import (
    FractionalDelay, ThiranDelay, FrequencyDependentLoss, StiffStringDispersion,
    StickSlipHysteresis, MultiportScatteringJunction, ToneHole, BellRadiation,
    JetDelay, jet_bore,
)


def test_fractional_delay_linear_direction_and_lagrange_finite():
    d=FractionalDelay(1.5,32,'linear')
    for x in [1.,2.,3.,4.]: d.write(x)
    # next write index is 4; 1.5 samples ago lies halfway between the two most recent samples (3 and 4)
    assert abs(d.read()-3.5)<1e-9
    q=FractionalDelay(2.25,32,'lagrange3')
    for x in np.linspace(-1,1,12): q.write(float(x))
    assert math.isfinite(q.read())


def test_thiran_is_bounded_for_sine():
    d=ThiranDelay(7.35,64)
    ys=[]
    for n in range(2000):
        ys.append(d.read()); d.write(math.sin(2*math.pi*0.03*n))
    assert np.isfinite(ys).all()
    assert max(abs(np.asarray(ys))) < 1.1


def test_frequency_loss_and_dispersion_bounded():
    loss=FrequencyDependentLoss(.999,.96,.4)
    disp=StiffStringDispersion(.8,4)
    x=1.0
    ys=[]
    for _ in range(2000):
        x=disp.process(loss.process(x)); ys.append(x)
    assert np.isfinite(ys).all()
    assert max(abs(np.asarray(ys))) <= 1.000001


def test_stick_slip_has_state():
    f=StickSlipHysteresis()
    _,s0=f.step(0.0,1.0)
    _,s1=f.step(1.0,1.0)
    assert s0=='stick' and s1=='slip'


def test_multiport_scattering_equal_impedance():
    j=MultiportScatteringJunction(np.ones(3))
    out=j.scatter(np.array([1.,0.,0.]))
    # junction pressure is 2/3, so outgoing is [-1/3, 2/3, 2/3]
    assert np.allclose(out,[-1/3,2/3,2/3])


def test_tonehole_and_bell_passive_finite():
    t=ToneHole(1e6,2e5,.7)
    r,rad=t.process(1.0)
    assert abs(r)<=1 and math.isfinite(rad)
    b=BellRadiation(900,48000,1.0)
    vals=[b.process(1.0 if i==0 else 0.0) for i in range(200)]
    assert np.isfinite(vals).all()


def test_jet_delay_and_common_flute():
    jd=JetDelay(.01,.4,.03,24000)
    assert jd.samples(20.0)>1
    r=jet_bore(440,.08,jet_velocity=20,sr=24000)
    assert len(r.audio)==1920 and np.isfinite(r.audio).all()
    f=genny.Flute().note(440,.08,sr=24000)
    assert len(f)==1920 and np.isfinite(f).all()


def test_existing_common_solvers_remain_finite():
    for fn,args in [
        (genny.CommonPhysicalSolver.hammer_string,(440,.05)),
        (genny.CommonPhysicalSolver.bow_string,(220,.05)),
        (genny.CommonPhysicalSolver.reed_bore,(220,.05)),
        (genny.CommonPhysicalSolver.lip_bore,(220,.05)),
        (genny.CommonPhysicalSolver.syrinx_trachea,((700,710),.05)),
    ]:
        r=fn(*args,sr=24000)
        assert np.isfinite(r.audio).all()
        assert np.max(np.abs(r.audio))<=1.000001
