import numpy as np
import genny
from genny import drums, sfx


def _ok(x, sr, dur, tol=.08):
    x=np.asarray(x)
    assert x.ndim == 1
    assert np.all(np.isfinite(x))
    assert abs(len(x)/sr-dur) <= tol
    assert np.max(np.abs(x)) <= 1.05
    assert np.max(np.abs(x)) > 1e-6


def test_physical_ports_and_passive_graph():
    n=128
    a=genny.PhysicalPort.mechanical(np.ones(n),np.ones(n)*.2,"a")
    b=genny.PhysicalPort.mechanical(np.ones(n)*.4,np.ones(n)*-.1,"b")
    g=genny.PhysicalGraph(); g.add("a",a); g.add("b",b); g.couple("a","b",.6)
    out=g.process()
    assert set(out)=={"a","b"}
    assert np.all(np.isfinite(out["a"].power))
    assert a.effort_unit=="N" and a.flow_unit=="m/s"


def test_kick_snare_multi_sr():
    for sr in (22050,44100,48000):
        _ok(genny.KickDrum().strike(.35,sr=sr),sr,.35)
        _ok(genny.SnareDrum().strike(.30,sr=sr),sr,.30)
        _ok(drums.REGISTRY["kick"]["fn"](sr=sr,vel=.8,tune=55,decay=.3,punch=.7),sr,.42,.1)
        _ok(drums.REGISTRY["snare"]["fn"](sr=sr,vel=.8,tune=180,decay=.22,snappy=.7),sr,.30,.1)


def test_machine_models():
    _ok(genny.ElectricMotor(4800).render(.35,sr=44100),44100,.35)
    m=genny.ElectricMotor(6000)
    assert m.commutation_hz > m.rotation_hz
    assert m.imbalance_force_n >= 0
    g=genny.GearTrain(1800,20,40)
    assert abs(g.mesh_hz-600)<1e-9
    assert abs(g.output_rpm-900)<1e-9
    _ok(g.render(.35,sr=44100),44100,.35)


def test_advanced_roll_and_explosion():
    _ok(genny.AdvancedRollingContact().render(.35,sr=48000),48000,.35)
    _ok(genny.Explosion(distance_m=2).render(.6,sr=44100),44100,.6)
    _ok(sfx.REGISTRY["explosion"]["fn"](sr=44100,dur=.6,energy=.8,distance=3),44100,.6)
    _ok(sfx.REGISTRY["electric_motor"]["fn"](sr=44100,dur=.35,rpm=5000),44100,.35)
    _ok(sfx.REGISTRY["gears"]["fn"](sr=44100,dur=.35,rpm=1600),44100,.35)
