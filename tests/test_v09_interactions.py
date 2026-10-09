import numpy as np
import genny


def _audio_ok(x, sr, dur):
    x=np.asarray(x)
    assert x.ndim==1 and np.all(np.isfinite(x))
    assert abs(len(x)/sr-dur) < .05
    assert np.max(np.abs(x)) > 1e-7


def _port_ok(p, domain):
    assert p.domain==domain
    assert p.effort.shape==p.flow.shape
    assert np.all(np.isfinite(p.effort)) and np.all(np.isfinite(p.flow))
    assert np.all(np.isfinite(p.power))


def test_hammer_string_interface_and_piano_wrapper():
    r=genny.HammerStringInteraction().solve(440,.12,velocity=.7,sr=44100,seed=1)
    _audio_ok(r.audio,44100,.12); _port_ok(r.drive_port,"mechanical"); _port_ok(r.response_port,"mechanical")
    assert r.metadata["felt_exponent"] > 2
    _audio_ok(genny.Piano().note(440,.12,sr=44100),44100,.12)


def test_bow_string_interface_and_wrapper():
    r=genny.BowStringInteraction().solve(220,.15,sr=48000,seed=2)
    _audio_ok(r.audio,48000,.15); _port_ok(r.drive_port,"mechanical"); _port_ok(r.response_port,"mechanical")
    _audio_ok(genny.BowedString().note(220,.15,sr=22050),22050,.15)


def test_air_column_interfaces():
    for obj,f in [(genny.ReedBoreInteraction(),220),(genny.LipBoreInteraction(),196)]:
        r=obj.solve(f,.15,sr=44100,seed=3)
        _audio_ok(r.audio,44100,.15); _port_ok(r.drive_port,"acoustic"); _port_ok(r.response_port,"acoustic")
        assert r.metadata["bore_impedance"] > 0
    _audio_ok(genny.Clarinet().note(220,.15,sr=44100),44100,.15)
    _audio_ok(genny.Brass().note(196,.15,sr=44100),44100,.15)


def test_syrinx_interface_and_wrapper():
    r=genny.SyrinxTracheaInteraction().solve((650,658),.12,sr=44100)
    _audio_ok(r.audio,44100,.12); _port_ok(r.drive_port,"acoustic"); _port_ok(r.response_port,"acoustic")
    assert r.metadata["trachea_impedance"] > 0
    _audio_ok(genny.BirdSyrinx().call(650,.12,sr=22050),22050,.12)


def test_passive_coupler_quadratic_norm_does_not_grow():
    rng=np.random.default_rng(0); n=512
    a=genny.PhysicalPort.mechanical(rng.normal(size=n),rng.normal(size=n),"a")
    b=genny.PhysicalPort.mechanical(rng.normal(size=n),rng.normal(size=n),"b")
    c=genny.PassiveCoupler(.63)
    ao,bo=c.connect(a,b)
    before=np.sum(a.effort*a.effort+b.effort*b.effort+a.flow*a.flow+b.flow*b.flow)
    after=np.sum(ao.effort*ao.effort+bo.effort*bo.effort+ao.flow*ao.flow+bo.flow*bo.flow)
    assert after <= before*(1+1e-12)
