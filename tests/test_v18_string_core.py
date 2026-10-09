import math
import numpy as np
import genny


def test_string_properties_consistent():
    p=genny.StringPhysicalProperties(length_m=.65,radius_m=.00035,material='steel',frequency_hz=440.)
    assert 50 < p.tension_n < 2000
    assert math.isclose(p.ideal_frequency_hz,440.,rel_tol=1e-10)
    assert p.characteristic_impedance_n_s_m > 0
    assert p.flexural_rigidity_n_m2 > 0
    assert p.inharmonicity_B > 0
    assert p.partial_frequency(4) > 4*440.
    assert p.longitudinal_frequency_hz > 440.


def test_material_changes_string_physics():
    s=genny.StringPhysicalProperties.from_frequency(220,material='steel')
    n=genny.StringPhysicalProperties.from_frequency(220,material='nylon')
    assert s.material.young_pa > n.material.young_pa
    assert not math.isclose(s.tension_n,n.tension_n)
    assert s.longitudinal_speed_m_s > n.longitudinal_speed_m_s


def test_dispersive_string_renders_finite_multiple_rates():
    for sr in (22050,44100,48000):
        p=genny.StringPhysicalProperties.from_frequency(220,length_m=.72,radius_m=.00045)
        s=genny.DispersiveString(p,sr=sr)
        y=s.render_pluck(.08,velocity=.7,position=.23)
        assert len(y)==round(.08*sr)
        assert np.isfinite(y).all()
        assert np.max(np.abs(y)) > 1e-5
        assert np.max(np.abs(y)) <= 1.000001


def test_piano_course_exposes_physical_specs():
    p=genny.PolyphonicPiano(sr=22050)
    c=p._course(60)
    assert hasattr(c,'physical_properties')
    assert len(c.physical_properties)==len(c.lines)
    assert all(x.characteristic_impedance_n_s_m>0 for x in c.physical_properties)
    assert c.physical_properties[0].frequency_hz > 200


def test_plucked_string_physical_note():
    inst=genny.PluckedString(kind='guitar')
    y=inst.physical_note(196,.06,sr=22050,velocity=.7)
    assert len(y)==round(.06*22050)
    assert np.isfinite(y).all()
    assert np.max(np.abs(y))>1e-5
