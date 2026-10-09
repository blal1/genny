import numpy as np
import genny
from genny import instruments as I


def test_wound_string_mass_and_pitch():
    w=genny.WoundStringPhysicalProperties.from_frequency(55.0,length_m=.864)
    plain=genny.StringPhysicalProperties.from_frequency(55.0,length_m=.864,radius_m=w.core_radius_m)
    assert w.linear_density_kg_m > plain.linear_density_kg_m
    assert abs(w.ideal_frequency_hz-55.0) < 1e-8
    assert w.outer_radius_m > w.core_radius_m
    assert w.helix_length_factor > 1
    assert w.characteristic_impedance_n_s_m > 0


def test_wound_material_changes_mass_and_rigidity():
    n=genny.WoundStringPhysicalProperties.from_frequency(82.41,winding_material='nickel')
    b=genny.WoundStringPhysicalProperties.from_frequency(82.41,winding_material='phosphor_bronze')
    assert not np.isclose(n.linear_density_kg_m,b.linear_density_kg_m)
    assert not np.isclose(n.flexural_rigidity_n_m2,b.flexural_rigidity_n_m2)


def test_shared_core_pluck_and_bow_finite():
    p=genny.StringPhysicalProperties.from_frequency(196.0,material='steel')
    for sr in (22050,44100,48000):
        a=genny.DispersiveString(p,sr=sr).render_pluck(.06,velocity=.7)
        b=genny.DispersiveString(p,sr=sr).render_bow(.06,bow_velocity=.18,bow_pressure=.7)
        assert len(a)==round(.06*sr)==len(b)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.max(np.abs(a))>1e-5 and np.max(np.abs(b))>1e-5


def test_public_string_wrappers_use_shared_core_paths():
    for name,f in [('guitar',196.),('harp',261.63),('mandolin',392.),('koto',293.66),('sitar',220.),('upright_bass',55.),('finger_bass',82.41),('violin',440.),('cello',146.83)]:
        y=I.render_note(name,f,.08,22050,.75)
        assert y.ndim==1 and np.isfinite(y).all(), name
        assert np.max(np.abs(y))>1e-5, name
