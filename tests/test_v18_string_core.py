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


def test_rauhala_dispersion_stretches_the_partials_like_a_stiff_string():
    import math
    from scipy.optimize import brentq
    from genny.strings import rauhala_dispersion
    sr = 48000
    def lag(w, coefs):
        return sum(-(math.atan2(-math.sin(w), a + math.cos(w)) - math.atan2(-a * math.sin(w), 1 + a * math.cos(w))) for a in coefs)
    for f0, B in ((82.4, 1.5e-5), (196.0, 1e-4), (65.4, 2e-4), (261.6, 3e-4)):
        c = rauhala_dispersion(f0, B, sr)
        assert len(c) == 4 and all(-1 < a < 0 for a in c)
        w0 = 2 * math.pi * f0 / sr
        N = (2 * math.pi - lag(w0, c)) / w0
        for k in (5, 10, 15):
            fk = brentq(lambda w: w * N + lag(w, c) - 2 * math.pi * k, 1e-6, 3.1) * sr / (2 * math.pi)
            want = k * f0 * math.sqrt((1 + B * k * k) / (1 + B))
            assert abs(1200 * math.log2(fk / want)) < 8.0, (f0, B, k)
    assert rauhala_dispersion(440.0, 0.0, sr) == ()


def test_tension_modulation_glides_down_to_the_unmodulated_pitch():
    from genny.strings import DispersiveString, StringPhysicalProperties
    sr = 48000
    def cents(y, t0, t1, f):
        seg = y[int(t0 * sr):int(t1 * sr)]
        Y = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
        fr = np.fft.rfftfreq(8 * len(seg), 1 / sr)
        m = (fr > f * .9) & (fr < f * 1.1)
        return 1200 * np.log2(fr[m][np.argmax(Y[m])] / f)
    p = StringPhysicalProperties.from_frequency(82.4, material="steel")
    off = DispersiveString(p, sr=sr).render_pluck(4.0, velocity=.9, position=.2)
    on = DispersiveString(p, sr=sr).render_pluck(4.0, velocity=.9, position=.2, glide_cents=30)
    assert np.isfinite(on).all()
    assert cents(on, .05, .35, 82.4) > cents(off, .05, .35, 82.4) + 10      # starts sharp
    assert abs(cents(on, 3.0, 3.9, 82.4) - cents(off, 3.0, 3.9, 82.4)) < 1.0  # and settles where it would have been
    assert np.array_equal(off, DispersiveString(p, sr=sr).render_pluck(4.0, velocity=.9, position=.2, glide_cents=0))
