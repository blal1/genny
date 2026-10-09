import numpy as np
import genny
from genny.strings import StringPhysicalProperties, WoundStringPhysicalProperties, DispersiveString, StringExciter, FretboardGeometry


def finite_audio(y,n=None):
    y=np.asarray(y,float); assert y.ndim==1 and len(y)>0; assert np.isfinite(y).all(); assert np.max(np.abs(y))>1e-8
    if n is not None: assert len(y)==n


def test_exciter_hardness_changes_contact_width():
    soft=StringExciter('finger',.1,.2).pulse(48000)
    hard=StringExciter('pick',.95,.2).pulse(48000)
    assert len(soft)>len(hard)>=2


def test_fret_geometry_equal_temperament():
    fb=FretboardGeometry(.648)
    assert abs(fb.vibrating_length(12)-.324)<1e-9
    assert abs(fb.frequency_ratio(12)-2.0)<1e-12


def test_fretted_and_buzz_render():
    p=StringPhysicalProperties.from_frequency(110,length_m=.648,material='steel')
    s=DispersiveString(p,sr=44100)
    y=s.render_fretted(5,.12,velocity=.8)
    finite_audio(y,round(.12*44100))
    s=DispersiveString(p,sr=44100)
    b=s.render_fret_buzz(5,.12,velocity=1.0,buzz=.9)
    finite_audio(b,len(y)); assert np.mean((b-y)**2)>1e-8


def test_palm_mute_reduces_tail_energy():
    p=StringPhysicalProperties.from_frequency(196,length_m=.648,material='steel')
    a=DispersiveString(p,sr=48000).render_exciter(.35,StringExciter('pick',.7,.2),velocity=.8,palm_mute=0)
    b=DispersiveString(p,sr=48000).render_exciter(.35,StringExciter('pick',.7,.2),velocity=.8,palm_mute=.9)
    tail=slice(int(.22*48000),None)
    assert np.mean(b[tail]**2)<np.mean(a[tail]**2)


def test_glissando_is_finite_at_common_rates():
    for sr in (22050,44100,48000):
        p=StringPhysicalProperties.from_frequency(110,length_m=.8,material='steel')
        y=DispersiveString(p,sr=sr).render_glissando(.10,110,165,velocity=.7)
        finite_audio(y,round(.10*sr))


def test_harmonic_and_slap_wound_string():
    p=WoundStringPhysicalProperties.from_frequency(55,length_m=.864)
    h=DispersiveString(p,sr=44100).render_natural_harmonic(3,.12)
    s=DispersiveString(p,sr=44100).render_slap(.12,velocity=.9)
    finite_audio(h); finite_audio(s)


def test_high_level_local_api_and_wrappers():
    pl=genny.PluckedString('guitar')
    for y in (pl.local_note(220,.08,technique='pick',sr=44100),
              pl.local_note(220,.08,technique='fret_buzz',fret=3,buzz=.7,sr=44100),
              pl.local_note(220,.08,technique='harmonic',harmonic=2,sr=44100),
              pl.glissando(220,330,.08,sr=44100),
              genny.BowedString('violin').glissando(220,330,.08,sr=44100)):
        finite_audio(y)
