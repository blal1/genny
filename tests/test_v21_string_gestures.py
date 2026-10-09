import numpy as np
import genny


def _finite_audio(x, n=None):
    x=np.asarray(x,float)
    assert x.ndim==1
    if n is not None: assert len(x)==n
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x))>1e-8
    assert np.max(np.abs(x))<=1.000001


def test_bend_changes_tension_and_renders():
    p=genny.StringPhysicalProperties.from_frequency(110.0,material='steel')
    s=genny.DispersiveString(p,sr=22050)
    y=s.render_bend(.12,cents=200,velocity=.7)
    _finite_audio(y,round(.12*22050))
    assert 1.24 < s.last_tension_ratio < 1.28  # 2^(400/1200)


def test_vibrato_and_slide_render_at_common_rates():
    for sr in (22050,44100,48000):
        p=genny.StringPhysicalProperties.from_frequency(196.0,material='steel')
        s=genny.DispersiveString(p,sr=sr)
        _finite_audio(s.render_vibrato(.08,depth_cents=20,rate_hz=5.0),round(.08*sr))
        s=genny.DispersiveString(p,sr=sr)
        _finite_audio(s.render_slide(.08,196,246.94,roughness=.5,pressure=.5,seed=1),round(.08*sr))


def test_feedback_fret_buzz_differs_from_plain_fretted():
    p=genny.StringPhysicalProperties.from_frequency(110.0,material='steel')
    fb=genny.FretboardGeometry(scale_length_m=p.length_m,action_m=.0012)
    a=genny.DispersiveString(p,sr=22050).render_fretted(3,.12,velocity=.9,fretboard=fb)
    b=genny.DispersiveString(p,sr=22050).render_fret_buzz_feedback(3,.12,velocity=.9,fretboard=fb,buzz=.9)
    _finite_audio(b)
    assert np.sqrt(np.mean((a-b)**2))>1e-5


def test_legato_tapping_dead_and_pinch():
    p=genny.StringPhysicalProperties.from_frequency(146.83,material='steel')
    fb=genny.FretboardGeometry(scale_length_m=p.length_m)
    funcs=[
        lambda: genny.DispersiveString(p,sr=22050).render_hammer_on(5,.10,velocity=.7,fretboard=fb),
        lambda: genny.DispersiveString(p,sr=22050).render_pull_off(3,.10,velocity=.7,fretboard=fb),
        lambda: genny.DispersiveString(p,sr=22050).render_tapping(12,.10,velocity=.7,fretboard=fb),
        lambda: genny.DispersiveString(p,sr=22050).render_dead_note(.10,velocity=.8),
        lambda: genny.DispersiveString(p,sr=22050).render_pinch_harmonic(4,.10,velocity=.8),
    ]
    for fn in funcs: _finite_audio(fn())


def test_high_level_expressive_facades():
    guitar=genny.PluckedString('guitar')
    for tech,kw in [
        ('bend',{}),('vibrato',{}),('slide',{'end_frequency':246.94}),
        ('hammer_on',{'fret':5}),('pull_off',{'fret':3}),('tapping',{'fret':12}),
        ('dead_note',{}),('pinch_harmonic',{'harmonic':4}),('fret_buzz_feedback',{'fret':3}),
    ]:
        y=guitar.expressive_note(196,.07,technique=tech,sr=22050,**kw)
        _finite_audio(y)
    violin=genny.BowedString(body='violin')
    _finite_audio(violin.vibrato_note(220,.07,sr=22050))
