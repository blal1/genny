import numpy as np

from genny.physical import bank_piano as bp

SR = 44100


def test_orth_radiation_profile_follows_measured_transfer():
    g = bp.orth_radiation_gain(np.array([400.0, 1000.0, 3000.0]))
    db = 20 * np.log10(g)
    assert -30 < db[0] < -22
    assert -16 < db[1] < -12
    assert abs(db[2]) < 1e-9
    assert bp.ORTH_RADIATION == 10 ** (-10.0 / 20.0)


def test_orth_profile_option_dims_the_horizontal_plane_in_the_bass():
    kw = dict(dur=0.4, sr=SR, tail=0.5, polarisations=2)
    flat = bp.note(110.0, **kw)
    prof = bp.note(110.0, orth_profile=True, **kw)
    one = bp.note(110.0, dur=0.4, sr=SR, tail=0.5, polarisations=1)
    n = min(flat.size, prof.size, one.size)
    d_flat = np.sqrt(np.mean((flat[:n] - one[:n]) ** 2))
    d_prof = np.sqrt(np.mean((prof[:n] - one[:n]) ** 2))
    assert d_prof < d_flat
