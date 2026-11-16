"""Lot 34 D: winds and percussion lines of Lecture profonde 3.4 that were closed by earlier waves, checked once on the signal.

Run:  uv run --with pytest pytest tests/test_lot34_d.py -q
"""
import inspect

import numpy as np

from genny import acoustic as A
from genny import fdstring, friction, klang, percussion
from genny.analysis import detect_clicks

SR = 44100


def _h(y, f0, k, t0=0.25, n=16384):
    s = y[int(t0 * SR):int(t0 * SR) + n]
    Y = np.abs(np.fft.rfft(s * np.hanning(s.size), 8 * s.size))
    fr = np.fft.rfftfreq(8 * s.size, 1 / SR)
    return Y[(fr > k * f0 * 0.97) & (fr < k * f0 * 1.03)].max()


def _even_odd_db(y, f0):
    ev = sum(_h(y, f0, k) ** 2 for k in (2, 4, 6))
    od = sum(_h(y, f0, k) ** 2 for k in (3, 5, 7))
    return 10 * np.log10(ev / od)


def test_oboe_even_option_lowers_even_harmonics_in_every_register():
    for f0 in (262.0, 440.0, 880.0):
        a = A.oboe(f0, 1.0, SR, 0.8, even=0.0)
        b = A.oboe(f0, 1.0, SR, 0.8, even=-12.0)
        assert not np.isnan(b).any() and len(detect_clicks(b, SR)) == 0
        assert _even_odd_db(b, f0) < _even_odd_db(a, f0) - 6.0


def test_bassoon_meyer_attack_is_slower_in_the_bass_than_default_table_value():
    assert A.attack_ms("bassoon", 100.0, 0.5) > A.attack_ms("bassoon", 400.0, 1.0) > 0
    y = klang.bassoon(110.0, 0.8, SR, 0.7, meyer=1, precursor=0.5)
    assert np.isfinite(y).all() and len(detect_clicks(y, SR)) == 0


def test_bar_options_exist_and_render():
    p = inspect.signature(fdstring.fd_bar).parameters
    assert {"glide", "scan", "scheme", "resonator", "tuning"} <= set(p)
    y = fdstring.fd_bar(440.0, 0.4, SR, 0.8, tuning="xylophone", scheme="implicit", glide=20.0, scan=2.0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0


def test_closed_requests_have_their_parameters():
    assert "sumtone" in inspect.signature(friction.whistling_blades).parameters
    assert "diameter" in inspect.signature(klang.distant_bell).parameters
    assert hasattr(percussion, "PERRIN_MODES") and hasattr(percussion, "triangle_modes")
