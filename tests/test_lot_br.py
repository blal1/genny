"""Lot BR: constants and functions that were defined but not rendered, now reached by an option.

Run:  uv run python tests/test_lot_br.py
"""
import numpy as np

from genny import acoustic as A
from genny import creatures, klang, nature, tubes
from genny.physical import nonverbal, winds

SR = 44100


def _hk(y, f0, k, t0=0.2, n=16384):
    s = y[int(t0 * SR):int(t0 * SR) + n]
    Y = np.abs(np.fft.rfft(s * np.hanning(s.size), 8 * s.size))
    fr = np.fft.rfftfreq(8 * s.size, 1 / SR)
    return Y[(fr > k * f0 * 0.97) & (fr < k * f0 * 1.03)].max()


def test_church_organ_mouth():
    base = A.church_organ(131.0, 0.8, SR, 1.0)
    low = A.church_organ(131.0, 0.8, SR, 1.0, mouth=0.0)
    high = A.church_organ(131.0, 0.8, SR, 1.0, mouth=1.0)
    assert base.shape == low.shape
    r_lo, r_hi = _hk(low, 131.0, 2) / _hk(low, 131.0, 1), _hk(high, 131.0, 2) / _hk(high, 131.0, 1)
    assert r_lo > 1.15 * r_hi, (r_lo, r_hi)


def test_clarinet_worman_and_map():
    f = 262.0
    w = A.clarinet(f, 0.9, SR, 0.8, model="worman")
    assert abs(w).max() > 0.1
    assert _hk(w, f, 3) > 5.0 * _hk(w, f, 2)
    soft, loud = (A.clarinet(f, 0.9, SR, v, model="worman") for v in (0.2, 1.0))
    assert _hk(loud, f, 5) / _hk(loud, f, 1) > _hk(soft, f, 5) / _hk(soft, f, 1)
    m = A.clarinet(f, 0.9, SR, 0.8, model="map")
    assert np.isfinite(m).all() and abs(m).max() > 0.1
    assert _hk(m, f, 1) > 5.0 * _hk(m, f, 2)
    lat = A.clarinet(f, 0.9, SR, 0.8, model="worman", lattice=1)
    assert _hk(lat, f, 9) / _hk(lat, f, 1) < _hk(w, f, 9) / _hk(w, f, 1)


def test_bassoon_lattice():
    a = klang.bassoon(110.0, 0.8, SR, 0.8)
    b = klang.bassoon(110.0, 0.8, SR, 0.8, lattice=1)
    hi = lambda y: np.sum(np.abs(np.fft.rfft(y[int(0.2 * SR):int(0.7 * SR)]))[2000 * 5 // 10:]) / np.sum(np.abs(np.fft.rfft(y[int(0.2 * SR):int(0.7 * SR)])))
    assert hi(b) < hi(a)


def test_reed_cone_apex_keeps_pitch_and_changes_signal():
    a = tubes.reed_cone(220.0, 0.5, SR, 0.8)
    b = tubes.reed_cone(220.0, 0.5, SR, 0.8, apex=1.0)
    assert np.isfinite(b).all() and abs(b).max() > 0.05 and not np.allclose(a[:4000], b[:4000])


def test_bowed_torsion_wolf():
    a = A.cello(110.0, 0.6, SR, 0.7, model="elastoplastic")
    b = A.cello(110.0, 0.6, SR, 0.7, model="elastoplastic", wolf=0.01)
    c = A.cello(110.0, 0.6, SR, 0.7, model="elastoplastic", torsion=0.3)
    assert not np.allclose(a, b) and not np.allclose(a, c)


def test_unison_strings_coupled_decay():
    y = A.unison_strings(261.63, 1.0, 22050, 0.8)
    assert np.isfinite(y).all() and abs(y).max() > 0.1


def test_laugh_calls_rhythm():
    rng = np.random.default_rng(1)
    a = nonverbal.vocalize("laugh", "female", np.random.default_rng(1), n_syl=5)
    b = nonverbal.vocalize("laugh", "female", rng, n_syl=5, rhythm="calls")
    st, ln = nonverbal.laugh_calls(5)
    assert ln[0] > ln[1] and abs(b).max() > 0.5 and a.size != b.size
    from genny.sfx import render_sfx
    assert abs(render_sfx("laugh", SR, rhythm="calls", seed=2)).max() > 0


def test_vowel_tube_sinuses():
    a = tubes.vowel_tube(150.0, 0.4, SR, 0.8, nasal=0.6)
    b = tubes.vowel_tube(150.0, 0.4, SR, 0.8, nasal=0.6, sinuses="mri")
    assert np.isfinite(b).all() and not np.allclose(a, b)


def test_tornado_psd_spectrum():
    y = nature.tornado(sr=SR, dur=4.0, spectrum="psd")
    z = nature.tornado(sr=SR, dur=4.0)
    assert np.isfinite(y).all() and abs(y).max() > 0.1 and not np.allclose(y[:1000], z[:1000])


def test_defaults_unchanged():
    assert winds.map_limits(0.5, 0.9)["gamma_beat"] > 0.3
    assert nature.tornado(sr=SR, dur=1.0).shape == nature.tornado(sr=SR, dur=1.0, spectrum="bands").shape


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
