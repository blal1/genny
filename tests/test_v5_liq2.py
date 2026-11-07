import math

import numpy as np

from genny import matter as M
from genny import reverbs as RV
from genny.physical import space as S
from genny.physical import voice as V
from genny.physical.core import SR


def _band_db(p, lo, hi, ref_lo, ref_hi):
    P = np.abs(np.fft.rfft(p)) ** 2
    f = np.fft.rfftfreq(len(p), 1 / SR)
    m = lambda a, b: P[(f >= a) & (f < b)].mean()
    return 10 * math.log10(m(lo, hi) / m(ref_lo, ref_hi))


def test_combustion_peak_and_slope():
    assert M.combustion_peak(0.05) == 200.0
    assert M.combustion_peak(0.5) == 20.0
    assert abs(M.combustion_spectrum(200.0, 200.0) - 0.5) < 1e-12
    # f^-2.5 above the peak: -7.5 dB per octave
    r = M.combustion_spectrum(3200.0, 200.0) / M.combustion_spectrum(1600.0, 200.0)
    assert abs(10 * math.log10(r) + 7.5) < 0.2


def test_combustion_roar_follows_spectrum():
    p = M.combustion_roar(4 * SR, np.random.default_rng(0), 1.0, 0.5, peak=200.0)
    for a in (400.0, 800.0, 1600.0, 3200.0):
        law = 10 * math.log10(M.combustion_spectrum(a, 200.0) / M.combustion_spectrum(200.0, 200.0))
        assert abs(_band_db(p, a / 1.1, a * 1.1, 180.0, 220.0) - law) < 1.5, a


def test_flame_length_option_changes_sound_default_kept():
    a = M.flame(kind="campfire", dur=0.5)
    b = M.flame(kind="campfire", dur=0.5, length=0.0)
    c = M.flame(kind="campfire", dur=0.5, length=0.05)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_pipe_reflection_levine_schwinger():
    assert abs(abs(S.pipe_reflection(0.1)) - (1 - 0.1 ** 2 / 2)) < 1e-3
    assert abs(abs(S.pipe_reflection(1.0)) - 0.694) < 0.01
    # low frequency: R -> -1 with phase of the 0.6133 a end correction
    r = S.pipe_reflection(0.01)
    assert abs(np.angle(-r) + 2 * 0.01 * 0.6133) < 1e-4


def test_pipe_directivity_rear_equals_reflection():
    for ka in (0.5, 1.0, 2.0):
        assert abs(S.pipe_directivity(ka, math.pi) - abs(S.pipe_reflection(ka))) < 1e-9
        assert abs(S.pipe_directivity(ka, 0.0) - 1.0) < 1e-9
    assert S.pipe_directivity(0.05, math.pi / 2) > 0.99


def test_distance_pipe_option():
    x = np.random.default_rng(1).standard_normal(SR // 2)
    a = RV.distance(x, SR, metres=5.0, source_radius=0.05, angle=180.0, air=False)
    b = RV.distance(x, SR, metres=5.0, source_radius=0.05, angle=180.0, air=False, pipe=True)
    assert np.std(b) < 0.5 * np.std(a)
    assert np.array_equal(a, RV.distance(x, SR, metres=5.0, source_radius=0.05, angle=180.0, air=False, pipe=False))


def test_slit_pore_impedance_limits():
    f = np.array([1.0, 1e5])
    z = S.ground_impedance(f, 300.0, "slit", 1.5, 0.4)
    # low frequency: Z -> sqrt(i sigma P0 / (w W)) / (rho c): 45 degrees
    assert abs(np.angle(z[0]) - math.pi / 4) < 0.02
    w = 2 * math.pi
    zl = math.sqrt(300e3 * 101325.0 / (w * 0.4)) / (1.2 * math.sqrt(1.4 * 101325.0 / 1.2))
    assert abs(abs(z[0]) / zl - 1) < 0.02
    # high frequency: Z -> sqrt(T) / W
    assert abs(z[1].real / (math.sqrt(1.5) / 0.4) - 1) < 0.1


def _f0_rms(y):
    s = y[int(0.3 * SR):]
    sp = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    fr = np.fft.rfftfreq(len(s), 1 / SR)
    return fr[1 + np.argmax(sp[1:])], float(np.sqrt(np.mean(s ** 2)))


def test_syrinx_fletcher_sounds_at_300_pa():
    f0, rms = _f0_rms(V.syrinx(300.0, (500.0,), dur=0.6, second_valve=False, membrane="fletcher"))
    _, quiet = _f0_rms(V.syrinx(300.0, (500.0,), dur=0.6, second_valve=False))
    assert rms > 1e-3 and quiet < 1e-5
    assert 400.0 < f0 < 520.0


def test_syrinx_piston_radiation_brighter():
    def centroid(y):
        s = y[int(0.3 * SR):]
        P = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(len(s), 1 / SR)
        return float(np.sum(f * P) / np.sum(P))
    a = V.syrinx(3000.0, (500.0,), dur=0.6, second_valve=False)
    b = V.syrinx(3000.0, (500.0,), dur=0.6, second_valve=False, radiation="piston")
    assert centroid(b) > 1.3 * centroid(a)
