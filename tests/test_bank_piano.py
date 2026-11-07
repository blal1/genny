"""Modal piano after Bank, Zambon & Fontana 2010 (genny.physical.bank_piano, the `piano` instrument).

Run:  uv run --with pytest pytest tests/test_bank_piano.py -q
"""
import math

import numpy as np

from genny.physical import bank_piano as BP
from genny.physical import piano_data as PD

SR = 44100


def _peak_cents(y, f_target):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 20))
    f = np.fft.rfftfreq(1 << 20, 1 / SR)
    band = (f > f_target * 0.985) & (f < f_target * 1.015)
    k = np.flatnonzero(band)[np.argmax(S[band])]
    return 1200 * math.log2(f[k] / f_target)


def test_partials_follow_the_steinway_stiff_string():
    for midi in (36, 60):
        f0, B = 440 * 2 ** ((midi - 69) / 12), PD.steinway_B(midi)
        y = BP.note(f0, 2.0, SR, 0.7, tail=0.0)[int(0.3 * SR):]
        for n in (2, 3, 5, 7):
            ft = n * f0 * math.sqrt(1 + B * n * n) / math.sqrt(1 + B)
            assert abs(_peak_cents(y, ft)) < 3.0, (midi, n)


def test_longitudinal_share_grows_with_hammer_speed(monkeypatch):
    # the tension term is quadratic in the transverse amplitude: its share of the bridge force rises with velocity
    f0 = 440 * 2 ** ((36 - 69) / 12)
    share = []
    for vel in (0.2, 1.0):
        full = BP.note(f0, 0.6, SR, vel, tail=0.0)
        monkeypatch.setattr(BP, "LONG_COUPLING", 0.0)
        trans = BP.note(f0, 0.6, SR, vel, tail=0.0)
        monkeypatch.undo()
        share.append(np.sum((full - trans) ** 2) / np.sum(trans ** 2))
    assert share[1] > 4 * share[0], share


def test_trichord_has_a_two_stage_decay():
    f0 = 440 * 2 ** ((60 - 69) / 12)
    assert BP.strings_per_key(60) == 3
    y = BP.note(f0, 11.0, SR, 0.7, tail=0.0)
    from scipy.signal import butter, sosfiltfilt
    env = np.abs(sosfiltfilt(butter(2, [f0 * 0.97, f0 * 1.03], "bandpass", fs=SR, output="sos"), y))
    t = np.arange(0.0, 10.5, 0.5)
    db = np.array([20 * np.log10(env[int(a * SR):int((a + 0.5) * SR)].max() + 1e-12) for a in t])
    rate = lambda a, b: -np.polyfit(t[(t >= a) & (t < b)], db[(t >= a) & (t < b)], 1)[0]  # noqa: E731  dB/s
    early, late = rate(0.0, 1.5), rate(3.0, 10.5)  # prompt sound, then the aftersound (beats average out)
    assert early > 2 * late > 0, (early, late)


def test_fxi_matches_bank_eq_5_36():
    """F_xi,k = F+ + F- of Bank's thesis eqs 5.36a-b, summed by brute force."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=30)
    nk = np.arange(1.0, 31.0)
    ES, L = 2.0e4, 1.2
    c = -ES * math.pi ** 3 / (8 * L * L)
    Y = lambda n: y[n - 1] if 1 <= n <= 30 else 0.0  # noqa: E731
    for k in (1, 2, 3, 4, 7, 10):
        fp = c * sum(Y(k - n) * Y(n) * k * (k - n) * n for n in range(1, k))
        fm = 2 * c * sum(Y(k + n) * Y(n) * k * (k + n) * n for n in range(1, 31))
        assert math.isclose(BP._fxi(y, nk, k, 2 * c), fp + fm, rel_tol=1e-12), k


def test_long_full_renders_and_differs():
    a = BP.note(110.0, 0.4, SR, 0.9, tail=0.1)
    b = BP.note(110.0, 0.4, SR, 0.9, tail=0.1, long_full=True)
    assert np.all(np.isfinite(b)) and a.shape == b.shape
    assert 1e-6 < np.max(np.abs(a - b)) < 0.5 * np.max(np.abs(a))


def test_parallel_filter_poles_and_least_squares_fit():
    from scipy.signal import lfilter
    a1, a2 = BP.parallel_poles(SR, 100, 0.98, 30.0, SR / 2)
    assert abs(math.sqrt(a2[-1]) - 0.98) < 1e-12           # |p| = R at fs/2
    rng = np.random.default_rng(0)
    d0, d1 = rng.normal(size=100), rng.normal(size=100)
    imp = np.zeros(4000)
    imp[0] = 1.0
    h = sum(lfilter([d0[k], d1[k]], [1, a1[k], a2[k]], imp) for k in range(100)) + 3.0 * imp
    fit = BP.fit_parallel_filter(h, SR, f_hi=SR / 2)
    y = BP.soundboard(imp, SR, fit=fit)
    assert fit[4].size == 1 and abs(fit[4][0] - 3.0) < 1e-6         # M = 0 keeps the direct term b_0
    assert np.linalg.norm(y - h) / np.linalg.norm(h) < 1e-9
    # a response made of other poles (half-way between the fitted ones on the log scale): no exact solution
    th = 2.0 * np.pi * np.geomspace(30.0, SR / 2, 100) / SR * (735.0 ** (1 / 99)) ** 0.5
    p = 0.98 ** (th / np.pi) * np.exp(1j * th)
    h = sum(lfilter([d0[k], d1[k]], [1, -2 * p[k].real, abs(p[k]) ** 2], imp) for k in range(99))
    y = BP.soundboard(imp, SR, fit=BP.fit_parallel_filter(h, SR, f_hi=SR / 2))
    assert np.linalg.norm(y - h) / np.linalg.norm(h) < 0.07                 # measured 0.059


def test_tuning_factor_and_inharmonicity_threshold():
    c = BP.unison_cents(0.999)
    assert abs(c[0] + 0.866) < 2e-3 and c[1] == 0.0 and abs(c[2] - 0.865) < 2e-3
    assert abs(PD.inharmonicity_threshold(440.0) / 1.0733e-4 - 1) < 1e-3
    assert PD.inharmonicity_audible(440.0, PD.steinway_B(69)) and not PD.inharmonicity_audible(3520.0, PD.steinway_B(105))
    a = BP.note(261.63, 0.3, SR, 0.6, tail=0.2)
    b = BP.note(261.63, 0.3, SR, 0.6, tail=0.2, tuning_factor=0.99)
    assert np.all(np.isfinite(b)) and np.max(np.abs(a - b)) > 1e-3 * np.max(np.abs(a))
