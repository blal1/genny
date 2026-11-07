"""Divers items of the deep-reading report (lecture profonde): measured checks of the sourced numbers.

Run:  uv run --with pytest pytest tests/test_divers.py -q
"""
import numpy as np

from genny import filters as F
from genny import fx as FX

SR = 48000


def _response(h, sr=SR):
    H = np.abs(np.fft.rfft(h))
    f = np.fft.rfftfreq(len(h), 1 / sr)
    return f, H


def test_lpf18_three_poles_and_resonance_at_cutoff():
    imp = np.zeros(1 << 15)
    imp[0] = 1e-4                                         # small: the tanh stages stay linear
    f, H = _response(F.ladder(imp, 1000.0, SR, 0.0, "lpf18", 2.0))
    db = lambda hz: 20 * np.log10(H[np.argmin(np.abs(f - hz))])   # noqa: E731
    slope = db(8000.0) - db(4000.0)
    assert -21.0 < slope < -16.0, slope                   # 3 poles: 18 dB per octave
    f, H = _response(F.ladder(imp, 1000.0, SR, 0.8, "lpf18", 2.0))
    assert abs(f[np.argmax(H)] / 1000.0 - 1.0) < 0.1, f[np.argmax(H)]
    loud = FX.ladder(np.sin(2 * np.pi * 110 * np.arange(SR) / SR), SR, cutoff=800, res=0.8, kind="lpf18", drive=2.0)
    assert np.all(np.isfinite(loud)) and np.max(np.abs(loud)) <= 1.0      # tanh output


def test_shaker_comb_taps_and_gain_range():
    x = np.zeros(SR)
    x[0] = 1.0
    y = FX.shaker_comb(x, SR, rate=4.0, ramp=0.2, seed=3)
    nz = np.flatnonzero(np.abs(y) > 1e-12)
    assert set(nz) <= {0, int(0.011 * SR), int(0.017 * SR), int(0.030 * SR)}, nz     # taps 0, 11, 17, 30 ms
    clicks = np.zeros(2 * SR)
    clicks[:: SR // 10] = 1.0                              # 10 Hz probe: read the four gains over time
    y = FX.shaker_comb(clicks, SR, rate=8.0, ramp=0.05, seed=1)
    g = np.array([y[k: k + int(0.031 * SR) + 1][[0, int(0.011 * SR), int(0.017 * SR), int(0.030 * SR)]] for k in range(0, 2 * SR - SR // 10, SR // 10)])
    assert g.min() >= -0.7 - 1e-9 and g.max() <= 1.3 + 1e-9, (g.min(), g.max())       # 2 U(0..999)/1000 - 0.7
    assert np.std(g[:, 0]) > 0.1, "the gains keep moving"


def test_width_ms_mode():
    rng = np.random.default_rng(0)
    st = rng.standard_normal((4800, 2))
    assert np.allclose(FX.width(st, SR, amount=0.5, mode="ms"), st), "0.5 is unchanged"
    m = FX.width(st, SR, amount=0.0, mode="ms")
    assert np.allclose(m[:, 0], m[:, 1])
    mono = rng.standard_normal(4800)
    w = FX.width(mono, SR, amount=1.0, mode="ms")
    assert np.allclose(w[:, 0], mono) and np.allclose(w[:, 1], mono), "a mono input stays mono"


def test_acid_phase_sync_resonance_at_cutoff_harmonic():
    from genny import instruments as I
    f = 110.0

    def harm(res):
        y = I.render_note("acid", f, 1.0, SR, 0.8, model="phase_sync", cutoff=10 * f, env=0, res=res)[int(0.2 * SR): int(0.8 * SR)]
        H = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        a = np.array([H[np.argmin(np.abs(fr - k * f))] for k in range(1, 16)])
        return 20 * np.log10(a / a[0])
    on, off = harm(0.8), harm(0.0)
    assert on[9] - off[9] > 15.0, (on[9], off[9])          # "accentuated harmonics around 10f" with rf = 10
    assert np.argmax(on[8:]) + 9 in (10, 11), on[8:]


def test_osc_classic_jumps_and_corners():
    from genny import osc as O
    n, f = SR, 100.0                                       # 100 Hz at 48 kHz: 480 samples a cycle, whole cycles in 1 s

    def amps(y):
        H = np.abs(np.fft.rfft(y)) * 2 / len(y)
        return np.array([H[int(k * f)] for k in range(1, 8)])
    k = np.arange(1, 8)
    sq = amps(O.classic(f, n, SR, jumps=((0.0, 1.0), (0.5, -1.0))))
    assert np.allclose(sq[::2], 2 / (np.pi * k[::2]), rtol=1e-6) and np.all(sq[1::2] < 1e-9)      # square: (2/pi)/k, odd only
    tri = amps(O.classic(f, n, SR, jumps=(), corners=((0.25, -8.0), (0.75, 8.0))))
    assert np.allclose(tri[::2], 8 / (np.pi ** 2 * k[::2] ** 2), rtol=1e-6) and np.all(tri[1::2] < 1e-9)   # triangle: (8/pi^2)/k^2
    t = O.classic(f, n, SR, jumps=(), corners=((0.25, -8.0), (0.75, 8.0)))
    assert abs(np.max(t) - 1.0) < 0.01                     # peak 1 for slopes +-4 per cycle
    saw = O.classic(f, n, SR)
    assert np.mean(np.diff(saw[100:380])) < 0, "a rise of 1 at 0, falling ramp in between (slope -sum d)"
