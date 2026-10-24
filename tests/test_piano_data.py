"""Steinway D string table (Chabassier & Durufle 2012) and the stiff-string dispersion allpass design.

Run:  uv run --with pytest pytest tests/test_piano_data.py -q
"""
import math

import numpy as np
from scipy.optimize import brentq

from genny.physical import piano_data as PD


def test_table_is_consistent_and_B_has_its_minimum_at_the_break():
    # every row: f0 = sqrt(T / (rho A)) / 2L within 1 % (the parse kept the columns together)
    for midi, f0, L, d, rho, T, *_ in PD.CHABASSIER_D:
        a = math.pi * (d * 1e-3) ** 2 / 4
        assert abs(math.sqrt(T / (rho * a)) / (2 * L) / f0 - 1) < 0.01, midi
    B = PD.steinway_B(np.arange(24, 108))
    lo = 24 + int(np.argmin(B))
    assert 30 <= lo <= 45                                       # bass/tenor break, not the lowest key
    assert abs(PD.steinway_B(69) - 6.9e-4) < 0.3e-4             # A4
    assert PD.steinway_B(96) > 20 * PD.steinway_B(60)


def _partials(f0, B, sr, a, M, delay):
    w0 = 2 * np.pi * f0 / sr
    N = 2 * np.pi / w0 - delay
    tot = lambda w: N * w + M * PD._ap_phase(w, a)              # noqa: E731
    out = []
    for n in range(1, 12):
        if n * f0 * math.sqrt(1 + B * n * n) > 0.4 * sr:
            break
        wn = brentq(lambda w: tot(w) - 2 * np.pi * n, 1e-6, np.pi - 1e-6)
        out.append(1200 * math.log2(wn * sr / (2 * np.pi) / (n * f0 * math.sqrt(1 + B * n * n) / math.sqrt(1 + B))))
    return out


def test_dispersion_allpass_places_the_partials():
    for sr in (44100, 48000):
        for midi in (36, 60, 72, 84, 96):
            f0, B = 440 * 2 ** ((midi - 69) / 12), PD.steinway_B(midi)
            a, delay, M = PD.dispersion_allpass(f0, B, sr)
            assert -1 < a < 0 and M > 0 and sr / f0 - delay >= 3
            err = _partials(f0, B, sr, a, M, delay)
            assert abs(err[0]) < 0.05 and max(map(abs, err)) < 8, (sr, midi, M, err)
    # Rauhala's closed form (M = 8, 44.1 kHz, B <= 1e-3) is close to the fitted optimum
    f0, B = 261.63, PD.steinway_B(60)
    a = PD.rauhala_coefficient(f0, B, 8)
    _, delay, _ = PD.dispersion_allpass(f0, B, 44100, 8)
    assert max(map(abs, _partials(f0, B, 44100, a, 8, 8 * PD._ap_phase(2 * np.pi * f0 / 44100, a) / (2 * np.pi * f0 / 44100)))) < 3
    assert PD.dispersion_allpass(100.0, 0.0, 44100) == (0.0, 0.0, 0)


def test_rendered_piano_partials_follow_the_steinway_scale():
    from genny.piano import PolyphonicPiano
    sr = 48000
    for midi, partials, tol in ((36, range(2, 9), 1.5), (60, (2, 3, 4, 6, 7, 8), 3.0)):
        p = PolyphonicPiano(sr=sr)
        p.note_on(midi, 0.8)
        y = np.asarray(p.render(2.0))
        y = y.mean(axis=1) if y.ndim == 2 else y
        seg = y[int(0.3 * sr):int(1.8 * sr)]
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 20))
        f = np.fft.rfftfreq(1 << 20, 1 / sr)
        f0, B = 440 * 2 ** ((midi - 69) / 12), PD.steinway_B(midi)
        for n in partials:
            ft = n * f0 * math.sqrt(1 + B * n * n) / math.sqrt(1 + B)
            band = (f > ft * 0.985) & (f < ft * 1.015)
            k = np.flatnonzero(band)[np.argmax(S[band])]
            assert abs(1200 * math.log2(f[k] / ft)) < tol, (midi, n, 1200 * math.log2(f[k] / ft))
