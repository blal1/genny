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
