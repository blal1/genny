"""Every instrument and drum renders cleanly, in tune, and without piercing highs.

Run:  uv run python tests/test_instruments.py     (plain asserts, no test framework needed)
"""
import numpy as np

from genny import drums as D
from genny import instruments as I
from genny.analysis import brightness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100


def cents_off(y, f0):
    """Pitch error of a rendered note, from the autocorrelation peak next to the expected period."""
    y = y[int(0.05 * SR):int(0.05 * SR) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = SR / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((SR * 4 / k) / f0)


def test_shapes_and_levels():
    for name, e in I.REGISTRY.items():
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 4.0, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[-1]) < 1e-3, (name, "ends on a step")
    for name in D.REGISTRY:
        for vel, sr in ((1.0, 44100), (0.2, 22050)):
            y = D.render_drum(name, sr, vel)
            assert y.ndim == 1 and np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 2.0, name


def test_tuning():
    # ensembles and beating / inharmonic voices are detuned on purpose; everything else must be within 10 cents
    loose = {"choir", "pad", "strings", "tubular_bell", "singing_bowl", "gamelan", "mandolin", "acid", "wobble", "didgeridoo", "piano"}
    # church and china bells are tuned on one stated partial of a strongly inharmonic spectrum, which an
    # autocorrelation pitch cannot follow; tests/test_foley.py checks that partial against the mode table
    inharmonic = {"church_bell", "china_bell"}
    # the chips' noise channels take a note only to choose a noise rate (tests/test_chiptune.py checks those rates)
    inharmonic |= {"nes_noise", "gb_noise"}
    for name, e in I.REGISTRY.items():
        if name in inharmonic:
            continue
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in (lo, (lo + hi) // 2, hi):
            c = cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9), midi_to_freq(m))
            assert abs(c) < (25 if name in loose else 10), (name, m, round(float(c), 1))


def test_not_piercing():
    """In its own register no instrument may be much sharper than a pure tone of the same note, nor carry
    real energy above 5 kHz. (`synth` is the raw building block: what you set is what you get.)"""
    for name, e in I.REGISTRY.items():
        if name == "synth":
            continue
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):   # up to C7; above that even a sine is sharp
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))


def test_level_matched():
    import math
    from genny.analysis import loudness
    for name, e in I.REGISTRY.items():
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        levels = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
                  for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        assert abs(sum(levels) / 3 + 12.0) < 1.0, (name, [round(v, 1) for v in levels], "run examples/level_match.py")
    assert math.isclose(I.key_gain(261.63), 1.0)


if __name__ == "__main__":
    for fn in (test_shapes_and_levels, test_tuning, test_not_piercing, test_level_matched):
        fn()
        print("ok", fn.__name__)
