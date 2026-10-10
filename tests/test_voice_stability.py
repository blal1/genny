"""A held note has one pitch source: it sits on the written pitch, carries no hum under it and does not flutter.

These are the measurements that found the faults fixed on 2026-10-10: a flute an octave low over a 60.7 Hz buzz, hybrid
brass and strings beating against their own second source, a string section 15 cents off that could not speak a
sixteenth note.
"""
import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

import genny.cli  # noqa: F401  registers every instrument
from genny import instruments as I

SR = 48000
FR = np.fft.rfftfreq(1 << 18, 1 / SR)


def held(name, freq, **params):
    y = np.asarray(I.render_note(name, freq, 2.0, SR, 0.8, **params), float)
    s = y[int(0.7 * SR):int(1.8 * SR)]
    p = np.abs(np.fft.rfft(s * np.hanning(len(s)), 1 << 18)) ** 2
    band = np.flatnonzero((FR > freq * 0.45) & (FR < freq * 2.2))
    cents = 1200 * np.log2(FR[band[np.argmax(p[band])]] / freq)
    cents = cents - 1200 * round(cents / 1200)                       # the octave partial may be the strongest
    under = p[(FR > 25) & (FR < 0.6 * freq)].sum() / p[FR > 25].sum()
    env = sosfiltfilt(butter(4, 30, fs=SR, output="sos"), np.abs(hilbert(s)))[int(0.15 * SR):-int(0.15 * SR)]
    return cents, under, env.std() / env.mean()                      # the filter's edges are left out


@pytest.mark.parametrize("name,freq", [("flute", 440.0), ("flute", 1046.5), ("flute", 2093.0), ("piccolo", 2093.0), ("recorder", 784.0),
                                       ("trumpet", 392.0), ("french_horn", 261.63), ("violin", 440.0), ("cello", 146.83)])
def test_held_note_sits_on_its_pitch_with_nothing_under_it(name, freq):
    cents, under, wobble = held(name, freq)
    assert abs(cents) < 6, f"{name} at {freq} Hz is {cents:+.1f} cents off"
    assert under < 0.01, f"{name} carries {100 * under:.1f} % of its energy under the note"
    assert wobble < 0.15, f"{name} wobbles by {100 * wobble:.0f} %"                # a player's vibrato moves the level a little


def test_the_flute_is_steady_without_vibrato_and_speaks_without_breath():
    assert held("flute", 659.26, vibrato=0.0)[2] < 0.03
    assert np.sqrt(np.mean(np.asarray(I.render_note("flute", 440.0, 1.0, SR, 0.8, breath=0.0)) ** 2)) > 0.02


def test_the_string_section_is_centred_and_bows_a_short_note():
    cents, under, wobble = held("strings", 440.0)
    assert abs(cents) < 6 and under < 0.01 and wobble < 0.15
    short = np.asarray(I.render_note("strings", 440.0, 0.05, SR, 0.8), float)
    long = np.asarray(I.render_note("strings", 440.0, 1.0, SR, 0.8), float)
    assert np.abs(short).max() > 0.3 * np.abs(long).max()          # a 50 ms note reaches a third of a held note's level
