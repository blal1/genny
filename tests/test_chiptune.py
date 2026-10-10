"""The chip models against the numbers of their hardware documentation (out/research/chips)."""
import numpy as np
import pytest

import genny.chiptune as C
import genny.spec as SP
from genny import instruments as I

SR = 48000


def peak_hz(y, lo=30.0, hi=20000.0):
    y = np.asarray(y, float)
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 19))
    fr = np.fft.rfftfreq(1 << 19, 1 / SR)
    band = (fr > lo) & (fr < hi)
    return fr[band][np.argmax(p[band])]


def harmonic(y, f, k):
    y = np.asarray(y, float)
    t = np.arange(len(y)) / SR
    return abs(np.sum(y * np.hanning(len(y)) * np.exp(-2j * np.pi * k * f * t)))


def repeat_hz(y):
    """The repeat rate of a looping noise: the first lag at which it matches itself again."""
    y = np.asarray(y, float)[:1 << 15]
    r = np.fft.irfft(np.abs(np.fft.rfft(y - y.mean(), 1 << 16)) ** 2)[:4000]
    return SR / (np.flatnonzero(r[30:] > 0.8 * r[30:].max())[0] + 30)


def raw(name, freq, dur=0.6, **params):
    y = np.asarray(I.REGISTRY[name]["fn"](freq, dur, SR, 1.0, mode="hardware_accurate", **params), float)
    return y[int(0.05 * SR):int((dur - 0.05) * SR)]


def test_nes_mixer_is_the_documented_formula():
    assert C.NES_PULSE_TAB[0] == 0 and C.NES_PULSE_TAB[30] == pytest.approx(95.88 / (8128 / 30 + 100))
    tnd = C.NES_TND_TAB.reshape(16, 16, 128)
    assert tnd[0, 0, 0] == 0
    assert tnd[15, 15, 127] == pytest.approx(159.79 / (1 / (15 / 8227 + 15 / 12241 + 127 / 22638) + 100))
    assert C.NES_PULSE_TAB[30] + tnd[15, 15, 127] == pytest.approx(1.0, abs=0.01)      # the full range is 0..1
    assert tnd[15, 0, 127] - tnd[0, 0, 127] < 0.75 * tnd[15, 0, 0]                    # a high DMC level quietens the triangle


@pytest.mark.parametrize("freq", [110.0, 440.0, 1760.0])
def test_nes_pitches_are_the_timer_pitches(freq):
    cpu = C.NES_CPU["ntsc"]
    t = round(cpu / (16 * freq) - 1)
    assert peak_hz(raw("nes_pulse", freq)) == pytest.approx(cpu / (16 * (t + 1)), rel=2e-4, abs=0.1)
    t = round(cpu / (32 * freq) - 1)
    assert peak_hz(raw("nes_triangle", freq)) == pytest.approx(cpu / (32 * (t + 1)), rel=2e-4, abs=0.1)


def test_nes_duty_cycles():
    f = C.NES_CPU["ntsc"] / (16 * (round(C.NES_CPU["ntsc"] / (16 * 220.0) - 1) + 1))
    half, quarter, eighth = (raw("nes_pulse", 220.0, duty=d) for d in (2, 1, 0))
    assert harmonic(half, f, 2) < 0.02 * harmonic(half, f, 1)                         # 50 %: no even harmonics
    assert harmonic(quarter, f, 4) < 0.02 * harmonic(quarter, f, 1)                   # 25 %: no 4th
    assert harmonic(eighth, f, 8) < 0.02 * harmonic(eighth, f, 1) and harmonic(eighth, f, 4) > 0.3 * harmonic(eighth, f, 1)
    assert harmonic(raw("nes_pulse", 220.0, duty=3), f, 1) == pytest.approx(harmonic(quarter, f, 1), rel=0.02)   # 75 % is 25 % negated


def test_nes_short_noise_repeats_every_93_steps():
    # APU Noise, the table of 93-step pitches: register $84 repeats at 300.7 Hz, $88 at 95.3 Hz
    hz = repeat_hz(raw("nes_noise", 440.0, dur=1.0, period=4, short=True))
    assert hz == pytest.approx(300.7, rel=0.02) or hz == pytest.approx(27965.2 / 31, rel=0.02)   # 93 or 31 steps
    assert np.std(raw("nes_noise", 440.0, period=4)) > 0.01                           # the long mode is noise too


def test_nes_hardware_envelope_and_sweep():
    y = np.asarray(I.REGISTRY["nes_pulse"]["fn"](440.0, 1.5, SR, 1.0, decay=3, mode="hardware_accurate"), float)
    env = np.array([np.abs(y[int(t * SR):int((t + 0.02) * SR)]).max() for t in (0.01, 0.5, 1.2)])
    assert env[0] > 1.8 * env[1] > 0 and env[2] < 0.02 * env[0]                       # 15 -> 0 in 16 * 4 / 240 = 1.07 s
    up = np.asarray(I.REGISTRY["nes_pulse"]["fn"](220.0, 0.6, SR, 1.0, sweep=[1, 3, 1], mode="hardware_accurate"), float)
    assert peak_hz(up[int(0.4 * SR):int(0.55 * SR)]) > 2.0 * peak_hz(up[:int(0.03 * SR)])   # negate: the pitch rises
    down = np.asarray(I.REGISTRY["nes_pulse"]["fn"](220.0, 1.0, SR, 1.0, sweep=[1, 2, 0], mode="hardware_accurate"), float)
    assert np.abs(down[int(0.6 * SR):int(0.9 * SR)]).max() < 0.02 * np.abs(down).max()   # the target passes $7FF: muted


@pytest.mark.parametrize("freq", [110.0, 440.0, 1760.0])
def test_game_boy_pitches_are_the_period_values(freq):
    assert peak_hz(raw("gb_pulse", freq)) == pytest.approx(131072 / round(131072 / freq), rel=2e-4, abs=0.1)
    assert peak_hz(raw("gb_wave", freq, wave="sine")) == pytest.approx(65536 / round(65536 / freq), rel=2e-4, abs=0.1)


def test_game_boy_wave_ram_envelope_and_noise():
    f = 65536 / round(65536 / 220.0)
    sq = raw("gb_wave", 220.0, wave="square")
    assert harmonic(sq, f, 2) < 0.02 * harmonic(sq, f, 1)
    assert harmonic(raw("gb_wave", 220.0, wave="square", level=2), f, 1) == pytest.approx(0.5 * harmonic(sq, f, 1), rel=0.1)
    y = np.asarray(I.REGISTRY["gb_pulse"]["fn"](440.0, 1.0, SR, 1.0, env=-2, mode="hardware_accurate"), float)
    assert np.abs(y[int(0.6 * SR):int(0.9 * SR)]).max() < 0.02 * np.abs(y).max()      # 15 steps of 2/64 s = 0.47 s
    y = raw("gb_noise", 440.0, dur=1.0, period=[2, 1], short=True)                     # LFSR clock 262144 / (1 * 4) = 65536 Hz
    assert repeat_hz(y) == pytest.approx(65536 / 127, rel=0.02)     # the 7-bit sequence is 127 steps long


def test_stylized_voices_are_in_tune():
    for name in ("nes_pulse", "nes_triangle", "gb_pulse", "gb_wave"):
        y = np.asarray(I.render_note(name, 1975.53, 0.6, SR, 0.9), float)             # B6, where the timers are coarse
        assert abs(1200 * np.log2(peak_hz(y[int(0.05 * SR):int(0.5 * SR)]) / 1975.53)) < 3, name


def test_chip_layer_is_one_voice_per_channel():
    nes = {"type": "chip", "system": "nes", "channels": {
        "pulse1": {"steps": "E5:0.5 G5:0.5 [C5,E5,G5]:1", "duty": 1, "vol": [15, 12, 9, 7, 5]},
        "pulse2": {"steps": "C4:1 G3:1", "duty": 2, "volume": 8},
        "triangle": {"steps": "C3:0.5 -:0.5 G2:1"},
        "noise": [{"steps": "C8:0.25 -:0.75 C8:0.25 -:0.75", "decay": 1}, {"steps": "-:0.5 C5:0.25 -:1.25", "decay": 3}]}}
    y, sr = SP.render_spec({"sr": SR, "bpm": 120, "layers": [nes]})
    assert abs(len(y) / sr - 1.0) < 0.05 and np.isfinite(y).all() and 0.05 < np.abs(y).max() <= 1.0
    with pytest.raises(SP.SpecError):
        SP.render_spec({"sr": SR, "layers": [{"type": "chip", "system": "nes", "channels": {"wave": {"steps": "C4"}}}]})
    gb = {"type": "chip", "system": "gb", "wave": "bass", "channels": {
        "pulse1": {"steps": "E5:0.5 G5:0.5", "env": -3, "pan": "L"}, "pulse2": {"steps": "C5:1", "duty": 0, "pan": "R"},
        "wave": {"steps": "C3:1"}, "noise": {"steps": "C7:0.25 -:0.75", "env": -1}}}
    y, sr = SP.render_spec({"sr": SR, "layers": [gb]})
    assert y.ndim == 2 and np.abs(y[:, 0] - y[:, 1]).max() > 0.05                     # NR51: hard panning


def fm(freq=440.0, dur=0.6, tail=0.6, mode="retro_stylized", **patch):
    base = dict(alg=7, fb=0, ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict()])
    return np.asarray(I.REGISTRY["ym2612"]["fn"](freq, dur, SR, 1.0, patch=dict(base, **patch), tail=tail, mode=mode), float)


def test_ym2612_tables_are_the_chip_formulas():
    assert C.OPN_SIN[0] == 0x859 and C.OPN_SIN[255] == 0 and C.OPN_SIN[256] == 0 and C.OPN_SIN[512] == 0x859 | 0x8000
    assert C.OPN_POW[0] == (0x3FA | 0x400) << 2 and C.OPN_POW[255] == 0x400 << 2
    assert len(C.OPN_INC) == 64 and C.OPN_DETUNE.shape == (32, 4)


def test_ym2612_operator_levels_and_pitch():
    mid = slice(int(0.1 * SR), int(0.5 * SR))
    one = fm()[mid]
    assert peak_hz(one) == pytest.approx(440.0, abs=0.2)
    assert harmonic(one, 440.0, 2) < 0.01 * harmonic(one, 440.0, 1)                   # one operator is a sine
    quiet = fm(ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict(tl=8)])[mid]
    assert np.abs(quiet).max() == pytest.approx(0.5 * np.abs(one).max(), rel=0.02)    # total level: 0.75 dB a step
    octave = fm(ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict(mul=2)])[mid]
    assert peak_hz(octave) == pytest.approx(880.0, abs=0.4)
    four = fm(ops=[dict(), dict(), dict(), dict()])[mid]
    assert np.abs(four).max() == pytest.approx(4 * np.abs(one).max(), rel=0.03)       # algorithm 7: four carriers


def test_ym2612_modulation_feedback_and_envelope():
    mid = slice(int(0.1 * SR), int(0.5 * SR))
    plain = fm()[mid]
    chain = fm(alg=0, ops=[dict(tl=127), dict(tl=127), dict(tl=24), dict()])[mid]     # operator 3 modulates operator 4
    assert harmonic(chain, 440.0, 2) > 0.1 * harmonic(chain, 440.0, 1) > 0
    saw = fm(alg=7, fb=6, ops=[dict(), dict(tl=127), dict(tl=127), dict(tl=127)])[mid]
    assert harmonic(saw, 440.0, 2) > 0.2 * harmonic(saw, 440.0, 1) and harmonic(plain, 440.0, 2) < 0.01 * harmonic(plain, 440.0, 1)
    slow = fm(ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict(ar=12)])
    assert np.abs(slow[:int(0.01 * SR)]).max() < 0.3 * np.abs(slow[int(0.4 * SR):int(0.5 * SR)]).max()   # the attack rises
    held, cut = (fm(tail=1.0, ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict(rr=r)]) for r in (2, 15))
    after = slice(int(0.8 * SR), int(0.9 * SR))
    assert np.abs(held[after]).max() > 0.3 * np.abs(held).max() and np.abs(cut[after]).max() < 0.01 * np.abs(cut).max()
    pluck = fm(dur=1.0, ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict(dr=14, sl=15, sr=0)])
    assert np.abs(pluck[int(0.8 * SR):int(0.9 * SR)]).max() < 0.2 * np.abs(pluck[:int(0.05 * SR)]).max()   # decay to the sustain level


def test_ym2612_nine_bit_output_and_chip_layer():
    raw9 = C.render_opn({0: [(0.0, 0.3, 440.0, 1.0, "sine", "C")]}, 0.3, SR, raw_dac=True)
    native = C._opn_core(2000, np.array([[0, 0, 1, 0, (4 << 11) | 1083, 0]], np.int64), np.array([C._fm_patch("sine")], np.int64), -1, 1,
                         C.OPN_SIN, C.OPN_POW, C.OPN_INC, C.OPN_DETUNE, C.OPN_PM, C.OPN_LFO_MAX, C.OPN_ALG)
    steps = np.unique(np.round(native[:, 0] * 6 * 260).astype(int))
    assert len(steps) <= 520                                                          # nine bits a channel
    assert not np.any((steps > 20) & (steps < 28)) or 24 in steps                     # the DAC jumps across zero: 21 or 28, never between
    assert np.isfinite(raw9).all()
    layer = {"type": "chip", "system": "ym2612", "lfo": 3, "channels": {
        "fm1": {"steps": "E4:0.5 G4:0.5 C5:1", "patch": "lead", "pan": "L"}, "fm2": {"steps": "C3:1 G2:1", "patch": "bass", "pan": "R"},
        "fm3": {"steps": "C4:2", "patch": "strings"}, "fm4": {"steps": "E4:2", "patch": "strings"}}}
    y, sr = SP.render_spec({"sr": SR, "layers": [layer]})
    assert y.ndim == 2 and np.isfinite(y).all() and 0.05 < np.abs(y).max() <= 1.0 and abs(len(y) / sr - 2.5) < 0.05
    with pytest.raises(SP.SpecError):
        SP.render_spec({"sr": SR, "layers": [{"type": "chip", "system": "ym2612", "channels": {"fm1": {"steps": "[C4,E4]:1"}}}]})
