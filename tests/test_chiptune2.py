"""SN76489, NES DMC, YM2151, YM2610, SID and POKEY against the numbers of their documentation (out/research/chips)."""
import numpy as np
import pytest

import genny.chiptune2 as C2
import genny.spec as SP
from genny import fx as FX
from genny import instruments as I

SR = 48000


def peak_hz(y, lo=30.0, hi=20000.0):
    y = np.asarray(y, float)
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 19))
    fr = np.fft.rfftfreq(1 << 19, 1 / SR)
    band = (fr > lo) & (fr < hi)
    return fr[band][np.argmax(p[band])]


def harmonic(y, f, k):
    t = np.arange(len(y)) / SR
    return abs(np.sum(np.asarray(y, float) * np.hanning(len(y)) * np.exp(-2j * np.pi * k * f * t)))


def raw(name, freq, dur=0.6, **params):
    y = np.asarray(I.REGISTRY[name]["fn"](freq, dur, SR, 1.0, mode="hardware_accurate", **params), float)
    return y[int(0.05 * SR):int((dur - 0.05) * SR)]


def test_sn76489_register_0fe_is_440_hz_and_a_step_is_2_db():
    # SMS Power: "0x0fe gives 440.4Hz"; "attenuates the volume by 2dB for each step"
    assert peak_hz(raw("sn_tone", 440.0)) == pytest.approx(3579545 / (2 * 0xFE * 16), abs=0.1)
    full, three_down = (np.abs(raw("sn_tone", 440.0, volume=v)).max() for v in (15, 12))
    assert 20 * np.log10(three_down / full) == pytest.approx(-6.0, abs=0.1)
    assert np.abs(raw("sn_tone", 440.0, volume=0)).max() < 1e-6                       # 15 steps down is silence


def test_sn76489_periodic_noise_is_a_sixteenth_of_the_tone():
    y = raw("sn_noise", 110.0, dur=1.0, white=False)                                   # the register looped: one pulse in 16 shifts
    assert peak_hz(y, 40, 400) == pytest.approx(110.0, rel=0.02)
    assert np.std(raw("sn_noise", 2000.0)) > 0.05
    assert np.array_equal(C2.SN_VARIANTS["sega"], (0x0009, 16)) and np.array_equal(C2.SN_VARIANTS["ti"], (0x0003, 15))


def test_dmc_moves_two_steps_a_clock_and_stays_in_seven_bits():
    up = C2._dmc(np.full(100, 127.0), 64)
    assert up[0] == 66 and up[1] == 68 and up.max() == 126                              # +2 a clock, never above 126
    down = C2._dmc(np.zeros(100), 64)
    assert down[0] == 62 and down.min() == 0
    t = np.arange(SR) / SR
    s = 0.8 * np.sin(2 * np.pi * 220 * t)
    fast, slow = (FX.apply_chain(s, [{"type": "dpcm", "rate": r}], SR) for r in (15, 0))
    assert fast.shape == s.shape and np.isfinite(fast).all()
    assert np.sqrt(np.mean((fast - s) ** 2)) < 0.3 * np.sqrt(np.mean(s ** 2))          # 33 kHz follows a 220 Hz tone
    assert np.abs(slow).max() < 0.5 * np.abs(s).max()                                  # 4 kHz cannot: slope overload
    assert C2.DMC_RATES[0] == 428 and C2.DMC_RATES[15] == 54


@pytest.mark.parametrize("freq", [55.0, 440.0, 3520.0])
def test_ym2151_pitch_table_is_in_tune(freq):
    y = np.asarray(I.REGISTRY["ym2151"]["fn"](freq, 0.8, SR, 1.0, patch="sine"), float)
    assert abs(1200 * np.log2(peak_hz(y[4000:30000]) / freq)) < 2.0                    # the table is in 1/64 semitone
    assert C2.OPM_STEP.shape == (768,) and C2.OPM_STEP[0] == 41568 and C2.OPM_STEP[767] == 82976
    assert C2.OPM_DT2 == (0, 384, 500, 608)


def test_ym2151_second_detune_raises_one_operator():
    plain = dict(alg=7, fb=0, ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict()])
    fifth = dict(alg=7, fb=0, ops=[dict(tl=127), dict(tl=127), dict(tl=127), dict(dt2=1)])
    a, b = (np.asarray(I.REGISTRY["ym2151"]["fn"](440.0, 0.8, SR, 1.0, patch=p), float)[4000:30000] for p in (plain, fifth))
    assert 1200 * np.log2(peak_hz(b) / peak_hz(a)) == pytest.approx(600.0, abs=3.0)     # DT2 = 1 is 600 cents


def test_sid_waveforms_envelope_and_filter():
    tri, saw, pulse = (raw("sid", 440.0, wave=w, s=15) for w in ("triangle", "saw", "pulse"))
    assert peak_hz(saw) == pytest.approx(440.0, abs=0.2)                               # 16-bit register: 0.06 Hz steps
    assert harmonic(tri, 440, 2) < 0.02 * harmonic(tri, 440, 1) and harmonic(pulse, 440, 2) < 0.02 * harmonic(pulse, 440, 1)
    assert harmonic(saw, 440, 2) == pytest.approx(0.5 * harmonic(saw, 440, 1), rel=0.05)
    thin = raw("sid", 440.0, wave="pulse", pw=0.125, s=15)
    assert harmonic(thin, 440, 8) < 0.03 * harmonic(thin, 440, 1) and harmonic(thin, 440, 2) > 0.5 * harmonic(thin, 440, 1)
    y = np.asarray(I.REGISTRY["sid"]["fn"](440.0, 2.5, SR, 1.0, wave="saw", a=9, d=9, s=8, r=9, mode="hardware_accurate"), float)
    lvl = lambda t: np.abs(y[int(t * SR):int((t + 0.03) * SR)]).max()
    assert lvl(0.11) == pytest.approx(0.5 * lvl(0.235), rel=0.2)                        # attack 9: 250 ms, linear
    assert lvl(2.0) == pytest.approx(lvl(0.235) * 8 / 15, rel=0.1)                      # sustain 8 of 15
    assert lvl(2.5 + 0.7) < 0.1 * lvl(2.0)                                              # release 9: 750 ms
    assert C2.SID_RATE[0] == 8 and C2.SID_RATE[15] == 31250
    cen = lambda v: (np.fft.rfftfreq(len(v), 1 / SR) * np.abs(np.fft.rfft(v)) ** 2).sum() / (np.abs(np.fft.rfft(v)) ** 2).sum()
    dark, bright = (raw("sid", 110.0, wave="saw", s=15, cutoff=c, res=0) for c in (100, 1800))
    assert cen(dark) < 0.7 * cen(bright)
    hp = raw("sid", 110.0, wave="saw", s=15, cutoff=1800, res=0, filter="hp")
    assert cen(hp) > 3 * cen(bright)


def test_pokey_dividers_and_polynomial_counters():
    assert (len(C2.POKEY_P4), len(C2.POKEY_P5), len(C2.POKEY_P9), len(C2.POKEY_P17)) == (15, 31, 511, 131071)   # maximal lengths
    assert peak_hz(raw("pokey", 440.0, clock="64k")) == pytest.approx(1789790 / 28 / (2 * 73), abs=0.2)         # AUDF 72
    assert peak_hz(raw("pokey", 440.0, clock="15k")) == pytest.approx(1789790 / 114 / (2 * 18), abs=0.2)        # AUDF 17
    assert peak_hz(raw("pokey", 5000.0, clock="1.79m")) == pytest.approx(1789790 / (2 * (175 + 4)), abs=1.0)    # AUDF + 4
    assert C2._pokey_clock(60.0, "auto") == "15k" and C2._pokey_clock(440.0, "auto") == "64k" and C2._pokey_clock(5000.0, "auto") == "1.79m"
    y = np.asarray(I.render_note("pokey", 1975.53, 0.6, SR, 0.9), float)
    assert abs(1200 * np.log2(peak_hz(y[3000:-3000]) / 1975.53)) < 3                    # stylized: played back in tune
    assert np.std(raw("pokey", 220.0, dist="noise")) > 0.05 and np.std(raw("pokey", 220.0, dist="buzz")) > 0.05


@pytest.mark.parametrize("system,channels", [
    ("sn76489", {"tone1": {"steps": "E5:0.5 G5:0.5 [C5,E5,G5]:1", "vol": [15, 12, 10]}, "noise": {"steps": "C3:2", "white": False}}),
    ("ym2151", {"fm1": {"steps": "E4:0.5 G4:0.5 C5:1", "patch": "lead", "pan": "L"}, "fm8": {"steps": "C3:2", "patch": "bass", "pan": "R"}}),
    ("ym2610", {"fm1": {"steps": "E4:1 G4:1", "patch": "epiano"}, "ssg1": {"steps": "C5:0.5 E5:0.5 G5:1", "level": 11}}),
    ("sid", {"voice1": {"steps": "C2:1 G2:1", "wave": "saw", "s": 12}, "voice2": {"steps": "[C4,E4,G4]:2", "pw": [0.1, 0.2, 0.3], "s": 10}}),
    ("pokey", {"ch1": {"steps": "E5:1 G5:1", "vol": [12, 9, 6]}, "ch2": {"steps": "C3:2", "dist": "buzz"}}),
])
def test_every_system_renders_through_the_chip_layer(system, channels):
    y, sr = SP.render_spec({"sr": SR, "layers": [{"type": "chip", "system": system, "channels": channels}]})
    assert np.isfinite(y).all() and 0.05 < np.abs(y).max() <= 1.0 and len(y) / sr >= 2.0
    with pytest.raises(SP.SpecError):
        SP.render_spec({"sr": SR, "layers": [{"type": "chip", "system": system, "channels": {"nope": {"steps": "C4"}}}]})
