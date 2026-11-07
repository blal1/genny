import math

import numpy as np
import pytest

from genny import chiptune as CH
from genny import creatures as C
from genny import retro as R

SR = 48000


def _peak_hz(x, sr=SR, lo=50.0, hi=20000.0):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (f > lo) & (f < hi)
    return f[m][np.argmax(X[m])]


def test_oec_helmholtz_frequency():
    assert C.oec_hz() == pytest.approx(1.0 / (2 * math.pi * math.sqrt(20.0 * 1.5e-8 / 350.0)), rel=1e-9)
    assert 5000 < C.oec_hz() < 6000
    assert C.OEC["lb"] == pytest.approx(1.2 * 0.01 / 0.4e-4)


def test_oec_beak_resonance_near_helmholtz():
    x = np.zeros(SR // 4)
    x[100] = 1.0
    y = C.oec_beak(x, 2500.0)
    assert _peak_hz(y, lo=1000.0, hi=12000.0) == pytest.approx(C.oec_hz(), rel=0.2)


@pytest.mark.parametrize("n,f1,f2", [(4, 945, 1403), (7, 782, 1194), (8, 795, 1227)])
def test_diana_tubes_formants(n, f1, f2):
    F = C.tube_formants(C.DIANA_TUBES[n])
    assert F[0] == pytest.approx(f1, rel=0.02)
    assert F[1] == pytest.approx(f2, rel=0.02)


def test_tube_formants_uniform_quarter_wave():
    F = C.tube_formants(((2.0, 10.0),))
    assert F[0] == pytest.approx(350.0 / 0.4, rel=1e-3)
    assert F[1] == pytest.approx(3 * 350.0 / 0.4, rel=1e-3)


def test_diana_alarm_length():
    p = C.DIANA_CALL
    y = C.diana_alarm(0)
    assert len(y) == C._n(p["calls"] * (p["dur"] + p["gap"]) + 0.05)


def test_latex_membrane_law():
    f = C.latex_membrane_hz(0.0035, 0.05)
    assert f == pytest.approx(2.405 / (2 * math.pi * 0.0035) * math.sqrt(1e6 * 0.05 / (0.5 * 930.0)), rel=1e-9)
    assert C.latex_membrane_hz(0.0035, 0.2) == pytest.approx(2.0 * f, rel=1e-9)


def test_labial_syrinx_threshold_and_pitch():
    n = SR // 2
    on = C.labial_syrinx(np.full(n, 3000.0), (np.full(n, 600.0), None), coupling=0.0)
    off = C.labial_syrinx(np.full(n, 800.0), (np.full(n, 600.0), None), coupling=0.0)
    tail = on[n // 2:]
    assert _peak_hz(tail, lo=200.0, hi=2000.0) == pytest.approx(600.0, rel=0.05)
    assert np.sqrt(np.mean(off[n // 2:] ** 2)) < 0.01 * np.sqrt(np.mean(tail ** 2))


def test_labial_two_sources_uncoupled_two_peaks():
    n = SR
    y = C.labial_syrinx(np.full(n, 3000.0), (np.full(n, 500.0), np.full(n, 700.0)), coupling=0.0)[n // 2:]
    X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    a5 = X[(f > 480) & (f < 520)].max()
    a7 = X[(f > 680) & (f < 720)].max()
    floor = np.median(X[(f > 300) & (f < 900)])
    assert a5 > 10 * floor and a7 > 10 * floor


def test_purr_voice_adds_200hz():
    def band(y):
        X = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1.0 / SR)
        return X[(f > 170) & (f < 230)].sum() / X[(f > 20) & (f < 2000)].sum()
    a = C.purr(sr=SR, dur=2.0, voice=0.0)
    b = C.purr(sr=SR, dur=2.0, voice=1.0)
    assert band(b) > 2.0 * band(a)


def test_birdsong_oec_tract_differs():
    a = C.birdsong(sr=SR, species="canary", source="normal_form", tract="bands")
    b = C.birdsong(sr=SR, species="canary", source="normal_form", tract="oec")
    assert len(a) == len(b) and not np.allclose(a, b)


def _mod(rows):
    b = bytearray(1084 + 64 * 4 * 4)
    b[20 + 22:20 + 24] = (100).to_bytes(2, "big")
    b[45] = 64
    b[950] = 1
    b[952] = 0
    b[1080:1084] = b"M.K."
    for row, ch, smp, per, fx, prm in rows:
        o = 1084 + (row * 4 + ch) * 4
        b[o:o + 4] = bytes([(smp & 0xF0) | (per >> 8), per & 0xFF, ((smp & 0x0F) << 4) | fx, prm])
    return bytes(b)


def test_mod_note_formula():
    assert R.mod_note(428) == int(70 - (math.log2(428) - 7) * 12) - 36 == 13


def test_mod_tracks_note_volume_break():
    tracks, total, row_s = R._mod_tracks({"data": _mod([(0, 0, 1, 428, 0xF, 3), (2, 1, 1, 214, 0xC, 32), (15, 0, 0, 0, 0xD, 0)])})
    assert total == 16
    assert row_s == pytest.approx(3 * 2.5 / 125)
    row, f, vel, r = tracks[0][0]
    assert row == 0 and f == pytest.approx(220.0 * 2 ** (1 / 12)) and vel == pytest.approx(1.0)
    assert r["pan"] == -1.0
    row, f, vel, r = tracks[1][0]
    assert f == pytest.approx(220.0 * 2 ** (13 / 12)) and vel == pytest.approx(0.5) and r["pan"] == 1.0


def test_frame_rates_and_tick_rows():
    assert R.FRAME_HZ["zx48"] == pytest.approx(50.08, abs=0.005)
    assert R.FRAME_HZ["zx128"] == pytest.approx(50.021, abs=0.001)
    layer = {"patterns": {"a": {"lead": "C-4 . . ."}}, "instruments": {"lead": {"zzfx": None}}, "ticks": 6, "frame": "zx48"}
    y = R.song(layer, SR, 0.5)
    assert y.shape[0] == int(round(4 * 6 / R.FRAME_HZ["zx48"] * SR))


def test_zzfxg_attack_offset():
    a = R._zz_array(None)
    assert len(R.zzfx_core(a, 44100.0, 0.5, 10 ** 7)) == 9 + 4410
    assert len(R.zzfx_core(a, 44100.0, 0.5, 10 ** 7, gen=True)) == 99 + 4410


def test_bfxr_strict_laws():
    P = np.zeros(22)
    P[0], P[1], P[3], P[4], P[5], P[16] = 0.0, 0.0, 0.05, 0.3, 0.5, 0.5
    Q = R.bfxr_strict(P)
    assert Q[5] == pytest.approx(0.5 ** 0.4 * 0.3)
    assert Q[1] == pytest.approx(0.01 * 0.18 / 0.06)
    assert Q[3] == pytest.approx(math.sqrt((0.05 * 3) ** 2 + 1e-4))
    assert Q[16] == pytest.approx(math.sqrt(0.2 * 0.125))


def test_gb_wave_from_spectrum():
    s = CH.gb_wave_from_spectrum([1.0])
    assert all(s[i] + s[i + 16] == 15 for i in range(16)) and s[7] == 15 and s[23] == 0
    w = CH.gb_wave_from_spectrum([1.0, 0.5])
    assert min(w) == 0 and max(w) == 15 and len(w) == 32


def test_beeper_rom_quantisation():
    hl = round(437500.0 / 440.0 - 30.125)
    assert R.beeper_hz(440.0) == pytest.approx(437500.0 / (hl + 30.125))
    y = R.beeper(440.0, 0.5, SR)
    assert set(np.unique(y)) == {-0.5, 0.5}
    assert _peak_hz(y, lo=100.0, hi=2000.0) == pytest.approx(R.beeper_hz(440.0), abs=3.0)
