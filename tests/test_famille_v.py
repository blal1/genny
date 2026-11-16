import math
import os

import numpy as np
import pytest

from genny import family_v as V
from genny.fx import REGISTRY as FX
from genny.instruments import REGISTRY as INST

SR = 44100
SOFA = os.path.join(os.path.dirname(__file__), "..", "..", "telech_20261107", "data", "hrtf", "mit_kemar_normal_pinna.sofa")


def _f0(y, sr=SR):
    y = y - y.mean()
    n = len(y)
    ac = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * n)) ** 2)[:n // 2]
    lo = int(sr / 3000)
    k = lo + int(np.argmax(ac[lo:]))
    thr = 0.6 * ac[k]
    for j in range(lo, n // 2 - 1):
        if ac[j] > thr and ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1]:
            k = j
            break
    a, b, c = ac[k - 1:k + 2]
    return sr / (k + 0.5 * (a - c) / (a - 2 * b + c))


def test_itd_woodworth_max():
    assert abs(V.itd_woodworth(90.0) - 0.000656) < 5e-6                  # (pi/2 + 1) a / c, a = 8.75 cm
    assert abs(V.itd_kuhn(90.0) / V.itd_woodworth(90.0) - 3.0 / (math.pi / 2 + 1)) < 1e-9
    assert V.itd_woodworth(0.0) == 0.0 and V.itd_woodworth(-45.0) == -V.itd_woodworth(45.0)


def test_parametric_hrir_itd_ild():
    h90 = V.hrir_param(90.0, 0.0, SR)
    assert 0.0006 < V.itd_measure(h90, SR) < 0.00078                     # explicit 0.656 ms + the shadow filter's low-frequency delay
    assert abs(V.itd_measure(V.hrir_param(0.0, 0.0, SR), SR)) < 2e-6
    assert V.itd_measure(V.hrir_param(-90.0, 0.0, SR), SR) < -0.0006
    assert V.ild_measure(h90, 2000, 8000, SR) > 10.0                     # right ear louder
    assert V.ild_measure(h90, 100, 400, SR) < V.ild_measure(h90, 2000, 8000, SR) / 2
    assert abs(V.ild_measure(V.hrir_param(0.0, 0.0, SR), 2000, 8000, SR)) < 0.5
    assert V.hrir_param(90.0, 0.0, SR, a=0.12).shape == (256, 2)
    assert V.itd_measure(V.hrir_param(90.0, 0.0, SR, a=0.12), SR) > V.itd_measure(h90, SR)


def test_pinna_notch_rises_with_elevation():
    def notch(el):
        s = np.abs(np.fft.rfft(V.hrir_param(0.0, el, SR, n=512), 2048, axis=0))[:, 1]
        f = np.fft.rfftfreq(2048, 1 / SR)
        m = (f > 4000) & (f < 14000)
        return f[m][np.argmin(s[m])]
    assert notch(40.0) > notch(-20.0)


def test_binaural_move_is_continuous_and_pans():
    x = np.random.default_rng(1).standard_normal(SR)
    y = V.binaural_path(x, SR, np.array([-80.0, 80.0]), 0.0, 2.0)
    assert not np.isnan(y).any() and y.shape[1] == 2
    l, r = np.sqrt(np.mean(y[:6000] ** 2, 0))
    assert l > 1.5 * r                                                   # starts on the left
    l, r = np.sqrt(np.mean(y[-9000:-3000] ** 2, 0))
    assert r > 1.5 * l                                                   # ends on the right
    far = V.binaural_path(x, SR, 0.0, 0.0, 4.0)
    near = V.binaural_path(x, SR, 0.0, 0.0, 1.0)
    assert abs(np.std(near) / np.std(far) / 4.0 - 1.0) < 0.05            # 1 / distance
    assert "binaural_move" in FX and "binaural_room" in FX


def test_doppler_shifts_pitch():
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * 1000 * t)
    y = V.binaural_path(x, SR, 0.0, 0.0, np.linspace(30.0, 3.0, x.size), doppler=True)[:, 0]
    seg = y[SR // 2:int(1.5 * SR)]
    f = np.argmax(np.abs(np.fft.rfft(seg * np.hanning(seg.size)))) * SR / seg.size
    v = 27.0 / 2.0                                                       # closing speed m/s
    assert abs(f / (1000 * 343.0 / (343.0 - v)) - 1.0) < 0.01


def test_sofa_reader_and_itd_vs_woodworth():
    pytest.importorskip("h5py")
    db = V.load_hrir(SOFA)
    assert db["hrir"].shape[2] == 2 and db["sr"] == 44100
    h = V.hrir_at(90.0, 0.0, SR, db, n=512)
    assert abs(db["az"][np.argmax(V._unit(db["az"], db["el"]) @ V._unit(90.0, 0.0))] - 90.0) < 6.0
    itd = V.itd_measure(h, SR)
    assert 0.0005 < itd < 0.0009                                         # measured KEMAR near Woodworth's 0.66 ms
    assert abs(itd - V.itd_measure(V.hrir_param(90.0, 0.0, SR), SR)) < 0.0001
    assert V.ild_measure(h, 2000, 8000, SR) > 6.0
    assert V.itd_measure(V.hrir_at(-90.0, 0.0, SR, db, n=512), SR) < 0


def test_binaural_room_ir():
    room = ((6, 8, 3), (3.5, 5, 1.5), (2.5, 2, 1.6))
    ir = V.binaural_room_ir(*room, "plaster", SR, 4)
    assert not np.isnan(ir).any()
    assert np.argmax(np.abs(ir[:, 1])) < 200                             # direct sound first
    assert V.itd_measure(ir[:200], SR) > 0.0001                          # source to the right of the listener (x 3.5 > 2.5)
    early = ir[int(0.01 * SR):int(0.1 * SR)]
    assert np.sum(early ** 2) > 0.05 * np.sum(ir[:200] ** 2)             # reflections exist
    turned = V.binaural_room_ir(*room, "plaster", SR, 3, yaw=90.0)
    assert V.itd_measure(turned[:200], SR) < V.itd_measure(ir[:200], SR)  # head turned right: the source moves to the left
    absorbent = V.binaural_room_ir(*room, "carpet", SR, 4, tail=False)
    hard = V.binaural_room_ir(*room, "marble", SR, 4, tail=False)
    assert np.sum(hard[200:] ** 2) > np.sum(absorbent[200:] ** 2)


def test_free_reed_pitch_tracks_the_blade():
    from genny.physical.winds import FREE_REED_F
    assert FREE_REED_F == (0.254, 0.593, 0.777)
    for preset in V.FREE_REED_PRESETS:
        for f in (196.0, 440.0, 1000.0):
            y = V.free_reed_voice(f, 0.8, SR, preset)
            assert not np.isnan(y).any()
            seg = y[SR // 4:int(0.7 * SR)]
            assert np.sqrt(np.mean(seg ** 2)) > 0.01                     # it speaks
            assert abs(_f0(seg) / f - 1.0) < 0.02                        # f proportional to the blade (within 2 %)


def test_free_reed_instruments():
    for name in ("harmonica_reed", "accordion_reed", "melodica_reed", "sho_reed", "khaen_reed"):
        y = INST[name]["fn"](330.0, 0.5, SR, 1.0)
        assert not np.isnan(y).any() and np.max(np.abs(y)) > 0.1 and abs(y[0]) < 0.05 and abs(y[-1]) < 0.05
    a = INST["accordion_reed"]["fn"](440.0, 1.2, SR, 1.0, musette=9.0)
    b = INST["accordion_reed"]["fn"](440.0, 1.2, SR, 1.0, musette=0.0)
    sp = lambda z: np.abs(np.fft.rfft(z[SR // 4:SR // 4 + 2 ** 15] * np.hanning(2 ** 15)))
    assert np.sum(sp(a) > 0.2 * sp(a).max()) > np.sum(sp(b) > 0.2 * sp(b).max())   # the second reed adds partials near each peak


def test_body_modes_at_published_frequencies():
    imp = np.zeros(16384)
    imp[0] = 1.0
    f = np.fft.rfftfreq(16384, 1 / SR)
    for body, names in (("violin", ("A0", "CBR", "B1-", "B1+")), ("guitar_classical", ("T(1,1)1", "T(1,1)2", "T(1,1)3")),
                        ("guitar_steel", ("T(1,1)2", "T(3,1)"))):
        s = np.abs(np.fft.rfft(V.body_apply(imp, SR, body)))
        for n, fk, q, g in V.BODY_MODES[body]:
            if n in names:
                m = (f > fk * 0.93) & (f < fk * 1.07)
                assert abs(f[m][np.argmax(s[m])] / fk - 1.0) < 0.025
                assert s[m].max() > 1.5
    assert np.allclose(np.abs(np.fft.rfft(V.body_apply(imp, SR, "violin", amount=0.0))), 1.0)
    h = V.body_apply(imp, SR, "violin")
    assert np.abs(h[SR // 10:]).max() < 0.2 * np.abs(h).max()            # the modes ring out
    assert V.BODY_MODES["violin"][1][:3] == ("A0", 275.0, 30.0)


def test_body_on_string():
    for name, f in (("guitar_modal", 220.0), ("violin_modal", 440.0)):
        y = INST[name]["fn"](f, 0.8, SR, 1.0)
        assert not np.isnan(y).any() and np.max(np.abs(y)) > 0.3
    raw = V._string(98.0, 0.8, SR, "pluck", 1.0, 0)
    out = V.body_apply(raw, SR, "guitar_classical")
    w = np.hanning(raw.size)
    r = np.abs(np.fft.rfft(out * w)) / (np.abs(np.fft.rfft(raw * w)) + 1e-9)
    f = np.fft.rfftfreq(raw.size, 1 / SR)
    assert r[(f > 190) & (f < 205)].max() > 2.0                          # the top-plate mode boosts the string's second partial


def test_tape_hysteresis_loop():
    H, M = V.tape_bh_loop()
    assert abs(M).max() > 0.9                                            # saturates toward Ms
    z = np.where(np.diff(np.sign(M)) != 0)[0]
    assert len(z) >= 2 and 0.5 < abs(H[z]).mean() < 2.5                  # coercive field of the order of k / Ms = 1
    assert abs(M[np.argmin(abs(H))]) > 0.2                               # remanence
    assert abs(np.trapezoid(M, H)) > 1.0                                 # a loop with area: energy lost per cycle
    i0 = np.where(np.diff(np.sign(H)) != 0)[0]
    assert abs(M[i0[0]] - M[i0[1]]) > 0.5                                # the two branches differ at H = 0


def test_tape_bias_and_saturation_harmonics():
    t = np.arange(SR) / SR

    def harm(bias, lev):
        s = np.abs(np.fft.rfft(V.tape_record(lev * np.sin(2 * np.pi * 1000 * t), bias)))
        return s[2000] / s[1000], s[3000] / s[1000], s[1000] * 2 / SR / lev
    assert harm(1.0, 0.3)[1] < 0.2 * harm(0.0, 0.3)[1]                   # HF bias linearises the record curve
    assert harm(1.0, 0.9)[1] > 3.0 * harm(1.0, 0.05)[1]                  # saturation: 3rd harmonic grows with level
    assert harm(1.0, 0.9)[0] < 0.01                                      # symmetric loop: even harmonics negligible
    assert harm(2.0, 0.3)[2] < harm(1.0, 0.3)[2]                         # over-bias: less output
    assert harm(0.0, 0.05)[2] > harm(1.0, 0.05)[2]                       # no bias: more output at low level (the dead zone fills)


def test_tape_bias_effect():
    t = np.arange(SR) / SR
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    y = FX["tape_bias"]["fn"](x, SR, hiss=-120)
    assert y.shape == x.shape and not np.isnan(y).any() and 0.1 < np.max(np.abs(y)) < 1.5
    z = FX["tape_bias"]["fn"](x, SR, wow=0.0, flutter=0.0, hiss=-120, speed=0.0, hf=0)
    s = np.abs(np.fft.rfft(z))
    assert s[440] > 50 * s[1320]                                         # clean at moderate level with linear bias
    w = FX["tape_bias"]["fn"](x, SR, wow=1.0, flutter=0.5, hiss=-120, speed=0.0)
    assert np.abs(np.fft.rfft(w))[440] < 0.9 * s[440]                    # wow and flutter smear the line
