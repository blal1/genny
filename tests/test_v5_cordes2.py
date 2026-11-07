"""V5_CORDES2: piano bank options, fdstring theta scheme and hammers, strings, winds and plates options.

Run:  uv run python -m pytest tests/test_v5_cordes2.py
"""
import math

import numpy as np

from genny import acoustic as A
from genny import chiptune2 as C2
from genny import fdstring as fd
from genny import klang as K
from genny import plates as PL
from genny import synths as S
from genny.physical import bank_piano as bp
from genny.physical import bow as BW
from genny.physical import waveguides as W

SR = 44100


def fin(y):
    return bool(np.all(np.isfinite(y))) and float(np.max(np.abs(y))) > 0


def pitch_cents(y, f0, sr=SR):
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


def band_db(y, lo, hi, sr=SR):
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(y.size, 1 / sr)
    return 10 * np.log10(Y[(f > lo) & (f < hi)].sum() + 1e-30)


# ------------------------------------------------------------------ bank_piano
def test_bridge_admittance_peak_and_q():
    f = np.linspace(100.0, 1000.0, 90001)
    y = bp.bridge_admittance(f, 10.0, bp.BRIDGE_RES_HZ)
    i = int(np.argmax(y.real))
    assert abs(f[i] - bp.BRIDGE_RES_HZ) < 1.0 and abs(y.real[i] / bp.Y_BRIDGE - 11.0) < 0.01
    q = math.pi * bp.BRIDGE_RES_HZ / (bp.SB_A * bp.BRIDGE_RES_HZ ** 2 + bp.SB_B * bp.BRIDGE_RES_HZ)
    assert abs(q - 39.6) < 0.5
    assert np.allclose(bp.bridge_admittance(f[:5]), bp.Y_BRIDGE)


def test_long_coupling_is_tangent():
    assert abs(bp.long_coupling(1.0) - math.tan(math.radians(1.0))) < 1e-12


def test_top_keys_undamped():
    assert bp.damped(60) and not bp.damped(bp.DAMPER_TOP_MIDI + 1)


def test_pedal_below_note_and_finite():
    y0 = bp.note(110.0, 0.5, SR, 0.8, tail=0.3)
    y1 = bp.note(110.0, 0.5, SR, 0.8, tail=0.3, pedal=True)
    d = y1[:y0.size] - y0[:y1.size]
    assert fin(y1) and 20 * np.log10(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(y0 ** 2))) < -3.0


def test_lid_closed_darkens():
    y = bp.note(440.0, 0.4, SR, 0.8, tail=0.2)
    yc = bp.lid(y, SR, "closed")
    assert band_db(yc, 4000, 12000) - band_db(y, 4000, 12000) < -6.0
    assert abs(band_db(yc, 50, 500) - band_db(y, 50, 500)) < 1.0


def test_ribs_more_poles_less_treble():
    a = bp.soundboard_filter(SR)[0]
    b = bp.soundboard_filter(SR, ribs=True)[0]
    assert b.size > a.size
    y0 = bp.note(261.6, 0.4, SR, 0.8, tail=0.2)
    y1 = bp.note(261.6, 0.4, SR, 0.8, tail=0.2, ribs=True)
    assert band_db(y1, 1200, 8000) - band_db(y1, 50, 1000) < band_db(y0, 1200, 8000) - band_db(y0, 50, 1000)


def test_bidirectional_and_noise_action_finite():
    assert fin(bp.note(110.0, 0.3, SR, 0.9, tail=0.2, bidirectional=True, polarisations=2))
    assert fin(bp.note(261.6, 0.3, SR, 0.8, tail=0.2, action=1.0, action_kind="noise"))


# ------------------------------------------------------------------ fdstring
def test_conservative_and_wide_hammer():
    y0 = fd.fd_piano(220.0, 0.3, SR, 0.8)
    y1 = fd.fd_piano(220.0, 0.3, SR, 0.8, hammer_model="conservative")
    # 220 Hz: N = 75, so 0.02 is 1.5 cells and the strike point 0.12 * 75 falls on a node (a point load);
    # 0.05 spreads the felt over several cells
    y2 = fd.fd_piano(220.0, 0.3, SR, 0.8, hammer_width=0.05)
    assert fin(y1) and fin(y2) and not np.allclose(y0, y1) and not np.allclose(y0, y2)
    assert abs(pitch_cents(y1, 220.0)) < 15.0


def test_theta_scheme_tunes_and_is_finite():
    for scheme in ("explicit", "theta4", "stable", "bensa"):
        y = fd.fd_piano(220.0, 0.4, SR, 0.8, scheme=scheme)
        assert fin(y), scheme
        assert abs(pitch_cents(y, 220.0)) < 15.0, scheme


def test_xylophone_tuning_1_3_6():
    b = fd.Bar(440.0, SR, ("free", "free"), None, 32, (3.0, 6.0))
    r = b.modes[:3] / b.modes[0]
    assert abs(r[1] - 3.0) < 0.03 and abs(r[2] - 6.0) < 0.06


def test_bead_lowers_fundamental():
    s = fd.StiffString(220.0, SR, B=1e-4, t60=None)
    s.mass(0.5, 0.3)
    s.pluck(0.2, 0.05, 0.001)
    y = s.run(int(0.5 * s.sr_i))[:, 0]
    Y = np.abs(np.fft.rfft(y * np.hanning(y.size), 16 * y.size))
    f = np.fft.rfftfreq(16 * y.size, 1 / s.sr_i)
    m = (f > 100) & (f < 300)
    assert f[m][np.argmax(Y[m])] < 200.0


def test_fd_bar_options_and_interceptor():
    for kw in (dict(glide=40.0), dict(scan=3.0), dict(scheme="implicit")):
        assert fin(fd.fd_bar(440.0, 0.4, SR, 0.9, **kw)), kw
    y0 = fd.interceptor_string(110.0, 0.4, SR, 0.9)
    y1 = fd.interceptor_string(110.0, 0.4, SR, 0.9, kind="tuba")
    assert fin(y0) and fin(y1) and not np.allclose(y0, y1)


def test_spring_uniform_wander():
    x = np.zeros(SR // 2)
    x[100] = 1.0
    a = fd.spring_reverb(x, SR)
    b = fd.spring_reverb(x, SR, wander="uniform")
    assert fin(b) and not np.allclose(a, b)


# ------------------------------------------------------------------ synths, klang, acoustic
def test_fm_bass_ratio_and_mex():
    y0 = S.fm_bass(55.0, 0.5, SR, 0.8)
    assert np.allclose(y0, S.fm_bass(55.0, 0.5, SR, 0.8, ratio=1.0))
    assert not np.allclose(y0, S.fm_bass(55.0, 0.5, SR, 0.8, ratio=2.0))
    y = S.fm_bass(55.0, 0.5, SR, 0.8, model="mex")
    assert fin(y) and abs(pitch_cents(y, 55.0)) < 10.0


def test_bassoon_meyer_and_bell_diameter():
    assert fin(K.bassoon(110.0, 0.5, SR, 0.8, meyer=1, precursor=0.5))
    a = K.distant_bell(SR, dur=1.0, diameter=0.5)
    b = K.distant_bell(SR, dur=1.0, diameter=1.5)
    assert fin(a) and fin(b) and not np.allclose(a[:min(a.size, b.size)], b[:min(a.size, b.size)])


def test_harp_low_end_in_tune():
    from genny import instruments as I
    from genny.notes import midi_to_freq, note_to_midi
    for m in (note_to_midi("Cb1"), note_to_midi("E1"), note_to_midi("A1")):
        f = midi_to_freq(m)
        assert abs(pitch_cents(I.render_note("harp", f, 0.6, SR, 0.9), f)) < 10.0, m
    assert A.harp_stretch(30.87) <= A.HARP_TUNE_CAP and A.harp_stretch(1000.0) < 0.5


def test_clav_tone_and_grey_chiff():
    y0 = A.clav(220.0, 0.5, SR, 0.8)
    assert np.allclose(y0, A.clav(220.0, 0.5, SR, 0.8, tone=""))
    yb = A.clav(220.0, 0.5, SR, 0.8, tone="B")
    ym = A.clav(220.0, 0.5, SR, 0.8, tone="M")
    assert band_db(yb, 2000, 8000) - band_db(yb, 100, 800) > band_db(ym, 2000, 8000) - band_db(ym, 100, 800)
    n = SR // 2
    c = A.grey_chiff(n, SR, 220.0, 1.0)
    m = int(A.CHIFF_S * SR)
    assert abs(20 * np.log10(np.sqrt(np.mean(c[:m] ** 2))) - A.CHIFF_DB) < 0.1
    assert fin(A.church_organ(220.0, 0.4, SR, 0.8, chiff="grey"))


def test_rhodes_tonebar_two_dof():
    f = 220.0
    L = A.rhodes_tine_length(f)
    (f1, a1), (f2, a2) = A.rhodes_tonebar_modes(f, f, L, f)
    assert abs((f2 - f1) / f - A.RHODES_FORK_SPLIT) < 2e-4 and abs(a1 - 0.5) < 0.01
    (f1, a1), (f2, a2) = A.rhodes_tonebar_modes(f, f * 2 ** (20 / 1200), L, f)
    assert a1 > 0.9
    assert abs(A.tonebar_freq(A.TONEBAR_L) - 412.2) < 1.0
    y0 = A.rhodes(f, 0.5, SR, 0.8)
    assert fin(A.rhodes(f, 0.5, SR, 0.8, tonebar="2dof"))
    assert pitch_cents(A.rhodes(f, 0.5, SR, 0.8, tuning_mass=0.05), f) < pitch_cents(y0, f) - 20.0


def test_horizontal_pickup_adds_even_harmonic():
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 110 * t)
    u0 = np.abs(np.fft.rfft(A.magnetic_pickup(x, SR)))
    u1 = np.abs(np.fft.rfft(A.magnetic_pickup(x, SR, horizontal=20.0)))
    assert u1[220] / u1[110] > u0[220] / u0[110]


def test_guitar_decay_targets():
    lv = []
    for f in (146.8, 196.0, 329.6):
        y = A.guitar(f, 2.4, SR, 0.8)
        w = int(0.05 * SR)
        e0 = np.sqrt(np.mean(y[int(0.02 * SR):int(0.02 * SR) + w] ** 2))
        e2 = np.sqrt(np.mean(y[2 * SR - w // 2:2 * SR + w // 2] ** 2))
        lv.append(20 * np.log10(e2 / e0))
    assert all(-40.0 < v < -12.0 for v in lv) and lv[0] > lv[-1]


def test_q_ranges():
    for f, q, _ in A._GUITAR_LOW:
        assert 10.0 <= q <= 100.0
    assert 10.0 <= BW.WOLF_Q <= 150.0
    for f in (110.0, 440.0, 1760.0):
        sigma = bp.SB_A * f * f + bp.SB_B * f
        assert 1.0 <= math.pi * f / sigma <= 1e4


# ------------------------------------------------------------------ plates, waveguides, chiptune2, bow
def test_airy_dst_matches_cholesky():
    mdl = PL._model(16000, 1.3, "ssss", 12.0, 0.0, 1.0, 0.0005, 0.3)
    f = PL.pulse(2000, 16000, 2e4, 1.0)
    a = PL.simulate_nl(mdl, f, "vk")[0]
    b = PL.simulate_nl(mdl, f, "vk", airy="dst")[0]
    assert fin(b) and abs(np.sqrt(np.mean(b ** 2)) / np.sqrt(np.mean(a ** 2)) - 1.0) < 0.02


def test_fd_drum_under_flip():
    y0 = PL.fd_drum(SR, 0.8)
    y1 = PL.fd_drum(SR, 0.8, under=0.6)
    y2 = PL.fd_drum(SR, 0.8, under=0.6, flip=1)
    assert fin(y1) and not np.allclose(y0, y1) and not np.allclose(y1, y2)


def test_modular_string_in_tune():
    for p in W.BRIDGE_PRESETS:
        y = W.modular_string(220.0, 1.0, np.random.default_rng(0), p)
        assert fin(y) and abs(pitch_cents(y, 220.0, W.SR)) < 2.0, p


def test_jaw_harp_paths_and_wolf():
    for path in ("figure8", "triangle"):
        assert fin(C2.jaw_harp(110.0, 0.4, SR, 0.8, path=path))
    y0 = BW.bowed(196.0, 0.4, SR)
    y1 = BW.bowed(196.0, 0.4, SR, wolf=0.01)
    assert fin(y1) and not np.allclose(y0, y1)
