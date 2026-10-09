"""genny.reverbs: measurable predictions of every model + a smoke loop over every registered name.

Run:  uv run --with pytest pytest tests/test_reverbs.py -q     (or: uv run python tests/test_reverbs.py)
"""
import math
import time

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from genny import fx as FX
from genny import instruments as I
from genny import reverbs as R
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100
FX_NAMES = ["jcrev", "satrev", "fdn_reverb", "room", "zita", "early_reflections", "distance", "doppler", "occlude",
            "leslie", "flanger", "phaser", "chorus", "tape_delay", "shimmer", "gated_reverb", "reverse_reverb"]
C = R.C_AIR


def impulse(n, at=0):
    x = np.zeros(n)
    x[at] = 1.0
    return x


def decay_time(ir, sr, lo=-10.0, hi=-30.0):
    """T60 from the slope of the Schroeder backward integral between ``lo`` and ``hi`` dB."""
    e = np.cumsum((ir ** 2)[::-1])[::-1]
    edc = 10 * np.log10(e / e[0] + 1e-30)
    i0, i1 = np.nonzero(edc <= lo)[0][0], np.nonzero(edc <= hi)[0][0]
    return -60.0 / np.polyfit(np.arange(i0, i1) / sr, edc[i0:i1], 1)[0]


def band_t60(ir, sr, fc, lo=-5.0, hi=-25.0):
    y = sosfiltfilt(butter(4, [fc / 2 ** 0.5, fc * 2 ** 0.5], "band", fs=sr, output="sos"), ir)
    return decay_time(y, sr, lo, hi)


def band_db(x, fc, sr=SR):
    return 20 * np.log10(rms(sosfiltfilt(butter(4, [fc / 1.1, fc * 1.1], "band", fs=sr, output="sos"), x)))


def peak_freq(seg, sr):
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    k = int(np.argmax(S))
    k = k + 0.5 * (S[k - 1] - S[k + 1]) / (S[k - 1] - 2 * S[k] + S[k + 1])
    return k * sr / (8 * len(seg))


def notch_freqs(y, sr, depth_db=20.0):
    H = 20 * np.log10(np.abs(np.fft.rfft(y)) + 1e-12)
    pk, _ = find_peaks(-H, height=depth_db)
    return np.fft.rfftfreq(len(y), 1 / sr)[pk]


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def cents_off(y, f0, sr=SR):
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


# ---------------------------------------------------------------------------------- FDN
def test_fdn_t60_low_and_high():
    for sr, tl, th, lines, mat in ((44100, 2.0, 1.0, 16, "householder"), (48000, 1.2, 0.5, 8, "hadamard"),
                                   (22050, 3.0, 1.5, 16, "hadamard")):
        y = R.fdn_reverb(impulse(int(sr * tl * 2.5)), sr, t60_low=tl, t60_high=th, lines=lines, matrix=mat,
                         mix=1.0, predelay=0.0, tail=0.0)[:, 0]
        m_lo, m_hi = band_t60(y, sr, 250.0), band_t60(y, sr, 4000.0)
        assert abs(m_lo / tl - 1) < 0.2, (sr, "250 Hz", tl, round(m_lo, 3))
        assert abs(m_hi / th - 1) < 0.2, (sr, "4 kHz", th, round(m_hi, 3))


def test_fdn_lossless_prototype():
    for n in (8, 16):
        for mat in ("householder", "hadamard"):
            A = R.feedback_matrix(mat, n)
            assert np.max(np.abs(A.T @ A - np.eye(n))) < 1e-12, (mat, n, "not unitary")
            e = R.fdn_impulse_energy(n, mat, SR, 6.0, 3.0)
            dev = float(np.max(np.abs(e / e[0] - 1.0)))
            assert len(e) > 30 and dev < 1e-6, (mat, n, dev)
    M = R.fdn_lengths(SR, 8.0, 16)
    assert all(math.gcd(int(a), int(b)) == 1 for i, a in enumerate(M) for b in M[:i])
    assert abs(M.mean() / (8.0 / C * SR) - 1) < 0.05                      # mean = mean free path / c


def test_fdn_damping_always_stable():
    for tl, th in ((0.1, 0.03), (2.0, 1.0), (30.0, 0.05), (10.0, 10.0), (5.0, 9.0)):
        for sr in (22050, 48000):
            g, p, alpha = R.fdn_damping(R.fdn_lengths(sr, 20.0, 16), sr, tl, th)
            assert np.all(g / (1 - p) < 1.0) and np.all(np.abs(p) < 1.0) and 0.1 <= alpha <= 1.0, (tl, th, sr)


# ---------------------------------------------------------------------------------- Schroeder
def test_schroeder_lengths_and_echo_density():
    assert R.schroeder_lengths("jcrev", 25000) == {"ap": [347, 113, 37], "comb": [1687, 1601, 2053, 2251]}
    assert R.schroeder_lengths("satrev", 25000) == {"ap": [125, 42, 12], "comb": [901, 778, 1011, 1123]}
    assert R.schroeder_lengths("samson", 25000) == {"ap": [1051, 337, 113], "comb": [4799, 4999, 5399, 5801]}
    for sr in (22050, 25000, 44100, 48000):
        # SATREV's published 125 / 42 / 12 share factors, so it is only coprime once rescaled
        for name in ("jcrev", "samson") + (() if sr == 25000 else ("satrev",)):
            L = R.schroeder_lengths(name, sr)
            a = L["ap"] + L["comb"]
            assert all(math.gcd(a[i], a[j]) == 1 for i in range(7) for j in range(i)), (name, sr, a)
            ref = R.SCHROEDER[name]["ap"] + R.SCHROEDER[name]["comb"]
            assert all(abs(v - r * sr / 25000) < max(3.0, 0.03 * v) for v, r in zip(a, ref)), (name, sr, a)
    for variant in ("mus10", "samson"):
        y = R.jcrev(impulse(SR), SR, mix=1.0, variant=variant, tail=0.0)[:, 0]
        thr = 1e-6 * np.max(np.abs(y))
        w = int(0.03 * SR)
        dens = [int(np.sum(np.abs(y[k * w:(k + 1) * w]) > thr)) for k in range(4)]   # echoes per 30 ms
        assert dens[0] < dens[1] < dens[2] <= dens[3], (variant, dens)
    # decay of the published comb gains: T60 = 3 N / (fs * -log10 g), longest-ringing comb
    y = R.jcrev(impulse(3 * SR), SR, mix=1.0, tail=0.0)[:, 0]
    spec = R.SCHROEDER["jcrev"]
    t_pred = max(-3 * n / (25000 * math.log10(g)) for n, g in zip(spec["comb"], spec["comb_g"]))
    t_meas = decay_time(y, SR, -20, -40)
    assert abs(t_meas / t_pred - 1) < 0.2, (t_meas, t_pred)


# ---------------------------------------------------------------------------------- propagation
def test_doppler_pitch_ratio():
    n = 4 * SR
    s = np.sin(2 * np.pi * 1000.0 * np.arange(n) / SR)
    for v in (10.0, 30.0, 60.0):
        y = R.doppler(s, SR, speed=v, closest=1.0, pan=False, air=False)
        fa = peak_freq(y[int(0.1 * SR):int(0.4 * SR)], SR) / 1000.0
        fl = peak_freq(y[-int(0.4 * SR):-int(0.1 * SR)], SR) / 1000.0
        assert abs(fa / (C / (C - v)) - 1) < 0.01, (v, fa, C / (C - v))
        assert abs(fl / (C / (C + v)) - 1) < 0.01, (v, fl, C / (C + v))
    # 1/r swell: loudest at the pass, and the pan crosses from left to right
    y = R.doppler(s, SR, speed=30.0, closest=5.0)
    env = np.convolve(np.abs(y).sum(axis=1), np.ones(2000) / 2000, "same")
    assert abs(np.argmax(env) / len(env) - 0.5) < 0.05
    q = len(y) // 4
    assert rms(y[:q, 0]) > 2 * rms(y[:q, 1]) and rms(y[-q:, 1]) > 2 * rms(y[-q:, 0])


def test_distance_inverse_law_and_air():
    nz = np.random.default_rng(0).standard_normal(SR)
    for m in (1.0, 2.0, 10.0, 40.0):
        v = rms(R.distance(nz, SR, metres=m, air=False)) / rms(nz)
        assert abs(v * m - 1) < 0.01, (m, v)                              # gain = 1 / r
    g3 = rms(R.distance(nz, SR, metres=4.0, air=False, line_source=True)) / rms(nz)
    assert abs(g3 - 0.5) < 0.01                                           # line source: 1 / sqrt(r)
    far, ref = R.distance(nz, SR, metres=201.0), R.distance(nz, SR, metres=201.0, air=False)
    loss4k, loss1k = band_db(ref, 4000) - band_db(far, 4000), band_db(ref, 1000) - band_db(far, 1000)
    assert abs(loss4k - 18.0) < 2.5 and abs(loss1k - 1.12) < 0.5, (loss4k, loss1k)   # PASP B.2: 90 and 5.6 dB/km
    dry = R.distance(nz, SR, metres=201.0, humidity=40)
    assert band_db(ref, 4000) - band_db(dry, 4000) > loss4k + 1.5         # drier air absorbs more at 4 kHz
    # reverb send: the direct / reverberant ratio falls with distance
    near, away = R.distance(nz, SR, metres=2.0, reverb=0.5), R.distance(nz, SR, metres=50.0, reverb=0.5)
    tail = slice(SR + 2000, SR + 20000)
    assert abs(rms(near[tail]) / rms(away[tail]) - 1) < 0.01             # same reverberant level ...
    dn = rms(R.distance(nz, SR, metres=2.0, air=False)) / rms(near[tail])
    da = rms(R.distance(nz, SR, metres=50.0, air=False)) / rms(away[tail])
    assert abs(dn / da / 25.0 - 1) < 0.01, (dn, da)                       # ... under a direct sound 25 x weaker


def test_occlusion_mass_law():
    nz = np.random.default_rng(1).standard_normal(SR)
    y = R.occlude(nz, SR, material="brick", thickness=0.1, leak=0.0)
    m = 1900.0 * 0.1
    for f in (500.0, 1000.0, 2000.0):
        tl = band_db(nz, f) - band_db(y, f)
        assert abs(tl - (20 * math.log10(f * m) - 47.0)) < 1.5, (f, tl)   # mass law, 6 dB per octave
    thin = R.occlude(nz, SR, material="glass", thickness=0.004, leak=0.0)
    assert band_db(thin, 1000) > band_db(y, 1000) + 20
    assert band_db(R.occlude(nz, SR, leak=0.5), 300) > band_db(y, 300) + 10   # the diffracted path fills in


def test_early_reflections_geometry():
    dims, src, lis = (6.0, 8.0, 3.4), np.array([3.5, 6.0, 1.4]), np.array([2.5, 2.0, 1.6])
    surf = ("plaster",) * 4 + ("wood", "plaster")
    delays, gains, sx, d0 = R.image_source_taps(dims, src, lis, surf, order=2, tmax=0.1)
    assert abs(d0 - np.linalg.norm(src - lis)) < 1e-9
    horiz = math.hypot(src[0] - lis[0], src[1] - lis[1])
    floor = (math.hypot(horiz, src[2] + lis[2]) - d0) / C                 # mirror the source in the floor
    ceil = (math.hypot(horiz, (3.4 - src[2]) + (3.4 - lis[2])) - d0) / C
    for t, mat in ((floor, "wood"), (ceil, "plaster")):
        i = int(np.argmin(np.abs(delays - t)))
        assert abs(delays[i] - t) < 1e-9
        beta = np.sqrt(1 - np.array(R.ABSORPTION[mat]))
        assert np.allclose(gains[i], d0 / (d0 + t * C) * beta, rtol=1e-9)
    assert np.all(np.diff(delays) >= 0) and len(delays) > 20
    dead = R.image_source_taps(dims, src, lis, ("open",) * 6, order=2)[1]
    assert np.max(np.abs(dead)) < 0.02                                    # open walls return (almost) nothing
    y = R.early_reflections(impulse(2000, 100), SR, preset="bathroom", mix=1.0)
    assert y.shape[1] == 2 and np.sum(y[400:] ** 2) > 0.2 * np.sum(y ** 2)


def test_room_presets_ordered_by_decay():
    indoor = ["closet", "studio", "bedroom", "bathroom", "hall", "church", "cave"]
    outdoor = ["forest", "street", "canyon"]
    assert set(indoor + outdoor) == set(R.ROOM_PRESETS)
    sab = {p: float(np.mean(R.room_info(p)["t60"][2:4])) for p in R.ROOM_PRESETS}
    meas = {p: decay_time(R.room(impulse(SR // 10, 10), SR, preset=p, mix=1.0).mean(axis=1), SR) for p in R.ROOM_PRESETS}
    for chain in (indoor, outdoor):
        for a, b in zip(chain, chain[1:]):
            assert sab[a] < sab[b], ("sabine", a, sab[a], b, sab[b])
            assert meas[a] < meas[b], ("measured", a, meas[a], b, meas[b])
    assert 1.6 < sab["hall"] < 2.3                                        # PASP: good halls about 1.9 s
    # explicit dimensions and materials: Sabine by hand, and a harder room rings longer
    kw_hard = dict(lx=4.0, ly=5.0, lz=3.0, walls="concrete", floor="concrete", ceiling="concrete")
    kw_soft = dict(lx=4.0, ly=5.0, lz=3.0, walls="fabric", floor="carpet", ceiling="fabric")
    info = R.room_info("bedroom", **kw_hard)
    assert info["volume"] == 60.0 and info["area"] == 94.0 and abs(info["mfp"] - 4 * 60 / 94) < 1e-9
    assert np.all(info["t60"] > R.room_info("bedroom", **kw_soft)["t60"])
    hard = decay_time(R.room(impulse(SR // 10, 10), SR, mix=1.0, **kw_hard).mean(axis=1), SR)
    dead = decay_time(R.room(impulse(SR // 10, 10), SR, mix=1.0, **kw_soft).mean(axis=1), SR)
    assert hard > 1.5 * dead, (hard, dead)
    for model in ("zita", "puckette", "farnell"):                         # the exposed space.py kernels
        y = R.room(impulse(SR // 10, 10), SR, preset="hall", model=model, mix=1.0)
        t = decay_time(y.mean(axis=1), SR)
        assert np.all(np.isfinite(y)) and 0.3 < t < 3.0, (model, t)


def test_zita_kernel_exposed():
    for sr in (44100, 22050):
        y = R.zita(impulse(int(2.5 * sr)), sr, mix=1.0, t60_low=1.5, t60_mid=1.0, predelay=0.0, tail=0.0).mean(axis=1)
        t = band_t60(y, sr, 1000.0, -10, -30)
        assert abs(t / 1.0 - 1) < 0.25, (sr, t)
    e = float(np.sum(R.zita(impulse(4 * SR), SR, mix=1.0)[:, 0] ** 2))
    assert 0.5 < e < 2.0, e                                               # unit-energy impulse response


# ---------------------------------------------------------------------------------- modulation effects
def test_phaser_notch_count_and_motion():
    for stages in (4, 6, 8):
        a = notch_freqs(R.phaser(impulse(1 << 16), SR, rate=0.0, stages=stages, phase=0.0), SR)
        b = notch_freqs(R.phaser(impulse(1 << 16), SR, rate=0.0, stages=stages, phase=0.25), SR)
        assert len(a) == stages // 2 and len(b) == stages // 2, (stages, a, b)
        assert np.all(b[:2] > 1.5 * a[:2]) and np.all(b >= a), (stages, a, b)   # the LFO moves the notches
        assert np.all(np.diff(np.log(a)) > 1.0)                           # exponentially spaced, not harmonic
    # 4 stages at 100/200/400/800 Hz: chain phase = 3 pi and pi -> two notches (analog prototype)
    f = np.array([100.0, 200.0, 400.0, 800.0])
    grid = np.linspace(20, 20000, 400000)
    ph = np.sum(2 * np.arctan(f[None, :] / grid[:, None]), axis=1)
    pred = [grid[np.argmin(np.abs(ph - k * np.pi))] for k in (3, 1)]
    got = notch_freqs(R.phaser(impulse(1 << 18), 4 * SR, rate=0.0, depth=0.0, stages=4), 4 * SR)
    assert np.allclose(got, pred, rtol=0.03), (got, pred)
    dc = np.abs(np.fft.rfft(R.phaser(impulse(1 << 16), SR, rate=0.0)))[0]
    assert abs(dc - 1.0) < 1e-2, dc                                       # DC gain 1 for an even number of stages


def test_flanger_notch_spacing_and_regeneration():
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    for d_ms in (1.0, 2.5):
        y = R.flanger(impulse(1 << 16), SR, rate=0.0, depth=0.0, delay=d_ms, feedback=0.0, mix=0.5)
        nf = notch_freqs(y, SR)
        nf = nf[nf < 12000]
        D = d_ms / 1000.0
        assert abs(np.mean(np.diff(nf)) * D - 1) < 0.01, (d_ms, np.mean(np.diff(nf)))   # spacing = 1 / delay
        assert abs(nf[0] * 2 * D - 1) < 0.02                                             # first notch at 1 / (2 delay)
    inv = R.flanger(impulse(1 << 16), SR, rate=0.0, depth=0.0, delay=1.0, feedback=0.0, mix=0.5, invert=True)
    assert np.abs(np.fft.rfft(inv))[0] < 1e-3 and abs(notch_freqs(inv, SR)[0] - 1000.0) < 15   # notch at DC, then 1/delay
    # true recursion: with regeneration a the peaks rise to (1 + 1 / (1 - a)) / 2 and the response rings on
    a = 0.7
    reg = R.flanger(impulse(1 << 16), SR, rate=0.0, depth=0.0, delay=1.0, feedback=a, mix=0.5)
    Hr = np.abs(np.fft.rfft(reg))
    k = int(np.argmin(np.abs(f - 2000.0)))
    assert abs(Hr[k - 40:k + 40].max() - 0.5 * (1 + 1 / (1 - a))) < 0.05, Hr[k - 40:k + 40].max()
    plain = R.flanger(impulse(1 << 16), SR, rate=0.0, depth=0.0, feedback=0.0)
    assert np.sum(reg[400:] ** 2) > 1e-4 and np.sum(plain[400:] ** 2) < 1e-12
    whole = R.flanger(impulse(4096), SR, rate=0.0, depth=0.0, delay=100e3 / SR, feedback=a, mix=0.5)   # 100 samples
    assert np.allclose(whole[100 * np.arange(1, 7)], 0.5 * a ** np.arange(6), atol=1e-6), whole[100 * np.arange(1, 7)]
    # the swept delay really moves: the first notch follows the LFO
    lo = notch_freqs(R.flanger(impulse(1 << 16), SR, rate=0.0, depth=4.0, feedback=0.0, phase=0.25), SR)[0]
    hi = notch_freqs(R.flanger(impulse(1 << 16), SR, rate=0.0, depth=4.0, feedback=0.0, phase=0.75), SR)[0]
    assert abs(lo - 100.0) < 3 and abs(hi - 500.0) < 8, (lo, hi)          # delays 5 ms and 1 ms


def test_old_parameter_names_still_work():
    x = 0.3 * np.random.default_rng(3).standard_normal(SR // 2)
    y = FX.apply_chain(x, [{"type": "flanger", "rate": 0.3, "depth": 2.0, "feedback": 0.5, "mix": 0.5}], SR)
    assert y.shape == x.shape
    y = FX.apply_chain(x, [{"type": "phaser", "rate": 0.5, "depth": 0.7, "stages": 4, "mix": 0.5}], SR)
    assert y.shape == x.shape
    y = FX.apply_chain(x, [{"type": "chorus", "rate": 0.8, "depth": 3.0, "mix": 0.5, "voices": 2}], SR)
    assert y.shape == (len(x), 2)
    for name, old in (("flanger", ("rate", "depth", "feedback", "mix")), ("phaser", ("rate", "depth", "stages", "mix")),
                      ("chorus", ("rate", "depth", "mix", "voices"))):
        assert FX.REGISTRY[name]["fn"].__module__ == "genny.reverbs"
        assert all(k in FX.REGISTRY[name]["params"] for k in old)
    # chorus voices are interpolated delays: a sine comes out frequency-modulated by 2 pi rate x swing
    s = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    c = R.chorus(s, SR, rate=2.0, depth=4.0, mix=1.0, voices=1)[:, 0]
    inst = np.diff(np.unwrap(np.angle(hilbert(c)))) * SR / (2 * np.pi)
    dev = (np.percentile(inst[4000:-4000], 99.5) - np.percentile(inst[4000:-4000], 0.5)) / 2 / 1000.0
    assert abs(dev / (2 * np.pi * 2.0 * 0.002) - 1) < 0.1, dev            # delay swing +-2 ms at 2 Hz
    v = R.vibrato(s, SR, rate=5.0, depth=0.5)
    inst = np.diff(np.unwrap(np.angle(hilbert(v)))) * SR / (2 * np.pi)
    dev = (np.percentile(inst[4000:-4000], 99.5) - np.percentile(inst[4000:-4000], 0.5)) / 2 / 1000.0
    assert v.shape == s.shape and abs(dev / (2 * np.pi * 5.0 * 0.0005) - 1) < 0.1, dev


def test_leslie_rotor_speeds():
    s = np.sin(2 * np.pi * 2000 * np.arange(3 * SR) / SR) * 0.3
    f = np.fft.rfftfreq(1 << 20, 1 / SR)
    for speed, rate in (("fast", 6.7), ("slow", 0.8)):
        y = R.leslie(s, SR, speed=speed, drive=0.0)
        env = np.abs(hilbert(y[:, 0]))[2000:-2000]
        E = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(len(env)), 1 << 20))
        got = f[np.argmax(E[: len(E) // 100])]
        assert abs(got - rate) < 0.25, (speed, got)                       # tremolo at the horn rate
    # far-field Doppler: w_l = w_s [1 - (r_s w_m / c) sin(w_m t)] -> peak deviation about r_s w_m / c
    # (measured on the direct horn path alone: the cabinet image adds a second, opposite shift)
    keep = R._CAB_GAIN
    try:
        R._CAB_GAIN = 0.0
        y = R.leslie(s, SR, speed="fast", drive=0.0, mic=3.0, spread=0.0)[:, 0]
    finally:
        R._CAB_GAIN = keep
    inst = np.diff(np.unwrap(np.angle(hilbert(y)))) * SR / (2 * np.pi)
    inst = np.convolve(inst, np.ones(200) / 200, "same")[5000:-5000]
    dev = (inst.max() - inst.min()) / 2 / 2000.0
    pred = R._HORN_R * 2 * np.pi * 6.7 / C
    assert abs(dev / pred - 1) < 0.15, (dev, pred)
    # brake: no modulation; slow -> fast accelerates
    b = R.leslie(s, SR, speed="brake", drive=0.0)[:, 0]
    assert np.std(np.abs(b[SR:2 * SR].reshape(-1, 441)).max(axis=1)) < 1e-3
    th = R._rotor_angle(3 * SR, SR, 0.8, 6.7, R._LESLIE_TAU[0])
    rate = np.diff(th) * SR / (2 * np.pi)
    assert abs(rate[0] - 0.8) < 0.01 and abs(rate[-1] - 6.7) < 0.2 and np.all(np.diff(rate) > 0)
    acc = R.leslie(s, SR, speed="fast", start="slow", drive=0.0)[:, 0]
    env = sosfiltfilt(butter(2, 30.0, "low", fs=SR, output="sos"), np.abs(hilbert(acc)))

    def trem(seg):
        E = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(len(seg)), 1 << 20))
        return f[np.argmax(E[: len(E) // 100])]
    early, late = trem(env[2000:SR]), trem(env[-SR:-2000])
    assert late > 1.5 * early and abs(late - 6.6) < 0.7, (early, late)   # the tremolo speeds up to 6.7 Hz


def test_tape_delay_echoes():
    T = 0.2
    y = R.tape_delay(impulse(SR, 10), SR, time=T, feedback=0.5, mix=1.0, wow=0.0, flutter=0.0, drive=0.0,
                     tone=20000.0, lowcut=5.0)
    k = int(T * SR)
    w = 300
    eL = [np.max(np.abs(y[10 + j * k - w:10 + j * k + w, 0])) for j in (1, 2, 3)]
    eR = [np.max(np.abs(y[10 + j * k - w:10 + j * k + w, 1])) for j in (1, 2, 3)]
    assert eL[0] > 0.5 and eR[0] < 1e-3 and eR[1] > 0.4 and eL[1] < 1e-3 and eL[2] > 0.2, (eL, eR)   # L, R, L ...
    assert abs(eL[2] / eL[0] - 0.5) < 0.1, eL                             # one round trip = feedback
    dark = R.tape_delay(impulse(SR, 10), SR, time=T, feedback=0.6, mix=1.0, pingpong=False, wow=0, flutter=0,
                        drive=0, tone=1500.0)
    c1 = brightness(dark[k - 500:k + 2000], SR)["centroid"]
    c4 = brightness(dark[4 * k - 500:4 * k + 2000], SR)["centroid"]
    assert dark.ndim == 1 and c4 < 0.75 * c1, (c1, c4)                    # each repeat is duller


def test_gated_reverse_shimmer():
    x = np.zeros(SR)
    x[:2000] = np.random.default_rng(4).standard_normal(2000) * np.hanning(2000) * 0.5
    g = R.gated_reverb(x, SR, mix=1.0, gate=0.2, release=0.02)
    e = np.abs(g).max(axis=1)
    assert e[int(0.15 * SR):int(0.2 * SR)].max() > 0.05 and e[int(0.32 * SR):].max() < 1e-3   # plateau, then cut
    r = R.reverse_reverb(x, SR, mix=1.0, t60=1.0)
    pre = np.abs(r[:SR]).max(axis=1)
    a, b = rms(pre[int(0.2 * SR):int(0.4 * SR)]), rms(pre[int(0.75 * SR):int(0.95 * SR)])
    assert len(r) == len(x) + SR and b > 4 * a, (a, b)                    # swells in ...
    assert rms(pre[SR // 2:]) > 10 * rms(np.abs(r[SR + 4000:]).max(axis=1))   # ... and nothing rings after
    s = np.sin(2 * np.pi * 500 * np.arange(SR // 2) / SR) * np.hanning(SR // 2) * 0.5
    f = np.fft.rfftfreq(32768, 1 / SR)

    def bands(y):
        S = np.abs(np.fft.rfft(y[SR // 2:SR // 2 + 32768] * np.hanning(32768)))
        return {fc: S[(f > fc * 0.94) & (f < fc * 1.06)].max() for fc in (500, 750, 1000, 1500, 2000)}
    on, off = bands(R.shimmer(s, SR, mix=1.0, shimmer=1.0)[:, 0]), bands(R.shimmer(s, SR, mix=1.0, shimmer=0.0)[:, 0])
    assert on[1000] > 0.05 * on[500] and on[1000] > 5 * on[750] and on[2000] > 2 * on[1500], on   # octaves appear
    assert off[1000] < 0.2 * on[1000], (off, on)


# ---------------------------------------------------------------------------------- instrument
def test_feedback_guitar():
    name = "feedback_guitar"
    e = I.REGISTRY[name]
    lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
    for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):                    # shapes and levels (as test_instruments.py)
        for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
            y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
            assert y.ndim == 1 and np.all(np.isfinite(y)), (m, dur)
            assert 1e-4 < np.max(np.abs(y)) < 1.5, (m, dur, float(np.max(np.abs(y))))
            assert abs(y[-1]) < 1e-3 and abs(y[0]) < 0.05 and abs(np.mean(y)) < 5e-3, (m, dur)
    for kw in ({}, {"feedback": 0.0}, {"feedback": 1.0, "drive": 12.0}, {"distance": 0.4, "pickup": 0.4}):   # in tune
        for m in (lo, (lo + hi) // 2, hi):
            c = cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9, **kw), midi_to_freq(m))
            assert abs(c) < 10, (kw, m, round(float(c), 2))
    for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):              # not piercing
        fq = midi_to_freq(m)
        y = I.render_note(name, fq, 0.5, SR, 0.9)
        pure = sharpness(I.render_note("sine", fq, 0.5, SR, 0.9), SR)
        s, b = sharpness(y, SR), brightness(y, SR)
        assert s < max(pure + 1.0, 1.5) and s < 2.4, (m, round(s, 2), round(pure, 2))
        assert b["above_5k"] < 0.03, (m, b["above_5k"])
    levels = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
              for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
    assert abs(sum(levels) / 3 + 12.0) < 3.0, levels
    # the point of the model: without feedback the note dies, with it the note sustains
    for m in (lo, (lo + hi) // 2, hi):
        fq = midi_to_freq(m)
        dead = I.render_note(name, fq, 2.5, SR, 0.9, feedback=0.0)
        live = I.render_note(name, fq, 2.5, SR, 0.9, feedback=0.7)
        late, early = slice(int(2.0 * SR), int(2.4 * SR)), slice(int(0.1 * SR), int(0.3 * SR))
        assert rms(dead[late]) < 0.2 * rms(dead[early]), (m, rms(dead[late]), rms(dead[early]))
        assert rms(live[late]) > 5 * rms(dead[late]) and rms(live[late]) > 0.4 * rms(live[early]), m
    a = I.render_note(name, 220.0, 0.3, SR, 0.8, seed=3)
    assert np.array_equal(a, I.render_note(name, 220.0, 0.3, SR, 0.8, seed=3))
    assert not np.array_equal(a, I.render_note(name, 220.0, 0.3, SR, 0.8, seed=4))


# ---------------------------------------------------------------------------------- smoke
def _test_signal(sr):
    n = sr
    t = np.arange(n) / sr
    x = 0.4 * np.sin(2 * np.pi * 220 * t) * np.exp(-3 * t) + 0.2 * np.random.default_rng(1).standard_normal(n) * np.exp(-8 * t)
    x[:64] *= np.linspace(0, 1, 64)
    x[-400:] *= np.linspace(1, 0, 400)
    return 0.5 * x / np.max(np.abs(x))


VARIANTS = {
    "jcrev": [{"variant": "samson"}, {"t60": 0.6, "damp": 0.5, "width": 0.0}],
    "satrev": [{"invert_right": True, "t60": 3.0}],
    "fdn_reverb": [{"lines": 8, "matrix": "hadamard", "t60_low": 0.3, "t60_high": 0.1, "size": 1.0},
                   {"t60_low": 12.0, "t60_high": 0.2, "size": 40.0, "diffusion": 0.0, "tonal": 0.0, "tail": 1.0}],
    "room": [{"preset": p} for p in ("closet", "bathroom", "canyon", "forest")]
            + [{"model": m, "preset": "church", "tail": 1.0} for m in ("zita", "puckette", "farnell")]
            + [{"lx": 3, "ly": 9, "lz": 2.4, "walls": "glass", "floor": "marble", "ceiling": "open", "distance": 6}],
    "zita": [{"preset": "cave", "tail": 1.0}, {"t60_low": 0.5, "t60_mid": 0.3, "damp": 2000, "xover": 400, "predelay": 0.0}],
    "early_reflections": [{"preset": "closet", "order": 4}, {"preset": "street", "stereo": False, "mix": 1.0}],
    "distance": [{"metres": 500.0, "humidity": 70}, {"metres": 0.3}, {"metres": 30, "reverb": 0.6, "delay": True}],
    "doppler": [{"speed": 250.0, "closest": 30.0, "direction": "rl"}, {"speed": 3.0, "closest": 0.5, "pass_at": 0.1, "pan": False}],
    "occlude": [{"material": "glass", "thickness": 0.004}, {"material": "concrete", "thickness": 0.3, "leak": 0.02}],
    "leslie": [{"speed": "slow", "start": "fast", "drive": 1.0, "mic": 0.3}, {"speed": "brake", "mix": 0.5, "spread": 180}],
    "flanger": [{"feedback": -0.95, "invert": True, "shape": "exp", "stereo": 0.25, "depth": 8.0}, {"shape": "triangle", "delay": 0.2}],
    "phaser": [{"stages": 8, "feedback": 0.9, "depth": 1.0, "stereo": 0.25}, {"stages": 6, "feedback": -0.9, "rate": 4.0}],
    "chorus": [{"voices": 8, "depth": 6.0, "feedback": 0.5}, {"voices": 1, "mix": 1.0}],
    "tape_delay": [{"feedback": 0.95, "drive": 1.0, "wow": 1.0, "flutter": 1.0, "tail": 2.0}, {"pingpong": False, "time": 0.05}],
    "shimmer": [{"shimmer": 1.0, "t60": 6.0, "tail": 1.5}],
    "gated_reverb": [{"gate": 0.05, "boost": 12.0}],
    "reverse_reverb": [{"t60": 0.3, "dark": 1.0}],
}


def test_smoke_every_effect():
    assert set(FX_NAMES) <= set(FX.REGISTRY) and set(VARIANTS) == set(FX_NAMES)
    for name in FX_NAMES:
        entry = FX.REGISTRY[name]
        assert entry["fn"].__module__ == "genny.reverbs", name
        assert all(len(v) == 2 and v[1] for v in entry["params"].values()), name
        for sr in (44100, 22050, 48000):
            x = _test_signal(sr)
            for kw in [{}] + (VARIANTS[name] if sr != 48000 else []):
                assert all(k in entry["params"] for k in kw), (name, kw)
                for inp in (x, np.stack([x, 0.7 * x[::-1]], axis=1)):
                    y = FX.apply_chain(inp, [{"type": name, **kw}], sr)
                    tag = (name, sr, kw, inp.ndim)
                    assert y.ndim in (1, 2) and y.shape[0] >= int(0.9 * len(x)), tag
                    assert np.all(np.isfinite(y)), tag
                    assert 1e-4 < np.max(np.abs(y)) < 1.5, tag + (float(np.max(np.abs(y))),)
                    assert np.max(np.abs(y[0])) < 2e-3 and np.max(np.abs(y[-1])) < 1e-3, tag + ("ends", float(np.max(np.abs(y[0]))), float(np.max(np.abs(y[-1]))))
                    assert np.max(np.abs(np.mean(y, axis=0))) < 3e-3, tag + ("dc",)
                    if kw == {} and inp.ndim == 1 and sr == 44100:
                        assert np.array_equal(y, FX.apply_chain(inp, [{"type": name}], sr)), tag + ("not deterministic",)


def test_speed():
    x = _test_signal(SR)
    for name in FX_NAMES:
        FX.apply_chain(x, [{"type": name}], SR)                           # JIT warm-up
        t0 = time.perf_counter()
        y = FX.apply_chain(x, [{"type": name}], SR)
        dt = time.perf_counter() - t0
        assert dt < 2.0 * len(y) / SR, (name, dt, len(y) / SR)
    I.render_note("feedback_guitar", 110.0, 1.0, SR, 0.9)
    t0 = time.perf_counter()
    I.render_note("feedback_guitar", 110.0, 1.0, SR, 0.9)
    assert time.perf_counter() - t0 < 2.0


if __name__ == "__main__":
    for _k, _fn in sorted(globals().items()):
        if _k.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _k)
