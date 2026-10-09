"""genny.spectral: STFT kernel, time/pitch, sines+noise model, envelopes and the registered effects.

Run:  uv run --with pytest pytest tests/test_spectral.py -q     (or: uv run python tests/test_spectral.py)
Every check is a measurable prediction of the source (JOS, SASP; out/research/g12-jos-sasp.md).
"""
import inspect
import os
import tempfile
import time

import numpy as np
from scipy.signal import hilbert, lfilter

from genny import fx as FX
from genny import spec as SP
from genny import spectral as S
from genny.core import write_wav

SR = 44100
NAMES = ["timestretch", "pitch_shift", "harmonizer", "vocoder", "cross_synth", "spectral_gate", "sines_only", "noise_only",
         "freeze", "spectral_blur", "freqshift", "robotize", "whisperize", "spectral_tilt"]


def peak_hz(y, sr, lo=20.0, hi=None):
    """Frequency of the strongest spectral line of the middle half of y (zero-padded, parabolic)."""
    y = y[len(y) // 4: 3 * len(y) // 4]
    N = 1 << 19
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)), N))
    a, b = int(lo * N / sr), int((hi or sr / 2 - 100) * N / sr)
    k = a + int(np.argmax(X[a:b]))
    l0, l1, l2 = np.log(X[k - 1:k + 2])
    return (k + 0.5 * (l0 - l2) / (l0 - 2 * l1 + l2)) * sr / N


def f0_acf(y, sr, lo=60.0, hi=500.0):
    """Pitch from the autocorrelation peak (parabolic)."""
    y = y[len(y) // 4: 3 * len(y) // 4]
    r = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)
    a, b = int(sr / hi), int(sr / lo)
    k = a + int(np.argmax(r[a:b]))
    return sr / (k + 0.5 * (r[k - 1] - r[k + 1]) / (r[k - 1] - 2 * r[k] + r[k + 1]))


def cents(f, ref):
    return 1200 * np.log2(f / ref)


def am_db(y):
    e = np.abs(hilbert(y))[len(y) // 8: -len(y) // 8]
    return 20 * np.log10(e.max() / e.min())


def sine(f, dur, sr=SR, a=0.5):
    return a * np.sin(2 * np.pi * f * np.arange(int(dur * sr)) / sr)


def voice(sr=SR, f0=110.0, dur=1.2, formants=((1000.0, 100.0),)):
    """Pulse train through formant resonators (the SASP test vowel recipe, g12 §9.3)."""
    n = int(dur * sr)
    x = np.zeros(n)
    x[np.arange(0, n, sr / f0).astype(int)] = 1.0
    for f, b in formants:
        r = np.exp(-np.pi * b / sr)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / sr), r * r], x)
    return 0.5 * x / np.abs(x).max()


def env_peak(y, sr, lo, hi):
    """Peak frequency of the long-term cepstral envelope (2 ms lifter) between lo and hi."""
    M = 2048
    mag = np.sqrt((np.abs(S.stft(y, M, N=2 * M))[12:-12] ** 2).mean(axis=0))
    E = S.cepstral_envelope(mag[None], int(0.002 * sr))[0]
    f = np.arange(len(E)) * sr / (2 * M)
    k = (f > lo) & (f < hi)
    return f[k][np.argmax(E[k])]


# ------------------------------------------------------------------------------------------ kernel

def test_cola():
    for M in (256, 1024, 2048):
        w = S.window(M, root=True)
        assert S.cola_error(w * w, M // 4) < 1e-9                    # root-Hann WOLA at hop M/4
        assert abs(np.sum(w * w) / (M // 4) - 2.0) < 1e-9            # ... and the constant is 2 (used by _pv)
        assert S.cola_error(S.window(M), M // 2) < 1e-9              # Hann at M/2
        assert S.cola_error(S.window(M, "hamming"), M // 2) < 1e-9
        assert S.cola_error(S.window(M, "blackman"), M // 4) < 1e-9
        assert S.cola_error(np.hanning(M), M // 2) > 1e-4            # the symmetric form is NOT COLA (g12 §2.2)


def test_roundtrip():
    x = np.random.default_rng(0).standard_normal(30001)
    for M, N in ((2048, 2048), (1024, 2048), (256, 256)):
        assert np.max(np.abs(S.istft(S.stft(x, M, N=N), len(x), M) - x)) < 1e-9
    assert np.max(np.abs(S.istft(S.stft(np.ones(5000), 512), 5000, 512) - 1.0)) < 1e-9      # dc test (g12 §15)
    st = np.stack([x, x[::-1]], axis=1)
    assert np.max(np.abs(S._wola(st, lambda X: X, 1024, 2048) - st)) < 1e-9


def test_zero_phase_peak():
    """Zero-phase framing: the phase at a peak is the sinusoid's phase at the frame centre (g12 §1.1)."""
    M, f, ph = 1024, 1000.0, 0.7
    n = np.arange(8 * M)
    X = S.stft(np.cos(2 * np.pi * f * n / SR + ph), M)
    m = 12
    centre = m * (M // 4) - M // 2
    k = int(round(f * M / SR))
    want = np.angle(np.exp(1j * (2 * np.pi * f * centre / SR + ph)))
    assert abs(np.angle(X[m, k] * np.exp(-1j * want))) < 1e-3                  # limit: leakage of the negative-frequency image
    assert abs(np.angle(X[m, k + 1] * np.exp(-1j * want))) < 1e-3    # same phase across the main lobe


# ------------------------------------------------------------------------------------------ time / pitch

def test_pv_identity():
    """Rate-1 read positions: the phase vocoder (with locking) is an identity system."""
    x = sine(440, 0.4) + 0.1 * np.random.default_rng(1).standard_normal(int(0.4 * SR))
    M = 2048
    F = len(x) // (M // 4) + 6
    y = S._pv(x, (np.arange(F) - 2) * (M // 4), M, lock=True)[M:M + len(x)]
    assert np.max(np.abs(y - x)) < 1e-9


def test_timestretch_sine():
    for sr in (22050, 44100, 48000):
        x = sine(440, 1.0, sr)
        for mode in ("vocoder", "sola"):
            y = FX.REGISTRY["timestretch"]["fn"](x, sr, factor=2.0, mode=mode)
            assert len(y) == 2 * len(x)
            assert abs(cents(peak_hz(y, sr), 440)) < 3, (sr, mode, peak_hz(y, sr))
            assert am_db(y) < 1.0, (sr, mode, am_db(y))
            assert abs(np.abs(hilbert(y))[len(y) // 2] - 0.5) < 0.01
    x = sine(440, 1.0)
    for fac in (0.5, 0.731, 1.37, 3.0):
        y = S.stretch(x, SR, fac)
        assert len(y) == round(len(x) * fac)
        assert abs(cents(peak_hz(y, SR), 440)) < 3
    st = S.stretch(np.stack([x, sine(660, 1.0)], axis=1), SR, 1.5)
    assert st.shape == (round(len(x) * 1.5), 2)
    assert abs(cents(peak_hz(st[:, 1], SR), 660)) < 3


def test_transient_not_stretched():
    """A click on a tone stays a click (g12 §9.7: translate transients); without it the vocoder smears it."""
    x = sine(220, 1.0, a=0.3)
    x[20000:20100] += np.hanning(100) * np.random.default_rng(2).standard_normal(100)

    def width(y):
        hp = np.diff(y, 2) ** 2
        k = int(np.argmax(hp))
        cs = np.cumsum(hp[k - 3000:k + 3000])
        return np.searchsorted(cs, 0.95 * cs[-1]) - np.searchsorted(cs, 0.05 * cs[-1])

    w_ref = width(x)
    for mode in ("vocoder", "sola"):
        assert width(S.stretch(x, SR, 2.0, mode)) <= w_ref + 8, mode
    assert width(S.stretch(x, SR, 2.0, "vocoder", transients=False)) > 5 * w_ref
    assert len(S.onsets(sine(400, 1.0) + sine(410, 1.0), SR)) == 1          # a 10 Hz beat is not a transient (only the start)


def test_pitch_shift():
    for sr in (22050, 44100):
        x = sine(440, 1.0, sr)
        y = FX.REGISTRY["pitch_shift"]["fn"](x, sr, semitones=12)
        assert len(y) == len(x)
        assert abs(cents(peak_hz(y, sr), 880)) < 5, peak_hz(y, sr)
        assert am_db(y) < 1.0
    y = S.pitch_shift(sine(440, 1.0), SR, semitones=-7, fine=50)
    assert abs(cents(peak_hz(y, SR), 440 * 2 ** (-6.5 / 12))) < 5


def test_formant_keep():
    """Formant-filtered pulse train: plain shift moves the envelope peak by the ratio, keep holds it."""
    x = voice()
    f_ref = env_peak(x, SR, 300, 4000)
    assert abs(f_ref / 1000.0 - 1) < 0.05
    for st in (-5, 4, 7):
        ratio = 2 ** (st / 12)
        moved = env_peak(S.shift_pitch(x, SR, st, "shift"), SR, 300, 4000)
        kept_y = S.shift_pitch(x, SR, st, "keep")
        kept = env_peak(kept_y, SR, 300, 4000)
        assert abs(moved / (f_ref * ratio) - 1) < 0.1, (st, moved)
        assert abs(kept / f_ref - 1) < 0.1, (st, kept)
        assert abs(cents(f0_acf(kept_y, SR), 110 * ratio)) < 10, st       # ... while the pitch did move


def test_freqshift():
    for sr in (22050, 48000):
        y = FX.REGISTRY["freqshift"]["fn"](sine(440, 1.0, sr), sr, hz=100)
        assert abs(peak_hz(y, sr) - 540) < 0.5
        assert am_db(y) < 0.5                                             # single sideband: no image at 340 Hz
    assert abs(peak_hz(S.freqshift(sine(440, 1.0), SR, hz=-140), SR) - 300) < 0.5


# ------------------------------------------------------------------------------------------ sines + noise

def _three():
    t = np.arange(int(1.5 * SR)) / SR
    fs, am, ph = [330.0, 787.3, 1512.9], [0.4, 0.25, 0.15], [0.3, 1.1, 2.0]
    ton = sum(a * np.cos(2 * np.pi * f * t + p) for f, a, p in zip(fs, am, ph))
    return fs, am, ton, 0.03 * np.random.default_rng(1).standard_normal(len(t))


def test_sines_noise_model():
    fs, am, ton, nz = _three()
    m = S.analyze(ton + nz, SR)
    top = np.argsort(-m["amp"].mean(axis=0))[:3]
    got = sorted((float(np.median(m["freq"][m["amp"][:, c] > 0, c])), float(np.median(m["amp"][m["amp"][:, c] > 0, c]))) for c in top)
    for (f, a), f0, a0 in zip(got, fs, am):
        assert abs(f - f0) < 2.0, got                                    # spec: 2 Hz; QIFFT gives ~0.02 Hz
        assert abs(20 * np.log10(a / a0)) < 0.5, got
    sl = slice(SR // 4, len(ton) - SR // 4)
    y = S.resynth(m, noise_gain=0)
    assert len(y) == len(ton)
    assert np.corrcoef(y[sl], ton[sl])[0, 1] > 0.9                      # cubic phase: the waveform itself is recovered
    assert abs(20 * np.log10(np.std(S.resynth(m, sines_gain=0)[sl]) / 0.03)) < 2.0      # residual level
    # transforms: exact length, transposed partial
    y2 = S.resynth(m, stretch=2.0, transpose=7, noise_gain=0)
    assert len(y2) == 2 * len(ton)
    assert abs(cents(peak_hz(y2, SR), 330 * 2 ** (7 / 12))) < 5
    # white noise in -> noise of the same level out
    w = 0.1 * np.random.default_rng(3).standard_normal(SR)
    assert abs(20 * np.log10(np.std(S.resynth(S.analyze(w, SR))[5000:-5000]) / 0.1)) < 1.5
    # other output rate
    y3 = S.resynth(m, sr=22050, noise_gain=0)
    assert abs(len(y3) - len(ton) // 2) <= 1 and np.corrcoef(y3[5000:30000], ton[::2][5000:30000])[0, 1] > 0.9
    # effects
    assert np.corrcoef(S.sines_only(ton + nz, SR)[sl], ton[sl])[0, 1] > 0.9
    assert abs(np.corrcoef(S.noise_only(ton + nz, SR)[sl], ton[sl])[0, 1]) < 0.1


def test_formant_in_resynth():
    """transpose with formant 0 keeps the envelope peak; formant == transpose moves it."""
    m = S.analyze(voice(), SR)
    f_ref = env_peak(S.resynth(m, noise_gain=0), SR, 300, 4000)
    assert abs(env_peak(S.resynth(m, transpose=5, noise_gain=0), SR, 300, 4000) / f_ref - 1) < 0.1
    assert abs(env_peak(S.resynth(m, transpose=5, formant=5, noise_gain=0), SR, 300, 4000) / (f_ref * 2 ** (5 / 12)) - 1) < 0.1


def test_mq_cubic_end_conditions():
    """McAulay-Quatieri cubic (g12 §17 H.4, reconstructed): theta(S) = theta1 (mod 2 pi), theta'(S) = w1."""
    Sn, sr = 512, 48000.0
    f = np.array([[1000.0], [1040.0], [1040.0]])
    w1 = 2 * np.pi * 1040.0 / sr
    w0 = 2 * np.pi * 1000.0 / sr
    t1 = 0.4 + 0.5 * (w0 + w1) * Sn + 0.3 - 2 * np.pi * 7                 # measured phase: 0.3 rad off the linear-frequency path, wrapped
    th = np.array([[0.4], [t1], [t1 + w1 * Sn]])
    y = S._osc_bank(f, np.ones((3, 1)), th, np.array([0, Sn, 2 * Sn]), 2 * Sn, sr, True)
    assert abs(y[0] - np.cos(0.4)) < 1e-12 and abs(y[1] - np.cos(0.4 + 2 * np.pi * 1000.0 / sr)) < 1e-3
    assert abs(y[Sn - 1] - np.cos(t1 - w1)) < 1e-3                       # arrives on the target phase with the target slope
    assert np.max(np.abs(y[Sn:] - np.cos(t1 + w1 * np.arange(Sn)))) < 1e-9
    z = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    inst = sr / np.diff(z + y[z] / (y[z] - y[z + 1]))                     # frequency from upward zero crossings
    assert inst.min() > 995 and inst.max() < 1050, (inst.min(), inst.max())      # "maximally smooth": no detour in frequency


def test_model_io_and_layer():
    fs, am, ton, nz = _three()
    m = S.analyze(ton[:SR // 2], SR)
    with tempfile.TemporaryDirectory() as d:
        for ext in ("npz", "json"):
            p = os.path.join(d, "m." + ext)
            S.save_model(m, p)
            assert np.allclose(S.resynth(S.load_model(p)), S.resynth(m))
        wav = os.path.join(d, "in.wav")
        write_wav(wav, ton, SR)
        y, sr = SP.render_spec({"sr": 22050, "normalize": None, "trim": False, "max_loudness": None, "declick": False,
                                "layers": [{"type": "resynth", "path": wav, "stretch": 1.5, "transpose": 12, "noise": 0}]})
        assert sr == 22050 and abs(len(y) - int(1.5 * 1.5 * 22050)) <= 2
        assert abs(cents(peak_hz(y, sr), 660)) < 5


# ------------------------------------------------------------------------------------------ envelopes

def test_envelopes():
    x = voice(formants=((700.0, 130.0), (2600.0, 160.0)))
    M = 2048
    mag = np.sqrt((np.abs(S.stft(x, M, N=2 * M))[12:-12] ** 2).mean(axis=0))[None]
    f = np.arange(mag.shape[1]) * SR / (2 * M)
    for E in (S.cepstral_envelope(mag, int(0.004 * SR))[0], S.lpc_envelope(mag, 8)[0]):
        for lo, hi, want in ((300, 1500, 700.0), (1800, 3500, 2600.0)):
            k = (f > lo) & (f < hi)
            assert abs(f[k][np.argmax(E[k])] / want - 1) < 0.1, (want, f[k][np.argmax(E[k])])
    # minimum phase from magnitude (g12 §3.2): recovers the impulse response of a minimum-phase filter
    N = 8192
    imp = np.zeros(N)
    imp[0] = 1.0
    h = lfilter([1.0, -0.5], [1.0, -1.6, 0.8], imp)
    H = S.minimum_phase(np.abs(np.fft.rfft(h)), N)
    assert np.max(np.abs(np.fft.irfft(H, N) - h)) < 1e-6
    hl = lfilter([-0.5, 1.0], [1.0, -1.6, 0.8], imp)                       # same magnitude, zero reflected outside the circle
    assert np.max(np.abs(np.fft.irfft(S.minimum_phase(np.abs(np.fft.rfft(hl)), N), N) - h)) < 1e-6


# ------------------------------------------------------------------------------------------ effects

def _band_db(y, sr, edges):
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return np.array([10 * np.log10(P[(f >= a) & (f < b)].sum() + 1e-20) for a, b in zip(edges[:-1], edges[1:])])


def test_vocoder_follows_modulator():
    rng = np.random.default_rng(4)
    n = int(1.5 * SR)
    mod = rng.standard_normal(n)
    for f, b in ((500.0, 150.0), (2400.0, 300.0)):
        r = np.exp(-np.pi * b / SR)
        mod = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r], mod)
    mod *= 0.5 / np.abs(mod).max() * (0.15 + 0.85 * (np.sin(2 * np.pi * 3 * np.arange(n) / SR) > 0))      # 3 Hz gating
    edges = np.array(S.VODER_EDGES, float)
    for carrier in ("saw", "noise", "chord"):
        y = S.vocoder(mod, SR, carrier=carrier)
        a, b = _band_db(mod, SR, edges), _band_db(y, SR, edges)
        assert np.max(np.abs(a - b)) < 4.5, (carrier, a - b)             # each of the ten bands (the follower rides the peaks: ~1-3 dB high)
        assert np.corrcoef(a, b)[0, 1] > 0.98
        ea, eb = (np.sqrt(np.convolve(v ** 2, np.ones(1024) / 1024, "same"))[::512] for v in (mod, y))
        assert np.corrcoef(ea, eb)[0, 1] > 0.9, carrier                  # and the level follows in time
    assert abs(peak_hz(S.vocoder(mod, SR, carrier="saw", freq=150), SR, 100, 220) - 150) < 1.0       # carrier pitch is the pitch
    e3 = np.array([300.0, 700.0, 800.0, 1300.0])
    dn, up = (_band_db(S.vocoder(mod, SR, formant=fs, bands=24), SR, e3) for fs in (0, 12))
    assert dn[0] > dn[2] + 6 and up[2] > up[0] + 3, (dn, up)              # formant +12: the 500 Hz resonance is now near 1 kHz


def test_cross_synth():
    x = voice(formants=((900.0, 120.0),))
    y = S.cross_synth(x, SR)                                              # noise carrier
    assert abs(env_peak(y, SR, 300, 4000) / 900 - 1) < 0.1
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "car.wav")
        write_wav(p, 0.5 * np.random.default_rng(5).uniform(-1, 1, 22050 // 3), 22050)       # other rate, shorter: resampled, looped
        y = S.cross_synth(x, SR, carrier=p)
        assert len(y) == len(x) and abs(env_peak(y, SR, 300, 4000) / 900 - 1) < 0.1


def test_spectral_gate():
    rng = np.random.default_rng(6)
    s, nz = sine(1000, 1.0, a=0.4), 0.02 * rng.standard_normal(SR)

    def parts(y):                                                         # (level at 1 kHz, level elsewhere) in dB
        P = np.abs(np.fft.rfft(y[4096:-4096] * np.hanning(len(y) - 8192))) ** 2
        f = np.fft.rfftfreq(len(y) - 8192, 1 / SR)
        k = np.abs(f - 1000) < 30
        return 10 * np.log10(P[k].sum()), 10 * np.log10(P[~k].sum())

    s0, n0 = parts(s + nz)
    for keep in ("all", "tonal"):
        s1, n1 = parts(S.spectral_gate(s + nz, SR, threshold=-40, reduction=30, keep=keep))
        assert abs(s1 - s0) < 0.5 and n1 < n0 - 20, (keep, s1 - s0, n1 - n0)
    s2, n2 = parts(S.spectral_gate(s + nz, SR, keep="noise"))
    assert s2 < s0 - 20 and abs(n2 - n0) < 1.5, (s2 - s0, n2 - n0)
    assert np.max(np.abs(S.spectral_gate(s + nz, SR, threshold=-200, smoothing=0) - (s + nz))) < 1e-9     # nothing below threshold: identity


def test_robot_whisper_freeze_blur_tilt():
    rng = np.random.default_rng(7)
    v = voice(f0=140.0)
    r = S.robotize(v, SR, pitch=100)
    assert abs(peak_hz(r, SR, 60, 180) - SR / round(SR / 100)) < 1.0      # pitch = frame rate, not the 140 Hz of the input
    assert abs(env_peak(r, SR, 300, 4000) / 1000 - 1) < 0.1               # formant kept

    def tonality(y):
        P = np.abs(np.fft.rfft(y[4096:-4096] * np.hanning(len(y) - 8192)))
        return P.max() / np.median(P[100:20000])

    w = S.whisperize(v, SR)
    assert tonality(w) < 0.05 * tonality(v)                               # harmonics gone
    assert abs(env_peak(w, SR, 300, 4000) / 1000 - 1) < 0.15              # formant still there
    # freeze: 0.5 s of 300 Hz then 0.5 s of 500 Hz, frozen at 0.25 s for 1 s
    x = np.concatenate([sine(300, 0.5), sine(500, 0.5)])
    y = S.freeze(x, SR, at=0.25, length=1.0)
    assert len(y) == len(x) + SR
    mid = y[int(0.4 * SR):int(1.1 * SR)]
    assert abs(peak_hz(mid, SR) - 300) < 1.0 and am_db(mid) < 1.0         # held, steady
    assert abs(peak_hz(y[int(1.6 * SR):], SR) - 500) < 2.0                # then the input continues
    # blur: a 5 ms burst is spread in time
    b = np.zeros(SR)
    b[20000:20220] = 0.5 * rng.standard_normal(220)

    def spread(z):
        e = np.cumsum(z ** 2)
        return np.searchsorted(e, 0.9 * e[-1]) - np.searchsorted(e, 0.1 * e[-1])

    assert spread(S.spectral_blur(b, SR, time=0.08)) > 10 * spread(b)
    # tilt: -6 dB/octave -> an octave apart differs by 6 dB
    t2 = S.spectral_tilt(sine(500, 0.5) + sine(2000, 0.5), SR, slope=-6.0, pivot=1000.0)
    P = np.abs(np.fft.rfft(t2))
    f = np.fft.rfftfreq(len(t2), 1 / SR)
    lv = [20 * np.log10(P[np.abs(f - q) < 20].max()) for q in (500, 2000)]
    assert abs((lv[0] - lv[1]) - 12.0) < 0.3


def test_harmonizer():
    y = S.harmonizer(sine(440, 1.0), SR, intervals="4 7", mix=0.5)          # formant keep must not silence voices of a pure tone
    P = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    for q in (440, 440 * 2 ** (4 / 12), 440 * 2 ** (7 / 12)):
        assert P[np.abs(f - q) < 5].max() > 0.1 * P.max(), q
    assert S._floats([3, 7]) == [3.0, 7.0] and S._floats("-12") == [-12.0] and S._floats(5) == [5.0]


def test_registered_and_smoke():
    """Every registered name: declared params == function params, both sample rates, mono and stereo,
    finite, bounded, deterministic, and fast."""
    assert "resynth" in SP.LAYER_TYPES
    for sr in (22050, 48000):
        rng = np.random.default_rng(8)
        n = sr
        x = voice(sr, 120.0, 1.0, ((700.0, 130.0), (1220.0, 70.0), (2600.0, 160.0)))[:n]
        x[n // 2:n // 2 + sr // 20] += 0.2 * rng.standard_normal(sr // 20) * np.hanning(sr // 20)
        x *= np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n)) / (0.01 * sr))
        st = np.stack([x, 0.7 * x[::-1]], axis=1)
        for name in NAMES:
            reg = FX.REGISTRY[name]
            sig = [p for p in inspect.signature(reg["fn"]).parameters if p not in ("x", "sr")]
            assert sorted(sig) == sorted(reg["params"]), name
            for k, (default, help_) in reg["params"].items():
                assert isinstance(help_, str) and help_, (name, k)
            kw = {"timestretch": {"factor": 1.5}, "pitch_shift": {"semitones": 3, "formant": "keep"}}.get(name, {})
            reg["fn"](x[:2000], sr, **kw)                                 # warm-up (numba, FFT plans)
            t0 = time.perf_counter()
            y = reg["fn"](x, sr, **kw)
            dt = time.perf_counter() - t0
            assert y.ndim == 1 and np.all(np.isfinite(y)), name
            assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, sr, np.max(np.abs(y)))
            assert abs(np.mean(y)) < abs(np.mean(x)) + 0.02, name
            assert dt < 2.0 * max(1.0, len(y) / sr), (name, sr, dt)
            assert np.array_equal(y, reg["fn"](x, sr, **kw)), name        # deterministic
            ys = reg["fn"](st, sr, **kw)
            assert ys.shape == (len(y), 2) and np.all(np.isfinite(ys)) and np.max(np.abs(ys)) < 1.5, name
            for tiny in (np.zeros(5), np.zeros((5, 2)), np.zeros(3000)):   # degenerate input does not blow up
                assert np.all(np.isfinite(reg["fn"](tiny, sr, **kw))), name
        for mode in ("vocoder", "sola"):
            for fac in (0.5, 2.0):
                y = FX.REGISTRY["timestretch"]["fn"](st, sr, factor=fac, mode=mode)
                assert y.shape == (round(n * fac), 2) and np.max(np.abs(y)) < 1.5
    assert FX.apply_chain(sine(440, 0.3), [FX.parse_fx_arg("pitch_shift:semitones=5,formant=keep"), FX.parse_fx_arg("freqshift:hz=30")], SR).shape == (int(0.3 * SR),)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            t0 = time.perf_counter()
            v()
            print(f"ok {k} ({time.perf_counter() - t0:.1f} s)")
