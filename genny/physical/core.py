# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Shared DSP core for the procedural models.

Everything runs at ``SR = 48000``. Constants tuned at another rate are converted with the
``rate_*`` helpers (PLAN.md rule 2):

* decay per sample  ``d48 = d ** (fs_src / SR)``
* pole radius       ``r48 = r ** (fs_src / SR)``
* per-sample prob.  ``p48 = p * fs_src / SR``
* delay in samples  ``n48 = n * SR / fs_src``

Sources
-------
* Contact force pulse ``(1 - cos)`` of duration tau: van den Doel, Kry & Pai, "FoleyAutomatic"
  (SIGGRAPH 2001) section 4 and research 06 §1 ("impact: F(t) = 1 - cos(2 pi t / tau)").
* Felt hammer force ``F = K * x**p`` (Hertz law with exponent p, 2.2-3.5 for piano felt):
  Stulov / Chaigne & Askenfelt via research 10 §5; the pulse shape below integrates that law
  for a mass hitting the felt (the usual half-sine-like but skewed pulse).
* Two-pole resonator with gain normalisation: Steiglitz / STK ``BiQuad::setResonance(normalize)``
  (zeros at +-1, b0 = 0.5 - 0.5 r^2), research 07.
* Time-varying band-pass: RBJ Audio-EQ-Cookbook constant-0dB-peak band-pass, coefficients
  recomputed every sample (PLAN.md rule 3).
* DC blocker: STK ``PoleZero::setBlockZero(0.99)`` form y = x - x1 + r*y1, research 07.
* Power-law envelopes (1-x)^p: research 09 §2 (Farnell "quartic decay" vline~ squared twice),
  research 10 §9.
* Loudness targets: game/assets/production/direction.toml [audio.classes]
  (sfx -20 LUFS / peak -3 dBFS, ui -20 / -6, bed -26, detail -30, music -18, all true peak <= -1).
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

SR: int = 48000


# ------------------------------------------------------------------------------------------
# Rate conversion (PLAN.md rule 2)
# ------------------------------------------------------------------------------------------

def rate_decay(d: float, fs_src: float) -> float:
    """Per-sample decay multiplier tuned at ``fs_src`` converted to SR."""
    return float(d) ** (fs_src / SR)


def rate_radius(r: float, fs_src: float) -> float:
    """Pole radius tuned at ``fs_src`` converted to SR."""
    return float(r) ** (fs_src / SR)


def rate_prob(p: float, fs_src: float) -> float:
    """Per-sample event probability tuned at ``fs_src`` converted to SR."""
    return float(p) * fs_src / SR


def rate_samples(n: float, fs_src: float) -> float:
    """Length in samples at ``fs_src`` converted to SR samples."""
    return float(n) * SR / fs_src


def seconds(t: float) -> int:
    """Seconds to a sample count at SR."""
    return int(round(t * SR))


def t60_to_radius(t60: float) -> float:
    """Pole radius whose envelope falls 60 dB in ``t60`` seconds: r = 10**(-3 / (t60*SR))."""
    return 10.0 ** (-3.0 / (max(t60, 1e-6) * SR))


def decay_rate_to_radius(d: float) -> float:
    """van den Doel decay rate d (1/s, envelope exp(-d t)) to pole radius exp(-d/SR)."""
    return float(np.exp(-d / SR))


# ------------------------------------------------------------------------------------------
# Envelopes (research 09 §2, 10 §9)
# ------------------------------------------------------------------------------------------

def pow_decay(n: int, power: float = 4.0) -> np.ndarray:
    """(1 - x)**power with x running over [0, 1) in n samples (Farnell quartic decay for p=4)."""
    x = np.arange(n) / max(n, 1)
    return (1.0 - x) ** power


def pow_attack(n: int, power: float = 2.0) -> np.ndarray:
    """x**(1/power) style rise is too abrupt; Farnell uses the square of a linear ramp mirrored:
    1 - (1 - x)**power, which starts steep and lands smoothly on 1."""
    x = np.arange(n) / max(n, 1)
    return 1.0 - (1.0 - x) ** power


def ar_env(n: int, attack: float, release: float, a_pow: float = 2.0, r_pow: float = 4.0) -> np.ndarray:
    """Attack (seconds) then power-law release to 0 over the remaining samples."""
    na = min(max(seconds(attack), 1), n)
    nr = max(n - na, 1)
    env = np.empty(n)
    env[:na] = pow_attack(na, a_pow)
    env[na:] = pow_decay(nr, r_pow)[: n - na]
    return env


def exp_decay(n: int, t60: float) -> np.ndarray:
    """Exponential envelope reaching -60 dB at t60 seconds."""
    return t60_to_radius(t60) ** np.arange(n)


# ------------------------------------------------------------------------------------------
# Contact / excitation pulses
# ------------------------------------------------------------------------------------------

def contact_pulse(ms: float) -> np.ndarray:
    """(1 - cos(2 pi t / tau)) force pulse of duration ``ms`` milliseconds, unit area.

    van den Doel et al. FoleyAutomatic §4: a hard contact of short duration tau excites
    frequencies up to about 1/tau; longer (softer / quieter) contacts give a darker strike.
    """
    n = max(int(round(ms * 1e-3 * SR)), 1)
    if n == 1:
        return np.ones(1)
    t = np.arange(n) / n
    p = 1.0 - np.cos(2 * np.pi * t)
    return p / p.sum()


def felt_pulse(ms: float, p: float = 2.5) -> np.ndarray:
    """Force of a mass on a nonlinear felt spring F = K x**p (Hertz law), unit area.

    Integrates m x'' = -K x**p from x=0 with unit entry speed (dimensionless) and scales the
    contact time to ``ms``. Exponent p 2.2-3.5 for piano felt (research 10 §5); p = 1.5 is
    the Hertz sphere-on-plane law. The result is the force curve during contact.
    """
    # Dimensionless integration: x'' = -x**p, x(0)=0, v(0)=1, until x returns to 0.
    dt = 1e-4
    x, v = 0.0, 1.0
    forces = []
    while True:
        f = x ** p if x > 0 else 0.0
        forces.append(f)
        v -= f * dt
        x += v * dt
        if x <= 0 and len(forces) > 10:
            break
        if len(forces) > 10_000_000:
            break
    forces = np.asarray(forces)
    n = max(int(round(ms * 1e-3 * SR)), 2)
    src = np.linspace(0, 1, len(forces))
    out = np.interp(np.linspace(0, 1, n), src, forces)
    s = out.sum()
    return out / s if s > 0 else np.ones(n) / n


# ------------------------------------------------------------------------------------------
# Filters (numba, per-sample)
# ------------------------------------------------------------------------------------------

@numba.njit(cache=True)
def biquad(x, b0, b1, b2, a1, a2):
    """Direct-form-I biquad y = b0 x + b1 x1 + b2 x2 - a1 y1 - a2 y2."""
    n = x.shape[0]
    y = np.empty(n)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for i in range(n):
        xi = x[i]
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = xi
        y2 = y1
        y1 = yi
        y[i] = yi
    return y


@numba.njit(cache=True)
def reson(x, f, r):
    """Two-pole resonator at f Hz, pole radius r, zeros at z = +-1, peak gain ~1.

    STK BiQuad::setResonance(f, r, normalize=true): a1 = -2 r cos(w), a2 = r^2,
    b0 = 0.5 - 0.5 r^2, b1 = 0, b2 = -b0.
    """
    w = 2.0 * np.pi * f / 48000.0
    a1 = -2.0 * r * np.cos(w)
    a2 = r * r
    b0 = 0.5 - 0.5 * r * r
    return biquad(x, b0, 0.0, -b0, a1, a2)


@numba.njit(cache=True)
def tv_bandpass(x, f, q):
    """RBJ constant-0dB-peak band-pass with coefficients recomputed every sample.

    ``f`` and ``q`` are arrays the same length as ``x`` (Hz, Q). Transposed direct form II,
    which stays well-behaved under per-sample coefficient changes.
    """
    n = x.shape[0]
    y = np.empty(n)
    s1 = 0.0
    s2 = 0.0
    for i in range(n):
        fi = f[i]
        if fi < 1.0:
            fi = 1.0
        if fi > 0.49 * 48000.0:
            fi = 0.49 * 48000.0
        w = 2.0 * np.pi * fi / 48000.0
        alpha = np.sin(w) / (2.0 * max(q[i], 1e-3))
        a0 = 1.0 + alpha
        b0 = alpha / a0
        b2 = -alpha / a0
        a1 = -2.0 * np.cos(w) / a0
        a2 = (1.0 - alpha) / a0
        xi = x[i]
        yi = b0 * xi + s1
        s1 = -a1 * yi + s2
        s2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


@numba.njit(cache=True)
def tv_lowpass(x, f, q):
    """RBJ low-pass with per-sample coefficients (TDF-II)."""
    n = x.shape[0]
    y = np.empty(n)
    s1 = 0.0
    s2 = 0.0
    for i in range(n):
        fi = min(max(f[i], 1.0), 0.49 * 48000.0)
        w = 2.0 * np.pi * fi / 48000.0
        cw = np.cos(w)
        alpha = np.sin(w) / (2.0 * max(q[i], 1e-3))
        a0 = 1.0 + alpha
        b0 = (1.0 - cw) * 0.5 / a0
        b1 = (1.0 - cw) / a0
        b2 = b0
        a1 = -2.0 * cw / a0
        a2 = (1.0 - alpha) / a0
        xi = x[i]
        yi = b0 * xi + s1
        s1 = b1 * xi - a1 * yi + s2
        s2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


@numba.njit(cache=True)
def one_pole_lp(x, a):
    """y = (1 - a) x + a y1 (a = pole, 0 <= a < 1)."""
    n = x.shape[0]
    y = np.empty(n)
    s = 0.0
    for i in range(n):
        s = (1.0 - a) * x[i] + a * s
        y[i] = s
    return y


@numba.njit(cache=True)
def tv_one_pole_lp(x, a):
    """One-pole low-pass with a per-sample pole array ``a``."""
    n = x.shape[0]
    y = np.empty(n)
    s = 0.0
    for i in range(n):
        s = (1.0 - a[i]) * x[i] + a[i] * s
        y[i] = s
    return y


@numba.njit(cache=True)
def dc_block(x, r=0.995):
    """y = x - x1 + r y1 (STK PoleZero::setBlockZero)."""
    n = x.shape[0]
    y = np.empty(n)
    x1 = 0.0
    y1 = 0.0
    for i in range(n):
        yi = x[i] - x1 + r * y1
        x1 = x[i]
        y1 = yi
        y[i] = yi
    return y


def one_pole_coef(fc: float) -> float:
    """Pole for a one-pole low-pass with -3 dB near fc: a = exp(-2 pi fc / SR)."""
    return float(np.exp(-2 * np.pi * fc / SR))


def lowpass(x: np.ndarray, fc: float, q: float = 0.7071) -> np.ndarray:
    """Static RBJ low-pass (biquad)."""
    w = 2 * np.pi * min(fc, 0.49 * SR) / SR
    alpha = np.sin(w) / (2 * q)
    cw = np.cos(w)
    a0 = 1 + alpha
    return biquad(np.ascontiguousarray(x, dtype=np.float64), (1 - cw) / 2 / a0, (1 - cw) / a0,
                  (1 - cw) / 2 / a0, -2 * cw / a0, (1 - alpha) / a0)


def highpass(x: np.ndarray, fc: float, q: float = 0.7071) -> np.ndarray:
    """Static RBJ high-pass (biquad)."""
    w = 2 * np.pi * min(fc, 0.49 * SR) / SR
    alpha = np.sin(w) / (2 * q)
    cw = np.cos(w)
    a0 = 1 + alpha
    return biquad(np.ascontiguousarray(x, dtype=np.float64), (1 + cw) / 2 / a0, -(1 + cw) / a0,
                  (1 + cw) / 2 / a0, -2 * cw / a0, (1 - alpha) / a0)


def bandpass(x: np.ndarray, fc: float, q: float) -> np.ndarray:
    """Static RBJ constant-0dB-peak band-pass (biquad)."""
    w = 2 * np.pi * min(fc, 0.49 * SR) / SR
    alpha = np.sin(w) / (2 * q)
    a0 = 1 + alpha
    return biquad(np.ascontiguousarray(x, dtype=np.float64), alpha / a0, 0.0, -alpha / a0,
                  -2 * np.cos(w) / a0, (1 - alpha) / a0)


# ------------------------------------------------------------------------------------------
# Nonlinearity, placement, panning
# ------------------------------------------------------------------------------------------

def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """tanh saturation normalised so small signals pass at unit gain."""
    drive = max(drive, 1e-6)
    return np.tanh(np.asarray(x) * drive) / drive


def pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan, -1 left .. 1 right."""
    angle = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return float(np.cos(angle)), float(np.sin(angle))


def to_stereo(x: np.ndarray, pan: float = 0.0) -> np.ndarray:
    gl, gr = pan_gains(pan)
    return np.stack([x * gl, x * gr], axis=1)


def place(buf: np.ndarray, sig: np.ndarray, at: float, gain: float = 1.0, pan: float = 0.0) -> None:
    """Add ``sig`` (mono or stereo) into ``buf`` (mono or stereo) starting at ``at`` seconds."""
    i = seconds(at)
    if i >= buf.shape[0] or i < 0:
        return
    sig = np.asarray(sig)[: buf.shape[0] - i] * gain
    m = sig.shape[0]
    if buf.ndim == 1:
        buf[i:i + m] += sig if sig.ndim == 1 else sig.mean(axis=1)
        return
    if sig.ndim == 1:
        gl, gr = pan_gains(pan)
        buf[i:i + m, 0] += sig * gl
        buf[i:i + m, 1] += sig * gr
    else:
        buf[i:i + m] += sig


def fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1))


# ------------------------------------------------------------------------------------------
# Mastering and output
# ------------------------------------------------------------------------------------------

# class: (integrated LUFS, sample-peak ceiling dBFS); direction.toml [audio.classes]
MASTER_TARGETS: dict[str, tuple[float, float]] = {
    "sfx": (-20.0, -3.0),
    "ui": (-20.0, -6.0),
    "bed": (-26.0, -1.0),
    "detail": (-30.0, -1.0),
    "music": (-18.0, -1.0),
}
TRUE_PEAK_DB = -1.0


def _true_peak(x: np.ndarray) -> float:
    """4x oversampled peak (ITU-R BS.1770 true-peak approximation)."""
    from scipy.signal import resample_poly
    y = resample_poly(x, 4, 1, axis=0)
    return float(np.max(np.abs(y)) + 1e-12)


def master(x: np.ndarray, cls: str = "sfx", loop: bool = False) -> np.ndarray:
    """Normalise to the class loudness target, respect the peak ceiling and true peak <= -1 dBTP.

    Removes DC with a 20 Hz high-pass, measures integrated loudness (pyloudnorm, BS.1770),
    gains to target, then if the peak exceeds the ceiling applies a soft knee (short sounds)
    or plain gain (loops, so the seam stays whole). Short sounds get a 4 ms power fade-out.
    """
    import pyloudnorm

    lufs, ceiling = MASTER_TARGETS[cls]
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = np.stack([highpass(x[:, c], 20.0) for c in range(x.shape[1])], axis=1)
    else:
        x = highpass(x, 20.0)
    need = seconds(0.5)
    measure = x if len(x) >= need else fit(x, need)
    meter = pyloudnorm.Meter(SR, block_size=min(0.4, len(measure) / SR * 0.99))
    loud = meter.integrated_loudness(measure)
    if not np.isfinite(loud) and np.max(np.abs(measure)) > 1e-9:
        # every 400 ms block sat under BS.1770's -70 LUFS absolute gate (a very quiet render):
        # bring the peak to -20 dBFS first, then measure again
        pre = 10 ** (-20 / 20) / float(np.max(np.abs(measure)))
        x = x * pre
        measure = measure * pre
        loud = meter.integrated_loudness(measure)
    if np.isfinite(loud):
        x = x * 10 ** ((lufs - loud) / 20)
    limit = min(10 ** (ceiling / 20), 10 ** (TRUE_PEAK_DB / 20))
    peak = _true_peak(x)
    if peak > limit:
        if loop:
            x = x * (limit / peak)
        else:
            x = x * min(1.0, 2 * limit / peak)
            x = limit * np.tanh(x / limit)
            x = x * min(1.0, limit / _true_peak(x))
    if not loop:
        nf = min(seconds(0.004), len(x) // 4)
        if nf > 1:
            ramp = pow_decay(nf, 2.0)
            x[-nf:] *= ramp if x.ndim == 1 else ramp[:, None]
    return x


def write_wav(path: str | Path, x: np.ndarray, subtype: str = "PCM_24") -> Path:
    """Write ``x`` to ``path`` at SR (creates folders). Clips at +-1 only as a safety net."""
    import soundfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0), SR, subtype=subtype)
    return path


DEV_DIR = Path(__file__).resolve().parent.parent / "out" / "dev"


# ------------------------------------------------------------------------------------------
# Measurement helpers for the self-tests
# ------------------------------------------------------------------------------------------

def spectrum_peaks(x: np.ndarray, n_peaks: int = 10, fmin: float = 20.0, fmax: float = 20000.0,
                   zero_pad: int = 4) -> list[tuple[float, float]]:
    """Largest local maxima of the Hann-windowed magnitude spectrum: [(Hz, dB)], by level."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n * zero_pad)))
    X = np.abs(np.fft.rfft(x * np.hanning(n), nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    mag = 20 * np.log10(X + 1e-12)
    loc = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
    loc = loc[(f[loc] >= fmin) & (f[loc] <= fmax)]
    loc = loc[np.argsort(mag[loc])[::-1][:n_peaks]]
    return [(float(f[i]), float(mag[i])) for i in loc]


def envelope_db(x: np.ndarray, hop: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """RMS envelope in dB per ``hop`` seconds: (times, dB)."""
    h = max(seconds(hop), 1)
    m = len(x) // h
    frames = x[: m * h].reshape(m, h)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    return np.arange(m) * h / SR, 20 * np.log10(rms)


def decay_rate(x: np.ndarray, start: float = 0.0, span_db: float = 30.0) -> float:
    """Fit the envelope's exponential decay after its maximum: returns d (1/s) in exp(-d t).

    Least-squares slope of the dB envelope from the peak down ``span_db`` dB.
    """
    t, db = envelope_db(x[seconds(start):], hop=0.002)
    i0 = int(np.argmax(db))
    tail = db[i0:]
    stop = np.where(tail < db[i0] - span_db)[0]
    j = stop[0] if len(stop) else len(tail)
    if j < 4:
        return float("nan")
    slope = np.polyfit(t[i0:i0 + j], tail[:j], 1)[0]  # dB/s
    return float(-slope / (20 / np.log(10)))


if __name__ == "__main__":
    out = DEV_DIR / "core"
    rng = np.random.default_rng(1)
    # 1. rate helpers
    assert abs(rate_decay(0.999, 44100) - 0.999 ** (44100 / 48000)) < 1e-15
    assert abs(rate_samples(441, 44100) - 480) < 1e-9
    # 2. contact pulse area and spectrum edge
    p = contact_pulse(1.0)
    print(f"contact_pulse(1 ms): n={len(p)} area={p.sum():.6f}")
    fp = felt_pulse(3.0, 2.5)
    print(f"felt_pulse(3 ms, p=2.5): n={len(fp)} area={fp.sum():.6f} peak at {np.argmax(fp) / len(fp):.2f} of contact")
    # 3. reson peak gain and frequency
    imp = np.zeros(SR)
    imp[0] = 1.0
    noise = rng.standard_normal(SR * 2)
    y = reson(noise, 1000.0, 0.999)
    pk = spectrum_peaks(y, 1)[0]
    print(f"reson 1000 Hz r=0.999: peak {pk[0]:.1f} Hz")
    # 4. tv_bandpass sweep stays finite and tracks
    n = SR * 2
    f = np.geomspace(200, 8000, n)
    y = tv_bandpass(noise[:n], f, np.full(n, 8.0))
    assert np.all(np.isfinite(y))
    seg = y[int(0.9 * n):]
    print(f"tv_bandpass sweep: finite, last-10% peak {spectrum_peaks(seg, 1, fmin=100)[0][0]:.0f} Hz (f there {f[int(0.95 * n)]:.0f})")
    # 5. decay estimate
    t60 = 0.5
    e = noise[:SR] * exp_decay(SR, t60)
    d = decay_rate(reson(e, 2000.0, 0.99))
    print(f"decay_rate of t60=0.5 s envelope: d={d:.2f} /s (expected {3 * np.log(10) / t60:.2f})")
    # 6. master
    y = master(to_stereo(e * 0.1), "sfx")
    print(f"master sfx: true peak {20 * np.log10(_true_peak(y)):.2f} dBTP")
    write_wav(out / "tv_bandpass_sweep.wav", tv_bandpass(noise[:n], f, np.full(n, 8.0)) * 0.3)
    print("core self-test OK")
