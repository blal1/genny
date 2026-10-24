"""Offline calibration and evaluation tools: parameters from recordings, and listening tests.

- ``estimate_B``: inharmonicity of a string tone, Rauhala, Lehtonen & Valimaki, "Fast automatic inharmonicity
  estimation algorithm", JASA 121(5), EL184 (2007): the partial frequency deviation (PFD) method with f1 refinement.
- ``esprit``: modes (frequency, decay, amplitude) of a strike by ESPRIT (shift invariance of the signal subspace of
  a Hankel matrix), plus the energy left in each octave band once they are removed.
- ``rayleigh_fit``: alpha, beta of the Rayleigh damping d = (alpha + beta w^2) / 2 that genny's materials use,
  least squares over fitted modes (the alpha / beta of ObjectFolder and RealImpact objects, from their recordings).
- ``texture_snr``: how close two textures are, as the SNR of their McDermott & Simoncelli statistics
  (``cyber.texture_stats``), per statistic class, the measure of McDermott & Simoncelli (Neuron 71, 2011).
- ``roughness``: Daniel & Weber roughness in asper (Acta Acust. 83, 1997) through MoSQITo (optional extra
  ``genny[eval]``; 1 asper = 1 kHz at 60 dB, 100 % AM at 70 Hz).
- ``webmushra``: a webMUSHRA test (Schoeffler et al., J. Open Res. Softw. 6, 2018) from a reference and conditions:
  the config YAML and the audio, ready to drop into a webMUSHRA install.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------------------- inharmonicity
def _pfd_peaks(x, sr, f1, kmax):
    n = 1 << 16
    X = np.abs(np.fft.rfft(x[:n] * np.blackman(min(len(x), n)), n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    keep = np.zeros(X.size, bool)                          # ten strongest bins in every 5 f1 wide band
    w = max(1, int(5 * f1 / (f[1] - f[0])))
    for a in range(0, min(X.size, int((kmax + 5) * f1 / f[1])), w):
        seg = X[a:a + w]
        keep[a + np.argsort(seg)[-10:]] = True
    return f, np.where(keep, X, 0.0)


def _deviation(f, X, f1, B, kmax):
    df = f[1] - f[0]
    k = np.arange(1, kmax + 1)
    fk = k * f1 / math.sqrt(1.0 + B) * np.sqrt(1.0 + B * k * k)
    fk = np.round(fk / df) * df                             # quantised to the FFT grid (their bias note)
    D = []
    for fh in fk:
        m = (f >= fh - 0.4 * f1) & (f <= fh + 0.4 * f1)
        if not np.any(X[m] > 0):
            break
        D.append(fh - f[m][np.argmax(X[m])])
    return np.array(D)


def estimate_B(x: np.ndarray, sr: int, f0: float, *, kmax: int = 50, B0: float = 1e-4) -> tuple[float, float]:
    """(B, f1) of a string tone near ``f0``. f1 = the strongest peak within a major third of f0, parabolic
    interpolation; then the PFD loop (B x 10^(+-delta), delta halved at each reversal of the deviation trend,
    40 rounds or |delta| < 1e-4), the f1 refinement (f1 x (1 +- mu) from the sign of the first half of D_k,
    mu 0.005 halved at reversals) and the B loop again. Trend sign: D_k = f_hat_k - f_k falling with k means
    the estimated partials fall behind, so B goes up."""
    x = np.asarray(x, float)
    n = 1 << 16
    X = np.abs(np.fft.rfft(x[:n] * np.blackman(min(len(x), n)), n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    band = np.flatnonzero((f > f0 * 2 ** (-4 / 12)) & (f < f0 * 2 ** (4 / 12)))
    i = band[np.argmax(X[band])]
    a, b, c = np.log(X[i - 1:i + 2] + 1e-30)
    f1 = f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
    fp, Xp = _pfd_peaks(x, sr, f1, kmax)

    def b_loop(B):
        delta, last = 1.0, 0
        for _ in range(40):
            D = _deviation(fp, Xp, f1, B, kmax)
            if D.size < 3:
                break
            trend = int(np.sign(np.sum(np.sign(np.diff(D)))))
            if trend == 0:
                break
            if last and trend != last:
                delta /= 2.0
            last = trend
            B *= 10.0 ** (-trend * delta)
            if delta < 1e-4:
                break
        return B

    B = b_loop(B0)
    mu, last = 0.005, 0
    for _ in range(100):
        D = _deviation(fp, Xp, f1, B, kmax)
        if D.size < 4:
            break
        s = int(np.sign(np.mean(D[:D.size // 2])))
        if s == 0:
            break
        if last and s != last:
            mu /= 2.0
        last = s
        f1 *= 1.0 - s * mu                                  # f_hat above the peaks: f1 too high
        if mu < 1e-5:
            break
    return b_loop(B), float(f1)


# ------------------------------------------------------------------------------------------- modes
def esprit(x: np.ndarray, sr: int, n_modes: int = 16, *, dur: float = 0.1, fmin: float = 20.0) -> dict:
    """Modes of the first ``dur`` s after the onset by ESPRIT: Hankel matrix of the segment, signal subspace of
    rank 2 n_modes, poles z from its shift invariance, f = arg z sr / 2 pi, decay = -ln|z| sr (1/s), complex
    amplitudes by least squares. Returns freqs, decays, amps (strongest first) and ``residual``: the energy
    left in each octave band (centres 63 Hz .. 16 kHz) as a share of the segment's energy in that band."""
    x = np.asarray(x, float)
    i0 = int(np.argmax(np.abs(x) > 0.1 * np.max(np.abs(x))))
    seg = x[i0:i0 + int(dur * sr)]
    N = seg.size
    L = N // 2
    H = np.lib.stride_tricks.sliding_window_view(seg, N - L + 1)[:L]
    U, _, _ = np.linalg.svd(H, full_matrices=False)
    r = min(2 * n_modes, U.shape[1] - 1)
    Us = U[:, :r]
    z = np.linalg.eigvals(np.linalg.pinv(Us[:-1]) @ Us[1:])
    V = z[None, :] ** np.arange(N)[:, None]
    amp = np.linalg.lstsq(V, seg.astype(complex), rcond=None)[0]
    fr = np.angle(z) * sr / (2 * np.pi)
    dec = -np.log(np.abs(z) + 1e-300) * sr
    keep = (fr > fmin) & (fr < 0.5 * sr)
    resid = seg - (V @ amp).real
    order = np.argsort(-np.abs(amp[keep]))
    bands = {}
    S, R = np.abs(np.fft.rfft(seg)) ** 2, np.abs(np.fft.rfft(resid)) ** 2
    fq = np.fft.rfftfreq(N, 1.0 / sr)
    for c in 1000.0 * 2.0 ** np.arange(-4, 5):
        m = (fq >= c / math.sqrt(2)) & (fq < c * math.sqrt(2))
        if np.any(m) and S[m].sum() > 0:
            bands[round(float(c))] = float(R[m].sum() / S[m].sum())
    return dict(freqs=fr[keep][order], decays=dec[keep][order], amps=2 * np.abs(amp[keep])[order], residual=bands)


def rayleigh_fit(freqs, decays) -> tuple[float, float]:
    """(alpha, beta) >= 0 of d = (alpha + beta w^2) / 2 fitted to mode decays (1/s) in least squares."""
    from scipy.optimize import nnls
    w2 = (2 * np.pi * np.asarray(freqs, float)) ** 2
    s = 1.0 / max(float(w2.max()), 1.0)                      # scale the beta column (w^2 ~ 1e9)
    sol = nnls(np.stack([np.full(w2.size, 0.5), 0.5 * w2 * s], axis=1), np.asarray(decays, float))[0]
    return float(sol[0]), float(sol[1] * s)


# ------------------------------------------------------------------------------------------- textures
def texture_snr(a: np.ndarray, b: np.ndarray, sr: int) -> dict:
    """SNR (dB) of b's statistics against a's, per class (envelope quantiles, modulation power, envelope
    correlations, C1, C2) and their mean: McDermott & Simoncelli's measure of how well a texture is matched."""
    from .cyber import texture_stats
    sa, sb = texture_stats(a, sr), texture_stats(b, sr)
    out = {}
    for k in ("quantiles", "modulation", "corr", "c1", "c2"):
        x, y = np.abs(np.asarray(sa[k])), np.abs(np.asarray(sb[k]))
        out[k] = float(10 * np.log10(np.sum(x * x) / (np.sum((x - y) ** 2) + 1e-30)))
    out["mean"] = float(np.mean(list(out.values())))
    return out


# ------------------------------------------------------------------------------------------- roughness
def roughness(x: np.ndarray, sr: int, *, level_db: float = 60.0) -> float:
    """Daniel & Weber roughness (asper), the median over 200 ms frames, of ``x`` played at ``level_db`` SPL
    (its RMS); needs ``pip install genny[eval]`` (MoSQITo)."""
    try:
        from mosqito.sq_metrics import roughness_dw
    except ImportError as e:                                # pragma: no cover - optional
        raise ImportError("roughness needs MoSQITo: pip install 'genny[eval]'") from e
    x = np.asarray(x, float)
    x = x.mean(axis=1) if x.ndim == 2 else x
    x = x / (np.sqrt(np.mean(x * x)) + 1e-30) * 2e-5 * 10 ** (level_db / 20)
    R = np.atleast_1d(roughness_dw(x, sr, overlap=0.5)[0])
    return float(np.median(R))


# ------------------------------------------------------------------------------------------- listening test
def _q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def webmushra(reference: str, conditions: dict, out_dir: str, *, name: str = "genny listening test",
              question: str = "Rate how close each sound is to the reference.", anchors: bool = True) -> Path:
    """Write ``out_dir/configs/<test>.yaml`` and copy the audio to ``out_dir/configs/resources/audio/``: one
    MUSHRA page (hidden reference, the conditions, the 3.5 and 7 kHz low-pass anchors unless ``anchors`` is
    false) and a finish page that writes the results. ``conditions``: label -> wav path."""
    out = Path(out_dir)
    audio = out / "configs" / "resources" / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    test = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_") or "genny"

    def put(src, label):
        dst = audio / f"{label}{Path(src).suffix or '.wav'}"
        shutil.copyfile(src, dst)
        return f"configs/resources/audio/{dst.name}"

    ref = put(reference, "reference")
    lines = [f"testname: {_q(name)}", f"testId: {test}", "bufferSize: 2048", "stopOnErrors: true",
             "showButtonPreviousPage: true", "remoteService: service/write.php", "", "pages:",
             "    - type: mushra", "      id: trial1", f"      name: {_q(name)}", f"      content: {_q(question)}",
             "      showWaveform: true", "      enableLooping: true", f"      reference: {ref}",
             f"      createAnchor35: {'true' if anchors else 'false'}",
             f"      createAnchor70: {'true' if anchors else 'false'}", "      stimuli:"]
    for i, (label, path) in enumerate(conditions.items(), 1):
        key = "".join(c if c.isalnum() else "_" for c in str(label)) or f"C{i}"
        lines.append(f"          {key}: {put(path, key)}")
    lines += ["    - type: finish", "      name: Thank you", "      content: Thank you!", "      showResults: true",
              "      writeResults: true", ""]
    cfg = out / "configs" / f"{test}.yaml"
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return cfg
