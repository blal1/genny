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
  the config YAML and the audio, ready to drop into a webMUSHRA install. ``align_stimuli`` levels the stimuli,
  ``mushra_screen`` post-screens the answers (BS.1534-3), ``sdt`` gives d', lambda, beta (Lindau & Weinzierl 2012).
- ``compare``: a render against a recording (PSD RMSE, DDSP loudness L1, ``fad`` on ``logmel_frames``).
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


def emg_fit(freqs, decays) -> tuple[float, float, float, float]:
    """(alpha, beta, sigma, eta) of decays c_i = (alpha + beta w^2) / 2 + n + e, n ~ N(0, sigma^2), e ~ Exp(eta)
    (a support adds damping, never removes it), by maximum likelihood; ``rayleigh_fit`` overestimates the damping."""
    from scipy.optimize import minimize
    from scipy.special import log_ndtr
    c = np.asarray(decays, float)
    w2 = (2 * np.pi * np.asarray(freqs, float)) ** 2
    s = 1.0 / max(float(w2.max()), 1.0)
    a0, b0 = rayleigh_fit(freqs, decays)
    r = c - 0.5 * (a0 + b0 * w2)
    p0 = [0.5 * a0, 0.5 * b0 / s, math.log(max(float(r.std()), 1e-6)), math.log(1.0 / max(float(np.abs(r).mean()), 1e-6))]

    def nll(p):
        m = p[0] + p[1] * s * w2
        sg, et = math.exp(p[2]), math.exp(p[3])
        return -float(np.sum(math.log(et / 2) + et * m + et * et * sg * sg / 2 - et * c + math.log(2.0)
                             + log_ndtr((c - m - et * sg * sg) / sg)))

    q = minimize(nll, p0, method="L-BFGS-B", bounds=[(0, None), (0, None), (-20, 10), (-20, 20)]).x
    return float(2 * q[0]), float(2 * q[1] * s), math.exp(q[2]), math.exp(q[3])


# ------------------------------------------------------------------------------------------- textures
def texture_snr(a: np.ndarray, b: np.ndarray, sr: int) -> dict:
    """SNR (dB) of b's statistics against a's, per class (envelope quantiles, modulation power, envelope
    correlations, C1, C2) and their mean: McDermott & Simoncelli's measure of how well a texture is matched."""
    from .cyber import texture_stats
    return stats_snr(texture_stats(a, sr), texture_stats(b, sr))


def stats_snr(sa: dict, sb: dict) -> dict:
    """``texture_snr`` on two ``texture_stats``. Signed or complex: C2's phase is the time direction (p. 938-939)."""
    out = {}
    for k in ("quantiles", "modulation", "corr", "c1", "c2"):
        x, y = np.asarray(sa[k]), np.asarray(sb[k])
        out[k] = float(10 * np.log10(np.sum(np.abs(x) ** 2) / (np.sum(np.abs(x - y) ** 2) + 1e-30)))
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


def align_stimuli(paths: dict, out_dir: str, lufs: float = -23.0) -> dict:
    """Copies of the stimuli at one integrated loudness (``spec.master_loudness``, true peak -1 dBTP) for a
    MUSHRA test: BS.1534-3 §8 asks for loudness-equalised stimuli. Returns label -> new path. Warns when a page
    would hold more than 12 signals (BS.1534-3 §5.1: at most 12 per trial, references and anchors included)."""
    import warnings
    from .core import read_wav, write_wav
    from .spec import master_loudness
    if len(paths) + 3 > 12:                                  # + hidden reference and the two anchors
        warnings.warn(f"{len(paths) + 3} signals on one MUSHRA page; BS.1534-3 §5.1 allows 12")
    out = Path(out_dir)
    res = {}
    for label, p in paths.items():
        y, sr = read_wav(p)
        res[label] = str(write_wav(out / f"{label}.wav", master_loudness(y, sr, lufs), sr, bits=24))
    return res


# ------------------------------------------------------------------------------------------- listening-test statistics
def sdt(p_hit: float, p_fa: float) -> dict:
    """Equal-variance signal detection estimates (Lindau & Weinzierl, Acta Acust. 98, 2012, eq. 1-3):
    lambda = Z(1 - pFA), d' = Z(pH) - Z(pFA), beta = phi(lambda - d') / phi(lambda); and the 2AFC detection rate
    that d' gives, Pc = Phi(d' / sqrt 2) (their Table I). Rates of exactly 0 or 1 are the caller's to correct."""
    from statistics import NormalDist
    N = NormalDist()
    lam = N.inv_cdf(1.0 - p_fa)
    d = N.inv_cdf(p_hit) - N.inv_cdf(p_fa)
    return {"d": d, "lambda": lam, "beta": N.pdf(lam - d) / N.pdf(lam), "pc": N.cdf(d / math.sqrt(2.0))}


def n_opt(d_min: float = 0.1777, alpha: float = 0.25, beta: float = 0.05) -> int:
    """Decisions a yes/no test needs to tell d' >= ``d_min`` from chance at type I / II errors ``alpha`` / ``beta``:
    (z_alpha + z_beta)^2 2 pi / d_min^2, from var(d') -> 2 pi / N near d' = 0."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    return math.ceil((z(1.0 - alpha) + z(1.0 - beta)) ** 2 * 2.0 * math.pi / float(d_min) ** 2)


def _quartiles(x):
    """BS.1534-3 §4.2 median and quartiles: Q1 / Q3 = medians of the lower / upper half, the middle value in both
    halves when n is odd."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    h = (n + 1) // 2
    return float(np.median(x[:h])), float(np.median(x)), float(np.median(x[n - h:]))


def mushra_screen(scores: dict, reference: str = "reference", anchor: str = "anchor70") -> dict:
    """BS.1534-3 §4.1.2 post-screening and §4.2 outliers. ``scores``: assessor -> item -> condition -> 0..100.
    An assessor is excluded when they score the hidden reference under 90 on more than 15 % of the items, or the
    mid-range anchor (7 kHz low-pass, webMUSHRA ``anchor70``) over 90 on more than 15 % of the items, items where
    more than 25 % of assessors do so not counting. Returns {"excluded", "kept", "stats": condition -> (median,
    IQR) over the kept assessors and all items, "outliers": [(assessor, item, condition, score)] outside
    [Q1 - 1.5 IQR, Q3 + 1.5 IQR] per condition and item}."""
    who = list(scores)
    items = sorted({i for a in who for i in scores[a]})
    lenient = {i for i in items
               if sum(scores[a].get(i, {}).get(anchor, 0) > 90 for a in who) > 0.25 * len(who)}
    excluded = []
    for a in who:
        its = list(scores[a])
        ref_bad = sum(scores[a][i].get(reference, 100) < 90 for i in its)
        anc_bad = sum(scores[a][i].get(anchor, 0) > 90 for i in its if i not in lenient)
        if ref_bad > 0.15 * len(its) or anc_bad > 0.15 * len(its):
            excluded.append(a)
    kept = [a for a in who if a not in excluded]
    conds = sorted({c for a in kept for i in scores[a] for c in scores[a][i]})
    stats, outliers = {}, []
    for c in conds:
        v = [scores[a][i][c] for a in kept for i in scores[a] if c in scores[a][i]]
        q1, q2, q3 = _quartiles(v)
        stats[c] = (q2, q3 - q1)
        for i in items:
            vi = [(a, scores[a][i][c]) for a in kept if c in scores[a].get(i, {})]
            if len(vi) < 2:
                continue
            q1, _, q3 = _quartiles([s for _, s in vi])
            outliers += [(a, i, c, s) for a, s in vi if s > q3 + 1.5 * (q3 - q1) or s < q1 - 1.5 * (q3 - q1)]
    return {"excluded": excluded, "kept": kept, "stats": stats, "outliers": outliers}


# ------------------------------------------------------------------------------------------- render against recording
def fad(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Frechet distance between two sets of embeddings (rows = frames), Kilgour et al. (FAD, arXiv 1812.08466,
    §3.1): |mu_a - mu_b|^2 + tr(S_a + S_b - 2 sqrt(S_a S_b)). Kilgour use VGGish (128-d); ``logmel_frames`` gives a
    dependency-free stand-in, whose numbers are not comparable to published FADs. They ask for >= 300 clips of 5 s
    per set (100 is unstable)."""
    from scipy.linalg import sqrtm
    a, b = np.atleast_2d(np.asarray(emb_a, float)), np.atleast_2d(np.asarray(emb_b, float))
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    sa, sb = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    cs = sqrtm(sa @ sb)
    cs = cs.real if np.iscomplexobj(cs) else cs
    return float(np.sum((ma - mb) ** 2) + np.trace(sa + sb - 2.0 * cs))


def logmel_frames(x: np.ndarray, sr: int, n_mels: int = 64, win: float = 0.025, hop: float = 0.010,
                  fmax: float = 7500.0) -> np.ndarray:
    """Log-mel frames (rows) as embeddings for ``fad``: 64 bands as VGGish's input, mel(f) =
    2595 log10(1 + f/700), triangular filters, Hann 25 ms, hop 10 ms, 125-7500 Hz (the VGGish front end)."""
    x = np.asarray(x, float)
    x = x.mean(axis=1) if x.ndim == 2 else x
    n, h = int(win * sr), int(hop * sr)
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    nfft = 1 << int(math.ceil(math.log2(n)))
    P = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, n)[::h] * np.hanning(n), nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    mel = lambda v: 2595.0 * np.log10(1.0 + np.asarray(v) / 700.0)
    edges = 700.0 * (10 ** (np.linspace(mel(125.0), mel(min(fmax, sr / 2)), n_mels + 2) / 2595.0) - 1.0)
    fb = np.maximum(0.0, np.minimum((f[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                                    (edges[2:, None] - f[None]) / (edges[2:, None] - edges[1:-1, None])))
    return np.log(P @ fb.T + 0.01)


def loudness_track(x: np.ndarray, sr: int, n_fft: int = 512, hop: int | None = None, range_db: float = 80.0) -> np.ndarray:
    """A-weighted log power per frame (DDSP's ``compute_loudness``: n_fft 512, power averaged over
    frequency, 80 dB range under the maximum), in dB. hop n_fft / 4 by default."""
    from .dsp import a_weighting_db
    x = np.asarray(x, float)
    x = x.mean(axis=1) if x.ndim == 2 else x
    hop = hop or n_fft // 4
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    S = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft), axis=1)) ** 2
    w = 10 ** (a_weighting_db(np.maximum(np.fft.rfftfreq(n_fft, 1.0 / sr), 1.0)) / 10)
    db = 10 * np.log10(np.mean(S * w, axis=1) + 1e-20)
    return np.maximum(db, db.max() - range_db)


def compare(render: np.ndarray, recording: np.ndarray, sr: int) -> dict:
    """How far a render is from a recording, both at ``sr``, after matching their mean power (a recording's gain
    is arbitrary): ``psd_rmse`` / ``psd_mae`` in dB between Welch PSDs (Hann, 2 s segments, hop 0.5 s, the
    pass-by validation, over 50 Hz .. 0.45 sr, each PSD floored 80 dB under its maximum; about 4 dB RMSE is a good
    car model), ``loudness_l1`` the mean |difference| of the two ``loudness_track`` (DDSP's loudness L1, on the
    common length), and
    ``fad`` on ``logmel_frames``."""
    from scipy.signal import welch
    a, b = (np.asarray(v, float) for v in (render, recording))
    a, b = (v.mean(axis=1) if v.ndim == 2 else v for v in (a, b))
    a = a / (np.sqrt(np.mean(a * a)) + 1e-30)
    b = b / (np.sqrt(np.mean(b * b)) + 1e-30)
    nper = min(2 * sr, a.size, b.size)
    nov = nper - min(sr // 2, nper)
    f, Pa = welch(a, sr, window="hann", nperseg=nper, noverlap=nov)
    _, Pb = welch(b, sr, window="hann", nperseg=nper, noverlap=nov)
    m = (f >= 50.0) & (f <= 0.45 * sr)
    db = lambda P: np.maximum(10 * np.log10(P[m] + 1e-30), 10 * np.log10(P[m].max() + 1e-30) - 80.0)
    d = db(Pa) - db(Pb)                                       # 80 dB floor as the loudness track: an empty band
                                                              # above a codec's cut-off would otherwise dominate
    la, lb = loudness_track(a, sr), loudness_track(b, sr)
    k = min(la.size, lb.size)
    return {"psd_rmse": float(np.sqrt(np.mean(d * d))), "psd_mae": float(np.mean(np.abs(d))),
            "loudness_l1": float(np.mean(np.abs(la[:k] - lb[:k]))),
            "fad": fad(logmel_frames(a, sr), logmel_frames(b, sr))}


def ir_params(ir: np.ndarray, sr: int, direct_ms: float = 2.5) -> dict:
    """Room parameters of a measured impulse response, the targets a reverb is calibrated to: Schroeder's
    backward-integrated decay, T20 (-5 to -25 dB), T30 (-5 to -35 dB), EDT (0 to -10 dB), each extrapolated to 60 dB, in s, by
    least squares on the decay curve; DRR in dB, energy within ``direct_ms`` of the peak against the rest.
    A slope the decay never reaches gives nan."""
    h = np.asarray(ir, float)
    h = h.mean(axis=1) if h.ndim == 2 else h
    e = h * h
    edc = np.cumsum(e[::-1])[::-1]
    db = 10.0 * np.log10(edc / max(edc[0], 1e-30) + 1e-30)
    t = np.arange(len(h)) / sr

    def slope(hi, lo):
        m = (db <= hi) & (db >= lo)
        if m.sum() < 3 or db.min() > lo:
            return float("nan")
        a = np.polyfit(t[m], db[m], 1)[0]
        return float(-60.0 / a) if a < 0 else float("nan")

    pk = int(np.argmax(np.abs(h)))
    w = max(1, int(direct_ms * 1e-3 * sr))
    direct = float(e[max(0, pk - w):pk + w + 1].sum())
    rest = float(e.sum()) - direct
    return {"t20": slope(-5.0, -25.0), "t30": slope(-5.0, -35.0), "edt": slope(0.0, -10.0),
            "drr": 10.0 * math.log10(direct / max(rest, 1e-30))}


ALLAMANCHE_FRAME = (0.4, 0.2)          # s: frame length and hop of the descriptor tracks
CLASS_METRIC = dict(rain="allamanche", wind="allamanche", stream="texture", fire="psd")


def allamanche_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame (loudness dB, spectral flatness, crest factor dB) of ``y``: frames x 3."""
    from .analysis import crest_factor_db, loudness, spectral_flatness
    y = np.asarray(y, float)
    y = y.mean(axis=1) if y.ndim == 2 else y
    L, H = (int(v * sr) for v in ALLAMANCHE_FRAME)
    rows = [(loudness(y[i:i + L], sr, window=ALLAMANCHE_FRAME[0]), spectral_flatness(y[i:i + L]), crest_factor_db(y[i:i + L]))
            for i in range(0, max(len(y) - L, 0) + 1, H) if np.any(y[i:i + L])]
    return np.array(rows, float).reshape(-1, 3)


def allamanche_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Distance between two textures on the loudness, flatness and crest-factor descriptors: the
    Euclidean distance of their mean descriptor vectors, each axis divided by the pooled standard deviation."""
    fa, fb = allamanche_features(a, sr), allamanche_features(b, sr)
    if not len(fa) or not len(fb):
        return float("nan")
    sd = np.sqrt(0.5 * (fa.var(axis=0) + fb.var(axis=0))) + 1e-9
    return float(np.sqrt(np.sum(((fa.mean(axis=0) - fb.mean(axis=0)) / sd) ** 2)))


def class_distance(render: np.ndarray, recording: np.ndarray, sr: int, kind: str) -> float:
    """The measure suited to a sound class (``CLASS_METRIC``): ``allamanche`` for rain and wind,
    McDermott-Simoncelli statistics (``texture_snr``, negated mean SNR) for a stream, PSD RMSE (``compare``) for fire."""
    m = CLASS_METRIC.get(str(kind))
    if m is None:
        raise ValueError(f"unknown class {kind!r}: {'|'.join(CLASS_METRIC)}")
    if m == "allamanche":
        return allamanche_distance(render, recording, sr)
    if m == "texture":
        return -texture_snr(recording, render, sr)["mean"]
    return compare(render, recording, sr)["psd_rmse"]


# ------------------------------------------------------------------------------------------- distances
MSS_FFTS = (2048, 1024, 512, 256, 128, 64)
MSS_OVERLAP = 0.75


def _stft_mag(x, n, overlap):
    hop = max(1, int(round(n * (1.0 - overlap))))
    x = np.asarray(x, float)
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    return np.abs(np.fft.rfft(fr, axis=1))


def multiscale_spectral_distance(a: np.ndarray, b: np.ndarray, ffts=MSS_FFTS, overlap: float = MSS_OVERLAP,
                                 w_mag: float = 1.0, w_log: float = 1.0) -> float:
    """Multi-scale spectral distance: sum over FFT sizes of w_mag |S - S~|_1 + w_log |log S - log S~|_1
    (means over bins and frames), Hann frames with ``overlap``. The shorter signal is zero-padded."""
    n = max(len(a), len(b))
    a, b = np.pad(np.asarray(a, float), (0, n - len(a))), np.pad(np.asarray(b, float), (0, n - len(b)))
    d = 0.0
    for k in ffts:
        A, B = _stft_mag(a, k, overlap), _stft_mag(b, k, overlap)
        d += w_mag * float(np.mean(np.abs(A - B)))
        if w_log:
            d += w_log * float(np.mean(np.abs(np.log(A + 1e-7) - np.log(B + 1e-7))))
    return d


def spectral_flux_error(x: np.ndarray, ref: np.ndarray, sr: int, win: float = 0.0232, hop: float = 0.0058) -> float:
    """Mean absolute difference of the half-wave rectified spectral flux curves (onsets, transients) of a render
    and a reference, each normalised by its own maximum; 0 = the same transient pattern."""
    def flux(y):
        n, h = int(round(win * sr)), int(round(hop * sr))
        A = _stft_mag(y, n, 1.0 - h / n)
        f = np.maximum(np.diff(A, axis=0), 0.0).sum(axis=1)
        return f / (f.max() + 1e-30)
    fa, fb = flux(x), flux(ref)
    m = min(fa.size, fb.size)
    return float(np.mean(np.abs(fa[:m] - fb[:m]))) if m else float("nan")


F0_CONFIDENCE = 0.85


def f0_l1_midi(f0_a, f0_b, mask=None) -> float:
    """Mean |MIDI(f0_a) - MIDI(f0_b)| in semitones over the frames where ``mask`` holds (a boolean array, or the
    confidence of the tracker compared with ``F0_CONFIDENCE``) and both tracks are voiced (> 0)."""
    a, b = np.asarray(f0_a, float), np.asarray(f0_b, float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ok = (a > 0) & (b > 0)
    if mask is not None:
        m = np.asarray(mask)[:n]
        ok &= m if m.dtype == bool else m >= F0_CONFIDENCE
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(12.0 * np.log2(a[ok] / b[ok]))))


# ------------------------------------------------------------------------------------------- loop filter
def loop_from_modes(freqs, decays) -> dict:
    """Waveguide loop seen by a set of near-harmonic modes: loop delay D = 2 pi / (w2 - w1) from the two
    lowest, per-mode loop gain |F(w_n)| = exp(-alpha_n D) (~ 1 - alpha_n D), loop phase w_n D - 2 pi n."""
    f = np.asarray(freqs, float)
    o = np.argsort(f)
    f, a = f[o], np.asarray(decays, float)[o]
    w = 2 * np.pi * f
    D = 2 * np.pi / (w[1] - w[0])
    nn = np.round(f * D).astype(int)
    return {"D": float(D), "freqs": f, "gain": np.exp(-a * D), "phase": w * D - 2 * np.pi * nn}


def loop_filter_from_recording(x: np.ndarray, sr: int, n_modes: int = 12, dur: float = 0.3) -> dict:
    """Loop filter of a plucked or struck string from a recording: ESPRIT modes, ``loop_from_modes``, and the
    Rayleigh fit of the decays: g0 = exp(-alpha/2 D) the loop gain at DC, ``beta`` its w^2 coefficient
    (decay = (alpha + beta w^2)/2), with the loop delay D."""
    m = esprit(x, sr, n_modes, dur=dur)
    lp = loop_from_modes(m["freqs"][:n_modes], m["decays"][:n_modes])
    alpha, beta = rayleigh_fit(lp["freqs"], -np.log(lp["gain"]) / lp["D"])
    return dict(lp, g0=float(math.exp(-0.5 * alpha * lp["D"])), alpha=alpha, beta=beta)


FAD_NOISE_SD = (1e-4, 3.1e-4, 1e-3, 3.1e-3, 0.01, 0.031, 0.1, 0.31)
FAD_LOWPASS = (4000.0, 2000.0, 1000.0, 500.0, 300.0)
FAD_HIGHPASS = (200.0, 500.0, 1000.0, 2000.0, 4000.0)
FAD_BITS = (9, 8, 7, 6, 5, 4, 3, 2)


def distortion_grid(x: np.ndarray, sr: int, seed: int = 0):
    """The Frechet-audio-distance validation grid: yields (label, distorted copy) for white noise of each SD,
    each low-pass and high-pass cutoff (4th-order Butterworth) and each quantisation depth."""
    from scipy.signal import butter, sosfilt
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    for s in FAD_NOISE_SD:
        yield f"noise_{s:g}", x + s * rng.standard_normal(x.shape)
    for f in FAD_LOWPASS:
        yield f"lowpass_{f:g}", sosfilt(butter(4, f, fs=sr, output="sos"), x, axis=0)
    for f in FAD_HIGHPASS:
        yield f"highpass_{f:g}", sosfilt(butter(4, f, "highpass", fs=sr, output="sos"), x, axis=0)
    for b in FAD_BITS:
        q = 2.0 ** (b - 1)
        yield f"bits_{b}", np.round(np.clip(x, -1, 1) * q) / q
