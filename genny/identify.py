"""Analysis/resynthesis helpers for fitting procedural models to recordings."""
from __future__ import annotations
import numpy as np
from .core import DEFAULT_SR, resample
from .physical import analysis as _a


def lpc_frames(x: np.ndarray, *, sr: int = DEFAULT_SR, order: int | None = None,
               frame_ms: float = 20.0, hop_ms: float = 10.0):
    """Return time-varying LPC analysis using stable reflection coefficients.

    The research engine works at 48 kHz; arbitrary-rate input is resampled first.
    """
    x = np.asarray(x, float)
    if sr != 48000:
        x = resample(x, sr, 48000)
    return _a.analyse(x, order=order, frame_ms=frame_ms, hop_ms=hop_ms)


def modal_fit(x: np.ndarray, *, sr: int = DEFAULT_SR, **kwargs):
    x = np.asarray(x, float)
    if sr != 48000:
        x = resample(x, sr, 48000)
    return _a.modal_fit(x, **kwargs)


# ---------------------------------------------------------------------------------------------
# Machines: from a recording to the parameters of the genny.cyber model that makes it
# ---------------------------------------------------------------------------------------------
# Peak picking with parabolic interpolation is Essentia's PeakDetection (standard/peakdetection.cpp)
# and librosa's piptrack; the envelope spectrum of a high band is the bearing-diagnostics method of
# Randall & Antoni (MSSP 25, 2011, section 5: the squared envelope). The decision rules and their
# thresholds are # UNSOURCED: set on genny's own renders and, for bearings, on the FSTF, AHU and SUBF
# recordings (results in docs/cyber.md, "Limits").
# ponytail: one steady operating point (no run-up tracking), four machine kinds, no fan or motor
# orders. Track the strongest line over time and work per segment when a recording changes speed.

_RATE = 24000


def spectral_lines(x: np.ndarray, sr: int, fmin: float = 15.0, fmax: float | None = None, prominence: float = 12.0):
    """Tonal lines of a steady sound: (Hz, dB, dB above the local noise floor), strongest first.
    The floor is a 40 Hz wide running median of the dB spectrum."""
    from scipy.fft import next_fast_len, rfft, rfftfreq
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks, get_window
    n = len(x)
    N = next_fast_len(4 * n)
    db = 20 * np.log10(np.abs(rfft(x * get_window("blackmanharris", n), N)) + 1e-12)   # -92 dB sidelobes: no false lines
    fr = rfftfreq(N, 1.0 / sr)
    over = db - median_filter(db, size=2 * int(20.0 * N / sr) + 1, mode="nearest")
    pk, _ = find_peaks(over, height=prominence, distance=12)
    pk = pk[(fr[pk] >= fmin) & (fr[pk] <= (fmax or 0.45 * sr)) & (pk > 0) & (pk < len(db) - 1)]
    a, b, c = db[pk - 1], db[pk], db[pk + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c - 1e-12)
    level = b - 0.25 * (a - c) * d
    order = np.argsort(-level)
    return ((pk + d) * sr / N)[order], level[order], over[pk][order]


def _has(lines: np.ndarray, f: float, tol: float = 0.4) -> bool:
    return bool(len(lines)) and float(np.min(np.abs(lines - f))) < tol


def identify_machine(x: np.ndarray, sr: int = DEFAULT_SR, rpm: float | None = None) -> dict:
    """Look at a steady recording of a machine and name the model and parameters that explain it.

    Returns ``{"kind": "transformer" | "bearing" | "gearbox" | "engine" | "unknown", "sfx": name, "params": {...},
    "evidence": {...}}``; ``params`` can be passed to ``genny.sfx.render_sfx(sfx, sr, **params)``.
    ``rpm`` is the shaft speed if it is known (it is then not estimated)."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    from scipy.stats import kurtosis
    x = np.asarray(x, float)
    x = x.mean(axis=1) if x.ndim == 2 else x
    if sr != _RATE:
        x = resample(x, sr, _RATE)
    x = x - x.mean()
    f, db, _ = spectral_lines(x, _RATE)
    out = {"kind": "unknown", "sfx": None, "params": {}, "evidence": {"lines_hz": [round(float(v), 2) for v in f[:12]]}}
    if len(f) == 0:
        return out
    p = 10 ** (db / 10)

    # 1. Transformer: nearly all the tonal power sits on the multiples of 50 or 60 Hz, most of it on
    # the even ones (magnetostriction doubles the frequency).
    # (A machine turning at exactly 3000 or 3600 rpm lands here too: pass ``rpm`` to skip this test.)
    share = {}
    for mains in (50.0, 60.0):
        k = np.round(f / mains)
        on = (k >= 1) & (np.abs(f - k * mains) < 0.5 + 0.002 * f)
        share[mains] = (float(p[on & (k % 2 == 0)].sum() / p.sum()), float(p[on & (k % 2 == 1)].sum() / p.sum()))
    mains = max(share, key=lambda m: sum(share[m]))
    even, odd = share[mains]
    if even + odd > 0.9 and even > odd and len(f) >= 2 and rpm is None:
        amp = lambda hz: 10 ** (float(np.max(db[np.abs(f - hz) < 1.0], initial=-240.0)) / 20)
        out.update(kind="transformer", sfx="transformer",
                   params={"mains": mains, "dc_bias": round(float(np.clip(0.6 * amp(mains) / (amp(2 * mains) + 1e-12), 0.0, 0.5)), 2)})
        out["evidence"].update(comb_share=round(even + odd, 3), odd_share=round(odd, 3))
        return out

    # 2. Bearing: the high band is modulated at a rate that is not a multiple of the shaft rate. The band
    # is impulsive, or (a soft outer-race defect is not) the modulation line has two harmonics.
    fr = float(rpm) / 60.0 if rpm else None
    if fr is None:
        low = f[f < 120.0]                                     # the once-per-turn line of the shaft
        fr = float(low[0]) if len(low) else None
    band = sosfiltfilt(butter(4, 1500.0, "highpass", fs=_RATE, output="sos"), x)
    env = np.abs(hilbert(band)) ** 2
    ef, el, _ = spectral_lines(env - env.mean(), _RATE, fmin=2.0, fmax=600.0, prominence=15.0)
    kurt = float(kurtosis(band))
    out["evidence"].update(shaft_hz=None if fr is None else round(fr, 2), band_kurtosis=round(kurt, 2),
                           envelope_lines_hz=[round(float(v), 2) for v in ef[:6]])
    if fr:
        ratio = ef / fr
        cand = (ratio > 2.5) & (ratio < 8.0)
        for c in (1, 2, 3, 4):                                  # not an order or a half / third / quarter order
            cand &= np.abs(ef - np.round(ratio * c) * fr / c) > np.maximum(0.2, 0.003 * ef)
        for mains in (50.0, 60.0):                              # an electric motor hums at the supply rate and its multiples
            cand &= np.abs(ef - np.round(ef / mains) * mains) > 0.01 * ef   # (every SUBF file, healthy or not)
        cand &= np.array([_has(ef, 2 * v, 0.6) and (kurt > 2.0 or _has(ef, 3 * v, 0.9)) for v in ef], dtype=bool)
        if cand.any():
            fe = float(ef[cand][0])
            # "Inner race faults would be modulated at shaft speed, and rolling element faults at cage
            # speed. For unidirectional load, an outer race fault would not be modulated" (Randall & Antoni
            # 2011, section 6): its harmonics then stand above whatever sidebands unbalance gives it.
            lv = lambda hz, tol=0.4: float(np.max(el[np.abs(ef - hz) < tol], initial=-np.inf))
            pair = lambda d: min(lv(fe - d), lv(fe + d))         # a modulation shows on both sides
            by_shaft, by_cage = pair(fr), max(pair(c * fr) for c in np.arange(0.34, 0.465, 0.01))
            if max(by_shaft, by_cage) < lv(2 * fe, 0.6) - 12.0:    # UNSOURCED: 12 dB, set on the AHU and FSTF recordings
                state, balls = "outer", fe / fr / 0.4            # BPFO = n / 2 (1 - d/D), with d/D = 0.2
            elif by_shaft >= by_cage:
                state, balls = "inner", fe / fr / 0.6            # BPFI = n / 2 (1 + d/D)
            else:
                state, balls = "ball", 9.0
            out.update(kind="bearing", sfx="bearing",
                       params={"rpm": round(60.0 * fr, 1), "state": state, "balls": int(np.clip(round(balls), 3, 60))})
            out["evidence"].update(fault_hz=round(fe, 2), fault_order=round(fe / fr, 3))
            return out
        # A ball defect: slip smears its own rate out of the envelope spectrum and leaves the cage rate
        # that modulates it (0.34 to 0.46 of the shaft rate for any usual geometry) with its multiples,
        # in an impulsive band. This is what the FSTF ball recordings show. The cage rate is not a line
        # of the sound itself ("the low harmonics of the repetition frequency have very low magnitude",
        # Randall & Antoni 2011, section 1): a knock once per turn is, and is not taken for a cage.
        for b in sorted({float(v) / m for v in ef for m in (1, 2)}):
            if (kurt > 2.0 and 0.34 < b / fr < 0.46 and sum(_has(ef, m * b, 0.6) for m in (1, 2, 3, 4, 6)) >= 3
                    and sum(_has(f, m * b, 0.6) for m in (1, 2, 3, 4, 6)) < 2):
                out.update(kind="bearing", sfx="bearing", params={"rpm": round(60.0 * fr, 1), "state": "ball"})
                out["evidence"].update(cage_hz=round(b, 2), cage_order=round(b / fr, 3))
                return out

    # 3. Gear mesh: a carrier with symmetric sidebands one shaft rate apart. The mesh order (the tooth
    # count) is the shaft harmonic that stands out of its neighbours, together with its double.
    lev = lambda hz: float(np.max(db[np.abs(f - hz) < 0.25], initial=float(db.min()) - 10.0))
    multiple = lambda a, b: round(a / b) >= 1 and abs(a / b - round(a / b)) < 0.02
    for fc in f[f > 150.0][:3]:
        found = []
        for g in f[(f > fc) & (f < 1.3 * fc)]:
            dlt = float(g - fc)
            z = round(fc / dlt)
            if dlt > 2.0 and 6 <= z <= 400 and _has(f, fc - dlt) and abs(fc / z - dlt) < 0.06:
                # ranked by its stronger side: amplitude and speed modulation together reinforce one
                # sideband and cancel the other (Randall 1982)
                found.append((max(lev(fc + dlt), lev(fc - dlt)), fc / z))
        base = [fr] if rpm else []
        for _, d in sorted(found, reverse=True):                 # strongest sidebands first
            mixed = any(abs(d - abs(a - b)) < 0.1 or abs(d - (a + b)) < 0.1 for a in base for b in base if a != b)
            if rpm or len(base) == 2 or mixed or any(multiple(d, b) for b in base):
                continue
            base = [b for b in base if not multiple(b, d)] + [d]
        if not base:
            continue
        fast = max(base)
        if fast < 4.0:                                           # under 240 rpm: flutter, not a shaft
            continue
        top = int(min(400, 0.45 * _RATE / fast))
        L = np.array([lev(m * fast) for m in range(top + 9)])
        excess = np.array([L[m] - np.median(np.r_[L[max(m - 4, 1):m], L[m + 1:m + 5]]) for m in range(top + 1)])
        score = excess + 0.5 * np.array([excess[2 * m] if 2 * m <= top else 0.0 for m in range(top + 1)])
        score[:8] = -np.inf                                      # no gear has fewer teeth
        score[excess < 10.0] = -np.inf
        if not np.isfinite(score).any():
            continue
        z = int(np.flatnonzero(score >= score.max() - 6.0)[0])
        if z * fast < 150.0 or L[1:z - 1].max() > L[z] + 6.0:    # a low order dominates: an engine, not a mesh
            continue
        if not (_has(f, (z - 1) * fast) and _has(f, (z + 1) * fast)):   # no sidebands: a tone, not a mesh
            continue
        params = {"rpm": round(float(60.0 * fast), 1), "teeth": z}
        if len(base) > 1:
            params["teeth2"] = int(round(z * fast / min(base)))
        out.update(kind="gearbox", sfx="gearbox", params=params)
        out["evidence"].update(mesh_hz=round(float(z * fast), 2), sideband_spacings_hz=[round(float(d), 2) for d in base])
        return out

    # 4. Four-stroke engine: a harmonic series on the firing rate, and between its lines the weaker
    # multiples of the cycle rate rpm / 120 (cylinders are never identical). firing / cycle = cylinders.
    f0 = float(f[0])
    if f0 < 600.0 and _has(f, 2 * f0, 1.0) and _has(f, 3 * f0, 1.0):
        between = (f < 6 * f0) & (np.abs(f / f0 - np.round(f / f0)) * f0 > 3.0)
        cyl = next((c for c in (1, 2, 3, 4, 5, 6, 8, 10, 12) if between.any() and
                    p[between & (np.abs(f * c / f0 - np.round(f * c / f0)) * f0 / c < 0.3)].sum() > 0.8 * p[between].sum()), None)
        out.update(kind="engine", sfx="pipe_engine", params={"rpm": round(120.0 * f0 / (cyl or 4), 1), "cylinders": cyl or 4})
        out["evidence"].update(firing_hz=round(f0, 2), cylinders_assumed=cyl is None)
    return out
