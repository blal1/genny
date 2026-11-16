"""Family V: binaural rendering, free reeds, measured instrument bodies, tape bias.

Registers
    effects      `binaural_move` (moving source through an HRTF), `binaural_room` (image-source room with an HRTF per arrival),
                 `body_measured` (modal body of a violin / guitar), `tape_bias` (Jiles-Atherton tape with HF bias, wow, flutter)
    instruments  `harmonica_reed`, `accordion_reed`, `melodica_reed`, `sho_reed`, `khaen_reed` (self-oscillating free reed),
                 `guitar_modal`, `violin_modal` (string excitation through the measured body)
HRTF data: the parametric sphere + pinna model, or a measured set read by `load_hrir` (.sofa through h5py, or a stereo .wav).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numba
import numpy as np
from scipy import signal as sg

from . import dsp as D
from . import filters as F
from . import presets
from . import reverbs as R
from .core import DEFAULT_SR, samples, to_mono
from .fx import effect
from .instruments import instrument

C_SOUND = 343.0
HEAD_A = 0.0875                                      # head radius (m): 8.75 cm, the Woodworth / Kuhn reference sphere


# ============================================================================ HRTF: parametric sphere + pinna
def itd_woodworth(az, el=0.0, a=HEAD_A, c=C_SOUND):
    """Interaural time difference (s, > 0 when the source is on the right: the left ear is later) of a rigid sphere,
    Woodworth: a/c (theta + sin theta), theta the lateral angle asin(sin az cos el); 0.656 ms at 90 deg for a = 8.75 cm."""
    th = math.asin(max(-1.0, min(1.0, math.sin(math.radians(az)) * math.cos(math.radians(el)))))
    return a / c * (th + math.sin(th))


def itd_kuhn(az, el=0.0, a=HEAD_A, c=C_SOUND):
    """Low-frequency (below ~500 Hz) interaural delay of the sphere, Kuhn: 3 a / c sin theta (1.5 x Woodworth at 90 deg)."""
    return 3.0 * a / c * math.sin(math.radians(az)) * math.cos(math.radians(el))


def head_shadow(f, theta_deg, a=HEAD_A, amin=0.1, thmin=150.0):
    """Brown-Duda head-shadow filter, theta the angle between the source and the ear axis: (1 + j alpha w / 2w0) / (1 + j w / 2w0),
    w0 = c / a, alpha(theta) = 1 + amin/2 + (1 - amin/2) cos(180 theta / thmin): +6 dB at the near ear, about -11 dB at 180 deg above w0."""
    w = 2.0 * np.pi * np.asarray(f, float) / (2.0 * C_SOUND / a)
    al = 1.0 + amin / 2.0 + (1.0 - amin / 2.0) * math.cos(math.radians(theta_deg) * 180.0 / thmin)
    return (1.0 + 1j * al * w) / (1.0 + 1j * w)


def _spec_param(az, el, sr, nfft, a=HEAD_A, itd_model="woodworth", pre=24.0):
    """rfft (nfft/2+1, 2) of the parametric HRIR (left, right): head shadow per ear, pinna reflection whose notch rises with
    elevation (f_n = 8 kHz + 40 Hz/deg, 5-12 kHz, gain 0.6 cos(theta/2)), rear high-frequency loss (up to 4 dB), whole-head delay
    `pre` samples, the far ear delayed by the ITD. Azimuth 0 ahead, +90 right; elevation 0 level, +90 up."""
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    w = 2.0 * np.pi * f
    cl = math.sin(math.radians(az)) * math.cos(math.radians(el))
    itd = (itd_kuhn if itd_model == "kuhn" else itd_woodworth)(az, el, a)
    back = max(0.0, -math.cos(math.radians(az)) * math.cos(math.radians(el)))
    rear = 10.0 ** (-4.0 * back / 20.0 * f * f / (f * f + 3000.0 ** 2))
    fn = float(np.clip(8000.0 + 40.0 * el, 5000.0, 12000.0))
    out = np.empty((f.size, 2), complex)
    for ch, sgn in ((0, -1.0), (1, 1.0)):                       # left ear axis -x, right +x
        th = math.degrees(math.acos(max(-1.0, min(1.0, sgn * cl))))
        pin = 1.0 + 0.6 * max(0.3, math.cos(math.radians(th) / 2.0)) * np.exp(-1j * w / (2.0 * fn))
        late = max(0.0, -sgn * itd)                             # the ear farther from the source
        out[:, ch] = head_shadow(f, th, a) * pin * rear * np.exp(-1j * w * (pre / sr + late))
    return out


def hrir_param(az, el=0.0, sr=DEFAULT_SR, a=HEAD_A, n=256, itd_model="woodworth"):
    """Parametric HRIR (n, 2), left then right."""
    return np.fft.irfft(_spec_param(az, el, sr, 2 * n, a, itd_model), 2 * n, axis=0)[:n]


# ============================================================================ measured HRIR: SOFA / wav
def load_hrir(path):
    """Measured HRIR set {'hrir' (M, N, 2) left/right, 'az' (M,) deg (0 ahead, +90 right), 'el' (M,) deg, 'sr'}.
    .sofa (SimpleFreeFieldHRIR, spherical or cartesian source positions; needs h5py) or a stereo .wav (one direction, ahead)."""
    p = str(path)
    if p.lower().endswith(".sofa"):
        try:
            import h5py
        except ImportError as e:
            raise ImportError("reading a .sofa file needs h5py (pip install h5py)") from e
        with h5py.File(p, "r") as f:
            ir = np.asarray(f["Data.IR"], float)
            sr = int(np.asarray(f["Data.SamplingRate"]).ravel()[0])
            pos = np.asarray(f["SourcePosition"], float)
            typ = f["SourcePosition"].attrs.get("Type", b"spherical")
            typ = typ.decode() if isinstance(typ, bytes) else str(typ)
        if typ.lower().startswith("cart"):
            az = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
            el = np.degrees(np.arcsin(np.clip(pos[:, 2] / np.maximum(np.linalg.norm(pos, axis=1), 1e-9), -1, 1)))
        else:
            az, el = pos[:, 0], pos[:, 1]
        return dict(hrir=np.transpose(ir, (0, 2, 1)), az=(-az + 180.0) % 360.0 - 180.0, el=el, sr=sr)   # SOFA azimuth is counter-clockwise
    import soundfile as sf
    y, sr = sf.read(p, always_2d=True)
    if y.shape[1] != 2:
        raise ValueError("an HRIR .wav must be stereo (left, right)")
    return dict(hrir=y[None], az=np.zeros(1), el=np.zeros(1), sr=int(sr))


def _unit(az, el):
    a, e = np.radians(np.asarray(az, float)), np.radians(np.asarray(el, float))
    return np.stack([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)], -1)


def _db_resampled(db, sr):
    if int(db["sr"]) == int(sr):
        return db["hrir"]
    g = math.gcd(int(sr), int(db["sr"]))
    return sg.resample_poly(db["hrir"], int(sr) // g, int(db["sr"]) // g, axis=1)


def _spec_at(az, el, sr, nfft, hrtf=None, a=HEAD_A):
    """rfft (nfft/2+1, 2) of the HRIR toward (az, el): nearest measured direction, else the parametric model."""
    if hrtf is None:
        return _spec_param(az, el, sr, nfft, a)
    if "_u" not in hrtf:
        hrtf["_u"] = _unit(hrtf["az"], hrtf["el"])
    i = int(np.argmax(hrtf["_u"] @ _unit(az, el)))
    h = _db_resampled(hrtf, sr)[i]
    return np.fft.rfft(h, nfft, axis=0)


def hrir_at(az, el=0.0, sr=DEFAULT_SR, hrtf=None, a=HEAD_A, n=256):
    """HRIR (n, 2) toward (az, el) from a measured set (`load_hrir`) or the parametric model."""
    if hrtf is None:
        return hrir_param(az, el, sr, a, n)
    h = np.fft.irfft(_spec_at(az, el, sr, 2048, hrtf, a), 2048, axis=0)
    return h[:n] if h.shape[0] >= n else np.pad(h, ((0, n - h.shape[0]), (0, 0)))


def itd_measure(h, sr, fmax=1500.0):
    """Interaural delay (s, > 0 when the right ear leads) of an HRIR (N, 2): cross-correlation peak of the low-passed channels,
    parabolic interpolation."""
    b, a = sg.butter(2, fmax / (sr / 2.0))
    l, r = sg.lfilter(b, a, h[:, 0]), sg.lfilter(b, a, h[:, 1])
    cc = sg.correlate(l, r, mode="full")
    k = int(np.argmax(cc))
    y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    return (k + d - (len(r) - 1)) / sr


def ild_measure(h, f0, f1, sr):
    """Level difference right - left (dB) of an HRIR (N, 2), mean over the band f0..f1."""
    nf = max(4096, h.shape[0])
    s = np.abs(np.fft.rfft(h, nf, axis=0))
    f = np.fft.rfftfreq(nf, 1.0 / sr)
    m = (f >= f0) & (f <= f1)
    return float(20.0 * np.log10(np.mean(s[m, 1] ** 2) ** 0.5 / np.mean(s[m, 0] ** 2) ** 0.5))


# ============================================================================ moving source
def binaural_path(x, sr, az, el=0.0, dist=1.0, hrtf=None, head=HEAD_A, block=256, doppler=False):
    """Mono source -> (n + N, 2) binaural, the source moving along az / el / dist (scalar, or arrays spread over the signal's
    duration). Blocks of 2 `block` samples with periodic Hann windows (they sum to 1 at a hop of `block`) are each convolved with the
    HRIR of the direction at their centre; gain 1 / dist (1 m = unity); `doppler` reads the source through the travel time dist / c."""
    x = to_mono(np.asarray(x, float))
    n = x.size
    t = np.arange(n)
    spread = lambda v: np.interp(t, np.linspace(0, n - 1, np.size(v)), np.ravel(v)) if np.ndim(v) else np.full(n, float(v))
    A, E, Dd = spread(az), spread(el), np.maximum(spread(dist), 0.1)
    if doppler:
        delay = Dd / C_SOUND * sr
        x = np.interp(t - (delay - delay[0]), t, x, left=0.0)
    x = x / Dd
    B = int(block)
    w = sg.windows.hann(2 * B, sym=False)
    nb = n // B + 2
    xp = np.concatenate([np.zeros(B), x, np.zeros(2 * B)])
    h0 = hrir_at(0.0, 0.0, sr, hrtf, head)
    N = h0.shape[0]
    out = np.zeros((nb * B + B + N, 2))
    for k in range(nb):
        c = min(k * B, n - 1)
        seg = xp[k * B:k * B + 2 * B] * w
        if not seg.any():
            continue
        h = hrir_at(A[c], E[c], sr, hrtf, head)
        for ch in range(2):
            y = sg.fftconvolve(seg, h[:, ch])
            out[k * B:k * B + y.size, ch] += y
    return out[B:B + n + N]


@effect("binaural_move", "Moving source heard through an HRTF: Woodworth ITD, head shadow, pinna notch (or a measured .sofa set); distance gain, optional Doppler. Stereo.",
        az0=(-60.0, "start azimuth, degrees (0 ahead, +90 right, 180 behind)"), az1=(60.0, "end azimuth"),
        el=(0.0, "elevation, degrees"), dist0=(2.0, "start distance, m"), dist1=(2.0, "end distance, m"),
        head=(HEAD_A, "head radius, m (parametric model)"), hrtf=("", "path of a .sofa (h5py) or stereo .wav HRIR; empty = parametric"),
        doppler=(0, "1 = read the source through its travel time (pitch shift when it moves)"))
def binaural_move(x, sr=DEFAULT_SR, az0=-60.0, az1=60.0, el=0.0, dist0=2.0, dist1=2.0, head=HEAD_A, hrtf="", doppler=0):
    db = load_hrir(hrtf) if hrtf else None
    return binaural_path(x, sr, np.array([az0, az1]), el, np.array([dist0, dist1]), db, float(head), doppler=bool(doppler))


# ============================================================================ binaural rooms
_BANDS_LOGF = np.log(np.asarray(R.BANDS, float))


def _band_weights(f):
    """(nf, 6) linear weights that interpolate six band values in log-frequency (flat outside)."""
    lf = np.log(np.maximum(f, 1.0))
    W = np.zeros((f.size, len(R.BANDS)))
    for i in range(len(R.BANDS)):
        e = np.zeros(len(R.BANDS))
        e[i] = 1.0
        W[:, i] = np.interp(lf, _BANDS_LOGF, e)
    return W


def binaural_room_ir(dims, src, lis, surfaces="plaster", sr=DEFAULT_SR, order=4, tmax=0.12, yaw=0.0, hrtf=None, head=HEAD_A,
                     tail=True, rt=None, seed=0):
    """Binaural room impulse response (N, 2): shoebox image sources (Allen & Berkley) to `order` reflections up to `tmax` s after the
    direct sound, each arrival filtered by the HRTF of its direction in the listener's frame (listener faces +y, `yaw` degrees to the
    right; x right, z up) and by the per-band wall reflection sqrt(1 - alpha); fractional delays in the spectrum; the direct sound has
    gain 1. After `tmax`: an uncorrelated exponential noise tail per ear at the Sabine decay time (`rt` overrides), level matched to the
    last 20 ms of the image-source part. `surfaces`: one material or six (x0, x1, y0, y1, floor, ceiling) of `reverbs.ABSORPTION`."""
    L = np.asarray(dims, float)
    s, r = np.asarray(src, float), np.asarray(lis, float)
    if np.any(s < 0) or np.any(s > L) or np.any(r < 0) or np.any(r > L):
        raise ValueError("source and listener must be inside the room")
    surf = (surfaces,) * 6 if isinstance(surfaces, str) else tuple(surfaces)
    alpha = np.array([R.ABSORPTION[m] for m in surf], float)             # (6 walls, 6 bands)
    lb = np.log(np.sqrt(1.0 - np.clip(alpha, 0.0, 0.9999)))
    K = int(order)
    g = np.arange(-K, K + 1)
    nn = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    arr = []
    for p in np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]):
        lo, hi = np.abs(nn - p), np.abs(nn)
        keep = (lo + hi).sum(1) <= K
        pos = (1 - 2 * p) * s + 2 * nn[keep] * L
        v = pos - r
        d = np.linalg.norm(v, axis=1)
        lbg = (lo[keep][:, :, None] * lb[0::2][None] + hi[keep][:, :, None] * lb[1::2][None]).sum(1)   # (taps, bands)
        arr.append((v, d, lbg))
    v = np.concatenate([a[0] for a in arr])
    d = np.concatenate([a[1] for a in arr])
    gb = np.exp(np.concatenate([a[2] for a in arr]))
    d0 = max(float(np.linalg.norm(s - r)), 0.05)
    d = np.maximum(d, 0.05)
    t = (d - d0) / C_SOUND
    ok = (t <= tmax) & ((t > 1e-9) | (np.abs(d - d0) < 1e-9))
    v, d, gb, t = v[ok], d[ok], gb[ok], t[ok]
    o = np.argsort(t)
    v, d, gb, t = v[o], d[o], gb[o], np.maximum(t[o], 0.0)
    nfft = 1024
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    W = _band_weights(f)
    n_out = int(tmax * sr) + nfft + 8
    rir = np.zeros((n_out, 2))
    az = np.degrees(np.arctan2(v[:, 0], v[:, 1])) - float(yaw)
    el = np.degrees(np.arcsin(np.clip(v[:, 2] / d, -1, 1)))
    for i in range(t.size):
        ti = t[i] * sr
        k = int(ti)
        sp = _spec_at((az[i] + 180.0) % 360.0 - 180.0, el[i], sr, nfft, hrtf, head)
        sp = sp * ((gb[i] @ W.T) * (d0 / d[i]))[:, None] * np.exp(-1j * 2 * np.pi * f * (ti - k) / sr)[:, None]
        rir[k:k + nfft] += np.fft.irfft(sp, nfft, axis=0)[:max(0, n_out - k)]
    if tail:
        V = float(np.prod(L))
        S = 2.0 * (L[0] * L[1] + L[0] * L[2] + L[1] * L[2])
        areas = np.array([L[1] * L[2]] * 2 + [L[0] * L[2]] * 2 + [L[0] * L[1]] * 2)
        am = float((areas * alpha[:, 2:4].mean(1)).sum() / S)
        T = float(rt) if rt else min(0.161 * V / max(S * am, 1e-6), 6.0)
        m = int(tmax * sr)
        ref = float(np.sqrt(np.mean(rir[max(0, m - int(0.02 * sr)):m] ** 2))) or 1e-6
        nt = int(min(T, 4.0) * sr)
        rng = np.random.default_rng(int(seed))
        te = np.arange(nt) / sr
        env = np.exp(-6.9078 * te / T) * np.minimum(te / 0.005, 1.0)
        tl = rng.standard_normal((nt, 2)) * env[:, None]
        tl *= ref / max(float(np.sqrt(np.mean(tl[:int(0.02 * sr)] ** 2))), 1e-9) * 0.7
        rir = np.concatenate([rir, np.zeros((nt, 2))])
        rir[m:m + nt] += tl
    return rir


@effect("binaural_room", "Binaural room: image-source reflections each through the HRTF of their arrival direction, Sabine tail. Mono in, stereo out.",
        lx=(6.0, "room width m"), ly=(8.0, "room length m"), lz=(3.0, "room height m"), src_x=(3.5, "source x, m"), src_y=(5.0, "source y, m"),
        src_z=(1.5, "source z, m"), lis_x=(2.5, "listener x, m"), lis_y=(2.0, "listener y, m"), lis_z=(1.6, "listener z, m"),
        material=("plaster", "wall material of reverbs.ABSORPTION (the floor and ceiling too)"), order=(4, "image-source order"),
        yaw=(0.0, "head turn, degrees to the right"), head=(HEAD_A, "head radius m"), hrtf=("", ".sofa / stereo .wav path, empty = parametric"),
        mix=(1.0, "wet 0..1"))
def binaural_room(x, sr=DEFAULT_SR, lx=6.0, ly=8.0, lz=3.0, src_x=3.5, src_y=5.0, src_z=1.5, lis_x=2.5, lis_y=2.0, lis_z=1.6,
                  material="plaster", order=4, yaw=0.0, head=HEAD_A, hrtf="", mix=1.0):
    m = to_mono(np.asarray(x, float))
    ir = binaural_room_ir((lx, ly, lz), (src_x, src_y, src_z), (lis_x, lis_y, lis_z), material, sr, int(order), yaw=yaw,
                          hrtf=load_hrir(hrtf) if hrtf else None, head=float(head))
    wet = np.stack([sg.fftconvolve(m, ir[:, c]) for c in range(2)], 1)
    dry = np.zeros_like(wet)
    dry[:m.size] = m[:, None]
    mx = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - mx) * dry + mx * wet


# ============================================================================ free reeds
# Blade modes of a clamped-free cantilever (frequency ratios), weighted by the coefficients F1-F3 of `physical.winds.FREE_REED_F`.
# The blade (unit modal mass, force per unit mass identical for every mode, so the static compliance of mode n falls as 1/ratio^2) opens
# a slot h = h0 + x; the plenum pressure p follows kappa (u0 - U) with U = h sqrt(p - p_pipe) the Bernoulli flow; the pressure drop
# pushes the blade. An optional pipe (closed at the reed, open at the far end: a quarter-wave round trip) loads the slot.
FREE_REED_RATIOS = (1.0, 6.267, 17.547)


@numba.njit(cache=True)
def _reed_kernel(u0, os, dt, w, Q, F, G, h0, kap, M, Z0, rr):
    n = u0.size
    out = np.zeros(n)
    q = np.zeros(3)
    qd = np.zeros(3)
    p = 0.0
    U = 0.0
    buf = np.zeros(max(M, 1))
    idx = 0
    lp = 0.0
    for i in range(n):
        acc = 0.0
        for _ in range(os):
            x = 0.0
            for k in range(3):
                x += F[k] * q[k]
            h = h0 + x
            if h < 0.0:
                h = 0.0
            inc = 0.0
            if M > 0:
                inc = buf[idx]
            dp = p - (2.0 * inc + Z0 * U)
            U = h * math.sqrt(abs(dp)) * (1.0 if dp >= 0.0 else -1.0)
            if M > 0:
                lp += 0.6 * ((inc + Z0 * U) - lp)
                buf[idx] = -rr * lp
                idx += 1
                if idx >= M:
                    idx = 0
            for k in range(3):
                qd[k] += dt * (w[0] * w[0] * G * F[k] * dp - w[k] / Q * qd[k] - w[k] * w[k] * q[k])
                q[k] += dt * qd[k]
            p += dt * kap * (u0[i] - U)
            acc += U
        out[i] = acc / os
    return out


FREE_REED_PRESETS = presets.schema("free_reed", {
    "Q": presets.F(float, lo=1.0, hi=1000.0), "h0": presets.F(float, lo=0.01, hi=5.0), "G": presets.F(float, lo=0.0, hi=100.0),
    "u0": presets.F(float, lo=0.0, hi=5.0), "pipe": presets.F(float, lo=0.0, hi=10.0), "cents": presets.F(float, lo=-1200.0, hi=1200.0, unit="cents"),
    "tone": presets.F(float, lo=100.0, hi=20000.0, unit="Hz")})
FREE_REED_SHIFT = 0.008


def free_reed_voice(freq, dur, sr=DEFAULT_SR, preset="harmonica", vel=1.0, attack=0.03, **over):
    """One self-oscillating free reed at `freq` (the reed's natural frequency is freq / (1 + FREE_REED_SHIFT): a reed blown open
    sounds slightly above it). Output is the radiated sound, d(flow)/dt, peak normalised. A reed loaded by a pipe sounds off its
    own frequency: a short pilot measures it and the reed is retuned."""
    if {**FREE_REED_PRESETS[preset], **over}["pipe"] > 0:
        from .friction import _spectral_peak
        want, freq = float(freq), float(freq)
        for _ in range(2):
            pilot = _free_reed_raw(freq, 0.4, sr, preset, vel, attack, **over)
            f = _spectral_peak(pilot[int(0.1 * sr):], want, sr)
            if f <= 0.0 or abs(np.log2(f / want)) < 0.0015:
                break
            freq *= float(np.clip(want / f, 0.97, 1.03))
    return _free_reed_raw(freq, dur, sr, preset, vel, attack, **over)


def _free_reed_raw(freq, dur, sr=DEFAULT_SR, preset="harmonica", vel=1.0, attack=0.03, **over):
    p = {**FREE_REED_PRESETS[preset], **over}
    os = 8
    fi = sr * os
    f0 = float(freq) / (1.0 + FREE_REED_SHIFT)
    n = samples(dur + 0.06, sr)
    t = np.arange(n) / sr
    u0 = p["u0"] * (0.7 + 0.3 * float(vel)) * np.clip(t / max(attack, 1e-3), 0, 1) * np.clip((dur + 0.06 - t) / 0.05, 0, 1)
    w = 2.0 * np.pi * f0 * np.array(FREE_REED_RATIOS)
    from .physical.winds import FREE_REED_F
    M = int(round(fi / (2.0 * f0 * p["pipe"]))) if p["pipe"] > 0 else 0
    y = _reed_kernel(u0, os, 1.0 / fi, w, float(p["Q"]), np.asarray(FREE_REED_F, float), float(p["G"]), float(p["h0"]),
                     3000.0 * f0 / 440.0, M, 0.3, 0.9)
    y = np.diff(y, prepend=0.0)
    y = F.highpass(y, 30.0, sr)
    y = F.lowpass(y, min(p["tone"], 0.45 * sr), sr)
    return y / max(float(np.max(np.abs(y))), 1e-9)


def _reed_instrument(name, desc, preset, extra=None):
    params = dict(second=(0.0, "semitones of a second reed sounding with the first (0 = none, 7 = fifth, -12 = drone below)"))
    if preset == "accordion":
        params["musette"] = (9.0, "cents the second reed is tuned above the first (0 = one reed)")

    @instrument(name, desc, family="winds", span=("C3", "C6"), **params)
    def make(freq, dur, sr=DEFAULT_SR, vel=1.0, second=0.0, musette=None):
        y = free_reed_voice(freq, dur, sr, preset, vel)
        c2 = 0.0
        if preset == "accordion":
            c2 = float(FREE_REED_PRESETS["accordion"]["cents"] if musette is None else musette)
            if c2:
                y = 0.6 * (y + free_reed_voice(freq * 2 ** (c2 / 1200.0), dur, sr, preset, vel))
        if second:
            y = 0.7 * (y + 0.8 * free_reed_voice(freq * 2 ** (float(second) / 12.0), dur, sr, preset, vel))
        y = F.lowpass(y, float(np.clip(6.0 * freq, 1800.0, 3200.0)), sr, order=2)
        t = np.arange(y.size) / sr
        return y * np.minimum(t / 0.004, 1.0) * np.clip((dur + 0.06 - t) / 0.03, 0, 1) * 0.5 * (0.5 + 0.5 * vel)
    make.__name__ = name
    return make


harmonica_reed = _reed_instrument("harmonica_reed", "Harmonica reed: a blade of three cantilever modes opening a slot, plenum pressure feedback (blown-open free reed).", "harmonica")
accordion_reed = _reed_instrument("accordion_reed", "Accordion reeds: free reed physical model, two reeds tuned a few cents apart (musette).", "accordion")
melodica_reed = _reed_instrument("melodica_reed", "Melodica: free reed in a small plenum, softer blade (lower Q).", "melodica")
sho_reed = _reed_instrument("sho_reed", "Shō: a brass free reed loaded by a bamboo pipe tuned to it; `second` adds an aitake pipe.", "sho")
khaen_reed = _reed_instrument("khaen_reed", "Khaen: free reed + long pipe, brighter and rougher than the shō; `second` -12 gives the drone pipe.", "khaen")


# ============================================================================ measured violin / guitar bodies
# (name, frequency Hz, Q, linear peak boost). Typical values of the published modal analyses, rounded; see SOURCES.md.
BODY_MODES = {
    "violin": (("T1", 165.0, 25.0, 0.6), ("A0", 275.0, 30.0, 3.0), ("CBR", 405.0, 35.0, 1.2), ("B1-", 480.0, 30.0, 2.0),
               ("B1+", 550.0, 35.0, 2.2), ("A1", 454.0, 40.0, 0.8), ("T2", 700.0, 30.0, 0.9), ("M7", 900.0, 30.0, 0.8),
               ("M9", 1200.0, 30.0, 0.7), ("bridge hill", 2500.0, 3.0, 1.0), ("M3k", 3100.0, 6.0, 0.5)),
    "guitar_classical": (("T(1,1)1", 98.0, 38.0, 3.0), ("T(1,1)2", 197.0, 42.0, 2.5), ("T(2,1)", 258.0, 38.0, 1.0),
                         ("T(0,1)", 285.0, 38.0, 0.8), ("T(1,1)3", 372.0, 38.0, 0.9), ("T(3,1)", 412.0, 38.0, 0.7),
                         ("T(2,2)", 504.0, 38.0, 0.7), ("T(1,3)", 562.0, 38.0, 0.6), ("T(4,1)", 700.0, 35.0, 0.5),
                         ("T(3,2)", 850.0, 35.0, 0.4), ("top 1.1k", 1100.0, 30.0, 0.4)),
    "guitar_steel": (("T(1,1)1", 105.0, 40.0, 3.0), ("T(1,1)2", 205.0, 45.0, 2.2), ("T(2,1)", 270.0, 40.0, 0.9),
                     ("T(1,1)3", 390.0, 40.0, 0.8), ("T(3,1)", 440.0, 40.0, 0.7), ("T(2,2)", 520.0, 38.0, 0.6),
                     ("T(1,3)", 600.0, 38.0, 0.5), ("T(4,1)", 750.0, 35.0, 0.4), ("top 1.2k", 1200.0, 30.0, 0.35)),
}


def _resonator(f, q, sr):
    r = math.exp(-math.pi * f / (q * sr))
    w = 2.0 * math.pi * f / sr
    b, a = np.array([1.0, 0.0, -1.0]), np.array([1.0, -2.0 * r * math.cos(w), r * r])
    return b / abs(sg.freqz(b, a, [w])[1][0]), a


def body_apply(y, sr, body="violin", amount=1.0, spread=0.0, seed=0):
    """y plus `amount` of the modal body: parallel unit-peak resonators at the published modes, scaled by each mode's boost
    (peak gain 1 + amount * boost at the mode). `spread`: relative s.d. of the mode frequencies (a different instrument)."""
    modes = BODY_MODES[body]
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y, float)
    out = y.copy()
    for _, f, q, g in modes:
        f = f * math.exp(float(spread) * rng.standard_normal()) if spread else f
        if f < 0.45 * sr:
            out += float(amount) * g * sg.lfilter(*_resonator(f, q, sr), y, axis=0)
    return out


@effect("body_measured", "Violin or guitar body: parallel modal filters at the published mode frequencies, Q and strengths (A0, CBR, B1-, B1+ / T(1,1)1, T(1,1)2...).",
        body=("violin", "violin | guitar_classical | guitar_steel"), amount=(1.0, "0..2 body strength"),
        spread=(0.0, "relative s.d. of the mode frequencies (0.06 = Woodhouse's spread between instruments)"), seed=(0, "int"))
def body_measured(x, sr=DEFAULT_SR, body="violin", amount=1.0, spread=0.0, seed=0):
    return body_apply(x, sr, str(body), amount, spread, seed)


@numba.njit(cache=True)
def _ks(n, N, damp, seed_noise, c):
    y = np.zeros(n)
    buf = seed_noise[:N].copy()
    prev = 0.0
    ax = 0.0
    ay = 0.0
    for i in range(n):
        j = i % N
        v = buf[j]
        y[i] = v
        avg = damp * 0.5 * (v + prev)
        prev = v
        o = c * avg + ax - c * ay
        ax = avg
        ay = o
        buf[j] = o
    return y


def _string(freq, dur, sr, kind, vel, seed):
    n = samples(dur + 0.3, sr)
    if kind == "pluck":
        D = sr / freq
        N = max(2, int(np.floor(D - 1.0)))
        d = D - 0.5 - N
        c = (1.0 - d) / (1.0 + d)
        noise = np.random.default_rng(int(seed)).uniform(-1, 1, N)
        noise = sg.lfilter([1.0 - 0.6], [1.0, -0.6], noise)
        y = _ks(n, N, 0.9985, noise, c)
    else:
        t = np.arange(n) / sr
        ph = np.cumsum(freq * (1 + 0.003 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(t / 0.4, 1.0))) / sr
        y = sum(np.sin(2 * np.pi * k * ph) / k for k in range(1, max(2, int(0.45 * sr / freq) + 1)))
        y = y * np.minimum(t / 0.06, 1.0) * np.clip((dur + 0.3 - t) / 0.2, 0, 1)
    return y / max(float(np.max(np.abs(y))), 1e-9) * (0.4 + 0.6 * vel)


@instrument("guitar_modal", "Plucked string (Karplus-Strong) through the measured guitar body modes (T(1,1)1, T(1,1)2, ...).", family="strings",
            span=("E2", "E5"), body=("guitar_classical", "guitar_classical | guitar_steel"), amount=(1.0, "body strength"), seed=(0, "int"))
def guitar_modal(freq, dur, sr=DEFAULT_SR, vel=1.0, body="guitar_classical", amount=1.0, seed=0):
    y = body_apply(_string(freq, dur, sr, "pluck", vel, seed), sr, body, amount)
    return y / max(float(np.max(np.abs(y))), 1e-9) * 0.7 * (0.5 + 0.5 * vel)


@instrument("violin_modal", "Bowed Helmholtz sawtooth through the measured violin body modes (A0, CBR, B1-, B1+, bridge hill).", family="strings",
            span=("G3", "E7"), amount=(1.0, "body strength"), seed=(0, "int"))
def violin_modal(freq, dur, sr=DEFAULT_SR, vel=1.0, amount=1.0, seed=0):
    y = body_apply(_string(freq, dur, sr, "bow", vel, seed), sr, "violin", amount)
    y = F.lowpass(y, float(np.clip(4.5 * freq, 1800.0, 3000.0)), sr, order=2)
    return y / max(float(np.max(np.abs(y))), 1e-9) * 0.7 * (0.5 + 0.5 * vel)


# ============================================================================ tape: Jiles-Atherton + HF bias
JA = dict(Ms=380.0, a=720.0, k=380.0, alpha=0.015, c=0.25)         # constants of `dsp.ja_hysteresis`; H in units of Ms (a/Ms = 1.9)
TAPE_HMAX = 6.0                                                      # head field at digital full scale
TAPE_BIAS_OPT = 2.0                                                  # bias amplitude that linearises the curve (units of H)


def _ja_M(H):
    """Magnetisation (-1..1) after each sample of the field series H, from the Jiles-Atherton kernel of `dsp`."""
    return D._ja_kernel(np.ascontiguousarray(H, float), 1.0, JA["Ms"], JA["a"], JA["k"], JA["alpha"], JA["c"], 4) * JA["c"] * JA["Ms"] / JA["a"]


def tape_bh_loop(hmax=TAPE_HMAX, cycles=3, points=256):
    """Major B-H (M-H) loop of the tape: (H, M) of the last of `cycles` sinusoidal sweeps of amplitude hmax."""
    j = np.arange(cycles * points)
    H = hmax * np.sin(2.0 * np.pi * j / points)
    M = _ja_M(H)
    return H[-points:], M[-points:]


@lru_cache(maxsize=32)
def tape_record_curve(bias=1.0, hmax=TAPE_HMAX, n=121, cycles=12, pts=8):
    """Remanent magnetisation left on the tape against the head field H: each tape element crosses the gap with the audio field H
    plus an HF bias of amplitude bias * TAPE_BIAS_OPT, both fading as it leaves; what stays is the recorded level. bias 0 = no bias
    (a dead zone and strong curvature: crossover distortion), 1 = the linear bias, > 1 over-bias (less output, less distortion)."""
    H = np.linspace(-hmax, hmax, n)
    j = np.arange(cycles * pts)
    env = 1.0 - j / (cycles * pts)
    sinb = float(bias) * TAPE_BIAS_OPT * np.sin(2.0 * np.pi * j / pts)
    M = np.array([_ja_M((h + sinb) * env)[-1] for h in H])
    return H, M


def tape_record(x, bias=1.0, hmax=TAPE_HMAX):
    """x (full scale 1) through the recording characteristic, referred to the linear-bias curve at half scale (unity gain there)."""
    H, M = tape_record_curve(float(bias), float(hmax))
    Hr, Mr = tape_record_curve(1.0, float(hmax))
    ref = float(np.interp(0.5 * hmax, Hr, Mr)) / (0.5 * hmax)
    return np.interp(np.asarray(x, float) * hmax, H, M) / (ref * hmax)


@effect("tape_bias", "Tape with a physical record stage: Jiles-Atherton hysteresis with HF bias (under / over-bias), wow, flutter, play-head loss, hiss.",
        bias=(1.0, "bias relative to the linear value: 0 none (crossover distortion), 1 linear, 2 over-biased"),
        drive=(0.6, "record level into the head, 0.05..2 (1 = field amplitude TAPE_HMAX)"),
        wow=(0.15, "wow depth %"), flutter=(0.08, "flutter depth %"), wow_rate=(0.6, "wow Hz"),
        speed=(7.5, "tape speed ips for the play-head loss; 0 = off"), hf=(14000, "playback roll-off Hz"), hiss=(-58, "hiss dB rms (-120 none)"),
        seed=(0, "int"))
def tape_bias(x, sr=DEFAULT_SR, bias=1.0, drive=0.6, wow=0.15, flutter=0.08, wow_rate=0.6, speed=7.5, hf=14000, hiss=-58, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(int(seed))
    n = x.shape[0]
    t = np.arange(n) / sr
    # ponytail: same wow / flutter recipe as `dsp.tape` (7.3 Hz flutter + smoothed drift), not factored out of it
    drift = D.moving_average(rng.standard_normal(n), int(sr / 20), 2)
    drift /= max(np.max(np.abs(drift)), 1e-9)
    rate = (1.0 + float(wow) / 100 * np.sin(2 * np.pi * float(wow_rate) * t + rng.uniform(0, 6.28))
            + float(flutter) / 100 * (0.7 * np.sin(2 * np.pi * 7.3 * t + rng.uniform(0, 6.28)) + 0.3 * drift))
    y = tape_record(np.clip(x * float(np.clip(drive, 0.05, 2.0)), -1.0, 1.0), bias) / float(np.clip(drive, 0.05, 2.0))
    if wow or flutter:
        y = D.varispeed(y, rate)
    if speed and speed > 0:
        y = D.playhead_loss(y, sr, float(speed))
    if hf and hf < 0.49 * sr:
        y = F.lowpass(y, float(hf), sr)
    if hiss > -119:
        h = F.highshelf(rng.standard_normal(n), 3000.0, sr, 6.0) * 10.0 ** (float(hiss) / 20.0) * 0.6
        y = y + (h[:, None] if y.ndim == 2 else h)
    return y
