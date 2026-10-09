"""genny.plates: 2D finite-difference membranes and plates (Bilbao 2009, ch.10-13; findings g03).

Run:  uv run --with pytest pytest tests/test_plates.py -q      (plain asserts; also runs as a script)
"""
import math

import numpy as np
import scipy.sparse as sp
from scipy.signal import butter, sosfiltfilt

from genny import drums as D
from genny import fx as FX
from genny import plates as P
from genny import sfx as S

DRUMS = ["fd_drum", "plate_crash", "plate_ride", "plate_china", "plate_gong", "tam_tam", "thunder_sheet"]
SFX = ["metal_plate", "glass_pane", "wood_panel", "plate_bow"]
EFFECTS = ["plate_reverb", "room2d"]


def cents(f, ref):
    return 1200.0 * np.log2(np.asarray(f) / np.asarray(ref))


def t60_band(h, fs, fc):
    """Schroeder decay time of the band around fc, from the -5..-25 dB slope."""
    b = sosfiltfilt(butter(4, [fc / 1.25, fc * 1.25], btype="band", fs=fs, output="sos"), h)
    db = 10.0 * np.log10(np.cumsum(b[::-1] ** 2)[::-1] / np.sum(b * b) + 1e-30)
    return 3.0 * (np.argmax(db < -25.0) - np.argmax(db < -5.0)) / fs


def lowest_peak(y, fs, lo, hi):
    n = 1 << 18
    sp_ = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    sel = (f > lo) & (f < hi)
    return f[sel][np.argmax(sp_[sel])]


# --- mode frequencies ---------------------------------------------------------------------------------
def test_membrane_modes_table_11_1():
    """Explicit scheme, gamma=1000, fs=16 kHz, square, fixed: the book's own numbers (Table 11.1)."""
    m = P._model(16000, 1.0, "ssss", 0.0, 1000.0)
    got = P.scheme_modes(m, 12)
    for hz in (707.0, 1114.0, 1413.1, 2009.2, 2117.5):
        assert np.min(np.abs(got - hz)) < 0.06, (hz, got)
    exact = P.membrane_modes(1000.0, 1.0, 4)
    assert abs(exact[0] - 707.1) < 0.05 and abs(cents(got[0], exact[0])) < 1.0
    assert np.all(np.abs(cents(got[:4], exact)) < 8.0)                # grid dispersion: -0.3, -6.2, -1.3 cents
    assert m.rho <= 4.0                                               # lambda <= 1/sqrt2


def test_plate_modes_table_12_1():
    """Explicit 13-point scheme, kappa=100, fs=44.1 kHz, simply supported (Table 12.1)."""
    m = P._model(44100, 1.0, "ssss", 100.0)
    got = P.scheme_modes(m, 12)
    for hz in (311.6, 764.1, 1217.4, 1470.6, 1926.1, 2366.5):
        assert np.min(np.abs(got - hz)) < 0.06, (hz, got)
    exact = P.plate_modes(100.0, 1.0, 4)
    assert abs(exact[0] - 314.2) < 0.05
    assert np.all(np.abs(cents(got[:4], exact)) < 60.0)               # -14, -48, -48, -55 cents in the book
    assert m.rho <= 4.0                                               # mu <= 1/4


def test_orthotropic_modes_table_12_2():
    m = P._model(44100, 1.248, "ssss", 0.0, 0.0, 0.0, 0.0, 0.3, (36.5, 8.41, 18.1))
    got = P.scheme_modes(m, 8)
    for hz in (56.4, 98.2, 176.1, 224.3, 287.8, 289.1):               # column r = sqrt(ky/kx)
        assert np.min(np.abs(got - hz)) < 0.2, (hz, got)
    assert abs(P.plate_modes(0.0, 1.248, 1, (36.5, 8.41, 18.1))[0] - 56.5) < 0.05
    assert abs(m.hy / m.hx - math.sqrt(8.41 / 36.5)) < 0.03           # eq 12.41


def test_rendered_mode_frequencies():
    """The numba kernel, not just the matrix: spectral peaks of a struck membrane and plate."""
    m = P._model(16000, 1.0, "ssss", 0.0, 1000.0)
    y = P.simulate(m, P.pulse(16000, 16000, 1.0, 1.0), (0.31, 0.37, 0.0), ((0.63, 0.29),))[:, 0]
    assert abs(cents(lowest_peak(y, 16000, 500, 900), 500.0 * math.sqrt(2.0))) < 5.0
    m = P._model(44100, 1.0, "ssss", 100.0)
    y = P.simulate(m, P.pulse(44100, 44100, 1.0, 1.0), (0.31, 0.37, 0.0), ((0.63, 0.29),))[:, 0]
    assert abs(cents(lowest_peak(y, 44100, 200, 500), math.pi * 100.0)) < 20.0     # book: -14.1 cents


def test_free_plate():
    """Free edges: three rigid-body modes, then Leissa's free square plate (nu = 0.3) eigenvalues
    w/kappa = 13.47, 19.60, 24.27, 34.80 (Leissa 1973, quoted from memory, not from the findings)."""
    m = P._model(44100, 1.0, "ffff", 20.0)
    f = P.scheme_modes(m, 8)
    assert np.all(f[:3] < 0.05) and f[3] > 30.0
    lam = 2.0 * math.pi * f[3:8] / 20.0
    assert np.all(np.abs(lam / np.array([13.47, 19.60, 24.27, 34.80, 34.80]) - 1.0) < 0.015), lam


def test_circular_membrane_and_air_cavity():
    """Staircase circle: Bessel-zero ratios; the cavity raises only modes with a net volume change."""
    m = P._model(16000, 1.0, "ssss", 0.0, 300.0, circle=True)
    f = P.scheme_modes(m, 6)
    assert abs(f[0] / (2.405 / math.pi * 300.0) - 1.0) < 0.02
    assert np.all(np.abs(f[[1, 3, 5]] / f[0] / np.array([1.5933, 2.1355, 2.2954]) - 1.0) < 0.01)
    sr = 44100
    pk = []
    for air in (0.0, 1.0):
        y = P.membrane(sr, 1.0, 1000.0, 0.33, 2.0, 0.0, 0.4, 0.9, air, 0.3, 0.0, False, 0.0, 1.0, False, 1.5)
        pk.append((lowest_peak(y, sr, 100, 225), lowest_peak(y, sr, 225.5, 240)))
    f01 = 2.405 / math.pi * math.sqrt(1000.0 / P.HEAD_DENSITY) / 0.33
    assert abs(pk[0][0] / f01 - 1.0) < 0.02                           # 142.3 vs 143.9 Hz
    assert pk[1][0] > 1.3 * pk[0][0]                                  # (0,1) raised by the air spring
    assert abs(pk[1][1] / pk[0][1] - 1.0) < 0.003                     # (1,1) untouched


# --- conservation, stability, parity ------------------------------------------------------------------
def _energy(m, u, u1):
    return (0.5 * np.sum(m.w * ((u - u1) / m.k) ** 2) + 0.5 * np.sum(m.w * u * (m.K @ u1))
            + 0.5 * m.cav * np.sum(m.w * u) * np.sum(m.w * u1))


def test_linear_energy_conservation():
    cases = {"membrane + cavity": P._model(16000, 1.3, "ssss", 0.0, 400.0, cav=400.0 ** 2 * 30.0),
             "plate supported": P._model(16000, 1.3, "ssss", 20.0),
             "plate free": P._model(16000, 1.3, "ffff", 20.0),
             "plate free/clamped": P._model(16000, 1.0, "fccc", 20.0),
             "stiff circular membrane": P._model(16000, 1.0, "ssss", 5.0, 300.0, circle=True),
             "room (Neumann)": P._model(8000, 1.4, "ffff", 0.0, 150.0)}
    for name, m in cases.items():
        rng = np.random.default_rng(1)
        u = rng.standard_normal(m.n)
        u1 = u + 0.1 * rng.standard_normal(m.n)
        e0, drift = _energy(m, u, u1), 0.0
        for _ in range(10):
            P.simulate(m, np.zeros(200), state=(u, u1))
            drift = max(drift, abs(_energy(m, u, u1) / e0 - 1.0))
        assert drift < 1e-10, (name, drift)
        assert m.rho <= 4.0 - m.k ** 2 * m.cav, name


def test_stability_bound_with_loss():
    """rho(k^2 K - 4 sigma1 k L) <= 4 for every boundary / loss / anisotropy combination; and a lossy run
    never gains energy."""
    for args in ((16000, 2.0, "ffff", 2.0, 0.0, 1.0, 0.02), (8000, 1.0, "cccc", 30.0, 0.0, 0.5, 0.2),
                 (16000, 1.5, "fscf", 8.0, 250.0, 2.0, 0.01), (22000, 3.0, "ssss", 0.0, 900.0, 3.0, 1e-4),
                 (16000, 1.2, "ssss", 0.0, 0.0, 1.0, 0.01, 0.3, (40.0, 5.0, 20.0))):
        m = P._model(*args)
        assert m.rho <= 4.0, args
        u = np.random.default_rng(2).standard_normal(m.n)
        u1 = u.copy()
        P.simulate(m, np.zeros(4000), state=(u, u1))
        assert np.all(np.isfinite(u)) and np.max(np.abs(u)) < 10.0, args


def test_kernel_matches_matrix_reference():
    m = P._model(16000, 1.3, "ffff", 20.0, 0.0, 2.0, 0.002)
    f = np.zeros(300)
    f[0] = 1.0
    out = P.simulate(m, f, (0.3, 0.4, 0.0), ((0.6, 0.3),))[:, 0]
    B, C = sp.csr_matrix(m.B, shape=(m.n, m.n)), sp.csr_matrix(m.C, shape=(m.n, m.n))
    ji, jw = P._spread(m, 0.3, 0.4, 0.0)
    pk = P._pickups(m, ((0.6, 0.3),))
    u, u1, ref = np.zeros(m.n), np.zeros(m.n), []
    for n in range(300):
        t = B @ u - C @ u1
        t[ji] += m.a * m.k ** 2 * f[n] * jw
        u1, u = u, t
        ref.append(P._read(u, m.idx, pk[0, 0], pk[0, 1]))
    assert np.max(np.abs(out - np.array(ref))) < 1e-10 * np.max(np.abs(ref))


def test_plate_t60_two_bands():
    fs = 16000
    for t1, t2 in ((4.0, 2.0), (2.0, 1.0)):
        s0, s1 = P.loss_coeffs(500.0, t1, 2000.0, t2, 2.0, 0.0, fs)
        m = P._model(fs, 2.0, "ffff", 2.0, 0.0, round(s0, 6), round(s1, 10))
        imp = np.zeros(int(fs * 0.8 * t1))
        imp[0] = 1.0
        h = P._vel(P.simulate(m, imp, (0.57, 0.5, 0.0), ((0.43, 0.43),)), fs)[:, 0]
        assert abs(t60_band(h, fs, 500.0) / t1 - 1.0) < 0.15, (t1, t60_band(h, fs, 500.0))
        assert abs(t60_band(h, fs, 2000.0) / t2 - 1.0) < 0.15, (t2, t60_band(h, fs, 2000.0))


# --- nonlinear plates ---------------------------------------------------------------------------------
def _vk_energy(m, u, u1):
    ch, hh = P._airy(m.Nx, m.Ny, m.hx, m.hy), m.hx * m.hy
    cc = np.empty((m.Nx, m.Ny))

    def airy(v):
        up = np.zeros((m.Nx + 1, m.Ny + 1))
        up[1:-1, 1:-1] = v.reshape(m.Nx - 1, m.Ny - 1)
        q = np.empty(m.n)
        P._bracket(up, up, cc, m.hx, m.hy, q)
        phi = -q
        P._chol_solve(ch, phi)
        return phi, -q                                                # Phi and BIH Phi

    return (0.5 * hh * np.sum(((u - u1) / m.k) ** 2) + 0.5 * hh * np.sum(u * (m.K @ u1))
            + 0.25 * m.kappa ** 2 * hh * np.sum(airy(u)[0] * airy(u1)[1]))


def test_bracket_and_nonlinear_energy():
    N, h = 40, 1.0 / 40
    X, Y = np.meshgrid(np.arange(N + 1) * h, np.arange(N + 1) * h, indexing="ij")
    a = np.sin(math.pi * X) * np.sin(math.pi * Y)
    a[[0, -1], :] = 0.0
    a[:, [0, -1]] = 0.0
    o, cc = np.empty((N - 1) ** 2), np.empty((N, N))
    P._bracket(a, a, cc, h, h, o)
    exact = 2 * math.pi ** 4 * ((np.sin(math.pi * X) * np.sin(math.pi * Y)) ** 2
                                - (np.cos(math.pi * X) * np.cos(math.pi * Y)) ** 2)[1:-1, 1:-1]
    assert np.max(np.abs(o.reshape(N - 1, N - 1) - exact)[3:-3, 3:-3]) < 0.005 * np.max(np.abs(exact))
    rng = np.random.default_rng(0)
    A, B, C = (np.pad(rng.standard_normal((N - 1, N - 1)), 1) for _ in range(3))
    o2 = np.empty_like(o)
    P._bracket(B, C, cc, h, h, o)
    P._bracket(A, B, cc, h, h, o2)
    lhs, rhs = np.sum(A[1:-1, 1:-1].ravel() * o), np.sum(o2 * C[1:-1, 1:-1].ravel())
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)                          # triple self-adjointness (eq 13.11)

    m = P._model(16000, 1.0, "ssss", 10.0)
    X, Y = np.meshgrid(np.arange(1, m.Nx) / m.Nx, np.arange(1, m.Ny) / m.Ny, indexing="ij")
    drift = {}
    for model in ("vk", "vk_simple"):                                 # first-mode shape, peak 12 thicknesses
        u = (12.0 * np.sin(math.pi * X) * np.sin(math.pi * Y)).ravel()
        u1 = u.copy()
        e0, d, hits = _vk_energy(m, u, u1), 0.0, 0
        for _ in range(10):
            hits += P.simulate_nl(m, np.zeros(100), model, state=(u, u1), ulim=1e9)[1]
            d = max(d, abs(_vk_energy(m, u, u1) / e0 - 1.0))
        drift[model] = d
        assert hits == 0
    assert drift["vk"] < 1e-9, drift
    assert drift["vk_simple"] > 1e3 * drift["vk"], drift               # the book's scheme is not conservative

    # Berger: E = kinetic + kappa^2 <u, BIH u->/2 + kappa^2 <grad u, grad u->^2 / 4
    hh = m.hx * m.hy
    berger = lambda u, u1: (0.5 * hh * np.sum(((u - u1) / m.k) ** 2) + 0.5 * hh * np.sum(u * (m.K @ u1))
                            + 0.25 * m.kappa ** 2 * (hh * np.sum(u * (m.L @ u1))) ** 2)
    u = (3.0 * np.sin(math.pi * X) * np.sin(2 * math.pi * Y) + 2.0 * np.sin(math.pi * X) * np.sin(math.pi * Y)).ravel()
    u1 = u.copy()
    e0, d = berger(u, u1), 0.0
    for _ in range(10):
        P.simulate_nl(m, np.zeros(100), "berger", state=(u, u1), ulim=1e9)
        d = max(d, abs(berger(u, u1) / e0 - 1.0))
    assert d < 1e-10, d


def test_pitch_glide():
    """Gong-like plate of Fig 13.4 (kappa=20, sigma0=1.38, sigma1=0.001, centre strike, 3 ms): the lowest
    mode starts sharp and glides down when struck hard, and sits at its linear frequency when struck softly."""
    fs = 16000
    m = P._model(fs, 1.0, "ssss", 20.0, 0.0, 1.38, 0.001)
    f_lin = P.scheme_modes(m, 1)[0]
    track = lambda v, t: lowest_peak(v[int(t * fs):int(t * fs) + 2048], fs, 40.0, 130.0)
    for model in ("berger", "vk"):
        soft = np.diff(P.simulate_nl(m, P.pulse(fs, fs, 1.9e4, 3.0), model, (0.5, 0.5, 0.0))[0][:, 0])
        hard = np.diff(P.simulate_nl(m, P.pulse(fs, fs, 1.9e5, 3.0), model, (0.5, 0.5, 0.0))[0][:, 0])
        assert abs(track(soft, 0.02) / f_lin - 1.0) < 0.01 and abs(track(soft, 0.8) / f_lin - 1.0) < 0.01, model
        f0, f1, f2 = track(hard, 0.02), track(hard, 0.3), track(hard, 0.8)
        assert f0 > 1.05 * f_lin and f0 > f1 > f2 >= 0.995 * f_lin, (model, f0, f1, f2, f_lin)


def test_crash_energy_cascade():
    """Crash plate of Fig 13.5 (kappa=5, sigma0=1.38, sigma1=0.005, strike (0.2,0.3), 16 ms, 16 kHz):
    after a hard strike the energy above 1 kHz and the spectral centroid keep rising once the mallet has
    left; after a soft one they fall. A linear plate cannot do the first."""
    fs = 16000
    m = P._model(fs, 1.0, "ssss", 5.0, 0.0, 1.38, 0.005)
    hp = butter(4, 1000.0, btype="high", fs=fs, output="sos")
    hp150 = butter(4, 150.0, btype="high", fs=fs, output="sos")

    def measure(fmax, model):
        out, hits = P.simulate_nl(m, P.pulse(int(0.3 * fs), fs, fmax, 16.0), model, (0.2, 0.3, 0.0), ((0.63, 0.41),))
        assert hits == 0 and np.all(np.isfinite(out))
        acc = sosfiltfilt(hp150, np.diff(out[:, 0], 2))
        hi = sosfiltfilt(hp, np.diff(out[:, 0]))
        a, b = slice(int(0.02 * fs), int(0.1 * fs)), slice(int(0.1 * fs), int(0.3 * fs))
        f = lambda s: np.fft.rfftfreq(len(s), 1.0 / fs)
        cen = lambda s: np.sum(f(s) * np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2) / np.sum(
            np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2)
        return 10 * np.log10(np.mean(hi[b] ** 2) / np.mean(hi[a] ** 2)), cen(acc[a]), cen(acc[b])

    soft, hard, lin = measure(1.3e3, "vk"), measure(1.3e5, "vk"), measure(1.3e5, "linear")
    print("cascade (dB change >1 kHz, centroid early, late):", soft, hard, lin)
    assert hard[0] > 1.0 and soft[0] < -3.0 and lin[0] < -3.0, (soft, hard, lin)       # dB change above 1 kHz
    assert hard[2] > hard[1] and soft[2] < 0.9 * soft[1], (soft, hard)                 # centroid: +3 % vs -20 %
    assert hard[1] > 1.4 * soft[1]                                                     # and brighter overall
    test_crash_energy_cascade.numbers = dict(soft=soft, hard=hard, linear=lin)


def test_mallet_contact_brightens_with_velocity():
    cen = []
    for v in (0.2, 1.0):
        y = P.membrane(44100, v, contact=True, dur=0.4)
        sp_ = np.abs(np.fft.rfft(y)) ** 2
        cen.append(np.sum(np.fft.rfftfreq(len(y), 1 / 44100) * sp_) / np.sum(sp_))
    assert cen[1] > cen[0], cen


# --- effects ------------------------------------------------------------------------------------------
def test_plate_reverb_impulse_response():
    sr = 44100
    x = np.zeros(2000)
    x[10] = 1.0
    y = P.plate_reverb(x, sr, mix=1.0, t60_low=2.0, t60_high=1.0, tail=2.0)
    assert y.shape == (2000 + 2 * sr, 2) and np.all(np.isfinite(y))
    for ch in range(2):
        assert abs(t60_band(y[:, ch], sr, 500.0) / 2.0 - 1.0) < 0.2, t60_band(y[:, ch], sr, 500.0)
        assert abs(t60_band(y[:, ch], sr, 2000.0) / 1.0 - 1.0) < 0.2, t60_band(y[:, ch], sr, 2000.0)
        assert abs(np.mean(y[:, ch])) < 1e-3 * np.sqrt(np.mean(y[:, ch] ** 2))          # no DC
    assert np.corrcoef(y[:, 0], y[:, 1])[0, 1] < 0.5                                   # two pickups: decorrelated
    mono = P.plate_reverb(x, sr, mix=1.0, t60_low=2.0, t60_high=1.0, tail=2.0, width=0.0)
    assert mono.ndim == 1
    assert np.allclose(P.plate_reverb(2.0 * x, sr, mix=1.0, tail=0.3), 2.0 * P.plate_reverb(x, sr, mix=1.0, tail=0.3))
    dry = P.plate_reverb(np.stack([x, x], axis=1), sr, mix=0.0, tail=0.2)
    assert dry.shape[1] == 2 and np.allclose(dry[:2000, 0], x)


# --- everything registered ------------------------------------------------------------------------------
def _ok(y, name):
    y = np.asarray(y)
    assert y.size > 0 and np.all(np.isfinite(y)), name
    peak = float(np.max(np.abs(y)))
    assert 1e-4 < peak < 1.5, (name, peak)
    m = y if y.ndim == 1 else y.mean(axis=1)
    assert abs(m[0]) < 1e-3 and abs(m[-1]) < 1e-3, (name, m[0], m[-1])
    assert abs(np.mean(m)) < 0.02 * peak, (name, "dc")


def _noise(sr, dur=0.3):
    x = np.random.default_rng(0).standard_normal(int(dur * sr)) * 0.3
    return x * np.hanning(len(x))


SHORT = {"plate_crash": dict(dur=0.4), "plate_ride": dict(dur=0.4), "plate_china": dict(dur=0.4),
         "plate_gong": dict(dur=0.5), "tam_tam": dict(dur=0.5), "thunder_sheet": dict(dur=0.4),
         "plate_bow": dict(dur=0.4), "fd_drum": {}, "metal_plate": dict(decay=0.6), "glass_pane": {},
         "wood_panel": {}}


def test_registered_names_render_at_two_rates():
    for name in DRUMS + SFX:
        reg = D.REGISTRY if name in DRUMS else S.REGISTRY
        assert name in reg and all(len(v) == 2 for v in reg[name]["params"].values()), name
        for sr in (22050, 48000):
            y = reg[name]["fn"](sr=sr, **SHORT[name])
            assert y.ndim == 1, name
            _ok(y, name)
        a, b = (reg[name]["fn"](sr=22050, **SHORT[name]) for _ in range(2))
        assert np.array_equal(a, b), name                              # deterministic
    for name in EFFECTS:
        assert name in FX.REGISTRY
        for sr in (22050, 48000):
            x = _noise(sr)
            for inp in (x, np.stack([x, 0.5 * x], axis=1)):
                y = FX.REGISTRY[name]["fn"](inp, sr=sr, tail=0.3)
                assert y.shape == (len(x) + int(0.3 * sr), 2), (name, y.shape)
                _ok(y, name)


def test_not_piercing_by_default():
    for name in DRUMS + SFX:
        reg = D.REGISTRY if name in DRUMS else S.REGISTRY
        y = reg[name]["fn"](sr=44100, **SHORT[name])
        sp_ = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / 44100)
        assert np.sum(sp_[f > 5000]) / np.sum(sp_) < 0.05, name


EXTREMES = {
    "fd_drum": [dict(tension=1, size=5, decay=0.001, damping=5, pos=-1, hardness=-1, air=9, depth=0, stiffness=9, quality=0.01),
                dict(tension=1e6, size=0.01, decay=99, damping=0, pos=3, hardness=3, air=1, depth=1e-6, stiffness=1, contact=1, scan=50, quality=1.5),
                dict(tension=300, size=0.6, air=1, depth=0.03, contact=1, vel=0.0, hardness=1)],
    "plate_crash": [dict(size=0.1, thickness=20, force=1e7, decay=0.01, dur=0.2, bright=2),
                    dict(size=3, thickness=0.2, force=1e7, decay=99, dur=0.2, quality=0.25, bright=-1),
                    dict(force=1e6, dur=0.3, model="vk_simple"), dict(force=1e6, dur=0.3, model="berger"),
                    dict(force=0.0, dur=0.2, model="linear", vel=0.0)],
    "plate_ride": [dict(force=1e7, dur=0.2), dict(size=0.1, thickness=20, dur=0.2)],
    "plate_china": [dict(force=1e7, dur=0.2), dict(size=3, thickness=0.2, dur=0.2, quality=0.25)],
    "plate_gong": [dict(force=1e8, dur=0.3), dict(size=0.1, thickness=20, force=1e8, dur=0.2)],
    "tam_tam": [dict(force=1e7, dur=0.2, quality=0.5), dict(size=3, thickness=0.2, force=1e6, dur=0.2, quality=0.25)],
    "thunder_sheet": [dict(force=1e5, rate=99, dur=0.2, quality=0.5, seed=3), dict(size=0.1, thickness=20, rate=0, dur=0.2)],
    "metal_plate": [dict(size=0.05, thickness=50, decay=0.01, aspect=9, edges="clamped", pos=2, hardness=2, vel=2, bright=2, quality=3),
                    dict(size=3, thickness=0.1, decay=0.3, edges="supported", pos=-1, hardness=-1, vel=0, bright=-1, quality=0.25),
                    dict(edges="nonsense", decay=0.2)],
    "glass_pane": [dict(size=0.05, thickness=50, decay=0.01, aspect=9), dict(size=3, thickness=0.1, decay=0.3, quality=0.25)],
    "wood_panel": [dict(size=0.01, thickness=99, decay=0.01), dict(size=9, thickness=0.01, decay=0.5, quality=0.25)],
    "plate_bow": [dict(dur=0.3, force=1e6, speed=99, pos=5, size=0.01, thickness=99, decay=0.01),
                  dict(dur=0.3, force=0.0, speed=0.0, pos=-5, size=5, thickness=0.01, decay=99)],
}
FX_EXTREMES = {
    "plate_reverb": [dict(mix=2, t60_low=99, t60_high=99, size=9, damping=-1, predelay=0.05, width=3, tail=0.2, quality=0.1),
                     dict(mix=0.5, t60_low=0.01, t60_high=5, size=0.01, damping=5, width=-1, tail=0.2, quality=2)],
    "room2d": [dict(mix=1, size=99, aspect=9, t60=99, damping=-1, tail=0.2, quality=0.5),
               dict(mix=1, size=0.1, aspect=0.1, t60=0.001, damping=9, predelay=0.02, tail=0.2, quality=2)],
}


def test_stable_at_parameter_extremes():
    for name, sets in EXTREMES.items():
        reg = D.REGISTRY if name in DRUMS else S.REGISTRY
        for kw in sets:
            _ok(reg[name]["fn"](sr=44100, **kw), (name, kw))
    x = _noise(22050, 0.15)
    for name, sets in FX_EXTREMES.items():
        for kw in sets:
            y = FX.REGISTRY[name]["fn"](x, sr=22050, **kw)
            assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 4.0, (name, kw, np.max(np.abs(y)))


if __name__ == "__main__":
    import time
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            t = time.time()
            v()
            print(f"ok {k} ({time.time() - t:.1f}s)", getattr(v, "numbers", ""))
