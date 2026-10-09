"""genny.matter: liquids, gases, fire / electricity, ice. Physical predictions + smoke."""
import numpy as np
from scipy.signal import hilbert, welch

import genny.matter as m
from genny.sfx import REGISTRY, render_sfx

SR = m.SR
NAMES = ["drip", "drops", "splash", "pour", "babble", "bubbles", "boil", "sizzle", "fizz", "gurgle", "drain",
         "surf", "cave_drips", "waterfall", "underwater_ambience", "steam", "air_leak", "spray", "kettle",
         "balloon", "gust", "flame", "fire", "match", "spark", "arc", "mains_hum", "tesla", "neon",
         "lightning", "ice_crack", "ice_cubes", "freeze"]


def _peak_hz(y, sr, lo=0.0, hi=1e9):
    n = len(y)
    N = 1 << int(np.ceil(np.log2(n)) + 2)
    sp = np.abs(np.fft.rfft(y * np.hanning(n), N))
    fr = np.fft.rfftfreq(N, 1.0 / sr)
    sp[(fr < lo) | (fr > hi)] = 0.0
    k = int(np.argmax(sp))
    a, b, c = np.log(sp[k - 1:k + 2] + 1e-300)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / N


def _inst_freq(y):
    return np.diff(np.unwrap(np.angle(hilbert(y)))) * SR / (2 * np.pi)


# ---------------------------------------------------------------- bubbles

def test_bubble_damping_table():
    # findings g07 section 2.2, derived table: r0 mm -> f0 Hz, delta total, beta 1/s
    rows = [(0.10, 32776, 0.0782, 8045), (0.25, 13124, 0.0555, 2289), (0.50, 6564, 0.0437, 901),
            (1.00, 3283, 0.0351, 362), (2.50, 1313, 0.0274, 113), (5.00, 657, 0.0234, 48), (10.0, 328, 0.0206, 21)]
    for r_mm, f0, delta, beta in rows:
        w0, b, wd, d = m.bubble_params(r_mm * 1e-3)
        assert abs(w0 / (2 * np.pi) / f0 - 1) < 0.005
        assert abs(d / delta - 1) < 0.01
        assert abs(b / beta - 1) < 0.02
        assert abs(m._beta_of(float(w0)) / b - 1) < 0.01       # the kernel's damping is the same law
    assert abs(m.G_TH / 1.60e6 - 1) < 0.01


def test_minnaert_frequency():
    for r in (1e-3, 5e-3):
        y = m.hf_bubble(r, depth=1.0)                           # deep: no pitch rise
        expected = np.sqrt(3 * 1.4 * 101325.0 / 1000.0) / (2 * np.pi * r)
        f = _peak_hz(y, SR)
        print(f"bubble r={r * 1e3:.0f} mm: {f:.1f} Hz, Minnaert {expected:.1f} Hz ({100 * (f / expected - 1):+.3f} %)")
        assert abs(f / expected - 1) < 0.02


def test_larger_bubbles_decay_slower():
    rates = []
    for r in (0.5e-3, 1e-3, 2.5e-3, 5e-3):
        y = m.hf_bubble(r, depth=1.0)
        beta = float(m.bubble_params(r)[1])
        env = np.abs(hilbert(y))
        t = np.arange(len(y)) / SR
        sel = (t > 0.5 / beta) & (t < 4.0 / beta)               # after the attack window
        fit = -np.polyfit(t[sel], np.log(env[sel]), 1)[0]
        print(f"r={r * 1e3:.1f} mm: decay {fit:.1f} 1/s, law {beta:.1f} 1/s, T60 {6.91 / fit * 1e3:.1f} ms")
        assert abs(fit / beta - 1) < 0.05
        rates.append(fit)
    assert all(a > b for a, b in zip(rates, rates[1:]))         # larger radius -> slower decay


def test_bubble_pitch_rises_as_it_surfaces():
    r = 2e-3
    f0 = float(m.minnaert(r))
    y = m.hf_bubble(r, depth=0.008, rise=1.0, phi0=-0.016)
    fi = _inst_freq(y)
    early = np.median(fi[int(0.002 * SR):int(0.006 * SR)])
    late = np.median(fi[int(0.016 * SR):int(0.022 * SR)])
    deep = _inst_freq(m.hf_bubble(r, depth=1.0))
    steady = np.median(deep[int(0.016 * SR):int(0.022 * SR)])
    print(f"surfacing 2 mm bubble: {early:.0f} -> {late:.0f} Hz (Minnaert {f0:.0f}); deep stays {steady:.0f} Hz")
    assert late > early * 1.03 and early > f0 * 0.99
    assert late < 3.05 * f0                                      # clamp
    assert abs(steady / f0 - 1) < 0.01


# ---------------------------------------------------------------- vessels, counts

def test_vessel_resonance_rises_with_fill():
    fill = np.linspace(0.0, 1.0, 101)
    cup = m.vessel_resonance(fill, 0.15, 0.035)
    bottle = m.vessel_resonance(fill, 0.25, 0.04, neck=0.01, neck_len=0.06)
    assert np.all(np.diff(cup) > 0) and np.all(np.diff(bottle) > 0)
    assert abs(cup[0] - 343.0 / (4 * (0.15 + 0.61 * 0.035))) < 1e-6      # quarter wave + end correction
    print(f"cup 15 cm: {cup[0]:.0f} -> {cup[50]:.0f} -> {cup[90]:.0f} Hz; bottle: {bottle[0]:.0f} -> {bottle[90]:.0f} Hz")


def test_pour_resonance_audible_and_rising():
    sr = 48000
    y = render_sfx("pour", sr, dur=4.0, fill_start=0.1, fill_end=0.7, resonance=1.0, seed=3)
    got, want = [], []
    for a, b in ((0.8, 1.6), (2.4, 3.2)):
        f, p = welch(y[int(a * sr):int(b * sr)], sr, nperseg=4096)
        fill = 0.1 + 0.6 * (a + b) / 2 / 4.0
        want.append(float(m.vessel_resonance(fill, 0.15, 0.035)))
        band = (f > 0.75 * want[-1]) & (f < 1.33 * want[-1])
        got.append(f[band][np.argmax(p[band])])
        assert p[band].max() > 3.0 * np.median(p[(f > 300) & (f < 3000)])      # a clear peak
    print(f"pour resonance: measured {got[0]:.0f} -> {got[1]:.0f} Hz, formula {want[0]:.0f} -> {want[1]:.0f} Hz")
    assert got[1] > got[0] * 1.15
    assert all(abs(g / w - 1) < 0.12 for g, w in zip(got, want))


def test_bubble_and_drop_counts():
    rng = np.random.default_rng(7)
    for name, rate, lo, hi in (("pour", m.HF_POUR_RATE, m.R_6K, m.R_300), ("babble", m.HF_BABBLE_RATE, m.R_5K, m.R_300)):
        for cluster in (0.0, 0.5):
            _, info = m.bubble_cloud(2.0, rng, rate, lo, hi, "two_band", cluster=cluster, return_info=True)
            f = m.minnaert(info["r"])
            print(f"{name}: {info['n'] / 2.0:.0f} bubbles/s (asked {rate:.0f}), {f.min():.0f}-{f.max():.0f} Hz")
            assert abs(info["n"] / 2.0 / rate - 1) < 0.15
            assert f.min() > 290 and f.max() < 6100
    assert abs(m.HF_POUR_RATE - 7896 / 5.0) < 2 and abs(m.HF_BABBLE_RATE - 26657 / 8.6) < 2
    # rate envelope: the count follows the integral
    env = np.linspace(0.0, 1.0, 3 * SR)
    assert abs(len(m.bubble_times(3 * SR, rng, 2000.0 * env)) / 3000.0 - 1) < 0.15
    for rate, jitter in ((4.0, 0.0), (4.0, 0.15), (10.0, 1.0), (0.5, 0.15)):
        k = np.mean([len(m.drop_times(40.0, np.random.default_rng(s), rate, jitter=jitter)) for s in range(8)])
        assert abs(k / (rate * 40.0) - 1) < 0.15
    per = np.mean([np.clip(rng.poisson(m.HF_DROP_BUBBLES), 1, 9) for _ in range(4000)])
    print(f"bubbles per drop: {per:.2f}")
    assert 4.0 <= per <= 5.0


def test_drops_audio_event_count():
    sr = 48000
    for seed in (0, 1, 2):
        y = render_sfx("drops", sr, dur=10.0, rate=1.0, jitter=0.0, seed=seed)
        env = np.convolve(np.abs(y), np.ones(240) / 240, mode="same")
        on = np.flatnonzero(env > 0.004 * env.max()) / sr
        groups = 1 + int(np.sum(np.diff(on) > 0.45))             # rebound droplets fall within 0.35 s
        print(f"drops rate 1/s for 10 s, seed {seed}: {groups} drops heard")
        assert abs(groups / 10.0 - 1) <= 0.15


# ---------------------------------------------------------------- fire, jets

def test_flame_spectral_slope():
    for alpha in (2.5, 3.0, 3.5):
        y = m.combustion_roar(8 * SR, np.random.default_rng(5), 1.0, 0.5, 60.0, alpha)
        f, p = welch(y, SR, nperseg=8192)
        sel = (f > 300) & (f < 6000)                             # above the 180 Hz hand-over
        slope = np.polyfit(np.log(f[sel]), np.log(p[sel]), 1)[0]
        hi = m._hp(y, 400.0, 2)
        print(f"alpha {alpha}: fitted power slope {slope:.2f}; kurtosis of the extension {np.mean(hi ** 4) / np.mean(hi ** 2) ** 2:.0f}")
        assert abs(slope + alpha) < 0.3
        assert np.max(np.abs(hi)) < 25 * np.std(hi)              # the beta cap keeps windows from spiking


def test_steady_flame_is_quiet():
    rng = np.random.default_rng(1)
    still = m.combustion_roar(2 * SR, rng, 1.0, 0.01, 60.0)
    ruffled = m.combustion_roar(2 * SR, rng, 1.0, 0.8, 60.0)
    assert np.std(ruffled) > 30 * np.std(still)                  # p = dI/dt: loudness follows change


def test_jet_peak_scales_with_velocity():
    d = 0.02
    grid = np.geomspace(60.0, 16000.0, 400)
    peaks, rms = [], []
    for u in (50.0, 100.0, 200.0):
        y = m.jet_noise(8 * SR, np.random.default_rng(2), u, d)
        f, p = welch(y, SR, nperseg=8192)
        lp = np.convolve(np.interp(grid, f, np.log(p + 1e-300)), np.ones(41) / 41, mode="same")
        peaks.append(grid[20:-20][np.argmax(lp[20:-20])])
        rms.append(np.std(y))
        print(f"jet U={u:.0f} m/s, D=20 mm: peak {peaks[-1]:.0f} Hz (St U / D = {0.2 * u / d:.0f}), rms {rms[-1]:.4f}")
        assert abs(peaks[-1] / (0.2 * u / d) - 1) < 0.2
    assert abs(peaks[1] / peaks[0] / 2 - 1) < 0.2 and abs(peaks[2] / peaks[1] / 2 - 1) < 0.2
    # level: power ~ U^8 -> amplitude x16 per doubling
    assert abs(rms[1] / rms[0] / 16 - 1) < 0.15 and abs(rms[2] / rms[1] / 16 - 1) < 0.15
    # and inversely with diameter
    y = m.jet_noise(8 * SR, np.random.default_rng(2), 100.0, 0.04)
    f, p = welch(y, SR, nperseg=8192)
    lp = np.convolve(np.interp(grid, f, np.log(p)), np.ones(41) / 41, mode="same")
    assert abs(grid[20:-20][np.argmax(lp[20:-20])] / 500.0 - 1) < 0.2


# ---------------------------------------------------------------- electricity, ice

def test_mains_hum_fundamental_exact():
    for mains in (50.0, 60.0):
        for sr in (44100, 48000):
            y = render_sfx("mains_hum", sr, mains=mains, dur=10.0)
            f = _peak_hz(y, sr, 20.0, 1.5 * mains)
            print(f"mains_hum {mains:.0f} Hz at {sr}: fundamental {f:.4f} Hz")
            assert abs(f - mains) < 0.02
            # harmonic series: the 2nd harmonic (magnetostriction) is present, slightly split by the beat
            assert abs(_peak_hz(y, sr, 1.8 * mains, 2.2 * mains) - 2 * mains) < 0.5


def test_ice_chirp_frequency_falls():
    y, f = m.ice_chirp(60.0, 0.05, return_freq=True)
    assert np.all(np.diff(f) < 0)
    fi = _inst_freq(y)
    n = len(y)
    meas = [float(np.median(fi[int(a * n):int(a * n) + 300])) for a in (0.03, 0.1, 0.25, 0.5, 0.8)]
    law = [float(f[int(a * n) + 150]) for a in (0.03, 0.1, 0.25, 0.5, 0.8)]
    print("ice chirp (60 m, 5 cm) instantaneous Hz:", [round(v) for v in meas], "law:", [round(v) for v in law])
    assert all(a > b for a, b in zip(meas, meas[1:]))
    assert all(abs(a / b - 1) < 0.1 for a, b in zip(meas, law))
    # f(t) t^2 = d^2 / (8 pi a) is constant along the chirp (flexural dispersion)
    t = 60.0 / m.ICE_C_MAX + np.arange(n) / SR
    assert np.ptp(f * t * t) / np.mean(f * t * t) < 1e-9
    # thin ice sings higher than thick ice; a farther crack gives a longer chirp
    assert m.ice_chirp(60.0, 0.03, return_freq=True)[1][0] > 2 * m.ice_chirp(60.0, 0.3, return_freq=True)[1][0]
    assert len(m.ice_chirp(200.0, 0.05)) > 2 * len(m.ice_chirp(50.0, 0.05))
    # the sfx itself starts high and ends low
    out = render_sfx("ice_crack", 48000, cracks=1, brittle=0.0, dur=0.6)
    k = int(np.flatnonzero(np.abs(out) > 0.02)[0])
    zc = lambda s: np.sum(np.diff(np.signbit(s).astype(int)) != 0) / 2 / (len(s) / 48000)
    assert zc(out[k:k + 480]) > 3 * zc(out[k + 2400:k + 4800])


# ---------------------------------------------------------------- catalog

def test_names_registered_with_help():
    for name in NAMES:
        assert name in REGISTRY, name
        entry = REGISTRY[name]
        assert entry["fn"].__module__ == "genny.matter", name    # pour and fire are replaced
        for k, v in entry["params"].items():
            assert isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], str) and v[1], (name, k)
        assert "seed" in entry["params"], name


def _check(y, sr, tag):
    assert y.ndim == 1 and len(y) > 64, tag
    assert np.all(np.isfinite(y)), tag
    peak = float(np.max(np.abs(y)))
    assert 1e-4 < peak < 1.5, (tag, peak)
    assert abs(float(np.mean(y))) < 0.01, (tag, "dc", float(np.mean(y)))
    assert abs(y[0]) < 1e-6 and abs(y[-1]) < 1e-6, (tag, y[0], y[-1])


def test_every_sfx_two_sample_rates():
    for name in NAMES:
        for sr in (22050, 48000):
            y = render_sfx(name, sr)
            _check(y, sr, (name, sr))
        dflt = REGISTRY[name]["params"].get("dur", (None,))[0]
        if dflt:
            assert abs(len(y) / 48000 - dflt) < 0.02, name
    assert np.array_equal(render_sfx("babble", 44100, dur=1.0), render_sfx("babble", 44100, dur=1.0))
    assert not np.array_equal(render_sfx("fire", 44100, dur=1.0), render_sfx("fire", 44100, dur=1.0, seed=1))


def test_kinds_and_parameter_extremes():
    sr = 44100
    for kind in m.FLAME_KINDS:
        _check(render_sfx("flame", sr, kind=kind, dur=1.5), sr, ("flame", kind))
    for through in m.GUST_SOURCES:
        for strength in (0.3, 0.5, 1.0):
            _check(render_sfx("gust", sr, through=through, dur=2.0, strength=strength), sr, ("gust", through))
    cases = [("pour", dict(neck=0.012, material="metal", fill_start=0.0, fill_end=1.0, flow=3.0, dur=1.5)),
             ("pour", dict(material="paper", flow=0.1, dur=0.3, fill_start=0.9, fill_end=0.1)),
             ("bubbles", dict(underwater=1, rate=0.2, size=12, dur=0.5)),
             ("bubbles", dict(rate=4000, size=0.3, dur=0.5, rise=5.0)),
             ("boil", dict(ramp=1, intensity=1.0, size=1.0, dur=2.0)), ("boil", dict(intensity=0.0, dur=1.0)),
             ("fizz", dict(rate=5000, half_life=0, size=0.3, bright=1.0, dur=0.5)),
             ("drops", dict(rate=40, rate_end=0.05, jitter=1.0, dur=1.0)), ("drops", dict(rate=0.05, dur=0.5)),
             ("air_leak", dict(burst=0.15, pressure=1.0, orifice=30, whistle=1.0)),
             ("air_leak", dict(pressure=0.0, orifice=0.3, whistle=1.0)),
             ("steam", dict(pressure=1.0, orifice=0.5, wet=1.0)), ("steam", dict(pressure=0.0, orifice=50, wet=0.0)),
             ("kettle", dict(tone=4000, onset=0.05, dur=1.0)), ("kettle", dict(tone=600, onset=5.0, dur=1.0, rumble=0)),
             ("fire", dict(size=1.0, crackle=150, wet=1.0, wind=1.0, dur=1.5)),
             ("fire", dict(size=0.0, crackle=0, wet=0.0, wind=0.0, dur=1.0)),
             ("arc", dict(mains=60, stability=1.0)), ("arc", dict(stability=0.0, dur=0.5)),
             ("tesla", dict(rate=600, chaos=1.0, sweep=2.0)), ("neon", dict(flicker=1.0, mains=60)),
             ("neon", dict(flicker=0.0, dur=0.5)), ("lightning", dict(distance=20, dur=1.0)),
             ("lightning", dict(distance=6000, delay=1, dur=2.0)), ("spark", dict(size=1.0, tone=0.5)),
             ("spark", dict(size=0.0, tone=1.5)), ("ice_crack", dict(thickness=1.0, distance=1000, cracks=20)),
             ("ice_crack", dict(thickness=0.01, distance=5, dur=0.2)), ("ice_cubes", dict(cubes=12, liquid=0, dur=0.4)),
             ("freeze", dict(rate=80, creak=1.0, pings=12, dur=1.0)), ("freeze", dict(rate=0.5, creak=0, pings=0, dur=0.5)),
             ("splash", dict(size=1.0, droplets=40)), ("splash", dict(size=0.0, droplets=0, dur=0.3)),
             ("sizzle", dict(intensity=1.0, decay=0.5, spit=60)), ("sizzle", dict(intensity=0.0, spit=0, hiss=0, dur=0.5)),
             ("gurgle", dict(rate=12, size=1.0)), ("drain", dict(size=1.0, dur=1.0)), ("surf", dict(dur=4.0, distance=10)),
             ("cave_drips", dict(dur=0.5, sources=1)), ("waterfall", dict(height=60, distance=200, dur=1.0)),
             ("underwater_ambience", dict(depth=100, bubbles=0, dur=1.0)), ("balloon", dict(size=1.0, flutter=0.0, dur=0.5)),
             ("spray", dict(nozzle=3.0, pressure=0.0, dur=0.2)), ("match", dict(dur=0.3)), ("mains_hum", dict(resonance=0.9, harmonics=1.0, beat=0, dur=1.0)),
             ("babble", dict(flow=3.0, size=2.0, turbulence=1.0, dur=1.0)), ("drip", dict(size=1.0, secondary=1.0, bubbles=9))]
    for name, params in cases:
        _check(render_sfx(name, sr, **params), sr, (name, params))
