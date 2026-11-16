import numpy as np

from genny import creatures as C, cyber, fx, klang, plates, reverbs, spectral
from genny.physical import ambience, voice


def test_applause_ritard_and_warm():
    p0 = C.applause_plan(30, 0.6, 6.0, seed=1)
    p1 = C.applause_plan(30, 0.6, 6.0, seed=1, ritard=0.05)
    gap = lambda plan: np.median([np.mean(np.diff(p[p > 4])) for p in plan if (p > 4).sum() > 2])
    assert gap(p1) > gap(p0)


def test_footsteps_ar9_correlated():
    g = np.diff(C.footsteps_plan("walk", 105, 300, 1.0, 0, rhythm="ar9"))
    z = g - g.mean()
    assert np.corrcoef(z[:-1], z[1:])[0, 1] > 0.2


def test_cavity_aspect_modes():
    from genny import contact
    f, _, _ = contact.cavity_modes("box", 0.5, 1.0, 2000.0, aspect=(0.5, 0.4))
    assert abs(f[0] - 350.0) < 1e-6 and abs(f[1] - 700.0) < 1e-6


def test_wood_panel_species_and_loop_and_distance():
    for sp in ("spruce", "beech"):
        assert np.isfinite(plates.wood_panel(species=sp)).all()
    sr = 48000
    x = np.sin(2 * np.pi * 50 * np.arange(5 * sr) / sr + 0.3)
    assert len(fx.loop(x, sr, 0.05, period=0.02, cycles=100, preroll=1.0)) == 100 * 0.02 * sr
    y = np.random.default_rng(0).normal(size=sr)
    r = lambda c, m: np.sqrt(np.mean(reverbs.distance(y, sr, m, air=False, curve=c, max_dist=100) ** 2))
    assert r("linear", 100) < 1e-6 and abs(r("linear", 50) / r("linear", 1) - 0.5) < 0.01


def test_nlp_duty_and_thunder_and_icon_and_iir():
    sh = []
    for sd in range(6):
        _, a = voice.roughness(np.full(480000, 150.0), np.random.default_rng(sd), shimmer=0.0, jitter=0.0, subharmonic=0.5, duty="healthy")
        sh.append(np.mean(np.abs(a - 1) > 0.1))
    assert 0.015 < np.mean(sh) < 0.05
    assert np.isfinite(ambience.thunder_channel(np.random.default_rng(1), horizontal=2000)).all()
    for ev in klang.AUDITORY_EVENTS:
        assert np.isfinite(klang.auditory_icon(event=ev)).all()
    b, a = spectral.fit_min_phase_iir([0, 200, 1000, 4000, 12000, 24000], [0, 0, -6, -20, -30, -40], 48000, 6)
    from scipy.signal import freqz
    _, H = freqz(b, a, worN=[1000, 4000], fs=48000)
    assert abs(20 * np.log10(abs(H[0])) + 6) < 1.0 and np.abs(np.roots(a)).max() < 1
    assert cyber.throttle_rpm(np.ones(48000), 800, 6000)[-1] > 800
