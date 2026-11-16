import numpy as np

from genny import contact, cyber, reverbs, spectral
from genny.physical import bank_piano, voice, vkmodal


def test_piano_horizontal_coupling_is_adjustable_and_guarded():
    sr = 22050
    late = []
    for hy in (0.1, 0.35, 0.5):
        y = bank_piano.note(220.0, 1.0, sr, 0.8, polarisations=2, horiz_y=hy)
        r = lambda a, b: np.sqrt(np.mean(y[int(a * sr):int(b * sr)] ** 2))
        late.append(r(1.4, 1.6) / r(0.3, 0.5))
    assert late[0] > late[1] > late[2]
    y = bank_piano.note(220.0, 0.5, sr, 0.8, polarisations=2, horiz_cents=50.0)
    assert np.isfinite(y).all()
    d = bank_piano.note(220.0, 0.5, sr, 0.8, polarisations=2)
    e = bank_piano.note(220.0, 0.5, sr, 0.8, polarisations=2, horiz_y=bank_piano.HORIZ_Y, horiz_cents=bank_piano.HORIZ_CENTS)
    assert np.array_equal(d, e)


def test_roughness_rough_duty_is_18_percent():
    sh = []
    for sd in range(4):
        _, a = voice.roughness(np.full(48000 * 60, 150.0), np.random.default_rng(sd), shimmer=0.0, jitter=0.0,
                               subharmonic=0.5, duty="rough")
        sh.append(np.mean(np.abs(a - 1) > 0.25))
    assert abs(np.mean(sh) - 0.18) < 0.02


def test_nlp_chain_sequences_phenomena_with_jump():
    n = 48000 * 20
    f0 = np.full(n, 200.0)
    f2, amp = voice.roughness(f0, np.random.default_rng(2), chain=True, duty=0.4, nlp={"p_chaos": 1.0, "p_bp": 1.0})
    assert np.isfinite(f2).all() and np.isfinite(amp).all() and amp.min() >= 0.0
    assert np.std(f2) > np.std(voice.roughness(f0, np.random.default_rng(2))[0])
    eps = voice._episodes(n, np.random.default_rng(2), 1.0, 0.4)
    assert eps and abs(sum(b - a for a, b in eps) / 20.0 - 0.4) < 0.02
    plain = voice.roughness(f0, np.random.default_rng(5), subharmonic=0.3)
    again = voice.roughness(f0, np.random.default_rng(5), subharmonic=0.3)
    assert np.array_equal(plain[1], again[1])
    fj, _ = voice.roughness(f0, np.random.default_rng(3), chain=True, jump=7.0, bridge=0.04)
    assert fj.max() > 200.0 * 2 ** (6.5 / 12) and np.isfinite(fj).all()


def test_vk_explicit_second_order_scheme():
    m = vkmodal.build(modes=12, airy=6, grid=20)
    f = np.zeros(4000)
    f[:20] = 50.0
    a = vkmodal.simulate(m, f, 8000.0, scheme="sav")
    b = vkmodal.simulate(m, f, 8000.0, scheme="explicit")
    assert np.isfinite(b).all()
    assert abs(np.abs(b).max() / np.abs(a).max() - 1.0) < 0.05
    f2 = np.zeros(8000)
    f2[:5] = 1e-3
    lin = vkmodal.simulate(m, f2, 8000.0, scheme="linear")
    ex = vkmodal.simulate(m, f2, 8000.0, scheme="explicit")
    assert np.max(np.abs(ex - lin)) < 1e-3 * np.max(np.abs(lin))


def test_conan_series_arma_second_order_and_free_correlation():
    ac = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1]
    a0, d0 = contact.conan_series(48000 * 20, 48000, (0, 0, 1), 0.5, 3)
    a2, d2 = contact.conan_series(48000 * 20, 48000, (0, 0, 1), 0.5, 3, ar2=0.5)
    assert ac(a2, 1) < ac(a0, 1) - 0.1 and abs(np.std(a2) / np.std(a0) - 1.0) < 0.35
    af, _ = contact.conan_series(48000 * 20, 48000, (1, 0, 0), 0.5, 3, ar_free=0.6)
    aw, _ = contact.conan_series(48000 * 20, 48000, (1, 0, 0), 0.5, 3)
    assert ac(af, 1) > 0.4 > abs(ac(aw, 1))
    assert np.array_equal(contact.conan_force(2000, 48000, (0.3, 0.3, 0.4), seed=1),
                          contact.conan_force(2000, 48000, (0.3, 0.3, 0.4), seed=1, scratch_dt=contact.CONAN_SCRATCH_DT))


def test_pipe_engine_intake_per_cylinder():
    a = cyber.pipe_engine(48000, 0.5, 2400.0, seed=1)
    b = cyber.pipe_engine(48000, 0.5, 2400.0, seed=1, intakes=[0.2] * 4)
    c = cyber.pipe_engine(48000, 0.5, 2400.0, seed=1, intakes=[0.1, 0.3, 0.5, 0.8])
    assert np.array_equal(a, b) and np.max(np.abs(a - c)) > 1e-3 and np.isfinite(c).all()
    up, down = cyber.throttle_lags()
    assert 0.3 < up < 1.0 and 2.5 < down < 4.0
    y = cyber.pipe_engine(48000, 1.0, 1000.0, throttle=[0, 1, 1, 0], lag_up=-1, lag_down=-1)
    assert np.isfinite(y).all()


def test_fallaway_and_distance_curve_with_flat_room():
    sr = 48000
    x = np.random.default_rng(0).standard_normal(2 * sr) * np.sin(np.linspace(0, 400, 2 * sr))
    y = spectral.fallaway(x, sr, 0.5, 1.5, 12.0, 2.0, -40.0)
    rms = lambda a, b: np.sqrt(np.mean(y[int(a * sr):int(b * sr)] ** 2))
    assert len(y) == 2 * sr and np.isfinite(y).all() and rms(1.9, 2.0) < 0.05 * rms(0.6, 0.7)
    g = lambda c, m: float(reverbs.distance(np.ones(4800), sr, metres=m, air=False, flat_height=3.0, curve=c, max_dist=100.0)[2000])
    assert g("linear", 100.0) < 1e-6 and abs(g("linear", 50.0) - 0.5051) < 0.01
    assert g("inverse_tapered", 50.0) < reverbs.distance(np.ones(4800), sr, metres=50.0, air=False, flat_height=3.0)[2000]
