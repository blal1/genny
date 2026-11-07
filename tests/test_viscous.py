"""Gore, mud, slime: the laws the module claims from its sources, measured on the signal, and every variant renders without a click."""
import numpy as np
import pytest

from genny import viscous as V
from genny.analysis import detect_clicks
from genny.sfx import REGISTRY

SR = 48000
VARIANTS = [(V.bone, dict(kind=k)) for k in ("craquement", "fracture", "os casse humide")] + [(V.slime, {})] \
    + [(V.gore, dict(kind=k)) for k in V.GORE_KINDS] + [(V.mud, dict(kind=k)) for k in V.MUD_KINDS] \
    + [(V.blood, dict(kind=k)) for k in V.BLOOD_KINDS] + [(V.chem, dict(kind=k)) for k in V.CHEM_KINDS] + [(V.burn_flesh, {})]


def _peak(y, lo, hi):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)), 8 * len(y)))
    f = np.fft.rfftfreq(8 * len(y), 1 / SR)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def _share(y, lo, hi):
    s = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(s[(f >= lo) & (f <= hi)].sum() / s.sum())


def _centroid(y):
    s = np.abs(np.fft.rfft(y)) ** 2
    return float(np.sum(np.fft.rfftfreq(len(y), 1 / SR) * s) / np.sum(s))


def _env(y, ms=5.0):
    k = max(1, int(ms * 1e-3 * SR))
    return np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))


def test_seventeen_variants_and_names_registered():
    assert len(VARIANTS) == 17
    assert {"bone", "slime", "gore", "mud", "blood", "chem", "burn_flesh"} <= set(REGISTRY)
    for f in (V.bone, V.gore, V.mud, V.blood, V.chem):
        with pytest.raises(ValueError):
            f(sr=SR, kind="nope")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("f,kw", VARIANTS, ids=lambda v: getattr(v, "__name__", None) or v.get("kind", ""))
def test_every_variant_renders_without_a_click(f, kw, seed):
    y = f(sr=SR, seed=seed, **kw)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
    assert len(detect_clicks(y)) == 0


# generator
def test_radial_line_is_the_damped_frequency_and_falls_with_viscosity_cpl():
    """[CPL]: more viscous, lower frequency and lower amplitude. The line is w_d of HF 36-38 with the liquid's own term."""
    r = 0.01
    seen = []
    for mu in (1.0, 100.0, 400.0):
        y = V._radial(r, mu, V.RHO_MUD, V.SIGMA_MUD)
        # A damped sine's zeros sit exactly at k pi / w_d; its spectral peak does not (sqrt(w_d^2 - beta^2)), which reads 29 % low at mu 400.
        z = np.flatnonzero(np.diff(np.sign(y[1:])) < 0)[0] + 1.5
        fd = V.bubble_modes(r, mu)[2] / (2 * np.pi)
        f = SR / (2 * z)
        assert abs(f - fd) / fd < 0.03, (mu, f, fd)
        seen.append((f, np.abs(y).max()))
    assert seen[0][0] > seen[1][0] > seen[2][0] and seen[0][1] > seen[1][1] > seen[2][1]


def test_inviscid_limit_is_minnaert():
    w0, beta, wd, _ = V.bubble_modes(0.003, 1e-3, 1000.0, 0.0726)
    assert abs(wd / (2 * np.pi) * 0.003 - 3.26) < 0.1 and beta < 0.1 * w0


def test_yield_stress_thickens_it():
    assert V.effective_viscosity(0.01, 1.0, 300.0) > 1.0 + 1.0 and V.effective_viscosity(0.01, 1.0, 0.0) == 1.0


def test_burst_fundamental_falls_linearly_with_size_cpb():
    rs = np.array([0.006, 0.012, 0.018, 0.024])
    fs = np.array([_peak(np.pad(V._burst(r, 1.0), (0, SR)), 60, 2000) for r in rs])     # one viscosity, thin enough that _thin stays at 1
    a, b = np.polyfit(rs, fs, 1)
    assert a < 0 and np.max(np.abs(a * rs + b - fs)) < 0.05 * fs.mean(), fs
    assert all(abs(fs - np.array([V.burst_freq(r) for r in rs])) < 0.2 * fs)


def test_burst_falls_with_viscosity_too_cpl():
    f = lambda mu: _peak(np.pad(V._burst(0.01, mu), (0, SR)), 30, 2000)
    assert f(300.0) < 0.8 * f(1.0)


def test_burst_is_an_n_wave_then_a_jet():
    y = V._burst(0.01, 1.0)
    per = int(SR / V.burst_freq(0.01))
    head = y[:per]
    assert head[:per // 2].max() > 0 and head[per // 2:].min() < 0 and np.argmax(head) < np.argmin(head)   # up, then down
    assert np.std(y[per:]) > 0.1 * np.std(head)


def test_rupture_time_quiets_the_burst_and_grows_with_viscosity():
    assert V.rupture_time(0.01, 100.0) > V.rupture_time(0.01, 1.0)
    assert np.abs(V._burst(0.01, 100.0)).max() < np.abs(V._burst(0.01, 1.0)).max()


def test_helmholtz_pitch_is_the_cavity_law_and_climbs():
    r = 0.02
    y = V._helmholtz(r, V.MU_MUD)
    k = int(V.rupture_time(r, V.MU_MUD) * SR)
    i = np.flatnonzero((y[:-1] <= 0) & (y[1:] > 0))
    zc = i + y[i] / (y[i] - y[i + 1])                  # to the sample's fraction: a period is 37 samples, one sample is 2.7 %
    early = SR / np.diff(zc[zc < k]).mean() if (zc < k).sum() > 1 else None
    late = SR / np.diff(zc[zc >= k + 1]).mean()
    assert abs(late - V.helmholtz_freq(r, V.neck_end(V.MU_MUD) * r)) / late < 0.03
    assert early is None or early < late
    assert V.helmholtz_freq(r, 0.1 * r) < V.helmholtz_freq(r, V.neck_end(V.MU_MUD) * r)


def test_helmholtz_formula():
    r, a = 0.03, 0.006
    want = 343 / (2 * np.pi) * np.sqrt(np.pi * a * a / (2 / 3 * np.pi * r ** 3 * (0.001 + 1.43 * a)))
    assert abs(V.helmholtz_freq(r, a) - want) < 1e-9


def test_viscosity_kills_the_shape_modes():
    rng = np.random.default_rng(0)
    assert np.abs(V._shape(0.01, 1e-3, 1000.0, 0.07, rng)).max() > 0
    assert np.abs(V._shape(0.01, 100.0, V.RHO_MUD, V.SIGMA_MUD, rng)).max() == 0


def test_helmholtz_ring_is_damped_by_radiation_divoux():
    # Q = 4 L^2 / (pi Phi^2): n = n0 + 8 L^2 / (2 pi^2 Phi^2) periods; a narrow opening (thick liquid) rings longer
    r, mu = 0.03, 300.0
    q = V.radiation_q(r, 2 * V.neck_end(mu) * r)
    assert abs(q - 4 / (np.pi * (2 * V.neck_end(mu)) ** 2)) < 1e-12
    y = V._helmholtz(r, mu)
    f = V.helmholtz_freq(r, V.neck_end(mu) * r)
    from scipy.signal import hilbert
    env = np.abs(hilbert(y))
    k = int(V.rupture_time(r, mu) * SR)
    a, b = k + int(0.005 * SR), k + int(0.02 * SR)
    tau = (b - a) / SR / np.log(env[a] / env[b])
    assert abs(tau / (q / (np.pi * f)) - 1) < 0.1
    assert V.radiation_q(r, 2 * V.neck_end(300.0) * r) > V.radiation_q(r, 2 * V.neck_end(0.5) * r)


def test_sfx_band_of_sus():
    y = V.mud(sr=SR, kind="bulles", dur=4.0)
    assert _share(y, 10, 10000) > 0.98      # [SUS] 10-10^4 Hz


# sinking, Johnson 6.66
def test_sink_law_is_6_66_cube_of_radius_linear_in_time():
    ts = np.array([0.01, 0.02, 0.08])
    a = V.sink_radius(ts, 800.0, 0.15, 5000.0)
    assert np.allclose(a ** 3 / ts, 3 * 0.15 * 800 / (16 * 5000))
    assert abs(V.sink_time(float(a[1]), 800.0, 0.15, 5000.0) - 0.02) < 1e-9


def test_floc_envelope_rises_as_cube_root_and_lasts_sink_time():
    T = 0.25
    rng = np.random.default_rng(1)
    y = np.mean([V._floc(T, rng) ** 2 for _ in range(60)], axis=0) ** 0.5
    k = int(0.05 * SR)
    a, b = np.sqrt(np.mean(y[k - 480:k + 480] ** 2)), np.sqrt(np.mean(y[8 * k // 5 * 2 - 480:8 * k // 5 * 2 + 480] ** 2))
    # 0.05 s then 0.16 s: (3.2)^(1/3) = 1.47
    assert abs(b / a - 3.2 ** (1 / 3)) < 0.2, b / a
    assert abs(np.argmax(np.convolve(y, np.ones(960) / 960, "same")) / SR - T) < 0.03


def test_rendered_floc_scales_with_viscosity():
    """The impact's floc length is in proportion to eta."""
    def top(eta):
        y = V.mud(sr=SR, kind="impact", eta=eta, mu=1e3, tau_y=0.0, seed=2)
        e = _env(V._band(y, 30, 350), 40)
        return np.flatnonzero(e > 0.5 * e.max())[-1] / SR      # t^(1/3) is flat near T, so the top is noise: the fall after T is what shows the length
    t1, t2 = top(10000.0), top(40000.0)
    want = lambda eta: V.sink_time(V.BODY_A, V.BODY_LOAD, V.BODY_R, eta)
    # release e^(-t / (.25 T + 10 ms)): half level .17 T + 7 ms after T
    assert abs(t2 - (1.17 * want(40000.0) + 0.007)) / t2 < 0.15, (t1, t2)
    assert abs(t1 - (1.17 * want(10000.0) + 0.007)) / t1 < 0.25, (t1, t2)


def test_marche_steps_fall_at_the_pace():
    y = V.mud(sr=SR, kind="marche", steps=4, pace=1.5, seed=0)      # each suck-out is over by .93 s: a quiet gap before every step
    e = _env(V._band(y, 40, 500), 10)
    for i in range(1, 4):
        t = 1.5 * i
        assert e[int((t - 0.07) * SR):int((t + 0.1) * SR)].max() > 3 * e[int((t - 0.35) * SR):int((t - 0.12) * SR)].max()


def test_aspiration_climbs():
    y = V.mud(sr=SR, kind="aspiration", pull=0.5, seed=1)
    assert _centroid(y[int(0.05 * SR):int(0.15 * SR)]) < _centroid(y[int(0.35 * SR):int(0.48 * SR)])


# the others
def test_bone_pulses_close_together_on_the_stick_under_10k():
    y = V.bone(sr=SR, kind="fracture")
    e = _env(y, 1)
    pk = np.flatnonzero((e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & (e[1:-1] > 0.15 * e.max()))
    assert len(pk) >= 3 and np.median(np.diff(pk)) < 0.05 * SR
    assert _share(y, 0, 10000) > 0.9   # [EGU] up to 10 kHz, rock and wood, not bone


def test_wet_bone_has_more_low_than_dry():
    assert _share(V.bone(sr=SR, kind="os casse humide"), 40, 800) > 1.5 * _share(V.bone(sr=SR, kind="fracture"), 40, 800)


def test_slime_is_higher_than_mud_and_has_strands():
    assert _centroid(V.slime(sr=SR, strands=0.0, seed=1)) > 1.5 * _centroid(V.mud(sr=SR, kind="bulles", dur=3.0, seed=1))
    assert _share(V.slime(sr=SR, seed=1), 900, 3200) > _share(V.slime(sr=SR, strands=0.0, seed=1), 900, 3200)


def test_strand_climbs():
    y = V._strand(np.random.default_rng(3))
    zc = np.flatnonzero(np.diff(np.sign(y)) > 0)
    assert np.diff(zc[:5]).mean() > 1.5 * np.diff(zc[-30:-5]).mean()


def test_blood_drops_are_slower_than_water():
    assert V.BLOOD_DRIP < 2.0
    y = V.blood(sr=SR, kind="goutte", dur=8.0, seed=1)
    e = _env(y, 10)
    above = e > 0.1 * e.max()
    assert 3 <= np.sum(np.diff(above.astype(int)) == 1) <= 40


def test_blood_stream_pulses_with_the_heart():
    y = V.blood(sr=SR, kind="flot", dur=6.0, seed=1)
    e = _env(y, 30)
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    T = 1 / V.PULSE
    a, b = int(0.7 * T * SR), int(1.3 * T * SR)
    assert abs((a + np.argmax(ac[a:b])) / SR - T) < 0.03


def test_acid_dies_and_reaction_gathers():
    a = V.chem(sr=SR, kind="acide", dur=4.0)
    r = V.chem(sr=SR, kind="reaction", dur=4.0)
    assert np.std(a[:SR]) > 1.5 * np.std(a[-2 * SR:-SR])
    assert np.std(r[-2 * SR:-SR]) > 1.5 * np.std(r[SR // 4:SR])


def test_burn_flesh_is_sizzle_with_slow_bubbles():
    y = V.burn_flesh(sr=SR, seed=1)
    from genny.matter import sizzle
    assert _share(y, 200, 1500) > 2 * _share(sizzle(sr=SR, dur=3.0, intensity=0.7, spit=4.0, hiss=0.5, seed=1), 200, 1500)
    assert _share(y, 2500, 16000) > 0.2


def test_gore_kinds_differ_as_described():
    sp, oz = V.gore(sr=SR, kind="splat"), V.gore(sr=SR, kind="ooze")
    k = SR // 20
    assert np.max(np.abs(sp[:k])) > 2 * np.median(_env(sp, 10)[:SR // 2] + 1e-9)       # a blow at the head
    assert len(oz) > 3 * len(sp)
    sq = V.gore(sr=SR, kind="squish")
    e = _env(sq, 40)
    assert e[len(e) // 2 - SR // 8:len(e) // 2 + SR // 8].mean() > 1.5 * e[:SR // 10].mean()      # one squeeze


def test_dislocation_pops_after_its_creep():
    y = V.gore(sr=SR, kind="deboitement", seed=1)
    a = np.abs(y)
    assert a[:int(0.19 * SR)].max() < 0.5 * a[int(0.2 * SR):int(0.26 * SR)].max()
