"""Physical checks for genny.contact: analytic mode frequencies, strike-position gains, the
internal-friction decay law, Hunt-Crossley contact (rebound, contact time, restitution, hardness and
velocity -> brightness), bounce / crumple statistics, fractal noise slope, rolling filter, cavity
modes, and a smoke loop over every registered name.

Run:  uv run --with pytest pytest tests/test_contact.py -q     (plain asserts)
"""
import math
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import hilbert, welch

from genny import contact as C
from genny.fx import REGISTRY as FX
from genny.sfx import REGISTRY as SFX

SR = 44100
SFX_NAMES = ("impact", "bounce", "drop", "smash", "crumple", "roll", "scrape", "rumble", "avalanche", "modal_engine")
FX_NAMES = ("resonate", "cavity")


def spectrum(y, sr):
    n = 1 << int(np.ceil(np.log2(len(y)))) + 1
    return np.fft.rfftfreq(n, 1 / sr), np.abs(np.fft.rfft(y * np.hanning(len(y)), n))


def level(y, sr, f, rel=0.015):
    fr, s = spectrum(y, sr)
    return float(s[(fr > f * (1 - rel)) & (fr < f * (1 + rel))].max())


def peak_near(y, sr, f, rel=0.03):
    fr, s = spectrum(y, sr)
    m = (fr > f * (1 - rel)) & (fr < f * (1 + rel))
    return float(fr[m][np.argmax(s[m])])


def centroid(y, sr):
    fr, s = spectrum(y, sr)
    return float((fr * s * s).sum() / (s * s).sum())


def t60_of_mode(y, sr, f, half=0.12):
    """T60 of one partial: isolate it in the spectrum, fit the slope of its log envelope."""
    spec = np.fft.rfft(y)
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    spec[(fr < f * (1 - half)) | (fr > f * (1 + half))] = 0
    env = np.abs(hilbert(np.fft.irfft(spec, len(y))))
    db = 20 * np.log10(env / env.max() + 1e-12)
    i0 = int(np.argmax(db > -5))
    sel = np.arange(i0, len(db))[(db[i0:] < -5) & (db[i0:] > -35)]
    sel = sel[sel < np.argmax(db < -35) + 1] if np.any(db < -35) else sel
    slope = np.polyfit(sel / sr, db[sel], 1)[0]
    return -60.0 / slope


def onsets(y, sr, min_gap=0.02):
    """Times of impact onsets: rises of the 1 ms energy envelope."""
    hop = max(1, sr // 2000)
    e = np.sqrt(np.convolve(y * y, np.ones(hop) / hop, "same"))[::hop]
    d = np.diff(e, prepend=0.0)
    thr = 0.12 * d.max()
    out, last = [], -1e9
    for i in np.nonzero(d > thr)[0]:
        if i * hop / sr - last > min_gap:
            out.append(i * hop / sr)
        last = i * hop / sr
    return np.array(out)


# ------------------------------------------------------------------------------------------
# ShapeBody
# ------------------------------------------------------------------------------------------

def test_shape_frequencies_match_analytic_ratios():
    r = lambda b: b.freqs / b.freqs[0]
    assert np.allclose(r(C.ShapeBody("string", 100, n_modes=6)), [1, 2, 3, 4, 5, 6])
    assert np.allclose(r(C.ShapeBody("bar", 100, n_modes=5)), [1, 2.756, 5.404, 8.933, 13.345], atol=2e-3)
    assert np.allclose(r(C.ShapeBody("bar", 100, boundary="clamped_clamped", n_modes=3)), [1, 2.756, 5.404], atol=2e-3)
    assert np.allclose(r(C.ShapeBody("bar", 100, boundary="clamped", n_modes=5)), [1, 6.267, 17.547, 34.386, 56.843], atol=5e-3)
    assert np.allclose(r(C.ShapeBody("bar", 100, boundary="supported", n_modes=4)), [1, 4, 9, 16])
    assert np.allclose(r(C.ShapeBody("circ_membrane", 100, n_modes=9)),
                       [1, 1.593, 2.136, 2.295, 2.653, 2.917, 3.155, 3.500, 3.598], atol=1.5e-3)
    assert np.allclose(r(C.ShapeBody("circ_plate", 100, n_modes=9)),
                       [1, 2.08, 3.41, 3.89, 5.00, 5.95, 6.82, 8.28, 8.72], atol=1e-2)
    for aspect in (1.0, 1.5, 7.0528):
        grid = sorted((m * m + aspect ** 2 * n * n) / (1 + aspect ** 2) for m in range(1, 13) for n in range(1, 13))
        assert np.allclose(r(C.ShapeBody("rect_plate", 50, aspect=aspect, n_modes=12)), grid[:12])
        assert np.allclose(r(C.ShapeBody("rect_membrane", 50, aspect=aspect, n_modes=12)), np.sqrt(grid[:12]))


def test_truncation_and_ranking():
    b = C.ShapeBody("rect_plate", 200, aspect=1.3, n_modes=40, fmax=3000.0, position=(0.3, 0.4))
    assert len(b.freqs) <= 40 and b.freqs.max() <= 3000.0 and np.all(np.diff(b.freqs) >= 0)
    order = b.rank()
    e = (b.gains(bright=0.5) * b.freqs * C._ear_weight(b.freqs)) ** 2
    assert np.all(np.diff(e[order]) <= 1e-30)                       # most important first
    p = b.pruned(keep=8)
    assert len(p.freqs) == 8 and set(p.freqs) == set(b.freqs[order[:8]])
    assert np.allclose(p.row((0.3, 0.4)), p.phi)                    # descriptors pruned with the modes
    q = b.pruned(floor_db=20)
    assert 0 < len(q.freqs) < len(b.freqs)


def test_rendered_partials_sit_on_the_analytic_frequencies():
    for sr in (22050, 44100):
        b = C.ShapeBody("bar", 300.0, material="steel", n_modes=4, position=0.12, scatter=0.0)
        y = C.ImpactInteractor(b, mass=0.005, k=1e9, mu=0.1).strike(1.0, 0.6, sr).audio
        for f in b.freqs:
            assert abs(peak_near(y, sr, f) / f - 1) < 2e-3, (sr, f, peak_near(y, sr, f))


def test_strike_position_sets_the_gains():
    # van den Doel's ideal-string model file (thesis appendix A; g06 B10a): a_n = |sin(n pi p)| / n
    table = {1 / 12: [0.258819, 0.25, 0.235702, 0.216506, 0.193185, 0.166667, 0.137989, 0.108253, 0.078567, 0.05],
             1 / 4: [0.707107, 0.5, 0.235702, 0.0, 0.141421, 0.166667, 0.101015, 0.0, 0.078567, 0.1],
             5 / 12: [0.965926, 0.25, 0.235702, 0.216506, 0.051764, 0.166667, 0.036974, 0.108253, 0.078567, 0.05]}
    for p, a in table.items():
        g = np.abs(C.ShapeBody("string", 440, n_modes=10, position=p).gains())
        assert np.allclose(g / g[0], np.array(a) / a[0], atol=2e-6)
    # a circular membrane or plate struck at the centre contains only the m = 0 modes
    for shape in ("circ_membrane", "circ_plate"):
        b = C.ShapeBody(shape, 100, n_modes=12, position=0.0)
        assert np.all((np.abs(b.phi) > 1e-9) == (b._desc["m"] == 0))
    # a rectangle struck at the centre has only odd-odd modes
    b = C.ShapeBody("rect_plate", 100, aspect=1.4, n_modes=20, position=(0.5, 0.5))
    odd = (b._desc["m"] % 2 == 1) & (b._desc["n"] % 2 == 1)
    assert np.all((np.abs(b.phi) > 1e-9) == odd)
    # near the boundary the high modes gain on the low ones (dull centre, bright rim)
    mem = lambda p: C.ShapeBody("rect_membrane", 500, n_modes=60, position=p).gains()
    hi_share = lambda g: np.sum(g[20:] ** 2) / np.sum(g ** 2)
    assert hi_share(mem((0.1, 0.1))) > 2 * hi_share(mem((0.5, 0.4)))


def test_node_at_strike_position_kills_that_mode():
    # string struck at 1/4: no 4th harmonic in the sound
    b = C.ShapeBody("string", 220.0, n_modes=6, position=0.25, scatter=0.0)
    y = C.ImpactInteractor(b, mass=0.002, k=1e9).strike(1.0, 0.5, SR).audio
    assert level(y, SR, 880.0) < 1e-4 * level(y, SR, 660.0)
    # free bar struck on the node of its first mode: the fundamental is gone, the others are not
    free = C.ShapeBody("bar", 400.0, n_modes=4, position=0.5, scatter=0.0)
    node = brentq(lambda x: free.row(x)[0], 0.15, 0.3)
    assert abs(node - 0.2242) < 1e-3
    at_node = C.ShapeBody("bar", 400.0, n_modes=4, position=node, scatter=0.0)
    y0 = C.ImpactInteractor(at_node, mass=0.002, k=1e9).strike(1.0, 0.5, SR).audio
    y1 = C.ImpactInteractor(free, mass=0.002, k=1e9).strike(1.0, 0.5, SR).audio
    assert level(y0, SR, 400.0) < 1e-4 * level(y0, SR, free.freqs[1])
    assert level(y1, SR, 400.0) > 10 * level(y1, SR, free.freqs[1])          # centre: mode 2 is antisymmetric
    assert level(y1, SR, free.freqs[1]) < 1e-4 * level(y1, SR, 400.0)


# ------------------------------------------------------------------------------------------
# Material law
# ------------------------------------------------------------------------------------------

def test_material_table_anchors():
    q = {k: m.q0 for k, m in C.MATERIALS.items()}
    assert 5 <= q["rubber"] <= 20 and 20 <= q["wood"] <= 150 and 150 <= q["glass"] <= 1500        # g04 §2.3
    assert 1500 <= q["steel"] <= 5000 and abs(q["aluminium"] - 5000) < 10
    assert q["rubber"] < q["wood"] < q["glass"] < q["steel"] < q["aluminium"]
    assert C.MATERIALS["wood"].fd == 35 and C.MATERIALS["metal"].fd == 477 and C.MATERIALS["steel"].fd == 1005
    assert C.MATERIALS["rubber"].k == 5e6 and C.MATERIALS["steel"].k == 1e10
    # Rayleigh anchors reproduce the decay rates derived in g07 §4.2
    ray = lambda m, f: float(C.material_damping([f], m, law="rayleigh")[0])
    assert abs(ray("wood", 1000) - 62) < 1 and abs(ray("plastic", 3000) - 182) < 2
    assert abs(ray("metal", 10000) - 44.9) < 0.5 and abs(ray("glass", 3000) - 34.6) < 0.5
    assert abs(ray("ceramic", 3000) - 15.0) < 0.2


def test_decay_proportional_to_frequency_under_the_phi_law():
    f = np.array([200.0, 800.0, 3200.0])
    d = C.material_damping(f, "wood", scatter=0.0, ext=0.0)
    assert np.allclose(d, math.pi * f * C.MATERIALS["wood"].tan_phi) and np.allclose(f / d, 35.0)
    assert np.allclose(C.material_damping(f, "wood", scatter=0.0, ext=7.0) - d, 7.0)
    # per-mode scatter: f/d spread like the measured objects (g06 B1), reproducible by seed
    fd = 1000.0 / C.material_damping(np.full(4000, 1000.0), "wood", ext=0.0, seed=3)
    assert 0.5 < np.std(fd) / np.mean(fd) < 0.76 and abs(np.mean(fd) / 35.0 - 1) < 0.05      # 35 +- 22 measured
    # measured: T60 of two modes of a rendered body, at two sample rates
    for sr in (44100, 22050):
        b = C.ShapeBody("bar", 400.0, material="wood", n_modes=2, position=0.1, scatter=0.0, ext=0.0)
        imp = np.zeros(int(1.6 * sr))
        imp[0] = 1.0
        y = C.modal_bank(imp, b, sr)[0]
        t = [t60_of_mode(y, sr, fm) for fm in b.freqs]
        want = 6.91 * 35.0 / b.freqs
        print(f"T60 sr={sr}: measured {t[0]:.4f}, {t[1]:.4f} s; law {want[0]:.4f}, {want[1]:.4f} s")
        assert np.allclose(t, want, rtol=0.03)
        assert abs((t[0] / t[1]) / (b.freqs[1] / b.freqs[0]) - 1) < 0.03


def test_bank_exposes_contact_displacement_and_velocity():
    b = C.ShapeBody("bar", 500.0, n_modes=3, position=0.3, scatter=0.0)
    imp = np.zeros(4000)
    imp[0] = 1.0
    _, u, ud = C.modal_bank(imp, b, SR)
    assert np.allclose(np.gradient(u, 1 / SR)[50:-50], ud[50:-50], atol=0.02 * np.abs(ud).max())
    # an impulse of 1 N x 1 sample gives the contact point the velocity sum phi^2 / (mass sr)
    assert abs(ud[:3].max() / (np.sum(b.phi ** 2) / (b.mass * SR)) - 1) < 0.1


# ------------------------------------------------------------------------------------------
# Hunt-Crossley impact
# ------------------------------------------------------------------------------------------

def test_contact_time_and_restitution_match_the_formulas():
    for m, k, alpha, mu, v in ((0.05, 1e7, 1.5, 0.3, 1.0), (0.01, 1.5e11, 2.8, 0.6, 2.0), (0.02, 5e6, 1.5, 0.6, 3.0)):
        it = C.ImpactInteractor(None, mass=m, k=k, alpha=alpha, mu=mu)
        r = it.strike(v, 0.02, SR, oversample=8)
        t0, e0 = it.contact_time(v), -C.hc_rebound(mu, v) / v
        print(f"m={m} k={k:g} a={alpha} mu={mu} v={v}: contact {r.contact_time * 1e3:.4f} ms (eq. 8.27: {t0 * 1e3:.4f}), "
              f"restitution {r.restitution:.4f} (phase plane: {e0:.4f}), peak force {r.force.max():.1f} N")
        assert abs(r.contact_time / t0 - 1) < 0.01
        assert abs(r.restitution - e0) < 2e-3 and r.restitution < 1.0
        assert len(r.contacts()) == 1 and r.force.min() >= 0.0
    # the book's reference set (Fig. 8.3): m = 10 g, k = 1.5e11, alpha = 2.8, mu = 0.6 -> 50-300 N for 1-4 m/s
    ref = C.ImpactInteractor(None, mass=1e-2, k=1.5e11, alpha=2.8, mu=0.6)
    assert 40 < ref.strike(1.0, 0.01, SR, oversample=8).force.max() < ref.strike(4.0, 0.01, SR, oversample=8).force.max() < 320
    # limits: e ~ 1 - (2/3) mu v for soft hits, |v_out| -> 1/mu for hard ones
    assert abs(-C.hc_rebound(0.3, 0.1) / 0.1 - (1 - 2 / 3 * 0.03)) < 1e-3
    assert 0.9 / 0.6 < -C.hc_rebound(0.6, 1e3) <= 1 / 0.6 + 1e-12
    # Hertz scaling for mu -> 0: t0 ~ (m/k)^(1/(a+1)) v^(-(a-1)/(a+1))
    t = lambda m, k, v: C.hc_contact_time(m, k, 1.5, 1e-9, v)
    assert abs(t(0.01, 1e8, 32.0) / t(0.01, 1e8, 1.0) - 32 ** -0.2) < 1e-3
    assert abs(t(0.32, 1e8, 1.0) / t(0.01, 1e8, 1.0) - 32 ** 0.4) < 1e-3
    assert abs(t(0.01, 1e8, 1.0) / (2.9432 * (1.25 * 0.01 / 1e8) ** 0.4) - 1) < 2e-3      # Hertz sphere: 2.94 (5m/4k)^(2/5)


def test_dropped_mass_rebounds_and_settles():
    it = C.ImpactInteractor(None, mass=0.02, k=1e7, mu=0.5)
    r = it.drop(0.05, 1.2, SR)
    c = r.contacts()
    assert len(c) >= 4
    apex = [r.pos[a[1]:b[0]].max() for a, b in zip(c[:-1], c[1:])]
    flight = np.array([(b[0] - a[1]) / r.fs for a, b in zip(c[:-1], c[1:])])
    v_in = math.sqrt(2 * C.G * 0.05)
    e1 = -C.hc_rebound(0.5, v_in) / v_in
    print(f"drop 5 cm: v_in {v_in:.3f} m/s, apex heights {np.round(apex[:4], 4)}, flights {np.round(flight[:4], 4)} s, e1 {e1:.3f}")
    assert 0.0 < apex[0] < 0.05 and np.all(np.diff(apex[:4]) < 0)            # it rebounds, lower each time
    assert np.all(np.diff(flight[:4]) < 0)                                   # accelerating pattern
    assert abs(apex[0] / 0.05 - e1 ** 2) < 0.03                              # restitution < 1, as predicted
    assert abs(flight[0] - 2 * e1 * v_in / C.G) < 0.02 * flight[0]           # t = 2 v_out / g (eq. 9.3)
    # with enough dissipation it ends in permanent contact, the contact force carrying the weight
    # (restitution -> 1 as v -> 0, so a weakly damped contact keeps hopping for a long time)
    r = C.ImpactInteractor(None, mass=0.02, k=1e7, mu=2.0).drop(0.02, 3.0, SR)
    last = r.contacts()[-1]
    print(f"mu=2: {len(r.contacts())} contacts, permanent contact from {last[0] / r.fs:.3f} s")
    assert last[1] == len(r.force) and abs(r.force[-200:].mean() / (0.02 * C.G) - 1) < 1e-6


def test_harder_striker_shorter_contact_and_brighter():
    body = C.ShapeBody.from_size("bar", 0.3, "wood", n_modes=24, position=0.28, seed=1)
    res = {}
    for m in (0.003, 0.02):
        for k in (5e6, 2.24e8, 1e10):
            r = C.ImpactInteractor(body, mass=m, k=k, mu=0.3).strike(1.5, 0.4, SR)
            res[m, k] = (r.contact_time, centroid(r.audio, SR), centroid(r.audio[:SR // 100], SR))
            print(f"striker {m * 1e3:.0f} g, k={k:g}: contact {r.contact_time * 1e3:.3f} ms, centroid {res[m, k][1]:.0f} Hz, "
                  f"first 10 ms {res[m, k][2]:.0f} Hz, {len(r.contacts())} contact(s)")
        assert res[m, 5e6][0] > res[m, 2.24e8][0] > res[m, 1e10][0]             # harder = shorter contact
        assert res[m, 5e6][2] < res[m, 2.24e8][2] < res[m, 1e10][2]             # ... = brighter attack
    # whole-sound centroid: monotonic for a light striker. A 20 g striker on this 70 g bar stays long
    # enough to interact with the low mode and splits the hardest hit into two contacts, so only
    # rubber < hard holds there (the heavy-hammer effect of g04 §4.4).
    assert res[0.003, 5e6][1] < res[0.003, 2.24e8][1] < res[0.003, 1e10][1]
    assert res[0.02, 5e6][1] < min(res[0.02, 2.24e8][1], res[0.02, 1e10][1])
    # the registered sound follows the same rule
    c = [centroid(SFX["impact"]["fn"](SR, hardness=h), SR) for h in (0.0, 0.5, 1.0)]
    assert c[0] < c[1] < c[2], c
    # heavier striker: longer contact, darker (g04 §1.2)
    light = C.ImpactInteractor(body, mass=0.002, k=2.24e8, mu=0.3).strike(1.5, 0.4, SR)
    heavy = C.ImpactInteractor(body, mass=0.2, k=2.24e8, mu=0.3).strike(1.5, 0.4, SR)
    assert heavy.contact_time > light.contact_time and centroid(heavy.audio, SR) < centroid(light.audio, SR)


def test_higher_velocity_is_brighter():
    body = C.ShapeBody.from_size("bar", 0.3, "wood", n_modes=24, position=0.28, seed=1)
    out = []
    for v in (0.25, 1.0, 4.0):
        r = C.ImpactInteractor(body, mass=0.02, k=5e7, mu=0.3).strike(v, 0.4, SR)
        out.append((r.contact_time, centroid(r.audio, SR), np.abs(r.audio).max()))
        print(f"v={v}: contact {out[-1][0] * 1e3:.3f} ms, centroid {out[-1][1]:.0f} Hz, peak {out[-1][2]:.3e}")
    assert out[0][0] > out[1][0] > out[2][0]                                # faster = shorter contact
    assert out[0][1] < out[1][1] < out[2][1]                                # ... = brighter
    assert out[0][2] < out[1][2] < out[2][2]                                # ... and louder
    c = [centroid(SFX["impact"]["fn"](SR, velocity=v, hardness=0.3), SR) for v in (0.3, 1.5, 8.0)]
    assert c[0] < c[1] < c[2], c


def test_restrike_of_a_ringing_body_differs_and_coupling_is_passive():
    body = C.ShapeBody("rect_plate", 180.0, material="steel", aspect=1.3, n_modes=20, position=(0.37, 0.41), mass=0.05, seed=2)
    it = C.ImpactInteractor(body, mass=0.02, k=1e9, mu=0.1)
    r = it.run(0.2, SR, events=([0.0, 0.0331], [1.0, 1.0]))
    c = r.contacts()
    first = [x for x in c if x[0] < 0.0331 * r.fs]
    second = [x for x in c if x[0] >= 0.0331 * r.fs]
    imp = lambda seg: sum(r.force[a:b].sum() for a, b in seg) / r.fs
    print(f"re-strike: {len(first)} + {len(second)} contacts, impulses {imp(first):.5f} / {imp(second):.5f} N s")
    assert abs(imp(first) / imp(second) - 1) > 0.01          # same parameters, different state, different hit
    assert np.all(np.isfinite(r.audio)) and r.restitution <= 1.0 + 1e-9
    # a light, compliant body under a heavy hard striker: several micro-contacts within one hit (g04 §4.4)
    assert len(first) > 1


# ------------------------------------------------------------------------------------------
# Patterns
# ------------------------------------------------------------------------------------------

def test_bounce_intervals_are_geometric():
    for factor in (0.6, 0.74, 0.88):
        t, v = C.bounce_schedule(0.3, 2.0, factor)
        gaps = np.diff(t)
        assert np.allclose(gaps[1:] / gaps[:-1], factor) and np.allclose(v[1:] / v[:-1], factor)
        assert abs(gaps[0] - 0.3) < 1e-12 and gaps[-1] >= 0.004 * factor          # buzz guard
    # irregular objects: only shorter / weaker than the spherical envelope
    t0, v0 = C.bounce_schedule(0.3, 2.0, 0.74)
    t1, v1 = C.bounce_schedule(0.3, 2.0, 0.74, dev_t=0.5, dev_v=0.5, seed=5)
    m = min(len(t0), len(t1)) - 1
    assert np.all(np.diff(t1)[:m] <= np.diff(t0)[:m] + 1e-12) and np.all(v1[:m] <= v0[:m] + 1e-12)
    assert np.std(np.diff(t1)[1:m] / np.diff(t1)[:m - 1]) > 0.02
    # rendered: onset intervals of the registered bounce follow the requested factor
    for factor in (0.6, 0.8):
        y = SFX["bounce"]["fn"](SR, first_interval=0.3, restitution=factor, surface="plastic")   # short ring: clean onsets
        on = onsets(y, SR)
        ratios = np.diff(on)[1:4] / np.diff(on)[:3]
        print(f"bounce restitution {factor}: onset intervals {np.round(np.diff(on)[:4], 4)} s, ratios {np.round(ratios, 3)}")
        assert abs(np.diff(on)[0] - 0.3) < 0.004 and np.allclose(ratios, factor, atol=0.03)


def test_break_schedule_decelerates():
    t, v, tr = C.break_schedule(1.5, trains=1, dev=0.0, seed=1)
    assert np.all(np.diff(np.diff(t)) > 0) and np.all(np.diff(v) < 0)          # intervals grow, hits weaken
    t, v, tr = C.break_schedule(1.5, trains=8, seed=1)
    assert len(np.unique(tr)) == 8 and np.all(np.diff(t) >= 0) and t.max() < 1.5
    early, late = np.sum(t < 0.15), np.sum((t >= 0.75) & (t < 0.9))
    assert early > 4 * max(late, 1)                                            # massive start, quick thinning


def test_crumple_energies_follow_the_power_law():
    assert abs(C.crumple_energy_floor(-1.5) - 0.4444) < 1e-3                   # (9.13), g05 §4.1
    for gamma in (-1.15, -1.5):
        m = C.crumple_energy_floor(gamma)
        assert abs((1 - m ** (gamma + 1)) / (gamma + 1) - 1) < 1e-9             # integral of E^gamma on [m, 1] = 1
        fits = {}
        for e_min in (None, 0.01):
            ev = C.crumple_events(1e9, gamma, 20.0, e_min=e_min, n_max=30000, seed=2)
            E = ev["energy"]
            lo = m if e_min is None else e_min
            assert E.min() >= lo - 1e-12 and E.max() <= 1.0 + 1e-12

            def nll(g):
                g1 = g + 1.0
                return -(len(E) * math.log(abs(g1 / (1.0 - lo ** g1))) + g * np.log(E).sum())
            fits[e_min] = minimize_scalar(nll, bounds=(-2.9, -1.01), method="bounded").x
        print(f"crumple gamma {gamma}: fitted {fits[None]:.3f} on [m, 1], {fits[0.01]:.3f} on [0.01, 1]")
        assert abs(fits[0.01] - gamma) < 0.03 and abs(fits[None] - gamma) < 0.12
    ev = C.crumple_events(0.03, -1.3, 20.0, seed=1)
    n = len(ev["t"])
    assert abs(ev["energy"].sum() * 7.5e-4 - 0.03) < 7.5e-4                    # stops when the budget is spent
    assert np.all(np.diff(ev["cut"]) <= 0) and ev["cut"][0] <= 1400 and abs(ev["cut"][-1] - 500) < 1e-9
    assert np.all(ev["left"] + ev["right"] <= 1.0 + 1e-12)
    assert (ev["left"] + ev["right"])[: n // 3].mean() > 2 * (ev["left"] + ev["right"])[-n // 3:].mean()   # facets shrink
    gaps = np.diff(C.crumple_events(1e9, -1.3, 20.0, n_max=20000, seed=4)["t"])
    assert abs(gaps.mean() * 20.0 - 1) < 0.03 and abs(gaps.std() / gaps.mean() - 1) < 0.05                 # Poisson


def test_fractal_noise_slope():
    for beta in (0.0, 0.5, 1.0, 1.81):
        x = C.fractal_noise(1 << 18, beta, SR, seed=1)
        f, p = welch(x, SR, nperseg=8192)
        m = (f > 60) & (f < 6000)
        slope = np.polyfit(np.log10(f[m]), 10 * np.log10(p[m]), 1)[0]
        print(f"fractal noise beta {beta}: {slope:.2f} dB/decade (target {-10 * beta:.1f})")
        assert abs(slope + 10 * beta) < 1.0 and abs(x.std() - 1) < 1e-9
    assert np.array_equal(C.fractal_noise(1000, 1.0, seed=3), C.fractal_noise(1000, 1.0, seed=3))


def test_rolling_filter():
    dx = 1e-4
    s = C.rolling_surface(0.5, dx, 0.6, seed=1)
    small, big = C.rolling_offset(s, 0.003, dx), C.rolling_offset(s, 0.05, dx)
    assert np.all(small >= s - 1e-15) and np.all(big >= small - 1e-15)        # the ball rides on the peaks
    rough = lambda z: np.std(np.diff(z, 2))
    assert rough(big) < rough(small) < rough(s)                               # bigger ball = smoother
    # a ball bridging a narrow deep gap of width w dips by r - sqrt(r^2 - (w/2)^2) ~ w^2/(8 r)
    flat = np.zeros(4000)
    flat[2000:2040] = -1e-3
    for r in (0.01, 0.03):
        dip = -C.rolling_offset(flat, r, dx).min()
        want = r - math.sqrt(r * r - 0.002 ** 2)
        assert abs(dip / want - 1) < 0.06, (r, dip, want)
    # all surface frequencies scale with speed
    c = [centroid(SFX["roll"]["fn"](SR, dur=1.0, speed=v, bright=0.0), SR) for v in (0.15, 1.2)]
    print(f"roll centroid at 0.15 / 1.2 m/s: {c[0]:.0f} / {c[1]:.0f} Hz")
    assert c[1] > 1.2 * c[0]


def test_scrape_reads_the_profile_at_speed_times_force():
    prof = C.surface_texture("grid", seed=0)
    f1 = C.scrape_force(prof, np.full(SR, 0.3), np.ones(SR), 1e-4, SR)
    assert np.allclose(C.scrape_force(prof, np.full(SR, 0.3), np.full(SR, 2.5), 1e-4, SR), 2.5 * f1)
    # grid texture: one ridge per 12 grains -> a line at v / (12 grain)
    for v in (0.3, 0.6):
        f = C.scrape_force(prof, np.full(SR, v), np.ones(SR), 1e-4, SR)
        fr, s = spectrum(f - f.mean(), SR)
        assert abs(fr[np.argmax(s)] / (v / (12 * 1e-4)) - 1) < 0.03
    c = [centroid(SFX["scrape"]["fn"](SR, speed=v, texture="white"), SR) for v in (0.05, 0.6)]
    assert c[1] > c[0]
    assert centroid(SFX["scrape"]["fn"](SR, roughness=1.0), SR) > centroid(SFX["scrape"]["fn"](SR, roughness=0.0), SR)


def test_cavity_modes_and_sabine_decay():
    # d = 0.5 m sphere: lowest partials 463.8 Hz and 742 Hz (Sounding Object §6.1; g04 §2.1)
    size = 0.25 / (3 / (4 * math.pi)) ** (1 / 3)
    f, d, _ = C.cavity_modes("sphere", size, 1.0, 5000.0, 10, 0)
    print(f"sphere d=0.5 m: {f[0]:.1f}, {f[1]:.1f} Hz")
    assert abs(f[0] - 463.8) < 0.5 and abs(f[1] / 742.0 - 1) < 0.005
    f, d, _ = C.cavity_modes("box", 0.5, 1.0, 5000.0, 10, 0)
    assert abs(f[0] - 350.0) < 1e-6 and abs(f[1] / f[0] - math.sqrt(2)) < 1e-9          # (1,0,0), (1,1,0)
    assert abs(6.91 / d[0] - 0.163 * 0.125 / (np.interp(350.0, [250, 500], [0.11, 0.10]) * 1.5)) < 1e-9   # Sabine
    # equal volume, sphere has less wall: longer decay (heard as brighter)
    fs_, ds_, _ = C.cavity_modes("sphere", 0.5, 1.0, 5000.0, 10, 0)
    assert np.interp(1000.0, fs_, ds_) < np.interp(1000.0, f, d)
    imp = np.zeros(SR // 2)
    imp[0] = 1.0
    f, d, _ = C.cavity_modes("box", 0.5, 0.2, 5000.0, 10, 0)                   # hard walls: narrow, measurable modes
    imp = np.zeros(2 * SR)
    imp[0] = 1.0
    y = FX["cavity"]["fn"](imp, SR, shape="box", size=0.5, wall=0.2, mix=1.0)
    t = t60_of_mode(y, SR, 350.0, half=0.15)
    print(f"box 0.5 m, wall 0.2: lowest mode {peak_near(y, SR, 350.0):.1f} Hz, T60 {t:.3f} s (Sabine {6.91 / d[0]:.3f} s)")
    assert abs(peak_near(y, SR, 350.0) - 350.0) < 2.0 and abs(t / (6.91 / d[0]) - 1) < 0.05


def test_resonate_effect():
    imp = np.zeros(SR // 2)
    imp[0] = 1.0
    b = C._make_body("bar", "steel", 0.3, 0.3, modes=6)
    y = FX["resonate"]["fn"](imp, SR, body="bar", material="steel", size=0.3, position=0.3, modes=6, tail=1.0)
    assert len(y) == len(imp) + SR
    for f in b.freqs[:3]:
        assert abs(peak_near(y, SR, f) / f - 1) < 2e-3
    dry = FX["resonate"]["fn"](imp, SR, mix=0.0, tail=0.0)
    assert np.allclose(dry, imp)
    assert np.all(FX["resonate"]["fn"](np.zeros(1000), SR) == 0)
    st = FX["resonate"]["fn"](np.stack([imp, 0.5 * imp], axis=1), SR, body="sy_vase", tail=0.2)
    assert st.shape == (len(imp) + int(0.2 * SR), 2) and np.all(np.isfinite(st))


# ------------------------------------------------------------------------------------------
# Everything registered
# ------------------------------------------------------------------------------------------

def sane(y, name):
    assert y.ndim == 1 and np.all(np.isfinite(y)), name
    peak = float(np.max(np.abs(y)))
    assert 1e-4 < peak < 1.5, (name, peak)
    assert abs(y.mean()) < 0.02 * peak, (name, "DC", y.mean())
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (name, y[0], y[-1])


def test_registered_and_documented():
    for n in SFX_NAMES:
        assert n in SFX and SFX[n]["fn"].__module__ == "genny.contact", n
        assert all(isinstance(p, tuple) and len(p) == 2 and p[1] for p in SFX[n]["params"].values()), n
    for n in FX_NAMES:
        assert n in FX and FX[n]["fn"].__module__ == "genny.contact", n


def test_smoke_every_name_at_three_rates():
    for sr in (22050, 44100, 48000):
        for n in SFX_NAMES:
            y = SFX[n]["fn"](sr)
            sane(y, (n, sr))
            assert np.array_equal(y, SFX[n]["fn"](sr)), (n, "not deterministic")
            if "seed" in SFX[n]["params"]:
                z = SFX[n]["fn"](sr, seed=7)
                assert len(z) != len(y) or not np.array_equal(z, y), (n, "seed has no effect")
        x = SFX["impact"]["fn"](sr)
        for n in FX_NAMES:
            y = FX[n]["fn"](x, sr)
            assert y.ndim == 1 and np.all(np.isfinite(y)) and 1e-4 < np.abs(y).max() < 1.5, (n, sr)
            assert abs(y[-1]) < 1e-3, n
            s = FX[n]["fn"](np.stack([x, x], axis=1), sr)
            assert s.shape == (len(y), 2) and np.allclose(s[:, 0], y)


def test_smoke_parameter_ranges():
    sr = 22050
    for m in C.MATERIALS:
        sane(SFX["impact"]["fn"](sr, material=m, dur=0.2), ("impact", m))
        sane(SFX["smash"]["fn"](sr, material=m, dur=0.3), ("smash", m))
    for s in ("string", "bar", "bar_clamped", "bar_supported", "bar_clamped_clamped", "rect_membrane", "rect_plate",
              "circ_membrane", "circ_plate", "sy_vase", "sy_wok"):
        for pos in (0.0, 0.5, 1.0):
            sane(SFX["impact"]["fn"](sr, shape=s, position=pos, dur=0.2), ("impact", s, pos))
        sane(SFX["scrape"]["fn"](sr, body=s, dur=0.2), ("scrape", s))
    for kw in ({"size": 0.01, "velocity": 30, "hardness": 1, "striker_mass": 1e-4}, {"size": 5, "striker_mass": 50, "hardness": 0},
               {"hollow": 1.0}, {"bright": 1.0, "damping": 0}, {"bright": 0, "damping": 200, "modes": 1}, {"freq": 6000}):
        sane(SFX["impact"]["fn"](sr, dur=0.2, **kw), ("impact", kw))
    for kw in ({"shape": "circ_plate", "size": 0.15, "irregularity": 1.0}, {"shape": "bar", "size": 0.2, "irregularity": 0.5},
               {"restitution": 0.05}, {"restitution": 0.99, "first_interval": 0.05}, {"material": "rubber", "surface": "concrete"},
               {"surface": "steel", "material": "glass", "size": 0.01, "velocity": 6}):
        sane(SFX["bounce"]["fn"](sr, **kw), ("bounce", kw))
    for kw in ({"roll": 0}, {"height": 2.0, "restitution": 0.9, "roll": 0.3}, {"material": "rubber", "size": 0.08, "irregularity": 1}):
        sane(SFX["drop"]["fn"](sr, **kw), ("drop", kw))
    for kind in ("can", "paper", "bottle", "foil", "bag"):
        for kw in ({}, {"energy": 0, "density": 0, "size": 0}, {"energy": 1, "density": 1, "size": 1, "dur": 1.0}):
            sane(SFX["crumple"]["fn"](sr, kind=kind, **kw), ("crumple", kind, kw))
    for kw in ({"dur": 0.14}, {"speed": [1.5, 0.8, 0.0], "joints": 0.1, "dents": 1.0, "asymmetry": 1.0},
               {"radius": 0.004, "material": "glass", "surface": "glass", "roughness": 1.0}, {"radius": 0.3, "speed": 8, "roughness": 0},
               {"speed": 0.0}):
        y = SFX["roll"]["fn"](sr, **kw)
        if kw != {"speed": 0.0}:
            sane(y, ("roll", kw))
        assert np.all(np.isfinite(y)) and np.abs(y).max() < 1.5
    for tex in sorted(C._TEXTURES):
        sane(SFX["scrape"]["fn"](sr, texture=tex, dur=0.3), ("scrape", tex))
    sane(SFX["scrape"]["fn"](sr, dur=0.14), "scrape short")
    sane(SFX["scrape"]["fn"](sr, speed=[0.0, 2.0, 0.1], force=[0.2, 1.0, 0.0], grain=2.0), "scrape curves")
    sane(SFX["avalanche"]["fn"](sr, dur=1.0, fmax=4000, modes=200), "avalanche")
    sane(SFX["rumble"]["fn"](sr, dur=0.3, rise=5, hold=1, fade=0.1, damping=0), "rumble")
    for kw in ({"cylinders": 4, "rpm": [900, 6000]}, {"rpm": 30000, "body_freq": 400, "damping": 300}, {"fan": 0, "cyl_spread": 0, "cylinders": 2}):
        sane(SFX["modal_engine"]["fn"](sr, dur=0.5, **kw), ("modal_engine", kw))
    x = SFX["impact"]["fn"](sr, dur=0.2)
    for kw in ({"shape": "sphere", "size": 0.05}, {"size": 8.0, "wall": 0.01}, {"wall": 50, "mix": 1.0, "tail": 0}):
        assert np.all(np.isfinite(FX["cavity"]["fn"](x, sr, **kw)))
    for kw in ({"body": "circ_membrane", "material": "rubber"}, {"body": "string", "size": 3.0, "bright": 1.0}, {"modes": 1, "damping": 500}):
        assert np.all(np.isfinite(FX["resonate"]["fn"](x, sr, **kw)))


def test_defaults_are_not_piercing_and_fast():
    from genny.analysis import brightness
    for n in SFX_NAMES:
        SFX[n]["fn"](SR)                                                       # JIT warm-up
        t = time.perf_counter()
        y = SFX[n]["fn"](SR)
        dt = time.perf_counter() - t
        b = brightness(y, SR)
        print(f"{n:13s} {len(y) / SR:5.2f} s rendered in {dt:5.2f} s, centroid {b['centroid']:5.0f} Hz, above 5 kHz {100 * b['above_5k']:4.1f} %")
        assert dt < 2.0 * max(len(y) / SR, 1.0), (n, dt)
        if n not in ("smash", "crumple"):                                      # broadband by nature, like cymbals
            assert b["above_5k"] < 0.05, (n, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
