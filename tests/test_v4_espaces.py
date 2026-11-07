"""Group V4_ESPACES: spring reverb, rooms, propagation, spatial effects (values in research/SOURCES.md)."""
import math

import numpy as np

from genny import fx as FX
from genny import fdstring as FS

SR = 44100


def _click(n):
    x = np.zeros(n)
    x[0] = 1.0
    return x


# ------------------------------------------------------------------ spring reverb

def test_spring_geometry_eq_18_19_and_unwound_length():
    g = FS.spring_geometry(1.0e-4, 2.5e-3, 100)
    c = math.sqrt(FS.STEEL_E / FS.STEEL_RHO)
    L = math.sqrt((2 * math.pi * 2.5e-3 * 100) ** 2)
    assert abs(g["L"] - L) < 1e-12
    assert abs(g["td"] - 4 * L * 2.5e-3 / (1.0e-4 * c)) < 1e-12
    assert 3500 < g["fc"] < 4100                          # steel, r 0.1 mm, R 2.5 mm: about 3.8 kHz
    assert abs(FS.spring_wire_length(2.5e-3, 10, 0.03) - math.hypot(2 * math.pi * 2.5e-3 * 10, 0.03)) < 1e-12
    for s in FS.SPRING_TANKS["three_spring"]:
        assert 0.029 < FS.spring_geometry(*s)["td"] < 0.052  # measured tanks: 29.8 to 51.3 ms


def test_spring_dispersion_low_limit_is_echo_spacing():
    r, R, N = 1.0e-4, 2.5e-3, 112
    t = FS.spring_echo_time([20.0], r, R, N)[0]
    assert abs(t / FS.spring_geometry(r, R, N)["td"] - 1.0) < 0.02
    b, f, vg, fc = FS.spring_dispersion(r, R)
    assert vg[-1] < 0.05 * vg[0] and f[-1] < fc        # group velocity falls to zero at the transition fC
    assert np.isinf(FS.spring_echo_time([1.2 * fc], r, R, N)[0])


def test_spring_chain_group_delay_and_cost():
    assert abs(FS.spring_chain_delay(0.0, 100, 5, 0.62, SR) - 100 * 5 * 0.38 / 1.62) < 1e-9
    assert abs(FS.spring_cost(100, 200, False) - 1023) < 1e-9
    assert abs(FS.spring_cost(100, 200, True) - 355.25) < 1e-9


def test_spring_eq_resonator_peak():
    from scipy.signal import freqz
    b, a = FS.spring_eq(95.0, 130.0, 5.1, SR)
    w, h = freqz(b, a, worN=np.array([95.0, 400.0, 1600.0]), fs=SR)
    db = 20 * np.log10(np.abs(h))
    assert abs(db[0]) < 0.5 and db[1] < -6 and db[2] < db[1]


def test_spring_presets_echo_spacing_and_parts():
    for name, td in (("leem1", 0.056), ("leem2", 0.044), ("sansui", 0.056)):
        w = FX.REGISTRY["spring_reverb"]["fn"](_click(SR), SR, mix=1.0, preset=name, hf=0.0)
        ac = np.correlate(w, w, "full")[len(w) - 1:]
        lag = (int(0.03 * SR) + int(np.argmax(ac[int(0.03 * SR):int(0.08 * SR)]))) / SR
        assert abs(lag - td) < 0.006, (name, lag)
    s = FX.REGISTRY["spring_reverb"]["fn"]
    base = s(_click(SR // 2), SR, mix=1.0)
    assert not np.array_equal(base, s(_click(SR // 2), SR, mix=1.0, seed=3))       # delay wander
    assert not np.array_equal(base, s(_click(SR // 2), SR, mix=1.0, clip=0.05))    # loop limiter
    for kw in (dict(efficient=True), dict(tank="three_spring"), dict(preset="leem_auto")):
        y = s(_click(SR // 2), SR, mix=1.0, **kw)
        assert np.all(np.isfinite(y)) and 1e-6 < np.max(np.abs(y)) < 2.0, kw


def test_spring_efficient_keeps_the_decay():
    s = FX.REGISTRY["spring_reverb"]["fn"]
    for eff in (False, True):
        w = s(_click(3 * SR), SR, mix=1.0, decay=2.0, hf=0.0, efficient=eff)
        r = [10 * np.log10(np.mean(w[int(a * SR):int((a + 0.3) * SR)] ** 2) + 1e-30) for a in (0.3, 1.3)]
        assert -40 < r[1] - r[0] < -20, (eff, r)


def test_spring_fit_recovers_chain():
    s = FX.REGISTRY["spring_reverb"]["fn"]
    w = s(_click(SR // 2), SR, mix=1.0, hf=0.0, preset="leem1")
    p = FS.spring_fit(w, SR, fc=4300.0)
    assert abs(p["td"] - 0.056) < 0.006 and 0.3 < p["a"] < 0.9


def test_spring_fd_model_rings_with_echoes():
    y = FS.spring_fd(_click(int(0.25 * SR)), SR, turns=60, decay=1.0, fmax=3000.0)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0
    td = FS.spring_geometry(1.0e-4, 2.5e-3, 60)["td"]
    lo = SR * 0.5 * td
    # energy keeps arriving after the first round trip (the wave comes back from the far end)
    assert np.sum(y[int(1.2 * td * SR):] ** 2) > 1e-4 * np.sum(y ** 2) and lo > 0


# ------------------------------------------------------------------ reverbs and space
from genny import reverbs as R
from genny.physical import space as S

SR2 = 22050


def _env_min(y, sr):
    e = np.sqrt(np.convolve(y ** 2, np.ones(int(0.01 * sr)) / int(0.01 * sr), "same"))
    return float(np.min(e[sr // 4:-sr // 10]))


def test_leslie_horn_floor_and_leak():
    assert R._HORN_FLOOR == 0.25                              # 0.25 + 0.75 (1 + cos)/2: -12 dB at the back
    s = np.sin(2 * np.pi * 3000 * np.arange(SR2) / SR2)
    a = R.leslie(s, SR2, speed="fast", drive=0.0, leak=-60.0)[:, 0]
    b = R.leslie(s, SR2, speed="fast", drive=0.0, leak=-10.0)[:, 0]
    assert _env_min(b, SR2) > _env_min(a, SR2)                # the fixed leak path fills the tremolo troughs


def test_moorer_sdn_raytrace_rooms_render():
    imp = _click(SR2 // 2)
    for model in ("moorer", "sdn", "raytrace"):
        y = R.room(imp, SR2, mix=1.0, model=model, preset="studio", rays=3000)
        assert y.ndim == 2 and np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0, model
        assert np.sum(y[int(0.05 * SR2):] ** 2) > 1e-6 * np.sum(y ** 2), model


def test_moorer_comb_loop_gives_t60():
    g, p = R._onepole_for(np.array([0.5]), np.array([0.25]), 0.0, 4000.0, SR)
    w = 2 * np.pi * 4000 / SR
    h = g * (1 - p) / np.abs(1 - p * np.exp(-1j * w))
    assert abs(h[0] - 0.25) < 1e-6 and abs(g[0] - 0.5) < 1e-12


def test_ray_trace_specular_decay_matches_eyring():
    alpha = np.full((6, 6), 0.3)
    surf = 2 * (80 + 32 + 40)
    t60_eyring = 0.161 * 320.0 / (-surf * np.log(1 - 0.3))

    def t60(scat):
        h = S.ray_trace((8.0, 10.0, 4.0), (3.0, 3.0, 1.5), (5.0, 7.0, 1.6), alpha, np.full(6, scat), np.zeros(6),
                        rays=4000, tmax=0.6, bin_s=0.01, seed=1)
        e = h[:, 2]
        t = (np.arange(e.size) + 0.5) * 0.01
        ok = (t > 0.05) & (t < 0.5) & (e > 0)
        return -60.0 / np.polyfit(t[ok], 10 * np.log10(e[ok]), 1)[0]
    # diffuse walls: Eyring holds; purely specular box: a non-diffuse field, the grazing paths decay slower
    assert abs(t60(1.0) / t60_eyring - 1.0) < 0.15
    assert t60(0.0) > t60(1.0)


def test_histogram_ir_follows_histogram_energy():
    hist = np.exp(-np.arange(40) * 0.2)[:, None] * np.full((40, 6), 1e-2)
    ir = R.histogram_ir(hist, 0.005, SR2, volume=300.0, seed=0)
    L = int(round(0.005 * SR2))
    e = np.add.reduceat(ir ** 2, np.arange(0, ir.size, L))
    assert abs(10 * np.log10(e[30] / e[10]) - 10 * np.log10(np.exp(-4.0))) < 3.0


def test_dark_velvet_is_darker_and_two_slope_knee():
    v = R.velvet_noise(SR, SR, 2205.0, np.random.default_rng(0))
    d = R.dark_velvet_noise(SR, SR, 2205.0, 1.0, np.random.default_rng(0))
    hi = lambda s: np.sum(np.abs(np.fft.rfft(s))[SR // 4:] ** 2) / np.sum(np.abs(np.fft.rfft(s)) ** 2)
    assert hi(d) < 0.5 * hi(v)
    env = R.two_slope_envelope([0.0, 0.5, 1.5], 1.5, 6.0, -20.0)
    assert abs(20 * np.log10(env[1]) + 20.0) < 1e-6 and abs(20 * np.log10(env[2]) + 30.0) < 1e-6


def test_velvet_kinds_and_hall_presets():
    x = _click(SR2 // 4)
    for kw in (dict(kind="dark"), dict(kind="edvn"), dict(kind="sparse"), dict(t60_late=4.0), dict(preset="musikverein")):
        y = R.velvet_reverb(x, SR2, mix=1.0, t60_low=1.0, t60_high=0.6, **kw)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0, kw
    assert R.HALL_PRESETS["pori"] == (2.3, 0.110) and R.HALL_PRESETS["musikverein"][1] == 0.052


def test_sparse_velvet_equals_dense_convolution():
    x = np.random.default_rng(3).standard_normal(2000)
    y = R.sparse_velvet_convolve(x, SR2, 0.3, 2205.0, segment=0.02, seed=4)
    n = R.samples(min(1.2 * 0.3 + 0.05, 12.0), SR2)
    v = R.velvet_noise(n, SR2, 2205.0, np.random.default_rng(4))
    pos = np.nonzero(v)[0]
    L = int(0.02 * SR2)
    edges = np.searchsorted(pos, np.arange(0, n + L, L))
    g = np.exp(-6.91 * (np.arange(edges.size - 1) * L + 0.5 * L) / SR2 / 0.3)
    g /= np.sqrt(np.sum(g ** 2 * np.diff(edges)))
    h = np.zeros(n)
    for s in range(edges.size - 1):
        k = pos[edges[s]:edges[s + 1]]
        h[k] = v[k] * g[s]
    assert np.allclose(y, np.convolve(x, h)[:2000], atol=1e-9)


def test_jpverb_greyhole_decay():
    x = _click(SR2 * 2)
    y = R.jpverb(x, SR2, t60=1.0, mix=1.0)
    e = lambda a, b: np.mean(y[int(a * SR2):int(b * SR2)] ** 2)
    assert -45 < 10 * np.log10(e(0.6, 0.8) / e(0.1, 0.3)) < -15     # about -30 dB in 0.5 s for t60 = 1 s
    g = R.greyhole(x, SR2, delay=0.1, feedback=0.5, mix=1.0)
    assert np.all(np.isfinite(g)) and np.max(np.abs(g[int(0.3 * SR2):])) < np.max(np.abs(g))


def test_room_feedback_howls_above_one_and_not_below():
    x = 1e-3 * np.random.default_rng(0).standard_normal(SR2 // 10)
    late = lambda y: np.sqrt(np.mean(y[-SR2 // 2:] ** 2))
    assert late(R.room_feedback(x, SR2, gain=1.4, preset="studio")) > 30 * late(R.room_feedback(x, SR2, gain=0.5, preset="studio"))
    assert np.all(np.isfinite(R.room_feedback(x, SR2, gain=1.0, shift=2.0, swell=0.5, pipe=110.0, preset="studio")))


def test_reflection_kinds():
    assert abs(R.reflector_cutoff(2.0, 5.0, 5.0) - 2 * R.C_AIR / 4.0 * 2.5) < 1e-9
    x = _click(SR2 // 2)
    y = R.reflection(x, SR2, kind="flutter", width=10.0, absorption=0.0, level=1.0)
    k = int(round(20.0 / R.C_AIR * SR2))
    assert abs(int(np.argmax(np.abs(y[10:]))) + 10 - k) <= 2
    for kind in ("fence", "stairs", "panel"):
        assert np.all(np.isfinite(R.reflection(x, SR2, kind=kind)))


def test_huggins_phase_and_delay():
    x = np.random.default_rng(1).standard_normal(SR2)
    y = R.huggins(x, SR2, freq=600.0)
    assert np.allclose(y[:, 0], x) and abs(np.std(y[:, 1]) / np.std(x) - 1.0) < 0.05
    d = R.huggins(x, SR2, mode="delay", delay=0.012)
    k = int(round(0.012 * SR2))
    assert np.allclose(d[k:, 1], x[:-k])


def test_bell_swing_seat_pv_reverb_soundboard():
    s = np.sin(2 * np.pi * 500 * np.arange(SR2 * 2) / SR2)
    y = R.bell_swing(s, SR2, period=2.0)
    assert y.shape[0] == s.shape[0] and np.all(np.isfinite(y))
    a, b = R.orchestra_seat(s, SR2, desk=1), R.orchestra_seat(s, SR2, desk=3)
    assert abs(20 * np.log10(np.std(b[SR2:]) / np.std(a[SR2:])) + 7.0) < 0.5
    p = R.pv_reverb(_click(SR2 // 2), SR2, mix=1.0)
    assert p.ndim == 2 and np.all(np.isfinite(p))
    sb = R.soundboard_fdn(_click(SR2 * 2), SR2)
    assert np.all(np.isfinite(sb)) and np.max(np.abs(sb)) > 0


def test_room_scattering_and_mic_pattern():
    x = _click(SR2 // 2)
    base = R.room(x, SR2, mix=1.0, preset="studio")
    sc = R.room(x, SR2, mix=1.0, preset="studio", scattering="rpg_qrd")
    card = R.room(x, SR2, mix=1.0, preset="studio", mic=0.5)
    assert not np.allclose(base, sc)
    late = lambda y: np.sum(y[int(0.1 * SR2):] ** 2)
    assert late(card) < late(base)                            # a cardioid picks up a third of the diffuse field
    assert "priory" in R.ROOM_PRESETS and R.room_info("priory")["t60"][2] > 4.0


def test_distance_shadow_duct_water_rolling():
    x = np.random.default_rng(2).standard_normal(SR2)
    still = R.distance(x, SR2, metres=2000.0, air=False)
    assert np.std(R.distance(x, SR2, metres=2000.0, air=False, gradient=-6.0)) < np.std(still)
    assert S.duct_gain(5000.0, 1.5, 6.0) > 1.0
    w = R.distance(x, SR2, metres=10.0, air=False, across_water=True)
    assert abs(20 * np.log10(np.std(w) / np.std(R.distance(x, SR2, metres=10.0, air=False))) + 29.5) < 0.3
    assert np.all(np.isfinite(R.distance(x, SR2, metres=50.0, ground="grass", rolling=True)))


def test_directivity_horn_wall_podium_split():
    x = np.random.default_rng(3).standard_normal(SR2)
    for kw in (dict(instrument="french_horn", angle=0.0), dict(podium=True), dict(split=True)):
        y = R.directivity(x, SR2, **kw)
        assert y.shape == x.shape and np.all(np.isfinite(y)), kw
    a = R.directivity(x, SR2, instrument="french_horn", angle=0.0)
    b = R.directivity(x, SR2, instrument="french_horn", angle=0.0, wall=True)
    assert np.std(b) > 2.0 * np.std(a)


def test_space_formulas():
    zc, g = S.delany_bazley([1000.0], 20000.0)
    assert abs(g[0].real - 18.8) < 0.3 and abs(g[0].imag - 31.1) < 0.3      # 0.188 + j0.311 Np/cm
    assert 0.0 < S.layer_absorption([1000.0], 20000.0, 0.05)[0] < 1.0
    assert abs(10 * np.log10(S.interface_transmission()) + 29.5) < 0.2
    assert abs(S.pa_range(15000.0, 2.0, 16.0)["r_max"] - 28.0) < 1.0
    assert abs(S.panel_absorber_f0(10.0, 5.0) - 600 / np.sqrt(50)) < 1e-9
    assert abs(S.helmholtz_absorber(100.0)["radius"] - 0.55) < 1e-12
    assert abs(S.qrd_depths(7, 500.0)[1] - (S.C_AIR / 500.0) * (2 * np.pi / 7) / (4 * np.pi)) < 1e-12
    st = S.transfer_statistics(2.0, 1000.0)
    assert st["max_spacing_hz"] == 2.0 and abs(st["peak_db"] - 4.34 * np.log(np.log(2000.0))) < 1e-9
    assert abs(S.diffuse_coherence([S.C_AIR / 0.4], 0.2)[0]) < 1e-9
    assert S.stc_rating([v + 40 for _, v in S.STC_CONTOUR]) == 42
    sd = S.schroeder_design(44100, 2.0)
    assert len(sd["allpass"]) == 5 and abs(sd["allpass"][0] - 4410) < 5 and np.all(sd["comb_g"] <= 0.85)
    gv = S.vbap_gains(0.0)
    assert abs(np.linalg.norm(gv) - 1.0) < 1e-12 and abs(gv[0] - gv[1]) < 1e-12
    b = S.bformat(np.ones(4), 90.0)
    assert np.allclose(b[:, 2], 1.0) and np.allclose(b[:, 0], 1 / np.sqrt(2))
    assert S.shadow_excess_db([1000.0], 500.0, 1.5, 1.5, -6.0)[0] == 0.0
    assert S.shadow_excess_db([1000.0], 5000.0, 1.5, 1.5, -6.0)[0] > 4.0
    assert abs(S.refraction_radius(-6.0) / 1000.0 - 97.7) < 1.0
    y = S.puckette_rev(_click(4800), kind="rev2", damp_hz=2000.0, damping=0.5)
    z = S.puckette_rev(_click(4800), capture=0.02)
    assert np.all(np.isfinite(y)) and np.abs(z[-500:]).max() > 0


def test_directivity_omni_below_winds_limit():
    from genny.physical.winds import OMNI_BELOW
    x = np.random.default_rng(5).standard_normal(SR2)
    y = R.directivity(x, SR2, instrument="trumpet", angle=180.0)
    nf = 1 << int(np.ceil(np.log2(x.shape[0] + int(0.02 * SR2))))
    f = np.fft.rfftfreq(nf, 1.0 / SR2)
    H = np.abs(np.fft.rfft(y, nf)) / (np.abs(np.fft.rfft(x, nf)) + 1e-12)
    lo = (f > 50) & (f < 0.9 * OMNI_BELOW["trumpet"])
    assert abs(np.mean(H[lo] ** 2) - 1.0) < 0.1
