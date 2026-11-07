"""V7_EFFETS6: last open T_Effets rows (filters, analysis, aeolian, contact, plates, fdstring, body morph, ...).

Run:  uv run --with pytest pytest tests/test_v7_effets6.py -q
"""
import math

import numpy as np

SR = 48000


def _peak_hz(y, sr=SR, lo=20.0, hi=None):
    s = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    m = (f >= lo) & (f <= (hi or sr / 2))
    return float(f[m][np.argmax(s[m])])


# ---------------------------------------------------------------- dsp / filters / calibrate
def test_butter2_r_and_peak_independent_of_order():
    from genny import dsp
    assert np.allclose(dsp.butter2_r(1.0 / math.sqrt(2.0), 1), [1.0 / math.sqrt(2.0)])
    x = np.zeros(1 << 15)
    x[0] = 1.0
    for N in (2, 3):
        h = dsp.butter2(x, 1000.0, SR, R=0.3, N=N)
        H = np.abs(np.fft.rfft(h))
        assert abs(20 * np.log10(H.max() / H[1]) - 4.847) < 0.3


def test_onepole_nl_zdf_hard_clips_state():
    from genny import filters as F
    x = np.full(4000, 10.0)
    y = F.onepole_nl(x, 2000.0, SR, kind="zdf_hard")
    assert np.all(np.isfinite(y)) and y[-1] > 1.0 and y[-1] < 10.0 + 1e-6
    y2 = F.onepole_nl(np.full(4000, 0.1), 2000.0, SR, kind="zdf_ota")
    assert abs(y2[-1] - 0.1) < 0.01


def test_f0_l1_midi_semitone():
    from genny.calibrate import f0_l1_midi
    a = np.full(100, 220.0)
    assert abs(f0_l1_midi(a, a * 2 ** (1 / 12)) - 1.0) < 1e-9
    b = a.copy()
    b[:50] = 0.0
    assert abs(f0_l1_midi(a, b * 2 ** (2 / 12)) - 2.0) < 1e-9


# ---------------------------------------------------------------- physical/analysis
def test_pitch_clean_drops_short_segment():
    from genny.physical.analysis import pitch_clean
    f0 = np.full(200, 200.0)
    f0[100:102] = 400.0
    y = pitch_clean(f0, 0.01)
    assert np.all(y[95:110] < 300.0)


def test_modal_fit_vote_rejects_noise_peak():
    from genny.physical.analysis import modal_fit
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 440 * t) * np.exp(-3 * t) + 0.001 * rng.standard_normal(SR)
    fr = np.asarray(modal_fit(x, 4, method="vote").freqs)
    assert np.any(np.abs(fr - 440.0) < 5.0)


def test_vector_mix_corners():
    from genny.physical.analysis import vector_mix
    a, b, c, d = (np.full(4, v) for v in (1.0, 2.0, 3.0, 4.0))
    assert np.allclose(vector_mix(a, b, c, d, 0.0, 0.0), 1.0)
    assert np.allclose(vector_mix(a, b, c, d, 0.5, 0.5), 2.5)


def test_ringmod_leak():
    from genny import fx
    x = np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    y0 = fx.ringmod(x, SR, freq=200, mix=1.0)
    y1 = fx.ringmod(x, SR, freq=200, mix=1.0, leak=0.5)
    s = np.abs(np.fft.rfft(y1))
    assert s[300] > 0.1 * s.max() and np.abs(np.fft.rfft(y0))[300] < 1e-3 * s.max()


# ---------------------------------------------------------------- chiptune2 / vehicle / cyber
def test_glitch_saw_two_pitches():
    from genny import chiptune2
    y = chiptune2.glitch_saw(220.0, 0.5, SR, freq2=1.5)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01


def test_car_engine_noise_adds_highband():
    from genny import vehicle
    a = vehicle.car_engine(SR, dur=0.5, model="ptr")
    b = vehicle.car_engine(SR, dur=0.5, model="ptr", noise=1.0)
    hf = lambda y: float(np.sum(np.abs(np.fft.rfft(y))[1000:] ** 2))
    assert hf(b) > hf(a)


def test_transformer_tesla_level_follows_leq():
    from genny import cyber
    lo = cyber.transformer(SR, dur=0.5, tesla=1.3)
    hi = cyber.transformer(SR, dur=0.5, tesla=1.9)
    assert np.sqrt(np.mean(hi ** 2)) > np.sqrt(np.mean(lo ** 2))


def test_firing_transients_fade_above_160hz():
    from genny import cyber
    rng = np.random.default_rng(0)
    slow = cyber.firing_transients(np.full(SR // 4, 15.0 / SR), 4, rng, 1.0)        # 60 Hz firing: under 80, full
    fast = cyber.firing_transients(np.full(SR // 4, 200.0 / SR), 4, np.random.default_rng(0), 1.0)
    assert np.abs(slow).max() > 0 and np.abs(fast).max() == 0


# ---------------------------------------------------------------- friction
def test_force_pitch_moves_to_target():
    from genny import friction
    t = np.arange(SR) / SR
    x = np.sign(np.sin(2 * np.pi * 200 * t)) * 0.3
    y = friction.force_pitch(x, SR, note="A3")
    assert abs(_peak_hz(y, lo=100, hi=400) - 220.0) < 12.0


def test_sympathetic_random_tuning_seeded():
    from genny import friction
    x = np.zeros(SR // 2)
    x[0] = 1.0
    a = friction.sympathetic(x, SR, tuning="random", seed=1)
    b = friction.sympathetic(x, SR, tuning="random", seed=2)
    assert not np.allclose(a, b)


# ---------------------------------------------------------------- physical/modal (aeolian)
def test_exp_sigmoid_bounds():
    from genny.physical.modal import exp_sigmoid
    s1 = 1.0 / (1.0 + math.exp(-1.0))
    assert abs(exp_sigmoid(1.0) - (2.0 * s1 ** math.log(10.0) + 1e-7)) < 1e-12   # max 2 sigmoid^ln 10 + 1e-7
    assert abs(exp_sigmoid(50.0) - 2.0) < 1e-6 and 1e-7 <= exp_sigmoid(-50.0) < 2e-7  # bounds: 1e-7 .. 2


def test_goldstein_intensity_scales_u6():
    from genny.physical.modal import goldstein_intensity
    i1 = goldstein_intensity(10.0, 0.005, 0.2)
    i2 = goldstein_intensity(20.0, 0.005, 0.2)
    assert abs(i2 / i1 - 64.0) < 1.0


def test_lock_state_hysteresis():
    from genny.physical.modal import lock_state
    s = lock_state([0.5, 0.9, 1.3, 0.75, 0.5], hyst=1.1)
    assert list(s) == [0, 1, 1, 1, 0]


def test_interp_rows_bilinear_grid():
    from genny.physical.modal import _interp_rows
    rows = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert abs(float(_interp_rows(rows, (0.5, 0.5), grid=(2, 2))[0]) - 1.5) < 1e-12


def test_wake_noise_level():
    from genny.physical.modal import wake_noise
    y = wake_noise(10.0, 0.005, 0.5, np.random.default_rng(0))
    assert np.isfinite(y).all() and np.abs(y).max() > 0


# ---------------------------------------------------------------- contact
def test_hertz_tau_formula():
    from genny.contact import hertz_tau
    tau = hertz_tau(0.01, 0.005, 2e11, 0.3, 1.0)
    es = 2e11 / (1 - 0.09)
    assert abs(tau - 2.87 * (0.01 ** 2 / (0.005 * es * es)) ** 0.2) < 1e-12


def test_acceleration_noise_peak():
    from genny.contact import acceleration_noise
    p = acceleration_noise(1e-4, 1.0, 0.005, SR)
    assert abs(p[0] - 1.2 * 0.005 ** 3 / (2 * 343.0) * math.pi ** 2 / (2 * 1e-8)) < 1e-9


def test_lod_gains_far_keeps_fewer():
    from genny.contact import lod_gains
    g = np.ones(20)
    r = np.linspace(1, 20, 20)
    near = lod_gains(g, r, r * 0.01, 1.0)
    far = lod_gains(g, r, r * 0.01, 20.0)
    assert np.count_nonzero(far) < np.count_nonzero(near)


def test_cavity_modes_gamma_between_box_and_sphere():
    from genny.contact import cavity_modes
    box = cavity_modes("box", 0.5, gamma=2.0)[0]
    mid = cavity_modes("box", 0.5, gamma=6.0)[0]
    assert not np.allclose(np.sort(box)[:3], np.sort(mid)[:3])


# ---------------------------------------------------------------- percussion / physics / ambience
def test_cubic_mode_hardening():
    from genny.percussion import cubic_mode
    lin = cubic_mode(100.0, 1.0, 1.0, 0.0, SR, SR)
    nl = cubic_mode(100.0, 1.0, 1.0, 5e6, SR, SR)
    assert _peak_hz(nl[:SR // 4], lo=50, hi=400) > _peak_hz(lin[:SR // 4], lo=50, hi=400)


def test_gong_hardness_changes_sound():
    from genny import percussion
    a = percussion.gong(SR, hardness=0.0)
    b = percussion.gong(SR, hardness=1.0)
    assert not np.allclose(a, b)


def test_thunder_post_changes_sound():
    from genny.physical.ambience import thunder
    a = thunder(np.random.default_rng(0), dur=3.0, fineberg=True)
    b = thunder(np.random.default_rng(0), dur=3.0, fineberg=True, post=True)
    assert not np.allclose(a, b)


# ---------------------------------------------------------------- tubes
def test_hole_rate_formula():
    from genny.tubes import ETA, RHO, hole_rate
    b, l, f = 0.004, 0.01, 300.0
    kb = 2 * math.pi * f * b / 343.0
    want = (2 / b) * math.sqrt(math.pi * f * ETA / RHO) + 343.0 * kb * kb / (4 * l)
    assert abs(hole_rate(b, l, f) - want) < 1e-9


def test_wah_pipe_low_pressure_only_fundamental():
    from genny import tubes
    y = tubes.wah_pipe(SR, length_m=2.0, dur=2.0, ramp=0.1, pressure=0.05)
    f1 = 343.0 / (2 * (2.0 + 2 * 0.6133 * 0.127))
    assert abs(_peak_hz(y[SR:], lo=40, hi=2000) - f1) < 3.0


# ---------------------------------------------------------------- plates
def test_room2d_absorb_shortens_tail():
    from genny import plates
    x = np.zeros(8000)
    x[10] = 1.0
    late = lambda z: float(np.sqrt(np.mean(z[-6000:] ** 2)) / np.sqrt(np.mean(z[:6000] ** 2)))
    a = plates.room2d(x, mix=1.0, width=0.0, tail=0.5)
    b = plates.room2d(x, mix=1.0, width=0.0, tail=0.5, absorb=0.5)
    assert late(b) < 0.1 * late(a)
    assert abs(plates.wall_impedance(0.5) - (3 + 2 * math.sqrt(2))) < 1e-9


def test_bracket_adjoint_identity():
    from genny import plates
    rng = np.random.default_rng(0)
    Nx, Ny, hx, hy = 7, 6, 0.13, 0.17
    a, v, phi = rng.standard_normal((3, Nx + 1, Ny + 1))
    phi[0], phi[-1], phi[:, 0], phi[:, -1] = 0, 0, 0, 0
    g, cc, o = np.zeros((Nx + 1, Ny + 1)), np.empty((Nx, Ny)), np.empty((Nx - 1) * (Ny - 1))
    plates._bracket_adj(a, phi, cc, hx, hy, g)
    plates._bracket(a, v, cc, hx, hy, o)
    assert abs((g * v).sum() - hx * hy * (phi[1:-1, 1:-1].ravel() * o).sum()) < 1e-8 * abs((g * v).sum())


def test_nl_plate_free_edges_runs():
    from genny import plates
    info = {}
    y = plates.nl_plate(dur=0.15, quality=0.5, edges="free", force=5.0, info=info)
    assert np.isfinite(y).all() and info["n"] > (info["n"] ** 0.5 - 2) ** 2 and info["guard_hits"] == 0


# ---------------------------------------------------------------- fdstring
def test_string_absorbing_end_kills_reflections():
    from genny import fdstring as S

    def ratio(**kw):
        s = S.StiffString(220.0, SR, B=0.0, t60=None)
        if kw:
            s.end("right", **kw)
        s.hammer(0.3, mass=0.3, w=3000.0, alpha=2.0, v=1.0)
        y = s.run(int(0.2 * s.sr_i))[:, 0]
        return float(np.sqrt(np.mean(y[y.size // 2:] ** 2)) / np.sqrt(np.mean(y[: y.size // 10] ** 2)))
    assert ratio() > 0.5 and ratio(absorb=1.0) < 1e-3


def test_arched_bar_ratios_under_max_rate():
    from genny import fdstring as S
    for th in (0.0, 0.25):
        b = S.Bar(1046.0, SR, ("free", "free"), (440.0, 1.0, 4000.0, 0.2), 32, (4.0, 10.0), theta=th)
        assert b.os * SR <= 1.5e6
        assert np.allclose(b.modes[1:3] / b.modes[0], (4.0, 10.0), rtol=3e-3)


# ---------------------------------------------------------------- acoustic body morph
def test_warped_allpole_stable_and_dc_gain():
    from genny import acoustic as A
    ir = A._body_ir(SR, 0, 0.5, False)
    a, ks = A.warped_lpc(ir)
    assert np.all(np.abs(ks) < 1)
    h = A.warped_allpole(np.r_[1.0, np.zeros(SR - 1)], a)
    assert np.isfinite(h).all() and abs(h[-SR // 10:]).max() < 1e-6
    assert abs(h.sum() - 1.0 / a.sum()) < 1e-3 * abs(1.0 / a.sum())


def test_body_morph_endpoints_differ():
    from genny import acoustic as A
    x = np.random.default_rng(0).standard_normal(SR // 4)
    a = A.body_morph(x, SR, morph=0.0)
    b = A.body_morph(x, SR, morph=1.0)
    assert np.isfinite(a).all() and not np.allclose(a, b)
