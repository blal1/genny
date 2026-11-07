import numpy as np

SR = 48000


def _hz(seg, sr=SR):
    z = np.nonzero(np.diff(np.signbit(seg)))[0]
    return (len(z) - 1) / 2 / ((z[-1] - z[0]) / sr)


def test_tr808_bd_loop_rings_at_49_5_with_stock_parts():
    from genny import drums as D
    assert D.TR808_BD_HZ == 49.5
    p = dict(D.TR808_BD_FEEDBACK)
    p["VR6"] = 0.5 * p["VR6"]
    rk = 1.0 / (1.0 / (p["R165"] + p["R166"]) + 1.0 / p["R161"])
    A = np.stack([D._bd_deriv(e, 0.0, rk, p["C41"], p["C42"], p) for e in np.eye(3)], axis=1)
    f = np.max(np.abs(np.linalg.eigvals(A).imag)) / (2 * np.pi)
    assert abs(f - 49.5) < 0.5
    y = D.kick808(SR, 1.0, model="circuit", sigh=0)
    assert abs(_hz(y[int(0.4 * SR):int(0.9 * SR)]) - 49.5) < 0.5


def test_tr808_pitch_sigh_q43_fit_and_settling():
    from genny import drums as D
    assert abs(D.sigh_current(-1.5) + 1.394e-5) < 0.01e-5
    assert abs(D.sigh_current(1.0)) < 1e-9
    on = D.kick808(SR, 1.0, model="circuit", sigh=1)
    off = D.kick808(SR, 1.0, model="circuit", sigh=0)
    early = lambda y: _hz(y[int(0.02 * SR):int(0.08 * SR)])
    assert early(on) > early(off) + 1.0
    assert abs(_hz(on[int(0.4 * SR):int(0.9 * SR)]) - 49.5) < 0.6


def test_keefe_closed_hole_inner_term_scavone_closhole():
    import math
    from genny.physical import winds as W
    a, b, t, f = 0.00945, 0.00476, 0.0034, 300.0
    plain = W.keefe_hole(a, b, t, f, open_=False)
    inner = W.keefe_hole(a, b, t, f, open_=False, inner=True)
    th, k = plain["th"], 2 * math.pi * f / W.air(26.85)["c"]
    rh = W.air(26.85)["rho"] * W.air(26.85)["c"] / (math.pi * b * b)
    ci = k * th * (0.25 * (b / th) ** 2 + 0.58 * (b / a) ** 2 - math.pi * b / (4 * th))
    assert abs((inner["Zs"] - plain["Zs"]) / (-1j * rh * ci) - 1.0) < 1e-6
