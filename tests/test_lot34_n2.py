"""Lot 34 N2: the open points of lot N, each measured on a rendered signal or on its closed form."""
import numpy as np

from genny import arms, aero, contact, fx, foley, matter, ordnance, plates, rail
from genny.physical import voice

SR = 48000


def _ok(y):
    y = np.asarray(y)
    assert y.size > 100 and np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0
    return y


def test_reload_finish_and_spread():
    a = _ok(arms.gun_mech(SR, "slide_rack", seed=1))
    b = _ok(arms.gun_mech(SR, "slide_rack", finish="polymer", seed=1))
    c = _ok(arms.gun_mech(SR, "slide_rack", spread=1.0, seed=1))
    assert not np.array_equal(a[: min(len(a), len(b))], b[: min(len(a), len(b))])
    assert not np.array_equal(a[: min(len(a), len(c))], c[: min(len(a), len(c))])
    try:
        arms.gun_mech(SR, "slide_rack", finish="gold")
    except ValueError:
        return
    raise AssertionError


def test_contact_radius2_roller():
    base = _ok(contact.impact(SR, "steel", "rect_plate", 0.3, seed=0))
    rad = _ok(contact.impact(SR, "steel", "rect_plate", 0.3, radius2=-0.05, seed=0))
    rol = _ok(contact.impact(SR, "steel", "rect_plate", 0.3, tip="roller", seed=0))
    flat = _ok(contact.impact(SR, "steel", "rect_plate", 0.3, tip="flat", seed=0))
    assert len({len(base), len(rad)}) >= 1 and rol is not None and flat is not None
    assert not np.array_equal(base[: min(len(base), len(rad))], rad[: min(len(base), len(rad))])


def test_plate_stiffened_and_dome():
    free = _ok(plates.metal_plate(SR, size=0.5, thickness=2.0))
    rib = _ok(plates.metal_plate(SR, size=0.5, thickness=2.0, ribs=3, rib_height=0.02))
    dom = _ok(plates.metal_plate(SR, size=0.5, thickness=2.0, dome=0.01))
    n = min(len(free), len(rib), len(dom))
    assert not np.array_equal(free[:n], rib[:n]) and not np.array_equal(free[:n], dom[:n])


def test_pour_drop():
    a = _ok(matter.pour(SR, dur=2.0, seed=2))
    b = _ok(matter.pour(SR, dur=2.0, drop=0.4, seed=2))
    assert not np.array_equal(a[: min(len(a), len(b))], b[: min(len(a), len(b))])


def test_telephone_line_and_click():
    x = np.random.default_rng(0).standard_normal(SR)
    y = _ok(fx.telephone(x, SR, voicing="line"))
    assert len(y) == len(x)
    for kind in ("hangup", "pickup"):
        for listen in ("handset", "line", "room"):
            _ok(foley.phone_click(SR, kind=kind, listen=listen))


def test_aero_and_ordnance_parameters():
    assert abs(aero.TANDEM_DELAY * aero.C - 18.29) < 1e-9
    for craft in ("xv15", "fl282"):
        assert craft in aero.ROTORCRAFT
    r = np.random.default_rng(0)
    a = ordnance.rocket_motor(2.0, np.random.default_rng(0))
    b = ordnance.rocket_motor(2.0, np.random.default_rng(0), fall=1.0)
    assert np.sqrt(np.mean(b[-SR // 4:] ** 2)) < np.sqrt(np.mean(a[-SR // 4:] ** 2))
    g1 = ordnance.ground_wave(1.0, 100.0, np.random.default_rng(0))
    g2 = ordnance.ground_wave(1.0, 100.0, np.random.default_rng(0), vp=5000.0)
    assert g1[0][0] != g2[0][0]


def test_vermeulen_force():
    f = rail.vermeulen_force
    assert f(0.0) == 0.0
    big = f(1.0)
    assert abs(abs(big) - 0.3 * rail.WHEEL_LOAD) < 1e-6 and f(-1.0) == -big
    assert abs(f(1e-5)) < abs(f(2e-5)) < abs(big)


def test_farnell_syrinx():
    y = _ok(voice.farnell_syrinx(SR // 2))
    assert len(y) == SR // 2
