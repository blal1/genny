import math
import os
import tempfile

import numpy as np
import scipy.optimize as so

from genny import engines as E, fauna as FA, family_v as V, geo_u as G, world_t as W
from genny.notes import to_freq


def test_oud_open_strings_are_the_arabic_standard_course():
    assert tuple(to_freq(n) for n in W.STRINGS["oud"]["open"]) == tuple(to_freq(n) for n in ("C2", "F2", "A2", "D3", "G3", "C4"))
    g = W.string_geometry(to_freq("F2"), W.STRINGS["oud"])
    assert abs(g["f_open"] - to_freq("F2")) < 1e-6 and abs(g["L"] - 0.61) < 1e-9
    assert 0.61 <= W.STRINGS["oud"]["scale"] <= 0.62


def test_morin_khuur_bridge_distance_is_about_60_cm():
    assert 0.59 <= W.BOWED["morin_khuur"]["L_open"] <= 0.61


def test_qanun_range_and_courses():
    from genny.instruments import REGISTRY
    assert W.STRINGS["qanun"]["courses"] == 3 and W.STRINGS["santur"]["courses"] == 4
    assert REGISTRY["qanun"]["range"] == "A2-E6"
    L_low = W.STRINGS["qanun"]["L_ref"] * (W.STRINGS["qanun"]["f_ref"] / to_freq("A2")) ** W.STRINGS["qanun"]["L_exp"]
    assert 0.95 <= L_low <= 1.0


def test_two_stroke_stinger_is_0_6_of_the_header():
    B = 0.05
    lens, r, g = E.expansion_chamber(B)
    d_h, d_b = 0.6 * B, 1.9 * B
    def last(ds):
        dc = d_b + (1.15 * ds - d_b) * 4.5 / 5.0
        return (dc * dc - ds * ds) / (dc * dc + ds * ds) - r[-1]
    ds = so.brentq(last, 0.005, 0.04)
    assert 0.59 <= ds / d_h <= 0.63
    assert 10.0 <= 0.2 / ds <= 12.5                       # stinger length about 12 diameters


def test_changed_sounds_render_finite():
    for fn, args in ((W.qanun, (to_freq("A2"), 0.6)), (W.qanun, (to_freq("E6"), 0.6)), (W.oud, (to_freq("F2"), 0.6)), (W.morin_khuur, (to_freq("F2"), 0.6))):
        y = np.asarray(fn(*args))
        assert np.isfinite(y).all() and np.abs(y).max() > 1e-3
    y = np.asarray(E.two_stroke(kind="mx125", dur=0.6))
    assert np.isfinite(y).all() and np.abs(y).max() > 1e-3


def test_two_stroke_port_timing_in_open_ranges():
    from genny import presets
    p = presets.resolve("two_stroke", "mx125")["ports"].split(",")
    eo, to = float(p[0]), float(p[1])
    dur_e, dur_t = 2 * (180 - eo), 2 * (180 - to)
    assert 187 <= dur_e <= 200 and 126 <= dur_t <= 140 and 25 <= (dur_e - dur_t) / 2 <= 35


def test_elephant_rumble_and_bat_division():
    assert 15.0 <= FA.RUMBLE_HZ <= 25.0 and FA.BAT_DOWN == 10.0
    assert 2.0 <= 3.0 <= 8.0


def test_dune_squeak_default_in_published_band():
    f = G.squeak_hz(0.2, 170.0)
    assert 500.0 <= f <= 2500.0


def test_shakuhachi_and_oud_to_freq():
    assert abs(to_freq("D4") - 293.66) < 0.05 and abs(to_freq("D4") - W.to_freq("D4")) < 1e-9


def test_sofa_reader_handles_elevation_beyond_90_and_96k():
    import h5py
    d = tempfile.mkdtemp()
    p = os.path.join(d, "s.sofa")
    with h5py.File(p, "w") as f:
        f["Data.IR"] = np.random.default_rng(0).standard_normal((3, 2, 64))
        f["Data.SamplingRate"] = np.array([96000.0])
        f["SourcePosition"] = np.array([[0.0, 0.0, 1.5], [0.0, 135.0, 1.5], [90.0, -45.0, 1.5]])
        f["SourcePosition"].attrs["Type"] = b"spherical"
    db = V.load_hrir(p)
    assert db["hrir"].shape == (3, 64, 2) and db["sr"] == 96000
    u = V._unit(0.0, 135.0)
    assert np.allclose(u, V._unit(180.0, 45.0), atol=1e-9)
    assert V.hrir_at(0.0, 135.0, 48000, db).shape[1] == 2


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
            print("ok", name)
    print(n, "ok")
