import numpy as np

from genny import chiptune_q as Q
from genny import psychoacoustics_p2 as P

SR = 44100


def _harm(tab):
    a = np.abs(np.fft.rfft(np.asarray(tab, float) - np.mean(tab)))[1:17]
    return a / a.max()


def test_prom_default_shape_and_range():
    p = Q.PROM_DEFAULT
    assert p.shape == (256,)
    assert p.min() >= 0 and p.max() <= 15 and np.all(p == np.round(p))
    for w in range(8):
        blk = p[w * 32:(w + 1) * 32]
        assert blk.max() - blk.min() >= 7
        assert abs(blk[0] - blk[-1]) <= 7 or w in (4, 5, 6, 7)


def test_prom_default_known_shapes():
    p = Q.PROM_DEFAULT
    h0 = _harm(p[0:32])
    assert int(np.argmax(h0)) == 0 and h0[1:].max() < 0.12
    h2 = _harm(p[64:96])
    assert int(np.argmax(h2)) == 1 and h2[0] < 0.3
    h6 = _harm(p[192:224])
    odd, even = h6[0::2], h6[1::2]
    assert even.max() < 0.01 and abs(odd[1] - odd[0] / 9) < 0.02
    h7 = _harm(p[224:256])
    assert h7[0] < 0.01 and abs(h7[3] / h7[1] - 0.5) < 0.05 and abs(h7[5] / h7[1] - 1 / 3) < 0.05
    assert p[224:256].tolist() == list(range(16)) * 2


def test_rendered_pitch_and_default_vs_generic():
    f = Q.wsg_hz(440.0)
    for w in (0, 2, 6, 7):
        y = Q.render_wsg({"v1": [(0.0, 1.0, 440.0, 1.0, dict(Q.WSG_PATCH, wave=w))]}, 1.0, SR)
        s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        sel = (fr > 100) & (fr < 4000)
        pk = fr[sel][np.argmax(s[sel])]
        assert abs(pk - f * (2 if w in (2, 7) else 1)) < 2.0, (w, pk)
    pat = dict(Q.WSG_PATCH, wave=1)
    d = Q.render_wsg({"v1": [(0.0, 0.5, 440.0, 1.0, pat)]}, 0.5, SR)
    g = Q.render_wsg({"v1": [(0.0, 0.5, 440.0, 1.0, pat)]}, 0.5, SR, generic=True)
    n = Q.render_wsg({"v1": [(0.0, 0.5, 440.0, 1.0, dict(Q.WSG_PATCH, wave="saw"))]}, 0.5, SR)
    assert np.allclose(g, n) and not np.allclose(d, g)
    assert np.isfinite(Q.namco_wsg(440.0, 0.2, SR, wave=3)).all()
    assert Q.namco_wsg(440.0, 0.2, SR, wave=3, generic=True).shape == Q.namco_wsg(440.0, 0.2, SR, wave=3).shape


def test_procedural_grid_covers_every_class():
    g = P.procedural_grid()
    assert set(k for k in g if k != "summary") == {"aerodynamic", "biotic", "contact", "machine", "nature"}
    assert g["summary"]["classes"] == 65 and g["summary"]["covered"] == 65
    assert all(not v["missing"] for k, c in g.items() if k != "summary" for v in c.values())
    assert len(g["summary"]["techniques"]) == 9 and all(g["summary"]["techniques"].values())


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
    print(n, "ok")
