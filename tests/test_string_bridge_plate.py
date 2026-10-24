"""String-bridge-plate of van Walstijn & Mehes, DAFx-17 (genny.physical.sbp; instrument string_bridge_plate).

Run:  uv run --with pytest pytest tests/test_string_bridge_plate.py -q
"""
import numpy as np

from genny.physical import sbp

SR = 48000.0


def _db(y, a, b):
    return 20 * np.log10(np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2)) + 1e-30)


def test_every_preset_is_stable():
    for p in sbp.PRESETS:
        y = sbp.render(80.0, 0.8, SR, p)
        assert np.all(np.isfinite(y)) and 0 < np.max(np.abs(y)) < 1.0, p


def test_light_bridge_drains_the_string_faster():
    # their Fig. 4: R_bs = 0.6 couples string and plate more and the energy goes more quickly than with R_bs = 6
    drop = {p: _db(y, 0.0, 0.25) - _db(y, 1.2, 1.45)
            for p, y in ((p, sbp.render(100.0, 1.5, SR, p)) for p in ("body", "coupled"))}
    assert drop["coupled"] > drop["body"] + 10, drop


def test_cubic_springs_spread_energy_upwards():
    # their Fig. 5: eta = 1 (alpha 3) grows dense high partials that the linear coupling (eta = 0) does not
    hf = {}
    for eta in (0.0, 1.0):
        sbp.PRESETS["_t"] = dict(sbp.PRESETS["plate"], eta=eta)
        y = sbp.render(47.3, 1.0, SR, "_t")[int(0.3 * SR):]
        S = np.abs(np.fft.rfft(y)) ** 2
        hf[eta] = S[np.fft.rfftfreq(y.size, 1 / SR) > 2000].sum() / S.sum()
    del sbp.PRESETS["_t"]
    assert hf[1.0] > 100 * hf[0.0], hf


def test_sitar_bridge_breaks_the_young_helmholtz_law():
    # Siddiq 2012, Fig. 11: plucked at 1/5, an ideal string has no 5th partial; with the flat bridge it has one
    from genny.fdstring import sitar_fd
    f0 = 131.0
    h5 = {}
    for buzz in (0.0, 0.5):
        y = sitar_fd(f0, 1.0, 48000, 0.9, buzz)[int(0.3 * 48000):int(1.2 * 48000)]
        S = np.abs(np.fft.rfft(y * np.hanning(y.size), 1 << 18))
        f = np.fft.rfftfreq(1 << 18, 1 / 48000)
        h = [S[(f > k * f0 * 0.97) & (f < k * f0 * 1.03)].max() for k in (1, 2, 3, 5)]
        h5[buzz] = 20 * np.log10(h[3] / max(h))
    assert h5[0.0] < -40 and h5[0.5] > h5[0.0] + 20, h5          # (grid and dispersion leak a little without it)


def test_instrument_fundamental_is_in_tune():
    from genny import instruments as I
    from genny.notes import midi_to_freq
    sr = 44100
    for m in (28, 46, 64):
        f0 = midi_to_freq(m)
        y = I.render_note("string_bridge_plate", f0, 0.6, sr, 0.9)[int(0.1 * sr):]
        S = np.abs(np.fft.rfft(y * np.hanning(y.size), 1 << 18))
        f = np.fft.rfftfreq(1 << 18, 1 / sr)
        b = (f > 0.9 * f0) & (f < 1.1 * f0)
        assert abs(1200 * np.log2(f[b][np.argmax(S[b])] / f0)) < 12, m
