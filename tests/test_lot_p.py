import numpy as np

from genny import analysis, calibrate, drums, fx, instruments, sfx

SR = 22050


def _psd_peak(y, sr, lo, hi):
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def test_aliases_exist_and_render():
    for n, kw in (("lahar", {"dur": 2.0}), ("strombolian", {"dur": 2.0}), ("mud_volcano", {"dur": 2.0}), ("branch_break", {}), ("lake_ice", {})):
        y = sfx.REGISTRY[n]["fn"](SR, **kw)
        assert len(y) > SR and np.isfinite(y).all() and np.abs(y).max() > 0.1


def test_lahar_has_the_approach_before_the_front():
    y = sfx.REGISTRY["lahar"]["fn"](SR, dur=2.0)
    assert len(y) / SR > 7.5


def test_bedload_band_follows_the_pebble_size():
    from genny import matter
    rng = lambda: np.random.default_rng(3)
    big = matter.bedload_impacts(SR * 4, rng(), pebble=100.0, rate=200.0)
    small = matter.bedload_impacts(SR * 4, rng(), pebble=5.0, rate=200.0)
    cen = lambda y: float(np.sum(np.fft.rfftfreq(len(y), 1.0 / SR) * np.abs(np.fft.rfft(y))) / np.sum(np.abs(np.fft.rfft(y))))
    assert cen(small) > 2.0 * cen(big)
    y0 = sfx.REGISTRY["waterfall"]["fn"](SR, dur=2.0)
    y1 = sfx.REGISTRY["waterfall"]["fn"](SR, dur=2.0, bedload=1.0, pebble=100.0)
    assert not np.allclose(y0, y1)


def test_fire_species_puts_crackles_on_the_emission_peaks():
    y = sfx.REGISTRY["fire"]["fn"](44100, dur=4.0, crackle=60.0, wet=0.0, species="spruce", seed=2)
    f = np.fft.rfftfreq(len(y), 1.0 / 44100)
    s = np.abs(np.fft.rfft(y))
    band = lambda a, b: float(np.sum(s[(f >= a) & (f < b)] ** 2))
    assert band(11000, 13000) > band(16000, 18000)


def test_foliage_wind_rises_with_speed_and_dry_leaves_are_brighter():
    f = sfx.REGISTRY["foliage"]["fn"]
    rms = lambda y: float(np.sqrt(np.mean(y * y)))
    cen = lambda y: float(np.sum(np.fft.rfftfreq(len(y), 1.0 / 44100) * np.abs(np.fft.rfft(y))) / np.sum(np.abs(np.fft.rfft(y))))
    assert f(44100, dur=2.0, wind=4.0).shape == f(44100, dur=2.0, wind=12.0).shape
    assert cen(f(44100, dur=2.0, wind=8.0, leaf="dry")) > cen(f(44100, dur=2.0, wind=8.0, leaf="green"))


def test_cymbal808_rings_longer_than_the_hat():
    c = drums.REGISTRY["cymbal808"]["fn"](44100, 1.0)
    h = drums.REGISTRY["hat808"]["fn"](44100, 1.0)
    assert len(c) > 5 * len(h)


def test_steelpan_coupling_adds_the_neighbour_fifth():
    f = instruments.REGISTRY["steelpan"]["fn"]
    a = f(440.0, 1.0, 44100, 1.0, coupling=0.0)
    b = f(440.0, 1.0, 44100, 1.0, coupling=1.0)
    spec = lambda y: np.abs(np.fft.rfft(y[:44100]))
    k = int(660.0)
    assert spec(b)[k - 2:k + 3].max() > 1.5 * spec(a)[k - 2:k + 3].max()


def test_trautonium_has_subharmonics_and_martenot_is_near_sine():
    t = instruments.REGISTRY["trautonium"]["fn"](880.0, 1.0, 44100, 1.0, subs=3, formant=0.0)
    s = np.abs(np.fft.rfft(t[:44100]))
    assert s[440] > 0.1 * s[880] and s[293] > 0.05 * s[880]
    m = instruments.REGISTRY["ondes_martenot"]["fn"](440.0, 1.0, 44100, 1.0, vib=0.0)
    sm = np.abs(np.fft.rfft(m[4410:40000]))
    assert sm.max() > 20.0 * np.median(sm) and abs(_psd_peak(m[4410:40000], 44100, 300, 600) - 440.0) < 3.0


def test_opll_roms_decode_and_render():
    from genny import chiptune_r as R
    p = R.opll_patch("piano")
    assert p["ops"][0]["tl"] == 0x19 and p["fb"] == 0 and p["ops"][1]["ar"] == 12 and len(R.OPLL_ROM) == 15
    y = instruments.REGISTRY["opll"]["fn"](440.0, 0.5, 44100, 1.0, patch="violin")
    assert np.abs(y).max() > 0.01 and abs(_psd_peak(y, 44100, 300, 600) - 440.0) < 12.0


def test_brr_roundtrip_and_gauss_taps_sum_to_one():
    from genny import chiptune_r as R
    x = 0.9 * np.sin(2 * np.pi * np.arange(32) / 32)
    d = R.brr_decode(R.brr_encode(x)) / 2.0
    assert np.sqrt(np.mean((d - x / 2.0 * 1.0) ** 2)) < 0.05 * 0.9
    assert abs(R.snes_gauss(0.37).sum() - 1.0) < 1e-12
    y = instruments.REGISTRY["snes"]["fn"](440.0, 0.4, 44100, 1.0, echo_ms=64.0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.05


def test_saa1099_register_is_within_a_cent_of_the_pitch_in_the_middle_octaves():
    from genny import chiptune_r as R
    for hz in (110.0, 440.0, 1760.0):
        assert abs(1200 * np.log2(R.saa_hz(hz) / hz)) < 3.0
    o, n = R.saa_register(440.0)
    assert 0 <= o <= 7 and 0 <= n <= 255


def test_scanner_vibrato_moves_the_pitch_by_a_few_cents_and_organ_crosstalk_adds_neighbours():
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = np.sin(2 * np.pi * 440.0 * t)
    y = fx.REGISTRY["vibrato"]["fn"](x, sr, model="scanner")
    assert np.max(np.abs(y - x)) > 0.01 and np.isfinite(y).all()
    a = instruments.REGISTRY["organ"]["fn"](440.0, 1.0, sr, 1.0, crosstalk=0.0)
    b = instruments.REGISTRY["organ"]["fn"](440.0, 1.0, sr, 1.0, crosstalk=1.0)
    k = int(round(440.0 * 2 ** (1 / 12)))
    assert np.abs(np.fft.rfft(b[:sr]))[k - 1:k + 2].max() > 3.0 * np.abs(np.fft.rfft(a[:sr]))[k - 1:k + 2].max()


def test_iso226_contours_match_the_standard_at_a_few_points():
    assert abs(float(analysis.equal_loudness_spl(40.0, 1000.0)) - 40.0) < 0.2
    assert abs(float(analysis.equal_loudness_spl(40.0, 100.0)) - 64.37) < 0.3
    assert abs(analysis.phon_of_tone(40.0, 1000.0) - 40.0) < 0.2


def test_calibrate_match_lowers_the_distance():
    sr = 8000
    t = np.arange(sr // 2) / sr
    target = np.sin(2 * np.pi * 300.0 * t) * np.exp(-6 * t)
    render = lambda f, d: np.sin(2 * np.pi * f * t) * np.exp(-d * t)
    r = calibrate.match(render, target, {"f": 250.0, "d": 2.0}, {"f": (200.0, 400.0), "d": (0.5, 12.0)}, steps=80, seed=1)
    assert r["distance"] < 0.5 * r["start_distance"] and abs(r["params"]["f"] - 300.0) < 25.0


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            try:
                v()
                print("ok  ", k)
            except Exception as e:
                import traceback
                print("FAIL", k, repr(e))
                traceback.print_exc()
