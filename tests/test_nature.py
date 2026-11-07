"""Nature, disasters, fireworks: every variant renders; clicks are looked at by cause; the sourced numbers are measured on the rendered signal."""
import math

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, welch

from genny import nature as N
from genny.analysis import detect_clicks
from genny.spec import render_spec

SR = 22050
P = 48000
LSB = 2.0 ** -15


def spec(y, sr):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return np.fft.rfftfreq(len(y), 1 / sr), Y


def top(y, sr, lo, hi):
    f, Y = spec(y, sr)
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(Y[m])]


def share(y, sr, lo, hi):
    f, Y = spec(y, sr)
    return Y[(f >= lo) & (f < hi)].sum() / Y.sum()


def centroid(y, sr):
    f, Y = spec(y, sr)
    return (f * Y).sum() / Y.sum()


def rms(y):
    return float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0


def band(y, sr, lo, hi):
    f = np.fft.rfftfreq(len(y), 1 / sr)
    Y = np.fft.rfft(y)
    Y[(f < lo) | (f >= hi)] = 0.0
    return np.fft.irfft(Y, len(y))


def lmax(x, sr, weight=False):
    """FAST (125 ms) maximum in dB, written apart from the module's own: IEC 61672 A curve from its poles."""
    if weight:
        f = np.fft.rfftfreq(len(x), 1 / sr)
        f2 = np.maximum(f, 1e-3) ** 2
        a = 1.2588966 * 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.598997 ** 2) * np.sqrt((f2 + 107.65265 ** 2) * (f2 + 737.86223 ** 2)) * (f2 + 12194.0 ** 2))
        x = np.fft.irfft(np.fft.rfft(x) * a, len(x))
    m = int(0.125 * sr)
    c = np.cumsum(np.r_[0.0, x * x])
    return 10 * math.log10(((c[m:] - c[:-m]) / m).max())


def stray(y, sr):
    """Flagged samples with the shape of a fault, rather than of a hit. A cut: one sample's jump over half the level so far, and the sound gone after it.
    A spike: one sample standing 4 x above all its neighbours within 8 samples. detect_clicks also flags every band-limited grain (it is meant for smooth
    voices): those of this module were looked at sample by sample (L16.md), they swing over several samples. test_stray_catches_a_cut keeps this from being empty."""
    c = detect_clicks(y)
    w, bad = int(0.005 * sr), []
    for i in c[(c > 10) & (c < len(y) - 10)]:
        before = y[max(0, i - w):i - 1]
        if len(before) and abs(y[i + 1] - y[i]) > 0.5 * np.abs(before).max() > 0 and rms(y[i + 2:i + 2 + w]) < 0.25 * rms(before):
            bad.append(round(i / sr, 3))
        if abs(y[i]) > 4 * max(np.abs(y[i - 8:i]).max(), np.abs(y[i + 1:i + 9]).max()):
            bad.append(round(i / sr, 3))
    return bad


def test_stray_catches_a_cut():
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    y = 0.5 * np.sin(2 * np.pi * 200 * t) * (t < 0.5012)                  # a tone cut at full level
    assert stray(y, SR)
    z = 0.01 * np.sin(2 * np.pi * 300 * t)
    z[9000] += 0.3                                                       # a one-sample spike in a quiet tone
    assert stray(z, SR)
    s = np.exp(-np.maximum(t - 0.2, 0.0) / 0.01) * (t >= 0.2) * rng.standard_normal(SR) * 0.5     # a hard attack, what a blast is
    assert stray(s, SR) == []


# Steady beds: a flagged sample here is a fault. Rain, cracks and blasts have attacks of their own: they go through stray().
SMOOTH = [(N.geothermal, dict(kind="fumerolle")), (N.tornado, {}), (N.volcano, dict(stage="tremor")), (N.tsunami, dict(stage="retrait")), (N.tsunami, dict(stage="vague")),
          (N.singing_sand, {}), (N.foliage, dict(kind="herbe")), (N.earthquake_stage, dict(stage="premices")), (N.dirt, dict(kind="coulee")), (N.dirt, dict(kind="poussiere"))]
SMOOTH += [(N.wind_kind, dict(kind=k)) for k in N.WIND_KINDS] + [(N.rapids, {})]
STRUCK = [(N.firework, dict(kind=k)) for k in N.FIREWORK] + [(N.firecracker, {}), (N.lava_flow, {})] + [(N.geothermal, dict(kind=k)) for k in ("solfatare",)]
STRUCK += [(N.volcano, dict(stage=k)) for k in ("eruption", "retombees", "complet")] + [(N.tsunami, dict(stage="debris")), (N.collapse, {})] + [(N.crack, dict(kind=k)) for k in N.CRACK]
STRUCK += [(N.rain_on, dict(surface=k)) for k in N.RAIN] + [(N.fire_out, {}), (N.storm, {})] + [(N.foliage, dict(kind=k)) for k in ("feuilles", "branches", "passage")]
STRUCK += [(N.earthquake_stage, dict(stage=k)) for k in ("secousse", "repliques")] + [(N.dirt, dict(kind=k)) for k in ("pelle", "poignee")] + [(N.tree, dict(kind=k)) for k in N.TREE] + [(N.fountain, {})]
# geo.geyser and geo.mud_pot: geo._ring stops dead at exp(-6) of its level. Known, not ours to edit: asked of L21 in reports/lecture_profonde/suivi/L16.md.
GEO = [(N.geothermal, dict(kind=k)) for k in ("geyser", "boue")]


def _id(c):
    return f"{c[0].__name__}-{'-'.join(map(str, c[1].values()))}"


def _clean(y):
    assert np.isfinite(y).all() and 0.01 < np.abs(y).max() <= 1.0
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


@pytest.mark.parametrize("case", SMOOTH, ids=_id)
def test_smooth_variant_has_no_click(case):
    y = np.asarray(case[0](sr=SR, **case[1]))
    _clean(y)
    assert len(detect_clicks(y)) == 0


@pytest.mark.parametrize("case", STRUCK, ids=_id)
def test_struck_variant_has_no_stray_click(case):
    y = np.asarray(case[0](sr=P, **case[1]))              # at 48 kHz: at 22 kHz a 6 kHz grain has 3 samples a cycle and reads as a jump
    _clean(y)
    assert stray(y, P) == []


@pytest.mark.parametrize("case", GEO, ids=_id)
def test_geo_variant_renders(case):
    _clean(np.asarray(case[0](sr=SR, **case[1])))


def test_ring_helper_ends_at_zero_unlike_geo():
    y = np.zeros(4800)
    N._ring(y, 0.0, 1000.0, 0.003, 1.0)
    on = np.flatnonzero(y)
    assert abs(y[0]) < 1e-3 and abs(y[on[-1]]) < 1e-4 * np.abs(y).max()


# ---------------------------------------------------------------------------------- fireworks
def _firework(kind, d=N.REF_M):
    y = np.asarray(N.firework(sr=P, kind=kind, distance=d))
    h = N.FIREWORK[kind][2]
    r = math.hypot(d, h)
    tb = N.T_LAUNCH + N.RISE_S + (r - d) / N.AIR
    i = int(tb * P) - int(0.02 * P)
    return y, y[:int(0.6 * P)], y[i:i + int(1.5 * P)], tb


@pytest.mark.parametrize("kind,a_gap", [("mortier", 16.0), ("bouquet", 17.0), ("pop", 19.0)])
def test_firework_burst_over_launch_a_weighted(kind, a_gap):
    _, la, bu, _ = _firework(kind)
    assert lmax(bu, P, True) - lmax(la, P, True) == pytest.approx(a_gap, abs=1.0)


@pytest.mark.parametrize("kind,lf_gap", [("mortier", 3.0), ("pop", 3.0), ("bouquet", -2.0)])
def test_firework_low_band_launch_against_burst(kind, lf_gap):
    # G weighting is about constant across one narrow band, so the gap in dB(G) is the gap at 16-25 Hz
    _, la, bu, _ = _firework(kind)
    assert lmax(band(la, P, 14, 28), P) - lmax(band(bu, P, 14, 28), P) == pytest.approx(lf_gap, abs=1.5)


def test_firework_low_peak_in_16_25():
    _, la, bu, _ = _firework("mortier")
    assert 16 <= top(la, P, 5, 200) <= 25 and 16 <= top(bu, P, 5, 200) <= 25


def test_launch_2500_third_octave_10_db_up():
    _, la, _, _ = _firework("mortier")
    f, Y = spec(la, P)
    L = {c: 10 * np.log10(Y[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum()) for c in (500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000)}
    others = np.median([v for c, v in L.items() if c != 2500])
    assert L[2500] - others == pytest.approx(10.0, abs=3.0)


@pytest.mark.parametrize("d", [50.0, 102.0, 400.0])
def test_burst_arrives_after_the_climb_and_its_extra_path(d):
    y, _, _, tb = _firework("pop", d)
    e = np.abs(y)
    on = np.flatnonzero(e[int(2.0 * P):] > 0.25 * e[int(2.0 * P):].max())[0] / P + 2.0
    assert on == pytest.approx(tb, abs=0.08)


def test_distance_6_db_per_doubling_of_each_range():
    gap = lambda d: (lambda _, la, bu, __: lmax(bu, P, True) - lmax(la, P, True))(*_firework("pop", d))
    want = 20 * math.log10(408.0 / 102.0) - 20 * math.log10(math.hypot(408.0, 120.0) / math.hypot(102.0, 120.0))
    assert gap(408.0) - gap(102.0) == pytest.approx(want, abs=1.0)


def test_whistle_falls_6_to_2_khz_as_its_tube_lengthens():
    y = np.asarray(N.firework(sr=P, kind="sifflet"))
    a, b = y[int(0.5 * P):int(0.7 * P)], y[int(2.5 * P):int(2.7 * P)]
    t0, t1 = 0.6 - 0.1, 2.6 - 0.1
    T = N.RISE_S - 0.15
    want = lambda t: 6000.0 / (1.0 + 2.0 * t / T)                         # f = c / 4L, L growing at a steady rate
    assert top(a, P, 1500, 8000) == pytest.approx(want(t0), rel=0.1)
    assert top(b, P, 1500, 8000) == pytest.approx(want(t1), rel=0.1)
    assert N.whistle_freq(0.0, T) == 6000.0 and N.whistle_freq(T, T) == pytest.approx(2000.0)


def test_kinds_differ_as_described():
    long_tail = lambda k: (lambda y, _, __, tb: rms(y[int((tb + 1.0) * P):int((tb + 2.0) * P)]))(*_firework(k))
    assert long_tail("mortier") > 2 * long_tail("pop")
    b, _, _, tb = _firework("bouquet")
    assert len(find_peaks(np.convolve(np.abs(b[int(tb * P):]), np.ones(960) / 960, "same"), height=0.1, distance=int(0.2 * P))[0]) >= 4


def test_firecracker_count_and_gaps():
    y = np.asarray(N.firecracker(sr=P, count=10, rate=6.0, seed=2))
    e = np.convolve(np.abs(y), np.ones(240) / 240, "same")
    pk, _ = find_peaks(e, height=0.2 * e.max(), distance=int(0.025 * P))
    assert len(pk) == 10
    assert ((np.diff(pk) / P >= 0.025) & (np.diff(pk) / P <= 0.21)).all()


# ---------------------------------------------------------------------------------- tornado
def _psd(y, sr, seg=8.0):
    f, p = welch(y, sr, nperseg=int(seg * sr), noverlap=int(seg * sr / 2))
    return f, p


def test_tornado_infrasound_peak_at_8_3_hz():
    f, p = _psd(np.asarray(N.tornado(sr=SR, distance=500.0, dur=32.0, seed=0)), SR)
    for s in (1, 2):
        p = p + _psd(np.asarray(N.tornado(sr=SR, distance=500.0, dur=32.0, seed=s)), SR)[1]
    m = (f >= 5) & (f <= 14)
    # the paper's smoothing: 1/50 decade, about 0.4 Hz here
    k = max(1, int(0.4 / (f[1] - f[0])))
    s = np.convolve(p, np.ones(k) / k, "same")
    assert f[m][np.argmax(s[m])] == pytest.approx(8.3, abs=0.8)


def test_tornado_infrasound_q_and_overtones():
    # On the layer: at 44 Hz the 40-200 Hz roar sits on top of the last overtones in the full render.
    x = N._infrasound(int(60.0 * P), np.random.default_rng(3))
    f, p = _psd(x, P)
    k = max(1, int(0.5 / (f[1] - f[0])))
    s = np.convolve(p, np.ones(k) / k, "same")
    for f0 in (8.3, 18.0, 29.0, 36.0, 44.0):
        m = (f >= f0 - 2.0) & (f <= f0 + 2.0)
        assert f[m][np.argmax(s[m])] == pytest.approx(f0, abs=2.0)
    assert s[np.argmin(abs(f - 18))] > s[np.argmin(abs(f - 13))]
    assert s[np.argmin(abs(f - 29))] > s[np.argmin(abs(f - 23.5))]


def test_tornado_audible_band_40_200_centre_80_100():
    # Below 50 Hz the overtones of the infrasound (up to 44 Hz, Elbing) are on top of the roar: peak and -3 dB points of the roar on its layer, its weight in the render.
    x = N._tornado_roar(30 * P, np.random.default_rng(2))
    f, p = _psd(x, P, 1.0)
    s = np.convolve(p, np.ones(9) / 9, "same")
    m = (f >= 30) & (f <= 500)
    pk = f[m][np.argmax(s[m])]
    assert 80 <= pk <= 100
    for edge in (40.0, 200.0):
        assert 10 * np.log10(s[np.argmin(abs(f - edge))] / s.max()) == pytest.approx(-3.0, abs=1.5)
    y = np.asarray(N.tornado(sr=SR, distance=500.0, dur=30.0, seed=2))
    f, p = _psd(y, SR, 2.0)
    s = np.convolve(p, np.ones(5) / 5, "same")
    m = (f >= 50) & (f <= 2000)
    assert 70 <= f[m][np.argmax(s[m])] <= 110
    assert share(y, SR, 40, 200) > 2 * share(y, SR, 200, 2000)


def test_tornado_loses_its_top_and_debris_far_away():
    near, far = (np.asarray(N.tornado(sr=SR, distance=d, dur=6.0)) for d in (80.0, 3000.0))
    assert share(near, SR, 1000, 8000) > 3 * share(far, SR, 1000, 8000)


# ---------------------------------------------------------------------------------- volcano, tsunami
def test_tremor_is_below_20_hz():
    y = np.asarray(N.volcano(sr=SR, stage="tremor", dur=8.0))
    assert share(y, SR, 1, 20) > 0.5


def test_volcano_complete_goes_tremor_blast_fall():
    y = np.asarray(N.volcano(sr=SR, stage="complet", dur=16.0))
    assert share(y[:int(4.0 * SR)], SR, 0, 100) > 0.9
    pk = np.argmax(np.abs(y)) / SR
    assert 4.5 < pk < 6.0
    last = y[int(13.0 * SR):int(15.0 * SR)]
    assert share(last, SR, 200, 8000) > 0.3


def test_tsunami_stages():
    r, v, d = (np.asarray(N.tsunami(sr=SR, stage=k, dur=8.0)) for k in N.TSUNAMI)
    assert centroid(r, SR) < centroid(v, SR)
    assert rms(r[-SR:]) < 0.5 * rms(r[int(2.4 * SR):int(3.4 * SR)])
    assert rms(v[:SR // 2]) < 0.5 * rms(v[3 * SR:4 * SR])
    cr = lambda y: (np.abs(y).max() / rms(y))
    assert cr(d) > cr(v)


# ---------------------------------------------------------------------------------- singing sand
@pytest.mark.parametrize("um,hz,err", [(160, 105, 10), (210, 90, 10), (270, 75, 10), (340, 63, 5)])
def test_song_frequency_table_i(um, hz, err):
    y = np.asarray(N.singing_sand(sr=SR, grain=um, depth=8.0, dur=6.0, seed=1))
    got = top(y[int(1.0 * SR):int(3.5 * SR)], SR, 30, 200)
    assert got == pytest.approx(N.song_freq(um), abs=1.5)
    assert abs(N.song_freq(um) - hz) <= err


def test_song_stops_below_a_quarter_of_c():
    dur, depth = 8.0, 6.0
    y = np.asarray(N.singing_sand(sr=SR, grain=160, depth=depth, dur=dur, seed=1))
    f = N.song_freq(160)
    hc = N.SONG_C / 4.0 / f * 1e3                                   # mm: 2.32
    rise = 0.1 * dur
    t_stop = dur - (dur - rise) * hc / depth
    tone = band(y, SR, 0.8 * f, 1.2 * f)
    before, after = tone[int((t_stop - 1.2) * SR):int((t_stop - 0.2) * SR)], tone[int((t_stop + 0.4) * SR):int((t_stop + 1.4) * SR)]
    assert 20 * np.log10(rms(before) / rms(after)) > 12


def test_song_needs_a_thick_enough_layer():
    # Below c/4f the same avalanche only rustles
    f = N.song_freq(160)
    sing, thin = (np.asarray(N.singing_sand(sr=SR, grain=160, depth=d, dur=4.0, seed=1)) for d in (6.0, 2.0))
    assert share(sing, SR, 0.8 * f, 1.2 * f) > 0.7
    assert share(thin, SR, 0.8 * f, 1.2 * f) < 0.5 * share(sing, SR, 0.8 * f, 1.2 * f)
    assert 1500 < centroid(band(thin, SR, 1000, 8000), SR) < 6000


def test_song_takes_time_to_come_up():
    y = np.asarray(N.singing_sand(sr=SR, grain=160, depth=6.0, dur=8.0, seed=1))
    assert rms(y[:int(0.15 * SR)]) < 0.3 * rms(y[int(1.5 * SR):int(2.5 * SR)])


def test_grains_lock_only_with_coupling():
    m, f = 4000, 100.0
    for k, lo, hi in ((0.0, 0.0, 0.25), (2 * np.pi * f * 0.1, 0.8, 1.0)):
        _, r = N._kuramoto(m, f, np.full(m, k), np.random.default_rng(0))
        assert lo <= r[-500:].mean() <= hi


# ---------------------------------------------------------------------------------- rain
def test_window_impulse_response():
    x = np.zeros(int(0.5 * P))
    x[0] = 1.0
    h = N.glass_window(x)
    # nothing comes out before the 3.7 ms line has been read
    assert np.flatnonzero(np.abs(h) > 1e-12)[0] == pytest.approx(3.7e-3 * P, abs=1)
    # bounded and dying: the author warns it easily goes unstable; loop gain 0.61 with these bands is stable
    a, b = rms(h[:int(0.05 * P)]), rms(h[int(0.2 * P):int(0.25 * P)])
    assert b < 1e-3 * a
    # the bands the sidebands group around
    f, Y = spec(h, P)
    assert Y[(f >= 1000) & (f < 3500)].sum() > 0.8 * Y.sum()


def test_window_is_dispersive_sidebands_at_ring_offsets():
    # One pass through ring + band-pass: 2007 Hz band, 254 Hz ring: excitation at 2007 + 254 comes out at 2007
    t = np.arange(int(0.5 * P)) / P
    y = N.glass_window(np.sin(2 * np.pi * (2007.0 + 254.0) * t))
    f, Y = spec(y[int(0.1 * P):], P)
    near = lambda c: Y[(f > c - 12) & (f < c + 12)].sum()
    assert near(2007.0) > 5 * near(1630.0)


def test_drops_layer_gives_sparse_positive_spikes_above_2_khz():
    n = 4 * P
    y = N.drop_spikes(n, np.random.default_rng(0))
    raw = N.drop_spikes(n, np.random.default_rng(0), hip=1.0)
    assert rms(y) > 0 and np.abs(y).max() / rms(y) > 6
    # thresholded: off most of the time, so the median sits far under the spikes
    assert np.abs(raw).max() > 20 * abs(np.median(raw))
    # hip~ is one pole: the fig. 18.46 instance takes 12.3 dB off at 500 Hz, 1 dB at 4 kHz
    f, a = welch(y, P, nperseg=4096)
    _, b = welch(raw, P, nperseg=4096)
    at = lambda c: 10 * np.log10(a[np.argmin(abs(f - c))] / b[np.argmin(abs(f - c))])
    assert at(500.0) == pytest.approx(-12.3, abs=1.5) and at(4000.0) == pytest.approx(-1.0, abs=1.0)


def test_raindrops_threshold_thins_the_rain():
    d = lambda th: len(find_peaks(N.raindrops(2 * P, 70.0, th, np.random.default_rng(1)), height=1e-3, distance=int(0.002 * P))[0])
    assert d(0.35) < d(0.2)
    assert d(0.2) > 10


@pytest.mark.parametrize("a,b", [("vitre", "bois"), ("toit", "bois")])
def test_surfaces_differ(a, b):
    x, y = (np.asarray(N.rain_on(sr=SR, surface=k, dur=2.0)) for k in (a, b))
    assert abs(centroid(x, SR) - centroid(y, SR)) > 200


def test_window_rain_sits_in_its_band():
    y = np.asarray(N.rain_on(sr=SR, surface="vitre", dur=3.0))
    assert share(y, SR, 1000, 3500) > 0.6


def test_wetness_moves_ground_rain_from_low_drops_to_puddles():
    dry, wet = (np.asarray(N.rain_on(sr=SR, surface="sol", wetness=w, dur=3.0)) for w in (0.0, 1.0))
    assert share(dry, SR, 500, 2000) > 0.2
    # bubbles of 1.5-2 mm radius: above the dry drops
    assert centroid(wet, SR) > centroid(dry, SR)


def test_wind_drives_density():
    calm, windy = (np.asarray(N.rain_on(sr=SR, surface="toit", intensity=0.2, wind=w, dur=3.0)) for w in (0.5, 3.0))
    # hits overlap more: the peak stands less far above the level
    assert np.abs(windy).max() / rms(windy) < np.abs(calm).max() / rms(calm)


def test_gust_delay_is_distance_over_wind():
    assert N.gust_delay(30.0, 30.0) == 1.0
    for v in (30.0, 15.0):
        p = N._storm_parts(0.5, v, 24.0, 2)
        e = [uniform_filter1d(np.abs(p[k]), 4800)[::480] for k in ("flag", "trees", "wires")]
        for a, b in ((0, 1), (1, 2)):                                     # flag, one delay, trees, one delay, wires (fig. 18.71)
            c = [np.corrcoef(e[a][:len(e[a]) - k], e[b][k:])[0, 1] for k in range(0, 400)]
            assert np.argmax(c) * 480 / P == pytest.approx(30.0 / v, abs=0.4)


def test_storm_rain_goes_from_dry_to_puddles():
    p = N._storm_parts(0.5, 30.0, 20.0, 1)
    r = p["rain"]
    # dry: the low drop signature; wet: bubbles of about 2 mm, 1.6 kHz (matter.bubbles: 1 mm = 3.3 kHz)
    dry, wet = r[:int(5 * P)], r[int(15 * P):int(20 * P)]
    assert share(wet, P, 1000, 3000) > 1.5 * share(dry, P, 1000, 3000)
    assert rms(p["thunder"]) > 0


def test_fire_out_order():
    y = np.asarray(N.fire_out(sr=SR, dur=5.0))
    assert share(y[:SR], SR, 3000, 11000) < share(y[int(2.3 * SR):int(3.3 * SR)], SR, 3000, 11000)
    assert rms(y[int(4.5 * SR):]) < 0.5 * rms(y[int(2 * SR):int(3 * SR)])


# ---------------------------------------------------------------------------------- the rest: relations only, every number UNSOURCED
def test_lava_flow_follows_flow():
    a, b = (np.asarray(N.lava_flow(sr=SR, flow=f, dur=6.0)) for f in (0.0, 1.0))
    assert share(b, SR, 20, 120) > share(a, SR, 20, 120)


def test_geothermal_kinds():
    fu, ge, so, bo = (np.asarray(N.geothermal(sr=SR, kind=k, dur=8.0)) for k in N.GEOTHERMAL)
    assert centroid(fu, SR) > centroid(bo, SR)
    # `geo.geyser`: 5 cm collapses, about 65 Hz (Kedar), before the jet at 0.4 x dur
    assert share(ge[int(0.5 * SR):int(3.0 * SR)], SR, 30, 250) > 0.5
    assert rms(ge[int(3.5 * SR):int(5.0 * SR)]) > rms(ge[:SR])
    assert share(so, SR, 1000, 11000) > 0.05


def test_collapse_size_lowers_and_lengthens():
    a, b = (np.asarray(N.collapse(sr=SR, size=s)) for s in (0.5, 2.5))
    assert len(b) > len(a) and centroid(b, SR) < centroid(a, SR)


def test_collapse_rejects_unknown_material():
    with pytest.raises(ValueError):
        N.collapse(sr=SR, material="jelly")


def test_crack_kinds():
    br, pl, os_, gl = (np.asarray(N.crack(sr=SR, kind=k)) for k in N.CRACK)
    assert len(os_) < len(br) < len(pl)
    assert centroid(os_, SR) > centroid(pl, SR)
    # the ice's falling pew
    h = gl[np.argmax(np.abs(gl) > 0.05 * np.abs(gl).max()):]
    k = int(0.05 * SR)
    assert top(h[:k], SR, 200, 8000) > top(h[k:2 * k], SR, 200, 8000) > 0


def test_foliage_kinds():
    he, fe, br, pa = (np.asarray(N.foliage(sr=SR, kind=k, dur=2.0)) for k in N.FOLIAGE)
    assert centroid(he, SR) > centroid(fe, SR) > centroid(br, SR)
    assert len(pa) == len(he)


def test_earthquake_stage_in_rumble_band():
    pr, se, re = (np.asarray(N.earthquake_stage(sr=SR, stage=k, dur=8.0)) for k in N.QUAKE)
    assert 35 <= top(pr, SR, 10, 500) <= 90
    assert share(se, SR, 300, 8000) > share(pr, SR, 300, 8000)
    e = np.convolve(np.abs(re), np.ones(2205) / 2205, "same")
    assert e.min() < 0.1 * e.max()


def test_dirt_kinds():
    pe, po, co, du = (np.asarray(N.dirt(sr=SR, kind=k)) for k in N.DIRT)
    assert centroid(du, SR) > centroid(co, SR)
    assert rms(co[SR:2 * SR]) > 0.5 * rms(co)
    assert rms(po[-SR // 4:]) < rms(po[:SR // 4])
    assert rms(pe[int(0.5 * SR):int(0.8 * SR)]) < rms(pe[int(0.9 * SR):int(1.2 * SR)])


def test_tree_gaps_close_towards_the_break():
    y = np.asarray(N.tree(sr=P, kind="craquement", seed=1))
    e = np.convolve(np.abs(y), np.ones(96) / 96, "same")
    pk, _ = find_peaks(e[:int(3.0 * P)], height=0.002, distance=int(0.003 * P))
    t = pk / P
    assert (t > 2.0).sum() > 2 * (t < 1.0).sum()
    assert 2.95 < np.argmax(np.abs(y)) / P < 3.3


def test_tree_fall_ends_on_the_ground():
    y = np.asarray(N.tree(sr=SR, kind="chute"))
    a, b = y[:SR], y[int(2.6 * SR):int(3.1 * SR)]
    assert share(b, SR, 20, 150) > share(a, SR, 20, 150)
    assert 2.5 < np.argmax(np.abs(y)) / SR < 3.0


def test_wind_kinds():
    i, s, t, f = (np.asarray(N.wind_kind(sr=SR, kind=k, dur=4.0)) for k in N.WIND_KINDS)
    assert centroid(i, SR) < centroid(f, SR)
    assert share(i, SR, 900 * 2, 11000) < 0.1
    fq, Y = spec(s, SR)
    assert Y.max() / np.median(Y) > 100         # a whistle has a line
    assert rms(t) > 0


def test_rapids_turbulence():
    a, b = (np.asarray(N.rapids(sr=SR, turbulence=t, dur=3.0)) for t in (0.0, 1.0))
    assert share(b, SR, 20, 150) > share(a, SR, 20, 150)


@pytest.mark.parametrize("h", [1.0, 2.0, 4.0])
def test_fountain_water_lands_after_its_flight(h):
    y = np.asarray(N.fountain(sr=SR, height=h, dur=5.0))
    x = band(y, SR, 150, 900)
    e = np.convolve(x * x, np.ones(2205) / 2205, "same")
    on = np.flatnonzero(e > 0.2 * np.median(e[int(N.fall_time(h) * SR) + SR:]))[0] / SR
    assert on == pytest.approx(N.fall_time(h), abs=0.3)
    assert N.fall_time(2.0) == pytest.approx(1.277, abs=0.002)


def test_suikinkutsu_recipe_rings_after_the_drops():
    y, sr = render_spec(dict(N.RECIPES["suikinkutsu"]))
    y = np.asarray(y)
    y = y.mean(axis=1) if y.ndim == 2 else y
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1
    e = np.convolve(np.abs(y), np.ones(int(0.02 * sr)) / int(0.02 * sr), "same")
    pk = int(np.argmax(e))
    # the jar's ring holds on after a drop
    assert e[pk + int(0.15 * sr)] > 0.02 * e[pk]


def test_earthquake_s_wave_rattle_arrives_0_119_s_per_km_after_p():
    from genny import geo as G
    assert G.S_MINUS_P == pytest.approx(0.119, abs=0.001)
    assert G.s_delay(20.0, 14.0) == pytest.approx(2.38, abs=0.01)   # halved, it was 1.2 s
    assert G.s_delay(200.0, 14.0) == 7.0
