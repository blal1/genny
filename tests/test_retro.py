"""genny.retro: sfxr / ZzFX parity with the reference JavaScript, presets, PSG divider, tracker song.

Run:  uv run --with pytest pytest tests/test_retro.py -q     (or: uv run python tests/test_retro.py)

The parity fixtures were produced with node from `jsfxr/sfxr.js` (`SoundEffect.getRawBuffer().normalized`,
sound_vol 0.5) and `ZzFX/ZzFX.js` (`ZZFX.buildSamples`, Math.random() = 0.5): the sample count, the sum and
the sum of squares of the whole buffer, and every `step`-th sample.
"""
import time

import numpy as np

import genny.spec as SP
from genny import instruments as I
from genny import retro as R
from genny import sfx as S
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100
VOICES = ("sfxr_voice", "zzfx_voice", "psg")
JS_NAMES = ["p_env_attack", "p_env_sustain", "p_env_punch", "p_env_decay", "p_base_freq", "p_freq_limit", "p_freq_ramp",
            "p_freq_dramp", "p_vib_strength", "p_vib_speed", "p_arp_mod", "p_arp_speed", "p_duty", "p_duty_ramp",
            "p_repeat_speed", "p_pha_offset", "p_pha_ramp", "p_lpf_freq", "p_lpf_ramp", "p_lpf_resonance", "p_hpf_freq",
            "p_hpf_ramp"]

# fmt: off
JSFXR = {
    'square': dict(p={'wave_type': 0, 'p_env_sustain': 0.1, 'p_env_decay': 0.2, 'p_base_freq': 0.3, 'p_duty': 0.3, 'p_env_punch': 0.4, 'p_duty_ramp': 0.2}, n=5002, step=397, sum=-616.2888947709, sumsq=1377.4667232107072, sub=[
        1.167645742063369, -0.9532074930468444, -0.742422469285451, -0.6017530106149155, -0.534277538438037, -0.4679216734388718,
        -0.40268967927342675, 0.38271432471213823, 0.317532066659795, 0.2509520487798951, -0.15168802366946832, -0.09241807030980112,
        -0.034226144415335134]),
    'saw_arp_flange': dict(p={'wave_type': 1, 'p_duty': 1, 'p_env_sustain': 0.12, 'p_env_decay': 0.15, 'p_base_freq': 0.4, 'p_arp_mod': -0.4, 'p_arp_speed': 0.6, 'p_pha_offset': -0.3, 'p_pha_ramp': -0.1}, n=3692, step=397, sum=18.898215514084228, sumsq=702.4109928779844, sub=[
        0.6392906347641489, -0.11547860105942555, 0.8931634584210677, 0.6043101950905247, 0.295034092470059, 0.020626500297016164,
        -0.1518729242266762, -0.22256998879962006, -0.19157038945789182, -0.05417282474901033]),
    'sine_filters': dict(p={'wave_type': 2, 'p_env_attack': 0.05, 'p_env_sustain': 0.1, 'p_env_decay': 0.2, 'p_base_freq': 0.35, 'p_vib_strength': 0.3, 'p_vib_speed': 0.5, 'p_lpf_freq': 0.5, 'p_lpf_resonance': 0.6, 'p_lpf_ramp': -0.2, 'p_hpf_freq': 0.1, 'p_hpf_ramp': 0.2}, n=5252, step=397, sum=-7.19652706446284, sumsq=2016.177852574238, sub=[
        1.935436526464012e-05, -1.0379962915194585, 0.5993201332888525, -1.294582464298818, -0.4823826652048856, -0.10382660324448505,
        -0.7242005071313936, -0.5753145803874308, 0.2972041364532083, -0.34930641711114013, -0.2759953969823349, 0.20643306330419747,
        -0.10600491962100317, -0.012703174918329247]),
    'saw_cutoff_repeat': dict(p={'wave_type': 1, 'p_duty': 0.6, 'p_env_sustain': 0.2, 'p_env_decay': 0.3, 'p_base_freq': 0.5, 'p_freq_limit': 0.2, 'p_freq_ramp': -0.25, 'p_freq_dramp': -0.1, 'p_repeat_speed': 0.6, 'p_arp_mod': 0.3, 'p_arp_speed': 0.7}, n=13002, step=397, sum=2.510805358106839, sumsq=3922.179766458763, sub=[
        -1.1506932828902556, -0.3841964774853664, -1.1644597381835962, -0.5177620604167935, 0.5693382358170509, 0.5925471957993524,
        0.17982531182460618, 0.9990217249338793, -0.28631005196013487, -1.1001023059472685, -0.938648185999449, 0.7522232570561883,
        0.6794954388197836, -1.0016260656559082, 0.3603978280497352, -0.8506638635257155, 0.5233902782576294, 0.2220381958917517, 0.16921249058513785,
        -0.3774781795427648, 0.19677297957720846, -0.2873894025965609, -0.5735593213429921, -0.36130508754761165, 0.30881210904971135,
        -0.2871998761985578, -0.3419791763197846, 0.0359477162352773, -0.0645902869345888, 0.10855500088658342, -0.035573905755737205,
        0.0015937780832245307, 0.015099412272474048]),
}

ZZFX = {
    'example': dict(p=[1, 0, 925, 0.04, 0.3, 0.6, 1, 0.3, 0, 6.27, -184, 0.09, 0.17], n=41454, step=397, sum=47.790732795603326, sumsq=14139.847847471965, sub=[
        0, -0.15836874180765784, -0.33429168305392676, 0.6682915546096393, -0.6462936135292253, -0.6730302717581358, 0.9421665689067408,
        -0.8916416801525879, 0.6585924024674153, 0.5185266620403925, -0.7635513218586802, -0.9284517776414982, -0.4485857318737173,
        0.8278725774113009, 0.9952640565405292, 0.8703515399353439, 0.76637513334045, 0.723983196080831, 0.773668445864884, 0.8043570892053555,
        -0.9460331050819445, 0.5497193756725315, 0.8485239219708206, -0.8902037243370235, -0.5042360843794929, 0.9783181155880426,
        -0.6047100252611431, -0.9215485020291255, 0.6310374055551345, 0.5105317553544299, 0.8905611522756941, 0.8769631579836036, -0.5546643495493846,
        -0.9859752821925472, 0.4703707049077353, 0.9157736026924019, -0.8469930552308218, 0.41246840497989457, 0.8905489327368048, 0.8534760939115981,
        0.8106121184888212, 0.7560855885164113, 0.6796551129041641, 0.5541361682142477, -0.29721418243786735, -0.6144280025754627,
        -0.7594810741188969, -0.8545398529417659, 0.30856038295848526, -0.3834382308883991, 0.2276272378592491, 0.46197680810174435,
        -0.6205783998771812, 0.7421434683820815, -0.634653305882396, -0.4244457386069181, 0.7207023686191448, -0.5827714176793264, 0.6794232087332649,
        -0.5049739288998326, -0.38662342026457885, 0.5759415733801657, -0.6116719152923225, 0.4937002453037128, -0.3134501028475739,
        -0.35091068551965643, 0.43304893448021387, 0.530836923942292, 0.44774748063662434, -0.19710938039501993, -0.4131060611365941,
        -0.46746295263415455, -0.4829859769289732, -0.4683194424457234, -0.4446321702266887, -0.38842901331417423, -0.4259238694255709,
        0.24839781101585134, 0.34632759870055246, -0.31270224282914555, -0.27011812126681933, 0.3182550151532803, 0.2218830370026856,
        -0.2914420532683269, -0.22813344231550642, 0.22956757006911324, 0.18841499792008687, -0.2058764570149745, -0.2225696652529856,
        0.15007412320852492, 0.19362146238141392, -0.16488313540443017, -0.10319923932487667, 0.1500186997585216, -0.15562978395970384,
        -0.07732003844222386, 0.051961203792275854, 0.07360971659514089, 0.0770667644941447, 0.07404608176726, 0.06566318931929252,
        0.044417361916103903, 0.02269809903361055, -0.014362196587265825, -0.005927173326515181]),
    'square_all': dict(p=[1.5, 0, 330, 0.01, 0.08, 0.1, 5, 0.7, 0, 0, 0, 0, 0.05, 0, 30, 0.05, 0.05, 0.6, 0.05, 0.3, -800], n=12789, step=397, sum=1406.340579186101, sumsq=3060.8995629272363, sub=[
        0, -0.6282293860335095, 0.6079459851694233, 0.419518687173544, -0.4688621884308052, -0.26221758821794094, 0.35829948541299494,
        0.3249868351031001, 0.37791519613036006, 0.09828191932224634, -0.2611736088463482, -0.24080342019196618, -0.3873047166648751,
        0.5133607834883293, 0.41138861141221184, 0.27588812950347896, -0.3256191858586321, -0.31205168090182844, 0.05335113841444675,
        -0.2221447900370861, 0.40422579148771476, 0.5249505996070346, 0.2425458191018517, -0.3935519727715331, -0.08660864163002438,
        -0.7941963155856199, 0.09237741518203318, 0.1921527724781129, 0.3440493961923246, -0.05958670072223823, 0.11004794087521069,
        -0.18965438547775854, -0.018034584679182716]),
    'saw_hp_slide': dict(p=[1, 0, 440, 0, 0.1, 0.1, 2, 1.5, -2, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 500], n=8829, step=397, sum=0.017604049310100123, sumsq=1354.5350293455192, sub=[
        0, 0.0635780032927946, -0.019552467363652744, 0.016681533415406957, -0.8657495924603874, 0.06356003886811593, 0.4348139668673234,
        0.04621277918373824, -0.873729435493656, -0.12764389191597153, 0.043430479417864355, -0.034891673241405075, 1.6179665362794453,
        0.7548087002894812, 1.3023322792762966, 0.00024752089351681326, -0.023420177770488076, -0.00928822169522564, 0.07578988228574415,
        0.003918031002878566, -0.019024069420321456, -0.001444152196538781, -0.00028799735280058596]),
    'tan_jump_repeat': dict(p=[0.8, 0, 150, 0.02, 0.1, 0.15, 3, 2, 1, 0, 120, 0.03, 0.06], n=11907, step=397, sum=11.284070983353793, sumsq=2544.7483234322926, sub=[
        0, -0.36009070294784584, -0.7201814058956917, 0.8, -0.0964284436040971, 0.12458715608721789, 0.8, -0.8, -0.8, -0.8, -0.8,
        3.3234809449864653e-05, 0.8, -0.06673434955404156, 0.13029675561493798, 0.6433910637348789, 0.671806500377929, 0.6237944066515495,
        -0.006162946932988885, -0.5277702191987906, 0.46962273073854244, 0.43174603174603177, -0.2762734617490112, 0.10367900292686072,
        0.019456115463030905, -0.14604059011062628, -0.19168556311413454, 0.1436734693877551, -0.09566137566137567, 0.016122633895725245]),
    'noise_head': dict(p=[1, 0, 60, 0, 0.05, 0.1, 4, 1, 0, 0, 0, 0, 0, 0.5], n=6624, step=61, sum=None, sumsq=724.9730511465041, sub=[
        0, 0.16633760300952216, 0.9294384904954812, -0.5090549270912228, 0.332733626097146, -0.22564691136880255, 0.9990799778559316,
        0.2990510138649674, -0.31239948200196854, -0.31014843792219926, -0.2672925818512846, -0.7001319092289201, -0.11895850799192947,
        -0.7063112051427535, 0.9814733021657295, -0.6249090726239707, -0.4808916861645477, 0.715867506773995, -0.7635680415052115,
        -0.9944286303511828, 0.3885552087027384, 0.7949600486085568, -0.9470717024485698, -0.9061736424070288, 0.7498666965626997]),
}

# fmt: on


def cents_off(y, f0, sr=SR):
    """Pitch error of a rendered note, from the autocorrelation peak next to the expected period
    (same method as tests/test_instruments.py)."""
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


def _clean(y, what, lo=1e-4, hi=1.5):
    assert y.ndim == 1 and np.all(np.isfinite(y)), what
    assert lo < np.max(np.abs(y)) < hi, (what, float(np.max(np.abs(y))))
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (what, "does not start/end at zero", float(y[0]), float(y[-1]))


def _compare(x, f, what):
    sub = np.array(f["sub"])
    err = float(np.max(np.abs(x[::f["step"]][:sub.size] - sub)))
    assert x.shape[0] == f["n"], (what, x.shape[0], f["n"])
    assert err <= 1e-6, (what, err)
    if f["sum"] is not None:
        assert abs(x.sum() - f["sum"]) < 1e-6 and abs((x * x).sum() - f["sumsq"]) < 1e-6, what
    return err


# ---------------------------------------------------------------------------- parity with the JS references
def test_jsfxr_parity():
    worst = 0.0
    for name, f in JSFXR.items():
        p = dict(R._KDEF)
        p.update({R.KNOBS[JS_NAMES.index(k)]: v for k, v in f["p"].items() if k != "wave_type"})
        x = R.sfxr_core(f["p"]["wave_type"], np.array([p[k] for k in R.KNOBS]), 0.5, 0, 10 ** 6, 0, 0.0)
        worst = max(worst, _compare(x, f, name))
    print(f"  jsfxr parity: max |diff| = {worst:.1e} over {len(JSFXR)} parameter sets")


def test_zzfx_parity():
    """`noise_head` has the noise slot on: only its first 1500 samples are stored (see `zzfx_core`)."""
    worst = 0.0
    for name, f in ZZFX.items():
        a = np.array(list(f["p"]) + R.ZZ_DEF[len(f["p"]):], dtype=np.float64)
        worst = max(worst, _compare(R.zzfx_core(a, 44100.0, 0.5, 10 ** 7), f, name))
    print(f"  ZzFX parity: max |diff| = {worst:.1e} over {len(ZZFX)} parameter sets")


def test_tone_preset_is_440_hz_for_one_second():
    for sr in (44100, 48000, 22050):
        y = S.render_sfx("sfxr", sr, preset="tone")
        assert abs(y.shape[0] / sr - 1.0) < 0.001, (sr, y.shape[0] / sr)
        up = np.nonzero((y[:-1] < 0) & (y[1:] >= 0))[0]
        up = up[(up > 0.05 * sr) & (up < 0.95 * sr)]
        z = up + y[up] / (y[up] - y[up + 1])                       # interpolated upward zero crossings
        f = (z.size - 1) * sr / (z[-1] - z[0])
        cents = 1200 * np.log2(f / 440.0)
        assert abs(cents) < 2.0, (sr, f)
        print(f"  tone preset at {sr}: {f:.3f} Hz ({cents:+.2f} cents), {y.shape[0] / sr:.4f} s")


# ---------------------------------------------------------------------------- sfx: every preset x 20 seeds
def test_presets_finite_and_bounded():
    for sr in (44100, 22050):
        for preset in R.SFXR_PRESETS:
            for seed in range(20):
                y = S.render_sfx("sfxr", sr, preset=preset, seed=seed)
                _clean(y, ("sfxr", preset, seed, sr))
                assert abs(y.mean()) < 0.03 and y.shape[0] <= 3.0 * sr + 2, ("sfxr", preset, seed)
        for preset in R.ZZFX_PRESETS:
            for seed in range(20):
                _clean(S.render_sfx("zzfx", sr, preset=preset, seed=seed), ("zzfx", preset, seed, sr))
    for seed in range(20):                                           # the Bfxr extras on top of the presets
        y = S.render_sfx("sfxr", SR, preset="random", seed=seed, engine="bfxr", wave=R.SFXR_WAVES[seed % 9],
                         harmonics=seed % 5, crush=0.1 * (seed % 4), crush_sweep=0.3, compression=0.5, mutate=2)
        _clean(y, ("bfxr", seed))


def test_sfxr_semantics():
    a = S.render_sfx("sfxr", SR, preset="pickup", seed=3)
    assert np.array_equal(a, S.render_sfx("sfxr", SR, preset="coin", seed=3, mutate=0)), "not deterministic / alias"
    assert not np.array_equal(a, S.render_sfx("sfxr", SR, preset="pickup", seed=4))
    assert not np.array_equal(a, S.render_sfx("sfxr", SR, preset="pickup", seed=3, mutate=1))
    assert np.array_equal(S.render_sfx("sfxr", SR, preset="explosion", seed="boss"), S.render_sfx("sfxr", SR, preset="explosion", seed="boss"))
    p = R.sfxr_params("laser", 5, 3, None, decay=0.25)
    assert p["decay"] == 0.25, "an explicit knob must win over preset and mutate"
    assert R.sfxr_params(wave="saw")["duty"] == 1.0 and R.sfxr_params()["duty"] == 0.0
    f = 8 * 44100 / np.floor(100.0 / (R.hz_to_knob(440.0) ** 2 + 0.001))
    assert abs(1200 * np.log2(f / 440.0)) < 1.0, f
    raw = S.render_sfx("sfxr", SR, wave="square", hz=880, soften=0.0)
    soft = S.render_sfx("sfxr", SR, wave="square", hz=880)
    assert brightness(soft, SR)["above_5k"] < 0.5 * brightness(raw, SR)["above_5k"], "soften must round the top end"
    y = S.render_sfx("sfxr", SR, slide=-0.3, freq=0.5, freq_limit=0.3, sustain=0.9)     # ends at the frequency cutoff
    assert y.shape[0] < 0.5 * SR
    assert S.render_sfx("sfxr", SR, sustain=1, decay=1, max_dur=0.5).shape[0] == int(0.5 * SR)
    for bad in (dict(wave="whistle"), dict(preset="nope"), dict(engine="x"), dict(nope=1)):
        try:
            S.render_sfx("sfxr", SR, **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")
    z = S.render_sfx("zzfx", SR, p=[None, 0, 925, 0.04, 0.3, 0.6, 1, 0.3, None, 6.27, -184, 0.09, 0.17])
    assert z.shape[0] == 41454 and np.array_equal(z, S.render_sfx("zzfx", SR, p=[1, 0, 925, 0.04, 0.3, 0.6, 1, 0.3, 0, 6.27, -184, 0.09, 0.17]))
    _clean(S.render_sfx("zzfx", SR, frequency=440, sustain=0.2, volume=0.01, filter=99999, shape_curve=-3), "degenerate slots")


# ---------------------------------------------------------------------------- PSG (AY-3-8910)
def test_psg_tone_frequency_is_clock_over_16_tp():
    worst = 0.0
    for clock in (1_773_400.0, 2_000_000.0, 1_000_000.0):
        for f in (110.0, 440.0, 1000.0, 2000.0):
            chip = clock / (16 * R.psg_tone_period(f, clock))
            for sr in (44100, 48000):
                c = float(cents_off(R.psg(f, 0.6, sr, clock=clock, bright=1.0), chip, sr))
                worst = max(worst, abs(c))
                assert abs(c) < 1.0, (clock, f, sr, c)
    # the divider grid is audible up high: 2000 Hz -> TP 55 -> 2015.2 Hz, 13.1 cents sharp
    assert R.psg_tone_period(2000.0) == 55 and abs(1200 * np.log2(1_773_400 / (16 * 55) / 2000.0) - 13.1) < 0.1
    assert abs(cents_off(R.psg(2000.0, 0.6, SR, bright=1.0), 2000.0) - 13.1) < 1.0
    x = R._psg_core(4000, 106, 1, 1, -1, 15, 0.0, 0, R.PSG_DAC)      # the raw kernel: exact period in clock/8 samples
    assert np.all(np.diff(np.nonzero(np.diff(x) != 0)[0]) == 106), "tone must flip every TP samples at clock/8 (period 16*TP clocks)"
    print(f"  psg: measured pitch within {worst:.2f} cents of clock/(16*TP) (3 clocks x 4 notes x 2 rates)")


def test_psg_dac_envelope_and_noise():
    d = R.PSG_DAC
    assert d[0] == 0 and d[15] == 1 and np.allclose(d[2:] / d[1:-1], 2 ** 0.5), "16 log steps, 3 dB each"

    def idx(shape, n=80):                                  # DAC index per envelope step (EP = 1: 2 samples per step)
        return [int(np.argmin(np.abs(d - v))) for v in R._psg_core(n * 2, 10 ** 6, 1, 1, shape, 15, 0.0, 0, d)[::2]]

    dn, up, z, h = list(range(15, -1, -1)), list(range(16)), [0] * 16, [15] * 16
    want = {0: dn + z * 4, 4: up + z * 4, 8: dn * 5, 9: dn + z * 4, 10: (dn + up) * 2 + dn, 11: dn + h * 4,
            12: up * 5, 13: up + h * 4, 14: (up + dn) * 2 + up, 15: up + z * 4}
    for shape, seq in want.items():
        assert idx(shape) == seq, (shape, idx(shape)[:40])
    assert all(idx(s) == want[0] for s in (1, 2, 3)) and all(idx(s) == want[4] for s in (5, 6, 7))
    # env_hz = ramps per second: shape 8 at 10 Hz repeats every 0.1 s
    y = np.abs(R.psg(220.0, 1.0, SR, shape=8, env_hz=10.0, bright=1.0))[:SR]
    blocks = y.reshape(-1, 441).max(axis=1)                # 10 ms peak envelope
    blocks = blocks - blocks.mean()
    ac = np.correlate(blocks, blocks, "full")[blocks.size - 1:]
    assert int(np.argmax(ac[5:20])) + 5 == 10, int(np.argmax(ac[5:20])) + 5
    # 17-bit LFSR: maximal length (2^17 - 1 states) shows the taps give a proper noise sequence
    s, n = 1, 0
    while True:
        s = (s >> 1) | (((s ^ (s >> 3)) & 1) << 16)
        n += 1
        if s == 1:
            break
    assert n == 2 ** 17 - 1, n
    hi, lo = (brightness(R.psg(440.0, 0.3, SR, noise=1.0, noise_period=k, bright=1.0), SR)["centroid"] for k in (1, 31))
    assert hi > 2 * lo, ("a longer noise period must lower the noise", hi, lo)


# ---------------------------------------------------------------------------- instruments (mirrors test_instruments.py)
def test_voices_shapes_and_levels():
    for name in VOICES:
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                _clean(I.render_note(name, midi_to_freq(m), dur, sr, vel), (name, m, dur, sr))
    for wave in R.SFXR_WAVES:
        _clean(I.render_note("sfxr_voice", 330.0, 0.2, SR, 0.8, wave=wave, harmonics=3, vib_depth=0.2, lpf=0.6, resonance=0.5), wave)
    for shape in range(6):
        _clean(I.render_note("zzfx_voice", 330.0, 0.2, SR, 0.8, shape=shape, shape_curve=0.7), shape)
    _clean(I.render_note("zzfx_voice", 330.0, 0.5, SR, 0.8, p=[2, 0.05, 99, 0, 0.02, 0.2, 2, 2], hold=False), "zzfxm semantics")
    for shape in range(-1, 16):
        for sr in (44100, 22050):
            _clean(I.render_note("psg", 220.0, 0.4, sr, 0.8, shape=shape, env_hz=7.0, noise=0.3), ("psg", shape))
    _clean(I.render_note("psg", 110.0, 0.4, SR, 0.8, shape=10, env_hz=110.0), "buzz bass")


def test_voices_in_tune():
    worst = {}
    for name in VOICES:
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in (lo, (lo + hi) // 2, hi):
            for sr in (44100, 48000):
                c = float(cents_off(I.render_note(name, midi_to_freq(m), 0.6, sr, 0.9), midi_to_freq(m), sr))
                worst[name] = max(worst.get(name, 0.0), abs(c))
                assert abs(c) < 10, (name, m, sr, round(c, 1))
    for wave in ("saw", "sine", "triangle", "whistle"):
        assert abs(cents_off(I.render_note("sfxr_voice", 523.25, 0.6, SR, 0.9, wave=wave), 523.25)) < 10, wave
    for shape in (1, 2, 5):
        assert abs(cents_off(I.render_note("zzfx_voice", 523.25, 0.6, SR, 0.9, shape=shape), 523.25)) < 10, shape
    print("  worst tuning error (cents): " + ", ".join(f"{k} {v:.2f}" for k, v in worst.items()))


def test_voices_not_piercing_and_level_matched():
    for name in VOICES:
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))
        levels = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
                  for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        assert abs(sum(levels) / 3 + 12.0) < 2.0, (name, [round(v, 1) for v in levels])
        print(f"  {name}: {sum(levels) / 3:.1f} dB K-weighted at vel 0.9")


# ---------------------------------------------------------------------------- song layer
SONG = {"type": "song", "bpm": 120, "rows": 4,
        "instruments": {"lead": {"zzfx": [1, 0, 220, 0.01, 0.1, 0.2, 1, 1.5], "pan": -0.5},
                        "bass": {"sfxr": {"wave": "triangle"}, "gain": 0.8},
                        "arp": {"inst": "psg", "params": {"shape": 9, "env_hz": 6}},
                        "hat": {"drum": "hihat", "gain": 0.3}},
        "patterns": {"A": {"lead": "C-5 . E-5 . | G-5 . ^^ . | C-6@0.5 . . . | G-5 . . .",
                           "bass": "C-2 . . . . . . . G-2 . . . . . . .",
                           "arp": "C-4 E-4 G-4 C-5 C-4 E-4 G-4 C-5",
                           "kick": "x . . . x . . . x . . . x . X .", "hat": ". . x . . . x . . . x . . . x o"},
                     "B": {"lead": "F#4 . . . ^^", "kick": "x . . . x . . ."}},
        "order": ["A", "B", "A"]}


def test_song_renders_to_the_exact_length():
    rows = 16 + 8 + 16
    for sr in (44100, 48000, 22050):
        y, at = SP.render_layer(SONG, sr)
        assert y.shape == (int(round(rows * 0.125 * sr)), 2) and np.all(np.isfinite(y)) and at == 0.0, (sr, y.shape)
        assert 1e-3 < np.max(np.abs(y)) < 4.0 and np.max(np.abs(y[-1])) < 1e-6
    assert np.sqrt(np.mean(y[:, 0] ** 2)) != np.sqrt(np.mean(y[:, 1] ** 2)), "pan must make the channels differ"
    y, _ = SP.render_layer(dict(SONG, bpm=90, rows=3, tail=0.5), SR)
    assert y.shape[0] == int(round(rows * (60 / 90 / 3) * SR)) + int(0.5 * SR)
    mono, _ = SP.render_layer({"type": "song", "patterns": {"A": {"chip": "C-4 . . ^^ . . . ."}}}, SR)   # no bpm: 1 beat = 1 s
    assert mono.shape == (2 * SR,)
    row = SR // 4
    assert np.max(np.abs(mono[10:3 * row - 400])) > 0.05 and np.max(np.abs(mono[3 * row + 4000:])) < 1e-4, "'^^' must end the note"
    full, _ = SP.render_spec({"bpm": 120, "layers": [SONG], "trim": False})
    assert full.shape[0] == int(round(rows * 0.125 * SR))
    for bad in ({"patterns": {"A": {"nope": "C-4"}}}, {"patterns": {"A": {"chip": "x"}}}, {"patterns": {"A": {"chip": "C-4"}}, "order": ["Z"]}):
        try:
            SP.render_layer(dict(bad, type="song"), SR)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_song_zzfxm_import():
    data = [[[1, 0, 220, None, 0.05, 0.1, 1], [2, 0, 440, None, None, 0.05, 2]],
            [[[0, -1, 13, None, None, None, 25, None, -1, None], [1, 1, None, None, 20.5, None, None, None, 20, None]],
             [[1, 0, 1, 5, 8, 13]]],
            [0, 1, 0], 150]
    beat = int(44100 / 150 * 60) >> 2                      # zzfxm.js: beatLength = zzfxR / BPM * 60 >> 2
    y, _ = SP.render_layer({"type": "song", "format": "zzfxm", "data": data}, SR)
    assert y.shape == ((8 + 4 + 8) * beat, 2) and np.all(np.isfinite(y)), y.shape
    assert np.max(np.abs(y[:beat, 1])) < 1e-9 < np.max(np.abs(y[:beat, 0])), "pan -1 is left only"
    ev = R._zzfxm_tracks(data)[0]
    assert abs(ev[0][0][1] - 220 * 2 ** (1 / 12)) < 1e-9 and ev[0][2][1] < 0, "note 12 is the instrument's own frequency; -1 = off"
    assert ev[1][0][2] == 0.5 and ev[1][1][2] == 1.0, "the fraction of a note is its attenuation"
    y48, _ = SP.render_layer({"type": "song", "format": "zzfxm", "data": data}, 48000)
    assert y48.shape[0] == int(round(20 * beat * 48000 / 44100))


def test_speed():
    for name, fn in (("sfxr", lambda: S.render_sfx("sfxr", 48000, preset="tone", engine="bfxr", harmonics=4)),
                     ("zzfx", lambda: S.render_sfx("zzfx", 48000, sustain=1.0, noise=0.2, filter=-900)),
                     ("sfxr_voice", lambda: I.render_note("sfxr_voice", 220.0, 1.0, 48000)),
                     ("zzfx_voice", lambda: I.render_note("zzfx_voice", 220.0, 1.0, 48000)),
                     ("psg", lambda: I.render_note("psg", 220.0, 1.0, 48000, shape=10, noise=0.5))):
        fn()
        t = time.perf_counter()
        y = fn()
        dt = time.perf_counter() - t
        assert dt < 2.0 and y.shape[0] > 0.9 * 48000, (name, dt)
        print(f"  {name}: {dt * 1000:.0f} ms for {y.shape[0] / 48000:.2f} s")


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
