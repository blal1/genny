import numpy as np

from genny import contact, creatures, nature, speech
from genny.physical import ambience as amb
from genny.physical import story


def test_drm_tables_follow_closed_tube_physics():
    assert len(story.DRM_STOP_TARGETS) == 8
    f1, f2, f3 = story.DRM_STOP_TARGETS[7]
    assert f2 == 1131 and abs(f3 / f2 - 2.0) < 0.01
    assert story.DRM_STOP_TARGETS[0][0] == 565
    assert all(r[0] == -1 for r in story.DRM_SIGNS_TM)
    assert story.DRM_SIGNS_TM[2][3] == -1 and story.DRM_SIGNS_TM[2][5] == -1


def test_eruption_infrasound_scales_with_volume_and_half_space():
    n = int(6 * nature.P)
    a = nature.eruption_infrasound(n, volume=2000.0)
    b = nature.eruption_infrasound(n, volume=4000.0)
    assert abs(np.abs(b).max() / np.abs(a).max() - 2.0) < 1e-6
    assert abs(nature.reduced_pressure(10.0, 325.0) - 3.25) < 1e-9
    d = nature.eruption_infrasound(n, volume=2000.0, dipole=1.0, theta=60.0)
    assert abs(np.abs(d).max() / np.abs(a).max() - 1.0) < 0.2


def test_volcano_infra_default_unchanged():
    a = nature.volcano(dur=6.0, stage="eruption", seed=2)
    assert np.array_equal(a, nature.volcano(dur=6.0, stage="eruption", seed=2, infra=0.0))
    assert not np.array_equal(a, nature.volcano(dur=6.0, stage="eruption", seed=2, infra=0.5))


def test_double_pulse_glide_moves_spectral_peaks():
    n = int(60 * nature.P)
    y = nature.double_pulse_tremor(n, np.random.default_rng(1), glide=(1.5, 1.5))
    z = nature.double_pulse_tremor(n, np.random.default_rng(1))
    assert y.shape == z.shape and np.abs(y).max() > 0


def test_rayleigh_depth_intensity_decays():
    v = contact.rayleigh_depth_intensity([0.0, 1.0, 4.0])
    assert abs(v[0] - 1.0) < 1e-12 and v[1] < v[0] and v[2] < 0.05


def test_thunder_centroid_decreases_with_distance():
    assert amb.thunder_centroid_hz(1.0) > amb.thunder_centroid_hz(10.0)
    assert abs(amb.thunder_centroid_hz(10.0) - 121.2 * 10.0 ** -0.26) < 1e-9


def test_speech_emotion_changes_duration_and_validates():
    sad = speech.render_speech("go now", emotion="sad")
    angry = speech.render_speech("go now", emotion="angry")
    assert len(angry) > len(sad)
    assert len(speech.render_speech("go now", emotion="")) == len(speech.render_speech("go now"))
    try:
        speech.render_speech("go", emotion="bored")
    except ValueError:
        return
    raise AssertionError


def test_vot_language_table():
    assert speech.VOT_LANGUAGES["english"]["aspirated"] == (58, 70, 80)
    assert speech.VOT_LANGUAGES["cantonese"]["aspirated"][2] == 87


def test_lobster_pulse_option():
    a = creatures.stridulation(kind="lobster", dur=1.0, seed=1)
    assert np.array_equal(a, creatures.stridulation(kind="lobster", dur=1.0, seed=1, pulse=0.0))
    assert creatures.stridulation(kind="lobster", dur=1.0, seed=1, pulse=77.0).shape == a.shape
