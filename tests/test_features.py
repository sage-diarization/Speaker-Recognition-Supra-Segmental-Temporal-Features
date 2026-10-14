import numpy as np
import pytest

from src.data.dataset import change_playback_speed, playback_sample_count
from src.data.features import (
    _bin_center_frequencies,
    apply_drc,
    compute_linear_spectrogram,
    compute_mel_spectrogram,
    normalise_standardize,
    vtlp_warp,
    warp_frequency_axis,
)
from src.config import TransformationConfig


def test_apply_drc_matches_formula():
    spectrogram = np.array([[0.0, 0.5], [1.0, 2.0]])
    expected = 10 * np.log10(1 + 10000 * spectrogram)
    np.testing.assert_allclose(apply_drc(spectrogram), expected)


def test_normalise_standardize_is_per_frame_across_frequency():
    spectrogram = np.array([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])  # (N=1, T=2, F=3)
    normalised = normalise_standardize(spectrogram)

    for t in range(2):
        frame = spectrogram[0, t]
        mu, std = frame.mean(), frame.std()
        expected = (frame - mu) / (std + 1e-5)
        np.testing.assert_allclose(normalised[0, t], expected)


def test_normalise_standardize_zero_mean_unit_ish_std_per_frame():
    rng = np.random.default_rng(0)
    spectrogram = rng.normal(5.0, 3.0, size=(2, 10, 16))
    normalised = normalise_standardize(spectrogram)
    np.testing.assert_allclose(normalised.mean(axis=-1), 0.0, atol=1e-6)


def test_mel_spectrogram_shape_for_one_second():
    config = TransformationConfig()
    waveform = np.zeros(config.sample_rate, dtype=np.float32)
    mel = compute_mel_spectrogram(waveform, config)
    assert mel.shape == (config.steps_per_second, config.n_mels)


def test_mel_spectrogram_scales_with_duration():
    config = TransformationConfig()
    waveform = np.zeros(config.sample_rate * 2, dtype=np.float32)
    mel = compute_mel_spectrogram(waveform, config)
    assert mel.shape[0] > config.steps_per_second


def test_linear_spectrogram_shape_matches_nfft_bins():
    # ResNet's front-end (context/src/00_configs/01_transformation/ResNet.json):
    # raw magnitude spectrogram, nfft // 2 + 1 frequency bins, no mel filterbank.
    config = TransformationConfig(type="linear", window="hamming", nfft=512,
                                   frame_length_s=0.025, frame_step_s=0.01)
    waveform = np.zeros(config.sample_rate, dtype=np.float32)
    spectrogram = compute_linear_spectrogram(waveform, config)
    assert spectrogram.shape == (config.steps_per_second, config.nfft // 2 + 1)


def test_transformation_config_num_freqs_depends_on_type():
    mel_config = TransformationConfig(type="mel", n_mels=128)
    assert mel_config.num_freqs == 128

    linear_config = TransformationConfig(type="linear", nfft=512)
    assert linear_config.num_freqs == 257


def test_hamming_window_produces_different_spectrogram_than_hann():
    hann_config = TransformationConfig(window="hann")
    hamming_config = TransformationConfig(window="hamming")
    rng = np.random.default_rng(0)
    waveform = rng.normal(size=hann_config.sample_rate).astype(np.float32)

    hann_spec = compute_mel_spectrogram(waveform, hann_config)
    hamming_spec = compute_mel_spectrogram(waveform, hamming_config)
    assert not np.allclose(hann_spec, hamming_spec)


@pytest.mark.parametrize("playback_speed", [0.8, 1.2, 2.0])
def test_change_playback_speed_scales_duration_and_pitch(playback_speed):
    sample_rate = 16000
    t = np.arange(sample_rate) / sample_rate
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    sped = change_playback_speed(waveform, playback_speed)

    assert len(sped) == playback_sample_count(len(waveform), playback_speed)
    assert len(sped) == pytest.approx(len(waveform) / playback_speed, abs=1)
    peak_hz = np.argmax(np.abs(np.fft.rfft(sped))) * sample_rate / len(sped)
    assert peak_hz == pytest.approx(440 * playback_speed, abs=2)


def test_change_playback_speed_one_is_identity():
    waveform = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    assert change_playback_speed(waveform, 1.0) is waveform


def _peak_spectrogram(config, peak_hz, num_frames=5):
    centers = _bin_center_frequencies(config)
    spectrogram = np.zeros((num_frames, len(centers)), dtype=np.float32)
    spectrogram[:, np.argmin(np.abs(centers - peak_hz))] = 1.0
    return spectrogram, centers


@pytest.mark.parametrize("kind", ["linear", "mel"])
@pytest.mark.parametrize("alpha", [0.9, 1.1])
def test_warp_frequency_axis_moves_peak_by_alpha_and_keeps_frames(kind, alpha):
    config = TransformationConfig(type=kind)
    spectrogram, centers = _peak_spectrogram(config, 1000.0)
    source_hz = centers[spectrogram[0].argmax()]

    warped = warp_frequency_axis(spectrogram, alpha, config)

    assert warped.shape == spectrogram.shape
    bin_width = np.diff(centers)[np.argmin(np.abs(centers - alpha * source_hz))]
    assert centers[warped[0].argmax()] == pytest.approx(alpha * source_hz, abs=bin_width)


def test_warp_frequency_axis_pins_zero_and_nyquist():
    config = TransformationConfig(type="linear")
    spectrogram = np.random.default_rng(0).random((3, config.nfft // 2 + 1)).astype(np.float32)
    for alpha in (0.9, 1.1):
        warped = warp_frequency_axis(spectrogram, alpha, config)
        np.testing.assert_allclose(warped[:, [0, -1]], spectrogram[:, [0, -1]], rtol=1e-6)


def test_vtlp_warp_equals_standardizing_the_warped_unstandardized_spectrogram():
    config = TransformationConfig()
    spectrogram = np.random.default_rng(0).random((4, config.n_mels)) * 30
    np.testing.assert_allclose(
        vtlp_warp(normalise_standardize(spectrogram), 1.1, config),
        normalise_standardize(warp_frequency_axis(spectrogram, 1.1, config)),
        atol=1e-4,
    )


def test_vtlp_alpha_one_is_identity():
    features = np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32)
    assert vtlp_warp(features, 1.0, TransformationConfig()) is features
