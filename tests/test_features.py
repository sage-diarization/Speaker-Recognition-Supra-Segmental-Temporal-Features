import numpy as np

from src.data.features import apply_drc, compute_linear_spectrogram, compute_mel_spectrogram, normalise_standardize
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
