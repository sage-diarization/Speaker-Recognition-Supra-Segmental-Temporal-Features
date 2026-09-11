import numpy as np
import torch
import torchaudio


def _stft_magnitude(waveform, config):
    """STFT magnitude, (frames, nfft // 2 + 1), matching
    context/src/setup/utils.py::transform_audio's tf.signal.stft step,
    parameterized over config.window ('hann' or 'hamming') since different
    original models use different windows (CNN/RNN's DeepVoice front-end
    uses hann, ResNet's uses hamming).

    Frames are cut and windowed manually (rather than via torch.stft)
    because tf.signal.stft always frames by frame_length/frame_step and only
    then zero-pads each frame up to fft_length; torch.stft instead frames by
    n_fft whenever win_length != n_fft, which silently mismatches TF's frame
    count for ResNet's front-end (frame_length=400 samples, nfft=512)."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    window_fn = torch.hamming_window if config.window == "hamming" else torch.hann_window
    window = window_fn(config.frame_length)
    frames = waveform.unfold(-1, config.frame_length, config.frame_step) * window
    return torch.fft.rfft(frames, n=config.nfft).abs()


def compute_mel_spectrogram(waveform, config):
    """Magnitude mel-spectrogram, (frames, n_mels), matching
    context/src/setup/utils.py::transform_audio's MEL_SPECTROGRAM path
    (STFT -> linear mel filterbank)."""
    magnitude = _stft_magnitude(waveform, config)
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=magnitude.shape[-1],
        f_min=config.fmin,
        f_max=config.fmax,
        n_mels=config.n_mels,
        sample_rate=config.sample_rate,
    )
    mel = magnitude @ mel_fb
    return mel.numpy()


def compute_linear_spectrogram(waveform, config):
    """Raw magnitude spectrogram, (frames, nfft // 2 + 1), matching
    context/src/setup/utils.py::transform_audio's SPECTROGRAM path (no mel
    filterbank) -- ResNet's front-end (context/src/00_configs/01_transformation/ResNet.json)."""
    return _stft_magnitude(waveform, config).numpy()


def apply_drc(spectrogram):
    """Dynamic range compression, matches TRANSFORM['DYNAMIC_RANGE_COMPRESSION']
    in context/src/generator/preprocessing.py."""
    return 10 * np.log10(1 + 10000 * spectrogram)


def normalise_standardize(spectrogram):
    """Per-frame z-score across the frequency axis (not global/per-utterance),
    matches NORMALISATION['STANDARDISATION'] in context/src/generator/preprocessing.py."""
    mu = spectrogram.mean(axis=-1, keepdims=True)
    std = spectrogram.std(axis=-1, keepdims=True)
    return (spectrogram - mu) / (std + 1e-5)
