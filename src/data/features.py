import numpy as np
import torch
import torchaudio


def compute_mel_spectrogram(waveform, config):
    """Magnitude mel-spectrogram, (frames, n_mels), matching
    context/src/setup/utils.py::transform_audio (hann STFT -> linear mel filterbank)."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    window = torch.hann_window(config.frame_length)
    stft = torch.stft(
        waveform,
        n_fft=config.nfft,
        hop_length=config.frame_step,
        win_length=config.frame_length,
        window=window,
        center=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = stft.abs().transpose(0, 1)

    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=magnitude.shape[-1],
        f_min=config.fmin,
        f_max=config.fmax,
        n_mels=config.n_mels,
        sample_rate=config.sample_rate,
    )
    mel = magnitude @ mel_fb
    return mel.numpy()


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
