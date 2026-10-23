import functools

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi


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


@functools.cache
def _w2vbert_feature_extractor():
    # Imported lazily: transformers is only needed for this front-end.
    # padding_value=1 matches facebook/w2v-bert-2.0's preprocessor_config.json
    # (it fills the odd last frame's stacked half), so no Hub access is needed.
    from transformers import SeamlessM4TFeatureExtractor

    return SeamlessM4TFeatureExtractor(padding_value=1.0)


def compute_w2vbert_features(waveform, config):
    """w2v-BERT 2.0's input features, (frames, 160): see
    TransformationConfig.type. Already normalized (per mel bin, over the
    utterance), so featurize_waveform applies no DRC/standardization."""
    output = _w2vbert_feature_extractor()(waveform, sampling_rate=config.sample_rate, return_tensors="np")
    return output["input_features"][0]


def compute_wespeaker_fbank(waveform, config):
    """WeSpeaker's input features, (frames, 80): Kaldi fbank of the
    16-bit-scaled waveform, hamming window, no dither (wespeaker/cli/speaker.py's
    compute_features), minus its mean over the utterance (CMN)."""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0) * (1 << 15), num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, sample_frequency=config.sample_rate, window_type="hamming",
    )
    return (fbank - fbank.mean(dim=0)).numpy()


def compute_wavlm_cnn_features(waveform, config):
    """WavLM's convolutional feature encoder output, (frames, 512), of the
    utterance layer-normalized as a whole (WavLM Large's cfg.normalize) --
    the last representation before the transformer mixes frames, see
    src/models/pretrained_wavlm.py."""
    # Imported lazily: the encoder weights come from the WavLM checkpoint.
    from ..models.pretrained_wavlm import wavlm_feature_encoder

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    with torch.no_grad():
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
        frames = wavlm_feature_encoder(config.wavlm_checkpoint)(waveform.unsqueeze(0))  # (1, 512, frames)
    return frames[0].T.numpy()


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


# Jaitly & Hinton 2013's VTLP boundary frequency: below f_hi * min(alpha, 1) / alpha
# the warp is a plain alpha scaling, above it a line pinning Nyquist in place.
VTLP_BOUNDARY_HZ = 4800.0


def _bin_center_frequencies(config):
    """Centre frequency (Hz) of each feature bin: the HTK mel points
    torchaudio.functional.melscale_fbanks (compute_mel_spectrogram) peaks at,
    or the STFT bin frequencies for a linear spectrogram."""
    if config.type == "linear":
        return np.arange(config.nfft // 2 + 1) * config.sample_rate / config.nfft
    if config.type != "mel":
        raise ValueError(f"VTLP is not supported for transformation type {config.type!r}")
    mel_min, mel_max = (2595.0 * np.log10(1.0 + f / 700.0) for f in (config.fmin, config.fmax))
    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2)[1:-1]
    return 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)


def warp_frequency_axis(spectrogram, alpha, config):
    """VTLP (Jaitly & Hinton 2013): spectrogram (frames, bins) with its
    frequency axis warped by W(f) = alpha * f, piecewise linear so 0 Hz and
    Nyquist stay fixed -- alpha > 1 moves formants/harmonics up (a shorter
    vocal tract), alpha < 1 down. Each bin takes the value at W^-1 of its
    centre frequency, linearly interpolated between neighbouring bins (for a
    mel spectrogram an approximation of warping the filterbank itself).
    Frame count and order are untouched."""
    if alpha == 1:
        return spectrogram
    nyquist = config.sample_rate / 2
    knee = VTLP_BOUNDARY_HZ * min(alpha, 1) / alpha
    centers = _bin_center_frequencies(config)
    sources = np.interp(centers, [0.0, alpha * knee, nyquist], [0.0, knee, nyquist])
    lower = np.clip(np.searchsorted(centers, sources, side="right") - 1, 0, len(centers) - 2)
    weight = np.clip((sources - centers[lower]) / (centers[lower + 1] - centers[lower]), 0.0, 1.0)
    warped = spectrogram[:, lower] * (1 - weight) + spectrogram[:, lower + 1] * weight
    return warped.astype(spectrogram.dtype)


def vtlp_warp(features, alpha, config):
    """warp_frequency_axis for already per-frame standardized features:
    interpolation is linear and keeps constants constant, so
    re-standardizing gives exactly normalise_standardize(warp(unstandardized))."""
    if alpha == 1:
        return features
    return normalise_standardize(warp_frequency_axis(features, alpha, config)).astype(features.dtype)
