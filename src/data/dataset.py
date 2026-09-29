import numpy as np
import torch
from typing import Optional
from torch.utils.data import Dataset

from .features import apply_drc, compute_linear_spectrogram, compute_mel_spectrogram, normalise_standardize
from .lazy_features import resolve_features, utterance_length
from .segments import DRAW_STRATEGIES
from .augment import FBACorruptor


def featurize_waveform(waveform, transformation_config):
    if transformation_config.type == "linear":
        spectrogram = compute_linear_spectrogram(waveform, transformation_config)
    else:
        spectrogram = compute_mel_spectrogram(waveform, transformation_config)
    spectrogram = apply_drc(spectrogram)
    return normalise_standardize(spectrogram)


def featurize_frame_range(read_samples_fn, path, start, stop, transformation_config):
    """Features of frames [start, stop) of the utterance at path, reading only
    the samples those frames cover -- equal (up to float32 rounding in the
    mel matmul) to featurize_waveform(whole utterance)[start:stop], since every step after
    framing (STFT, mel filterbank, DRC, per-frame standardization) is
    frame-local and frame i spans samples [i * frame_step, i * frame_step +
    frame_length) (src/data/features.py's unpadded framing)."""
    first_sample = start * transformation_config.frame_step
    last_sample = (stop - 1) * transformation_config.frame_step + transformation_config.frame_length
    return featurize_waveform(read_samples_fn(path, first_sample, last_sample), transformation_config)


class SegmentDataset(Dataset):
    """Draws a fresh segment (OS/SS/SU) per access, re-sampled every epoch
    since DataLoader calls __getitem__ again each pass over the dataset.
    Each utterance entry is either a plain ndarray (already featurized) or a
    LazyFeatures instance (featurized on first access -- see
    src/data/lazy_features.py), transparently via resolve_features/
    utterance_length.
    
    FBA corruption augmentation (Strategy 1) can be optionally applied during
    training to degrade spectral cues and force models to rely on SST features.
    """

    def __init__(self, utterances, segment_length, draw_strategy, seed=None, 
                 fba_corruption_config: Optional[FBACorruptionConfig] = None,
                 is_training: bool = True):
        """
        Args:
            utterances: List of (features, label) pairs
            segment_length: Number of frames in each segment
            draw_strategy: One of "OS", "SS", "SU"
            seed: Random seed for segment drawing
            fba_corruption_config: Optional FBACorruptionConfig for spectral augmentation.
                                  Only applied when is_training=True.
            is_training: Whether this dataset is used for training (vs validation/test)
        """
        self.segment_length = segment_length
        self.draw_fn = DRAW_STRATEGIES[draw_strategy]
        self.rng = np.random.default_rng(seed)
        self.utterances = [(f, label) for f, label in utterances if utterance_length(f) > segment_length]
        if not self.utterances:
            raise ValueError("no utterance is longer than segment_length")
        
        # Setup FBA corruption if configured and in training mode
        self._corruptor = None
        if (fba_corruption_config is not None and 
            fba_corruption_config.is_active() and 
            is_training):
            self._corruptor = FBACorruptor(fba_corruption_config)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        features_or_loader, label = self.utterances[idx]
        # A windowed LazyFeatures is handed to the draw function as-is: its
        # slicing featurizes just the drawn frames (see featurize_frame_range).
        if getattr(features_or_loader, "windowed", False):
            features = features_or_loader
        else:
            features = resolve_features(features_or_loader)
        segment = self.draw_fn(features, self.segment_length, self.rng)
        
        # Convert to tensor
        tensor = torch.from_numpy(segment).float().unsqueeze(0)
        
        # Apply FBA corruption if configured
        if self._corruptor is not None:
            # Squeeze the channel dimension, corrupt, then unsqueeze back
            tensor = self._corruptor(tensor.squeeze(0)).unsqueeze(0)
        
        return tensor, label
