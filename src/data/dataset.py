import numpy as np
import torch
from torch.utils.data import Dataset

from .features import apply_drc, compute_mel_spectrogram, normalise_standardize
from .segments import DRAW_STRATEGIES


def featurize_waveform(waveform, transformation_config):
    mel = compute_mel_spectrogram(waveform, transformation_config)
    mel = apply_drc(mel)
    return normalise_standardize(mel)


class SegmentDataset(Dataset):
    """Draws a fresh segment (OS/SS/SU) per access, re-sampled every epoch
    since DataLoader calls __getitem__ again each pass over the dataset."""

    def __init__(self, utterances, segment_length, draw_strategy, seed=None):
        self.segment_length = segment_length
        self.draw_fn = DRAW_STRATEGIES[draw_strategy]
        self.rng = np.random.default_rng(seed)
        self.utterances = [(f, label) for f, label in utterances if f.shape[0] > segment_length]
        if not self.utterances:
            raise ValueError("no utterance is longer than segment_length")

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        features, label = self.utterances[idx]
        segment = self.draw_fn(features, self.segment_length, self.rng)
        tensor = torch.from_numpy(segment).float().unsqueeze(0)
        return tensor, label
