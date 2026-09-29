"""Spectral corruption augmentations for forcing models to rely on SST features.

This implements Strategy 1: Make FBA unreliable during training to force models
to rely on supra-segmental temporal (SST) features instead of frame-based acoustic (FBA) cues.

The augmentations degrade spectral content while preserving temporal structure (durations, frame order).
"""

import random
import torch
import torchaudio
from typing import Optional, Tuple


class FBACorruptor:
    """Applies spectral corruption that degrades frame-based acoustic (FBA) speaker cues
    while preserving temporal structure (supra-segmental features).
    
    This implements Strategy 1 from the research: Make FBA unreliable during training
    to force models to rely on SST features.
    
    The corruption is applied on-the-fly during training and can be configured via
    FBACorruptionConfig from src.config. When disabled or probability is 0, samples pass 
    through unchanged.
    """
    
    def __init__(self, config):
        """
        Args:
            config: FBACorruptionConfig instance with all parameters
        """
        self.config = config
        self._enabled = config.enabled
        self._probability = config.probability
        self._formant_range = config.formant_shift_range
        self._noise_range = config.noise_std_range
        self._invert_prob = config.spectral_invert_probability
        
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply corruption to a feature tensor (spectrogram frames).
        
        This implementation works on *feature tensors* (spectrogram frames) rather than
        raw waveforms, since the existing pipeline already converts to spectrograms.
        We corrupt the spectral representation directly.
        
        Args:
            features: Tensor of shape (frames, freqs) - a spectrogram segment
            
        Returns:
            Corrupted features of the same shape
        """
        if not self._enabled or random.random() > self._probability:
            return features
        
        # Choose which corruption(s) to apply
        corruptions = []
        
        # Formant-like spectral shift (simulated on spectrogram)
        # We approximate formant shifting by shifting frequency bins
        if self._formant_range != (0, 0):
            shift_semitones = random.uniform(*self._formant_range)
            # Convert semitones to frequency bin shift (approximate)
            # 1 semitone ≈ 1/12 octave, and frequency bins are logarithmic
            # This is a simplified approximation
            n_bins = features.shape[1]
            shift_bins = int(shift_semitones * n_bins / 12.0)
            if shift_bins != 0:
                corruptions.append(('formant', shift_bins))
        
        # Additive noise
        if self._noise_range != (0, 0):
            noise_std = random.uniform(*self._noise_range)
            corruptions.append(('noise', noise_std))
        
        # Spectral inversion
        if random.random() < self._invert_prob:
            corruptions.append(('invert', None))
        
        # Apply all selected corruptions
        result = features.clone()
        for corr_type, param in corruptions:
            if corr_type == 'formant':
                result = self._apply_formant_shift(result, param)
            elif corr_type == 'noise':
                result = self._apply_noise(result, param)
            elif corr_type == 'invert':
                result = self._apply_spectral_invert(result)
        
        return result
    
    def _apply_formant_shift(self, features: torch.Tensor, shift_bins: int) -> torch.Tensor:
        """Apply approximate formant shift by rolling frequency bins."""
        if shift_bins == 0:
            return features
        
        n_bins = features.shape[1]
        # Roll frequency bins (positive = shift up, negative = shift down)
        result = torch.roll(features, shifts=shift_bins, dims=1)
        
        # Handle edge bins by zeroing out the wrapped-around content
        # This is a simple approach; more sophisticated would use proper phase vocoder
        if shift_bins > 0:
            result[:, :shift_bins] = 0
        elif shift_bins < 0:
            result[:, shift_bins:] = 0
        
        return result
    
    def _apply_noise(self, features: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to spectrogram."""
        noise = torch.randn_like(features) * std
        return features + noise
    
    def _apply_spectral_invert(self, features: torch.Tensor) -> torch.Tensor:
        """Reverse frequency bins (spectral inversion)."""
        return torch.flip(features, dims=[1])


# Global corruptor instance - will be configured at dataset initialization
_current_corruptor: Optional[FBACorruptor] = None


def get_corruptor() -> Optional[FBACorruptor]:
    """Get the currently configured global corruptor (if any)."""
    return _current_corruptor


def set_corruptor(corruptor: Optional[FBACorruptor]):
    """Set the global corruptor instance."""
    global _current_corruptor
    _current_corruptor = corruptor
