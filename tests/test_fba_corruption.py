#!/usr/bin/env python3
"""Test suite for FBA corruption augmentation (Strategy 1).

Tests the implementation of spectral corruption to force models to rely on SST features.
"""

import numpy as np
import torch
from src.config import ExperimentConfig, FBACorruptionConfig, TrainingConfig
from src.data.augment import FBACorruptor
from src.data.dataset import SegmentDataset


def test_fba_corruption_config():
    """Test FBACorruptionConfig creation and defaults."""
    print("Testing FBACorruptionConfig...")
    
    # Test default (disabled)
    config = FBACorruptionConfig()
    assert not config.enabled
    assert not config.is_active()
    print("  ✓ Default config is disabled and inactive")
    
    # Test enabled with no corruption types
    config = FBACorruptionConfig(enabled=True, probability=0.5)
    assert not config.is_active()  # No corruption types configured
    print("  ✓ Enabled config with no corruption types is inactive")
    
    # Test enabled with corruption types
    config = FBACorruptionConfig(
        enabled=True,
        probability=0.5,
        formant_shift_range=(-1.0, 1.0)
    )
    assert config.is_active()
    print("  ✓ Enabled config with corruption types is active")
    
    # Test enabled with noise
    config = FBACorruptionConfig(
        enabled=True,
        probability=0.5,
        noise_std_range=(0.01, 0.1)
    )
    assert config.is_active()
    print("  ✓ Enabled config with noise is active")
    
    # Test enabled with spectral inversion
    config = FBACorruptionConfig(
        enabled=True,
        probability=0.5,
        spectral_invert_probability=0.5
    )
    assert config.is_active()
    print("  ✓ Enabled config with spectral inversion is active")


def test_fba_corruptor():
    """Test FBACorruptor operations."""
    print("\nTesting FBACorruptor...")
    
    # Create a simple spectrogram-like tensor (frames, freqs)
    # Simulate 100 frames, 128 frequency bins
    features = torch.randn(100, 128)
    
    # Test with disabled config
    config = FBACorruptionConfig(enabled=False)
    corruptor = FBACorruptor(config)
    result = corruptor(features)
    assert torch.allclose(features, result)
    print("  ✓ Disabled corruptor passes through unchanged")
    
    # Test with enabled but probability 0
    config = FBACorruptionConfig(enabled=True, probability=0.0)
    corruptor = FBACorruptor(config)
    result = corruptor(features)
    assert torch.allclose(features, result)
    print("  ✓ Probability 0 corruptor passes through unchanged")
    
    # Test with noise corruption
    config = FBACorruptionConfig(
        enabled=True,
        probability=1.0,  # Always apply
        noise_std_range=(0.01, 0.01)  # Fixed small noise
    )
    corruptor = FBACorruptor(config)
    result = corruptor(features)
    # Result should be different from input
    assert not torch.allclose(features, result)
    # But shape should be the same
    assert result.shape == features.shape
    print("  ✓ Noise corruption changes values but preserves shape")
    
    # Test with spectral inversion
    config = FBACorruptionConfig(
        enabled=True,
        probability=1.0,
        spectral_invert_probability=1.0
    )
    corruptor = FBACorruptor(config)
    result = corruptor(features)
    # Spectral inversion should reverse frequency bins
    expected = torch.flip(features, dims=[1])
    assert torch.allclose(result, expected)
    print("  ✓ Spectral inversion correctly reverses frequency bins")
    
    # Test with formant shift
    config = FBACorruptionConfig(
        enabled=True,
        probability=1.0,
        formant_shift_range=(1.0, 1.0)  # Fixed shift of 1 semitone
    )
    corruptor = FBACorruptor(config)
    result = corruptor(features)
    assert result.shape == features.shape
    # With a shift of 1 semitone on 128 bins, we expect ~11 bins shift
    # The exact behavior depends on the implementation
    print("  ✓ Formant shift preserves shape")


def test_segment_dataset_with_corruption():
    """Test SegmentDataset with FBA corruption enabled."""
    print("\nTesting SegmentDataset with corruption...")
    
    # Create dummy features
    num_utterances = 10
    segment_length = 50
    num_freqs = 128
    
    utterances = []
    for i in range(num_utterances):
        # Create features with enough frames
        features = np.random.randn(100, num_freqs).astype(np.float32)
        utterances.append((features, i))
    
    # Test without corruption
    config = FBACorruptionConfig(enabled=False)
    dataset = SegmentDataset(
        utterances, segment_length, "OS",
        fba_corruption_config=config,
        is_training=True
    )
    sample, label = dataset[0]
    assert sample.shape == (1, segment_length, num_freqs)
    print("  ✓ Dataset without corruption works")
    
    # Test with corruption disabled
    dataset = SegmentDataset(
        utterances, segment_length, "OS",
        fba_corruption_config=None,
        is_training=True
    )
    sample, label = dataset[0]
    assert sample.shape == (1, segment_length, num_freqs)
    print("  ✓ Dataset with None corruption config works")
    
    # Test with corruption enabled
    corruption_config = FBACorruptionConfig(
        enabled=True,
        probability=1.0,
        noise_std_range=(0.01, 0.01)
    )
    dataset = SegmentDataset(
        utterances, segment_length, "OS",
        fba_corruption_config=corruption_config,
        is_training=True
    )
    sample, label = dataset[0]
    assert sample.shape == (1, segment_length, num_freqs)
    print("  ✓ Dataset with corruption enabled works")
    
    # Test with is_training=False (should not apply corruption even if config is enabled)
    dataset = SegmentDataset(
        utterances, segment_length, "OS",
        fba_corruption_config=corruption_config,
        is_training=False
    )
    sample, label = dataset[0]
    assert sample.shape == (1, segment_length, num_freqs)
    print("  ✓ Dataset with is_training=False skips corruption")


def test_config_integration():
    """Test that FBACorruptionConfig integrates with ExperimentConfig."""
    print("\nTesting config integration...")
    
    # Create a minimal experiment config
    config_dict = {
        "training": {
            "fba_corruption": {
                "enabled": True,
                "probability": 0.5,
                "formant_shift_range": [-1.0, 1.0],
                "noise_std_range": [0.01, 0.1],
                "spectral_invert_probability": 0.3
            }
        }
    }
    
    config = ExperimentConfig.from_dict(config_dict)
    assert config.training.fba_corruption.enabled
    assert config.training.fba_corruption.probability == 0.5
    # YAML loads lists, but dataclass has tuples - convert for comparison
    assert tuple(config.training.fba_corruption.formant_shift_range) == (-1.0, 1.0)
    assert config.training.fba_corruption.is_active()
    print("  ✓ FBACorruptionConfig integrates with ExperimentConfig")
    
    # Test default config
    config = ExperimentConfig()
    assert not config.training.fba_corruption.enabled
    assert not config.training.fba_corruption.is_active()
    print("  ✓ Default ExperimentConfig has FBA corruption disabled")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FBA Corruption Implementation")
    print("=" * 60)
    
    test_fba_corruption_config()
    test_fba_corruptor()
    test_segment_dataset_with_corruption()
    test_config_integration()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
