import numpy as np
import pytest

from src.config import ExperimentConfig, TransformationConfig
from src.data.dataset import featurize_waveform


@pytest.fixture
def transformation_config():
    return TransformationConfig()


@pytest.fixture
def small_config():
    config = ExperimentConfig()
    config.training.num_epochs = 3
    config.training.batch_size = 4
    config.evaluation.sc_num_speakers = 3
    config.evaluation.sc_utterances_per_speaker = 2
    return config


def make_synthetic_waveform(frequency, duration_s, sample_rate, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    noise = rng.normal(0, 0.01, size=t.shape)
    return (signal + noise).astype(np.float32)


@pytest.fixture
def synthetic_utterances(transformation_config):
    """Builds (features, speaker_label) pairs from distinguishable synthetic
    sine-wave 'speakers' run through the real feature pipeline, so tests
    don't depend on a real (licensed) TIMIT copy being present."""

    def _build(num_speakers=4, utterances_per_speaker=6, duration_s=2.0, seed=0):
        utterances = []
        for speaker in range(num_speakers):
            frequency = 200 + 150 * speaker
            for utterance_idx in range(utterances_per_speaker):
                waveform = make_synthetic_waveform(
                    frequency,
                    duration_s,
                    transformation_config.sample_rate,
                    seed=seed * 1000 + speaker * 100 + utterance_idx,
                )
                features = featurize_waveform(waveform, transformation_config)
                utterances.append((features, speaker))
        return utterances

    return _build
