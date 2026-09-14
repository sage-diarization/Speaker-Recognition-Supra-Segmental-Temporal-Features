import numpy as np

from src.config import TransformationConfig
from src.data.lazy_features import LazyFeatures, expected_frame_count, resolve_features, utterance_length


def test_resolve_features_calls_lazy_features_exactly_when_needed():
    calls = []

    def compute():
        calls.append(1)
        return np.zeros((10, 4))

    lazy = LazyFeatures(compute, length=10)
    assert calls == []  # not computed until resolved

    features = resolve_features(lazy)
    assert calls == [1]
    assert features.shape == (10, 4)

    # A plain ndarray passes through unchanged, without being called.
    ndarray = np.ones((5, 3))
    assert resolve_features(ndarray) is ndarray


def test_utterance_length_dispatches_on_entry_type():
    lazy = LazyFeatures(lambda: np.zeros((1, 1)), length=42)
    assert utterance_length(lazy) == 42

    ndarray = np.zeros((7, 4))
    assert utterance_length(ndarray) == 7


def test_expected_frame_count_matches_stft_framing():
    config = TransformationConfig()  # frame_length=1024, frame_step=160 samples at 16kHz/0.064s/0.01s
    # Exactly one frame's worth of samples -> exactly 1 frame.
    assert expected_frame_count(config.frame_length, config) == 1
    # One frame_step short of a second frame -> still only 1 frame.
    assert expected_frame_count(config.frame_length + config.frame_step - 1, config) == 1
    # Exactly one more frame_step -> 2 frames.
    assert expected_frame_count(config.frame_length + config.frame_step, config) == 2


def test_expected_frame_count_is_zero_for_too_short_audio():
    config = TransformationConfig()
    assert expected_frame_count(config.frame_length - 1, config) == 0
    assert expected_frame_count(0, config) == 0
