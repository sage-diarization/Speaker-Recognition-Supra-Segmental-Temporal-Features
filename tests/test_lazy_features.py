import numpy as np
import pytest

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


def _windowed_entry(waveform, transformation):
    """A LazyFeatures reading samples straight out of `waveform`, with a log
    of every (start, stop) sample range read."""
    import functools

    from src.data.dataset import featurize_frame_range, featurize_waveform

    reads = []

    def read_samples(path, start, stop):
        reads.append((start, stop))
        return waveform[start:stop]

    length = expected_frame_count(waveform.shape[0], transformation)
    frames_fn = functools.partial(featurize_frame_range, read_samples, "unused", transformation_config=transformation)
    return LazyFeatures(lambda: featurize_waveform(waveform, transformation), length, frames_fn), reads


@pytest.mark.parametrize("kind", ["mel", "linear"])
def test_windowed_frame_slices_equal_slicing_the_fully_featurized_utterance(kind):
    from src.data.dataset import featurize_waveform

    transformation = TransformationConfig()
    if kind == "linear":  # ResNet's front-end
        transformation = TransformationConfig(type="linear", window="hamming", nfft=512, frame_length_s=0.025)
    waveform = np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32)
    full = featurize_waveform(waveform, transformation)
    entry, reads = _windowed_entry(waveform, transformation)

    assert entry.length == full.shape[0]
    assert entry.shape == (full.shape[0],)
    for start, stop in [(0, 95), (37, 132), (full.shape[0] - 95, full.shape[0]), (10, 11)]:
        # Equal up to float32 rounding: the mel matmul's BLAS blocking depends
        # on how many frames go through it at once.
        np.testing.assert_allclose(entry[start:stop], full[start:stop], rtol=0, atol=1e-5)
    # Only the samples those frames cover were read, never the whole file.
    assert reads[1] == (37 * transformation.frame_step, 131 * transformation.frame_step + transformation.frame_length)


@pytest.mark.parametrize("strategy", ["OS", "SS", "SU"])
def test_segment_dataset_draws_identical_segments_from_windowed_entries(strategy):
    from src.data.dataset import SegmentDataset, featurize_waveform

    transformation = TransformationConfig()
    waveforms = [np.random.default_rng(i).standard_normal(16000 * 3).astype(np.float32) for i in range(3)]
    eager = [(featurize_waveform(w, transformation), i) for i, w in enumerate(waveforms)]
    windowed = [(_windowed_entry(w, transformation)[0], i) for i, w in enumerate(waveforms)]

    eager_dataset = SegmentDataset(eager, 95, strategy, seed=7)
    windowed_dataset = SegmentDataset(windowed, 95, strategy, seed=7)
    for _ in range(2):
        for idx in range(3):
            eager_segment, eager_label = eager_dataset[idx]
            windowed_segment, windowed_label = windowed_dataset[idx]
            assert eager_label == windowed_label
            # Same draws (identical RNG consumption), same values up to float32 rounding.
            np.testing.assert_allclose(windowed_segment.numpy(), eager_segment.numpy(), rtol=0, atol=1e-5)
