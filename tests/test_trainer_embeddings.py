import numpy as np
import pytest

from src.models.common import BackendOutput
from src.training.trainer import extract_embeddings


class _MeanModel:
    """Stand-in model whose backend embedding is just the mean value of its
    input segment (broadcast to a length-1 vector) -- lets us verify
    extract_embeddings' hop-window averaging arithmetic exactly, without
    depending on a real (nonlinear) backend's numbers."""

    def eval(self):
        pass

    def to(self, device):
        return self

    def __call__(self, x):
        backend = x.mean(dim=(1, 2, 3))
        return BackendOutput(backend=backend.unsqueeze(-1), bottleneck=None)


def _os_windows(features, segment_length, step, num_windows):
    return [features[i * step:i * step + segment_length] for i in range(num_windows)]


def test_extract_embeddings_averages_sequential_hop_windows():
    """OS windows must walk sequential, evenly-spaced start positions
    (matching context/src/setup/setup.py's `current_start += step` loop),
    not a fresh random position per window -- otherwise averaging no longer
    guarantees systematic coverage of the whole utterance."""
    segment_length, step, utterance_length = 10, 5, 40  # windows at 0,5,...,30 -> 7 windows
    features = np.arange(utterance_length, dtype=np.float64)[:, None] * np.ones((1, 3))

    embeddings, labels = extract_embeddings(_MeanModel(), [(features, 5)], segment_length, "OS")

    expected_windows = _os_windows(features, segment_length, step, 7)
    expected = np.mean([w.mean() for w in expected_windows])
    np.testing.assert_allclose(embeddings[0, 0], expected)
    assert labels[0] == 5


def test_extract_embeddings_uses_single_window_for_short_utterance():
    """An utterance only slightly longer than segment_length has just one
    hop window, matching the original loop's guaranteed first iteration."""
    segment_length = 10
    features = np.arange(segment_length + 1, dtype=np.float64)[:, None] * np.ones((1, 3))

    embeddings, _ = extract_embeddings(_MeanModel(), [(features, 0)], segment_length, "OS")

    assert embeddings[0, 0] == pytest.approx(features[:segment_length].mean())


def test_extract_embeddings_hop_fraction_controls_window_count():
    """hop_fraction=1.0 ('H100', non-overlapping windows) should draw fewer,
    non-overlapping windows than the default 0.5 ('H50')."""
    segment_length, utterance_length = 10, 40
    features = np.arange(utterance_length, dtype=np.float64)[:, None] * np.ones((1, 3))

    embeddings, _ = extract_embeddings(
        _MeanModel(), [(features, 0)], segment_length, "OS", hop_fraction=1.0
    )

    expected_windows = _os_windows(features, segment_length, 10, 4)  # starts 0,10,20,30
    expected = np.mean([w.mean() for w in expected_windows])
    np.testing.assert_allclose(embeddings[0, 0], expected)


def test_extract_embeddings_ss_shuffles_frame_order_within_each_window():
    """SS windows are the same contiguous crop as OS but with frame order
    destroyed, freshly reshuffled per window (matching setup.py's
    per-iteration np.random.shuffle(rseg_dist))."""
    segment_length = 6
    features = np.arange(segment_length + 1, dtype=np.float64)[:, None] * np.ones((1, 3))
    captured = {}

    class _CapturingModel(_MeanModel):
        def __call__(self, x):
            captured["x"] = x.clone()
            return super().__call__(x)

    extract_embeddings(_CapturingModel(), [(features, 0)], segment_length, "SS", seed=0)

    window = captured["x"][0, 0].numpy()
    np.testing.assert_allclose(sorted(window[:, 0]), features[:segment_length, 0])
    assert not np.array_equal(window[:, 0], features[:segment_length, 0])


def test_extract_embeddings_su_draws_with_replacement_from_whole_utterance():
    """SU/RF must draw segment_length frames with replacement from the
    *entire* utterance for every window (matching setup.py's
    np.random.choice(time_dist, segment_length)), not from a bounded local
    sub-window (that's src/data/segments.py::draw_su's on-the-fly *training*
    sampler behavior, ported from context/src/generator/sampler.py::load_RF).
    This is the dominant source of the SC misclassification-rate gap against
    Neururer et al. 2024's Table 1 for long, multi-sentence SC utterances."""
    segment_length, step, utterance_length, seed = 4, 2, 10, 0  # windows at 0,2,4,6,8 -> 4 windows... see below
    num_windows = (utterance_length - segment_length) // step + 1
    features = np.arange(utterance_length, dtype=np.float64)[:, None] * np.ones((1, 3))

    embeddings, _ = extract_embeddings(_MeanModel(), [(features, 0)], segment_length, "SU", seed=seed)

    reference_rng = np.random.default_rng(seed)
    expected_means = []
    for _ in range(num_windows):
        indices = reference_rng.integers(0, utterance_length, segment_length)
        expected_means.append(features[indices].mean())
    np.testing.assert_allclose(embeddings[0, 0], np.mean(expected_means))
