import numpy as np
import pytest

from src.data.segments import draw_os, draw_ss, draw_su


@pytest.fixture
def utterance():
    return np.arange(300 * 8, dtype=np.float32).reshape(300, 8)


def test_os_returns_contiguous_crop_of_correct_length(utterance):
    segment = draw_os(utterance, 94, np.random.default_rng(0))
    assert segment.shape == (94, 8)
    row_indices = segment[:, 0] / 8
    assert np.all(np.diff(row_indices) == 1)


def test_os_raises_when_utterance_too_short():
    short = np.zeros((50, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        draw_os(short, 94, np.random.default_rng(0))


def test_ss_is_a_permutation_of_the_matching_os_crop(utterance):
    os_segment = draw_os(utterance, 94, np.random.default_rng(42))
    ss_segment = draw_ss(utterance, 94, np.random.default_rng(42))

    os_sorted = np.sort(os_segment[:, 0])
    ss_sorted = np.sort(ss_segment[:, 0])
    np.testing.assert_array_equal(os_sorted, ss_sorted)
    assert not np.array_equal(os_segment, ss_segment)


def test_su_returns_correct_length_and_valid_frames(utterance):
    segment = draw_su(utterance, 94, np.random.default_rng(1))
    assert segment.shape == (94, 8)
    row_ids = segment[:, 0] / 8
    assert np.all(row_ids >= 0) and np.all(row_ids < 300)
    assert len(set(row_ids.tolist())) == 94  # drawn without replacement


def test_su_raises_when_utterance_too_short():
    short = np.zeros((50, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        draw_su(short, 94, np.random.default_rng(0))


@pytest.mark.parametrize("draw_fn", [draw_os, draw_ss, draw_su])
def test_draws_are_deterministic_given_a_seed(utterance, draw_fn):
    a = draw_fn(utterance, 94, np.random.default_rng(7))
    b = draw_fn(utterance, 94, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
