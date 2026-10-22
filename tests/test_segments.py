import numpy as np
import pytest

from src.data.segments import DRAW_STRATEGIES, draw_bs, draw_os, draw_ss, draw_su, shuffle_blocks


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


@pytest.mark.parametrize("block_size", [2, 4, 16])
def test_bs_permutes_blocks_of_the_os_crop_keeping_each_blocks_order(utterance, block_size):
    os_segment = draw_os(utterance, 96, np.random.default_rng(3))
    bs_segment = draw_bs(utterance, 96, np.random.default_rng(3), block_size=block_size)
    assert bs_segment.shape == (96, 8)
    assert not np.array_equal(os_segment, bs_segment)

    def blocks(segment):
        return sorted(tuple(segment[st:st + block_size, 0]) for st in range(0, 96, block_size))

    assert blocks(bs_segment) == blocks(os_segment)


def test_bs_with_a_block_spanning_the_segment_is_os(utterance):
    np.testing.assert_array_equal(
        draw_bs(utterance, 94, np.random.default_rng(5), block_size=94),
        draw_os(utterance, 94, np.random.default_rng(5)),
    )


def test_mix_draws_both_ordered_and_shuffled_segments(utterance):
    rng = np.random.default_rng(0)
    ordered = [np.all(np.diff(DRAW_STRATEGIES["MIX"](utterance, 94, rng)[:, 0]) == 8) for _ in range(200)]
    assert 60 < sum(ordered) < 140


def test_randbs_draws_every_block_size_and_os(utterance):
    rng = np.random.default_rng(0)
    runs = set()
    for _ in range(300):
        rows = DRAW_STRATEGIES["RANDBS"](utterance, 192, rng)[:, 0] / 8
        # Length of the first in-order run: the block size (or 192 for OS).
        breaks = np.flatnonzero(np.diff(rows) != 1)
        runs.add(int(breaks[0]) + 1 if len(breaks) else 192)
    assert {1, 8, 16, 32, 64, 192} <= runs


def test_shuffle_blocks_never_returns_the_original_order():
    segment = np.arange(10)[:, None]
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert np.array_equal(shuffle_blocks(segment, 5, rng)[:, 0], [5, 6, 7, 8, 9, 0, 1, 2, 3, 4])


def test_draw_strategies_register_block_shuffles_and_mixtures():
    assert {"OS", "SS", "SU", "BS4", "BS64", "MIX", "RANDBS"} <= set(DRAW_STRATEGIES)


def test_gapped_segment_spreads_ordered_blocks_over_the_utterance(utterance):
    from src.data.segments import gapped_segment

    # 300 frames, 6 blocks of 16: gap (300 - 96) // 6 = 34, slack 204 - 5 * 34 = 34.
    rows = gapped_segment(utterance, 96, 16, 0.0)[:, 0] / 8
    assert len(rows) == 96
    for i in range(6):
        np.testing.assert_array_equal(rows[i * 16:(i + 1) * 16], np.arange(16) + i * 50)
    last = gapped_segment(utterance, 96, 16, 1.0)[:, 0] / 8
    assert last[0] == 34 and last[-1] == 299


def test_gapped_segment_keeps_a_shorter_last_block(utterance):
    from src.data.segments import gapped_segment

    rows = gapped_segment(utterance, 40, 16, 0.0)[:, 0] / 8  # blocks of 16, 16, 8; gap 260 // 3 = 86
    np.testing.assert_array_equal(rows, np.concatenate([np.arange(16), np.arange(16) + 102, np.arange(8) + 204]))


@pytest.mark.parametrize("block_size", [4, 16])
def test_gs_shuffles_exactly_the_blocks_of_the_matching_go_draw(utterance, block_size):
    from src.data.segments import draw_go, draw_gs

    go = draw_go(utterance, 96, np.random.default_rng(3), block_size=block_size)
    gs = draw_gs(utterance, 96, np.random.default_rng(3), block_size=block_size)
    assert not np.array_equal(go, gs)

    def blocks(segment):
        return sorted(tuple(segment[st:st + block_size, 0]) for st in range(0, 96, block_size))

    assert blocks(gs) == blocks(go)


def test_draw_strategies_register_gapped_block_pairs():
    assert {"GO8", "GS8", "GO64", "GS64"} <= set(DRAW_STRATEGIES)
