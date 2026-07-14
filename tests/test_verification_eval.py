import numpy as np
import pytest

from src.evaluation.verification import build_verification_pairs, cosine_similarity, equal_error_rate


def test_build_verification_pairs_covers_all_unordered_pairs():
    idx1, idx2, labels = build_verification_pairs([0, 0, 1])
    assert len(idx1) == 3  # C(3, 2)
    np.testing.assert_array_equal(labels, [1, 0, 0])


def test_cosine_similarity_of_identical_vectors_is_one():
    a = np.array([[1.0, 2.0, 3.0]])
    assert cosine_similarity(a, a)[0] == pytest.approx(1.0)


def test_eer_near_zero_for_well_separated_speakers():
    rng = np.random.default_rng(0)
    num_speakers, dim = 20, 16
    embeddings, speaker_ids = [], []
    for speaker in range(num_speakers):
        base = np.zeros(dim)
        base[speaker % dim] = 1.0
        for _ in range(5):
            embeddings.append(base + rng.normal(0, 0.01, size=dim))
            speaker_ids.append(speaker)

    eer = equal_error_rate(np.array(embeddings), np.array(speaker_ids))
    assert eer < 0.05


def test_eer_near_half_for_random_unrelated_embeddings():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(0, 1, size=(200, 16))
    speaker_ids = rng.integers(0, 20, size=200)

    eer = equal_error_rate(embeddings, speaker_ids)
    assert abs(eer - 0.5) < 0.15
