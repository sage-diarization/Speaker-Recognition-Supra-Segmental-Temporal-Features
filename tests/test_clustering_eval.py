import numpy as np

from src.evaluation.clustering import best_misclassification_rate


def test_well_separated_clusters_have_near_zero_mr():
    rng = np.random.default_rng(0)
    num_speakers, dim = 5, 16
    embeddings, speaker_ids = [], []
    for speaker in range(num_speakers):
        base = np.zeros(dim)
        base[speaker % dim] = 1.0
        for _ in range(4):
            embeddings.append(base + rng.normal(0, 0.01, size=dim))
            speaker_ids.append(speaker)

    mr = best_misclassification_rate(np.array(embeddings), speaker_ids)
    assert mr < 0.05


def test_mr_is_within_valid_range_for_random_embeddings():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(0, 1, size=(30, 8))
    speaker_ids = rng.integers(0, 6, size=30)

    mr = best_misclassification_rate(embeddings, speaker_ids)
    assert 0.0 <= mr <= 1.0
