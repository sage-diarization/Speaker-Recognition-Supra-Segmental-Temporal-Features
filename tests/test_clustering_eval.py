import numpy as np
from scipy.spatial.distance import cdist

from src.evaluation import clustering
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


def test_clusters_on_cosine_distance_of_distance_profiles(monkeypatch):
    """context/src/evaluation/clustering.py builds a pairwise cosine-distance
    matrix and feeds it into scipy's linkage with metric='cosine' -- since
    scipy treats a 2-D input as raw observation vectors, this clusters on
    the cosine distance between each item's distance-profile row, not
    directly on embedding cosine distance. Locks in that this quirk is
    reproduced (linkage is called with the distance matrix, not the raw
    embeddings), since it's what produced the paper's reported numbers."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(8, 5))
    speaker_ids = np.repeat(np.arange(4), 2)

    captured = {}
    real_linkage = clustering.linkage

    def spy_linkage(y, method, metric):
        captured["y"] = y
        return real_linkage(y, method, metric)

    monkeypatch.setattr(clustering, "linkage", spy_linkage)
    best_misclassification_rate(embeddings, speaker_ids)

    expected = cdist(embeddings, embeddings, metric="cosine")
    np.testing.assert_allclose(captured["y"], expected)
