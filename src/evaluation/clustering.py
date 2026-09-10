import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from .metrics import misclassification_rate


def best_misclassification_rate(embeddings, speaker_ids):
    """Cosine-distance complete-linkage clustering, sweeping every merge
    threshold in the linkage and reporting the best (lowest) MR, matching
    context/src/evaluation/clustering.py::evaluate_clustering.

    That original builds an explicit pairwise cosine-distance matrix and
    feeds it into scipy's `linkage` with metric='cosine'; since scipy treats
    a 2-D input as raw observation vectors rather than a precomputed
    distance matrix, this clusters on the cosine distance *between each
    item's distance-profile row*, not directly on embedding cosine distance.
    That's very likely an unintentional quirk in the original codebase, but
    it's what produced the paper's Table 1/2 numbers, so it's reproduced
    here rather than "fixed" to a more standard direct-cosine clustering."""
    speaker_ids = np.asarray(speaker_ids)
    distance_matrix = cdist(embeddings, embeddings, metric="cosine")
    z = linkage(distance_matrix, method="complete", metric="cosine")
    best = None
    for threshold in z[:, 2]:
        predicted = fcluster(z, threshold, "distance")
        mr = misclassification_rate(speaker_ids, predicted)
        if best is None or mr < best:
            best = mr
    return float(best)
