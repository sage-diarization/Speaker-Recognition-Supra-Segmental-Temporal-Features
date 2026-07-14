import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .metrics import misclassification_rate


def best_misclassification_rate(embeddings, speaker_ids):
    """Cosine-distance complete-linkage clustering, sweeping every merge
    threshold in the linkage and reporting the best (lowest) MR, matching
    context/src/evaluation/clustering.py::evaluate_clustering."""
    speaker_ids = np.asarray(speaker_ids)
    z = linkage(embeddings, method="complete", metric="cosine")
    best = None
    for threshold in z[:, 2]:
        predicted = fcluster(z, threshold, "distance")
        mr = misclassification_rate(speaker_ids, predicted)
        if best is None or mr < best:
            best = mr
    return float(best)
