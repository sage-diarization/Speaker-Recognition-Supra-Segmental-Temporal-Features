import itertools

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=-1)


def build_verification_pairs(speaker_ids):
    """All unordered pairs of the given per-utterance speaker ids."""
    n = len(speaker_ids)
    idx1, idx2, labels = [], [], []
    for i, j in itertools.combinations(range(n), 2):
        idx1.append(i)
        idx2.append(j)
        labels.append(int(speaker_ids[i] == speaker_ids[j]))
    return np.array(idx1), np.array(idx2), np.array(labels)


def equal_error_rate(embeddings, speaker_ids):
    idx1, idx2, labels = build_verification_pairs(speaker_ids)
    scores = cosine_similarity(embeddings[idx1], embeddings[idx2])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)
