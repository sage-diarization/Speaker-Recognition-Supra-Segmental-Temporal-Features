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


def trial_list_equal_error_rate(embeddings, utterance_ids, trial_pairs):
    """EER over a fixed list of (label, utterance_id1, utterance_id2) trial
    pairs -- the standard VoxCeleb verification protocol (a curated trial
    list, not every possible pair) -- scoring only those specific
    comparisons via cosine similarity, unlike equal_error_rate's exhaustive
    all-vs-all pairing (appropriate for TIMIT's SV protocol, but not how
    VoxCeleb's veri_test-style evaluation works, and not computationally
    sane at VoxCeleb's utterance count).

    `embeddings`/`utterance_ids` are extract_embeddings' usual parallel
    arrays, with utterance_ids holding each row's relative wav path (as used
    by src/data/voxceleb.py) rather than a speaker id."""
    embedding_by_id = dict(zip(utterance_ids, embeddings))
    labels = np.array([int(label) for label, _, _ in trial_pairs])
    embeddings1 = np.stack([embedding_by_id[utterance_id1] for _, utterance_id1, _ in trial_pairs])
    embeddings2 = np.stack([embedding_by_id[utterance_id2] for _, _, utterance_id2 in trial_pairs])
    scores = cosine_similarity(embeddings1, embeddings2)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)
