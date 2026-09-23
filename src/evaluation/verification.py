import itertools
from typing import NamedTuple

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


class IndexedTrials(NamedTuple):
    """A trial list stored as parallel arrays over a table of unique
    utterance ids, rather than trial_list_equal_error_rate's list of
    (label, id1, id2) string tuples -- TidyVoiceX's official Dev list has 12M
    trials over only ~59k utterances, which as Python tuples alone would cost
    several GB of RAM."""

    utterance_ids: list
    labels: np.ndarray
    idx1: np.ndarray
    idx2: np.ndarray


def indexed_trial_equal_error_rate(embeddings, utterance_ids, trials, chunk_size=200_000):
    """Same EER as trial_list_equal_error_rate, but over IndexedTrials and
    scored chunk_size trials at a time: gathering both sides' embeddings for
    all of TidyVoiceX's 12M trials at once (as trial_list_equal_error_rate
    does) would need ~49 GB at 512 dims.

    Every id in trials.utterance_ids must be present in `utterance_ids`
    (extract_embeddings silently skips utterances no longer than one
    segment, so callers must drop trials referencing those beforehand)."""
    row_by_id = {utterance_id: row for row, utterance_id in enumerate(utterance_ids)}
    order = np.array([row_by_id[utterance_id] for utterance_id in trials.utterance_ids])
    normed = embeddings[order] / np.linalg.norm(embeddings[order], axis=-1, keepdims=True)
    scores = np.empty(len(trials.labels), dtype=np.float32)
    for start in range(0, len(scores), chunk_size):
        end = start + chunk_size
        scores[start:end] = np.einsum("ij,ij->i", normed[trials.idx1[start:end]], normed[trials.idx2[start:end]])
    fpr, tpr, _ = roc_curve(trials.labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)
