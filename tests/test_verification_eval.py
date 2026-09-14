import numpy as np
import pytest

from src.evaluation.verification import (
    build_verification_pairs,
    cosine_similarity,
    equal_error_rate,
    trial_list_equal_error_rate,
)


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


def test_trial_list_eer_scores_only_the_given_pairs_near_zero_for_well_separated_speakers():
    # Same well-separated-speaker construction as
    # test_eer_near_zero_for_well_separated_speakers, but each utterance has
    # its own id (a VoxCeleb-style relative path) and only a curated set of
    # trial pairs is scored -- not every possible pair, as equal_error_rate
    # would do (and as would be nonsensical here: utterance_ids are unique
    # per row, so equal_error_rate would treat every pair as "different
    # speaker").
    rng = np.random.default_rng(0)
    num_speakers, per_speaker, dim = 20, 5, 16
    embeddings, utterance_ids, speaker_of = [], [], {}
    for speaker in range(num_speakers):
        base = np.zeros(dim)
        base[speaker % dim] = 1.0
        for i in range(per_speaker):
            utterance_id = f"id{speaker:04d}/clip{i}/00001.wav"
            embeddings.append(base + rng.normal(0, 0.01, size=dim))
            utterance_ids.append(utterance_id)
            speaker_of[utterance_id] = speaker
    embeddings = np.array(embeddings)

    ids_by_speaker = {
        speaker: [uid for uid, spk in speaker_of.items() if spk == speaker] for speaker in range(num_speakers)
    }
    trial_pairs = []
    for speaker in range(num_speakers):
        own = ids_by_speaker[speaker]
        other = ids_by_speaker[(speaker + 1) % num_speakers]
        trial_pairs.append((1, own[0], own[1]))  # same speaker
        trial_pairs.append((0, own[0], other[0]))  # different speaker
    assert {label for label, _, _ in trial_pairs} == {0, 1}  # both classes present

    eer = trial_list_equal_error_rate(embeddings, utterance_ids, trial_pairs)
    assert eer < 0.05


def test_trial_list_eer_only_looks_up_ids_referenced_by_the_trial_pairs():
    # An embedding for an utterance that appears in utterance_ids but not in
    # any trial pair must not be required/looked up -- trial_list_equal_error_rate
    # only needs the embeddings for ids the trial pairs actually reference.
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    utterance_ids = ["a", "b", "c", "unused"]
    trial_pairs = [(1, "a", "b"), (0, "a", "c")]

    eer = trial_list_equal_error_rate(embeddings, utterance_ids, trial_pairs)
    assert eer == pytest.approx(0.0)
