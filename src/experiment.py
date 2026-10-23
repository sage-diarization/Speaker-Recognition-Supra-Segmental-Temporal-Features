import argparse
import faulthandler
import functools
import itertools
import json
import os
import signal
import statistics
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from . import tracking
from .config import ExperimentConfig
from .data.dataset import (
    SegmentDataset,
    change_playback_speed,
    featurize_frame_range,
    featurize_waveform,
    playback_sample_count,
)
from .data.features import vtlp_warp
from .data.lazy_features import LazyFeatures, expected_frame_count
from .data.segments import BLOCK_SIZES, DRAW_STRATEGIES
from .data.tidyvoicex import TidyVoiceXCorpus
from .data.timit import TimitCorpus
from .data.voxceleb import VoxCelebCorpus
from .device import resolve_device
from .evaluation.clustering import best_misclassification_rate
from .evaluation.verification import (
    IndexedTrials,
    all_pairs_min_dcf,
    all_pairs_scores,
    eer_from_scores,
    equal_error_rate,
    indexed_trial_equal_error_rate,
    indexed_trial_min_dcf,
    indexed_trial_scores,
    trial_list_equal_error_rate,
    trial_list_min_dcf,
    trial_list_scores,
)
from .models.losses import build_loss
from .models.registry import build_model
from .training.trainer import best_checkpoint_path, extract_embeddings_by_strategy, train

STRATEGIES = ("OS", "SS", "SU")


class RunStatistics(NamedTuple):
    """Mean and (population) SD of a metric, as raw [0, 1] fractions, over
    config.num_runs independent train/test runs -- matches Neururer et al.
    2024 Section 2.3: 'For each reported EER/MR, we average over 5 train/test
    runs and report mean and SD.'"""

    mean: float
    std: float


def _run_statistics(values):
    return RunStatistics(mean=statistics.fmean(values), std=statistics.pstdev(values))


def _featurize_split(corpus, split, transformation, playback_speed=1.0):
    speakers = corpus.speakers(split)
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    utterances = []
    for speaker in speakers:
        for path in corpus.utterance_paths(split, speaker):
            waveform, _ = corpus.load_waveform(path)
            features = featurize_waveform(change_playback_speed(waveform, playback_speed), transformation)
            utterances.append((features, label_map[speaker]))
    return utterances, label_map


def _lazy_featurize_train_dev_split(corpus, transformation, dev_holdout_per_speaker, playback_speed=1.0):
    """Lazy analogue of _featurize_train_dev_split (see its docstring) for a
    corpus too large to eagerly featurize upfront -- currently VoxCeleb's
    TRAIN split, whose per-speaker segments run far more numerous than TIMIT's ~10 sentences/speaker."""
    speakers = corpus.speakers("TRAIN")
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    train_utterances, dev_utterances = [], []
    for speaker in speakers:
        paths = sorted(corpus.utterance_paths("TRAIN", speaker))
        for i, path in enumerate(paths):
            is_dev = i < dev_holdout_per_speaker
            speed = 1.0 if is_dev else playback_speed
            length = expected_frame_count(playback_sample_count(corpus.raw_sample_count(path), speed), transformation)
            compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, path, transformation, speed)
            target = dev_utterances if is_dev else train_utterances
            target.append((LazyFeatures(compute_fn, length), label_map[speaker]))
    return train_utterances, dev_utterances, label_map


def _featurize_train_dev_split(corpus, transformation, dev_holdout_per_speaker, playback_speed=1.0):
    """Splits TRAIN into an actual-training pool and a held-out dev pool
    (dev_holdout_per_speaker of each speaker's utterances, deterministically
    the first sorted paths), so periodic checkpoint selection during training
    draws from TRAIN speakers' own held-out utterances instead of the TEST
    split -- context/src's generator.py does the analogous thing, splitting
    AUDIO_LIST_TRAIN/AUDIO_LIST_VAL out of the same speaker pool
    (load_train_val_locs) rather than ever touching TEST before final
    scoring. Both returned utterance lists share one label_map, so dev EER's
    same-speaker/different-speaker pairing is correct."""
    speakers = corpus.speakers("TRAIN")
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    train_utterances, dev_utterances = [], []
    for speaker in speakers:
        paths = sorted(corpus.utterance_paths("TRAIN", speaker))
        for i, path in enumerate(paths):
            waveform, _ = corpus.load_waveform(path)
            is_dev = i < dev_holdout_per_speaker
            if not is_dev:
                waveform = change_playback_speed(waveform, playback_speed)
            features = featurize_waveform(waveform, transformation)
            target = dev_utterances if is_dev else train_utterances
            target.append((features, label_map[speaker]))
    return train_utterances, dev_utterances, label_map


def _featurize_dev_speakers(corpus, split, transformation, speaker_ids):
    """All of split's utterances of speaker_ids (matched case-insensitively, as
    TIMIT copies differ in speaker-directory case), for checkpoint selection on
    speakers never trained on -- context/src selects on its "development" SV
    list the same way, rather than on training speakers' own utterances."""
    available = {speaker.upper(): speaker for speaker in corpus.speakers(split)}
    missing = [speaker for speaker in speaker_ids if speaker.upper() not in available]
    if missing:
        raise ValueError(f"dev speakers not found in {split} split: {missing}")
    utterances = []
    for label, speaker in enumerate(available[speaker.upper()] for speaker in speaker_ids):
        for path in corpus.utterance_paths(split, speaker):
            waveform, _ = corpus.load_waveform(path)
            utterances.append((featurize_waveform(waveform, transformation), label))
    return utterances


def build_sc_utterances(corpus, split, transformation, num_speakers, speaker_ids=None):
    """Builds the SC task's per-speaker utterances by concatenating sentences
    rather than using single raw sentences, per Neururer et al. 2024 Section
    2.2: "2 utterances (comprising 2 and 8 concatenated sentences) per 40
    speakers". Sentences are sorted deterministically (not left in TIMIT's
    filesystem-extraction order, which is arbitrary per speaker) before being
    split into a long (all but the last 2) and short (last 2) concatenation,
    matching the SC data prep of the paper's reference [13]
    (github.com/stdm/ZHAW_deep_voice, common/spectrogram/speaker_train_splitter.py:
    the last 20% of each speaker's files form the second utterance). Taking
    the *first* 2 sorted files instead would make every speaker's short
    utterance TIMIT's SA1+SA2 -- the two dialect sentences all speakers read
    with identical text.

    speaker_ids selects the speakers explicitly (see
    EvaluationConfig.sc_speakers); otherwise the first num_speakers sorted by
    speaker id are used -- on real TIMIT that's 40 female speakers only
    (female ids start with "F"), a much harder, non-paper SC set."""
    if speaker_ids:
        # Case-insensitive: TIMIT copies differ in speaker-directory case.
        available = {speaker.upper(): speaker for speaker in corpus.speakers(split)}
        missing = [speaker for speaker in speaker_ids if speaker.upper() not in available]
        if missing:
            raise ValueError(f"SC speakers not found in {split} split: {missing}")
        speakers = [available[speaker.upper()] for speaker in speaker_ids]
    else:
        speakers = corpus.speakers(split)[:num_speakers]
    utterances = []
    for label, speaker in enumerate(speakers):
        paths = sorted(corpus.utterance_paths(split, speaker))
        for group in (paths[:-2], paths[-2:]):
            waveform = np.concatenate([corpus.load_waveform(path)[0] for path in group])
            features = featurize_waveform(waveform, transformation)
            utterances.append((features, label))
    return utterances


def _vtlp_augment(segment, rng, alpha_range, transformation):
    return vtlp_warp(segment, rng.uniform(*alpha_range), transformation)


def _load_and_featurize(load_waveform_fn, path, transformation, playback_speed=1.0):
    waveform, _ = load_waveform_fn(path)
    return featurize_waveform(change_playback_speed(waveform, playback_speed), transformation)


def _lazy_featurize_split(corpus, split, transformation, playback_speed=1.0):
    """Like _featurize_split, but each utterance is a LazyFeatures instance
    (featurized on first access, not upfront) -- see src/data/voxceleb.py's
    module docstring for why VoxCeleb's ~148k training utterances can't be
    eagerly loaded into memory the way TIMIT's ~5.5k can (VoxCeleb's
    segments are similarly numerous, see src/data/voxceleb.py).
    corpus.raw_sample_count(path) supplies each entry's frame count without
    featurizing it -- a cheap file-header probe for TIMIT/VoxCeleb's
    plain-file paths.

    A corpus with load_samples(path, start, stop) (VoxCelebCorpus) also gets
    each entry a frames_fn, so training segment draws read only the drawn
    frames' samples instead of the whole file (see featurize_frame_range) --
    unless playback_speed != 1, which needs the whole waveform to resample,
    or the transformation isn't frame-local (w2vbert)."""
    speakers = corpus.speakers(split)
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    load_samples = getattr(corpus, "load_samples", None)
    utterances = []
    for speaker in speakers:
        for path in corpus.utterance_paths(split, speaker):
            length = expected_frame_count(playback_sample_count(corpus.raw_sample_count(path), playback_speed), transformation)
            compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, path, transformation, playback_speed)
            frames_fn = None
            # w2vbert features are normalized per utterance, so not frame-local.
            if load_samples is not None and playback_speed == 1 and transformation.type != "w2vbert":
                frames_fn = functools.partial(featurize_frame_range, load_samples, path, transformation_config=transformation)
            utterances.append((LazyFeatures(compute_fn, length, frames_fn), label_map[speaker]))
    return utterances, label_map


def _lazy_trial_utterances(corpus, trial_pairs, transformation):
    """One lazy utterance entry per unique path referenced in trial_pairs,
    labeled by that relative path (not a speaker id) so
    trial_list_equal_error_rate can look embeddings back up by utterance id
    after extract_embeddings runs."""
    relative_paths = sorted({path for _, path1, path2 in trial_pairs for path in (path1, path2)})
    utterances = []
    for relative_path in relative_paths:
        full_path = corpus.trial_utterance_path(relative_path)
        length = expected_frame_count(corpus.raw_sample_count(full_path), transformation)
        compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, full_path, transformation)
        utterances.append((LazyFeatures(compute_fn, length), relative_path))
    return utterances


def _lazy_indexed_trial_utterances(corpus, transformation, segment_length):
    """IndexedTrials analogue of _lazy_trial_utterances (TidyVoiceX's
    12M-trial Dev list), restricted to trials whose both utterances are
    longer than one segment: extract_embeddings silently skips shorter ones
    (Common Voice clips can be under a second, unlike VoxCeleb's >=4s
    ones), which would otherwise leave their trials unscoreable."""
    trials = corpus.trials
    keep = np.zeros(len(trials.utterance_ids), dtype=bool)
    utterances = []
    for row, relative_path in enumerate(trials.utterance_ids):
        full_path = corpus.trial_utterance_path(relative_path)
        length = expected_frame_count(corpus.raw_sample_count(full_path), transformation)
        if length <= segment_length:
            continue
        keep[row] = True
        compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, full_path, transformation)
        utterances.append((LazyFeatures(compute_fn, length), relative_path))

    trial_mask = keep[trials.idx1] & keep[trials.idx2]
    new_row = (np.cumsum(keep) - 1).astype(np.int32)
    kept_trials = IndexedTrials(
        utterance_ids=[utterance_id for utterance_id, kept in zip(trials.utterance_ids, keep) if kept],
        labels=trials.labels[trial_mask],
        idx1=new_row[trials.idx1[trial_mask]],
        idx2=new_row[trials.idx2[trial_mask]],
    )
    print(
        f"==> Trial list: kept {len(kept_trials.labels)}/{len(trials.labels)} trials "
        f"({int((~keep).sum())} utterances too short for a {segment_length}-frame segment)"
    )
    return utterances, kept_trials


def _holdout_dev_trials(dev_utterances, segment_length):
    """Turns TRAIN's held-out dev utterances (see _lazy_featurize_train_dev_split)
    into a small trial list for periodic checkpoint selection, instead of
    equal_error_rate's exhaustive all-vs-all pairing -- at TidyVoiceX's 3,666
    training speakers that's ~27M pairs per dev eval. Per speaker: every
    same-speaker pair of its held-out utterances (target), plus each one
    paired with the next speaker's same-position one (nontarget) -- 1 target
    to 2 nontarget per speaker at the default dev_holdout_per_speaker=2,
    matching the official Dev list's ratio. Returned utterances are relabeled
    by row index, which the returned IndexedTrials refers to."""
    kept = [(features, speaker) for features, speaker in dev_utterances if features.length > segment_length]
    rows_by_speaker = {}
    for row, (_, speaker) in enumerate(kept):
        rows_by_speaker.setdefault(speaker, []).append(row)
    speakers = sorted(rows_by_speaker)
    labels, idx1, idx2 = [], [], []
    for i, speaker in enumerate(speakers):
        rows = rows_by_speaker[speaker]
        for row1, row2 in itertools.combinations(rows, 2):
            labels.append(1)
            idx1.append(row1)
            idx2.append(row2)
        if len(speakers) > 1:
            for row1, row2 in zip(rows, rows_by_speaker[speakers[(i + 1) % len(speakers)]]):
                labels.append(0)
                idx1.append(row1)
                idx2.append(row2)
    utterances = [(features, row) for row, (features, _) in enumerate(kept)]
    trials = IndexedTrials(
        utterance_ids=list(range(len(kept))),
        labels=np.array(labels, dtype=np.int8),
        idx1=np.array(idx1, dtype=np.int32),
        idx2=np.array(idx2, dtype=np.int32),
    )
    return utterances, trials


def _block_frames(strategy, segment_length):
    """Frames per shuffled block of a test strategy: 1 for SS, b for BS{b},
    the whole segment for OS (nothing shuffled). None for strategies that
    aren't block shuffles (SU, mixtures)."""
    if strategy == "OS":
        return segment_length
    if strategy == "SS":
        return 1
    if strategy.startswith("BS"):
        return int(strategy[2:])
    return None


EER_CHANCE = 0.5

# results keys of sst_bracket's measures, with their report headings.
SST_MEASURES = {
    "L": "SST lower estimate L(b) = EER(GS_b) - EER(GO_b) [EER %-points]",
    "U": "SST upper estimate U(b) = EER(BS_b) - EER(OS) [EER %-points]",
    "L_norm": "SST lower estimate, normalized L(b) / (50% - EER(OS)) [%]",
    "U_norm": "SST upper estimate, normalized U(b) / (50% - EER(OS)) [%]",
}


def sst_bracket(eers, block_size):
    """SST measures of one run at block size b, from its test EERs
    ({test strategy: EER}); each is left out when its strategies weren't
    tested:

    - L: EER(GS_b) - EER(GO_b). GO_b and GS_b have the same blocks and
      number of joins, so splicing artefacts largely cancel (GS's joins can
      be rougher, see temporal_probes.join_discontinuity), but the gaps
      already remove structure across joins in both: tends to
      underestimate the true cost T(b).
    - U: EER(BS_b) - EER(OS), the plain block shuffle: also counts the
      artefacts at its joins, so tends to overestimate T(b).
    - L_norm, U_norm: L and U divided by the OS model's headroom to chance,
      EER_chance - EER(OS) -- one common denominator, so the normalized band
      brackets T(b) / (EER_chance - EER(OS)) and compares across models of
      different baseline EER.

    L <= T(b) <= U holds if artefacts never improve EER and combine
    roughly additively with the information lost."""
    measures = {}
    go, gs, bs, os_ = (eers.get(strategy) for strategy in (f"GO{block_size}", f"GS{block_size}", f"BS{block_size}", "OS"))
    if go is not None and gs is not None:
        measures["L"] = gs - go
    if bs is not None and os_ is not None:
        measures["U"] = bs - os_
    if os_ is not None:
        for measure in ("L", "U"):
            if measure in measures:
                measures[f"{measure}_norm"] = measures[measure] / (EER_CHANCE - os_)
    return measures


def sst_bootstrap_intervals(labels, scores, num_resamples, rng, level=0.95):
    """Percentile intervals of sst_bracket's measures over num_resamples
    resamples of the trials with replacement. scores maps test strategies
    to their per-trial scores over the same trials (labels), and every
    resample is shared by all strategies, so the EER differences are paired:
    noise common to both conditions of a difference cancels. Returns
    {f"{measure}_b{b}": (low, high)}.

    Only the trial sampling is resampled: the trained model and the random
    block orders stay fixed (see the placebo and the SD over runs for
    those). Trials sharing an utterance aren't independent, so the
    intervals are somewhat too narrow."""
    labels = np.asarray(labels)
    samples = {}
    for _ in range(num_resamples):
        indices = rng.integers(0, len(labels), len(labels))
        eers = {strategy: eer_from_scores(labels[indices], np.asarray(values)[indices]) for strategy, values in scores.items()}
        for block_size in BLOCK_SIZES:
            for measure, value in sst_bracket(eers, block_size).items():
                samples.setdefault(f"{measure}_b{block_size}", []).append(value)
    tail = (1 - level) / 2
    return {key: (float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))) for key, values in samples.items()}


def _log_eer_by_block(run, key, eers_by_strategy, segment_length, transformation, title):
    """Line chart of SV EER [%] over the signal one shuffled block spans,
    (b - 1) * frame step + frame length, for the block-shuffle test strategies
    (see _block_frames). eers_by_strategy maps each test strategy to one EER
    per series, e.g. {"OS": {"run0": 0.1, "mean": 0.1}, ...}."""
    points = sorted(
        [
            (_block_frames(strategy, segment_length), eers)
            for strategy, eers in eers_by_strategy.items()
            if _block_frames(strategy, segment_length) is not None
        ],
        key=lambda point: point[0],
    )
    if len(points) < 2:
        return
    block_ms = [round(((frames - 1) * transformation.frame_step_s + transformation.frame_length_s) * 1000) for frames, _ in points]
    series = {name: [eers[name] * 100 for _, eers in points] for name in points[0][1]}
    tracking.log_line_series(run, key, block_ms, series, title, xname="signal spanned by one shuffled block [ms]")


def _sweep_key(config, train_strategy):
    run_postfix = ""
    if config.wandb.run_postfix:
        run_postfix = f"-{config.wandb.run_postfix}"
    return f"{config.model.type}-{config.data.dataset.lower()}-{train_strategy}{run_postfix}"


def _sweep_dir(config, train_strategy):
    return Path(config.training.checkpoint_dir) / _sweep_key(config, train_strategy)


def _checkpoint_path(config, train_strategy, run_idx):
    return _sweep_dir(config, train_strategy) / f"run{run_idx}.pt"


def _manifest_path(config, train_strategy):
    return _sweep_dir(config, train_strategy) / "manifest.json"


def _load_manifest(path):
    """A sweep's resume state: which wandb run to reattach to, and each
    already-completed repeat's raw SV/SC metrics (not just a "done" flag --
    needed so the aggregate mean/SD recomputed after a resume is identical to
    an uninterrupted run's)."""
    if not path.exists():
        return {"wandb_run_id": None, "completed": {}}
    with open(path) as f:
        manifest = json.load(f)
    manifest["completed"] = {int(run_idx): record for run_idx, record in manifest.get("completed", {}).items()}
    return manifest


def _save_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp_path, path)


def run_experiment(config, corpus=None, strategies=None):
    """strategies restricts which of config.training.strategies to
    train/evaluate (default: all of them) -- lets a caller split the
    strategies across separate processes/GPUs, since each is already
    independently checkpointed, manifested, and wandb-tracked (see
    _sweep_dir, _manifest_path, tracking.start_run), so running a subset
    here never races with another process running a different subset
    against the same config. Every trained model is evaluated under each of
    config.evaluation.test_strategies."""
    strategies = strategies or config.training.strategies
    test_strategies = config.evaluation.test_strategies
    unknown = [s for s in (*strategies, *test_strategies) if s not in DRAW_STRATEGIES]
    if unknown:
        raise ValueError(f"unknown draw strategies {unknown}, expected some of {list(DRAW_STRATEGIES)}")
    config.device = resolve_device(config.device)
    print(f"==> Using device: {config.device}")

    dataset_name = config.data.dataset.lower()
    transformation = config.transformation
    segment_length = config.data.segment_length(transformation)

    if dataset_name == "voxceleb":
        # Neururer et al. 2024 explicitly omits the SC task for VoxCeleb
        # ("we omit SC results on VoxCeleb as experiments in Section 2 led
        # to similar conclusions") -- sc_utterances stays None throughout.
        # SV evaluation is over a fixed trial-pairs list (the standard
        # VoxCeleb protocol), not the exhaustive all-vs-all pairing TIMIT's
        # SV task uses, hence the different eval function below.
        #
        # Checkpoint selection uses corpus.dev_trial_pairs and reported SV
        # uses corpus.trial_pairs: VoxCeleb1-O vs. VoxCeleb1-H under the
        # paper's VoxCeleb2-train protocol (context/src's
        # 04_evaluation/VOX-00_ORIGINAL.json), the same list otherwise (see
        # src/data/voxceleb.py).
        corpus = corpus or VoxCelebCorpus(config)
        train_utterances, train_label_map = _lazy_featurize_split(corpus, "TRAIN", transformation, config.training.playback_speed)
        sv_eval_utterances = _lazy_trial_utterances(corpus, corpus.trial_pairs, transformation)
        if corpus.dev_trial_pairs is corpus.trial_pairs:
            dev_utterances = sv_eval_utterances
        else:
            dev_utterances = _lazy_trial_utterances(corpus, corpus.dev_trial_pairs, transformation)
        print(
            f"==> VoxCeleb: {len(train_utterances)} training utterances of {len(train_label_map)} speakers, "
            f"{len(dev_utterances)} selection / {len(sv_eval_utterances)} evaluation trial utterances"
        )
        sc_utterances = None

        def sv_eval_fn(embeddings, utterance_ids):
            return trial_list_equal_error_rate(embeddings, utterance_ids, corpus.trial_pairs)

        def sv_min_dcf_fn(embeddings, utterance_ids):
            return trial_list_min_dcf(embeddings, utterance_ids, corpus.trial_pairs)

        def sv_scores_fn(embeddings, utterance_ids):
            return trial_list_scores(embeddings, utterance_ids, corpus.trial_pairs)

        # train()'s periodic dev-eval must also use the trial-list function --
        # plain equal_error_rate would be nonsensical here (utterance_ids are
        # unique per-utterance path strings, not speaker ids, so every pair
        # would count as "different speaker").
        def dev_eval_fn(embeddings, utterance_ids):
            return trial_list_equal_error_rate(embeddings, utterance_ids, corpus.dev_trial_pairs)

    elif dataset_name == "tidyvoicex":
        # VoxCeleb-shaped (see src/data/tidyvoicex.py): train on every Train
        # speaker, score SV over the official Dev trial list. Unlike VoxCeleb,
        # checkpoint selection/early stopping never sees the evaluation
        # trials: dev EER comes from utterances held out of TRAIN (as for
        # TIMIT), scored over a small generated trial list since
        # exhaustive pairing doesn't scale to 3,666 speakers. SC is omitted
        # -- the dataset's terms permit only verification use.
        corpus = corpus or TidyVoiceXCorpus(config)
        train_utterances, dev_pool, train_label_map = _lazy_featurize_train_dev_split(
            corpus, transformation, config.evaluation.dev_holdout_per_speaker, config.training.playback_speed
        )
        dev_utterances, dev_trials = _holdout_dev_trials(dev_pool, segment_length)
        sv_eval_utterances, eval_trials = _lazy_indexed_trial_utterances(corpus, transformation, segment_length)
        sc_utterances = None

        def sv_eval_fn(embeddings, utterance_ids):
            return indexed_trial_equal_error_rate(embeddings, utterance_ids, eval_trials)

        def sv_min_dcf_fn(embeddings, utterance_ids):
            return indexed_trial_min_dcf(embeddings, utterance_ids, eval_trials)

        def sv_scores_fn(embeddings, utterance_ids):
            return indexed_trial_scores(embeddings, utterance_ids, eval_trials, chunk_size=200_000)

        def dev_eval_fn(embeddings, utterance_ids):
            return indexed_trial_equal_error_rate(embeddings, utterance_ids, dev_trials)
    else:
        if corpus is None:
            corpus = TimitCorpus(config.data)
            corpus.check_standard_size()
        if config.evaluation.dev_speakers:
            # Speaker-disjoint dev set from TEST (the standard TIMIT
            # development speakers, see the TIMIT configs), matching
            # context/src's "development" SV list; TRAIN is trained on in full.
            train_utterances, train_label_map = _featurize_split(corpus, "TRAIN", transformation, config.training.playback_speed)
            dev_utterances = _featurize_dev_speakers(corpus, "TEST", transformation, config.evaluation.dev_speakers)
        else:
            train_utterances, dev_utterances, train_label_map = _featurize_train_dev_split(
                corpus, transformation, config.evaluation.dev_holdout_per_speaker, config.training.playback_speed
            )
        sv_eval_utterances, _ = _featurize_split(corpus, "TEST", transformation)
        sc_utterances = build_sc_utterances(
            corpus, "TEST", transformation, config.evaluation.sc_num_speakers, config.evaluation.sc_speakers
        )
        sv_eval_fn = equal_error_rate
        sv_min_dcf_fn = all_pairs_min_dcf
        sv_scores_fn = all_pairs_scores
        # train()'s own default (trainer.py's equal_error_rate) is exactly
        # this same function -- no need to route dev-eval through this
        # module's reference for TIMIT.
        dev_eval_fn = None

    results = {"SV": {}, "SV_minDCF": {}, "SV_paper_comparable": {}}
    if sc_utterances is not None:
        results["SC"] = {}

    for train_strategy in strategies:
        manifest_path = _manifest_path(config, train_strategy)
        manifest = _load_manifest(manifest_path)

        # One wandb run covers this whole strategy's config.num_runs repeats
        # plus their aggregate -- reattached via the manifest's persisted
        # run id if a previous invocation already started it (crash resume),
        # matching Neururer et al. 2024 Section 2.3's "average over 5
        # train/test runs and report mean and SD" (the aggregate is what's
        # actually of interest, not each repeat in isolation).
        run = tracking.start_run(
            config.wandb,
            name=_sweep_key(config, train_strategy),
            tags=[config.model.type, config.data.dataset.lower(), train_strategy],
            run_config=config.to_dict(),
            run_id=manifest["wandb_run_id"],
        )
        if run is not None and manifest["wandb_run_id"] is None:
            manifest["wandb_run_id"] = run.id
            _save_manifest(manifest_path, manifest)

        # Raw [0, 1] fractions from every one of config.num_runs independent
        # train/test runs for this strategy, keyed by test_strategy, reduced
        # to mean/SD below (Neururer et al. 2024 Section 2.3). Repeats already
        # recorded in the manifest (a prior invocation completed them before
        # a crash/requeue) contribute their persisted values here instead of
        # being retrained, so the aggregate is identical to an uninterrupted run.
        raw_sv = {test_strategy: [] for test_strategy in test_strategies}
        raw_sv_min_dcf = {test_strategy: [] for test_strategy in test_strategies}
        raw_sc = {test_strategy: [] for test_strategy in test_strategies} if sc_utterances is not None else None
        # Best dev EER over only the 11 epochs Neururer et al. 2024's checkpoint
        # selection actually searches (see trainer.train's paper_comparable_best) --
        # one value per run (not per test_strategy: dev eval during training only
        # ever uses draw_strategy=train_strategy). Older manifests predate this
        # field, hence the .get.
        raw_sv_paper_comparable = []
        # Per run: sst_bootstrap_intervals' intervals, and the GS placebo
        # differences by block size (see evaluate below), when computed.
        raw_sst_ci = []
        raw_sst_placebo = []
        # Run index of each entry in the raw_* lists above.
        raw_run_indices = []
        # Completed runs trained for fewer epochs than configured now continue
        # from their kept latest checkpoint (training.keep_checkpoints) and
        # are evaluated again. Their best checkpoint is then selected over all
        # epochs evaluated in either phase. Records predating "num_epochs"
        # count as fully trained.
        continued = set()
        for run_idx, completed_record in manifest["completed"].items():
            trained_epochs = completed_record.get("num_epochs", config.training.num_epochs)
            if trained_epochs < config.training.num_epochs:
                if not _checkpoint_path(config, train_strategy, run_idx).exists():
                    raise ValueError(
                        f"run {run_idx} in {manifest_path} trained {trained_epochs} of {config.training.num_epochs} "
                        "epochs, but its checkpoint is gone (set training.keep_checkpoints to keep it), so it can't "
                        "be continued"
                    )
                continued.add(run_idx)

        def collect(run_idx, record):
            for test_strategy in test_strategies:
                raw_sv[test_strategy].append(record["sv"][test_strategy])
                # Older manifests predate minDCF.
                if test_strategy in record.get("sv_min_dcf", {}):
                    raw_sv_min_dcf[test_strategy].append(record["sv_min_dcf"][test_strategy])
                if raw_sc is not None:
                    raw_sc[test_strategy].append(record["sc"][test_strategy])
            if record.get("sv_paper_comparable") is not None:
                raw_sv_paper_comparable.append(record["sv_paper_comparable"])
            if "sst_ci" in record:
                raw_sst_ci.append(record["sst_ci"])
            if "sst_placebo" in record:
                raw_sst_placebo.append(record["sst_placebo"])
            raw_run_indices.append(run_idx)

        def evaluate(model, run_idx, record, strategies_to_test):
            """Scores model under strategies_to_test into record (raw [0, 1]
            fractions) and returns the matching summary entries (percentages,
            matching Tables 1/2 in Neururer et al. 2024). One pass per
            utterance set for all of them, each with its own Generator seeded
            as a separate extract_embeddings(seed=run_idx) call would be --
            so scoring a strategy later gives the same result as scoring it
            together with the others.

            Scoring all test strategies also computes the SST bracket's
            quality checks, when configured: paired trial-bootstrap intervals
            (evaluation.sst_bootstrap_resamples) and the placebo, every GS{b}
            scored again with a second permutation seed
            (evaluation.sst_placebo)."""
            summary = {}
            started = time.perf_counter()
            print(f"==> Final evaluation: embedding {len(sv_eval_utterances)} utterances", flush=True)
            sv_by_strategy = extract_embeddings_by_strategy(
                model, sv_eval_utterances, segment_length,
                {test_strategy: np.random.default_rng(run_idx) for test_strategy in strategies_to_test},
                device=config.device, num_workers=config.training.num_workers, log_every=10_000,
            )
            print(f"==> Final evaluation: embedded in {time.perf_counter() - started:.0f}s", flush=True)
            if sc_utterances is not None:
                sc_by_strategy = extract_embeddings_by_strategy(
                    model, sc_utterances, segment_length,
                    {test_strategy: np.random.default_rng(run_idx) for test_strategy in strategies_to_test},
                    device=config.device, num_workers=config.training.num_workers,
                )
            for test_strategy in strategies_to_test:
                sv_embeddings, sv_labels = sv_by_strategy[test_strategy]
                eer = sv_eval_fn(sv_embeddings, sv_labels)
                record["sv"][test_strategy] = eer
                summary[f"run{run_idx}/final/SV_EER_test-{test_strategy}"] = eer * 100
                dcf = sv_min_dcf_fn(sv_embeddings, sv_labels)
                record.setdefault("sv_min_dcf", {})[test_strategy] = dcf
                summary[f"run{run_idx}/final/SV_minDCF_test-{test_strategy}"] = dcf

                if sc_utterances is not None:
                    sc_embeddings, sc_labels = sc_by_strategy[test_strategy]
                    mr = best_misclassification_rate(sc_embeddings, sc_labels)
                    record["sc"][test_strategy] = mr
                    summary[f"run{run_idx}/final/SC_MR_test-{test_strategy}"] = mr * 100

            if set(strategies_to_test) != set(test_strategies):
                return summary
            if config.evaluation.sst_bootstrap_resamples:
                labels, scores = None, {}
                for test_strategy in test_strategies:
                    if test_strategy == "OS" or test_strategy[:2] in ("BS", "GO", "GS"):
                        labels, scores[test_strategy] = sv_scores_fn(*sv_by_strategy[test_strategy])
                record["sst_ci"] = sst_bootstrap_intervals(
                    labels, scores, config.evaluation.sst_bootstrap_resamples, np.random.default_rng(run_idx)
                ) if scores else {}
                for key, (low, high) in record["sst_ci"].items():
                    summary[f"run{run_idx}/final/SST_{key}_ci_low"] = low * 100
                    summary[f"run{run_idx}/final/SST_{key}_ci_high"] = high * 100
            if config.evaluation.sst_placebo:
                gs_strategies = [test_strategy for test_strategy in test_strategies if test_strategy.startswith("GS")]
                # The same windows and blocks as the first GS scoring, in another random order.
                placebo_by_strategy = extract_embeddings_by_strategy(
                    model, sv_eval_utterances, segment_length,
                    {test_strategy: np.random.default_rng((run_idx, 1)) for test_strategy in gs_strategies},
                    device=config.device, num_workers=config.training.num_workers, log_every=10_000,
                ) if gs_strategies else {}
                record["sst_placebo"] = {
                    f"b{test_strategy[2:]}": sv_eval_fn(*placebo_by_strategy[test_strategy]) - record["sv"][test_strategy]
                    for test_strategy in gs_strategies
                }
                for block, value in record["sst_placebo"].items():
                    summary[f"run{run_idx}/final/SST_placebo_{block}"] = value * 100
            return summary

        def log_run(run_idx, record, summary):
            tracking.log_summary(run, summary)
            _log_eer_by_block(
                run, f"final/SV_EER_by_block/run{run_idx}",
                {test_strategy: {f"run{run_idx}": eer} for test_strategy, eer in record["sv"].items()},
                segment_length, transformation, f"SV EER [%] by shuffled block span, run {run_idx}",
            )

        for run_idx, completed_record in manifest["completed"].items():
            if run_idx in continued:
                continue
            missing = [t for t in test_strategies if t not in completed_record["sv"]]
            quality_missing = (
                (config.evaluation.sst_bootstrap_resamples and "sst_ci" not in completed_record)
                or (config.evaluation.sst_placebo and "sst_placebo" not in completed_record)
            )
            if missing or quality_missing:
                # The quality checks need every strategy's scores, which the
                # manifest doesn't keep: score them all again.
                if config.evaluation.sst_bootstrap_resamples or config.evaluation.sst_placebo:
                    missing = list(test_strategies)
                # Scored from the run's kept best weights (training.keep_checkpoints).
                best_path = best_checkpoint_path(_checkpoint_path(config, train_strategy, run_idx))
                if not best_path.exists():
                    raise ValueError(
                        f"{manifest_path} has completed runs never evaluated under {missing}, and their best "
                        "checkpoint is gone; set a different wandb.run_postfix to train a fresh sweep"
                    )
                print(f"==> Evaluating completed run {run_idx} under {missing}", flush=True)
                model = build_model(config)
                model.load_state_dict(torch.load(best_path, map_location=config.device)["model_state_dict"])
                model.to(config.device)
                log_run(run_idx, completed_record, evaluate(model, run_idx, completed_record, missing))
                _save_manifest(manifest_path, manifest)
            collect(run_idx, completed_record)

        for run_idx in range(config.num_runs):
            if run_idx in manifest["completed"] and run_idx not in continued:
                continue
            if run_idx in continued:
                print(f"==> Continuing run {run_idx} to {config.training.num_epochs} epochs", flush=True)

            train_segment_length = config.data.train_segment_length(transformation, config.training.playback_speed)
            augment_fn = None
            if config.training.vtlp_alpha_range:
                if transformation.normalization != "frame":
                    raise ValueError("vtlp_alpha_range needs transformation.normalization 'frame' (see vtlp_warp)")
                augment_fn = functools.partial(
                    _vtlp_augment, alpha_range=config.training.vtlp_alpha_range, transformation=transformation
                )
            dataset = SegmentDataset(train_utterances, train_segment_length, train_strategy, augment_fn=augment_fn)
            model = build_model(config)
            loss_module = build_loss(config, bottleneck_dim=512, num_speakers=len(train_label_map))
            checkpoint_path = _checkpoint_path(config, train_strategy, run_idx)
            # dev_utterances (TIMIT: held out from TRAIN, never gradient-trained on or
            # touched again until here; VoxCeleb: the trial-pairs pool, already excluded
            # from training -- see run_experiment's is_voxceleb branch) + draw_strategy=
            # train_strategy matches context/src's periodic dev-set EER checkpointing
            # (Neururer et al. 2024's reported numbers come from the best such
            # checkpoint, not the final epoch), while keeping TEST/sv_eval_utterances
            # itself untouched until final scoring below.
            _, paper_comparable_best = train(
                model, loss_module, dataset, config,
                dev_utterances=dev_utterances, segment_length=segment_length, draw_strategy=train_strategy,
                run=run, run_idx=run_idx, device=config.device,
                checkpoint_path=checkpoint_path,
                checkpoint_every_epochs=config.training.checkpoint_every_epochs,
                early_stopping_patience=config.training.early_stopping_patience,
                min_improvement_rate=config.training.early_stopping_min_improvement_rate,
                dev_eval_fn=dev_eval_fn,
            )

            record = {"sv": {}, "sv_min_dcf": {}, "num_epochs": config.training.num_epochs}
            if sc_utterances is not None:
                record["sc"] = {}
            record["sv_paper_comparable"] = paper_comparable_best
            summary = evaluate(model, run_idx, record, test_strategies)
            if paper_comparable_best is not None:
                summary[f"run{run_idx}/final/SV_EER_paper_comparable"] = paper_comparable_best * 100
            log_run(run_idx, record, summary)
            collect(run_idx, record)

            # Persisted before deleting the checkpoint: this run_idx is now
            # durably done, so a crash from here on skips straight past it.
            manifest["completed"][run_idx] = record
            _save_manifest(manifest_path, manifest)
            if not config.training.keep_checkpoints:
                for path in (checkpoint_path, best_checkpoint_path(checkpoint_path)):
                    if path.exists():
                        path.unlink()

        # Once all of this strategy's runs are accounted for (freshly run or
        # already in the manifest), log the mean/SD over config.num_runs runs
        # to this same run's summary (Neururer et al. 2024 Section 2.3).
        aggregate_summary = {}
        for test_strategy in test_strategies:
            sv_stats = _run_statistics(raw_sv[test_strategy])
            results["SV"][(train_strategy, test_strategy)] = sv_stats
            aggregate_summary[f"final/SV_EER_test-{test_strategy}_mean"] = sv_stats.mean * 100
            aggregate_summary[f"final/SV_EER_test-{test_strategy}_std"] = sv_stats.std * 100
            if raw_sv_min_dcf[test_strategy]:
                dcf_stats = _run_statistics(raw_sv_min_dcf[test_strategy])
                results["SV_minDCF"][(train_strategy, test_strategy)] = dcf_stats
                aggregate_summary[f"final/SV_minDCF_test-{test_strategy}_mean"] = dcf_stats.mean
                aggregate_summary[f"final/SV_minDCF_test-{test_strategy}_std"] = dcf_stats.std

            if raw_sc is not None:
                sc_stats = _run_statistics(raw_sc[test_strategy])
                results["SC"][(train_strategy, test_strategy)] = sc_stats
                aggregate_summary[f"final/SC_MR_test-{test_strategy}_mean"] = sc_stats.mean * 100
                aggregate_summary[f"final/SC_MR_test-{test_strategy}_std"] = sc_stats.std * 100
        # Per run first, then mean/SD over runs (not measures of mean EERs).
        for block_size in BLOCK_SIZES:
            per_run = [
                sst_bracket({test_strategy: raw_sv[test_strategy][i] for test_strategy in test_strategies}, block_size)
                for i in range(len(raw_run_indices))
            ]
            for measure in SST_MEASURES:
                if per_run and measure in per_run[0]:
                    stats = _run_statistics([values[measure] for values in per_run])
                    results.setdefault(f"SST_{measure}", {})[(train_strategy, f"b{block_size}")] = stats
                    aggregate_summary[f"final/SST_{measure}_b{block_size}_mean"] = stats.mean * 100
                    aggregate_summary[f"final/SST_{measure}_b{block_size}_std"] = stats.std * 100
        for block_size in BLOCK_SIZES:
            block = f"b{block_size}"
            placebo = [values[block] for values in raw_sst_placebo if block in values]
            if placebo:
                stats = _run_statistics(placebo)
                results.setdefault("SST_placebo", {})[(train_strategy, block)] = stats
                aggregate_summary[f"final/SST_placebo_{block}_mean"] = stats.mean * 100
                aggregate_summary[f"final/SST_placebo_{block}_std"] = stats.std * 100
            for measure in SST_MEASURES:
                half_widths = [(high - low) / 2 for low, high in (ci[f"{measure}_{block}"] for ci in raw_sst_ci if f"{measure}_{block}" in ci)]
                if half_widths:
                    stats = _run_statistics(half_widths)
                    results.setdefault(f"SST_{measure}_ci", {})[(train_strategy, block)] = stats
                    aggregate_summary[f"final/SST_{measure}_{block}_ci_halfwidth_mean"] = stats.mean * 100
        if raw_sv_paper_comparable:
            paper_stats = _run_statistics(raw_sv_paper_comparable)
            results["SV_paper_comparable"][train_strategy] = paper_stats
            aggregate_summary["final/SV_EER_paper_comparable_mean"] = paper_stats.mean * 100
            aggregate_summary["final/SV_EER_paper_comparable_std"] = paper_stats.std * 100
        tracking.log_summary(run, aggregate_summary)
        _log_eer_by_block(
            run, "final/SV_EER_by_block",
            {
                test_strategy: {
                    **{f"run{i}": eer for i, eer in zip(raw_run_indices, raw_sv[test_strategy])},
                    "mean": statistics.fmean(raw_sv[test_strategy]),
                }
                for test_strategy in test_strategies
            },
            segment_length, transformation, f"SV EER [%] by shuffled block span, {train_strategy}-trained",
        )
        tracking.finish(run)

    return results


def format_results(results):
    """Reports EER/MR as `mean% σstd%` (mean and SD over config.num_runs
    runs, as percentages), matching Tables 1/2 in Neururer et al. 2024;
    `results` itself stores RunStatistics of raw [0, 1] fractions. "SC" is
    absent for VoxCeleb runs (Neururer et al. 2024 omits the SC task there),
    so only tasks actually present in `results` are reported. minDCF is
    already normalized and is reported unscaled. The SST_* tables report
    sst_bracket's measures per block size b, also on a 0-100 scale, with
    `±` the trial-bootstrap 95% interval's half-width (mean over runs) when
    computed; SST_placebo the GS placebo (see run_experiment's evaluate)."""
    lines = []
    tasks = [("SV", "SV (EER) [%]", 100, "test"), ("SV_minDCF", "SV (minDCF)", 1, "test"), ("SC", "SC (MR) [%]", 100, "test")]
    tasks += [(f"SST_{measure}", heading, 100, "block") for measure, heading in SST_MEASURES.items()]
    tasks += [("SST_placebo", "SST placebo P(b) = EER(GS_b, 2nd seed) - EER(GS_b) [EER %-points]", 100, "block")]
    for task, heading, scale, column in tasks:
        if not results.get(task):
            continue
        # Only trained strategies have rows -- a --strategy-restricted run
        # (see run_experiment) leaves the others absent from results[task].
        trained_strategies = list(dict.fromkeys(s for s, _ in results[task]))
        test_strategies = list(dict.fromkeys(t for _, t in results[task]))
        ci = results.get(f"{task}_ci", {})
        rows = {}
        for train_strategy in trained_strategies:
            rows[train_strategy] = []
            for test_strategy in test_strategies:
                stats = results[task][(train_strategy, test_strategy)]
                cell = f"{stats.mean * scale:6.2f} σ{stats.std * scale:5.2f}"
                if (train_strategy, test_strategy) in ci:
                    cell += f" ±{ci[(train_strategy, test_strategy)].mean * scale:5.2f}"
                rows[train_strategy].append(cell)
        width = max(14, *(len(cell) for row in rows.values() for cell in row))
        lines.append(f"{heading}:")
        header = "train\\" + column
        lines.append(f"{header:<11} " + "  ".join(f"{s:>{width}}" for s in test_strategies))
        for train_strategy, row in rows.items():
            lines.append(f"{train_strategy:<11} " + "  ".join(f"{cell:>{width}}" for cell in row))
        lines.append("")

    if results.get("SV_paper_comparable"):
        # Dev EER of train_strategy against itself, best-of-11 fixed epochs
        # only -- directly comparable to Neururer et al. 2024's own
        # checkpoint-selection search (see trainer.train's paper_comparable_best),
        # unlike the SV table above which searches every epoch.
        lines.append("SV (dev EER, paper-comparable best-of-11) [%]:")
        for train_strategy, stats in results["SV_paper_comparable"].items():
            lines.append(f"{train_strategy:<11} {stats.mean:6.2f} σ{stats.std:5.2f}")
        lines.append("")
    return "\n".join(lines)


def main():
    # `kill -USR1 <pid>` prints every thread's Python stack to the log (also
    # in DataLoader workers, which inherit the handler) -- the cluster doesn't
    # allow attaching a debugger such as py-spy.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    parser.add_argument(
        "--strategy", choices=list(DRAW_STRATEGIES), default=None,
        help="restrict this invocation to a single train_strategy (default: all of the config's "
             "training.strategies), so multiple invocations can run concurrently on different GPUs "
             "against the same config",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    strategies = (args.strategy,) if args.strategy else None
    results = run_experiment(config, strategies=strategies)
    print(format_results(results))


if __name__ == "__main__":
    main()
