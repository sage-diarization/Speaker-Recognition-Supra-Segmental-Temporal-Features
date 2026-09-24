import argparse
import functools
import itertools
import json
import os
import statistics
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import tracking
from .config import ExperimentConfig
from .data.aishell4 import Aishell4Corpus
from .data.dataset import SegmentDataset, featurize_frame_range, featurize_waveform
from .data.lazy_features import LazyFeatures, expected_frame_count
from .data.tidyvoicex import TidyVoiceXCorpus
from .data.timit import TimitCorpus
from .data.voxceleb import VoxCelebCorpus
from .device import resolve_device
from .evaluation.clustering import best_misclassification_rate
from .evaluation.verification import (
    IndexedTrials,
    equal_error_rate,
    indexed_trial_equal_error_rate,
    trial_list_equal_error_rate,
)
from .models.losses import build_loss
from .models.registry import build_model
from .training.trainer import best_checkpoint_path, extract_embeddings, train

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


def _featurize_split(corpus, split, transformation):
    speakers = corpus.speakers(split)
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    utterances = []
    for speaker in speakers:
        for path in corpus.utterance_paths(split, speaker):
            waveform, _ = corpus.load_waveform(path)
            features = featurize_waveform(waveform, transformation)
            utterances.append((features, label_map[speaker]))
    return utterances, label_map


def _lazy_featurize_train_dev_split(corpus, transformation, dev_holdout_per_speaker):
    """Lazy analogue of _featurize_train_dev_split (see its docstring) for a
    corpus too large to eagerly featurize upfront -- currently AISHELL-4's
    TRAIN split, whose per-speaker TextGrid-turn segments run far more
    numerous than TIMIT's ~10 sentences/speaker."""
    speakers = corpus.speakers("TRAIN")
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    train_utterances, dev_utterances = [], []
    for speaker in speakers:
        paths = sorted(corpus.utterance_paths("TRAIN", speaker))
        for i, path in enumerate(paths):
            length = expected_frame_count(corpus.raw_sample_count(path), transformation)
            compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, path, transformation)
            target = dev_utterances if i < dev_holdout_per_speaker else train_utterances
            target.append((LazyFeatures(compute_fn, length), label_map[speaker]))
    return train_utterances, dev_utterances, label_map


def _featurize_train_dev_split(corpus, transformation, dev_holdout_per_speaker):
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
            features = featurize_waveform(waveform, transformation)
            target = dev_utterances if i < dev_holdout_per_speaker else train_utterances
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


def _load_and_featurize(load_waveform_fn, path, transformation):
    waveform, _ = load_waveform_fn(path)
    return featurize_waveform(waveform, transformation)


def _lazy_featurize_split(corpus, split, transformation):
    """Like _featurize_split, but each utterance is a LazyFeatures instance
    (featurized on first access, not upfront) -- see src/data/voxceleb.py's
    module docstring for why VoxCeleb's ~148k training utterances can't be
    eagerly loaded into memory the way TIMIT's ~5.5k can (AISHELL-4's
    TextGrid-turn segments are similarly numerous, see src/data/aishell4.py).
    corpus.raw_sample_count(path) supplies each entry's frame count without
    featurizing it -- a cheap file-header probe for TIMIT/VoxCeleb's
    plain-file paths, or no I/O at all for AISHELL-4's segment refs, which
    already know their own sample count from the TextGrid times.

    A corpus with load_samples(path, start, stop) (VoxCelebCorpus) also gets
    each entry a frames_fn, so training segment draws read only the drawn
    frames' samples instead of the whole file (see featurize_frame_range)."""
    speakers = corpus.speakers(split)
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    load_samples = getattr(corpus, "load_samples", None)
    utterances = []
    for speaker in speakers:
        for path in corpus.utterance_paths(split, speaker):
            length = expected_frame_count(corpus.raw_sample_count(path), transformation)
            compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, path, transformation)
            frames_fn = None
            if load_samples is not None:
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


def _sweep_key(config, train_strategy):
    return f"{config.model.type}-{config.data.dataset.lower()}-{train_strategy}"


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
    """strategies restricts which of STRATEGIES to train/evaluate (default:
    all three) -- lets a caller split the 3 strategies across separate
    processes/GPUs, since each is already independently checkpointed,
    manifested, and wandb-tracked (see _sweep_dir, _manifest_path,
    tracking.start_run), so running a subset here never races with another
    process running a different subset against the same config."""
    strategies = strategies or STRATEGIES
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
        train_utterances, train_label_map = _lazy_featurize_split(corpus, "TRAIN", transformation)
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

        # train()'s periodic dev-eval must also use the trial-list function --
        # plain equal_error_rate would be nonsensical here (utterance_ids are
        # unique per-utterance path strings, not speaker ids, so every pair
        # would count as "different speaker").
        def dev_eval_fn(embeddings, utterance_ids):
            return trial_list_equal_error_rate(embeddings, utterance_ids, corpus.dev_trial_pairs)
    elif dataset_name == "aishell4":
        # AISHELL-4 has a real per-speaker TRAIN/TEST split like TIMIT (not
        # VoxCeleb's trial-pairs protocol) -- see src/data/aishell4.py's
        # module docstring for what "speaker" and "TRAIN"/"TEST" mean here --
        # so SV evaluation is the same exhaustive all-vs-all pairing TIMIT
        # uses. Segment counts are far larger than TIMIT's, hence the lazy
        # (LazyFeatures-based) featurization VoxCeleb also uses instead of
        # TIMIT's eager one.
        corpus = corpus or Aishell4Corpus(config)
        train_utterances, dev_utterances, train_label_map = _lazy_featurize_train_dev_split(
            corpus, transformation, config.evaluation.dev_holdout_per_speaker
        )
        sv_eval_utterances, _ = _lazy_featurize_split(corpus, "TEST", transformation)
        # Neururer et al. 2024's SC recipe (2-vs-8 concatenated sentences per
        # speaker) doesn't transfer to AISHELL-4's TextGrid-turn segments --
        # counts/durations per (session, tier) speaker are too irregular for
        # it to make sense -- so it's omitted here, the same way it's
        # omitted for VoxCeleb.
        sc_utterances = None
        sv_eval_fn = equal_error_rate
        # train()'s own default (trainer.py's equal_error_rate) is exactly
        # this same function -- no need to route dev-eval through this
        # module's reference, same as TIMIT.
        dev_eval_fn = None
    elif dataset_name == "tidyvoicex":
        # VoxCeleb-shaped (see src/data/tidyvoicex.py): train on every Train
        # speaker, score SV over the official Dev trial list. Unlike VoxCeleb,
        # checkpoint selection/early stopping never sees the evaluation
        # trials: dev EER comes from utterances held out of TRAIN (as for
        # TIMIT/AISHELL-4), scored over a small generated trial list since
        # exhaustive pairing doesn't scale to 3,666 speakers. SC is omitted
        # -- the dataset's terms permit only verification use.
        corpus = corpus or TidyVoiceXCorpus(config)
        train_utterances, dev_pool, train_label_map = _lazy_featurize_train_dev_split(
            corpus, transformation, config.evaluation.dev_holdout_per_speaker
        )
        dev_utterances, dev_trials = _holdout_dev_trials(dev_pool, segment_length)
        sv_eval_utterances, eval_trials = _lazy_indexed_trial_utterances(corpus, transformation, segment_length)
        sc_utterances = None

        def sv_eval_fn(embeddings, utterance_ids):
            return indexed_trial_equal_error_rate(embeddings, utterance_ids, eval_trials)

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
            train_utterances, train_label_map = _featurize_split(corpus, "TRAIN", transformation)
            dev_utterances = _featurize_dev_speakers(corpus, "TEST", transformation, config.evaluation.dev_speakers)
        else:
            train_utterances, dev_utterances, train_label_map = _featurize_train_dev_split(
                corpus, transformation, config.evaluation.dev_holdout_per_speaker
            )
        sv_eval_utterances, _ = _featurize_split(corpus, "TEST", transformation)
        sc_utterances = build_sc_utterances(
            corpus, "TEST", transformation, config.evaluation.sc_num_speakers, config.evaluation.sc_speakers
        )
        sv_eval_fn = equal_error_rate
        # train()'s own default (trainer.py's equal_error_rate) is exactly
        # this same function -- no need to route dev-eval through this
        # module's reference for TIMIT.
        dev_eval_fn = None

    results = {"SV": {}, "SV_paper_comparable": {}}
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
        raw_sv = {test_strategy: [] for test_strategy in STRATEGIES}
        raw_sc = {test_strategy: [] for test_strategy in STRATEGIES} if sc_utterances is not None else None
        # Best dev EER over only the 11 epochs Neururer et al. 2024's checkpoint
        # selection actually searches (see trainer.train's paper_comparable_best) --
        # one value per run (not per test_strategy: dev eval during training only
        # ever uses draw_strategy=train_strategy). Older manifests predate this
        # field, hence the .get.
        raw_sv_paper_comparable = []
        for completed_record in manifest["completed"].values():
            for test_strategy in STRATEGIES:
                raw_sv[test_strategy].append(completed_record["sv"][test_strategy])
                if raw_sc is not None:
                    raw_sc[test_strategy].append(completed_record["sc"][test_strategy])
            if completed_record.get("sv_paper_comparable") is not None:
                raw_sv_paper_comparable.append(completed_record["sv_paper_comparable"])

        for run_idx in range(config.num_runs):
            if run_idx in manifest["completed"]:
                continue

            dataset = SegmentDataset(train_utterances, segment_length, train_strategy)
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

            # summary reports this single run's numbers as percentages, matching
            # Tables 1/2 in Neururer et al. 2024 and context/src/train.py's console output.
            summary = {}
            record = {"sv": {}, "sc": {}} if sc_utterances is not None else {"sv": {}}
            record["sv_paper_comparable"] = paper_comparable_best
            if paper_comparable_best is not None:
                raw_sv_paper_comparable.append(paper_comparable_best)
                summary[f"run{run_idx}/final/SV_EER_paper_comparable"] = paper_comparable_best * 100
            for test_strategy in STRATEGIES:
                sv_embeddings, sv_labels = extract_embeddings(
                    model, sv_eval_utterances, segment_length, test_strategy, seed=run_idx, device=config.device
                )
                eer = sv_eval_fn(sv_embeddings, sv_labels)
                raw_sv[test_strategy].append(eer)
                record["sv"][test_strategy] = eer
                summary[f"run{run_idx}/final/SV_EER_test-{test_strategy}"] = eer * 100

                if sc_utterances is not None:
                    sc_embeddings, sc_labels = extract_embeddings(
                        model, sc_utterances, segment_length, test_strategy, seed=run_idx, device=config.device
                    )
                    mr = best_misclassification_rate(sc_embeddings, sc_labels)
                    raw_sc[test_strategy].append(mr)
                    record["sc"][test_strategy] = mr
                    summary[f"run{run_idx}/final/SC_MR_test-{test_strategy}"] = mr * 100

            tracking.log_summary(run, summary)

            # Persisted before deleting the checkpoint: this run_idx is now
            # durably done, so a crash from here on skips straight past it.
            manifest["completed"][run_idx] = record
            _save_manifest(manifest_path, manifest)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            best_path = best_checkpoint_path(checkpoint_path)
            if best_path.exists():
                best_path.unlink()

        # Once all of this strategy's runs are accounted for (freshly run or
        # already in the manifest), log the mean/SD over config.num_runs runs
        # to this same run's summary (Neururer et al. 2024 Section 2.3).
        aggregate_summary = {}
        for test_strategy in STRATEGIES:
            sv_stats = _run_statistics(raw_sv[test_strategy])
            results["SV"][(train_strategy, test_strategy)] = sv_stats
            aggregate_summary[f"final/SV_EER_test-{test_strategy}_mean"] = sv_stats.mean * 100
            aggregate_summary[f"final/SV_EER_test-{test_strategy}_std"] = sv_stats.std * 100

            if raw_sc is not None:
                sc_stats = _run_statistics(raw_sc[test_strategy])
                results["SC"][(train_strategy, test_strategy)] = sc_stats
                aggregate_summary[f"final/SC_MR_test-{test_strategy}_mean"] = sc_stats.mean * 100
                aggregate_summary[f"final/SC_MR_test-{test_strategy}_std"] = sc_stats.std * 100
        if raw_sv_paper_comparable:
            paper_stats = _run_statistics(raw_sv_paper_comparable)
            results["SV_paper_comparable"][train_strategy] = paper_stats
            aggregate_summary["final/SV_EER_paper_comparable_mean"] = paper_stats.mean * 100
            aggregate_summary["final/SV_EER_paper_comparable_std"] = paper_stats.std * 100
        tracking.log_summary(run, aggregate_summary)
        tracking.finish(run)

    return results


def format_results(results):
    """Reports EER/MR as `mean% σstd%` (mean and SD over config.num_runs
    runs, as percentages), matching Tables 1/2 in Neururer et al. 2024;
    `results` itself stores RunStatistics of raw [0, 1] fractions. "SC" is
    absent for VoxCeleb runs (Neururer et al. 2024 omits the SC task there),
    so only tasks actually present in `results` are reported."""
    lines = []
    for task in ("SV", "SC"):
        if task not in results:
            continue
        # Only trained strategies have rows -- a --strategy-restricted run
        # (see run_experiment) leaves the others absent from results[task].
        trained_strategies = [s for s in STRATEGIES if any((s, t) in results[task] for t in STRATEGIES)]
        lines.append(f"{task} ({'EER' if task == 'SV' else 'MR'}) [%]:")
        lines.append("train\\test  " + "  ".join(f"{s:>14}" for s in STRATEGIES))
        for train_strategy in trained_strategies:
            row = []
            for test_strategy in STRATEGIES:
                stats = results[task][(train_strategy, test_strategy)]
                row.append(f"{stats.mean * 100:6.2f} σ{stats.std * 100:5.2f}")
            lines.append(f"{train_strategy:<11} " + "  ".join(row))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    parser.add_argument(
        "--strategy", choices=STRATEGIES, default=None,
        help="restrict this invocation to a single train_strategy (default: run all %s), "
             "so multiple invocations can run concurrently on different GPUs against the same config" % (STRATEGIES,),
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    strategies = (args.strategy,) if args.strategy else None
    results = run_experiment(config, strategies=strategies)
    print(format_results(results))


if __name__ == "__main__":
    main()
