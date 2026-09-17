import argparse
import functools
import json
import os
import statistics
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

from . import tracking
from .config import ExperimentConfig
from .data.dataset import SegmentDataset, featurize_waveform
from .data.lazy_features import LazyFeatures, expected_frame_count
from .data.timit import TimitCorpus
from .data.voxceleb import VoxCelebCorpus
from .device import resolve_device
from .evaluation.clustering import best_misclassification_rate
from .evaluation.verification import equal_error_rate, trial_list_equal_error_rate
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


def build_sc_utterances(corpus, split, transformation, num_speakers):
    """Builds the SC task's per-speaker utterances by concatenating sentences
    rather than using single raw sentences, per Neururer et al. 2024 Section
    2.2: "2 utterances (comprising 2 and 8 concatenated sentences) per 40
    speakers". Sentences are sorted deterministically (not left in TIMIT's
    filesystem-extraction order, which is arbitrary per speaker) before being
    split into a short (first 2) and long (remaining) concatenation; for
    standard TIMIT's 10-sentences-per-speaker layout this reproduces the
    paper's 2-vs-8 split (the original speaker-list files that pick the 40
    speakers themselves aren't recoverable from context/src, so we take the
    first num_speakers sorted by speaker id instead)."""
    speakers = corpus.speakers(split)[:num_speakers]
    utterances = []
    for label, speaker in enumerate(speakers):
        paths = sorted(corpus.utterance_paths(split, speaker))
        for group in (paths[:2], paths[2:]):
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
    eagerly loaded into memory the way TIMIT's ~5.5k can."""
    speakers = corpus.speakers(split)
    label_map = {speaker: i for i, speaker in enumerate(speakers)}
    utterances = []
    for speaker in speakers:
        for path in corpus.utterance_paths(split, speaker):
            length = expected_frame_count(sf.info(str(path)).frames, transformation)
            compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, path, transformation)
            utterances.append((LazyFeatures(compute_fn, length), label_map[speaker]))
    return utterances, label_map


def _lazy_trial_utterances(corpus, transformation):
    """One lazy utterance entry per unique path referenced in the corpus's
    verification trial pairs, labeled by that relative path (not a speaker
    id) so trial_list_equal_error_rate can look embeddings back up by
    utterance id after extract_embeddings runs."""
    relative_paths = sorted({path for _, path1, path2 in corpus.trial_pairs for path in (path1, path2)})
    utterances = []
    for relative_path in relative_paths:
        full_path = corpus.trial_utterance_path(relative_path)
        length = expected_frame_count(sf.info(str(full_path)).frames, transformation)
        compute_fn = functools.partial(_load_and_featurize, corpus.load_waveform, full_path, transformation)
        utterances.append((LazyFeatures(compute_fn, length), relative_path))
    return utterances


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

    is_voxceleb = config.data.dataset.lower() == "voxceleb"
    transformation = config.transformation
    segment_length = config.data.segment_length(transformation)

    if is_voxceleb:
        # Neururer et al. 2024 explicitly omits the SC task for VoxCeleb
        # ("we omit SC results on VoxCeleb as experiments in Section 2 led
        # to similar conclusions") -- sc_utterances stays None throughout.
        # SV evaluation is over a fixed trial-pairs list (the standard
        # VoxCeleb protocol), not the exhaustive all-vs-all pairing TIMIT's
        # SV task uses, hence the different eval function below.
        corpus = corpus or VoxCelebCorpus(config)
        train_utterances, train_label_map = _lazy_featurize_split(corpus, "TRAIN", transformation)
        sv_eval_utterances = _lazy_trial_utterances(corpus, transformation)
        sc_utterances = None

        def sv_eval_fn(embeddings, utterance_ids):
            return trial_list_equal_error_rate(embeddings, utterance_ids, corpus.trial_pairs)

        # train()'s periodic dev-eval must also use the trial-list function --
        # plain equal_error_rate would be nonsensical here (utterance_ids are
        # unique per-utterance path strings, not speaker ids, so every pair
        # would count as "different speaker").
        dev_eval_fn = sv_eval_fn
    else:
        corpus = corpus or TimitCorpus(config.data)
        train_utterances, train_label_map = _featurize_split(corpus, "TRAIN", transformation)
        sv_eval_utterances, _ = _featurize_split(corpus, "TEST", transformation)
        sc_utterances = build_sc_utterances(corpus, "TEST", transformation, config.evaluation.sc_num_speakers)
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
            # dev_utterances=sv_eval_utterances + draw_strategy=train_strategy matches
            # context/src's periodic dev-set EER checkpointing (Neururer et al. 2024's
            # reported numbers come from the best such checkpoint, not the final epoch).
            _, paper_comparable_best = train(
                model, loss_module, dataset, config,
                dev_utterances=sv_eval_utterances, segment_length=segment_length, draw_strategy=train_strategy,
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
