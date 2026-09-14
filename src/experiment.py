import argparse
import json
import os
import statistics
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import tracking
from .config import ExperimentConfig
from .data.dataset import SegmentDataset, featurize_waveform
from .data.timit import TimitCorpus
from .device import resolve_device
from .evaluation.clustering import best_misclassification_rate
from .evaluation.verification import equal_error_rate
from .models.losses import build_loss
from .models.registry import build_model
from .training.trainer import extract_embeddings, train

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


def run_experiment(config, corpus=None):
    config.device = resolve_device(config.device)
    print(f"==> Using device: {config.device}")

    corpus = corpus or TimitCorpus(config.data)
    transformation = config.transformation
    segment_length = config.data.segment_length(transformation)

    train_utterances, train_label_map = _featurize_split(corpus, "TRAIN", transformation)
    test_utterances, _ = _featurize_split(corpus, "TEST", transformation)

    sc_utterances = build_sc_utterances(corpus, "TEST", transformation, config.evaluation.sc_num_speakers)

    results = {"SV": {}, "SC": {}}
    for train_strategy in STRATEGIES:
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
        raw_sc = {test_strategy: [] for test_strategy in STRATEGIES}
        for completed_record in manifest["completed"].values():
            for test_strategy in STRATEGIES:
                raw_sv[test_strategy].append(completed_record["sv"][test_strategy])
                raw_sc[test_strategy].append(completed_record["sc"][test_strategy])

        for run_idx in range(config.num_runs):
            if run_idx in manifest["completed"]:
                continue

            dataset = SegmentDataset(train_utterances, segment_length, train_strategy)
            model = build_model(config)
            loss_module = build_loss(config, bottleneck_dim=512, num_speakers=len(train_label_map))
            checkpoint_path = _checkpoint_path(config, train_strategy, run_idx)
            # dev_utterances=test_utterances + draw_strategy=train_strategy matches
            # context/src's periodic dev-set EER checkpointing (Neururer et al. 2024's
            # reported numbers come from the best such checkpoint, not the final epoch).
            train(
                model, loss_module, dataset, config,
                dev_utterances=test_utterances, segment_length=segment_length, draw_strategy=train_strategy,
                run=run, run_idx=run_idx, device=config.device,
                checkpoint_path=checkpoint_path,
                checkpoint_every_epochs=config.training.checkpoint_every_epochs,
                early_stopping_patience=config.training.early_stopping_patience,
            )

            # summary reports this single run's numbers as percentages, matching
            # Tables 1/2 in Neururer et al. 2024 and context/src/train.py's console output.
            summary = {}
            record = {"sv": {}, "sc": {}}
            for test_strategy in STRATEGIES:
                sv_embeddings, sv_labels = extract_embeddings(
                    model, test_utterances, segment_length, test_strategy, device=config.device
                )
                eer = equal_error_rate(sv_embeddings, sv_labels)
                raw_sv[test_strategy].append(eer)
                record["sv"][test_strategy] = eer
                summary[f"run{run_idx}/final/SV_EER_test-{test_strategy}"] = eer * 100

                sc_embeddings, sc_labels = extract_embeddings(
                    model, sc_utterances, segment_length, test_strategy, device=config.device
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

        # Once all of this strategy's runs are accounted for (freshly run or
        # already in the manifest), log the mean/SD over config.num_runs runs
        # to this same run's summary (Neururer et al. 2024 Section 2.3).
        aggregate_summary = {}
        for test_strategy in STRATEGIES:
            sv_stats = _run_statistics(raw_sv[test_strategy])
            sc_stats = _run_statistics(raw_sc[test_strategy])
            results["SV"][(train_strategy, test_strategy)] = sv_stats
            results["SC"][(train_strategy, test_strategy)] = sc_stats
            aggregate_summary[f"final/SV_EER_test-{test_strategy}_mean"] = sv_stats.mean * 100
            aggregate_summary[f"final/SV_EER_test-{test_strategy}_std"] = sv_stats.std * 100
            aggregate_summary[f"final/SC_MR_test-{test_strategy}_mean"] = sc_stats.mean * 100
            aggregate_summary[f"final/SC_MR_test-{test_strategy}_std"] = sc_stats.std * 100
        tracking.log_summary(run, aggregate_summary)
        tracking.finish(run)

    return results


def format_results(results):
    """Reports EER/MR as `mean% σstd%` (mean and SD over config.num_runs
    runs, as percentages), matching Tables 1/2 in Neururer et al. 2024;
    `results` itself stores RunStatistics of raw [0, 1] fractions."""
    lines = []
    for task in ("SV", "SC"):
        lines.append(f"{task} ({'EER' if task == 'SV' else 'MR'}) [%]:")
        lines.append("train\\test  " + "  ".join(f"{s:>14}" for s in STRATEGIES))
        for train_strategy in STRATEGIES:
            row = []
            for test_strategy in STRATEGIES:
                stats = results[task][(train_strategy, test_strategy)]
                row.append(f"{stats.mean * 100:6.2f} σ{stats.std * 100:5.2f}")
            lines.append(f"{train_strategy:<11} " + "  ".join(row))
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    results = run_experiment(config)
    print(format_results(results))


if __name__ == "__main__":
    main()
