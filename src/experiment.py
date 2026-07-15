import argparse
import statistics
from typing import NamedTuple

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


def select_clustering_subset(utterances, num_speakers, utterances_per_speaker):
    """Picks a fixed number of speakers and utterances/speaker for the SC
    task (a reasonable-fidelity stand-in for the paper's exact "2 and 8
    concatenated sentences per 40 speakers" protocol, whose original TIMIT
    list files aren't recoverable from context/src)."""
    speakers = sorted({label for _, label in utterances})[:num_speakers]
    subset = []
    for speaker in speakers:
        speaker_utterances = [u for u in utterances if u[1] == speaker]
        subset.extend(speaker_utterances[:utterances_per_speaker])
    return subset


def run_experiment(config, corpus=None):
    config.device = resolve_device(config.device)
    print(f"==> Using device: {config.device}")

    corpus = corpus or TimitCorpus(config.data)
    transformation = config.transformation
    segment_length = config.data.segment_length(transformation)

    train_utterances, train_label_map = _featurize_split(corpus, "TRAIN", transformation)
    test_utterances, _ = _featurize_split(corpus, "TEST", transformation)

    sc_utterances = select_clustering_subset(
        test_utterances, config.evaluation.sc_num_speakers, config.evaluation.sc_utterances_per_speaker
    )

    results = {"SV": {}, "SC": {}}
    for train_strategy in STRATEGIES:
        # Raw [0, 1] fractions from every one of config.num_runs independent
        # train/test runs for this strategy, keyed by test_strategy, reduced
        # to mean/SD below (Neururer et al. 2024 Section 2.3).
        raw_sv = {test_strategy: [] for test_strategy in STRATEGIES}
        raw_sc = {test_strategy: [] for test_strategy in STRATEGIES}

        for run_idx in range(config.num_runs):
            run = tracking.start_run(
                config.wandb,
                name=f"{config.model.type}-{train_strategy}-run{run_idx}",
                tags=[config.model.type, train_strategy],
                run_config=config.to_dict(),
            )

            dataset = SegmentDataset(train_utterances, segment_length, train_strategy)
            model = build_model(config)
            loss_module = build_loss(config, bottleneck_dim=512, num_speakers=len(train_label_map))
            train(model, loss_module, dataset, config, run=run, device=config.device)

            # summary reports this single run's numbers as percentages, matching
            # Tables 1/2 in Neururer et al. 2024 and context/src/train.py's console output.
            summary = {}
            for test_strategy in STRATEGIES:
                sv_embeddings, sv_labels = extract_embeddings(
                    model, test_utterances, segment_length, test_strategy, device=config.device
                )
                eer = equal_error_rate(sv_embeddings, sv_labels)
                raw_sv[test_strategy].append(eer)
                summary[f"final/SV_EER_test-{test_strategy}"] = eer * 100

                sc_embeddings, sc_labels = extract_embeddings(
                    model, sc_utterances, segment_length, test_strategy, device=config.device
                )
                mr = best_misclassification_rate(sc_embeddings, sc_labels)
                raw_sc[test_strategy].append(mr)
                summary[f"final/SC_MR_test-{test_strategy}"] = mr * 100

            tracking.log_summary(run, summary)
            tracking.finish(run)

        # Once all of this strategy's runs are done, log the mean/SD over
        # config.num_runs runs to their own wandb run (Neururer et al. 2024
        # Section 2.3: "we average over 5 train/test runs and report mean and SD").
        aggregate_run = tracking.start_run(
            config.wandb,
            name=f"{config.model.type}-{train_strategy}-aggregate",
            tags=[config.model.type, train_strategy, "aggregate"],
            run_config=config.to_dict(),
        )
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
        tracking.log_summary(aggregate_run, aggregate_summary)
        tracking.finish(aggregate_run)

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
