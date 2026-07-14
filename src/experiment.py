import argparse

from .config import ExperimentConfig
from .data.dataset import SegmentDataset, featurize_waveform
from .data.timit import TimitCorpus
from .evaluation.clustering import best_misclassification_rate
from .evaluation.verification import equal_error_rate
from .models.losses import build_loss
from .models.registry import build_model
from .training.trainer import extract_embeddings, train

STRATEGIES = ("OS", "SS", "SU")


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
        dataset = SegmentDataset(train_utterances, segment_length, train_strategy)
        model = build_model(config)
        loss_module = build_loss(config, bottleneck_dim=512, num_speakers=len(train_label_map))
        train(model, loss_module, dataset, config)

        for test_strategy in STRATEGIES:
            sv_embeddings, sv_labels = extract_embeddings(model, test_utterances, segment_length, test_strategy)
            results["SV"][(train_strategy, test_strategy)] = equal_error_rate(sv_embeddings, sv_labels)

            sc_embeddings, sc_labels = extract_embeddings(model, sc_utterances, segment_length, test_strategy)
            results["SC"][(train_strategy, test_strategy)] = best_misclassification_rate(sc_embeddings, sc_labels)

    return results


def format_results(results):
    lines = []
    for task in ("SV", "SC"):
        lines.append(f"{task} ({'EER' if task == 'SV' else 'MR'}):")
        lines.append("train\\test  " + "  ".join(f"{s:>8}" for s in STRATEGIES))
        for train_strategy in STRATEGIES:
            row = [f"{results[task][(train_strategy, s)]:8.4f}" for s in STRATEGIES]
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
