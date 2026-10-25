"""Checks whether GO{b}'s EER is driven by its per-utterance gap
(src/data/segments.py::gapped_segment): GO{b} spreads its blocks over the
whole utterance, so its gap -- and with it the playback speedup
(b + gap) / b -- grows with the utterance's length, and a gap of 0 is a
plain contiguous crop (OS). Hypothesis: GO2 at 3 s scores badly because
trials mix gap-0 (natural speed) with gapped (sped up) utterances.

Without --score, only counts gaps and trials from the utterance lengths
(no featurization, no model). With --score, also embeds the trial
utterances under OS, GO{b} and GS{b} as the experiment does (run 0's seed)
and reports each strategy's EER per trial group.

Usage: python -m src.check_go_gaps --config configs/pretrained-w2vbert-voxceleb-3s.yaml [--block-size 2] [--score]
"""

import argparse
from collections import Counter

import numpy as np

from .config import ExperimentConfig
from .data.voxceleb import VoxCelebCorpus
from .device import resolve_device
from .evaluation.verification import eer_from_scores, trial_list_scores
from .experiment import _lazy_trial_utterances
from .models.registry import build_model
from .training.trainer import extract_embeddings_by_strategy


def gap(length, segment_length, block_size):
    """gapped_segment's gap between blocks, in frames."""
    return (length - segment_length) // -(-segment_length // block_size)


def trial_groups(trial_pairs, gaps):
    """Boolean masks over the trials, by the gaps of their two sides."""
    gaps1 = np.array([gaps[path1] for _, path1, _ in trial_pairs])
    gaps2 = np.array([gaps[path2] for _, _, path2 in trial_pairs])
    zeros = (gaps1 == 0).astype(int) + (gaps2 == 0)
    return {
        "all": np.ones(len(trial_pairs), dtype=bool),
        "no gap-0 side": zeros == 0,
        "one gap-0 side": zeros == 1,
        "both gap-0": zeros == 2,
        "same gap": gaps1 == gaps2,
        "different gap": gaps1 != gaps2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config (VoxCeleb)")
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--score", action="store_true", help="embed the trial utterances and report EERs per trial group")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    transformation = config.transformation
    segment_length = config.data.segment_length(transformation)
    corpus = VoxCelebCorpus(config)
    trial_pairs = corpus.trial_pairs
    utterances = _lazy_trial_utterances(corpus, trial_pairs, transformation)
    lengths = {path: features.length for features, path in utterances}
    gaps = {path: gap(length, segment_length, args.block_size) for path, length in lengths.items()}

    print(f"segment_length {segment_length} frames, block_size {args.block_size}, {len(utterances)} utterances")
    print("gap [frames]  speedup  utterances")
    for value, count in sorted(Counter(gaps.values()).items()):
        print(f"{value:12d}  {(args.block_size + value) / args.block_size:6.2f}x  {count:10d}")

    groups = trial_groups(trial_pairs, gaps)
    labels = np.array([int(label) for label, _, _ in trial_pairs])
    print("\ntrial group       trials  target")
    for name, mask in groups.items():
        print(f"{name:16s} {mask.sum():7d} {labels[mask].sum():7d}")
    if not args.score:
        return

    config.device = resolve_device(config.device)
    model = build_model(config).to(config.device)
    strategies = ("OS", f"GO{args.block_size}", f"GS{args.block_size}")
    embeddings = extract_embeddings_by_strategy(
        model, utterances, segment_length, {strategy: np.random.default_rng(0) for strategy in strategies},
        device=config.device, num_workers=config.training.num_workers, log_every=10_000,
    )
    scores = {strategy: trial_list_scores(*embeddings[strategy], trial_pairs)[1] for strategy in strategies}

    print("\nSV EER [%] per trial group")
    print(f"{'trial group':16s} " + " ".join(f"{strategy:>7s}" for strategy in strategies))
    for name, mask in groups.items():
        # A group needs target and non-target trials for an EER.
        eers = [
            eer_from_scores(labels[mask], scores[strategy][mask]) * 100 if 0 < labels[mask].sum() < mask.sum() else float("nan")
            for strategy in strategies
        ]
        print(f"{name:16s} " + " ".join(f"{eer:7.2f}" for eer in eers))


if __name__ == "__main__":
    main()
