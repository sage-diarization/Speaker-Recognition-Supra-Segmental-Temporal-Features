"""Measures d*: the separation at which two frames of one utterance are about
as unrelated as two random frames of it. A GO{b} join splices frames gap + 1
apart, a GS{b} join frames from anywhere in the span, so a gap of at least
d* - 1 frames makes GO{b}'s joins about as foreign as GS{b}'s and keeps join
roughness out of L(b) (see temporal_probes.join_discontinuity).

Per trial utterance (corpus.trial_pairs, VoxCeleb1-H with the pretrained
configs), in the config's input features: D(d), the mean Euclidean distance
between frames d apart, for d = 1..max separation, and D_rand, the mean over
all pairs of its frames. R(d) = D(d) / D_rand is averaged over utterances,
each weighing equally, and d*_q is the smallest d with mean R(d) >= q, with a
bootstrap interval over utterances.

Usage: python -m src.measure_decorrelation --config configs/pretrained-resnet34-voxceleb-2s.yaml [--max-separation-s 3.0] [--max-frames 2000] [--output logs/decorrelation.npz]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data.voxceleb import VoxCelebCorpus
from .device import resolve_device
from .experiment import _lazy_trial_utterances
from .training.trainer import resolved_utterances

QUANTILES = (0.9, 0.95, 0.99)


def separation_profile(features, max_separation, device="cpu"):
    """(D(d) for d = 1..max_separation, D_rand) of one utterance's (frames,
    features) array, see the module docstring."""
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    length = x.shape[0]
    distances = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
    position = torch.arange(length, device=device)
    offsets = position[None, :] - position[:, None]
    upper = offsets > 0
    sums = torch.bincount(offsets[upper], weights=distances[upper].double(), minlength=length)
    pairs = length - torch.arange(1, max_separation + 1, device=device)
    profile = sums[1:max_separation + 1] / pairs
    return profile.cpu().numpy(), distances[upper].double().mean().item()


def first_reaching(ratio, quantile):
    """Smallest separation d (1-based) with ratio[d - 1] >= quantile, NaN if none."""
    reached = np.flatnonzero(ratio >= quantile)
    return float(reached[0] + 1) if reached.size else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config (VoxCeleb)")
    parser.add_argument("--max-separation-s", type=float, default=3.0, help="largest separation measured, in seconds")
    parser.add_argument("--max-frames", type=int, default=2000, help="frames of each utterance used, from its start (caps the pairwise distances' memory)")
    parser.add_argument("--resamples", type=int, default=200, help="bootstrap resamples over utterances")
    parser.add_argument("--output", default=None, help="npz with the curves (default logs/decorrelation-<config>.npz)")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    transformation = config.transformation
    frame_s = transformation.frame_period_s
    max_separation = round(args.max_separation_s / frame_s)
    device = resolve_device(config.device)
    corpus = VoxCelebCorpus(config)
    utterances = _lazy_trial_utterances(corpus, corpus.trial_pairs, transformation)
    print(f"{len(utterances)} utterances of {len(corpus.trial_pairs)} trials, {transformation.type} frames every "
          f"{frame_s * 1000:.0f} ms, separations 1..{max_separation} frames, device {device}")

    profiles, randoms, skipped = [], [], 0
    for i, (features, _) in enumerate(resolved_utterances(utterances, config.training.num_workers)):
        features = features[:args.max_frames]
        if features.shape[0] <= max_separation:
            skipped += 1
            continue
        profile, random = separation_profile(features, max_separation, device)
        profiles.append(profile)
        randoms.append(random)
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1}/{len(utterances)} utterances")
    profiles, randoms = np.array(profiles), np.array(randoms)
    ratios = profiles / randoms[:, None]
    ratio = ratios.mean(axis=0)
    print(f"{len(randoms)} utterances measured, {skipped} shorter than {max_separation + 1} frames left out")

    rng = np.random.default_rng(0)
    boot = np.array([
        np.bincount(rng.integers(0, len(ratios), len(ratios)), minlength=len(ratios)) @ ratios / len(ratios)
        for _ in range(args.resamples)
    ])
    d_star = {q: first_reaching(ratio, q) for q in QUANTILES}
    d_star_boot = {q: np.array([first_reaching(r, q) for r in boot]) for q in QUANTILES}

    print("\nseparation      R(d) = D(d) / D_rand  [95% CI]")
    for d in sorted({1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256} & set(range(1, max_separation + 1))):
        low, high = np.percentile(boot[:, d - 1], [2.5, 97.5])
        print(f"{d:5d} frames {d * frame_s * 1000:6.0f} ms   {ratio[d - 1]:.3f}  [{low:.3f}, {high:.3f}]")
    print("\nquantile  d* [frames]  d* [ms]  95% CI [ms]      min gap [frames]")
    for q in QUANTILES:
        low, high = np.nanpercentile(d_star_boot[q], [2.5, 97.5]) if np.isfinite(d_star_boot[q]).any() else (np.nan, np.nan)
        print(f"{q:8.2f}  {d_star[q]:11.0f}  {d_star[q] * frame_s * 1000:7.0f}  "
              f"[{low * frame_s * 1000:.0f}, {high * frame_s * 1000:.0f}]  {d_star[q] - 1:16.0f}")

    output = Path(args.output or f"logs/decorrelation-{Path(args.config).stem}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output, separation_frames=np.arange(1, max_separation + 1), frame_s=frame_s,
        profile=profiles.mean(axis=0), random=randoms.mean(), ratio=ratio,
        ratio_ci=np.percentile(boot, [2.5, 97.5], axis=0), quantiles=np.array(QUANTILES),
        d_star=np.array([d_star[q] for q in QUANTILES]), d_star_boot=np.array([d_star_boot[q] for q in QUANTILES]),
    )
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
