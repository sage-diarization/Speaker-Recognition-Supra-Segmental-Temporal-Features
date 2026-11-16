"""Measures how much a pretrained speaker model combines frames d apart: the
interaction between two input patches as a function of their distance d.
Unlike the splicing tests (GO/GS, BS), the joins this adds don't matter: the
effect of each patch on its own, joins included, cancels in the interaction.

Per trial utterance (corpus.trial_pairs, VoxCeleb1-H with the pretrained
configs) its middle test window x (data.segment_duration) is embedded, and
again with a patch of `patch` frames at position i, at j = i + d, or at both
replaced by the frames at the same positions of another utterance's window
(x_i, x_j, x_ij). With f an output of the model,

    I(d) = f(x_ij) - f(x_i) - f(x_j) + f(x)

is 0 when the patches' effects add up, i.e. when nothing the model computes
depends on both patches together. R(d) = |I| / sqrt(|f(x_i) - f(x)|^2 +
|f(x_j) - f(x)|^2) scales it by the patches' own effects (0: additive).

Two outputs f:
- "frames": the mean over time of the frame-level features the model pools
  (WeSpeaker ResNet34: layer4's output; w2v-BERT 2.0: the adapters' output).
  A mean is additive, so any interaction comes from the model mixing frames
  over time, the temporal integration this probe is after.
- "embedding": the speaker embedding. Its statistics pooling (standard
  deviation, attention weights) also makes far apart frames interact, even
  in a model that never mixes frames, so read it against "frames".

R(d) is averaged over `pairs` random positions i per distance and utterance,
then over utterances, each weighing equally, with a bootstrap interval over
utterances. The floor is x embedded again in a forward pass of its own,
relative to the patches' effects: GPU arithmetic noise, below which
R(d) means nothing. TF32 is off for this probe.

Usage: python -m src.measure_interaction --config configs/pretrained-resnet34-voxceleb-2s.yaml [--utterances 2000] [--output logs/interaction.npz]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data.voxceleb import VoxCelebCorpus
from .device import resolve_device
from .experiment import _lazy_trial_utterances
from .models.pretrained_w2vbert import W2VBertSV
from .models.pretrained_wespeaker import WeSpeakerResNet34
from .models.registry import build_model
from .training.trainer import resolved_utterances

DISTANCES_S = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 1.8)
LEVELS = ("frames", "embedding")


def interaction_windows(window, donor, distances, patch, pairs, rng):
    """The windows one utterance's interactions need, (1 + 3 * len(distances)
    * pairs, time, features): x, then per distance d and pair x_i, x_j and
    x_ij with i drawn uniformly so both patches fit. Patches take donor's
    frames at the same positions."""
    length = window.shape[0]
    windows = [window]
    for d in distances:
        if d < patch or d + patch > length:
            raise ValueError(f"distance {d} needs non-overlapping patches of {patch} frames in {length} frames")
        for i in rng.integers(0, length - patch - d + 1, pairs):
            for positions in ((i,), (i + d,), (i, i + d)):
                patched = window.copy()
                for p in positions:
                    patched[p:p + patch] = donor[p:p + patch]
                windows.append(patched)
    return np.stack(windows)


def interaction_ratios(outputs, num_distances, pairs, repeat):
    """R(d) per distance (mean over the pairs) and the floor, from one
    utterance's outputs (windows, dim) in interaction_windows' order and
    repeat, x's output from a forward pass of its own."""
    base = outputs[0]
    f_i, f_j, f_ij = (outputs[1:].reshape(num_distances, pairs, 3, -1)[:, :, k] for k in range(3))
    effect = np.sqrt(np.sum((f_i - base) ** 2, axis=-1) + np.sum((f_j - base) ** 2, axis=-1))
    interaction = np.linalg.norm(f_ij - f_i - f_j + base, axis=-1)
    ratios = (interaction / effect).mean(axis=1)
    floor = np.linalg.norm(repeat - base) / effect.mean()
    return ratios, floor


class PooledFrames:
    """Captures the mean over time of the frame-level features a pretrained
    model pools, on every forward pass while active (see the module docstring)."""

    def __init__(self, model):
        self.value = None
        for module in model.modules():
            if isinstance(module, WeSpeakerResNet34):
                # (batch, channels, freq, time) -> (batch, channels * freq)
                self.handle = module.layer4.register_forward_hook(
                    lambda m, args, out: setattr(self, "value", out.mean(dim=-1).flatten(1)))
                return
            if isinstance(module, W2VBertSV):
                # (batch, time, dim) -> (batch, dim)
                self.handle = module.pooling.register_forward_pre_hook(
                    lambda m, args: setattr(self, "value", args[0].mean(dim=1)))
                return
        raise ValueError(f"no pooled frame features known for {type(model).__name__}")

    def remove(self):
        self.handle.remove()


def embed_levels(model, pooled, windows, batch_size, device):
    """{level: (windows, dim)} of LEVELS for windows (n, time, features)."""
    outputs = {level: [] for level in LEVELS}
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).float().unsqueeze(1).to(device)
            embedding = model(batch).backend
            outputs["frames"].append(pooled.value.double().cpu().numpy())
            outputs["embedding"].append(embedding.double().cpu().numpy())
    return {level: np.concatenate(values) for level, values in outputs.items()}


def middle(features, length):
    start = (features.shape[0] - length) // 2
    return features[start:start + length]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config of a pretrained model (VoxCeleb)")
    parser.add_argument("--utterances", type=int, default=2000, help="trial utterances measured, drawn at random")
    parser.add_argument("--patch-s", type=float, default=0.1, help="patch length in seconds")
    parser.add_argument("--distances-s", type=float, nargs="+", default=DISTANCES_S, help="distances i to j in seconds")
    parser.add_argument("--pairs", type=int, default=4, help="random positions i per distance and utterance")
    parser.add_argument("--batch-size", type=int, default=128, help="windows per forward pass")
    parser.add_argument("--resamples", type=int, default=200, help="bootstrap resamples over utterances")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="npz with the curves (default logs/interaction-<config>.npz)")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    transformation = config.transformation
    frame_s = transformation.frame_period_s
    length = config.data.segment_length(transformation)
    patch = round(args.patch_s / frame_s)
    distances = sorted({round(d / frame_s) for d in args.distances_s})
    device = resolve_device(config.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(config).to(device).eval()
    pooled = PooledFrames(model)

    rng = np.random.default_rng(args.seed)
    corpus = VoxCelebCorpus(config)
    utterances = _lazy_trial_utterances(corpus, corpus.trial_pairs, transformation)
    # Twice as many as needed, in case some are shorter than a window.
    chosen = [utterances[k] for k in rng.permutation(len(utterances))[:2 * args.utterances]]
    windows = [middle(features, length) for features, _ in resolved_utterances(chosen, config.training.num_workers)
               if features.shape[0] >= length][:args.utterances]
    print(f"{len(windows)} utterances, {transformation.type} frames every {frame_s * 1000:.0f} ms, windows of {length} "
          f"frames, patches of {patch}, distances {distances} frames, {args.pairs} pairs each, device {device}")

    ratios = {level: [] for level in LEVELS}
    floors = {level: [] for level in LEVELS}
    for k, window in enumerate(windows):
        # The donor: the next utterance's window (another trial utterance, usually another speaker).
        batch = interaction_windows(window, windows[(k + 1) % len(windows)], distances, patch, args.pairs, rng)
        outputs = embed_levels(model, pooled, batch, args.batch_size, device)
        repeat = embed_levels(model, pooled, window[None], 1, device)
        for level in LEVELS:
            r, floor = interaction_ratios(outputs[level], len(distances), args.pairs, repeat[level][0])
            ratios[level].append(r)
            floors[level].append(floor)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(windows)} utterances", flush=True)
    pooled.remove()

    saved = {"distance_frames": np.array(distances), "frame_s": frame_s, "patch_frames": patch, "window_frames": length}
    for level in LEVELS:
        values = np.array(ratios[level])  # (utterances, distances)
        boot = np.array([values[rng.integers(0, len(values), len(values))].mean(axis=0) for _ in range(args.resamples)])
        mean, ci = values.mean(axis=0), np.percentile(boot, [2.5, 97.5], axis=0)
        floor = float(np.mean(floors[level]))
        print(f"\n{level}: R(d) = |I| / |patch effects|  [95% CI], floor {floor:.2e}")
        for d, m, (low, high) in zip(distances, mean, ci.T):
            print(f"{d:5d} frames {d * frame_s * 1000:6.0f} ms   {m:.4f}  [{low:.4f}, {high:.4f}]")
        saved.update({f"ratios_{level}": values, f"ratio_{level}": mean, f"ratio_ci_{level}": ci, f"floor_{level}": floor})

    output = Path(args.output or f"logs/interaction-{Path(args.config).stem}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **saved)
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
