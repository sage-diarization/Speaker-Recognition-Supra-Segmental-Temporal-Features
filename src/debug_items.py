"""Visual inspection tool for SegmentDataset items: plots one utterance's full
featurized spectrogram above the OS/SS/SU items SegmentDataset draws from it,
on a shared diverging color scale (features are per-frame z-scored, so 0 is
the meaningful center). The source frames of each drawn item are marked under
the full utterance, located by exact row match against the full features.
With --only-strategies, just the items of the given strategies are plotted.
With --no-normalisation, the per-frame z-scoring is skipped and the
dynamic-range-compressed spectrogram is plotted on a sequential color scale.

Usage: python -m src.debug_items path/to/utterance.WAV output.png [--config configs/X.yaml] [--only-strategies os ss su] [--no-normalisation]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .config import ExperimentConfig, TransformationConfig
from .data.dataset import SegmentDataset
from .data.features import apply_drc, compute_linear_spectrogram, compute_mel_spectrogram, normalise_standardize
from .data.segments import DRAW_STRATEGIES


def _source_frames(full, segment):
    """Index into full of each segment frame (first exact row match)."""
    matches = (segment[:, None, :] == full[None, :, :]).all(axis=-1)
    return matches.argmax(axis=1)


def _imshow(ax, features, transformation, color_scale):
    """Plots (frames, freqs) features with time in seconds on x."""
    return ax.imshow(
        features.T, aspect="auto", origin="lower", interpolation="nearest", **color_scale,
        extent=[0, features.shape[0] * transformation.frame_step_s, 0, features.shape[1]],
    )


def plot_items(wav_path, output_path, segment_length, seed=None, transformation=None, only_strategies=None,
               normalise=True):
    """Draws one item per strategy from a SegmentDataset holding only the
    utterance at wav_path and writes the comparison figure to output_path.
    If only_strategies is given, plots just those strategies' items, without
    the full utterance and source-frame panels. If normalise is False, the
    features are featurized like featurize_waveform minus its per-frame
    standardization."""
    transformation = transformation or TransformationConfig()
    waveform, _ = sf.read(str(wav_path), dtype="float32")
    if transformation.type == "linear":
        full = apply_drc(compute_linear_spectrogram(waveform, transformation))
    else:
        full = apply_drc(compute_mel_spectrogram(waveform, transformation))
    if normalise:
        full = normalise_standardize(full)
    label = Path(wav_path).parent.name

    items = {}
    for name in only_strategies or DRAW_STRATEGIES:
        dataset = SegmentDataset([(full, label)], segment_length, name, seed=seed)
        tensor, _ = dataset[0]
        items[name] = tensor[0].numpy()  # drop channel dim -> (frames, freqs)

    shown = list(items.values()) if only_strategies else [full, *items.values()]
    if normalise:
        vmax = max(np.abs(s).max() for s in shown)
        color_scale = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        color_label = "z-score (per frame)"
    else:
        color_scale = dict(cmap="magma", vmin=0, vmax=max(s.max() for s in shown))
        # DRC output is uncalibrated (see apply_drc), so only relative level is shown.
        color_label = None # "level"
    freq_label = "mel bin" if transformation.type == "mel" else "freq bin"

    if only_strategies:
        fig = plt.figure(figsize=(14, 4), layout="constrained")
        grid = fig.add_gridspec(1, len(items))
        fig.suptitle(f"{Path(wav_path).name} (speaker {label})")
        item_row = 0
    else:
        fig = plt.figure(figsize=(14, 8), layout="constrained")
        grid = fig.add_gridspec(3, len(items), height_ratios=[3, 1, 3])
        item_row = 2

        ax_full = fig.add_subplot(grid[0, :])
        _imshow(ax_full, full, transformation, color_scale)
        ax_full.set(title=f"full utterance: {Path(wav_path).name} (speaker {label})", ylabel=freq_label)

        # One row per strategy marking which frames of the full utterance it drew.
        ax_src = fig.add_subplot(grid[1, :], sharex=ax_full)
        for row, (name, segment) in enumerate(items.items()):
            times = _source_frames(full, segment) * transformation.frame_step_s
            ax_src.eventplot(times, lineoffsets=row, linelengths=0.8, linewidths=0.5, color=f"C{row}")
        ax_src.set(yticks=range(len(items)), yticklabels=list(items), xlabel="time [s]",
                   title="source frames of each item")
        ax_src.invert_yaxis()

    for col, (name, segment) in enumerate(items.items()):
        ax = fig.add_subplot(grid[item_row, col])
        im = _imshow(ax, segment, transformation, color_scale)
        ax.set(title=f"{name} item {tuple(segment.shape)}", xlabel="time [s]",
               ylabel=freq_label if col == 0 else None)

    colorbar = fig.colorbar(im, ax=fig.axes, label=color_label, shrink=0.6)
    if not normalise:
        colorbar.set_ticks([color_scale["vmin"], color_scale["vmax"]], labels=["low", "high"])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", help="path to a single utterance .wav file")
    parser.add_argument("output_path", help="path of the .png figure to write")
    parser.add_argument("--config", help="configs/*.yaml to take the transformation and segment duration from (defaults otherwise)")
    parser.add_argument("--segment-duration", type=float, help="segment length in seconds (overrides the config's data.segment_duration)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-strategies", nargs="+", type=str.upper, choices=list(DRAW_STRATEGIES),
                        help="plot just these strategies' items (os/ss/su), without the full utterance and source frames")
    parser.add_argument("--no-normalisation", action="store_true",
                        help="skip the per-frame z-scoring and plot the dynamic-range-compressed spectrogram")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    if args.segment_duration is not None:
        config.data.segment_duration = args.segment_duration
    transformation = config.transformation
    segment_length = config.data.segment_length(transformation)
    plot_items(args.wav_path, args.output_path, segment_length, seed=args.seed, transformation=transformation,
               only_strategies=args.only_strategies, normalise=not args.no_normalisation)


if __name__ == "__main__":
    main()
