"""Manual/visual inspection tool for the OS/SS/SU segment-drawing strategies:
writes the full featurized utterance and each strategy's drawn segment to disk
as .npy arrays and spectrogram heatmap .png images.

Usage: python -m src.debug_segments path/to/utterance.WAV output_dir/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .config import TransformationConfig
from .data.dataset import featurize_waveform
from .data.segments import DRAW_STRATEGIES


def _save_spectrogram_png(features, path, title):
    fig, ax = plt.subplots()
    ax.imshow(features.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def dump_segments(wav_path, output_dir, segment_length, seed=None, transformation=None):
    """Writes the full featurized utterance plus its OS/SS/SU segment draws
    (each as a .npy array and a spectrogram .png) to output_dir."""
    transformation = transformation or TransformationConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, _ = sf.read(str(wav_path), dtype="float32")
    features = featurize_waveform(waveform, transformation)
    np.save(output_dir / "full.npy", features)
    _save_spectrogram_png(features, output_dir / "full.png", "full utterance")

    for name, draw_fn in DRAW_STRATEGIES.items():
        segment = draw_fn(features, segment_length, np.random.default_rng(seed))
        np.save(output_dir / f"{name}.npy", segment)
        _save_spectrogram_png(segment, output_dir / f"{name}.png", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", help="path to a single utterance .wav file")
    parser.add_argument("output_dir", help="directory to write .npy/.png files to")
    parser.add_argument("--segment-duration", type=float, default=1.0, help="segment length in seconds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    transformation = TransformationConfig()
    segment_length = int(args.segment_duration * transformation.steps_per_second)
    dump_segments(args.wav_path, args.output_dir, segment_length, seed=args.seed, transformation=transformation)


if __name__ == "__main__":
    main()