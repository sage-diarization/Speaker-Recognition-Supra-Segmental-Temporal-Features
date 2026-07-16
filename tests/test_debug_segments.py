import numpy as np
import soundfile as sf

from src.config import TransformationConfig
from src.debug_segments import dump_segments
from tests.conftest import make_synthetic_waveform


def test_dump_segments_writes_full_and_all_strategies(tmp_path):
    transformation = TransformationConfig()
    waveform = make_synthetic_waveform(220, 2.0, transformation.sample_rate, seed=0)
    segment_length = int(1.0 * transformation.steps_per_second)

    wav_path = tmp_path / "utterance.wav"
    sf.write(str(wav_path), waveform, transformation.sample_rate)

    output_dir = tmp_path / "out"
    dump_segments(wav_path, output_dir, segment_length, seed=0, transformation=transformation)

    for name in ("full", "OS", "SS", "SU"):
        assert (output_dir / f"{name}.npy").exists()
        assert (output_dir / f"{name}.png").exists()

    full = np.load(output_dir / "full.npy")
    os_segment = np.load(output_dir / "OS.npy")
    assert os_segment.shape == (segment_length, full.shape[1])