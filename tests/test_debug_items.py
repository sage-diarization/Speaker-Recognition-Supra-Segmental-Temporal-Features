import soundfile as sf

from src.config import TransformationConfig
from src.debug_items import plot_items
from tests.conftest import make_synthetic_waveform


def test_plot_items_writes_figure(tmp_path):
    transformation = TransformationConfig()
    waveform = make_synthetic_waveform(220, 2.0, transformation.sample_rate, seed=0)
    segment_length = int(1.0 * transformation.steps_per_second)

    wav_path = tmp_path / "utterance.wav"
    sf.write(str(wav_path), waveform, transformation.sample_rate)

    output_path = tmp_path / "items.png"
    plot_items(wav_path, output_path, segment_length, seed=0, transformation=transformation)

    assert output_path.stat().st_size > 0


def test_plot_items_only_strategies_writes_figure(tmp_path):
    transformation = TransformationConfig()
    waveform = make_synthetic_waveform(220, 2.0, transformation.sample_rate, seed=0)
    segment_length = int(1.0 * transformation.steps_per_second)

    wav_path = tmp_path / "utterance.wav"
    sf.write(str(wav_path), waveform, transformation.sample_rate)

    output_path = tmp_path / "items.png"
    plot_items(wav_path, output_path, segment_length, seed=0, transformation=transformation,
               only_strategies=["OS", "SU"])

    assert output_path.stat().st_size > 0


def test_plot_items_without_normalisation_writes_figure(tmp_path):
    transformation = TransformationConfig()
    waveform = make_synthetic_waveform(220, 2.0, transformation.sample_rate, seed=0)
    segment_length = int(1.0 * transformation.steps_per_second)

    wav_path = tmp_path / "utterance.wav"
    sf.write(str(wav_path), waveform, transformation.sample_rate)

    output_path = tmp_path / "items.png"
    plot_items(wav_path, output_path, segment_length, seed=0, transformation=transformation, normalise=False)

    assert output_path.stat().st_size > 0
