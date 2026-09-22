import numpy as np
import pytest
import soundfile as sf
from praatio.data_classes.interval_tier import IntervalTier
from praatio.data_classes.textgrid import Textgrid

from src.config import ExperimentConfig
from src.data.aishell4 import Aishell4Corpus, Aishell4NotAvailableError, SegmentRef


def _write_wav(path, duration_s, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sf.write(str(path), waveform, sample_rate)


def _write_textgrid(path, tiers, max_time):
    path.parent.mkdir(parents=True, exist_ok=True)
    tg = Textgrid()
    for name, entries in tiers.items():
        tg.addTier(IntervalTier(name, entries, 0, max_time))
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def _write_session(split_dir, session_id, tiers, duration_s=6.0):
    _write_wav(split_dir / "wav" / f"{session_id}.wav", duration_s)
    _write_textgrid(split_dir / "TextGrid" / f"{session_id}.TextGrid", tiers, duration_s)


@pytest.fixture
def aishell4_root(tmp_path):
    root = tmp_path / "aishell4"
    _write_session(
        root / "train_S",
        "train_sess1",
        {
            "SPK1": [(0.0, 2.0, "hello"), (3.0, 4.0, "world")],
            "SPK2": [(2.0, 3.0, "hi")],
        },
    )
    _write_session(
        root / "test",
        "test_sess1",
        {
            "SPK1": [(0.0, 2.0, "foo")],  # unrelated to train_sess1's SPK1
        },
    )
    return root


def _config_for(root):
    config = ExperimentConfig()
    config.aishell4.root = str(root)
    return config


def test_speakers_are_session_scoped(aishell4_root):
    corpus = Aishell4Corpus(_config_for(aishell4_root))

    assert corpus.speakers("TRAIN") == ["train_sess1__SPK1", "train_sess1__SPK2"]
    assert corpus.speakers("TEST") == ["test_sess1__SPK1"]


def test_utterance_paths_are_one_segment_per_non_silence_interval(aishell4_root):
    corpus = Aishell4Corpus(_config_for(aishell4_root))

    segments = corpus.utterance_paths("TRAIN", "train_sess1__SPK1")
    assert len(segments) == 2
    assert all(isinstance(segment, SegmentRef) for segment in segments)
    # 2.0s and 1.0s intervals at 16kHz.
    assert sorted(corpus.raw_sample_count(s) for s in segments) == [16000, 32000]


def test_load_waveform_reads_only_the_segment_slice(aishell4_root):
    corpus = Aishell4Corpus(_config_for(aishell4_root))

    segments = sorted(corpus.utterance_paths("TRAIN", "train_sess1__SPK2"))
    assert len(segments) == 1
    waveform, sample_rate = corpus.load_waveform(segments[0])

    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert waveform.shape[0] == corpus.raw_sample_count(segments[0])


def test_raises_when_root_has_no_train_or_test_sessions(tmp_path):
    root = tmp_path / "empty_aishell4"
    root.mkdir()
    with pytest.raises(Aishell4NotAvailableError):
        Aishell4Corpus(_config_for(root))


def test_raises_when_root_is_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("AISHELL4_ROOT", raising=False)
    config = ExperimentConfig()
    with pytest.raises(Aishell4NotAvailableError):
        Aishell4Corpus(config)
