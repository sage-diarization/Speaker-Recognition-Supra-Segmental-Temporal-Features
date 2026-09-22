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


def test_prefers_pre_converted_mono_wav_over_flac(tmp_path):
    # The official release ships <session_id>.flac (8-channel) under wav/;
    # a locally pre-converted <session_id>_16k_mono.wav sitting alongside it
    # should be used instead (cheaper to read, already single-channel).
    root = tmp_path / "aishell4"
    wav_dir = root / "train_S" / "wav"
    wav_dir.mkdir(parents=True)
    _write_wav(wav_dir / "sess1.flac", duration_s=6.0)
    _write_wav(wav_dir / "sess1_16k_mono.wav", duration_s=6.0)
    _write_textgrid(root / "train_S" / "TextGrid" / "sess1.TextGrid", {"SPK1": [(0.0, 2.0, "hi")]}, 6.0)
    _write_session(root / "test", "test_sess1", {"SPK1": [(0.0, 2.0, "foo")]})

    corpus = Aishell4Corpus(_config_for(root))

    segment = corpus.utterance_paths("TRAIN", "sess1__SPK1")[0]
    assert segment.wav_path == wav_dir / "sess1_16k_mono.wav"


def test_falls_back_to_flac_when_no_mono_wav_present(tmp_path):
    # A copy that only has the official release's own audio (no local
    # pre-conversion step) should still work, reading the .flac directly.
    root = tmp_path / "aishell4"
    wav_dir = root / "train_S" / "wav"
    wav_dir.mkdir(parents=True)
    _write_wav(wav_dir / "sess1.flac", duration_s=6.0)
    _write_textgrid(root / "train_S" / "TextGrid" / "sess1.TextGrid", {"SPK1": [(0.0, 2.0, "hi")]}, 6.0)
    _write_session(root / "test", "test_sess1", {"SPK1": [(0.0, 2.0, "foo")]})

    corpus = Aishell4Corpus(_config_for(root))

    segment = corpus.utterance_paths("TRAIN", "sess1__SPK1")[0]
    assert segment.wav_path == wav_dir / "sess1.flac"
    waveform, sample_rate = corpus.load_waveform(segment)
    assert sample_rate == 16000
    assert waveform.shape[0] == corpus.raw_sample_count(segment)


def test_ignores_rttm_sibling_files_in_textgrid_dir(tmp_path):
    # TextGrid/ also holds a same-stem .rttm per session -- must not be
    # mistaken for a second session with no matching audio.
    root = tmp_path / "aishell4"
    _write_session(root / "train_S", "sess1", {"SPK1": [(0.0, 2.0, "hi")]})
    (root / "train_S" / "TextGrid" / "sess1.rttm").write_text("SPEAKER sess1 1 0.0 2.0 <NA> <NA> SPK1 <NA> <NA>\n")
    _write_session(root / "test", "test_sess1", {"SPK1": [(0.0, 2.0, "foo")]})

    corpus = Aishell4Corpus(_config_for(root))

    assert corpus.speakers("TRAIN") == ["sess1__SPK1"]
