import numpy as np
import pytest
import soundfile as sf

from src.config import ExperimentConfig
from src.data import voxceleb as voxceleb_module
from src.data.voxceleb import VoxCelebCorpus


class _FakeVoxCeleb1Verification:
    """Stands in for torchaudio.datasets.VoxCeleb1Verification so this test
    never touches the network -- VoxCelebCorpus only reads its `_flist`
    attribute and relies on it to have downloaded/extracted `root/wav/`,
    which this test builds directly instead."""

    last_kwargs = None

    def __init__(self, root, meta_url=None, download=False):
        _FakeVoxCeleb1Verification.last_kwargs = {"root": root, "meta_url": meta_url, "download": download}
        self._flist = [
            ("1", "id10001/clipA/00001.wav", "id10001/clipB/00001.wav"),
            ("0", "id10001/clipA/00001.wav", "id10002/clipA/00001.wav"),
        ]


def _write_wav(path, duration_s=2.0, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sf.write(str(path), waveform, sample_rate)


@pytest.fixture
def voxceleb_root(tmp_path, monkeypatch):
    monkeypatch.setattr(voxceleb_module, "VoxCeleb1Verification", _FakeVoxCeleb1Verification)
    root = tmp_path / "voxceleb"
    wav_root = root / "wav"
    # id10001 and id10002 appear in the fake trial pairs above; id10003 does not.
    for speaker, clip in [("id10001", "clipA"), ("id10001", "clipB"), ("id10002", "clipA"), ("id10003", "clipA")]:
        _write_wav(wav_root / speaker / clip / "00001.wav")
    return root


def test_train_speakers_exclude_every_trial_pair_speaker(voxceleb_root):
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    corpus = VoxCelebCorpus(config)

    assert corpus.speakers("TRAIN") == ["id10003"]
    assert len(corpus.utterance_paths("TRAIN", "id10003")) == 1


def test_trial_pairs_and_utterance_loading(voxceleb_root):
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    corpus = VoxCelebCorpus(config)

    assert corpus.trial_pairs == [
        ("1", "id10001/clipA/00001.wav", "id10001/clipB/00001.wav"),
        ("0", "id10001/clipA/00001.wav", "id10002/clipA/00001.wav"),
    ]

    path = corpus.trial_utterance_path("id10001/clipA/00001.wav")
    waveform, sample_rate = corpus.load_waveform(path)
    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert waveform.shape[0] > 0


def test_speakers_and_utterance_paths_reject_non_train_split(voxceleb_root):
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    corpus = VoxCelebCorpus(config)

    with pytest.raises(AssertionError):
        corpus.speakers("TEST")
    with pytest.raises(AssertionError):
        corpus.utterance_paths("TEST", "id10003")


def test_verification_downloader_receives_configured_root_and_trial_url(voxceleb_root):
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    config.voxceleb.trial_meta_url = "https://example.invalid/custom_trial_list.txt"
    VoxCelebCorpus(config)

    assert _FakeVoxCeleb1Verification.last_kwargs == {
        "root": str(voxceleb_root),
        "meta_url": "https://example.invalid/custom_trial_list.txt",
        "download": True,
    }


def test_raises_when_every_speaker_is_a_trial_speaker(tmp_path, monkeypatch):
    class _AllSpeakersAreEvalSpeakers(_FakeVoxCeleb1Verification):
        def __init__(self, root, meta_url=None, download=False):
            self._flist = [("1", "id10003/clipA/00001.wav", "id10003/clipA/00001.wav")]

    monkeypatch.setattr(voxceleb_module, "VoxCeleb1Verification", _AllSpeakersAreEvalSpeakers)
    root = tmp_path / "voxceleb"
    _write_wav(root / "wav" / "id10003" / "clipA" / "00001.wav")

    config = ExperimentConfig()
    config.voxceleb.root = str(root)
    with pytest.raises(RuntimeError, match="no non-evaluation speakers"):
        VoxCelebCorpus(config)


_TRIALS_BY_URL = {
    "https://example.invalid/o.txt": [("1", "id10001/clipA/00001.wav", "id10001/clipB/00001.wav"),
                                      ("0", "id10001/clipA/00001.wav", "id10002/clipA/00001.wav")],
    "https://example.invalid/h.txt": [("1", "id10003/clipA/00001.wav", "id10003/clipA/00001.wav"),
                                      ("0", "id10003/clipA/00001.wav", "id10002/clipA/00001.wav")],
}


class _PerUrlVoxCeleb1Verification:
    def __init__(self, root, meta_url=None, download=False):
        self._flist = _TRIALS_BY_URL[meta_url]


def test_vox2_root_trains_on_voxceleb2_and_splits_selection_and_reported_trials(voxceleb_root, tmp_path, monkeypatch):
    monkeypatch.setattr(voxceleb_module, "VoxCeleb1Verification", _PerUrlVoxCeleb1Verification)
    vox2_root = tmp_path / "vox2" / "dev" / "wav"  # nesting depth doesn't matter
    for speaker, video in [("id00012", "vidA"), ("id00012", "vidB"), ("id00015", "vidA")]:
        _write_wav(vox2_root / speaker / video / "00001.wav")

    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    config.voxceleb.vox2_root = str(tmp_path / "vox2")
    config.voxceleb.trial_meta_url = "https://example.invalid/o.txt"
    config.voxceleb.eval_trial_meta_url = "https://example.invalid/h.txt"
    corpus = VoxCelebCorpus(config)

    assert corpus.speakers("TRAIN") == ["id00012", "id00015"]
    assert len(corpus.utterance_paths("TRAIN", "id00012")) == 2
    assert corpus.dev_trial_pairs == _TRIALS_BY_URL["https://example.invalid/o.txt"]
    assert corpus.trial_pairs == _TRIALS_BY_URL["https://example.invalid/h.txt"]


def test_without_eval_trial_list_selection_and_reporting_share_one_list(voxceleb_root):
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    corpus = VoxCelebCorpus(config)
    assert corpus.dev_trial_pairs is corpus.trial_pairs


def test_voxceleb1_substitute_excludes_speakers_of_both_trial_lists(voxceleb_root, monkeypatch):
    monkeypatch.setattr(voxceleb_module, "VoxCeleb1Verification", _PerUrlVoxCeleb1Verification)
    _write_wav(voxceleb_root / "wav" / "id10004" / "clipA" / "00001.wav")
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    config.voxceleb.trial_meta_url = "https://example.invalid/o.txt"
    config.voxceleb.eval_trial_meta_url = "https://example.invalid/h.txt"  # adds id10003

    assert VoxCelebCorpus(config).speakers("TRAIN") == ["id10004"]


def test_vox2_root_without_wavs_points_at_aac_conversion(voxceleb_root, tmp_path):
    aac = tmp_path / "vox2" / "id00012" / "vidA" / "00001.m4a"
    aac.parent.mkdir(parents=True)
    aac.write_bytes(b"not-real-audio")
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    config.voxceleb.vox2_root = str(tmp_path / "vox2")
    with pytest.raises(RuntimeError, match="convert it to 16 kHz mono WAV"):
        VoxCelebCorpus(config)


def test_load_samples_reads_only_the_requested_range(voxceleb_root):
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    corpus = VoxCelebCorpus(config)
    path = corpus.utterance_paths("TRAIN", "id10003")[0]
    full, _ = corpus.load_waveform(path)
    np.testing.assert_array_equal(corpus.load_samples(path, 1000, 5000), full[1000:5000])


def test_wav_index_is_built_once_then_read_from_its_cache(voxceleb_root, tmp_path, monkeypatch):
    monkeypatch.setattr(voxceleb_module, "VoxCeleb1Verification", _PerUrlVoxCeleb1Verification)
    vox2_root = tmp_path / "vox2"
    _write_wav(vox2_root / "aac" / "id00012" / "vidA" / "00001.wav", duration_s=1.5)
    _write_wav(vox2_root / "aac" / "id00015" / "vidB" / "00007.wav", duration_s=2.0)
    config = ExperimentConfig()
    config.voxceleb.root = str(voxceleb_root)
    config.voxceleb.vox2_root = str(vox2_root)
    config.voxceleb.trial_meta_url = "https://example.invalid/o.txt"
    config.voxceleb.eval_trial_meta_url = "https://example.invalid/h.txt"

    first = VoxCelebCorpus(config)
    assert (vox2_root / voxceleb_module._INDEX_FILE).exists()
    assert (voxceleb_root / voxceleb_module._INDEX_FILE).exists()

    def _no_header_reads(path):
        raise AssertionError(f"header read despite cached index: {path}")

    monkeypatch.setattr(voxceleb_module.sf, "info", _no_header_reads)
    second = VoxCelebCorpus(config)

    assert second.speakers("TRAIN") == first.speakers("TRAIN") == ["id00012", "id00015"]
    train_path = second.utterance_paths("TRAIN", "id00012")[0]
    assert train_path == vox2_root / "aac" / "id00012" / "vidA" / "00001.wav"
    assert second.raw_sample_count(train_path) == 24000
    trial_path = second.trial_utterance_path("id10001/clipA/00001.wav")
    assert second.raw_sample_count(trial_path) == 32000


def test_empty_wav_index_is_not_cached(tmp_path):
    cache_path = tmp_path / voxceleb_module._INDEX_FILE
    assert voxceleb_module._wav_index(tmp_path / "missing", cache_path, "test") == {}
    assert not cache_path.exists()
