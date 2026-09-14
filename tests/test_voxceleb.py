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
