import numpy as np
import pytest
import soundfile as sf

from src.config import ExperimentConfig
from src.data.tidyvoicex import TidyVoiceXCorpus, TidyVoiceXNotAvailableError

TRIAL_FILE_NAME = "TidyVocieX_Dev_trialPairs.txt"


def _write_wav(path, duration_s=2.0, sample_rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sf.write(str(path), waveform, sample_rate)


def make_tidyvoicex_root(root, train, dev, trial_lines, duration_s=2.0):
    """Builds a TidyVoiceX_ASV-shaped tree: <root>/TidyVoiceX_ASV/TidyVoiceX_{Train,Dev}/
    <speaker>/<language>/<utterance>.wav (nested one level, as the MDC archive
    extracts), plus the trial list at <root>/TidyVocieX_Dev_trialPairs.txt.
    `train`/`dev` map speaker -> list of "<language>/<utterance>.wav"."""
    for split_name, speakers in (("TidyVoiceX_Train", train), ("TidyVoiceX_Dev", dev)):
        for speaker, relative_paths in speakers.items():
            for relative_path in relative_paths:
                _write_wav(root / "TidyVoiceX_ASV" / split_name / speaker / relative_path, duration_s)
    (root / TRIAL_FILE_NAME).write_text("".join(f"{line}\n" for line in trial_lines))
    return root


def _config_for(root, trial_file=""):
    config = ExperimentConfig()
    config.tidyvoicex.root = str(root)
    config.tidyvoicex.trial_file = trial_file
    return config


@pytest.fixture
def tidyvoicex_root(tmp_path):
    return make_tidyvoicex_root(
        tmp_path / "tidyx",
        train={"id000001": ["en/en_1.wav", "de/de_2.wav"], "id000002": ["fr/fr_3.wav"]},
        dev={"id014001": ["en/en_10.wav", "de/de_11.wav"], "id014002": ["en/en_12.wav"]},
        trial_lines=[
            "1 id014001/en/en_10.wav id014001/de/de_11.wav",
            "0 id014001/en/en_10.wav id014002/en/en_12.wav",
        ],
    )


def test_train_speakers_and_utterances_come_from_the_train_split(tidyvoicex_root):
    corpus = TidyVoiceXCorpus(_config_for(tidyvoicex_root))

    assert corpus.speakers("TRAIN") == ["id000001", "id000002"]
    assert [p.name for p in corpus.utterance_paths("TRAIN", "id000001")] == ["de_2.wav", "en_1.wav"]


def test_trials_are_indexed_over_unique_dev_utterances(tidyvoicex_root):
    corpus = TidyVoiceXCorpus(_config_for(tidyvoicex_root))
    trials = corpus.trials

    assert trials.utterance_ids == ["id014001/en/en_10.wav", "id014001/de/de_11.wav", "id014002/en/en_12.wav"]
    np.testing.assert_array_equal(trials.labels, [1, 0])
    np.testing.assert_array_equal(trials.idx1, [0, 0])
    np.testing.assert_array_equal(trials.idx2, [1, 2])

    waveform, sample_rate = corpus.load_waveform(corpus.trial_utterance_path(trials.utterance_ids[0]))
    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert corpus.raw_sample_count(corpus.trial_utterance_path(trials.utterance_ids[0])) == waveform.shape[0]


def test_explicit_trial_file_is_resolved_relative_to_root(tidyvoicex_root):
    (tidyvoicex_root / TRIAL_FILE_NAME).rename(tidyvoicex_root / "custom.txt")
    corpus = TidyVoiceXCorpus(_config_for(tidyvoicex_root, trial_file="custom.txt"))
    assert len(corpus.trials.labels) == 2


def test_missing_trial_file_points_at_the_separate_download(tidyvoicex_root):
    (tidyvoicex_root / TRIAL_FILE_NAME).unlink()
    with pytest.raises(TidyVoiceXNotAvailableError, match="tidyvoice_trials.zip"):
        TidyVoiceXCorpus(_config_for(tidyvoicex_root))


def test_trial_utterance_missing_from_dev_split_raises(tidyvoicex_root):
    (tidyvoicex_root / TRIAL_FILE_NAME).write_text("1 id014001/en/en_10.wav id099999/en/en_99.wav\n")
    with pytest.raises(TidyVoiceXNotAvailableError, match="id099999/en/en_99.wav"):
        TidyVoiceXCorpus(_config_for(tidyvoicex_root))


def test_missing_root_raises(tmp_path):
    with pytest.raises(TidyVoiceXNotAvailableError, match="not a directory"):
        TidyVoiceXCorpus(_config_for(tmp_path / "does-not-exist"))


def test_only_a_train_split_is_exposed(tidyvoicex_root):
    corpus = TidyVoiceXCorpus(_config_for(tidyvoicex_root))
    with pytest.raises(AssertionError):
        corpus.speakers("TEST")
