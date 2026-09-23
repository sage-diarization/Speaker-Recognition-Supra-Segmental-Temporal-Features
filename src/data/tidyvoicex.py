"""TidyVoiceX_ASV (TidyVoice 2026 challenge, Mozilla Data Collective dataset
cmihtsewu023so207xot1iqqw) corpus loader.

A multilingual Common Voice derivative curated for cross-lingual speaker
verification: speaker-disjoint Train (3,666 speakers) and Dev (808 speakers)
splits, each laid out as `<speaker_id>/<language>/<utterance>.wav` (16 kHz),
plus a separately distributed Dev trial list (`TidyVocieX_Dev_trialPairs.txt`
-- sic, from the challenge's tidyvoice_trials.zip) of `<0|1> <enroll> <test>`
lines with paths relative to the Dev directory: 12M trials (4M target, 8M
nontarget, half of each same-language, half cross-language) over ~59k
unique utterances. See the reference recipe at
https://github.com/areffarhadi/wespeaker/tree/master/examples/tidyvocie.

Structurally this is VoxCeleb-shaped rather than TIMIT-shaped: training
uses every Train speaker, and SV evaluation is over the fixed trial list, so
this exposes a single "TRAIN" split via speakers()/utterance_paths()/
load_waveform() plus the trial list -- as IndexedTrials (parallel arrays),
not VoxCelebCorpus's list of string tuples, given its 12M-trial size.

The dataset's terms forbid using it for speaker identification (only
verification is permitted), so src/experiment.py omits the SC task here,
as it already does for VoxCeleb.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from ..evaluation.verification import IndexedTrials


class TidyVoiceXNotAvailableError(RuntimeError):
    pass


_TRIALS_URL = "https://drive.google.com/file/d/1OLEKewhcGi_W_gmqEDpjx-fZwQGz2kWU"


def _is_split_dir_name(name, split_name):
    name = name.lower()
    return name == split_name or name.endswith(f"_{split_name}")


def _dirs_breadth_first(root):
    """Every directory under root (root included), shallowest first --
    extracted archives vary in nesting (the MDC download is itself a tar.gz
    that may wrap further archives), so nothing here assumes a fixed depth.
    Split directories (*_train, *_dev) are yielded but never descended into,
    so locating them never crawls their ~300k utterance files."""
    level = [Path(root)]
    while level:
        yield from level
        level = sorted(
            child
            for directory in level
            if not any(_is_split_dir_name(directory.name, split) for split in ("train", "dev"))
            for child in directory.iterdir()
            if child.is_dir()
        )


def _find_split_dir(root, split_name):
    """The shallowest directory named `<split_name>` or `*_<split_name>`
    (case-insensitive, e.g. TidyVoiceX_Train/TidyVoiceX_Dev) that actually
    holds <speaker>/<language>/*.wav files."""
    for candidate in _dirs_breadth_first(root):
        if _is_split_dir_name(candidate.name, split_name) and next(candidate.glob("*/*/*.wav"), None):
            return candidate
    raise TidyVoiceXNotAvailableError(
        f"Could not locate a *_{split_name}/<speaker>/<language>/*.wav directory under {root}"
    )


def _find_trial_file(root, trial_file):
    if trial_file:
        path = Path(trial_file).expanduser()
        path = path if path.is_absolute() else Path(root) / path
        if path.is_file():
            return path
    else:
        for directory in _dirs_breadth_first(root):
            found = sorted(directory.glob("*trialPairs*.txt"))
            if found:
                return found[0]
    raise TidyVoiceXNotAvailableError(
        f"No TidyVoiceX Dev trial list found ({trial_file or f'no *trialPairs*.txt under {root}'}). It isn't part "
        f"of the Mozilla Data Collective archive -- download tidyvoice_trials.zip from {_TRIALS_URL} and unzip it "
        "into tidyvoicex.root (or point tidyvoicex.trial_file at the extracted TidyVocieX_Dev_trialPairs.txt)."
    )


def _read_trials(path):
    row_by_id, labels, idx1, idx2 = {}, [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            label, enroll, test = parts
            labels.append(int(label))
            idx1.append(row_by_id.setdefault(enroll, len(row_by_id)))
            idx2.append(row_by_id.setdefault(test, len(row_by_id)))
    return IndexedTrials(
        utterance_ids=list(row_by_id),
        labels=np.array(labels, dtype=np.int8),
        idx1=np.array(idx1, dtype=np.int32),
        idx2=np.array(idx2, dtype=np.int32),
    )


class TidyVoiceXCorpus:
    def __init__(self, config):
        root = Path(config.tidyvoicex.root).expanduser()
        if not root.is_dir():
            raise TidyVoiceXNotAvailableError(f"{root} is not a directory (see tidyvoicex.root)")

        self._dev_root = _find_split_dir(root, "dev")
        self.trials = _read_trials(_find_trial_file(root, config.tidyvoicex.trial_file))
        dev_files = {path.relative_to(self._dev_root).as_posix() for path in self._dev_root.glob("*/*/*.wav")}
        missing = [utterance_id for utterance_id in self.trials.utterance_ids if utterance_id not in dev_files]
        if missing:
            raise TidyVoiceXNotAvailableError(
                f"{len(missing)} trial-list utterances not found under {self._dev_root} (e.g. {missing[0]})"
            )

        train_root = _find_split_dir(root, "train")
        train_utterances = {}
        for wav_path in sorted(train_root.glob("*/*/*.wav")):
            train_utterances.setdefault(wav_path.relative_to(train_root).parts[0], []).append(wav_path)
        self._train_utterances = train_utterances

    def speakers(self, split):
        assert split == "TRAIN", "TidyVoiceXCorpus only has a TRAIN split -- evaluation is trial-based (see trials)"
        return sorted(self._train_utterances.keys())

    def utterance_paths(self, split, speaker_id):
        assert split == "TRAIN"
        return self._train_utterances[speaker_id]

    def trial_utterance_path(self, relative_path):
        return self._dev_root / relative_path

    @staticmethod
    def raw_sample_count(path):
        return sf.info(str(path)).frames

    @staticmethod
    def load_waveform(path):
        waveform, sample_rate = sf.read(str(path), dtype="float32")
        return waveform, sample_rate
