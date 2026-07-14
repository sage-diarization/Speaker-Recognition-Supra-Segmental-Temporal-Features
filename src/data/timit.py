import os
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf


class TimitNotAvailableError(RuntimeError):
    pass


_NOT_AVAILABLE_MSG = (
    "TIMIT is an LDC-licensed corpus (catalog.ldc.upenn.edu/LDC93S1) with no "
    "legitimate free redistribution mirror, so it cannot be auto-downloaded from "
    "a hardcoded URL. Point this at your own licensed copy by setting one of:\n"
    "  - data.timit_root / TIMIT_ROOT: an already-extracted TIMIT directory\n"
    "  - data.archive_path / TIMIT_ARCHIVE_PATH: a local .zip of your TIMIT copy\n"
    "  - data.download_url / TIMIT_DOWNLOAD_URL: a URL to YOUR OWN hosted copy"
)


def _find_split_roots(root):
    """TIMIT archives vary in nesting (e.g. data/lisa/data/timit/raw/TIMIT/...),
    so locate TRAIN/TEST by name instead of assuming a fixed depth."""
    splits = {}
    for path in Path(root).rglob("*"):
        if path.is_dir() and path.name.upper() in ("TRAIN", "TEST") and path.name.upper() not in splits:
            splits[path.name.upper()] = path
    if "TRAIN" not in splits or "TEST" not in splits:
        raise TimitNotAvailableError(f"Could not locate TRAIN/ and TEST/ under {root}")
    return splits


class TimitCorpus:
    def __init__(self, config):
        self.root = self._resolve_root(config)
        self._splits = _find_split_roots(self.root)
        self._utterances = {split: self._scan_split(path) for split, path in self._splits.items()}

    @staticmethod
    def _scan_split(split_root):
        utterances = {}
        for wav_path in split_root.rglob("*"):
            if wav_path.suffix.upper() != ".WAV":
                continue
            speaker_id = wav_path.parent.name
            utterances.setdefault(speaker_id, []).append(wav_path)
        return utterances

    def _resolve_root(self, config):
        timit_root = config.timit_root or os.environ.get("TIMIT_ROOT")
        if timit_root and Path(timit_root).expanduser().is_dir():
            return Path(timit_root).expanduser()

        cache_dir = Path(config.cache_dir).expanduser()
        extracted = cache_dir / "extracted"
        if extracted.is_dir() and any(extracted.iterdir()):
            return extracted

        archive_path = config.archive_path or os.environ.get("TIMIT_ARCHIVE_PATH")
        url = config.download_url or os.environ.get("TIMIT_DOWNLOAD_URL")
        if not archive_path and not url:
            raise TimitNotAvailableError(_NOT_AVAILABLE_MSG)

        cache_dir.mkdir(parents=True, exist_ok=True)
        if not archive_path:
            archive_path = cache_dir / "timit_archive.zip"
            urllib.request.urlretrieve(url, archive_path)

        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extracted)
        return extracted

    def speakers(self, split):
        return sorted(self._utterances[split.upper()].keys())

    def utterance_paths(self, split, speaker_id):
        return self._utterances[split.upper()][speaker_id]

    @staticmethod
    def load_waveform(path):
        waveform, sample_rate = sf.read(str(path), dtype="float32")
        return waveform, sample_rate
