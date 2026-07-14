import zipfile
from pathlib import Path

import pytest

from src.config import DataConfig
from src.data.timit import TimitCorpus, TimitNotAvailableError


def _make_fake_timit_tree(root):
    for split, speaker in (("TRAIN", "MABC0"), ("TEST", "FXYZ0")):
        speaker_dir = Path(root) / split / "DR1" / speaker
        speaker_dir.mkdir(parents=True)
        (speaker_dir / "SA1.WAV").write_bytes(b"not-real-audio")
    return root


def test_resolves_timit_root_from_config_field(tmp_path):
    _make_fake_timit_tree(tmp_path)
    corpus = TimitCorpus(DataConfig(timit_root=str(tmp_path)))
    assert corpus.speakers("TRAIN") == ["MABC0"]
    assert corpus.speakers("TEST") == ["FXYZ0"]


def test_resolves_timit_root_from_env_var_when_config_field_is_empty(tmp_path, monkeypatch):
    _make_fake_timit_tree(tmp_path)
    monkeypatch.setenv("TIMIT_ROOT", str(tmp_path))
    corpus = TimitCorpus(DataConfig(timit_root=""))
    assert corpus.speakers("TRAIN") == ["MABC0"]


def test_extracts_and_caches_from_local_archive_path(tmp_path):
    source_dir = tmp_path / "source"
    _make_fake_timit_tree(source_dir)

    archive_path = tmp_path / "timit.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        for wav in source_dir.rglob("*.WAV"):
            zf.write(wav, wav.relative_to(source_dir))

    cache_dir = tmp_path / "cache"
    config = DataConfig(archive_path=str(archive_path), cache_dir=str(cache_dir))
    corpus = TimitCorpus(config)
    assert corpus.speakers("TRAIN") == ["MABC0"]
    assert (cache_dir / "extracted").is_dir()


def test_raises_actionable_error_when_nothing_is_configured(tmp_path):
    config = DataConfig(cache_dir=str(tmp_path / "cache"))
    with pytest.raises(TimitNotAvailableError, match="LDC-licensed"):
        TimitCorpus(config)
