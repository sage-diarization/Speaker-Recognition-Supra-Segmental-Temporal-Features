import os

import pytest

from src.config import DataConfig, TransformationConfig
from src.data.dataset import featurize_waveform
from src.data.timit import TimitCorpus

pytestmark = pytest.mark.skipif(
    not any(os.environ.get(var) for var in ("TIMIT_ROOT", "TIMIT_ARCHIVE_PATH", "TIMIT_DOWNLOAD_URL")),
    reason=(
        "no real TIMIT corpus configured - set TIMIT_ROOT (an extracted TRAIN/TEST "
        "tree), TIMIT_ARCHIVE_PATH (a local .zip), or TIMIT_DOWNLOAD_URL to your own "
        "licensed copy to exercise this smoke test"
    ),
)


@pytest.fixture(scope="module")
def real_corpus():
    # DataConfig fields are left blank on purpose: TimitCorpus._resolve_root falls
    # back to the TIMIT_ROOT/TIMIT_ARCHIVE_PATH/TIMIT_DOWNLOAD_URL env vars, so this
    # test never hardcodes a path and works with whatever licensed copy the machine
    # running it has configured.
    return TimitCorpus(DataConfig())


def test_nist_sphere_wav_loads_and_featurizes(real_corpus):
    speakers = real_corpus.speakers("TEST")
    assert len(speakers) >= 1

    path = real_corpus.utterance_paths("TEST", speakers[0])[0]
    waveform, sample_rate = real_corpus.load_waveform(path)
    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert waveform.shape[0] > 0

    transformation = TransformationConfig()
    features = featurize_waveform(waveform, transformation)
    assert features.shape[1] == transformation.n_mels
    assert features.shape[0] > 0