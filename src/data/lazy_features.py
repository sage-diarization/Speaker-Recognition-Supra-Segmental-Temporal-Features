"""Lazy per-utterance feature loading, for corpora too large to featurize
entirely into memory upfront (see src/data/voxceleb.py -- VoxCeleb1's
~148k training utterances, at a rough average mel-spectrogram size, would be
tens of GB held eagerly in RAM). TIMIT's utterances stay plain, already
-featurized ndarrays (src/experiment.py's _featurize_split); this module is
only used for the VoxCeleb training/evaluation path.
"""


class LazyFeatures:
    """Wraps a zero-arg `compute_fn` that featurizes an utterance on first
    call, plus a `length` (expected feature-frame count) known upfront from
    just the raw waveform's sample count -- so a length-based filter (e.g.
    SegmentDataset's "too short for this segment_length" check) doesn't
    require featurizing (STFT + mel filterbank + DRC + standardize) every
    utterance just to filter out the rare short ones."""

    def __init__(self, compute_fn, length):
        self._compute_fn = compute_fn
        self.length = length

    def __call__(self):
        return self._compute_fn()


def resolve_features(entry):
    """entry is either a plain ndarray (already featurized -- TIMIT) or a
    LazyFeatures instance (featurized on demand -- VoxCeleb)."""
    return entry() if callable(entry) else entry


def utterance_length(entry):
    return entry.length if hasattr(entry, "length") else entry.shape[0]


def expected_frame_count(raw_sample_count, transformation):
    """The number of STFT frames raw_sample_count raw audio samples will
    produce, matching src/data/features.py::_stft_magnitude's
    waveform.unfold(-1, frame_length, frame_step) framing -- computed from
    just the sample count, without decoding or transforming any audio."""
    return max(0, (raw_sample_count - transformation.frame_length) // transformation.frame_step + 1)
