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

    def __init__(self, compute_fn, length, frames_fn=None):
        self._compute_fn = compute_fn
        self.length = length
        # Optional frames_fn(start, stop) featurizing only frames [start, stop)
        # (see src/data/dataset.py's featurize_frame_range) -- lets training
        # draw a segment by reading just its samples, not the whole file.
        self._frames_fn = frames_fn

    def __call__(self):
        return self._compute_fn()

    @property
    def windowed(self):
        return self._frames_fn is not None

    @property
    def shape(self):
        # Only the time axis is known without featurizing -- all that
        # src/data/segments.py's draw functions read from `.shape`.
        return (self.length,)

    def __getitem__(self, key):
        """Contiguous frame slices only (all src/data/segments.py's draw
        functions take), featurized via frames_fn."""
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("LazyFeatures supports only contiguous frame slices")
        start, stop, _ = key.indices(self.length)
        return self._frames_fn(start, stop)


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
