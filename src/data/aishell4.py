"""AISHELL-4 (Fu et al. 2021, OpenSLR SLR111) corpus loader.

Structurally very different from TIMIT/VoxCeleb: AISHELL-4 isn't shipped as
pre-segmented per-speaker utterance files but as long multi-speaker meeting
recordings (one 8-channel .wav per session) plus a per-session TextGrid
annotating who spoke when. Two deliberate simplifications follow from that,
chosen together with the project owner rather than assumed silently:

- Speaker identity: AISHELL-4's TextGrid tiers are speaker labels *local to
  their own session* (e.g. two different sessions' "SPK1" tiers are
  unrelated people, and there is no corpus-wide speaker-ID metadata linking
  sessions to real identities). "Speaker" here is therefore defined as
  (session_id, tier_name) -- distinct sessions never share a speaker id --
  and train/test speaker-disjointness comes for free from AISHELL-4's own
  train_*/ vs test/ session split, no separate held-out-session logic needed.
- Audio channel: only the first of the (typically 8) array channels is used,
  no beamforming -- matching how TIMIT/VoxCeleb are consumed as
  single-channel audio elsewhere in this project.

Each speaker's "utterances" are the individual (non-silence) intervals of
its TextGrid tier, read directly out of the session's channel 0 via a
sample-offset slice (soundfile's start=/frames=, so a multi-hour session
file is never decoded or copied in full to extract one segment). Segment
counts run far higher than TIMIT's ~10 sentences/speaker and durations vary
widely (sub-second interjections to long turns) -- both handled by the same
"too short for this segment_length" filtering SegmentDataset/
extract_embeddings already apply to every corpus, no AISHELL-4-specific
filtering needed here.

Exposes the same speakers()/utterance_paths()/load_waveform() interface as
TimitCorpus (a real TRAIN/TEST speaker split, unlike VoxCeleb's trial-pairs
protocol -- src/experiment.py's TIMIT-shaped SV/dev-holdout helpers apply,
just via LazyFeatures given the much larger utterance count here, see
_lazy_featurize_train_dev_split), plus raw_sample_count() (this module's
segments know their own sample count directly from the TextGrid interval
times, no soundfile.info probe needed).

Following Neururer et al. 2024's own precedent of omitting the SC task for
a corpus that doesn't fit its "2 vs 8 concatenated sentences per speaker"
recipe (they do this for VoxCeleb), src/experiment.py omits SC for AISHELL-4
too: segment counts/durations per (session, tier) speaker are too irregular
for that recipe to make sense.

Assumes every TextGrid tier is a genuine per-participant speaker tier (true
of AISHELL-4's official annotation format); a local copy with additional
non-speaker tiers would need pre-filtering before pointing this at it.
"""

import os
from pathlib import Path
from typing import NamedTuple

import soundfile as sf
from praatio import textgrid as praatio_textgrid


class Aishell4NotAvailableError(RuntimeError):
    pass


_NOT_AVAILABLE_MSG = (
    "AISHELL-4 (openslr.org/111) is freely downloadable, but this project "
    "doesn't auto-fetch it (its archives are multi-gigabyte 8-channel "
    "audio). Download+extract it yourself, then point this at the result "
    "via one of:\n"
    "  - data.aishell4_root / AISHELL4_ROOT: the extracted directory tree "
    "(containing train_S/, train_M/ and/or train_L/, plus test/, each with "
    "wav/ and TextGrid/ subdirectories)"
)


class SegmentRef(NamedTuple):
    """One (non-silence) TextGrid interval: a sample-offset slice of a
    session's channel-0 audio, resolved lazily by load_waveform rather than
    materialized as its own file on disk."""

    wav_path: Path
    start_sample: int
    end_sample: int


def _find_session_dirs(root):
    """Locates every (wav/, TextGrid/) directory pair under root, classifying
    each by its parent directory's name: anything starting with "train"
    (case-insensitive -- AISHELL-4 ships train_S/train_M/train_L split by
    recording-room size; however many of these the local copy has are
    combined into one TRAIN split) contributes to TRAIN, and "test"
    contributes to TEST. Search is recursive rather than depth-assuming, for
    the same reason as TimitCorpus's _find_split_roots: real extracted
    archives vary in nesting."""
    sessions = {"TRAIN": {}, "TEST": {}}
    for wav_dir in Path(root).rglob("wav"):
        if not wav_dir.is_dir():
            continue
        textgrid_dir = wav_dir.parent / "TextGrid"
        if not textgrid_dir.is_dir():
            continue
        parent_name = wav_dir.parent.name.lower()
        if parent_name.startswith("train"):
            split = "TRAIN"
        elif parent_name == "test":
            split = "TEST"
        else:
            continue
        for wav_path in sorted(wav_dir.glob("*.wav")):
            textgrid_path = textgrid_dir / f"{wav_path.stem}.TextGrid"
            if textgrid_path.is_file():
                sessions[split][wav_path.stem] = (wav_path, textgrid_path)
    if not sessions["TRAIN"] or not sessions["TEST"]:
        raise Aishell4NotAvailableError(
            f"Could not locate any train_*/{{wav,TextGrid}} and test/{{wav,TextGrid}} directories under {root}"
        )
    return sessions


def _scan_session(session_id, wav_path, textgrid_path):
    """One (session, tier) pair per speaker in this session's TextGrid, each
    mapped to its non-silence intervals as SegmentRefs."""
    sample_rate = sf.info(str(wav_path)).samplerate
    tg = praatio_textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
    segments_by_speaker = {}
    for tier_name in tg.tierNames:
        segments = []
        for interval in tg.getTier(tier_name).entries:
            start_sample = int(round(interval.start * sample_rate))
            end_sample = int(round(interval.end * sample_rate))
            if end_sample > start_sample:
                segments.append(SegmentRef(wav_path, start_sample, end_sample))
        if segments:
            segments_by_speaker[f"{session_id}__{tier_name}"] = segments
    return segments_by_speaker


class Aishell4Corpus:
    def __init__(self, config):
        self.root = self._resolve_root(config)
        session_dirs = _find_session_dirs(self.root)
        self._utterances = {"TRAIN": {}, "TEST": {}}
        for split, sessions in session_dirs.items():
            for session_id, (wav_path, textgrid_path) in sessions.items():
                self._utterances[split].update(_scan_session(session_id, wav_path, textgrid_path))

    @staticmethod
    def _resolve_root(config):
        root = config.aishell4.root or os.environ.get("AISHELL4_ROOT")
        if not root:
            raise Aishell4NotAvailableError(_NOT_AVAILABLE_MSG)
        root = Path(root).expanduser()
        if not root.is_dir():
            raise Aishell4NotAvailableError(f"{root} is not a directory")
        return root

    def speakers(self, split):
        return sorted(self._utterances[split.upper()].keys())

    def utterance_paths(self, split, speaker_id):
        return self._utterances[split.upper()][speaker_id]

    @staticmethod
    def raw_sample_count(segment_ref):
        return segment_ref.end_sample - segment_ref.start_sample

    @staticmethod
    def load_waveform(segment_ref):
        frames = segment_ref.end_sample - segment_ref.start_sample
        data, sample_rate = sf.read(
            str(segment_ref.wav_path), start=segment_ref.start_sample, frames=frames,
            dtype="float32", always_2d=True,
        )
        return data[:, 0], sample_rate
