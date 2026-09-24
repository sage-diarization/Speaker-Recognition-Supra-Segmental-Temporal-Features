import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from torchaudio.datasets import VoxCeleb1Verification


class VoxCelebCorpus:
    """Speaker-disjoint train/verification corpus over VoxCeleb1, fetched via
    torchaudio's built-in downloader (torchaudio.datasets.VoxCeleb1Verification)
    rather than a custom fetcher, per this project's policy of using a
    standardized library instead of building one.

    This substitutes for Neururer et al. 2024's actual VoxCeleb2-train /
    VoxCeleb1-test protocol: torchaudio ships no VoxCeleb2 downloader, so
    training utterances are instead drawn from VoxCeleb1's own speakers,
    excluding whichever speakers appear in the verification trial list so
    train/eval speakers stay disjoint (the standard open-set verification
    setup, and how src/experiment.py builds train_utterances/trial
    evaluation utterances from this corpus). That exclusion is also why
    config.voxceleb.trial_meta_url must stay the *original* 'veri_test2.txt'
    list (40 held-out speakers) rather than the VoxSRC "hard"/"extended"
    lists (list_test_hard2.txt / list_test_all2.txt): fetched and inspected
    directly, those span 1190 of VoxCeleb1's 1251 speakers -- so excluding
    their speakers from training would leave almost nothing to train on, and
    *not* excluding them would leak most training speakers into evaluation.
    Both problems are specific to this VoxCeleb1-only substitute protocol:
    the paper's actual VoxCeleb2-train/VoxCeleb1-test setup has no such
    conflict, since the two are separate, speaker-disjoint corpora.

    Setting config.voxceleb.vox2_root instead selects the paper's own
    protocol: training on every speaker of a local VoxCeleb2 dev WAV tree
    (speaker-disjoint from VoxCeleb1 by construction, so nothing is excluded),
    checkpoint selection on trial_meta_url (VoxCeleb1-O cleaned) and reported
    SV on eval_trial_meta_url (VoxCeleb1-H cleaned) -- context/src's
    04_evaluation/VOX-00_ORIGINAL.json and the paper's Section 3.2. In both
    modes dev_trial_pairs is the checkpoint-selection list and trial_pairs
    the reported one; they're the same list object when eval_trial_meta_url
    is unset.

    Exposes a single "TRAIN" split via the same speakers()/utterance_paths()/
    load_waveform() interface as TimitCorpus (so src/experiment.py's
    TIMIT-shaped helpers apply unchanged), plus trial_pairs -- there is no
    per-speaker "TEST" split, since VoxCeleb evaluation is defined over the
    fixed trial-pairs list, not a set of per-speaker test utterances (and
    Neururer et al. 2024 explicitly omits the SC task for VoxCeleb, so no
    analogous SC utterance grouping is needed either)."""

    def __init__(self, config):
        voxceleb = config.voxceleb
        root = Path(voxceleb.root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.dev_trial_pairs = _download_trial_pairs(root, voxceleb.trial_meta_url)
        self.trial_pairs = self.dev_trial_pairs
        if voxceleb.eval_trial_meta_url:
            self.trial_pairs = _download_trial_pairs(root, voxceleb.eval_trial_meta_url)

        self._wav_root = root / "wav"
        vox1_counts = _wav_index(self._wav_root, root / _INDEX_FILE, "VoxCeleb1")
        self._sample_counts = {str(self._wav_root / path): count for path, count in vox1_counts.items()}
        if voxceleb.vox2_root:
            vox2_root = Path(voxceleb.vox2_root).expanduser()
            vox2_counts = _wav_index(vox2_root, vox2_root / _INDEX_FILE, "VoxCeleb2")
            if not vox2_counts:
                hint = ""
                if next(vox2_root.rglob("*.m4a"), None) is not None:
                    hint = " -- found .m4a files instead: VoxCeleb2 ships as AAC, convert it to 16 kHz mono WAV first (e.g. with ffmpeg)"
                raise RuntimeError(f"no VoxCeleb2 .wav files found under {vox2_root}{hint}")
            self._sample_counts.update({str(vox2_root / path): count for path, count in vox2_counts.items()})
            train_utterances = {}
            for path in sorted(vox2_counts):
                # <speaker>/<video>/<utterance>.wav, at any depth under vox2_root.
                train_utterances.setdefault(Path(path).parent.parent.name, []).append(vox2_root / path)
            self._train_utterances = train_utterances
            return

        eval_speakers = {
            path.split("/")[0]
            for pairs in (self.dev_trial_pairs, self.trial_pairs)
            for _, path1, path2 in pairs
            for path in (path1, path2)
        }

        train_utterances = {}
        for path in sorted(vox1_counts):
            parts = Path(path).parts  # <speaker>/<video>/<utterance>.wav
            if len(parts) != 3 or parts[0] in eval_speakers:
                continue
            train_utterances.setdefault(parts[0], []).append(self._wav_root / path)
        if not train_utterances:
            raise RuntimeError(
                f"no non-evaluation speakers found under {self._wav_root} -- "
                "check that trial_meta_url doesn't reference nearly every speaker in the corpus"
            )
        self._train_utterances = train_utterances

    def speakers(self, split):
        assert split == "TRAIN", "VoxCelebCorpus only has a TRAIN split -- evaluation is trial-pair-based (see trial_pairs)"
        return sorted(self._train_utterances.keys())

    def utterance_paths(self, split, speaker_id):
        assert split == "TRAIN"
        return self._train_utterances[speaker_id]

    def trial_utterance_path(self, relative_path):
        return self._wav_root / relative_path

    def raw_sample_count(self, path):
        # From the cached index (see _wav_index), not a header read per file.
        return self._sample_counts[str(path)]

    @staticmethod
    def load_waveform(path):
        waveform, sample_rate = sf.read(str(path), dtype="float32")
        return waveform, sample_rate

    @staticmethod
    def load_samples(path, start, stop):
        """Samples [start, stop) only -- a seek, not a full decode, for WAV."""
        waveform, _ = sf.read(str(path), start=start, stop=stop, dtype="float32")
        return waveform


def _download_trial_pairs(root, meta_url):
    # torchaudio downloads/extracts VoxCeleb1's wavs only once, and caches
    # each trial list under root by its file name.
    verification = VoxCeleb1Verification(root=str(root), meta_url=meta_url, download=True)
    return [(label, path1, path2) for label, path1, path2 in verification._flist]


# Cached next to each dataset: one line "<relative path>\t<sample count>" per
# .wav. Delete it to rebuild the index after the dataset changes.
_INDEX_FILE = ".sst-wav-index.tsv"
# Directory listings and header reads mostly wait on the (network) file
# system, so threads parallelize them well despite the GIL.
_INDEX_THREADS = 64


def _wav_index(root, cache_path, name):
    """{path relative to root: sample count} of every .wav under root. The
    first call lists root and reads each file's header in parallel (on a
    network file system, ~1.2M files one at a time took most of an hour),
    then writes cache_path; later calls read just that file. An empty
    result isn't cached, so a dataset added later is still picked up."""
    if cache_path.exists():
        with open(cache_path) as f:
            index = {path: int(count) for path, count in (line.rstrip("\n").split("\t") for line in f)}
        print(f"==> {name}: {len(index)} utterances (cached index {cache_path})")
        return index

    print(f"==> {name}: indexing {root} (first run only; cached in {cache_path})...")
    with ThreadPoolExecutor(_INDEX_THREADS) as pool:
        paths = _list_wavs(root, pool)
        print(f"==> {name}: found {len(paths)} .wav files, reading their lengths...")
        counts = []
        for i, count in enumerate(pool.map(lambda path: sf.info(path).frames, paths, chunksize=256), start=1):
            counts.append(count)
            if i % 100_000 == 0:
                print(f"==> {name}: {i}/{len(paths)} lengths read")
    index = {os.path.relpath(path, root): count for path, count in zip(paths, counts)}
    if index:
        # Per-process temp file: jobs starting together (e.g. OS and SS) may
        # both build the index; os.replace keeps one complete copy.
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with open(tmp_path, "w") as f:
            f.writelines(f"{path}\t{count}\n" for path, count in index.items())
        os.replace(tmp_path, cache_path)
    return index


def _list_wavs(root, pool):
    """Every .wav path under root, listing each directory level in parallel."""
    wavs, directories = [], [str(root)]
    while directories:
        next_directories = []
        for subdirectories, files in pool.map(_list_directory, directories):
            next_directories.extend(subdirectories)
            wavs.extend(files)
        directories = next_directories
    return wavs


def _list_directory(directory):
    subdirectories, wavs = [], []
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:  # e.g. VoxCeleb1's wav/ before it's downloaded
        return subdirectories, wavs
    for entry in entries:
        if entry.is_dir():
            subdirectories.append(entry.path)
        elif entry.name.endswith(".wav"):
            wavs.append(entry.path)
    return subdirectories, wavs
