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

    Exposes a single "TRAIN" split via the same speakers()/utterance_paths()/
    load_waveform() interface as TimitCorpus (so src/experiment.py's
    TIMIT-shaped helpers apply unchanged), plus trial_pairs -- there is no
    per-speaker "TEST" split, since VoxCeleb evaluation is defined over the
    fixed trial-pairs list, not a set of per-speaker test utterances (and
    Neururer et al. 2024 explicitly omits the SC task for VoxCeleb, so no
    analogous SC utterance grouping is needed either)."""

    def __init__(self, config):
        root = Path(config.voxceleb.root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        verification = VoxCeleb1Verification(root=str(root), meta_url=config.voxceleb.trial_meta_url, download=True)
        self.trial_pairs = [(label, path1, path2) for label, path1, path2 in verification._flist]

        self._wav_root = root / "wav"
        eval_speakers = {path.split("/")[0] for _, path1, path2 in self.trial_pairs for path in (path1, path2)}

        train_utterances = {}
        for wav_path in sorted(self._wav_root.glob("*/*/*.wav")):
            speaker_id = wav_path.relative_to(self._wav_root).parts[0]
            if speaker_id in eval_speakers:
                continue
            train_utterances.setdefault(speaker_id, []).append(wav_path)
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

    @staticmethod
    def load_waveform(path):
        waveform, sample_rate = sf.read(str(path), dtype="float32")
        return waveform, sample_rate
