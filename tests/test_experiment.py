from src.config import ExperimentConfig
from src.experiment import format_results, run_experiment, select_clustering_subset
from tests.conftest import make_synthetic_waveform


def test_select_clustering_subset_limits_speakers_and_utterances():
    utterances = []
    for speaker in range(6):
        for _ in range(5):
            utterances.append((object(), speaker))

    subset = select_clustering_subset(utterances, num_speakers=3, utterances_per_speaker=2)

    speakers_present = {label for _, label in subset}
    assert speakers_present == {0, 1, 2}
    assert len(subset) == 6


class _StubCorpus:
    """Minimal TimitCorpus-shaped stub (speakers/utterance_paths/load_waveform)
    so run_experiment can be exercised end-to-end without a real (licensed)
    TIMIT copy: "paths" are (split, speaker_id, index) tuples."""

    def __init__(self, waveforms_by_split):
        self._waveforms_by_split = waveforms_by_split

    def speakers(self, split):
        return sorted(self._waveforms_by_split[split].keys())

    def utterance_paths(self, split, speaker_id):
        n = len(self._waveforms_by_split[split][speaker_id])
        return [(split, speaker_id, i) for i in range(n)]

    def load_waveform(self, path):
        split, speaker_id, i = path
        return self._waveforms_by_split[split][speaker_id][i], 16000


def test_run_experiment_end_to_end_on_synthetic_corpus():
    config = ExperimentConfig()
    config.training.num_epochs = 3
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.evaluation.sc_utterances_per_speaker = 2

    waveforms = {"TRAIN": {}, "TEST": {}}
    seed = 0
    for split in waveforms:
        for speaker in range(3):
            speaker_id = f"SPK{speaker}"
            waveforms[split][speaker_id] = [
                make_synthetic_waveform(200 + 150 * speaker, 2.0, 16000, seed=seed)
                for seed in range(seed, seed + 4)
            ]
            seed += 4

    corpus = _StubCorpus(waveforms)
    results = run_experiment(config, corpus=corpus)

    assert set(results.keys()) == {"SV", "SC"}
    for task in ("SV", "SC"):
        for train_strategy in ("OS", "SS", "SU"):
            for test_strategy in ("OS", "SS", "SU"):
                value = results[task][(train_strategy, test_strategy)]
                assert 0.0 <= value <= 1.0

    report = format_results(results)
    assert "SV (EER)" in report
    assert "SC (MR)" in report
