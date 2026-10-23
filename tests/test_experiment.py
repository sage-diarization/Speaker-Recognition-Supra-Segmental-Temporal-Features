import functools
import itertools
import json
import statistics
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src import experiment as experiment_module
from src.config import ExperimentConfig
from src.data.dataset import SegmentDataset, featurize_waveform
from src.data.features import normalise_standardize
from src.data.tidyvoicex import TidyVoiceXCorpus
from src.experiment import RunStatistics, STRATEGIES, build_sc_utterances, format_results, run_experiment
from tests.conftest import make_synthetic_waveform
from tests.test_tidyvoicex import _write_wav, make_tidyvoicex_root


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


def _make_voxceleb_stub_corpus(tmp_path, num_speakers=3, utterances_per_speaker=3, duration_s=2.0):
    """Minimal VoxCelebCorpus-shaped stub (speakers/utterance_paths/
    load_waveform/trial_pairs/trial_utterance_path) so run_experiment's
    VoxCeleb path can be exercised end-to-end without a real download.
    Backed by real (temporary) wav files on disk, since the lazy
    featurization path (src/experiment.py::_lazy_featurize_split/
    _lazy_trial_utterances) probes each file's length via soundfile.info
    before featurizing it."""
    root = tmp_path / "voxceleb_stub"

    train_paths_by_speaker = {}
    for speaker in range(num_speakers):
        speaker_id = f"id{speaker:05d}"
        paths = []
        for i in range(utterances_per_speaker):
            path = root / speaker_id / f"clip{i}" / "00001.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            waveform = make_synthetic_waveform(200 + 150 * speaker, duration_s, 16000, seed=speaker * 100 + i)
            sf.write(str(path), waveform, 16000)
            paths.append(path)
        train_paths_by_speaker[speaker_id] = paths

    # A separate pool of "held out" utterances for the trial list -- disjoint
    # from train_paths_by_speaker's speakers, matching how the real
    # VoxCelebCorpus excludes trial speakers from training.
    trial_paths = {}
    for speaker_idx, speaker_id in enumerate(("idEVAL0", "idEVAL1")):
        for clip in range(2):
            relative_path = f"{speaker_id}/clip{clip}/00001.wav"
            path = root / "trial" / speaker_id / f"clip{clip}" / "00001.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            waveform = make_synthetic_waveform(500 + 150 * speaker_idx, duration_s, 16000, seed=900 + speaker_idx * 10 + clip)
            sf.write(str(path), waveform, 16000)
            trial_paths[relative_path] = path

    trial_pairs = [
        (1, "idEVAL0/clip0/00001.wav", "idEVAL0/clip1/00001.wav"),
        (0, "idEVAL0/clip0/00001.wav", "idEVAL1/clip0/00001.wav"),
    ]

    class _VoxCelebStubCorpus:
        def __init__(self):
            self.trial_pairs = trial_pairs
            self.dev_trial_pairs = trial_pairs

        def speakers(self, split):
            assert split == "TRAIN"
            return sorted(train_paths_by_speaker.keys())

        def utterance_paths(self, split, speaker_id):
            assert split == "TRAIN"
            return train_paths_by_speaker[speaker_id]

        def trial_utterance_path(self, relative_path):
            return trial_paths[relative_path]

        @staticmethod
        def raw_sample_count(path):
            return sf.info(str(path)).frames

        @staticmethod
        def load_waveform(path):
            waveform, sample_rate = sf.read(str(path), dtype="float32")
            return waveform, sample_rate

        @staticmethod
        def load_samples(path, start, stop):
            return sf.read(str(path), start=start, stop=stop, dtype="float32")[0]

    return _VoxCelebStubCorpus()



def test_build_sc_utterances_splits_short_and_long_per_speaker():
    # Neururer et al. 2024 Section 2.2's SC protocol clusters, per speaker,
    # one utterance built from 2 concatenated sentences and one from the
    # remaining sentences (8 for standard TIMIT's 10-sentences-per-speaker
    # layout) -- not raw single sentences.
    waveforms = {"TEST": {}}
    for speaker in range(4):
        speaker_id = f"SPK{speaker}"
        waveforms["TEST"][speaker_id] = [
            make_synthetic_waveform(200 + 150 * speaker, 1.0, 16000, seed=speaker * 10 + i) for i in range(5)
        ]
    corpus = _StubCorpus(waveforms)
    transformation = ExperimentConfig().transformation

    utterances = build_sc_utterances(corpus, "TEST", transformation, num_speakers=2)

    labels_present = {label for _, label in utterances}
    assert labels_present == {0, 1}
    assert len(utterances) == 4  # 2 speakers x (short, long)

    lengths_by_speaker = {}
    for features, label in utterances:
        lengths_by_speaker.setdefault(label, []).append(features.shape[0])
    for lengths in lengths_by_speaker.values():
        assert len(lengths) == 2
        assert min(lengths) < max(lengths)  # short (2 sentences) vs. long (remaining 3)


def test_build_sc_utterances_short_utterance_is_last_two_sorted_sentences():
    # TIMIT sorts SA1, SA2 (identical text for every speaker) first -- the
    # short utterance must be the last 2 sorted sentences, not SA1+SA2.
    durations = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    waveforms = {"TEST": {"SPK0": [make_synthetic_waveform(200, d, 16000, seed=i) for i, d in enumerate(durations)]}}
    corpus = _StubCorpus(waveforms)
    transformation = ExperimentConfig().transformation

    long_utterance, short_utterance = [f for f, _ in build_sc_utterances(corpus, "TEST", transformation, num_speakers=1)]

    expected_short = featurize_waveform(np.concatenate(waveforms["TEST"]["SPK0"][-2:]), transformation)
    expected_long = featurize_waveform(np.concatenate(waveforms["TEST"]["SPK0"][:-2]), transformation)
    np.testing.assert_allclose(short_utterance, expected_short)
    np.testing.assert_allclose(long_utterance, expected_long)


def test_build_sc_utterances_uses_explicit_speaker_ids_case_insensitively():
    waveforms = {"TEST": {}}
    for speaker_id in ("fabc0", "fdef0", "mabc0"):
        waveforms["TEST"][speaker_id] = [make_synthetic_waveform(200, 1.0, 16000, seed=i) for i in range(4)]
    corpus = _StubCorpus(waveforms)
    transformation = ExperimentConfig().transformation

    utterances = build_sc_utterances(corpus, "TEST", transformation, num_speakers=2, speaker_ids=["MABC0", "FDEF0"])
    assert len(utterances) == 4
    assert [label for _, label in utterances] == [0, 0, 1, 1]

    with pytest.raises(ValueError, match="MXYZ0"):
        build_sc_utterances(corpus, "TEST", transformation, num_speakers=2, speaker_ids=["MABC0", "MXYZ0"])


def test_featurize_train_dev_split_holds_out_configured_utterances_per_speaker():
    # Checkpoint-selection dev EER must come from TRAIN speakers' own held-out
    # utterances (never gradient-trained on), not from TEST -- see
    # run_experiment's use of this split and the "dev/test leakage" fix it's
    # part of.
    waveforms = {"TRAIN": {}}
    for speaker in range(3):
        speaker_id = f"SPK{speaker}"
        waveforms["TRAIN"][speaker_id] = [
            make_synthetic_waveform(200 + 150 * speaker, 1.0, 16000, seed=speaker * 10 + i) for i in range(5)
        ]
    corpus = _StubCorpus(waveforms)
    transformation = ExperimentConfig().transformation

    train_utterances, dev_utterances, label_map = experiment_module._featurize_train_dev_split(
        corpus, transformation, dev_holdout_per_speaker=2
    )

    assert set(label_map.keys()) == {"SPK0", "SPK1", "SPK2"}
    assert len(dev_utterances) == 3 * 2  # 2 held-out utterances x 3 speakers
    assert len(train_utterances) == 3 * 3  # remaining 3 utterances x 3 speakers
    # Every speaker appears in both pools (held-out utterances, not held-out speakers).
    assert {label for _, label in dev_utterances} == set(label_map.values())
    assert {label for _, label in train_utterances} == set(label_map.values())


def test_playback_speed_transforms_only_the_training_pool():
    # TrainingConfig.playback_speed is training-data augmentation: held-out
    # dev utterances must stay at the original speed, in both the eager and
    # the lazy split, and lazy lengths must match the transformed features.
    waveforms = {"TRAIN": {"SPK0": [make_synthetic_waveform(200, 1.0, 16000, seed=i) for i in range(3)]}}
    corpus = _StubCorpus(waveforms)
    corpus.raw_sample_count = lambda path: len(corpus.load_waveform(path)[0])
    transformation = ExperimentConfig().transformation
    original = experiment_module._featurize_train_dev_split(corpus, transformation, dev_holdout_per_speaker=1)

    eager_train, eager_dev, _ = experiment_module._featurize_train_dev_split(
        corpus, transformation, dev_holdout_per_speaker=1, playback_speed=1.25
    )
    lazy_train, lazy_dev, _ = experiment_module._lazy_featurize_train_dev_split(
        corpus, transformation, dev_holdout_per_speaker=1, playback_speed=1.25
    )

    np.testing.assert_array_equal(eager_dev[0][0], original[1][0][0])
    np.testing.assert_array_equal(lazy_dev[0][0](), original[1][0][0])
    for (eager, _), (lazy, _), (unchanged, _) in zip(eager_train, lazy_train, original[0]):
        assert eager.shape[0] == lazy.length == lazy().shape[0]
        # Samples scale exactly by 1 / 1.25; frames only up to the frame_length offset.
        assert eager.shape[0] == pytest.approx(unchanged.shape[0] / 1.25, abs=2)


def test_run_experiment_dev_checkpoint_selection_excludes_test_split(monkeypatch):
    # Regression test for the dev/test leakage fix: before it, dev_utterances
    # passed to train() *was* sv_eval_utterances (the TEST split), so its size
    # equaled TEST's full utterance count. Now it must be TRAIN's own held-out
    # pool, sized dev_holdout_per_speaker x num_train_speakers instead.
    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.evaluation.dev_holdout_per_speaker = 2  # >=2 so the dev pool has genuine (same-speaker) pairs
    config.num_runs = 1

    corpus = _tiny_stub_corpus()  # 2 TRAIN/TEST speakers, 3 utterances each
    real_train = experiment_module.train
    captured_dev_utterance_counts = []

    def _spy_train(model, loss_module, dataset, cfg, **kwargs):
        captured_dev_utterance_counts.append(len(kwargs["dev_utterances"]))
        return real_train(model, loss_module, dataset, cfg, **kwargs)

    monkeypatch.setattr(experiment_module, "train", _spy_train)
    run_experiment(config, corpus=corpus, strategies=("OS",))

    assert captured_dev_utterance_counts == [4]  # 2 held-out utterances x 2 TRAIN speakers, not TEST's 6


def test_run_experiment_dev_speakers_select_on_test_speakers_and_train_on_all_of_train(monkeypatch):
    # evaluation.dev_speakers (the TIMIT configs' standard 50-speaker dev set):
    # checkpoint selection uses all TEST utterances of those speakers, matched
    # case-insensitively, and no TRAIN utterance is held out of training.
    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.evaluation.dev_speakers = ["spk1", "spk0"]
    config.num_runs = 1

    corpus = _tiny_stub_corpus()  # 2 TRAIN/TEST speakers, 3 utterances each
    real_train = experiment_module.train
    captured = []

    def _spy_train(model, loss_module, dataset, cfg, **kwargs):
        captured.append((len(dataset), kwargs["dev_utterances"]))
        return real_train(model, loss_module, dataset, cfg, **kwargs)

    monkeypatch.setattr(experiment_module, "train", _spy_train)
    run_experiment(config, corpus=corpus, strategies=("OS",))

    [(train_size, dev_utterances)] = captured
    assert train_size == 6  # all 2 x 3 TRAIN utterances
    expected = [featurize_waveform(corpus.load_waveform(path)[0], config.transformation)
                for speaker in ("SPK1", "SPK0") for path in corpus.utterance_paths("TEST", speaker)]
    assert [label for _, label in dev_utterances] == [0, 0, 0, 1, 1, 1]
    for (features, _), expected_features in zip(dev_utterances, expected):
        np.testing.assert_allclose(features, expected_features)

    config.evaluation.dev_speakers = ["SPK9"]
    with pytest.raises(ValueError, match="SPK9"):
        run_experiment(config, corpus=corpus, strategies=("OS",))


def test_run_experiment_end_to_end_on_synthetic_corpus():
    config = ExperimentConfig()
    config.training.num_epochs = 3
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1

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

    assert set(results.keys()) == {"SV", "SV_minDCF", "SC", "SV_paper_comparable"}
    for task in ("SV", "SC"):
        for train_strategy in ("OS", "SS", "SU"):
            for test_strategy in ("OS", "SS", "SU"):
                stats = results[task][(train_strategy, test_strategy)]
                assert isinstance(stats, RunStatistics)
                assert 0.0 <= stats.mean <= 1.0
                assert stats.std >= 0.0

    report = format_results(results)
    assert "SV (EER)" in report
    assert "SC (MR)" in report
    assert "SV (minDCF)" in report


def test_run_experiment_end_to_end_with_conformer_model():
    # Same synthetic-corpus pipeline as the CNN test above, but selecting the
    # Conformer backend via config.model.type -- exercises the model-choice
    # plumbing (registry lookup, config.conformer hyperparameters) through a
    # full train/extract/evaluate pass, not just the model's forward pass.
    config = ExperimentConfig()
    config.model.type = "Conformer"
    config.conformer.num_layers = 1
    config.conformer.encoder_dim = 16
    config.conformer.num_heads = 2
    config.conformer.ff_expansion_factor = 2
    config.conformer.conv_kernel_size = 3
    config.training.num_epochs = 2
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1

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

    assert set(results.keys()) == {"SV", "SV_minDCF", "SC", "SV_paper_comparable"}
    for task in ("SV", "SC"):
        for train_strategy in ("OS", "SS", "SU"):
            for test_strategy in ("OS", "SS", "SU"):
                stats = results[task][(train_strategy, test_strategy)]
                assert isinstance(stats, RunStatistics)
                assert 0.0 <= stats.mean <= 1.0
                assert stats.std >= 0.0


def test_format_results_reports_percentages_matching_paper_tables():
    # Neururer et al. 2024 Tables 1/2 report MR/EER as mean/SD on a 0-100 scale
    # (e.g. MR mean/SD of 37.50/4.18, EER up to 23.41), while the metric
    # functions themselves return [0, 1] fractions. format_results must
    # rescale both mean and SD for the report to be comparable.
    results = {
        "SV": {(s1, s2): RunStatistics(mean=0.0638, std=0.0012) for s1 in STRATEGIES for s2 in STRATEGIES},
        "SC": {(s1, s2): RunStatistics(mean=0.375, std=0.0418) for s1 in STRATEGIES for s2 in STRATEGIES},
    }

    report = format_results(results)

    assert "[%]" in report
    assert "6.38" in report
    assert "0.12" in report
    assert "37.50" in report
    assert "4.18" in report
    assert "0.0638" not in report
    assert "0.3750" not in report


def test_run_experiment_strategies_argument_restricts_which_strategies_run():
    # Lets a caller split STRATEGIES across separate processes/GPUs (see
    # main()'s --strategy flag) -- only the requested strategy should be
    # trained/evaluated, and format_results must still render cleanly with
    # the other two absent from results rather than KeyError.
    config = ExperimentConfig()
    config.training.num_epochs = 2
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1

    results = run_experiment(config, corpus=_tiny_stub_corpus(), strategies=("SS",))

    for task in ("SV", "SC"):
        trained = {train_strategy for train_strategy, _ in results[task]}
        assert trained == {"SS"}

    report = format_results(results)
    row_labels = {line.split()[0] for line in report.splitlines() if line[:2] in ("OS", "SS", "SU")}
    assert row_labels == {"SS"}


def test_run_experiment_trains_and_tests_on_configured_strategies():
    config = ExperimentConfig()
    config.training.num_epochs = 2
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1
    config.training.strategies = ["MIX"]
    config.evaluation.test_strategies = ["OS", "BS4"]

    results = run_experiment(config, corpus=_tiny_stub_corpus())

    for task in ("SV", "SV_minDCF", "SC"):
        assert set(results[task]) == {("MIX", "OS"), ("MIX", "BS4")}
    report = format_results(results)
    assert any(line.split()[:2] == ["train\\test", "OS"] and "BS4" in line for line in report.splitlines())


def test_run_experiment_stage1_setup_with_segment_normalization():
    # The stage-1 configs' combination: per-segment normalization, RANDBS
    # training, and the frame-order-invariant Conformer.
    config = ExperimentConfig()
    config.model.type = "Conformer"
    config.transformation.normalization = "segment"
    config.training.num_epochs = 2
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.evaluation.test_strategies = ["OS", "SS", "BS16"]
    config.num_runs = 1

    config.training.strategies = ["RANDBS"]
    results = run_experiment(config, corpus=_tiny_stub_corpus())
    assert set(results["SV"]) == {("RANDBS", "OS"), ("RANDBS", "SS"), ("RANDBS", "BS16")}

    config.training.strategies = ["OS"]
    config.conformer.use_relative_positional_encoding = False
    config.conformer.conv_kernel_size = 1
    config.conformer.subsampling_time_kernel = 1
    config.conformer.subsampling_time_stride = 1
    results = run_experiment(config, corpus=_tiny_stub_corpus())
    # Invariant to frame order: SS and BS16 test windows embed exactly like OS.
    assert results["SV"][("OS", "SS")].mean == pytest.approx(results["SV"][("OS", "OS")].mean, abs=1e-6)
    assert results["SV"][("OS", "BS16")].mean == pytest.approx(results["SV"][("OS", "OS")].mean, abs=1e-6)


def test_run_experiment_logs_eer_by_block_span_per_run_and_aggregated(monkeypatch):
    charts = []
    monkeypatch.setattr(
        experiment_module.tracking, "log_line_series",
        lambda run, key, xs, series, title, xname: charts.append((key, xs, series)),
    )
    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 2
    config.training.strategies = ["OS"]
    config.evaluation.test_strategies = ["BS8", "OS", "SU", "SS"]

    results = run_experiment(config, corpus=_tiny_stub_corpus())

    segment_length = config.data.segment_length(config.transformation)
    expected_ms = [64, 134, round(((segment_length - 1) * 0.01 + 0.064) * 1000)]  # SS, BS8, OS; SU isn't a block shuffle
    assert [key for key, _, _ in charts] == [
        "final/SV_EER_by_block/run0", "final/SV_EER_by_block/run1", "final/SV_EER_by_block",
    ]
    assert all(xs == expected_ms for _, xs, _ in charts)
    final = charts[-1][2]
    assert set(final) == {"run0", "run1", "mean"}
    order = ["SS", "BS8", "OS"]
    assert final["mean"] == pytest.approx([results["SV"][("OS", s)].mean * 100 for s in order])
    assert charts[0][2]["run0"] == final["run0"]


def _continuation_config(num_epochs, keep_checkpoints=True):
    config = ExperimentConfig()
    config.training.num_epochs = num_epochs
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1
    config.training.strategies = ["OS"]
    config.evaluation.test_strategies = ["OS"]
    config.training.keep_checkpoints = keep_checkpoints
    return config


@pytest.mark.parametrize("keep_checkpoints", [True, False])
def test_run_experiment_keeps_checkpoints_only_when_configured(keep_checkpoints):
    config = _continuation_config(2, keep_checkpoints)
    run_experiment(config, corpus=_tiny_stub_corpus())
    checkpoint = experiment_module._checkpoint_path(config, "OS", 0)
    assert checkpoint.exists() == keep_checkpoints
    assert experiment_module.best_checkpoint_path(checkpoint).exists() == keep_checkpoints


def test_run_experiment_continues_completed_runs_when_num_epochs_is_raised(monkeypatch):
    corpus = _tiny_stub_corpus()
    run_experiment(_continuation_config(2), corpus=corpus)

    logged_epochs = []
    real_log = experiment_module.tracking.log
    def _spy_log(run, metrics, step=None):
        if "run0/train/loss" in metrics:  # one per trained epoch
            logged_epochs.append(metrics["run0/epoch"])
        return real_log(run, metrics, step)
    monkeypatch.setattr(experiment_module.tracking, "log", _spy_log)

    config = _continuation_config(3)
    results = run_experiment(config, corpus=corpus)

    assert logged_epochs == [2]  # only the one new epoch is trained
    checkpoint = torch.load(experiment_module._checkpoint_path(config, "OS", 0), weights_only=False)
    assert checkpoint["epoch"] == 2
    manifest = experiment_module._load_manifest(experiment_module._manifest_path(config, "OS"))
    assert manifest["completed"][0]["num_epochs"] == 3
    assert results["SV"][("OS", "OS")].mean == pytest.approx(manifest["completed"][0]["sv"]["OS"])

    # Already at the configured epochs: nothing is trained again.
    logged_epochs.clear()
    run_experiment(config, corpus=corpus)
    assert logged_epochs == []


def test_run_experiment_refuses_to_continue_a_run_without_its_checkpoint():
    corpus = _tiny_stub_corpus()
    run_experiment(_continuation_config(2, keep_checkpoints=False), corpus=corpus)
    with pytest.raises(ValueError, match="can't be continued"):
        run_experiment(_continuation_config(3), corpus=corpus)


def test_run_experiment_rejects_unknown_strategies():
    config = ExperimentConfig()
    config.evaluation.test_strategies = ["OS", "XX"]
    with pytest.raises(ValueError, match="XX"):
        run_experiment(config, corpus=_tiny_stub_corpus())


def test_run_experiment_refuses_manifests_missing_a_test_strategy(tmp_path):
    config = ExperimentConfig()
    config.training.checkpoint_dir = str(tmp_path)
    config.training.strategies = ["OS"]
    config.evaluation.test_strategies = ["OS", "BS4"]
    config.num_runs = 1
    manifest = {"wandb_run_id": None, "completed": {"0": {"sv": {"OS": 0.1, "SS": 0.2, "SU": 0.3}}}}
    experiment_module._save_manifest(experiment_module._manifest_path(config, "OS"), manifest)
    with pytest.raises(ValueError, match="BS4"):
        run_experiment(config, corpus=_tiny_stub_corpus())


def _tiny_stub_corpus():
    waveforms = {"TRAIN": {}, "TEST": {}}
    seed = 0
    for split in waveforms:
        for speaker in range(2):
            speaker_id = f"SPK{speaker}"
            waveforms[split][speaker_id] = [
                make_synthetic_waveform(200 + 150 * speaker, 2.0, 16000, seed=seed)
                for seed in range(seed, seed + 3)
            ]
            seed += 3
    return _StubCorpus(waveforms)


def test_run_experiment_resolves_auto_device_onto_the_config(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1
    assert config.device == "auto"

    run_experiment(config, corpus=_tiny_stub_corpus())

    assert config.device == "cpu"


def test_run_experiment_honors_an_explicit_device_override():
    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1
    config.device = "cpu"

    run_experiment(config, corpus=_tiny_stub_corpus())

    assert config.device == "cpu"


def _is_aggregate_summary(summary):
    return any(key.endswith("_mean") for key in summary)


def test_run_experiment_logs_wandb_summary_on_a_0_100_scale(monkeypatch):
    # tracking.log_summary receives both each individual run's EER/MR
    # (namespaced by run_idx, since config.num_runs repeats now share a
    # single wandb run) and, once a strategy's runs all finish, its aggregate
    # mean/SD -- all on the same 0-100 scale as format_results/the paper's
    # tables, even though `results` itself keeps the raw [0, 1]
    # fractions/RunStatistics. num_runs=1 here so each per-run summary's
    # single raw value is directly comparable to the (trivial) aggregate mean.
    captured_summaries = []
    monkeypatch.setattr(
        experiment_module.tracking, "log_summary", lambda run, metrics: captured_summaries.append(metrics)
    )

    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1

    results = run_experiment(config, corpus=_tiny_stub_corpus())

    per_run_summaries = [s for s in captured_summaries if not _is_aggregate_summary(s)]
    aggregate_summaries = [s for s in captured_summaries if _is_aggregate_summary(s)]

    assert len(per_run_summaries) == len(STRATEGIES) * config.num_runs
    assert len(aggregate_summaries) == len(STRATEGIES)

    for summary, train_strategy in zip(per_run_summaries, STRATEGIES):
        for test_strategy in STRATEGIES:
            eer_fraction = results["SV"][(train_strategy, test_strategy)].mean
            mr_fraction = results["SC"][(train_strategy, test_strategy)].mean
            assert summary[f"run0/final/SV_EER_test-{test_strategy}"] == pytest.approx(eer_fraction * 100)
            assert summary[f"run0/final/SC_MR_test-{test_strategy}"] == pytest.approx(mr_fraction * 100)

    for summary, train_strategy in zip(aggregate_summaries, STRATEGIES):
        for test_strategy in STRATEGIES:
            sv_stats = results["SV"][(train_strategy, test_strategy)]
            sc_stats = results["SC"][(train_strategy, test_strategy)]
            assert summary[f"final/SV_EER_test-{test_strategy}_mean"] == pytest.approx(sv_stats.mean * 100)
            assert summary[f"final/SV_EER_test-{test_strategy}_std"] == pytest.approx(sv_stats.std * 100)
            assert summary[f"final/SC_MR_test-{test_strategy}_mean"] == pytest.approx(sc_stats.mean * 100)
            assert summary[f"final/SC_MR_test-{test_strategy}_std"] == pytest.approx(sc_stats.std * 100)


def test_run_experiment_aggregates_mean_and_std_over_num_runs(monkeypatch):
    # Neururer et al. 2024 (Section 2.3) train/test each strategy repeatedly
    # and report mean/SD of EER/MR over those runs. equal_error_rate and
    # best_misclassification_rate are stubbed with deterministic, strictly
    # increasing sequences so the exact per-cell mean/SD can be predicted from
    # the call order and checked against run_experiment's actual aggregation
    # (catches both wrong grouping across cells and a wrong run count), and
    # against what gets logged to wandb's aggregate summary.
    eer_counter = itertools.count()
    monkeypatch.setattr(experiment_module, "equal_error_rate", lambda *a, **k: next(eer_counter) / 100.0)
    mr_counter = itertools.count()
    monkeypatch.setattr(experiment_module, "best_misclassification_rate", lambda *a, **k: next(mr_counter) / 100.0)

    captured_summaries = []
    monkeypatch.setattr(
        experiment_module.tracking, "log_summary", lambda run, metrics: captured_summaries.append(metrics)
    )

    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 3

    results = run_experiment(config, corpus=_tiny_stub_corpus())

    aggregate_summaries = [s for s in captured_summaries if _is_aggregate_summary(s)]
    assert len(aggregate_summaries) == len(STRATEGIES)

    # Both stubs are called once per (train_strategy, run_idx, test_strategy)
    # iteration, in that nesting order, so the same running index predicts
    # both the EER and MR sequences.
    call = itertools.count()
    for train_strategy, aggregate_summary in zip(STRATEGIES, aggregate_summaries):
        per_test_strategy_values = {test_strategy: [] for test_strategy in STRATEGIES}
        for _run_idx in range(config.num_runs):
            for test_strategy in STRATEGIES:
                per_test_strategy_values[test_strategy].append(next(call) / 100.0)

        for test_strategy in STRATEGIES:
            values = per_test_strategy_values[test_strategy]
            expected = RunStatistics(mean=statistics.fmean(values), std=statistics.pstdev(values))
            sv_stats = results["SV"][(train_strategy, test_strategy)]
            sc_stats = results["SC"][(train_strategy, test_strategy)]
            assert sv_stats.mean == pytest.approx(expected.mean)
            assert sv_stats.std == pytest.approx(expected.std)
            assert sc_stats.mean == pytest.approx(expected.mean)
            assert sc_stats.std == pytest.approx(expected.std)

            assert aggregate_summary[f"final/SV_EER_test-{test_strategy}_mean"] == pytest.approx(expected.mean * 100)
            assert aggregate_summary[f"final/SV_EER_test-{test_strategy}_std"] == pytest.approx(expected.std * 100)
            assert aggregate_summary[f"final/SC_MR_test-{test_strategy}_mean"] == pytest.approx(expected.mean * 100)
            assert aggregate_summary[f"final/SC_MR_test-{test_strategy}_std"] == pytest.approx(expected.std * 100)


def test_run_experiment_resumes_across_a_crash_without_retraining_completed_runs(monkeypatch, tmp_path):
    # Stubbed with deterministic, strictly increasing sequences (as in
    # test_run_experiment_aggregates_mean_and_std_over_num_runs above) so the
    # exact call order -- and therefore whether any run got redone -- is
    # verifiable regardless of real model training's inherent randomness.
    eer_counter = itertools.count()
    monkeypatch.setattr(experiment_module, "equal_error_rate", lambda *a, **k: next(eer_counter) / 100.0)
    mr_counter = itertools.count()
    monkeypatch.setattr(experiment_module, "best_misclassification_rate", lambda *a, **k: next(mr_counter) / 100.0)

    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 3
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")

    corpus = _tiny_stub_corpus()
    real_train = experiment_module.train

    # Simulate a crash partway through the very first strategy's sweep: let
    # run_idx 0 finish normally, then blow up before run_idx 1 trains at all
    # (matching a real crash, which leaves no checkpoint for the run that was
    # never even started).
    call_count = itertools.count()

    def _crash_on_second_call(*args, **kwargs):
        if next(call_count) == 1:
            raise RuntimeError("simulated crash")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "train", _crash_on_second_call)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_experiment(config, corpus=corpus)

    # "Resume": a fresh invocation with the same config (same checkpoint_dir)
    # must not retrain OS's run_idx 0 -- but SS/SU haven't started at all yet
    # (the crash happened during OS's sweep), so they still train all 3.
    train_calls = []

    def _spy_train(model, loss_module, dataset, cfg, **kwargs):
        train_calls.append((kwargs.get("draw_strategy"), kwargs.get("run_idx")))
        return real_train(model, loss_module, dataset, cfg, **kwargs)

    monkeypatch.setattr(experiment_module, "train", _spy_train)
    results = run_experiment(config, corpus=corpus)

    assert ("OS", 0) not in train_calls
    assert sorted(strategy_run for strategy_run in train_calls if strategy_run[0] == "OS") == [("OS", 1), ("OS", 2)]
    for strategy in ("SS", "SU"):
        assert sorted(run_idx for s, run_idx in train_calls if s == strategy) == [0, 1, 2]

    manifest_path = Path(config.training.checkpoint_dir) / "CNN-timit-OS" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert sorted(manifest["completed"].keys()) == ["0", "1", "2"]

    # Completed repeats' checkpoints are cleaned up, not left to accumulate.
    remaining_checkpoints = list(Path(config.training.checkpoint_dir).rglob("*.pt"))
    assert remaining_checkpoints == []

    for task in ("SV", "SC"):
        for test_strategy in STRATEGIES:
            stats = results[task][("OS", test_strategy)]
            assert 0.0 <= stats.mean <= 1.0


def test_run_experiment_reuses_the_same_wandb_run_id_across_invocations(monkeypatch, tmp_path):
    config = ExperimentConfig()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.evaluation.sc_num_speakers = 2
    config.num_runs = 1
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.wandb.enabled = True
    config.wandb.mode = "disabled"  # exercises the real wandb.init/finish path without network

    corpus = _tiny_stub_corpus()
    run_experiment(config, corpus=corpus)

    manifest_path = Path(config.training.checkpoint_dir) / "CNN-timit-OS" / "manifest.json"
    first_run_id = json.loads(manifest_path.read_text())["wandb_run_id"]
    assert first_run_id is not None

    captured_run_ids = []
    real_start_run = experiment_module.tracking.start_run

    def _spy_start_run(*args, **kwargs):
        captured_run_ids.append(kwargs.get("run_id"))
        return real_start_run(*args, **kwargs)

    monkeypatch.setattr(experiment_module.tracking, "start_run", _spy_start_run)

    # All 3 strategies' runs are already complete, so this invocation should
    # skip straight to reattaching + re-logging the aggregate for each.
    run_experiment(config, corpus=corpus)

    assert captured_run_ids[0] == first_run_id


def test_run_experiment_voxceleb_end_to_end_skips_sc_and_uses_trial_list_eval(tmp_path):
    # Neururer et al. 2024 omits SC for VoxCeleb, and VoxCeleb's SV
    # evaluation is trial-pairs-based rather than TIMIT's exhaustive
    # all-vs-all pairing -- both should show up structurally in the result
    # of a run through run_experiment's VoxCeleb dispatch (src/experiment.py).
    config = ExperimentConfig()
    config.data.dataset = "VoxCeleb"
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.num_runs = 1

    corpus = _make_voxceleb_stub_corpus(tmp_path)
    results = run_experiment(config, corpus=corpus)

    assert set(results.keys()) == {"SV", "SV_minDCF", "SV_paper_comparable"}
    for train_strategy in STRATEGIES:
        for test_strategy in STRATEGIES:
            stats = results["SV"][(train_strategy, test_strategy)]
            assert isinstance(stats, RunStatistics)
            assert 0.0 <= stats.mean <= 1.0

    report = format_results(results)
    assert "SV (EER)" in report
    assert "SC (MR)" not in report


def test_run_experiment_voxceleb_selects_on_dev_trials_and_reports_on_eval_trials(monkeypatch, tmp_path):
    # Paper protocol (VoxCeleb2 training): checkpoint selection on VoxCeleb1-O
    # (corpus.dev_trial_pairs), reported SV on VoxCeleb1-H (corpus.trial_pairs).
    # Also runs training through DataLoader workers over windowed entries.
    config = ExperimentConfig()
    config.data.dataset = "VoxCeleb"
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.training.num_workers = 2
    config.loss.type = "SOFTMAX"
    config.num_runs = 1

    corpus = _make_voxceleb_stub_corpus(tmp_path)
    corpus.dev_trial_pairs = [
        (1, "idEVAL1/clip0/00001.wav", "idEVAL1/clip1/00001.wav"),
        (0, "idEVAL1/clip1/00001.wav", "idEVAL0/clip1/00001.wav"),
    ]
    evaluated_ids = {"dev": [], "eval": []}
    real_eer = experiment_module.trial_list_equal_error_rate

    def _spy_eer(embeddings, utterance_ids, trial_pairs):
        evaluated_ids["dev" if trial_pairs is corpus.dev_trial_pairs else "eval"].append(set(utterance_ids))
        return real_eer(embeddings, utterance_ids, trial_pairs)

    captured_train_entries = []
    real_train = experiment_module.train

    def _spy_train(model, loss_module, dataset, cfg, **kwargs):
        captured_train_entries.extend(entry for entry, _ in dataset.utterances)
        return real_train(model, loss_module, dataset, cfg, **kwargs)

    monkeypatch.setattr(experiment_module, "trial_list_equal_error_rate", _spy_eer)
    monkeypatch.setattr(experiment_module, "train", _spy_train)
    run_experiment(config, corpus=corpus, strategies=("OS",))

    assert all(entry.windowed for entry in captured_train_entries)
    assert evaluated_ids["dev"] == [{"idEVAL1/clip0/00001.wav", "idEVAL1/clip1/00001.wav", "idEVAL0/clip1/00001.wav"}]
    assert len(evaluated_ids["eval"]) == len(STRATEGIES)
    assert all(ids == {"idEVAL0/clip0/00001.wav", "idEVAL0/clip1/00001.wav", "idEVAL1/clip0/00001.wav"}
               for ids in evaluated_ids["eval"])


def test_run_experiment_voxceleb_wandb_tags_and_name_identify_the_dataset(monkeypatch, tmp_path):
    captured = []
    real_start_run = experiment_module.tracking.start_run

    def _spy_start_run(wandb_config, name, tags, run_config, run_id=None):
        captured.append((name, tags))
        return real_start_run(wandb_config, name, tags, run_config, run_id=run_id)

    monkeypatch.setattr(experiment_module.tracking, "start_run", _spy_start_run)

    config = ExperimentConfig()
    config.data.dataset = "VoxCeleb"
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.num_runs = 1
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")

    corpus = _make_voxceleb_stub_corpus(tmp_path)
    run_experiment(config, corpus=corpus)

    names = [name for name, _ in captured]
    assert names == ["CNN-voxceleb-OS", "CNN-voxceleb-SS", "CNN-voxceleb-SU"]
    for _, tags in captured:
        assert "voxceleb" in tags



def _make_tidyvoicex_corpus(tmp_path, short_dev_utterance=False):
    """A real (tiny, on-disk) TidyVoiceX_ASV-shaped tree + trial list, so
    run_experiment's TidyVoiceX path is exercised against the actual
    src/data/tidyvoicex.py loader rather than a stub."""
    train = {f"id00000{s}": [f"en/en_{s}{i}.wav" for i in range(2)] + [f"de/de_{s}{i}.wav" for i in range(2)] for s in range(3)}
    dev = {f"id01400{s}": ["en/en_a.wav", "fr/fr_b.wav"] for s in range(3)}
    trial_lines = []
    for s in range(3):
        trial_lines.append(f"1 id01400{s}/en/en_a.wav id01400{s}/fr/fr_b.wav")
        trial_lines.append(f"0 id01400{s}/en/en_a.wav id01400{(s + 1) % 3}/en/en_a.wav")
    root = make_tidyvoicex_root(tmp_path / "tidyx", train=train, dev=dev, trial_lines=trial_lines)
    if short_dev_utterance:
        _write_wav(root / "TidyVoiceX_ASV" / "TidyVoiceX_Dev" / "id014000" / "it" / "it_c.wav", 0.5)
        with open(root / "TidyVocieX_Dev_trialPairs.txt", "a") as f:
            f.write("1 id014000/en/en_a.wav id014000/it/it_c.wav\n")

    config = ExperimentConfig()
    config.tidyvoicex.root = str(root)
    return TidyVoiceXCorpus(config)


def test_lazy_indexed_trial_utterances_drops_trials_with_too_short_utterances(tmp_path):
    config = ExperimentConfig()
    corpus = _make_tidyvoicex_corpus(tmp_path, short_dev_utterance=True)
    segment_length = config.data.segment_length(config.transformation)

    utterances, trials = experiment_module._lazy_indexed_trial_utterances(corpus, config.transformation, segment_length)

    assert len(corpus.trials.labels) == 7
    assert len(trials.labels) == 6
    assert "id014000/it/it_c.wav" not in trials.utterance_ids
    assert [utterance_id for _, utterance_id in utterances] == trials.utterance_ids
    # Re-indexed trials still point at the same utterance pairs as the originals.
    original = {(corpus.trials.utterance_ids[a], corpus.trials.utterance_ids[b]) for a, b in zip(corpus.trials.idx1, corpus.trials.idx2)}
    kept = {(trials.utterance_ids[a], trials.utterance_ids[b]) for a, b in zip(trials.idx1, trials.idx2)}
    assert kept <= original


def test_holdout_dev_trials_pair_within_and_across_neighbouring_speakers():
    class _Features:
        def __init__(self, length):
            self.length = length

    segment_length = 10
    dev_pool = [(_Features(20), 0), (_Features(20), 0), (_Features(20), 1), (_Features(20), 1), (_Features(5), 1)]

    utterances, trials = experiment_module._holdout_dev_trials(dev_pool, segment_length)

    assert [label for _, label in utterances] == [0, 1, 2, 3]  # too-short entry dropped, relabeled by row
    pairs = sorted(zip(trials.labels.tolist(), trials.idx1.tolist(), trials.idx2.tolist()))
    assert pairs == [(0, 0, 2), (0, 1, 3), (0, 2, 0), (0, 3, 1), (1, 0, 1), (1, 2, 3)]


def test_run_experiment_tidyvoicex_end_to_end_skips_sc_and_uses_trial_list_eval(monkeypatch, tmp_path):
    captured = []
    real_start_run = experiment_module.tracking.start_run

    def _spy_start_run(wandb_config, name, tags, run_config, run_id=None):
        captured.append((name, tags))
        return real_start_run(wandb_config, name, tags, run_config, run_id=run_id)

    monkeypatch.setattr(experiment_module.tracking, "start_run", _spy_start_run)

    config = ExperimentConfig()
    config.data.dataset = "TidyVoiceX"
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.num_runs = 1
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")

    corpus = _make_tidyvoicex_corpus(tmp_path)
    results = run_experiment(config, corpus=corpus)

    assert set(results.keys()) == {"SV", "SV_minDCF", "SV_paper_comparable"}
    for train_strategy in STRATEGIES:
        for test_strategy in STRATEGIES:
            stats = results["SV"][(train_strategy, test_strategy)]
            assert isinstance(stats, RunStatistics)
            assert 0.0 <= stats.mean <= 1.0

    report = format_results(results)
    assert "SV (EER)" in report
    assert "SC (MR)" not in report

    names = [name for name, _ in captured]
    assert names == ["CNN-tidyvoicex-OS", "CNN-tidyvoicex-SS", "CNN-tidyvoicex-SU"]
    for _, tags in captured:
        assert "tidyvoicex" in tags


@pytest.mark.parametrize("playback_speed, expected", [(1.0, 94), (1.2, 75), (0.8, 113)])
def test_train_segment_length_follows_playback_speed(playback_speed, expected):
    config = ExperimentConfig()
    assert config.data.segment_length(config.transformation) == 94
    assert config.data.train_segment_length(config.transformation, playback_speed) == expected


def test_vtlp_augment_draws_fresh_alpha_per_segment_reproducibly():
    transformation = ExperimentConfig().transformation
    features = normalise_standardize(np.random.default_rng(0).random((300, transformation.n_mels)) * 30)
    augment_fn = functools.partial(
        experiment_module._vtlp_augment, alpha_range=[0.9, 1.1], transformation=transformation
    )

    def draws(augment_fn):
        dataset = SegmentDataset([(features, 0)], 94, "OS", seed=0, augment_fn=augment_fn)
        return [dataset[0][0] for _ in range(2)]

    plain, warped, warped_again = draws(None), draws(augment_fn), draws(augment_fn)
    for p, w, w2 in zip(plain, warped, warped_again):
        assert w.shape == p.shape
        assert not torch.equal(w, p)
        assert torch.equal(w, w2)

    alphas = []

    def recording_augment(segment, rng):
        alphas.append(rng.uniform(0.9, 1.1))
        return segment

    draws(recording_augment)
    assert len(set(alphas)) == 2  # a fresh alpha for every drawn segment


def test_sst_bracket_computes_both_estimates_raw_and_normalized_by_os_headroom():
    eers = {"OS": 0.10, "BS16": 0.16, "GO16": 0.20, "GS16": 0.23}
    measures = experiment_module.sst_bracket(eers, 16)
    assert measures["L"] == pytest.approx(0.03)
    assert measures["U"] == pytest.approx(0.06)
    assert measures["L_norm"] == pytest.approx(0.03 / 0.40)
    assert measures["U_norm"] == pytest.approx(0.06 / 0.40)
    assert experiment_module.sst_bracket({"OS": 0.1, "BS16": 0.2}, 16).keys() == {"U", "U_norm"}
    assert experiment_module.sst_bracket({"GO16": 0.2, "GS16": 0.3}, 16).keys() == {"L"}
    assert experiment_module.sst_bracket(eers, 8) == {}


def test_run_experiment_reports_sst_measures_per_block_size():
    config = _continuation_config(1, keep_checkpoints=False)
    config.evaluation.test_strategies = ["OS", "BS16", "GO16", "GS16", "GO32", "GS32"]

    results = run_experiment(config, corpus=_tiny_stub_corpus())

    sv = {test: results["SV"][("OS", test)].mean for test in config.evaluation.test_strategies}
    assert set(results["SST_L_norm"]) == {("OS", "b16"), ("OS", "b32")}
    assert set(results["SST_U"]) == {("OS", "b16")}
    assert results["SST_L"][("OS", "b32")].mean == pytest.approx(sv["GS32"] - sv["GO32"])
    assert results["SST_L_norm"][("OS", "b16")].mean == pytest.approx((sv["GS16"] - sv["GO16"]) / (0.5 - sv["OS"]))
    report = format_results(results)
    assert "normalized L(b)" in report and "train\\block" in report


def test_run_experiment_evaluates_completed_runs_under_new_test_strategies_from_their_best_checkpoint(monkeypatch):
    corpus = _tiny_stub_corpus()
    first = run_experiment(_continuation_config(2), corpus=corpus)

    trained = []
    monkeypatch.setattr(experiment_module, "train", lambda *args, **kwargs: trained.append(1))
    config = _continuation_config(2)
    config.evaluation.test_strategies = ["OS", "GO16", "GS16"]
    results = run_experiment(config, corpus=corpus)

    assert trained == []
    assert results["SV"][("OS", "OS")].mean == first["SV"][("OS", "OS")].mean
    manifest = experiment_module._load_manifest(experiment_module._manifest_path(config, "OS"))
    assert set(manifest["completed"][0]["sv"]) == {"OS", "GO16", "GS16"}
    assert ("OS", "b16") in results["SST_L"]


def test_sst_bootstrap_intervals_are_paired_and_contain_the_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.repeat([1, 0], 200)
    os_scores = rng.normal(labels * 1.5, 1.0)
    scores = {"OS": os_scores, "GO16": os_scores - 0.3 * labels, "GS16": os_scores - 0.3 * labels, "BS16": os_scores - 0.6 * labels}
    intervals = experiment_module.sst_bootstrap_intervals(labels, scores, 100, np.random.default_rng(1))

    assert set(intervals) == {"L_b16", "U_b16", "L_norm_b16", "U_norm_b16"}
    # GO and GS score every trial alike: paired resampling leaves no spread.
    assert intervals["L_b16"] == (0.0, 0.0)
    eers = {strategy: experiment_module.eer_from_scores(labels, values) for strategy, values in scores.items()}
    low, high = intervals["U_b16"]
    assert low <= experiment_module.sst_bracket(eers, 16)["U"] <= high
    assert low > 0


def _sst_quality_config():
    config = _continuation_config(1)
    config.evaluation.test_strategies = ["OS", "BS16", "GO16", "GS16"]
    config.evaluation.sst_bootstrap_resamples = 20
    config.evaluation.sst_placebo = True
    return config


def test_run_experiment_reports_bootstrap_intervals_and_the_gs_placebo():
    config = _sst_quality_config()
    results = run_experiment(config, corpus=_tiny_stub_corpus())

    record = experiment_module._load_manifest(experiment_module._manifest_path(config, "OS"))["completed"][0]
    assert set(record["sst_ci"]) == {"L_b16", "U_b16", "L_norm_b16", "U_norm_b16"}
    assert all(low <= high for low, high in record["sst_ci"].values())
    assert set(record["sst_placebo"]) == {"b16"}
    assert results["SST_placebo"][("OS", "b16")].mean == pytest.approx(record["sst_placebo"]["b16"])
    assert ("OS", "b16") in results["SST_L_ci"]
    report = format_results(results)
    assert "SST placebo" in report and "±" in report


def test_run_experiment_adds_the_quality_checks_to_completed_runs_without_retraining(monkeypatch):
    corpus = _tiny_stub_corpus()
    config = _sst_quality_config()
    config.evaluation.sst_bootstrap_resamples = 0
    config.evaluation.sst_placebo = False
    first = run_experiment(config, corpus=corpus)

    monkeypatch.setattr(experiment_module, "train", lambda *args, **kwargs: pytest.fail("retrained"))
    config = _sst_quality_config()
    results = run_experiment(config, corpus=corpus)

    record = experiment_module._load_manifest(experiment_module._manifest_path(config, "OS"))["completed"][0]
    assert "sst_ci" in record and "sst_placebo" in record
    for strategy in config.evaluation.test_strategies:
        assert results["SV"][("OS", strategy)].mean == pytest.approx(first["SV"][("OS", strategy)].mean)
