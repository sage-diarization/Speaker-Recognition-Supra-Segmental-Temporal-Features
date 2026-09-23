import itertools
import json
import statistics
from pathlib import Path

import pytest
import soundfile as sf

from src import experiment as experiment_module
from src.config import ExperimentConfig
from src.data.aishell4 import Aishell4Corpus
from src.data.tidyvoicex import TidyVoiceXCorpus
from src.experiment import RunStatistics, STRATEGIES, build_sc_utterances, format_results, run_experiment
from tests.conftest import make_synthetic_waveform
from tests.test_aishell4 import _write_session
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

    return _VoxCelebStubCorpus()


def _make_aishell4_corpus(tmp_path, num_speakers=3, utterances_per_speaker=4, duration_s=2.0):
    """Builds a real (tiny, on-disk) AISHELL-4-shaped directory tree --
    train_S/{wav,TextGrid} and test/{wav,TextGrid}, one session per speaker
    -- and constructs a real Aishell4Corpus from it, so run_experiment's
    AISHELL4 path is exercised against the actual TextGrid-segment-slicing
    code (src/data/aishell4.py), not a hand-rolled stub."""
    root = tmp_path / "aishell4_stub"
    gap_s = 0.5
    for split_dir_name in ("train_S", "test"):
        for speaker in range(num_speakers):
            session_id = f"{split_dir_name}_sess{speaker}"
            intervals, t = [], 0.0
            for i in range(utterances_per_speaker):
                intervals.append((t, t + duration_s, f"utt{i}"))
                t += duration_s + gap_s
            _write_session(root / split_dir_name, session_id, {"SPK0": intervals}, duration_s=t)

    config = ExperimentConfig()
    config.aishell4.root = str(root)
    return Aishell4Corpus(config)


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

    assert set(results.keys()) == {"SV", "SC", "SV_paper_comparable"}
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

    assert set(results.keys()) == {"SV", "SC", "SV_paper_comparable"}
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

    assert set(results.keys()) == {"SV", "SV_paper_comparable"}
    for train_strategy in STRATEGIES:
        for test_strategy in STRATEGIES:
            stats = results["SV"][(train_strategy, test_strategy)]
            assert isinstance(stats, RunStatistics)
            assert 0.0 <= stats.mean <= 1.0

    report = format_results(results)
    assert "SV (EER)" in report
    assert "SC (MR)" not in report


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


def test_run_experiment_aishell4_end_to_end_skips_sc_and_uses_exhaustive_eval(tmp_path):
    # AISHELL-4 has a real per-(session, TextGrid-tier) speaker TRAIN/TEST
    # split (like TIMIT), not VoxCeleb's trial-pairs protocol -- SV should be
    # the same exhaustive all-vs-all pairing TIMIT's SV task uses. SC is
    # omitted (see run_experiment's AISHELL4 branch), same as VoxCeleb.
    config = ExperimentConfig()
    config.data.dataset = "AISHELL4"
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.num_runs = 1

    corpus = _make_aishell4_corpus(tmp_path)
    results = run_experiment(config, corpus=corpus)

    assert set(results.keys()) == {"SV", "SV_paper_comparable"}
    for train_strategy in STRATEGIES:
        for test_strategy in STRATEGIES:
            stats = results["SV"][(train_strategy, test_strategy)]
            assert isinstance(stats, RunStatistics)
            assert 0.0 <= stats.mean <= 1.0

    report = format_results(results)
    assert "SV (EER)" in report
    assert "SC (MR)" not in report


def test_run_experiment_aishell4_wandb_tags_and_name_identify_the_dataset(monkeypatch, tmp_path):
    captured = []
    real_start_run = experiment_module.tracking.start_run

    def _spy_start_run(wandb_config, name, tags, run_config, run_id=None):
        captured.append((name, tags))
        return real_start_run(wandb_config, name, tags, run_config, run_id=run_id)

    monkeypatch.setattr(experiment_module.tracking, "start_run", _spy_start_run)

    config = ExperimentConfig()
    config.data.dataset = "AISHELL4"
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.loss.type = "SOFTMAX"
    config.num_runs = 1
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")

    corpus = _make_aishell4_corpus(tmp_path)
    run_experiment(config, corpus=corpus)

    names = [name for name, _ in captured]
    assert names == ["CNN-aishell4-OS", "CNN-aishell4-SS", "CNN-aishell4-SU"]
    for _, tags in captured:
        assert "aishell4" in tags


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

    assert set(results.keys()) == {"SV", "SV_paper_comparable"}
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
