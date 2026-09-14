import itertools
import json
import statistics
from pathlib import Path

import pytest

from src import experiment as experiment_module
from src.config import ExperimentConfig
from src.experiment import RunStatistics, STRATEGIES, build_sc_utterances, format_results, run_experiment
from tests.conftest import make_synthetic_waveform


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

    assert set(results.keys()) == {"SV", "SC"}
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

    assert set(results.keys()) == {"SV", "SC"}
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
