from types import SimpleNamespace

import numpy as np
import torch

from src.config import ExperimentConfig
from src.data.dataset import SegmentDataset
from src.models.losses import build_loss
from src.models.registry import build_model
from src.training import trainer
from src.training.trainer import train


def test_checkpoint_written_every_checkpoint_every_epochs_and_at_final_epoch(monkeypatch, synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 7
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    checkpointed_epochs = []
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda path, epoch, *a, **k: checkpointed_epochs.append(epoch))
    monkeypatch.setattr(trainer, "_save_best_checkpoint", lambda *a, **k: None)

    train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path="unused-because-save-is-stubbed.pt", checkpoint_every_epochs=3,
    )

    # 7 epochs (0..6): flushes at (epoch+1) % 3 == 0 -> epochs 2, 5, plus
    # always at the final epoch (6), even though 7 isn't a multiple of 3.
    assert checkpointed_epochs == [2, 5, 6]


def test_resume_continues_from_the_next_epoch_with_restored_weights(tmp_path, synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)
    checkpoint_path = tmp_path / "run0.pt"

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)
    config.training.num_epochs = 3
    history_first_half, _ = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path=checkpoint_path, checkpoint_every_epochs=1,
    )
    assert len(history_first_half) == 3

    # Fresh model/dataset objects, as a real process restart would force --
    # only the checkpoint file on disk survives a crash.
    dataset2 = SegmentDataset(utterances, segment_length, "OS", seed=1)
    model2 = build_model(config)
    loss_module2 = build_loss(config, bottleneck_dim=512, num_speakers=3)
    config.training.num_epochs = 5
    history_second_half, _ = train(
        model2, loss_module2, dataset2, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path=checkpoint_path, checkpoint_every_epochs=1,
    )

    # Only epochs 3, 4 remain (resumed from epoch 2's checkpoint), not a
    # fresh 5-epoch run.
    assert len(history_second_half) == 2

    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["epoch"] == 4
    # train() restores the best-dev-EER epoch's weights before returning
    # (which isn't necessarily the last-trained epoch), so the model's final
    # state must match the checkpoint's best snapshot, not its live one.
    for key in model2.state_dict():
        assert torch.equal(model2.state_dict()[key], checkpoint["best_model_state_dict"][key])


def test_resume_from_checkpoint_missing_dev_eer_history_key_does_not_crash(tmp_path, synthetic_utterances):
    """Checkpoints written by a version of this code that predates
    dev_eer_history tracking don't have that key at all -- this is exactly
    what a real resume on the cluster hit (KeyError: 'dev_eer_history'),
    since every test that exercises resume creates its checkpoints with the
    current code and so always has the key. Resuming from such a checkpoint
    must not crash; it should just start the min_improvement_rate
    condition's history fresh from the resume point."""
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 6
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)
    checkpoint_path = tmp_path / "run0.pt"

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)
    optimizer = torch.optim.Adam(
        trainer._weight_decay_param_groups(model, loss_module, config.optimizer.weight_decay),
        lr=config.optimizer.learning_rate,
    )

    # Hand-crafted old-format checkpoint: epoch 2 done, best at epoch 0, no
    # "dev_eer_history" key.
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_module_state_dict": loss_module.state_dict(),
            "best_metric": 0.1,
            "best_epoch": 0,
            "best_model_state_dict": model.state_dict(),
            "rng_state": trainer._rng_state(dataset, np.random.default_rng()),
        },
        checkpoint_path,
    )

    history, _ = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path=checkpoint_path, checkpoint_every_epochs=1,
        early_stopping_patience=3, min_improvement_rate=0.10,
    )

    # Epochs 3, 4, 5 remain (resumed from epoch 2's checkpoint).
    assert len(history) == 3


def test_resume_produces_bit_identical_training_to_an_uninterrupted_run(tmp_path, synthetic_utterances):
    """Crashing after 3 of 6 epochs and resuming from a fresh process (fresh
    model/dataset/optimizer objects, as a real crash would force) must
    reproduce exactly the same remaining-epoch losses and final weights as an
    uninterrupted 6-epoch run -- only possible if every source of randomness
    (torch's global RNG for dropout/DataLoader shuffling, the dataset's own
    segment-draw RNG, and the dev-eval RNG) is captured in the checkpoint and
    restored bit-exactly."""
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    def _run(num_epochs, checkpoint_path=None):
        torch.manual_seed(42)
        model = build_model(config)
        loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)
        dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
        config.training.num_epochs = num_epochs
        history, _ = train(
            model, loss_module, dataset, config,
            dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
            checkpoint_path=checkpoint_path, checkpoint_every_epochs=1,
        )
        return model, history

    model_a, history_a = _run(num_epochs=6)

    checkpoint_path = tmp_path / "run0.pt"
    _, history_first_half = _run(num_epochs=3, checkpoint_path=checkpoint_path)
    model_b, history_second_half = _run(num_epochs=6, checkpoint_path=checkpoint_path)

    assert history_first_half == history_a[:3]
    assert history_second_half == history_a[3:]

    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        torch.testing.assert_close(state_a[key], state_b[key])


def test_early_stopping_halts_training_after_patience_epochs_without_improvement(monkeypatch, synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 20
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    # Best dev EER occurs at epoch 0; every later epoch is strictly worse, so
    # with patience=3 training must stop after epoch 3 (3 epochs with no
    # improvement past the epoch-0 best), well short of num_epochs=20.
    eer_sequence = iter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: next(eer_sequence))

    history, _ = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        early_stopping_patience=3,
    )

    assert len(history) == 4


def test_early_stopping_halts_on_insufficient_relative_improvement(monkeypatch, synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 20
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    # Every epoch is a (tiny) improvement over the last, so the plain
    # no-improvement-at-all patience check alone would never fire. But with
    # min_improvement_rate=0.10 and patience=3, epoch 3's dev EER (0.188)
    # is only a 6% relative drop from epoch 0's (0.20) -- short of the
    # required 10% -- so training must stop right after epoch 3.
    eer_sequence = iter([0.20, 0.196, 0.192, 0.188, 0.184, 0.180])
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: next(eer_sequence))

    history, _ = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        early_stopping_patience=3, min_improvement_rate=0.10,
    )

    assert len(history) == 4


def test_config_early_stopping_patience_null_disables_early_stopping(monkeypatch, synthetic_utterances):
    """training.early_stopping_patience: null in a YAML config must round-trip
    to Python None and, passed straight through to train(), train the full
    num_epochs regardless of dev EER never improving -- matching the paper's
    own protocol (see src/README.md), which never stops early."""
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig.from_dict({"training": {"early_stopping_patience": None}})
    assert config.training.early_stopping_patience is None
    config.training.num_epochs = 6
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    # Dev EER never improves past epoch 0 -- with any finite patience this
    # would stop early, same setup as the patience=3 test above.
    eer_sequence = iter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: next(eer_sequence))

    history, _ = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        early_stopping_patience=config.training.early_stopping_patience,
    )

    assert len(history) == 6


def test_resume_does_not_retrain_past_an_already_exhausted_patience_window(monkeypatch, tmp_path, synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 20
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)
    checkpoint_path = tmp_path / "run0.pt"

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    eer_sequence = iter([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: next(eer_sequence))
    train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path=checkpoint_path, checkpoint_every_epochs=1, early_stopping_patience=3,
    )
    # Training already stopped early (epoch 3, best at epoch 0) and the eer
    # stub above has no more values queued -- if resume incorrectly kept
    # training, the next equal_error_rate() call would raise StopIteration.

    model2 = build_model(config)
    loss_module2 = build_loss(config, bottleneck_dim=512, num_speakers=3)
    dataset2 = SegmentDataset(utterances, segment_length, "OS", seed=1)
    history, _ = train(
        model2, loss_module2, dataset2, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path=checkpoint_path, checkpoint_every_epochs=1, early_stopping_patience=3,
    )

    assert history == []


def test_best_checkpoint_updates_on_every_improvement_not_just_periodic_cadence(monkeypatch, tmp_path, synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 5
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)
    checkpoint_path = tmp_path / "run0.pt"

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    # Best dev EER occurs at epoch 2; later epochs are worse but not worse
    # long enough to trigger early stopping (none is configured here).
    eer_sequence = iter([0.5, 0.4, 0.3, 0.35, 0.4])
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: next(eer_sequence))

    main_checkpoint_epochs = []
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda path, epoch, *a, **k: main_checkpoint_epochs.append(epoch))

    train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path=checkpoint_path, checkpoint_every_epochs=100,
    )

    # checkpoint_every_epochs=100 never divides evenly within a 5-epoch run,
    # so the periodic/final-epoch checkpoint only ever fires once, at the
    # very last epoch (4).
    assert main_checkpoint_epochs == [4]

    # The best-checkpoint companion file must already reflect the true best
    # epoch (2), not epoch 4 -- proving it was written immediately on the
    # epoch-2 improvement rather than waiting for the periodic/final save.
    best_checkpoint = torch.load(trainer.best_checkpoint_path(checkpoint_path))
    assert best_checkpoint["epoch"] == 2
    assert best_checkpoint["metric"] == 0.3


def test_paper_comparable_best_only_considers_the_fixed_11_epoch_schedule(monkeypatch, synthetic_utterances):
    # Matches context/src/utils.py's TEST_EPOCHS default: np.linspace(0, 20, 11)
    # over num_epochs=21 selects exactly the even epochs {0, 2, 4, ..., 20}.
    # The global dev-EER minimum below sits at epoch 1 -- an odd, excluded
    # epoch -- so paper_comparable_best must report the best *even* epoch's
    # value (0.05, epoch 6), not the true global best (0.01, epoch 1), which
    # is what the ordinary every-epoch best_metric tracking would use instead.
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 21
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    eer_sequence = [0.9] * 21
    eer_sequence[1] = 0.01
    eer_sequence[6] = 0.05
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: eer_sequence.pop(0))

    _, paper_comparable_best = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
    )

    assert paper_comparable_best == 0.05


def test_paper_comparable_best_schedule_is_fixed_against_configured_num_epochs(monkeypatch, tmp_path, synthetic_utterances):
    # If the 11-epoch schedule were instead computed from however many epochs
    # this run actually completes (rather than the configured num_epochs), it
    # would be contaminated by early stopping's own epoch-by-epoch dev-EER
    # signal (patience-based stopping always ends close to the true best
    # epoch -- see the reasoning in trainer.train's docstring). Here
    # num_epochs=21 fixes the schedule at {0, 2, 4, ..., 20} *before* training
    # starts; early stopping (patience=2, best epoch=1) then cuts the run
    # short after epoch 3, so only schedule epochs {0, 2} were ever reached.
    # The correct result is min(epoch 0, epoch 2) = 0.4 -- if the schedule
    # were instead recomputed over the actual 4-epoch run length, epochs 1
    # and 3 would also be in-bounds and the (wrong) result would be 0.01.
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 21
    config.training.batch_size = 4
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    eer_sequence = iter([0.5, 0.01, 0.4, 0.6])
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: next(eer_sequence))

    history, paper_comparable_best = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        early_stopping_patience=2,
    )

    assert len(history) == 4  # confirms early stopping actually cut the 21-epoch run short
    assert paper_comparable_best == 0.4


def test_patience_none_never_triggers_regardless_of_other_arguments():
    assert trainer._early_stopping_triggered(0, 0, [], None, None) is False
    assert trainer._early_stopping_triggered(100, 0, [0.9] * 101, None, 0.5) is False


def test_plain_no_improvement_patience_is_a_closed_lower_bound():
    # min_improvement_rate=None: pure epoch/best_epoch arithmetic, patience is
    # a ">=" (closed) bound, not a ">" one.
    assert trainer._early_stopping_triggered(4, 2, [], 3, None) is False  # 2 epochs since best: not yet
    assert trainer._early_stopping_triggered(5, 2, [], 3, None) is True   # exactly patience: stop
    assert trainer._early_stopping_triggered(6, 2, [], 3, None) is True   # past patience: stop


def test_plain_check_ignores_dev_eer_history_contents():
    # With min_improvement_rate=None, dev_eer_history is never consulted --
    # only epoch/best_epoch arithmetic matters, however the history is shaped.
    assert trainer._early_stopping_triggered(5, 2, [], 3, None) is True
    assert trainer._early_stopping_triggered(5, 2, [0.99, 0.01, 0.5], 3, None) is True


def test_rate_check_never_triggers_before_a_full_patience_window_of_history_exists():
    # epoch - patience < 0: even a wildly non-improving history must not
    # trigger, since there isn't yet a full patience-epoch window to judge.
    assert trainer._early_stopping_triggered(1, 0, [0.9, 0.9], 3, 0.10) is False


def test_rate_check_activates_starting_exactly_at_epoch_equals_patience():
    # patience=3: epoch=2 (epoch-patience=-1) is one epoch short of a full
    # window -- must not trigger regardless of the (here, flat/bad) history.
    assert trainer._early_stopping_triggered(2, 0, [0.5, 0.5, 0.5], 3, 0.10) is False
    # epoch=3 (epoch-patience=0): the first epoch with a full patience-sized
    # window available -- flat history here has no significant improvement.
    assert trainer._early_stopping_triggered(3, 0, [0.5, 0.5, 0.5, 0.5], 3, 0.10) is True


def test_significant_improvement_at_the_very_start_of_the_window_prevents_stop():
    history = [1.0, 1.0, 0.5, 0.9, 0.95]  # epoch 2 (first epoch of the window) is the big win
    assert trainer._early_stopping_triggered(4, 2, history, 3, 0.10) is False


def test_significant_improvement_buried_in_the_middle_of_the_window_prevents_stop():
    # Regression test for the bug this function used to have: it only
    # compared the window's two endpoints (dev_eer_history[epoch-patience] vs
    # dev_eer_history[epoch]), so a big win in the middle of the window
    # followed by a partial regression back toward the window's starting
    # value was invisible to it -- epoch 2's 90% improvement here would have
    # been missed entirely, incorrectly stopping training.
    history = [0.50, 0.48, 0.05, 0.30, 0.35, 0.49]
    assert trainer._early_stopping_triggered(5, 2, history, 5, 0.10) is False


def test_significant_improvement_only_at_the_final_epoch_of_the_window_prevents_stop():
    history = [1.0, 1.0, 0.95, 0.93, 0.5]  # epoch 4 (last epoch of the window) is the big win
    assert trainer._early_stopping_triggered(4, 4, history, 3, 0.10) is False


def test_no_epoch_in_the_window_reaches_the_required_rate_triggers_stop():
    # Every epoch is a (tiny) improvement over the last -- so the plain
    # no-improvement check alone would never fire -- but none clears the 10%
    # bar measured against the running best just before it.
    history = [0.20, 0.196, 0.192, 0.188]
    assert trainer._early_stopping_triggered(3, 3, history, 3, 0.10) is True


def test_flat_or_worsening_history_triggers_stop():
    history = [0.5, 0.5, 0.6, 0.7]
    assert trainer._early_stopping_triggered(3, 0, history, 3, 0.10) is True


def test_improvement_exactly_at_the_threshold_counts_as_significant():
    # <=, not <: a relative drop of exactly min_improvement_rate must count.
    assert trainer._early_stopping_triggered(1, 1, [1.0, 0.8], 1, 0.20) is False


def test_improvement_just_short_of_the_threshold_triggers_stop():
    assert trainer._early_stopping_triggered(1, 1, [1.0, 0.81], 1, 0.20) is True  # 19% < 20%


def test_rate_zero_treats_any_improvement_or_tie_as_significant():
    assert trainer._early_stopping_triggered(1, 1, [0.5, 0.5], 1, 0.0) is False   # tie counts
    assert trainer._early_stopping_triggered(1, 1, [0.5, 0.51], 1, 0.0) is True   # strictly worse -> stop


def test_rate_one_requires_hitting_zero():
    assert trainer._early_stopping_triggered(1, 1, [0.5, 0.0], 1, 1.0) is False
    assert trainer._early_stopping_triggered(1, 1, [0.5, 0.0001], 1, 1.0) is True


def test_short_history_after_a_pre_history_tracking_checkpoint_resume_does_not_crash():
    # A checkpoint written before dev_eer_history was tracked resumes with a
    # fresh, shorter history no longer aligned to absolute epoch numbers (see
    # test_resume_from_checkpoint_missing_dev_eer_history_key_does_not_crash) --
    # epoch=5 would index dev_eer_history[5] on a 1-element list without this
    # guard. Too little real history to judge -- don't trigger, don't crash.
    assert trainer._early_stopping_triggered(5, 0, [0.01], 3, 0.10) is False


def test_best_epoch_argument_is_unused_once_min_improvement_rate_is_set():
    # Once a rate is given, the rate check fully replaces the plain
    # best_epoch-based check (see the function's docstring) -- the result
    # must depend only on dev_eer_history, not on whatever best_epoch says.
    history = [0.20, 0.196, 0.192, 0.188]
    assert trainer._early_stopping_triggered(3, -999, history, 3, 0.10) is True
    assert trainer._early_stopping_triggered(3, 3, history, 3, 0.10) is True


def test_window_excludes_epochs_older_than_patience_even_if_they_improved_a_lot():
    # patience=2, epoch=3 -> window is epochs {2, 3} only. Epoch 1's huge win
    # is now stale (outside the window) -- it still lowers the running-best
    # baseline the window is judged against (correctly raising the bar), but
    # it must not itself count as a "recent" significant improvement.
    history = [1.0, 0.01, 0.95, 0.94]
    assert trainer._early_stopping_triggered(3, 1, history, 2, 0.10) is True


def test_restore_rng_state_moves_the_torch_state_tensors_back_to_cpu(monkeypatch):
    """train()'s torch.load(checkpoint_path, map_location=device) moves every
    tensor in the checkpoint onto `device`, including the CPU-only tensors
    returned by torch.get_rng_state() and torch.cuda.get_rng_state_all() --
    on a CUDA device that silently turns them into torch.cuda.ByteTensors,
    which the CPU-ByteTensor-only set_rng_state()/set_rng_state_all() then
    reject with "RNG state must be a torch.ByteTensor". Can't reproduce the
    real failure without a GPU, so this stands in fake tensors and checks
    .cpu() is called on each before it reaches set_rng_state/set_rng_state_all."""

    class FakeCudaTensor:
        def __init__(self):
            self.cpu_called = False

        def cpu(self):
            self.cpu_called = True
            return self

    fake_cpu_state = FakeCudaTensor()
    fake_cuda_states = [FakeCudaTensor(), FakeCudaTensor()]
    received_cpu = []
    received_cuda = []
    monkeypatch.setattr(torch, "set_rng_state", lambda state: received_cpu.append(state))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: received_cuda.append(states))

    dataset = SimpleNamespace(rng=SimpleNamespace(bit_generator=SimpleNamespace(state=None)))
    dev_eval_rng = SimpleNamespace(bit_generator=SimpleNamespace(state=None))

    trainer._restore_rng_state(
        {
            "torch": fake_cpu_state,
            "torch_cuda": fake_cuda_states,
            "dataset": "dataset-state",
            "dev_eval": "dev-eval-state",
        },
        dataset, dev_eval_rng,
    )

    assert fake_cpu_state.cpu_called
    assert received_cpu == [fake_cpu_state]
    assert all(t.cpu_called for t in fake_cuda_states)
    assert received_cuda == [fake_cuda_states]


def test_paper_checkpoint_epochs_restricts_best_checkpoint_to_the_11_epoch_schedule(monkeypatch, tmp_path, synthetic_utterances):
    # Same schedule as above (num_epochs=21 -> even epochs only): with
    # paper_checkpoint_epochs set, the odd-epoch global minimum (epoch 1) must
    # not be kept; the best even epoch (6) is.
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    config = ExperimentConfig()
    config.training.num_epochs = 21
    config.training.batch_size = 4
    config.training.paper_checkpoint_epochs = True
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=3)

    eer_sequence = [0.9] * 21
    eer_sequence[1] = 0.01
    eer_sequence[6] = 0.05
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: eer_sequence.pop(0))

    kept_epochs = []
    monkeypatch.setattr(trainer, "_save_best_checkpoint", lambda path, epoch, metric, state: kept_epochs.append(epoch))

    train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        checkpoint_path=tmp_path / "run.pt",
    )

    assert kept_epochs == [0, 6]
