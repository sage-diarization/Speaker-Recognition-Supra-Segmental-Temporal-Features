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
    history_first_half = train(
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
    history_second_half = train(
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
        history = train(
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

    history = train(
        model, loss_module, dataset, config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
        early_stopping_patience=3,
    )

    assert len(history) == 4


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
    history = train(
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
