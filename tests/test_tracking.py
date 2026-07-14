from src import tracking
from src.config import WandbConfig
from src.data.dataset import SegmentDataset
from src.models.losses import build_loss
from src.models.registry import build_model
from src.training.trainer import train


def test_disabled_config_never_starts_a_run():
    run = tracking.start_run(WandbConfig(enabled=False), name="x", tags=[], run_config={})
    assert run is None


def test_log_summary_and_finish_are_safe_noops_for_no_run():
    tracking.log(None, {"a": 1})
    tracking.log_summary(None, {"a": 1})
    tracking.finish(None)


def test_enabled_config_in_disabled_mode_starts_a_real_run_without_network():
    # wandb's own mode="disabled" fully no-ops (no login/network needed), so this
    # exercises the real wandb.init/log/summary/finish call path in CI.
    config = WandbConfig(enabled=True, mode="disabled", project="pytest-project")
    run = tracking.start_run(config, name="run-name", tags=["OS"], run_config={"a": 1})
    assert run is not None

    tracking.log(run, {"train/loss": 0.5}, step=0)
    tracking.log_summary(run, {"final/SV_EER_test-OS": 0.1})
    tracking.finish(run)


class _FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((step, metrics))


def test_train_logs_loss_to_the_given_run_every_epoch(small_config, synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    small_config.loss.type = "SOFTMAX"
    segment_length = small_config.data.segment_length(small_config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(small_config)
    loss_module = build_loss(small_config, bottleneck_dim=512, num_speakers=3)

    run = _FakeRun()
    train(model, loss_module, dataset, small_config, run=run)

    assert len(run.logged) == small_config.training.num_epochs
    steps = [step for step, _ in run.logged]
    assert steps == list(range(small_config.training.num_epochs))
    assert all("train/loss" in metrics for _, metrics in run.logged)
