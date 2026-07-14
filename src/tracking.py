"""Optional Weights & Biases progress tracking.

Scoped one run per training-strategy (OS/SS/SU), mirroring
context/src/train.py's per-run wandb.init + per-epoch callback logging (minus
the Keras-specific weight-histogram logging, which isn't meaningful for a
plain PyTorch loop). `run` is either a wandb Run or None throughout this
module and the training loop, so callers don't need to branch on whether
tracking is enabled.
"""


def start_run(wandb_config, name, tags, run_config):
    if not wandb_config.enabled:
        return None

    import wandb

    return wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity or None,
        mode=wandb_config.mode,
        name=name,
        tags=[*tags, *wandb_config.tags],
        config=run_config,
        reinit=True,
    )


def log(run, metrics, step=None):
    if run is not None:
        run.log(metrics, step=step)


def log_summary(run, metrics):
    if run is not None:
        for key, value in metrics.items():
            run.summary[key] = value


def finish(run):
    if run is not None:
        run.finish()
