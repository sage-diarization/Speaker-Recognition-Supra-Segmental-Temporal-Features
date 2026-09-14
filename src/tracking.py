"""Optional Weights & Biases progress tracking.

Scoped one run per (model, dataset, training-strategy) -- covering all of
that strategy's config.num_runs repeats plus their aggregate mean/SD, since a
sweep's individual repeats aren't independently interesting (only their
aggregate is), mirroring context/src/train.py's per-epoch callback logging
(minus the Keras-specific weight-histogram logging, which isn't meaningful
for a plain PyTorch loop) but collapsed into one run per strategy instead of
one per repeat. `run` is either a wandb Run or None throughout this module
and the training loop, so callers don't need to branch on whether tracking
is enabled.
"""


def start_run(wandb_config, name, tags, run_config, run_id=None):
    """run_id, when given, reattaches to a previously started run (see
    src/experiment.py's per-sweep manifest) instead of starting a new one --
    used both to resume logging into the same run after a crash and to let a
    strategy's config.num_runs repeats share a single run."""
    if not wandb_config.enabled:
        return None

    import wandb

    return wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity or None,
        mode=wandb_config.mode,
        id=run_id,
        resume="allow" if run_id is not None else None,
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
