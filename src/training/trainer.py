import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import tracking
from ..data.lazy_features import resolve_features
from ..evaluation.verification import equal_error_rate


def _rng_state(dataset, dev_eval_rng):
    """Full random state needed for bit-exact resume: torch's global RNG
    (consumed both by dropout during the forward pass and by DataLoader's
    default RandomSampler, which reseeds itself from the global RNG at the
    start of every epoch's iteration), the SegmentDataset's own numpy
    Generator (its on-the-fly per-epoch segment draws), and this module's own
    persistent Generator for periodic dev-eval segment draws (see
    extract_embeddings' `rng` param)."""
    state = {
        "torch": torch.get_rng_state(),
        "dataset": dataset.rng.bit_generator.state,
        "dev_eval": dev_eval_rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, dataset, dev_eval_rng):
    # state["torch"] is torch.get_rng_state()'s CPU-only ByteTensor (the CUDA
    # generators' state is restored separately below), but torch.load's
    # map_location=device in train() applies to every tensor in the
    # checkpoint indiscriminately -- on a CUDA device that turns this one
    # into a torch.cuda.ByteTensor too, which the CPU-only set_rng_state()
    # rejects. Force it back to CPU regardless of what map_location did to it.
    torch.set_rng_state(state["torch"].cpu())
    dataset.rng.bit_generator.state = state["dataset"]
    dev_eval_rng.bit_generator.state = state["dev_eval"]
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_checkpoint(path, epoch, model, optimizer, loss_module, best_metric, best_epoch, best_state, rng_state):
    """Writes to a temp file then renames into place (atomic on POSIX) so a
    crash mid-write can never leave a corrupt checkpoint that fails to load
    on resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_module_state_dict": loss_module.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_model_state_dict": best_state,
            "rng_state": rng_state,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def best_checkpoint_path(checkpoint_path):
    """Companion file next to checkpoint_path holding just the best-dev-EER
    weights so far, rewritten on every improvement (see
    _save_best_checkpoint) -- unlike checkpoint_path itself, not gated behind
    checkpoint_every_epochs."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.best{checkpoint_path.suffix}")


def _save_best_checkpoint(path, epoch, metric, model_state):
    """Rewritten on every dev-EER improvement (not just every
    checkpoint_every_epochs epochs like _save_checkpoint), so the best
    weights found so far are never more than one epoch stale on disk even if
    the process is killed before the next periodic checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"epoch": epoch, "metric": metric, "model_state_dict": model_state}, tmp_path)
    os.replace(tmp_path, path)


def train(model, loss_module, dataset, config, dev_utterances=None, segment_length=None,
          draw_strategy=None, run=None, device="cpu", run_idx=None,
          checkpoint_path=None, checkpoint_every_epochs=None, early_stopping_patience=None,
          dev_eval_fn=None):
    """Trains for up to config.training.num_epochs. When dev_utterances is
    given, evaluates dev-set SV EER after every epoch, stops once dev EER
    hasn't improved for early_stopping_patience epochs, and keeps the
    best-scoring checkpoint's weights on the model at the end of training --
    matches context/src's EvalCallback + get_reference_data, which is what
    Neururer et al. 2024's Tables 1/2 numbers are actually computed from (the
    best dev checkpoint, not whatever state training happens to end in).

    dev_eval_fn(embeddings, labels) -> eer defaults to equal_error_rate's
    exhaustive all-pairs comparison (TIMIT's SV protocol); pass a different
    function (e.g. a closure over trial_list_equal_error_rate) for a dataset
    whose SV evaluation is instead defined over a fixed trial-pairs list
    (VoxCeleb -- see src/experiment.py).

    If checkpoint_path is given, a checkpoint is written every
    checkpoint_every_epochs epochs (and always on the epoch that triggers
    early stopping or completes training) and automatically resumed from if
    the file already exists -- including full bit-exact RNG state (torch's
    global RNG, the dataset's segment-draw RNG, and this function's own
    dev-eval RNG), so a crash-and-resume continues the exact same random
    sequence an uninterrupted run would have produced. If the reloaded
    best_epoch already implies early_stopping_patience was exhausted before
    the crash, training is not resumed at all -- just the best weights are
    restored -- since that run had already finished.

    Independently of that periodic cadence, best_checkpoint_path(checkpoint_path)
    is (re)written on every dev-EER improvement, so the best weights on disk
    are never more than one epoch stale even if the process is killed between
    two periodic checkpoints."""
    model.to(device)
    loss_module.to(device)
    loader = DataLoader(dataset, batch_size=min(config.training.batch_size, len(dataset)), shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    start_epoch = 0
    best_metric, best_epoch, best_state = None, -1, None
    dev_eval_rng = np.random.default_rng()
    metric_prefix = f"run{run_idx}/" if run_idx is not None else ""
    dev_eval_fn = dev_eval_fn or equal_error_rate

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_module.load_state_dict(checkpoint["loss_module_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        best_state = checkpoint["best_model_state_dict"]
        _restore_rng_state(checkpoint["rng_state"], dataset, dev_eval_rng)

    already_stopped = (
        early_stopping_patience is not None
        and best_epoch >= 0
        and (start_epoch - 1 - best_epoch) >= early_stopping_patience
    )

    model.train()
    loss_module.train()
    history = []
    if not already_stopped:
        for epoch in range(start_epoch, config.training.num_epochs):
            epoch_loss, n_batches = 0.0, 0
            for x, labels in loader:
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss, _ = loss_module(output.bottleneck, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            mean_loss = epoch_loss / max(n_batches, 1)
            history.append(mean_loss)
            tracking.log(run, {f"{metric_prefix}train/loss": mean_loss})

            if dev_utterances is not None:
                dev_embeddings, dev_labels = extract_embeddings(
                    model, dev_utterances, segment_length, draw_strategy, device=device, rng=dev_eval_rng
                )
                dev_eer = dev_eval_fn(dev_embeddings, dev_labels)
                tracking.log(run, {f"{metric_prefix}dev/EER": dev_eer})
                if best_metric is None or dev_eer < best_metric:
                    best_metric, best_epoch, best_state = dev_eer, epoch, copy.deepcopy(model.state_dict())
                    if checkpoint_path is not None:
                        _save_best_checkpoint(best_checkpoint_path(checkpoint_path), best_epoch, best_metric, best_state)
                model.train()

                stopped_early = (
                    early_stopping_patience is not None and (epoch - best_epoch) >= early_stopping_patience
                )
                should_checkpoint = checkpoint_path is not None and (
                    stopped_early
                    or epoch == config.training.num_epochs - 1
                    or (checkpoint_every_epochs is not None and (epoch + 1) % checkpoint_every_epochs == 0)
                )
                if should_checkpoint:
                    _save_checkpoint(
                        checkpoint_path, epoch, model, optimizer, loss_module,
                        best_metric, best_epoch, best_state, _rng_state(dataset, dev_eval_rng),
                    )
                if stopped_early:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _eval_windows(features, segment_length, step, num_windows, draw_strategy, rng):
    """Builds the num_windows hop-window segments for one utterance, matching
    context/src/setup/setup.py's precomputed EVAL distributions -- the ones
    context/src/generator/sampler.py::load_test actually reads at test time
    -- rather than src/data/segments.py's on-the-fly *training* sampler
    (a port of context/src/generator/sampler.py::load_OT/RS/RF), which draws
    a fresh random position per call. OS/SS windows walk
    sequential, evenly-spaced start positions (current_start stepping by
    `step`) instead of resampling a random position each time; SS also
    reshuffles each window's frame order fresh, matching setup.py's
    per-iteration np.random.shuffle(rseg_dist). SU -- 'RF' -- draws
    segment_length frames with replacement from the *entire* utterance for
    every window (np.random.choice(time_dist, segment_length), time_dist
    spanning the whole utterance), independent of hop position -- not the
    bounded local sub-window that src/data/segments.py::draw_su uses for
    on-the-fly training batches. That bounded-window/full-utterance
    distinction barely matters for single-sentence SV test utterances but is
    the dominant source of the SC misclassification-rate gap against
    Neururer et al. 2024's Table 1, since SC test utterances are much
    longer (2-8 concatenated sentences)."""
    if draw_strategy == "SU":
        length = features.shape[0]
        return [features[rng.integers(0, length, segment_length)] for _ in range(num_windows)]
    segments = []
    for i in range(num_windows):
        start = i * step
        segment = features[start:start + segment_length].copy()
        if draw_strategy == "SS":
            rng.shuffle(segment)
        segments.append(segment)
    return segments


def extract_embeddings(model, utterances, segment_length, draw_strategy, seed=None, device="cpu", hop_fraction=0.5, rng=None):
    """Draws multiple hop-window segments per utterance (step =
    hop_fraction * segment_length; hop_fraction=0.5, i.e. 'H50', is the
    setting Neururer et al. 2024's actual Table 1/2 numbers come from --
    context/src/evaluation/utils.py hardcodes 'H50' into its reference-lookup
    key) and averages their embeddings into one per-utterance embedding,
    matching context/src/generator/generator.py:399's
    np.mean(current_embeddings[indices], axis=0) over hop-window embeddings
    (see _eval_windows for how each window is drawn). A single segment's
    worth of utterance just yields that one segment's embedding, matching
    the original's `while (current_start + segment_length) <= sample_length`
    loop, which always runs at least once.

    `rng`, when given, is used (and mutated) directly instead of building a
    fresh Generator from `seed` -- lets a caller (train()'s periodic dev
    eval) own a single persistent, checkpoint-able Generator across many
    calls instead of reseeding every time."""
    if rng is None:
        rng = np.random.default_rng(seed)
    step = max(int(hop_fraction * segment_length), 1)

    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for features_or_loader, label in utterances:
            features = resolve_features(features_or_loader)
            length = features.shape[0]
            if length <= segment_length:
                continue
            num_windows = (length - segment_length) // step + 1
            segments = _eval_windows(features, segment_length, step, num_windows, draw_strategy, rng)
            tensor = torch.from_numpy(np.stack(segments)).float().unsqueeze(1).to(device)
            output = model(tensor)
            embeddings.append(output.backend.mean(dim=0).cpu().numpy())
            labels.append(label)
    return np.stack(embeddings), np.array(labels)
