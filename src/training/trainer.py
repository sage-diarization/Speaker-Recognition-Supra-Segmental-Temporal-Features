import copy
import gc
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import tracking
from ..data.lazy_features import resolve_features
from ..data.segments import MIXTURES, gapped_segment, shuffle_blocks
from ..evaluation.verification import equal_error_rate


def _weight_decay_param_groups(model, loss_module, weight_decay):
    """context/src only ever L2-regularizes Conv2D/Dense kernels (and, for
    ResNet's aggregation head, their biases) via Keras' kernel_regularizer/
    bias_regularizer -- BatchNorm's affine params, GhostVLAD's cluster
    centers (context/src/models/aggregation/blocks/vlad.py's `self.cluster`,
    added via add_weight with no regularizer), and the angular-margin loss's
    classifier matrix (context/src/models/losses/angular_margin.py's `W`,
    same) are never regularized there. torch.optim.Adam's weight_decay
    applies indiscriminately to every parameter passed to it, so split
    params into decay/no-decay groups to match, instead of handing it
    model.parameters() + loss_module.parameters() directly."""
    decay, no_decay = [], []
    for module in model.modules():
        is_norm = isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
        for name, param in module.named_parameters(recurse=False):
            if is_norm or name == "cluster":
                no_decay.append(param)
            else:
                decay.append(param)
    no_decay.extend(loss_module.parameters())
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _seed_worker_segment_rng(worker_id):
    """DataLoader worker_init_fn: each worker process starts with a copy of
    the SegmentDataset, including its segment-draw Generator's state, so
    without reseeding every worker would draw identical segment positions.
    The worker's torch seed (base seed + worker id, the base seed drawn fresh
    each epoch from the main process's torch RNG) is unique per worker and
    epoch, and reproducible on resume since that RNG is checkpointed."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)


def _fork_context(num_workers):
    # "fork" (where available) hands workers the dataset copy-on-write --
    # Python 3.14's forkserver default would instead pickle all of it (1M+
    # LazyFeatures entries on VoxCeleb2) into every worker, every epoch.
    fork = num_workers > 0 and "fork" in multiprocessing.get_all_start_methods()
    if fork:
        # A worker's first full garbage collection would otherwise visit every
        # object inherited from this process, copying the memory pages they
        # live on (~1 GB per worker on VoxCeleb2) -- with 32 workers living
        # for hours during the final evaluation, enough to get the job
        # OOM-killed. Frozen objects are skipped by the garbage collector
        # (here and in the workers); reference counting still frees them.
        gc.freeze()
    return "fork" if fork else None


class _ResolvedUtterances(torch.utils.data.Dataset):
    def __init__(self, utterances):
        self.utterances = utterances

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        features_or_loader, label = self.utterances[idx]
        try:
            return resolve_features(features_or_loader), label
        except Exception as error:
            # Formatted here, in the worker: DataLoader re-raises a worker's
            # exception by calling its class with just the message, which
            # e.g. soundfile's LibsndfileError (expecting an error code)
            # can't take, losing the file and the reason.
            raise RuntimeError(f"Featurizing utterance {label!r} failed: {type(error).__name__}: {error}") from error


def _keep_numpy(item):
    # DataLoader's default (batch_size=None) collate would turn the features
    # into tensors; extract_embeddings' window draws expect ndarrays.
    return item


def resolved_utterances(utterances, num_workers=0):
    """(features, label) for every utterance entry, in order, with lazy
    entries (src/data/lazy_features.py) featurized -- by num_workers
    DataLoader worker processes when > 0, so reading and featurizing
    evaluation utterances runs in parallel instead of one at a time in the
    main process."""
    if num_workers == 0:
        return ((resolve_features(features_or_loader), label) for features_or_loader, label in utterances)
    return DataLoader(
        _ResolvedUtterances(utterances), batch_size=None, collate_fn=_keep_numpy,
        num_workers=num_workers, multiprocessing_context=_fork_context(num_workers),
        # Its own Generator: starting the workers draws a seed from it, which
        # from torch's global RNG would shift training's shuffling/dropout.
        generator=torch.Generator(),
    )


def _synchronize(device):
    """Waits for queued GPU work, so wall-clock timers measure it too."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


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
    # torch.get_rng_state() and torch.cuda.get_rng_state_all() both return
    # CPU-only ByteTensors (even the latter, despite describing the CUDA
    # generators' state), but torch.load's map_location=device in train()
    # applies to every tensor in the checkpoint indiscriminately -- on a CUDA
    # device that turns both into torch.cuda.ByteTensors, which set_rng_state
    # and set_rng_state_all (both CPU-ByteTensor-only) then reject. Force
    # everything back to CPU regardless of what map_location did to it.
    torch.set_rng_state(state["torch"].cpu())
    dataset.rng.bit_generator.state = state["dataset"]
    dev_eval_rng.bit_generator.state = state["dev_eval"]
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["torch_cuda"]])


def _save_checkpoint(path, epoch, model, optimizer, loss_module, best_metric, best_epoch, best_state,
                      dev_eer_history, rng_state, paper_comparable_best):
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
            "dev_eer_history": dev_eer_history,
            "rng_state": rng_state,
            "paper_comparable_best": paper_comparable_best,
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


def _early_stopping_triggered(epoch, best_epoch, dev_eer_history, patience, min_improvement_rate):
    """True if patience epochs have passed since the last epoch that improved
    on the running-best dev EER at all (plain no-improvement stopping) -- or,
    when min_improvement_rate is set, since the last epoch that improved on
    the running-best-as-of-just-before-it by at least that relative rate
    (e.g. rate=0.10 requires at least a 10% relative drop). The rate check
    scans every epoch in the trailing patience-epoch window individually
    against its own immediately-preceding running best, rather than just
    comparing the window's two endpoints: an epoch deep inside the window can
    be the one significant improvement, even if a later epoch in the same
    window regresses back up toward the window's starting value -- comparing
    only the endpoints would incorrectly stop training in that case, missing
    the recent significant improvement entirely. Requiring *any* significant
    improvement anywhere in the window is a strictly stronger bar to avoid
    stopping than requiring *any* (however tiny) improvement, so when
    min_improvement_rate is set it fully replaces (rather than being OR'd
    with) the plain no-improvement-at-all check -- zero improvement is never
    "significant", so the plain check can never fire on its own once a rate
    is set without the rate check also firing."""
    if patience is None:
        return False
    if min_improvement_rate is None:
        return (epoch - best_epoch) >= patience
    # epoch >= len(dev_eer_history): a checkpoint written before dev_eer_history
    # was tracked resumes with a fresh, shorter history no longer aligned to
    # absolute epoch numbers (see the resume tests) -- too little real history
    # to judge, so don't trigger rather than index past it.
    if epoch - patience < 0 or epoch >= len(dev_eer_history):
        return False
    window_start = epoch - patience + 1
    running_best = min(dev_eer_history[:window_start])
    for i in range(window_start, epoch + 1):
        if dev_eer_history[i] <= running_best * (1 - min_improvement_rate):
            return False
        running_best = min(running_best, dev_eer_history[i])
    return True


def train(model, loss_module, dataset, config, dev_utterances=None, segment_length=None,
          draw_strategy=None, run=None, device="cpu", run_idx=None,
          checkpoint_path=None, checkpoint_every_epochs=None, early_stopping_patience=None,
          min_improvement_rate=None, dev_eval_fn=None):
    """Trains for up to config.training.num_epochs. When dev_utterances is
    given, evaluates dev-set SV EER after every epoch, stops once either:
    dev EER hasn't improved at all for early_stopping_patience epochs, or (if
    min_improvement_rate is given) no epoch within the last
    early_stopping_patience epochs improved on the running-best dev EER as of
    just before it by at least that relative rate (see
    _early_stopping_triggered) -- and keeps the best-scoring checkpoint's
    weights on the model at the end of training --
    matches context/src's EvalCallback + get_reference_data, which is what
    Neururer et al. 2024's Tables 1/2 numbers are actually computed from (the
    best dev checkpoint, not whatever state training happens to end in).
    With config.training.paper_checkpoint_epochs set, only the 11 epochs that
    EvalCallback evaluates (see paper_comparable_best below) are candidates.

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
    sequence an uninterrupted run would have produced. The checkpoint also
    carries the full per-epoch dev-EER history, so if the reloaded state
    already implies either early-stopping condition was met before the
    crash, training is not resumed at all -- just the best weights are
    restored -- since that run had already finished.

    Independently of that periodic cadence, best_checkpoint_path(checkpoint_path)
    is (re)written on every dev-EER improvement, so the best weights on disk
    are never more than one epoch stale even if the process is killed between
    two periodic checkpoints.

    Returns (history, paper_comparable_best): history is the per-epoch mean
    training loss; paper_comparable_best is the best dev EER seen at only the
    11 epochs np.linspace(0, config.training.num_epochs-1, 11) selects (or
    None if dev_utterances wasn't given, or none of those epochs have been
    reached yet -- e.g. early stopping fired first). Those 11 fixed positions
    match context/src/utils.py's TEST_EPOCHS default and
    context/src/evaluation/callback.py's EvalCallback, which is what Neururer
    et al. 2024's Tables 1/2 numbers actually search over -- unlike
    best_metric/best_state above, which search every epoch and so read
    optimistic relative to the paper. The schedule is fixed from the
    *configured* num_epochs, not from however many epochs this run actually
    completes, so it stays independent of early stopping's own epoch-by-epoch
    dev-EER signal: early stopping's cutoff is itself a function of where the
    best epoch was (patience-based stopping always ends ~patience epochs
    after it), so computing the linspace over the run's actual length would
    let one of the 11 samples (always the last one) land suspiciously close
    to the true optimum, reintroducing the same kind of search-power bias
    this metric exists to avoid. Fixing the schedule against num_epochs means
    a truncated run simply yields fewer than 11 candidates -- more
    conservative than the paper's search, never more lenient."""
    model.to(device)
    loss_module.to(device)
    num_workers = config.training.num_workers
    loader = DataLoader(
        dataset, batch_size=min(config.training.batch_size, len(dataset)), shuffle=True, drop_last=True,
        num_workers=num_workers, worker_init_fn=_seed_worker_segment_rng if num_workers > 0 else None,
        multiprocessing_context=_fork_context(num_workers), pin_memory=torch.device(device).type == "cuda",
    )
    optimizer = torch.optim.Adam(
        _weight_decay_param_groups(model, loss_module, config.optimizer.weight_decay),
        lr=config.optimizer.learning_rate,
    )

    start_epoch = 0
    best_metric, best_epoch, best_state = None, -1, None
    dev_eer_history = []
    paper_comparable_best = None
    paper_comparable_epochs = frozenset(np.linspace(0, config.training.num_epochs - 1, 11).astype(int).tolist())
    dev_eval_rng = np.random.default_rng()
    metric_prefix = f"run{run_idx}/" if run_idx is not None else ""
    dev_eval_fn = dev_eval_fn or equal_error_rate
    if run_idx is not None:
        tracking.define_repeat_metrics(run, metric_prefix)

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
        # .get(..., []): checkpoints written before this history was tracked
        # won't have the key -- resuming from one just means the
        # min_improvement_rate condition has no history yet and only
        # evaluates once enough post-resume epochs have accumulated.
        dev_eer_history = checkpoint.get("dev_eer_history", [])
        paper_comparable_best = checkpoint.get("paper_comparable_best")
        _restore_rng_state(checkpoint["rng_state"], dataset, dev_eval_rng)

    already_stopped = best_epoch >= 0 and _early_stopping_triggered(
        start_epoch - 1, best_epoch, dev_eer_history, early_stopping_patience, min_improvement_rate
    )

    # With checkpoint selection restricted to the 11 paper epochs and no early
    # stopping (which reads every epoch's dev EER), no other epoch's dev EER
    # can affect the outcome, so only those 11 are evaluated.
    evaluate_every_epoch = not config.training.paper_checkpoint_epochs or early_stopping_patience is not None
    if dev_utterances is not None and not already_stopped:
        # Featurized once (in parallel) and kept in memory, instead of
        # re-reading and re-featurizing every dev utterance at every dev eval.
        started = time.perf_counter()
        dev_utterances = list(resolved_utterances(dev_utterances, num_workers))
        print(f"==> Featurized {len(dev_utterances)} dev utterances in {time.perf_counter() - started:.0f}s")

    # Training batches only (fixed shape thanks to drop_last): CUDA graphs
    # replay each step with a single launch instead of hundreds, so the step
    # no longer waits on a CPU shared with the DataLoader workers. Dev/test
    # evaluation keeps the plain model -- its per-utterance window count
    # varies, which would recompile constantly -- and checkpoints keep saving
    # the plain model's state_dict.
    forward = torch.compile(model, mode="reduce-overhead") if config.training.compile else model

    model.train()
    loss_module.train()
    history = []
    if not already_stopped:
        for epoch in range(start_epoch, config.training.num_epochs):
            epoch_started = time.perf_counter()
            data_wait = 0.0
            # Detached per-batch losses, read back once per epoch: a .item()
            # per batch would make the CPU wait for every GPU step instead of
            # queueing the next one while the GPU works.
            batch_losses = []
            wait_started = time.perf_counter()
            for x, labels in loader:
                data_wait += time.perf_counter() - wait_started
                x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                if config.training.compile:
                    torch.compiler.cudagraph_mark_step_begin()
                output = forward(x)
                loss, _ = loss_module(output.bottleneck, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.detach())
                wait_started = time.perf_counter()
            # Summed as Python floats in batch order, as the per-batch .item() did.
            epoch_loss = sum(torch.stack(batch_losses).cpu().tolist()) if batch_losses else 0.0
            n_batches = len(batch_losses)
            _synchronize(device)
            train_time = time.perf_counter() - epoch_started
            mean_loss = epoch_loss / max(n_batches, 1)
            history.append(mean_loss)
            timings = {
                "time/train_s": train_time,
                "time/data_wait_s": data_wait,
                "time/data_wait_fraction": data_wait / train_time if train_time > 0 else 0.0,
                "throughput/segments_per_s": n_batches * loader.batch_size / train_time if train_time > 0 else 0.0,
            }

            if dev_utterances is not None:
                if evaluate_every_epoch or epoch in paper_comparable_epochs:
                    eval_started = time.perf_counter()
                    dev_embeddings, dev_labels = extract_embeddings(
                        model, dev_utterances, segment_length, draw_strategy, device=device, rng=dev_eval_rng
                    )
                    dev_eer = dev_eval_fn(dev_embeddings, dev_labels)
                    dev_eer_history.append(dev_eer)
                    tracking.log(run, {f"{metric_prefix}epoch": epoch, f"{metric_prefix}dev/EER": dev_eer})
                    if epoch in paper_comparable_epochs and (paper_comparable_best is None or dev_eer < paper_comparable_best):
                        paper_comparable_best = dev_eer
                    is_candidate = not config.training.paper_checkpoint_epochs or epoch in paper_comparable_epochs
                    if is_candidate and (best_metric is None or dev_eer < best_metric):
                        best_metric, best_epoch, best_state = dev_eer, epoch, copy.deepcopy(model.state_dict())
                        if checkpoint_path is not None:
                            _save_best_checkpoint(best_checkpoint_path(checkpoint_path), best_epoch, best_metric, best_state)
                    model.train()
                    timings["time/dev_eval_s"] = time.perf_counter() - eval_started

                stopped_early = _early_stopping_triggered(
                    epoch, best_epoch, dev_eer_history, early_stopping_patience, min_improvement_rate
                )
                should_checkpoint = checkpoint_path is not None and (
                    stopped_early
                    or epoch == config.training.num_epochs - 1
                    or (checkpoint_every_epochs is not None and (epoch + 1) % checkpoint_every_epochs == 0)
                )
                if should_checkpoint:
                    checkpoint_started = time.perf_counter()
                    _save_checkpoint(
                        checkpoint_path, epoch, model, optimizer, loss_module,
                        best_metric, best_epoch, best_state, dev_eer_history, _rng_state(dataset, dev_eval_rng),
                        paper_comparable_best,
                    )
                    timings["time/checkpoint_s"] = time.perf_counter() - checkpoint_started

            tracking.log(run, {
                f"{metric_prefix}epoch": epoch,
                f"{metric_prefix}train/loss": mean_loss,
                **{f"{metric_prefix}{key}": value for key, value in timings.items()},
            })
            print(
                f"==> Epoch {epoch}: loss {mean_loss:.4f}, train {train_time:.0f}s "
                f"(data wait {data_wait:.0f}s = {timings['time/data_wait_fraction']:.0%}, "
                f"{timings['throughput/segments_per_s']:.0f} segments/s)"
                + "".join(f", {key.removeprefix('time/').removesuffix('_s')} {timings[key]:.0f}s"
                          for key in ("time/dev_eval_s", "time/checkpoint_s") if key in timings),
                flush=True,
            )
            if dev_utterances is not None and stopped_early:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, paper_comparable_best


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
    longer (2-8 concatenated sentences).

    BS{b} walks the same sequential windows as OS/SS and permutes each
    window's b-frame blocks fresh (see src/data/segments.py::draw_bs); a
    mixture (MIX, RANDBS, see src/data/segments.py's MIXTURES) treats each
    window by one of its strategies, picked at random per window.

    GO{b}/GS{b} windows instead spread their blocks over the whole utterance
    (src/data/segments.py::gapped_segment), the num_windows windows at
    evenly-spaced positions through its slack. The positions use no
    randomness, so GO{b} and GS{b} take the same blocks in every window and
    differ only in GS{b}'s block order."""
    if draw_strategy == "SU":
        length = features.shape[0]
        return [features[rng.integers(0, length, segment_length)] for _ in range(num_windows)]
    segments = []
    for i in range(num_windows):
        strategy = draw_strategy
        if strategy in MIXTURES:
            strategy = MIXTURES[strategy][rng.integers(len(MIXTURES[strategy]))]
        if strategy.startswith(("GO", "GS")):
            block_size = int(strategy[2:])
            segment = gapped_segment(features, segment_length, block_size, i / max(num_windows - 1, 1))
            if strategy.startswith("GS"):
                segment = shuffle_blocks(segment, block_size, rng)
            segments.append(segment)
            continue
        start = i * step
        segment = features[start:start + segment_length].copy()
        if strategy == "SS":
            rng.shuffle(segment)
        elif strategy.startswith("BS"):
            segment = shuffle_blocks(segment, int(strategy[2:]), rng)
        segments.append(segment)
    return segments


def extract_embeddings(model, utterances, segment_length, draw_strategy, seed=None, device="cpu", hop_fraction=0.5, rng=None,
                       num_workers=0):
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
    calls instead of reseeding every time.

    num_workers > 0 reads and featurizes lazy utterance entries in that many
    worker processes (see resolved_utterances); the result is the same."""
    if rng is None:
        rng = np.random.default_rng(seed)
    return extract_embeddings_by_strategy(
        model, utterances, segment_length, {draw_strategy: rng}, device, hop_fraction, num_workers
    )[draw_strategy]


# Utterances featurized (in memory) per round of extract_embeddings_by_strategy,
# and input frames per forward pass: 2048 windows of 1 s (94 frames), half as
# many of 2 s. Counted in frames because a forward pass's memory grows with
# window length; e.g. a 2 s batch of 2048 windows took 14 GB in the first
# convolution of a Conformer without time subsampling.
EMBEDDING_CHUNK_UTTERANCES = 5000
EMBEDDING_BATCH_FRAMES = 2048 * 94


def extract_embeddings_by_strategy(model, utterances, segment_length, rngs, device="cpu", hop_fraction=0.5, num_workers=0,
                                   log_every=None, resume_dir=None):
    """extract_embeddings for several draw strategies at once -- rngs maps
    each strategy to the Generator its windows are drawn with -- featurizing
    every utterance only once instead of once per strategy. Returns
    {strategy: (embeddings, labels)}, each equal to a separate
    extract_embeddings call with that strategy and Generator. log_every
    prints progress every that many utterances.

    Utterances are featurized EMBEDDING_CHUNK_UTTERANCES at a time, and the
    DataLoader workers have exited before that chunk's forward passes run:
    with workers alive next to eager GPU work, the VoxCeleb1-H evaluation
    slowed to a crawl. The windows of many utterances share a forward pass
    (up to EMBEDDING_BATCH_FRAMES frames; only an utterance with more windows
    than that alone exceeds it) instead of one small pass per utterance and
    strategy; each window is embedded independently (eval mode), so this
    changes at most float rounding.

    resume_dir, when given, keeps every finished chunk's embeddings there,
    with the Generators' states after it: a call interrupted (e.g. a
    requeued job) and made again with the same arguments continues after
    the last finished chunk, with the same result. The caller deletes the
    directory once it no longer needs it."""
    step = max(int(hop_fraction * segment_length), 1)
    batch_windows = max(EMBEDDING_BATCH_FRAMES // segment_length, 1)
    started = time.perf_counter()

    def embed(windows, counts):
        """Per-utterance mean backend embedding, for consecutive utterances
        whose windows (counts[i] each) are concatenated in `windows`."""
        tensor = torch.from_numpy(np.concatenate(windows)).float().unsqueeze(1).to(device)
        backend = model(tensor).backend
        return [part.mean(dim=0).cpu().numpy() for part in torch.split(backend, counts)]

    model.eval()
    embeddings, labels = {strategy: [] for strategy in rngs}, []
    pending = {strategy: ([], []) for strategy in rngs}  # windows, window count per utterance
    if resume_dir is not None:
        resume_dir = Path(resume_dir)
        resume_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for chunk_start in range(0, len(utterances), EMBEDDING_CHUNK_UTTERANCES):
            chunk = utterances[chunk_start:chunk_start + EMBEDDING_CHUNK_UTTERANCES]
            chunk_path = resume_dir / f"chunk{chunk_start}.npz" if resume_dir is not None else None
            if chunk_path is not None and chunk_path.exists():
                with np.load(chunk_path) as saved:
                    labels.extend(saved["labels"])
                    for strategy, rng in rngs.items():
                        embeddings[strategy].extend(saved[f"embeddings_{strategy}"])
                        rng.bit_generator.state = json.loads(str(saved[f"rng_{strategy}"]))
                continue
            chunk_first = {strategy: len(embeddings[strategy]) for strategy in rngs}, len(labels)
            for features, label in list(resolved_utterances(chunk, num_workers)):
                length = features.shape[0]
                if length <= segment_length:
                    continue
                num_windows = (length - segment_length) // step + 1
                for strategy, rng in rngs.items():
                    windows, counts = pending[strategy]
                    if counts and sum(counts) + num_windows > batch_windows:
                        embeddings[strategy].extend(embed(windows, counts))
                        windows, counts = pending[strategy] = ([], [])
                    windows.append(np.stack(_eval_windows(features, segment_length, step, num_windows, strategy, rng)))
                    counts.append(num_windows)
                labels.append(label)
            if chunk_path is not None:
                # Embed what's pending, so the chunk is complete on disk.
                for strategy, (windows, counts) in pending.items():
                    if counts:
                        embeddings[strategy].extend(embed(windows, counts))
                        pending[strategy] = ([], [])
                first_embedding, first_label = chunk_first
                saved = {"labels": np.array(labels[first_label:])}
                for strategy, rng in rngs.items():
                    saved[f"embeddings_{strategy}"] = np.array(embeddings[strategy][first_embedding[strategy]:])
                    saved[f"rng_{strategy}"] = np.array(json.dumps(rng.bit_generator.state))
                # Written under a temporary name first: a chunk file on disk is always complete.
                partial_path = chunk_path.with_suffix(".partial.npz")
                np.savez(partial_path, **saved)
                os.replace(partial_path, chunk_path)
            done = chunk_start + len(chunk)
            if log_every and (done // log_every) > (chunk_start // log_every):
                print(f"==> Embedded {done}/{len(utterances)} utterances ({time.perf_counter() - started:.0f}s)", flush=True)
        for strategy, (windows, counts) in pending.items():
            if counts:
                embeddings[strategy].extend(embed(windows, counts))
    return {strategy: (np.stack(embeddings[strategy]), np.array(labels)) for strategy in rngs}
