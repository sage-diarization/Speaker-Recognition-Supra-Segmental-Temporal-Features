import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import tracking
from ..evaluation.verification import equal_error_rate


def _checkpoint_epochs(num_epochs):
    """11 evenly spaced epochs across training at which to evaluate a dev
    checkpoint, matching context/src/utils.py's TEST_EPOCHS default
    (np.linspace(0, NUM_EPOCHS-1, 11))."""
    return set(np.linspace(0, num_epochs - 1, 11).astype(int).tolist())


def train(model, loss_module, dataset, config, dev_utterances=None, segment_length=None,
          draw_strategy=None, run=None, device="cpu"):
    """Trains for config.training.num_epochs. When dev_utterances is given,
    periodically evaluates dev-set SV EER and keeps the best-scoring
    checkpoint's weights on the model at the end of training -- matches
    context/src's EvalCallback + get_reference_data, which is what Neururer
    et al. 2024's Tables 1/2 numbers are actually computed from (the best dev
    checkpoint, not whatever state training happens to end in)."""
    model.to(device)
    loss_module.to(device)
    loader = DataLoader(dataset, batch_size=min(config.training.batch_size, len(dataset)), shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    checkpoint_epochs = _checkpoint_epochs(config.training.num_epochs) if dev_utterances is not None else set()
    best_eer, best_state = None, None

    model.train()
    loss_module.train()
    history = []
    for epoch in range(config.training.num_epochs):
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
        tracking.log(run, {"train/loss": mean_loss}, step=epoch)

        if epoch in checkpoint_epochs:
            dev_embeddings, dev_labels = extract_embeddings(
                model, dev_utterances, segment_length, draw_strategy, device=device
            )
            dev_eer = equal_error_rate(dev_embeddings, dev_labels)
            tracking.log(run, {"dev/EER": dev_eer}, step=epoch)
            if best_eer is None or dev_eer < best_eer:
                best_eer, best_state = dev_eer, copy.deepcopy(model.state_dict())
            model.train()

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


def extract_embeddings(model, utterances, segment_length, draw_strategy, seed=None, device="cpu", hop_fraction=0.5):
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
    loop, which always runs at least once."""
    rng = np.random.default_rng(seed)
    step = max(int(hop_fraction * segment_length), 1)

    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for features, label in utterances:
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
