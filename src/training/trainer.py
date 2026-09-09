import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import tracking
from ..data.segments import DRAW_STRATEGIES
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


def extract_embeddings(model, utterances, segment_length, draw_strategy, seed=None, device="cpu"):
    """Draws one segment per utterance (via the given strategy) and returns
    the model's backend (1024-d) embedding for each, plus their labels."""
    draw_fn = DRAW_STRATEGIES[draw_strategy]
    rng = np.random.default_rng(seed)

    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for features, label in utterances:
            if features.shape[0] <= segment_length:
                continue
            segment = draw_fn(features, segment_length, rng)
            tensor = torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor)
            embeddings.append(output.backend.squeeze(0).cpu().numpy())
            labels.append(label)
    return np.stack(embeddings), np.array(labels)
