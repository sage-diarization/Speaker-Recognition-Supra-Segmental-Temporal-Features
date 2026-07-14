import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.segments import DRAW_STRATEGIES


def train(model, loss_module, dataset, config, device="cpu"):
    model.to(device)
    loss_module.to(device)
    loader = DataLoader(dataset, batch_size=min(config.training.batch_size, len(dataset)), shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=config.optimizer.learning_rate,
    )

    model.train()
    loss_module.train()
    history = []
    for _ in range(config.training.num_epochs):
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
        history.append(epoch_loss / max(n_batches, 1))
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
