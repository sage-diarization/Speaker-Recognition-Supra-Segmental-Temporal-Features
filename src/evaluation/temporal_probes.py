"""Label-free probes of how much a trained model's embeddings depend on the
temporal order of their input (no verification trials needed)."""

import numpy as np
import torch

from ..models.conformer import RelativeMultiHeadSelfAttention
from ..training.trainer import extract_embeddings_by_strategy
from .verification import cosine_similarity


def permutation_sensitivity(model, utterances, segment_length, strategies, device="cpu", seed=0):
    """Mean cosine distance 1 - cos(e_OS, e_S) between each utterance's OS
    embedding and its embedding under every strategy S (e.g. BS{b}). Both
    use the same hop windows (see trainer._eval_windows), so they differ
    only in frame order. Returns {strategy: mean distance}."""
    rngs = {strategy: np.random.default_rng(seed) for strategy in ("OS", *strategies)}
    embeddings = extract_embeddings_by_strategy(model, utterances, segment_length, rngs, device)
    reference = embeddings["OS"][0]
    return {
        strategy: float(np.mean(1.0 - cosine_similarity(reference, embeddings[strategy][0])))
        for strategy in strategies
    }


def attention_distances(model, segments, step_ms):
    """Attention-weighted mean |query - key| distance in ms, per Conformer
    layer and head, averaged over queries and `segments` (batch, 1, time,
    freq). step_ms is the encoder's time step: the frame step times 4 with
    temporal subsampling, the frame step itself without. Returns an array of
    shape (layers, heads)."""
    distances = []

    def hook(module, args):
        x, pos_k = args
        weights = module.attention_weights(x, pos_k)  # (batch, heads, query, key)
        position = torch.arange(x.shape[1], device=x.device)
        offset = (position[:, None] - position[None, :]).abs().float()
        distances.append((weights * offset).sum(-1).mean(dim=(0, 2)).cpu().numpy() * step_ms)

    modules = [m for m in model.modules() if isinstance(m, RelativeMultiHeadSelfAttention)]
    handles = [m.register_forward_pre_hook(hook) for m in modules]
    model.eval()
    try:
        with torch.no_grad():
            model(segments)
    finally:
        for handle in handles:
            handle.remove()
    return np.stack(distances)
