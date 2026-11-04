"""Label-free probes of how much a trained model's embeddings depend on the
temporal order of their input (no verification trials needed)."""

import numpy as np
import torch

from ..models.conformer import RelativeMultiHeadSelfAttention
from ..training.trainer import _eval_windows, extract_embeddings_by_strategy, resolved_utterances
from .verification import cosine_similarity


def permutation_sensitivity(model, utterances, segment_length, strategies, device="cpu", seed=0, gap_ratio=None):
    """Mean cosine distance 1 - cos(e_R, e_S) between each utterance's
    embedding under a reference strategy R and under every strategy S. R is
    OS, except for GS{b}, whose reference is the matched control GO{b}. Each
    pair uses the same windows (see trainer._eval_windows) and differs only
    in frame order; GO{b} and GS{b} also have the same number of joins.
    gap_ratio sets GO{b}/GS{b}'s gap between blocks; utterances too short for it
    are left out of their means. Returns {strategy: mean distance}."""
    references = {strategy: f"GO{strategy[2:]}" if strategy.startswith("GS") else "OS" for strategy in strategies}
    rngs = {strategy: np.random.default_rng(seed) for strategy in (*references.values(), *strategies)}
    embeddings = extract_embeddings_by_strategy(model, utterances, segment_length, rngs, device, gap_ratio=gap_ratio)
    return {
        strategy: float(np.nanmean(1.0 - cosine_similarity(embeddings[references[strategy]][0], embeddings[strategy][0])))
        for strategy in strategies
    }


def join_discontinuity(utterances, segment_length, strategies, seed=0, hop_fraction=0.5, gap_ratio=None):
    """Euclidean distance between the two frames at every join of each
    strategy's evaluation windows (as extract_embeddings would draw them):
    every two consecutive window positions whose frames weren't consecutive
    in the utterance. Also between every two consecutive frames of the OS
    windows ("adjacent", the natural frame-to-frame change). Input-only, no
    model: if GS{b}'s joins are rougher than GO{b}'s, the join artefacts
    don't fully cancel in L(b). Blocks that stay original neighbours make
    no join, and a shorter last block can land anywhere, so joins are found
    by drawing every window a second time from the frame indices (with an
    identically seeded Generator), not at multiples of b. The per-window
    band means of segment normalization cancel in the differences. gap_ratio
    sets GO{b}/GS{b}'s gap between blocks. Returns {strategy: 1-D array of
    distances}."""
    step = max(int(hop_fraction * segment_length), 1)
    rngs = {strategy: (np.random.default_rng(seed), np.random.default_rng(seed)) for strategy in strategies}
    distances = {strategy: [] for strategy in ("adjacent", *strategies)}
    for features, _ in resolved_utterances(utterances):
        if features.shape[0] <= segment_length:
            continue
        num_windows = (features.shape[0] - segment_length) // step + 1
        for window in _eval_windows(features, segment_length, step, num_windows, "OS", None):
            distances["adjacent"].append(np.linalg.norm(np.diff(window, axis=0), axis=1))
        for strategy in strategies:
            distances[strategy].extend(
                _join_distances(features, segment_length, step, num_windows, strategy, rngs[strategy], gap_ratio)
            )
    return {strategy: np.concatenate(values) for strategy, values in distances.items()}


def utterance_join_roughness(utterances, segment_length, strategies, seed=0, hop_fraction=0.5, gap_ratio=None):
    """Mean distance between the two frames at the joins of each
    utterance's evaluation windows (see join_discontinuity), per strategy.
    One value per row of extract_embeddings_by_strategy's embeddings: drawn
    with the same seed (a Generator default_rng(seed) per strategy, as the
    final evaluation draws with default_rng(run_idx)), these are the joins
    of exactly the windows and block orders that were embedded. NaN for an
    utterance without joins, e.g. one too short for GO{b}/GS{b}'s gaps
    (whose embedding is NaN too). Returns ({strategy: 1-D array}, labels),
    labels as extract_embeddings_by_strategy returns them, to check the
    rows line up."""
    step = max(int(hop_fraction * segment_length), 1)
    rngs = {strategy: (np.random.default_rng(seed), np.random.default_rng(seed)) for strategy in strategies}
    roughness, labels = {strategy: [] for strategy in strategies}, []
    for features, label in resolved_utterances(utterances):
        if features.shape[0] <= segment_length:
            continue
        num_windows = (features.shape[0] - segment_length) // step + 1
        for strategy in strategies:
            distances = _join_distances(features, segment_length, step, num_windows, strategy, rngs[strategy], gap_ratio)
            joined = np.concatenate(distances) if distances else np.empty(0)
            roughness[strategy].append(joined.mean() if joined.size else np.nan)
        labels.append(label)
    return {strategy: np.array(values) for strategy, values in roughness.items()}, labels


def _join_distances(features, segment_length, step, num_windows, strategy, rngs, gap_ratio):
    """Per evaluation window of strategy, the distances between the two
    frames at each of its joins. rngs holds two identically seeded
    Generators: one draws the windows from the features, the other the
    same windows from the frame indices, which locate the joins (see
    join_discontinuity)."""
    windows = _eval_windows(features, segment_length, step, num_windows, strategy, rngs[0], gap_ratio)
    frame_indices = np.arange(features.shape[0])[:, None]
    sources = _eval_windows(frame_indices, segment_length, step, num_windows, strategy, rngs[1], gap_ratio)
    distances = []
    for window, source in zip(windows, sources):
        joins = np.flatnonzero(np.diff(source[:, 0]) != 1) + 1
        distances.append(np.linalg.norm(window[joins] - window[joins - 1], axis=1))
    return distances


def attention_offset_profiles(model, segments):
    """Distribution of attention weight over |query - key| offset, per
    Conformer layer and head, averaged over queries and `segments` (batch, 1,
    time, freq). Returns an array of shape (layers, heads, encoder steps):
    entry d is the mean weight a query puts on keys d encoder steps away, so
    each (layer, head) row sums to 1."""
    profiles = []

    def hook(module, args):
        x, pos_k = args
        weights = module.attention_weights(x, pos_k)  # (batch, heads, query, key)
        position = torch.arange(x.shape[1], device=x.device)
        offset = (position[:, None] - position[None, :]).abs()
        one_hot = torch.nn.functional.one_hot(offset, x.shape[1]).float()  # (query, key, offset)
        profile = torch.einsum("bhqk,qkd->hd", weights, one_hot) / (x.shape[0] * x.shape[1])
        profiles.append(profile.cpu().numpy())

    modules = [m for m in model.modules() if isinstance(m, RelativeMultiHeadSelfAttention)]
    handles = [m.register_forward_pre_hook(hook) for m in modules]
    model.eval()
    try:
        with torch.no_grad():
            model(segments)
    finally:
        for handle in handles:
            handle.remove()
    return np.stack(profiles)


def attention_distances(model, segments, step_ms):
    """Attention-weighted mean |query - key| distance in ms, per Conformer
    layer and head, averaged over queries and `segments` (batch, 1, time,
    freq). step_ms is the encoder's time step: the frame step times 4 with
    temporal subsampling, the frame step itself without. Returns an array of
    shape (layers, heads)."""
    profiles = attention_offset_profiles(model, segments)
    return profiles @ np.arange(profiles.shape[-1]) * step_ms
