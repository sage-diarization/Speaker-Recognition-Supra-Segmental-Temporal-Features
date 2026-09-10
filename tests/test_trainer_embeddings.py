import numpy as np
import pytest

from src.models.common import BackendOutput
from src.training import trainer
from src.training.trainer import extract_embeddings


class _MeanModel:
    """Stand-in model whose backend embedding is just the mean value of its
    input segment (broadcast to a length-1 vector) -- lets us verify
    extract_embeddings' hop-window averaging arithmetic exactly, without
    depending on a real (nonlinear) backend's numbers."""

    def eval(self):
        pass

    def to(self, device):
        return self

    def __call__(self, x):
        backend = x.mean(dim=(1, 2, 3))
        return BackendOutput(backend=backend.unsqueeze(-1), bottleneck=None)


def _fake_draw_returning(values):
    values = iter(values)

    def draw(features, segment_length, rng):
        return np.full((segment_length, features.shape[1]), float(next(values)))

    return draw


def test_extract_embeddings_averages_all_hop_windows(monkeypatch):
    """context/src/generator/generator.py:399 averages the embeddings of
    every hop-window segment drawn from an utterance (step = 0.5 *
    segment_length, i.e. 'H50') into one per-utterance embedding, instead of
    embedding a single random segment. This locks in both the hop-window
    count formula (matching context/src/setup/setup.py's
    `while (current_start + segment_length) <= sample_length` loop) and the
    averaging itself."""
    segment_length = 10
    utterance_length = 40  # windows start at 0,5,10,15,20,25,30 -> 7 windows

    monkeypatch.setattr(
        trainer, "DRAW_STRATEGIES", {"OS": _fake_draw_returning(range(7))}
    )

    features = np.zeros((utterance_length, 3))
    embeddings, labels = extract_embeddings(_MeanModel(), [(features, 5)], segment_length, "OS")

    np.testing.assert_allclose(embeddings[0, 0], np.mean(range(7)))
    assert labels[0] == 5


def test_extract_embeddings_uses_single_window_for_short_utterance(monkeypatch):
    """An utterance only slightly longer than segment_length has just one
    hop window, matching the original loop's guaranteed first iteration --
    same behavior as before the hop-averaging fix."""
    segment_length = 10
    monkeypatch.setattr(trainer, "DRAW_STRATEGIES", {"OS": _fake_draw_returning([42.0])})

    features = np.zeros((segment_length + 1, 3))
    embeddings, _ = extract_embeddings(_MeanModel(), [(features, 0)], segment_length, "OS")

    assert embeddings[0, 0] == pytest.approx(42.0)


def test_extract_embeddings_hop_fraction_controls_window_count(monkeypatch):
    """hop_fraction=1.0 ('H100', non-overlapping windows) should draw fewer,
    non-overlapping windows than the default 0.5 ('H50')."""
    segment_length = 10
    utterance_length = 40

    monkeypatch.setattr(trainer, "DRAW_STRATEGIES", {"OS": _fake_draw_returning(range(100))})
    features = np.zeros((utterance_length, 3))

    embeddings, _ = extract_embeddings(
        _MeanModel(), [(features, 0)], segment_length, "OS", hop_fraction=1.0
    )
    # windows at 0,10,20,30 -> 4 windows, values 0..3
    np.testing.assert_allclose(embeddings[0, 0], np.mean(range(4)))
