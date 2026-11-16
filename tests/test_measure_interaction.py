import numpy as np
import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from src.measure_interaction import PooledFrames, embed_levels, interaction_ratios, interaction_windows
from src.models.pretrained_w2vbert import W2VBertSV
from src.models.pretrained_wespeaker import WeSpeakerResNet34


def test_interaction_windows_patch_i_j_and_both_with_the_donors_frames():
    window = np.zeros((20, 3), dtype=np.float32)
    donor = np.ones((20, 3), dtype=np.float32)
    windows = interaction_windows(window, donor, [6], 2, 1, np.random.default_rng(0))
    assert windows.shape == (4, 20, 3)
    np.testing.assert_array_equal(windows[0], window)
    i = int(np.flatnonzero(windows[1][:, 0])[0])
    np.testing.assert_array_equal(np.flatnonzero(windows[1][:, 0]), [i, i + 1])
    np.testing.assert_array_equal(np.flatnonzero(windows[2][:, 0]), [i + 6, i + 7])
    np.testing.assert_array_equal(np.flatnonzero(windows[3][:, 0]), [i, i + 1, i + 6, i + 7])


def test_interaction_windows_reject_overlapping_or_unfitting_patches():
    window = np.zeros((20, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        interaction_windows(window, window, [1], 2, 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        interaction_windows(window, window, [19], 2, 1, np.random.default_rng(0))


def _outputs(windows, f):
    return np.stack([f(w) for w in windows])


def test_interaction_ratios_are_zero_for_an_additive_output_and_positive_for_a_product():
    rng = np.random.default_rng(0)
    window, donor = rng.standard_normal((2, 30, 4))
    windows = interaction_windows(window, donor, [5, 10, 20], 3, 2, rng)
    additive = _outputs(windows, lambda w: np.tanh(w).sum(axis=0))
    ratios, floor = interaction_ratios(additive, 3, 2, additive[0])
    np.testing.assert_allclose(ratios, 0, atol=1e-12)
    assert floor == 0
    # Every frame times every other: patches interact at any distance.
    product = _outputs(windows, lambda w: np.outer(w.sum(axis=0), w.sum(axis=0)).ravel())
    ratios, _ = interaction_ratios(product, 3, 2, product[0])
    assert (ratios > 0.1).all()


def _tiny_w2vbert_sv():
    config = Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        output_hidden_size=32, conv_depthwise_kernel_size=3,
    )
    return W2VBertSV(Wav2Vec2BertModel(config), adapter_dim=8, embed_dim=16).eval()


@pytest.mark.parametrize("model, features, dim", [
    (WeSpeakerResNet34().eval(), 80, 256 * 10),
    (_tiny_w2vbert_sv(), 160, 8 * 3),
])
def test_pooled_frames_capture_the_mean_of_the_pooled_frame_features(model, features, dim):
    pooled = PooledFrames(model)
    windows = np.random.default_rng(0).standard_normal((5, 24, features)).astype(np.float32)
    outputs = embed_levels(model, pooled, windows, 2, "cpu")
    pooled.remove()
    assert outputs["frames"].shape == (5, dim)
    assert outputs["embedding"].shape[0] == 5
    # The hook is gone: another pass leaves the captured value as it was.
    last = pooled.value
    with torch.no_grad():
        model(torch.from_numpy(windows[:1]).unsqueeze(1))
    assert pooled.value is last


def test_pooled_frames_reject_an_unknown_model():
    with pytest.raises(ValueError):
        PooledFrames(torch.nn.Linear(2, 2))
