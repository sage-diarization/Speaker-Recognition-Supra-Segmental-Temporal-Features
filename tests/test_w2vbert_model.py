import numpy as np
import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from src.config import TransformationConfig
from src.data.dataset import featurize_waveform
from src.data.lazy_features import expected_frame_count
from src.models.w2vbert import W2VBertBackend


def _tiny_encoder():
    # Randomly initialized, same architecture family as facebook/w2v-bert-2.0,
    # so these tests need no download.
    config = Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        output_hidden_size=32, conv_depthwise_kernel_size=3,
    )
    return Wav2Vec2BertModel(config)


def test_forward_returns_backend_and_bottleneck_for_any_segment_length():
    model = W2VBertBackend(_tiny_encoder()).eval()
    for frames in (49, 100):
        output = model(torch.randn(4, 1, frames, 160))
        assert output.backend.shape == (4, 1024)
        assert output.bottleneck.shape == (4, 512)


def test_frozen_encoder_stays_out_of_state_dict_optimizer_and_train_mode():
    model = W2VBertBackend(_tiny_encoder())
    assert not any(key.startswith("encoder") for key in model.state_dict())
    assert {name for name, _ in model.named_parameters()} == {
        "layer_weights", "backend_head.weight", "backend_head.bias",
        "bottleneck_head.0.weight", "bottleneck_head.0.bias",
        "bottleneck_head.2.weight", "bottleneck_head.2.bias",
    }
    model.train()
    assert not model.encoder.training

    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    layer_weights_before = model.layer_weights.detach().clone()
    model(torch.randn(4, 1, 49, 160)).bottleneck.sum().backward()
    optimizer.step()

    assert not torch.equal(model.layer_weights, layer_weights_before)
    for key, value in model.encoder.state_dict().items():
        assert torch.equal(value, encoder_before[key])


def test_to_moves_the_encoder_too():
    model = W2VBertBackend(_tiny_encoder()).to(torch.float64)
    assert all(p.dtype == torch.float64 for p in model.encoder.parameters())


def _w2vbert_transformation():
    return TransformationConfig(type="w2vbert", frame_length_s=0.025, frame_step_s=0.01)


@pytest.mark.parametrize("num_samples", [16000, 16159, 16400, 23456])
def test_featurize_waveform_frame_count_matches_expected_frame_count(num_samples):
    transformation = _w2vbert_transformation()
    waveform = (np.random.default_rng(0).standard_normal(num_samples) * 0.1).astype(np.float32)

    features = featurize_waveform(waveform, transformation)

    assert features.shape == (expected_frame_count(num_samples, transformation), transformation.num_freqs)
    if num_samples == 16000:
        assert features.shape[0] == transformation.steps_per_second


def test_w2vbert_transformation_rejects_other_frame_parameters():
    with pytest.raises(ValueError, match="w2vbert"):
        TransformationConfig(type="w2vbert")
