import pytest
import torch

from src.config import ExperimentConfig
from src.models.resnet import GhostVladPooling, ResNetBackend, build_resnet

# 64 is the smallest num_freqs that survives ResNet34s's stem maxpool + three
# stride-2 stages without a spatial dimension collapsing below what the next
# layer needs (see test_num_freqs_too_small_raises below).
NUM_FREQS = 64


def _tiny_model(**overrides):
    kwargs = dict(num_freqs=NUM_FREQS, vlad_clusters=4, ghost_clusters=2, bottleneck_dim=16)
    kwargs.update(overrides)
    return ResNetBackend(**kwargs)


def test_forward_pass_shapes():
    model = _tiny_model()
    x = torch.randn(3, 1, 32, NUM_FREQS)
    out = model(x)
    assert out.backend.shape == (3, 16)
    assert out.bottleneck.shape == (3, 16)


def test_backend_and_bottleneck_are_the_same_tensor():
    # Unlike CNN/RNN/Conformer, ResNet34s.py's original CUT setting is
    # 'AGGREGATION': the SV/SC eval embedding is the GhostVLAD/loss-facing
    # embedding itself, not a separate larger pre-loss layer.
    model = _tiny_model()
    out = model(torch.randn(2, 1, 32, NUM_FREQS))
    assert out.backend is out.bottleneck


def test_forward_pass_batch_size_one():
    model = _tiny_model()
    model.eval()
    x = torch.randn(1, 1, 32, NUM_FREQS)
    out = model(x)
    assert out.backend.shape == (1, 16)


def test_variable_time_length_still_produces_fixed_size_output():
    model = _tiny_model()
    model.eval()
    for time_steps in (16, 32, 64):
        out = model(torch.randn(2, 1, time_steps, NUM_FREQS))
        assert out.backend.shape == (2, 16)


def test_gradients_flow_to_backbone_stem_weights():
    model = _tiny_model()
    x = torch.randn(4, 1, 32, NUM_FREQS)
    out = model(x)
    loss = out.bottleneck.sum()
    loss.backward()

    first_conv = model.backbone.stem[0]
    assert first_conv.weight.grad is not None
    assert torch.any(first_conv.weight.grad != 0)


def test_num_freqs_too_small_raises():
    with pytest.raises(RuntimeError):
        _tiny_model(num_freqs=8)


def test_build_resnet_reads_hyperparameters_from_config():
    config = ExperimentConfig()
    config.transformation.type = "linear"
    config.transformation.nfft = 126  # num_freqs = 126 // 2 + 1 = 64
    config.resnet.vlad_clusters = 4
    config.resnet.ghost_clusters = 2
    config.resnet.bottleneck = 16

    model = build_resnet(config)
    assert isinstance(model, ResNetBackend)

    out = model(torch.randn(2, 1, 32, config.transformation.num_freqs))
    assert out.backend.shape == (2, 16)


def test_ghost_vlad_pooling_output_shape_and_l2_normalized():
    pooling = GhostVladPooling(k_centers=4, g_centers=2, feature_dim=8)
    feat = torch.randn(3, 8, 1, 10)
    cluster_score = torch.randn(3, 4 + 2, 1, 10)

    out = pooling(feat, cluster_score)
    assert out.shape == (3, 4 * 8)

    norms = out.reshape(3, 4, 8).norm(p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))
