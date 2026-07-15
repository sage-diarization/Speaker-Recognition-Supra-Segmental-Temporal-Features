import pytest
import torch

from src.config import ExperimentConfig
from src.models.conformer import ConformerBackend, build_conformer


def _tiny_model(**overrides):
    kwargs = dict(
        num_freqs=20,
        encoder_dim=16,
        num_layers=2,
        num_heads=2,
        ff_expansion_factor=2,
        conv_kernel_size=3,
        dropout=0.1,
        use_relative_positional_encoding=True,
    )
    kwargs.update(overrides)
    return ConformerBackend(**kwargs)


def test_forward_pass_shapes():
    model = _tiny_model()
    x = torch.randn(3, 1, 32, 20)
    out = model(x)
    assert out.backend.shape == (3, 1024)
    assert out.bottleneck.shape == (3, 512)


def test_forward_pass_batch_size_one():
    model = _tiny_model()
    model.eval()
    x = torch.randn(1, 1, 32, 20)
    out = model(x)
    assert out.backend.shape == (1, 1024)
    assert out.bottleneck.shape == (1, 512)


def test_variable_time_length_still_produces_fixed_size_output():
    # Unlike CNNBackend's AdaptiveAvgPool2d trace, nothing here is sized to a
    # specific segment_length at construction time -- the encoder attends
    # over whatever time axis it's given and the backend mean-pools over it.
    model = _tiny_model()
    model.eval()
    for time_steps in (16, 32, 64):
        out = model(torch.randn(2, 1, time_steps, 20))
        assert out.backend.shape == (2, 1024)
        assert out.bottleneck.shape == (2, 512)


def test_gradients_flow_to_subsampling_conv_weights():
    model = _tiny_model()
    x = torch.randn(4, 1, 32, 20)
    out = model(x)
    loss = out.bottleneck.sum()
    loss.backward()

    first_conv = model.subsampling.conv[0]
    assert first_conv.weight.grad is not None
    assert torch.any(first_conv.weight.grad != 0)


def test_relative_positional_encoding_can_be_disabled():
    model = _tiny_model(use_relative_positional_encoding=False)
    assert model.encoder.positional_encoding is None

    out = model(torch.randn(2, 1, 32, 20))
    assert out.backend.shape == (2, 1024)
    assert out.bottleneck.shape == (2, 512)


def test_conv_kernel_size_must_be_odd_for_same_padding():
    with pytest.raises(AssertionError):
        _tiny_model(conv_kernel_size=4)


def test_encoder_dim_must_be_divisible_by_num_heads():
    with pytest.raises(AssertionError):
        _tiny_model(encoder_dim=15, num_heads=4)


def test_build_conformer_reads_hyperparameters_from_config():
    config = ExperimentConfig()
    config.transformation.n_mels = 20
    config.conformer.encoder_dim = 16
    config.conformer.num_layers = 2
    config.conformer.num_heads = 2
    config.conformer.ff_expansion_factor = 2
    config.conformer.conv_kernel_size = 3

    model = build_conformer(config)
    assert isinstance(model, ConformerBackend)

    out = model(torch.randn(2, 1, 32, 20))
    assert out.backend.shape == (2, 1024)
    assert out.bottleneck.shape == (2, 512)
