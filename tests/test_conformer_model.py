import numpy as np
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


def test_without_temporal_context_the_backend_ignores_frame_order():
    model = _tiny_model(conv_kernel_size=1, use_relative_positional_encoding=False, subsampling_time_kernel=1,
                         subsampling_time_stride=1)
    model.eval()
    x = torch.randn(2, 1, 32, 20)
    shuffled = x[:, :, torch.randperm(32)]
    torch.testing.assert_close(model(shuffled).backend, model(x).backend, rtol=1e-4, atol=1e-5)


def test_default_backend_depends_on_frame_order():
    model = _tiny_model()
    model.eval()
    x = torch.randn(2, 1, 32, 20)
    assert not torch.allclose(model(x[:, :, torch.randperm(32)]).backend, model(x).backend, atol=1e-5)


def test_build_conformer_passes_subsampling_time_kernel_and_stride():
    config = ExperimentConfig()
    config.conformer.subsampling_time_kernel = 5
    config.conformer.subsampling_time_stride = 1
    conv = build_conformer(config).subsampling.conv[0]
    assert (conv.kernel_size, conv.stride, conv.padding) == ((5, 3), (1, 2), (2, 1))


@pytest.mark.parametrize("time_kernel, time_stride, expected_steps", [(3, 2, 8), (1, 2, 8), (3, 1, 32), (1, 1, 32)])
def test_subsampling_time_stride_sets_the_encoder_length(time_kernel, time_stride, expected_steps):
    model = _tiny_model(subsampling_time_kernel=time_kernel, subsampling_time_stride=time_stride)
    assert model.subsampling(torch.randn(2, 1, 32, 20)).shape == (2, expected_steps, 16)


def test_even_subsampling_time_kernel_is_rejected():
    with pytest.raises(ValueError):
        _tiny_model(subsampling_time_kernel=2)


def test_attention_distances_has_one_value_per_layer_and_head():
    from src.evaluation.temporal_probes import attention_distances

    model = _tiny_model()
    distances = attention_distances(model, torch.randn(2, 1, 32, 20), step_ms=40)
    assert distances.shape == (2, 2)
    # 8 encoder steps: offsets range 0..7 steps.
    assert np.all((distances >= 0) & (distances <= 7 * 40))
