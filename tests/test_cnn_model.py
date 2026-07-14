import torch

from src.models.cnn import CNNBackend


def test_forward_pass_shapes():
    model = CNNBackend(segment_length=94, num_freqs=128)
    x = torch.randn(3, 1, 94, 128)
    out = model(x)
    assert out.backend.shape == (3, 1024)
    assert out.bottleneck.shape == (3, 512)


def test_forward_pass_batch_size_one():
    model = CNNBackend(segment_length=94, num_freqs=128)
    model.eval()
    x = torch.randn(1, 1, 94, 128)
    out = model(x)
    assert out.backend.shape == (1, 1024)
    assert out.bottleneck.shape == (1, 512)


def test_variable_time_length_still_flattens_to_fixed_size():
    model = CNNBackend(segment_length=94, num_freqs=128)
    model.eval()
    x = torch.randn(2, 1, 188, 128)
    out = model(x)
    assert out.backend.shape == (2, 1024)
    assert out.bottleneck.shape == (2, 512)


def test_gradients_flow_to_conv_weights():
    model = CNNBackend(segment_length=94, num_freqs=128)
    x = torch.randn(4, 1, 94, 128)
    out = model(x)
    loss = out.bottleneck.sum()
    loss.backward()

    first_conv = model.conv[0]
    assert first_conv.weight.grad is not None
    assert torch.any(first_conv.weight.grad != 0)
