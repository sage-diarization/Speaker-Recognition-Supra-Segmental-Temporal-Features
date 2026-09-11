import torch

from src.config import ExperimentConfig
from src.models.rnn import RNNBackend, build_rnn


def test_forward_pass_shapes():
    model = RNNBackend(num_freqs=20, hidden_size=8)
    x = torch.randn(3, 1, 10, 20)
    out = model(x)
    assert out.backend.shape == (3, 16)  # 2 * hidden_size (bidirectional concat)
    assert out.bottleneck.shape == (3, 512)


def test_forward_pass_batch_size_one():
    model = RNNBackend(num_freqs=20, hidden_size=8)
    model.eval()
    x = torch.randn(1, 1, 10, 20)
    out = model(x)
    assert out.backend.shape == (1, 16)
    assert out.bottleneck.shape == (1, 512)


def test_variable_time_length_still_produces_fixed_size_output():
    # LSTMs process any time length directly; nothing here is traced to a
    # specific segment_length at construction time (unlike CNNBackend).
    model = RNNBackend(num_freqs=20, hidden_size=8)
    model.eval()
    for time_steps in (5, 10, 40):
        out = model(torch.randn(2, 1, time_steps, 20))
        assert out.backend.shape == (2, 16)
        assert out.bottleneck.shape == (2, 512)


def test_gradients_flow_to_lstm_weights():
    model = RNNBackend(num_freqs=20, hidden_size=8)
    x = torch.randn(4, 1, 10, 20)
    out = model(x)
    loss = out.bottleneck.sum()
    loss.backward()

    assert model.lstm1.weight_ih_l0.grad is not None
    assert torch.any(model.lstm1.weight_ih_l0.grad != 0)


def test_build_rnn_reads_hyperparameters_from_config():
    config = ExperimentConfig()
    config.transformation.n_mels = 20
    config.rnn.hidden_size = 8

    model = build_rnn(config)
    assert isinstance(model, RNNBackend)

    out = model(torch.randn(2, 1, 10, 20))
    assert out.backend.shape == (2, 16)
    assert out.bottleneck.shape == (2, 512)
