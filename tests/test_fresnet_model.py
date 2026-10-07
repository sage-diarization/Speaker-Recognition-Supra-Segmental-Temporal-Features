import torch

from src.models.fresnet import FResNetBackend

NUM_FREQS = 40
NUM_FRAMES = 100


def test_forward_pass_shapes():
    model = FResNetBackend(num_freqs=NUM_FREQS, bottleneck_dim=16)
    out = model(torch.randn(3, 1, NUM_FRAMES, NUM_FREQS))
    assert out.backend.shape == (3, 16)
    assert out.backend is out.bottleneck


def test_encoder_runs_on_original_freq_time_layout():
    # clovaai's ResNetSE34L feeds (batch, 1, n_mels, time): the stem's stride
    # (2, 1) and layer2/3's (2, 2) downsample frequency 8x and time 4x, and
    # SAP then attends over time frames. This repo's datasets yield
    # (batch, 1, time, freq), so the backend must swap the axes first.
    model = FResNetBackend(num_freqs=NUM_FREQS, bottleneck_dim=16)
    shapes = []
    model.layer4.register_forward_hook(lambda _m, _i, out: shapes.append(out.shape))
    model(torch.randn(2, 1, NUM_FRAMES, NUM_FREQS))
    assert shapes[0][-2:] == (NUM_FREQS // 8, NUM_FRAMES // 4)
