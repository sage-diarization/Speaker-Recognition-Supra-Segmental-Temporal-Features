import torch
import torch.nn as nn

from .common import KERAS_BATCH_NORM, BackendOutput


class CNNBackend(nn.Module):
    """PyTorch port of context/src/models/backend/CNN.py."""

    def __init__(self, segment_length, num_freqs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.ReLU(),
            nn.BatchNorm2d(32, **KERAS_BATCH_NORM),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.BatchNorm2d(64, **KERAS_BATCH_NORM),
            nn.MaxPool2d(kernel_size=4, stride=2),
        )

        # AdaptiveAvgPool2d target is traced once from the configured
        # segment_length so ALLOW_FULL (variable T at eval time) still
        # flattens to a fixed size instead of breaking the Linear layer.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, segment_length, num_freqs)
            traced_shape = self.conv(dummy).shape[-2:]
        self.pool = nn.AdaptiveAvgPool2d(tuple(traced_shape))
        flat_dim = 64 * traced_shape[0] * traced_shape[1]

        self.flatten = nn.Flatten()
        self.backend_head = nn.Linear(flat_dim, 1024)

        self.bottleneck_head = nn.Sequential(
            nn.BatchNorm1d(1024, **KERAS_BATCH_NORM),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
        )

        # Keras Conv2D/Dense defaults: glorot_uniform kernels, zero biases
        # (torch defaults to kaiming_uniform(a=sqrt(5)) + uniform biases).
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        backend = self.backend_head(x)
        bottleneck = self.bottleneck_head(backend)
        return BackendOutput(backend, bottleneck)


def build_cnn(config):
    segment_length = config.data.segment_length(config.transformation)
    return CNNBackend(segment_length, config.transformation.n_mels)
