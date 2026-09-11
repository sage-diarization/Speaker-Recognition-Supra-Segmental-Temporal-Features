import torch
import torch.nn as nn

from .common import BackendOutput


class RNNBackend(nn.Module):
    """PyTorch port of context/src/models/backend/LSTM.py: two stacked
    bidirectional LSTMs (the second returning only its final hidden state)
    followed by the same Dense(1024)->Dropout->Dense(512) head CNNBackend
    uses, matching Neururer et al. 2024's RNN [13]."""

    def __init__(self, num_freqs, hidden_size=512):
        super().__init__()
        self.lstm1 = nn.LSTM(num_freqs, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)

        self.backend_head = nn.Linear(2 * hidden_size, 1024)
        self.bottleneck_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        x = x.squeeze(1)  # (batch, 1, T, F) -> (batch, T, F)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        _, (h_n, _) = self.lstm2(x)
        # h_n: (num_directions=2, batch, hidden_size) -- concat forward/backward
        # final hidden states, matching Keras Bidirectional(..., return_sequences=False)'s
        # default merge_mode='concat'.
        backend = torch.cat([h_n[0], h_n[1]], dim=-1)
        bottleneck = self.bottleneck_head(self.backend_head(backend))
        return BackendOutput(backend, bottleneck)


def build_rnn(config):
    return RNNBackend(config.transformation.n_mels, config.rnn.hidden_size)
