"""PyTorch Conformer speaker-embedding backend, following the encoder
architecture of Gulati et al. 2020 ("Conformer: Convolution-augmented
Transformer for Speech Recognition", see context/docs/). The relative
positional multi-head attention and conformer block structure are adapted
from https://github.com/BUTSpeechFIT/DiariZen/blob/main/diarizen/models/module/conformer.py.

Unlike the ASR encoder in the paper (which feeds a decoder per time step),
this backend mean-pools the encoder output over time to produce a
fixed-size speaker embedding, matching CNNBackend's (backend=1024-d,
bottleneck=512-d) contract (see src/models/common.py).
"""

import torch
import torch.nn as nn

from .common import BackendOutput


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class RelativePositionalEncoding(nn.Module):
    """Transformer-XL-style relative positional embedding table: a learned
    per-head-dim vector for each relative offset (i - j) between query
    position i and key position j, clamped to +/-max_relative_position."""

    def __init__(self, dim, max_relative_position=1000):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embedding = nn.Embedding(2 * max_relative_position, dim)

    def forward(self, seq_len, device):
        position = torch.arange(seq_len, device=device)
        relative_position = (position[:, None] - position[None, :]).clamp(
            -self.max_relative_position, self.max_relative_position - 1
        )
        return self.embedding(relative_position + self.max_relative_position)


class RelativeMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with an additive relative-position score
    term (Gulati et al. 2020 Section 2.1)."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        assert dim % num_heads == 0, "encoder_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_k=None):
        batch, seq_len, _ = x.shape
        q = self.query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", q, k)
        if pos_k is not None:
            scores = scores + torch.einsum("bhid,ijd->bhij", q, pos_k)
        scores = scores / (self.head_dim**0.5)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhij,bhjd->bhid", weights, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out(out)


class ConformerSelfAttentionModule(nn.Module):
    """Pre-norm MHSA with residual connection and dropout (Figure 3)."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = RelativeMultiHeadSelfAttention(dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_k=None):
        return x + self.dropout(self.attention(self.norm(x), pos_k))


class ConformerFeedForward(nn.Module):
    """Pre-norm feed-forward module with a half-step residual (Figure 4 /
    the macaron-style 1/2 FFN(x) terms in Eq. 1)."""

    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerConvolutionModule(nn.Module):
    """Gating + depthwise convolution module (Figure 2)."""

    def __init__(self, dim, kernel_size, dropout):
        super().__init__()
        assert kernel_size % 2 == 1, "conv_kernel_size must be odd for SAME padding"
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size, padding=(kernel_size - 1) // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, dim)
        residual = x
        x = self.norm(x).transpose(1, 2)  # (batch, dim, time)
        x = self.glu(self.pointwise_conv1(x))
        x = self.activation(self.batch_norm(self.depthwise_conv(x)))
        x = self.dropout(self.pointwise_conv2(x))
        return residual + x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """FFN -> MHSA -> Conv -> FFN -> LayerNorm (Eq. 1)."""

    def __init__(self, dim, num_heads, ffn_hidden_dim, conv_kernel_size, dropout):
        super().__init__()
        self.ffn1 = ConformerFeedForward(dim, ffn_hidden_dim, dropout)
        self.self_attention = ConformerSelfAttentionModule(dim, num_heads, dropout)
        self.conv = ConformerConvolutionModule(dim, conv_kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(dim, ffn_hidden_dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos_k=None):
        x = self.ffn1(x)
        x = self.self_attention(x, pos_k)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads, ffn_hidden_dim, conv_kernel_size, dropout, use_relative_positional_encoding):
        super().__init__()
        self.positional_encoding = (
            RelativePositionalEncoding(dim // num_heads) if use_relative_positional_encoding else None
        )
        self.blocks = nn.ModuleList(
            [ConformerBlock(dim, num_heads, ffn_hidden_dim, conv_kernel_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        pos_k = self.positional_encoding(x.shape[1], x.device) if self.positional_encoding is not None else None
        for block in self.blocks:
            x = block(x, pos_k)
        return x


class ConvSubsampling(nn.Module):
    """Halves the time and frequency resolution twice ("Convolution
    Subsampling" in Figure 1), then projects the flattened channel*frequency
    features to encoder_dim. Only num_freqs (not segment length) is needed:
    the encoder that follows operates on a variable-length time axis, and
    the backend pools over time, so no segment_length-dependent fixed size
    is baked in (contrast CNNBackend's AdaptiveAvgPool2d trace)."""

    def __init__(self, num_freqs, encoder_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 8, num_freqs)
            subsampled_freq = self.conv(dummy).shape[-1]
        self.projection = nn.Linear(encoder_dim * subsampled_freq, encoder_dim)

    def forward(self, x):
        # x: (batch, 1, time, freq)
        x = self.conv(x)  # (batch, encoder_dim, time', freq')
        batch, channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, time, channels * freq)
        return self.projection(x)  # (batch, time', encoder_dim)


class ConformerBackend(nn.Module):
    def __init__(
        self,
        num_freqs,
        encoder_dim,
        num_layers,
        num_heads,
        ff_expansion_factor,
        conv_kernel_size,
        dropout,
        use_relative_positional_encoding,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(num_freqs, encoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder = ConformerEncoder(
            dim=encoder_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_dim=encoder_dim * ff_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_relative_positional_encoding=use_relative_positional_encoding,
        )
        self.backend_head = nn.Linear(encoder_dim, 1024)
        self.bottleneck_head = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        x = self.subsampling(x)
        x = self.dropout(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        backend = self.backend_head(pooled)
        bottleneck = self.bottleneck_head(backend)
        return BackendOutput(backend, bottleneck)


def build_conformer(config):
    return ConformerBackend(
        num_freqs=config.transformation.n_mels,
        encoder_dim=config.conformer.encoder_dim,
        num_layers=config.conformer.num_layers,
        num_heads=config.conformer.num_heads,
        ff_expansion_factor=config.conformer.ff_expansion_factor,
        conv_kernel_size=config.conformer.conv_kernel_size,
        dropout=config.conformer.dropout,
        use_relative_positional_encoding=config.conformer.use_relative_positional_encoding,
    )
