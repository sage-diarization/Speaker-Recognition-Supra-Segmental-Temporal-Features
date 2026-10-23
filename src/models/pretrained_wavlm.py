"""WavLM Large + ECAPA-TDNN speaker verification model (Chen et al. 2022,
github.com/microsoft/UniSpeech, downstreams/speaker_verification), evaluated
with its released fine-tuned weights (the "WavLM large, Fix pre-train: No"
checkpoint: 1.154% EER on VoxCeleb1-H). The checkpoint is on Google Drive
only, so it has to be downloaded by hand (see the pretrained-wavlm configs).

Re-implemented on the stock transformers WavLMModel (an exact port of the
original WavLM.py) with the checkpoint's fairseq parameter names mapped to
it, rather than installing s3prl and fairseq. The ECAPA-TDNN head keeps the
original's parameter names.

The model is split where frames first get mixed: the convolutional feature
encoder (25 ms receptive field, 20 ms stride) runs on the whole utterance as
the front-end (TransformationConfig.type "wavlm_cnn", see
wavlm_feature_encoder), so segment draws shuffle its output frames; WavLMECAPA
runs the rest -- feature projection, positional convolution, the 24
transformer layers with their gated relative position bias, a softmax-weighted
sum of all 25 hidden states, then the ECAPA-TDNN.
"""

import functools
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BackendOutput, resolve_checkpoint

WAVLM_CONFIG = "microsoft/wavlm-large"

# The original WavLM's parameter names (inside the checkpoint's
# "feature_extract.model." prefix) -> transformers' WavLMModel's, after
# transformers' own convert_wavlm_original_pytorch_checkpoint_to_pytorch.py.
_KEY_RENAMES = (
    (r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
    (r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.", r"feature_extractor.conv_layers.\1.layer_norm."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"^encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
    (r"^encoder\.pos_conv\.0\.weight_g$", "encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
    (r"^encoder\.pos_conv\.0\.weight_v$", "encoder.pos_conv_embed.conv.parametrizations.weight.original1"),
    (r"\.self_attn\.grep_linear\.", ".attention.gru_rel_pos_linear."),
    (r"\.self_attn\.grep_a$", ".attention.gru_rel_pos_const"),
    (r"\.self_attn\.relative_attention_bias\.", ".attention.rel_attn_embed."),
    (r"\.self_attn\.", ".attention."),
    (r"\.self_attn_layer_norm\.", ".layer_norm."),
    (r"\.fc1\.", ".feed_forward.intermediate_dense."),
    (r"\.fc2\.", ".feed_forward.output_dense."),
)
_WAVLM_PREFIX = "feature_extract.model."


def _wavlm_state_dict(checkpoint_state):
    """The checkpoint's WavLM parameters under transformers' names."""
    state = {}
    for key, value in checkpoint_state.items():
        if not key.startswith(_WAVLM_PREFIX):
            continue
        key = key[len(_WAVLM_PREFIX):]
        if key == "mask_emb":  # pretraining only
            continue
        for pattern, replacement in _KEY_RENAMES:
            key = re.sub(pattern, replacement, key)
        state[key] = value
    return state


def _load_wavlm(checkpoint_state):
    from transformers import WavLMConfig, WavLMModel

    wavlm = WavLMModel(WavLMConfig.from_pretrained(WAVLM_CONFIG))
    missing, unexpected = wavlm.load_state_dict(_wavlm_state_dict(checkpoint_state), strict=False)
    if missing != ["masked_spec_embed"] or unexpected:
        raise ValueError(f"WavLM checkpoint doesn't match: missing {missing}, unexpected {unexpected}")
    return wavlm


def _load_checkpoint_state(spec):
    path = resolve_checkpoint(spec)
    try:
        # Memory-mapped: every DataLoader worker loads it (wavlm_feature_encoder).
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialization can't be memory-mapped
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"]


@functools.cache
def wavlm_feature_encoder(checkpoint):
    """The checkpoint's convolutional feature encoder, (batch, samples) ->
    (batch, 512, frames), loaded once per process (DataLoader workers each
    load their own)."""
    return _load_wavlm(_load_checkpoint_state(checkpoint)).feature_extractor.requires_grad_(False).eval()


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size, padding, dilation, scale):
        super().__init__()
        self.width = channels // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, padding=padding, dilation=dilation) for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])

    def forward(self, x):
        splits = torch.split(x, self.width, 1)
        out = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            sp = splits[i] if i == 0 else sp + splits[i]
            sp = bn(F.relu(conv(sp)))
            out.append(sp)
        out.append(splits[-1])
        return torch.cat(out, dim=1)


class SEConnect(nn.Module):
    def __init__(self, channels, bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, channels)

    def forward(self, x):
        out = torch.sigmoid(self.linear2(F.relu(self.linear1(x.mean(dim=2)))))
        return x * out.unsqueeze(2)


class SERes2Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(channels, channels)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(channels, 3, padding=dilation, dilation=dilation, scale=8)
        self.Conv1dReluBn2 = Conv1dReluBn(channels, channels)
        self.SE_Connect = SEConnect(channels)

    def forward(self, x):
        return x + self.SE_Connect(self.Conv1dReluBn2(self.Res2Conv1dReluBn(self.Conv1dReluBn1(x))))


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        std = torch.sqrt((torch.sum(alpha * x ** 2, dim=2) - mean ** 2).clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class WavLMECAPA(nn.Module):
    def __init__(self, wavlm, channels=512, embed_dim=256):
        super().__init__()
        # Only the parts after the feature encoder, see the module docstring.
        self.feature_projection = wavlm.feature_projection
        self.encoder = wavlm.encoder
        hidden_size = wavlm.config.hidden_size
        self.feature_weight = nn.Parameter(torch.zeros(wavlm.config.num_hidden_layers + 1))
        self.instance_norm = nn.InstanceNorm1d(hidden_size)
        self.layer1 = Conv1dReluBn(hidden_size, channels, kernel_size=5, padding=2)
        self.layer2 = SERes2Block(channels, dilation=2)
        self.layer3 = SERes2Block(channels, dilation=3)
        self.layer4 = SERes2Block(channels, dilation=4)
        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536)
        self.bn = nn.BatchNorm1d(1536 * 2)
        self.linear = nn.Linear(1536 * 2, embed_dim)

    def weighted_hidden_states(self, frames):
        """Softmax(feature_weight)-weighted sum of the 25 hidden states
        s3prl's WavLM upstream hands the ECAPA-TDNN: each transformer layer's
        input, then the final layer-normed output. Summed up layer by layer:
        holding all 25 of an evaluation batch would take tens of GB."""
        weights = torch.softmax(self.feature_weight, dim=0)
        x = self.feature_projection(frames)[0]
        x = x + self.encoder.pos_conv_embed(x)
        total, position_bias = 0, None
        for weight, layer in zip(weights, self.encoder.layers):
            total = total + weight * x
            x, position_bias = layer(x, position_bias=position_bias)
        return total + weights[-1] * self.encoder.layer_norm(x)

    def forward(self, x):
        # x: (batch, 1, time, 512)
        x = self.weighted_hidden_states(x.squeeze(1))
        x = self.instance_norm(x.transpose(1, 2) + 1e-6)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        embedding = self.linear(self.bn(self.pooling(out)))
        return BackendOutput(embedding, embedding)


def build_wavlm_ecapa(config):
    state = _load_checkpoint_state(config.pretrained.checkpoint)
    model = WavLMECAPA(_load_wavlm(state))
    head = {k: v for k, v in state.items() if not k.startswith("feature_extract.")}
    missing, unexpected = model.load_state_dict(head, strict=False)
    missing = [k for k in missing if not k.startswith(("feature_projection.", "encoder."))]
    if missing or unexpected:
        raise ValueError(f"ECAPA-TDNN checkpoint doesn't match: missing {missing}, unexpected {unexpected}")
    return model.requires_grad_(False).eval()
