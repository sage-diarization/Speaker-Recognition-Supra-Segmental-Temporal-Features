"""w2v-BERT 2.0 speaker verification model of Li, Cheng & Li 2025
("Enhancing Speaker Verification with w2v-BERT 2.0 and Knowledge
Distillation guided Structured Pruning", github.com/ZXHY-82/w2v-BERT-2.0_SV),
evaluated with its released weights (zl389/w2v-bert-2.0_SV,
Adapter_MFA_voxceleb2/s3: fully fine-tuned on VoxCeleb2 with large-margin
fine-tuning, 0.81% EER on VoxCeleb1-H without score calibration).

Re-implemented on the stock transformers Wav2Vec2BertModel rather than
copied: the authors' code (CC BY-NC-SA 4.0) runs a patched transformers. Its
"Adapter_MFA" head: every hidden state (the feature projection's output and
the 24 layers') through its own small adapter MLP, concatenated, attentive
statistics pooling (order-invariant), one linear embedding layer.

Its input frames are w2v-BERT's 160-dim stacked fbank
(TransformationConfig.type "w2vbert"), what segment draws shuffle, as in
src/models/w2vbert.py.
"""

import torch
import torch.nn as nn

from .common import BackendOutput, resolve_checkpoint


class AttentiveStatisticsPooling(nn.Module):
    """The authors' ASP: per-dimension attention weights over time, then the
    weighted mean and standard deviation."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # x: (batch, time, dim)
        w = self.attention(x.transpose(1, 2)).transpose(1, 2)
        mu = torch.sum(x * w, dim=1)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
        return torch.cat([mu, sg], dim=1)


class W2VBertSV(nn.Module):
    def __init__(self, encoder, adapter_dim=128, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        num_hidden_states = encoder.config.num_hidden_layers + 1
        hidden_size = encoder.config.hidden_size
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.ReLU(True),
                nn.Linear(adapter_dim, adapter_dim),
            ) for _ in range(num_hidden_states)
        ])
        self.pooling = AttentiveStatisticsPooling(adapter_dim * num_hidden_states, adapter_dim)
        self.bottleneck = nn.Linear(2 * adapter_dim * num_hidden_states, embed_dim)

    def forward(self, x):
        # x: (batch, 1, time, 160). The encoder's layers are run one by one,
        # as the authors' forward_w2v_bert does, so each hidden state goes
        # through its adapter right away: holding all 25 full-width states
        # of an evaluation batch would take tens of GB.
        x = self.encoder.feature_projection(x.squeeze(1))[0]
        adapted = [self.adapter_layers[0](x)]
        for layer, adapter in zip(self.encoder.encoder.layers, self.adapter_layers[1:]):
            x = layer(x)
            adapted.append(adapter(x))
        embedding = self.bottleneck(self.pooling(torch.cat(adapted, dim=-1)))
        return BackendOutput(embedding, embedding)


def build_w2vbert_sv(config):
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    # Architecture from the base model; all weights from the fine-tuned checkpoint.
    model = W2VBertSV(Wav2Vec2BertModel(Wav2Vec2BertConfig.from_pretrained(config.w2vbert.pretrained)))
    checkpoint = torch.load(resolve_checkpoint(config.pretrained.checkpoint), map_location="cpu", weights_only=False)
    state = {k.replace("front.encoder.", "encoder.", 1): v for k, v in checkpoint["modules"]["spk_model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The authors drop the pretraining-only mask embedding.
    if missing != ["encoder.masked_spec_embed"] or unexpected:
        raise ValueError(f"checkpoint doesn't match: missing {missing}, unexpected {unexpected}")
    return model.requires_grad_(False).eval()
