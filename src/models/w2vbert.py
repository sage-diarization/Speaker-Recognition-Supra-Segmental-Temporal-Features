"""Frozen pretrained w2v-BERT 2.0 (Barrault et al. 2023, Seamless) encoder
with a trainable speaker-embedding head, to measure with the OS/SS/SU test
how much supra-segmental temporal information a large self-supervised
model exploits.

The segment draws (src/data/segments.py) shuffle the encoder's *input*
frames -- its 160-dim stacked log-fbank (TransformationConfig.type
"w2vbert") -- since that is the last point before any temporal context is
mixed in: shuffling the encoder's outputs would leave SST already encoded in
every frame. Only the head is trained, and it pools over time
order-invariantly (mean + std), so any OS vs SS/SU difference comes from the
pretrained encoder itself.

Head: a learned softmax-weighted sum over all hidden states (the SUPERB
benchmark's standard probe, as speaker information concentrates in lower
layers), mean + std pooling, then the same backend/bottleneck layers as
src/models/conformer.py.
"""

import torch
import torch.nn as nn

from .common import BackendOutput


class W2VBertBackend(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        encoder.requires_grad_(False)
        encoder.eval()
        # Kept out of the module tree on purpose: the frozen encoder (~580M
        # parameters) then stays out of state_dict() -- checkpoints hold only
        # the trainable head, the encoder is reloaded from its pretrained id --
        # out of the optimizer, and in eval mode (no dropout) whatever
        # .train() is called on this module. _apply forwards .to(device) etc.
        object.__setattr__(self, "encoder", encoder)
        num_hidden_states = encoder.config.num_hidden_layers + 1
        hidden_size = encoder.config.hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(num_hidden_states))
        self.backend_head = nn.Linear(2 * hidden_size, 1024)
        self.bottleneck_head = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
        )

    def _apply(self, fn, recurse=True):
        self.encoder._apply(fn, recurse)
        return super()._apply(fn, recurse)

    def forward(self, x):
        # x: (batch, 1, time, 160)
        with torch.no_grad():
            hidden_states = self.encoder(input_features=x.squeeze(1), output_hidden_states=True).hidden_states
        weights = torch.softmax(self.layer_weights, dim=0)
        x = (weights[:, None, None, None] * torch.stack(hidden_states)).sum(dim=0)  # (batch, time, hidden)
        pooled = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        backend = self.backend_head(pooled)
        bottleneck = self.bottleneck_head(backend)
        return BackendOutput(backend, bottleneck)


def build_w2vbert(config):
    from transformers import Wav2Vec2BertModel

    return W2VBertBackend(Wav2Vec2BertModel.from_pretrained(config.w2vbert.pretrained))
