"""WeSpeaker's ResNet34 speaker-embedding model (Wang et al. 2023,
github.com/wenet-e2e/wespeaker, wespeaker/models/resnet.py, Apache-2.0),
evaluated with its released VoxCeleb2-trained weights
(Wespeaker/wespeaker-voxceleb-resnet34-LM: large-margin fine-tuned, 1.695%
EER on VoxCeleb1-H without score normalization).

Its input frames are the 80-bin Kaldi fbank of TransformationConfig.type
"wespeaker_fbank" -- what segment draws shuffle, since the first 3x3
convolution is the first layer that mixes neighbouring frames. Parameter
names follow wespeaker's ResNet, so its checkpoint loads as is.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BackendOutput, resolve_checkpoint


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class WeSpeakerResNet34(nn.Module):
    """ResNet34 (blocks 3-4-6-3, 32 base channels) + temporal statistics
    pooling (mean and std over time, order-invariant) + one linear
    embedding layer."""

    def __init__(self, feat_dim=80, embed_dim=256, m_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        in_planes = m_channels
        for index, (num_blocks, multiplier, stride) in enumerate(((3, 1, 1), (4, 2, 2), (6, 4, 2), (3, 8, 2)), start=1):
            planes = m_channels * multiplier
            blocks = [BasicBlock(in_planes if i == 0 else planes, planes, stride if i == 0 else 1) for i in range(num_blocks)]
            setattr(self, f"layer{index}", nn.Sequential(*blocks))
            in_planes = planes
        self.seg_1 = nn.Linear(2 * (feat_dim // 8) * m_channels * 8, embed_dim)

    def forward(self, x):
        # x: (batch, 1, time, freq) -> wespeaker's (batch, 1, freq, time)
        out = F.relu(self.bn1(self.conv1(x.transpose(2, 3))))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = out.flatten(1, 2)  # (batch, channels * freq, time)
        stats = torch.cat([out.mean(dim=-1), torch.sqrt(out.var(dim=-1) + 1e-7)], dim=1)
        embedding = self.seg_1(stats)
        return BackendOutput(embedding, embedding)


def build_wespeaker_resnet34(config):
    model = WeSpeakerResNet34()
    state = torch.load(resolve_checkpoint(config.pretrained.checkpoint), map_location="cpu")
    # projection.* is the training-only speaker classifier.
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("projection.")})
    return model.requires_grad_(False).eval()
