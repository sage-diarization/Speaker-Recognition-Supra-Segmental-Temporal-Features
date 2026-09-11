import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BackendOutput


class _ConvBlock(nn.Module):
    """context/src/models/backend/blocks/ResNet.py::conv_block_2D: a
    strided 1x1 "reduce" entry (with a matching strided 1x1 shortcut
    projection) around a stride-1, SAME-padded 3x3 conv."""

    def __init__(self, in_channels, filters, stride):
        super().__init__()
        f1, f2, f3 = filters
        self.reduce = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride, bias=False)
        self.reduce_bn = nn.BatchNorm2d(f1)
        self.conv = nn.Conv2d(f1, f2, kernel_size=3, padding=1, bias=False)
        self.conv_bn = nn.BatchNorm2d(f2)
        self.increase = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.increase_bn = nn.BatchNorm2d(f3)
        self.shortcut = nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(f3)

    def forward(self, x):
        identity = self.shortcut_bn(self.shortcut(x))
        out = F.relu(self.reduce_bn(self.reduce(x)))
        out = F.relu(self.conv_bn(self.conv(out)))
        out = self.increase_bn(self.increase(out))
        return F.relu(out + identity)


class _IdentityBlock(nn.Module):
    """context/src/models/backend/blocks/ResNet.py::identity_block_2D: the
    same bottleneck path as _ConvBlock but stride-1 with no shortcut
    projection (in_channels must equal filters[2])."""

    def __init__(self, channels, filters):
        super().__init__()
        f1, f2, f3 = filters
        self.reduce = nn.Conv2d(channels, f1, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(f1)
        self.conv = nn.Conv2d(f1, f2, kernel_size=3, padding=1, bias=False)
        self.conv_bn = nn.BatchNorm2d(f2)
        self.increase = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.increase_bn = nn.BatchNorm2d(f3)

    def forward(self, x):
        out = F.relu(self.reduce_bn(self.reduce(x)))
        out = F.relu(self.conv_bn(self.conv(out)))
        out = self.increase_bn(self.increase(out))
        return F.relu(out + x)


class ResNet34sBackbone(nn.Module):
    """PyTorch port of context/src/models/backend/ResNet34s.py's
    convolutional backbone (everything before GhostVLAD): a 7x7 stem
    followed by 4 bottleneck-block stages (2/3/3/3 blocks), then the
    (3, 1)-kernel 'mpool2' that further squeezes the frequency axis. Operates
    on (batch, 1, freq, time) -- ResNet34s.py's ["F", "T", 1] layout, unlike
    every other backend in this project which is (batch, 1, time, freq) --
    see ResNetBackend.forward for the transpose that bridges the two."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            _ConvBlock(64, (48, 48, 96), stride=1),
            _IdentityBlock(96, (48, 48, 96)),
        )
        self.stage3 = nn.Sequential(
            _ConvBlock(96, (96, 96, 128), stride=2),
            _IdentityBlock(128, (96, 96, 128)),
            _IdentityBlock(128, (96, 96, 128)),
        )
        self.stage4 = nn.Sequential(
            _ConvBlock(128, (128, 128, 256), stride=2),
            _IdentityBlock(256, (128, 128, 256)),
            _IdentityBlock(256, (128, 128, 256)),
        )
        self.stage5 = nn.Sequential(
            _ConvBlock(256, (256, 256, 512), stride=2),
            _IdentityBlock(512, (256, 256, 512)),
            _IdentityBlock(512, (256, 256, 512)),
        )
        self.mpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.mpool2(x)


class GhostVladPooling(nn.Module):
    """context/src/models/aggregation/blocks/vlad.py::VladPooling ported to
    PyTorch, specialized to the 'gvlad' mode ResNet34s.py uses. feat and
    cluster_score arrive as (batch, channels, 1, time) -- the frequency axis
    has already been collapsed to 1 by the caller's full-height convolution,
    matching the original's H=1 after its own full-height Conv2D -- and are
    squeezed/transposed to (batch, time, channels) before reproducing the
    original's softmax-weighted-residual computation over the cluster axis,
    summed over time."""

    def __init__(self, k_centers, g_centers, feature_dim):
        super().__init__()
        self.k_centers = k_centers
        self.cluster = nn.Parameter(torch.empty(k_centers + g_centers, feature_dim))
        nn.init.orthogonal_(self.cluster)

    def forward(self, feat, cluster_score):
        feat = feat.squeeze(2).transpose(1, 2)  # (batch, time, D)
        cluster_score = cluster_score.squeeze(2).transpose(1, 2)  # (batch, time, k+g)

        assignment = torch.softmax(cluster_score, dim=-1).unsqueeze(-1)  # (batch, time, k+g, 1)
        residual = feat.unsqueeze(2) - self.cluster  # (batch, time, k+g, D)
        cluster_res = (assignment * residual).sum(dim=1)  # (batch, k+g, D)
        cluster_res = cluster_res[:, : self.k_centers, :]
        cluster_res = F.normalize(cluster_res, p=2, dim=-1)
        return cluster_res.reshape(cluster_res.shape[0], -1)  # (batch, k_centers * D)


def _orthogonal_init(module):
    """Matches ResNet34s.py/GhostVlad.py's kernel_initializer='orthogonal',
    bias_initializer='zeros' (Keras' Conv2D/Dense default) on every Conv2d
    and Linear layer."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ResNetBackend(nn.Module):
    """PyTorch port of context/src/models/backend/ResNet34s.py +
    context/src/models/aggregation/GhostVlad.py (Neururer et al. 2024's
    ResNet [27]). Unlike CNNBackend/RNNBackend/ConformerBackend, the
    original's CUT setting for this model is 'AGGREGATION'
    (context/src/00_configs/05_model/RES34S.json), meaning the SV/SC
    evaluation embedding *is* the GhostVLAD-aggregated, loss-facing
    embedding -- there is no separate larger pre-loss layer as in the other
    backends -- so backend and bottleneck below are the same tensor."""

    def __init__(self, num_freqs, vlad_clusters=10, ghost_clusters=2, bottleneck_dim=512):
        super().__init__()
        self.backbone = ResNet34sBackbone()
        # Traced (not computed from a formula) since Keras 'same'/'valid'
        # padding rounding through 4 downsampling stages is fiddly to get
        # exactly right by hand. eval() + batch=2 avoids BatchNorm2d's
        # training-mode "more than 1 value per channel" restriction, which
        # would otherwise spuriously trip on the small spatial maps this
        # trace produces (a genuinely too-small num_freqs still raises
        # PyTorch's own "output size too small" error here).
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 1, num_freqs, 8)
            freq_shape = self.backbone(dummy).shape[2]
        self.backbone.train()

        self.fc = nn.Conv2d(512, bottleneck_dim, kernel_size=(freq_shape, 1))
        self.cluster_conv = nn.Conv2d(512, vlad_clusters + ghost_clusters, kernel_size=(freq_shape, 1))
        self.vlad = GhostVladPooling(vlad_clusters, ghost_clusters, bottleneck_dim)
        self.fc6 = nn.Linear(vlad_clusters * bottleneck_dim, bottleneck_dim)

        _orthogonal_init(self)

    def forward(self, x):
        x = x.transpose(2, 3)  # (batch, 1, time, freq) -> (batch, 1, freq, time)
        x = self.backbone(x)
        x_fc = F.relu(self.fc(x))
        cluster_score = self.cluster_conv(x)
        pooled = self.vlad(x_fc, cluster_score)
        embedding = F.relu(self.fc6(pooled))
        return BackendOutput(backend=embedding, bottleneck=embedding)


def build_resnet(config):
    return ResNetBackend(
        num_freqs=config.transformation.num_freqs,
        vlad_clusters=config.resnet.vlad_clusters,
        ghost_clusters=config.resnet.ghost_clusters,
        bottleneck_dim=config.resnet.bottleneck,
    )
