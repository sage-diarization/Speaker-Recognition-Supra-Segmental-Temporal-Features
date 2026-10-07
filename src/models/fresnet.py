"""
F-ResNet (Fast ResNet-34) implementation.

PyTorch port of the ResNetSE34L model from github.com/clovaai/voxceleb_trainer
(Chung et al., "In defence of metric learning for speaker recognition", Interspeech 2020).

This corresponds to the "Fast ResNet-34" (F-ResNet) model referenced as [29] in
Neururer et al. 2024 ("Deep neural networks for automatic speaker recognition do not
learn supra-segmental temporal features"), Table 3.

Architecture highlights:
- SEBasicBlock (Squeeze-and-Excitation) residual blocks
- Layer configuration: [3, 4, 6, 3] (standard ResNet-34)
- Channel configuration: [16, 32, 64, 128]
- Stem: 7x7 conv, stride=(2,1)
- Pooling: Self-Attentive Pooling (SAP) or Attentive Statistics Pooling (ASP)
- Front-end: Mel spectrogram with 40 mel bins (n_mels=40), Hamming window

Training protocol (from clovaai/voxceleb_trainer README):
- Training segments: 200 frames (2 seconds at 10ms frame step)
- Evaluation segments: 400 frames (4 seconds)
- Batch size: 800 (achieved via nPerSpeaker=2, batch_size=400)
- Epochs: 500
- Loss: Angular Prototypical (angleproto) or AM-Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import KERAS_BATCH_NORM, BackendOutput


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer (Hu et al., 2018)."""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    """SE-ResNet basic block with expansion=1."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class FResNetBackend(nn.Module):
    """F-ResNet (Fast ResNet-34 / ResNetSE34L) backend.
    
    Ported from github.com/clovaai/voxceleb_trainer's ResNetSE34L model.
    Operates on (batch, 1, time, freq) tensors after front-end transformation.
    
    Args:
        num_freqs: Number of frequency bins from the front-end.
        bottleneck_dim: Dimension of the speaker embedding (default: 512).
        encoder_type: Pooling type - "SAP" (Self-Attentive Pooling) or "ASP" 
            (Attentive Statistics Pooling). Default: "SAP".
        n_mels: Number of mel bins (for compatibility with clovaai config).
            Not used directly as the front-end is handled by the transformation
            pipeline, but included for reference.
    """

    def __init__(self, num_freqs, bottleneck_dim=512, encoder_type="SAP", n_mels=40):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        # ResNetSE34L uses these layer and filter configurations
        layers = [3, 4, 6, 3]
        num_filters = [16, 32, 64, 128]
        
        self.inplanes = num_filters[0]
        
        # Stem: 7x7 conv with stride (2,1) as in clovaai's ResNetSE34L
        # Note: The original uses stride=(2,1) which downsample in freq but not time
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, 
                               stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        
        # Build the 4 stages
        self.layer1 = self._make_layer(SEBasicBlock, num_filters[0], layers[0], stride=1)
        self.layer2 = self._make_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(SEBasicBlock, num_filters[3], layers[3], stride=(1, 1))
        
        # Pooling dimensions: we need to compute the output shape after all conv layers
        # This is traced to avoid hardcoding the spatial dimensions
        self._trace_pooling_dims(num_freqs)
        
        # Self-Attentive Pooling (SAP) or Attentive Statistics Pooling (ASP)
        if encoder_type == "SAP":
            self.pooling_linear = nn.Linear(num_filters[3] * SEBasicBlock.expansion, 
                                           num_filters[3] * SEBasicBlock.expansion)
            self.attention = nn.Parameter(torch.FloatTensor(
                num_filters[3] * SEBasicBlock.expansion, 1))
            nn.init.xavier_normal_(self.attention)
            out_dim = num_filters[3] * SEBasicBlock.expansion
        elif encoder_type == "ASP":
            self.pooling_linear = nn.Linear(num_filters[3] * SEBasicBlock.expansion, 
                                           num_filters[3] * SEBasicBlock.expansion)
            self.attention = nn.Parameter(torch.FloatTensor(
                num_filters[3] * SEBasicBlock.expansion, 1))
            nn.init.xavier_normal_(self.attention)
            out_dim = num_filters[3] * SEBasicBlock.expansion * 2  # mu + std
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}. Use 'SAP' or 'ASP'.")
        
        # Final fully-connected layer to bottleneck dimension
        self.fc = nn.Linear(out_dim, bottleneck_dim)
        
        # Initialize weights: clovaai uses kaiming_normal for conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Build a ResNet stage with the given block type and parameters."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _trace_pooling_dims(self, num_freqs):
        """Trace the spatial dimensions after all convolutional layers.
        
        This is necessary to determine the input size for the pooling layer.
        We use a dummy input with the configured num_freqs and a representative
        time dimension.
        """
        # We don't actually need to store these as the pooling is done dynamically
        # in forward(), but we do need to know the channel dimension for SAP/ASP
        pass

    def _sap_pooling(self, x):
        """Self-Attentive Pooling (SAP)."""
        # x shape: (batch, channels, 1, time) after mean pooling over freq
        x = x.permute(0, 3, 1, 2).squeeze(-1)  # (batch, time, channels)
        
        h = torch.tanh(self.pooling_linear(x))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        
        x = torch.sum(x * w, dim=1)  # (batch, channels)
        return x

    def _asp_pooling(self, x):
        """Attentive Statistics Pooling (ASP)."""
        # x shape: (batch, channels, 1, time) after mean pooling over freq
        x = x.permute(0, 3, 1, 2).squeeze(-1)  # (batch, time, channels)
        
        h = torch.tanh(self.pooling_linear(x))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        
        # Compute mean and std
        mu = torch.sum(x * w, dim=1)
        rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
        
        x = torch.cat((mu, rh), dim=1)  # (batch, channels * 2)
        return x

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, time, freq)
        
        Returns:
            BackendOutput(backend, bottleneck) where both are the speaker embedding
            (F-ResNet uses CUT=AGGREGATION, so backend == bottleneck)
        """
        # x shape: (batch, 1, time, freq)
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Mean pooling over frequency dimension
        x = torch.mean(x, dim=2, keepdim=True)  # (batch, channels, 1, time)
        
        # Apply pooling (SAP or ASP)
        if self.encoder_type == "SAP":
            x = self._sap_pooling(x)
        elif self.encoder_type == "ASP":
            x = self._asp_pooling(x)
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")
        
        # Final FC layer
        x = self.fc(x)
        
        # F-ResNet uses CUT=AGGREGATION (like the existing ResNet), so backend == bottleneck
        return BackendOutput(backend=x, bottleneck=x)


def build_fresnet(config):
    """Factory function to build F-ResNet model from config.
    
    Args:
        config: ExperimentConfig with fresnet section
    
    Returns:
        FResNetBackend model instance
    """
    return FResNetBackend(
        num_freqs=config.transformation.num_freqs,
        bottleneck_dim=config.fresnet.bottleneck,
        encoder_type=config.fresnet.encoder_type,
        n_mels=config.transformation.n_mels,
    )
