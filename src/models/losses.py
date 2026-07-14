import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxLoss(nn.Module):
    """Port of context/src/models/losses/softmax.py."""

    def __init__(self, bottleneck_dim, num_speakers):
        super().__init__()
        self.head = nn.Linear(bottleneck_dim, num_speakers)

    def forward(self, bottleneck, labels):
        logits = self.head(bottleneck)
        return F.cross_entropy(logits, labels), logits


class AngularMarginLoss(nn.Module):
    """CosFace-style angular margin loss, port of
    context/src/models/losses/angular_margin.py. Returns pre-scale cosine
    similarities as `logits` (bounded in [-1, 1]) for inspection/testing."""

    def __init__(self, bottleneck_dim, num_speakers, margin_cosface, margin_arcface, margin_sphereface, scale):
        super().__init__()
        self.W = nn.Parameter(torch.empty(bottleneck_dim, num_speakers))
        nn.init.xavier_uniform_(self.W)
        self.margin_cosface = margin_cosface
        self.margin_arcface = margin_arcface
        self.margin_sphereface = margin_sphereface
        self.scale = scale

    def forward(self, bottleneck, labels):
        x = F.normalize(bottleneck, dim=1)
        w = F.normalize(self.W, dim=0)
        cos_theta = x @ w

        target_logits = cos_theta
        if self.margin_sphereface != 1.0 or self.margin_arcface != 0.0:
            eps = 1e-7
            theta = torch.acos(cos_theta.clamp(-1.0 + eps, 1.0 - eps))
            if self.margin_sphereface != 1.0:
                theta = theta * self.margin_sphereface
            if self.margin_arcface != 0.0:
                theta = theta + self.margin_arcface
            target_logits = torch.cos(theta)

        if self.margin_cosface != 0.0:
            target_logits = target_logits - self.margin_cosface

        one_hot = F.one_hot(labels, cos_theta.shape[1]).float()
        scaled_logits = (cos_theta * (1 - one_hot) + target_logits * one_hot) * self.scale
        return F.cross_entropy(scaled_logits, labels), cos_theta


def build_softmax(config, bottleneck_dim, num_speakers):
    return SoftmaxLoss(bottleneck_dim, num_speakers)


def build_angular_margin(config, bottleneck_dim, num_speakers):
    return AngularMarginLoss(
        bottleneck_dim,
        num_speakers,
        config.loss.margin_cosface,
        config.loss.margin_arcface,
        config.loss.margin_sphereface,
        config.loss.scale,
    )


LOSS_REGISTRY = {
    "SOFTMAX": build_softmax,
    "ANGULAR_MARGIN": build_angular_margin,
}


def build_loss(config, bottleneck_dim, num_speakers):
    return LOSS_REGISTRY[config.loss.type](config, bottleneck_dim, num_speakers)
