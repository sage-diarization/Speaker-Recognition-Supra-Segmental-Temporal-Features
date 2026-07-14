import copy

import torch

from src.models.losses import AngularMarginLoss, SoftmaxLoss


def test_softmax_loss_scalar_and_backward():
    loss_module = SoftmaxLoss(bottleneck_dim=512, num_speakers=5)
    bottleneck = torch.randn(6, 512, requires_grad=True)
    labels = torch.randint(0, 5, (6,))

    loss, logits = loss_module(bottleneck, labels)
    assert loss.dim() == 0
    assert logits.shape == (6, 5)
    loss.backward()
    assert bottleneck.grad is not None


def test_angular_margin_loss_scalar_backward_and_bounded_logits():
    loss_module = AngularMarginLoss(
        bottleneck_dim=512, num_speakers=5,
        margin_cosface=0.3, margin_arcface=0.0, margin_sphereface=1.0, scale=40.0,
    )
    bottleneck = torch.randn(6, 512, requires_grad=True)
    labels = torch.randint(0, 5, (6,))

    loss, cos_theta = loss_module(bottleneck, labels)
    assert loss.dim() == 0
    assert torch.all(cos_theta >= -1.0 - 1e-5) and torch.all(cos_theta <= 1.0 + 1e-5)
    loss.backward()
    assert bottleneck.grad is not None


def test_cosface_margin_makes_the_task_harder_than_zero_margin():
    torch.manual_seed(0)
    base = AngularMarginLoss(
        bottleneck_dim=32, num_speakers=4,
        margin_cosface=0.0, margin_arcface=0.0, margin_sphereface=1.0, scale=40.0,
    )
    margined = AngularMarginLoss(
        bottleneck_dim=32, num_speakers=4,
        margin_cosface=0.3, margin_arcface=0.0, margin_sphereface=1.0, scale=40.0,
    )
    margined.W = copy.deepcopy(base.W)

    bottleneck = torch.randn(8, 32)
    labels = torch.randint(0, 4, (8,))

    base_loss, _ = base(bottleneck, labels)
    margined_loss, _ = margined(bottleneck, labels)

    assert margined_loss.item() >= base_loss.item()
