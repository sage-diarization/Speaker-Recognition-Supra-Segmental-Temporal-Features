import torch.nn as nn

from .cnn import build_cnn
from .conformer import build_conformer
from .fresnet import build_fresnet
from .resnet import build_resnet
from .rnn import build_rnn
from .w2vbert import build_w2vbert

MODEL_REGISTRY = {
    "CNN": build_cnn,
    "Conformer": build_conformer,
    "FResNet": build_fresnet,
    "RNN": build_rnn,
    "ResNet": build_resnet,
    "W2VBert": build_w2vbert,
}


class SegmentMeanNormalization(nn.Module):
    """Subtracts each frequency band's mean over time from every input
    segment (batch, 1, time, freq) before the wrapped model -- the
    transformation.normalization "segment" option. The mean ignores frame
    order, so shuffled and ordered segments are normalized alike."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x - x.mean(dim=2, keepdim=True))


def build_model(config):
    model = MODEL_REGISTRY[config.model.type](config)
    if config.transformation.normalization == "segment" and config.transformation.type != "w2vbert":
        model = SegmentMeanNormalization(model)
    return model
