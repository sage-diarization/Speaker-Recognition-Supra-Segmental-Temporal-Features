from .cnn import build_cnn
from .conformer import build_conformer
from .resnet import build_resnet
from .rnn import build_rnn

MODEL_REGISTRY = {
    "CNN": build_cnn,
    "Conformer": build_conformer,
    "RNN": build_rnn,
    "ResNet": build_resnet,
}


def build_model(config):
    return MODEL_REGISTRY[config.model.type](config)
