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


def build_model(config):
    return MODEL_REGISTRY[config.model.type](config)
