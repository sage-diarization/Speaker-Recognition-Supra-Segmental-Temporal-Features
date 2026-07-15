from .cnn import build_cnn
from .conformer import build_conformer

MODEL_REGISTRY = {
    "CNN": build_cnn,
    "Conformer": build_conformer,
}


def build_model(config):
    return MODEL_REGISTRY[config.model.type](config)
