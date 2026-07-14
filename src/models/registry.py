from .cnn import build_cnn

MODEL_REGISTRY = {
    "CNN": build_cnn,
}


def build_model(config):
    return MODEL_REGISTRY[config.model.type](config)
