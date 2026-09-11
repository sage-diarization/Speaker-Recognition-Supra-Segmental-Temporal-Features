import pytest
import torch

from src.config import ExperimentConfig
from src.models.cnn import CNNBackend
from src.models.conformer import ConformerBackend
from src.models.registry import build_model
from src.models.resnet import ResNetBackend
from src.models.rnn import RNNBackend


def test_build_model_selects_cnn_by_default():
    config = ExperimentConfig()
    model = build_model(config)
    assert isinstance(model, CNNBackend)


def test_build_model_selects_conformer():
    config = ExperimentConfig()
    config.model.type = "Conformer"
    config.conformer.num_layers = 1
    config.conformer.encoder_dim = 16
    config.conformer.num_heads = 2
    config.conformer.ff_expansion_factor = 2
    model = build_model(config)
    assert isinstance(model, ConformerBackend)


def test_build_model_selects_rnn():
    config = ExperimentConfig()
    config.model.type = "RNN"
    config.rnn.hidden_size = 8
    model = build_model(config)
    assert isinstance(model, RNNBackend)


def test_build_model_selects_resnet():
    config = ExperimentConfig()
    config.model.type = "ResNet"
    config.transformation.type = "linear"
    config.transformation.nfft = 126  # num_freqs = 126 // 2 + 1 = 64, the smallest viable size
    config.resnet.vlad_clusters = 4
    config.resnet.ghost_clusters = 2
    config.resnet.bottleneck = 16
    model = build_model(config)
    assert isinstance(model, ResNetBackend)


def test_build_model_rejects_unknown_type():
    config = ExperimentConfig()
    config.model.type = "NOT_A_MODEL"
    with pytest.raises(KeyError):
        build_model(config)


def test_cnn_and_conformer_are_interchangeable_for_a_given_input_shape():
    config = ExperimentConfig()
    config.conformer.num_layers = 1
    config.conformer.encoder_dim = 16
    config.conformer.num_heads = 2
    config.conformer.ff_expansion_factor = 2

    x = torch.randn(2, 1, 94, config.transformation.n_mels)

    config.model.type = "CNN"
    cnn_out = build_model(config)(x)

    config.model.type = "Conformer"
    conformer_out = build_model(config)(x)

    assert cnn_out.backend.shape == conformer_out.backend.shape == (2, 1024)
    assert cnn_out.bottleneck.shape == conformer_out.bottleneck.shape == (2, 512)
