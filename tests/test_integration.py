import copy
import itertools

import numpy as np
import torch

from src.config import ExperimentConfig, TransformationConfig
from src.data.dataset import SegmentDataset, featurize_waveform
from src.evaluation.clustering import best_misclassification_rate
from src.evaluation.verification import equal_error_rate
from src.models.losses import build_loss
from src.models.registry import build_model
from src.training import trainer
from src.training.trainer import extract_embeddings, train

STRATEGIES = ("OS", "SS", "SU")


def _split(utterances, num_speakers, train_n, test_n):
    train_set, test_set = [], []
    for speaker in range(num_speakers):
        speaker_utterances = [u for u in utterances if u[1] == speaker]
        train_set.extend(speaker_utterances[:train_n])
        test_set.extend(speaker_utterances[train_n:train_n + test_n])
    return train_set, test_set


def test_training_loss_decreases_on_synthetic_speakers(synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=4, utterances_per_speaker=6)
    train_set, _ = _split(utterances, 4, 6, 0)

    config = ExperimentConfig()
    config.training.num_epochs = 20
    config.training.batch_size = 8
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    dataset = SegmentDataset(train_set, segment_length, "OS", seed=0)
    model = build_model(config)
    loss_module = build_loss(config, bottleneck_dim=512, num_speakers=4)

    history = train(model, loss_module, dataset, config)

    assert history[-1] < history[0]


def test_full_train_eval_grid_produces_valid_metrics(synthetic_utterances):
    utterances = synthetic_utterances(num_speakers=4, utterances_per_speaker=8)
    train_set, test_set = _split(utterances, 4, 6, 2)

    config = ExperimentConfig()
    config.training.num_epochs = 5
    config.training.batch_size = 8
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    for train_strategy in STRATEGIES:
        dataset = SegmentDataset(train_set, segment_length, train_strategy, seed=0)
        model = build_model(config)
        loss_module = build_loss(config, bottleneck_dim=512, num_speakers=4)
        train(model, loss_module, dataset, config)

        for test_strategy in STRATEGIES:
            sv_embeddings, sv_labels = extract_embeddings(model, test_set, segment_length, test_strategy, seed=1)
            eer = equal_error_rate(sv_embeddings, sv_labels)
            assert 0.0 <= eer <= 1.0

            mr = best_misclassification_rate(sv_embeddings, sv_labels)
            assert 0.0 <= mr <= 1.0


def test_rnn_full_train_eval_grid_produces_valid_metrics(synthetic_utterances):
    # Same shape as test_full_train_eval_grid_produces_valid_metrics, but for
    # the RNN backend (context/src/models/backend/LSTM.py), which shares the
    # CNN's mel front-end so the same synthetic_utterances fixture applies.
    utterances = synthetic_utterances(num_speakers=4, utterances_per_speaker=8)
    train_set, test_set = _split(utterances, 4, 6, 2)

    config = ExperimentConfig()
    config.model.type = "RNN"
    config.rnn.hidden_size = 8
    config.training.num_epochs = 3
    config.training.batch_size = 8
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    for train_strategy in STRATEGIES:
        dataset = SegmentDataset(train_set, segment_length, train_strategy, seed=0)
        model = build_model(config)
        loss_module = build_loss(config, bottleneck_dim=512, num_speakers=4)
        train(model, loss_module, dataset, config)

        for test_strategy in STRATEGIES:
            sv_embeddings, sv_labels = extract_embeddings(model, test_set, segment_length, test_strategy, seed=1)
            eer = equal_error_rate(sv_embeddings, sv_labels)
            assert 0.0 <= eer <= 1.0

            mr = best_misclassification_rate(sv_embeddings, sv_labels)
            assert 0.0 <= mr <= 1.0


def test_resnet_full_train_eval_grid_produces_valid_metrics():
    # ResNet uses its own (linear-spectrogram) front-end, so it can't reuse
    # the synthetic_utterances fixture, which is built on the default mel
    # TransformationConfig -- build synthetic utterances directly here
    # instead. nfft=126 -> num_freqs=64, the smallest ResNet34s survives.
    resnet_transformation = TransformationConfig(
        type="linear", window="hamming", nfft=126, frame_length_s=0.025, frame_step_s=0.01
    )
    rng = np.random.default_rng(0)

    def make_utterance(speaker, utterance_idx):
        frequency = 200 + 150 * speaker
        duration_s = 1.5
        t = np.arange(int(duration_s * resnet_transformation.sample_rate)) / resnet_transformation.sample_rate
        signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        noise = rng.normal(0, 0.01, size=t.shape)
        waveform = (signal + noise).astype(np.float32)
        return featurize_waveform(waveform, resnet_transformation), speaker

    utterances = [make_utterance(speaker, i) for speaker in range(4) for i in range(8)]
    train_set, test_set = _split(utterances, 4, 6, 2)

    config = ExperimentConfig()
    config.transformation = resnet_transformation
    config.model.type = "ResNet"
    config.resnet.vlad_clusters = 4
    config.resnet.ghost_clusters = 2
    config.resnet.bottleneck = 16
    config.training.num_epochs = 2
    config.training.batch_size = 8
    config.loss.type = "SOFTMAX"
    segment_length = config.data.segment_length(config.transformation)

    for train_strategy in STRATEGIES:
        dataset = SegmentDataset(train_set, segment_length, train_strategy, seed=0)
        model = build_model(config)
        loss_module = build_loss(config, bottleneck_dim=16, num_speakers=4)
        train(model, loss_module, dataset, config)

        for test_strategy in STRATEGIES:
            sv_embeddings, sv_labels = extract_embeddings(model, test_set, segment_length, test_strategy, seed=1)
            eer = equal_error_rate(sv_embeddings, sv_labels)
            assert 0.0 <= eer <= 1.0

            mr = best_misclassification_rate(sv_embeddings, sv_labels)
            assert 0.0 <= mr <= 1.0


def test_train_keeps_best_dev_checkpoint_not_the_final_epoch(monkeypatch, small_config, synthetic_utterances):
    # Neururer et al. 2024's reported numbers come from the best dev-EER
    # checkpoint seen during training (context/src's EvalCallback +
    # get_reference_data), not from whatever state training happens to end
    # in. Here dev EER strictly worsens after the very first checkpoint, so
    # the model's weights after train() must match that first checkpoint,
    # not the fully-trained final epoch.
    utterances = synthetic_utterances(num_speakers=3, utterances_per_speaker=4)
    small_config.loss.type = "SOFTMAX"
    segment_length = small_config.data.segment_length(small_config.transformation)

    dataset = SegmentDataset(utterances, segment_length, "OS", seed=0)
    model = build_model(small_config)
    loss_module = build_loss(small_config, bottleneck_dim=512, num_speakers=3)

    eer_sequence = iter([0.05, 0.4, 0.6, 0.8, 0.9])
    monkeypatch.setattr(trainer, "equal_error_rate", lambda *a, **k: next(eer_sequence))

    captured_states = []
    real_deepcopy = copy.deepcopy

    def _capturing_deepcopy(obj, memo=None):
        is_top_level_call = memo is None  # recursive internal deepcopy calls pass a memo dict
        state = real_deepcopy(obj, memo)
        if is_top_level_call:
            captured_states.append(state)
        return state

    monkeypatch.setattr(trainer.copy, "deepcopy", _capturing_deepcopy)

    train(
        model, loss_module, dataset, small_config,
        dev_utterances=utterances, segment_length=segment_length, draw_strategy="OS",
    )

    # Only the first checkpoint ever improves on the running-best dev EER, so
    # exactly one checkpoint gets captured -- but training continues for two
    # more epochs afterwards, so this only passes if the final weights were
    # actually rolled back rather than left at the (worse) last epoch.
    assert len(captured_states) == 1
    first_checkpoint_state = captured_states[0]
    final_state = model.state_dict()
    for key in first_checkpoint_state:
        assert torch.equal(final_state[key], first_checkpoint_state[key])
