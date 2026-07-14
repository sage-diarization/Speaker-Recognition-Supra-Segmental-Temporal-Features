import itertools

from src.config import ExperimentConfig
from src.data.dataset import SegmentDataset
from src.evaluation.clustering import best_misclassification_rate
from src.evaluation.verification import equal_error_rate
from src.models.losses import build_loss
from src.models.registry import build_model
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
