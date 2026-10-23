import numpy as np
import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel, WavLMConfig, WavLMModel

from src.config import TransformationConfig
from src.data.dataset import featurize_waveform
from src.data.lazy_features import expected_frame_count
from src.models import pretrained_wavlm
from src.models.pretrained_w2vbert import W2VBertSV
from src.models.pretrained_wavlm import WavLMECAPA, _wavlm_state_dict
from src.models.pretrained_wespeaker import WeSpeakerResNet34

# Randomly initialized, same architecture families as the released
# checkpoints, so these tests need no download.


def _tiny_wavlm():
    config = WavLMConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=True, feat_extract_norm="layer", conv_bias=False,
    )
    return WavLMModel(config).eval()


def _tiny_w2vbert():
    config = Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        output_hidden_size=32, conv_depthwise_kernel_size=3,
    )
    return Wav2Vec2BertModel(config).eval()


@pytest.mark.parametrize("frames", [98, 298])
def test_wespeaker_resnet34_embeds_any_segment_length(frames):
    model = WeSpeakerResNet34().eval()
    output = model(torch.randn(3, 1, frames, 80))
    assert output.backend.shape == (3, 256)
    assert torch.equal(output.backend, output.bottleneck)


def test_w2vbert_sv_feeds_every_hidden_state_through_its_adapter():
    encoder = _tiny_w2vbert()
    model = W2VBertSV(encoder, adapter_dim=8, embed_dim=16).eval()
    x = torch.randn(2, 1, 49, 160)
    with torch.no_grad():
        output = model(x)
        # The layer-by-layer loop sees the same hidden states as transformers' own forward.
        hidden_states = encoder(input_features=x.squeeze(1), output_hidden_states=True).hidden_states
        adapted = torch.cat([adapter(h) for adapter, h in zip(model.adapter_layers, hidden_states)], dim=-1)
        expected = model.bottleneck(model.pooling(adapted))
    assert len(hidden_states) == len(model.adapter_layers) == 3
    assert output.backend.shape == (2, 16)
    assert torch.allclose(output.backend, expected, atol=1e-5)


def test_wavlm_ecapa_weights_the_layer_inputs_and_the_layer_normed_output():
    wavlm = _tiny_wavlm()
    model = WavLMECAPA(wavlm, channels=16, embed_dim=8).eval()
    with torch.no_grad():
        model.feature_weight.normal_()
        waveform = torch.randn(2, 16000)
        frames = wavlm.feature_extractor(waveform).transpose(1, 2)
        # transformers' hidden_states end with the last layer's output before
        # the encoder's layer norm; s3prl's (which the ECAPA-TDNN was trained
        # on) with the output after it.
        output = wavlm(waveform, output_hidden_states=True)
        states = [*output.hidden_states[:-1], output.last_hidden_state]
        weights = torch.softmax(model.feature_weight, dim=0)
        expected = sum(w * h for w, h in zip(weights, states))
        assert torch.allclose(model.weighted_hidden_states(frames), expected, atol=1e-5)
        assert model(frames.unsqueeze(1)).backend.shape == (2, 8)


def test_wavlm_checkpoint_names_map_onto_transformers():
    renamed = _wavlm_state_dict({
        "feature_extract.model.feature_extractor.conv_layers.3.0.weight": 1,
        "feature_extract.model.feature_extractor.conv_layers.3.2.1.bias": 2,
        "feature_extract.model.layer_norm.weight": 3,
        "feature_extract.model.post_extract_proj.bias": 4,
        "feature_extract.model.encoder.pos_conv.0.weight_g": 5,
        "feature_extract.model.encoder.layers.0.self_attn.relative_attention_bias.weight": 6,
        "feature_extract.model.encoder.layers.7.self_attn.grep_a": 7,
        "feature_extract.model.encoder.layers.7.self_attn.q_proj.weight": 8,
        "feature_extract.model.encoder.layers.7.self_attn_layer_norm.weight": 9,
        "feature_extract.model.encoder.layers.7.fc1.weight": 10,
        "feature_extract.model.encoder.layer_norm.bias": 11,
        "feature_extract.model.mask_emb": 12,
        "layer1.conv.weight": 13,
    })
    assert renamed == {
        "feature_extractor.conv_layers.3.conv.weight": 1,
        "feature_extractor.conv_layers.3.layer_norm.bias": 2,
        "feature_projection.layer_norm.weight": 3,
        "feature_projection.projection.bias": 4,
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0": 5,
        "encoder.layers.0.attention.rel_attn_embed.weight": 6,
        "encoder.layers.7.attention.gru_rel_pos_const": 7,
        "encoder.layers.7.attention.q_proj.weight": 8,
        "encoder.layers.7.layer_norm.weight": 9,
        "encoder.layers.7.feed_forward.intermediate_dense.weight": 10,
        "encoder.layer_norm.bias": 11,
    }


def _waveform(num_samples):
    return (np.random.default_rng(0).standard_normal(num_samples) * 0.1).astype(np.float32)


@pytest.mark.parametrize("num_samples", [16000, 16159, 23456])
def test_wespeaker_fbank_frame_count_matches_expected_frame_count(num_samples):
    transformation = TransformationConfig(type="wespeaker_fbank", frame_length_s=0.025, frame_step_s=0.01)
    features = featurize_waveform(_waveform(num_samples), transformation)
    assert features.shape == (expected_frame_count(num_samples, transformation), transformation.num_freqs)
    assert np.allclose(features.mean(axis=0), 0, atol=1e-4)  # utterance mean-normalized
    if num_samples == 16000:
        assert features.shape[0] == transformation.steps_per_second


@pytest.mark.parametrize("num_samples", [16000, 16159, 23456])
def test_wavlm_cnn_frame_count_matches_expected_frame_count(monkeypatch, num_samples):
    monkeypatch.setattr(pretrained_wavlm, "wavlm_feature_encoder", lambda checkpoint: _tiny_wavlm().feature_extractor)
    transformation = TransformationConfig(
        type="wavlm_cnn", frame_length_s=0.025, frame_step_s=0.02, wavlm_checkpoint="unused"
    )
    features = featurize_waveform(_waveform(num_samples), transformation)
    assert features.shape == (expected_frame_count(num_samples, transformation), transformation.num_freqs)
    if num_samples == 16000:
        assert features.shape[0] == transformation.steps_per_second


def test_pretrained_transformations_reject_other_frame_parameters():
    with pytest.raises(ValueError, match="wespeaker_fbank"):
        TransformationConfig(type="wespeaker_fbank")
    with pytest.raises(ValueError, match="wavlm_cnn"):
        TransformationConfig(type="wavlm_cnn", frame_length_s=0.025, frame_step_s=0.01, wavlm_checkpoint="x")
    with pytest.raises(ValueError, match="wavlm_checkpoint"):
        TransformationConfig(type="wavlm_cnn", frame_length_s=0.025, frame_step_s=0.02)


def test_block_span_counts_each_front_ends_own_frame_step():
    spans_ms = {
        transformation.type: [round(transformation.block_span_s(b) * 1000) for b in (2, 4)]
        for transformation in (
            TransformationConfig(),  # mel: 64 ms frames every 10 ms
            TransformationConfig(type="wespeaker_fbank", frame_length_s=0.025, frame_step_s=0.01),
            TransformationConfig(type="w2vbert", frame_length_s=0.025, frame_step_s=0.01),  # 2 stacked fbank frames
            TransformationConfig(type="wavlm_cnn", frame_length_s=0.025, frame_step_s=0.02, wavlm_checkpoint="x"),
        )
    }
    assert spans_ms == {"mel": [74, 94], "wespeaker_fbank": [35, 55], "w2vbert": [55, 95], "wavlm_cnn": [45, 85]}
