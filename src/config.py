from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class TransformationConfig:
    sample_rate: int = 16000
    frame_length_s: float = 0.064
    frame_step_s: float = 0.01
    nfft: int = 1024
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0
    # "mel" (default, matches CNN/RNN's DeepVoice front-end) or "linear" (raw
    # magnitude spectrogram, no mel filterbank -- matches ResNet's front-end,
    # context/src/00_configs/01_transformation/ResNet.json's "SPECTROGRAM" type).
    type: str = "mel"
    # "hann" (default) or "hamming" -- ResNet's front-end uses hamming
    # (context/src/setup/utils.py's window_map).
    window: str = "hann"

    @property
    def frame_length(self):
        return int(self.sample_rate * self.frame_length_s)

    @property
    def frame_step(self):
        return int(self.sample_rate * self.frame_step_s)

    @property
    def steps_per_second(self):
        return int((self.sample_rate + self.frame_step - self.frame_length) / self.frame_step)

    @property
    def num_freqs(self):
        return self.n_mels if self.type == "mel" else self.nfft // 2 + 1


@dataclass
class DataConfig:
    segment_duration: float = 1.0
    timit_root: str = ""
    archive_path: str = ""
    download_url: str = ""
    cache_dir: str = "~/.cache/sst-experiment/timit"

    def segment_length(self, transformation: TransformationConfig):
        return int(self.segment_duration * transformation.steps_per_second)


@dataclass
class ModelConfig:
    type: str = "CNN"
    allow_full: bool = False


@dataclass
class ConformerConfig:
    """Gulati et al. 2020 Conformer encoder hyperparameters. Defaults follow
    the paper's Conformer(S) dimensions (encoder_dim=144, num_heads=4,
    conv_kernel_size=31 -- the paper's Table 1 rounds this to 32, but the
    convolution module requires an odd kernel for symmetric SAME padding),
    except num_layers, which the paper sets to 16 for LibriSpeech ASR; that's
    substantially more compute than this project's short TIMIT segments
    warrant, so it defaults to 4 here (override in config to match the
    paper's scale)."""

    encoder_dim: int = 144
    num_layers: int = 4
    num_heads: int = 4
    ff_expansion_factor: int = 4
    conv_kernel_size: int = 31
    dropout: float = 0.1
    use_relative_positional_encoding: bool = True


@dataclass
class RNNConfig:
    """context/src/models/backend/LSTM.py hyperparameters (hidden_size=512
    per direction, matching the paper's RNN [13])."""

    hidden_size: int = 512


@dataclass
class ResNetConfig:
    """context/src/models/backend/ResNet34s.py + its GhostVLAD aggregation
    (context/src/00_configs/06_aggregation/GVLAD.json) hyperparameters."""

    vlad_clusters: int = 10
    ghost_clusters: int = 2
    bottleneck: int = 512


@dataclass
class LossConfig:
    type: str = "ANGULAR_MARGIN"
    margin_cosface: float = 0.3
    margin_arcface: float = 0.0
    margin_sphereface: float = 1.0
    scale: float = 40.0


@dataclass
class OptimizerConfig:
    type: str = "ADAM"
    learning_rate: float = 1e-4
    # L2 weight decay -- 0.0 (off) matches CNN/Conformer/RNN's original recipes;
    # ResNet's original config sets 1e-4 (context/src/00_configs/05_model/RES34S.json).
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    num_epochs: int = 128
    batch_size: int = 100
    segment_draw: str = "OS"
    num_speakers: int = 462


@dataclass
class EvaluationConfig:
    segment_draw: str = "OS"
    sv_max_sentences: int = 0
    sc_num_speakers: int = 40


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "speaker-verification"
    entity: str = ""
    # "online" needs network+login; "offline" writes locally for a later `wandb sync`
    # (useful on SLURM compute nodes without internet); "disabled" is a full no-op.
    mode: str = "online"
    tags: list = field(default_factory=list)


@dataclass
class ExperimentConfig:
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    resnet: ResNetConfig = field(default_factory=ResNetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # "auto" picks cuda > mps > cpu at runtime (see src/device.py); set explicitly
    # (e.g. "cpu") to override autodetection.
    device: str = "auto"
    # Neururer et al. 2024 (Section 2.3) train/test each strategy 5 times and
    # report mean and SD of EER/MR over those runs (Tables 1/2).
    num_runs: int = 5

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw):
        kwargs = {}
        for section_name, section_cls in (
            ("transformation", TransformationConfig),
            ("data", DataConfig),
            ("model", ModelConfig),
            ("conformer", ConformerConfig),
            ("rnn", RNNConfig),
            ("resnet", ResNetConfig),
            ("loss", LossConfig),
            ("optimizer", OptimizerConfig),
            ("training", TrainingConfig),
            ("evaluation", EvaluationConfig),
            ("wandb", WandbConfig),
        ):
            kwargs[section_name] = section_cls(**raw.get(section_name, {}))
        kwargs["device"] = raw.get("device", "auto")
        kwargs["num_runs"] = raw.get("num_runs", 5)
        return cls(**kwargs)

    def to_dict(self):
        return asdict(self)
