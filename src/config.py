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

    @property
    def frame_length(self):
        return int(self.sample_rate * self.frame_length_s)

    @property
    def frame_step(self):
        return int(self.sample_rate * self.frame_step_s)

    @property
    def steps_per_second(self):
        return int((self.sample_rate + self.frame_step - self.frame_length) / self.frame_step)


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
    sc_utterances_per_speaker: int = 2


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
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # "auto" picks cuda > mps > cpu at runtime (see src/device.py); set explicitly
    # (e.g. "cpu") to override autodetection.
    device: str = "auto"

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
            ("loss", LossConfig),
            ("optimizer", OptimizerConfig),
            ("training", TrainingConfig),
            ("evaluation", EvaluationConfig),
            ("wandb", WandbConfig),
        ):
            kwargs[section_name] = section_cls(**raw.get(section_name, {}))
        kwargs["device"] = raw.get("device", "auto")
        return cls(**kwargs)

    def to_dict(self):
        return asdict(self)
