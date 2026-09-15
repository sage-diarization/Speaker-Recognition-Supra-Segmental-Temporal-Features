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
    # "TIMIT" (default) or "VoxCeleb" -- selects the corpus + evaluation task
    # set in src/experiment.py, and is folded into wandb run names/tags so
    # runs from different datasets are distinguishable at a glance.
    dataset: str = "TIMIT"
    segment_duration: float = 1.0
    timit_root: str = ""
    archive_path: str = ""
    download_url: str = ""
    cache_dir: str = "~/.cache/sst-experiment/timit"

    def segment_length(self, transformation: TransformationConfig):
        return int(self.segment_duration * transformation.steps_per_second)


@dataclass
class VoxCelebConfig:
    """torchaudio.datasets.VoxCeleb1Verification-backed corpus (see
    src/data/voxceleb.py). Substitutes for Neururer et al. 2024's actual
    VoxCeleb2-train/VoxCeleb1-test protocol, since torchaudio ships no
    VoxCeleb2 downloader: training utterances instead come from VoxCeleb1's
    own speakers, excluding whichever speakers appear in the verification
    trial list so train/eval speakers stay disjoint (the standard open-set
    setup) -- see src/data/voxceleb.py's module docstring for why that
    constrains trial_meta_url to the *original* 40-speaker list rather than
    the "hard"/"extended" VoxSRC lists."""

    root: str = "~/.cache/sst-experiment/voxceleb"
    # The verification trial-pairs list to download+evaluate against
    # (default: the cleaned original 40-held-out-speaker list). Passed
    # straight through to VoxCeleb1Verification's meta_url.
    trial_meta_url: str = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"


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
    # Directory persisted checkpoints and resume manifests are written under
    # (see src/training/trainer.py's checkpoint/resume support and
    # src/experiment.py's per-sweep manifest).
    checkpoint_dir: str = "checkpoints"
    # Disk-checkpoint cadence: a checkpoint is written every this-many epochs
    # (and always on the epoch that triggers early stopping or completes
    # training), so a crashed run never has to redo more than this many
    # epochs of dev-eval history. TIMIT's cheap epochs tolerate a coarser
    # cadence (default 25); VoxCeleb's expensive epochs should checkpoint
    # every single one (override to 1 in VoxCeleb configs).
    checkpoint_every_epochs: int = 25
    # Stop training once dev EER hasn't improved for this many epochs. The
    # paper itself never stops early -- it always trains the full fixed
    # schedule and reports the best dev checkpoint after the fact -- this is
    # a pragmatic compute-saving addition on top of that. Set to null/None to
    # disable early stopping and match the paper's protocol exactly (every
    # run trains the full num_epochs; the best dev checkpoint is still what
    # gets kept/reported).
    early_stopping_patience: int | None = 15


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
    voxceleb: VoxCelebConfig = field(default_factory=VoxCelebConfig)
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
            ("voxceleb", VoxCelebConfig),
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
