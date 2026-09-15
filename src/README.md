# Supra-segmental temporal feature test (CNN / RNN / ResNet / Conformer)

PyTorch reimplementation of the Section 2 time-scrambling test from
Neururer et al. 2024 ("Deep neural networks for automatic speaker
recognition do not learn supra-segmental temporal features"). See
`../context/docs/` for the paper and `../context/src/` for the original
TensorFlow reference implementation this was ported from.

Four speaker-embedding backends are available via `model.type`:

- `"CNN"` (default): a direct port of `context/src/models/backend/CNN.py`,
  the CNN [12] backend of Neururer et al. 2024's Table 1/2.
- `"RNN"`: a direct port of `context/src/models/backend/LSTM.py` (two
  stacked bidirectional LSTMs), the RNN [13] backend. Shares the CNN's mel
  front-end, so it reuses `configs/cnn_timit.yaml`'s `transformation:`
  defaults (see `configs/rnn_timit.yaml`).
- `"ResNet"`: a direct port of `context/src/models/backend/ResNet34s.py` +
  its GhostVLAD aggregation (`context/src/models/aggregation/GhostVlad.py`),
  the ResNet [27] backend. Unlike the other three backends, its original
  front-end is a raw (non-mel) magnitude spectrogram with a hamming window
  (`context/src/00_configs/01_transformation/ResNet.json`) -- see
  `configs/resnet_timit.yaml`'s `transformation:`/`resnet:` sections and
  `src/models/resnet.py`'s module docstrings. The paper's fourth model,
  F-ResNet (Fast ResNet-34, sourced from the external
  `github.com/clovaai/voxceleb_trainer` rather than `context/src`, and
  called "out of competition" on TIMIT due to its VoxCeleb-tuned front-end),
  isn't ported here.
- `"Conformer"`: a from-scratch PyTorch implementation of the Gulati et al.
  2020 Conformer encoder ("Conformer: Convolution-augmented Transformer for
  Speech Recognition", see `../context/docs/`), adapted from an ASR encoder
  into a speaker embedding backend by mean-pooling the encoder output over
  time. See `src/models/conformer.py`'s module docstring for architecture
  details and configs/conformer_timit.yaml for its hyperparameters
  (`conformer:` section).

All four expose the same `(backend, bottleneck)` output contract (see
"Adding a new model backend" below), so switching `model.type` swaps only
the embedding backbone -- everything else (loss, training loop, evaluation)
stays identical, which is what makes the runs comparable. The one exception
is ResNet's own front-end transformation (mel vs. linear spectrogram),
which the original also varies per model for the same reason (Section 2.2:
front-end/hyperparameters are kept faithful to each model's own source
paper, not unified across models).

Trains a speaker embedding model for each of the three training-time
segment-draw strategies (OS/SS/SU), repeated `num_runs` times per strategy
(default 5, see `configs/cnn_timit.yaml`), and evaluates each run against all
three test-time strategies. On TIMIT (`data.dataset: "TIMIT"`, the default),
reports speaker verification EER and speaker clustering MR as mean/SD over
those runs in a 3x3 grid (reproducing the structure and mean/SD reporting of
the paper's Tables 1/2, not the exact numbers — the original list-file
splits for dev/final partitions aren't recoverable from the reference repo).
On VoxCeleb (`data.dataset: "VoxCeleb"`, see "Running on VoxCeleb" below),
only SV/EER is reported, matching the paper's Table 3 and its stated reason
for omitting SC there ("experiments in Section 2 led to similar
conclusions" for both tasks).

Each run evaluates dev-set SV EER after *every* epoch and keeps the
best-scoring checkpoint's weights, matching `context/src`'s `EvalCallback` +
`get_reference_data`: the paper's Tables 1/2 numbers come from that best dev
checkpoint, not from whatever state training happens to end in. The dev set
reuses the held-out TEST-split utterances (the same pool the "full" final
numbers are drawn from), since the original's own `development` list is
itself a subset of that pool rather than a disjoint speaker split.

Training stops early once dev EER hasn't improved for
`training.early_stopping_patience` epochs (default 15) -- a pragmatic
compute-saving addition on top of the paper's own methodology, which always
trains the full fixed schedule and only picks the best checkpoint after the
fact. A checkpoint is written to disk every `training.checkpoint_every_epochs`
epochs (default 25; set to 1 for datasets with expensive epochs, see the
VoxCeleb configs) under `training.checkpoint_dir`, and automatically resumed
from -- including full bit-exact random state (torch's global RNG, the
segment-draw RNG, and the dev-eval RNG) -- if that file already exists, so a
crashed or requeued SLURM job continues the exact same training run rather
than restarting it. Independently of that periodic cadence, a lightweight
`<checkpoint>.best.pt` companion file is rewritten on every dev-EER
improvement, so the best weights on disk are never more than one epoch stale
even if the process is killed between two periodic checkpoints. Beyond the
single in-progress run, `run_experiment`
itself tracks which of a strategy's `num_runs` repeats are already complete
in a small manifest alongside the checkpoints, so re-invoking
`python -m src.experiment --config ...` after a crash skips straight past
finished repeats instead of retraining them.

- **OS** (Original Segment): a contiguous crop — has both FBA and SST.
- **SS** (Shuffled within Segment): the OS crop with frame order destroyed.
- **SU** (Shuffled within Utterance): frames drawn from a wider window.

## Getting TIMIT

TIMIT is an LDC-licensed corpus (catalog.ldc.upenn.edu/LDC93S1) — there is
no free redistribution mirror this project can hardcode a download for.
Point the pipeline at your own licensed copy via `configs/cnn_timit.yaml`
(or the equivalent env vars), any one of:

- `data.timit_root` / `TIMIT_ROOT`: an already-extracted TIMIT directory
  (containing `TRAIN/` and `TEST/`, standard NIST layout).
- `data.archive_path` / `TIMIT_ARCHIVE_PATH`: a local `.zip` of your copy —
  extracted once and cached under `data.cache_dir`.
- `data.download_url` / `TIMIT_DOWNLOAD_URL`: a URL to a copy you control
  (e.g. your own S3/blob storage) — downloaded and cached on first use.

## Running on VoxCeleb

Set `data.dataset: "VoxCeleb"` (see `configs/cnn_voxceleb.yaml`) to run
against VoxCeleb instead of TIMIT. `src/data/voxceleb.py`'s `VoxCelebCorpus`
fetches and caches the corpus automatically via torchaudio's own downloader
(`torchaudio.datasets.VoxCeleb1Verification`, `voxceleb.root`) — no manual
download/credential step needed, unlike TIMIT.

**This is a deliberate deviation from Neururer et al. 2024's actual
protocol**, worth understanding before comparing numbers to the paper: the
paper trains on VoxCeleb2 (5,994 speakers) and evaluates on VoxCeleb1's
"hard" test set (Section 3.2). torchaudio ships no VoxCeleb2 downloader, so
this project instead trains on VoxCeleb1's *own* speakers, excluding
whichever 40 speakers appear in the verification trial list so train/eval
speakers stay disjoint (the standard open-set setup) — i.e. a
VoxCeleb1-train/VoxCeleb1-test substitute for the paper's
VoxCeleb2-train/VoxCeleb1-test protocol. This is also why
`voxceleb.trial_meta_url` must stay the *original* `veri_test2.txt` list (40
held-out speakers) rather than the "hard"/"extended" VoxSRC lists
(`list_test_hard2.txt` / `list_test_all2.txt`): fetched and inspected
directly while building this, those span 1,190 of VoxCeleb1's 1,251
speakers, so excluding their speakers from training would leave almost
nothing to train on, and *not* excluding them would leak most training
speakers into evaluation. See `src/data/voxceleb.py`'s module docstring for
the full detail.

Two further consequences of VoxCeleb's scale and structure vs. TIMIT's,
both handled automatically but worth knowing about:

- **Lazy featurization.** VoxCeleb1 has ~148k training utterances (vs.
  TIMIT's ~5,500); eagerly featurizing all of them upfront (as TIMIT's path
  does) would need tens of GB of RAM. `src/data/lazy_features.py`'s
  `LazyFeatures` instead defers each utterance's featurization to first
  access (during training/eval), using just its raw sample count (a cheap
  file-header probe, not a full decode) to filter out utterances shorter
  than a training segment upfront.
- **No cap on hop-window count for evaluation.** VoxCeleb utterances vary
  from a few seconds to several minutes (unlike TIMIT's uniform ~3s
  sentences); `extract_embeddings`' hop-window averaging (see "Tracking
  progress with wandb" below and its own docstring) generates one window per
  ~0.5s of audio with no upper bound, matching `context/src/setup/setup.py`'s
  equally uncapped original behavior. A handful of very long utterances in
  the trial list could therefore make a single embedding-extraction call
  noticeably more expensive than the rest. `evaluation.sv_max_sentences`
  exists as an unused config hook if this proves impractical in practice and
  a cap is wanted.

## Running the experiment

```
pip install -r requirements.txt
python -m src.experiment --config configs/cnn_timit.yaml         # CNN backend on TIMIT
python -m src.experiment --config configs/rnn_timit.yaml         # RNN backend on TIMIT
python -m src.experiment --config configs/resnet_timit.yaml      # ResNet backend on TIMIT
python -m src.experiment --config configs/conformer_timit.yaml   # Conformer backend on TIMIT
python -m src.experiment --config configs/cnn_voxceleb.yaml      # CNN backend on VoxCeleb
python -m src.experiment --config configs/rnn_voxceleb.yaml      # RNN backend on VoxCeleb
python -m src.experiment --config configs/resnet_voxceleb.yaml   # ResNet backend on VoxCeleb
python -m src.experiment --config configs/conformer_voxceleb.yaml   # Conformer backend on VoxCeleb
```

`device: "auto"` (the default, see `configs/cnn_timit.yaml`) picks the best
available accelerator at startup — cuda, then mps, then cpu — so the same
config runs unchanged on a SLURM GPU node, an Apple Silicon laptop, or a
CPU-only machine. Set it explicitly (e.g. `device: "cpu"`) to override.

## Tracking progress with wandb

Set `wandb.enabled: true` in the config to track runs (see
`configs/cnn_timit.yaml`). One wandb run covers each (model, dataset,
training-strategy) combination -- named e.g. `CNN-timit-OS` (or
`CNN-voxceleb-OS` for a VoxCeleb run) and tagged `[model.type, dataset,
strategy]` so runs from different models/datasets are distinguishable at a
glance -- and stays open across all of that strategy's
`num_runs` repeats, since the individual repeats aren't independently
interesting on their own (only their aggregate is). Each repeat's per-epoch
`train/loss`/`dev/EER` curves are logged under a `run{idx}/` -prefixed key so
they remain separately visible, and each repeat's final SV/SC numbers are
logged the same way (`run{idx}/final/SV_EER_test-*`) as soon as that repeat
finishes. Once all `num_runs` repeats are accounted for, the mean/SD
(`final/..._mean` / `final/..._std`, matching Tables 1/2 of Neururer et al.
2024) is logged to that same run's summary and the run is closed. If the
process crashes and `python -m src.experiment --config ...` is re-invoked,
the run is reattached (via an id persisted in the resume manifest, see
above) rather than starting a new one, so a strategy's wandb history stays
in one place across restarts.

On a SLURM cluster whose compute nodes have no internet access, set
`wandb.mode: "offline"` — logs are written locally and synced later with
`wandb sync <run-dir>` from a login node. `wandb.mode: "disabled"` fully
no-ops (useful for local smoke runs). When `wandb.enabled: false` (the
default), nothing wandb-related is imported or called at all.

## Adding a new model backend

Register a `build_<name>(config) -> nn.Module` in `src/models/registry.py`'s
`MODEL_REGISTRY`, where the module's `forward` returns a `(backend,
bottleneck)` pair (see `src/models/common.py`'s `BackendOutput`; backend =
evaluation embedding, bottleneck = the dimension the training loss operates
on -- CNN/RNN/Conformer use 1024-d/512-d, but a new backend isn't required
to: ResNet's original CUT=AGGREGATION setting means its backend and
bottleneck are the same 512-d tensor, see `src/models/resnet.py`). Set
`model.type` in the config to the new registry key, and
add a `<name>: ExperimentConfig` section if the model needs its own
hyperparameters (see `ConformerConfig` in `src/config.py`).
