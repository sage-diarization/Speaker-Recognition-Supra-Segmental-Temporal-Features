# Supra-segmental temporal feature test (CNN / Conformer)

PyTorch reimplementation of the Section 2 time-scrambling test from
Neururer et al. 2024 ("Deep neural networks for automatic speaker
recognition do not learn supra-segmental temporal features"). See
`../context/docs/` for the paper and `../context/src/` for the original
TensorFlow reference implementation this was ported from.

Two speaker-embedding backends are available via `model.type`:

- `"CNN"` (default): a direct port of `context/src/models/backend/CNN.py`,
  the backend Neururer et al. 2024 uses.
- `"Conformer"`: a from-scratch PyTorch implementation of the Gulati et al.
  2020 Conformer encoder ("Conformer: Convolution-augmented Transformer for
  Speech Recognition", see `../context/docs/`), adapted from an ASR encoder
  into a speaker embedding backend by mean-pooling the encoder output over
  time. See `src/models/conformer.py`'s module docstring for architecture
  details and configs/conformer_timit.yaml for its hyperparameters
  (`conformer:` section).

Both expose the same `(backend, bottleneck)` output contract (see "Adding a
new model backend" below), so switching `model.type` swaps only the
embedding backbone -- everything else (loss, training loop, evaluation)
stays identical, which is what makes the two runs comparable.

Trains a speaker embedding model for each of the three training-time
segment-draw strategies (OS/SS/SU), repeated `num_runs` times per strategy
(default 5, see `configs/cnn_timit.yaml`), and evaluates each run against all
three test-time strategies. Reports speaker verification EER and speaker
clustering MR as mean/SD over those runs in a 3x3 grid (reproducing the
structure and mean/SD reporting of the paper's Tables 1/2, not the exact
numbers — the original list-file splits for dev/final partitions aren't
recoverable from the reference repo).

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

## Running the experiment

```
pip install -r requirements.txt
python -m src.experiment --config configs/cnn_timit.yaml         # CNN backend
python -m src.experiment --config configs/conformer_timit.yaml   # Conformer backend
```

`device: "auto"` (the default, see `configs/cnn_timit.yaml`) picks the best
available accelerator at startup — cuda, then mps, then cpu — so the same
config runs unchanged on a SLURM GPU node, an Apple Silicon laptop, or a
CPU-only machine. Set it explicitly (e.g. `device: "cpu"`) to override.

## Tracking progress with wandb

Set `wandb.enabled: true` in the config to track runs (see
`configs/cnn_timit.yaml`). One wandb run is started per training-strategy
per repeat (OS/SS/SU x `num_runs`), each with a live per-epoch `train/loss`
curve and, once training finishes, the resulting SV/EER and SC/MR numbers
against all three test strategies logged to that run's summary — this
mirrors `context/src/train.py`'s per-run `wandb.init` + `EvalCallback`
logging, minus the Keras-specific weight-histogram logging. Once all
`num_runs` repeats of a training-strategy finish, one further
`{model}-{strategy}-aggregate` wandb run logs the mean/SD of those repeats
(`final/..._mean` / `final/..._std`), matching the mean/SD reported in
Tables 1/2 of Neururer et al. 2024.

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
on -- both CNN and Conformer use 1024-d/512-d, but a new backend isn't
required to). Set `model.type` in the config to the new registry key, and
add a `<name>: ExperimentConfig` section if the model needs its own
hyperparameters (see `ConformerConfig` in `src/config.py`).
