# Supra-segmental temporal feature test (CNN)

PyTorch reimplementation of the Section 2 time-scrambling test from
Neururer et al. 2024 ("Deep neural networks for automatic speaker
recognition do not learn supra-segmental temporal features"), for a CNN
backend. See `../context/docs/` for the paper and `../context/src/` for the
original TensorFlow reference implementation this was ported from.

Trains a CNN speaker embedding model three times (once per training-time
segment-draw strategy: OS/SS/SU) and evaluates each against all three
test-time strategies, reporting speaker verification EER and speaker
clustering MR in a 3x3 grid (reproducing the structure of the paper's
Tables 1/2, not the exact numbers — the original list-file splits for
dev/final partitions aren't recoverable from the reference repo).

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
python -m src.experiment --config configs/cnn_timit.yaml
```

`device: "auto"` (the default, see `configs/cnn_timit.yaml`) picks the best
available accelerator at startup — cuda, then mps, then cpu — so the same
config runs unchanged on a SLURM GPU node, an Apple Silicon laptop, or a
CPU-only machine. Set it explicitly (e.g. `device: "cpu"`) to override.

## Tracking progress with wandb

Set `wandb.enabled: true` in the config to track runs (see
`configs/cnn_timit.yaml`). One wandb run is started per training-strategy
(OS/SS/SU), each with a live per-epoch `train/loss` curve and, once training
finishes, the resulting SV/EER and SC/MR numbers against all three test
strategies logged to that run's summary — this mirrors
`context/src/train.py`'s per-run `wandb.init` + `EvalCallback` logging,
minus the Keras-specific weight-histogram logging.

On a SLURM cluster whose compute nodes have no internet access, set
`wandb.mode: "offline"` — logs are written locally and synced later with
`wandb sync <run-dir>` from a login node. `wandb.mode: "disabled"` fully
no-ops (useful for local smoke runs). When `wandb.enabled: false` (the
default), nothing wandb-related is imported or called at all.

## Adding a new model backend (e.g. Conformer)

Register a `build_<name>(config) -> nn.Module` in `src/models/registry.py`'s
`MODEL_REGISTRY`, where the module's `forward` returns a `(backend,
bottleneck)` pair (backend = evaluation embedding, bottleneck = the
dimension the training loss operates on). Set `model.type` in the config to
the new registry key.
