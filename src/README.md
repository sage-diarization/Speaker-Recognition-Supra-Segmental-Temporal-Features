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
splits for dev/final partitions aren't recoverable from the reference repo;
the SC speaker list is, from the paper's reference [13] -- see
`evaluation.sc_speakers` in the TIMIT configs).
On VoxCeleb (`data.dataset: "VoxCeleb"`, see "Running on VoxCeleb" below),
AISHELL-4 (`data.dataset: "AISHELL4"`, see "Running on AISHELL-4" below) and
TidyVoiceX (`data.dataset: "TidyVoiceX"`, see "Running on TidyVoiceX" below),
only SV/EER is reported, matching the paper's Table 3 and its stated reason
for omitting SC on VoxCeleb ("experiments in Section 2 led to similar
conclusions" for both tasks) — extended here to AISHELL-4 for the same
reason (see "Running on AISHELL-4" below), and to TidyVoiceX because its
terms permit verification use only.

Each run evaluates dev-set SV EER after *every* epoch and keeps the
best-scoring checkpoint's weights, matching `context/src`'s `EvalCallback` +
`get_reference_data`: the paper's Tables 1/2 numbers come from that best dev
checkpoint, not from whatever state training happens to end in. On TIMIT,
the configs set `evaluation.dev_speakers` to the standard 50-speaker TIMIT
development set (TEST-split speakers never trained on; Kaldi's
`egs/timit/s5/conf/dev_spk.list`), the closest match to context/src's
`development` SV list (`00_configs/04_evaluation/TIMIT-00_ORIGINAL.json`,
whose list file isn't in the reference repo), and all TRAIN utterances are
trained on. As in the paper, final SV is scored on the *full* TEST split,
so it includes these 50 speakers. With `training.paper_checkpoint_epochs`
(set in the TIMIT configs) only the 11 epochs `np.linspace(0, 127, 11)`
that context/src's `EvalCallback` evaluates are best-checkpoint candidates.
Without `dev_speakers`, the dev set falls back to
`evaluation.dev_holdout_per_speaker` (default 2) of each TRAIN speaker's
own utterances (see `experiment.py`'s `_featurize_train_dev_split`) -- on
TIMIT these are SA1/SA2, which have identical text for every speaker.

Training stops early once dev EER hasn't improved for
`training.early_stopping_patience` epochs (default 15) -- a pragmatic
compute-saving addition on top of the paper's own methodology, which always
trains the full fixed schedule and only picks the best checkpoint after the
fact. Set `training.early_stopping_patience` to `null` in the YAML config to
disable early stopping and match that protocol exactly (every run trains the
full `training.num_epochs`; the best dev checkpoint is still what gets
kept/reported). A checkpoint is written to disk every `training.checkpoint_every_epochs`
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

Set `data.dataset: "VoxCeleb"` (see `configs/cnn-voxceleb.yaml`) to run
against VoxCeleb instead of TIMIT. The VoxCeleb configs follow Neururer et
al. 2024's protocol (Section 3.2, `context/src`'s
`04_evaluation/VOX-00_ORIGINAL.json`):

- **Training** on VoxCeleb2 dev (5,994 speakers, ~1.09M utterances) from
  `voxceleb.vox2_root`. VoxCeleb2 can't be auto-downloaded and ships as AAC
  (`.m4a`): download it yourself and convert it to 16 kHz mono WAV (e.g.
  with ffmpeg), keeping the `<speaker>/<video>/<utterance>.wav` layout (at
  any depth under `vox2_root`).
- **Checkpoint selection** on VoxCeleb1-O cleaned (`voxceleb.trial_meta_url`,
  `veri_test2.txt`), at the paper's 11 epochs (`training.paper_checkpoint_epochs`),
  over the full 128 epochs (no early stopping).
- **Reported SV** on VoxCeleb1-H cleaned (`voxceleb.eval_trial_meta_url`,
  `list_test_hard2.txt`, ~550k trials).

VoxCeleb1 itself (both trial lists' audio) is fetched and cached
automatically under `voxceleb.root` via torchaudio's own downloader
(`torchaudio.datasets.VoxCeleb1Verification`).

As in the paper, an epoch draws one 1 s segment per training utterance, not
every frame of it. Training reads only the samples behind each drawn segment
(`LazyFeatures`' `frames_fn`, see `src/data/dataset.py`'s
`featurize_frame_range`; equal to featurizing the whole file then slicing,
up to float32 rounding) rather than decoding whole files, spread over
`training.num_workers` DataLoader workers (8 in the VoxCeleb configs, matched
by the SLURM jobs' `--cpus-per-task`). Every utterance's length is needed
upfront; the first run lists both datasets and reads each file's header in
parallel (with progress output), then caches the result as
`.sst-wav-index.tsv` in `vox2_root` and `voxceleb.root`, so later runs start
in seconds. Delete those files if a dataset changes. The full run is longer than the SLURM
jobs' 24h limit, so resubmit the job: it resumes from `checkpoint_every_epochs: 1`.

**VoxCeleb1-only substitute.** Removing `vox2_root` and `eval_trial_meta_url`
falls back to training on VoxCeleb1's *own* speakers, excluding whichever
speakers appear in the trial list(s) so train/eval speakers stay disjoint,
and selecting and reporting on that same list -- **a deliberate deviation
from the paper**. In that mode `voxceleb.trial_meta_url` must stay the
*original* `veri_test2.txt` list (40 held-out speakers) rather than the
"hard"/"extended" VoxSRC lists (`list_test_hard2.txt` / `list_test_all2.txt`):
fetched and inspected directly while building this, those span 1,190 of
VoxCeleb1's 1,251 speakers, so excluding their speakers from training would
leave almost nothing to train on, and *not* excluding them would leak most
training speakers into evaluation. See `src/data/voxceleb.py`'s module
docstring for the full detail.

Two further consequences of VoxCeleb's scale and structure vs. TIMIT's,
both handled automatically but worth knowing about:

- **Lazy featurization.** VoxCeleb has ~148k (VoxCeleb1) or ~1.09M (VoxCeleb2) training utterances (vs.
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

## Running on AISHELL-4

Set `data.dataset: "AISHELL4"` (see `configs/cnn-aishell4.yaml`) to run
against AISHELL-4 (Fu et al. 2021, `openslr.org/111`) instead of TIMIT.
Unlike TIMIT, AISHELL-4 itself is freely downloadable with no license
agreement -- but this project doesn't auto-fetch it (its archives are
multi-gigabyte 8-channel meeting recordings). Download and extract it
yourself, then point `aishell4.root` (or the `AISHELL4_ROOT` env var) at the
result -- a directory containing `train_S/`, `train_M/` and/or `train_L/`
(AISHELL-4's three subsets, split by recording-room size; however many of
these your copy has are combined into one TRAIN split), plus `test/`, each
with `wav/` and `TextGrid/` subdirectories.

**AISHELL-4 isn't shaped like TIMIT/VoxCeleb**, and this project's
adaptation of it is a deliberate simplification worth understanding before
comparing numbers across datasets -- see `src/data/aishell4.py`'s module
docstring for the full detail. In short:

- AISHELL-4 ships long multi-speaker meeting recordings (one 8-channel
  `.wav` per session) plus a per-session `TextGrid` annotating who spoke
  when, not pre-segmented per-speaker utterance files. Each speaker's
  "utterances" are the individual non-silence intervals of its TextGrid
  tier, sliced directly out of the session's first audio channel (no
  beamforming) by sample offset -- no separate segment files are ever
  written to disk.
- AISHELL-4's TextGrid tiers are speaker labels *local to their own
  session* -- there's no corpus-wide speaker-ID metadata linking sessions to
  real identities. "Speaker" is therefore defined as `(session_id,
  tier_name)`, and train/test speaker-disjointness comes for free from
  AISHELL-4's own `train_*/` vs `test/` session split (the same real,
  per-speaker TRAIN/TEST structure TIMIT has, unlike VoxCeleb's trial-pairs
  protocol), so SV evaluation is TIMIT's exhaustive all-vs-all pairing, not
  a trial list.
- SC is omitted, the same way it's omitted for VoxCeleb: Neururer et al.
  2024's "2 vs 8 concatenated sentences per speaker" SC recipe doesn't
  transfer to AISHELL-4's TextGrid-turn segments, whose count/duration per
  speaker is far more irregular than TIMIT's uniform ~10 sentences.
- Segment counts run much higher than TIMIT's ~5,500 utterances, so, like
  VoxCeleb, training/TEST utterances are handled via
  `src/data/lazy_features.py`'s `LazyFeatures` rather than featurized
  eagerly upfront.

Parsing the TextGrid annotations uses the
[`praatio`](https://github.com/timmahrt/praatIO) library (see
`requirements.txt`) rather than a hand-rolled parser, per this project's
existing preference for standardized libraries over custom fetchers/parsers
(the same reasoning `src/data/voxceleb.py`'s module docstring gives for
using torchaudio's own VoxCeleb downloader).

## Running on TidyVoiceX

Set `data.dataset: "TidyVoiceX"` (see `configs/{cnn,rnn,resnet,conformer}-tidyvoicex.yaml`
and the matching `slurm/*-tidyvoicex-{os,ss}.submit` jobs) to run against
TidyVoiceX_ASV, the multilingual Common Voice derivative of the TidyVoice
2026 cross-lingual speaker-verification challenge
([Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmihtsewu023so207xot1iqqw),
reference recipe:
[wespeaker `examples/tidyvocie`](https://github.com/areffarhadi/wespeaker/tree/master/examples/tidyvocie)).
Its download needs a Data Collective API key, so it isn't auto-fetched:

1. Download and extract the dataset archive into `tidyvoicex.root`
   (default `~/.cache/datasets/tidyx`). Its `TidyVoiceX_Train/` and
   `TidyVoiceX_Dev/` directories (`<speaker>/<language>/<utterance>.wav`)
   are located at whatever nesting depth they ended up in.
2. The Dev trial list is **not** part of that archive: download
   `tidyvoice_trials.zip` (linked from the challenge site / the recipe's
   `local/download_tidyvoice.sh`) and unzip its
   `TidyVocieX_Dev_trialPairs.txt` (sic) anywhere under `tidyvoicex.root`, or
   set `tidyvoicex.trial_file` to its path.

The protocol is VoxCeleb-shaped: training uses all 3,666 Train speakers, and
SV is scored over the official Dev trial list (808 speakers disjoint from
Train; 12M trials, 1:2 target:nontarget, half same-language/half
cross-language). There are three differences from the VoxCeleb path:

- **Checkpoint selection never sees the evaluation trials.**
  `evaluation.dev_holdout_per_speaker` utterances per Train speaker are held
  out of training (as for TIMIT/AISHELL-4) and scored over a small generated
  trial list (each speaker's held-out pair as target, paired with the next
  speaker's as nontarget), because exhaustive all-vs-all pairing over 3,666
  speakers would be ~27M pairs per epoch.
- **Short clips are dropped.** Common Voice clips can be shorter than one
  training segment, which `extract_embeddings` can't embed. Trials referencing
  such clips are excluded from scoring, and the count is printed at
  startup.
- **Chunked scoring.** The 12M trials are held as index arrays
  (`IndexedTrials`) and scored in chunks by
  `indexed_trial_equal_error_rate`. Parsing takes about 7s and scoring
  about 15s, at roughly 1.3 GB peak.

## Running the experiment

```
pip install -r requirements.txt
python -m src.experiment --config configs/cnn-timit.yaml         # CNN backend on TIMIT
python -m src.experiment --config configs/rnn-timit.yaml         # RNN backend on TIMIT
python -m src.experiment --config configs/resnet-timit.yaml      # ResNet backend on TIMIT
python -m src.experiment --config configs/conformer-timit.yaml   # Conformer backend on TIMIT
python -m src.experiment --config configs/cnn-voxceleb.yaml      # CNN backend on VoxCeleb
python -m src.experiment --config configs/rnn-voxceleb.yaml      # RNN backend on VoxCeleb
python -m src.experiment --config configs/resnet-voxceleb.yaml   # ResNet backend on VoxCeleb
python -m src.experiment --config configs/conformer-voxceleb.yaml   # Conformer backend on VoxCeleb
python -m src.experiment --config configs/cnn-aishell4.yaml      # CNN backend on AISHELL-4
python -m src.experiment --config configs/rnn-aishell4.yaml      # RNN backend on AISHELL-4
python -m src.experiment --config configs/resnet-aishell4.yaml   # ResNet backend on AISHELL-4
python -m src.experiment --config configs/conformer-aishell4.yaml   # Conformer backend on AISHELL-4
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
