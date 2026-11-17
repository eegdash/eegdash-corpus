# EEGDash Corpus

**eegdash-corpus-v0** is 77 public EEG datasets served through [EEGDash](https://github.com/eegdash/EEGDash) from
OpenNeuro and NEMAR, converted once with the preprocessing recipe of the REVE EEG foundation model into one Zarr store
of 632 buckets, 14,074.7 hours and 6.21 billion tokens per epoch (3.6 TB). This repository holds the code and the
records that built it; the data is on Hugging Face, gated, at
[EEGDash/eegdash-corpus-v0](https://huggingface.co/datasets/EEGDash/eegdash-corpus-v0) (tag `v1.0`), and the corpus
is described in *EEGDash Corpus: A Training-Ready Public Corpus for EEG Foundation Models* (MLESP 2026, with IEEE
BIBM 2026; preprint [arXiv:2611.06356](https://arxiv.org/abs/2611.06356); see [Citation](#10-citation)).

| | eegdash-corpus-v0 | from |
|---|---:|---|
| datasets | 77 (56 OpenNeuro, 21 NEMAR) | `corpus/eegdash-corpus-v0.csv` |
| subjects | 7,272 | `corpus/eegdash-corpus-v0.csv` |
| sessions | 41,133 | `conversion/eegdash-corpus-v0-CONVERSION_REPORT.md` |
| hours of 200 Hz EEG | 14,074.7 | `conversion/eegdash-corpus-v0-CONVERSION_REPORT.md` |
| shards | 418 | `conversion/eegdash-corpus-v0-CONVERSION_REPORT.md` |
| buckets | 632 | `conversion/eegdash-corpus-v0-buckets.csv` |
| training samples (10 s, 2,000 time steps) | 5,045,111 | `conversion/eegdash-corpus-v0-buckets.csv` |
| tokens per epoch (samples × channels × 11 patches) | 6,206,191,189 | `conversion/eegdash-corpus-v0-buckets.csv` |
| files in the store | 44,801 | `conversion/eegdash-corpus-v0-SHA256SUMS` |

A bucket is every session of one dataset (within one shard) that shares one exact channel set; a bucket never spans
two datasets.

## 1. Quick start: pretrain REVE on the corpus

Three steps, the same on a workstation or a cluster; the tutorial page
[Pretraining REVE](https://eegdash.org/corpus/pretraining_reve.html) walks through them.

```bash
# 1. Download the corpus (3.6 TB), with the newest Hugging Face client in a small environment of its own
git clone https://github.com/eegdash/eegdash-corpus.git && cd eegdash-corpus
python3.11 -m venv ../hf-env && source ../hf-env/bin/activate
pip install -U huggingface_hub hf_xet
hf auth login                         # an account that accepted the dataset's terms on Hugging Face
bash download.sh /data
deactivate

# 2. Install the training environment and REVE's code with the patches, beside this repository
python3.11 -m venv ../reve-env && source ../reve-env/bin/activate
pip install -r requirements-pretraining.txt
git clone https://github.com/elouayas/reve_eeg.git ../reve_eeg && git -C ../reve_eeg checkout 06a7059
bash patches/apply_patches.sh ../reve_eeg

# 3. Pretrain REVE-Base with the paper's settings (17 epochs), on one machine
DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
```

**Download.** The download has its own environment because the training environment pins an older Hugging Face client (the one
REVE's code asks for), which spends one request per file: a free account gets 1,000 per five minutes, and the store
has 44,801 files. Everything lands side by side in the folder you start in: this repository, `hf-env`, `reve-env`
and `reve_eeg`. Where there is no `python3.11` command, any Python 3.11 does in its place in the two `venv` lines,
a conda environment's `python` for instance.

The download runs for hours, so start it where it survives a logout (`tmux`, `screen` or a job); run again after
an interruption, it goes on where it stopped. `VERIFY=1 JOBS=16 bash download.sh /data` also checks every file
against the published SHA-256 list and ends with `checksums OK`. On an HPC cluster the download took about
5.5 hours (near 200 MB/s) and the check about 3.5 hours.

**Pretrain on one machine.** Step 3 above: one line, and `pretrain.sh` uses every GPU the machine has.

```bash
DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
```

**Pretrain on several machines.** The paper's runs used 16 GPUs: 4 machines with 4 GPUs each. The command is run
once on every machine, four times for four machines, each time with three more variables in front. By hand, that is
a login to each machine; a job scheduler can start the four for you.

```bash
# on node01, the first machine
NNODES=4 NODE_RANK=0 MAIN_HOST=node01 DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
# on node02
NNODES=4 NODE_RANK=1 MAIN_HOST=node01 DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
# on node03
NNODES=4 NODE_RANK=2 MAIN_HOST=node01 DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
# on node04
NNODES=4 NODE_RANK=3 MAIN_HOST=node01 DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
```

- `NNODES`: the number of machines, 4 here.
- `NODE_RANK`: the index of the machine the line runs on, 0 on the first, then 1, 2 and 3.
- `MAIN_HOST`: the name of the first machine, the one with `NODE_RANK=0`; the others connect to it.

Each machine uses every GPU it has, so 4 machines with 4 GPUs give 16 training processes. Every machine needs the
same number of GPUs and must see the same `DATA` and `CKPT` folders; training starts once the command runs on all
of them.

**Results.** The paper trained REVE-Base twice this way (seeds 2 and 3), 17 epochs each: 23.5 hours on A40 GPUs
and 29.7 hours on V100 GPUs. A run writes the encoder, `$CKPT/run/encoder.pth`, and the loss of every epoch,
`$CKPT/run/loss.csv`.

The two encoders, scored on seven public OpenEEGBench tasks, against the released REVE-Base (chance-adjusted
balanced accuracy, 0 = chance; mean of the two runs):

| Linear probe | PhysioNet-MI | BCIC-IV-2a | BCIC-2020-3 | FACED | Mumtaz | Zyma | CHB-MIT | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| released REVE-Base | 0.377 | 0.191 | 0.088 | 0.167 | 0.681 | 0.295 | 0.529 | **0.333** |
| pretrained on the corpus | 0.373 | 0.230 | 0.095 | 0.159 | 0.694 | 0.340 | 0.423 | **0.331** |

| Full fine-tuning | PhysioNet-MI | BCIC-IV-2a | BCIC-2020-3 | FACED | Mumtaz | Zyma | CHB-MIT | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| released REVE-Base | 0.436 | 0.352 | 0.041 | 0.183 | 0.675 | 0.323 | 0.264 | **0.325** |
| pretrained on the corpus | 0.379 | 0.296 | 0.041 | 0.160 | 0.667 | 0.226 | 0.144 | **0.273** |

The released model's scores are the OpenEEGBench leaderboard's; CHB-MIT, which the leaderboard does not score, is
the released weights run through the same scoring.

## 2. Settings of the scripts

Every setting is an environment variable written before the command, as `DATA` and `CKPT` are above; `download.sh`
also takes one argument, the target directory.

### `download.sh`

```bash
bash download.sh <directory>
```

| setting | default | meaning |
|---|---|---|
| `<directory>` | required | The one argument: a directory with 4 TB free. The store lands in `<directory>/eegdash-corpus-v0/eegdash-corpus-v0.zarr`. |
| `VERIFY` | `0` | `1` checks every file against the published SHA-256 list after the download (hours on the full store) and prints `checksums OK`. |
| `JOBS` | `8` | With `VERIFY=1`, the number of files hashed at a time. |
| `WAIT` | `60` | Seconds to wait before going on when the Hub's rate limit pauses the download. |

### `pretrain.sh`

```bash
DATA=<store> CKPT=<directory> bash pretrain.sh
```

| setting | default | meaning |
|---|---|---|
| `DATA` | required | The store, `…/eegdash-corpus-v0.zarr`; `download.sh` prints the path. |
| `CKPT` | required | The directory the run writes to. Run again with the same one, the run continues after its last saved epoch. |
| `SEED` | `2` | Seeds the weights and the batch order. The paper's two runs used 2 and 3. |
| `RUN` | `reve-base-v0-seed$SEED` | The run's name: it names the saved states under `$CKPT/states/`. |
| `REVE_EEG` | `../reve_eeg` | The patched checkout of REVE's code, by default beside this repository. |
| `GPUS` | all of the machine's | The number of GPUs to use on each machine. |
| `NNODES` | `1` | The number of machines. |
| `NODE_RANK` | required when `NNODES` > 1 | The index of the machine the line runs on, 0 on the first. |
| `MAIN_HOST` | required when `NNODES` > 1 | The name of the first machine; the others connect to it. |

### `train.sh`

```bash
DATA=<store> RUN=<name> CKPT=<directory> REVE_EEG=<checkout> SEED=<n> bash train.sh
```

`pretrain.sh` calls `train.sh`, the script that launches REVE's trainer, and passes these settings on; they can be
set on the `pretrain.sh` line too. Called by itself, `train.sh` needs `DATA`, `RUN`, `CKPT`, `REVE_EEG` and `SEED`,
and takes `NNODES` (default 1) and `NPROC`, the number of processes over all machines (default 1).

| setting | default | meaning |
|---|---|---|
| `EPOCHS` | `17` | The number of epochs; 17 is the paper's. |
| `ENCODER` | `base` | The model size, one of the encoder configurations of REVE's code: `base` (the paper's), `tiny`, `small` or `large`. |
| `LR` | `2.4e-4` | The peak learning rate. |
| `BS` | `300` | REVE's batch-size setting: a batch holds 16 × `BS` ÷ channels samples on each GPU. |
| `WORKERS` | `8` | Data-loading workers per GPU. |
| `MIXED_PRECISION` | `fp16` | `fp16` runs on every recent NVIDIA GPU; `bf16` on those that support it (A100, H100). |
| `EXCLUDE_DATASETS` | none | Dataset ids left out of training, separated by commas (`EXCLUDE_DATASETS=ds004395,nm000338`). |
| `REVE_ENCODER_EVERY` | `5` | Saves a snapshot of the encoder, `encoder_epoch_<k>.pth`, every so many epochs; `0` turns it off. |
| `RESUME` | `False` | `True` continues from the last saved state in `$CKPT`. `pretrain.sh` sets it when that state exists. |
| `ACCELERATE_ARGS` | none | Extra flags for `accelerate launch`. `pretrain.sh` sets them on several machines. |

A run writes under `$CKPT`: `run/encoder.pth` (the pretrained encoder, at the end), `run/encoder_epoch_<k>.pth` (the
snapshots), `run/loss.csv` (one row per epoch) and `states/$RUN/` (a state per epoch, to continue from).

## 3. What is here

| path | content |
|---|---|
| `corpus/` | `eegdash-corpus-v0.csv`: the 77 datasets (ids, paradigm, archive, subjects, sessions, hours, channels, native rate, licence); `LICENSES.csv`: one row per dataset with its licence, licence URL, authors and source DOI; `excluded-datasets.csv`: the datasets kept out by their rights; `reve-corpus.csv`: every dataset of REVE's own pretraining corpus and release card, with what became of it here |
| `conversion/` | the records of the released store: `eegdash-corpus-v0-CONVERSION_REPORT.md` (sessions and hours per dataset, skipped recordings, dropped channels), `-buckets.csv` (one row per bucket: dataset, shard, samples, sessions, channels), `-manifest.csv` (one row per session: source BIDS file, bucket, channels, duration, native rate, subject, session, run, task), `-shards.tsv` (the 418 shards: dataset, mode, subjects, records, estimated hours), `-SHA256SUMS` (every file of the store), `-hf-verify.txt` (the check of the Hugging Face copy against that list: 0 mismatches) |
| `scripts/eegdash_to_reve.py` | converts one shard (one dataset, or part of its subjects) from EEGDash's cache into REVE's layout; `--format zarr` writes the published form |
| `scripts/make_shards.py` | plans the conversion: scans recording headers and sidecars in the raw cache (no data read) and packs subjects into shards |
| `scripts/order_shards.py` | orders a shard list by measured minutes, largest first, finished shards left out |
| `scripts/merge_parts.py` | merges finished parts into one store, verifies it and writes the three side files (manifest, buckets, conversion report); `relabel` renames a dataset id in a merged store, `licence` writes each dataset's licence into its buckets |
| `scripts/check_layout.py` | loads a store with reve_eeg's own data-loading code and pulls training batches: the test that a store is what `train.py` reads |
| `scripts/export_encoder.py` | converts reve_eeg's `encoder.pth` (or an `encoder_epoch_<k>.pth` snapshot) into braindecode's REVE state dict |
| `scripts/run_openeegbench.py` | scores one set of weights on OpenEEGBench (`run`), tabulates the results (`summarize`, `compare`, strategies named probe and fine-tune), checks and stages the task datasets (`cache`, `stage`) and merges a label scored on two machines (`merge`) |
| `scripts/plot_loss.py`, `scripts/estimate_epochs.py` | read the trainer's log: the loss per epoch and per iteration; how many epochs fit in a time budget |
| `scripts/*.txt`, `scripts/dry_shards.tsv` | id lists for `make_shards.py` (`openneuro_ids_other.txt`, `_hbn.txt`, `_peers.txt`: the 56 OpenNeuro datasets; `nemar_ids.txt`: 9 of the 21 NEMAR datasets, the other 12 are listed in `corpus/eegdash-corpus-v0.csv`); a test fixture |
| `patches/` | the changes to reve_eeg's trainer: `PATCHES.md` (what each edit does and why), `apply_patches.sh` (applies them), `zarr_loader.py` (the Zarr reader, patch 8) |
| `download.sh` | downloads the released store from Hugging Face until every file of the SHA-256 list is there, waiting whenever the Hub's rate limit pauses it; run it with the newest Hugging Face client (Quick start, step 1); with `VERIFY=1` it also checks every file against that list |
| `pretrain.sh` | pretrains REVE-Base on the store, on one machine or several: checks the patched reve_eeg checkout, counts the GPUs, resumes a run that saved a state, then calls `train.sh` |
| `train.sh` | launches reve_eeg's `src/train.py` on the store with the paper's settings; its header lists every setting |
| `tests/` | pytest suite for the scripts, the patches and the loader |
| `CITATION.bib` | the corpus, REVE, and each of the 77 datasets keyed by its dataset id |
| `requirements-pretraining.txt` | the short list `pretrain.sh` needs, at the versions `requirements.txt` pins |
| `requirements.txt` | the exact Python environment of every conversion, training and evaluation run |
| `NOTICE` | the licence of the reve_eeg lines quoted in `patches/` and `tests/test_patches.py` |

## 4. Reading the store

The store is Zarr v3, one group per bucket, named `0` … `631`. The names below are the ones written by
`scripts/eegdash_to_reve.py` (`ZarrGroupWriter`) and `scripts/merge_parts.py` (the merge and `licence`).

| array | shape | dtype | content |
|---|---|---|---|
| `<b>/data` | (N, 2000, C) | float32 | the 10-s training samples, µV, not normalised; one chunk per sample, 128 samples per shard file, Blosc-zstd level 5 |
| `<b>/session` | (N,) | int32 | the session (row of `stats`, entry of the `sessions` attribute) each sample comes from |
| `<b>/stats` | (S, 2, C) | float32 | per-session mean and standard deviation per channel, µV |
| `<b>/positions` | (C, 3) | float32 | electrode positions, metres |

| attributes of | keys |
|---|---|
| the root | `format` (`"reve-samples"`), `version`, `sfreq` (200), `window` (2000), `units` (`"uV"`), `band_pass_hz` ([0.5, 99.5]), `datasets`, `n_buckets`, `n_samples`, `buckets` (the bucket index), `licence` (`"other"`), `licences` (dataset → licence), `licence_file` (`"LICENSES.csv"`) |
| a bucket | `dataset`, `channels` (names, sorted), `sessions` (per session: source `file`, `duration` in samples at 200 Hz, `n_samples`, `first_sample`), `bucket`, `source_shard`, `licence`, `licence_url`, `source` |

```python
import numpy as np
import zarr  # zarr >= 3

root = zarr.open("eegdash-corpus-v0.zarr", mode="r")
b = root["0"]
print(b.attrs["dataset"], b.attrs["licence"], b.attrs["channels"][:4])
x = b["data"][0]                              # (2000, C) float32, µV: one 10-s sample
mean, sd = b["stats"][int(b["session"][0])]   # its session's per-channel mean and SD
z = np.clip((x - mean) / sd, -15, 15)         # REVE's normalisation, applied at load time
pos = b["positions"][:]                       # (C, 3), metres
```

## 5. Patches to reve_eeg

`patches/apply_patches.sh <checkout>` edits a reve_eeg checkout at commit 06a7059 in place. It is idempotent (a second
run changes nothing) and stops if an anchor line is missing. It applies patches 1 and 3–10; patch 2 is documented only.
Patch 8 appends `patches/zarr_loader.py` to `src/utils/data_loading.py` (a newer copy replaces the block, never
duplicates it).

| patch | file | change |
|---|---|---|
| 1 | `src/utils/data_loading.py` | single-session guard in `compute_group_segments`: a bucket holding one session no longer crashes |
| 2 | `src/train.py` | not applied: continue pretraining from `brain-bzh/reve-base` |
| 3 | `src/train.py` | per-epoch loss written to `loss.csv` in the run directory |
| 4 | `src/train.py` | multi-GPU: the type check that rejected Accelerate's `DistributedDataParallel` wrapper is dropped |
| 5 | `src/utils/ddp_setup.py` | Weights & Biases online unless `wandb.offline`, with the configured entity |
| 6 | `src/train.py` | Weights & Biases logging every 100 iterations, with a fractional epoch and the learning rate |
| 7 | `src/train.py` | encoder snapshot `encoder_epoch_<k>.pth` every `REVE_ENCODER_EVERY` epochs (default 5) |
| 8 | `src/utils/data_loading.py` | a `data.path` ending in `.zarr` is read from the store (`ZarrEEGDataset`); `REVE_EXCLUDE_DATASETS` leaves datasets out by id |
| 9 | `src/train.py` | a resumed run continues at the epoch after its last saved state |
| 10 | `src/train.py`, `src/utils/data_loading.py` | seeding: the config's `seed` sets the weights and the batch order, and each epoch covers every batch once across ranks |

## 6. Fetch the raw corpus by name

`EEGDashCorpus`, in EEGDash's current `develop` version
([EEGDash #521](https://github.com/eegdash/EEGDash/pull/521)), fetches the raw recordings the store was converted
from, out of the archives. It reads its list of datasets from the corpus table, `corpus/eegdash-corpus-v0.csv`.
In Python the corpus's name is enough, because EEGDash ships a copy of the table:

```python
from eegdash import EEGDashCorpus

corpus = EEGDashCorpus("eegdash-corpus-v0", cache_dir="eegdash-cache")
corpus.download()
corpus.verify(level="hash")
for recording in corpus.records():
    raw = recording.raw
```

The command line takes the table's path, not the name, so have the file first: clone this repository, or save
the table from the [corpus page](https://eegdash.org/corpus/v0.html#download).

```bash
python -m eegdash.corpus download corpus/eegdash-corpus-v0.csv --cache-dir eegdash-cache
python -m eegdash.corpus verify corpus/eegdash-corpus-v0.csv --cache-dir eegdash-cache --level hash
```

This fetches the archives' raw recordings, not the converted Zarr store. To train on the released store, follow
the [Quick start](#1-quick-start-pretrain-reve-on-the-corpus); to read it, [Reading the store](#4-reading-the-store).
The corpus page gives the [full dataset and paradigm breakdown](https://eegdash.org/corpus/v0.html#paradigms).

## 7. Reproduce the conversion

The steps below document how the released store was built. They are not needed to read it.

The usage lines of the scripts, and the commands below, use six shell variables; set them to your own paths:
`$EEGDASH_CACHE` (the raw cache), `$CONVROOT` (shard plans, parts and the store), `$CKPTROOT` (checkpoints),
`$OUTROOT` (results), `$LOGS` (logs) and `$REVE_EEG` (the patched reve_eeg checkout).

**1. Fetch the raw datasets into EEGDash's cache.** EEGDash stores a recording at
`$EEGDASH_CACHE/<dataset>/<BIDS path>`, the archive's own path, so an S3 mirror of OpenNeuro produces the same tree.
The OpenNeuro datasets (`ds*`) were mirrored from OpenNeuro's public bucket; the NEMAR datasets (`nm*`) are git-annex
objects that only EEGDash's downloader resolves.

```bash
for id in $(cat scripts/openneuro_ids_*.txt); do
  s5cmd --no-sign-request sync "s3://openneuro.org/$id/*" "$EEGDASH_CACHE/$id/"
done
```

```python
from eegdash import EEGDashDataset
EEGDashDataset(cache_dir=EEGDASH_CACHE, dataset="nm000338").download_all(n_jobs=8)   # once per NEMAR id
```

The converter also downloads a NEMAR dataset itself when run with `--online`. EEGDash serves most OpenNeuro ids under
their NEMAR re-host `on<digits>` and caches them under that name, while the converter's offline mode reads
`$EEGDASH_CACHE/ds<digits>`: fill the OpenNeuro part of the cache from S3 as above, or link the two names.

**2. Plan the shards.** Subjects are packed into shards of at most 200 h or 400 recordings.

```bash
python scripts/make_shards.py --cache-dir $EEGDASH_CACHE --out $CONVROOT/plan --workers 8 \
    --offline-ids scripts/openneuro_ids_other.txt scripts/openneuro_ids_hbn.txt scripts/openneuro_ids_peers.txt \
    --online-ids scripts/nemar_ids.txt
```

This writes `records_<id>.csv` per dataset, `plan_summary.json` and `shards.tsv`. The released store's own list is
`conversion/eegdash-corpus-v0-shards.tsv`: 418 shards in the order that fixed the bucket numbers, each naming its
dataset, its mode (`offline`: read from the cache; `online`: fetched by EEGDash) and its subjects. The commands
below use it.

**3. Convert each shard.** One invocation per shard, independent of the others, so shards can run in parallel on as
many machines as are available. The converter refuses the source datasets of the held-out OpenEEGBench tasks by id.

```bash
tail -n +2 $CONVROOT/plan/eegdash-corpus-v0.tsv | while IFS=$'\t' read -r shard dataset mode subjects _; do
  args=(--dataset "$dataset" --cache-dir "$EEGDASH_CACHE" --out "$CONVROOT/parts_zarr/$shard" --format zarr --n-jobs 8)
  [ "$mode" = online ] && args+=(--online)
  [ "$subjects" != all ] && args+=(--subjects $subjects)
  python scripts/eegdash_to_reve.py "${args[@]}" > "$LOGS/shard-$shard.out" 2>&1
done
```

A finished part carries `_DONE` and a second run skips it. `order_shards.py` sorts a list by the minutes each shard
took (read from those logs), for scheduling.

**4. Merge and check.** The merge hard-links the parts' chunks into `<out>/<name>.zarr` and writes
`<name>-manifest.csv`, `<name>-buckets.csv` and `<name>-CONVERSION_REPORT.md` beside it (the released ones are in
`conversion/`). `licence` then writes each dataset's licence into its buckets and the root. `check_layout.py` needs
the patched reve_eeg checkout of step 5.

```bash
python scripts/merge_parts.py --parts $CONVROOT/parts_zarr --out $CONVROOT/zarr \
    --name eegdash-corpus-v0 --shards $CONVROOT/plan/eegdash-corpus-v0.tsv
python scripts/merge_parts.py licence --store $CONVROOT/zarr/eegdash-corpus-v0.zarr --table corpus/eegdash-corpus-v0.csv
REVE_EEG=$REVE_EEG python scripts/check_layout.py $CONVROOT/zarr/eegdash-corpus-v0.zarr
```

A downloaded copy of the store is checked against the released list with
`sha256sum -c conversion/eegdash-corpus-v0-SHA256SUMS`, run from the directory that holds `eegdash-corpus-v0.zarr/`.

**5. Train and score.** Clone reve_eeg at the commit the patches target, patch it, and launch `train.sh` (its header
lists every setting; the paper's two runs used `SEED=2` and `SEED=3`, 17 epochs, learning rate 2.4e-4, batch 300).
`pretrain.sh` is the short form of the same launch: it fills in the run name, the GPU count and the resume flag and
calls `train.sh` (the [Quick start](#1-quick-start-pretrain-reve-on-the-corpus)).

```bash
git clone https://github.com/elouayas/reve_eeg.git $REVE_EEG && git -C $REVE_EEG checkout 06a7059
bash patches/apply_patches.sh $REVE_EEG
DATA=$CONVROOT/zarr/eegdash-corpus-v0.zarr RUN=reve-base-v0-run2 CKPT=$CKPTROOT/reve-base-v0-run2 \
    SEED=2 NPROC=4 REVE_EEG=$REVE_EEG bash train.sh
```

Then export the encoder to braindecode's format and score it on the seven OpenEEGBench tasks the paper reports (the
eighth public task, ISRUC sleep staging, is left out: the corpus holds no sleep recordings):

```bash
python scripts/export_encoder.py --encoder $CKPTROOT/reve-base-v0-run2/run/encoder.pth \
    --out $CKPTROOT/reve-base-v0-run2/export/reve_braindecode_epfinal.pth
python scripts/run_openeegbench.py run --label reve-base-v0-run2-epfinal \
    --checkpoint $CKPTROOT/reve-base-v0-run2/export/reve_braindecode_epfinal.pth \
    --datasets bcic2a physionet bcic2020_3 faced mdd_mumtaz2016 arithmetic_zyma2019 chbmit \
    --out $OUTROOT/evals/openeegbench
python scripts/run_openeegbench.py summarize $OUTROOT/evals/openeegbench
```

## 8. Environment

Two files. `requirements-pretraining.txt` is the short list that pretraining on the store needs
(15 packages, the same versions); the download runs apart from it, with the newest `huggingface_hub` and `hf_xet`
(Quick start, step 1). `requirements.txt` is the exact environment of every run (its header gives the commands): Python 3.11, torch 2.5.1
and torchaudio 2.5.1 built for CUDA 12.1 (from the PyTorch index), zarr 3.1.6, mne 1.13.0, accelerate 1.15.0,
hydra-core 1.3.7, open-eeg-bench 0.6.0. braindecode and eegdash are editable installs of the commits named in the
header (braindecode `59a1e8b8`, eegdash `0a70e9e`). reve_eeg is run from its checkout and never pip-installed: its
`pyproject.toml` would replace torch.

```bash
pytest -q tests
```

The tests need numpy, scipy, pandas, zarr, numcodecs, mne, mne-bids, pybv and edfio (to write the fixtures),
matplotlib, torch, braindecode and eegdash; huggingface_hub is optional. Tests that load reve_eeg's code (the patches,
the Zarr loader, the converter's end-to-end check, the encoder export, the Hydra overrides) need a checkout of
reve_eeg at 06a7059, patched or not, with its training dependencies (accelerate, hydra-core, omegaconf, einops,
transformers); they look for it at `$REVE_EEG`, then `~/GitHub/reve_eeg` and `~/Documents/GitHub/reve_eeg`, and skip
when none is found.

## 9. Licences

The code is under the BSD 3-Clause License (`LICENSE`). The lines of reve_eeg quoted in `patches/` and
`tests/test_patches.py` remain under reve_eeg's MIT licence (`NOTICE`). The EEG data is not in this repository: each
dataset keeps its own licence, listed in `corpus/LICENSES.csv` (52 CC0-1.0, 12 CC-BY-SA-4.0, 11 CC-BY-4.0, 1
CC-BY-NC-4.0, 1 GPL-3.0), and the Hugging Face gate asks users to accept them before downloading.

## 10. Citation

Cite the corpus and every dataset whose buckets you use (`CITATION.bib`, keyed by dataset id).

```bibtex
@inproceedings{eegdash_corpus_v0,
  title     = {{EEGDash Corpus}: A Training-Ready Public Corpus for {EEG} Foundation Models},
  author    = {Shahi, Kuber and Aristimunha, Bruno and Delorme, Arnaud},
  booktitle = {7th International Workshop on Machine Learning for EEG Signal Processing (MLESP 2026), with IEEE BIBM 2026},
  year      = {2026},
  note      = {arXiv:2611.06356. Dataset: \url{https://huggingface.co/datasets/EEGDash/eegdash-corpus-v0}}
}
```
