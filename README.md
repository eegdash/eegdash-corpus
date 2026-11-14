# EEGDash Corpus

**eegdash-corpus-v0** is 77 public EEG datasets served through [EEGDash](https://github.com/eegdash/EEGDash) from
OpenNeuro and NEMAR, converted once with the preprocessing recipe of the REVE EEG foundation model into one Zarr store
of 632 buckets, 14,074.7 hours and 6.21 billion tokens per epoch (3.6 TB). This repository holds the code and the
records that built it; the data is on Hugging Face, gated, at
[EEGDash/eegdash-corpus-v0](https://huggingface.co/datasets/EEGDash/eegdash-corpus-v0) (tag `v1.0`), and the corpus
is described in *EEGDash Corpus: A Training-Ready Public Corpus for EEG Foundation Models* (MLESP 2026, with IEEE
BIBM 2026; preprint [arXiv:2611.06356](https://arxiv.org/abs/2611.06356); see [Citation](#citation)).

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

## Quick start: pretrain REVE on the corpus

Three steps, the same on a workstation or a cluster; the tutorial page
[Pretraining REVE](https://eegdash.org/corpus/pretraining_reve.html) walks through them.

```bash
# 1. Install, all in the folder you start in: this repository, a Python 3.11 environment and REVE's code with the patches
git clone https://github.com/eegdash/eegdash-corpus.git && cd eegdash-corpus
python3.11 -m venv ../reve-env && source ../reve-env/bin/activate
pip install -r requirements-pretraining.txt
git clone https://github.com/elouayas/reve_eeg.git ../reve_eeg && git -C ../reve_eeg checkout 06a7059
bash patches/apply_patches.sh ../reve_eeg
hf auth login                         # an account that accepted the dataset's terms on Hugging Face

# 2. Download the corpus (3.6 TB)
bash download.sh /data

# 3. Pretrain REVE-Base with the paper's settings (17 epochs)
DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
```

`pretrain.sh` uses every GPU of the machine. The paper's runs used 16 GPUs on 4 machines: there the same line runs
once per machine with `NNODES=4`, `NODE_RANK` (0 to 3) and `MAIN_HOST` (the first machine's name); an epoch then
takes about 1.3 to 1.8 hours depending on the GPU. The result is `$CKPT/run/encoder.pth`; `scripts/export_encoder.py`
converts it for braindecode ([Reproduce the conversion](#reproduce-the-conversion), step 5).

## What is here

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
| `download.sh` | downloads the released store from Hugging Face and, with `VERIFY=1`, checks every file against the SHA-256 list |
| `pretrain.sh` | pretrains REVE-Base on the store, on one machine or several: checks the patched reve_eeg checkout, counts the GPUs, resumes a run that saved a state, then calls `train.sh` |
| `train.sh` | launches reve_eeg's `src/train.py` on the store with the paper's settings; its header lists every setting |
| `tests/` | pytest suite for the scripts, the patches and the loader |
| `CITATION.bib` | the corpus, REVE, and each of the 77 datasets keyed by its dataset id |
| `requirements-pretraining.txt` | the short list `download.sh` and `pretrain.sh` need, at the versions `requirements.txt` pins |
| `requirements.txt` | the exact Python environment of every conversion, training and evaluation run |
| `NOTICE` | the licence of the reve_eeg lines quoted in `patches/` and `tests/test_patches.py` |

## Fetch the raw corpus by name

With EEGDash's current `develop` version (the named-corpus API was merged in
[EEGDash #521](https://github.com/eegdash/EEGDash/pull/521)), the corpus table is included in the package:

```python
from eegdash import EEGDashCorpus

corpus = EEGDashCorpus("eegdash-corpus-v0", cache_dir="eegdash-cache")
corpus.download()
corpus.verify(level="hash")
for recording in corpus.records():
    raw = recording.raw
```

This fetches the archives' raw recordings, not the converted Zarr store. To train on the released
store, download it from [Hugging Face](https://huggingface.co/datasets/EEGDash/eegdash-corpus-v0)
(`download.sh`) and follow the [Quick start](#quick-start-pretrain-reve-on-the-corpus) or
[Reading the store](#reading-the-store). The corpus web page gives the
[full dataset and paradigm breakdown](https://eegdash.org/corpus/v0.html#paradigms).

## Reproduce the conversion

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
calls `train.sh` (the [Quick start](#quick-start-pretrain-reve-on-the-corpus)).

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

## Reading the store

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

## Patches to reve_eeg

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

## Environment

Two files. `requirements-pretraining.txt` is the short list that downloading the store and pretraining on it need
(15 packages, the same versions). `requirements.txt` is the exact environment of every run (its header gives the commands): Python 3.11, torch 2.5.1
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

## Licences

The code is under the BSD 3-Clause License (`LICENSE`). The lines of reve_eeg quoted in `patches/` and
`tests/test_patches.py` remain under reve_eeg's MIT licence (`NOTICE`). The EEG data is not in this repository: each
dataset keeps its own licence, listed in `corpus/LICENSES.csv` (52 CC0-1.0, 12 CC-BY-SA-4.0, 11 CC-BY-4.0, 1
CC-BY-NC-4.0, 1 GPL-3.0), and the Hugging Face gate asks users to accept them before downloading.

## Citation

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
