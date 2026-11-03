# Hand edits to reve_eeg (commit 06a7059)

`bash patches/apply_patches.sh <reve_eeg checkout>` applies 1 and 3 to 10; it is idempotent and fails loudly if the anchor lines are not found. Patch 2 is documented only and **not applied** for the from-scratch run.

## 4. Multi-GPU — `src/train.py`

`mae = ensure_type(accelerator.prepare(mae), MAE)` raises `TypeError` on more than one GPU, because Accelerate returns a `DistributedDataParallel` wrapper (Expanse job 54353570, 4 V100). The check is dropped; the rest of the loop already goes through `accelerator.unwrap_model(mae)`. Without this, `train.py` at 06a7059 only runs on a single GPU.

## 5. Weights & Biases online — `src/utils/ddp_setup.py`

Two lines. (a) `if "lustre" in args.data.path or args.wandb.offline` forced offline mode for any Lustre path — their cluster's compute nodes had no network; Expanse's do — now only `wandb.offline` decides. (b) the tracker was initialised with `entity="online"` (a literal) when not offline; now always `args.wandb.entity`. Result: the loss curve streams live to wandb.ai during training; `wandb sync` is only needed if a run is launched with `WANDB_OFFLINE=1`.

## 1. Single-session guard — `src/utils/data_loading.py::compute_group_segments`

`if index_data[-1]["flag_reduce"] != index_data[-2]["flag_reduce"]` crashes when a big recording holds one session. Guarded with `len(index_data) > 1`. Lets the converter keep channel sets that occur once in a dataset (e.g. ds003004 has 27 unique montages) instead of discarding them.

## 3. Per-epoch loss CSV — `src/train.py`

After `save_state(...)` at the end of each epoch, the main process appends `epoch, loss_ema, iterations, seconds` to `loss.csv` in the run directory (Hydra `chdir`s there, next to `encoder.pth`). `wandb.log` is off in our runs; this gives a curve that can be plotted at any point (`scripts/plot_loss.py`).

## 6. W&B every 100 iterations — `src/train.py`

REVE logs `loss`, `loss_ema`, `epoch`, `it` to Weights & Biases on **every** iteration (rank 0's batch) — ~106k noisy points over a 20-epoch run. Now every 100th iteration (~53 points per epoch on the full corpus), with `epoch` as a fraction (`epoch + it / iterations_per_epoch`, so the x-axis reads in epochs) and the current learning rate. Vocabulary: an iteration is one optimizer step = one batch per GPU (`accumulate_grad_batches: 1`); the log's `it. N/M` counts per GPU, `Train batches` is the total over GPUs. Per-epoch values stay in `loss.csv` (patch 3).

## 7. Encoder snapshots every N epochs — `src/train.py`

After the `loss.csv` row (main process only), every `REVE_ENCODER_EVERY` epochs (default 5; `train_reve.sbatch` exports it) the encoder's state dict is saved as `encoder_epoch_<k>.pth` in the run directory (`checkpoints/<RUN>/run/`), skipping the final epoch which REVE already writes as `encoder.pth`. ~280 MB each for Base. Purpose: `export_encoder.py` an intermediate snapshot and run the sprint-1 linear probe on ds003810 while training continues — pretraining has no validation loss (`train_no_val`), so a downstream probe is the only "is it learning something useful" signal. Not written into the epoch state dirs because REVE's rotation (`max_states: 5`) deletes those.

## 8. Train from the published Zarr corpus — `src/utils/data_loading.py`

A `data.path` ending in `.zarr` (e.g. `training_data_converted/zarr/eegdash-corpus-v0.zarr`, or `eegdash-corpus-v1.zarr` with `REVE_EXCLUDE_DATASETS=nm000289` for the same 632 buckets) is read from the Zarr store written by `eegdash_to_reve.py --format zarr` and merged by `merge_parts.py`; any other path takes REVE's own memmap code, unchanged (that layout is retired; the first runs used it). Two edits: (a) one line at the top of `get_train_val_loaders` dispatches on the path; (b) the contents of `patches/zarr_loader.py` are appended to the module (re-running with a newer `zarr_loader.py` replaces the block, never duplicates it). `ZarrEEGDataset` subclasses `EEGDataset` and overrides only the file handling: stats and positions are held in memory, and a bucket's memmap is replaced by an object whose `[i : i + 2000]` returns sample row `i`. Keys keep REVE's `bucket_-_session_-_offset` form, with the offset being the sample's row, so normalisation, clipping, masking and `GroupedSampler` run unchanged. Tests (`tests/test_zarr_loader.py`): the same sessions written both ways give the same groups and bit-identical items, `get_train_val_loaders` yields the same number of samples and batches. `data.subset=all` only; there is no validation split. **Reading (2026-09-25).** On the full v0 store (then `reve-public-nc`) on Lustre, Zarr's own per-sample read (re-open the shard file, re-read its index, read the chunk: three round trips) starved the GPUs: 3.61 s per step with 8 workers per GPU, 1.29 s with 36 (GPU utilisation ≈ 50 %), because 72 % of the samples come in 240-sample I-CARE batches. The block therefore reads shards directly: `_ShardReader` keeps each worker's shard files open and their indexes in memory, so a sample is one `pread` (missing shards and empty chunks read as zeros, the fill value, as in Zarr; an unfamiliar array layout falls back to Zarr), and `ZarrEEGDataset.__getitems__` reads a whole batch's samples in parallel threads before REVE's own `__getitem__` normalises and masks each one, so batches are identical. Measured on the real store (login node): one I-CARE batch 50 s through Zarr, 25–37 s direct, 5–9 s direct with 16 threads, samples identical. Knobs: `REVE_ZARR_DIRECT=0` returns to Zarr's reads; `REVE_ZARR_READ_THREADS` (default 8) threads per DataLoader worker. **Leaving datasets out (2026-10-28).** `REVE_EXCLUDE_DATASETS=nm000289,ds004395` drops those datasets' buckets (by each bucket's `dataset` attribute) from the sample list before the sampler sees it, for the camera-ready's composition ablations on `eegdash-corpus-v1`; an id that no bucket carries raises, and rank 0 prints how many samples were left out. `train_reve.sbatch` sets it from `EXCLUDE_DATASETS` and refuses it for a memmap layout.

Open files (2026-09-27, job 54458868): the reader keeps one descriptor budget per process shared by every bucket, `REVE_ZARR_MAX_OPEN` (default 1,024), and at most `REVE_ZARR_MAX_INDEX` shard indexes (default 16,384). The first version capped descriptors per bucket; 968 buckets (`reve-public-nc`) × 144 loader processes per node filled a node's kernel file table after 20 minutes of `base-run3` (runbook EXP 14c). Forked workers (2026-09-30, job 54491935): a new worker's threads each reset the process-wide table on their first read, and one closed a descriptor another had just opened (errno 9 at an epoch start); the reset is now locked and happens once per process (runbook EXP 14e). Truncated reads (2026-10-02, job 54576735): one chunk that eight epochs had read fine failed blosc decompression (-1) in epoch 9 and ended `base-run3`; `os.pread` may return fewer bytes than asked, so the reader now reads until a chunk (or index) is whole, and retries a failed chunk once from a fresh descriptor and index before raising with the file and row.

## 9. Resume at the next epoch — `src/train.py`

`train.py` reloads the saved state (`checkpointing.load_last_state`) and then loops `range(trainer.epochs)` from 0
again, so a resumed run trains `epochs` more epochs and reuses the epoch numbers of the first part: its snapshots
`encoder_epoch_4/8.pth`, its `states/epoch_<k>` folders and its `loss.csv` rows would be overwritten. Seen 2026-10-02
in a resume test on DeltaAI (job 3292081, log `Epoch 0 … 0/17` after `All model weights loaded successfully`), before
any real resume had run. The loop now starts one past the highest `states/epoch_<k>`: 7 for `base-run3` (epoch_6
saved), 11 for `base-run4` (epoch_10). `REVE_START_EPOCH` overrides it, for a copied state that holds only `last`. The
learning rate is a cycle per epoch (`CyclicTrapezoidLR`: warm-up, plateau, cool-down within each epoch), so the epoch
number changes only how many epochs run and how they are named, not the schedule.

## 2. Optional, NOT applied: continue pretraining from `brain-bzh/reve-base`

In `src/train.py`, after `mae = MAE(args)` and before `accelerator.prepare(...)`:
```python
    pretrained_id = getattr(args.init, "pretrained_id", None)
    if pretrained_id:
        from models.encoder import REVE as LocalREVE
        enc, _ = LocalREVE.from_pretrained(pretrained_id)
        missing, unexpected = mae.encoder.load_state_dict(enc.state_dict(), strict=False)
        accelerator.print(f"init from {pretrained_id}: missing={len(missing)} unexpected={len(unexpected)}")
```
plus `pretrained_id: null` in `src/configs/init/mae.yaml`; launch with `init.pretrained_id=brain-bzh/reve-base optimizer.lr=1e-4`. Needs HF access on the node that builds the model (or a populated `HF_HOME`).

## Not patched, worked around at launch

- `trainer.n_gpus: ${gpu_count:}` is 0 on a machine without CUDA and is used as a divisor → pass `trainer.n_gpus=1` for CPU/MPS smoke tests.
- `hydra.job.chdir: true` → pass `hydra.run.dir=<run dir>` so `encoder.pth`, `loss.csv` and `.hydra/` land in a known place.
- `flash_attn` is optional (`use_flash=FLASH_AVAILABLE`); without it attention uses PyTorch SDPA. Fine on V100 (no flash kernels there anyway).

## 10. Seeding: reproducible weights, one pass per epoch across ranks (2026-10-28)

REVE's `train.py` never applied its config's `seed: 42`: the weights, the masks and the batch order came from unseeded generators, so no run could be repeated, and `base-run4` and `small-run1` are each one unrepeatable draw. The batch order had a second problem. `GroupedSampler` shuffles with Python's global `random`, which Accelerate does not synchronise across processes (its `rng_types` covers torch generators only), so on 16 GPUs each rank shuffled its own permutation and `BatchSamplerShard` kept every 16th batch of it: an "epoch" covered about 64 % of the batches (1 − (15/16)^16) with repeats, and the corpus was sampled evenly only in expectation over the 17 epochs. Confirmed on 2026-10-28 by running the sampler in two processes (different orders) and reading `accelerate.data_loader` (`rng_types` defaults to `["generator"]`). Every run before this patch trained that way, REVE's own release presumably too.

The patch, six anchors: (a–c) `GroupedSampler(seed=None)` owns a `random.Random(seed + epoch)`, shuffles copies of the group lists (so epoch *k* depends on `seed + k` alone, not on earlier in-place shuffles) and gains `set_epoch`; unseeded it behaves as before. (d) `_build_loader` passes `args.seed` for training (the Zarr loader in `zarr_loader.py` does the same). (e) `train.py` seeds Python, NumPy and torch with `seed + rank` right after the accelerator exists: masking and dropout differ per rank, and DDP broadcasts rank 0's initialisation, so the weights follow `seed` alone. (f) The loop calls `set_epoch(epoch)` before each epoch, through Accelerate's shard wrapper on more than one GPU, so a resumed run (patch 9) shuffles exactly as the uninterrupted one would. `train_reve.sbatch`: `SEED=` (Hydra `seed=`; unset leaves the config's 42, which the first log line prints). Tests (`test_zarr_loader.py`, `test_patches.py`): two seeded samplers agree and change with the epoch, the groups are not mutated, Accelerate's shards on two ranks are disjoint and cover every batch, both loaders pass the seed, rank *r* seeds torch with `seed + r`. **For the paper:** the camera-ready's runs on v1 are the first where an epoch is one pass over the corpus; the submitted `base-run4` was not, and the text says so.
