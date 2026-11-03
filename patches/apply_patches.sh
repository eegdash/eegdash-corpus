#!/bin/bash
# Apply our hand edits to a reve_eeg checkout (commit 06a7059). Idempotent: re-running is a no-op.
#   bash patches/apply_patches.sh ~/GitHub/reve_eeg
# See PATCHES.md for what each edit does and why.
set -euo pipefail
REVE=${1:?path to reve_eeg checkout}
PY=${PYTHON:-python}
"$PY" - "$REVE" "$(dirname "$0")" <<'EOF'
import sys
from pathlib import Path

reve, here = Path(sys.argv[1]), Path(sys.argv[2])

def patch(path, old, new, tag, required=True):
    p = reve / path
    src = p.read_text()
    if new in src:
        print(f"{tag}: already applied ({path})"); return True
    if old not in src:
        assert not required, f"{tag}: anchor not found in {path} — is this commit 06a7059?"
        return False
    p.write_text(src.replace(old, new, 1))
    print(f"{tag}: applied ({path})"); return True

# 1. compute_group_segments indexes index_data[-2]; a big recording with one session crashed.
patch("src/utils/data_loading.py",
      '            if index_data[-1]["flag_reduce"] != index_data[-2]["flag_reduce"]:',
      '            if len(index_data) > 1 and index_data[-1]["flag_reduce"] != index_data[-2]["flag_reduce"]:',
      "patch 1 (single-session guard)")

# 8. Train from the published Zarr corpus: a data.path ending in .zarr is read by ZarrEEGDataset (zarr_loader.py,
#    appended to the module); a memmap layout takes REVE's own path, unchanged.
P8_ANCHOR = '    """Entry point: build train (and optionally val) data loaders from config."""\n'
patch("src/utils/data_loading.py", P8_ANCHOR,
      P8_ANCHOR +
      '    if str(args.data.path).rstrip("/").endswith(".zarr"):  # PATCH 8: the published Zarr corpus\n'
      '        return _zarr_train_loader(args)\n',
      "patch 8a (Zarr dispatch)")
P8_END = "    return train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler\n"
P8_BLOCK = (here / "zarr_loader.py").read_text()
dl = reve / "src/utils/data_loading.py"
src = dl.read_text()
if src.endswith(P8_END + P8_BLOCK):
    print("patch 8b (Zarr loader): already applied (src/utils/data_loading.py)")
else:  # a block from an older zarr_loader.py is replaced, never appended twice
    assert P8_END in src, "patch 8b: anchor not found in src/utils/data_loading.py — is this commit 06a7059?"
    dl.write_text(src[:src.index(P8_END) + len(P8_END)] + P8_BLOCK)
    print("patch 8b (Zarr loader): applied (src/utils/data_loading.py)")

# 3. Per-epoch loss to loss.csv in the run directory (a plottable curve even without wandb).
patch("src/train.py",
      "        save_state(accelerator, args, epoch)\n",
      "        save_state(accelerator, args, epoch)\n"
      "        if accelerator.is_main_process:  # PATCH 3: per-epoch loss curve without wandb\n"
      "            new = not os.path.exists(\"loss.csv\")\n"
      "            with open(\"loss.csv\", \"a\") as f:\n"
      "                if new:\n"
      "                    f.write(\"epoch,loss_ema,iterations,seconds\\n\")\n"
      "                f.write(f\"{epoch},{loss_ema},{batch_idx + 1},{time.time() - start:.1f}\\n\")\n",
      "patch 3 (loss.csv)")

# 7. Encoder snapshot every N epochs (N = REVE_ENCODER_EVERY, default 5) into the run directory, so intermediate
#    checkpoints can be exported and probed downstream while training runs. REVE's own save_encoder() writes into
#    the epoch state dir, which its rotation (max_states=5) deletes - hence the run dir.
P3_LAST = '                f.write(f"{epoch},{loss_ema},{batch_idx + 1},{time.time() - start:.1f}\\n")\n'
patch("src/train.py", P3_LAST,
      P3_LAST +
      "            every = int(os.environ.get(\"REVE_ENCODER_EVERY\", \"5\"))  # PATCH 7: encoder snapshot every N epochs\n"
      "            if every > 0 and (epoch + 1) % every == 0 and epoch + 1 < args.trainer.epochs:\n"
      "                accelerator.save(accelerator.unwrap_model(mae).encoder.state_dict(), f\"encoder_epoch_{epoch + 1}.pth\")\n"
      "                accelerator.print(f\"Saved encoder snapshot encoder_epoch_{epoch + 1}.pth\")\n",
      "patch 7 (encoder snapshot every N epochs)")

# 4. Multi-GPU: Accelerate wraps the model in DistributedDataParallel and REVE's type check raised
#    TypeError (seen on Expanse job 54353570). Downstream code already uses accelerator.unwrap_model.
patch("src/train.py",
      "    mae = ensure_type(accelerator.prepare(mae), MAE)\n",
      "    mae = accelerator.prepare(mae)  # PATCH 4: DDP-wrapped on >1 GPU; the MAE type check rejected it\n",
      "patch 4 (multi-GPU prepare)")

# 6. wandb logging every 100 iterations (not every one) with a fractional epoch and the learning rate:
#    ~1k readable points over a 20-epoch run instead of ~100k noisy ones; loss.csv (patch 3) stays per epoch.
P6_NEW = ('            if args.wandb.log and batch_idx % 100 == 0:  # PATCH 6: every 100 iterations, epoch as a fraction\n'
          '                accelerator.log({"epoch": epoch + batch_idx / max(1, n_iter_per_train), "it": batch_idx,\n'
          '                                 "loss_ema": loss_ema, "loss": loss_g, "lr": optimizer.param_groups[0]["lr"]})\n')
P6_V50 = P6_NEW.replace("% 100 == 0:  # PATCH 6: every 100", "% 50 == 0:  # PATCH 6: every 50")
patch("src/train.py", P6_V50, P6_NEW, "patch 6 (upgrade 50 -> 100 iterations)", required=False) or patch(
      "src/train.py",
      '            if args.wandb.log:\n'
      '                accelerator.log({"epoch": epoch, "it": batch_idx, "loss_ema": loss_ema, "loss": loss_g})\n',
      P6_NEW, "patch 6 (wandb every 100 iterations)")

# 5. wandb: REVE forced offline mode for any data path containing "lustre" (their cluster had no network
#    on compute nodes; Expanse does) and sent the literal entity "online" when not offline.
patch("src/utils/ddp_setup.py",
      '        if "lustre" in args.data.path or args.wandb.offline:\n',
      '        if args.wandb.offline:  # PATCH 5: offline only when asked, not for every lustre path\n',
      "patch 5a (wandb offline rule)")
patch("src/utils/ddp_setup.py",
      '                "entity": args.wandb.entity if args.wandb.offline else "online",\n',
      '                "entity": args.wandb.entity,  # PATCH 5: was the literal "online" when not offline\n',
      "patch 5b (wandb entity)")

# 9. Resume at the next epoch. train.py reloaded the state and then looped range(trainer.epochs) from 0 again, so a
#    resumed run trained `epochs` more epochs and reused epoch numbers: snapshots encoder_epoch_4/8 and the
#    states/epoch_<k> folders of the first part were overwritten (seen 2026-10-02 in DeltaAI resume test 3292081,
#    before any real resume ran). The next epoch is one past the highest states/epoch_<k>; REVE_START_EPOCH overrides
#    it (a copied state with only `last`). The learning rate repeats every epoch (CyclicTrapezoidLR), so the epoch
#    number changes only how many epochs run and how they are named.
P9_OLD = ('    if args.checkpointing.load_last_state:\n'
          '        accelerator.load_state(pjoin(args.checkpointing.state_path, "last"))\n'
          '\n'
          '    pbar = tqdm(range(args.trainer.epochs))\n')
P9_NEW = ('    first_epoch = 0\n'
          '    if args.checkpointing.load_last_state:\n'
          '        accelerator.load_state(pjoin(args.checkpointing.state_path, "last"))\n'
          '        # PATCH 9: continue after the last saved epoch instead of restarting at 0\n'
          '        done = [int(f.split("_")[-1]) for f in os.listdir(args.checkpointing.state_path) if f.startswith("epoch_")]\n'
          '        first_epoch = int(os.environ.get("REVE_START_EPOCH", max(done) + 1 if done else 0))\n'
          '        accelerator.print(f"Resuming at epoch {first_epoch} of {args.trainer.epochs}")\n'
          '\n'
          '    pbar = tqdm(range(first_epoch, args.trainer.epochs))\n')
patch("src/train.py", P9_OLD, P9_NEW, "patch 9 (resume at the next epoch)")

# 10. Seeding (2026-10-28). train.py never used args.seed: weights, masks and the batch order came from unseeded
#     generators, so no run could be repeated. Worse, GroupedSampler shuffles with Python's global `random`, which
#     Accelerate does not synchronise across processes (it syncs torch generators only): on 16 GPUs each rank shuffled
#     its own permutation and kept every 16th batch of it, so an "epoch" covered about 64 % of the batches
#     (1 - (15/16)^16) with repeats. Every run up to base-run4 and small-run1 trained that way. Now the sampler owns a
#     random.Random(seed + epoch), identical on every rank and new each epoch (set_epoch, also right after a resume),
#     so Accelerate's shard split covers each batch exactly once per epoch; torch is seeded with seed + rank (masking,
#     dropout), and DDP broadcasts rank 0's initialisation, so the weights follow `seed` alone. SEED= in train_reve.sbatch.
P10A_OLD = ('    def __init__(self, dataset, batch_size, drop_last, n_gpu, mode="train"):\n'
            '        self.dataset = dataset\n')
P10A_NEW = ('    def __init__(self, dataset, batch_size, drop_last, n_gpu, mode="train", seed=None):\n'
            '        self.seed, self.epoch = seed, 0  # PATCH 10: seeded, the shuffle is the same on every rank (set_epoch)\n'
            '        self.dataset = dataset\n')
patch("src/utils/data_loading.py", P10A_OLD, P10A_NEW, "patch 10a (sampler seed)")
P10B_OLD = ('            for c in n_chans:\n'
            '                self.local_batch_size = get_local_batch_size(c, self.global_batch_size)\n'
            '                group_indices = self.groups[c]\n'
            '                random.shuffle(group_indices)\n'
            '                indices += [\n'
            '                    group_indices[i * self.local_batch_size : (i + 1) * self.local_batch_size]\n'
            '                    for i in range(len(group_indices) // self.local_batch_size + self.drop_last)\n'
            '                ]\n'
            '            random.shuffle(indices)\n')
P10B_NEW = ('            rng = random if self.seed is None else random.Random(self.seed + self.epoch)  # PATCH 10: per epoch, per seed\n'
            '            for c in n_chans:\n'
            '                self.local_batch_size = get_local_batch_size(c, self.global_batch_size)\n'
            '                group_indices = self.groups[c] if self.seed is None else list(self.groups[c])  # a copy: epoch k depends on seed + k only\n'
            '                rng.shuffle(group_indices)\n'
            '                indices += [\n'
            '                    group_indices[i * self.local_batch_size : (i + 1) * self.local_batch_size]\n'
            '                    for i in range(len(group_indices) // self.local_batch_size + self.drop_last)\n'
            '                ]\n'
            '            rng.shuffle(indices)\n')
patch("src/utils/data_loading.py", P10B_OLD, P10B_NEW, "patch 10b (sampler shuffle)")
P10C_OLD = ('        return iter(self.indices)\n'
            '\n'
            '    def __len__(self):\n'
            '        return len(self.indices)\n')
P10C_NEW = ('        return iter(self.indices)\n'
            '\n'
            '    def set_epoch(self, epoch):  # PATCH 10: train.py calls it before each epoch, so a resumed run shuffles as a fresh one would\n'
            '        self.epoch = epoch\n'
            '\n'
            '    def __len__(self):\n'
            '        return len(self.indices)\n')
patch("src/utils/data_loading.py", P10C_OLD, P10C_NEW, "patch 10c (sampler set_epoch)")
P10D_OLD = ('        n_gpu=args.trainer.n_gpus * args.trainer.n_nodes,\n'
            '        mode="train" if train else "val",\n'
            '    )\n')
P10D_NEW = ('        n_gpu=args.trainer.n_gpus * args.trainer.n_nodes,\n'
            '        mode="train" if train else "val",\n'
            '        seed=args.seed if train else None,  # PATCH 10\n'
            '    )\n')
patch("src/utils/data_loading.py", P10D_OLD, P10D_NEW, "patch 10d (memmap loader seed)")
P10E_OLD = ('    accelerator = get_accelerator(args)\n'
            '    logger.info("Starting training")\n')
P10E_NEW = ('    accelerator = get_accelerator(args)\n'
            '    logger.info("Starting training")\n'
            '    # PATCH 10: torch seeded per rank (masking, dropout); DDP broadcasts rank 0, so the weights follow args.seed alone.\n'
            '    # The batch order is seeded in the sampler itself (set_epoch below). REVE never applied args.seed.\n'
            '    import random as _random\n'
            '    import numpy as _np\n'
            '    import torch as _torch\n'
            '    _seed = int(args.seed) + accelerator.process_index\n'
            '    _random.seed(_seed); _np.random.seed(_seed); _torch.manual_seed(_seed); _torch.cuda.manual_seed_all(_seed)\n'
            '    accelerator.print(f"seed {args.seed} (torch: + rank)")\n')
patch("src/train.py", P10E_OLD, P10E_NEW, "patch 10e (train seed)")
P10F_OLD = ('    for epoch in pbar:\n'
            '        start = time.time()\n')
P10F_NEW = ('    for epoch in pbar:\n'
            '        _bs = train_loader.batch_sampler  # PATCH 10: REVE\'s GroupedSampler, inside Accelerate\'s shard on >1 GPU\n'
            '        getattr(_bs, "batch_sampler", _bs).set_epoch(epoch)\n'
            '        start = time.time()\n')
patch("src/train.py", P10F_OLD, P10F_NEW, "patch 10f (set_epoch per epoch)")
EOF
