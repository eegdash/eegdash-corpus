#!/bin/bash
# Pretrain REVE from scratch on eegdash-corpus-v0 with reve_eeg's own trainer (src/train.py), patched by
# patches/apply_patches.sh so that a data.path ending in .zarr is read from the store. Cluster-neutral: run it once per
# node, inside whatever scheduler you use.
#
#   DATA=/data/eegdash-corpus-v0.zarr RUN=reve-base-v0-run2 CKPT=/ckpt/reve-base-v0-run2 REVE_EEG=~/reve_eeg SEED=2 NPROC=4 bash train.sh
#
# Knobs (environment variables):
#   DATA              the Zarr store                                   DATA=/data/eegdash-corpus-v0.zarr
#   RUN               run name (Hydra name=)                           RUN=reve-base-v0-run2
#   CKPT              checkpoint directory of the run                  CKPT=/ckpt/reve-base-v0-run2
#   REVE_EEG          the patched reve_eeg checkout (commit 06a7059)   REVE_EEG=~/reve_eeg
#   SEED              required: initialisation and batch order         SEED=2            (the paper's runs used 2 and 3)
#   NNODES            number of machines, default 1                    NNODES=4
#   NPROC             processes (GPUs) over all machines, default 1    NPROC=16
#   ENCODER           REVE's Hydra group encoder=, default base        ENCODER=small
#   EPOCHS            default 17                                       EPOCHS=17
#   BS                global batch size, default 300                   BS=300
#   LR                peak learning rate, default 2.4e-4               LR=2.4e-4
#   WORKERS           DataLoader workers per process, default 8        WORKERS=8
#   EXCLUDE_DATASETS  optional, comma-separated dataset ids left out   EXCLUDE_DATASETS=ds004395,nm000338
#                     at load (exported as REVE_EXCLUDE_DATASETS, patch 8)
#   RESUME            continue from $CKPT's last state, default False  RESUME=True
#   MIXED_PRECISION   default fp16 (V100); bf16 on A100/H100           MIXED_PRECISION=bf16
#   REVE_ENCODER_EVERY  encoder snapshot every N epochs, default 5     REVE_ENCODER_EVERY=0   (0 = off, patch 7)
#   ACCELERATE_ARGS   extra flags passed to `accelerate launch`        ACCELERATE_ARGS="--machine_rank 1 --main_process_ip node01 --main_process_port 29500"
#
# Multi-node launches need the usual Accelerate rendezvous flags on every node (--machine_rank, --main_process_ip,
# --main_process_port), passed through ACCELERATE_ARGS; the paper's runs derived them from the scheduler (machine rank =
# the node's index, main process = the first node, port 29500) and added --multi_gpu when NNODES > 1, as this script does.
#
# Writes under $CKPT: states/$RUN/epoch_* and states/$RUN/last (Accelerate states, for RESUME=True) and run/ (Hydra's
# working directory: encoder.pth, encoder_epoch_<k>.pth, loss.csv, .hydra/). Weights & Biases logging is off.
set -euo pipefail

: "${DATA:?set DATA to the eegdash-corpus-v0.zarr store}"
: "${RUN:?set RUN to the run name}"
: "${CKPT:?set CKPT to the run's checkpoint directory}"
: "${REVE_EEG:?set REVE_EEG to the patched reve_eeg checkout}"
: "${SEED:?set SEED (the paper's runs used 2 and 3)}"
NNODES=${NNODES:-1}; NPROC=${NPROC:-1}
ENCODER=${ENCODER:-base}; EPOCHS=${EPOCHS:-17}; BS=${BS:-300}; LR=${LR:-2.4e-4}; WORKERS=${WORKERS:-8}
RESUME=${RESUME:-False}; MIXED_PRECISION=${MIXED_PRECISION:-fp16}
export REVE_EXCLUDE_DATASETS=${EXCLUDE_DATASETS:-}
export REVE_ENCODER_EVERY=${REVE_ENCODER_EVERY:-5}
MULTI=""; [ "$NNODES" -gt 1 ] && MULTI="--multi_gpu"

echo "run $RUN  encoder $ENCODER  lr $LR  seed $SEED  nodes $NNODES  processes $NPROC  epochs $EPOCHS  batch $BS  resume $RESUME  data $DATA${EXCLUDE_DATASETS:+  excluding $EXCLUDE_DATASETS}  start $(date '+%F %T')"
mkdir -p "$CKPT"
cd "$REVE_EEG"
# shellcheck disable=SC2086  # ACCELERATE_ARGS and MULTI are meant to split into words
accelerate launch --num_processes "$NPROC" --num_machines "$NNODES" --mixed_precision "$MIXED_PRECISION" $MULTI ${ACCELERATE_ARGS:-} \
    src/train.py encoder="$ENCODER" \
    data.path="$DATA" data.subset=all data.loader.num_workers="$WORKERS" data.loader.prefetch_factor=4 \
    trainer.epochs="$EPOCHS" trainer.batch_size="$BS" trainer.n_nodes="$NNODES" trainer.lr="$LR" seed="$SEED" \
    checkpointing.path="$CKPT" checkpointing.load_last_state="$RESUME" name="$RUN" \
    hydra.run.dir="$CKPT/run" wandb.log=false mode=train
echo "end $(date '+%F %T')"
