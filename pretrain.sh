#!/bin/bash
# Pretrain REVE-Base from scratch on eegdash-corpus-v0 with the paper's settings (17 epochs, peak learning rate
# 2.4e-4, global batch 300), with REVE's own trainer (reve_eeg) patched to read the Zarr store.
#
#   DATA=/data/eegdash-corpus-v0/eegdash-corpus-v0.zarr CKPT=/ckpt/reve-base-v0 bash pretrain.sh
#
# One machine: every GPU it has is used. Several machines (the paper: 16 GPUs on 4 machines): run the same line once
# on each machine, with the number of machines, this machine's index and the first machine's name:
#
#   NNODES=4 NODE_RANK=0 MAIN_HOST=node01 DATA=... CKPT=... bash pretrain.sh     # on node01; NODE_RANK=1, 2, 3 on the others
#
# REVE's patched code is expected beside this repository (../reve_eeg, as the Quick start clones it); REVE_EEG names
# another place. Run again with the same CKPT, it continues after the last saved epoch. It writes $CKPT/run/encoder.pth (the
# pretrained encoder) and $CKPT/run/loss.csv. Every other setting (SEED, EPOCHS, ENCODER=small, WORKERS,
# MIXED_PRECISION, ...) is train.sh's, which this script calls: its header lists them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${DATA:?set DATA to the eegdash-corpus-v0.zarr store (download.sh prints it)}"
: "${CKPT:?set CKPT to a directory for the checkpoints}"
export DATA CKPT SEED=${SEED:-2} REVE_EEG=${REVE_EEG:-$(dirname "$HERE")/reve_eeg}   # REVE's code: beside this repository unless REVE_EEG says otherwise
export RUN=${RUN:-reve-base-v0-seed$SEED}

grep -q "PATCH 8" "$REVE_EEG/src/utils/data_loading.py" 2> /dev/null || {
  echo "no patched reve_eeg at $REVE_EEG: clone it at 06a7059 and run patches/apply_patches.sh (README, Quick start)" >&2; exit 1; }

GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}                    # the GPUs of this machine
export NNODES=${NNODES:-1}
export NPROC=$((NNODES * GPUS))                           # processes over all machines, one per GPU
if [ "$NNODES" -gt 1 ]; then                              # where the machines find each other
  : "${MAIN_HOST:?several machines: set MAIN_HOST to the name of the first machine}"
  : "${NODE_RANK:?several machines: set NODE_RANK to the index of this machine, 0 on the first}"
  export ACCELERATE_ARGS="--machine_rank $NODE_RANK --main_process_ip $MAIN_HOST --main_process_port 29500"
fi
if [ -d "$CKPT/states/$RUN/last" ]; then export RESUME=True; fi
bash "$HERE/train.sh"
