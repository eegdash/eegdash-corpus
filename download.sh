#!/bin/bash
# Download eegdash-corpus-v0, the converted Zarr store (3.6 TB, 44,801 files), from Hugging Face.
#
#   bash download.sh /data                      # -> /data/eegdash-corpus-v0/eegdash-corpus-v0.zarr
#   VERIFY=1 JOBS=16 bash download.sh /data     # also check every file against the published SHA-256 list
#
# Needs a Hugging Face account that accepted the dataset's terms and a token on this machine (`hf auth login`).
# The download resumes: after an interruption, run the same line again.
set -euo pipefail
OUT=${1:?usage: bash download.sh <a directory with 4 TB free>}
DEST=$OUT/eegdash-corpus-v0

hf auth whoami > /dev/null || { echo "not logged in: run 'hf auth login' and accept the terms on the dataset's page" >&2; exit 1; }
hf download EEGDash/eegdash-corpus-v0 --repo-type dataset --revision v1.0 --local-dir "$DEST"

if [ "${VERIFY:-0}" = 1 ]; then   # hashes every file, JOBS at a time: hours on the full store
  parts=$(mktemp -d)
  split -l 500 "$DEST/eegdash-corpus-v0-SHA256SUMS" "$parts/part."
  (cd "$DEST" && ls "$parts"/part.* | xargs -P "${JOBS:-8}" -n 1 sha256sum --quiet -c)
  rm -r "$parts"
  echo "checksums OK"
fi
echo "DATA=$DEST/eegdash-corpus-v0.zarr"
