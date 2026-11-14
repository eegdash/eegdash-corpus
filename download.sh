#!/bin/bash
# Download eegdash-corpus-v0, the converted Zarr store (3.6 TB, 44,801 files), from Hugging Face.
#
#   bash download.sh /data                      # -> /data/eegdash-corpus-v0/eegdash-corpus-v0.zarr
#   VERIFY=1 JOBS=16 bash download.sh /data     # also check every file against the published SHA-256 list
#
# Needs a Hugging Face account that accepted the dataset's terms, a token on this machine (`hf auth login`) and the
# newest Hugging Face client, in an environment of its own (README, Quick start, step 1). The training environment
# pins an older client that spends a request per file, and a free account gets 1,000 per five minutes: with it the
# script still finishes, pausing whenever the Hub refuses. After an interruption, run the same line again.
set -euo pipefail
OUT=${1:?usage: bash download.sh <a directory with 4 TB free>}
DEST=$OUT/eegdash-corpus-v0

hf auth whoami > /dev/null || { echo "not logged in: run 'hf auth login' and accept the terms on the dataset's page" >&2; exit 1; }
python - "$DEST" << 'EOF'
import os, sys, time, traceback
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

dest = Path(sys.argv[1])
listed = dest / "eegdash-corpus-v0-SHA256SUMS"                 # names every file of the store
for per_file in ("huggingface_hub.xet_get", "huggingface_hub.http_get"):
    disable_progress_bars(per_file)                            # leaves one bar that counts the finished files
while True:
    try:
        snapshot_download("EEGDash/eegdash-corpus-v0", repo_type="dataset", revision="v1.0", local_dir=dest)
        if listed.exists() and all((dest / line.split()[1]).exists() for line in listed.open()):
            break                                              # all there (a refused request can also return early)
    except Exception as error:
        if "429 Client Error" not in "".join(traceback.format_exception(error)):
            raise                                              # anything but the rate limit: stop
    print("paused by the Hub's rate limit: finished files are kept, going on in a minute", flush=True)
    time.sleep(float(os.environ.get("WAIT", 60)))
EOF

if [ "${VERIFY:-0}" = 1 ]; then   # hashes every file, JOBS at a time: hours on the full store
  parts=$(mktemp -d)
  split -l 500 "$DEST/eegdash-corpus-v0-SHA256SUMS" "$parts/part."
  (cd "$DEST" && ls "$parts"/part.* | xargs -P "${JOBS:-8}" -n 1 sha256sum --quiet -c)
  rm -r "$parts"
  echo "checksums OK"
fi
echo "DATA=$DEST/eegdash-corpus-v0.zarr"
