"""Order a shard list for short conversion jobs: measured minutes per shard, largest first, finished shards left out.

  python order_shards.py --shards $CONVROOT/plan/v0/all_shards.tsv --logs $LOGS \
      --parts $CONVROOT/parts_zarr --out $CONVROOT/plan/v0/zarr_todo.tsv

Minutes come from an earlier conversion of the same shards (the converter's `done …, <n> s` line in
logs/shard-<id>.out, the memmap run of 2026-09-18/19); the preprocessing is the same, so the time carries over. The
output adds an `est_min` column, which convert_worker.sbatch uses to claim only shards that fit its remaining time,
so 1-2 h jobs (backfilled sooner than long ones) never start a shard they cannot finish. Shards without a measured
time get an empty `est_min` and fall back to the worker's RESERVE rule. Largest first, so the long shards go to the
first workers while they still have their whole time.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

DONE_LINE = re.compile(r"^done .*, (\d+(?:\.\d+)?) s ->", re.M)


def measured_minutes(logs: Path, shard_ids: list[str], prefix: str = "shard-") -> dict[str, int]:
    """Whole minutes (rounded up) of the last completed conversion of each shard, where its log has one."""
    out = {}
    for sid in shard_ids:
        log = logs / f"{prefix}{sid}.out"
        if log.exists():
            hits = DONE_LINE.findall(log.read_text(errors="replace"))
            if hits:
                out[sid] = max(1, math.ceil(float(hits[-1]) / 60))
    return out


def order(shards: Path, logs: Path, parts: Path, out: Path, prefix: str = "shard-") -> dict:
    with open(shards, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fields = (list(rows[0].keys()) if rows else ["shard_id"]) + ["est_min"]   # before est_min is added to the rows
    todo = [r for r in rows if not (parts / r["shard_id"] / "_DONE").exists()]
    minutes = measured_minutes(logs, [r["shard_id"] for r in todo], prefix)
    for r in todo:
        r["est_min"] = minutes.get(r["shard_id"], "")
    todo.sort(key=lambda r: -(r["est_min"] if r["est_min"] != "" else 10**6))  # unmeasured first: unknown size
    with open(out, "w", newline="") as f:
        # "\n", not csv's default "\r\n": bash reads the last column as "87\r" and its arithmetic fails (2026-09-24,
        # every worker on this list exited in seconds)
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(todo)
    return dict(total=len(rows), done=len(rows) - len(todo), todo=len(todo), measured=len(minutes),
                minutes=sum(minutes.values()), largest=max(minutes.values(), default=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="the plan's shards.tsv")
    ap.add_argument("--logs", required=True, help="folder with shard-<id>.out logs of an earlier conversion")
    ap.add_argument("--parts", required=True, help="parts folder of the conversion to continue (skips _DONE)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="shard-", help="log name prefix (shard- for the memmap run)")
    a = ap.parse_args()
    s = order(Path(a.shards), Path(a.logs), Path(a.parts), Path(a.out), a.prefix)
    print(f"{s['todo']} of {s['total']} shards to do ({s['done']} done); {s['measured']} with a measured time, "
          f"{s['minutes'] / 60:.0f} worker-hours, largest {s['largest']} min -> {a.out}")


if __name__ == "__main__":
    main()
