"""Read a train.py log and say how many epochs fit in a time budget.

train.py's progress bar prints, per iteration:
  Epoch   0 (it.  35/ 36) Loss (EMA/loss):0.913/0.884 LR 0.00000100, shape   2, time 57.7
where `time` is seconds since the epoch started and `36` is iterations per epoch (per process).

  python estimate_epochs.py $LOGS/train-<jobid>.out --budget-hours 24
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# A number is digits with at most one decimal point: sixteen ranks write the job log over each other, and a torn line
# can read `time 3793.4.` (job 3479398, 2026-10-30), which `[\d.]+` swallowed and float() refused.
NUM = r"(nan|\d+(?:\.\d+)?)"
LINE = re.compile(rf"Epoch\s+(\d+) \(it\.\s*(\d+)/\s*(\d+)\) Loss \(EMA/loss\):{NUM}/{NUM} .*?time\s+(\d+(?:\.\d+)?)")


def parse(text: str) -> list[dict]:
    """Every well-formed progress line; a line torn by another rank's output is skipped, not fatal."""
    rows = []
    for m in LINE.finditer(text.replace("\r", "\n")):
        try:
            rows.append(dict(epoch=int(m[1]), it=int(m[2]), n_iter=int(m[3]), loss_ema=float(m[4]), loss=float(m[5]),
                             seconds=float(m[6])))
        except ValueError:
            continue
    return rows


def estimate(rows: list[dict], budget_hours: float, warmup_iters: int = 20) -> dict:
    """Seconds per iteration from the steady part of the latest epoch; epochs that fit the budget."""
    if not rows:
        raise SystemExit("no progress lines found - is this a train.py log?")
    last_epoch = rows[-1]["epoch"]
    ep = [r for r in rows if r["epoch"] == last_epoch and r["it"] >= warmup_iters]
    if len(ep) < 2:
        ep = [r for r in rows if r["epoch"] == last_epoch]
    a, b = ep[0], ep[-1]
    sec_per_it = (b["seconds"] - a["seconds"]) / max(1, b["it"] - a["it"])
    n_iter = b["n_iter"]
    epoch_h = sec_per_it * n_iter / 3600
    nan = any(r["loss"] != r["loss"] for r in rows)
    return dict(iterations_per_epoch=n_iter, seconds_per_iteration=round(sec_per_it, 2), epoch_hours=round(epoch_h, 2),
                epochs_in_budget=int(budget_hours // epoch_h) if epoch_h > 0 else 0, last_loss_ema=rows[-1]["loss_ema"],
                first_loss=rows[0]["loss"], nan_seen=nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--budget-hours", type=float, default=24.0)
    a = ap.parse_args()
    info = estimate(parse(Path(a.log).read_text(errors="replace")), a.budget_hours)
    for k, v in info.items():
        print(f"{k:22s} {v}")
    if info["nan_seen"]:
        print("!! NaN loss seen - fp16 instability; lower trainer.lr before a real run")
    print(f"-> EPOCHS={info['epochs_in_budget']} for a {a.budget_hours:g} h budget on this many GPUs")


if __name__ == "__main__":
    main()
