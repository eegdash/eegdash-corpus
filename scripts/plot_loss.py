"""The pretraining loss of a run, per epoch and per iteration: a table and a figure a reader can act on.

  python plot_loss.py $CKPTROOT/<run>/run/loss.csv --log $LOGS/train-<jobid>.out --out $OUTROOT/runs/<run>/loss.png

Inputs: loss.csv (patch 3: the EMA at the last step of each epoch) and the job log, whose progress lines carry every
iteration's loss and EMA. Outputs: loss.png (every logged step, the EMA and the epoch means over the run), loss-epochs.png (the initial loss at\nepoch 0, then one point per epoch at 1..E) and loss-epochs.csv with one row per epoch: iterations,
seconds, first_loss (the loss of the epoch's first logged step; epoch 0's is the run's initial loss), mean_loss,
median_loss, mean_loss_ema, last_loss_ema (the value loss.csv records). The mean over an epoch is the number to read:
the last-step EMA depends on which channel-count group the final batches came from (a 19-channel and a 129-channel batch
differ a lot in loss), so loss.csv alone jumps around between epochs (reve-base-v0-icare-run1, 2026-10-30).

The loss is REVE's masked-patch L1 reconstruction loss (`F.l1_loss` in models/mae.py), the EMA its 0.95 running mean.
Sixteen ranks write one job log, so a few lines are torn: a row whose iteration count differs from the run's, whose
step exceeds it, or whose loss or EMA is far above the run's median is dropped before anything is drawn.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from estimate_epochs import parse  # noqa: E402


def clean(rows: list[dict], loss_cap_factor: float = 20.0) -> tuple[list[dict], int]:
    """The well-formed rows and the run's iterations per epoch (the most frequent count)."""
    if not rows:
        return [], 0
    n = Counter(r["n_iter"] for r in rows).most_common(1)[0][0]
    med = statistics.median(r["loss"] for r in rows)
    med_ema = statistics.median(r["loss_ema"] for r in rows)
    keep = [r for r in rows if r["n_iter"] == n and r["it"] <= n
            and r["loss"] <= loss_cap_factor * max(med, 1e-6) and r["loss_ema"] <= loss_cap_factor * max(med_ema, 1e-6)]
    return keep, n


def epoch_table(rows: list[dict], csv_rows: list[dict]) -> list[dict]:
    """One row per epoch from the iterations, joined with loss.csv's end-of-epoch EMA."""
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["epoch"]].append(r)
    end = {int(r["epoch"]): r for r in csv_rows}
    out = []
    for e in sorted(set(by) | set(end)):
        it = sorted(by.get(e, []), key=lambda r: r["it"])
        row = {"epoch": e, "iterations": it[0]["n_iter"] if it else end.get(e, {}).get("iterations", ""),
               "logged_steps": len(it), "seconds": end[e]["seconds"] if e in end else (f"{it[-1]['seconds']:.1f}" if it else "")}
        if it:
            losses = [r["loss"] for r in it]
            row.update(first_loss=f"{it[0]['loss']:.4f}", mean_loss=f"{sum(losses) / len(losses):.4f}",
                       median_loss=f"{statistics.median(losses):.4f}",
                       mean_loss_ema=f"{sum(r['loss_ema'] for r in it) / len(it):.4f}", last_loss_ema=f"{it[-1]['loss_ema']:.4f}")
        else:
            row.update(first_loss="", mean_loss="", median_loss="", mean_loss_ema="", last_loss_ema="")
        row["loss_csv_ema"] = f"{float(end[e]['loss_ema']):.4f}" if e in end else ""
        out.append(row)
    return out


def _plain_log_ticks(ax) -> None:
    """A fixed set of plain decimal ticks on the log axis (0.03 … 1), no 10^-1 and no crowded minor labels."""
    lo, hi = ax.get_ylim()
    ticks = [v for v in (0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0) if lo <= v <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())


def draw_epochs(table: list[dict], out: Path, run: str, subtitle: str) -> None:
    """The initial loss at epoch 0, then the mean EMA over each epoch at 1..E, and the EMA at each epoch's last step."""
    ep = [t for t in table if t["mean_loss_ema"]]
    first = next((float(t["first_loss"]) for t in table if t["first_loss"]), None)
    xs = ([0] if first is not None else []) + [t["epoch"] + 1 for t in ep]
    ys = ([first] if first is not None else []) + [float(t["mean_loss_ema"]) for t in ep]
    last_step = [t for t in ep if t["last_loss_ema"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(xs, ys, "-", color="#1f4e79", lw=1.6, zorder=3)
    ax.plot(xs, ys, "o", color="#1f4e79", ms=5, mec="white", mew=0.8, zorder=4, label="mean EMA loss (epoch 0: the initial loss)")
    ax.plot([t["epoch"] + 1 for t in last_step], [float(t["last_loss_ema"]) for t in last_step], "D", ms=4.5, mfc="none",
            mec="#c8102e", mew=1.1, zorder=2, label="EMA loss at the epoch's last step")
    if len(xs) > 1:
        for xx, yy, dx in ((xs[0], ys[0], 6), (xs[-1], ys[-1], 6)):
            ax.annotate(f"{yy:.3f}", (xx, yy), textcoords="offset points", xytext=(dx, 5), fontsize=8, color="#1f4e79")
    allv = ys + [float(t["last_loss_ema"]) for t in last_step]
    ax.set(xlabel="epoch", ylabel="masked-patch L1 loss", yscale="log", ylim=(min(allv) / 1.1, max(allv) * 1.1))
    ax.set_xticks(range(0, len(ep) + 1))
    _plain_log_ticks(ax)
    ax.grid(alpha=0.25, which="major", lw=0.6)
    ax.grid(alpha=0.12, which="minor", lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle(f"REVE pretraining loss, {run}", fontsize=12, y=0.98)
    fig.text(0.5, 0.885, subtitle, ha="center", va="center", fontsize=8.5, color="dimgray")
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("loss_csv")
    ap.add_argument("--log", default=None, help="train.py stdout for the per-iteration curve and the per-epoch table")
    ap.add_argument("--out", default="loss.png", help="the figure; loss-epochs.csv goes beside it")
    ap.add_argument("--epochs-csv", default=None, help="the per-epoch table (default: loss-epochs.csv beside --out)")
    ap.add_argument("--title", default="", help="the run's name for the titles (default: the run folder's name)")
    a = ap.parse_args()
    out = Path(a.out)
    csv_rows = list(csv.DictReader(open(a.loss_csv)))
    rows, n = ([], 0)
    if a.log:
        rows, n = clean(parse(Path(a.log).read_text(errors="replace")))
    table = epoch_table(rows, csv_rows)
    epochs_csv = Path(a.epochs_csv) if a.epochs_csv else out.with_name("loss-epochs.csv")
    with open(epochs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()) if table else ["epoch"])
        w.writeheader()
        w.writerows(table)
    folder = Path(a.loss_csv).resolve().parent          # $CKPTROOT/<run>/run/loss.csv on a cluster, results/runs/<run>/loss.csv here
    run = a.title or (folder.parent.name if folder.name == "run" else folder.name)
    first = next((t["first_loss"] for t in table if t["first_loss"]), "")
    last = next((t["mean_loss_ema"] for t in reversed(table) if t["mean_loss_ema"]), "")
    subtitle = (f"masked-patch L1 reconstruction loss (REVE's MAE), EMA 0.95\n"
                f"{n:,} steps per epoch per GPU, {len(table)} epochs; initial loss {first}, last epoch {last}")
    if rows:
        draw_epochs(table, out, run, subtitle)
    print(f"{len(table)} epochs -> {out} and {epochs_csv}; {len(rows)} logged steps kept")


if __name__ == "__main__":
    main()
