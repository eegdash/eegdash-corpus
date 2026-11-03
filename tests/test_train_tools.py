"""Training job helpers: log parsing / epoch estimate, loss plot, sbatch syntax, Hydra override names."""

import csv
import subprocess
import sys

import pytest

from estimate_epochs import estimate, parse

LOG = "\r".join(
    f"Epoch   0 (it. {i:3d}/ 400) Loss (EMA/loss):{1.0 - i * 0.001:.3f}/{0.95 - i * 0.001:.3f} LR 0.00100000, shape  37, time {1.5 * i + 3:.1f}"
    for i in range(0, 120, 5)
)


def test_parse_and_estimate():
    rows = parse(LOG)
    assert len(rows) == 24 and rows[0]["n_iter"] == 400 and rows[-1]["it"] == 115
    info = estimate(rows, budget_hours=24)
    assert abs(info["seconds_per_iteration"] - 1.5) < 0.01
    assert abs(info["epoch_hours"] - 1.5 * 400 / 3600) < 0.01
    assert info["epochs_in_budget"] == int(24 // (1.5 * 400 / 3600))
    assert info["nan_seen"] is False


def test_nan_detected():
    rows = parse("Epoch   0 (it.   3/ 10) Loss (EMA/loss):nan/nan LR 0.00100000, shape   2, time 5.0")
    assert estimate(rows, 1)["nan_seen"] is True


def test_plot_loss(tmp_path, scripts_on_path):
    csv_p = tmp_path / "loss.csv"
    csv_p.write_text("epoch,loss_ema,iterations,seconds\n0,0.91,36,60.0\n1,0.85,36,58.0\n")
    log_p = tmp_path / "train.out"
    log_p.write_text(LOG)
    out = tmp_path / "loss.png"
    # two torn lines (the ranks write over each other): a mangled iteration count and a loss in the hundreds
    log_p.write_text(LOG + "\rEpoch   0 (it. 300/  4) Loss (EMA/loss):0.900/0.900 LR 0.00100000, shape  37, time 9.0"
                     + "\rEpoch   1 (it.  10/ 400) Loss (EMA/loss):0.850/270.0 LR 0.00100000, shape  37, time 20.0"
                     + "\rEpoch   1 (it.  20/ 400) Loss (EMA/loss):0.850/0.840 LR 0.00100000, shape  37, time 30.0")
    r = subprocess.run([sys.executable, str(scripts_on_path / "plot_loss.py"), str(csv_p), "--log", str(log_p), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 1000 and "25 logged steps kept" in r.stdout   # 24 well-formed of epoch 0 + 1 of epoch 1
    rows = list(csv.DictReader(open(tmp_path / "loss-epochs.csv")))
    assert [r["epoch"] for r in rows] == ["0", "1"] and rows[0]["first_loss"] == "0.9500" and rows[0]["logged_steps"] == "24"
    assert abs(float(rows[0]["mean_loss"]) - (0.95 - 0.001 * sum(range(0, 120, 5)) / 24)) < 1e-3
    assert rows[0]["loss_csv_ema"] == "0.9100" and rows[1]["mean_loss"] == "0.8400"   # the 270 is dropped


def test_train_overrides_are_valid_hydra_keys(reve_src, tmp_path):
    """Every override used in train_reve.sbatch must exist in REVE's config (a typo would only fail on the GPU node)."""
    overrides = ["encoder=small", "trainer.lr=0.00024", "data.path=/x", "data.subset=all", "data.loader.num_workers=8", "data.loader.prefetch_factor=4",
                 "trainer.epochs=10", "trainer.batch_size=300", "trainer.n_nodes=1", "checkpointing.path=/x",
                 "checkpointing.load_last_state=False", "name=v0", f"hydra.run.dir={tmp_path}", "mode=train",
                 # the W&B block train_reve.sbatch adds when WANDB_ENTITY is set
                 "wandb.log=True", "wandb.offline=True", "wandb.path=/x/wandb", "wandb.entity=someone",
                 "wandb.project=reve-retrain", "wandb.comment=v0", "seed=1"]   # SEED knob -> seed (patch 10)
    r = subprocess.run([sys.executable, str(reve_src / "train.py"), "--cfg", "job", *overrides],
                       capture_output=True, text=True, cwd=str(reve_src.parent))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "epochs: 10" in r.stdout and "batch_size: 300" in r.stdout and "n_nodes: 1" in r.stdout
    assert "depth: 4" in r.stdout      # encoder=small (ENCODER knob) swaps the encoder group: 4 layers instead of 22
    assert "lr: 0.00024" in r.stdout   # LR knob -> trainer.lr (the paper's Table 5 value; run1 diverged at the default 1e-3)
    assert "seed: 1" in r.stdout


def test_parse_survives_lines_torn_by_other_ranks():
    """Job 3479398: sixteen ranks wrote over each other and one line read `time 3793.4.`; parse must skip it, not die."""
    from estimate_epochs import parse
    text = ("Epoch   3 (it. 100/10435) Loss (EMA/loss):0.150/0.120 LR 0.00024000, shape  38, time 60.0:   0%|\r"
            "Epoch   3 (it. 200/10435) Loss (EMA/loss):0.149/0.118 LR 0.00024000, shape  38, time 3793.4.\r"
            "E):0.152/0.157 LR 0.00024000, shape  37, time 2210.2:  24%|\r"
            "Epoch   3 (it. 300/10435) Loss (EMA/loss):0.148/nan LR 0.00024000, shape  38, time 180.0:   0%|\r")
    rows = parse(text)
    assert [r["it"] for r in rows] == [100, 200, 300] and rows[1]["seconds"] == 3793.4
    assert rows[2]["loss"] != rows[2]["loss"]   # nan kept as nan

