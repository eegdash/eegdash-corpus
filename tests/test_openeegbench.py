"""run_openeegbench.py: the summary (no package needed) groups per-seed CSVs by label and strategy, reports raw and
chance-adjusted balanced accuracy, and places the leaderboard's REVE rows on the same tasks; the sbatch is well-formed."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from run_openeegbench import (DEFAULT_TASKS, INFRA_MODE, QUICK_TASKS, REVISIONS, SLOW_TASKS, STRATEGIES, TASKS, adjusted, audit, expand_tasks, stage_repo,
                               fetch, merge_rows, model_kwargs_for, weights_id)

REPO = Path(__file__).resolve().parents[1]


def test_strategy_names_are_registry_names():
    assert STRATEGIES == ["frozen", "full_finetune"]   # "full" is the config kind, not a registry key: job 54361519 failed on it


def test_all_eight_open_tasks_are_known():
    assert set(TASKS) == {"bcic2a", "physionet", "bcic2020_3", "isruc_sleep", "faced", "mdd_mumtaz2016", "arithmetic_zyma2019", "chbmit"}
    assert TASKS["isruc_sleep"][1] == 5 and TASKS["bcic2020_3"][1] == 5
    assert TASKS["chbmit"] == ("braindecode/chbmit", 2, "CHB-MIT")


def test_quick_and_slow_phases_cover_the_eight_open_tasks():
    assert DEFAULT_TASKS == QUICK_TASKS == ["bcic2a", "physionet", "bcic2020_3", "mdd_mumtaz2016", "arithmetic_zyma2019"]
    assert SLOW_TASKS == ["faced", "isruc_sleep", "chbmit"] and set(QUICK_TASKS + SLOW_TASKS) == set(TASKS)


def test_chbmit_rows_render_and_the_leaderboard_shows_a_dash(tmp_path):
    """CHB-MIT rows map back to their task (braindecode/chbmit -> chbmit, column CHB-MIT); the leaderboard has no
    CHB-MIT score, so its rows print "—" there, and the mean stays over the tasks every row has."""
    rows = []
    for label, tasks in (("reve-base-retested", ["bcic2a", "chbmit"]), ("reve-small-run1-epfinal", ["bcic2a", "chbmit"]), ("old", ["bcic2a"])):
        for strat in ("frozen",):
            for task in tasks:
                for seed in range(2):
                    rows.append(dict(label=label, dataset=TASKS[task][0], finetuning=strat, seed=seed, status="completed",
                                     test_balanced_accuracy=0.75))
    for label, sub in pd.DataFrame(rows).groupby("label"):
        sub.to_csv(tmp_path / f"{label}.csv", index=False)
    (tmp_path / "leaderboard_2026-09-19.json").write_text(json.dumps([{"fullname": "REVE", "adapter": "probe", "bcic2a_accuracy": 0.19}]))
    script = str(REPO / "scripts/run_openeegbench.py")
    r = subprocess.run([sys.executable, script, "summarize", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "| model | strategy | BCIC-IV-2a | CHB-MIT | mean adj. (1 tasks) |" in r.stdout      # "old" lacks CHB-MIT
    assert "| reve-base-retested | probe (2 seeds) | 0.750 (0.667) | 0.750 (0.500) | 0.667 |" in r.stdout
    assert "| reve-base-released (leaderboard) | probe | — (0.190) | — | 0.190 |" in r.stdout
    c = subprocess.run([sys.executable, script, "compare", str(tmp_path)], capture_output=True, text=True)
    assert c.returncode == 0, c.stdout + c.stderr
    assert "CHB-MIT (Δ)" in c.stdout and "| reve-base-released (leaderboard, probe) | 0.190 | — | 0.190 |" in c.stdout
    (tmp_path / "old.csv").unlink()                                   # every row has CHB-MIT: the leaderboard cannot be averaged
    c = subprocess.run([sys.executable, script, "compare", str(tmp_path)], capture_output=True, text=True)
    assert "| reve-base-released (leaderboard, probe) | 0.190 | — | — |" in c.stdout     # the row is kept, not dropped


def test_exca_mode_retries_failed_cells():
    assert INFRA_MODE == "retry"   # "cached" would reload the 6 rate-limited cells of job 54361573 as failed forever


def test_adjusted_balanced_accuracy():
    assert adjusted(0.25, 4) == 0 and adjusted(1.0, 4) == 1 and abs(adjusted(0.514, 4) - 0.352) < 1e-3   # REVE full FT on BCIC-IV-2a
    assert abs(adjusted(0.737, 2) - 0.474) < 1e-3                                                        # our ds003810 probe


def test_summary_table(tmp_path):
    rows = []
    for label, base in (("reve-base-retested", 0.6), ("reve-small-run1-ep10", 0.4)):
        for strat in ("frozen", "full_finetune"):
            for task in ["bcic2a", "physionet", "faced", "mdd_mumtaz2016", "arithmetic_zyma2019"]:   # the five tasks the reference rows were run on
                hf = TASKS[task][0]
                for seed in range(2):
                    rows.append(dict(label=label, backbone="REVE", dataset=hf, training="default", finetuning=strat, head="linear",
                                     seed=seed, status="completed", test_balanced_accuracy=base + 0.01 * seed))
    df = pd.DataFrame(rows)
    for label, sub in df.groupby("label"):
        sub.to_csv(tmp_path / f"{label}.csv", index=False)
    lb = [{"fullname": "REVE", "adapter": "probe", "bcic2a_accuracy": 0.1911, "physionet_accuracy": 0.3767, "faced_accuracy": 0.1672,
           "mdd_mumtaz2016_accuracy": 0.6811, "arithmetic_zyma2019_accuracy": 0.2954, "#Params (M)": 69.19},
          {"fullname": "BIOT", "adapter": "probe", "bcic2a_accuracy": 0.0183}]
    (tmp_path / "leaderboard_2026-09-19.json").write_text(json.dumps(lb))
    out = tmp_path / "summary.md"
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "summarize", str(tmp_path), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    text = out.read_text()
    assert "| reve-base-retested | probe (2 seeds) | 0.605 (0.473)" in text            # mean over seeds 0.605; adjusted for K=4
    assert "| reve-small-run1-ep10 | fine-tune (2 seeds) | 0.405 (0.207)" in text
    assert text.count("| reve-base-released (leaderboard) | probe |") == 1 and "BIOT" not in text
    assert "| model | strategy | BCIC-IV-2a | PhysioNet-MI | FACED | Mumtaz | Zyma | mean adj. (5 tasks) |" in text
    # a label with a missing seed on one task is flagged, not silently averaged as complete
    part = pd.read_csv(tmp_path / "reve-base-retested.csv"); part = part[~((part.dataset == "braindecode/faced") & (part.seed == 1) & (part.finetuning == "frozen"))]
    part["label"] = "partial"; part.to_csv(tmp_path / "partial.csv", index=False)
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "summarize", str(tmp_path)], capture_output=True, text=True)
    assert "| partial | probe (1–2 seeds, incomplete) |" in r.stdout and "| partial | fine-tune (2 seeds) |" in r.stdout


def test_mean_column_uses_only_the_tasks_every_row_has(tmp_path):
    """A row run without FACED must not be averaged over 4 tasks against a 5-task row: the mean uses the shared set."""
    rows = []
    for label, base, tasks in (("five-task", 0.6, ["bcic2a", "physionet", "faced", "mdd_mumtaz2016", "arithmetic_zyma2019"]), ("four-task", 0.6, ["bcic2a", "physionet", "mdd_mumtaz2016", "arithmetic_zyma2019"])):
        for strat in ("frozen",):
            for task in tasks:
                for seed in range(2):
                    rows.append(dict(label=label, backbone="REVE", dataset=TASKS[task][0], training="d", finetuning=strat,
                                     head="linear", seed=seed, status="completed", test_balanced_accuracy=base))
    df = pd.DataFrame(rows)
    for label, sub in df.groupby("label"):
        sub.to_csv(tmp_path / f"{label}.csv", index=False)
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "summarize", str(tmp_path)],
                       capture_output=True, text=True, check=True)
    assert "mean adj. (4 tasks)" in r.stdout                     # FACED excluded from the mean, still a column
    assert "| FACED |" in r.stdout and "| four-task | probe (2 seeds) | 0.600 (0.467) | 0.600 (0.467) | — |" in r.stdout
    five = [l for l in r.stdout.splitlines() if l.startswith("| five-task")][0]
    four = [l for l in r.stdout.splitlines() if l.startswith("| four-task")][0]
    assert five.rsplit("|", 2)[1].strip() == four.rsplit("|", 2)[1].strip()   # identical scores -> identical mean


def test_run_requires_exactly_one_weight_source():
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "run", "--label", "x"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and ("exactly one" in r.stderr or "No module named" in r.stderr)


def test_model_kwargs_follow_the_checkpoint_size(tmp_path):
    """A Small export must build a 4-layer REVE in the benchmark (job 54362047 built Base and found 108 tensors missing)."""
    from braindecode.models import REVE

    small = REVE(n_outputs=2, n_chans=19, n_times=2000, sfreq=200, depth=4,
                 chs_info=[{"ch_name": n} for n in ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2"]])
    p = tmp_path / "small.pth"; torch.save(small.state_dict(), p)
    assert model_kwargs_for(str(p), {"x": 1}) == {"x": 1, "embed_dim": 512, "depth": 4, "heads": 8}
    assert model_kwargs_for(None, {"x": 1}) == {"x": 1}          # Hub weights: the preset's kwargs untouched


def test_merge_rows_upserts_and_keeps_the_rest():
    """A 10-seed Zyma top-up under the same label must add seeds 3-9, replace seeds 0-2 for Zyma, and leave every other task."""
    base = pd.DataFrame([dict(dataset=TASKS[t][0], finetuning="frozen", seed=s, test_balanced_accuracy=0.5) for t in DEFAULT_TASKS for s in range(3)])
    top = pd.DataFrame([dict(dataset=TASKS["arithmetic_zyma2019"][0], finetuning="frozen", seed=s, test_balanced_accuracy=0.9) for s in range(10)])
    m = merge_rows(base, top)
    assert len(m) == (len(DEFAULT_TASKS) - 1) * 3 + 10   # the other quick tasks untouched, Zyma replaced by 10 seeds
    z = m[m.dataset == TASKS["arithmetic_zyma2019"][0]]
    assert len(z) == 10 and (z.test_balanced_accuracy == 0.9).all()          # recomputed cells replaced, new seeds added
    assert (m[m.dataset == TASKS["bcic2a"][0]].test_balanced_accuracy == 0.5).all() and len(m[m.dataset == TASKS["bcic2a"][0]]) == 3


def _rows(weights, acc, tasks=DEFAULT_TASKS, seeds=3):
    return pd.DataFrame([dict(dataset=TASKS[t][0], weights=weights, finetuning="frozen", seed=s, test_balanced_accuracy=acc)
                         for t in tasks for s in range(seeds)])


def test_merge_refuses_rows_from_other_weights():
    """The 2026-09-21 loss: base-run1 and base-run2 both labelled reve-base-12k-ep5, so the second run's cells replaced
    the first's key for key, with nothing in the file to show it had happened."""
    run1, run2 = _rows("sha256:1111111111111111", 0.30), _rows("sha256:2222222222222222", 0.80)
    with pytest.raises(SystemExit, match="refusing to merge"):
        merge_rows(run1, run2)
    m = merge_rows(run1, run2, force=True)                       # the deliberate overwrite is still available
    assert len(m) == len(run1) and (m.test_balanced_accuracy == 0.80).all()
    same = merge_rows(run1, _rows("sha256:1111111111111111", 0.55, tasks=["bcic2a"]))    # a top-up of the same weights
    assert len(same) == len(run1) and (same[same.dataset == TASKS["bcic2a"][0]].test_balanced_accuracy == 0.55).all()
    legacy = merge_rows(run1.drop(columns=["weights"]), run2)    # CSVs written before the column existed still merge
    assert len(legacy) == len(run1)


def test_weights_id_follows_the_file_not_its_name(tmp_path):
    a, b, copy = tmp_path / "ep5.pth", tmp_path / "ep5.pth.other", tmp_path / "moved.pth"
    a.write_bytes(b"encoder-one"); b.write_bytes(b"encoder-two"); copy.write_bytes(b"encoder-one")
    assert weights_id(None, str(a)) == weights_id(None, str(copy)) != weights_id(None, str(b))
    assert weights_id("brain-bzh/reve-base", None) == "hub:brain-bzh/reve-base"


def test_compare_tables_show_deltas_to_the_reference(tmp_path):
    rows = []
    for label, base in (("reve-base-retested", 0.6), ("reve-small-run1-ep2", 0.7)):
        for strat in ("frozen", "full"):
            for task in ["bcic2a", "physionet", "mdd_mumtaz2016", "arithmetic_zyma2019"]:
                for seed in range(3):
                    rows.append(dict(label=label, backbone="REVE", dataset=TASKS[task][0], training="d", finetuning=strat, head="linear",
                                     seed=seed, status="completed", test_balanced_accuracy=base + 0.01 * seed))
    df = pd.DataFrame(rows)
    for label, sub in df.groupby("label"):
        sub.to_csv(tmp_path / f"{label}.csv", index=False)
    (tmp_path / "leaderboard_2026-09-19.json").write_text(json.dumps([{"fullname": "REVE", "adapter": "probe", "bcic2a_accuracy": 0.19,
        "physionet_accuracy": 0.38, "mdd_mumtaz2016_accuracy": 0.68, "arithmetic_zyma2019_accuracy": 0.30}]))
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "compare", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "### probe" in r.stdout and "### fine-tune" in r.stdout   # the tables say probe and fine-tune (2026-11-03); the CSV keeps the package's names
    assert "| **reve-base-retested** |" in r.stdout                                    # the reference row is marked and carries no delta
    # small: raw 0.71 -> adjusted (0.71-0.25)/0.75 = 0.613 on 4-class, delta vs base (0.61 -> 0.480) = +0.133
    assert "0.613 (+0.133)" in r.stdout
    assert "| reve-base-released (leaderboard, probe) |" in r.stdout and "(leaderboard, fine-tune" not in r.stdout   # only under its own strategy
    bad = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "compare", str(tmp_path), "--ref", "nope"],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "has no CSV" in bad.stderr


def _hf_snapshot(root, files):
    """A Hugging Face cache layout: content stored once under blobs/, every file of the snapshot a link to its blob."""
    import hashlib
    import os

    snap = root / "snapshots" / "rev"
    for path, data in files.items():
        blob = root / "blobs" / hashlib.sha256(data).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True); blob.write_bytes(data)
        f = snap / path; f.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, f.parent), f)
    return snap


def test_cache_audit_finds_missing_wrong_and_extra_files(tmp_path):
    """Identical files sharing one blob are fine; a missing file, a wrong size, a wrong hash and a local-only file are reported."""
    import hashlib

    same = b"x" * 10
    snap = _hf_snapshot(tmp_path, {"a/1.bin": same, "a/2.bin": same, "b.bin": b"short", "c.bin": b"y" * 8, "local-only.tmp": b"z"})
    remote = [{"path": "a/1.bin", "size": 10, "sha256": hashlib.sha256(same).hexdigest()},
              {"path": "a/2.bin", "size": 10, "sha256": hashlib.sha256(same).hexdigest()},   # deduplicated on disk: still correct
              {"path": "b.bin", "size": 9, "sha256": None},                                  # wrong size
              {"path": "c.bin", "size": 8, "sha256": "0" * 64},                              # right size, wrong content
              {"path": "d.bin", "size": 3, "sha256": None}]                                  # missing
    assert len(list((tmp_path / "blobs").iterdir())) == 4                                      # two files, one blob
    r = audit(snap, remote)
    assert r["missing"] == ["d.bin"] and r["wrong"] == ["b.bin"] and r["extra"] == ["local-only.tmp"] and r["files"] == 5
    assert audit(snap, remote, hashes=True)["wrong"] == ["b.bin", "c.bin"]                     # the hash catches what the size cannot
    assert audit(tmp_path / "snapshots" / "absent", remote)["missing"] == sorted(x["path"] for x in remote)


def test_cache_fetch_waits_out_a_429_and_throttles():
    """A 429 is waited out and the file retried; other errors stop; every file is re-downloaded, not trusted from the cache."""
    calls, sleeps = [], []

    def download(repo, path, **kw):
        calls.append((path, kw["revision"], kw["force_download"]))
        if path == "b" and calls.count(("b", "r", True)) == 1:
            raise RuntimeError("429 Client Error: Too Many Requests")

    assert fetch("braindecode/x", "r", ["a", "b"], per_second=5, download=download, sleep=sleeps.append) == 2
    assert [c[0] for c in calls] == ["a", "b", "b"] and 60 in sleeps and sleeps.count(0.2) == 2
    with pytest.raises(RuntimeError, match="404"):
        fetch("braindecode/x", "r", ["c"], download=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("404 Not Found")), sleep=lambda s: None)


def test_merge_upserts_cells_from_another_cluster_and_refuses_other_weights(tmp_path):
    """Two cells scored on another machine go into the label's 42-row CSV: the failed rows are replaced, the rest kept;
    rows from other weights are refused unless forced."""
    def rows(weights, cells):
        return pd.DataFrame([dict(label="reve-base-v0-run2-epfinal", weights=weights, dataset=TASKS[t][0], finetuning=f, seed=s,
                                  status=st, test_balanced_accuracy=acc) for t, f, s, st, acc in cells])
    into = tmp_path / "into.csv"; new = tmp_path / "new.csv"
    rows("sha256:aa", [("faced", "frozen", 0, "failed", None), ("faced", "frozen", 1, "completed", 0.25), ("chbmit", "frozen", 0, "failed", None)]).to_csv(into, index=False)
    rows("sha256:aa", [("faced", "frozen", 0, "completed", 0.26), ("chbmit", "frozen", 0, "completed", 0.8)]).to_csv(new, index=False)
    r = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "merge", str(into), str(new)], capture_output=True, text=True)
    assert r.returncode == 0 and "3 rows + 2 merged -> 3 rows, 0 not completed" in r.stdout, r.stdout + r.stderr
    got = pd.read_csv(into)
    assert list(got["status"]) == ["completed"] * 3 and got.loc[got.dataset == TASKS["faced"][0]].sort_values("seed")["test_balanced_accuracy"].tolist() == [0.26, 0.25]
    rows("sha256:bb", [("faced", "frozen", 2, "completed", 0.3)]).to_csv(new, index=False)
    bad = subprocess.run([sys.executable, str(REPO / "scripts/run_openeegbench.py"), "merge", str(into), str(new)], capture_output=True, text=True)
    assert bad.returncode != 0 and "refusing to merge" in bad.stderr and len(pd.read_csv(into)) == 3   # untouched


def test_every_task_has_a_pinned_revision():
    assert set(REVISIONS) == set(TASKS) and all(len(v) == 40 for v in REVISIONS.values())


def test_expand_tasks_takes_names_and_groups():
    assert expand_tasks(None) == QUICK_TASKS and expand_tasks(["all"]) == QUICK_TASKS + SLOW_TASKS
    assert expand_tasks(["quick", "faced", "chbmit"]) == QUICK_TASKS + ["faced", "chbmit"]   # the job's DATASETS line, in order, no repeats
    with pytest.raises(SystemExit, match="unknown task"):
        expand_tasks(["bcic2a", "nope"])


def test_stage_copies_one_revision_with_its_blobs_and_skips_what_is_there(tmp_path):
    """The destination is a Hugging Face cache huggingface_hub reads offline: the snapshot's links, the blobs they point
    to (one copy per content), refs; other revisions stay behind; a second run copies nothing; a missing snapshot stops."""
    import os

    src, dst = tmp_path / "src", tmp_path / "dst"
    same = b"x" * 10
    folder = "datasets--braindecode--bcic2a"
    _hf_snapshot(src / folder, {"a/1.bin": same, "a/2.bin": same, "b.bin": b"other"})
    (src / folder / "refs").mkdir(); (src / folder / "refs" / "main").write_text("rev")
    old = src / folder / "snapshots" / "old"; old.mkdir(); (old / "stale.bin").write_bytes(b"stale")   # an older revision, not wanted
    r = stage_repo(src, dst, folder, "rev")
    assert (r["files"], r["copied"], r["skipped"]) == (3, 15, 0)                       # two files share one blob: copied once
    snap = dst / folder / "snapshots" / "rev"
    assert (snap / "a" / "1.bin").is_symlink() and (snap / "a" / "1.bin").read_bytes() == same and (snap / "b.bin").read_bytes() == b"other"
    assert os.readlink(snap / "a" / "1.bin") == os.readlink(snap / "a" / "2.bin") and len(list((dst / folder / "blobs").iterdir())) == 2
    assert (dst / folder / "refs" / "main").read_text() == "rev" and not (dst / folder / "snapshots" / "old").exists()
    assert stage_repo(src, dst, folder, "rev")["copied"] == 0 and stage_repo(src, dst, folder, "rev")["skipped"] == 15   # incremental
    assert stage_repo(src, dst, folder)["files"] == 4                                     # no revision: every snapshot (the model repos)
    with pytest.raises(SystemExit, match="no snapshot"):
        stage_repo(src, dst, folder, "absent")
    with pytest.raises(SystemExit, match="no snapshot"):
        stage_repo(src, dst, "datasets--braindecode--faced", "rev")
