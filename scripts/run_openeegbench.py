"""OpenEEGBench on Expanse, one backbone per job, on the public (ungated) tasks.

  run        evaluate one set of weights with the `open_eeg_bench` package and write a CSV of per-seed scores
  summarize  turn those CSVs into one markdown table next to the leaderboard's REVE-Base rows, reve-base-released (no package needed)
  compare    per-task side-by-side tables, chance-adjusted only, with the difference to a reference label (default reve-base-retested)
             and the seed spread, one table per strategy

  python scripts/run_openeegbench.py run --label reve-base-retested --hub-repo brain-bzh/reve-base --out results/evals/openeegbench
  python scripts/run_openeegbench.py run --label reve-small-run1-ep10 --checkpoint $CKPTROOT/small-run1/export/reve_braindecode_epfinal.pth \
         --out results/evals/openeegbench --cache $ARTROOT/oeb_cache
  python scripts/run_openeegbench.py summarize results/evals/openeegbench --out results/evals/openeegbench/SUMMARY.md
  python scripts/run_openeegbench.py compare   results/evals/openeegbench --out results/evals/openeegbench/COMPARE.md
  python scripts/run_openeegbench.py run --label reve-small-run1-ep2 --checkpoint … --datasets arithmetic_zyma2019 --seeds 10
             # a top-up: only the new cells are computed (exca cache) and they are MERGED into the label's CSV

One label = one set of weights. The label names the CSV, and a top-up merges into it cell by cell, so two checkpoints
under one label overwrite each other silently — which is what happened to base-run1's epoch-5 scores on 2026-09-21.
Every row therefore records a `weights` identity (Hub repo, or the SHA-256 of the exported state dict) and a merge
across identities is refused. Label snapshots `reve-<run>-ep<k>`, never `reve-<model>-<corpus>-ep<k>`.

Backbone config = the package's own REVE preset (`default_configs.backbones.reve`: braindecode REVE, window z-score clipped
at 15 SD, LoRA on the feed-forward Linears) with the weights swapped: `--hub-repo` (Hub) or `--checkpoint` (a braindecode
state dict from export_encoder.py; the export's head is ignored, the benchmark attaches its own). Strategies follow the
leaderboard: `frozen` (its "probe") and `full_finetune`. Scores are balanced accuracy on the test split,
mean over seeds; the summary also prints the chance-adjusted value (BA - 1/K)/(1 - 1/K), which is what the leaderboard
appears to show (0 = chance).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

# Hub datasets under braindecode/, ungated. ISRUC is 13 GB, CHB-MIT 4.7 GB, and BCIC-2020-3 is near chance for every model
# on the board, so the large ones and FACED are not in the default set; the write-up rows run all eight (--datasets all)
TASKS = {  # name -> (hf dataset id, number of classes, display name); the leaderboard's public tasks plus CHB-MIT, which it does not score
    "bcic2a": ("braindecode/bcic2a", 4, "BCIC-IV-2a"),
    "physionet": ("braindecode/physionet", 4, "PhysioNet-MI"),
    "bcic2020_3": ("braindecode/bcic2020-3", 5, "BCIC-2020-3"),
    "isruc_sleep": ("braindecode/isruc-sleep", 5, "ISRUC"),
    "faced": ("braindecode/faced", 9, "FACED"),
    "mdd_mumtaz2016": ("braindecode/mdd_mumtaz2016", 2, "Mumtaz"),
    "arithmetic_zyma2019": ("braindecode/arithmetic_zyma2019", 2, "Zyma"),
    "chbmit": ("braindecode/chbmit", 2, "CHB-MIT"),   # seizure detection, 10 s windows, subject split fixed by the package; no leaderboard column
}
# Phase 1, the default: five quick tasks (minutes each for Small). Phase 2: FACED (45 min for Small), ISRUC (13 GB download,
# sequence task) and CHB-MIT (4.7 GB download). The job runs phase 1, writes the tables, then phase 2 when asked (DATASETS=all), so quick scores land first.
QUICK_TASKS = ["bcic2a", "physionet", "bcic2020_3", "mdd_mumtaz2016", "arithmetic_zyma2019"]
SLOW_TASKS = ["faced", "isruc_sleep", "chbmit"]
DEFAULT_TASKS = QUICK_TASKS
STRATEGIES = ["frozen", "full_finetune"]   # registry names (default_configs.ALL_FINETUNING); the leaderboard calls them probe / full_finetune
INFRA_MODE = "retry"   # exca: keep completed cells, recompute failed ones (default "cached" reloads a failure as failed — job 54361573
                       # lost 6 cells to Hugging Face 429 rate limits during the first dataset fetch and needs exactly this)


def model_kwargs_for(checkpoint: str | None, base_kwargs: dict | None = None) -> dict:
    """The size the model must be built with. braindecode's REVE defaults to Base (22 layers); a Small snapshot
    (4 layers) would leave 108 tensors "missing" (job 54362047), so read embed_dim/depth/heads off the file."""
    kwargs = dict(base_kwargs or {})
    if checkpoint:
        import torch
        from export_encoder import arch_from_state_dict   # same folder

        kwargs.update(arch_from_state_dict(torch.load(checkpoint, map_location="cpu")))
    return kwargs


def adjusted(ba: float, k: int) -> float:
    """Chance-adjusted balanced accuracy: 0 at chance (1/K), 1 when perfect."""
    return (ba - 1 / k) / (1 - 1 / k)


def run(a) -> None:
    import open_eeg_bench as oeb
    from open_eeg_bench.default_configs.backbones import reve

    if bool(a.hub_repo) == bool(a.checkpoint):
        raise SystemExit("give exactly one of --hub-repo or --checkpoint")
    bb = reve(hub_repo=a.hub_repo, checkpoint_path=a.checkpoint)
    tasks = {"all": QUICK_TASKS + SLOW_TASKS, "quick": QUICK_TASKS, "slow": SLOW_TASKS}.get((a.datasets or ["quick"])[0], a.datasets) if not a.datasets or len(a.datasets) == 1 else a.datasets
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; known: {list(TASKS)}")
    infra = {"folder": a.cache, "mode": INFRA_MODE}   # exca's cache: finished cells reused, failed ones recomputed on a re-run
    model_kwargs = model_kwargs_for(a.checkpoint, bb.model_kwargs)
    print(f"label {a.label}  weights {a.hub_repo or a.checkpoint}  model_kwargs {model_kwargs}  tasks {tasks}  strategies {a.strategies}  seeds {a.seeds}  device {a.device}")
    df = oeb.benchmark(bb.model_cls, hub_repo=bb.hub_repo, checkpoint_path=bb.checkpoint_path, model_kwargs=model_kwargs,
                       peft_target_modules=bb.peft_target_modules, peft_ff_modules=bb.peft_ff_modules,
                       head_module_name=bb.head_module_name, normalization=bb.normalization,
                       datasets=tasks, finetuning_strategies=a.strategies, n_seeds=a.seeds, infra=infra,
                       max_workers=a.max_workers, device=a.device)
    df = pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df
    df.insert(0, "label", a.label)
    df.insert(1, "weights", weights_id(a.hub_repo, a.checkpoint))   # so a label can never silently cover two models
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{a.label}.csv"
    df = merge_rows(pd.read_csv(p), df, force=a.force) if p.exists() else df
    df.to_csv(p, index=False)
    print(f"wrote {p} ({len(df)} rows)")
    if "test_balanced_accuracy" in df:
        print(df.groupby(["finetuning", "dataset"])["test_balanced_accuracy"].agg(["mean", "std", "count"]).round(3))
    failed = df[df.get("status", "completed") != "completed"] if "status" in df else df.iloc[0:0]
    if len(failed):
        print(f"WARNING: {len(failed)} experiment(s) not completed:\n{failed[['dataset', 'finetuning', 'seed'] + [c for c in ('status', 'exception') if c in failed]].to_string()}")


KEY = ["dataset", "finetuning", "seed"]


def weights_id(hub_repo: str | None, checkpoint: str | None) -> str:
    """Which model produced a row. A Hub row is named by its repo; a snapshot by the SHA-256 of the exported state dict,
    so two runs' epoch-5 exports are never mistaken for each other however their files are named or moved."""
    if hub_repo:
        return f"hub:{hub_repo}"
    h = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


def merge_rows(existing: pd.DataFrame, new: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Upsert `new` into `existing` on (dataset, strategy, seed): a top-up run (more seeds on one task) adds rows and
    replaces the cells it recomputed, and never drops the rest of the label's CSV.

    Refuses when the label's CSV already holds rows from *different* weights, which is how base-run1's epoch-5 scores
    were silently overwritten by base-run2's on 2026-09-21: both runs were labelled `reve-base-12k-ep5`, the keys
    matched cell for cell, and the upsert replaced every one of them. A label names one set of weights; use a
    run-scoped label (`reve-<run>-ep<k>`) instead. CSVs written before the `weights` column existed carry no identity
    and are merged as before."""
    if not force and "weights" in new:
        old = set(existing["weights"].dropna().unique()) if "weights" in existing else set()
        clash = old - set(new["weights"].unique())
        if clash:
            raise SystemExit(
                f"refusing to merge: this CSV already holds rows from other weights ({', '.join(sorted(clash))}) "
                f"and the new rows come from {', '.join(sorted(set(new['weights'].unique())))}. A label names one set "
                f"of weights — give this run its own label (e.g. reve-<run>-ep<k>), or pass --force to overwrite.")
    keep = existing.merge(new[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    keep = existing[(keep["_merge"] == "left_only").values]
    return pd.concat([keep, new], ignore_index=True).sort_values(KEY).reset_index(drop=True)


def load_runs(folder: str) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, "*.csv")))]
    frames = [f for f in frames if "test_balanced_accuracy" in f]
    if not frames:
        raise SystemExit(f"no benchmark CSVs with test_balanced_accuracy in {folder}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["test_balanced_accuracy"].notna()]
    return df


def leaderboard_rows(folder: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(folder, "leaderboard_*.json")))
    if not files:
        return []
    rows = json.load(open(files[-1]))
    return [r for r in rows if r.get("fullname") == "REVE"]


def summarize(a) -> None:
    df = load_runs(a.folder)
    by_id = {v[0]: (k, v[1], v[2]) for k, v in TASKS.items()}
    tasks = [t for t in TASKS if TASKS[t][0] in set(df["dataset"])] or list(TASKS)
    df["task"] = df["dataset"].map(lambda d: by_id.get(d, (d, None, d))[0])
    g = df.groupby(["label", "finetuning", "task"])["test_balanced_accuracy"].agg(["mean", "std", "count"])
    # the mean column must compare like with like: average over the tasks EVERY row has (rows evaluated after
    # FACED was dropped from the default would otherwise average 4 tasks against someone else's 5)
    have = {k: {t for t in tasks if (k[0], k[1], t) in g.index} for k in {(l, f) for l, f, _ in g.index}}
    common = [t for t in tasks if all(t in v for v in have.values())] or tasks
    lines = ["| model | strategy | " + " | ".join(TASKS[t][2] for t in tasks) + f" | mean adj. ({len(common)} tasks) |",
             "|---|---|" + "---:|" * (len(tasks) + 1)]

    def cells(vals: dict[str, float | None]) -> str:
        out = ["—" if vals.get(t) is None else f"{vals[t]:.3f} ({adjusted(vals[t], TASKS[t][1]):.3f})" for t in tasks]
        adj = [adjusted(vals[t], TASKS[t][1]) for t in common if vals.get(t) is not None]
        return " | ".join(out) + (f" | {sum(adj) / len(adj):.3f}" if len(adj) == len(common) else " | —")

    for (label, strat), sub in df.groupby(["label", "finetuning"]):
        vals = {t: float(g.loc[(label, strat, t), "mean"]) for t in tasks if (label, strat, t) in g.index}
        counts = sub.groupby("task")["seed"].nunique() if "seed" in sub else pd.Series(dtype=int)
        n = f"{counts.min()} seeds" if len(counts) and counts.min() == counts.max() else f"{counts.min()}–{counts.max()} seeds, incomplete" if len(counts) else "no seeds"
        lines.append(f"| {label} | {strat} ({n}) | {cells(vals)} |")
    for r in leaderboard_rows(a.folder):   # the arena's published REVE-Base rows (reve-base-released), already chance-adjusted
        adj = {t: r.get(f"{t}_accuracy") for t in tasks}
        got = [adj[t] for t in common if adj.get(t) is not None]
        lines.append(f"| reve-base-released (leaderboard) | {r['adapter']} | " + " | ".join("—" if adj[t] is None else f"— ({adj[t]:.3f})" for t in tasks)
                     + (f" | {sum(got) / len(got):.3f} |" if len(got) == len(common) else " | — |"))
    text = ("Balanced accuracy on the test split, mean over seeds; in brackets the chance-adjusted value (BA − 1/K)/(1 − 1/K). "
            f"The mean column averages the adjusted values over the {len(common)} task(s) every row has "
            f"({', '.join(TASKS[t][2] for t in common)}). reve-base-released rows are the leaderboard's published scores (saved snapshot), adjusted values only.\n\n"
            + "\n".join(lines) + "\n")
    if a.out:
        Path(a.out).write_text(text); print(f"wrote {a.out}")
    print(text)


def compare(a) -> None:
    """One table per strategy: rows = labels, columns = tasks (chance-adjusted), plus the difference to --ref per task
    and on the shared-task mean. The leaderboard's published REVE-Base rows (reve-base-released) are appended for the same tasks."""
    df = load_runs(a.folder)
    by_id = {v[0]: k for k, v in TASKS.items()}
    df["task"] = df["dataset"].map(by_id)
    tasks = [t for t in TASKS if t in set(df["task"])]
    g = df.groupby(["label", "finetuning", "task"])["test_balanced_accuracy"].agg(["mean", "std", "count"])
    labels = list(dict.fromkeys(df["label"]))
    if a.ref not in labels:
        raise SystemExit(f"--ref {a.ref!r} has no CSV in {a.folder}; labels: {labels}")
    out = []
    for strat in sorted(set(df["finetuning"])):
        have = {l: {t for t in tasks if (l, strat, t) in g.index} for l in labels if any((l, strat, t) in g.index for t in tasks)}
        common = [t for t in tasks if all(t in v for v in have.values())] or tasks
        ref = {t: adjusted(g.loc[(a.ref, strat, t), "mean"], TASKS[t][1]) for t in tasks if (a.ref, strat, t) in g.index}
        ref_mean = sum(ref[t] for t in common) / len(common) if all(t in ref for t in common) else None
        out.append(f"\n### {strat} — chance-adjusted balanced accuracy, mean over seeds; Δ = difference to `{a.ref}`\n")
        out.append("| model | " + " | ".join(f"{TASKS[t][2]} (Δ)" for t in tasks) + f" | mean of {len(common)} shared (Δ) |")
        out.append("|---|" + "---:|" * (len(tasks) + 1))
        for l in have:
            cells = []
            for t in tasks:
                if (l, strat, t) not in g.index:
                    cells.append("—"); continue
                r = g.loc[(l, strat, t)]; v = adjusted(r["mean"], TASKS[t][1])
                d = f" ({v - ref[t]:+.3f})" if t in ref and l != a.ref else ""
                cells.append(f"{v:.3f}{d}" + (f" ±{r['std'] / (1 - 1 / TASKS[t][1]):.3f}" if r["count"] > 1 else "") + f" n={int(r['count'])}")
            adj = [adjusted(g.loc[(l, strat, t), "mean"], TASKS[t][1]) for t in common if (l, strat, t) in g.index]
            mean = sum(adj) / len(adj) if len(adj) == len(common) else None
            dm = f" ({mean - ref_mean:+.3f})" if mean is not None and ref_mean is not None and l != a.ref else ""
            out.append(f"| {'**' + l + '**' if l == a.ref else l} | " + " | ".join(cells) + f" | {'—' if mean is None else f'{mean:.3f}{dm}'} |")
        for r in leaderboard_rows(a.folder):
            adapter = {"probe": "frozen", "full_finetune": "full"}.get(r["adapter"], r["adapter"])
            if adapter != strat:
                continue
            vals = {t: r.get(f"{t}_accuracy") for t in tasks}
            got = [vals[t] for t in common if vals.get(t) is not None]
            mean = f"{sum(got) / len(got):.3f}" if len(got) == len(common) else "—"   # e.g. CHB-MIT, which the leaderboard lacks
            out.append(f"| reve-base-released (leaderboard, {r['adapter']}) | " + " | ".join("—" if vals[t] is None else f"{vals[t]:.3f}" for t in tasks)
                       + f" | {mean} |")
    out.append("\n±: seed standard deviation on the adjusted scale; n: seeds. A Δ smaller than about 2× the larger ± of the pair is noise at this seed count.\n")
    text = "\n".join(out)
    if a.out:
        Path(a.out).write_text(text); print(f"wrote {a.out}")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate one set of weights")
    r.add_argument("--label", required=True, help="row name, reve-<model>-<source>[-ep<k>]: reve-base-retested, reve-small-random-init, reve-small-run1-ep10; one label = one set of weights")
    r.add_argument("--force", action="store_true", help="merge even though the label's CSV holds rows from other weights (overwrites them)")
    r.add_argument("--hub-repo", default=None); r.add_argument("--checkpoint", default=None)
    r.add_argument("--datasets", nargs="+", default=None, help=f"task names, or quick (default: {QUICK_TASKS}), slow ({SLOW_TASKS}), all")
    r.add_argument("--strategies", nargs="+", default=STRATEGIES)
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--device", default="cuda")
    r.add_argument("--max-workers", type=int, default=1, help="experiments in flight; 1 on a single GPU")
    r.add_argument("--cache", default=os.path.expanduser("~/.cache/exca"), help="exca folder (finished cells are reused)")
    r.add_argument("--out", default="results/evals/openeegbench")
    r.set_defaults(fn=run)
    s = sub.add_parser("summarize", help="markdown table from the CSVs in a folder")
    s.add_argument("folder", nargs="?", default="results/evals/openeegbench")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=summarize)
    c = sub.add_parser("compare", help="per-task side-by-side tables (adjusted only) with deltas to a reference label")
    c.add_argument("folder", nargs="?", default="results/evals/openeegbench")
    c.add_argument("--ref", default="reve-base-retested", help="label the deltas are measured against")
    c.add_argument("--out", default=None)
    c.set_defaults(fn=compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
