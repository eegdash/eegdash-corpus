"""apply_patches.sh applies every edit to a pristine reve_eeg checkout and is idempotent."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RETRAIN = Path(__file__).resolve().parents[1]
FILES = ("src/train.py", "src/utils/data_loading.py", "src/utils/ddp_setup.py")


@pytest.fixture
def pristine(reve_src, tmp_path):
    """Copy the three patched files from the checkout, then undo our edits to get the 06a7059 text."""
    root = tmp_path / "reve_eeg"
    for f in FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(reve_src.parent / f, root / f)
    undo = {
        "src/utils/data_loading.py": [('            if len(index_data) > 1 and index_data[-1]["flag_reduce"] != index_data[-2]["flag_reduce"]:',
                                       '            if index_data[-1]["flag_reduce"] != index_data[-2]["flag_reduce"]:')],
        "src/train.py": [("    mae = accelerator.prepare(mae)  # PATCH 4: DDP-wrapped on >1 GPU; the MAE type check rejected it\n",
                          "    mae = ensure_type(accelerator.prepare(mae), MAE)\n"),
                         ('            if args.wandb.log and batch_idx % 100 == 0:  # PATCH 6: every 100 iterations, epoch as a fraction\n'
                          '                accelerator.log({"epoch": epoch + batch_idx / max(1, n_iter_per_train), "it": batch_idx,\n'
                          '                                 "loss_ema": loss_ema, "loss": loss_g, "lr": optimizer.param_groups[0]["lr"]})\n',
                          '            if args.wandb.log:\n'
                          '                accelerator.log({"epoch": epoch, "it": batch_idx, "loss_ema": loss_ema, "loss": loss_g})\n')],
        "src/utils/ddp_setup.py": [('        if args.wandb.offline:  # PATCH 5: offline only when asked, not for every lustre path\n',
                                    '        if "lustre" in args.data.path or args.wandb.offline:\n'),
                                   ('                "entity": args.wandb.entity,  # PATCH 5: was the literal "online" when not offline\n',
                                    '                "entity": args.wandb.entity if args.wandb.offline else "online",\n')],
    }
    for f, pairs in undo.items():
        p = root / f
        s = p.read_text()
        for new, old in pairs:
            s = s.replace(new, old)
        if f == "src/utils/data_loading.py":  # strip patch 8: the dispatch lines and the appended block
            s = s.replace('    if str(args.data.path).rstrip("/").endswith(".zarr"):  # PATCH 8: the published Zarr corpus\n'
                          '        return _zarr_train_loader(args)\n', "")
            if "\n\n# ---- PATCH 8" in s:
                s = s[:s.index("\n\n# ---- PATCH 8")]
        if f == "src/utils/data_loading.py":  # patch 10 a-d back to REVE's unseeded sampler
            s = s.replace('    def __init__(self, dataset, batch_size, drop_last, n_gpu, mode="train", seed=None):\n'
                          '        self.seed, self.epoch = seed, 0  # PATCH 10: seeded, the shuffle is the same on every rank (set_epoch)\n',
                          '    def __init__(self, dataset, batch_size, drop_last, n_gpu, mode="train"):\n')
            s = s.replace('            rng = random if self.seed is None else random.Random(self.seed + self.epoch)  # PATCH 10: per epoch, per seed\n', "")
            s = s.replace('                group_indices = self.groups[c] if self.seed is None else list(self.groups[c])  # a copy: epoch k depends on seed + k only\n'
                          '                rng.shuffle(group_indices)\n',
                          '                group_indices = self.groups[c]\n                random.shuffle(group_indices)\n')
            s = s.replace('            rng.shuffle(indices)\n', '            random.shuffle(indices)\n')
            s = s.replace('    def set_epoch(self, epoch):  # PATCH 10: train.py calls it before each epoch, so a resumed run shuffles as a fresh one would\n'
                          '        self.epoch = epoch\n\n', "")
            s = s.replace('        seed=args.seed if train else None,  # PATCH 10\n', "")
        if f == "src/train.py" and "# PATCH 10" in s:  # patch 10 e-f back to REVE's unseeded loop
            a, b = s.index("    # PATCH 10: torch seeded per rank"), s.index('    accelerator.print(f"seed {args.seed} (torch: + rank)")\n')
            s = s[:a] + s[b + len('    accelerator.print(f"seed {args.seed} (torch: + rank)")\n'):]
            s = s.replace("        _bs = train_loader.batch_sampler  # PATCH 10: REVE's GroupedSampler, inside Accelerate's shard on >1 GPU\n"
                          '        getattr(_bs, "batch_sampler", _bs).set_epoch(epoch)\n', "")
        if f == "src/train.py" and "# PATCH 9" in s:  # patch 9 back to REVE's resume lines
            a, b = s.index("    first_epoch = 0\n"), s.index("    pbar = tqdm(range(first_epoch, args.trainer.epochs))\n")
            s = (s[:a] + '    if args.checkpointing.load_last_state:\n'
                 '        accelerator.load_state(pjoin(args.checkpointing.state_path, "last"))\n\n'
                 '    pbar = tqdm(range(args.trainer.epochs))\n' + s[b + len("    pbar = tqdm(range(first_epoch, args.trainer.epochs))\n"):])
        if f == "src/train.py":  # strip the patch 3 + 7 block (from the PATCH 3 line to the blank line after it)
            start = s.find("        if accelerator.is_main_process:  # PATCH 3")
            if start != -1:
                end = s.find("\n\n", start)
                s = s[:start] + s[end + 1:]
        p.write_text(s)
    return root


def run(root):
    return subprocess.run(["bash", str(RETRAIN / "patches" / "apply_patches.sh"), str(root)],
                          capture_output=True, text=True, env={"PYTHON": sys.executable, "PATH": "/usr/bin:/bin"})


def test_apply_then_noop(pristine):
    r = run(pristine)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("applied (") == 16 and "already" not in r.stdout, r.stdout
    train = (pristine / "src/train.py").read_text()
    assert "ensure_type(accelerator.prepare(mae), MAE)" not in train and all(f"PATCH {k}" in train for k in (3, 4, 6, 7, 9, 10))
    ddp = (pristine / "src/utils/ddp_setup.py").read_text()
    assert '"lustre" in args.data.path' not in ddp and 'else "online"' not in ddp
    dl = (pristine / "src/utils/data_loading.py").read_text()
    assert "len(index_data) > 1" in dl and "PATCH 8" in dl and dl.count("class ZarrEEGDataset") == 1
    assert dl.count("seed=args.seed") == 2 and "def set_epoch" in dl   # patch 10: both loaders seed the sampler
    r2 = run(pristine)
    assert r2.returncode == 0 and r2.stdout.count("already applied") == 16, r2.stdout


def _resume_epochs(train_text, state_dir, env=None, epochs=17):
    """Run the patched resume block of train.py against a states folder; return the epochs the loop would run."""
    import os, textwrap
    from os.path import join as pjoin
    from types import SimpleNamespace as NS
    a = train_text.index("    first_epoch = 0\n")
    b = train_text.index("    pbar = tqdm(range(first_epoch, args.trainer.epochs))\n")
    block = textwrap.dedent(train_text[a:b]) + "pbar = list(range(first_epoch, args.trainer.epochs))\n"
    loaded = []
    acc = NS(load_state=loaded.append, print=lambda *x: None)
    args = NS(checkpointing=NS(load_last_state=True, state_path=str(state_dir)), trainer=NS(epochs=epochs))
    old = dict(os.environ)
    os.environ.update(env or {})
    try:
        ns = {"os": os, "pjoin": pjoin, "accelerator": acc, "args": args}
        exec(block, ns)
    finally:
        os.environ.clear(); os.environ.update(old)
    assert loaded == [pjoin(str(state_dir), "last")]
    return ns["pbar"]


def test_a_resumed_run_continues_after_its_last_saved_epoch(pristine, tmp_path):
    """DeltaAI test 3292081 (2026-10-02): a resume looped from epoch 0 again, so it would have run 17 more epochs and
    overwritten the first part's snapshots. Now base-run3 (epoch_5, epoch_6 saved) resumes at 7, base-run4 (epoch_10)
    at 11, and a copied state holding only `last` starts where REVE_START_EPOCH says."""
    run(pristine)
    train = (pristine / "src/train.py").read_text()
    states = tmp_path / "states"
    for d in ("epoch_5", "epoch_6", "last"):
        (states / d).mkdir(parents=True)
    assert _resume_epochs(train, states) == list(range(7, 17))
    (states / "epoch_10").mkdir()
    assert _resume_epochs(train, states) == list(range(11, 17))
    copy = tmp_path / "copy"; (copy / "last").mkdir(parents=True)
    assert _resume_epochs(train, copy) == list(range(0, 17))
    assert _resume_epochs(train, copy, {"REVE_START_EPOCH": "11"}) == list(range(11, 17))


def test_zarr_loader_replaced_not_duplicated_when_it_changes(pristine):
    """Editing zarr_loader.py and re-running on Expanse must leave one copy of the block, the new one."""
    run(pristine)
    dl = pristine / "src/utils/data_loading.py"
    dl.write_text(dl.read_text().replace("class _ZarrSamples:", "class _ZarrSamples:  # older version"))
    r = run(pristine)
    assert r.returncode == 0 and "patch 8b (Zarr loader): applied" in r.stdout, r.stdout
    s = dl.read_text()
    assert s.count("# ---- PATCH 8") == 1 and "older version" not in s


def test_dry_shards_tsv_parses():
    rows = [l.split("\t") for l in (RETRAIN / "scripts" / "dry_shards.tsv").read_text().splitlines()]
    assert rows[0] == ["shard_id", "dataset", "mode", "subjects", "n_records", "est_hours"]
    assert [r[1] for r in rows[1:]] == ["ds004840", "ds005262", "nm000146"]
    assert all(len(r) == 6 for r in rows) and rows[3][2] == "online"


def test_train_seeds_torch_per_rank_from_the_config(pristine):
    """Patch 10: rank r seeds torch with args.seed + r (DDP then broadcasts rank 0's weights); the print names the seed."""
    import textwrap
    from types import SimpleNamespace as NS
    import torch
    run(pristine)
    train = (pristine / "src/train.py").read_text()
    a, b = train.index("    # PATCH 10: torch seeded per rank"), train.index('    accelerator.print(f"seed {args.seed} (torch: + rank)")\n')
    block = textwrap.dedent(train[a:b + len('    accelerator.print(f"seed {args.seed} (torch: + rank)")\n')])
    printed = []
    for rank in (0, 3):
        exec(block, {"args": NS(seed=42), "accelerator": NS(process_index=rank, print=printed.append)})
        assert torch.initial_seed() == 42 + rank
    assert printed == ["seed 42 (torch: + rank)"] * 2
    assert 'getattr(_bs, "batch_sampler", _bs).set_epoch(epoch)' in train
