"""Load a REVE layout with reve_eeg's own data_loading code and pull one training batch.

The definitive test that a converted (or merged) directory is what train.py expects:
file sizes match the CSVs, compute_group_segments runs, EEGDataset returns windows, GroupedSampler
builds batches. Exit code 0 = OK.

  REVE_EEG=~/GitHub/reve_eeg python check_layout.py $CONVROOT/memmap
  REVE_EEG=~/GitHub/reve_eeg python check_layout.py <dir>/eegdash-corpus-v0.zarr [--compare <memmap layout>] [--bench 2000]   # or -v1, reve-public-nc

A path ending in .zarr is read through patch 8 (ZarrEEGDataset). --compare checks every item of the store against
the memmap layout converted from the same shards: same channel groups, bit-identical normalised samples and
positions. --bench times N items read in random order through a DataLoader (block masking included, --workers
processes), for both paths when --compare is given: whether Zarr reads keep up with the GPUs.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

WINDOW = 2000
MASKING = dict(window_duration=WINDOW, clip=15, block_masking=True, masking_window=200, masking_overlap=20,
               masking_ratio=0.55, radius_spat_mask=0.03, radius_temp_mask=3, dropout_ratio=0.1, dropout_radius=0.04)


def add_reve_src(reve_eeg: str | None) -> None:
    for cand in (reve_eeg, os.environ.get("REVE_EEG"), "~/GitHub/reve_eeg", "~/Documents/GitHub/reve_eeg"):
        if cand and (Path(cand).expanduser() / "src" / "utils" / "data_loading.py").exists():
            sys.path.insert(0, str(Path(cand).expanduser() / "src"))
            return
    raise SystemExit("reve_eeg checkout not found; pass --reve-eeg or set REVE_EEG")


def check_sizes(layout: Path, big, stats) -> None:
    for b, s in zip(big, stats):
        i, T, C = int(b["big_recording_index"]), int(b["duration"]), int(b["n_chans"])
        assert int(s["big_recording_index"]) == i
        rec = (layout / "recordings" / f"recording_-_eeg_-_{i}.npy").stat().st_size
        st = (layout / "stats" / f"recording_-_stats_-_{i}.npy").stat().st_size
        pos = np.load(layout / "positions" / f"recording_-_positions_-_{i}.npy")
        assert rec == T * C * 4, f"big recording {i}: file has {rec} bytes, CSV says {T}x{C}x4={T * C * 4}"
        assert st == int(s["n_sessions"]) * 2 * C * 4, f"big recording {i}: stats size mismatch"
        assert pos.shape == (C, 3), f"big recording {i}: positions {pos.shape} != ({C}, 3)"


def check_layout(layout: Path, batch_size: int = 300, n_batches: int = 1) -> dict:
    from utils.data_loading import EEGDataset, GroupedSampler, _read_csv, compute_group_segments
    from torch.utils.data import DataLoader

    csvd = layout / "csv_recordings"
    big, corr, stats = (_read_csv(str(csvd / n)) for n in ("df_big.csv", "df_corrected.csv", "df_stats_tmp.csv"))
    assert big and corr and stats, "empty CSVs"
    check_sizes(layout, big, stats)
    for r in corr:
        assert int(r["duration"]) > WINDOW, f"session {r['file']} has duration {r['duration']} <= {WINDOW}"
    groups, segments = compute_group_segments(corr, big, WINDOW)
    ds = EEGDataset(segments, groups, big, stats, str(layout / "recordings"), **MASKING)
    sampler = GroupedSampler(ds, batch_size=batch_size, drop_last=True, n_gpu=1, mode="train")
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    shapes = []
    for k, (x, pos, b_m, b_u) in enumerate(loader):
        assert x.ndim == 3 and x.shape[2] == WINDOW and pos.shape[:2] == x.shape[:2] and pos.shape[2] == 3
        assert np.isfinite(x.numpy()).all() and x.abs().max() <= 15
        shapes.append(tuple(x.shape))
        if k + 1 >= n_batches:
            break
    info = dict(big_recordings=len(big), sessions=len(corr), hours=round(sum(int(b["duration"]) for b in big) / 200 / 3600, 2),
                channel_groups={c: len(v) for c, v in groups.items()}, windows=len(segments), batches=len(sampler),
                batch_shapes=shapes)
    return info


def reve_data_loading():
    from utils import data_loading

    return data_loading


def memmap_dataset(layout: Path, dl=None, **kw):
    """REVE's own EEGDataset over a memmap layout, built the way _build_loader builds it."""
    dl = dl or reve_data_loading()
    csvd = layout / "csv_recordings"
    big, corr, stats = (dl._read_csv(str(csvd / n)) for n in ("df_big.csv", "df_corrected.csv", "df_stats_tmp.csv"))
    groups, segments = dl.compute_group_segments(corr, big, WINDOW)
    return dl.EEGDataset(segments, groups, big, stats, str(layout / "recordings"), **{**MASKING, **kw})


def zarr_dataset(store: Path, dl=None, **kw):
    """Patch 8's ZarrEEGDataset over a store; fails with a clear message on an unpatched checkout."""
    import zarr

    dl = dl or reve_data_loading()
    if not hasattr(dl, "ZarrEEGDataset"):
        raise SystemExit("this reve_eeg checkout has no Zarr loader: bash patches/apply_patches.sh <checkout>")
    groups, segments = dl.zarr_group_segments(zarr.open_group(store, mode="r"), WINDOW)
    return dl.ZarrEEGDataset(str(store), segments, groups, **{**MASKING, **kw})


def check_zarr(store: Path, batch_size: int = 300, n_batches: int = 1, dl=None) -> dict:
    """The store's buckets pass the merge checks and REVE's sampler builds finite, clipped batches from it."""
    from merge_parts import zarr_problems
    from torch.utils.data import DataLoader

    dl = dl or reve_data_loading()
    problems = zarr_problems(store)
    assert not problems, problems
    ds = zarr_dataset(store, dl)
    sampler = dl.GroupedSampler(ds, batch_size=batch_size, drop_last=True, n_gpu=1, mode="train")
    shapes = []
    for k, (x, pos, b_m, b_u) in enumerate(DataLoader(ds, batch_sampler=sampler, num_workers=0)):
        assert x.ndim == 3 and x.shape[2] == WINDOW and pos.shape[:2] == x.shape[:2] and pos.shape[2] == 3
        assert np.isfinite(x.numpy()).all() and x.abs().max() <= 15
        shapes.append(tuple(x.shape))
        if k + 1 >= n_batches:
            break
    return dict(channel_groups={c: len(v) for c, v in ds.groups.items()}, samples=len(ds), batches=len(sampler),
                batch_shapes=shapes)


def compare(layout: Path, store: Path, dl=None) -> dict:
    """Every item of the Zarr store equals the memmap layout's item under the same bucket and session, in order."""
    import torch

    mm, zr = memmap_dataset(layout, dl, no_masking=True), zarr_dataset(store, dl, no_masking=True)
    g_mm, g_zr = {c: len(k) for c, k in mm.groups.items()}, {c: len(k) for c, k in zr.groups.items()}
    assert g_mm == g_zr, f"channel groups differ: memmap {g_mm} vs zarr {g_zr}"
    for k_mm, k_zr in zip(mm.segments, zr.segments):
        assert k_mm.split("_-_")[:2] == k_zr.split("_-_")[:2], f"order differs at {k_mm} vs {k_zr}"
        (x_mm, p_mm), (x_zr, p_zr) = mm[k_mm], zr[k_zr]
        assert torch.equal(x_mm, x_zr), f"sample differs: memmap {k_mm} vs zarr {k_zr}"
        assert torch.equal(p_mm, p_zr), f"positions differ: memmap {k_mm} vs zarr {k_zr}"
    return dict(items_identical=len(mm.segments), channel_groups=g_mm)


def bench(ds, n: int, workers: int, seed: int = 0) -> dict:
    """Items per second for n keys in random order, the access pattern of REVE's shuffled sampler."""
    from torch.utils.data import DataLoader

    keys = random.Random(seed).sample(ds.segments, min(n, len(ds.segments)))
    loader = DataLoader(ds, batch_size=None, sampler=keys, num_workers=workers)
    t0 = time.perf_counter()
    mb = sum(x.numel() * 4 for x, *_ in loader) / 1e6
    dt = time.perf_counter() - t0
    return dict(items=len(keys), seconds=round(dt, 2), items_per_s=round(len(keys) / dt, 1), mb_per_s=round(mb / dt, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout")
    ap.add_argument("--reve-eeg", default=None)
    ap.add_argument("--batch-size", type=int, default=300)
    ap.add_argument("--n-batches", type=int, default=1)
    ap.add_argument("--compare", default=None, help="Zarr store only: the memmap layout converted from the same shards")
    ap.add_argument("--bench", type=int, default=0, help="time this many random-order items (0 = off)")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers for --bench (train_reve.sbatch: 8)")
    a = ap.parse_args()
    add_reve_src(a.reve_eeg)
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # merge_parts, for the store checks
    layout = Path(a.layout)
    is_zarr = layout.suffix == ".zarr"
    info = check_zarr(layout, a.batch_size, a.n_batches) if is_zarr else check_layout(layout, a.batch_size, a.n_batches)
    if a.compare:
        if not is_zarr:
            raise SystemExit("--compare takes a memmap layout and needs the positional path to be a .zarr store")
        info |= compare(Path(a.compare), layout)
    if a.bench:
        info["bench_" + ("zarr" if is_zarr else "memmap")] = bench(
            zarr_dataset(layout) if is_zarr else memmap_dataset(layout), a.bench, a.workers)
        if a.compare:
            info["bench_memmap"] = bench(memmap_dataset(Path(a.compare)), a.bench, a.workers)
    for k, v in info.items():
        print(f"{k:16s} {v}")
    print(("ZARR OK" + (", IDENTICAL TO MEMMAP" if a.compare else "")) if is_zarr else "LAYOUT OK")


if __name__ == "__main__":
    main()
