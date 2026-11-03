"""Patch 8: REVE's loader reads the Zarr corpus and hands train.py exactly what it reads from the memmap layout."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from eegdash_to_reve import WINDOW, Bank, GroupWriter, Prepared, ZarrGroupWriter
from check_layout import bench, check_zarr, compare, memmap_dataset, zarr_dataset
from merge_parts import merge

RETRAIN = Path(__file__).resolve().parents[1]
NAMES = ["C3", "C4", "Cz", "Fz", "Pz", "F3", "F4", "P3", "P4", "O1"]
META = dict(subject="01", session="", run="1", task="x")


@pytest.fixture(scope="module")
def dl(reve_src, tmp_path_factory):
    """data_loading.py from the checkout with apply_patches.sh run on a copy, loaded under its own module name."""
    root = tmp_path_factory.mktemp("reve_eeg")
    for f in ("src/train.py", "src/utils/data_loading.py", "src/utils/ddp_setup.py"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(reve_src.parent / f, root / f)
    r = subprocess.run(["bash", str(RETRAIN / "patches/apply_patches.sh"), str(root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    spec = importlib.util.spec_from_file_location("data_loading_patched", root / "src/utils/data_loading.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The same sessions through both writers and both merges: (memmap layout dir, Zarr store)."""
    tmp = tmp_path_factory.mktemp("corpus")
    bank = Bank(NAMES, np.random.default_rng(0).normal(0, 0.05, (len(NAMES), 3)).astype(np.float32))
    sessions = {"dsA": [(NAMES[:8], 30000), (NAMES[:8], 4000), (NAMES, 26001), (NAMES[:8], 12345)],
                "dsB": [(NAMES, 50000), (NAMES[:9], 9000)]}
    for fmt, writer in (("mm", GroupWriter), ("zr", ZarrGroupWriter)):
        for ds, rows in sessions.items():
            w = writer(tmp / f"parts_{fmt}" / f"{ds}_000", ds, bank)
            for k, (chans, T) in enumerate(rows):
                rng = np.random.default_rng(hash((ds, k)) % 2**32)
                data = (rng.normal(0, 20, (T, len(chans))) + rng.normal(0, 5, len(chans))).astype("<f4")
                w.append(Prepared(data, sorted(chans), [], 500.0), f"{ds}/r{k}", META)
            w.finalize({})
    merge(tmp / "parts_mm", tmp / "memmap")
    (tmp / "shards.tsv").write_text("shard_id\ndsA_000\ndsB_000\n")
    merge(tmp / "parts_zr", tmp / "zarr", tmp / "shards.tsv", name="eegdash-corpus-v0")
    return tmp / "memmap", tmp / "zarr" / "eegdash-corpus-v0.zarr"


def args_for(path, subset="all", window=WINDOW):
    masking = NS(use_block=True, masking_window=200, masking_overlap=20, ratio=0.55, radius_spat_mask=0.03,
                 radius_temp_mask=3, dropout_ratio=0.1, dropout_radius=0.04)
    return NS(seed=0, data=NS(path=str(path), subset=subset, loader=NS(num_workers=0, prefetch_factor=None)),
              preprocessing=NS(window_duration=window, clip=15, masking=masking),
              trainer=NS(batch_size=4, n_gpus=1, n_nodes=1))


def test_same_samples_in_the_same_groups(dl, corpus):
    """One key per sample, grouped by channel count exactly as compute_group_segments groups the memmap windows."""
    mm, zr = memmap_dataset(corpus[0], dl, no_masking=True), zarr_dataset(corpus[1], dl, no_masking=True)
    assert {c: len(k) for c, k in mm.groups.items()} == {c: len(k) for c, k in zr.groups.items()}
    assert len(mm) == len(zr) > 0
    # same bucket and session for every sample, in the same order
    assert [k.split("_-_")[:2] for k in mm.segments] == [k.split("_-_")[:2] for k in zr.segments]


def test_every_item_identical_to_the_memmap_path(dl, corpus):
    """Normalised, clipped sample and positions: bit-identical for every key, the memmap window vs the Zarr row."""
    mm, zr = memmap_dataset(corpus[0], dl, no_masking=True), zarr_dataset(corpus[1], dl, no_masking=True)
    for k_mm, k_zr in zip(mm.segments, zr.segments):
        (x_mm, p_mm), (x_zr, p_zr) = mm[k_mm], zr[k_zr]
        assert torch.equal(x_mm, x_zr) and torch.equal(p_mm, p_zr), (k_mm, k_zr)
        assert x_zr.shape[1] == WINDOW


def test_train_py_entry_point_dispatches_on_the_path(dl, corpus):
    """get_train_val_loaders: a .zarr path goes to the Zarr loader, a memmap layout to REVE's own; both yield
    masked batches of the same shape and count."""
    out = {}
    for name, path in (("memmap", corpus[0]), ("zarr", corpus[1])):
        loader, val, n, _, n_batches, _ = dl.get_train_val_loaders(args_for(path), return_val=False)
        x, pos, b_mask, b_unmask = next(iter(loader))
        assert val is None and x.shape[2] == WINDOW and pos.shape[:2] == x.shape[:2]
        assert isinstance(loader.dataset, dl.ZarrEEGDataset) == (name == "zarr")
        out[name] = (n, n_batches)
    assert out["memmap"] == out["zarr"]


def test_excluded_datasets_leave_their_buckets_out(dl, corpus, monkeypatch, capsys):
    """REVE_EXCLUDE_DATASETS drops a dataset's buckets from the sample list and the batches; the rest is unchanged."""
    import zarr

    root = zarr.open_group(corpus[1], mode="r")
    by_ds = {}
    for name, g in root.groups():
        by_ds.setdefault(g.attrs["dataset"], set()).add(name)
    _, every = dl.zarr_group_segments(root, WINDOW)
    groups, kept = dl.zarr_group_segments(root, WINDOW, {"dsB"})
    assert {k.split("_-_")[0] for k in kept} == by_ds["dsA"]
    assert kept == [k for k in every if k.split("_-_")[0] in by_ds["dsA"]]
    assert sum(len(v) for v in groups.values()) == len(kept) < len(every)
    assert "excluded datasets ['dsB']" in capsys.readouterr().out
    monkeypatch.setenv("REVE_EXCLUDE_DATASETS", "dsB")
    _, _, n, _, _, _ = dl.get_train_val_loaders(args_for(corpus[1]), return_val=False)
    assert n == len(kept)


def test_an_excluded_id_no_bucket_carries_is_refused(dl, corpus, monkeypatch):
    """A typo (nm00289 for nm000289) would train on the whole store: refused."""
    monkeypatch.setenv("REVE_EXCLUDE_DATASETS", "dsB, dsZ")
    with pytest.raises(ValueError, match=r"\['dsZ'\]"):
        dl.get_train_val_loaders(args_for(corpus[1]), return_val=False)


def test_seeded_sampler_is_identical_on_every_rank_and_new_each_epoch(dl, corpus):
    """Patch 10: two ranks build the same batch order from the seed; set_epoch changes it; the groups are not mutated.
    Unseeded (REVE's behaviour), two samplers over the same data shuffle differently."""
    zr = zarr_dataset(corpus[1], dl, no_masking=True)
    groups_before = {c: list(k) for c, k in zr.groups.items()}
    r0 = dl.GroupedSampler(zr, batch_size=4, drop_last=True, n_gpu=2, mode="train", seed=7)
    r1 = dl.GroupedSampler(zr, batch_size=4, drop_last=True, n_gpu=2, mode="train", seed=7)
    e0 = list(iter(r0))
    assert e0 == list(iter(r1)) and len(e0) > 2
    r0.set_epoch(1); r1.set_epoch(1)
    e1 = list(iter(r0))
    assert e1 == list(iter(r1)) and e1 != e0                                     # a new permutation, shared again
    assert len(e1) == len(e0) and sorted(map(len, e1)) == sorted(map(len, e0))   # the same batch count and sizes
    r0.set_epoch(0)
    assert list(iter(r0)) == e0                                                   # epoch k depends on seed + k only
    assert {c: list(k) for c, k in zr.groups.items()} == groups_before
    u0 = dl.GroupedSampler(zr, batch_size=4, drop_last=True, n_gpu=2, mode="train")
    u1 = dl.GroupedSampler(zr, batch_size=4, drop_last=True, n_gpu=2, mode="train")
    assert list(iter(u0)) != list(iter(u1))


def test_accelerate_shards_cover_every_batch_once_when_seeded(dl, corpus):
    """The point of patch 10: with Accelerate's BatchSamplerShard on two ranks, the union of the batches is the whole
    epoch and the ranks are disjoint. Unseeded, the ranks permute independently and the epoch has gaps and repeats."""
    from accelerate.data_loader import BatchSamplerShard

    zr = zarr_dataset(corpus[1], dl, no_masking=True)

    def ranks(seed):
        out = []
        for r in range(2):
            s = dl.GroupedSampler(zr, batch_size=4, drop_last=True, n_gpu=2, mode="train", seed=seed)
            out.append([tuple(b) for b in BatchSamplerShard(s, num_processes=2, process_index=r, split_batches=False, even_batches=False)])
        return out
    a, b = ranks(seed=3)
    every = {tuple(x) for x in iter(dl.GroupedSampler(zr, batch_size=4, drop_last=True, n_gpu=2, mode="train", seed=3))}
    assert set(a).isdisjoint(b) and set(a) | set(b) == every and len(a) + len(b) == len(every)
    a, b = ranks(seed=None)
    assert set(a) | set(b) != every or not set(a).isdisjoint(b)


def test_loaders_pass_the_config_seed_to_the_sampler(dl, corpus):
    for path in corpus:
        loader, *_ = dl.get_train_val_loaders(args_for(path), return_val=False)
        assert loader.batch_sampler.seed == 0 and loader.batch_sampler.epoch == 0


def test_refuses_other_subsets_and_window_lengths(dl, corpus):
    with pytest.raises(ValueError, match="subset=all"):
        dl.get_train_val_loaders(args_for(corpus[1], subset="small"), return_val=False)
    with pytest.raises(ValueError, match="window_duration"):
        dl.get_train_val_loaders(args_for(corpus[1], window=1000), return_val=False)


def test_reopen_after_the_file_handle_cycle(dl, corpus):
    """REVE closes and reopens its files every 500k items; the Zarr dataset must survive the same cycle."""
    zr = zarr_dataset(corpus[1], dl, no_masking=True)
    key = zr.segments[3]
    before = zr[key][0]
    zr.counter = 500000
    assert torch.equal(zr[key][0], before) and zr.counter == 1


def test_check_layout_on_a_store(dl, corpus):
    """check_layout.py's Zarr mode: store checks pass, a batch comes out, and the comparison with memmap holds."""
    info = check_zarr(corpus[1], batch_size=4, dl=dl)
    assert info["samples"] == 62 and info["batch_shapes"]
    assert compare(corpus[0], corpus[1], dl=dl)["items_identical"] == 62
    b = bench(zarr_dataset(corpus[1], dl), n=20, workers=0)
    assert b["items"] == 20 and b["items_per_s"] > 0


def test_compare_catches_one_changed_value(dl, corpus, tmp_path):
    """A single sample value off in the store must fail the comparison."""
    import zarr

    store = tmp_path / "copy.zarr"
    shutil.copytree(corpus[1], store)
    arr = zarr.open_group(store, mode="r+")["2"]["data"]
    sample = arr[5]
    sample[100, 3] += 1.0
    arr[5] = sample
    with pytest.raises(AssertionError, match="sample differs"):
        compare(corpus[0], store, dl=dl)


# ---- the direct shard reader (one pread per sample) and batched reads ----------------------------------------------

def _store_with_zeros(tmp_path, bank, compress=True):
    """One bucket: a normal session, then 300 samples of zeros (a whole shard of zeros is never written, and zero
    chunks inside a written shard are left out of it), then another normal session."""
    w = ZarrGroupWriter(tmp_path / "z", "dsZ", bank, compress=compress)
    rng = np.random.default_rng(5)
    sessions = [rng.normal(0, 20, (WINDOW * 100 + 7, 8)), np.zeros((WINDOW * 300 + 7, 8)), rng.normal(0, 20, (WINDOW * 60 + 7, 8))]
    for k, d in enumerate(sessions):
        w.append(Prepared(d.astype("<f4"), sorted(NAMES[:8]), [], 200.0), f"s{k}", dict(subject="1", session="", run="1", task="x"))
    w.finalize({})
    return tmp_path / "z/samples.zarr"


@pytest.fixture
def bank8():
    from eegdash_to_reve import Bank
    return Bank(NAMES, np.random.default_rng(0).normal(0, 0.05, (len(NAMES), 3)).astype(np.float32))


def test_direct_reader_equals_zarr_for_every_sample(dl, corpus):
    import zarr
    root = zarr.open_group(corpus[1], mode="r")
    for name, g in root.groups():
        r = dl._ShardReader(corpus[1] / name / "data")
        assert r.ok
        for i in range(g["data"].shape[0]):
            assert np.array_equal(r.read(i), g["data"][i]), (name, i)


@pytest.mark.parametrize("compress", [True, False])
def test_zeros_read_back_as_zeros_missing_shard_and_empty_chunks(dl, tmp_path, bank8, compress):
    import os, zarr
    store = _store_with_zeros(tmp_path, bank8, compress)
    arr = zarr.open_group(store, mode="r")["0"]["data"]
    shards = {int(x) for x in os.listdir(store / "0/data/c")}
    assert len(shards) < -(-arr.shape[0] // 128)                     # the fixture really has a missing shard
    r = dl._ShardReader(store / "0/data")
    assert r.ok and r.compressed == compress
    for i in range(arr.shape[0]):
        assert np.array_equal(r.read(i), arr[i]), i


def test_batch_via_getitems_equals_item_by_item(dl, corpus):
    ds = zarr_dataset(corpus[1], dl, no_masking=True)
    keys = ds.segments[::3]
    batched = ds.__getitems__(keys)
    single = [ds[k] for k in keys]
    assert len(batched) == len(single)
    for (xb, pb), (xs, ps) in zip(batched, single):
        assert torch.equal(xb, xs) and torch.equal(pb, ps)
    assert all(not f.prefetched for f in ds.files.values())           # nothing left behind for the next batch


def test_dataloader_batches_use_getitems_and_match(dl, corpus):
    from torch.utils.data import DataLoader
    ds = zarr_dataset(corpus[1], dl, no_masking=True)
    groups = [v for v in ds.groups.values()]
    batches = [groups[0][:5], groups[-1][:4]]
    calls = []
    orig = ds.__getitems__
    ds.__getitems__ = lambda ks: calls.append(len(ks)) or orig(ks)
    out = list(DataLoader(ds, batch_sampler=batches, num_workers=0))
    assert calls == [5, 4]
    for (x, p), keys in zip(out, batches):
        ref = torch.stack([ds[k][0] for k in keys])
        assert torch.equal(x, ref)


def test_old_descriptors_are_closed_between_batches(dl, tmp_path, bank8):
    import zarr
    store = _store_with_zeros(tmp_path, bank8)
    ds = zarr_dataset(store, dl, no_masking=True)
    reader = ds.files[0].reader
    reader.MAX_OPEN, reader.MAX_INDEX = 1, 1
    keys = ds.segments[::40]                                         # rows spread over several shards
    out = ds.__getitems__(keys)
    assert len(reader.fds) <= 1 and len(reader.index) <= 1
    arr = zarr.open_group(store, mode="r")["0"]["data"]
    ref = zarr_dataset(store, dl, no_masking=True)
    assert all(torch.equal(a[0], ref[k][0]) for a, k in zip(out, keys))


def test_a_forked_worker_opens_its_own_files(dl, corpus, monkeypatch):
    """In a new process the inherited descriptors are closed (the child's copies) and the worker opens its own."""
    import os
    r = dl._ShardReader(corpus[1] / "0" / "data")
    r.read(0)
    parent_fds, parent_table = set(r.fds.values()), r.fds
    assert parent_fds
    closed, opened = [], []
    real_close, real_open = os.close, os.open
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd) or real_close(fd))
    monkeypatch.setattr(os, "open", lambda *a, **k: opened.append(a[0]) or real_open(*a, **k))
    monkeypatch.setattr(os, "getpid", lambda: r.pid + 1)
    x = r.read(1)
    assert parent_fds <= set(closed)                                 # the inherited descriptors are released
    assert opened and r.fds is not parent_table                      # a fresh table, filled by this process's own open
    assert x.shape == tuple(r.shape[1:])


def test_one_descriptor_budget_for_all_buckets_in_a_process(dl, corpus, monkeypatch):
    """Job 54458868 (2026-09-27): a cap per bucket let 968 readers x 144 processes fill a node's file table. The cap
    is now shared: however many buckets a batch touches, the process holds at most MAX_OPEN descriptors after it."""
    ds = zarr_dataset(corpus[1], dl, no_masking=True)
    readers = {id(f.reader): f.reader for f in ds.files.values() if f.reader is not None}
    assert len(readers) >= 2                                         # the fixture has several buckets
    monkeypatch.setattr(dl._ShardReader, "MAX_OPEN", 2)
    keys = [k for g in ds.groups.values() for k in g[::7]]           # every bucket, several shards each
    out = ds.__getitems__(keys)
    r = next(iter(readers.values()))
    assert len(r.fds) <= 2 and all(x.fds is r.fds for x in readers.values())
    ref = zarr_dataset(corpus[1], dl, no_masking=True)
    assert all(torch.equal(a[0], ref[k][0]) for a, k in zip(out, keys))


def test_unknown_layout_falls_back_to_zarr(dl, tmp_path):
    import zarr
    a = zarr.create_array(tmp_path / "plain.zarr", shape=(4, WINDOW, 3), chunks=(1, WINDOW, 3), dtype="float32")
    assert not dl._ShardReader(tmp_path / "plain.zarr").ok
    assert dl._ZarrSamples(a, tmp_path / "plain.zarr").reader is None


def test_direct_reads_can_be_switched_off(dl, corpus, monkeypatch):
    monkeypatch.setenv("REVE_ZARR_DIRECT", "0")
    ds = zarr_dataset(corpus[1], dl, no_masking=True)
    assert all(f.reader is None for f in ds.files.values())
    assert torch.equal(ds[ds.segments[0]][0], zarr_dataset(corpus[1], dl, no_masking=True)[ds.segments[0]][0])


def test_threads_of_a_new_worker_do_not_close_each_others_files(dl, corpus, monkeypatch):
    """Job 54491935 (2026-09-30): at an epoch start the DataLoader forks fresh workers, and a worker's first batch is
    read by several threads at once; each saw a new process and reset the shared table, one closing a descriptor
    another had just opened (errno 9 on pread, epoch 7 of 17). Reads in a new process must survive that start."""
    import os
    import threading
    import time
    r = dl._ShardReader(corpus[1] / "0" / "data")
    rows = list(range(0, r.shape[0], max(1, r.per_shard // 2)))[:8] or [0]
    real_close, fake = os.close, [r.pid]
    monkeypatch.setattr(os, "close", lambda fd: time.sleep(0.002) or real_close(fd))   # widens the window
    monkeypatch.setattr(os, "getpid", lambda: fake[0])
    errors = []

    def work(barrier):
        barrier.wait()
        try:
            for row in rows:
                r.read(row)
        except OSError as e:
            errors.append(e)

    for round_ in range(30):
        fake[0] += 1                                                 # a new worker process
        barrier = threading.Barrier(8)
        threads = [threading.Thread(target=work, args=(barrier,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"round {round_}: {errors[0]}"


@pytest.mark.parametrize("compress", [True, False])
def test_short_reads_are_completed(dl, tmp_path, bank8, compress, monkeypatch):
    """POSIX lets pread return fewer bytes than asked; the reader keeps reading until the chunk is whole."""
    import os, zarr
    store = _store_with_zeros(tmp_path, bank8, compress)
    arr = zarr.open_group(store, mode="r")["0"]["data"]
    real = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, n, off: real(fd, max(1, n // 3), off))
    r = dl._ShardReader(store / "0/data")
    for i in range(arr.shape[0]):
        assert np.array_equal(r.read(i), arr[i]), i


def test_a_failed_decompression_is_retried_once_then_reported(dl, tmp_path, bank8, monkeypatch):
    """Job 54576735 (2026-10-02): one blosc failure (-1) on a chunk read fine for eight epochs ended the run. A single
    bad read now costs one retry; a chunk that stays unreadable raises with its file and row."""
    import zarr
    from numcodecs import blosc
    store = _store_with_zeros(tmp_path, bank8, True)
    arr = zarr.open_group(store, mode="r")["0"]["data"]
    row = next(i for i in range(arr.shape[0]) if np.any(arr[i]))
    expected = arr[row]                                              # read before decompress is replaced
    real, calls = blosc.decompress, []

    def flaky(buf, *a):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("error during blosc decompression: -1")
        return real(buf, *a)

    monkeypatch.setattr(blosc, "decompress", flaky)
    r = dl._ShardReader(store / "0/data")
    assert np.array_equal(r.read(row), expected) and len(calls) == 2
    monkeypatch.setattr(blosc, "decompress", lambda buf, *a: (_ for _ in ()).throw(RuntimeError("blosc -1")))
    with pytest.raises(RuntimeError, match=rf"0/data/c/\d+/0/0 row {row}: unreadable after a retry"):
        r.read(row)
