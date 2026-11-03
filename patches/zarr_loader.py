

# ---- PATCH 8 (appended by replicating-reve apply_patches.sh): train from the published Zarr corpus -------------------
# A data.path ending in .zarr selects this path; a memmap layout still goes through the code above, unchanged.
# Store (eegdash_to_reve.py --format zarr, merged by merge_parts.py): one group per bucket (all sessions sharing one
# channel set) holding data (N, 2000, C) = the training samples REVE cuts from that bucket, session (N,), stats (S, 2, C)
# and positions (C, 3). Keys keep REVE's "bucket_-_session_-_offset" form with offset = the sample's row, so
# EEGDataset.__getitem__ (normalisation, clipping, masking) runs unchanged; only the file handling differs.


class _ShardReader:
    """Reads one sample of a sharded Zarr array with a single pread: shard files stay open and their indexes are kept in
    memory, per process. Zarr's own read re-opens the shard file and re-reads its index for every sample, three
    round trips to Lustre; on reve-public-nc (72 % of samples in 240-sample I-CARE batches) that starved the GPUs at
    3.6 s per step (debug run 54454169, 2026-09-25). Missing shards and chunks read as zeros, the fill value, as in
    Zarr (all-zero I-CARE recordings are stored that way). Returns None for arrays it does not understand, and the
    caller then uses Zarr."""

    # One budget per process, shared by every bucket's reader (2026-09-27, job 54458868): with a cap per reader, 968
    # buckets x 144 worker processes per node kept a descriptor for nearly every shard touched, and after 20 minutes a
    # node's kernel file table was full ("Too many open files in system", errno 23) and Slurm failed the node.
    # Sampling is random over ~127,000 shard files, so a larger cache would barely raise the hit rate anyway.
    MAX_OPEN = int(__import__("os").environ.get("REVE_ZARR_MAX_OPEN", "1024"))      # descriptors per process
    MAX_INDEX = int(__import__("os").environ.get("REVE_ZARR_MAX_INDEX", "16384"))   # shard indexes per process (~2 KB each)
    _fds, _index, _pid = {}, {}, None                                                # process-wide, insertion-ordered
    # The first batch of a freshly forked worker is read by several threads at once, and each sees the new process:
    # unlocked, one thread's reset closed a descriptor another had just opened (errno 9, job 54491935, epoch 7).
    _reset_lock = __import__("threading").Lock()

    def __init__(self, array_dir):
        import json

        self.array_dir = str(array_dir)
        meta = json.load(open(f"{self.array_dir}/zarr.json"))
        self.ok = False
        try:
            (codec,) = meta["codecs"]
            cfg = codec["configuration"]
            inner = [c["name"] for c in cfg["codecs"]]
            index = [c["name"] for c in cfg["index_codecs"]]
            sep = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
            self.ok = (codec["name"] == "sharding_indexed" and cfg.get("index_location", "end") == "end"
                       and inner in (["bytes"], ["bytes", "blosc"]) and index == ["bytes", "crc32c"]
                       and cfg["codecs"][0].get("configuration", {}).get("endian", "little") == "little"
                       and meta["data_type"] == "float32" and meta.get("fill_value", 0) in (0, 0.0) and sep == "/"
                       and cfg["chunk_shape"][0] == 1)
        except (KeyError, TypeError, ValueError):
            self.ok = False
        if self.ok:
            self.shape = meta["shape"]
            self.per_shard = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
            self.compressed = inner[-1] == "blosc"
            self.index_bytes = self.per_shard * 16 + 4
        self._check_process()

    @property
    def fds(self):
        return type(self)._fds

    @property
    def index(self):
        return type(self)._index

    @property
    def pid(self):
        return type(self)._pid

    @classmethod
    def _check_process(cls):
        """In a new process (a forked DataLoader worker), close the descriptors inherited from the parent and start empty."""
        import os
        import resource

        if cls._pid == os.getpid():
            return
        with cls._reset_lock:
            if cls._pid == os.getpid():             # another thread of this process reset it first
                return
            for fd in list(cls._fds.values()):
                try:
                    os.close(fd)                    # the child's copy only; the parent's stays open
                except OSError:
                    pass
            cls._fds, cls._index = {}, {}
            cls._pid = os.getpid()                  # last: a thread that sees it can use the new table
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)   # nodes often default to 1024 open files a process
        want = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
                soft = want
            except (ValueError, OSError):
                pass
        if soft != resource.RLIM_INFINITY:
            cls.MAX_OPEN = min(cls.MAX_OPEN, soft // 2)

    def read(self, row):
        """Sample `row` as a (window, C) float32 array; safe to call from several threads between two trims."""
        import os

        cls = type(self)
        if os.getpid() != cls._pid:
            cls._check_process()
        shard, j = divmod(row, self.per_shard)
        path = f"{self.array_dir}/c/{shard}/0/0"
        fd = cls._fds.get(path)
        if fd is None:
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError:               # a shard of zeros is never written
                return np.zeros(self.shape[1:], dtype=np.float32)
            kept = cls._fds.setdefault(path, fd)
            if kept != fd:                          # another thread opened it meanwhile: keep theirs, close ours
                os.close(fd)
                fd = kept
        idx = cls._index.get(path)
        if idx is None:
            idx = cls._index.setdefault(path, self._read_index(fd))
        try:
            return self._read_chunk(fd, idx, j)
        except (RuntimeError, ValueError, OSError) as first:
            # Job 54576735 (2026-10-02): one read of a chunk every earlier epoch had read fine failed to decompress
            # (blosc -1) and ended the run. Retry once from a fresh descriptor and a fresh index before giving up.
            fresh = os.open(path, os.O_RDONLY)
            try:
                return self._read_chunk(fresh, self._read_index(fresh), j)
            except (RuntimeError, ValueError, OSError) as second:
                raise RuntimeError(f"{path} row {row}: unreadable after a retry ({first}; then {second})") from second
            finally:
                os.close(fresh)

    @staticmethod
    def _pread_full(fd, nbytes, offset):
        """os.pread may return fewer bytes than asked (allowed by POSIX; seen as truncated chunks on Lustre)."""
        import os

        parts, got = [], 0
        while got < nbytes:
            part = os.pread(fd, nbytes - got, offset + got)
            if not part:
                break
            parts.append(part)
            got += len(part)
        return b"".join(parts)

    def _read_index(self, fd):
        """The shard's chunk index, stored at its end: (offset, nbytes) per chunk plus a 4-byte checksum."""
        import os

        end = os.fstat(fd).st_size
        raw = self._pread_full(fd, self.index_bytes, end - self.index_bytes)
        if len(raw) != self.index_bytes:
            raise OSError(f"index: {len(raw)} of {self.index_bytes} bytes")
        return np.frombuffer(raw[:-4], dtype="<u8").reshape(self.per_shard, 2)

    def _read_chunk(self, fd, idx, j):
        off, nbytes = int(idx[j, 0]), int(idx[j, 1])
        if off == 2**64 - 1:                        # an all-zero chunk inside a written shard
            return np.zeros(self.shape[1:], dtype=np.float32)
        buf = self._pread_full(fd, nbytes, off)
        if len(buf) != nbytes:
            raise OSError(f"chunk: {len(buf)} of {nbytes} bytes at offset {off}")
        if self.compressed:
            from numcodecs import blosc

            buf = blosc.decompress(buf)
        return np.frombuffer(buf, dtype="<f4").reshape(self.shape[1:]).copy()

    def trim(self):
        """Close the oldest descriptors beyond the process budget and drop the oldest indexes beyond theirs; called
        between batches, never while threads read. `self.MAX_OPEN` may be set on one reader (tests)."""
        import os

        cls = type(self)
        while len(cls._fds) > self.MAX_OPEN:
            os.close(cls._fds.pop(next(iter(cls._fds))))
        while len(cls._index) > self.MAX_INDEX:
            cls._index.pop(next(iter(cls._index)))


class _ZarrSamples:
    """Stands in for a bucket's memmap: [i : i + window] returns sample row i, which is that window. Rows fetched ahead
    by ZarrEEGDataset.__getitems__ are taken from `prefetched`."""

    def __init__(self, array, array_dir=None):
        self.array = array
        self.prefetched = {}
        self.reader = _ShardReader(array_dir) if array_dir is not None else None
        if self.reader is not None and not self.reader.ok:
            self.reader = None

    def fetch(self, row):
        return self.reader.read(row) if self.reader is not None else self.array[row]

    def __getitem__(self, s):
        row = s.start
        hit = self.prefetched.pop(row, None)
        return hit if hit is not None else self.fetch(row)


def zarr_group_segments(root, window_duration, exclude=()):
    """The keys compute_group_segments builds from the CSVs, read from the store: one per sample, grouped by channel
    count. Buckets whose `dataset` attribute is in *exclude* are left out: the camera-ready's v0 runs on Delta read
    eegdash-corpus-v1 without I-CARE (REVE_EXCLUDE_DATASETS=nm000289, 2026-10-29: the v0 store's 632 buckets). An id
    that no bucket carries is refused, since a misspelled one would silently train on everything."""
    exclude, seen, dropped = set(exclude), set(), 0
    dict_groups = defaultdict(list)
    for name, g in sorted(root.groups(), key=lambda kv: int(kv[0])):
        n, w, c = g["data"].shape
        if w != window_duration:
            raise ValueError(f"bucket {name}: samples of {w} timesteps, preprocessing.window_duration={window_duration}")
        if exclude and g.attrs.get("dataset") in exclude:
            seen.add(g.attrs["dataset"])
            dropped += n
            continue
        dict_groups[c] += [f"{name}_-_{s}_-_{i}" for i, s in enumerate(g["session"][:].tolist())]
    if exclude - seen:
        raise ValueError(f"REVE_EXCLUDE_DATASETS names {sorted(exclude - seen)}, which no bucket of the store carries")
    if exclude and __import__("os").environ.get("RANK", "0") == "0":
        print(f"excluded datasets {sorted(exclude)}: {dropped} samples left out", flush=True)
    segments = [k for keys in dict_groups.values() for k in keys]  # flattened before sorting, as REVE does
    return dict(sorted(dict_groups.items())), segments


def excluded_datasets():
    """REVE_EXCLUDE_DATASETS as a set: ids separated by commas or spaces; empty when unset."""
    return set(__import__("os").environ.get("REVE_EXCLUDE_DATASETS", "").replace(",", " ").split())


class ZarrEEGDataset(EEGDataset):
    """EEGDataset over one Zarr store. Stats and positions are small and held in memory; a sample is one chunk read,
    by _ShardReader (one pread) unless REVE_ZARR_DIRECT=0, and a batch's samples are read in parallel (__getitems__).
    Each DataLoader worker opens its own files after fork."""

    def __init__(self, store_path, segments, groups, **kwargs):
        self.store_path = store_path
        super().__init__(segments, groups, None, None, store_path, **kwargs)

    def init_files_pos(self):
        import os
        import zarr

        root = zarr.open_group(self.store_path, mode="r")
        direct = os.environ.get("REVE_ZARR_DIRECT", "1") == "1"
        self.files, self.positions_cache, self.stats = {}, {}, {}
        for name, g in root.groups():
            b = int(name)
            self.files[b] = _ZarrSamples(g["data"], f"{self.store_path}/{name}/data" if direct else None)
            self.positions_cache[b] = g["positions"][:]
            self.stats[b] = g["stats"][:]
        self._pool, self._pool_pid = None, None

    def del_files(self):
        del self.files, self.positions_cache, self.stats
        gc.collect()

    def __getitems__(self, keys):
        """A whole batch at once (PyTorch's DataLoader calls this when the dataset has it): the batch's samples are
        read in parallel, then REVE's own __getitem__ normalises and masks each one, so the batch is the same as item by
        item. REVE_ZARR_READ_THREADS (default 8) sets the threads per worker; os.pread releases the GIL."""
        import os
        from concurrent.futures import ThreadPoolExecutor

        if self.counter + len(keys) >= 500000:          # REVE's file reopen falls inside this batch: do it item by item
            return [self.__getitem__(k) for k in keys]
        if self._pool is None or self._pool_pid != os.getpid():
            self._pool = ThreadPoolExecutor(int(os.environ.get("REVE_ZARR_READ_THREADS", "8")))
            self._pool_pid = os.getpid()
        wanted = []
        for k in keys:
            b, _, row = k.split("_-_")[:3]
            wanted.append((self.files[int(b)], int(row)))
        rows = list(self._pool.map(lambda fr: fr[0].fetch(fr[1]), wanted))
        for (f, row), x in zip(wanted, rows):
            f.prefetched[row] = x
        try:
            return [self.__getitem__(k) for k in keys]
        finally:
            for f, _ in wanted:
                f.prefetched.clear()
                if f.reader is not None:
                    f.reader.trim()


def _zarr_train_loader(args):
    """get_train_val_loaders for a Zarr corpus: the same dataset settings, sampler and DataLoader as _build_loader
    with train=True. No validation split (REVE's val subsets index its own layout)."""
    import zarr

    if args.data.subset != "all":
        raise ValueError(f"a Zarr corpus trains with data.subset=all, not {args.data.subset!r}")
    root = zarr.open_group(args.data.path, mode="r")
    dict_groups, segments = zarr_group_segments(root, args.preprocessing.window_duration, excluded_datasets())
    masking_cfg = args.preprocessing.masking
    dataset = ZarrEEGDataset(
        args.data.path,
        segments,
        dict_groups,
        window_duration=args.preprocessing.window_duration,
        clip=args.preprocessing.clip,
        block_masking=masking_cfg.use_block,
        masking_window=masking_cfg.masking_window,
        masking_overlap=masking_cfg.masking_overlap,
        masking_ratio=masking_cfg.ratio,
        radius_spat_mask=masking_cfg.radius_spat_mask,
        radius_temp_mask=masking_cfg.radius_temp_mask,
        dropout_ratio=masking_cfg.dropout_ratio,
        dropout_radius=masking_cfg.dropout_radius,
        no_masking=False,
        manual_seed=False,
    )
    sampler = GroupedSampler(
        dataset,
        batch_size=args.trainer.batch_size,
        drop_last=True,
        n_gpu=args.trainer.n_gpus * args.trainer.n_nodes,
        mode="train",
        seed=args.seed,  # PATCH 10: the same shuffle on every rank, new each epoch
    )
    sampler.__iter__()
    len_sampler = len(sampler)
    loader = DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=sampler,
        num_workers=args.data.loader.num_workers,
        persistent_workers=False,
        prefetch_factor=args.data.loader.prefetch_factor,
    )
    return loader, None, len(dataset), None, len_sampler, None
