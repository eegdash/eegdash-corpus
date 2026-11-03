"""Convert EEGDash recordings into the on-disk layout REVE's own train.py reads.

One invocation = one shard: one dataset, optionally a subset of its subjects. Shards are
independent parts (merged later by merge_parts.py), so a Slurm array can run many at once.

  python eegdash_to_reve.py --dataset ds004840 --cache-dir $EEGDASH_CACHE \
      --out $CONVROOT/parts/ds004840_000 [--subjects 01 02] [--online] [--n-jobs 16] [--format zarr] \
      [--no-dataset-positions]

--format zarr writes the published form instead: <out>/samples.zarr, one group per channel set holding the training
samples themselves, data (N, 2000, C), cut exactly as REVE's loader cuts them (see ZarrGroupWriter, sample_offsets).
Preprocessing is identical in both formats; only the writer differs. The memmap layout is described below.

Layout written under --out (contract: reve_eeg/src/utils/data_loading.py, EEGDataset.init_files_pos):

  recordings/recording_-_eeg_-_{g}.npy   RAW float32 little-endian, shape (T, C), time first, NO .npy header
                                          (the loader opens it with np.memmap(shape=(T, C)))
  positions/recording_-_positions_-_{g}.npy  real .npy (np.load), (C, 3) metres, from brain-bzh/reve-positions
                                          (+ MNE's biosemi256 template; + positions aligned from the dataset, see below)
  stats/recording_-_stats_-_{g}.npy       RAW float32, shape (n_sessions, 2, C): per-session [mean, std]
  csv_recordings/df_big.csv               big_recording_index, duration, n_chans, chans_names, config
  csv_recordings/df_corrected.csv         file, big_recording_index, index, duration, n_chans, mult, dataset,
                                          class, flag_remove, flag_reduce, n_chans_to_remove
  csv_recordings/df_stats_tmp.csv         big_recording_index, n_sessions, n_chans
  manifest_part.csv, summary.json, _DONE  ours: provenance per session, shard summary, completion marker

A "big recording" g = every session of this shard whose resolved channel set is identical, appended
along time. A "session" = one recording (subject x session x run). Channel names are sorted inside a
big recording so that two files listing the same electrodes in a different order share one group.

Recipe (REVE paper section 3.1.1): EEG channels only, channels with a known position (the bank, rules below:
BioSemi-256 caps, a zero typed for O, coordinates from the dataset's electrodes.tsv fitted to the bank),
band-pass 0.5-99.5 Hz (upper edge below Nyquist), resample to 200 Hz, float32, drop recordings of
10 s or less. Normalisation is NOT baked in: train.py z-scores each window with its session's
mean/std at load time and clips at 15 SD. Signals are stored in microvolts.

Rules that bite (the first and the last read from the loader source):
  * sessions with duration <= 2000 samples are dropped by compute_group_segments BEFORE stats are
    indexed, so such a session must never be written (it would shift every later stats row);
  * the position bank is case-sensitive; its 18 case-fold duplicates share coordinates, so a
    case-insensitive match is safe. Bare A1..D32 are 10-20 names (A1/A2 = earlobes) but a BioSemi-128
    cap uses the same labels for its ring positions: >= 32 such names => map to biosemi128_*;
  * a BioSemi-256 cap labels 8 rings A1..H32: >= 32 labels of rings A-D and >= 64 of rings E-H => every ring label
    maps to biosemi256_* (MNE's template, appended to the bank), never to the 128 cap's ring or GSN's E1..E32;
  * a zero typed for the letter O (P03, P07, P0z: SparrKULee) is tried as PO3, PO7, POz after the exact name fails;
  * an EEG channel the bank does not know is kept when the recording's *_electrodes.tsv (BIDS inheritance) places it:
    the file is fitted to the bank on the channels both know (rotation + uniform scale + translation, >= 8 shared,
    RMS residual <= 10 mm, so units and frames need not be trusted), the channel keeps its dataset name and gets the
    fitted position; otherwise it is dropped (--no-dataset-positions turns this off);
  * a big recording with a single session needs patch 1 (patches/) in the loader.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SFREQ = 200
WINDOW = 2000  # samples; preprocessing.window_duration in REVE's config
L_FREQ, H_FREQ = 0.5, 99.5
MIN_RESOLVABLE = 3
HELD_OUT = {"nm000139", "ds004362", "nm000112", "nm000114", "nm000109", "nm000110", "nm000113"}  # OpenEEGBench sources in EEGDash: BCIC-IV-2a, PhysioNet-MI, FACED, Mumtaz, Zyma, CHB-MIT, BCIC-2020-3
BIOSEMI_RE = re.compile(r"[A-D]([1-9]|[12]\d|3[0-2])")  # A1..D32
RING_E_RE = re.compile(r"E([1-9]|[12]\d|3[0-2])")       # E1..E32: the fifth ring of a BioSemi-160 or -256 cap (and GSN's first 32 names)
RING_FGH_RE = re.compile(r"[F-H]([1-9]|[12]\d|3[0-2])")  # F1..H32: the three outer rings of a BioSemi-256 cap
RING_EH_RE = re.compile(r"[E-H]([1-9]|[12]\d|3[0-2])")
# A 256 cap needs many outer labels, not just some: a 10-10 cap has F1..F10 beside C1..C6, a BioSemi-160 cap one ring E.
BIOSEMI_MIN_INNER = 32   # >= 32 labels from rings A-D: a BioSemi cap (a 10-10 cap has at most A1, A2 and C1..C6)
BIOSEMI256_MIN_FGH = 4   # then any real presence of rings F-H makes it a 256 cap (bad channels removed per recording: ds003004)
BIOSEMI160_MIN_E = 8     # else a ring E makes it a 160 cap; else the 128 cap
ZERO_FOR_O_RE = re.compile(r"^([A-Za-z]+)0(\d+|[zZ])$")  # P03, P0z: a zero typed for O (SparrKULee, 42 of 659 files)
ALIGN_MIN_SHARED = 8      # electrodes a dataset's electrodes.tsv must share with the bank before it is fitted
ALIGN_MAX_RMS_M = 0.020   # RMS residual of that fit over the shared electrodes, metres; above it the file is not used. 20 mm, not
# 10: a file that projects the electrodes onto a unit sphere (EEGLAB's export, ds002680) fits the template head to 16 mm, about
# the spread of real 10-20 positions between people; a mirrored or random file sits far above (2026-10-18)
FILTER_ONE_CORE_BYTES = 1_000_000_000   # a recording larger than this in memory is filtered on one core (filter_jobs)


def filter_jobs(n_bytes: int, n_jobs: int) -> int:
    """MNE's filter and resample hand the channels to joblib workers, each with its own copy of the data: a 3 GB
    recording (ds005697: 64 channels, 1.7 h at 1,000 Hz, float64) on ten cores is 30 GB and was killed in a 19 GB job
    (rc=137, 2026-10-18). Above FILTER_ONE_CORE_BYTES one core does the work; it is a few minutes per recording."""
    return 1 if n_bytes > FILTER_ONE_CORE_BYTES else max(1, n_jobs)
# Channel names that are not scalp EEG, never placed from a dataset's coordinates. The same pattern as
# corpus_tables.NON_EEG_RE (the dropped-channels audit); a test keeps the two equal.
NON_EEG_RE = re.compile(r"(EXG|EOG|HEO|VEO|E[CK]G|EMG|GSR|Erg\d*|Resp|Plet|Pulse|Optic|Temp|Status|Trig|Audio|Mastoid|Eye|SpO2|Dia-|\bHR\b"
                        r"|^(Empty|M[12]|REF\w*|GND|Ground|MK|\d+|[A-HJ-Z])$)", re.IGNORECASE)   # a bare I is ds002680's inion electrode
# A modality prefix some EDF writers put on every label ("EEG Fp1", "EEG-Cz", "EEG_O1"): Siena (nm000385) labels
# all its electrodes so, and without stripping it the bank matches none and every recording is skipped (2026-10-13).
EEG_PREFIX_RE = re.compile(r"^EEG[\s_-]+", re.IGNORECASE)


def strip_eeg_prefix(name: str) -> str:
    """`EEG Fp1` -> `Fp1`; a name without the prefix, or one that would become empty, is returned unchanged."""
    bare = EEG_PREFIX_RE.sub("", name.strip())
    return bare or name
CORR_FIELDS = ["file", "big_recording_index", "index", "duration", "n_chans", "mult", "dataset", "class",
               "flag_remove", "flag_reduce", "n_chans_to_remove"]


# ----------------------------------------------------------------------------- position bank
def with_biosemi_templates(names: list[str], xyz: np.ndarray) -> tuple[list[str], np.ndarray]:
    """The bank plus MNE's biosemi160 and biosemi256 templates as biosemi160_A1..E32 and biosemi256_A1..H32 (metres,
    the bank's head frame: its biosemi128_* equal MNE's biosemi128 to 0.0 mm). REVE's bank has only the 128 cap;
    without these, rings A-D of a bigger cap took the 128 cap's positions, ring E GSN's and rings F-H were dropped.
    Names already present are not added twice."""
    import mne

    have = set(names)
    extra = []
    for cap in ("biosemi160", "biosemi256"):
        ch_pos = mne.channels.make_standard_montage(cap).get_positions()["ch_pos"]
        extra += [(f"{cap}_{k}", v) for k, v in ch_pos.items() if f"{cap}_{k}" not in have]
    if not extra:
        return list(names), np.asarray(xyz, dtype=np.float32)
    return (list(names) + [n for n, _ in extra],
            np.concatenate([np.asarray(xyz, dtype=np.float32), np.array([v for _, v in extra], dtype=np.float32)]))


with_biosemi256 = with_biosemi_templates   # the first name of the helper (2026-10-18)


def zero_for_o(name: str) -> str | None:
    """`P03` -> `PO3`, `P0z` -> `POz`; None for a name without a zero after its letters."""
    m = ZERO_FOR_O_RE.match(name)
    return f"{m[1]}O{m[2]}" if m else None


class Bank:
    """Electrode name -> 3-D position (metres), with REVE's naming quirks handled."""

    def __init__(self, names: list[str], xyz: np.ndarray):
        assert len(names) == len(xyz) and xyz.shape[1] == 3
        self.names = list(names)
        self.xyz = np.asarray(xyz, dtype=np.float32)
        self.exact = {n: i for i, n in enumerate(self.names)}
        self.fold: dict[str, int] = {}
        for i, n in enumerate(self.names):
            self.fold.setdefault(n.casefold(), i)

    @classmethod
    def from_braindecode(cls) -> "Bank":
        """brain-bzh/reve-positions via braindecode's cached copy (REVE_POSITIONS_PATH or ~/mne_data)."""
        from braindecode.models.reve import RevePositionBank

        bank = RevePositionBank()
        return cls(*with_biosemi_templates(list(bank.position_names), bank.embedding.detach().cpu().numpy()))

    @classmethod
    def from_json(cls, path: str | Path) -> "Bank":
        """A {name: [x, y, z]} bank, plus biosemi256_* so that it resolves as the braindecode bank does."""
        d = json.loads(Path(path).read_text())
        return cls(*with_biosemi_templates(list(d), np.array(list(d.values()), dtype=np.float32)))

    def _lookup(self, name: str) -> int | None:
        i = self.exact.get(name)
        return i if i is not None else self.fold.get(name.casefold())

    def resolve(self, ch_names: list[str]) -> tuple[list[int], list[str], list[str]]:
        """Return (indices kept, canonical bank names in the same order, names dropped)."""
        inner = sum(bool(BIOSEMI_RE.fullmatch(n)) for n in ch_names)
        cap = None   # which BioSemi cap the ring labels belong to, decided once per recording from the rings present
        if inner >= BIOSEMI_MIN_INNER:
            if sum(bool(RING_FGH_RE.fullmatch(n)) for n in ch_names) >= BIOSEMI256_MIN_FGH:
                cap = "biosemi256"
            elif sum(bool(RING_E_RE.fullmatch(n)) for n in ch_names) >= BIOSEMI160_MIN_E:
                cap = "biosemi160"
        cap256 = cap == "biosemi256"
        biosemi = inner >= BIOSEMI_MIN_INNER and cap is None
        kept, canonical, dropped, seen = [], [], [], set()
        for i, n in enumerate(ch_names):
            bare = strip_eeg_prefix(n)
            if cap and (BIOSEMI_RE.fullmatch(n) or RING_EH_RE.fullmatch(n)):
                cands = [f"{cap}_{n}"]   # never the 128 cap's ring, the 10-20 name or GSN's E1..E32
            else:
                cands = ([f"biosemi128_{n}"] if biosemi and BIOSEMI_RE.fullmatch(n) else []) + [n] + ([bare] if bare != n else [])
                cands += [o for o in map(zero_for_o, cands) if o]   # fallback only: after every exact candidate
            hit = next((self._lookup(c) for c in cands if self._lookup(c) is not None), None)
            # unknown name, or the same electrode twice (Cz and CZ are both bank entries with one position)
            if hit is None or self.names[hit].casefold() in seen:
                dropped.append(n)
                continue
            seen.add(self.names[hit].casefold())
            kept.append(i)
            canonical.append(self.names[hit])
        return kept, canonical, dropped

    def positions(self, canonical: list[str]) -> np.ndarray:
        return np.stack([self.xyz[self.exact[n]] for n in canonical]).astype(np.float32)


# ----------------------------------------------------------------------------- positions from the dataset
ENTITY_RE = re.compile(r"(?:^|_)(sub|ses)-([^_]+)")


def find_electrodes_tsv(root: Path, relpath: str) -> Path | None:
    """The *_electrodes.tsv that applies to the recording at root/relpath, by BIDS inheritance: its own folder first,
    then the session and subject folders. Within a folder: a file whose sub/ses entities match the recording's, then
    one without `space-`; another subject's file never."""
    root = Path(root)
    rec = root / relpath
    ents = dict(ENTITY_RE.findall(rec.name))
    for d in rec.parents:
        if d == root or root not in d.parents:
            break
        ranked = []
        for f in sorted(d.glob("*_electrodes.tsv")):
            fe = dict(ENTITY_RE.findall(f.name))
            if "sub" in fe and "sub" in ents and fe["sub"] != ents["sub"]:
                continue
            ranked.append((not all(ents.get(k) == v for k, v in fe.items()), "space-" in f.name, f.name, f))
        if ranked:
            return min(ranked)[-1]
    return None


def read_electrodes_tsv(path: Path) -> dict[str, np.ndarray]:
    """name -> (x, y, z) in the file's own units and frame; rows with n/a or all-zero coordinates are skipped (some
    datasets write 0 0 0 for an electrode they did not digitise). The first of two spellings wins."""
    out: dict[str, np.ndarray] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                xyz = np.array([float(row[k]) for k in ("x", "y", "z")])
            except (KeyError, TypeError, ValueError):
                continue
            name = str(row.get("name") or "").strip()
            if name and np.all(np.isfinite(xyz)) and np.any(xyz != 0):
                out.setdefault(name, xyz)
    return out


def similarity_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale s, rotation R and translation t minimising |dst - (s R src + t)|^2 (Umeyama 1991). R is a proper
    rotation (det +1): a mirrored file must fail the residual test, not be silently flipped."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x, y = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(y.T @ x / len(src))
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1
    r = u @ np.diag(sign) @ vt
    var = (x ** 2).sum() / len(src)
    s = float((d * sign).sum() / var) if var > 0 else 0.0
    return s, r, mu_d - s * r @ mu_s


def dataset_positions(path: Path, names: list[str], kept: list[int], canonical: list[str], bank: Bank
                      ) -> tuple[list[int], list[str], np.ndarray, float | None]:
    """Channels of `names` the bank does not know, placed from the dataset's electrodes.tsv after fitting the file to
    the bank on the channels `resolve` kept (ds002680's primed 10-20 names, CB1/CB2, EP1/EP2). Returns (indices into
    names, canonical names = the file's spelling, positions (k, 3) metres in the bank's frame, RMS residual of the fit
    in metres or None when no fit was made). Empty when the file is unreadable, shares too few channels or fits badly."""
    none = ([], [], np.zeros((0, 3), dtype=np.float32))
    try:
        coords = read_electrodes_tsv(path)
    except (OSError, UnicodeDecodeError, csv.Error):
        return (*none, None)
    folded: dict[str, str] = {}
    for k in coords:
        folded.setdefault(k.casefold(), k)

    def lookup(n: str) -> str | None:
        for c in dict.fromkeys((n, strip_eeg_prefix(n))):
            if c in coords:
                return c
            if c.casefold() in folded:
                return folded[c.casefold()]
        return None

    pairs = [(coords[k], bank.xyz[bank.exact[c]]) for i, c in zip(kept, canonical) if (k := lookup(names[i]))]
    if len(pairs) < ALIGN_MIN_SHARED:
        return (*none, None)
    src, dst = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=np.float64)
    s, r, t = similarity_fit(src, dst)
    rms = float(np.sqrt(((s * src @ r.T + t - dst) ** 2).sum(1).mean()))
    if not np.isfinite(rms) or s <= 0 or rms > ALIGN_MAX_RMS_M:
        return (*none, rms)
    taken = {c.casefold() for c in canonical}
    idx, out, pos = [], [], []
    keep = set(kept)
    for i, n in enumerate(names):
        if i in keep or NON_EEG_RE.search(n.strip()) or bank._lookup(n) is not None or bank._lookup(strip_eeg_prefix(n)) is not None:
            continue   # an electrode the bank knows (a duplicate, or a ring label of a cap the bank lacks) is not re-placed
        k = lookup(n)
        if k is None or k.casefold() in taken:
            continue
        taken.add(k.casefold())
        idx.append(i)
        out.append(k)
        pos.append(s * r @ coords[k] + t)
    if not idx:
        return (*none, rms)
    return idx, out, np.array(pos, dtype=np.float32), rms


# ----------------------------------------------------------------------------- preprocessing
@dataclass
class Prepared:
    data: np.ndarray  # (T, C) float32, channels in sorted canonical order
    names: list[str]  # sorted canonical names
    dropped: list[str]
    sfreq_native: float
    typed_from_names: bool = False  # channel types came from the position bank, not from the file's metadata
    positions: np.ndarray | None = None  # (C, 3) float32 metres, same order as names; None = all from the writer's bank
    aligned: list[str] = field(default_factory=list)  # names placed from the dataset's electrodes.tsv, not the bank


def positions_of(prep: Prepared, bank: Bank) -> np.ndarray:
    """The positions a writer stores for a session: the recording's own when preprocessing set them."""
    return prep.positions if prep.positions is not None else bank.positions(prep.names)


@dataclass
class Skipped:
    reason: str
    detail: str = ""


def preprocess_recording(raw, bank: Bank, n_jobs: int = 1, electrodes: Path | None = None) -> Prepared | Skipped:
    """REVE contract on one MNE Raw (not yet loaded): EEG only, known positions, 0.5-99.5 Hz, 200 Hz. `electrodes`,
    the recording's *_electrodes.tsv, places EEG channels the bank does not know (dataset_positions)."""
    sfreq = float(raw.info["sfreq"])
    if sfreq < SFREQ - 1e-6:
        return Skipped("sfreq_lt_200", f"{sfreq:g} Hz")
    if raw.n_times / sfreq <= WINDOW / SFREQ:  # <= 10 s -> would give <= 2000 samples
        return Skipped("too_short", f"{raw.n_times / sfreq:.1f} s")
    types = raw.get_channel_types()
    eeg = [n for n, t in zip(raw.ch_names, types) if t == "eeg"]
    typed_from_names = False
    if not eeg:
        # Some datasets ship channels.tsv with type "n/a" (ds005697, ds002680, ds004315); EEGDash then marks
        # every channel "misc". Fall back to all channels and let the position bank decide: a name that
        # resolves to a scalp electrode is EEG.
        eeg, typed_from_names = list(raw.ch_names), True
    kept, canonical, dropped = bank.resolve(eeg)
    if typed_from_names and not kept:
        return Skipped("no_eeg", "no EEG-typed channels and no name resolves in the position bank")
    if len(kept) < MIN_RESOLVABLE:
        return Skipped("lt3_resolvable", f"{len(kept)} of {len(eeg)}")
    a_idx, a_names, a_pos = [], [], np.zeros((0, 3), dtype=np.float32)
    if electrodes is not None and dropped:
        a_idx, a_names, a_pos, _ = dataset_positions(Path(electrodes), eeg, kept, canonical, bank)
        placed = {eeg[i] for i in a_idx}
        dropped = [n for n in dropped if n not in placed]
    idx, canonical = kept + a_idx, canonical + a_names
    pos = np.concatenate([bank.positions(canonical[:len(kept)]), a_pos])
    by_file = np.argsort(idx, kind="stable")  # pick in file order, so the data rows follow idx whatever pick does
    idx, canonical, pos = [idx[k] for k in by_file], [canonical[k] for k in by_file], pos[by_file]
    raw = raw.copy().pick([eeg[i] for i in idx]).load_data(verbose="ERROR")
    if typed_from_names:  # MNE filters "data" channels only; these are EEG in all but their metadata
        raw.set_channel_types({n: "eeg" for n in raw.ch_names}, verbose="ERROR")
    jobs = filter_jobs(len(raw.ch_names) * raw.n_times * 8, n_jobs)   # float64 in memory; no copy to count it
    raw.filter(L_FREQ, min(H_FREQ, 0.45 * sfreq), n_jobs=jobs, verbose="ERROR")
    if abs(sfreq - SFREQ) > 1e-6:
        raw.resample(SFREQ, method="polyphase", n_jobs=jobs, verbose="ERROR")
    order = np.argsort(canonical)
    data = raw.get_data(units="uV")[order].T  # (T, C)
    prep = Prepared(np.ascontiguousarray(data, dtype="<f4"), [canonical[i] for i in order], dropped, sfreq,
                    typed_from_names=typed_from_names, positions=np.ascontiguousarray(pos[order], dtype=np.float32),
                    aligned=sorted(a_names))
    return prep


# ----------------------------------------------------------------------------- writer
@dataclass
class Group:
    index: int
    names: list[str]
    n_sessions: int = 0
    duration: int = 0
    corr_rows: list[dict] = field(default_factory=list)


class GroupWriter:
    """Streams sessions into REVE's raw memmap layout, one big recording per channel set."""

    def __init__(self, out: Path, dataset: str, bank: Bank):
        self.out, self.dataset, self.bank = Path(out), dataset, bank
        for d in ("recordings", "positions", "stats", "csv_recordings"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.groups: dict[tuple, Group] = {}
        self.manifest: list[dict] = []

    def append(self, prep: Prepared, file: str, meta: dict) -> int:
        T, C = prep.data.shape
        if T <= WINDOW:  # never write a session the loader will drop (stats rows would misalign)
            raise ValueError(f"session {file} has {T} <= {WINDOW} samples; skip it before calling append")
        key = tuple(prep.names)
        grp = self.groups.get(key)
        if grp is None:
            grp = self.groups[key] = Group(len(self.groups), list(prep.names))
            # positions of the group's first session: a bank name has one position; a name placed from a dataset's
            # electrodes.tsv may differ by a few mm between subjects of one shard, and the first wins
            np.save(self.out / "positions" / f"recording_-_positions_-_{grp.index}.npy", positions_of(prep, self.bank))
        g = grp.index
        with open(self.out / "recordings" / f"recording_-_eeg_-_{g}.npy", "ab") as f:
            f.write(prep.data.tobytes(order="C"))
        stats = np.stack([prep.data.mean(0), prep.data.std(0)]).astype("<f4")  # (2, C)
        with open(self.out / "stats" / f"recording_-_stats_-_{g}.npy", "ab") as f:
            f.write(stats.tobytes(order="C"))
        grp.corr_rows.append(dict(file=file, big_recording_index=g, index=grp.n_sessions, duration=T, n_chans=C,
                                  mult=1, dataset=self.dataset, **{"class": "pretrain"}, flag_remove="False",
                                  flag_reduce="[]", n_chans_to_remove=0))
        self.manifest.append(dict(dataset_id=self.dataset, file=file, big_recording_index=g, session_index=grp.n_sessions,
                                  n_chans=C, duration_s=round(T / SFREQ, 2), sfreq_native=prep.sfreq_native,
                                  n_dropped_no_position=len(prep.dropped), n_positions_aligned=len(prep.aligned), **meta))
        grp.n_sessions += 1
        grp.duration += T
        return g

    def finalize(self, summary: dict) -> None:
        groups = sorted(self.groups.values(), key=lambda g: g.index)
        big = [dict(big_recording_index=g.index, duration=g.duration, n_chans=len(g.names),
                    chans_names=json.dumps(g.names), config=self.dataset) for g in groups]
        corr = [r for g in groups for r in g.corr_rows]
        stats = [dict(big_recording_index=g.index, n_sessions=g.n_sessions, n_chans=len(g.names)) for g in groups]
        csvd = self.out / "csv_recordings"
        write_csv(csvd / "df_big.csv", big, ["big_recording_index", "duration", "n_chans", "chans_names", "config"])
        write_csv(csvd / "df_corrected.csv", corr, CORR_FIELDS)
        write_csv(csvd / "df_stats_tmp.csv", stats, ["big_recording_index", "n_sessions", "n_chans"])
        write_csv(self.out / "manifest_part.csv", self.manifest,
                  list(self.manifest[0].keys()) if self.manifest else ["dataset_id", "file"])
        summary.update(n_big_recordings=len(groups), n_sessions=len(corr),
                       hours_written=round(sum(g.duration for g in groups) / SFREQ / 3600, 3))
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        (self.out / "_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


SHARD_SAMPLES = 128  # samples per Zarr shard file: ~130 MB at 129 channels, so millions of samples are thousands of files


def sample_offsets(T: int) -> np.ndarray:
    """Start of every training sample in a session of T timesteps, exactly as REVE cuts them.

    reve_eeg's _make_window_keys uses np.arange(start, end - window, window). The exclusive end means a session of
    exactly k*2000 timesteps yields k-1 samples, not k — the last full window is dropped. Reproduced here on purpose,
    so the Zarr corpus holds precisely the samples REVE trains on."""
    return np.arange(0, T - WINDOW, WINDOW)


@dataclass
class ZarrGroup:
    index: int
    names: list[str]
    data: object                                    # the zarr array (N, WINDOW, C)
    positions: np.ndarray                           # (C, 3) metres, of the group's first session (as GroupWriter)
    n_written: int = 0
    pending: list = field(default_factory=list)      # sample blocks not yet written, each (n, WINDOW, C)
    n_pending: int = 0
    session_of_sample: list = field(default_factory=list)
    stats: list = field(default_factory=list)        # (2, C) per session
    sessions: list = field(default_factory=list)     # provenance per session
    corr_rows: list = field(default_factory=list)


class ZarrGroupWriter:
    """Streams sessions into one Zarr store of training samples, one group per channel set.

    Layout (`<out>/samples.zarr`):  <g>/data (N, 2000, C) float32 — one row per REVE training sample
                                    <g>/session (N,) int32      — which session each sample came from
                                    <g>/stats (S, 2, C) float32 — per-session mean/std, for REVE's normalisation
                                    <g>/positions (C, 3) float32 — metres, from the position bank or the dataset
    One chunk per sample, grouped SHARD_SAMPLES to a file. Signals are microvolts, not normalised."""

    def __init__(self, out: Path, dataset: str, bank: Bank, compress: bool = True):
        import zarr
        from zarr.codecs import BloscCodec

        self.out, self.dataset, self.bank = Path(out), dataset, bank
        self.out.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(self.out / "samples.zarr", mode="w")
        self.root.attrs.update(format="reve-samples", version=1, sfreq=SFREQ, window=WINDOW, units="uV",
                               band_pass_hz=[L_FREQ, H_FREQ], dataset=dataset)
        self.compressors = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle") if compress else None
        self.groups: dict[tuple, ZarrGroup] = {}
        self.manifest: list[dict] = []

    def _group(self, prep: Prepared) -> ZarrGroup:
        names = prep.names
        key = tuple(names)
        if key not in self.groups:
            idx, C = len(self.groups), len(names)
            g = self.root.create_group(str(idx))
            data = g.create_array("data", shape=(0, WINDOW, C), chunks=(1, WINDOW, C),
                                  shards=(SHARD_SAMPLES, WINDOW, C), dtype="float32", compressors=self.compressors)
            self.groups[key] = ZarrGroup(idx, list(names), data, positions_of(prep, self.bank))
        return self.groups[key]

    def _flush(self, grp: ZarrGroup, final: bool = False) -> None:
        """Write pending samples in whole shards (a partial shard only at the end), so no shard is rewritten."""
        if not grp.n_pending:
            return
        block = np.concatenate(grp.pending)
        n = len(block) if final else (len(block) // SHARD_SAMPLES) * SHARD_SAMPLES
        if n:
            grp.data.resize((grp.n_written + n, WINDOW, len(grp.names)))
            grp.data[grp.n_written:grp.n_written + n] = block[:n]
            grp.n_written += n
        grp.pending, grp.n_pending = ([block[n:]] if n < len(block) else []), len(block) - n

    def append(self, prep: Prepared, file: str, meta: dict) -> int:
        T, C = prep.data.shape
        if T <= WINDOW:  # never write a session the loader will drop (stats rows would misalign)
            raise ValueError(f"session {file} has {T} <= {WINDOW} samples; skip it before calling append")
        grp = self._group(prep)
        s = len(grp.sessions)
        n = len(sample_offsets(T))
        samples = prep.data[: n * WINDOW].reshape(n, WINDOW, C)   # offsets are 0, 2000, 4000, … so this is exact
        first = grp.n_written + grp.n_pending
        grp.pending.append(samples)
        grp.n_pending += n
        grp.session_of_sample += [s] * n
        grp.stats.append(np.stack([prep.data.mean(0), prep.data.std(0)]).astype("<f4"))  # over the whole session
        grp.sessions.append(dict(file=file, duration=T, n_samples=n, first_sample=first))
        grp.corr_rows.append(dict(file=file, big_recording_index=grp.index, index=s, duration=T, n_chans=C,
                                  mult=1, dataset=self.dataset, **{"class": "pretrain"}, flag_remove="False",
                                  flag_reduce="[]", n_chans_to_remove=0))
        self.manifest.append(dict(dataset_id=self.dataset, file=file, big_recording_index=grp.index, session_index=s,
                                  n_chans=C, duration_s=round(T / SFREQ, 2), sfreq_native=prep.sfreq_native,
                                  n_dropped_no_position=len(prep.dropped), n_positions_aligned=len(prep.aligned),
                                  n_samples=n, **meta))
        self._flush(grp)
        return grp.index

    def finalize(self, summary: dict) -> None:
        groups = sorted(self.groups.values(), key=lambda g: g.index)
        for grp in groups:
            self._flush(grp, final=True)
            g = self.root[str(grp.index)]
            g.create_array("session", data=np.asarray(grp.session_of_sample, dtype="int32"))
            g.create_array("stats", data=np.stack(grp.stats))
            g.create_array("positions", data=grp.positions)
            g.attrs.update(channels=grp.names, dataset=self.dataset, sessions=grp.sessions)
        write_csv(self.out / "manifest_part.csv", self.manifest,
                  list(self.manifest[0].keys()) if self.manifest else ["dataset_id", "file"])
        summary.update(format="zarr", n_big_recordings=len(groups), n_sessions=sum(len(g.sessions) for g in groups),
                       n_samples=sum(g.n_written for g in groups),
                       hours_written=round(sum(s["duration"] for g in groups for s in g.sessions) / SFREQ / 3600, 3),
                       hours_in_samples=round(sum(g.n_written for g in groups) * WINDOW / SFREQ / 3600, 3))
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        (self.out / "_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- records
def discover_records(cache_dir: Path, dataset: str, subjects: list[str] | None, offline: bool):
    """EEGDash records of one dataset. offline=True indexes the local BIDS mirror (no API);
    offline=False queries the catalog and downloads lazily (needed for NEMAR nm* ids)."""
    from eegdash import EEGDashDataset

    kwargs = dict(cache_dir=str(cache_dir), dataset=dataset, download=not offline, n_jobs=1)
    if subjects:
        kwargs["subject"] = list(subjects)
    try:
        recs = list(EEGDashDataset(**kwargs).datasets)
    except ValueError as err:  # braindecode: "datasets should not be an empty iterable"
        if "empty" not in str(err):
            raise
        where = f"{cache_dir / dataset}" if offline else "the EEGDash catalog"
        raise SystemExit(f"no recordings found for {dataset} subjects={subjects or 'all'} in {where} "
                         f"(wrong subject labels? dataset not downloaded?)") from None

    def sort_key(rec):
        d = rec.description
        return tuple(str(d.get(k) or "") for k in ("subject", "session", "run")) + (record_file(rec),)

    return sorted(recs, key=sort_key)


PRIMARY_EXT = (".set", ".vhdr", ".edf", ".bdf", ".fif", ".gdf", ".cnt")  # data files; .eeg/.fdt/.vmrk are companions


@dataclass
class LocalRecord:
    """A recording of a local BIDS tree read without EEGDash, shaped like the EEGDash records the converter uses."""
    path: Path
    root: Path

    @property
    def record(self) -> dict:
        return {"bids_relpath": self.path.relative_to(self.root).as_posix()}

    @property
    def description(self) -> dict:
        from mne_bids import get_entities_from_fname

        e = get_entities_from_fname(self.path.name)
        return {"subject": e.get("subject"), "session": e.get("session"), "run": e.get("run"), "task": e.get("task")}

    @property
    def raw(self):
        from mne_bids import BIDSPath, get_entities_from_fname, read_raw_bids

        e = {k: v for k, v in get_entities_from_fname(self.path.name).items() if v is not None}
        bp = BIDSPath(root=self.root, datatype="eeg", suffix="eeg", extension=self.path.suffix, check=False, **e)
        return read_raw_bids(bp, verbose="ERROR")


def discover_bids(root: Path, subjects: list[str] | None) -> list[LocalRecord]:
    """Recordings of a local BIDS tree EEGDash does not carry (I-CARE), listed per subject folder. EEGDash's offline
    discovery walks the whole tree for every shard (I-CARE's 36,123 files: over 30 min before the first recording,
    job 54436119) and records paths relative to the resolved root, which breaks behind a symlink (2026-09-24)."""
    subs = [f"sub-{s}" for s in subjects] if subjects else sorted(d.name for d in root.glob("sub-*") if d.is_dir())
    files = sorted(p for s in subs for p in (root / s).glob("**/eeg/*_eeg.*") if p.suffix in PRIMARY_EXT)
    if not files:
        raise SystemExit(f"no recordings found for subjects={subjects or 'all'} under {root}")
    return [LocalRecord(f, root) for f in files]


def record_file(rec) -> str:
    return str(rec.record.get("bids_relpath") or rec.record.get("bidspath") or rec.record.get("data_name") or "?")


def record_electrodes(rec, cache_dir: Path, dataset: str) -> Path | None:
    """The *_electrodes.tsv of a record: LocalRecord paths are relative to its root, EEGDash's bids_relpath to its
    bids_root (cache_dir/dataset unless the record says otherwise)."""
    if isinstance(rec, LocalRecord):
        return find_electrodes_tsv(rec.root, rec.path.relative_to(rec.root).as_posix())
    rel = rec.record.get("bids_relpath")
    if not rel:
        return None
    return find_electrodes_tsv(Path(getattr(rec, "bids_root", None) or Path(cache_dir) / dataset), str(rel))


def record_meta(rec) -> dict:
    d = rec.description
    return {k: ("" if d.get(k) is None else str(d.get(k))) for k in ("subject", "session", "run", "task")}


# ----------------------------------------------------------------------------- shard
def convert_shard(dataset: str, subjects: list[str] | None, cache_dir: Path, out: Path, *, offline: bool = True,
                  n_jobs: int = 1, max_records: int | None = None, bank: Bank | None = None,
                  fmt: str = "memmap", compress: bool = True, bids_root: Path | None = None,
                  use_dataset_positions: bool = True) -> dict:
    """One shard. use_dataset_positions=False (--no-dataset-positions) drops every channel the bank does not know, as
    before electrodes.tsv files were read."""
    if dataset in HELD_OUT:
        raise SystemExit(f"{dataset} is an OpenEEGBench evaluation set and must not be pretrained on")
    out = Path(out)
    if (out / "_DONE").exists():
        print(f"{out} already complete (_DONE); nothing to do")
        return json.loads((out / "summary.json").read_text())
    if out.exists():
        print(f"{out} exists without _DONE: incomplete shard, wiping")
        shutil.rmtree(out)
    t0 = time.time()
    bank = bank or Bank.from_braindecode()
    recs = discover_bids(Path(bids_root), subjects) if bids_root else discover_records(cache_dir, dataset, subjects, offline)
    if max_records:
        recs = recs[:max_records]
    print(f"== {dataset} subjects={subjects or 'all'} offline={offline}: {len(recs)} recordings")

    writer = GroupWriter(out, dataset, bank) if fmt == "memmap" else ZarrGroupWriter(out, dataset, bank, compress)
    skipped, dropped_names, aligned_names, sfreqs = [], Counter(), Counter(), Counter()
    n_typed_from_names = 0
    for k, rec in enumerate(recs, 1):
        file = record_file(rec)
        try:
            electrodes = record_electrodes(rec, cache_dir, dataset) if use_dataset_positions else None
            res = preprocess_recording(rec.raw, bank, n_jobs=n_jobs, electrodes=electrodes)
        except Exception as err:  # keep the shard going; the summary lists the failure
            res = Skipped("load_error", f"{err.__class__.__name__}: {err}"[:300])
        if isinstance(res, Skipped):
            skipped.append(dict(file=file, reason=res.reason, detail=res.detail))
            print(f"   [{k}/{len(recs)}] skip {file}: {res.reason} {res.detail}", flush=True)
            continue
        sfreqs[str(int(res.sfreq_native))] += 1
        dropped_names.update(res.dropped)
        aligned_names.update(res.aligned)
        n_typed_from_names += res.typed_from_names
        g = writer.append(res, file, record_meta(rec))
        print(f"   [{k}/{len(recs)}] {file}: {res.data.shape[1]} ch, {res.data.shape[0] / SFREQ:.0f} s, "
              f"{int(res.sfreq_native)} Hz -> group {g}" + (f", dropped {len(res.dropped)}" if res.dropped else "")
              + (f", aligned {len(res.aligned)}" if res.aligned else "")
              + (" (types from names)" if res.typed_from_names else ""), flush=True)

    summary = dict(dataset=dataset, subjects=subjects, offline=offline, n_records_seen=len(recs),
                   n_records_typed_from_names=n_typed_from_names,
                   skipped=skipped, skipped_by_reason=dict(Counter(s["reason"] for s in skipped)),
                   channels_dropped_no_position=dict(dropped_names.most_common()),
                   channels_aligned_from_dataset=dict(aligned_names.most_common()), sfreq_native_hist=dict(sfreqs),
                   elapsed_s=round(time.time() - t0, 1))
    writer.finalize(summary)
    print(f"done {dataset}: {summary['n_sessions']} sessions in {summary['n_big_recordings']} big recordings, "
          f"{summary['hours_written']} h, {len(skipped)} skipped, {summary['elapsed_s']} s -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="EEGDash dataset id (ds*, nm*)")
    ap.add_argument("--subjects", nargs="*", default=None, help="subject labels for this shard (default: all)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True, help="this shard's part directory")
    ap.add_argument("--online", action="store_true", help="query the EEGDash catalog and download (NEMAR); "
                                                          "default is offline indexing of the local mirror")
    ap.add_argument("--n-jobs", type=int, default=1, help="threads for MNE filter/resample")
    ap.add_argument("--max-records", type=int, default=None, help="cap for dry runs")
    ap.add_argument("--positions-json", default=None, help="position bank JSON (default: braindecode's cached copy)")
    ap.add_argument("--format", choices=["memmap", "zarr"], default="memmap",
                    help="memmap: REVE's raw layout; zarr: training samples (N, 2000, C) per channel set, one chunk each")
    ap.add_argument("--no-compress", action="store_true", help="zarr only: store chunks uncompressed")
    ap.add_argument("--bids-root", default=None, help="read this local BIDS tree directly, not through EEGDash "
                    "(datasets EEGDash does not carry, e.g. I-CARE); --dataset then only names the output")
    ap.add_argument("--no-dataset-positions", action="store_true", help="drop every channel the position bank does not "
                    "know, even when the recording's electrodes.tsv places it")
    a = ap.parse_args()
    bank = Bank.from_json(a.positions_json) if a.positions_json else None
    convert_shard(a.dataset, a.subjects, Path(a.cache_dir), Path(a.out), offline=not a.online,
                  n_jobs=a.n_jobs, max_records=a.max_records, bank=bank, fmt=a.format, compress=not a.no_compress,
                  bids_root=Path(a.bids_root) if a.bids_root else None, use_dataset_positions=not a.no_dataset_positions)


if __name__ == "__main__":
    main()
