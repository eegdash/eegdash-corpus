"""Plan the conversion: scan recording headers (no data read) and pack subjects into shards.

  python make_shards.py --cache-dir $EEGDASH_CACHE --out $CONVROOT/plan \
      --offline-ids openneuro_ids_other.txt openneuro_ids_hbn.txt --online-ids nemar_ids.txt [--workers 8]

Writes  <out>/records_<dsid>.csv   one row per recording: subject session run task file sfreq nchans ntimes hours
        <out>/shards.tsv           shard_id  dataset  mode  subjects  n_records  est_hours   (tab-separated;
                                   subjects = space-separated labels or "all"; mode = offline|online)
        <out>/plan_summary.json

Scanner: the BIDS tree is walked directly (sub-*/[ses-*/]eeg/*_eeg.<set|vhdr|edf|bdf|...>) and
SamplingFrequency / RecordingDuration / EEGChannelCount are read from the recording's sidecar, following
BIDS inheritance (a dataset-level task-<x>_eeg.json covers recordings without their own JSON - 8 of the
55 datasets are like that). When no sidecar gives the duration, the data file's header is read with MNE
(preload=False). EEGDash's offline
discovery would give the same numbers but globs the tree through mne_bids for every record, which on
Lustre ran at ~2.5 s per recording (2026-09-18); the converter still goes through EEGDash, this is only
the plan. Datasets are scanned in parallel processes.

Online datasets (NEMAR nm*, small): one shard each, all subjects, converted with --online.
Subjects are packed greedily in sorted order until a shard would exceed --max-hours or --max-records.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

HELD_OUT = {"nm000139", "ds004362", "nm000112", "nm000114", "nm000109", "nm000110", "nm000113"}
DEFAULT_HOURS = 0.25  # per record when neither the sidecar nor the header gives a duration
ENTITY = re.compile(r"(?:^|_)(sub|ses|run|task|acq)-([^_.]+)")


def read_ids(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def entities(name: str) -> dict:
    return {k: v for k, v in ENTITY.findall(name)}


def header_duration(path: Path) -> tuple[float | None, float | None, int | None]:
    """(sfreq, hours, n_eeg) from the data file header via MNE; None on failure."""
    try:
        import mne

        raw = mne.io.read_raw(path, preload=False, verbose="ERROR")
        n_eeg = sum(t == "eeg" for t in raw.get_channel_types())
        return float(raw.info["sfreq"]), raw.n_times / raw.info["sfreq"] / 3600, n_eeg
    except Exception:
        return None, None, None


PRIMARY_EXT = (".set", ".vhdr", ".edf", ".bdf", ".fif", ".gdf", ".cnt", ".mff")  # .eeg/.fdt/.vmrk are companions


def find_sidecar(data: Path, root: Path) -> dict:
    """BIDS inheritance: <stem>.json next to the file, else the nearest *_eeg.json up the tree whose
    entities are all present in the file name (typically a dataset-level task-<x>_eeg.json)."""
    own = data.with_suffix(".json")
    if own.exists():
        try:
            return json.loads(own.read_text())
        except Exception:
            return {}
    ent = entities(data.name)
    for d in [data.parent, *data.parents]:
        for cand in sorted(d.glob("*_eeg.json")):
            cent = entities(cand.name)
            if all(ent.get(k) == v for k, v in cent.items()):
                try:
                    return json.loads(cand.read_text())
                except Exception:
                    return {}
        if d == root:
            break
    return {}


def scan_dataset(cache_dir: Path, dsid: str, threads: int = 16) -> list[dict]:
    """Header-only record list of one locally mirrored dataset: BIDS sidecar (with inheritance), MNE header
    as fallback. Walks the data files, so recordings without their own JSON are still found. One tree walk and
    threaded sidecar reads: on Lustre each small-file open waits on the metadata server, and I-CARE's 36,123
    recordings took over 30 min read serially with one walk per extension (2026-09-24)."""
    root = cache_dir / dsid
    files = sorted(p for p in root.glob("sub-*/**/eeg/*_eeg.*") if p.suffix in PRIMARY_EXT)
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda data: scan_record(data, root, dsid), files))


def scan_record(data: Path, root: Path, dsid: str) -> dict:
    """One recording's row: sidecar first, the data file's header only when the sidecar lacks duration or rate."""
    ent = entities(data.name)
    meta = find_sidecar(data, root)
    sfreq, dur_s, n_eeg = meta.get("SamplingFrequency"), meta.get("RecordingDuration"), meta.get("EEGChannelCount")
    hours = dur_s / 3600 if isinstance(dur_s, (int, float)) and dur_s > 0 else None
    if hours is None or not sfreq:
        h_sfreq, h_hours, h_n = header_duration(data)
        sfreq, hours, n_eeg = sfreq or h_sfreq, hours if hours is not None else h_hours, n_eeg or h_n
    return dict(dataset=dsid, subject=ent.get("sub", ""), session=ent.get("ses", ""), run=ent.get("run", ""),
                task=ent.get("task", ""), file=str(data.relative_to(root)), sfreq=sfreq, nchans=n_eeg,
                ntimes=int(hours * 3600 * sfreq) if (hours is not None and sfreq) else None,
                hours=round(hours, 4) if hours is not None else None)


def pack_subjects(records: list[dict], max_hours: float, max_records: int) -> list[list[str]]:
    """Greedy packing of subjects (sorted) into shards; a single subject may exceed the limits alone."""
    per_subject: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for r in records:
        h, n = per_subject[r["subject"]]
        per_subject[r["subject"]] = (h + (r["hours"] if r["hours"] is not None else DEFAULT_HOURS), n + 1)
    shards, cur, cur_h, cur_n = [], [], 0.0, 0
    for subj in sorted(per_subject):
        h, n = per_subject[subj]
        if cur and (cur_h + h > max_hours or cur_n + n > max_records):
            shards.append(cur)
            cur, cur_h, cur_n = [], 0.0, 0
        cur.append(subj)
        cur_h += h
        cur_n += n
    if cur:
        shards.append(cur)
    return shards


def shard_rows(dsid: str, mode: str, records: list[dict], groups: list[list[str]]) -> list[dict]:
    rows = []
    for k, subs in enumerate(groups):
        sel = [r for r in records if r["subject"] in set(subs)] if subs != ["all"] else records
        rows.append(dict(shard_id=f"{dsid}_{k:03d}", dataset=dsid, mode=mode, subjects=" ".join(subs),
                         n_records=len(sel),
                         est_hours=round(sum((r["hours"] if r["hours"] is not None else DEFAULT_HOURS) for r in sel), 2)))
    return rows


def _scan_job(args):
    cache_dir, dsid = args
    return dsid, scan_dataset(Path(cache_dir), dsid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offline-ids", nargs="*", default=[], help="id lists of locally mirrored datasets")
    ap.add_argument("--online-ids", nargs="*", default=[], help="id lists converted through the EEGDash API (NEMAR)")
    ap.add_argument("--max-hours", type=float, default=200.0)
    ap.add_argument("--max-records", type=int, default=400)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    offline = [i for p in a.offline_ids for i in read_ids(p)]
    online = [i for p in a.online_ids for i in read_ids(p)]
    bad = (set(offline) | set(online)) & HELD_OUT
    if bad:
        raise SystemExit(f"held-out dataset(s) in the id lists: {sorted(bad)}")
    present = [d for d in offline if (Path(a.cache_dir) / d).exists()]
    for d in offline:
        if d not in present:
            print(f"!! {d}: not in {a.cache_dir}, skipped (download not finished?)", flush=True)

    shards, summary = [], {d: dict(status="missing") for d in offline if d not in present}
    with ProcessPoolExecutor(a.workers) as pool:
        for dsid, recs in pool.map(_scan_job, [(a.cache_dir, d) for d in present]):
            with open(out / f"records_{dsid}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[0].keys()) if recs else ["dataset"])
                w.writeheader()
                w.writerows(recs)
            groups = pack_subjects(recs, a.max_hours, a.max_records)
            rows = shard_rows(dsid, "offline", recs, groups)
            shards += rows
            hours = sum(r["est_hours"] for r in rows)
            n_unknown = sum(r["hours"] is None for r in recs)
            summary[dsid] = dict(status="ok", n_records=len(recs), n_subjects=len({r["subject"] for r in recs}),
                                 est_hours=round(hours, 1), n_shards=len(rows), n_records_without_duration=n_unknown)
            print(f"{dsid}: {len(recs)} records, {summary[dsid]['n_subjects']} subjects, ~{hours:.0f} h -> {len(rows)} shard(s)"
                  + (f"  ({n_unknown} without duration)" if n_unknown else ""), flush=True)
    for dsid in online:
        shards += shard_rows(dsid, "online", [], [["all"]])
        summary[dsid] = dict(status="online", n_shards=1)
        print(f"{dsid}: online, 1 shard", flush=True)

    shards.sort(key=lambda r: (-r["est_hours"], r["shard_id"]))  # biggest first: long shards start early in the array
    with open(out / "shards.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shard_id", "dataset", "mode", "subjects", "n_records", "est_hours"], delimiter="\t")
        w.writeheader()
        w.writerows(shards)
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=1))
    total_h = sum(r["est_hours"] for r in shards)
    print(f"\n{len(shards)} shards, ~{total_h:.0f} h of EEG -> {out / 'shards.tsv'}", flush=True)
    print(f"convert each row of {out / 'shards.tsv'} with scripts/eegdash_to_reve.py (one job per shard)", flush=True)


if __name__ == "__main__":
    main()
