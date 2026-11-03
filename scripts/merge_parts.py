"""Merge finished shard parts into one REVE layout, verify it, and write the manifest + report; relabel a dataset id
in a merged Zarr store.

  python merge_parts.py [merge] --parts $CONVROOT/parts --out $CONVROOT/memmap \
      [--shards $CONVROOT/plan/v1/shards.tsv] [--allow-missing]
  python merge_parts.py [merge] --parts $CONVROOT/parts_zarr --out $CONVROOT/zarr \
      --name eegdash-corpus-v1 --shards $CONVROOT/plan/eegdash-corpus-v1.tsv [--results $OUTROOT/conversion/zarr]
  python merge_parts.py relabel --store $CONVROOT/zarr/eegdash-corpus-v0.zarr --from i-care --to nm000289 \
      [--parts $CONVROOT/parts_zarr] [--plans $CONVROOT/plan/eegdash-corpus-v0.tsv ...] [--side-only] [--dry-run]
  python merge_parts.py licence --store $CONVROOT/zarr/eegdash-corpus-v1.zarr --table corpus/eegdash-corpus-v1.csv [--dry-run]

* Parts are taken in shards.tsv order (or sorted directory order); every part must carry `_DONE`
  unless --allow-missing (then missing/incomplete parts are listed and skipped).
* big_recording_index is renumbered globally; recording/stats/positions files are HARD-LINKED into
  the output (same filesystem, no copy) - fall back to a copy if linking fails.
* Verifies: file size == duration * n_chans * 4, stats size == n_sessions * 2 * n_chans * 4,
  every session duration > 2000, no held-out dataset id anywhere.
* Writes csv_recordings/{df_big,df_corrected,df_stats_tmp}.csv (+ df_stats.csv, the release's name),
  manifest.csv (all manifest_part.csv rows with the new indices) and CONVERSION_REPORT.md.

Zarr parts (converter --format zarr) are detected and merged into one corpus store, <out>/<name>.zarr, from the parts
named in --shards (required: the shard list is what defines a corpus; parts_zarr/ holds more than one corpus) and
--name (required: a store is published under its name, e.g. eegdash-corpus-v0, eegdash-corpus-v1), with the
same guarantees: shard order, _DONE required, damaged parts refused, held-out ids refused, buckets renumbered 0..k-1 and
hard-linked (only the small zarr.json metadata files are copied, so provenance can be added without touching the
shard). Beside the store: <name>-buckets.csv, <name>-manifest.csv and <name>-CONVERSION_REPORT.md (outside it,
because Zarr warns about any foreign file in a store; prefixed, because eegdash-corpus-v0 and -v1 share <out>).
--results DIR also copies those three files into DIR (created if needed; $OUTROOT/conversion/zarr, which results_sync
brings to the laptop). Parts of the two formats cannot be mixed.

relabel renames a dataset id where a merged Zarr store and its side files record it, without opening an array: the
attributes `dataset` and `source_shard` of every bucket group labelled --from, the root attributes `datasets` and
`buckets`, the `dataset`/`shard` columns of <name>-buckets.csv, `dataset_id` of <name>-manifest.csv, the id and shard
names in <name>-CONVERSION_REPORT.md; shard names carrying the id as a prefix (i-care_046) become <to>_046. Each metadata
file is rewritten whole and swapped in by rename, so a hard link into a part is never written through. --parts renames
the part folders named after the id and rewrites the id inside every part named <from>_NNN or <to>_NNN (bucket and root
attributes of samples.zarr, `dataset_id` of manifest_part.csv, `dataset` of summary.json), so a re-merge keeps <to>;
--plans rewrites the shard lists that name them (shard_id and dataset columns). --side-only rewrites the side files
alone (copies kept elsewhere, e.g. in the repository). Everything is planned before anything is written: a target id
that already labels other buckets is refused, and a second run changes nothing. The arrays, hence the store's
`.verified` marker (file count and REVE's loader over every bucket), are untouched.

licence writes each dataset's licence into the store, so a bucket carries its own terms wherever it is copied
(corpus/HUGGINGFACE.md §4): every bucket group gets `licence` (the table's id, e.g. CC-BY-NC-SA-4.0), `licence_url` and
`source` (the dataset's page on OpenNeuro or NEMAR); the root gets `licence: "other"`, `licences` (dataset -> licence)
and `licence_file: "LICENSES.csv"`. The licences come from the corpus table (`dataset_id`, `licence`; a store id the
table spells otherwise is read through TABLE_IDS: nm000289 -> icare). Like relabel it
edits only zarr.json files, by rename, after planning everything: a dataset missing from the table, without a licence,
or with a licence that has no known URL stops the run before anything is written; a second run changes nothing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HELD_OUT = {"nm000139", "ds004362", "nm000112", "nm000114", "nm000109", "nm000110", "nm000113"}
# A store's dataset id -> the corpus tables' id where the two differ (corpus_tables.MANIFEST_ALIASES, kept equal by a
# test): the v1 store records I-CARE as nm000289, the tables call it icare since 2026-10-30.
TABLE_IDS = {"i-care": "icare", "nm000289": "icare"}
WINDOW = 2000
SFREQ = 200


def read_csv(p: Path, delimiter=",") -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(p: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Columns = the union over the rows, in order of first appearance, blanks where a row lacks one: parts converted
    before a column existed (n_positions_aligned, 2026-10-18) merge with parts converted after it."""
    if not fields:
        fields = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def part_problems(part: Path) -> list[str]:
    """File sizes of a finished part vs its CSVs. A mismatch means two workers wrote the same part
    concurrently (2026-09-19: ds005511_000 came out 1.9x oversized after a lock-takeover race)."""
    big = read_csv(part / "csv_recordings" / "df_big.csv")
    stats = {int(s["big_recording_index"]): s for s in read_csv(part / "csv_recordings" / "df_stats_tmp.csv")}
    out = []
    for b in big:
        i, T, C = int(b["big_recording_index"]), int(b["duration"]), int(b["n_chans"])
        rec = (part / "recordings" / f"recording_-_eeg_-_{i}.npy").stat().st_size
        st = (part / "stats" / f"recording_-_stats_-_{i}.npy").stat().st_size
        exp_rec, exp_st = T * C * 4, int(stats[i]["n_sessions"]) * 2 * C * 4
        if rec != exp_rec or st != exp_st:
            out.append(f"big recording {i}: file {rec / exp_rec:.2f}x the CSV size, stats {st / exp_st:.2f}x")
    return out


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


PART_STORE = "samples.zarr"                            # the store inside each part folder; see eegdash_to_reve.py


def is_zarr_part(part: Path) -> bool:
    return (part / PART_STORE).is_dir()


def zarr_problems(store: Path) -> list[str]:
    """A Zarr part is consistent when, for every bucket, samples, session ids, stats, channels and positions agree
    with the per-session record in its attrs. Same purpose as part_problems: catch a part two workers wrote at once."""
    import zarr

    out = []
    root = zarr.open_group(store, mode="r")
    for name, g in sorted(root.groups(), key=lambda kv: int(kv[0])):
        sessions, channels = g.attrs["sessions"], g.attrs["channels"]
        n, w, c = g["data"].shape
        session = g["session"][:]
        expected = [s for s, x in enumerate(sessions) for _ in range(x["n_samples"])]
        if w != WINDOW or c != len(channels) or g["positions"].shape != (c, 3):
            out.append(f"bucket {name}: data {g['data'].shape} vs {len(channels)} channels, positions {g['positions'].shape}")
        if n != len(session) or session.tolist() != expected:
            out.append(f"bucket {name}: {n} samples, {len(session)} session ids, {len(expected)} per the session record")
        if g["stats"].shape != (len(sessions), 2, c):
            out.append(f"bucket {name}: stats {g['stats'].shape} for {len(sessions)} sessions")
        if any(x["duration"] <= WINDOW for x in sessions):
            out.append(f"bucket {name}: a session of <= {WINDOW} timesteps")
    return out


def link_tree(src: Path, dst: Path) -> None:
    """Hard-link a Zarr group's chunk files; copy its small zarr.json metadata files, so attributes can be edited in
    the merged store without reaching back into the shard it came from."""
    for dirpath, _, files in os.walk(src):
        target = dst / Path(dirpath).relative_to(src)
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            (shutil.copy2 if f == "zarr.json" else link)(Path(dirpath) / f, target / f)


def merge_zarr(parts: list[Path], out: Path, incomplete: list[str], name: str, results: Path | None = None) -> dict:
    """One corpus store, <out>/<name>.zarr: every bucket of every shard, renumbered 0..k-1 in shard order; the side
    files beside it and, given `results`, copied there too."""
    import zarr

    damaged = {p.name: zarr_problems(p / PART_STORE) for p in parts}
    damaged = {k: v for k, v in damaged.items() if v}
    if damaged:
        for shard, msgs in damaged.items():
            for m in msgs:
                print(f"DAMAGED {shard}: {m}")
        raise SystemExit(f"{len(damaged)} damaged part(s) - delete them so a worker redoes them: {' '.join(damaged)}")

    store = out / f"{name}.zarr"
    if store.exists():
        shutil.rmtree(store)
    out.mkdir(parents=True, exist_ok=True)
    corpus = zarr.open_group(store, mode="w")
    index, corr, manifest, summaries = [], [], [], {}
    nxt = 0
    for part in parts:
        src = zarr.open_group(part / PART_STORE, mode="r")
        attrs = dict(src.attrs)
        pm = read_csv(part / "manifest_part.csv") if (part / "manifest_part.csv").stat().st_size > 20 else []
        remap = {}
        for key, g in sorted(src.groups(), key=lambda kv: int(kv[0])):
            remap[int(key)] = nxt
            link_tree(part / PART_STORE / key, store / str(nxt))
            index.append(dict(bucket=nxt, dataset=g.attrs["dataset"], shard=part.name, n_samples=g["data"].shape[0],
                              n_sessions=len(g.attrs["sessions"]), n_chans=g["data"].shape[2]))
            corr += [dict(dataset=g.attrs["dataset"], duration=s["duration"], file=s["file"]) for s in g.attrs["sessions"]]
            nxt += 1
        for r in pm:
            r["big_recording_index"] = remap[int(r["big_recording_index"])]
        manifest += pm
        summaries[part.name] = json.loads((part / "summary.json").read_text())

    datasets = {r["dataset"] for r in index}
    bad = datasets & HELD_OUT
    if bad:
        shutil.rmtree(store)
        raise SystemExit(f"held-out dataset(s) present in the merged layout: {sorted(bad)} - discard this run")

    corpus = zarr.open_group(store, mode="r+")
    for row in index:                                  # provenance lives with the data, not only in a side file
        corpus[str(row["bucket"])].attrs.update(bucket=row["bucket"], source_shard=row["shard"])
    corpus.attrs.update({k: attrs[k] for k in ("format", "version", "sfreq", "window", "units", "band_pass_hz")},
                        datasets=sorted(datasets), n_buckets=nxt, n_samples=sum(r["n_samples"] for r in index),
                        buckets=index)
    side = out / name                                  # <name>-buckets.csv, <name>-manifest.csv, ...
    write_csv(Path(f"{side}-buckets.csv"), index)
    write_csv(Path(f"{side}-manifest.csv"), manifest)

    verify_zarr(store, index)
    report = build_report(index, corr, summaries, incomplete)
    report += f"\nZarr corpus: {nxt} buckets, {sum(r['n_samples'] for r in index):,} training samples of {WINDOW} timesteps.\n"
    Path(f"{side}-CONVERSION_REPORT.md").write_text(report)
    print(report)
    if results is not None:                            # copies, never links: $OUTROOT must not alias the store's files
        results.mkdir(parents=True, exist_ok=True)
        for f in side_files(store).values():
            shutil.copy2(f, results / f.name)
        print(f"side files copied to {results}")
    return dict(format="zarr", big_recordings=nxt, sessions=len(corr), samples=sum(r["n_samples"] for r in index),
                hours=round(sum(int(r["duration"]) for r in corr) / SFREQ / 3600, 1), datasets=sorted(datasets),
                incomplete=incomplete)


def verify_zarr(store: Path, index: list[dict]) -> None:
    """The merged store re-opens, holds exactly the indexed buckets, and every bucket passes the part checks."""
    import zarr

    root = zarr.open_group(store, mode="r")
    names = sorted(int(k) for k, _ in root.groups())
    assert names == list(range(len(index))), f"buckets {names[:5]}... do not match the index of {len(index)}"
    for row in index:
        g = root[str(row["bucket"])]
        assert g["data"].shape == (row["n_samples"], WINDOW, row["n_chans"]), f"bucket {row['bucket']}: shape changed"
        assert g["session"].shape[0] == row["n_samples"] and g["stats"].shape[0] == row["n_sessions"]
    problems = zarr_problems(store)
    assert not problems, problems
    print(f"verify OK: {len(index)} buckets, {sum(r['n_samples'] for r in index):,} samples")


def list_parts(parts_dir: Path, shards_tsv: Path | None) -> list[Path]:
    if shards_tsv:
        return [parts_dir / r["shard_id"] for r in read_csv(shards_tsv, delimiter="\t")]
    return sorted(p for p in parts_dir.iterdir() if p.is_dir())


def merge(parts_dir: Path, out: Path, shards_tsv: Path | None = None, allow_missing: bool = False,
          name: str | None = None, results: Path | None = None) -> dict:
    parts = list_parts(parts_dir, shards_tsv)
    incomplete = [p.name for p in parts if not (p / "_DONE").exists()]
    if incomplete and not allow_missing:
        raise SystemExit(f"{len(incomplete)} part(s) without _DONE (rerun their array tasks): {incomplete[:10]}")
    parts = [p for p in parts if (p / "_DONE").exists()]
    kinds = {is_zarr_part(p) for p in parts}
    if len(kinds) > 1:
        raise SystemExit("parts mix the memmap and Zarr formats; convert them all with the same --format")
    if kinds == {True}:
        if shards_tsv is None:  # parts_zarr/ holds the parts of every corpus version; a default merge would mix them
            raise SystemExit("a Zarr corpus is defined by its shard list: pass --shards (eegdash-corpus-v0: the 66 "
                             "datasets' shards; eegdash-corpus-v1: the same plus the datasets v1 adds)")
        if not name:            # a store is published and trained on by its name; no default could be right
            raise SystemExit("a Zarr store must be named: pass --name (e.g. eegdash-corpus-v0, eegdash-corpus-v1)")
        return merge_zarr(parts, out, incomplete, name, results)
    if results is not None:
        raise SystemExit("--results copies a Zarr store's side files; a memmap merge writes its own into --out")
    damaged = [p.name for p in parts if part_problems(p)]
    if damaged:
        for p in parts:
            for msg in part_problems(p):
                print(f"DAMAGED {p.name}: {msg}")
        raise SystemExit(f"{len(damaged)} damaged part(s) - delete them so a worker redoes them: {' '.join(damaged)}")

    for d in ("recordings", "positions", "stats", "csv_recordings"):
        (out / d).mkdir(parents=True, exist_ok=True)
    big, corr, stats, manifest, summaries = [], [], [], [], {}
    nxt = 0
    for part in parts:
        csvd = part / "csv_recordings"
        pb, pc, ps = read_csv(csvd / "df_big.csv"), read_csv(csvd / "df_corrected.csv"), read_csv(csvd / "df_stats_tmp.csv")
        pm = read_csv(part / "manifest_part.csv") if (part / "manifest_part.csv").stat().st_size > 20 else []
        remap = {}
        for r in pb:
            old = int(r["big_recording_index"])
            remap[old] = nxt
            for sub, pat in (("recordings", "recording_-_eeg_-_{}.npy"), ("positions", "recording_-_positions_-_{}.npy"),
                             ("stats", "recording_-_stats_-_{}.npy")):
                link(part / sub / pat.format(old), out / sub / pat.format(nxt))
            r["big_recording_index"] = nxt
            nxt += 1
        for rows in (pc, ps, pm):
            for r in rows:
                r["big_recording_index"] = remap[int(r["big_recording_index"])]
        big += pb
        corr += pc
        stats += ps
        manifest += pm
        summaries[part.name] = json.loads((part / "summary.json").read_text())

    datasets = {r["dataset"] for r in corr} | {r["config"] for r in big}
    bad = datasets & HELD_OUT
    if bad:
        raise SystemExit(f"held-out dataset(s) present in the merged layout: {sorted(bad)} - discard this run")

    csvd = out / "csv_recordings"
    write_csv(csvd / "df_big.csv", big, ["big_recording_index", "duration", "n_chans", "chans_names", "config"])
    write_csv(csvd / "df_corrected.csv", corr)
    write_csv(csvd / "df_stats_tmp.csv", stats, ["big_recording_index", "n_sessions", "n_chans"])
    shutil.copy(csvd / "df_stats_tmp.csv", csvd / "df_stats.csv")
    write_csv(out / "manifest.csv", manifest)

    verify(out, big, corr, stats)
    report = build_report(big, corr, summaries, incomplete)
    (out / "CONVERSION_REPORT.md").write_text(report)
    print(report)
    return dict(big_recordings=nxt, sessions=len(corr), hours=round(sum(int(b["duration"]) for b in big) / SFREQ / 3600, 1),
                datasets=sorted(datasets), incomplete=incomplete)


def verify(out: Path, big, corr, stats) -> None:
    by_idx = {int(s["big_recording_index"]): s for s in stats}
    for b in big:
        i, T, C = int(b["big_recording_index"]), int(b["duration"]), int(b["n_chans"])
        rec = (out / "recordings" / f"recording_-_eeg_-_{i}.npy").stat().st_size
        assert rec == T * C * 4, f"big recording {i}: {rec} bytes != {T}*{C}*4"
        st = (out / "stats" / f"recording_-_stats_-_{i}.npy").stat().st_size
        assert st == int(by_idx[i]["n_sessions"]) * 2 * C * 4, f"big recording {i}: stats size mismatch"
        assert (out / "positions" / f"recording_-_positions_-_{i}.npy").exists()
    sess_per_big = Counter(int(r["big_recording_index"]) for r in corr)
    for i, s in by_idx.items():
        assert sess_per_big[i] == int(s["n_sessions"]), f"big recording {i}: {sess_per_big[i]} sessions vs stats {s['n_sessions']}"
    for r in corr:
        assert int(r["duration"]) > WINDOW, f"session {r['file']} duration {r['duration']} <= {WINDOW}"
    print(f"verify OK: {len(big)} big recordings, {len(corr)} sessions")


def build_report(big, corr, summaries, incomplete) -> str:
    hours = defaultdict(float)
    sessions = Counter()
    for r in corr:
        hours[r["dataset"]] += int(r["duration"]) / SFREQ / 3600
        sessions[r["dataset"]] += 1
    skipped = Counter()
    dropped = Counter()
    for s in summaries.values():
        skipped.update(s.get("skipped_by_reason", {}))
        dropped.update(s.get("channels_dropped_no_position", {}))
    lines = ["# Conversion report", "",
             f"{len(big)} big recordings, {len(corr)} sessions, **{sum(hours.values()):.1f} h** of 200 Hz EEG from "
             f"{len(hours)} datasets, {len(summaries)} shards.", ""]
    if incomplete:
        lines += [f"**{len(incomplete)} shard(s) not merged (no _DONE):** {' '.join(incomplete)}", ""]
    lines += ["| dataset | sessions | hours |", "|---|---:|---:|"]
    lines += [f"| {d} | {sessions[d]} | {hours[d]:.1f} |" for d in sorted(hours, key=lambda d: -hours[d])]
    lines += ["", "Skipped recordings by reason: " + (", ".join(f"{k} {v}" for k, v in skipped.most_common()) or "none"), "",
              "Channels dropped for lacking a position (top 20): " +
              (", ".join(f"{k} {v}" for k, v in dropped.most_common(20)) or "none"), ""]
    return "\n".join(lines)



# ---------------------------------------------------------------- relabel

def side_files(store: Path) -> dict[str, Path]:
    """The files merge_zarr writes beside <out>/<name>.zarr."""
    side = store.parent / store.name.removesuffix(".zarr")
    return {k: Path(f"{side}-{k}") for k in ("buckets.csv", "manifest.csv", "CONVERSION_REPORT.md")}


def renamed_shard(shard: str, old: str, new: str) -> str:
    """i-care_046 -> nm000289_046; any other name unchanged."""
    return new + shard[len(old):] if shard.startswith(f"{old}_") else shard


def write_atomic(path: Path, text: str) -> None:
    """Write beside, then rename over: never writes through a hard link (a store's zarr.json copied from a part)."""
    tmp = path.with_name(f".{path.name}.relabel-tmp")
    tmp.write_text(text, newline="")
    os.replace(tmp, path)


def read_rows(path: Path, delimiter: str = ",") -> tuple[list[list[str]], str, str]:
    """(rows as lists, header first; the line ending; the text), so a rewrite changes only the relabelled cells."""
    with open(path, newline="") as f:                 # keep \r\n: csv.writer's default ending, as merge wrote it
        text = f.read()
    ending = "\r\n" if text.split("\n", 1)[0].endswith("\r") else "\n"
    return list(csv.reader(text.splitlines(keepends=True), delimiter=delimiter)), ending, text


def rows_text(rows: list[list[str]], ending: str, delimiter: str = ",") -> str:
    buf = io.StringIO(newline="")
    csv.writer(buf, delimiter=delimiter, lineterminator=ending).writerows(rows)
    return buf.getvalue()


def relabel_csv(path: Path, old: str, new: str, id_col: str, shard_col: str | None = None,
                delimiter: str = ",") -> tuple[str | None, int]:
    """(new text, or None when nothing changes; rows whose id or shard changed). Refuses a file where both ids label
    rows. A shard list (shards.tsv) is the same call with id_col "dataset", shard_col "shard_id", delimiter tab."""
    rows, ending, before = read_rows(path, delimiter)
    head = rows[0]
    i, j = head.index(id_col), (head.index(shard_col) if shard_col else None)
    ids = {r[i] for r in rows[1:] if r}
    if old in ids and new in ids:
        raise SystemExit(f"{path.name}: rows labelled both {old} and {new}; refusing to merge two datasets")
    n = 0
    for r in rows[1:]:
        if not r:
            continue
        was = list(r)
        if r[i] == old:
            r[i] = new
        if j is not None:
            r[j] = renamed_shard(r[j], old, new)
        n += r != was
    text = rows_text(rows, ending, delimiter)
    return (text if text != before else None), n


def id_pattern(old: str) -> re.Pattern:
    """The id as a word, alone or as a shard prefix (i-care, i-care_046), never inside another id."""
    return re.compile(rf"(?<![\w.-]){re.escape(old)}(?=_\d|[^\w-]|$)", re.M)


def group_meta(path: Path) -> dict:
    """A Zarr v3 node's zarr.json; a v2 store (.zattrs) is refused rather than half-handled."""
    if not path.is_file():
        raise SystemExit(f"{path} missing: not a Zarr v3 store (relabel edits zarr.json only)")
    return json.loads(path.read_text())


def meta_text(meta: dict) -> str:
    return json.dumps(meta, indent=2, allow_nan=True)  # as zarr 3 writes it


def relabel_part(part: Path, old: str, new: str) -> dict[Path, str]:
    """The rewrites inside one part folder (the files, not its name). The 2026-10-18 relabel only renamed the folders,
    so the next merge read `i-care` back from their contents (found 2026-10-24)."""
    writes: dict[Path, str] = {}
    store = part / PART_STORE
    if store.is_dir():
        root = group_meta(store / "zarr.json")
        if root.get("attributes", {}).get("dataset") == old:
            root["attributes"]["dataset"] = new
            writes[store / "zarr.json"] = meta_text(root)
        for key in sorted((p.name for p in store.iterdir() if p.is_dir() and p.name.isdigit()), key=int):
            path = store / key / "zarr.json"
            meta = group_meta(path)
            a, before = meta.get("attributes", {}), meta_text(meta)
            if a.get("dataset") == old:
                a["dataset"] = new
            if "source_shard" in a:
                a["source_shard"] = renamed_shard(a["source_shard"], old, new)
            if meta_text(meta) != before:
                writes[path] = meta_text(meta)
    if (part / "manifest_part.csv").is_file():
        text, _ = relabel_csv(part / "manifest_part.csv", old, new, "dataset_id")
        if text is not None:
            writes[part / "manifest_part.csv"] = text
    if (part / "summary.json").is_file():          # the converter's `dataset` key; file paths do not carry the id
        summ = json.loads((part / "summary.json").read_text())
        if summ.get("dataset") == old:
            summ["dataset"] = new
            writes[part / "summary.json"] = json.dumps(summ, indent=1, default=str)  # as eegdash_to_reve.py writes it
    return writes


def relabel(store: Path, old: str, new: str, dry_run: bool = False, parts: Path | None = None,
            plans: list[Path] = (), side_only: bool = False) -> dict:
    """Plan every change, refuse a clash, print the plan, then (unless dry_run) write it. Returns the counts."""
    if old == new:
        raise SystemExit("--from and --to are the same id")
    writes: dict[Path, str] = {}
    summary = {"groups": 0, "groups labelled": 0, "root": False, "buckets rows": 0, "manifest rows": 0,
               "report lines": 0, "plan rows": 0, "parts": [], "part files": 0}

    if not side_only:
        if not store.is_dir():
            raise SystemExit(f"{store}: no store (pass --side-only to rewrite the side files alone)")
        root = group_meta(store / "zarr.json")
        attrs = root.get("attributes", {})
        if root.get("consolidated_metadata"):
            raise SystemExit(f"{store}: consolidated metadata present; relabel would leave it stale")
        datasets = attrs.get("datasets", [])
        if old in datasets and new in datasets:
            raise SystemExit(f"{store}: {new} already labels other buckets; refusing to merge two datasets")
        for key in sorted((p.name for p in store.iterdir() if p.is_dir() and p.name.isdigit()), key=int):
            path = store / key / "zarr.json"
            meta = group_meta(path)
            a = meta.get("attributes", {})
            if a.get("dataset") == old:
                a["dataset"] = new
                if "source_shard" in a:
                    a["source_shard"] = renamed_shard(a["source_shard"], old, new)
                writes[path] = meta_text(meta)
                summary["groups"] += 1
            summary["groups labelled"] += a.get("dataset") == new
        if old in datasets or any(b.get("dataset") == old for b in attrs.get("buckets", [])):
            attrs["datasets"] = sorted({new if d == old else d for d in datasets})
            for b in attrs.get("buckets", []):
                if b.get("dataset") == old:
                    b["dataset"] = new
                b["shard"] = renamed_shard(b.get("shard", ""), old, new)
            writes[store / "zarr.json"] = meta_text(root)
            summary["root"] = True

    side = side_files(store)
    for name, id_col, shard_col, key in (("buckets.csv", "dataset", "shard", "buckets rows"),
                                         ("manifest.csv", "dataset_id", None, "manifest rows")):
        if side[name].is_file():
            text, summary[key] = relabel_csv(side[name], old, new, id_col, shard_col)
            if text is not None:
                writes[side[name]] = text
        else:
            print(f"   no {side[name].name}: skipped")
    if side["CONVERSION_REPORT.md"].is_file():
        report = side["CONVERSION_REPORT.md"].read_text()
        lines = sum(bool(id_pattern(old).search(l)) for l in report.splitlines())
        if lines:
            writes[side["CONVERSION_REPORT.md"]] = id_pattern(old).sub(new, report)
            summary["report lines"] = lines
    for plan in plans:
        text, n = relabel_csv(plan, old, new, "dataset", "shard_id", "\t")
        if text is not None:
            writes[plan] = text
            summary["plan rows"] += n
    if parts is not None:
        for p in sorted(parts.iterdir()):
            if not (p.is_dir() and (p.name.startswith(f"{old}_") or p.name.startswith(f"{new}_"))):
                continue
            inside = relabel_part(p, old, new)         # paths under the old name: written before the folder moves
            writes.update(inside)
            summary["part files"] += len(inside)
            if p.name.startswith(f"{old}_"):
                target = parts / renamed_shard(p.name, old, new)
                if target.exists():
                    raise SystemExit(f"{target} exists; refusing to rename {p.name} onto it")
                summary["parts"].append((p, target))

    store_tag = "side files of " if side_only else ""
    w = max(27, *(len(f.name) + 2 for f in side.values()))
    print(f"relabel {old} -> {new} in {store_tag}{store}{'  (dry run: nothing written)' if dry_run else ''}")
    if not side_only:
        print(f"   {'bucket groups changed':<{w}}{summary['groups']} (labelled {new} afterwards: {summary['groups labelled']})")
        print(f"   {'root attributes':<{w}}{'changed' if summary['root'] else 'unchanged'}")
    print(f"   {side['buckets.csv'].name:<{w}}{summary['buckets rows']} rows changed")
    print(f"   {side['manifest.csv'].name:<{w}}{summary['manifest rows']} rows changed")
    print(f"   {side['CONVERSION_REPORT.md'].name:<{w}}{summary['report lines']} lines changed")
    if plans:
        print(f"   {'shard lists':<{w}}{summary['plan rows']} rows changed in {len(plans)} file(s)")
    if parts is not None:
        print(f"   {'part folders renamed':<{w}}{len(summary['parts'])}")
        print(f"   {'part files rewritten':<{w}}{summary['part files']}")
    if not writes and not summary["parts"]:
        print("   nothing to change")
    if dry_run:
        return summary
    for path, text in writes.items():
        write_atomic(path, text)
    for src, dst in summary["parts"]:
        os.rename(src, dst)
    print(f"relabel done: {len(writes)} file(s) rewritten, {len(summary['parts'])} folder(s) renamed")
    return summary


LICENCE_URLS = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "GPL-3.0": "https://www.gnu.org/licenses/gpl-3.0.html",
}


def source_url(dataset: str) -> str:
    """The dataset's page: OpenNeuro for ds ids (the original even when EEGDash serves the NEMAR twin), NEMAR otherwise."""
    if re.fullmatch(r"ds\d+", dataset):
        return f"https://openneuro.org/datasets/{dataset}"
    return f"https://nemar.org/dataexplorer/detail?dataset_id={dataset}"


def licence(store: Path, table: Path, dry_run: bool = False) -> dict:
    """Plan the licence attributes of every bucket and the root, refuse a dataset without a known licence, then write."""
    licences = {r["dataset_id"]: r.get("licence", "").strip() for r in read_csv(table)}
    for store_id, ours in TABLE_IDS.items():           # a bucket labelled nm000289 takes the table's icare row
        if ours in licences:
            licences.setdefault(store_id, licences[ours])
    root = group_meta(store / "zarr.json")
    if root.get("consolidated_metadata"):
        raise SystemExit(f"{store}: consolidated metadata present; licence would leave it stale")
    keys = sorted((p.name for p in store.iterdir() if p.is_dir() and p.name.isdigit()), key=int)
    metas = {k: group_meta(store / k / "zarr.json") for k in keys}
    used = sorted({m.get("attributes", {}).get("dataset", "") for m in metas.values()})
    bad = [f"{d}: not in {table.name}" if d not in licences else f"{d}: no licence" if not licences[d]
           else f"{d}: licence {licences[d]} has no known URL" for d in used
           if d not in licences or licences[d] not in LICENCE_URLS]
    if bad:
        raise SystemExit(f"{store}: refusing, nothing written:\n   " + "\n   ".join(bad))
    writes: dict[Path, str] = {}
    for k, meta in metas.items():
        a = meta.setdefault("attributes", {})
        before = meta_text(meta)
        d = a["dataset"]
        a.update(licence=licences[d], licence_url=LICENCE_URLS[licences[d]], source=source_url(d))
        if meta_text(meta) != before:
            writes[store / k / "zarr.json"] = meta_text(meta)
    attrs = root.setdefault("attributes", {})
    before = meta_text(root)
    attrs.update(licence="other", licences={d: licences[d] for d in used}, licence_file="LICENSES.csv")
    if meta_text(root) != before:
        writes[store / "zarr.json"] = meta_text(root)
    per = Counter(licences[metas[k]["attributes"]["dataset"]] for k in keys)
    print(f"licence {store} from {table}{'  (dry run: nothing written)' if dry_run else ''}")
    print(f"   {len(used)} datasets, {len(keys)} buckets")
    for lic, n in sorted(per.items(), key=lambda x: -x[1]):
        print(f"   {lic:<18}{sum(licences[d] == lic for d in used):>4} datasets {n:>5} buckets")
    print(f"   {len(writes)} zarr.json file(s) to rewrite" if writes else "   nothing to change")
    if not dry_run:
        for path, text in writes.items():
            write_atomic(path, text)
        print(f"licence done: {len(writes)} file(s) rewritten")
    return {"datasets": len(used), "buckets": len(keys), "rewritten": 0 if dry_run else len(writes),
            "planned": len(writes), "per_licence": dict(per)}


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("merge", "relabel", "licence", "-h", "--help"):
        argv.insert(0, "merge")                        # the documented merge lines carry no subcommand
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="merge finished parts into one layout or Zarr store (the default)")
    m.add_argument("--parts", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--shards", default=None, help="shards.tsv; fixes the part order (default: sorted dirs)")
    m.add_argument("--allow-missing", action="store_true")
    m.add_argument("--name", default=None,
                   help="Zarr parts only, required there: corpus name -> <out>/<name>.zarr, e.g. eegdash-corpus-v1")
    m.add_argument("--results", default=None,
                   help="Zarr parts only: also copy the three side files here, e.g. $OUTROOT/conversion/zarr")
    r = sub.add_parser("relabel", help="rename a dataset id in a merged Zarr store and its side files")
    r.add_argument("--store", required=True, help="<out>/<name>.zarr; the side files beside it are found from the name")
    r.add_argument("--from", dest="old", required=True, help="the id the store records, e.g. i-care")
    r.add_argument("--to", dest="new", required=True, help="the new id, e.g. nm000289")
    r.add_argument("--parts", default=None, help="rename the part folders <from>_NNN here to <to>_NNN; rewrite the id "
                                                "inside them and inside parts already named <to>_NNN")
    r.add_argument("--plans", nargs="*", default=[], help="shard lists (TSV) naming those parts, rewritten to match")
    r.add_argument("--side-only", action="store_true", help="rewrite the side files only (no store needed)")
    r.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    lc = sub.add_parser("licence", help="write each dataset's licence into the bucket and root attributes of a Zarr store")
    lc.add_argument("--store", required=True, help="<out>/<name>.zarr")
    lc.add_argument("--table", required=True, help="the corpus table with dataset_id and licence, e.g. corpus/eegdash-corpus-v1.csv")
    lc.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    a = ap.parse_args(argv)
    if a.cmd == "licence":
        licence(Path(a.store), Path(a.table), a.dry_run)
        return
    if a.cmd == "relabel":
        relabel(Path(a.store), a.old, a.new, a.dry_run, Path(a.parts) if a.parts else None,
                [Path(p) for p in a.plans], a.side_only)
        return
    info = merge(Path(a.parts), Path(a.out), Path(a.shards) if a.shards else None, a.allow_missing, a.name,
                 Path(a.results) if a.results else None)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
