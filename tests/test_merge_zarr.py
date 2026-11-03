"""merge_parts.py on Zarr parts: one corpus store with the same guarantees as the memmap merge."""

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from eegdash_to_reve import WINDOW, Bank, GroupWriter, Prepared, ZarrGroupWriter
import merge_parts as mp
from merge_parts import main as merge_main, merge as merge_parts, relabel


def merge(parts, out, **kw):
    """merge_parts.merge over every part present, via the shard list and the name a Zarr merge requires."""
    shards = parts.parent / "shards.tsv"
    shards.write_text("shard_id\n" + "".join(f"{p.name}\n" for p in sorted(parts.iterdir()) if p.is_dir()))
    return merge_parts(parts, out, shards, **{"name": "eegdash-corpus-v0", **kw})

META = dict(subject="01", session="", run="1", task="x")
NAMES = ["C3", "C4", "Cz", "Fz", "Pz"]


def out_dir(tmp_path):
    return tmp_path / "zarr"


def out_store(tmp_path):
    return out_dir(tmp_path) / "eegdash-corpus-v0.zarr"


@pytest.fixture
def bank():
    return Bank(NAMES, np.arange(15, dtype=np.float32).reshape(5, 3) / 100)


def prep(names, T, seed):
    rng = np.random.default_rng(seed)
    return Prepared(rng.normal(0, 20, (T, len(names))).astype("<f4"), sorted(names), [], 500.0)


def make_part(parts, name, dataset, bank, sessions, writer=ZarrGroupWriter):
    """A finished part: sessions is a list of (channels, timesteps)."""
    w = writer(parts / name, dataset, bank)
    for k, (chans, T) in enumerate(sessions):
        w.append(prep(chans, T, seed=hash((name, k)) % 2**32), f"{dataset}/sub-01/r{k}", META)
    w.finalize({"dataset": dataset})                   # convert_shard's summary starts with the id
    return parts / name


@pytest.fixture
def two_parts(tmp_path, bank):
    parts = tmp_path / "parts"
    make_part(parts, "dsA_000", "dsA", bank, [(["C3", "C4"], 5000), (["Cz", "Fz", "Pz"], 4500), (["C3", "C4"], 6001)])
    make_part(parts, "dsB_000", "dsB", bank, [(["C3", "Cz"], 9000)])
    return parts


def open_part(parts, name, bucket):
    return zarr.open_group(parts / name / "samples.zarr", mode="r")[str(bucket)]


def test_buckets_renumbered_in_shard_order_and_data_unchanged(tmp_path, two_parts):
    info = merge(two_parts, out_dir(tmp_path))
    root = zarr.open_group(out_dir(tmp_path) / "eegdash-corpus-v0.zarr", mode="r")
    sources = [("dsA_000", 0), ("dsA_000", 1), ("dsB_000", 0)]
    assert sorted(int(k) for k, _ in root.groups()) == [0, 1, 2]
    for new, (shard, old) in enumerate(sources):
        g, src = root[str(new)], open_part(two_parts, shard, old)
        for key in ("data", "session", "stats", "positions"):
            assert np.array_equal(g[key][:], src[key][:]), (new, key)
        assert g.attrs["source_shard"] == shard and g.attrs["bucket"] == new
        assert g.attrs["sessions"] == src.attrs["sessions"]
    assert info["format"] == "zarr" and info["big_recordings"] == 3 and info["datasets"] == ["dsA", "dsB"]
    assert root.attrs["n_samples"] == info["samples"] == sum(root[str(i)]["data"].shape[0] for i in range(3))
    assert root.attrs["window"] == WINDOW and root.attrs["format"] == "reve-samples"


def test_chunks_are_hard_linked_and_parts_left_untouched(tmp_path, two_parts):
    """Merging must not copy the corpus (tens of TB on Lustre) nor edit the shards it came from."""
    before = (two_parts / "dsB_000/samples.zarr/0/zarr.json").read_bytes()
    merge(two_parts, out_dir(tmp_path))
    chunk_files = [os.path.join(d, f) for d, _, fs in os.walk(out_dir(tmp_path) / "eegdash-corpus-v0.zarr/2/data") for f in fs
                   if f != "zarr.json"]
    assert chunk_files and all(os.stat(f).st_nlink == 2 for f in chunk_files)
    assert os.stat(out_dir(tmp_path) / "eegdash-corpus-v0.zarr/2/zarr.json").st_nlink == 1
    assert (two_parts / "dsB_000/samples.zarr/0/zarr.json").read_bytes() == before


def test_side_files_and_report(tmp_path, two_parts):
    merge(two_parts, out_dir(tmp_path))
    with open(out_dir(tmp_path) / "eegdash-corpus-v0-buckets.csv") as f:
        buckets = list(csv.DictReader(f))
    assert [(b["bucket"], b["dataset"], b["shard"]) for b in buckets] == [
        ("0", "dsA", "dsA_000"), ("1", "dsA", "dsA_000"), ("2", "dsB", "dsB_000")]
    with open(out_dir(tmp_path) / "eegdash-corpus-v0-manifest.csv") as f:
        manifest = list(csv.DictReader(f))
    assert [m["big_recording_index"] for m in manifest] == ["0", "1", "0", "2"]
    report = (out_dir(tmp_path) / "eegdash-corpus-v0-CONVERSION_REPORT.md").read_text()
    assert "3 big recordings, 4 sessions" in report and "Zarr corpus: 3 buckets" in report


def test_rerun_replaces_the_store(tmp_path, two_parts):
    merge(two_parts, out_dir(tmp_path))
    merge(two_parts, out_dir(tmp_path))
    assert len(list(zarr.open_group(out_dir(tmp_path) / "eegdash-corpus-v0.zarr", mode="r").groups())) == 3


def test_refuses_held_out_dataset(tmp_path, two_parts, bank):
    make_part(two_parts, "ds004362_000", "ds004362", bank, [(["C3", "C4"], 5000)])
    with pytest.raises(SystemExit, match="held-out"):
        merge(two_parts, out_dir(tmp_path))
    assert not (out_dir(tmp_path) / "eegdash-corpus-v0.zarr").exists()


def test_refuses_damaged_part(tmp_path, two_parts):
    """A session record that disagrees with the stored samples is what two concurrent writers leave behind."""
    g = zarr.open_group(two_parts / "dsB_000/samples.zarr", mode="r+")["0"]
    sessions = g.attrs["sessions"]
    sessions[0]["n_samples"] += 1
    g.attrs["sessions"] = sessions
    with pytest.raises(SystemExit, match="damaged"):
        merge(two_parts, out_dir(tmp_path))


def test_refuses_incomplete_part_unless_allowed(tmp_path, two_parts):
    (two_parts / "dsB_000/_DONE").unlink()
    with pytest.raises(SystemExit, match="without _DONE"):
        merge(two_parts, out_dir(tmp_path))
    info = merge(two_parts, out_dir(tmp_path), allow_missing=True)
    assert info["incomplete"] == ["dsB_000"] and info["big_recordings"] == 2


def test_refuses_mixed_formats(tmp_path, two_parts, bank):
    make_part(two_parts, "dsC_000", "dsC", bank, [(["C3", "C4"], 5000)], writer=GroupWriter)
    with pytest.raises(SystemExit, match="mix the memmap and Zarr"):
        merge(two_parts, out_dir(tmp_path))


def test_summary_counts_match_the_parts(tmp_path, two_parts):
    info = merge(two_parts, out_dir(tmp_path))
    parts = [json.loads((two_parts / p / "summary.json").read_text()) for p in ("dsA_000", "dsB_000")]
    assert info["samples"] == sum(p["n_samples"] for p in parts)
    assert info["sessions"] == sum(p["n_sessions"] for p in parts) == 4


def test_store_holds_only_zarr_and_side_files_sit_beside_it(tmp_path, two_parts):
    """Zarr warns about any foreign file in a store, so the manifest and report go beside it, named after it."""
    merge(two_parts, out_dir(tmp_path))
    assert {p.name for p in out_store(tmp_path).iterdir()} == {"zarr.json", "0", "1", "2"}
    assert {p.name for p in out_dir(tmp_path).iterdir()} == {"eegdash-corpus-v0.zarr",
        "eegdash-corpus-v0-buckets.csv", "eegdash-corpus-v0-manifest.csv", "eegdash-corpus-v0-CONVERSION_REPORT.md"}


def test_two_corpora_share_the_folder(tmp_path, two_parts):
    """Two corpus versions live side by side; merging one leaves the other alone."""
    merge(two_parts, out_dir(tmp_path))
    merge(two_parts, out_dir(tmp_path), name="eegdash-corpus-v1")
    names = {p.name for p in out_dir(tmp_path).iterdir()}
    assert {"eegdash-corpus-v0.zarr", "eegdash-corpus-v1.zarr", "eegdash-corpus-v0-manifest.csv", "eegdash-corpus-v1-manifest.csv"} <= names
    assert len(list(zarr.open_group(out_store(tmp_path), mode="r").groups())) == 3


def test_zarr_merge_requires_a_name(tmp_path, two_parts):
    """A store is published and trained on by its name: no default, and nothing written without one."""
    shards = tmp_path / "shards.tsv"
    shards.write_text("shard_id\ndsA_000\ndsB_000\n")
    with pytest.raises(SystemExit, match="pass --name"):
        merge_parts(two_parts, out_dir(tmp_path), shards)
    with pytest.raises(SystemExit, match="pass --name"):
        merge_main(["--parts", str(two_parts), "--out", str(out_dir(tmp_path)), "--shards", str(shards)])
    assert not out_dir(tmp_path).exists()


def test_results_receives_copies_of_the_side_files(tmp_path, two_parts):
    """--results puts the manifest, bucket index and report where results_sync finds them; copies, not links."""
    shards = tmp_path / "shards.tsv"
    shards.write_text("shard_id\ndsA_000\ndsB_000\n")
    results = tmp_path / "outroot/conversion/zarr"
    merge_main(["--parts", str(two_parts), "--out", str(out_dir(tmp_path)), "--shards", str(shards),
                "--name", "eegdash-corpus-v1", "--results", str(results)])
    names = ["eegdash-corpus-v1-manifest.csv", "eegdash-corpus-v1-buckets.csv", "eegdash-corpus-v1-CONVERSION_REPORT.md"]
    assert sorted(p.name for p in results.iterdir()) == sorted(names)
    for n in names:
        assert (results / n).read_bytes() == (out_dir(tmp_path) / n).read_bytes() and os.stat(results / n).st_nlink == 1
    assert not (results / "eegdash-corpus-v1.zarr").exists()


def test_results_is_refused_for_a_memmap_merge(tmp_path, bank):
    parts = tmp_path / "parts_memmap"
    make_part(parts, "dsC_000", "dsC", bank, [(["C3", "C4"], 5000)], writer=GroupWriter)
    with pytest.raises(SystemExit, match="--results"):
        merge_parts(parts, tmp_path / "memmap", results=tmp_path / "r")
    assert not (tmp_path / "r").exists()


def test_zarr_merge_requires_a_shard_list(tmp_path, two_parts):
    """parts_zarr/ holds the parts of every corpus version: merging whatever is there would mix them."""
    with pytest.raises(SystemExit, match="pass --shards"):
        merge_parts(two_parts, out_dir(tmp_path))


def test_shard_list_selects_the_corpus(tmp_path, two_parts):
    """v0 from a subset of the parts, v1 from all of them, side by side, no reconversion."""
    one = tmp_path / "public.tsv"
    one.write_text("shard_id\ndsA_000\n")
    info = merge_parts(two_parts, out_dir(tmp_path), one, name="eegdash-corpus-v0")
    assert info["datasets"] == ["dsA"] and info["big_recordings"] == 2
    info_nc = merge(two_parts, out_dir(tmp_path), name="eegdash-corpus-v1")
    assert info_nc["datasets"] == ["dsA", "dsB"]
    assert info["samples"] < info_nc["samples"]


# ---- relabel: a dataset id renamed in the merged store and its side files, arrays untouched ----------------------



@pytest.fixture
def nc(tmp_path, bank):
    """eegdash-corpus-v0 from an I-CARE part (two buckets) and another dataset's part, with its shard list."""
    parts = tmp_path / "parts"
    make_part(parts, "dsB_000", "dsB", bank, [(["C3", "Cz"], 9000)])
    make_part(parts, "i-care_000", "i-care", bank, [(["C3", "C4"], 5000), (["Cz", "Fz", "Pz"], 4500)])
    plan = tmp_path / "nc_shards.tsv"
    plan.write_text("shard_id\tdataset\tmode\ndsB_000\tdsB\toffline\ni-care_000\ti-care\toffline\n")
    merge(parts, out_dir(tmp_path), name="eegdash-corpus-v0")
    return out_dir(tmp_path) / "eegdash-corpus-v0.zarr", parts, plan


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_relabel_rewrites_attributes_and_side_files_and_leaves_arrays_alone(tmp_path, nc):
    store, parts, _ = nc
    before = snapshot(store)
    info = relabel(store, "i-care", "nm000289")
    root = zarr.open_group(store, mode="r")
    labels = {k: (g.attrs["dataset"], g.attrs["source_shard"]) for k, g in root.groups()}
    assert labels == {"0": ("dsB", "dsB_000"), "1": ("nm000289", "nm000289_000"), "2": ("nm000289", "nm000289_000")}
    assert root.attrs["datasets"] == ["dsB", "nm000289"]
    assert [(b["dataset"], b["shard"]) for b in root.attrs["buckets"]] == [("dsB", "dsB_000"), ("nm000289", "nm000289_000"),
                                                                           ("nm000289", "nm000289_000")]
    after = snapshot(store)
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"zarr.json", "1/zarr.json", "2/zarr.json"} and set(after) == set(before)
    assert np.array_equal(root["1"]["data"][:], open_part(parts, "i-care_000", 0)["data"][:])
    side = out_dir(tmp_path) / "eegdash-corpus-v0"
    assert [(b["dataset"], b["shard"]) for b in rows(f"{side}-buckets.csv")] == [
        ("dsB", "dsB_000"), ("nm000289", "nm000289_000"), ("nm000289", "nm000289_000")]
    assert [m["dataset_id"] for m in rows(f"{side}-manifest.csv")] == ["dsB", "nm000289", "nm000289"]
    report = Path(f"{side}-CONVERSION_REPORT.md").read_text()
    assert "| nm000289 |" in report and "i-care" not in report
    assert info["groups"] == 2 and info["buckets rows"] == 2 and info["manifest rows"] == 2


def test_relabel_never_writes_through_a_hard_link_into_the_part(tmp_path, nc):
    """The store's zarr.json files are copies, but even a linked one must be replaced, not written in place."""
    store, parts, _ = nc
    part_meta = parts / "i-care_000/samples.zarr/0/zarr.json"
    os.unlink(store / "1/zarr.json")
    os.link(part_meta, store / "1/zarr.json")
    before = part_meta.read_bytes()
    relabel(store, "i-care", "nm000289")
    assert part_meta.read_bytes() == before and json.loads((store / "1/zarr.json").read_text())["attributes"]["dataset"] == "nm000289"


def test_relabel_is_idempotent(tmp_path, nc, capsys):
    store, *_ = nc
    relabel(store, "i-care", "nm000289")
    before = snapshot(out_dir(tmp_path))
    info = relabel(store, "i-care", "nm000289")
    assert snapshot(out_dir(tmp_path)) == before
    assert info["groups"] == 0 and info["buckets rows"] == 0 and info["manifest rows"] == 0 and not info["root"]
    assert "nothing to change" in capsys.readouterr().out


def test_relabel_dry_run_changes_nothing(tmp_path, nc, capsys):
    store, parts, plan = nc
    before, before_parts, before_plan = snapshot(out_dir(tmp_path)), sorted(p.name for p in parts.iterdir()), plan.read_bytes()
    info = relabel(store, "i-care", "nm000289", dry_run=True, parts=parts, plans=[plan])
    assert snapshot(out_dir(tmp_path)) == before and sorted(p.name for p in parts.iterdir()) == before_parts
    assert plan.read_bytes() == before_plan
    assert info["groups"] == 2 and "dry run" in capsys.readouterr().out


def test_relabel_refuses_a_target_that_labels_other_buckets(tmp_path, nc, bank):
    store, parts, _ = nc
    make_part(parts, "nm000289_000", "nm000289", bank, [(["C3", "C4"], 5000)])
    merge(parts, out_dir(tmp_path), name="eegdash-corpus-v0")
    before = snapshot(out_dir(tmp_path))
    with pytest.raises(SystemExit, match="already labels other buckets"):
        relabel(store, "i-care", "nm000289")
    assert snapshot(out_dir(tmp_path)) == before


def test_relabel_renames_part_folders_and_their_shard_list(tmp_path, nc):
    store, parts, plan = nc
    other = snapshot(parts / "dsB_000")
    relabel(store, "i-care", "nm000289", parts=parts, plans=[plan])
    assert sorted(p.name for p in parts.iterdir()) == ["dsB_000", "nm000289_000"]
    assert snapshot(parts / "dsB_000") == other                              # another dataset's part untouched
    assert plan.read_text() == "shard_id\tdataset\tmode\ndsB_000\tdsB\toffline\nnm000289_000\tnm000289\toffline\n"


def part_labels(part):
    """Every place inside a part folder that records its dataset id."""
    root = zarr.open_group(part / "samples.zarr", mode="r")
    return {"root": root.attrs["dataset"], "buckets": sorted({g.attrs["dataset"] for _, g in root.groups()}),
            "manifest": sorted({r["dataset_id"] for r in rows(part / "manifest_part.csv")}),
            "summary": json.loads((part / "summary.json").read_text())["dataset"]}


NEW_LABELS = {"root": "nm000289", "buckets": ["nm000289"], "manifest": ["nm000289"], "summary": "nm000289"}


def test_relabel_rewrites_inside_a_part_it_renames_and_a_second_run_changes_nothing(tmp_path, nc, capsys):
    store, parts, plan = nc
    data = open_part(parts, "i-care_000", 0)["data"][:]
    info = relabel(store, "i-care", "nm000289", parts=parts, plans=[plan])
    part = parts / "nm000289_000"
    assert part_labels(part) == NEW_LABELS and info["part files"] == 5           # root, 2 buckets, manifest, summary
    assert np.array_equal(open_part(parts, "nm000289_000", 0)["data"][:], data)
    assert "part files rewritten" in capsys.readouterr().out
    before = snapshot(tmp_path)
    info = relabel(store, "i-care", "nm000289", parts=parts, plans=[plan])
    assert snapshot(tmp_path) == before and info["part files"] == 0
    assert "nothing to change" in capsys.readouterr().out


def test_relabel_rewrites_a_part_already_renamed_without_its_contents(tmp_path, nc, capsys):
    """The 2026-10-18 relabel renamed i-care_NNN to nm000289_NNN and left `i-care` inside."""
    store, parts, plan = nc
    relabel(store, "i-care", "nm000289", plans=[plan])                         # store and plan relabelled, parts not
    os.rename(parts / "i-care_000", parts / "nm000289_000")
    assert part_labels(parts / "nm000289_000")["summary"] == "i-care"
    capsys.readouterr()
    relabel(store, "i-care", "nm000289", parts=parts, plans=[plan])
    assert part_labels(parts / "nm000289_000") == NEW_LABELS
    assert sorted(p.name for p in parts.iterdir()) == ["dsB_000", "nm000289_000"]
    assert "nothing to change" not in capsys.readouterr().out
    relabel(store, "i-care", "nm000289", parts=parts, plans=[plan])
    assert "nothing to change" in capsys.readouterr().out


def test_relabel_dry_run_leaves_the_inside_of_parts_alone(tmp_path, nc, capsys):
    store, parts, plan = nc
    relabel(store, "i-care", "nm000289", plans=[plan])
    os.rename(parts / "i-care_000", parts / "nm000289_000")
    before = snapshot(tmp_path)
    capsys.readouterr()
    info = relabel(store, "i-care", "nm000289", dry_run=True, parts=parts, plans=[plan])
    out = capsys.readouterr().out
    assert snapshot(tmp_path) == before and info["part files"] == 5
    assert re.search(r"part files rewritten +5\n", out) and "nothing to change" not in out


def test_a_merge_after_the_relabel_keeps_the_new_id(tmp_path, nc):
    """Found 2026-10-24: re-merged from renamed but unrewritten parts, both stores came back labelled i-care."""
    store, parts, plan = nc
    relabel(store, "i-care", "nm000289", parts=parts, plans=[plan])
    merge_parts(parts, out_dir(tmp_path), plan, name="eegdash-corpus-v0")
    root = zarr.open_group(store, mode="r")
    assert root.attrs["datasets"] == ["dsB", "nm000289"]
    assert {g.attrs["dataset"] for _, g in root.groups()} == {"dsB", "nm000289"}
    assert {g.attrs["source_shard"] for _, g in root.groups()} == {"dsB_000", "nm000289_000"}
    side = out_dir(tmp_path) / "eegdash-corpus-v0"
    assert {m["dataset_id"] for m in rows(f"{side}-manifest.csv")} == {"dsB", "nm000289"}
    assert "i-care" not in Path(f"{side}-buckets.csv").read_text() + Path(f"{side}-CONVERSION_REPORT.md").read_text()


def test_relabel_side_only_rewrites_copies_without_a_store(tmp_path, nc):
    """The repository keeps copies of the side files, not the store."""
    copies = tmp_path / "copies"
    copies.mkdir()
    for f in out_dir(tmp_path).glob("eegdash-corpus-v0-*"):
        (copies / f.name).write_bytes(f.read_bytes())
    with pytest.raises(SystemExit, match="no store"):
        relabel(copies / "eegdash-corpus-v0.zarr", "i-care", "nm000289")
    merge_main(["relabel", "--store", str(copies / "eegdash-corpus-v0.zarr"), "--from", "i-care", "--to", "nm000289", "--side-only"])
    assert [m["dataset_id"] for m in rows(copies / "eegdash-corpus-v0-manifest.csv")] == ["dsB", "nm000289", "nm000289"]
    original = (out_dir(tmp_path) / "eegdash-corpus-v0-manifest.csv").read_bytes().replace(b"\ni-care,", b"\nnm000289,")
    assert (copies / "eegdash-corpus-v0-manifest.csv").read_bytes() == original   # only the id cells change, line ends kept


def test_the_merge_line_without_a_subcommand_still_works(tmp_path, two_parts):
    shards = tmp_path / "shards.tsv"
    shards.write_text("shard_id\ndsA_000\ndsB_000\n")
    merge_main(["--parts", str(two_parts), "--out", str(out_dir(tmp_path)), "--shards", str(shards), "--name", "eegdash-corpus-v0"])
    assert out_store(tmp_path).is_dir()


def test_write_csv_takes_the_union_of_columns_so_old_and_new_parts_merge(tmp_path):
    """Parts converted before n_positions_aligned existed (2026-10-18) merge with parts converted after it."""
    out = tmp_path / "m.csv"
    mp.write_csv(out, [{"a": "1", "b": "2"}, {"a": "3", "b": "4", "n_positions_aligned": "5"}])
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == ["a", "b", "n_positions_aligned"] and rows[0]["n_positions_aligned"] == "" and rows[1]["n_positions_aligned"] == "5"


# ---- licence: each dataset's licence written into its buckets and the root, arrays untouched ----------------------

REPO = Path(__file__).resolve().parents[1]


def licence_table(tmp_path, **licences):
    t = tmp_path / "table.csv"
    t.write_text("dataset_id,licence\n" + "".join(f"{d},{lic}\n" for d, lic in licences.items()))
    return t


def test_licence_writes_bucket_and_root_attributes_and_leaves_arrays_alone(tmp_path, nc):
    store, parts, _ = nc
    before = snapshot(store)
    info = mp.licence(store, licence_table(tmp_path, dsB="CC0-1.0", **{"i-care": "CC-BY-NC-SA-4.0"}))
    root = zarr.open_group(store, mode="r")
    got = {k: (g.attrs["dataset"], g.attrs["licence"], g.attrs["source"]) for k, g in root.groups()}
    assert got == {"0": ("dsB", "CC0-1.0", "https://nemar.org/dataexplorer/detail?dataset_id=dsB"),
                   "1": ("i-care", "CC-BY-NC-SA-4.0", "https://nemar.org/dataexplorer/detail?dataset_id=i-care"),
                   "2": ("i-care", "CC-BY-NC-SA-4.0", "https://nemar.org/dataexplorer/detail?dataset_id=i-care")}
    assert root["1"].attrs["licence_url"] == "https://creativecommons.org/licenses/by-nc-sa/4.0/"
    assert root.attrs["licence"] == "other" and root.attrs["licence_file"] == "LICENSES.csv"
    assert root.attrs["licences"] == {"dsB": "CC0-1.0", "i-care": "CC-BY-NC-SA-4.0"}
    after = snapshot(store)
    assert {k for k in before if before[k] != after[k]} == {"zarr.json", "0/zarr.json", "1/zarr.json", "2/zarr.json"}
    assert np.array_equal(root["1"]["data"][:], open_part(parts, "i-care_000", 0)["data"][:])
    assert info["per_licence"] == {"CC0-1.0": 1, "CC-BY-NC-SA-4.0": 2}


def test_licence_reads_a_store_id_under_the_id_of_the_tables(tmp_path, nc):
    """The v1 store says nm000289 (or i-care, as here), the tables icare since 2026-10-30: the bucket keeps its id."""
    store, *_ = nc
    mp.licence(store, licence_table(tmp_path, dsB="CC0-1.0", icare="CC-BY-NC-SA-4.0"))
    root = zarr.open_group(store, mode="r")
    assert (root["1"].attrs["dataset"], root["1"].attrs["licence"]) == ("i-care", "CC-BY-NC-SA-4.0")
    assert root.attrs["licences"] == {"dsB": "CC0-1.0", "i-care": "CC-BY-NC-SA-4.0"}


def test_licence_source_points_openneuro_ids_to_openneuro():
    assert mp.source_url("ds004395") == "https://openneuro.org/datasets/ds004395"
    assert mp.source_url("nm000289") == "https://nemar.org/dataexplorer/detail?dataset_id=nm000289"


def test_licence_is_idempotent(tmp_path, nc, capsys):
    store, *_ = nc
    table = licence_table(tmp_path, dsB="CC0-1.0", **{"i-care": "CC-BY-NC-SA-4.0"})
    mp.licence(store, table)
    before = snapshot(store)
    assert mp.licence(store, table)["rewritten"] == 0 and snapshot(store) == before
    assert "nothing to change" in capsys.readouterr().out


@pytest.mark.parametrize("licences, why", [({"dsB": "CC0-1.0"}, "i-care: not in"),
                                           ({"dsB": "CC0-1.0", "i-care": ""}, "i-care: no licence"),
                                           ({"dsB": "CC0-1.0", "i-care": "custom"}, "no known URL")])
def test_licence_refuses_a_dataset_without_a_known_licence_and_writes_nothing(tmp_path, nc, licences, why):
    store, *_ = nc
    before = snapshot(store)
    with pytest.raises(SystemExit, match=why):
        mp.licence(store, licence_table(tmp_path, **licences))
    assert snapshot(store) == before


def test_licence_dry_run_changes_nothing(tmp_path, nc, capsys):
    store, *_ = nc
    before = snapshot(store)
    info = mp.licence(store, licence_table(tmp_path, dsB="CC0-1.0", **{"i-care": "CC-BY-NC-SA-4.0"}), dry_run=True)
    assert snapshot(store) == before and info["planned"] == 4 and info["rewritten"] == 0
    assert "dry run" in capsys.readouterr().out


def test_licence_never_writes_through_a_hard_link_into_the_part(tmp_path, nc):
    store, parts, _ = nc
    part_meta = parts / "i-care_000/samples.zarr/0/zarr.json"
    os.unlink(store / "1/zarr.json")
    os.link(part_meta, store / "1/zarr.json")
    before = part_meta.read_bytes()
    merge_main(["licence", "--store", str(store), "--table",
                str(licence_table(tmp_path, dsB="CC0-1.0", **{"i-care": "CC-BY-NC-SA-4.0"}))])
    assert part_meta.read_bytes() == before
    assert json.loads((store / "1/zarr.json").read_text())["attributes"]["licence"] == "CC-BY-NC-SA-4.0"


def test_the_v0_table_gives_every_v0_bucket_a_known_licence():
    """The licence step stops on a gap; the released corpus table and buckets side file show it here first."""
    table = {r["dataset_id"]: r["licence"] for r in rows(REPO / "corpus/eegdash-corpus-v0.csv")}
    used = {b["dataset"] for b in rows(REPO / "conversion/eegdash-corpus-v0-buckets.csv")}
    assert len(used) == 77 and all(table.get(mp.TABLE_IDS.get(d, d)) in mp.LICENCE_URLS for d in used)
