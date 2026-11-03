"""Merging shard parts: renumbering, hard links, verification, held-out refusal."""

import csv

import numpy as np
import pytest

from eegdash_to_reve import Bank, GroupWriter, Prepared
from merge_parts import merge


@pytest.fixture
def bank():
    return Bank(["C3", "C4", "Cz", "Fz", "Pz"], np.arange(15, dtype=np.float32).reshape(5, 3) / 100)


def prep(names, T, seed):
    rng = np.random.default_rng(seed)
    return Prepared(rng.normal(0, 20, (T, len(names))).astype("<f4"), sorted(names), [], 500.0)


def make_part(root, dataset, sessions, bank, done=True):
    w = GroupWriter(root / f"{dataset}_000", dataset, bank)
    for k, (names, T) in enumerate(sessions):
        w.append(prep(names, T, k), f"sub-0{k}/x", dict(subject=f"0{k}", session="", run="1", task="t"))
    w.finalize({"skipped_by_reason": {"too_short": 1}, "channels_dropped_no_position": {"XX": 2}})
    if not done:
        (root / f"{dataset}_000" / "_DONE").unlink()
    return root / f"{dataset}_000"


def read(p):
    return list(csv.DictReader(open(p, newline="")))


def test_merge_renumbers_links_and_verifies(tmp_path, bank):
    parts = tmp_path / "parts"
    make_part(parts, "dsA", [(["C3", "C4", "Cz"], 3000), (["C3", "C4", "Cz"], 2500), (["Fz", "Pz", "C3"], 4000)], bank)
    make_part(parts, "dsB", [(["Cz", "Fz"], 5000)], bank)
    out = tmp_path / "merged"
    info = merge(parts, out)
    assert info["big_recordings"] == 3 and info["sessions"] == 4 and info["datasets"] == ["dsA", "dsB"]

    big, corr, stats = (read(out / "csv_recordings" / n) for n in ("df_big.csv", "df_corrected.csv", "df_stats_tmp.csv"))
    assert [int(b["big_recording_index"]) for b in big] == [0, 1, 2]
    assert [int(r["big_recording_index"]) for r in corr] == [0, 0, 1, 2]
    assert [int(s["n_sessions"]) for s in stats] == [2, 1, 1]
    assert (out / "recordings" / "recording_-_eeg_-_2.npy").stat().st_size == 5000 * 2 * 4
    # hard link, not copy
    src = parts / "dsB_000" / "recordings" / "recording_-_eeg_-_0.npy"
    assert (out / "recordings" / "recording_-_eeg_-_2.npy").stat().st_ino == src.stat().st_ino
    man = read(out / "manifest.csv")
    assert len(man) == 4 and {m["dataset_id"] for m in man} == {"dsA", "dsB"}
    assert man[-1]["big_recording_index"] == "2"
    report = (out / "CONVERSION_REPORT.md").read_text()
    assert "| dsA |" in report and "too_short 2" in report and (out / "csv_recordings" / "df_stats.csv").exists()


def test_incomplete_part_blocks_unless_allowed(tmp_path, bank):
    parts = tmp_path / "parts"
    make_part(parts, "dsA", [(["C3", "C4", "Cz"], 3000)], bank)
    make_part(parts, "dsC", [(["C3", "C4", "Cz"], 3000)], bank, done=False)
    with pytest.raises(SystemExit):
        merge(parts, tmp_path / "m1")
    info = merge(parts, tmp_path / "m2", allow_missing=True)
    assert info["incomplete"] == ["dsC_000"] and info["datasets"] == ["dsA"]


def test_held_out_dataset_refused(tmp_path, bank):
    parts = tmp_path / "parts"
    make_part(parts, "ds004362", [(["C3", "C4", "Cz"], 3000)], bank)
    with pytest.raises(SystemExit):
        merge(parts, tmp_path / "m")


def test_damaged_part_is_named_before_anything_is_linked(tmp_path, bank):
    parts = tmp_path / "parts"
    make_part(parts, "dsA", [(["C3", "C4", "Cz"], 3000)], bank)
    bad = make_part(parts, "dsB", [(["Cz", "Fz"], 5000)], bank)
    with open(bad / "recordings" / "recording_-_eeg_-_0.npy", "ab") as f:  # a second writer appended
        f.write(b"\0" * 5000 * 2 * 4)
    with pytest.raises(SystemExit, match="dsB_000"):
        merge(parts, tmp_path / "m")
    assert not (tmp_path / "m" / "recordings").exists() or not list((tmp_path / "m" / "recordings").iterdir())
