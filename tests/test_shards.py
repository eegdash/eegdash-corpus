"""Subject packing into shards, and the sidecar-based header scan."""

import json

import pytest

from make_shards import DEFAULT_HOURS, entities, pack_subjects, scan_dataset, shard_rows


def recs(spec):
    """spec: {subject: [hours per record]}"""
    return [dict(dataset="dsX", subject=s, hours=h) for s, hs in spec.items() for h in hs]


def test_packs_in_sorted_order_until_limits():
    r = recs({"03": [50, 50], "01": [120], "02": [90], "04": [15]})
    shards = pack_subjects(r, max_hours=200, max_records=400)
    assert shards == [["01"], ["02", "03"], ["04"]]  # 120 | 90+100=190 (+15 would exceed 200) | 15


def test_record_limit():
    r = recs({"a": [0.1] * 300, "b": [0.1] * 150, "c": [0.1] * 10})
    assert pack_subjects(r, max_hours=1e9, max_records=400) == [["a"], ["b", "c"]]


def test_single_subject_over_limit_gets_own_shard():
    r = recs({"big": [500], "s": [1]})
    assert pack_subjects(r, max_hours=200, max_records=400) == [["big"], ["s"]]


def test_unknown_duration_uses_default():
    r = [dict(dataset="dsX", subject="a", hours=None)] * 3
    rows = shard_rows("dsX", "offline", r, pack_subjects(r, 200, 400))
    assert rows[0]["est_hours"] == round(3 * DEFAULT_HOURS, 2) and rows[0]["shard_id"] == "dsX_000"


def test_online_dataset_is_one_shard():
    rows = shard_rows("nm000146", "online", [], [["all"]])
    assert rows == [dict(shard_id="nm000146_000", dataset="nm000146", mode="online", subjects="all", n_records=0, est_hours=0)]


def test_entities():
    assert entities("sub-NDARAB123_ses-01_task-RestingState_run-2_eeg.json") == {"sub": "NDARAB123", "ses": "01", "task": "RestingState", "run": "2"}
    assert entities("sub-6_ses-0_task-imagery_run-0_eeg.bdf") == {"sub": "6", "ses": "0", "task": "imagery", "run": "0"}


def write_edf(path, seconds=120):
    import mne
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.RawArray(np.zeros((3, 250 * seconds)), mne.create_info(["Cz", "Pz", "Fz"], 250.0, "eeg"), verbose="ERROR")
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="ERROR")


@pytest.fixture
def bids_tree(tmp_path):
    """sub-01: own sidecar with RecordingDuration. sub-02: own sidecar without duration (MNE header fallback).
    sub-03: no own sidecar, covered by a dataset-level task-rest_eeg.json (BIDS inheritance).
    sub-04: task-other, no sidecar anywhere (header only). sub-05: orphan sidecar without data (ignored)."""
    root = tmp_path / "ds999902"
    write_edf(root / "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.edf")
    (root / "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.json").write_text(
        json.dumps({"SamplingFrequency": 250, "EEGChannelCount": 3, "RecordingDuration": 120.0}))
    write_edf(root / "sub-02/eeg/sub-02_task-rest_eeg.edf")
    (root / "sub-02/eeg/sub-02_task-rest_eeg.json").write_text(json.dumps({"SamplingFrequency": 250, "EEGChannelCount": 3}))
    write_edf(root / "sub-03/eeg/sub-03_task-rest_eeg.edf", seconds=60)
    (root / "task-rest_eeg.json").write_text(json.dumps({"SamplingFrequency": 250, "EEGChannelCount": 3, "RecordingDuration": 60.0}))
    write_edf(root / "sub-04/eeg/sub-04_task-other_eeg.edf", seconds=30)
    (root / "sub-05/eeg").mkdir(parents=True)
    (root / "sub-05/eeg/sub-05_task-rest_eeg.json").write_text("{}")
    return tmp_path


def test_scan_dataset_sidecars_inheritance_and_header_fallback(bids_tree):
    rows = {r["subject"]: r for r in scan_dataset(bids_tree, "ds999902")}
    assert set(rows) == {"01", "02", "03", "04"}
    assert all(r["sfreq"] == 250 and r["nchans"] == 3 for r in rows.values())
    assert rows["01"]["hours"] == round(120 / 3600, 4) and rows["01"]["session"] == "01"
    assert abs(rows["02"]["hours"] - 120 / 3600) < 1e-3          # own sidecar lacks duration -> EDF header
    assert rows["03"]["hours"] == round(60 / 3600, 4)              # dataset-level task-rest_eeg.json
    assert abs(rows["04"]["hours"] - 30 / 3600) < 1e-3 and rows["04"]["task"] == "other"   # no sidecar at all
    assert rows["01"]["file"] == "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.edf" and rows["01"]["ntimes"] == 30000


def test_dataset_outside_eegdash_through_a_symlink(tmp_path):
    """I-CARE: the BIDS tree sits in open/icare/bids and is planned as `i-care` through a symlink, no copy."""
    from make_shards import scan_dataset
    from test_converter_e2e import make_bids

    make_bids(tmp_path / "icare" / "bids", [("0284", "01", 40), ("0286", "01", 30)])
    (tmp_path / "icare" / "i-care").symlink_to("bids")
    rows = scan_dataset(tmp_path / "icare", "i-care")
    assert [(r["dataset"], r["subject"]) for r in rows] == [("i-care", "0284"), ("i-care", "0286")]
    assert all(r["hours"] and r["sfreq"] for r in rows)
