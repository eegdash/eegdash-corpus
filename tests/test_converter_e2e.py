"""End to end, offline: synthetic BIDS dataset -> EEGDash local discovery -> converter -> REVE loader batch."""

import json
import warnings

import mne
import numpy as np
import pytest
from mne_bids import BIDSPath, write_raw_bids

from eegdash_to_reve import Bank, convert_shard

DSID = "ds999901"
CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def make_bids(root, runs):
    """runs: list of (subject, run, seconds). 250 Hz, 19 EEG (mixed-case 10-20 names) + EOG + one unknown name."""
    rng = np.random.default_rng(0)
    for subject, run, seconds in runs:
        names, types = CH + ["EOG1", "XX7"], ["eeg"] * len(CH) + ["eog", "eeg"]
        info = mne.create_info(names, 250.0, types)
        data = rng.normal(0, 20e-6, (len(names), int(250 * seconds)))
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.set_meas_date(0)
        bp = BIDSPath(root=root, subject=subject, task="rest", run=run, datatype="eeg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_raw_bids(raw, bp, format="BrainVision", allow_preload=True, overwrite=True, verbose="ERROR")


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    make_bids(cache / DSID, [("01", "01", 40), ("01", "02", 30), ("02", "01", 25), ("02", "02", 6)])  # last one is < 10 s
    return cache


@pytest.fixture(scope="module")
def bank():
    xyz = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    names = [n for n in CH if n in xyz] + ["CZ", "biosemi128_A1"]
    pos = np.array([xyz[n] for n in CH] + [xyz["Cz"], [0, 0, 0.095]], dtype=np.float32)
    return Bank(names, pos)


def test_convert_shard_offline(cache, bank, tmp_path, reve_src, scripts_on_path):
    out = tmp_path / "part"
    summary = convert_shard(DSID, None, cache, out, offline=True, n_jobs=1, bank=bank)
    assert summary["n_records_seen"] == 4
    assert summary["skipped_by_reason"] == {"too_short": 1}
    assert summary["n_sessions"] == 3 and summary["n_big_recordings"] == 1
    assert summary["channels_dropped_no_position"] == {"XX7": 3}  # EOG never reaches the bank; XX7 has no position
    assert abs(summary["hours_written"] - (40 + 30 + 25) / 3600) < 1e-3

    corr = list(open(out / "csv_recordings/df_corrected.csv"))
    assert len(corr) == 4  # header + 3 sessions
    assert all(f"{int(s * 200)}" in line for s, line in zip((40, 30, 25), corr[1:]))  # exact sample counts at 200 Hz
    big = json.loads(list(__import__("csv").DictReader(open(out / "csv_recordings/df_big.csv")))[0]["chans_names"])
    assert big == sorted(CH)  # sorted canonical names, EOG and XX7 gone
    man = list(__import__("csv").DictReader(open(out / "manifest_part.csv")))
    assert {m["subject"] for m in man} == {"01", "02"} and man[0]["sfreq_native"] == "250.0"

    # the real consumer: reve_eeg's loader builds a batch from it
    from check_layout import check_layout
    info = check_layout(out, batch_size=8)
    # _make_window_keys = arange(start, end - 2000, 2000): the last 10 s of every session is never used,
    # so 40 / 30 / 25 s give 3 + 2 + 2 windows.
    assert info["channel_groups"] == {19: 7}
    assert info["batch_shapes"][0][1:] == (19, 2000)


def test_second_run_is_noop_when_done(cache, bank, tmp_path, scripts_on_path):
    out = tmp_path / "part"
    convert_shard(DSID, ["02"], cache, out, offline=True, bank=bank)
    marker = (out / "_DONE").read_text()
    convert_shard(DSID, ["02"], cache, out, offline=True, bank=bank)  # _DONE present -> skipped
    assert (out / "_DONE").read_text() == marker


def test_held_out_refused(cache, bank, tmp_path):
    with pytest.raises(SystemExit):
        convert_shard("ds004362", None, cache, tmp_path / "x", bank=bank)


def test_convert_shard_to_zarr_matches_what_reve_trains_on(cache, bank, tmp_path, reve_src, scripts_on_path):
    """Raw BIDS -> converter --format zarr, the pipeline the paper publishes. The same raw files through the memmap
    path feed REVE's own loader 7 samples; the Zarr store must hold exactly those 7, byte for byte."""
    import zarr

    from eegdash_to_reve import WINDOW

    mm, zr = tmp_path / "mm", tmp_path / "zr"
    convert_shard(DSID, None, cache, mm, offline=True, n_jobs=1, bank=bank)
    summary = convert_shard(DSID, None, cache, zr, offline=True, n_jobs=1, bank=bank, fmt="zarr")
    assert summary["format"] == "zarr" and summary["n_sessions"] == 3 and summary["n_samples"] == 7
    assert summary["skipped_by_reason"] == {"too_short": 1}

    g = zarr.open_group(zr / "samples.zarr", mode="r")["0"]
    assert g["data"].shape == (7, WINDOW, 19) and g.attrs["channels"] == sorted(CH)
    assert g["session"][:].tolist() == [0, 0, 0, 1, 1, 2, 2]          # 40 / 30 / 25 s -> 3 + 2 + 2

    rows = list(__import__("csv").DictReader(open(mm / "csv_recordings/df_corrected.csv")))
    durations = [int(r["duration"]) for r in rows]
    memmap = np.fromfile(mm / "recordings/recording_-_eeg_-_0.npy", dtype="<f4").reshape(sum(durations), 19)
    starts = np.cumsum([0] + durations[:-1])
    reve = [memmap[o:o + WINDOW] for s, d in zip(starts, durations) for o in np.arange(s, s + d - WINDOW, WINDOW)]
    assert np.array_equal(g["data"][:], np.stack(reve))
    stats = np.fromfile(mm / "stats/recording_-_stats_-_0.npy", dtype="<f4").reshape(3, 2, 19)
    assert np.array_equal(g["stats"][:], stats)


def test_bids_root_reads_a_tree_eegdash_does_not_carry(cache, bank, tmp_path):
    """--bids-root (I-CARE): the same tree read directly gives byte-identical samples to the EEGDash path, full
    relative paths in the manifest, and only the shard's subjects are opened."""
    import csv
    import zarr

    via_eegdash = convert_shard(DSID, ["01"], cache, tmp_path / "a", bank=bank, fmt="zarr")
    direct = convert_shard("i-care", ["01"], tmp_path / "unused", tmp_path / "b", bank=bank, fmt="zarr",
                           bids_root=cache / DSID)
    assert direct["n_records_seen"] == 2 and direct["n_sessions"] == via_eegdash["n_sessions"] == 2
    a = zarr.open_group(tmp_path / "a/samples.zarr", mode="r")["0"]
    b = zarr.open_group(tmp_path / "b/samples.zarr", mode="r")["0"]
    for key in ("data", "session", "stats", "positions"):
        assert np.array_equal(a[key][:], b[key][:]), key
    assert b.attrs["dataset"] == "i-care" and b.attrs["channels"] == a.attrs["channels"]
    files = [r["file"] for r in csv.DictReader(open(tmp_path / "b/manifest_part.csv"))]
    assert files == ["sub-01/eeg/sub-01_task-rest_run-01_eeg.vhdr", "sub-01/eeg/sub-01_task-rest_run-02_eeg.vhdr"]
    meta = next(csv.DictReader(open(tmp_path / "b/manifest_part.csv")))
    assert (meta["subject"], meta["run"], meta["task"]) == ("01", "01", "rest")


def test_bids_root_behind_a_symlink_keeps_full_paths(cache, bank, tmp_path):
    """EEGDash recorded bare file names behind a symlinked root (2026-09-24); the direct reader must not."""
    import csv

    (tmp_path / "link").symlink_to(cache / DSID)
    convert_shard("i-care", ["02"], tmp_path, tmp_path / "c", bank=bank, fmt="zarr", bids_root=tmp_path / "link")
    files = [r["file"] for r in csv.DictReader(open(tmp_path / "c/manifest_part.csv"))]
    assert files == ["sub-02/eeg/sub-02_task-rest_run-01_eeg.vhdr"]      # run-02 is < 10 s and skipped


TSV_DSID = "ds999902"
XX7_TRUE = "Oz"   # where the fixture's unknown channel XX7 really sits: a standard_1005 position the bank lacks


@pytest.fixture(scope="module")
def cache_tsv(tmp_path_factory):
    """Two subjects; sub-01 ships an electrodes.tsv (millimetres, rotated, shifted) that also places XX7."""
    from scipy.spatial.transform import Rotation

    cache = tmp_path_factory.mktemp("cache_tsv")
    make_bids(cache / TSV_DSID, [("01", "01", 30), ("02", "01", 30)])
    ch_pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    r = Rotation.from_euler("zyx", [30, -10, 5], degrees=True)
    rows = ["name\tx\ty\tz"] + [f"{n}\t" + "\t".join(f"{v:.3f}" for v in r.apply(ch_pos[m]) * 1000 + [3, -8, 20])
                                for n, m in [(c, c) for c in CH] + [("XX7", XX7_TRUE)]]
    eeg = cache / TSV_DSID / "sub-01" / "eeg"
    (eeg / "sub-01_electrodes.tsv").write_text("\n".join(rows) + "\n")
    (eeg / "sub-01_coordsystem.json").write_text(json.dumps({"EEGCoordinateSystem": "CapTrak", "EEGCoordinateUnits": "mm"}))
    return cache


def test_electrodes_tsv_places_a_channel_the_bank_lacks(cache_tsv, bank, tmp_path):
    """sub-01's XX7 is kept with the position its electrodes.tsv gives after the fit; sub-02 has no file and drops it.
    Both writers store the same positions; --no-dataset-positions drops XX7 everywhere."""
    import csv
    import zarr

    truth = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"][XX7_TRUE]
    s = convert_shard(TSV_DSID, None, cache_tsv, tmp_path / "mm", bank=bank)
    assert s["channels_aligned_from_dataset"] == {"XX7": 1} and s["channels_dropped_no_position"] == {"XX7": 1}
    assert s["n_big_recordings"] == 2
    man = {r["subject"]: r for r in csv.DictReader(open(tmp_path / "mm/manifest_part.csv"))}
    assert (man["01"]["n_positions_aligned"], man["02"]["n_positions_aligned"]) == ("1", "0")
    g = int(man["01"]["big_recording_index"])
    names = json.loads(next(r for r in csv.DictReader(open(tmp_path / "mm/csv_recordings/df_big.csv"))
                            if int(r["big_recording_index"]) == g)["chans_names"])
    pos = np.load(tmp_path / f"mm/positions/recording_-_positions_-_{g}.npy")
    assert names == sorted(CH + ["XX7"]) and np.linalg.norm(pos[names.index("XX7")] - truth) < 1e-3

    convert_shard(TSV_DSID, None, cache_tsv, tmp_path / "zr", bank=bank, fmt="zarr")
    zg = zarr.open_group(tmp_path / "zr/samples.zarr", mode="r")[str(g)]
    assert zg.attrs["channels"] == names and np.array_equal(zg["positions"][:], pos)

    off = convert_shard(TSV_DSID, None, cache_tsv, tmp_path / "off", bank=bank, use_dataset_positions=False)
    assert off["channels_aligned_from_dataset"] == {} and off["channels_dropped_no_position"] == {"XX7": 2}
    assert off["n_big_recordings"] == 1
