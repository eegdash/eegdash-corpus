"""GroupWriter writes exactly what data_loading.py's memmaps expect."""

import csv
import json

import numpy as np
import pytest

from eegdash_to_reve import WINDOW, Bank, GroupWriter, Prepared


@pytest.fixture
def bank():
    names = ["C3", "C4", "Cz", "Fz", "Pz"]
    return Bank(names, np.arange(15, dtype=np.float32).reshape(5, 3) / 100)


def prep(names, T, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.normal(0, 20, (T, len(names))) + rng.normal(0, 5, len(names))).astype("<f4")
    return Prepared(data, sorted(names), [], 500.0)


def read(p):
    return list(csv.DictReader(open(p, newline="")))


def test_streams_sessions_into_groups(tmp_path, bank):
    w = GroupWriter(tmp_path, "dsTEST", bank)
    g0 = w.append(prep(["C3", "C4", "Cz"], 3000), "sub-01/r1", dict(subject="01", session="", run="1", task="x"))
    g1 = w.append(prep(["C3", "C4", "Cz"], 2500, 1), "sub-01/r2", dict(subject="01", session="", run="2", task="x"))
    g2 = w.append(prep(["Fz", "Pz", "C3", "C4"], 4000, 2), "sub-02/r1", dict(subject="02", session="", run="1", task="x"))
    assert (g0, g1, g2) == (0, 0, 1)
    w.finalize({})

    big = read(tmp_path / "csv_recordings/df_big.csv")
    corr = read(tmp_path / "csv_recordings/df_corrected.csv")
    stats = read(tmp_path / "csv_recordings/df_stats_tmp.csv")
    assert [int(b["duration"]) for b in big] == [5500, 4000]
    assert [int(b["n_chans"]) for b in big] == [3, 4]
    assert json.loads(big[0]["chans_names"]) == ["C3", "C4", "Cz"]
    assert [(int(r["big_recording_index"]), int(r["index"]), int(r["duration"])) for r in corr] == [(0, 0, 3000), (0, 1, 2500), (1, 0, 4000)]
    assert [int(s["n_sessions"]) for s in stats] == [2, 1]

    # raw memmaps: sizes and content round-trip through np.memmap with the CSV shapes
    for b in big:
        i, T, C = int(b["big_recording_index"]), int(b["duration"]), int(b["n_chans"])
        assert (tmp_path / f"recordings/recording_-_eeg_-_{i}.npy").stat().st_size == T * C * 4
        mm = np.memmap(tmp_path / f"recordings/recording_-_eeg_-_{i}.npy", mode="r", dtype="float32", shape=(T, C))
        st = np.memmap(tmp_path / f"stats/recording_-_stats_-_{i}.npy", mode="r", dtype="float32",
                       shape=(int(stats[i]["n_sessions"]), 2, C))
        pos = np.load(tmp_path / f"positions/recording_-_positions_-_{i}.npy")
        assert pos.shape == (C, 3)
        assert np.isfinite(mm).all() and np.isfinite(st).all()
    # session 1 of group 0 starts at offset 3000 and its stats row is index 1
    mm0 = np.memmap(tmp_path / "recordings/recording_-_eeg_-_0.npy", mode="r", dtype="float32", shape=(5500, 3))
    st0 = np.memmap(tmp_path / "stats/recording_-_stats_-_0.npy", mode="r", dtype="float32", shape=(2, 2, 3))
    sess1 = np.asarray(mm0[3000:])
    assert np.allclose(sess1.mean(0), st0[1, 0], atol=1e-3) and np.allclose(sess1.std(0), st0[1, 1], atol=1e-3)
    assert (tmp_path / "_DONE").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_sessions"] == 3 and summary["n_big_recordings"] == 2
    assert len(read(tmp_path / "manifest_part.csv")) == 3


def test_refuses_short_session(tmp_path, bank):
    w = GroupWriter(tmp_path, "dsTEST", bank)
    with pytest.raises(ValueError):
        w.append(prep(["C3", "C4", "Cz"], WINDOW), "sub-01/short", {})


def test_preprocess_falls_back_to_names_when_no_channel_is_typed_eeg(bank):
    """channels.tsv with type n/a makes EEGDash mark everything misc (ds005697): the bank decides instead."""
    import mne

    from eegdash_to_reve import Prepared, preprocess_recording

    rng = np.random.default_rng(0)
    info = mne.create_info(["Cz", "Pz", "Fz", "ECG"], 250.0, ["misc"] * 4)
    raw = mne.io.RawArray(rng.normal(0, 20e-6, (4, 250 * 60)), info, verbose="ERROR")
    res = preprocess_recording(raw, bank)
    assert isinstance(res, Prepared) and res.typed_from_names
    assert res.names == ["Cz", "Fz", "Pz"] and res.dropped == ["ECG"] and res.data.shape == (200 * 60, 3)
