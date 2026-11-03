"""ZarrGroupWriter stores exactly the training samples REVE's loader would cut from the same sessions."""

import json

import numpy as np
import pytest
import zarr

from eegdash_to_reve import SHARD_SAMPLES, WINDOW, Bank, GroupWriter, Prepared, ZarrGroupWriter, sample_offsets

META = dict(subject="01", session="", run="1", task="x")


@pytest.fixture
def bank():
    names = ["C3", "C4", "Cz", "Fz", "Pz"]
    return Bank(names, np.arange(15, dtype=np.float32).reshape(5, 3) / 100)


def prep(names, T, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.normal(0, 20, (T, len(names))) + rng.normal(0, 5, len(names))).astype("<f4")
    return Prepared(data, sorted(names), [], 500.0)


def reve_offsets(start, end):
    """The expression in reve_eeg/src/utils/data_loading.py::_make_window_keys, verbatim."""
    return np.arange(start, end - WINDOW, WINDOW)


@pytest.mark.parametrize("T", [2001, 3999, 4000, 4001, 7777, 10000])
def test_sample_count_matches_reve_including_its_exact_multiple_quirk(T):
    """A session of exactly k*2000 timesteps gives k-1 samples in REVE (exclusive end); ours must agree."""
    assert np.array_equal(sample_offsets(T), reve_offsets(0, T))
    if T == 4000:
        assert len(sample_offsets(T)) == 1   # the documented quirk: two full windows fit, REVE takes one


def test_samples_equal_the_slices_reve_would_read(tmp_path, bank):
    """Every stored sample is bit-identical to REVE's slice of the concatenated bucket at the same offset."""
    sessions = [prep(["C3", "C4", "Cz"], T, seed=i) for i, T in enumerate([4000, 7777, 2500, 6001])]
    w = ZarrGroupWriter(tmp_path, "dsTEST", bank)
    for i, p in enumerate(sessions):
        w.append(p, f"sub-01/r{i}", META)
    w.finalize({})

    g = zarr.open_group(tmp_path / "samples.zarr", mode="r")["0"]
    data, session = g["data"][:], g["session"][:]
    bucket = np.concatenate([p.data for p in sessions])        # what the memmap layout would hold
    starts = np.cumsum([0] + [len(p.data) for p in sessions[:-1]])
    expected, expected_session = [], []
    for s, (p, start) in enumerate(zip(sessions, starts)):
        for off in reve_offsets(start, start + len(p.data)):
            expected.append(bucket[off:off + WINDOW])
            expected_session.append(s)
    assert data.shape == (len(expected), WINDOW, 3)
    assert np.array_equal(data, np.stack(expected))            # lossless: compression changes nothing
    assert session.tolist() == expected_session
    assert [x["n_samples"] for x in g.attrs["sessions"]] == [1, 3, 1, 3]


def test_stats_positions_and_groups(tmp_path, bank):
    """Stats are over the whole session (tail included), as the memmap writer computes them; a second channel set
    becomes a second group with its own positions."""
    a, b = prep(["C3", "C4", "Cz"], 5000, 1), prep(["Fz", "Pz"], 3000, 2)
    w = ZarrGroupWriter(tmp_path, "dsTEST", bank)
    assert w.append(a, "a", META) == 0 and w.append(b, "b", META) == 1
    w.finalize({})
    root = zarr.open_group(tmp_path / "samples.zarr", mode="r")
    g0, g1 = root["0"], root["1"]
    np.testing.assert_array_equal(g0["stats"][0], np.stack([a.data.mean(0), a.data.std(0)]))
    np.testing.assert_array_equal(g0["positions"][:], bank.positions(["C3", "C4", "Cz"]))
    assert g1.attrs["channels"] == ["Fz", "Pz"] and g1["data"].shape[2] == 2
    assert root.attrs["window"] == WINDOW and root.attrs["sfreq"] == 200


def test_one_chunk_per_sample_and_whole_shards(tmp_path, bank):
    """A sample is one chunk (one fetch); samples are grouped SHARD_SAMPLES to a file. Enough sessions to cross
    several shard boundaries mid-session, which is where an off-by-one in the buffering would show."""
    w = ZarrGroupWriter(tmp_path, "dsTEST", bank)
    rng = np.random.default_rng(0)
    lengths = rng.integers(2001, 60000, 20)
    ps = [prep(["C3", "C4"], int(T), seed=k) for k, T in enumerate(lengths)]
    for k, p in enumerate(ps):
        w.append(p, f"s{k}", META)
    w.finalize({})
    arr = zarr.open_group(tmp_path / "samples.zarr", mode="r")["0"]["data"]
    assert arr.chunks == (1, WINDOW, 2) and arr.shards == (SHARD_SAMPLES, WINDOW, 2)
    total = sum(len(sample_offsets(len(p.data))) for p in ps)
    assert arr.shape[0] == total > 2 * SHARD_SAMPLES
    stored = arr[:]
    expected = np.concatenate([p.data[:len(sample_offsets(len(p.data))) * WINDOW].reshape(-1, WINDOW, 2) for p in ps])
    assert np.array_equal(stored, expected)


def test_same_preprocessing_as_the_memmap_writer(tmp_path, bank):
    """Both writers receive the same Prepared sessions; the Zarr samples are exactly REVE's windows over the memmap."""
    ps = [prep(["C3", "C4", "Cz"], T, seed=i) for i, T in enumerate([4500, 8000])]
    mm, zr = GroupWriter(tmp_path / "mm", "dsTEST", bank), ZarrGroupWriter(tmp_path / "zr", "dsTEST", bank)
    for i, p in enumerate(ps):
        mm.append(p, f"r{i}", META)
        zr.append(p, f"r{i}", META)
    mm.finalize({})
    zr.finalize({})
    T = sum(len(p.data) for p in ps)
    memmap = np.fromfile(tmp_path / "mm/recordings/recording_-_eeg_-_0.npy", dtype="<f4").reshape(T, 3)
    stats_mm = np.fromfile(tmp_path / "mm/stats/recording_-_stats_-_0.npy", dtype="<f4").reshape(2, 2, 3)
    g = zarr.open_group(tmp_path / "zr/samples.zarr", mode="r")["0"]
    starts = [0, len(ps[0].data)]
    windows = [memmap[o:o + WINDOW] for p, s in zip(ps, starts) for o in reve_offsets(s, s + len(p.data))]
    assert np.array_equal(g["data"][:], np.stack(windows))
    assert np.array_equal(g["stats"][:], stats_mm)
    s = json.loads((tmp_path / "zr/summary.json").read_text())
    assert s["format"] == "zarr" and s["n_samples"] == len(windows) and (tmp_path / "zr/_DONE").exists()


def test_uncompressed_option(tmp_path, bank):
    w = ZarrGroupWriter(tmp_path, "dsTEST", bank, compress=False)
    p = prep(["C3", "C4"], 6001)
    w.append(p, "r", META)
    w.finalize({})
    arr = zarr.open_group(tmp_path / "samples.zarr", mode="r")["0"]["data"]
    assert np.array_equal(arr[:], p.data[:3 * WINDOW].reshape(3, WINDOW, 2))


def test_refuses_short_session(tmp_path, bank):
    w = ZarrGroupWriter(tmp_path, "dsTEST", bank)
    with pytest.raises(ValueError):
        w.append(prep(["C3", "C4"], WINDOW), "short", META)
