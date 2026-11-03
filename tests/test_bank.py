"""Channel-name resolution against the position bank."""

import numpy as np
import pytest

from eegdash_to_reve import Bank


@pytest.fixture
def bank():
    names = ["Cz", "CZ", "Fp1", "C3", "A1", "A2", "B7", "AFF1h", "E1", "E129"]
    names += [f"biosemi128_{p}{i}" for p in "ABCD" for i in range(1, 33)]
    rng = np.random.default_rng(0)
    xyz = rng.normal(0, 0.05, (len(names), 3)).astype(np.float32)
    xyz[1] = xyz[0]  # Cz / CZ share coordinates, like the real bank
    return Bank(names, xyz)


def test_exact_and_casefold(bank):
    kept, canon, dropped = bank.resolve(["FP1", "cz", "aff1h", "XX9"])
    assert kept == [0, 1, 2]
    assert canon == ["Fp1", "Cz", "AFF1h"]
    assert dropped == ["XX9"]


def test_duplicate_electrode_kept_once(bank):
    kept, canon, dropped = bank.resolve(["Cz", "CZ", "E1"])
    assert canon == ["Cz", "E1"] and dropped == ["CZ"]


def test_ten_twenty_cap_keeps_bare_names(bank):
    kept, canon, dropped = bank.resolve(["A1", "A2", "C3", "Cz"])
    assert canon == ["A1", "A2", "C3", "Cz"]  # earlobes and 10-20 C3, not the BioSemi ring


def test_biosemi_cap_maps_to_biosemi_names(bank):
    names = [f"{p}{i}" for p in "ABCD" for i in range(1, 33)] + ["EXG1"]
    kept, canon, dropped = bank.resolve(names)
    assert len(canon) == 128 and all(c.startswith("biosemi128_") for c in canon)
    assert canon[0] == "biosemi128_A1" and dropped == ["EXG1"]


def test_positions_follow_canonical_order(bank):
    canon = ["E129", "Cz"]
    pos = bank.positions(canon)
    assert pos.shape == (2, 3) and np.allclose(pos[0], bank.xyz[bank.exact["E129"]])


def test_eeg_prefix_is_stripped_before_the_lookup(bank):
    """Siena (nm000385) labels every electrode `EEG Fp1`; the bank matches only the bare name (2026-10-13)."""
    kept, canon, dropped = bank.resolve(["EEG Fp1", "EEG-Cz", "eeg_C3", "EKG EKG", "SPO2", "EEG XX9"])
    assert canon == ["Fp1", "Cz", "C3"] and dropped == ["EKG EKG", "SPO2", "EEG XX9"]


# ----------------------------------------------------------------------------- BioSemi-256 caps
RINGS_128 = [f"{p}{i}" for p in "ABCD" for i in range(1, 33)]
RINGS_256 = [f"{p}{i}" for p in "ABCDEFGH" for i in range(1, 33)]


@pytest.fixture
def bank256():
    """The real bank's traps: bare A1..D32 (10-20 names), the 128 cap's rings, GSN's E1..E129, 10-10 F/C names."""
    from eegdash_to_reve import with_biosemi_templates

    names = ["Cz", "Fz", "F1", "F2", "F3", "F4", "C1", "C2", "C3", "C4", "PO3", "POz", "PO7", "F10"] + RINGS_128
    names += [f"biosemi128_{n}" for n in RINGS_128] + [f"E{i}" for i in range(1, 130)]
    xyz = np.random.default_rng(1).normal(0, 0.05, (len(names), 3)).astype(np.float32)
    return Bank(*with_biosemi_templates(names, xyz))


def test_biosemi256_cap_maps_every_ring_to_biosemi256(bank256):
    import mne

    kept, canon, dropped = bank256.resolve(RINGS_256 + ["EXG1", "Status"])
    assert canon == [f"biosemi256_{n}" for n in RINGS_256] and dropped == ["EXG1", "Status"]
    template = mne.channels.make_standard_montage("biosemi256").get_positions()["ch_pos"]
    pos = bank256.positions(canon)
    assert np.allclose(pos, np.array([template[n] for n in RINGS_256]), atol=1e-7)   # rings A-D too: not the 128 cap's
    b128 = bank256.positions([f"biosemi128_{n}" for n in RINGS_128])
    assert np.linalg.norm(pos[:128] - b128, axis=1).max() > 0.01


def test_biosemi128_cap_still_maps_to_biosemi128(bank256):
    kept, canon, dropped = bank256.resolve(RINGS_128)
    assert canon == [f"biosemi128_{n}" for n in RINGS_128] and not dropped


def test_gsn_net_is_untouched_by_the_256_rule(bank256):
    names = [f"E{i}" for i in range(1, 130)]
    kept, canon, dropped = bank256.resolve(names)
    assert canon == names and not dropped


def test_ten_ten_cap_with_f_and_c_names_is_not_a_256_cap(bank256):
    """F1..F4 and C1..C4 are ring labels of both halves of a 256 cap; a 10-10 cap must keep its own names."""
    names = ["F1", "F2", "F3", "F4", "C1", "C2", "C3", "C4", "Cz", "Fz"]
    kept, canon, dropped = bank256.resolve(names)
    assert canon == names and not dropped


def test_from_json_appends_biosemi256_once(tmp_path):
    import json

    from eegdash_to_reve import with_biosemi_templates

    (tmp_path / "bank.json").write_text(json.dumps({"Cz": [0, 0, 0.09], "Fz": [0, 0.07, 0.06]}))
    b = Bank.from_json(tmp_path / "bank.json")
    assert len(b.names) == 2 + 160 + 256 and b.names[:2] == ["Cz", "Fz"] and b.names[-1] == "biosemi256_H32"
    names, xyz = with_biosemi_templates(b.names, b.xyz)   # a bank that already has them (the committed JSON) gains nothing
    assert names == b.names and np.array_equal(xyz, b.xyz)


# ----------------------------------------------------------------------------- a zero typed for O
def test_zero_for_o_is_a_fallback(bank256):
    """SparrKULee writes P03, P07, P0z for PO3, PO7, POz; a name with a genuine zero (F10) is not touched."""
    kept, canon, dropped = bank256.resolve(["P03", "P0z", "P07", "F10", "X01"])
    assert canon == ["PO3", "POz", "PO7", "F10"] and dropped == ["X01"]


def test_zero_for_o_never_duplicates_the_real_name(bank256):
    kept, canon, dropped = bank256.resolve(["PO3", "P03"])
    assert canon == ["PO3"] and dropped == ["P03"]


# ----------------------------------------------------------------------------- positions from the dataset
KNOWN = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4", "P3", "P4"]   # 12 electrodes the bank knows
CB1_TRUE = np.array([-0.032, -0.092, -0.035])                                          # metres, in the bank's frame


@pytest.fixture
def bank1005():
    import mne

    ch_pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    return Bank(KNOWN, np.array([ch_pos[n] for n in KNOWN], dtype=np.float32))


def to_file_frame(xyz: np.ndarray) -> np.ndarray:
    """A dataset's own frame: rotated 30 degrees about a tilted axis, in millimetres, shifted."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_rotvec(np.deg2rad(30) * np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5]))
    return r.apply(xyz) * 1000 + np.array([5.0, -12.0, 40.0])


def write_electrodes(path, coords: dict[str, np.ndarray]) -> None:
    rows = ["name\tx\ty\tz\timpedance"] + [f"{n}\t{x:.4f}\t{y:.4f}\t{z:.4f}\tn/a" for n, (x, y, z) in coords.items()]
    path.write_text("\n".join(rows + ["GND\tn/a\tn/a\tn/a\tn/a"]) + "\n")


def raw_with(names, types):
    import mne

    info = mne.create_info(names, 250.0, types)
    return mne.io.RawArray(np.random.default_rng(0).normal(0, 20e-6, (len(names), 250 * 30)), info, verbose="ERROR")


def file_coords(bank, names=KNOWN):
    coords = {n: bank.xyz[bank.exact[n]].astype(float) for n in names}
    coords["CB1"], coords["EXG3"] = CB1_TRUE, np.array([0.0, -0.1, -0.06])
    return {n: to_file_frame(v) for n, v in coords.items()}


def test_dataset_positions_place_an_electrode_the_bank_lacks(bank1005, tmp_path):
    from eegdash_to_reve import GroupWriter, Prepared, preprocess_recording

    write_electrodes(tmp_path / "e.tsv", file_coords(bank1005))
    raw = raw_with(KNOWN + ["CB1", "EXG3", "EOG1"], ["eeg"] * 14 + ["eog"])
    res = preprocess_recording(raw, bank1005, electrodes=tmp_path / "e.tsv")
    assert isinstance(res, Prepared) and res.aligned == ["CB1"] and res.dropped == ["EXG3"]   # EXG never placed
    assert res.names == sorted(KNOWN + ["CB1"]) and res.positions.shape == (13, 3)
    pos = dict(zip(res.names, res.positions))
    assert np.linalg.norm(pos["CB1"] - CB1_TRUE) < 1e-3
    assert all(np.array_equal(pos[n], bank1005.xyz[bank1005.exact[n]]) for n in KNOWN)   # bank names keep the bank's

    w = GroupWriter(tmp_path / "out", "dsTEST", bank1005)
    w.append(res, "sub-01/r1", dict(subject="01", session="", run="1", task="x"))
    w.finalize({})
    man = list(__import__("csv").DictReader(open(tmp_path / "out/manifest_part.csv")))
    assert man[0]["n_positions_aligned"] == "1" and man[0]["n_dropped_no_position"] == "1"
    assert np.array_equal(np.load(tmp_path / "out/positions/recording_-_positions_-_0.npy"), res.positions)


def test_a_file_that_does_not_fit_is_not_used(bank1005, tmp_path):
    from eegdash_to_reve import preprocess_recording

    rng = np.random.default_rng(3)
    write_electrodes(tmp_path / "e.tsv", {n: rng.normal(0, 80, 3) for n in KNOWN + ["CB1"]})
    res = preprocess_recording(raw_with(KNOWN + ["CB1"], ["eeg"] * 13), bank1005, electrodes=tmp_path / "e.tsv")
    assert res.aligned == [] and res.dropped == ["CB1"] and res.names == sorted(KNOWN)


def test_a_mirrored_file_is_not_flipped_back(bank1005, tmp_path):
    coords = {n: v * np.array([-1.0, 1.0, 1.0]) for n, v in file_coords(bank1005).items()}   # left and right swapped
    write_electrodes(tmp_path / "e.tsv", coords)
    from eegdash_to_reve import preprocess_recording

    res = preprocess_recording(raw_with(KNOWN + ["CB1"], ["eeg"] * 13), bank1005, electrodes=tmp_path / "e.tsv")
    assert res.aligned == [] and res.dropped == ["CB1"]


def test_too_few_shared_electrodes_are_not_fitted(bank1005, tmp_path):
    from eegdash_to_reve import preprocess_recording

    write_electrodes(tmp_path / "e.tsv", file_coords(bank1005, KNOWN[:7]))
    res = preprocess_recording(raw_with(KNOWN + ["CB1"], ["eeg"] * 13), bank1005, electrodes=tmp_path / "e.tsv")
    assert res.aligned == [] and res.dropped == ["CB1"]


def test_unreadable_file_drops_quietly(bank1005, tmp_path):
    from eegdash_to_reve import preprocess_recording

    (tmp_path / "e.tsv").write_bytes(b"\xff\xfe\x00garbage")
    res = preprocess_recording(raw_with(KNOWN + ["CB1"], ["eeg"] * 13), bank1005, electrodes=tmp_path / "e.tsv")
    assert res.aligned == [] and res.dropped == ["CB1"]
    res = preprocess_recording(raw_with(KNOWN + ["CB1"], ["eeg"] * 13), bank1005, electrodes=tmp_path / "absent.tsv")
    assert res.aligned == [] and res.dropped == ["CB1"]


def test_find_electrodes_tsv_follows_bids_inheritance(tmp_path):
    from eegdash_to_reve import find_electrodes_tsv

    rel = "sub-01/ses-1/eeg/sub-01_ses-1_task-x_run-1_eeg.set"
    eeg = tmp_path / "sub-01/ses-1/eeg"
    eeg.mkdir(parents=True)
    assert find_electrodes_tsv(tmp_path, rel) is None
    (tmp_path / "sub-01/sub-01_electrodes.tsv").write_text("name\tx\ty\tz\n")
    assert find_electrodes_tsv(tmp_path, rel) == tmp_path / "sub-01/sub-01_electrodes.tsv"
    for name in ("sub-02_ses-1_electrodes.tsv", "sub-01_ses-1_space-CapTrak_electrodes.tsv"):
        (eeg / name).write_text("name\tx\ty\tz\n")
    assert find_electrodes_tsv(tmp_path, rel) == eeg / "sub-01_ses-1_space-CapTrak_electrodes.tsv"   # own folder first
    (eeg / "sub-01_ses-1_electrodes.tsv").write_text("name\tx\ty\tz\n")
    assert find_electrodes_tsv(tmp_path, rel) == eeg / "sub-01_ses-1_electrodes.tsv"   # then the one without space-
    (eeg / "sub-01_ses-1_electrodes.tsv").unlink()
    (eeg / "sub-01_ses-1_space-CapTrak_electrodes.tsv").unlink()
    assert find_electrodes_tsv(tmp_path, rel) == tmp_path / "sub-01/sub-01_electrodes.tsv"   # never sub-02's


def test_biosemi160_cap_maps_to_the_160_template(bank256):
    """Rings A-E only: the 160 cap, neither the 128 cap's positions for A-D nor GSN's for E (a GSN-like E* list alone stays GSN)."""
    names = [f"{r}{i}" for r in "ABCDE" for i in range(1, 33)]
    kept, canon, dropped = bank256.resolve(names)
    assert len(kept) == 160 and not dropped and all(c.startswith("biosemi160_") for c in canon)


def test_a_256_cap_with_bad_channels_removed_is_still_a_256_cap(bank256):
    """ds003004 ships 76 A-D, 8 E and a few F-H labels in its sparsest recording: still the 256 cap."""
    names = [f"{r}{i}" for r in "ABCD" for i in range(1, 20)] + [f"E{i}" for i in range(1, 9)] + ["F3", "G12", "H1", "H30"]
    kept, canon, dropped = bank256.resolve(names)
    assert not dropped and all(c.startswith("biosemi256_") for c in canon)


def test_filter_jobs_uses_one_core_above_a_gigabyte():
    """Ten joblib workers each copy the recording: a 3 GB recording was killed in a 19 GB job (2026-10-18)."""
    from eegdash_to_reve import FILTER_ONE_CORE_BYTES, filter_jobs
    assert filter_jobs(64 * 6_100_900 * 8, 10) == 1 and filter_jobs(FILTER_ONE_CORE_BYTES, 10) == 10 and filter_jobs(10, 0) == 1


def test_a_unit_sphere_file_still_places_the_unknown_electrode(bank1005, tmp_path):
    """ds002680 projects its electrodes onto a unit sphere (EEGLAB): 16 mm RMS against the template head, under the
    20 mm bar, so its twelve primed electrodes are placed, each within a couple of centimetres of where it belongs."""
    from eegdash_to_reve import dataset_positions
    centre = np.array([0.0, 0.0, 0.04])   # roughly the head centre of the template
    p3 = bank1005.xyz[bank1005.exact["P3"]] - centre
    true_cb1 = centre + np.linalg.norm(p3) * (p3 + [0.0, -0.04, -0.04]) / np.linalg.norm(p3 + [0.0, -0.04, -0.04])   # on the head, below and behind P3
    on_sphere = {n: (p - centre) / np.linalg.norm(p - centre) for n, p in
                 [(k, bank1005.xyz[bank1005.exact[k]]) for k in KNOWN] + [("CB1", true_cb1)]}
    write_electrodes(tmp_path / "e.tsv", on_sphere)
    names = KNOWN + ["CB1"]
    kept, canon, _ = bank1005.resolve(names)
    idx, out, pos, rms = dataset_positions(tmp_path / "e.tsv", names, kept, canon, bank1005)
    assert out == ["CB1"] and rms is not None and rms < 0.020 and np.linalg.norm(pos[0] - true_cb1) < 0.03
