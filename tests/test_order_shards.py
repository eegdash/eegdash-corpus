"""order_shards.py: measured minutes from earlier logs, largest first, finished shards left out."""

import csv

from order_shards import order


def test_orders_by_measured_minutes_and_skips_done(tmp_path):
    shards = tmp_path / "shards.tsv"
    shards.write_text("shard_id\tdataset\tmode\tsubjects\tn_records\test_hours\n"
                      "a_000\tdsA\toffline\tall\t1\t10\nb_000\tdsB\toffline\tall\t1\t20\n"
                      "c_000\tdsC\toffline\tall\t1\t30\nd_000\tdsD\toffline\tall\t1\t40\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "shard-a_000.out").write_text("noise\ndone dsA: 3 sessions in 1 big recordings, 10 h, 0 skipped, 125.0 s -> x\n")
    (logs / "shard-b_000.out").write_text("done dsB: 1 sessions in 1 big recordings, 20 h, 0 skipped, 3001.5 s -> x\n")
    (logs / "shard-c_000.out").write_text("Traceback: the conversion failed, no done line\n")
    parts = tmp_path / "parts_zarr"
    (parts / "d_000").mkdir(parents=True)
    (parts / "d_000" / "_DONE").write_text("x")                      # finished: left out
    out = tmp_path / "todo.tsv"
    s = order(shards, logs, parts, out)
    rows = list(csv.DictReader(open(out), delimiter="\t"))
    assert [(r["shard_id"], r["est_min"]) for r in rows] == [("c_000", ""), ("b_000", "51"), ("a_000", "3")]
    assert rows[0]["dataset"] == "dsC" and rows[0]["est_hours"] == "30"   # original columns kept
    header = open(out).readline().rstrip("\n").split("\t")
    assert header.count("est_min") == 1 and header[-1] == "est_min" and b"\r" not in out.read_bytes()
    assert s["todo"] == 3 and s["done"] == 1 and s["measured"] == 2 and s["largest"] == 51
