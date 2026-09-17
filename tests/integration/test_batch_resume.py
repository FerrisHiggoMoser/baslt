"""Resuming a batch: only runs whose inputs changed are checked again."""

from __future__ import annotations

import json
import os

import pytest

from baslt.reqs.api import check
from reference.batch_runs import write_batch, write_run

pytestmark = pytest.mark.minimal

PEAKS = {"r1": 4.0, "r2": 6.0, "r3": 3.0}


def summary(out):
    data = json.loads((out / "summary.json").read_text())
    for key in ("seconds", "jobs"):
        data.pop(key)
    data["counts"].pop("cached")
    for run in data["runs"]:
        run.pop("cached")
    return data


def cached(result) -> list[str]:
    return [run.id for run in result.batch.runs if run.summary["cached"]]


def test_nothing_changed_nothing_rechecked(tmp_path):
    runs, table, params = write_batch(tmp_path, PEAKS)
    out = tmp_path / "out"
    first = check(runs, table, params=params, output=out, jobs=1)
    before = summary(out)
    again = check(runs, table, params=params, output=out, jobs=1, resume=True)
    assert cached(again) == ["r1", "r2", "r3"] and again.batch.counts()["cached"] == 3
    assert summary(out) == before
    assert again.exit_code == first.exit_code == 1
    assert len((out / "index.jsonl").read_text().splitlines()) == 3


def test_changed_inputs_are_rechecked(tmp_path):
    runs, table, params = write_batch(tmp_path, PEAKS)
    out = tmp_path / "out"
    check(runs, table, params=params, output=out, jobs=1)
    write_run(runs / "r2.csv", 4.0)  # r2 now passes
    stamp = (runs / "r2.csv").stat()
    os.utime(runs / "r2.csv", ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10**9))
    result = check(runs, table, params=params, output=out, jobs=1, resume=True)
    assert cached(result) == ["r1", "r3"]
    assert result.exit_code == 0 and result.batch.stats["PEAK"]["counts"]["fail"] == 0
    params.write_text("run,peak,family\nr1,4,z\nr2,6.0,b\nr3,3,a\n", encoding="utf-8")  # only r1 changes
    assert cached(check(runs, table, params=params, output=out, jobs=1, resume=True)) == ["r2", "r3"]
    table.write_text(table.read_text() + "EXTRA,,x,,<= 100 m,,,,,,,,\n", encoding="utf-8")
    assert cached(check(runs, table, params=params, output=out, jobs=1, resume=True)) == []


def test_a_missing_page_is_made_again(tmp_path):
    runs, table, params = write_batch(tmp_path, PEAKS)
    out = tmp_path / "out"
    check(runs, table, params=params, output=out, jobs=1)
    (out / "runs" / "r2.html").unlink()
    result = check(runs, table, params=params, output=out, jobs=1, resume=True)
    assert cached(result) == ["r1", "r3"] and (out / "runs" / "r2.html").is_file()
    assert cached(check(runs, table, params=params, output=out, jobs=1, resume=True, pages="all")) == ["r2"]
    assert cached(check(runs, table, params=params, output=out, jobs=1, pages="all")) == []


def test_a_damaged_record_is_rechecked(tmp_path):
    runs, table, params = write_batch(tmp_path, PEAKS)
    out = tmp_path / "out"
    check(runs, table, params=params, output=out, jobs=1)
    (out / "runs" / "r1.json").write_text("{not json", encoding="utf-8")
    assert cached(check(runs, table, params=params, output=out, jobs=1, resume=True)) == ["r2", "r3"]
