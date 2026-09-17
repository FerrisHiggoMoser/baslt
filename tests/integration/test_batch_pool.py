"""Checking a batch in several processes gives the same answer, and a crashing run does not stop the batch."""

from __future__ import annotations

import json
import multiprocessing
import os

import pytest

from baslt.reqs import batch as batch_module
from baslt.reqs.api import check
from reference.batch_runs import write_batch

PEAKS = {f"r{k}": 3.0 + 0.5 * k for k in range(8)}


def comparable(out):
    data = json.loads((out / "summary.json").read_text())
    for key in ("seconds", "jobs"):
        data.pop(key)
    return data


def test_processes_give_the_same_results(tmp_path):
    runs, table, params = write_batch(tmp_path, PEAKS)
    one = check(runs, table, params=params, output=tmp_path / "one", jobs=1)
    many = check(runs, table, params=params, output=tmp_path / "many", jobs=3)
    assert comparable(tmp_path / "one") == comparable(tmp_path / "many")
    assert one.exit_code == many.exit_code == 1 and many.batch.jobs == 3
    for name in PEAKS:
        a = json.loads((tmp_path / "one" / "runs" / f"{name}.json").read_text())
        b = json.loads((tmp_path / "many" / "runs" / f"{name}.json").read_text())
        for record in (a, b):
            record.pop("timing")
        assert a == b


def crashing_worker(task):
    if task["id"] == "r5":
        os._exit(13)  # the process dies the way a native crash would end it
    return batch_module.check_one(task)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs the fork start method")
def test_a_crashing_run_becomes_an_error(tmp_path):
    runs, table, params = write_batch(tmp_path, PEAKS)
    result = check(runs, table, params=params, output=tmp_path / "out", jobs=3, _worker=crashing_worker,
                   _context="fork")
    statuses = {run.id: run.status for run in result.batch.runs}
    assert statuses["r5"] == "error"
    assert result.batch.runs[5].error == "the worker process stopped while checking this run"
    assert all(status != "error" for name, status in statuses.items() if name != "r5")
    assert result.exit_code == 1


def test_the_worker_setup_limits_threads(monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(os, "nice", lambda value: 0)
    batch_module._init_worker()
    assert os.environ["OMP_NUM_THREADS"] == "1" and os.environ["OPENBLAS_NUM_THREADS"] == "1"
