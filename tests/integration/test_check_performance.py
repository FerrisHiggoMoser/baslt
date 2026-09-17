"""Requirement checks stay fast at realistic sizes (generous bounds; docs/performance.md has the targets)."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from baslt.reqs.api import check, lint, load
from baslt.reqs.run import check_run

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]


def bench():
    spec = importlib.util.spec_from_file_location("bench_check", ROOT / "benchmarks" / "bench_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seconds(fn):
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started


def test_linting_a_large_table(tmp_path):
    b = bench()
    table = b.write_table(tmp_path / "lint.csv", b.requirement_rows(2000, 50))
    result, took = seconds(lambda: lint(table))
    assert result.exit_code == 0 and took < 3.0


def test_one_large_run(tmp_path):
    b = bench()
    data = b.signals(120_000, 50, 1000.0)
    reqs = b.write_table(tmp_path / "reqs.csv", b.requirement_rows(200, 50))
    reqset = load(reqs)
    (result, _), took = seconds(lambda: check_run(data, reqset))
    assert result.error is None and len(result.results) == 200
    assert not [r for r in result.results if r.verdict == "error"]
    assert took < 5.0
    outcome, took = seconds(lambda: check(data, reqs, output=tmp_path / "out"))
    assert took < 8.0 and outcome.outputs["report"].stat().st_size < 1_000_000


def test_a_batch_and_its_resume(tmp_path):
    spec = importlib.util.spec_from_file_location("rocket_batch_perf", ROOT / "examples" / "rocket_batch.py")
    rocket = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rocket)
    rocket.write(tmp_path / "gen", 40, 100, 100.0, "csv")
    args = dict(mapping=ROOT / "examples" / "rocket_mapping.json", params=tmp_path / "gen" / "params.csv",
                output=tmp_path / "out", jobs=4)
    first, took = seconds(lambda: check(tmp_path / "gen" / "runs", ROOT / "examples" / "rocket_requirements.csv",
                                        **args))
    assert first.batch.counts()["runs"] == 40 and took < 60.0
    again, took = seconds(lambda: check(tmp_path / "gen" / "runs", ROOT / "examples" / "rocket_requirements.csv",
                                        resume=True, **args))
    assert again.batch.counts()["cached"] == 40 and took < 10.0
