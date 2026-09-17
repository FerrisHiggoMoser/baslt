"""The batch example: its runs, checked with the example requirements, fail exactly where the generator says."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from baslt.reqs.api import check

pytestmark = pytest.mark.minimal

ROOT = Path(__file__).resolve().parents[2]


def example():
    spec = importlib.util.spec_from_file_location("rocket_batch_example", ROOT / "examples/rocket_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_batch_matches_the_generator(tmp_path, capsys):
    batch = example()
    assert batch.main(["--runs", "12", "--out", str(tmp_path / "gen"), "--rate", "100", "--format", "csv"]) == 0
    assert "wrote 12 runs" in capsys.readouterr().out
    params = list(csv.DictReader((tmp_path / "gen" / "params.csv").open(encoding="utf-8")))
    assert [row["run"] for row in params] == [f"run_{k:04d}" for k in range(12)]
    assert params[5]["anomaly"] == "q_spike" and params[11]["anomaly"] == "alpha_chatter"
    expected = {row.pop("run"): row for row in csv.DictReader((tmp_path / "gen" / "expected.csv").open())}
    result = check(tmp_path / "gen" / "runs", ROOT / "examples/rocket_requirements.csv",
                   mapping=ROOT / "examples/rocket_mapping.json", params=tmp_path / "gen" / "params.csv",
                   output=tmp_path / "out", jobs=2, pages="none")
    got = {run.id: {rid: run.summary["results"][rid]["v"] for rid in ("LV-007", "LV-005", "LV-001")}
           for run in result.batch.runs}
    assert got == expected
    assert got["run_0005"]["LV-001"] == "fail"
    assert any(row["LV-007"] == "fail" for row in expected.values())
    assert any(row["LV-007"] == "pass" for row in expected.values())
    assert result.batch.runs[0].summary["params"]["payload"] == float(params[0]["payload"])


def test_the_plan_is_deterministic():
    batch = example()
    assert batch.plan(5, 100) == batch.plan(5, 100)
    assert batch.plan(5, 100) != batch.plan(5, 101)
