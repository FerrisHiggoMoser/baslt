"""--archive: a verified .baslt copy of each checked run, made from the requirements."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

import baslt
from baslt.api import verify
from baslt.reqs.api import check, load
from baslt.reqs.archive import derived_policy
from baslt.reqs.bind import bind_requirements
from baslt.sources import open_source
from reference.batch_runs import write_batch
from reference.reqs_check import load_rows

pytestmark = pytest.mark.minimal

ROOT = Path(__file__).resolve().parents[2]


def policy_for(rows, mapping, source, tmp_path):
    reqset = load_rows(rows, mapping, folder=tmp_path)
    infos = open_source(source).list_signals()
    return derived_policy(reqset, bind_requirements(reqset, infos), infos)


def test_the_policy_protects_what_is_checked(tmp_path):
    import numpy as np

    t = np.linspace(0, 10, 11)
    source = {"t": t, "q": t * 1000.0, "alpha": np.sin(t), "mode": (t > 5).astype(np.int64), "odd": t}
    mapping = {"signals": {"qk": {"path": "q", "unit": "kPa"}, "gee": {"path": "odd", "unit": "g0"},
                           "state": {"path": "mode", "kind": "discrete", "labels": {0: "A", 1: "B"}}},
               "events": {"hot": {"signal": "qk", "rises_above": "5 kPa", "debounce": "10 ms"},
                          "weird": {"signal": "gee", "rises_above": "2 g0"}},
               "time": {"signal": "t"}}
    rows = [
        {"id": "A", "check": "qk", "limit": "<= 7", "tolerance": "0.5 s"},
        {"id": "B", "check": "alpha", "limit": "[-2, 2]"},
        {"id": "C", "check": "alpha * 2", "limit": "<= 3"},
        {"id": "D", "check": "qk > 3", "limit": "<= 4 s"},
        {"id": "E", "check": "max(alpha)", "limit": "<= 3"},
        {"id": "F", "check": "state == 'B'", "type": "assert", "when": "t > 6 s"},
        {"id": "G", "check": "gee", "limit": "<= 1"},
        {"id": "H", "check": "nope", "limit": "<= 1"},
    ]
    policy = policy_for(rows, mapping, source, tmp_path)
    assert policy["signals"]["include"] == ["q", "alpha", "mode", "odd"]
    assert policy["signals"]["time"] == "t"
    assert policy["signals"]["decl"] == {"qk": {"path": "q", "unit": "kPa"}, "state": {"path": "mode",
                                                                                       "kind": "discrete"},
                                         "gee": {"path": "odd"}}
    assert policy["hard"] == {
        "qk": {"global_extrema": {}, "violation": [{"above": 7.0, "min_duration": "0.5 s"}],
               "threshold_crossing": [{"value": 3.0}]},
        "alpha": {"global_extrema": {}, "violation": [{"above": 2.0}, {"below": -2.0}]},
        "state": {"state_transitions": {}},
        "gee": {"global_extrema": {}},  # a unit the policy cannot read: no levels
    }
    assert policy["events"] == {"hot": {"when": {"signal": "qk", "rises_above": "5 kPa", "debounce": "10 ms"},
                                        "occurrence": "all", "keep": {"before": "1 s", "after": "1 s"}}}
    compiled = baslt.compile(source, policy)
    assert compiled.status in ("pass", "pass_with_warnings")


def test_a_single_run_archive_verifies_against_the_run(tmp_path):
    spec = importlib.util.spec_from_file_location("rocket_archive_example", ROOT / "examples/rocket_sim.py")
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    run = sim.write_csv(tmp_path / "q_spike.csv", sim.simulate(anomaly="q_spike", rate=200))
    result = check(run, ROOT / "examples/rocket_requirements.csv", mapping=ROOT / "examples/rocket_mapping.json",
                   output=tmp_path / "out", archive=True, html=False, xlsx=False)
    artifact = result.outputs["archive"]
    assert artifact == tmp_path / "out" / "run.baslt"
    report = verify(artifact.read_bytes(), source=run)
    assert report.status in ("pass", "pass_with_warnings"), report.status
    manifest = baslt.read_artifact(artifact.read_bytes()).manifest
    assert "hard.q.violation[0]" in json.dumps(manifest["requirements"])
    assert result.runs[0].issues == []


def test_a_budget_that_cannot_hold_the_run_is_an_issue(tmp_path):
    runs, table, params = write_batch(tmp_path, {"r1": 6.0})
    result = check(runs / "r1.csv", table, output=tmp_path / "out", archive=True, max_size="1 KiB", html=False)
    assert "archive" not in result.outputs
    issues = result.runs[0].issues
    assert len(issues) == 1 and issues[0].message.startswith("the run could not be archived")
    record = json.loads((tmp_path / "out" / "results.json").read_text())
    assert record["issues"][0]["path"] == "archive"


def test_batch_archives_and_resume(tmp_path):
    runs, table, params = write_batch(tmp_path, {"r1": 4.0, "r2": 6.0})
    out = tmp_path / "out"
    result = check(runs, table, params=params, output=out, jobs=1, archive=True)
    assert sorted(p.name for p in (out / "runs").glob("*.baslt")) == ["r1.baslt", "r2.baslt"]
    assert verify((out / "runs" / "r2.baslt").read_bytes(), source=runs / "r2.csv").status in (
        "pass", "pass_with_warnings")
    assert [run.summary["archive"] for run in result.batch.runs] == ["runs/r1.baslt", "runs/r2.baslt"]
    (out / "runs" / "r1.baslt").unlink()
    again = check(runs, table, params=params, output=out, jobs=1, archive=True, resume=True)
    assert [run.id for run in again.batch.runs if run.summary["cached"]] == ["r2"]
    without = check(runs, table, params=params, output=out, jobs=1, resume=True)
    assert [run.summary["cached"] for run in without.batch.runs] == [False, False]  # the settings changed


def test_the_command_line(tmp_path, capsys):
    from baslt.cli import main

    runs, table, _ = write_batch(tmp_path, {"r1": 4.0})
    assert main(["check", str(runs / "r1.csv"), "-r", str(table), "-o", str(tmp_path / "o"), "--archive",
                 "--max-size", "1 MiB"]) == 0
    assert "run.baslt" in capsys.readouterr().out
    assert (tmp_path / "o" / "run.baslt").stat().st_size < 1024 * 1024


def test_reqif_requirements_give_a_csv_checked_copy(tmp_path):
    from reference.reqif_fixtures import write_reqif

    runs, _, _ = write_batch(tmp_path, {"r1": 6.0})
    path = write_reqif(tmp_path / "export.reqif")
    mapping = {"requirements": {"where": {"ReqIF Type": ["Requirement"]}},
               "signals": {"q": {"path": "x", "unit": "kPa"}, "alpha": {"path": "x", "unit": "deg"}}}
    result = check(runs / "r1.csv", path, mapping=mapping, output=tmp_path / "out", html=False)
    assert result.outputs["annotated"].name == "export.checked.csv"
    rows = list(csv.reader(result.outputs["annotated"].open(encoding="utf-8")))
    assert rows[0][:2] == ["ReqIF.ForeignID", "ReqIF.Name"] and rows[0][-4:] == ["Verdict", "Result", "Margin",
                                                                                   "Evidence"]
    verdicts = {row[0]: row[-4] for row in rows[1:] if row[0]}
    assert verdicts == {"LV-001": "PASS", "LV-002": "PASS", "LV-003": "NOT COVERED", "LV-009": "PASS"}
    assert load(path, mapping=mapping).requirements[0].loc.text == "export.reqif:LV-001"
