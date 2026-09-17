"""Checking a batch of runs: verdicts per run, the combined files and the command line."""

from __future__ import annotations

import csv
import json

import pytest

from baslt.cli import main
from baslt.reqs.api import check
from reference.batch_runs import write_batch

pytestmark = pytest.mark.minimal

PEAKS = {"r1": 4.0, "r2": 4.8, "r3": 6.0, "r4": 3.0}


@pytest.fixture()
def batch(tmp_path):
    runs, table, params = write_batch(tmp_path, PEAKS, gate_end={"r4": 9.5})
    return runs, table, params, tmp_path / "out"


def test_every_run_gets_its_verdicts(batch):
    runs, table, params, out = batch
    result = check(runs, table, params=params, output=out, jobs=1)
    assert result.exit_code == 1 and result.status == "fail"
    summary = result.batch
    by_run = {run.id: run.summary for run in summary.runs}
    assert [run.id for run in summary.runs] == ["r1", "r2", "r3", "r4"]
    assert {k: v["status"] for k, v in by_run.items()} == {"r1": "pass", "r2": "warn", "r3": "fail", "r4": "fail"}
    assert by_run["r3"]["results"]["PEAK"]["margin"] == pytest.approx(-1.0)
    assert by_run["r4"]["results"]["GATE"]["v"] == "fail"
    assert by_run["r2"]["params"] == {"peak": 4.8, "family": "b"}
    stats = summary.stats["PEAK"]
    assert stats["counts"]["pass"] == 2 and stats["counts"]["warn"] == 1 and stats["counts"]["fail"] == 1
    assert stats["worst"]["run"] == "r3" and stats["pass_rate"] == 0.5
    assert summary.sentence("PEAK") == "FAIL in 1/4 runs, worst -1 m (r3)"
    assert summary.sentence("MEAN") == "PASS in all 4 runs, least margin 6.181 m (r3)"
    assert summary.counts() == {"pass": 1, "warn": 1, "fail": 2, "not_applicable": 0, "error": 0, "runs": 4,
                                "cached": 0, "not_covered": 0}


def test_the_files(batch):
    runs, table, params, out = batch
    result = check(runs, table, params=params, output=out, jobs=1)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["index.html", "index.jsonl", "reqs.checked.csv", "results.xlsx", "runs", "summary.json"]
    assert sorted(p.name for p in (out / "runs").iterdir()) == [
        "r1.json", "r2.html", "r2.json", "r3.html", "r3.json", "r4.html", "r4.json"]  # pages for problems only
    lines = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    assert [line["run"] for line in lines] == ["r1", "r2", "r3", "r4"] and lines[2]["page"] == "runs/r3.html"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "fail" and summary["counts"]["runs"] == 4
    assert summary["requirements"][0]["sentence"] == "FAIL in 1/4 runs, worst -1 m (r3)"
    record = json.loads((out / "runs" / "r3.json").read_text())
    assert record["status"] == "fail" and record["cache_key"] and record["plots"]["PEAK"]["upper"] == 5.0
    checked = list(csv.reader((out / "reqs.checked.csv").open(encoding="utf-8")))
    assert checked[0][-4:] == ["Verdict", "Result", "Margin", "Evidence"]
    assert checked[1][-4:] == ["FAIL", "6 m", "'-1 m (-20.0 %)", "FAIL in 1/4 runs, worst -1 m (r3)"]
    assert checked[3][-4] == "FAIL" and checked[3][-1] == "FAIL in 1/4 runs"  # GATE: no margin to show
    page = (out / "index.html").read_text(encoding="utf-8")
    assert 'id="manifest"' in page and "FAIL in 1/4 runs" in page
    assert set(result.outputs) == {"report", "xlsx", "annotated", "json", "index"}
    assert "Result: FAIL (4 runs: 1 pass, 1 warn, 2 fail, 0 error, 0 n/a)" in result.text


def test_the_results_workbook(batch):
    openpyxl = pytest.importorskip("openpyxl")
    runs, table, params, out = batch
    check(runs, table, params=params, output=out, jobs=1)
    book = openpyxl.load_workbook(out / "results.xlsx")
    assert book.sheetnames == ["Summary", "Matrix", "Margins", "Requirements", "Violations", "Runs"]
    matrix = list(book["Matrix"].iter_rows(values_only=True))
    assert matrix[0] == ("Run", "Status", "PEAK", "MEAN", "GATE")
    assert matrix[3] == ("r3", "FAIL", "FAIL", "PASS", "PASS")
    margins = list(book["Margins"].iter_rows(values_only=True))
    assert margins[3][2] == pytest.approx(-1.0)
    reqs = list(book["Requirements"].iter_rows(values_only=True))
    assert reqs[1][:10] == ("PEAK", "Peak", None, "x", "FAIL", 2, 1, 1, 0, 0)
    runs_sheet = list(book["Runs"].iter_rows(values_only=True))
    assert runs_sheet[0][-2:] == ("peak", "family") and runs_sheet[3][-2:] == (6.0, "a")
    assert book["Runs"]["J4"].hyperlink.target == "runs/r3.html"
    violations = list(book["Violations"].iter_rows(values_only=True))
    assert [row[:2] for row in violations[1:]] == [("r3", "PEAK"), ("r4", "GATE")]


@pytest.mark.parametrize(("pages", "expected"), [("all", 4), ("none", 0), ("failed", 3)])
def test_pages(batch, pages, expected):
    runs, table, params, out = batch
    check(runs, table, params=params, output=out, jobs=1, pages=pages)
    assert len(list((out / "runs").glob("*.html"))) == expected


def test_run_problems_are_errors_not_crashes(batch):
    runs, table, params, out = batch
    (runs / "broken.mat").write_bytes(b"not a MAT-file")
    result = check(runs, table, params=params, output=out, jobs=1)
    broken = [run for run in result.batch.runs if run.id == "broken"][0]
    assert broken.status == "error" and "MAT" in broken.error
    assert result.exit_code == 1
    assert "ERROR    run broken:" in result.text
    only_broken = check([runs / "broken.mat", runs / "r1.csv"], table, output=out, jobs=1)
    assert only_broken.exit_code == 3 and only_broken.status == "error"


def test_the_command_line(batch, capsys):
    runs, table, params, out = batch
    code = main(["check", str(runs), "-r", str(table), "--params", str(params), "-o", str(out), "--jobs", "1"])
    captured = capsys.readouterr()
    assert code == 1
    assert "[3/4] r3" in captured.err and "FAIL  PEAK" in captured.err
    assert captured.out.splitlines()[0] == "Checks   reqs.csv   3 requirements, 0 not covered"
    table_lines = [line for line in captured.out.splitlines() if line.startswith(("FAIL", "WARN"))]
    assert [line.split()[1] for line in table_lines] == ["PEAK", "GATE"]
    assert main(["check", str(runs / "r1.csv"), str(runs / "r2.csv"), "-r", str(table), "-o", str(out),
                 "--json", "--jobs", "1"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["status"] == "warn" and [r["run"] for r in payload["runs"]] == ["r1", "r2"]
    assert payload["requirements"][0]["id"] == "PEAK"
    assert main(["check", str(runs), "-r", str(table), "-o", str(out), "--jobs", "0"]) == 3
    assert "jobs must be at least 1" in capsys.readouterr().err


def test_the_default_output_folder(batch):
    runs, table, params, _ = batch
    result = check(str(runs / "*.csv"), table, jobs=1)
    assert result.outputs["json"] == runs.parent / "runs.check" / "summary.json"
