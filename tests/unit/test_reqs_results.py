"""Result records and files: text formatting, the terminal table, JSON, CSV and the results workbook."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from baslt.reqs.results import evidence_text, fmt_time, fmt_value, render_run, run_sheets, write_csv, write_json
from reference.reqs_check import check

pytestmark = pytest.mark.minimal

T11 = np.arange(11.0)
PEAK = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
MAPPING = {"events": {"go": {"signal": "x", "rises_above": 0.5}}, "time": {"t0": "go"},
           "conditions": {"rising": "t < 5 s", "high": "x > 3 m"}}
ROWS = [
    {"id": "A-1", "title": "Peak", "check": "x", "limit": "<= 4", "unit": "m", "when": "rising or t >= 5 s"},
    {"id": "A-2", "title": "=HYPERLINK(\"x\")", "check": "max(x)", "limit": "<= 5.2", "unit": "m", "margin": "0.5"},
    {"id": "A-3", "title": "Broken", "check": "nope", "limit": "<= 1"},
    {"id": "A-4", "title": "Never", "check": "x", "limit": "<= 1", "when": "t > 50 s"},
    {"id": "A-5", "title": "Fine", "check": "x", "limit": "<= 10", "unit": "m"},
    {"id": "A-6", "title": "Prose only"},
]


@pytest.fixture(scope="module")
def run():
    return check(ROWS, {"x": (T11, PEAK, "m")}, MAPPING)


@pytest.mark.parametrize(("value", "unit", "text"), [
    (66930.0, "Pa", "66.93 kPa"), (2.5e6, "Pa", "2.5 MPa"), (999.0, "Pa", "999 Pa"), (-2660.0, "Pa", "-2.66 kPa"),
    (2152760.0, "N", "2.153 MN"), (46032.4, "m", "46.03 km"), (500.0, "m", "500 m"), (12345.6, None, "12346"),
    (1.5e12, None, "1.5e+12"), (0.000123, "s", "0.000123 s"), (3.0, "1", "3"), (3.0, "?", "3"),
    (math.nan, "m", "n/a"), (math.inf, None, "inf"), (-math.inf, None, "-inf"), (None, "m", "-"),
    ("light", None, "light"), (41.33, "m/s^2", "41.33 m/s^2"),
])
def test_values_are_shown_with_four_digits_and_a_readable_unit(value, unit, text):
    assert fmt_value(value, unit) == text


def test_times_are_shown_relative_to_time_zero():
    assert fmt_time(62.0, 1.5) == "T+60.500 s"
    assert fmt_time(0.5, 1.0) == "T-0.500 s"
    assert fmt_time(3.0) == "3.000 s"
    assert fmt_time(None) == fmt_time(math.nan) == "-"


def test_the_run_record(run):
    counts = run.counts()
    assert counts == {"pass": 1, "warn": 1, "fail": 1, "not_applicable": 1, "error": 1, "not_covered": 1}
    assert run.status == "fail" and run.not_covered == ["A-6"]
    assert run.t0 == 0.5 and run.t0_event == "go" and run.events["go"]["times"] == [0.5]
    assert run.signals == 1 and run.samples == 11 and run.duration == 10.0 and run.format == "numpy"
    by_id = {r.id: r for r in run.results}
    assert by_id["A-1"].context == {"conditions": ["high"], "event": "go", "since": 4.5}


def test_evidence_sentences(run):
    by_id = {r.id: r for r in run.results}
    assert evidence_text(by_id["A-1"], run.t0) == \
        "worst 5 m, at T+4.500 s, limit <= 4 m, margin -1 m (-25.0 %), (limit exceeded 1 time for 2 s)"
    assert evidence_text(by_id["A-2"], run.t0) == \
        "value 5 m, at T+4.500 s, limit <= 5.2 m, margin 0.2 m (3.8 %), (5 is within the warning margin of 5.2)"
    assert evidence_text(by_id["A-3"], run.t0).startswith("not checked: unknown name 'nope'")
    assert evidence_text(by_id["A-4"], run.t0) == "the condition never holds"
    assert evidence_text(by_id["A-5"], None) == "worst 5 m, at 5.000 s, limit <= 10 m, margin 5 m (50.0 %)"


def test_the_terminal_table_lists_problems_first(run):
    text = render_run(run)
    lines = text.splitlines()
    assert lines[0] == "Run      in-memory run   numpy, 10.0 s, 1 signals"
    assert lines[1] == "Checks   reqs.csv   5 requirements, 1 not covered"
    assert lines[3].split()[:3] == ["VERDICT", "ID", "TITLE"]
    assert [line.split()[0] for line in lines[4:8]] == ["FAIL", "ERROR", "WARN", "N/A"]
    assert "A-5" not in text and "reqs.csv:4:3" in text
    assert lines[4].endswith("T+4.500 s  high; 4.5 s after go")
    assert lines[-1] == "Result: FAIL (1 pass, 1 warn, 1 fail, 1 error, 1 n/a)"
    assert "A-5" in render_run(run, show_all=True)


def test_json_output(tmp_path, run):
    path = write_json(tmp_path / "results.json", run)
    data = json.loads(path.read_text())
    assert data["status"] == "fail" and data["counts"]["fail"] == 1 and data["t0"] == 0.5
    first = data["results"][0]
    assert first["id"] == "A-1" and first["verdict"] == "fail" and first["runs"][0]["start"] == 4.0
    assert first["cases"][0]["verdict"] == "fail" and "trace" not in first
    assert data["results"][2]["issues"][0]["location"] == "reqs.csv:4:3"
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_csv_output_is_safe_to_open_in_a_spreadsheet(tmp_path, run):
    path = write_csv(tmp_path / "results.csv", run)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert [row["id"] for row in rows] == ["A-1", "A-2", "A-3", "A-4", "A-5"]
    assert rows[1]["title"] == "'=HYPERLINK(\"x\")"
    assert rows[0]["margin"] == "-1.0" and rows[0]["verdict"] == "FAIL" and rows[0]["at"] == "5.0"
    assert rows[0]["at_t0"] == "4.5" and rows[0]["first_violation"] == "4.0"
    assert rows[0]["limit"] == "<= 4 m" and rows[0]["location"] == "reqs.csv:2"
    assert rows[3]["value"] == "" and rows[3]["verdict"] == "N/A"


def test_the_results_workbook(tmp_path, run):
    openpyxl = pytest.importorskip("openpyxl")
    from baslt.tabular import write_xlsx

    path = write_xlsx(tmp_path / "results.xlsx", run_sheets(run), title="Requirement check")
    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["Summary", "Results", "Cases", "Violations", "Events", "Issues"]
    results = list(book["Results"].iter_rows(values_only=True))
    assert results[0][:4] == ("ID", "Title", "Verdict", "Reason")
    assert results[1][:3] == ("A-1", "Peak", "FAIL") and results[1][9] == -1.0
    assert results[2][1] == "=HYPERLINK(\"x\")" and book["Results"]["B3"].data_type == "s"
    assert results[-1][:3] == ("A-6", None, "NOT COVERED")
    fill = book["Results"]["C2"].fill.fgColor.rgb
    assert fill != book["Results"]["C6"].fill.fgColor.rgb
    violations = list(book["Violations"].iter_rows(values_only=True))
    assert violations[1][:4] == ("A-1", 4.0, 6.0, 2.0)
    summary = dict(book["Summary"].iter_rows(values_only=True))
    assert summary["Status"] == "FAIL" and summary["Not covered"] == "A-6" and summary["Time zero"] == "go at 0.5 s"
    issues = list(book["Issues"].iter_rows(values_only=True))
    assert ("error", "A-3") in [row[:2] for row in issues]
