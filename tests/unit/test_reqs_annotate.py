"""The checked copy of a requirements table: results written next to each requirement, nothing else changed."""

from __future__ import annotations

import csv
import zipfile

import numpy as np
import pytest

from baslt.reqs.config import load_config
from baslt.reqs.load import load_requirements
from baslt.reqs.results import annotate
from baslt.reqs.run import check_run
from reference.reqs_check import write_rows
from reference.reqs_runs import run

pytestmark = pytest.mark.minimal

T11 = np.arange(11.0)
PEAK = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
SIGNALS = {"x": (T11, PEAK, "m")}
HEADER = ["ID", "Title", "Check", "Limit", "Unit", "Case", "When", "Verdict", "Owner"]
BODY = [
    ["", "Heading", "", "", "", "", "", "", ""],
    ["R-1", "Peak early", "x", "<= 3", "m", "early", "t < 5 s", "old", "=SUM(A1:A2)"],
    ["R-1", "Peak late", "", "<= 10", "m", "late", "", "", "-kept"],
    ["R-2", "Mean", "mean(x)", "<= 1", "m", "", "", "", ""],
    ["R-3", "Prose only", "", "", "", "", "", "", ""],
]


def checked(path, mapping=None):
    reqset = load_requirements(path, load_config(mapping or {}, workbook=path))
    result, _ = check_run(run(**SIGNALS), reqset)
    return reqset, {r.id: r for r in result.results}, result


def test_csv_copies_keep_the_layout_and_only_guard_written_cells(tmp_path):
    source = tmp_path / "reqs.csv"
    with open(source, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerows([HEADER, *BODY])
    reqset, results, run_result = checked(source)
    assert reqset.table.delimiter == ";"
    dst, notes = annotate(reqset, results, tmp_path / "reqs.checked.csv", t0=None)
    assert notes == []
    rows = list(csv.reader(dst.open(newline="", encoding="utf-8"), delimiter=";"))
    assert rows[0] == HEADER[:7] + ["Verdict", "Owner", "Result", "Margin", "Evidence"]
    assert rows[1] == BODY[0] + ["", "", ""]
    early, late, mean, prose = rows[2:6]
    assert early[:7] == BODY[1][:7] and early[8] == "=SUM(A1:A2)"  # the user's own cells stay as they are
    assert early[7] == "FAIL" and early[9] == "4 m" and early[10] == "'-1 m (-33.3 %)"
    assert early[11] == "worst 4 m, at 4.000 s, limit <= 3 m, margin -1 m (-33.3 %), (limit exceeded 1 time for " \
                        "1.17 s)"
    assert late[7] == "PASS" and late[8] == "-kept" and late[9] == "5 m" and late[10] == "5 m" and late[11] == ""
    assert mean[7] == "FAIL" and mean[9] == "2.5 m" and mean[11].startswith("value 2.5 m, limit <= 1 m")
    assert prose[7] == "NOT COVERED" and prose[11] == "no check"
    assert run_result.not_covered == ["R-3"]


def test_the_early_case_decides_the_headline(tmp_path):
    source = write_rows(tmp_path / "reqs.csv", [
        {"id": "R-1", "check": "x", "limit": "<= 3", "unit": "m", "case": "early", "when": "t < 5 s"},
        {"id": "R-1", "limit": "<= 10", "unit": "m", "case": "late"}])
    _, results, _ = checked(source)
    result = results["R-1"]
    assert result.verdict == "fail" and result.case == "early" and result.value == 4.0 and result.at == 4.0


def test_written_back_fields_follow_the_mapping(tmp_path):
    source = write_rows(tmp_path / "reqs.csv", [{"id": "R-1", "check": "x", "limit": "<= 3", "unit": "m"}])
    mapping = {"requirements": {"write_back": {"verdict": "Verification Result", "at": "When (T+)",
                                                "margin_pct": "Margin %", "runs": "Runs"}}}
    reqset, results, _ = checked(source, mapping)
    dst, _ = annotate(reqset, results, tmp_path / "out.csv", t0=1.0)
    rows = list(csv.reader(dst.open(newline="", encoding="utf-8")))
    assert rows[0][-4:] == ["Verification Result", "When (T+)", "Margin %", "Runs"]
    assert rows[1][-4:] == ["FAIL", "T+4.000 s", "-0.666667", "1"]  # a number needs no guard


def test_aggregate_evidence_replaces_the_run_sentence(tmp_path):
    source = write_rows(tmp_path / "reqs.csv", [{"id": "R-1", "check": "x", "limit": "<= 3", "unit": "m"}])
    reqset, results, _ = checked(source)
    dst, _ = annotate(reqset, results, tmp_path / "out.csv", aggregate={"R-1": "FAIL in 3/10 runs"})
    rows = list(csv.reader(dst.open(newline="", encoding="utf-8")))
    assert rows[1][-1] == "FAIL in 3/10 runs"


def test_xlsx_copies_change_only_the_requirements_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from baslt.tabular import Sheet, write_xlsx

    source = write_xlsx(tmp_path / "reqs.xlsx", [
        Sheet("Requirements", [HEADER, *BODY]),
        Sheet("Other", [["keep", 1.5]]),
    ])
    reqset, results, _ = checked(source)
    dst, notes = annotate(reqset, results, tmp_path / "reqs.checked.xlsx")
    assert notes == []
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(dst) as after:
        assert before.namelist() == after.namelist()
        changed = [name for name in before.namelist() if before.read(name) != after.read(name)]
    assert changed == ["xl/worksheets/sheet1.xml"]
    book = openpyxl.load_workbook(dst)
    rows = list(book["Requirements"].iter_rows(values_only=True))
    assert rows[0] == (*HEADER, "Result", "Margin", "Evidence")
    assert rows[2][7] == "FAIL" and rows[2][8] == "=SUM(A1:A2)" and rows[2][9] == "4 m"
    assert rows[2][10] == "-1 m (-33.3 %)"  # typed text cells need no guard
    assert book["Requirements"].cell(3, 11).data_type == "s"
    assert rows[5][7] == "NOT COVERED"
    assert list(book["Other"].iter_rows(values_only=True)) == [("keep", 1.5)]
