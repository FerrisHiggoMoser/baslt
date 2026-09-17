"""Loading requirement tables: header detection, column mapping, filters, cases, joins and error locations."""

from __future__ import annotations

import csv

import pytest

from baslt.errors import RequirementsError
from baslt.reqs.api import load
from baslt.tabular import Sheet, Styled, write_xlsx

pytestmark = pytest.mark.minimal


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


BASIC = [
    ["Requirements for LV-3", "", "", "", "", ""],
    ["exported 2026-09-01", "", "", "", "", ""],
    ["Req ID", "Name", "Signal", "Type", "Limit", "Unit"],
    ["R1", "Max q", "q", "Upper limit", "<= 70", "kPa"],
    ["", "", "", "", "", ""],
    ["R2", "AoA", "alpha", "range", "[-7, 7]", "deg"],
    ["", "Heading only", "", "", "", ""],
    ["R3", "Unchecked", "", "", "", ""],
]


def test_title_rows_above_the_header_and_synonyms(tmp_path):
    reqs = load(write_csv(tmp_path / "r.csv", BASIC))
    assert reqs.header_row == 3
    assert reqs.columns == {"id": 1, "check": 3, "kind": 4, "limit": 5, "title": 2, "unit": 6}
    assert [r.id for r in reqs.requirements] == ["R1", "R2", "R3"]
    r1 = reqs.requirements[0]
    assert (r1.title, r1.check, r1.kind) == ("Max q", "q", "upper")
    assert r1.cases[0].limit.upper.quantity.text == "70 kPa" and r1.cases[0].label == ""
    assert r1.loc.text == "r.csv:4" and r1.cases[0].cells["limit"] == "r.csv:4:5"
    assert reqs.not_covered == ["R3"] and not reqs.requirements[2].covered
    assert reqs.skipped_rows == 0 and reqs.case_count == 2 and len(reqs.covered) == 2
    assert len(reqs.sha256) == 64


def test_no_header_row(tmp_path):
    with pytest.raises(RequirementsError, match="no header row found in the first 20 rows"):
        load(write_csv(tmp_path / "r.csv", [["a", "b"], ["1", "2"]]))
    with pytest.raises(RequirementsError, match="no ID column found"):
        load(write_csv(tmp_path / "s.csv", [["Check", "Limit"], ["q", "1"]]))


def test_configured_columns_rows_and_vocabulary(tmp_path):
    path = write_csv(tmp_path / "r.csv", [
        ["Key", "What", "Kind of check", "Threshold", "Phase", "State"],
        ["A-1", "q", "Ceiling", "70 kPa", "ascent", "Approved"],
        ["A-2", "q", "Maximum", ">= 1 kPa", "", "draft"],
        ["A-3", "q", "Frobnicate", "70 kPa", "", "approved"],
    ])
    mapping = {"requirements": {
        "header_row": 1,
        "columns": {"id": "Key", "check": "What", "kind": "Kind of check", "limit": "Threshold", "when": "Phase"},
        "where": {"State": ["approved"]},
        "vocab": {"kind": {"frobnicate": "lower"}},
        "passthrough": ["State"],
    }}
    reqs = load(path, mapping=mapping)
    assert [r.id for r in reqs.requirements] == ["A-1", "A-3"] and reqs.skipped_rows == 1
    assert reqs.requirements[0].kind == "upper" and reqs.requirements[0].cases[0].when == "ascent"
    assert reqs.requirements[1].kind == "lower"  # mapped by the vocabulary
    assert reqs.requirements[0].passthrough == {"State": "Approved"}
    with pytest.raises(RequirementsError, match="requirements.where names column 'Status'"):
        load(path, mapping={"requirements": {"where": {"Status": ["x"]}}})
    with pytest.raises(RequirementsError, match="requirements.columns.check names 'Signal'"):
        load(path, mapping={"requirements": {"columns": {"id": "Key", "check": "Signal"}}})


def test_cases_share_an_id(tmp_path):
    path = write_csv(tmp_path / "r.csv", [
        ["ID", "Check", "Type", "Limit", "When", "Case", "Tolerance", "Warn margin"],
        ["L-1", "load", "upper", "5 g0", "relief", "relief", "100 ms", "5 %"],
        ["L-2", "q", "upper", "70 kPa", "", "", "", ""],
        ["L-1", "", "", "6 g0", "ascent", "", "3 samples", ""],
    ])
    reqs = load(path, mapping={"defaults": {"margin": "1 g0"}})
    load_req = reqs.requirements[0]
    assert [c.label for c in load_req.cases] == ["relief", "row 4"] and load_req.rows == [2, 4]
    first, second = load_req.cases
    assert first.tolerance.seconds == pytest.approx(0.1) and first.margin.relative == pytest.approx(0.05)
    assert second.tolerance.samples == 3 and second.margin.absolute.text == "1 g0"  # the default margin
    assert second.limit.upper.quantity.text == "6 g0" and second.when == "ascent"
    assert not load_req.issues and reqs.requirements[1].cases[0].margin.absolute.text == "1 g0"


def test_problems_stay_with_their_requirement(tmp_path):
    path = write_csv(tmp_path / "r.csv", [
        ["ID", "Check", "Type", "Limit", "Min", "Max", "Tolerance", "Count"],
        ["E-1", "q", "upper", "<= 70 kpa", "", "", "", ""],
        ["E-2", "q", "upper", "70", "1", "", "", ""],
        ["E-3", "q", "sideways", "70", "", "", "", ""],
        ["E-4", "q", "upper", "70", "", "", "soon", "x"],
        ["E-5", "q", "", "", "-7", "7", "", ""],
        ["E-6", "q", "upper", "", "", "", "", ""],
        ["E-7", "q", "upper", "70", "", "", "", ""],
        ["E-7", "p", "upper", "80", "", "", "", ""],
        ["", "q", "", "70", "", "", "", ""],
    ])
    reqs = load(path)
    issues = {r.id: [(i.message, i.location) for i in r.issues] for r in reqs.requirements}
    assert issues["E-1"] == [("cannot read limit '<= 70 kpa': unknown unit 'kpa'; did you mean 'kPa'?", "r.csv:2:4")]
    assert "either in the Limit column or in the Min/Max columns" in issues["E-2"][0][0]
    assert "unknown check type 'sideways'" in issues["E-3"][0][0] and issues["E-3"][0][1] == "r.csv:4"
    assert [m for m, _ in issues["E-4"]][0].startswith("cannot read tolerance 'soon'")
    assert "whole number" in [m for m, _ in issues["E-4"]][1]
    assert issues["E-5"] == [] and reqs.requirements[4].cases[0].limit.kind == "range"
    assert issues["E-6"] == [("an upper check needs a limit", "r.csv:7")]
    assert "different checks ('q' in row 8, 'p' in row 9)" in issues["E-7"][0][0]
    assert [w.message for w in reqs.warnings] == ["a row with a check has no ID and was skipped"]


def test_excel_tables_with_numbers_percentages_and_sheets(tmp_path):
    rows = [
        [Styled("ID", "header"), "Check", "Type", "Min", "Max", "Unit", "Warn margin", "Count"],
        ["X-1", "alpha", "range", -7.0, 7.0, "deg", "", ""],
        ["X-2", "q", "upper", "", 70000, "Pa", Styled(0.05, "pct"), 2],
    ]
    path = write_xlsx(tmp_path / "r.xlsx", [
        Sheet("Guide", [["read me"]]), Sheet("Signals", [["Alias", "Path"], ["q", "aero/q"]]),
        Sheet("Checks", rows),
    ])
    reqs = load(path)
    assert reqs.table.sheet == "Checks"  # no other sheet holds requirements
    x1, x2 = reqs.requirements
    assert x1.cases[0].limit.kind == "range" and x1.cases[0].limit.lower.quantity.text == "-7 deg"
    assert x2.cases[0].limit.upper.quantity.text == "70000 Pa"
    assert x2.cases[0].margin.relative == pytest.approx(0.05) and x2.cases[0].count == 2
    assert x2.loc.text == "r.xlsx:Checks!row 3" and x2.cases[0].cells["margin"] == "r.xlsx:Checks!G3"
    assert reqs.config.signals["q"].path == "aero/q"  # config sheets of the same workbook


def test_sheet_choice_prefers_requirements(tmp_path):
    path = write_xlsx(tmp_path / "r.xlsx", [
        Sheet("Overview", [["nothing"]]),
        Sheet("Requirements", [["ID", "Check"], ["R", "q"]]),
    ])
    assert load(path).table.sheet == "Requirements"
    assert load(path, mapping={"requirements": {"sheet": "Overview", "header_row": 1,
                                                "columns": {"id": "nothing"}}}).requirements == []


def test_checks_joined_by_id(tmp_path):
    export = write_csv(tmp_path / "polarion.csv", [
        ["ID", "Title", "Type", "Status"],
        ["P-1", "Max q", "Requirement", "approved"],
        ["P-2", "Modes", "Requirement", "approved"],
        ["P-0", "Loads", "Heading", ""],
        ["P-3", "Draft", "Requirement", "draft"],
    ])
    checks = write_csv(tmp_path / "checks.csv", [
        ["Requirement", "Check", "Limit", "Case"],
        ["P-1", "q", "<= 70 kPa", "ascent"],
        ["P-1", "q", "<= 50 kPa", "coast"],
        ["P-3", "q", "<= 1 kPa", ""],
        ["P-9", "x", "<= 1", ""],
    ])
    mapping = {"requirements": {"where": {"Type": ["Requirement"], "Status": ["approved"]}},
               "checks": {"file": checks.name, "key": "Requirement"}}
    reqs = load(export, mapping=mapping)
    assert [r.id for r in reqs.requirements] == ["P-1", "P-2"]
    p1, p2 = reqs.requirements
    assert p1.title == "Max q" and p1.check == "q" and [c.label for c in p1.cases] == ["ascent", "coast"]
    assert p1.cases[1].loc.text == "checks.csv:3" and p1.loc.text == "polarion.csv:2"
    assert not p2.covered and reqs.not_covered == ["P-2"]
    assert reqs.unmatched_checks == ["P-9"]  # P-3 exists, it is only filtered out
    assert any("P-9" == w.path for w in reqs.warnings)
    assert reqs.skipped_rows == 2


def test_only_filters_ids(tmp_path):
    reqs = load(write_csv(tmp_path / "r.csv", BASIC), only=["R2", "R*3"])
    assert [r.id for r in reqs.requirements] == ["R2", "R3"]


def test_duplicate_headers_and_missing_passthrough(tmp_path):
    path = write_csv(tmp_path / "r.csv", [["ID", "Check", "check", "Owner"], ["R", "q", "p", "me"]])
    reqs = load(path, mapping={"requirements": {"passthrough": ["Owner", "Team"]}})
    assert reqs.requirements[0].check == "q"
    messages = [w.message for w in reqs.warnings]
    assert any("appears twice" in m for m in messages) and "column 'Team' is not in the header row" in messages
    assert reqs.requirements[0].passthrough == {"Owner": "me"}


def test_a_filter_column_is_not_a_check_field(tmp_path):
    path = tmp_path / "polarion.csv"
    path.write_text("ID,Title,Type,Status,Check,Limit\n,Loads,Heading,,,\nLV-1,Max q,Requirement,approved,q,<= 70\n"
                    "LV-2,Prose,Requirement,approved,,\n", encoding="utf-8")
    reqset = load(path, mapping={"requirements": {"where": {"Type": ["Requirement"], "Status": ["approved"]}}})
    assert [(r.id, r.kind, r.issues) for r in reqset.requirements] == [("LV-1", None, []), ("LV-2", None, [])]
    assert "kind" not in reqset.columns
    assert any("filters rows" in w.message for w in reqset.warnings)
    path.write_text("ID,Type,Check,Limit\nLV-1,upper,q,70\nLV-2,assert,q > 1,\n", encoding="utf-8")
    reqset = load(path, mapping={"requirements": {"where": {"Type": ["upper"]}, "columns": {"kind": "Type"}}})
    assert [(r.id, r.kind) for r in reqset.requirements] == [("LV-1", "upper")]
