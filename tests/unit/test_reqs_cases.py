"""Requirements with several cases: rows sharing an ID, each with its own conditions and limit."""

from __future__ import annotations

import numpy as np
import pytest

from reference.reqs_check import results

pytestmark = pytest.mark.minimal

T11 = np.arange(11.0)
PEAK = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
SIGNALS = {"x": (T11, PEAK)}


def two_cases(first: dict, second: dict, params=None, signals=SIGNALS, mapping=None):
    rows = [{"id": "R-1", "check": "x", **first}, {"id": "R-1", **second}]
    return results(rows, signals, mapping, params)["R-1"]


def test_cases_filtered_by_run_parameters():
    heavy = {"case": "heavy", "applies_to": "payload == 'heavy'", "limit": "<= 4"}
    default = {"case": "default", "limit": "<= 6"}
    result = two_cases(heavy, default, {"payload": "heavy"})
    assert result.verdict == "fail" and result.case == "heavy" and result.limit == "<= 4"
    assert [c.verdict for c in result.cases] == ["fail", "not_applicable"]
    assert result.cases[1].reason == "no data while its condition holds"
    assert [c.label for c in result.cases] == ["heavy", "default"] and [c.row for c in result.cases] == [2, 3]
    result = two_cases(heavy, default, {"payload": "light"})
    assert result.verdict == "pass" and result.case == "default" and result.margin == 1.0
    assert result.cases[0].verdict == "not_applicable" and result.cases[0].applies is False
    assert result.cases[0].reason == "Applies to does not match this run"


def test_numeric_parameters_and_missing_parameters():
    big = {"case": "big", "applies_to": "mass > 1 t", "limit": "<= 4"}
    small = {"case": "small", "applies_to": "mass <= 1 t", "limit": "<= 6"}
    mapping = {"params": {"units": {"mass": "kg"}}}
    assert two_cases(big, small, {"mass": 1500}, mapping=mapping).verdict == "fail"
    assert two_cases(big, small, {"mass": 500}, mapping=mapping).verdict == "pass"
    result = two_cases(big, small, {"payload": "x"}, mapping=mapping)
    assert result.verdict == "error" and "unknown run parameter 'mass'" in result.reason


def test_when_no_case_applies_the_requirement_is_not_applicable():
    first = {"case": "a", "applies_to": "payload == 'heavy'", "limit": "<= 4"}
    second = {"case": "b", "applies_to": "payload == 'medium'", "limit": "<= 6"}
    result = two_cases(first, second, {"payload": "light"})
    assert result.verdict == "not_applicable" and result.reason == "no case applies to this run"
    value = results([{"id": "V", "check": "max(x)", "limit": "<= 4", "applies_to": "payload == 'heavy'"}],
                    SIGNALS, None, {"payload": "light"})["V"]
    assert value.verdict == "not_applicable" and value.reason == "Applies to does not match this run"


def test_each_sample_belongs_to_the_first_case_whose_conditions_hold():
    early = {"case": "early", "when": "t < 5 s", "limit": "<= 3"}
    late = {"case": "late", "limit": "<= 10"}
    result = two_cases(early, late)
    assert result.verdict == "fail" and result.case == "early" and result.value == 4.0 and result.at == 4.0
    run = result.runs[0]
    assert run["case"] == "early" and run["start"] == 3.0 and run["end"] == pytest.approx(4 + 1 / 6)
    assert [c.verdict for c in result.cases] == ["fail", "pass"]
    assert result.cases[1].margin == 5.0 and result.cases[1].at == 5.0
    assert result.cases[0].active_time == 4.0 and result.cases[1].active_time == 5.0
    assert result.trace.claims.tolist() == [0] * 5 + [1] * 6


def test_a_violation_continues_across_a_limit_switch_with_the_strictest_tolerance():
    signals = {"x": (T11, np.full(11, 5.0))}
    early = {"case": "early", "when": "t < 5 s", "limit": "<= 4", "tolerance": "20 s"}
    late = {"case": "late", "limit": "<= 4.5"}
    result = two_cases(early, late, signals=signals)
    assert result.verdict == "fail" and result.totals["runs"] == 1
    assert result.runs[0]["duration"] == 10.0 and result.runs[0]["case"] == "early"
    assert result.margin == -1.0
    lenient = two_cases(early, {**late, "tolerance": "20 s"}, signals=signals)
    assert lenient.verdict == "pass" and lenient.totals["tolerated"] == 1


def test_overlapping_cases_can_be_an_error():
    first = {"case": "a", "when": "t < 6.5 s", "limit": "<= 10"}
    second = {"case": "b", "when": "t > 4.5 s", "limit": "<= 10"}  # both hold at 5 s and 6 s
    result = two_cases(first, second)
    assert result.verdict == "pass" and result.totals["overlap_time"] == 1.0
    result = two_cases(first, second, mapping={"defaults": {"on_case_overlap": "error"}})
    assert result.verdict == "error" and result.reason == "cases overlap for 1 s"


def test_single_number_cases_are_checked_one_by_one():
    rows = [{"id": "V", "check": "max(x)", "case": "early", "when": "t < 5 s", "limit": "<= 4"},
            {"id": "V", "case": "late", "when": "t >= 5 s", "limit": "<= 4.5", "severity": "critical"}]
    result = results(rows, SIGNALS)["V"]
    assert result.verdict == "fail" and result.case == "late" and result.value == 5.0 and result.margin == -0.5
    early, late = result.cases
    assert (early.verdict, early.value, early.margin) == ("pass", 4.0, 0.0)
    assert (late.verdict, late.value, late.severity) == ("fail", 5.0, "critical")
    assert result.severity == "critical"


def test_the_headline_is_the_passing_case_with_the_least_margin():
    rows = [{"id": "V", "check": "max(x)", "case": "a", "when": "t < 5 s", "limit": "<= 10"},
            {"id": "V", "case": "b", "when": "t >= 5 s", "limit": "<= 6"}]
    result = results(rows, SIGNALS)["V"]
    assert result.verdict == "pass" and result.case == "b" and result.margin == 1.0


def test_a_broken_case_makes_the_requirement_an_error():
    rows = [{"id": "R", "check": "x", "case": "a", "limit": "<= 4"},
            {"id": "R", "case": "b", "when": "nope > 1", "limit": "<= 6"}]
    result = results(rows, SIGNALS)["R"]
    assert result.verdict == "error" and "unknown name 'nope'" in result.reason
    assert [c.verdict for c in result.cases] == ["pass", "error"]
    assert result.issues[0].path == "R.when" and result.issues[0].location == "reqs.csv:3:7"
