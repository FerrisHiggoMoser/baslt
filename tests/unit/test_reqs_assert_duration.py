"""Conditions that must hold (assert) and the time a condition holds (duration)."""

from __future__ import annotations

import numpy as np
import pytest

from reference.reqs_check import one

pytestmark = pytest.mark.minimal

T11 = np.arange(11.0)
PEAK = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
MAPPING = {
    "signals": {"mode": {"path": "mode", "kind": "discrete", "labels": {0: "IDLE", 1: "BURN", 2: "COAST"}}},
    "events": {"MECO": {"signal": "thrust", "falls_below": 50}},
    "conditions": {"burning": "mode == 'BURN'", "coast": "after('MECO')"},
}
THRUST = np.where(T11 < 6, 100.0, 0.0)  # falls below 50 between 5 and 6 s: MECO at 5.5 s


def signals(mode):
    return {"mode": (T11, np.asarray(mode), None, "discrete"), "thrust": (T11, THRUST), "x": (T11, PEAK)}


def test_an_assert_passes_while_the_condition_holds_in_its_window():
    result = one({"check": "mode == 'COAST'", "when": "after('MECO')"}, signals([1] * 6 + [2] * 5), MAPPING)
    assert result.kind == "assert" and result.verdict == "pass"
    assert result.value is None and result.margin is None and result.totals["active_time"] == 4.0


def test_an_assert_fails_on_the_first_false_run():
    mode = [1] * 6 + [2, 2, 1, 2, 2]
    result = one({"check": "mode == 'COAST'", "when": "coast", "type": "assert"}, signals(mode), MAPPING)
    assert result.verdict == "fail" and result.reason == "condition failed 1 time for 1 s"
    assert result.at == 8.0 and result.first_violation == 8.0
    assert result.runs == [{"start": 8.0, "end": 9.0, "duration": 1.0, "samples": 1, "worst_t": 8.0,
                            "tolerated": False, "case": "", "component": 0, "flags": []}]
    assert result.cases[0].verdict == "fail" and result.cases[0].runs == 1
    tolerated = one({"check": "mode == 'COAST'", "when": "coast", "tolerance": "1 s"}, signals(mode), MAPPING)
    assert tolerated.verdict == "pass" and tolerated.totals["tolerated"] == 1


def test_an_assert_on_a_continuous_comparison_changes_at_midpoints():
    result = one({"check": "x < 5", "type": "assert"}, signals([0] * 11), MAPPING)
    assert result.verdict == "fail"
    assert (result.runs[0]["start"], result.runs[0]["end"]) == (4.5, 5.5)
    assert result.totals["violating_time"] == 1.0


def test_unknown_parts_of_an_assert_follow_on_gap():
    x = PEAK.copy()
    x[3:5] = np.nan
    row = {"check": "x < 10", "type": "assert"}
    result = one(row, {"x": (T11, x)})
    assert result.verdict == "warn" and result.totals["unknown_time"] == 3.0
    assert "unknown condition of 3 s" in result.reason
    assert one(row, {"x": (T11, x)}, {"defaults": {"on_gap": "ignore"}}).verdict == "pass"


def test_asserts_take_no_limit_and_need_a_condition():
    result = one({"check": "mode == 'COAST'", "type": "assert", "limit": "<= 3"}, signals([0] * 11), MAPPING)
    assert result.verdict == "error" and "takes no limit" in result.reason
    result = one({"check": "x", "type": "assert"}, signals([0] * 11), MAPPING)
    assert result.verdict == "error" and "needs a condition" in result.reason


def test_durations_of_comparisons_use_exact_crossings():
    result = one({"check": "x > 2.5", "limit": "<= 4 s"}, signals([0] * 11), MAPPING)
    assert result.kind == "duration" and result.unit == "s"
    assert result.verdict == "fail" and result.value == pytest.approx(5.0) and result.margin == pytest.approx(-1.0)
    assert result.reason == "5 is above 4"
    assert result.totals["count"] == 1 and result.totals["longest"] == pytest.approx(5.0)
    touching = one({"check": "x >= 5", "limit": "<= 4 s"}, signals([0] * 11), MAPPING)
    assert touching.verdict == "pass" and touching.value == pytest.approx(0.0, abs=1e-9)
    below = one({"check": "x < 2.5", "type": "duration", "limit": "[4.9 s, 5.1 s]"}, signals([0] * 11), MAPPING)
    assert below.verdict == "pass" and below.value == pytest.approx(5.0)


def test_durations_inside_a_window():
    result = one({"check": "x > 2.5", "limit": ">= 2 s", "when": "t < 5 s"}, signals([0] * 11), MAPPING)
    assert result.verdict == "fail" and result.value == pytest.approx(1.5)  # 2.5 s to the window's last sample
    assert result.cases[0].active_time == 4.0


def test_durations_of_states_and_named_conditions():
    mode = [0, 1, 1, 0, 1, 0, 0, 0, 2, 2, 2]
    result = one({"check": "burning", "limit": "<= 2.5 s"}, signals(mode), MAPPING)
    assert result.kind == "duration" and result.verdict == "fail" and result.value == 3.0
    assert result.totals["count"] == 2 and result.totals["longest"] == 2.0
    result = one({"check": "mode == 'COAST'", "limit": ">= 2 s", "type": "duration"}, signals(mode), MAPPING)
    assert result.verdict == "pass" and result.value == 2.0  # the last sample's step ends with the run


def test_a_condition_that_never_holds_lasts_zero_seconds():
    result = one({"check": "x > 100", "limit": ">= 1 s"}, signals([0] * 11), MAPPING)
    assert result.verdict == "fail" and result.value == 0.0 and result.totals["count"] == 0


def test_a_duration_needs_a_limit_and_a_condition():
    result = one({"check": "x > 1", "type": "duration"}, signals([0] * 11), MAPPING)
    assert result.verdict == "error" and "needs a limit" in result.reason
    result = one({"check": "x", "type": "duration", "limit": "<= 1 s"}, signals([0] * 11), MAPPING)
    assert result.verdict == "error" and "needs a condition" in result.reason
    result = one({"check": "x > 1", "limit": "<= 1 kPa"}, signals([0] * 11), MAPPING)
    assert result.verdict == "error" and "kPa" in result.reason
