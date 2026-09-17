"""Single-number checks (aggregates, event functions, parameters) and event checks."""

from __future__ import annotations

import numpy as np
import pytest

from reference.reqs_check import one

pytestmark = pytest.mark.minimal

T11 = np.arange(11.0)
PEAK = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
THRUST = np.where(T11 < 6, 100.0, 0.0)
MAPPING = {
    "signals": {"mode": {"path": "mode", "kind": "discrete", "labels": {0: "IDLE", 1: "BURN", 2: "COAST"}}},
    "events": {
        "MECO": {"signal": "thrust", "falls_below": 50},
        "peak": {"signal": "x", "rises_above": 4.5},
        "cross": {"signal": "x", "rises_above": 0.5, "hysteresis": 0.1},
        "never": {"signal": "x", "rises_above": 100},
        "late": {"signal": "x", "rises_above": 0.5, "debounce": "20 s"},
        "sep": {"param": "sep_time"},
    },
    "params": {"units": {"mass": "kg"}},
}
SIGNALS = {"x": (T11, PEAK, "m"), "thrust": (T11, THRUST, "kN"), "mode": (T11, np.array([1] * 6 + [2] * 5), None,
                                                                            "discrete")}


def check(row, params=None, mapping=MAPPING):
    return one(row, SIGNALS, mapping, params)


@pytest.mark.parametrize(("row", "value"), [
    ({"check": "x", "type": "max", "limit": "<= 6 m"}, 5.0),
    ({"check": "x", "type": "min", "limit": ">= 0 m"}, 0.0),
    ({"check": "x", "type": "initial", "limit": ">= 0 m"}, 0.0),
    ({"check": "x", "type": "final", "limit": ">= 0 m"}, 0.0),
    ({"check": "x", "type": "mean", "limit": "<= 3 m"}, 2.5),
    ({"check": "max(x)", "limit": "<= 6 m"}, 5.0),
    ({"check": "max(x)", "limit": "<= 6 m", "when": "t <= 3 s"}, 3.0),
    ({"check": "mean(thrust)", "limit": ">= 50 kN", "when": "mode == 'BURN'"}, 100.0),
    ({"check": "integral(x)", "limit": "<= 30"}, 25.0),
    ({"check": "duration(mode == 'COAST')", "limit": ">= 4 s"}, 4.0),
    ({"check": "time('MECO')", "limit": "5 .. 6", "unit": "s"}, 5.5),
    ({"check": "at('MECO', x)", "limit": ">= 4 m"}, 4.5),
    ({"check": "count('peak')", "limit": "= 1"}, 1.0),
    ({"check": "time('MECO') - time('peak')", "limit": "<= 2 s"}, 1.0),
    ({"check": "param.mass", "limit": "<= 2 t"}, 1500.0),
    ({"check": "max(x) / 2", "limit": "<= 3 m"}, 2.5),
])
def test_values(row, value):
    result = check({"limit": "", **row}, {"mass": 1500, "sep_time": 3.0})
    assert result.kind == "value" and result.verdict == "pass", result.reason
    assert result.value == pytest.approx(value)
    assert result.cases[0].value == pytest.approx(value)


def test_value_limits_and_margins():
    result = check({"check": "max(x)", "limit": "<= 4.5 m"})
    assert result.verdict == "fail" and result.reason == "5 is above 4.5"
    assert result.margin == -0.5 and result.margin_pct == pytest.approx(-0.5 / 4.5) and result.limit == "<= 4.5 m"
    result = check({"check": "max(x)", "limit": "5 ± 0.2 m", "margin": "0.25 m"})
    assert result.verdict == "warn" and result.margin == pytest.approx(0.2) and result.limit_value == 5.2
    assert result.reason == "5 is within the warning margin of 5.2"
    result = check({"check": "max(x)", "limit": "(4, 5)"})
    assert result.verdict == "fail" and result.margin == 0.0  # strict: exactly at the limit is beyond it
    assert check({"check": "max(x)", "limit": "== 5"}).verdict == "pass"
    result = check({"check": "max(x)", "limit": "== 4"})
    assert result.verdict == "fail" and result.reason == "5 instead of 4" and result.margin is None
    assert check({"check": "max(x)", "limit": "!= 4"}).verdict == "pass"


def test_values_in_other_units_are_converted():
    result = check({"check": "max(x)", "limit": "<= 0.004 km"})
    assert result.verdict == "fail" and result.limit_value == pytest.approx(4.0) and result.unit == "m"
    result = check({"check": "mean(thrust)", "limit": ">= 40000", "unit": "N"})
    assert result.verdict == "pass" and result.limit_value == pytest.approx(40.0)


def test_missing_values_are_not_applicable_or_warn():
    result = check({"check": "max(x)", "limit": "<= 10 m", "when": "t > 20 s"})
    assert result.verdict == "not_applicable" and result.reason == "the condition never holds"
    result = check({"check": "at('never', x)", "limit": "<= 10 m"})
    assert result.verdict == "warn" and result.reason == "event never did not happen"
    assert result.cases[0].verdict == "not_applicable"
    result = check({"check": "time('never')", "limit": "<= 10 s"}, mapping={**MAPPING, "defaults":
                                                                           {"on_missing_event": "fail"}})
    assert result.verdict == "fail"


def test_parameters():
    assert check({"check": "param.mass", "limit": "<= 1 t"}, {"mass": 1500}).verdict == "fail"
    result = check({"check": "param.mass", "limit": "<= 1 t"}, {"mass": "heavy"})
    assert result.verdict == "error" and "the value is the text 'heavy'" in result.reason
    result = check({"check": "param.mass", "limit": "<= 1 t"}, {"other": 1})
    assert result.verdict == "error" and "unknown run parameter 'mass'" in result.reason


def test_text_parameters_compare_as_text():
    assert check({"check": "param.payload", "limit": "== heavy"}, {"payload": "heavy"}).verdict == "pass"
    assert check({"check": "param.payload", "limit": "== 'heavy'"}, {"payload": "heavy"}).verdict == "pass"
    assert check({"check": "param.payload", "limit": "!= heavy"}, {"payload": "light"}).verdict == "pass"
    result = check({"check": "param.payload", "limit": "== heavy"}, {"payload": "light"})
    assert result.verdict == "fail" and result.value == "light" and result.reason == "'light' is not 'heavy'"
    result = check({"check": "param.payload", "limit": "<= 3"}, {"payload": "light"})
    assert result.verdict == "error" and "compare it with == or !=" in result.reason


def test_value_checks_need_one_number():
    result = check({"check": "x", "type": "value", "limit": "<= 3 m"})
    assert result.verdict == "error" and "needs a single number, such as max(x)" in result.reason
    result = check({"check": "max(x)", "limit": "<= x"})
    assert result.verdict == "error" and "changes over time" in result.reason
    result = check({"check": "max(x)", "limit": "5 m"})
    assert result.verdict == "error" and "write the limit with a comparison" in result.reason
    result = check({"check": "max(x)", "type": "upper", "limit": "<= 3 m"})
    assert result.verdict == "error" and "use the value type" in result.reason


def test_an_event_that_happens_once_inside_its_window_passes():
    result = check({"check": "MECO", "type": "event", "limit": "5 .. 6", "unit": "s"})
    assert result.kind == "event" and result.verdict == "pass" and result.value == 5.5 and result.at == 5.5
    assert result.totals["count"] == 1 and result.totals["times"] == [5.5] and result.events == ["MECO"]
    assert result.margin == pytest.approx(0.5)
    inferred = check({"check": "MECO", "limit": "<= 6 s"})
    assert inferred.kind == "event" and inferred.verdict == "pass"
    bare = check({"check": "MECO"})
    assert bare.verdict == "pass" and bare.limit is None


def test_event_counts_and_times():
    result = check({"check": "cross", "limit": "<= 10 s"})
    assert result.verdict == "pass" and result.value == 0.5
    result = check({"check": "cross", "count": "2"})
    assert result.verdict == "fail" and result.reason == "cross happened 1 time, expected 2"
    result = check({"check": "MECO", "limit": ">= 6 s"})
    assert result.verdict == "fail" and result.reason == "MECO at 5.5 is below 6"
    result = check({"check": "MECO#last", "limit": "<= 6 s"})
    assert result.verdict == "pass" and result.value == 5.5
    result = check({"check": "never", "limit": "<= 6 s"})
    assert result.verdict == "fail" and result.reason == "never happened 0 times, expected 1; never did not happen"
    result = check({"check": "never", "count": "0"})
    assert result.verdict == "pass" and result.value is None


def test_events_still_pending_at_the_end_warn():
    result = check({"check": "late", "count": "0"})
    assert result.verdict == "warn" and "still pending" in result.reason
    assert result.totals["pending_at_end"] is True


def test_events_from_parameters():
    result = check({"check": "sep", "limit": "<= 5 s"}, {"sep_time": 3.0})
    assert result.verdict == "pass" and result.value == 3.0
    result = check({"check": "max(x)", "limit": "<= 10 m", "when": "after('sep')"}, {"sep_time": 3.0})
    assert result.verdict == "pass" and result.value == 5.0
    result = check({"check": "max(x)", "limit": "<= 10 m", "when": "after('sep')"}, {"sep_time": 7.5})
    assert result.value == 2.0  # the samples after 7.5 s


def test_unknown_events_are_errors():
    result = check({"check": "LOS", "type": "event"})
    assert result.verdict == "error" and "unknown event 'LOS'" in result.reason
