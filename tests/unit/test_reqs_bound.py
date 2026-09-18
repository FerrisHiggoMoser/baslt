"""Limits on signals: exact limits, strict limits, tolerances, windows, discrete steps, vectors and curves."""

from __future__ import annotations

import numpy as np
import pytest

from reference.reqs_check import one, results

pytestmark = pytest.mark.minimal

T11 = np.arange(11.0)
PEAK = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
T = np.round(np.arange(1001) * 0.01, 10)  # 0 .. 10 s at 100 Hz


def spikes(*spans, height=10.0):
    """Zeros with `height` on the sample ranges [first, last]."""
    x = np.zeros(T.shape[0])
    for first, last in spans:
        x[first:last + 1] = height
    return x


def test_a_value_exactly_at_the_limit_passes_unless_the_limit_is_strict():
    signals = {"x": (T11, PEAK)}
    inclusive = one({"check": "x", "limit": "<= 5"}, signals)
    assert inclusive.verdict == "pass" and inclusive.margin == 0.0 and inclusive.value == 5.0
    assert inclusive.kind == "bound" and inclusive.at == 5.0 and inclusive.limit == "<= 5"
    strict = one({"check": "x", "limit": "< 5"}, signals)
    assert strict.verdict == "fail" and strict.margin == 0.0
    assert strict.totals["runs"] == 1 and strict.runs[0]["worst_t"] == 5.0
    lower = one({"check": "x", "type": "lower", "limit": "0"}, signals)
    assert lower.verdict == "pass" and lower.margin == 0.0
    strict_lower = one({"check": "x", "limit": "> 0"}, signals)
    assert strict_lower.verdict == "fail" and strict_lower.totals["runs"] == 2  # both ends sit on the limit


def test_bare_limits_take_their_direction_from_the_type():
    signals = {"x": (T11, PEAK)}
    assert one({"check": "x", "type": "upper", "limit": "4"}, signals).verdict == "fail"
    assert one({"check": "x", "type": "lower", "limit": "-1"}, signals).verdict == "pass"
    assert one({"check": "x", "limit": "4"}, signals).verdict == "fail"  # upper by default
    result = one({"check": "x", "type": "range", "limit": "4"}, signals)
    assert result.verdict == "error" and "a range check needs a range" in result.reason
    result = one({"check": "x", "type": "lower", "limit": "<= 4"}, signals)
    assert result.verdict == "error" and "cannot use an upper limit" in result.reason


def test_violations_start_and_end_at_the_linear_crossings():
    result = one({"check": "x", "limit": "<= 5.5"}, {"x": (T11, T11)})
    assert result.verdict == "fail"
    run = result.runs[0]
    assert run["start"] == pytest.approx(5.5) and run["end"] == 10.0 and run["flags"] == ["open_end"]
    assert result.first_violation == pytest.approx(5.5)
    assert result.totals["violating_time"] == pytest.approx(4.5)
    assert result.value == 10.0 and result.at == 10.0 and result.margin == pytest.approx(-4.5)
    assert result.margin_pct == pytest.approx(-4.5 / 5.5)


def test_ranges_report_the_side_that_is_exceeded():
    x = np.array([0, -2, 0, 0.5, 0], dtype=float)
    result = one({"check": "x", "limit": "[-1, 1]"}, {"x": (np.arange(5.0), x)})
    assert result.verdict == "fail" and result.value == -2.0 and result.limit_value == -1.0
    assert result.limit == ">= -1" and result.margin == -1.0 and result.runs[0]["limit"] == -1.0
    result = one({"check": "x", "limit": "[-3, 1]"}, {"x": (np.arange(5.0), x)})
    assert result.verdict == "pass" and result.limit_value == 1.0 and result.margin == 0.5


def test_short_violations_within_the_tolerance_pass_and_are_listed():
    x = spikes((100, 104), (500, 519))  # 0.05 s and 0.2 s long
    result = one({"check": "x", "limit": "<= 5", "tolerance": "100 ms"}, {"x": (T, x)})
    assert result.verdict == "fail" and result.totals["runs"] == 2 and result.totals["tolerated"] == 1
    assert [run["tolerated"] for run in result.runs] == [True, False]
    assert result.runs[0]["duration"] == pytest.approx(0.05)
    assert result.first_violation == pytest.approx(4.995)
    assert result.totals["violating_time"] == pytest.approx(0.2)
    short = one({"check": "x", "limit": "<= 5", "tolerance": "100 ms"}, {"x": (T, spikes((100, 104)))})
    assert short.verdict == "pass" and short.margin == -5.0
    assert short.notes == ["1 short violation within the tolerance"]
    warned = one({"check": "x", "limit": "<= 5", "tolerance": "100 ms"}, {"x": (T, spikes((100, 104)))},
                 {"defaults": {"on_tolerated": "warn"}})
    assert warned.verdict == "warn" and "within the tolerance" in warned.reason


def test_without_a_tolerance_every_violating_sample_counts():
    result = one({"check": "x", "limit": "<= 5"}, {"x": (T, spikes((100, 100)))})
    assert result.verdict == "fail" and result.totals["tolerated"] == 0


def test_sample_tolerances_count_samples():
    row = {"check": "x", "limit": "<= 5", "tolerance": "5 samples"}
    assert one(row, {"x": (T, spikes((100, 104)))}).verdict == "pass"
    assert one(row, {"x": (T, spikes((100, 105)))}).verdict == "fail"


def test_the_default_tolerance_comes_from_the_mapping():
    x = spikes((100, 104))
    assert one({"check": "x", "limit": "<= 5"}, {"x": (T, x)}, {"defaults": {"tolerance": "60 ms"}}).verdict == \
        "pass"
    assert one({"check": "x", "limit": "<= 5", "tolerance": "0 s"}, {"x": (T, x)},
               {"defaults": {"tolerance": "60 ms"}}).verdict == "fail"


def test_only_samples_inside_the_window_count():
    x = spikes((100, 150))  # 1.0 .. 1.5 s
    gate = (T >= 3.0).astype(float)
    signals = {"x": (T, x), "gate": (T, gate)}
    result = one({"check": "x", "limit": "<= 5", "when": "gate > 0.5"}, signals)
    assert result.verdict == "pass" and result.totals["active_time"] == pytest.approx(7.0, abs=0.011)
    assert result.value == 0.0
    result = one({"check": "x", "limit": "<= 5", "when": "t < 1.2 s"}, signals)
    assert result.verdict == "fail"
    assert result.runs[0]["end"] == pytest.approx(1.19) and "gap_end" in result.runs[0]["flags"]  # the last sample


def test_a_window_that_never_opens_is_not_applicable():
    result = one({"check": "x", "limit": "<= 5", "when": "x > 100"}, {"x": (T11, PEAK)})
    assert result.verdict == "not_applicable" and result.reason == "the condition never holds"
    assert result.value is None


def test_discrete_signals_hold_their_value_until_the_next_sample():
    mode = np.array([0, 0, 3, 3, 0, 0, 0, 0, 0, 5, 5])
    result = one({"check": "mode", "limit": "<= 2"}, {"mode": (T11, mode, None, "discrete")})
    assert result.verdict == "fail"
    first, second = result.runs
    assert (first["start"], first["end"], first["duration"]) == (2.0, 4.0, 2.0)
    assert (second["start"], second["end"], second["flags"]) == (9.0, 10.0, ["open_end"])
    assert result.value == 5.0 and result.at == 9.0
    assert result.totals["violating_time"] == 3.0 and result.totals["active_time"] == 10.0


def test_vectors_are_checked_per_component():
    position = np.column_stack([T11 * 0.1, T11 * 0.2, PEAK])
    result = one({"check": "pos", "limit": "<= 4.5"}, {"pos": (T11, position, "m", "vector")})
    assert result.verdict == "fail" and result.totals["component"] == 2
    assert result.value == 5.0 and {run["component"] for run in result.runs} == {2}
    assert one({"check": "pos.x", "limit": "<= 4.5"}, {"pos": (T11, position, "m", "vector")}).verdict == "pass"
    norm = one({"check": "norm(pos)", "limit": "<= 10 m"}, {"pos": (T11, position, "m", "vector")})
    assert norm.verdict == "pass" and norm.unit == "m"


def test_curve_limits_report_the_closest_approach():
    mapping = {"curves": {"envelope": {"points": [[0, 10], [10, 6]], "x": "y"}}}
    signals = {"x": (T11, np.full(11, 5.0)), "y": (T11, T11)}
    result = one({"check": "x", "limit": "curve(envelope)"}, signals, mapping)
    assert result.verdict == "pass" and result.margin == pytest.approx(1.0) and result.at == 10.0
    assert result.limit_value == pytest.approx(6.0)
    assert result.limit == "<= curve(envelope)"
    tight = one({"check": "x", "limit": "curve(envelope)"}, {**signals, "x": (T11, np.full(11, 7.0))}, mapping)
    assert tight.verdict == "fail" and tight.first_violation == pytest.approx(7.5)
    assert tight.margin == pytest.approx(-1.0) and tight.notes == []


def test_curves_used_outside_their_points_are_noted():
    mapping = {"curves": {"envelope": {"points": [[0, 10], [5, 8]], "x": "y"}}}
    signals = {"x": (T11, np.full(11, 5.0)), "y": (T11, T11)}
    result = one({"check": "x", "limit": "curve(envelope)"}, signals, mapping)
    assert result.verdict == "pass" and result.limit_value == 8.0
    assert result.notes == ["curve envelope was used outside its range at 5 samples"]


def test_limits_are_converted_into_the_signal_unit():
    q = np.array([0, 4000, 5100, 3000], dtype=float)
    result = one({"check": "q", "limit": "<= 5", "unit": "kPa"}, {"q": (np.arange(4.0), q, "Pa")})
    assert result.verdict == "fail" and result.unit == "Pa" and result.margin == pytest.approx(-100.0)
    assert result.limit_value == 5000.0 and result.limit == "<= 5 kPa"
    result = one({"check": "q", "limit": "<= 5 kPa", "margin": "0.2 kPa"}, {"q": (np.arange(4.0), q * 0.97, "Pa")})
    assert result.verdict == "warn" and result.margin == pytest.approx(53.0)


def test_warning_margins():
    x = np.array([0, 4.9, 0], dtype=float)
    signals = {"x": (np.arange(3.0), x)}
    assert one({"check": "x", "limit": "<= 5", "margin": "0.2"}, signals).verdict == "warn"
    assert one({"check": "x", "limit": "<= 5", "margin": "5 %"}, signals).verdict == "warn"
    assert one({"check": "x", "limit": "<= 5", "margin": "1 %"}, signals).verdict == "pass"
    result = one({"check": "x", "limit": "<= 5", "margin": "0.2"}, signals)
    assert result.reason == "inside the warning margin" and result.margin == pytest.approx(0.1)
    assert result.cases[0].verdict == "warn"


def test_data_gaps_inside_the_window_follow_on_gap():
    x = np.array([0, 1, np.nan, np.nan, 1, 0], dtype=float)
    signals = {"x": (np.arange(6.0), x)}
    row = {"check": "x", "limit": "<= 5"}
    result = one(row, signals)
    assert result.verdict == "warn" and result.totals["gap_time"] == 3.0
    assert result.reason == "data gap of 3 s inside the checked window"
    ignored = one(row, signals, {"defaults": {"on_gap": "ignore"}})
    assert ignored.verdict == "pass" and ignored.notes == ["data gap of 3 s inside the checked window"]
    assert one(row, signals, {"defaults": {"on_gap": "fail"}}).verdict == "fail"
    gated = one({**row, "when": "t > 2.5 s"}, signals)
    assert gated.verdict == "warn" and gated.totals["gap_time"] == 1.0  # only the gap part inside the window
    assert one({**row, "when": "t > 3.5 s"}, signals).verdict == "pass"  # the window opens after the gap


def test_only_nan_data_is_not_applicable():
    result = one({"check": "x", "limit": "<= 5"}, {"x": (np.arange(3.0), np.full(3, np.nan))})
    assert result.verdict == "not_applicable" and result.reason == "no data while the condition holds"


def test_missing_events_warn_unless_the_mapping_says_otherwise():
    mapping = {"events": {"E": {"signal": "x", "rises_above": 100}}}
    row = {"check": "x", "limit": "<= 50", "when": "before('E')"}
    signals = {"x": (T11, PEAK)}
    result = one(row, signals, mapping)
    assert result.verdict == "warn" and result.reason == "event E did not happen" and result.events == ["E"]
    failing = one(row, signals, {**mapping, "defaults": {"on_missing_event": "fail"}})
    assert failing.verdict == "fail"
    na = one(row, signals, {**mapping, "defaults": {"on_missing_event": "na"}})
    assert na.verdict == "pass" and na.notes == ["event E did not happen"]
    hidden = one({"check": "x", "limit": "<= 1", "when": "before('E')"}, signals, mapping)
    assert hidden.verdict == "fail"  # a missing event never hides a failure
    after = one({"check": "x", "limit": "<= 50", "when": "after('E')"}, signals, mapping)
    assert after.verdict == "warn" and after.cases[0].verdict == "not_applicable"


def test_problems_of_one_requirement_do_not_stop_the_others():
    rows = [{"id": "A", "check": "x", "limit": "<= 5 s"}, {"id": "B", "check": "nope", "limit": "<= 5"},
            {"id": "C", "check": "x", "limit": "<= 5"}]
    out = results(rows, {"x": (T11, PEAK, "Pa")})
    assert out["A"].verdict == "error" and "'5 s' is a time but the values are in Pa" in out["A"].reason
    assert out["A"].issues[0].location == "reqs.csv:2:5"  # line 2, column 5 (Limit)
    assert out["B"].verdict == "error" and "unknown name 'nope'" in out["B"].reason
    assert out["C"].verdict == "pass"


def test_limits_that_change_over_time():
    signals = {"x": (T11, PEAK), "cap": (T11, 9 - T11)}
    result = one({"check": "x", "limit": "<= cap"}, signals)
    assert result.verdict == "fail" and result.first_violation == pytest.approx(4.5)
    assert result.at == 5.0 and result.value == 5.0 and result.limit_value == 4.0 and result.margin == -1.0
    assert result.runs[0]["limit"] == 4.0 and result.runs[0]["end"] == 10.0
    assert result.limit == "<= cap"


def test_enum_signals_compare_with_their_codes():
    mode = np.array([0, 1, 2, 1, 0])
    result = one({"check": "mode", "limit": "<= 1"}, {"mode": (np.arange(5.0), mode, None, "discrete")})
    assert result.verdict == "fail" and result.value == 2.0


def test_vectors_stored_as_one_column_per_component():
    split = {"pos_x": (T11, T11 * 0.1, "m"), "pos_y": (T11, T11 * 0.2, "m"), "pos_z": (T11, PEAK, "m")}
    result = one({"check": "pos", "limit": "<= 4.5 m"}, split)
    assert result.verdict == "fail" and result.totals["component"] == 2 and result.unit == "m"
    assert one({"check": "`pos`.z", "limit": "<= 5 m"}, split).verdict == "pass"
    mapping = {"signals": {"alt": {"expr": "`pos`.z", "unit": "m"}, "p": {"path": "pos", "unit": "km"}}}
    assert one({"check": "alt", "type": "max", "limit": ">= 5 m"}, split, mapping).value == 5.0
    assert one({"check": "norm(p)", "type": "max", "limit": "<= 6 km"}, split, mapping).unit == "km"
    bracketed = {"g[0]": (T11, T11), "g[1]": (T11, -T11), "g[2]": (T11, T11 * 2)}
    result = one({"check": "norm(g)", "limit": "<= 30"}, bracketed)
    assert result.verdict == "pass" and result.value == pytest.approx(10 * 6 ** 0.5)
    assert one({"check": "`g`.z", "limit": "<= 25"}, bracketed).value == 20.0
    numbered = {"v_1": (T11, T11), "v_2": (T11, -T11)}
    assert one({"check": "norm(v)", "limit": "<= 15"}, numbered).value == pytest.approx(10 * 2 ** 0.5)
    result = one({"check": "w", "limit": "<= 1"}, {"w_1": (T11, T11)})
    assert result.verdict == "error" and "unknown name 'w'" in result.reason
