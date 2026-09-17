"""Verdicts of limit checks agree with a sample-by-sample reading and keep their basic promises."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from reference import reqs_loop as loop  # noqa: E402
from reference.reqs_check import one, results  # noqa: E402

SEVERITY = {"not_applicable": 0, "pass": 1, "warn": 2, "error": 3, "fail": 4}

values = st.one_of(st.integers(-4, 4).map(float), st.floats(-5, 5, allow_nan=False).map(lambda v: round(v, 3)))


@st.composite
def runs(draw, *, gaps=False):
    n = draw(st.integers(2, 24))
    steps = draw(st.lists(st.sampled_from([0.5, 1.0, 0.25, 2.0]), min_size=n - 1, max_size=n - 1))
    t = np.concatenate([[0.0], np.cumsum(steps)])
    element = st.one_of(values, st.just(math.nan)) if gaps else values
    x = np.array(draw(st.lists(element, min_size=n, max_size=n)))
    gate = np.array(draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n)))
    return t, x, gate


limits = st.sampled_from([-2.0, 0.0, 0.5, 1.0, 3.0])
operators = st.sampled_from(["<=", "<"])
tolerances = st.sampled_from([0.0, 0.5, 1.0, 3.0])


@given(runs(), limits, operators, tolerances, st.booleans())
def test_upper_limits_agree_with_the_sample_loop(run, limit, op, tolerance, gated):
    t, x, gate = run
    row = {"check": "x", "limit": f"{op} {limit}", "tolerance": f"{tolerance} s"}
    if gated:
        row["when"] = "gate > 0.5"
    result = one(row, {"x": (t, x), "gate": (t, gate)})
    active = gate > 0.5 if gated else np.ones(t.shape[0], dtype=bool)
    expected = loop.upper_limit_check(t, x, active, limit, strict=op == "<", tolerance=tolerance)
    assert result.verdict == expected["verdict"], (result.reason, expected)
    if expected["margin"] is not None:
        assert result.margin == pytest.approx(expected["margin"], abs=1e-12)
    got = [(run["start"], run["end"], run["tolerated"]) for run in result.runs]
    assert len(got) == len(expected["runs"])
    for (s1, e1, k1), (s2, e2, k2) in zip(got, expected["runs"]):
        assert s1 == pytest.approx(s2, abs=1e-9) and e1 == pytest.approx(e2, abs=1e-9) and k1 == k2


@given(runs(), limits, operators)
def test_without_a_tolerance_a_negative_margin_is_a_failure(run, limit, op):
    t, x, _ = run
    result = one({"check": "x", "limit": f"{op} {limit}"}, {"x": (t, x)})
    if op == "<=":
        assert (result.verdict == "fail") == (result.margin < 0)
    else:
        assert (result.verdict == "fail") == (result.margin <= 0)


@given(runs(gaps=True), limits, tolerances)
def test_a_window_that_always_holds_changes_nothing(run, limit, tolerance):
    t, x, _ = run
    row = {"check": "x", "limit": f"<= {limit}", "tolerance": f"{tolerance} s"}
    plain = one(row, {"x": (t, x)})
    windowed = one({**row, "when": "t >= -1 s"}, {"x": (t, x)})
    assert (plain.verdict, plain.reason) == (windowed.verdict, windowed.reason)
    assert plain.margin == windowed.margin and plain.runs == windowed.runs
    assert plain.totals == windowed.totals


@given(runs(), limits, limits)
def test_complementary_cases_equal_one_switching_limit(run, first, second):
    t, x, gate = run
    signals = {"x": (t, x), "gate": (t, gate)}
    cases = results([{"id": "R", "check": "x", "when": "gate > 0.5", "limit": f"<= {first}"},
                     {"id": "R", "when": "not (gate > 0.5)", "limit": f"<= {second}"}], signals)["R"]
    switching = one({"check": "x", "limit": f"<= where(gate > 0.5, {first}, {second})"}, signals)
    assert cases.verdict == switching.verdict
    assert cases.margin == pytest.approx(switching.margin)
    assert [(r["start"], r["end"]) for r in cases.runs] == [(r["start"], r["end"]) for r in switching.runs]


@given(runs(gaps=True), limits, st.sampled_from([0.0, 0.1, 1.0, 10.0]), tolerances, st.booleans())
def test_raising_a_limit_never_makes_a_check_worse(run, limit, raise_by, tolerance, value_check):
    t, x, _ = run
    signals = {"x": (t, x)}
    if value_check:
        low = one({"check": "max(x)", "limit": f"<= {limit}"}, signals)
        high = one({"check": "max(x)", "limit": f"<= {limit + raise_by}"}, signals)
    else:
        low = one({"check": "x", "limit": f"<= {limit}", "tolerance": f"{tolerance} s"}, signals)
        high = one({"check": "x", "limit": f"<= {limit + raise_by}", "tolerance": f"{tolerance} s"}, signals)
    assert SEVERITY[high.verdict] <= SEVERITY[low.verdict]
    if low.margin is not None:
        assert high.margin == pytest.approx(low.margin + raise_by)
