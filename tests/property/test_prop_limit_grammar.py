"""Every way of writing a limit reads back as the same limit."""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from baslt.reqs.limits import parse_limit  # noqa: E402

UNITS = ["", "kPa", "deg", "s", "m/s", "g0", "%", "N*m"]
numbers = st.one_of(st.integers(-10**6, 10**6).map(float),
                    st.floats(-1e6, 1e6, allow_nan=False).map(lambda x: round(x, 4)))


def fmt(x):
    return str(int(x)) if x == int(x) else repr(x)


def sides(spec):
    out = []
    for bound in (spec.lower, spec.upper):
        out.append(None if bound is None else (bound.quantity.value, bound.quantity.unit, bound.inclusive))
    return spec.kind, tuple(out)


@settings(max_examples=200, deadline=None)
@given(numbers, st.sampled_from(UNITS), st.booleans())
def test_upper_and_lower_forms(value, unit, strict):
    u = f" {unit}" if unit else ""
    expected_upper = ("upper", (None, (value, unit or None, not strict)))
    expected_lower = ("lower", ((value, unit or None, not strict), None))
    op = "<" if strict else "<="
    assert sides(parse_limit(f"{op} {fmt(value)}{u}")) == expected_upper
    assert sides(parse_limit(f"{op}{fmt(value)}", unit=unit)) == expected_upper
    op = ">" if strict else ">="
    assert sides(parse_limit(f"{op} {fmt(value)}{u}")) == expected_lower
    if not strict:
        assert sides(parse_limit(f"at most {fmt(value)}{u}")) == expected_upper
        assert sides(parse_limit(f"≥ {fmt(value)}{u}")) == expected_lower


@settings(max_examples=200, deadline=None)
@given(numbers, numbers, st.sampled_from(UNITS))
def test_range_forms(a, b, unit):
    low, high = min(a, b), max(a, b)
    u = f" {unit}" if unit else ""
    expected = ("range", ((low, unit or None, True), (high, unit or None, True)))
    for text in (f"[{fmt(low)}, {fmt(high)}]{u}", f"{fmt(low)} .. {fmt(high)}{u}", f"{fmt(low)} to {fmt(high)}{u}",
                 f"between {fmt(low)} and {fmt(high)}{u}", f"[{fmt(low)}{u}, {fmt(high)}{u}]"):
        assert sides(parse_limit(text)) == expected, text
    center, spread = (low + high) / 2, (high - low) / 2
    parsed = parse_limit(f"{fmt(center)} ± {fmt(spread)}{u}")
    assert parsed.kind == "range"
    assert parsed.lower.quantity.value == pytest.approx(center - spread, abs=1e-9)
    assert parsed.upper.quantity.value == pytest.approx(center + spread, abs=1e-9)
