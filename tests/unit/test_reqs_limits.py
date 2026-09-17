"""Limit, tolerance, margin and count cells in the forms engineers write them."""

from __future__ import annotations

import re

import pytest

from baslt.reqs.limits import (
    LimitError,
    parse_count,
    parse_limit,
    parse_limit_columns,
    parse_margin,
    parse_tolerance,
)
from baslt.reqs.units_ext import UnitTable
from baslt.units import UnitDef

pytestmark = pytest.mark.minimal


def shape(spec):
    """(kind, (lower text, inclusive) | None, (upper text, inclusive) | None)."""
    if spec is None:
        return None
    side = lambda b: None if b is None else (b.describe(), b.inclusive)  # noqa: E731
    return spec.kind, side(spec.lower), side(spec.upper)


@pytest.mark.parametrize(("cell", "expected"), [
    ("<= 70 kPa", ("upper", None, ("70 kPa", True))),
    ("≤70 kPa", ("upper", None, ("70 kPa", True))),
    ("< 70", ("upper", None, ("70", False))),
    ("=< 70", ("upper", None, ("70", True))),
    ("max 70 kPa", ("upper", None, ("70 kPa", True))),
    ("Not to exceed 70 kPa", ("upper", None, ("70 kPa", True))),
    ("at most 3 s", ("upper", None, ("3 s", True))),
    (">= 100 km", ("lower", ("100 km", True), None)),
    ("> 0 s", ("lower", ("0 s", False), None)),
    ("at least 2 kN", ("lower", ("2 kN", True), None)),
    ("[-7, 7] deg", ("range", ("-7 deg", True), ("7 deg", True))),
    ("[-7 deg; 7 deg]", ("range", ("-7 deg", True), ("7 deg", True))),
    ("(0, 1)", ("range", ("0", False), ("1", False))),
    ("[0, 1)", ("range", ("0", True), ("1", False))),
    ("99 .. 102 s", ("range", ("99 s", True), ("102 s", True))),
    ("99 to 102 s", ("range", ("99 s", True), ("102 s", True))),
    ("99–102 s", ("range", ("99 s", True), ("102 s", True))),
    ("−5 – 5", ("range", ("-5", True), ("5", True))),
    ("between 2 and 3 kN", ("range", ("2 kN", True), ("3 kN", True))),
    ("5.5 ± 1 s", ("range", ("4.5 s", True), ("6.5 s", True))),
    ("±5 deg", ("range", ("-5 deg", True), ("5 deg", True))),
    ("5.5 +/- 1 s", ("range", ("4.5 s", True), ("6.5 s", True))),
    ("== 1", ("equal", ("1", True), None)),
    ("= 1", ("equal", ("1", True), None)),
    ("!= 0", ("not_equal", ("0", True), None)),
    ("70 kPa", ("bare", None, ("70 kPa", True))),
    ("5 %", ("bare", None, ("5 %", True))),
    ("4.5 g0", ("bare", None, ("4.5 g0", True))),
    ("curve(q_max_vs_mach, mach)", ("bare", None, ("curve(q_max_vs_mach, mach)", True))),
    ("TABLE( q_max )", ("bare", None, ("curve(q_max)", True))),
    ("<= curve(q_max)", ("upper", None, ("curve(q_max)", True))),
    ("= 0.9 * param.q_design", ("bare", None, ("= 0.9 * param.q_design", True))),
    ("<= 1.1 * mean(thrust)", ("upper", None, ("= 1.1 * mean(thrust)", True))),
    ("> ref_limit", ("lower", ("= ref_limit", False), None)),
    ("", None), ("  ", None), ("-", None), ("n/a", None), (None, None),
])
def test_limit_forms(cell, expected):
    assert shape(parse_limit(cell)) == expected


def test_unit_column_and_numbers():
    assert shape(parse_limit("<= 70", unit="kPa")) == ("upper", None, ("70 kPa", True))
    assert shape(parse_limit("<= 70 Pa", unit="kPa")) == ("upper", None, ("70 Pa", True))  # the cell wins
    assert shape(parse_limit(70.0, unit="kPa")) == ("bare", None, ("70 kPa", True))
    assert shape(parse_limit(0.25)) == ("bare", None, ("0.25", True))
    assert shape(parse_limit("[1, 2]", unit="s")) == ("range", ("1 s", True), ("2 s", True))
    assert parse_limit("1,5 .. 2,5 kPa", decimal_comma=True).upper.quantity.value == 2.5


def test_user_units():
    units = UnitTable({"g": UnitDef("g", "acceleration", 9.80665)})
    spec = parse_limit("<= 5 g", units=units)
    assert units.convert(spec.upper.quantity, "m/s^2", delta=False) == pytest.approx(49.03325)


@pytest.mark.parametrize(("cell", "message"), [
    ("<= 70 kpa", "unknown unit 'kpa'; did you mean 'kPa'?"),
    ("[7, -7]", "lower end of '[7, -7]' is above its upper end"),
    ("[1 s, 2 kPa]", "do not match"),
    ("<=", "has an operator but no value"),
    ("5 ± -1", "must not be negative"),
    (float("nan"), "must be finite"),
])
def test_limit_errors(cell, message):
    with pytest.raises(LimitError, match=re.escape(message)):
        parse_limit(cell)
    with pytest.raises(LimitError, match="unknown unit 'furlong'"):
        parse_limit("<= 5", unit="furlong")


def test_min_max_columns():
    assert shape(parse_limit_columns(-7.0, 7.0, unit="deg")) == ("range", ("-7 deg", True), ("7 deg", True))
    assert shape(parse_limit_columns(None, "70 kPa")) == ("upper", None, ("70 kPa", True))
    assert shape(parse_limit_columns("2", "", unit="kN")) == ("lower", ("2 kN", True), None)
    assert parse_limit_columns(None, "  ") is None
    with pytest.raises(LimitError, match="above its upper end"):
        parse_limit_columns(5.0, 1.0)


def test_tolerances():
    assert parse_tolerance("100 ms").seconds == pytest.approx(0.1)
    assert parse_tolerance("0.2").seconds == 0.2 and parse_tolerance(0.2).seconds == 0.2
    assert parse_tolerance("3 samples").samples == 3 and parse_tolerance("1 sample").samples == 1
    assert parse_tolerance("2 min").seconds == 120.0
    assert parse_tolerance("").seconds == 0.0
    default = parse_tolerance("50 ms")
    assert parse_tolerance(None, default=default) is default
    for bad, message in (("5 kPa", "must be a duration"), ("-1 s", "must not be negative"), ("soon", "cannot read")):
        with pytest.raises(LimitError, match=message):
            parse_tolerance(bad)


def test_margins():
    assert parse_margin("5%").relative == pytest.approx(0.05)
    assert parse_margin("2.5 %").relative == pytest.approx(0.025)
    assert parse_margin("2 kPa").absolute.text == "2 kPa"
    assert parse_margin(0.5).absolute.value == 0.5
    assert parse_margin("") is None and parse_margin("none") is None
    with pytest.raises(LimitError, match="must not be negative"):
        parse_margin("-1 kPa")
    with pytest.raises(LimitError, match="cannot read margin"):
        parse_margin("lots")


def test_counts():
    assert parse_count("3") == 3 and parse_count(2.0) == 2 and parse_count("") is None and parse_count(None) is None
    for bad in ("1.5", "-1", "once", 1.5):
        with pytest.raises(LimitError, match="whole number"):
            parse_count(bad)
