"""Limit, tolerance, margin and count cells as engineers write them.

    <= 70 kPa     < 70 kPa     max 70 kPa     at most 70 kPa     not exceed 70 kPa     (upper)
    >= 100 km     > 0 s        min 2 kN       at least 2 kN                            (lower)
    [-7, 7] deg   (0, 1)       [0, 1)         99 .. 102 s    99 to 102 s    99–102 s   (range; brackets set
    between 2 and 3 kN         5.5 ± 1 s      ±5 deg         5.5 +/- 1 s                inclusiveness)
    == 1          != 0                                                                 (equal, not equal)
    70 kPa        (bare: the direction comes from the Type column)
    curve(q_max_vs_mach, mach)     table(q_max_vs_mach)     <= curve(q_max_vs_mach)
    = 0.9 * param.q_design         <= 1.1 * mean(thrust)    (an expression)

A unit written after a range applies to both ends. A unit inside the cell wins over the Unit column, which wins over
the unit of the checked expression (applied when the requirement is bound to a run).
"""

from __future__ import annotations

import math
import re

from ..units import Quantity
from .model import Bound, LimitSpec, Margin, Tolerance
from .units_ext import UnitsError, UnitTable

__all__ = ["LimitError", "parse_count", "parse_limit", "parse_limit_columns", "parse_margin", "parse_tolerance"]

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UPPER_WORDS = ("not to exceed", "shall not exceed", "not exceed", "no more than", "at most", "maximum", "max",
                "below", "under", "less than or equal to")
_LOWER_WORDS = ("no less than", "at least", "minimum", "min", "above", "over", "greater than or equal to")
_OPERATORS = {"<=": "upper", "=<": "upper", "<": "upper_strict", ">=": "lower", "=>": "lower", ">": "lower_strict",
              "==": "equal", "!=": "not_equal", "<>": "not_equal"}
_BRACKET_RE = re.compile(rf"^([\[(])\s*(.+?)\s*[,;]\s*(.+?)\s*([\])])\s*(.*)$")
_BETWEEN_RE = re.compile(r"^between\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE)
_DOTS_RE = re.compile(rf"^({NUMBER})\s*(\S*?)\s*(?:\.\.|…|\bto\b|–|—)\s*({NUMBER})\s*(.*)$", re.IGNORECASE)
_PM_RE = re.compile(rf"^({NUMBER})?\s*(?:±|\+/-|\+-)\s*({NUMBER})\s*(.*)$")
_CURVE_RE = re.compile(r"^(?:curve|table)\s*\(\s*([A-Za-z_][\w.-]*)\s*(?:,\s*(.+?))?\s*\)$", re.IGNORECASE)
_SAMPLES_RE = re.compile(r"^(\d+)\s*(?:samples?|pts?|points?|steps?)$", re.IGNORECASE)
_NONE_WORDS = ("", "-", "none", "n/a", "na", "—")


class LimitError(ValueError):
    """A cell that does not read as a limit, tolerance, margin or count."""


def _clean(text: object, decimal_comma: bool) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("\u2264", "<=").replace("\u2265", ">=").replace("\u2260", "!=").replace("\u00b1", "±")
    raw = re.sub(r"[\u00a0\u2009\u202f\s]+", " ", raw).strip()
    # An en or em dash between two numbers separates a range; any other dash is a minus sign.
    raw = re.sub(r"(\d)\s*[\u2013\u2014]\s*(?=[-+\u2212]?[\d.])", r"\1 .. ", raw)
    raw = raw.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    raw = re.sub(r"(^|[\s(\[,;=<>])([-+]) (?=[\d.])", r"\1\2", raw)  # "- 3" is the number -3
    if decimal_comma:
        raw = re.sub(r"(?<=\d),(?=\d)", ".", raw)
    return raw


def _quantity(text: str, unit: str | None, units: UnitTable) -> Quantity:
    """A number with its own unit, or with `unit` when it has none."""
    q = units.parse(text)
    if q.unit is None and unit:
        if not units.known(unit):
            raise UnitsError(units.unknown(unit))
        return Quantity(value=q.value, unit=unit, text=f"{q.text} {unit}")
    return q


def _rhs(text: str, unit: str | None, units: UnitTable, inclusive: bool) -> Bound:
    """The right-hand side of a comparison: a quantity, a curve or an expression."""
    text = text.strip()
    curve = _CURVE_RE.match(text)
    if curve is not None:
        return Bound(curve=curve.group(1), arg=curve.group(2), inclusive=inclusive)
    if text.startswith("="):
        return Bound(expr=text[1:].strip(), inclusive=inclusive)
    try:
        return Bound(quantity=_quantity(text, unit, units), inclusive=inclusive)
    except UnitsError as exc:
        if re.fullmatch(rf"{NUMBER}(\s*\S+)?", text):  # a number with a bad unit is not an expression
            raise
        return Bound(expr=text, inclusive=inclusive)  # anything else is an expression such as 0.9 * param.q


def _range(low: str, high: str, unit: str | None, units: UnitTable, text: str, *, low_inclusive: bool = True,
           high_inclusive: bool = True) -> LimitSpec:
    lower = Bound(quantity=_quantity(low, unit, units), inclusive=low_inclusive)
    upper = Bound(quantity=_quantity(high, unit, units), inclusive=high_inclusive)
    _check_order(lower, upper, units, text)
    return LimitSpec("range", lower, upper, text)


def _check_order(lower: Bound, upper: Bound, units: UnitTable, text: str) -> None:
    a, b = lower.quantity, upper.quantity
    if a is None or b is None:
        return
    try:
        b_value = units.convert(b, a.unit, delta=False) if a.unit and b.unit else b.value
    except UnitsError as exc:
        raise LimitError(f"the two ends of {text!r} do not match: {exc}") from None
    if a.value > b_value:
        raise LimitError(f"the lower end of {text!r} is above its upper end")


def parse_limit(cell: object, *, unit: str | None = None, units: UnitTable | None = None,
                decimal_comma: bool = False) -> LimitSpec | None:
    """A limit cell as a LimitSpec, or None for an empty cell. Raises LimitError."""
    units = units or UnitTable()
    unit = (unit or "").strip() or None
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        if not math.isfinite(float(cell)):
            raise LimitError(f"a limit must be finite, got {cell!r}")
        q = _quantity(repr(float(cell)) if not float(cell).is_integer() else str(int(cell)), unit, units)
        return LimitSpec("bare", None, Bound(quantity=q), q.text)
    text = _clean(cell, decimal_comma)
    if text.lower() in _NONE_WORDS:
        return None
    try:
        return _parse_text(text, unit, units)
    except UnitsError as exc:
        raise LimitError(f"cannot read limit {text!r}: {exc}") from None


def _parse_text(text: str, unit: str | None, units: UnitTable) -> LimitSpec:
    if text.startswith("=") and not text.startswith(("==", "=<", "=>")):
        rest = text[1:].strip()
        if re.fullmatch(rf"{NUMBER}(\s*\S+)?", rest):
            try:
                return LimitSpec("equal", Bound(quantity=_quantity(rest, unit, units)), None, text)
            except UnitsError:
                pass
        return LimitSpec("bare", None, Bound(expr=rest), text)

    bracket = _BRACKET_RE.match(text)
    if bracket is not None:
        open_, low, high, close, tail = bracket.groups()
        return _range(low, high, tail.strip() or unit, units, text,
                      low_inclusive=open_ == "[", high_inclusive=close == "]")
    between = _BETWEEN_RE.match(text)
    if between is not None:
        low, high = between.groups()
        high_q = units.parse(high)
        return _range(low, high, high_q.unit or unit, units, text)
    dots = _DOTS_RE.match(text)
    if dots is not None:
        low, low_unit, high, tail = dots.groups()
        shared = tail.strip() or low_unit or unit
        return _range(f"{low} {low_unit}".strip() if low_unit else low, high, shared, units, text)
    pm = _PM_RE.match(text)
    if pm is not None:
        center, spread, tail = pm.groups()
        u = tail.strip() or unit
        mid = _quantity(center or "0", u, units)
        width = _quantity(spread, u, units)
        if width.value < 0:
            raise LimitError(f"the spread of {text!r} must not be negative")
        lower = Quantity(mid.value - width.value, mid.unit, f"{_fmt(mid.value - width.value)}{_suffix(mid.unit)}")
        upper = Quantity(mid.value + width.value, mid.unit, f"{_fmt(mid.value + width.value)}{_suffix(mid.unit)}")
        return LimitSpec("range", Bound(quantity=lower), Bound(quantity=upper), text)

    for symbol in sorted(_OPERATORS, key=len, reverse=True):
        if text.startswith(symbol):
            direction = _OPERATORS[symbol]
            rhs = text[len(symbol):].strip()
            if not rhs:
                raise LimitError(f"{text!r} has an operator but no value")
            strict = direction.endswith("_strict")
            bound = _rhs(rhs, unit, units, inclusive=not strict)
            kind = direction.removesuffix("_strict")
            if kind == "upper":
                return LimitSpec("upper", None, bound, text)
            if kind == "lower":
                return LimitSpec("lower", bound, None, text)
            return LimitSpec(kind, bound, None, text)

    lowered = text.lower()
    for words, kind in ((_UPPER_WORDS, "upper"), (_LOWER_WORDS, "lower")):
        for word in words:
            if lowered.startswith(word + " "):
                bound = _rhs(text[len(word):].strip(), unit, units, inclusive=True)
                return LimitSpec(kind, None, bound, text) if kind == "upper" else LimitSpec(kind, bound, None, text)

    return LimitSpec("bare", None, _rhs(text, unit, units, inclusive=True), text)


def _fmt(value: float) -> str:
    return str(int(value)) if value == int(value) and abs(value) < 2**53 else repr(value)


def _suffix(unit: str | None) -> str:
    return f" {unit}" if unit else ""


def parse_limit_columns(lower: object, upper: object, *, unit: str | None = None, units: UnitTable | None = None,
                        decimal_comma: bool = False) -> LimitSpec | None:
    """Separate Min and Max cells as one limit: lower, upper or range."""
    units = units or UnitTable()
    parts: dict[str, Bound] = {}
    for side, cell in (("lower", lower), ("upper", upper)):
        if cell is None or (isinstance(cell, str) and _clean(cell, decimal_comma).lower() in _NONE_WORDS):
            continue
        text = _fmt(float(cell)) if isinstance(cell, (int, float)) else _clean(cell, decimal_comma)
        try:
            parts[side] = _rhs(text, unit, units, inclusive=True)
        except UnitsError as exc:
            raise LimitError(f"cannot read the {side} limit {cell!r}: {exc}") from None
    if not parts:
        return None
    shown = " .. ".join(parts[side].describe() if side in parts else "" for side in ("lower", "upper")).strip()
    if len(parts) == 2:
        _check_order(parts["lower"], parts["upper"], units, shown)
        return LimitSpec("range", parts["lower"], parts["upper"], f"[{parts['lower'].describe()}, "
                                                                   f"{parts['upper'].describe()}]")
    if "lower" in parts:
        return LimitSpec("lower", parts["lower"], None, f">= {parts['lower'].describe()}")
    return LimitSpec("upper", None, parts["upper"], f"<= {parts['upper'].describe()}")


def parse_tolerance(cell: object, *, units: UnitTable | None = None, default: Tolerance | None = None,
                    decimal_comma: bool = False) -> Tolerance:
    """`100 ms`, `0.1` (seconds), `3 samples`; empty gives `default` (no tolerance)."""
    units = units or UnitTable()
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return _seconds(units.parse(cell), str(cell))
    text = _clean(cell, decimal_comma)
    if text.lower() in _NONE_WORDS:
        return default if default is not None else Tolerance()
    samples = _SAMPLES_RE.match(text)
    if samples is not None:
        return Tolerance(seconds=0.0, samples=int(samples.group(1)), text=text)
    try:
        return _seconds(units.parse(text), text)
    except UnitsError as exc:
        raise LimitError(f"cannot read tolerance {text!r}: {exc}; write a duration such as '100 ms' or "
                         "a count such as '3 samples'") from None


def _seconds(q: Quantity, text: str) -> Tolerance:
    table = UnitTable()
    if q.unit is not None and table.dimension(q.unit) != "time":
        raise LimitError(f"tolerance {text!r} must be a duration or a number of samples")
    seconds = q.value if q.unit is None else q.value * table.factor(q.unit, "s")
    if seconds < 0:
        raise LimitError(f"tolerance {text!r} must not be negative")
    return Tolerance(seconds=seconds, samples=None, text=text)


def parse_margin(cell: object, *, units: UnitTable | None = None, decimal_comma: bool = False) -> Margin | None:
    """`5 %` (of the limit), `2 kPa`, a bare number (in the checked unit); empty gives None."""
    units = units or UnitTable()
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return _absolute(units.parse(cell), str(cell))
    text = _clean(cell, decimal_comma)
    if text.lower() in _NONE_WORDS:
        return None
    percent = re.fullmatch(rf"({NUMBER})\s*%", text)
    if percent is not None:
        fraction = float(percent.group(1)) / 100.0
        if fraction < 0:
            raise LimitError(f"margin {text!r} must not be negative")
        return Margin(relative=fraction, text=text)
    try:
        return _absolute(units.parse(text), text)
    except UnitsError as exc:
        raise LimitError(f"cannot read margin {text!r}: {exc}") from None


def _absolute(q: Quantity, text: str) -> Margin:
    if q.value < 0:
        raise LimitError(f"margin {text!r} must not be negative")
    return Margin(absolute=q, text=text)


def parse_count(cell: object) -> int | None:
    if cell is None:
        return None
    if isinstance(cell, float) and not isinstance(cell, bool):
        if cell.is_integer() and cell >= 0:
            return int(cell)
        raise LimitError(f"a count must be a whole number, got {cell!r}")
    text = str(cell).strip()
    if text.lower() in _NONE_WORDS:
        return None
    if not re.fullmatch(r"\d+", text):
        raise LimitError(f"a count must be a whole number, got {text!r}")
    return int(text)

