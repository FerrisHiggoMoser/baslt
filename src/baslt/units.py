"""Physical units used in policies: an explicit symbol table, quantity parsing and conversion.

There is no generic prefix parsing; every accepted symbol is listed in `UNITS` exactly as in
docs/policy.md. Symbols are case-sensitive. SI value = number * factor + offset for absolute
values, or number * factor for differences.
"""

from __future__ import annotations

import difflib
import math
import numbers
import re
from dataclasses import dataclass
from fractions import Fraction

from .errors import Issue, PolicyError


@dataclass(frozen=True, slots=True)
class UnitDef:
    """One unit symbol and its conversion to the SI base unit of its dimension."""

    symbol: str
    dimension: str
    factor: float
    offset: float = 0.0


@dataclass(frozen=True, slots=True)
class Quantity:
    """A parsed policy quantity. `unit` is None for a bare number."""

    value: float
    unit: str | None
    text: str


_DEG = math.pi / 180.0

_TABLE: tuple[tuple[str, tuple[tuple[str, float, float], ...]], ...] = (
    (
        "time",
        (
            ("s", 1.0, 0.0),
            ("ms", 1e-3, 0.0),
            ("us", 1e-6, 0.0),
            ("µs", 1e-6, 0.0),
            ("ns", 1e-9, 0.0),
            ("min", 60.0, 0.0),
            ("h", 3600.0, 0.0),
        ),
    ),
    (
        "angle",
        (
            ("rad", 1.0, 0.0),
            ("mrad", 1e-3, 0.0),
            ("deg", _DEG, 0.0),
            ("°", _DEG, 0.0),
            ("degree", _DEG, 0.0),
            ("degrees", _DEG, 0.0),
        ),
    ),
    ("angular_rate", (("rad/s", 1.0, 0.0), ("deg/s", _DEG, 0.0))),
    (
        "pressure",
        (
            ("Pa", 1.0, 0.0),
            ("hPa", 100.0, 0.0),
            ("kPa", 1e3, 0.0),
            ("MPa", 1e6, 0.0),
            ("bar", 1e5, 0.0),
            ("mbar", 100.0, 0.0),
            ("psi", 6894.757293168361, 0.0),
        ),
    ),
    (
        "length",
        (
            ("m", 1.0, 0.0),
            ("mm", 1e-3, 0.0),
            ("cm", 1e-2, 0.0),
            ("km", 1e3, 0.0),
            ("ft", 0.3048, 0.0),
            ("in", 0.0254, 0.0),
            ("nmi", 1852.0, 0.0),
        ),
    ),
    (
        "velocity",
        (
            ("m/s", 1.0, 0.0),
            ("km/s", 1e3, 0.0),
            ("km/h", 1.0 / 3.6, 0.0),
            ("ft/s", 0.3048, 0.0),
            ("kn", 1852.0 / 3600.0, 0.0),
        ),
    ),
    ("acceleration", (("m/s^2", 1.0, 0.0), ("m/s2", 1.0, 0.0), ("ft/s^2", 0.3048, 0.0))),
    ("force", (("N", 1.0, 0.0), ("kN", 1e3, 0.0), ("MN", 1e6, 0.0), ("lbf", 4.4482216152605, 0.0))),
    ("mass", (("g", 1e-3, 0.0), ("kg", 1.0, 0.0), ("t", 1e3, 0.0))),
    (
        "temperature",
        (("K", 1.0, 0.0), ("degC", 1.0, 273.15), ("degF", 5.0 / 9.0, 255.37222222222223)),
    ),
    ("frequency", (("Hz", 1.0, 0.0), ("kHz", 1e3, 0.0))),
    ("dimensionless", (("1", 1.0, 0.0), ("%", 0.01, 0.0))),
    (
        "bytes",
        (
            ("B", 1.0, 0.0),
            ("KB", 1e3, 0.0),
            ("MB", 1e6, 0.0),
            ("GB", 1e9, 0.0),
            ("KiB", 1024.0, 0.0),
            ("MiB", 1048576.0, 0.0),
            ("GiB", 1073741824.0, 0.0),
        ),
    ),
)

UNITS: dict[str, UnitDef] = {
    symbol: UnitDef(symbol, dimension, factor, offset)
    for dimension, rows in _TABLE
    for symbol, factor, offset in rows
}

DIMENSIONS: tuple[str, ...] = tuple(dimension for dimension, _ in _TABLE)

# Noun phrases used in messages: "'7 deg' is an angle but signal q_dyn is a pressure (Pa)".
DIMENSION_PHRASES: dict[str, str] = {
    "time": "a time",
    "angle": "an angle",
    "angular_rate": "an angular rate",
    "pressure": "a pressure",
    "length": "a length",
    "velocity": "a velocity",
    "acceleration": "an acceleration",
    "force": "a force",
    "mass": "a mass",
    "temperature": "a temperature",
    "frequency": "a frequency",
    "dimensionless": "dimensionless",
    "bytes": "a byte size",
}

_QUANTITY_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S*)")


def _fail(path: str, message: str) -> PolicyError:
    return PolicyError([Issue(path=path, message=message)])


def describe_dimension(dimension: str) -> str:
    """Noun phrase for a dimension, e.g. "an angle"."""
    return DIMENSION_PHRASES.get(dimension, dimension)


def lookup_unit(symbol: str | None) -> UnitDef | None:
    """The definition of a unit symbol, or None for None and for symbols not in the table."""
    if symbol is None:
        return None
    return UNITS.get(symbol)


def suggest_unit(symbol: str) -> str | None:
    """A close known unit symbol for a probable typo, or None."""
    folded = [u for u in UNITS if u.lower() == symbol.lower()]
    if folded:
        return folded[0]
    close = difflib.get_close_matches(symbol, list(UNITS), n=1, cutoff=0.6)
    return close[0] if close else None


def unknown_unit_message(unit: str, text: str) -> str:
    message = f"unknown unit {unit!r} in {text!r}"
    suggestion = suggest_unit(unit)
    if suggestion is not None:
        message += f"; did you mean {suggestion!r}?"
    return message


def parse_quantity(x: str | int | float, *, path: str = "", allow_unknown: bool = False) -> Quantity:
    """Parse `<number> <unit>`, `<number><unit>` or a bare number.

    A unit that is not in the table is an error unless `allow_unknown` is set, in which case
    it is kept so that binding can compare it with an opaque signal unit.
    """
    if isinstance(x, bool):
        raise _fail(path, "expected a number or a quantity such as '100 ms', got a boolean")
    if isinstance(x, numbers.Real):
        try:
            value = float(x)
        except OverflowError:
            raise _fail(path, f"number {x!r} is too large") from None
        if not math.isfinite(value):
            raise _fail(path, f"expected a finite number, got {value!r}")
        text = str(int(x)) if isinstance(x, numbers.Integral) else repr(value)
        return Quantity(value=value, unit=None, text=text)
    if not isinstance(x, str):
        kind = "null" if x is None else type(x).__name__
        raise _fail(path, f"expected a number or a quantity such as '100 ms', got {kind}")
    text = x.strip()
    match = _QUANTITY_RE.fullmatch(text)
    if match is None:
        raise _fail(path, f"expected a number with an optional unit such as '100 ms', got {x!r}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise _fail(path, f"expected a finite number, got {x!r}")
    unit = match.group(2) or None
    if unit is not None and unit not in UNITS and not allow_unknown:
        raise _fail(path, unknown_unit_message(unit, text))
    return Quantity(value=value, unit=unit, text=text)


def convert(
    q: Quantity,
    target_unit: str | None,
    *,
    delta: bool,
    path: str = "",
    expect_dimension: str | None = None,
) -> float:
    """Express `q` in `target_unit`.

    A bare number is returned unchanged (it is already in the target's unit). A unit equal to
    the target symbol is returned unchanged. Otherwise both units must be known and share a
    dimension; `delta` ignores affine offsets.
    """
    if q.unit is None:
        return q.value
    source = lookup_unit(q.unit)
    if expect_dimension is not None and source is not None and source.dimension != expect_dimension:
        raise _fail(
            path,
            f"{q.text!r} is {describe_dimension(source.dimension)} "
            f"but {describe_dimension(expect_dimension)} is required",
        )
    if target_unit is None:
        raise _fail(path, f"{q.text!r} has a unit but the signal has no unit; write a bare number")
    if q.unit == target_unit:
        return q.value
    target = lookup_unit(target_unit)
    if target is None:
        raise _fail(
            path,
            f"{q.text!r} does not match the signal unit {target_unit!r}, which is not a known unit; "
            f"write a bare number or use exactly {target_unit!r}",
        )
    if source is None:
        raise _fail(path, unknown_unit_message(q.unit, q.text))
    if source.dimension != target.dimension:
        raise _fail(
            path,
            f"{q.text!r} is {describe_dimension(source.dimension)} but the signal unit "
            f"{target_unit} is {describe_dimension(target.dimension)}",
        )
    if delta:
        return q.value * source.factor / target.factor
    si = q.value * source.factor + source.offset
    return (si - target.offset) / target.factor


def to_seconds(q: Quantity, *, path: str = "") -> float:
    """A duration in seconds. The quantity must be a time or a bare number (seconds)."""
    if q.unit is None:
        return q.value
    unit = lookup_unit(q.unit)
    if unit is None:
        raise _fail(path, unknown_unit_message(q.unit, q.text))
    if unit.dimension != "time":
        raise _fail(path, f"{q.text!r} is {describe_dimension(unit.dimension)} but a duration (time) is required")
    return q.value * unit.factor


def parse_bytes(x: str | int, *, path: str = "") -> int:
    """A byte count from a plain integer or a byte quantity such as '2 MiB'."""
    if isinstance(x, bool):
        raise _fail(path, "expected a byte size such as '2 MiB' or an integer number of bytes, got a boolean")
    if isinstance(x, numbers.Integral):
        count = int(x)
    elif isinstance(x, str):
        q = parse_quantity(x, path=path)
        if q.unit is None:
            factor = 1.0
        else:
            unit = UNITS[q.unit]
            if unit.dimension != "bytes":
                raise _fail(
                    path, f"{q.text!r} is {describe_dimension(unit.dimension)} but a byte size is required"
                )
            factor = unit.factor
        # Exact decimal arithmetic on the written digits: every byte factor is an exact power of
        # 10 or 2, so '2.2 MB' is exactly 2200000 bytes while '1000000000.4 B' stays fractional
        # and is rejected however large it is.
        digits = _QUANTITY_RE.fullmatch(q.text).group(1)  # type: ignore[union-attr]
        exact = Fraction(digits) * Fraction(factor)
        if exact.denominator != 1:
            raise _fail(path, f"{q.text!r} is not a whole number of bytes")
        count = int(exact)
    else:
        kind = "null" if x is None else type(x).__name__
        raise _fail(path, f"expected a byte size such as '2 MiB' or an integer number of bytes, got {kind}")
    if count < 0:
        raise _fail(path, f"byte size must not be negative, got {x!r}")
    return count
