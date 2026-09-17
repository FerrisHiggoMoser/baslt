"""Units for requirement checks: the policy table plus the units engineering limits need, and user units.

The policy unit table (`baslt.units.UNITS`) is mirrored by the verifier and stays as it is. Requirements need
more: load factors (`g0`), torques, flows, powers, rates. `UnitTable` looks a symbol up in the mapping's own
`units:` first (which may redefine `g` as a load factor), then in `EXTRA_UNITS`, then in the policy table.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping

from ..units import DIMENSION_PHRASES, UNITS, Quantity, UnitDef

__all__ = ["DERIVATIVE_UNITS", "EXTRA_UNITS", "UnitTable", "UnitsError"]

_DEG = math.pi / 180.0
G0 = 9.80665

_EXTRA: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
    ("acceleration", (("g0", G0), ("m/s²", 1.0))),
    ("torque", (("N*m", 1.0), ("N·m", 1.0), ("Nm", 1.0), ("kN*m", 1e3), ("kN·m", 1e3), ("kNm", 1e3))),
    ("mass_flow", (("kg/s", 1.0), ("g/s", 1e-3))),
    ("power", (("W", 1.0), ("kW", 1e3), ("MW", 1e6))),
    ("energy", (("J", 1.0), ("kJ", 1e3), ("MJ", 1e6))),
    ("voltage", (("V", 1.0), ("mV", 1e-3), ("kV", 1e3))),
    ("current", (("A", 1.0), ("mA", 1e-3))),
    ("heat_flux", (("W/m^2", 1.0), ("W/m2", 1.0), ("kW/m^2", 1e3), ("kW/m2", 1e3))),
    ("angular_acceleration", (("rad/s^2", 1.0), ("rad/s2", 1.0), ("deg/s^2", _DEG), ("deg/s2", _DEG))),
    ("pressure_rate", (("Pa/s", 1.0), ("kPa/s", 1e3), ("MPa/s", 1e6))),
    ("angular_rate", (("rpm", 2.0 * math.pi / 60.0), ("°/s", _DEG))),
    ("frequency", (("1/s", 1.0),)),
    ("dimensionless", (("-", 1.0),)),
)

EXTRA_UNITS: dict[str, UnitDef] = {
    symbol: UnitDef(symbol, dimension, factor) for dimension, rows in _EXTRA for symbol, factor in rows
}

EXTRA_PHRASES: dict[str, str] = {
    "torque": "a torque",
    "mass_flow": "a mass flow",
    "power": "a power",
    "energy": "an energy",
    "voltage": "a voltage",
    "current": "a current",
    "heat_flux": "a heat flux",
    "angular_acceleration": "an angular acceleration",
    "pressure_rate": "a pressure rate",
}

# unit -> unit of its time derivative (and the reverse for integrals)
DERIVATIVE_UNITS: dict[str, str] = {
    "m": "m/s", "km": "km/s", "ft": "ft/s", "m/s": "m/s^2", "ft/s": "ft/s^2",
    "rad": "rad/s", "deg": "deg/s", "°": "°/s", "rad/s": "rad/s^2", "deg/s": "deg/s^2",
    "Pa": "Pa/s", "kPa": "kPa/s", "MPa": "MPa/s", "kg": "kg/s", "g": "g/s", "J": "W", "kJ": "kW", "MJ": "MW",
}
INTEGRAL_UNITS: dict[str, str] = {derived: base for base, derived in DERIVATIVE_UNITS.items()}
INTEGRAL_UNITS.update({"m/s2": "m/s", "rad/s2": "rad/s", "deg/s2": "deg/s", "m/s²": "m/s"})

_QUANTITY_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S.*)?")


class UnitsError(ValueError):
    """A unit that is unknown, or does not fit where it is used."""


class UnitTable:
    """Unit lookup and conversion for requirement checks."""

    def __init__(self, extra: Mapping[str, UnitDef] | None = None) -> None:
        self.user: dict[str, UnitDef] = dict(extra or {})

    def lookup(self, symbol: str | None) -> UnitDef | None:
        if symbol is None:
            return None
        return self.user.get(symbol) or EXTRA_UNITS.get(symbol) or UNITS.get(symbol)

    def known(self, symbol: str | None) -> bool:
        return self.lookup(symbol) is not None

    def symbols(self) -> list[str]:
        return list(dict.fromkeys([*self.user, *EXTRA_UNITS, *UNITS]))

    def dimension(self, symbol: str | None) -> str | None:
        unit = self.lookup(symbol)
        return None if unit is None else unit.dimension

    def phrase(self, dimension: str) -> str:
        return DIMENSION_PHRASES.get(dimension) or EXTRA_PHRASES.get(dimension) or f"a {dimension.replace('_', ' ')}"

    def unknown(self, symbol: str) -> str:
        close = difflib.get_close_matches(symbol, self.symbols(), n=1, cutoff=0.6)
        hint = f"; did you mean {close[0]!r}?" if close else "; declare it under units: in the mapping"
        return f"unknown unit {symbol!r}{hint}"

    def parse(self, text: str | float | int) -> Quantity:
        """A quantity `<number> <unit>` whose unit this table knows, or a bare number."""
        if isinstance(text, bool):
            raise UnitsError("expected a number, got a boolean")
        if isinstance(text, (int, float)):
            value = float(text)
            if not math.isfinite(value):
                raise UnitsError(f"expected a finite number, got {text!r}")
            shown = str(int(value)) if value == int(value) and abs(value) < 2**53 else repr(value)
            return Quantity(value=value, unit=None, text=shown)
        cleaned = normalize_number_text(text)
        match = _QUANTITY_RE.fullmatch(cleaned)
        if match is None:
            raise UnitsError(f"expected a number with an optional unit such as '70 kPa', got {text!r}")
        value = float(match.group(1))
        if not math.isfinite(value):
            raise UnitsError(f"expected a finite number, got {text!r}")
        unit = (match.group(2) or "").strip() or None
        if unit is not None and not self.known(unit):
            raise UnitsError(self.unknown(unit))
        return Quantity(value=value, unit=unit, text=cleaned)

    def convert(self, q: Quantity, target: str | None, *, delta: bool) -> float:
        """Express `q` in `target`. A bare number is taken to be in the target unit already."""
        if q.unit is None or q.unit == target:
            return q.value
        source = self.lookup(q.unit)
        if source is None:
            raise UnitsError(self.unknown(q.unit))
        if target is None:
            raise UnitsError(f"{q.text!r} has a unit, but what it is compared with has none; write a bare number "
                             "or declare the unit (an alias `unit:` or as_unit)")
        goal = self.lookup(target)
        if goal is None:
            raise UnitsError(f"{q.text!r} cannot be compared with values in {target!r}, which is not a known unit; "
                             f"write a bare number or use exactly {target!r}")
        if source.dimension != goal.dimension:
            raise UnitsError(f"{q.text!r} is {self.phrase(source.dimension)} but the values are in {target} "
                             f"({self.phrase(goal.dimension)})")
        if delta:
            return q.value * source.factor / goal.factor
        return (q.value * source.factor + source.offset - goal.offset) / goal.factor

    def factor(self, source: str, target: str) -> float:
        """Multiplier from `source` to `target` for differences (both known, same dimension)."""
        a, b = self.lookup(source), self.lookup(target)
        if a is None or b is None or a.dimension != b.dimension:
            raise UnitsError(f"cannot convert {source!r} to {target!r}")
        return a.factor / b.factor

    def is_affine(self, symbol: str | None) -> bool:
        unit = self.lookup(symbol)
        return unit is not None and unit.offset != 0.0

    def derivative_unit(self, symbol: str | None) -> str | None:
        return self._related(symbol, DERIVATIVE_UNITS)

    def integral_unit(self, symbol: str | None) -> str | None:
        return self._related(symbol, INTEGRAL_UNITS)

    def _related(self, symbol: str | None, table: Mapping[str, str]) -> str | None:
        if symbol is None or symbol not in table:
            return None
        builtin = EXTRA_UNITS.get(symbol) or UNITS.get(symbol)
        if builtin is not None and symbol in self.user and self.user[symbol].dimension != builtin.dimension:
            return None  # the mapping gave the symbol another meaning
        return table[symbol]


def normalize_number_text(text: str, *, decimal_comma: bool = False) -> str:
    """Typographic minus signs and spaces to ASCII; a decimal comma to a point when asked."""
    cleaned = (str(text).replace("−", "-").replace("–", "-").replace(" ", " ")
               .replace(" ", " ").replace(" ", " ").strip())
    if decimal_comma:
        cleaned = re.sub(r"(?<=\d),(?=\d)", ".", cleaned)
    return cleaned
