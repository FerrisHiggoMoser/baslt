"""The verifier's SI table matches docs/policy.md bit for bit and agrees exactly with the compiler's unit table.

structure.bind re-binds the policy with the verifier's table and compares bound parameters bitwise, so every
factor, offset and converted value here is compared by its IEEE-754 bits, never with a tolerance.
"""

from __future__ import annotations

import importlib
import math
import struct
import subprocess
import sys
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from baslt.verify.si_table import SI_UNITS, dimension, to_si

pytestmark = pytest.mark.minimal

# Transcribed by hand from the Units table in docs/policy.md: symbol -> (dimension, factor, offset).
# Factors are written as the docs write them (1e-3, π/180, 1/3.6, 1852/3600, 5/9) so they evaluate to the same float64.
DOC_TABLE: dict[str, tuple[str, float, float]] = {
    "s": ("time", 1.0, 0.0),
    "ms": ("time", 1e-3, 0.0),
    "us": ("time", 1e-6, 0.0),
    "µs": ("time", 1e-6, 0.0),
    "ns": ("time", 1e-9, 0.0),
    "min": ("time", 60.0, 0.0),
    "h": ("time", 3600.0, 0.0),
    "rad": ("angle", 1.0, 0.0),
    "mrad": ("angle", 1e-3, 0.0),
    "deg": ("angle", math.pi / 180, 0.0),
    "°": ("angle", math.pi / 180, 0.0),
    "degree": ("angle", math.pi / 180, 0.0),
    "degrees": ("angle", math.pi / 180, 0.0),
    "rad/s": ("angular_rate", 1.0, 0.0),
    "deg/s": ("angular_rate", math.pi / 180, 0.0),
    "Pa": ("pressure", 1.0, 0.0),
    "hPa": ("pressure", 100.0, 0.0),
    "kPa": ("pressure", 1e3, 0.0),
    "MPa": ("pressure", 1e6, 0.0),
    "bar": ("pressure", 1e5, 0.0),
    "mbar": ("pressure", 100.0, 0.0),
    "psi": ("pressure", 6894.757293168361, 0.0),
    "m": ("length", 1.0, 0.0),
    "mm": ("length", 1e-3, 0.0),
    "cm": ("length", 1e-2, 0.0),
    "km": ("length", 1e3, 0.0),
    "ft": ("length", 0.3048, 0.0),
    "in": ("length", 0.0254, 0.0),
    "nmi": ("length", 1852.0, 0.0),
    "m/s": ("velocity", 1.0, 0.0),
    "km/s": ("velocity", 1e3, 0.0),
    "km/h": ("velocity", 1 / 3.6, 0.0),
    "ft/s": ("velocity", 0.3048, 0.0),
    "kn": ("velocity", 1852 / 3600, 0.0),
    "m/s^2": ("acceleration", 1.0, 0.0),
    "m/s2": ("acceleration", 1.0, 0.0),
    "ft/s^2": ("acceleration", 0.3048, 0.0),
    "N": ("force", 1.0, 0.0),
    "kN": ("force", 1e3, 0.0),
    "MN": ("force", 1e6, 0.0),
    "lbf": ("force", 4.4482216152605, 0.0),
    "g": ("mass", 1e-3, 0.0),
    "kg": ("mass", 1.0, 0.0),
    "t": ("mass", 1e3, 0.0),
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, 273.15),
    "degF": ("temperature", 5 / 9, 255.37222222222223),
    "Hz": ("frequency", 1.0, 0.0),
    "kHz": ("frequency", 1e3, 0.0),
    "1": ("dimensionless", 1.0, 0.0),
    "%": ("dimensionless", 0.01, 0.0),
    "B": ("bytes", 1.0, 0.0),
    "KB": ("bytes", 1e3, 0.0),
    "MB": ("bytes", 1e6, 0.0),
    "GB": ("bytes", 1e9, 0.0),
    "KiB": ("bytes", 1024.0, 0.0),
    "MiB": ("bytes", 1048576.0, 0.0),
    "GiB": ("bytes", 1073741824.0, 0.0),
}

DIMENSIONS = {
    "time", "angle", "angular_rate", "pressure", "length", "velocity", "acceleration",
    "force", "mass", "temperature", "frequency", "dimensionless", "bytes",
}  # fmt: skip

# Inputs chosen so that reordered formulas give different bits: v / 3.6, v * π / 180, v * 5 / 9 and v * 1852 / 3600
# each differ from v * factor for at least one of these.
PROBE_VALUES = (3.0, 65.0, -7.25, 0.1, 123456.789)


def bits(x: float) -> bytes:
    """IEEE-754 binary64 bits of `x`; unlike ==, distinguishes 0.0 from -0.0."""
    return struct.pack("<d", x)


def table_mismatches(
    actual: Mapping[str, tuple[str, float, float]], expected: Mapping[str, tuple[str, float, float]]
) -> list[str]:
    """Every difference between two unit tables, comparing factors and offsets bitwise."""
    problems = [f"{symbol}: missing" for symbol in sorted(set(expected) - set(actual))]
    problems += [f"{symbol}: not documented" for symbol in sorted(set(actual) - set(expected))]
    for symbol in sorted(set(actual) & set(expected)):
        dim, factor, offset = actual[symbol]
        want_dim, want_factor, want_offset = expected[symbol]
        if dim != want_dim:
            problems.append(f"{symbol}: dimension {dim!r} != {want_dim!r}")
        for label, got, want in (("factor", factor, want_factor), ("offset", offset, want_offset)):
            if type(got) is not float:
                problems.append(f"{symbol}: {label} is {type(got).__name__}, not float")
            elif bits(got) != bits(want):
                problems.append(f"{symbol}: {label} {got!r} ({got.hex()}) != {want!r} ({float(want).hex()})")
    return problems


def compiler_table(units_module) -> dict[str, tuple[str, float, float]]:
    """The compiler's UNITS as symbol -> (dimension, factor, offset), checking each UnitDef's own symbol."""
    table = {}
    for symbol, udef in units_module.UNITS.items():
        assert udef.symbol == symbol, f"UNITS[{symbol!r}].symbol is {udef.symbol!r}"
        table[symbol] = (udef.dimension, udef.factor, udef.offset)
    return table


def test_table_has_exactly_the_documented_symbols():
    assert len(DOC_TABLE) == 58
    assert set(SI_UNITS) == set(DOC_TABLE)


def test_every_documented_dimension_is_used():
    assert {dim for dim, _, _ in SI_UNITS.values()} == DIMENSIONS


def test_table_matches_docs_bitwise():
    assert table_mismatches(SI_UNITS, DOC_TABLE) == []


@pytest.mark.parametrize("symbol", sorted(DOC_TABLE))
def test_entry_matches_docs(symbol):
    dim, factor, offset = SI_UNITS[symbol]
    doc_dim, doc_factor, doc_offset = DOC_TABLE[symbol]
    assert dim == doc_dim
    assert type(factor) is float and type(offset) is float
    assert bits(factor) == bits(doc_factor), (factor.hex(), doc_factor.hex())
    assert bits(offset) == bits(doc_offset), (offset.hex(), doc_offset.hex())


@pytest.mark.parametrize("symbol", sorted(DOC_TABLE))
def test_every_symbol_converts(symbol):
    doc_dim, doc_factor, doc_offset = DOC_TABLE[symbol]
    for value in PROBE_VALUES:
        assert bits(to_si(value, symbol, delta=True)) == bits(value * doc_factor), value
        assert bits(to_si(value, symbol, delta=False)) == bits(value * doc_factor + doc_offset), value
    assert dimension(symbol) == doc_dim


def test_probe_values_separate_formula_orders():
    """A converter that divided or multiplied in another order would give different bits for some probe value."""
    reorderings = {
        "km/h": lambda v: v / 3.6,
        "deg": lambda v: v * math.pi / 180,
        "degF": lambda v: v * 5 / 9,
        "kn": lambda v: v * 1852 / 3600,
    }
    for symbol, other in reorderings.items():
        factor = DOC_TABLE[symbol][1]
        assert any(bits(other(v)) != bits(v * factor) for v in PROBE_VALUES), symbol


# Worked examples, SI values computed by hand from the docs formula
# SI = number * factor + offset (absolute) or number * factor (delta). Each literal is the exact float64 result.
@pytest.mark.parametrize(
    ("value", "unit", "delta", "expected"),
    [
        (65.0, "kPa", False, 65000.0),
        (65.0, "kPa", True, 65000.0),
        (1.0, "kPa", True, 1000.0),
        (100.0, "ms", False, 0.1),
        (1.0, "s", False, 1.0),
        (2.0, "min", False, 120.0),
        (1.5, "h", False, 5400.0),
        (7.0, "deg", False, 0.12217304763960307),
        (-7.0, "deg", False, -0.12217304763960307),
        (7.0, "°", True, 0.12217304763960307),
        (180.0, "deg/s", False, 3.141592653589793),
        (300.0, "K", False, 300.0),
        (25.0, "degC", False, 298.15),
        (0.0, "degC", False, 273.15),
        (2.0, "degC", True, 2.0),
        (100.0, "degF", False, 310.9277777777778),
        (100.0, "degF", True, 55.55555555555556),
        (0.0, "degF", False, 255.37222222222223),
        (2.0, "MiB", False, 2097152.0),
        (3.0, "KiB", False, 3072.0),
        (1.0, "GiB", False, 1073741824.0),
        (2.0, "MB", False, 2000000.0),
        (1.0, "kn", False, 0.5144444444444445),
        (36.0, "km/h", False, 10.0),
        (10.0, "ft", False, 3.048),
        (1.0, "nmi", False, 1852.0),
        (100.0, "N", False, 100.0),
        (1.0, "lbf", False, 4.4482216152605),
        (1.0, "psi", False, 6894.757293168361),
        (65.0, "psi", False, 448159.2240559434),
        (250.0, "g", False, 0.25),
        (50.0, "%", False, 0.5),
        (2.0, "kHz", False, 2000.0),
    ],
)
def test_worked_examples(value, unit, delta, expected):
    _, doc_factor, doc_offset = DOC_TABLE[unit]
    from_docs = value * doc_factor if delta else value * doc_factor + doc_offset
    assert bits(from_docs) == bits(expected), "worked example literal is not the exact docs result"
    got = to_si(value, unit, delta=delta)
    assert bits(got) == bits(expected), (got.hex(), expected.hex())


@pytest.mark.parametrize("direction", [math.inf, -math.inf])
@pytest.mark.parametrize("symbol", sorted(DOC_TABLE))
def test_one_ulp_factor_change_is_detected(symbol, direction):
    dim, factor, offset = DOC_TABLE[symbol]
    shifted = dict(DOC_TABLE)
    shifted[symbol] = (dim, math.nextafter(factor, direction), offset)
    problems = table_mismatches(shifted, DOC_TABLE)
    assert len(problems) == 1 and problems[0].startswith(f"{symbol}: factor"), problems


@pytest.mark.parametrize("symbol", ["degC", "degF"])
def test_one_ulp_offset_change_is_detected(symbol):
    dim, factor, offset = DOC_TABLE[symbol]
    shifted = {**DOC_TABLE, symbol: (dim, factor, math.nextafter(offset, math.inf))}
    problems = table_mismatches(shifted, DOC_TABLE)
    assert len(problems) == 1 and problems[0].startswith(f"{symbol}: offset"), problems


def test_mismatch_report_covers_every_kind_of_difference():
    actual = {**DOC_TABLE, "kPa": ("force", 1e3, 0.0), "furlong": ("length", 201.168, 0.0), "Pa": ("pressure", 1, 0.0)}
    del actual["psi"]
    assert table_mismatches(actual, DOC_TABLE) == [
        "psi: missing",
        "furlong: not documented",
        "Pa: factor is int, not float",
        "kPa: dimension 'force' != 'pressure'",
    ]


def test_exact_examples_are_bit_exact():
    assert to_si(65.0, "kPa", delta=False) == 65000.0
    assert to_si(2.0, "MiB", delta=False) == 2097152.0
    assert to_si(100.0, "degF", delta=True) == 55.55555555555556
    assert to_si(1.0, "kn", delta=False) == 0.5144444444444445
    assert to_si(7.0, "deg", delta=False) == 0.12217304763960307


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("value", [0.0, -0.0, -3.25, 1e300, 65.0])
def test_bare_number_is_unchanged(value, delta):
    assert bits(to_si(value, None, delta=delta)) == bits(value)


def test_bare_number_keeps_non_finite():
    assert math.isnan(to_si(math.nan, None, delta=False))
    assert to_si(math.inf, None, delta=True) == math.inf


@pytest.mark.parametrize("unit", ["furlong", "", "kpa", "KPA", " kPa", "kPa ", "deg C", "mib", "s "])
def test_unknown_unit_raises(unit):
    with pytest.raises(ValueError, match="unknown unit"):
        to_si(1.0, unit, delta=False)
    with pytest.raises(ValueError, match="unknown unit"):
        to_si(1.0, unit, delta=True)


def test_dimension():
    assert dimension(None) is None
    assert dimension("kPa") == "pressure"
    assert dimension("deg/s") == "angular_rate"
    assert dimension("m/s^2") == "acceleration"
    assert dimension("degF") == "temperature"
    assert dimension("1") == "dimensionless"
    assert dimension("MiB") == "bytes"
    assert dimension("furlong") is None  # opaque unit


def test_import_is_light():
    heavy = ("numpy", "h5py", "scipy", "yaml")
    code = (
        "import sys, baslt.verify, baslt.verify.si_table; "
        f"print(','.join(m for m in {heavy!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_matches_compiler_unit_table():
    """Cross-check against baslt.units.UNITS in both directions, bitwise."""
    units = importlib.import_module("baslt.units")
    problems = table_mismatches(compiler_table(units), SI_UNITS)
    assert not problems, "baslt.units differs from verify.si_table:\n" + "\n".join(problems)


def test_compiler_cross_check_detects_one_ulp_difference():
    """A compiler factor one ulp off changes bound parameters bitwise, so the cross-check must fail on it."""
    dim, factor, offset = SI_UNITS["psi"]
    lower = math.nextafter(factor, -math.inf)
    assert bits(65.0 * lower) != bits(65.0 * factor)
    fake_units = SimpleNamespace(
        UNITS={
            symbol: SimpleNamespace(
                symbol=symbol, dimension=d, factor=lower if symbol == "psi" else f, offset=o
            )
            for symbol, (d, f, o) in SI_UNITS.items()
        }
    )
    problems = table_mismatches(compiler_table(fake_units), SI_UNITS)
    assert len(problems) == 1 and problems[0].startswith("psi: factor"), problems
