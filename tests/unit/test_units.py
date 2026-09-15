"""Unit table, quantity parsing and conversion."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from baslt.errors import PolicyError
from baslt.units import (
    DIMENSIONS,
    UNITS,
    Quantity,
    UnitDef,
    convert,
    describe_dimension,
    lookup_unit,
    parse_bytes,
    parse_quantity,
    suggest_unit,
    to_seconds,
)

PI_180 = math.pi / 180

# Written from the unit table in docs/policy.md: symbol -> (dimension, factor, offset).
DOC_UNITS = {
    "s": ("time", 1.0, 0.0),
    "ms": ("time", 1e-3, 0.0),
    "us": ("time", 1e-6, 0.0),
    "µs": ("time", 1e-6, 0.0),
    "ns": ("time", 1e-9, 0.0),
    "min": ("time", 60.0, 0.0),
    "h": ("time", 3600.0, 0.0),
    "rad": ("angle", 1.0, 0.0),
    "mrad": ("angle", 1e-3, 0.0),
    "deg": ("angle", PI_180, 0.0),
    "°": ("angle", PI_180, 0.0),
    "degree": ("angle", PI_180, 0.0),
    "degrees": ("angle", PI_180, 0.0),
    "rad/s": ("angular_rate", 1.0, 0.0),
    "deg/s": ("angular_rate", PI_180, 0.0),
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

_DOC_DIMENSION = {
    "time (s)": "time",
    "angle (rad)": "angle",
    "angular rate (rad/s)": "angular_rate",
    "pressure (Pa)": "pressure",
    "length (m)": "length",
    "velocity (m/s)": "velocity",
    "acceleration (m/s^2)": "acceleration",
    "force (N)": "force",
    "mass (kg)": "mass",
    "temperature (K)": "temperature",
    "frequency (Hz)": "frequency",
    "dimensionless (1)": "dimensionless",
    "bytes (B)": "bytes",
}


def _eval_factor(expr: str) -> float:
    expr = expr.strip().replace("π", "pi")
    assert re.fullmatch(r"[0-9.e+\-/pi ]+", expr), expr
    parts = expr.split("/")
    value = math.pi if parts[0] == "pi" else float(parts[0])
    for part in parts[1:]:
        value /= float(part)
    return value


def _doc_table() -> dict[str, tuple[str, float, float]]:
    doc = Path(__file__).resolve().parents[2] / "docs" / "policy.md"
    text = doc.read_text(encoding="utf-8")
    section = text.split("## Units", 1)[1].split("## Errors", 1)[0]
    table: dict[str, tuple[str, float, float]] = {}
    for line in section.splitlines():
        if not line.startswith("| ") or line.startswith("| Dimension"):
            continue
        dim_text, units_text = (cell.strip() for cell in line.strip("|").split("|"))
        dimension = _DOC_DIMENSION[dim_text]
        for entry in units_text.split(" · "):
            match = re.fullmatch(r"`([^`]+)` ([^,]+)(?:, offset (.+))?", entry.strip())
            assert match, entry
            offset = float(match.group(3)) if match.group(3) else 0.0
            table[match.group(1)] = (dimension, _eval_factor(match.group(2)), offset)
    return table


def test_table_matches_literal_doc_values():
    assert set(UNITS) == set(DOC_UNITS)
    for symbol, (dimension, factor, offset) in DOC_UNITS.items():
        unit = UNITS[symbol]
        assert unit.symbol == symbol
        assert unit.dimension == dimension
        assert unit.factor == pytest.approx(factor, rel=1e-15)
        assert unit.offset == offset


def test_table_matches_docs_markdown():
    table = _doc_table()
    assert set(table) == set(UNITS)
    for symbol, (dimension, factor, offset) in table.items():
        assert UNITS[symbol].dimension == dimension, symbol
        assert UNITS[symbol].factor == pytest.approx(factor, rel=1e-15), symbol
        assert UNITS[symbol].offset == offset, symbol


def test_micro_and_degree_code_points():
    assert "µs" in UNITS and UNITS["µs"].factor == 1e-6
    assert "μs" not in UNITS  # Greek mu is a different symbol
    assert "°" in UNITS and UNITS["°"].factor == PI_180


def test_exact_factors_for_common_units():
    assert UNITS["deg"].factor == math.pi / 180
    assert UNITS["kPa"].factor == 1000.0
    assert UNITS["MiB"].factor == 1048576.0


def test_dimensions_and_phrases():
    assert set(DIMENSIONS) == {u.dimension for u in UNITS.values()}
    assert describe_dimension("angle") == "an angle"
    assert describe_dimension("pressure") == "a pressure"
    assert describe_dimension("dimensionless") == "dimensionless"
    assert describe_dimension("unknown_dim") == "unknown_dim"


def test_dataclasses_are_frozen():
    unit = UNITS["s"]
    with pytest.raises(AttributeError):
        unit.factor = 2.0  # type: ignore[misc]
    q = Quantity(1.0, "s", "1 s")
    with pytest.raises(AttributeError):
        q.value = 2.0  # type: ignore[misc]
    assert UnitDef("x", "time", 1.0).offset == 0.0


def test_lookup_unit():
    assert lookup_unit(None) is None
    assert lookup_unit("kPa") is UNITS["kPa"]
    assert lookup_unit("kpa") is None
    assert lookup_unit("counts") is None


@pytest.mark.parametrize("symbol", sorted(DOC_UNITS))
def test_every_symbol_parses_with_and_without_space(symbol):
    forms = [f"2 {symbol}", f"  2   {symbol} "]
    if not symbol[0].isdigit():  # "21" is the bare number 21, not 2 of unit "1"
        forms.append(f"2{symbol}")
    for text in forms:
        q = parse_quantity(text)
        assert q.value == 2.0
        assert q.unit == symbol
        assert q.text == text.strip()


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("65 kPa", 65.0, "kPa"),
        ("100ms", 100.0, "ms"),
        ("-7 deg", -7.0, "deg"),
        ("+7deg", 7.0, "deg"),
        ("1.5e3 Pa", 1500.0, "Pa"),
        ("1E-3s", 1e-3, "s"),
        ("2.5e+2 m", 250.0, "m"),
        (".5 s", 0.5, "s"),
        ("5. s", 5.0, "s"),
        ("42", 42.0, None),
        ("-0.25", -0.25, None),
        ("1e3", 1000.0, None),
        ("10 m/s^2", 10.0, "m/s^2"),
        ("5%", 5.0, "%"),
        ("3 1", 3.0, "1"),
    ],
)
def test_parse_quantity_syntax(text, value, unit):
    q = parse_quantity(text)
    assert q.value == pytest.approx(value)
    assert q.unit == unit
    assert q.text == text


def test_parse_numbers():
    assert parse_quantity(5) == Quantity(5.0, None, "5")
    assert parse_quantity(-2.5) == Quantity(-2.5, None, "-2.5")
    assert parse_quantity(0) == Quantity(0.0, None, "0")
    assert parse_quantity(1e-3).text == "0.001"


@pytest.mark.parametrize("value", [True, False])
def test_parse_rejects_bool(value):
    with pytest.raises(PolicyError, match="got a boolean") as info:
        parse_quantity(value, path="a.b")
    assert info.value.issues[0].path == "a.b"


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        (None, "got null"),
        ([1], "got list"),
        ({"a": 1}, "got dict"),
        ("kPa", "expected a number with an optional unit"),
        ("", "expected a number with an optional unit"),
        ("65 kPa extra", "expected a number with an optional unit"),
        ("1,5 s", "expected a number with an optional unit"),
        ("nan", "expected a number with an optional unit"),
        ("inf s", "expected a number with an optional unit"),
        ("1e999 s", "finite"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (10**400, "too large"),
    ],
)
def test_parse_errors(value, fragment):
    with pytest.raises(PolicyError, match=fragment) as info:
        parse_quantity(value, path="x.y")
    assert info.value.issues[0].path == "x.y"


def test_units_are_case_sensitive_with_suggestions():
    with pytest.raises(PolicyError) as info:
        parse_quantity("65 kpa", path="p")
    assert str(info.value) == "p: unknown unit 'kpa' in '65 kpa'; did you mean 'kPa'?"
    for text in ("1 S", "1 pa", "1 mPa", "1 DEG", "1 hz"):
        with pytest.raises(PolicyError, match="unknown unit"):
            parse_quantity(text)


def test_unknown_unit_without_suggestion():
    with pytest.raises(PolicyError) as info:
        parse_quantity("3 zzzzzz")
    assert str(info.value) == "unknown unit 'zzzzzz' in '3 zzzzzz'"


def test_allow_unknown_keeps_the_unit():
    q = parse_quantity("12 counts", allow_unknown=True)
    assert q == Quantity(12.0, "counts", "12 counts")


def test_suggest_unit():
    assert suggest_unit("kpa") == "kPa"
    assert suggest_unit("mib") == "MiB"
    assert suggest_unit("degc") == "degC"
    assert suggest_unit("qqqqqq") is None


class TestConvert:
    def test_bare_number_unchanged(self):
        q = parse_quantity(12.5)
        assert convert(q, "Pa", delta=False) == 12.5
        assert convert(q, None, delta=True) == 12.5
        assert convert(q, "counts", delta=False) == 12.5

    def test_same_unit_is_exact(self):
        q = parse_quantity("0.1 deg")
        assert convert(q, "deg", delta=False) == 0.1
        assert convert(parse_quantity("5 counts", allow_unknown=True), "counts", delta=False) == 5.0

    def test_scaled(self):
        assert convert(parse_quantity("65 kPa"), "Pa", delta=False) == 65000.0
        assert convert(parse_quantity("1 bar"), "kPa", delta=False) == pytest.approx(100.0)
        assert convert(parse_quantity("7 deg"), "rad", delta=False) == pytest.approx(7 * PI_180)
        assert convert(parse_quantity("1 rad"), "°", delta=False) == pytest.approx(180 / math.pi)
        assert convert(parse_quantity("1 nmi"), "km", delta=True) == pytest.approx(1.852)
        assert convert(parse_quantity("36 km/h"), "m/s", delta=False) == pytest.approx(10.0)
        assert convert(parse_quantity("1 psi"), "Pa", delta=False) == pytest.approx(6894.757293168361)
        assert convert(parse_quantity("50 %"), "1", delta=False) == pytest.approx(0.5)

    def test_absolute_temperature(self):
        assert convert(parse_quantity("20 degC"), "K", delta=False) == pytest.approx(293.15)
        assert convert(parse_quantity("300 K"), "degC", delta=False) == pytest.approx(26.85)
        assert convert(parse_quantity("32 degF"), "degC", delta=False) == pytest.approx(0.0, abs=1e-12)
        assert convert(parse_quantity("212 degF"), "K", delta=False) == pytest.approx(373.15)
        assert convert(parse_quantity("100 degC"), "degF", delta=False) == pytest.approx(212.0)

    def test_delta_temperature_ignores_offsets(self):
        assert convert(parse_quantity("2 degC"), "K", delta=True) == pytest.approx(2.0)
        assert convert(parse_quantity("9 degF"), "degC", delta=True) == pytest.approx(5.0)
        assert convert(parse_quantity("5 K"), "degF", delta=True) == pytest.approx(9.0)

    def test_dimension_mismatch(self):
        with pytest.raises(PolicyError) as info:
            convert(parse_quantity("7 deg"), "Pa", delta=False, path="hard.q.v")
        issue = info.value.issues[0]
        assert issue.path == "hard.q.v"
        assert issue.message == "'7 deg' is an angle but the signal unit Pa is a pressure"

    def test_unit_on_unitless_target(self):
        with pytest.raises(PolicyError, match="has a unit but the signal has no unit"):
            convert(parse_quantity("7 deg"), None, delta=False)

    def test_opaque_target_with_other_unit(self):
        with pytest.raises(PolicyError, match="not a known unit") as info:
            convert(parse_quantity("7 deg"), "counts", delta=False, path="p")
        assert info.value.issues[0].path == "p"

    def test_unknown_source_unit(self):
        q = parse_quantity("7 kpa", allow_unknown=True)
        with pytest.raises(PolicyError, match="unknown unit 'kpa'.*did you mean 'kPa'"):
            convert(q, "Pa", delta=False)

    def test_expect_dimension(self):
        assert convert(parse_quantity("1 km"), "m", delta=True, expect_dimension="length") == 1000.0
        assert convert(parse_quantity(3), "m", delta=True, expect_dimension="length") == 3.0
        with pytest.raises(PolicyError) as info:
            convert(parse_quantity("1 s"), "m", delta=True, expect_dimension="length", path="traj")
        assert info.value.issues[0].message == "'1 s' is a time but a length is required"


class TestToSeconds:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("100 ms", 0.1),
            ("2 s", 2.0),
            ("3 min", 180.0),
            ("1.5 h", 5400.0),
            ("250 us", 250e-6),
            ("250 µs", 250e-6),
            ("7 ns", 7e-9),
            ("0.25", 0.25),
        ],
    )
    def test_values(self, text, seconds):
        assert to_seconds(parse_quantity(text)) == pytest.approx(seconds)

    def test_bare_number(self):
        assert to_seconds(parse_quantity(4)) == 4.0

    def test_non_time(self):
        with pytest.raises(PolicyError) as info:
            to_seconds(parse_quantity("5 m"), path="d")
        assert str(info.value) == "d: '5 m' is a length but a duration (time) is required"

    def test_unknown_unit(self):
        with pytest.raises(PolicyError, match="unknown unit 'sec'"):
            to_seconds(parse_quantity("5 sec", allow_unknown=True))


class TestParseBytes:
    @pytest.mark.parametrize(
        ("value", "count"),
        [
            (2048, 2048),
            (0, 0),
            ("2048", 2048),
            ("2 MiB", 2097152),
            ("2MiB", 2097152),
            ("1.5 KiB", 1536),
            ("2.2 MB", 2200000),
            ("3 GiB", 3221225472),
            ("1 GB", 1000000000),
            ("7 KB", 7000),
            ("5 B", 5),
            ("1e3 B", 1000),
            ("1234.5678 GB", 1234567800000),
        ],
    )
    def test_values(self, value, count):
        result = parse_bytes(value)
        assert result == count
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        ("value", "fragment"),
        [
            (True, "got a boolean"),
            (1.5, "got float"),
            (None, "got null"),
            (-1, "must not be negative"),
            ("-2 KiB", "must not be negative"),
            ("1.3 B", "not a whole number of bytes"),
            ("10.5", "not a whole number of bytes"),
            # A fractional byte count is rejected however large it is, not silently rounded.
            ("1000000000.4 B", "not a whole number of bytes"),
            ("1073741824.5", "not a whole number of bytes"),
            ("2.0000000001 GiB", "not a whole number of bytes"),
            ("2 s", "is a time but a byte size is required"),
            ("2 mib", "unknown unit 'mib'"),
            ("big", "expected a number"),
        ],
    )
    def test_errors(self, value, fragment):
        with pytest.raises(PolicyError, match=fragment) as info:
            parse_bytes(value, path="artifact.max_size")
        assert info.value.issues[0].path == "artifact.max_size"


def test_units_module_does_not_import_numpy():
    import subprocess
    import sys

    code = "import sys, baslt.units, baslt.policy; print('numpy' in sys.modules, 'yaml' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"
