"""The requirement unit table: policy units, extra engineering units and user units."""

from __future__ import annotations

import pytest

from baslt.reqs.units_ext import EXTRA_UNITS, UnitsError, UnitTable
from baslt.units import UNITS, UnitDef

pytestmark = pytest.mark.minimal


def test_lookup_order_and_conversion():
    table = UnitTable({"g": UnitDef("g", "acceleration", 9.80665)})
    assert table.lookup("kPa") is UNITS["kPa"]
    assert table.lookup("g0") is EXTRA_UNITS["g0"]
    assert table.dimension("g") == "acceleration"  # the mapping's meaning wins over grams
    assert UnitTable().dimension("g") == "mass"
    q = table.parse("5 g")
    assert table.convert(q, "m/s^2", delta=False) == pytest.approx(49.03325)
    assert table.convert(table.parse("2 g0"), "g", delta=True) == pytest.approx(2.0)
    assert table.convert(table.parse("0 degC"), "K", delta=False) == pytest.approx(273.15)
    assert table.convert(table.parse("10 degC"), "K", delta=True) == pytest.approx(10.0)
    assert table.convert(table.parse("3"), "kPa", delta=False) == 3.0
    assert table.convert(table.parse("300 rpm"), "rad/s", delta=False) == pytest.approx(31.4159265)
    assert table.factor("kN*m", "N*m") == 1e3
    assert table.is_affine("degF") and not table.is_affine("K")


def test_errors():
    table = UnitTable()
    with pytest.raises(UnitsError, match="did you mean 'kPa'"):
        table.parse("5 kpa")
    with pytest.raises(UnitsError, match="is a pressure but the values are in s"):
        table.convert(table.parse("5 kPa"), "s", delta=False)
    with pytest.raises(UnitsError, match="has none"):
        table.convert(table.parse("5 kPa"), None, delta=False)
    with pytest.raises(UnitsError, match="not a known unit"):
        table.convert(table.parse("5 kPa"), "furlong", delta=False)
    with pytest.raises(UnitsError, match="expected a number"):
        table.parse("five")
    with pytest.raises(UnitsError, match="boolean"):
        table.parse(True)


def test_derivative_and_integral_units():
    table = UnitTable()
    assert table.derivative_unit("m") == "m/s" and table.derivative_unit("m/s") == "m/s^2"
    assert table.derivative_unit("deg") == "deg/s" and table.integral_unit("kg/s") == "kg"
    assert table.integral_unit("m/s^2") == "m/s" and table.integral_unit("W") == "J"
    assert table.derivative_unit("K") is None and table.derivative_unit(None) is None
    redefined = UnitTable({"g": UnitDef("g", "acceleration", 9.80665)})
    assert redefined.derivative_unit("g") is None


def test_parse_keeps_typographic_forms():
    table = UnitTable()
    assert table.parse("−3 kPa").value == -3.0
    assert table.parse(2).text == "2" and table.parse(2.5).text == "2.5"
    assert table.parse("9.81 m/s²").unit == "m/s²"
