"""The numpy-free CSV text helpers shared by the signal reader and the table reader."""

from __future__ import annotations

import subprocess
import sys

import pytest

from baslt.sources.csv_text import WHITESPACE, non_blank, parse_header_field, sniff_delimiter, split_fields

pytestmark = pytest.mark.minimal


@pytest.mark.parametrize(("lines", "expected"), [
    (["a,b,c", "1,2,3"], ","),
    (["a;b", "1,5;2,5"], ";"),
    (["a\tb\tc", "1\t2\t3"], "\t"),
    (["time  q [Pa]", "0.0  1.0"], WHITESPACE),
    (["only", "1"], None),
    (['"x, y",z', '"1,2",3'], ","),
])
def test_sniff_delimiter(lines, expected):
    assert sniff_delimiter(lines) == expected


def test_split_fields_keeps_quoted_commas_and_unit_tokens():
    assert split_fields('ID,"Title, long", Limit', ",") == ["ID", "Title, long", "Limit"]
    assert split_fields("t   q [kPa]  alpha (deg)", WHITESPACE) == ["t", "q [kPa]", "alpha (deg)"]


@pytest.mark.parametrize(("raw", "expected"), [
    ("q [kPa]", ("q", "kPa")),
    ("alpha (deg)", ("alpha", "deg")),
    ('"thrust [N]"', ("thrust", "N")),
    ("mode", ("mode", None)),
    ("x []", ("x", None)),
])
def test_parse_header_field(raw, expected):
    assert parse_header_field(raw) == expected


def test_non_blank():
    assert non_blank(["a"]) and not non_blank([]) and not non_blank(["  "])


def test_importing_the_helpers_does_not_load_numpy():
    code = "import sys, baslt.sources.csv_text; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


@pytest.mark.parametrize(("header", "expected"), [
    ("q [Pa]", ("q", "Pa")),
    ("q (Pa)", ("q", "Pa")),
    ("q [1]", ("q", "1")),                      # a dimensionless unit, written with a space
    ("gyro_rad[0]", ("gyro_rad[0]", None)),     # PX4 array columns: the index belongs to the name
    ("delta_xy[1]", ("delta_xy[1]", None)),
    ("x(2)", ("x(2)", None)),
    ("accel[0] [m/s^2]", ("accel[0]", "m/s^2")),
    ("value[]", ("value", None)),
])
def test_array_indexes_are_part_of_the_name(header, expected):
    assert parse_header_field(header) == expected
