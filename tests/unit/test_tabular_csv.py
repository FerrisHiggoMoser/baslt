"""CSV and TSV tables: delimiters, quoting, encodings and locations."""

from __future__ import annotations

import subprocess
import sys

import pytest

from baslt.errors import TableError
from baslt.tabular import list_sheets, read_table
from baslt.tabular.table import sniff_table_delimiter

pytestmark = pytest.mark.minimal


def texts(table):
    return [[cell.text for cell in row] for row in table.rows]


def test_quoted_cells_with_delimiters_and_newlines(tmp_path):
    path = tmp_path / "reqs.csv"
    path.write_text('ID,Title,Limit\nLV-001,"Max q, ascent","<= 70 kPa"\nLV-002,"two\nlines", 5 \n\n\n',
                    encoding="utf-8")
    table = read_table(path)
    assert table.delimiter == "," and table.encoding == "utf-8" and table.format == "csv"
    assert texts(table) == [["ID", "Title", "Limit"], ["LV-001", "Max q, ascent", "<= 70 kPa"],
                            ["LV-002", "two\nlines", "5"]]
    assert table.rows[2][2].value == "5" and table.rows[2][2].ref == "C3"
    assert table.location(3, 2) == "reqs.csv:3:2"


@pytest.mark.parametrize(("text", "delimiter"), [
    ("a;b;c\n1,5;2;3\n", ";"),
    ("a\tb\n1\t2\n", "\t"),
    ("a,b\n1,2\n", ","),
    ("only\none\n", ","),
    ('x;"y;z"\n1;2\n', ";"),
])
def test_delimiter_sniffing(text, delimiter):
    assert sniff_table_delimiter(text) == delimiter


def test_explicit_delimiter_and_encoding(tmp_path):
    path = tmp_path / "params.tsv"
    path.write_bytes("run\tpayload [kg]\nrün_1\t1200\n".encode("latin-1"))
    table = read_table(path)
    assert table.encoding == "cp1252" and "Windows-1252" in table.issues[0]
    assert texts(table)[1] == ["rün_1", "1200"]
    assert texts(read_table(path, delimiter="tab", encoding="latin-1"))[0] == ["run", "payload [kg]"]
    with pytest.raises(TableError, match="one character"):
        read_table(path, delimiter=";;")
    with pytest.raises(TableError, match="cannot decode"):
        read_table(path, encoding="utf-8")


def test_bom_and_blank_cells(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("﻿ID,Unit\nA,\n".encode("utf-8"))
    table = read_table(path)
    assert texts(table) == [["ID", "Unit"], ["A", ""]]
    assert table.rows[1][1].value is None and table.rows[1][1].is_empty


def test_missing_file_and_single_sheet(tmp_path):
    with pytest.raises(TableError, match="no such file"):
        read_table(tmp_path / "nope.csv")
    path = tmp_path / "reqs.txt"
    path.write_text("a,b\n")
    assert [s.name for s in list_sheets(path)] == ["reqs"]
    assert texts(read_table(path)) == [["a", "b"]]


def test_the_table_package_does_not_load_numpy():
    code = ("import sys, baslt.tabular, baslt.tabular.xlsx_read, baslt.tabular.xlsx_write, "
            "baslt.tabular.xlsx_patch; print('numpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
