"""Reading .xlsx sheets written part by part: strings, numbers, formats, formulas, layout and hostile files."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from baslt.errors import TableError
from baslt.tabular import Table, list_sheets, read_table
from reference import xlsx_fixtures as fx

pytestmark = pytest.mark.minimal


def texts(table):
    return [[None if cell is None else cell.text for cell in row] for row in table.rows]


def values(table):
    return [[None if cell is None else cell.value for cell in row] for row in table.rows]


def test_shared_strings_rich_text_phonetics_and_escapes(tmp_path):
    strings = [
        "<si><t>ID</t></si>",
        '<si><r><rPr><b/></rPr><t>Max </t></r><r><t xml:space="preserve">q </t></r><rPh sb="0" eb="1"><t>ignored</t></rPh></si>',
        "<si><t>line_x000D_break_x005F_x000D_kept</t></si>",
        "<si><t/></si>",
    ]
    rows = ('<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c><c r="C2" t="s"><v>99</v></c></row>')
    table = read_table(fx.simple(tmp_path / "s.xlsx", rows, strings=strings))
    assert texts(table) == [["ID", "Max q "], ["line\rbreak_x000D_kept", "", ""]]
    assert table.rows[1][1].value is None and table.rows[1][1].is_empty


def test_cell_types(tmp_path):
    rows = (
        '<row r="1">'
        '<c r="A1" t="inlineStr"><is><r><t>in</t></r><r><t>line</t></r></is></c>'
        '<c r="B1" t="str"><f>A1&amp;"x"</f><v>inlinex</v></c>'
        '<c r="C1" t="b"><v>1</v></c>'
        '<c r="D1" t="b"><v>0</v></c>'
        '<c r="E1" t="e"><v>#N/A</v></c>'
        '<c r="F1" t="d"><v>2024-05-06T07:08:09</v></c>'
        '<c r="G1"><v>70</v></c>'
        '<c r="H1" t="n"><v>-1.25E-3</v></c>'
        '<c r="I1"><v>12345678901234567890</v></c>'
        "</row>"
    )
    table = read_table(fx.simple(tmp_path / "t.xlsx", rows))
    assert texts(table) == [["inline", "inlinex", "TRUE", "FALSE", "#N/A", "2024-05-06T07:08:09", "70", "-0.00125",
                             "1.2345678901234567e+19"]]
    assert values(table)[0][:4] == ["inline", "inlinex", True, False]
    assert values(table)[0][4] is None
    assert values(table)[0][6] == 70.0 and isinstance(values(table)[0][6], float)


def test_formulas_without_a_saved_value_are_flagged(tmp_path):
    rows = '<row r="1"><c r="A1"><f>1+1</f></c><c r="B1"><f>2+2</f><v>4</v></c></row>'
    table = read_table(fx.simple(tmp_path / "f.xlsx", rows))
    assert table.rows[0][0].formula_uncached and table.rows[0][0].text == ""
    assert not table.rows[0][1].formula_uncached and table.rows[0][1].value == 4.0
    assert len(table.issues) == 1 and "A1" in table.issues[0] and "save it once" in table.issues[0]


def test_percent_and_date_formats(tmp_path):
    rows = (
        '<row r="1">'
        '<c r="A1" s="1"><v>0.05</v></c>'
        '<c r="B1" s="2"><v>0.125</v></c>'
        '<c r="C1" s="3"><v>0.5</v></c>'
        '<c r="D1" s="4"><v>45000</v></c>'
        '<c r="E1" s="5"><v>45000.5</v></c>'
        '<c r="F1" s="6"><v>0.5</v></c>'
        '<c r="G1" s="7"><v>3</v></c>'
        "</row>"
    )
    xfs = [0, 9, 10, 165, 14, 166, 167, 168]
    custom = {165: "0.0%", 166: "yyyy-mm-dd hh:mm", 167: '"%"0.00', 168: "0.00"}
    table = read_table(fx.simple(tmp_path / "p.xlsx", rows, xfs=xfs, custom=custom))
    assert texts(table) == [["5%", "12.5%", "50%", "2023-03-15", "2023-03-15T12:00:00", "0.5", "3"]]
    assert values(table)[0][0] == 0.05


def test_the_1904_date_system(tmp_path):
    rows = '<row r="1"><c r="A1" s="1"><v>0</v></c></row>'
    table = read_table(fx.simple(tmp_path / "d.xlsx", rows, xfs=[0, 14], date1904=True))
    assert texts(table) == [["1904-01-01"]]


def test_merged_cells_and_sparse_rows(tmp_path):
    rows = '<row r="2"><c r="C2" t="inlineStr"><is><t>x</t></is></c></row><row r="4"/>'
    extra = '<mergeCells count="1"><mergeCell ref="A1:C1"/></mergeCells>'
    table = read_table(fx.simple(tmp_path / "m.xlsx", rows, sheet_extra=extra))
    assert texts(table) == [[], [None, None, "x"]]
    assert table.merged == [(1, 1, 1, 3)]
    assert table.cell(2, 3).ref == "C2" and table.cell(9, 9) is None and table.text(1, 1) == ""


def test_rows_and_cells_without_references(tmp_path):
    rows = ('<row><c t="inlineStr"><is><t>a</t></is></c><c t="inlineStr"><is><t>b</t></is></c></row>'
            '<row><c r="C2"><v>1</v></c><c><v>2</v></c></row>')
    table = read_table(fx.simple(tmp_path / "r.xlsx", rows))
    assert texts(table) == [["a", "b"], [None, None, "1", "2"]]
    assert table.rows[1][3].ref == "D2"


def _multi(tmp_path, *, main=fx.MAIN, rel=fx.REL, prefix="", absolute=False):
    targets = [("worksheet", "/xl/worksheets/sheet1.xml" if absolute else "worksheets/sheet1.xml"),
               ("worksheet", "worksheets/sheet2.xml"),
               ("chartsheet", "chartsheets/sheet1.xml"),
               ("worksheet", "worksheets/sheet3.xml")]
    p = f"{prefix}:" if prefix else ""
    parts = {
        "[Content_Types].xml": fx.CONTENT_TYPES,
        "_rels/.rels": fx.package_rels(rel=rel),
        "xl/workbook.xml": fx.workbook([("Meta", "hidden"), ("Requirements", "visible"), ("Chart", "visible"),
                                        ("Secret", "veryHidden")], main=main, rel=rel, prefix=prefix),
        "xl/_rels/workbook.xml.rels": fx.workbook_rels(targets, rel=rel),
        "xl/worksheets/sheet1.xml": fx.sheet(f'<{p}row r="1"><{p}c r="A1"><{p}v>1</{p}v></{p}c></{p}row>',
                                             main=main, prefix=prefix),
        "xl/worksheets/sheet2.xml": fx.sheet(f'<{p}row r="1"><{p}c r="A1"><{p}v>2</{p}v></{p}c></{p}row>',
                                             main=main, prefix=prefix),
        "xl/worksheets/sheet3.xml": fx.sheet("", main=main, prefix=prefix),
    }
    return fx.build(tmp_path / "multi.xlsx", parts)


@pytest.mark.parametrize("variant", [
    {}, {"main": fx.STRICT_MAIN, "rel": fx.STRICT_REL}, {"prefix": "x"}, {"absolute": True},
])
def test_sheet_selection_in_every_package_flavour(tmp_path, variant):
    path = _multi(tmp_path, **variant)
    assert [(s.name, s.hidden) for s in list_sheets(path)] == [("Meta", True), ("Requirements", False),
                                                             ("Secret", True)]
    assert texts(read_table(path)) == [["2"]]  # the first visible sheet
    assert texts(read_table(path, sheet="meta")) == [["1"]]
    assert texts(read_table(path, sheet=1)) == [["1"]]
    assert read_table(path, sheet="Secret").rows == []
    with pytest.raises(TableError, match=r"has no sheet 'Chart'; its sheets are 'Meta', 'Requirements', 'Secret'"):
        read_table(path, sheet="Chart")
    with pytest.raises(TableError, match="there is no sheet 4"):
        read_table(path, sheet=4)


def test_locations():
    xlsx = Table(path=Path("/x/reqs.xlsx"), sheet="Requirements", rows=[], format="xlsx")
    csv = Table(path=Path("/x/reqs.csv"), sheet=None, rows=[], format="csv")
    assert xlsx.location(12, 6) == "reqs.xlsx:Requirements!F12"
    assert xlsx.location(3) == "reqs.xlsx:Requirements!row 3"
    assert csv.location(12, 6) == "reqs.csv:12:6" and csv.location(2) == "reqs.csv:2"


def test_hostile_and_broken_files(tmp_path):
    ole = tmp_path / "old.xls"
    ole.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100)
    with pytest.raises(TableError, match="save it as .xlsx without a password"):
        read_table(ole)
    fake = tmp_path / "fake.xlsx"
    fake.write_text("ID,Title\n")
    with pytest.raises(TableError, match="not a valid .xlsx workbook"):
        read_table(fake)
    with pytest.raises(TableError, match="no such file"):
        read_table(tmp_path / "missing.xlsx")

    empty = fx.build(tmp_path / "empty.xlsx", {"[Content_Types].xml": fx.CONTENT_TYPES})
    with pytest.raises(TableError, match="has no workbook part"):
        read_table(empty)


def _replace_sheet(path, text):
    with pytest.warns(UserWarning, match="Duplicate name"), zipfile.ZipFile(path, "a") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", text)  # the later entry wins when the file is read
    return path


def test_doctype_and_malformed_sheets(tmp_path):
    bomb = _replace_sheet(fx.simple(tmp_path / "entity.xlsx", ""),
                          '<?xml version="1.0"?><!DOCTYPE lol [<!ENTITY lol "lol">]><worksheet/>')
    with pytest.raises(TableError, match="DOCTYPE"):
        read_table(bomb)
    broken = _replace_sheet(fx.simple(tmp_path / "broken.xlsx", ""), "<worksheet><sheetData>")
    with pytest.raises(TableError, match="not well-formed"):
        read_table(broken)


def test_oversized_parts_are_refused(tmp_path, monkeypatch):
    from baslt.tabular import xlsx_read

    path = fx.simple(tmp_path / "big.xlsx", '<row r="1"><c r="A1"><v>1</v></c></row>' * 50)
    monkeypatch.setattr(xlsx_read, "MAX_PART_BYTES", 200)
    with pytest.raises(TableError, match="too large to read safely"):
        read_table(path)
