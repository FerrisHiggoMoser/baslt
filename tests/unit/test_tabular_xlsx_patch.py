"""Patching values into existing workbooks: edits land where they belong and nothing else changes."""

from __future__ import annotations

import zipfile

import pytest

from baslt.tabular import CellEdit, patch_xlsx, read_table
from baslt.tabular.xlsx_patch import patch_sheet_xml
from reference import xlsx_fixtures as fx

pytestmark = pytest.mark.minimal

EXCEL_SHEET = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" mc:Ignorable="x14ac xr xr2 xr3" '
    'xmlns:x14ac="http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac" '
    'xmlns:xr="http://schemas.microsoft.com/office/spreadsheetml/2014/revision" '
    'xmlns:xr2="http://schemas.microsoft.com/office/spreadsheetml/2015/revision2" '
    'xmlns:xr3="http://schemas.microsoft.com/office/spreadsheetml/2016/revision3" '
    'xr:uid="{00000000-0001-0000-0000-000000000000}">'
    '<dimension ref="A1:C4"/>'
    '<sheetViews><sheetView tabSelected="1" workbookViewId="0"/></sheetViews>'
    '<sheetFormatPr defaultRowHeight="15" x14ac:dyDescent="0.25"/>'
    '<sheetData>'
    '<row r="1" spans="1:3" x14ac:dyDescent="0.25"><c r="A1" s="1" t="s"><v>0</v></c>'
    '<c r="B1" s="1" t="s"><v>1</v></c><c r="C1" s="1" t="s"><v>2</v></c></row>'
    '<row r="2" spans="1:3" x14ac:dyDescent="0.25"><c r="A2" t="s"><v>3</v></c><c r="C2"><v>70</v></c></row>'
    '<row r="4" spans="1:3" x14ac:dyDescent="0.25"><c r="A4" t="s"><v>4</v></c>'
    '<c r="C4"><f>C2*2</f><v>140</v></c></row>'
    '<row r="5" x14ac:dyDescent="0.25"/>'
    '</sheetData>'
    '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>'
    '</worksheet>'
)
STRINGS = ["<si><t>ID</t></si>", "<si><t>Title</t></si>", "<si><t>Limit</t></si>", "<si><t>LV-001</t></si>",
           "<si><t>LV-002</t></si>"]


def excel_like(tmp_path):
    path = fx.simple(tmp_path / "polarion.xlsx", "", strings=STRINGS, xfs=[0, 0])
    with zipfile.ZipFile(path) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    members["xl/worksheets/sheet1.xml"] = EXCEL_SHEET.encode()
    members["customXml/item1.xml"] = b"<polarion round-trip='metadata'/>"
    return fx.build(tmp_path / "polarion.xlsx", members)


def grid(path, sheet=None):
    return [[None if c is None else c.text for c in row] for row in read_table(path, sheet=sheet).rows]


def test_untouched_parts_and_bytes_stay_identical(tmp_path):
    src = excel_like(tmp_path)
    dst = tmp_path / "checked.xlsx"
    notes = patch_xlsx(src, dst, sheet="Requirements",
                       edits=[CellEdit(1, 4, "Verification Result"), CellEdit(2, 4, "FAIL"), CellEdit(4, 4, "PASS")])
    assert notes == []
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst) as b:
        assert a.namelist() == b.namelist()
        for info in a.infolist():
            if info.filename != "xl/worksheets/sheet1.xml":
                assert a.read(info) == b.read(info.filename), info.filename
                assert b.getinfo(info.filename).compress_type == info.compress_type
        patched = b.read("xl/worksheets/sheet1.xml").decode()
    head, tail = EXCEL_SHEET.split("<sheetData>")
    assert patched.startswith(head.replace('<dimension ref="A1:C4"/>', '<dimension ref="A1:D4"/>'))
    assert patched.endswith(tail.split("</sheetData>", 1)[1])
    assert 'mc:Ignorable="x14ac xr xr2 xr3"' in patched
    assert '<row r="1" x14ac:dyDescent="0.25">' in patched and "spans" not in patched.split('<row r="5"')[0]
    assert '<row r="5" x14ac:dyDescent="0.25"/>' in patched  # an untouched row keeps its bytes
    assert grid(dst) == [["ID", "Title", "Limit", "Verification Result"], ["LV-001", None, "70", "FAIL"], [],
                         ["LV-002", None, "140", "PASS"]]


def test_insertions_replacements_and_new_rows():
    sheet = fx.sheet('<row r="2"><c r="B2" s="3" t="s"><v>0</v></c><c r="D2"><v>4</v></c></row>'
                     '<row r="5"/>').encode()
    edits = [CellEdit(2, 1, "a"), CellEdit(2, 3, 3), CellEdit(2, 4, 4.5), CellEdit(2, 6, True), CellEdit(2, 2, None),
             CellEdit(1, 2, "new first"), CellEdit(3, 1, "between"), CellEdit(5, 2, "into empty"),
             CellEdit(9, 1, "at the end")]
    patched, notes = patch_sheet_xml(sheet, edits)
    text = patched.decode()
    assert notes == []
    assert ('<row r="2"><c r="A2" t="inlineStr"><is><t xml:space="preserve">a</t></is></c>'
            '<c r="B2" s="3"/><c r="C2"><v>3</v></c><c r="D2"><v>4.5</v></c><c r="F2" t="b"><v>1</v></c></row>') in text
    order = [text.index(f'<row r="{r}"') for r in (1, 2, 3, 5, 9)]
    assert order == sorted(order)
    assert '<row r="5"><c r="B5" t="inlineStr">' in text


def test_self_closing_sheet_data_and_prefixed_elements():
    empty = fx.sheet("").replace("<sheetData></sheetData>", "<sheetData/>").encode()
    patched, _ = patch_sheet_xml(empty, [CellEdit(1, 1, "x")])
    assert b'<sheetData><row r="1"><c r="A1" t="inlineStr">' in patched and patched.count(b"sheetData") == 2

    prefixed = fx.sheet('<x:row r="1"><x:c r="A1"><x:v>1</x:v></x:c></x:row>', prefix="x").encode()
    patched, _ = patch_sheet_xml(prefixed, [CellEdit(1, 2, "y"), CellEdit(2, 1, 2)])
    assert b'<x:c r="B1" t="inlineStr"><x:is><x:t xml:space="preserve">y</x:t></x:is></x:c></x:row>' in patched
    assert b'<x:row r="2"><x:c r="A2"><x:v>2</x:v></x:c></x:row>' in patched


def test_rows_and_cells_without_references_get_them():
    sheet = fx.sheet('<row><c><v>1</v></c><c><v>2</v></c></row><row><c><v>3</v></c></row>').encode()
    patched, _ = patch_sheet_xml(sheet, [CellEdit(1, 3, "z")])
    text = patched.decode()
    assert '<row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c><c r="C1" t="inlineStr">' in text
    assert '<row r="2"><c><v>3</v></c></row>' in text  # an untouched row gains only its number


def test_formula_cells_are_left_alone():
    sheet = fx.sheet('<row r="1"><c r="A1"><f>1+1</f><v>2</v></c></row>').encode()
    patched, notes = patch_sheet_xml(sheet, [CellEdit(1, 1, "x")])
    assert notes == ["A1 holds a formula and was left unchanged"]
    assert b"<f>1+1</f>" in patched


def test_values_that_need_escaping():
    sheet = fx.sheet("").encode()
    patched, _ = patch_sheet_xml(sheet, [CellEdit(1, 1, 'a < b & "c"\x02'), CellEdit(1, 2, float("nan"))])
    assert b"a &lt; b &amp; \"c\"_x0002_" in patched and b">nan<" in patched
    with pytest.raises(ValueError, match="start at 1"):
        patch_sheet_xml(sheet, [CellEdit(0, 1, "x")])
    assert patch_sheet_xml(sheet, []) == (sheet, [])


@pytest.mark.filterwarnings("ignore:Workbook contains no default style")  # the fixture's minimal styles part
def test_openpyxl_opens_the_patched_workbook(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    src = excel_like(tmp_path)
    dst = tmp_path / "checked.xlsx"
    patch_xlsx(src, dst, sheet=None, edits=[CellEdit(2, 4, "FAIL"), CellEdit(7, 1, 1.5)])
    sheet = openpyxl.load_workbook(dst)["Requirements"]
    assert sheet["D2"].value == "FAIL" and sheet["A7"].value == 1.5 and sheet["C4"].value == "=C2*2"
    assert sheet["A1"].value == "ID"
