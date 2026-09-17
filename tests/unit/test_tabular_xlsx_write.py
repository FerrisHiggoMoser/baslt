"""Writing workbooks: valid parts, deterministic bytes, cell types, styles, links, limits."""

from __future__ import annotations

import io
import shutil
import subprocess
import zipfile

import pytest

from baslt.tabular import Link, Sheet, Styled, read_table, write_xlsx, xlsx_bytes
from baslt.tabular import xlsx_write
from baslt.tabular.xlsx_write import STYLES, sanitize_sheet_names

pytestmark = pytest.mark.minimal

ROWS = [
    ["ID", "Value", "Flag", "Note"],
    ["LV-001", 70.5, True, Styled("FAIL", "fail")],
    ["LV-002", float("nan"), False, Link("run page", "runs/run_0001.html")],
    ["LV-003", float("-inf"), None, Link("back", "#'Other sheet'!A1")],
    ["LV-004", 2**60, -3, "tab\there\x01 and _x000D_ literal"],
]


def parts(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def workbook():
    return [Sheet("Results", ROWS), Sheet("Other sheet", [["x"], ["y"]], freeze=(0, 0), autofilter=False)]


def test_bytes_are_deterministic_and_parts_are_well_formed(tmp_path):
    data = xlsx_bytes(workbook())
    assert data == xlsx_bytes(workbook())
    names = list(parts(data))
    assert names[0] == "[Content_Types].xml"
    assert {"xl/workbook.xml", "xl/styles.xml", "xl/sharedStrings.xml", "xl/worksheets/sheet1.xml",
            "xl/worksheets/_rels/sheet1.xml.rels", "docProps/core.xml", "docProps/app.xml"} <= set(names)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    if shutil.which("xmllint") is None:
        pytest.skip("xmllint is not installed")
    for name, content in parts(data).items():
        subprocess.run(["xmllint", "--noout", "-"], input=content, check=True, capture_output=True)


def test_values_read_back(tmp_path):
    path = write_xlsx(tmp_path / "out.xlsx", workbook())
    table = read_table(path)
    assert table.sheet == "Results"
    assert [[c.text if c else None for c in row] for row in table.rows] == [
        ["ID", "Value", "Flag", "Note"],
        ["LV-001", "70.5", "TRUE", "FAIL"],
        ["LV-002", "NaN", "FALSE", "run page"],
        ["LV-003", "-inf", None, "back"],
        ["LV-004", str(2**60), "-3", "tab\there\x01 and _x000D_ literal"],
    ]
    assert table.rows[1][1].value == 70.5 and table.rows[4][2].value == -3.0


def test_styles_links_and_layout():
    content = parts(xlsx_bytes(workbook()))
    sheet1 = content["xl/worksheets/sheet1.xml"].decode()
    assert f'<c r="D2" s="{STYLES["fail"]}" t="s">' in sheet1
    assert '<hyperlink ref="D3" r:id="rId1"/>' in sheet1
    assert "location=\"'Other sheet'!A1\"" in sheet1
    assert 'state="frozen"' in sheet1 and 'ySplit="1"' in sheet1
    assert '<autoFilter ref="A1:D5"/>' in sheet1 and '<dimension ref="A1:D5"/>' in sheet1
    rels = content["xl/worksheets/_rels/sheet1.xml.rels"].decode()
    assert 'Target="runs/run_0001.html" TargetMode="External"' in rels
    book = content["xl/workbook.xml"].decode()
    assert "_xlnm._FilterDatabase" in book and "&apos;Results&apos;" not in book
    assert "'Results'!$A$1:$D$5" in book
    sheet2 = content["xl/worksheets/sheet2.xml"].decode()
    assert "autoFilter" not in sheet2 and "frozen" not in sheet2
    assert b"<dc:creator>baslt</dc:creator>" in content["docProps/core.xml"]


def test_sheet_names_are_made_valid():
    assert sanitize_sheet_names(["a/b", "A_B", "", "'quoted'", "x" * 40, "x" * 40]) == [
        "a_b", "A_B (2)", "Sheet", "quoted", "x" * 31, "x" * 27 + " (2)"]


def test_hidden_sheets_and_empty_workbooks():
    content = parts(xlsx_bytes([Sheet("Meta", [["k"]], hidden=True), Sheet("Shown", [["v"]])]))
    book = content["xl/workbook.xml"].decode()
    assert 'state="hidden"' in book and 'activeTab="1"' in book
    with pytest.raises(ValueError, match="at least one sheet"):
        xlsx_bytes([])
    with pytest.raises(ValueError, match="visible"):
        xlsx_bytes([Sheet("Meta", [], hidden=True)])
    empty = parts(xlsx_bytes([Sheet("Empty", [])]))["xl/worksheets/sheet1.xml"].decode()
    assert "<sheetData></sheetData>" in empty and "autoFilter" not in empty


def test_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(xlsx_write, "MAX_ROWS", 3)
    monkeypatch.setattr(xlsx_write, "MAX_TEXT", 5)
    path = write_xlsx(tmp_path / "cut.xlsx", [Sheet("S", [["abcdefgh"], ["2"], ["3"], ["4"], ["5"]])])
    rows = [[c.text for c in row] for row in read_table(path).rows]
    assert rows == [["abcd…"], ["2"], ["… 3 more rows are not shown (Excel's row limit)"]]


def test_openpyxl_reads_the_workbook(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = write_xlsx(tmp_path / "out.xlsx", workbook(), title="Results")
    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["Results", "Other sheet"]
    sheet = book["Results"]
    assert sheet["B2"].value == 70.5 and sheet["C3"].value is False and sheet["D2"].value == "FAIL"
    assert sheet["D2"].fill.fgColor.rgb == "FFFFC7CE"
    assert sheet["D3"].hyperlink.target == "runs/run_0001.html"
    assert sheet.freeze_panes == "A2" and sheet.auto_filter.ref == "A1:D5"
    assert book.properties.title == "Results" and book.properties.creator == "baslt"
