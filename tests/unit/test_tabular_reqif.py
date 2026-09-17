"""ReqIF exports read as tables and as requirements."""

from __future__ import annotations

import pytest

from baslt.errors import TableError
from baslt.reqs.config import load_config
from baslt.reqs.load import load_requirements
from baslt.tabular import list_sheets, read_table
from reference.reqif_fixtures import write_reqif, write_reqifz

pytestmark = pytest.mark.minimal

HEADER = ["ReqIF.ForeignID", "ReqIF.Name", "ReqIF.ChapterName", "Check", "Limit", "ReqIF.Text", "Status", "Priority",
          "ReqIF Type", "ReqIF Level"]


def test_objects_become_rows_in_hierarchy_order(tmp_path):
    table = read_table(write_reqif(tmp_path / "export.reqif"))
    assert table.format == "reqif" and table.row_texts(1) == HEADER
    rows = [table.row_texts(r) for r in range(2, table.n_rows + 1)]
    assert [row[0] for row in rows] == ["", "LV-001", "LV-002", "LV-003", "LV-009"]
    heading, first, second, prose, orphan = rows
    assert heading[2] == "Structural loads" and heading[8:] == ["Heading", "1"]
    assert first[5] == "The dynamic pressure shall not exceed\n70 kPa."
    assert first[6] == "approved" and first[7] == "2" and first[9] == "2"
    assert second[4] == "[-7, 7] deg" and second[6] == "draft"
    assert prose[6] == "approved, reviewed" and prose[9] == "3"
    assert orphan[9] == "" and orphan[8] == "Requirement"
    assert table.location(3, 5) == "export.reqif:LV-001 (Limit)"
    assert table.location(2) == "export.reqif:obj-h1"
    assert [s.name for s in list_sheets(tmp_path / "export.reqif")] == ["export"]


def test_reqifz_archives(tmp_path):
    table = read_table(write_reqifz(tmp_path / "export.reqifz"))
    assert table.n_rows == 6 and table.text(3, 1) == "LV-001"


@pytest.mark.parametrize(("text", "message"), [
    ('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><REQ-IF/>', "DOCTYPE"),
    ("<REQ-IF><unclosed></REQ-IF>", "not valid ReqIF XML"),
    ("<SOMETHING/>", "not a ReqIF file"),
    ("<REQ-IF><CORE-CONTENT/></REQ-IF>", "no REQ-IF-CONTENT"),
])
def test_bad_files(tmp_path, text, message):
    path = tmp_path / "bad.reqif"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TableError, match=message):
        read_table(path)


def test_an_empty_archive(tmp_path):
    import zipfile

    with zipfile.ZipFile(tmp_path / "empty.reqifz", "w") as archive:
        archive.writestr("readme.txt", "nothing")
    with pytest.raises(TableError, match="holds no .reqif file"):
        read_table(tmp_path / "empty.reqifz")


def test_a_polarion_export_as_requirements(tmp_path):
    path = write_reqif(tmp_path / "export.reqif")
    config = load_config({"requirements": {"where": {"Status": ["approved", "approved, reviewed"],
                                                     "ReqIF Type": ["Requirement"]}}})
    reqset = load_requirements(path, config)
    assert [req.id for req in reqset.requirements] == ["LV-001", "LV-003", "LV-009"]
    first = reqset.requirements[0]
    assert first.title == "Max dynamic pressure" and first.check == "q" and first.covered
    assert first.loc.text == "export.reqif:LV-001"
    assert reqset.not_covered == ["LV-003"]
