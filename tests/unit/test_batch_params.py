"""Run parameters from a table, matched to runs by id, file name or path."""

from __future__ import annotations

import pytest

from baslt.errors import RequirementsError
from baslt.reqs.config import load_config
from baslt.reqs.params import load_params, row_for

pytestmark = pytest.mark.minimal


def table(tmp_path, text, name="params.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_rows_are_found_by_id_name_stem_or_relative_path(tmp_path):
    config = load_config({})
    params = load_params(table(tmp_path, "Run ID,mass,family\nr1,1500,heavy\nr2.csv,900,\nsub/r3,1.5e3,x\n"),
                         config)
    assert params.key == "Run ID" and params.names() == {"mass", "family"}
    assert row_for(params, tmp_path / "r1.h5", config) == {"mass": 1500.0, "family": "heavy"}
    assert row_for(params, tmp_path / "r2.csv", config) == {"mass": 900.0, "family": ""}
    assert row_for(params, tmp_path / "sub" / "r3.mat", config, root=tmp_path) == {"mass": 1500.0, "family": "x"}
    assert row_for(params, tmp_path / "other.csv", config, run_id="r1") == {"mass": 1500.0, "family": "heavy"}
    assert row_for(params, tmp_path / "unknown.csv", config) == {}


def test_the_key_column_can_be_named(tmp_path):
    config = load_config({"params": {"key": "file"}})
    params = load_params(table(tmp_path, "case,file,mass\nA,r1.csv,1\n"), config)
    assert row_for(params, tmp_path / "r1.csv", config) == {"case": "A", "mass": 1.0}
    with pytest.raises(RequirementsError, match="params.key names column 'file'"):
        load_params(table(tmp_path, "case,mass\nA,1\n", "other.csv"), config)


def test_problems(tmp_path):
    config = load_config({})
    with pytest.raises(RequirementsError, match="run 'r1' appears twice"):
        load_params(table(tmp_path, "run,mass\nr1,1\nr1,2\n"), config)
    with pytest.raises(RequirementsError, match="empty"):
        load_params(table(tmp_path, "", "empty.csv"), config)


def test_a_workbook_table(tmp_path):
    from baslt.tabular import Sheet, write_xlsx

    path = write_xlsx(tmp_path / "params.xlsx", [Sheet("Runs", [["run", "mass", "ok"], ["r1", 1500, True]])])
    params = load_params(path, load_config({}))
    assert row_for(params, tmp_path / "r1.csv", load_config({}))["mass"] == 1500.0
