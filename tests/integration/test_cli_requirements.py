"""`baslt requirements init` and `baslt requirements lint` from the command line."""

from __future__ import annotations

import csv
import json
import subprocess
import sys

import pytest

from baslt.cli import main
from baslt.tabular import list_sheets, read_table

pytestmark = pytest.mark.minimal


@pytest.mark.parametrize("template", ["generic", "polarion"])
def test_the_workbook_template_lints_clean(tmp_path, capsys, template):
    out = tmp_path / "reqs.xlsx"
    assert main(["requirements", "init", "-o", str(out), "--template", template]) == 0
    assert "Wrote" in capsys.readouterr().out
    assert [s.name for s in list_sheets(out)] == ["Requirements", "Signals", "Events", "Conditions", "Curves",
                                                  "Units", "Settings", "Guide"]
    assert main(["requirements", "lint", str(out)]) == 0
    text = capsys.readouterr().out
    assert "0 errors, 1 warning)" in text and "not covered" in text


def test_the_csv_template_writes_a_mapping(tmp_path, capsys):
    out = tmp_path / "reqs.csv"
    mapping = tmp_path / "reqs.json"
    assert main(["requirements", "init", "-o", str(out), "--mapping-output", str(mapping), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass" and payload["mapping"] == str(mapping)
    data = json.loads(mapping.read_text())
    assert "sheet" not in data.get("requirements", {}) and data["events"]["MECO"]["falls_below"] == "1 MN"
    assert main(["requirements", "lint", str(out), "-m", str(mapping), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass_with_warnings" and result["not_covered"] == ["LV-017"]
    assert result["requirements"] == 17 and result["skipped_rows"] == 3


def test_existing_files_need_force(tmp_path, capsys):
    out = tmp_path / "reqs.xlsx"
    assert main(["requirements", "init", "-o", str(out)]) == 0
    assert main(["requirements", "init", "-o", str(out)]) == 3
    assert "already exists" in capsys.readouterr().err
    assert main(["requirements", "init", "-o", str(out), "--force"]) == 0
    assert main(["requirements", "init", "-o", str(tmp_path / "reqs.txt")]) == 3


def test_a_source_fills_the_signals_sheet(tmp_path):
    run = tmp_path / "run.csv"
    run.write_text("t,thrust [N],gnc/mode\n0,1.5,0\n1,2.5,1\n")
    out = tmp_path / "mine.xlsx"
    assert main(["requirements", "init", "-o", str(out), "--source", str(run), "-q"]) == 0
    signals = read_table(out, sheet="Signals")
    assert [[c.text for c in row][:4] for row in signals.rows[1:]] == [["thrust", "thrust", "N", ""],
                                                                       ["mode", "gnc/mode", "", "discrete"]]
    assert read_table(out, sheet="Events").n_rows == 1  # the rocket's events are left out
    assert main(["requirements", "lint", str(out), "-q"]) == 0  # the example rows are not approved


def test_lint_errors_exit_3(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="") as handle:
        csv.writer(handle).writerows([["ID", "Check", "Type", "Limit"], ["B-1", "q", "range", "<= 3"],
                                      ["B-2", "q", "assert", "1"], ["B-3", "q", "upper", "<= 3 kpa"]])
    assert main(["requirements", "lint", str(path)]) == 3
    text = capsys.readouterr().out
    assert "ERROR    B-1  a range check cannot use an upper limit '<= 3' (bad.csv:2)" in text
    assert "B-2  an assert check takes no limit" in text
    assert "B-3  cannot read limit '<= 3 kpa'" in text and "Result: FAIL (3 errors, 0 warnings)" in text
    assert main(["requirements", "lint", str(path), "--only", "B-2"]) == 3
    assert main(["requirements", "lint", str(tmp_path / "missing.csv")]) == 3


def test_requirements_commands_stay_light():
    code = ("import sys, baslt.cli, baslt.reqs.api, baslt.reqs.load, baslt.reqs.lint, baslt.reqs.templates, "
            "baslt.reqs.expr; "
            "print(sorted(m for m in ('numpy', 'yaml', 'h5py', 'scipy') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
