"""`baslt check` from the command line: outputs, exit codes and options."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from baslt.cli import main
from reference.reqs_check import write_rows

pytestmark = pytest.mark.minimal

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def write_run(path: Path) -> Path:
    t = np.arange(11.0)
    x = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=float)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t [s]", "x [m]", "gate"])
        for i in range(t.shape[0]):
            writer.writerow([t[i], x[i], 1 if i > 3 else 0])
    return path


ROWS = {
    "pass": {"id": "P", "check": "x", "limit": "<= 10 m"},
    "warn": {"id": "W", "check": "x", "limit": "<= 5.1 m", "margin": "0.5 m"},
    "fail": {"id": "F", "check": "x", "limit": "<= 4 m"},
    "error": {"id": "E", "check": "nope", "limit": "<= 1"},
    "na": {"id": "N", "check": "x", "limit": "<= 1 m", "when": "x > 100 m"},
}


@pytest.fixture()
def files(tmp_path):
    return write_run(tmp_path / "run.csv"), tmp_path


def reqs(folder: Path, *names: str) -> Path:
    return write_rows(folder / "reqs.csv", [ROWS[name] for name in names])


@pytest.mark.parametrize(("names", "options", "code"), [
    (("pass",), [], 0),
    (("pass", "warn"), [], 0),
    (("pass", "warn"), ["--fail-on", "warn"], 1),
    (("pass", "fail"), [], 1),
    (("pass", "fail"), ["--fail-on", "none"], 0),
    (("pass", "error"), [], 3),
    (("fail", "error"), [], 1),
    (("na",), [], 0),
])
def test_exit_codes(files, capsys, names, options, code):
    run, folder = files
    assert main(["check", str(run), "-r", str(reqs(folder, *names)), "-o", str(folder / "out"), *options]) == code
    assert "Result:" in capsys.readouterr().out


def test_outputs_and_the_terminal_table(files, capsys):
    run, folder = files
    table = reqs(folder, "pass", "warn", "fail", "error", "na")
    assert main(["check", str(run), "-r", str(table)]) == 1
    text = capsys.readouterr().out
    out = folder / "run.check"
    assert sorted(p.name for p in out.iterdir()) == ["report.html", "reqs.checked.csv", "results.csv",
                                                    "results.json", "results.xlsx"]
    assert text.splitlines()[0] == "Run      run.csv   csv, 10.0 s, 1 signals"
    assert f"Output   {out}" in text
    assert "report.html, results.xlsx, reqs.checked.csv, results.json, results.csv" in text
    page = (out / "report.html").read_text(encoding="utf-8")
    assert '<details class="card v-fail" id="req-F"' in page and 'id="req-E"' in page
    lines = [line for line in text.splitlines() if line[:4] in ("FAIL", "ERRO", "WARN", "N/A ")]
    assert [line.split()[1] for line in lines] == ["F", "E", "W", "N"]
    assert "PASS" not in text
    assert main(["check", str(run), "-r", str(table), "--all", "--no-xlsx", "--no-annotate", "--no-html"]) == 1
    assert "PASS     P" in capsys.readouterr().out
    data = json.loads((out / "results.json").read_text())
    assert data["counts"] == {"pass": 1, "warn": 1, "fail": 1, "not_applicable": 1, "error": 1, "not_covered": 0}
    assert data["digest"].startswith("sha256-sampled-v1 ")


def test_json_output(files, capsys):
    run, folder = files
    code = main(["check", str(run), "-r", str(reqs(folder, "pass", "fail")), "-o", str(folder / "o"), "--json",
                 "--hash", "none", "--no-xlsx", "--no-html"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail" and payload["exit_code"] == 1
    assert payload["counts"]["fail"] == 1 and set(payload["outputs"]) == {"json", "csv", "annotated"}
    run_json = payload["runs"][0]
    assert run_json["digest"] is None and [r["id"] for r in run_json["results"]] == ["P", "F"]


def test_only_selected_requirements(files, capsys):
    run, folder = files
    table = reqs(folder, "pass", "fail", "error")
    assert main(["check", str(run), "-r", str(table), "-o", str(folder / "o"), "--only", "P", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in payload["runs"][0]["results"]] == ["P"]


def test_parameters_from_a_table(files, capsys):
    run, folder = files
    table = write_rows(folder / "reqs.csv", [
        {"id": "C", "check": "x", "case": "heavy", "applies_to": "payload == 'heavy'", "limit": "<= 4 m"},
        {"id": "C", "case": "other", "limit": "<= 10 m"},
    ])
    params = folder / "params.csv"
    params.write_text("run,payload\nrun,heavy\nother,light\n", encoding="utf-8")
    assert main(["check", str(run), "-r", str(table), "-o", str(folder / "o"), "--params", str(params)]) == 1
    capsys.readouterr()
    params.write_text("run,payload\nrun.csv,light\n", encoding="utf-8")
    assert main(["check", str(run), "-r", str(table), "-o", str(folder / "o"), "--params", str(params),
                 "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["params"] == {"payload": "light"}


def test_problems_before_checking(files, capsys):
    run, folder = files
    table = reqs(folder, "pass")
    assert main(["check", str(run), "-r", str(folder / "missing.csv")]) == 3
    assert "baslt: error" in capsys.readouterr().err
    assert main(["check", str(folder / "missing.csv"), "-r", str(table), "-o", str(folder / "o")]) == 3
    assert "run file not found" in capsys.readouterr().err and not (folder / "o").exists()
    bad = folder / "bad.mat"
    bad.write_bytes(b"not a MAT-file")
    assert main(["check", str(bad), "-r", str(table), "-o", str(folder / "o")]) == 3
    text = capsys.readouterr().out
    assert "ERROR    the run could not be checked: " in text
    assert "1 requirements, 0 not covered" in text and "reqs.checked.csv" not in text
    assert main(["check", str(folder / "none*.csv"), "-r", str(table)]) == 3  # usage problems exit with 3 too
    assert "no run files match" in capsys.readouterr().err
    assert main(["check", str(run), "-r", str(table), "--jobs", "many"]) == 3
    assert "--jobs takes a number or auto" in capsys.readouterr().err
    assert main(["check", str(run), "-r", str(table), "--archive"]) == 3
    capsys.readouterr()
    assert main(["check", str(run), "-r", str(table), "--fail-on", "sometimes"]) == 3


def test_the_rocket_example_from_csv(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("rocket_cli_example", EXAMPLES / "rocket_sim.py")
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    run = sim.write_csv(tmp_path / "q_spike.csv", sim.simulate(anomaly="q_spike"))
    table = EXAMPLES / "rocket_requirements.csv"
    mapping = EXAMPLES / "rocket_mapping.json"
    assert main(["requirements", "lint", str(table), "-m", str(mapping), "--source", str(run)]) == 0
    assert "Result: OK (0 errors, 1 warning)" in capsys.readouterr().out
    assert main(["check", str(run), "-r", str(table), "-m", str(mapping), "-o", str(tmp_path / "out")]) == 1
    text = capsys.readouterr().out
    failing = [line.split()[1] for line in text.splitlines() if line.startswith("FAIL")]
    assert failing == ["LV-004", "LV-001"]
    assert "Result: FAIL (13 pass, 1 warn, 2 fail, 0 error, 0 n/a)" in text
    checked = list(csv.reader((tmp_path / "out" / "rocket_requirements.checked.csv").open(encoding="utf-8")))
    assert checked[0][-4:] == ["Verdict", "Result", "Margin", "Evidence"]
    assert checked[2][0] == "LV-001" and checked[2][-4:-2] == ["FAIL", "72.66 kPa"]


def test_lint_against_a_run(files, capsys):
    run, folder = files
    table = write_rows(folder / "reqs.csv", [
        {"id": "A", "check": "x", "limit": "<= 4 s"},
        {"id": "B", "check": "y", "limit": "<= 4"},
        {"id": "C", "check": "x", "limit": "<= 4", "applies_to": "payload == 'heavy'"},
        {"id": "D", "check": "gate == 1", "type": "assert"},
    ])
    assert main(["requirements", "lint", str(table)]) == 0
    capsys.readouterr()
    params = folder / "params.csv"
    params.write_text("run,mass\nrun,5\n", encoding="utf-8")
    assert main(["requirements", "lint", str(table), "--source", str(run), "--params", str(params)]) == 3
    text = capsys.readouterr().out
    assert "ERROR    A" in text and "'4 s' is a time but the values are in m" in text
    assert "ERROR    B" in text and "unknown name 'y'" in text
    assert "ERROR    C" in text and "unknown run parameter 'payload'" in text
    assert "    D " not in text
