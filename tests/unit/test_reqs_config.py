"""The mapping: files, workbook config sheets, merge order and validation messages."""

from __future__ import annotations

import json

import pytest

from baslt.errors import RequirementsError, UsageError
from baslt.reqs.config import config_from_workbook, deep_merge, load_config
from baslt.tabular import Sheet, write_xlsx

pytestmark = pytest.mark.minimal

FULL = {
    "version": 1,
    "name": "LV-3",
    "requirements": {
        "sheet": "Reqs", "header_row": 3, "where": {"Status": ["approved", "reviewed"], "Type": "Requirement"},
        "columns": {"check": "Verification Check", "limit": ["Limit", "Verification Limit"]},
        "passthrough": ["Owner"], "vocab": {"kind": {"Max": "upper"}}, "decimal_comma": True,
        "write_back": {"verdict": "Verification Result"},
    },
    "checks": {"file": "checks.xlsx", "key": "Requirement"},
    "time": {"signal": "tout", "t0": "liftoff"},
    "signals": {
        "q": {"path": "aero/q", "unit": "Pa"},
        "mode": {"path": "gnc/mode", "kind": "discrete", "labels": {0: "IDLE", 1: "BURN"}},
        "mode2": {"path": "gnc/mode2", "labels": "0=A; 1=B"},
        "mode3": {"path": "gnc/mode3", "labels": ["X", "Y"]},
        "speed": {"expr": "norm(`nav/velocity`)", "unit": "m/s"},
        "raw": "aero/raw",
    },
    "units": {"g": "9.80665 m/s^2", "furlong": {"dimension": "length", "factor": 201.168}},
    "events": {
        "MECO": {"signal": "thrust", "falls_below": "1 MN", "debounce": "250 ms"},
        "liftoff": {"when": "alt > 1 m"},
        "sep": {"param": "sep_time", "occurrence": "last"},
    },
    "conditions": {"ascent": "between('liftoff', 'MECO')"},
    "curves": {"qmax": {"points": [[0, 72], [1.2, 70]], "x_unit": "1", "y_unit": "kPa", "x": "mach"},
               "table2": {"x": [0, 1], "y": [5, 6]}},
    "params": {"key": "run", "file": "runs.csv", "units": {"payload": "kg"}},
    "defaults": {"tolerance": "50 ms", "margin": "5 %", "on_gap": "fail", "on_missing_event": "na"},
    "report": {"title": "Ascent", "pages": "all", "plot_bins": 256},
    "archive": {"max_size": "2 MiB"},
}


def test_a_complete_mapping(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(FULL))
    config = load_config(path)
    layout = config.requirements
    assert (layout.sheet, layout.header_row, layout.decimal_comma) == ("Reqs", 3, True)
    assert layout.where == {"Status": ["approved", "reviewed"], "Type": ["Requirement"]}
    assert layout.columns == {"check": ["Verification Check"], "limit": ["Limit", "Verification Limit"]}
    assert layout.vocab == {"kind": {"max": "upper"}} and layout.write_back == {"verdict": "Verification Result"}
    assert config.checks.file == "checks.xlsx" and config.checks.key == "Requirement"
    assert config.signals["mode"].labels == {0: "IDLE", 1: "BURN"}
    assert config.signals["mode2"].labels == {0: "A", 1: "B"}
    assert config.signals["mode3"].labels == {0: "X", 1: "Y"}
    assert config.signals["raw"].path == "aero/raw" and config.signals["speed"].expr.startswith("norm")
    assert config.units.dimension("g") == "acceleration" and config.units.lookup("furlong").factor == 201.168
    meco = config.events["MECO"]
    assert (meco.signal, meco.condition, meco.value, meco.debounce) == ("thrust", "falls_below", "1 MN", "250 ms")
    assert config.events["sep"].occurrence == "last" and config.events["liftoff"].when == "alt > 1 m"
    assert config.curves["qmax"].points == [(0.0, 72.0), (1.2, 70.0)] and config.curves["qmax"].x == "mach"
    assert config.curves["table2"].points == [(0.0, 5.0), (1.0, 6.0)]
    assert config.params.units == {"payload": "kg"}
    assert (config.defaults.tolerance, config.defaults.margin, config.defaults.on_gap) == ("50 ms", "5 %", "fail")
    assert (config.report.pages, config.report.plot_bins) == ("all", 256)
    assert config.archive_max_size == "2 MiB"
    assert config.base == tmp_path.resolve() and config.sources == ["map.json"]
    assert len(config.sha256) == 64


@pytest.mark.parametrize(("data", "message"), [
    ({"requirments": {}}, "unknown key 'requirments'; did you mean 'requirements'?"),
    ({"version": 2}, "mapping version 1"),
    ({"requirements": {"columns": {"chek": "X"}}}, "unknown field 'chek'; did you mean 'check'?"),
    ({"requirements": {"write_back": {"verdikt": "X"}}}, "unknown result field 'verdikt'"),
    ({"requirements": {"header_row": 0}}, "whole number of at least 1"),
    ({"requirements": {"decimal_comma": "maybe"}}, "expected true or false"),
    ({"signals": {"bad name": "x"}}, "must be a name made of letters"),
    ({"signals": {"q": {"unit": "Pa"}}}, "give either a path or an expr"),
    ({"signals": {"q": {"path": "a", "kind": "boolean"}}}, "expected one of continuous, discrete, vector"),
    ({"signals": {"m": {"path": "a", "labels": {"x": "A"}}}}, "label codes are whole numbers"),
    ({"signals": {"m": {"path": "a", "labels": {0: "A", 1: "A"}}}}, "label 'A' is used twice"),
    ({"events": {"E": {"signal": "x"}}}, "needs falls_below, rises_above or equals"),
    ({"events": {"E": {"falls_below": 3}}}, "falls_below needs a signal"),
    ({"events": {"E": {"when": "x > 1", "param": "p"}}}, "exactly one of"),
    ({"events": {"E": {"when": "x > 1", "debounce": "5 kPa"}}}, "is not a duration"),
    ({"events": {"E": {"when": "x > 1", "occurrence": "sometimes"}}}, "first, last or a trigger number"),
    ({"curves": {"c": {"points": [[0, 1]]}}}, "at least two points"),
    ({"curves": {"c": {"points": [[1, 1], [0, 2]]}}}, "strictly increasing"),
    ({"curves": {"c": {"points": [[0, 1], [1, 2]], "y_unit": "kpa"}}}, "did you mean 'kPa'"),
    ({"units": {"g": "9.8"}}, "with a unit"),
    ({"units": {"g": {"dimension": "acceleration", "factor": -1}}}, "must be positive"),
    ({"defaults": {"tolerance": "soon"}}, "cannot read tolerance"),
    ({"defaults": {"on_gap": "explode"}}, "expected one of warn, ignore, fail"),
    ({"report": {"plot_bins": 3}}, "between 64 and 8192"),
    ({"signals": {"x": "a"}, "conditions": {"x": "t > 1"}}, "both a signal alias and a condition"),
])
def test_validation_messages(data, message):
    with pytest.raises(RequirementsError, match=message.replace("(", r"\(").replace(")", r"\)").replace("?", r"\?")):
        load_config(data)


def test_every_problem_is_reported_with_its_location(tmp_path):
    yaml = pytest.importorskip("yaml")
    assert yaml
    path = tmp_path / "map.yaml"
    path.write_text("version: 1\nrequirements:\n  sheet: 1\n  colums: {}\nsignals:\n  q: {unit: Pa}\n")
    with pytest.raises(RequirementsError) as caught:
        load_config(path)
    text = str(caught.value)
    assert "requirements.colums: unknown key 'colums'; did you mean 'columns'? (map.yaml:4:3)" in text
    assert "signals.q: give either a path or an expr (map.yaml:6:3)" in text


def test_yaml_needs_pyyaml(tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__

    def no_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    path = tmp_path / "map.yaml"
    path.write_text("version: 1\n")
    with pytest.raises(UsageError, match=r"baslt\[yaml\]"):
        load_config(path)


def test_file_errors(tmp_path):
    with pytest.raises(UsageError, match="mapping file not found"):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "map.txt").write_text("{}")
    with pytest.raises(UsageError, match="cannot tell the mapping format"):
        load_config(tmp_path / "map.txt")
    (tmp_path / "list.json").write_text("[1]")
    with pytest.raises(RequirementsError, match="must hold a mapping"):
        load_config(tmp_path / "list.json")
    (tmp_path / "empty.json").write_text("{}")
    assert load_config(tmp_path / "empty.json").signals == {}


def workbook(tmp_path):
    header = lambda *names: list(names)  # noqa: E731
    sheets = [
        Sheet("Requirements", [["ID", "Check"], ["R1", "q"]]),
        Sheet("Signals", [header("Alias", "Path", "Unit", "Kind", "Labels", "Expression"),
                          ["q", "aero/q", "Pa", "", "", ""], ["mode", "gnc/mode", "", "discrete", "0=A; 1=B", ""],
                          ["speed", "", "m/s", "", "", "norm(`v`)"], ["", "ignored", "", "", "", ""]]),
        Sheet("Events", [header("Name", "Signal", "Condition", "Value", "When", "Debounce"),
                         ["MECO", "thrust", "falls below", "1 MN", "", "250 ms"], ["lift", "", "", "", "alt > 1", ""]]),
        Sheet("Conditions", [header("Name", "Expression"), ["ascent", "after('lift')"]]),
        Sheet("Curves", [header("Name", "X", "Y", "X unit", "Y unit"), ["c", "0", "1", "1", "kPa"],
                         ["c", "2", "3", "", ""]]),
        Sheet("Units", [header("Symbol", "Definition"), ["gee", "9.80665 m/s^2"]]),
        Sheet("Settings", [header("Key", "Value"), ["requirements.where.Status", "approved, reviewed"],
                           ["defaults.margin", "5 %"], ["requirements.decimal_comma", "yes"],
                           ["requirements.columns.check", "Check | Signal"]]),
    ]
    return write_xlsx(tmp_path / "reqs.xlsx", sheets)


def test_config_sheets(tmp_path):
    path = workbook(tmp_path)
    data, locations = config_from_workbook(path)
    assert data["signals"]["mode"] == {"path": "gnc/mode", "kind": "discrete", "labels": {"0": "A", "1": "B"}}
    assert data["events"]["MECO"] == {"signal": "thrust", "falls_below": "1 MN", "debounce": "250 ms"}
    assert data["curves"]["c"]["points"] == [["0", "1"], ["2", "3"]]
    assert locations["signals.mode"] == "reqs.xlsx:Signals!row 3"
    config = load_config(workbook=path)
    assert config.signals["speed"].expr == "norm(`v`)" and config.events["lift"].when == "alt > 1"
    assert config.curves["c"].points == [(0.0, 1.0), (2.0, 3.0)] and config.curves["c"].y_unit == "kPa"
    assert config.requirements.where == {"Status": ["approved", "reviewed"]}
    assert config.requirements.columns == {"check": ["Check", "Signal"]}
    assert config.requirements.decimal_comma and config.defaults.margin == "5 %"
    assert config.units.dimension("gee") == "acceleration"
    assert config.sources == ["reqs.xlsx (config sheets)"]


def test_merge_order(tmp_path):
    path = workbook(tmp_path)
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"signals": {"q": {"path": "other/q"}}, "defaults": {"margin": "1 kPa"}}))
    config = load_config(mapping, workbook=path, overrides={"defaults": {"margin": "2 kPa"}})
    assert config.signals["q"].path == "other/q" and config.signals["q"].unit == "Pa"  # merged key by key
    assert config.signals["mode"].kind == "discrete"  # from the workbook
    assert config.defaults.margin == "2 kPa"
    assert config.sources == ["reqs.xlsx (config sheets)", "map.json"]


def test_config_sheet_problems_point_at_their_row(tmp_path):
    path = write_xlsx(tmp_path / "bad.xlsx", [
        Sheet("Requirements", [["ID", "Check"]]),
        Sheet("Settings", [["Key", "Value"], ["report.title", "x"], ["defaults.on_gap", "explode"]]),
        Sheet("Signals", [["Alias", "Unit"], ["q", "Pa"]]),
    ])
    with pytest.raises(RequirementsError) as caught:
        load_config(workbook=path)
    text = str(caught.value)
    assert "defaults.on_gap: expected one of warn, ignore, fail, got 'explode' (bad.xlsx:Settings!row 3)" in text
    assert "signals.q: give either a path or an expr (bad.xlsx:Signals!row 2)" in text


def test_deep_merge_does_not_touch_its_inputs():
    base = {"a": {"b": [1]}, "c": 1}
    update = {"a": {"d": 2}}
    merged = deep_merge(base, update)
    assert merged == {"a": {"b": [1], "d": 2}, "c": 1} and base == {"a": {"b": [1]}, "c": 1}
    merged["a"]["b"].append(2)
    assert base["a"]["b"] == [1]


def test_source_settings(tmp_path):
    config = load_config({"source": {"format": "csv", "delimiter": ";", "encoding": "cp1252",
                                     "units_row": "yes", "decimal_comma": "no"}})
    assert config.source.options() == {"format": "csv", "delimiter": ";", "encoding": "cp1252",
                                       "units_row": True, "decimal_comma": False}
    assert load_config({}).source.options() == {}
    assert load_config({"source": {"units_row": "auto"}}).source.options() == {}
    with pytest.raises(RequirementsError, match="source.units_row: expected true or false"):
        load_config({"source": {"units_row": "sometimes"}})
    with pytest.raises(RequirementsError, match="unknown key 'delimeter'"):
        load_config({"source": {"delimeter": ";"}})
