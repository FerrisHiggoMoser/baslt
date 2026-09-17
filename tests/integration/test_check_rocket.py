"""The rocket example requirements on the simulated ascent: the designed failures and nothing else."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from baslt.reqs.api import init_template, load
from baslt.reqs.run import check_run

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
GOLDEN = ROOT / "tests/golden/rocket_check_results.json"
ANOMALIES = (None, "q_spike", "alpha_chatter")


def _simulator():
    spec = importlib.util.spec_from_file_location("rocket_check_example", EXAMPLES / "rocket_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rounded(value):
    if isinstance(value, float) and math.isfinite(value):
        return float(f"{value:.9g}")
    return value


def summary(results) -> dict:
    return {r.id: {"verdict": r.verdict, "value": _rounded(r.value), "margin": _rounded(r.margin),
                   "at": _rounded(r.at), "reason": r.reason} for r in results}


@pytest.fixture(scope="module")
def runs():
    sim = _simulator()
    return {str(anomaly): sim.simulate(anomaly=anomaly) for anomaly in ANOMALIES}


@pytest.fixture(scope="module")
def checked(runs):
    reqset = load(EXAMPLES / "rocket_requirements.csv", mapping=EXAMPLES / "rocket_mapping.yaml")
    return {name: check_run(data, reqset)[0] for name, data in runs.items()}


def test_the_designed_failures(checked):
    problems = {name: {r.id: r.verdict for r in run.results if r.verdict != "pass"} for name, run in checked.items()}
    assert problems == {
        "None": {"LV-010": "warn"},
        "q_spike": {"LV-001": "fail", "LV-004": "fail", "LV-010": "warn"},
        "alpha_chatter": {"LV-009": "fail", "LV-010": "warn"},
    }
    for run in checked.values():
        assert run.not_covered == ["LV-017"] and run.error is None and len(run.results) == 16
        assert run.t0 == pytest.approx(1.0885, abs=1e-3) and run.events["MECO"]["count"] == 1
    spike = {r.id: r for r in checked["q_spike"].results}["LV-001"]
    assert spike.value == pytest.approx(72659.66, rel=1e-6) and spike.margin == pytest.approx(-2659.66, rel=1e-6)
    assert spike.context["conditions"][:1] == ["ascent"]
    tolerated = {r.id: r for r in checked["alpha_chatter"].results}["LV-002"]
    assert tolerated.verdict == "pass" and tolerated.totals["tolerated"] >= 1


def test_results_match_the_golden_file(checked):
    expected = json.loads(GOLDEN.read_text())
    got = {name: summary(run.results) for name, run in checked.items()}
    assert got.keys() == expected.keys()
    for name in expected:
        assert got[name].keys() == expected[name].keys()
        for rid, want in expected[name].items():
            have = got[name][rid]
            assert have["verdict"] == want["verdict"] and have["reason"] == want["reason"], (name, rid)
            for key in ("value", "margin", "at"):
                if isinstance(want[key], float):
                    assert have[key] == pytest.approx(want[key], rel=1e-6, abs=1e-9), (name, rid, key)
                else:
                    assert have[key] == want[key], (name, rid, key)


def test_the_json_mapping_and_the_workbook_give_the_same_results(tmp_path, runs, checked):
    by_json = load(EXAMPLES / "rocket_requirements.csv", mapping=EXAMPLES / "rocket_mapping.json")
    workbook = tmp_path / "reqs.xlsx"
    init_template(workbook)
    by_workbook = load(workbook)
    data = runs["q_spike"]
    expected = summary(checked["q_spike"].results)
    assert summary(check_run(data, by_json)[0].results) == expected
    assert summary(check_run(data, by_workbook)[0].results) == expected


def test_the_example_files_match_the_template(tmp_path):
    init_template(tmp_path / "rocket_requirements.csv", mapping_output=tmp_path / "rocket_mapping.yaml")
    init_template(tmp_path / "again.csv", mapping_output=tmp_path / "rocket_mapping.json")
    for name in ("rocket_requirements.csv", "rocket_mapping.yaml", "rocket_mapping.json"):
        assert (tmp_path / name).read_bytes() == (EXAMPLES / name).read_bytes(), name


def test_only_the_needed_signals_are_loaded(checked):
    assert all(run.signals == 8 for run in checked.values())
