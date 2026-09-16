"""Public compile/verify integration, atomic output and deterministic rocket evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import baslt
from baslt.api import compile, verify
from baslt.errors import InfeasibleBudget, SelfVerifyFailed

ROOT = Path(__file__).resolve().parents[2]
POLICY = {"version": 1, "hard": {"q": {"global_extrema": {}, "threshold_crossing": {"value": 1.5}}}}


def data():
    return {"t": np.arange(7, dtype=float), "q": np.array([0., 1., 2., 4., 2., 1., 0.])}


@pytest.mark.minimal
def test_public_roundtrip_and_source_verification(tmp_path):
    path = tmp_path / "run.baslt"
    result = baslt.compile(data(), POLICY, output=path)
    assert result.status == "pass"
    assert path.read_bytes() == result.bytes
    assert verify(result.bytes, source=data()).status == "pass"
    assert baslt.verify_artifact(path).status == "pass"
    artifact = baslt.read_artifact(path)
    assert artifact.signal_names() == ["q"]
    assert result.bytes == compile(data(), POLICY).bytes
    assert baslt.inspect(path)["size_bytes"] == result.size


@pytest.mark.minimal
def test_failed_compile_does_not_replace_output(tmp_path):
    path = tmp_path / "run.baslt"
    path.write_bytes(b"previous")
    with pytest.raises(InfeasibleBudget):
        compile(data(), POLICY, output=path, max_size=1)
    assert path.read_bytes() == b"previous"
    result = compile(data(), POLICY, output=path, max_size=1, on_error="record")
    assert result.status == "error" and result.exit_code == 2
    report = json.loads(path.with_suffix(".error.json").read_text())
    assert report["status"] in ("error", "infeasible")
    assert report["minimum_bytes"] > 1
    assert path.read_bytes() == b"previous"


@pytest.mark.minimal
def test_record_mode_survives_report_write_failure(tmp_path):
    result = compile(data(), {"version": 90}, on_error="record", error_report=tmp_path / "missing" / "e.json")
    assert result.status == "error"
    assert "report_error" in result.error


@pytest.mark.minimal
def test_self_verify_failure_emits_nothing(monkeypatch, tmp_path):
    from baslt.verify.checks import VerifyResult
    from baslt.verify.structure import Check
    monkeypatch.setattr("baslt.verify.checks.verify_artifact", lambda _: VerifyResult([Check("injected", "fail")]))
    path = tmp_path / "bad.baslt"
    with pytest.raises(SelfVerifyFailed):
        compile(data(), POLICY, output=path)
    assert not path.exists()


@pytest.mark.minimal
def test_declarations_apply_time_scale_and_kind():
    policy = {"version": 1, "signals": {"time": "clock", "time_unit": "ms",
              "decl": {"mode": {"kind": "discrete"}}}, "hard": {"mode": {"state_transitions": {}}}}
    result = compile({"clock": np.arange(5.) * 1000, "mode": np.array([0., 0., 1., 1., 2.])}, policy)
    artifact = baslt.read_artifact(result.bytes)
    assert artifact.index["signals"][0]["kind"] == "discrete"
    assert artifact.array("mode", "t")[-1] == 4.


def rocket_summary():
    spec = importlib.util.spec_from_file_location("rocket_example", ROOT / "examples/rocket_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcome = compile(module.simulate(), ROOT / "tests/golden/m5_extrema_crossings.yaml")
    artifact = baslt.read_artifact(outcome.bytes)
    assert verify(outcome.bytes).status == "pass"
    return {"status": outcome.status, "requirements": outcome.manifest["requirements"],
            "retained": [{"name": entry["name"], "n": entry["n"]} for entry in artifact.index["signals"]]}


def test_rocket_golden():
    assert rocket_summary() == json.loads((ROOT / "tests/golden/rocket_m5_evidence.json").read_text())
