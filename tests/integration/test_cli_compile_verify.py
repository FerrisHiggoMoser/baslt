"""CLI roundtrips, status JSON and documented exit codes."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.minimal


def cli(*args):
    return subprocess.run([sys.executable, "-m", "baslt", *map(str, args)], capture_output=True, text=True)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "run.csv"
    source.write_text("t,q\n0,0\n1,2\n2,0\n")
    policy = tmp_path / "p.json"
    policy.write_text(json.dumps({"version": 1, "hard": {"q": {"global_extrema": {}}}}))
    return source, policy


def test_cli_roundtrip(inputs):
    source, policy = inputs
    result = cli("compile", source, "--policy", policy, "--json")
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    artifact = source.with_suffix(".baslt")
    result = cli("verify", artifact, "--source", source, "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "pass"
    assert cli("inspect", source, "--json").returncode == 0
    assert cli("inspect", artifact, "--json").returncode == 0
    assert cli("verify", artifact, "-q").stdout == ""
    assert "Result: PASS" in cli("verify", artifact).stdout


def test_budget_failure_json_and_cli_override(inputs):
    source, policy = inputs
    result = cli("compile", source, "--policy", policy, "--max-size", "1 B", "--json")
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "error"
    assert not source.with_suffix(".baslt").exists()
    result = cli("compile", source, "--policy", policy, "--max-size", "1 MiB", "--json")
    assert result.returncode == 0
    assert json.loads(result.stdout)["manifest"]["budget"]["source"] == "cli"


@pytest.mark.parametrize("args", [("compile",), ("bogus",), ("inspect", "/nonexistent"),
                                  ("verify", "/nonexistent"), ("compile", "run", "--policy", "/missing")])
def test_usage_and_missing_inputs_return_three(args):
    result = cli(*args, "--json")
    assert result.returncode == 3
    assert json.loads(result.stdout)["status"] == "error"


def test_corrupt_artifact_is_integrity_failure(tmp_path):
    path = tmp_path / "corrupt.baslt"
    path.write_bytes(b"invalid")
    result = cli("verify", path, "--json")
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "fail"
