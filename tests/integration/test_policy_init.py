"""Starter policies: valid for every source, compile as written, never overwrite by accident."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from baslt.api import compile, init_policy, verify
from baslt.errors import SourceError, UsageError
from baslt.policy import load_policy
from baslt.policy.scaffold import budget_for, render_yaml, starter_policy
from baslt.signals import SignalInfo

yaml = pytest.importorskip("yaml")
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rocket_sim.py"


def _rocket():
    if "rocket_sim" in sys.modules:
        return sys.modules["rocket_sim"]
    spec = importlib.util.spec_from_file_location("rocket_sim", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rocket_sim"] = module
    spec.loader.exec_module(module)
    return module


def _info(name, kind="continuous", unit="Pa", shape=(10,), dtype="<f8", time_ref="t"):
    return SignalInfo(name=name, path=name, shape=shape, dtype=dtype, unit=unit, kind=kind, n=shape[0],
                      time_ref=time_ref)


@pytest.mark.parametrize("size, expected", [
    # a fiftieth of the source: 256 KiB covers sources up to 50 * 256 KiB = 12.5 MiB
    (None, "1 MiB"), (0, "1 MiB"), (1000, "256 KiB"), (50 * 256 * 1024, "256 KiB"), (50 * 256 * 1024 + 1, "512 KiB"),
    (100 * 2**20, "2 MiB"), (100 * 2**20 + 1, "4 MiB"), (400 * 2**20, "8 MiB"), (10 * 2**40, "8 MiB"),
])
def test_budget_scales_with_the_source(size, expected):
    assert budget_for(size) == expected


def test_operator_choice_and_untimed_signals():
    infos = [
        _info("aero/q"), _info("nav/pos", kind="vector", shape=(10, 3)),
        _info("gnc/mode", kind="discrete", unit=None, dtype="<i8"), _info("orphan", time_ref=None),
    ]
    policy = starter_policy(infos, name="flight", source_size=None)
    assert policy["hard"] == {
        "aero/q": {"global_extrema": {}}, "nav/pos": {"global_extrema": {}}, "gnc/mode": {"state_transitions": {}},
    }
    assert "orphan" in starter_policy(infos, require_time=False)["hard"]
    text = render_yaml(policy, infos, source_label="run.h5")
    assert "# Left out because no time signal was found: orphan." in text
    assert load_policy(text, format="yaml").sha256 == load_policy(policy).sha256


def test_awkward_names_are_quoted_so_yaml_reads_them_back():
    names = ["on", "yes", "Null", "1e3", "Gain 2", "a: b", "x #y", "ünit", "-lead", "007"]
    infos = [_info(name) for name in names]
    policy = starter_policy(infos, name="true")
    loaded = yaml.safe_load(render_yaml(policy, infos, source_label="odd.h5"))
    assert list(loaded["hard"]) == names
    assert loaded["name"] == "true"
    assert load_policy(loaded).name == "true"


def test_missing_units_get_a_placeholder_hint():
    infos = [_info("prop/thrust", unit=None), _info("gnc/mode", kind="discrete", unit=None)]
    text = render_yaml(starter_policy(infos), infos, source_label="run.mat")
    assert "# 1 signal(s) have no unit in the source." in text
    assert "#       thrust: {path: prop/thrust, unit: <unit>}" in text
    with_units = [_info("aero/q")]
    assert "no unit" not in render_yaml(starter_policy(with_units), with_units, source_label="run.h5")


@pytest.mark.parametrize("fmt", ["h5", "csv", "mat", "mat73"])
def test_starter_policy_compiles_and_verifies_for_every_format(tmp_path, fmt):
    rs = _rocket()
    data = rs.simulate(duration=20.0, rate=100.0)
    path = tmp_path / f"run.{ {'mat73': 'mat'}.get(fmt, fmt) }"
    if fmt == "h5":
        rs.write_h5(path, data)
    elif fmt == "csv":
        rs.write_csv(path, data)
    else:
        rs.write_mat(path, data, version="7.3" if fmt == "mat73" else "5")
    result = init_policy(path, output=tmp_path / "policy.yaml")
    assert result["protected"] == result["signals"] > 0
    compiled = compile(path, tmp_path / "policy.yaml", output=tmp_path / "run.baslt")
    assert compiled.status == "pass", compiled.to_json()
    assert verify(tmp_path / "run.baslt", source=path).status == "pass"


def test_in_memory_sources_and_json_output(tmp_path):
    t = np.linspace(0.0, 1.0, 11)
    result = init_policy({"t": t, "x": np.sin(t), "y": (t, np.cos(t))}, output=tmp_path / "p.json")
    assert result["format"] == "json" and result["output"] == str(tmp_path / "p.json")
    written = json.loads((tmp_path / "p.json").read_text())
    assert written == result["policy"]
    assert set(written["hard"]) == {"x", "y"}
    assert written["artifact"]["max_size"] == "1 MiB"


def test_refuses_to_overwrite_unless_forced(tmp_path):
    t = np.linspace(0.0, 1.0, 11)
    target = tmp_path / "p.yaml"
    target.write_text("keep me")
    with pytest.raises(UsageError, match="already exists"):
        init_policy({"t": t, "x": t}, output=target)
    assert target.read_text() == "keep me"
    init_policy({"t": t, "x": t}, output=target, force=True)
    assert target.read_text().startswith("# Starter policy for in-memory data.")


def test_rejects_bad_arguments_and_empty_sources(tmp_path):
    t = np.linspace(0.0, 1.0, 11)
    with pytest.raises(UsageError, match="format must be"):
        init_policy({"t": t, "x": t}, format="toml")
    with pytest.raises(SourceError, match="no signals found"):
        init_policy({"t": t})
    source = tmp_path / "run.csv"
    source.write_text("t,x\n0,1\n1,2\n")
    with pytest.raises(UsageError, match="must differ from the source"):
        init_policy(source, output=source, force=True)


def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "baslt", *map(str, args)], capture_output=True, text=True, cwd=cwd)


def test_command_line(tmp_path):
    source = tmp_path / "run.csv"
    source.write_text("t,q [Pa],mode\n0,1.5,0\n1,2.5,1\n2,0.5,1\n")
    printed = _cli("policy", "init", source)
    assert printed.returncode == 0 and printed.stdout.startswith("# Starter policy for run.csv.")
    assert not (tmp_path / "policy.yaml").exists()

    written = _cli("policy", "init", source, "-o", tmp_path / "policy.yaml")
    assert written.returncode == 0
    assert written.stdout.strip() == f"Wrote {tmp_path / 'policy.yaml'}: 2 of 2 signals protected"

    again = _cli("policy", "init", source, "-o", tmp_path / "policy.yaml")
    assert again.returncode == 3 and "already exists" in again.stderr
    forced = _cli("policy", "init", source, "-o", tmp_path / "policy.yaml", "--force", "--json")
    payload = json.loads(forced.stdout)
    assert forced.returncode == 0 and payload["status"] == "pass" and "text" not in payload
    assert payload["policy"]["hard"] == {"q": {"global_extrema": {}}, "mode": {"state_transitions": {}}}

    assert _cli("policy").returncode == 3
    assert _cli("policy", "init", tmp_path / "missing.csv").returncode == 3
