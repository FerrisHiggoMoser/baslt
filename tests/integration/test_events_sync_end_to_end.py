"""Events and sync groups through compile and verify, including tampered artifacts."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.api import compile, verify
from baslt.container.reader import read_artifact
from reference.artifact_edit import position, rebuild

T = np.linspace(0.0, 10.0, 2001)
THRUST = np.where(T < 6.0, 1000.0, 50.0) + np.random.default_rng(0).normal(0, 1, T.size)
Q = np.sin(T) * 100
ALPHA = np.cos(T * 3)
T2 = np.linspace(0.0, 10.0, 777)
FAST = np.sin(T2 * 5)
MODE = ((T > 3).astype(np.int64) + (T > 7))
SOURCE = {"q": (T, Q), "thrust": (T, THRUST), "alpha": (T, ALPHA), "fast": (T2, FAST), "mode": (T, MODE)}
POLICY = {
    "version": 1,
    "hard": {"q": {"global_extrema": {}}},
    "events": {
        "MECO": {"when": {"signal": "thrust", "falls_below": 100}, "expect": 1,
                 "keep": {"before": "0.5 s", "after": "1 s", "signals": ["q", "fast"]}},
        "Stage": {"when": {"signal": "mode", "equals": 2}, "occurrence": "all",
                  "keep": {"after": "20 s", "signals": ["alpha"]}},
    },
    "sync_groups": {"dyn": {"members": ["q", "alpha", "fast"]}},
}


@pytest.fixture(scope="module")
def artifact():
    result = compile(SOURCE, POLICY)
    assert result.status == "pass_with_warnings"  # the sync group brackets the slower clock
    return result.bytes


def checks_of(data, **kwargs):
    result = verify(data, **kwargs)
    return result, {c.id: c for c in result.checks}


def legend_bit(data, signal, legend_id):
    entry = next(e for e in read_artifact(data).index["signals"] if e["name"] == signal)
    return next(r["bit"] for r in entry["roles"] if r["id"] == legend_id)


def test_well_formed_artifact_verifies_with_and_without_the_source(artifact):
    result, checks = checks_of(artifact)
    assert result.status == "pass_with_warnings", result.render()
    assert checks["events.MECO.trigger"].status == "pass"
    assert checks["events.MECO.windows"].status == "pass"
    assert checks["sync_groups.dyn.alignment"].status == "warn"
    result, checks = checks_of(artifact, source=SOURCE)
    assert result.status == "pass_with_warnings", result.render()
    assert checks["events.MECO.source"].status == "pass"
    assert checks["events.Stage.source"].status == "pass"


def test_missing_window_sample_is_caught(artifact):
    window = read_artifact(artifact).manifest["events"][0]["evidence"]["windows"][0]
    middle = (window["start"] + window["end"]) // 2

    def drop(decoded):
        arrays = decoded[window["signal"]]
        keep = np.ones(arrays["idx"].shape[0], bool)
        keep[position(arrays, middle)] = False
        for name in ("t", "v", "idx", "roles"):
            arrays[name] = arrays[name][keep]

    _, checks = checks_of(rebuild(artifact, arrays=drop))
    assert checks["events.MECO.windows"].status == "fail"


def test_window_that_does_not_cover_its_range_is_caught(artifact):
    def shrink(manifest):
        window = manifest["events"][0]["evidence"]["windows"][0]
        window["start"] += 5

    _, checks = checks_of(rebuild(artifact, manifest=shrink))
    assert checks["events.MECO.windows"].status == "fail"
    assert "inside the window" in checks["events.MECO.windows"].message


def test_moved_trigger_time_is_caught(artifact):
    def move(manifest):
        manifest["events"][0]["evidence"]["triggers"][0]["t"] += 0.01

    _, checks = checks_of(rebuild(artifact, manifest=move))
    assert checks["events.MECO.trigger"].status == "fail"


def test_hidden_trigger_is_caught(artifact):
    def hide(manifest):
        manifest["events"][1]["evidence"]["found"] = 0
        manifest["events"][1]["evidence"]["triggers"] = []

    _, checks = checks_of(rebuild(artifact, manifest=hide))
    assert checks["events.Stage.trigger"].status == "fail"


def test_expect_mismatch_must_be_marked_warn(artifact):
    def lie(index):
        next(e for e in index["events"] if e["name"] == "MECO")["expect"] = 2

    _, checks = checks_of(rebuild(artifact, index=lie))
    assert checks["structure.bind"].status == "fail"
    assert checks["events.MECO.trigger"].status == "fail"


def test_changed_event_value_fails_rebinding(artifact):
    def change(index):
        next(e for e in index["events"] if e["name"] == "MECO")["value"] = 150.0

    _, checks = checks_of(rebuild(artifact, index=change))
    assert checks["structure.bind"].status == "fail"
    assert "events.MECO.value" in checks["structure.bind"].message


def test_event_only_in_one_file_is_caught(artifact):
    def forget(manifest):
        manifest["events"] = [e for e in manifest["events"] if e["name"] != "Stage"]

    _, checks = checks_of(rebuild(artifact, manifest=forget))
    assert checks["events.Stage"].status == "fail"


def test_dropped_sync_bracket_is_caught(artifact):
    bit = legend_bit(artifact, "fast", "sync.dyn")

    def unflag(decoded):
        roles = decoded["fast"]["roles"]
        flagged = np.flatnonzero((roles.astype(np.uint64) & np.uint64(1 << bit)) != 0)
        roles[flagged[len(flagged) // 2]] &= ~roles.dtype.type(1 << bit)

    _, checks = checks_of(rebuild(artifact, arrays=unflag))
    assert checks["sync_groups.dyn.alignment"].status == "fail"


def test_wrong_sync_counts_are_caught(artifact):
    def recount(manifest):
        manifest["sync_groups"][0]["evidence"]["unaligned"] -= 1

    _, checks = checks_of(rebuild(artifact, manifest=recount))
    assert checks["sync_groups.dyn.alignment"].status == "fail"


def test_rocket_run_with_every_supported_feature(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "examples" / "rocket_sim.py"
    spec = importlib.util.spec_from_file_location("rocket_sim", example)
    rs = sys.modules.get("rocket_sim") or importlib.util.module_from_spec(spec)
    if "rocket_sim" not in sys.modules:
        sys.modules["rocket_sim"] = rs
        spec.loader.exec_module(rs)
    data = rs.simulate(duration=110.0, rate=200.0)
    path = rs.write_h5(tmp_path / "run.h5", data)
    policy = {
        "version": 1,
        "hard": {
            "aero/q": {"global_extrema": {}, "local_extrema": {"prominence": "2 kPa", "separation": "1 s"},
                       "threshold_crossing": [{"value": "65 kPa", "hysteresis": "1 kPa", "debounce": "100 ms"}]},
            "aero/alpha": {"violation": [{"above": "7 deg"}]},
            "gnc/mode": {"state_transitions": {}},
        },
        "events": {"MECO": {"when": {"signal": "prop/thrust", "falls_below": "100 N", "debounce": "250 ms"},
                            "expect": 1, "keep": {"before": "2 s", "after": "3 s",
                                                  "signals": ["aero/q", "aero/alpha", "gnc/mode"]}}},
        "sync_groups": {"aero": {"members": ["aero/q", "aero/alpha"]}},
    }
    result = compile(path, policy, output=tmp_path / "run.baslt")
    assert result.status == "pass", result.to_json()
    assert result.manifest["events"][0]["evidence"]["found"] == 1
    checked = verify(tmp_path / "run.baslt", source=path)
    assert checked.status == "pass", checked.render()
