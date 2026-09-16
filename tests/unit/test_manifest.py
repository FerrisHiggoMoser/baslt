"""The manifest shape and its size fixed point, and the compile step that produces both."""

from __future__ import annotations

import re
import sys
import types
from itertools import cycle

import numpy as np
import pytest

import baslt.verify as verify_package
from baslt.container.reader import read_artifact
from baslt.container.spec import canonical_json
from baslt.container.zipwriter import zip_size
from baslt.errors import CompileError, InfeasibleBudget, SelfVerifyFailed
from baslt.hashing import hash_arrays, hash_none
from baslt.manifest import (
    EVIDENCE_LIMIT,
    OVERALL_FAIL,
    OVERALL_PASS,
    OVERALL_PASS_WITH_WARNINGS,
    STATUS_FAIL,
    STATUS_NOT_APPLICABLE,
    STATUS_PASS,
    STATUS_WARN,
    RequirementEntry,
    SignalBudget,
    build_manifest,
    cap_evidence,
    overall_status,
)
from baslt.ops import global_extrema, threshold_crossing, violation
from baslt.ops._common import extent_samples, gap_samples
from baslt.plan import compile_run, evaluate_run
from baslt.plan.compile import CHECK_ENTRY_POINTS, run_self_verify
from baslt.policy import bind_policy, load_policy
from baslt.signals import Run, SourceMeta, normalize_signal

NAN = float("nan")
RATIO = re.compile(r"^\d\.\d{6}e[+-]\d{2}$")


# ----- fixtures -------------------------------------------------------------------------------


def wave(n: int = 40) -> np.ndarray:
    return np.sin(np.arange(n, dtype=np.float64) / 3.0) * 10.0 + 50.0


def make_run(source_bytes: int = 4096, **arrays) -> Run:
    n = max(len(np.asarray(v)) for v in arrays.values())
    t = np.arange(n, dtype=np.float64)
    signals = {}
    for name, values in arrays.items():
        unit = "Pa" if name == "q" else None
        sig, _ = normalize_signal(name, t, np.asarray(values), path=f"/{name}", unit=unit)
        signals[name] = sig
    meta = SourceMeta(path="run.npz", format="numpy", size_bytes=source_bytes, issues=["a source note"])
    return Run(signals=signals, meta=meta)


def demo_run() -> Run:
    values = wave()
    values[7] = NAN
    values[8] = NAN
    return make_run(
        q=values,
        mode=np.repeat([0, 1, 2], [10, 15, 15]).astype(np.int32),
        label=np.repeat(["idle", "burn"], 20),
        pos=np.stack([wave(), wave() * 2, wave() * 3], axis=1),
        flat=np.linspace(0.0, 1.0, 40),
        flat2=np.linspace(0.0, 2.0, 40),
    )


DEMO_HARD = {
    "q": {
        "global_extrema": {},
        "threshold_crossing": [{"value": 55.0}],
        "violation": [{"above": 58.0, "min_duration": "0.5 s"}],
    },
    "mode": {"state_transitions": {}},
    "label": {"state_transitions": {}},
    "pos": {"global_extrema": {}},
}


def bind(run: Run, **sections):
    return bind_policy(load_policy({"version": 1, "name": "demo", **sections}), run.infos())


def compile_demo(run: Run | None = None, **sections):
    run = run if run is not None else demo_run()
    bound = bind(run, hard=DEMO_HARD, **sections)
    digest = hash_arrays({name: sig.v for name, sig in run.signals.items()})
    return run, bound, compile_run(run, bound, digest=digest)


def simple_policy():
    return load_policy({"version": 1, "name": "p"})


def rows():
    return [SignalBudget(name="q", retained=12, hard=12, soft=0, bytes=300)]


def entries():
    return [
        RequirementEntry(
            id="hard.q.global_extrema", signal="q", op="global_extrema", severity="info",
            status=STATUS_PASS, evidence={"components": []},
        )
    ]


def sized(members=(("baslt.json", 105), ("index.json", 420), ("policy.json", 260))):
    """A realistic size_of: the manifest is STORED, so its member size is its own length."""
    return lambda length: zip_size([*members, ("manifest.json", length)])


# ----- status roll-up -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("statuses", "on_not_applicable", "expected"),
    [
        ([], "warn", OVERALL_PASS),
        ([STATUS_PASS, STATUS_PASS], "warn", OVERALL_PASS),
        ([STATUS_PASS, STATUS_WARN], "warn", OVERALL_PASS_WITH_WARNINGS),
        ([STATUS_PASS, STATUS_NOT_APPLICABLE], "warn", OVERALL_PASS_WITH_WARNINGS),
        ([STATUS_WARN, STATUS_NOT_APPLICABLE], "warn", OVERALL_PASS_WITH_WARNINGS),
        ([STATUS_PASS, STATUS_FAIL], "warn", OVERALL_FAIL),
        ([STATUS_FAIL, STATUS_NOT_APPLICABLE], "fail", OVERALL_FAIL),
        # not_applicable fails the run only when the policy asks for it; a warning never does.
        ([STATUS_PASS, STATUS_NOT_APPLICABLE], "fail", OVERALL_FAIL),
        ([STATUS_PASS, STATUS_WARN], "fail", OVERALL_PASS_WITH_WARNINGS),
        ([STATUS_PASS], "fail", OVERALL_PASS),
    ],
)
def test_overall_status(statuses, on_not_applicable, expected):
    assert overall_status(statuses, on_not_applicable=on_not_applicable) == expected


def test_evidence_lists_are_capped_but_counts_are_not():
    evidence = {"count": 1000, "runs": [{"i": i} for i in range(1000)], "nested": {"items": list(range(300))}}
    capped = cap_evidence(evidence)

    assert capped["count"] == 1000
    assert len(capped["runs"]) == EVIDENCE_LIMIT
    assert capped["runs"][0] == {"i": 0}
    assert len(capped["nested"]["items"]) == EVIDENCE_LIMIT


# ----- manifest shape -------------------------------------------------------------------------


def test_manifest_shape_matches_the_container_document():
    manifest, size = build_manifest(
        source=SourceMeta(path="run.h5", format="hdf5", size_bytes=8_421_000, issues=[]),
        digest=hash_none(),
        policy=simple_policy(),
        requirements=entries(),
        signals=rows(),
        samples_total=418_234,
        size_of=sized(),
        max_bytes=2_097_152,
        budget_source="policy",
    )

    assert set(manifest) == {
        "status", "baslt", "source", "policy", "artifact", "budget",
        "requirements", "events", "trajectories", "sync_groups",
    }
    assert set(manifest["baslt"]) == {"version", "container"}
    assert set(manifest["source"]) == {
        "path", "format", "size_bytes", "digest", "signals", "samples_total", "issues"
    }
    assert set(manifest["policy"]) == {"name", "sha256"}
    assert set(manifest["artifact"]) == {"size_bytes", "max_bytes", "ratio", "codec", "level"}
    assert set(manifest["budget"]) == {
        "source", "required_bytes", "discretionary_bytes", "overhead_bytes", "signals"
    }
    assert set(manifest["requirements"][0]) == {"id", "signal", "op", "severity", "status", "evidence"}
    assert set(manifest["budget"]["signals"][0]) == {
        "name", "retained", "hard", "soft", "bytes", "soft_max_abs_err"
    }
    assert manifest["events"] == [] and manifest["trajectories"] == [] and manifest["sync_groups"] == []
    assert manifest["status"] == OVERALL_PASS
    assert manifest["source"]["digest"] == {"algorithm": "none", "mode": "none", "covered_bytes": 0, "value": ""}
    assert manifest["artifact"]["max_bytes"] == 2_097_152
    assert manifest["budget"]["source"] == "policy"
    assert size > 0


def test_size_bytes_is_a_twenty_digit_string_and_ratio_is_scientific():
    manifest, size = build_manifest(
        source=SourceMeta(path=None, format="numpy", size_bytes=8_421_000, issues=[]),
        digest=None, policy=simple_policy(), requirements=entries(), signals=rows(),
        samples_total=10, size_of=sized(),
    )

    assert manifest["artifact"]["size_bytes"] == f"{size:020d}"
    assert len(manifest["artifact"]["size_bytes"]) == 20
    assert RATIO.match(manifest["artifact"]["ratio"])
    assert float(manifest["artifact"]["ratio"]) == pytest.approx(8_421_000 / size)
    assert manifest["artifact"]["max_bytes"] is None
    assert manifest["source"]["digest"]["mode"] == "none"


def test_ratio_is_zero_without_a_source_size():
    manifest, _ = build_manifest(
        source=SourceMeta(path=None, format="numpy", size_bytes=None, issues=[]),
        digest=None, policy=simple_policy(), requirements=entries(), signals=rows(),
        samples_total=10, size_of=sized(),
    )
    assert manifest["artifact"]["ratio"] == "0.000000e+00"
    assert manifest["source"]["size_bytes"] is None


def test_size_is_a_fixed_point_of_its_own_manifest_length():
    size_of = sized()
    manifest, size = build_manifest(
        source=SourceMeta(path=None, format="numpy", size_bytes=1_000_000, issues=[]),
        digest=None, policy=simple_policy(), requirements=entries(), signals=rows(),
        samples_total=10, size_of=size_of,
    )

    # The manifest describes exactly the file its own length produces.
    assert size == size_of(len(canonical_json(manifest)))
    budget = manifest["budget"]
    assert budget["required_bytes"] + budget["discretionary_bytes"] + budget["overhead_bytes"] == size
    assert budget["required_bytes"] == 300 and budget["discretionary_bytes"] == 0


def test_size_that_never_settles_is_a_compile_error():
    sizes = cycle([10**5, 10**6])  # the overhead digits change on every iteration
    with pytest.raises(CompileError, match="did not settle"):
        build_manifest(
            source=None, digest=None, policy=simple_policy(), requirements=entries(), signals=rows(),
            samples_total=10, size_of=lambda _length: next(sizes),
        )


def test_status_is_rolled_up_from_the_requirements():
    requirements = entries() + [
        RequirementEntry(id="hard.q.window_extrema", signal="q", op="window_extrema", severity="info",
                         status=STATUS_NOT_APPLICABLE, evidence={})
    ]
    manifest, _ = build_manifest(
        source=None, digest=None, policy=simple_policy(), requirements=requirements, signals=rows(),
        samples_total=10, size_of=sized(),
    )
    assert manifest["status"] == OVERALL_PASS_WITH_WARNINGS

    strict = load_policy({"version": 1, "artifact": {"on_not_applicable": "fail"}})
    manifest, _ = build_manifest(
        source=None, digest=None, policy=strict, requirements=requirements, signals=rows(),
        samples_total=10, size_of=sized(),
    )
    assert manifest["status"] == OVERALL_FAIL


# ----- compile --------------------------------------------------------------------------------


def test_compiled_artifact_reads_back():
    run, bound, outcome = compile_demo()
    artifact = read_artifact(outcome.bytes)

    assert artifact.signal_names() == bound.included
    assert artifact.manifest["artifact"]["size_bytes"] == f"{len(outcome.bytes):020d}"
    assert int(artifact.manifest["artifact"]["size_bytes"]) == outcome.size
    assert artifact.manifest["policy"]["sha256"] == bound.policy.sha256
    assert artifact.policy["sha256"] == bound.policy.sha256
    assert artifact.policy["canonical"] == bound.policy.canonical
    assert artifact.header["codec"] == "deflate"
    assert [entry.name for entry in artifact.entries][:4] == [
        "baslt.json", "index.json", "manifest.json", "policy.json"
    ]

    budget = artifact.manifest["budget"]
    assert budget["required_bytes"] + budget["discretionary_bytes"] + budget["overhead_bytes"] == outcome.size
    assert budget["discretionary_bytes"] == 0
    assert {row["name"] for row in budget["signals"]} == set(bound.included)
    assert all(row["hard"] == row["retained"] and row["soft"] == 0 for row in budget["signals"])


def test_retained_samples_are_bit_identical_to_the_source():
    run, _bound, outcome = compile_demo()
    artifact = read_artifact(outcome.bytes)

    for name, sig in run.signals.items():
        idx = artifact.array(name, "idx").astype(np.int64)
        assert np.all(np.diff(idx) > 0) and (idx < sig.n).all()
        t = artifact.array(name, "t")
        v = artifact.array(name, "v")
        assert t.tobytes() == np.ascontiguousarray(sig.t[idx]).tobytes()
        assert v.tobytes() == np.ascontiguousarray(sig.v[idx].astype(v.dtype)).tobytes()


def test_retained_samples_match_the_operators_own_output():
    run, _bound, outcome = compile_demo()
    sig = run.signals["q"]
    artifact = read_artifact(outcome.bytes)

    # Re-derive what the contracts ask for, straight from the operators and the implicit retention.
    expected = extent_samples(sig.n, 0)
    expected = expected.union(gap_samples(sig.v, 0))
    expected = expected.union(global_extrema.evaluate(sig, {}, {"main": 0}).samples)
    expected = expected.union(threshold_crossing.evaluate(sig, {"value": 55.0}, {"main": 0}).samples)
    expected = expected.union(
        violation.evaluate(sig, {"above": 58.0, "min_duration": 0.5}, {"edge": 0, "worst": 0}).samples
    )

    idx = artifact.array("q", "idx").astype(np.int64)
    assert idx.tolist() == expected.materialize()[0].tolist()


def test_index_describes_signals_roles_and_bound_parameters():
    run, _bound, outcome = compile_demo()
    artifact = read_artifact(outcome.bytes)

    q = artifact.signal("q")
    assert (q["kind"], q["interp"], q["unit"]) == ("continuous", "linear", "Pa")
    assert (q["n_source"], q["components"], q["labels"]) == (40, 1, [])
    assert [array["name"] for array in q["arrays"]] == ["t", "v", "idx", "roles"]
    assert [role["id"] for role in q["roles"]][:2] == ["extent", "gap"]
    crossing = next(r for r in q["requirements"] if r["op"] == "threshold_crossing")
    assert {p["name"]: p["value"] for p in crossing["params"]} == {
        "value": 55.0, "edge": "both", "hysteresis": 0.0,
        "debounce": 0.0, "tolerance": 0.0, "interpolate": "linear",
    }
    assert crossing["bits"] == [
        role["bit"] for role in q["roles"] if role["id"].startswith(crossing["id"])
    ]

    mode = artifact.signal("mode")
    assert (mode["kind"], mode["interp"]) == ("discrete", "hold")
    label = artifact.signal("label")
    assert label["labels"] == ["burn", "idle"]
    assert artifact.signal("pos")["components"] == 3
    assert artifact.array("pos", "v").shape == (artifact.signal("pos")["n"], 3)


def test_signals_with_identical_retained_timestamps_share_a_clock_member():
    run, _bound, outcome = compile_demo()
    artifact = read_artifact(outcome.bytes)

    def clock(name):
        return next(a["member"] for a in artifact.signal(name)["arrays"] if a["name"] == "t")

    # flat and flat2 carry no requirement, so both retain exactly their extent: one shared clock.
    assert clock("flat") == clock("flat2")
    assert clock("q") != clock("flat")
    members = [entry.name for entry in artifact.entries]
    assert members.count(clock("flat")) == 1
    assert sum(name.startswith("t/") for name in members) < len(run.signals)


def test_manifest_records_the_source_and_digest():
    run, _bound, outcome = compile_demo()
    manifest = outcome.manifest

    assert manifest["source"]["path"] == "run.npz"
    assert manifest["source"]["format"] == "numpy"
    assert manifest["source"]["issues"] == ["a source note"]
    assert manifest["source"]["signals"] == len(run.signals)
    assert manifest["source"]["samples_total"] == sum(sig.n for sig in run.signals.values())
    assert manifest["source"]["digest"]["mode"] == "arrays"
    assert manifest["status"] == outcome.status


def test_not_applicable_can_fail_the_whole_run():
    run = make_run(q=np.full(20, NAN))
    hard = {"q": {"global_extrema": {}}}
    outcome = compile_run(run, bind(run, hard=hard), digest=None)
    assert outcome.manifest["requirements"][0]["status"] == STATUS_NOT_APPLICABLE
    assert outcome.manifest["status"] == OVERALL_PASS_WITH_WARNINGS

    strict = bind(run, hard=hard, artifact={"on_not_applicable": "fail"})
    with pytest.raises(SelfVerifyFailed):
        compile_run(run, strict, digest=None)
    assert compile_run(run, strict, digest=None, self_verify=False).manifest["status"] == OVERALL_FAIL


def test_infeasible_budget_reports_a_breakdown():
    run = demo_run()
    bound = bind(run, hard=DEMO_HARD, artifact={"max_size": 512})

    with pytest.raises(InfeasibleBudget) as exc:
        compile_run(run, bound, digest=None)

    assert exc.value.exit_code == 2
    report = exc.value.report
    assert report["kind"] == "infeasible_budget"
    assert report["max_bytes"] == 512
    assert report["budget_source"] == "policy"
    assert report["minimum_bytes"] > 512
    assert report["excess_bytes"] == report["minimum_bytes"] - 512
    assert report["required_bytes"] + report["overhead_bytes"] == report["minimum_bytes"]
    assert {row["id"] for row in report["requirements"]} == {req.id for req in bound.reqs}
    for row in report["requirements"]:
        assert set(row) == {"id", "signal", "op", "samples", "standalone_bytes"}
        assert row["samples"] >= 0 and row["standalone_bytes"] >= 0
    assert any(row["samples"] > 0 for row in report["requirements"])
    assert "artifact.max_size" in str(exc.value)


def test_a_budget_that_fits_is_compiled():
    run = demo_run()
    bound = bind(run, hard=DEMO_HARD, artifact={"max_size": "1 MiB"})
    outcome = compile_run(run, bound, digest=None)

    assert outcome.size <= 1024 * 1024
    assert outcome.manifest["artifact"]["max_bytes"] == 1024 * 1024
    assert outcome.manifest["budget"]["source"] == "policy"


# ----- self-verify ----------------------------------------------------------------------------


def install_checks(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "baslt.verify.checks", module)
    monkeypatch.setattr(verify_package, "checks", module, raising=False)


def checks_module(name="verify_artifact", result=True):
    module = types.ModuleType("baslt.verify.checks")
    setattr(module, name, lambda blob: result)
    return module


def test_self_verify_passes_quietly(monkeypatch):
    install_checks(monkeypatch, checks_module(result=True))
    assert run_self_verify(b"artifact") == []


def test_self_verify_failure_stops_the_compile(monkeypatch):
    install_checks(monkeypatch, checks_module(result=False))
    run = demo_run()
    bound = bind(run, hard=DEMO_HARD)

    with pytest.raises(SelfVerifyFailed) as exc:
        compile_run(run, bound, digest=None)
    assert exc.value.exit_code == 1
    assert "failed its own verification" in str(exc.value)

    # ... and can be turned off.
    assert compile_run(run, bound, digest=None, self_verify=False).size > 0


@pytest.mark.parametrize(
    ("result", "ok"),
    [
        (True, True),
        (False, False),
        ({"ok": True}, True),
        ({"ok": False}, False),
        ({"status": "pass"}, True),
        ({"status": "fail"}, False),
        (types.SimpleNamespace(status="pass_with_warnings"), True),
        (types.SimpleNamespace(status="failed"), False),
        (types.SimpleNamespace(ok=True), True),
        (types.SimpleNamespace(failures=[]), True),
        (types.SimpleNamespace(failures=["structure.size"]), False),
    ],
)
def test_self_verify_reads_the_verifiers_verdict(monkeypatch, result, ok):
    install_checks(monkeypatch, checks_module(result=result))
    if ok:
        assert run_self_verify(b"artifact") == []
    else:
        with pytest.raises(SelfVerifyFailed):
            run_self_verify(b"artifact")


@pytest.mark.parametrize(
    "module",
    [
        types.ModuleType("baslt.verify.checks"),  # no entry point at all
        checks_module(result=object()),  # an unreadable verdict
        None,  # the attribute exists but the module does not
    ],
)
def test_self_verify_fails_closed_when_it_cannot_run(monkeypatch, module):
    install_checks(monkeypatch, module)
    with pytest.raises(CompileError):
        run_self_verify(b"artifact")


def test_self_verify_entry_points_are_tried_in_order(monkeypatch):
    calls = []
    module = types.ModuleType("baslt.verify.checks")
    for name in CHECK_ENTRY_POINTS:
        setattr(module, name, lambda blob, name=name: calls.append(name) or True)
    install_checks(monkeypatch, module)

    assert run_self_verify(b"artifact") == []
    assert calls == [CHECK_ENTRY_POINTS[0]]


def test_compile_self_verifies():
    _run, _bound, outcome = compile_demo()
    notes = [note for note in outcome.notes if note.startswith("self-verify")]
    assert notes == []
