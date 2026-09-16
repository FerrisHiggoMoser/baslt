"""The soft layer and the budget search: preview samples, weights, caps, accounting and reconstruction errors."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.api import compile, load_run
from baslt.container.reader import read_artifact
from baslt.errors import InfeasibleBudget
from baslt.plan import compile as compile_module
from baslt.plan.compile import DEFAULT_POINTS, MAX_EVALUATIONS, compile_run, minimum_size
from baslt.policy import load_policy
from baslt.reduce.rank import preview_order
from baslt.verify import reference

pytestmark = pytest.mark.minimal

N = 30001
T = np.linspace(0.0, 300.0, N)
RNG = np.random.default_rng(3)
X = np.sin(T / 5.0) + 0.3 * np.sin(T * 1.7) + RNG.normal(0.0, 0.02, N)
Y = np.cos(T / 7.0) + RNG.normal(0.0, 0.02, N)
SOURCE = {"x": (T, X), "y": (T, Y)}


def policy(*, soft=None, max_size=None, hard=None, **extra) -> dict:
    out = {"version": 1, "name": "soft_probe", "hard": hard if hard is not None else {"x": {"global_extrema": {}}}}
    out["soft"] = soft if soft is not None else [{"match": "*", "priority": "none"}]
    if max_size is not None:
        out["artifact"] = {"max_size": max_size}
    out.update(extra)
    return out


def retained(data, name) -> np.ndarray:
    return read_artifact(data).array(name, "idx").astype(np.int64)


def row(result, name) -> dict:
    return next(r for r in result.manifest["budget"]["signals"] if r["name"] == name)


def taken(data, base, name, values) -> int:
    """How many preview samples a signal took: the shortest ranking prefix that explains its extra samples."""
    extra = np.setdiff1d(retained(data, name), base)
    full = preview_order(values, values.shape[0])
    if extra.size == 0:
        return 0
    position = np.empty(values.shape[0], dtype=np.int64)
    position[full] = np.arange(values.shape[0])
    k = int(position[extra].max()) + 1
    assert np.array_equal(np.setdiff1d(full[:k], base), extra), "the extra samples are not a ranking prefix"
    return k


@pytest.fixture(autouse=True)
def _reports_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recorded error report is written next to the run id


@pytest.fixture(scope="module")
def hard_only():
    result = compile(SOURCE, policy())
    return {name: retained(result.bytes, name) for name in ("x", "y")}


# --- without a budget ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("priority", ["high", "medium", "low"])
def test_default_points_follow_the_priority(priority, hard_only):
    result = compile(SOURCE, policy(soft=[{"match": "x", "priority": priority}, {"match": "*", "priority": "none"}]))
    weight = {"high": 4, "medium": 2, "low": 1}[priority]
    assert taken(result.bytes, hard_only["x"], "x", X) == DEFAULT_POINTS[weight]
    assert retained(result.bytes, "y").tolist() == hard_only["y"].tolist()


def test_unmatched_signals_default_to_medium(hard_only):
    result = compile(SOURCE, {"version": 1, "hard": {"x": {"global_extrema": {}}}})
    assert taken(result.bytes, hard_only["y"], "y", Y) == DEFAULT_POINTS[2]


def test_priority_none_keeps_only_the_hard_samples_and_no_soft_role(hard_only):
    result = compile(SOURCE, policy())
    artifact = read_artifact(result.bytes)
    for name in ("x", "y"):
        assert "soft" not in [role["id"] for role in artifact.signal(name)["roles"]]
        assert row(result, name)["soft"] == 0
    assert result.manifest["budget"]["discretionary_bytes"] == 0


def test_max_points_caps_a_signal(hard_only):
    result = compile(SOURCE, policy(soft=[{"match": "x", "priority": "high", "max_points": 100},
                                          {"match": "*", "priority": "none"}]))
    assert taken(result.bytes, hard_only["x"], "x", X) == 100


def test_a_short_signal_is_kept_whole():
    t = np.arange(50.0)
    result = compile({"s": (t, np.sin(t))}, {"version": 1})
    assert retained(result.bytes, "s").tolist() == list(range(50))
    assert row(result, "s")["soft_max_abs_err"] == "0.000000e+00"


# --- with a budget -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [6000, 9000, 20000, 60000, 150000])
def test_the_budget_is_never_exceeded_and_mostly_used(budget):
    result = compile(SOURCE, policy(soft=[{"match": "*", "priority": "medium"}]), max_size=budget)
    assert result.size <= budget
    # Everything fits at 150 kB only if the source is small; otherwise the search gets close to the budget.
    assert result.size >= 0.95 * budget or row(result, "x")["retained"] == N


def test_weights_split_the_preview_points(hard_only):
    soft = [{"match": "x", "priority": "high"}, {"match": "y", "priority": "low"}]
    result = compile(SOURCE, policy(soft=soft), max_size=40000)
    high = taken(result.bytes, hard_only["x"], "x", X)
    low = taken(result.bytes, hard_only["y"], "y", Y)
    assert low > 0
    # One common scale s: floor(4 s) and floor(s) points.
    assert 4 * low <= high <= 4 * low + 3


def test_a_budget_that_holds_everything_keeps_everything():
    result = compile(SOURCE, policy(soft=[{"match": "*", "priority": "low"}]), max_size="4 MiB")
    assert row(result, "x")["retained"] == N and row(result, "y")["retained"] == N
    assert row(result, "x")["soft_max_abs_err"] == "0.000000e+00"


def test_the_hard_samples_survive_every_budget(hard_only):
    for budget in (5000, 12000, 50000):
        result = compile(SOURCE, policy(soft=[{"match": "*", "priority": "high"}]), max_size=budget)
        assert np.isin(hard_only["x"], retained(result.bytes, "x")).all()


def test_accounting_splits_hard_soft_and_bytes():
    result = compile(SOURCE, policy(soft=[{"match": "*", "priority": "high"}]), max_size=30000)
    budget = result.manifest["budget"]
    assert budget["required_bytes"] + budget["discretionary_bytes"] + budget["overhead_bytes"] == result.size
    assert budget["discretionary_bytes"] > 0
    roomy = compile(SOURCE, policy(soft=[{"match": "*", "priority": "high"}]), max_size="1 GiB")
    for entry in budget["signals"]:
        assert entry["hard"] + entry["soft"] == entry["retained"]
        assert entry["soft"] > 0
    # Without soft samples the hard counts are the whole artifact.
    none = compile(SOURCE, policy())
    assert [r["hard"] for r in none.manifest["budget"]["signals"]] == [r["hard"] for r in budget["signals"]]
    assert roomy.manifest["budget"]["discretionary_bytes"] > budget["discretionary_bytes"]


def test_the_search_builds_a_bounded_number_of_trials(monkeypatch):
    builds = []
    original = compile_module._Assembler.build

    def counting(self, *args, **kwargs):
        builds.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(compile_module._Assembler, "build", counting)
    compile(SOURCE, policy(soft=[{"match": "*", "priority": "high"}]), max_size=25000)
    # base + at most MAX_EVALUATIONS search trials + the final build with the errors recorded
    assert 3 <= len(builds) <= MAX_EVALUATIONS + 2


def test_compiles_are_deterministic():
    pol = policy(soft=[{"match": "*", "priority": "high"}])
    assert compile(SOURCE, pol, max_size=20000).bytes == compile(SOURCE, pol, max_size=20000).bytes


# --- infeasible budgets --------------------------------------------------------------------------------------


def bound_for(pol, max_size=None):
    run, bound = load_run(SOURCE, load_policy(pol), max_size=max_size)
    return run, bound


def test_minimum_size_is_the_exact_feasibility_edge():
    pol = policy(soft=[{"match": "*", "priority": "high"}], hard={"x": {"window_extrema": {"interval": "0.5 s"}}})
    run, bound = bound_for(pol)
    minimum = minimum_size(run, bound)
    run, bound = bound_for(pol, max_size=minimum)
    assert len(compile_run(run, bound, digest=None).bytes) == minimum
    run, bound = bound_for(pol, max_size=minimum - 1)
    with pytest.raises(InfeasibleBudget) as caught:
        compile_run(run, bound, digest=None)
    report = caught.value.report
    assert report["minimum_bytes"] == minimum
    assert report["excess_bytes"] == 1
    assert report["digest"]["algorithm"] == "none"
    assert report["required_bytes"] + report["overhead_bytes"] == minimum
    assert "baslt explain" in str(caught.value)


def test_the_report_records_the_digest_it_was_sized_with():
    result = compile(SOURCE, policy(hard={"x": {"window_extrema": {"interval": "0.1 s"}}}),
                     max_size=2000, on_error="record", id="probe")
    assert result.status == "error"
    assert result.error["digest"]["mode"] == "arrays"


# --- sync groups and soft samples ----------------------------------------------------------------------------


def test_soft_samples_propagate_through_sync_groups(hard_only):
    pol = policy(soft=[{"match": "x", "priority": "high"}, {"match": "*", "priority": "none"}],
                 sync_groups={"g": {"members": ["x", "y"]}})
    result = compile(SOURCE, pol, max_size=30000)
    x_idx, y_idx = retained(result.bytes, "x"), retained(result.bytes, "y")
    assert x_idx.tolist() == y_idx.tolist()  # one clock: every x timestamp is kept on y as well
    y_row = row(result, "y")
    assert y_row["soft"] > 0  # y has no soft priority, but x's previews reach it through the group
    assert "soft" not in [r["id"] for r in read_artifact(result.bytes).signal("y")["roles"]]
    assert result.size <= 30000


# --- reconstruction errors -----------------------------------------------------------------------------------


def expected_error(data, name, t, values, hold=False) -> float:
    artifact = read_artifact(data)
    idx = artifact.array(name, "idx").astype(np.int64)
    kept = artifact.array(name, "v").astype(np.float64)
    rebuild = reference.reconstruct_hold if hold else reference.reconstruct_linear
    approx = rebuild(t[idx], kept, t)
    diff = np.abs(np.asarray(values, dtype=np.float64) - approx)
    diff[idx] = 0.0
    diff = diff[np.isfinite(diff)]
    return float(diff.max()) if diff.size else 0.0


def test_errors_match_the_contract_reconstruction():
    t = np.linspace(0.0, 10.0, 4001)
    gappy = np.sin(t * 3)
    gappy[1000:1100] = np.nan
    mode = (np.sin(t * 40) > 0).astype(np.int64) + (t > 5)  # more transitions than preview points
    vector = np.stack([np.sin(t), np.cos(t), t], axis=1)
    source = {"gappy": (t, gappy), "mode": (t, mode), "vector": (t, vector)}
    pol = {"version": 1, "signals": {"decl": {"mode": {"kind": "discrete"}, "vector": {"kind": "vector"}}},
           "soft": [{"match": "*", "priority": "low", "max_points": 60}]}
    result = compile(source, pol)
    for name, values, hold in (("gappy", gappy, False), ("mode", mode, True), ("vector", vector, False)):
        claimed = float(row(result, name)["soft_max_abs_err"])
        assert claimed > 0
        assert claimed == pytest.approx(expected_error(result.bytes, name, t, values, hold), rel=1e-6)


def test_errors_are_fixed_width_strings():
    result = compile(SOURCE, policy(soft=[{"match": "*", "priority": "low"}]), max_size=20000)
    for entry in result.manifest["budget"]["signals"]:
        assert isinstance(entry["soft_max_abs_err"], str) and len(entry["soft_max_abs_err"]) == 12
