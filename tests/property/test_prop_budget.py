"""The budget search on random runs: never over budget, infeasible exactly below the minimum, always verifiable."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from baslt.api import compile, load_run, verify  # noqa: E402
from baslt.container.reader import read_artifact  # noqa: E402
from baslt.errors import InfeasibleBudget  # noqa: E402
from baslt.hashing import HashInfo  # noqa: E402
from baslt.plan.compile import minimum_size  # noqa: E402
from baslt.policy import load_policy  # noqa: E402

PRIORITIES = ("high", "medium", "low", "none")
PASSING = ("pass", "pass_with_warnings")


@st.composite
def runs(draw):
    count = draw(st.integers(1, 3))
    n = draw(st.integers(2, 400))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    shared = np.cumsum(rng.choice([0.0, 0.01, 0.02, 0.05], size=n))
    source, decl, hard, soft = {}, {}, {}, []
    for k in range(count):
        name = f"s{k}"
        t = shared if draw(st.booleans()) else np.cumsum(rng.choice([0.01, 0.03], size=draw(st.integers(2, 300))))
        shape = draw(st.sampled_from(["smooth", "noise", "steps", "gappy"]))
        if shape == "steps":
            values = rng.integers(0, 4, size=t.shape[0]).astype(np.int64)
            decl[name] = {"kind": "discrete"}
            if draw(st.booleans()):
                hard[name] = {"state_transitions": {}}
        else:
            values = np.sin(t * rng.uniform(0.5, 40.0)) + (rng.normal(0, 0.3, t.shape[0]) if shape == "noise" else 0)
            if shape == "gappy":
                values[rng.random(t.shape[0]) < 0.1] = np.nan
            choice = draw(st.sampled_from(["none", "extrema", "crossing"]))
            if choice == "extrema":
                hard[name] = {"global_extrema": {}}
            elif choice == "crossing":
                hard[name] = {"threshold_crossing": [{"value": 0.25}]}
        source[name] = (t, values)
        soft.append({"match": name, "priority": draw(st.sampled_from(PRIORITIES))})
    policy = {"version": 1, "name": "prop", "signals": {"decl": decl}, "hard": hard, "soft": soft}
    names = list(source)
    if len(names) > 1 and draw(st.booleans()):
        policy["sync_groups"] = {"g": {"members": names}}
    budget = draw(st.integers(1500, 40000))
    return source, policy, budget


def attempt(source, policy, budget):
    try:
        return compile(source, policy, max_size=budget), None
    except InfeasibleBudget as exc:
        return None, exc


@settings(max_examples=60, deadline=None)
@given(runs())
def test_the_budget_holds_and_infeasibility_is_exact(case):
    source, policy, budget = case
    result, infeasible = attempt(source, policy, budget)
    run, bound = load_run(source, load_policy(policy), max_size=budget)
    if result is None:
        report = infeasible.report
        assert report["max_bytes"] == budget
        assert report["minimum_bytes"] > budget
        assert report["minimum_bytes"] == minimum_size(run, bound, digest=HashInfo.from_json(report["digest"]))
    else:
        assert result.size <= budget
        assert len(result.bytes) == result.size
        assert result.status in PASSING


@settings(max_examples=40, deadline=None)
@given(runs())
def test_a_feasible_budget_stays_feasible_when_doubled(case):
    source, policy, budget = case
    result, _ = attempt(source, policy, budget)
    if result is not None:
        larger, infeasible = attempt(source, policy, 2 * budget)
        assert infeasible is None
        assert larger.size <= 2 * budget


@settings(max_examples=40, deadline=None)
@given(runs())
def test_every_accepted_artifact_verifies(case):
    source, policy, budget = case
    result, _ = attempt(source, policy, budget)
    if result is None:
        return
    alone = verify(result.bytes)
    assert alone.status in PASSING, alone.render()
    against = verify(result.bytes, source=source)
    assert against.status in PASSING, against.render()
    ids = {check.id for check in against.checks}
    assert {"budget.accounting", "budget.errors", "source.samples"} <= ids


@settings(max_examples=40, deadline=None)
@given(runs())
def test_the_hard_samples_are_always_kept(case):
    source, policy, budget = case
    result, _ = attempt(source, policy, budget)
    if result is None:
        return
    bare = compile(source, {**policy, "soft": [{"match": "*", "priority": "none"}]})
    kept, base = read_artifact(result.bytes), read_artifact(bare.bytes)
    for name in source:
        wanted = base.array(name, "idx").astype(np.int64)
        got = kept.array(name, "idx").astype(np.int64)
        assert np.isin(wanted, got).all()
    rows = {r["name"]: r for r in result.manifest["budget"]["signals"]}
    for name in source:
        assert rows[name]["hard"] == base.signal(name)["n"]
