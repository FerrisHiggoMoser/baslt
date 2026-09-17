"""The vectorized evaluator agrees with a sample-by-sample reading on random expressions and aggregates."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from baslt.reqs.evaluate import Tri  # noqa: E402
from reference import reqs_loop as loop  # noqa: E402
from reference.reqs_runs import context  # noqa: E402

LABELS = {"IDLE": 0, "BURN": 1, "COAST": 2}
MAPPING = {"signals": {"mode": {"path": "m", "kind": "discrete", "labels": {v: k for k, v in LABELS.items()}}}}

values = st.one_of(st.floats(-100, 100, allow_nan=False).map(lambda v: round(v, 2)), st.just(math.nan),
                   st.sampled_from([0.0, 1.0, -1.0]))
numbers = st.one_of(st.integers(-5, 5).map(float), st.sampled_from([0.5, -2.5, 10.0, 0.0]))


def numeric(depth):
    # `t` carries seconds, so it would mix units with the unitless signals; unit rules are tested elsewhere
    leaves = st.one_of(numbers.map(lambda v: ("num", v)), st.sampled_from(["a", "b", "c"]).map(lambda n: ("sig", n)))
    if depth <= 0:
        return leaves
    sub = numeric(depth - 1)
    return st.one_of(
        leaves,
        st.tuples(st.sampled_from(["add", "sub", "mul", "div"]), sub, sub),
        st.tuples(st.just("neg"), sub),
        st.tuples(st.just("abs"), sub),
        st.tuples(st.sampled_from(["max", "min"]), sub, sub),
        st.tuples(st.just("clip"), sub, sub, sub),
        st.tuples(st.just("where"), condition(depth - 1), sub, sub),
    )


def condition(depth):
    compare = st.tuples(st.just("cmp"), st.sampled_from(["<", "<=", ">", ">=", "==", "!="]),
                        numeric(max(depth - 1, 0)), numeric(max(depth - 1, 0)))
    label = st.tuples(st.just("label"), st.just("mode"), st.sampled_from(sorted(LABELS)))
    if depth <= 0:
        return st.one_of(compare, label)
    sub = condition(depth - 1)
    return st.one_of(compare, label, st.tuples(st.sampled_from(["and", "or"]), sub, sub),
                     st.tuples(st.just("not"), sub))


@st.composite
def runs(draw):
    n = draw(st.integers(1, 25))
    steps = draw(st.lists(st.sampled_from([0.0, 0.5, 1.0, 2.0]), min_size=n, max_size=n))
    t = np.cumsum(np.asarray(steps)) + 1.0
    a = np.array(draw(st.lists(values, min_size=n, max_size=n)))
    b = np.array(draw(st.lists(values, min_size=n, max_size=n)))
    m = np.array(draw(st.lists(st.sampled_from([0, 1, 2]), min_size=n, max_size=n)))
    return t, a, b, m


def env_of(t, a, b, m):
    return {"t": t, "a": a, "b": b, "c": t * 2.0 - 3.0, "mode": m.astype(float), "labels": LABELS}


def signals(t, a, b, m, discrete=False):
    return {"a": (t, a, None, "discrete" if discrete else None), "b": (t, b), "c": (t, t * 2.0 - 3.0),
            "m": (t, m, None, "discrete")}


settings_ = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@settings_
@given(runs(), numeric(3))
def test_numeric_expressions(data, tree):
    t, a, b, m = data
    ctx = context(signals(t, a, b, m), MAPPING)
    bound = ctx.bind(loop.render(tree))
    grid = ctx.grid_for(["a"])
    got = np.broadcast_to(np.asarray(ctx.value(bound.root, grid), dtype=float), t.shape)
    env = env_of(t, a, b, m)
    for i in range(t.shape[0]):
        want = loop.value(tree, env, i)
        assert (math.isnan(want) and math.isnan(got[i])) or got[i] == want or \
            math.isclose(got[i], want, rel_tol=1e-12, abs_tol=1e-12), (loop.render(tree), i, got[i], want)


@settings_
@given(runs(), condition(3))
def test_conditions(data, tree):
    t, a, b, m = data
    ctx = context(signals(t, a, b, m), MAPPING)
    bound = ctx.bind(loop.render(tree))
    grid = ctx.grid_for(["a"])
    got = ctx.truth(bound.root, grid)
    assert isinstance(got, Tri)
    env = env_of(t, a, b, m)
    for i in range(t.shape[0]):
        want = loop.truth(tree, env, i)
        have = None if not got.known[i] else bool(got.val[i])
        assert have == want, (loop.render(tree), i, have, want)


@settings_
@given(runs(), condition(2), st.sampled_from(["max", "min", "initial", "final", "mean", "rms", "integral"]),
       st.booleans())
def test_aggregates(data, window_tree, fn, discrete):
    t, a, b, m = data
    ctx = context(signals(t, a, b, m, discrete), MAPPING)
    grid = ctx.grid_for(["a"])
    window = ctx.truth(ctx.bind(loop.render(window_tree)).root, grid)
    got = ctx.scalar(ctx.bind(f"{fn}(a)").root, grid, window)
    env = env_of(t, a, b, m)
    active = [loop.truth(window_tree, env, i) is True for i in range(t.shape[0])]
    want = loop.aggregate(fn, t, a, active, discrete)
    assert (math.isnan(want) and math.isnan(got)) or math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9), \
        (fn, got, want)
