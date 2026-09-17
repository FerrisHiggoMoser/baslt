"""Evaluating requirement expressions on runs: values, conditions, functions, aggregates, events and grids."""

from __future__ import annotations

import math

import numpy as np
import pytest

from baslt.ops import threshold_crossing
from baslt.reqs.evaluate import EvalError, Tri, measure, time_weighted
from reference.reqs_runs import context, evaluate

pytestmark = pytest.mark.minimal

T = np.arange(11, dtype=float)


def tri(value):
    assert isinstance(value, Tri)
    return [None if not k else bool(v) for v, k in zip(np.broadcast_to(value.val, T.shape).tolist(),
                                                        np.broadcast_to(value.known, T.shape).tolist())]


def test_arithmetic_and_units():
    ctx = context({"q": (T, T * 1000, "Pa"), "qk": (T, T, "kPa")})
    assert np.array_equal(evaluate(ctx, "q + 1 kPa"), T * 1000 + 1000)
    assert np.array_equal(evaluate(ctx, "q - qk"), np.zeros(11))
    assert np.allclose(evaluate(ctx, "q / 2 kPa"), T / 2)
    assert np.array_equal(evaluate(ctx, "-q * 2"), -T * 2000)
    assert np.array_equal(evaluate(ctx, "q ^ 2"), (T * 1000) ** 2)
    assert np.allclose(evaluate(ctx, "q * 50 %"), T * 500)


def test_three_valued_conditions():
    x = np.array([0, 1, np.nan, 3, 4, np.nan, 6, 7, 8, 9, 10], dtype=float)
    ctx = context({"x": (T, x)})
    assert tri(evaluate(ctx, "x > 2")) == [False, False, None, True, True, None, True, True, True, True, True]
    assert tri(evaluate(ctx, "not (x > 2)"))[:4] == [True, True, None, False]
    # unknown and false is false; unknown or true is true
    assert tri(evaluate(ctx, "x > 2 and t < 1"))[2] is False
    assert tri(evaluate(ctx, "x > 2 or t < 3"))[2] is True
    assert tri(evaluate(ctx, "x > 2 or t > 3"))[2] is None
    assert tri(evaluate(ctx, "(x > 2) == (t > 2)"))[:4] == [True, True, None, True]
    assert tri(evaluate(ctx, "x in (1, 3)"))[:4] == [False, True, None, True]
    assert tri(evaluate(ctx, "x > 2 if t < 5 else x < 8"))[3:8] == [True, True, None, True, True]


def test_states_with_labels():
    mode = np.array([0, 0, 1, 1, 2, 2, 1, 0, 0, 3, 3])
    ctx = context({"gnc__mode": (T, mode), "phase": (T, np.where(T > 4, "BURN", "IDLE"))},
                  {"signals": {"mode": {"path": "gnc/mode", "kind": "discrete",
                                        "labels": {0: "OFF", 1: "ON", 2: "HOLD", 3: "DONE"}}}})
    assert tri(evaluate(ctx, "mode == 'ON'"))[:7] == [False, False, True, True, False, False, True]
    assert tri(evaluate(ctx, "mode != 'OFF' and mode != DONE"))[8:] == [False, False, False]
    assert tri(evaluate(ctx, "mode in ('HOLD', 'DONE')")).count(True) == 4
    assert tri(evaluate(ctx, "mode == 2")).count(True) == 2
    assert tri(evaluate(ctx, "phase == 'BURN'")).count(True) == 6
    with pytest.raises(EvalError, match="is not a state"):
        ctx.code_of(ctx.bind("phase").root, "COAST")


def test_functions():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])
    ctx = context({"x": (T, x, "deg"), "v": (T, np.column_stack([T, T, T]), "m", "vector")},
                  {"curves": {"c": {"points": [[0, 10], [4, 20]], "y_unit": "kPa"},
                              "s": {"points": [[0, 10], [4, 20]], "mode": "step", "outside": "none"},
                              "e": {"points": [[0, 10], [4, 20]], "outside": "error"}}})
    assert np.array_equal(evaluate(ctx, "abs(x)")[:3], [2, 1, 0])
    assert np.array_equal(evaluate(ctx, "sign(x)")[:3], [-1, -1, 0])
    assert np.array_equal(evaluate(ctx, "max(x, 0)")[:2], [0, 0]) and evaluate(ctx, "max(x, 0)")[6] == 0
    assert np.array_equal(evaluate(ctx, "min(x, 1, 3)")[:5], [-2, -1, 0, 1, 1])
    assert np.array_equal(evaluate(ctx, "clip(x, -1, 2)")[:6], [-1, -1, 0, 1, 2, 2])
    assert np.allclose(evaluate(ctx, "hypot(x, x)")[4], 2 * math.sqrt(2))
    assert np.allclose(evaluate(ctx, "norm(v)"), T * math.sqrt(3))
    assert np.allclose(evaluate(ctx, "v.y + v[2]"), 2 * T)
    assert np.allclose(evaluate(ctx, "sin(x)")[5], math.sin(math.radians(3)))
    assert np.array_equal(evaluate(ctx, "where(x > 0, x, 0)")[:6], [0, 0, 0, 1, 2, 3])
    assert np.isnan(evaluate(ctx, "where(x > 0, x, 0)")[6])
    assert np.allclose(evaluate(ctx, "curve(c, t)")[:6], [10, 12.5, 15, 17.5, 20, 20])
    stepped = evaluate(ctx, "curve(s, t)")
    assert stepped[:5].tolist() == [10, 10, 10, 10, 20] and np.isnan(stepped[5])
    with pytest.raises(EvalError, match="outside its range"):
        evaluate(ctx, "curve(e, t)")
    assert ctx.curve_outside["s"] == 6


def test_derivatives_and_moving_means():
    t = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    x = np.array([0.0, 1.0, 4.0, 5.0, 9.0, 16.0])
    ctx = context({"x": (t, x, "m")})
    d = evaluate(ctx, "deriv(x)")
    assert d[0] == 1.0 and d[-1] == 7.0 and np.isnan(d[2])  # repeated timestamps give no derivative
    ctx = context({"y": (T, T * 2.0)})
    assert np.allclose(evaluate(ctx, "movmean(y, 2 s)"), np.concatenate(([0.0, 1.0], T[2:] * 2 - 2)))
    assert np.allclose(evaluate(ctx, "movmean(y, 100 s)")[-1], 10.0)
    gap = context({"y": (T, np.where(T == 5, np.nan, T))})
    smoothed = evaluate(gap, "movmean(y, 2 s)")
    assert np.all(np.isfinite(smoothed[np.arange(11) != 5])) and smoothed[7] == pytest.approx(6.5)


def test_aggregates_and_windows():
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    x = np.array([1.0, 3.0, np.nan, 5.0, 1.0])
    ctx = context({"x": (t, x, "kN"), "s": (t, np.array([0, 1, 1, 0, 0]), None, "discrete")})
    assert evaluate(ctx, "max(x)") == 5.0 and evaluate(ctx, "min(x)") == 1.0
    assert evaluate(ctx, "initial(x)") == 1.0 and evaluate(ctx, "final(x)") == 1.0
    # trapezoids over pairs of finite samples only: 0-1 s gives 2, 4-8 s gives 12 -> 14 over 5 s
    assert evaluate(ctx, "integral(x)") == pytest.approx(14.0)
    assert evaluate(ctx, "mean(x)") == pytest.approx(14 / 5)
    rms = math.sqrt((1 * (1 + 9 + 3) / 3 + 4 * (25 + 1 + 5) / 3) / 5)
    assert evaluate(ctx, "rms(x)") == pytest.approx(rms)
    assert evaluate(ctx, "mean(s)") == pytest.approx(3 / 8)  # held values: 1 from 1 s to 4 s
    assert evaluate(ctx, "duration(s == 1)") == pytest.approx(3.0)
    assert evaluate(ctx, "duration(x > 2)") == pytest.approx(0.0)  # no two consecutive finite samples above 2
    bound = ctx.bind("mean(x)")
    grid = ctx.grid_for(["x"])
    window = ctx.truth(ctx.bind("t >= 4").root, grid)
    assert ctx.scalar(bound.root, grid, window) == pytest.approx(3.0)  # (5 + 1) / 2 over 4-8 s
    nothing = ctx.truth(ctx.bind("t > 100").root, grid)
    assert math.isnan(ctx.scalar(bound.root, grid, nothing))


def test_measure_and_time_weighted_helpers():
    t = np.array([0.0, 1.0, 3.0, 6.0])
    mask = np.array([True, True, False, True])
    assert measure(t, mask, discrete=False) == 1.0 and measure(t, mask, discrete=True) == 3.0
    assert measure(np.array([1.0]), np.array([True]), discrete=False) == 0.0
    area, span = time_weighted(t, np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, bool), discrete=True, what="value")
    assert (area, span) == (1 + 4 + 9, 6.0)


EVENTS = {
    "burn_end": {"signal": "thrust", "falls_below": "5 kN", "debounce": "1 s"},
    "on": {"signal": "mode", "equals": "ON"},
    "high": {"when": "thrust > 7 kN"},
    "low": {"when": "thrust < 2 kN"},
    "cond": {"when": "mode == 'ON' or thrust > 9 kN"},
    "sep": {"param": "sep_time"},
    "sep_ms": {"param": "sep_ms"},
    "never": {"signal": "thrust", "rises_above": "1 MN"},
    "second_on": {"signal": "mode", "equals": "ON", "occurrence": "2"},
}
THRUST = np.array([0.0, 4.0, 8.0, 10.0, 8.0, 6.0, 4.0, 6.0, 3.0, 1.0, 0.0])
MODE = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])


@pytest.fixture
def ectx():
    return context({"prop__thrust": (T, THRUST, "kN"), "gnc__mode": (T, MODE)},
                   {"signals": {"thrust": {"path": "prop/thrust"},
                                "mode": {"path": "gnc/mode", "kind": "discrete", "labels": {0: "OFF", 1: "ON"}}},
                    "events": EVENTS, "params": {"units": {"sep_ms": "ms"}}},
                   params={"sep_time": 6.5, "sep_ms": 2500})


def test_events(ectx):
    expected = threshold_crossing.detect(T, THRUST, 5.0, edge="falling", debounce=1.0).t
    assert np.array_equal(ectx.event("burn_end").times, expected)
    assert ectx.event("on").times.tolist() == [1.0, 5.0]
    assert ectx.event("high").times.tolist() == pytest.approx([1.75])  # the exact crossing of 7 kN
    assert ectx.event("low").times.tolist() == pytest.approx([8.5])  # true from the start is no trigger
    assert ectx.event("cond").times.tolist() == pytest.approx([0.5, 4.5])  # midpoints of an indicator
    assert ectx.event("sep").times.tolist() == [6.5] and ectx.event("sep_ms").times.tolist() == [2.5]
    assert ectx.event("never").count == 0
    assert ectx.event("second_on").selected().tolist() == [5.0]
    with pytest.raises(EvalError, match="unknown event 'zzz'"):
        ectx.event("zzz")


def test_event_helpers(ectx):
    assert tri(evaluate(ectx, "after('on')", "prop/thrust")) == [False] + [True] * 10
    assert tri(evaluate(ectx, "after('on', 2 s)", "prop/thrust"))[:4] == [False, False, False, True]
    assert tri(evaluate(ectx, "after('on#last')", "prop/thrust"))[:6] == [False] * 5 + [True]
    assert tri(evaluate(ectx, "after('on#all')", "prop/thrust")) == [False] + [True] * 10
    assert tri(evaluate(ectx, "before('high')", "prop/thrust"))[:3] == [True, True, False]
    assert tri(evaluate(ectx, "between('on', 'low')", "prop/thrust")) == [False] + [True] * 8 + [False] * 2
    assert ectx.missing == set()
    # the second "on" has no "high" after it: that window runs to the end and "high" counts as missing
    assert tri(evaluate(ectx, "between('on#all', 'high')", "prop/thrust")) == [False, True] + [False] * 3 + [True] * 6
    assert ectx.missing == {"high"}
    ectx.missing.clear()
    assert tri(evaluate(ectx, "during('on#2', 2 s)", "prop/thrust")) == [False] * 5 + [True, True] + [False] * 4
    since = evaluate(ectx, "since('high')", "prop/thrust")
    assert np.isnan(since[1]) and since[2] == pytest.approx(0.25)
    expected = threshold_crossing.detect(T, THRUST, 5.0, edge="falling", debounce=1.0).t
    assert evaluate(ectx, "time('burn_end')") == pytest.approx(expected[0])
    assert evaluate(ectx, "count('on')") == 2.0
    assert evaluate(ectx, "at('high', thrust)") == pytest.approx(7.0)
    assert evaluate(ectx, "at('on#2', mode)") == 1.0
    assert evaluate(ectx, "time('sep') - time('on')") == pytest.approx(5.5)
    assert ectx.missing == set()


def test_missing_events_are_recorded(ectx):
    assert tri(evaluate(ectx, "after('never')", "prop/thrust")) == [False] * 11
    assert tri(evaluate(ectx, "before('never')", "prop/thrust")) == [True] * 11
    assert math.isnan(evaluate(ectx, "time('never')")) and math.isnan(evaluate(ectx, "at('never', thrust)"))
    assert evaluate(ectx, "count('never')") == 0.0
    assert tri(evaluate(ectx, "between('high', 'never')", "prop/thrust")) == [False, False] + [True] * 9
    assert ectx.missing == {"never"}


def test_parameters():
    ctx = context({"x": (T, T)}, params={"payload": "heavy", "mass": "1200", "empty": ""})
    assert tri(evaluate(ctx, "param.payload == 'heavy'"))[0] is True
    assert tri(evaluate(ctx, "param.mass > 1000"))[0] is True
    with pytest.raises(EvalError, match="run parameter 'empty' is not given"):
        evaluate(ctx, "param.empty > 1")
    with pytest.raises(EvalError, match="run parameter 'missing' is not given"):
        evaluate(ctx, "param.missing == 'x'")


def test_grids_and_resampling():
    fast_t = np.linspace(0, 10, 101)
    ctx = context({"slow": (T, T * 10.0), "fast": (fast_t, fast_t), "late": (T + 5, T),
                   "mode": (T, (T > 4.5).astype(int), None, "discrete"), "same": (T.copy(), T * 2)})
    assert evaluate(ctx, "slow - fast", "slow").tolist() == (T * 9).tolist()
    on_fast = evaluate(ctx, "slow", "fast")
    assert on_fast[5] == pytest.approx(5.0) and on_fast.shape == (101,)
    held = evaluate(ctx, "mode", "fast")
    assert held[50] == 1.0 and held[49] == 0.0 and held[45] == 0.0  # previous value, not interpolated
    late = evaluate(ctx, "late", "slow")
    assert np.isnan(late[:5]).all() and late[5] == 0.0  # nothing before the signal starts
    grid = ctx.grid_for(["slow", "same"])
    assert ctx.signals.on("same", grid) is ctx.signals.values("same")  # identical clocks are not resampled
    union = ctx.grid_for(["slow", "late"], mode="union")
    assert union.t.tolist() == list(range(16)) and union.ref is None
    assert ctx.grid_for(["fast"], mode="slow").ref == "slow"
    with pytest.raises(EvalError, match="names no signal"):
        ctx.grid_for(["fast"], mode="1 + 1")


def test_duplicate_timestamps_take_the_later_sample():
    t = np.array([0.0, 1.0, 1.0, 2.0])
    ctx = context({"step": (t, np.array([0.0, 0.0, 10.0, 10.0])), "clock": (np.array([0.0, 1.0, 1.5, 2.0]),
                                                                              np.zeros(4))})
    assert evaluate(ctx, "step", "clock").tolist() == [0.0, 10.0, 10.0, 10.0]


def test_the_cache_stays_within_its_budget():
    ctx = context({"x": (T, T)})
    ctx.signals.cache_limit = 200
    for k in range(20):
        evaluate(ctx, f"x * {k} + 1")
    assert ctx.signals._cache_bytes <= 200
