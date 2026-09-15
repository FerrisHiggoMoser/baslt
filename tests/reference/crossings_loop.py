"""Naive per-sample reference for the threshold_crossing and violation contracts.

Written step by step from docs/contracts.md with explicit Python loops over samples. It does not import the code
under test.

Two floating-point details the formulas leave open are resolved from the surrounding contract text, not from the
operator:

- a crossing time is clamped into its bracket `[t[i], t[i+1]]`. Step 1 places the crossing "between consecutive
  finite samples i, i+1" and step 4 measures run lengths between crossing times, which a time outside its own
  bracket would make negative; rounding can push the formula one ulp past `t[i+1]`.
- with hysteresis, a sample marks the state low only when `x <= lo` and `x < V`. In exact arithmetic
  `lo = V - H/2 < V` for every `H > 0`, so a low mark is always below `V` and step 3 always finds a falling level
  crossing (step 1 needs `V > x[i+1]`); float64 collapses `V - H/2` onto `V` for a band narrower than one ulp of
  `V`, and without the second condition step 3 would have no crossing to timestamp the flip with.

Because this oracle and the operator resolve those two details the same way, it cannot by itself show that the
reading is right. tests/property/test_prop_crossings_ops.py therefore also cross-checks the operator against the
repo's other independent implementation of the same contract, src/baslt/verify/reference.py.
"""

from __future__ import annotations

import math


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def crossing_time(t0: float, t1: float, x0: float, x1: float, level: float) -> float:
    """t0 + (t1 - t0) * (level - x0) / (x1 - x0), kept inside [t0, t1]."""
    tc = t0 + (t1 - t0) * (level - x0) / (x1 - x0)
    if tc < t0:
        tc = t0
    if tc > t1:
        tc = t1
    return tc


def threshold_crossings(t, x, value, edge="both", hysteresis=0.0, debounce=0.0, interpolate="linear") -> dict:
    """Steps 1-5 of the threshold_crossing contract.

    Returns {"crossings": [...], "pending_at_end": bool, "level_crossings": int, "flips": int, "gap_flips": int};
    every crossing is a dict with t, edge (+1/-1), index_before, index_after, confirm and gap. `gap_flips` counts
    the flips of step 3 whose level crossing spans a gap, which is the contract's "count of gap flips" evidence:
    "flip" is the step-3 term, so debounce and the edge filter do not reduce it.
    """
    t = _floats(t)
    x = _floats(x)
    level = float(value)
    n = len(x)

    # Step 1: level crossings between consecutive finite samples.
    level_crossings = []
    previous = None
    for i in range(n):
        if not math.isfinite(x[i]):
            continue
        if previous is not None:
            a = previous
            if x[a] < level <= x[i]:
                direction = +1
            elif x[a] >= level > x[i]:
                direction = -1
            else:
                direction = 0
            if direction != 0:
                if interpolate == "linear":
                    tc = crossing_time(t[a], t[i], x[a], x[i], level)
                else:
                    tc = t[i]
                level_crossings.append(
                    {"before": a, "after": i, "direction": direction, "t": tc, "gap": i != a + 1}
                )
        previous = i

    # Steps 2 and 3: state machine; each flip takes the last same-direction level crossing with after <= j.
    hi = level + hysteresis / 2
    lo = level - hysteresis / 2
    state = None
    initial_state = None
    flips = []
    for j in range(n):
        if not math.isfinite(x[j]):
            continue  # non-finite samples hold the state
        if state is None:
            state = x[j] >= level
            initial_state = state
            continue
        if hysteresis == 0:
            new_state = x[j] >= level
        elif x[j] >= hi:
            new_state = True
        elif x[j] <= lo and x[j] < level:
            new_state = False
        else:
            new_state = state
        if new_state != state:
            direction = +1 if new_state else -1
            chosen = None
            for crossing in level_crossings:
                if crossing["direction"] == direction and crossing["after"] <= j:
                    chosen = crossing
            assert chosen is not None, "a flip always has a level crossing before it"
            flips.append(
                {
                    "t": chosen["t"],
                    "edge": direction,
                    "index_before": chosen["before"],
                    "index_after": chosen["after"],
                    "confirm": j,
                    "gap": chosen["gap"],
                }
            )
            state = new_state

    # Step 4: debounce.
    flips.sort(key=lambda f: f["t"])  # stable: equal times keep sample order
    accepted_state = initial_state
    transitions = []
    pending_at_end = False
    for k, flip in enumerate(flips):
        if k + 1 < len(flips):
            run_length = flips[k + 1]["t"] - flip["t"]
        else:
            run_length = t[n - 1] - flip["t"]
            if run_length < debounce:
                pending_at_end = True
        if run_length >= debounce:
            flip_state = flip["edge"] > 0
            if flip_state != accepted_state:
                transitions.append(flip)
            accepted_state = flip_state

    # Step 5: edge filter.
    crossings = []
    for flip in transitions:
        if edge == "both" or (edge == "rising" and flip["edge"] > 0) or (edge == "falling" and flip["edge"] < 0):
            crossings.append(flip)
    return {
        "crossings": crossings,
        "pending_at_end": pending_at_end,
        "level_crossings": len(level_crossings),
        "flips": len(flips),
        "gap_flips": sum(1 for flip in flips if flip["gap"]),
    }


def violation_runs(t, x, above=None, below=None, min_duration=0.0) -> list[dict]:
    """Violating runs kept by min_duration.

    Each run is a dict with start, end, index_start, index_end, worst and flags (a set of flag names).
    """
    t = _floats(t)
    x = _floats(x)
    n = len(x)
    if above is not None:
        limit = float(above)

        def outside(value):
            return value > limit

        def more_extreme(a, b):
            return a > b
    else:
        limit = float(below)

        def outside(value):
            return value < limit

        def more_extreme(a, b):
            return a < b

    runs = []
    i = 0
    while i < n:
        if not (math.isfinite(x[i]) and outside(x[i])):
            i += 1
            continue
        first = i
        worst = i
        while i + 1 < n and math.isfinite(x[i + 1]) and outside(x[i + 1]):
            i += 1
            if more_extreme(x[i], x[worst]):
                worst = i
        last = i
        flags = set()
        if first == 0:
            start = t[first]
            flags.add("open_start")
        elif not math.isfinite(x[first - 1]):
            start = t[first]
            flags.add("gap_start")
        else:
            start = crossing_time(t[first - 1], t[first], x[first - 1], x[first], limit)
        if last == n - 1:
            end = t[last]
            flags.add("open_end")
        elif not math.isfinite(x[last + 1]):
            end = t[last]
            flags.add("gap_end")
        else:
            end = crossing_time(t[last], t[last + 1], x[last], x[last + 1], limit)
        if end - start >= min_duration:
            runs.append(
                {"start": start, "end": end, "index_start": first, "index_end": last, "worst": worst, "flags": flags}
            )
        i = last + 1
    return runs
