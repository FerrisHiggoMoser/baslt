"""Detection on any superset of the retained samples equals detection on the source.

This is why nothing added after the hard pass -- event windows, sync propagation, later the soft layer -- can make
the verifier's re-detection disagree with the compiler: crossings, violations and event triggers retain every
candidate, so the extra samples never create or hide one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from baslt.ops import threshold_crossing, violation  # noqa: E402
from baslt.ops._common import extent_samples, gap_samples  # noqa: E402
from baslt.signals import normalize_signal  # noqa: E402
from baslt.verify import reference  # noqa: E402


@st.composite
def series(draw):
    n = draw(st.integers(2, 60))
    values = draw(st.lists(st.sampled_from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan]), min_size=n, max_size=n))
    steps = draw(st.lists(st.sampled_from([0.0, 0.5, 1.0, 2.0]), min_size=n, max_size=n))
    t = np.cumsum(steps)
    extra = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return t, np.array(values), np.array(extra)


def _superset(sig, samples, extra):
    keep = samples.union(extent_samples(sig.n, 30)).union(gap_samples(sig.v, 31))
    idx = keep.materialize()[0]
    return np.union1d(idx, np.flatnonzero(extra))


@settings(max_examples=300, deadline=None)
@given(series(), st.sampled_from([1.5, 2.0, 2.5]), st.sampled_from(["rising", "falling", "both"]),
       st.sampled_from([0.0, 1.0, 2.0]), st.sampled_from([0.0, 1.0, 3.0]))
def test_crossings_survive_any_superset(data, level, edge, hysteresis, debounce):
    t, x, extra = data
    sig, _ = normalize_signal("x", t, x)
    params = {"value": level, "edge": edge, "hysteresis": hysteresis, "debounce": debounce,
              "tolerance": 0.0, "interpolate": "linear"}
    out = threshold_crossing.evaluate(sig, params, {"main": 0})
    idx = _superset(sig, out.samples, extra)
    source = reference.crossings(t, x, level, edge=edge, hysteresis=hysteresis, debounce=debounce)
    kept = reference.crossings(t[idx], x[idx], level, edge=edge, hysteresis=hysteresis, debounce=debounce)
    assert kept["pending_at_end"] == source["pending_at_end"]
    assert len(kept["accepted"]) == len(source["accepted"])
    for a, b in zip(kept["accepted"], source["accepted"]):
        assert idx[a["index_before"]] == b["index_before"] and idx[a["index_after"]] == b["index_after"]
        assert abs(a["t"] - b["t"]) <= 4 * reference.ulp(max(abs(a["t"]), abs(b["t"]), 1.0))


@settings(max_examples=300, deadline=None)
@given(series(), st.sampled_from([1.5, 2.5, 3.5]), st.booleans(), st.sampled_from([0.0, 0.5, 2.0]))
def test_violations_survive_any_superset(data, limit, above, min_duration):
    t, x, extra = data
    sig, _ = normalize_signal("x", t, x)
    bound = {"above": limit} if above else {"below": limit}
    out = violation.evaluate(sig, {**bound, "min_duration": min_duration}, {"edge": 0, "worst": 1})
    idx = _superset(sig, out.samples, extra)
    kwargs = {"above": limit} if above else {"below": limit}
    source = reference.violation_runs(t, x, min_duration=min_duration, **kwargs)
    kept = reference.violation_runs(t[idx], x[idx], min_duration=min_duration, **kwargs)
    assert [(r["start"], r["end"]) for r in kept] == [(r["start"], r["end"]) for r in source]


@settings(max_examples=200, deadline=None)
@given(series(), st.sampled_from(["falls_below", "rises_above"]), st.sampled_from([0.0, 1.0]),
       st.sampled_from(["first", "last", "all"]))
def test_event_triggers_survive_any_superset(data, condition, debounce, occurrence):
    t, x, extra = data
    sig, _ = normalize_signal("x", t, x)
    edge = "falling" if condition == "falls_below" else "rising"
    out = threshold_crossing.evaluate(sig, {"value": 2.5, "edge": edge, "hysteresis": 0.0, "debounce": debounce,
                                            "tolerance": 0.0, "interpolate": "linear"}, {"main": 0})
    idx = _superset(sig, out.samples, extra)
    source = reference.event_triggers(t, x, condition, 2.5, debounce=debounce, occurrence=occurrence)
    kept = reference.event_triggers(t[idx], x[idx], condition, 2.5, debounce=debounce, occurrence=occurrence)
    assert kept["found"] == source["found"]
    assert [idx[k["index_before"]] for k in kept["triggers"]] == [s["index_before"] for s in source["triggers"]]
