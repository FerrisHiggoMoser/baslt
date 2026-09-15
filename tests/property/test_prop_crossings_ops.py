"""Property tests for threshold_crossing and violation.

Three independent checks:

1. **Oracle.** The vectorized detectors agree exactly with the naive per-sample loops in
   tests/reference/crossings_loop.py.
2. **Cross-implementation.** They agree with the repo's other implementation of the same contract,
   src/baslt/verify/reference.py, which was written from docs/contracts.md independently and is what the verifier
   runs: structure exactly, times within the contract's `4 ulp`. The oracle alone cannot show that a reading of
   the contract is right, only that it is self-consistent, because it resolves the same two float64 details (the
   clamped crossing time, the low mark of a band that rounds onto V) the same way the operator does. Inputs whose
   float64 hysteresis band collapses onto V are excluded with `assume`: there the two implementations disagree by
   design, and the verifier reference raises rather than returning (see the module docstring of
   baslt.ops.threshold_crossing).
3. **Fidelity.** Running steps 1-5 on the reconstruction returns exactly the accepted crossings, and detection on
   the reconstruction returns exactly the kept runs. The reconstruction is built from the mandatory retained set
   -- the operator's own samples plus the implicit `extent` and `gap` retention of docs/contracts.md, which every
   artifact carries for a signal with a hard contract -- and from that set plus arbitrary extra samples, because
   soft picks and other requirements add more. Both the operator and the verifier's own detector are run on both
   sides. docs/verification.md gives these checks basis `artifact`, so they must hold with no appeal to the
   source.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from reference.crossings_loop import threshold_crossings, violation_runs

from baslt.ops._common import gap_samples
from baslt.ops.threshold_crossing import EDGE_NAMES, detect
from baslt.ops.threshold_crossing import evaluate as evaluate_crossing
from baslt.ops.violation import all_runs, detect_runs, flag_names
from baslt.ops.violation import evaluate as evaluate_violation
from baslt.signals import Signal
from baslt.verify import reference as vref

# ------------------------------------------------------------------------------------------------ strategies

finite_values = st.one_of(
    st.integers(-3, 3).map(float),
    st.integers(-6, 6).map(lambda k: k / 2),
    st.floats(-10, 10),
    st.floats(-1e300, 1e300),
)
sample_values = st.one_of(finite_values, finite_values, finite_values, st.sampled_from([math.nan, math.inf, -math.inf]))
time_steps = st.one_of(st.just(0.0), st.integers(0, 8).map(lambda k: k / 4), st.floats(0, 5))
levels = st.one_of(st.integers(-2, 2).map(float), st.integers(-4, 4).map(lambda k: k / 2), st.floats(-5, 5))
hysteresis = st.one_of(st.just(0.0), st.integers(1, 6).map(lambda k: k / 2), st.floats(0, 5))
debounce = st.one_of(st.just(0.0), st.integers(1, 8).map(lambda k: k / 4), st.floats(0, 6))
edges = st.sampled_from(["both", "rising", "falling"])
interpolations = st.sampled_from(["linear", "none"])
durations = st.one_of(st.just(0.0), st.integers(1, 8).map(lambda k: k / 4), st.floats(0, 6))


@st.composite
def series(draw, max_size=40):
    n = draw(st.integers(0, max_size))
    t0 = draw(st.one_of(st.just(0.0), st.floats(-100, 100), st.integers(0, 10**6).map(float)))
    steps = draw(st.lists(time_steps, min_size=n, max_size=n))
    values = draw(st.lists(sample_values, min_size=n, max_size=n))
    t = t0 + np.cumsum(np.asarray(steps, dtype=np.float64))
    return t, np.asarray(values, dtype=np.float64)


@st.composite
def series_with_extra(draw):
    t, x = draw(series())
    mask = draw(st.lists(st.booleans(), min_size=t.shape[0], max_size=t.shape[0]))
    return t, x, np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


# ------------------------------------------------------------------------------------------------ helpers


def _sig(t, x):
    return Signal("s", "s", np.asarray(t, np.float64), np.asarray(x, np.float64), "continuous", None, None, "<f8")


def _extent(n):
    return np.array([0, n - 1] if n else [], dtype=np.int64)


def _gaps(x):
    return gap_samples(np.asarray(x, np.float64), 0).indices_with(1)


def _union(*parts):
    out = np.empty(0, np.int64)
    for p in parts:
        out = np.union1d(out, np.asarray(p, dtype=np.int64))
    return out.astype(np.int64)


def _crossing_params(value, hyst, bounce, edge, interp):
    return {"value": value, "edge": edge, "hysteresis": hyst, "debounce": bounce, "tolerance": 0.0,
            "interpolate": interp}


def _detect(t, x, p):
    return detect(t, x, p["value"], edge=p["edge"], hysteresis=p["hysteresis"], debounce=p["debounce"],
                  interpolate=p["interpolate"])


def _band_is_well_defined(value, hyst):
    """False when float64 collapses `V - H/2` onto `V` (a band narrower than one ulp of `V`)."""
    return hyst == 0 or (value - hyst / 2 < value and value + hyst / 2 > value)


def _contract_retained(t, x, p):
    return evaluate_crossing(_sig(t, x), p, {"main": 0}).samples.indices_with(1)


def _mandatory(t, x, p):
    """What an artifact always retains here: the operator's samples plus implicit `extent` and `gap` retention."""
    return _union(_contract_retained(t, x, p), _extent(len(t)), _gaps(x))


def _times_close(t, got, want, before, after):
    """`|Δt| <= 4·ulp(max(|t[i]|, |t[i+1]|))`, the contract's tolerance with τ = 0."""
    if len(before) == 0:
        return True
    scale = np.maximum(np.abs(t[np.asarray(before, np.int64)]), np.abs(t[np.asarray(after, np.int64)]))
    return bool((np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64)) <= 4 * np.spacing(scale)).all())


# ------------------------------------------------------------------------------------------------ oracle


@given(series(), levels, hysteresis, debounce, edges, interpolations)
def test_detect_matches_naive_oracle(data, value, hyst, bounce, edge, interp):
    t, x = data
    d = detect(t, x, value, edge=edge, hysteresis=hyst, debounce=bounce, interpolate=interp)
    ref = threshold_crossings(t, x, value, edge, hyst, bounce, interp)
    got = list(zip(d.t.tolist(), d.edge.tolist(), d.index_before.tolist(), d.index_after.tolist(),
                   d.confirm.tolist(), d.gap.tolist()))
    want = [(c["t"], c["edge"], c["index_before"], c["index_after"], c["confirm"], c["gap"])
            for c in ref["crossings"]]
    assert got == want
    assert d.pending_at_end == ref["pending_at_end"]
    assert d.n_level_crossings == ref["level_crossings"]
    assert d.n_flips == ref["flips"]
    # The evidence counts gap flips of step 3, before debounce and the edge filter.
    assert d.gap_flips == ref["gap_flips"]
    assert evaluate_crossing(_sig(t, x), _crossing_params(value, hyst, bounce, edge, interp),
                             {"main": 0}).evidence["gap_flips"] == ref["gap_flips"]
    if interp == "none":
        assert d.discretization_error.tolist() == (t[d.index_after] - t[d.index_before]).tolist()


@given(series(), levels, st.booleans(), durations)
def test_detect_runs_matches_naive_oracle(data, limit, is_above, min_duration):
    t, x = data
    kw = {"above": limit} if is_above else {"below": limit}
    r = detect_runs(t, x, min_duration=min_duration, **kw)
    ref = violation_runs(t, x, min_duration=min_duration, **kw)
    got = [(s, e, i, j, w, set(flag_names(f))) for s, e, i, j, w, f in
           zip(r.start.tolist(), r.end.tolist(), r.index_start.tolist(), r.index_end.tolist(), r.worst.tolist(),
               r.flags.tolist())]
    want = [(q["start"], q["end"], q["index_start"], q["index_end"], q["worst"], q["flags"]) for q in ref]
    assert got == want
    assert bool(((r.end - r.start) >= min_duration).all())


# ------------------------------------------------------------------ the verifier's own independent implementation


def _ref_cross(t, x, p):
    return vref.crossings(t, x, value=p["value"], edge=p["edge"], hysteresis=p["hysteresis"],
                          debounce=p["debounce"], interpolate=p["interpolate"])


def _ref_cross_struct(res, idx=None):
    """Accepted crossings as (edge, index_before, index_after, confirm, gap) in source indices."""
    out = []
    for c in res["accepted"]:
        a, b, j = c["index_before"], c["index_after"], c["confirm_index"]
        if idx is not None:
            a, b, j = int(idx[a]), int(idx[b]), int(idx[j])
        out.append((c["edge"], a, b, j, c["gap"]))
    return out


def _detection_struct(d, idx=None):
    out = []
    for e, a, b, j, g in zip(d.edge.tolist(), d.index_before.tolist(), d.index_after.tolist(),
                             d.confirm.tolist(), d.gap.tolist()):
        if idx is not None:
            a, b, j = int(idx[a]), int(idx[b]), int(idx[j])
        out.append((EDGE_NAMES[e], a, b, j, g))
    return out


@given(series(), levels, hysteresis, debounce, edges, interpolations)
def test_detect_matches_the_verifier_reference(data, value, hyst, bounce, edge, interp):
    t, x = data
    assume(_band_is_well_defined(value, hyst))
    p = _crossing_params(value, hyst, bounce, edge, interp)
    d = _detect(t, x, p)
    r = _ref_cross(t, x, p)
    assert _detection_struct(d) == _ref_cross_struct(r)
    assert _times_close(t, d.t, [c["t"] for c in r["accepted"]], d.index_before, d.index_after)
    assert (d.n_flips, d.gap_flips, d.n_level_crossings, d.pending_at_end) == (
        r["flips"], r["gap_flips"], r["raw_level_crossings"], r["pending_at_end"])
    # The evidence the operator publishes is the same count the verifier computes.
    evidence = evaluate_crossing(_sig(t, x), p, {"main": 0}).evidence
    assert evidence["gap_flips"] == r["gap_flips"]
    assert evidence["pending_at_end"] == r["pending_at_end"]
    assert evidence["count"] == len(r["accepted"])


def _ref_runs_struct(runs, idx=None):
    out = []
    for q in runs:
        a, b, w = q["first_index"], q["last_index"], q["worst_index"]
        if idx is not None:
            a, b, w = int(idx[a]), int(idx[b]), int(idx[w])
        flags = {name for name in ("open_start", "open_end", "gap_start", "gap_end") if q[name]}
        out.append((a, b, w, flags))
    return out


def _runs_struct(r, idx=None):
    out = []
    for a, b, w, f in zip(r.index_start.tolist(), r.index_end.tolist(), r.worst.tolist(), r.flags.tolist()):
        if idx is not None:
            a, b, w = int(idx[a]), int(idx[b]), int(idx[w])
        out.append((a, b, w, set(flag_names(f))))
    return out


def _run_times_close(t, r, ref_runs):
    n = t.shape[0]
    if r.count == 0:
        return True
    s, e = r.index_start, r.index_end
    s_scale = np.maximum(np.abs(t[np.maximum(s - 1, 0)]), np.abs(t[s]))
    e_scale = np.maximum(np.abs(t[e]), np.abs(t[np.minimum(e + 1, n - 1)]))
    starts = np.asarray([q["start"] for q in ref_runs], np.float64)
    ends = np.asarray([q["end"] for q in ref_runs], np.float64)
    return bool((np.abs(r.start - starts) <= 4 * np.spacing(s_scale)).all()) and bool(
        (np.abs(r.end - ends) <= 4 * np.spacing(e_scale)).all()
    )


@given(series(), levels, st.booleans(), durations)
def test_detect_runs_matches_the_verifier_reference(data, limit, is_above, min_duration):
    t, x = data
    kw = {"above": limit} if is_above else {"below": limit}
    r = detect_runs(t, x, min_duration=min_duration, **kw)
    ref_runs = vref.violation_runs(t, x, min_duration=min_duration, **kw)
    assert _runs_struct(r) == _ref_runs_struct(ref_runs)
    assert _run_times_close(t, r, ref_runs)


# ------------------------------------------------------------------------------------------------ fidelity


def _crossings_equal(src, t, sub, idx, *, gap):
    """Same accepted crossings: edges, source brackets, times within 4 ulp, and optionally gap flags."""
    if sub.count != src.count:
        return False
    if sub.edge.tolist() != src.edge.tolist():
        return False
    if idx[sub.index_before].tolist() != src.index_before.tolist():
        return False
    if idx[sub.index_after].tolist() != src.index_after.tolist():
        return False
    if not _times_close(t, sub.t, src.t, src.index_before, src.index_after):
        return False
    return not gap or sub.gap.tolist() == src.gap.tolist()


def _assert_fidelity(t, x, p, idx, *, gap=True):
    """The operator and the verifier's own detector both reproduce the accepted crossings on the reconstruction."""
    src = _detect(t, x, p)
    sub = _detect(t[idx], x[idx], p)
    assert _crossings_equal(src, t, sub, idx, gap=gap), (
        f"source {list(zip(src.t.tolist(), src.edge.tolist(), src.index_before.tolist()))} vs "
        f"reconstruction {list(zip(sub.t.tolist(), sub.edge.tolist(), idx[sub.index_before].tolist()))}"
    )
    if _band_is_well_defined(p["value"], p["hysteresis"]):
        ref_src = _ref_cross(t, x, p)
        ref_sub = _ref_cross(t[idx], x[idx], p)
        assert _ref_cross_struct(ref_sub, idx) == _ref_cross_struct(ref_src)
        assert _times_close(t, [c["t"] for c in ref_sub["accepted"]], [c["t"] for c in ref_src["accepted"]],
                            [c["index_before"] for c in ref_src["accepted"]],
                            [c["index_after"] for c in ref_src["accepted"]])


@given(series_with_extra(), levels, hysteresis, debounce, edges, interpolations)
def test_crossing_fidelity_on_the_mandatory_retained_set(data, value, hyst, bounce, edge, interp):
    t, x, extra = data
    p = _crossing_params(value, hyst, bounce, edge, interp)
    src = _detect(t, x, p)
    idx = _mandatory(t, x, p)
    retained = set(idx.tolist())

    # The contract's first clause: the bracket of every accepted crossing, and its confirming sample with H > 0.
    assert set(src.index_before.tolist()) | set(src.index_after.tolist()) <= retained
    if hyst > 0:
        assert set(src.confirm.tolist()) <= retained
    # What the second clause needs on top: the bracket of every flip, including the rejected and filtered ones.
    f = src.flips
    assert set(f.index_before.tolist()) | set(f.index_after.tolist()) <= retained
    if hyst > 0:
        assert set(f.confirm.tolist()) <= retained

    # The contract's second clause, for the mandatory set and for any superset of it.
    _assert_fidelity(t, x, p, idx)
    _assert_fidelity(t, x, p, _union(idx, extra))


def _runs_equal(src, t, sub, idx):
    if sub.count != src.count:
        return False
    if idx[sub.index_start].tolist() != src.index_start.tolist():
        return False
    if idx[sub.index_end].tolist() != src.index_end.tolist():
        return False
    if idx[sub.worst].tolist() != src.worst.tolist() or sub.flags.tolist() != src.flags.tolist():
        return False
    n = t.shape[0]
    s, e = src.index_start, src.index_end
    s_scale = np.maximum(np.abs(t[np.maximum(s - 1, 0)]), np.abs(t[s]))
    e_scale = np.maximum(np.abs(t[e]), np.abs(t[np.minimum(e + 1, n - 1)]))
    return bool((np.abs(sub.start - src.start) <= 4 * np.spacing(s_scale)).all()) and bool(
        (np.abs(sub.end - src.end) <= 4 * np.spacing(e_scale)).all()
    )


def _assert_run_fidelity(t, x, idx, kw, min_duration):
    src = detect_runs(t, x, min_duration=min_duration, **kw)
    sub = detect_runs(t[idx], x[idx], min_duration=min_duration, **kw)
    assert _runs_equal(src, t, sub, idx), (
        f"source {list(zip(src.start.tolist(), src.end.tolist(), src.index_start.tolist()))} vs "
        f"reconstruction {list(zip(sub.start.tolist(), sub.end.tolist(), idx[sub.index_start].tolist()))}"
    )
    ref_src = vref.violation_runs(t, x, min_duration=min_duration, **kw)
    ref_sub = vref.violation_runs(t[idx], x[idx], min_duration=min_duration, **kw)
    assert _ref_runs_struct(ref_sub, idx) == _ref_runs_struct(ref_src)


@given(series_with_extra(), levels, st.booleans(), durations)
def test_violation_fidelity_on_the_mandatory_retained_set(data, limit, is_above, min_duration):
    t, x, extra = data
    kw = {"above": limit} if is_above else {"below": limit}
    params = {"above": None, "below": None, "min_duration": min_duration, **kw}
    res = evaluate_violation(_sig(t, x), params, {"edge": 0, "worst": 1})
    src = detect_runs(t, x, min_duration=min_duration, **kw)

    # `worst` marks exactly the kept runs; `edge` marks the boundary pairs of every candidate run.
    assert set(src.worst.tolist()) == set(res.samples.indices_with(0b10).tolist())
    every = all_runs(t, x, **kw)
    assert set(every.edge_indices(len(t)).tolist()) <= set(res.samples.indices_with(0b01).tolist())

    idx = _union(res.samples.indices_with(0b11), _extent(len(t)), _gaps(x))
    _assert_run_fidelity(t, x, idx, kw, min_duration)
    _assert_run_fidelity(t, x, _union(idx, extra), kw, min_duration)


# --------------------------------------------------------------------------------- former contract-gap cases
#
# Each case below used to break the contract's fidelity clause with the retained set the operators shipped. They
# are kept as regressions: the mandatory retained set now reproduces the source detection on every one of them.

T11 = [float(k) for k in range(11)]

CROSSING_CASES = [
    pytest.param([0, 1, 2, 3, 4, 5], [0, 2, 2, 2, 2, 2], {"debounce": 2.0}, [],
                 id="debounce_needs_last_sample"),
    pytest.param([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 0, 0.75, 2, 2, 2], {"hysteresis": 1.0, "edge": "rising"}, [],
                 id="hysteresis_edge_filter_misses_opposite_confirm"),
    pytest.param(T11, [0, 2, 2, 2, 0, 0, 0, 0.75, 2, 2, 2], {"debounce": 2.0, "edge": "rising"}, [],
                 id="debounce_edge_filter_moves_opposite_crossing"),
    pytest.param(T11, [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0], {"debounce": 2.0}, [],
                 id="pending_at_end_flip_not_retained"),
    pytest.param(T11, [0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2], {"debounce": 2.0}, [5],
                 id="debounce_extra_sample_inside_rejected_excursion"),
    pytest.param(T11, [0, 2, 2, 2, 2, 0, math.nan, 2, 2, 2, 2], {"debounce": 2.0}, [],
                 id="debounce_gap_samples_inside_rejected_excursion"),
]


@pytest.mark.parametrize(("t", "x", "overrides", "extra"), CROSSING_CASES)
def test_crossing_fidelity_on_former_contract_gaps(t, x, overrides, extra):
    t = np.asarray(t, np.float64)
    x = np.asarray(x, np.float64)
    p = _crossing_params(1.0, 0.0, 0.0, "both", "linear")
    p.update(overrides)
    _assert_fidelity(t, x, p, _union(_mandatory(t, x, p), extra))


VIOLATION_CASES = [
    pytest.param(T11, [0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0], 2.0, [2], id="extra_sample_inside_unkept_run"),
    pytest.param([0, 9, 10, 11, 12, 13, 14, 15], [0, 0, 2, math.nan, 0, 5, 5, 0], 2.0, [],
                 id="gap_sample_inside_unkept_run"),
]


@pytest.mark.parametrize(("t", "x", "min_duration", "extra"), VIOLATION_CASES)
def test_violation_fidelity_on_former_contract_gaps(t, x, min_duration, extra):
    t = np.asarray(t, np.float64)
    x = np.asarray(x, np.float64)
    params = {"above": 1.0, "below": None, "min_duration": min_duration}
    res = evaluate_violation(_sig(t, x), params, {"edge": 0, "worst": 1})
    idx = _union(res.samples.indices_with(0b11), _extent(len(t)), _gaps(x), extra)
    _assert_run_fidelity(t, x, idx, {"above": 1.0}, min_duration)
