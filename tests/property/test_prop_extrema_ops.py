"""Property tests: extrema and state-transition operators against the naive reference loops."""

from __future__ import annotations

import struct

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from baslt.ops import global_extrema, state_transitions, window_extrema
from baslt.ops.window_extrema import bucket_ids
from baslt.signals import normalize_signal
from reference.extrema_loop import (
    global_extrema_loop,
    json_values_equal,
    state_transitions_loop,
    window_buckets_loop,
    window_extrema_loop,
)

BITS = {"main": 1}

# Small value alphabets make ties, NaN runs and constant stretches common.
FLOAT_VALUES = st.sampled_from([0.0, -0.0, 1.0, -1.0, 2.5, 1e300, -1e300, np.nan, np.inf, -np.inf])
FINITE_OR_NAN = st.one_of(FLOAT_VALUES, st.floats(-1e6, 1e6, allow_nan=False, width=64))

NAN64_A = np.frombuffer(struct.pack("<Q", 0x7FF8000000000001), dtype="<f8")[0]
NAN64_B = np.frombuffer(struct.pack("<Q", 0xFFF8000000000123), dtype="<f8")[0]
NAN32_A = np.frombuffer(struct.pack("<I", 0x7FC00007), dtype="<f4")[0]
FLOAT64_STATES = [np.float64(x) for x in (0.0, -0.0, 1.0, np.nan, np.inf)] + [NAN64_A, NAN64_B]
FLOAT32_STATES = [np.float32(x) for x in (0.0, -0.0, 1.0, np.nan, -np.inf)] + [NAN32_A]


@st.composite
def clocks(draw, max_n: int = 60):
    n = draw(st.integers(0, max_n))
    base = draw(st.sampled_from([0.0, -3.0, 1.7e9, 12345.678]))
    step = draw(st.sampled_from([0.001, 0.01, 0.1, 0.25, 1.0 / 3.0]))
    ks = sorted(draw(st.lists(st.integers(0, 400), min_size=n, max_size=n)))  # duplicates allowed
    t = base + np.asarray(ks, dtype=np.float64) * step
    return np.maximum.accumulate(t) if n else t


@st.composite
def signals(draw, max_n: int = 60, allow_vector: bool = True):
    t = draw(clocks(max_n))
    n = t.shape[0]
    k = draw(st.sampled_from([1, 1, 2, 3])) if allow_vector else 1
    flat = draw(st.lists(FINITE_OR_NAN, min_size=n * k, max_size=n * k))
    v = np.asarray(flat, dtype=np.float64)
    v = v.reshape(n, k) if k > 1 else v
    sig, _ = normalize_signal("s", t, v, kind="vector" if k > 1 else "continuous")
    return sig


def _retained(res) -> set[int]:
    idx, roles = res.samples.materialize()
    assert (roles == np.uint64(1 << BITS["main"])).all()
    return set(idx.tolist())


@given(signals())
def test_global_extrema_matches_loop(sig):
    res = global_extrema.evaluate(sig, {}, BITS)
    per_comp, retained = global_extrema_loop(sig.v)
    assert _retained(res) == retained
    comps = res.evidence["components"]
    assert len(comps) == len(per_comp)
    for item, (hi, lo) in zip(comps, per_comp, strict=True):
        got_hi = None if item["max"] is None else item["max"]["index"]
        got_lo = None if item["min"] is None else item["min"]["index"]
        assert (got_hi, got_lo) == (hi, lo)
        if hi is not None:
            assert item["max"]["t"] == float(sig.t[hi])
    # Not applicable exactly when there is no finite sample (every component then reports None).
    assert len({hi is None for hi, _ in per_comp}) <= 1
    applicable = any(hi is not None for hi, _ in per_comp)
    assert res.status == ("pass" if applicable else "not_applicable")


@given(signals(), st.sampled_from([0.001, 0.01, 0.1, 0.25, 1.0, 7.5]), st.sampled_from([0.0, -0.05, 0.3, 1.7e9]))
def test_window_extrema_matches_loop(sig, interval, origin):
    if sig.n:
        u = np.abs((sig.t - origin) / interval)
        assume(bool((u < 2**50).all()))
    res = window_extrema.evaluate(sig, {"interval": interval, "origin": origin}, BITS)
    ref = window_extrema_loop(sig.t, sig.v, interval, origin)
    assert _retained(res) == ref["retained"]
    assert res.evidence["buckets"] == ref["buckets"]
    assert res.evidence["buckets_with_finite"] == ref["buckets_with_finite"]
    assert res.status == ("pass" if ref["applicable"] else "not_applicable")


@given(clocks(200), st.sampled_from([0.001, 0.01, 0.1, 1.0 / 3.0, 2.0]), st.sampled_from([0.0, -0.05, 0.1, -1.7e9]))
def test_bucket_ids_match_loop(t, interval, origin):
    if t.size:
        assume(bool((np.abs((t - origin) / interval) < 2**50).all()))
    ids, boundary = bucket_ids(t, interval, origin)
    assert bool((np.diff(ids) >= 0).all())
    got = [[int(ids[i])] + ([int(ids[i]) - 1] if boundary[i] else []) for i in range(t.shape[0])]
    assert got == window_buckets_loop(t, interval, origin)


@given(
    st.integers(0, 80).flatmap(
        lambda n: st.one_of(
            st.lists(st.integers(-2, 2), min_size=n, max_size=n).map(lambda x: np.asarray(x, dtype=np.int32)),
            st.lists(st.booleans(), min_size=n, max_size=n).map(lambda x: np.asarray(x, dtype=bool)),
            st.lists(st.sampled_from(FLOAT64_STATES), min_size=n, max_size=n).map(
                lambda x: np.asarray(x, dtype=np.float64)
            ),
            st.lists(st.sampled_from(FLOAT32_STATES), min_size=n, max_size=n).map(
                lambda x: np.asarray(x, dtype=np.float32)
            ),
        )
    )
)
def test_state_transitions_matches_loop(v):
    sig, _ = normalize_signal("m", np.arange(v.shape[0], dtype=np.float64), v, kind="discrete")
    res = state_transitions.evaluate(sig, {}, BITS)
    ref = state_transitions_loop(v)
    retained = _retained(res)
    assert retained == ref["retained"]
    assert res.evidence["transitions"] == ref["transitions"]
    assert res.evidence["distinct_count"] == ref["distinct_count"]
    assert json_values_equal(res.evidence["distinct_values"], ref["distinct_values"][:256])
    assert res.status == ("pass" if v.shape[0] else "not_applicable")
    if v.shape[0]:
        # The contract's own guarantee, checked directly: the hold reproduces the source bits everywhere.
        idx = np.asarray(sorted(retained))
        held = v[idx[np.searchsorted(idx, np.arange(v.shape[0]), side="right") - 1]]
        assert np.ascontiguousarray(held).tobytes() == np.ascontiguousarray(v).tobytes()
