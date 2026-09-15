"""Unit tests for the window_extrema operator and its bucket formula."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from baslt.ops import window_extrema
from baslt.ops.window_extrema import bucket_delta, bucket_ids
from baslt.signals import normalize_signal
from reference.extrema_loop import window_buckets_loop, window_extrema_loop

BITS = {"main": 5}


def _sig(t, v, kind=None):
    sig, _ = normalize_signal("s", np.asarray(t, dtype=np.float64), np.asarray(v), kind=kind)
    return sig


def _run(t, v, interval, origin=0.0, kind=None):
    sig = _sig(t, v, kind)
    params = {"interval": interval, "origin": origin}
    res = window_extrema.evaluate(sig, params, BITS)
    idx, roles = res.samples.materialize()
    assert (roles == np.uint64(1 << BITS["main"])).all()
    return res, idx.tolist()


def _neighbour_sets(t, interval, origin):
    ids, boundary = bucket_ids(t, interval, origin)
    out = []
    for i in range(ids.shape[0]):
        s = [int(ids[i])]
        if boundary[i]:
            s.append(int(ids[i]) - 1)
        out.append(s)
    return out


# ---------------------------------------------------------------- bucket_ids


def test_bucket_ids_empty():
    ids, boundary = bucket_ids(np.empty(0), 1.0, 0.0)
    assert ids.dtype == np.int64 and ids.size == 0
    assert boundary.dtype == bool and boundary.size == 0


def test_bucket_ids_formula_matches_contract_literally():
    t = np.array([-1.0, -0.25, 0.0, 0.3, 0.999, 1.0, 2.5])
    interval, origin = 0.5, 0.25
    ids, _ = bucket_ids(t, interval, origin)
    delta = 8.0 * math.ulp(max(abs(t[0]), abs(t[-1])) / interval)
    expected = [math.floor((float(x) - origin) / interval + delta) for x in t]
    assert ids.tolist() == expected


def test_one_khz_clock_boundaries():
    k = np.arange(2001)
    t = k * 0.001
    ids, boundary = bucket_ids(t, 0.1, 0.0)
    # Every sample on a 100 ms boundary belongs to the bucket that starts there and to the one before it.
    assert ids.tolist() == (k // 100).tolist()
    assert np.flatnonzero(boundary).tolist() == list(range(0, 2001, 100))
    assert _neighbour_sets(t, 0.1, 0.0) == window_buckets_loop(t, 0.1, 0.0)
    # 0.3 / 0.1 is 2.9999999999999996 in float64; delta still puts it in bucket 3.
    assert (300 * 0.001) / 0.1 < 3.0
    assert ids[300] == 3 and boundary[300]


def test_negative_origin():
    k = np.arange(1001)
    t = k * 0.001
    ids, boundary = bucket_ids(t, 0.1, -0.05)
    assert ids.tolist() == ((k + 50) // 100).tolist()
    assert np.flatnonzero(boundary).tolist() == list(range(50, 1001, 100))
    assert _neighbour_sets(t, 0.1, -0.05) == window_buckets_loop(t, 0.1, -0.05)


def test_negative_times():
    t = np.linspace(-2.0, 0.0, 401)
    ids, _ = bucket_ids(t, 0.25, 0.0)
    assert ids[0] == -8 and ids[-1] == 0
    assert _neighbour_sets(t, 0.25, 0.0) == window_buckets_loop(t, 0.25, 0.0)


def test_large_epoch_times():
    k = np.arange(1001)
    base = 1.7e9
    t = base + k * 0.001
    ids, boundary = bucket_ids(t, 0.1, 0.0)
    assert (ids - int(base / 0.1)).tolist() == (k // 100).tolist()
    assert np.flatnonzero(boundary).tolist() == list(range(0, 1001, 100))
    assert _neighbour_sets(t, 0.1, 0.0) == window_buckets_loop(t, 0.1, 0.0)


def test_epoch_origin_near_timestamps():
    t = 1.7e9 + np.arange(501) * 0.001
    ids, boundary = bucket_ids(t, 0.1, 1.7e9 + 0.05)
    assert _neighbour_sets(t, 0.1, 1.7e9 + 0.05) == window_buckets_loop(t, 0.1, 1.7e9 + 0.05)
    assert ids[0] == -1 and ids[50] == 0 and boundary[50]


@pytest.mark.parametrize("interval", [0.0, -1.0, math.inf, math.nan])
def test_bad_interval(interval):
    with pytest.raises(ValueError):
        bucket_ids(np.array([0.0, 1.0]), interval, 0.0)


def test_bad_origin():
    with pytest.raises(ValueError):
        bucket_ids(np.array([0.0, 1.0]), 1.0, math.nan)


def test_interval_too_small_for_timestamps():
    with pytest.raises(ValueError, match="delta"):
        bucket_ids(np.array([1.7e9, 1.7e9 + 1.0]), 1e-9, 0.0)


@pytest.mark.parametrize("interval", [2e-5, 1e-5])
def test_regression_epoch_interval_with_delta_up_to_quarter_is_accepted(interval):
    # delta is 0.125 at 20 us and 0.25 at 10 us; both keep the ids - 1 neighbour rule exact.
    k = np.arange(2000)
    t = 1.7e9 + k * interval
    assert bucket_delta(t, interval) == (0.125 if interval == 2e-5 else 0.25)
    assert _neighbour_sets(t, interval, 0.0) == window_buckets_loop(t, interval, 0.0)
    v = np.sin(k * 0.37)
    res, idx = _run(t, v, interval)
    ref = window_extrema_loop(t, v, interval, 0.0)
    assert res.status == "pass"
    assert set(idx) == ref["retained"]
    assert res.evidence["buckets"] == ref["buckets"]
    assert res.evidence["buckets_with_finite"] == ref["buckets_with_finite"]


def test_regression_limit_is_about_six_microseconds_at_epoch():
    t = np.array([1.7e9, 1.7e9 + 1e-3])
    assert bucket_delta(t, 6.1e-6) == 0.25
    bucket_ids(t, 6.1e-6, 0.0)
    assert bucket_delta(t, 6.0e-6) == 0.5
    with pytest.raises(ValueError, match=r"delta = .* = 0\.5"):
        bucket_ids(t, 6.0e-6, 0.0)


def test_regression_delta_half_is_not_applicable_with_reason():
    t = 1.7e9 + np.arange(200) * 5e-6
    res, idx = _run(t, np.arange(200.0), 5e-6)
    assert res.status == "not_applicable"
    assert idx == []
    assert res.evidence["buckets"] is None and res.evidence["buckets_with_finite"] is None
    assert len(res.notes) == 1 and "delta" in res.notes[0] and "0.5" in res.notes[0]
    assert "2**52" not in res.notes[0]
    json.dumps(res.evidence)


def test_regression_bucket_numbers_too_large_has_its_own_message():
    # delta is tiny (timestamps near 0), but the origin puts u at 2**53.
    t = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=r"2\*\*52") as info:
        bucket_ids(t, 1.0, -(2.0**53))
    assert "delta" not in str(info.value)
    res, idx = _run(t, [1.0, 2.0], 1.0, origin=-(2.0**53))
    assert res.status == "not_applicable"
    assert idx == []
    assert "2**52" in res.notes[0]


# ---------------------------------------------------------------- evaluate


def test_basic_buckets_and_evidence():
    t = [0.0, 0.2, 0.4, 0.6, 1.2, 1.4, 1.7]
    v = [1.0, 3.0, 2.0, 0.5, 9.0, 9.0, -1.0]
    res, idx = _run(t, v, 1.0)
    # bucket 0: t=0 (also bucket -1), 0.2, 0.4, 0.6 ; bucket 1: 1.2, 1.4, 1.7
    assert res.status == "pass"
    assert res.evidence == {"buckets": 3, "buckets_with_finite": 3, "interval": 1.0, "origin": 0.0}
    assert idx == [0, 1, 3, 4, 6]
    json.dumps(res.evidence)


def test_ties_take_lowest_index_per_bucket():
    t = np.arange(8) * 0.25 + 0.1
    v = [5.0, 5.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    res, idx = _run(t, v, 1.0)
    # bucket 0 holds samples 0..3 (t < 1.0), bucket 1 holds samples 4..7.
    assert idx == [0, 2, 4]
    assert res.evidence["buckets"] == 2


def test_nan_runs_and_all_nan_bucket():
    t = [0.1, 0.5, 1.1, 1.5, 2.1, 2.5]
    v = [np.nan, 4.0, np.nan, np.nan, 7.0, np.inf]
    res, idx = _run(t, v, 1.0)
    assert res.status == "pass"
    assert res.evidence["buckets"] == 3
    assert res.evidence["buckets_with_finite"] == 2
    assert idx == [1, 4]


def test_all_nan_signal_not_applicable():
    res, idx = _run([0.0, 1.0, 2.0], [np.nan, np.nan, np.nan], 1.0)
    assert res.status == "not_applicable"
    assert idx == []
    assert res.evidence["buckets_with_finite"] == 0
    assert res.evidence["buckets"] == 4  # buckets -1, 0, 1, 2 (every sample sits on a boundary)
    assert res.notes == ["no finite sample"]


def test_constant_signal():
    t = np.arange(10) * 0.3
    res, idx = _run(t, np.full(10, 2.0), 1.0)
    assert res.status == "pass"
    ref = window_extrema_loop(t, np.full(10, 2.0), 1.0, 0.0)
    assert set(idx) == ref["retained"]
    assert idx == [0, 4, 7]


def test_single_sample():
    res, idx = _run([0.37], [1.0], 0.1)
    assert res.status == "pass"
    assert idx == [0]
    assert res.evidence["buckets"] == 1


def test_empty_signal():
    res, idx = _run(np.empty(0), np.empty(0), 1.0)
    assert res.status == "not_applicable"
    assert idx == []
    assert res.evidence == {"buckets": 0, "buckets_with_finite": 0, "interval": 1.0, "origin": 0.0}


def test_missing_origin_defaults_to_zero():
    sig = _sig([0.0, 0.5, 1.5], [1.0, 2.0, 3.0])
    res = window_extrema.evaluate(sig, {"interval": 1.0}, BITS)
    assert res.evidence["origin"] == 0.0


def test_boundary_sample_feeds_both_buckets():
    # Sample 2 sits exactly on t=1.0; it is the max of bucket 1 and, via the boundary rule, also of bucket 0.
    t = [0.2, 0.6, 1.0, 1.4]
    v = [0.0, 1.0, 5.0, 3.0]
    res, idx = _run(t, v, 1.0)
    ref = window_extrema_loop(t, v, 1.0, 0.0)
    assert set(idx) == ref["retained"]
    assert ref["per_bucket"][0] == [(2, 0)]
    assert ref["per_bucket"][1] == [(2, 3)]


def test_one_khz_clock_evaluate_matches_reference():
    k = np.arange(1001)
    t = k * 0.001
    v = np.sin(k * 0.0314)
    # Boundary samples dominate and grow, so bucket b's max is the boundary sample it borrows from bucket b + 1.
    v[k % 100 == 0] = 10.0 + k[k % 100 == 0] / 100.0
    res, idx = _run(t, v, 0.1)
    ref = window_extrema_loop(t, v, 0.1, 0.0)
    assert set(idx) == ref["retained"]
    # t=0 and t=1 sit on boundaries, so buckets -1 and 10 exist besides 0..9.
    assert res.evidence["buckets"] == ref["buckets"] == 12
    assert ref["per_bucket"][-1][0] == (0, 0)
    for b in range(0, 10):
        assert ref["per_bucket"][b][0][0] == 100 * (b + 1)
        assert 100 * (b + 1) in idx
    assert ref["per_bucket"][10][0] == (1000, 1000)


def test_vectors_use_fully_finite_samples():
    t = [0.1, 0.4, 0.8, 1.2, 1.6]
    v = np.array([[1.0, 5.0], [2.0, np.nan], [0.0, 6.0], [np.nan, 1.0], [4.0, 0.5]])
    res, idx = _run(t, v, 1.0, kind="vector")
    ref = window_extrema_loop(t, v, 1.0, 0.0)
    # bucket 0: finite samples 0 and 2; bucket 1: finite sample 4 only.
    assert set(idx) == ref["retained"] == {0, 2, 4}
    assert ref["per_bucket"][0] == [(0, 2), (2, 0)]
    assert res.evidence["buckets_with_finite"] == ref["buckets_with_finite"] == 2
    assert res.status == "pass"


def test_vector_all_nan_component():
    t = [0.1, 0.4, 1.2]
    v = np.array([[1.0, np.nan], [2.0, np.nan], [0.0, np.nan]])
    res, idx = _run(t, v, 1.0, kind="vector")
    assert res.status == "not_applicable"
    assert idx == []
    assert res.evidence["buckets"] == 2
    assert res.evidence["buckets_with_finite"] == 0


def test_regression_partial_nan_sample_is_not_a_bucket_max():
    v = np.array([[5.0, np.nan], [1.0, 1.0], [3.0, 0.0]])
    t = [0.1, 0.2, 0.3]
    res, idx = _run(t, v, 1.0, kind="vector")
    ref = window_extrema_loop(t, v, 1.0, 0.0)
    assert ref["per_bucket"][0] == [(2, 1), (1, 2)]
    assert idx == [1, 2]
    assert set(idx) == ref["retained"]
    assert res.evidence["buckets_with_finite"] == 1


def test_regression_bucket_without_fully_finite_sample():
    v = np.array([[1.0, np.nan], [np.nan, 2.0]])
    res, idx = _run([0.0, 1.0], v, 10.0, origin=-5.0, kind="vector")
    assert res.status == "not_applicable"
    assert idx == []
    assert res.evidence["buckets"] == 1
    assert res.evidence["buckets_with_finite"] == 0


def test_regression_partial_nan_bucket_is_not_counted():
    # Bucket 0 only has samples with one NaN component; bucket 1 has finite samples.
    t = [0.2, 0.5, 1.2, 1.5]
    v = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0], [0.0, 9.0]])
    res, idx = _run(t, v, 1.0, kind="vector")
    ref = window_extrema_loop(t, v, 1.0, 0.0)
    assert res.status == "pass"
    assert res.evidence["buckets"] == ref["buckets"] == 2
    assert res.evidence["buckets_with_finite"] == ref["buckets_with_finite"] == 1
    assert idx == [2, 3]
    assert set(idx) == ref["retained"]


def test_large_epoch_evaluate():
    k = np.arange(3001)
    t = 1.7e9 + k * 0.001
    rng = np.random.default_rng(7)
    v = rng.integers(0, 5, size=k.size).astype(np.float64)
    v[rng.random(k.size) < 0.1] = np.nan
    res, idx = _run(t, v, 0.1, origin=-12.3)
    ref = window_extrema_loop(t, v, 0.1, -12.3)
    assert set(idx) == ref["retained"]
    assert res.evidence["buckets"] == ref["buckets"]
    assert res.evidence["buckets_with_finite"] == ref["buckets_with_finite"]


def test_duplicate_timestamps():
    t = [0.0, 0.5, 0.5, 0.5, 1.0, 1.0]
    v = [1.0, 2.0, 2.0, -1.0, 7.0, 7.0]
    res, idx = _run(t, v, 1.0)
    ref = window_extrema_loop(t, v, 1.0, 0.0)
    assert set(idx) == ref["retained"]
    # bucket -1: {0}; bucket 0: {0..3} plus borrowed {4, 5}; bucket 1: {4, 5}
    assert idx == [0, 3, 4]
    assert res.evidence["buckets"] == 3


def test_invalid_interval_param():
    sig = _sig([0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        window_extrema.evaluate(sig, {"interval": 0.0}, BITS)
