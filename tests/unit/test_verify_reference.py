"""Hand-worked examples for the verifier's reference detectors (baslt.verify.reference).

Every expected answer below is written out by hand from the rules in docs/contracts.md, with the arithmetic in a
comment. None of them comes from running compiler code.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pytest

from baslt.verify.reference import (
    bucket_delta,
    crossings,
    event_triggers,
    event_window_clipped,
    event_window_indices,
    finite,
    global_extrema,
    local_peaks,
    reconstruct_hold,
    reconstruct_linear,
    sed_max,
    state_transitions,
    ulp,
    violation_runs,
    window_buckets,
    window_evidence,
    window_extrema,
)

pytestmark = pytest.mark.minimal

NAN = float("nan")
INF = float("inf")


@pytest.fixture(autouse=True)
def _floating_point_strict():
    """No detector may emit a numpy floating-point warning (division by zero, invalid, overflow)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            yield


def arange_t(n: int, step: float = 1.0) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * step


def summary(result: dict) -> list[tuple]:
    return [(c["edge"], c["index_before"], c["confirm_index"], c["gap"]) for c in result["accepted"]]


# ---------------------------------------------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------------------------------------------


def test_ulp_matches_float64_spacing():
    # ulp(1) = 2^-52, ulp(-2) = 2^-51 (sign ignored), ulp(0) = smallest subnormal 2^-1074.
    assert ulp(1.0) == 2.0**-52
    assert ulp(-2.0) == 2.0**-51
    assert ulp(0.0) == 2.0**-1074
    # 1.7e10 lies in [2^33, 2^34), so its ulp is 2^(33 - 52) = 2^-19.
    assert ulp(1.7e10) == 2.0**-19


def test_finite_scalar_and_vector():
    assert finite([1.0, NAN, INF, -INF, 0.0]).tolist() == [True, False, False, False, True]
    # A vector sample is finite only when every component is.
    assert finite([[1.0, 2.0], [NAN, 1.0], [3.0, -INF], [4.0, 5.0]]).tolist() == [True, False, False, True]


# ---------------------------------------------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------------------------------------------


def test_reconstruct_linear_interpolates_between_retained_samples():
    t_ret, x_ret = [0.0, 1.0, 3.0], [0.0, 10.0, 30.0]
    # 0.5: 0 + (10 - 0) * 0.5 / 1 = 5; 2: 10 + (30 - 10) * (2 - 1) / (3 - 1) = 20.
    out = reconstruct_linear(t_ret, x_ret, [0.0, 0.5, 1.0, 2.0, 3.0])
    assert out.tolist() == [0.0, 5.0, 10.0, 20.0, 30.0]


def test_reconstruct_linear_is_bit_identical_at_retained_timestamps():
    t_ret = np.array([0.1, 0.30000000000000004, 0.7])
    x_ret = np.array([0.1 + 0.2, math.pi, -1e-300])
    out = reconstruct_linear(t_ret, x_ret, t_ret)
    assert out.tobytes() == x_ret.tobytes()


def test_reconstruct_linear_nan_across_non_finite_retained_sample():
    t_ret, x_ret = [0.0, 1.0, 2.0, 3.0], [0.0, NAN, 2.0, 4.0]
    # (0, 1) and (1, 2) touch the NaN -> gap; at t = 1 the retained value itself is NaN;
    # 2.5: 2 + (4 - 2) * 0.5 = 3; t = 0 is the retained value 0.
    out = reconstruct_linear(t_ret, x_ret, [0.5, 1.0, 1.5, 2.5, 0.0])
    assert np.isnan(out[:3]).all()
    assert out[3:].tolist() == [3.0, 0.0]


def test_reconstruct_linear_duplicate_timestamps_use_later_sample():
    t_ret, x_ret = [0.0, 1.0, 1.0, 2.0], [0.0, 10.0, 20.0, 40.0]
    # 0.5: between (0, 0) and the first t=1 sample (10): 5. At 1.0: later sample 20.
    # 1.5: between the second t=1 sample (20) and (2, 40): 20 + 20 * 0.5 = 30.
    out = reconstruct_linear(t_ret, x_ret, [0.5, 1.0, 1.5])
    assert out.tolist() == [5.0, 20.0, 30.0]


def test_reconstruct_linear_outside_retained_span_is_nan_and_vectors_work():
    assert np.isnan(reconstruct_linear([0.0, 1.0], [0.0, 1.0], [-1.0, 2.0])).all()
    # Components interpolate independently: at 0.25 -> (0 + 4 * 0.25, 8 - 8 * 0.25) = (1, 6).
    out = reconstruct_linear([0.0, 1.0], [[0.0, 8.0], [4.0, 0.0]], [0.25])
    assert out.tolist() == [[1.0, 6.0]]


def test_reconstruct_hold():
    t_ret, x_ret = [0.0, 2.0, 5.0], [1.0, 3.0, 7.0]
    # previous retained value: [0, 2) -> 1, [2, 5) -> 3, [5, ...) -> 7
    assert reconstruct_hold(t_ret, x_ret, [0.0, 1.0, 2.0, 4.9, 5.0, 6.0]).tolist() == [1, 1, 3, 3, 7, 7]
    assert np.isnan(reconstruct_hold(t_ret, x_ret, [-1.0])).all()
    # Repeated timestamps hold the later sample.
    assert reconstruct_hold([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 1.5]).tolist() == [3.0, 3.0]
    # Integer states keep their dtype.
    held = reconstruct_hold([0.0, 1.0], np.array([4, 9], dtype=np.int32), [0.5, 1.0])
    assert held.dtype == np.int32 and held.tolist() == [4, 9]


def test_reconstruction_shape_follows_the_query_shape():
    tr, xr = [0.0, 1.0], [0.0, 1.0]
    vectors = [[0.0, 8.0], [4.0, 0.0]]
    # the result has shape t_query.shape + x_ret.shape[1:]: a 2-D query keeps its shape, a scalar query is 0-d
    assert reconstruct_linear(tr, xr, np.array([[0.25, 0.5], [0.75, 1.0]])).shape == (2, 2)
    assert reconstruct_linear(tr, xr, 0.5).shape == ()
    assert float(reconstruct_linear(tr, xr, 0.5)) == 0.5
    assert reconstruct_linear(tr, vectors, [0.25, 0.5]).shape == (2, 2)
    assert reconstruct_linear(tr, vectors, 0.25).tolist() == [1.0, 6.0]
    assert reconstruct_linear([], [], 0.5).shape == ()
    assert reconstruct_hold(tr, xr, 0.5).shape == ()
    assert reconstruct_hold([], [], 0.5).shape == ()

    # The interpolation is the trajectories contract's weight-first form, x[a] + w * (x[b] - x[a]). Multiplying
    # before dividing is a different rounding: on this pair the two orders differ by one ulp.
    ta, tb, xa, xb, tq = 0.0, 3.0, 5.0, -2.0, 0.5
    weight_first = xa + ((tq - ta) / (tb - ta)) * (xb - xa)
    assert float(reconstruct_linear([ta, tb], [xa, xb], tq)) == weight_first
    assert weight_first != xa + (xb - xa) * (tq - ta) / (tb - ta)


# ---------------------------------------------------------------------------------------------------------------
# global_extrema
# ---------------------------------------------------------------------------------------------------------------


def test_global_extrema_ties_and_non_finite():
    # finite values: 3@0, 7@1, 7@3, -2@4, -2@5 (inf@6 and nan@2 are not finite)
    # max 7: lowest index 1; min -2: lowest index 4
    assert global_extrema([3.0, 7.0, NAN, 7.0, -2.0, -2.0, INF]) == {"max_index": 1, "min_index": 4}


def test_global_extrema_not_applicable():
    assert global_extrema([NAN, INF]) == {"max_index": None, "min_index": None}
    assert global_extrema([]) == {"max_index": None, "min_index": None}


def test_global_extrema_per_component_uses_whole_row_finiteness():
    # Row 2 is not a finite sample: component 1 is NaN. The Conventions rule makes the whole row ineligible, so
    # component 0's maximum is 3.0 @3, not 9.0 @2; component 1 is a constant 0 -> max and min both @0.
    P = np.array([[1.0, 0.0], [2.0, 0.0], [9.0, NAN], [3.0, 0.0]])
    assert finite(P).tolist() == [True, True, False, True]
    assert global_extrema(P) == [{"max_index": 3, "min_index": 0}, {"max_index": 0, "min_index": 0}]
    # slicing a component and calling the 1-D form would instead use per-column finiteness, and pick row 2
    assert global_extrema(P[:, 0]) == {"max_index": 2, "min_index": 0}
    # no finite sample in any component -> not applicable, per component
    assert global_extrema(np.array([[1.0, NAN], [2.0, INF]])) == [
        {"max_index": None, "min_index": None},
        {"max_index": None, "min_index": None},
    ]


# ---------------------------------------------------------------------------------------------------------------
# window_extrema
# ---------------------------------------------------------------------------------------------------------------


def test_bucket_delta_by_hand():
    # 1 kHz clock over one second: max(|0|, |0.999|) / 0.1 = 9.99 in [8, 16) -> ulp = 2^(3 - 52); δ = 8 * 2^-49
    assert bucket_delta(np.arange(1000) / 1000, 0.1) == 8 * 2.0**-49
    # epoch scale: 1.7000000009990e10 in [2^33, 2^34) -> ulp = 2^-19; δ = 8 * 2^-19 = 2^-16
    assert bucket_delta(1.7e9 + np.arange(1000) / 1000, 0.1) == 2.0**-16
    # an empty signal has no t[0] and no bucket to widen; the two callers already answer it without raising
    assert bucket_delta([], 0.1) == 0.0
    assert window_buckets([], 0.1)[0].tolist() == []
    assert window_extrema([], [], 0.1) == {}


def test_window_buckets_1khz_clock_boundaries():
    t = np.arange(1000) / 1000
    k = np.arange(1000)
    # t = 0.3 is 0.299999999999999988898; 0.3 / 0.1 = 2.9999999999999996 so plain floor would give bucket 2.
    # With δ = 1.42e-14: 2.9999999999999996 + δ = 3.0000000000000138 -> bucket 3, as intended.
    assert math.floor(t[300] / 0.1) == 2
    primary, near = window_buckets(t, 0.1, 0.0)
    assert primary.tolist() == (k // 100).tolist()
    # Samples at exact multiples of 0.1 s sit on a boundary (|u - round(u)| <= 4.4e-16 <= δ) and also belong to the
    # bucket before; every other sample is at least 0.01 away from a boundary.
    assert np.flatnonzero(near).tolist() == list(range(0, 1000, 100))


def test_window_buckets_epoch_scale_times():
    k = np.arange(2000)
    t = 1.7e9 + k / 1000  # ulp(1.7e9) = 2^-22 ≈ 2.4e-7, so the grid is not exact
    primary, near = window_buckets(t, 0.1, 1.7e9)
    # (t - o) / 0.1 is within ~1e-6 of k / 100, and δ = 2^-16 ≈ 1.5e-5 absorbs that error.
    assert primary.tolist() == (k // 100).tolist()
    assert np.flatnonzero(near).tolist() == list(range(0, 2000, 100))
    # Without δ, some boundary samples fall into the previous bucket.
    assert (np.floor((t - 1.7e9) / 0.1).astype(np.int64) != k // 100).any()


def test_window_extrema_with_boundary_neighbours_and_ties():
    t = np.arange(300) / 1000  # buckets 0, 1, 2 of 0.1 s; boundary samples 0, 100, 200
    x = np.zeros(300)
    x[50] = 3.0
    x[100] = 5.0
    x[150] = 4.0
    x[200] = -1.0
    x[250] = -1.0
    got = window_extrema(t, x, 0.1, 0.0)
    assert got == {
        # bucket -1 is not a bucket of this signal: m(i) gives each sample exactly one bucket and none of them
        # is -1. Sample 0 sits on its boundary and would be borrowed into it, which does not make it exist.
        # bucket 0 = samples 0..99 plus boundary sample 100: max 5 @100; min 0, lowest index 0
        0: (100, 0),
        # bucket 1 = samples 100..199 plus boundary sample 200: max 5 @100; min -1 @200
        1: (100, 200),
        # bucket 2 = samples 200..299: max 0, lowest index 201; min -1 @200 and @250 -> 200
        2: (201, 200),
    }
    assert window_evidence(t, x, 0.1, 0.0) == {"buckets": 3, "buckets_with_finite": 3}


def test_window_extrema_skips_non_finite_samples():
    t = np.arange(300) / 1000
    x = np.arange(300, dtype=np.float64)
    x[100:200] = NAN
    got = window_extrema(t, x, 0.1, 0.0)
    # bucket 0: 0..99 finite (boundary sample 100 is NaN, not a member): max 99, min 0
    # bucket 1: all its own samples are NaN, but boundary sample 200 is borrowed into it and is finite, so the
    # bucket exists and does contain a finite sample -> (200, 200)
    # bucket 2: 200..299: max 299, min 200
    assert got == {0: (99, 0), 1: (200, 200), 2: (299, 200)}
    assert window_evidence(t, x, 0.1, 0.0) == {"buckets": 3, "buckets_with_finite": 3}
    assert window_extrema(t, np.full(300, NAN), 0.1, 0.0) == {}
    assert window_evidence(t, np.full(300, NAN), 0.1, 0.0) == {"buckets": 3, "buckets_with_finite": 0}


def test_window_extrema_has_no_bucket_below_the_signals_first_bucket():
    # Sample 0 lies exactly on the origin boundary, so it is borrowed into bucket -1 as well. Bucket -1 is still
    # not reported: no sample has it as its own bucket, so it is not a bucket of this signal.
    assert window_extrema([0.0, 0.05], [1.0, 2.0], 0.1) == {0: (1, 0)}
    assert window_evidence([0.0, 0.05], [1.0, 2.0], 0.1) == {"buckets": 1, "buckets_with_finite": 1}
    # an interior boundary sample is still borrowed backwards, into a bucket that does exist
    assert window_extrema([0.05, 0.1], [1.0, 2.0], 0.1) == {0: (1, 0), 1: (1, 1)}


def test_window_extrema_per_component_uses_whole_row_finiteness():
    t = np.array([0.0, 0.04, 0.08, 0.12])  # buckets 0, 0, 0, 1
    P = np.array([[1.0, 5.0], [9.0, NAN], [2.0, 4.0], [3.0, 6.0]])
    # Sample 1 is not a finite sample (component 1 is NaN), so it is eligible in no component -- 9.0 is never a
    # bucket maximum. bucket 0 = samples 0, 2: component 0 max 2 @2 / min 1 @0; component 1 max 5 @0 / min 4 @2.
    # bucket 1 = sample 3 alone in both components.
    assert window_extrema(t, P, 0.1) == [{0: (2, 0), 1: (3, 3)}, {0: (0, 2), 1: (3, 3)}]
    assert window_evidence(t, P, 0.1) == {"buckets": 2, "buckets_with_finite": 2}
    # slicing a component and calling the 1-D form uses per-column finiteness instead, and reports 9.0 @1
    assert window_extrema(t, P[:, 0], 0.1) == {0: (1, 0), 1: (3, 3)}


# ---------------------------------------------------------------------------------------------------------------
# local_extrema
# ---------------------------------------------------------------------------------------------------------------


def peaks_summary(peaks: list[dict]) -> list[tuple]:
    return [(p["index"], p["prominence"], p["left_base"], p["right_base"], p["kind"]) for p in peaks]


def test_local_peaks_basic_prominence():
    x = [0.0, 2.0, 1.0, 3.0, 1.0, 0.0]
    t = arange_t(6)
    # peak @1 (2): left [0..1] -> base 0 (0); right stops at @3 (3 > 2): [1..2] -> base 2 (1); prom 2 - max(0, 1) = 1
    # peak @3 (3): left to start: min 0 @0; right to end: min 0 @5; prom 3 - 0 = 3
    assert peaks_summary(local_peaks(t, x)) == [(1, 1.0, 0, 2, "max"), (3, 3.0, 0, 5, "max")]
    # prominence >= p is inclusive: p = 1 keeps both, p = 1.5 keeps only @3
    assert [p["index"] for p in local_peaks(t, x, prominence=1.0)] == [1, 3]
    assert [p["index"] for p in local_peaks(t, x, prominence=1.5)] == [3]


def test_local_peaks_plateau_middle_rounded_down_and_edge_runs():
    # run [1..4] of 5 between 0 and 1 -> middle (1 + 4) // 2 = 2
    # left: [0..2] min 0 @0; right: [2..6] min 1 @5; prom 5 - max(0, 1) = 4
    # the last run (2 @6) is higher than its neighbour but is never a peak
    x = [0.0, 5.0, 5.0, 5.0, 5.0, 1.0, 2.0]
    assert peaks_summary(local_peaks(arange_t(7), x)) == [(2, 4.0, 0, 5, "max")]
    # a two-sample plateau [1, 2] -> (1 + 2) // 2 = 1
    assert [p["index"] for p in local_peaks(arange_t(4), [0.0, 4.0, 4.0, 0.0])] == [1]
    # the first run is never a peak either
    assert local_peaks(arange_t(3), [5.0, 1.0, 0.0]) == []
    assert local_peaks(arange_t(3), [1.0, 1.0, 1.0]) == []
    assert local_peaks(arange_t(2), [0.0, 1.0]) == []


def test_local_peaks_base_ties_take_index_nearest_the_peak():
    x = [0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0]
    # peak @3: left [0..3] minima 0 @0 and @2 -> nearest 2; right [3..6] minima @4 and @6 -> nearest 4; prom 3
    # peak @1: left base 0; right stops at @3: base 2; prom 1.  peak @5: left stops at @3: base 4; right base 6
    assert peaks_summary(local_peaks(arange_t(7), x)) == [
        (1, 1.0, 0, 2, "max"),
        (3, 3.0, 2, 4, "max"),
        (5, 1.0, 4, 6, "max"),
    ]


def test_local_peaks_non_finite_samples_are_walls():
    x = [0.0, 3.0, 1.0, NAN, 2.0, 5.0, 0.0]
    # peak @1: right scan stops at the wall @3: base @2 (1); prom 3 - max(0, 1) = 2
    # sample @2 and @4 touch the wall (higher neighbour) -> not peaks
    # peak @5: left scan stops at the wall: base @4 (2); right base @6 (0); prom 5 - max(2, 0) = 3
    # (without the wall the left interval would reach 0 @0 and give prominence 5)
    assert peaks_summary(local_peaks(arange_t(7), x)) == [(1, 2.0, 0, 2, "max"), (5, 3.0, 4, 6, "max")]
    assert local_peaks(arange_t(4), [0.0, 4.0, INF, 0.0]) == []


def test_local_peaks_separation_by_value_then_index():
    x = [0.0, 5.0, 0.0, 5.0, 0.0, 4.0, 0.0, 6.0, 0.0]
    t = arange_t(9)
    # peaks @1 (5), @3 (5), @5 (4), @7 (6); visit 7, 1, 3, 5
    # keep 7; 1: |1 - 7| = 6 >= 2.5 keep; 3: |3 - 1| = 2 < 2.5 drop; 5: |5 - 7| = 2 < 2.5 drop
    assert [p["index"] for p in local_peaks(t, x, separation=2.5)] == [1, 7]
    # |Δt| < s is strict: with s = 2 nothing is within 2
    assert [p["index"] for p in local_peaks(t, x, separation=2.0)] == [1, 3, 5, 7]
    # equal values: the lower index is visited first and wins
    assert [p["index"] for p in local_peaks(arange_t(5), [0.0, 5.0, 0.0, 5.0, 0.0], separation=3.0)] == [1]
    # separation compares only against kept peaks: s = 2.5 on [0, 5, 0, 4, 0, 3, 0] visits @1 (5), @3 (4), @5 (3);
    # keep @1; @3 is 2 s from @1 -> drop; @5 is 2 s from the dropped @3 but 4 s from kept @1 -> keep -> [1, 5]
    assert [p["index"] for p in local_peaks(arange_t(7), [0.0, 5.0, 0.0, 4.0, 0.0, 3.0, 0.0], separation=2.5)] == [
        1,
        5,
    ]


def test_local_peaks_separation_uses_time_not_samples():
    # peaks @1 and @3 are two samples apart but t = 0, 0.1, 0.2, 5, 5.1 puts them 4.9 s apart
    t = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    x = [0.0, 2.0, 0.0, 3.0, 0.0]
    assert [p["index"] for p in local_peaks(t, x, separation=4.0)] == [1, 3]
    assert [p["index"] for p in local_peaks(t, x, separation=5.0)] == [3]


def test_local_peaks_minima_and_both():
    x = [5.0, 1.0, 4.0, 0.0, 3.0]
    # -x = [-5, -1, -4, 0, -3]
    # min @1: left base @0 (-5); right stops at @3 (0 > -1): base @2 (-4); prom -1 - max(-5, -4) = 3
    # min @3: left to start: min -5 @0; right base @4 (-3); prom 0 - max(-5, -3) = 3
    # max @2 (4): left stops at @0 (5 > 4): base @1 (1); right to end: min 0 @3; prom 4 - max(1, 0) = 3
    assert peaks_summary(local_peaks(arange_t(5), x, kind="min")) == [(1, 3.0, 0, 2, "min"), (3, 3.0, 0, 4, "min")]
    assert peaks_summary(local_peaks(arange_t(5), x, kind="both")) == [
        (1, 3.0, 0, 2, "min"),
        (2, 3.0, 1, 3, "max"),
        (3, 3.0, 0, 4, "min"),
    ]


def test_local_peaks_long_scans_cross_chunk_boundaries():
    # A single high peak in the middle of a long ramp-and-noise signal: the scan has to walk thousands of samples.
    n = 10_001
    x = np.zeros(n)
    x[1::2] = 1.0  # many small peaks of prominence 1
    x[5000] = 10.0
    x[123] = -2.0  # left base of the big peak (the only -2 to its left)
    x[9876] = -3.0  # right base
    peaks = {p["index"]: p for p in local_peaks(arange_t(n), x, prominence=2.0)}
    # peak @5000: left base @123 (-2), right base @9876 (-3); prom 10 - max(-2, -3) = 12
    assert list(peaks) == [5000]
    assert (peaks[5000]["left_base"], peaks[5000]["right_base"], peaks[5000]["prominence"]) == (123, 9876, 12.0)


def test_local_peaks_agree_with_scipy_on_random_plateaus():
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(7)
    x = np.round(rng.normal(size=3000), 1)  # rounding creates plateaus and tied bases
    ours = local_peaks(arange_t(x.size), x)
    idx, _ = signal.find_peaks(x)
    prom, left, right = signal.peak_prominences(x, idx)
    assert [p["index"] for p in ours] == idx.tolist()
    assert [p["prominence"] for p in ours] == prom.tolist()
    assert [p["left_base"] for p in ours] == left.tolist()
    assert [p["right_base"] for p in ours] == right.tolist()


# ---------------------------------------------------------------------------------------------------------------
# threshold_crossing
# ---------------------------------------------------------------------------------------------------------------


def test_crossings_linear_rising_and_falling():
    t, x = arange_t(5), [0.0, 2.0, 4.0, 2.0, 0.0]
    got = crossings(t, x, 3.0)
    # rising (1, 2): 2 < 3 <= 4, tc = 1 + 1 * (3 - 2) / (4 - 2) = 1.5, state high at 2
    # falling (2, 3): 4 >= 3 > 2, tc = 2 + 1 * (3 - 4) / (2 - 4) = 2.5, state low at 3
    assert [(c["t"], c["edge"], c["index_before"], c["index_after"], c["confirm_index"], c["gap"])
            for c in got["accepted"]] == [(1.5, "rising", 1, 2, 2, False), (2.5, "falling", 2, 3, 3, False)]
    assert (got["pending_at_end"], got["raw_level_crossings"], got["flips"], got["gap_flips"]) == (False, 2, 2, 0)


def test_crossings_level_equal_to_threshold():
    t, x = arange_t(4), [2.0, 3.0, 3.0, 2.0]
    got = crossings(t, x, 3.0)
    # (0, 1): 2 < 3 <= 3 rising, tc = 0 + (3 - 2) / (3 - 2) = 1
    # (1, 2): 3 >= 3 > 3 is false -> no crossing
    # (2, 3): 3 >= 3 > 2 falling, tc = 2 + (3 - 3) / (2 - 3) = 2
    assert [(c["t"], c["edge"]) for c in got["accepted"]] == [(1.0, "rising"), (2.0, "falling")]
    assert got["raw_level_crossings"] == 2


def test_crossings_edge_filter_and_interpolate_none():
    t, x = arange_t(5), [0.0, 2.0, 4.0, 2.0, 0.0]
    assert [c["t"] for c in crossings(t, x, 3.0, edge="rising")["accepted"]] == [1.5]
    assert [c["t"] for c in crossings(t, x, 3.0, edge="falling")["accepted"]] == [2.5]
    # interpolate none: tc = t[i + 1] -> rising 2.0, falling 3.0
    got = crossings(t, x, 3.0, interpolate="none")
    assert [(c["t"], c["index_before"]) for c in got["accepted"]] == [(2.0, 1), (3.0, 2)]


def test_crossings_across_a_non_finite_gap():
    t, x = arange_t(5), [0.0, 1.0, NAN, 5.0, 6.0]
    got = crossings(t, x, 3.0)
    # finite samples 0, 1, 3, 4; pair (1, 3) spans the NaN: 1 < 3 <= 5 rising
    # tc = 1 + (3 - 1) * (3 - 1) / (5 - 1) = 1 + 2 * 2 / 4 = 2
    assert [(c["t"], c["index_before"], c["index_after"], c["confirm_index"], c["gap"])
            for c in got["accepted"]] == [(2.0, 1, 3, 3, True)]
    assert got["gap_flips"] == 1


def test_crossings_leading_nan_sets_initial_state_from_first_finite():
    got = crossings(arange_t(4), [NAN, NAN, 5.0, 1.0], 3.0)
    # initial state: x[2] = 5 >= 3 -> high; (2, 3) falling, tc = 2 + (3 - 5) / (1 - 5) = 2.5
    assert [(c["t"], c["edge"], c["gap"]) for c in got["accepted"]] == [(2.5, "falling", False)]


def test_crossings_hysteresis_suppresses_chatter():
    t = arange_t(10)
    x = [-2.0, 0.5, -0.5, 0.5, -0.5, 2.0, 0.5, -0.5, 0.5, -2.0]
    # V = 0: level crossings at pairs (0,1) r, (1,2) f, (2,3) r, (3,4) f, (4,5) r, (6,7) f, (7,8) r, (8,9) f -> 8
    plain = crossings(t, x, 0.0)
    assert plain["raw_level_crossings"] == 8 and plain["flips"] == 8 and len(plain["accepted"]) == 8

    # H = 2: hi = 1, lo = -1. Initial -2 < 0 -> low. Only @5 (2 >= 1) and @9 (-2 <= -1) change the state.
    got = crossings(t, x, 0.0, hysteresis=2.0)
    # flip @5 takes the last rising crossing with i + 1 <= 5: (4, 5), tc = 4 + (0 + 0.5) / (2 + 0.5) = 4.2
    # flip @9 takes the last falling crossing with i + 1 <= 9: (8, 9), tc = 8 + (0 - 0.5) / (-2 - 0.5) = 8.2
    assert summary(got) == [("rising", 4, 5, False), ("falling", 8, 9, False)]
    assert [c["t"] for c in got["accepted"]] == [pytest.approx(4.2, abs=1e-12), pytest.approx(8.2, abs=1e-12)]
    assert (got["raw_level_crossings"], got["flips"]) == (8, 2)


def test_crossings_hysteresis_confirmation_after_the_crossing():
    # H = 2 (hi = 1, lo = -1). Initial -0.5 >= 0 is false -> low. @1 (0.5) holds; @2 (1 >= hi) turns high.
    # Last rising crossing with i + 1 <= 2: (0, 1): -0.5 < 0 <= 0.5, tc = 0 + 0.5 / 1 = 0.5
    got = crossings(arange_t(3), [-0.5, 0.5, 1.0], 0.0, hysteresis=2.0)
    assert [(c["t"], c["index_before"], c["index_after"], c["confirm_index"]) for c in got["accepted"]] == [
        (0.5, 0, 1, 2)
    ]
    # Exactly lo confirms low: initial 0 >= 0 -> high; @1 = 1 stays high; @2 = 0 holds; @3 = -1 <= lo -> low.
    # Last falling crossing with i + 1 <= 3: (2, 3): 0 >= 0 > -1, tc = 2 + (0 - 0) / (-1 - 0) = 2
    got = crossings(arange_t(4), [0.0, 1.0, 0.0, -1.0], 0.0, hysteresis=2.0)
    assert [(c["t"], c["edge"], c["index_before"], c["confirm_index"]) for c in got["accepted"]] == [
        (2.0, "falling", 2, 3)
    ]


def test_crossings_debounce_rejection_and_pending_at_end():
    t = arange_t(10)
    x = [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
    # flips: up @1 tc 0.5, down @2 tc 1.5, up @4 tc 3.5, down @8 tc 7.5; last sample time 9
    # runs: 0.5 -> 1.5 = 1.0, 1.5 -> 3.5 = 2.0, 3.5 -> 7.5 = 4.0, 7.5 -> 9 = 1.5
    # D = 1.5: up@0.5 rejected; down@1.5 accepted but the accepted state is already low (no transition);
    # up@3.5 accepted -> rising 3.5; down@7.5 run 1.5 >= 1.5 accepted -> falling 7.5
    got = crossings(t, x, 0.0, debounce=1.5)
    assert [(c["t"], c["edge"]) for c in got["accepted"]] == [(3.5, "rising"), (7.5, "falling")]
    assert got["pending_at_end"] is False and got["flips"] == 4

    # D = 2: the final run 1.5 < 2 -> pending_at_end, and the falling flip is not accepted
    got = crossings(t, x, 0.0, debounce=2.0)
    assert [(c["t"], c["edge"]) for c in got["accepted"]] == [(3.5, "rising")]
    assert got["pending_at_end"] is True

    # pending_at_end comes from step 4, before the edge filter; here the flip that left the run pending is the
    # falling one, so the edge-filtered flag agrees with it
    got = crossings(t, x, 0.0, debounce=2.0, edge="falling")
    assert got["accepted"] == [] and got["pending_at_end"] is True
    assert got["pending_at_end_edge"] is True


def test_crossings_report_the_step_4_caveats_raw_and_per_edge():
    # [0, nan, 2, 0, ...] at V = 1: the rising flip's level crossing spans the NaN and is a gap flip; the falling
    # one, pair (2, 3), does not. A `falling` requirement never claimed the rising flip.
    t, x = arange_t(8), [0.0, NAN, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    got = crossings(t, x, 1.0, edge="falling")
    assert [(c["edge"], c["gap"]) for c in got["accepted"]] == [("falling", False)]
    assert (got["gap_flips"], got["gap_flips_accepted"]) == (1, 0)
    # with both edges the gap flip is one of the accepted crossings, so both counts see it
    both = crossings(t, x, 1.0)
    assert (both["gap_flips"], both["gap_flips_accepted"]) == (1, 1)

    # x = [2, 2, 2, 0 ... 0, 2] at V = 1 with D = 2: falling flip tc 2.5 (run 6 -> accepted), rising flip tc 8.5
    # whose run to the last sample time is 9 - 8.5 = 0.5 < 2, so step 4 reports pending_at_end.
    t2, x2 = arange_t(10), [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    falling = crossings(t2, x2, 1.0, debounce=2.0, edge="falling")
    assert [(c["t"], c["edge"]) for c in falling["accepted"]] == [(2.5, "falling")]
    # the pending run was started by the rising flip, which this requirement filtered out
    assert falling["pending_at_end"] is True and falling["pending_at_end_edge"] is False
    assert crossings(t2, x2, 1.0, debounce=2.0, edge="rising")["pending_at_end_edge"] is True
    assert crossings(t2, x2, 1.0, debounce=2.0)["pending_at_end_edge"] is True
    # an event triggers on one fixed edge and inherits the filtered flag
    event = event_triggers(t2, x2, "falls_below", 1.0, debounce=2.0)
    assert [g["t"] for g in event["triggers"]] == [2.5]
    assert event["pending_at_end"] is True and event["pending_at_end_edge"] is False
    assert event_triggers(t2, x2, "rises_above", 1.0, debounce=2.0)["pending_at_end_edge"] is True
    assert event_triggers(arange_t(3), [1, 3, 3], "equals", 3)["pending_at_end_edge"] is False


def test_crossings_debounce_uses_last_sample_time_for_final_run():
    # one flip @2 (tc = 1 + (0 + 1) / 2 = 1.5); last sample time 4 -> final run 2.5
    t, x = arange_t(5), [-1.0, -1.0, 1.0, 1.0, 1.0]
    assert [c["t"] for c in crossings(t, x, 0.0, debounce=2.5)["accepted"]] == [1.5]
    got = crossings(t, x, 0.0, debounce=2.6)
    assert got["accepted"] == [] and got["pending_at_end"] is True


def test_crossings_duplicate_timestamps_and_degenerate_inputs():
    # pair (1, 2) has equal timestamps: tc = 1 + 0 * (5 - 0) / (10 - 0) = 1, no division by zero
    got = crossings([0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 10.0, 10.0], 5.0)
    assert [(c["t"], c["index_before"]) for c in got["accepted"]] == [(1.0, 1)]
    empty = {
        "accepted": [],
        "pending_at_end": False,
        "pending_at_end_edge": False,
        "raw_level_crossings": 0,
        "flips": 0,
        "gap_flips": 0,
        "gap_flips_accepted": 0,
    }
    assert crossings([], [], 0.0) == empty
    assert crossings(arange_t(3), [NAN, NAN, NAN], 0.0) == empty
    assert crossings([0.0], [1.0], 0.0) == empty
    with pytest.raises(ValueError):
        crossings(arange_t(2), [0.0, 1.0], 0.5, edge="up")


def test_crossings_epoch_scale_times():
    t = 1.7e9 + np.arange(5) / 1000  # 1 kHz clock at epoch scale, ulp(1.7e9) = 2^-22
    x = [0.0, 0.0, 1.0, 1.0, 0.0]
    got = crossings(t, x, 0.25)
    # rising (1, 2): tc = t[1] + (t[2] - t[1]) * 0.25; falling (3, 4): tc = t[3] + (t[4] - t[3]) * 0.75
    expected = [t[1] + 0.00025, t[3] + 0.00075]
    for c, want, i in zip(got["accepted"], expected, (1, 3)):
        assert abs(c["t"] - want) <= 4 * ulp(max(abs(t[i]), abs(t[i + 1])))
        assert t[i] <= c["t"] <= t[i + 1]


# ---------------------------------------------------------------------------------------------------------------
# violation
# ---------------------------------------------------------------------------------------------------------------


def runs_summary(runs: list[dict]) -> list[tuple]:
    return [
        (r["first_index"], r["last_index"], r["worst_index"], r["open_start"], r["open_end"], r["gap_start"],
         r["gap_end"])
        for r in runs
    ]


def test_violation_runs_above_with_interpolated_edges():
    t = arange_t(8)
    x = [0.0, 3.0, 4.0, 1.0, 5.0, 5.0, 2.0, 3.0]
    runs = violation_runs(t, x, above=2.0)
    # run [1, 2]: start 0 + (2 - 0) / (3 - 0) = 2/3; end 2 + (2 - 4) / (1 - 4) = 8/3; worst 4 @2
    # run [4, 5]: start 3 + (2 - 1) / (5 - 1) = 3.25; end 5 + (2 - 5) / (2 - 5) = 6; worst tie 5 -> @4
    # x[6] = 2 is not > 2, so the run ends before it
    # run [7]: start 6 + (2 - 2) / (3 - 2) = 6; ends at the signal end -> open_end, end = t[7] = 7
    assert runs_summary(runs) == [
        (1, 2, 2, False, False, False, False),
        (4, 5, 4, False, False, False, False),
        (7, 7, 7, False, True, False, False),
    ]
    assert [(r["start"], r["end"]) for r in runs] == [
        (pytest.approx(2 / 3), pytest.approx(8 / 3)),
        (3.25, 6.0),
        (6.0, 7.0),
    ]
    # durations 2, 2.75, 1 -> min_duration 1.5 keeps the first two
    assert [r["first_index"] for r in violation_runs(t, x, above=2.0, min_duration=1.5)] == [1, 4]


def test_violation_runs_below_open_start_and_gaps():
    t = arange_t(6, 0.5)  # 0, 0.5, 1, 1.5, 2, 2.5
    x = [-1.0, -2.0, NAN, -3.0, 0.5, -1.0]
    runs = violation_runs(t, x, below=0.0)
    # run [0, 1]: open_start (start t[0] = 0); ends at the NaN -> gap_end, end t[1] = 0.5; worst -2 @1
    # run [3]: after the NaN -> gap_start, start t[3] = 1.5; end 1.5 + 0.5 * (0 + 3) / (0.5 + 3) = 1.5 + 3/7
    # run [5]: start 2 + 0.5 * (0 - 0.5) / (-1 - 0.5) = 2 + 1/6; open_end, end t[5] = 2.5
    assert runs_summary(runs) == [
        (0, 1, 1, True, False, False, True),
        (3, 3, 3, False, False, True, False),
        (5, 5, 5, False, True, False, False),
    ]
    assert [(r["start"], r["end"]) for r in runs] == [
        (0.0, 0.5),
        (1.5, pytest.approx(1.5 + 3 / 7)),
        (pytest.approx(2 + 1 / 6), 2.5),
    ]


def test_violation_runs_degenerate_inputs():
    assert violation_runs([], [], above=0.0) == []
    assert violation_runs(arange_t(3), [NAN, NAN, NAN], above=0.0) == []
    assert violation_runs(arange_t(3), [0.0, 0.0, 0.0], above=0.0) == []
    # a single violating sample in the middle of a run of NaN: gap on both sides, duration 0
    (run,) = violation_runs(arange_t(3), [NAN, 5.0, NAN], above=0.0)
    assert (run["start"], run["end"], run["gap_start"], run["gap_end"]) == (1.0, 1.0, True, True)
    assert violation_runs(arange_t(3), [NAN, 5.0, NAN], above=0.0, min_duration=0.1) == []
    with pytest.raises(ValueError):
        violation_runs(arange_t(2), [0.0, 1.0])
    with pytest.raises(ValueError):
        violation_runs(arange_t(2), [0.0, 1.0], above=1.0, below=0.0)


# ---------------------------------------------------------------------------------------------------------------
# state_transitions
# ---------------------------------------------------------------------------------------------------------------


def test_state_transitions():
    # first 0; changes at 2 (1 -> 2) and 5 (2 -> 1); last 5
    assert state_transitions([1, 1, 2, 2, 2, 1]) == [0, 2, 5]
    # NaN equals NaN: changes at 2 (nan -> 1) and 3 (1 -> nan); last 4
    assert state_transitions([NAN, NAN, 1.0, NAN, NAN]) == [0, 2, 3, 4]
    assert state_transitions(np.array([3, 3, 3], dtype=np.int8)) == [0, 2]
    assert state_transitions([7]) == [0]
    assert state_transitions([]) == []
    # vector rows differ when any component differs (NaN == NaN per component)
    assert state_transitions([[1.0, NAN], [1.0, NAN], [1.0, 2.0], [1.0, 2.0]]) == [0, 2, 3]


# ---------------------------------------------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------------------------------------------


def test_event_triggers_falls_below_occurrence():
    t, x = arange_t(4), [2.0, 0.0, 2.0, 0.0]
    # falling crossings of 1: (0, 1) tc = 0 + (1 - 2) / (0 - 2) = 0.5; (2, 3) tc = 2.5 (rising 1.5 is filtered)
    all_ = event_triggers(t, x, "falls_below", 1.0)
    assert all_["found"] == 2
    assert [(g["t"], g["index_before"], g["index_after"], g["at_start"]) for g in all_["triggers"]] == [
        (0.5, 0, 1, False),
        (2.5, 2, 3, False),
    ]
    assert [g["t"] for g in event_triggers(t, x, "falls_below", 1.0, occurrence="first")["triggers"]] == [0.5]
    assert [g["t"] for g in event_triggers(t, x, "falls_below", 1.0, occurrence="last")["triggers"]] == [2.5]
    # rises_above: (1, 2) tc = 1 + (1 - 0) / (2 - 0) = 1.5
    assert [g["t"] for g in event_triggers(t, x, "rises_above", 1.0)["triggers"]] == [1.5]


def test_event_triggers_rises_above_with_hysteresis_and_debounce():
    t = arange_t(10)
    x = [-2.0, 0.5, -0.5, 0.5, -0.5, 2.0, 0.5, -0.5, 0.5, -2.0]
    # same chatter as the crossing test: with H = 2 the only rising flip is (4, 5) at 4.2, run 4.2 -> 8.2 = 4
    got = event_triggers(t, x, "rises_above", 0.0, hysteresis=2.0, debounce=3.0)
    assert [(g["index_before"], g["index_after"]) for g in got["triggers"]] == [(4, 5)]
    # the falling flip at 8.2 has a final run of 9 - 8.2 = 0.8 < 3 -> pending
    assert got["pending_at_end"] is True


def test_event_triggers_equals_runs():
    t, x = arange_t(6, 0.1), [3, 3, 1, 3, 2, 3]
    got = event_triggers(t, x, "equals", 3)
    # runs of x == 3 start at 0 (at_start), 3 and 5
    assert [(g["index_before"], g["index_after"], g["at_start"]) for g in got["triggers"]] == [
        (None, 0, True),
        (2, 3, False),
        (4, 5, False),
    ]
    assert [g["t"] for g in got["triggers"]] == [t[0], t[3], t[5]]
    assert event_triggers(t, x, "equals", 9)["found"] == 0


def test_event_window_indices():
    t = arange_t(10)  # 0 .. 9
    # tau 4.5, before 1, after 2: lo = first t >= 3.5 -> 4; hi = last t <= 6.5 -> 6; retain [3, 7]
    assert event_window_indices(t, 4.5, 1.0, 2.0) == (3, 7)
    assert event_window_clipped(t, 4.5, 1.0, 2.0) is False
    # exact hits: lo = first t >= 4 -> 4, hi = last t <= 6 -> 6 -> [3, 7]
    assert event_window_indices(t, 5.0, 1.0, 1.0) == (3, 7)
    # window between two samples: lo = first t >= 4.4 -> 5, hi = last t <= 4.6 -> 4 -> [4, 5], the bracketing pair
    assert event_window_indices(t, 4.5, 0.1, 0.1) == (4, 5)
    # clipped at the start: lo = first t >= -1.5 -> 0, hi = last t <= 0.7 -> 0 -> [max(-1, 0), 1]
    assert event_window_indices(t, 0.5, 2.0, 0.2) == (0, 1)
    assert event_window_clipped(t, 0.5, 2.0, 0.2) is True
    # clipped at the end: lo = first t >= 8.5 -> 9, hi = last t <= 13.5 -> 9 -> [8, min(10, 9)]
    assert event_window_indices(t, 8.5, 0.0, 5.0) == (8, 9)
    assert event_window_clipped(t, 8.5, 0.0, 5.0) is True
    # entirely past the end: lo = 10 (none), hi = 9 -> [9, 9]
    assert event_window_indices(t, 20.0, 1.0, 1.0) == (9, 9)
    assert event_window_indices([], 1.0, 1.0, 1.0) is None


def test_event_window_indices_1khz_epoch_clock():
    t = 1.7e9 + np.arange(5000) / 1000
    tau = float(t[2000])  # an exact sample time
    # before = after = 0.1 s: samples 1900 .. 2100 qualify up to rounding of t; the one-neighbour extension
    # retains 1899 .. 2101 when the grid is exact at those samples.
    lo, hi = event_window_indices(t, tau, 0.1, 0.1)
    first = int(np.flatnonzero(t >= tau - 0.1)[0])
    last = int(np.flatnonzero(t <= tau + 0.1)[-1])
    assert (lo, hi) == (first - 1, last + 1)
    assert abs(first - 1900) <= 1 and abs(last - 2100) <= 1


# ---------------------------------------------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------------------------------------------


def test_sed_max_basic_vectors():
    t = arange_t(5)
    P = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]])
    keep = [0, 4]
    # recon(t) = (t, 0); SED = [0, 1, 0, 1, 0]; ties -> lowest index 1
    assert sed_max(t, P, t[keep], P[keep]) == (1.0, 1)
    # keeping everything gives zero error
    assert sed_max(t, P, t, P) == (0.0, 0)


def test_sed_max_three_four_five():
    t = arange_t(3)
    P = np.array([[0.0, 0.0], [6.0, 4.0], [6.0, 0.0]])
    # recon(1) = (0, 0) + 0.5 * (6, 0) = (3, 0); P[1] - recon = (3, 4) -> 5
    assert sed_max(t, P, t[[0, 2]], P[[0, 2]]) == (5.0, 1)


def test_sed_max_equal_retained_timestamps_are_a_discontinuity():
    t = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])
    P = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 10.0], [1.5, 10.0], [2.0, 10.0]])
    keep = [0, 2, 3, 5]
    # 0.5 lies between (0, [0, 0]) and the first t = 1 sample [1, 0] -> recon [0.5, 0], SED 0
    # 1.5 lies between the second t = 1 sample [1, 10] and [2, 10] -> recon [1.5, 10], SED 0
    # at t = 1 both source samples equal one of the retained samples at that time -> SED 0
    assert sed_max(t, P, t[keep], P[keep]) == (0.0, 0)
    # move P[4] to [1.5, 13]: SED = |13 - 10| = 3 at index 4
    moved = P.copy()
    moved[4, 1] = 13.0
    assert sed_max(t, moved, t[keep], moved[keep]) == (3.0, 4)
    # a source sample at t = 1 that matches neither retained value: nearest is [1, 10] at distance 3 for [1, 7]
    t3 = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    P3 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 7.0], [1.0, 10.0], [2.0, 10.0]])
    keep3 = [0, 1, 3, 4]
    # distances to [1, 0] and [1, 10]: 7 and 3 -> 3
    assert sed_max(t3, P3, t3[keep3], P3[keep3]) == (3.0, 2)


def test_sed_max_three_retained_samples_at_one_timestamp():
    t = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    P = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 5.0], [1.0, 6.0], [1.0, 10.0], [2.0, 10.0]])
    keep = [0, 1, 2, 4, 5]
    # source [1, 6] is not retained; retained values at t = 1 are [1, 0], [1, 5], [1, 10] -> nearest [1, 5], SED 1
    assert sed_max(t, P, t[keep], P[keep]) == (1.0, 3)


def test_sed_max_gaps_outside_span_and_non_finite_sources():
    t = arange_t(5)
    P = np.array([[0.0, 0.0], [1.0, 0.0], [NAN, NAN], [3.0, 0.0], [4.0, 0.0]])
    # the NaN source sample is not evaluated; retained [0, 2, 4] brackets 1 and 3 with a NaN -> no reconstruction
    assert sed_max(t, P, t[[0, 2, 4]], P[[0, 2, 4]]) == (INF, 1)
    # retained [0, 1, 2, 3, 4] reproduces everything finite
    assert sed_max(t, P, t, P) == (0.0, 0)
    # a finite source sample before the first retained timestamp has no reconstruction
    assert sed_max(t, P, t[[1, 4]], P[[1, 4]]) == (INF, 0)
    assert sed_max(t, np.full((5, 2), NAN), t, np.full((5, 2), NAN)) == (0.0, None)
    # scalar positions are treated as one component: recon(1) = 0 + 0.5 * 4 = 2, |5 - 2| = 3
    assert sed_max([0.0, 1.0, 2.0], [0.0, 5.0, 4.0], [0.0, 2.0], [0.0, 4.0]) == (3.0, 1)


# ---------------------------------------------------------------------------------------------------------------
# Performance smoke test
# ---------------------------------------------------------------------------------------------------------------


def test_crossings_and_violation_runs_on_five_million_samples():
    n = 5_000_000
    t = np.arange(n, dtype=np.float64) * 1e-3  # 1 kHz for 5000 s
    x = np.sin(2 * np.pi * t / 500.0)  # ten slow periods
    x[1_000_000:1_000_010] = np.nan

    start = time.perf_counter()
    found = crossings(t, x, 0.5, hysteresis=0.1, debounce=1.0)
    runs = violation_runs(t, x, above=0.9, min_duration=1.0)
    elapsed = time.perf_counter() - start

    # sin crosses 0.5 upward once and downward once per period -> 10 + 10
    assert len(found["accepted"]) == 20
    # sin > 0.9 once per period -> 10 runs
    assert len(runs) == 10
    assert elapsed < 5.0, f"took {elapsed:.2f} s"


def test_crossings_and_violation_runs_on_noise_sitting_on_the_threshold():
    # The sine above has 20 flips and 10 runs in 5 M samples, so it says nothing about per-event cost. Noise on
    # the threshold is the other extreme: ~n/2 flips and ~n/4 violating runs, a candidate every few samples.
    n = 5_000_000
    rng = np.random.default_rng(5)
    x = rng.random(n) - 0.5
    t = arange_t(n, 1e-3)

    start = time.perf_counter()
    # A debounce of 50 ms (50 samples) rejects every flip -- a run of 50 samples of one sign has probability
    # 2**-49 per position -- while steps 1 to 4 still see all of them.
    found = crossings(t, x, 0.0, debounce=0.05)
    runs = violation_runs(t, x, above=0.0, min_duration=0.05)
    # The same machine on a fifth of the samples with every crossing kept, so records are built as well.
    reported = crossings(t[: n // 5], x[: n // 5], 0.0)
    elapsed = time.perf_counter() - start

    assert found["flips"] > n // 3 and found["accepted"] == []
    assert runs == []
    # flips alternate direction, so with no debounce every flip is an accepted transition
    assert len(reported["accepted"]) == reported["flips"] > n // 15
    assert elapsed < 5.0, f"took {elapsed:.2f} s"


def test_local_peaks_on_a_signal_where_every_other_sample_is_a_peak():
    # A rising zigzag, x = [0, 3, 2, 5, 4, 7, ...]: every odd sample is a peak and nothing to its left is higher,
    # so scanning the base interval per peak walks back to sample 0 every time, which is quadratic. Peak @k has
    # value k + 2, left base @0 (value 0) and right base @k+1 (value k + 1), so prominence (k + 2) - (k + 1) = 1.
    n = 1_000_000
    x = np.arange(n, dtype=np.float64)
    x[1::2] += 2.0
    t = arange_t(n, 1e-3)
    # An ADC-quantized signal dithering over one LSB has the same shape: runs of 1.0 between runs of 0.0, whose
    # bases are the far ends of the signal because nothing is strictly higher than a peak.
    rng = np.random.default_rng(11)
    dithered = (rng.random(n) < 0.3).astype(np.float64)

    start = time.perf_counter()
    peaks = local_peaks(t, x)
    quantized = local_peaks(t, dithered)
    elapsed = time.perf_counter() - start

    # the last run is never a peak, so the peaks are the odd indices up to n - 3
    assert len(peaks) == n // 2 - 1
    assert peaks_summary(peaks[:1]) == [(1, 1.0, 0, 2, "max")]
    assert peaks_summary(peaks[-1:]) == [(n - 3, 1.0, 0, n - 2, "max")]
    middle = peaks[len(peaks) // 2]
    assert (middle["left_base"], middle["right_base"], middle["prominence"]) == (0, middle["index"] + 1, 1.0)
    assert quantized and {p["prominence"] for p in quantized} == {1.0}
    assert elapsed < 5.0, f"took {elapsed:.2f} s"
