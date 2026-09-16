"""local_extrema: the contract on hand-built signals, the range queries, agreement with the verifier."""

from __future__ import annotations

import time

import numpy as np
import pytest

from baslt.ops import local_extrema as op
from baslt.ops.local_extrema import _previous_greater, _RangeMin, peaks
from baslt.signals import normalize_signal
from baslt.verify import reference as ref

BITS = {"peak": 3, "base": 4}


def _eval(x, t=None, **params):
    x = np.asarray(x, dtype=float)
    t = np.arange(x.shape[0], dtype=float) if t is None else np.asarray(t, dtype=float)
    sig, _ = normalize_signal("s", t, x, kind="vector" if x.ndim == 2 else None)
    params.setdefault("prominence", 0.0)
    result = op.evaluate(sig, params, BITS)
    idx, roles = result.samples.materialize()
    peaks_at = idx[(roles & np.uint64(1 << 3)) != 0].tolist()
    bases_at = idx[(roles & np.uint64(1 << 4)) != 0].tolist()
    return result, peaks_at, bases_at


# ---------------------------------------------------------------------------------------------- contract


def test_simple_peak_and_its_bases():
    #        0  1  2  3  4  5  6
    x = [0, 3, 1, 5, 2, 4, 0]
    found = peaks(np.arange(7.0), np.array(x, float), 0.0, 0.0)
    assert found["index"].tolist() == [1, 3, 5]
    # peak 1: stops at 3 on the right, bases 0 and 2; peak 3 (value 5): no higher sample on either side, so
    # its bases are the minima of the whole left and right parts, 0 at 0 and 0 at 6; peak 5: stops at 3, bases 4, 6
    assert found["left_base"].tolist() == [0, 0, 4]
    assert found["right_base"].tolist() == [2, 6, 6]
    assert found["prominence"].tolist() == [2.0, 5.0, 2.0]


def test_plateau_peak_index_is_the_middle_rounded_down():
    x = np.array([0, 2, 2, 2, 2, 0], float)
    assert peaks(np.arange(6.0), x, 0.0, 0.0)["index"].tolist() == [2]


def test_first_and_last_runs_are_never_peaks():
    assert peaks(np.arange(3.0), np.array([5.0, 1.0, 5.0]), 0.0, 0.0)["index"].size == 0
    assert peaks(np.arange(3.0), np.array([5.0, 5.0, 5.0]), 0.0, 0.0)["index"].size == 0


def test_non_finite_samples_are_walls():
    # a finite run next to a wall is never a peak; walls also stop the base search
    x = np.array([0, 3, np.nan, 1, 2, 1, np.inf, 0], float)
    found = peaks(np.arange(8.0), x, 0.0, 0.0)
    assert found["index"].tolist() == [4]
    assert found["left_base"].tolist() == [3] and found["right_base"].tolist() == [5]
    assert found["prominence"].tolist() == [1.0]


def test_base_ties_take_the_sample_nearest_the_peak():
    x = np.array([0, 1, 0, 1, 0, 9, 0, 1, 0], float)
    found = peaks(np.arange(9.0), x, 3.0, 0.0)
    assert found["index"].tolist() == [5]
    assert found["left_base"].tolist() == [4] and found["right_base"].tolist() == [6]


def test_base_interval_stops_at_the_nearest_strictly_higher_sample():
    x = np.array([0, 9, 1, 4, 2, 6, 3, 5, 0], float)
    found = peaks(np.arange(9.0), x, 0.0, 0.0)
    record = dict(zip(found["index"].tolist(), zip(found["left_base"].tolist(), found["right_base"].tolist(),
                                                     found["prominence"].tolist())))
    # peak 7 (value 5): left stops at 5 (value 6), base 6 (value 3); right base 8 (value 0)
    assert record[7] == (6, 8, 2.0)
    # peak 3 (value 4): left stops at 1 (value 9), base 2; right stops at 5, base 4 -> prominence 4 - 2 = 2
    assert record[3] == (2, 4, 2.0)


def test_prominence_filter_and_separation_greedy():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    x = np.array([0, 5, 0, 7, 0, 7, 0, 0], float)
    assert peaks(t, x, 6.0, 0.0)["index"].tolist() == [3, 5]
    # equal heights: the lower index is visited first, the other is within 2.5 s and dropped
    assert peaks(t, x, 0.0, 2.5)["index"].tolist() == [3]
    assert peaks(t, x, 0.0, 2.0)["index"].tolist() == [1, 3, 5]  # exactly 2 s apart is allowed
    # duplicate timestamps are zero seconds apart
    same_t = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    assert peaks(same_t, x, 0.0, 0.1)["index"].tolist() == [3]


def test_evaluate_minima_both_and_roles():
    x = [0, 3, 1, 5, 2, 4, 0]
    result, at_peaks, at_bases = _eval(x, prominence=1.5, kind="both")
    assert result.status == "pass"
    assert result.evidence["maxima"] == 3 and result.evidence["minima"] == 2
    assert at_peaks == [1, 2, 3, 4, 5]
    assert set(at_bases) >= {0, 2, 4, 6}
    kinds = {(p["index"], p["kind"]) for p in result.evidence["peaks"]}
    assert (2, "min") in kinds and (3, "max") in kinds
    only_max, peaks_max, _ = _eval(x, prominence=1.5, kind="max")
    assert only_max.evidence["minima"] == 0 and peaks_max == [1, 3, 5]


def test_vector_samples_are_finite_only_when_every_component_is():
    x = np.array([[0, 0], [5, 1], [0, 0], [4, np.nan], [0, 0], [3, 2], [0, 0]], float)
    result, at_peaks, _ = _eval(x, prominence=0.5, kind="max")
    assert at_peaks == [1, 5]
    comps = sorted((p["index"], p["component"]) for p in result.evidence["peaks"])
    assert comps == [(1, 0), (1, 1), (5, 0), (5, 1)]


def test_not_applicable_and_parameter_errors():
    result, at_peaks, _ = _eval([np.nan, np.nan, np.nan], prominence=1.0)
    assert result.status == "not_applicable" and at_peaks == []
    assert _eval([], prominence=1.0)[0].status == "not_applicable"
    assert _eval([1.0, 1.0], prominence=1.0)[0].status == "pass"
    with pytest.raises(ValueError, match="kind"):
        _eval([0, 1, 0], kind="peak")
    with pytest.raises(ValueError, match="prominence"):
        _eval([0, 1, 0], prominence=-1.0)
    with pytest.raises(ValueError, match="separation"):
        _eval([0, 1, 0], separation=float("nan"))


def test_evidence_is_capped_but_counts_are_exact():
    x = np.tile([0.0, 1.0], 400)
    result, at_peaks, _ = _eval(x, prominence=0.5, kind="max")
    assert result.evidence["maxima"] == 399
    assert len(result.evidence["peaks"]) == 256
    assert len(at_peaks) == 399


# ---------------------------------------------------------------------------------------------- queries


def test_range_min_matches_a_naive_scan_with_rightmost_ties():
    rng = np.random.default_rng(0)
    for n in (1, 2, 15, 16, 17, 64, 333):
        values = rng.integers(0, 4, n).astype(float)
        rmq = _RangeMin(values)
        a = rng.integers(0, n, 400)
        b = rng.integers(0, n, 400)
        a, b = np.minimum(a, b), np.maximum(a, b)
        got = rmq.query(a, b)
        for lo, hi, g in zip(a.tolist(), b.tolist(), got.tolist()):
            segment = values[lo : hi + 1]
            want = lo + int(np.flatnonzero(segment == segment.min())[-1])
            assert g == want, (n, lo, hi)


def test_previous_greater_matches_a_naive_scan():
    rng = np.random.default_rng(1)
    for n in (1, 5, 16, 17, 100, 1000):
        values = rng.integers(0, 6, n).astype(float)
        values[rng.random(n) < 0.05] = np.inf
        targets = np.sort(rng.choice(n, size=min(n, 50), replace=False))
        got = _previous_greater(values, targets)
        for e, g in zip(targets.tolist(), got.tolist()):
            earlier = np.flatnonzero(values[:e] > values[e])
            assert g == (int(earlier[-1]) if earlier.size else -1)


# ---------------------------------------------------------------------------------------------- agreement


def test_agrees_with_the_verifier_reference_on_awkward_series():
    rng = np.random.default_rng(7)
    for _ in range(600):
        n = int(rng.integers(0, 90))
        y = rng.integers(0, 4, n).astype(float) if rng.random() < 0.5 else rng.standard_normal(n)
        if n:
            y[rng.random(n) < 0.08] = rng.choice([np.nan, np.inf, -np.inf])
        t = np.cumsum(rng.random(n) * (rng.random() < 0.8))
        p = float(rng.choice([0.0, 0.5, 1.5]))
        s = float(rng.choice([0.0, 0.8, 4.0]))
        for label, yy in (("max", y), ("min", -y)):
            got = peaks(t, yy, p, s)
            want = ref.local_peaks(t, y, p, s, kind=label)
            assert got["index"].tolist() == [w["index"] for w in want]
            assert got["left_base"].tolist() == [w["left_base"] for w in want]
            assert got["right_base"].tolist() == [w["right_base"] for w in want]
            assert got["prominence"].tolist() == [w["prominence"] for w in want]


def test_prominence_on_any_retained_superset_is_at_least_the_source_prominence():
    rng = np.random.default_rng(11)
    for _ in range(300):
        n = int(rng.integers(3, 150))
        y = np.cumsum(rng.standard_normal(n))
        if rng.random() < 0.3:
            y[rng.random(n) < 0.05] = np.nan
        t = np.arange(n, dtype=float)
        found = peaks(t, y, 0.5, 0.0)
        keep = set(found["index"].tolist()) | set(found["left_base"].tolist()) | set(found["right_base"].tolist())
        keep |= {0, n - 1} | set(np.flatnonzero(rng.random(n) < 0.2).tolist())
        idx = np.array(sorted(keep))
        at = np.searchsorted(idx, found["index"])
        retained = ref.prominences_at(y[idx], at)
        assert np.all(retained >= found["prominence"]), (retained, found["prominence"])


def test_large_adversarial_signals_stay_fast():
    n = 1_000_000
    k = np.arange(n)
    ringing = np.exp(-k / 2e5) * np.sin(k * 0.37)
    ringing[-10] = 5.0
    noise = np.random.default_rng(2).standard_normal(n)
    t = k * 1e-3
    start = time.perf_counter()
    for y in (ringing, -ringing, noise):
        peaks(t, y, 1e-3, 0.0)
    peaks(t, noise, 0.5, 0.05)
    assert time.perf_counter() - start < 6.0
