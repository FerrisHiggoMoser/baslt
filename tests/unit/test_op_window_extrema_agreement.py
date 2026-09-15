"""The compiler and the independent verifier must agree on bucket existence and membership.

A disagreement here would make every artifact fail its own self-check, so these tests compare
baslt.ops.window_extrema against baslt.verify.reference on the awkward cases in docs/contracts.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from baslt.ops import window_extrema as op
from baslt.signals import normalize_signal
from baslt.verify import reference as ref


def _evidence(t, x, interval, origin=0.0):
    sig, _ = normalize_signal("s", t, x)
    result = op.evaluate(sig, {"interval": interval, "origin": origin}, {"main": 0})
    return result, ref.window_evidence(t, x, interval, origin)


def test_first_sample_on_a_boundary_does_not_invent_a_bucket():
    # t = 0 sits exactly on boundary 0, so it is borrowed into bucket -1, which must not exist.
    t = np.arange(0.0, 0.5, 0.001)
    x = np.sin(t * 10.0)
    result, verifier = _evidence(t, x, 0.1)
    assert result.evidence["buckets"] == 5
    assert result.evidence["buckets"] == verifier["buckets"]
    assert result.evidence["buckets_with_finite"] == verifier["buckets_with_finite"]


def test_bucket_whose_own_samples_are_all_nan_still_counts_when_it_borrows_one():
    t = np.arange(0.0, 0.5, 0.001)
    x = np.sin(t * 10.0)
    x[200:300] = np.nan  # bucket 2 keeps only the boundary sample it borrows from bucket 3
    result, verifier = _evidence(t, x, 0.1)
    assert result.evidence["buckets"] == 5 == verifier["buckets"]
    assert result.evidence["buckets_with_finite"] == verifier["buckets_with_finite"] == 5


def test_retention_covers_no_more_than_the_existing_buckets():
    t = np.arange(0.0, 0.5, 0.001)
    x = np.sin(t * 10.0)
    sig, _ = normalize_signal("s", t, x)
    result = op.evaluate(sig, {"interval": 0.1, "origin": 0.0}, {"main": 0})
    idx, _ = result.samples.materialize()
    ids, _ = op.bucket_ids(t, 0.1, 0.0)
    # every retained sample belongs to a bucket that exists, and at most 2 per bucket per component
    assert set(ids[idx].tolist()) <= set(np.unique(ids).tolist())
    assert idx.size <= 2 * int(np.unique(ids).size)


@pytest.mark.parametrize(
    "t0, interval",
    [(0.0, 0.1), (0.05, 0.1), (-1.0, 0.25), (1_700_000_000.0, 1.0)],
)
def test_counts_agree_across_clocks(t0, interval):
    t = t0 + np.arange(0, 1200) * 0.001
    x = np.cos(t)
    result, verifier = _evidence(t, x, interval)
    assert result.evidence["buckets"] == verifier["buckets"]
    assert result.evidence["buckets_with_finite"] == verifier["buckets_with_finite"]


def test_gap_in_time_creates_no_empty_buckets():
    t = np.concatenate([np.arange(0.0, 0.2, 0.001), np.arange(5.0, 5.2, 0.001)])
    x = np.arange(t.size, dtype=float)
    result, verifier = _evidence(t, x, 0.1)
    ids, _ = op.bucket_ids(t, 0.1, 0.0)
    # the long empty stretch between 0.2 s and 5.0 s must not create buckets
    assert result.evidence["buckets"] == verifier["buckets"] == int(np.unique(ids).size)
    assert result.evidence["buckets"] < 10
