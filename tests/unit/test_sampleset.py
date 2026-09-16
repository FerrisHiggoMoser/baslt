from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from baslt.sampleset import SampleSet


def _model(ops):
    """Naive reference: dict index -> OR-ed role mask."""
    roles: dict[int, int] = {}
    for kind, a, b, bit in ops:
        if kind == "p":
            idx = [a]
        else:
            idx = range(a, b)
        for i in idx:
            roles[i] = roles.get(i, 0) | (1 << bit)
    return roles


def _apply(ops):
    s = SampleSet.empty()
    for kind, a, b, bit in ops:
        if kind == "p":
            s = s.add_points([a], bit)
        else:
            s = s.add_ranges([a], [b], bit)
    return s


def test_empty():
    s = SampleSet.empty()
    idx, roles = s.materialize()
    assert s.count() == 0 and idx.size == 0 and roles.size == 0
    assert s.is_empty


def test_points_dedupe_and_merge_adjacent():
    s = SampleSet.from_points([5, 3, 4, 4, 9], bit=0)
    assert s.starts.tolist() == [3, 9]
    assert s.stops.tolist() == [6, 10]
    assert s.count() == 4


def test_union_ors_roles_on_overlap():
    a = SampleSet.from_ranges([0], [10], bit=0)
    b = SampleSet.from_ranges([5], [15], bit=1)
    u = a.union(b)
    assert u.starts.tolist() == [0, 5, 10]
    assert u.stops.tolist() == [5, 10, 15]
    assert u.roles.tolist() == [1, 3, 2]
    idx, roles = u.materialize()
    assert idx.tolist() == list(range(15))
    assert roles[7] == 3


def test_large_window_costs_one_range():
    s = SampleSet.from_ranges([0], [60_000_000], bit=3)
    assert s.starts.shape == (1,)
    assert s.count() == 60_000_000


def test_roles_at_and_contains():
    s = SampleSet.from_ranges([2, 10], [4, 12], bit=1).add_points([20], bit=2)
    q = np.array([-1, 0, 2, 3, 4, 10, 11, 12, 20, 21])
    assert s.contains(q).tolist() == [False, False, True, True, False, True, True, False, True, False]
    assert s.roles_at(q)[8] == 4


def test_with_bits_and_indices_with():
    s = SampleSet.from_ranges([0], [5], bit=0).add_points([2, 7], bit=1)
    only1 = s.with_bits(1 << 1)
    assert only1.materialize()[0].tolist() == [2, 7]
    assert (only1.roles == 2).all()
    assert s.indices_with(1).tolist() == [0, 1, 2, 3, 4]


def test_clip():
    s = SampleSet.from_ranges([-3, 8], [2, 20], bit=0).clip(10)
    assert s.materialize()[0].tolist() == [0, 1, 8, 9]


def test_bits_present():
    s = SampleSet.from_points([1], 0).add_points([3], 5)
    assert s.bits_present() == (1 | 32)


def test_high_bits():
    s = SampleSet.from_points([0], 63)
    assert int(s.roles[0]) == 1 << 63


op = st.one_of(
    st.tuples(st.just("p"), st.integers(0, 60), st.just(0), st.integers(0, 6)),
    st.tuples(st.just("r"), st.integers(0, 60), st.integers(0, 60), st.integers(0, 6)),
)


@given(st.lists(op, max_size=25))
def test_matches_naive_model(ops):
    s = _apply(ops)
    model = _model(ops)
    idx, roles = s.materialize()
    assert idx.tolist() == sorted(model)
    assert [int(r) for r in roles] == [model[i] for i in sorted(model)]
    # canonical form: sorted, disjoint, no zero roles, no mergeable neighbours
    assert (s.stops > s.starts).all()
    assert (s.roles != 0).all()
    if s.starts.size > 1:
        assert (s.starts[1:] >= s.stops[:-1]).all()
        mergeable = (s.starts[1:] == s.stops[:-1]) & (s.roles[1:] == s.roles[:-1])
        assert not mergeable.any()
    assert s.count() == len(model)


@given(st.lists(op, max_size=15), st.lists(op, max_size=15))
def test_union_is_commutative(a_ops, b_ops):
    a, b = _apply(a_ops), _apply(b_ops)
    ab, ba = a.union(b), b.union(a)
    assert ab.starts.tolist() == ba.starts.tolist()
    assert ab.stops.tolist() == ba.stops.tolist()
    assert ab.roles.tolist() == ba.roles.tolist()
