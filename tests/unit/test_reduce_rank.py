"""`preview_order`: the nested ranking the soft layer spends its bytes on."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.reduce.rank import _bit_reversed, preview_order

pytestmark = pytest.mark.minimal

RNG = np.random.default_rng(7)


def signals():
    n = 257
    noisy = RNG.normal(size=n)
    gappy = RNG.normal(size=n)
    gappy[40:60] = np.nan
    gappy[200] = np.inf
    return {
        "noisy": noisy,
        "flat": np.zeros(n),
        "steps": np.repeat([0.0, 3.0, 1.0, 1.0, 2.0], 52)[:n],
        "gappy": gappy,
        "all_nan": np.full(n, np.nan),
        "ints": RNG.integers(0, 4, size=n).astype(np.int32),
        "vector": RNG.normal(size=(n, 3)),
        "tiny": np.array([5.0]),
    }


@pytest.mark.parametrize("name", list(signals()))
def test_the_full_order_is_a_permutation(name):
    v = signals()[name]
    order = preview_order(v, v.shape[0])
    assert order.dtype == np.int64
    assert sorted(order.tolist()) == list(range(v.shape[0]))


@pytest.mark.parametrize("name", list(signals()))
def test_every_cap_is_a_prefix_of_the_full_order(name):
    v = signals()[name]
    full = preview_order(v, v.shape[0])
    for cap in (0, 1, 2, 3, 5, 8, 16, 17, 31, 64, 100, 128, 200, v.shape[0], v.shape[0] + 10):
        part = preview_order(v, cap)
        assert part.tolist() == full[:cap].tolist(), cap


def test_the_global_extremes_come_first():
    v = signals()["noisy"]
    first = set(preview_order(v, 2).tolist())
    assert first == {int(np.argmin(v)), int(np.argmax(v))}


def test_each_component_contributes_its_extremes_first():
    v = signals()["vector"]
    head = set(preview_order(v, 6).tolist())
    wanted = {int(f(v[:, k])) for k in range(3) for f in (np.argmin, np.argmax)}
    assert head == wanted


def test_gap_boundaries_follow_the_extremes():
    v = signals()["gappy"]
    finite = v[np.isfinite(v)]
    extremes = {int(np.flatnonzero(v == finite.min())[0]), int(np.flatnonzero(v == finite.max())[0])}
    # The first non-finite run spans 40..59, the single inf sits at 200.
    boundaries = {39, 40, 59, 60, 199, 200, 201}
    head = preview_order(v, len(extremes) + len(boundaries)).tolist()
    assert set(head) == extremes | boundaries


def test_non_finite_samples_are_ranked_only_at_the_end_or_as_gap_edges():
    v = signals()["all_nan"]
    order = preview_order(v, v.shape[0])
    # Nothing is finite: the gap run's own edges come first, then the even fill.
    assert order[:2].tolist() == [0, v.shape[0] - 1]
    assert order[2:].tolist() == [i for i in _bit_reversed(v.shape[0]).tolist() if i not in (0, v.shape[0] - 1)]


def test_a_step_signal_gets_its_edges_before_the_fill():
    v = signals()["steps"]
    edges = {int(i) for i in np.flatnonzero(np.diff(v)) + 1}
    head = set(preview_order(v, 16).tolist())
    # Every level keeps the first sample of each new value, so the first samples after each step are ranked early.
    assert edges <= head


def test_ties_pick_the_lowest_index():
    v = np.array([1.0, 3.0, 3.0, 1.0, 2.0, 3.0])
    assert preview_order(v, 2).tolist() == [0, 1]


def test_empty_and_negative_caps():
    assert preview_order(np.zeros(0), 5).tolist() == []
    assert preview_order(np.zeros(4), 0).tolist() == []
    assert preview_order(np.zeros(4), -3).tolist() == []


def test_bit_reversed_order_refines_evenly():
    assert _bit_reversed(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]
    assert _bit_reversed(5).tolist() == [0, 4, 2, 1, 3]
    assert _bit_reversed(1).tolist() == [0]
    assert _bit_reversed(0).tolist() == []


def test_the_order_is_deterministic():
    v = signals()["noisy"]
    assert preview_order(v, 50).tolist() == preview_order(v.copy(), 50).tolist()
