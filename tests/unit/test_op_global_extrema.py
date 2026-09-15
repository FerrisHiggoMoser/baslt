"""Unit tests for the global_extrema operator."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from baslt.ops import global_extrema
from baslt.signals import normalize_signal
from reference.extrema_loop import global_extrema_loop

BITS = {"main": 3}


def _sig(v, t=None, kind=None):
    v = np.asarray(v)
    if t is None:
        t = np.arange(v.shape[0], dtype=np.float64) * 0.5
    sig, _ = normalize_signal("s", t, v, kind=kind)
    return sig


def _run(v, t=None, kind=None):
    sig = _sig(v, t, kind)
    res = global_extrema.evaluate(sig, {}, BITS)
    idx, roles = res.samples.materialize()
    assert (roles == np.uint64(1 << BITS["main"])).all()
    return res, idx.tolist()


def _pairs(res):
    return [
        (None if c["max"] is None else c["max"]["index"], None if c["min"] is None else c["min"]["index"])
        for c in res.evidence["components"]
    ]


def test_basic_max_min_and_evidence():
    res, idx = _run([1.0, 5.0, -2.0, 3.0])
    assert res.status == "pass"
    assert idx == [1, 2]
    (comp,) = res.evidence["components"]
    assert comp == {
        "component": 0,
        "max": {"index": 1, "t": 0.5, "value": 5.0},
        "min": {"index": 2, "t": 1.0, "value": -2.0},
    }
    json.dumps(res.evidence)


def test_ties_take_lowest_index():
    res, idx = _run([2.0, 7.0, 7.0, 0.0, 0.0, 7.0])
    comp = res.evidence["components"][0]
    assert comp["max"]["index"] == 1
    assert comp["min"]["index"] == 3
    assert idx == [1, 3]


def test_nan_and_inf_are_ignored():
    res, idx = _run([np.nan, np.inf, 4.0, np.nan, np.nan, -np.inf, 1.0, np.nan])
    comp = res.evidence["components"][0]
    assert (comp["max"]["index"], comp["min"]["index"]) == (2, 6)
    assert res.status == "pass"
    assert idx == [2, 6]


def test_all_nan_is_not_applicable():
    res, idx = _run([np.nan, np.nan, np.nan])
    assert res.status == "not_applicable"
    assert idx == []
    assert res.evidence["components"] == [{"component": 0, "max": None, "min": None}]
    assert res.notes == ["no finite sample"]


def test_constant_passes_with_first_sample():
    res, idx = _run([4.0, 4.0, 4.0])
    assert res.status == "pass"
    comp = res.evidence["components"][0]
    assert comp["max"]["index"] == comp["min"]["index"] == 0
    assert idx == [0]


def test_constant_after_nan_uses_first_finite():
    res, idx = _run([np.nan, 4.0, 4.0])
    assert idx == [1]


def test_single_sample():
    res, idx = _run([9.5], t=[3.0])
    assert res.status == "pass"
    assert idx == [0]
    assert res.evidence["components"][0]["max"] == {"index": 0, "t": 3.0, "value": 9.5}


def test_empty_signal():
    res, idx = _run(np.empty(0), t=np.empty(0))
    assert res.status == "not_applicable"
    assert idx == []
    assert res.samples.is_empty
    assert res.notes == ["empty signal"]


def test_vector_uses_only_fully_finite_samples():
    # Samples 0 and 2 each have a NaN component, so only samples 1 and 3 are finite.
    v = np.array(
        [
            [1.0, np.nan, 0.0],
            [3.0, 2.0, 0.0],
            [np.nan, -1.0, 0.0],
            [-4.0, 8.0, 0.0],
        ]
    )
    res, idx = _run(v, kind="vector")
    assert _pairs(res) == [(1, 3), (3, 1), (1, 1)]
    assert idx == [1, 3]
    assert res.status == "pass"
    assert (_pairs(res), set(idx)) == global_extrema_loop(v)


def test_vector_nan_component_means_no_finite_sample():
    v = np.array([[1.0, np.nan], [5.0, np.nan], [2.0, np.nan]])
    res, idx = _run(v, kind="vector")
    assert res.status == "not_applicable"
    assert res.evidence["components"] == [
        {"component": 0, "max": None, "min": None},
        {"component": 1, "max": None, "min": None},
    ]
    assert idx == []


def test_regression_partial_nan_sample_is_not_a_max_candidate():
    # Per-component NaN skipping would pick index 0 (value 5) as component 0's max; sample 0 is not finite.
    v = np.array([[5.0, np.nan], [1.0, 1.0], [3.0, 0.0]])
    res, idx = _run(v, t=[0.1, 0.2, 0.3], kind="vector")
    comps = res.evidence["components"]
    assert comps[0]["max"] == {"index": 2, "t": 0.3, "value": 3.0}
    assert comps[0]["min"] == {"index": 1, "t": 0.2, "value": 1.0}
    assert comps[1]["max"]["index"] == 1 and comps[1]["min"]["index"] == 2
    assert idx == [1, 2]
    assert res.status == "pass"


def test_regression_staggered_nans_leave_no_finite_sample():
    v = np.array([[1.0, np.nan], [np.nan, 2.0]])
    res, idx = _run(v, t=[0.0, 1.0], kind="vector")
    assert res.status == "not_applicable"
    assert idx == []
    assert _pairs(res) == [(None, None), (None, None)]


def test_regression_infinite_component_excludes_sample():
    v = np.array([[9.0, np.inf], [1.0, 0.0], [2.0, -np.inf], [0.5, 3.0]])
    res, idx = _run(v, kind="vector")
    assert _pairs(res) == [(1, 3), (3, 1)]
    assert idx == [1, 3]


def test_integer_and_bool_values():
    res, idx = _run(np.array([3, 9, -2, 9, -2], dtype=np.int64))
    comp = res.evidence["components"][0]
    assert (comp["max"]["value"], comp["min"]["value"]) == (9, -2)
    assert isinstance(comp["max"]["value"], int)
    assert idx == [1, 2]
    res, idx = _run(np.array([False, True, True, False]))
    assert idx == [0, 1]
    assert res.evidence["components"][0]["max"]["value"] is True


def test_large_int64_values_are_exact():
    big = 2**62
    res, idx = _run(np.array([big, big + 1, big - 1], dtype=np.int64))
    assert idx == [1, 2]


def test_float32_values():
    res, idx = _run(np.array([np.nan, 1.5, -3.25, 1.5], dtype=np.float32))
    comp = res.evidence["components"][0]
    assert comp["max"] == {"index": 1, "t": 0.5, "value": 1.5}
    assert comp["min"]["value"] == -3.25
    assert idx == [1, 2]


def test_signed_zero_ties_lowest_index():
    res, idx = _run([0.0, -0.0, -1.0, 0.0])
    assert res.evidence["components"][0]["max"]["index"] == 0
    assert math.copysign(1.0, res.evidence["components"][0]["max"]["value"]) == 1.0


def test_component_extrema_helper():
    assert global_extrema.component_extrema(np.array([np.nan])) == (None, None)
    assert global_extrema.component_extrema(np.empty(0)) == (None, None)
    assert global_extrema.component_extrema(np.array([np.nan, 2.0, 2.0])) == (1, 1)
    x = np.array([7.0, 1.0, 4.0, -2.0])
    assert global_extrema.component_extrema(x, np.array([False, True, True, False])) == (2, 1)
    assert global_extrema.component_extrema(x, np.zeros(4, dtype=bool)) == (None, None)
    big = np.array([2**62 + 1, 2**62, 2**62 + 2], dtype=np.int64)
    assert global_extrema.component_extrema(big, np.array([True, True, False])) == (0, 1)


@pytest.mark.parametrize("bit", [0, 17, 63])
def test_roles_use_given_bit(bit):
    sig = _sig([1.0, 2.0])
    res = global_extrema.evaluate(sig, {}, {"main": bit})
    _, roles = res.samples.materialize()
    assert set(roles.tolist()) == {1 << bit}
