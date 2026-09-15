"""Unit tests for the state_transitions operator."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from baslt.ops import state_transitions
from baslt.signals import Signal, normalize_signal
from reference.extrema_loop import json_values_equal, state_transitions_loop

BITS = {"main": 2}


def _nan64(payload: int) -> np.float64:
    return np.frombuffer(struct.pack("<Q", 0x7FF8000000000000 | payload), dtype="<f8")[0]


def _nan32(payload: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", 0x7FC00000 | payload), dtype="<f4")[0]


def _sig(v, kind=None):
    v = np.asarray(v)
    sig, _ = normalize_signal("mode", np.arange(v.shape[0], dtype=np.float64), v, kind=kind)
    return sig


def _run(v, kind=None):
    res = state_transitions.evaluate(_sig(v, kind), {}, BITS)
    idx, roles = res.samples.materialize()
    assert (roles == np.uint64(1 << BITS["main"])).all()
    return res, idx.tolist()


def _hold_bitwise_equals_source(v, idx):
    """Sample-and-hold over retained indices reproduces the source bits at every sample."""
    v = np.ascontiguousarray(v)
    pos = np.searchsorted(np.asarray(idx), np.arange(v.shape[0]), side="right") - 1
    held = np.ascontiguousarray(v[np.asarray(idx)[pos]])
    return held.tobytes() == v.tobytes()


def test_integer_modes():
    v = np.array([0, 0, 1, 1, 1, 2, 0, 0], dtype=np.int32)
    res, idx = _run(v)
    assert idx == [0, 2, 5, 6, 7]
    assert res.status == "pass"
    assert res.evidence == {"transitions": 3, "distinct_count": 3, "distinct_values": [0, 1, 2]}
    assert all(isinstance(x, int) for x in res.evidence["distinct_values"])
    assert _hold_bitwise_equals_source(v, idx)
    json.dumps(res.evidence)


def test_bool_values():
    v = np.array([False, False, True, False, False])
    res, idx = _run(v)
    assert idx == [0, 2, 3, 4]
    assert res.evidence == {"transitions": 2, "distinct_count": 2, "distinct_values": [False, True]}


def test_float_with_nan_runs():
    v = np.array([1.0, np.nan, np.nan, np.nan, 1.0, 1.0, np.nan, 2.0])
    res, idx = _run(v, kind="discrete")
    assert idx == [0, 1, 4, 6, 7]
    assert res.evidence["transitions"] == 4
    assert res.evidence["distinct_count"] == 3
    dv = res.evidence["distinct_values"]
    assert dv[:2] == [1.0, 2.0] and len(dv) == 3 and math.isnan(dv[2])
    assert _hold_bitwise_equals_source(v, idx)


def test_all_nan_passes():
    res, idx = _run(np.full(4, np.nan))
    assert res.status == "pass"
    assert idx == [0, 3]
    assert res.evidence["transitions"] == 0
    assert res.evidence["distinct_count"] == 1


def test_constant():
    res, idx = _run(np.full(6, 3, dtype=np.int64))
    assert idx == [0, 5]
    assert res.evidence == {"transitions": 0, "distinct_count": 1, "distinct_values": [3]}


def test_change_at_last_sample():
    res, idx = _run(np.array([1, 1, 1, 2]))
    assert idx == [0, 3]
    assert res.evidence["transitions"] == 1


def test_single_sample():
    res, idx = _run(np.array([7]))
    assert idx == [0]
    assert res.status == "pass"
    assert res.evidence == {"transitions": 0, "distinct_count": 1, "distinct_values": [7]}


def test_empty_signal():
    res, idx = _run(np.empty(0, dtype=np.int32))
    assert res.status == "not_applicable"
    assert idx == []
    assert res.evidence == {"transitions": 0, "distinct_count": 0, "distinct_values": []}


def test_regression_signed_zero_is_a_transition():
    v = np.array([0.0, -0.0, 0.0])
    res, idx = _run(v, kind="discrete")
    assert idx == [0, 1, 2]
    assert res.evidence["transitions"] == 2
    assert res.evidence["distinct_count"] == 2
    dv = res.evidence["distinct_values"]
    assert [math.copysign(1.0, x) for x in dv] == [-1.0, 1.0]
    assert _hold_bitwise_equals_source(v, idx)


def test_regression_signed_zero_hold_is_bitwise():
    v = np.array([0.0, -0.0, -0.0, 1.0])
    res, idx = _run(v, kind="discrete")
    assert idx == [0, 1, 3]
    assert res.evidence["transitions"] == 2
    assert _hold_bitwise_equals_source(v, idx)
    ref = state_transitions_loop(v)
    assert set(idx) == ref["retained"]
    assert json_values_equal(res.evidence["distinct_values"], ref["distinct_values"])


def test_regression_nan_payload_change_is_retained_but_not_counted():
    a, b = _nan64(1), _nan64(2)
    v = np.array([a, b, b, 1.0])
    res, idx = _run(v, kind="discrete")
    # NaN equals NaN, so index 1 is not a transition, but it is retained for a bitwise hold.
    assert idx == [0, 1, 3]
    assert res.evidence["transitions"] == 1
    assert res.evidence["distinct_count"] == 2
    assert _hold_bitwise_equals_source(v, idx)
    ref = state_transitions_loop(v)
    assert set(idx) == ref["retained"] and ref["transitions"] == 1


def test_regression_float32_signed_zero_and_nan_payload():
    v = np.array([np.float32(0.0), np.float32(-0.0), _nan32(3), _nan32(5), _nan32(5), np.float32(2.5)], dtype="<f4")
    res, idx = _run(v, kind="discrete")
    assert idx == [0, 1, 2, 3, 5]
    assert res.evidence["transitions"] == 3
    assert res.evidence["distinct_count"] == 4
    assert _hold_bitwise_equals_source(v, idx)
    ref = state_transitions_loop(v)
    assert set(idx) == ref["retained"]
    assert json_values_equal(res.evidence["distinct_values"], ref["distinct_values"])


def test_regression_big_endian_floats():
    v = np.array([1.0, -0.0, 0.0, 0.0, 1.0], dtype=">f8")
    sig = Signal("m", "m", np.arange(5.0), v, "discrete", None, None, ">f8")
    res = state_transitions.evaluate(sig, {}, BITS)
    idx = res.samples.materialize()[0].tolist()
    assert idx == [0, 1, 2, 4]
    assert res.evidence["transitions"] == 3
    assert _hold_bitwise_equals_source(v, idx)


def test_distinct_values_capped():
    v = np.arange(1000, dtype=np.int64)
    res, idx = _run(v)
    assert res.evidence["transitions"] == 999
    assert len(res.evidence["distinct_values"]) == 256
    assert res.evidence["distinct_count"] == 1000
    assert len(idx) == 1000


def test_regression_distinct_count_is_exact_past_the_cap():
    v = np.arange(300, dtype=np.int32)
    res, _ = _run(v, kind="discrete")
    assert res.evidence["distinct_count"] == 300
    assert res.evidence["distinct_values"] == list(range(256))


def test_distinct_values_order_and_infinities():
    v = np.array([np.inf, 0.0, np.nan, -np.inf, -0.0, 3.0, _nan64(9)])
    res, _ = _run(v, kind="discrete")
    dv = res.evidence["distinct_values"]
    assert res.evidence["distinct_count"] == 6
    assert dv[0] == -np.inf and dv[3] == 3.0 and dv[4] == np.inf and math.isnan(dv[5])
    assert math.copysign(1.0, dv[1]) == -1.0 and math.copysign(1.0, dv[2]) == 1.0


def test_vector_rejected():
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        state_transitions.evaluate(_sig(v, kind="vector"), {}, BITS)


def test_two_dimensional_values_rejected_even_if_mislabelled():
    sig = Signal("x", "x", np.arange(3.0), np.zeros((3, 2)), "discrete", None, None, "<f8")
    with pytest.raises(ValueError):
        state_transitions.evaluate(sig, {}, BITS)


def test_change_mask_helper():
    m = state_transitions.change_mask(np.array([np.nan, np.nan, 1.0, np.inf, np.inf, -0.0, 0.0]))
    assert m.tolist() == [False, True, True, False, True, True]
    value_changed, bits_changed = state_transitions.change_masks(np.array([_nan64(1), _nan64(2), 1.0]))
    assert value_changed.tolist() == [False, True]
    assert bits_changed.tolist() == [True, True]
