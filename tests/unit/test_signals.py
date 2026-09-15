from __future__ import annotations

import numpy as np
import pytest

from baslt.errors import SourceError
from baslt.signals import infer_kind, normalize_signal


def test_float_signal_is_not_copied():
    t = np.linspace(0, 1, 11)
    v = np.sin(t)
    sig, issues = normalize_signal("x", t, v)
    assert issues == []
    assert np.shares_memory(sig.t, t)
    assert np.shares_memory(sig.v, v)
    assert sig.kind == "continuous"
    assert sig.n == 11 and sig.components == 1


def test_column_and_row_vectors_are_flattened():
    t = np.arange(5.0)
    sig, _ = normalize_signal("x", t.reshape(1, 5), np.arange(5.0).reshape(5, 1))
    assert sig.t.shape == (5,) and sig.v.shape == (5,)


def test_vector_signal_transposed_when_needed():
    t = np.arange(4.0)
    v = np.arange(12.0).reshape(3, 4)  # (k, n)
    sig, _ = normalize_signal("pos", t, v)
    assert sig.v.shape == (4, 3)
    assert sig.kind == "vector"
    assert sig.components == 3


def test_non_finite_timestamps_are_dropped():
    t = np.array([0.0, np.nan, 2.0, np.inf, 4.0])
    v = np.arange(5.0)
    sig, issues = normalize_signal("x", t, v)
    assert sig.t.tolist() == [0.0, 2.0, 4.0]
    assert sig.v.tolist() == [0.0, 2.0, 4.0]
    assert len(issues) == 1 and "2 samples" in issues[0]


def test_decreasing_time_raises_or_sorts():
    t = np.array([0.0, 2.0, 1.0, 3.0])
    v = np.array([10.0, 20.0, 15.0, 30.0])
    with pytest.raises(SourceError, match="decreases at sample 2"):
        normalize_signal("x", t, v)
    sig, issues = normalize_signal("x", t, v, on_non_monotonic="sort")
    assert sig.t.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert sig.v.tolist() == [10.0, 15.0, 20.0, 30.0]
    assert issues


def test_duplicate_timestamps_are_kept():
    sig, issues = normalize_signal("x", [0.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0])
    assert sig.n == 4 and issues == []


def test_string_values_become_enum_codes():
    sig, _ = normalize_signal("mode", [0, 1, 2, 3], np.array(["idle", "burn", "burn", "coast"]))
    assert sig.kind == "discrete"
    assert sig.labels == ["burn", "coast", "idle"]
    assert sig.v.dtype == np.int32
    assert [sig.labels[c] for c in sig.v] == ["idle", "burn", "burn", "coast"]
    assert sig.source_dtype == "str"


def test_string_values_with_given_labels():
    sig, _ = normalize_signal("mode", [0, 1], np.array(["b", "a"]), labels=["a", "b", "c"])
    assert sig.v.tolist() == [1, 0]
    with pytest.raises(SourceError, match="not in its labels"):
        normalize_signal("mode", [0, 1], np.array(["b", "z"]), labels=["a", "b"])


def test_time_scale_and_integer_time():
    sig, _ = normalize_signal("x", np.array([0, 1000, 2000]), [1.0, 2.0, 3.0], time_scale=1e-3)
    assert sig.t.dtype == np.float64
    assert sig.t.tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "t, v, message",
    [
        ([0.0, 1.0], [1.0, 2.0, 3.0], "2 timestamps but 3 values"),
        ([0.0, 1.0], np.array([1 + 1j, 2j]), "complex"),
        ([0.0, 1.0], np.zeros((2, 2, 2)), "1-D or 2-D"),
        (np.zeros((2, 2)), [1.0, 2.0], "time must be one-dimensional"),
    ],
)
def test_bad_inputs(t, v, message):
    with pytest.raises(SourceError, match=message):
        normalize_signal("x", t, v)


def test_kind_inference():
    assert infer_kind(np.array([True, False])) == "discrete"
    assert infer_kind(np.array([1, 2], dtype=np.int16)) == "discrete"
    assert infer_kind(np.array([1.0, 2.0])) == "continuous"
    assert infer_kind(np.zeros((3, 3))) == "vector"
    with pytest.raises(SourceError):
        infer_kind(np.zeros((3, 3), dtype=np.int64))


def test_declared_kind_consistency():
    with pytest.raises(SourceError, match="declare kind: vector"):
        normalize_signal("x", [0.0, 1.0], np.zeros((2, 3)), kind="continuous")
    sig, _ = normalize_signal("x", [0.0, 1.0], [1, 2], kind="continuous")
    assert sig.kind == "continuous"


def test_info_describes_signal():
    sig, _ = normalize_signal("pos", np.arange(3.0), np.zeros((3, 3)), unit="m", path="nav/pos")
    info = sig.info()
    assert info.name == "pos" and info.path == "nav/pos" and info.shape == (3, 3)
    assert info.unit == "m" and info.kind == "vector" and info.n == 3
