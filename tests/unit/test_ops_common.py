from __future__ import annotations

import numpy as np

from baslt.ops._common import (
    EVIDENCE_LIMIT,
    capped,
    components,
    extent_samples,
    finite_mask,
    fnum,
    gap_samples,
    value_of,
)


def test_finite_mask_scalar_vector_and_integer():
    assert finite_mask(np.array([1.0, np.nan, np.inf])).tolist() == [True, False, False]
    assert finite_mask(np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, -np.inf]])).tolist() == [True, False, False]
    assert finite_mask(np.array([1, 2, 3], dtype=np.int16)).tolist() == [True, True, True]
    assert finite_mask(np.array([True, False])).tolist() == [True, True]


def test_extent_samples():
    assert extent_samples(0, 0).is_empty
    assert extent_samples(1, 0).materialize()[0].tolist() == [0]
    assert extent_samples(6, 2).materialize()[0].tolist() == [0, 5]
    assert int(extent_samples(6, 2).roles[0]) == 4


def test_gap_samples_bracket_every_nan_run():
    v = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0, np.nan, 8.0])
    idx = gap_samples(v, 1).materialize()[0].tolist()
    # run [2,4): neighbours 1 and 4, run edges 2 and 3; run [6,7): neighbours 5 and 7, edge 6
    assert idx == [1, 2, 3, 4, 5, 6, 7]


def test_gap_samples_at_signal_edges_and_clean_signal():
    assert gap_samples(np.array([np.nan, 1.0, 2.0]), 0).materialize()[0].tolist() == [0, 1]
    assert gap_samples(np.array([1.0, 2.0, np.nan]), 0).materialize()[0].tolist() == [1, 2]
    assert gap_samples(np.array([1.0, 2.0, 3.0]), 0).is_empty
    assert gap_samples(np.empty(0), 0).is_empty


def test_gap_samples_vector_component_nan():
    v = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 3.0]])
    assert gap_samples(v, 0).materialize()[0].tolist() == [0, 1, 2]


def test_json_helpers():
    assert fnum(np.float32(1.5)) == 1.5 and isinstance(fnum(np.float32(1.5)), float)
    assert fnum(np.int64(3)) == 3 and isinstance(fnum(np.int64(3)), int)
    assert fnum(np.bool_(True)) is True
    assert value_of(np.array([[1.0, 2.0]]), 0) == [1.0, 2.0]
    assert value_of(np.array([7], dtype=np.int32), 0) == 7
    assert len(capped(list(range(1000)))) == EVIDENCE_LIMIT
    comps = list(components(np.zeros((4, 3))))
    assert [k for k, _ in comps] == [0, 1, 2] and comps[0][1].shape == (4,)
    assert [k for k, _ in components(np.zeros(4))] == [0]
