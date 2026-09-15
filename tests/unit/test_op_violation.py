"""Hand-built cases for the violation operator (docs/contracts.md, violation)."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.ops._common import STATUS_NOT_APPLICABLE, STATUS_PASS
from baslt.ops.violation import GAP_END, GAP_START, OPEN_END, OPEN_START, detect_runs, evaluate, flag_names
from baslt.signals import Signal

NAN = float("nan")
BITS = {"edge": 2, "worst": 4}


def _sig(t, v):
    v = np.asarray(v, dtype=np.float64)
    kind = "vector" if v.ndim == 2 else "continuous"
    return Signal("s", "s", np.asarray(t, dtype=np.float64), v, kind, None, None, v.dtype.str)


def _roles(result):
    edge = result.samples.indices_with(1 << BITS["edge"]).tolist()
    worst = result.samples.indices_with(1 << BITS["worst"]).tolist()
    return edge, worst


def test_run_above_with_interpolated_boundaries():
    t = [0, 1, 2, 3, 4, 5, 6]
    x = [0, 2, 3, 2, 0, 0, 0]
    r = detect_runs(t, x, above=1.0)
    assert r.start.tolist() == [0.5]
    assert r.end.tolist() == [3.5]
    assert r.index_start.tolist() == [1] and r.index_end.tolist() == [3]
    assert r.worst.tolist() == [2]
    assert r.flags.tolist() == [0]

    res = evaluate(_sig(t, x), {"above": 1.0, "below": None, "min_duration": 0.0}, BITS)
    assert res.status == STATUS_PASS
    assert _roles(res) == ([0, 1, 3, 4], [2])
    assert res.samples.bits_present() == (1 << 2) | (1 << 4)
    assert res.evidence == {
        "count": 1,
        "total_duration": 3.0,
        "runs": [{"start": 0.5, "end": 3.5, "worst_t": 2.0, "worst_value": 3.0, "component": 0, "flags": []}],
    }


def test_run_below_and_worst_lowest_index_on_ties():
    t = [0, 1, 2, 3, 4]
    x = [0, -2, -3, -3, 0]
    r = detect_runs(t, x, below=-1.0)
    assert r.start.tolist() == [0.5]
    assert r.end.tolist() == [3 + 2 / 3]
    assert r.worst.tolist() == [2]
    r = detect_runs(t, [0, 3, 3, 1.5, 0], above=1.0)
    assert r.worst.tolist() == [1]


def test_value_equal_to_limit_is_not_violating():
    assert detect_runs([0, 1, 2], [1, 1, 1], above=1.0).count == 0
    assert detect_runs([0, 1, 2], [1, 1, 1], below=1.0).count == 0
    # Neighbours exactly at the limit: the crossing times are the neighbours' times.
    r = detect_runs([0, 1, 2], [1, 2, 1], above=1.0)
    assert (r.start.tolist(), r.end.tolist()) == ([0.0], [2.0])
    assert r.flags.tolist() == [0]


def test_open_start_and_open_end():
    t = [0, 1, 2, 3, 4]
    x = [2, 2, 0, 2, 2]
    r = detect_runs(t, x, above=1.0)
    assert r.start.tolist() == [0.0, 2.5]
    assert r.end.tolist() == [1.5, 4.0]
    assert r.flags.tolist() == [OPEN_START, OPEN_END]
    res = evaluate(_sig(t, x), {"above": 1.0}, BITS)
    assert [run["flags"] for run in res.evidence["runs"]] == [["open_start"], ["open_end"]]
    assert _roles(res) == ([0, 1, 2, 3, 4], [0, 3])


def test_gap_start_and_gap_end():
    t = [0, 1, 2, 3, 4]
    x = [0, 2, NAN, 2, 0]
    r = detect_runs(t, x, above=1.0)
    assert r.start.tolist() == [0.5, 3.0]
    assert r.end.tolist() == [1.0, 3.5]
    assert r.flags.tolist() == [GAP_END, GAP_START]
    res = evaluate(_sig(t, x), {"above": 1.0}, BITS)
    # Boundary pairs include the non-finite neighbour.
    assert _roles(res) == ([0, 1, 2, 3, 4], [1, 3])
    assert [run["flags"] for run in res.evidence["runs"]] == [["gap_end"], ["gap_start"]]


def test_infinite_samples_end_runs():
    r = detect_runs([0, 1, 2, 3], [2, np.inf, 2, -np.inf], above=1.0)
    assert r.flags.tolist() == [OPEN_START | GAP_END, GAP_START | GAP_END]


def test_min_duration_inclusive():
    t = list(range(8))
    x = [0, 2, 0, 0, 2, 2, 2, 0]
    r = detect_runs(t, x, above=1.0, min_duration=1.0)
    assert r.start.tolist() == [0.5, 3.5]
    assert (r.end - r.start).tolist() == [1.0, 3.0]
    r = detect_runs(t, x, above=1.0, min_duration=1.5)
    assert r.start.tolist() == [3.5]
    assert r.n_candidates == 2
    res = evaluate(_sig(t, x), {"above": 1.0, "min_duration": 1.5}, BITS)
    assert res.evidence["count"] == 1 and res.evidence["total_duration"] == 3.0
    # Boundary pairs of both runs, including the one min_duration rejected; `worst` only on the kept run.
    assert _roles(res) == ([0, 1, 2, 3, 4, 6, 7], [4])
    assert res.raw[0]["runs"].count == 2 and res.raw[0]["kept"].tolist() == [False, True]


def test_boundary_pairs_of_runs_rejected_by_min_duration_are_retained():
    # Without sample 1 the reconstruction stretches the 0.5 s run at sample 2 back to sample 0 and reports it as a
    # 5 s violation the source never had (docs/contracts.md, violation: "detection on the reconstruction yields
    # the same runs"). The mandatory `gap` retention alone retains samples 2, 3 and 4 here.
    t = [0, 9, 10, 11, 12, 13, 14, 15]
    x = [0, 0, 2, NAN, 0, 5, 5, 0]
    res = evaluate(_sig(t, x), {"above": 1.0, "min_duration": 2.0}, BITS)
    assert res.evidence["count"] == 1
    assert res.evidence["runs"][0]["start"] == 12.2 and res.evidence["runs"][0]["end"] == 14.8
    assert _roles(res) == ([1, 2, 3, 4, 5, 6, 7], [5])


def test_single_sample_signal_and_run():
    r = detect_runs([5.0], [3.0], above=1.0)
    assert r.start.tolist() == [5.0] and r.end.tolist() == [5.0]
    assert flag_names(int(r.flags[0])) == ["open_start", "open_end"]
    assert detect_runs([5.0], [3.0], above=1.0, min_duration=0.1).count == 0
    res = evaluate(_sig([5.0], [3.0]), {"above": 1.0}, BITS)
    assert _roles(res) == ([0], [0])


def test_duplicate_timestamps():
    with np.errstate(all="raise"):
        r = detect_runs([0, 1, 1, 2], [0, 0, 2, 2], above=1.0)
        assert r.start.tolist() == [1.0] and r.end.tolist() == [2.0]
        r = detect_runs([0, 1, 1, 2], [0, 2, 0, 0], above=1.0)
        assert r.start.tolist() == [0.5] and r.end.tolist() == [1.0]


def test_constant_and_empty_signals():
    res = evaluate(_sig([0, 1, 2], [0, 0, 0]), {"above": 1.0}, BITS)
    assert res.status == STATUS_PASS
    assert res.evidence == {"count": 0, "total_duration": 0.0, "runs": []}
    assert res.samples.is_empty
    res = evaluate(_sig([0, 1], [NAN, NAN]), {"above": 1.0}, BITS)
    assert res.status == STATUS_NOT_APPLICABLE and res.samples.is_empty
    res = evaluate(_sig([], []), {"below": 1.0}, BITS)
    assert res.status == STATUS_NOT_APPLICABLE
    assert detect_runs([], [], above=0.0).count == 0


def test_vector_per_component():
    t = [0, 1, 2]
    v = [[0, 0], [2, -2], [0, 0]]
    res = evaluate(_sig(t, v), {"above": 1.0}, BITS)
    assert res.evidence["count"] == 1
    run = res.evidence["runs"][0]
    assert (run["start"], run["end"], run["component"], run["worst_value"]) == (0.5, 1.5, 0, 2.0)
    res = evaluate(_sig(t, v), {"below": -1.0}, BITS)
    assert res.evidence["runs"][0]["component"] == 1
    assert res.evidence["runs"][0]["worst_value"] == -2.0
    assert _roles(res) == ([0, 1, 2], [1])


def test_vector_non_finite_component_ends_runs_in_every_component():
    t = [0, 1, 2, 3]
    v = [[2, 2], [2, NAN], [2, 2], [0, 0]]
    res = evaluate(_sig(t, v), {"above": 1.0}, BITS)
    runs = [(r["component"], r["start"], r["end"], r["flags"]) for r in res.evidence["runs"]]
    assert runs == [
        (0, 0.0, 0.0, ["open_start", "gap_end"]),
        (1, 0.0, 0.0, ["open_start", "gap_end"]),
        (0, 2.0, 2.5, ["gap_start"]),
        (1, 2.0, 2.5, ["gap_start"]),
    ]


def test_evidence_capped_counts_exact():
    n = 1000
    x = np.tile([0.0, 2.0], n // 2)
    res = evaluate(_sig(np.arange(n, dtype=float), x), {"above": 1.0}, BITS)
    assert res.evidence["count"] == n // 2
    assert len(res.evidence["runs"]) == 256
    assert res.evidence["total_duration"] == pytest.approx(n // 2 - 0.5)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        detect_runs([0, 1], [0, 1])
    with pytest.raises(ValueError):
        detect_runs([0, 1], [0, 1], above=1.0, below=0.0)
    with pytest.raises(ValueError):
        detect_runs([0, 1], [0, 1], above=NAN)
    with pytest.raises(ValueError):
        detect_runs([0, 1], [0, 1], above=1.0, min_duration=-1)
    with pytest.raises(ValueError):
        evaluate(_sig([0, 1], [0, 1]), {"above": None, "below": None}, BITS)
