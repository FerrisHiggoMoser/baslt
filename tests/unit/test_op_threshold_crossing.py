"""Hand-built cases for the threshold_crossing operator (docs/contracts.md, threshold_crossing)."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.ops._common import EVIDENCE_LIMIT, STATUS_NOT_APPLICABLE, STATUS_PASS, STATUS_WARN
from baslt.ops.threshold_crossing import FALLING, RISING, detect, evaluate, level_time
from baslt.signals import Signal

NAN = float("nan")


def _sig(t, v, kind=None):
    v = np.asarray(v, dtype=np.float64)
    kind = kind or ("vector" if v.ndim == 2 else "continuous")
    return Signal("s", "s", np.asarray(t, dtype=np.float64), v, kind, None, None, v.dtype.str)


def _params(value, **kw):
    p = {"value": value, "edge": "both", "hysteresis": 0.0, "debounce": 0.0, "tolerance": 0.0,
         "interpolate": "linear"}
    p.update(kw)
    return p


def _indices(result, bit=3):
    return result.samples.indices_with(1 << bit).tolist()


# ---------------------------------------------------------------------------------------------- step 1


def test_rising_and_falling_linear_times():
    t = [0, 1, 2, 3, 4]
    x = [0, 2, 2, 0, 0]
    d = detect(t, x, 1.0)
    assert d.t.tolist() == [0.5, 2.5]
    assert d.edge.tolist() == [RISING, FALLING]
    assert d.index_before.tolist() == [0, 2]
    assert d.index_after.tolist() == [1, 3]
    assert d.confirm.tolist() == [1, 3]
    assert d.gap.tolist() == [False, False]
    assert (d.n_level_crossings, d.n_flips, d.pending_at_end) == (2, 2, False)
    assert d.initial_state is False


def test_sample_exactly_on_level():
    # Rising needs x[i] < V <= x[i+1]; falling needs x[i] >= V > x[i+1].
    d = detect([0, 1, 2], [0, 1, 0], 1.0)
    assert d.t.tolist() == [1.0, 1.0]
    assert d.edge.tolist() == [RISING, FALLING]
    # Touching the level from above is not a crossing.
    d = detect([0, 1, 2], [2, 1, 2], 1.0)
    assert d.count == 0 and d.n_level_crossings == 0
    assert d.initial_state is True


def test_plateau_on_level():
    d = detect([0, 1, 2, 3, 4], [0, 1, 1, 1, 0], 1.0)
    assert d.t.tolist() == [1.0, 3.0]
    assert d.index_before.tolist() == [0, 3]
    assert d.edge.tolist() == [RISING, FALLING]


def test_edge_filter():
    t, x = [0, 1, 2, 3, 4], [0, 2, 2, 0, 0]
    assert detect(t, x, 1.0, edge="rising").t.tolist() == [0.5]
    assert detect(t, x, 1.0, edge="falling").t.tolist() == [2.5]
    assert detect(t, x, 1.0, edge="falling").n_flips == 2


def test_interpolate_none_reports_sample_after_and_discretization_error():
    t, x = [0, 1, 2, 3, 4], [0, 2, 2, 0, 0]
    d = detect(t, x, 1.0, interpolate="none")
    assert d.t.tolist() == [1.0, 3.0]
    assert d.discretization_error.tolist() == [1.0, 1.0]
    assert detect(t, x, 1.0).discretization_error.tolist() == [0.0, 0.0]

    r = evaluate(_sig([0, 1, 2, 4, 5], x), _params(1.0, interpolate="none"), {"main": 3})
    assert [c["t"] for c in r.evidence["crossings"]] == [1.0, 4.0]
    assert [c["discretization_error"] for c in r.evidence["crossings"]] == [1.0, 2.0]
    assert r.evidence["max_discretization_error"] == 2.0
    assert "max_discretization_error" not in evaluate(_sig(t, x), _params(1.0), {"main": 3}).evidence


# ---------------------------------------------------------------------------------------------- gaps


def test_crossing_spanning_nan_gap_uses_bracketing_finite_samples():
    t = [0, 1, 2, 3]
    x = [0, NAN, NAN, 3]
    d = detect(t, x, 1.0)
    assert d.t.tolist() == [1.0]
    assert d.index_before.tolist() == [0]
    assert d.index_after.tolist() == [3]
    assert d.confirm.tolist() == [3]
    assert d.gap.tolist() == [True]
    assert d.gap_flips == 1

    r = evaluate(_sig(t, x), _params(1.0), {"main": 3})
    assert r.status == STATUS_WARN
    assert r.evidence["gap_flips"] == 1
    assert _indices(r) == [0, 3]
    assert r.notes


def test_evidence_gap_flips_counts_flips_before_debounce_and_the_edge_filter():
    # "count of gap flips" is the step-3 term: a gap flip still counts when no crossing is reported for it.
    t = [float(k) for k in range(11)]
    x = [0, NAN, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    r = evaluate(_sig(t, x), _params(1.0, debounce=2.0), {"main": 0})
    assert r.evidence["count"] == 0 and r.evidence["gap_flips"] == 1
    assert r.status == STATUS_WARN and r.notes

    r = evaluate(_sig([0, 1, 2, 3, 4, 5], [0, NAN, 2, 2, 0, 0]), _params(1.0, edge="falling"), {"main": 0})
    assert r.evidence["count"] == 1 and r.evidence["rising"] == 0
    assert r.evidence["gap_flips"] == 1  # the rising flip that the edge filter drops spans the gap
    assert r.status == STATUS_WARN

    # Detection.gap_flips is the same count.
    assert detect(t, x, 1.0, debounce=2.0).gap_flips == 1
    assert detect(t, x, 1.0, debounce=2.0).count == 0


def test_non_finite_samples_hold_the_state():
    # Hysteresis 1 -> band (0.5, 1.5). NaN and in-band samples hold; sample 3 confirms the rise.
    d = detect([0, 1, 2, 3], [0, 1.25, NAN, 2], 1.0, hysteresis=1.0)
    assert d.t.tolist() == [0.8]
    assert d.index_before.tolist() == [0]
    assert d.confirm.tolist() == [3]
    assert d.gap.tolist() == [False]


def test_leading_and_trailing_non_finite_samples():
    d = detect([0, 1, 2, 3, 4], [NAN, 0, 2, NAN, np.inf], 1.0)
    assert d.t.tolist() == [1.5]
    assert d.initial_state is False


# ---------------------------------------------------------------------------------------------- hysteresis


def test_chatter_inside_hysteresis_band():
    t = [0, 1, 2, 3, 4, 5, 6, 7]
    x = [0, 1.25, 0.75, 1.25, 0.75, 1.25, 2, 2]
    d = detect(t, x, 1.0, hysteresis=1.0)
    assert d.n_level_crossings == 5
    assert d.n_flips == 1
    # The flip at j=6 takes the last rising level crossing with i+1 <= 6: bracket (4, 5).
    assert d.t.tolist() == [4.5]
    assert d.index_before.tolist() == [4]
    assert d.confirm.tolist() == [6]

    r = evaluate(_sig(t, x), _params(1.0, hysteresis=1.0), {"main": 3})
    assert _indices(r) == [4, 5, 6]
    assert r.status == STATUS_PASS

    # Without hysteresis every level crossing is a flip.
    assert detect(t, x, 1.0).count == 5
    r0 = evaluate(_sig(t, x), _params(1.0), {"main": 3})
    assert _indices(r0) == [0, 1, 2, 3, 4, 5]


def test_hysteresis_initial_state_uses_level_not_band():
    # First finite sample 1.25 is inside the band but >= V, so the initial state is high.
    d = detect([0, 1, 2, 3], [1.25, 0.75, 0.25, 0.25], 1.0, hysteresis=1.0)
    assert d.initial_state is True
    assert d.edge.tolist() == [FALLING]
    assert d.t.tolist() == [0.5]
    assert d.confirm.tolist() == [2]


def test_band_rounding_onto_level_never_marks_low_at_level():
    # V = -1 with a tiny H: hi rounds above V but lo rounds onto V. A sample at exactly V stays high.
    value = -1.0
    h = 1.5 * 2.0**-53
    assert value + h / 2 > value and value - h / 2 == value
    d = detect([0, 1, 2, 3], [-2, 0, -1, -1], value, hysteresis=h)
    assert d.edge.tolist() == [RISING]
    assert d.n_flips == 1


def test_band_rounding_onto_level_keeps_step_3_well_defined():
    # Step 3 timestamps a flip with "the last level crossing in the same direction", so every flip must have one.
    # Reading step 2's low mark as the bare `x <= lo` breaks that when the band rounds onto V: a sample at exactly
    # V would mark low although step 1 sees no falling crossing anywhere (falling needs V > x[i+1]). The extra
    # `x < V` condition is what keeps the two steps consistent; it only ever applies to a sub-ulp band.
    value = -1.0
    h = 1.5 * 2.0**-53
    for t, x in (([0, 1, 2, 3], [-2, 0, -1, -1]), ([0, 1, 2, 3, 4], [-2, 0, -1.5, 0, -1])):
        x = np.asarray(x, dtype=np.float64)
        d = detect(t, x, value, hysteresis=h)
        f = d.flips
        assert f.t.shape[0] == d.n_flips
        for e, i, j in zip(f.edge.tolist(), f.index_before.tolist(), f.index_after.tolist()):
            if e == RISING:
                assert x[i] < value <= x[j]
            else:
                assert x[i] >= value > x[j]
    # The first signal has no falling level crossing at all, so it may not produce a falling flip.
    d = detect([0, 1, 2, 3], [-2, 0, -1, -1], value, hysteresis=h)
    assert d.n_level_crossings == 1 and d.flips.edge.tolist() == [RISING]


def test_infinite_hysteresis_never_flips():
    d = detect([0, 1, 2], [0, 5, -5], 1.0, hysteresis=np.inf)
    assert d.n_level_crossings == 2 and d.n_flips == 0 and d.count == 0


# ---------------------------------------------------------------------------------------------- debounce


def test_debounce_rejects_short_excursion():
    t = list(range(11))
    x = [0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2]
    d = detect(t, x, 1.0, debounce=1.5)
    assert d.t.tolist() == [1.5]
    assert d.edge.tolist() == [RISING]
    assert d.pending_at_end is False
    f = d.flips
    assert f.t.tolist() == [1.5, 4.5, 5.5]
    assert f.duration.tolist() == [3.0, 1.0, 4.5]
    assert f.accepted.tolist() == [True, False, True]
    assert f.transition.tolist() == [True, False, False]
    assert detect(t, x, 1.0).count == 3


def test_debounce_inclusive_bound():
    t = list(range(11))
    x = [0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2]
    assert detect(t, x, 1.0, debounce=1.0).count == 3


def test_transition_timestamped_by_flip_starting_accepted_run():
    t = [0, 1, 2, 3, 4, 5, 6]
    x = [0, 2, 0, 2, 2, 2, 2]
    d = detect(t, x, 1.0, debounce=1.5)
    assert d.t.tolist() == [2.5]
    assert d.index_before.tolist() == [2]


def test_pending_at_end():
    t = [0, 1, 2, 3, 4, 5]
    x = [0, 0, 0, 0, 2, 2]
    d = detect(t, x, 1.0, debounce=2.0)
    assert d.count == 0
    assert d.pending_at_end is True
    r = evaluate(_sig(t, x), _params(1.0, debounce=2.0), {"main": 3})
    assert r.status == STATUS_WARN
    assert r.evidence["pending_at_end"] is True
    assert r.evidence["count"] == 0
    # No crossing is reported, but the pending flip's bracket is retained: without it the reconstruction would
    # place the flip elsewhere and could accept it (docs/contracts.md, threshold_crossing, second clause).
    assert _indices(r) == [3, 4]


def test_pending_at_end_after_accepted_crossing():
    t = list(range(10))
    x = [0, 2, 2, 2, 2, 2, 2, 2, 2, 0]
    d = detect(t, x, 1.0, debounce=2.0)
    assert d.t.tolist() == [0.5]
    assert d.pending_at_end is True


# ---------------------------------------------------------------------------------------------- degenerate


def test_single_sample_and_constant_signals():
    d = detect([0.0], [5.0], 1.0)
    assert d.count == 0 and d.n_level_crossings == 0 and d.initial_state is True
    d = detect([], [], 1.0)
    assert d.count == 0 and d.initial_state is None

    r = evaluate(_sig([0.0], [5.0]), _params(1.0), {"main": 3})
    assert r.status == STATUS_NOT_APPLICABLE and r.samples.is_empty
    assert r.evidence == {"count": 0, "rising": 0, "falling": 0, "crossings": [], "pending_at_end": False,
                          "gap_flips": 0}

    r = evaluate(_sig([0, 1, 2], [NAN, 3, NAN]), _params(1.0), {"main": 3})
    assert r.status == STATUS_NOT_APPLICABLE

    r = evaluate(_sig([0, 1, 2], [2, 2, 2]), _params(1.0), {"main": 3})
    assert r.status == STATUS_PASS and r.evidence["count"] == 0 and r.samples.is_empty

    r = evaluate(_sig([0, 1, 2], [1, 1, 1]), _params(1.0), {"main": 3})
    assert r.status == STATUS_PASS and r.evidence["count"] == 0


def test_duplicate_timestamps_do_not_divide_by_zero():
    with np.errstate(all="raise"):
        d = detect([0, 1, 1, 2], [0, 0, 2, 2], 1.0)
        assert d.t.tolist() == [1.0]
        d = detect([0, 1, 1, 1, 3], [0, 0, 2, 0, 2], 1.0)
        assert d.t.tolist() == [1.0, 1.0, 2.0]
        assert d.flips.duration.tolist() == [0.0, 1.0, 1.0]
        d = detect([0, 1, 1, 1, 3], [0, 0, 2, 0, 2], 1.0, debounce=0.5)
        assert d.t.tolist() == [2.0]
        d = detect([0, 0], [0, 2], 1.0, interpolate="none")
        assert d.t.tolist() == [0.0] and d.discretization_error.tolist() == [0.0]


def test_crossing_time_clamped_into_bracket():
    ta = 3 * 2.0**-53
    tb = 1 + 3 * 2.0**-52
    assert ta + (tb - ta) * 1.0 > tb  # the raw formula lands one ulp past the bracket
    d = detect([ta, tb, 2.0], [0.0, 1.0, 0.0], 1.0)
    assert d.t.tolist() == [tb, tb]
    assert d.flips.duration.tolist() == [0.0, 2.0 - tb]


def test_level_time_survives_value_overflow():
    t = np.array([0.0, 2.0])
    x = np.array([-1e308, 1e308])
    with np.errstate(all="raise"):
        assert level_time(t, x, np.array([0]), np.array([1]), 0.0).tolist() == [1.0]
    assert detect(t, x, 0.0).t.tolist() == [1.0]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        detect([0, 1], [0, 1], 0.5, edge="up")
    with pytest.raises(ValueError):
        detect([0, 1], [0, 1], 0.5, interpolate="cubic")
    with pytest.raises(ValueError):
        detect([0, 1], [0, 1], 0.5, hysteresis=-1)
    with pytest.raises(ValueError):
        detect([0, 1], [0, 1], NAN)
    with pytest.raises(ValueError):
        detect([0, 1], [0, 1, 2], 0.5)
    with pytest.raises(ValueError):
        evaluate(_sig([0.0], [1.0]), _params(0.5, edge="up"), {"main": 0})


# ---------------------------------------------------------------------------------------------- evaluate


def test_evaluate_bits_and_evidence():
    t, x = [0, 1, 2, 3, 4], [0, 2, 2, 0, 0]
    r = evaluate(_sig(t, x), _params(1.0), {"main": 5})
    assert r.samples.bits_present() == 1 << 5
    assert _indices(r, 5) == [0, 1, 2, 3]
    assert r.status == STATUS_PASS
    assert r.evidence == {
        "count": 2,
        "rising": 1,
        "falling": 1,
        "crossings": [
            {"t": 0.5, "edge": "rising", "index_before": 0, "component": 0},
            {"t": 2.5, "edge": "falling", "index_before": 2, "component": 0},
        ],
        "pending_at_end": False,
        "gap_flips": 0,
    }
    assert isinstance(r.evidence["crossings"][0]["index_before"], int)
    assert isinstance(r.evidence["crossings"][0]["t"], float)
    assert len(r.raw) == 1 and r.raw[0].count == 2


def test_evaluate_retains_every_flip_bracket_not_only_the_reported_crossings():
    t, x = [0, 1, 2, 3, 4], [0, 2, 2, 0, 0]
    r = evaluate(_sig(t, x), _params(1.0, edge="falling"), {"main": 0})
    # Only the falling crossing is reported, but dropping the rising flip's bracket would move the run boundaries
    # the reconstruction measures.
    assert _indices(r, 0) == [0, 1, 2, 3]
    assert r.evidence["count"] == 1 and r.evidence["rising"] == 0


def test_evaluate_retains_brackets_of_flips_rejected_by_debounce():
    t = list(range(11))
    x = [0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2]
    r = evaluate(_sig(t, x), _params(1.0, debounce=2.0), {"main": 0})
    assert r.evidence["count"] == 1  # only the rise at 0.5 survives debounce
    # Brackets of all three flips: (0, 1) accepted, (4, 5) and (5, 6) rejected as a 1 s excursion.
    assert _indices(r, 0) == [0, 1, 4, 5, 6]


def test_evaluate_retains_the_confirming_sample_of_every_flip_with_hysteresis():
    t = [0, 1, 2, 3, 4, 5, 6, 7]
    x = [0, 2, 2, 0, 0.75, 2, 2, 2]
    r = evaluate(_sig(t, x), _params(1.0, hysteresis=1.0, edge="rising"), {"main": 0})
    assert r.evidence["count"] == 2 and r.evidence["falling"] == 0
    # Sample 3 confirms the falling flip that the edge filter drops; without it the reconstruction never leaves
    # the high state and reports a single rise.
    assert _indices(r, 0) == [0, 1, 2, 3, 4, 5]


def test_evaluate_vector_per_component():
    t = [0, 1, 2, 3]
    v = [[0, 2], [2, 0], [2, 0], [0, 2]]
    r = evaluate(_sig(t, v), _params(1.0), {"main": 1})
    ev = r.evidence
    assert (ev["count"], ev["rising"], ev["falling"]) == (4, 2, 2)
    assert [(c["t"], c["edge"], c["component"]) for c in ev["crossings"]] == [
        (0.5, "rising", 0),
        (0.5, "falling", 1),
        (2.5, "falling", 0),
        (2.5, "rising", 1),
    ]
    assert _indices(r, 1) == [0, 1, 2, 3]
    assert len(r.raw) == 2


def test_evaluate_vector_non_finite_component_makes_sample_non_finite():
    t = [0, 1, 2]
    v = [[0, 0], [NAN, 5], [2, 0]]
    r = evaluate(_sig(t, v), _params(1.0), {"main": 1})
    assert r.evidence["count"] == 1
    assert r.evidence["crossings"][0]["component"] == 0
    assert r.evidence["gap_flips"] == 1
    assert r.status == STATUS_WARN
    assert _indices(r, 1) == [0, 2]


def test_evidence_list_is_capped_but_counts_are_exact():
    n = 600
    x = np.tile([0.0, 2.0], n // 2)
    r = evaluate(_sig(np.arange(n, dtype=float), x), _params(1.0), {"main": 0})
    assert r.evidence["count"] == n - 1
    assert len(r.evidence["crossings"]) == EVIDENCE_LIMIT
    assert r.samples.count() == n
