"""The planner's required layer: role legends, implicit retention, statuses and rejected features."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.errors import PolicyError, UsageError
from baslt.ops import global_extrema, state_transitions, threshold_crossing, violation
from baslt.ops._common import STATUS_NOT_APPLICABLE, STATUS_PASS, STATUS_WARN
from baslt.plan import MAX_ROLES, evaluate_run, legend_of
from baslt.policy import bind_policy, load_policy
from baslt.signals import Run, SourceMeta, normalize_signal

NAN = float("nan")


def make_run(**arrays) -> Run:
    """A run whose signals all share one clock; each value array may be (n,) or (n, k)."""
    signals = {}
    n = max(len(np.asarray(v)) for v in arrays.values())
    t = np.arange(n, dtype=np.float64)
    for name, values in arrays.items():
        sig, _ = normalize_signal(name, t, np.asarray(values), path=f"/{name}")
        signals[name] = sig
    return Run(signals=signals, meta=SourceMeta(path=None, format="numpy", size_bytes=0, issues=[]))


def bind(run: Run, **sections):
    return bind_policy(load_policy({"version": 1, **sections}), run.infos())


def plan_of(run: Run, **sections):
    return evaluate_run(run, bind(run, **sections))


def wave(n: int = 40) -> np.ndarray:
    return np.sin(np.arange(n, dtype=np.float64) / 3.0) * 10.0 + 50.0


# ----- legend ---------------------------------------------------------------------------------


def test_legend_order_extent_gap_then_requirements_in_policy_order():
    run = make_run(q=wave(), mode=np.repeat([0, 1, 2], [10, 15, 15]).astype(np.int32))
    hard = {
        "q": {
            "global_extrema": {},
            "threshold_crossing": [{"value": 55.0}, {"value": 45.0}],
            "violation": [{"above": 58.0}],
            "window_extrema": {"interval": "2 s"},
        },
        "mode": {"state_transitions": {}},
    }
    required = plan_of(run, hard=hard)

    assert [(role.bit, role.id) for role in required.plan("q").legend] == [
        (0, "extent"),
        (1, "gap"),
        (2, "hard.q.global_extrema"),
        (3, "hard.q.threshold_crossing[0]"),
        (4, "hard.q.threshold_crossing[1]"),
        (5, "hard.q.violation[0]#edge"),
        (6, "hard.q.violation[0]#worst"),
        (7, "hard.q.window_extrema"),
    ]
    assert [(role.bit, role.id) for role in required.plan("mode").legend] == [
        (0, "extent"),
        (1, "gap"),
        (2, "hard.mode.state_transitions"),
    ]


def test_legend_bits_are_contiguous_and_match_the_requirement_results():
    run = make_run(q=wave())
    required = plan_of(run, hard={"q": {"global_extrema": {}, "violation": [{"above": 58.0}]}})
    plan = required.plan("q")

    assert [role.bit for role in plan.legend] == list(range(len(plan.legend)))
    by_id = {result.id: result for result in plan.requirements}
    assert by_id["hard.q.global_extrema"].roles == {"main": 2}
    assert by_id["hard.q.violation[0]"].roles == {"edge": 3, "worst": 4}
    assert by_id["hard.q.violation[0]"].bits == [3, 4]
    assert by_id["hard.q.violation[0]"].mask == (1 << 3) | (1 << 4)


def test_signal_without_hard_requirements_has_no_gap_bit():
    run = make_run(q=wave(), other=wave())
    required = plan_of(run, hard={"q": {"global_extrema": {}}})

    assert [role.id for role in required.plan("other").legend] == ["extent"]
    assert [role.id for role in required.plan("q").legend][:2] == ["extent", "gap"]


def test_more_than_64_roles_is_a_policy_error():
    run = make_run(q=wave())
    specs = [{"value": float(i)} for i in range(MAX_ROLES - 1)]  # extent + gap + 63 = 65 bits
    with pytest.raises(PolicyError) as exc:
        plan_of(run, hard={"q": {"threshold_crossing": specs}})

    message = str(exc.value)
    assert "65 role bits" in message and str(MAX_ROLES) in message
    assert exc.value.issues[0].path == "hard.q"


def test_legend_of_is_the_single_rule_for_the_legend():
    run = make_run(q=wave())
    bound = bind(run, hard={"q": {"global_extrema": {}, "violation": [{"above": 58.0}]}})
    legend = legend_of(bound.reqs_by_signal["q"])

    assert [role.id for role in legend] == [role.id for role in evaluate_run(run, bound).plan("q").legend]
    assert [role.id for role in legend_of([])] == ["extent"]


# ----- implicit retention ---------------------------------------------------------------------


def test_extent_is_retained_for_every_included_signal():
    values = wave()
    run = make_run(q=values, other=values)
    required = plan_of(run, hard={"q": {"global_extrema": {}}})

    for name in ("q", "other"):
        plan = required.plan(name)
        extent = plan.samples.indices_with(1 << 0)
        assert extent.tolist() == [0, len(values) - 1]
    assert required.plan("other").samples.materialize()[0].tolist() == [0, len(values) - 1]


def test_gap_retention_only_for_signals_with_a_hard_contract():
    values = wave()
    values[7] = NAN
    values[8] = NAN
    run = make_run(q=values, other=values.copy())
    required = plan_of(run, hard={"q": {"global_extrema": {}}})

    # The non-finite run's own boundaries plus the finite samples around it.
    assert required.plan("q").samples.indices_with(1 << 1).tolist() == [6, 7, 8, 9]
    assert required.plan("other").samples.materialize()[0].tolist() == [0, len(values) - 1]


def test_extent_of_an_empty_signal_retains_nothing():
    run = make_run(q=np.empty(0, dtype=np.float64))
    required = plan_of(run, hard={"q": {"global_extrema": {}}})
    plan = required.plan("q")

    assert plan.retained == 0
    assert plan.samples.is_empty
    assert plan.requirements[0].status == STATUS_NOT_APPLICABLE


# ----- operator results -----------------------------------------------------------------------


def test_retained_samples_are_the_operators_own_output():
    values = wave()
    values[7] = NAN
    run = make_run(q=values)
    required = plan_of(
        run,
        hard={"q": {"global_extrema": {}, "threshold_crossing": [{"value": 55.0}], "violation": [{"above": 58.0}]}},
    )
    plan = required.plan("q")
    sig = run.signals["q"]

    expected = {
        "hard.q.global_extrema": global_extrema.evaluate(sig, {}, {"main": 0}),
        "hard.q.threshold_crossing[0]": threshold_crossing.evaluate(sig, {"value": 55.0}, {"main": 0}),
        "hard.q.violation[0]": violation.evaluate(sig, {"above": 58.0}, {"edge": 0, "worst": 1}),
    }
    for result in plan.requirements:
        own = expected[result.id].samples.materialize()[0]
        assert plan.samples.indices_with(result.mask).tolist() == own.tolist()
        assert result.evidence == expected[result.id].evidence


def test_statuses_come_from_the_operators():
    run = make_run(
        empty_of_finite=np.full(20, NAN),
        mode=np.repeat([0, 1], [10, 10]).astype(np.int32),
    )
    required = plan_of(
        run,
        hard={"empty_of_finite": {"global_extrema": {}}, "mode": {"state_transitions": {}}},
    )

    assert required.plan("empty_of_finite").requirements[0].status == STATUS_NOT_APPLICABLE
    assert required.plan("mode").requirements[0].status == STATUS_PASS
    assert required.statuses() == [STATUS_NOT_APPLICABLE, STATUS_PASS]


def test_pending_at_end_is_a_warning():
    # A last run shorter than the debounce leaves the crossing pending (docs/contracts.md, step 4).
    values = np.array([0.0, 0.0, 10.0, 10.0, 10.0, 0.0], dtype=np.float64)
    run = make_run(q=values)
    required = plan_of(run, hard={"q": {"threshold_crossing": [{"value": 5.0, "debounce": "2 s"}]}})
    result = required.plan("q").requirements[0]

    assert result.status == STATUS_WARN
    assert result.evidence["pending_at_end"] is True
    assert any("pending_at_end" in note for note in required.notes)


def test_requirements_are_reported_in_policy_order_across_signals():
    run = make_run(q=wave(), mode=np.repeat([0, 1], 20).astype(np.int32))
    required = plan_of(
        run,
        hard={"q": {"global_extrema": {}}, "mode": {"state_transitions": {}}},
    )
    assert [result.id for result in required.requirements] == [
        "hard.q.global_extrema",
        "hard.mode.state_transitions",
    ]


def test_state_transitions_evidence_and_retention():
    mode = np.repeat([0, 1, 2], [10, 15, 15]).astype(np.int32)
    run = make_run(mode=mode)
    required = plan_of(run, hard={"mode": {"state_transitions": {}}})
    plan = required.plan("mode")

    own = state_transitions.evaluate(run.signals["mode"], {}, {"main": 0})
    assert plan.samples.materialize()[0].tolist() == own.samples.materialize()[0].tolist()
    assert plan.requirements[0].evidence["transitions"] == 2


def test_operator_parameter_errors_become_policy_errors():
    run = make_run(q=wave())
    bound = bind(run, hard={"q": {"violation": [{"above": 58.0}]}})
    bound.reqs_by_signal["q"][0].params = {}  # neither above nor below

    with pytest.raises(PolicyError) as exc:
        evaluate_run(run, bound)
    assert exc.value.issues[0].path == "hard.q.violation[0]"
    assert "above" in str(exc.value)


# ----- rejected policies ----------------------------------------------------------------------


def test_unsupported_operator_is_named(monkeypatch):
    # every operator the policy schema allows is implemented, so simulate a build without one
    from baslt.plan import required

    monkeypatch.delitem(required.OP_EVALUATORS, "local_extrema")
    run = make_run(q=wave())
    with pytest.raises(UsageError) as exc:
        plan_of(run, hard={"q": {"local_extrema": {"prominence": 1.0}}})

    message = str(exc.value)
    assert "hard.q.local_extrema: the local_extrema operator" in message
    assert "global_extrema" in message  # the message lists what this build does support


def test_unsupported_sections_are_named():
    run = make_run(q=wave(), thrust=wave(), pos=np.zeros((40, 3)))
    sections = {
        "events": {"MECO": {"when": {"signal": "thrust", "falls_below": 10.0}}},
        "trajectories": {"ascent": {"position": "pos", "max_position_error": 1.0}},
        "sync_groups": {"dyn": {"members": ["q", "thrust"]}},
        "soft": [{"match": "q", "priority": "high"}],
    }
    with pytest.raises(UsageError) as exc:
        plan_of(run, **sections)

    message = str(exc.value)
    assert "events.MECO: events" in message
    assert "trajectories.ascent: trajectories" in message
    assert "sync_groups.dyn: sync groups" in message
    assert "soft[0] (match 'q'): the soft layer" in message


@pytest.mark.parametrize(
    ("section", "needle"),
    [
        ({"events": {"MECO": {"when": {"signal": "q", "falls_below": 10.0}}}}, "events"),
        ({"sync_groups": {"dyn": {"members": ["q", "thrust"]}}}, "sync groups"),
        ({"soft": [{"match": "*", "priority": "none"}]}, "the soft layer"),
    ],
)
def test_each_unsupported_section_alone_is_rejected(section, needle):
    run = make_run(q=wave(), thrust=wave())
    with pytest.raises(UsageError, match=needle):
        plan_of(run, **section)


def test_supported_policy_without_extras_is_accepted():
    run = make_run(q=wave())
    required = plan_of(run, hard={"q": {"global_extrema": {}}})
    assert required.plan("q").retained > 0


def test_signal_missing_from_the_run_is_a_usage_error():
    run = make_run(q=wave(), other=wave())
    bound = bind(run, hard={"q": {"global_extrema": {}}})
    del run.signals["other"]

    with pytest.raises(UsageError) as exc:
        evaluate_run(run, bound)
    assert "'other'" in str(exc.value) and "loaded signals: q" in str(exc.value)
