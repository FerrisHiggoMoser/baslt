"""Events and sync groups in the planner: retained samples, evidence and statuses."""

from __future__ import annotations

import numpy as np
import pytest

from baslt.errors import PolicyError
from baslt.plan.required import evaluate_run, event_role_id, propagates, sync_role_id
from baslt.policy import bind_policy, load_policy
from baslt.signals import Run, SourceMeta, normalize_signal


def run_of(**signals) -> Run:
    """Each value is (t, v) or v on the clock t = 0, 1, 2, ..."""
    out = {}
    for name, value in signals.items():
        t, v = value if isinstance(value, tuple) else (np.arange(len(value), dtype=float), value)
        labels = None
        if isinstance(v, list) and v and isinstance(v[0], str):
            v = np.array(v)
        out[name], _ = normalize_signal(name, np.asarray(t, float), np.asarray(v), labels=labels)
    return Run(signals=out, meta=SourceMeta(path=None, format="numpy", size_bytes=0, issues=[]))


def plan(run: Run, **sections):
    policy = load_policy({"version": 1, **sections})
    return evaluate_run(run, bind_policy(policy, run.infos()))


def retained(result, signal: str, role: str | None = None) -> list[int]:
    p = result.signals[signal]
    if role is None:
        return p.samples.materialize()[0].tolist()
    return p.samples.indices_with(1 << p.bit_of(role)).tolist()


def test_legend_order_adds_event_and_sync_roles_after_hard_roles():
    run = run_of(q=np.arange(10.0), thrust=np.arange(10.0)[::-1])
    result = plan(run, hard={"q": {"global_extrema": {}}},
                  events={"E": {"when": {"signal": "thrust", "falls_below": 3}, "keep": {"signals": ["q", "thrust"]}}},
                  sync_groups={"g": {"members": ["q", "thrust"]}})
    assert [r.id for r in result.signals["q"].legend] == [
        "extent", "gap", "hard.q.global_extrema", "event.E#window", "sync.g"]
    assert [r.id for r in result.signals["thrust"].legend] == [
        "extent", "event.E#trigger", "event.E#window", "sync.g"]
    assert propagates("event.E#window") and propagates("hard.q.global_extrema")
    assert not propagates("extent") and not propagates("gap") and not propagates("sync.g")


def test_falling_trigger_windows_and_clipping():
    thrust = np.array([9, 9, 9, 9, 9, 1, 1, 1, 1, 1], float)
    q = np.arange(10.0)
    result = plan(run_of(q=q, thrust=thrust),
                  events={"MECO": {"when": {"signal": "thrust", "falls_below": 5},
                                   "keep": {"before": "1 s", "after": "2 s", "signals": ["q"]}}})
    event = result.events[0]
    assert event.status == "pass" and event.evidence["found"] == 1
    trigger = event.evidence["triggers"][0]
    assert trigger == {"t": 4.5, "index_before": 4, "index_after": 5, "at_start": False, "gap": False}
    # window [3.5, 6.5] -> samples 4..6, plus one neighbour each side -> 3..7
    assert event.evidence["windows"] == [
        {"signal": "q", "trigger": 4.5, "start": 3, "end": 7, "samples": 5, "clipped": False}]
    assert retained(result, "q", event_role_id("MECO", "window")) == [3, 4, 5, 6, 7]
    assert retained(result, "thrust", event_role_id("MECO", "trigger")) == [4, 5]


def test_windows_past_the_signal_are_clipped_and_empty_windows_keep_their_bracket():
    thrust = np.array([9, 1, 1, 1], float)
    sparse_t = np.array([0.0, 10.0, 20.0])
    result = plan(run_of(thrust=thrust, slow=(sparse_t, np.array([1.0, 2.0, 3.0]))),
                  events={"E": {"when": {"signal": "thrust", "falls_below": 5},
                                "keep": {"before": "5 s", "after": "1 s", "signals": ["thrust", "slow"]}}})
    windows = {w["signal"]: w for w in result.events[0].evidence["windows"]}
    assert windows["thrust"] == {"signal": "thrust", "trigger": 0.5, "start": 0, "end": 2, "samples": 3,
                                 "clipped": True}
    # no sample of "slow" lies in [-4.5, 1.5]: the window keeps the samples around it
    assert windows["slow"]["start"] == 0 and windows["slow"]["end"] == 1


def test_every_candidate_flip_is_retained_even_when_debounce_rejects_it():
    #                 short dip at 2..3 is rejected by the 3 s debounce; the real fall at 6 is kept
    thrust = np.array([9, 9, 1, 9, 9, 9, 1, 1, 1, 1, 1, 1], float)
    result = plan(run_of(thrust=thrust),
                  events={"E": {"when": {"signal": "thrust", "falls_below": 5, "debounce": "3 s"}}})
    event = result.events[0]
    assert [t["index_before"] for t in event.evidence["triggers"]] == [5]
    assert retained(result, "thrust", event_role_id("E", "trigger")) == [1, 2, 3, 5, 6]


def test_occurrence_and_expect():
    x = np.array([9, 1, 9, 1, 9, 1], float)
    first = plan(run_of(x=x), events={"E": {"when": {"signal": "x", "falls_below": 5}, "expect": 2}}).events[0]
    assert first.evidence["found"] == 3 and len(first.evidence["triggers"]) == 1
    assert first.status == "warn" and "expected 2" in first.notes[0]
    last = plan(run_of(x=x), events={"E": {"when": {"signal": "x", "falls_below": 5}, "occurrence": "last"}})
    assert last.events[0].evidence["triggers"][0]["index_before"] == 4
    every = plan(run_of(x=x), events={"E": {"when": {"signal": "x", "falls_below": 5}, "occurrence": "all",
                                            "expect": 3}})
    assert every.events[0].status == "pass" and len(every.events[0].evidence["triggers"]) == 3


def test_no_trigger_is_a_pass_with_no_windows():
    result = plan(run_of(x=np.full(5, 9.0), q=np.arange(5.0)),
                  events={"E": {"when": {"signal": "x", "falls_below": 5}, "keep": {"signals": ["q"]}}})
    event = result.events[0]
    assert event.status == "pass" and event.evidence["found"] == 0
    assert event.evidence["windows"] == [] and event.evidence["windows_total"] == 0


def test_equals_on_codes_and_labels():
    mode = np.array([0, 0, 2, 2, 1, 2, 2], dtype=np.int32)
    result = plan(run_of(mode=mode), events={"E": {"when": {"signal": "mode", "equals": 2}, "occurrence": "all"}})
    triggers = result.events[0].evidence["triggers"]
    assert [(t["index_before"], t["index_after"]) for t in triggers] == [(1, 2), (4, 5)]
    assert retained(result, "mode", event_role_id("E", "trigger")) == [1, 2, 4, 5]

    labels = plan(run_of(phase=["burn", "burn", "coast", "burn"]),
                  events={"E": {"when": {"signal": "phase", "equals": "burn"}, "occurrence": "all"}})
    triggers = labels.events[0].evidence["triggers"]
    assert [(t["index_after"], t["at_start"]) for t in triggers] == [(0, True), (3, False)]
    assert triggers[0]["index_before"] is None


def test_event_errors_name_the_event():
    with pytest.raises(PolicyError, match="events.E.*not one of the labels"):
        plan(run_of(phase=["burn", "coast"]), events={"E": {"when": {"signal": "phase", "equals": "idle"}}})
    with pytest.raises(PolicyError, match="events.E.*must be scalar"):
        plan(run_of(pos=np.zeros((4, 3))), events={"E": {"when": {"signal": "pos", "falls_below": 1}}})


def test_sync_propagates_exact_and_bracketed_timestamps():
    fast_t = np.arange(0.0, 10.0, 1.0)
    slow_t = np.array([0.0, 2.5, 5.0, 7.5, 9.0])
    result = plan(
        run_of(fast=(fast_t, np.sin(fast_t)), slow=(slow_t, np.array([5.0, 1.0, 3.0, 4.0, 2.0]))),
        hard={"slow": {"global_extrema": {}}},
        sync_groups={"g": {"members": ["fast", "slow"]}},
    )
    group = result.sync_groups[0]
    # propagating: extrema of slow (max 5 at 0.0, min 1 at 2.5) -> times 0.0 and 2.5
    assert group.evidence == {"members": ["fast", "slow"], "propagating": 2, "aligned": 3, "unaligned": 1,
                              "out_of_range": 0}
    assert group.status == "warn"
    assert retained(result, "fast", sync_role_id("g")) == [0, 2, 3]  # 0.0 exact, 2.5 bracketed by 2 and 3
    assert retained(result, "slow", sync_role_id("g")) == [0, 1]


def test_sync_samples_do_not_propagate_and_out_of_range_is_counted():
    a_t = np.arange(0.0, 5.0)
    b_t = np.arange(2.0, 4.0)
    c_t = np.arange(0.0, 5.0) + 0.5
    result = plan(
        run_of(a=(a_t, a_t), b=(b_t, b_t), c=(c_t, c_t)),
        hard={"a": {"global_extrema": {}}},
        sync_groups={"one": {"members": ["a", "b"]}, "two": {"members": ["b", "c"]}},
    )
    one, two = result.sync_groups
    # a's extrema are at 0 and 4, both outside b's span [2, 3]
    assert one.evidence["propagating"] == 2 and one.evidence["out_of_range"] == 2
    # nothing in b or c carries a propagating role, so group two copies nothing
    assert two.evidence["propagating"] == 0
    assert retained(result, "c", sync_role_id("two")) == []


def test_event_windows_propagate_through_sync_groups():
    t = np.arange(0.0, 20.0)
    result = plan(
        run_of(x=(t, np.where(t < 10, 9.0, 1.0)), q=(t, t), r=(t + 0.25, t)),
        events={"E": {"when": {"signal": "x", "falls_below": 5}, "keep": {"before": "1 s", "after": "1 s",
                                                                          "signals": ["q"]}}},
        sync_groups={"g": {"members": ["q", "r"]}},
    )
    window = retained(result, "q", event_role_id("E", "window"))
    assert window == [8, 9, 10, 11]
    group = result.sync_groups[0]
    # the window times 8..11 fall between r's samples (r runs 0.25 s late)
    assert group.evidence["propagating"] == 4 and group.evidence["unaligned"] == 4
    assert group.evidence["aligned"] == 4  # q itself
    assert retained(result, "r", sync_role_id("g")) == [7, 8, 9, 10, 11]
