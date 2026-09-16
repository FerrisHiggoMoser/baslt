"""What the hard requirements of a bound policy need from a run.

`evaluate_run` runs one operator per bound requirement and unions the sample sets they ask for with the
implicit retention of docs/contracts.md:

- `extent`: the first and last source sample of every included signal;
- `gap`: for every signal carrying at least one hard contract, the boundaries of every non-finite run and
  the finite samples around it.

Role legend (the `roles` list of a signal in docs/container.md) in a deterministic order:

    bit 0  extent
    bit 1  gap                       only for a signal with at least one hard requirement
    bit 2+ one bit per requirement role, requirements in policy order

A requirement contributes one bit per sub-role its operator fills: operators with a single role (`main`)
give the requirement id itself, and the others suffix it, so `violation` occupies
`hard.q_dyn.violation[0]#edge` and `hard.q_dyn.violation[0]#worst`. A `roles` bitmask is at most 64 bits
wide (docs/container.md), so more than 64 roles on one signal is a PolicyError.

After the hard pass, events and sync groups add their own roles, still in policy order:

    event.<name>#trigger   on the event's trigger signal
    event.<name>#window    on every signal the event keeps a window of
    sync.<name>            on every member of a sync group

Events retain every candidate trigger (as threshold_crossing retains every candidate flip), so detecting the
event again on the retained samples finds exactly what the source had, and then the windows around the selected
triggers. Sync groups run last, in one pass: the timestamps carried by hard and event roles of any member are
retained in every member, exactly when present and otherwise by their bracketing pair. Samples added that way
carry only a sync role and do not propagate again (docs/contracts.md).

Detection of crossings, violations and event triggers on any superset of these samples gives the same result as on
the source, so nothing added later (sync, and later the soft layer) can create or hide one.

A policy that asks for a feature this build does not implement -- trajectories or the soft layer -- is rejected
with a UsageError naming the feature instead of being compiled into an artifact that would silently not carry it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..errors import Issue, PolicyError, UsageError
from ..ops import (
    global_extrema,
    local_extrema,
    state_transitions,
    threshold_crossing,
    violation,
    window_extrema,
)
from ..ops._common import STATUS_NOT_APPLICABLE, STATUS_PASS, STATUS_WARN, capped, extent_samples, gap_samples
from ..sampleset import SampleSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..policy.schema import BoundEvent, BoundPolicy, BoundReq, BoundSyncGroup
    from ..signals import Run, Signal

EXTENT_ROLE = "extent"
GAP_ROLE = "gap"
MAX_ROLES = 64  # a roles bitmask is at most <u8 (docs/container.md)

MAIN = "main"
_MAIN_ONLY: tuple[str, ...] = (MAIN,)

# Sub-roles every operator fills, in legend order (docs/contracts.md "Roles").
OP_ROLES: dict[str, tuple[str, ...]] = {
    "global_extrema": _MAIN_ONLY,
    "local_extrema": ("peak", "base"),
    "window_extrema": _MAIN_ONLY,
    "threshold_crossing": _MAIN_ONLY,
    "violation": ("edge", "worst"),
    "state_transitions": _MAIN_ONLY,
}

OP_EVALUATORS = {
    "global_extrema": global_extrema.evaluate,
    "local_extrema": local_extrema.evaluate,
    "window_extrema": window_extrema.evaluate,
    "threshold_crossing": threshold_crossing.evaluate,
    "violation": violation.evaluate,
    "state_transitions": state_transitions.evaluate,
}

SUPPORTED_OPS: tuple[str, ...] = tuple(sorted(OP_EVALUATORS))


TRIGGER = "trigger"
WINDOW = "window"
NON_PROPAGATING_PREFIXES: tuple[str, ...] = ("sync.", "link.")


def role_id(req_id: str, role: str) -> str:
    """Legend id of one sub-role: the requirement id itself for `main`, `<id>#<role>` otherwise."""
    return req_id if role == MAIN else f"{req_id}#{role}"


def event_role_id(name: str, role: str) -> str:
    return f"event.{name}#{role}"


def sync_role_id(name: str) -> str:
    return f"sync.{name}"


def propagates(legend_id: str) -> bool:
    """Whether samples carrying this role are copied across a sync group (docs/contracts.md, "sync groups")."""
    return legend_id not in (EXTENT_ROLE, GAP_ROLE) and not legend_id.startswith(NON_PROPAGATING_PREFIXES)


@dataclass(frozen=True, slots=True)
class Role:
    """One entry of a signal's role legend."""

    bit: int
    id: str

    def to_json(self) -> dict:
        return {"bit": int(self.bit), "id": self.id}


@dataclass(slots=True)
class RequirementResult:
    """One evaluated hard requirement: where its samples are, what it found and how it ended."""

    req: BoundReq
    roles: dict[str, int]  # sub-role name -> legend bit
    status: str
    evidence: dict
    notes: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.req.id

    @property
    def signal(self) -> str:
        return self.req.signal

    @property
    def op(self) -> str:
        return self.req.op

    @property
    def severity(self) -> str:
        return self.req.severity

    @property
    def bits(self) -> list[int]:
        """Legend bits of this requirement, ascending."""
        return sorted(self.roles.values())

    @property
    def mask(self) -> int:
        mask = 0
        for bit in self.roles.values():
            mask |= 1 << bit
        return mask


@dataclass(slots=True)
class SignalPlan:
    """Everything the planner decided about one included signal."""

    name: str
    signal: Signal
    legend: list[Role]
    samples: SampleSet
    requirements: list[RequirementResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def retained(self) -> int:
        return self.samples.count()

    def legend_json(self) -> list[dict]:
        return [role.to_json() for role in self.legend]

    def bit_of(self, legend_id: str) -> int:
        for role in self.legend:
            if role.id == legend_id:
                return role.bit
        raise KeyError(legend_id)

    def propagating_mask(self) -> int:
        mask = 0
        for role in self.legend:
            if propagates(role.id):
                mask |= 1 << role.bit
        return mask


@dataclass(slots=True)
class EventResult:
    """One evaluated event: its triggers and the windows kept around the selected ones."""

    event: BoundEvent
    status: str
    evidence: dict
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.event.name

    @property
    def signal(self) -> str:
        return self.event.signal

    @property
    def severity(self) -> str:
        return self.event.severity

    @property
    def triggered(self) -> bool:
        return bool(self.evidence.get("triggers"))


@dataclass(slots=True)
class SyncResult:
    """One sync group after propagation."""

    group: BoundSyncGroup
    status: str
    evidence: dict

    @property
    def name(self) -> str:
        return self.group.name


@dataclass(slots=True)
class RequiredResult:
    """The plan for a whole run: one SignalPlan per included signal, requirements in policy order."""

    signals: dict[str, SignalPlan]
    requirements: list[RequirementResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    events: list[EventResult] = field(default_factory=list)
    sync_groups: list[SyncResult] = field(default_factory=list)

    def statuses(self) -> list[str]:
        return ([result.status for result in self.requirements] + [result.status for result in self.events]
                + [result.status for result in self.sync_groups])

    def plan(self, name: str) -> SignalPlan:
        return self.signals[name]


def legend_of(reqs: Sequence[BoundReq], extra: Sequence[str] = ()) -> list[Role]:
    """Role legend for a signal carrying `reqs` and the event and sync roles `extra`, in the documented order."""
    entries = [EXTENT_ROLE]
    if reqs:
        entries.append(GAP_ROLE)
    for req in reqs:
        entries.extend(role_id(req.id, role) for role in OP_ROLES[req.op])
    entries.extend(extra)
    return [Role(bit=bit, id=name) for bit, name in enumerate(entries)]


def _check_legend(name: str, legend: list[Role], reqs: Sequence[BoundReq]) -> None:
    if len(legend) <= MAX_ROLES:
        return
    raise PolicyError(
        [
            Issue(
                path=f"hard.{name}",
                message=(
                    f"signal {name!r} needs {len(legend)} role bits for {len(reqs)} hard requirements and its "
                    f"events and sync groups, but a role bitmask holds at most {MAX_ROLES}; split them over "
                    "several policies or drop some of them"
                ),
            )
        ]
    )


def _unsupported(bound: BoundPolicy) -> None:
    """Reject every policy feature this milestone does not implement, naming each one."""
    problems: list[str] = []
    for trajectory in bound.trajectories:
        problems.append(f"trajectories.{trajectory.name}: trajectories")
    for i, rule in enumerate(bound.policy.soft):
        problems.append(f"soft[{i}] (match {rule.match!r}): the soft layer")
    for req in bound.reqs:
        if req.op not in OP_EVALUATORS:
            problems.append(f"{req.id}: the {req.op} operator")
    if not problems:
        return
    raise UsageError(
        "this build of baslt cannot compile these policy features yet:\n"
        + "\n".join(f"  {problem}" for problem in problems)
        + "\nit implements the hard operators "
        + ", ".join(SUPPORTED_OPS)
        + " with events and sync groups but no trajectories or soft layer; remove the features above"
    )


def _evaluate_signal(name: str, sig: Signal, reqs: Sequence[BoundReq], extra: Sequence[str] = ()) -> SignalPlan:
    legend = legend_of(reqs, extra)
    _check_legend(name, legend, reqs)
    bits = {role.id: role.bit for role in legend}

    samples = extent_samples(sig.n, bits[EXTENT_ROLE])
    if reqs:
        samples = samples.union(gap_samples(sig.v, bits[GAP_ROLE]))

    notes: list[str] = []
    results: list[RequirementResult] = []
    for req in reqs:
        roles = {role: bits[role_id(req.id, role)] for role in OP_ROLES[req.op]}
        try:
            out = OP_EVALUATORS[req.op](sig, req.params, roles)
        except ValueError as exc:
            # The operator rejected its own bound parameters or the signal's shape; that is a policy
            # problem, reported with the requirement's path like every other policy issue.
            raise PolicyError([Issue(path=req.id, message=str(exc))]) from None
        samples = samples.union(out.samples)
        results.append(
            RequirementResult(req=req, roles=roles, status=out.status, evidence=out.evidence, notes=list(out.notes))
        )
        notes.extend(out.notes)

    return SignalPlan(
        name=name,
        signal=sig,
        legend=legend,
        samples=samples.clip(sig.n),
        requirements=results,
        notes=notes,
    )


def evaluate_run(run: Run, bound: BoundPolicy) -> RequiredResult:
    """Evaluate every hard requirement of `bound` against `run`.

    Returns the retained samples, role legend, evidence and status of every included signal. Raises
    UsageError for a policy feature this milestone does not implement or for a signal the run did not
    load, and PolicyError for a legend wider than 64 bits or an operator that rejects its parameters.
    """
    _unsupported(bound)

    extra: dict[str, list[str]] = {name: [] for name in bound.included}
    for event in bound.events:
        extra[event.signal].append(event_role_id(event.name, TRIGGER))
        for name in event.signals:
            extra[name].append(event_role_id(event.name, WINDOW))
    for group in bound.sync_groups:
        for name in group.members:
            extra[name].append(sync_role_id(group.name))

    signals: dict[str, SignalPlan] = {}
    requirements: list[RequirementResult] = []
    notes: list[str] = []
    for name in bound.included:
        sig = run.signals.get(name)
        if sig is None:
            available = ", ".join(sorted(run.signals)) or "none"
            raise UsageError(
                f"the policy includes signal {name!r} but the run does not carry it "
                f"(loaded signals: {available})"
            )
        plan = _evaluate_signal(name, sig, bound.reqs_by_signal.get(name, ()), extra[name])
        signals[name] = plan
        notes.extend(plan.notes)

    # Requirements in policy order, not in signal order.
    by_id = {result.id: result for plan in signals.values() for result in plan.requirements}
    requirements = [by_id[req.id] for req in bound.reqs if req.id in by_id]

    events = [_evaluate_event(event, signals) for event in bound.events]
    for result in events:
        notes.extend(result.notes)
    sync_groups = _propagate_sync(bound.sync_groups, signals)
    for plan in signals.values():
        plan.samples = plan.samples.clip(plan.signal.n)
    return RequiredResult(signals=signals, requirements=requirements, notes=notes, events=events,
                          sync_groups=sync_groups)


# --------------------------------------------------------------------------------------------------------------
# Events


def _event_issue(event: BoundEvent, message: str) -> PolicyError:
    return PolicyError([Issue(path=f"events.{event.name}", message=message)])


def _equals_code(event: BoundEvent, sig: Signal) -> float:
    value = event.value
    if isinstance(value, str):
        if not sig.labels:
            raise _event_issue(event, f"equals {value!r} is text, but signal {sig.name!r} has no text labels")
        if value not in sig.labels:
            raise _event_issue(event, f"equals {value!r} is not one of the labels of {sig.name!r}: "
                                      + ", ".join(repr(label) for label in sig.labels))
        return float(sig.labels.index(value))
    return float(value)


def _evaluate_event(event: BoundEvent, plans: dict[str, SignalPlan]) -> EventResult:
    """Triggers on the trigger signal, then windows on every listed signal (docs/contracts.md, "events")."""
    plan = plans[event.signal]
    sig = plan.signal
    if sig.v.ndim != 1:
        raise _event_issue(event, f"the trigger signal {sig.name!r} must be scalar, not a vector")
    bit = plan.bit_of(event_role_id(event.name, TRIGGER))
    notes: list[str] = []
    pending = False
    status = STATUS_PASS

    if event.trigger in ("falls_below", "rises_above"):
        params = {
            "value": float(event.value),
            "edge": "falling" if event.trigger == "falls_below" else "rising",
            "hysteresis": float(event.hysteresis),
            "debounce": float(event.debounce),
            "tolerance": 0.0,
            "interpolate": "linear",
        }
        try:
            out = threshold_crossing.evaluate(sig, params, {MAIN: bit})
        except ValueError as exc:
            raise _event_issue(event, str(exc)) from None
        samples = out.samples
        if out.status == STATUS_NOT_APPLICABLE:
            status = STATUS_NOT_APPLICABLE
            notes.extend(out.notes)
        detection = out.raw[0] if out.raw else None
        triggers: list[dict] = []
        if detection is not None:
            pending = bool(detection.pending_at_end)
            for tc, before, after, gap in zip(detection.t.tolist(), detection.index_before.tolist(),
                                              detection.index_after.tolist(), detection.gap.tolist()):
                triggers.append({"t": tc, "index_before": before, "index_after": after, "at_start": False,
                                 "gap": bool(gap)})
    elif event.trigger == "equals":
        if sig.kind != "discrete":
            raise _event_issue(event, f"equals needs a discrete signal, but {sig.name!r} is {sig.kind}")
        code = _equals_code(event, sig)
        hit = sig.v == code
        previous = np.concatenate(([False], hit[:-1])) if hit.size else hit
        starts = np.flatnonzero(hit & ~previous)
        samples = SampleSet.from_points(np.concatenate([starts, starts[starts > 0] - 1]), bit)
        triggers = [
            {"t": float(sig.t[s]), "index_before": s - 1 if s > 0 else None, "index_after": s,
             "at_start": s == 0, "gap": False}
            for s in starts.tolist()
        ]
        if sig.n == 0:
            status = STATUS_NOT_APPLICABLE
            notes.append(f"{sig.name}: empty signal")
    else:
        raise _event_issue(event, f"unknown trigger {event.trigger!r}")
    plan.samples = plan.samples.union(samples)

    found = len(triggers)
    if event.occurrence == "first":
        selected = triggers[:1]
    elif event.occurrence == "last":
        selected = triggers[-1:]
    else:
        selected = triggers

    windows: list[dict] = []
    for name in event.signals:
        target = plans[name]
        t = target.signal.t
        n = int(t.shape[0])
        if n == 0 or not selected:
            continue
        taus = np.array([trigger["t"] for trigger in selected], dtype=np.float64)
        lo = np.searchsorted(t, taus - event.before, side="left")
        hi = np.searchsorted(t, taus + event.after, side="right") - 1
        start = np.maximum(lo - 1, 0)
        end = np.minimum(hi + 1, n - 1)
        clipped = (taus - event.before < t[0]) | (taus + event.after > t[-1])
        target.samples = target.samples.union(
            SampleSet.from_ranges(start, end + 1, target.bit_of(event_role_id(event.name, WINDOW))))
        for tau, a, b, c in zip(taus.tolist(), start.tolist(), end.tolist(), clipped.tolist()):
            windows.append({"signal": name, "trigger": tau, "start": a, "end": b, "samples": b - a + 1,
                            "clipped": bool(c)})

    if status != STATUS_NOT_APPLICABLE and event.expect is not None and found != event.expect:
        status = STATUS_WARN
        notes.append(f"event {event.name}: expected {event.expect} occurrence(s), found {found}")
    evidence = {
        "condition": event.trigger,
        "value": event.value,
        "occurrence": event.occurrence,
        "expect": event.expect,
        "found": found,
        "pending_at_end": pending,
        "triggers": capped(selected),
        "windows": capped(windows),
        "windows_total": len(windows),
    }
    return EventResult(event=event, status=status, evidence=evidence, notes=notes)


# --------------------------------------------------------------------------------------------------------------
# Sync groups


def _unique_bits(values: np.ndarray) -> np.ndarray:
    """Distinct float64 values, compared bitwise, in ascending order."""
    if values.size == 0:
        return values.astype(np.float64)
    bits = np.unique(np.ascontiguousarray(values, dtype=np.float64).view(np.int64))
    out = bits.view(np.float64)
    return out[np.argsort(out, kind="stable")]


def _propagate_sync(groups: Sequence[BoundSyncGroup], plans: dict[str, SignalPlan]) -> list[SyncResult]:
    """One propagation pass over every group, from the samples retained before any group ran."""
    stamps: list[np.ndarray] = []
    for group in groups:
        parts = []
        for name in group.members:
            plan = plans[name]
            idx = plan.samples.indices_with(plan.propagating_mask())
            parts.append(plan.signal.t[idx])
        stamps.append(_unique_bits(np.concatenate(parts)) if parts else np.empty(0))

    results: list[SyncResult] = []
    for group, times in zip(groups, stamps):
        aligned = unaligned = out_of_range = 0
        for name in group.members:
            plan = plans[name]
            t = plan.signal.t
            n = int(t.shape[0])
            if n == 0:
                out_of_range += int(times.shape[0])
                continue
            inside = (times >= t[0]) & (times <= t[-1])
            wanted = times[inside]
            out_of_range += int(times.shape[0] - wanted.shape[0])
            pos = np.searchsorted(t, wanted, side="left")
            at = np.minimum(pos, n - 1)
            exact = (pos < n) & (t[at].view(np.int64) == wanted.view(np.int64))
            aligned += int(np.count_nonzero(exact))
            unaligned += int(np.count_nonzero(~exact))
            loose = pos[~exact]
            points = np.concatenate([pos[exact], loose - 1, loose])
            points = points[(points >= 0) & (points < n)]
            plan.samples = plan.samples.union(SampleSet.from_points(points, plan.bit_of(sync_role_id(group.name))))
        evidence = {
            "members": list(group.members),
            "propagating": int(times.shape[0]),
            "aligned": aligned,
            "unaligned": unaligned,
            "out_of_range": out_of_range,
        }
        results.append(SyncResult(group=group, status=STATUS_WARN if unaligned else STATUS_PASS, evidence=evidence))
    return results
