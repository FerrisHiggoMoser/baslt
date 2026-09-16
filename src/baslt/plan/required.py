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

This milestone implements six operators. A policy that asks for anything else -- another operator,
events, trajectories, sync groups or the soft layer -- is rejected with a UsageError naming the feature
instead of being compiled into an artifact that would silently not carry it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import Issue, PolicyError, UsageError
from ..ops import (
    global_extrema,
    local_extrema,
    state_transitions,
    threshold_crossing,
    violation,
    window_extrema,
)
from ..ops._common import extent_samples, gap_samples
from ..sampleset import SampleSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..policy.schema import BoundPolicy, BoundReq
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


def role_id(req_id: str, role: str) -> str:
    """Legend id of one sub-role: the requirement id itself for `main`, `<id>#<role>` otherwise."""
    return req_id if role == MAIN else f"{req_id}#{role}"


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


@dataclass(slots=True)
class RequiredResult:
    """The plan for a whole run: one SignalPlan per included signal, requirements in policy order."""

    signals: dict[str, SignalPlan]
    requirements: list[RequirementResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def statuses(self) -> list[str]:
        return [result.status for result in self.requirements]

    def plan(self, name: str) -> SignalPlan:
        return self.signals[name]


def legend_of(reqs: Sequence[BoundReq]) -> list[Role]:
    """Role legend for a signal carrying `reqs`, in the documented order."""
    entries = [EXTENT_ROLE]
    if reqs:
        entries.append(GAP_ROLE)
    for req in reqs:
        entries.extend(role_id(req.id, role) for role in OP_ROLES[req.op])
    return [Role(bit=bit, id=name) for bit, name in enumerate(entries)]


def _check_legend(name: str, legend: list[Role], reqs: Sequence[BoundReq]) -> None:
    if len(legend) <= MAX_ROLES:
        return
    raise PolicyError(
        [
            Issue(
                path=f"hard.{name}",
                message=(
                    f"signal {name!r} needs {len(legend)} role bits for {len(reqs)} hard requirements, "
                    f"but a role bitmask holds at most {MAX_ROLES}; split the requirements over several "
                    "policies or drop some of them"
                ),
            )
        ]
    )


def _unsupported(bound: BoundPolicy) -> None:
    """Reject every policy feature this milestone does not implement, naming each one."""
    problems: list[str] = []
    for event in bound.events:
        problems.append(f"events.{event.name}: events")
    for trajectory in bound.trajectories:
        problems.append(f"trajectories.{trajectory.name}: trajectories")
    for group in bound.sync_groups:
        problems.append(f"sync_groups.{group.name}: sync groups")
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
        + " with no soft layer; remove the features above from the policy"
    )


def _evaluate_signal(name: str, sig: Signal, reqs: Sequence[BoundReq]) -> SignalPlan:
    legend = legend_of(reqs)
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
        plan = _evaluate_signal(name, sig, bound.reqs_by_signal.get(name, ()))
        signals[name] = plan
        notes.extend(plan.notes)

    # Requirements in policy order, not in signal order.
    by_id = {result.id: result for plan in signals.values() for result in plan.requirements}
    requirements = [by_id[req.id] for req in bound.reqs if req.id in by_id]
    return RequiredResult(signals=signals, requirements=requirements, notes=notes)
