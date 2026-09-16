"""Compile one run into `.baslt` container bytes.

    evaluate -> materialize per signal -> encode arrays -> index.json and policy.json
             -> manifest (size fixed point) -> write_zip -> self-verify

Every signal gets one `s/<k>` member holding its `v`, `idx` and `roles` arrays back to back, and points
its `t` descriptor at a `t/<j>` member. Signals whose retained timestamps are bit-identical share that
member, which is what makes a clock stored once for a whole run.

This milestone compiles the hard layer only: every retained sample is required by a hard contract, an event, a
sync group or the implicit `extent`/`gap` retention, so `budget.discretionary_bytes` is 0 and the artifact is the
smallest one the policy allows. A budget it does not fit is therefore infeasible outright, reported with
an itemized breakdown instead of a search for a smaller artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..container.encode import container_dtype, encode_array
from ..container.spec import (
    CODEC_METHODS,
    CONTAINER_VERSION,
    DEFAULT_LEVEL,
    ENC_DELTA_SHUFFLE,
    ENC_SHUFFLE,
    MEMBER_HEADER,
    MEMBER_INDEX,
    MEMBER_MANIFEST,
    MEMBER_POLICY,
    SIGNAL_MEMBER_PREFIX,
    TIME_MEMBER_PREFIX,
    canonical_json,
    header_json,
    idx_dtype,
    roles_dtype,
)
from ..container.zipwriter import Member, compress_member, write_zip, zip_size
from ..errors import CompileError, InfeasibleBudget, SelfVerifyFailed
from ..manifest import EventEntry, RequirementEntry, SignalBudget, SyncEntry, build_manifest
from ..policy.schema import OP_SCHEMAS
from .required import RequiredResult, SignalPlan, evaluate_run

if TYPE_CHECKING:
    from ..hashing import HashInfo
    from ..policy.schema import BoundPolicy
    from ..signals import Run

# Supported self-verification entry points, tried in order.
CHECK_ENTRY_POINTS: tuple[str, ...] = ("verify_artifact", "check_artifact", "run_checks", "verify")
FAILED_STATUSES = frozenset({"fail", "failed", "error"})


@dataclass(slots=True)
class CompileOutcome:
    """The compiled artifact, its manifest and the evidence the planner gathered."""

    bytes: bytes
    manifest: dict
    evidence: RequiredResult
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.bytes)

    @property
    def status(self) -> str:
        return self.manifest["status"]


@dataclass(slots=True)
class _Encoded:
    """One signal's retained samples, already encoded as container bytes."""

    plan: SignalPlan
    idx: np.ndarray  # int64 source indices
    t: bytes
    v: bytes
    ids: bytes
    roles: bytes
    t_dtype: str
    v_dtype: str
    ids_dtype: str
    roles_dtype: str
    components: int

    @property
    def n(self) -> int:
        return int(self.idx.shape[0])


def _encode_signal(plan: SignalPlan) -> _Encoded:
    sig = plan.signal
    idx, roles = plan.samples.materialize()
    t = sig.t[idx]
    v = sig.v[idx]

    ids_code = idx_dtype(sig.n)
    ids = idx.astype(np.uint32) if ids_code == "<u4" else idx.astype(np.float64)
    roles_code = roles_dtype(len(plan.legend))
    narrow = roles.astype(np.dtype(roles_code))

    return _Encoded(
        plan=plan,
        idx=idx,
        t=encode_array(t, ENC_SHUFFLE),
        v=encode_array(v, ENC_SHUFFLE),
        ids=encode_array(ids, ENC_DELTA_SHUFFLE),
        roles=encode_array(narrow, ENC_SHUFFLE),
        t_dtype=container_dtype(t.dtype),
        v_dtype=container_dtype(v.dtype),
        ids_dtype=ids_code,
        roles_dtype=roles_code,
        components=int(sig.components),
    )


def _desc(name: str, member: str, offset: int, data: bytes, n: int, components: int, dtype: str, enc: str) -> dict:
    return {
        "name": name,
        "member": member,
        "offset": int(offset),
        "nbytes": len(data),
        "n": int(n),
        "components": int(components),
        "dtype": dtype,
        "enc": enc,
    }


def _requirement_params(req) -> list[dict]:
    """Bound parameters in the documented order, so `structure.bind` can re-derive them."""
    schema = OP_SCHEMAS.get(req.op, {})
    names = [name for name in schema if name in req.params]
    names += sorted(name for name in req.params if name not in schema)
    return [{"name": name, "value": req.params[name]} for name in names]


def _index_and_members(
    encoded: list[_Encoded], bound: BoundPolicy, method: int
) -> tuple[dict, list[Member], list[Member], dict[str, list[str]]]:
    """index.json, the `s/` and `t/` members, and which members each signal owns for byte accounting."""
    signals: list[dict] = []
    signal_members: list[Member] = []
    time_members: list[Member] = []
    time_name_of: dict[bytes, str] = {}
    owned: dict[str, list[str]] = {}

    for k, item in enumerate(encoded):
        plan = item.plan
        sig = plan.signal
        s_name = f"{SIGNAL_MEMBER_PREFIX}{k}"
        t_name = time_name_of.get(item.t)
        mine = [s_name]
        if t_name is None:
            t_name = f"{TIME_MEMBER_PREFIX}{len(time_name_of)}"
            time_name_of[item.t] = t_name
            time_members.append(Member(t_name, item.t, method))
            mine.append(t_name)  # the first signal on a shared clock carries its bytes
        owned[plan.name] = mine
        signal_members.append(Member(s_name, item.v + item.ids + item.roles, method))

        kind = bound.kinds.get(plan.name, sig.kind)
        signals.append(
            {
                "name": plan.name,
                "path": sig.path,
                "kind": kind,
                "interp": "hold" if kind == "discrete" else "linear",
                "unit": bound.units.get(plan.name, sig.unit),
                "labels": list(sig.labels or []),
                "n_source": int(sig.n),
                "n": item.n,
                "components": item.components,
                "arrays": [
                    _desc("t", t_name, 0, item.t, item.n, 1, item.t_dtype, ENC_SHUFFLE),
                    _desc("v", s_name, 0, item.v, item.n, item.components, item.v_dtype, ENC_SHUFFLE),
                    _desc("idx", s_name, len(item.v), item.ids, item.n, 1, item.ids_dtype, ENC_DELTA_SHUFFLE),
                    _desc(
                        "roles",
                        s_name,
                        len(item.v) + len(item.ids),
                        item.roles,
                        item.n,
                        1,
                        item.roles_dtype,
                        ENC_SHUFFLE,
                    ),
                ],
                "roles": plan.legend_json(),
                "requirements": [
                    {
                        "id": result.id,
                        "op": result.op,
                        "bits": result.bits,
                        "params": _requirement_params(result.req),
                    }
                    for result in plan.requirements
                ],
            }
        )

    events = [
        {
            "name": event.name,
            "signal": event.signal,
            "condition": event.trigger,
            "value": event.value,
            "hysteresis": float(event.hysteresis),
            "debounce": float(event.debounce),
            "occurrence": event.occurrence,
            "expect": event.expect,
            "before": float(event.before),
            "after": float(event.after),
            "signals": list(event.signals),
            "severity": event.severity,
        }
        for event in bound.events
    ]
    sync_groups = [{"name": group.name, "members": list(group.members)} for group in bound.sync_groups]
    index = {"container": CONTAINER_VERSION, "signals": signals, "events": events, "sync_groups": sync_groups}
    return index, signal_members, time_members, owned


def _policy_json(bound: BoundPolicy) -> dict:
    policy = bound.policy
    return {
        "canonical": policy.canonical,
        "sha256": policy.sha256,
        "source_format": policy.source_format,
        "source_text": policy.source_text,
    }


def _standalone_bytes(item: _Encoded, mask: int, method: int, level: int) -> int:
    """Compressed bytes the samples selected by `mask` would cost on their own."""
    plan = item.plan
    sig = plan.signal
    idx = plan.samples.indices_with(mask)
    if idx.shape[0] == 0:
        return 0
    ids_code = idx_dtype(sig.n)
    ids = idx.astype(np.uint32) if ids_code == "<u4" else idx.astype(np.float64)
    roles = plan.samples.roles_at(idx).astype(np.dtype(item.roles_dtype))
    payload = (
        encode_array(sig.t[idx], ENC_SHUFFLE)
        + encode_array(sig.v[idx], ENC_SHUFFLE)
        + encode_array(ids, ENC_DELTA_SHUFFLE)
        + encode_array(roles, ENC_SHUFFLE)
    )
    return len(compress_member(Member("s/0", payload, method), level=level)[1])


def _infeasible(
    required: RequiredResult,
    encoded: list[_Encoded],
    rows: list[SignalBudget],
    size: int,
    max_bytes: int,
    budget_source: str,
    method: int,
    level: int,
) -> InfeasibleBudget:
    by_name = {item.plan.name: item for item in encoded}
    requirements = []
    for result in required.requirements:
        item = by_name[result.signal]
        requirements.append(
            {
                "id": result.id,
                "signal": result.signal,
                "op": result.op,
                "samples": int(item.plan.samples.with_bits(result.mask).count()),
                "standalone_bytes": _standalone_bytes(item, result.mask, method, level),
            }
        )
    for event in required.events:
        samples = 0
        standalone = 0
        for item in encoded:
            mask = 0
            for role in item.plan.legend:
                if role.id.startswith(f"event.{event.name}#"):
                    mask |= 1 << role.bit
            if mask:
                samples += int(item.plan.samples.with_bits(mask).count())
                standalone += _standalone_bytes(item, mask, method, level)
        requirements.append({"id": f"events.{event.name}", "signal": event.signal, "op": "event",
                             "samples": samples, "standalone_bytes": standalone})
    report = {
        "kind": "infeasible_budget",
        "max_bytes": int(max_bytes),
        "budget_source": budget_source,
        "minimum_bytes": int(size),
        "excess_bytes": int(size - max_bytes),
        "required_bytes": sum(int(row.bytes) for row in rows),
        "overhead_bytes": int(size - sum(int(row.bytes) for row in rows)),
        "requirements": requirements,
        "signals": [row.to_json() for row in rows],
    }
    source = "artifact.max_size" if budget_source == "policy" else "--max-size"
    return InfeasibleBudget(
        f"the hard requirements need {size} bytes but {source} allows {max_bytes}; "
        f"{size - max_bytes} bytes too many. Every retained sample is required by a hard contract, "
        "so no smaller artifact satisfies this policy",
        report,
    )


def _interpret(result: object) -> bool | None:
    """True/False when a self-verify result can be read, None when its shape is unknown."""
    if isinstance(result, bool):
        return result
    if isinstance(result, Mapping):
        if isinstance(result.get("ok"), bool):
            return bool(result["ok"])
        status = result.get("status")
        if isinstance(status, str):
            return status.lower() not in FAILED_STATUSES
        return None
    for attribute in ("ok", "passed", "success"):
        value = getattr(result, attribute, None)
        if isinstance(value, bool):
            return value
    status = getattr(result, "status", None)
    if isinstance(status, str):
        return status.lower() not in FAILED_STATUSES
    failures = getattr(result, "failures", getattr(result, "failed", None))
    if isinstance(failures, int) and not isinstance(failures, bool):
        return failures == 0
    if isinstance(failures, (list, tuple)):
        return len(failures) == 0
    return None


def run_self_verify(blob: bytes) -> list[str]:
    """Verify freshly compiled bytes with the independent verifier. Returns notes about what it did.

    Raises SelfVerifyFailed when the verifier rejects the artifact. The verifier is imported here, not
    at module import, to keep startup inexpensive. An unavailable verifier is an error.
    """
    try:
        from ..verify import checks
    except ImportError as exc:
        raise CompileError(f"self-verification is unavailable: {exc}") from exc
    if checks is None:
        raise CompileError("self-verification is unavailable")
    for name in CHECK_ENTRY_POINTS:
        entry = getattr(checks, name, None)
        if callable(entry):
            break
    else:
        raise CompileError("self-verification has no entry point")
    result = entry(blob)
    verdict = _interpret(result)
    if verdict is None:
        raise CompileError("self-verification returned an unrecognized result")
    if not verdict:
        raise SelfVerifyFailed(
            f"the compiled artifact failed its own verification (baslt.verify.checks.{name}); "
            "nothing was written",
            result,
        )
    return []


def compile_run(
    run: Run, bound: BoundPolicy, *, digest: HashInfo | None, self_verify: bool = True
) -> CompileOutcome:
    """Compile `run` under `bound` into container bytes.

    `digest` is the source digest recorded in the manifest (None writes the "none" digest). Pass
    `self_verify=False` to skip the independent verification of the compiled bytes.

    Raises InfeasibleBudget when the hard requirements do not fit `bound.max_bytes`, SelfVerifyFailed
    when the artifact fails its own verification, and UsageError or PolicyError for a policy this
    milestone cannot compile.
    """
    policy = bound.policy
    codec = policy.artifact.codec
    method = CODEC_METHODS[codec]
    level = DEFAULT_LEVEL

    required = evaluate_run(run, bound)
    encoded = [_encode_signal(required.signals[name]) for name in bound.included]

    index, signal_members, time_members, owned = _index_and_members(encoded, bound, method)
    data_members = signal_members + time_members
    csize = {member.name: len(compress_member(member, level=level)[1]) for member in data_members}

    header_bytes = header_json(codec, level)
    index_bytes = canonical_json(index)
    policy_bytes = canonical_json(_policy_json(bound))
    index_member = Member(MEMBER_INDEX, index_bytes, method)
    policy_member = Member(MEMBER_POLICY, policy_bytes, method)
    fixed = [
        (MEMBER_HEADER, len(header_bytes)),
        (MEMBER_INDEX, len(compress_member(index_member, level=level)[1])),
        (MEMBER_POLICY, len(compress_member(policy_member, level=level)[1])),
        *csize.items(),
    ]

    rows = [
        SignalBudget(
            name=item.plan.name,
            retained=item.n,
            hard=item.n,  # no soft layer yet: every retained sample is required
            soft=0,
            bytes=sum(csize[name] for name in owned[item.plan.name]),
        )
        for item in encoded
    ]
    requirements = [
        RequirementEntry(
            id=result.id,
            signal=result.signal,
            op=result.op,
            severity=result.severity,
            status=result.status,
            evidence=result.evidence,
        )
        for result in required.requirements
    ]
    events = [
        EventEntry(name=result.name, signal=result.signal, severity=result.severity, status=result.status,
                   evidence=result.evidence)
        for result in required.events
    ]
    sync_groups = [
        SyncEntry(name=result.name, status=result.status, evidence=result.evidence)
        for result in required.sync_groups
    ]

    manifest, size = build_manifest(
        source=run.meta,
        digest=digest,
        policy=policy,
        requirements=requirements,
        signals=rows,
        samples_total=sum(int(sig.n) for sig in run.signals.values()),
        source_signals=len(run.signals),
        size_of=lambda length: zip_size([*fixed, (MEMBER_MANIFEST, length)]),
        max_bytes=bound.max_bytes,
        budget_source=bound.budget_source,
        codec=codec,
        level=level,
        events=events,
        sync_groups=sync_groups,
    )

    if bound.max_bytes is not None and size > bound.max_bytes:
        raise _infeasible(required, encoded, rows, size, bound.max_bytes, bound.budget_source, method, level)

    members = [
        Member(MEMBER_HEADER, header_bytes, method),
        index_member,
        Member(MEMBER_MANIFEST, canonical_json(manifest), method),
        policy_member,
        *data_members,
    ]
    blob = write_zip(members, level=level)
    if len(blob) != size:
        raise CompileError(
            f"the artifact is {len(blob)} bytes but its manifest records {size}; "
            "this is an internal compiler error"
        )

    notes = list(required.notes)
    if self_verify:
        notes.extend(run_self_verify(blob))
    return CompileOutcome(bytes=blob, manifest=manifest, evidence=required, notes=notes)
