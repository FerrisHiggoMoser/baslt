"""Compile one run into `.baslt` container bytes.

    evaluate hard -> [add soft samples -> close sync -> encode -> index.json, policy.json -> manifest (size fixed
    point)] repeated by the budget search -> reconstruction errors -> write_zip -> self-verify

Every signal gets one `s/<k>` member holding its `v`, `idx` and `roles` arrays back to back, and points
its `t` descriptor at a `t/<j>` member. Signals whose retained timestamps are bit-identical share that
member, which is what makes a clock stored once for a whole run.

Hard requirements, events, sync groups and the implicit `extent`/`gap` retention form the base artifact. When it
does not fit the budget the compile is infeasible and reports an itemized breakdown. Otherwise the rest of the
budget goes to preview samples ranked by `reduce.rank.preview_order`: every signal with soft weight `w` takes the
first `min(cap, floor(s * w))` of its ranking for one common scale `s`, and the search keeps the largest scale
whose complete artifact (sync propagation included) was measured to fit. Compressed members are cached by content,
so a trial only compresses the members that changed.
"""

from __future__ import annotations

import hashlib
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
from ..manifest import NO_DIGEST, EventEntry, RequirementEntry, SignalBudget, SyncEntry, build_manifest
from ..policy.schema import OP_SCHEMAS
from ..reduce.rank import preview_order
from .required import RequiredResult, SignalPlan, close, evaluate_hard

if TYPE_CHECKING:
    from ..hashing import HashInfo
    from ..policy.schema import BoundPolicy
    from ..signals import Run

# Supported self-verification entry points, tried in order.
CHECK_ENTRY_POINTS: tuple[str, ...] = ("verify_artifact", "check_artifact", "run_checks", "verify")
FAILED_STATUSES = frozenset({"fail", "failed", "error"})

# Preview samples per signal without a budget, by soft weight (high 4, medium 2, low 1).
DEFAULT_POINTS: dict[int, int] = {4: 16384, 2: 4096, 1: 1024}
# A budget of B bytes never needs more than this many preview candidates per signal.
CAP_SAMPLES_PER_BYTE = 16
MAX_EVALUATIONS = 16


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
    digest: HashInfo | None = None,
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
        "digest": NO_DIGEST if digest is None else digest.to_json(),
    }
    source = "artifact.max_size" if budget_source == "policy" else "--max-size"
    return InfeasibleBudget(
        f"the hard requirements need {size} bytes but {source} allows {max_bytes}; "
        f"{size - max_bytes} bytes too many. Every retained sample is required by a hard contract, an event or a "
        "sync group, so no smaller artifact satisfies this policy; baslt explain suggests what to relax",
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


@dataclass(slots=True)
class _Assembly:
    """One candidate artifact: everything needed to write it, plus its exact size."""

    required: RequiredResult
    encoded: list[_Encoded]
    rows: list[SignalBudget]
    members: list[Member]
    manifest: dict
    size: int
    data_bytes: int


class _Assembler:
    """Builds candidate artifacts for one run, caching compressed members across budget trials."""

    def __init__(self, run: Run, bound: BoundPolicy, digest: HashInfo | None) -> None:
        self.run = run
        self.bound = bound
        self.digest = digest
        self.codec = bound.policy.artifact.codec
        self.method = CODEC_METHODS[self.codec]
        self.level = DEFAULT_LEVEL
        self._sizes: dict[tuple[str, bytes], int] = {}
        # Set from the artifact without soft samples; later trials are accounted against it.
        self.base_data_bytes: int | None = None
        self.base_counts: dict[str, int] | None = None

    def _csize(self, member: Member) -> int:
        key = (member.name, hashlib.sha1(member.data).digest() + len(member.data).to_bytes(8, "little"))
        size = self._sizes.get(key)
        if size is None:
            size = len(compress_member(member, level=self.level)[1])
            self._sizes[key] = size
        return size

    def build(self, required: RequiredResult, *, errors: dict[str, float] | None = None) -> _Assembly:
        bound = self.bound
        encoded = [_encode_signal(required.signals[name]) for name in bound.included]
        index, signal_members, time_members, owned = _index_and_members(encoded, bound, self.method)
        data_members = signal_members + time_members
        csize = {member.name: self._csize(member) for member in data_members}

        header_bytes = header_json(self.codec, self.level)
        index_member = Member(MEMBER_INDEX, canonical_json(index), self.method)
        policy_member = Member(MEMBER_POLICY, canonical_json(_policy_json(bound)), self.method)
        fixed = [
            (MEMBER_HEADER, len(header_bytes)),
            (MEMBER_INDEX, self._csize(index_member)),
            (MEMBER_POLICY, self._csize(policy_member)),
            *csize.items(),
        ]

        rows: list[SignalBudget] = []
        for item in encoded:
            name = item.plan.name
            hard = item.n if self.base_counts is None else self.base_counts[name]
            rows.append(SignalBudget(
                name=name,
                retained=item.n,
                hard=hard,
                soft=item.n - hard,
                bytes=sum(csize[member] for member in owned[name]),
                soft_max_abs_err=(errors or {}).get(name, 0.0),
            ))
        data_bytes = sum(row.bytes for row in rows)
        base_bytes = data_bytes if self.base_data_bytes is None else self.base_data_bytes
        discretionary = max(0, data_bytes - base_bytes)

        requirements = [
            RequirementEntry(id=r.id, signal=r.signal, op=r.op, severity=r.severity, status=r.status,
                             evidence=r.evidence)
            for r in required.requirements
        ]
        events = [
            EventEntry(name=r.name, signal=r.signal, severity=r.severity, status=r.status, evidence=r.evidence)
            for r in required.events
        ]
        sync_groups = [SyncEntry(name=r.name, status=r.status, evidence=r.evidence) for r in required.sync_groups]
        manifest, size = build_manifest(
            source=self.run.meta,
            digest=self.digest,
            policy=bound.policy,
            requirements=requirements,
            signals=rows,
            samples_total=sum(int(sig.n) for sig in self.run.signals.values()),
            source_signals=len(self.run.signals),
            size_of=lambda length: zip_size([*fixed, (MEMBER_MANIFEST, length)]),
            max_bytes=bound.max_bytes,
            budget_source=bound.budget_source,
            codec=self.codec,
            level=self.level,
            events=events,
            sync_groups=sync_groups,
            discretionary_bytes=discretionary,
        )
        members = [
            Member(MEMBER_HEADER, header_bytes, self.method),
            index_member,
            Member(MEMBER_MANIFEST, canonical_json(manifest), self.method),
            policy_member,
            *data_members,
        ]
        return _Assembly(required=required, encoded=encoded, rows=rows, members=members, manifest=manifest,
                         size=size, data_bytes=data_bytes)


# --------------------------------------------------------------------------------------------------------------
# Soft layer and budget search


def _soft_plan(run: Run, bound: BoundPolicy) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    """Preview rankings, point caps and weights of every signal that can take soft samples."""
    rankings: dict[str, np.ndarray] = {}
    caps: dict[str, int] = {}
    weights: dict[str, int] = {}
    for name in bound.included:
        weight, max_points = bound.soft.get(name, (0, None))
        sig = run.signals[name]
        if weight <= 0 or sig.n == 0:
            continue
        if bound.max_bytes is None:
            cap = DEFAULT_POINTS.get(weight, DEFAULT_POINTS[2])
        else:
            cap = CAP_SAMPLES_PER_BYTE * bound.max_bytes
        if max_points is not None:
            cap = min(cap, int(max_points))
        cap = min(cap, sig.n)
        if cap <= 0:
            continue
        rankings[name] = preview_order(sig.v, cap)
        caps[name] = int(rankings[name].shape[0])
        weights[name] = int(weight)
    return rankings, caps, weights


def _points(scale: float, caps: dict[str, int], weights: dict[str, int]) -> dict[str, int]:
    return {name: min(cap, int(np.floor(scale * weights[name]))) for name, cap in caps.items()}


def _reconstruction_errors(encoded: list[_Encoded]) -> dict[str, float]:
    """Largest absolute difference between each source signal and its reconstruction from the retained samples.

    The reconstruction is the one of docs/contracts.md, a function of time: the retained value at a retained
    timestamp (the later one when timestamps repeat), linear interpolation `a + w * (b - a)` between consecutive
    retained timestamps, NaN across a non-finite retained value; discrete signals hold the previous retained value.
    Retained samples count as exact and non-finite differences are ignored.
    """
    errors: dict[str, float] = {}
    for item in encoded:
        sig = item.plan.signal
        idx = item.idx
        if sig.n == 0 or idx.shape[0] == 0:
            errors[item.plan.name] = 0.0
            continue
        t_kept = sig.t[idx]
        left = np.searchsorted(t_kept, sig.t, side="right") - 1
        left = np.maximum(left, 0)
        right = np.minimum(left + 1, idx.shape[0] - 1)
        knot = t_kept[left] == sig.t
        hold = sig.kind == "discrete"
        if not hold:
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = (sig.t - t_kept[left]) / (t_kept[right] - t_kept[left])
            weight[knot] = 0.0
        values = sig.v.reshape(sig.n, -1)
        worst = 0.0
        for k in range(values.shape[1]):
            column = values[:, k].astype(np.float64)
            kept = column[idx]
            a = kept[left]
            with np.errstate(invalid="ignore", over="ignore"):
                approx = a if hold else np.where(knot, a, a + weight * (kept[right] - a))
                diff = np.abs(column - approx)
            diff[idx] = 0.0
            diff = diff[np.isfinite(diff)]
            if diff.size:
                worst = max(worst, float(diff.max()))
        errors[item.plan.name] = worst
    return errors


def _search(
    trial, base: _Assembly, caps: dict[str, int], weights: dict[str, int], max_bytes: int
) -> tuple[_Assembly, int]:
    """The largest measured preview scale whose artifact fits `max_bytes`, and the number of trials built.

    Trial sizes are monotone in the scale. Each step interpolates the next scale from the two measured bracket
    sizes, treating bytes as linear in the number of preview samples, and falls back to bisection when one side
    has moved twice in a row. The search stops once the accepted artifact uses 99.5% of the budget, the bracket
    is narrower than one preview sample of the highest-weight signal, or MAX_EVALUATIONS trials were built.
    """
    cap_arr = np.array([caps[name] for name in caps], dtype=np.int64)
    weight_arr = np.array([weights[name] for name in caps], dtype=np.float64)

    def total(scale: float) -> int:
        return int(np.minimum(cap_arr, np.floor(scale * weight_arr)).sum())

    top_scale = float(np.max(cap_arr / weight_arr))
    top = trial(_points(top_scale, caps, weights))
    evaluations = 1
    if top.size <= max_bytes:
        return top, evaluations

    def scale_for(picks: float, lo: float, hi: float) -> float:
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if total(mid) >= picks:
                hi = mid
            else:
                lo = mid
        return hi

    step = 1.0 / float(weight_arr.max())
    lo_s, lo_r, hi_s, hi_r = 0.0, base, top_scale, top
    lo_points, hi_points = _points(lo_s, caps, weights), _points(hi_s, caps, weights)
    streak = 0  # positive: low moved last, negative: high moved last
    for _ in range(4 * MAX_EVALUATIONS):
        if evaluations >= MAX_EVALUATIONS or hi_s - lo_s <= max(step, 0.005 * hi_s):
            break
        if lo_r.size >= 0.995 * max_bytes:
            break
        if abs(streak) >= 2:
            scale = (lo_s + hi_s) / 2.0
        else:
            lo_p, hi_p = total(lo_s), total(hi_s)
            fraction = (max_bytes - lo_r.size) / max(hi_r.size - lo_r.size, 1)
            fraction = min(max(fraction, 0.02), 0.98)
            scale = scale_for(lo_p + fraction * (hi_p - lo_p), lo_s, hi_s)
        if not lo_s < scale < hi_s:
            scale = (lo_s + hi_s) / 2.0
        points = _points(scale, caps, weights)
        if points == lo_points:
            lo_s = scale
            continue
        if points == hi_points:
            hi_s = scale
            continue
        result = trial(points)
        evaluations += 1
        if result.size <= max_bytes:
            lo_s, lo_r, lo_points = scale, result, points
            streak = streak + 1 if streak > 0 else 1
        else:
            hi_s, hi_r, hi_points = scale, result, points
            streak = streak - 1 if streak < 0 else -1
    return lo_r, evaluations


def minimum_size(run: Run, bound: BoundPolicy, *, digest: HashInfo | None = None) -> int:
    """Size in bytes of the artifact `bound` allows with no preview samples; nothing is written or verified.

    This is the smallest artifact except when preview samples let very short signals share a clock member.
    """
    hard = evaluate_hard(run, bound)
    return _Assembler(run, bound, digest).build(close(hard, bound)).size


def compile_run(
    run: Run, bound: BoundPolicy, *, digest: HashInfo | None, self_verify: bool = True
) -> CompileOutcome:
    """Compile `run` under `bound` into container bytes.

    Hard requirements, events and sync groups are always kept. The rest of `bound.max_bytes` is spent on preview
    samples: every trial size is measured on the complete artifact, soft samples and sync propagation included, and
    only a measured size within the budget is accepted, so the result never exceeds it. Without a budget, every
    signal gets a default number of preview samples by priority.

    `digest` is the source digest recorded in the manifest (None writes the "none" digest). Pass
    `self_verify=False` to skip the independent verification of the compiled bytes.

    Raises InfeasibleBudget when the hard requirements alone do not fit `bound.max_bytes`, SelfVerifyFailed when
    the artifact fails its own verification, and UsageError or PolicyError for a policy this build cannot compile.
    """
    assembler = _Assembler(run, bound, digest)
    hard = evaluate_hard(run, bound)
    rankings, caps, weights = _soft_plan(run, bound)

    kept: dict[str, np.ndarray] = {}

    def trial(points: dict[str, int]) -> _Assembly:
        # Samples the base artifact keeps anyway are not previews: they keep their roles, and their bytes.
        soft = {}
        for name, count in points.items():
            picks = rankings[name][:count]
            soft[name] = picks[~np.isin(picks, kept[name], assume_unique=True)] if count > 0 else picks[:0]
        return assembler.build(close(hard, bound, soft))

    base = trial({})
    kept.update((item.plan.name, item.idx) for item in base.encoded)
    assembler.base_data_bytes = base.data_bytes
    assembler.base_counts = {row.name: row.retained for row in base.rows}
    max_bytes = bound.max_bytes

    if not caps:
        chosen = base
    elif max_bytes is None:
        chosen = trial(dict(caps))
    elif base.size <= max_bytes:
        chosen, _ = _search(trial, base, caps, weights, max_bytes)
    else:
        # More samples can make an artifact smaller when they give signals identical timestamps, which then share
        # one clock member; that only matters for very short signals, so one full-preview trial settles it.
        chosen = trial(dict(caps))
    if max_bytes is not None and chosen.size > max_bytes:
        raise _infeasible(base.required, base.encoded, base.rows, base.size, max_bytes, bound.budget_source,
                          assembler.method, assembler.level, digest)

    # The errors are formatted at a fixed width, so recording them leaves the size unchanged.
    final = assembler.build(chosen.required, errors=_reconstruction_errors(chosen.encoded))
    if final.size != chosen.size:
        raise CompileError(
            f"the artifact size changed from {chosen.size} to {final.size} bytes when the reconstruction errors "
            "were recorded; this is an internal compiler error"
        )
    blob = write_zip(final.members, level=assembler.level)
    if len(blob) != final.size:
        raise CompileError(
            f"the artifact is {len(blob)} bytes but its manifest records {final.size}; "
            "this is an internal compiler error"
        )

    notes = list(final.required.notes)
    if self_verify:
        notes.extend(run_self_verify(blob))
    return CompileOutcome(bytes=blob, manifest=final.manifest, evidence=final.required, notes=notes)
