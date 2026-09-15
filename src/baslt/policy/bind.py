"""Binding a validated policy to the signals of a source.

Resolves signal references (alias, canonical name, unique leaf), applies include/exclude,
checks signal kinds, converts quantities to signal units and seconds, and computes soft weights,
the byte budget and the options used to load the selected signals.
"""

from __future__ import annotations

import difflib
import numbers
from collections.abc import Sequence
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from ..errors import PolicyError, UsageError
from ..units import Quantity, convert, describe_dimension, lookup_unit, suggest_unit, to_seconds, unknown_unit_message
from .schema import (
    EVENT_KINDS,
    OP_KINDS,
    OP_SCHEMAS,
    SOFT_WEIGHTS,
    BoundEvent,
    BoundPolicy,
    BoundReq,
    BoundSyncGroup,
    BoundTrajectory,
    Policy,
)
from .validate import IssueCollector, item_path, join_path

if TYPE_CHECKING:
    from ..signals import SignalInfo


class _Unresolved(Exception):
    pass


def _kinds_text(kinds: Sequence[str]) -> str:
    return " or ".join(kinds)


def _is_text_dtype(dtype: str) -> bool:
    return dtype == "str" or dtype.lstrip("<>|=")[:1] in ("U", "S", "O")


def _components(shape: tuple[int, ...]) -> int:
    return int(shape[1]) if len(shape) == 2 else 1


class _Binder:
    def __init__(self, policy: Policy, infos: Sequence[SignalInfo]) -> None:
        self.policy = policy
        self.issues = IssueCollector(policy.locations)
        self.infos: dict[str, SignalInfo] = {}
        for info in infos:
            self.infos.setdefault(info.name, info)
        self.by_path: dict[str, str] = {}
        self.leaves: dict[str, list[str]] = {}
        for name, info in self.infos.items():
            stripped = info.path.lstrip("/")
            if stripped not in self.infos:
                self.by_path.setdefault(stripped, name)
            self.leaves.setdefault(name.rsplit("/", 1)[-1], []).append(name)
        self.alias_map: dict[str, str | None] = {}
        self.units: dict[str, str | None] = {n: i.unit for n, i in self.infos.items()}
        self.kinds: dict[str, str] = {n: i.kind for n, i in self.infos.items()}
        self.time_refs: dict[str, str | None] = {n: i.time_ref for n, i in self.infos.items()}
        self.excluded: set[str] = set()
        self.referenced: set[str] = set()
        self.failed: set[str] = set()

    # ----- names ------------------------------------------------------------------------

    def lookup(self, ref: str) -> str:
        if ref in self.infos:
            return ref
        stripped = ref.lstrip("/")
        if stripped in self.infos:
            return stripped
        if stripped in self.by_path:
            return self.by_path[stripped]
        if "/" not in stripped:
            candidates = self.leaves.get(stripped, [])
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise _Unresolved(
                    f"ambiguous signal {ref!r}; it matches {', '.join(sorted(candidates))}. "
                    "Use the full name or declare an alias in signals.decl"
                )
        pool = list(self.alias_map) + list(self.infos) + [leaf for leaf, c in self.leaves.items() if len(c) == 1]
        close = difflib.get_close_matches(ref, list(dict.fromkeys(pool)), n=1, cutoff=0.6)
        if close:
            raise _Unresolved(f"unknown signal {ref!r}; did you mean {close[0]!r}?")
        raise _Unresolved(f"unknown signal {ref!r}")

    def resolve(self, ref: str, path: str) -> str | None:
        """Canonical name of a reference; each failing reference is reported once, at its first use."""
        if ref in self.alias_map:
            return self.alias_map[ref]
        if ref in self.failed:
            return None
        try:
            return self.lookup(ref)
        except _Unresolved as exc:
            self.failed.add(ref)
            self.issues.add(path, str(exc))
            return None

    def reference(self, ref: str, path: str) -> str | None:
        """Resolve a signal used by a requirement, event, trajectory, sync group or review."""
        canonical = self.resolve(ref, path)
        if canonical is None:
            return None
        if canonical in self.excluded:
            if ref not in self.failed:
                self.failed.add(ref)
                shown = ref if ref == canonical else f"{ref} ({canonical})"
                self.issues.add(path, f"signal {shown} is excluded by signals.exclude but referenced here")
            return None
        self.referenced.add(canonical)
        return canonical

    def time_name(self, ref: str) -> str:
        if ref in self.alias_map and self.alias_map[ref] is not None:
            return self.alias_map[ref]  # type: ignore[return-value]
        try:
            return self.lookup(ref)
        except _Unresolved:
            return ref.lstrip("/")

    # ----- declarations -----------------------------------------------------------------

    def declarations(self) -> None:
        declared_by: dict[str, str] = {}
        for alias, decl in self.policy.signals.decls.items():
            path = join_path("signals.decl", alias)
            target = decl.path if decl.path is not None else alias
            try:
                canonical = self.lookup(target)
            except _Unresolved as exc:
                self.issues.add(join_path(path, "path") if decl.path is not None else path, str(exc))
                self.alias_map[alias] = None
                continue
            self.alias_map[alias] = canonical
            if canonical in declared_by:
                self.issues.add(path, f"signal {canonical!r} is already declared by {declared_by[canonical]!r}")
                continue
            declared_by[canonical] = alias
            info = self.infos[canonical]
            if decl.unit is not None:
                self.units[canonical] = decl.unit
            if decl.time is not None:
                self.time_refs[canonical] = self.time_name(decl.time)
            if decl.kind is not None:
                self.kinds[canonical] = decl.kind
                kpath = join_path(path, "kind")
                comps = _components(info.shape)
                if decl.kind == "vector" and not (len(info.shape) == 2 and comps > 1):
                    self.issues.add(kpath, f"signal {canonical!r} has shape {info.shape}; kind vector needs shape (n, k)")
                elif decl.kind != "vector" and comps > 1:
                    self.issues.add(kpath, f"signal {canonical!r} has {comps} components; only kind vector fits")
                elif decl.kind != "discrete" and _is_text_dtype(info.dtype):
                    self.issues.add(kpath, f"signal {canonical!r} holds strings; only kind discrete fits")

    def selection(self) -> None:
        names_of: dict[str, list[str]] = {name: [name] for name in self.infos}
        for alias, canonical in self.alias_map.items():
            if canonical is not None and alias != canonical:
                names_of[canonical].append(alias)
        spec = self.policy.signals

        def matches(name: str, patterns: Sequence[str]) -> bool:
            # include and exclude are globs over canonical names only; aliases take part in soft
            # rules, where docs/policy.md asks for "canonical names and aliases".
            return any(fnmatchcase(name, p) for p in patterns)

        self.names_of = names_of
        self.excluded = {name for name in self.infos if matches(name, spec.exclude)}
        self.selected = {name for name in self.infos if name not in self.excluded and matches(name, spec.include)}

    # ----- quantities -------------------------------------------------------------------

    def to_unit(self, q: Quantity, unit: str | None, subject: str, *, delta: bool, path: str) -> float | None:
        if q.unit is None:
            return q.value
        if q.unit == unit:
            return q.value
        target = lookup_unit(unit)
        source = lookup_unit(q.unit)
        if unit is not None and target is None:
            message = (
                f"{q.text!r} does not match the unit of {subject}: {unit!r} is not a known unit, "
                f"so values must be bare numbers or use exactly {unit!r}"
            )
            hint = suggest_unit(unit)
            if hint is not None:
                message += f" (did you mean to declare unit {hint!r}?)"
            self.issues.add(path, message)
            return None
        # A mistyped or unknown unit is named before the signal's own unit is discussed, so that
        # '65 kpa' on a signal without a unit still gets its "did you mean 'kPa'?".
        if source is None:
            self.issues.add(path, unknown_unit_message(q.unit, q.text))
            return None
        if unit is None:
            self.issues.add(
                path,
                f"{q.text!r} has a unit but {subject} has no unit; write a bare number or declare its unit in signals.decl",
            )
            return None
        if source.dimension != target.dimension:
            self.issues.add(
                path,
                f"{q.text!r} is {describe_dimension(source.dimension)} but {subject} is "
                f"{describe_dimension(target.dimension)} ({unit})",
            )
            return None
        try:
            return convert(q, unit, delta=delta, path=path)
        except PolicyError as exc:  # pragma: no cover - all mismatch cases are handled above
            self.issues.absorb(exc)
            return None

    def seconds(self, q: Quantity, path: str) -> float | None:
        try:
            return to_seconds(q, path=path)
        except PolicyError as exc:  # pragma: no cover - validation already checked durations
            self.issues.absorb(exc)
            return None

    # ----- sections ---------------------------------------------------------------------

    def hard(self) -> list[BoundReq]:
        reqs: list[BoundReq] = []
        resolved: dict[str, str | None] = {}
        kind_checked: dict[tuple[str, str], bool] = {}
        for req in self.policy.hard:
            if req.signal not in resolved:
                resolved[req.signal] = self.reference(req.signal, join_path("hard", req.signal))
            canonical = resolved[req.signal]
            if canonical is None:
                continue
            key = (req.signal, req.op)
            if key not in kind_checked:
                kind = self.kinds[canonical]
                allowed = OP_KINDS[req.op]
                kind_checked[key] = kind in allowed
                if kind not in allowed:
                    self.issues.add(
                        join_path(join_path("hard", req.signal), req.op),
                        f"{req.op} needs a {_kinds_text(allowed)} signal, but {req.signal} is {kind}",
                    )
            if not kind_checked[key]:
                continue
            params: dict[str, float | str | int] = {}
            ok = True
            subject = f"signal {req.signal}"
            for name, value in req.params.items():
                param = OP_SCHEMAS[req.op][name]
                ppath = join_path(req.id, name)
                if param.kind in ("value", "delta"):
                    out = self.to_unit(value, self.units[canonical], subject, delta=param.kind == "delta", path=ppath)  # type: ignore[arg-type]
                elif param.kind == "duration":
                    out = self.seconds(value, ppath)  # type: ignore[arg-type]
                else:
                    out = value  # type: ignore[assignment]
                if out is None:
                    ok = False
                else:
                    params[name] = out
            if ok:
                reqs.append(BoundReq(id=req.id, signal=canonical, op=req.op, params=params, severity=req.severity))
        return reqs

    def events(self) -> list[tuple[BoundEvent, bool]]:
        out: list[tuple[BoundEvent, bool]] = []
        for ev in self.policy.events:
            base = join_path("events", ev.name)
            when = join_path(base, "when")
            canonical = self.reference(ev.signal, join_path(when, "signal"))
            keep_signals: list[str] = []
            keep_ok = True
            if ev.signals is not None:
                for i, ref in enumerate(ev.signals):
                    c = self.reference(ref, item_path(join_path(join_path(base, "keep"), "signals"), i))
                    if c is None:
                        keep_ok = False
                    elif c not in keep_signals:
                        keep_signals.append(c)
            before = self.seconds(ev.before, join_path(base, "keep.before"))
            after = self.seconds(ev.after, join_path(base, "keep.after"))
            if canonical is None or not keep_ok or before is None or after is None:
                continue
            kind = self.kinds[canonical]
            allowed = EVENT_KINDS[ev.trigger]
            tpath = join_path(when, ev.trigger)
            if kind not in allowed:
                self.issues.add(tpath, f"{ev.trigger} needs a {_kinds_text(allowed)} signal, but {ev.signal} is {kind}")
                continue
            if ev.trigger == "equals":
                value: float | str | int | None = ev.value  # type: ignore[assignment]
                hysteresis: float | None = 0.0
                debounce: float | None = 0.0
            else:
                subject = f"signal {ev.signal}"
                unit = self.units[canonical]
                value = self.to_unit(ev.value, unit, subject, delta=False, path=tpath)  # type: ignore[arg-type]
                hysteresis = self.to_unit(ev.hysteresis, unit, subject, delta=True, path=join_path(when, "hysteresis"))  # type: ignore[arg-type]
                debounce = self.seconds(ev.debounce, join_path(when, "debounce"))  # type: ignore[arg-type]
            if value is None or hysteresis is None or debounce is None:
                continue
            bound = BoundEvent(
                name=ev.name,
                signal=canonical,
                trigger=ev.trigger,
                value=value,
                hysteresis=hysteresis,
                debounce=debounce,
                occurrence=ev.occurrence,
                expect=ev.expect,
                before=before,
                after=after,
                signals=keep_signals,
                severity=ev.severity,
            )
            out.append((bound, ev.signals is None))
        return out

    def trajectories(self) -> list[BoundTrajectory]:
        out: list[BoundTrajectory] = []
        for tr in self.policy.trajectories:
            base = join_path("trajectories", tr.name)
            ppath = join_path(base, "position")
            if isinstance(tr.position, str):
                canonical = self.reference(tr.position, ppath)
                if canonical is None:
                    continue
                info = self.infos[canonical]
                kind = self.kinds[canonical]
                if kind != "vector" or len(info.shape) != 2 or info.shape[1] != 3:
                    self.issues.add(
                        ppath,
                        f"position signal {tr.position} must be a vector signal with shape (n, 3), "
                        f"got {kind} with shape {info.shape}",
                    )
                    continue
                position = [canonical]
                unit = self.units[canonical]
                subject = f"position signal {tr.position}"
            else:
                position = []
                for i, ref in enumerate(tr.position):
                    c = self.reference(ref, item_path(ppath, i))
                    if c is None:
                        continue
                    kind = self.kinds[c]
                    if kind != "continuous" or _components(self.infos[c].shape) != 1:
                        self.issues.add(item_path(ppath, i), f"position component {ref} must be a continuous scalar signal, got {kind}")
                        continue
                    position.append(c)
                if len(position) != 3:
                    continue
                if len(set(position)) != 3:
                    self.issues.add(ppath, f"position components must be three different signals, got {', '.join(position)}")
                    continue
                counts = [self.infos[c].n for c in position]
                clocks = [self.time_refs[c] for c in position]
                if len(set(counts)) > 1:
                    self.issues.add(
                        ppath,
                        f"position components must share one clock, but they have {', '.join(map(str, counts))} samples",
                    )
                    continue
                # Only the known clocks are compared: one unknown time_ref must not hide a real
                # mismatch between the other two components.
                if len({c for c in clocks if c is not None}) > 1:
                    shown = ", ".join("unknown" if c is None else c for c in clocks)
                    self.issues.add(ppath, f"position components must share one clock, but they use {shown}")
                    continue
                units = [self.units[c] for c in position]
                if len(set(units)) > 1:
                    self.issues.add(
                        ppath, f"position components have different units: {', '.join(repr(u) for u in units)}"
                    )
                    continue
                unit = units[0]
                subject = "position [" + ", ".join(tr.position) + "]"
            unit_def = lookup_unit(unit)
            if unit_def is not None and unit_def.dimension != "length":
                self.issues.add(
                    ppath, f"{subject} is {describe_dimension(unit_def.dimension)} ({unit}) but a length is required"
                )
                continue
            eps = self.to_unit(tr.max_position_error, unit, subject, delta=True, path=join_path(base, "max_position_error"))
            max_time = None if tr.max_time_error is None else self.seconds(tr.max_time_error, join_path(base, "max_time_error"))
            linked: list[str] = []
            linked_ok = True
            for i, ref in enumerate(tr.linked):
                c = self.reference(ref, item_path(join_path(base, "linked"), i))
                if c is None:
                    linked_ok = False
                elif c not in linked:
                    linked.append(c)
            if eps is None or not linked_ok or (tr.max_time_error is not None and max_time is None):
                continue
            out.append(
                BoundTrajectory(
                    name=tr.name,
                    position=position,
                    max_position_error=eps,
                    max_time_error=max_time,
                    linked=linked,
                    unit=unit,
                )
            )
        return out

    def sync_groups(self) -> list[BoundSyncGroup]:
        out: list[BoundSyncGroup] = []
        for group in self.policy.sync_groups:
            mpath = join_path(join_path("sync_groups", group.name), "members")
            members: list[str] = []
            first_ref: dict[str, str] = {}
            ok = True
            for i, ref in enumerate(group.members):
                c = self.reference(ref, item_path(mpath, i))
                if c is None:
                    ok = False
                    continue
                if c in first_ref:
                    self.issues.add(item_path(mpath, i), f"members {first_ref[c]!r} and {ref!r} are the same signal ({c})")
                    ok = False
                    continue
                first_ref[c] = ref
                members.append(c)
            if ok:
                out.append(BoundSyncGroup(name=group.name, members=members))
        return out

    def review_list(self, refs: Sequence[str], path: str) -> list[str]:
        out: list[str] = []
        for i, ref in enumerate(refs):
            c = self.reference(ref, item_path(path, i))
            if c is not None and c not in out:
                out.append(c)
        return out

    def soft(self, included: Sequence[str]) -> dict[str, tuple[int, int | None]]:
        weights: dict[str, tuple[int, int | None]] = {}
        for name in included:
            weight: tuple[int, int | None] = (SOFT_WEIGHTS["medium"], None)
            for rule in self.policy.soft:
                if any(fnmatchcase(n, rule.match) for n in self.names_of[name]):
                    weight = (SOFT_WEIGHTS[rule.priority], rule.max_points)
                    break
            weights[name] = weight
        return weights


def bind_policy(
    policy: Policy, infos: Sequence[SignalInfo], *, max_bytes_override: int | None = None
) -> BoundPolicy:
    """Bind a validated policy to a source's signals. Raises PolicyError with every problem found."""
    if max_bytes_override is not None:
        if isinstance(max_bytes_override, bool) or not isinstance(max_bytes_override, numbers.Integral) or max_bytes_override <= 0:
            raise UsageError(f"the maximum size must be a positive number of bytes, got {max_bytes_override!r}")

    binder = _Binder(policy, infos)
    binder.declarations()
    binder.selection()
    reqs = binder.hard()
    events = binder.events()
    trajectories = binder.trajectories()
    sync_groups = binder.sync_groups()
    thumbnails = binder.review_list(policy.review.thumbnails, "review.thumbnails")
    kpis = binder.review_list(policy.review.kpis, "review.kpis")
    binder.issues.raise_if_any()

    included = [name for name in binder.infos if name in binder.selected or name in binder.referenced]
    bound_events: list[BoundEvent] = []
    for event, keep_all in events:
        if keep_all:
            event.signals = list(included)
        bound_events.append(event)

    reqs_by_signal: dict[str, list[BoundReq]] = {name: [] for name in included}
    for req in reqs:
        reqs_by_signal[req.signal].append(req)

    if max_bytes_override is not None:
        max_bytes, budget_source = int(max_bytes_override), "cli"
    elif policy.artifact.max_bytes is not None:
        max_bytes, budget_source = policy.artifact.max_bytes, "policy"
    else:
        max_bytes, budget_source = None, "none"

    signals = policy.signals
    time_hints: dict[str, str] = {}
    decl_units: dict[str, str] = {}
    decl_kinds: dict[str, str] = {}
    for alias, decl in signals.decls.items():
        canonical = binder.alias_map.get(alias)
        if canonical is None:
            continue
        if decl.time is not None:
            time_hints[canonical] = binder.time_name(decl.time)
        if decl.unit is not None:
            decl_units[canonical] = decl.unit
        if decl.kind is not None:
            decl_kinds[canonical] = decl.kind
    time_unit = lookup_unit(signals.time_unit)
    load_options = {
        "time_hints": time_hints,
        "global_time": None if signals.time is None else binder.time_name(signals.time),
        "time_scale": time_unit.factor if time_unit is not None else 1.0,
        "on_non_monotonic": signals.on_non_monotonic,
        "units": decl_units,
        "kinds": decl_kinds,
    }

    return BoundPolicy(
        policy=policy,
        included=included,
        aliases={alias: c for alias, c in binder.alias_map.items() if c is not None},
        reqs=reqs,
        reqs_by_signal=reqs_by_signal,
        events=bound_events,
        trajectories=trajectories,
        sync_groups=sync_groups,
        soft=binder.soft(included),
        max_bytes=max_bytes,
        budget_source=budget_source,
        thumbnails=thumbnails,
        kpis=kpis,
        units={name: binder.units[name] for name in included},
        kinds={name: binder.kinds[name] for name in included},
        load_options=load_options,
    )
