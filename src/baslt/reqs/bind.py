"""Binding a requirements table to a source: kinds, expressions and limits in the units of what they check.

Binding happens once per source layout, before any data is loaded, so its result also says which signals a check
needs. A requirement that cannot be bound keeps its problems (with cell locations) and is reported as ERROR; the
others are checked as usual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import Issue
from ..units import Quantity
from .config import Config
from .events import split_reference
from .expr import UNKNOWN, B, Binder, BoundExpr, ExprError, TypeInfo
from .model import AGGREGATE_KINDS, BOUND_KINDS, Bound, Case, Requirement, RequirementSet
from .units_ext import UnitsError

__all__ = ["BoundCase", "BoundLimit", "BoundRequirement", "BoundSet", "bind_requirements", "lint_against_source",
           "open_run_source"]

SCALAR_KINDS = ("value", "duration", "event", *AGGREGATE_KINDS)


@dataclass
class BoundSide:
    node: B
    inclusive: bool
    text: str


@dataclass
class BoundLimit:
    kind: str  # upper, lower, range, equal, not_equal
    lower: BoundSide | None
    upper: BoundSide | None
    text: str

    @property
    def series(self) -> bool:
        return any(side is not None and side.node.type.series for side in (self.lower, self.upper))


@dataclass
class BoundCase:
    case: Case
    when: BoundExpr | None = None
    applies_to: BoundExpr | None = None
    limit: BoundLimit | None = None
    margin_abs: float | None = None
    margin_rel: float | None = None
    tol_seconds: float = 0.0
    tol_samples: int | None = None
    count: int | None = None
    issues: list[Issue] = field(default_factory=list)


@dataclass
class BoundRequirement:
    req: Requirement
    kind: str = ""
    check: BoundExpr | None = None
    event: str | None = None  # the event reference of an event check
    unit: str | None = None
    cases: list[BoundCase] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    grid: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and not any(case.issues for case in self.cases)

    @property
    def all_issues(self) -> list[Issue]:
        return [*self.issues, *(issue for case in self.cases for issue in case.issues)]

    def signals(self) -> list[str]:
        out: list[str] = []
        exprs = [self.check] + [c.when for c in self.cases] + [c.applies_to for c in self.cases]
        for expr in exprs:
            for name in (expr.signals if expr is not None else []):
                if name not in out:
                    out.append(name)
        for case in self.cases:
            for side in (case.limit.lower, case.limit.upper) if case.limit is not None else ():
                if side is not None:
                    for name in _signals_of(side.node):
                        if name not in out:
                            out.append(name)
        return out

    def events(self) -> set[str]:
        names: set[str] = set()
        if self.event:
            names.add(split_reference(self.event)[0])
        for expr in [self.check] + [c.when for c in self.cases]:
            if expr is not None:
                names |= expr.events
        return names

    def grid_names(self) -> list[str]:
        """The clock order of docs/requirements.md: the check's signals, then the cases' conditions."""
        out = list(self.check.signals) if self.check is not None else []
        for case in self.cases:
            for name in (case.when.signals if case.when is not None else []):
                if name not in out:
                    out.append(name)
        for name in self.signals():
            if name not in out:
                out.append(name)
        return out


@dataclass
class BoundSet:
    requirements: list[BoundRequirement]
    reqset: RequirementSet
    signals: list[str]  # every signal any check, event or condition needs
    issues: list[Issue] = field(default_factory=list)  # problems of the mapping on this source


def _signals_of(node: B) -> list[str]:
    out: list[str] = []

    def walk(n: B) -> None:
        if n.op == "sig" and n.value[0] not in out:
            out.append(n.value[0])
        for arg in n.args:
            if isinstance(arg, B):
                walk(arg)

    walk(node)
    return out


def _text_literal(bound: Bound) -> str | None:
    """The text of a limit such as `== heavy` or `== 'heavy'`, compared with a text value."""
    if bound.expr is None:
        return None
    text = bound.expr.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text if text.isidentifier() else None


class _RequirementBinder:
    def __init__(self, binder: Binder, config: Config) -> None:
        self.binder = binder
        self.config = config
        self.units = config.units

    def issue(self, req: Requirement, field_name: str, message: str, case: Case | None = None) -> Issue:
        location = None
        if case is not None:
            location = case.cells.get(field_name) or case.loc.text
        return Issue(path=f"{req.id}.{field_name}", message=message, location=location or req.loc.text)

    def bind(self, req: Requirement) -> BoundRequirement:
        out = BoundRequirement(req=req, grid=req.grid or None)
        if req.issues:
            out.issues = list(req.issues)
            return out
        kind = req.kind
        first = req.cases[0] if req.cases else None
        try:
            if kind == "event" or (kind is None and self._is_event(req.check)):
                kind = "event"
                name, occurrence = split_reference(req.check)
                if name not in self.config.events:
                    raise ExprError(f"unknown event {name!r}; define it under events:")
                out.event = req.check.strip()
                out.unit = "s"
            else:
                text = req.check
                if kind in AGGREGATE_KINDS:
                    text = f"{kind}({text})"
                out.check = self.binder.bind(text, role="check")
                out.notes.extend(out.check.notes)
                kind = self._kind(req, kind, out.check.type)
                if kind == "duration":
                    out.unit = "s"
                else:
                    out.unit = out.check.type.unit
        except (ExprError, UnitsError) as exc:
            out.issues.append(self.issue(req, "check", str(exc), first))
            out.kind = kind or "unknown"
            return out
        out.kind = kind
        for case in req.cases:
            out.cases.append(self._case(req, out, case))
        return out

    def _is_event(self, text: str) -> bool:
        name, _ = split_reference(text)
        if name not in self.config.events:
            return False
        if name in self.config.signals or name in self.config.conditions:
            return False
        try:
            self.binder.index.lookup(name)
        except LookupError:
            return True
        return False

    def _kind(self, req: Requirement, kind: str | None, t: TypeInfo) -> str:
        limits = [case.limit for case in req.cases if case.limit is not None]
        if kind in AGGREGATE_KINDS:
            if t.shape != "scalar":
                raise ExprError(f"{kind} needs a single-valued signal")
            return "value"
        if kind in BOUND_KINDS:
            if t.dtype == "bool":
                raise ExprError(f"an {kind} check needs a signal or a number expression, but {req.check!r} is a "
                                "condition; use the assert or duration type")
            if t.shape != "series":
                raise ExprError(f"an {kind} check needs a signal; {req.check!r} is a single number, so use the value "
                                "type")
            if t.dtype not in ("num", "enum"):
                raise ExprError(f"an {kind} check needs numbers")
            return "bound"
        if kind == "assert":
            if t.dtype != "bool" or t.shape != "series":
                raise ExprError("an assert check needs a condition on signals, such as mode == 'COAST'")
            return "assert"
        if kind == "duration":
            if t.dtype != "bool" or t.shape != "series":
                raise ExprError("a duration check needs a condition on signals, such as q > 60 kPa")
            return "duration"
        if kind == "value":
            if t.shape != "scalar":
                raise ExprError(f"a value check needs a single number, such as max({req.check}) or "
                                f"at('EVENT', {req.check})")
            return "value"
        # no type given: infer it
        if t.dtype == "bool" and t.shape == "series":
            return "duration" if limits else "assert"
        if t.shape == "scalar":
            return "value"
        if t.dtype in ("num", "enum"):
            return "bound"
        raise ExprError(f"cannot tell what kind of check {req.check!r} is; set its Type")

    # ----- cases ------------------------------------------------------------------------------------------

    def _case(self, req: Requirement, out: BoundRequirement, case: Case) -> BoundCase:
        bound = BoundCase(case=case, tol_seconds=case.tolerance.seconds, tol_samples=case.tolerance.samples,
                          count=case.count)
        try:
            if case.when:
                bound.when = self.binder.bind(case.when, role="when")
                if bound.when.type.dtype != "bool":
                    raise ExprError(f"When {case.when!r} is not a condition")
                out.notes.extend(bound.when.notes)
        except (ExprError, UnitsError) as exc:
            bound.issues.append(self.issue(req, "when", str(exc), case))
        try:
            if case.applies_to:
                bound.applies_to = self.binder.bind(case.applies_to, role="applies_to")
                if bound.applies_to.type.dtype != "bool":
                    raise ExprError(f"Applies to {case.applies_to!r} is not a condition")
        except (ExprError, UnitsError) as exc:
            bound.issues.append(self.issue(req, "applies_to", str(exc), case))
        try:
            bound.limit = self._limit(req, out, case)
        except (ExprError, UnitsError) as exc:
            bound.issues.append(self.issue(req, "limit", str(exc), case))
        try:
            self._margin(out, case, bound)
        except (ExprError, UnitsError) as exc:
            bound.issues.append(self.issue(req, "margin", str(exc), case))
        if out.kind == "event" and bound.count is None:
            bound.count = 1
        return bound

    def _limit(self, req: Requirement, out: BoundRequirement, case: Case) -> BoundLimit | None:
        spec = case.limit
        kind = out.kind
        if spec is None:
            if kind in ("assert", "event"):
                return None
            raise ExprError(f"a {req.kind or kind} check needs a limit")
        if kind == "assert":
            raise ExprError("an assert check takes no limit; its Check is the condition")
        if spec.kind == "bare":
            if kind == "bound":
                spec = spec.with_direction(req.kind or "upper")
                if req.kind == "range":
                    raise ExprError(f"a range check needs a range, such as [{spec.text}, ...]")
            else:
                raise ExprError(f"write the limit with a comparison or a range, such as '<= {spec.text}'")
        if kind == "bound":
            wanted = req.kind if req.kind in BOUND_KINDS else None
            if wanted is not None and spec.kind != wanted:
                names = {"upper": "an upper limit", "lower": "a lower limit", "range": "a range",
                         "equal": "an equality", "not_equal": "an inequality"}
                raise ExprError(f"an {wanted} check cannot use {names[spec.kind]} {spec.text!r}")
            if spec.kind in ("equal", "not_equal"):
                raise ExprError("a limit on a signal is an upper limit, a lower limit or a range; use an assert "
                                "check for equality")
        unit = out.unit
        scalar_only = kind != "bound"
        if spec.kind in ("equal", "not_equal") and out.check is not None \
                and out.check.type.dtype in ("any", "str", "enum"):
            text = _text_literal(spec.lower or spec.upper)
            if text is not None:
                side = BoundSide(B("str", (), text, TypeInfo("scalar", "str")), True, text)
                return BoundLimit(spec.kind, side, None, spec.text)
        lower = self._side(spec.lower, unit, scalar_only) if spec.lower is not None else None
        upper = self._side(spec.upper, unit, scalar_only) if spec.upper is not None else None
        if spec.kind in ("equal", "not_equal"):
            lower, upper = lower or upper, None
        return BoundLimit(spec.kind, lower, upper, spec.text)

    def _side(self, bound: Bound, unit: str | None, scalar_only: bool) -> BoundSide:
        text = bound.describe()
        if bound.quantity is not None:
            node = self._quantity(bound.quantity, unit)
        elif bound.curve is not None:
            arg = f", {bound.arg}" if bound.arg else ""
            expr = self.binder.bind(f"curve({bound.curve}{arg})", role="limit")
            node = self._in_unit(expr.root, unit, text)
        else:
            expr = self.binder.bind(bound.expr, role="limit")
            if expr.type.dtype not in ("num", "any"):
                raise ExprError(f"the limit {bound.expr!r} is not a number")
            node = self._in_unit(expr.root, unit, text)
        if scalar_only and node.type.series:
            raise ExprError(f"the limit {text!r} changes over time, but this check compares one number; use a "
                            "single-number limit")
        return BoundSide(node, bound.inclusive, text)

    def _quantity(self, q: Quantity, unit: str | None) -> B:
        if q.unit is not None and unit == UNKNOWN:
            raise ExprError(f"{q.text!r} has a unit, but the check's unit cannot be worked out; declare it with "
                            "as_unit() or an alias unit")
        value = self.units.convert(q, unit if unit != UNKNOWN else None, delta=False)
        return B("num", (), value, TypeInfo("scalar", "num", unit))

    def _in_unit(self, node: B, unit: str | None, text: str) -> B:
        current = node.type.unit
        if current == unit or current is None or unit is None:
            return node
        if UNKNOWN in (current, unit):
            raise ExprError(f"the limit {text!r} is in {current if current != UNKNOWN else 'an unknown unit'} and "
                            f"the check in {unit if unit != UNKNOWN else 'an unknown unit'}; declare them with "
                            "as_unit()")
        try:
            return self.binder._convert(node, unit)
        except (ExprError, UnitsError) as exc:
            raise ExprError(f"the limit {text!r}: {exc}") from None

    def _margin(self, out: BoundRequirement, case: Case, bound: BoundCase) -> None:
        margin = case.margin
        if margin is None:
            return
        if margin.relative is not None:
            bound.margin_rel = margin.relative
            return
        q = margin.absolute
        unit = out.unit
        if q.unit is not None and unit in (None, UNKNOWN):
            raise ExprError(f"the margin {q.text!r} has a unit, but the check has none")
        bound.margin_abs = self.units.convert(q, unit if unit != UNKNOWN else None, delta=True)


def _event_signals(binder: Binder, config: Config, names: set[str], issues: list[Issue]) -> list[str]:
    """Signals the named events need (following events that name other events)."""
    out: list[str] = []
    seen: set[str] = set()
    todo = list(names)
    while todo:
        name = todo.pop()
        if name in seen or name not in config.events:
            continue
        seen.add(name)
        defn = config.events[name]
        text = defn.signal if defn.condition is not None else defn.when
        if text is None:
            continue
        try:
            expr = binder.bind(text, role="event")
        except (ExprError, UnitsError) as exc:
            issues.append(Issue(path=f"events.{name}", message=str(exc), location=defn.loc))
            continue
        for signal in expr.signals:
            if signal not in out:
                out.append(signal)
        todo.extend(expr.events)
    return out


def bind_requirements(reqset: RequirementSet, infos: Sequence, *, param_names: set[str] | None = None,
                      labels: Mapping[str, Sequence[str]] | None = None) -> BoundSet:
    """Bind every covered requirement against a source's signals."""
    from ..policy import SignalIndex

    config = reqset.config
    index = SignalIndex(list(infos))
    binder = Binder(index, {info.name: info for info in infos}, config, param_names=param_names,
                    param_units=config.params.units, labels=labels)
    requirement_binder = _RequirementBinder(binder, config)
    bound = [requirement_binder.bind(req) for req in reqset.requirements if req.covered]
    signals: list[str] = []
    events: set[str] = set()
    for item in bound:
        for name in item.signals():
            if name not in signals:
                signals.append(name)
        events |= item.events()
    issues: list[Issue] = []
    for name in _event_signals(binder, config, events, issues):
        if name not in signals:
            signals.append(name)
    return BoundSet(requirements=bound, reqset=reqset, signals=signals, issues=issues)


def open_run_source(source, config: Config):
    """The source adapter with the mapping's time options, and the load options to use with it."""
    from ..sources import open_source

    hints: dict[str, str] = {}
    for alias in config.signals.values():
        if alias.path is not None and alias.time is not None:
            hints[alias.path.lstrip("/")] = alias.time
    adapter = open_source(source, time_hints=hints or None, global_time=config.time.signal)
    scale = 1.0
    if config.time.unit:
        scale = config.units.factor(config.time.unit, "s")
    options = {"time_scale": scale, "on_non_monotonic": config.time.on_non_monotonic}
    return adapter, options


def lint_against_source(reqset: RequirementSet, source, *, param_names: set[str] | None = None) -> list:
    """Lint items for names, types and units that do not fit the given run."""
    from .lint import LintItem

    adapter, _ = open_run_source(source, reqset.config)
    infos = adapter.list_signals()
    bound = bind_requirements(reqset, infos, param_names=param_names)
    items = []
    for issue in bound.issues:
        items.append(LintItem("error", issue.path, issue.message, issue.location))
    for item in bound.requirements:
        for issue in item.all_issues:
            if issue in item.req.issues:
                continue  # already reported by the static lint
            items.append(LintItem("error", item.req.id, issue.message, issue.location))
        for note in dict.fromkeys(item.notes):
            items.append(LintItem("warning", item.req.id, note, item.req.loc.text))
    untimed = [info.name for info in infos if info.time_ref is None and info.name in bound.signals]
    for name in untimed:
        items.append(LintItem("error", "", f"signal {name!r} has no time signal in {Path(str(source)).name}; set "
                                           "time.signal in the mapping or give the alias a time", None))
    return items
