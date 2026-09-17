"""Checking a requirements table before any run: every cell readable, every row consistent, every name defined.

Static lint needs no run and no numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..errors import Issue
from .model import AGGREGATE_KINDS, BOUND_KINDS, Requirement, RequirementSet

__all__ = ["LintItem", "LintResult", "lint_static"]

LIMIT_KINDS_FOR: dict[str, tuple[str, ...]] = {
    "upper": ("upper", "bare"),
    "lower": ("lower", "bare"),
    "range": ("range",),
    "duration": ("upper", "lower", "range", "equal", "not_equal", "bare"),
    "value": ("upper", "lower", "range", "equal", "not_equal"),
    "event": ("upper", "lower", "range", "equal"),
    **{kind: ("upper", "lower", "range", "equal", "not_equal") for kind in AGGREGATE_KINDS},
}


@dataclass(slots=True)
class LintItem:
    level: str  # "error" or "warning"
    id: str
    message: str
    location: str | None = None

    def to_json(self) -> dict:
        return {"level": self.level, "id": self.id, "message": self.message, "location": self.location}


@dataclass(slots=True)
class LintResult:
    label: str
    requirements: int
    cases: int
    covered: int
    not_covered: list[str]
    skipped_rows: int
    items: list[LintItem] = field(default_factory=list)

    @property
    def errors(self) -> list[LintItem]:
        return [item for item in self.items if item.level == "error"]

    @property
    def warnings(self) -> list[LintItem]:
        return [item for item in self.items if item.level == "warning"]

    @property
    def status(self) -> str:
        return "fail" if self.errors else ("pass_with_warnings" if self.warnings else "pass")

    @property
    def exit_code(self) -> int:
        return 3 if self.errors else 0

    def to_json(self) -> dict:
        return {
            "status": self.status, "requirements": self.requirements, "cases": self.cases,
            "covered": self.covered, "not_covered": self.not_covered, "skipped_rows": self.skipped_rows,
            "source": self.label, "items": [item.to_json() for item in self.items],
        }

    def render(self) -> str:
        lines = [
            f"Requirements  {self.label}   {self.requirements} requirements ({self.cases} cases), "
            f"{len(self.not_covered)} not covered, {self.skipped_rows} rows skipped",
        ]
        if self.items:
            width = max(len(item.id) for item in self.items)
            lines.append("")
            for item in self.items:
                where = f" ({item.location})" if item.location else ""
                lines.append(f"{item.level.upper():<8} {item.id:<{width}}  {item.message}{where}")
        lines.append("")
        summary = f"{_count(len(self.errors), 'error')}, {_count(len(self.warnings), 'warning')}"
        lines.append(("Result: FAIL (" if self.errors else "Result: OK (") + summary + ")")
        return "\n".join(lines) + "\n"


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _items(level: str, issues: Sequence[Issue], default_id: str = "") -> list[LintItem]:
    out = []
    for issue in issues:
        rid, _, rest = issue.path.partition(".")
        out.append(LintItem(level, rid or default_id, issue.message, issue.location))
    return out


def _requirement_items(req: Requirement) -> list[LintItem]:
    items = _items("error", req.issues, req.id)
    if not req.covered or req.issues:
        return items
    for case in req.cases:
        where = case.loc.text
        limit = case.limit
        kind = req.kind
        if kind in ("assert",) and limit is not None:
            items.append(LintItem("error", req.id, "an assert check takes no limit; its Check is the condition", where))
            continue
        if kind is None or kind == "assert":
            continue
        if limit is None:
            if kind != "event":
                article = "an" if kind[0] in "aeiou" else "a"
                items.append(LintItem("error", req.id, f"{article} {kind} check needs a limit", where))
            continue
        allowed = LIMIT_KINDS_FOR.get(kind, ())
        if limit.kind not in allowed:
            shown = {"bare": "a bare value", "upper": "an upper limit", "lower": "a lower limit", "range": "a range",
                     "equal": "an equality", "not_equal": "an inequality"}[limit.kind]
            hint = " (write it as '<= X', '>= X' or a range)" if limit.kind == "bare" else ""
            items.append(LintItem("error", req.id, f"a {kind} check cannot use {shown} {limit.text!r}{hint}", where))
        if kind in BOUND_KINDS and any(b is not None and b.expr is not None and b.curve is None
                                       for b in (limit.lower, limit.upper)):
            pass  # expression limits are checked when names are bound
    return items


def lint_static(reqset: RequirementSet) -> LintResult:
    table = reqset.table
    label = table.path.name + (f":{table.sheet}" if table.sheet else "")
    result = LintResult(
        label=label,
        requirements=len(reqset.requirements),
        cases=reqset.case_count,
        covered=len(reqset.covered),
        not_covered=list(reqset.not_covered),
        skipped_rows=reqset.skipped_rows,
    )
    result.items.extend(_items("error", reqset.issues))
    result.items.extend(_items("warning", reqset.warnings))
    for req in reqset.requirements:
        result.items.extend(_requirement_items(req))
        if not req.covered:
            result.items.append(LintItem("warning", req.id, "no check: the requirement is not covered", req.loc.text))
    return result
