"""What a requirements table says, before any run is looked at: requirements, their cases and parsed cells."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..errors import Issue

if TYPE_CHECKING:
    from ..tabular import Table
    from ..units import Quantity
    from .config import Config

__all__ = [
    "AGGREGATE_KINDS",
    "BOUND_KINDS",
    "KINDS",
    "VERDICTS",
    "VERDICT_LABELS",
    "Bound",
    "Case",
    "LimitSpec",
    "Loc",
    "Margin",
    "Requirement",
    "RequirementSet",
    "Tolerance",
    "Verdict",
    "worst",
]

Verdict = Literal["pass", "warn", "fail", "not_applicable", "error"]
VERDICTS: tuple[str, ...] = ("pass", "warn", "fail", "not_applicable", "error")
VERDICT_LABELS: dict[str, str] = {
    "pass": "PASS", "warn": "WARN", "fail": "FAIL", "not_applicable": "N/A", "error": "ERROR",
}
# Worst first: a requirement's verdict is the first of these that any of its cases has.
PRECEDENCE: tuple[str, ...] = ("fail", "error", "warn", "pass", "not_applicable")

BOUND_KINDS: tuple[str, ...] = ("upper", "lower", "range")
AGGREGATE_KINDS: tuple[str, ...] = ("max", "min", "initial", "final", "mean", "rms", "integral")
KINDS: tuple[str, ...] = (*BOUND_KINDS, "assert", "duration", "value", "event", *AGGREGATE_KINDS)


def worst(verdicts: Iterable[str]) -> Verdict:
    """The requirement verdict of a set of case verdicts: FAIL > ERROR > WARN > PASS > N/A."""
    present = set(verdicts)
    for verdict in PRECEDENCE:
        if verdict in present:
            return verdict  # type: ignore[return-value]
    return "not_applicable"


@dataclass(frozen=True, slots=True)
class Loc:
    """Where something was written: a table cell or row, or a key of a mapping file."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(slots=True)
class Bound:
    """One side of a limit: a quantity, an expression, or a curve evaluated at an argument."""

    quantity: Quantity | None = None
    expr: str | None = None
    curve: str | None = None
    arg: str | None = None
    inclusive: bool = True

    def describe(self) -> str:
        if self.quantity is not None:
            return self.quantity.text
        if self.curve is not None:
            return f"curve({self.curve}{', ' + self.arg if self.arg else ''})"
        return f"= {self.expr}"


@dataclass(slots=True)
class LimitSpec:
    """A parsed limit cell. `kind` is upper, lower, range, equal, not_equal, or bare (direction from the Type)."""

    kind: str
    lower: Bound | None
    upper: Bound | None
    text: str

    def with_direction(self, kind: str) -> LimitSpec:
        """A bare limit read as an upper or lower one."""
        if self.kind != "bare":
            return self
        bound = self.upper or self.lower
        if kind in ("lower", "min_limit"):
            return LimitSpec("lower", bound, None, self.text)
        return LimitSpec("upper", None, bound, self.text)


@dataclass(slots=True)
class Tolerance:
    """How long, or for how many samples, a violation may last before it counts."""

    seconds: float = 0.0
    samples: int | None = None
    text: str = ""


@dataclass(slots=True)
class Margin:
    """The warning band inside a limit: an absolute quantity or a fraction of the limit."""

    absolute: Quantity | None = None
    relative: float | None = None
    text: str = ""


@dataclass(slots=True)
class Case:
    """One row of a requirement: a limit that applies under its own condition."""

    label: str
    when: str | None
    applies_to: str | None
    limit: LimitSpec | None
    tolerance: Tolerance
    margin: Margin | None
    severity: str | None
    count: int | None
    unit: str | None
    loc: Loc
    row: int
    cells: dict[str, str] = field(default_factory=dict)  # field -> the cell location, for messages


@dataclass(slots=True)
class Requirement:
    id: str
    title: str
    kind: str | None
    check: str
    cases: list[Case]
    loc: Loc
    row: int
    when: str | None = None
    grid: str | None = None
    passthrough: dict[str, str] = field(default_factory=dict)
    covered: bool = True
    issues: list[Issue] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)  # every table row of the requirement, in order

    @property
    def has_errors(self) -> bool:
        return bool(self.issues)


@dataclass(slots=True)
class RequirementSet:
    requirements: list[Requirement]
    config: Config
    table: Table
    sha256: str
    header_row: int
    columns: dict[str, int]  # field -> 1-based column of the requirements table
    issues: list[Issue] = field(default_factory=list)  # problems of the table or mapping as a whole
    warnings: list[Issue] = field(default_factory=list)  # things worth knowing that do not stop a check
    skipped_rows: int = 0
    not_covered: list[str] = field(default_factory=list)
    unmatched_checks: list[str] = field(default_factory=list)
    checks_table: Table | None = None

    @property
    def covered(self) -> list[Requirement]:
        return [req for req in self.requirements if req.covered]

    @property
    def case_count(self) -> int:
        return sum(len(req.cases) for req in self.requirements)
