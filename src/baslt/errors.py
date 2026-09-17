"""Error types. Every Baslt error carries the process exit code the CLI should use.

Exit codes:
    0  success
    1  contract or integrity failure (verification failed)
    2  infeasible budget or internal compile error
    3  usage, policy, source, or container input error
"""

from __future__ import annotations

from dataclasses import dataclass


class BasltError(Exception):
    """Base class for all Baslt errors."""

    exit_code: int = 2

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class UsageError(BasltError):
    """Bad command-line usage, or a feature that needs an optional extra that is not installed."""

    exit_code = 3


@dataclass
class Issue:
    """One problem found while reading or validating a policy."""

    path: str
    message: str
    location: str | None = None  # e.g. "flight_review.yaml:23:9"

    def render(self) -> str:
        loc = f" ({self.location})" if self.location else ""
        prefix = f"{self.path}: " if self.path else ""
        return f"{prefix}{self.message}{loc}"


class PolicyError(BasltError):
    """The policy could not be parsed, validated, or bound to the source signals."""

    exit_code = 3

    def __init__(self, issues: list[Issue] | str) -> None:
        if isinstance(issues, str):
            issues = [Issue(path="", message=issues)]
        self.issues: list[Issue] = list(issues)
        super().__init__("\n".join(issue.render() for issue in self.issues))


class RequirementsError(PolicyError):
    """A requirements table or its mapping could not be read or validated. Issues carry sheet and cell locations."""


class SourceError(BasltError):
    """The simulation source could not be read or is malformed."""

    exit_code = 3


class ContainerError(BasltError):
    """A .baslt file is unreadable, corrupt, or uses an unsupported container feature."""

    exit_code = 3


class CompileError(BasltError):
    """An internal compile failure (for example the closure did not converge)."""

    exit_code = 2


class InfeasibleBudget(BasltError):
    """The hard requirements cannot fit inside the requested byte budget.

    `report` holds the itemized minimum-feasible-size breakdown.
    """

    exit_code = 2

    def __init__(self, message: str, report: dict | None = None) -> None:
        super().__init__(message)
        self.report: dict = report or {}


class SelfVerifyFailed(BasltError):
    """The compiled artifact failed its own verification. Nothing is written."""

    exit_code = 1

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result
