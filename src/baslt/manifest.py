"""The artifact manifest: provenance, budget accounting, per-requirement evidence and statuses.

The shape is fixed by docs/container.md. Two fields are written so that the manifest's own length does
not depend on the artifact size: `artifact.size_bytes` is a 20-digit zero-padded string and
`artifact.ratio` is `%.6e`. The remaining size-dependent field is `budget.overhead_bytes`, whose digit
count can still change, so the manifest is solved as a fixed point: `manifest.json` is STORED, so its
member length is exactly the length of its canonical JSON, and `build_manifest` iterates until the
length it assumed is the length it produced. Sixteen iterations are more than a digit count can need;
failing to settle is a CompileError rather than a manifest whose `size_bytes` lies.

This module formats dicts only: no numpy, and nothing imported from the planner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._version import __version__
from .container.spec import CONTAINER_VERSION, DEFAULT_LEVEL, canonical_json
from .errors import CompileError

if TYPE_CHECKING:
    from .hashing import HashInfo
    from .policy.schema import Policy
    from .signals import SourceMeta

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_FAIL = "fail"
STATUSES: tuple[str, ...] = (STATUS_PASS, STATUS_WARN, STATUS_NOT_APPLICABLE, STATUS_FAIL)

OVERALL_PASS = "pass"
OVERALL_PASS_WITH_WARNINGS = "pass_with_warnings"
OVERALL_FAIL = "fail"

EVIDENCE_LIMIT = 256  # docs/container.md: evidence lists hold at most 256 items, counts stay exact
SIZE_DIGITS = 20
MAX_SIZE_ITERATIONS = 16

NO_DIGEST: dict = {"algorithm": "none", "mode": "none", "covered_bytes": 0, "value": ""}


def overall_status(statuses: Sequence[str], *, on_not_applicable: str = "warn") -> str:
    """Roll per-requirement statuses up (docs/contracts.md, "Statuses").

    `fail` if anything failed; otherwise `pass_with_warnings` if anything warned or could not be
    evaluated -- but a `not_applicable` requirement fails the run when the policy says
    `artifact.on_not_applicable: fail`.
    """
    if any(status == STATUS_FAIL for status in statuses):
        return OVERALL_FAIL
    not_applicable = any(status == STATUS_NOT_APPLICABLE for status in statuses)
    if not_applicable and on_not_applicable == "fail":
        return OVERALL_FAIL
    if not_applicable or any(status == STATUS_WARN for status in statuses):
        return OVERALL_PASS_WITH_WARNINGS
    return OVERALL_PASS


def cap_evidence(evidence: object, limit: int = EVIDENCE_LIMIT) -> object:
    """Truncate every list in an evidence tree to `limit` items. Counts beside them stay exact."""
    if isinstance(evidence, Mapping):
        return {key: cap_evidence(value, limit) for key, value in evidence.items()}
    if isinstance(evidence, (list, tuple)):
        return [cap_evidence(item, limit) for item in list(evidence)[:limit]]
    return evidence


@dataclass(slots=True)
class RequirementEntry:
    """One row of `manifest.requirements`."""

    id: str
    signal: str
    op: str
    severity: str
    status: str
    evidence: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "signal": self.signal,
            "op": self.op,
            "severity": self.severity,
            "status": self.status,
            "evidence": cap_evidence(dict(self.evidence)),
        }


@dataclass(slots=True)
class EventEntry:
    """One row of `manifest.events`."""

    name: str
    signal: str
    severity: str
    status: str
    evidence: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "signal": self.signal,
            "severity": self.severity,
            "status": self.status,
            "evidence": cap_evidence(dict(self.evidence)),
        }


@dataclass(slots=True)
class SyncEntry:
    """One row of `manifest.sync_groups`."""

    name: str
    status: str
    evidence: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"name": self.name, "status": self.status, "evidence": cap_evidence(dict(self.evidence))}


@dataclass(slots=True)
class SignalBudget:
    """One row of `manifest.budget.signals`."""

    name: str
    retained: int
    hard: int
    soft: int
    bytes: int
    soft_max_abs_err: float = 0.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "retained": int(self.retained),
            "hard": int(self.hard),
            "soft": int(self.soft),
            "bytes": int(self.bytes),
            "soft_max_abs_err": fixed_width_error(self.soft_max_abs_err),
        }


def fixed_width_error(value: float) -> str:
    """A non-negative error as a 12-character `%.6e` string, so the manifest length does not depend on it.

    The budget search sizes the artifact before the preview error is known; a fixed width keeps that size exact.
    """
    import math

    number = float(value)
    if math.isnan(number):
        return "nan".rjust(12)
    if math.isinf(number) or number >= 1e100:
        return "inf".rjust(12)
    if number < 1e-99:
        return "0.000000e+00"
    return "%.6e" % number


def _source_block(source: SourceMeta | None, digest: HashInfo | None, signals: int, samples_total: int) -> dict:
    return {
        "path": getattr(source, "path", None),
        "format": getattr(source, "format", "numpy"),
        "size_bytes": getattr(source, "size_bytes", None),
        "digest": NO_DIGEST if digest is None else digest.to_json(),
        "signals": int(signals),
        "samples_total": int(samples_total),
        "issues": list(getattr(source, "issues", []) or []),
    }


def build_manifest(
    *,
    source: SourceMeta | None,
    digest: HashInfo | None,
    policy: Policy,
    requirements: Sequence[RequirementEntry],
    signals: Sequence[SignalBudget],
    samples_total: int,
    size_of: Callable[[int], int],
    source_signals: int | None = None,
    max_bytes: int | None = None,
    budget_source: str = "none",
    codec: str = "deflate",
    level: int = DEFAULT_LEVEL,
    discretionary_bytes: int = 0,
    events: Sequence[EventEntry] = (),
    sync_groups: Sequence[SyncEntry] = (),
) -> tuple[dict, int]:
    """Build the manifest and the artifact size it describes.

    `size_of(manifest_length)` returns the total file size when `manifest.json` holds that many bytes;
    the two are solved together. Returns `(manifest, size)`.
    """
    discretionary_bytes = int(discretionary_bytes)
    required_bytes = sum(int(entry.bytes) for entry in signals) - discretionary_bytes
    source_size = getattr(source, "size_bytes", None)

    base = {
        "status": overall_status(
            [entry.status for entry in requirements]
            + [entry.status for entry in events]
            + [entry.status for entry in sync_groups],
            on_not_applicable=policy.artifact.on_not_applicable,
        ),
        "baslt": {"version": __version__, "container": CONTAINER_VERSION},
        "source": _source_block(
            source,
            digest,
            len(signals) if source_signals is None else source_signals,
            samples_total,
        ),
        "policy": {"name": policy.name, "sha256": policy.sha256},
        "requirements": [entry.to_json() for entry in requirements],
        "events": [entry.to_json() for entry in events],
        "trajectories": [],
        "sync_groups": [entry.to_json() for entry in sync_groups],
    }
    budget_signals = [entry.to_json() for entry in signals]

    def assemble(size: int, overhead: int) -> dict:
        ratio = (float(source_size) / size) if (source_size and size > 0) else 0.0
        return {
            **base,
            "artifact": {
                "size_bytes": f"{int(size):0{SIZE_DIGITS}d}",
                "max_bytes": None if max_bytes is None else int(max_bytes),
                "ratio": "%.6e" % ratio,
                "codec": codec,
                "level": int(level),
            },
            "budget": {
                "source": budget_source,
                "required_bytes": required_bytes,
                "discretionary_bytes": discretionary_bytes,
                "overhead_bytes": int(overhead),
                "signals": budget_signals,
            },
        }

    # Fixed point: the assumed manifest length decides the file size, which decides overhead_bytes,
    # whose digits decide the manifest length again.
    length = len(canonical_json(assemble(0, 0)))
    for _ in range(MAX_SIZE_ITERATIONS):
        size = int(size_of(length))
        manifest = assemble(size, size - required_bytes - discretionary_bytes)
        settled = len(canonical_json(manifest))
        if settled == length:
            return manifest, size
        length = settled
    raise CompileError(
        f"the manifest size did not settle in {MAX_SIZE_ITERATIONS} iterations "
        f"(last manifest length {length} bytes); this is an internal compiler error"
    )
