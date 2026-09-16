"""Rendering of a verification result: the text table of docs/cli.md, and its JSON form.

The layout is the one the command-line reference prints:

```
Policy   flight_review   sha256 3f9a1c0b…
Source   run.h5          sha256 (full) 5e02…
Artifact 1.86 MiB of 2.00 MiB   ratio 4.64e+03

STATUS  CHECK                                         BASIS     SOURCE         ARTIFACT       TOLERANCE
PASS    structure                                     artifact  -              -              -
PASS    hard.q_dyn.global_extrema max                 attested  68142.3 Pa     68142.3 Pa     0

Result: PASS WITH WARNINGS (11 pass, 1 warn, 1 n/a, 0 fail)
```

Two details of that example are rules rather than accidents:

- The CHECK column writes a check id `<requirement>.<aspect>` as `<requirement> aspect`. A requirement that could
  not be evaluated has no aspect and is printed bare, which is why the aspect is recognised by name (`ASPECTS`,
  the aspect vocabulary of docs/verification.md) instead of by splitting at the last dot.
- `structure` is one row while all six structure checks pass, and becomes one row per check as soon as any of them
  does not, so a failure names the check that found it.

Columns keep the widths of the example and grow only for content that would not fit, so the table stays aligned
either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .structure import BASIS_ARTIFACT, FAIL, NOT_APPLICABLE, PASS, WARN, Check

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ASPECTS", "check_label", "render", "to_json"]

STATUS_LABELS: dict[str, str] = {PASS: "PASS", WARN: "WARN", NOT_APPLICABLE: "N/A", FAIL: "FAIL"}
RESULT_LABELS: dict[str, str] = {
    "pass": "PASS",
    "pass_with_warnings": "PASS WITH WARNINGS",
    "fail": "FAIL",
}

HEADERS: tuple[str, ...] = ("STATUS", "CHECK", "BASIS", "SOURCE", "ARTIFACT", "TOLERANCE")
WIDTHS: tuple[int, ...] = (8, 46, 10, 15, 15, 0)  # the last column is not padded
LABEL_WIDTH = 9  # "Artifact " in the header block
NAME_WIDTH = 16
DASH = "-"

# Aspect names of docs/verification.md, plus `source` for a fact recomputed from the source run.
ASPECTS: frozenset[str] = frozenset(
    {
        "container", "descriptors", "invariants", "size", "policy", "bind",
        "max", "min", "buckets", "prominence", "separation", "brackets", "fidelity", "runs",
        "transitions", "trigger", "windows", "sed", "alignment", "accounting", "unaligned",
        "crossings", "digest", "samples", "source", "peaks", "errors",
    }
)  # fmt: skip


def check_label(check_id: str) -> str:
    """`structure.container` -> `structure container`; a bare requirement id is unchanged."""
    head, _, last = check_id.rpartition(".")
    return f"{head} {last}" if head and last in ASPECTS else check_id


def _cell(value: object) -> str:
    return DASH if value is None or value == "" else str(value)


def _collapse_structure(checks: Sequence[Check]) -> list[Check]:
    structure = [c for c in checks if c.id == "structure" or c.id.startswith("structure.")]
    if not structure or not all(c.status == PASS for c in structure):
        return list(checks)
    rows: list[Check] = []
    for check in checks:
        if check in structure:
            if structure[0] is check:
                rows.append(Check("structure", PASS, BASIS_ARTIFACT, message="every structure check passed"))
            continue
        rows.append(check)
    return rows


def _row(fields: Sequence[str], widths: Sequence[int]) -> str:
    out = ""
    for value, width in zip(fields, widths):
        out += value.ljust(width)
    return out.rstrip()


def table_lines(checks: Sequence[Check]) -> list[str]:
    """The header row and one row per check."""
    rows = [
        (
            STATUS_LABELS.get(check.status, check.status.upper()),
            check_label(check.id),
            _cell(check.basis),
            _cell(check.claimed),
            _cell(check.measured),
            _cell(check.allowed),
        )
        for check in _collapse_structure(checks)
    ]
    widths = list(WIDTHS)
    for column in range(len(HEADERS) - 1):
        longest = max((len(row[column]) for row in rows), default=0)
        widths[column] = max(widths[column], len(HEADERS[column]) + 2, longest + 2)
    return [_row(HEADERS, widths), *(_row(row, widths) for row in rows)]


def _fmt_bytes(count: object) -> str:
    if not isinstance(count, (int, float)) or isinstance(count, bool):
        return "unknown size"
    value = float(count)
    if value >= 1048576:
        return f"{value / 1048576:.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{int(value)} B"


def _short(value: object) -> str:
    return f"{value[:8]}…" if isinstance(value, str) and value else "(none)"


def header_lines(result: object) -> list[str]:
    """The Policy / Source / Artifact block above the table."""
    policy = getattr(result, "policy", {}) or {}
    source = getattr(result, "source", {}) or {}
    artifact = getattr(result, "artifact", {}) or {}
    names = [str(policy.get("name") or "policy")]
    if source:
        names.append(str(source.get("path") or source.get("format") or "source"))
    width = max(NAME_WIDTH, max(len(name) for name in names) + 2)

    lines = [f"{'Policy'.ljust(LABEL_WIDTH)}{names[0].ljust(width)}sha256 {_short(policy.get('sha256'))}"]
    if not policy and not source:
        lines = []
    if source:
        digest = source.get("digest") or {}
        mode = digest.get("mode")
        if not mode or mode == "none":
            detail = "no digest"
        else:
            detail = f"{digest.get('algorithm') or 'sha256'} ({mode}) {_short(digest.get('value'))}"
        lines.append(f"{'Source'.ljust(LABEL_WIDTH)}{names[1].ljust(width)}{detail}")

    size = _fmt_bytes(artifact.get("size_bytes"))
    max_bytes = artifact.get("max_bytes")
    span = size if max_bytes is None else f"{size} of {_fmt_bytes(max_bytes)}"
    ratio = artifact.get("ratio")
    tail = "" if not isinstance(ratio, (int, float)) else f"   ratio {float(ratio):.2e}"
    lines.append(f"{'Artifact'.ljust(LABEL_WIDTH)}{span}{tail}")
    return lines


def result_line(result: object) -> str:
    counts = result.counts  # type: ignore[attr-defined]
    label = RESULT_LABELS.get(result.status, str(result.status).upper())  # type: ignore[attr-defined]
    return (
        f"Result: {label} ({counts[PASS]} pass, {counts[WARN]} warn, "
        f"{counts[NOT_APPLICABLE]} n/a, {counts[FAIL]} fail)"
    )


def render(result: object) -> str:
    """The whole report: header block, table, result line."""
    lines = [*header_lines(result), "", *table_lines(result.checks), "", result_line(result)]  # type: ignore[attr-defined]
    return "\n".join(lines) + "\n"


def to_json(result: object) -> dict:
    """The machine-readable form, with a top-level "status" as docs/cli.md requires of every command."""
    counts = result.counts  # type: ignore[attr-defined]
    return {
        "status": result.status,  # type: ignore[attr-defined]
        "strict": bool(getattr(result, "strict", False)),
        "exit_code": result.exit_code,  # type: ignore[attr-defined]
        "policy": dict(getattr(result, "policy", {}) or {}),
        "source": dict(getattr(result, "source", {}) or {}),
        "artifact": dict(getattr(result, "artifact", {}) or {}),
        "counts": {
            "pass": counts[PASS],
            "warn": counts[WARN],
            "not_applicable": counts[NOT_APPLICABLE],
            "fail": counts[FAIL],
        },
        "checks": [
            {
                "id": check.id,
                "status": check.status,
                "basis": check.basis,
                "claimed": check.claimed,
                "measured": check.measured,
                "allowed": check.allowed,
                "message": check.message,
            }
            for check in result.checks  # type: ignore[attr-defined]
        ],
    }
