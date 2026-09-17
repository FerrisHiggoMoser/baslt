"""Results of requirement checks: the run record, JSON, CSV, the Excel workbook and the annotated requirements copy.

Times in tables are shown relative to `time.t0` (an event such as liftoff) when the mapping names one: "T+62.0 s".
Numbers keep six significant digits in text and full precision in JSON and Excel cells. CSV text that a
spreadsheet would read as a formula is prefixed with an apostrophe.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .._io import write_atomic
from ..errors import Issue
from .check import RequirementResult
from .model import VERDICT_LABELS, RequirementSet

__all__ = ["RunResult", "annotate", "fmt_time", "fmt_value", "render_run", "run_sheets", "write_csv",
           "write_json"]

STYLE_OF = {"pass": "pass", "warn": "warn", "fail": "fail", "not_applicable": "na", "error": "error"}
ORDER = {"fail": 0, "error": 1, "warn": 2, "not_applicable": 3, "pass": 4}
# large values of these units are shown with a prefix: 46032 m reads as 46.03 km
PREFIXED = {
    "Pa": (("MPa", 1e6), ("kPa", 1e3)),
    "N": (("MN", 1e6), ("kN", 1e3)),
    "m": (("km", 1e3),),
    "W": (("MW", 1e6), ("kW", 1e3)),
    "J": (("MJ", 1e6), ("kJ", 1e3)),
}
DEFAULT_WRITE_BACK = {"verdict": "Verdict", "value": "Result", "margin": "Margin", "evidence": "Evidence"}


@dataclass
class RunResult:
    run_id: str
    source: str
    format: str
    requirements_file: str
    duration: float | None = None
    signals: int = 0
    samples: int = 0
    results: list[RequirementResult] = field(default_factory=list)
    events: dict = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    not_covered: list[str] = field(default_factory=list)
    t0: float | None = None
    t0_event: str | None = None
    params: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    digest: str | None = None
    error: str | None = None  # the run could not be checked at all
    covered: int = 0  # requirements with a check in the table

    def counts(self) -> dict[str, int]:
        out = {verdict: 0 for verdict in ("pass", "warn", "fail", "not_applicable", "error")}
        for result in self.results:
            out[result.verdict] += 1
        out["not_covered"] = len(self.not_covered)
        return out

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        present = {r.verdict for r in self.results}
        for verdict in ("fail", "error", "warn", "pass"):
            if verdict in present:
                return verdict
        return "not_applicable" if present else "pass"

    def to_json(self, *, with_results: bool = True) -> dict:
        out = {
            "run": self.run_id, "source": self.source, "format": self.format, "status": self.status,
            "requirements": self.requirements_file, "duration": _num(self.duration), "signals": self.signals,
            "samples": self.samples, "counts": self.counts(), "not_covered": self.not_covered,
            "events": self.events, "t0": _num(self.t0), "t0_event": self.t0_event, "params": self.params,
            "timing": self.timing, "digest": self.digest, "error": self.error,
            "issues": [{"path": i.path, "message": i.message, "location": i.location} for i in self.issues],
        }
        if with_results:
            out["results"] = [r.to_json() for r in self.results]
        return out


def _num(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def fmt_value(value, unit: str | None = None, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    value = float(value)
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    for name, factor in PREFIXED.get(unit or "", ()):
        if abs(value) >= factor:
            value, unit = value / factor, name
            break
    text = f"{value:.{digits}g}"
    if "e" in text and 10 ** digits <= abs(value) < 1e9:
        text = f"{value:.0f}"
    return f"{text} {unit}" if unit and unit not in ("1", "?") else text


def fmt_time(seconds, t0: float | None = None) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if not math.isfinite(seconds):
        return "-"
    if t0 is not None and math.isfinite(t0):
        delta = seconds - t0
        return f"T{'+' if delta >= 0 else '-'}{abs(delta):.3f} s"
    return f"{seconds:.3f} s"


def _margin_text(result: RequirementResult) -> str:
    if result.margin is None:
        return "-"
    text = fmt_value(result.margin, result.unit)
    if result.margin_pct is not None and math.isfinite(result.margin_pct):
        text += f" ({result.margin_pct * 100:.1f} %)"
    return text


def _context_text(result: RequirementResult, t0: float | None) -> str:
    context = result.context or {}
    parts = []
    if context.get("conditions"):
        parts.append(", ".join(context["conditions"]))
    if context.get("event") is not None:
        parts.append(f"{context['since']:.1f} s after {context['event']}")
    return "; ".join(parts)


def _tolerated_text(result: RequirementResult) -> str:
    count = result.totals.get("tolerated", 0)
    return f"{count} short violation{'s' if count != 1 else ''} within the tolerance"


def evidence_text(result: RequirementResult, t0: float | None) -> str:
    """One sentence for a requirement's evidence column."""
    if result.verdict == "error":
        return f"not checked: {result.reason}"
    if result.verdict == "not_applicable":
        return result.reason or "not applicable"
    pieces = []
    if result.value is not None and not isinstance(result.value, str):
        label = "worst" if result.kind in ("bound",) else "value"
        pieces.append(f"{label} {fmt_value(result.value, result.unit)}")
    elif isinstance(result.value, str):
        pieces.append(f"value {result.value}")
    if result.at is not None:
        pieces.append(f"at {fmt_time(result.at, t0)}")
    if result.limit:
        pieces.append(f"limit {result.limit}")
    if result.margin is not None:
        pieces.append(f"margin {_margin_text(result)}")
    if result.kind == "assert" and result.verdict == "pass":
        checked = result.totals.get("active_time")
        pieces.append("the condition held" + (f" for all {checked:.4g} s checked" if checked else ""))
    if result.reason and result.verdict != "pass":
        pieces.append(f"({result.reason})")
    elif result.verdict == "pass" and result.totals.get("tolerated"):
        pieces.append(f"({_tolerated_text(result)})")
    return ", ".join(pieces) or VERDICT_LABELS[result.verdict]


# ------------------------------------------------------------------------------------------------------------
# text


def _table(rows: Sequence[Sequence[str]], widths: Sequence[int] | None = None) -> list[str]:
    widths = widths or [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def render_run(run: RunResult, *, show_all: bool = False, outputs: Mapping[str, Path] | None = None) -> str:
    counts = run.counts()
    duration = f"{run.duration:.1f} s" if run.duration is not None else "-"
    lines = [
        f"Run      {Path(run.source).name if run.source else run.run_id}   {run.format}, {duration}, "
        f"{run.signals} signals",
        f"Checks   {run.requirements_file}   {run.covered} requirements, {counts['not_covered']} not covered",
    ]
    if run.error:
        lines += ["", f"ERROR    the run could not be checked: {run.error}"]
    shown = [r for r in run.results if show_all or r.verdict != "pass"]
    shown.sort(key=lambda r: (ORDER[r.verdict], r.first_violation if r.first_violation is not None else math.inf))
    if shown:
        rows = [("VERDICT", "ID", "TITLE", "WORST", "LIMIT", "MARGIN", "AT", "")]
        for r in shown:
            if r.verdict == "error":
                where = f" ({r.issues[0].location})" if r.issues and r.issues[0].location else ""
                rows.append((VERDICT_LABELS[r.verdict], r.id, r.title[:28], (r.reason or "") + where, "", "", "",
                             ""))
                continue
            value = fmt_value(r.value, r.unit) if r.value is not None else "-"
            detail = _context_text(r, run.t0)
            if r.verdict in ("warn", "not_applicable") and r.reason:
                detail = "; ".join(filter(None, [r.reason, detail]))
            elif r.verdict == "pass" and r.totals.get("tolerated"):
                detail = _tolerated_text(r)
            rows.append((VERDICT_LABELS[r.verdict], r.id, r.title[:28], value, r.limit or "-", _margin_text(r),
                         fmt_time(r.at, run.t0), detail))
        lines += [""] + _table(rows)
    parts = [f"{counts[v]} {label}" for v, label in (("pass", "pass"), ("warn", "warn"), ("fail", "fail"),
                                                       ("error", "error"), ("not_applicable", "n/a"))]
    status = VERDICT_LABELS.get(run.status, run.status.upper())
    lines += ["", f"Result: {status} ({', '.join(parts)})"]
    if outputs:
        order = ("report", "xlsx", "annotated", "json", "csv", "archive")
        paths = [Path(outputs[key]) for key in order if key in outputs]
        folders = {path.parent for path in paths}
        if len(folders) == 1:
            lines.append(f"Output   {folders.pop()}{os.sep}  {', '.join(path.name for path in paths)}")
        else:
            lines.extend(f"Output   {path}" for path in paths)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------
# files


def _safe(text) -> object:
    if isinstance(text, str) and text[:1] in ("=", "+", "-", "@", "\t", "\r") and not _is_number(text):
        return "'" + text
    return text


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


RESULT_COLUMNS = ("id", "title", "verdict", "reason", "kind", "check", "value", "unit", "limit", "margin",
                  "margin_pct", "at", "at_t0", "first_violation", "case", "severity", "runs", "violating_time",
                  "location")


def _result_row(r: RequirementResult, t0: float | None) -> list:
    """`at` is the time of the reported value (the worst point of a series check)."""
    at = r.at
    return [
        r.id, r.title, VERDICT_LABELS[r.verdict], r.reason or "", r.kind, r.check,
        r.value if not isinstance(r.value, float) or math.isfinite(r.value) else None, r.unit or "",
        r.limit or "", r.margin, r.margin_pct, at, (at - t0) if (at is not None and t0 is not None) else None,
        r.first_violation, r.case or "", r.severity or "", r.totals.get("runs", r.totals.get("count")),
        r.totals.get("violating_time"), r.loc,
    ]


def write_json(path: Path, run: RunResult) -> Path:
    write_atomic(path, (json.dumps(run.to_json(), indent=2, ensure_ascii=False, sort_keys=False) + "\n").encode())
    return path


def write_csv(path: Path, run: RunResult) -> Path:
    passthrough = list(dict.fromkeys(key for r in run.results for key in r.passthrough))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([*RESULT_COLUMNS, *passthrough])
    for r in sorted(run.results, key=lambda r: r.rows[0] if r.rows else 0):
        row = _result_row(r, run.t0) + [r.passthrough.get(key, "") for key in passthrough]
        writer.writerow(["" if v is None else _safe(v) for v in row])
    write_atomic(path, buffer.getvalue().encode("utf-8"))
    return path


def run_sheets(run: RunResult, reqset: RequirementSet | None = None) -> list:
    from ..tabular import Sheet, Styled

    def head(names):
        return [Styled(name, "header") for name in names]

    counts = run.counts()
    summary = [
        head(["Item", "Value"]),
        ["Run", run.run_id], ["Source", run.source], ["Format", run.format],
        ["Requirements", run.requirements_file], ["Status", Styled(VERDICT_LABELS.get(run.status, run.status),
                                                                    STYLE_OF.get(run.status, "bold"))],
        ["Duration (s)", run.duration], ["Signals loaded", run.signals],
        *[[VERDICT_LABELS[v], counts[v]] for v in ("pass", "warn", "fail", "error", "not_applicable")],
        ["Not covered", ", ".join(run.not_covered) or "none"],
        ["Time zero", f"{run.t0_event} at {run.t0:.6g} s" if run.t0 is not None else "none"],
    ]
    if run.digest:
        summary.append(["Source digest", run.digest])
    for key, value in run.params.items():
        summary.append([f"param {key}", value])
    if run.error:
        summary.append(["Error", run.error])

    passthrough = list(dict.fromkeys(key for r in run.results for key in r.passthrough))
    names = ["ID", "Title", "Verdict", "Reason", "Type", "Check", "Value", "Unit", "Limit", "Margin", "Margin %",
             "At (s)", "At (T+)", "First violation (s)", "Case", "Severity", "Runs", "Violating time (s)",
             "Location", *passthrough]
    rows = [head(names)]
    for r in sorted(run.results, key=lambda r: r.rows[0] if r.rows else 0):
        values = _result_row(r, run.t0)
        values[2] = Styled(values[2], STYLE_OF[r.verdict])
        if values[10] is not None:
            values[10] = Styled(values[10], "pct")
        rows.append(values + [r.passthrough.get(key, "") for key in passthrough])
    for rid in run.not_covered:
        rows.append([rid, None, Styled("NOT COVERED", "na"), "no check"])

    cases = [head(["ID", "Case", "Row", "Verdict", "Reason", "Applies", "Value", "Limit", "Margin", "Margin %",
                   "At (s)", "Runs", "Active time (s)", "Severity"])]
    for r in run.results:
        for c in r.cases:
            cases.append([r.id, c.label, c.row, Styled(VERDICT_LABELS[c.verdict], STYLE_OF[c.verdict]),
                          c.reason or "", c.applies, c.value if not isinstance(c.value, float) or
                          math.isfinite(c.value) else None, c.limit, c.margin,
                          Styled(c.margin_pct, "pct") if c.margin_pct is not None else None, c.at, c.runs,
                          c.active_time, c.severity or ""])

    violations = [head(["ID", "Start (s)", "End (s)", "Duration (s)", "Samples", "Worst at (s)", "Worst value",
                        "Limit", "Excess", "Tolerated", "Case", "Component", "Flags"])]
    for r in run.results:
        for run_record in r.runs:
            violations.append([r.id, run_record["start"], run_record["end"], run_record["duration"],
                               run_record["samples"], run_record["worst_t"], run_record.get("worst_value"),
                               run_record.get("limit"), run_record.get("excess"), run_record["tolerated"],
                               run_record.get("case") or "", run_record["component"],
                               ", ".join(run_record["flags"])])

    events = [head(["Event", "Count", "First (s)", "Last (s)", "Times (s)", "Pending at end", "Problem"])]
    for name, info in run.events.items():
        times = info.get("times") or []
        events.append([name, info.get("count"), times[0] if times else None, times[-1] if times else None,
                       ", ".join(f"{v:.6g}" for v in times[:50]), info.get("pending", False),
                       info.get("error") or ""])

    issues = [head(["Level", "ID", "Message", "Location"])]
    for issue in run.issues:
        issues.append(["error", issue.path, issue.message, issue.location or ""])
    for r in run.results:
        for issue in r.issues:
            issues.append(["error", r.id, issue.message, issue.location or ""])
        for note in r.notes:
            issues.append(["note", r.id, note, r.loc])
    if reqset is not None:
        for warning in reqset.warnings:
            issues.append(["warning", warning.path, warning.message, warning.location or ""])

    return [
        Sheet("Summary", summary, widths=[20, 80], autofilter=False),
        Sheet("Results", rows),
        Sheet("Cases", cases),
        Sheet("Violations", violations),
        Sheet("Events", events),
        Sheet("Issues", issues),
    ]


# ------------------------------------------------------------------------------------------------------------
# write-back


def write_back_values(reqset: RequirementSet, results: Mapping[str, RequirementResult], t0: float | None,
                      aggregate: Mapping[str, str] | None = None) -> dict[int, dict[str, object]]:
    """Table row -> {result field: value} for every requirement row."""
    rows: dict[int, dict[str, object]] = {}
    for req in reqset.requirements:
        result = results.get(req.id)
        first = req.row
        if not req.covered:
            rows[first] = {"verdict": "NOT COVERED", "evidence": "no check"}
            continue
        if result is None:
            continue
        values = {
            "verdict": VERDICT_LABELS[result.verdict],
            "value": fmt_value(result.value, result.unit) if result.value is not None else "",
            "limit": result.limit or "",
            "margin": _margin_text(result) if result.margin is not None else "",
            "margin_pct": result.margin_pct,
            "at": fmt_time(result.at, t0),
            "evidence": aggregate.get(req.id) if aggregate else evidence_text(result, t0),
            "runs": result.totals.get("runs", ""),
        }
        rows[first] = values
        if reqset.config.checks is None:
            for case_result in result.cases:
                if case_result.row is not None and case_result.row != first:
                    rows[case_result.row] = {"verdict": VERDICT_LABELS[case_result.verdict],
                                             "value": fmt_value(case_result.value, result.unit)
                                             if case_result.value is not None else "",
                                             "margin": fmt_value(case_result.margin, result.unit)
                                             if case_result.margin is not None else "",
                                             "evidence": case_result.reason or ""}
    return rows


def annotate(reqset: RequirementSet, results: Mapping[str, RequirementResult], dst: Path, *,
             t0: float | None = None, aggregate: Mapping[str, str] | None = None) -> tuple[Path, list[str]]:
    """Write a copy of the requirements table with result columns filled in (existing ones reused)."""
    from ..tabular import CellEdit, patch_xlsx
    from .config import normalize_header

    table = reqset.table
    write_back = reqset.config.requirements.write_back or DEFAULT_WRITE_BACK
    header = [normalize_header(text) for text in table.row_texts(reqset.header_row)]
    columns: dict[str, int] = {}
    next_col = max(len(header), 1) + 1
    edits: list = []
    for field_name, name in write_back.items():
        wanted = normalize_header(name)
        if wanted in header:
            columns[field_name] = header.index(wanted) + 1
        else:
            columns[field_name] = next_col
            edits.append(CellEdit(reqset.header_row, next_col, name))
            next_col += 1
    values = write_back_values(reqset, results, t0, aggregate)
    for row, fields in values.items():
        for field_name, col in columns.items():
            if field_name in fields:
                value = fields[field_name]
                edits.append(CellEdit(row, col, value if not isinstance(value, float) or math.isfinite(value)
                                      else None))
    notes: list[str] = []
    if table.format == "xlsx":
        notes = patch_xlsx(table.path, dst, sheet=table.sheet, edits=edits)
        return dst, notes
    # CSV: rewrite with the same delimiter and encoding, cells as text
    grid = [[cell.text if cell is not None else "" for cell in row] for row in table.rows]
    for edit in edits:
        while len(grid) < edit.row:
            grid.append([])
        row = grid[edit.row - 1]
        while len(row) < edit.col:
            row.append("")
        value = edit.value
        text = "" if value is None else (f"{value:.6g}" if isinstance(value, float) else str(value))
        row[edit.col - 1] = _safe(text)  # only written values are guarded; the user's own cells stay as they are
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=table.delimiter or ",", lineterminator="\n")
    width = max(len(row) for row in grid) if grid else 0
    for row in grid:
        writer.writerow(row + [""] * (width - len(row)))
    encoding = "utf-8" if (table.encoding or "utf-8") == "utf-8" else table.encoding
    write_atomic(dst, buffer.getvalue().encode(encoding, errors="replace"))
    return dst, notes
