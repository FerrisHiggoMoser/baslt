"""Checking small requirement tables against small in-memory runs, for requirement-check tests."""

from __future__ import annotations

import csv
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from baslt.reqs.config import load_config
from baslt.reqs.load import load_requirements
from baslt.reqs.run import check_run
from reference.reqs_runs import run

COLUMNS = ("ID", "Title", "Check", "Type", "Limit", "Unit", "When", "Applies to", "Tolerance", "Warn margin",
           "Count", "Case", "Severity")
KEYS = {"id": "ID", "title": "Title", "check": "Check", "type": "Type", "limit": "Limit", "unit": "Unit",
        "when": "When", "applies_to": "Applies to", "tolerance": "Tolerance", "margin": "Warn margin",
        "count": "Count", "case": "Case", "severity": "Severity"}


def write_rows(path: Path, rows: Sequence[Mapping[str, object]], *, delimiter: str = ",") -> Path:
    """Rows as `{"id": ..., "check": ..., ...}` (keys from KEYS) into a CSV requirements table."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(COLUMNS)
        for row in rows:
            unknown = set(row) - set(KEYS)
            if unknown:
                raise KeyError(f"unknown row keys {sorted(unknown)}")
            cells = {KEYS[key]: "" if value is None else str(value) for key, value in row.items()}
            cells.setdefault("Title", str(row.get("id", "")))
            writer.writerow([cells.get(name, "") for name in COLUMNS])
    return path


def load_rows(rows, mapping: Mapping | None = None, *, folder: Path | None = None):
    """A RequirementSet from rows; the CSV lives in `folder` or a temporary folder."""
    if folder is not None:
        path = write_rows(Path(folder) / "reqs.csv", rows)
        return load_requirements(path, load_config(mapping or {}, workbook=path))
    with tempfile.TemporaryDirectory() as tmp:
        path = write_rows(Path(tmp) / "reqs.csv", rows)
        return load_requirements(path, load_config(mapping or {}, workbook=path))


def check(rows, signals: Mapping, mapping: Mapping | None = None, params: Mapping | None = None):
    """RunResult of `rows` on a run built from `signals` (as in reference.reqs_runs.run)."""
    reqset = load_rows(rows, mapping)
    result, _ = check_run(run(**signals), reqset, params=params)
    return result


def results(rows, signals: Mapping, mapping: Mapping | None = None, params: Mapping | None = None) -> dict:
    """{requirement id: RequirementResult}."""
    outcome = check(rows, signals, mapping, params)
    assert outcome.error is None, outcome.error
    return {r.id: r for r in outcome.results}


def one(row: Mapping, signals: Mapping, mapping: Mapping | None = None, params: Mapping | None = None):
    """The RequirementResult of a single-row table (the row's id defaults to R-1)."""
    row = {"id": "R-1", **row}
    return results([row], signals, mapping, params)[row["id"]]
