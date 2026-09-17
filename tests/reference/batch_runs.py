"""Small batches for batch-check tests: CSV runs whose peak depends on a parameter, and a requirements table."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from reference.reqs_check import write_rows

ROWS = [
    {"id": "PEAK", "title": "Peak", "check": "x", "limit": "<= 5 m", "margin": "0.5 m"},
    {"id": "MEAN", "title": "Mean", "check": "mean(x)", "limit": "<= 10 m"},
    {"id": "GATE", "title": "Gate closes", "check": "gate == 0", "type": "assert", "when": "t > 9 s"},
]


def write_run(path: Path, peak: float, *, gate_end: float = 9.0, n: int = 101) -> Path:
    t = np.linspace(0.0, 10.0, n)
    x = peak * np.sin(np.pi * t / 10.0)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t [s]", "x [m]", "gate"])
        for i in range(n):
            writer.writerow([f"{t[i]:.6g}", f"{x[i]:.9g}", 1 if t[i] < gate_end else 0])
    return path


def write_batch(folder: Path, peaks: dict[str, float], *, params: bool = True, gate_end: dict | None = None
                ) -> tuple[Path, Path, Path | None]:
    """(runs folder, requirements table, params table) for runs named by `peaks`."""
    runs = folder / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for name, peak in peaks.items():
        write_run(runs / f"{name}.csv", peak, gate_end=(gate_end or {}).get(name, 9.0))
    table = write_rows(folder / "reqs.csv", ROWS)
    table_params = None
    if params:
        table_params = folder / "params.csv"
        with open(table_params, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run", "peak", "family"])
            for k, (name, peak) in enumerate(peaks.items()):
                writer.writerow([name, peak, "a" if k % 2 == 0 else "b"])
    return runs, table, table_params
