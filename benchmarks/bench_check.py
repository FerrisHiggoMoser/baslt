"""Timings of requirement checks: linting a large table, one large run, a long run, the batch workbook and a batch.

    python benchmarks/bench_check.py            # every scenario at full size
    python benchmarks/bench_check.py --quick    # smaller sizes, for a smoke run

Each line gives the measured seconds and the target from docs/performance.md.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from baslt.reqs.api import check, lint, load  # noqa: E402
from baslt.reqs.run import check_run  # noqa: E402


def write_table(path: Path, rows: list[list[str]]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Title", "Check", "Type", "Limit", "Unit", "When", "Tolerance", "Warn margin"])
        writer.writerows(rows)
    return path


def requirement_rows(count: int, signals: int) -> list[list[str]]:
    """A mix of limit, duration, value and assert checks over `signals` signals."""
    rows = []
    for k in range(count):
        s = f"s{k % signals}"
        form = k % 5
        if form == 0:
            rows.append([f"R-{k:04d}", f"Limit {k}", s, "", "[-3, 3]", "", "t > 1 s", "10 ms", "5 %"])
        elif form == 1:
            rows.append([f"R-{k:04d}", f"Peak {k}", s, "max", "<= 4", "", "", "", ""])
        elif form == 2:
            rows.append([f"R-{k:04d}", f"Time above {k}", f"{s} > 1", "", "<= 60 s", "", "", "", ""])
        elif form == 3:
            rows.append([f"R-{k:04d}", f"Mean {k}", f"mean(abs({s}))", "", "<= 2", "", "t < 100 s", "", ""])
        else:
            rows.append([f"R-{k:04d}", f"Hold {k}", f"abs({s}) < 5", "assert", "", "", "", "", ""])
    return rows


def signals(n: int, count: int, rate: float, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    out: dict[str, np.ndarray] = {"t": t}
    for k in range(count):
        walk = np.cumsum(rng.normal(0.0, 1.0, n)) / np.sqrt(n) * 2.0
        out[f"s{k}"] = walk + 0.3 * np.sin(2 * np.pi * t / (7.0 + k))
    return out


def timed(label: str, target: float | None, fn):
    started = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - started
    if target is None:
        print(f"{label:<58} {seconds:8.2f} s", flush=True)
    else:
        status = "ok" if seconds <= target else "SLOW"
        print(f"{label:<58} {seconds:8.2f} s   target {target:6.1f} s   {status}", flush=True)
    return value, seconds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller sizes")
    parser.add_argument("--keep", type=Path, help="keep the generated files in this folder")
    args = parser.parse_args(argv)
    scale = 0.1 if args.quick else 1.0
    folder = Path(args.keep or tempfile.mkdtemp(prefix="baslt-bench-"))
    folder.mkdir(parents=True, exist_ok=True)
    print(f"baslt requirement-check benchmark ({'quick' if args.quick else 'full'}), files in {folder}")
    print(f"{os.cpu_count()} CPUs, numpy {np.__version__}, Python {sys.version.split()[0]}\n")

    # 1. lint a large table
    table = write_table(folder / "lint.csv", requirement_rows(int(2000 * scale) or 20, 50))
    timed(f"lint {int(2000 * scale)} rows", 1.0, lambda: lint(table))

    # 2. one run: 120 s at 1 kHz, 50 signals, 200 requirements
    n = int(120_000 * scale)
    data = signals(n, 50, 1000.0)
    reqs = write_table(folder / "reqs.csv", requirement_rows(200, 50))
    reqset = load(reqs)
    (result, _), seconds = timed(f"evaluate 200 requirements, 50 signals x {n} samples", 1.5,
                                 lambda: check_run(data, reqset))
    verdicts = {v: sum(r.verdict == v for r in result.results) for v in ("pass", "warn", "fail", "error")}
    print(f"{'':<4}verdicts {verdicts}")
    timed("  ... with report, workbook and checked copy", 3.0,
          lambda: check(data, reqs, output=folder / "single"))

    # 3. a long run: 10M samples x 20 signals, 50 requirements
    n_long = int(10_000_000 * scale)
    long_data = signals(n_long, 20, 1000.0, seed=2)
    long_reqs = write_table(folder / "long.csv", requirement_rows(50, 20))
    timed(f"evaluate 50 requirements, 20 signals x {n_long} samples", 30.0,
          lambda: check_run(long_data, load(long_reqs)))
    del long_data

    # 4. the batch workbook for a 1000 x 200 matrix
    sys.path.insert(0, str(HERE.parent / "tests"))
    sys.path.insert(0, str(HERE.parent / "tests" / "unit"))
    from test_dashboard_size import fake_batch  # noqa: E402

    runs = int(1000 * scale) or 10
    batch = fake_batch(folder, runs, 200, signals=40)
    timed(f"results.xlsx for {runs} runs x 200 requirements", 3.0,
          lambda: batch.write_xlsx(folder / "batch.xlsx"))
    timed(f"index.html for {runs} runs x 200 requirements", 10.0,
          lambda: batch.write_dashboard(folder / "batch.html"))

    # 5. a batch of simulated ascents, then resumed
    spec = importlib.util.spec_from_file_location("rocket_batch", HERE.parent / "examples" / "rocket_batch.py")
    rocket = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rocket)
    count = int(1000 * scale) or 10
    fmt = "h5" if importlib.util.find_spec("h5py") else "csv"
    timed(f"write {count} simulated runs ({fmt}, 200 Hz)", None,
          lambda: rocket.write(folder / "gen", count, 100, 200.0, fmt))
    examples = HERE.parent / "examples"
    mapping = examples / ("rocket_mapping.yaml" if importlib.util.find_spec("yaml") else "rocket_mapping.json")

    def run_batch(resume: bool):
        return check(folder / "gen" / "runs", examples / "rocket_requirements.csv", mapping=mapping,
                     params=folder / "gen" / "params.csv", output=folder / "batch-out", jobs="auto", resume=resume)

    first, seconds = timed(f"check {count} runs, jobs auto", 240.0 * count / 1000, lambda: run_batch(False))
    print(f"{'':<4}{first.batch.counts()}")
    timed(f"check {count} runs again with --resume", 5.0, lambda: run_batch(True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
