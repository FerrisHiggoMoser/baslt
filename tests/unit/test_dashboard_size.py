"""The batch dashboard stays small for a thousand runs."""

from __future__ import annotations

import time

import numpy as np
import pytest

from baslt.reqs.aggregate import BatchSummary
from baslt.reqs.batch import BatchRun, _quantize
from reference.reqs_check import load_rows

pytestmark = pytest.mark.minimal

VERDICTS = ("pass", "warn", "fail", "not_applicable", "error")


def fake_batch(tmp_path, runs: int, reqs: int, signals: int = 10, seed: int = 1):
    rng = np.random.default_rng(seed)
    reqset = load_rows([{"id": f"R-{k:03d}", "title": f"Requirement {k}", "check": f"s{k % signals}",
                         "limit": "<= 5"} for k in range(reqs)], folder=tmp_path)
    out = []
    for i in range(runs):
        series = {}
        for k in range(signals):
            walk = np.cumsum(rng.normal(size=256)) * 0.1
            series[f"sig{k}#0"] = {"name": f"s{k}", "unit": "m", "labels": None, "a": -1.0, "b": 119.0 + i % 3,
                                   **_quantize(walk - 0.05, walk + 0.05)}
        results = {}
        plots = {}
        for k in range(reqs):
            verdict = VERDICTS[0] if rng.random() > 0.03 else VERDICTS[int(rng.integers(1, 5))]
            margin = float(rng.normal(1.0, 0.5))
            results[f"R-{k:03d}"] = {"v": verdict, "value": 5.0 - margin, "margin": margin, "pct": margin / 5,
                                     "at": 1.0, "first": None, "reason": "limit exceeded" if verdict == "fail" else
                                     None, "case": None, "limit": "<= 5 m", "unit": "m", "runs": 0}
            plots[f"R-{k:03d}"] = {"s": f"sig{k % signals}#0", "upper": 5.0}
        status = "fail" if any(r["v"] == "fail" for r in results.values()) else "pass"
        summary = {"id": f"run_{i:04d}", "source": f"runs/run_{i:04d}.h5", "status": status,
                   "counts": {v: sum(r["v"] == v for r in results.values()) for v in VERDICTS},
                   "params": {"payload": float(rng.uniform(0, 2000)), "thrust_scale": float(rng.uniform(0.99, 1.04)),
                              "vehicle": "heavy" if i % 3 == 0 else "light"},
                   "error": None, "t0": 1.0, "duration": 120.0, "results": results, "violations": {},
                   "series": series, "plots": plots, "page": f"runs/run_{i:04d}.html" if status == "fail" else None,
                   "cached": False, "timing": {}, "key": "x", "format": "hdf5"}
        out.append(BatchRun(i, summary))
    return BatchSummary(reqset, out, output=tmp_path / "out", root=tmp_path / "runs", seconds=1.0, jobs=4)


def test_a_thousand_runs_and_fifty_requirements(tmp_path):
    batch = fake_batch(tmp_path, 1000, 50)
    started = time.perf_counter()
    path = batch.write_dashboard(tmp_path / "index.html")
    seconds = time.perf_counter() - started
    size = path.stat().st_size
    assert size <= 1_500_000, size
    assert seconds < 10
    text = path.read_text(encoding="utf-8")
    assert text.count('<tr class="pick"') == 50


@pytest.mark.slow
def test_a_thousand_runs_and_two_hundred_requirements(tmp_path):
    batch = fake_batch(tmp_path, 1000, 200, signals=40)
    path = batch.write_dashboard(tmp_path / "index.html")
    assert path.stat().st_size <= 4_000_000
