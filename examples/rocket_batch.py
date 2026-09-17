"""A batch of simulated ascents with different payloads and engines, for `baslt check` on many runs.

    python examples/rocket_batch.py --runs 40 --out examples/out/batch
    baslt check examples/out/batch/runs -r examples/rocket_requirements.csv -m examples/rocket_mapping.yaml \\
        --params examples/out/batch/params.csv -o examples/out/batch-check

Each run gets a payload (kg) and a thrust scale; a few get the q-spike or AoA-chatter anomaly. `params.csv` lists
them, one row per run, and `expected.csv` says which runs should fail the velocity-at-MECO, altitude and max-q
requirements, worked out here straight from the simulated arrays.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MECO_THRUST = 1.0e6  # N, the MECO event of rocket_mapping.yaml
V_MECO_MIN = 1800.0  # m/s, LV-007
ALTITUDE_MIN = 40_000.0  # m, LV-005
Q_MAX = 70_000.0  # Pa, LV-001


def _simulator():
    spec = importlib.util.spec_from_file_location("rocket_sim", HERE / "rocket_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(runs: int, seed: int) -> list[dict]:
    """The run settings: payload, thrust scale and anomaly for every run."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(runs):
        anomaly = "q_spike" if k % 17 == 5 else ("alpha_chatter" if k % 23 == 11 else "")
        out.append({
            "run": f"run_{k:04d}",
            "payload": float(np.round(rng.uniform(0.0, 2000.0), -1)),
            "thrust_scale": float(np.round(rng.uniform(0.99, 1.04), 4)),
            "vehicle": "heavy" if k % 3 == 0 else "light",
            "anomaly": anomaly,
            "seed": int(seed + k),
        })
    return out


def expected(data: dict) -> dict:
    """Which of LV-007, LV-005 and LV-001 this run should fail, from the arrays themselves."""
    t = data["t"]
    thrust = data["prop/thrust"]
    speed = np.linalg.norm(data["nav/velocity"], axis=1)
    below = np.flatnonzero((thrust[:-1] >= MECO_THRUST) & (thrust[1:] < MECO_THRUST))
    k = int(below[0])
    frac = (thrust[k] - MECO_THRUST) / (thrust[k] - thrust[k + 1])
    t_meco = t[k] + frac * (t[k + 1] - t[k])
    v_meco = float(np.interp(t_meco, t, speed))
    altitude = data["nav/position"][:, 2]
    lifted = np.flatnonzero(altitude > 1.0)
    t_liftoff = t[lifted[0]]
    ascent = (t >= t_liftoff) & (t < t_meco)
    return {
        "LV-007": "fail" if v_meco < V_MECO_MIN else "pass",
        "LV-005": "fail" if float(np.max(altitude)) < ALTITUDE_MIN else "pass",
        "LV-001": "fail" if float(np.max(data["aero/q"][ascent])) > Q_MAX else "pass",
    }


def write(out: Path, runs: int, seed: int, rate: float, fmt: str) -> list[dict]:
    sim = _simulator()
    folder = out / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    rows = plan(runs, seed)
    truth = []
    for row in rows:
        data = sim.simulate(seed=row["seed"], rate=rate, anomaly=row["anomaly"] or None,
                            thrust_scale=row["thrust_scale"], payload=row["payload"])
        path = folder / f"{row['run']}.{fmt}"
        if fmt == "h5":
            sim.write_h5(path, data)
        else:
            sim.write_csv(path, data)
        truth.append({"run": row["run"], **expected(data)})
    with open(out / "params.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / "expected.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    return truth


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write a batch of simulated ascents with a parameter table.")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--rate", type=float, default=200.0, help="samples per second")
    parser.add_argument("--format", choices=("h5", "csv"), default=None,
                        help="run file format (default: h5 when h5py is installed, else csv)")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    fmt = args.format
    if fmt is None:
        fmt = "h5" if importlib.util.find_spec("h5py") is not None else "csv"
    truth = write(args.out, args.runs, args.seed, args.rate, fmt)
    failing = sum(1 for row in truth if "fail" in row.values())
    print(f"wrote {args.runs} runs to {args.out / 'runs'} ({failing} expected to fail), params.csv and "
          "expected.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
