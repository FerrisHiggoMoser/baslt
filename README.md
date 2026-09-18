# Example output of `baslt check`

Generated output only — no source. The code is on `main`.

Both HTML pages are self-contained: no CDN, no external requests. Download and
open them in any browser, offline.

## single-run/

`baslt check sim.h5 -r examples/rocket_requirements.csv -m examples/rocket_mapping.yaml`

on a 120 s, 8-signal run with a dynamic-pressure spike injected
(`examples/rocket_sim.py --anomaly q_spike`). Result: FAIL, 13 pass, 1 warn, 2 fail.

| file | what it is |
| --- | --- |
| `report.html` | one page: what went wrong, phase timeline, a plot per requirement |
| `results.xlsx` | Summary, Results, Cases, Violations, Events, Issues |
| `results.csv`, `results.json` | the same results, flat |
| `rocket_requirements.checked.csv` | the requirements table with Verdict, Result, Margin and Evidence filled in |

## batch/

The same check over 40 generated runs (`examples/rocket_batch.py --runs 40`),
with run parameters from `params.csv`. Result: 7 runs fail, 33 warn.

| file | what it is |
| --- | --- |
| `index.html` | dashboard: pass rate per requirement, run x requirement matrix, envelope over all runs, margin against payload, filterable run table |
| `runs/run_XXXX.html` | the per-run page, linked from the matrix |
| `runs/run_XXXX.json` | the per-run results |
| `results.xlsx` | Summary, Matrix, Margins, Requirements, Violations, Runs |
| `summary.json`, `index.jsonl` | the dashboard data |

Start at `batch/index.html` — the run links only resolve if `runs/` sits next to it.

## Verdicts against ground truth

`rocket_batch.py` writes `expected.csv` with the true verdict for LV-001, LV-005
and LV-007 in every run. All 120 verdicts matched, and the 5 runs with injected
anomalies are exactly the ones flagged.
