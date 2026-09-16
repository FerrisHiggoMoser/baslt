# Baslt

**Status: pre-alpha, under active construction.** Nothing here is ready for real review work yet.

Baslt compiles a large simulation or test run into a small review artifact, and proves that the engineering facts
you care about survived the reduction. You write a policy that says what must never be lost — global extrema,
threshold crossings, limit violations, event windows, trajectory accuracy, synchronized signals — and give it a byte
budget. Baslt keeps exactly what the policy protects, fills the rest of the budget with a shape-preserving preview,
and writes a manifest with the evidence. If the budget cannot hold everything the policy protects, compilation fails
and tells you the smallest size that would work, instead of quietly dropping something.

It is designed to stay out of the way of simulation pipelines: runs are compiled after the fact, one tiny artifact
per run, so a sweep of thousands of runs can be triaged in a browser without opening the raw logs.

## Available now

The first end-to-end compiler supports NumPy arrays, CSV, HDF5 and MATLAB MAT-files (v5 to v7.3, including
Simulink "structure with time" logs), with global, window and local extrema (prominence and separation), threshold
crossings, violations, discrete state transitions, events with windows around their triggers, and sync groups that
keep related signals sampled at the same instants. `compile`, `verify`, `inspect` and `policy init` are available in the CLI. Compilation verifies
its output before writing it. Unsupported policy features produce an explicit error.

```sh
baslt policy init run.mat -o review.yaml     # starter policy for every signal in the file
baslt compile run.csv --policy review.json -o run.baslt
baslt verify run.baslt --source run.csv
baslt inspect run.baslt --json
```

```python
import baslt

result = baslt.compile({"t": t, "pressure": pressure}, {
    "version": 1,
    "hard": {"pressure": {"global_extrema": {}}},
}, output="run.baslt")
verification = baslt.verify_artifact("run.baslt")
```

`baslt.api.verify` also accepts `source=` for source-backed checks. For a pipeline hook, use
`on_error="record", suggestions="off", threads=1`; the returned result carries failures and the adjacent
`.error.json` explains them. See [the CLI reference](docs/cli.md) and [contracts](docs/contracts.md).

## Planned v0.1 scope

- Inputs: MATLAB `.mat` (v5/v7 and v7.3), HDF5, CSV, and NumPy arrays in memory. MAT-files carry no units, so
  declare them under `signals.decl` when a limit has one.
- Policy file with hard guarantees and soft preferences, units included (`65 kPa`, `7 deg`, `100 ms`, `2 MiB`).
- `baslt compile`, `baslt verify`, `baslt report`, `baslt sweep`, `baslt inspect`, `baslt policy init`,
  `baslt explain`.
- A self-contained HTML viewer for one run and a dashboard for a whole sweep.
- A MATLAB package that can open artifacts without Python and call the compiler through Python when available.

## License

MIT. See `LICENSE`.
