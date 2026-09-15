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

## Planned v0.1 scope

- Inputs: MATLAB `.mat` (v5/v7 and v7.3), HDF5, CSV, and NumPy arrays in memory.
- Policy file with hard guarantees and soft preferences, units included (`65 kPa`, `7 deg`, `100 ms`, `2 MiB`).
- `baslt compile`, `baslt verify`, `baslt report`, `baslt sweep`, `baslt inspect`, `baslt policy init`,
  `baslt explain`.
- A self-contained HTML viewer for one run and a dashboard for a whole sweep.
- A MATLAB package that can open artifacts without Python and call the compiler through Python when available.

## License

MIT. See `LICENSE`.
