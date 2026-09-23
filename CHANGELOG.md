# Changelog

## 0.1.0 - 2026-09-23

- `baslt check` tests runs against requirement tables (Excel, CSV, ReqIF) with cases per phase, event window, run
  parameter or limit curve; limit, assert, duration, value and event checks with tolerances, warning margins, gap and
  missing-event rules; exact violation times; results as text, JSON, CSV, a workbook and the requirements table with
  verdicts written back (Excel files change only in the result cells).
- A report page per run with linked, zoomable plots of every requirement, and full-resolution windows around
  failures, in one file that loads nothing from the network.
- Batches of runs in parallel processes, with parameter tables, resume, crash isolation, a batch workbook and a
  dashboard (verdict matrix, each requirement over all runs, margins against parameters).
- `--archive` compiles each checked run with a policy made from its requirements.
- `baslt requirements init` and `lint`, an expression language with units, events and aggregates, run parameters
  read from HDF5 and MAT-files, and a standard-library reader, writer and patcher for `.xlsx`.
- Batch example `examples/rocket_batch.py`; `rocket_sim.simulate` takes a thrust scale and a payload.
- Atomic writes give files the usual permissions instead of owner-only.
- Soft layer: every signal gets a nested min/max preview weighted by its priority, and the compiler searches for the
  largest preview that fits the budget, measuring each trial exactly.
- The manifest splits the budget into required, discretionary and overhead bytes and hard and soft samples per
  signal, and records each signal's reconstruction error; the verifier checks all of it.
- `baslt explain` suggests the smallest budget that works and, given the source, measures relaxing each requirement.
- State transitions verify against the source when a value changes between samples sharing a timestamp.
- A MATLAB v7.3 file saved with an `.h5` name opens as a MAT-file, a clock in a parent HDF5 group serves the groups
  below it, `policy init` excludes signals without a clock so its policy compiles, and a file holding only MATLAB
  objects says why nothing was found.
- Local extrema with prominence and separation, checked independently on the retained samples.
- Events (falling, rising and equals triggers with windows) and sync groups, with verifier checks.
- MATLAB MAT-file sources, v5 through scipy and v7.3 through h5py, including Simulink structure-with-time logs.
- `baslt policy init` writes a starter policy that compiles as written.
- The rocket example writes MAT-files in both versions.
- Compile and verify the first hard-contract artifacts through Python and the CLI.
- Independent contract verification, atomic output, error reports and deterministic rocket evidence.
- NumPy, CSV and HDF5 sources; validated policies, units, source digests and the v1 container.
