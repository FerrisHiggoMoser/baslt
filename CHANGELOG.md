# Changelog

## Unreleased

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
