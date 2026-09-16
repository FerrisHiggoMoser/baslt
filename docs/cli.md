# Command-line reference

Every command accepts `--json` (machine-readable output with a top-level `"status"`), `-q` (quiet) and `--help`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success, or verification passed (warnings allowed unless `--strict`). |
| 1 | A contract or integrity check failed, `--strict` saw a warning, or `--fail-on` matched. |
| 2 | Infeasible budget, or an internal compile error. |
| 3 | Usage, policy, source or container input error. |

## `baslt compile`

```
baslt compile SOURCE --policy POLICY [-o OUT] [--max-size SIZE] [--codec deflate|store|zstd]
              [--hash full|sampled|none] [--threads N] [--no-self-verify] [--error-report PATH]
```

Compiles one run. `OUT` defaults to `SOURCE` with the extension replaced by `.baslt`. `--max-size` overrides
`artifact.max_size` and the artifact records `budget.source: cli`. On an infeasible budget nothing is written, the
itemized minimum-size report is printed (or written to `--error-report`), and the exit code is 2.

## `baslt verify`

```
baslt verify ARTIFACT [--source SOURCE] [--strict] [--json]
```

Checks the artifact against its embedded policy and prints one row per check:

```
Policy   flight_review   sha256 3f9a1c0b…
Source   run.h5          sha256 (full) 5e02…
Artifact 1.86 MiB of 2.00 MiB   ratio 4.64e+03

STATUS  CHECK                                         BASIS     SOURCE         ARTIFACT       TOLERANCE
PASS    structure                                     artifact  -              -              -
PASS    hard.q_dyn.global_extrema max                 attested  68142.3 Pa     68142.3 Pa     0
PASS    hard.q_dyn.threshold_crossing[0] crossings    artifact  4              4              5 ms
WARN    sync.flight_dynamics unaligned                artifact  -              3              0
N/A     hard.skin_temp.global_extrema                 -         -              -              -

Result: PASS WITH WARNINGS (11 pass, 1 warn, 1 n/a, 0 fail)
```

With `--source`, attested checks are recomputed from the source and their basis becomes `source`.

## `baslt report`

```
baslt report ARTIFACT [-o OUT.html]
baslt report SWEEP_DIR [-o OUT.html]
```

Writes a single self-contained HTML file: a run view for an artifact, or the sweep dashboard for a sweep directory.

## `baslt sweep`

```
baslt sweep INPUTS... --policy POLICY -o OUTDIR [--jobs auto|N] [--resume] [--watch] [--params runs.csv]
            [--max-size SIZE] [--hash sampled|full|none] [--fail-on limit_violations|warning|compiler_failure]
```

`INPUTS` are files, directories or glob patterns. Runs are compiled out of band at low CPU and I/O priority with one
thread per worker. `OUTDIR` receives `index.jsonl` (one summary per run, appended as runs finish), `summary.json`,
`sweep.html`, `runs/<id>.baslt`, `runs/<id>.html` for limit-violation runs, and `runs/<id>.error.json` for failures.
`--resume` skips runs already in `index.jsonl`. `--params` joins a CSV with a `run` column to add parameter columns.

## `baslt inspect`

```
baslt inspect SOURCE|ARTIFACT [--json]
```

Lists signals with shape, dtype, unit, kind and time reference, or summarizes an artifact.

## `baslt policy init`

```
baslt policy init SOURCE [-o POLICY] [--format yaml|json] [--force]
```

Writes a starter policy for the signals in `SOURCE`: global extrema for every continuous or vector signal, state
transitions for every discrete one, and an artifact budget of a fiftieth of the source (rounded up to a power of two,
between 256 KiB and 8 MiB). Without `-o` the policy is printed. The YAML form starts with a table of the signals
found, a hint when signals have no unit (MAT-files never do), and the names of signals left out because no clock was
found. The format follows the output suffix (`.json` gives JSON) unless `--format` is given. An existing file is only
replaced with `--force`. The starter policy always compiles as written.

## `baslt explain`

```
baslt explain ERROR_REPORT.json [--source SOURCE --policy POLICY]
```

Turns an infeasible-budget report into concrete suggestions (shorter windows, looser trajectory error, hysteresis for
chattering thresholds, demoting requirements to soft, or the smallest budget that works).
