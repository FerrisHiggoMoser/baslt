# Command-line reference

Every command accepts `--json` (machine-readable output with a top-level `"status"`), `-q` (quiet) and `--help`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success, or verification passed (warnings allowed unless `--strict`). |
| 1 | A contract or integrity check failed, `--strict` saw a warning, a requirement failed (`check`), or `--fail-on` matched. |
| 2 | Infeasible budget, or an internal compile error. |
| 3 | Usage, policy, source or container input error. |

## `baslt compile`

```
baslt compile SOURCE --policy POLICY [-o OUT] [--max-size SIZE] [--codec deflate|store|zstd]
              [--hash full|sampled|none] [--threads N] [--no-self-verify] [--error-report PATH]
```

Compiles one run. `OUT` defaults to `SOURCE` with the extension replaced by `.baslt`. `--max-size` overrides
`artifact.max_size` and the artifact records `budget.source: cli`. Everything the policy protects is kept; the rest
of the budget goes to preview samples, and the artifact never exceeds the budget. On an infeasible budget nothing is
written, the itemized minimum-size report is printed (or written to `--error-report`), and the exit code is 2.

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

## `baslt check`

```
baslt check RUN... -r REQUIREMENTS [-m MAP]... [--set KEY=VALUE]... [--params TABLE] [-o OUT] [--jobs auto|N]
            [--resume]
            [--fail-on fail|warn|none] [--pages failed|all|none] [--only ID] [--no-html] [--no-xlsx]
            [--no-annotate] [--archive [--max-size SIZE]] [--hash sampled|full|none] [--all]
```

Tests runs against a requirements table (`.xlsx`, `.csv`, `.reqif` or `.reqifz`). One `RUN` file is checked in this
process and its results go to `OUT` (default `<run>.check/`): `report.html`, `results.xlsx`, `results.json`,
`results.csv` and `<requirements>.checked.<ext>`, the table with the verdicts written in. Several files, a folder
or a glob pattern are a batch: runs are checked in `--jobs` processes, progress goes to stderr, and `OUT` (default
`<folder>.check/`) gets `index.html`, `results.xlsx`, `summary.json`, `index.jsonl`, the checked copy and
`runs/<id>.json` (plus `runs/<id>.html` per `--pages`). `--resume` skips runs whose stored result is up to date.

The terminal shows every requirement that did not pass (`--all` shows every one) with its worst value, limit,
margin, time and where it happened:

```
VERDICT  ID      TITLE                 WORST      LIMIT      MARGIN              AT
FAIL     LV-001  Max dynamic pressure  72.66 kPa  <= 70 kPa  -2.66 kPa (-3.8 %)  T+60.953 s  ascent, high_q; 56.0 s after pitch_start
```

`-m` gives a mapping (YAML or JSON) and may be repeated: later files are laid over earlier ones, key by key, so a
vehicle build's file need only carry what it changes. `--set key=value` (repeatable) changes one setting and wins
over the files. Config sheets inside the workbook sit underneath both. `--params` is a table with one
row per run. `--only` (repeatable, globs) checks a subset of ids. `--archive` also compiles each run to a `.baslt`
artifact protecting what the requirements check; `--max-size` is its budget. `--hash` sets how the run file is
fingerprinted in the results.

Exit codes: 0 when nothing failed, 1 when a requirement failed (or warned, with `--fail-on warn`; never with
`--fail-on none`), 3 for unusable requirements, mappings or options and when nothing failed but something ended in
ERROR, 2 for an internal error. See [requirements.md](requirements.md).

## `baslt requirements init`

```
baslt requirements init [-o reqs.xlsx] [--template generic|polarion] [--mapping-output MAP] [--source RUN] [--force]
```

Writes a requirements template: an `.xlsx` workbook with example rows, the config sheets (Signals, Events,
Conditions, Curves, Units, Settings) and a Guide sheet, or a `.csv` table with its mapping next to it
(`--mapping-output`, YAML when PyYAML is installed, else JSON). The examples check `examples/rocket_sim.py`; with
`--source` the Signals sheet lists the run's own signals instead. `polarion` lays the table out like a Polarion
export with verification fields, a `where` filter and `write_back`.

## `baslt requirements lint`

```
baslt requirements lint REQUIREMENTS [-m MAP]... [--set KEY=VALUE]... [--source RUN] [--params TABLE] [--only ID]
```

Reads the table and mapping and reports every problem with its cell (`reqs.xlsx:Requirements!E12`): unknown
types, limits that do not parse, rows not covered, mapping errors. With `--source`, names, units and limits are also
resolved against that run, as `check` would; with `--params` too, `param.` names are checked. Exit code 0 when
there are no errors (warnings allowed), 3 otherwise.

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
found, a hint when signals have no unit (MAT-files never do), and the signals excluded (`signals.exclude`) because no
clock was found. A file in which no signal has a clock is refused with a message saying how to name one. The format
follows the output suffix (`.json` gives JSON) unless `--format` is given. An existing file is only replaced with
`--force`. The starter policy always compiles as written.

## `baslt explain`

```
baslt explain ERROR_REPORT.json [--source SOURCE --policy POLICY] [--max-size SIZE] [--json]
```

Turns an infeasible-budget report into suggestions. The report alone gives the smallest budget that works (rounded up
to 64 KiB) and the largest requirements with the relaxation that usually helps each operator:

```
The hard requirements need 24,283 bytes; the budget is 12,288 (11,995 over).

Smallest budget that works: max_size: 64 KiB

Largest requirements
  hard.aoa.window_extrema         241 samples  3,811 bytes  use a longer interval: every window keeps its own maximum and minimum
  hard.aoa.threshold_crossing[1]  227 samples  2,748 bytes  add hysteresis or debounce: a noisy signal near the level crosses it many times
```

With `--source` and `--policy` (both are needed) each relaxation is measured: longer windows (×2, ×4), larger
prominence, a separation, hysteresis (1 % of the signal's range, then 2 %) and debounce (10 and 20 sample spacings)
for thresholds, a longer `min_duration` for violations, shorter event windows, and demoting a requirement or removing
an event. The eight largest requirements and every event are tried. Each row gives the smallest artifact the changed
policy allows, sized the way the compiler sizes it; `FITS` leaves 64 bytes for the policy text the edit itself adds.
The mildest fitting change per requirement is then combined, loosening before dropping, until the run fits:

```
Measured on the source (smallest artifact; budget 12,288 bytes, FITS leaves 64 bytes for the policy edit)
  FITS  11,743  hard.aoa.threshold_crossing[1]    demote to soft
        14,952  hard.aoa.threshold_crossing[1]    debounce 0 s -> 0.02 s
        20,349  hard.aoa.window_extrema           demote to soft
        21,590  hard.aoa.window_extrema           interval 1 s -> 4 s
        ...
  ... 12 more with --json

Together (fits at 11,654 bytes):
  1. hard.aoa.threshold_crossing[1]: debounce 0 s -> 0.02 s
  2. hard.aoa.window_extrema: interval 1 s -> 4 s
  3. hard.aoa.threshold_crossing[0]: debounce 0 s -> 0.02 s
```

(The rocket example with `--anomaly alpha_chatter`, `tests/golden/m5_extrema_crossings.yaml` and `--max-size 12KiB`.)

The budget measured against is `--max-size`, else the report's own when it came from `compile --max-size`, else the
policy's. Nothing is written; the suggestions are edits to make in the policy file.
