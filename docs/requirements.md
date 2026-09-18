# Requirement checks

`baslt check` tests simulation runs against a requirements table and says what went wrong: which requirement
failed, by how much, when, and in which flight phase. It reads the same run files as the compiler (HDF5, MATLAB
MAT-files, CSV, NumPy arrays) and requirements from Excel, CSV or a ReqIF export (Polarion, DOORS). One run gives a
results workbook, a report page with a plot per requirement, and a copy of the requirements table with the verdicts
written in. Many runs give a dashboard with a verdict matrix and each requirement over all runs.

This is a layer next to the compiler. It reads the full run data and does not change the `.baslt` format;
`--archive` compiles a verified artifact of each checked run as well.

## Quick start

```sh
baslt requirements init -o reqs.xlsx              # a template workbook with example rows and a guide sheet
baslt requirements lint reqs.xlsx --source run.h5 # every name, unit and limit resolved against a real run
baslt check run.h5 -r reqs.xlsx                   # results in run.check/
baslt check runs/ -r reqs.xlsx --params runs.csv  # every run in the folder, in parallel, with a dashboard
```

```
Run      q_spike.h5   hdf5, 120.0 s, 8 signals
Checks   reqs.xlsx:Requirements   16 requirements, 1 not covered

VERDICT  ID      TITLE                        WORST      LIMIT                          MARGIN               AT
FAIL     LV-004  q against the Mach envelope  72.66 kPa  <= curve(q_max_vs_mach, mach)  -3.146 kPa (-4.5 %)  T+60.953 s  ascent, load_relief, high_q; 56.0 s after pitch_start
FAIL     LV-001  Max dynamic pressure         72.66 kPa  <= 70 kPa                      -2.66 kPa (-3.8 %)   T+60.953 s  ascent, load_relief, high_q; 56.0 s after pitch_start
WARN     LV-010  Skin temperature             355.9 K    <= 420 K                       64.11 K (15.3 %)     T+98.936 s  data gap of 0.351 s inside the checked window

Result: FAIL (13 pass, 1 warn, 2 fail, 0 error, 0 n/a)
Output   run.check/  report.html, results.xlsx, reqs.checked.xlsx, results.json, results.csv
```

```python
import baslt

result = baslt.check("run.h5", "reqs.xlsx")                      # or a list, a folder or a glob of runs
print(result.status, result.exit_code, result.outputs["report"])
lint = baslt.lint_requirements("reqs.xlsx", source="run.h5")
```

The template's example rows check the simulated ascent of `examples/rocket_sim.py`. With `--source`, the Signals
sheet lists your run's signals instead and the example rows get Status `example`, so nothing is checked until you
point a row's Check at your signals and set its Status to `approved`; lint says when no row is selected. The same
examples are in `examples/rocket_requirements.csv` with `examples/rocket_mapping.yaml` (and `.json`), and
`examples/rocket_batch.py` writes a batch with a parameter table.

## The requirements table

One requirement per row, with a header row. The header can be anywhere in the first 20 rows; it is the first row
whose cells name at least two known fields. Names are matched ignoring case, spaces and punctuation, so `Warn
margin`, `WARN_MARGIN` and `warn-margin` are the same.

| Field | Header names recognized | Meaning |
|---|---|---|
| `id` | ID, Req ID, Requirement ID, Identifier, Key, Number, ReqIF.ForeignID | Requirement id; rows sharing an id are cases of one requirement |
| `title` | Title, Name, Summary, Description, Text, ReqIF.Name | Shown in reports |
| `check` | Check, Signal, Expression, Measure, Quantity | What is checked: a signal or an expression (below) |
| `kind` | Type, Kind, Check type, Limit type | `upper`, `lower`, `range`, `assert`, `duration`, `value`, `event`, or an aggregate `max`, `min`, `initial`, `final`, `mean`, `rms`, `integral`; empty means inferred |
| `limit` | Limit, Threshold, Criterion, Value | The limit cell (grammar below) |
| `lower`, `upper` | Min, Max, Lower limit, Upper limit, LSL, USL | Limits in two columns instead of one |
| `unit` | Unit, Units, UoM | Unit of the limit when the cell has none |
| `when` | When, Condition, Phase, During | Where in the run the requirement applies |
| `applies_to` | Applies to, Applicability, Configuration, Variant | Which runs it applies to, from run parameters |
| `tolerance` | Tolerance, Persistence, Allowed duration, Debounce | How long a violation may last: `100 ms` or `3 samples` |
| `margin` | Margin, Warn margin, Warning margin | Distance to the limit that gives WARN: `2 kPa` or `5 %` |
| `case` | Case, Subcase | Name of this row's case |
| `count` | Count, Expected count, Occurrences | How often an event must happen (default 1) |
| `severity` | Severity, Criticality, Priority, Category | Carried to the results |
| `grid` | Grid, Time base | The clock to evaluate on (below) |
| `notes` | Notes, Comment, Remarks, Rationale | Carried to the results |

Other names are set in the mapping (`requirements.columns`, below). Words in the Type column are read generously:
`max`, `upper limit`, `not to exceed` and `≤` mean an upper limit; `always`, `invariant` and `shall hold` mean
`assert`; `time above` means `duration`; `peak` means `max`; `occurs` means `event`. `requirements.vocab` adds more.

Rows without an id are skipped (headings). A row with an id and no check is **not covered**: it appears in the
results and the checked copy as `NOT COVERED`, so prose requirements are counted, not lost. `requirements.where`
keeps only rows whose cells hold allowed values, for example `Status: [approved]` or `Type: [Requirement]` in a
Polarion export. A column used in `where` is not also read as a field (a Polarion `Type` is not the check type)
unless `requirements.columns` names it.

### Kinds

With an empty Type the kind is inferred from the check: a signal or signal expression with a limit is a limit check
(`bound`), a condition without a limit is `assert`, a condition with a limit is `duration`, a single number is
`value`, and the name of an event is `event`.

| Kind | Check | FAIL when |
|---|---|---|
| `upper`, `lower`, `range` (`bound`) | a signal or expression | a violation lasts longer than the tolerance |
| `assert` | a condition such as `mode == 'COAST'` | the condition is false longer than the tolerance while the requirement applies |
| `duration` | a condition such as `q > 60 kPa` | the total time it holds is outside the limit |
| `value` | one number: an aggregate, an event function, a parameter or arithmetic on those | the number is outside the limit |
| `max`, `min`, `initial`, `final`, `mean`, `rms`, `integral` | a signal | the aggregate over the window is outside the limit (a `value` check of `max(signal)` and so on) |
| `event` | an event name, `MECO`, `MECO#last` | it does not happen exactly `count` times, or its time is outside the limit |

### Limit cells

```
<= 70 kPa     < 70 kPa     max 70 kPa     at most 70 kPa     not exceed 70 kPa     upper limit
>= 100 km     > 0 s        min 2 kN       at least 2 kN                            lower limit
[-7, 7] deg   (0, 1)       [0, 1)         99 .. 102 s    99 to 102 s              range; brackets set inclusiveness
between 2 and 3 kN         5.5 ± 1 s      ±5 deg         5.5 +/- 1 s
== 1          != 0         == heavy                                                equal, not equal (numbers or texts)
70 kPa                                                                             bare: direction from the Type column
curve(q_max_vs_mach, mach)     table(q_max_vs_mach)                                a limit curve over another signal
= 0.9 * param.q_design         <= 1.1 * mean(thrust)                               an expression
```

A unit after a range applies to both ends. The unit in the cell wins over the Unit column, which wins over the unit
of the checked expression. A bare number is in the unit of what is checked. A limit is converted into the checked
unit, so `<= 5 g0` works on an acceleration in m/s². `decimal_comma: true` reads `1,5` as 1.5.

A limit on a signal (`bound`) may change over time (`<= curve(...)`, `<= 0.9 * other_signal`); single-number checks
need a single-number limit.

### Cases

Rows with the same id are cases of one requirement: the check and kind come from the first row, and each row has its
own `applies_to`, `when`, limit, tolerance, margin, count and severity.

- `applies_to` looks at run parameters: `payload == 'heavy'`, `mass > 5 t`. A case whose `applies_to` is false is
  `N/A` for that run.
- For limit and assert checks, each sample belongs to the first case (in row order) whose conditions hold there. A
  violation is a run of consecutive samples beyond their own case's limit; it continues across a switch between
  cases, takes the strictest tolerance it touches, and belongs to the case of its worst sample.
- Single-number checks evaluate each case in its own window.
- The requirement's verdict is its worst case: FAIL, then ERROR, WARN, PASS, N/A. The headline value is the failing
  case with the least margin, or the passing one with the least margin.
- Cases that hold at the same time are allowed (the first wins); `defaults.on_case_overlap: error` makes that an
  ERROR.

```
ID      Check  Type   Limit  Unit  When         Tolerance  Case
LV-003  load   upper  5      g0    load_relief  100 ms     load relief
LV-003                6      g0    ascent       100 ms     powered
```

## Expressions

Checks, conditions, limits and derived signals share one small language. It is Python-like, is parsed with a fixed
list of allowed forms, and never runs code.

### Names

A bare name is, in this order:

1. `t` (the run's time, in seconds) or a function name;
2. an alias from the mapping (`signals:`), or a named condition (`conditions:`);
3. a signal of the run: its full name (`aero/q`), or its last part when that is unique in the run (`q`).

Backticks refer to a signal by its path, whatever its name: `` `aero/q` ``, `` `nav/position`.z ``. A vector stored as
one CSV column per component (`nav/position_x`, `_y`, `_z`, or `_1`, `_2`, ...) is put back together under its base
name. Run parameters are `param.name`; inside `Applies to`, bare names are parameters. Unknown names get a "did you
mean" suggestion. Aliases and conditions may refer to each other but not in a loop.

### Operators

`+ - * / **`, comparisons `< <= > >= == !=` (chains such as `2 < x < 5` work), `and`, `or`, `not` (also `AND`, `OR`,
`NOT`), `x in ('A', 'B')`, `a if cond else b`, component access `.x`, `.y`, `.z` or `[k]`. `≤ ≥ ≠ − ^` and a
lone `=` are read as `<= >= != - ** ==`. Quantities are written `60 kPa`, `60[kPa]` or `100 ms`.

States are compared by label: `mode == 'COAST'` or `mode == COAST` (a bare word next to a state is taken as a label,
and lint says so). Labels come from the file (text signals) or the mapping (`labels: {0: IDLE, 1: BURN}` for integer
modes).

### Functions

| Function | Result |
|---|---|
| `abs(x)`, `sign(x)`, `sqrt(x)`, `exp(x)`, `log(x)` | element by element; `sqrt`, `exp`, `log` need a unitless value |
| `sin(x)`, `cos(x)`, `tan(x)` | of an angle (degrees are converted) |
| `min(a, b, ...)`, `max(a, b, ...)`, `clip(x, lo, hi)`, `hypot(a, b)`, `where(cond, a, b)` | element by element |
| `norm(v)` | length of a vector signal |
| `deriv(x)` | time derivative (unit `x/s`: Pa/s, deg/s, m/s from m) |
| `movmean(x, 200 ms)` | moving average over a time window |
| `as_unit(x, 'kN')` | declares the unit of an expression whose unit cannot be worked out |
| `curve(name, x)` | a limit curve from the mapping at `x` (the curve's own argument when `x` is left out) |
| `max(x)`, `min(x)`, `initial(x)`, `final(x)`, `mean(x)`, `rms(x)`, `integral(x)` | one number over the requirement's window (time-weighted for `mean`, `rms`, `integral`) |
| `duration(cond)` | total time a condition holds in the window |
| `after('E')`, `after('E', 2 s)`, `before('E', 1 s)`, `during('E', 5 s)`, `between('A', 'B')` | conditions from events |
| `since('E')` | seconds since the event (a signal) |
| `time('E')`, `count('E')`, `at('E', x)` | an event's time, how often it happened, and a signal's value at that time |

An event reference is `'E'` (its first trigger), `'E#last'`, `'E#2'`, or `'E#all'` (every trigger, for `after`,
`during` and `between`).

### Units

Units follow the signals: the unit of an alias or signal, kept by `+`, `-`, `abs`, `min`, `max`, `movmean` and
friends. Adding or comparing values of the same dimension converts the right side into the left side's unit (`q >
qk`, `q > 5 kPa`); different dimensions are an error ("a pressure (Pa) and a time (s) cannot be compared").
Multiplying or dividing by a plain number keeps the unit; dividing two values of the same dimension gives a unitless
number; other products give an unknown unit, which must be declared with `as_unit()` or an alias `unit:` before a
limit with a unit can be compared with it.

The unit table is the compiler's (see `policy.md`) plus `g0` (standard gravity), `N*m`, `kg/s`, `W`, `J`, `V`, `A`,
`rpm`, `1/s`, `Pa/s`, `rad/s^2`, `deg/s^2`, `W/m^2` and their prefixed forms. `units:` in the mapping adds more:

```yaml
units:
  gee: 9.80665 m/s^2                       # a new unit defined by a quantity
  counts: {dimension: dimensionless, factor: 1}
```

### Reading run files

The readers cope with what recording tools write: a units row under a CSV header (`s`, `m/s`), numbers with a
decimal comma in files the comma does not split (`0,5` with `;` separators), comment lines before the header,
columns for the elements of a vector (`pos_x`, `pos[0]`, `pos_1`, read as one vector under `pos`), one clock per
group or per topic, and timestamps in other units (`time.unit: us`). A column of whole numbers is read as a state
signal (held between samples, compared by label); give it `kind: continuous` in `signals:` when it is really a
measurement. `baslt inspect RUN` shows what a reader made of a file, and says what it had to skip.

Nothing has to be guessed. `source:` settles how the run files are read, and what it says wins:

```yaml
source:
  format: csv            # read the files as this format whatever they are called
  delimiter: ";"         # CSV: one character, or "whitespace"
  encoding: cp1252       # CSV
  units_row: true        # true, false, or auto (the default: a row of units is recognized)
  decimal_comma: true    # true, false, or auto (the default: 0,5 is a number where the comma does not split)
```

## Where and when

A requirement is evaluated on the clock of the first signal its check reads, then its condition's and cases'.
Other signals are resampled onto that clock: linearly for continuous and vector signals, holding the last value for
discrete ones, and NaN outside a signal's time span or across its gaps. `grid: union` evaluates on every sample of
every signal involved; `grid: <alias>` uses that signal's clock.

A condition that cannot be decided (a NaN input) counts as not holding, and the time it is unknown inside the window
is reported. Condition edges are resolved to the spacing of the clock.

Violation times are exact where the values allow: a continuous run starts and ends where the excess crosses zero
(linearly between samples); a discrete run lasts from its first sample to the sample after its last. Where a run
touches the edge of the data, a gap or the edge of the window, it ends at the last sample there and says so
(`from the start of the data`, `ends at a gap or window edge`).

`duration` of a comparison such as `q > 60 kPa` uses the same exact crossings; `duration` of any other condition
counts whole sample intervals.

## Verdicts and evidence

| Verdict | Meaning |
|---|---|
| PASS | Checked and within the limit |
| WARN | Within the limit but inside the warning margin, or checked with a data gap, an unknown condition or a missing event (see the `on_*` settings) |
| FAIL | Outside the limit |
| N/A | Nothing to check in this run: no case applies, the window never opens, or there is no data in it |
| ERROR | The requirement could not be checked: an unknown name, a unit that does not fit, a limit that does not parse. The message says where (`reqs.xlsx:Requirements!E12`) |

One requirement's ERROR never stops the others. A run that cannot be read at all is reported as an ERROR run.

**Margin** is the distance to the limit at the worst sample: positive inside the limit, negative outside, in the
checked unit and as a percentage of the limit. For a limit that changes over time (a curve), the worst sample is the
closest approach, not the highest value. For a range, it is the nearer side.

**Warning margin**: a check whose margin is smaller than the warning margin is WARN (per case; `5 %` is relative to
the limit). There is no warning margin unless the row or `defaults.margin` gives one.

**Tolerance**: a violation that lasts no longer than the tolerance is tolerated. Tolerated violations PASS, are
listed in the evidence and drawn hatched in the report; `defaults.on_tolerated: warn` makes them WARN. Without a
tolerance every violating sample counts. `3 samples` tolerates runs of up to three samples.

**Gaps**: missing data (NaN) or an undecided condition inside the checked window gives WARN; `defaults.on_gap:
ignore` makes it a note and `fail` a FAIL.

**Missing events**: when a condition or value refers to an event that did not happen, `after()` and `between()` never
hold, `before()` always holds, and `time()`, `since()` and `at()` have no value. The requirement is then WARN
(`defaults.on_missing_event: fail` or `na` change that), which never hides a FAIL. An `event` check whose event is
missing FAILs. An event still waiting for its debounce when the run ends gives WARN.

Evidence (in `results.json`, the workbook and the report) holds the worst value, its time, the limit, the margin,
every violation (start, end, duration, samples, worst value, limit, tolerated, case, component; the first 256), the
checked time, gap time, unknown time and overlap time, and where the worst point was: the named conditions holding
there and the last event before it ("ascent, high_q; 1.1 s after maxq").

Times are shown relative to `time.t0` (an event such as liftoff) when the mapping names one: `T+60.953 s`. Limits
and event times in the table are always in the run's own time.

## The mapping

Everything about the layout and the run is set in a mapping: a YAML or JSON file given with `-m`, or sheets inside
the requirements workbook itself. Both are optional. Settings merge in the order built-in defaults, workbook sheets,
mapping file. Unknown keys are errors with a suggestion.

```yaml
version: 1
name: LV-3 ascent requirements

requirements:
  sheet: Requirements            # name or number; default: a sheet named Requirements, else the first other one
  header_row: auto               # or a row number
  where: {Status: [approved], Type: [Requirement]}
  columns: {id: "Polarion ID", check: "Verification Check|Check"}
  passthrough: [Status, Owner]   # columns carried to the results
  vocab: {kind: {Apogee: max}}   # more words for the Type column
  decimal_comma: false
  write_back: {verdict: "Verification Result", evidence: "Verification Evidence"}
  encoding: utf-8                # CSV only
  delimiter: ","                 # CSV only; sniffed when left out

checks:                          # an engineer-owned sheet of checks, joined by id to a requirements export
  file: checks.xlsx
  sheet: Checks
  key: ID

source:                          # how to read the run files (below); everything here is optional
  delimiter: ";"

time:
  signal: t                      # the run's clock when signals do not name their own
  unit: s                        # unit of the run's time values
  on_non_monotonic: error        # error | sort | drop
  t0: liftoff                    # show times as T+ from this event

signals:
  q: {path: aero/q, unit: Pa}
  mode: {path: gnc/mode, kind: discrete, labels: {0: VERTICAL_RISE, 1: PITCH_PROGRAM, 2: LOAD_RELIEF, 3: COAST}}
  alt: {expr: "`nav/position`.z", unit: m}
  load: {expr: "norm(deriv(movmean(`nav/velocity`, 200 ms)))", unit: m/s^2}
  q_ref: {path: ref/q, time: ref/time}   # a signal with its own clock

events:
  liftoff: {when: "alt > 1 m", debounce: 100 ms}
  pitch_start: {signal: mode, equals: PITCH_PROGRAM}
  MECO: {signal: thrust, falls_below: 1 MN, debounce: 250 ms}
  separation: {param: sep_time}            # a time given as a run parameter
  # also: rises_above, hysteresis, occurrence (first, last, or a trigger number)

conditions:
  ascent: "between('liftoff', 'MECO')"
  high_q: "q > 20 kPa"

curves:
  q_max_vs_mach:
    x: mach                                # the argument when curve() is called with only the name
    x_unit: "1"
    y_unit: kPa
    points: [[0.0, 72.0], [1.2, 70.0], [3.0, 69.0]]   # x strictly increasing
    mode: linear                           # linear | step
    outside: clamp                         # clamp | none (no limit outside) | error

units:
  gee: 9.80665 m/s^2

params:
  file: runs.csv                           # the parameter table (also --params)
  key: run                                 # its key column; default: the first of run, id, name, file, case, path
  units: {mass: kg}
  from_source: [mass, "config/*"]          # parameters stored in the run files

defaults:
  tolerance: 0 s
  margin: 5 %
  grid: first                              # first | union
  on_gap: warn                             # warn | ignore | fail
  on_missing_event: warn                   # warn | fail | na
  on_case_overlap: first                   # first | error
  on_tolerated: pass                       # pass | warn

report:
  title: LV-3 ascent
  pages: failed                            # batch pages: failed | all | none
  plot_bins: 512                           # points kept per plotted signal (twice this)

archive:
  max_size: 2 MiB                          # budget of --archive artifacts
```

### Sheets instead of a mapping file

A workbook can carry its own mapping in sheets named `Signals`, `Events`, `Conditions`, `Curves`, `Units` and
`Settings` (any order, any of them). `baslt requirements init` writes all of them with examples and a `Guide`
sheet.

| Sheet | Columns |
|---|---|
| Signals | Alias, Path, Unit, Kind, Labels (`0=IDLE; 1=BURN`), Expression, Time |
| Events | Name, Signal, Condition (`falls below`, `rises above`, `equals`), Value, When, Parameter, Hysteresis, Debounce, Occurrence |
| Conditions | Name, Expression |
| Curves | Name, X, Y (one row per point), X unit, Y unit, Argument, Mode, Outside |
| Units | Symbol, Definition (`9.80665 m/s^2`), or Dimension, Factor, Offset |
| Settings | Key (`time.t0`, `defaults.on_gap`, `requirements.where.Status`), Value |

### Polarion and other tools

- **Excel round trip.** Export the work items to Excel, check against the export, and import the checked copy back.
  `requirements.write_back` names the existing Polarion fields that receive the verdict, value, limit, margin, time,
  evidence and violation count (result fields `verdict`, `value`, `limit`, `margin`, `margin_pct`, `at`,
  `evidence`, `runs`); fields the sheet does not have are added after the last column. Only the result cells change:
  every other part of the workbook is copied byte for byte, so the import sees the same file it exported.
- **ReqIF.** `.reqif` and `.reqifz` files read as tables: one row per requirement in document order, one column per
  attribute (`ReqIF.ForeignID`, `ReqIF.Name`, `ReqIF.Text`, custom fields), plus `ReqIF Type` (Heading,
  Requirement, ...) and `ReqIF Level`. XHTML text is flattened and enumerations show their names. Filter headings
  with `where: {ReqIF Type: [Requirement]}`. The checked copy of a ReqIF file is a CSV table.
- **Checks kept apart.** When the export should stay untouched, keep the checks in a separate sheet (`checks:`) with
  an ID column and the check columns; it is joined by id. Requirements without a check are reported as not covered,
  and checks without a requirement are listed.

## Run parameters

Parameters describe a run: its payload, its configuration, a dispersion's seed. They select cases (`Applies to`),
feed values (`param.mass`) and events (`param:`), and appear in the batch dashboard's filters and scatter plots.

- **A table** (`--params runs.csv`, or `params.file`) with one row per run, matched by run id, file name, file stem or
  path below the batch folder. Numbers are read as numbers; `params.units` gives them units.
- **The run file** (`params.from_source`): HDF5 scalar datasets and file attributes, MAT-file scalars and character
  vectors (struct fields as `cfg/gain`), and single values next to in-memory arrays. They are named by their last
  part (`mass`), or their whole path with `_` when last parts repeat. The table wins when both give a value.

`requirements lint --source run.h5 --params runs.csv` checks that every `param.` name exists.

## Outputs

### One run

`OUT/` (default: `<run>.check/` next to the run file):

| File | Contents |
|---|---|
| `report.html` | The report: what went wrong, a timeline of phases and events, every requirement with its plot, the events and the issues |
| `results.xlsx` | Sheets Summary, Results, Cases, Violations, Events, Issues |
| `<reqs>.checked.xlsx` (or `.csv`) | The requirements table with Verdict, Result, Margin and Evidence columns (or the `write_back` fields) filled in |
| `results.json` | Everything, for scripts |
| `results.csv` | One row per requirement; text a spreadsheet would read as a formula starts with `'` |
| `run.baslt` | With `--archive` |

The report is one file that opens from disk and loads nothing from the network. Its plots are drawn from the
checked values: the full signal (every sample up to 1024, else the lowest and highest sample of 512 bins, so
spikes survive) and full resolution around failures. Drag across a plot to zoom, double-click to zoom out; the time
axes are linked. `report.html?t=60.7,61.2` opens zoomed in. Clicking a row of "What went wrong" zooms to it.

### Many runs

Give a folder, a glob or several files. Runs are checked in `--jobs` processes (default: one less than the CPU
count), each with one thread and a lower priority. `OUT/` (default: `<folder>.check/`):

| File | Contents |
|---|---|
| `index.html` | The dashboard: pass rate per requirement, the verdict matrix (runs by requirements), each requirement's signal over all runs (range, 5-95 % band, median, and the failing and closest runs), margin against a run parameter, and a filterable run list (`payload>3000 status=fail`) |
| `results.xlsx` | Sheets Summary, Matrix, Margins, Requirements, Violations, Runs |
| `<reqs>.checked.xlsx` | One sentence per requirement: "FAIL in 3/1000 runs, worst -1.2 kPa (run_0457)" |
| `summary.json` | Counts, per-requirement statistics and every run's status |
| `index.jsonl` | One line per run, appended as runs finish |
| `runs/<id>.json` | Each run's full results |
| `runs/<id>.html` | The report of each run with a problem (`--pages all` for every run, `none` for none) |
| `runs/<id>.baslt` | With `--archive` |

`--resume` skips runs whose stored result was made from the same run file (path, size, modification time), the same
requirements table, mapping and parameters, and the same baslt version. A run that crashes its process is tried
again alone and, if it crashes again, reported as ERROR.

## Archive

`--archive` also compiles each checked run with a policy made from its requirements (see `policy.md`):

| Requirement | Protected in the artifact |
|---|---|
| a limit on a signal | global extrema, and a violation for each constant limit, with the tolerance as its minimum duration |
| `duration` of `signal > value` | threshold crossings at the value |
| `max()`, `min()` of a signal | global extrema |
| any state signal read | state transitions |
| mapping events on one signal | the same events, with a second of data on each side |

Checks on derived signals keep their input signals without extra guarantees. The artifact verifies against the run
(`baslt verify run.baslt --source run.h5`). A budget too small for what the requirements protect is reported as an
issue of the run; the check itself stands.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Every requirement passed or warned (with `--fail-on warn`, only passed), or `--fail-on none` |
| 1 | A requirement failed in some run (or warned, with `--fail-on warn`) |
| 2 | An internal error |
| 3 | The requirements, mapping or options could not be used, the table selects no requirement to check, or nothing failed but a requirement or run ended in ERROR |

## Limits of this version

- The report and dashboard need a browser that can unpack raw DEFLATE data with `DecompressionStream` (Chrome 103,
  Firefox 113, Safari 16.4 or later). The tables of the report are complete without scripts.
- Written copies were checked with the reader in this package and with openpyxl; opening them in Excel and a Polarion
  round-trip import should be tried once with real files.
- Column names cover common Polarion and DOORS exports; others are set with `requirements.columns`.
