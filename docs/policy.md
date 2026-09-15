# Policy file format

A policy says which facts about a run must survive (hard requirements), how to spend any remaining bytes (soft
preferences), and how big the artifact may be. Policies are YAML (needs `baslt[yaml]`), JSON, or a Python `dict`.
The same policy written in any of these forms has the same canonical hash.

## Complete example

```yaml
version: 1
name: flight_review

artifact:
  max_size: 2 MiB            # hard ceiling; hard requirements are never dropped to meet it
  codec: deflate             # deflate | store | zstd
  hash: full                 # full | sampled | none   (default: full for files, arrays for in-memory)
  soft_value_dtype: source   # source | float32 (float32 only applies to signals with no hard contract)
  on_not_applicable: warn    # warn | fail   (a hard requirement on an empty / all-NaN / constant signal)

signals:
  time: t                    # default time signal for signals without their own
  time_unit: s               # unit of time arrays in the source (converted to seconds)
  on_non_monotonic: error    # error | sort
  include: ["*"]             # glob patterns over canonical signal names
  exclude: []
  decl:                      # optional per-signal declarations; keys become aliases
    q_dyn:     {path: aero/q, unit: Pa}
    aoa:       {path: aero/alpha, unit: deg}
    thrust:    {path: prop/thrust, unit: N}
    position:  {path: nav/position, unit: m}
    flight_mode: {path: gnc/mode, kind: discrete}

hard:
  q_dyn:
    global_extrema: {}
    threshold_crossing:
      - {value: 65 kPa, edge: both, hysteresis: 1 kPa, debounce: 100 ms, tolerance: 5 ms}
    violation:
      - {above: 70 kPa, min_duration: 100 ms}
  aoa:
    local_extrema: {prominence: 0.5 deg, separation: 100 ms}
    window_extrema: {interval: 1 s}
    threshold_crossing:
      - {value: -7 deg}
      - {value: 7 deg}
  flight_mode:
    state_transitions: {}

events:
  MECO:
    when: {signal: thrust, falls_below: 100 N, debounce: 250 ms}
    occurrence: first          # first | last | all
    expect: 1                  # optional; a mismatch is a warning
    keep: {before: 5 s, after: 10 s, signals: [thrust, q_dyn, aoa, flight_mode]}
    severity: info             # info | limit

trajectories:
  ascent:
    position: position         # one (n,3) signal, or a list of three scalar signals [x, y, z]
    max_position_error: 1 m
    max_time_error: 20 ms      # optional
    linked: [aoa]              # signals that get a sample at every trajectory knot

sync_groups:
  flight_dynamics:
    members: [q_dyn, aoa]

soft:
  - {match: q_dyn, priority: high}
  - {match: "*", priority: low, max_points: 4000}

review:
  thumbnails: [q_dyn, aoa]     # signals drawn in sweep overlays (default: hard-requirement signals, at most 4)
  kpis: [q_dyn]                # signals whose extrema appear as sweep table columns
```

Only `version` is required. Every other section is optional.

## Top-level keys

| Key | Type | Meaning |
|---|---|---|
| `version` | int | Must be `1`. |
| `name` | string | Label shown in reports. Default `"policy"`. |
| `artifact` | mapping | Size budget and encoding options. |
| `signals` | mapping | Signal selection, declarations and time handling. |
| `hard` | mapping | `signal → op → spec or list of specs`. |
| `events` | mapping | `name → event spec`. |
| `trajectories` | mapping | `name → trajectory spec`. |
| `sync_groups` | mapping | `name → {members}`. |
| `soft` | list | Ordered soft allocation rules; the first matching rule wins per signal. |
| `review` | mapping | Sweep dashboard options. |

Unknown keys anywhere are errors, with a "did you mean" suggestion.

## Signal names

A signal is referred to by:

1. a key of `signals.decl` (an alias),
2. its canonical name: the source path without a leading `/` (`aero/q`, `sim/q_dyn`), or for CSV the column
   header without its unit, or for in-memory input the dict key,
3. a leaf name (the part after the last `/`) when exactly one signal has that leaf.

An ambiguous leaf is an error that lists the candidates.

## Hard operators and parameters

Parameter kinds:

- **value**: an absolute level in the signal's unit (`65 kPa`, `7 deg`, `300 K`).
- **delta**: a difference in the signal's unit; affine offsets are ignored (`1 kPa`, `2 degC` means 2 K).
- **duration**: a time span, stored in seconds (`100 ms`).
- **enum**: one of the listed strings.

| Op | Parameter | Kind | Required | Default |
|---|---|---|---|---|
| `global_extrema` | — | | | |
| `local_extrema` | `prominence` | delta | yes | |
| | `separation` | duration | no | `0 s` |
| | `kind` | enum `max`, `min`, `both` | no | `both` |
| `window_extrema` | `interval` | duration | yes | |
| | `origin` | duration | no | `0 s` |
| `threshold_crossing` | `value` | value | yes | |
| | `edge` | enum `rising`, `falling`, `both` | no | `both` |
| | `hysteresis` | delta | no | `0` |
| | `debounce` | duration | no | `0 s` |
| | `tolerance` | duration | no | `0 s` |
| | `interpolate` | enum `linear`, `none` | no | `linear` |
| `violation` | `above` | value | one of `above`/`below` | |
| | `below` | value | one of `above`/`below` | |
| | `min_duration` | duration | no | `0 s` |
| `state_transitions` | — | | | |

Every hard spec also accepts `severity: info | limit`. The default is `limit` for `violation` and `info` otherwise.
A spec can be a single mapping or a list of mappings. Requirement ids are the YAML path:
`hard.q_dyn.global_extrema`, `hard.q_dyn.threshold_crossing[0]`.

Kind restrictions: `state_transitions` and `equals` events need a discrete signal; `local_extrema`,
`window_extrema`, `threshold_crossing` and `violation` need a continuous or vector signal (vectors are evaluated per
component).

## Events

```yaml
events:
  <name>:
    when: {signal: S, falls_below: V}      # or rises_above: V, or equals: V (discrete signals)
      # optional inside when: hysteresis (delta), debounce (duration)
    occurrence: first | last | all          # default first
    expect: N                               # optional
    keep: {before: D, after: D, signals: [..]}   # signals default to every included signal
    severity: info | limit                  # default info
```

## Trajectories

`position` is one signal with shape `(n, 3)` or a list of exactly three scalar signals sharing one clock.
`max_position_error` is a length. `max_time_error` is optional. `linked` signals receive a sample at every retained
trajectory knot time.

## Soft rules

`match` is a glob over canonical names and aliases. `priority` is `high` (weight 4), `medium` (2), `low` (1) or
`none` (0, keep only hard samples). `max_points` caps soft points for matching signals. Signals no rule matches get
`medium`.

## Units

Quantities are written `<number> <unit>` or `<number><unit>` (`65 kPa`, `100ms`). A bare number is taken to be in
the signal's own unit. A signal whose unit is not in this table is opaque: its policy values must be bare numbers or
use exactly the same unit string.

SI value = number × factor + offset (absolute), or number × factor (delta).

| Dimension | Unit symbols (factor to SI base) |
|---|---|
| time (s) | `s` 1 · `ms` 1e-3 · `us` 1e-6 · `µs` 1e-6 · `ns` 1e-9 · `min` 60 · `h` 3600 |
| angle (rad) | `rad` 1 · `mrad` 1e-3 · `deg` π/180 · `°` π/180 · `degree` π/180 · `degrees` π/180 |
| angular rate (rad/s) | `rad/s` 1 · `deg/s` π/180 |
| pressure (Pa) | `Pa` 1 · `hPa` 100 · `kPa` 1e3 · `MPa` 1e6 · `bar` 1e5 · `mbar` 100 · `psi` 6894.757293168361 |
| length (m) | `m` 1 · `mm` 1e-3 · `cm` 1e-2 · `km` 1e3 · `ft` 0.3048 · `in` 0.0254 · `nmi` 1852 |
| velocity (m/s) | `m/s` 1 · `km/s` 1e3 · `km/h` 1/3.6 · `ft/s` 0.3048 · `kn` 1852/3600 |
| acceleration (m/s^2) | `m/s^2` 1 · `m/s2` 1 · `ft/s^2` 0.3048 |
| force (N) | `N` 1 · `kN` 1e3 · `MN` 1e6 · `lbf` 4.4482216152605 |
| mass (kg) | `g` 1e-3 · `kg` 1 · `t` 1e3 |
| temperature (K) | `K` 1 · `degC` 1, offset 273.15 · `degF` 5/9, offset 255.37222222222223 |
| frequency (Hz) | `Hz` 1 · `kHz` 1e3 |
| dimensionless (1) | `1` 1 · `%` 0.01 |
| bytes (B) | `B` 1 · `KB` 1e3 · `MB` 1e6 · `GB` 1e9 · `KiB` 1024 · `MiB` 1048576 · `GiB` 1073741824 |

`artifact.max_size` accepts a bytes quantity or a plain integer number of bytes. The CLI `--max-size` overrides it
and the artifact records `budget.source: cli`.

## Errors

Every problem found in a policy is reported with its path and, for YAML, its line and column:

```
hard.q_dyn.threshold_crossing[0].edge: expected one of rising, falling, both, got 'up' (flight_review.yaml:23:9)
hard.q_dyn.threshold_crossing[0].value: '7 deg' is an angle but signal q_dyn is a pressure (Pa)
events.MECO.when.signal: unknown signal 'thrust_n'; did you mean 'thrust'?
```

Up to 50 issues are collected before reporting.

## Canonical form and hash

The canonical form is the validated policy as JSON with sorted keys, compact separators, and quantities kept as their
original strings. `policy.sha256` is the SHA-256 of that JSON encoded as UTF-8. Defaults are filled in before hashing,
so omitting a default and writing it explicitly give the same hash.
