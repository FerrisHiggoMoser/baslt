# Fidelity contracts

This document is the single definition of what every hard requirement guarantees. The compiler (planner) and the
verifier are both implemented from it, independently.

## Conventions

- A run is a set of signals. Each signal has timestamps `t[0..n-1]` in seconds (non-decreasing, finite) and values
  `x[0..n-1]`, scalar or vector `(n, k)`.
- A sample is **finite** when every component is finite.
- The artifact retains a subset of source sample indices `idx` (strictly increasing) with their exact timestamps and
  values (bit-identical) and a role bitmask per retained sample.
- **Reconstruction** of a continuous signal from retained samples is linear interpolation between consecutive
  retained samples; a discrete signal uses sample-and-hold (previous retained value). Reconstruction never interpolates
  across a non-finite retained sample: the interval between a finite and a non-finite retained sample is a gap.
- `ulp(x)` is the gap between `|x|` and the next larger float64.

## Implicit retention

- `extent`: the first and last source sample of every included signal.
- `gap`: for every signal with at least one hard contract, the first and last sample of each run of non-finite
  samples, and the finite samples immediately before and after that run.

## global_extrema

Guarantee: for each component, the retained samples include the source sample with the maximum finite value and the
one with the minimum finite value, lowest index on ties. Value error 0.

Evidence per component: `max {index, t, value}`, `min {index, t, value}`.
Roles: `hard.<sig>.global_extrema`.
Not applicable: no finite sample.

## window_extrema (interval Δ, origin o)

Bucket of sample i: `m(i) = floor((t[i] - o) / Δ + δ)` with `δ = 8 · ulp(max(|t[0]|, |t[n-1]|) / Δ)`. Planner and
verifier use exactly this formula. A sample whose `(t[i] - o)/Δ` lies within `δ` of an integer boundary also belongs
to the neighbouring bucket.

Guarantee: for every bucket containing a finite sample, per component, the finite max and min samples of the bucket
(lowest index on ties) are retained.

Evidence: bucket count, number of buckets with finite samples.

## local_extrema (prominence p, separation s, kind)

Peaks (maxima; minima are peaks of `-x`):

1. Equal-value runs are merged. A run is a candidate peak when both neighbouring runs are strictly lower. Its index
   is the middle sample of the run, rounded down. The first and last run are never peaks. Non-finite samples split
   the signal: they act like `+∞` walls.
2. Prominence (scipy definition): extend left from the peak until a strictly higher finite sample, a wall or the
   signal start; the left base is the minimum finite sample in that interval. Same to the right. Prominence = peak
   value − max(left base value, right base value). Base ties take the index nearest the peak.
3. Keep peaks with prominence ≥ p.
4. Separation (applied after prominence): visit kept peaks by value descending (lowest index first on ties); keep a
   peak if no already-kept peak has `|t_a − t_b| < s`.

Guarantee: every peak surviving steps 1–4 is retained, together with its left and right base samples. Only these
claimed peaks are guaranteed; the reconstruction may show other bumps.

Evidence: count of claimed maxima/minima, list of `{index, t, value, prominence}` (up to 256).
Roles: `#peak`, `#base`.

## threshold_crossing (V, edge, hysteresis H, debounce D, tolerance τ, interpolate)

1. **Level crossings** between consecutive finite samples i, i+1:
   rising if `x[i] < V ≤ x[i+1]`, falling if `x[i] ≥ V > x[i+1]`.
   Crossing time with `interpolate: linear`: `tc = t[i] + (t[i+1] − t[i]) · (V − x[i]) / (x[i+1] − x[i])`.
   With `interpolate: none`: `tc = t[i+1]`, and the evidence records `t[i+1] − t[i]` as discretization error.
2. **State.** With H = 0 the state of sample j is `x[j] ≥ V`. With H > 0: `hi = V + H/2`, `lo = V − H/2`; the
   state becomes high at a sample with `x ≥ hi`, low at a sample with `x ≤ lo` **and** `x < V`, and otherwise holds.
   The initial state is `x[first finite] ≥ V`. Non-finite samples hold the state.

   The extra `x < V` condition is what exact arithmetic gives: for every `H > 0` the band's lower edge lies strictly
   below the level, so a sample at exactly `V` is never low. It matters only when `H` is smaller than one unit in the
   last place of `V`, where `lo` rounds onto `V`. Without it a sample at exactly `V` could flip the state low while
   step 1 reports no falling level crossing anywhere, leaving step 3 with no crossing to timestamp.
3. **Flip.** When the state changes at sample j, the flip's crossing is the last level crossing in the same direction
   with `i + 1 ≤ j`. Its time is that crossing's `tc`. If the level crossing spans a non-finite gap, the bracketing
   finite samples are used and the flip is marked `gap`.
4. **Debounce.** Order flips by time. A run between consecutive flips lasts `tc[k+1] − tc[k]`; the last run ends at
   the last sample time. A state change is accepted only if its run lasts ≥ D. Accepted transitions are the changes in
   the accepted state sequence, each timestamped with the `tc` of the flip that started its accepted run. A final run
   shorter than D is reported as `pending_at_end`.
5. **Edge filter** keeps rising, falling or both.

Guarantee: for every accepted crossing, both bracketing samples `i` and `i+1` of its level crossing (and, with H > 0,
the confirming sample j) are retained. Running steps 1–5 on the reconstruction yields exactly the same accepted
crossings, with times equal within `τ + 4·ulp(max(|t[i]|, |t[i+1]|))`.

Evidence: count by edge, list of `{t, edge, index_before}` (up to 256), `pending_at_end`, count of `gap` flips.

## violation (above A | below B, min_duration m)

A violating run is a maximal run of consecutive finite samples with `x > A` (or `x < B`); non-finite samples end a run.
Start time: the crossing time of A between the sample before the run and the first sample of the run (linear
interpolation), or the first sample's time if the run starts at the signal start (`open_start`) or after a gap
(`gap_start`). End time: symmetric. Keep runs with `end − start ≥ m`.

Guarantee: for every kept run, both boundary pairs and the most extreme sample of the run are retained. Detection on
the reconstruction yields the same runs with start/end times within tolerance `4·ulp`.

Evidence: count, total violating duration, list of `{start, end, worst_t, worst_value}` (up to 256).
Roles: `#edge`, `#worst`. Default severity: `limit`.

## state_transitions

Guarantee: the first sample, every sample whose value differs from the previous sample (NaN equals NaN), and the last
sample are retained. Hold reconstruction equals the source value at every source timestamp.

Evidence: transition count, distinct values.

## events

Trigger:

- `falls_below V` / `rises_above V`: threshold crossing machine with fixed edge, optional hysteresis and debounce.
- `equals V` (discrete): the first sample of each run with `x == V`; a run starting at sample 0 is marked `at_start`.

`occurrence` selects `first`, `last` or `all` accepted triggers. `expect: N` produces a warning if the number of
triggers differs.

Window per listed signal S: `lo = first index with t ≥ τ − before`, `hi = last index with t ≤ τ + after`; retain
every source sample in `[max(lo − 1, 0), min(hi + 1, n − 1)]`. Overlapping windows merge. Windows past a signal's
extent are clipped and marked `clipped`.

Guarantee: the trigger's bracketing samples are retained, and every source sample of every window is retained.

Evidence: occurrences found, triggers kept `{t, at_start}`, windows `{signal, start, end, samples, clipped}`.
Roles: `event.<name>#trigger`, `event.<name>#window`.

## trajectories (ε, max_time_error Δt, linked)

All position components share one clock. SED of source sample i between retained samples a < i < b:

`SED(i) = ‖P[i] − (P[a] + w · (P[b] − P[a]))‖`, `w = (t[i] − t[a]) / (t[b] − t[a])`.

Equal timestamps `t[a] = t[b]` are a discontinuity: both samples are retained and the segment has no interior.

Guarantee: `SED(i) ≤ ε + s` for every source sample i, with `s = 16 · ulp(max |coordinate|)`. The planner enforces
`SED ≤ ε − s`. Because both paths are linear between source timestamps, the bound holds at all times.

`max_time_error` (optional) is defined as: for every source sample i, `min over |τ − t[i]| ≤ Δt of ‖recon(τ) − P[i]‖
≤ ε`. SED ≤ ε implies it; the manifest records `max_time_error_used = 0`.

`linked` signals retain, at every knot time, the sample with that exact timestamp or the bracketing pair.

Evidence: `max_sed`, `allowed`, `knots`, `max_time_error_allowed`, `max_time_error_used`, linked alignment counts.
Roles: `trajectory.<name>#knot`, `link.<name>`.

## sync groups

Propagating timestamps of a group: every retained timestamp of every member that carries a hard, event, knot or soft
role. For each propagating timestamp T and each member M: if M has a sample with timestamp exactly T (bitwise), it is
retained; otherwise the pair of samples bracketing T is retained and counted as `unaligned`. Samples added by
propagation do not propagate again. Soft timestamps are chosen once per group from the union of member picks.

Guarantee: every propagating timestamp is either present in every member or bracketed, and the unaligned count is
exact.

Evidence: propagating timestamps, aligned, unaligned. Status is WARN when unaligned > 0.
Role: `sync.<name>`.

## Statuses

Each requirement ends as:

| Status | Meaning |
|---|---|
| `pass` | Evaluated and the guarantee holds. |
| `warn` | Evaluated and holds, with a caveat (unaligned sync, `expect` mismatch, `pending_at_end`, gap flips). |
| `not_applicable` | Could not be evaluated (empty, all-NaN, or constant where the op needs variation). |
| `fail` | The guarantee does not hold. |

A hard requirement is `pass` only if it was evaluated. Overall status: `fail` if any fail; otherwise
`pass_with_warnings` if any warn or not_applicable (or `fail` when `artifact.on_not_applicable: fail`); otherwise
`pass`.

## Sweep triage

| Triage | Rule |
|---|---|
| `limit_violations` | Any kept `violation` run, or any triggered event with `severity: limit`. |
| `compiler_failure` | Exception, infeasible budget, or self-verify failure. |
| `warning` | Overall `pass_with_warnings`. |
| `nominal` | Everything else. |

## Verification bases

| Basis | Meaning |
|---|---|
| `artifact` | Provable from the artifact alone (for example adjacency of crossing brackets, contiguity of event windows, sync alignment, prominence lower bound). |
| `attested` | Depends on source samples that were not retained (for example "no larger value elsewhere"); checked for internal consistency only. |
| `source` | Recomputed from the source with `verify --source`. |
