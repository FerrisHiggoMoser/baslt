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
verifier use exactly this formula.

**Existence and membership are different.** The buckets of a signal are exactly the values `m(i)` taken by its
samples; no other bucket exists. Membership is wider: sample i belongs to bucket `m(i)`, and also to bucket
`m(i) − 1` when `|u − round(u)| ≤ δ` for `u = (t[i] − o)/Δ`, that is when the sample sits on a bucket boundary. A
bucket that exists may therefore contain a finite sample only by borrowing a boundary sample from its neighbour, and
it is still covered by the guarantee.

Guarantee: for every bucket containing a finite sample, per component, the finite max and min samples of the bucket
(lowest index on ties) are retained. A sample is finite only when all its components are finite.

Evidence: `buckets` (how many exist), `buckets_with_finite` (how many contain a finite sample under the membership
rule above), `interval`, `origin`. If the interval is so small relative to the timestamps that `δ ≥ 0.5`, or bucket
numbers would exceed 2^52, the requirement is `not_applicable` with the reason recorded.

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

Evidence: `maxima`, `minima` (exact counts), `peaks`: a list of `{index, t, value, prominence, kind, component}`
(up to 256, sorted by index), and the bound `prominence` and `separation`. A signal with no finite sample is
`not_applicable`; otherwise the requirement passes, with zero peaks if there are none.
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

`pending_at_end` and the `gap` flip count are **step-4 values**: they describe what the state machine saw, before the
step-5 edge filter removed anything. Implementations may additionally report the edge-filtered counterparts as
`pending_at_end_edge` and `gap_flips_accepted`, but the two step-4 fields are the ones compared between compiler and
verifier.

Retention covers every **candidate**, not only the reported ones: the bracketing pair of every step-3 flip (and its
confirming sample when H > 0) is retained, including flips that debounce or the edge filter later discards. Without
this, a rejected excursion can reappear in the reconstruction as a crossing the source never had. The same rule
applies to `violation`: the boundary pairs of every candidate run are retained, while the `worst` sample is retained
only for runs that survive `min_duration`.

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

Guarantee: the bracketing samples of **every candidate** trigger are retained (for crossing triggers, every flip
of step 3, as for threshold_crossing; for `equals`, every run start and the sample before it), so detecting the
event again on the retained samples finds exactly the source's triggers. Every source sample of every window around
a selected trigger is retained.

The window range of one trigger on one signal is always kept as computed, even when windows overlap, so the evidence
lists one window per (selected trigger, listed signal) pair; the retained samples are their union. A signal with no
samples gets no window.

Status: `warn` when `expect` is set and differs from the number found; `not_applicable` when the trigger signal
has fewer than two finite samples (crossing triggers) or none (`equals`); otherwise `pass`, including when the event
never happened. The trigger signal must be scalar, and `equals` needs a discrete signal; a text value must be one of
the signal's labels.

Evidence: `condition`, `value`, `occurrence`, `expect`, `found` (triggers before selection), `pending_at_end` (the
step-4 flag), `triggers` (selected, up to 256) as `{t, index_before, index_after, at_start, gap}`, `windows` (up to
256) as `{signal, trigger, start, end, samples, clipped}`, and the exact `windows_total`.
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

Propagating timestamps of a group: the distinct (bitwise) timestamps of every retained sample of every member that
carries a propagating role, taken before any group runs. Every role propagates except `extent`, `gap`, `sync.*`
and `link.*`: today that is hard requirement and event roles; trajectory knots and soft picks join them when those
layers exist. Soft timestamps are chosen once per group from the union of member picks.

For each propagating timestamp T and each member M (including the member T came from):

- T outside M's time span `[t[0], t[n−1]]`: nothing is retained, and the pair counts as `out_of_range`;
- M has a sample with timestamp exactly T (bitwise): the first such sample is retained (`aligned`);
- otherwise the two samples bracketing T are retained (`unaligned`).

Samples added by propagation carry only the group's role and do not propagate again, so one pass over all groups is
the whole step.

Guarantee: every propagating timestamp inside a member's span is either present in that member or bracketed by two
adjacent source samples, and the three counts are exact.

Evidence: `members`, `propagating`, `aligned`, `unaligned`, `out_of_range` (the last three count
(timestamp, member) pairs). Status is `warn` when `unaligned` > 0.
Role: `sync.<name>`.

## Detection on supersets

Crossings, violations and event triggers retain every candidate (every flip bracket and confirming sample, every
candidate run's boundary pairs, every candidate trigger). Detecting them again on any superset of the retained
samples therefore gives exactly the source's result. Samples added by events, sync groups and, later, the soft layer
never create or hide one, so the compiler needs no repair step for them; a property test enforces this.

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
