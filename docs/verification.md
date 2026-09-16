# Verification

`baslt verify` answers one question: can this small file be trusted in place of the source run for the facts its
policy protects? The verifier is a separate code path from the compiler. It is written from `contracts.md`, uses its
own unit table and its own simple reference algorithms, and may not import the compiler's modules.

## Bases

| Basis | What it proves |
|---|---|
| `artifact` | The claim follows from the artifact alone. |
| `attested` | The claim depends on source samples that were not retained; the artifact is checked for consistency only. |
| `source` | The claim was recomputed from the source (`--source`). |

## Check ids

| Id | Basis | What is checked |
|---|---|---|
| `structure.container` | artifact | ZIP profile, header versions, CRCs, member order. |
| `structure.descriptors` | artifact | Every array descriptor fits its member; sizes match dtype and shape. |
| `structure.invariants` | artifact | Per signal: `t` finite and non-decreasing, `idx` strictly increasing and `< n_source`, equal lengths, role bits within the legend, `extent` on the first and last source index. |
| `structure.size` | artifact | `manifest.artifact.size_bytes` equals the file size and does not exceed `max_bytes`. |
| `structure.policy` | artifact | The embedded policy hashes to its recorded sha256, which matches the manifest. |
| `structure.bind` | artifact | Re-binding the embedded policy with the verifier's own unit table reproduces every bound parameter in `index.json`, for requirements and events. |
| `<requirement id>.<aspect>` | varies | Per-contract checks below. |
| `events.<name>.<aspect>`, `sync_groups.<name>.alignment` | varies | Event and sync-group checks below. |
| `source.digest` | source | The source digest matches (a sampled digest is reported as a fingerprint, never as sha256). |
| `source.samples` | source | Every retained `(t, v)` is bit-identical to the source at its index. |
| `budget.accounting`, `budget.errors` | artifact, source | The byte and sample accounting, and the reconstruction errors (below). |

## Per-contract checks

| Contract | Aspect | Basis | Check |
|---|---|---|---|
| global_extrema | `max`, `min` | attested | Flagged samples exist, equal the manifest values bitwise, and no retained sample exceeds them. |
| window_extrema | `buckets` | attested | Each flagged bucket's flagged max/min equal the retained max/min of that bucket. |
| local_extrema | `prominence` | artifact | Each claimed peak and its bases are retained and the artifact prominence is at least `p`. |
| local_extrema | `separation` | artifact | Listed peaks of the same kind and component are at least `separation` apart. |
| local_extrema | `peaks` | source | The source's peaks are exactly the claimed ones, with the claimed counts, and all are retained. |
| threshold_crossing | `brackets` | artifact | Every claimed crossing has adjacent retained source indices that straddle `V`. |
| threshold_crossing | `fidelity` | artifact | The verifier's own detector on the reconstruction returns exactly the claimed crossings within tolerance. |
| violation | `runs` | artifact | Boundary pairs straddle the limit, durations meet `min_duration`, the worst sample lies inside. |
| violation | `fidelity` | artifact | Detection on the reconstruction returns exactly the claimed runs. |
| state_transitions | `transitions` | artifact | Hold transitions over retained samples equal the flagged transitions. |
| state_transitions | `source` | source | Every source change is retained and flagged, the counts match, and holding the last retained sample at or before each source index reproduces the source. |
| event | `trigger` | artifact | Detecting the event on the retained samples returns the claimed `found` count, `pending_at_end` and selected triggers (times within tolerance, bracket indices exact), and every bracket sample carries the trigger role. An `expect` mismatch must be marked `warn`. |
| event | `windows` | artifact | The window count is the selected triggers times the non-empty listed signals; each listed window is a contiguous run of retained source samples carrying the window role, bounded by samples outside the window (or the signal's ends), with the right `clipped` flag. |
| event | `source` | source | Triggers and window bounds recomputed from the source match the claims. |
| trajectory | `sed` | attested | Knots present, claimed `max_sed ≤ ε`, linked signals aligned or bracketed at every knot. |
| sync_group | `alignment` | artifact | Every propagating timestamp inside a member's span is present bitwise (with the group's role) or bracketed by two retained samples with consecutive source indices (both with the role); `propagating`, `aligned`, `unaligned` and `out_of_range` match, and the status is `warn` exactly when something is unaligned. |
| budget | `accounting` | artifact | Required + discretionary + overhead bytes equal the file size, and required + discretionary equal the data members; `budget.source` agrees with `max_bytes`; one row per signal in index order with `retained = n = hard + soft`, `bytes` equal to the members the signal pays for, a well-formed `soft_max_abs_err`, at most `hard` samples carrying a contract role (anything but `soft` and `sync.*`), at most `soft` samples carrying only `soft`, and no discretionary bytes without soft samples. |
| budget | `errors` | source | Each signal's `soft_max_abs_err` equals the reconstruction error recomputed from the source within `1e-6` relative plus `16·ulp(max|v|)`. |

With `--source` the attested checks are recomputed: extrema with `nanmax`/`nanmin`, peaks with a naive prominence scan,
crossings and violations with a readable state loop, SED at every source timestamp.

## Tolerances

- Values: bitwise equality.
- Times: `|Δt| ≤ tolerance + 4·ulp(max(|t⁻|, |t⁺|))`.
- Trajectory SED: `≤ ε + 16·ulp(max |coordinate|)`.
- Prominence: `≥ p` computed on retained samples.

Floating-point slack is reported separately from the policy tolerance.

## Tamper resistance

The test suite decodes artifacts, mutates arrays or JSON, re-encodes them with valid CRCs and sizes, and asserts that
verification fails with the expected check id. Edits to soft-only samples cannot be detected from the artifact alone;
they are caught by `--source` (`source.samples`, and `budget.errors` when the edit changes the reconstruction).
