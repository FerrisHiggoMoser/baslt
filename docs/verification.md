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
| `structure.bind` | artifact | Re-binding the embedded policy with the verifier's own unit table reproduces every bound parameter in `index.json`. |
| `<requirement id>.<aspect>` | varies | Per-contract checks below. |
| `source.digest` | source | The source digest matches (a sampled digest is reported as a fingerprint, never as sha256). |
| `source.samples` | source | Every retained `(t, v)` is bit-identical to the source at its index. |

## Per-contract checks

| Contract | Aspect | Basis | Check |
|---|---|---|---|
| global_extrema | `max`, `min` | attested | Flagged samples exist, equal the manifest values bitwise, and no retained sample exceeds them. |
| window_extrema | `buckets` | attested | Each flagged bucket's flagged max/min equal the retained max/min of that bucket. |
| local_extrema | `prominence` | artifact | Each claimed peak and its bases are retained and the artifact prominence is at least `p`. |
| local_extrema | `separation` | artifact | Claimed peaks respect the separation rule. |
| threshold_crossing | `brackets` | artifact | Every claimed crossing has adjacent retained source indices that straddle `V`. |
| threshold_crossing | `fidelity` | artifact | The verifier's own detector on the reconstruction returns exactly the claimed crossings within tolerance. |
| violation | `runs` | artifact | Boundary pairs straddle the limit, durations meet `min_duration`, the worst sample lies inside. |
| violation | `fidelity` | artifact | Detection on the reconstruction returns exactly the claimed runs. |
| state_transitions | `transitions` | artifact | Hold transitions over retained samples equal the flagged transitions. |
| event | `trigger`, `windows` | artifact | Trigger bracket valid; each window is a contiguous run of source indices covering `[τ − before, τ + after]` plus one neighbour each side. |
| trajectory | `sed` | attested | Knots present, claimed `max_sed ≤ ε`, linked signals aligned or bracketed at every knot. |
| sync_group | `alignment` | artifact | Every propagating timestamp present bitwise in every member or bracketed; unaligned count matches. |
| budget | `accounting` | artifact | Required + discretionary + overhead bytes equal the file size. |

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
they are caught by `--source`.
