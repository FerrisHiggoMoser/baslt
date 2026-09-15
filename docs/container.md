# The `.baslt` container (profile v1)

A `.baslt` file is a strict subset of ZIP: a few JSON members plus raw little-endian arrays. It is designed to be
decoded exactly by Python's standard library, by MATLAB (Java `Inflater` or `unzip`), and by a browser
(`DecompressionStream('deflate-raw')`), without 64-bit integer arithmetic.

## ZIP subset

All integers are little-endian.

**Local file header** (30 bytes + name), immediately followed by the member data:

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 4 | signature | `50 4B 03 04` |
| 4 | 2 | version needed | 20 (63 for zstd) |
| 6 | 2 | flags | 0 |
| 8 | 2 | method | 0 store, 8 deflate, 93 zstd |
| 10 | 2 | mod time | 0 |
| 12 | 2 | mod date | 0x0021 (1980-01-01) |
| 14 | 4 | CRC-32 | of uncompressed data |
| 18 | 4 | compressed size | |
| 22 | 4 | uncompressed size | |
| 26 | 2 | name length | |
| 28 | 2 | extra length | 0 |
| 30 | n | name | ASCII |

**Central directory entry** (46 bytes + name): signature `50 4B 01 02`, version made by 20, version needed, flags 0,
method, time 0, date 0x0021, CRC-32, compressed size, uncompressed size, name length, extra length 0, comment length
0, disk 0, internal attributes 0, external attributes 0, local header offset, name.

**End of central directory** (22 bytes, always the last 22 bytes of the file): signature `50 4B 05 06`, disk 0,
central directory disk 0, entries on disk, total entries, central directory size, central directory offset, comment
length 0.

Profile rules: no ZIP64 (file < 4 GiB, < 65535 members), no extra fields, no comments, no data descriptors, no
encryption, no directory entries, member names match `[a-z0-9._/]+`. A member is written STORED when deflate does not
make it smaller. Deflate uses zlib level 6, raw (no zlib header).

File size: `22 + Σ (76 + 2·len(name) + compressed_size)`.

## Members, in order

| Name | Method | Content |
|---|---|---|
| `baslt.json` | always STORED | Header. Its name at byte 30 identifies the format. |
| `index.json` | deflate | Signal descriptors, array locations, role legends, bound requirement parameters. |
| `manifest.json` | always STORED | Provenance, budget, per-requirement evidence and statuses. |
| `policy.json` | deflate | Canonical policy, original text, sha256. |
| `s/<k>` | deflate | Arrays of signal k, concatenated. |
| `t/<k>` | deflate | A time array shared by several signals. |

All JSON is UTF-8, sorted keys, separators `,` and `:`, no NaN or Infinity tokens (non-finite numbers are written as
the strings `"NaN"`, `"Infinity"`, `"-Infinity"`).

### baslt.json

```json
{"byte_order":"little","codec":"deflate","container":1,"format":"baslt","level":6,"reader_min":1}
```

### index.json

```json
{
  "container": 1,
  "signals": [
    {
      "name": "q_dyn",
      "path": "aero/q",
      "kind": "continuous",
      "interp": "linear",
      "unit": "Pa",
      "labels": [],
      "n_source": 120001,
      "n": 5231,
      "components": 1,
      "arrays": [
        {"name": "t",     "member": "t/0", "offset": 0,     "nbytes": 41848, "n": 5231, "components": 1, "dtype": "<f8", "enc": "shuffle"},
        {"name": "v",     "member": "s/0", "offset": 0,     "nbytes": 41848, "n": 5231, "components": 1, "dtype": "<f8", "enc": "shuffle"},
        {"name": "idx",   "member": "s/0", "offset": 41848, "nbytes": 20924, "n": 5231, "components": 1, "dtype": "<u4", "enc": "delta+shuffle"},
        {"name": "roles", "member": "s/0", "offset": 62772, "nbytes": 10462, "n": 5231, "components": 1, "dtype": "<u2", "enc": "shuffle"}
      ],
      "roles": [
        {"bit": 0, "id": "extent"},
        {"bit": 1, "id": "hard.q_dyn.global_extrema"},
        {"bit": 2, "id": "soft"}
      ],
      "requirements": [
        {"id": "hard.q_dyn.threshold_crossing[0]", "op": "threshold_crossing", "bits": [3],
         "params": [{"name": "value", "value": 65000.0}, {"name": "edge", "value": "both"}]}
      ]
    }
  ]
}
```

Every element of every list has the same keys, so MATLAB `jsondecode` returns struct arrays. `labels` is an empty
list for non-enum signals. Signals sharing a clock point their `t` descriptor at the same member.

## Array encodings

An array descriptor gives `n` elements × `components` of element type `dtype` (NumPy notation: `<f8`, `<f4`, `<i8`,
`<i4`, `<i2`, `|i1`, `<u8`, `<u4`, `<u2`, `|u1`). Vector arrays are component-sequential: all of component 0, then
component 1, and so on (MATLAB `reshape(a, n, k)`; NumPy `a.reshape(k, n).T`). Let `N = n · components` and `w` the
element width in bytes.

| `enc` | Stored bytes | Decode |
|---|---|---|
| `raw` | the elements' little-endian bytes | read directly |
| `shuffle` | byte planes: `out[p·N + i] = raw[i·w + p]` | invert the byte-plane permutation |
| `delta+shuffle` | shuffle of `d[0] = a[0]`, `d[i] = a[i] − a[i−1]` | unshuffle, then cumulative sum in float64 |

Rules that keep every reader exact:

- Times and float values are stored `shuffle` in their source dtype. They are never delta-encoded.
- `delta+shuffle` is used only for `idx`. `idx` is strictly increasing, so deltas are positive. It is `<u4` when
  `n_source < 2^32`, else `<f8` holding exact integers. Cumulative sums stay below 2^53 and are exact in float64.
- `int64`/`uint64` values are `shuffle` only. JavaScript decodes them as (low, high) uint32 pairs.
- `roles` is the narrowest of `|u1`, `<u2`, `<u4`, `<u8` that holds the legend. Bit k means legend entry with `bit`
  k. `<u8` roles are decoded as (low, high) uint32 pairs.
- Booleans are stored as `|u1`. Enums are `<i4` codes into `labels`.
- `soft_value_dtype: float32` stores `<f4` values only for signals with no hard, event or trajectory role.

## manifest.json (key fields)

```json
{
  "status": "pass",
  "baslt": {"version": "0.1.0", "container": 1},
  "source": {"path": "run.h5", "format": "hdf5", "size_bytes": 8421000000,
             "digest": {"algorithm": "sha256", "mode": "full", "covered_bytes": 8421000000, "value": "..."},
             "signals": 42, "samples_total": 418234901, "issues": []},
  "policy": {"name": "flight_review", "sha256": "..."},
  "artifact": {"size_bytes": "00000000000001984213", "max_bytes": 2097152, "ratio": "4.243812e+03",
               "codec": "deflate", "level": 6},
  "budget": {"source": "policy", "required_bytes": 828412, "discretionary_bytes": 1155801,
             "overhead_bytes": 6120, "signals": [{"name": "q_dyn", "retained": 5231, "hard": 912, "soft": 4319,
             "bytes": 72100, "soft_max_abs_err": 12.5}]},
  "requirements": [{"id": "hard.q_dyn.global_extrema", "signal": "q_dyn", "op": "global_extrema",
                    "severity": "info", "status": "pass", "evidence": {}}],
  "events": [], "trajectories": [], "sync_groups": []
}
```

`size_bytes` is a 20-digit zero-padded string and `ratio` is `%.6e` so the manifest's own length does not depend on
the artifact size. `size_bytes` must equal the file length. Evidence lists hold at most 256 items; counts are always
exact.

### policy.json

```json
{"canonical": {"version": 1, "...": "..."}, "sha256": "...", "source_format": "yaml", "source_text": "version: 1\n..."}
```

## Reader checks

A conforming reader must check: EOCD at `size − 22`; member 0 is `baslt.json` and STORED; methods are 0, 8 or 93
(93 only if zstd is available); flags and extra lengths are 0; CRC-32 and sizes match; `container ≤ supported` and
`reader_min ≤ supported`; every array descriptor lies inside its member and `nbytes = N·w`.

## Determinism

Fixed dates and attributes, fixed member order, canonical JSON and no wall-clock fields make the bytes identical for
the same source, policy, Baslt version and zlib build.
