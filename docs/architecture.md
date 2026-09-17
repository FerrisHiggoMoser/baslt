# Architecture

This page fixes the internal interfaces so modules can be built and tested independently. Semantics live in
`contracts.md`, the policy format in `policy.md`, the file format in `container.md`.

## Rules

- Python ≥ 3.10. `from __future__ import annotations` in every module. snake_case, dataclasses, `pathlib`.
- The only hard runtime dependency is numpy. `h5py`, `scipy` and `yaml` are imported inside functions; a missing
  one raises `UsageError("... install baslt[hdf5]")` (or `[mat]`, `[yaml]`).
- `import baslt`, `import baslt.cli` and `import baslt.errors` must not import numpy.
- No pandas. Vectorized numpy; Python loops only over signals, requirements or reported items, never over samples.
- `baslt.verify.*` must not import `baslt.ops`, `baslt.reduce`, `baslt.plan` or `baslt.manifest` (enforced by an AST
  test). It keeps its own unit table (`verify/si_table.py`) written from `policy.md`.
- Every error is a subclass of `baslt.errors.BasltError` with an `exit_code`.

## Module interfaces

### `baslt.units`

```python
@dataclass(frozen=True, slots=True)
class UnitDef:
    symbol: str
    dimension: str      # "time", "angle", "angular_rate", "pressure", "length", "velocity", "acceleration",
                        # "force", "mass", "temperature", "frequency", "dimensionless", "bytes"
    factor: float       # SI = value * factor + offset
    offset: float = 0.0

@dataclass(frozen=True, slots=True)
class Quantity:
    value: float
    unit: str | None    # None for a bare number
    text: str           # original text, used in canonical form and messages

UNITS: dict[str, UnitDef]
def lookup_unit(symbol: str | None) -> UnitDef | None
def parse_quantity(x: str | int | float, *, path: str = "") -> Quantity          # PolicyError on bad syntax/unit
def convert(q: Quantity, target_unit: str | None, *, delta: bool, path: str = "",
            expect_dimension: str | None = None) -> float
    # bare number -> q.value unchanged; known units -> via SI; dimension mismatch, unit on a unitless signal,
    # or a unit differing from an opaque signal unit -> PolicyError with the path
def to_seconds(q: Quantity, *, path: str = "") -> float                           # quantity must be time or bare
def parse_bytes(x: str | int, *, path: str = "") -> int
```

### `baslt.signals`

```python
Kind = Literal["continuous", "discrete", "vector"]

@dataclass(slots=True)
class SignalInfo:              # cheap description, no data loaded
    name: str                  # canonical name
    path: str                  # path inside the source
    shape: tuple[int, ...]
    dtype: str                 # numpy dtype string, e.g. "<f8"
    unit: str | None
    kind: Kind
    n: int
    time_ref: str | None       # canonical name of its time signal, if known

@dataclass(slots=True)
class Signal:
    name: str
    path: str
    t: np.ndarray              # float64, shape (n,), seconds, non-decreasing, finite
    v: np.ndarray              # shape (n,) or (n, k)
    kind: Kind
    unit: str | None
    labels: list[str] | None   # enum labels for discrete string signals
    source_dtype: str

@dataclass(slots=True)
class SourceMeta:
    path: str | None
    format: str                # "numpy", "csv", "hdf5", "mat", "mat73"
    size_bytes: int | None
    issues: list[str]

@dataclass(slots=True)
class Run:
    signals: dict[str, Signal]
    meta: SourceMeta

def infer_kind(v: np.ndarray) -> Kind
def normalize_signal(name: str, t, v, *, path: str | None = None, unit: str | None = None,
                     kind: Kind | None = None, labels: list[str] | None = None,
                     time_scale: float = 1.0, on_non_monotonic: str = "error") -> tuple[Signal, list[str]]
    # drops non-finite timestamps (recorded as issues), sorts or raises SourceError on decreasing time,
    # rejects complex values and length mismatch; returns (signal, issues). Never copies contiguous float64 input.
```

### `baslt.sources`

```python
class SourceAdapter(Protocol):
    format: ClassVar[str]
    extensions: ClassVar[tuple[str, ...]]
    @classmethod
    def sniff(cls, path: Path, head: bytes) -> bool: ...
    def list_signals(self) -> list[SignalInfo]: ...
    def load(self, names: Sequence[str] | None = None, *, time_hints: Mapping[str, str] | None = None,
             global_time: str | None = None, time_scale: float = 1.0,
             on_non_monotonic: str = "error") -> Run: ...

def open_source(source: str | Path | Mapping, *, format: str | None = None, **options) -> SourceAdapter
```

Adapters may also provide `parameters() -> dict[str, object]`: single numbers, bools and texts stored in the file
(HDF5 scalar datasets and file attributes, MAT-file scalars and character vectors, scalars next to in-memory arrays;
CSV has none). Requirement checks read them with `params.from_source`.

Time association order: `time_hints[name]` → dataset attribute `time`/`t` → sibling named `t`, `time`, `Time`,
`timestamp` or `tout` in the same group/struct → `global_time` → such a clock in the nearest parent group (so a
file-level `/time` serves `/simout/q_dyn`) → top-level `tout` → `SourceError`. Time signals themselves are not listed
as data signals.

Registered formats, in sniffing order: `mat` (aliases `mat73`, `matlab`), `hdf5`, `csv`; `numpy` is chosen for
in-memory input. MAT comes before HDF5 because a v7.3 MAT-file is also a valid HDF5 file, with reversed dimensions.
For the same reason a `.h5`/`.hdf5` file whose header text starts with `MATLAB` (`save('simout.h5', ..., '-v7.3')`)
opens with the MAT reader; `format="hdf5"` still reads it as plain HDF5. MATLAB objects (timeseries, Simulink
datasets) cannot be read without MATLAB; a source with nothing else says so and points to the Structure With Time
and Array logging formats.

MAT-files (`sources/mat_src.py`, `sources/matv73.py`): versions 4–7.2 are read with `scipy.io.loadmat` using MATLAB's
own types (integer-valued doubles stay doubles); a file that holds complex data is read a second time with native
types only to skip those values, because scipy would otherwise cast them to real. Version 7.3 files are read with
h5py: dimensions are reversed back, `MATLAB_class` decides the type, char arrays are UTF-16, cells and struct arrays
follow object references into `#refs#`, and contiguous datasets are memory-mapped. Both build one tree that is
flattened the same way: struct fields become `s/q`, struct array elements `s/1/q`, cell arrays of character vectors
become discrete string signals, and a Simulink "structure with time" (`time` plus `signals`) becomes one signal per
element named by its label and timed by its own `time`. Scalars, char arrays and empty values are not signals;
MATLAB objects, sparse, complex and 3-D arrays are skipped and listed in `Run.meta.issues`. MAT-files have no units.

### `baslt.policy`

```python
def load_policy(source: str | Path | Mapping, *, format: str | None = None) -> Policy   # parse + validate + hash
def bind_policy(policy: Policy, infos: Sequence[SignalInfo], *, max_bytes_override: int | None = None) -> BoundPolicy

@dataclass(frozen=True)
class Param:
    kind: str                    # "value" | "delta" | "duration" | "enum" | "int"
    required: bool = False
    default: object = None       # a string quantity like "0 s", or an enum/int value
    choices: tuple[str, ...] = ()

OP_SCHEMAS: dict[str, dict[str, Param]]      # exactly the table in policy.md
OP_KINDS: dict[str, tuple[str, ...]]         # allowed signal kinds per op

@dataclass
class HardReq:
    id: str; signal: str; op: str; params: dict[str, object]; severity: str   # params hold Quantity or str/int

@dataclass
class Policy:
    version: int
    name: str
    artifact: ArtifactSpec       # max_bytes: int | None, codec, hash: str | None, soft_value_dtype, on_not_applicable
    signals: SignalsSpec         # time, time_unit, on_non_monotonic, include, exclude, decls: dict[str, SignalDecl]
    hard: list[HardReq]
    events: list[EventSpec]
    trajectories: list[TrajectorySpec]
    sync_groups: list[SyncGroupSpec]
    soft: list[SoftRule]
    review: ReviewSpec
    canonical: dict
    sha256: str
    source_format: str           # "yaml" | "json" | "dict"
    source_text: str | None
    locations: dict[str, str]    # path -> "file:line:col" for YAML input

@dataclass
class BoundReq:
    id: str; signal: str; op: str; params: dict[str, float | str | int]; severity: str
    # value/delta params converted to the signal's unit, durations to seconds

@dataclass
class BoundPolicy:
    policy: Policy
    included: list[str]                          # canonical names, source order
    aliases: dict[str, str]                      # alias -> canonical
    reqs: list[BoundReq]
    reqs_by_signal: dict[str, list[BoundReq]]
    events: list[BoundEvent]
    trajectories: list[BoundTrajectory]
    sync_groups: list[BoundSyncGroup]
    soft: dict[str, tuple[int, int | None]]       # canonical -> (weight, max_points)
    max_bytes: int | None
    budget_source: str                           # "policy" | "cli" | "none"
```

### `baslt.container`

```python
@dataclass(slots=True)
class ArrayDesc:
    name: str; member: str; offset: int; nbytes: int; n: int; components: int; dtype: str; enc: str

@dataclass(slots=True)
class Member:
    name: str
    data: bytes          # uncompressed
    method: int          # 0 store, 8 deflate, 93 zstd (writer may downgrade 8 -> 0 when not smaller)

def encode_array(arr: np.ndarray, enc: str) -> bytes            # "raw" | "shuffle" | "delta+shuffle"
def decode_array(buf: bytes | memoryview, desc: ArrayDesc) -> np.ndarray
def compress_member(m: Member, *, level: int = 6) -> tuple[int, bytes, int]    # (method, compressed, crc32)
def write_zip(members: Sequence[Member], *, level: int = 6) -> bytes
def zip_size(names_and_csizes: Iterable[tuple[str, int]]) -> int

@dataclass
class Artifact:
    header: dict; index: dict; manifest: dict; policy: dict; size: int
    def signal_names(self) -> list[str]
    def array(self, signal: str, name: str) -> np.ndarray      # "t", "v", "idx", "roles"

def read_artifact(source: str | Path | bytes) -> Artifact      # stdlib zipfile + profile checks; ContainerError
```

### `baslt.tabular`

Numpy-free tables for requirement files, standard library only.

```python
@dataclass(slots=True)
class Cell:                    # value: str | float | bool | None; text: what a person sees; ref: "F12"
@dataclass(slots=True)
class Table:                   # rows of cells, 1-based access; location(row, col) -> "reqs.xlsx:Requirements!F12"
def read_table(path, *, sheet=None, encoding=None, delimiter=None) -> Table   # .csv/.tsv, .xlsx, .reqif/.reqifz
def list_sheets(path) -> list[SheetInfo]
def write_xlsx(path, sheets: Sequence[Sheet], *, title=None) -> Path       # deterministic bytes, atomic
def patch_xlsx(src, dst, *, sheet, edits: Sequence[CellEdit]) -> list[str] # changes only the edited cells
```

The `.xlsx` reader uses `zipfile` and `ElementTree`, handles shared and inline strings, rich text, percent and date
styles, merged cells and the strict namespace, and refuses DOCTYPE declarations, legacy `.xls` files and parts
that expand too much. The writer emits a minimal workbook (fixed timestamps and part order, verdict fills, number and
percent formats); text that looks like a formula stays text. The patcher rewrites the cells it is given in one sheet
part and copies every other member byte for byte. `reqif.py` reads ReqIF: one row per SPEC-OBJECT in hierarchy
order, one column per attribute long name.

### `baslt.reqs`

Requirement checks, separate from the compiler (`docs/requirements.md`). `api`, `load`, `lint`, `templates`,
`config`, `limits`, `model` and `units_ext` import no numpy.

```
config.load_config(mapping, workbook=, overrides=) -> Config          # YAML/JSON/sheets, merged, validated
load.load_requirements(path, config, only=) -> RequirementSet         # header, where, columns, cases, coverage
limits.parse_limit(cell, unit=, units=) -> LimitSpec                   # the limit-cell grammar
expr.parse(text) -> N ; expr.Binder(index, infos, config).bind(text, role=) -> BoundExpr   # whitelisted AST, types, units
bind.bind_requirements(reqset, infos, param_names=) -> BoundSet        # kinds, limits in the checked unit
align.Signals / Grid                                                   # clocks and resampling (verify.reference)
evaluate.RunContext(run, config, params=)                              # values, three-valued conditions, events, aggregates
check.evaluate_requirement(bound_requirement, ctx) -> RequirementResult
run.check_run(source, reqset, params=, run_id=, digest=) -> (RunResult, RunContext)
results                                                                # text, JSON, CSV, workbook sheets, checked copy
plotdata.build_plots(run, ctx, packer) / report.write_run_report(path, run, ctx, reqset)
batch.check_batch(...) / aggregate.BatchSummary                        # processes, resume, dashboard, batch workbook
archive.derived_policy(reqset, bound, infos) / archive_run(...)        # --archive through api.compile
api.check(runs, requirements, ...) -> CheckResult                      # what the CLI calls
```

A check opens the source, binds the requirements against its signal list (so a requirement with an unknown name is
an ERROR before any data is read), loads only the signals the requirements and events need, detects the events,
and evaluates each requirement on its grid. Expression values are cached per (expression, grid) in a byte-bounded
cache. Batch workers run `batch.check_one` in a `forkserver` process pool (one thread, lower priority) and return
compact summaries with 256-bin envelopes of each plotted signal; the parent writes `index.jsonl` as they arrive and
merges the envelopes onto one time axis for the dashboard.

### `baslt.report`

```python
class Packer:                                  # arrays -> one raw-DEFLATE, base64 blob; refs {"o","n","k",...}
def page(title, body, *, styles, scripts, data=None, blob=None, description=None) -> str
def render_run_page(run, reqset, plots, packer) -> str          # check_page.py
def render_batch_page(summary, data, packer) -> str             # dashboard.py
```

Pages are single files. The content security policy allows only the page's own script and style (by SHA-256 hash)
and loads nothing else. `assets/report.js` and `assets/dash.js` inflate the blob with `DecompressionStream`, draw
canvas plots, and run a self-test with `?selftest=1` that writes array digests and draw counts into
`<pre id="selftest">` for the browser tests.

## Pipeline

```
open_source -> list_signals -> load_policy -> bind_policy -> load(included)
  -> plan.required.evaluate_hard (ops per signal, events)
  -> reduce.rank.preview_order (once per soft signal)
  -> budget search: plan.required.close (soft picks, sync groups) -> encode -> manifest, per trial
  -> reconstruction errors -> container.write_zip -> self-verify (verify.checks, artifact only) -> bytes
```

- `plan.required.evaluate_hard(run, bound)` evaluates the operators and events once. `close(hard, bound, soft)`
  copies the plans, adds the soft picks with the `soft` role and propagates the sync groups; the hard result is
  never changed, so every trial starts from the same place.
- `plan.compile` owns the search. Its assembler caches compressed members by content, so a trial only compresses the
  members whose bytes changed. `minimum_size(run, bound, digest=)` sizes the base artifact without writing it.
- `reduce.rank.preview_order(v, cap)` returns the first `cap` entries of one fixed order of the signal's indices.
- `baslt.advice` (`baslt explain`) reads an infeasible-budget report and, given the source and policy, re-sizes the
  base artifact with one requirement or event relaxed at a time through `minimum_size`.
