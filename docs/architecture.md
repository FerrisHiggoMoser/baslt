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

Time association order: `time_hints[name]` → dataset attribute `time`/`t` → sibling named `t`, `time`, `Time`,
`timestamp` or `tout` in the same group/struct → `global_time` → top-level `tout` → `SourceError`. Time signals
themselves are not listed as data signals.

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

## Pipeline

```
open_source -> list_signals -> load_policy -> bind_policy -> load(included)
  -> prepass -> ops (per signal) -> closure -> budget search (closure inside every evaluation)
  -> container.write_zip -> self-verify (verify.checks, artifact only) -> bytes
```
