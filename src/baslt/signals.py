"""In-memory run model shared by sources, the compiler and the verifier.

numpy is imported inside functions so that importing this module (for example to use the
SignalInfo type in policy binding) stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .errors import SourceError

if TYPE_CHECKING:
    import numpy as np

Kind = Literal["continuous", "discrete", "vector"]
KINDS: tuple[str, ...] = ("continuous", "discrete", "vector")


@dataclass(slots=True)
class SignalInfo:
    """Cheap description of a signal available in a source. No data is loaded."""

    name: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    unit: str | None
    kind: Kind
    n: int
    time_ref: str | None = None


@dataclass(slots=True)
class Signal:
    """One loaded signal: timestamps in seconds and values."""

    name: str
    path: str
    t: np.ndarray
    v: np.ndarray
    kind: Kind
    unit: str | None
    labels: list[str] | None
    source_dtype: str

    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    @property
    def components(self) -> int:
        return 1 if self.v.ndim == 1 else int(self.v.shape[1])

    def info(self) -> SignalInfo:
        return SignalInfo(
            name=self.name,
            path=self.path,
            shape=tuple(int(s) for s in self.v.shape),
            dtype=self.v.dtype.str,
            unit=self.unit,
            kind=self.kind,
            n=self.n,
            time_ref=None,
        )


@dataclass(slots=True)
class SourceMeta:
    path: str | None
    format: str
    size_bytes: int | None
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Run:
    signals: dict[str, Signal]
    meta: SourceMeta

    def infos(self) -> list[SignalInfo]:
        return [s.info() for s in self.signals.values()]


def infer_kind(v: np.ndarray) -> Kind:
    """Discrete for bool, integer and string values; vector for 2-D floats; otherwise continuous."""
    import numpy as np

    if v.ndim == 2 and v.shape[1] > 1:
        if not np.issubdtype(v.dtype, np.floating):
            raise SourceError("vector signals must have floating-point values")
        return "vector"
    if v.dtype.kind in "biuUSO":
        return "discrete"
    return "continuous"


def _as_1d(a: np.ndarray, what: str, name: str) -> np.ndarray:
    if a.ndim == 1:
        return a
    if a.ndim == 2 and 1 in a.shape:
        return a.reshape(-1)
    if a.ndim == 0:
        return a.reshape(1)
    raise SourceError(f"signal {name!r}: {what} must be one-dimensional, got shape {a.shape}")


def normalize_signal(
    name: str,
    t,
    v,
    *,
    path: str | None = None,
    unit: str | None = None,
    kind: Kind | None = None,
    labels: list[str] | None = None,
    time_scale: float = 1.0,
    on_non_monotonic: str = "error",
) -> tuple[Signal, list[str]]:
    """Validate and canonicalize one signal.

    - time becomes float64 seconds (multiplied by `time_scale` when it is not 1)
    - samples with non-finite timestamps are dropped and reported
    - decreasing time raises SourceError, or is stably sorted with `on_non_monotonic="sort"`
    - values may be (n,), (n, 1), (1, n) or (n, k); a (k, n) array with k != n is transposed
    - string values become int32 codes with `labels`
    - contiguous float64 input is not copied
    """
    import numpy as np

    issues: list[str] = []
    t_arr = _as_1d(np.asarray(t), "time", name)
    if t_arr.dtype.kind not in "fiu":
        raise SourceError(f"signal {name!r}: time must be numeric, got dtype {t_arr.dtype}")
    t_arr = np.asarray(t_arr, dtype=np.float64)
    if time_scale != 1.0:
        t_arr = t_arr * float(time_scale)

    v_arr = np.asarray(v)
    n = t_arr.shape[0]
    if v_arr.dtype.kind == "c":
        raise SourceError(f"signal {name!r}: complex values are not supported")
    if v_arr.ndim == 0:
        v_arr = v_arr.reshape(1)
    if v_arr.ndim == 2:
        if v_arr.shape[0] != n and v_arr.shape[1] == n:
            v_arr = v_arr.T
        if v_arr.shape[1] == 1 or (v_arr.shape[0] == 1 and n == v_arr.shape[1]):
            v_arr = v_arr.reshape(-1)
    elif v_arr.ndim > 2:
        raise SourceError(f"signal {name!r}: values must be 1-D or 2-D, got shape {v_arr.shape}")
    if v_arr.shape[0] != n:
        raise SourceError(f"signal {name!r}: {n} timestamps but {v_arr.shape[0]} values")

    source_dtype = v_arr.dtype.str
    if v_arr.dtype.kind in "USO":
        if v_arr.ndim != 1:
            raise SourceError(f"signal {name!r}: string values must be one-dimensional")
        found, codes = np.unique(v_arr.astype(str), return_inverse=True)
        if labels is None:
            labels = [str(x) for x in found]
            v_arr = codes.astype(np.int32)
        else:
            lookup = {label: i for i, label in enumerate(labels)}
            try:
                v_arr = np.array([lookup[str(x)] for x in found], dtype=np.int32)[codes]
            except KeyError as exc:
                raise SourceError(f"signal {name!r}: value {exc.args[0]!r} is not in its labels") from None
        source_dtype = "str"

    finite_t = np.isfinite(t_arr)
    if not finite_t.all():
        dropped = int((~finite_t).sum())
        issues.append(f"{name}: dropped {dropped} samples with non-finite timestamps")
        t_arr = t_arr[finite_t]
        v_arr = v_arr[finite_t]

    if t_arr.shape[0] > 1:
        decreasing = np.diff(t_arr) < 0
        if decreasing.any():
            if on_non_monotonic != "sort":
                first = int(np.flatnonzero(decreasing)[0])
                raise SourceError(
                    f"signal {name!r}: time decreases at sample {first + 1} "
                    f"({t_arr[first]!r} -> {t_arr[first + 1]!r}); set signals.on_non_monotonic: sort to sort it"
                )
            order = np.argsort(t_arr, kind="stable")
            t_arr = t_arr[order]
            v_arr = v_arr[order]
            issues.append(f"{name}: sorted {int(decreasing.sum())} out-of-order timestamps")

    resolved_kind = kind or infer_kind(v_arr)
    if resolved_kind == "vector" and v_arr.ndim != 2:
        raise SourceError(f"signal {name!r}: declared vector but values are one-dimensional")
    if resolved_kind != "vector" and v_arr.ndim == 2:
        raise SourceError(f"signal {name!r}: values have {v_arr.shape[1]} components; declare kind: vector")

    signal = Signal(
        name=name,
        path=path if path is not None else name,
        t=t_arr,
        v=v_arr,
        kind=resolved_kind,
        unit=unit,
        labels=labels,
        source_dtype=source_dtype,
    )
    return signal, issues
