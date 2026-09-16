"""`verify_artifact`: every check of docs/verification.md for the operators this build compiles.

The verifier starts from the file and works outwards. `structure.py` establishes that the bytes are an artifact and
that the parameters it was compiled with are the policy's own. This module then takes each requirement the manifest
claims and asks what the contract for that operator guarantees:

- what the artifact alone proves (basis `artifact`): the bracketing samples of a claimed crossing really straddle
  the level and really are retained; a claimed violating run really is long enough and really contains its worst
  sample; and -- the check that makes the artifact worth trusting -- re-detecting crossings and violations on the
  *reconstruction* with `verify/reference.py` returns exactly the claimed facts, so a reviewer reading the small
  file sees what the run did.
- what depends on samples nobody kept (basis `attested`): "no larger value exists anywhere" cannot follow from a
  subset. Those claims are checked for internal consistency -- the flagged sample exists, matches the manifest
  bitwise, and no *retained* sample contradicts it -- and are marked `attested` so nobody mistakes them for proof.
- what `--source` can settle (basis `source`): the digest, a bit comparison of every retained sample against the
  source at its index, and an independent recomputation of every claimed fact from the full run. An attested claim
  becomes a `source` claim here, which is the whole point of passing `--source`.

`source` is *loaded* run data -- a `Run`, or any mapping of signal name to `(t, v)` -- and may also be the source
file's path, which is enough for the digest. The verifier deliberately does not open source files: the format
readers are `baslt.sources`, which docs/architecture.md keeps on the compiler's side of the wall (the import test
refuses even to walk into it). The CLI opens the file and hands the loaded run over, so `--source` still recomputes
every fact while the verifier shares no code with the thing it is checking.

Values compare bitwise and times within `tolerance + 4·ulp(max(|t⁻|, |t⁺|))`, as docs/verification.md sets out;
floating-point slack is reported in the TOLERANCE column separately from the policy's own tolerance.

Nothing here imports the compiler. The detectors are `reference.py`, written from the contracts independently, so
agreement between the two is evidence rather than a tautology.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..errors import ContainerError
from . import reference, table
from .structure import (
    BASIS_ARTIFACT,
    BASIS_ATTESTED,
    BASIS_SOURCE,
    FAIL,
    NO_BASIS,
    NOT_APPLICABLE,
    PASS,
    WARN,
    Check,
    structure_checks,
    verdict,
)

if TYPE_CHECKING:
    from ..container import Artifact

__all__ = ["EVIDENCE_LIMIT", "Check", "VerifyResult", "verify_artifact"]

EVIDENCE_LIMIT = 256  # docs/container.md: evidence lists hold at most 256 items, counts are exact

SUPPORTED_OPS: tuple[str, ...] = (
    "global_extrema",
    "local_extrema",
    "window_extrema",
    "threshold_crossing",
    "violation",
    "state_transitions",
)

# Parameter defaults of docs/policy.md, applied when index.json records only the parameters that were written.
PARAM_DEFAULTS: dict[str, dict[str, object]] = {
    "global_extrema": {},
    "local_extrema": {"separation": 0.0, "kind": "both"},
    "window_extrema": {"origin": 0.0},
    "threshold_crossing": {
        "edge": "both",
        "hysteresis": 0.0,
        "debounce": 0.0,
        "tolerance": 0.0,
        "interpolate": "linear",
    },
    "violation": {"min_duration": 0.0},
    "state_transitions": {},
}
REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "local_extrema": ("prominence",),
    "window_extrema": ("interval",),
    "threshold_crossing": ("value",),
}


# --- result ------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class VerifyResult:
    """Every check, with the header facts the report prints above them."""

    checks: list[Check] = field(default_factory=list)
    policy: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    artifact: dict = field(default_factory=dict)
    strict: bool = False
    on_not_applicable: str = "warn"

    @property
    def counts(self) -> dict[str, int]:
        counts = {PASS: 0, WARN: 0, NOT_APPLICABLE: 0, FAIL: 0}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return counts

    @property
    def status(self) -> str:
        """`fail`, `pass_with_warnings` or `pass` (docs/contracts.md, "Statuses")."""
        counts = self.counts
        if counts[FAIL]:
            return "fail"
        if counts[NOT_APPLICABLE] and self.on_not_applicable == "fail":
            return "fail"
        if counts[WARN] or counts[NOT_APPLICABLE]:
            return "pass_with_warnings"
        return "pass"

    @property
    def exit_code(self) -> int:
        """docs/cli.md: 1 when a check failed or `--strict` saw a warning, otherwise 0."""
        status = self.status
        if status == "fail":
            return 1
        return 1 if self.strict and status != "pass" else 0

    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == FAIL]

    def by_id(self, check_id: str) -> Check | None:
        return next((check for check in self.checks if check.id == check_id), None)

    def render(self) -> str:
        return table.render(self)

    def to_json(self) -> dict:
        return table.to_json(self)


# --- retained samples --------------------------------------------------------------------------------------


@dataclass(slots=True)
class SignalView:
    """One signal's retained samples, decoded once and shared by every check on it."""

    name: str
    entry: dict
    t: np.ndarray  # float64 timestamps
    v: np.ndarray  # (n,) or (n, k) values, in the stored dtype
    idx: np.ndarray  # int64 source indices, strictly increasing
    roles: np.ndarray  # uint64 role masks
    n_source: int
    unit: str | None

    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    @property
    def components(self) -> int:
        return 1 if self.v.ndim == 1 else int(self.v.shape[1])

    def column(self, component: int) -> np.ndarray:
        return self.v if self.v.ndim == 1 else self.v[:, component]

    def finite(self) -> np.ndarray:
        return reference.finite(self.v)

    def position(self, source_index: object) -> int:
        """Retained position of a source index, or -1 when that sample was not kept."""
        if isinstance(source_index, bool) or not isinstance(source_index, (int, float, np.integer, np.floating)):
            return -1
        wanted = int(source_index)
        if wanted != source_index:
            return -1
        at = int(np.searchsorted(self.idx, wanted))
        return at if at < self.n and int(self.idx[at]) == wanted else -1

    def flagged(self, bit: int | None) -> np.ndarray:
        """Mask of retained samples carrying `bit`; all False when the requirement has no such role."""
        if bit is None:
            return np.zeros(self.n, dtype=bool)
        mask = np.uint64(1) << np.uint64(bit)
        return np.bitwise_and(self.roles.astype(np.uint64), mask) != np.uint64(0)

    def source_index(self, position: int) -> int:
        return int(self.idx[position])


def _view(artifact: Artifact, entry: dict) -> tuple[SignalView | None, str | None]:
    name = entry.get("name")
    try:
        t = np.asarray(artifact.array(name, "t"), dtype=np.float64)
        values = artifact.array(name, "v")
        idx = artifact.array(name, "idx")
        roles = artifact.array(name, "roles")
    except ContainerError as exc:
        return None, str(exc)
    unit = entry.get("unit")
    return (
        SignalView(
            name=str(name),
            entry=entry,
            t=t,
            v=values,
            idx=idx.astype(np.int64),
            roles=roles.astype(np.uint64),
            n_source=int(entry.get("n_source", t.shape[0])),
            unit=unit if isinstance(unit, str) else None,
        ),
        None,
    )


@dataclass(slots=True)
class SourceSignal:
    """The full signal from `--source`, for the checks that may look at it."""

    t: np.ndarray
    v: np.ndarray

    def column(self, component: int) -> np.ndarray:
        return self.v if self.v.ndim == 1 else self.v[:, component]


# --- small comparisons -------------------------------------------------------------------------------------


def _number(value: object) -> float | None:
    """A manifest number as float64. Non-finite values are written as strings (docs/container.md)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "NaN":
            return float("nan")
        if text == "Infinity":
            return float("inf")
        if text == "-Infinity":
            return float("-inf")
    return None


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def _same_scalar(claimed: object, measured: object) -> bool:
    """Bitwise equality of a claimed value and a retained one (docs/verification.md: values compare bitwise)."""
    if isinstance(claimed, bool) or isinstance(measured, (bool, np.bool_)):
        return bool(claimed) == bool(measured)
    if isinstance(measured, (int, np.integer)) and isinstance(claimed, (int, np.integer)):
        return int(claimed) == int(measured)
    left, right = _number(claimed), _number(measured)
    if left is None or right is None:
        return claimed == measured
    return _bits(left) == _bits(right)


def _same_array(left: np.ndarray, right: np.ndarray) -> int:
    """-1 when every element matches bitwise, else the first differing position."""
    if left.shape != right.shape:
        return 0
    a = np.ascontiguousarray(left)
    b = np.ascontiguousarray(right.astype(left.dtype, copy=False))
    if a.tobytes() == b.tobytes():
        return -1
    differs = a != b
    if a.dtype.kind == "f":
        differs = differs & ~(np.isnan(a) & np.isnan(b))
        if a.dtype.itemsize in (2, 4, 8):
            raw_a = a.view(f"u{a.dtype.itemsize}")
            raw_b = b.view(f"u{b.dtype.itemsize}")
            differs = differs | ((raw_a != raw_b) & ~(np.isnan(a) & np.isnan(b)))
    if differs.ndim == 2:
        differs = differs.any(axis=1)
    hits = np.flatnonzero(differs)
    return int(hits[0]) if hits.size else -1


def _value_changes(values: np.ndarray) -> np.ndarray:
    """Mask of samples differing from the previous one: bitwise, except that any two NaNs are equal."""
    if values.dtype.kind == "f" and values.dtype.itemsize in (2, 4, 8):
        raw = np.ascontiguousarray(values).view(f"u{values.dtype.itemsize}")
        differs = raw[1:] != raw[:-1]
        nan = np.isnan(values)
        differs = differs & ~(nan[1:] & nan[:-1])
    else:
        differs = values[1:] != values[:-1]
    return differs.any(axis=1) if differs.ndim == 2 else differs


def _distinct_count(values: np.ndarray) -> int:
    """Distinct values under the same equality as `_value_changes`."""
    if values.dtype.kind == "f" and values.dtype.itemsize in (2, 4, 8):
        nan = np.isnan(values)
        raw = np.ascontiguousarray(values[~nan]).view(f"u{values.dtype.itemsize}")
        return int(np.unique(raw).size) + int(bool(nan.any()))
    return int(np.unique(values).size)


def _ulp_slack(*times: float, factor: float = 4.0) -> float:
    largest = max((abs(float(t)) for t in times if t is not None), default=0.0)
    return factor * reference.ulp(largest)


# --- formatting --------------------------------------------------------------------------------------------


def _fmt_value(value: object, unit: str | None = None) -> str | None:
    number = _number(value)
    if number is None:
        return None if value is None else str(value)
    text = f"{number:.6g}"
    return f"{text} {unit}" if unit else text


def _fmt_seconds(seconds: object) -> str:
    number = _number(seconds)
    if number is None:
        return "-"
    if number == 0.0:
        return "0"
    if abs(number) < 1e-3:
        return f"{number * 1e6:.6g} us"
    if abs(number) < 1.0:
        return f"{number * 1e3:.6g} ms"
    return f"{number:.6g} s"


def _fmt_count(value: object) -> str:
    return "-" if value is None else str(value)


# --- requirement parameters --------------------------------------------------------------------------------


def _bound_params(requirement: Mapping, op: str) -> tuple[dict, list[str]]:
    """The bound parameters index.json records, with the documented defaults filled in."""
    params: dict = dict(PARAM_DEFAULTS.get(op, {}))
    problems: list[str] = []
    for item in requirement.get("params", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            problems.append("a parameter entry has no string 'name'")
            continue
        params[item["name"]] = item.get("value")
    for name in REQUIRED_PARAMS.get(op, ()):
        if name not in params:
            problems.append(f"index.json does not bind the required parameter {name!r}")
    if op == "violation":
        given = [name for name in ("above", "below") if params.get(name) is not None]
        if len(given) != 1:
            problems.append("violation needs exactly one of 'above' and 'below'")
    return params, problems


def _role_bits(entry: Mapping, requirement: Mapping) -> dict[str, int]:
    """Role name -> legend bit for one requirement. A role id `<requirement>#edge` is the role `edge`."""
    legend = {
        item["bit"]: item.get("id", "")
        for item in entry.get("roles", []) or []
        if isinstance(item, dict) and isinstance(item.get("bit"), int) and not isinstance(item.get("bit"), bool)
    }
    bits: dict[str, int] = {}
    for bit in requirement.get("bits", []) or []:
        if not isinstance(bit, int) or isinstance(bit, bool):
            continue
        role_id = legend.get(bit, "")
        name = role_id.split("#", 1)[1] if "#" in role_id else "main"
        bits.setdefault(name, bit)
    return bits


def _float(value: object, fallback: float = 0.0) -> float:
    number = _number(value)
    return fallback if number is None else number


# --- global_extrema ----------------------------------------------------------------------------------------


def _global_extrema_checks(
    view: SignalView, req_id: str, bits: Mapping[str, int], evidence: Mapping, src: SourceSignal | None
) -> list[Check]:
    items = evidence.get("components")
    if not isinstance(items, list) or not items:
        return [Check(f"{req_id}.max", FAIL, BASIS_ATTESTED, message="the evidence lists no components")]
    by_component: dict[int, dict] = {}
    for position, item in enumerate(items):
        if isinstance(item, dict):
            key = item.get("component", position)
            by_component[int(key) if isinstance(key, int) and not isinstance(key, bool) else position] = item

    bit = bits.get("main")
    finite = view.finite()
    recomputed = None
    if src is not None:
        found = reference.global_extrema(src.v)
        recomputed = found if isinstance(found, list) else [found]

    basis = BASIS_SOURCE if src is not None else BASIS_ATTESTED
    checks: list[Check] = []
    for aspect in ("max", "min"):
        problems: list[str] = []
        claimed_text: str | None = None
        measured_text: str | None = None
        evaluated = 0
        for component in range(view.components):
            item = by_component.get(component)
            if item is None:
                problems.append(f"component {component} is missing from the evidence")
                continue
            claim = item.get(aspect)
            if claim is None:
                continue  # this component has no finite sample; the requirement status covers it
            evaluated += 1
            found, claimed_text, measured_text = _extremum_problems(
                view, component, aspect, claim, bit, finite, recomputed
            )
            problems.extend(found)
        if evaluated == 0:
            checks.append(
                Check(f"{req_id}.{aspect}", NOT_APPLICABLE, NO_BASIS, message=f"no component claims a {aspect}")
            )
            continue
        checks.append(
            verdict(
                f"{req_id}.{aspect}",
                problems,
                basis=basis,
                claimed=claimed_text,
                measured=measured_text,
                allowed="0",
                message=f"the flagged {aspect} matches the manifest and no retained sample beats it",
            )
        )
    return checks


def _extremum_problems(
    view: SignalView,
    component: int,
    aspect: str,
    claim: Mapping,
    bit: int | None,
    finite: np.ndarray,
    recomputed: Sequence[Mapping] | None,
) -> tuple[list[str], str | None, str | None]:
    where = f"{aspect} of component {component}"
    if not isinstance(claim, dict):
        return [f"{where}: the evidence entry is not an object"], None, None
    claimed_value = claim.get("value")
    claimed_text = _fmt_value(claimed_value, view.unit)
    index = claim.get("index")
    at = view.position(index)
    if at < 0:
        return [f"{where}: source sample {index!r} is claimed but not retained"], claimed_text, None

    problems: list[str] = []
    column = view.column(component)
    measured_text = _fmt_value(column[at], view.unit)
    if bit is not None and not view.flagged(bit)[at]:
        problems.append(f"{where}: retained sample {index} does not carry the requirement's role")
    if not _same_scalar(claim.get("t"), view.t[at]):
        problems.append(f"{where}: t is {view.t[at]!r} in the artifact but {claim.get('t')!r} in the manifest")
    if not _same_scalar(claimed_value, column[at]):
        problems.append(f"{where}: value is {column[at]!r} in the artifact but {claimed_value!r} in the manifest")

    number = _number(claimed_value)
    if number is None or not np.isfinite(number):
        problems.append(f"{where}: the claimed value {claimed_value!r} is not a finite number")
    else:
        beats = (column > number) if aspect == "max" else (column < number)
        offenders = np.flatnonzero(finite & beats)
        if offenders.size:
            worst = int(offenders[0])
            problems.append(
                f"{where}: retained source sample {view.source_index(worst)} is {column[worst]!r}, "
                f"which beats the claimed {aspect} {number!r}"
            )

    if recomputed is not None and component < len(recomputed):
        wanted = recomputed[component].get(f"{aspect}_index")
        if wanted is None:
            problems.append(f"{where}: the source has no finite sample, but the manifest claims one")
        elif int(wanted) != int(index):
            problems.append(f"{where}: the source's {aspect} is sample {int(wanted)}, not {int(index)}")
    return problems, claimed_text, measured_text


# --- local_extrema -----------------------------------------------------------------------------------------


def _peak_column(values: np.ndarray, finite: np.ndarray, component: int) -> np.ndarray:
    """One component as float64, NaN wherever the sample is not finite in every component."""
    column = np.asarray(values if values.ndim == 1 else values[:, component], dtype=np.float64)
    return np.where(finite, column, np.nan) if values.ndim == 2 else column


def _peak_labels(kind: object) -> tuple[str, ...]:
    return ("max", "min") if kind == "both" else (str(kind),)


def _local_extrema_checks(
    view: SignalView,
    req_id: str,
    bits: Mapping[str, int],
    params: Mapping,
    evidence: Mapping,
    src: SourceSignal | None,
) -> list[Check]:
    prominence = _float(params.get("prominence"), float("nan"))
    separation = _float(params.get("separation"), 0.0)
    kind = params.get("kind") or "both"
    if kind not in reference.PEAK_KINDS:
        return [Check(f"{req_id}.prominence", FAIL, NO_BASIS, message=f"unknown kind {kind!r}")]
    if not np.isfinite(prominence) or prominence < 0 or not np.isfinite(separation) or separation < 0:
        return [Check(f"{req_id}.prominence", FAIL, NO_BASIS,
                      message=f"bound prominence {params.get('prominence')!r} or separation "
                              f"{params.get('separation')!r} is not a non-negative number")]
    labels = _peak_labels(kind)
    items = [item for item in evidence.get("peaks", []) or [] if isinstance(item, dict)]
    counts = {label: evidence.get("maxima" if label == "max" else "minima") for label in ("max", "min")}
    total = sum(c for c in counts.values() if isinstance(c, int) and not isinstance(c, bool))

    flagged = view.flagged(bits.get("peak"))
    positions = np.flatnonzero(flagged)
    finite = view.finite()
    prominences: dict[tuple[int, str], np.ndarray] = {}
    for component in range(view.components):
        column = _peak_column(view.v, finite, component)
        for label in labels:
            prominences[(component, label)] = reference.prominences_at(column if label == "max" else -column,
                                                                        positions)

    problems: list[str] = []
    if positions.size > total:
        problems.append(f"{positions.size} samples carry the peak role, more than the {total} peaks claimed")
    if total and not positions.size:
        problems.append(f"{total} peaks are claimed but no sample carries the peak role")
    if total and not np.any(view.flagged(bits.get("base"))):
        problems.append("peaks are claimed but no sample carries the base role")
    valid = np.zeros(positions.size, dtype=bool)
    for measured in prominences.values():
        valid |= measured >= prominence
    if not valid.all():
        bad = positions[~valid]
        problems.append(
            f"{bad.size} flagged peak(s) are not peaks of prominence >= {prominence!r} on the retained samples, "
            f"first at source sample {view.source_index(int(bad[0]))}"
        )
    for item in items:
        index, component, label = item.get("index"), item.get("component", 0), item.get("kind")
        where = f"peak at source sample {index!r}"
        at = view.position(index)
        if at < 0 or not flagged[at]:
            problems.append(f"{where}: claimed but not retained with the peak role")
            continue
        if not isinstance(component, int) or not 0 <= component < view.components or label not in labels:
            problems.append(f"{where}: component {component!r} or kind {label!r} does not match the requirement")
            continue
        if not _same_scalar(item.get("t"), view.t[at]):
            problems.append(f"{where}: t is {view.t[at]!r} in the artifact but {item.get('t')!r} in the manifest")
        if not _same_scalar(item.get("value"), view.column(component)[at]):
            problems.append(f"{where}: value differs between the artifact and the manifest")
        here = prominences[(component, label)][int(np.searchsorted(positions, at))]
        claimed = _number(item.get("prominence"))
        if claimed is None or claimed < prominence:
            problems.append(f"{where}: claimed prominence {item.get('prominence')!r} is below {prominence!r}")
        elif not here >= claimed:
            problems.append(
                f"{where}: prominence on the retained samples is {here!r}, below the claimed {claimed!r}; "
                "a base sample is missing"
            )
    checks = [
        verdict(
            f"{req_id}.prominence",
            problems,
            basis=BASIS_ARTIFACT,
            claimed=str(total),
            measured=str(int(positions.size)),
            allowed=f">= {prominence!r}",
            message="every flagged peak keeps at least its prominence on the retained samples",
        )
    ]

    if separation > 0:
        spacing: list[str] = []
        groups: dict[tuple[int, str], list[float]] = {}
        for item in items:
            number = _number(item.get("t"))
            if number is not None:
                groups.setdefault((item.get("component", 0), item.get("kind")), []).append(number)
        for (component, label), times in groups.items():
            times.sort()
            gaps = np.diff(times)
            if gaps.size and gaps.min() < separation:
                spacing.append(f"two claimed {label} peaks of component {component} are {gaps.min()!r} s apart")
        complete = len(items) == total
        checks.append(
            verdict(
                f"{req_id}.separation",
                spacing,
                basis=BASIS_ARTIFACT,
                claimed=_fmt_seconds(separation),
                measured=str(len(items)),
                allowed=_fmt_seconds(separation),
                message="claimed peaks keep the separation"
                + ("" if complete else f" ({len(items)} of {total} peaks are listed in the manifest)"),
            )
        )

    if src is not None:
        checks.append(_local_extrema_source_check(view, req_id, src, prominence, separation, labels, counts, items))
    return checks


def _local_extrema_source_check(
    view: SignalView,
    req_id: str,
    src: SourceSignal,
    prominence: float,
    separation: float,
    labels: Sequence[str],
    counts: Mapping[str, object],
    items: Sequence[Mapping],
) -> Check:
    finite = reference.finite(src.v)
    found: dict[tuple[int, str], list[int]] = {}
    for component in range(1 if src.v.ndim == 1 else src.v.shape[1]):
        column = _peak_column(src.v, finite, component)
        for label in labels:
            peaks = reference.local_peaks(src.t, column, prominence, separation, kind=label)
            found[(component, label)] = [int(peak["index"]) for peak in peaks]
    problems: list[str] = []
    for label in labels:
        expected = sum(len(v) for (c, lab), v in found.items() if lab == label)
        if counts.get(label) != expected:
            problems.append(f"the source has {expected} {label} peaks, the manifest claims {counts.get(label)!r}")
    source_set = {index for indices in found.values() for index in indices}
    for item in items:
        key = (item.get("component", 0), item.get("kind"))
        if item.get("index") not in found.get(key, []):
            problems.append(f"the source has no {item.get('kind')} peak at sample {item.get('index')!r}")
    missing = [index for index in source_set if view.position(index) < 0]
    if missing:
        problems.append(f"{len(missing)} source peak(s) are not retained, first at sample {min(missing)}")
    total = sum(len(v) for v in found.values())
    return verdict(
        f"{req_id}.peaks",
        problems,
        basis=BASIS_SOURCE,
        claimed=str(sum(c for c in counts.values() if isinstance(c, int) and not isinstance(c, bool))),
        measured=str(total),
        allowed="0",
        message="the source's peaks are exactly the claimed ones and all are retained",
    )


# --- window_extrema ----------------------------------------------------------------------------------------


def _window_extrema_checks(
    view: SignalView,
    req_id: str,
    bits: Mapping[str, int],
    params: Mapping,
    evidence: Mapping,
    src: SourceSignal | None,
) -> list[Check]:
    check_id = f"{req_id}.buckets"
    interval = _float(params.get("interval"), float("nan"))
    origin = _float(params.get("origin"), 0.0)
    basis = BASIS_SOURCE if src is not None else BASIS_ATTESTED
    if not np.isfinite(interval) or interval <= 0:
        return [Check(check_id, FAIL, basis, message=f"the bound interval {params.get('interval')!r} is not positive")]

    problems: list[str] = []
    for name, bound in (("interval", interval), ("origin", origin)):
        if name in evidence and not _same_scalar(evidence[name], bound):
            problems.append(f"the evidence records {name} {evidence[name]!r}, but index.json binds {bound!r}")

    try:
        per_component = reference.window_extrema(view.t, view.v, interval, origin)
    except ValueError as exc:
        return [Check(check_id, FAIL, basis, message=f"bucketing the retained samples failed: {exc}")]
    maps = per_component if isinstance(per_component, list) else [per_component]

    flagged = view.flagged(bits.get("main"))
    buckets: set[int] = set()
    for component, bucket_map in enumerate(maps):
        for bucket, (highest, lowest) in bucket_map.items():
            buckets.add(int(bucket))
            for role, at in (("max", highest), ("min", lowest)):
                if not flagged[at]:
                    problems.append(
                        f"bucket {bucket} of component {component}: the retained {role} "
                        f"(source sample {view.source_index(at)}) is not flagged"
                    )
    measured = len(buckets)
    claimed = evidence.get("buckets_with_finite")
    if isinstance(claimed, int) and not isinstance(claimed, bool) and measured > claimed:
        problems.append(f"the retained samples fall in {measured} buckets, more than the {claimed} claimed")

    if src is not None:
        problems.extend(_window_source_problems(view, src, interval, origin, evidence, flagged))
    return [
        verdict(
            check_id,
            problems,
            basis=basis,
            claimed=_fmt_count(claimed),
            measured=str(measured),
            allowed="0",
            message="every bucket's retained extrema are flagged",
        )
    ]


def _window_source_problems(
    view: SignalView,
    src: SourceSignal,
    interval: float,
    origin: float,
    evidence: Mapping,
    flagged: np.ndarray,
) -> list[str]:
    problems: list[str] = []
    try:
        counts = reference.window_evidence(src.t, src.v, interval, origin)
        per_component = reference.window_extrema(src.t, src.v, interval, origin)
    except ValueError as exc:
        return [f"bucketing the source failed: {exc}"]
    for name in ("buckets", "buckets_with_finite"):
        if name in evidence and not _same_scalar(evidence[name], counts[name]):
            problems.append(f"the source has {counts[name]} {name}, but the manifest claims {evidence[name]!r}")
    maps = per_component if isinstance(per_component, list) else [per_component]
    for component, bucket_map in enumerate(maps):
        for bucket, (highest, lowest) in bucket_map.items():
            for role, source_index in (("max", highest), ("min", lowest)):
                at = view.position(source_index)
                if at < 0:
                    problems.append(
                        f"bucket {bucket} of component {component}: the source {role} (sample {source_index}) "
                        "is not retained"
                    )
                elif not flagged[at]:
                    problems.append(
                        f"bucket {bucket} of component {component}: the source {role} (sample {source_index}) "
                        "is retained but not flagged"
                    )
    return problems


# --- threshold_crossing ------------------------------------------------------------------------------------


def _detect_crossings(t: np.ndarray, values: np.ndarray, params: Mapping) -> tuple[list[dict], dict]:
    """`reference.crossings` per component, merged the way the evidence is ordered: by time, then component."""
    level = _float(params.get("value"))
    edge = str(params.get("edge", "both"))
    hysteresis = _float(params.get("hysteresis"))
    debounce = _float(params.get("debounce"))
    interpolate = str(params.get("interpolate", "linear"))
    columns = [values] if values.ndim == 1 else [values[:, c] for c in range(values.shape[1])]

    records: list[dict] = []
    aggregate = {"count": 0, "rising": 0, "falling": 0, "pending_at_end": False, "gap_flips": 0}
    for component, column in enumerate(columns):
        found = reference.crossings(
            t, column, level, edge=edge, hysteresis=hysteresis, debounce=debounce, interpolate=interpolate
        )
        aggregate["pending_at_end"] = aggregate["pending_at_end"] or bool(found["pending_at_end"])
        aggregate["gap_flips"] += int(found["gap_flips"])
        for item in found["accepted"]:
            records.append({**item, "component": component})
            aggregate["count"] += 1
            aggregate["rising" if item["edge"] == "rising" else "falling"] += 1
    records.sort(key=lambda item: (item["t"], item["component"]))
    return records, aggregate


def _crossing_checks(
    view: SignalView,
    req_id: str,
    bits: Mapping[str, int],
    params: Mapping,
    evidence: Mapping,
    src: SourceSignal | None,
) -> list[Check]:
    claimed_items = [item for item in (evidence.get("crossings") or []) if isinstance(item, dict)]
    tolerance = _float(params.get("tolerance"))
    checks = [_crossing_brackets_check(view, req_id, bits, params, evidence, claimed_items)]
    checks.append(_crossing_fidelity_check(view, req_id, params, evidence, claimed_items, tolerance))
    if src is not None:
        checks.append(_crossing_source_check(src, req_id, params, evidence, claimed_items))
    return checks


def _crossing_brackets_check(
    view: SignalView,
    req_id: str,
    bits: Mapping[str, int],
    params: Mapping,
    evidence: Mapping,
    claimed_items: Sequence[Mapping],
) -> Check:
    level = _float(params.get("value"))
    flagged = view.flagged(bits.get("main"))
    problems: list[str] = []
    checked = 0
    for item in claimed_items:
        component = int(item.get("component", 0) or 0)
        index_before = item.get("index_before")
        edge = item.get("edge")
        where = f"crossing at t={item.get('t')!r}"
        at = view.position(index_before)
        if at < 0:
            problems.append(f"{where}: the bracketing sample {index_before!r} is not retained")
            continue
        if at + 1 >= view.n:
            problems.append(f"{where}: no retained sample follows source sample {index_before}")
            continue
        checked += 1
        column = view.column(component)
        finite_after = np.flatnonzero(np.isfinite(column[at + 1:]))
        if not finite_after.size:
            problems.append(f"{where}: no finite sample follows its bracket")
            continue
        right = at + 1 + int(finite_after[0])
        before, after = float(column[at]), float(column[right])
        if edge == "rising":
            straddles = before < level <= after
        elif edge == "falling":
            straddles = before >= level > after
        else:
            problems.append(f"{where}: unknown edge {edge!r}")
            continue
        if not straddles:
            problems.append(
                f"{where}: the retained pair {before!r} -> {after!r} does not straddle {level!r} {edge}"
            )
        if not (flagged[at] and flagged[right]):
            problems.append(f"{where}: its bracketing pair is retained but not flagged")
        if view.source_index(right) != int(index_before) + 1 and (right == at + 1 or not int(evidence.get("gap_flips", 0) or 0)):
            problems.append(
                f"{where}: source samples {index_before} and {view.source_index(right)} are not adjacent, "
                "but the evidence records no gap flips"
            )
    return verdict(
        f"{req_id}.brackets",
        problems,
        claimed=_fmt_count(evidence.get("count")),
        measured=str(checked),
        allowed="0",
        message="every claimed crossing is bracketed by adjacent retained samples that straddle the level",
    )


def _crossing_fidelity_check(
    view: SignalView,
    req_id: str,
    params: Mapping,
    evidence: Mapping,
    claimed_items: Sequence[Mapping],
    tolerance: float,
) -> Check:
    check_id = f"{req_id}.fidelity"
    allowed = _fmt_seconds(tolerance) + " + 4 ulp"
    try:
        records, aggregate = _detect_crossings(view.t, view.v, params)
    except (ValueError, AssertionError) as exc:
        return Check(check_id, FAIL, BASIS_ARTIFACT, message=f"re-detection on the reconstruction failed: {exc}")

    problems = _compare_crossing_counts(evidence, aggregate)
    problems.extend(
        _compare_crossing_items(
            claimed_items,
            records,
            tolerance=tolerance,
            source_index=view.source_index,
            times=view.t,
            where="the reconstruction",
        )
    )
    caveats = _crossing_caveats(evidence, aggregate, strict=False)
    return _crossing_verdict(
        check_id,
        problems,
        caveats,
        basis=BASIS_ARTIFACT,
        claimed=_fmt_count(evidence.get("count")),
        measured=str(aggregate["count"]),
        allowed=allowed,
        message="re-detection on the reconstruction returns exactly the claimed crossings",
    )


def _crossing_source_check(
    src: SourceSignal, req_id: str, params: Mapping, evidence: Mapping, claimed_items: Sequence[Mapping]
) -> Check:
    check_id = f"{req_id}.source"
    try:
        records, aggregate = _detect_crossings(src.t, src.v, params)
    except (ValueError, AssertionError) as exc:
        return Check(check_id, FAIL, BASIS_SOURCE, message=f"re-detection on the source failed: {exc}")
    problems = _compare_crossing_counts(evidence, aggregate)
    problems.extend(
        _compare_crossing_items(
            claimed_items,
            records,
            tolerance=0.0,
            source_index=lambda position: position,
            times=src.t,
            where="the source",
        )
    )
    caveats = _crossing_caveats(evidence, aggregate, strict=True)
    return _crossing_verdict(
        check_id,
        problems + [c for c in caveats if c.startswith("the source")],
        [c for c in caveats if not c.startswith("the source")],
        basis=BASIS_SOURCE,
        claimed=_fmt_count(evidence.get("count")),
        measured=str(aggregate["count"]),
        allowed="4 ulp",
        message="detection on the source returns exactly the claimed crossings",
    )


def _crossing_verdict(
    check_id: str,
    problems: list[str],
    caveats: list[str],
    *,
    basis: str,
    claimed: str | None,
    measured: str | None,
    allowed: str | None,
    message: str,
) -> Check:
    if problems:
        return verdict(check_id, problems, basis=basis, claimed=claimed, measured=measured, allowed=allowed)
    if caveats:
        return Check(check_id, WARN, basis, claimed, measured, allowed, "; ".join(caveats))
    return Check(check_id, PASS, basis, claimed, measured, allowed, message)


def _compare_crossing_counts(evidence: Mapping, aggregate: Mapping) -> list[str]:
    problems: list[str] = []
    for name in ("count", "rising", "falling"):
        claimed = evidence.get(name)
        if claimed is None:
            continue
        if not _same_scalar(claimed, aggregate[name]):
            problems.append(f"{name}: the manifest claims {claimed!r}, detection finds {aggregate[name]}")
    return problems


def _compare_crossing_items(
    claimed_items: Sequence[Mapping],
    records: Sequence[Mapping],
    *,
    tolerance: float,
    source_index,
    times: np.ndarray,
    where: str,
) -> list[str]:
    """Compare the evidence list, which holds the first 256 crossings in time order, with the detected ones."""
    problems: list[str] = []
    listed = list(records)[:EVIDENCE_LIMIT]
    if len(claimed_items) != min(len(listed), EVIDENCE_LIMIT) and len(claimed_items) != len(listed):
        problems.append(f"the evidence lists {len(claimed_items)} crossings, {where} has {len(listed)}")
    for position, claim in enumerate(claimed_items):
        if position >= len(listed):
            problems.append(f"crossing {position} at t={claim.get('t')!r} is claimed but not found in {where}")
            continue
        found = listed[position]
        if claim.get("edge") != found["edge"]:
            problems.append(
                f"crossing {position}: the manifest says {claim.get('edge')!r}, {where} gives {found['edge']!r}"
            )
        wanted = source_index(found["index_before"])
        if not _same_scalar(claim.get("index_before"), wanted):
            problems.append(
                f"crossing {position}: index_before is {claim.get('index_before')!r} in the manifest, "
                f"{wanted} in {where}"
            )
        claimed_time = _number(claim.get("t"))
        if claimed_time is None:
            problems.append(f"crossing {position}: t is {claim.get('t')!r}, not a number")
            continue
        slack = tolerance + _ulp_slack(times[found["index_before"]], times[found["index_after"]])
        if not abs(claimed_time - found["t"]) <= slack:
            problems.append(
                f"crossing {position}: t is {claimed_time!r} in the manifest but {found['t']!r} in {where}, "
                f"outside {slack!r}"
            )
    return problems


def _crossing_caveats(evidence: Mapping, aggregate: Mapping, *, strict: bool) -> list[str]:
    """`pending_at_end` and gap flips are WARN caveats; against the source a disagreement is a failure.

    On the artifact alone the two step-4 values are measured on the reconstruction rather than on the run, so a
    disagreement is reported as a caveat instead of a contradiction.
    """
    caveats: list[str] = []
    claimed_pending = bool(evidence.get("pending_at_end", False))
    claimed_gaps = int(evidence.get("gap_flips", 0) or 0)
    if claimed_pending:
        caveats.append("the final run is shorter than the debounce (pending_at_end)")
    if claimed_gaps:
        caveats.append(f"{claimed_gaps} state flip(s) span non-finite samples")
    prefix = "the source " if strict else ""
    if claimed_pending != bool(aggregate["pending_at_end"]):
        caveats.append(f"{prefix}pending_at_end is {aggregate['pending_at_end']}, the manifest claims {claimed_pending}")
    if claimed_gaps != int(aggregate["gap_flips"]):
        caveats.append(f"{prefix}gap flips are {aggregate['gap_flips']}, the manifest claims {claimed_gaps}")
    return caveats


# --- violation ---------------------------------------------------------------------------------------------


def _detect_runs(t: np.ndarray, values: np.ndarray, params: Mapping) -> list[dict]:
    """`reference.violation_runs` per component, ordered the way the evidence is: by start, then component."""
    above = _number(params.get("above"))
    below = _number(params.get("below"))
    limits = {"above": above} if above is not None else {"below": below}
    minimum = _float(params.get("min_duration"))
    columns = [values] if values.ndim == 1 else [values[:, c] for c in range(values.shape[1])]
    records: list[dict] = []
    for component, column in enumerate(columns):
        for item in reference.violation_runs(t, column, min_duration=minimum, **limits):
            records.append({**item, "component": component})
    records.sort(key=lambda item: (item["start"], item["component"]))
    return records


def _violation_checks(
    view: SignalView,
    req_id: str,
    bits: Mapping[str, int],
    params: Mapping,
    evidence: Mapping,
    src: SourceSignal | None,
) -> list[Check]:
    claimed_items = [item for item in (evidence.get("runs") or []) if isinstance(item, dict)]
    checks = [_violation_runs_check(view, req_id, bits, params, evidence, claimed_items)]
    checks.append(_violation_fidelity_check(view, req_id, params, evidence, claimed_items))
    if src is not None:
        checks.append(_violation_source_check(src, req_id, params, evidence, claimed_items))
    return checks


def _violation_runs_check(
    view: SignalView,
    req_id: str,
    bits: Mapping[str, int],
    params: Mapping,
    evidence: Mapping,
    claimed_items: Sequence[Mapping],
) -> Check:
    above = _number(params.get("above"))
    limit = above if above is not None else _number(params.get("below"))
    minimum = _float(params.get("min_duration"))
    edge_flagged = view.flagged(bits.get("edge"))
    worst_flagged = view.flagged(bits.get("worst"))
    finite = view.finite()
    problems: list[str] = []
    checked = 0

    for item in claimed_items:
        component = int(item.get("component", 0) or 0)
        start, end = _number(item.get("start")), _number(item.get("end"))
        where = f"run [{item.get('start')!r}, {item.get('end')!r}]"
        if start is None or end is None or limit is None:
            problems.append(f"{where}: start, end or the limit is not a number")
            continue
        checked += 1
        slack = _ulp_slack(start, end)
        if end - start < minimum - slack:
            problems.append(f"{where}: lasts {end - start!r}, under the {minimum!r} s minimum duration")

        column = view.column(component)
        violating = finite & ((column > limit) if above is not None else (column < limit))
        inside = np.flatnonzero(violating & (view.t >= start - slack) & (view.t <= end + slack))
        if inside.size == 0:
            problems.append(f"{where}: no retained sample inside it violates the limit {limit!r}")
            continue
        problems.extend(
            _violation_edge_problems(view, item, inside, violating, finite, edge_flagged, where)
        )
        problems.extend(
            _violation_worst_problems(view, item, component, above, limit, worst_flagged, start, end, slack, where)
        )

    total = evidence.get("total_duration")
    return verdict(
        f"{req_id}.runs",
        problems,
        claimed=_fmt_count(evidence.get("count")),
        measured=str(checked),
        allowed=_fmt_seconds(minimum),
        message=f"every claimed run straddles the limit and holds its worst sample (total {total!r} s)",
    )


def _violation_edge_problems(
    view: SignalView,
    item: Mapping,
    inside: np.ndarray,
    violating: np.ndarray,
    finite: np.ndarray,
    edge_flagged: np.ndarray,
    where: str,
) -> list[str]:
    flags = set(item.get("flags") or [])
    problems: list[str] = []
    first, last = int(inside[0]), int(inside[-1])
    for side, at, step in (("start", first, -1), ("end", last, 1)):
        if f"open_{side}" in flags:
            wanted = 0 if step < 0 else view.n_source - 1
            if view.source_index(at) != wanted:
                problems.append(f"{where}: open_{side} but the run's {side} is source sample {view.source_index(at)}")
            continue
        neighbour = at + step
        if not 0 <= neighbour < view.n:
            problems.append(f"{where}: the {side} boundary has no retained neighbour")
            continue
        if f"gap_{side}" in flags:
            if finite[neighbour]:
                problems.append(f"{where}: gap_{side} but the neighbouring retained sample is finite")
            continue
        if violating[neighbour]:
            problems.append(f"{where}: the sample beyond the {side} boundary also violates the limit")
        if not (edge_flagged[at] and edge_flagged[neighbour]):
            problems.append(f"{where}: its {side} boundary pair is not flagged with the edge role")
    return problems


def _violation_worst_problems(
    view: SignalView,
    item: Mapping,
    component: int,
    above: float | None,
    limit: float,
    worst_flagged: np.ndarray,
    start: float,
    end: float,
    slack: float,
    where: str,
) -> list[str]:
    worst_t = _number(item.get("worst_t"))
    worst_value = item.get("worst_value")
    if worst_t is None:
        return [f"{where}: worst_t is {item.get('worst_t')!r}, not a number"]
    if not (start - slack <= worst_t <= end + slack):
        return [f"{where}: its worst sample at t={worst_t!r} lies outside the run"]
    column = view.column(component)
    low = int(np.searchsorted(view.t, worst_t, side="left"))
    high = int(np.searchsorted(view.t, worst_t, side="right"))
    for at in range(low, high):
        if _same_scalar(worst_value, column[at]) and worst_flagged[at]:
            beyond = column[at] > limit if above is not None else column[at] < limit
            if not beyond:
                return [f"{where}: its worst sample {worst_value!r} does not pass the limit {limit!r}"]
            return []
    return [f"{where}: no retained sample at t={worst_t!r} holds the worst value {worst_value!r} with the worst role"]


def _violation_fidelity_check(
    view: SignalView, req_id: str, params: Mapping, evidence: Mapping, claimed_items: Sequence[Mapping]
) -> Check:
    check_id = f"{req_id}.fidelity"
    try:
        records = _detect_runs(view.t, view.v, params)
    except ValueError as exc:
        return Check(check_id, FAIL, BASIS_ARTIFACT, message=f"re-detection on the reconstruction failed: {exc}")
    problems = _compare_runs(evidence, claimed_items, records, view.column, "the reconstruction")
    return verdict(
        check_id,
        problems,
        claimed=_fmt_count(evidence.get("count")),
        measured=str(len(records)),
        allowed="4 ulp",
        message="re-detection on the reconstruction returns exactly the claimed runs",
    )


def _violation_source_check(
    src: SourceSignal, req_id: str, params: Mapping, evidence: Mapping, claimed_items: Sequence[Mapping]
) -> Check:
    check_id = f"{req_id}.source"
    try:
        records = _detect_runs(src.t, src.v, params)
    except ValueError as exc:
        return Check(check_id, FAIL, BASIS_SOURCE, message=f"detection on the source failed: {exc}")
    problems = _compare_runs(evidence, claimed_items, records, src.column, "the source")
    return verdict(
        check_id,
        problems,
        basis=BASIS_SOURCE,
        claimed=_fmt_count(evidence.get("count")),
        measured=str(len(records)),
        allowed="4 ulp",
        message="detection on the source returns exactly the claimed runs",
    )


def _compare_runs(
    evidence: Mapping,
    claimed_items: Sequence[Mapping],
    records: Sequence[Mapping],
    column_of,
    where: str,
) -> list[str]:
    problems: list[str] = []
    claimed_count = evidence.get("count")
    if claimed_count is not None and not _same_scalar(claimed_count, len(records)):
        problems.append(f"the manifest claims {claimed_count!r} runs, {where} has {len(records)}")
    listed = list(records)[:EVIDENCE_LIMIT]
    for position, claim in enumerate(claimed_items):
        if position >= len(listed):
            problems.append(f"run {position} starting at {claim.get('start')!r} is claimed but not found in {where}")
            continue
        found = listed[position]
        for name in ("start", "end"):
            claimed_time = _number(claim.get(name))
            if claimed_time is None:
                problems.append(f"run {position}: {name} is {claim.get(name)!r}, not a number")
                continue
            slack = _ulp_slack(claimed_time, found[name])
            if not abs(claimed_time - found[name]) <= slack:
                problems.append(
                    f"run {position}: {name} is {claimed_time!r} in the manifest but {found[name]!r} in {where}"
                )
        column = column_of(found["component"])
        if not _same_scalar(claim.get("worst_value"), column[found["worst_index"]]):
            problems.append(
                f"run {position}: the worst value is {column[found['worst_index']]!r} in {where}, "
                f"but {claim.get('worst_value')!r} in the manifest"
            )
    total = _number(evidence.get("total_duration"))
    if total is not None and records:
        measured_total = float(sum(item["end"] - item["start"] for item in records))
        slack = _ulp_slack(total, measured_total, factor=4.0 * max(len(records), 1))
        if not abs(total - measured_total) <= slack:
            problems.append(f"total_duration is {total!r} in the manifest but {measured_total!r} in {where}")
    return problems


# --- state_transitions -------------------------------------------------------------------------------------


def _state_transition_checks(
    view: SignalView, req_id: str, bits: Mapping[str, int], evidence: Mapping, src: SourceSignal | None
) -> list[Check]:
    check_id = f"{req_id}.transitions"
    flagged = view.flagged(bits.get("main"))
    problems: list[str] = []
    for at in reference.state_transitions(view.v):
        if not flagged[at]:
            problems.append(f"retained sample {view.source_index(at)} is a hold transition but is not flagged")

    measured = int(np.count_nonzero(_value_changes(view.v)))
    claimed = evidence.get("transitions")
    if claimed is not None and not _same_scalar(claimed, measured):
        problems.append(f"the manifest claims {claimed!r} transitions, the retained samples show {measured}")
    distinct = _distinct_count(view.v)
    claimed_distinct = evidence.get("distinct_count")
    if claimed_distinct is not None and not _same_scalar(claimed_distinct, distinct):
        problems.append(f"the manifest claims {claimed_distinct!r} distinct values, the artifact holds {distinct}")

    checks = [
        verdict(
            check_id,
            problems,
            claimed=_fmt_count(claimed),
            measured=str(measured),
            allowed="0",
            message="every hold transition over the retained samples is flagged and counted",
        )
    ]
    if src is not None:
        checks.append(_state_transition_source_check(view, src, req_id, bits, evidence))
    return checks


def _state_transition_source_check(
    view: SignalView, src: SourceSignal, req_id: str, bits: Mapping[str, int], evidence: Mapping
) -> Check:
    flagged = view.flagged(bits.get("main"))
    problems: list[str] = []
    for source_index in reference.state_transitions(src.v):
        at = view.position(source_index)
        if at < 0:
            problems.append(f"source sample {source_index} changes value but is not retained")
        elif not flagged[at]:
            problems.append(f"source sample {source_index} changes value but is not flagged")

    measured = int(np.count_nonzero(_value_changes(src.v)))
    claimed = evidence.get("transitions")
    if claimed is not None and not _same_scalar(claimed, measured):
        problems.append(f"the manifest claims {claimed!r} transitions, the source has {measured}")
    distinct = _distinct_count(src.v)
    claimed_distinct = evidence.get("distinct_count")
    if claimed_distinct is not None and not _same_scalar(claimed_distinct, distinct):
        problems.append(f"the manifest claims {claimed_distinct!r} distinct values, the source has {distinct}")

    held = reference.reconstruct_hold(view.t, view.v, src.t)
    differs = _same_array(np.ascontiguousarray(src.v), held)
    if differs >= 0:
        problems.append(
            f"hold reconstruction differs from the source at sample {differs}: "
            f"{held[differs]!r} instead of {src.v[differs]!r}"
        )
    return verdict(
        f"{req_id}.source",
        problems,
        basis=BASIS_SOURCE,
        claimed=_fmt_count(claimed),
        measured=str(measured),
        allowed="0",
        message="hold reconstruction equals the source at every source timestamp",
    )


# --- source checks -----------------------------------------------------------------------------------------


def _digest_target(source: object, mode: str) -> object | None:
    """What a digest of this mode must be recomputed over: the source file, or the arrays that were hashed."""
    if mode == "arrays":
        signals = getattr(source, "signals", None)
        if isinstance(signals, Mapping):
            return {f"{name}/{part}": getattr(sig, part) for name, sig in signals.items() for part in ("t", "v")}
        # The arrays digest covers exactly the mapping that was hashed; a mapping of (t, v) pairs is run data,
        # not that mapping, and hashing it would report a mismatch that says nothing about the source.
        if isinstance(source, Mapping) and source and all(isinstance(v, np.ndarray) for v in source.values()):
            return source
        return None
    if isinstance(source, (str, Path)):
        return source
    path = getattr(getattr(source, "meta", None), "path", None)
    return path if isinstance(path, (str, Path)) else None


def _digest_check(artifact: Artifact, source: object) -> Check:
    from .. import hashing

    recorded = (artifact.manifest.get("source") or {}).get("digest")
    if not isinstance(recorded, dict):
        return Check("source.digest", NOT_APPLICABLE, NO_BASIS, message="the manifest records no source digest")
    info = hashing.HashInfo.from_json(recorded)
    if info.mode == "none":
        return Check("source.digest", NOT_APPLICABLE, NO_BASIS, message="the artifact was compiled with --hash none")
    claimed = f"{info.algorithm} ({info.mode}) {info.value[:8]}…" if info.value else info.algorithm
    target = _digest_target(source, info.mode)
    if target is None:
        needed = "the source arrays" if info.mode == "arrays" else "the path of the source file"
        return Check(
            "source.digest", NOT_APPLICABLE, BASIS_SOURCE, claimed,
            message=f"rechecking the {info.mode} digest needs {needed}",
        )
    try:
        found = hashing.recompute(target, info)
    except (ValueError, OSError) as exc:
        return Check("source.digest", NOT_APPLICABLE, BASIS_SOURCE, claimed, message=str(exc))

    measured = f"{found.algorithm} ({found.mode}) {found.value[:8]}…"
    if found.value != info.value:
        return Check(
            "source.digest",
            FAIL,
            BASIS_SOURCE,
            claimed,
            measured,
            "0",
            f"the source digests to {found.value}, but the manifest records {info.value}",
        )
    if not info.is_proof:
        return Check(
            "source.digest",
            WARN,
            BASIS_SOURCE,
            claimed,
            measured,
            "0",
            f"a {info.mode} digest covers {info.covered_bytes} bytes: it is a fingerprint, not a sha256 match",
        )
    return Check("source.digest", PASS, BASIS_SOURCE, claimed, measured, "0", "the source digest matches")


def _sample_check(views: Sequence[SignalView], signals: Mapping[str, SourceSignal]) -> Check:
    problems: list[str] = []
    compared = 0
    for view in views:
        signal = signals.get(view.name)
        if signal is None:
            problems.append(f"signal {view.name!r} is in the artifact but not in the source")
            continue
        t = np.asarray(signal.t, dtype=np.float64)
        values = np.asarray(signal.v)
        if t.shape[0] != view.n_source:
            problems.append(
                f"signal {view.name!r}: the source has {t.shape[0]} samples, the artifact records {view.n_source}"
            )
            continue
        if view.n and int(view.idx[-1]) >= t.shape[0]:
            problems.append(f"signal {view.name!r}: idx reaches {int(view.idx[-1])}, past the source")
            continue
        compared += view.n
        differs = _same_array(view.t, t[view.idx])
        if differs >= 0:
            problems.append(
                f"signal {view.name!r}: retained timestamp {differs} is {view.t[differs]!r}, "
                f"but the source has {t[view.idx[differs]]!r} at source sample {view.source_index(differs)}"
            )
        kept = values[view.idx]
        if kept.dtype != view.v.dtype:
            problems.append(
                f"signal {view.name!r}: values are stored as {view.v.dtype} but the source is {kept.dtype}"
            )
            continue
        differs = _same_array(view.v, kept)
        if differs >= 0:
            problems.append(
                f"signal {view.name!r}: retained value {differs} is {view.v[differs]!r}, "
                f"but the source has {kept[differs]!r} at source sample {view.source_index(differs)}"
            )
    return verdict(
        "source.samples",
        problems,
        basis=BASIS_SOURCE,
        claimed=str(compared),
        measured=str(compared - len(problems)),
        allowed="0",
        message=f"all {compared} retained samples are bit-identical to the source",
    )


def _source_signals(source: object) -> dict[str, SourceSignal]:
    """Signals from loaded run data: a `Run` (any object with a `signals` mapping) or a mapping of `(t, v)` pairs.

    Entries that are bare arrays are skipped rather than guessed at: pairing a value array with its clock is the
    source adapter's job, and the adapter lives on the compiler's side of the import wall.
    """
    items = getattr(source, "signals", None)
    if not isinstance(items, Mapping):
        items = source if isinstance(source, Mapping) else None
    if items is None:
        return {}
    out: dict[str, SourceSignal] = {}
    for name, value in items.items():
        t, values = getattr(value, "t", None), getattr(value, "v", None)
        if t is None and isinstance(value, (tuple, list)) and len(value) == 2:
            t, values = value
        if t is None or values is None:
            continue
        out[str(name)] = SourceSignal(t=np.asarray(t, dtype=np.float64), v=np.asarray(values))
    return out


# --- the whole pass ----------------------------------------------------------------------------------------


def _requirement_checks(
    view: SignalView | None,
    claim: Mapping,
    requirement: Mapping,
    entry: Mapping,
    src: SourceSignal | None,
) -> list[Check]:
    req_id = str(claim.get("id"))
    op = claim.get("op") or requirement.get("op")
    status = claim.get("status")
    if view is None:
        return [Check(req_id, FAIL, NO_BASIS, message="its signal could not be decoded")]
    if op not in SUPPORTED_OPS:
        return [Check(req_id, FAIL, NO_BASIS, message=f"operator {op!r} is not supported by this verifier")]

    bits = _role_bits(entry, requirement)
    evidence = claim.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    if status == "not_applicable":
        flagged = int(np.count_nonzero(view.flagged(bits.get("main")))) if "main" in bits else 0
        if flagged:
            return [
                Check(req_id, FAIL, NO_BASIS, message=f"it is not applicable but {flagged} sample(s) carry its role")
            ]
        notes = claim.get("notes") or []
        reason = "; ".join(str(note) for note in notes) if isinstance(notes, list) else ""
        return [Check(req_id, NOT_APPLICABLE, NO_BASIS, message=reason or "the operator could not be evaluated")]

    params, problems = _bound_params(requirement, op)
    if problems:
        return [Check(req_id, FAIL, NO_BASIS, message="; ".join(problems))]

    if op == "global_extrema":
        return _global_extrema_checks(view, req_id, bits, evidence, src)
    if op == "local_extrema":
        return _local_extrema_checks(view, req_id, bits, params, evidence, src)
    if op == "window_extrema":
        return _window_extrema_checks(view, req_id, bits, params, evidence, src)
    if op == "threshold_crossing":
        return _crossing_checks(view, req_id, bits, params, evidence, src)
    if op == "violation":
        return _violation_checks(view, req_id, bits, params, evidence, src)
    return _state_transition_checks(view, req_id, bits, evidence, src)


def _header(artifact: Artifact) -> tuple[dict, dict, dict, str]:
    manifest = artifact.manifest
    policy_section = manifest.get("policy") if isinstance(manifest.get("policy"), dict) else {}
    canonical = artifact.policy.get("canonical") if isinstance(artifact.policy.get("canonical"), dict) else {}
    policy = {
        "name": policy_section.get("name") or canonical.get("name") or "policy",
        "sha256": policy_section.get("sha256") or artifact.policy.get("sha256"),
    }
    source_section = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    source = {
        "path": source_section.get("path"),
        "format": source_section.get("format"),
        "size_bytes": source_section.get("size_bytes"),
        "digest": source_section.get("digest") or {},
    }
    artifact_section = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    ratio = _number(artifact_section.get("ratio"))
    if ratio is None and isinstance(source.get("size_bytes"), int) and artifact.size:
        ratio = float(source["size_bytes"]) / float(artifact.size)
    max_bytes = artifact_section.get("max_bytes")
    info = {
        "size_bytes": artifact.size,
        "max_bytes": max_bytes if isinstance(max_bytes, int) and not isinstance(max_bytes, bool) else None,
        "ratio": ratio,
    }
    on_not_applicable = (canonical.get("artifact") or {}).get("on_not_applicable")
    return policy, source, info, on_not_applicable if on_not_applicable in ("warn", "fail") else "warn"


def verify_artifact(
    artifact: str | Path | bytes,
    *,
    source: object = None,
    strict: bool = False,
) -> VerifyResult:
    """Verify one `.baslt` artifact, optionally against the source run it was compiled from.

    `artifact` is a path or the file's bytes. `source` is loaded run data -- a `Run`, or a mapping of signal name
    to `(t, v)` -- which turns the attested claims into recomputed ones; a path or a `Run` carrying one also lets
    the recorded digest be rechecked. Opening source files is the caller's job (see the module docstring).
    """
    parsed, checks = structure_checks(artifact)
    result = VerifyResult(checks=checks, strict=strict)
    if parsed is None:
        return result
    result.policy, result.source, result.artifact, result.on_not_applicable = _header(parsed)

    entries = [entry for entry in parsed.index.get("signals", []) if isinstance(entry, dict)]
    views: dict[str, SignalView] = {}
    for entry in entries:
        view, problem = _view(parsed, entry)
        if view is None:
            checks.append(Check(f"signal {entry.get('name')!r}", FAIL, BASIS_ARTIFACT, message=str(problem)))
        else:
            views[view.name] = view

    source_signals: dict[str, SourceSignal] = {}
    if source is not None:
        checks.append(_digest_check(parsed, source))
        source_signals = _source_signals(source)
        if source_signals:
            checks.append(_sample_check(list(views.values()), source_signals))
        else:
            checks.append(
                Check(
                    "source.samples",
                    NOT_APPLICABLE,
                    BASIS_SOURCE,
                    message="no loaded signals were given; pass the loaded run to compare retained samples",
                )
            )

    indexed: dict[str, tuple[dict, dict]] = {}
    for entry in entries:
        for requirement in entry.get("requirements", []) or []:
            if isinstance(requirement, dict) and isinstance(requirement.get("id"), str):
                indexed[requirement["id"]] = (entry, requirement)

    claimed_ids: set[str] = set()
    for claim in parsed.manifest.get("requirements", []) or []:
        if not isinstance(claim, dict) or not isinstance(claim.get("id"), str):
            checks.append(Check("manifest.requirements", FAIL, NO_BASIS, message="a requirement has no string 'id'"))
            continue
        req_id = claim["id"]
        claimed_ids.add(req_id)
        found = indexed.get(req_id)
        if found is None:
            checks.append(Check(req_id, FAIL, NO_BASIS, message="the manifest claims it, but index.json does not"))
            continue
        entry, requirement = found
        view = views.get(str(entry.get("name")))
        src = source_signals.get(str(entry.get("name")))
        checks.extend(_requirement_checks(view, claim, requirement, entry, src))

    for req_id in indexed:
        if req_id not in claimed_ids:
            checks.append(Check(req_id, FAIL, NO_BASIS, message="index.json records it, but the manifest does not"))
    return result
