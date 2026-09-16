"""The `structure.*` checks of docs/verification.md.

Six checks answer "is this file a well-formed artifact, and does it agree with the policy it carries?":

| check | what it proves |
|---|---|
| `structure.container` | `container.read_artifact` accepts the file: ZIP profile, header versions, CRCs, member order. |
| `structure.descriptors` | Every array descriptor fits its member; `nbytes` matches dtype, `n` and `components`. |
| `structure.invariants` | Per signal: `t` finite and non-decreasing, `idx` strictly increasing and `< n_source`, equal array lengths, role bits within the legend, `extent` on the first and last source index. |
| `structure.size` | `manifest.artifact.size_bytes` equals the file length and does not exceed `max_bytes`. |
| `structure.policy` | The embedded canonical policy hashes to its recorded sha256, which the manifest repeats. |
| `structure.bind` | Re-binding the embedded policy's numeric parameters reproduces every bound parameter in `index.json`. |

The container profile is checked by `container.read_artifact`, which is the one reader the format has. Everything
else is the verifier's own work: the descriptor arithmetic, the per-signal invariants and, above all, the
re-binding, which parses the policy's quantities again and converts them with `verify/si_table.py` alone. Importing
the compiler's unit table or its policy binder would make `structure.bind` compare the compiler with itself.

`read_artifact` validates descriptors too, so a bad descriptor makes it raise and `structure.container` fail. The
verifier still runs its own descriptor pass over a best-effort `zipfile` read of the same bytes, so such a file
fails with `structure.descriptors` naming the offending array as well, rather than only with a reader message.

Bitwise, never approximate: a bound parameter one ulp away from the policy's own arithmetic is a failure, because
the compiler used that parameter to decide what to keep.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..container import read_artifact
from ..errors import ContainerError
from .si_table import SI_UNITS, to_si

if TYPE_CHECKING:
    from ..container import Artifact

__all__ = [
    "BASIS_ARTIFACT",
    "BASIS_ATTESTED",
    "BASIS_SOURCE",
    "FAIL",
    "NOT_APPLICABLE",
    "NO_BASIS",
    "PASS",
    "WARN",
    "Check",
    "bind_problems",
    "canonical_sha256",
    "descriptor_problems",
    "invariant_problems",
    "read_bytes",
    "structure_checks",
]

# --- statuses and bases (docs/contracts.md "Statuses", docs/verification.md "Bases") ----------------------------

PASS = "pass"
WARN = "warn"
NOT_APPLICABLE = "not_applicable"
FAIL = "fail"
STATUSES: tuple[str, ...] = (PASS, WARN, NOT_APPLICABLE, FAIL)

BASIS_ARTIFACT = "artifact"
BASIS_ATTESTED = "attested"
BASIS_SOURCE = "source"
NO_BASIS = "-"


@dataclass(frozen=True, slots=True)
class Check:
    """One row of the verification result.

    `claimed` is what the artifact says (it came from the source run), `measured` is what the verifier found, and
    `allowed` is the tolerance the comparison was given. All three are rendered text or None for "not applicable
    to this check"; the table prints None as a dash.
    """

    id: str
    status: str
    basis: str = BASIS_ARTIFACT
    claimed: str | None = None
    measured: str | None = None
    allowed: str | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (PASS, WARN)


def verdict(
    check_id: str,
    problems: list[str],
    *,
    basis: str = BASIS_ARTIFACT,
    claimed: str | None = None,
    measured: str | None = None,
    allowed: str | None = None,
    message: str = "",
    limit: int = 3,
) -> Check:
    """A passing check, or one failing check carrying the first `limit` problems."""
    if problems:
        shown = "; ".join(problems[:limit])
        if len(problems) > limit:
            shown += f" (and {len(problems) - limit} more)"
        return Check(check_id, FAIL, basis, claimed, measured, allowed, shown)
    return Check(check_id, PASS, basis, claimed, measured, allowed, message)


# --- container profile constants, transcribed from docs/container.md -------------------------------------------

MEMBER_HEADER = "baslt.json"
MEMBER_INDEX = "index.json"
MEMBER_MANIFEST = "manifest.json"
MEMBER_POLICY = "policy.json"
JSON_MEMBERS: tuple[str, ...] = (MEMBER_HEADER, MEMBER_INDEX, MEMBER_MANIFEST, MEMBER_POLICY)
DATA_PREFIXES: tuple[str, ...] = ("s/", "t/")

DTYPE_WIDTHS: dict[str, int] = {
    "<f8": 8, "<f4": 4, "<i8": 8, "<i4": 4, "<i2": 2, "|i1": 1, "<u8": 8, "<u4": 4, "<u2": 2, "|u1": 1,
}  # fmt: skip
ENCODINGS: tuple[str, ...] = ("raw", "shuffle", "delta+shuffle")
DELTA_DTYPES: tuple[str, ...] = ("<u4", "<f8")
ARRAY_NAMES: tuple[str, ...] = ("t", "v", "idx", "roles")
DESCRIPTOR_KEYS: tuple[str, ...] = ("name", "member", "offset", "nbytes", "n", "components", "dtype", "enc")
SIZE_DIGITS = 20  # manifest.artifact.size_bytes is zero-padded so the manifest's length is size-independent
MASK64 = (1 << 64) - 1

# Hard-operator parameter kinds, transcribed from the table in docs/policy.md.
OP_PARAMS: dict[str, dict[str, str]] = {
    "global_extrema": {},
    "local_extrema": {"prominence": "delta", "separation": "duration", "kind": "enum"},
    "window_extrema": {"interval": "duration", "origin": "duration"},
    "threshold_crossing": {
        "value": "value",
        "edge": "enum",
        "hysteresis": "delta",
        "debounce": "duration",
        "tolerance": "duration",
        "interpolate": "enum",
    },
    "violation": {"above": "value", "below": "value", "min_duration": "duration"},
    "state_transitions": {},
}

# `<number> <unit>` or `<number><unit>`, as docs/policy.md writes quantities.
_QUANTITY_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S*)")
_REQUIREMENT_ID_RE = re.compile(r"hard\.(?P<rest>.+?)(?:\[(?P<index>\d+)\])?$")


# --- reading ---------------------------------------------------------------------------------------------------


def read_bytes(source: str | Path | bytes | bytearray | memoryview) -> bytes:
    """The artifact's bytes, from a path or from memory. Raises ContainerError when the file cannot be read."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    path = Path(source)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ContainerError(f"cannot read {path}: {exc.strerror or exc}") from exc


def _loose_members(data: bytes) -> dict[str, bytes] | None:
    """Members read with plain `zipfile`, ignoring the profile, or None when even that fails.

    Used only to attribute a `read_artifact` failure: it lets the descriptor pass run on a file the strict reader
    rejected, so the result names the bad array instead of only quoting the reader.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    except (zipfile.BadZipFile, zlib.error, OSError, ValueError, EOFError, RuntimeError, NotImplementedError):
        return None


def _json_object(raw: bytes | None) -> dict | None:
    if raw is None:
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _member_order_problems(artifact: Artifact) -> list[str]:
    """docs/container.md fixes the member order: the four JSON members, then the data members."""
    names = [entry.name for entry in artifact.entries]
    problems: list[str] = []
    head = names[: len(JSON_MEMBERS)]
    if head != list(JSON_MEMBERS):
        problems.append(f"members start with {head}, expected {list(JSON_MEMBERS)}")
    for name in names[len(JSON_MEMBERS) :]:
        if not name.startswith(DATA_PREFIXES):
            problems.append(f"member {name!r} is neither a JSON member nor an s/ or t/ data member")
    return problems


# --- structure.descriptors -------------------------------------------------------------------------------------


def descriptor_problems(index: dict, member_sizes: dict[str, int]) -> list[str]:
    """Every descriptor must fit its member and its `nbytes` must match dtype, `n` and `components`."""
    problems: list[str] = []
    signals = index.get("signals")
    if not isinstance(signals, list):
        return ["index.json has no 'signals' list"]
    for position, entry in enumerate(signals):
        if not isinstance(entry, dict):
            problems.append(f"index.json signals[{position}] is not an object")
            continue
        name = entry.get("name", f"[{position}]")
        arrays = entry.get("arrays")
        if not isinstance(arrays, list):
            problems.append(f"signal {name!r} has no 'arrays' list")
            continue
        for descriptor in arrays:
            problems.extend(_descriptor_problems(descriptor, member_sizes, name))
    return problems


def _descriptor_problems(descriptor: object, member_sizes: dict[str, int], signal: str) -> list[str]:
    if not isinstance(descriptor, dict):
        return [f"signal {signal!r}: an array descriptor is not an object"]
    missing = [key for key in DESCRIPTOR_KEYS if key not in descriptor]
    if missing:
        return [f"signal {signal!r}: an array descriptor is missing {', '.join(missing)}"]
    where = f"signal {signal!r} array {descriptor['name']!r}"
    for key in ("name", "member", "dtype", "enc"):
        if not isinstance(descriptor[key], str):
            return [f"{where}: field {key!r} must be a string"]
    for key in ("offset", "nbytes", "n", "components"):
        value = descriptor[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return [f"{where}: field {key!r} must be a non-negative integer, got {value!r}"]

    problems: list[str] = []
    dtype, enc = descriptor["dtype"], descriptor["enc"]
    width = DTYPE_WIDTHS.get(dtype)
    if width is None:
        problems.append(f"{where}: unsupported dtype {dtype!r}")
    if enc not in ENCODINGS:
        problems.append(f"{where}: unsupported encoding {enc!r}")
    if descriptor["components"] < 1:
        problems.append(f"{where}: components must be at least 1")
    if enc == "delta+shuffle":
        if dtype not in DELTA_DTYPES:
            problems.append(f"{where}: delta+shuffle needs dtype <u4 or <f8, got {dtype!r}")
        if descriptor["components"] != 1:
            problems.append(f"{where}: delta+shuffle arrays must have exactly one component")
    if width is not None:
        expected = descriptor["n"] * descriptor["components"] * width
        if descriptor["nbytes"] != expected:
            problems.append(
                f"{where}: nbytes {descriptor['nbytes']} is not n*components*width = "
                f"{descriptor['n']}*{descriptor['components']}*{width} = {expected}"
            )

    member = descriptor["member"]
    if member in JSON_MEMBERS or member not in member_sizes:
        problems.append(f"{where}: member {member!r} is not a data member of this file")
    else:
        end = descriptor["offset"] + descriptor["nbytes"]
        if end > member_sizes[member]:
            problems.append(
                f"{where}: bytes {descriptor['offset']}..{end} lie outside member {member!r} "
                f"({member_sizes[member]} bytes)"
            )
    return problems


# --- structure.invariants --------------------------------------------------------------------------------------


def invariant_problems(artifact: Artifact) -> list[str]:
    """Per-signal invariants of docs/verification.md, for every signal in `index.json`."""
    problems: list[str] = []
    for entry in artifact.index.get("signals", []):
        if isinstance(entry, dict):
            problems.extend(_signal_invariants(artifact, entry))
    return problems


def _signal_invariants(artifact: Artifact, entry: dict) -> list[str]:
    name = entry.get("name")
    where = f"signal {name!r}"
    descriptors = {d.get("name"): d for d in entry.get("arrays", []) if isinstance(d, dict)}
    missing = [a for a in ARRAY_NAMES if a not in descriptors]
    if missing:
        return [f"{where}: no {', '.join(missing)} array"]

    n, n_source, components = entry.get("n"), entry.get("n_source"), entry.get("components", 1)
    for label, value in (("n", n), ("n_source", n_source), ("components", components)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return [f"{where}: {label} must be a non-negative integer, got {value!r}"]

    problems: list[str] = []
    if n > n_source:
        problems.append(f"{where}: keeps {n} samples of a source of {n_source}")
    for array_name, descriptor in descriptors.items():
        if descriptor.get("n") != n:
            problems.append(f"{where} array {array_name!r}: n is {descriptor.get('n')!r}, expected {n}")
        wanted = components if array_name == "v" else 1
        if descriptor.get("components") != wanted:
            problems.append(
                f"{where} array {array_name!r}: components is {descriptor.get('components')!r}, expected {wanted}"
            )
    if problems:
        return problems  # decoding arrays of inconsistent lengths would only repeat the same cause

    try:
        t = artifact.array(name, "t")
        values = artifact.array(name, "v")
        idx = artifact.array(name, "idx")
        roles = artifact.array(name, "roles")
    except ContainerError as exc:
        return [f"{where}: {exc}"]

    if n == 0:
        return problems
    problems.extend(_time_problems(t, where))
    problems.extend(_index_problems(idx, n_source, where))
    legend, legend_problems = _legend(entry, where)
    problems.extend(legend_problems)
    if legend is not None:
        problems.extend(_role_problems(roles, legend, where))
        problems.extend(_extent_problems(idx, roles, legend, n_source, where))
    if values.shape[0] != n:
        problems.append(f"{where}: the v array decodes to {values.shape[0]} samples, expected {n}")
    return problems


def _time_problems(t: np.ndarray, where: str) -> list[str]:
    problems: list[str] = []
    if not bool(np.isfinite(t).all()):
        first = int(np.flatnonzero(~np.isfinite(t))[0])
        problems.append(f"{where}: retained timestamp {first} is {t[first]!r}, which is not finite")
    elif t.shape[0] > 1 and not bool((np.diff(t) >= 0).all()):
        first = int(np.flatnonzero(np.diff(t) < 0)[0])
        problems.append(f"{where}: time decreases at retained sample {first + 1} ({t[first]!r} -> {t[first + 1]!r})")
    return problems


def _as_source_indices(idx: np.ndarray, where: str) -> tuple[np.ndarray | None, list[str]]:
    """`idx` as int64. A `delta+shuffle` array decodes to float64 holding exact integers."""
    if idx.dtype.kind == "f":
        if not bool(np.isfinite(idx).all()) or not bool((np.floor(idx) == idx).all()):
            return None, [f"{where}: the idx array holds values that are not exact integers"]
    return idx.astype(np.int64), []


def _index_problems(idx: np.ndarray, n_source: int, where: str) -> list[str]:
    source_index, problems = _as_source_indices(idx, where)
    if source_index is None:
        return problems
    if source_index.shape[0] > 1 and not bool((np.diff(source_index) > 0).all()):
        first = int(np.flatnonzero(np.diff(source_index) <= 0)[0])
        problems.append(
            f"{where}: idx is not strictly increasing at retained sample {first + 1} "
            f"({int(source_index[first])} -> {int(source_index[first + 1])})"
        )
    if int(source_index[0]) < 0:
        problems.append(f"{where}: idx starts at {int(source_index[0])}, which is negative")
    if int(source_index[-1]) >= n_source:
        problems.append(f"{where}: idx reaches {int(source_index[-1])}, but the source has {n_source} samples")
    return problems


def _legend(entry: dict, where: str) -> tuple[dict[int, str] | None, list[str]]:
    raw = entry.get("roles")
    if not isinstance(raw, list):
        return None, [f"{where}: no role legend"]
    legend: dict[int, str] = {}
    problems: list[str] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("bit"), int) or isinstance(item.get("bit"), bool):
            problems.append(f"{where}: a role legend entry has no integer 'bit'")
            continue
        bit, role_id = item["bit"], item.get("id")
        if not 0 <= bit < 64:
            problems.append(f"{where}: role bit {bit} is outside 0..63")
            continue
        if bit in legend:
            problems.append(f"{where}: role bit {bit} is used by {legend[bit]!r} and {role_id!r}")
            continue
        legend[bit] = role_id if isinstance(role_id, str) else ""
    return (None, problems) if problems else (legend, problems)


def _role_problems(roles: np.ndarray, legend: dict[int, str], where: str) -> list[str]:
    mask = 0
    for bit in legend:
        mask |= 1 << bit
    outside = np.uint64((~mask) & MASK64)
    marked = roles.astype(np.uint64)
    bad = np.flatnonzero(np.bitwise_and(marked, outside) != np.uint64(0))
    if bad.size:
        position = int(bad[0])
        return [
            f"{where}: retained sample {position} carries role bits {int(marked[position]):#x} outside the "
            f"legend ({len(legend)} entries)"
        ]
    return []


def _extent_problems(
    idx: np.ndarray, roles: np.ndarray, legend: dict[int, str], n_source: int, where: str
) -> list[str]:
    source_index, problems = _as_source_indices(idx, where)
    if source_index is None or n_source == 0:
        return problems
    first, last = int(source_index[0]), int(source_index[-1])
    if first != 0 or last != n_source - 1:
        problems.append(
            f"{where}: extent retention needs source samples 0 and {n_source - 1}, but idx runs {first}..{last}"
        )
    bits = [bit for bit, role_id in legend.items() if role_id == "extent"]
    if not bits:
        problems.append(f"{where}: the role legend has no 'extent' entry")
        return problems
    mask = np.uint64(1) << np.uint64(bits[0])
    flagged = set(source_index[np.bitwise_and(roles.astype(np.uint64), mask) != np.uint64(0)].tolist())
    wanted = {0, n_source - 1}
    if flagged != wanted:
        problems.append(f"{where}: the extent role marks source samples {sorted(flagged)}, expected {sorted(wanted)}")
    return problems


# --- structure.size --------------------------------------------------------------------------------------------


def size_check(artifact: Artifact) -> Check:
    section = artifact.manifest.get("artifact")
    if not isinstance(section, dict):
        return Check("structure.size", FAIL, BASIS_ARTIFACT, message="manifest.json has no 'artifact' section")
    raw = section.get("size_bytes")
    problems: list[str] = []
    claimed: int | None = None
    if isinstance(raw, str):
        if not (len(raw) == SIZE_DIGITS and raw.isdigit()):
            problems.append(f"manifest.artifact.size_bytes {raw!r} is not a {SIZE_DIGITS}-digit zero-padded string")
        if raw.isdigit():
            claimed = int(raw)
    elif isinstance(raw, int) and not isinstance(raw, bool):
        claimed = raw
    else:
        problems.append(f"manifest.artifact.size_bytes is {raw!r}, expected a {SIZE_DIGITS}-digit string")

    if claimed is not None and claimed != artifact.size:
        problems.append(f"manifest.artifact.size_bytes is {claimed}, but the file is {artifact.size} bytes")
    max_bytes = section.get("max_bytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        max_bytes = None
    if max_bytes is not None and artifact.size > max_bytes:
        problems.append(f"the file is {artifact.size} bytes, over the {max_bytes}-byte budget")
    return verdict(
        "structure.size",
        problems,
        claimed=None if claimed is None else str(claimed),
        measured=str(artifact.size),
        allowed=None if max_bytes is None else str(max_bytes),
        message="size_bytes equals the file length",
    )


# --- structure.policy ------------------------------------------------------------------------------------------


def canonical_json(obj: object) -> str:
    """The canonical form of docs/policy.md: sorted keys, compact separators, UTF-8 unescaped, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_sha256(obj: object) -> str:
    """SHA-256 of `canonical_json(obj)` encoded as UTF-8."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def policy_check(artifact: Artifact) -> Check:
    policy = artifact.policy
    canonical = policy.get("canonical")
    if not isinstance(canonical, dict):
        return Check("structure.policy", FAIL, BASIS_ARTIFACT, message="policy.json has no canonical policy")
    recorded = policy.get("sha256")
    try:
        recomputed = canonical_sha256(canonical)
    except (TypeError, ValueError) as exc:
        return Check("structure.policy", FAIL, BASIS_ARTIFACT, message=f"the canonical policy is not JSON: {exc}")

    problems: list[str] = []
    if not isinstance(recorded, str) or not recorded:
        problems.append("policy.json has no sha256")
    elif recorded != recomputed:
        problems.append(f"the canonical policy hashes to {recomputed}, but policy.json records {recorded}")
    manifest_sha = artifact.manifest.get("policy", {}).get("sha256") if isinstance(artifact.manifest, dict) else None
    if isinstance(manifest_sha, str) and manifest_sha and manifest_sha != recomputed:
        problems.append(f"manifest.policy.sha256 is {manifest_sha}, but the embedded policy hashes to {recomputed}")
    return verdict(
        "structure.policy",
        problems,
        claimed=_short_hash(recorded),
        measured=_short_hash(recomputed),
        message="the embedded policy hashes to its recorded sha256",
    )


def _short_hash(value: object) -> str | None:
    return f"{value[:8]}…" if isinstance(value, str) and value else None


# --- structure.bind --------------------------------------------------------------------------------------------


def parse_quantity(text: object) -> tuple[float, str | None] | None:
    """`(value, unit)` of a policy quantity, or None when the text is not one."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text), None
    if not isinstance(text, str):
        return None
    match = _QUANTITY_RE.fullmatch(text.strip())
    if match is None:
        return None
    return float(match.group(1)), (match.group(2) or None)


def rebind(kind: str, written: object, unit: str | None) -> tuple[float | str | None, str | None]:
    """Re-bind one policy parameter. Returns `(bound value, problem)`.

    `value` and `delta` end in the signal's own unit, `duration` in seconds, `enum` as its string. The arithmetic
    is the one docs/policy.md states -- SI = number * factor + offset, deltas without the offset -- evaluated in
    that order with `si_table` factors, so the result matches the compiler's bit for bit or not at all.
    """
    if kind == "enum":
        return (written, None) if isinstance(written, str) else (None, f"{written!r} is not an enum value")
    parsed = parse_quantity(written)
    if parsed is None:
        return None, f"{written!r} is not a quantity"
    value, symbol = parsed
    if kind == "duration":
        if symbol is None:
            return value, None
        entry = SI_UNITS.get(symbol)
        if entry is None:
            return None, f"unknown unit {symbol!r}"
        if entry[0] != "time":
            return None, f"{written!r} is a {entry[0]}, but a duration is required"
        return to_si(value, symbol, delta=True), None

    if symbol is None or symbol == unit:
        return value, None  # a bare number is already in the signal's unit
    source = SI_UNITS.get(symbol)
    target = SI_UNITS.get(unit) if unit is not None else None
    if source is None:
        return None, f"unknown unit {symbol!r}"
    if unit is None:
        return None, f"{written!r} has a unit but the signal has none"
    if target is None:
        return None, f"{written!r} does not match the opaque signal unit {unit!r}"
    if source[0] != target[0]:
        return None, f"{written!r} is a {source[0]} but the signal unit {unit} is a {target[0]}"
    if kind == "delta":
        return to_si(value, symbol, delta=True) / target[1], None
    return (to_si(value, symbol, delta=False) - target[2]) / target[1], None


def _same_number(claimed: object, bound: float) -> bool:
    """Bitwise equality of a bound parameter: one ulp apart is a different parameter."""
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)):
        return False
    return struct.pack("<d", float(claimed)) == struct.pack("<d", float(bound))


def _policy_spec(hard: dict, requirement_id: str, op: str) -> tuple[dict | None, str | None]:
    """The policy spec a requirement id points at. Ids are the YAML path: `hard.<signal>.<op>` or `...[k]`."""
    match = _REQUIREMENT_ID_RE.fullmatch(requirement_id)
    if match is None:
        return None, f"{requirement_id!r} is not a hard requirement id"
    reference, _, op_in_id = match.group("rest").rpartition(".")
    if not reference:
        return None, f"{requirement_id!r} does not name a signal"
    if op_in_id != op:
        return None, f"{requirement_id!r} names operator {op_in_id!r}, but index.json says {op!r}"
    ops = hard.get(reference)
    if not isinstance(ops, dict) or op not in ops:
        return None, f"the embedded policy has no {op} for signal {reference!r}"
    spec = ops[op]
    position = match.group("index")
    if position is None:
        if not isinstance(spec, dict):
            return None, f"the embedded policy writes {requirement_id!r} as a list of specs"
        return spec, None
    if not isinstance(spec, list) or int(position) >= len(spec):
        return None, f"the embedded policy has no {op}[{position}] for signal {reference!r}"
    item = spec[int(position)]
    if not isinstance(item, dict):
        return None, f"{requirement_id!r} is not a mapping in the embedded policy"
    return item, None


def bind_problems(artifact: Artifact) -> tuple[list[str], int]:
    """Re-bind every parameter `index.json` records. Returns `(problems, parameters compared)`."""
    canonical = artifact.policy.get("canonical")
    if not isinstance(canonical, dict):
        return ["policy.json has no canonical policy to re-bind"], 0
    hard = canonical.get("hard")
    if not isinstance(hard, dict):
        hard = {}
    problems: list[str] = []
    compared = 0
    for entry in artifact.index.get("signals", []):
        if not isinstance(entry, dict):
            continue
        unit = entry.get("unit")
        unit = unit if isinstance(unit, str) else None
        for requirement in entry.get("requirements", []) or []:
            if not isinstance(requirement, dict):
                problems.append(f"signal {entry.get('name')!r}: a requirement entry is not an object")
                continue
            count, found = _requirement_bind_problems(requirement, hard, unit)
            problems.extend(found)
            compared += count
    problems.extend(_budget_problems(artifact, canonical))
    return problems, compared


def _requirement_bind_problems(requirement: dict, hard: dict, unit: str | None) -> tuple[int, list[str]]:
    requirement_id = requirement.get("id")
    op = requirement.get("op")
    if not isinstance(requirement_id, str) or not isinstance(op, str):
        return 0, ["a requirement in index.json has no string 'id' and 'op'"]
    schema = OP_PARAMS.get(op)
    if schema is None:
        return 0, [f"{requirement_id}: unknown operator {op!r}"]
    spec, problem = _policy_spec(hard, requirement_id, op)
    if spec is None:
        return 0, [f"{requirement_id}: {problem}"]

    compared = 0
    problems: list[str] = []
    for parameter in requirement.get("params", []) or []:
        if not isinstance(parameter, dict) or not isinstance(parameter.get("name"), str):
            problems.append(f"{requirement_id}: a parameter entry has no string 'name'")
            continue
        name, claimed = parameter["name"], parameter.get("value")
        if name == "severity":
            compared += 1
            if claimed != spec.get("severity"):
                problems.append(f"{requirement_id}.severity: index.json says {claimed!r}, the policy {spec.get('severity')!r}")
            continue
        kind = schema.get(name)
        if kind is None:
            problems.append(f"{requirement_id}: {op} has no parameter {name!r}")
            continue
        if name not in spec:
            problems.append(f"{requirement_id}: index.json binds {name!r}, which the embedded policy does not set")
            continue
        bound, problem = rebind(kind, spec[name], unit)
        compared += 1
        if problem is not None:
            problems.append(f"{requirement_id}.{name}: {problem}")
        elif kind == "enum":
            if claimed != bound:
                problems.append(f"{requirement_id}.{name}: index.json says {claimed!r}, the policy {bound!r}")
        elif not _same_number(claimed, bound):  # type: ignore[arg-type]
            problems.append(
                f"{requirement_id}.{name}: index.json says {claimed!r}, but {spec[name]!r} binds to {bound!r}"
            )
    return compared, problems


def parse_bytes(written: object) -> tuple[int | None, str | None]:
    """A byte count from a plain integer or a byte quantity such as '2 MiB'."""
    if isinstance(written, bool):
        return None, "a byte size must not be a boolean"
    if isinstance(written, int):
        return written, None
    parsed = parse_quantity(written)
    if parsed is None:
        return None, f"{written!r} is not a byte size"
    _value, symbol = parsed
    factor = 1.0
    if symbol is not None:
        entry = SI_UNITS.get(symbol)
        if entry is None or entry[0] != "bytes":
            return None, f"{written!r} is not a byte size"
        factor = entry[1]
    digits = _QUANTITY_RE.fullmatch(str(written).strip()).group(1)  # type: ignore[union-attr]
    exact = Fraction(digits) * Fraction(factor)
    if exact.denominator != 1:
        return None, f"{written!r} is not a whole number of bytes"
    return int(exact), None


def _budget_problems(artifact: Artifact, canonical: dict) -> list[str]:
    """`artifact.max_size` binds to `manifest.artifact.max_bytes` when the budget came from the policy."""
    manifest = artifact.manifest
    budget = manifest.get("budget") if isinstance(manifest.get("budget"), dict) else {}
    if budget.get("source") != "policy":
        return []
    written = (canonical.get("artifact") or {}).get("max_size")
    if written is None:
        return []
    wanted, problem = parse_bytes(written)
    if problem is not None:
        return [f"artifact.max_size: {problem}"]
    recorded = (manifest.get("artifact") or {}).get("max_bytes")
    if isinstance(recorded, bool) or not isinstance(recorded, int):
        return [f"artifact.max_size: the policy sets {written!r} but the manifest records no max_bytes"]
    if recorded != wanted:
        return [f"artifact.max_size: {written!r} is {wanted} bytes, but the manifest records {recorded}"]
    return []


# --- the whole structure pass ----------------------------------------------------------------------------------


def structure_checks(source: str | Path | bytes) -> tuple[Artifact | None, list[Check]]:
    """Run every `structure.*` check. Returns the artifact (None when it could not be read) and the checks."""
    data = read_bytes(source)
    artifact: Artifact | None = None
    error: str | None = None
    try:
        artifact = read_artifact(data)
    except ContainerError as exc:
        error = str(exc)

    if artifact is not None:
        index: dict | None = artifact.index
        sizes: dict[str, int] | None = {name: len(buf) for name, buf in artifact.members.items()}
        problems = _member_order_problems(artifact)
    else:
        members = _loose_members(data)
        index = None if members is None else _json_object(members.get(MEMBER_INDEX))
        sizes = None if members is None else {name: len(buf) for name, buf in members.items()}
        problems = [error or "the artifact could not be read"]

    checks = [
        verdict(
            "structure.container",
            problems,
            message="profile v1: header versions, CRCs, member order",
            limit=1,
        )
    ]
    if index is None or sizes is None:
        checks.append(
            Check("structure.descriptors", NOT_APPLICABLE, BASIS_ARTIFACT, message="index.json could not be read")
        )
    else:
        checks.append(
            verdict(
                "structure.descriptors",
                descriptor_problems(index, sizes),
                measured=str(sum(len(s.get("arrays", [])) for s in index.get("signals", []) if isinstance(s, dict))),
                message="every array descriptor fits its member",
            )
        )

    if artifact is None:
        for name in ("invariants", "size", "policy", "bind"):
            checks.append(
                Check(f"structure.{name}", NOT_APPLICABLE, BASIS_ARTIFACT, message="the artifact could not be read")
            )
        return None, checks

    checks.append(
        verdict(
            "structure.invariants",
            invariant_problems(artifact),
            measured=str(len(artifact.index.get("signals", []))),
            message="timestamps, indices, lengths, role bits and extent hold for every signal",
        )
    )
    checks.append(size_check(artifact))
    checks.append(policy_check(artifact))
    bind_found, compared = bind_problems(artifact)
    checks.append(
        verdict(
            "structure.bind",
            bind_found,
            measured=str(compared),
            allowed="0",
            message=f"{compared} bound parameter(s) reproduced from the embedded policy",
        )
    )
    return artifact, checks
