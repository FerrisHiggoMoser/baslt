"""Policy data model: operator parameter schemas and the validated and bound dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..units import Quantity


@dataclass(frozen=True)
class Param:
    """One hard-operator parameter. `kind` is "value", "delta", "duration", "enum" or "int"."""

    kind: str
    required: bool = False
    default: object = None
    choices: tuple[str, ...] = ()


OP_SCHEMAS: dict[str, dict[str, Param]] = {
    "global_extrema": {},
    "local_extrema": {
        "prominence": Param("delta", required=True),
        "separation": Param("duration", default="0 s"),
        "kind": Param("enum", default="both", choices=("max", "min", "both")),
    },
    "window_extrema": {
        "interval": Param("duration", required=True),
        "origin": Param("duration", default="0 s"),
    },
    "threshold_crossing": {
        "value": Param("value", required=True),
        "edge": Param("enum", default="both", choices=("rising", "falling", "both")),
        "hysteresis": Param("delta", default=0),
        "debounce": Param("duration", default="0 s"),
        "tolerance": Param("duration", default="0 s"),
        "interpolate": Param("enum", default="linear", choices=("linear", "none")),
    },
    "violation": {
        "above": Param("value"),
        "below": Param("value"),
        "min_duration": Param("duration", default="0 s"),
    },
    "state_transitions": {},
}

_ALL_KINDS: tuple[str, ...] = ("continuous", "discrete", "vector")
_NUMERIC_KINDS: tuple[str, ...] = ("continuous", "vector")

OP_KINDS: dict[str, tuple[str, ...]] = {
    "global_extrema": _ALL_KINDS,
    "local_extrema": _NUMERIC_KINDS,
    "window_extrema": _NUMERIC_KINDS,
    "threshold_crossing": _NUMERIC_KINDS,
    "violation": _NUMERIC_KINDS,
    "state_transitions": ("discrete",),
}

# Allowed trigger signal kinds per event trigger.
EVENT_KINDS: dict[str, tuple[str, ...]] = {
    "falls_below": _ALL_KINDS,
    "rises_above": _ALL_KINDS,
    "equals": ("discrete",),
}

SEVERITIES: tuple[str, ...] = ("info", "limit")
SOFT_WEIGHTS: dict[str, int] = {"high": 4, "medium": 2, "low": 1, "none": 0}
DEFAULT_SOFT_PRIORITY = "medium"
MAX_THUMBNAILS = 4


def default_severity(op: str) -> str:
    return "limit" if op == "violation" else "info"


@dataclass
class ArtifactSpec:
    max_size: str | int | None = None  # as written in the policy
    max_bytes: int | None = None
    codec: str = "deflate"
    hash: str | None = None  # None: full for files, arrays for in-memory input
    soft_value_dtype: str = "source"
    on_not_applicable: str = "warn"


@dataclass
class SignalDecl:
    alias: str
    path: str | None = None  # None: the alias itself names the signal
    unit: str | None = None
    kind: str | None = None
    time: str | None = None


@dataclass
class SignalsSpec:
    time: str | None = None
    time_unit: str = "s"
    on_non_monotonic: str = "error"
    include: list[str] = field(default_factory=lambda: ["*"])
    exclude: list[str] = field(default_factory=list)
    decls: dict[str, SignalDecl] = field(default_factory=dict)


@dataclass
class HardReq:
    id: str
    signal: str  # the signal reference as written in the policy
    op: str
    params: dict[str, object]  # Quantity for value/delta/duration, str for enum
    severity: str


@dataclass
class EventSpec:
    name: str
    signal: str
    trigger: str  # "falls_below" | "rises_above" | "equals"
    value: Quantity | str | int | float  # Quantity unless trigger is "equals"
    hysteresis: Quantity | None = None  # None for equals
    debounce: Quantity | None = None  # None for equals
    occurrence: str = "first"
    expect: int | None = None
    before: Quantity = field(default_factory=lambda: Quantity(0.0, "s", "0 s"))
    after: Quantity = field(default_factory=lambda: Quantity(0.0, "s", "0 s"))
    signals: list[str] | None = None  # None: every included signal
    severity: str = "info"


@dataclass
class TrajectorySpec:
    name: str
    position: str | list[str]
    max_position_error: Quantity
    max_time_error: Quantity | None = None
    linked: list[str] = field(default_factory=list)


@dataclass
class SyncGroupSpec:
    name: str
    members: list[str]


@dataclass
class SoftRule:
    match: str
    priority: str = DEFAULT_SOFT_PRIORITY
    max_points: int | None = None


@dataclass
class ReviewSpec:
    thumbnails: list[str] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)


@dataclass
class Policy:
    version: int
    name: str
    artifact: ArtifactSpec
    signals: SignalsSpec
    hard: list[HardReq]
    events: list[EventSpec]
    trajectories: list[TrajectorySpec]
    sync_groups: list[SyncGroupSpec]
    soft: list[SoftRule]
    review: ReviewSpec
    canonical: dict
    sha256: str
    source_format: str  # "yaml" | "json" | "dict"
    source_text: str | None
    locations: dict[str, str]  # path -> "file:line:col" for YAML input


@dataclass
class BoundReq:
    id: str
    signal: str  # canonical name
    op: str
    params: dict[str, float | str | int]  # value/delta in the signal unit, durations in seconds
    severity: str


@dataclass
class BoundEvent:
    name: str
    signal: str
    trigger: str
    value: float | str | int  # in the trigger signal unit; equals keeps the value as given
    hysteresis: float
    debounce: float
    occurrence: str
    expect: int | None
    before: float
    after: float
    signals: list[str]
    severity: str


@dataclass
class BoundTrajectory:
    name: str
    position: list[str]  # one (n, 3) signal or three scalar signals
    max_position_error: float  # in the position unit
    max_time_error: float | None  # seconds
    linked: list[str]
    unit: str | None


@dataclass
class BoundSyncGroup:
    name: str
    members: list[str]


@dataclass
class BoundPolicy:
    policy: Policy
    included: list[str]
    aliases: dict[str, str]
    reqs: list[BoundReq]
    reqs_by_signal: dict[str, list[BoundReq]]
    events: list[BoundEvent]
    trajectories: list[BoundTrajectory]
    sync_groups: list[BoundSyncGroup]
    soft: dict[str, tuple[int, int | None]]
    max_bytes: int | None
    budget_source: str  # "policy" | "cli" | "none"
    thumbnails: list[str] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)
    units: dict[str, str | None] = field(default_factory=dict)  # effective unit per included signal
    kinds: dict[str, str] = field(default_factory=dict)  # effective kind per included signal
    load_options: dict = field(default_factory=dict)
