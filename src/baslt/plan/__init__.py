"""The planner: evaluate hard requirements and compile a run into a `.baslt` artifact.

`required.evaluate_run` decides which source samples every hard requirement needs and which role bit
records why. `compile.compile_run` turns that plan into container bytes.
"""

from __future__ import annotations

from .compile import CompileOutcome, compile_run
from .required import (
    EXTENT_ROLE,
    GAP_ROLE,
    MAX_ROLES,
    OP_ROLES,
    SUPPORTED_OPS,
    RequiredResult,
    RequirementResult,
    Role,
    SignalPlan,
    evaluate_run,
    legend_of,
)

__all__ = [
    "EXTENT_ROLE",
    "GAP_ROLE",
    "MAX_ROLES",
    "OP_ROLES",
    "SUPPORTED_OPS",
    "CompileOutcome",
    "RequiredResult",
    "RequirementResult",
    "Role",
    "SignalPlan",
    "compile_run",
    "evaluate_run",
    "legend_of",
]
