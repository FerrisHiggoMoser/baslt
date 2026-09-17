"""Checking one run: open the source, bind the requirements, load only what they need, evaluate."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from ..errors import BasltError, Issue
from .bind import bind_requirements, open_run_source
from .check import evaluate_requirement
from .evaluate import EvalError, RunContext
from .expr import ExprError
from .model import RequirementSet
from .results import RunResult
from .units_ext import UnitsError

__all__ = ["check_run", "run_label"]


def run_label(source) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    return "in-memory run"


def _param_names(params: Mapping[str, object] | None) -> set[str] | None:
    return None if params is None else set(params)


def source_parameters(adapter, patterns) -> dict[str, object]:
    """Parameters stored in the run file whose path or leaf name matches `patterns` (globs), under names that
    `param.<name>` can use: the leaf name, or the whole path with `_` where the leaf repeats."""
    import fnmatch
    import re

    reader = getattr(adapter, "parameters", None)
    if not patterns or reader is None:
        return {}
    found = {path: value for path, value in reader().items()
             if any(fnmatch.fnmatchcase(path, p) or fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], p)
                    for p in patterns)}
    leaves = [path.rsplit("/", 1)[-1] for path in found]
    out: dict[str, object] = {}
    for path, leaf in zip(found, leaves):
        name = leaf if leaves.count(leaf) == 1 else path
        out[re.sub(r"\W", "_", name)] = found[path]
    return out


def check_run(source, reqset: RequirementSet, *, params: Mapping[str, object] | None = None,
              run_id: str | None = None, digest: str | None = None) -> tuple[RunResult, RunContext | None]:
    """Every covered requirement of `reqset` on one run. Problems of the run itself are recorded, not raised."""
    config = reqset.config
    started = time.perf_counter()
    label = reqset.table.path.name + (f":{reqset.table.sheet}" if reqset.table.sheet else "")
    result = RunResult(
        run_id=run_id or (Path(source).stem if isinstance(source, (str, Path)) else "run"),
        source=str(source) if isinstance(source, (str, Path)) else run_label(source),
        format="numpy",
        requirements_file=label,
        not_covered=list(reqset.not_covered),
        covered=sum(1 for req in reqset.requirements if req.covered),
        params={k: v for k, v in (params or {}).items()},
        digest=digest,
    )
    try:
        adapter, options = open_run_source(source, config)
        result.format = adapter.format
        if config.params.from_source:
            params = {**source_parameters(adapter, config.params.from_source), **(params or {})}
            result.params = dict(params)
        infos = adapter.list_signals()
        bound = bind_requirements(reqset, infos, param_names=_param_names(params))
        result.issues.extend(bound.issues)
        needed = [name for name in bound.signals]
        missing_clock = [info.name for info in infos if info.name in needed and info.time_ref is None
                         and adapter.format != "numpy"]
        if missing_clock:
            raise BasltError(f"no time signal found for {', '.join(missing_clock)}; set time.signal in the mapping")
        loaded = time.perf_counter()
        run = adapter.load(needed, **options) if needed else adapter.load([], **options)
        result.timing["load"] = time.perf_counter() - loaded
    except BasltError as exc:
        result.error = str(exc)
        result.timing["total"] = time.perf_counter() - started
        return result, None
    result.signals = len(run.signals)
    result.samples = sum(sig.n for sig in run.signals.values())
    spans = [(float(sig.t[0]), float(sig.t[-1])) for sig in run.signals.values() if sig.n]
    if spans:
        result.duration = max(end for _, end in spans) - min(start for start, _ in spans)
    ctx = RunContext(run, config, params=params)
    for name in config.events:
        try:
            event = ctx.event(name)
            result.events[name] = {"count": event.count, "times": [float(v) for v in event.times[:256]],
                                   "pending": event.pending}
        except (EvalError, ExprError, UnitsError, LookupError) as exc:
            result.events[name] = {"count": None, "times": [], "pending": False, "error": str(exc)}
    if config.time.t0:
        t0_event = result.events.get(config.time.t0)
        result.t0_event = config.time.t0
        if t0_event and t0_event.get("times"):
            result.t0 = t0_event["times"][0]
        elif config.time.t0 not in config.events:
            result.issues.append(Issue(path="time.t0", message=f"time.t0 names {config.time.t0!r}, which is not an "
                                                              "event"))
    evaluated = time.perf_counter()
    for item in bound.requirements:
        result.results.append(evaluate_requirement(item, ctx))
    result.timing["evaluate"] = time.perf_counter() - evaluated
    result.timing["total"] = time.perf_counter() - started
    return result, ctx
