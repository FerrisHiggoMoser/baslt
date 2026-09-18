"""Checking many runs: finding them, their parameters, a process pool, resuming, and the combined outputs.

Each run is checked in a worker process that writes `runs/<id>.json` (and `runs/<id>.html` for the runs the page
setting asks for) and hands back a compact summary. The parent appends every finished run to `index.jsonl`, so an
interrupted batch keeps what it did; `resume` skips runs whose stored result was made from the same run file,
requirements, mapping, parameters and baslt version. A run that crashes its worker is retried alone and, if it
crashes again, reported as ERROR.

The combined outputs are `summary.json`, `results.xlsx` (verdict matrix, margins, per-requirement statistics,
violations, runs), the checked copy of the requirements table with one sentence per requirement, and `index.html`.
Worker envelopes (256 bins of each plotted signal on the run's own time axis, relative to its time zero) are merged
in the parent onto one axis for the dashboard's overlays.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import json
import math
import os
import re
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._io import write_atomic
from .._version import __version__
from ..errors import BasltError, SourceError, UsageError

__all__ = ["BatchRun", "check_batch", "discover", "run_ids"]

RUN_SUFFIXES = (".h5", ".hdf5", ".mat", ".csv")
ENVELOPE_BINS = 256
NAN_BYTE = 255
VIOLATION_ROWS = 100_000
VIOLATIONS_PER_REQUIREMENT = 20


# ------------------------------------------------------------------------------------------------------------
# finding runs


def _has_magic(text: str) -> bool:
    return any(ch in text for ch in "*?[")


def discover(runs: str | Path | Sequence[str | Path], *, skip: Path | None = None) -> tuple[list[Path], Path]:
    """Run files from files, folders (searched recursively) and glob patterns, and the folder they share."""
    items = [runs] if isinstance(runs, (str, Path)) else list(runs)
    found: list[Path] = []
    seen: set[Path] = set()
    skip_dir = skip.resolve() if skip is not None else None

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if skip_dir is not None and (resolved == skip_dir or skip_dir in resolved.parents):
            return
        seen.add(resolved)
        found.append(path)

    for item in items:
        text = str(item)
        if _has_magic(text):
            matches = sorted(Path(m) for m in glob.glob(text, recursive=True))
            files = [m for m in matches if m.is_file()]
            if not files:
                raise UsageError(f"no run files match {text!r}")
            for match in files:
                add(match)
            continue
        path = Path(item)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in RUN_SUFFIXES
                           and not any(part.startswith(".") for part in p.relative_to(path).parts))
            if not files:
                raise UsageError(f"no run files ({', '.join(RUN_SUFFIXES)}) in folder {path}")
            for file in files:
                add(file)
        elif path.is_file():
            add(path)
        else:
            raise SourceError(f"run file not found: {path}")
    if not found:
        raise UsageError("no runs to check")
    parents = [str(p.resolve().parent) for p in found]
    return found, Path(os.path.commonpath(parents))


def run_ids(paths: Sequence[Path], root: Path) -> list[str]:
    """File stems as run ids; where stems repeat, the path below `root` instead. Ids are safe file names."""
    stems = [p.stem for p in paths]
    repeated = {s for s in stems if stems.count(s) > 1}
    out: list[str] = []
    used: set[str] = set()
    for path, stem in zip(paths, stems):
        name = stem
        if stem in repeated:
            try:
                name = path.resolve().relative_to(root).with_suffix("").as_posix()
            except ValueError:
                name = path.with_suffix("").as_posix()
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.replace("/", "__")).strip("._") or "run"
        unique, k = name, 2
        while unique.lower() in used:
            unique, k = f"{name}-{k}", k + 1
        used.add(unique.lower())
        out.append(unique)
    return out


# ------------------------------------------------------------------------------------------------------------
# envelopes


def _quantize(lo: np.ndarray, hi: np.ndarray) -> dict:
    both = np.concatenate([lo, hi])
    finite = np.isfinite(both)
    vmin = float(both[finite].min()) if finite.any() else 0.0
    vmax = float(both[finite].max()) if finite.any() else 0.0
    span = vmax - vmin
    codes = np.full(both.shape[0], NAN_BYTE, dtype=np.uint8)
    if span > 0:
        codes[finite] = np.rint((both[finite] - vmin) / span * 254).astype(np.uint8)
    else:
        codes[finite] = 0
    return {"min": vmin, "max": vmax, "q": base64.b64encode(codes.tobytes()).decode("ascii")}


def _dequantize(env: dict) -> tuple[np.ndarray, np.ndarray]:
    codes = np.frombuffer(base64.b64decode(env["q"]), dtype=np.uint8)
    values = env["min"] + codes.astype(np.float64) * ((env["max"] - env["min"]) / 254)
    values[codes == NAN_BYTE] = np.nan
    half = codes.shape[0] // 2
    return values[:half], values[half:]


def _bins(t: np.ndarray, values: np.ndarray, a: float, b: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin minimum and maximum of `values` over `bins` equal time bins of [a, b]; NaN for empty bins."""
    edges = a + (b - a) * np.arange(bins + 1) / bins
    bounds = np.searchsorted(t, edges, side="left")
    bounds[-1] = t.shape[0]  # the last bin holds the sample at b
    lo = np.full(bins, np.nan)
    hi = np.full(bins, np.nan)
    filled = np.diff(bounds) > 0
    if filled.any():
        # consecutive filled bins touch, so each reduceat segment is exactly one bin's samples
        starts = bounds[:-1][filled]
        lo[filled] = np.fmin.reduceat(values, starts)
        hi[filled] = np.fmax.reduceat(values, starts)
    return lo, hi


def envelopes(run, ctx, bins: int = ENVELOPE_BINS) -> tuple[dict, dict]:
    """(series, plots): binned signals of the run's plots keyed by expression, and each requirement's reference."""
    from .plotdata import PLOT_ERRORS, _describe

    series: dict[str, dict] = {}
    plots: dict[str, dict] = {}
    if ctx is None:
        return series, plots
    base = float(run.t0) if run.t0 is not None else 0.0
    for result in run.results:
        trace = result.trace
        if trace is None or trace.subject is None or trace.grid is None or trace.grid.n == 0:
            continue
        subject = trace.subject
        component = int(result.totals.get("component", 0)) if trace.x is not None else 0
        key = f"{subject.key}#{component}"
        t = trace.grid.t - base
        a, b = float(t[0]), float(t[-1])
        if key not in series:
            try:
                values = trace.x if trace.x is not None else np.asarray(ctx.series(subject, trace.grid), dtype=float)
            except PLOT_ERRORS:  # a plot is optional; the verdict stands without it
                continue
            values = np.asarray(values, dtype=np.float64)
            if values.ndim == 2:
                values = values[:, min(component, values.shape[1] - 1)]
            if not (b > a):
                continue
            lo, hi = _bins(t, values, a, b, bins)
            labels = None
            if subject.type.dtype == "enum":
                try:
                    found = ctx.labels_of(subject)
                except PLOT_ERRORS:
                    found = None
                if isinstance(found, Mapping):
                    labels = [[int(k), str(v)] for k, v in sorted(found.items())]
                elif found:
                    labels = [[k, str(v)] for k, v in enumerate(found)]
            unit = result.unit if trace.x is not None else subject.type.unit
            series[key] = {"name": _describe(subject), "unit": unit if unit not in ("1", "?") else None,
                           "labels": labels, "a": a, "b": b, **_quantize(lo, hi)}
        entry: dict = {"s": key}
        for side, values in (("upper", trace.upper), ("lower", trace.lower)):
            if values is None:
                continue
            finite = np.isfinite(values)
            if not finite.any():
                continue
            picked = values[finite]
            if np.all(picked == picked[0]):
                entry[side] = float(picked[0])
            else:
                lo, hi = _bins(t, values, a, b, bins)
                entry[side] = {"a": a, "b": b, **_quantize(lo, hi)}
        for kind, value in trace.lines:
            if kind in ("upper", "lower", "threshold") and math.isfinite(value):
                entry.setdefault(kind if kind != "threshold" else "upper", float(value))
        plots[result.id] = entry
    return series, plots


# ------------------------------------------------------------------------------------------------------------
# one run in a worker


_REQSETS: dict[str, object] = {}


def _reqset(task: dict):
    from .api import load

    key = json.dumps([task["requirements"], task["mapping"], task["overrides"], task["only"]], sort_keys=True,
                     default=str)
    found = _REQSETS.get(key)
    if found is None:
        found = load(task["requirements"], mapping=task["mapping"], overrides=task["overrides"],
                     only=task["only"])
        _REQSETS.clear()
        _REQSETS[key] = found
    return found


def _compact(record: dict, *, page: str | None, cached: bool) -> dict:
    results = {}
    violations = {}
    for r in record.get("results", []):
        results[r["id"]] = {
            "v": r["verdict"], "value": r["value"], "margin": r["margin"], "pct": r["margin_pct"], "at": r["at"],
            "first": r["first_violation"], "reason": r["reason"], "case": r["case"], "limit": r["limit"],
            "unit": r["unit"], "runs": r["totals"].get("runs", r["totals"].get("count")),
        }
        rows = [run for run in r.get("runs", []) if not run.get("tolerated")][:VIOLATIONS_PER_REQUIREMENT]
        if rows:
            violations[r["id"]] = rows
    return {
        "id": record["run"], "source": record["source"], "status": record["status"], "counts": record["counts"],
        "params": record.get("params", {}), "error": record.get("error"), "t0": record.get("t0"),
        "duration": record.get("duration"), "results": results, "violations": violations,
        "series": record.get("envelopes", {}), "plots": record.get("plots", {}), "page": page, "cached": cached,
        "timing": record.get("timing", {}), "key": record.get("cache_key"), "format": record.get("format"),
        "archive": record.get("archive"), "issues": record.get("issues", []),
    }


def check_one(task: dict) -> dict:
    """Check one run and write its files; never raises for problems of the run."""
    from ..hashing import hash_file
    from .results import RunResult
    from .run import check_run

    folder = Path(task["output"]) / "runs"
    path = Path(task["path"])
    try:
        reqset = _reqset(task)
        digest = None
        ctx = None
        try:
            if task["hash"] != "none":
                info = hash_file(path, task["hash"])
                digest = f"{info.algorithm} {info.value}"
            run, ctx = check_run(path, reqset, params=task["params"], run_id=task["id"], digest=digest)
        except (BasltError, OSError) as exc:
            run = RunResult(run_id=task["id"], source=str(path), format="unknown",
                            requirements_file=reqset.table.path.name, params=dict(task["params"]), error=str(exc),
                            not_covered=list(reqset.not_covered))
        archived = None
        if task.get("archive") and run.error is None:
            from .archive import archive_run

            archived, notes = archive_run(path, reqset, run, folder / f"{task['id']}.baslt",
                                          max_size=task.get("max_size"), hash=task["hash"])
            run.issues.extend(notes)
        series, plots = envelopes(run, ctx)
        record = run.to_json()
        record.update({"cache_key": task["key"], "envelopes": series, "plots": plots, "baslt": __version__})
        page = None
        wants_page = task["pages"] == "all" or (task["pages"] == "failed" and run.status in ("fail", "warn",
                                                                                              "error"))
        if wants_page:
            from .report import write_run_report

            write_run_report(folder / f"{task['id']}.html", run, ctx, reqset)
            page = f"runs/{task['id']}.html"
        record["page"] = page
        record["archive"] = f"runs/{task['id']}.baslt" if archived is not None else None
        write_atomic(folder / f"{task['id']}.json", (json.dumps(record, ensure_ascii=False) + "\n").encode())
        return _compact(record, page=page, cached=False)
    except Exception as exc:  # noqa: BLE001 - one broken run must not stop the batch
        return failed(task, f"internal error: {exc}", traceback.format_exc())


def failed(task: dict, message: str, detail: str | None = None) -> dict:
    record = {"run": task["id"], "source": str(task["path"]), "status": "error", "counts": {}, "results": [],
              "params": task["params"], "error": message, "cache_key": None, "detail": detail}
    return _compact(record, page=None, cached=False)


def _init_worker() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------------------
# the batch


@dataclass
class BatchRun:
    """A finished run as the parent keeps it."""

    index: int
    summary: dict

    @property
    def id(self) -> str:
        return self.summary["id"]

    @property
    def status(self) -> str:
        return self.summary["status"]

    @property
    def results(self) -> list:
        return [_Verdict(rid, item["v"]) for rid, item in self.summary["results"].items()]

    @property
    def error(self) -> str | None:
        return self.summary.get("error")


@dataclass
class _Verdict:
    id: str
    verdict: str


def _mapping_digest(mapping) -> str:
    """One digest for however many mappings are laid over each other."""
    if mapping is None:
        return "none"
    if isinstance(mapping, (str, Path, Mapping)):
        layers = [mapping]
    else:
        layers = list(mapping)
    parts = []
    for layer in layers:
        if isinstance(layer, Mapping):
            parts.append(hashlib.sha256(json.dumps(layer, sort_keys=True, default=str).encode()).hexdigest())
        else:
            path = Path(layer)
            parts.append(hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else str(path))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def cache_key(task: dict, reqset, mapping_digest: str) -> str:
    path = Path(task["path"])
    stat = path.stat() if path.exists() else None
    parts = {
        "baslt": __version__, "requirements": reqset.sha256, "mapping": mapping_digest, "only": task["only"],
        "params": task["params"], "path": str(path.resolve()), "size": stat.st_size if stat else None,
        "overrides": task.get("overrides"),
        "mtime": stat.st_mtime_ns if stat else None, "hash": task["hash"],
        "archive": [bool(task.get("archive")), task.get("max_size")],
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _jobs(jobs, count: int) -> int:
    if jobs in (None, "", "auto"):
        n = max(1, (os.cpu_count() or 2) - 1)
    else:
        try:
            n = int(jobs)
        except (TypeError, ValueError):
            raise UsageError(f"jobs takes a number or auto, got {jobs!r}") from None
        if n < 1:
            raise UsageError("jobs must be at least 1")
    return max(1, min(n, count))


def _cached(task: dict, output: Path) -> dict | None:
    record_path = output / "runs" / f"{task['id']}.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("cache_key") != task["key"]:
        return None
    page = record.get("page")
    wants_page = task["pages"] == "all" or (task["pages"] == "failed" and record.get("status") in ("fail", "warn",
                                                                                                   "error"))
    if wants_page and (page is None or not (output / page).is_file()):
        return None
    if not wants_page:
        page = page if page and (output / page).is_file() else None
    if task.get("archive") and record.get("status") != "error" and not (output / "runs" /
                                                                          f"{task['id']}.baslt").is_file():
        return None
    return _compact(record, page=page, cached=True)


def _context(name: str | None):
    import multiprocessing

    if name is not None:
        return multiprocessing.get_context(name)
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")


def run_pool(tasks: Sequence[dict], jobs: int, on_done: Callable[[dict, dict], None], *,
             worker: Callable[[dict], dict] | None = None, context: str | None = None) -> None:
    """Check `tasks` with `jobs` processes (in this process for one job), calling `on_done` as each finishes."""
    worker = worker or check_one
    if jobs <= 1:
        for task in tasks:
            on_done(task, worker(task))
        return
    pending = list(tasks)
    breaks: dict[str, int] = {}
    mp = _context(context)
    while pending:
        round_tasks, pending = pending, []
        alone = [t for t in round_tasks if breaks.get(t["id"], 0) >= 2]
        shared = [t for t in round_tasks if breaks.get(t["id"], 0) < 2]
        for task in alone:
            try:
                with ProcessPoolExecutor(max_workers=1, mp_context=mp, initializer=_init_worker) as pool:
                    on_done(task, pool.submit(worker, task).result())
            except BrokenProcessPool:
                on_done(task, failed(task, "the worker process stopped while checking this run"))
        if not shared:
            continue
        with ProcessPoolExecutor(max_workers=min(jobs, len(shared)), mp_context=mp,
                                 initializer=_init_worker) as pool:
            futures = {pool.submit(worker, task): task for task in shared}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    outcome = future.result()
                except BrokenProcessPool:
                    breaks[task["id"]] = breaks.get(task["id"], 0) + 1
                    pending.append(task)
                    continue
                except Exception as exc:  # noqa: BLE001
                    outcome = failed(task, f"internal error: {exc}")
                on_done(task, outcome)


def check_batch(runs, reqset, *, requirements, mapping=None, only=None, params=None, output=None,
                fail_on: str = "fail", xlsx: bool = True, annotate: bool = True, html: bool = True,
                overrides=None, pages: str | None = None, jobs="auto", resume: bool = False,
                hash: str = "sampled",
                show_all: bool = False, progress: Callable[[str], None] | None = None, archive: bool = False,
                max_size=None,
                context: str | None = None, worker: Callable[[dict], dict] | None = None):
    from .api import CheckResult, _exit_code
    from .params import load_params, row_for

    config = reqset.config
    pages = pages or config.report.pages or "failed"
    if pages not in ("failed", "all", "none"):
        raise UsageError("pages must be failed, all or none")
    out_guess = Path(output) if output is not None else None
    paths, root = discover(runs, skip=out_guess)
    if output is None:
        output = root.parent / f"{root.name}.check" if root.name else Path("check")
    output = Path(output)
    (output / "runs").mkdir(parents=True, exist_ok=True)
    ids = run_ids(paths, root)
    table = None
    params_file = params if params is not None and not isinstance(params, Mapping) else config.params.file
    if params_file:
        table = load_params(params_file, config)
    mapping_digest = _mapping_digest(mapping)
    tasks = []
    for index, (path, rid) in enumerate(zip(paths, ids)):
        row = {}
        if isinstance(params, Mapping):
            row = dict(params.get(rid, {})) if all(isinstance(v, Mapping) for v in params.values()) else dict(params)
        elif table is not None:
            row = row_for(table, path, config, run_id=rid, root=root)
        task = {"index": index, "id": rid, "path": str(path), "params": row, "requirements": str(requirements),
                "mapping": _portable(mapping), "overrides": dict(overrides) if overrides else None,
                "only": list(only) if only else None, "output": str(output), "pages": pages if html else "none",
                "hash": hash, "archive": bool(archive), "max_size": None if max_size is None else str(max_size)}
        task["key"] = cache_key(task, reqset, mapping_digest)
        tasks.append(task)
    count = len(tasks)
    workers = _jobs(jobs, count)
    started = time.perf_counter()
    finished: dict[int, dict] = {}
    index_path = output / "index.jsonl"
    todo = []
    for task in tasks:
        found = _cached(task, output) if resume else None
        if found is not None:
            finished[task["index"]] = found
        else:
            todo.append(task)
    with open(index_path, "a" if resume else "w", encoding="utf-8") as index_file:
        done = len(finished)

        def on_done(task: dict, summary: dict) -> None:
            nonlocal done
            done += 1
            finished[task["index"]] = summary
            index_file.write(json.dumps(_index_line(summary), ensure_ascii=False) + "\n")
            index_file.flush()
            if progress is not None:
                failing = [rid for rid, item in summary["results"].items() if item["v"] == "fail"]
                if summary["status"] in ("fail", "error") or done == count or done % max(1, count // 10) == 0:
                    detail = summary["error"] or ", ".join(failing[:6]) + (" ..." if len(failing) > 6 else "")
                    progress(f"[{done:>{len(str(count))}}/{count}] {summary['id']:<24} "
                             f"{summary['status'].upper():<5} {detail}".rstrip())

        if progress is not None and resume and finished:
            progress(f"{len(finished)} of {count} runs are up to date")
        run_pool(todo, workers, on_done, worker=worker, context=context)
    batch = [BatchRun(i, finished[i]) for i in range(count)]
    lines = [json.dumps(_index_line(run.summary), ensure_ascii=False) for run in batch]
    write_atomic(index_path, ("\n".join(lines) + "\n").encode())

    from .aggregate import BatchSummary

    summary = BatchSummary(reqset, batch, output=output, root=root, params_file=params_file,
                           seconds=time.perf_counter() - started, jobs=workers)
    outputs: dict[str, Path] = {}
    if html:
        outputs["report"] = summary.write_dashboard(output / "index.html")
    if xlsx:
        outputs["xlsx"] = summary.write_xlsx(output / "results.xlsx")
    if annotate:
        path, notes = summary.write_annotated(output)
        if path is not None:
            outputs["annotated"] = path
        summary.notes.extend(notes)
    outputs["json"] = summary.write_json(output / "summary.json")
    outputs["index"] = index_path
    code = _exit_code(batch, fail_on)
    result = CheckResult(status=summary.status, exit_code=code, outputs=outputs, runs=[], counts=summary.counts())
    result.batch = summary
    result.text = summary.render(outputs, show_all=show_all)
    return result


def _portable(mapping):
    """The mappings as something a worker process can be handed: paths as text, dicts as they are."""
    if mapping is None or isinstance(mapping, Mapping):
        return mapping
    if isinstance(mapping, (str, Path)):
        return str(mapping)
    return [layer if isinstance(layer, Mapping) else str(layer) for layer in mapping]


def _index_line(summary: dict) -> dict:
    return {"run": summary["id"], "status": summary["status"], "counts": summary["counts"],
            "source": summary["source"], "page": summary["page"], "cached": summary["cached"],
            "error": summary["error"], "params": summary["params"]}
