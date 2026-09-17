"""Public requirement-check operations: load, lint, check and templates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import Issue

from .config import Config, load_config
from .model import RequirementSet

__all__ = ["CheckResult", "check", "init_template", "lint", "load"]


def load(requirements: str | Path, *, mapping: str | Path | Mapping | None = None,
         overrides: Mapping | None = None, only: Sequence[str] | None = None) -> RequirementSet:
    """Read a requirements table with its mapping (config sheets in the workbook, then the mapping file)."""
    from .load import load_requirements

    path = Path(requirements)
    config: Config = load_config(mapping, workbook=path, overrides=overrides)
    return load_requirements(path, config, only=only)


def lint(requirements: str | Path, *, mapping: str | Path | Mapping | None = None, source=None, params=None,
         only: Sequence[str] | None = None):
    """Check a requirements table without running anything; with `source`, also resolve its names on that run."""
    from .lint import lint_static

    reqset = load(requirements, mapping=mapping, only=only)
    result = lint_static(reqset)
    if source is not None:
        from .bind import lint_against_source

        names = None
        if params is not None:
            names = set(_params_table(params, reqset.config, source)) if not isinstance(params, Mapping) else \
                set(params)
        result.items.extend(lint_against_source(reqset, source, param_names=names))
    return result


@dataclass
class CheckResult:
    status: str
    exit_code: int
    outputs: dict[str, Path] = field(default_factory=dict)
    runs: list = field(default_factory=list)  # RunResult of a single run
    counts: dict = field(default_factory=dict)
    text: str = ""
    batch: object = None  # reqs.aggregate.BatchSummary of several runs

    def to_json(self) -> dict:
        out = {
            "status": self.status, "exit_code": self.exit_code, "counts": self.counts,
            "outputs": {key: str(path) for key, path in self.outputs.items()},
            "runs": [run.to_json(with_results=len(self.runs) == 1) for run in self.runs],
        }
        if self.batch is not None:
            data = self.batch.to_json()
            out["runs"] = data["runs"]
            out["requirements"] = data["requirements"]
        return out


def _params_table(params, config, source) -> dict:
    """The parameter row of a single run: from a mapping, or the row of a table that matches the run."""
    if params is None:
        return {}
    if isinstance(params, Mapping):
        return dict(params)
    from .params import load_params, row_for

    table = load_params(params, config)
    return row_for(table, source, config)


def _exit_code(runs, fail_on: str) -> int:
    statuses = [run.status for run in runs]
    verdicts = {r.verdict for run in runs for r in run.results}
    if fail_on != "none" and "fail" in verdicts:
        return 1
    if fail_on == "warn" and "warn" in verdicts:
        return 1
    if "error" in statuses or "error" in verdicts:
        return 3
    return 0


def check(runs, requirements: str | Path, *, mapping: str | Path | Mapping | None = None, params=None,
          output: str | Path | None = None, fail_on: str = "fail", only: Sequence[str] | None = None,
          xlsx: bool = True, annotate: bool = True, html: bool = True, pages: str | None = None,
          jobs: int | str = "auto", resume: bool = False, archive: bool = False, max_size=None,
          hash: str = "sampled", show_all: bool = False, progress=None, _worker=None,
          _context: str | None = None) -> CheckResult:
    """Check one run (a path or in-memory arrays) or many runs against a requirements table.

    Many runs are a list of paths, a folder or a glob pattern; they are checked in `jobs` processes and summed
    up in a dashboard. Writes results to `output` (a folder; by default `<run>.check` next to a single run file or
    `<folder>.check` next to the runs' folder, nothing for in-memory data) and returns the verdicts with the exit
    code the command line would use. `progress` receives a line of text as runs finish. `_worker` and
    `_context` replace the per-run function and the process start method (for tests).
    """
    from ..errors import UsageError

    if fail_on not in ("fail", "warn", "none"):
        raise UsageError("fail_on must be fail, warn or none")
    reqset = load(requirements, mapping=mapping, only=only)
    many = isinstance(runs, (list, tuple)) or (isinstance(runs, (str, Path)) and _is_batch(runs))
    if archive:
        raise UsageError("--archive is not available yet")
    if many:
        from .batch import check_batch

        return check_batch(runs, reqset, requirements=requirements, mapping=mapping, only=only, params=params,
                           output=output, fail_on=fail_on, xlsx=xlsx, annotate=annotate, html=html, pages=pages,
                           jobs=jobs, resume=resume, hash=hash, show_all=show_all, progress=progress,
                           worker=_worker, context=_context)
    return _check_single(runs, reqset, params=params, output=output, fail_on=fail_on, xlsx=xlsx,
                         annotate=annotate, html=html, archive=archive, max_size=max_size, hash=hash,
                         show_all=show_all)


def _is_batch(runs) -> bool:
    path = Path(runs)
    return path.is_dir() or any(ch in str(runs) for ch in "*?[")


def _check_single(source, reqset: RequirementSet, *, params, output, fail_on, xlsx, annotate, html, archive,
                  max_size, hash, show_all) -> CheckResult:
    from ..errors import SourceError
    from ..hashing import hash_file
    from .results import annotate as write_annotated
    from .results import render_run, run_sheets, write_csv, write_json
    from .run import check_run

    is_file = isinstance(source, (str, Path))
    if is_file and not Path(source).is_file():
        raise SourceError(f"run file not found: {source}")
    row = _params_table(params, reqset.config, source)
    digest = None
    if is_file and hash != "none":
        info = hash_file(source, hash)
        digest = f"{info.algorithm} {info.value}"
    run, ctx = check_run(source, reqset, params=row, digest=digest)
    outputs: dict[str, Path] = {}
    folder = Path(output) if output is not None else (Path(source).with_suffix(".check") if is_file else None)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        outputs["json"] = write_json(folder / "results.json", run)
        outputs["csv"] = write_csv(folder / "results.csv", run)
        if xlsx:
            from ..tabular import write_xlsx

            outputs["xlsx"] = write_xlsx(folder / "results.xlsx", run_sheets(run, reqset), title="Requirement check")
        if annotate and run.error is None:
            table = reqset.table
            name = f"{table.path.stem}.checked{table.path.suffix}"
            path, notes = write_annotated(reqset, {r.id: r for r in run.results}, folder / name, t0=run.t0)
            outputs["annotated"] = path
            for note in notes:
                run.issues.append(Issue(path="annotate", message=note, location=name))
        if html:
            from .report import write_run_report

            outputs["report"] = write_run_report(folder / "report.html", run, ctx, reqset)
    code = _exit_code([run], fail_on)
    status = run.status
    result = CheckResult(status=status, exit_code=code, outputs=outputs, runs=[run], counts=run.counts())
    result.text = render_run(run, show_all=show_all, outputs=outputs)
    return result


def init_template(output: str | Path, *, template: str = "generic", mapping_output: str | Path | None = None,
                  source=None, force: bool = False) -> dict:
    from .templates import init_template as write

    return write(output, template=template, mapping_output=mapping_output, source=source, force=force)
