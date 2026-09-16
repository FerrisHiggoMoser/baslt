"""Public operations for compiling, inspecting and verifying simulation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import os
import tempfile

from .errors import BasltError, CompileError, SourceError, UsageError


@dataclass(slots=True)
class CompileResult:
    """A compile outcome; recorded failures carry a report instead of raising."""

    status: str
    bytes: bytes = b""
    manifest: dict = field(default_factory=dict)
    output: Path | None = None
    error: dict | None = None
    exit_code: int = 0

    @property
    def size(self) -> int:
        return len(self.bytes)

    def to_json(self) -> dict:
        return {"status": self.status, "output": str(self.output) if self.output else None,
                "size_bytes": self.size, "manifest": self.manifest, "error": self.error,
                "exit_code": self.exit_code}


def _write(path: Path, data: bytes) -> None:
    """Replace the destination only after every byte has been written successfully."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_run(source, policy, *, max_size=None):
    """Resolve policy declarations before loading the included signals."""
    from .policy import bind_policy
    from .sources import open_source
    from .units import parse_bytes

    hints = {(decl.path or alias).lstrip("/"): decl.time for alias, decl in policy.signals.decls.items()
             if decl.time is not None}
    adapter = open_source(source, time_hints=hints, global_time=policy.signals.time)
    ceiling = None if max_size is None else parse_bytes(max_size, path="max_size")
    bound = bind_policy(policy, adapter.list_signals(), max_bytes_override=ceiling)
    options = {key: value for key, value in bound.load_options.items() if key not in ("units", "kinds")}
    run = adapter.load(bound.included, **options)
    run.signals = {name: replace(sig, unit=bound.units[name], kind=bound.kinds[name])
                   for name, sig in run.signals.items()}
    return run, bound


def _run_arrays(run) -> dict:
    return {f"{name}/{part}": getattr(sig, part) for name, sig in run.signals.items() for part in ("t", "v")}


def compile(source, policy, *, output=None, max_size=None, codec=None, hash=None, threads=1,
            self_verify=True, on_error="raise", error_report=None, suggestions="off", id="run") -> CompileResult:
    """Compile a path, mapping of arrays or Run, optionally writing an artifact.

    With ``on_error='record'`` ordinary failures return status ``error`` and write an error report.
    If the report itself cannot be written, its failure is included in the returned report.
    """
    try:
        from .hashing import hash_arrays, hash_file, hash_none
        from .plan.compile import compile_run
        from .policy import load_policy
        from .policy.schema import Policy

        if on_error not in ("raise", "record"):
            raise UsageError("on_error must be 'raise' or 'record'")
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise UsageError("threads must be a positive integer")
        if suggestions not in ("off", "lazy"):
            raise UsageError("suggestions must be 'off' or 'lazy'; use baslt explain for relaxation suggestions")
        parsed = policy if isinstance(policy, Policy) else load_policy(policy)
        if codec is not None:
            if codec not in ("store", "deflate", "zstd"):
                raise UsageError("codec must be store, deflate or zstd")
            parsed = replace(parsed, artifact=replace(parsed.artifact, codec=codec))
        run, bound = load_run(source, parsed, max_size=max_size)
        mode = hash or parsed.artifact.hash or ("full" if isinstance(source, (str, Path)) else "arrays")
        if mode == "none":
            digest = hash_none()
        elif isinstance(source, (str, Path)) and mode in ("full", "sampled"):
            digest = hash_file(source, mode)
        elif not isinstance(source, (str, Path)) and mode in ("arrays", "full"):
            digest = hash_arrays(_run_arrays(run))
        else:
            raise UsageError(f"hash mode {mode!r} is incompatible with this source")
        outcome = compile_run(run, bound, digest=digest, self_verify=self_verify)
        destination = Path(output) if output is not None else None
        if destination is not None:
            if isinstance(source, (str, Path)) and destination.resolve() == Path(source).resolve():
                raise UsageError("output must differ from the source file")
            _write(destination, outcome.bytes)
        return CompileResult(outcome.status, outcome.bytes, outcome.manifest, destination,
                             exit_code=1 if outcome.status == "fail" else 0)
    except Exception as exc:
        error = exc if isinstance(exc, BasltError) else (
            SourceError(str(exc)) if isinstance(exc, OSError) else CompileError(str(exc)))
        if on_error != "record":
            if error is exc:
                raise
            raise error from exc
        report = {"status": "error", "error": type(error).__name__, "message": str(error),
                  "exit_code": error.exit_code, **getattr(error, "report", {})}
        base = Path(output) if output is not None else (Path(source) if isinstance(source, (str, Path)) else Path(id))
        path = Path(error_report) if error_report is not None else base.with_suffix(".error.json")
        try:
            _write(path, (json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
        except Exception as report_error:
            report["report_error"] = str(report_error)
        return CompileResult("error", error=report, exit_code=error.exit_code)


def verify(artifact, *, source=None, strict=False):
    """Verify an artifact, loading an optional source with its embedded policy."""
    from .container import read_artifact
    from .policy import load_policy
    from .verify.checks import verify_artifact

    if source is None:
        return verify_artifact(artifact, strict=strict)
    parsed = read_artifact(artifact)
    run, _ = load_run(source, load_policy(parsed.policy["canonical"]))
    return verify_artifact(artifact, source=run, strict=strict)


def inspect(source) -> dict:
    """Describe a source's signals or summarize an artifact without compiling it."""
    from dataclasses import asdict
    from .container import read_artifact
    from .sources import open_source

    if isinstance(source, bytes) or (isinstance(source, (str, Path)) and Path(source).suffix == ".baslt"):
        artifact = read_artifact(source)
        return {"status": artifact.manifest.get("status", "pass"), "format": "baslt",
                "size_bytes": artifact.size, "signals": artifact.index["signals"], "manifest": artifact.manifest}
    adapter = open_source(source)
    return {"status": "pass", "format": adapter.format,
            "signals": [asdict(info) for info in adapter.list_signals()]}


def init_policy(source, *, output=None, force=False, format=None) -> dict:
    """Write (or return) a starter policy for a source's signals.

    The format is `format` ("yaml" or "json"), else taken from the output suffix, else YAML. An existing output is
    only replaced with `force`. The result holds the policy text, its mapping and where it was written.
    """
    from .policy import load_policy
    from .policy.scaffold import render_json, render_yaml, starter_policy
    from .sources import open_source

    destination = Path(output) if output is not None else None
    fmt = format or ("json" if destination is not None and destination.suffix.lower() == ".json" else "yaml")
    if fmt not in ("yaml", "json"):
        raise UsageError("format must be 'yaml' or 'json'")
    is_file = isinstance(source, (str, Path))
    adapter = open_source(source)
    infos = adapter.list_signals()
    if not infos:
        raise SourceError(f"no signals found in {source if is_file else 'the in-memory source'}")
    name = Path(source).stem if is_file else "run"
    size = Path(source).stat().st_size if is_file else None
    policy = starter_policy(infos, name=name, source_size=size, require_time=is_file)
    load_policy(policy)  # the starter policy must always be valid
    label = Path(source).name if is_file else "in-memory data"
    text = render_json(policy) if fmt == "json" else render_yaml(policy, infos, source_label=label,
                                                                 require_time=is_file)
    if destination is not None:
        if is_file and destination.resolve() == Path(source).resolve():
            raise UsageError("output must differ from the source file")
        if destination.exists() and not force:
            raise UsageError(f"{destination} already exists; use --force (force=True) to replace it")
        _write(destination, text.encode("utf-8"))
    protected = len(policy.get("hard", {}))
    return {"status": "pass", "output": str(destination) if destination is not None else None, "format": fmt,
            "signals": len(infos), "protected": protected, "policy": policy, "text": text}
