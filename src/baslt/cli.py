"""Command-line interface with inexpensive help and lazy operation imports."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Handler = Callable[[argparse.Namespace], int]


class Parser(argparse.ArgumentParser):
    """Report usage failures with the documented exit code."""

    def error(self, message):
        from .errors import UsageError
        raise UsageError(message)


def _common(parser):
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS)


def _build_parser() -> tuple[argparse.ArgumentParser, dict[str, Handler]]:
    from ._version import __version__

    parser = Parser(prog="baslt", description="Compile simulation runs into small review artifacts with verified fidelity contracts.")
    parser.add_argument("--version", action="version", version=f"baslt {__version__}")
    _common(parser)
    parser.set_defaults(json=False, quiet=False)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    compile_parser = commands.add_parser("compile", help="Compile one simulation run")
    compile_parser.add_argument("source")
    compile_parser.add_argument("--policy", required=True)
    compile_parser.add_argument("-o", "--output")
    compile_parser.add_argument("--max-size")
    compile_parser.add_argument("--codec", choices=("deflate", "store", "zstd"))
    compile_parser.add_argument("--hash", choices=("full", "sampled", "none"))
    compile_parser.add_argument("--threads", type=int, default=1)
    compile_parser.add_argument("--no-self-verify", action="store_true")
    compile_parser.add_argument("--error-report")
    _common(compile_parser)
    verify_parser = commands.add_parser("verify", help="Verify an artifact's protected facts")
    verify_parser.add_argument("artifact")
    verify_parser.add_argument("--source")
    verify_parser.add_argument("--strict", action="store_true")
    _common(verify_parser)
    inspect_parser = commands.add_parser("inspect", help="List signals or summarize an artifact")
    inspect_parser.add_argument("source")
    _common(inspect_parser)
    policy_parser = commands.add_parser("policy", help="Work with policy files")
    policy_actions = policy_parser.add_subparsers(dest="policy_action", metavar="ACTION", required=True)
    init_parser = policy_actions.add_parser("init", help="Write a starter policy for a source")
    init_parser.add_argument("source")
    init_parser.add_argument("-o", "--output", help="policy file to write; printed to stdout when omitted")
    init_parser.add_argument("--format", choices=("yaml", "json"))
    init_parser.add_argument("--force", action="store_true", help="replace an existing output file")
    _common(init_parser)
    explain_parser = commands.add_parser("explain", help="Suggest how to fit an infeasible budget")
    explain_parser.add_argument("report", help="the .error.json an infeasible compile wrote")
    explain_parser.add_argument("--source", help="measure each relaxation on this source (needs --policy)")
    explain_parser.add_argument("--policy", help="the policy the report was compiled with")
    explain_parser.add_argument("--max-size", help="budget to measure against instead of the policy's")
    _common(explain_parser)
    requirements_parser = commands.add_parser("requirements", help="Work with requirement tables")
    requirement_actions = requirements_parser.add_subparsers(dest="requirements_action", metavar="ACTION",
                                                             required=True)
    template_parser = requirement_actions.add_parser("init", help="Write a starter requirements workbook")
    template_parser.add_argument("-o", "--output", default="requirements.xlsx",
                                 help="requirements file to write (.xlsx or .csv)")
    template_parser.add_argument("--template", choices=("generic", "polarion"), default="generic")
    template_parser.add_argument("--mapping-output", help="also write the mapping as a .yaml or .json file")
    template_parser.add_argument("--source", help="list this run's signals in the Signals sheet")
    template_parser.add_argument("--force", action="store_true", help="replace existing files")
    _common(template_parser)
    lint_parser = requirement_actions.add_parser("lint", help="Report every problem in a requirements table")
    lint_parser.add_argument("requirements")
    lint_parser.add_argument("-m", "--mapping", help="mapping file (.yaml, .json or .xlsx)")
    lint_parser.add_argument("--source", help="also resolve names against this run")
    lint_parser.add_argument("--params", help="run parameters table (for Applies to)")
    lint_parser.add_argument("--only", action="append", help="lint only these requirement IDs (glob)")
    _common(lint_parser)
    check_parser = commands.add_parser("check", help="Check runs against a requirements table")
    check_parser.add_argument("runs", nargs="+", help="run files, folders or glob patterns")
    check_parser.add_argument("-r", "--requirements", required=True, help="requirements table (.xlsx or .csv)")
    check_parser.add_argument("-m", "--mapping", help="mapping file (.yaml, .json or .xlsx)")
    check_parser.add_argument("--params", help="run parameters table (one row per run)")
    check_parser.add_argument("-o", "--output", help="output folder")
    check_parser.add_argument("--jobs", default="1", help="parallel runs: a number or auto")
    check_parser.add_argument("--resume", action="store_true", help="skip runs whose results are up to date")
    check_parser.add_argument("--fail-on", choices=("fail", "warn", "none"), default="fail")
    check_parser.add_argument("--pages", choices=("failed", "all", "none"), help="per-run report pages (batch)")
    check_parser.add_argument("--only", action="append", help="check only these requirement IDs (glob)")
    check_parser.add_argument("--no-html", action="store_true", help="do not write HTML reports")
    check_parser.add_argument("--no-xlsx", action="store_true", help="do not write the results workbook")
    check_parser.add_argument("--no-annotate", action="store_true", help="do not write the checked copy")
    check_parser.add_argument("--archive", action="store_true", help="also compile a .baslt artifact per run")
    check_parser.add_argument("--max-size", help="size budget of --archive artifacts")
    check_parser.add_argument("--hash", choices=("sampled", "full", "none"), default="sampled")
    check_parser.add_argument("--all", action="store_true", help="list passing requirements too")
    _common(check_parser)
    return parser, {"compile": _compile, "verify": _verify, "inspect": _inspect, "policy": _policy,
                    "explain": _explain, "requirements": _requirements, "check": _check}


def _emit(args, payload, message):
    if args.json:
        import json
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif not args.quiet:
        print(message, end="" if message.endswith("\n") else "\n")


def _compile(args):
    from pathlib import Path
    from .api import compile

    output = args.output or str(Path(args.source).with_suffix(".baslt"))
    result = compile(args.source, args.policy, output=output, max_size=args.max_size, codec=args.codec,
                     hash=args.hash, threads=args.threads, self_verify=not args.no_self_verify,
                     error_report=args.error_report, on_error="record" if args.error_report else "raise")
    _emit(args, result.to_json(), result.error["message"] if result.error else
          f"{result.status.upper()}: {result.output} ({result.size} bytes)")
    return result.exit_code


def _verify(args):
    from .api import verify
    result = verify(args.artifact, source=args.source, strict=args.strict)
    _emit(args, result.to_json(), result.render())
    return result.exit_code


def _inspect(args):
    from .api import inspect
    result = inspect(args.source)
    lines = [f"Format: {result['format']}", "SIGNAL  SHAPE  DTYPE  UNIT  KIND"]
    for entry in result["signals"]:
        dtype = entry.get("dtype") or next((a["dtype"] for a in entry.get("arrays", []) if a["name"] == "v"), "-")
        shape = entry.get("shape", [entry.get("n"), entry.get("components", 1)])
        lines.append(f"{entry['name']}  {shape}  {dtype}  {entry.get('unit') or '-'}  {entry['kind']}")
    _emit(args, result, "\n".join(lines))
    return 0


def _policy(args):
    from .api import init_policy

    result = init_policy(args.source, output=args.output, force=args.force, format=args.format)
    payload = {key: value for key, value in result.items() if key != "text"}
    if args.json:
        _emit(args, payload, "")
    elif args.output is None:
        print(result["text"], end="")
    elif not args.quiet:
        print(f"Wrote {result['output']}: {result['protected']} of {result['signals']} signals protected")
    return 0


def _explain(args):
    from .advice import explain, render

    result = explain(args.report, source=args.source, policy=args.policy, max_size=args.max_size)
    _emit(args, result, render(result))
    return 0


def _requirements(args):
    from .reqs import api

    if args.requirements_action == "init":
        result = api.init_template(args.output, template=args.template, mapping_output=args.mapping_output,
                                   source=args.source, force=args.force)
        written = " and ".join(result["written"])
        _emit(args, result, f"Wrote {written}: {result['rows']} example rows, {result['signals']} signals")
        return 0
    result = api.lint(args.requirements, mapping=args.mapping, source=args.source, params=args.params,
                      only=args.only)
    _emit(args, result.to_json(), result.render())
    return result.exit_code


def _check(args):
    from .errors import UsageError
    from .reqs import api

    jobs = args.jobs
    if jobs != "auto":
        try:
            jobs = int(jobs)
        except ValueError:
            raise UsageError(f"--jobs takes a number or auto, got {jobs!r}") from None
    runs = args.runs[0] if len(args.runs) == 1 else list(args.runs)
    result = api.check(runs, args.requirements, mapping=args.mapping, params=args.params, output=args.output,
                       fail_on=args.fail_on, only=args.only, xlsx=not args.no_xlsx, annotate=not args.no_annotate,
                       html=not args.no_html, pages=args.pages, jobs=jobs, resume=args.resume,
                       archive=args.archive, max_size=args.max_size, hash=args.hash, show_all=args.all)
    _emit(args, result.to_json(), result.text)
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    from .errors import BasltError
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        parser, handlers = _build_parser()
        args = parser.parse_args(raw)
        if not args.command:
            parser.print_help()
            return 0
        return handlers[args.command](args)
    except (BasltError, OSError) as exc:
        code = getattr(exc, "exit_code", 3)
        if "--json" in raw:
            import json
            print(json.dumps({"status": "error", "error": type(exc).__name__, "message": str(exc),
                              "exit_code": code, "report": getattr(exc, "report", {})}))
        else:
            print(f"baslt: error: {exc}", file=sys.stderr)
        return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
