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
    return parser, {"compile": _compile, "verify": _verify, "inspect": _inspect}


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
