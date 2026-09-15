"""Command-line interface.

Only the standard library is imported at module level so that `baslt --help` and
`baslt --version` start instantly. Each subcommand imports what it needs inside its handler.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Handler = Callable[[argparse.Namespace], int]


def _build_parser() -> tuple[argparse.ArgumentParser, dict[str, Handler]]:
    from ._version import __version__

    parser = argparse.ArgumentParser(
        prog="baslt",
        description="Compile simulation runs into small review artifacts with verified fidelity contracts.",
    )
    parser.add_argument("--version", action="version", version=f"baslt {__version__}")
    parser.add_subparsers(dest="command", metavar="COMMAND")
    handlers: dict[str, Handler] = {}
    return parser, handlers


def main(argv: Sequence[str] | None = None) -> int:
    parser, handlers = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handler = handlers[args.command]
    from .errors import BasltError

    try:
        return handler(args)
    except BasltError as exc:
        print(f"baslt: error: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
