"""The HTML report of one checked run."""

from __future__ import annotations

from pathlib import Path

from .._io import write_atomic

__all__ = ["render_run_report", "write_run_report"]


def render_run_report(run, ctx, reqset, *, packer=None) -> str:
    """The page for `run`; `packer` (a report.html.Packer) collects its arrays."""
    from ..report.check_page import render_run_page
    from ..report.html import Packer
    from .plotdata import build_plots

    packer = Packer() if packer is None else packer
    plots = build_plots(run, ctx, packer) if ctx is not None else {"base": 0.0, "span": [0.0, 0.0], "series": [],
                                                                     "plots": {}, "events": [], "phases": []}
    return render_run_page(run, reqset, plots, packer)


def write_run_report(path, run, ctx, reqset) -> Path:
    path = Path(path)
    write_atomic(path, render_run_report(run, ctx, reqset).encode("utf-8"))
    return path
