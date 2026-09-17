"""The batch dashboard of `baslt check`: how each requirement did over all runs, a verdict matrix, overlays of the
checked signals, margins against run parameters, and the runs. The requirements table is plain HTML; the scripts
draw the rest from the embedded data.
"""

from __future__ import annotations

import math

from .._version import __version__
from .check_page import VERDICT_CLASS, _badge, _table
from .html import Packer, asset, esc, page

__all__ = ["render_batch_page"]

BAR_WIDTH = 120


def _bar(counts: list[int]) -> str:
    """A stacked bar of pass, warn, fail, n/a and error counts (SVG with classes, so the page's colors apply)."""
    total = sum(counts) or 1
    x = 0.0
    rects = []
    for verdict, count in zip(("pass", "warn", "fail", "na", "error"), counts):
        if not count:
            continue
        width = BAR_WIDTH * count / total
        rects.append(f'<rect class="bar-{verdict}" x="{x:.2f}" y="0" width="{width:.2f}" height="10"></rect>')
        x += width
    label = ", ".join(f"{c} {v}" for v, c in zip(("pass", "warn", "fail", "n/a", "error"), counts) if c)
    return (f'<svg class="bar" viewBox="0 0 {BAR_WIDTH} 10" width="{BAR_WIDTH}" height="10" role="img" '
            f'aria-label="{esc(label)}">{"".join(rects)}</svg>')


def render_batch_page(summary, data: dict, packer: Packer, *, title: str | None = None) -> str:
    from ..reqs.model import VERDICT_LABELS
    from ..reqs.results import fmt_value

    labels = dict(VERDICT_LABELS)
    config = summary.reqset.config
    heading = title or config.report.title or "Requirement check"
    counts = summary.counts()
    n = counts["runs"]
    meta = [f"{n} runs in {summary.root}", f"checked against {summary.reqset.table.path.name}"]
    if summary.params_file:
        meta.append(f"parameters from {str(summary.params_file).replace(chr(92), '/').rsplit('/', 1)[-1]}")
    meta.append(f"{summary.seconds:.1f} s with {summary.jobs} process{'es' if summary.jobs != 1 else ''}")
    chips = []
    for verdict in ("fail", "error", "warn", "pass", "not_applicable"):
        if counts[verdict] or verdict in ("fail", "pass"):
            chips.append(f'<span class="chip v-{VERDICT_CLASS[verdict]}"><b>{counts[verdict]}</b> '
                         f'{"run" if counts[verdict] == 1 else "runs"} {esc(labels[verdict].lower())}</span>')
    if counts["not_covered"]:
        chips.append(f'<a class="chip v-na" href="#issues"><b>{counts["not_covered"]}</b> not covered</a>')
    header = (
        '<header class="top"><div class="head">'
        f'<h1>{esc(heading)}</h1>{_badge(summary.status, labels)}</div>'
        f'<p class="meta">{" · ".join(esc(m) for m in meta)}</p>'
        f'<div class="chips">{"".join(chips)}</div>'
        '<nav class="toc"><a href="#requirements">Requirements</a><a href="#matrix">Verdict matrix</a>'
        '<a href="#details">Over all runs</a><a href="#runs">Runs</a><a href="#issues">Issues</a></nav>'
        "</header>"
    )

    rows = []
    order = sorted(range(len(summary.ids)), key=lambda k: (
        {"fail": 0, "error": 1, "warn": 2, "pass": 3, "not_applicable": 4}[summary.stats[summary.ids[k]]["verdict"]],
        k))
    for k in order:
        rid = summary.ids[k]
        req = summary.requirements[k]
        stat = summary.stats[rid]
        worst = stat.get("worst") or {}
        rate = "" if stat["pass_rate"] is None else f"{stat['pass_rate'] * 100:.1f} %"
        margin = ""
        if worst.get("margin") is not None:
            margin = fmt_value(worst["margin"], worst["unit"])
            if worst.get("pct") is not None and math.isfinite(worst["pct"]):
                margin += f" ({worst['pct'] * 100:.1f} %)"
        run_link = ""
        if worst.get("run"):
            target = summary.runs[worst["index"]].summary["page"] or f"runs/{worst['run']}.json"
            run_link = f'<a href="{esc(target)}">{esc(worst["run"])}</a>'
        rows.append(
            f'<tr class="pick" data-req="{k}"><td>{_badge(stat["verdict"], labels)}</td>'
            f'<td class="id">{esc(rid)}</td><td>{esc(req.title)}</td>'
            f'<td>{_bar([stat["counts"][v] for v in ("pass", "warn", "fail", "not_applicable", "error")])}</td>'
            f'<td class="num">{esc(rate)}</td><td class="num">{esc(margin)}</td><td>{run_link}</td>'
            f'<td>{esc(summary.sentence(rid))}</td></tr>')
    requirements = (
        '<div class="scroll"><table class="reqs"><thead><tr><th>Verdict</th><th>ID</th><th>Title</th>'
        '<th>Runs</th><th>Pass rate</th><th>Worst margin</th><th>Worst run</th><th>Over all runs</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="hint">Click a requirement to see it over all runs.</p>'
    )
    matrix = (
        '<p class="hint">One row per run, one column per requirement. Hover for details, click to select.</p>'
        '<div class="matrix"><canvas></canvas><div class="readout" hidden></div></div>'
        '<p class="selection" id="selection"></p>'
    )
    options = "".join(f'<option value="{k}">{esc(rid)}: {esc(summary.requirements[k].title)}</option>'
                      for k, rid in enumerate(summary.ids))
    param_options = '<option value="">run order</option>' + "".join(
        f'<option value="{esc(p["name"])}">{esc(p["name"])}</option>' for p in data["params"])
    details = (
        '<div class="tools"><select id="req-select" aria-label="Requirement">' + options + '</select>'
        '<label>Margin against <select id="param-select">' + param_options + '</select></label>'
        '<label><input type="checkbox" id="pct"> in %</label></div>'
        '<p id="req-sentence" class="sentence"></p>'
        '<div class="pair"><div class="plot overlay"><canvas></canvas><div class="legend"></div>'
        '<div class="readout" hidden></div></div>'
        '<div class="plot scatter"><canvas></canvas><div class="legend"></div>'
        '<div class="readout" hidden></div></div></div>'
        '<p class="hint">Left: the checked signal over all runs (range, 5 to 95 % band and median, with the '
        'failing and closest runs drawn one by one). Right: each run\'s margin; below zero is a violation. '
        'Click a point to open that run.</p>'
    )
    runs_section = (
        '<div class="tools"><input type="search" id="run-filter" placeholder="Filter runs: text, or payload>3000, '
        'status=fail" aria-label="Filter runs"><span id="run-count" class="hint"></span></div>'
        '<div class="vtable" id="run-table"><div class="vhead"></div><div class="vbody"><div class="vspace">'
        '</div></div></div>'
        '<noscript><p class="empty">The run list needs scripts; results.xlsx has every run.</p></noscript>'
    )
    issue_rows = []
    for run in summary.runs:
        if run.error:
            target = f"runs/{run.id}.json"
            issue_rows.append(["run error", f'<a href="{esc(target)}">{esc(run.id)}</a>', esc(run.error)])
    for rid in summary.reqset.not_covered:
        issue_rows.append(["not covered", esc(rid), "no check is given for this requirement"])
    for warning in summary.reqset.warnings:
        issue_rows.append(["warning", esc(warning.path), esc(warning.message + (f" ({warning.location})"
                                                                                if warning.location else ""))])
    for note in summary.notes:
        issue_rows.append(["note", "", esc(note)])
    issues = _table(["Level", "Where", "Message"], issue_rows, "issues") if issue_rows else \
        '<p class="empty ok">No problems with the requirements or the runs.</p>'
    body = (
        header
        + f'<main><section id="requirements"><h2>Requirements</h2>{requirements}</section>'
        + f'<section id="matrix"><h2>Verdict matrix</h2>{matrix}</section>'
        + f'<section id="details"><h2>Over all runs</h2>{details}</section>'
        + f'<section id="runs"><h2>Runs</h2>{runs_section}</section>'
        + f'<section id="issues"><h2>Issues</h2>{issues}</section></main>'
        + f'<footer>Runs in {esc(summary.root)} · baslt {esc(__version__)}</footer>'
        + '<pre id="selftest" hidden></pre>'
    )
    manifest = {"version": 1, "labels": labels, **data}
    description = (f"{labels.get(summary.status, summary.status)}: {counts['fail']} of {n} runs failed, "
                   f"{counts['warn']} warned")
    return page(f"{heading} · {n} runs", body, styles=[asset("report.css"), asset("dash.css")],
                scripts=[asset("dash.js")], data={"manifest": manifest}, blob=packer.blob(),
                description=description)
