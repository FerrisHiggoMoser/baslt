"""The run report page of `baslt check`: what went wrong, a timeline, every requirement with its plot, events and
issues. Tables are plain HTML, so the page reads and prints without scripts; the scripts add plots, sorting and
filters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .._version import __version__
from .html import Packer, asset, esc, page

__all__ = ["render_run_page"]

VERDICT_CLASS = {"pass": "pass", "warn": "warn", "fail": "fail", "not_applicable": "na", "error": "error"}
ORDER = {"fail": 0, "error": 1, "warn": 2, "not_applicable": 3, "pass": 4}
RUN_ROWS = 20
EDGE_TEXT = {"open_start": "from the start of the data", "open_end": "until the end of the data",
             "gap_start": "starts at a gap or window edge", "gap_end": "ends at a gap or window edge"}


def _badge(verdict: str, labels: Mapping[str, str]) -> str:
    return f'<span class="badge v-{VERDICT_CLASS.get(verdict, "na")}">{esc(labels.get(verdict, verdict))}</span>'


def _table(head: Sequence[str], rows: Sequence[Sequence[str]], cls: str = "") -> str:
    """`rows` hold ready HTML."""
    out = [f'<div class="scroll"><table class="{cls}"><thead><tr>'
           + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    out += ["<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows]
    out.append("</tbody></table></div>")
    return "".join(out)


def render_run_page(run, reqset, plots: Mapping, packer: Packer, *, title: str | None = None) -> str:
    """`plots` is the plot manifest of reqs.plotdata.build_plots; its arrays are in `packer`."""
    from ..reqs.model import VERDICT_LABELS
    from ..reqs.results import evidence_text, fmt_time, fmt_value

    labels = dict(VERDICT_LABELS)
    t0 = run.t0
    counts = run.counts()
    heading = title or (reqset.config.report.title if reqset is not None and reqset.config.report.title else
                        "Requirement check")
    source_name = run.source.replace("\\", "/").rsplit("/", 1)[-1]

    def when(seconds) -> str:
        return esc(fmt_time(seconds, t0))

    def margin(result) -> str:
        if result.margin is None:
            return ""
        text = fmt_value(result.margin, result.unit)
        if result.margin_pct is not None and math.isfinite(result.margin_pct):
            text += f" ({result.margin_pct * 100:.1f} %)"
        return esc(text)

    def context(result) -> str:
        ctx = result.context or {}
        parts = []
        if ctx.get("conditions"):
            parts.append(", ".join(ctx["conditions"]))
        if ctx.get("event") is not None:
            parts.append(f"{ctx['since']:.1f} s after {ctx['event']}")
        return "; ".join(parts)

    # ----- header -----------------------------------------------------------------------------------------
    meta = [source_name, run.format]
    if run.duration is not None:
        meta.append(f"{run.duration:.1f} s")
    meta.append(f"{run.signals} signals")
    meta.append(f"checked against {run.requirements_file}")
    if t0 is not None:
        meta.append(f"T+0 = {run.t0_event} at {t0:.6g} s")
    chips = []
    for verdict in ("fail", "error", "warn", "pass", "not_applicable"):
        chips.append(f'<button type="button" class="chip v-{VERDICT_CLASS[verdict]}" data-filter="{verdict}" '
                     f'aria-pressed="true"><b>{counts[verdict]}</b> {esc(labels[verdict].lower())}</button>')
    if counts["not_covered"]:
        chips.append(f'<a class="chip v-na" href="#issues"><b>{counts["not_covered"]}</b> not covered</a>')
    header = (
        '<header class="top"><div class="head">'
        f'<h1>{esc(heading)}</h1>{_badge(run.status, labels)}</div>'
        f'<p class="meta">{" · ".join(esc(m) for m in meta)}</p>'
        f'<div class="chips">{"".join(chips)}</div>'
        '<nav class="toc"><a href="#wrong">What went wrong</a><a href="#timeline">Timeline</a>'
        '<a href="#requirements">Requirements</a><a href="#events">Events</a><a href="#issues">Issues</a></nav>'
        "</header>"
    )
    if run.error:
        header += f'<p class="run-error">The run could not be checked: {esc(run.error)}</p>'

    # ----- what went wrong --------------------------------------------------------------------------------
    problems = [r for r in run.results if r.verdict in ("fail", "error", "warn")]
    problems.sort(key=lambda r: (ORDER[r.verdict], r.first_violation if r.first_violation is not None
                                 else (r.at if r.at is not None else math.inf)))
    if problems:
        rows = []
        for r in problems:
            start = r.first_violation if r.first_violation is not None else r.at
            end = start
            failing = [run_ for run_ in r.runs if not run_["tolerated"]]
            if failing:
                end = failing[0]["end"]
            reason = r.reason or ""
            if r.verdict == "error" and r.issues and r.issues[0].location:
                reason += f" ({r.issues[0].location})"
            detail = "; ".join(p for p in (reason, context(r)) if p)
            data = ""
            if start is not None:
                base = t0 or 0.0
                data = f' data-s="{start - base:.9g}" data-e="{(end if end is not None else start) - base:.9g}"'
            rows.append(
                f'<tr class="jump" data-req="{esc(r.id)}"{data}><td>{_badge(r.verdict, labels)}</td>'
                f'<td class="id"><a href="#req-{esc(r.id)}">{esc(r.id)}</a></td><td>{esc(r.title)}</td>'
                f'<td class="num">{esc(fmt_value(r.value, r.unit)) if r.value is not None else ""}</td>'
                f'<td>{esc(r.limit or "")}</td><td class="num">{margin(r)}</td><td class="num">{when(start)}</td>'
                f'<td>{esc(detail)}</td></tr>')
        wrong = ('<div class="scroll"><table class="wrong"><thead><tr><th>Verdict</th><th>ID</th><th>Title</th>'
                 '<th>Worst</th><th>Limit</th><th>Margin</th><th>From</th><th>Why</th></tr></thead><tbody>'
                 + "".join(rows) + "</tbody></table></div>")
    elif run.error:
        wrong = '<p class="empty">Nothing was checked.</p>'
    else:
        wrong = '<p class="empty ok">Nothing went wrong: every requirement passed or does not apply.</p>'

    # ----- requirements -----------------------------------------------------------------------------------
    cards = []
    ordered = sorted(run.results, key=lambda r: (r.rows[0] if r.rows else 0))
    for position, r in enumerate(ordered):
        open_attr = " open" if r.verdict in ("fail", "warn", "error") else ""
        at = r.first_violation if r.first_violation is not None else r.at
        sort_margin = r.margin_pct if r.margin_pct is not None else (r.margin if r.margin is not None else "")
        summary = (
            f'<summary><span>{_badge(r.verdict, labels)}</span><span class="id">{esc(r.id)}</span>'
            f'<span class="title">{esc(r.title)}</span>'
            f'<span class="num">{esc(fmt_value(r.value, r.unit)) if r.value is not None else ""}</span>'
            f'<span>{esc(r.limit or "")}</span><span class="num">{margin(r)}</span>'
            f'<span class="num">{when(at) if at is not None else ""}</span></summary>'
        )
        facts = [("Check", f"<code>{esc(r.check)}</code>"), ("Type", esc(r.kind))]
        if r.case:
            facts.append(("Case", esc(r.case)))
        if r.severity:
            facts.append(("Severity", esc(r.severity)))
        facts.append(("Evidence", esc(evidence_text(r, t0))))
        ctx_text = context(r)
        if ctx_text:
            facts.append(("Where", esc(ctx_text)))
        totals = r.totals or {}
        if totals.get("active_time") is not None:
            facts.append(("Checked for", esc(f"{totals['active_time']:.4g} s")))
        for note in r.notes:
            facts.append(("Note", esc(note)))
        for issue in r.issues:
            facts.append(("Problem", esc(issue.message + (f" ({issue.location})" if issue.location else ""))))
        facts.append(("Source", esc(r.loc)))
        body = ['<dl class="facts">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts) + "</dl>"]
        if r.id in plots["plots"]:
            body.insert(0, f'<div class="plot" data-req="{esc(r.id)}"><canvas></canvas>'
                           '<div class="legend"></div><div class="readout" hidden></div></div>')
        if r.runs:
            with_cases = any(item.get("case") for item in r.runs)
            rows = []
            for item in r.runs[:RUN_ROWS]:
                row = [esc("tolerated" if item["tolerated"] else "violation"), when(item["start"]),
                       when(item["end"]), esc(f"{item['duration']:.4g} s"), esc(item["samples"]),
                       esc(fmt_value(item.get("worst_value"), r.unit)) if "worst_value" in item else ""]
                if with_cases:
                    row.append(esc(item.get("case") or ""))
                row.append(esc(", ".join(EDGE_TEXT.get(flag, flag) for flag in item["flags"])))
                rows.append(row)
            more = f'<p class="more">{len(r.runs) - RUN_ROWS} more in results.xlsx</p>' \
                if len(r.runs) > RUN_ROWS else ""
            head = ["Result", "From", "To", "Duration", "Samples", "Worst", *(["Case"] if with_cases else []),
                    "Note"]
            body.append(_table(head, rows, "runs") + more)
        if len(r.cases) > 1:
            rows = [[_badge(c.verdict, labels), esc(c.label or ""), esc(fmt_value(c.value, r.unit))
                     if c.value is not None else "", esc(fmt_value(c.limit, r.unit)) if c.limit is not None else "",
                     esc(fmt_value(c.margin, r.unit)) if c.margin is not None else "", esc(c.reason or "")]
                    for c in r.cases]
            body.append(_table(["Verdict", "Case", "Value", "Limit", "Margin", "Why"], rows, "cases"))
        cards.append(
            f'<details class="card v-{VERDICT_CLASS[r.verdict]}" id="req-{esc(r.id)}" data-verdict="{r.verdict}" '
            f'data-order="{position}" data-rank="{ORDER[r.verdict]}" data-margin="{esc(sort_margin)}" '
            f'data-at="{"" if at is None else f"{at:.9g}"}"{open_attr}>{summary}'
            f'<div class="body">{"".join(body)}</div></details>')
    requirements = (
        '<div class="tools"><input type="search" id="filter" placeholder="Filter by ID, title or check" '
        'aria-label="Filter requirements"><label><input type="checkbox" id="link" checked> Link time axes</label>'
        '<button type="button" id="reset">Reset zoom</button><button type="button" id="expand">Open all</button>'
        '</div>'
        '<div class="list" role="table"><div class="list-head" role="row">'
        '<button type="button" data-sort="rank">Verdict</button><button type="button" data-sort="order">ID</button>'
        '<span>Title</span><span class="num">Value</span><span>Limit</span>'
        '<button type="button" class="num" data-sort="margin">Margin</button>'
        '<button type="button" class="num" data-sort="at">At</button></div>'
        + "".join(cards) + '<p class="empty" id="nomatch" hidden>No requirement matches the filter.</p></div>'
    )

    # ----- events and issues ------------------------------------------------------------------------------
    event_rows = []
    for name, info in run.events.items():
        times = info.get("times") or []
        shown = ", ".join(fmt_time(v, t0) for v in times[:12]) + (" ..." if len(times) > 12 else "")
        state = info.get("error") or ("still pending at the end" if info.get("pending") else "")
        event_rows.append([esc(name), esc(info.get("count") if info.get("count") is not None else "-"),
                           esc(shown), esc(state)])
    events = _table(["Event", "Count", "Times", "Note"], event_rows, "events") if event_rows else \
        '<p class="empty">No events are defined in the mapping.</p>'
    issue_rows = []
    for issue in run.issues:
        issue_rows.append(["error", esc(issue.path), esc(issue.message), esc(issue.location or "")])
    for r in run.results:
        for issue in r.issues:
            issue_rows.append(["error", esc(r.id), esc(issue.message), esc(issue.location or "")])
    for rid in run.not_covered:
        issue_rows.append(["not covered", esc(rid), "no check is given for this requirement", ""])
    if reqset is not None:
        for warning in reqset.warnings:
            issue_rows.append(["warning", esc(warning.path), esc(warning.message), esc(warning.location or "")])
    issues = _table(["Level", "Where", "Message", "Location"], issue_rows, "issues") if issue_rows else \
        '<p class="empty ok">No problems with the requirements or the run.</p>'

    provenance = [f"Source {esc(run.source)}"]
    if run.digest:
        provenance.append(f"digest {esc(run.digest)}")
    provenance.append(f"baslt {esc(__version__)}")
    body = (
        header
        + f'<main><section id="wrong"><h2>What went wrong</h2>{wrong}</section>'
        + '<section id="timeline"><h2>Timeline</h2><div class="plot timeline"><canvas></canvas>'
          '<div class="readout" hidden></div></div><p class="hint">Drag across a plot to zoom, double-click to '
          'zoom out. Click a row above to jump to it.</p></section>'
        + f'<section id="requirements"><h2>Requirements</h2>{requirements}</section>'
        + f'<section id="events"><h2>Events</h2>{events}</section>'
        + f'<section id="issues"><h2>Issues</h2>{issues}</section></main>'
        + f'<footer>{" · ".join(provenance)}</footer><pre id="selftest" hidden></pre>'
        + '<noscript><p class="empty">Plots need scripts; the tables above are complete without them.</p>'
          '</noscript>'
    )
    manifest = {"version": 1, "t0": t0, "t0_event": run.t0_event, "labels": labels, **plots}
    description = f"{labels.get(run.status, run.status)}: {counts['fail']} failed, {counts['warn']} warnings, " \
                  f"{counts['pass']} passed for {source_name}"
    return page(f"{heading} · {source_name}", body, styles=[asset("report.css")], scripts=[asset("report.js")],
                data={"manifest": manifest}, blob=packer.blob(), description=description)
