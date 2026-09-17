"""What a batch of runs says about each requirement: counts, worst runs, overlays and the batch files."""

from __future__ import annotations

import json
import math
import os
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .._io import write_atomic
from .batch import ENVELOPE_BINS, VIOLATION_ROWS, _dequantize
from .model import VERDICT_LABELS
from .results import STYLE_OF, fmt_value

__all__ = ["BatchSummary"]

VERDICTS = ("pass", "warn", "fail", "not_applicable", "error")
CODES = {"pass": 0, "warn": 1, "fail": 2, "not_applicable": 3, "error": 4}
NOT_CHECKED = 5
SEVERITY = {"fail": 0, "error": 1, "warn": 2, "pass": 3, "not_applicable": 4}
OVERLAY_TRACES = 16  # runs drawn one by one per requirement
SERIES_TRACES = 32  # runs stored per signal, unless more of them fail
DISTINCT_LIMITS = 4


def _worst_verdict(verdicts) -> str:
    present = set(verdicts)
    for verdict in ("fail", "error", "warn", "pass"):
        if verdict in present:
            return verdict
    return "not_applicable"


class BatchSummary:
    def __init__(self, reqset, runs, *, output: Path, root: Path, params_file=None, seconds: float = 0.0,
                 jobs: int = 1) -> None:
        self.reqset = reqset
        self.runs = runs
        self.output = Path(output)
        self.root = root
        self.params_file = params_file
        self.seconds = seconds
        self.jobs = jobs
        self.notes: list[str] = []
        self.requirements = [req for req in reqset.requirements if req.covered]
        self.ids = [req.id for req in self.requirements]
        self.param_names = list(dict.fromkeys(k for run in runs for k in run.summary["params"]))
        self.stats = {rid: self._stat(rid) for rid in self.ids}

    # ----- numbers ----------------------------------------------------------------------------------------

    def item(self, run, rid: str) -> dict | None:
        return run.summary["results"].get(rid)

    def _stat(self, rid: str) -> dict:
        counts = dict.fromkeys(VERDICTS, 0)
        counts["not_checked"] = 0
        worst = None
        reasons: dict[str, int] = {}
        for run in self.runs:
            item = self.item(run, rid)
            if item is None:
                counts["not_checked"] += 1
                continue
            counts[item["v"]] += 1
            if item["v"] in ("fail", "error", "warn") and item["reason"]:
                reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
            margin = item["margin"]
            if margin is None or not math.isfinite(margin):
                continue
            rank = (0 if item["v"] == "fail" else 1, margin)
            if worst is None or rank < worst[0]:
                worst = (rank, run)
        checked = counts["pass"] + counts["warn"] + counts["fail"]
        out = {"counts": counts, "checked": checked, "pass_rate": counts["pass"] / checked if checked else None,
               "verdict": _worst_verdict(v for v in VERDICTS if counts[v]),
               "reason": max(reasons, key=reasons.get) if reasons else None}
        if worst is not None:
            run = worst[1]
            item = self.item(run, rid)
            out["worst"] = {"run": run.id, "index": run.index, "margin": item["margin"], "pct": item["pct"],
                            "value": item["value"], "limit": item["limit"], "unit": item["unit"],
                            "verdict": item["v"]}
        return out

    def counts(self) -> dict:
        out = dict.fromkeys(VERDICTS, 0)
        for run in self.runs:
            out[run.status] = out.get(run.status, 0) + 1
        out["runs"] = len(self.runs)
        out["cached"] = sum(1 for run in self.runs if run.summary.get("cached"))
        out["not_covered"] = len(self.reqset.not_covered)
        return out

    @property
    def status(self) -> str:
        return _worst_verdict(run.status for run in self.runs)

    def sentence(self, rid: str) -> str:
        """One sentence about a requirement over all runs, for the checked copy."""
        stat = self.stats[rid]
        counts = stat["counts"]
        n = len(self.runs)
        worst = stat.get("worst")
        tail = ""
        if worst is not None:
            label = "worst" if worst["verdict"] == "fail" else "least margin"
            tail = f", {label} {fmt_value(worst['margin'], worst['unit'])} ({worst['run']})"
        for verdict in ("fail", "error", "warn"):
            if counts[verdict]:
                text = f"{VERDICT_LABELS[verdict]} in {counts[verdict]}/{n} runs"
                if verdict == "error" and stat["reason"]:
                    return f"{text}: {stat['reason']}"
                return text + tail
        if counts["pass"]:
            scope = f"all {n} runs" if counts["pass"] == n else f"{counts['pass']}/{n} runs"
            return f"PASS in {scope}{tail}"
        return f"N/A in all {n} runs" if counts["not_applicable"] == n else "not checked"

    # ----- files ------------------------------------------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "status": self.status, "counts": self.counts(), "seconds": self.seconds, "jobs": self.jobs,
            "requirements_file": self.reqset.table.path.name, "not_covered": list(self.reqset.not_covered),
            "requirements": [{"id": rid, "title": req.title, **self.stats[rid], "sentence": self.sentence(rid)}
                             for rid, req in zip(self.ids, self.requirements)],
            "runs": [{"run": run.id, "status": run.status, "counts": run.summary["counts"],
                      "page": run.summary["page"], "error": run.error, "params": run.summary["params"],
                      "cached": run.summary["cached"]} for run in self.runs],
            "notes": self.notes,
        }

    def write_json(self, path: Path) -> Path:
        write_atomic(path, (json.dumps(_clean(self.to_json()), indent=2, ensure_ascii=False) + "\n").encode())
        return path

    def sheets(self) -> list:
        from ..tabular import Link, Sheet, Styled

        def head(names):
            return [Styled(name, "header") for name in names]

        counts = self.counts()
        summary = [
            head(["Item", "Value"]),
            ["Requirements", self.reqset.table.path.name],
            ["Runs", counts["runs"]],
            ["Status", Styled(VERDICT_LABELS.get(self.status, self.status), STYLE_OF.get(self.status, "bold"))],
            *[[f"Runs {VERDICT_LABELS[v]}", counts[v]] for v in VERDICTS],
            ["Runs from cache", counts["cached"]],
            ["Not covered", ", ".join(self.reqset.not_covered) or "none"],
            ["Run folder", str(self.root)],
            ["Parameters", str(self.params_file) if self.params_file else "none"],
            ["Seconds", round(self.seconds, 3)],
            ["Processes", self.jobs],
        ]
        matrix = [head(["Run", "Status", *self.ids])]
        margins = [head(["Run", "Status", *self.ids])]
        for run in self.runs:
            row = [run.id, Styled(VERDICT_LABELS.get(run.status, run.status), STYLE_OF.get(run.status, "bold"))]
            numbers = [run.id, VERDICT_LABELS.get(run.status, run.status)]
            for rid in self.ids:
                item = self.item(run, rid)
                if item is None:
                    row.append(None)
                    numbers.append(None)
                    continue
                row.append(Styled(VERDICT_LABELS[item["v"]], STYLE_OF[item["v"]]))
                numbers.append(item["margin"])
            matrix.append(row)
            margins.append(numbers)
        requirements = [head(["ID", "Title", "Type", "Check", "Result", "Pass", "Warn", "Fail", "N/A", "Error",
                              "Pass rate", "Worst margin", "Worst margin %", "Worst value", "Unit", "Worst run",
                              "Summary"])]
        for rid, req in zip(self.ids, self.requirements):
            stat = self.stats[rid]
            worst = stat.get("worst") or {}
            c = stat["counts"]
            requirements.append([
                rid, req.title, req.kind or None, req.check, Styled(VERDICT_LABELS[stat["verdict"]],
                                                                  STYLE_OF[stat["verdict"]]),
                c["pass"], c["warn"], c["fail"], c["not_applicable"], c["error"],
                Styled(stat["pass_rate"], "pct") if stat["pass_rate"] is not None else None,
                worst.get("margin"), Styled(worst["pct"], "pct") if worst.get("pct") is not None else None,
                worst.get("value") if not isinstance(worst.get("value"), str) else worst.get("value"),
                worst.get("unit") or None, worst.get("run"), self.sentence(rid)])
        for rid in self.reqset.not_covered:
            requirements.append([rid, "", "", "", Styled("NOT COVERED", "na")])
        violations = [head(["Run", "ID", "Start (s)", "End (s)", "Duration (s)", "Samples", "Worst value", "Limit",
                            "Case", "Edges"])]
        rows = 0
        for run in self.runs:
            for rid, items in run.summary["violations"].items():
                for item in items:
                    if rows >= VIOLATION_ROWS:
                        break
                    violations.append([run.id, rid, item["start"], item["end"], item["duration"], item["samples"],
                                       item.get("worst_value"), item.get("limit"), item.get("case") or None,
                                       ", ".join(item.get("flags", [])) or None])
                    rows += 1
        runs = [head(["Run", "Status", "Pass", "Warn", "Fail", "N/A", "Error", "Duration (s)", "Source", "Page",
                      "Problem", *self.param_names])]
        for run in self.runs:
            s = run.summary
            c = s["counts"]
            page = Link(s["page"], s["page"]) if s["page"] else None
            runs.append([run.id, Styled(VERDICT_LABELS.get(run.status, run.status), STYLE_OF.get(run.status,
                                                                                                "bold")),
                         c.get("pass"), c.get("warn"), c.get("fail"), c.get("not_applicable"), c.get("error"),
                         s.get("duration"), s["source"], page, s.get("error") or None,
                         *[s["params"].get(name) for name in self.param_names]])
        return [
            Sheet("Summary", summary, widths=[22, 70], autofilter=False),
            Sheet("Matrix", matrix, freeze=(1, 2)),
            Sheet("Margins", margins, freeze=(1, 2)),
            Sheet("Requirements", requirements),
            Sheet("Violations", violations),
            Sheet("Runs", runs),
        ]

    def write_xlsx(self, path: Path) -> Path:
        from ..tabular import write_xlsx

        return write_xlsx(path, self.sheets(), title="Requirement check of a batch")

    def write_annotated(self, output: Path) -> tuple[Path | None, list[str]]:
        from .check import RequirementResult
        from .results import annotate, annotated_name

        results = {}
        for rid, req in zip(self.ids, self.requirements):
            stat = self.stats[rid]
            worst = stat.get("worst") or {}
            results[rid] = RequirementResult(
                id=rid, title=req.title, kind=req.kind or "", check=req.check, verdict=stat["verdict"],
                unit=worst.get("unit"), value=worst.get("value"), limit=worst.get("limit"),
                margin=worst.get("margin"), margin_pct=worst.get("pct"),
                totals={"runs": stat["counts"]["fail"]})
        name = annotated_name(self.reqset.table)
        aggregate = {rid: self.sentence(rid) for rid in self.ids}
        return annotate(self.reqset, results, output / name, aggregate=aggregate)

    def write_dashboard(self, path: Path, *, packer=None) -> Path:
        """index.html; `packer` (a report.html.Packer) collects its arrays."""
        from ..report.dashboard import render_batch_page
        from ..report.html import Packer

        packer = Packer() if packer is None else packer
        text = render_batch_page(self, self.dashboard_data(packer), packer)
        write_atomic(path, text.encode("utf-8"))
        return path

    # ----- dashboard data ---------------------------------------------------------------------------------

    def dashboard_data(self, packer, bins: int = ENVELOPE_BINS) -> dict:
        n_runs, n_reqs = len(self.runs), len(self.ids)
        codes = np.full((n_runs, n_reqs), NOT_CHECKED, dtype=np.uint8)
        margins = np.full((n_runs, n_reqs), np.nan)
        pcts = np.full((n_runs, n_reqs), np.nan)
        for i, run in enumerate(self.runs):
            for j, rid in enumerate(self.ids):
                item = self.item(run, rid)
                if item is None:
                    continue
                codes[i, j] = CODES[item["v"]]
                if item["margin"] is not None:
                    margins[i, j] = item["margin"]
                if item["pct"] is not None:
                    pcts[i, j] = item["pct"]
        spans = [(env["a"], env["b"]) for run in self.runs for env in run.summary["series"].values()]
        span = [min(a for a, _ in spans), max(b for _, b in spans)] if spans else [0.0, 1.0]
        if not span[1] > span[0]:
            span[1] = span[0] + 1.0
        series: list[dict] = []
        series_index: dict[str, int] = {}
        reqs = []
        for j, (rid, req) in enumerate(zip(self.ids, self.requirements)):
            stat = self.stats[rid]
            entry = {"id": rid, "title": req.title, "kind": req.kind or "", "check": req.check,
                     "counts": [stat["counts"][v] for v in VERDICTS], "verdict": stat["verdict"],
                     "sentence": self.sentence(rid), "worst": stat.get("worst"),
                     "margin": packer.uint16(margins[:, j]), "unit": (stat.get("worst") or {}).get("unit")}
            scale = _common_scale(margins[:, j], pcts[:, j])
            if scale is not None:
                entry["pct_scale"] = scale  # one limit for every run: the percentage is margin / |limit|
            else:
                entry["pct"] = packer.uint16(pcts[:, j])
            overlay = self._overlay(rid, j, codes, margins, span, bins, packer, series, series_index)
            if overlay is not None:
                entry["plot"] = overlay
            reqs.append(entry)
        for entry in series:
            pending = entry.pop("pending")
            order = sorted(pending)
            entry["trace_runs"] = order
            if order:
                entry["trace_lo"] = packer.uint8q(np.concatenate([pending[i][0] for i in order]), *entry["bounds"])
                entry["trace_hi"] = packer.uint8q(np.concatenate([pending[i][1] for i in order]), *entry["bounds"])
        runs = []
        for run in self.runs:
            s = run.summary
            runs.append({"id": run.id, "status": run.status, "page": s["page"],
                         "file": f"runs/{run.id}.json", "params": s["params"], "error": s.get("error"),
                         "counts": [s["counts"].get(v, 0) for v in VERDICTS]})
        params = []
        for name in self.param_names:
            values = [run.summary["params"].get(name) for run in self.runs]
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v not in
                          (None, ""))
            params.append({"name": name, "numeric": numeric})
        return {"span": span, "bins": bins, "runs": runs, "reqs": reqs, "series": series, "params": params,
                "matrix": packer.uint8(codes.reshape(-1)), "t0_event": self.reqset.config.time.t0}

    def _overlay(self, rid, j, codes, margins, span, bins, packer, series, series_index) -> dict | None:
        refs = [(i, run.summary["plots"].get(rid)) for i, run in enumerate(self.runs)]
        refs = [(i, ref) for i, ref in refs if ref and ref.get("s") in self.runs[i].summary["series"]]
        if not refs:
            return None
        key = refs[0][1]["s"]
        refs = [(i, ref) for i, ref in refs if ref["s"] == key]
        rows = [i for i, _ in refs]
        lo, hi = self._resample([self.runs[i].summary["series"][key] for i in rows], span, bins)
        if key not in series_index:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mid = (lo + hi) / 2
                bands = {"min": np.nanmin(lo, axis=0), "p05": np.nanpercentile(mid, 5, axis=0),
                         "p50": np.nanpercentile(mid, 50, axis=0), "p95": np.nanpercentile(mid, 95, axis=0),
                         "max": np.nanmax(hi, axis=0)}
            finite = np.concatenate([b[np.isfinite(b)] for b in bands.values()])
            bounds = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
            first = self.runs[rows[0]].summary["series"][key]
            series_index[key] = len(series)
            series.append({"name": first["name"], "unit": first["unit"], "labels": first["labels"],
                           "runs": len(rows), "bounds": list(bounds), "pending": {},
                           "bands": {k: packer.uint16(v, *bounds) for k, v in bands.items()}})
        entry = series[series_index[key]]
        # the runs to draw one by one: failures first, then warnings, then the smallest margins
        order = sorted(range(len(rows)), key=lambda k: (
            {2: 0, 1: 1, 4: 2}.get(int(codes[rows[k], j]), 3),
            margins[rows[k], j] if np.isfinite(margins[rows[k], j]) else math.inf, rows[k]))
        traces = []
        for k in order:
            if len(traces) >= OVERLAY_TRACES:
                break
            if not (np.isfinite(lo[k]).any() or np.isfinite(hi[k]).any()):
                continue
            code = int(codes[rows[k], j])
            stored = rows[k] in entry["pending"]
            # a run's trace is stored once per signal; past the budget only failures add new ones
            if not stored and len(entry["pending"]) >= SERIES_TRACES and code != CODES["fail"]:
                continue
            entry["pending"].setdefault(rows[k], (lo[k], hi[k]))
            traces.append({"run": rows[k], "v": code})
        overlay = {"series": series_index[key], "traces": traces}
        for side in ("upper", "lower"):
            scalars = sorted({ref[side] for _, ref in refs if isinstance(ref.get(side), (int, float))})
            curves = [ref[side] for _, ref in refs if isinstance(ref.get(side), Mapping)]
            if scalars:
                overlay[side] = scalars[:DISTINCT_LIMITS]
            if curves:
                clo, chi = self._resample(curves, span, bins)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    tight = np.nanmin(clo, axis=0) if side == "upper" else np.nanmax(chi, axis=0)
                overlay[f"{side}_curve"] = packer.uint16(tight)
        return overlay

    @staticmethod
    def _resample(envs, span, bins) -> tuple[np.ndarray, np.ndarray]:
        """Run envelopes (each on its own time axis) as rows of min and max on the shared axis."""
        count = len(envs)
        lo = np.full((count, bins), np.nan)
        hi = np.full((count, bins), np.nan)
        width = (span[1] - span[0]) / bins
        for r, env in enumerate(envs):
            elo, ehi = _dequantize(env)
            k = elo.shape[0]
            centers = env["a"] + (np.arange(k) + 0.5) * (env["b"] - env["a"]) / k
            target = np.clip(((centers - span[0]) / width).astype(np.int64), 0, bins - 1)
            np.fmin.at(lo[r], target, elo)
            np.fmax.at(hi[r], target, ehi)
        return lo, hi

    # ----- terminal ---------------------------------------------------------------------------------------

    def render(self, outputs: Mapping[str, Path], *, show_all: bool = False) -> str:
        from .results import _table

        counts = self.counts()
        n = counts["runs"]
        lines = [
            f"Checks   {self.reqset.table.path.name}   {len(self.ids)} requirements, "
            f"{counts['not_covered']} not covered",
            f"Runs     {n} runs in {self.root}   {self.jobs} process{'es' if self.jobs != 1 else ''}, "
            f"{self.seconds:.1f} s, {counts['cached']} up to date",
        ]
        shown = [rid for rid in self.ids if show_all or self.stats[rid]["verdict"] != "pass"]
        shown.sort(key=lambda rid: (SEVERITY[self.stats[rid]["verdict"]], -self.stats[rid]["counts"]["fail"]))
        if shown:
            rows = [("VERDICT", "ID", "TITLE", "RUNS", "WORST", "LIMIT", "MARGIN", "RUN")]
            for rid, req in ((rid, self.requirements[self.ids.index(rid)]) for rid in shown):
                stat = self.stats[rid]
                verdict = stat["verdict"]
                c = stat["counts"]
                amount = c[verdict] if verdict != "pass" else c["pass"]
                worst = stat.get("worst") or {}
                margin = "-"
                if worst.get("margin") is not None:
                    margin = fmt_value(worst["margin"], worst["unit"])
                    if worst.get("pct") is not None:
                        margin += f" ({worst['pct'] * 100:.1f} %)"
                value = fmt_value(worst.get("value"), worst.get("unit")) if worst else "-"
                if verdict == "error" and stat["reason"]:
                    value, margin = stat["reason"][:60], ""
                rows.append((VERDICT_LABELS[verdict], rid, req.title[:28], f"{amount}/{n}", value,
                             worst.get("limit") or "-", margin, worst.get("run", "")))
            lines += [""] + _table(rows)
        broken = [run for run in self.runs if run.error]
        if broken:
            lines.append("")
            for run in broken[:10]:
                lines.append(f"ERROR    run {run.id}: {run.error}")
            if len(broken) > 10:
                lines.append(f"         and {len(broken) - 10} more runs could not be checked")
        parts = ", ".join(f"{counts[v]} {label}" for v, label in (("pass", "pass"), ("warn", "warn"),
                                                                   ("fail", "fail"), ("error", "error"),
                                                                   ("not_applicable", "n/a")))
        lines += ["", f"Result: {VERDICT_LABELS.get(self.status, self.status)} ({n} runs: {parts})"]
        order = ("report", "xlsx", "annotated", "json", "index")
        paths = [Path(outputs[key]) for key in order if key in outputs]
        if paths:
            folder = self.output
            names = [str(p.relative_to(folder)) if folder in p.parents else str(p) for p in paths]
            lines.append(f"Output   {folder}{os.sep}  {', '.join(names)}, runs{os.sep}")
        return "\n".join(lines) + "\n"


def _common_scale(margins: np.ndarray, pcts: np.ndarray) -> float | None:
    """The factor with pct == margin * factor for every run, when there is one."""
    both = np.isfinite(margins) & np.isfinite(pcts)
    if not both.any() or np.any(np.isfinite(margins) != np.isfinite(pcts)):
        return None
    nonzero = both & (margins != 0)
    if not nonzero.any():
        return None
    factor = float(np.median(pcts[nonzero] / margins[nonzero]))
    if not np.allclose(pcts[both], margins[both] * factor, rtol=1e-9, atol=1e-12):
        return None
    return factor


def _clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value
