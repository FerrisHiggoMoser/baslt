"""Plot data of the run report: overview traces keep what matters, and the page stays small."""

from __future__ import annotations

import base64
import json
import re
import zlib

import numpy as np
import pytest

from baslt.report.html import Packer
from baslt.reqs.plotdata import build_plots, overview_indices
from baslt.reqs.report import render_run_report
from baslt.reqs.run import check_run
from reference.reqs_check import load_rows
from reference.reqs_runs import run as make_run

pytestmark = pytest.mark.minimal


def test_short_series_keep_every_sample():
    assert overview_indices(np.arange(1024.0)).tolist() == list(range(1024))


def test_long_series_keep_extremes_ends_and_gap_starts():
    rng = np.random.default_rng(5)
    x = rng.normal(size=100_000)
    x[31_337] = 50.0
    x[77_000] = -50.0
    x[50_000:50_010] = np.nan
    x[90_000] = np.nan
    keep = overview_indices(x)
    assert keep.shape[0] <= 2 * 512 + 2 + 256
    assert {0, 99_999, 31_337, 77_000, 50_000, 90_000} <= set(keep.tolist())
    assert np.all(np.diff(keep) > 0)
    finite = np.isfinite(x)
    for lo in range(0, 100_000, 10_000):  # every tenth of the run keeps its own extremes
        part = x[lo:lo + 10_000]
        kept = x[keep[(keep >= lo) & (keep < lo + 10_000)]]
        assert np.nanmax(kept) == np.max(part[finite[lo:lo + 10_000]])


def test_many_gaps_keep_the_longest():
    x = np.zeros(20_000)
    x[1::4] = np.nan  # 5000 one-sample gaps
    x[10_000:10_500] = np.nan
    keep = overview_indices(x)
    assert 10_000 in keep.tolist()
    assert np.count_nonzero(np.isnan(x[keep])) <= 256


T = np.linspace(0.0, 100.0, 100_001)
Q = 50.0 + 10.0 * np.sin(T / 10.0)
Q[60_000:60_100] = 80.0
MAPPING = {"events": {"go": {"signal": "q", "rises_above": 55}}, "time": {"t0": "go"},
           "conditions": {"early": "t < 50 s"},
           "signals": {"mode": {"path": "mode", "kind": "discrete", "labels": {0: "IDLE", 1: "BURN"}}}}
ROWS = [
    {"id": "A", "check": "q", "limit": "<= 70"},
    {"id": "B", "check": "q", "limit": "<= 75", "when": "early"},
    {"id": "C", "check": "max(q)", "limit": "<= 100"},
    {"id": "D", "check": "q > 55", "limit": "<= 60 s"},
    {"id": "E", "check": "go", "limit": "0 .. 10 s"},
    {"id": "F", "check": "mode == 'BURN'", "type": "assert", "when": "t > 90 s"},
    {"id": "G", "check": "time('go')", "limit": "<= 10 s"},
]


@pytest.fixture(scope="module")
def checked():
    signals = {"q": (T, Q), "mode": (T[::1000], (T[::1000] > 80).astype(int), None, "discrete")}
    reqset = load_rows(ROWS, MAPPING)
    result, ctx = check_run(make_run(**signals), reqset)
    return result, ctx, reqset


def test_plots_for_each_kind(checked):
    run, ctx, _ = checked
    packer = Packer()
    plots = build_plots(run, ctx, packer)
    specs = plots["plots"]
    assert set(specs) == {"A", "B", "C", "D", "E", "F"}  # G reads no signal
    assert specs["A"]["series"] == specs["B"]["series"] == specs["C"]["series"]  # one q trace, shared
    assert len(plots["series"]) == 2 and plots["series"][1]["labels"] == [[0, "IDLE"], [1, "BURN"]]
    assert [s["name"] for s in plots["series"]] == ["q", "mode"]
    assert plots["base"] == run.t0 and plots["span"] == [pytest.approx(-run.t0), pytest.approx(100 - run.t0)]
    a = specs["A"]
    assert a["limits"][0]["step"] is True and a["limits"][0]["side"] == "upper"
    assert a["marker"]["v"] == 80.0 and len(a["details"]) == 1 and a["spans"][0]["tol"] is False
    assert specs["B"]["details"] == [] and specs["B"]["windows"][0][0] == pytest.approx(-run.t0)
    assert specs["C"]["lines"] == [{"kind": "value", "v": 80.0}, {"kind": "upper", "v": 100.0}]
    assert specs["C"]["marker"]["t"] == pytest.approx(60.0 - run.t0)
    assert specs["D"]["lines"] == [{"kind": "threshold", "v": 55.0}] and specs["D"]["spans"][0]["held"]
    assert specs["E"]["band"] == [pytest.approx(-run.t0), pytest.approx(10 - run.t0)]
    assert specs["F"]["series"] == [1] and specs["F"]["spans"] == []
    assert [p["name"] for p in plots["phases"]] == ["early"]
    assert plots["events"][0]["name"] == "go" and plots["events"][0]["times"][0] == 0.0  # times from T+0
    assert len(plots["events"][0]["times"]) == run.events["go"]["count"] == 3


def test_the_page_is_small_and_complete(checked):
    run, ctx, reqset = checked
    packer = Packer()
    text = render_run_report(run, ctx, reqset, packer=packer)
    assert len(text.encode()) < 120_000
    manifest = json.loads(re.search(r'<script type="application/json" id="manifest">(.*?)</script>', text).group(1))
    blob = re.search(r'<script type="application/octet-stream" id="blob">(.*?)</script>', text).group(1)
    assert len(zlib.decompress(base64.b64decode(blob), -15)) == packer.size
    assert set(manifest["plots"]) == {"A", "B", "C", "D", "E", "F"}
    for rid in "ABCDEFG":
        assert f'id="req-{rid}"' in text
    assert text.count('class="plot" data-req=') == 6
    assert render_run_report(run, ctx, reqset) == text  # the same run gives the same page


def test_two_hundred_requirements_fit_the_budget():
    t = np.linspace(0.0, 120.0, 120_001)
    rng = np.random.default_rng(11)
    signals = {f"s{k}": (t, rng.normal(size=t.shape[0]).cumsum() * 0.01) for k in range(40)}
    rows = []
    for k in range(200):
        limit = "<= 0.5" if k % 20 == 0 else "<= 1000"
        rows.append({"id": f"R-{k:03d}", "check": f"s{k % 40}", "limit": limit,
                     "when": f"t > {k % 7} s"})
    reqset = load_rows(rows)
    result, ctx = check_run(make_run(**signals), reqset)
    assert sum(r.verdict == "fail" for r in result.results) >= 5
    text = render_run_report(result, ctx, reqset)
    assert len(text.encode()) <= 700_000
