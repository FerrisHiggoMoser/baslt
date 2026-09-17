"""The run report in a real browser: its data unpacks to the same bytes, every plot draws, nothing fails."""

from __future__ import annotations

import html
import importlib.util
import json
import re
from pathlib import Path

import pytest

from baslt.report.html import Packer
from baslt.reqs.api import load
from baslt.reqs.report import render_run_report
from baslt.reqs.run import check_run
from reference.browser import dump_dom, find_chromium, screenshot

pytestmark = pytest.mark.browser

ROOT = Path(__file__).resolve().parents[2]
CHROMIUM = find_chromium()
if CHROMIUM is None:
    pytest.skip("needs a headless Chromium (set BASLT_CHROMIUM)", allow_module_level=True)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("rocket_browser_example", ROOT / "examples/rocket_sim.py")
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    reqset = load(ROOT / "examples/rocket_requirements.csv", mapping=ROOT / "examples/rocket_mapping.yaml")
    run, ctx = check_run(sim.simulate(anomaly="q_spike"), reqset, run_id="q_spike")
    packer = Packer()
    folder = tmp_path_factory.mktemp("report")
    path = folder / "report.html"
    path.write_text(render_run_report(run, ctx, reqset, packer=packer), encoding="utf-8")
    return path, packer, folder


def selftest(path: Path, home: Path, query: str = "selftest=1") -> dict:
    dom = dump_dom(CHROMIUM, f"{path.as_uri()}?{query}", home, until='data-done="1"')
    found = re.search(r'<pre id="selftest"[^>]*data-done="1"[^>]*>(.*?)</pre>', dom, re.S)
    assert found, dom[-2000:]
    return json.loads(html.unescape(found.group(1)))


def test_the_page_unpacks_and_draws_everything(report):
    path, packer, folder = report
    result = selftest(path, folder)
    assert result["errors"] == []
    assert result["digests"] == packer.digests
    assert result["plots"] == result["drawn"] == 15
    assert result["ms"] < 3000


def test_a_zoom_link_still_draws(report):
    path, packer, folder = report
    result = selftest(path, folder, "selftest=1&t=60.7,61.15")
    assert result["errors"] == [] and result["drawn"] == result["plots"]


@pytest.mark.parametrize(("size", "dark"), [((1280, 1800), False), ((390, 1800), True)])
def test_screenshots(report, size, dark):
    path, _, folder = report
    shot = screenshot(CHROMIUM, path.as_uri(), folder / f"shot-{size[0]}-{dark}.png", folder, size=size, dark=dark)
    data = shot.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 30_000
