"""The batch dashboard in a real browser: its data unpacks to the same bytes and every view draws."""

from __future__ import annotations

import html
import json
import re

import pytest

from baslt.report.html import Packer
from baslt.reqs.api import check
from reference.batch_runs import write_batch
from reference.browser import dump_dom, find_chromium, screenshot

pytestmark = pytest.mark.browser

CHROMIUM = find_chromium()
if CHROMIUM is None:
    pytest.skip("needs a headless Chromium (set BASLT_CHROMIUM)", allow_module_level=True)


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    folder = tmp_path_factory.mktemp("batch")
    runs, table, params = write_batch(folder, {f"r{k}": 3.0 + 0.4 * k for k in range(9)}, gate_end={"r2": 9.5})
    result = check(runs, table, params=params, output=folder / "out", jobs=1)
    packer = Packer()
    path = result.batch.write_dashboard(folder / "out" / "index.html", packer=packer)
    return path, packer, folder


def test_the_dashboard_unpacks_and_draws(dashboard):
    path, packer, folder = dashboard
    dom = dump_dom(CHROMIUM, f"{path.as_uri()}?selftest=1", folder, until='data-done="1"')
    found = re.search(r'<pre id="selftest"[^>]*data-done="1"[^>]*>(.*?)</pre>', dom, re.S)
    assert found, dom[-2000:]
    result = json.loads(html.unescape(found.group(1)))
    assert result["errors"] == []
    assert result["digests"] == packer.digests
    assert result["reqs"] == result["drawn"] == 3 and result["runs"] == 9
    assert result["rows"] == 9 and result["ms"] < 3000


@pytest.mark.parametrize(("size", "dark"), [((1280, 1600), False), ((390, 1600), True)])
def test_screenshots(dashboard, size, dark):
    path, _, folder = dashboard
    shot = screenshot(CHROMIUM, path.as_uri(), folder / f"dash-{size[0]}.png", folder, size=size, dark=dark)
    data = shot.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 30_000
