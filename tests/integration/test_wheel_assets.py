"""The built wheel carries the report's scripts and styles, and nothing that does not belong to the package."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_the_wheel_has_the_report_assets(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv to build the wheel")
    done = subprocess.run([uv, "build", "--wheel", "--offline", "--out-dir", str(tmp_path), str(ROOT)],
                          capture_output=True, text=True, timeout=300, check=False)
    if done.returncode != 0:
        pytest.skip(f"the wheel could not be built offline: {done.stderr[-300:]}")
    (wheel,) = tmp_path.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    for asset in ("report.js", "report.css"):
        assert f"baslt/report/assets/{asset}" in names
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    assert all(n.startswith(("baslt/", "baslt-")) for n in names)  # only the package and its metadata
    with zipfile.ZipFile(wheel) as archive:
        packed = archive.read("baslt/report/assets/report.js")
    assert packed == (ROOT / "src/baslt/report/assets/report.js").read_bytes()
