"""Importing baslt and its CLI must not pull in heavy dependencies."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

HEAVY = ("numpy", "h5py", "scipy", "yaml")


@pytest.mark.minimal
def test_import_does_not_load_heavy_modules():
    code = (
        "import sys, baslt, baslt.cli, baslt.errors; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


@pytest.mark.minimal
def test_version_flag():
    from baslt import __version__

    out = subprocess.run(
        [sys.executable, "-m", "baslt", "--version"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == f"baslt {__version__}"


@pytest.mark.minimal
def test_help_is_fast():
    best = min(_time_help() for _ in range(3))
    assert best < 0.5, f"baslt --help took {best:.3f}s"


def _time_help() -> float:
    start = time.perf_counter()
    subprocess.run([sys.executable, "-m", "baslt", "--help"], capture_output=True, check=True)
    return time.perf_counter() - start
