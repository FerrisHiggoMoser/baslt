"""Headless Chromium for page tests: find it, dump a page's DOM after its scripts ran, take a screenshot."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CANDIDATES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")


def find_chromium() -> str | None:
    wanted = os.environ.get("BASLT_CHROMIUM")
    if wanted:
        return wanted if Path(wanted).exists() or shutil.which(wanted) else None
    for name in CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _run(binary: str, args: list[str], home: Path, timeout: float) -> subprocess.CompletedProcess:
    profile = Path(tempfile.mkdtemp(prefix="profile-", dir=home))  # a fresh profile: no lock left by an earlier run
    env = {**os.environ, "HOME": str(home)}
    base = [binary, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run", "--disable-extensions",
            f"--user-data-dir={profile}", "--virtual-time-budget=10000"]
    try:
        return subprocess.run([*base, *args], capture_output=True, text=True, timeout=timeout, env=env, check=False)
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def dump_dom(binary: str, url: str, home: Path, *, timeout: float = 90, until: str | None = None,
             attempts: int = 5) -> str:
    """The DOM after the page's scripts ran. Chromium's virtual time does not wait for stream decompression,
    so with `until` the dump is repeated until that text shows up."""
    dom = ""
    for _ in range(attempts if until else 1):
        dom = _run(binary, ["--dump-dom", url], home, timeout).stdout
        if until is None or until in dom:
            break
    return dom


def screenshot(binary: str, url: str, path: Path, home: Path, *, size: tuple[int, int] = (1280, 1600),
               dark: bool = False, timeout: float = 90) -> Path:
    args = [f"--window-size={size[0]},{size[1]}", "--hide-scrollbars", f"--screenshot={path}"]
    if dark:
        args += ["--force-dark-mode", "--blink-settings=preferredColorScheme=0"]
    _run(binary, [*args, url], home, timeout)
    return path
