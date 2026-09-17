"""Atomic writes leave complete files with the permissions a plain write would give."""

from __future__ import annotations

import os
import stat

import pytest

from baslt._io import write_atomic

pytestmark = [pytest.mark.minimal, pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")]


def test_new_files_follow_the_umask(tmp_path):
    old = os.umask(0o027)
    try:
        write_atomic(tmp_path / "a.bin", b"abc")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "a.bin").stat().st_mode) == 0o640
    assert (tmp_path / "a.bin").read_bytes() == b"abc"


def test_replaced_files_keep_their_permissions(tmp_path):
    path = tmp_path / "b.bin"
    path.write_bytes(b"old")
    path.chmod(0o600)
    write_atomic(path, b"new")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and path.read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == ["b.bin"]
