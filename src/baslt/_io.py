"""Small file helpers shared by every writer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _mode_for(path: Path) -> int:
    """The permissions a plain open() would give: the existing file's, or 0o666 less the umask."""
    try:
        return path.stat().st_mode & 0o7777
    except OSError:
        mask = os.umask(0)
        os.umask(mask)
        return 0o666 & ~mask


def write_atomic(path: str | os.PathLike[str], data: bytes) -> None:
    """Replace the destination only after every byte has been written successfully."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        if os.name == "posix":
            os.chmod(temporary, _mode_for(path))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
