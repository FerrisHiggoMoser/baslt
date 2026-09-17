"""Small file helpers shared by every writer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: str | os.PathLike[str], data: bytes) -> None:
    """Replace the destination only after every byte has been written successfully."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
