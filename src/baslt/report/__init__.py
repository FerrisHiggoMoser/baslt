"""Self-contained HTML pages: one file that opens from disk, loads nothing from the network and carries its data.

`html` builds the page shell (content security policy, embedded JSON and a compressed binary blob). The page
scripts and styles live in `assets/`.
"""

from __future__ import annotations

__all__ = ["html"]
