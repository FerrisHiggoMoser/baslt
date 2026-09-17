"""The page shell: escaping, embedded JSON, packed arrays and a content security policy.

Arrays travel in one `<script type="application/octet-stream">` element: their little-endian bytes, each padded to
a multiple of 4, compressed with raw DEFLATE and base64-encoded. The page inflates them with the browser's
`DecompressionStream("deflate-raw")`. A reference to an array is a small dict:

    {"o": byte offset, "n": count, "k": "f4"}                          float32 values
    {"o": byte offset, "n": count, "k": "u2", "lo": lo, "hi": hi}      uint16 steps between lo and hi, 65535 = NaN

Scripts and styles are allowed by their SHA-256 hashes; nothing else may load, so the page works from a file and
never contacts a server.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import zlib
from collections.abc import Mapping, Sequence
from importlib import resources

import numpy as np

__all__ = ["Packer", "asset", "esc", "json_script", "page"]

NAN_CODE = 65535
STEPS = 65534


def esc(text: object) -> str:
    """Text for HTML content and attribute values."""
    return html.escape("" if text is None else str(text), quote=True)


def json_text(data: object) -> str:
    """JSON that is safe inside a <script> element."""
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def json_script(data: object, element_id: str) -> str:
    return f'<script type="application/json" id="{esc(element_id)}">{json_text(data)}</script>'


def asset(name: str) -> str:
    return resources.files("baslt.report").joinpath("assets", name).read_text(encoding="utf-8")


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


class Packer:
    """Collects arrays into one compressed blob and hands out references to them."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self._size = 0
        self.digests: list[str] = []  # sha256 of each array's bytes, in order (the page's self-test repeats them)

    def _add(self, data: bytes, count: int, kind: str, **extra) -> dict:
        ref = {"o": self._size, "n": int(count), "k": kind, **extra}
        self.digests.append(hashlib.sha256(data).hexdigest())
        pad = (-len(data)) % 4
        self._parts.append(data + b"\0" * pad)
        self._size += len(data) + pad
        return ref

    def float32(self, values) -> dict:
        array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
        return self._add(array.tobytes(), array.shape[0], "f4")

    def uint16(self, values, lo: float | None = None, hi: float | None = None) -> dict:
        """Values quantized to 65534 steps between lo and hi (their finite extremes by default)."""
        x = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(x)
        if lo is None or hi is None:
            if finite.any():
                lo, hi = float(np.min(x[finite])), float(np.max(x[finite]))
            else:
                lo, hi = 0.0, 0.0
        span = hi - lo
        codes = np.full(x.shape[0], NAN_CODE, dtype="<u2")
        if span > 0 and np.isfinite(span):
            scaled = np.rint((np.clip(x[finite], lo, hi) - lo) / span * STEPS)
            codes[finite] = scaled.astype("<u2")
        else:
            codes[finite] = 0
        return self._add(codes.tobytes(), x.shape[0], "u2", lo=_finite_or_none(lo) or 0.0,
                         hi=_finite_or_none(hi) or 0.0)

    @property
    def size(self) -> int:
        return self._size

    def blob(self) -> str:
        compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
        packed = compressor.compress(b"".join(self._parts)) + compressor.flush()
        return base64.b64encode(packed).decode("ascii")


def _hash(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


def page(title: str, body: str, *, styles: Sequence[str], scripts: Sequence[str],
         data: Mapping[str, object] | None = None, blob: str | None = None, description: str | None = None) -> str:
    """A complete HTML document. `data` becomes JSON script elements by id; `blob` is the packed arrays."""
    style = "\n".join(styles)
    script = "\n".join(scripts)
    policy = "; ".join([
        "default-src 'none'",
        f"script-src {_hash(script)}",
        f"style-src {_hash(style)}",
        "img-src data: blob:",
        "base-uri 'none'",
        "form-action 'none'",
    ])
    head = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f'<meta http-equiv="Content-Security-Policy" content="{esc(policy)}">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="color-scheme" content="light dark">',
        '<meta name="generator" content="baslt">',
    ]
    if description:
        head.append(f'<meta name="description" content="{esc(description)}">')
    head += [f"<title>{esc(title)}</title>", f"<style>{style}</style>", "</head>"]
    parts = [*head, "<body>", body]
    for element_id, value in (data or {}).items():
        parts.append(json_script(value, element_id))
    if blob is not None:
        parts.append(f'<script type="application/octet-stream" id="blob">{blob}</script>')
    parts += [f"<script>{script}</script>", "</body>", "</html>"]
    return "\n".join(parts) + "\n"
