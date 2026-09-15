"""The `.baslt` container: array encodings, deterministic ZIP writer and checked reader.

Names are resolved lazily so importing this package does not import numpy.
"""

from __future__ import annotations

from importlib import import_module

_LAZY: dict[str, str] = {
    "ArrayDesc": "spec",
    "canonical_json": "spec",
    "header_dict": "spec",
    "header_json": "spec",
    "idx_dtype": "spec",
    "roles_dtype": "spec",
    "zstd_available": "spec",
    "container_dtype": "encode",
    "encode_array": "encode",
    "decode_array": "encode",
    "Member": "zipwriter",
    "compress_member": "zipwriter",
    "write_zip": "zipwriter",
    "zip_size": "zipwriter",
    "Artifact": "reader",
    "MemberEntry": "reader",
    "read_artifact": "reader",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'baslt.container' has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
