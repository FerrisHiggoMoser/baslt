"""Reading policies from YAML, JSON or a mapping.

YAML is read with `yaml.SafeLoader` (the optional `baslt[yaml]` extra). The composed node tree
gives a "file:line:col" location for every mapping key and sequence item, which validation and
binding attach to their messages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ..errors import Issue, PolicyError, UsageError
from .schema import Policy
from .validate import MAX_ISSUES, item_path, join_path, validate_policy

_EXTENSIONS = {".yaml": "yaml", ".yml": "yaml", ".json": "json"}
_FORMATS = {"yaml": "yaml", "yml": "yaml", "json": "json", "dict": "dict"}


def load_policy(source: str | Path | Mapping, *, format: str | None = None) -> Policy:
    """Parse, validate and hash a policy.

    `source` is a file path (format taken from the .yaml/.yml/.json extension unless given), a
    raw YAML or JSON text when `format` is given with a str, or a mapping.
    """
    fmt = None
    if format is not None:
        fmt = _FORMATS.get(format.lower()) if isinstance(format, str) else None
        if fmt is None:
            raise UsageError(f"unknown policy format {format!r}; expected yaml or json")

    if isinstance(source, Mapping):
        if fmt not in (None, "dict"):
            raise UsageError(f"format={format!r} does not apply to a mapping policy")
        return validate_policy(source, source_format="dict", source_text=None, locations={})

    if isinstance(source, Path) or (isinstance(source, str) and fmt is None):
        path = Path(source)
        if fmt is None:
            fmt = _EXTENSIONS.get(path.suffix.lower())
            if fmt is None:
                if isinstance(source, str) and ("\n" in source or source.lstrip().startswith("{")):
                    raise UsageError("policy text needs format='yaml' or format='json'")
                raise UsageError(
                    f"cannot tell the policy format from {str(source)!r}; use a .yaml, .yml or .json file"
                )
        if fmt == "dict":
            raise UsageError("format='dict' needs a mapping, not a file")
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise UsageError(f"policy file not found: {path}") from None
        except IsADirectoryError:
            raise UsageError(f"policy path is a directory: {path}") from None
        except UnicodeDecodeError as exc:
            raise PolicyError(f"policy file {path} is not valid UTF-8: {exc}") from None
        return _load_text(text, fmt, str(source))

    if isinstance(source, str):
        if fmt == "dict":
            raise UsageError("format='dict' needs a mapping, not a string")
        return _load_text(source, fmt, "<string>")

    raise UsageError(f"a policy must be a file path, YAML/JSON text or a mapping, got {type(source).__name__}")


def _load_text(text: str, fmt: str, filename: str) -> Policy:
    if fmt == "yaml":
        data, locations = parse_yaml(text, filename)
    else:
        data, locations = parse_json(text, filename), {}
    return validate_policy(data, source_format=fmt, source_text=text, locations=locations)


def parse_json(text: str, filename: str = "<string>") -> object:
    """Parse JSON text; duplicate keys and NaN/Infinity tokens are errors."""

    def pairs(items: list[tuple[str, object]]) -> dict:
        out: dict = {}
        for key, value in items:
            if key in out:
                raise PolicyError([Issue("", f"duplicate key {key!r}", filename)])
            out[key] = value
        return out

    def constant(token: str) -> object:
        raise PolicyError([Issue("", f"invalid JSON: {token} is not allowed", filename)])

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as exc:
        raise PolicyError([Issue("", f"invalid JSON: {exc.msg}", f"{filename}:{exc.lineno}:{exc.colno}")]) from None


def parse_yaml(text: str, filename: str = "<string>") -> tuple[object, dict[str, str]]:
    """Parse YAML with SafeLoader and return (data, locations)."""
    try:
        import yaml
    except ImportError:
        raise UsageError("reading YAML policies needs PyYAML; install baslt[yaml]") from None

    loader = yaml.SafeLoader(text)
    locations: dict[str, str] = {}
    try:
        issues: list[Issue] = []
        try:
            node = loader.get_single_node()
            if node is not None:
                _walk(node, "", filename, locations, issues, set())
            if issues:
                raise PolicyError(issues)
            data = loader.construct_document(node) if node is not None else None
        except yaml.YAMLError as exc:
            raise PolicyError([_yaml_issue(exc, filename)]) from None
        except RecursionError:
            # Composing, walking and constructing all recurse; a deeply nested document must still
            # leave as a BasltError with an exit code, not a traceback.
            raise PolicyError([Issue("", "policy is nested too deeply", filename)]) from None
    finally:
        loader.dispose()
    return data, locations


def _mark(mark: object, filename: str) -> str:
    return f"{filename}:{mark.line + 1}:{mark.column + 1}"  # type: ignore[attr-defined]


def _yaml_issue(exc: Exception, filename: str) -> Issue:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    problem = getattr(exc, "problem", None) or str(exc)
    location = _mark(mark, filename) if mark is not None else filename
    return Issue("", f"invalid YAML: {problem}", location)


def _walk(node: object, path: str, filename: str, locations: dict[str, str], issues: list[Issue], seen: set[int]) -> None:
    """Record a location for every mapping key and sequence item, and report duplicate keys.

    PyYAML's composer shares one node object between every use of an anchor, so a node already
    walked is skipped: its locations were recorded at the path it was first reached by, which is
    what `locations.setdefault` already means. Skipping also ends recursive aliases, and keeps
    nested aliases linear instead of costing one walk per expanded path.
    """
    import yaml

    if id(node) in seen:
        return
    if isinstance(node, yaml.MappingNode):
        seen.add(id(node))
        keys: set[str] = set()
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = str(key_node.value) if isinstance(key_node, yaml.ScalarNode) else "?"
            child = join_path(path, key)
            where = _mark(key_node.start_mark, filename)
            if isinstance(key_node, yaml.ScalarNode):
                if key in keys:
                    issues.append(Issue(child, f"duplicate key {key!r}", where))
                    if len(issues) >= MAX_ISSUES:  # the same cap validation and binding use
                        raise PolicyError(issues)
                keys.add(key)
            locations.setdefault(child, where)
            _walk(value_node, child, filename, locations, issues, seen)
    elif isinstance(node, yaml.SequenceNode):
        seen.add(id(node))
        for i, item in enumerate(node.value):
            child = item_path(path, i)
            locations.setdefault(child, _mark(item.start_mark, filename))
            _walk(item, child, filename, locations, issues, seen)
