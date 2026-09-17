"""Public requirement-check operations: load, lint, check and templates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .config import Config, load_config
from .model import RequirementSet

__all__ = ["init_template", "lint", "load"]


def load(requirements: str | Path, *, mapping: str | Path | Mapping | None = None,
         overrides: Mapping | None = None, only: Sequence[str] | None = None) -> RequirementSet:
    """Read a requirements table with its mapping (config sheets in the workbook, then the mapping file)."""
    from .load import load_requirements

    path = Path(requirements)
    config: Config = load_config(mapping, workbook=path, overrides=overrides)
    return load_requirements(path, config, only=only)


def lint(requirements: str | Path, *, mapping: str | Path | Mapping | None = None,
         only: Sequence[str] | None = None):
    """Check a requirements table without running anything."""
    from .lint import lint_static

    return lint_static(load(requirements, mapping=mapping, only=only))


def init_template(output: str | Path, *, template: str = "generic", mapping_output: str | Path | None = None,
                  source=None, force: bool = False) -> dict:
    from .templates import init_template as write

    return write(output, template=template, mapping_output=mapping_output, source=source, force=force)
