"""Shared pytest configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make tests/reference importable as `reference.<module>` from any test file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from hypothesis import HealthCheck, settings
except ImportError:  # minimal environments run without hypothesis
    pass
else:
    settings.register_profile("dev", max_examples=50, deadline=None)
    settings.register_profile(
        "ci", max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
