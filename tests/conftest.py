"""Shared pytest setup for repo-local imports and async tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio for deterministic local behavior."""
    return "asyncio"
