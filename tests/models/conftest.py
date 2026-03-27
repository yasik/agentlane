"""Shared fixtures for model tests."""

from typing import Any

import pytest


@pytest.fixture()
def mock_output_schema() -> Any:
    """Return a minimal object that mimics OutputSchema for prompt tests."""

    class _MockOutputSchema:
        def response_format(self) -> dict[str, object]:
            """Return mock response-format data."""
            return {"type": "mock", "ok": True}

    return _MockOutputSchema()
