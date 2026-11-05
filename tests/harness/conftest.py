"""Shared test fixtures for harness tests."""

import shutil

import pytest


@pytest.fixture(name="jq_binary")
def fixture_jq_binary() -> str:
    """Return the installed jq executable for command parity tests."""
    executable = shutil.which("jq")
    if executable is None:
        pytest.skip("jq is not installed")

    return executable
