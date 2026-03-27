"""Shared fixtures for tracing package tests."""

from collections.abc import Iterator

import pytest

from agentlane.tracing import (
    DefaultTraceProvider,
    clear_all_collectors,
    reset_metrics_registry,
    set_trace_provider,
)


@pytest.fixture(name="reset_tracing_globals", autouse=True)
def fixture_reset_tracing_globals() -> Iterator[None]:
    """Reset tracing globals between tests."""
    reset_metrics_registry()
    clear_all_collectors()
    set_trace_provider(DefaultTraceProvider())
    yield
    reset_metrics_registry()
    clear_all_collectors()
    set_trace_provider(DefaultTraceProvider())
