# pylint: disable=W0603

"""Global trace provider management."""

from collections.abc import Callable
from typing import Any

_global_trace_provider: Any | None = None
"""The global trace provider."""
_trace_provider_factory: Callable[[], Any] | None = None
"""Factory used to lazily create the default global trace provider."""


def set_trace_provider(provider: Any) -> None:
    """Set the global trace provider.

    Args:
        provider: The trace provider to use globally.
    """
    global _global_trace_provider
    _global_trace_provider = provider


def set_trace_provider_factory(factory: Callable[[], Any]) -> None:
    """Set the factory used for lazy default provider creation."""
    global _trace_provider_factory
    _trace_provider_factory = factory


def get_trace_provider() -> Any:
    """Get the global trace provider.

    Returns:
        The global trace provider.

    Raises:
        RuntimeError: If no provider is configured.
    """
    if _global_trace_provider is None:
        if _trace_provider_factory is None:
            raise RuntimeError("Trace provider is not configured")
        set_trace_provider(_trace_provider_factory())
    if _global_trace_provider is None:
        raise RuntimeError("Trace provider is not configured")
    return _global_trace_provider
