"""LiteLLM wrapper with enhanced functionality.

This module provides a wrapper around the LiteLLM client that allows for
easier configuration and usage of the LiteLLM client. It also provides a
factory for creating LiteLLM clients.
"""

from ._client import Client, Factory

__all__ = [
    "Client",
    "Factory",
]
