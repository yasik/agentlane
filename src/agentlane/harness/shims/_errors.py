"""Errors raised while built-in shims prepare a model turn."""


class ToolNameCollisionError(ValueError):
    """Two tool contributions use a name that requires a unique owner."""
