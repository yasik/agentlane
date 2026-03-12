"""Time-related utility helpers."""

from time import time


def utc_now_ms() -> int:
    """Return the current UTC epoch time in milliseconds."""
    return int(time() * 1000)
