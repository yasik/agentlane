"""Small byte-stream contracts with no filesystem or harness dependencies."""

from typing import Protocol


class Reader(Protocol):
    """Read bytes. Short reads are valid; empty bytes mean end of stream."""

    def read(self, size: int = -1, /) -> bytes:
        """Read up to size bytes, or the remaining data when size is negative."""
        ...


class Writer(Protocol):
    """Write bytes and return the number accepted. Short writes are valid."""

    def write(self, data: bytes, /) -> int:
        """Return a positive byte count, or raise OSError on failure."""
        ...
