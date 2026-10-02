"""Byte-stream interfaces and operations independent of storage providers."""

from ._operations import buffered_reader, read_all, write_all
from ._types import Reader, Writer

__all__ = ["Reader", "Writer", "buffered_reader", "read_all", "write_all"]
