"""Buffering and complete transfers over the small stream contracts."""

from collections.abc import Buffer
from io import BufferedReader, RawIOBase

from ._types import Reader, Writer


class _ReaderAdapter(RawIOBase):
    """Adapt a Reader to Python buffering without taking stream ownership."""

    def __init__(self, reader: Reader) -> None:
        super().__init__()
        self._reader = reader

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        # Count bytes even when the supplied buffer has a wider element type.
        view = memoryview(buffer).cast("B")
        data = self._reader.read(len(view))
        if len(data) > len(view):
            raise OSError("reader returned more bytes than requested")

        view[: len(data)] = data
        return len(data)


def buffered_reader(reader: Reader) -> BufferedReader:
    """Add standard Python buffering and line reads; the caller owns reader."""
    # Closing RawIOBase closes only the adapter, leaving provider cleanup to its owner.
    return BufferedReader(_ReaderAdapter(reader))


def read_all(reader: Reader) -> bytes:
    """Read to EOF, including streams that return short reads."""
    with buffered_reader(reader) as stream:
        return stream.read()


def write_all(writer: Writer, data: bytes) -> None:
    """Write all bytes, rejecting invalid counts and writes with no progress."""
    offset = 0

    while offset < len(data):
        # Bound each copy when a writer repeatedly accepts only a few bytes.
        chunk = data[offset : offset + 64 * 1024]
        written = writer.write(chunk)
        if written <= 0 or written > len(chunk):
            raise OSError("writer returned an invalid byte count")

        # Retry only bytes that the writer has not acknowledged.
        offset += written
