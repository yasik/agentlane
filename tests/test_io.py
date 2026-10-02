"""The stream contracts require no paths, seeking, or line operations."""

from io import BytesIO

import pytest

from agentlane.io import buffered_reader, read_all, write_all


class ShortReader:
    def __init__(self, content: bytes) -> None:
        self.source = BytesIO(content)

    def read(self, size: int = -1, /) -> bytes:
        # Even an unbounded request may need several reads to reach EOF.
        return self.source.read(min(size, 3) if size >= 0 else 3)


class ShortWriter:
    def __init__(self) -> None:
        self.content = bytearray()

    def write(self, data: bytes, /) -> int:
        self.content.extend(data[:3])
        return min(3, len(data))


def test_buffering_adds_line_reads_without_owning_source() -> None:
    source = ShortReader(b"one\r\ntwo\nlast")

    # Interleave line and byte reads across the source's short-read boundaries.
    with buffered_reader(source) as buffered:
        assert buffered.readline() == b"one\r\n"
        assert buffered.read(2) == b"tw"
        assert buffered.readline() == b"o\n"
        assert buffered.read() == b"last"

    assert not source.source.closed


def test_complete_transfers_handle_short_reads_and_writes() -> None:
    content = b"abc\x00\xff" * 15000
    result = read_all(ShortReader(content))
    assert result == content

    writer = ShortWriter()
    write_all(writer, result)
    assert writer.content == content


@pytest.mark.parametrize("count", [0, -1, 100])
def test_write_rejects_invalid_progress(count: int) -> None:
    # Reject both stalled writes and counts outside the supplied byte range.
    class InvalidWriter:
        def write(self, data: bytes, /) -> int:
            return count

    with pytest.raises(OSError, match="invalid byte count"):
        write_all(InvalidWriter(), b"content")
