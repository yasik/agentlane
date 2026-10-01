"""Non-seekable storage and permission fixtures for native file tools."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import PurePosixPath

import pytest

from agentlane.harness.filesystem import BinaryReader, FileInfo
from agentlane.harness.tools import (
    ToolPermissionDecision,
    ToolPermissionRequest,
)


class NonSeekableReader:
    """Byte stream with short reads and no seek or iterator methods."""

    def __init__(self, content: bytes) -> None:
        self.stream = BytesIO(content)

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(min(size, 113) if size >= 0 else size)

    def readline(self, size: int = -1) -> bytes:
        return self.stream.readline(size)


class MemoryFileSystem:
    """Observe byte storage calls without using the host filesystem."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = {"."}
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.writes: list[tuple[str, bytes]] = []
        self.thread_ids: list[int] = []
        self.failure: OSError | None = None
        self.write_started = threading.Event()
        self.write_released = threading.Event()
        self.write_released.set()

    @contextmanager
    def open_read(self, path: str) -> Iterator[BinaryReader]:
        self.thread_ids.append(threading.get_ident())
        self.opened.append(path)
        if self.failure is not None:
            raise self.failure
        if path in self.directories:
            raise IsADirectoryError(path)
        if path not in self.files:
            raise FileNotFoundError(path)

        stream = NonSeekableReader(self.files[path])
        try:
            yield stream
        finally:
            stream.stream.close()
            self.closed.append(path)

    def stat(self, path: str) -> FileInfo | None:
        self.thread_ids.append(threading.get_ident())
        if path in self.directories:
            return FileInfo(is_directory=True)
        if path in self.files:
            return FileInfo(is_directory=False)
        return None

    def write(self, path: str, content: bytes) -> None:
        self.thread_ids.append(threading.get_ident())
        self.write_started.set()
        if not self.write_released.wait(timeout=5):
            raise TimeoutError("Test write was not released")
        if self.failure is not None:
            raise self.failure

        self.directories.update(str(parent) for parent in PurePosixPath(path).parents)
        self.files[path] = content
        self.writes.append((path, content))


class RecordingPolicy:
    """Record each permission check with a configurable decision."""

    def __init__(self) -> None:
        self.requests: list[ToolPermissionRequest] = []
        self.decision = ToolPermissionDecision.allow()

    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        self.requests.append(request)
        return self.decision


@pytest.fixture(name="storage")
def fixture_storage() -> MemoryFileSystem:
    return MemoryFileSystem()


@pytest.fixture(name="policy")
def fixture_policy() -> RecordingPolicy:
    return RecordingPolicy()
