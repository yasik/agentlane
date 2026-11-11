"""Non-seekable storage and permission fixtures for native file tools."""

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from io import BytesIO
from pathlib import PurePosixPath

import pytest

from agentlane.harness.filesystem import (
    BinaryReader,
    DirectoryEntry,
    FileInfo,
    FileRevisionConflictError,
)
from agentlane.harness.tools import (
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
)
from agentlane.io import Writer


class NonSeekableReader:
    """Byte stream with short reads and no seek or iterator methods."""

    def __init__(self, content: bytes) -> None:
        self.stream = BytesIO(content)

    def read(self, size: int = -1) -> bytes:
        # A short read must not be treated as EOF by a tool's buffering layer.
        return self.stream.read(min(size, 113) if size >= 0 else size)


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
    def open_read(self, path: str) -> Generator[BinaryReader, None, None]:
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

    def list_directory(self, path: str) -> tuple[DirectoryEntry, ...]:
        self.thread_ids.append(threading.get_ident())

        if path in self.files:
            raise NotADirectoryError(path)

        if path not in self.directories:
            raise FileNotFoundError(path)

        entries = [
            DirectoryEntry(PurePosixPath(name).name, name in self.directories)
            for name in self.directories | self.files.keys()
            if name != "." and str(PurePosixPath(name).parent) == path
        ]
        return tuple(reversed(entries))

    @contextmanager
    def open_write(self, path: str) -> Generator[Writer, None, None]:
        with BytesIO() as stream:
            yield stream

            # Publish only after normal context exit, like an atomic backend.
            self.write(path, stream.getvalue())

    def write(self, path: str, content: bytes) -> None:
        self.thread_ids.append(threading.get_ident())

        # Pause at commit so cancellation tests control the write boundary.
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


class SessionWritePolicy:
    """Allow reads and mutations within one canonical session directory."""

    def __init__(self, root: str) -> None:
        self.root = PurePosixPath(root)
        self.requests: list[ToolPermissionRequest] = []

    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        self.requests.append(request)
        if request.operation in (ToolOperation.READ_FILE, ToolOperation.SEARCH_FILES):
            return ToolPermissionDecision.allow()
        if request.path is not None and request.path.is_relative_to(self.root):
            return ToolPermissionDecision.allow()
        return ToolPermissionDecision.deny()


@pytest.fixture(name="session_policy")
def fixture_session_policy() -> SessionWritePolicy:
    return SessionWritePolicy("/workspace/sessions/one")


class RevisionedMemoryFileSystem(MemoryFileSystem):
    """Check revisions on write, like a backend with conditional replacement."""

    def __init__(self) -> None:
        super().__init__()
        self.revisions: dict[str, int] = {}
        self.conflicts = 0
        self.before_checked_write: list[Callable[[], object]] = []

    @contextmanager
    def open_read_revision(
        self, path: str
    ) -> Generator[tuple[BinaryReader, str | None], None, None]:
        revision = str(self.revisions.get(path, 0))
        with self.open_read(path) as stream:
            yield stream, revision

    @contextmanager
    def open_write_revision(
        self, path: str, revision: str
    ) -> Generator[Writer, None, None]:
        with BytesIO() as stream:
            yield stream

            # A queued action stands in for another writer that commits first.
            if self.before_checked_write:
                self.before_checked_write.pop(0)()

            if str(self.revisions.get(path, 0)) != revision:
                self.conflicts += 1
                raise FileRevisionConflictError(path)

            self.write(path, stream.getvalue())

    def write(self, path: str, content: bytes) -> None:
        super().write(path, content)
        self.revisions[path] = self.revisions.get(path, 0) + 1


@pytest.fixture(name="revisioned_storage")
def fixture_revisioned_storage() -> RevisionedMemoryFileSystem:
    return RevisionedMemoryFileSystem()
