"""In-memory file I/O for skill loader contract tests."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from io import BytesIO
from threading import get_ident

from agentlane.harness.filesystem import BinaryReader, DirectoryEntry


class MemorySkillReader:
    """Record file access without exposing any host filesystem paths."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.listed: list[str] = []
        self.thread_ids: list[int] = []

    @contextmanager
    def open_read(self, path: str) -> Generator[BinaryReader, None, None]:
        self.thread_ids.append(get_ident())
        self.opened.append(path)

        if path not in self.files:
            raise FileNotFoundError(path)

        try:
            with BytesIO(self.files[path]) as stream:
                yield stream
        finally:
            self.closed.append(path)

    def list_directory(self, path: str) -> Sequence[DirectoryEntry]:
        self.thread_ids.append(get_ident())
        self.listed.append(path)
        prefix = "" if path == "." else f"{path}/"
        children: dict[str, bool] = {}

        # Infer immediate directories from stored keys, as an object store would.
        for name in self.files:
            if not name.startswith(prefix):
                continue

            child, separator, _ = name.removeprefix(prefix).partition("/")
            children[child] = children.get(child, False) or bool(separator)

        if not children:
            raise FileNotFoundError(path)

        # Reverse order makes stable SDK discovery independent of storage order.
        return tuple(
            DirectoryEntry(name=name, is_directory=is_directory)
            for name, is_directory in reversed(tuple(children.items()))
        )
