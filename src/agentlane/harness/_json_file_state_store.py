"""Atomic JSON-file persistence for one harness agent snapshot."""

import json
import os
import tempfile
from pathlib import Path

from ._snapshot import AgentSnapshot
from ._state_store import StateStore


def _sync_directory(path: Path) -> None:
    """Sync one directory after a durable entry change on POSIX."""
    if os.name != "posix":
        return

    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class JsonFileStateStore(StateStore):
    """Persist one agent snapshot at a local JSON file path.

    Writes replace the file atomically. Revision checks protect the documented
    single-writer workflow from stale sequential saves; coordinating concurrent
    writers remains the application's responsibility.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> AgentSnapshot | None:
        """Load the current snapshot, or return ``None`` when absent."""
        if not self.path.exists():
            return None

        return AgentSnapshot.from_json(
            json.loads(self.path.read_text(encoding="utf-8"))
        )

    def save(
        self,
        snapshot: AgentSnapshot,
        *,
        expected_revision: int | None,
    ) -> None:
        """Atomically replace the snapshot when its stored revision matches.

        ``expected_revision=None`` requires the state file to be absent.
        """
        current = self.load()
        actual_revision = current.revision if current is not None else None
        if actual_revision != expected_revision:
            raise ValueError(
                f"State revision mismatch for {self.path}: expected "
                f"{expected_revision}, found {actual_revision}."
            )
        if expected_revision is not None and snapshot.revision <= expected_revision:
            raise ValueError(
                f"New state revision must be greater than {expected_revision}, "
                f"got {snapshot.revision}."
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    snapshot.to_json(),
                    temporary_file,
                    indent=2,
                    ensure_ascii=False,
                )
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, self.path)
            temporary_path = None
            _sync_directory(self.path.parent)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def delete(self, *, expected_revision: int | None) -> None:
        """Delete the snapshot when its stored revision matches.

        ``expected_revision=None`` requires the state file to be absent.
        """
        current = self.load()
        actual_revision = current.revision if current is not None else None
        if actual_revision != expected_revision:
            raise ValueError(
                f"State revision mismatch for {self.path}: expected "
                f"{expected_revision}, found {actual_revision}."
            )

        if current is None:
            return

        self.path.unlink()
        _sync_directory(self.path.parent)
