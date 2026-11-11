"""Revision-checked storage keeps concurrent patches from overwriting each other."""

import asyncio
from contextlib import AbstractContextManager
from typing import cast

import pytest

from agentlane.harness._storage_write import write_content
from agentlane.harness.filesystem import (
    BinaryReader,
    FileRevisionConflictError,
    MountedFileSystem,
    SkillReader,
)
from agentlane.harness.tools import patch_tool
from agentlane.io import read_all

from ..tools_test_utils import run_tool
from .conftest import MemoryFileSystem, RevisionedMemoryFileSystem

_REPORT = (
    '{"causes": [\n  {"name": "A", "text": ""},\n  {"name": "B", "text": ""}\n]}\n'
)


def _edit(name: str, text: str) -> str:
    return (
        "<<<<<<< SEARCH\n"
        f'  {{"name": "{name}", "text": ""}}\n'
        "=======\n"
        f'  {{"name": "{name}", "text": "{text}"}}\n'
        ">>>>>>> REPLACE"
    )


def test_patch_revision_conflict_rereads_and_keeps_both_edits(
    revisioned_storage: RevisionedMemoryFileSystem,
) -> None:
    revisioned_storage.files["report.json"] = _REPORT.encode()
    other = patch_tool(reader=revisioned_storage, writer=revisioned_storage)

    # The other writer commits its slot after this patch read the file.
    revisioned_storage.before_checked_write.append(
        lambda: run_tool(other, path="report.json", edits=_edit("B", "second"))
    )
    result = run_tool(
        patch_tool(reader=revisioned_storage, writer=revisioned_storage),
        path="report.json",
        edits=_edit("A", "first"),
    )

    content = revisioned_storage.files["report.json"].decode()
    assert result == "Applied 1 edit to report.json."
    assert '"text": "first"' in content
    assert '"text": "second"' in content
    assert revisioned_storage.conflicts == 1


def test_patch_revision_conflict_on_every_attempt_reports_changed_file(
    revisioned_storage: RevisionedMemoryFileSystem,
) -> None:
    revisioned_storage.files["report.json"] = _REPORT.encode()

    def change_file() -> None:
        revisioned_storage.write("report.json", _REPORT.encode())

    revisioned_storage.before_checked_write.extend([change_file] * 3)
    result = run_tool(
        patch_tool(reader=revisioned_storage, writer=revisioned_storage),
        path="report.json",
        edits=_edit("A", "first"),
    )

    assert result == (
        "File changed while it was patched: report.json. "
        "Read it again and repeat the patch."
    )
    assert revisioned_storage.files["report.json"] == _REPORT.encode()
    assert revisioned_storage.conflicts == 3


def test_patch_without_revisions_writes_without_a_check(
    storage: MemoryFileSystem,
) -> None:
    storage.files["report.json"] = _REPORT.encode()

    result = run_tool(
        patch_tool(reader=storage, writer=storage),
        path="report.json",
        edits=_edit("A", "first"),
    )

    assert result == "Applied 1 edit to report.json."
    assert '"text": "first"' in storage.files["report.json"].decode()


def test_mounted_read_revision_reports_child_revision_or_none(
    storage: MemoryFileSystem, revisioned_storage: RevisionedMemoryFileSystem
) -> None:
    storage.files["plain.txt"] = b"plain"
    revisioned_storage.files["checked.txt"] = b"checked"
    mounted = MountedFileSystem(
        {
            "plain": cast(SkillReader, storage),
            "checked": cast(SkillReader, revisioned_storage),
        }
    )

    def read(
        opened: AbstractContextManager[tuple[BinaryReader, str | None]],
    ) -> tuple[bytes, str | None]:
        with opened as (source, revision):
            return read_all(source), revision

    assert read(mounted.open_read_revision("/plain/plain.txt")) == (b"plain", None)
    assert read(mounted.open_read_revision("/checked/checked.txt")) == (
        b"checked",
        "0",
    )


def test_mounted_write_revision_requires_a_revisioned_mount(
    storage: MemoryFileSystem,
) -> None:
    mounted = MountedFileSystem({"plain": cast(SkillReader, storage)})

    with pytest.raises(PermissionError):
        mounted.open_write_revision("/plain/notes.txt", "0")


def test_revision_write_after_file_changes_inside_open_writer_conflicts(
    revisioned_storage: RevisionedMemoryFileSystem,
) -> None:
    revisioned_storage.files["report.json"] = _REPORT.encode()
    with revisioned_storage.open_read_revision("report.json") as (_, revision):
        assert revision is not None

    # The check belongs to the commit, so a change after the writer opens still
    # fails the write and keeps the competing content.
    with pytest.raises(FileRevisionConflictError):
        with revisioned_storage.open_write_revision("report.json", revision) as stream:
            stream.write(b"stale replacement")
            revisioned_storage.write("report.json", b"competing")

    assert revisioned_storage.files["report.json"] == b"competing"


def test_patch_revision_reader_with_unchecked_writer_is_rejected(
    storage: MemoryFileSystem, revisioned_storage: RevisionedMemoryFileSystem
) -> None:
    with pytest.raises(ValueError, match="writer must implement FileRevisions"):
        patch_tool(reader=revisioned_storage, writer=storage)


def test_patch_max_attempts_one_returns_conflict_without_replay(
    revisioned_storage: RevisionedMemoryFileSystem,
) -> None:
    revisioned_storage.files["report.json"] = _REPORT.encode()

    def change_file() -> None:
        revisioned_storage.write("report.json", _REPORT.encode())

    revisioned_storage.before_checked_write.extend([change_file] * 3)
    result = run_tool(
        patch_tool(
            reader=revisioned_storage, writer=revisioned_storage, max_attempts=1
        ),
        path="report.json",
        edits=_edit("A", "first"),
    )

    assert result == (
        "File changed while it was patched: report.json. "
        "Read it again and repeat the patch."
    )
    assert revisioned_storage.conflicts == 1


def test_patch_max_attempts_below_one_is_rejected(
    revisioned_storage: RevisionedMemoryFileSystem,
) -> None:
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        patch_tool(reader=revisioned_storage, writer=revisioned_storage, max_attempts=0)


def test_revision_write_with_unchecked_writer_raises_type_error(
    storage: MemoryFileSystem,
) -> None:
    with pytest.raises(TypeError, match="FileRevisions"):
        asyncio.run(write_content(storage, "report.json", b"x", revision="0"))
