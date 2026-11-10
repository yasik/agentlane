"""Revision-checked storage keeps concurrent patches from overwriting each other."""

from contextlib import AbstractContextManager
from typing import cast

import pytest

from agentlane.harness.filesystem import (
    BinaryReader,
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
