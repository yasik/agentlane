"""Tests for the pluggable skill filesystem: loader and read tool over any store."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from agentlane.harness.skills import (
    FilesystemSkillLoader,
    LocalSkillFilesystem,
    SkillFilesystem,
    SkillFilesystemEntry,
    SkillResource,
    filesystem_read_tool,
)

from .tools_test_utils import run_tool


class _Entry(SkillFilesystemEntry):
    """One in-memory directory entry."""

    def __init__(self, name: str, is_dir: bool) -> None:
        self._name = name
        self._is_dir = is_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_dir(self) -> bool:
        return self._is_dir


class _InMemoryFilesystem(SkillFilesystem):
    """A `SkillFilesystem` over `{root: {posix_path: bytes}}`; directories are implied."""

    def __init__(self, trees: dict[str, dict[str, bytes]]) -> None:
        self._trees = trees

    async def read_bytes(self, root: str, path: str, /) -> bytes:
        files = self._trees.get(root, {})
        if path in files:
            return files[path]
        if _is_directory(files, path):
            raise IsADirectoryError(path)
        raise FileNotFoundError(path)

    async def list_dir(self, root: str, path: str, /) -> Sequence[SkillFilesystemEntry]:
        files = self._trees.get(root)
        if files is None:
            raise FileNotFoundError(root)
        if path != "" and path in files:
            raise NotADirectoryError(path)
        if path != "" and not _is_directory(files, path):
            raise FileNotFoundError(path)

        prefix = f"{path}/" if path else ""
        children: dict[str, bool] = {}
        for name in files:
            if prefix and not name.startswith(prefix):
                continue
            head, separator, _ = name[len(prefix) :].partition("/")
            if head == "":
                continue
            children[head] = children.get(head, False) or bool(separator)
        return tuple(_Entry(name, is_dir) for name, is_dir in children.items())


def _is_directory(files: dict[str, bytes], path: str) -> bool:
    """Return whether any stored file sits under `path` as a directory."""
    prefix = f"{path}/"
    return any(name.startswith(prefix) for name in files)


def _skill_bytes(name: str, description: str, *, body: str) -> bytes:
    """Render one `SKILL.md` document as bytes."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n".encode()


def test_local_skill_filesystem_lists_and_reads(tmp_path: Path) -> None:
    """The default filesystem lists directories and reads files from local disk."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.txt").write_text("hello", encoding="utf-8")
    filesystem = LocalSkillFilesystem()

    async def scenario() -> None:
        entries = await filesystem.list_dir(str(tmp_path), "")
        assert [(entry.name, entry.is_dir) for entry in entries] == [("sub", True)]
        assert await filesystem.read_bytes(str(tmp_path), "sub/file.txt") == b"hello"

        with pytest.raises(FileNotFoundError):
            await filesystem.read_bytes(str(tmp_path), "missing.txt")
        with pytest.raises(IsADirectoryError):
            await filesystem.read_bytes(str(tmp_path), "sub")
        with pytest.raises(NotADirectoryError):
            await filesystem.list_dir(str(tmp_path), "sub/file.txt")
        with pytest.raises(FileNotFoundError):
            await filesystem.list_dir(str(tmp_path), "missing-dir")

    asyncio.run(scenario())


def test_filesystem_skill_loader_over_custom_filesystem_discovers_and_loads() -> None:
    """The loader discovers and loads skills, with resource ordering preserved."""
    filesystem = _InMemoryFilesystem(
        {
            "practitioner": {
                "refund-policy/SKILL.md": _skill_bytes(
                    "refund-policy",
                    "Handle refunds.",
                    body="# Refund Policy",
                ),
                "refund-policy/scripts/run.py": b"print('ok')",
                "refund-policy/references/policy.md": b"Policy",
                "refund-policy/notes.txt": b"Notes",
            },
        }
    )
    loader = FilesystemSkillLoader(roots=("practitioner",), filesystem=filesystem)

    async def scenario() -> None:
        manifests = await loader.discover()
        assert [manifest.name for manifest in manifests] == ["refund-policy"]

        loaded = await loader.load("refund-policy")
        assert loaded.manifest.root == Path("practitioner/refund-policy")
        assert loaded.instructions == "# Refund Policy"
        assert loaded.resources == (
            SkillResource(path="scripts/run.py"),
            SkillResource(path="references/policy.md"),
            SkillResource(path="notes.txt"),
        )

    asyncio.run(scenario())


def test_filesystem_skill_loader_first_root_wins_on_name_clash() -> None:
    """An earlier root keeps a clashing skill name over a later root."""
    filesystem = _InMemoryFilesystem(
        {
            "practitioner": {
                "refund-policy/SKILL.md": _skill_bytes(
                    "refund-policy", "Practitioner copy.", body="# Practitioner"
                ),
            },
            "shared": {
                "refund-policy/SKILL.md": _skill_bytes(
                    "refund-policy", "Shared copy.", body="# Shared"
                ),
            },
        }
    )
    loader = FilesystemSkillLoader(
        roots=("practitioner", "shared"), filesystem=filesystem
    )

    async def scenario() -> None:
        manifests = await loader.discover()
        assert [manifest.name for manifest in manifests] == ["refund-policy"]

        loaded = await loader.load("refund-policy")
        assert loaded.instructions == "# Practitioner"
        assert loaded.manifest.root == Path("practitioner/refund-policy")

    asyncio.run(scenario())


def test_filesystem_skill_loader_skips_malformed_skill_over_custom_filesystem() -> None:
    """A skill whose frontmatter is invalid is skipped, not raised."""
    filesystem = _InMemoryFilesystem(
        {
            "shared": {
                "broken/SKILL.md": b"---\ndescription: missing name\n---\n\nBroken",
                "valid/SKILL.md": _skill_bytes("valid", "Valid skill.", body="# Valid"),
            },
        }
    )
    loader = FilesystemSkillLoader(roots=("shared",), filesystem=filesystem)

    async def scenario() -> None:
        manifests = await loader.discover()
        assert [manifest.name for manifest in manifests] == ["valid"]

    asyncio.run(scenario())


def test_filesystem_read_tool_reads_resource_from_matching_root() -> None:
    """The read tool returns the first root that has the resource."""
    filesystem = _InMemoryFilesystem(
        {
            "practitioner": {},
            "shared": {"refund-policy/references/policy.md": b"line1\nline2\nline3\n"},
        }
    )
    tool = filesystem_read_tool(filesystem, roots=("practitioner", "shared"))

    result = run_tool(tool, path="refund-policy/references/policy.md")

    assert result == "line1\nline2\nline3"


def test_filesystem_read_tool_windows_with_offset_and_limit() -> None:
    """Offset and limit return a bounded window with a continuation note."""
    content = "\n".join(f"line{number}" for number in range(1, 6)) + "\n"
    filesystem = _InMemoryFilesystem({"shared": {"doc.md": content.encode("utf-8")}})
    tool = filesystem_read_tool(filesystem, roots=("shared",))

    result = run_tool(tool, path="doc.md", offset=2, limit=2)

    assert result == "line2\nline3\n\n[Showing lines 2-3. Use offset=4 to continue.]"


def test_filesystem_read_tool_reports_missing_directory_and_binary() -> None:
    """Missing files, directories, and binary content each return a tool error."""
    filesystem = _InMemoryFilesystem(
        {
            "shared": {
                "dir/inner.txt": b"x",
                "bin.dat": b"\x00\x01\x02",
                "ok.txt": b"hi",
            },
        }
    )
    tool = filesystem_read_tool(filesystem, roots=("shared",))

    assert run_tool(tool, path="missing.txt") == "file not found: `missing.txt`"
    assert run_tool(tool, path="dir") == "path is a directory: `dir`"
    assert (
        run_tool(tool, path="bin.dat")
        == "file appears to be binary and cannot be read as text: `bin.dat`"
    )


def test_filesystem_read_tool_rejects_unsafe_and_invalid_arguments() -> None:
    """Absolute paths, traversal, empty paths, and bad windows are rejected."""
    filesystem = _InMemoryFilesystem({"shared": {"ok.txt": b"hi"}})
    tool = filesystem_read_tool(filesystem, roots=("shared",))

    assert (
        run_tool(tool, path="/etc/passwd")
        == "path is not a valid skill-relative path: `/etc/passwd`"
    )
    assert (
        run_tool(tool, path="../secrets")
        == "path is not a valid skill-relative path: `../secrets`"
    )
    assert run_tool(tool, path="   ") == "path must not be empty"
    assert run_tool(tool, path="ok.txt", offset=0) == (
        "offset must be a 1-indexed line number"
    )
    assert run_tool(tool, path="ok.txt", limit=0) == "limit must be greater than zero"
