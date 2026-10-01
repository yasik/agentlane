"""File I/O contracts and local storage behavior."""

import os
from pathlib import Path, PurePosixPath

import pytest

from agentlane.harness.filesystem import (
    FileReader,
    FileWriter,
    LocalFileSystem,
    SkillReader,
    normalize_relative_path,
)
from agentlane.harness.tools import (
    PathScopeToolPermissionPolicy,
    ToolOperation,
    ToolPermissionRequest,
    WorkspaceToolPermissionPolicy,
)


@pytest.mark.parametrize(
    ("path", "root", "expected"),
    [
        ("references/a.md", "skills/example", "skills/example/references/a.md"),
        ("../other/a.md", "skills/example", "skills/other/a.md"),
        ("./assets/../a.md", ".", "a.md"),
        ("~user/file.md", ".", "~user/file.md"),
        (".", ".", "."),
    ],
)
def test_relative_path_valid_input_resolves_without_host_paths(
    path: str,
    root: str,
    expected: str,
) -> None:
    resolved = normalize_relative_path(path, root=root)

    assert resolved == PurePosixPath(expected)
    assert not isinstance(resolved, Path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "  ",
        "/etc/passwd",
        "../escape",
        "a/../../escape",
        "C:/secret",
        "C:secret",
        "a\\b",
        "a\x00b",
    ],
)
def test_relative_path_invalid_input_rejects_escape(path: str) -> None:
    with pytest.raises(ValueError):
        normalize_relative_path(path)


def test_relative_path_invalid_root_rejects_even_if_join_returns_inside() -> None:
    with pytest.raises(ValueError):
        normalize_relative_path("inside/file", root="../outside")


def test_local_filesystem_round_trip_preserves_bytes_and_protocols(
    tmp_path: Path,
) -> None:
    storage = LocalFileSystem(tmp_path)
    writer: FileWriter = storage
    reader: FileReader = storage
    skills: SkillReader = storage
    content = b"hello\r\n\xff\x00"

    writer.write("notes/raw.bin", content)
    with reader.open_read("notes/raw.bin") as stream:
        assert stream.read() == content

    assert writer.stat("missing") is None
    directory = writer.stat("notes")
    assert directory is not None and directory.is_directory
    file_info = writer.stat("notes/raw.bin")
    assert file_info is not None and not file_info.is_directory
    assert [entry.name for entry in skills.list_directory("notes")] == ["raw.bin"]


def test_local_filesystem_failed_replace_preserves_original_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalFileSystem(tmp_path)
    storage.write("existing.txt", b"original")

    def fail_replace(self: Path, target: str | Path) -> Path:
        del self, target
        raise PermissionError("replacement blocked")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError):
        storage.write("existing.txt", b"replacement")

    assert (tmp_path / "existing.txt").read_bytes() == b"original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing.txt"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires named pipes")
def test_local_filesystem_named_pipe_is_not_opened_or_listed_as_a_file(
    tmp_path: Path,
) -> None:
    os.mkfifo(tmp_path / "SKILL.md")
    storage = LocalFileSystem(tmp_path)

    with pytest.raises(OSError, match="not a regular file"):
        storage.open_read("SKILL.md")

    assert storage.list_directory(".") == ()


def test_local_filesystem_captured_root_survives_working_directory_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(original)
    storage = LocalFileSystem()
    monkeypatch.chdir(elsewhere)

    storage.write("target.txt", b"content")

    assert (original / "target.txt").read_bytes() == b"content"
    assert not (elsewhere / "target.txt").exists()


def test_local_policies_injected_path_reject_without_host_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies = (
        WorkspaceToolPermissionPolicy(tmp_path),
        PathScopeToolPermissionPolicy((tmp_path,)),
    )
    request = ToolPermissionRequest(
        tool_name="read",
        operation=ToolOperation.READ_FILE,
        cwd=PurePosixPath("."),
        path=PurePosixPath("skills/guide.md"),
    )

    def unexpected_resolve(self: Path, strict: bool = False) -> Path:
        del self, strict
        raise AssertionError("injected paths must not resolve against the host")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)
    for policy in policies:
        assert not policy.check(request).allowed
