"""File I/O contracts and local storage behavior."""

import os
from pathlib import Path, PurePosixPath

import pytest

from agentlane.harness.filesystem import (
    FileReader,
    FileStat,
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
from agentlane.io import write_all


@pytest.mark.parametrize(
    ("path", "root", "expected"),
    [
        ("references/a.md", "skills/example", "skills/example/references/a.md"),
        ("../other/a.md", "skills/example", "skills/other/a.md"),
        ("./assets/../a.md", ".", "a.md"),
        ("report~backup/file.md", ".", "report~backup/file.md"),
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


@pytest.mark.parametrize(
    "path",
    [
        "~",
        "~/file",
        "~user/file",
        "./~/file",
        "sub/../~/file",
        "mount/~/file",
        "mount/~user/file",
    ],
)
@pytest.mark.parametrize("as_root", [False, True])
def test_relative_path_rejects_home_prefixes(path: str, as_root: bool) -> None:
    # Mount routing can make a nested component the start of a local path.
    with pytest.raises(ValueError, match="home-directory"):
        if as_root:
            normalize_relative_path("../inside", root=path)
        else:
            normalize_relative_path(path)


def test_local_filesystem_round_trip_preserves_bytes_and_protocols(
    tmp_path: Path,
) -> None:
    storage = LocalFileSystem(tmp_path)
    writer: FileWriter = storage
    reader: FileReader = storage
    skills: SkillReader = storage
    content = b"hello\r\n\xff\x00"

    with writer.open_write("notes/raw.bin") as stream:
        write_all(stream, content)
    metadata: FileStat = storage
    with reader.open_read("notes/raw.bin") as stream:
        assert stream.read() == content

    assert metadata.stat("missing") is None
    directory = metadata.stat("notes")
    assert directory is not None and directory.is_directory
    file_info = metadata.stat("notes/raw.bin")
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

    # Local policies must reject logical paths before host path resolution.
    for policy in policies:
        assert not policy.check(request).allowed


@pytest.mark.parametrize("exists", [True, False])
def test_local_open_write_aborts_on_failure(tmp_path: Path, exists: bool) -> None:
    storage = LocalFileSystem(tmp_path)
    if exists:
        storage.write("target", b"original")

    # An exception after partial output must abort the whole replacement.
    with pytest.raises(RuntimeError, match="abort"):
        with storage.open_write("target") as stream:
            stream.write(b"partial")
            raise RuntimeError("abort")

    if exists:
        assert (tmp_path / "target").read_bytes() == b"original"
    else:
        assert not (tmp_path / "target").exists()

    assert {path.name for path in tmp_path.iterdir()} == (
        {"target"} if exists else set()
    )


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX permissions")
def test_local_new_file_uses_normal_creation_mode(tmp_path: Path) -> None:
    # A reference file observes the current umask without changing global state.
    reference = tmp_path / "reference"
    reference.write_bytes(b"x")
    LocalFileSystem(tmp_path).write("created", b"x")

    assert (tmp_path / "created").stat().st_mode == reference.stat().st_mode


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX permissions")
def test_local_replace_preserves_executable_mode(tmp_path: Path) -> None:
    target = tmp_path / "script"
    target.write_bytes(b"old")
    target.chmod(0o751)

    # Replacing the inode must not discard the target's executable mode bits.
    LocalFileSystem(tmp_path).write("script", b"new")
    assert target.stat().st_mode & 0o777 == 0o751


@pytest.mark.parametrize("exists", [False, True])
def test_local_write_supports_long_valid_filenames(
    tmp_path: Path, exists: bool
) -> None:
    # A valid target can exceed the name limit if a temp suffix is appended.
    name = "x" * 230
    target = tmp_path / name
    if exists:
        target.write_bytes(b"old")

    LocalFileSystem(tmp_path).write(name, b"new")

    assert target.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [target]
