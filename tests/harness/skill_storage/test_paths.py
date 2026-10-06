"""Virtual paths have one lexical root and no host filesystem dependencies."""

from pathlib import PurePosixPath

import pytest

from agentlane.harness.filesystem import normalize_virtual_path


@pytest.mark.parametrize(
    ("path", "cwd", "expected"),
    [
        ("notes.md", "/sessions/current", "/sessions/current/notes.md"),
        ("/skills/guide.md", "/sessions/current", "/skills/guide.md"),
        ("../other/notes.md", "/sessions/current", "/sessions/other/notes.md"),
        ("local/notes.md", "/sessions/current", "/sessions/current/local/notes.md"),
        ("./local/notes.md", "/sessions/current", "/sessions/current/local/notes.md"),
        ("notes.md", "sessions/current", "/sessions/current/notes.md"),
        ("notes.md", ".", "/notes.md"),
        (".", "/", "/"),
        ("..", "/sessions", "/"),
        ("/one//two/./../three/", "/", "/one/three"),
        ("//one///two", "/", "/one/two"),
    ],
)
def test_virtual_path_valid_input_resolves_from_cwd(
    path: str, cwd: str, expected: str
) -> None:
    assert normalize_virtual_path(path, cwd=cwd) == PurePosixPath(expected)


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        "../escape",
        "/../escape",
        "/one/../../escape",
        "one/../../one/file",
        "one\\file",
        "one/bad\x00name",
        "C:/file",
        "/one/C:/file",
        "/one/a:b",
        "~/file",
        "/one/~user/file",
    ],
)
def test_virtual_path_invalid_input_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        normalize_virtual_path(path)


@pytest.mark.parametrize("cwd", ["", "../escape", "/../escape", "C:/", "/~user"])
def test_virtual_path_invalid_cwd_rejected_for_absolute_target(cwd: str) -> None:
    with pytest.raises(ValueError):
        normalize_virtual_path("/valid/file", cwd=cwd)


def test_virtual_path_parent_traversal_above_cwd_root_rejected() -> None:
    with pytest.raises(ValueError, match="virtual root"):
        normalize_virtual_path("../../../file", cwd="/sessions/current")
