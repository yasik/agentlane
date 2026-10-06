"""Mounted paths keep a stable cwd across tools, policies, and sessions."""

import asyncio
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from pydantic import BaseModel

from agentlane.harness.filesystem import MountedFileSystem
from agentlane.harness.tools import (
    HarnessToolDefinition,
    ToolOperation,
    find_tool,
    patch_tool,
    read_tool,
    write_tool,
)
from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ..tools_test_utils import run_tool
from .conftest import MemoryFileSystem, RecordingPolicy, SessionWritePolicy

CWD = "/workspace/sessions/one"
EDITS = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"


def test_mounted_tools_empty_session_share_canonical_cwd(
    storage: MemoryFileSystem, session_policy: SessionWritePolicy
) -> None:
    filesystem = MountedFileSystem({"workspace": storage})
    read = read_tool(reader=filesystem, cwd=CWD, permissions=session_policy)
    write = write_tool(writer=filesystem, cwd=CWD, permissions=session_policy)
    patch = patch_tool(
        reader=filesystem, writer=filesystem, cwd=CWD, permissions=session_policy
    )
    find = find_tool(reader=filesystem, cwd=CWD, permissions=session_policy)
    assert storage.directories == {"."}

    assert run_tool(write, path="notes.md", content="old\n") == (
        f"Wrote 4 bytes to {CWD}/notes.md."
    )
    assert run_tool(read, path="notes.md") == "old"
    assert run_tool(patch, path="notes.md", edits=EDITS) == (
        f"Applied 1 edit to {CWD}/notes.md."
    )
    assert run_tool(find, pattern="*.md") == f"Search directory: {CWD}\nnotes.md"
    assert run_tool(read, path="./notes.md") == "new"
    assert storage.files == {"sessions/one/notes.md": b"new\n"}
    assert all(request.cwd == PurePosixPath(CWD) for request in session_policy.requests)
    assert all(
        request.path is not None and request.path.is_relative_to(CWD)
        for request in session_policy.requests
    )
    assert session_policy.requests[0].operation == ToolOperation.CREATE_DIRECTORY


def test_find_other_mount_result_uses_search_base_for_readback(
    storage: MemoryFileSystem,
) -> None:
    references = MemoryFileSystem()
    references.write("skill/references/guide.md", b"guide")
    filesystem = MountedFileSystem({"workspace": storage, "references": references})
    read = read_tool(reader=filesystem, cwd=CWD)
    find = find_tool(reader=filesystem, cwd=CWD)

    result = run_tool(find, path="/references/skill", pattern="**/*.md")

    header, relative_path = result.splitlines()
    assert header == "Search directory: /references/skill"
    assert relative_path == "references/guide.md"
    base = header.removeprefix("Search directory: ")
    assert run_tool(read, path=f"{base}/{relative_path}") == "guide"
    assert run_tool(read, path=relative_path).startswith("file not found:")
    assert run_tool(
        write_tool(writer=filesystem, cwd=CWD), path="notes.md", content="x"
    )
    assert storage.files == {"sessions/one/notes.md": b"x"}


@pytest.mark.parametrize(
    ("path", "target"),
    [
        ("../two/notes.md", "/workspace/sessions/two/notes.md"),
        ("/workspace/sessions/one/../two/notes.md", "/workspace/sessions/two/notes.md"),
        ("/references/notes.md", "/references/notes.md"),
        ("/workspace/../references/notes.md", "/references/notes.md"),
    ],
)
@pytest.mark.parametrize("operation", ["write", "patch"])
def test_mounted_mutation_outside_session_denies_canonical_target(
    storage: MemoryFileSystem,
    session_policy: SessionWritePolicy,
    path: str,
    target: str,
    operation: str,
) -> None:
    references = MemoryFileSystem()
    storage.write("sessions/two/notes.md", b"old\n")
    references.write("notes.md", b"old\n")
    storage.writes.clear()
    references.writes.clear()
    filesystem = MountedFileSystem({"workspace": storage, "references": references})
    if operation == "write":
        definition = write_tool(writer=filesystem, cwd=CWD, permissions=session_policy)
        result = run_tool(definition, path=path, content="new\n")
    else:
        definition = patch_tool(
            reader=filesystem, writer=filesystem, cwd=CWD, permissions=session_policy
        )
        result = run_tool(definition, path=path, edits=EDITS)

    assert result == f"permission denied: {operation} is not allowed for `{target}`"
    assert session_policy.requests[-1].path == PurePosixPath(target)
    assert session_policy.requests[-1].cwd == PurePosixPath(CWD)
    assert storage.opened == references.opened == []
    assert storage.writes == references.writes == []


@pytest.mark.parametrize(
    "path", ["../../../../escape", "/../escape", r"C:\secret", "a\x00b"]
)
@pytest.mark.parametrize("name", ["read", "write", "patch", "find"])
def test_mounted_invalid_path_fails_before_policy_and_backend(
    storage: MemoryFileSystem, policy: RecordingPolicy, path: str, name: str
) -> None:
    filesystem = MountedFileSystem({"workspace": storage})
    definitions = {
        "read": read_tool(reader=filesystem, cwd=CWD, permissions=policy),
        "write": write_tool(writer=filesystem, cwd=CWD, permissions=policy),
        "patch": patch_tool(
            reader=filesystem, writer=filesystem, cwd=CWD, permissions=policy
        ),
        "find": find_tool(reader=filesystem, cwd=CWD, permissions=policy),
    }
    args: dict[str, object] = {"path": path}
    if name == "write":
        args["content"] = "new"
    elif name == "patch":
        args["edits"] = EDITS
    elif name == "find":
        args["pattern"] = "*"

    result = run_tool(definitions[name], **args)
    if "\x00" in path and name in ("write", "patch"):
        assert result == "path contains a null byte"
    else:
        assert result.startswith("failed to")
    assert policy.requests == []
    assert storage.thread_ids == []


def test_mounted_absolute_host_path_never_falls_back_to_local_storage(
    storage: MemoryFileSystem, tmp_path: Path
) -> None:
    host_note = tmp_path / "notes.md"
    host_note.write_text("host only")
    filesystem = MountedFileSystem({"workspace": storage})

    assert run_tool(read_tool(reader=filesystem, cwd=CWD), path=str(host_note)) != (
        "host only"
    )
    assert not run_tool(
        write_tool(writer=filesystem, cwd=CWD), path=str(host_note), content="changed"
    ).startswith("Wrote")
    assert host_note.read_text() == "host only"
    assert storage.thread_ids == []


def test_new_mount_does_not_change_relative_directory_resolution(
    storage: MemoryFileSystem,
) -> None:
    original = MountedFileSystem({"workspace": storage})
    references = MemoryFileSystem()
    extended = MountedFileSystem({"workspace": storage, "references": references})

    for filesystem in (original, extended):
        write = write_tool(writer=filesystem, cwd=CWD)
        assert run_tool(write, path="references/notes.md", content="session") == (
            f"Wrote 7 bytes to {CWD}/references/notes.md."
        )
        assert run_tool(
            read_tool(reader=filesystem, cwd=CWD), path="references/notes.md"
        ) == ("session")

    assert storage.files == {"sessions/one/references/notes.md": b"session"}
    assert references.files == {}


async def _run(definition: HarnessToolDefinition, **arguments: object) -> str:
    tool = cast(Tool[BaseModel, str], definition.tool)
    return await tool.run(
        tool.args_type().model_validate(arguments), CancellationToken()
    )


@pytest.mark.asyncio
async def test_shared_filesystem_concurrent_session_tools_keep_independent_cwd(
    storage: MemoryFileSystem,
) -> None:
    filesystem = MountedFileSystem({"workspace": storage})

    async def session(name: str) -> tuple[str, str]:
        cwd = f"/workspace/sessions/{name}"
        policy = SessionWritePolicy(cwd)
        await _run(
            write_tool(writer=filesystem, cwd=cwd, permissions=policy),
            path="notes.md",
            content=name,
        )
        # A separately built helper captures the same cwd without changing the FS.
        helper_read = read_tool(reader=filesystem, cwd=cwd, permissions=policy)
        value = await _run(helper_read, path="notes.md")
        search = await _run(find_tool(reader=filesystem, cwd=cwd), pattern="*.md")
        return value, search

    results = await asyncio.gather(session("one"), session("two"))

    assert list(results) == [
        ("one", "Search directory: /workspace/sessions/one\nnotes.md"),
        ("two", "Search directory: /workspace/sessions/two\nnotes.md"),
    ]
    assert storage.files == {
        "sessions/one/notes.md": b"one",
        "sessions/two/notes.md": b"two",
    }
