"""Storage adapters retain native output and permission behavior."""

import asyncio
import threading
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from pydantic import BaseModel

from agentlane.harness.filesystem import FileInfo
from agentlane.harness.tools import (
    TEXT_MAX_BYTES,
    HarnessToolDefinition,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionRequest,
    WorkspaceToolPermissionPolicy,
    base_harness_tools,
    read_tool,
    write_tool,
)
from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ..tools_test_utils import run_tool
from .conftest import MemoryFileSystem, RecordingPolicy


def test_read_nonseekable_sample_boundary_preserves_lines(
    storage: MemoryFileSystem,
) -> None:
    storage.files["notes.txt"] = b"first\r\n" + b"a" * 4088 + "é\nlast\n".encode()

    result = run_tool(read_tool(reader=storage), path="notes.txt", offset=2)

    assert result == "a" * 4088 + "é\nlast"
    assert storage.opened == storage.closed == ["notes.txt"]
    assert all(identifier != threading.get_ident() for identifier in storage.thread_ids)


def test_read_nonseekable_slice_retains_continuation(
    storage: MemoryFileSystem,
) -> None:
    storage.files["notes.txt"] = b"one\ntwo\nthree\n"

    result = run_tool(read_tool(reader=storage), path="notes.txt", offset=2, limit=1)

    assert result == "two\n\n[Showing lines 2-2. Use offset=3 to continue.]"
    assert storage.closed == ["notes.txt"]


def test_read_nonseekable_short_reads_still_detect_binary_sample(
    storage: MemoryFileSystem,
) -> None:
    storage.files["data.bin"] = b"a" * 4000 + b"\x00"

    result = run_tool(read_tool(reader=storage), path="data.bin")

    assert result == "file appears to be binary and cannot be read as text: `data.bin`"
    assert storage.closed == ["data.bin"]


def test_read_nonseekable_output_cap_is_preserved(storage: MemoryFileSystem) -> None:
    storage.files["wide.txt"] = b"ok\n" + b"a" * TEXT_MAX_BYTES

    result = run_tool(read_tool(reader=storage), path="wide.txt")

    assert result == (
        f"ok\n\n[Showing lines 1-1 ({TEXT_MAX_BYTES} byte limit). "
        "Use offset=2 to continue.]"
    )


def test_read_injected_cwd_uses_relative_permission_path(
    storage: MemoryFileSystem, policy: RecordingPolicy
) -> None:
    storage.files["skills/guide.md"] = b"guide"

    result = run_tool(
        read_tool(reader=storage, cwd="skills", permissions=policy),
        path="./guide.md",
    )

    assert result == "guide"
    assert policy.requests[0].path == PurePosixPath("skills/guide.md")
    assert not isinstance(policy.requests[0].path, Path)
    assert policy.requests[0].cwd == PurePosixPath("skills")
    assert storage.opened == ["skills/guide.md"]


def test_read_denied_policy_does_not_open_stream(
    storage: MemoryFileSystem, policy: RecordingPolicy
) -> None:
    policy.decision = ToolPermissionDecision.deny()

    result = run_tool(read_tool(reader=storage, permissions=policy), path="secret")

    assert result == "permission denied: read is not allowed for `secret`"
    assert storage.opened == []


@pytest.mark.parametrize("path", ["/etc/passwd", "../escape", "a/../../escape"])
def test_injected_path_outside_namespace_does_not_access_storage(
    storage: MemoryFileSystem, path: str
) -> None:
    assert run_tool(read_tool(reader=storage), path=path) == "failed to read file"
    assert (
        run_tool(write_tool(writer=storage), path=path, content="x")
        == "failed to write file"
    )
    assert storage.thread_ids == []


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError("private"), "file not found"),
        (IsADirectoryError("private"), "path is a directory"),
        (PermissionError("private"), "permission denied"),
        (OSError("private"), "failed to read file"),
    ],
)
def test_read_storage_error_has_native_sanitized_output(
    storage: MemoryFileSystem, error: OSError, message: str
) -> None:
    storage.failure = error
    assert run_tool(read_tool(reader=storage), path="notes") == f"{message}: `notes`"


def test_write_injected_storage_preserves_bytes_and_permission_operations(
    storage: MemoryFileSystem, policy: RecordingPolicy
) -> None:
    definition = write_tool(writer=storage, permissions=policy)

    assert (
        run_tool(definition, path="nested/notes", content="µ\r\n")
        == "Wrote 4 bytes to nested/notes."
    )
    assert [request.operation for request in policy.requests] == [
        ToolOperation.CREATE_DIRECTORY,
        ToolOperation.CREATE_FILE,
    ]
    assert [request.path for request in policy.requests] == [
        PurePosixPath("nested"),
        PurePosixPath("nested/notes"),
    ]
    policy.requests.clear()
    assert (
        run_tool(definition, path="nested/notes", content="")
        == "Wrote 0 bytes to nested/notes."
    )
    assert [request.operation for request in policy.requests] == [
        ToolOperation.OVERWRITE_FILE
    ]
    assert storage.writes == [("nested/notes", b"\xc2\xb5\r\n"), ("nested/notes", b"")]
    assert all(identifier != threading.get_ident() for identifier in storage.thread_ids)


def test_write_approval_denied_leaves_storage_unchanged(
    storage: MemoryFileSystem, policy: RecordingPolicy
) -> None:
    policy.decision = ToolPermissionDecision.require_approval()

    result = run_tool(
        write_tool(writer=storage, permissions=policy), path="new/notes", content="x"
    )

    assert result.startswith("approval required:")
    assert storage.writes == []
    assert storage.directories == {"."}


def test_write_approval_is_complete_before_side_effects(
    storage: MemoryFileSystem, policy: RecordingPolicy
) -> None:
    policy.decision = ToolPermissionDecision.require_approval()
    approved: list[ToolOperation] = []

    async def approve(
        request: ToolPermissionRequest, _decision: ToolPermissionDecision
    ) -> ToolPermissionDecision:
        assert storage.writes == []
        assert storage.directories == {"."}
        approved.append(request.operation)
        return ToolPermissionDecision.allow()

    result = run_tool(
        write_tool(writer=storage, permissions=policy, approval_callback=approve),
        path="new/notes",
        content="x",
    )

    assert result == "Wrote 1 bytes to new/notes."
    assert approved == [ToolOperation.CREATE_DIRECTORY, ToolOperation.CREATE_FILE]
    assert storage.files == {"new/notes": b"x"}


def test_write_parent_file_reports_native_error(storage: MemoryFileSystem) -> None:
    storage.files["blocked"] = b"file"

    result = run_tool(write_tool(writer=storage), path="blocked/child", content="x")

    assert result == "parent path is not a directory: `blocked`"
    assert storage.writes == []


@pytest.mark.parametrize("denied", [False, True])
def test_write_invalid_parent_metadata_checks_policy_before_diagnostic(
    storage: MemoryFileSystem,
    policy: RecordingPolicy,
    monkeypatch: pytest.MonkeyPatch,
    denied: bool,
) -> None:
    def invalid_parent(_path: str) -> FileInfo | None:
        raise NotADirectoryError("private storage details")

    monkeypatch.setattr(storage, "stat", invalid_parent)
    if denied:
        policy.decision = ToolPermissionDecision.deny()

    result = run_tool(
        write_tool(writer=storage, permissions=policy),
        path="blocked/child",
        content="x",
    )

    expected = (
        "permission denied: write is not allowed for `blocked`"
        if denied
        else "parent path is not a directory: `blocked`"
    )
    assert result == expected
    assert policy.requests
    assert storage.writes == []


def test_base_tools_forward_storage_adapters(storage: MemoryFileSystem) -> None:
    definitions = base_harness_tools(
        reader=storage, writer=storage, include=("read", "write")
    )
    read, write = definitions

    assert run_tool(write, path="note", content="saved") == "Wrote 5 bytes to note."
    assert run_tool(read, path="note") == "saved"


def test_injected_read_rejects_local_workspace_policy(
    storage: MemoryFileSystem, tmp_path: Path
) -> None:
    result = run_tool(
        read_tool(reader=storage, permissions=WorkspaceToolPermissionPolicy(tmp_path)),
        path="note",
    )

    assert result.startswith("permission denied:")
    assert storage.opened == []


async def _run(
    definition: HarnessToolDefinition, token: CancellationToken, **arguments: object
) -> str:
    tool = cast(Tool[BaseModel, str], definition.tool)
    return await tool.run(tool.args_type()(**arguments), token)


@pytest.mark.asyncio
async def test_precancelled_tools_do_not_access_storage(
    storage: MemoryFileSystem,
) -> None:
    token = CancellationToken()
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await _run(read_tool(reader=storage), token, path="notes")
    with pytest.raises(asyncio.CancelledError):
        await _run(write_tool(writer=storage), token, path="notes", content="x")

    assert storage.thread_ids == []


@pytest.mark.asyncio
async def test_cancelled_write_finishes_before_handler_exits(
    storage: MemoryFileSystem,
) -> None:
    storage.write_released.clear()
    task = asyncio.create_task(
        _run(write_tool(writer=storage), CancellationToken(), path="notes", content="x")
    )
    try:
        assert await asyncio.to_thread(storage.write_started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        # A repeated cancellation must also leave the worker protected.
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        storage.write_released.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert storage.writes == [("notes", b"x")]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (PermissionError("private details"), "permission denied"),
        (OSError("private details"), "failed to write file"),
    ],
)
def test_write_storage_error_has_native_sanitized_output(
    storage: MemoryFileSystem, error: OSError, message: str
) -> None:
    storage.failure = error

    result = run_tool(write_tool(writer=storage), path="notes", content="x")

    assert result == f"{message}: `notes`"
    assert storage.writes == []


@pytest.mark.asyncio
async def test_write_token_cancelled_during_approval_does_not_write(
    storage: MemoryFileSystem, policy: RecordingPolicy
) -> None:
    token = CancellationToken()
    policy.decision = ToolPermissionDecision.require_approval()

    async def approve(
        _request: ToolPermissionRequest, _decision: ToolPermissionDecision
    ) -> ToolPermissionDecision:
        token.cancel()
        return ToolPermissionDecision.allow()

    definition = write_tool(
        writer=storage, permissions=policy, approval_callback=approve
    )
    with pytest.raises(asyncio.CancelledError):
        await _run(definition, token, path="notes", content="x")

    assert storage.writes == []


def test_read_injected_oversized_first_line_keeps_bash_guidance(
    storage: MemoryFileSystem,
) -> None:
    storage.files["wide.txt"] = b"a" * (TEXT_MAX_BYTES + 1)

    result = run_tool(read_tool(reader=storage), path="wide.txt")

    assert result == (
        f"[Line 1 is {TEXT_MAX_BYTES + 1} bytes, exceeds "
        f"{TEXT_MAX_BYTES} byte limit. Use bash to inspect it.]"
    )
