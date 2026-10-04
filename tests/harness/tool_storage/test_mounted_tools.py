"""Mounted tools use interfaces; process tools use their workspace filesystem."""

import asyncio
import json
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from pydantic import BaseModel

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.filesystem import (
    BinaryReader,
    DirectoryEntry,
    FileInfo,
    LocalFileSystem,
    MountedFileSystem,
    MountedReader,
)
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    HarnessToolDefinition,
    HarnessToolsShim,
    ToolPermissionDecision,
    base_harness_tools,
    find_tool,
    patch_tool,
    write_tool,
)
from agentlane.models import Tool, Tools
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from ..skill_storage._reader import MemorySkillReader
from ..tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_tool,
)
from .conftest import MemoryFileSystem, RecordingPolicy

EDITS = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"


async def _run(
    definition: HarnessToolDefinition, token: CancellationToken, **args: object
) -> str:
    tool = cast(Tool[BaseModel, str], definition.tool)
    return await tool.run(tool.args_type().model_validate(args), token)


def test_file_tools_share_mounted_paths_and_process_tools_keep_workspace(
    storage: MemoryFileSystem,
    tmp_path: Path,
) -> None:
    mounted = MountedFileSystem({"tenant": storage})
    (tmp_path / "process.txt").write_text("local only\n")
    tools = {d.tool.name: d for d in base_harness_tools(reader=mounted, cwd=tmp_path)}
    assert (
        run_tool(tools["write"], path="tenant/note.txt", content="old\n")
        == "Wrote 4 bytes to tenant/note.txt."
    )
    assert run_tool(tools["read"], path="tenant/note.txt") == "old"
    assert (
        run_tool(tools["find"], pattern="**/*.txt")
        == "Search directory: .\ntenant/note.txt"
    )
    assert (
        run_tool(tools["patch"], path="tenant/note.txt", edits=EDITS)
        == "Applied 1 edit to tenant/note.txt."
    )
    assert storage.files == {"note.txt": b"new\n"}

    # Mounted storage does not create a host mount or expose bytes to processes.
    assert (
        run_tool(tools["grep"], pattern="local only")
        == "Search path: .\nprocess.txt:1:local only"
    )
    assert run_tool(tools["bash"], command="cat process.txt").strip() == "local only"
    assert "No matches" in run_tool(tools["grep"], pattern="new")
    assert not (tmp_path / "tenant").exists()


def test_process_and_storage_working_directories_are_independent(
    storage: MemoryFileSystem, tmp_path: Path
) -> None:
    mounted = MountedFileSystem({"tenant": storage})
    tools = {
        d.tool.name: d
        for d in base_harness_tools(reader=mounted, cwd=tmp_path, storage_cwd="tenant")
    }
    run_tool(tools["write"], path="note", content="old\n")
    assert run_tool(tools["read"], path="note") == "old"
    assert run_tool(tools["find"], pattern="*") == "Search directory: tenant\nnote"
    assert run_tool(tools["bash"], command="pwd").strip() == str(tmp_path)


@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("name", ["read", "write", "find", "patch"])
def test_injected_local_tools_reject_home_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mounted: bool,
    name: str,
) -> None:
    # Use a real local backend so the test covers expansion after validation.
    home = tmp_path / "home"
    home.mkdir()
    note = home / "note.txt"
    note.write_text("old\n")
    monkeypatch.setenv("HOME", str(home))
    local = LocalFileSystem(tmp_path / "workspace")
    reader = MountedFileSystem({"local": local}) if mounted else local
    tools = {d.tool.name: d for d in base_harness_tools(reader=reader)}
    path = "local/~" if mounted else "~"
    args: dict[str, object] = {"path": path if name == "find" else f"{path}/note.txt"}
    if name == "write":
        args["content"] = "new\n"
    elif name == "patch":
        args["edits"] = EDITS
    elif name == "find":
        args["pattern"] = "*.txt"

    result = run_tool(tools[name], **args)

    assert result.startswith("failed to")
    assert note.read_text() == "old\n"


def test_mounted_writes_preserve_readonly_mounts_and_roots(
    storage: MemoryFileSystem,
) -> None:
    readonly = MemorySkillReader({"note": b"original"})
    mounted = MountedFileSystem({"rw": storage, "ro": readonly})
    assert mounted.stat(".") == mounted.stat("ro") == FileInfo(True)
    assert mounted.stat("ro/note") == FileInfo(False)
    assert mounted.stat("unknown/path") is None
    assert (
        run_tool(write_tool(writer=mounted), path="ro/note", content="x")
        == "permission denied: `ro/note`"
    )
    for root in (".", "rw", "ro"):
        with pytest.raises(IsADirectoryError):
            mounted.open_write(root)
    with pytest.raises(FileNotFoundError):
        mounted.open_write("unknown/note")
    assert readonly.files["note"] == b"original"


def test_find_uses_one_ordering_for_local_and_injected_metadata(tmp_path: Path) -> None:
    local = LocalFileSystem(tmp_path)
    local.write("a.py", b"old\n")
    local.write("b.py", b"old\n")
    native = run_tool(find_tool(cwd=tmp_path), pattern="*.py").splitlines()[1:]
    injected = run_tool(find_tool(reader=local), pattern="*.py").splitlines()[1:]
    assert native == injected
    assert all(entry.modified_time is not None for entry in local.list_directory("."))


def test_find_storage_respects_same_root_ignore_rules(
    storage: MemoryFileSystem,
) -> None:
    for path in ("a.py", "z.py", "nested/b.py", "ignored/x.py"):
        storage.write(path, b"x")
    storage.write(".gitignore", b"ignored/\nz.py\n")
    assert (
        run_tool(find_tool(reader=storage), pattern="**/*.py")
        == "Search directory: .\na.py\nnested/b.py"
    )


@pytest.mark.parametrize("name", ["find", "patch"])
@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../escape", "x/../../escape", r"C:\secret"]
)
def test_injected_file_tools_reject_host_paths(
    storage: MemoryFileSystem, name: str, path: str
) -> None:
    tools = {
        d.tool.name: d
        for d in base_harness_tools(reader=storage, include=("find", "patch"))
    }
    args = (
        {"path": path, "edits": EDITS}
        if name == "patch"
        else {"path": path, "pattern": "x"}
    )
    assert run_tool(tools[name], **args).startswith("failed to")
    assert storage.opened == []
    assert storage.writes == []


@pytest.mark.parametrize("name", ["find", "patch"])
def test_permissions_precede_injected_file_access(
    storage: MemoryFileSystem, policy: RecordingPolicy, name: str
) -> None:
    policy.decision = ToolPermissionDecision.deny()
    tools = {
        d.tool.name: d
        for d in base_harness_tools(
            reader=storage,
            storage_cwd="logical",
            permissions=policy,
            include=("find", "patch"),
        )
    }
    args = {"path": "note", "edits": EDITS} if name == "patch" else {"pattern": "x"}
    assert run_tool(tools[name], **args).startswith(f"permission denied: {name}")

    # Even metadata probes count as backend access and must follow permission.
    assert storage.thread_ids == []
    assert policy.requests[0].cwd == PurePosixPath("logical")


def test_patch_failed_edit_never_opens_a_writer(storage: MemoryFileSystem) -> None:
    storage.write("note", b"old\n")
    storage.writes.clear()

    # Failure in a later edit must not publish the earlier successful edit.
    edits = EDITS + "\n<<<<<<< SEARCH\nmissing\n=======\nx\n>>>>>>> REPLACE"
    assert "Could not find edit 2" in run_tool(
        patch_tool(reader=storage, writer=storage), path="note", edits=edits
    )
    assert storage.files["note"] == b"old\n"
    assert storage.writes == []


def test_patch_injected_bom_and_line_endings_match_local_engine(
    storage: MemoryFileSystem, tmp_path: Path
) -> None:
    content = b"\xef\xbb\xbfold\r\n"
    storage.write("note", content)
    (tmp_path / "note").write_bytes(content)
    assert "Applied 1 edit" in run_tool(
        patch_tool(reader=storage, writer=storage), path="note", edits=EDITS
    )
    assert "Applied 1 edit" in run_tool(
        patch_tool(cwd=tmp_path), path="note", edits=EDITS
    )
    assert (
        storage.files["note"]
        == (tmp_path / "note").read_bytes()
        == b"\xef\xbb\xbfnew\r\n"
    )


@pytest.mark.asyncio
async def test_patch_waits_for_started_write_on_cancellation(
    storage: MemoryFileSystem,
) -> None:
    storage.write("note", b"old\n")
    storage.write_started.clear()
    storage.write_released.clear()
    task = asyncio.create_task(
        _run(
            patch_tool(reader=storage, writer=storage),
            CancellationToken(),
            path="note",
            edits=EDITS,
        )
    )
    try:
        assert await asyncio.to_thread(storage.write_started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)

        # Repeated task cancellation must still wait for the active commit.
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        storage.write_released.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert storage.files["note"] == b"new\n"


def test_readonly_reader_can_supply_read_find_and_process_tools(tmp_path: Path) -> None:
    mounted = MountedReader({"tenant": MemorySkillReader({"note": b"x"})})
    tools = base_harness_tools(reader=mounted, cwd=tmp_path, exclude=("write", "patch"))
    assert {d.tool.name for d in tools} >= {"read", "find", "grep", "bash"}
    with pytest.raises(ValueError, match="writer"):
        base_harness_tools(reader=mounted)


@pytest.mark.asyncio
async def test_mounted_tools_run_through_runner(storage: MemoryFileSystem) -> None:
    calls = (
        ("write", {"path": "tenant/note", "content": "old\n"}),
        ("patch", {"path": "tenant/note", "edits": EDITS}),
        ("read", {"path": "tenant/note"}),
    )
    responses = [
        make_assistant_response(
            None,
            tool_calls=[
                make_tool_call(
                    tool_id=f"call_{i}", name=name, arguments=json.dumps(args)
                )
            ],
        )
        for i, (name, args) in enumerate(calls)
    ]
    model = SequenceModel([*responses, make_assistant_response("complete")])
    runner = Runner()
    agent = Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        descriptor=AgentDescriptor(
            name="Mounted files",
            model=model,
            tools=Tools(
                tools=[
                    d.tool
                    for d in base_harness_tools(
                        reader=MountedFileSystem({"tenant": storage}),
                        include=("read", "write", "patch"),
                    )
                ]
            ),
        ),
    )
    state = RunState(
        instructions="Use mounted files", history=["Update note"], responses=[]
    )
    result = await runner.run(agent, state)
    assert result.final_output == "complete"
    assert storage.files["note"] == b"new\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_parallel_patches_read_after_prior_commit_settles(
    storage: MemoryFileSystem, cancel_first: bool
) -> None:
    storage.files["note"] = b"old\nsecond\n"
    storage.write_released.clear()

    # Both calls share one tool because its lock owns this serialization scope.
    definition = patch_tool(reader=storage, writer=storage)
    first = asyncio.create_task(
        _run(definition, CancellationToken(), path="note", edits=EDITS)
    )
    second = None
    try:
        assert await asyncio.to_thread(storage.write_started.wait, 2)
        if cancel_first:
            first.cancel()

        # A second read before the first commit would overwrite that first edit.
        second = asyncio.create_task(
            _run(
                definition,
                CancellationToken(),
                path="note",
                edits="<<<<<<< SEARCH\nsecond\n=======\nSECOND\n>>>>>>> REPLACE",
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert storage.opened == ["note"]
        assert not first.done()
    finally:
        storage.write_released.set()

    if cancel_first:
        with pytest.raises(asyncio.CancelledError):
            await first
    else:
        assert await first == "Applied 1 edit to note."

    assert second is not None
    assert await second == "Applied 1 edit to note."
    assert storage.files["note"] == b"new\nSECOND\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_task", [False, True])
async def test_cancelled_patch_waiter_does_not_read_or_write(
    storage: MemoryFileSystem, cancel_task: bool
) -> None:
    storage.files["note"] = b"old\n"
    storage.write_released.clear()
    definition = patch_tool(reader=storage, writer=storage)
    first = asyncio.create_task(
        _run(definition, CancellationToken(), path="note", edits=EDITS)
    )
    token = CancellationToken()
    second = None
    try:
        assert await asyncio.to_thread(storage.write_started.wait, 2)
        second = asyncio.create_task(_run(definition, token, path="note", edits=EDITS))
        await asyncio.sleep(0)

        # Both cancellation paths must stop a waiter before it reads a snapshot.
        if cancel_task:
            second.cancel()
        else:
            token.cancel()
    finally:
        storage.write_released.set()

    await first
    assert second is not None
    with pytest.raises(asyncio.CancelledError):
        await second

    assert storage.opened == ["note"]
    assert len(storage.writes) == 1


@pytest.mark.parametrize("failure_path", [".", "child"])
def test_find_reports_backend_listing_failures(
    storage: MemoryFileSystem, monkeypatch: pytest.MonkeyPatch, failure_path: str
) -> None:
    storage.write("child/note", b"x")
    original = storage.list_directory

    def listing(path: str) -> tuple[DirectoryEntry, ...]:
        if path == failure_path:
            raise TimeoutError("private provider details")

        return original(path)

    monkeypatch.setattr(storage, "list_directory", listing)

    # A failed subtree is a failed search, not a valid empty or partial result.
    assert run_tool(find_tool(reader=storage), pattern="**/*") == "failed to find files"


@pytest.mark.parametrize("count", [10, 1000])
def test_mounted_find_lists_each_directory_once_without_file_stat(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    # This reader has no stat method; sorting must reuse directory entries.
    reader = MemorySkillReader({f"{index:04}.txt": b"x" for index in range(count)})
    original = reader.list_directory
    listed: list[str] = []

    def listing(path: str) -> tuple[DirectoryEntry, ...]:
        listed.append(path)
        return tuple(original(path))

    monkeypatch.setattr(reader, "list_directory", listing)
    result = run_tool(
        find_tool(reader=MountedReader({"data": reader})), pattern="**/*.txt", limit=5
    )
    assert "data/0000.txt" in result
    assert listed == ["."]


@pytest.mark.parametrize("injected", [False, True])
def test_find_skips_directory_symlink_cycles(tmp_path: Path, injected: bool) -> None:
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "note.txt").write_text("x")

    # Following this directory link would revisit the search root indefinitely.
    (tmp_path / "folder" / "loop").symlink_to(tmp_path, target_is_directory=True)
    definition = (
        find_tool(reader=LocalFileSystem(tmp_path))
        if injected
        else find_tool(cwd=tmp_path)
    )
    result = run_tool(definition, pattern="**/*.txt")
    assert result.splitlines()[1:] == ["folder/note.txt"]


@pytest.mark.asyncio
async def test_injected_tools_prompt_explains_both_namespaces(
    storage: MemoryFileSystem, tmp_path: Path
) -> None:
    definitions = base_harness_tools(reader=storage, cwd=tmp_path, storage_cwd="tenant")
    shim = HarnessToolsShim(definitions)
    bound = await shim.bind(cast(ShimBindingContext, object()))
    state = RunState(instructions="Base", history=[], responses=[], turn_count=1)
    await bound.prepare_turn(PreparedTurn(run_state=state, tools=None, model_args=None))

    # Check the model-visible instructions, not only the tool metadata.
    prompt = cast(str, state.instructions)
    assert "working directory 'tenant'" in prompt
    assert f"configured workspace '{tmp_path}'" in prompt
    assert "Storage paths and process paths are not interchangeable" in prompt
    assert "explicit path mapping" in prompt


@pytest.mark.asyncio
async def test_task_cancelled_during_patch_read_cannot_commit_later(
    storage: MemoryFileSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage.files["note"] = b"old\n"
    started = threading.Event()
    released = threading.Event()
    finished = threading.Event()
    original = storage.open_read

    @contextmanager
    def blocked_read(path: str) -> Generator[BinaryReader, None, None]:
        with original(path) as stream:
            started.set()
            try:
                assert released.wait(timeout=2)
                yield stream
            finally:
                finished.set()

    monkeypatch.setattr(storage, "open_read", blocked_read)
    definition = patch_tool(reader=storage, writer=storage)
    task = asyncio.create_task(
        _run(definition, CancellationToken(), path="note", edits=EDITS)
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)

        # The worker keeps reading after cancellation; it must never reach write.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        released.set()

    assert await asyncio.to_thread(finished.wait, 2)
    assert storage.writes == []

    # Cancellation must also release the tool lock for the next valid call.
    monkeypatch.setattr(storage, "open_read", original)
    assert (
        await _run(definition, CancellationToken(), path="note", edits=EDITS)
        == "Applied 1 edit to note."
    )
    assert storage.files["note"] == b"new\n"
