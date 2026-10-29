"""Test saved MCP results across agent runs."""

import asyncio
import json
import re
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pytest
from mcp import types

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.mcp import (
    MCPAuthorizationContext,
    MCPClientManager,
    MCPResultPolicy,
    MCPServer,
    MCPStdioTransport,
    MCPToolsShim,
)
from agentlane.harness.mcp._connection import MCPConnection
from agentlane.harness.mcp._result import render_mcp_result
from agentlane.harness.tools import read_tool
from agentlane.models import ToolCall, ToolFailure, Tools
from agentlane.runtime import CancellationToken

from ..tool_storage.conftest import MemoryFileSystem
from ..tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_tool,
)
from .helpers import ConnectionInstaller, acquire_lease

_SERVER_NAME = "Clinical Notes"
_TOOL_NAME = "get/document"
_VISIBLE_NAME = "clinical_notes__get_document"
_CONTINUATION = re.compile(
    r"\n\n\[Showing lines [^\n]+ Use offset=(\d+) to continue\.\]$"
)


@dataclass
class _Peer:
    result: types.CallToolResult
    opened: list[MCPConnection] = field(default_factory=list[MCPConnection])
    calls: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list[tuple[str, dict[str, Any]]]
    )
    discoveries: int = 0

    async def list_tools(
        self, *, cursor: str | None, cache_mode: Literal["bypass"]
    ) -> types.ListToolsResult:
        del cursor, cache_mode
        self.discoveries += 1
        return types.ListToolsResult(
            tools=[types.Tool(name=_TOOL_NAME, input_schema={"type": "object"})]
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> types.CallToolResult:
        del read_timeout_seconds
        self.calls.append((name, arguments))
        # Simulate a catalog-change notification before the next model turn.
        for connection in self.opened:
            connection.catalog_revision += 1
        return self.result

    @asynccontextmanager
    async def listen(
        self, *, tools_list_changed: bool
    ) -> AsyncGenerator[AsyncIterator[object], None]:
        del tools_list_changed

        async def events() -> AsyncIterator[object]:
            yield object()

        yield events()


@pytest.fixture(name="output_peer")
def fixture_output_peer(install_connection: ConnectionInstaller) -> _Peer:
    peer = _Peer(types.CallToolResult(content=[types.TextContent(text="x" * 1000)]))

    async def prepare(connection: MCPConnection) -> None:
        peer.opened.append(connection)
        connection.client = peer

    install_connection(prepare)
    return peer


def _server(*, max_chars: int = 512) -> MCPServer:
    return MCPServer(
        name=_SERVER_NAME,
        transport=MCPStdioTransport(command="fixture"),
        result_policy=MCPResultPolicy(max_text_chars=max_chars),
    )


def _call(
    name: str = _VISIBLE_NAME, *, call_id: str | None = None, **arguments: object
) -> ToolCall:
    return make_tool_call(
        tool_id=call_id or f"call-{name}", name=name, arguments=json.dumps(arguments)
    )


@pytest.mark.asyncio
async def test_runner_reads_complete_saved_output_through_native_pagination(
    output_peer: _Peer, tmp_path: Path
) -> None:
    transcript = "BEGIN_TRANSCRIPT\n" + "Я🙂界" * 11_000 + "\x00\x07\nEND_TRANSCRIPT"
    output_peer.result = types.CallToolResult(
        content=[types.TextContent(text=transcript)],
        structured_content={"separate_fact": "STRUCTURED_ONLY"},
    )
    filesystem = MountedFileSystem({"saved": LocalFileSystem(tmp_path)})
    model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call(meeting_id="meeting-7")]),
            make_assistant_response("complete"),
        ]
    )

    async with MCPClientManager() as manager:
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="reader",
                model=model,
                shims=(
                    MCPToolsShim(
                        servers=(_server(),),
                        client_manager=manager,
                        result_writer=filesystem,
                        result_directory="saved/./sessions/../tool-results",
                    ),
                ),
            )
        )
        result = await agent.run("Read all of this meeting.")

    assert result.final_output == "complete"
    preview = next(
        message["content"] for message in model.calls[-1] if message["role"] == "tool"
    )
    assert isinstance(preview, str)
    assert len(preview) <= 512
    reference = re.search(r"\n\n\[Full MCP result: (.+)\]$", preview)
    assert reference is not None
    read_path = json.loads(reference.group(1))

    pages: list[str] = []
    offset = 1
    while True:
        page = await asyncio.to_thread(
            run_tool,
            read_tool(reader=filesystem),
            path=read_path,
            offset=offset,
            limit=7,
        )
        continuation = _CONTINUATION.search(page)
        pages.append(page[: continuation.start()] if continuation else page)
        if continuation is None:
            break
        next_offset = int(continuation.group(1))
        assert next_offset > offset
        offset = next_offset

    assert len(pages) > 1
    local_path = tmp_path / PurePosixPath(read_path).relative_to("/saved")
    saved = local_path.read_text(encoding="utf-8")
    assert "\n".join(pages) == saved
    restored = json.loads("".join(pages).replace("\n", ""))
    assert restored["content"][0]["text"] == transcript
    assert restored["structuredContent"] == {"separate_fact": "STRUCTURED_ONLY"}
    assert output_peer.calls == [(_TOOL_NAME, {"meeting_id": "meeting-7"})]
    assert list((tmp_path / "tool-results").iterdir()) == [local_path]
    assert read_path == f"/saved/tool-results/{local_path.name}"
    notice = f"\n\n[Full MCP result: {json.dumps(read_path)}]"
    inline = json.loads(preview.removesuffix(notice))
    assert inline["content"] == []
    assert inline["structuredContent"] == restored["structuredContent"]


@pytest.mark.asyncio
async def test_child_inherits_overflow_storage(
    output_peer: _Peer, tmp_path: Path
) -> None:
    filesystem = LocalFileSystem(tmp_path)
    child_model = SequenceModel(
        [
            make_assistant_response(
                None, tool_calls=[_call(meeting_id="child-meeting")]
            ),
            make_assistant_response("child complete"),
        ]
    )
    child = AgentDescriptor(name="child", model=child_model)
    parent_model = SequenceModel(
        [
            make_assistant_response(None, tool_calls=[_call("child")]),
            make_assistant_response("parent complete"),
        ]
    )

    async with MCPClientManager() as manager:
        await DefaultAgent(
            descriptor=AgentDescriptor(
                name="parent",
                model=parent_model,
                tools=Tools(tools=[child.as_tool()]),
                shims=(
                    MCPToolsShim(
                        servers=(_server(),),
                        client_manager=manager,
                        result_writer=filesystem,
                        result_directory="./sessions/../tool-results",
                    ),
                ),
            )
        ).run("Ask the child to read the meeting.")

    assert output_peer.calls == [(_TOOL_NAME, {"meeting_id": "child-meeting"})]
    (read_path,) = (tmp_path / "tool-results").glob("*.txt")
    assert f"tool-results/{read_path.name}" in json.dumps(child_model.calls[-1])
    assert list((tmp_path / "tool-results").iterdir()) == [read_path]
    assert len(output_peer.opened) == 1
    assert output_peer.discoveries > 1


@pytest.mark.asyncio
async def test_concurrent_shims_keep_writers_separate_after_catalog_refresh(
    output_peer: _Peer,
) -> None:
    labels = ("first", "second")
    backends = [MemoryFileSystem() for _ in labels]
    for backend in backends:
        backend.write_released.clear()
    filesystems = [MountedFileSystem({"saved": backend}) for backend in backends]
    models = [
        SequenceModel(
            [
                make_assistant_response(
                    None,
                    tool_calls=[
                        _call(
                            call_id=f"{label}-{index}",
                            meeting_id=f"{label}-{index}",
                        )
                    ],
                )
                for index in range(2)
            ]
            + [make_assistant_response("complete")]
        )
        for label in labels
    ]

    async with MCPClientManager() as manager:
        agents = [
            DefaultAgent(
                descriptor=AgentDescriptor(
                    name=label,
                    model=model,
                    shims=(
                        MCPToolsShim(
                            servers=(_server(),),
                            client_manager=manager,
                            result_writer=filesystem,
                            result_directory=f"/saved/{label}/tool-results",
                        ),
                    ),
                )
            )
            for label, filesystem, model in zip(
                labels, filesystems, models, strict=True
            )
        ]
        runs = [
            asyncio.create_task(agent.run("Read two meetings.")) for agent in agents
        ]
        try:
            # Both writers must start before either session can finish its save.
            started = await asyncio.gather(
                *(
                    asyncio.to_thread(backend.write_started.wait, 5)
                    for backend in backends
                )
            )
            assert all(started)
            assert len(output_peer.opened) == 1
            for backend in backends:
                backend.write_released.set()
            await asyncio.gather(*runs)
        finally:
            for backend in backends:
                backend.write_released.set()
            for run in runs:
                if not run.done():
                    run.cancel()
            await asyncio.gather(*runs, return_exceptions=True)

    for label, backend, model in zip(labels, backends, models, strict=True):
        assert len(backend.writes) == 2
        history = json.dumps(model.calls[-1])
        for path in backend.files:
            assert path.startswith(f"{label}/tool-results/")
            assert f"/saved/{path}" in history
        other = "second" if label == "first" else "first"
        assert f"/saved/{other}" not in history

    assert sorted(arguments["meeting_id"] for _, arguments in output_peer.calls) == [
        f"{label}-{index}" for label in labels for index in range(2)
    ]
    assert output_peer.discoveries > 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["shutdown", "token", "task"])
async def test_tool_cancellation_waits_for_active_file_write(
    monkeypatch: pytest.MonkeyPatch,
    cancel: Literal["shutdown", "token", "task"],
) -> None:
    backend = MemoryFileSystem()
    backend.write_released.clear()
    filesystem = MountedFileSystem({"saved": backend})
    provider_calls: list[str] = []

    # A real connection exercises shutdown cleanup, which the test stub omits.
    server = MCPServer(
        name="math",
        result_policy=MCPResultPolicy(max_text_chars=128),
        transport=MCPStdioTransport(
            command=sys.executable,
            args=(str(Path(__file__).parent / "fixtures/math_server.py"),),
        ),
    )
    manager = MCPClientManager()
    lease = await acquire_lease(
        manager, server, MCPAuthorizationContext(key="shutdown")
    )
    entry = lease._entry  # pyright: ignore[reportPrivateUsage]
    assert entry.connection is not None
    connection = entry.connection
    client = connection.get_client()
    original_call = client.call_tool

    async def counted_call(
        name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> object:
        provider_calls.append(name)
        return await original_call(
            name, arguments, read_timeout_seconds=read_timeout_seconds
        )

    monkeypatch.setattr(client, "call_tool", counted_call)
    (tool,) = await lease.tools(
        result_writer=filesystem,
        result_directory="/saved/tool-results",
    )
    token = CancellationToken()
    call_task = asyncio.create_task(
        tool.run(
            tool.args_type().model_validate({"left": 10**200, "right": 3}),
            token,
        )
    )
    shutdown: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(backend.write_started.wait, 5)
        if cancel == "shutdown":
            shutdown = asyncio.create_task(manager.aclose())
        elif cancel == "token":
            token.cancel()
        else:
            call_task.cancel()

        # Wait for cancellation to reach the writer before releasing the write.
        async with asyncio.timeout(5):
            while not any(task.cancelling() for task in connection.active_calls):
                await asyncio.sleep(0.01)
        assert backend.files == {}
        assert shutdown is None or not shutdown.done()
        assert not call_task.done()

        backend.write_released.set()
        if shutdown is not None:
            await asyncio.wait_for(shutdown, timeout=5)
        if cancel == "task":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(call_task, timeout=5)
        else:
            result = await asyncio.wait_for(call_task, timeout=5)
            assert isinstance(result, ToolFailure)
            assert result.error.kind == (
                "cancelled" if cancel == "token" else "mcp_transport"
            )
            assert "[Full MCP result:" not in result

        assert len(backend.writes) == 1
        assert len(backend.files) == 1
        assert provider_calls == ["add"]
    finally:
        backend.write_released.set()
        if not call_task.done():
            call_task.cancel()
        await asyncio.gather(call_task, return_exceptions=True)
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        await lease.release()
        await manager.aclose()


@pytest.mark.asyncio
async def test_cancellation_after_mcp_response_does_not_start_storage(
    output_peer: _Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = MemoryFileSystem()
    token = CancellationToken()
    original_call = output_peer.call_tool

    async def cancel_after_response(
        name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> types.CallToolResult:
        result = await original_call(
            name, arguments, read_timeout_seconds=read_timeout_seconds
        )
        # Complete the MCP task before cancellation, but cancel before its caller resumes.
        asyncio.get_running_loop().call_soon(token.cancel)
        return result

    monkeypatch.setattr(output_peer, "call_tool", cancel_after_response)
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, _server(), MCPAuthorizationContext(key="cancel-after-response")
        )
        try:
            (tool,) = await lease.tools(
                result_writer=backend, result_directory="tool-results"
            )
            result = await tool.run(tool.args_type().model_validate({}), token)
        finally:
            await lease.release()

    assert token.is_cancelled
    assert isinstance(result, ToolFailure)
    assert result.error.kind == "cancelled"
    assert "[Full MCP result:" not in result
    assert output_peer.calls == [(_TOOL_NAME, {})]
    assert not backend.write_started.is_set()
    assert backend.writes == []
    assert backend.files == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [10, 100_000])
async def test_no_writer_keeps_existing_response(output_peer: _Peer, size: int) -> None:
    output_peer.result = types.CallToolResult(
        content=[types.TextContent(text="x" * size)]
    )
    server = _server()
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="inline")
        )
        try:
            (tool,) = await lease.tools()
            response = await tool.run(
                tool.args_type().model_validate({}), CancellationToken()
            )
        finally:
            await lease.release()

    assert response == render_mcp_result(output_peer.result, server.result_policy)
    assert output_peer.calls == [(_TOOL_NAME, {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", [False, True])
@pytest.mark.parametrize("include_structured_content", [False, True])
@pytest.mark.parametrize("size", [10, 10_000])
async def test_structured_content_policy_applies_with_and_without_storage(
    output_peer: _Peer,
    tmp_path: Path,
    storage: bool,
    include_structured_content: bool,
    size: int,
) -> None:
    output_peer.result = types.CallToolResult(
        content=[types.TextContent(text="transcript " + "x" * size)],
        structured_content={"details": "structured-only " + "y" * size},
    )
    server = replace(
        _server(),
        result_policy=MCPResultPolicy(
            max_text_chars=512, include_structured_content=include_structured_content
        ),
    )
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager, server, MCPAuthorizationContext(key="policy")
        )
        try:
            (tool,) = await lease.tools(
                result_writer=LocalFileSystem(tmp_path) if storage else None,
                result_directory="results" if storage else None,
            )
            response = await tool.run(
                tool.args_type().model_validate({}), CancellationToken()
            )
        finally:
            await lease.release()

    paths = tuple(tmp_path.rglob("*.txt"))
    if storage and size > 512:
        (path,) = paths
        payload = json.loads("".join(path.read_text().splitlines()))
        assert payload["content"][0]["text"] == "transcript " + "x" * size
        assert "[Full MCP result: " in response
    else:
        assert not paths
        payload = json.loads(response)
        if include_structured_content:
            assert not payload["content"]
    assert ("structuredContent" in payload) is include_structured_content
    assert ("structured-only" in str(payload)) is include_structured_content
    assert len(response) <= 512


@pytest.mark.parametrize("extra_chars", [-1, 0, 1])
def test_shim_validates_resolved_reference_for_each_server(extra_chars: int) -> None:
    filesystem = MountedFileSystem({"saved": MemoryFileSystem()})
    directory = '/saved/results/../é\n"quoted"'
    resolved = filesystem.resolve_path(directory)
    reference = f"\n\n[Full MCP result: {json.dumps(str(resolved / ('0' * 36 + '.txt')), ensure_ascii=False)}]"
    limit = max(128, len(reference) + 1)
    # Extend the resolved path to exercise the exact JSON-escaped boundary.
    directory += "x" * (limit + extra_chars - len(reference))
    servers = (
        _server(max_chars=limit + 100),
        replace(_server(max_chars=limit), name="second", required=False),
    )
    if extra_chars > 0:
        with pytest.raises(ValueError, match="reference exceeds max_text_chars"):
            MCPToolsShim(
                servers=servers, result_writer=filesystem, result_directory=directory
            )
    else:
        MCPToolsShim(
            servers=servers, result_writer=filesystem, result_directory=directory
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_chars", [-1, 0, 1])
async def test_lease_validates_reference_before_discovery(
    output_peer: _Peer, extra_chars: int
) -> None:
    directory = 'results/é\n"quoted"'
    reference = f"\n\n[Full MCP result: {json.dumps(directory + '/' + '0' * 36 + '.txt', ensure_ascii=False)}]"
    directory += "x" * (128 + extra_chars - len(reference))
    async with MCPClientManager() as manager:
        lease = await acquire_lease(
            manager,
            _server(max_chars=128),
            MCPAuthorizationContext(key="reference-budget"),
        )
        try:
            if extra_chars > 0:
                with pytest.raises(
                    ValueError, match="reference exceeds max_text_chars"
                ):
                    await lease.tools(
                        result_writer=MemoryFileSystem(), result_directory=directory
                    )
                assert output_peer.discoveries == 0
            else:
                tools = await lease.tools(
                    result_writer=MemoryFileSystem(), result_directory=directory
                )
                assert len(tools) == 1
        finally:
            await lease.release()
    assert output_peer.calls == []
