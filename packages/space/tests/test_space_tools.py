"""Exercise the native HTTP path with real guest worker subprocesses."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from agentlane_space import SpaceError, SpaceErrorKind, space_harness_tools
from conftest import LocalSpace, sandbox_for

from agentlane.harness.tools import BASE_TOOL_NAMES, base_harness_tools
from agentlane.models import Tool, ToolCall, ToolExecutor, ToolFailure, Tools
from agentlane.runtime import CancellationToken


async def _invoke(tool: Tool[Any, Any], arguments: dict[str, Any]) -> Any:
    return await tool.run(
        tool.args_type().model_validate(arguments), CancellationToken()
    )


@pytest.mark.asyncio
async def test_all_environment_tools_use_guest_cwd_and_preserve_host_tools(
    local_space: LocalSpace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = tmp_path / "host"
    guest = tmp_path / "guest"
    host.mkdir()
    guest.mkdir()
    (host / "note.txt").write_text("host sentinel")
    (guest / ".git").mkdir()
    monkeypatch.chdir(host)
    service = local_space
    async with sandbox_for(service) as sandbox:
        await sandbox.check(cwd=str(guest))
        definitions = space_harness_tools(sandbox, cwd=str(guest))
        assert tuple(d.tool.name for d in definitions) == BASE_TOOL_NAMES
        tools = {d.tool.name: cast(Tool[Any, Any], d.tool) for d in definitions}
        assert not isinstance(tools["agent"], Tool)
        assert tools["write_plan"].retry_on_timeout
        assert all(
            not tools[name].retry_on_timeout
            for name in ("read", "find", "grep", "patch", "write", "bash")
        )

        await _invoke(tools["write"], {"path": "note.txt", "content": "alpha\nβeta\n"})
        assert await _invoke(tools["read"], {"path": "note.txt"}) == "alpha\nβeta"
        assert "note.txt" in await _invoke(tools["find"], {"pattern": "*.txt"})
        assert "note.txt:1:alpha" in await _invoke(tools["grep"], {"pattern": "alpha"})
        await _invoke(
            tools["patch"],
            {
                "path": "note.txt",
                "edits": "<<<<<<< SEARCH\nalpha\n=======\ngamma\n>>>>>>> REPLACE",
            },
        )
        assert await _invoke(tools["read"], {"path": "note.txt"}) == "gamma\nβeta"
        assert (
            await _invoke(tools["bash"], {"command": "cat note.txt"}) == "gamma\nβeta"
        )
        failure = await _invoke(tools["bash"], {"command": "exit 7"})
        assert isinstance(failure, ToolFailure)
        assert failure.error.kind == "nonzero_exit"
        assert (host / "note.txt").read_text() == "host sentinel"
        assert all(command.cwd == str(guest) for command in service.commands.values())
        assert str(host) not in repr(definitions)

        local = {d.tool.name: d.tool.schema for d in base_harness_tools(cwd=guest)}
        assert all(
            d.tool.schema["parameters"] == local[d.tool.name]["parameters"]
            for d in definitions
        )


@pytest.mark.asyncio
async def test_bundle_limits_fail_before_remote_dispatch(
    local_space: LocalSpace, tmp_path: Path
) -> None:
    service = local_space
    async with sandbox_for(service) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        tools = {
            d.tool.name: cast(Tool[Any, Any], d.tool)
            for d in space_harness_tools(sandbox, cwd=str(tmp_path))
        }
        for name, arguments in (
            ("write", {"path": "target", "content": "é" * 25000}),
            ("patch", {"path": "target", "edits": "é" * 25000}),
            ("bash", {"command": "é" * 25000}),
        ):
            result = await _invoke(tools[name], arguments)
            assert isinstance(result, ToolFailure)
            assert result.error.kind == "input_limit"
        assert len(service.commands) == 1
        assert not (tmp_path / "target").exists()


@pytest.mark.asyncio
async def test_remote_failure_never_falls_back_and_blocks_reuse(
    local_space: LocalSpace, tmp_path: Path
) -> None:
    service = local_space
    async with sandbox_for(service) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        tool = cast(
            Tool[Any, Any],
            space_harness_tools(sandbox, cwd=str(tmp_path), include=("write",))[0].tool,
        )
        service.fail_stream = True
        result = await _invoke(
            tool, {"path": "target", "content": "must not be written"}
        )
        assert isinstance(result, ToolFailure)
        assert sandbox.needs_recovery
        assert not (tmp_path / "target").exists()
        second = await _invoke(tool, {"path": "target", "content": "still blocked"})
        assert isinstance(second, ToolFailure)
        assert second.error.kind == "recovery_required"
        assert len(service.commands) == 2


@pytest.mark.asyncio
async def test_normal_tool_executor_retains_guest_results_and_events(
    local_space: LocalSpace,
    tmp_path: Path,
) -> None:
    service = local_space
    completed: list[object] = []
    async with sandbox_for(service) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        definitions = space_harness_tools(sandbox, cwd=str(tmp_path), include=("bash",))
        call = ToolCall.model_validate(
            {
                "id": "test-call",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"exit 3"}'},
            }
        )

        def on_end(_: ToolCall, result: object) -> None:
            completed.append(result)

        messages = await ToolExecutor().execute(
            tool_calls=[call],
            tools=Tools(tools=tuple(d.tool for d in definitions)),
            on_tool_end=on_end,
        )
        assert messages[0]["tool_call_id"] == "test-call"
        assert "code 3" in messages[0]["content"]
        assert len(completed) == 1
        assert isinstance(completed[0], ToolFailure)
        assert completed[0].error.kind == "nonzero_exit"
        assert len(service.commands) == 2


@pytest.mark.asyncio
async def test_missing_check_and_invalid_selectors_fail_before_dispatch(
    local_space: LocalSpace,
    tmp_path: Path,
) -> None:
    service = local_space
    async with sandbox_for(service) as sandbox:
        with pytest.raises(SpaceError):
            space_harness_tools(sandbox, cwd=str(tmp_path))
        assert not service.commands
        await sandbox.check(cwd=str(tmp_path))
        with pytest.raises(ValueError):
            space_harness_tools(sandbox, cwd=str(tmp_path), include=("unknown",))
        with pytest.raises(ValueError):
            space_harness_tools(
                sandbox, cwd=str(tmp_path), include=("read",), exclude=("read",)
            )
        assert len(service.commands) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, expected_kind", [("stream", "protocol"), ("connect", "transport")]
)
async def test_bash_bundle_preserves_native_failure_kind(
    local_space: LocalSpace, tmp_path: Path, failure: str, expected_kind: str
) -> None:
    service = local_space
    async with sandbox_for(service) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        tool = cast(
            Tool[Any, Any],
            space_harness_tools(sandbox, cwd=str(tmp_path), include=("bash",))[0].tool,
        )
        service.fail_stream = failure == "stream"
        service.fail_connect = failure == "connect"

        result = await _invoke(tool, {"command": "touch must-not-exist"})

        assert isinstance(result, ToolFailure)
        assert result.error.kind == expected_kind
        assert "failed to execute bash command" not in result
        assert sandbox.needs_recovery
        assert not (tmp_path / "must-not-exist").exists()
        second = await _invoke(tool, {"command": "true"})
        assert isinstance(second, ToolFailure)
        assert second.error.kind == "recovery_required"
        assert len(service.commands) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "invalid_json",
        "nonzero",
        "unexpected_info",
        "missing_protocol",
        "boolean_protocol",
        "oversized",
    ],
)
async def test_invalid_worker_result_blocks_queued_dispatch(
    local_space: LocalSpace, tmp_path: Path, failure: str
) -> None:
    (tmp_path / "source.txt").write_text("source")
    service = local_space
    async with sandbox_for(service) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        tools = {
            definition.tool.name: cast(Tool[Any, Any], definition.tool)
            for definition in space_harness_tools(
                sandbox, cwd=str(tmp_path), include=("read", "write")
            )
        }
        if failure == "invalid_json":
            service.worker_response = b"{invalid"
        elif failure == "nonzero":
            service.worker_exit_code = 1
        elif failure == "missing_protocol":
            service.worker_response = b'{"text":"unversioned"}'
        elif failure == "boolean_protocol":
            service.worker_response = b'{"protocol":true,"text":"wrong type"}'
        elif failure == "oversized":
            service.worker_response = json.dumps(
                {"protocol": 1, "text": "x" * (1024 * 1024)}
            ).encode()
        else:
            service.worker_response = json.dumps(
                {
                    "protocol": 1,
                    "info": {
                        "version": "x",
                        "fingerprint": "x",
                        "cwd": str(tmp_path),
                        "tools": [],
                    },
                }
            ).encode()
        gate = asyncio.Event()
        service.stream_gate = gate
        first = asyncio.create_task(_invoke(tools["read"], {"path": "source.txt"}))
        await asyncio.wait_for(service.stream_entered.wait(), timeout=5)
        second = asyncio.create_task(
            _invoke(tools["write"], {"path": "must-not-exist", "content": "blocked"})
        )
        # Let the second invocation queue behind the first native command.
        await asyncio.sleep(0)
        gate.set()
        first_result, second_result = await asyncio.wait_for(
            asyncio.gather(first, second), timeout=10
        )

        assert isinstance(first_result, ToolFailure)
        assert first_result.error.kind == "protocol"
        assert isinstance(second_result, ToolFailure)
        assert second_result.error.kind == "recovery_required"
        assert sandbox.needs_recovery
        assert len(service.commands) == 2  # Readiness check and first tool only.
        assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("project_module", ["agentlane.py", "agentlane_space.py"])
async def test_installed_worker_ignores_modules_in_guest_cwd(
    local_space: LocalSpace, tmp_path: Path, project_module: str
) -> None:
    (tmp_path / project_module).write_text(
        'raise RuntimeError("project module imported")\n'
    )
    (tmp_path / "source.txt").write_text("installed worker")
    service = local_space
    async with sandbox_for(service) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        tool = cast(
            Tool[Any, Any],
            space_harness_tools(sandbox, cwd=str(tmp_path), include=("read",))[0].tool,
        )

        assert await _invoke(tool, {"path": "source.txt"}) == "installed worker"


@pytest.mark.asyncio
async def test_missing_guest_dependency_fails_readiness_without_writing_files(
    local_space: LocalSpace, tmp_path: Path
) -> None:
    local_space.environment["PATH"] = str(tmp_path)
    async with sandbox_for(local_space) as sandbox:
        with pytest.raises(SpaceError, match="guest check failed"):
            await sandbox.check(cwd=str(tmp_path))
        with pytest.raises(SpaceError):
            space_harness_tools(sandbox, cwd=str(tmp_path))
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("find", {"pattern": "**/*.txt"}),
        ("grep", {"pattern": "needle", "outputMode": "content"}),
        ("grep", {"pattern": "needle", "outputMode": "files_with_matches"}),
        ("grep", {"pattern": "needle", "outputMode": "count"}),
    ],
)
async def test_guest_search_preserves_native_ignore_and_symlink_rules(
    local_space: LocalSpace,
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    (tmp_path / "keep.txt").write_text("needle\nneedle again\n")
    (tmp_path / "ignored.txt").write_text("needle hidden\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "cycle").symlink_to(tmp_path, target_is_directory=True)
    native = cast(
        Tool[Any, Any], base_harness_tools(cwd=tmp_path, include=(tool_name,))[0].tool
    )
    async with sandbox_for(local_space) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        remote = cast(
            Tool[Any, Any],
            space_harness_tools(sandbox, cwd=str(tmp_path), include=(tool_name,))[
                0
            ].tool,
        )
        actual = await _invoke(remote, arguments)

    assert actual == await _invoke(native, arguments)
    assert "keep.txt" in actual
    assert "ignored.txt" not in actual
    assert "cycle" not in actual


@pytest.mark.asyncio
async def test_concurrent_failed_recheck_leaves_no_checked_metadata(
    local_space: LocalSpace, tmp_path: Path
) -> None:
    responses = 0

    def fail_second_check(payload: bytes) -> bytes:
        nonlocal responses
        responses += 1
        if responses == 2:
            response = json.loads(payload)
            response["info"]["fingerprint"] = "different-build"
            return json.dumps(response).encode()
        return payload

    local_space.transform_response = fail_second_check
    gate = asyncio.Event()
    local_space.stream_gate = gate
    async with sandbox_for(local_space) as sandbox:
        first = asyncio.create_task(sandbox.check(cwd=str(tmp_path)))
        await asyncio.wait_for(local_space.stream_entered.wait(), 5)
        second = asyncio.create_task(sandbox.check(cwd=str(tmp_path)))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.wait_for(first, 5)
        with pytest.raises(SpaceError):
            await asyncio.wait_for(second, 5)
        with pytest.raises(SpaceError) as unchecked:
            space_harness_tools(sandbox, cwd=str(tmp_path))
        assert unchecked.value.kind is SpaceErrorKind.NOT_CHECKED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("write", {"path": "must-not-exist", "content": "blocked"}),
        ("bash", {"command": "touch must-not-exist"}),
    ],
)
async def test_queued_tool_rechecks_readiness_before_dispatch(
    local_space: LocalSpace,
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    async with sandbox_for(local_space) as sandbox:
        await sandbox.check(cwd=str(tmp_path))
        tool = cast(
            Tool[Any, Any],
            space_harness_tools(sandbox, cwd=str(tmp_path), include=(tool_name,))[
                0
            ].tool,
        )
        gate = asyncio.Event()
        local_space.stream_gate = gate
        active = asyncio.create_task(
            sandbox.execute(
                ("true",), cwd=str(tmp_path), cancellation_token=CancellationToken()
            )
        )
        await asyncio.wait_for(local_space.stream_entered.wait(), 5)
        local_space.worker_response = b'{"protocol":1}'
        recheck = asyncio.create_task(sandbox.check(cwd=str(tmp_path)))
        await asyncio.sleep(0)
        queued = asyncio.create_task(_invoke(tool, arguments))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.wait_for(active, 5)
        with pytest.raises(SpaceError):
            await asyncio.wait_for(recheck, 5)
        result = await asyncio.wait_for(queued, 5)

        assert isinstance(result, ToolFailure)
        assert result.error.kind == "not_checked"
        assert len(local_space.commands) == 3
        assert not (tmp_path / "must-not-exist").exists()
