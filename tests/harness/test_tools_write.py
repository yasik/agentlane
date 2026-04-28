import asyncio
from pathlib import Path
from typing import cast

import pytest
from tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_state,
    run_tool,
)

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import HarnessToolsShim, ToolPathResolver, write_tool
from agentlane.models import Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


def test_write_tool_creates_new_file_and_reports_byte_count(tmp_path: Path) -> None:
    output = run_tool(
        write_tool(cwd=tmp_path),
        path="notes/today.txt",
        content="hello\nworld",
    )

    target = tmp_path / "notes" / "today.txt"
    assert output == f"Wrote 11 bytes to {target}."
    assert target.read_text(encoding="utf-8") == "hello\nworld"


def test_write_tool_resolves_relative_paths_from_configured_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    output = run_tool(
        write_tool(cwd=workspace),
        path="notes.txt",
        content="relative",
    )

    assert output == f"Wrote 8 bytes to {workspace / 'notes.txt'}."
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "relative"


def test_write_tool_accepts_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "absolute.txt"

    output = run_tool(
        write_tool(cwd=tmp_path / "elsewhere"),
        path=str(target),
        content="absolute",
    )

    assert output == f"Wrote 8 bytes to {target}."
    assert target.read_text(encoding="utf-8") == "absolute"


def test_write_tool_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old content", encoding="utf-8")

    output = run_tool(
        write_tool(cwd=tmp_path),
        path="existing.txt",
        content="replacement",
    )

    assert output == f"Wrote 11 bytes to {target}."
    assert target.read_text(encoding="utf-8") == "replacement"


def test_write_tool_preserves_exact_and_empty_content(tmp_path: Path) -> None:
    definition = write_tool(cwd=tmp_path)
    exact_content = "  leading\n\ntrailing  "

    exact_output = run_tool(
        definition,
        path="exact.txt",
        content=exact_content,
    )
    empty_output = run_tool(
        definition,
        path="empty.txt",
        content="",
    )

    assert exact_output == f"Wrote 21 bytes to {tmp_path / 'exact.txt'}."
    assert empty_output == f"Wrote 0 bytes to {tmp_path / 'empty.txt'}."
    assert (tmp_path / "exact.txt").read_text(encoding="utf-8") == exact_content
    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""


def test_write_tool_reports_directory_path(tmp_path: Path) -> None:
    output = run_tool(write_tool(cwd=tmp_path), path=".", content="content")

    assert output == f"path is a directory: `{tmp_path}`"


def test_write_tool_rejects_invalid_path_and_content(tmp_path: Path) -> None:
    definition = write_tool(cwd=tmp_path)

    assert run_tool(definition, path="", content="content") == "path must not be empty"
    assert run_tool(definition, path="bad\x00name.txt", content="content") == (
        "path contains a null byte"
    )
    assert run_tool(definition, path="notes.txt", content="\ud800") == (
        "content is not valid UTF-8"
    )


def test_write_tool_reports_parent_path_file(tmp_path: Path) -> None:
    blocking_file = tmp_path / "blocked"
    blocking_file.write_text("not a directory", encoding="utf-8")

    output = run_tool(
        write_tool(cwd=tmp_path),
        path="blocked/child.txt",
        content="content",
    )

    assert output == f"parent path is not a directory: `{blocking_file}`"
    assert (blocking_file / "child.txt").exists() is False


def test_write_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(self: ToolPathResolver, path: str | Path) -> Path:
        del self
        del path
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(ToolPathResolver, "resolve", raise_unexpected_error)

    output = run_tool(write_tool(cwd=tmp_path), path="notes.txt", content="content")

    assert output == "failed to write file"


def test_write_tool_sanitizes_os_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_os_error(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> int:
        del self
        del args
        del kwargs
        raise OSError("Traceback (most recent call last): private details")

    monkeypatch.setattr(Path, "write_text", raise_os_error)

    output = run_tool(write_tool(cwd=tmp_path), path="notes.txt", content="content")

    assert output == f"failed to write file: `{tmp_path / 'notes.txt'}`"


def test_write_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="write",
                            arguments='{"path":"runner.txt","content":"from runner"}',
                        )
                    ],
                ),
                make_assistant_response(content="done"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="WriteRunner",
                model=model,
                tools=Tools(
                    tools=[write_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"write": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Write the requested file before answering.",
            history=["create runner.txt"],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        assert (tmp_path / "runner.txt").read_text(encoding="utf-8") == "from runner"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["write"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "write",
            "content": f"Wrote 11 bytes to {tmp_path / 'runner.txt'}.",
        }

    asyncio.run(scenario())


def test_write_tool_error_result_does_not_crash_runner_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="write",
                            arguments=(
                                '{"path":"bad\\u0000name.txt",'
                                '"content":"from runner"}'
                            ),
                        )
                    ],
                ),
                make_assistant_response(content="handled"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="WriteRunner",
                model=model,
                tools=Tools(tools=[write_tool(cwd=tmp_path).tool]),
            ),
        )
        state = RunState(
            instructions="Write the requested file before answering.",
            history=["create a file with an invalid path"],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "handled"
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "write",
            "content": "path contains a null byte",
        }

    asyncio.run(scenario())


def test_write_tool_prompt_metadata_renders_through_shim(tmp_path: Path) -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((write_tool(cwd=tmp_path),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["write"]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- write: Create or overwrite files\n\n"
            "Guidelines:\n"
            "- Use write only for new files or complete rewrites.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())
