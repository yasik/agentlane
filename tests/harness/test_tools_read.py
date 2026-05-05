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
from agentlane.harness.tools import (
    TEXT_MAX_BYTES,
    TEXT_MAX_LINES,
    HarnessToolsShim,
    ToolPathResolver,
    read_tool,
)
from agentlane.models import Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


def test_read_tool_reads_basic_text_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbravo\n", encoding="utf-8")

    output = run_tool(read_tool(cwd=tmp_path), path="notes.txt")

    assert output == "alpha\nbravo"


def test_read_tool_resolves_relative_paths_from_configured_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("relative\n", encoding="utf-8")

    output = run_tool(read_tool(cwd=workspace), path="notes.txt")

    assert output == "relative"


def test_read_tool_accepts_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "absolute.txt"
    target.write_text("absolute\n", encoding="utf-8")

    output = run_tool(read_tool(cwd=tmp_path / "elsewhere"), path=str(target))

    assert output == "absolute"


def test_read_tool_starts_at_offset(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    output = run_tool(read_tool(cwd=tmp_path), path="notes.txt", offset=2)

    assert output == "two\nthree"


def test_read_tool_applies_caller_limit_before_global_truncation(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    output = run_tool(read_tool(cwd=tmp_path), path="notes.txt", limit=2)

    assert output == (
        "one\n" "two\n" "\n" "[Showing lines 1-2. Use offset=3 to continue.]"
    )


def test_read_tool_truncates_by_global_line_limit(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text(
        "".join("x\n" for _ in range(TEXT_MAX_LINES + 1)),
        encoding="utf-8",
    )

    output = run_tool(read_tool(cwd=tmp_path), path="long.txt")
    lines = output.splitlines()

    assert lines[0] == "x"
    assert lines[TEXT_MAX_LINES - 1] == "x"
    assert lines[-1] == (
        f"[Showing lines 1-{TEXT_MAX_LINES}. "
        f"Use offset={TEXT_MAX_LINES + 1} to continue.]"
    )


def test_read_tool_truncates_by_global_byte_limit(tmp_path: Path) -> None:
    (tmp_path / "wide.txt").write_text(
        "ok\n" + ("a" * TEXT_MAX_BYTES), encoding="utf-8"
    )

    output = run_tool(read_tool(cwd=tmp_path), path="wide.txt")

    assert output == (
        "ok\n"
        "\n"
        f"[Showing lines 1-1 ({TEXT_MAX_BYTES} byte limit). "
        "Use offset=2 to continue.]"
    )


def test_read_tool_reports_oversized_requested_line(tmp_path: Path) -> None:
    (tmp_path / "wide.txt").write_text("a" * (TEXT_MAX_BYTES + 1), encoding="utf-8")

    output = run_tool(read_tool(cwd=tmp_path), path="wide.txt")

    assert output == (
        f"[Line 1 is {TEXT_MAX_BYTES + 1} bytes, exceeds "
        f"{TEXT_MAX_BYTES} byte limit. Use bash to inspect it.]"
    )


def test_read_tool_reports_missing_file(tmp_path: Path) -> None:
    output = run_tool(read_tool(cwd=tmp_path), path="missing.txt")

    assert output == f"file not found: `{tmp_path / 'missing.txt'}`"


def test_read_tool_reports_directory_path(tmp_path: Path) -> None:
    output = run_tool(read_tool(cwd=tmp_path), path=".")

    assert output == f"path is a directory: `{tmp_path}`"


def test_read_tool_reports_binary_file(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"text\x00more")

    output = run_tool(read_tool(cwd=tmp_path), path="data.bin")

    assert output == (
        "file appears to be binary and cannot be read as text: "
        f"`{tmp_path / 'data.bin'}`"
    )


def test_read_tool_decodes_invalid_utf8_with_replacement(tmp_path: Path) -> None:
    (tmp_path / "latin1.txt").write_bytes(b"\xff")

    output = run_tool(read_tool(cwd=tmp_path), path="latin1.txt")

    assert output == "\ufffd"


def test_read_tool_rejects_invalid_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")
    definition = read_tool(cwd=tmp_path)

    assert run_tool(definition, path="") == "path must not be empty"
    assert run_tool(definition, path="notes.txt", offset=0) == (
        "offset must be a 1-indexed line number"
    )
    assert run_tool(definition, path="notes.txt", limit=0) == (
        "limit must be greater than zero"
    )


def test_read_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(self: ToolPathResolver, path: str | Path) -> Path:
        del self
        del path
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(ToolPathResolver, "resolve", raise_unexpected_error)

    output = run_tool(read_tool(cwd=tmp_path), path="notes.txt")

    assert output == "failed to read file"


def test_read_tool_sanitizes_os_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_os_error(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self
        del args
        del kwargs
        raise OSError("Traceback (most recent call last): private details")

    monkeypatch.setattr(Path, "open", raise_os_error)

    output = run_tool(read_tool(cwd=tmp_path), path="notes.txt")

    assert output == f"failed to read file: `{tmp_path / 'notes.txt'}`"


def test_read_tool_reports_offset_beyond_file_length(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")

    output = run_tool(read_tool(cwd=tmp_path), path="notes.txt", offset=3)

    assert output == "offset exceeds file length"


def test_read_tool_returns_empty_output_for_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    output = run_tool(read_tool(cwd=tmp_path), path="empty.txt")

    assert output == ""


def test_read_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "runner.txt").write_text("runner content\n", encoding="utf-8")
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="read",
                            arguments='{"path":"runner.txt"}',
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
                name="ReadRunner",
                model=model,
                tools=Tools(
                    tools=[read_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"read": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Read the requested file before answering.",
            history=["inspect runner.txt"],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["read"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message["role"] == "tool"
        assert tool_message["name"] == "read"
        assert tool_message["content"] == "runner content"

    asyncio.run(scenario())


def test_read_tool_prompt_metadata_renders_through_shim(tmp_path: Path) -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((read_tool(cwd=tmp_path),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["read"]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- read: Read file contents\n\n"
            "Guidelines:\n"
            "- Use read to examine files instead of cat or sed.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())
