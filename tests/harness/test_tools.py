import asyncio
import os
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

import agentlane.harness.tools._find as find_module
import agentlane.harness.tools._grep as grep_module
import agentlane.harness.tools._read as read_module
import agentlane.harness.tools._write as write_module
from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    FIND_DEFAULT_LIMIT,
    TEXT_MAX_BYTES,
    TEXT_MAX_LINES,
    GitignoreMatcher,
    HarnessToolDefinition,
    HarnessToolsShim,
    ToolPathResolver,
    base_harness_tools,
    find_tool,
    grep_tool,
    plan_tool,
    read_tool,
    truncate_output,
    write_tool,
)
from agentlane.models import MessageDict, Model, ModelResponse, Tool, ToolCall, Tools
from agentlane.models.run import DefaultRunContext
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


class _EchoArgs(BaseModel):
    text: str


async def _echo(args: _EchoArgs, cancellation_token: CancellationToken) -> str:
    del cancellation_token
    return args.text


def _tool(name: str) -> Tool[_EchoArgs, str]:
    return Tool(
        name=name,
        description=f"{name} tool.",
        args_model=_EchoArgs,
        handler=_echo,
    )


def _run_state(*, turn_count: int = 1) -> RunState:
    return RunState(
        instructions="Base",
        history=[],
        responses=[],
        turn_count=turn_count,
    )


def _run_tool(definition: HarnessToolDefinition, **arguments: object) -> str:
    args_model = definition.tool.args_type()
    return asyncio.run(
        definition.tool.run(
            args_model(**arguments),
            CancellationToken(),
        )
    )


def _run_plan(definition: HarnessToolDefinition, **arguments: object) -> str:
    return _run_tool(definition, **arguments)


def _run_write(definition: HarnessToolDefinition, **arguments: object) -> str:
    return _run_tool(definition, **arguments)


def _run_grep(definition: HarnessToolDefinition, **arguments: object) -> str:
    return _run_tool(definition, **arguments)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _set_mtime(path: Path, mtime: float) -> None:
    """Pin a file's mtime so mtime-sorted assertions are deterministic."""
    os.utime(path, (mtime, mtime))


def _make_assistant_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


def _make_tool_call(*, tool_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    )


class _SequenceModel(Model[ModelResponse]):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[MessageDict]] = []
        self.call_tools: list[Tools | None] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args
        del schema
        del cancellation_token
        del kwargs

        self.calls.append([dict(message) for message in messages])
        self.call_tools.append(tools)
        return self._responses.pop(0)


def test_base_harness_tools_includes_current_tool_set() -> None:
    definitions = base_harness_tools()

    assert [definition.tool.name for definition in definitions] == [
        "read",
        "find",
        "grep",
        "write",
        "update_plan",
    ]


def test_harness_tools_shim_merges_tools_and_appends_prompt_once() -> None:
    async def scenario() -> None:
        existing = _tool("existing")
        definition = HarnessToolDefinition(
            tool=_tool("read"),
            prompt_snippet="Read file contents",
            prompt_guidelines=[
                "Use read to examine files instead of cat or sed.",
                "Use read to examine files instead of cat or sed.",
            ],
        )
        assert definition.prompt_guidelines == (
            "Use read to examine files instead of cat or sed.",
            "Use read to examine files instead of cat or sed.",
        )
        shim = HarnessToolsShim((definition,))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = _run_state()

        first_turn = PreparedTurn(
            run_state=state,
            tools=Tools(tools=[existing]),
            model_args=None,
        )
        await bound.prepare_turn(first_turn)

        assert first_turn.tools is not None
        assert [tool.name for tool in first_turn.tools.normalized_tools] == [
            "existing",
            "read",
        ]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- read: Read file contents\n\n"
            "Guidelines:\n"
            "- Use read to examine files instead of cat or sed.\n"
            "</default_tools>"
        )
        instructions_after_first = state.instructions

        second_turn = PreparedTurn(
            run_state=state,
            tools=Tools(tools=[existing]),
            model_args=None,
        )
        await bound.prepare_turn(second_turn)

        assert state.instructions == instructions_after_first

    asyncio.run(scenario())


def test_harness_tools_shim_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="Duplicate harness tool name: read"):
        HarnessToolsShim(
            (
                HarnessToolDefinition(tool=_tool("read")),
                HarnessToolDefinition(tool=_tool("read")),
            )
        )


def test_plan_tool_updates_plan_with_codex_success_message() -> None:
    output = _run_plan(
        plan_tool(),
        explanation="Track the implementation.",
        plan=[
            {"step": "Inspect implementation", "status": "completed"},
            {"step": "Add tests", "status": "in_progress"},
            {"step": "Update docs", "status": "pending"},
        ],
    )

    assert output == "Plan updated"


def test_plan_tool_accepts_pending_completed_and_empty_plans() -> None:
    assert (
        _run_plan(
            plan_tool(),
            plan=[{"step": "Start", "status": "pending"}],
        )
        == "Plan updated"
    )
    assert (
        _run_plan(
            plan_tool(),
            plan=[{"step": "Finish", "status": "completed"}],
        )
        == "Plan updated"
    )
    assert _run_plan(plan_tool(), plan=[]) == "Plan updated"


def test_plan_tool_sanitizes_unexpected_error_text() -> None:
    def raise_unexpected_error(snapshot: dict[str, object]) -> None:
        del snapshot
        raise RuntimeError("Traceback (most recent call last): private details")

    output = _run_plan(
        plan_tool(persist_to=raise_unexpected_error),
        plan=[{"step": "Start", "status": "pending"}],
    )

    assert output == "failed to update plan"


def test_plan_tool_persists_latest_plan_through_harness_tools_shim() -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((plan_tool(),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = _run_state()
        await bound.on_run_start(state, DefaultRunContext())

        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)
        assert turn.tools is not None
        bound_plan = turn.tools.executable_tools[0]
        args_model = bound_plan.args_type()

        first = args_model(
            plan=[{"step": "Start", "status": "in_progress"}],
        )
        second = args_model(
            plan=[{"step": "Finish", "status": "completed"}],
        )

        await bound_plan.run(first, CancellationToken())
        assert state.shim_state["harness-tools:plan"] == {
            "explanation": None,
            "plan": [{"step": "Start", "status": "in_progress"}],
        }

        await bound_plan.run(second, CancellationToken())
        assert state.shim_state["harness-tools:plan"] == {
            "explanation": None,
            "plan": [{"step": "Finish", "status": "completed"}],
        }

    asyncio.run(scenario())


def test_plan_tool_runs_through_normal_runner_execution() -> None:
    async def scenario() -> None:
        model = _SequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            name="update_plan",
                            arguments=(
                                '{"plan":['
                                '{"step":"Create plan","status":"completed"}]}'
                            ),
                        )
                    ],
                ),
                _make_assistant_response(content="done"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Planner",
                model=model,
                instructions="You create plans.",
                shims=(HarnessToolsShim((plan_tool(),)),),
            )
        )

        result = await agent.run("Plan this task.")

        assert result.final_output == "done"
        run_state = result.run_state
        if run_state is None:
            raise AssertionError("Expected DefaultAgent to return run state.")
        assert run_state.shim_state["harness-tools:plan"] == {
            "explanation": None,
            "plan": [{"step": "Create plan", "status": "completed"}],
        }
        assert any(
            isinstance(message, dict)
            and message["role"] == "tool"
            and message["name"] == "update_plan"
            and message["content"] == "Plan updated"
            for message in run_state.history
        )

    asyncio.run(scenario())


def test_plan_tool_prompt_metadata_renders_through_shim() -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((plan_tool(),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = _run_state()
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)

        assert isinstance(state.instructions, str)
        assert "- update_plan: Update the task plan" in state.instructions
        assert (
            "Use `update_plan` to maintain a visible, step-by-step plan"
            in state.instructions
        )
        assert state.instructions.endswith("</default_tools>")

    asyncio.run(scenario())


def test_tool_path_resolver_captures_and_normalizes_cwd(tmp_path: Path) -> None:
    resolver = ToolPathResolver(cwd=tmp_path)

    assert resolver.resolve("notes/today.md") == tmp_path / "notes" / "today.md"
    assert resolver.resolve(tmp_path / "absolute.txt") == tmp_path / "absolute.txt"


def test_tool_path_resolver_rejects_empty_string(tmp_path: Path) -> None:
    resolver = ToolPathResolver(cwd=tmp_path)

    with pytest.raises(ValueError, match="path must not be empty"):
        resolver.resolve("")


def test_truncate_output_limits_head_by_line_count() -> None:
    output = truncate_output("a\nb\nc\n", max_lines=2, max_bytes=100)

    assert output.truncated is True
    assert output.text == (
        "[output truncated: showing first 2 lines or 100 bytes]\n" "a\nb\n"
    )


def test_truncate_output_limits_tail_by_byte_count() -> None:
    output = truncate_output(
        "alpha\nbravo\ncharlie\n", max_lines=10, max_bytes=8, tail=True
    )

    assert output.truncated is True
    assert output.text == (
        "[output truncated: showing last 10 lines or 8 bytes]\n" "charlie\n"
    )


def test_read_tool_reads_basic_text_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbravo\n", encoding="utf-8")

    output = _run_tool(read_tool(cwd=tmp_path), path="notes.txt")

    assert output == (
        f"Absolute path: {tmp_path / 'notes.txt'}\n" "L1: alpha\n" "L2: bravo"
    )


def test_read_tool_resolves_relative_paths_from_configured_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("relative\n", encoding="utf-8")

    output = _run_tool(read_tool(cwd=workspace), path="notes.txt")

    assert output == f"Absolute path: {workspace / 'notes.txt'}\nL1: relative"


def test_read_tool_accepts_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "absolute.txt"
    target.write_text("absolute\n", encoding="utf-8")

    output = _run_tool(read_tool(cwd=tmp_path / "elsewhere"), path=str(target))

    assert output == f"Absolute path: {target}\nL1: absolute"


def test_read_tool_starts_at_offset(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    output = _run_tool(read_tool(cwd=tmp_path), path="notes.txt", offset=2)

    assert output == f"Absolute path: {tmp_path / 'notes.txt'}\nL2: two\nL3: three"


def test_read_tool_applies_caller_limit_before_global_truncation(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    output = _run_tool(read_tool(cwd=tmp_path), path="notes.txt", limit=2)

    assert output == (
        f"Absolute path: {tmp_path / 'notes.txt'}\n"
        "L1: one\n"
        "L2: two\n"
        "More than 2 lines found"
    )


def test_read_tool_truncates_by_global_line_limit(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text(
        "".join("x\n" for _ in range(TEXT_MAX_LINES + 1)),
        encoding="utf-8",
    )

    output = _run_tool(read_tool(cwd=tmp_path), path="long.txt")
    lines = output.splitlines()

    assert lines[0] == f"Absolute path: {tmp_path / 'long.txt'}"
    assert lines[1] == "L1: x"
    assert lines[TEXT_MAX_LINES] == f"L{TEXT_MAX_LINES}: x"
    assert lines[-1] == f"More than {TEXT_MAX_LINES} lines found"


def test_read_tool_truncates_by_global_byte_limit(tmp_path: Path) -> None:
    (tmp_path / "wide.txt").write_text("a" * (TEXT_MAX_BYTES + 1), encoding="utf-8")

    output = _run_tool(read_tool(cwd=tmp_path), path="wide.txt")
    lines = output.splitlines()

    assert lines[0] == f"Absolute path: {tmp_path / 'wide.txt'}"
    assert lines[-1] == f"Output truncated after {TEXT_MAX_BYTES} bytes"
    assert lines[1].startswith("L1: ")
    assert len(lines[1].encode("utf-8")) == TEXT_MAX_BYTES


def test_read_tool_reports_missing_file(tmp_path: Path) -> None:
    output = _run_tool(read_tool(cwd=tmp_path), path="missing.txt")

    assert output == f"file not found: `{tmp_path / 'missing.txt'}`"


def test_read_tool_reports_directory_path(tmp_path: Path) -> None:
    output = _run_tool(read_tool(cwd=tmp_path), path=".")

    assert output == f"path is a directory: `{tmp_path}`"


def test_read_tool_reports_binary_file(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"text\x00more")

    output = _run_tool(read_tool(cwd=tmp_path), path="data.bin")

    assert output == (
        "file appears to be binary and cannot be read as text: "
        f"`{tmp_path / 'data.bin'}`"
    )


def test_read_tool_decodes_invalid_utf8_with_replacement(tmp_path: Path) -> None:
    (tmp_path / "latin1.txt").write_bytes(b"\xff")

    output = _run_tool(read_tool(cwd=tmp_path), path="latin1.txt")

    assert output == f"Absolute path: {tmp_path / 'latin1.txt'}\nL1: \ufffd"


def test_read_tool_rejects_invalid_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")
    definition = read_tool(cwd=tmp_path)

    assert _run_tool(definition, path="") == "path must not be empty"
    assert _run_tool(definition, path="notes.txt", offset=0) == (
        "offset must be a 1-indexed line number"
    )
    assert _run_tool(definition, path="notes.txt", limit=0) == (
        "limit must be greater than zero"
    )


def test_read_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(args: object, *, resolver: object) -> str:
        del args
        del resolver
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(read_module, "_read_file", raise_unexpected_error)

    output = _run_tool(read_tool(cwd=tmp_path), path="notes.txt")

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

    output = _run_tool(read_tool(cwd=tmp_path), path="notes.txt")

    assert output == f"failed to read file: `{tmp_path / 'notes.txt'}`"


def test_read_tool_reports_offset_beyond_file_length(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")

    output = _run_tool(read_tool(cwd=tmp_path), path="notes.txt", offset=3)

    assert output == "offset exceeds file length"


def test_read_tool_returns_header_only_for_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    output = _run_tool(read_tool(cwd=tmp_path), path="empty.txt")

    assert output == f"Absolute path: {tmp_path / 'empty.txt'}"


def test_read_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "runner.txt").write_text("runner content\n", encoding="utf-8")
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            name="read",
                            arguments='{"path":"runner.txt"}',
                        )
                    ],
                ),
                _make_assistant_response(content="done"),
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
        assert "L1: runner content" in cast(str, tool_message["content"])

    asyncio.run(scenario())


def test_read_tool_prompt_metadata_renders_through_shim(tmp_path: Path) -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((read_tool(cwd=tmp_path),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = _run_state()
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


def test_write_tool_creates_new_file_and_reports_byte_count(tmp_path: Path) -> None:
    output = _run_write(
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

    output = _run_write(
        write_tool(cwd=workspace),
        path="notes.txt",
        content="relative",
    )

    assert output == f"Wrote 8 bytes to {workspace / 'notes.txt'}."
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "relative"


def test_write_tool_accepts_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "absolute.txt"

    output = _run_write(
        write_tool(cwd=tmp_path / "elsewhere"),
        path=str(target),
        content="absolute",
    )

    assert output == f"Wrote 8 bytes to {target}."
    assert target.read_text(encoding="utf-8") == "absolute"


def test_write_tool_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old content", encoding="utf-8")

    output = _run_write(
        write_tool(cwd=tmp_path),
        path="existing.txt",
        content="replacement",
    )

    assert output == f"Wrote 11 bytes to {target}."
    assert target.read_text(encoding="utf-8") == "replacement"


def test_write_tool_preserves_exact_and_empty_content(tmp_path: Path) -> None:
    definition = write_tool(cwd=tmp_path)
    exact_content = "  leading\n\ntrailing  "

    exact_output = _run_write(
        definition,
        path="exact.txt",
        content=exact_content,
    )
    empty_output = _run_write(
        definition,
        path="empty.txt",
        content="",
    )

    assert exact_output == f"Wrote 21 bytes to {tmp_path / 'exact.txt'}."
    assert empty_output == f"Wrote 0 bytes to {tmp_path / 'empty.txt'}."
    assert (tmp_path / "exact.txt").read_text(encoding="utf-8") == exact_content
    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""


def test_write_tool_reports_directory_path(tmp_path: Path) -> None:
    output = _run_write(write_tool(cwd=tmp_path), path=".", content="content")

    assert output == f"path is a directory: `{tmp_path}`"


def test_write_tool_rejects_invalid_path_and_content(tmp_path: Path) -> None:
    definition = write_tool(cwd=tmp_path)

    assert (
        _run_write(definition, path="", content="content") == "path must not be empty"
    )
    assert _run_write(definition, path="bad\x00name.txt", content="content") == (
        "path contains a null byte"
    )
    assert _run_write(definition, path="notes.txt", content="\ud800") == (
        "content is not valid UTF-8"
    )


def test_write_tool_reports_parent_path_file(tmp_path: Path) -> None:
    blocking_file = tmp_path / "blocked"
    blocking_file.write_text("not a directory", encoding="utf-8")

    output = _run_write(
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
    def raise_unexpected_error(args: object, *, resolver: object) -> str:
        del args
        del resolver
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(write_module, "_write_file", raise_unexpected_error)

    output = _run_write(write_tool(cwd=tmp_path), path="notes.txt", content="content")

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

    output = _run_write(write_tool(cwd=tmp_path), path="notes.txt", content="content")

    assert output == f"failed to write file: `{tmp_path / 'notes.txt'}`"


def test_write_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            name="write",
                            arguments='{"path":"runner.txt","content":"from runner"}',
                        )
                    ],
                ),
                _make_assistant_response(content="done"),
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
        model = _SequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            name="write",
                            arguments=(
                                '{"path":"bad\\u0000name.txt",'
                                '"content":"from runner"}'
                            ),
                        )
                    ],
                ),
                _make_assistant_response(content="handled"),
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
        state = _run_state()
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


def test_gitignore_matcher_respects_root_rules_and_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(
        "ignored.txt\nbuild/\n*.log\n",
        encoding="utf-8",
    )
    (tmp_path / "build").mkdir()

    matcher = GitignoreMatcher.from_path(tmp_path)

    assert matcher.is_ignored("ignored.txt") is True
    assert matcher.is_ignored(tmp_path / "ignored.txt") is True
    assert matcher.is_ignored(tmp_path / "build", is_dir=True) is True
    assert matcher.is_ignored(tmp_path / "build" / "artifact.js") is True
    assert matcher.is_ignored(tmp_path / "debug.log") is True
    assert matcher.is_ignored(tmp_path / ".git" / "config") is True
    assert matcher.is_ignored(tmp_path / "visible.txt") is False


def test_find_tool_matches_simple_glob(tmp_path: Path) -> None:
    _touch(tmp_path / "alpha.py")
    _touch(tmp_path / "nested" / "beta.py")
    _touch(tmp_path / "notes.md")

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.py")

    assert output == f"Search directory: {tmp_path}\nalpha.py"


def test_find_tool_matches_recursive_glob(tmp_path: Path) -> None:
    _touch(tmp_path / "alpha.json")
    _touch(tmp_path / "nested" / "beta.json")
    _touch(tmp_path / "nested" / "notes.md")
    _set_mtime(tmp_path / "alpha.json", 2000)
    _set_mtime(tmp_path / "nested" / "beta.json", 1000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="**/*.json")

    assert output == f"Search directory: {tmp_path}\nalpha.json\nnested/beta.json"


def test_find_tool_uses_explicit_search_path(tmp_path: Path) -> None:
    _touch(tmp_path / "outside.py")
    _touch(tmp_path / "src" / "inside.py")

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.py", path="src")

    assert output == f"Search directory: {tmp_path / 'src'}\ninside.py"


def test_find_tool_includes_dotfiles(tmp_path: Path) -> None:
    _touch(tmp_path / ".env")
    _touch(tmp_path / "visible.txt")
    _set_mtime(tmp_path / ".env", 2000)
    _set_mtime(tmp_path / "visible.txt", 1000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*")

    assert output == f"Search directory: {tmp_path}\n.env\nvisible.txt"


def test_find_tool_matches_case_insensitively(tmp_path: Path) -> None:
    _touch(tmp_path / "Alpha.py")

    output = _run_tool(find_tool(cwd=tmp_path), pattern="alpha.py")

    assert output == f"Search directory: {tmp_path}\nAlpha.py"


def test_find_tool_supports_brace_expansion(tmp_path: Path) -> None:
    _touch(tmp_path / "alpha.py")
    _touch(tmp_path / "beta.ts")
    _touch(tmp_path / "gamma.md")
    _set_mtime(tmp_path / "alpha.py", 1000)
    _set_mtime(tmp_path / "beta.ts", 2000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.{py,ts}")

    assert output == f"Search directory: {tmp_path}\nbeta.ts\nalpha.py"


def test_find_tool_strips_leading_slash_from_pattern(tmp_path: Path) -> None:
    _touch(tmp_path / "src" / "alpha.py")

    output = _run_tool(find_tool(cwd=tmp_path), pattern="/src/*.py")

    assert output == f"Search directory: {tmp_path}\nsrc/alpha.py"


def test_find_tool_respects_gitignore_and_skips_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(
        "ignored/\n*.tmp\n",
        encoding="utf-8",
    )
    _touch(tmp_path / ".git" / "hidden.py")
    _touch(tmp_path / "ignored" / "skip.py")
    _touch(tmp_path / "visible" / "keep.py")
    _touch(tmp_path / "visible" / "skip.tmp")
    _set_mtime(tmp_path / ".gitignore", 2000)
    _set_mtime(tmp_path / "visible" / "keep.py", 1000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="**/*")

    assert output == f"Search directory: {tmp_path}\n.gitignore\nvisible/keep.py"


def test_find_tool_sorts_by_mtime_newest_first(tmp_path: Path) -> None:
    _touch(tmp_path / "old.txt")
    _touch(tmp_path / "middle.txt")
    _touch(tmp_path / "newest.txt")
    _set_mtime(tmp_path / "old.txt", 1000)
    _set_mtime(tmp_path / "middle.txt", 2000)
    _set_mtime(tmp_path / "newest.txt", 3000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.txt")

    assert output == (f"Search directory: {tmp_path}\nnewest.txt\nmiddle.txt\nold.txt")


def test_find_tool_breaks_mtime_ties_alphabetically(tmp_path: Path) -> None:
    _touch(tmp_path / "zeta.txt")
    _touch(tmp_path / "alpha.txt")
    _touch(tmp_path / "middle.txt")
    _set_mtime(tmp_path / "zeta.txt", 2000)
    _set_mtime(tmp_path / "alpha.txt", 2000)
    _set_mtime(tmp_path / "middle.txt", 1000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.txt")

    assert output == (f"Search directory: {tmp_path}\nalpha.txt\nzeta.txt\nmiddle.txt")


def test_find_tool_marks_result_limit_truncation(tmp_path: Path) -> None:
    _touch(tmp_path / "a.py")
    _touch(tmp_path / "b.py")
    _touch(tmp_path / "c.py")
    _set_mtime(tmp_path / "a.py", 3000)
    _set_mtime(tmp_path / "b.py", 2000)
    _set_mtime(tmp_path / "c.py", 1000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.py", limit=2)

    assert output == (
        f"Search directory: {tmp_path}\n"
        "a.py\n"
        "b.py\n"
        "3 files matched; returned first 2. "
        f"Refine the pattern or raise `limit` (max {FIND_DEFAULT_LIMIT})."
    )


def test_find_tool_marks_limit_capped_at_maximum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(find_module, "FIND_DEFAULT_LIMIT", 2)
    _touch(tmp_path / "a.py")
    _touch(tmp_path / "b.py")
    _touch(tmp_path / "c.py")
    _set_mtime(tmp_path / "a.py", 3000)
    _set_mtime(tmp_path / "b.py", 2000)
    _set_mtime(tmp_path / "c.py", 1000)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.py", limit=10)

    assert output == (
        f"Search directory: {tmp_path}\n"
        "a.py\n"
        "b.py\n"
        "3 files matched; returned first 2 (maximum). "
        "Refine the pattern or narrow `path`."
    )


def test_find_tool_marks_byte_limit_truncation(tmp_path: Path) -> None:
    for index in range(800):
        _touch(tmp_path / f"{index:04d}-{'x' * 72}.txt")

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.txt")
    lines = output.splitlines()

    assert lines[0] == f"Search directory: {tmp_path}"
    assert lines[-1] == (
        f"Output truncated at {TEXT_MAX_BYTES} bytes; "
        "refine the pattern or narrow `path`."
    )
    assert len("\n".join(lines[1:-1]).encode("utf-8")) <= TEXT_MAX_BYTES


def test_find_tool_reports_no_matches(tmp_path: Path) -> None:
    _touch(tmp_path / "notes.md")

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.py")

    assert output == f"Search directory: {tmp_path}\nNo files matched."


def test_find_tool_reports_non_directory_path(tmp_path: Path) -> None:
    _touch(tmp_path / "notes.md")

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.md", path="notes.md")

    assert output == f"path is not a directory: `{tmp_path / 'notes.md'}`"


def test_find_tool_rejects_invalid_pattern_path_and_limit(tmp_path: Path) -> None:
    definition = find_tool(cwd=tmp_path)

    assert _run_tool(definition, pattern=" ") == "pattern must not be empty"
    assert _run_tool(definition, pattern="*.py", path="") == "path must not be empty"
    assert _run_tool(definition, pattern="*.py", limit=0) == (
        "limit must be greater than zero"
    )


def test_grep_tool_searches_directory_with_regex(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def main() -> None:\n    return None\nmain()\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("nothing here\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern=r"main\(", path=".")

    assert result == (
        "Search path: .\n" "app.py:1:def main() -> None:\n" "app.py:3:main()"
    )


def test_grep_tool_searches_explicit_file_path(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path="second.txt")

    assert result == "Search path: second.txt\nsecond.txt:1:needle"


def test_grep_tool_supports_literal_and_ignore_case(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text(
        "axb\nA.B\n",
        encoding="utf-8",
    )

    literal_result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="a.b",
        path="notes.txt",
        literal=True,
        ignoreCase=True,
    )
    regex_result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="a.b",
        path="notes.txt",
        ignoreCase=True,
    )

    assert literal_result == "Search path: notes.txt\nnotes.txt:2:A.B"
    assert regex_result == (
        "Search path: notes.txt\n" "notes.txt:1:axb\n" "notes.txt:2:A.B"
    )


def test_grep_tool_filters_candidates_by_glob(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("needle\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".", glob="*.py")

    assert result == "Search path: .\napp.py:1:needle"


def test_grep_tool_filters_explicit_file_path_by_glob(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="notes.txt",
        glob="*.py",
    )

    assert result == "Search path: notes.txt\nNo matches."


def test_grep_tool_includes_context_and_merges_overlapping_groups(
    tmp_path: Path,
) -> None:
    (tmp_path / "story.txt").write_text(
        "\n".join(
            [
                "zero",
                "one needle",
                "two",
                "three needle",
                "four",
                "five",
                "six",
                "seven",
                "eight needle",
                "nine",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="story.txt",
        context=1,
    )

    assert result == (
        "Search path: story.txt\n"
        "story.txt:1:zero\n"
        "story.txt:2:one needle\n"
        "story.txt:3:two\n"
        "story.txt:4:three needle\n"
        "story.txt:5:four\n"
        "--\n"
        "story.txt:8:seven\n"
        "story.txt:9:eight needle\n"
        "story.txt:10:nine"
    )


def test_grep_tool_respects_gitignore_and_skips_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "Search path: .\nvisible.txt:1:needle"


def test_grep_tool_respects_gitignore_for_explicit_file_path(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path="ignored.txt")

    assert result == "Search path: ignored.txt\nNo matches."


def test_grep_tool_searches_hidden_files_but_skips_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".env").write_text("needle\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == ("Search path: .\n.env:1:needle\nvisible.txt:1:needle")


def test_grep_tool_truncates_long_lines_and_match_limit(tmp_path: Path) -> None:
    long_line = "needle " + ("x" * 600)
    (tmp_path / "long.txt").write_text(
        f"{long_line}\nneedle second\n",
        encoding="utf-8",
    )

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="long.txt",
        limit=1,
    )
    lines = result.splitlines()

    assert lines[0] == "Search path: long.txt"
    assert lines[1].startswith("long.txt:1:needle ")
    assert "needle second" not in result
    assert len(lines[1].split(":", maxsplit=2)[2]) == 500
    assert lines[-2] == "Showing first 1 matches; more remain."
    assert lines[-1] == (
        "One or more matching lines were truncated after 500 characters"
    )


def test_grep_tool_truncates_by_byte_limit(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "\n".join(f"needle {index:04d} " + ("x" * 90) for index in range(700)),
        encoding="utf-8",
    )

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="large.txt",
        limit=1000,
    )

    assert result.endswith(f"\nOutput truncated after {TEXT_MAX_BYTES} bytes")
    assert len(result.encode("utf-8")) < 53 * 1024


def test_grep_tool_reports_no_matches_and_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "text.txt").write_text("plain text\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00needle\x00")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "Search path: .\nNo matches."


def test_grep_tool_continues_when_directory_contains_binary_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x00needle\x00")
    (tmp_path / "text.txt").write_text("needle in text\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "Search path: .\ntext.txt:1:needle in text"


def test_grep_tool_parser_ignores_binary_warnings() -> None:
    output = "\n".join(
        (
            'rg: ./binary.bin: binary file matches (found "\\0" byte around offset 0)',
            (
                '{"type":"match","data":{"path":{"text":"/workspace/text.txt"},'
                '"lines":{"text":"needle\\n"},"line_number":1,'
                '"absolute_offset":0,"submatches":[]}}'
            ),
        )
    )

    parse_ripgrep_output = cast(Any, grep_module)._parse_ripgrep_output
    parsed = parse_ripgrep_output(output)

    assert parsed.error is None
    assert parsed.match_count == 1
    assert parsed.rows[0].path == Path("/workspace/text.txt")


def test_grep_tool_parser_ignores_per_file_warnings_alongside_matches() -> None:
    output = "\n".join(
        (
            "rg: ./locked.txt: Permission denied (os error 13)",
            (
                '{"type":"match","data":{"path":{"text":"/workspace/visible.py"},'
                '"lines":{"text":"hit\\n"},"line_number":1,'
                '"absolute_offset":0,"submatches":[]}}'
            ),
        )
    )

    parse_ripgrep_output = cast(Any, grep_module)._parse_ripgrep_output
    parsed = parse_ripgrep_output(output)

    assert parsed.error is None
    assert parsed.match_count == 1
    assert parsed.rows[0].path == Path("/workspace/visible.py")


def test_grep_tool_listing_mode_filters_warning_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "alpha.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "beta.py").write_text("needle\n", encoding="utf-8")

    class _FakeRipgrep:
        def __init__(self, output: str) -> None:
            self._output = output

        @property
        def as_string(self) -> str:
            return self._output

    fake_output = "\n".join(
        [
            f"{tmp_path / 'alpha.py'}",
            "rg: ./locked.txt: Permission denied (os error 13)",
            f"{tmp_path / 'beta.py'}",
        ]
    )

    real_run = grep_module.Ripgrepy.run

    def fake_run(self: object) -> _FakeRipgrep:
        del self
        return _FakeRipgrep(fake_output)

    monkeypatch.setattr(grep_module.Ripgrepy, "run", fake_run)
    try:
        result = _run_grep(
            grep_tool(cwd=tmp_path),
            pattern="needle",
            path=".",
            outputMode="files_with_matches",
        )
    finally:
        monkeypatch.setattr(grep_module.Ripgrepy, "run", real_run)

    assert result == "Search path: .\nalpha.py\nbeta.py"


def test_grep_tool_reports_binary_file_for_explicit_file_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x00needle\x00")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path="binary.bin")

    assert result == (
        "file appears to be binary and cannot be searched as text: "
        f"`{tmp_path / 'binary.bin'}`"
    )


def test_grep_tool_reports_invalid_regex(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="(", path=".")

    assert result == "invalid regex pattern"


def test_grep_tool_reports_invalid_glob(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".", glob="[")

    assert result == "invalid glob pattern"


@pytest.mark.parametrize(
    "output_mode",
    ["content", "files_with_matches", "count"],
)
def test_grep_tool_reports_invalid_file_type(
    tmp_path: Path,
    output_mode: str,
) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode=output_mode,
        type="notatype",
    )

    assert result == "invalid file type"


def test_grep_tool_reports_missing_ripgrep_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_ripgrep(*args: object, **kwargs: object) -> object:
        del args
        del kwargs
        raise grep_module.RipGrepNotFound("private details")

    monkeypatch.setattr(grep_module, "Ripgrepy", missing_ripgrep)

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "ripgrep executable not found"


def test_grep_tool_rejects_negative_context() -> None:
    args_model = grep_tool().tool.args_type()

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        args_model(pattern="needle", context=-1)



def test_find_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(
        args: object,
        *,
        resolver: object,
        cancellation_token: object,
    ) -> str:
        del args
        del resolver
        del cancellation_token
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(find_module, "_find_files", raise_unexpected_error)

    output = _run_tool(find_tool(cwd=tmp_path), pattern="*.py")

    assert output == "failed to find files"


def test_find_tool_runs_through_runner_tool_execution(tmp_path: Path) -> None:
    async def scenario() -> None:
        _touch(tmp_path / "alpha.py")
        tool_call = _make_tool_call(
            tool_id="call_1",
            name="find",
            arguments='{"pattern":"*.py"}',
        )
        model = _SequenceModel(
            [
                _make_assistant_response(None, tool_calls=[tool_call]),
                _make_assistant_response("complete"),
            ]
        )
        runner = Runner()
        runtime = SingleThreadedRuntimeEngine()
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="Finder",
                model=model,
                tools=Tools(
                    tools=[find_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"find": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Find requested files before answering.",
            history=["Find Python files."],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "complete"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["find"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message["role"] == "tool"
        assert tool_message["name"] == "find"
        assert "alpha.py" in cast(str, tool_message["content"])

    asyncio.run(scenario())


def test_find_tool_prompt_snippet_through_harness_tools_shim(tmp_path: Path) -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((find_tool(cwd=tmp_path),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = _run_state()

        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )
        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["find"]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- find: Find files by glob pattern (use `**/` for recursion)\n\n"
            "Guidelines:\n"
            "- Use find to locate files instead of shelling out to find or ls.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())


def test_find_tool_limit_schema_documents_default_result_limit() -> None:
    schema = find_tool().tool.schema

    assert schema["parameters"]["properties"]["limit"]["default"] == FIND_DEFAULT_LIMIT


def test_grep_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(
        args: object,
        *,
        resolver: object,
    ) -> str:
        del args
        del resolver
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(grep_module, "_search_files", raise_unexpected_error)

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "failed to search files"


def test_grep_tool_sanitizes_os_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

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

    result = _run_grep(grep_tool(cwd=tmp_path), pattern="needle", path="notes.txt")

    assert result == f"failed to read file: `{tmp_path / 'notes.txt'}`"


def test_grep_tool_files_with_matches_mode(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("needle in haystack\n", encoding="utf-8")
    (tmp_path / "bravo.py").write_text("another needle\n", encoding="utf-8")
    (tmp_path / "charlie.py").write_text("no match here\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="files_with_matches",
    )

    assert result == "Search path: .\nalpha.py\nbravo.py"


def test_grep_tool_files_with_matches_mode_reports_no_matches(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("plain text\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="files_with_matches",
    )

    assert result == "Search path: .\nNo matches."


def test_grep_tool_count_mode(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text(
        "needle one\nneedle two\nneedle three\n",
        encoding="utf-8",
    )
    (tmp_path / "bravo.py").write_text("needle solo\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="count",
    )

    assert result == "Search path: .\nalpha.py:3\nbravo.py:1"


def test_grep_tool_filters_by_type(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("needle\n", encoding="utf-8")
    (tmp_path / "script.js").write_text("needle\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        type="py",
    )

    assert result == "Search path: .\napp.py:1:needle"


def test_grep_tool_multiline_pattern_matches_across_newlines(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text(
        "def foo():\n    return 42\n\ndef bar():\n    return 99\n",
        encoding="utf-8",
    )

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern=r"def foo\(\):.+?return",
        path="code.py",
        multiline=True,
    )

    assert "code.py:1:def foo():" in result


def test_grep_tool_caps_files_mode_with_limit(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"file_{index}.txt").write_text("needle\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="files_with_matches",
        limit=2,
    )

    lines = result.splitlines()
    assert lines[0] == "Search path: ."
    assert lines[1] == "file_0.txt"
    assert lines[2] == "file_1.txt"
    assert lines[-1] == "Showing first 2 files; more remain."


def test_grep_tool_count_mode_renders_relative_paths(tmp_path: Path) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "deep.py").write_text("needle one\nneedle two\n", encoding="utf-8")

    result = _run_grep(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="count",
    )

    assert result == "Search path: .\nsrc/deep.py:2"


def test_grep_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = _SequenceModel(
            [
                _make_assistant_response(
                    content=None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            name="grep",
                            arguments='{"pattern":"needle","path":"."}',
                        )
                    ],
                ),
                _make_assistant_response(content="done"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="GrepRunner",
                model=model,
                tools=Tools(
                    tools=[grep_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"grep": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Search the requested files before answering.",
            history=["inspect TODO mentions"],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["grep"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message["role"] == "tool"
        assert tool_message["name"] == "grep"
        assert "notes.txt:1:needle" in cast(str, tool_message["content"])

    asyncio.run(scenario())


def test_grep_tool_prompt_metadata_renders_through_harness_tools_shim(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        definition = grep_tool(cwd=tmp_path)
        shim = HarnessToolsShim((definition,))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = _run_state()

        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["grep"]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- grep: Search file contents for patterns\n"
            "\n"
            "Guidelines:\n"
            '- Use grep with output_mode="files_with_matches" or "count" to '
            "discover where a symbol appears; switch to content mode when you "
            "need surrounding lines.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())
