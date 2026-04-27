import asyncio
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

import agentlane.harness.tools._read as read_module
import agentlane.harness.tools._write as write_module
from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    TEXT_MAX_BYTES,
    TEXT_MAX_LINES,
    GitignoreMatcher,
    HarnessToolDefinition,
    HarnessToolsShim,
    ToolPathResolver,
    base_harness_tools,
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


def _run_read(definition: HarnessToolDefinition, **arguments: object) -> str:
    args_model = definition.tool.args_type()
    return asyncio.run(
        definition.tool.run(
            args_model(**arguments),
            CancellationToken(),
        )
    )


def _run_plan(definition: HarnessToolDefinition, **arguments: object) -> str:
    args_model = definition.tool.args_type()
    return asyncio.run(
        definition.tool.run(
            args_model(**arguments),
            CancellationToken(),
        )
    )


def _run_write(definition: HarnessToolDefinition, **arguments: object) -> str:
    args_model = definition.tool.args_type()
    return asyncio.run(
        definition.tool.run(
            args_model(**arguments),
            CancellationToken(),
        )
    )


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

    output = _run_read(read_tool(cwd=tmp_path), path="notes.txt")

    assert output == (
        f"Absolute path: {tmp_path / 'notes.txt'}\n" "L1: alpha\n" "L2: bravo"
    )


def test_read_tool_resolves_relative_paths_from_configured_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("relative\n", encoding="utf-8")

    output = _run_read(read_tool(cwd=workspace), path="notes.txt")

    assert output == f"Absolute path: {workspace / 'notes.txt'}\nL1: relative"


def test_read_tool_accepts_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "absolute.txt"
    target.write_text("absolute\n", encoding="utf-8")

    output = _run_read(read_tool(cwd=tmp_path / "elsewhere"), path=str(target))

    assert output == f"Absolute path: {target}\nL1: absolute"


def test_read_tool_starts_at_offset(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    output = _run_read(read_tool(cwd=tmp_path), path="notes.txt", offset=2)

    assert output == f"Absolute path: {tmp_path / 'notes.txt'}\nL2: two\nL3: three"


def test_read_tool_applies_caller_limit_before_global_truncation(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    output = _run_read(read_tool(cwd=tmp_path), path="notes.txt", limit=2)

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

    output = _run_read(read_tool(cwd=tmp_path), path="long.txt")
    lines = output.splitlines()

    assert lines[0] == f"Absolute path: {tmp_path / 'long.txt'}"
    assert lines[1] == "L1: x"
    assert lines[TEXT_MAX_LINES] == f"L{TEXT_MAX_LINES}: x"
    assert lines[-1] == f"More than {TEXT_MAX_LINES} lines found"


def test_read_tool_truncates_by_global_byte_limit(tmp_path: Path) -> None:
    (tmp_path / "wide.txt").write_text("a" * (TEXT_MAX_BYTES + 1), encoding="utf-8")

    output = _run_read(read_tool(cwd=tmp_path), path="wide.txt")
    lines = output.splitlines()

    assert lines[0] == f"Absolute path: {tmp_path / 'wide.txt'}"
    assert lines[-1] == f"Output truncated after {TEXT_MAX_BYTES} bytes"
    assert lines[1].startswith("L1: ")
    assert len(lines[1].encode("utf-8")) == TEXT_MAX_BYTES


def test_read_tool_reports_missing_file(tmp_path: Path) -> None:
    output = _run_read(read_tool(cwd=tmp_path), path="missing.txt")

    assert output == f"file not found: `{tmp_path / 'missing.txt'}`"


def test_read_tool_reports_directory_path(tmp_path: Path) -> None:
    output = _run_read(read_tool(cwd=tmp_path), path=".")

    assert output == f"path is a directory: `{tmp_path}`"


def test_read_tool_reports_binary_file(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"text\x00more")

    output = _run_read(read_tool(cwd=tmp_path), path="data.bin")

    assert output == (
        "file appears to be binary and cannot be read as text: "
        f"`{tmp_path / 'data.bin'}`"
    )


def test_read_tool_decodes_invalid_utf8_with_replacement(tmp_path: Path) -> None:
    (tmp_path / "latin1.txt").write_bytes(b"\xff")

    output = _run_read(read_tool(cwd=tmp_path), path="latin1.txt")

    assert output == f"Absolute path: {tmp_path / 'latin1.txt'}\nL1: \ufffd"


def test_read_tool_rejects_invalid_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")
    definition = read_tool(cwd=tmp_path)

    assert _run_read(definition, path="") == "path must not be empty"
    assert _run_read(definition, path="notes.txt", offset=0) == (
        "offset must be a 1-indexed line number"
    )
    assert _run_read(definition, path="notes.txt", limit=0) == (
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

    output = _run_read(read_tool(cwd=tmp_path), path="notes.txt")

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

    output = _run_read(read_tool(cwd=tmp_path), path="notes.txt")

    assert output == f"failed to read file: `{tmp_path / 'notes.txt'}`"


def test_read_tool_reports_offset_beyond_file_length(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")

    output = _run_read(read_tool(cwd=tmp_path), path="notes.txt", offset=3)

    assert output == "offset exceeds file length"


def test_read_tool_returns_header_only_for_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    output = _run_read(read_tool(cwd=tmp_path), path="empty.txt")

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
