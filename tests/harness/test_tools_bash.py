"""Tests for the first-party bash harness tool."""

import asyncio
import importlib
import json
import os
import shlex
import signal
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    BashExecutionRequest,
    BashExecutionResult,
    BashPolicyDecision,
    HarnessToolsShim,
    bash_tool,
)
from agentlane.harness.tools._output import TruncatedOutput
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    ToolCall,
    ToolExecutor,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def _full_output_path_from_output(output: str) -> Path:
    full_output_line = next(
        line for line in output.splitlines() if "Full output: " in line
    )
    return Path(
        full_output_line.rsplit("Full output: ", maxsplit=1)[1].removesuffix("]")
    )


def _unlink_full_output_path(output: str) -> None:
    _full_output_path_from_output(output).unlink(missing_ok=True)


def _run_bash(
    command: str,
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> str:
    async def scenario() -> str:
        definition = bash_tool(cwd=cwd)
        args_model = definition.tool.args_type()
        kwargs: dict[str, Any] = {"command": command}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await definition.tool.run(args_model(**kwargs), CancellationToken())

    return asyncio.run(scenario())


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


def _make_bash_tool_call(command: str) -> ToolCall:
    return _make_tool_call(
        tool_id="call_1",
        name="bash",
        arguments=json.dumps({"command": command}),
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


class _FakeBashExecutor:
    def __init__(self) -> None:
        self.requests: list[BashExecutionRequest] = []

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        del cancellation_token
        self.requests.append(request)
        return BashExecutionResult(
            command=request.command,
            cwd=request.cwd,
            exit_code=0,
            timed_out=False,
            cancelled=False,
            timeout_seconds=request.timeout_seconds,
            output=TruncatedOutput(text="fake output\n", truncated=False),
            full_output_path=None,
        )


class _DenyBashPolicy:
    def check(self, request: BashExecutionRequest) -> BashPolicyDecision:
        del request
        return BashPolicyDecision(allowed=False, reason="bash command denied by test")


def test_bash_tool_runs_successful_command(tmp_path: Path) -> None:
    output = _run_bash("printf 'hello\\n'", cwd=tmp_path)

    assert output == "hello"


def test_bash_tool_rejects_invalid_default_timeout() -> None:
    with pytest.raises(ValueError, match="default_timeout must be greater than zero"):
        bash_tool(default_timeout=0)


def test_bash_tool_rejects_invalid_command_arguments(tmp_path: Path) -> None:
    definition = bash_tool(cwd=tmp_path)
    args_model = definition.tool.args_type()

    assert (
        asyncio.run(definition.tool.run(args_model(command=""), CancellationToken()))
        == "command must not be empty"
    )
    assert (
        asyncio.run(
            definition.tool.run(
                args_model(command="pwd", timeout=0), CancellationToken()
            )
        )
        == "timeout must be greater than zero"
    )


def test_bash_tool_adapter_passes_request_to_executor(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, list[BashExecutionRequest]]:
        executor = _FakeBashExecutor()
        definition = bash_tool(cwd=tmp_path, executor=executor)
        args_model = definition.tool.args_type()
        output = await definition.tool.run(
            args_model(command="pwd", timeout=3),
            CancellationToken(),
        )
        return output, executor.requests

    output, requests = asyncio.run(scenario())

    assert output == "fake output"
    assert len(requests) == 1
    assert requests[0] == BashExecutionRequest(
        command="pwd",
        cwd=tmp_path,
        timeout_seconds=3,
    )


def test_bash_tool_policy_can_deny_before_executor_runs(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, list[BashExecutionRequest]]:
        executor = _FakeBashExecutor()
        definition = bash_tool(
            cwd=tmp_path,
            executor=executor,
            policy=_DenyBashPolicy(),
        )
        args_model = definition.tool.args_type()
        output = await definition.tool.run(
            args_model(command="pwd"),
            CancellationToken(),
        )
        return output, executor.requests

    output, requests = asyncio.run(scenario())

    assert output == "bash command denied by test"
    assert requests == []


def test_bash_tool_returns_nonzero_exit_code() -> None:
    output = _run_bash("printf 'before fail\\n'; exit 7")

    assert output == "before fail\n\n[Command exited with code 7]"


def test_bash_tool_renders_empty_successful_output() -> None:
    output = _run_bash(":")

    assert output == "(no output)"


def test_bash_tool_captures_stderr() -> None:
    output = _run_bash("printf 'problem\\n' >&2")

    assert output == "problem"


def test_bash_tool_captures_stdout_and_stderr() -> None:
    output = _run_bash("printf 'out\\n'; printf 'err\\n' >&2")

    assert "out" in output
    assert "err" in output


def test_bash_tool_uses_configured_cwd(tmp_path: Path) -> None:
    (tmp_path / "visible.txt").write_text("content\n", encoding="utf-8")

    output = _run_bash("pwd; ls", cwd=tmp_path)

    assert str(tmp_path) in output
    assert "visible.txt" in output


def test_bash_tool_reports_missing_configured_cwd(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    output = _run_bash("pwd", cwd=missing)

    assert output == f"working directory not found: `{missing}`"


def test_bash_tool_honors_per_call_timeout(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    output = _run_bash(f"sleep 5; touch {marker}", cwd=tmp_path, timeout=0.1)

    assert "[Command timed out after 0.1 seconds]" in output
    assert marker.exists() is False


def test_bash_tool_honors_cancellation_token(tmp_path: Path) -> None:
    async def scenario() -> str:
        definition = bash_tool(cwd=tmp_path)
        args_model = definition.tool.args_type()
        token = CancellationToken()
        task = asyncio.create_task(
            definition.tool.run(args_model(command="sleep 5"), token)
        )

        await asyncio.sleep(0.1)
        token.cancel()
        return await task

    output = asyncio.run(scenario())

    assert output == "(no output)\n\n[Command cancelled]"


def test_bash_tool_does_not_hang_on_inherited_pipe_handles(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires POSIX fork semantics")

    python_code = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(5)\n"
        "else:\n"
        "    print('done', flush=True)\n"
        "    os._exit(0)\n"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(python_code)}"
    output = _run_bash(command, cwd=tmp_path, timeout=1)

    assert output == "done"


def test_bash_tool_does_not_execute_when_token_already_cancelled(
    tmp_path: Path,
) -> None:
    async def scenario() -> str:
        marker = tmp_path / "should-not-exist"
        definition = bash_tool(cwd=tmp_path)
        args_model = definition.tool.args_type()
        token = CancellationToken()
        token.cancel()
        output = await definition.tool.run(
            args_model(command=f"touch {marker}"),
            token,
        )
        assert marker.exists() is False
        return output

    output = asyncio.run(scenario())

    assert output == "(no output)\n\n[Command cancelled]"


def test_bash_tool_tail_truncates_large_output_and_preserves_full_log() -> None:
    output = _run_bash(
        "for ((i=0; i<3000; i++)); do "
        "printf 'line-%04d-abcdefghijklmnopqrstuvwxyz0123456789\\n' \"$i\"; "
        "done"
    )

    try:
        assert "Showing last 2000 lines or 51200 bytes. Full output:" in output
        assert "[output truncated:" not in output
        assert "line-2999" in output
        assert "line-0000" not in output

        full_output_path = _full_output_path_from_output(output)
        assert full_output_path.exists()
        full_output_text = full_output_path.read_text(encoding="utf-8")
        assert "line-0000" in full_output_text
        assert "line-2999" in full_output_text
    finally:
        _unlink_full_output_path(output)


def test_bash_tool_line_truncation_preserves_full_log() -> None:
    output = _run_bash(
        "for ((i=0; i<2500; i++)); do printf 'line-%04d\\n' \"$i\"; done"
    )

    try:
        assert "Showing last 2000 lines or 51200 bytes. Full output:" in output
        assert "[output truncated:" not in output
        assert "line-2499" in output
        assert "line-0000" not in output

        full_output_path = _full_output_path_from_output(output)
        assert full_output_path.exists()
        full_output_text = full_output_path.read_text(encoding="utf-8")
        assert "line-0000" in full_output_text
        assert "line-2499" in full_output_text
    finally:
        _unlink_full_output_path(output)


def test_bash_tool_windows_graceful_signal_uses_process_group_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWindowsProcess:
        pid = 123
        returncode: int | None = None

        def __init__(self) -> None:
            self.signals: list[object] = []
            self.terminated = False

        def send_signal(self, sig: object) -> None:
            self.signals.append(sig)

        def terminate(self) -> None:
            self.terminated = True

    ctrl_break_event = object()
    process = FakeWindowsProcess()

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", ctrl_break_event, raising=False)

    bash_executor = importlib.import_module("agentlane.harness.tools._bash_executor")
    helper_name = "_send_process_signal"
    send_process_signal = getattr(bash_executor, helper_name)
    send_process_signal(cast(Any, process), signal.SIGTERM)

    assert process.signals == [ctrl_break_event]
    assert process.terminated is False


def test_bash_tool_supports_quotes_and_shell_expansion() -> None:
    output = _run_bash("name='Agent Lane'; printf \"hello ${name}\\n\"")

    assert output == "hello Agent Lane"


def test_bash_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raise_unexpected_error(*args: object, **kwargs: object) -> object:
        del args
        del kwargs
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", raise_unexpected_error)

    output = _run_bash("printf 'hidden\\n'", cwd=tmp_path)

    assert output == "failed to execute bash command"


def test_bash_tool_executes_through_tool_executor() -> None:
    async def scenario() -> list[dict[str, Any]]:
        definition = bash_tool()
        executor = ToolExecutor()
        return await executor.execute(
            tool_calls=[_make_bash_tool_call("printf 'executor\\n'")],
            tools=Tools(tools=[definition.tool]),
        )

    messages = asyncio.run(scenario())

    assert messages[0]["role"] == "tool"
    assert messages[0]["name"] == "bash"
    assert messages[0]["content"] == "executor"


def test_bash_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
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
                            name="bash",
                            arguments='{"command":"printf \\"runner content\\\\n\\""}',
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
                name="BashRunner",
                model=model,
                tools=Tools(
                    tools=[bash_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"bash": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Run the requested shell command before answering.",
            history=["run the command"],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["bash"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message["role"] == "tool"
        assert tool_message["name"] == "bash"
        assert tool_message["content"] == "runner content"

    asyncio.run(scenario())


def test_bash_tool_prompt_snippet_through_harness_tools_shim() -> None:
    async def scenario() -> str:
        shim = HarnessToolsShim((bash_tool(),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = RunState(
            instructions="Base",
            history=[],
            responses=[],
            turn_count=1,
        )
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)
        return cast(str, state.instructions)

    instructions = asyncio.run(scenario())

    assert (
        instructions == "Base\n\n"
        "<default_tools>\n"
        "Available tools:\n"
        "- bash: Execute non-interactive bash commands\n\n"
        "Guidelines:\n"
        "- Use dedicated file tools for direct file reads, writes, searches, "
        "and patches when they fit.\n"
        "- Use bash for shell workflows, short inspection commands, and "
        "commands that need existing CLIs.\n"
        "- Prefer `rg` over `grep` or `find` when searching from bash.\n"
        "- Avoid interactive commands; bash does not accept follow-up stdin.\n"
        "- Set `timeout` for commands that may hang or run for a long time.\n"
        "- The bash tool does not provide sandboxing, approvals, or "
        "permission enforcement.\n"
        "</default_tools>"
    )
