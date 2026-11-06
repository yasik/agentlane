"""Public contract tests for the jq harness tool."""

import asyncio
import json
import shlex
import subprocess
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.filesystem import BinaryReader, LocalFileSystem, MountedReader
from agentlane.harness.tools import (
    BASH_MAX_LINES,
    BashExecutionRequest,
    BashExecutionResult,
    HarnessToolsShim,
    LocalBashExecutor,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionGrantPolicy,
    ToolPermissionRequest,
    TruncatedOutput,
    WorkspaceToolPermissionPolicy,
    base_harness_tools,
    bash_tool,
    jq_tool,
    parse_tool_permission_grants,
)
from agentlane.models import Tool, ToolFailure
from agentlane.runtime import CancellationToken

from .tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_tool,
)


class _RecordingExecutor:
    """Capture input at the public executor boundary without starting a process."""

    supports_stdin: bool = True

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
            output=TruncatedOutput(text="accepted\n", truncated=False),
            full_output_path=None,
        )


class _LegacyExecutor:
    """Wrap an executor using the original run-only command contract."""

    def __init__(self, delegate: _RecordingExecutor) -> None:
        self._delegate = delegate

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        return await self._delegate.run(request, cancellation_token)


class _CapturingExecutor(_RecordingExecutor):
    """Return an explicit capture result without assuming an executor filesystem."""

    supports_stdout_capture: bool = True

    def __init__(
        self, stdout: bytes | None = b"42\n", *, truncated: bool = False
    ) -> None:
        super().__init__()
        self.stdout = stdout
        self.truncated = truncated

    async def run(
        self,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        result = await super().run(request, cancellation_token)

        return replace(result, stdout=self.stdout, stdout_truncated=self.truncated)


class _UnopenedReader:
    """Fail if permission or argument rejection opens the input."""

    def open_read(self, path: str) -> AbstractContextManager[BinaryReader]:
        raise AssertionError(f"unexpected read: {path}")


class _RecordingPolicy:
    """Record permission requests and optionally deny one operation."""

    def __init__(self, denied_operation: ToolOperation | None = None) -> None:
        self.requests: list[ToolPermissionRequest] = []
        self.denied_operation = denied_operation

    def check(self, request: ToolPermissionRequest) -> ToolPermissionDecision:
        self.requests.append(request)
        if request.operation == self.denied_operation:
            return ToolPermissionDecision.deny()

        return ToolPermissionDecision.allow()


@pytest.mark.parametrize(
    ("source", "jq_filter", "options", "flags"),
    [
        (
            '{"items":[{"id":1,"active":true},{"id":2,"active":false}]}',
            ".items[] | select(.active) | {id}",
            {},
            [],
        ),
        ('{"id":1}\n{"id":2}\n', ".id", {}, []),
        ('{"id":1}\n{"id":2}\n', "map(.id)", {"slurp": True}, ["--slurp"]),
        (
            '{"body":"first\\nsecond — é"}',
            ".body",
            {"raw_output": True},
            ["--raw-output"],
        ),
        (
            '{"value":null,"enabled":false}',
            "., .value, .enabled",
            {"compact_output": True},
            ["--compact-output"],
        ),
    ],
)
def test_jq_matches_installed_cli(
    tmp_path: Path,
    jq_binary: str,
    source: str,
    jq_filter: str,
    options: dict[str, object],
    flags: list[str],
) -> None:
    payload = source.encode("utf-8")
    (tmp_path / "data.json").write_bytes(payload)
    expected = (
        subprocess.run(  # noqa: S603 - installed jq, fixed test input
            [jq_binary, "--monochrome-output", *flags, "--", jq_filter],
            input=payload,
            capture_output=True,
            check=True,
        )
        .stdout.decode("utf-8")
        .rstrip("\n")
    )

    output = run_tool(
        jq_tool(cwd=tmp_path), path="data.json", filter=jq_filter, **options
    )

    assert output == expected
    assert not isinstance(output, ToolFailure)


def test_jq_empty_result_is_distinct_from_json_null(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_text("null", encoding="utf-8")
    definition = jq_tool(cwd=tmp_path)

    assert run_tool(definition, path="data.json", filter="empty") == "(no output)"
    assert run_tool(definition, path="data.json", filter=".") == "null"


def test_jq_treats_filter_and_file_names_as_literal_data(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    name = "$(touch PATH_EXECUTED) ' data.json"
    (tmp_path / name).write_text("null", encoding="utf-8")
    value = "$(touch FILTER_EXECUTED) `touch BACKTICK_EXECUTED` ; 'quoted'"

    output = run_tool(
        jq_tool(cwd=tmp_path),
        path=name,
        filter=json.dumps(value),
        raw_output=True,
    )

    assert output == value
    assert sorted(path.name for path in tmp_path.iterdir()) == [name]


def test_jq_mount_input_is_independent_of_process_working_directory(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    storage = tmp_path / "storage"
    storage.mkdir()
    process_directory = tmp_path / "process"
    process_directory.mkdir()
    (storage / "data.json").write_text('{"id":7}', encoding="utf-8")
    reader = MountedReader({"scratchpad": LocalFileSystem(storage)})

    output = run_tool(
        jq_tool(cwd="/scratchpad", reader=reader, process_cwd=process_directory),
        path="data.json",
        filter="{id, cwd: env.PWD, filename: input_filename}",
    )

    assert json.loads(output) == {
        "id": 7,
        "cwd": str(process_directory),
        "filename": "<stdin>",
    }


def test_jq_passes_unchanged_bytes_and_options_to_custom_executor(
    tmp_path: Path,
) -> None:
    payload = b'{"value":1.0000000000000000000000001}\nfalse\n'
    (tmp_path / "data.json").write_bytes(payload)
    executor = _RecordingExecutor()

    output = run_tool(
        jq_tool(cwd=tmp_path, process_cwd="/remote/work", executor=executor),
        path="data.json",
        filter=".[0].value",
        raw_output=True,
        slurp=True,
        compact_output=True,
        timeout=2.5,
    )

    assert output == "accepted"
    (request,) = executor.requests
    assert request.stdin == payload
    assert request.cwd == Path("/remote/work")
    assert request.timeout_seconds == 2.5
    assert request.shell_startup is False
    arguments = shlex.split(request.command)
    assert arguments[0] == "jq"
    assert set(arguments[1:-2]) == {
        "--monochrome-output",
        "--raw-output",
        "--slurp",
        "--compact-output",
    }
    assert arguments[-2:] == ["--", ".[0].value"]


def test_jq_preserves_custom_executor_default_timeout(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text("null", encoding="utf-8")
    executor = _RecordingExecutor()

    output = run_tool(
        jq_tool(cwd=tmp_path, executor=executor, default_timeout=10.0),
        path="data.json",
        filter=".",
    )

    assert output == "accepted"
    assert executor.requests[0].timeout_seconds is None


@pytest.mark.parametrize("custom_executor", [False, True])
def test_jq_uses_owning_executor_default_timeout(
    tmp_path: Path, jq_binary: str, custom_executor: bool
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_text("null", encoding="utf-8")
    definition = (
        jq_tool(
            cwd=tmp_path,
            executor=LocalBashExecutor(default_timeout=0.1),
            default_timeout=10.0,
        )
        if custom_executor
        else jq_tool(cwd=tmp_path, default_timeout=0.1)
    )

    output = run_tool(definition, path="data.json", filter="until(false; .)")

    assert isinstance(output, ToolFailure)
    assert output.error.kind == "timeout"
    assert "Command timed out after 0.1 seconds" in output


@pytest.mark.parametrize("is_file", [False, True])
def test_jq_reports_local_process_directory_errors_before_input_read(
    tmp_path: Path, is_file: bool
) -> None:
    process_directory = tmp_path / "process"
    if is_file:
        process_directory.write_text("not a directory", encoding="utf-8")

    output = run_tool(
        jq_tool(reader=_UnopenedReader(), process_cwd=process_directory),
        path="data.json",
        filter=".",
    )

    message = (
        "working directory is not a directory"
        if is_file
        else "working directory not found"
    )
    assert output == f"{message}: `{process_directory.resolve()}`"
    assert isinstance(output, ToolFailure)


@pytest.mark.parametrize("declares_capability", [False, True])
def test_jq_rejects_unsupported_executor_before_opening_input(
    declares_capability: bool,
) -> None:
    executor = _RecordingExecutor()
    executor.supports_stdin = False
    supplied_executor = executor if declares_capability else _LegacyExecutor(executor)

    output = run_tool(
        jq_tool(reader=_UnopenedReader(), executor=supplied_executor),
        path="data.json",
        filter=".",
    )

    assert output == "jq requires an executor that supports stdin"
    assert isinstance(output, ToolFailure)
    assert not executor.requests


@pytest.mark.parametrize("failure_stage", ["reader", "executor"])
def test_jq_sanitizes_unexpected_backend_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    (tmp_path / "data.json").write_text("null", encoding="utf-8")
    executor = _RecordingExecutor()

    def fail_open(
        self: LocalFileSystem, path: str
    ) -> AbstractContextManager[BinaryReader]:
        del self, path
        raise RuntimeError("private credentials: secret-canary")

    async def fail_run(
        self: _RecordingExecutor,
        request: BashExecutionRequest,
        cancellation_token: CancellationToken,
    ) -> BashExecutionResult:
        del self, request, cancellation_token
        raise RuntimeError("private credentials: secret-canary")

    if failure_stage == "reader":
        monkeypatch.setattr(LocalFileSystem, "open_read", fail_open)
    else:
        monkeypatch.setattr(_RecordingExecutor, "run", fail_run)

    output = run_tool(
        jq_tool(cwd=tmp_path, executor=executor), path="data.json", filter="."
    )

    assert output == "failed to run jq"
    assert isinstance(output, ToolFailure)
    assert output.error.kind == "error"
    assert output.error.message == "failed to run jq"


@pytest.mark.parametrize(
    "denied_operation", [ToolOperation.READ_FILE, ToolOperation.EXECUTE_COMMAND]
)
def test_jq_checks_permissions_before_reading_or_starting_process(
    tmp_path: Path, denied_operation: ToolOperation
) -> None:
    policy = _RecordingPolicy(denied_operation)
    executor = _RecordingExecutor()

    output = run_tool(
        jq_tool(
            reader=_UnopenedReader(),
            process_cwd=tmp_path,
            executor=executor,
            permissions=policy,
        ),
        path="data.json",
        filter=".",
    )

    assert output.startswith("permission denied: jq")
    assert not executor.requests
    assert policy.requests[0].operation == ToolOperation.READ_FILE
    assert policy.requests[-1].operation == denied_operation
    if denied_operation == ToolOperation.EXECUTE_COMMAND:
        assert policy.requests[-1].cwd == tmp_path
        assert policy.requests[-1].command is not None


def test_jq_path_scope_does_not_implicitly_allow_command_execution(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.json").write_text("null", encoding="utf-8")
    executor = _RecordingExecutor()

    output = run_tool(
        jq_tool(
            cwd=tmp_path,
            executor=executor,
            permissions=WorkspaceToolPermissionPolicy(tmp_path),
        ),
        path="data.json",
        filter=".",
    )

    assert output.startswith("permission denied: jq")
    assert not executor.requests


def test_jq_input_byte_limit_accepts_exact_fit_and_rejects_overflow(
    tmp_path: Path,
) -> None:
    payload = '"é"'.encode()
    target = tmp_path / "data.json"
    target.write_bytes(payload)
    executor = _RecordingExecutor()

    assert (
        run_tool(
            jq_tool(cwd=tmp_path, executor=executor, max_input_bytes=len(payload)),
            path="data.json",
            filter=".",
        )
        == "accepted"
    )
    assert len(executor.requests) == 1

    output = run_tool(
        jq_tool(cwd=tmp_path, executor=executor, max_input_bytes=len(payload) - 1),
        path="data.json",
        filter=".",
    )

    assert output == (f"file exceeds {len(payload) - 1} byte input limit: `{target}`")
    assert len(executor.requests) == 1


@pytest.mark.parametrize(
    ("path", "message"),
    [("missing.json", "file not found"), (".", "path is a directory")],
)
def test_jq_reports_input_path_errors(tmp_path: Path, path: str, message: str) -> None:
    executor = _RecordingExecutor()

    output = run_tool(jq_tool(cwd=tmp_path, executor=executor), path=path, filter=".")

    assert output == f"{message}: `{(tmp_path / path).resolve()}`"
    assert not executor.requests


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"path": "", "filter": "."}, "path must not be empty"),
        ({"path": "data.json", "filter": " "}, "filter must not be empty"),
        ({"path": "data.json", "filter": ".", "timeout": 0}, "timeout"),
        ({"path": "data.json", "filter": ".", "timeout": float("inf")}, "timeout"),
        ({"path": "data.json", "filter": ".", "output_path": " "}, "output_path"),
        (
            {"path": "data.json", "filter": ".", "output_path": "bad\x00.json"},
            "null byte",
        ),
    ],
)
def test_jq_rejects_invalid_arguments_before_io(
    arguments: dict[str, object], message: str
) -> None:
    executor = _RecordingExecutor()

    output = run_tool(jq_tool(reader=_UnopenedReader(), executor=executor), **arguments)

    assert message in output
    assert not executor.requests


def test_jq_keeps_partial_json_stream_output_and_reports_parse_failure(
    tmp_path: Path, jq_binary: str
) -> None:
    source = b'{"id":1}\n{"id":'
    (tmp_path / "data.jsonl").write_bytes(source)
    expected = subprocess.run(  # noqa: S603 - installed jq, fixed test input
        [jq_binary, "--monochrome-output", ".id"],
        input=source,
        capture_output=True,
        check=False,
    )

    output = run_tool(jq_tool(cwd=tmp_path), path="data.jsonl", filter=".id")

    assert expected.returncode != 0
    assert isinstance(output, ToolFailure)
    assert output.error.kind == "nonzero_exit"
    assert expected.stdout.decode().strip() in output
    assert expected.stderr.decode().strip() in output
    assert f"Command exited with code {expected.returncode}" in output


def test_jq_invalid_filter_returns_compiler_diagnostic_and_failure(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_text("null", encoding="utf-8")

    output = run_tool(jq_tool(cwd=tmp_path), path="data.json", filter=".[")

    assert isinstance(output, ToolFailure)
    assert output.error.kind == "nonzero_exit"
    assert "compile error" in output


def test_jq_honors_timeout_for_nonterminating_filter(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_text("null", encoding="utf-8")

    output = run_tool(
        jq_tool(cwd=tmp_path),
        path="data.json",
        filter="until(false; .)",
        timeout=0.1,
    )

    assert isinstance(output, ToolFailure)
    assert output.error.kind == "timeout"


@pytest.mark.asyncio
async def test_jq_honors_cancellation_token_during_filter_execution(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_text("null", encoding="utf-8")
    tool = cast(Tool[BaseModel, str], jq_tool(cwd=tmp_path).tool)
    token = CancellationToken()
    task = asyncio.create_task(
        tool.run(tool.args_type()(path="data.json", filter="until(false; .)"), token)
    )
    await asyncio.sleep(0.1)
    token.cancel()

    output = await asyncio.wait_for(task, timeout=5)

    assert isinstance(output, ToolFailure)
    assert output.error.kind == "cancelled"


def test_jq_preserves_large_output_in_the_existing_command_log(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_text("null", encoding="utf-8")

    output = run_tool(
        jq_tool(cwd=tmp_path),
        path="data.json",
        filter=f"range(0; {BASH_MAX_LINES + 10})",
    )

    assert f"Showing last {BASH_MAX_LINES} lines" in output
    log_line = next(line for line in output.splitlines() if "Full output: " in line)
    log_path = Path(log_line.rsplit("Full output: ", maxsplit=1)[1].removesuffix("]"))
    try:
        full_output = log_path.read_text(encoding="utf-8")
        assert "\n0\n1\n" in full_output
        assert f"\n{BASH_MAX_LINES + 9}\n" in full_output
    finally:
        log_path.unlink(missing_ok=True)


@pytest.mark.parametrize("save", [False, True])
def test_jq_executes_through_runner_with_prompt_and_permission_context(
    tmp_path: Path, jq_binary: str, save: bool
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_text('{"answer":42}', encoding="utf-8")

    async def scenario() -> None:
        policy = _RecordingPolicy()
        definition = jq_tool(cwd=tmp_path, permissions=policy)
        arguments = {"path": "data.json", "filter": ".answer"}
        if save:
            arguments["output_path"] = "answer.json"
        model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="jq-call",
                            name="jq",
                            arguments=json.dumps(arguments),
                        )
                    ],
                ),
                make_assistant_response(content="done"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="JsonReader",
                model=model,
                shims=(HarnessToolsShim((definition,)),),
            ),
        )

        result = await agent.run("Read the answer from data.json.")

        assert result.final_output == "done"
        tools = model.call_tools[0]
        assert tools is not None
        assert [tool.name for tool in tools.normalized_tools] == ["jq"]
        tool_message = next(
            item for item in model.calls[-1] if item.get("role") == "tool"
        )
        assert tool_message["name"] == "jq"
        if save:
            saved_path = (tmp_path / "answer.json").resolve()
            assert saved_path.read_bytes() == b"42\n"
            assert tool_message["content"] == f"Wrote 3 bytes to {saved_path}.\n\n42"
        else:
            assert tool_message["content"] == "42"
        assert definition.prompt_snippet is not None
        assert any(
            definition.prompt_snippet in str(message.get("content", ""))
            for message in model.calls[0]
        )
        expected_operations = [
            ToolOperation.READ_FILE,
            ToolOperation.EXECUTE_COMMAND,
        ]
        if save:
            expected_operations.append(ToolOperation.CREATE_FILE)
        assert [request.operation for request in policy.requests] == expected_operations
        assert all(request.agent_name == "JsonReader" for request in policy.requests)
        assert all(request.tool_call_id == "jq-call" for request in policy.requests)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("jq_filter", "options", "expected"),
    [
        (".count += 1", {}, b'{\n  "count": 2\n}\n'),
        ("., .", {"compact_output": True}, b'{"count":1}\n{"count":1}\n'),
        ('"é\\n\\u0000"', {"raw_output": True}, "é\n\x00\n".encode()),
        ("empty", {}, b""),
    ],
)
def test_jq_saves_exact_stdout_to_input_path(
    tmp_path: Path,
    jq_binary: str,
    jq_filter: str,
    options: dict[str, object],
    expected: bytes,
) -> None:
    del jq_binary
    source = tmp_path / "data.json"
    source.write_bytes(b'{"count":1}')

    output = run_tool(
        jq_tool(cwd=tmp_path),
        path="data.json",
        filter=jq_filter,
        output_path="data.json",
        **options,
    )

    assert not isinstance(output, ToolFailure)
    assert source.read_bytes() == expected
    assert output == (
        f"Wrote {len(expected)} bytes to {source.resolve()}.\n\n"
        + (expected.decode().rstrip("\n") or "(no output)")
    )


@pytest.mark.parametrize("executor_kind", ["default", "local", "factory"])
@pytest.mark.parametrize("save", [False, True])
def test_jq_excludes_shell_startup_output(
    tmp_path: Path,
    jq_binary: str,
    monkeypatch: pytest.MonkeyPatch,
    executor_kind: str,
    save: bool,
) -> None:
    del jq_binary
    source = tmp_path / "data.json"
    source.write_bytes(b'{"count":1}')
    startup = tmp_path / "startup.sh"
    startup.write_text("printf 'startup-noise\\n'\n", encoding="utf-8")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("JQ_TEST_VALUE", "retained")
    executor = LocalBashExecutor()
    if executor_kind == "factory":
        (definition,) = base_harness_tools(
            cwd=tmp_path, bash_executor=executor, include=("jq",)
        )
    else:
        definition = jq_tool(
            cwd=tmp_path, executor=executor if executor_kind == "local" else None
        )

    output = run_tool(
        definition,
        path="data.json",
        filter=".count += 1 | .value = $ENV.JQ_TEST_VALUE",
        compact_output=True,
        output_path="data.json" if save else None,
    )

    expected = b'{"count":2,"value":"retained"}\n'
    assert not isinstance(output, ToolFailure)
    assert "startup-noise" not in output
    assert output.endswith(expected.decode().rstrip("\n"))
    assert source.read_bytes() == (expected if save else b'{"count":1}')

    # Reusing the executor for Bash keeps its existing startup behavior.
    bash_output = run_tool(
        bash_tool(cwd=tmp_path, executor=executor), command="printf bash-output"
    )
    assert bash_output == "startup-noise\nbash-output"


def test_jq_saves_stdout_without_stderr(tmp_path: Path, jq_binary: str) -> None:
    del jq_binary
    (tmp_path / "data.json").write_bytes(b'{"count":1}')

    output = run_tool(
        jq_tool(cwd=tmp_path),
        path="data.json",
        filter="debug | .count += 1",
        output_path="new/copy.json",
    )

    assert not isinstance(output, ToolFailure)
    assert (tmp_path / "new/copy.json").read_bytes() == b'{\n  "count": 2\n}\n'
    assert (tmp_path / "data.json").read_bytes() == b'{"count":1}'
    assert "DEBUG" in output


def test_jq_saves_full_large_string_despite_preview_truncation(
    tmp_path: Path, jq_binary: str
) -> None:
    del jq_binary
    source = tmp_path / "data.json"
    original = {"body": "x" * (4 * 1024 * 1024), "count": 1}
    source.write_text(json.dumps(original), encoding="utf-8")

    output = run_tool(
        jq_tool(cwd=tmp_path),
        path="data.json",
        filter=".count += 1",
        output_path="data.json",
    )

    log_line = next(line for line in output.splitlines() if "Full output: " in line)
    log_path = Path(log_line.rsplit("Full output: ", maxsplit=1)[1].removesuffix("]"))
    try:
        assert not isinstance(output, ToolFailure)
        assert json.loads(source.read_bytes()) == {**original, "count": 2}
        assert source.stat().st_size > 4 * 1024 * 1024
        assert len(output.encode()) < 52_000
        assert "Showing last" in output
    finally:
        log_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("source", "jq_filter", "timeout", "limit", "kind"),
    [
        (b'{"id":1}\n{"id":', ".id", None, 1024, "nonzero_exit"),
        (b"null", '., error("failed")', None, 1024, "nonzero_exit"),
        (b"null", ".[", None, 1024, "nonzero_exit"),
        (b"null", "until(false; .)", 0.05, 1024, "timeout"),
        (b"null", "range(0; infinite)", 2, 64, "error"),
    ],
)
def test_jq_failed_save_preserves_source(
    tmp_path: Path,
    jq_binary: str,
    source: bytes,
    jq_filter: str,
    timeout: float | None,
    limit: int,
    kind: str,
) -> None:
    del jq_binary
    path = tmp_path / "data.json"
    path.write_bytes(source)

    output = run_tool(
        jq_tool(cwd=tmp_path, max_output_bytes=limit),
        path="data.json",
        filter=jq_filter,
        output_path="data.json",
        timeout=timeout,
    )

    assert isinstance(output, ToolFailure)
    assert output.error.kind == kind
    assert path.read_bytes() == source
    if kind == "error":
        assert f"maximum {limit} bytes" in output


@pytest.mark.parametrize("limit", [2, 3])
def test_jq_save_output_limit_is_exact(
    tmp_path: Path, jq_binary: str, limit: int
) -> None:
    del jq_binary
    (tmp_path / "data.json").write_bytes(b"42")

    output = run_tool(
        jq_tool(cwd=tmp_path, max_output_bytes=limit),
        path="data.json",
        filter=".",
        output_path="output.json",
    )

    if limit == 3:
        assert not isinstance(output, ToolFailure)
        assert (tmp_path / "output.json").read_bytes() == b"42\n"
    else:
        assert isinstance(output, ToolFailure)
        assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize(
    "denied_operation",
    [
        ToolOperation.CREATE_DIRECTORY,
        ToolOperation.CREATE_FILE,
        ToolOperation.OVERWRITE_FILE,
    ],
)
def test_jq_save_checks_write_permissions_before_read_or_execution(
    tmp_path: Path, denied_operation: ToolOperation
) -> None:
    policy = _RecordingPolicy(denied_operation)
    executor = _CapturingExecutor()
    target = "new/output.json"
    if denied_operation == ToolOperation.OVERWRITE_FILE:
        target = "output.json"
        (tmp_path / target).write_bytes(b"original")

    output = run_tool(
        jq_tool(
            reader=_UnopenedReader(),
            writer=LocalFileSystem(tmp_path),
            executor=executor,
            permissions=policy,
        ),
        path="data.json",
        filter=".",
        output_path=target,
    )

    assert isinstance(output, ToolFailure)
    assert "permission denied" in output
    assert policy.requests[-1].operation == denied_operation
    assert all(request.tool_name == "jq" for request in policy.requests)
    assert not executor.requests
    assert not (tmp_path / "new").exists()
    if denied_operation == ToolOperation.OVERWRITE_FILE:
        assert (tmp_path / target).read_bytes() == b"original"


def test_jq_save_rejects_missing_capture_capability_before_read(tmp_path: Path) -> None:
    executor = _RecordingExecutor()

    output = run_tool(
        jq_tool(
            reader=_UnopenedReader(),
            writer=LocalFileSystem(tmp_path),
            executor=executor,
        ),
        path="data.json",
        filter=".",
        output_path="output.json",
    )

    assert isinstance(output, ToolFailure)
    assert "executor that captures stdout" in output
    assert not executor.requests
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("stdout", "truncated"), [(None, False), (b"42", True)])
def test_jq_save_rejects_incomplete_capture_even_after_success(
    tmp_path: Path, stdout: bytes | None, truncated: bool
) -> None:
    path = tmp_path / "data.json"
    path.write_bytes(b"1")
    executor = _CapturingExecutor(stdout, truncated=truncated)

    output = run_tool(
        jq_tool(cwd=tmp_path, executor=executor, max_output_bytes=10),
        path="data.json",
        filter=".",
        output_path="data.json",
    )

    assert isinstance(output, ToolFailure)
    assert path.read_bytes() == b"1"
    assert executor.requests[0].stdout_capture_limit == 10


def test_jq_save_configuration_rejects_ambiguous_storage_and_invalid_limit() -> None:
    with pytest.raises(ValueError, match="writer requires a reader"):
        jq_tool(writer=LocalFileSystem())
    with pytest.raises(ValueError, match="max_output_bytes must be greater than zero"):
        jq_tool(max_output_bytes=0)


@pytest.mark.parametrize("grants_text", ["jq", "jq:read_file,jq:execute_command"])
def test_jq_save_requires_write_grants(
    tmp_path: Path, jq_binary: str, grants_text: str
) -> None:
    del jq_binary
    source = tmp_path / "data.json"
    source.write_bytes(b"1")
    grants, invalid_entries = parse_tool_permission_grants(grants_text)
    assert invalid_entries == ()

    output = run_tool(
        jq_tool(cwd=tmp_path, permissions=ToolPermissionGrantPolicy(grants)),
        path="data.json",
        filter=". + 1",
        output_path="data.json",
    )

    if grants_text == "jq":
        assert not isinstance(output, ToolFailure)
        assert source.read_bytes() == b"2\n"
    else:
        assert isinstance(output, ToolFailure)
        assert "permission denied" in output
        assert source.read_bytes() == b"1"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["token", "task"])
async def test_jq_cancelled_execution_does_not_save(
    tmp_path: Path, jq_binary: str, stop: str
) -> None:
    del jq_binary
    source = tmp_path / "data.json"
    source.write_bytes(b"null")
    tool = cast(Tool[BaseModel, str], jq_tool(cwd=tmp_path).tool)
    token = CancellationToken()
    task = asyncio.create_task(
        tool.run(
            tool.args_type()(
                path="data.json", filter="until(false; .)", output_path="data.json"
            ),
            token,
        )
    )
    await asyncio.sleep(0.1)
    if stop == "token":
        token.cancel()
        output = await asyncio.wait_for(task, timeout=5)
        assert isinstance(output, ToolFailure)
        assert output.error.kind == "cancelled"
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    assert source.read_bytes() == b"null"
